//! Explicit, non-overwriting Git hook ownership; see v1.md §4.5.

use std::{fs, path::Path};

const NAMES: [&str; 4] = ["post-commit", "post-checkout", "post-merge", "post-rewrite"];
const SCRIPT: &str = "#!/bin/sh\n# Deepkit Code managed hook v1\nif command -v dkc >/dev/null 2>&1; then\n    dkc index --quick || printf '%s\\n' 'dkc: quick index failed; run dkc index --quick to retry' >&2\nelse\n    printf '%s\\n' 'dkc: executable missing from PATH; run dkc index --quick after restoring it' >&2\nfi\nexit 0\n";

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("cannot discover Git repository: {0}")]
    Git(#[source] Box<gix::discover::Error>),
    #[error("cannot update Git hooks: {0}")]
    Io(#[from] std::io::Error),
    #[cfg(unix)]
    #[error("cannot access Git hooks safely: {0}")]
    Filesystem(#[from] rustix::io::Errno),
    #[error("refusing hooks: {0}")]
    Refused(&'static str),
}

#[derive(serde::Serialize)]
pub(crate) struct Report {
    installed: bool,
    changed: Vec<&'static str>,
}

/// Changes only exact owned hooks after checking repository hook managers.
/// Merely indexing never calls this function: installation is explicit opt-in.
pub(crate) fn change(root: &Path, install: bool) -> Result<Report, Error> {
    let repo = gix::discover(root).map_err(|error| Error::Git(Box::new(error)))?;
    let worktree = repo.workdir().ok_or(Error::Refused("bare repository"))?;
    if repo.config_snapshot().string("core.hooksPath").is_some() {
        return Err(Error::Refused("core.hooksPath is configured"));
    }
    for name in [
        ".husky",
        ".huskyrc",
        ".huskyrc.json",
        ".huskyrc.js",
        ".huskyrc.yaml",
        ".huskyrc.yml",
        "husky.config.js",
        "husky.config.cjs",
        "lefthook.yml",
        "lefthook.yaml",
        ".lefthook.yml",
        ".lefthook.yaml",
        "lefthook.toml",
        ".lefthook.toml",
        ".pre-commit-config.yaml",
    ] {
        match fs::symlink_metadata(worktree.join(name)) {
            Ok(_) => {
                return Err(Error::Refused(
                    "husky, lefthook or pre-commit configuration exists",
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    let package = worktree.join("package.json");
    if package.is_file() {
        use std::io::Read;
        let mut bytes = Vec::new();
        fs::File::open(package)?
            .take(1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > 1024 * 1024 {
            return Err(Error::Refused(
                "package.json is too large to check hook ownership",
            ));
        }
        if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes)
            && value.get("husky").is_some()
        {
            return Err(Error::Refused("package.json contains husky configuration"));
        }
    }
    change_directory(repo.common_dir(), install)
}

#[cfg(not(unix))]
fn change_directory(_git: &Path, _install: bool) -> Result<Report, Error> {
    Err(Error::Refused(
        "safe hook installation is unavailable on this platform",
    ))
}

#[cfg(unix)]
fn change_directory(git: &Path, install: bool) -> Result<Report, Error> {
    use rustix::fs::{self as rfs, AtFlags, FileType, Mode, OFlags};
    use std::{
        io::{Read, Write},
        os::fd::OwnedFd,
        sync::atomic::{AtomicU64, Ordering},
    };
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let git = rfs::open(git, flags, Mode::empty())?;
    let directory = match rfs::openat(&git, "hooks", flags, Mode::empty()) {
        Ok(fd) => fd,
        Err(rustix::io::Errno::NOENT) if install => {
            rfs::mkdirat(&git, "hooks", Mode::from_raw_mode(0o755))?;
            rfs::openat(&git, "hooks", flags, Mode::empty())?
        }
        Err(rustix::io::Errno::NOENT) => {
            return Ok(Report {
                installed: false,
                changed: Vec::new(),
            });
        }
        Err(error) => return Err(error.into()),
    };
    fn owned(directory: &OwnedFd, name: &str) -> Result<bool, Error> {
        let fd = match rfs::openat(
            directory,
            name,
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
            Mode::empty(),
        ) {
            Ok(fd) => fd,
            Err(rustix::io::Errno::NOENT) => return Ok(false),
            Err(error) => return Err(error.into()),
        };
        let stat = rfs::fstat(&fd)?;
        if !FileType::from_raw_mode(stat.st_mode).is_file()
            || stat.st_nlink != 1
            || stat.st_size != SCRIPT.len() as i64
        {
            return Err(Error::Refused(
                "an existing hook is not an unchanged Deepkit hook",
            ));
        }
        let mut bytes = Vec::new();
        fs::File::from(fd)
            .take(SCRIPT.len() as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes != SCRIPT.as_bytes() {
            return Err(Error::Refused(
                "an existing hook is not an unchanged Deepkit hook",
            ));
        }
        Ok(true)
    }
    // Preflight every destination before installing or removing any script.
    let existing = NAMES.map(|name| owned(&directory, name));
    let existing = existing.into_iter().collect::<Result<Vec<_>, _>>()?;
    struct Temporary<'a> {
        directory: &'a OwnedFd,
        name: String,
    }
    impl Drop for Temporary<'_> {
        fn drop(&mut self) {
            let _ = rfs::unlinkat(self.directory, self.name.as_str(), AtFlags::empty());
        }
    }
    let mut changed = Vec::new();
    for (name, exists) in NAMES.into_iter().zip(existing) {
        if install && !exists {
            let temporary_name = format!(
                ".dkc-hook-{}-{}.tmp",
                std::process::id(),
                SEQUENCE.fetch_add(1, Ordering::Relaxed)
            );
            let fd = rfs::openat(
                &directory,
                temporary_name.as_str(),
                OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::from_raw_mode(0o755),
            )?;
            let temporary = Temporary {
                directory: &directory,
                name: temporary_name,
            };
            let mut file = fs::File::from(fd);
            rfs::fchmod(&file, Mode::from_raw_mode(0o755))?;
            file.write_all(SCRIPT.as_bytes())?;
            file.sync_all()?;
            // linkat refuses an existing destination, including a concurrent hook install.
            rfs::linkat(
                &directory,
                temporary.name.as_str(),
                &directory,
                name,
                AtFlags::empty(),
            )?;
            changed.push(name);
        } else if !install && exists && owned(&directory, name)? {
            rfs::unlinkat(&directory, name, AtFlags::empty())?;
            changed.push(name);
        }
    }
    Ok(Report {
        installed: install,
        changed,
    })
}
