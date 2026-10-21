//! Shared checked repository setup for init and zero-setup serve; v1 §§4.2, 10.2.
/// Typed setup errors preserve filesystem causes without capturing file contents.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("{0}")]
    Invalid(&'static str),
    #[error("Repository setup I/O failed")]
    Io(#[from] std::io::Error),
    #[cfg(unix)]
    #[error("Repository setup filesystem operation failed")]
    Filesystem(#[from] rustix::io::Errno),
    #[cfg(unix)]
    #[error("{message}")]
    Destination {
        message: &'static str,
        #[source]
        source: rustix::io::Errno,
    },
}
type Result<T> = std::result::Result<T, Error>;

/// Start marker shared by automatic and explicit setup.
pub(crate) const EXCLUDE_START: &str = "# >>> deepkit-code >>>";
/// End marker of the owned exclude block.
pub(crate) const EXCLUDE_END: &str = "# <<< deepkit-code <<<";
/// Only machine-local index data, overrides and prompts are excluded.
pub(crate) const EXCLUDE_CONTENT: &str =
    ".deepkit/code.db*\n.deepkit/config.local.toml\n.deepkit/prompts/\n";

/// Closed set of opt-in IDE destinations; content merging remains with the caller.
#[derive(Clone, Copy)]
pub(crate) enum ExtraFile {
    Cursor,
    Claude,
    Codex,
}
impl ExtraFile {
    #[cfg(unix)]
    fn parts(self) -> Vec<&'static str> {
        match self {
            Self::Cursor => vec![".cursor", "mcp.json"],
            Self::Claude => vec![".mcp.json"],
            Self::Codex => vec![".codex", "config.toml"],
        }
    }
}
use std::path::Path;

#[cfg(unix)]
pub(crate) use unix::Prepared;

/// Reads only an init configuration snapshot through bounded no-follow descriptors.
/// Missing files remain absent; unsafe files are refused before parsing or probing.
pub(crate) fn read_configuration(root: &Path, name: &str) -> Result<Option<String>> {
    if !matches!(name, "config.toml" | "config.local.toml") {
        return Err(Error::Invalid("Unsupported init configuration filename"));
    }
    #[cfg(unix)]
    {
        unix::read_configuration(root, name)
    }
    #[cfg(not(unix))]
    {
        let _ = root;
        Err(Error::Invalid(
            "Checked init configuration reading is not implemented on this platform",
        ))
    }
}

/// Reads one supported project MCP file through the shared bounded no-follow reader.
/// No parser or external command runs, and an absent file is not created.
pub(crate) fn read_extra(root: &Path, target: ExtraFile) -> Result<Option<Vec<u8>>> {
    #[cfg(unix)]
    {
        unix::read_extra(root, target)
    }
    #[cfg(not(unix))]
    {
        let _ = (root, target);
        Err(Error::Invalid(
            "Checked project MCP reading is not implemented on this platform",
        ))
    }
}

/// Captures both destinations before provider verification without creating files.
pub(crate) fn prepare(
    root: &Path,
    gitignore: bool,
    original_config: Option<&str>,
    proposed_config: Option<&str>,
) -> Result<Prepared> {
    #[cfg(unix)]
    {
        unix::prepare(root, gitignore, Some((original_config, proposed_config)))
    }
    #[cfg(not(unix))]
    {
        let _ = (root, gitignore, original_config, proposed_config);
        Err(Error::Invalid(
            "Checked init publication is not implemented on this platform",
        ))
    }
}

/// Prepares only the repository-local excludes used by zero-setup serve.
/// It never reads or writes configuration or falls back to a shared .gitignore.
pub(crate) fn prepare_excludes(root: &Path) -> Result<Prepared> {
    #[cfg(unix)]
    {
        unix::prepare(root, false, None)
    }
    #[cfg(not(unix))]
    {
        let _ = root;
        Err(Error::Invalid(
            "Checked init publication is not implemented on this platform",
        ))
    }
}

#[cfg(not(unix))]
pub(crate) struct Prepared;
#[cfg(not(unix))]
impl Prepared {
    /// Refuses additional files until checked filesystem operations exist.
    pub(crate) fn with_file<E: From<Error>>(
        self,
        _target: ExtraFile,
        _render: impl FnOnce(Option<&[u8]>) -> std::result::Result<Vec<u8>, E>,
    ) -> std::result::Result<Self, E> {
        Err(Error::Invalid("Checked init publication is not implemented on this platform").into())
    }
    /// Refuses publication until equivalent checked filesystem operations exist.
    pub(crate) fn check(&self) -> Result<()> {
        Err(Error::Invalid(
            "Checked init publication is not implemented on this platform",
        ))
    }
    /// Refuses publication until equivalent checked filesystem operations exist.
    pub(crate) fn publish(&self) -> Result<()> {
        self.check()
    }
}

fn exclude_text(existing: &str) -> Result<String> {
    let starts = existing.match_indices(EXCLUDE_START).collect::<Vec<_>>();
    let ends = existing.match_indices(EXCLUDE_END).collect::<Vec<_>>();
    let block = format!("{EXCLUDE_START}\n{EXCLUDE_CONTENT}{EXCLUDE_END}\n");
    match (starts.as_slice(), ends.as_slice()) {
        ([], []) => Ok(format!(
            "{existing}{}{block}",
            if existing.is_empty() || existing.ends_with('\n') {
                ""
            } else {
                "\n"
            }
        )),
        ([(start, _)], [(end, _)]) if start < end => {
            // Marker substrings within another line are not an owned block.
            for (offset, marker) in [(*start, EXCLUDE_START), (*end, EXCLUDE_END)] {
                let after = &existing[offset + marker.len()..];
                if (offset != 0 && !existing[..offset].ends_with('\n'))
                    || (!after.is_empty() && !after.starts_with('\n'))
                {
                    return Err(Error::Invalid("Malformed Deepkit Code exclude block"));
                }
            }
            let suffix = &existing[end + EXCLUDE_END.len()..];
            Ok(format!(
                "{}{block}{}",
                &existing[..*start],
                suffix.strip_prefix('\n').unwrap_or(suffix)
            ))
        }
        _ => Err(Error::Invalid(
            "Malformed or duplicate Deepkit Code exclude block",
        )),
    }
}

#[cfg(unix)]
mod unix {
    use super::*;
    use rustix::fs::{self as rfs, AtFlags, FileType, Mode, OFlags};
    use std::{
        fs::{self, File},
        io::{Read, Write},
        os::fd::OwnedFd,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
    const MAX_EXISTING_BYTES: u64 = 1024 * 1024;

    #[derive(Debug, PartialEq, Eq)]
    struct Identity {
        device: rfs::Dev,
        inode: u64,
    }
    fn identity(fd: &OwnedFd) -> Result<Identity> {
        let stat = rfs::fstat(fd)?;
        Ok(Identity {
            device: stat.st_dev,
            inode: stat.st_ino,
        })
    }
    fn directory_flags() -> OFlags {
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC
    }
    #[derive(PartialEq, Eq)]
    struct Existing {
        identity: Identity,
        bytes: Vec<u8>,
        mode: Mode,
    }
    #[derive(PartialEq, Eq)]
    struct Snapshot {
        parents: Vec<Option<Identity>>,
        file: Option<Existing>,
    }
    struct Target {
        parts: Vec<&'static str>,
        before: Snapshot,
        proposed: Option<Vec<u8>>,
    }

    /// Original identities and bytes remain authoritative through verification.
    pub(crate) struct Prepared {
        root: PathBuf,
        root_identity: Identity,
        targets: Vec<Target>,
    }

    pub(super) fn read_extra(root: &Path, target: ExtraFile) -> Result<Option<Vec<u8>>> {
        let root = fs::canonicalize(root)?;
        let directory = rfs::open(&root, directory_flags(), Mode::empty())?;
        Ok(snapshot(&directory, &target.parts())?
            .file
            .map(|file| file.bytes))
    }

    pub(crate) fn read_configuration(root: &Path, name: &str) -> Result<Option<String>> {
        let root = fs::canonicalize(root)?;
        let directory = rfs::open(&root, directory_flags(), Mode::empty())?;
        snapshot(&directory, &[".deepkit", name])?
            .file
            .map(|file| {
                String::from_utf8(file.bytes)
                    .map_err(|_| Error::Invalid("Init configuration is not valid UTF-8"))
            })
            .transpose()
    }

    pub(crate) fn prepare(
        root: &Path,
        gitignore: bool,
        config: Option<(Option<&str>, Option<&str>)>,
    ) -> Result<Prepared> {
        let root = fs::canonicalize(root)?;
        let directory = rfs::open(&root, directory_flags(), Mode::empty())?;
        let root_identity = identity(&directory)?;
        let mut targets = Vec::new();
        if let Some((original_config, proposed_config)) = config {
            let parts = vec![".deepkit", "config.toml"];
            let before = snapshot(&directory, &parts)?;
            if before.file.as_ref().map(|file| file.bytes.as_slice())
                != original_config.map(str::as_bytes)
            {
                return Err(Error::Invalid(
                    "Configuration changed during init; retry with the new settings",
                ));
            }
            targets.push(Target {
                parts,
                before,
                proposed: proposed_config.map(|text| text.as_bytes().to_vec()),
            });
        }
        if !gitignore
            && fs::symlink_metadata(root.join(".git")).is_ok_and(|metadata| metadata.is_file())
        {
            return Err(Error::Invalid(
                "Init excludes for a Git worktree file require --gitignore",
            ));
        }
        let exclude_parts = if gitignore {
            vec![".gitignore"]
        } else {
            vec![".git", "info", "exclude"]
        };
        let exclude_before = snapshot(&directory, &exclude_parts)?;
        let existing = exclude_before
            .file
            .as_ref()
            .map_or(&[][..], |file| file.bytes.as_slice());
        let text = std::str::from_utf8(existing)
            .map_err(|_| Error::Invalid("Exclude file is not valid UTF-8"))?;
        let exclude = exclude_text(text)?.into_bytes();
        targets.push(Target {
            parts: exclude_parts,
            before: exclude_before,
            proposed: Some(exclude),
        });
        let prepared = Prepared {
            root,
            root_identity,
            targets,
        };
        prepared.check()?;
        Ok(prepared)
    }

    fn read(directory: &OwnedFd, name: &str) -> Result<Option<Existing>> {
        let fd = match rfs::openat(
            directory,
            name,
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
            Mode::empty(),
        ) {
            Ok(fd) => fd,
            Err(rustix::io::Errno::NOENT) => return Ok(None),
            Err(source) => {
                return Err(Error::Destination {
                    message: "Init destination is unavailable or not a regular file",
                    source,
                });
            }
        };
        let stat = rfs::fstat(&fd)?;
        if !FileType::from_raw_mode(stat.st_mode).is_file()
            || stat.st_nlink != 1
            || stat.st_size < 0
            || stat.st_size as u64 > MAX_EXISTING_BYTES
        {
            return Err(Error::Invalid(
                "Init destination must be a single-link regular file of at most 1 MiB",
            ));
        }
        let original_identity = identity(&fd)?;
        let mut bytes = Vec::new();
        File::from(fd)
            .take(MAX_EXISTING_BYTES + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_EXISTING_BYTES {
            return Err(Error::Invalid("Init destination exceeds 1 MiB"));
        }
        Ok(Some(Existing {
            identity: original_identity,
            bytes,
            mode: Mode::from_raw_mode(stat.st_mode & 0o777),
        }))
    }
    fn snapshot(root: &OwnedFd, parts: &[&str]) -> Result<Snapshot> {
        let (name, parents) = parts
            .split_last()
            .ok_or(Error::Invalid("Init destination has no filename"))?;
        let mut directory = rfs::openat(root, ".", directory_flags(), Mode::empty())?;
        let mut captured = Vec::new();
        let mut absent = false;
        for parent in parents {
            if absent {
                captured.push(None);
                continue;
            }
            match rfs::openat(&directory, *parent, directory_flags(), Mode::empty()) {
                Ok(next) => {
                    captured.push(Some(identity(&next)?));
                    directory = next;
                }
                Err(rustix::io::Errno::NOENT) => {
                    captured.push(None);
                    absent = true;
                }
                Err(source) => {
                    return Err(Error::Destination {
                        message: "Init destination ancestry must contain ordinary directories",
                        source,
                    });
                }
            }
        }
        let file = if absent {
            None
        } else {
            read(&directory, name)?
        };
        Ok(Snapshot {
            parents: captured,
            file,
        })
    }
    impl Prepared {
        /// Snapshots an explicitly selected destination and delegates its pure merge.
        /// The callback can preserve its own typed errors; no writes happen here.
        pub(crate) fn with_file<E: From<Error>>(
            mut self,
            target: ExtraFile,
            render: impl FnOnce(Option<&[u8]>) -> std::result::Result<Vec<u8>, E>,
        ) -> std::result::Result<Self, E> {
            let directory = self.root_directory()?;
            let parts = target.parts();
            let before = snapshot(&directory, &parts)?;
            let proposed = render(before.file.as_ref().map(|file| file.bytes.as_slice()))?;
            self.targets.push(Target {
                parts,
                before,
                proposed: Some(proposed),
            });
            self.check()?;
            Ok(self)
        }
        fn root_directory(&self) -> Result<OwnedFd> {
            if fs::canonicalize(&self.root)? != self.root {
                return Err(Error::Invalid("Repository location changed during init"));
            }
            let directory = rfs::open(&self.root, directory_flags(), Mode::empty())?;
            if identity(&directory)? != self.root_identity {
                return Err(Error::Invalid("Repository location changed during init"));
            }
            Ok(directory)
        }
        /// Rechecks original destinations without writes before credential storage.
        pub(crate) fn check(&self) -> Result<()> {
            let directory = self.root_directory()?;
            for target in &self.targets {
                if snapshot(&directory, &target.parts)? != target.before {
                    return Err(Error::Invalid(
                        "Init destination changed during verification; retry init",
                    ));
                }
            }
            Ok(())
        }
        /// Publishes each requested file atomically; this is not a multi-file transaction.
        pub(crate) fn publish(&self) -> Result<()> {
            self.check()?;
            for target in &self.targets {
                let Some(bytes) = &target.proposed else {
                    continue;
                };
                if target.before.file.as_ref().map(|file| &file.bytes) == Some(bytes) {
                    continue;
                }
                self.publish_target(target, bytes)?;
            }
            Ok(())
        }
        fn publish_target(&self, target: &Target, bytes: &[u8]) -> Result<()> {
            let root = self.root_directory()?;
            if snapshot(&root, &target.parts)? != target.before {
                return Err(Error::Invalid(
                    "Init destination changed before publication; retry init",
                ));
            }
            let (name, parents) = target
                .parts
                .split_last()
                .ok_or(Error::Invalid("Init destination has no filename"))?;
            let mut directory = root;
            for (parent, expected) in parents.iter().zip(&target.before.parents) {
                if expected.is_none() {
                    // A competing creation is a changed destination, not permission
                    // to adopt an unverified directory supplied during verification.
                    rfs::mkdirat(&directory, *parent, Mode::from_raw_mode(0o700))?;
                }
                let next = rfs::openat(&directory, *parent, directory_flags(), Mode::empty())?;
                if let Some(expected) = expected
                    && identity(&next)? != *expected
                {
                    return Err(Error::Invalid(
                        "Init destination directory changed before publication",
                    ));
                }
                directory = next;
            }
            let temporary_name = format!(
                ".dkc-init-{}-{}.tmp",
                std::process::id(),
                TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
            );
            let mode = target
                .before
                .file
                .as_ref()
                .map_or(Mode::RUSR | Mode::WUSR, |file| file.mode);
            let fd = rfs::openat(
                &directory,
                temporary_name.as_str(),
                OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                mode,
            )?;
            let mut temporary = Temporary {
                directory: &directory,
                name: temporary_name,
                remove: true,
            };
            let mut file = File::from(fd);
            rfs::fchmod(&file, mode)?;
            file.write_all(bytes)?;
            file.sync_all()?;
            // Reopen the current ancestry and compare the directory descriptor.
            // A renamed directory must not redirect publication outside the repo.
            let mut current = self.root_directory()?;
            for parent in parents {
                current = rfs::openat(&current, *parent, directory_flags(), Mode::empty())?;
            }
            if identity(&current)? != identity(&directory)?
                || read(&directory, name)? != target.before.file
            {
                return Err(Error::Invalid(
                    "Init destination changed during publication; retry init",
                ));
            }
            let original = rfs::fstat(&file)?;
            let named = rfs::statat(
                &directory,
                temporary.name.as_str(),
                AtFlags::SYMLINK_NOFOLLOW,
            )?;
            if original.st_nlink != 1
                || original.st_dev != named.st_dev
                || original.st_ino != named.st_ino
            {
                return Err(Error::Invalid(
                    "Init temporary file changed during publication",
                ));
            }
            rfs::renameat(&directory, temporary.name.as_str(), &directory, *name)?;
            temporary.remove = false;
            rfs::fsync(&directory)?;
            Ok(())
        }
    }
    struct Temporary<'a> {
        directory: &'a OwnedFd,
        name: String,
        remove: bool,
    }
    impl Drop for Temporary<'_> {
        fn drop(&mut self) {
            if self.remove {
                let _ = rfs::unlinkat(self.directory, self.name.as_str(), AtFlags::empty());
            }
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::{fs, os::unix::fs::symlink};

    #[cfg(unix)]
    use crate::sync::source::mkfifo;

    #[test]
    fn configuration_reads_capture_both_names_without_creating_or_rewriting_files() {
        let root = tempfile::tempdir().unwrap();
        for name in ["config.toml", "config.local.toml"] {
            assert_eq!(read_configuration(root.path(), name).unwrap(), None);
        }
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
        fs::create_dir(root.path().join(".deepkit")).unwrap();
        for name in ["config.toml", "config.local.toml"] {
            let text = format!("# {name}\n[index]\nrepo_name='kept'\n");
            let path = root.path().join(".deepkit").join(name);
            fs::write(&path, &text).unwrap();
            assert_eq!(
                read_configuration(root.path(), name).unwrap(),
                Some(text.clone())
            );
            assert_eq!(fs::read_to_string(&path).unwrap(), text);
        }
        assert_eq!(
            fs::read_dir(root.path().join(".deepkit")).unwrap().count(),
            2
        );
        assert!(read_configuration(root.path(), "../outside").is_err());
    }

    #[test]
    fn configuration_reads_refuse_hostile_files_and_fifo_without_blocking() {
        use std::{sync::mpsc, time::Duration};
        for name in ["config.toml", "config.local.toml"] {
            for kind in ["fifo", "symlink", "hardlink", "directory", "oversize"] {
                let root = tempfile::tempdir().unwrap();
                let outside = tempfile::tempdir().unwrap();
                fs::create_dir(root.path().join(".deepkit")).unwrap();
                let target = root.path().join(".deepkit").join(name);
                let external = outside.path().join("untouched");
                fs::write(&external, "unchanged").unwrap();
                match kind {
                    "fifo" => {
                        mkfifo(&target, rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR).unwrap()
                    }
                    "symlink" => symlink(&external, &target).unwrap(),
                    "hardlink" => fs::hard_link(&external, &target).unwrap(),
                    "directory" => fs::create_dir(&target).unwrap(),
                    _ => fs::File::create(&target)
                        .unwrap()
                        .set_len(1024 * 1024 + 1)
                        .unwrap(),
                }
                let path = root.path().to_path_buf();
                let (sender, receiver) = mpsc::channel();
                let worker = std::thread::spawn(move || {
                    sender
                        .send(read_configuration(&path, name).is_err())
                        .unwrap();
                });
                assert!(
                    receiver.recv_timeout(Duration::from_secs(2)).unwrap(),
                    "{name}/{kind}"
                );
                worker.join().unwrap();
                assert_eq!(fs::read_to_string(&external).unwrap(), "unchanged");
                assert_eq!(fs::read_dir(root.path()).unwrap().count(), 1);
                assert_eq!(
                    fs::read_dir(root.path().join(".deepkit")).unwrap().count(),
                    1
                );
            }
        }
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("config.local.toml"), "kept").unwrap();
        symlink(outside.path(), root.path().join(".deepkit")).unwrap();
        assert!(read_configuration(root.path(), "config.local.toml").is_err());
        assert_eq!(
            fs::read_to_string(outside.path().join("config.local.toml")).unwrap(),
            "kept"
        );
    }

    #[test]
    fn preflight_is_read_only_and_publication_is_idempotent() {
        let root = tempfile::tempdir().unwrap();
        let prepared =
            prepare(root.path(), false, None, Some("[index]\nrepo_name='new'\n")).unwrap();
        prepared.check().unwrap();
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
        prepared.publish().unwrap();
        let config = fs::read_to_string(root.path().join(".deepkit/config.toml")).unwrap();
        let exclude = fs::read(root.path().join(".git/info/exclude")).unwrap();
        let again = prepare(root.path(), false, Some(&config), None).unwrap();
        again.publish().unwrap();
        assert_eq!(
            fs::read(root.path().join(".git/info/exclude")).unwrap(),
            exclude
        );
        assert_eq!(
            fs::read_to_string(root.path().join(".deepkit/config.toml")).unwrap(),
            config
        );
    }

    #[test]
    fn malformed_markers_refuse_before_configuration_creation() {
        for text in [
            "# >>> deepkit-code >>>\n",
            "# <<< deepkit-code <<<\n",
            "# <<< deepkit-code <<<\n# >>> deepkit-code >>>\n",
            "# >>> deepkit-code >>>\n# >>> deepkit-code >>>\n# <<< deepkit-code <<<\n",
            "prefix # >>> deepkit-code >>>\n# <<< deepkit-code <<<\n",
        ] {
            let root = tempfile::tempdir().unwrap();
            fs::write(root.path().join(".gitignore"), text).unwrap();
            assert!(prepare(root.path(), true, None, Some("new")).is_err());
            assert!(!root.path().join(".deepkit").exists());
            assert_eq!(
                fs::read_to_string(root.path().join(".gitignore")).unwrap(),
                text
            );
        }
    }

    #[test]
    fn unsafe_destinations_and_worktree_files_are_explicitly_refused() {
        for target in [".deepkit", ".git", ".gitignore"] {
            let root = tempfile::tempdir().unwrap();
            let outside = tempfile::tempdir().unwrap();
            let destination = if target == ".gitignore" {
                let path = outside.path().join("file");
                fs::write(&path, "kept").unwrap();
                path
            } else {
                outside.path().into()
            };
            symlink(&destination, root.path().join(target)).unwrap();
            assert!(prepare(root.path(), target == ".gitignore", None, Some("new")).is_err());
        }
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let file = outside.path().join("file");
        fs::write(&file, "kept").unwrap();
        fs::hard_link(&file, root.path().join(".gitignore")).unwrap();
        assert!(prepare(root.path(), true, None, Some("new")).is_err());
        assert_eq!(fs::read_to_string(&file).unwrap(), "kept");
        fs::write(root.path().join(".git"), "gitdir: /other/worktree\n").unwrap();
        let error = prepare(root.path(), false, None, Some("new"))
            .err()
            .unwrap();
        assert!(error.to_string().contains("--gitignore"));
    }

    #[test]
    fn changed_config_or_ancestry_is_not_overwritten() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join(".deepkit")).unwrap();
        let config = root.path().join(".deepkit/config.toml");
        fs::write(&config, "original").unwrap();
        let prepared = prepare(root.path(), true, Some("original"), Some("proposed")).unwrap();
        fs::write(&config, "concurrent").unwrap();
        assert!(prepared.check().is_err());
        assert!(prepared.publish().is_err());
        assert_eq!(fs::read_to_string(&config).unwrap(), "concurrent");
        assert!(!root.path().join(".gitignore").exists());
        let prepared = prepare(root.path(), true, Some("concurrent"), None).unwrap();
        fs::rename(root.path().join(".deepkit"), root.path().join("old")).unwrap();
        fs::create_dir(root.path().join(".deepkit")).unwrap();
        fs::write(&config, "concurrent").unwrap();
        assert!(prepared.check().is_err());
        assert!(prepared.publish().is_err());
        assert!(!root.path().join(".gitignore").exists());
    }
    #[test]
    fn excludes_only_is_read_only_then_idempotent_and_never_creates_configuration() {
        let root = tempfile::tempdir().unwrap();
        let prepared = prepare_excludes(root.path()).unwrap();
        prepared.check().unwrap();
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
        prepared.publish().unwrap();
        let path = root.path().join(".git/info/exclude");
        let bytes = fs::read(&path).unwrap();
        assert_eq!(
            String::from_utf8(bytes.clone()).unwrap(),
            format!("{EXCLUDE_START}\n{EXCLUDE_CONTENT}{EXCLUDE_END}\n")
        );
        prepare_excludes(root.path()).unwrap().publish().unwrap();
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 1);
        assert!(!root.path().join(".deepkit").exists());
        assert!(!root.path().join(".gitignore").exists());
    }

    #[test]
    fn excludes_only_does_not_access_configuration_and_refuses_changed_excludes() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        symlink(outside.path(), root.path().join(".deepkit")).unwrap();
        fs::create_dir_all(root.path().join(".git/info")).unwrap();
        let path = root.path().join(".git/info/exclude");
        fs::write(&path, "kept\n").unwrap();
        let prepared = prepare_excludes(root.path()).unwrap();
        fs::write(&path, "concurrent\n").unwrap();
        assert!(prepared.check().is_err());
        assert!(prepared.publish().is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), "concurrent\n");
        assert_eq!(fs::read_dir(outside.path()).unwrap().count(), 0);
        prepare_excludes(root.path()).unwrap().publish().unwrap();
        assert!(
            fs::read_to_string(&path)
                .unwrap()
                .starts_with("concurrent\n")
        );
        assert!(!root.path().join(".gitignore").exists());
    }

    #[test]
    fn excludes_only_refuses_linked_worktree_without_fallback() {
        let root = tempfile::tempdir().unwrap();
        let original = "gitdir: /other/worktree\n";
        fs::write(root.path().join(".git"), original).unwrap();
        assert!(prepare_excludes(root.path()).is_err());
        assert_eq!(
            fs::read_to_string(root.path().join(".git")).unwrap(),
            original
        );
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 1);
    }
}
