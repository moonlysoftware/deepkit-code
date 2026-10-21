//! Verified-key persistence outside repositories; v1 §§10.2 and 11.
use std::path::Path;

/// Bounds both credential reads and writes before TOML parsing.
pub(crate) const MAX_FILE_BYTES: u64 = 1024 * 1024;

/// Fixed diagnostics never include credential text or parser excerpts.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("credential destination must be a private regular file outside the repository")]
    UnsafePath,
    #[error("credential file contains invalid or duplicate host entries")]
    InvalidFile,
    #[error("credential host or key is invalid")]
    InvalidInput,
    #[error("credential destination changed during preparation; retry init")]
    Changed,
    #[error("credential file I/O failed ({0:?})")]
    Io(std::io::ErrorKind),
    #[cfg(not(unix))]
    #[error("secure credential writing is not implemented on this platform")]
    Unsupported,
}
impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.kind())
    }
}

/// Stores a caller-verified key, preserving other hosts and never logging keys.
/// Unix uses private descriptor-relative temporary files and atomic replacement;
/// other platforms fail closed until equivalent semantics are implemented.
/// Credential verification belongs to the caller and must precede this operation.
pub(crate) fn upsert(
    root: &Path,
    credentials_path: &Path,
    host: &str,
    key: &str,
) -> Result<(), Error> {
    #[cfg(unix)]
    {
        unix::upsert(root, credentials_path, host, key)
    }
    #[cfg(not(unix))]
    {
        let _ = (root, credentials_path, host, key);
        Err(Error::Unsupported)
    }
}

#[cfg(unix)]
mod unix {
    use super::*;
    use rustix::fs::{self as rfs, AtFlags, FileType, FlockOperation, Mode, OFlags};
    use serde::{Deserialize, Serialize};
    use std::{
        collections::BTreeSet,
        ffi::OsStr,
        fs::{self, File},
        io::{Read, Write},
        os::{fd::OwnedFd, unix::fs::MetadataExt},
        path::{Component, PathBuf},
        sync::atomic::{AtomicU64, Ordering},
    };
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    #[derive(Default, Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct Document {
        #[serde(default)]
        credentials: Vec<Entry>,
    }
    #[derive(Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct Entry {
        host: String,
        api_key: String,
    }
    #[derive(PartialEq)]
    struct Existing {
        text: String,
        device: rfs::Dev,
        inode: u64,
    }
    fn io(error: rustix::io::Errno) -> Error {
        std::io::Error::from(error).into()
    }
    fn host_name(host: &str) -> Result<String, Error> {
        if host.is_empty() || host.chars().any(char::is_whitespace) {
            return Err(Error::InvalidInput);
        }
        let url =
            reqwest::Url::parse(&format!("https://{host}")).map_err(|_| Error::InvalidInput)?;
        if !url.username().is_empty()
            || url.password().is_some()
            || url.port().is_some()
            || url.path() != "/"
            || url.query().is_some()
            || url.fragment().is_some()
            || host.contains('/')
            || host.contains('\\')
            || host.contains('@')
        {
            return Err(Error::InvalidInput);
        }
        url.host_str().map(str::to_owned).ok_or(Error::InvalidInput)
    }
    fn key_valid(key: &str) -> bool {
        !key.trim().is_empty() && !key.chars().any(char::is_control)
    }

    /// Resolves existing ancestry before creating anything, including symlink aliases.
    fn destination(root: &Path, path: &Path) -> Result<(PathBuf, PathBuf), Error> {
        let root = fs::canonicalize(root)?;
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()?.join(path)
        };
        if absolute
            .components()
            .any(|part| matches!(part, Component::ParentDir))
        {
            return Err(Error::UnsafePath);
        }
        let name = absolute.file_name().ok_or(Error::UnsafePath)?;
        let mut parent = absolute.parent().ok_or(Error::UnsafePath)?.to_path_buf();
        let mut missing = Vec::new();
        loop {
            match fs::canonicalize(&parent) {
                Ok(mut resolved) => {
                    if !resolved.is_dir() {
                        return Err(Error::UnsafePath);
                    }
                    for component in missing.iter().rev() {
                        resolved.push(component);
                    }
                    let destination = resolved.join(name);
                    if destination.starts_with(&root) {
                        return Err(Error::UnsafePath);
                    }
                    return Ok((root, destination));
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    // A dangling symlink is not an absent directory to create.
                    if fs::symlink_metadata(&parent).is_ok() {
                        return Err(Error::UnsafePath);
                    }
                    missing.push(parent.file_name().ok_or(Error::UnsafePath)?.to_os_string());
                    if !parent.pop() {
                        return Err(Error::UnsafePath);
                    }
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
    fn directory(path: &Path) -> Result<OwnedFd, Error> {
        let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
        let mut fd = rfs::open("/", flags, Mode::empty()).map_err(io)?;
        for component in path.components() {
            match component {
                Component::RootDir | Component::CurDir => continue,
                Component::Normal(name) => match rfs::openat(&fd, name, flags, Mode::empty()) {
                    Ok(next) => fd = next,
                    Err(rustix::io::Errno::NOENT) => {
                        match rfs::mkdirat(&fd, name, Mode::from_raw_mode(0o700)) {
                            Ok(()) | Err(rustix::io::Errno::EXIST) => {}
                            Err(error) => return Err(io(error)),
                        }
                        fd = rfs::openat(&fd, name, flags, Mode::empty()).map_err(io)?;
                    }
                    Err(error) => return Err(io(error)),
                },
                _ => return Err(Error::UnsafePath),
            }
        }
        Ok(fd)
    }
    fn read(directory: &OwnedFd, name: &OsStr) -> Result<Option<Existing>, Error> {
        let fd = match rfs::openat(
            directory,
            name,
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
            Mode::empty(),
        ) {
            Ok(fd) => fd,
            Err(rustix::io::Errno::NOENT) => return Ok(None),
            Err(rustix::io::Errno::LOOP) => return Err(Error::UnsafePath),
            Err(error) => return Err(io(error)),
        };
        let stat = rfs::fstat(&fd).map_err(io)?;
        if !FileType::from_raw_mode(stat.st_mode).is_file()
            || stat.st_nlink != 1
            || stat.st_mode & 0o077 != 0
            || stat.st_size < 0
            || stat.st_size as u64 > MAX_FILE_BYTES
        {
            return Err(Error::UnsafePath);
        }
        let mut bytes = Vec::new();
        File::from(fd)
            .take(MAX_FILE_BYTES + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_FILE_BYTES {
            return Err(Error::InvalidFile);
        }
        let text = String::from_utf8(bytes).map_err(|_| Error::InvalidFile)?;
        Ok(Some(Existing {
            text,
            device: stat.st_dev,
            inode: stat.st_ino,
        }))
    }
    fn check_directory(root: &Path, parent: &Path, directory: &OwnedFd) -> Result<(), Error> {
        let resolved = fs::canonicalize(parent)?;
        let metadata = fs::metadata(parent)?;
        let descriptor = File::from(directory.try_clone()?).metadata()?;
        if resolved != parent
            || resolved.starts_with(root)
            || metadata.dev() != descriptor.dev()
            || metadata.ino() != descriptor.ino()
        {
            return Err(Error::Changed);
        }
        Ok(())
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
    pub(super) fn upsert(root: &Path, path: &Path, host: &str, key: &str) -> Result<(), Error> {
        upsert_observed(root, path, host, key, || {})
    }
    pub(super) fn upsert_observed(
        root: &Path,
        path: &Path,
        host: &str,
        key: &str,
        before_publish: impl FnOnce(),
    ) -> Result<(), Error> {
        let host = host_name(host)?;
        if !key_valid(key) {
            return Err(Error::InvalidInput);
        }
        let (root, path) = destination(root, path)?;
        let parent = path.parent().ok_or(Error::UnsafePath)?;
        let name = path.file_name().ok_or(Error::UnsafePath)?;
        let directory = directory(parent)?;
        // Serialize cooperating writers without another persistent lock file or waiting forever.
        rfs::flock(&directory, FlockOperation::NonBlockingLockExclusive).map_err(io)?;
        check_directory(&root, parent, &directory)?;
        let before = read(&directory, name)?;
        let mut document: Document = match &before {
            Some(existing) => toml::from_str(&existing.text).map_err(|_| Error::InvalidFile)?,
            None => Document::default(),
        };
        let mut seen = BTreeSet::new();
        for entry in &mut document.credentials {
            entry.host = host_name(&entry.host).map_err(|_| Error::InvalidFile)?;
            if !key_valid(&entry.api_key) || !seen.insert(entry.host.clone()) {
                return Err(Error::InvalidFile);
            }
        }
        if let Some(entry) = document
            .credentials
            .iter_mut()
            .find(|entry| entry.host == host)
        {
            entry.api_key = key.into();
        } else {
            document.credentials.push(Entry {
                host,
                api_key: key.into(),
            });
        }
        let rendered = toml::to_string(&document).map_err(|_| Error::InvalidFile)?;
        if rendered.len() as u64 > MAX_FILE_BYTES {
            return Err(Error::InvalidFile);
        }
        let temp_name = format!(
            ".credentials-{}-{}.tmp",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        );
        let fd = rfs::openat(
            &directory,
            temp_name.as_str(),
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::from_raw_mode(0o600),
        )
        .map_err(io)?;
        let mut temporary = Temporary {
            directory: &directory,
            name: temp_name,
            remove: true,
        };
        let mut file = File::from(fd);
        rfs::fchmod(&file, Mode::from_raw_mode(0o600)).map_err(io)?;
        file.write_all(rendered.as_bytes())?;
        file.sync_all()?;
        before_publish();
        check_directory(&root, parent, &directory)?;
        if before != read(&directory, name)? {
            return Err(Error::Changed);
        }
        let stat = rfs::fstat(&file).map_err(io)?;
        let named = rfs::statat(
            &directory,
            temporary.name.as_str(),
            AtFlags::SYMLINK_NOFOLLOW,
        )
        .map_err(io)?;
        if stat.st_nlink != 1
            || stat.st_mode & 0o777 != 0o600
            || stat.st_dev != named.st_dev
            || stat.st_ino != named.st_ino
        {
            return Err(Error::Changed);
        }
        rfs::renameat(&directory, temporary.name.as_str(), &directory, name).map_err(io)?;
        temporary.remove = false;
        rfs::fsync(&directory).map_err(io)?;
        Ok(())
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::{
        fs,
        os::unix::fs::{PermissionsExt, symlink},
    };
    fn secret(label: &str) -> String {
        ["fixture", label].join("-")
    }
    fn private(path: &Path, text: &str) {
        fs::write(path, text).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
    }
    fn no_temps(directory: &Path) {
        assert!(fs::read_dir(directory).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".credentials-")
        }));
    }
    #[test]
    fn upsert_preserves_other_hosts_and_writes_private_complete_document() {
        let repo = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let path = outside.path().join("new-directory/credentials.toml");
        let first = secret("first");
        let second = secret("second");
        let changed = secret("changed");
        upsert(repo.path(), &path, "ONE.example", &first).unwrap();
        upsert(repo.path(), &path, "two.example", &second).unwrap();
        upsert(repo.path(), &path, "one.example", &changed).unwrap();
        let document: toml::Value = toml::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        let rows = document["credentials"].as_array().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["host"].as_str(), Some("one.example"));
        assert_eq!(rows[0]["api_key"].as_str(), Some(changed.as_str()));
        assert_eq!(rows[1]["api_key"].as_str(), Some(second.as_str()));
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        no_temps(path.parent().unwrap());
    }
    #[test]
    fn repository_aliases_symlinks_hardlinks_and_nonregular_targets_are_refused() {
        let repo = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let key = secret("new");
        let inside = repo.path().join("nested/credentials.toml");
        assert!(upsert(repo.path(), &inside, "example.org", &key).is_err());
        assert!(!repo.path().join("nested").exists());
        let alias = outside.path().join("repo-alias");
        symlink(repo.path(), &alias).unwrap();
        assert!(
            upsert(
                repo.path(),
                &alias.join("new/credentials.toml"),
                "example.org",
                &key
            )
            .is_err()
        );
        let original = repo.path().join("original");
        private(&original, "unchanged");
        let target = outside.path().join("credentials.toml");
        symlink(&original, &target).unwrap();
        assert!(upsert(repo.path(), &target, "example.org", &key).is_err());
        fs::remove_file(&target).unwrap();
        fs::hard_link(&original, &target).unwrap();
        assert!(upsert(repo.path(), &target, "example.org", &key).is_err());
        assert_eq!(fs::read_to_string(&original).unwrap(), "unchanged");
        fs::remove_file(&target).unwrap();
        fs::create_dir(&target).unwrap();
        assert!(upsert(repo.path(), &target, "example.org", &key).is_err());
        no_temps(outside.path());
    }
    #[test]
    fn refused_publication_removes_its_private_temporary_file() {
        let repo = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let target = outside.path().join("credentials.toml");
        let key = secret("pending");
        let changed = "# concurrently created file\n";
        let result = unix::upsert_observed(repo.path(), &target, "example.org", &key, || {
            private(&target, changed);
        });
        assert!(matches!(result, Err(Error::Changed)));
        assert_eq!(fs::read_to_string(&target).unwrap(), changed);
        no_temps(outside.path());
    }
    #[test]
    fn malformed_duplicate_or_public_existing_files_remain_unchanged() {
        let repo = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let target = outside.path().join("credentials.toml");
        let key = secret("test");
        let valid = format!("[[credentials]]\nhost='example.org'\napi_key='{key}'\n");
        for text in [
            "not toml [".to_owned(),
            format!("{valid}{valid}"),
            valid.replace("example.org", "bad/path"),
            format!("{valid}unexpected=true\n"),
        ] {
            private(&target, &text);
            let error = upsert(repo.path(), &target, "another.example", &key).unwrap_err();
            assert!(!format!("{error:?} {error}").contains(&key));
            assert_eq!(fs::read_to_string(&target).unwrap(), text);
            no_temps(outside.path());
        }
        private(&target, &valid);
        fs::set_permissions(&target, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(upsert(repo.path(), &target, "example.org", &key).is_err());
        assert_eq!(fs::read_to_string(&target).unwrap(), valid);
        no_temps(outside.path());
    }
}
