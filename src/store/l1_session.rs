//! Process-shared L1 orchestration admission; v1 §4.2–4.6.

use super::{Error, Store};
use std::{
    fs::File,
    path::{Path, PathBuf},
    time::Duration,
};

/// Holds only orchestration admission, never a SQLite transaction. Closing the
/// handle releases it after cancellation or process death; the empty sidecar stays.
pub(crate) struct Session(Option<File>);

impl Drop for Session {
    fn drop(&mut self) {
        if let Some(file) = &self.0 {
            let _ = file.unlock();
        }
    }
}

impl Store {
    /// Serializes multi-transaction L1 producers across CLI, watcher and query
    /// repair. Existing generation/source fences still protect every publication.
    /// The caller's cancellation/deadline also bounds lock contention.
    pub(crate) fn l1_session(
        &self,
        root: &Path,
        check: impl Fn() -> Result<(), Error>,
    ) -> Result<Session, Error> {
        check()?;
        let root = super::path::system_path(root);
        let database = super::checked_writer::with_writer(self, &check, |writer| {
            Ok::<_, Error>(writer.path().map(PathBuf::from))
        })?;
        let Some(database) = database.filter(|path| !path.as_os_str().is_empty()) else {
            // Private in-memory fixtures have no process-shared producer.
            return Ok(Session(None));
        };
        let relative = database
            .strip_prefix(&root)
            .map_err(|_| Error::L1InputsChanged)?;
        let mut name = relative.as_os_str().to_owned();
        name.push(".l1-lock");
        let path = PathBuf::from(name);
        let file = open_lock(&root, &path).map_err(|source| Error::Io { source })?;
        let metadata = file.metadata().map_err(|source| Error::Io { source })?;
        if !metadata.is_file() {
            return Err(Error::L1InputsChanged);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if metadata.nlink() != 1 {
                return Err(Error::L1InputsChanged);
            }
        }
        loop {
            check()?;
            match file.try_lock() {
                Ok(()) => {
                    check()?;
                    return Ok(Session(Some(file)));
                }
                Err(std::fs::TryLockError::WouldBlock) => {
                    std::thread::sleep(Duration::from_millis(5))
                }
                Err(std::fs::TryLockError::Error(source)) => return Err(Error::Io { source }),
            }
        }
    }
}

// Resolve every untrusted descendant against an open directory, as source
// opening does. Leaf-only NOFOLLOW would still create bookkeeping through a
// replaced .deepkit ancestor. The canonical repository root is the trust anchor.
#[cfg(unix)]
fn open_lock(root: &Path, relative: &Path) -> std::io::Result<File> {
    use rustix::fs::{Mode, OFlags, open, openat};
    use std::path::Component;
    let parts = relative
        .components()
        .map(|part| match part {
            Component::Normal(name) => Ok(name),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "invalid L1 lock path",
            )),
        })
        .collect::<std::io::Result<Vec<_>>>()?;
    let Some((name, parents)) = parts.split_last() else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "empty L1 lock path",
        ));
    };
    let flags = OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK;
    let mut directory = open(
        root,
        flags | OFlags::RDONLY | OFlags::DIRECTORY,
        Mode::empty(),
    )?;
    for parent in parents {
        directory = openat(
            &directory,
            *parent,
            flags | OFlags::RDONLY | OFlags::DIRECTORY,
            Mode::empty(),
        )?;
    }
    Ok(File::from(openat(
        &directory,
        *name,
        flags | OFlags::RDWR | OFlags::CREATE,
        Mode::RUSR | Mode::WUSR,
    )?))
}

#[cfg(not(unix))]
fn open_lock(_root: &Path, _relative: &Path) -> std::io::Result<File> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "verified L1 coordination is not implemented on this platform",
    ))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn independent_connections_wait_cooperatively_and_release_on_drop() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("code.db");
        let first = super::super::open(&path, Default::default()).unwrap();
        let second = super::super::open(&path, Default::default()).unwrap();
        let held = first.l1_session(root.path(), || Ok(())).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_millis(30);
        assert!(
            second
                .l1_session(root.path(), || if std::time::Instant::now() >= deadline {
                    Err(Error::L1InputsChanged)
                } else {
                    Ok(())
                })
                .is_err()
        );
        drop(held);
        let _acquired = second.l1_session(root.path(), || Ok(())).unwrap();
    }

    #[test]
    fn l1_coordination_refuses_replaced_ancestors_and_linked_lock_files() {
        use std::{fs, os::unix::fs::symlink};
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let owned = root.path().join(".deepkit");
        let store = super::super::open(owned.join("code.db"), Default::default()).unwrap();
        fs::rename(&owned, root.path().join("retained")).unwrap();
        symlink(outside.path(), &owned).unwrap();
        assert!(store.l1_session(root.path(), || Ok(())).is_err());
        assert!(!outside.path().join("code.db.l1-lock").exists());
        fs::remove_file(&owned).unwrap();
        fs::rename(root.path().join("retained"), &owned).unwrap();
        fs::write(outside.path().join("keep"), "unchanged").unwrap();
        let lock = owned.join("code.db.l1-lock");
        symlink(outside.path().join("keep"), &lock).unwrap();
        assert!(store.l1_session(root.path(), || Ok(())).is_err());
        fs::remove_file(&lock).unwrap();
        fs::hard_link(outside.path().join("keep"), &lock).unwrap();
        assert!(store.l1_session(root.path(), || Ok(())).is_err());
        assert_eq!(
            fs::read_to_string(outside.path().join("keep")).unwrap(),
            "unchanged"
        );
    }
}
