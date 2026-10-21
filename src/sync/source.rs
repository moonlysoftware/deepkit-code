//! Descriptor-relative repository source opening; see `v1.md` §10.1 and §4.6.
use std::{fs::File, io, path::Path};

/// A short-lived identity for bytes observed through one source descriptor.
/// It is never persisted or used as proof for a different indexing session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Stamp {
    device: u64,
    inode: u64,
    size: u64,
    mtime: (i64, i64),
    ctime: (i64, i64),
    uid: u32,
    gid: u32,
}

impl Stamp {
    /// Requires every Git stat field, including nanoseconds and inode identity.
    pub(super) fn matches_index(self, index: &gix::index::entry::Stat) -> bool {
        self.size <= u64::from(u32::MAX)
            && self.size == u64::from(index.size)
            && self.device as u32 == index.dev
            && self.inode as u32 == index.ino
            && self.uid == index.uid
            && self.gid == index.gid
            && self.mtime == (i64::from(index.mtime.secs), i64::from(index.mtime.nsecs))
            && self.ctime == (i64::from(index.ctime.secs), i64::from(index.ctime.nsecs))
    }
}

/// Captures inode and change-time evidence in addition to ordinary size/mtime.
/// Platforms without this evidence retain full byte validation.
pub(crate) fn stamp(metadata: &std::fs::Metadata) -> Option<Stamp> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        // A seconds-only change clock can miss two writes in one tick. Treat
        // even a coincidental exact-second timestamp conservatively.
        if metadata.ctime_nsec() == 0 {
            return None;
        }
        Some(Stamp {
            device: metadata.dev(),
            inode: metadata.ino(),
            size: metadata.len(),
            mtime: (metadata.mtime(), metadata.mtime_nsec()),
            ctime: (metadata.ctime(), metadata.ctime_nsec()),
            uid: metadata.uid(),
            gid: metadata.gid(),
        })
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        None
    }
}

/// Opens a regular source beneath the trusted repository root without following
/// descendant symlinks or blocking on a FIFO substituted during traversal.
#[cfg(unix)]
pub(crate) fn open_regular(root: &Path, relative: &Path) -> io::Result<File> {
    open_regular_observed(root, relative, |_| {})
}

#[cfg(unix)]
fn open_regular_observed(
    root: &Path,
    relative: &Path,
    mut after_directory: impl FnMut(usize),
) -> io::Result<File> {
    use rustix::fs::{FileType, Mode, OFlags, fstat, open, openat};
    use std::path::Component;

    let components = relative
        .components()
        .map(|component| match component {
            Component::Normal(name) => Ok(name),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "source path must contain only normal relative components",
            )),
        })
        .collect::<io::Result<Vec<_>>>()?;
    let Some((filename, parents)) = components.split_last() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "source path is empty",
        ));
    };
    let flags = OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK;
    // The root is the caller's trusted anchor. All untrusted descendants are
    // resolved one component at a time against an already-open directory.
    let mut directory = open(root, flags | OFlags::DIRECTORY, Mode::empty())?;
    for (index, parent) in parents.iter().enumerate() {
        directory = openat(
            &directory,
            *parent,
            flags | OFlags::DIRECTORY,
            Mode::empty(),
        )?;
        after_directory(index);
    }
    let source = openat(&directory, *filename, flags, Mode::empty())?;
    if !FileType::from_raw_mode(fstat(&source)?.st_mode).is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "source descriptor is not a regular file",
        ));
    }
    Ok(File::from(source))
}

/// Fails closed until descriptor-relative source opening is implemented and
/// verified on non-Unix platforms; pathname checks cannot provide this contract.
#[cfg(not(unix))]
pub(crate) fn open_regular(_root: &Path, _relative: &Path) -> io::Result<File> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "verified source opening is not implemented on this platform",
    ))
}

/// Creates a FIFO fixture, using the system utility where rustix lacks mkfifoat.
#[cfg(all(test, unix))]
pub(crate) fn mkfifo(path: &Path, mode: rustix::fs::Mode) -> io::Result<()> {
    #[cfg(not(target_vendor = "apple"))]
    {
        rustix::fs::mkfifoat(rustix::fs::CWD, path, mode).map_err(io::Error::from)
    }
    #[cfg(target_vendor = "apple")]
    {
        let status = std::process::Command::new("mkfifo")
            .arg("-m")
            .arg(format!("{:o}", mode.bits()))
            .arg(path)
            .status()?;
        if status.success() {
            Ok(())
        } else {
            Err(io::Error::other("mkfifo fixture creation failed"))
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::{fs, io::Read, os::unix::fs::symlink, sync::mpsc, time::Duration};

    #[test]
    fn nested_regular_source_and_invalid_paths() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir_all(root.path().join("src/nested")).unwrap();
        fs::write(root.path().join("src/nested/source.rs"), "original source").unwrap();
        let mut text = String::new();
        open_regular(root.path(), Path::new("src/nested/source.rs"))
            .unwrap()
            .read_to_string(&mut text)
            .unwrap();
        assert_eq!(text, "original source");
        for path in [
            "",
            ".",
            "../source.rs",
            "src/../source.rs",
            "/source.rs",
            "src",
        ] {
            assert!(
                open_regular(root.path(), Path::new(path)).is_err(),
                "{path}"
            );
        }
    }

    #[test]
    fn symlinks_cannot_redirect_source_or_ancestors() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("source.rs"), "outside source").unwrap();
        symlink(outside.path(), root.path().join("src")).unwrap();
        symlink(
            outside.path().join("source.rs"),
            root.path().join("source.rs"),
        )
        .unwrap();
        assert!(open_regular(root.path(), Path::new("src/source.rs")).is_err());
        assert!(open_regular(root.path(), Path::new("source.rs")).is_err());
    }

    #[test]
    fn replacing_opened_ancestor_cannot_redirect_remaining_traversal() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::create_dir_all(root.path().join("src/nested")).unwrap();
        fs::create_dir(outside.path().join("nested")).unwrap();
        fs::write(root.path().join("src/nested/source.rs"), "original source").unwrap();
        fs::write(outside.path().join("nested/source.rs"), "outside source").unwrap();
        let mut source =
            open_regular_observed(root.path(), Path::new("src/nested/source.rs"), |index| {
                if index == 0 {
                    fs::rename(root.path().join("src"), root.path().join("retained")).unwrap();
                    symlink(outside.path(), root.path().join("src")).unwrap();
                }
            })
            .unwrap();
        let mut text = String::new();
        source.read_to_string(&mut text).unwrap();
        assert_eq!(text, "original source");
    }

    #[test]
    fn fifo_replacement_is_rejected_without_waiting_for_a_writer() {
        use rustix::fs::Mode;

        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("src")).unwrap();
        fs::write(root.path().join("src/source.rs"), "original source").unwrap();
        let (send, receive) = mpsc::channel();
        // A regression must fail the assertion within a bounded time; joining
        // a thread blocked in a FIFO open would hang the test process instead.
        std::thread::spawn(move || {
            let result = open_regular_observed(root.path(), Path::new("src/source.rs"), |_| {
                let path = root.path().join("src/source.rs");
                fs::remove_file(&path).unwrap();
                mkfifo(&path, Mode::RUSR | Mode::WUSR).unwrap();
            });
            send.send(result.is_err()).unwrap();
        });
        assert!(receive.recv_timeout(Duration::from_secs(2)).unwrap());
    }
}
