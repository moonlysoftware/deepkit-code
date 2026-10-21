//! Shared index destination identity preflight; v1 §2.1–2.2/§13.
use std::{
    fs,
    path::{Component, Path, PathBuf},
};

/// Static diagnoses do not expose repository configuration or database contents.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("{0}")]
    Invalid(&'static str),
    #[error("Bootstrap database destination could not be inspected")]
    Io(#[from] std::io::Error),
}

#[cfg(unix)]
#[derive(Debug, PartialEq, Eq)]
struct Identity {
    device: u64,
    inode: u64,
    mode: u32,
    links: u64,
}

#[cfg(unix)]
fn identity(metadata: &fs::Metadata) -> Identity {
    use std::os::unix::fs::MetadataExt;
    Identity {
        device: metadata.dev(),
        inode: metadata.ino(),
        mode: metadata.mode(),
        // Creating an unrelated child (including our .git/info excludes) changes
        // a directory's link count without replacing that directory.
        links: if metadata.is_dir() {
            0
        } else {
            metadata.nlink()
        },
    }
}

#[cfg(unix)]
#[derive(Debug, PartialEq, Eq)]
struct Snapshot {
    root: Identity,
    parents: Vec<Option<Identity>>,
    files: Vec<Option<Identity>>,
}

/// Captured destination identities for a second read-only check before SQLite open.
/// This is a preflight, not an atomic reservation: pathname swaps after check remain
/// possible because SQLite subsequently opens its own pathname and sidecars.
#[cfg(unix)]
pub(crate) struct Prepared {
    root: PathBuf,
    database: PathBuf,
    before: Snapshot,
}

#[cfg(not(unix))]
pub(crate) struct Prepared;

/// Checks only metadata; large databases are permitted and contents are never opened.
/// The caller supplies its already canonical repository root. Relative database paths
/// are resolved there; absolute paths must still remain under its .deepkit directory.
pub(crate) fn prepare(root: &Path, configured_database: &Path) -> Result<Prepared, Error> {
    #[cfg(unix)]
    {
        let root = system_path(root);
        if !root.is_absolute() || fs::canonicalize(&root)? != root {
            return Err(Error::Invalid(
                "Bootstrap repository root must be canonical",
            ));
        }
        // Refuse parent traversal rather than assigning potentially surprising
        // semantics to paths that leave and re-enter the owned directory.
        if configured_database
            .components()
            .any(|part| matches!(part, Component::ParentDir))
        {
            return Err(Error::Invalid(
                "Bootstrap database path must not contain parent traversal",
            ));
        }
        let candidate = if configured_database.is_absolute() {
            system_path(configured_database)
        } else {
            root.join(configured_database)
        };
        let database: PathBuf = candidate
            .components()
            .filter(|part| !matches!(part, Component::CurDir))
            .collect();
        let owned = root.join(".deepkit");
        if database == owned || !database.starts_with(&owned) || database.file_name().is_none() {
            return Err(Error::Invalid(
                "Bootstrap database must be inside the repository .deepkit directory",
            ));
        }
        let before = snapshot(&root, &database)?;
        Ok(Prepared {
            root,
            database,
            before,
        })
    }
    #[cfg(not(unix))]
    {
        let _ = (root, configured_database);
        Err(Error::Invalid(
            "Checked bootstrap database setup is not implemented on this platform",
        ))
    }
}

impl Prepared {
    /// Removes only the admitted database and WAL sidecars inside .deepkit.
    /// Descriptor-relative traversal prevents a renamed or linked parent from
    /// redirecting destructive cleanup to an unrelated filesystem location.
    pub(crate) fn remove(self) -> Result<Vec<PathBuf>, Error> {
        #[cfg(unix)]
        {
            use rustix::fs::{self as rfs, AtFlags, FileType, Mode, OFlags};

            self.check()?;
            let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
            let mut directory =
                rfs::open(&self.root, flags, Mode::empty()).map_err(std::io::Error::from)?;
            let metadata = fs::File::from(directory.try_clone()?).metadata()?;
            if identity(&metadata) != self.before.root {
                return Err(Error::Invalid("Cleanup repository identity changed"));
            }
            let parent = self
                .database
                .parent()
                .ok_or(Error::Invalid("Cleanup database has no parent"))?;
            let relative = parent.strip_prefix(&self.root).map_err(|_| {
                Error::Invalid("Cleanup database must remain inside its repository")
            })?;
            for (index, component) in relative.components().enumerate() {
                let Component::Normal(name) = component else {
                    return Err(Error::Invalid("Cleanup database ancestry is invalid"));
                };
                directory = match rfs::openat(&directory, name, flags, Mode::empty()) {
                    Ok(next) => next,
                    Err(rustix::io::Errno::NOENT) => {
                        self.check()?;
                        return Ok(Vec::new());
                    }
                    Err(error) => return Err(std::io::Error::from(error).into()),
                };
                let metadata = fs::File::from(directory.try_clone()?).metadata()?;
                if self.before.parents.get(index) != Some(&Some(identity(&metadata))) {
                    return Err(Error::Invalid("Cleanup database ancestry changed"));
                }
            }
            // Check every destination before deleting any. Unlinking is relative
            // to the verified directory, never a newly resolved absolute path.
            self.check()?;
            let name = self
                .database
                .file_name()
                .ok_or(Error::Invalid("Cleanup database has no filename"))?;
            let mut removed = Vec::new();
            for suffix in ["", "-wal", "-shm"] {
                let mut filename = name.to_os_string();
                filename.push(suffix);
                match rfs::statat(&directory, &filename, AtFlags::SYMLINK_NOFOLLOW) {
                    Ok(stat)
                        if FileType::from_raw_mode(stat.st_mode).is_file()
                            && stat.st_nlink == 1 => {}
                    Ok(_) => {
                        return Err(Error::Invalid(
                            "Cleanup database and sidecars must be single-link regular files",
                        ));
                    }
                    Err(rustix::io::Errno::NOENT) => continue,
                    Err(error) => return Err(std::io::Error::from(error).into()),
                }
                rfs::unlinkat(&directory, &filename, AtFlags::empty())
                    .map_err(std::io::Error::from)?;
                removed.push(parent.join(filename));
            }
            Ok(removed)
        }
        #[cfg(not(unix))]
        {
            Err(Error::Invalid(
                "Checked database cleanup is not implemented on this platform",
            ))
        }
    }

    /// Rechecks an existing database and its ancestry while allowing SQLite to
    /// create/remove safe ordinary WAL bookkeeping files. The database itself
    /// must retain its original identity; no missing database is admitted.
    pub(crate) fn check_existing_database(&self) -> Result<(), Error> {
        #[cfg(unix)]
        {
            let current = snapshot(&self.root, &self.database)?;
            if self.before.files.first().is_none_or(Option::is_none)
                || current.root != self.before.root
                || current.parents != self.before.parents
                || current.files.first() != self.before.files.first()
            {
                return Err(Error::Invalid(
                    "Existing database identity changed or is absent",
                ));
            }
            Ok(())
        }
        #[cfg(not(unix))]
        {
            Err(Error::Invalid(
                "Checked existing database opening is not implemented on this platform",
            ))
        }
    }

    /// Rechecks an ordinary open using the captured database's existence. An
    /// existing database may gain or lose safe SQLite sidecars while another
    /// process opens it; an initially absent database keeps the strict snapshot.
    /// Destructive cleanup must continue to use `check` instead.
    pub(crate) fn check_for_open(&self) -> Result<(), Error> {
        #[cfg(unix)]
        if self.before.files.first().is_some_and(Option::is_some) {
            return self.check_existing_database();
        }
        self.check()
    }

    /// Refuses changed ancestry or file identities immediately before ordinary SQLite open.
    /// No file or directory is created, locked or reserved by this check.
    pub(crate) fn check(&self) -> Result<(), Error> {
        #[cfg(unix)]
        {
            if snapshot(&self.root, &self.database)? != self.before {
                return Err(Error::Invalid(
                    "Bootstrap database destination changed during setup",
                ));
            }
            Ok(())
        }
        #[cfg(not(unix))]
        {
            Err(Error::Invalid(
                "Checked bootstrap database setup is not implemented on this platform",
            ))
        }
    }
}

/// Normalizes macOS system aliases without resolving repository-controlled links.
/// SQLite NOFOLLOW and descriptor walks must still inspect every user component.
pub(crate) fn system_path(path: &Path) -> PathBuf {
    #[cfg(target_os = "macos")]
    for (alias, physical) in [
        ("/var", "/private/var"),
        ("/tmp", "/private/tmp"),
        ("/etc", "/private/etc"),
    ] {
        if let Ok(rest) = path.strip_prefix(alias) {
            return Path::new(physical).join(rest);
        }
    }
    path.to_path_buf()
}

#[cfg(unix)]
fn metadata(path: &Path) -> Result<Option<fs::Metadata>, Error> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

#[cfg(unix)]
fn snapshot(root: &Path, database: &Path) -> Result<Snapshot, Error> {
    if fs::canonicalize(root)? != root {
        return Err(Error::Invalid("Bootstrap repository ancestry changed"));
    }
    let root_metadata = fs::symlink_metadata(root)?;
    if !root_metadata.file_type().is_dir() {
        return Err(Error::Invalid(
            "Bootstrap repository must be an ordinary directory",
        ));
    }
    let parent = database
        .parent()
        .ok_or(Error::Invalid("Bootstrap database has no parent"))?;
    let relative = parent
        .strip_prefix(root)
        .map_err(|_| Error::Invalid("Bootstrap database must remain inside its repository"))?;
    let mut current = root.to_path_buf();
    let mut parents = Vec::new();
    let mut absent = false;
    for part in relative.components() {
        current.push(part);
        if absent {
            parents.push(None);
            continue;
        }
        match metadata(&current)? {
            Some(metadata) => {
                if !metadata.file_type().is_dir() {
                    return Err(Error::Invalid(
                        "Bootstrap database ancestry must contain ordinary directories",
                    ));
                }
                parents.push(Some(identity(&metadata)));
            }
            None => {
                absent = true;
                parents.push(None);
            }
        }
    }
    let mut files = Vec::new();
    for suffix in ["", "-wal", "-shm", "-journal"] {
        let mut path = database.as_os_str().to_os_string();
        path.push(suffix);
        let metadata = if absent {
            None
        } else {
            metadata(Path::new(&path))?
        };
        files.push(match metadata {
            Some(metadata) => {
                use std::os::unix::fs::MetadataExt;
                if !metadata.file_type().is_file() || metadata.nlink() != 1 {
                    return Err(Error::Invalid(
                        "Bootstrap database and sidecars must be single-link regular files",
                    ));
                }
                Some(identity(&metadata))
            }
            None => None,
        });
    }
    Ok(Snapshot {
        root: identity(&root_metadata),
        parents,
        files,
    })
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    #[cfg(unix)]
    use crate::sync::source::mkfifo;

    #[test]
    fn canonical_root_and_owned_descendants_must_not_hide_symlinks() {
        let temporary = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(temporary.path()).unwrap();
        fs::create_dir_all(root.join("repo/.deepkit/real")).unwrap();
        symlink(root.join("repo"), root.join("alias")).unwrap();
        assert!(prepare(&root.join("alias"), Path::new(".deepkit/code.db")).is_err());
        let repo = root.join("repo");
        symlink(repo.join(".deepkit/real"), repo.join(".deepkit/link")).unwrap();
        assert!(prepare(&repo, Path::new(".deepkit/link/code.db")).is_err());
        fs::write(repo.join(".deepkit/real/code.db"), "unchanged").unwrap();
        assert!(prepare(&repo, Path::new(".deepkit/link/code.db")).is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn system_aliases_preserve_all_repository_components() {
        for (alias, physical) in [
            ("/var", "/private/var"),
            ("/tmp", "/private/tmp"),
            ("/etc", "/private/etc"),
        ] {
            let path = Path::new(alias).join("repo/link/../code.db");
            assert_eq!(
                system_path(&path),
                Path::new(physical).join("repo/link/../code.db")
            );
        }
        for path in ["/variable/repo", "/private/var/repo", "repo/link"] {
            assert_eq!(system_path(Path::new(path)), Path::new(path));
        }
        let root = tempfile::tempdir().unwrap();
        prepare(root.path(), &root.path().join(".deepkit/code.db"))
            .unwrap()
            .check()
            .unwrap();
    }

    #[test]
    fn fresh_and_existing_large_database_are_read_only_and_contained() {
        let root = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(root.path()).unwrap();
        prepare(&root, Path::new(".deepkit/code.db"))
            .unwrap()
            .check()
            .unwrap();
        assert_eq!(fs::read_dir(&root).unwrap().count(), 0);
        fs::create_dir(root.join(".deepkit")).unwrap();
        let database = root.join(".deepkit/code.db");
        fs::File::create(&database)
            .unwrap()
            .set_len(2 * 1024 * 1024)
            .unwrap();
        for suffix in ["-wal", "-shm", "-journal"] {
            fs::write(root.join(format!(".deepkit/code.db{suffix}")), "unchanged").unwrap();
        }
        prepare(&root, &database).unwrap().check().unwrap();
        assert_eq!(fs::metadata(&database).unwrap().len(), 2 * 1024 * 1024);
        for suffix in ["-wal", "-shm", "-journal"] {
            assert_eq!(
                fs::read_to_string(root.join(format!(".deepkit/code.db{suffix}"))).unwrap(),
                "unchanged"
            );
        }
        for path in [
            root.join("outside.db"),
            root.join(".deepkit"),
            root.join(".deepkit/../outside.db"),
        ] {
            assert!(prepare(&root, &path).is_err());
        }
    }

    #[test]
    fn symlink_ancestry_and_every_hostile_database_sidecar_are_refused() {
        for name in ["code.db", "code.db-wal", "code.db-shm", "code.db-journal"] {
            for kind in ["symlink", "dangling", "hardlink", "fifo", "directory"] {
                let root = tempfile::tempdir().unwrap();
                let outside = tempfile::tempdir().unwrap();
                fs::create_dir(root.path().join(".deepkit")).unwrap();
                let external = outside.path().join("untouched");
                fs::write(&external, "kept").unwrap();
                let target = root.path().join(".deepkit").join(name);
                match kind {
                    "symlink" => symlink(&external, &target).unwrap(),
                    "dangling" => symlink(outside.path().join("missing"), &target).unwrap(),
                    "hardlink" => fs::hard_link(&external, &target).unwrap(),
                    "fifo" => {
                        mkfifo(&target, rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR).unwrap()
                    }
                    _ => fs::create_dir(&target).unwrap(),
                }
                assert!(
                    prepare(
                        &fs::canonicalize(root.path()).unwrap(),
                        Path::new(".deepkit/code.db")
                    )
                    .is_err(),
                    "{name}/{kind}"
                );
                assert_eq!(fs::read_to_string(&external).unwrap(), "kept");
                assert!(!outside.path().join("missing").exists());
                assert_eq!(fs::read_dir(outside.path()).unwrap().count(), 1);
            }
        }
        for nested in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let outside = tempfile::tempdir().unwrap();
            let relative = if nested {
                fs::create_dir(root.path().join(".deepkit")).unwrap();
                symlink(outside.path(), root.path().join(".deepkit/nested")).unwrap();
                ".deepkit/nested/code.db"
            } else {
                symlink(outside.path(), root.path().join(".deepkit")).unwrap();
                ".deepkit/code.db"
            };
            assert!(prepare(&fs::canonicalize(root.path()).unwrap(), Path::new(relative)).is_err());
            assert_eq!(fs::read_dir(outside.path()).unwrap().count(), 0);
        }
    }

    #[test]
    fn snapshots_refuse_parent_or_database_replacement_before_open() {
        let root = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(root.path()).unwrap();
        let prepared = prepare(&root, Path::new(".deepkit/code.db")).unwrap();
        fs::create_dir(root.join(".deepkit")).unwrap();
        assert!(prepared.check().is_err());
        assert!(prepared.check_for_open().is_err());
        fs::write(root.join(".deepkit/code.db"), "original").unwrap();
        let prepared = prepare(&root, Path::new(".deepkit/code.db")).unwrap();
        fs::rename(root.join(".deepkit/code.db"), root.join(".deepkit/old.db")).unwrap();
        fs::write(root.join(".deepkit/code.db"), "replacement").unwrap();
        assert!(prepared.check().is_err());
        assert!(prepared.check_for_open().is_err());
        let prepared = prepare(&root, Path::new(".deepkit/code.db")).unwrap();
        fs::rename(root.join(".deepkit"), root.join("old")).unwrap();
        fs::create_dir(root.join(".deepkit")).unwrap();
        fs::write(root.join(".deepkit/code.db"), "replacement").unwrap();
        assert!(prepared.check().is_err());
        assert!(prepared.check_for_open().is_err());
    }

    #[test]
    fn existing_database_open_allows_safe_sidecar_churn_but_cleanup_stays_strict() {
        for suffix in ["-wal", "-shm", "-journal"] {
            let root = tempfile::tempdir().unwrap();
            let root = fs::canonicalize(root.path()).unwrap();
            fs::create_dir(root.join(".deepkit")).unwrap();
            let database = root.join(".deepkit/code.db");
            let sidecar = root.join(format!(".deepkit/code.db{suffix}"));
            fs::write(&database, "unchanged database").unwrap();

            let absent = prepare(&root, &database).unwrap();
            fs::write(&sidecar, "first sidecar").unwrap();
            absent.check_for_open().unwrap();
            absent.check_existing_database().unwrap();
            assert!(absent.check().is_err());

            let present = prepare(&root, &database).unwrap();
            fs::remove_file(&sidecar).unwrap();
            present.check_for_open().unwrap();
            present.check_existing_database().unwrap();
            assert!(present.check().is_err());

            fs::write(&sidecar, "second sidecar").unwrap();
            let replaced = prepare(&root, &database).unwrap();
            // Keep the old inode alive so the replacement cannot reuse it.
            fs::rename(&sidecar, root.join(".deepkit/retained-sidecar")).unwrap();
            fs::write(&sidecar, "replacement sidecar").unwrap();
            replaced.check_for_open().unwrap();
            replaced.check_existing_database().unwrap();
            assert!(replaced.check().is_err());
            assert_eq!(fs::read_to_string(&database).unwrap(), "unchanged database");
        }
    }

    #[test]
    fn open_keeps_strict_snapshot_when_database_was_initially_absent() {
        let root = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(root.path()).unwrap();
        fs::create_dir(root.join(".deepkit")).unwrap();
        let database = root.join(".deepkit/code.db");
        let prepared = prepare(&root, &database).unwrap();
        prepared.check_for_open().unwrap();
        assert!(prepared.check_existing_database().is_err());
        fs::write(&database, "appeared after capture").unwrap();
        assert!(prepared.check_for_open().is_err());

        let existing = prepare(&root, &database).unwrap();
        fs::remove_file(&database).unwrap();
        assert!(existing.check_for_open().is_err());

        let absent = prepare(&root, &database).unwrap();
        fs::write(root.join(".deepkit/code.db-wal"), "appeared after capture").unwrap();
        assert!(absent.check_for_open().is_err());
    }

    #[test]
    fn existing_database_open_refuses_hostile_sidecars_created_after_capture() {
        for suffix in ["-wal", "-shm", "-journal"] {
            for kind in ["symlink", "dangling", "hardlink", "fifo", "directory"] {
                let root = tempfile::tempdir().unwrap();
                let root = fs::canonicalize(root.path()).unwrap();
                let outside = tempfile::tempdir().unwrap();
                fs::create_dir(root.join(".deepkit")).unwrap();
                let database = root.join(".deepkit/code.db");
                let sidecar = root.join(format!(".deepkit/code.db{suffix}"));
                let external = outside.path().join("untouched");
                fs::write(&database, "unchanged database").unwrap();
                fs::write(&external, "kept").unwrap();
                let prepared = prepare(&root, &database).unwrap();
                match kind {
                    "symlink" => symlink(&external, &sidecar).unwrap(),
                    "dangling" => symlink(outside.path().join("missing"), &sidecar).unwrap(),
                    "hardlink" => fs::hard_link(&external, &sidecar).unwrap(),
                    "fifo" => {
                        mkfifo(&sidecar, rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR).unwrap()
                    }
                    _ => fs::create_dir(&sidecar).unwrap(),
                }
                assert!(prepared.check_for_open().is_err(), "{suffix}/{kind}");
                assert!(
                    prepared.check_existing_database().is_err(),
                    "{suffix}/{kind}"
                );
                assert_eq!(fs::read_to_string(&external).unwrap(), "kept");
                assert_eq!(fs::read_to_string(&database).unwrap(), "unchanged database");
                assert!(!outside.path().join("missing").exists());
            }
        }
    }

    #[test]
    fn owned_exclude_directories_do_not_invalidate_database_ancestry() {
        let root = tempfile::tempdir().unwrap();
        let canonical = fs::canonicalize(root.path()).unwrap();
        let prepared = prepare(&canonical, Path::new(".deepkit/code.db")).unwrap();
        crate::sync::setup::prepare_excludes(&canonical)
            .unwrap()
            .publish()
            .unwrap();
        prepared.check().unwrap();
        assert!(!canonical.join(".deepkit").exists());
        assert!(canonical.join(".git/info/exclude").is_file());
    }
}
