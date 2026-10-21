//! Affected-path discovery for watcher publication; v1.md §4.1–4.5.

use super::{Error, Scan, SyncLimits, SyncStatus, discovery, l1::Work};
use crate::store::IndexSnapshot;
use std::{collections::BTreeSet, path::Path};

/// Observes only the event scope through bounded descriptor reads. Unobserved
/// siblings are never hashed and cannot become deletion candidates.
pub(crate) fn scan(
    root: &Path,
    paths: &BTreeSet<String>,
    limits: SyncLimits,
    snapshot: &IndexSnapshot,
    work: Work<'_>,
) -> Result<Scan, Error> {
    work.check()?;
    let existing = snapshot
        .files
        .iter()
        .filter(|file| discovery::affected(paths, &file.path))
        .count();
    let unaffected = snapshot.files.len().saturating_sub(existing);
    // Lowering a cap retains existing rows. The store still checks the global
    // count before admitting any new path in the atomic catalogue transaction.
    let allowance = limits.max_files.saturating_sub(unaffected).max(existing);
    let candidates = discovery::discover_paths(root, paths, allowance, work)?;
    let files = candidates
        .paths
        .iter()
        .map(|path| super::l1::observe_source(root, path, limits.max_file_bytes, work, |_| {}))
        .collect::<Result<Vec<_>, _>>()?;
    if super::head::read(root, work)? != candidates.head {
        return Err(Error::HeadChanged);
    }
    let status = SyncStatus {
        max_files_reached: candidates.max_files_reached,
        files_over_max_bytes: files.iter().filter(|file| file.content_capped).count(),
        skipped_non_unicode_paths: candidates.skipped_non_unicode_paths,
        messages: if candidates.max_files_reached {
            vec![super::MAX_FILES_MESSAGE.into()]
        } else {
            Vec::new()
        },
        ..SyncStatus::default()
    };
    Ok(Scan {
        files,
        dirs: Vec::new(),
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::Config,
        store::{self, Store},
        sync::index,
    };
    use std::{fs, sync::atomic::AtomicBool};

    fn fixture() -> (tempfile::TempDir, Config, Store) {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join(".deepkit")).unwrap();
        fs::create_dir(root.path().join("nested")).unwrap();
        let mut settings = Config::default();
        settings.embedder.profile = "local".into();
        settings.llm.enabled = false;
        fs::write(
            root.path().join(".deepkit/config.toml"),
            settings.dump().unwrap(),
        )
        .unwrap();
        for (name, contents) in [
            ("a.rs", "pub fn before() {}\n"),
            ("other.rs", "pub fn unrelated() {}\n"),
            ("nested/old.rs", "pub fn renamed() {}\n"),
        ] {
            fs::write(root.path().join(name), contents).unwrap();
        }
        let settings = crate::config::load(root.path(), None).unwrap().config;
        let store = store::open(root.path().join(".deepkit/code.db"), Default::default()).unwrap();
        index::run(input(root.path(), &settings, &store), |_| {}).unwrap();
        (root, settings, store)
    }
    fn input<'a>(root: &'a Path, settings: &'a Config, store: &'a Store) -> index::Input<'a> {
        index::Input {
            config_root: root,
            root,
            settings,
            store,
            full: false,
            work: Work::default(),
        }
    }

    #[test]
    fn watcher_scope_changes_only_affected_files_and_preserves_rename_identity() {
        let (root, settings, store) = fixture();
        let before = store.index_snapshot().unwrap();
        let identity = store.l1_identity().unwrap();
        let renamed = before
            .files
            .iter()
            .find(|f| f.path == "nested/old.rs")
            .unwrap()
            .id;
        let rows: Vec<i64> = store
            .writer()
            .unwrap()
            .prepare("SELECT id FROM units WHERE file_id=?1 ORDER BY id")
            .unwrap()
            .query_map([renamed], |r| r.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        fs::write(root.path().join("a.rs"), "pub fn after() {}\n").unwrap();
        fs::write(root.path().join("added.rs"), "pub fn added() {}\n").unwrap();
        fs::rename(
            root.path().join("nested/old.rs"),
            root.path().join("nested/new.rs"),
        )
        .unwrap();
        // No event covers this deletion; reading or deleting it would exceed scope.
        fs::remove_file(root.path().join("other.rs")).unwrap();
        let paths = ["a.rs", "added.rs", "nested/old.rs", "nested/new.rs"]
            .map(String::from)
            .into();
        let report =
            index::run_paths(input(root.path(), &settings, &store), &paths, |_| {}).unwrap();
        assert_eq!(report.scan.files().count(), 3);
        assert_eq!(report.publication.changes.modified, ["a.rs"]);
        assert_eq!(report.publication.changes.added, ["added.rs"]);
        assert_eq!(report.publication.changes.renamed.len(), 1);
        assert!(report.publication.changes.deleted.is_empty());
        let after = store.index_snapshot().unwrap();
        assert_eq!(
            after
                .files
                .iter()
                .find(|f| f.path == "nested/new.rs")
                .unwrap()
                .id,
            renamed
        );
        assert_eq!(
            after.files.iter().find(|f| f.path == "other.rs"),
            before.files.iter().find(|f| f.path == "other.rs")
        );
        let after_rows: Vec<i64> = store
            .writer()
            .unwrap()
            .prepare("SELECT id FROM units WHERE file_id=?1 ORDER BY id")
            .unwrap()
            .query_map([renamed], |r| r.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(rows, after_rows);
        assert!(store.pending_l1_paths().unwrap().is_empty());
        assert_eq!(
            store.l1_identity().unwrap().indexed_head,
            identity.indexed_head
        );
        assert_eq!(store.l1_identity().unwrap().completed, identity.completed);
        assert_eq!(store.index_snapshot().unwrap().dirs, before.dirs);
        assert_eq!(
            store.get_meta("graph_pending").unwrap().as_deref(),
            Some("0")
        );
        let deletion = index::run_paths(
            input(root.path(), &settings, &store),
            &["other.rs".into()].into(),
            |_| {},
        )
        .unwrap();
        assert_eq!(deletion.publication.changes.deleted, ["other.rs"]);
    }

    #[test]
    fn watcher_scope_rechecks_ignores_and_refuses_cancellation_and_capacity() {
        let (root, mut settings, store) = fixture();
        let before = store.index_snapshot().unwrap();
        let cancelled = AtomicBool::new(true);
        let mut stopped = input(root.path(), &settings, &store);
        stopped.work.cancelled = Some(&cancelled);
        assert!(index::run_paths(stopped, &["a.rs".into()].into(), |_| {}).is_err());
        assert_eq!(store.index_snapshot().unwrap(), before);

        fs::write(root.path().join("a.rs"), "pub fn replacement() {}\n").unwrap();
        let changed = index::run_paths(
            input(root.path(), &settings, &store),
            &["a.rs".into()].into(),
            |progress| {
                if matches!(progress, index::Progress::Files { processed: 0, .. }) {
                    fs::write(root.path().join(".deepkitignore"), "a.rs\n").unwrap();
                }
            },
        );
        assert!(changed.is_err());
        assert!(store.pending_l1_paths().unwrap().contains(&"a.rs".into()));
        let removed = index::run_paths(
            input(root.path(), &settings, &store),
            &["a.rs".into()].into(),
            |_| {},
        )
        .unwrap();
        assert_eq!(removed.publication.changes.deleted, ["a.rs"]);
        assert_eq!(removed.scan.files().count(), 0);

        settings.limits.max_files = 2;
        fs::write(
            root.path().join(".deepkit/config.toml"),
            settings.dump().unwrap(),
        )
        .unwrap();
        fs::write(root.path().join("new.rs"), "pub fn new() {}\n").unwrap();
        let capped = index::run_paths(
            input(root.path(), &settings, &store),
            &["new.rs".into()].into(),
            |_| {},
        )
        .unwrap();
        assert!(!capped.scan.is_complete());
        assert_eq!(store.counts().unwrap().files, 2);
        assert!(capped.publication.changes.deleted.is_empty());
        assert!(
            index::run_paths(
                input(root.path(), &settings, &store),
                &["../outside".into()].into(),
                |_| {}
            )
            .is_err()
        );
    }
}
