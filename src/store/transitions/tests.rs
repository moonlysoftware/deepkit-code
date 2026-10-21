//! Persisted routing generations, kind-specific completion and writer races.
use super::*;
use crate::store::{
    self, StoreLimits,
    vectors::{VectorInsert, VectorTableSpec},
};
fn routes(table: &str) -> Routes {
    UnitKind::ALL
        .into_iter()
        .map(|kind| (kind.as_str().into(), table.into()))
        .collect()
}
fn prepare(store: &Store, desired: &Routes) -> Result<State, Error> {
    store.prepare_transition(|_| Ok::<_, Error>(desired.clone()))
}
fn state(store: &Store) -> State {
    store.read_snapshot(read).unwrap().unwrap().unwrap()
}
fn table(store: &Store, slug: &str, known: bool) {
    store
        .ensure_vector_table(VectorTableSpec {
            slug,
            model: slug,
            profile: "ollama",
            dims: 2,
            created_at: 0,
            active: true,
        })
        .unwrap();
    if known {
        store
            .bind_embedding_table(
                slug,
                &embedding_bindings::Binding {
                    profile: "ollama".into(),
                    model: slug.into(),
                    dims: 2,
                    endpoint: format!("http://localhost/{slug}/api/embed"),
                    artifact_hash: None,
                    query_prefix: "query: ".into(),
                    api_key_env: None,
                },
            )
            .unwrap();
    }
}
fn put(store: &Store, table: &str, kind: UnitKind, ordinal: u8) -> (i64, [u8; 32]) {
    let text = format!("fixture {} {ordinal}", kind.as_str());
    let hash = *blake3::hash(text.as_bytes()).as_bytes();
    let id = store
        .insert_unit(store::UnitInsert {
            hash: &hash,
            source_key: Some(&text),
            duplicate_ordinal: 0,
            file_id: None,
            kind,
            byte_start: None,
            byte_end: None,
            line_start: None,
            line_end: None,
            scope_path: None,
            header: "fixture",
            text: &text,
            tokens: 1,
            ladder_level: 1,
            parent_unit: None,
            created_at: 0,
            updated_at: 0,
            fts: store::FtsColumns {
                header: "fixture",
                idents: "",
                body: &text,
                prose: "",
            },
        })
        .unwrap();
    replace(store, table, id, &hash);
    (id, hash)
}
fn replace(store: &Store, table: &str, id: i64, hash: &[u8; 32]) {
    store
        .replace_vector(VectorInsert {
            table_slug: table,
            unit_id: id,
            ctx_hash: hash,
            embedding_hash: hash,
            embedding: &[1.0, 0.0],
        })
        .unwrap();
}
#[test]
fn resume_rejects_another_change_and_completion_counts_only_affected_kinds() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("index.db");
    let store = store::open(&path, StoreLimits::default()).unwrap();
    for slug in ["a", "b", "c"] {
        table(&store, slug, true);
    }
    assert_eq!(prepare(&store, &routes("a")).unwrap().generation, 1);
    put(&store, "a", UnitKind::Commit, 1);
    let (history, hash) = put(&store, "a", UnitKind::History, 2);
    let mut desired = routes("a");
    desired.insert("history".into(), "b".into());
    let pending = prepare(&store, &desired).unwrap();
    assert_eq!(pending.generation, 2);
    assert_eq!(pending.previous.len(), 1);
    assert_eq!(
        pending.previous[0].kinds,
        BTreeSet::from(["history".into()])
    );
    assert!(pending.previous[0].known);
    assert!(matches!(
        prepare(&store, &routes("c")),
        Err(Error::EmbeddingTransition { .. })
    ));
    drop(store);
    let store = store::open(&path, StoreLimits::default()).unwrap();
    assert_eq!(state(&store), pending);
    assert_eq!(prepare(&store, &desired).unwrap(), pending);
    assert_eq!(store.finish_transition(2).unwrap(), pending);
    replace(&store, "b", history, &hash);
    assert!(store.finish_transition(2).unwrap().previous.is_empty());
    assert_eq!(store.count_vectors().unwrap(), 2); // Unchanged commit still lives in a.
    let third = prepare(&store, &routes("c")).unwrap();
    assert_eq!(third.generation, 3);
    assert_eq!(third.previous.len(), 2);
    assert!(store.finish_transition(2).is_err());
    assert_eq!(state(&store), third);
}
#[test]
fn competing_connections_admit_only_one_change() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("index.db");
    let store = store::open(&path, StoreLimits::default()).unwrap();
    for slug in ["a", "b", "c"] {
        table(&store, slug, true);
    }
    prepare(&store, &routes("a")).unwrap();
    put(&store, "a", UnitKind::Commit, 1);
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
    let threads = ["b", "c"].map(|target| {
        let path = path.clone();
        let barrier = barrier.clone();
        std::thread::spawn(move || {
            let store = store::open(path, StoreLimits::default()).unwrap();
            barrier.wait();
            prepare(&store, &routes(target))
        })
    });
    barrier.wait();
    let results = threads.map(|thread| thread.join().unwrap());
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(result, Err(Error::EmbeddingTransition { .. })))
            .count(),
        1
    );
    assert_eq!(state(&store).generation, 2);
    assert_eq!(store.count_vectors().unwrap(), 1);
}
#[test]
fn capacity_refusal_preserves_routes_and_resumes_without_inventing_completion() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("index.db");
    let mut store = store::open(path, StoreLimits::default()).unwrap();
    table(&store, "a", true);
    table(&store, "b", true);
    let initial = prepare(&store, &routes("a")).unwrap();
    let (id, hash) = put(&store, "a", UnitKind::Commit, 1);
    let cap = store.wal_max_bytes;
    store.wal_max_bytes = 1;
    assert_eq!(prepare(&store, &routes("a")).unwrap(), initial);
    assert!(prepare(&store, &routes("b")).is_err());
    assert_eq!(state(&store), initial);
    store.wal_max_bytes = cap;
    let pending = prepare(&store, &routes("b")).unwrap();
    replace(&store, "b", id, &hash);
    store.wal_max_bytes = 1;
    assert!(store.finish_transition(pending.generation).is_err());
    assert_eq!(state(&store), pending);
    assert_eq!(prepare(&store, &routes("b")).unwrap(), pending);
    store.wal_max_bytes = cap;
    assert!(
        store
            .finish_transition(pending.generation)
            .unwrap()
            .previous
            .is_empty()
    );
}
#[test]
fn upgrade_bootstraps_observed_legacy_sources_without_guessing_provider_history() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("index.db");
    let store = store::open(&path, StoreLimits::default()).unwrap();
    table(&store, "old", false);
    put(&store, "old", UnitKind::Commit, 1);
    store::migration_tests::drop_l1_delete_indexes(&store.writer().unwrap());
    store.writer().unwrap().execute_batch("DROP TABLE embedding_publications; DROP TABLE embedding_transition; UPDATE meta SET value='12' WHERE key='schema_version'").unwrap();
    drop(store);
    let store = store::open(&path, StoreLimits::default()).unwrap();
    assert!(store.read_snapshot(read).unwrap().unwrap().is_none());
    let pending = prepare(&store, &routes("new")).unwrap();
    assert_eq!(pending.generation, 1);
    assert_eq!(pending.previous.len(), 1);
    assert!(!pending.previous[0].known);
    assert_eq!(store.count_vectors().unwrap(), 1);
    let crowded = store::open(":memory:", StoreLimits::default()).unwrap();
    for (index, slug) in ["a", "b", "c"].into_iter().enumerate() {
        table(&crowded, slug, false);
        put(&crowded, slug, UnitKind::Commit, index as u8);
    }
    assert!(matches!(
        prepare(&crowded, &routes("new")),
        Err(Error::EmbeddingTransition { .. })
    ));
    assert!(crowded.read_snapshot(read).unwrap().unwrap().is_none());
    assert_eq!(crowded.count_vectors().unwrap(), 3);
}
