//! Provenance survives reopen, conflicts and migration without invented history.
use super::*;
use crate::store::{
    self, StoreLimits,
    vectors::{VectorInsert, VectorTableSpec},
};
fn binding() -> Binding {
    Binding {
        profile: "ollama".into(),
        model: "model".into(),
        dims: 2,
        endpoint: "http://localhost:11434/api/embed".into(),
        artifact_hash: None,
        query_prefix: "query: ".into(),
        api_key_env: None,
    }
}
fn table(store: &Store) {
    store
        .ensure_vector_table(VectorTableSpec {
            slug: "bound",
            model: "model",
            profile: "ollama",
            dims: 2,
            created_at: 0,
            active: true,
        })
        .unwrap();
}
fn get(store: &Store) -> Option<Binding> {
    store
        .read_snapshot(|tx| read(tx, "bound"))
        .unwrap()
        .unwrap()
}
#[test]
fn binding_survives_reopen_and_only_query_settings_can_change() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("index.db");
    let store = store::open(&path, StoreLimits::default()).unwrap();
    table(&store);
    let initial = binding();
    assert_eq!(
        store.bind_embedding_table("bound", &initial).unwrap(),
        State::Known
    );
    drop(store);
    let mut store = store::open(&path, StoreLimits::default()).unwrap();
    assert!(get(&store).as_ref() == Some(&initial));
    let mut changed = initial.clone();
    changed.endpoint = "http://localhost:11435/api/embed".into();
    assert!(matches!(
        store.bind_embedding_table("bound", &changed),
        Err(Error::EmbeddingBinding { .. })
    ));
    assert!(get(&store).as_ref() == Some(&initial));
    changed = initial.clone();
    changed.query_prefix = "new query: ".into();
    let cap = store.wal_max_bytes;
    store.wal_max_bytes = 1;
    store.bind_embedding_table("bound", &initial).unwrap();
    assert!(store.bind_embedding_table("bound", &changed).is_err());
    assert!(get(&store).as_ref() == Some(&initial));
    store.wal_max_bytes = cap;
    store.bind_embedding_table("bound", &changed).unwrap();
    assert!(get(&store).as_ref() == Some(&changed));
    // A SQL trigger proves the exact-match path performs no write.
    store.writer().unwrap().execute_batch("CREATE TRIGGER reject_binding_update BEFORE UPDATE ON embedding_bindings BEGIN SELECT RAISE(ABORT,'hold'); END").unwrap();
    store.bind_embedding_table("bound", &changed).unwrap();
    assert!(store.bind_embedding_table("bound", &initial).is_err());
    assert!(get(&store).as_ref() == Some(&changed));
    store.drop_vector_table("bound").unwrap();
    assert!(get(&store).is_none());
}
#[test]
fn populated_legacy_tables_stay_unknown_and_invalid_endpoints_are_not_persisted() {
    let store = store::open(":memory:", StoreLimits::default()).unwrap();
    table(&store);
    let mut hostile = binding();
    hostile.endpoint = "http://localhost/api/embed?invalid=fixture".into();
    assert!(store.bind_embedding_table("bound", &hostile).is_err());
    hostile.endpoint = "http://fixture@localhost/api/embed".into();
    assert!(store.bind_embedding_table("bound", &hostile).is_err());
    assert!(get(&store).is_none());
    populate(&store);
    assert_eq!(
        store.bind_embedding_table("bound", &binding()).unwrap(),
        State::LegacyUnknown
    );
    assert!(get(&store).is_none());
    assert_eq!(store.count_vectors().unwrap(), 1);
}

fn populate(store: &Store) {
    let hash = *blake3::hash(b"body").as_bytes();
    let id = store
        .insert_unit(store::UnitInsert {
            hash: &hash,
            source_key: Some("commit:fixture"),
            duplicate_ordinal: 0,
            file_id: None,
            kind: store::UnitKind::Commit,
            byte_start: None,
            byte_end: None,
            line_start: None,
            line_end: None,
            scope_path: None,
            header: "fixture",
            text: "body",
            tokens: 1,
            ladder_level: 1,
            parent_unit: None,
            created_at: 0,
            updated_at: 0,
            fts: store::FtsColumns {
                header: "fixture",
                idents: "",
                body: "body",
                prose: "",
            },
        })
        .unwrap();
    store
        .insert_vector(VectorInsert {
            table_slug: "bound",
            unit_id: id,
            ctx_hash: &hash,
            embedding_hash: &hash,
            embedding: &[1.0, 0.0],
        })
        .unwrap();
}

#[test]
fn upgrading_schema_eleven_preserves_vectors_without_fabricating_a_binding() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("index.db");
    let store = store::open(&path, StoreLimits::default()).unwrap();
    table(&store);
    populate(&store);
    // Reconstruct the previous schema with actual vector/FTS data in place.
    store::migration_tests::drop_l1_delete_indexes(&store.writer().unwrap());
    store
        .writer()
        .unwrap()
        .execute_batch(
            "DROP TABLE embedding_publications; DROP TABLE embedding_transition; DROP TABLE embedding_bindings; UPDATE meta SET value='11' WHERE key='schema_version'",
        )
        .unwrap();
    drop(store);
    let store = store::open(&path, StoreLimits::default()).unwrap();
    assert_eq!(
        store.get_meta("schema_version").unwrap(),
        Some(store::LATEST_SCHEMA_VERSION.to_string())
    );
    assert_eq!(store.count_vectors().unwrap(), 1);
    assert_eq!(store.query_fts("body", 10).unwrap().len(), 1);
    assert!(get(&store).is_none());
    assert_eq!(
        store.bind_embedding_table("bound", &binding()).unwrap(),
        State::LegacyUnknown
    );
    assert!(get(&store).is_none());
}

#[test]
fn local_artifact_and_remote_key_reference_round_trip_without_cross_profile_fields() {
    let store = store::open(":memory:", StoreLimits::default()).unwrap();
    for (profile, endpoint, artifact, key) in [
        ("local", "", Some([42; 32]), None),
        (
            "remote",
            "https://example.invalid/v1/embeddings",
            None,
            Some("FIXTURE_KEY_ENV"),
        ),
    ] {
        store
            .ensure_vector_table(VectorTableSpec {
                slug: profile,
                model: "model",
                profile,
                dims: 2,
                created_at: 0,
                active: true,
            })
            .unwrap();
        let original = Binding {
            profile: profile.into(),
            model: "model".into(),
            dims: 2,
            endpoint: endpoint.into(),
            artifact_hash: artifact,
            query_prefix: "query: ".into(),
            api_key_env: key.map(str::to_owned),
        };
        store.bind_embedding_table(profile, &original).unwrap();
        assert!(
            store
                .read_snapshot(|tx| read(tx, profile))
                .unwrap()
                .unwrap()
                .as_ref()
                == Some(&original)
        );
        let mut changed = original.clone();
        changed.artifact_hash = Some([43; 32]);
        assert!(store.bind_embedding_table(profile, &changed).is_err());
        assert!(
            store
                .read_snapshot(|tx| read(tx, profile))
                .unwrap()
                .unwrap()
                .as_ref()
                == Some(&original)
        );
    }
}

#[test]
fn destination_identity_ignores_query_settings_but_separates_vector_spaces() {
    let original = binding();
    let mut query = original.clone();
    query.query_prefix = "different query: ".into();
    assert_eq!(original.destination_slug(), query.destination_slug());
    for changed in [
        Binding {
            endpoint: "http://localhost:11435/api/embed".into(),
            ..original.clone()
        },
        Binding {
            model: "another".into(),
            ..original.clone()
        },
        Binding {
            dims: 3,
            ..original.clone()
        },
    ] {
        assert_ne!(original.destination_slug(), changed.destination_slug());
    }
    let local = Binding {
        profile: "local".into(),
        endpoint: String::new(),
        artifact_hash: Some([1; 32]),
        ..original.clone()
    };
    let changed = Binding {
        artifact_hash: Some([2; 32]),
        ..local.clone()
    };
    assert_ne!(local.destination_slug(), changed.destination_slug());
    assert!(super::super::vectors::vector_table_name(&original.destination_slug()).is_ok());
}

#[test]
fn destination_reuses_only_proven_legacy_names_and_prefers_the_full_identity() {
    let store = store::open(":memory:", StoreLimits::default()).unwrap();
    let binding = binding();
    let canonical =
        super::super::vectors::embedding_table_slug(&binding.profile, &binding.model, binding.dims);
    let destination = binding.destination_slug();
    assert_eq!(store.embedding_destination(&binding).unwrap(), destination);
    store
        .ensure_vector_table(VectorTableSpec {
            slug: &canonical,
            model: "model",
            profile: "ollama",
            dims: 2,
            created_at: 0,
            active: true,
        })
        .unwrap();
    assert_eq!(store.embedding_destination(&binding).unwrap(), destination); // Unknown is not proof.
    store.bind_embedding_table(&canonical, &binding).unwrap();
    assert_eq!(store.embedding_destination(&binding).unwrap(), canonical);
    store
        .ensure_vector_table(VectorTableSpec {
            slug: &destination,
            model: "model",
            profile: "ollama",
            dims: 2,
            created_at: 1,
            active: true,
        })
        .unwrap();
    store.bind_embedding_table(&destination, &binding).unwrap();
    assert_eq!(store.embedding_destination(&binding).unwrap(), destination);
    store.drop_vector_table(&canonical).unwrap();
    assert_eq!(store.embedding_destination(&binding).unwrap(), destination);
}
