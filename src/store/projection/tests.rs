//! Parent exclusion before lexical/name/graph/vector limits; v1 §9.
use crate::store::{
    self, FileKind, FileUpsert, FtsColumns, GraphEdgeKind, Store, StoreLimits, SymbolInsert,
    UnitEdgeInsert, UnitInsert, UnitKind,
    vectors::{VectorInsert, VectorScanPredicate, VectorTableSpec},
};

fn file(store: &Store, path: &str) -> i64 {
    store
        .upsert_file(FileUpsert {
            path,
            blake3: blake3::hash(path.as_bytes()).as_bytes(),
            size: 10000,
            mtime_ns: 1,
            lang: Some("rust"),
            kind: if path.ends_with(".png") {
                FileKind::Image
            } else {
                FileKind::Code
            },
            git_blob: None,
            max_level: 2,
            treatment: "fixture",
            indexed_at: Some(1),
            needs_vlm: false,
        })
        .unwrap()
}

fn scoped_unit(
    store: &Store,
    file: Option<i64>,
    parent: Option<i64>,
    kind: UnitKind,
    ordinal: u32,
    body: &str,
) -> (i64, [u8; 32]) {
    let hash = *blake3::hash(format!("{file:?}:{parent:?}:{kind:?}:{ordinal}:{body}").as_bytes())
        .as_bytes();
    let source_key = format!("scope:{}", blake3::hash(&hash).to_hex());
    let id = store
        .insert_unit(UnitInsert {
            hash: &hash,
            source_key: Some(&source_key),
            duplicate_ordinal: ordinal,
            file_id: file,
            kind,
            byte_start: None,
            byte_end: None,
            line_start: None,
            line_end: None,
            scope_path: None,
            header: "",
            text: body,
            tokens: 1,
            ladder_level: if kind == UnitKind::Image { 0 } else { 3 },
            parent_unit: parent,
            created_at: 1,
            updated_at: 1,
            fts: FtsColumns {
                header: "",
                idents: "",
                body,
                prose: "",
            },
        })
        .unwrap();
    (id, hash)
}

fn scoped_vector(store: &Store, table: &str, unit: (i64, [u8; 32]), embedding: &[f32]) {
    store
        .insert_vector(VectorInsert {
            table_slug: table,
            unit_id: unit.0,
            ctx_hash: &unit.1,
            embedding_hash: &unit.1,
            embedding,
        })
        .unwrap();
}

#[test]
fn summary_owner_scope_precedes_limits_without_changing_vector_route_identity() {
    let store = store::open(":memory:", StoreLimits::default()).unwrap();
    for slug in ["summary_scope", "summary_destination"] {
        store
            .ensure_vector_table(VectorTableSpec {
                slug,
                model: "fixture",
                profile: "local",
                dims: 2,
                created_at: 1,
                active: true,
            })
            .unwrap();
    }
    let outside = scoped_unit(
        &store,
        Some(file(&store, "other/outside.png")),
        None,
        UnitKind::Image,
        0,
        "",
    )
    .0;
    let parent = scoped_unit(
        &store,
        Some(file(&store, "images/target.png")),
        None,
        UnitKind::Image,
        0,
        "",
    )
    .0;
    for ordinal in 0..450 {
        let unit = scoped_unit(
            &store,
            None,
            Some(outside),
            UnitKind::Summary,
            ordinal,
            "scope_needle",
        );
        scoped_vector(&store, "summary_scope", unit, &[1.0, 0.0]);
    }
    let intermediate = scoped_unit(
        &store,
        None,
        Some(parent),
        UnitKind::Summary,
        0,
        "intermediate",
    )
    .0;
    let target = scoped_unit(
        &store,
        None,
        Some(intermediate),
        UnitKind::Summary,
        0,
        &format!("scope_needle {}", "padding ".repeat(100)),
    );
    scoped_vector(&store, "summary_scope", target, &[0.0, 1.0]);
    let globs = ["missing/*".to_owned(), "images/target.png".to_owned()];
    for scope in [
        store::SearchScope::from("images/target.png"),
        store::SearchScope::Globs(globs.to_vec()),
    ] {
        let hits = store
            .query_fts_scoped("scope_needle", 1, Some(&scope), &[])
            .unwrap();
        assert_eq!(
            hits.iter().map(|h| h.unit_id).collect::<Vec<_>>(),
            [target.0]
        );
        for excluded in [parent, intermediate, target.0] {
            assert!(
                store
                    .query_fts_scoped("scope_needle", 1, Some(&scope), &[excluded])
                    .unwrap()
                    .is_empty()
            );
        }
    }
    assert!(
        store
            .query_fts_scoped("scope_needle", 1, Some(&"code".into()), &[])
            .unwrap()
            .is_empty()
    );
    let predicate = VectorScanPredicate {
        kinds: &[UnitKind::Summary],
        scope_kinds: &[UnitKind::Image],
        path_globs: Some(&globs),
        ..Default::default()
    };
    let scan = |predicate: VectorScanPredicate<'_>| {
        store
            .scan_vectors("summary_scope", &[1.0, 0.0], 1, predicate)
            .unwrap()
    };
    assert_eq!(
        scan(predicate)
            .iter()
            .map(|h| h.unit_id)
            .collect::<Vec<_>>(),
        [target.0]
    );
    for excluded in [parent, intermediate, target.0] {
        assert!(
            scan(VectorScanPredicate {
                exclude_ids: &[excluded],
                ..predicate
            })
            .is_empty()
        );
    }
    assert!(
        scan(VectorScanPredicate {
            kinds: &[UnitKind::Image],
            ..predicate
        })
        .is_empty(),
        "stored routing kind remains summary"
    );
    assert!(
        scan(VectorScanPredicate {
            scope_kinds: &[UnitKind::Symbol],
            ..predicate
        })
        .is_empty(),
        "image summaries cannot satisfy code scope"
    );
    scoped_vector(&store, "summary_destination", target, &[0.0, 1.0]);
    assert!(
        scan(VectorScanPredicate {
            replaced_by: &[(UnitKind::Summary, "summary_destination".into())],
            ..predicate
        })
        .is_empty()
    );
    assert_eq!(
        scan(VectorScanPredicate {
            replaced_by: &[(UnitKind::Image, "summary_destination".into())],
            ..predicate
        })
        .len(),
        1,
        "migration checks actual stored kind, not inherited image scope"
    );
}

#[test]
fn scoped_summary_cycles_and_missing_owners_cannot_claim_a_file_path() {
    let store = store::open(":memory:", StoreLimits::default()).unwrap();
    store
        .ensure_vector_table(VectorTableSpec {
            slug: "broken_summary_scope",
            model: "fixture",
            profile: "local",
            dims: 2,
            created_at: 1,
            active: true,
        })
        .unwrap();
    let image = file(&store, "images/target.png");
    let first = scoped_unit(
        &store,
        Some(image),
        None,
        UnitKind::Summary,
        0,
        "scope_needle",
    );
    let second = scoped_unit(
        &store,
        None,
        Some(first.0),
        UnitKind::Summary,
        0,
        "scope_needle",
    );
    store
        .writer()
        .unwrap()
        .execute(
            "UPDATE units SET parent_unit=?2 WHERE id=?1",
            [first.0, second.0],
        )
        .unwrap();
    let missing = scoped_unit(
        &store,
        Some(image),
        None,
        UnitKind::Summary,
        1,
        "scope_needle",
    );
    for unit in [first, second, missing] {
        scoped_vector(&store, "broken_summary_scope", unit, &[1.0, 0.0]);
    }
    assert_eq!(
        store
            .query_fts_scoped("scope_needle", 10, None, &[])
            .unwrap()
            .len(),
        3
    );
    for scope in ["images/target.png", "images"] {
        assert!(
            store
                .query_fts_scoped("scope_needle", 10, Some(&scope.into()), &[])
                .unwrap()
                .is_empty()
        );
    }
    assert!(
        store
            .scan_vectors(
                "broken_summary_scope",
                &[1.0, 0.0],
                10,
                VectorScanPredicate {
                    scope_kinds: &[UnitKind::Image],
                    path_globs: Some(&["images/target.png".into()]),
                    ..Default::default()
                }
            )
            .unwrap()
            .is_empty()
    );
}
fn unit(
    store: &Store,
    file: i64,
    ordinal: u32,
    kind: UnitKind,
    range: (i64, i64),
    name: &str,
    body: &str,
) -> (i64, [u8; 32]) {
    let hash = *blake3::hash(format!("{}:{name}:{body}", kind.as_str()).as_bytes()).as_bytes();
    let edge = UnitEdgeInsert {
        src_symbol_index: 0,
        dst_name: "should_retry",
        dst_symbol: None,
        dst_symbol_index: None,
        kind: GraphEdgeKind::Ref,
        line: Some(1),
    };
    let id = store
        .insert_indexed_unit(
            UnitInsert {
                hash: &hash,
                source_key: None,
                duplicate_ordinal: ordinal,
                file_id: Some(file),
                kind,
                byte_start: Some(range.0),
                byte_end: Some(range.1),
                line_start: Some(1),
                line_end: Some(1),
                scope_path: None,
                header: name,
                text: body,
                tokens: 1,
                ladder_level: 1,
                parent_unit: None,
                created_at: 1,
                updated_at: 1,
                fts: FtsColumns {
                    header: "",
                    idents: name,
                    body,
                    prose: "",
                },
            },
            &[SymbolInsert {
                unit_id: None,
                name,
                qualified: Some(name),
                kind: "function",
                signature: None,
                line_start: 1,
                line_end: 1,
            }],
            if name == "seed" {
                std::slice::from_ref(&edge)
            } else {
                &[]
            },
        )
        .unwrap()
        .unit_id;
    (id, hash)
}

#[test]
fn excluded_parent_cannot_fill_any_retrieval_window_with_its_children() {
    let store = store::open(":memory:", StoreLimits::default()).unwrap();
    let blocked = file(&store, "src/blocked.rs");
    let (parent, _) = unit(
        &store,
        blocked,
        0,
        UnitKind::Symbol,
        (0, 10000),
        "container",
        "parent",
    );
    store
        .ensure_vector_table(VectorTableSpec {
            slug: "projection",
            model: "fixture",
            profile: "local",
            dims: 2,
            created_at: 1,
            active: true,
        })
        .unwrap();
    for i in 0..450 {
        let (id, hash) = unit(
            &store,
            blocked,
            i as u32,
            UnitKind::Chunk,
            (i * 10, i * 10 + 5),
            "should_retry",
            "should_retry",
        );
        store
            .insert_vector(VectorInsert {
                table_slug: "projection",
                unit_id: id,
                ctx_hash: &hash,
                embedding_hash: &hash,
                embedding: &[1.0, 0.0],
            })
            .unwrap();
    }
    let (target, hash) = unit(
        &store,
        file(&store, "src/eligible.rs"),
        0,
        UnitKind::Symbol,
        (0, 1000),
        "should_retry",
        &format!("should_retry {}", "padding ".repeat(100)),
    );
    store
        .insert_vector(VectorInsert {
            table_slug: "projection",
            unit_id: target,
            ctx_hash: &hash,
            embedding_hash: &hash,
            embedding: &[0.0, 1.0],
        })
        .unwrap();
    let (seed, _) = unit(
        &store,
        file(&store, "src/seed.rs"),
        0,
        UnitKind::Symbol,
        (0, 100),
        "seed",
        "seed",
    );
    store
        .writer()
        .unwrap()
        .execute(
            "UPDATE symbols SET pagerank=CASE WHEN unit_id=?1 THEN 0.0 ELSE 1.0 END",
            [target],
        )
        .unwrap();
    assert_ne!(
        store
            .query_fts_scoped("should_retry", 1, None, &[])
            .unwrap()[0]
            .unit_id,
        target
    );
    assert_ne!(
        store
            .scan_vectors("projection", &[1.0, 0.0], 1, VectorScanPredicate::default())
            .unwrap()[0]
            .unit_id,
        target
    );
    assert_ne!(
        store.graph_neighbours(&[seed], 1, None, &[]).unwrap()[0].unit_id,
        target
    );
    assert_eq!(
        store
            .query_fts_scoped("should_retry", 1, None, &[parent])
            .unwrap()
            .iter()
            .map(|hit| hit.unit_id)
            .collect::<Vec<_>>(),
        [target]
    );
    assert_eq!(
        store
            .exact_name_units("should_retry", 1, None, &[parent])
            .unwrap(),
        [target]
    );
    assert_eq!(
        store
            .graph_neighbours(&[seed], 1, None, &[parent])
            .unwrap()
            .iter()
            .map(|hit| hit.unit_id)
            .collect::<Vec<_>>(),
        [target]
    );
    assert_eq!(
        store
            .scan_vectors(
                "projection",
                &[1.0, 0.0],
                1,
                VectorScanPredicate {
                    exclude_ids: &[parent],
                    ..VectorScanPredicate::default()
                }
            )
            .unwrap()
            .iter()
            .map(|hit| hit.unit_id)
            .collect::<Vec<_>>(),
        [target]
    );
}
