//! Regression fixtures for the shared lexical pipeline (`v1.md` §9).
use super::*;
use crate::store::{
    FileKind, FileUpsert, FtsColumns, GraphEdgeKind, StoreLimits, SymbolInsert, UnitEdgeInsert,
    UnitInsert, UnitKind,
};
use std::error::Error;

pub(super) fn add(
    store: &Store,
    path: &str,
    name: &str,
    body: &str,
    edge: Option<&str>,
) -> Result<i64, Box<dyn Error>> {
    let hash = blake3::hash(path.as_bytes());
    let file = store.upsert_file(FileUpsert {
        path,
        blake3: hash.as_bytes(),
        size: body.len() as i64,
        mtime_ns: 1,
        lang: Some("rust"),
        kind: FileKind::Code,
        git_blob: None,
        max_level: 1,
        treatment: "source",
        indexed_at: Some(1),
        needs_vlm: false,
    })?;
    let edges = edge
        .map(|name| UnitEdgeInsert {
            src_symbol_index: 0,
            dst_name: name,
            dst_symbol: None,
            dst_symbol_index: None,
            kind: GraphEdgeKind::Ref,
            line: Some(1),
        })
        .into_iter()
        .collect::<Vec<_>>();
    Ok(store
        .insert_indexed_unit(
            UnitInsert {
                hash: hash.as_bytes(),
                source_key: None,
                duplicate_ordinal: 0,
                file_id: Some(file),
                kind: UnitKind::Symbol,
                byte_start: Some(0),
                byte_end: Some(body.len() as i64),
                line_start: Some(1),
                line_end: Some(1),
                scope_path: None,
                header: name,
                text: body,
                tokens: 10,
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
            &edges,
        )?
        .unit_id)
}

#[test]
fn relaxed_queries_preserve_phrases_and_promote_exact_names() -> Result<(), Box<dyn Error>> {
    let db = tempfile::NamedTempFile::new()?;
    let store = store::open(db.path(), StoreLimits::default())?;
    let distractor = add(
        &store,
        "src/a.rs",
        "should_retry_extra",
        "should_retry should_retry should_retry",
        None,
    )?;
    let target = add(
        &store,
        "src/b.rs",
        "should_retry",
        "network failure retry policy",
        None,
    )?;
    let settings = Search {
        graph_prior_weight: 0.0,
        ..Search::default()
    };
    let hits = lexical(
        &store,
        "where is the network failure retried",
        10,
        None,
        &settings,
    )?;
    assert!(hits.iter().any(|hit| hit.unit.id == target));
    let hits = lexical(&store, "should_retry", 10, None, &settings)?;
    assert_eq!(hits[0].unit.id, target);
    assert!(hits[0].exact_name);
    assert!(hits.iter().any(|hit| hit.unit.id == distractor));
    assert!(lexical(&store, "\"network retry\" absent", 10, None, &settings)?.is_empty());
    assert!(!lexical(&store, "\"network failure\" absent", 10, None, &settings)?.is_empty());
    assert!(lexical(&store, "!?", 10, None, &settings)?.is_empty());
    Ok(())
}

#[test]
fn relaxed_questions_do_not_rank_grammar_only_matches_above_requested_content()
-> Result<(), Box<dyn Error>> {
    let db = tempfile::NamedTempFile::new()?;
    let store = store::open(db.path(), StoreLimits::default())?;
    let noise = add(&store, "src/noise.rs", "noise", "where is the", None)?;
    let target = add(
        &store,
        "src/connection.rs",
        "connection",
        &format!("connection timeout {}", "padding ".repeat(100)),
        None,
    )?;
    let text = "where is the connection timeout";
    assert!(store.query_fts(&build_match(text), 10)?.is_empty());
    // The old OR query lets short, grammar-only content beat the requested code.
    let baseline = store.query_fts(&crate::treat::fts::build_or_match(text), 10)?;
    assert_eq!(baseline.first().map(|hit| hit.unit_id), Some(noise));
    let settings = Search {
        graph_prior_weight: 0.0,
        ..Search::default()
    };
    let hits = lexical(&store, text, 1, None, &settings)?;
    assert_eq!(hits.first().map(|hit| hit.unit.id), Some(target));
    assert!(hits.iter().all(|hit| hit.unit.id != noise));

    // An adequate strict result still wins before any relaxed query is built.
    let strict = add(&store, "src/strict.rs", "strict", text, None)?;
    let hits = lexical(&store, text, 1, None, &settings)?;
    assert_eq!(hits.first().map(|hit| hit.unit.id), Some(strict));

    // Explicit syntax retains the previous OR order, even when it contains
    // grammar words; single-word and all-grammar requests also stay meaningful.
    for explicit in [
        "lex:where is the connection timeout",
        "where is the \"connection timeout\"",
        "where",
        "where is the",
    ] {
        let parsed = query::parse(explicit, fusion::Mode::Lexical)?;
        let expected = store
            .query_fts(
                &crate::treat::fts::build_or_match(parsed.lexical.as_deref().unwrap_or_default()),
                10,
            )?
            .into_iter()
            .map(|hit| hit.unit_id)
            .collect::<Vec<_>>();
        let actual = lexical(&store, explicit, 10, None, &settings)?
            .into_iter()
            .map(|hit| hit.unit.id)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "{explicit}");
    }
    Ok(())
}

#[test]
fn relaxed_questions_keep_identifier_references_without_an_exact_name_match()
-> Result<(), Box<dyn Error>> {
    let db = tempfile::NamedTempFile::new()?;
    let store = store::open(db.path(), StoreLimits::default())?;
    add(&store, "src/noise.rs", "noise", "configured", None)?;
    let target = add(&store, "src/input.rs", "load_input", "CAN terminal", None)?;
    assert!(store.exact_name_units("CAN", 10, None, &[])?.is_empty());
    let settings = Search {
        graph_prior_weight: 0.0,
        ..Search::default()
    };
    let hits = lexical(&store, "Where is CAN configured", 10, None, &settings)?;
    assert!(hits.iter().any(|hit| hit.unit.id == target));
    Ok(())
}

#[test]
fn graph_seeds_follow_lexical_rank_and_configured_weight() -> Result<(), Box<dyn Error>> {
    let db = tempfile::NamedTempFile::new()?;
    let store = store::open(db.path(), StoreLimits::default())?;
    let excluded = add(&store, "src/excluded.rs", "excluded", "unrelated", None)?;
    let included = add(&store, "src/included.rs", "included", "unrelated", None)?;
    // Low rowids fill the old BTreeMap-derived seed set, but have poor BM25 ranks.
    for i in 0..20 {
        add(
            &store,
            &format!("src/noisy{i}.rs"),
            &format!("noisy{i}"),
            &format!("needle {}", "padding ".repeat(100)),
            Some("excluded"),
        )?;
    }
    for i in 0..20 {
        add(
            &store,
            &format!("src/strong{i}.rs"),
            &format!("strong{i}"),
            "needle",
            Some("included"),
        )?;
    }
    let settings = Search {
        max_k: 100,
        ..Search::default()
    };
    let hits = lexical(&store, "needle", 100, None, &settings)?;
    assert!(hits.iter().any(|hit| hit.unit.id == included && hit.graph));
    assert!(!hits.iter().any(|hit| hit.unit.id == excluded));
    let score = hits
        .iter()
        .find(|hit| hit.unit.id == included)
        .ok_or("missing neighbour")?
        .score;
    assert!((score - 0.3 / 61.0).abs() < 1e-12);
    let settings = Search {
        graph_prior_weight: 0.0,
        ..settings
    };
    let hits = lexical(&store, "needle", 100, None, &settings)?;
    assert!(hits.iter().all(|hit| !hit.graph));
    assert!(!hits.iter().any(|hit| hit.unit.id == included));
    let settings = Search {
        graph_prior_weight: 0.6,
        ..settings
    };
    let hits = lexical(&store, "needle", 100, None, &settings)?;
    assert!(
        (hits
            .iter()
            .find(|hit| hit.unit.id == included)
            .ok_or("missing neighbour")?
            .score
            - 2.0 * score)
            .abs()
            < 1e-12
    );
    Ok(())
}

#[test]
fn scopes_and_exact_names_filter_before_candidate_limits() -> Result<(), Box<dyn Error>> {
    let db = tempfile::NamedTempFile::new()?;
    let store = store::open(db.path(), StoreLimits::default())?;
    for i in 0..205 {
        add(
            &store,
            &format!("outside/{i}.rs"),
            "TargetNameExtra",
            "needle needle",
            None,
        )?;
    }
    let target = add(
        &store,
        "inside/target.rs",
        "TargetName",
        "needle padding",
        None,
    )?;
    let settings = Search::default();
    let hits = lexical(&store, "needle", 1, Some("inside/*"), &settings)?;
    assert_eq!(hits[0].unit.id, target);
    let hits = lexical(&store, "TargetName", 1, None, &settings)?;
    assert_eq!(hits[0].unit.id, target);
    assert!(lexical(&store, "needle", 10, Some("docs"), &settings)?.is_empty());
    assert!(
        lexical(&store, "TargetName", 10, Some("outside/*"), &settings)?
            .iter()
            .all(|hit| hit.unit.id != target)
    );
    Ok(())
}

#[test]
fn open_range_inside_a_symbol_does_not_require_an_overlapping_chunk() -> Result<(), Box<dyn Error>>
{
    let store = store::open(":memory:", StoreLimits::default())?;
    let id = add(&store, "source.rs", "outer", "first chunk", None)?;
    store
        .writer()?
        .execute("UPDATE symbols SET line_end=10", [])?;
    assert!(store.unit_by_path("source.rs", Some(5), Some(5))?.is_none());
    let opened = navigation::open(&store, "source.rs:5-5", 100)?;
    assert_eq!(
        opened.id,
        navigation::public_id(&store.unit_by_id(id)?.ok_or("missing unit")?)
    );
    for target in [
        "source.rs:0-1",
        "source.rs:5-2",
        "source.rs:1-18446744073709551615",
    ] {
        assert!(matches!(
            navigation::open(&store, target, 100),
            Err(super::Error::InvalidQuery { .. })
        ));
    }
    Ok(())
}

#[test]
fn shared_navigation_roundtrips_ids_and_bounds_unicode_text() -> Result<(), Box<dyn Error>> {
    let db = tempfile::NamedTempFile::new()?;
    let store = store::open(db.path(), StoreLimits::default())?;
    let target = add(&store, "src/target.rs", "TargetName", "aé🙂z", None)?;
    add(
        &store,
        "src/caller.rs",
        "CallerName",
        "call TargetName",
        Some("TargetName"),
    )?;
    let symbols = navigation::symbol(&store, "TargetName")?;
    assert_eq!(symbols.len(), 1);
    assert_eq!(symbols[0].callers, 1);
    let opened = navigation::open(&store, &symbols[0].id, 100)?;
    assert_eq!(opened.path.as_deref(), Some("src/target.rs"));
    let occurrence = store.unit_by_id(target)?.ok_or("missing target")?;
    assert_eq!(opened.id, navigation::public_id(&occurrence));
    for cap in 0..=9 {
        let text = navigation::open(&store, &opened.id, cap)?.text;
        assert!(text.len() <= cap);
        assert!("aé🙂z".starts_with(&text));
    }
    assert_eq!(
        navigation::open(&store, "src/target.rs:1-1", 100)?.id,
        opened.id
    );
    let refs = navigation::refs(&store, &symbols[0].id, Some(1), false)?;
    assert_eq!(refs.total, 1);
    assert_eq!(refs.groups[0].path, "src/caller.rs");
    assert!(
        navigation::refs(&store, &symbols[0].id, None, true)?
            .groups
            .is_empty()
    );
    assert!(matches!(
        navigation::refs(&store, "TargetName", Some(0), false),
        Err(super::Error::InvalidReferenceLimit)
    ));
    assert!(matches!(
        navigation::open(&store, "missing.rs", 100),
        Err(super::Error::NotFound { .. })
    ));
    Ok(())
}

#[test]
fn verbose_rrf_details_preserve_compact_response_and_order() -> Result<(), Box<dyn Error>> {
    let store = store::open(":memory:", StoreLimits::default())?;
    add(&store, "first.rs", "FirstTarget", "needle", None)?;
    add(&store, "second.rs", "SecondTarget", "needle", None)?;
    let hits = lexical(&store, "needle", 2, None, &Search::default())?;
    let compact = format_response(&store, "needle", &hits)?;
    let mut verbose = format_response_verbose(&store, "needle", &hits, true)?;
    assert_eq!(
        compact,
        format_response_verbose(&store, "needle", &hits, false)?
    );
    for (position, hit) in verbose["hits"]
        .as_array_mut()
        .ok_or("missing hits")?
        .iter_mut()
        .enumerate()
    {
        assert_eq!(hit["rrf"]["lexical"]["rank"], position + 1);
        assert!(hit["rrf"]["dense"].is_null());
        assert!(hit["rrf"]["graph"].is_null());
        assert_eq!(hit["score"], hit["rrf"]["lexical"]["score"]);
        hit.as_object_mut().ok_or("hit object")?.remove("rrf");
    }
    assert_eq!(compact, verbose);
    Ok(())
}

#[test]
fn typed_lexical_phrases_do_not_turn_prefixes_or_identifiers_into_extra_hits()
-> Result<(), Box<dyn Error>> {
    let store = store::open(":memory:", StoreLimits::default())?;
    let wanted = add(&store, "wanted.rs", "RetryPolicy", "charge retry", None)?;
    add(&store, "other.rs", "OtherName", "retry then charge", None)?;
    let hits = lexical(&store, "lex:\"charge retry\"", 10, None, &Search::default())?;
    assert_eq!(
        hits.iter().map(|hit| hit.unit.id).collect::<Vec<_>>(),
        vec![wanted]
    );
    assert!(
        lexical(
            &store,
            "lex:\"RetryPolicy missing phrase\"",
            10,
            None,
            &Search::default()
        )?
        .is_empty()
    );
    assert_eq!(
        lexical(&store, "lex:RetryPolicy", 10, None, &Search::default())?[0]
            .unit
            .id,
        wanted
    );
    assert!(matches!(
        lexical(&store, "vec:charge retry", 10, None, &Search::default()),
        Err(super::Error::DenseUnavailable)
    ));
    Ok(())
}

#[test]
fn glob_union_filters_all_legs_before_limits_and_deduplicates() -> Result<(), Box<dyn Error>> {
    let db = tempfile::NamedTempFile::new()?;
    let store = store::open(db.path(), StoreLimits::default())?;
    for i in 0..205 {
        add(
            &store,
            &format!("outside/{i}.rs"),
            "TargetName",
            "needle",
            None,
        )?;
    }
    let left = add(
        &store,
        "left/a.rs",
        "TargetName",
        "needle padding",
        Some("Neighbour"),
    )?;
    let right = add(&store, "right/b.rs", "TargetName", "needle padding", None)?;
    let outside = add(&store, "outside/n.rs", "Neighbour", "other", None)?;
    let neighbour = add(&store, "right/n.rs", "Neighbour", "other", None)?;
    let scope =
        store::SearchScope::Globs(vec!["left/*".into(), "right/*".into(), "left/a.rs".into()]);
    let ids = store
        .query_fts_scoped("needle", 2, Some(&scope), &[])?
        .into_iter()
        .map(|h| h.unit_id)
        .collect::<Vec<_>>();
    assert_eq!(ids, [left, right]);
    assert_eq!(
        store.exact_name_units("TargetName", 2, Some(&scope), &[])?,
        [left, right]
    );
    let neighbours = store.graph_neighbours(&[left], 1, Some(&scope), &[])?;
    assert_eq!(
        neighbours.iter().map(|h| h.unit_id).collect::<Vec<_>>(),
        [neighbour]
    );
    assert!(outside < neighbour);
    assert!(
        store
            .graph_neighbours(&[left], 1, Some(&scope), &[neighbour])?
            .is_empty()
    );
    let parsed = query::parse("TargetName", fusion::Mode::Lexical)?;
    let hits = lexical_with_exclusions(
        &store,
        &parsed,
        10,
        Some(&scope),
        &Search::default(),
        &[left],
    )?;
    assert_eq!(hits.iter().map(|h| h.unit.id).collect::<Vec<_>>(), [right]);
    let empty = store::SearchScope::Globs(Vec::new());
    assert!(
        store
            .query_fts_scoped("needle", 10, Some(&empty), &[])?
            .is_empty()
    );
    assert!(
        store
            .exact_name_units("TargetName", 10, Some(&empty), &[])?
            .is_empty()
    );
    let literal = add(&store, "docs", "LiteralName", "needle", None)?;
    let literal_scope = store::SearchScope::Globs(vec!["docs".into()]);
    assert_eq!(
        store.query_fts_scoped("needle", 10, Some(&literal_scope), &[])?[0].unit_id,
        literal
    );
    assert!(
        store
            .query_fts_scoped("needle", 10, Some(&"docs".into()), &[])?
            .is_empty()
    );
    Ok(())
}

#[test]
fn symbol_checked_preserves_matching_counts_and_cancels_every_boundary()
-> Result<(), Box<dyn Error>> {
    use std::cell::Cell;
    let store = store::open(":memory:", StoreLimits::default())?;
    add(
        &store,
        "target.rs",
        "TargetName",
        "target",
        Some("TargetName"),
    )?;
    add(&store, "partial.rs", "OtherTargetName", "partial", None)?;
    add(&store, "caller.rs", "Caller", "caller", Some("TargetName"))?;
    add(&store, "short.rs", "xy", "short", None)?;
    let exact = navigation::symbol(&store, "TargetName")?;
    assert_eq!(exact.len(), 2);
    assert_eq!(exact[0].path, "target.rs");
    assert_eq!((exact[0].callers, exact[0].callees), (2, 1));
    let before = store.writer()?.total_changes();
    for (query, count) in [
        ("TargetName", 2),
        ("targetname", 2),
        ("xy", 1),
        (exact[0].id.as_str(), 1),
        ("missing", 0),
    ] {
        let calls = Cell::new(0);
        let expected = navigation::symbol_checked(&store, query, &|| {
            calls.set(calls.get() + 1);
            Ok(())
        })?;
        assert_eq!(expected.len(), count);
        for stop in 1..=calls.get() {
            let seen = Cell::new(0);
            let result = navigation::symbol_checked(&store, query, &|| {
                seen.set(seen.get() + 1);
                if seen.get() >= stop {
                    Err(store::Error::L1InputsChanged)
                } else {
                    Ok(())
                }
            });
            assert!(
                matches!(
                    result,
                    Err(super::Error::Store(store::Error::L1InputsChanged))
                ),
                "query={query}, boundary={stop}"
            );
            assert!(seen.get() >= stop);
        }
        assert_eq!(
            serde_json::to_value(navigation::symbol(&store, query)?)?,
            serde_json::to_value(expected)?
        );
    }
    assert_eq!(store.writer()?.total_changes(), before);
    Ok(())
}

#[test]
fn symbol_checked_readers_expire_cancel_and_retry() -> Result<(), Box<dyn Error>> {
    use std::{
        sync::atomic::{AtomicBool, Ordering},
        time::{Duration, Instant},
    };
    let store = store::open(":memory:", StoreLimits::default())?;
    add(
        &store,
        "target.rs",
        "TargetName",
        "target",
        Some("TargetName"),
    )?;
    for operation in 0..5 {
        let held = store.writer()?;
        let deadline = Instant::now() + Duration::from_millis(15);
        let check = || {
            if Instant::now() >= deadline {
                Err(store::Error::L1InputsChanged)
            } else {
                Ok(())
            }
        };
        let result = match operation {
            0 => navigation::symbol_checked(&store, "TargetName", &check).map(|_| ()),
            1 => store
                .query_fts_names_checked("Target", 100, &check)
                .map(|_| ())
                .map_err(Into::into),
            _ => store
                .graph_snapshot_checked(&check)
                .map(|_| ())
                .map_err(Into::into),
        };
        assert!(matches!(
            result,
            Err(super::Error::Store(store::Error::L1InputsChanged))
        ));
        assert!(Instant::now() < deadline + Duration::from_secs(1));
        drop(held);
    }
    let cancelled = AtomicBool::new(false);
    let held = store.writer()?;
    std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(15));
            cancelled.store(true, Ordering::Release);
        });
        assert!(
            navigation::symbol_checked(&store, "TargetName", &|| {
                if cancelled.load(Ordering::Acquire) {
                    Err(store::Error::L1InputsChanged)
                } else {
                    Ok(())
                }
            })
            .is_err()
        );
    });
    drop(held);
    assert_eq!(navigation::symbol(&store, "TargetName")?.len(), 1);
    assert_eq!(store.graph_snapshot()?.edges.len(), 1);
    assert!(!store.query_fts_names("Target", 100)?.is_empty());
    Ok(())
}

#[test]
fn refs_checked_preserves_groups_filters_counts_and_cancels_every_boundary()
-> Result<(), Box<dyn Error>> {
    use std::cell::Cell;
    let store = store::open(":memory:", StoreLimits::default())?;
    add(&store, "target.rs", "TargetName", "definition", None)?;
    add(
        &store,
        "caller.rs",
        "Caller",
        "TargetName",
        Some("TargetName"),
    )?;
    add(&store, "lexical.rs", "Lexical", "TargetName", None)?;
    add(&store, "test.rs", "Test", "TargetName", Some("TargetName"))?;
    store.writer()?.execute(
        "UPDATE edges SET kind='test_of' WHERE file_id=(SELECT id FROM files WHERE path='test.rs')",
        [],
    )?;
    // The helper indexes only the definition name in idents. Model a lexical
    // mention without introducing a graph edge or another target definition.
    store.writer()?.execute_batch(
        "DELETE FROM fts_units WHERE rowid=(SELECT unit_id FROM symbols WHERE name='Lexical');
         INSERT INTO fts_units(rowid,header,idents,body,prose)
         SELECT unit_id,'','TargetName','TargetName','' FROM symbols WHERE name='Lexical';",
    )?;
    let id = navigation::symbol(&store, "TargetName")?[0].id.clone();
    let before = store.writer()?.total_changes();
    for (query, k, tests_only, total, paths) in [
        (
            "TargetName",
            None,
            false,
            3,
            vec!["caller.rs", "lexical.rs", "test.rs"],
        ),
        ("TargetName", Some(1), false, 3, vec!["caller.rs"]),
        (id.as_str(), None, true, 1, vec!["test.rs"]),
        ("missing", None, false, 0, vec![]),
    ] {
        let calls = Cell::new(0);
        let expected = navigation::refs_checked(&store, query, k, tests_only, &|| {
            calls.set(calls.get() + 1);
            Ok(())
        })?;
        assert_eq!(expected.total, total);
        assert_eq!(
            expected
                .groups
                .iter()
                .map(|g| g.path.as_str())
                .collect::<Vec<_>>(),
            paths
        );
        assert!(expected.groups.iter().all(|g| g.lines == [1].into()));
        if tests_only {
            assert_eq!(expected.groups[0].kind, "test_of");
        }
        for stop in 1..=calls.get() {
            let seen = Cell::new(0);
            assert!(
                matches!(
                    navigation::refs_checked(&store, query, k, tests_only, &|| {
                        seen.set(seen.get() + 1);
                        if seen.get() >= stop {
                            Err(store::Error::L1InputsChanged)
                        } else {
                            Ok(())
                        }
                    }),
                    Err(super::Error::Store(store::Error::L1InputsChanged))
                ),
                "query={query}, boundary={stop}"
            );
        }
        assert_eq!(
            serde_json::to_value(navigation::refs(&store, query, k, tests_only)?)?,
            serde_json::to_value(expected)?
        );
    }
    for query in ["TargetName", "missing", ""] {
        assert_eq!(
            format!("{:?}", store.query_fts(query, 200)?),
            format!("{:?}", store.query_fts_checked(query, 200, &|| Ok(()))?)
        );
    }
    assert_eq!(store.writer()?.total_changes(), before);
    Ok(())
}

#[test]
fn refs_request_cache_reselects_after_local_and_external_edge_writes() -> Result<(), Box<dyn Error>>
{
    let database = tempfile::NamedTempFile::new()?;
    let store = store::open(database.path(), StoreLimits::default())?;
    add(&store, "target.rs", "TargetName", "definition", None)?;
    add(
        &store,
        "caller.rs",
        "Caller",
        "TargetName",
        Some("TargetName"),
    )?;
    let check = || Ok(());
    let mut cache = None;
    let initial =
        navigation::refs_checked_reuse(&store, &mut cache, "TargetName", None, true, &check)?;
    assert!(initial.groups.is_empty());
    assert_eq!(
        navigation::refs_checked_reuse(&store, &mut cache, "TargetName", None, true, &check)?,
        initial
    );

    store
        .writer()?
        .execute("UPDATE edges SET kind='test_of'", [])?;
    let local =
        navigation::refs_checked_reuse(&store, &mut cache, "TargetName", None, true, &check)?;
    assert_eq!(local.groups.len(), 1);
    assert_eq!(
        local,
        navigation::refs_checked(&store, "TargetName", None, true, &check)?
    );

    let external = rusqlite::Connection::open(database.path())?;
    external.execute("UPDATE edges SET kind='import'", [])?;
    let changed =
        navigation::refs_checked_reuse(&store, &mut cache, "TargetName", None, true, &check)?;
    assert!(changed.groups.is_empty());
    assert_eq!(
        changed,
        navigation::refs_checked(&store, "TargetName", None, true, &check)?
    );

    for (target, k, tests_only) in [
        ("TargetName", Some(1), true),
        ("TargetName", None, false),
        ("missing", None, true),
    ] {
        assert_eq!(
            navigation::refs_checked_reuse(&store, &mut cache, target, k, tests_only, &check,)?,
            navigation::refs_checked(&store, target, k, tests_only, &check)?
        );
    }
    Ok(())
}

#[test]
fn refs_checked_readers_expire_cancel_and_retry() -> Result<(), Box<dyn Error>> {
    use std::{
        sync::atomic::{AtomicBool, Ordering},
        time::{Duration, Instant},
    };
    let store = store::open(":memory:", StoreLimits::default())?;
    add(
        &store,
        "caller.rs",
        "Caller",
        "TargetName",
        Some("TargetName"),
    )?;
    for operation in 0..2 {
        for cancel in [false, true] {
            let held = store.writer()?;
            let started = Instant::now();
            let cancelled = AtomicBool::new(false);
            std::thread::scope(|scope| {
                if cancel {
                    scope.spawn(|| {
                        std::thread::sleep(Duration::from_millis(15));
                        cancelled.store(true, Ordering::Release);
                    });
                }
                let check = || {
                    if cancelled.load(Ordering::Acquire)
                        || (!cancel && started.elapsed() >= Duration::from_millis(15))
                    {
                        Err(store::Error::L1InputsChanged)
                    } else {
                        Ok(())
                    }
                };
                let result = if operation == 0 {
                    navigation::refs_checked(&store, "TargetName", None, false, &check).map(|_| ())
                } else {
                    store
                        .query_fts_checked("TargetName", 200, &check)
                        .map(|_| ())
                        .map_err(Into::into)
                };
                assert!(matches!(
                    result,
                    Err(super::Error::Store(store::Error::L1InputsChanged))
                ));
            });
            assert!(started.elapsed() < Duration::from_secs(1));
            drop(held);
            assert_eq!(
                navigation::refs(&store, "TargetName", None, false)?.total,
                1
            );
            assert_eq!(
                store
                    .query_fts_checked("TargetName", 200, &|| Ok(()))?
                    .len(),
                1
            );
        }
    }
    Ok(())
}

#[test]
fn retained_selection_checks_each_stage_and_retries_without_writes() -> Result<(), Box<dyn Error>> {
    let db = tempfile::NamedTempFile::new()?;
    let store = store::open(db.path(), StoreLimits::default())?;
    add(&store, "retry.rs", "should_retry", "retry payment", None)?;
    add(
        &store,
        "caller.rs",
        "caller",
        "unrelated",
        Some("should_retry"),
    )?;
    let parsed = query::parse("should_retry", fusion::Mode::Lexical)?;
    let settings = Search::default();
    let calls = std::cell::Cell::new(0);
    let (_, original) = lexical_with_selection(&store, &parsed, 1, None, &settings, &[], &|| {
        calls.set(calls.get() + 1);
        Ok(())
    })?;
    let writes = store.writer()?.total_changes();
    assert!(calls.get() >= 8);
    for stop in 1..=calls.get() {
        let seen = std::cell::Cell::new(0);
        let result = lexical_with_selection(&store, &parsed, 1, None, &settings, &[], &|| {
            seen.set(seen.get() + 1);
            if seen.get() >= stop {
                Err(super::Error::Deadline)
            } else {
                Ok(())
            }
        });
        assert!(
            matches!(result, Err(super::Error::Deadline)),
            "stage {stop}"
        );
        assert_eq!(seen.get(), stop);
        let (_, retry) =
            lexical_with_selection(&store, &parsed, 1, None, &settings, &[], &|| Ok(()))?;
        assert!(retry == original);
    }
    assert_eq!(store.writer()?.total_changes(), writes);
    Ok(())
}

#[test]
fn scoped_lexical_readers_preserve_filters_and_refuse_every_boundary() -> Result<(), Box<dyn Error>>
{
    use std::cell::Cell;
    let store = store::open(":memory:", StoreLimits::default())?;
    for i in 0..5 {
        add(
            &store,
            &format!("outside/{i}.rs"),
            "TargetName",
            "needle",
            None,
        )?;
    }
    let excluded = add(&store, "inside/excluded.rs", "TargetName", "needle", None)?;
    let first = add(&store, "inside/a.rs", "TargetName", "needle", None)?;
    let second = add(&store, "inside/b.rs", "TargetName", "needle", None)?;
    let scope = store::SearchScope::Globs(vec!["inside/*".into(), "inside/a.rs".into()]);
    let writes = store.writer()?.total_changes();
    for exact in [false, true] {
        for query in [
            "",
            "xy",
            "missing",
            if exact { "TargetName" } else { "needle" },
        ] {
            let read = |check: &dyn Fn() -> Result<(), store::Error>| {
                if exact {
                    store.exact_name_units_checked(query, 2, Some(&scope), &[excluded], check)
                } else {
                    store
                        .query_fts_scoped_checked(query, 2, Some(&scope), &[excluded], check)
                        .map(|hits| hits.into_iter().map(|hit| hit.unit_id).collect())
                }
            };
            let calls = Cell::new(0);
            let expected = read(&|| {
                calls.set(calls.get() + 1);
                Ok(())
            })?;
            if query == "TargetName" || query == "needle" {
                assert_eq!(expected, [first, second]);
            } else {
                assert!(expected.is_empty());
            }
            for stop in 1..=calls.get() {
                let seen = Cell::new(0);
                assert!(
                    matches!(
                        read(&|| {
                            seen.set(seen.get() + 1);
                            if seen.get() >= stop {
                                Err(store::Error::L1InputsChanged)
                            } else {
                                Ok(())
                            }
                        }),
                        Err(store::Error::L1InputsChanged)
                    ),
                    "exact={exact}, query={query}, boundary={stop}"
                );
            }
            assert_eq!(read(&|| Ok(()))?, expected);
            let ordinary = if exact {
                store.exact_name_units(query, 2, Some(&scope), &[excluded])?
            } else {
                store
                    .query_fts_scoped(query, 2, Some(&scope), &[excluded])?
                    .into_iter()
                    .map(|hit| hit.unit_id)
                    .collect()
            };
            assert_eq!(ordinary, expected);
        }
    }
    assert_eq!(store.writer()?.total_changes(), writes);
    assert_eq!(
        store
            .writer()?
            .query_row("PRAGMA busy_timeout", [], |r| r.get::<_, i64>(0))?,
        5000
    );
    Ok(())
}

#[test]
fn scoped_lexical_reader_contention_obeys_original_work_and_retries() -> Result<(), Box<dyn Error>>
{
    use std::{
        sync::{
            atomic::{AtomicBool, Ordering},
            mpsc,
        },
        thread,
        time::{Duration, Instant},
    };
    let store = store::open(":memory:", StoreLimits::default())?;
    let id = add(&store, "target.rs", "TargetName", "needle", None)?;
    let writes = store.writer()?.total_changes();
    for operation in 0..7 {
        for cancel in [false, true] {
            let held = store.writer()?;
            let cancelled = AtomicBool::new(false);
            thread::scope(|scope| {
                let (send, receive) = mpsc::channel();
                let cancelled = &cancelled;
                let store = &store;
                scope.spawn(move || {
                    let deadline = Instant::now() + Duration::from_millis(30);
                    let query_check = || {
                        if cancelled.load(Ordering::Acquire) || Instant::now() >= deadline {
                            Err(super::Error::Deadline)
                        } else {
                            Ok(())
                        }
                    };
                    let store_check = || query_check().map_err(|_| store::Error::L1InputsChanged);
                    let refused = match operation {
                        0 => matches!(
                            store.query_fts_scoped_checked("needle", 2, None, &[], &store_check),
                            Err(store::Error::L1InputsChanged)
                        ),
                        1 => matches!(
                            store.exact_name_units_checked(
                                "TargetName",
                                2,
                                None,
                                &[],
                                &store_check
                            ),
                            Err(store::Error::L1InputsChanged)
                        ),
                        3 => matches!(
                            store.symbol_seeds_checked(&[id], 20, &store_check),
                            Err(store::Error::L1InputsChanged)
                        ),
                        4 => matches!(
                            store.graph_neighbours_checked(&[id], 100, None, &[], &store_check),
                            Err(store::Error::L1InputsChanged)
                        ),
                        5 => matches!(
                            store.units_by_ids_checked(&[id], &store_check),
                            Err(store::Error::L1InputsChanged)
                        ),
                        6 => matches!(
                            collapse::prepare_with_work(
                                store,
                                &[id],
                                &BTreeSet::new(),
                                Some(&query_check)
                            ),
                            Err(super::Error::Deadline)
                        ),
                        _ => {
                            let parsed = query::parse("TargetName", fusion::Mode::Lexical).unwrap();
                            matches!(
                                lexical_with_selection(
                                    store,
                                    &parsed,
                                    1,
                                    None,
                                    &Search::default(),
                                    &[],
                                    &query_check
                                ),
                                Err(super::Error::Deadline)
                            )
                        }
                    };
                    send.send(refused).unwrap();
                });
                if cancel {
                    cancelled.store(true, Ordering::Release);
                }
                let result = receive.recv_timeout(Duration::from_secs(2));
                drop(held);
                assert!(result.unwrap(), "operation={operation}, cancel={cancel}");
            });
            let parsed = query::parse("TargetName", fusion::Mode::Lexical)?;
            let (hits, _) =
                lexical_with_selection(&store, &parsed, 1, None, &Search::default(), &[], &|| {
                    Ok(())
                })?;
            assert_eq!(hits[0].unit.id, id);
        }
    }
    assert_eq!(store.writer()?.total_changes(), writes);
    Ok(())
}

#[test]
fn graph_readers_preserve_rank_scope_seed_order_and_cancel_every_boundary()
-> Result<(), Box<dyn Error>> {
    use std::cell::Cell;
    let store = store::open(":memory:", StoreLimits::default())?;
    let source = add(
        &store,
        "inside/source.rs",
        "SourceName",
        "needle",
        Some("TargetName"),
    )?;
    let outside = add(&store, "outside/target.rs", "TargetName", "outside", None)?;
    let excluded = add(&store, "inside/excluded.rs", "TargetName", "excluded", None)?;
    let low = add(&store, "inside/low.rs", "TargetName", "low", None)?;
    let high = add(&store, "inside/high.rs", "TargetName", "high", None)?;
    let doc = add(&store, "inside/doc.rs", "Document", "doc", None)?;
    store
        .writer()?
        .execute("UPDATE units SET kind='doc_section' WHERE id=?1", [doc])?;
    store.writer()?.execute(
        "UPDATE symbols SET pagerank=CASE unit_id WHEN ?1 THEN 0.8 WHEN ?2 THEN 0.4 ELSE 0.9 END",
        [high, low],
    )?;
    let writes = store.writer()?.total_changes();
    let scope = store::SearchScope::Globs(vec!["inside/*".into(), "inside/high.rs".into()]);
    let ordered = [doc, high, -1, source, low, high];
    assert_eq!(
        store.symbol_seeds_checked(&ordered, 3, &|| Ok(()))?,
        [high, source, low]
    );
    let graph = |seeds: &[i64], limit, check: &dyn Fn() -> Result<(), store::Error>| {
        store
            .graph_neighbours_checked(seeds, limit, Some(&scope), &[excluded], check)
            .map(|hits| {
                hits.into_iter()
                    .map(|hit| (hit.unit_id, hit.rank))
                    .collect::<Vec<_>>()
            })
    };
    assert_eq!(graph(&[source, source], 1, &|| Ok(()))?, [(high, 0.8)]);
    assert_eq!(graph(&[source], 10, &|| Ok(()))?, [(high, 0.8), (low, 0.4)]);
    assert_eq!(graph(&[high], 10, &|| Ok(()))?, [(source, 0.9)]);
    assert!(outside < high);
    for seeds in [&ordered[..], &[][..], &[-1][..]] {
        for limit in [0, 1, 20] {
            for expand in [false, true] {
                let read = |check: &dyn Fn() -> Result<(), store::Error>| {
                    if expand {
                        graph(seeds, limit, check)
                    } else {
                        store
                            .symbol_seeds_checked(seeds, limit, check)
                            .map(|ids| ids.into_iter().map(|id| (id, 0.0)).collect())
                    }
                };
                let calls = Cell::new(0);
                let expected = read(&|| {
                    calls.set(calls.get() + 1);
                    Ok(())
                })?;
                if expand && seeds.len() == ordered.len() {
                    assert!(
                        calls.get() > ordered.len() * 2,
                        "checks must cover rows and seeds"
                    );
                }
                for stop in 1..=calls.get() {
                    let seen = Cell::new(0);
                    assert!(
                        matches!(
                            read(&|| {
                                seen.set(seen.get() + 1);
                                if seen.get() >= stop {
                                    Err(store::Error::L1InputsChanged)
                                } else {
                                    Ok(())
                                }
                            }),
                            Err(store::Error::L1InputsChanged)
                        ),
                        "expand={expand} limit={limit} stop={stop}"
                    );
                }
                assert_eq!(read(&|| Ok(()))?, expected);
                let ordinary = if expand {
                    store
                        .graph_neighbours(seeds, limit, Some(&scope), &[excluded])?
                        .into_iter()
                        .map(|hit| (hit.unit_id, hit.rank))
                        .collect::<Vec<_>>()
                } else {
                    store
                        .symbol_seeds(seeds, limit)?
                        .into_iter()
                        .map(|id| (id, 0.0))
                        .collect()
                };
                assert_eq!(ordinary, expected);
            }
        }
    }
    assert_eq!(store.writer()?.total_changes(), writes);
    Ok(())
}

#[test]
fn compact_snippets_bound_long_unicode_lines_and_preserve_short_source() {
    for body in [
        "let repeated_identifier = 123456789; ".repeat(1000),
        "日本語🙂 café ".repeat(1000),
        "first\nsecond\nthird\n".to_owned() + &"abcdef ".repeat(1000),
    ] {
        let result = snippet(&body);
        assert!(crate::treat::count_tokens(&result) <= 60);
        assert!(result.lines().count() <= 4);
        let prefix = result.strip_suffix('…').unwrap();
        assert!(body.starts_with(prefix));
    }
    assert_eq!(
        snippet("one\ntwo\nthree\nfour\nfive"),
        "one\ntwo\nthree\nfour"
    );
    assert_eq!(snippet(""), "");
}

#[test]
fn diagnostic_window_preserves_direct_pool_k_relaxation_and_exclusions()
-> Result<(), Box<dyn Error>> {
    let db = tempfile::NamedTempFile::new()?;
    let store = store::open(db.path(), StoreLimits::default())?;
    let mut ids = Vec::new();
    for i in 0..210 {
        ids.push(add(
            &store,
            &format!("src/diagnostic_{i}.rs"),
            &format!("entry_{i}"),
            if i < 2 { "shared uncommon" } else { "shared" },
            None,
        )?);
    }
    let settings = Search {
        graph_prior_weight: 0.0,
        ..Search::default()
    };
    let parsed = query::parse("shared uncommon", fusion::Mode::Lexical)?;
    let strict = lexical_candidates(&store, &parsed, 1, None, &settings, &[], None)?;
    assert_eq!(strict.hits.len(), 2);
    let relaxed = lexical_candidates(&store, &parsed, 3, None, &settings, &[], None)?;
    assert_eq!(relaxed.hits.len(), LEXICAL_LIMIT);
    let baseline = lexical(&store, "shared uncommon", 3, None, &settings)?;
    let shared = collapse::finish(relaxed.hits, 3, &settings, &relaxed.prepared)?;
    assert_eq!(
        format_response(&store, "shared uncommon", &baseline)?,
        format_response(&store, "shared uncommon", &shared)?
    );
    let excluded = lexical_candidates(&store, &parsed, 3, None, &settings, &ids[..2], None)?;
    assert_eq!(excluded.hits.len(), LEXICAL_LIMIT);
    assert!(
        excluded
            .hits
            .iter()
            .all(|hit| !ids[..2].contains(&hit.unit.id))
    );
    let stop = || Err(super::Error::Deadline);
    assert!(matches!(
        lexical_candidates(&store, &parsed, 3, None, &settings, &[], Some(&stop)),
        Err(super::Error::Deadline)
    ));
    Ok(())
}
