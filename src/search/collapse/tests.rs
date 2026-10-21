//! Parent identity, evidence and per-file result limits; v1 §9 step 8.
use super::*;
use crate::{
    search::{format_response_verbose, fusion, materialize},
    store::StoreLimits,
};
use rusqlite::params;

fn unit(
    store: &Store,
    id: i64,
    kind: &str,
    path: Option<&str>,
    range: Option<(i64, i64)>,
    parent: Option<i64>,
) {
    let hash = *blake3::hash(&id.to_le_bytes()).as_bytes();
    let mut writer = store.writer().unwrap();
    let tx = writer.transaction().unwrap();
    let file=path.map(|path| {
        tx.execute("INSERT OR IGNORE INTO files(path,blake3,size,mtime_ns,kind,max_level,treatment) VALUES(?1,?2,200,1,'code',3,'fixture')",params![path,blake3::hash(path.as_bytes()).as_bytes().as_slice()]).unwrap();
        tx.query_row("SELECT id FROM files WHERE path=?1",[path],|r|r.get::<_,i64>(0)).unwrap()
    });
    tx.execute(
        "INSERT INTO unit_contents(hash,text,tokens) VALUES(?1,?2,1)",
        params![hash.as_slice(), format!("body {id}")],
    )
    .unwrap();
    tx.execute("INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,byte_start,byte_end,line_start,line_end,header,parent_unit,created_at,updated_at) VALUES(?1,?2,?2,'collapse:' || ?1,?3,?4,?5,?6,1,1,?7,?8,1,1)",params![id,hash.as_slice(),file,kind,range.map(|r|r.0),range.map(|r|r.1),format!("unit {id}"),parent]).unwrap();
    tx.commit().unwrap();
}
fn fixture() -> Store {
    let store = store::open(":memory:", StoreLimits::default()).unwrap();
    unit(&store, 1, "symbol", Some("src/a.rs"), Some((0, 100)), None);
    unit(&store, 2, "symbol", Some("src/a.rs"), Some((20, 80)), None);
    unit(&store, 3, "chunk", Some("src/a.rs"), Some((30, 40)), None);
    unit(&store, 4, "summary", None, None, Some(3));
    unit(&store, 5, "symbol", Some("src/b.rs"), Some((0, 100)), None);
    unit(&store, 6, "chunk", Some("src/a.rs"), Some((110, 120)), None);
    unit(&store, 7, "chunk", Some("src/a.rs"), None, None);
    unit(&store, 8, "chunk", Some("src/a.rs"), Some((70, 120)), None);
    unit(&store, 9, "chunk", Some("src/b.rs"), Some((30, 40)), None);
    unit(&store, 10, "commit", None, None, None);
    unit(&store, 11, "commit", None, None, None);
    store
}
fn candidate(store: &Store, id: i64, score: f64, leg: &str) -> RankedCandidate {
    let contribution = Some(fusion::Contribution { rank: 1, score });
    let mut scores = fusion::Scores::default();
    match leg {
        "lexical" => scores.lexical = contribution,
        "dense" => scores.dense = contribution,
        "graph" => scores.graph = contribution,
        _ => panic!("unknown test leg"),
    }
    let units = store.units_by_ids(&[id]).unwrap();
    materialize(
        vec![fusion::Hit { id, score, scores }],
        &BTreeSet::new(),
        &mut units.into_iter().map(|unit| (unit.id, unit)).collect(),
    )
    .pop()
    .unwrap()
}

#[test]
fn nested_projection_keeps_winning_score_and_every_original_match() {
    let store = fixture();
    let hits = apply(
        &store,
        vec![
            candidate(&store, 4, 0.9, "dense"),
            candidate(&store, 3, 0.8, "lexical"),
            candidate(&store, 2, 0.5, "graph"),
        ],
        10,
        &Search::default(),
        &BTreeSet::new(),
    )
    .unwrap();
    assert_eq!(hits.len(), 1);
    let hit = &hits[0];
    assert_eq!(hit.unit.id, 2);
    assert_eq!(hit.unit.text, "body 2");
    assert_eq!(hit.score, 0.9);
    assert!(hit.summary && hit.chunk && hit.dense && hit.lexical && hit.graph);
    assert!(hit.scores.lexical.is_none()); // No invented aggregate RRF score.
    assert_eq!(hit.scores.dense.as_ref().unwrap().score, 0.9);
    assert_eq!(
        hit.matches.iter().map(|e| e.unit_id).collect::<Vec<_>>(),
        [4, 3, 2]
    );
    let verbose = format_response_verbose(&store, "query", &hits, true).unwrap();
    assert_eq!(verbose["hits"][0]["matches"].as_array().unwrap().len(), 3);
    assert_eq!(
        verbose["hits"][0]["matches"][1]["rrf"]["lexical"]["score"],
        0.8
    );
    assert!(verbose["hits"][0]["matches"][0].get("unit_id").is_none());
    assert!(
        verbose["hits"][0]["why"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("summary"))
    );
    let compact = format_response_verbose(&store, "query", &hits, false).unwrap();
    assert!(compact["hits"][0].get("matches").is_none());
}

#[test]
fn containment_uses_same_file_bytes_and_exclusions_cannot_be_reintroduced() {
    let store = fixture();
    let input = [6, 7, 8, 9]
        .into_iter()
        .map(|id| candidate(&store, id, 0.5, "lexical"))
        .collect();
    let settings = Search {
        max_hits_per_file: 10,
        ..Search::default()
    };
    let hits = apply(&store, input, 10, &settings, &BTreeSet::new()).unwrap();
    assert_eq!(
        hits.iter().map(|h| h.unit.id).collect::<Vec<_>>(),
        [6, 7, 8, 5]
    );
    for excluded in [2, 3, 4] {
        assert!(
            apply(
                &store,
                vec![candidate(&store, 4, 0.9, "dense")],
                10,
                &settings,
                &BTreeSet::from([excluded])
            )
            .unwrap()
            .is_empty()
        );
    }
}

#[test]
fn file_caps_follow_projection_and_do_not_group_unrelated_pathless_units() {
    let store = fixture();
    let hits = apply(
        &store,
        [4, 6, 9, 10, 11]
            .into_iter()
            .map(|id| candidate(&store, id, 0.5, "lexical"))
            .collect(),
        10,
        &Search {
            max_hits_per_file: 1,
            ..Search::default()
        },
        &BTreeSet::new(),
    )
    .unwrap();
    assert_eq!(
        hits.iter().map(|h| h.unit.id).collect::<Vec<_>>(),
        [2, 5, 10, 11]
    );
    let hits = apply(
        &store,
        hits,
        3,
        &Search {
            max_hits_per_file: 1,
            ..Search::default()
        },
        &BTreeSet::new(),
    )
    .unwrap();
    assert_eq!(
        hits.iter().map(|h| h.unit.id).collect::<Vec<_>>(),
        [2, 5, 10]
    );
}

#[test]
fn malformed_summary_parents_fail_and_deleted_occurrences_are_omitted() {
    let store = fixture();
    unit(&store, 12, "summary", None, None, None);
    unit(&store, 13, "summary", None, None, Some(12));
    store
        .writer()
        .unwrap()
        .execute("UPDATE units SET parent_unit=13 WHERE id=12", [])
        .unwrap();
    assert!(matches!(
        apply(
            &store,
            vec![candidate(&store, 12, 0.5, "dense")],
            10,
            &Search::default(),
            &BTreeSet::new()
        ),
        Err(Error::InvalidSummaryParent { unit_id: 12 })
    ));
    unit(&store, 14, "summary", None, None, None);
    assert!(matches!(
        apply(
            &store,
            vec![candidate(&store, 14, 0.5, "dense")],
            10,
            &Search::default(),
            &BTreeSet::new()
        ),
        Err(Error::InvalidSummaryParent { unit_id: 14 })
    ));
    let hit = candidate(&store, 10, 0.5, "lexical");
    store
        .writer()
        .unwrap()
        .execute("DELETE FROM units WHERE id=10", [])
        .unwrap();
    assert!(
        apply(&store, vec![hit], 10, &Search::default(), &BTreeSet::new())
            .unwrap()
            .is_empty()
    );
}

fn eligible(store: &Store, excluded: &[i64]) -> Vec<i64> {
    let rule = store::projection::exclusion_sql(excluded, 1);
    store
        .writer()
        .unwrap()
        .prepare(&format!(
            "{}SELECT u.id FROM units AS u WHERE {} ORDER BY u.id",
            rule.prefix, rule.predicate
        ))
        .unwrap()
        .query_map([serde_json::json!(excluded).to_string()], |row| row.get(0))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

#[test]
fn early_exclusions_match_nested_collapse_and_follow_summary_chains() {
    let store = fixture();
    unit(&store, 12, "summary", None, None, Some(4));
    unit(&store, 13, "summary", None, None, Some(12));
    for excluded in [
        vec![1],
        vec![2],
        vec![3],
        vec![4],
        vec![5],
        vec![9],
        vec![2, 10],
        Vec::new(),
    ] {
        let allowed = eligible(&store, &excluded);
        for id in 1..=13 {
            let collapsed = apply(
                &store,
                vec![candidate(&store, id, 0.5, "dense")],
                1,
                &Search::default(),
                &excluded.iter().copied().collect(),
            )
            .unwrap();
            assert_eq!(
                !collapsed.is_empty(),
                allowed.contains(&id),
                "source {id}, exclusions {excluded:?}"
            );
        }
    }
    assert!(eligible(&store, &[1]).contains(&3)); // Inner symbol is a different result.
    assert!(!eligible(&store, &[2]).contains(&13)); // Transitive summary of a chunk.
}

#[test]
fn exclusion_expansion_rechecks_parent_ranges_and_terminates_summary_cycles() {
    let store = fixture();
    assert!(!eligible(&store, &[2]).contains(&3));
    store
        .writer()
        .unwrap()
        .execute("UPDATE units SET byte_end=25 WHERE id=2", [])
        .unwrap();
    assert!(eligible(&store, &[2]).contains(&3)); // Its collapse target is now outer symbol 1.
    assert!(!eligible(&store, &[1]).contains(&3));
    unit(&store, 12, "summary", None, None, None);
    unit(&store, 13, "summary", None, None, Some(12));
    store
        .writer()
        .unwrap()
        .execute("UPDATE units SET parent_unit=13 WHERE id=12", [])
        .unwrap();
    let allowed = eligible(&store, &[12]);
    assert!(!allowed.contains(&12));
    assert!(!allowed.contains(&13));
}

#[test]
fn later_projection_suppression_replaces_the_original_occurrence_too() {
    let store = fixture();
    let hit = candidate(&store, 3, 0.7, "dense");
    let mut prepared = prepare(&store, &[3], &BTreeSet::new()).unwrap();
    assert!(prepared.units.contains_key(&3));
    prepared.extend(prepare(&store, &[3], &BTreeSet::from([2])).unwrap());
    assert!(!prepared.units.contains_key(&3));
    assert!(
        finish(vec![hit], 10, &Search::default(), &prepared)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn reranked_child_winner_keeps_its_score_and_all_parent_match_scores() {
    let store = fixture();
    let mut hits = vec![
        candidate(&store, 4, 0.04, "dense"),
        candidate(&store, 3, 0.03, "lexical"),
        candidate(&store, 2, 0.02, "graph"),
        candidate(&store, 5, 0.01, "dense"),
    ];
    crate::search::rerank::apply_scores(
        &mut hits,
        &[0.7, 0.9, 0.8, 0.85],
        &crate::config::Rerank::default(),
        &Search::default(),
    )
    .unwrap();
    let hits = apply(&store, hits, 10, &Search::default(), &BTreeSet::new()).unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.unit.id).collect::<Vec<_>>(),
        [2, 5]
    );
    assert_eq!(hits[0].rerank_score, Some(0.9));
    assert_eq!(hits[0].rrf_score, 0.03);
    assert_eq!(
        hits[0]
            .matches
            .iter()
            .map(|hit| (hit.unit_id, hit.rerank_score))
            .collect::<Vec<_>>(),
        [(3, Some(0.9)), (2, Some(0.8)), (4, Some(0.7))]
    );
    let verbose = format_response_verbose(&store, "needle", &hits, true).unwrap();
    assert_eq!(
        verbose["hits"][0]["matches"][1]["rrf_score"],
        serde_json::json!(0.02)
    );
}

#[test]
fn checked_hydration_preserves_order_duplicates_and_refuses_every_boundary() {
    use std::cell::Cell;
    let store = fixture();
    let writes = store.writer().unwrap().total_changes();
    for ids in [vec![], vec![4, 999, 3, 4, 5]] {
        let read = |check: &dyn Fn() -> Result<(), store::Error>| {
            store.units_by_ids_checked(&ids, check).map(|units| {
                units
                    .into_iter()
                    .map(|unit| (unit.id, unit.text))
                    .collect::<Vec<_>>()
            })
        };
        let count = Cell::new(0);
        let expected = read(&|| {
            count.set(count.get() + 1);
            Ok(())
        })
        .unwrap();
        let ordinary = store
            .units_by_ids(&ids)
            .unwrap()
            .into_iter()
            .map(|unit| (unit.id, unit.text))
            .collect::<Vec<_>>();
        assert_eq!(expected, ordinary);
        if !ids.is_empty() {
            assert_eq!(
                expected.iter().map(|entry| entry.0).collect::<Vec<_>>(),
                [4, 3, 4, 5]
            );
        }
        for stop in 1..=count.get() {
            let calls = Cell::new(0);
            assert!(
                matches!(
                    read(&|| {
                        calls.set(calls.get() + 1);
                        if calls.get() >= stop {
                            Err(store::Error::L1InputsChanged)
                        } else {
                            Ok(())
                        }
                    }),
                    Err(store::Error::L1InputsChanged)
                ),
                "boundary={stop}"
            );
        }
        assert_eq!(read(&|| Ok(())).unwrap(), expected);
    }
    assert_eq!(store.writer().unwrap().total_changes(), writes);
}

#[test]
fn checked_collapse_cancels_parent_chains_and_preserves_projection_and_first_error() {
    use std::cell::Cell;
    let store = fixture();
    for id in 12..32 {
        unit(
            &store,
            id,
            "summary",
            None,
            None,
            Some(if id == 12 { 4 } else { id - 1 }),
        );
    }
    let writes = store.writer().unwrap().total_changes();
    for ids in [vec![], vec![31, 3, 31, 999, 5, 6, 7, 8, 9]] {
        for excluded in [BTreeSet::new(), BTreeSet::from([2]), BTreeSet::from([20])] {
            let describe = |prepared: Prepared| {
                let originals = prepared.units.keys().copied().collect::<Vec<_>>();
                let projections = prepared
                    .projections
                    .into_iter()
                    .map(|(id, p)| (id, p.map(|p| (p.unit.id, p.unit.text, p.summary, p.chunk))))
                    .collect::<Vec<_>>();
                (originals, projections, prepared.links)
            };
            let count = Cell::new(0);
            let expected = describe(
                prepare_with_work(
                    &store,
                    &ids,
                    &excluded,
                    Some(&|| {
                        count.set(count.get() + 1);
                        Ok(())
                    }),
                )
                .unwrap(),
            );
            assert_eq!(
                describe(prepare(&store, &ids, &excluded).unwrap()),
                expected
            );
            if !ids.is_empty() && excluded.is_empty() {
                assert!(
                    count.get() > 30,
                    "parent chain must be checked inside the snapshot"
                );
                assert_eq!(expected.1[0].1.as_ref().unwrap().0, 2);
            }
            for stop in 1..=count.get() {
                let calls = Cell::new(0);
                let result = prepare_with_work(
                    &store,
                    &ids,
                    &excluded,
                    Some(&|| {
                        calls.set(calls.get() + 1);
                        if calls.get() == stop {
                            Err(Error::Deadline)
                        } else if calls.get() > stop {
                            Err(Error::QueryTask)
                        } else {
                            Ok(())
                        }
                    }),
                );
                assert!(matches!(result, Err(Error::Deadline)), "boundary={stop}");
            }
            assert_eq!(
                describe(prepare_with_work(&store, &ids, &excluded, Some(&|| Ok(()))).unwrap()),
                expected
            );
        }
    }
    assert_eq!(store.writer().unwrap().total_changes(), writes);
    assert_eq!(
        store
            .writer()
            .unwrap()
            .query_row("PRAGMA busy_timeout", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        5000
    );
}
