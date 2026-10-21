//! Real SQLite publication, cap and cascade regressions.
use super::*;
use crate::store::{self, StoreLimits};

fn fixture() -> Store {
    let store = store::open(
        ":memory:",
        StoreLimits {
            max_l3_units: 2,
            ..Default::default()
        },
    )
    .unwrap();
    store.writer().unwrap().execute_batch("INSERT INTO files(id,path,blake3,size,mtime_ns,kind,max_level,treatment)
        VALUES(1,'source.rs',zeroblob(32),5,0,'code',3,'source');
        INSERT INTO unit_contents(hash,text,tokens) VALUES(zeroblob(32),'parent source',2);
        INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,created_at,updated_at)
        VALUES(1,zeroblob(32),zeroblob(32),'source.rs',1,'chunk','parent',0,0);
        INSERT INTO symbols(id,file_id,unit_id,name,kind,line_start,line_end)
        VALUES(1,1,1,'first','function',1,1),(2,1,1,'second','function',1,1),(3,1,1,'third','function',1,1);").unwrap();
    store
}
fn input(owner: i64, text: &str) -> Publication<'_> {
    Publication {
        owner: Owner::Symbol(owner),
        parent_unit: 1,
        parent_hash: &[0; 32],
        desired_hash: &[7; 32],
        text,
        sources: &[],
        now: 10,
    }
}

#[test]
fn summary_generation_tracks_only_committed_changed_output() {
    let store = fixture();
    assert_eq!(store.summary_generation().unwrap(), 0);
    let l1 = store.l1_identity().unwrap();
    {
        let mut db = store.writer().unwrap();
        let tx = db.transaction().unwrap();
        store
            .publish_summary_in(&tx, input(1, "first output"))
            .unwrap();
        assert_eq!(summary_generation_in(&tx).unwrap(), 1);
        // Simulated failed worker completion must not wake catalogue consumers.
    }
    assert_eq!(store.summary_generation().unwrap(), 0);
    for (text, expected) in [
        ("first output", 1),
        ("first output", 1),
        ("updated output", 2),
    ] {
        let mut db = store.writer().unwrap();
        let tx = db.transaction().unwrap();
        store.publish_summary_in(&tx, input(1, text)).unwrap();
        assert_eq!(summary_generation_in(&tx).unwrap(), expected);
        tx.commit().unwrap();
        drop(db);
        assert_eq!(store.summary_generation().unwrap(), expected);
    }
    assert_eq!(store.l1_identity().unwrap(), l1);
    for invalid in ["-1", "corrupt", "9223372036854775807"] {
        store.set_meta("summary_generation", invalid).unwrap();
        {
            let mut db = store.writer().unwrap();
            let tx = db.transaction().unwrap();
            assert!(matches!(
                store.publish_summary_in(&tx, input(1, "must roll back")),
                Err(Error::SummaryGeneration)
            ));
        }
        assert!(!store.query_fts("updated", 10).unwrap().is_empty());
        assert!(store.query_fts("must", 10).unwrap().is_empty());
    }
}

#[test]
fn replacement_preserves_other_owners_and_allows_lowered_count_cap() {
    let mut store = fixture();
    let (first, second) = {
        let mut writer = store.writer().unwrap();
        let tx = writer.transaction().unwrap();
        let first = store
            .publish_summary_in(&tx, input(1, "original first description"))
            .unwrap();
        let second = store
            .publish_summary_in(&tx, input(2, "retained second description"))
            .unwrap();
        assert_ne!(first, second);
        tx.commit().unwrap();
        (first, second)
    };
    store.max_l3_units = 0;
    let replacement = {
        let mut writer = store.writer().unwrap();
        let tx = writer.transaction().unwrap();
        let replacement = store
            .publish_summary_in(
                &tx,
                Publication {
                    sources: &[Source::File(1), Source::Symbol(2)],
                    ..input(1, "replacement description")
                },
            )
            .unwrap();
        assert_eq!(
            tx.query_row("SELECT summary_unit FROM symbols WHERE id=2", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            second
        );
        assert_eq!(
            tx.query_row("SELECT count(*) FROM units WHERE kind='summary'", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            2
        );
        assert_eq!(
            tx.query_row(
                "SELECT count(*) FROM unit_sources WHERE unit_id=?1",
                [replacement],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
            3
        );
        tx.commit().unwrap();
        replacement
    };
    assert!(store.query_fts("original", 10).unwrap().is_empty());
    assert!(!store.query_fts("replacement", 10).unwrap().is_empty());
    {
        let mut writer = store.writer().unwrap();
        let tx = writer.transaction().unwrap();
        let same = store
            .publish_summary_in(
                &tx,
                Publication {
                    sources: &[Source::File(1), Source::Symbol(2)],
                    ..input(1, "replacement description")
                },
            )
            .unwrap();
        assert_eq!(same, replacement);
        tx.commit().unwrap();
    }
    {
        let mut writer = store.writer().unwrap();
        let tx = writer.transaction().unwrap();
        assert!(matches!(
            store.publish_summary_in(&tx, input(3, "blocked new coverage")),
            Err(Error::SummaryCapReached { .. })
        ));
        // Completion must roll back all writes, including owner pointers and FTS.
        tx.rollback().unwrap();
    }
    assert!(store.query_fts("blocked", 10).unwrap().is_empty());
    assert_eq!(store.enrichment_status().unwrap().summaries, 2);
    // Rowids may be reused after deletion; identity is checked through text and FTS.
    let _ = first;
    store.delete_unit(1).unwrap();
    assert_eq!(store.enrichment_status().unwrap().summaries, 0);
    assert!(store.query_fts("description", 10).unwrap().is_empty());
}

#[test]
fn stale_parent_foreign_owner_and_unknown_provenance_cannot_replace_summary() {
    let store = fixture();
    let mut writer = store.writer().unwrap();
    let tx = writer.transaction().unwrap();
    let first = store
        .publish_summary_in(&tx, input(1, "retained original"))
        .unwrap();
    assert!(matches!(
        store.publish_summary_in(
            &tx,
            Publication {
                parent_hash: &[1; 32],
                ..input(1, "stale replacement")
            }
        ),
        Err(Error::L1InputsChanged)
    ));
    assert!(matches!(
        store.publish_summary_in(
            &tx,
            Publication {
                sources: &[Source::File(999)],
                ..input(1, "unknown provenance")
            }
        ),
        Err(Error::L1InputsChanged)
    ));
    tx.execute("UPDATE symbols SET summary_unit=?1 WHERE id=2", [first])
        .unwrap();
    assert!(matches!(
        store.publish_summary_in(&tx, input(2, "stolen summary")),
        Err(Error::L1InputsChanged)
    ));
    assert_eq!(
        tx.query_row("SELECT count(*) FROM units WHERE kind='summary'", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    tx.rollback().unwrap();
}

#[test]
fn composite_owner_pointers_publish_and_cascade_with_their_parent() {
    let store = fixture();
    let mut writer = store.writer().unwrap();
    let tx = writer.transaction().unwrap();
    for (id, kind) in [(11, "pr"), (12, "module")] {
        tx.execute(
            "INSERT INTO units(id,hash,content_hash,source_key,kind,header,created_at,updated_at)
            VALUES(?1,?2,zeroblob(32),?3,?4,'composite',0,0)",
            params![id, vec![id as u8; 32], format!("{kind}:{id}"), kind],
        )
        .unwrap();
    }
    tx.execute(
        "INSERT INTO prs(id,title,merge_sha,strategy,commit_count,merged_at,unit_id)
        VALUES(9,'fixture',zeroblob(20),'merge',1,10,11)",
        [],
    )
    .unwrap();
    tx.execute(
        "INSERT INTO dirs(path,merkle,file_count,module_unit) VALUES('src',zeroblob(32),5,12)",
        [],
    )
    .unwrap();
    let pr = store
        .publish_summary_in(
            &tx,
            Publication {
                owner: Owner::Pr(9),
                parent_unit: 11,
                ..input(1, "pullrequest summary")
            },
        )
        .unwrap();
    let module = store
        .publish_summary_in(
            &tx,
            Publication {
                owner: Owner::Module(12),
                parent_unit: 12,
                ..input(1, "module summary")
            },
        )
        .unwrap();
    assert_eq!(
        tx.query_row("SELECT summary_unit FROM prs WHERE id=9", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        pr
    );
    assert_eq!(
        tx.query_row("SELECT summary_unit FROM dirs WHERE path='src'", [], |r| {
            r.get::<_, i64>(0)
        })
        .unwrap(),
        module
    );
    tx.execute("DELETE FROM units WHERE id IN (11,12)", [])
        .unwrap();
    assert_eq!(
        tx.query_row("SELECT count(*) FROM units WHERE kind='summary'", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        tx.query_row("SELECT summary_unit FROM prs WHERE id=9", [], |r| r
            .get::<_, Option<i64>>(0))
            .unwrap(),
        None
    );
    assert_eq!(
        tx.query_row("SELECT summary_unit FROM dirs WHERE path='src'", [], |r| {
            r.get::<_, Option<i64>>(0)
        })
        .unwrap(),
        None
    );
    tx.commit().unwrap();
}

#[test]
fn soft_database_threshold_rejects_new_summary_and_rollback_keeps_source() {
    let mut store = fixture();
    let allocated = store.enrichment_status().unwrap().allocated_bytes;
    store.db_max_bytes = allocated * 100 / 85;
    let mut writer = store.writer().unwrap();
    let tx = writer.transaction().unwrap();
    assert!(matches!(
        store.publish_summary_in(&tx, input(1, "refused growth")),
        Err(Error::EnrichmentDatabasePaused { .. })
    ));
    tx.rollback().unwrap();
    drop(writer);
    assert_eq!(store.enrichment_status().unwrap().summaries, 0);
    assert!(store.query_fts("refused", 10).unwrap().is_empty());
}

#[test]
fn replacement_claims_preserve_new_slots_and_skip_paused_growth_at_higher_priority() {
    use crate::queue;
    let store = fixture();
    let enqueue = |target_id, desired: &[u8; 32], provider, priority| {
        queue::enqueue(
            &store,
            &queue::Enqueue {
                kind: queue::Kind::DescribeSymbol,
                target: queue::Target::Symbol,
                target_id,
                target_hash: &[0; 32],
                desired_hash: desired,
                provider,
                priority,
                tokens: 1,
            },
            0,
        )
    };
    // This new-coverage job was queued before the cap became full.
    let growth = enqueue(3, &[1; 32], "first", 99.0).unwrap();
    {
        let mut writer = store.writer().unwrap();
        let tx = writer.transaction().unwrap();
        store
            .publish_summary_in(&tx, input(1, "first summary"))
            .unwrap();
        store
            .publish_summary_in(&tx, input(2, "second summary"))
            .unwrap();
        tx.commit().unwrap();
    }
    let replacement = enqueue(1, &[2; 32], "first", 0.1).unwrap();
    assert!(matches!(
        enqueue(3, &[3; 32], "first", 1.0),
        Err(queue::Error::Store(Error::SummaryCapReached { .. }))
    ));
    let owner = queue::Owner::new(&store).unwrap();
    let leases = queue::claim(&store, "first", &owner, 1, 10, 10).unwrap();
    assert_eq!(leases.len(), 1);
    assert_eq!(leases[0].id, replacement);
    assert_ne!(leases[0].id, growth);
    let other = enqueue(2, &[4; 32], "second", 1.0).unwrap();
    let leases = queue::claim(
        &store,
        "second",
        &queue::Owner::new(&store).unwrap(),
        1,
        10,
        10,
    )
    .unwrap();
    assert_eq!(leases.len(), 1);
    assert_eq!(leases[0].id, other);
    let status = serde_json::to_value(queue::status::snapshot(&store, 2).unwrap()).unwrap();
    assert_eq!(status["enrichment_claims"]["active_leases"], 2);
    assert_eq!(status["enrichment_claims"]["growth_leases"], 0);
    assert_eq!(status["enrichment_claims"]["available_slots"], 0);
}
