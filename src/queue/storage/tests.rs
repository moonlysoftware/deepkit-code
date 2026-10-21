//! File-backed queue admission, billing retention and rollback; v1 §3.5/§8.2.
use super::*;
use crate::{
    queue::{self, capacity, retry, usage},
    store::{self, StoreLimits},
};
use std::error::Error as StdError;

#[test]
fn held_reader_preserves_control_headroom_then_reports_exhaustion_without_losing_billing()
-> Result<(), Box<dyn StdError>> {
    for size in [4096_u32, 65536] {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("code.db");
        store::register_vector_extension()?;
        {
            let db = Connection::open(&path)?;
            db.pragma_update(None, "page_size", size)?;
            db.execute_batch("VACUUM")?;
        }
        let cap = 8 * 1024 * 1024;
        let limits = StoreLimits {
            wal_max_bytes: cap,
            ..Default::default()
        };
        let store = store::open(&path, limits)?;
        let second = store::open(&path, limits)?;
        store.writer()?.execute_batch(
            "CREATE TABLE publication(value BLOB); INSERT INTO publication VALUES('original')",
        )?;
        for id in 1..=2 {
            queue::enqueue(&store, &queue::tests::input(id, &[1; 32]), 0)?;
        }
        let owner = queue::Owner::new(&store)?;
        let leases = queue::claim(&store, "embed:local", &owner, 0, 2, 100)?;
        let budget = usage::Budget {
            calls: None,
            tokens: None,
        };
        let usage::Admission::Reserved(reservation) =
            usage::reserve(&store, "embed:local", 10, 0, budget, Some(&leases[0]), || 1)?
        else {
            panic!("admission");
        };
        assert_eq!(
            usage::mark_dispatched(
                &store,
                &reservation,
                Some(&leases[0]),
                || 1,
                |_, _| Ok(usage::Limits {
                    budget,
                    rpm: None,
                    tpm: None
                })
            )?,
            usage::Dispatch::Recorded
        );
        wal::checkpoint(&store)?;
        let mut reader = Connection::open(&path)?;
        let read = reader.transaction()?;
        assert_eq!(
            read.query_row("SELECT value FROM publication", [], |r| r
                .get::<_, String>(0))?,
            "original"
        );
        let error = queue::complete(
            &store,
            &leases[0],
            || 2,
            |_, _| {
                Ok(Some(queue::CurrentInput {
                    target_hash: [7; 32],
                    desired_hash: [1; 32],
                }))
            },
            |tx| {
                tx.execute("UPDATE publication SET value=zeroblob(16777216)", [])?;
                Ok(())
            },
        )
        .unwrap_err();
        assert!(matches!(
            capacity::reason(&error),
            Some(capacity::Reason::Wal)
        ));
        assert!(wal::wal_bytes(&read)? <= cap);
        // Even with the reader preventing reclamation, a control write can
        // release the lease and record the pause after failed publication.
        assert_eq!(
            retry::schedule(
                &store,
                &leases[0],
                retry::Outcome::Capacity(capacity::Reason::Wal),
                || 2
            )?,
            retry::Scheduled::RetryAt(capacity::PAUSED)
        );
        let snapshot = serde_json::to_value(queue::status::snapshot(&store, 2)?)?;
        assert_eq!(
            snapshot["providers"]["embed:local"]["deferrals"]["cap_wal"]["jobs"],
            1
        );
        assert_eq!(snapshot["providers"]["embed:local"]["uncertain_calls"], 1);
        assert_eq!(
            store
                .writer()?
                .query_row("SELECT value FROM publication", [], |r| r
                    .get::<_, String>(0))?,
            "original"
        );
        // New admission stops before consuming the control region.
        let mut admitted = 0;
        for id in 3..1000 {
            match queue::enqueue(&second, &queue::tests::input(id, &[1; 32]), 2) {
                Ok(_) => admitted += 1,
                Err(Error::Store(store::Error::WalCapacity { .. })) => break,
                Err(error) => return Err(error.into()),
            }
            assert!(wal::wal_bytes(&read)? <= cap);
        }
        assert!(admitted < 997);
        let mut renewals = 0;
        for time in 3..2000 {
            match queue::renew(&second, &leases[1], time) {
                Ok(true) => renewals += 1,
                Err(Error::Store(store::Error::WalCapacity { .. })) => break,
                other => panic!("unexpected renewal: {other:?}"),
            }
            assert!(wal::wal_bytes(&read)? <= cap);
        }
        assert!(renewals > 0 && renewals < 1997);
        assert!(matches!(
            usage::reconcile(&store, &reservation, Some(7)),
            Err(Error::Store(store::Error::WalCapacity { .. }))
        ));
        let snapshot = queue::status::snapshot(&store, 2)?;
        assert!(snapshot.summary().contains("control paused"));
        let snapshot = serde_json::to_value(snapshot)?;
        assert_eq!(snapshot["wal"]["growth_paused"], true);
        assert_eq!(snapshot["wal"]["control_paused"], true);
        assert_eq!(snapshot["providers"]["embed:local"]["uncertain_calls"], 1);
        let physical = wal::wal_bytes(&read)?;
        assert!(physical <= cap);
        read.commit()?;
        // The queue boundary now checkpoints on demand and admits exactly once.
        assert!(usage::reconcile(&store, &reservation, Some(7))?);
        assert_eq!(capacity::resume(&store, "embed:local", || 2001)?, 1);
        assert_eq!(
            store
                .writer()?
                .query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
            "ok"
        );
        println!(
            "Queue WAL backpressure: page_size={size}, cap={cap}, failed_then_admitted={admitted}, renewals={renewals}, held_wal={physical}, billing_recovered=7"
        );
    }
    Ok(())
}

#[test]
fn writer_owned_completion_commits_publication_after_lease_deadline_on_disk()
-> Result<(), Box<dyn StdError>> {
    let dir = tempfile::tempdir()?;
    let store = store::open(dir.path().join("code.db"), StoreLimits::default())?;
    store.writer()?.execute_batch(
        "CREATE TABLE publication(value TEXT); INSERT INTO publication VALUES('original')",
    )?;
    queue::enqueue(&store, &queue::tests::input(1, &[1; 32]), 0)?;
    let lease = queue::claim(
        &store,
        "embed:local",
        &queue::Owner::new(&store)?,
        0,
        1,
        100,
    )?
    .remove(0);
    let mut now = 0;
    assert_eq!(
        queue::complete(
            &store,
            &lease,
            || {
                let previous = now;
                now = 121;
                previous
            },
            |_, _| Ok(Some(queue::CurrentInput {
                target_hash: [7; 32],
                desired_hash: [1; 32]
            })),
            |tx| {
                tx.execute("UPDATE publication SET value='uncommitted'", [])?;
                Ok(())
            }
        )?,
        queue::Completion::Published
    );
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT value FROM publication", [], |r| r
                .get::<_, String>(0))?,
        "uncommitted"
    );
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT state FROM jobs", [], |r| r.get::<_, String>(0))?,
        "done"
    );
    Ok(())
}

#[test]
fn bulk_retry_and_capacity_resume_shrink_oversized_batches_and_keep_progress()
-> Result<(), Box<dyn StdError>> {
    for size in [4096_u32, 65536] {
        for resume in [false, true] {
            let directory = tempfile::tempdir()?;
            let path = directory.path().join("code.db");
            store::register_vector_extension()?;
            {
                let db = Connection::open(&path)?;
                db.pragma_update(None, "page_size", size)?;
                db.execute_batch("VACUUM")?;
            }
            let cap = 8 * 1024 * 1024;
            let store = store::open(
                &path,
                StoreLimits {
                    wal_max_bytes: cap,
                    ..Default::default()
                },
            )?;
            {
                let mut db = store.writer()?;
                let tx = db.transaction()?;
                for id in 1..=257 {
                    queue::enqueue_in(&tx, &queue::tests::input(id, &[1; 32]), 0, None)?;
                }
                tx.execute(
                    "UPDATE jobs SET state=?1,attempts=2,not_before=?2,last_error=?3",
                    rusqlite::params![
                        if resume { "queued" } else { "failed" },
                        if resume { capacity::PAUSED } else { 86400 },
                        capacity::Reason::Wal.message()
                    ],
                )?;
                tx.execute_batch("CREATE TABLE bulk_events(id INTEGER PRIMARY KEY,payload BLOB);
                    CREATE TRIGGER bulk_payload AFTER UPDATE ON jobs BEGIN INSERT INTO bulk_events VALUES(NEW.id,zeroblob(32768)); END;")?;
                tx.commit()?;
            }
            wal::checkpoint(&store)?;
            // Prove this workload cannot fit the former single control transaction.
            let unbounded = transaction(&store, Kind::Control, |tx| {
                tx.execute("UPDATE jobs SET updated_at=10", [])?;
                tx.commit()?;
                Ok(())
            });
            assert!(matches!(
                unbounded,
                Err(Error::Store(store::Error::WalCapacity { .. }))
            ));
            assert_eq!(
                store
                    .writer()?
                    .query_row("SELECT count(*) FROM bulk_events", [], |r| r
                        .get::<_, i64>(0))?,
                0
            );
            wal::checkpoint(&store)?;
            let changed = if resume {
                capacity::resume(&store, "embed:local", || 11)?
            } else {
                retry::request(&store, None, 11)?
            };
            assert_eq!(changed, 257);
            assert_eq!(
                store
                    .writer()?
                    .query_row("SELECT count(*) FROM bulk_events", [], |r| r
                        .get::<_, i64>(0))?,
                257
            );
            let valid: i64 = store.writer()?.query_row("SELECT count(*) FROM jobs WHERE state='queued' AND attempts=2 AND updated_at=11 AND not_before=?1", [if resume {11} else {86400}], |r|r.get(0))?;
            assert_eq!(valid, 257);
            assert!(wal::wal_bytes(&*store.writer()?)? <= cap);
            assert_eq!(
                store
                    .writer()?
                    .query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
                "ok"
            );
            println!(
                "Bulk queue WAL: page_size={size}, resume={resume}, changed={changed}, event_bytes={}, cap={cap}",
                257 * 32768
            );
        }
    }
    Ok(())
}

#[test]
fn bulk_pass_keeps_starting_id_boundary_and_rechecks_later_eligibility()
-> Result<(), Box<dyn StdError>> {
    let store = store::open(":memory:", StoreLimits::default())?;
    for id in 1..=130 {
        queue::enqueue(&store, &queue::tests::input(id, &[1; 32]), 0)?;
    }
    store.writer()?.execute_batch("UPDATE jobs SET state='failed',attempts=1,not_before=86400;
        CREATE TRIGGER change_later_batch AFTER UPDATE OF state ON jobs WHEN NEW.id=1 BEGIN
          UPDATE jobs SET attempts=8 WHERE id=100;
          INSERT INTO jobs(kind,target_kind,target_id,target_hash,desired_hash,provider,priority,tokens,state,attempts,not_before,created_at,updated_at)
          VALUES('embed','unit',131,zeroblob(32),zeroblob(32),'embed:local',1,10,'failed',1,86400,0,0);
        END;")?;
    assert_eq!(retry::request(&store, None, 1)?, 129);
    assert_eq!(
        store.writer()?.query_row(
            "SELECT count(*) FROM jobs WHERE id IN(100,131) AND state='failed'",
            [],
            |r| r.get::<_, i64>(0)
        )?,
        2
    );
    assert_eq!(retry::request(&store, None, 2)?, 1);
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT attempts FROM jobs WHERE id=100", [], |r| r
                .get::<_, i64>(0))?,
        8
    );
    Ok(())
}

#[test]
fn bulk_wal_backpressure_keeps_committed_batches_and_resumes_after_reader_release()
-> Result<(), Box<dyn StdError>> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("code.db");
    let cap = 4 * 1024 * 1024;
    let store = store::open(
        &path,
        StoreLimits {
            wal_max_bytes: cap,
            ..Default::default()
        },
    )?;
    {
        let mut db = store.writer()?;
        let tx = db.transaction()?;
        for id in 1..=257 {
            queue::enqueue_in(&tx, &queue::tests::input(id, &[1; 32]), 0, None)?;
        }
        tx.execute_batch("UPDATE jobs SET state='failed';
            CREATE TABLE bulk_events(id INTEGER PRIMARY KEY,payload BLOB);
            CREATE TRIGGER bulk_payload AFTER UPDATE ON jobs BEGIN INSERT INTO bulk_events VALUES(NEW.id,zeroblob(32768)); END;")?;
        tx.commit()?;
    }
    wal::checkpoint(&store)?;
    let mut reader = Connection::open(&path)?;
    let read = reader.transaction()?;
    assert_eq!(
        read.query_row("SELECT count(*) FROM bulk_events", [], |r| r
            .get::<_, i64>(0))?,
        0
    );
    let error = retry::request(&store, None, 1).unwrap_err();
    let Error::BulkIncomplete { changed, source } = error else {
        panic!("expected partial progress: {error:?}");
    };
    assert!(changed > 0 && changed < 257);
    assert!(matches!(
        *source,
        Error::Store(store::Error::WalCapacity { .. })
    ));
    let physical = wal::wal_bytes(&read)?;
    assert!(physical <= cap);
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT count(*) FROM bulk_events", [], |r| r
                .get::<_, i64>(0))?,
        changed as i64
    );
    assert_eq!(
        read.query_row("SELECT count(*) FROM bulk_events", [], |r| r
            .get::<_, i64>(0))?,
        0
    );
    assert!(wal::checkpoint(&store)?.busy);
    read.commit()?;
    let resumed = retry::request(&store, None, 2)?;
    assert_eq!(changed + resumed, 257);
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT count(*) FROM bulk_events", [], |r| r
                .get::<_, i64>(0))?,
        257
    );
    assert!(wal::wal_bytes(&*store.writer()?)? <= cap);
    assert_eq!(
        store
            .writer()?
            .query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
        "ok"
    );
    println!(
        "Bulk reader backpressure: cap={cap}, committed_before_pause={changed}, resumed={resumed}, held_wal={physical}"
    );
    Ok(())
}

#[test]
fn recovery_batches_bound_both_tables_and_preserve_dispatched_charges()
-> Result<(), Box<dyn StdError>> {
    for size in [4096_u32, 65536] {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("code.db");
        store::register_vector_extension()?;
        {
            let db = Connection::open(&path)?;
            db.pragma_update(None, "page_size", size)?;
            db.execute_batch("VACUUM")?;
        }
        let cap = 8 * 1024 * 1024;
        let store = store::open(
            &path,
            StoreLimits {
                wal_max_bytes: cap,
                ..Default::default()
            },
        )?;
        {
            let mut db = store.writer()?;
            let tx = db.transaction()?;
            for id in 1..=257 {
                queue::enqueue_in(&tx, &queue::tests::input(id, &[1; 32]), 0, None)?;
                for state in ["reserved", "dispatched"] {
                    tx.execute(
                        "INSERT INTO usage_reservations
                        (id,provider,day,job_id,lease_generation,calls,tokens,state,created_at)
                        VALUES(?1,'embed:local','1970-01-01',?2,1,1,10,?3,0)",
                        rusqlite::params![format!("{state}-{id:04}"), id, state],
                    )?;
                }
            }
            tx.execute_batch("UPDATE jobs SET state='leased',lease_generation=1,lease_owner='old',leased_until=120,attempts=2;
                CREATE TABLE recovery_events(kind TEXT,id TEXT,payload BLOB,PRIMARY KEY(kind,id));
                CREATE TRIGGER recovery_jobs AFTER UPDATE ON jobs BEGIN
                    INSERT INTO recovery_events VALUES('job',NEW.id,zeroblob(32768)); END;
                CREATE TRIGGER recovery_usage AFTER UPDATE ON usage_reservations BEGIN
                    INSERT INTO recovery_events VALUES('usage',NEW.id,zeroblob(32768)); END;")?;
            tx.commit()?;
        }
        wal::checkpoint(&store)?;
        let unbounded = transaction(&store, Kind::Control, |tx| {
            let result = retry::recover_in(&tx, "embed:local", 120)?;
            tx.commit()?;
            Ok(result)
        });
        assert!(matches!(
            unbounded,
            Err(Error::Store(store::Error::WalCapacity { .. }))
        ));
        wal::checkpoint(&store)?;
        let mut samples = 0;
        assert_eq!(
            retry::recover(&store, "embed:local", || {
                samples += 1;
                120
            })?,
            retry::Recovered {
                jobs: 257,
                reservations: 257
            }
        );
        assert_eq!(samples, 1);
        let db = store.writer()?;
        assert_eq!(
            db.query_row("SELECT count(*) FROM recovery_events", [], |r| r
                .get::<_, i64>(0))?,
            514
        );
        assert_eq!(db.query_row("SELECT count(*) FROM jobs WHERE state='failed' AND attempts=2 AND lease_generation=1 AND lease_owner IS NULL", [], |r| r.get::<_,i64>(0))?, 257);
        assert_eq!(db.query_row("SELECT count(*) FROM usage_reservations WHERE state='dispatched' AND calls=1 AND tokens=10", [], |r| r.get::<_,i64>(0))?, 257);
        assert_eq!(
            db.query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
            "ok"
        );
        assert!(wal::wal_bytes(&db)? <= cap);
        drop(db);
        assert_eq!(
            retry::recover(&store, "embed:local", || 120)?,
            retry::Recovered::default()
        );
        println!(
            "Recovery WAL: page_size={size}, jobs=257, released=257, dispatched=257, cap={cap}"
        );
    }
    Ok(())
}

#[test]
fn claim_batches_rollback_before_resizing_and_resample_cancellation()
-> Result<(), Box<dyn StdError>> {
    for size in [4096_u32, 65536] {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("code.db");
        store::register_vector_extension()?;
        {
            let db = Connection::open(&path)?;
            db.pragma_update(None, "page_size", size)?;
            db.execute_batch("VACUUM")?;
        }
        let cap = 8 * 1024 * 1024;
        let store = store::open(
            &path,
            StoreLimits {
                wal_max_bytes: cap,
                ..Default::default()
            },
        )?;
        {
            let mut db = store.writer()?;
            let tx = db.transaction()?;
            for id in 1..=257 {
                queue::enqueue_in(&tx, &queue::tests::input(id, &[1; 32]), 0, None)?;
            }
            tx.execute_batch(
                "CREATE TABLE claim_events(id INTEGER PRIMARY KEY,payload BLOB);
                CREATE TRIGGER claim_payload AFTER UPDATE ON jobs WHEN NEW.state='leased' BEGIN
                    INSERT INTO claim_events VALUES(NEW.id,zeroblob(131072)); END;",
            )?;
            tx.commit()?;
        }
        wal::checkpoint(&store)?;
        let owner = queue::Owner::new(&store)?;
        let mut samples = 0;
        let cancelled = queue::claim_with_clock(
            &store,
            "embed:local",
            &owner,
            || {
                samples += 1;
                if samples >= 3 { -1 } else { 10 }
            },
            257,
            10000,
        );
        assert!(matches!(cancelled, Err(Error::Invalid("lease timestamp"))));
        assert_eq!(samples, 3);
        assert_eq!(
            store
                .writer()?
                .query_row("SELECT count(*) FROM claim_events", [], |r| r
                    .get::<_, i64>(0))?,
            0
        );
        assert_eq!(
            store.writer()?.query_row(
                "SELECT count(*) FROM jobs WHERE lease_generation=0 AND state='queued'",
                [],
                |r| r.get::<_, i64>(0)
            )?,
            257
        );
        let mut ids = std::collections::BTreeSet::new();
        loop {
            let batch = queue::claim(&store, "embed:local", &owner, 10, 257, 10000)?;
            if batch.is_empty() {
                break;
            }
            assert!(batch.len() < 64);
            for lease in batch {
                assert_eq!(lease.generation, 1);
                assert!(ids.insert(lease.id));
            }
            assert!(wal::wal_bytes(&*store.writer()?)? <= cap);
        }
        assert_eq!(ids.len(), 257);
        let db = store.writer()?;
        assert_eq!(
            db.query_row("SELECT count(*) FROM claim_events", [], |r| r
                .get::<_, i64>(0))?,
            257
        );
        assert_eq!(db.query_row("SELECT count(*) FROM jobs WHERE state='leased' AND attempts=0 AND leased_until=130", [], |r| r.get::<_,i64>(0))?,257);
        assert_eq!(
            db.query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
            "ok"
        );
        println!(
            "Claim WAL: page_size={size}, unique_leases=257, cancelled_replay_samples={samples}, cap={cap}"
        );
    }
    Ok(())
}
