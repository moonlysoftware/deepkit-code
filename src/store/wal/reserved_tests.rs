//! Fixed WAL allowances on large indexes and failed publication; v1 §3.5/§8.2.
use super::*;
use crate::store::{self, StoreLimits};
use std::{
    error::Error as StdError,
    sync::atomic::{AtomicBool, Ordering},
};

fn database(path: &std::path::Path, size: u32) -> Result<Store, Box<dyn StdError>> {
    store::register_vector_extension()?;
    {
        let db = Connection::open(path)?;
        db.pragma_update(None, "page_size", size)?;
        db.execute_batch("VACUUM")?;
    }
    let store = store::open(path, StoreLimits::default())?;
    store.writer()?.execute_batch(
        "CREATE TABLE reserve_fixture(id INTEGER PRIMARY KEY,value BLOB);
         INSERT INTO reserve_fixture VALUES(1,zeroblob(8388608)),(2,'original');
         CREATE VIRTUAL TABLE reserve_fts USING fts5(text, content='');",
    )?;
    assert!(!checkpoint(&store)?.busy);
    Ok(store)
}

#[test]
fn small_writes_to_large_database_pause_for_reader_then_resume() -> Result<(), Box<dyn StdError>> {
    for size in [4096_u32, 65536] {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("code.db");
        let store = database(&path, size)?;
        let second = store::open(&path, StoreLimits::default())?;
        checkpoint(&store)?;
        let reservation = 8 * (u64::from(size) + 24) + 32;
        let cap = 2 * reservation;
        assert!(fs::metadata(&path)?.len() > cap);
        assert!(matches!(
            write(&store, cap, 0, |_| Ok(())),
            Err(Error::WalCapacity { .. })
        ));
        let mut reader = Connection::open(&path)?;
        let read = reader.transaction()?;
        assert_eq!(
            read.query_row("SELECT value FROM reserve_fixture WHERE id=2", [], |r| {
                r.get::<_, String>(0)
            })?,
            "original"
        );
        let mut completed = 0;
        for n in 0..100 {
            let worker = if n % 2 == 0 { &store } else { &second };
            let called = AtomicBool::new(false);
            let result = write_reserved(worker, cap, reservation, 0, |tx| {
                called.store(true, Ordering::Relaxed);
                tx.execute("UPDATE reserve_fixture SET value=?1 WHERE id=2", [n])?;
                Ok(())
            });
            assert!(wal_bytes(&read)? <= cap);
            match result {
                Ok(()) => completed += 1,
                Err(Error::WalCapacity { .. }) => {
                    assert!(
                        !called.load(Ordering::Relaxed),
                        "preflight must refuse before mutation"
                    );
                    break;
                }
                Err(error) => return Err(error.into()),
            }
        }
        assert!(completed > 0 && completed < 100);
        assert!(checkpoint(&store)?.busy);
        assert_eq!(
            read.query_row("SELECT value FROM reserve_fixture WHERE id=2", [], |r| {
                r.get::<_, String>(0)
            })?,
            "original"
        );
        let physical = wal_bytes(&read)?;
        read.commit()?;
        assert!(!checkpoint(&store)?.busy);
        write_reserved(&store, cap, reservation, 0, |tx| {
            tx.execute("UPDATE reserve_fixture SET value='resumed' WHERE id=2", [])?;
            Ok(())
        })?;
        assert_eq!(
            store
                .writer()?
                .query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
            "ok"
        );
        println!(
            "Fixed WAL allowance: page_size={size}, main_bytes={}, cap={cap}, reservation={reservation}, writes_before_pause={completed}, held_wal={physical}",
            fs::metadata(&path)?.len()
        );
    }
    Ok(())
}

#[test]
fn overflow_during_spill_commit_and_fts_sync_rolls_back_without_crossing_cap()
-> Result<(), Box<dyn StdError>> {
    for size in [4096_u32, 65536] {
        for mode in ["spill", "commit", "fts"] {
            let directory = tempfile::tempdir()?;
            let store = database(&directory.path().join("code.db"), size)?;
            store.writer()?.pragma_update(
                None,
                "cache_size",
                if mode == "spill" { 1 } else { -65536 },
            )?;
            // FULL introduces possible commit padding; the VFS must count that too.
            store.writer()?.pragma_update(None, "synchronous", "FULL")?;
            let cap = 3 * (u64::from(size) + 24) + 32;
            let old_limit = native_limit(&*store.writer()?)?;
            let old_ceiling: u32 = store
                .writer()?
                .query_row("PRAGMA max_page_count", [], |r| r.get(0))?;
            let old_spill: i64 = store
                .writer()?
                .query_row("PRAGMA cache_spill", [], |r| r.get(0))?;
            let completed_action = AtomicBool::new(false);
            let result = write_reserved(&store, cap, cap, 4096, |tx| {
                tx.execute(
                    "UPDATE reserve_fixture SET value='uncommitted' WHERE id=2",
                    [],
                )?;
                if mode == "fts" {
                    let text: String = (0..30000).map(|i| format!("distinct{i} ")).collect();
                    tx.execute("INSERT INTO reserve_fts(text) VALUES(?1)", [text])?;
                } else {
                    tx.execute(
                        "INSERT INTO reserve_fixture VALUES(3,randomblob(2097152))",
                        [],
                    )?;
                }
                completed_action.store(true, Ordering::Relaxed);
                Ok(())
            });
            assert!(
                matches!(result, Err(Error::WalCapacity { .. })),
                "{mode}: {result:?}"
            );
            if mode == "spill" {
                assert!(
                    !completed_action.load(Ordering::Relaxed),
                    "spill must fail during mutation"
                );
            } else if mode == "commit" {
                assert!(
                    completed_action.load(Ordering::Relaxed),
                    "commit boundary must be exercised"
                );
            }
            let db = store.writer()?;
            let physical = wal_bytes(&db)?;
            assert!(physical <= cap);
            assert_eq!(native_limit(&db)?, old_limit);
            assert_eq!(
                db.query_row("PRAGMA max_page_count", [], |r| r.get::<_, u32>(0))?,
                old_ceiling
            );
            assert_eq!(
                db.query_row("PRAGMA cache_spill", [], |r| r.get::<_, i64>(0))?,
                old_spill
            );
            assert_eq!(
                db.query_row("SELECT value FROM reserve_fixture WHERE id=2", [], |r| {
                    r.get::<_, String>(0)
                })?,
                "original"
            );
            assert_eq!(
                db.query_row("SELECT count(*) FROM reserve_fixture", [], |r| r
                    .get::<_, i64>(0))?,
                2
            );
            assert_eq!(
                db.query_row("SELECT count(*) FROM reserve_fts", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            assert_eq!(
                db.query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
                "ok"
            );
            drop(db);
            assert!(!checkpoint(&store)?.busy);
            assert_eq!(wal_bytes(&*store.writer()?)?, 0);
            write_reserved(&store, cap, cap, 0, |tx| {
                tx.execute("UPDATE reserve_fixture SET value='resumed' WHERE id=2", [])?;
                Ok(())
            })?;
            println!(
                "WAL overflow rollback: page_size={size}, boundary={mode}, cap={cap}, failed_wal={physical}, integrity=ok"
            );
        }
    }
    Ok(())
}

#[test]
fn native_limit_rejects_invalid_values_and_reservation_respects_existing_ceiling()
-> Result<(), Box<dyn StdError>> {
    let directory = tempfile::tempdir()?;
    let store = database(&directory.path().join("code.db"), 4096)?;
    {
        let db = store.writer()?;
        db.pragma_update(None, "dkc_wal_limit", 16384)?;
        for invalid in ["-1", "garbage", "9223372036854775808", "", "1.5"] {
            assert!(db.pragma_update(None, "dkc_wal_limit", invalid).is_err());
            assert_eq!(native_limit(&db)?, 16384);
        }
    }
    assert!(matches!(
        write_reserved(&store, 32768, 32768, 0, |_| -> Result<(), Error> {
            panic!("unadmitted callback")
        }),
        Err(Error::WalCapacity { .. })
    ));
    assert_eq!(native_limit(&*store.writer()?)?, 16384);
    assert!(matches!(
        write_reserved(&store, 32768, 0, 0, |_| -> Result<(), Error> {
            panic!("zero reservation")
        }),
        Err(Error::WalCapacity { .. })
    ));
    let memory = store::open(":memory:", StoreLimits::default())?;
    assert!(matches!(
        write_reserved(&memory, 32768, 16384, 0, |_| Ok(())),
        Err(Error::WalReservationUnsupported)
    ));
    Ok(())
}
