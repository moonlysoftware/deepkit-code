//! Conservative WAL reservations and checkpoint backpressure; v1 §3.5/§8.2.
#![expect(
    dead_code,
    reason = "T1.5 reservation foundation; queue/control write integration follows"
)]

use super::{Error, Store};
use rusqlite::{Connection, Transaction, TransactionBehavior};
use std::{fs, path::PathBuf, time::Duration};

/// A checkpoint can return normally while a reader still prevents reclamation.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Checkpoint {
    pub(crate) busy: bool,
    pub(crate) log_frames: i64,
    pub(crate) checkpointed_frames: i64,
}

/// Attempts WAL truncation without waiting for readers. Never deletes WAL files
/// directly; SQLite decides what committed snapshots still require. Run on a
/// blocking thread; the connection's usual busy timeout is restored afterwards.
pub(crate) fn checkpoint(store: &Store) -> Result<Checkpoint, Error> {
    checkpoint_in(&*store.writer()?)
}

/// Checkpoints an already admitted connection without acquiring the Store mutex.
pub(crate) fn checkpoint_in(db: &Connection) -> Result<Checkpoint, Error> {
    let timeout: i64 = db.query_row("PRAGMA busy_timeout", [], |r| r.get(0))?;
    db.busy_timeout(Duration::ZERO)?;
    let result = db.query_row("PRAGMA main.wal_checkpoint(TRUNCATE)", [], |r| {
        Ok(Checkpoint {
            busy: r.get::<_, i64>(0)? != 0,
            log_frames: r.get(1)?,
            checkpointed_frames: r.get(2)?,
        })
    });
    db.pragma_update(None, "busy_timeout", timeout)?;
    Ok(result?)
}

/// Observes allocated WAL bytes without mutating or checkpointing a snapshot.
pub(crate) fn wal_bytes(db: &Connection) -> Result<u64, Error> {
    let path = db
        .path()
        .filter(|path| !path.is_empty())
        .ok_or(Error::WalReservationUnsupported)?;
    let mut wal = PathBuf::from(path).into_os_string();
    wal.push("-wal");
    match fs::metadata(PathBuf::from(wal)) {
        Ok(metadata) => Ok(metadata.len()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(source) => Err(Error::Io { source }),
    }
}

fn native_limit(db: &Connection) -> Result<i64, Error> {
    let value: String = db.query_row("PRAGMA main.dkc_wal_limit", [], |r| r.get(0))?;
    value.parse().map_err(|_| Error::WalReservationUnsupported)
}

fn native_limit_hit(db: &Connection) -> Result<bool, Error> {
    let value: String = db.query_row("PRAGMA main.dkc_wal_limit_hit", [], |r| r.get(0))?;
    match value.as_str() {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(Error::WalReservationUnsupported),
    }
}

/// Reserves an explicit WAL byte allowance under BEGIN IMMEDIATE, independent of
/// database size. The named VFS rejects writes/truncation beyond the reservation;
/// the allowance includes frame headers, commit padding and any spilled pages.
/// Overflow rolls back publication and retains partial WAL bytes for checkpoint
/// reclamation. Already allocated WAL is conservatively counted even if reusable.
///
/// Cache spilling stays enabled; this does not prove an overall RSS bound. The
/// caller bounds input and work, and handles refusal with checkpoint/backpressure.
/// SQLite virtual-table commit work is covered at the same file-write boundary.
/// Callback SQL must not change pragmas, attach databases or manage transactions.
/// Settings are restored on normal return/errors; callbacks must not panic.
/// Existing queue/L1/bootstrap writers are not yet routed through this API.
pub(crate) fn write_reserved<T>(
    store: &Store,
    wal_max_bytes: u64,
    reservation_bytes: u64,
    growth_pages: u32,
    action: impl FnOnce(&Transaction<'_>) -> Result<T, Error>,
) -> Result<T, Error> {
    transaction_reserved(
        store,
        wal_max_bytes,
        reservation_bytes,
        Some(growth_pages),
        |tx| {
            let value = action(&tx)?;
            if native_limit_hit(&tx)? {
                return Err(Error::WalCapacity {
                    wal_max_bytes,
                    allocated_bytes: wal_bytes(&tx)?,
                    reservation_bytes,
                });
            }
            tx.commit()?;
            Ok(value)
        },
    )
}

/// Applies a reservation while preserving the caller's explicit commit/rollback
/// decisions. Used by queue transactions whose successful early returns may
/// intentionally roll back publication after a lease fence changes. Callbacks
/// must propagate write failures, never suppress them and commit partial work.
/// No external work or WAL-policy changes are allowed in the callback. With no
/// temporary growth allowance, authoritative main-file cap updates are retained.
pub(crate) fn transaction_reserved<T, E>(
    store: &Store,
    wal_max_bytes: u64,
    reservation_bytes: u64,
    growth_pages: Option<u32>,
    action: impl FnOnce(Transaction<'_>) -> Result<T, E>,
) -> Result<T, E>
where
    E: From<Error> + From<rusqlite::Error>,
{
    transaction_with_reservation(
        store,
        wal_max_bytes,
        Some(reservation_bytes),
        growth_pages,
        action,
    )
}

/// Reserves the remaining WAL growth space under the writer lock for an atomic
/// schema migration. Other writers cannot consume the allowance before commit.
pub(crate) fn transaction_remaining<T, E>(
    store: &Store,
    wal_max_bytes: u64,
    action: impl FnOnce(Transaction<'_>) -> Result<T, E>,
) -> Result<T, E>
where
    E: From<Error> + From<rusqlite::Error>,
{
    transaction_with_reservation(store, wal_max_bytes, None, None, action)
}

fn transaction_with_reservation<T, E>(
    store: &Store,
    wal_max_bytes: u64,
    requested_bytes: Option<u64>,
    growth_pages: Option<u32>,
    action: impl FnOnce(Transaction<'_>) -> Result<T, E>,
) -> Result<T, E>
where
    E: From<Error> + From<rusqlite::Error>,
{
    let mut db = store.writer()?;
    transaction_in(
        &mut db,
        wal_max_bytes,
        requested_bytes,
        growth_pages,
        None,
        action,
    )
}

/// Reserves on the caller's writer connection. A bounded admission signal only
/// polls lock acquisition; source/config validation belongs in the action. The
/// caller must disable SQLite busy waiting when supplying a signal.
pub(crate) fn transaction_in<T, E>(
    db: &mut Connection,
    wal_max_bytes: u64,
    requested_bytes: Option<u64>,
    growth_pages: Option<u32>,
    check: Option<&dyn Fn() -> Result<(), Error>>,
    action: impl FnOnce(Transaction<'_>) -> Result<T, E>,
) -> Result<T, E>
where
    E: From<Error> + From<rusqlite::Error>,
{
    let mut reservation_bytes = requested_bytes.unwrap_or(0);
    let mode: String = db.query_row("PRAGMA main.journal_mode", [], |r| r.get(0))?;
    if mode != "wal" {
        return Err(Error::WalReservationUnsupported.into());
    }
    let old_limit = native_limit(db)?;
    let old_ceiling: u32 = db.query_row("PRAGMA main.max_page_count", [], |r| r.get(0))?;
    let spill: i64 = db.query_row("PRAGMA cache_spill", [], |r| r.get(0))?;
    db.pragma_update(None, "cache_spill", true)?;
    let result = (|| {
        let tx = match check {
            Some(check) => super::checked_writer::begin(db, check)?,
            None => db.transaction_with_behavior(TransactionBehavior::Immediate)?,
        };
        let allocated_bytes = wal_bytes(&tx)?;
        reservation_bytes =
            requested_bytes.unwrap_or_else(|| wal_max_bytes.saturating_sub(allocated_bytes));
        let capacity = || Error::WalCapacity {
            wal_max_bytes,
            allocated_bytes,
            reservation_bytes,
        };
        let page_size: u32 = tx.query_row("PRAGMA main.page_size", [], |r| r.get(0))?;
        let end = allocated_bytes
            .checked_add(reservation_bytes)
            .filter(|end| *end <= wal_max_bytes)
            .and_then(|end| i64::try_from(end).ok())
            .filter(|end| *end <= old_limit)
            .ok_or_else(capacity)?;
        if reservation_bytes < u64::from(page_size) + 24 + 32 {
            return Err(capacity().into());
        }
        let pages: u32 = tx.query_row("PRAGMA main.page_count", [], |r| r.get(0))?;
        let ceiling = growth_pages.map_or(old_ceiling, |growth| {
            pages.saturating_add(growth).min(old_ceiling)
        });
        if ceiling < pages {
            return Err(Error::DatabaseGrowthLimit.into());
        }
        tx.pragma_update(None, "dkc_wal_limit", end)?;
        if native_limit(&tx)? != end {
            return Err(Error::WalReservationUnsupported.into());
        }
        let actual: u32 =
            tx.pragma_update_and_check(None, "max_page_count", ceiling, |r| r.get(0))?;
        if actual != ceiling {
            return Err(Error::DatabaseGrowthLimit.into());
        }
        action(tx)
    })();
    // Inspect before resetting the native flag; attempt every restoration even
    // when a previous one fails, as in the conservative reservation API.
    let hit = native_limit_hit(db);
    let restore_limit = db.pragma_update(None, "dkc_wal_limit", old_limit);
    let restore_ceiling = if growth_pages.is_some() {
        db.pragma_update(None, "max_page_count", old_ceiling)
    } else {
        Ok(())
    };
    let restore_spill = db.pragma_update(None, "cache_spill", spill);
    restore_limit?;
    restore_ceiling?;
    restore_spill?;
    if hit? {
        return Err(Error::WalCapacity {
            wal_max_bytes,
            allocated_bytes: wal_bytes(db)?,
            reservation_bytes,
        }
        .into());
    }
    result
}

/// Guards VACUUM's SQLite-owned transaction with an absolute native WAL ceiling.
/// A competing writer between preflight and VACUUM can consume room, but cannot
/// make this connection write past that ceiling. Failure rolls back atomically.
/// Temporary rebuild storage is disk-backed; cache spilling remains enabled.
pub(crate) fn compact(store: &Store, wal_max_bytes: u64) -> Result<(), Error> {
    checkpoint(store)?;
    let mut db = store.writer()?;
    let file_backed = db.path().is_some_and(|path| !path.is_empty());
    let old_limit = if file_backed {
        Some(native_limit(&db)?)
    } else {
        None
    };
    let spill: i64 = db.query_row("PRAGMA cache_spill", [], |r| r.get(0))?;
    let temp_store: i64 = db.query_row("PRAGMA temp_store", [], |r| r.get(0))?;
    let mut reservation_bytes = 0;
    let result = (|| -> Result<(), Error> {
        {
            let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let pages: i64 = tx.query_row("PRAGMA page_count", [], |r| r.get(0))?;
            let free: i64 = tx.query_row("PRAGMA freelist_count", [], |r| r.get(0))?;
            if free.saturating_mul(4) <= pages {
                return Ok(());
            }
            super::capacity::limit_main_growth(&tx, store.db_max_bytes)?;
            if let Some(old_limit) = old_limit {
                let allocated_bytes = wal_bytes(&tx)?;
                let ceiling = i64::try_from(wal_max_bytes)
                    .map_err(|_| Error::WalReservationUnsupported)?
                    .min(old_limit);
                reservation_bytes = (ceiling as u64).saturating_sub(allocated_bytes);
                let page_size: u32 = tx.query_row("PRAGMA page_size", [], |r| r.get(0))?;
                if reservation_bytes < u64::from(page_size) + 56 {
                    return Err(Error::WalCapacity {
                        wal_max_bytes,
                        allocated_bytes,
                        reservation_bytes,
                    });
                }
                tx.pragma_update(None, "dkc_wal_limit", ceiling)?;
                if native_limit(&tx)? != ceiling {
                    return Err(Error::WalReservationUnsupported);
                }
            }
            // No SQL data changes; release the lock for VACUUM's own transaction.
            tx.commit()?;
        }
        db.pragma_update(None, "temp_store", "FILE")?;
        db.pragma_update(None, "cache_spill", true)?;
        db.execute_batch("VACUUM")?;
        Ok(())
    })();
    let hit = if file_backed {
        native_limit_hit(&db)
    } else {
        Ok(false)
    };
    let restore_limit = old_limit.map_or(Ok(()), |limit| {
        db.pragma_update(None, "dkc_wal_limit", limit)
    });
    let restore_spill = db.pragma_update(None, "cache_spill", spill);
    let restore_temp = db.pragma_update(None, "temp_store", temp_store);
    restore_limit?;
    restore_spill?;
    restore_temp?;
    if hit? {
        return Err(Error::WalCapacity {
            wal_max_bytes,
            allocated_bytes: wal_bytes(&db)?,
            reservation_bytes,
        });
    }
    result
}

/// Runs one ordinary-table write with a conservative reservation under SQLite's
/// cross-process writer lock. Reserves all possible database pages plus one frame
/// and the WAL header; repeated modifications cannot spill before commit. Existing
/// physical WAL bytes are counted even if SQLite might be able to reuse them.
///
/// `growth_pages` bounds additional main-file pages during this transaction, also
/// respecting the installed main-file ceiling. Callback SQL must not ATTACH,
/// change pragmas, flush caches, manage transactions, or write virtual tables.
/// NORMAL synchronous WAL is required so commit does not pad frames for syncing.
/// Settings are restored on success and ordinary errors; callbacks must not panic.
///
/// This whole-database reservation deliberately rejects large databases under
/// small WAL caps. It is a safe foundation, not the final per-operation reservation
/// or a claim that existing unguarded queue/L1/bootstrap writes enforce wal_max_bytes.
pub(crate) fn write<T>(
    store: &Store,
    wal_max_bytes: u64,
    growth_pages: u32,
    action: impl FnOnce(&Transaction<'_>) -> Result<T, Error>,
) -> Result<T, Error> {
    let mut db = store.writer()?;
    let mode: String = db.query_row("PRAGMA main.journal_mode", [], |r| r.get(0))?;
    let synchronous: i64 = db.query_row("PRAGMA main.synchronous", [], |r| r.get(0))?;
    if mode != "wal" || synchronous != 1 {
        return Err(Error::WalReservationUnsupported);
    }
    let spill: i64 = db.query_row("PRAGMA cache_spill", [], |r| r.get(0))?;
    let old_ceiling: u32 = db.query_row("PRAGMA main.max_page_count", [], |r| r.get(0))?;
    db.pragma_update(None, "cache_spill", false)?;
    let result = (|| {
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let pages: u32 = tx.query_row("PRAGMA main.page_count", [], |r| r.get(0))?;
        let page_size: u32 = tx.query_row("PRAGMA main.page_size", [], |r| r.get(0))?;
        let ceiling = pages.saturating_add(growth_pages).min(old_ceiling);
        if pages == 0 || ceiling < pages {
            return Err(Error::WalReservationUnsupported);
        }
        let reservation = (u64::from(ceiling) + 1)
            .checked_mul(u64::from(page_size) + 24)
            .and_then(|frames| frames.checked_add(32))
            .ok_or(Error::WalReservationUnsupported)?;
        let allocated = wal_bytes(&tx)?;
        if allocated
            .checked_add(reservation)
            .is_none_or(|total| total > wal_max_bytes)
        {
            return Err(Error::WalCapacity {
                wal_max_bytes,
                allocated_bytes: allocated,
                reservation_bytes: reservation,
            });
        }
        let actual: u32 =
            tx.pragma_update_and_check(None, "max_page_count", ceiling, |r| r.get(0))?;
        if actual != ceiling {
            return Err(Error::DatabaseGrowthLimit);
        }
        let value = action(&tx)?;
        tx.commit()?;
        Ok(value)
    })();
    // Try both restorations even if the first fails. Never return a successful
    // result with a connection silently left under an operation-specific policy.
    let restore_ceiling = db.pragma_update(None, "max_page_count", old_ceiling);
    let restore_spill = db.pragma_update(None, "cache_spill", spill);
    restore_ceiling?;
    restore_spill?;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{self, StoreLimits};
    use std::{
        error::Error as StdError,
        sync::atomic::{AtomicBool, Ordering},
    };

    #[test]
    fn held_reader_stops_guarded_writes_before_cap_and_release_allows_resumption()
    -> Result<(), Box<dyn StdError>> {
        for page_size in [4096, 65536] {
            let directory = tempfile::tempdir()?;
            let path = directory.path().join("code.db");
            store::register_vector_extension()?;
            {
                let db = Connection::open(&path)?;
                db.pragma_update(None, "page_size", page_size)?;
                db.execute_batch("VACUUM")?;
            }
            let store = store::open(&path, StoreLimits::default())?;
            store.writer()?.execute_batch("CREATE TABLE wal_fixture(id INTEGER PRIMARY KEY,value BLOB); INSERT INTO wal_fixture VALUES(1,zeroblob(65536))")?;
            assert!(!checkpoint(&store)?.busy);
            // This second writer represents another embedded worker/process. Each
            // reservation observes the shared WAL after acquiring BEGIN IMMEDIATE.
            let second = store::open(&path, StoreLimits::default())?;
            let mut reader = Connection::open(&path)?;
            let read = reader.transaction()?;
            let original: Vec<u8> =
                read.query_row("SELECT value FROM wal_fixture WHERE id=1", [], |r| r.get(0))?;
            let (pages, size): (u32, u32) = store.writer()?.query_row(
                "SELECT page_count,page_size FROM pragma_page_count(),pragma_page_size()",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )?;
            let reservation = (u64::from(pages) + 1) * (u64::from(size) + 24) + 32;
            let cap = reservation + 4 * (u64::from(size) + 24);
            let mut completed = 0;
            for index in 0..100 {
                let worker = if index % 2 == 0 { &store } else { &second };
                let called = AtomicBool::new(false);
                let result = write(worker, cap, 0, |tx| {
                    called.store(true, Ordering::Relaxed);
                    tx.execute(
                        "UPDATE wal_fixture SET value=randomblob(65536) WHERE id=1",
                        [],
                    )?;
                    Ok(())
                });
                assert!(wal_bytes(&*store.writer()?)? <= cap);
                match result {
                    Ok(()) => completed += 1,
                    Err(Error::WalCapacity { .. }) => {
                        assert!(!called.load(Ordering::Relaxed));
                        break;
                    }
                    Err(error) => return Err(error.into()),
                }
            }
            assert!(completed > 0 && completed < 100);
            assert!(checkpoint(&store)?.busy);
            assert_eq!(
                read.query_row("SELECT value FROM wal_fixture WHERE id=1", [], |r| r
                    .get::<_, Vec<u8>>(0))?,
                original
            );
            read.commit()?;
            assert!(!checkpoint(&store)?.busy);
            assert_eq!(wal_bytes(&*store.writer()?)?, 0);
            write(&store, cap, 0, |tx| {
                tx.execute(
                    "UPDATE wal_fixture SET value=zeroblob(65536) WHERE id=1",
                    [],
                )?;
                Ok(())
            })?;
            assert!(wal_bytes(&*store.writer()?)? <= cap);
            assert_eq!(
                store
                    .writer()?
                    .query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
                "ok"
            );
            println!(
                "WAL reservation fixture: page_size={size}, pages={pages}, cap={cap}, reserved={reservation}, completed_before_pause={completed}, final_wal_bytes={}",
                wal_bytes(&*store.writer()?)?
            );
        }
        Ok(())
    }

    #[test]
    fn insufficient_reservations_and_failed_growth_preserve_data_and_connection_settings()
    -> Result<(), Box<dyn StdError>> {
        let directory = tempfile::tempdir()?;
        let store = store::open(directory.path().join("code.db"), StoreLimits::default())?;
        store.writer()?.execute_batch(
            "CREATE TABLE wal_fixture(value BLOB); INSERT INTO wal_fixture VALUES('committed')",
        )?;
        checkpoint(&store)?;
        let settings = || -> Result<(i64, i64, i64), Error> {
            let db = store.writer()?;
            Ok((
                db.query_row("PRAGMA cache_spill", [], |r| r.get(0))?,
                db.query_row("PRAGMA max_page_count", [], |r| r.get(0))?,
                db.query_row("PRAGMA busy_timeout", [], |r| r.get(0))?,
            ))
        };
        let before = settings()?;
        assert!(matches!(
            write(&store, 0, 0, |_| -> Result<(), Error> {
                panic!("unadmitted callback")
            }),
            Err(Error::WalCapacity { .. })
        ));
        assert_eq!(settings()?, before);
        let error = write(&store, 64 * 1024 * 1024, 0, |tx| {
            tx.execute("UPDATE wal_fixture SET value='changed'", [])?;
            tx.execute("INSERT INTO wal_fixture VALUES(zeroblob(16777216))", [])?;
            Ok(())
        })
        .unwrap_err();
        assert!(
            matches!(error, Error::Sqlite(ref error) if error.sqlite_error_code() == Some(rusqlite::ErrorCode::DiskFull))
        );
        assert_eq!(settings()?, before);
        assert_eq!(
            store
                .writer()?
                .query_row("SELECT value FROM wal_fixture", [], |r| r
                    .get::<_, String>(0))?,
            "committed"
        );
        assert_eq!(wal_bytes(&*store.writer()?)?, 0);
        let memory = store::open(":memory:", StoreLimits::default())?;
        assert!(matches!(
            write(&memory, 1024, 0, |_| Ok(())),
            Err(Error::WalReservationUnsupported)
        ));
        Ok(())
    }
}

#[cfg(all(test, unix))]
mod process_tests;

#[cfg(test)]
mod reserved_tests;

#[cfg(test)]
mod compact_tests;
