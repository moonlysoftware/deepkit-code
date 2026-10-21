//! Shared writer WAL admission and control headroom; v1 §3.5/§8.2.
use super::Error;
use crate::store::{Store, wal};
use rusqlite::{Connection, Transaction, TransactionBehavior};
use serde::Serialize;

/// Growth/publication leaves headroom for existing lease and billing updates.
#[derive(Clone, Copy)]
pub(crate) enum Kind {
    Growth,
    /// Whole catalogue/file publication cannot be split without losing fences.
    L1Growth,
    /// Catalogue/maintenance may use main control space but retain WAL headroom.
    Catalogue,
    /// A branch catalogue is indivisible and may reserve remaining growth space.
    AtomicCatalogue,
    /// Atomic schema rewrites need more room than ordinary file publication.
    Migration,
    Control,
}

/// Observed allocation and admission status; inspection never checkpoints.
#[derive(Debug, Serialize)]
pub(crate) struct Status {
    allocated_bytes: u64,
    limit_bytes: u64,
    control_headroom_bytes: u64,
    growth_paused: bool,
    l1_growth_paused: bool,
    control_paused: bool,
}

impl Status {
    /// Distinguishes write backpressure from job/provider errors without mutation.
    pub(crate) fn summary(&self) -> String {
        format!(
            "WAL: {} / {} bytes; growth {}, L1 growth {}, control {} (checkpoint may need readers to finish)",
            self.allocated_bytes,
            self.limit_bytes,
            if self.growth_paused {
                "paused"
            } else {
                "admissible"
            },
            if self.l1_growth_paused {
                "paused"
            } else {
                "admissible"
            },
            if self.control_paused {
                "paused"
            } else {
                "admissible"
            }
        )
    }
}

fn allowance(cap: u64, kind: Kind) -> (u64, u64) {
    // These are internal transaction bounds, not new configuration defaults.
    // Native enforcement rolls back work too large for an allowance.
    let headroom = (cap / 4).min(1024 * 1024);
    match kind {
        Kind::Growth => {
            let ceiling = cap - headroom;
            (ceiling, (ceiling / 2).min(4 * 1024 * 1024))
        }
        Kind::L1Growth | Kind::Catalogue => {
            let ceiling = cap - headroom;
            (ceiling, ceiling / 2)
        }
        Kind::Migration | Kind::AtomicCatalogue => (cap - headroom, cap - headroom),
        Kind::Control => (cap, headroom),
    }
}

/// Starts ordinary PASSIVE checkpointing before growth admission runs out of
/// room. One bounded growth transaction may cross the watermark; retain another
/// reservation after that commit. Pinned readers still enforce normal cap refusal.
pub(super) fn autocheckpoint_frames(wal_cap: u64, page_size: u32) -> u32 {
    let (ceiling, reserve) = allowance(wal_cap, Kind::Growth);
    let watermark = ceiling.saturating_sub(reserve.saturating_mul(2));
    let frames = watermark.saturating_sub(32) / (u64::from(page_size) + 24);
    // Preserve SQLite's default frequency when its 1000-frame watermark is safe.
    // Zero disables automatic checkpoints, so tiny caps must use at least one.
    frames.clamp(1, 1000) as u32
}

/// Compacts without enclosing SQLite's own VACUUM transaction in BEGIN, while
/// retaining the same WAL control headroom as atomic schema rewrites.
pub(crate) fn compact(store: &Store) -> Result<(), Error> {
    wal::compact(store, allowance(store.wal_max_bytes, Kind::Migration).0)
}

/// Reports file-backed writer backpressure even when there is no room to persist
/// a pause record. Memory-only test databases have no WAL admission policy.
pub(crate) fn status(db: &Connection, store: &Store) -> Result<Option<Status>, Error> {
    if db.path().is_none_or(str::is_empty) {
        return Ok(None);
    }
    let allocated_bytes = wal::wal_bytes(db)?;
    let size: u32 = db.query_row("PRAGMA page_size", [], |r| r.get(0))?;
    let blocked = |kind| {
        let (cap, reserve) = allowance(store.wal_max_bytes, kind);
        reserve < u64::from(size) + 56 || allocated_bytes.saturating_add(reserve) > cap
    };
    Ok(Some(Status {
        allocated_bytes,
        limit_bytes: store.wal_max_bytes,
        control_headroom_bytes: allowance(store.wal_max_bytes, Kind::Control).1,
        growth_paused: blocked(Kind::Growth),
        l1_growth_paused: blocked(Kind::L1Growth),
        control_paused: blocked(Kind::Control),
    }))
}

/// Reserves one bounded writer transaction, trying a nonwaiting checkpoint only
/// when allocation would prevent admission. The exact reservation is checked
/// again under the cross-process writer lock; callbacks execute at most once.
/// Callers own commit decisions and must propagate every write failure. A failed
/// control write retains the lease/uncertain charge for subsequent recovery.
pub(crate) fn transaction<T, E>(
    store: &Store,
    kind: Kind,
    action: impl FnOnce(Transaction<'_>) -> Result<T, E>,
) -> Result<T, E>
where
    E: From<Error> + From<rusqlite::Error>,
{
    transaction_checked(store, kind, None, action)
}

/// Uses a single admitted writer for observation, checkpoint and reservation.
/// An explicit cheap work signal enables cooperative lock waits; None preserves
/// ordinary indexing/queue busy policy. Mutation callbacks are never replayed.
pub(crate) fn transaction_checked<T, E>(
    store: &Store,
    kind: Kind,
    check: Option<&dyn Fn() -> Result<(), Error>>,
    action: impl FnOnce(Transaction<'_>) -> Result<T, E>,
) -> Result<T, E>
where
    E: From<Error> + From<rusqlite::Error>,
{
    let run = |writer: &mut Connection| transaction_in(store, writer, kind, check, action);
    match check {
        Some(check) => super::checked_writer::with_writer(store, check, run),
        None => run(&mut *store.writer()?),
    }
}

fn transaction_in<T, E>(
    store: &Store,
    writer: &mut Connection,
    kind: Kind,
    check: Option<&dyn Fn() -> Result<(), Error>>,
    action: impl FnOnce(Transaction<'_>) -> Result<T, E>,
) -> Result<T, E>
where
    E: From<Error> + From<rusqlite::Error>,
{
    let action = |transaction: Transaction<'_>| {
        let cap = match kind {
            Kind::Growth | Kind::L1Growth => super::capacity::growth_limit(store.db_max_bytes),
            Kind::Catalogue | Kind::AtomicCatalogue | Kind::Migration | Kind::Control => {
                store.db_max_bytes
            }
        };
        super::capacity::limit_main_growth(&transaction, cap)?;
        action(transaction)
    };
    if let Some(check) = check {
        check()?;
    }
    let observed = status(writer, store)?;
    let Some(observed) = observed else {
        return action(match check {
            Some(check) => super::checked_writer::begin(writer, check)?,
            None => writer.transaction_with_behavior(TransactionBehavior::Immediate)?,
        });
    };
    if matches!(kind, Kind::Migration | Kind::AtomicCatalogue) {
        // VACUUM-like schema rewrites cannot be divided into smaller committed
        // pieces. Reclaim what readers permit, then reserve available growth
        // space under the lock rather than racing an observed allocation.
        wal::checkpoint_in(writer)?;
        return wal::transaction_in(
            writer,
            allowance(store.wal_max_bytes, kind).0,
            None,
            None,
            check,
            action,
        );
    }
    if match kind {
        Kind::Growth => observed.growth_paused,
        Kind::L1Growth | Kind::Catalogue | Kind::AtomicCatalogue | Kind::Migration => {
            let (cap, reserve) = allowance(store.wal_max_bytes, kind);
            observed.growth_paused || observed.allocated_bytes.saturating_add(reserve) > cap
        }
        Kind::Control => observed.control_paused,
    } {
        let (cap, reserve) = allowance(store.wal_max_bytes, kind);
        // Serving readers can briefly pin an otherwise reclaimable WAL. Give
        // checked work a bounded, cancellable admission window before refusing;
        // no transaction or mutation callback has started, so nothing is replayed.
        let started = std::time::Instant::now();
        loop {
            if let Some(check) = check {
                check()?;
            }
            let checkpoint = wal::checkpoint_in(writer)?;
            if !checkpoint.busy
                || check.is_none()
                || wal::wal_bytes(writer)?.saturating_add(reserve) <= cap
                || started.elapsed() >= std::time::Duration::from_millis(250)
            {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
    let (cap, reserve) = allowance(store.wal_max_bytes, kind);
    wal::transaction_in(writer, cap, Some(reserve), None, check, action)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{
        FileKind, FileUpsert, GraphFileRank, GraphSymbolRank, StoreLimits, SymbolInsert, open,
        register_vector_extension,
    };
    use std::error::Error as StdError;

    fn deadline_check(deadline: std::time::Instant) -> impl Fn() -> Result<(), Error> {
        move || {
            if std::time::Instant::now() >= deadline {
                Err(Error::L1InputsChanged)
            } else {
                Ok(())
            }
        }
    }

    fn policy(store: &Store) -> (i64, i64, String) {
        let writer = store.writer().unwrap();
        (
            writer
                .query_row("PRAGMA busy_timeout", [], |r| r.get(0))
                .unwrap(),
            writer
                .query_row("PRAGMA cache_spill", [], |r| r.get(0))
                .unwrap(),
            writer
                .query_row("PRAGMA dkc_wal_limit", [], |r| r.get(0))
                .unwrap(),
        )
    }

    #[test]
    fn checked_admission_stops_at_local_writer_and_retries_without_replay() {
        use std::{
            sync::mpsc,
            thread,
            time::{Duration, Instant},
        };
        for file_backed in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let path = if file_backed {
                dir.path().join("code.db")
            } else {
                ":memory:".into()
            };
            let store = open(&path, Default::default()).unwrap();
            let held = store.writer().unwrap();
            thread::scope(|scope| {
                let (send, receive) = mpsc::channel();
                let store = &store;
                scope.spawn(move || {
                    let check = deadline_check(Instant::now() + Duration::from_millis(30));
                    let result: Result<(), Error> =
                        transaction_checked(store, Kind::Catalogue, Some(&check), |_| {
                            panic!("refused action")
                        });
                    send.send(matches!(result, Err(Error::L1InputsChanged)))
                        .unwrap();
                });
                let result = receive.recv_timeout(Duration::from_secs(2));
                drop(held);
                assert!(result.unwrap());
            });
            let calls = std::cell::Cell::new(0);
            transaction_checked(
                &store,
                Kind::Catalogue,
                Some(&|| Ok(())),
                |tx| -> Result<(), Error> {
                    calls.set(calls.get() + 1);
                    tx.execute("INSERT INTO meta VALUES('admission_test', 'published')", [])?;
                    tx.commit()?;
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(calls.get(), 1);
            assert_eq!(
                store.get_meta("admission_test").unwrap().as_deref(),
                Some("published")
            );
            assert_eq!(
                store
                    .writer()
                    .unwrap()
                    .query_row("PRAGMA busy_timeout", [], |r| r.get::<_, i64>(0))
                    .unwrap(),
                5000
            );
        }
    }

    #[test]
    fn checked_wal_admission_restores_policy_on_contention_overflow_and_refusal() {
        use std::time::{Duration, Instant};
        for page_size in [4096, 65536] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("code.db");
            register_vector_extension().unwrap();
            let db = Connection::open(&path).unwrap();
            db.pragma_update(None, "page_size", page_size).unwrap();
            db.execute_batch("VACUUM").unwrap();
            drop(db);
            let store = open(
                &path,
                StoreLimits {
                    wal_max_bytes: 8 * 1024 * 1024,
                    ..Default::default()
                },
            )
            .unwrap();
            let original = policy(&store);
            let mut other = Connection::open(&path).unwrap();
            let held = other
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .unwrap();
            let start = Instant::now();
            let check = deadline_check(start + Duration::from_millis(30));
            let result: Result<(), Error> =
                transaction_checked(&store, Kind::Catalogue, Some(&check), |_| {
                    panic!("contended action")
                });
            assert!(matches!(result, Err(Error::L1InputsChanged)));
            assert!(start.elapsed() < Duration::from_secs(2));
            assert_eq!(policy(&store), original);
            held.rollback().unwrap();

            // Dirty work is rolled back on a late source/work refusal; native
            // overflow also rolls back without replaying the mutation callback.
            for overflow in [false, true] {
                wal::checkpoint(&store).unwrap();
                let calls = std::cell::Cell::new(0);
                let result: Result<(), Error> =
                    transaction_checked(&store, Kind::L1Growth, Some(&|| Ok(())), |tx| {
                        calls.set(calls.get() + 1);
                        tx.execute("INSERT INTO meta VALUES('admission_test', 'temporary')", [])?;
                        if overflow {
                            tx.execute(
                                "INSERT INTO meta VALUES('admission_big', zeroblob(16777216))",
                                [],
                            )?;
                            tx.commit()?;
                            Ok(())
                        } else {
                            Err(Error::L1InputsChanged)
                        }
                    });
                if overflow {
                    assert!(matches!(result, Err(Error::WalCapacity { .. })));
                } else {
                    assert!(matches!(result, Err(Error::L1InputsChanged)));
                }
                assert_eq!(calls.get(), 1);
                assert_eq!(policy(&store), original);
                assert!(store.get_meta("admission_test").unwrap().is_none());
                assert!(store.get_meta("admission_big").unwrap().is_none());
                assert!(wal::wal_bytes(&store.writer().unwrap()).unwrap() <= store.wal_max_bytes);
            }
            transaction_checked(
                &store,
                Kind::Control,
                Some(&|| Ok(())),
                |tx| -> Result<(), Error> {
                    tx.execute("INSERT INTO meta VALUES('admission_test', 'retry')", [])?;
                    tx.commit()?;
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(policy(&store), original);
            assert_eq!(
                store.get_meta("admission_test").unwrap().as_deref(),
                Some("retry")
            );
        }
    }

    #[test]
    fn checked_checkpoint_refusal_keeps_reader_and_policy_then_resumes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("code.db");
        let mut store = open(&path, Default::default()).unwrap();
        store.set_meta("admission_test", "before").unwrap();
        wal::checkpoint(&store).unwrap();
        let mut reader = Connection::open(&path).unwrap();
        let read = reader.transaction().unwrap();
        let read_value = || {
            read.query_row(
                "SELECT value FROM meta WHERE key='admission_test'",
                [],
                |r| r.get::<_, String>(0),
            )
            .unwrap()
        };
        assert_eq!(read_value(), "before");
        transaction(&store, Kind::L1Growth, |tx| -> Result<(), Error> {
            tx.execute(
                "INSERT INTO meta VALUES('large_checkpoint_test', zeroblob(2097152))",
                [],
            )?;
            tx.commit()?;
            Ok(())
        })
        .unwrap();
        store.wal_max_bytes = 1024 * 1024;
        let original = policy(&store);
        let check = deadline_check(std::time::Instant::now() + std::time::Duration::from_secs(1));
        let result: Result<(), Error> =
            transaction_checked(&store, Kind::Catalogue, Some(&check), |_| {
                panic!("pinned WAL must refuse before callback")
            });
        assert!(matches!(result, Err(Error::WalCapacity { .. })));
        assert_eq!(policy(&store), original);
        assert_eq!(read_value(), "before");
        read.rollback().unwrap();
        transaction_checked(
            &store,
            Kind::Catalogue,
            Some(&|| Ok(())),
            |tx| -> Result<(), Error> {
                tx.execute(
                    "UPDATE meta SET value='after' WHERE key='admission_test'",
                    [],
                )?;
                tx.commit()?;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(policy(&store), original);
        assert_eq!(
            store.get_meta("admission_test").unwrap().as_deref(),
            Some("after")
        );
        assert!(wal::wal_bytes(&store.writer().unwrap()).unwrap() <= store.wal_max_bytes);
    }

    #[test]
    fn standalone_graph_and_catalogue_failures_preserve_committed_state()
    -> Result<(), Box<dyn StdError>> {
        for page_size in [4096, 65536] {
            for operation in ["rank", "symbols", "rename", "delete", "metadata", "rebuild"] {
                let dir = tempfile::tempdir()?;
                let path = dir.path().join("code.db");
                register_vector_extension()?;
                {
                    let db = Connection::open(&path)?;
                    db.pragma_update(None, "page_size", page_size)?;
                    db.execute_batch("VACUUM")?;
                }
                let store = open(
                    &path,
                    StoreLimits {
                        wal_max_bytes: 8 * 1024 * 1024,
                        ..Default::default()
                    },
                )?;
                let file_id = store.upsert_file(FileUpsert {
                    path: "src/original.rs",
                    blake3: &[1; 32],
                    size: 10,
                    mtime_ns: 1,
                    lang: Some("rust"),
                    kind: FileKind::Code,
                    git_blob: None,
                    max_level: 2,
                    treatment: "default",
                    indexed_at: Some(1),
                    needs_vlm: false,
                })?;
                let symbols = store.replace_file_symbols(
                    file_id,
                    &[SymbolInsert {
                        unit_id: None,
                        name: "original",
                        qualified: Some("original"),
                        kind: "function",
                        signature: Some("fn original()"),
                        line_start: 1,
                        line_end: 1,
                    }],
                )?;
                let original = store.graph_snapshot()?;
                store.set_meta("test_marker", "before")?;
                let generation = store.l1_generation()?;
                let rank_symbols = [GraphSymbolRank {
                    symbol_id: symbols[0].id,
                    rank: 0.7,
                }];
                let rank_files = [GraphFileRank { file_id, rank: 0.7 }];
                let run = || -> Result<(), Error> {
                    match operation {
                        "rank" => store.update_graph_ranks(generation, &rank_symbols, &rank_files),
                        "symbols" => store.replace_file_symbols(file_id, &[]).map(|_| ()),
                        "rename" => store
                            .rename_file("src/original.rs", "src/new.rs")
                            .map(|_| ()),
                        "delete" => store.delete_file("src/original.rs"),
                        "metadata" => store.set_meta("test_marker", "after"),
                        _ => store.rebuild_fts(&[]),
                    }
                };
                // FTS virtual tables cannot have triggers: grow their postings directly
                // for rebuild, and use a base-table trigger for the other paths.
                if operation == "rebuild" {
                    let huge = (0..400_000)
                        .map(|i| format!("token{i} "))
                        .collect::<String>();
                    let row = crate::store::FtsRebuildRow {
                        unit_id: 1,
                        columns: crate::store::FtsColumns {
                            header: &huge,
                            idents: "",
                            body: "",
                            prose: "",
                        },
                    };
                    wal::checkpoint(&store)?;
                    assert!(matches!(
                        store.rebuild_fts(&[row]),
                        Err(Error::WalCapacity { .. })
                    ));
                } else {
                    let event = match operation {
                        "rank" | "rename" => "AFTER UPDATE ON files",
                        "symbols" => "AFTER DELETE ON symbols",
                        "delete" => "AFTER DELETE ON files",
                        _ => "AFTER UPDATE ON meta",
                    };
                    store.writer()?.execute_batch(&format!(
                        "CREATE TABLE payload(value BLOB); CREATE TRIGGER overflow {event} BEGIN INSERT INTO payload VALUES(zeroblob(16777216)); END;"
                    ))?;
                    wal::checkpoint(&store)?;
                    assert!(
                        matches!(run(), Err(Error::WalCapacity { .. })),
                        "{operation}"
                    );
                    assert_eq!(
                        store
                            .writer()?
                            .query_row("SELECT count(*) FROM payload", [], |r| r
                                .get::<_, i64>(0))?,
                        0
                    );
                    store.writer()?.execute_batch("DROP TRIGGER overflow")?;
                }
                assert_eq!(store.graph_snapshot()?, original, "{operation}");
                assert_eq!(store.get_meta("graph_pending")?.as_deref(), Some("1"));
                assert_eq!(store.get_meta("test_marker")?.as_deref(), Some("before"));
                assert!(!store.query_fts_names("original", 10)?.is_empty());
                assert!(wal::wal_bytes(&*store.writer()?)? <= 8 * 1024 * 1024);
                run()?;
                assert!(wal::wal_bytes(&*store.writer()?)? <= 8 * 1024 * 1024);
                if operation == "rank" {
                    assert_eq!(store.get_meta("graph_pending")?.as_deref(), Some("0"));
                    assert_eq!(store.graph_snapshot()?.symbols[0].pagerank, 0.7);
                    assert!(matches!(
                        store.update_graph_ranks(generation + 1, &rank_symbols, &rank_files),
                        Err(Error::L1ScanSuperseded { .. })
                    ));
                }
                assert_eq!(
                    store
                        .writer()?
                        .query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
                    "ok"
                );
            }
        }
        Ok(())
    }
    #[test]
    fn standalone_metadata_backpressure_resumes_after_reader_release()
    -> Result<(), Box<dyn StdError>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("code.db");
        let limits = StoreLimits {
            wal_max_bytes: 8 * 1024 * 1024,
            ..Default::default()
        };
        let first = open(&path, limits)?;
        let second = open(&path, limits)?;
        first.set_meta("test_marker", "initial")?;
        wal::checkpoint(&first)?;
        let reader = Connection::open(&path)?;
        reader.execute_batch("BEGIN")?;
        let read = || {
            reader.query_row("SELECT value FROM meta WHERE key='test_marker'", [], |r| {
                r.get::<_, String>(0)
            })
        };
        assert_eq!(read()?, "initial");
        let payload = "x".repeat(8192);
        let mut completed = 0;
        let mut blocked = false;
        for i in 0..4096 {
            let store = if i % 2 == 0 { &first } else { &second };
            match store.set_meta("test_marker", &format!("{i}{payload}")) {
                Ok(()) => completed += 1,
                Err(Error::WalCapacity { .. }) => {
                    blocked = true;
                    break;
                }
                Err(error) => return Err(error.into()),
            }
            assert!(wal::wal_bytes(&*store.writer()?)? <= limits.wal_max_bytes);
        }
        assert!(
            blocked && completed > 0,
            "blocked={blocked}, completed={completed}"
        );
        assert_eq!(read()?, "initial");
        assert_eq!(
            first.get_meta("test_marker")?,
            Some(format!("{}{payload}", completed - 1))
        );
        reader.execute_batch("ROLLBACK")?;
        second.set_meta("test_marker", "resumed")?;
        assert_eq!(first.get_meta("test_marker")?.as_deref(), Some("resumed"));
        assert!(wal::wal_bytes(&*first.writer()?)? <= limits.wal_max_bytes);
        Ok(())
    }

    #[test]
    fn checked_checkpoint_wait_releases_transient_reader_and_calls_action_once() {
        use std::cell::Cell;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("code.db");
        let mut store = open(&path, Default::default()).unwrap();
        store.set_meta("admission_test", "before").unwrap();
        wal::checkpoint(&store).unwrap();
        let reader = Connection::open(&path).unwrap();
        reader.execute_batch("BEGIN").unwrap();
        let before: String = reader
            .query_row(
                "SELECT value FROM meta WHERE key='admission_test'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(before, "before");
        transaction(&store, Kind::L1Growth, |tx| -> Result<(), Error> {
            tx.execute(
                "INSERT INTO meta VALUES('large_checkpoint_test', zeroblob(2097152))",
                [],
            )?;
            tx.commit()?;
            Ok(())
        })
        .unwrap();
        store.wal_max_bytes = 1024 * 1024;
        let original = policy(&store);
        let checks = Cell::new(0);
        let released = Cell::new(false);
        let calls = Cell::new(0);
        let check = || {
            checks.set(checks.get() + 1);
            // Release after repeated cooperative polls, beyond all initial writer
            // checks. Without checkpoint retries admission refuses before reaching
            // this point. No thread scheduling or guessed sleep enables release.
            if checks.get() == 16 {
                reader.execute_batch("ROLLBACK")?;
                released.set(true);
            }
            Ok(())
        };
        transaction_checked(
            &store,
            Kind::Catalogue,
            Some(&check),
            |tx| -> Result<(), Error> {
                calls.set(calls.get() + 1);
                assert!(released.get(), "callback must wait for reclaimable WAL");
                tx.execute(
                    "UPDATE meta SET value='after' WHERE key='admission_test'",
                    [],
                )?;
                tx.commit()?;
                Ok(())
            },
        )
        .unwrap();
        assert!(released.get());
        assert_eq!(calls.get(), 1);
        assert_eq!(policy(&store), original);
        assert_eq!(
            store.get_meta("admission_test").unwrap().as_deref(),
            Some("after")
        );
        assert!(wal::wal_bytes(&store.writer().unwrap()).unwrap() <= store.wal_max_bytes);
    }

    #[test]
    fn checked_checkpoint_wait_cancels_without_callback_or_policy_change() {
        use std::cell::Cell;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("code.db");
        let mut store = open(&path, Default::default()).unwrap();
        store.set_meta("admission_test", "before").unwrap();
        wal::checkpoint(&store).unwrap();
        let reader = Connection::open(&path).unwrap();
        reader.execute_batch("BEGIN").unwrap();
        let read = || {
            reader
                .query_row(
                    "SELECT value FROM meta WHERE key='admission_test'",
                    [],
                    |r| r.get::<_, String>(0),
                )
                .unwrap()
        };
        assert_eq!(read(), "before");
        transaction(&store, Kind::L1Growth, |tx| -> Result<(), Error> {
            tx.execute(
                "INSERT INTO meta VALUES('large_checkpoint_test', zeroblob(2097152))",
                [],
            )?;
            tx.commit()?;
            Ok(())
        })
        .unwrap();
        store.wal_max_bytes = 1024 * 1024;
        let original = policy(&store);
        let checks = Cell::new(0);
        let calls = Cell::new(0);
        let check = || {
            checks.set(checks.get() + 1);
            if checks.get() >= 16 {
                Err(Error::L1InputsChanged)
            } else {
                Ok(())
            }
        };
        let result: Result<(), Error> =
            transaction_checked(&store, Kind::Catalogue, Some(&check), |_| {
                calls.set(calls.get() + 1);
                Ok(())
            });
        assert!(matches!(result, Err(Error::L1InputsChanged)));
        assert_eq!(calls.get(), 0);
        assert_eq!(policy(&store), original);
        assert_eq!(read(), "before", "cancellation must preserve pinned reader");
        reader.execute_batch("ROLLBACK").unwrap();
        transaction_checked(
            &store,
            Kind::Catalogue,
            Some(&|| Ok(())),
            |tx| -> Result<(), Error> {
                tx.execute(
                    "UPDATE meta SET value='resumed' WHERE key='admission_test'",
                    [],
                )?;
                tx.commit()?;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(policy(&store), original);
        assert_eq!(
            store.get_meta("admission_test").unwrap().as_deref(),
            Some("resumed")
        );
        assert!(wal::wal_bytes(&store.writer().unwrap()).unwrap() <= store.wal_max_bytes);
    }
}

#[cfg(test)]
mod autocheckpoint_tests {
    use super::*;
    #[test]
    fn watermark_accounts_for_page_headers_and_two_growth_reservations() {
        assert_eq!(autocheckpoint_frames(64 * 1024 * 1024, 65536), 879);
        assert_eq!(autocheckpoint_frames(64 * 1024 * 1024, 4096), 1000);
        assert_eq!(autocheckpoint_frames(0, 65536), 1);
    }

    #[test]
    fn atomic_catalogue_uses_remaining_space_but_rolls_back_at_pinned_cap() {
        let directory = tempfile::tempdir().unwrap();
        let database = directory.path().join("code.db");
        let limits = super::super::StoreLimits {
            wal_max_bytes: 16 * 1024 * 1024,
            ..Default::default()
        };
        let store = super::super::open(&database, limits).unwrap();
        store.writer().unwrap().execute_batch(
            "CREATE TABLE catalogue_fixture(value BLOB); INSERT INTO catalogue_fixture VALUES(zeroblob(10485760));"
        ).unwrap();
        wal::checkpoint(&store).unwrap();
        let reader = Connection::open(&database).unwrap();
        reader.execute_batch("BEGIN").unwrap();
        let value = |db: &Connection| {
            db.query_row(
                "SELECT hex(substr(value,1,1)) FROM catalogue_fixture",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap()
        };
        assert_eq!(value(&reader), "00");
        let calls = std::cell::Cell::new(0);
        let update = |byte: u8| -> Result<(), Error> {
            transaction(&store, Kind::AtomicCatalogue, |tx| {
                calls.set(calls.get() + 1);
                tx.execute(
                    "UPDATE catalogue_fixture SET value=?1",
                    [vec![byte; 10 * 1024 * 1024]],
                )?;
                tx.commit()?;
                Ok(())
            })
        };
        update(1).unwrap();
        assert!(matches!(update(2), Err(Error::WalCapacity { .. })));
        assert_eq!(calls.get(), 2, "mutation is never replayed");
        assert_eq!(value(&store.writer().unwrap()), "01");
        assert_eq!(value(&reader), "00");
        assert!(
            wal::wal_bytes(&store.writer().unwrap()).unwrap()
                <= allowance(limits.wal_max_bytes, Kind::AtomicCatalogue).0
        );
        reader.execute_batch("ROLLBACK").unwrap();
        update(2).unwrap();
        assert_eq!(calls.get(), 3);
        assert_eq!(value(&store.writer().unwrap()), "02");
        assert!(
            wal::wal_bytes(&store.writer().unwrap()).unwrap()
                <= allowance(limits.wal_max_bytes, Kind::AtomicCatalogue).0
        );
    }

    #[test]
    fn normal_checkpoints_recycle_unpinned_wal_but_pinned_reader_still_refuses() {
        let directory = tempfile::tempdir().unwrap();
        let database = directory.path().join("code.db");
        let limits = super::super::StoreLimits {
            wal_max_bytes: 16 * 1024 * 1024,
            ..Default::default()
        };
        let store = super::super::open(&database, limits).unwrap();
        store.writer().unwrap().execute_batch(
            "CREATE TABLE checkpoint_fixture(value BLOB);             INSERT INTO checkpoint_fixture VALUES(zeroblob(1048576));"
        ).unwrap();
        wal::checkpoint(&store).unwrap();
        let (ceiling, reserve) = allowance(limits.wal_max_bytes, Kind::Growth);
        let update = |value: u8| {
            wal::write_reserved(&store, ceiling, reserve, 0, |tx| {
                tx.execute(
                    "UPDATE checkpoint_fixture SET value=?1",
                    [vec![value; 1048576]],
                )?;
                Ok(())
            })
        };
        // Bypass the admission helper's emergency TRUNCATE: only SQLite's normal
        // post-commit automatic checkpoint can keep these writes admissible.
        for value in 1..=80 {
            update(value).unwrap();
            let writer = store.writer().unwrap();
            assert!(wal::wal_bytes(&writer).unwrap() + reserve <= ceiling);
        }
        wal::checkpoint(&store).unwrap();
        let mut reader = Connection::open(&database).unwrap();
        let snapshot = reader.transaction().unwrap();
        let before: Vec<u8> = snapshot
            .query_row("SELECT value FROM checkpoint_fixture", [], |row| row.get(0))
            .unwrap();
        let mut refused = false;
        for value in 81..=160 {
            match update(value) {
                Ok(()) => {}
                Err(Error::WalCapacity { .. }) => {
                    refused = true;
                    break;
                }
                Err(error) => panic!("unexpected write failure: {error}"),
            }
        }
        assert!(
            refused,
            "automatic checkpoint must not bypass pinned-reader limits"
        );
        let retained: Vec<u8> = snapshot
            .query_row("SELECT value FROM checkpoint_fixture", [], |row| row.get(0))
            .unwrap();
        assert_eq!(retained, before);
        assert!(wal::wal_bytes(&store.writer().unwrap()).unwrap() <= ceiling);
        assert!(wal::checkpoint(&store).unwrap().busy);
        snapshot.commit().unwrap();
        assert!(!wal::checkpoint(&store).unwrap().busy);
        update(161).unwrap();
        let value: Vec<u8> = store
            .writer()
            .unwrap()
            .query_row("SELECT value FROM checkpoint_fixture", [], |row| row.get(0))
            .unwrap();
        assert_eq!(value, vec![161; 1048576]);
    }
}
