//! Cooperative writer admission for budgeted L1 repair; v1 §3.1/§4.6.

use super::{Error, Store};
use rusqlite::{Connection, ErrorCode, Transaction, TransactionBehavior};
use std::{sync::TryLockError, thread, time::Duration};

const POLL: Duration = Duration::from_millis(1);

/// Acquires the local writer cooperatively and disables SQLite's blocking busy
/// timeout for this operation only. The callback executes at most once and must
/// retain WAL/main-cap admission; this helper alone does not authorize writes.
/// Ordinary writer users keep their configured timeout. Callbacks must not panic
/// or change the busy handler, and must release transactions before returning.
pub(crate) fn with_writer<T, E>(
    store: &Store,
    check: &(impl Fn() -> Result<(), Error> + ?Sized),
    action: impl FnOnce(&mut Connection) -> Result<T, E>,
) -> Result<T, E>
where
    E: From<Error> + From<rusqlite::Error>,
{
    let mut writer = loop {
        check()?;
        match store.writer.try_lock() {
            Ok(writer) => break writer,
            Err(TryLockError::WouldBlock) => thread::sleep(POLL),
            Err(TryLockError::Poisoned(_)) => {
                return Err(Error::LockPoisoned {
                    connection: "writer",
                }
                .into());
            }
        }
    };
    check()?;
    let timeout: i64 = writer.query_row("PRAGMA busy_timeout", [], |row| row.get(0))?;
    writer.busy_timeout(Duration::ZERO)?;
    let result = (|| {
        check()?;
        action(&mut writer)
    })();
    // Restore even when admission or the callback refused. A panic poisons the
    // mutex, so the connection cannot silently return to ordinary writer use.
    writer.pragma_update(None, "busy_timeout", timeout)?;
    result
}

/// Retries only BEGIN IMMEDIATE contention, checking caller work between attempts
/// and after admission. Use inside with_writer so SQLite itself never sleeps for
/// its ordinary timeout. No mutation callback is replayed; late refusal drops the
/// transaction before returning. Subsequent work still needs precommit checks.
pub(crate) fn begin<'a>(
    writer: &'a mut Connection,
    check: &(impl Fn() -> Result<(), Error> + ?Sized),
) -> Result<Transaction<'a>, Error> {
    loop {
        check()?;
        // Runtime transaction checks allow retrying BEGIN against one borrowed
        // connection; no transaction escapes until admission succeeds.
        match Transaction::new_unchecked(writer, TransactionBehavior::Immediate) {
            Ok(transaction) => {
                check()?;
                return Ok(transaction);
            }
            Err(rusqlite::Error::SqliteFailure(error, _))
                if error.code == ErrorCode::DatabaseBusy =>
            {
                thread::sleep(POLL);
            }
            Err(error) => return Err(error.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, sync::mpsc, time::Instant};

    fn before(deadline: Instant) -> impl Fn() -> Result<(), Error> {
        move || {
            if Instant::now() >= deadline {
                Err(Error::L1InputsChanged)
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn held_local_writer_stops_before_owner_releases_and_retry_succeeds() {
        let store =
            super::super::open(std::path::Path::new(":memory:"), Default::default()).unwrap();
        let held = store.writer().unwrap();
        thread::scope(|scope| {
            let (send, receive) = mpsc::channel();
            let store = &store;
            scope.spawn(move || {
                let result: Result<(), Error> = with_writer(
                    store,
                    &before(Instant::now() + Duration::from_millis(30)),
                    |_| panic!("expired waiter must not run"),
                );
                send.send(matches!(result, Err(Error::L1InputsChanged)))
                    .unwrap();
            });
            let stopped = receive.recv_timeout(Duration::from_secs(2));
            drop(held);
            assert!(stopped.unwrap());
        });
        with_writer(&store, &|| Ok(()), |writer| -> Result<(), Error> {
            begin(writer, &|| Ok(()))?.commit()?;
            Ok(())
        })
        .unwrap();
        assert_eq!(
            store
                .writer()
                .unwrap()
                .query_row("PRAGMA busy_timeout", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            5000
        );
    }

    #[test]
    fn competing_sqlite_writer_times_out_without_mutation_and_restores_policy() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("code.db");
        let store = super::super::open(&path, Default::default()).unwrap();
        let mut other = Connection::open(&path).unwrap();
        let held = other
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .unwrap();
        let started = Instant::now();
        let check = before(started + Duration::from_millis(30));
        let called = Cell::new(false);
        let result: Result<(), Error> = with_writer(&store, &check, |writer| {
            let tx = begin(writer, &check)?;
            called.set(true);
            tx.execute("INSERT INTO meta VALUES ('checked_writer_test', 'new')", [])?;
            tx.commit()?;
            Ok(())
        });
        assert!(matches!(result, Err(Error::L1InputsChanged)));
        assert!(!called.get());
        assert!(started.elapsed() < Duration::from_secs(2));
        assert_eq!(
            store
                .writer()
                .unwrap()
                .query_row("PRAGMA busy_timeout", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            5000
        );
        held.rollback().unwrap();
        with_writer(&store, &|| Ok(()), |writer| -> Result<(), Error> {
            let tx = begin(writer, &|| Ok(()))?;
            tx.execute("INSERT INTO meta VALUES ('checked_writer_test', 'new')", [])?;
            tx.commit()?;
            Ok(())
        })
        .unwrap();
        assert_eq!(
            store.get_meta("checked_writer_test").unwrap().as_deref(),
            Some("new")
        );
    }

    #[test]
    fn local_writer_cancellation_is_observed_while_owner_still_holds_lock() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let store =
            super::super::open(std::path::Path::new(":memory:"), Default::default()).unwrap();
        let held = store.writer().unwrap();
        let cancelled = AtomicBool::new(false);
        thread::scope(|scope| {
            let (entered, waiting) = mpsc::channel();
            let (done, stopped) = mpsc::channel();
            let store = &store;
            let cancelled = &cancelled;
            scope.spawn(move || {
                let announced = Cell::new(false);
                let result: Result<(), Error> = with_writer(
                    store,
                    &|| {
                        if !announced.replace(true) {
                            entered.send(()).unwrap();
                        }
                        if cancelled.load(Ordering::Acquire) {
                            Err(Error::L1InputsChanged)
                        } else {
                            Ok(())
                        }
                    },
                    |_| panic!("cancelled writer must not run"),
                );
                done.send(matches!(result, Err(Error::L1InputsChanged)))
                    .unwrap();
            });
            let ready = waiting.recv_timeout(Duration::from_secs(2));
            cancelled.store(true, Ordering::Release);
            let result = stopped.recv_timeout(Duration::from_secs(2));
            drop(held);
            ready.unwrap();
            assert!(result.unwrap());
        });
    }

    #[test]
    fn cancellation_after_admission_rolls_back_and_callback_errors_restore_timeout() {
        let store =
            super::super::open(std::path::Path::new(":memory:"), Default::default()).unwrap();
        let checks = Cell::new(0);
        let result: Result<(), Error> = with_writer(&store, &|| Ok(()), |writer| {
            let _tx = begin(writer, &|| {
                checks.set(checks.get() + 1);
                if checks.get() == 2 {
                    Err(Error::L1InputsChanged)
                } else {
                    Ok(())
                }
            })?;
            panic!("post-BEGIN refusal must prevent publication");
        });
        assert!(matches!(result, Err(Error::L1InputsChanged)));
        assert!(store.writer().unwrap().is_autocommit());
        let result: Result<(), Error> = with_writer(&store, &|| Ok(()), |writer| {
            let tx = begin(writer, &|| Ok(()))?;
            tx.execute("INSERT INTO meta VALUES ('rolled_back_test', 'new')", [])?;
            Err(Error::L1InputsChanged)
        });
        assert!(matches!(result, Err(Error::L1InputsChanged)));
        assert!(store.get_meta("rolled_back_test").unwrap().is_none());
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
