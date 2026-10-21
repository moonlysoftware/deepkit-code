//! Shared quota, uncertainty and post-lock deadline regressions; v1 §§8.2, 9.
use super::*;
use crate::store::{self, StoreLimits};
use std::{sync::atomic::AtomicUsize, time::Duration};

const BUDGET: usage::Budget = usage::Budget {
    calls: Some(5),
    tokens: Some(100),
};
struct Query {
    calls: AtomicUsize,
    rpm: i64,
    allowed: bool,
    pending: bool,
    fence: Option<Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>>,
    entered: AtomicBool,
}
#[async_trait]
impl Work for Query {
    type Output = i64;
    fn tokens(&self) -> i64 {
        10
    }
    fn endpoint(&self) -> Option<usage::Endpoint> {
        Some(usage::Endpoint {
            kind: "embed",
            url: "https://example.com/v1/embeddings".into(),
            units: 1,
        })
    }
    fn limits(&self, _: &Transaction<'_>, _: &str) -> Result<Option<usage::Limits>, queue::Error> {
        self.entered.store(true, Ordering::SeqCst);
        if let Some(fence) = &self.fence {
            let (lock, condition) = &**fence;
            let mut released = lock.lock().unwrap();
            while !*released {
                released = condition.wait(released).unwrap();
            }
        }
        Ok(self.allowed.then_some(usage::Limits {
            budget: BUDGET,
            rpm: Some(self.rpm),
            tpm: None,
        }))
    }
    async fn call(&self) -> Result<Response<i64>, AttemptError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.pending {
            std::future::pending::<()>().await;
        }
        Ok(Response {
            output: 42,
            tokens: Some(3),
        })
    }
}
fn query(rpm: i64, allowed: bool, pending: bool) -> Arc<Query> {
    Arc::new(Query {
        calls: AtomicUsize::new(0),
        rpm,
        allowed,
        pending,
        fence: None,
        entered: AtomicBool::new(false),
    })
}
fn store() -> Arc<Store> {
    Arc::new(store::open(":memory:", StoreLimits::default()).unwrap())
}
async fn run(store: &Arc<Store>, work: &Arc<Query>, ms: u64) -> Result<i64, Error> {
    execute(
        Arc::clone(store),
        "embed:test".into(),
        BUDGET,
        Instant::now() + Duration::from_millis(ms),
        Arc::clone(work),
        || 100,
    )
    .await
}
fn count(store: &Store, state: &str) -> i64 {
    store
        .writer()
        .unwrap()
        .query_row(
            "SELECT count(*) FROM usage_reservations WHERE state=?1",
            [state],
            |r| r.get(0),
        )
        .unwrap()
}

#[tokio::test]
async fn settled_queries_share_minute_limits_with_queue_dispatches() {
    let store = store();
    let work = query(1, true, false);
    assert_eq!(run(&store, &work, 2000).await.unwrap(), 42);
    assert_eq!(count(&store, "reconciled"), 1);
    let tokens: i64 = store
        .writer()
        .unwrap()
        .query_row("SELECT tokens FROM usage", [], |r| r.get(0))
        .unwrap();
    assert_eq!(tokens, 3);
    let usage::Admission::Reserved(held) =
        usage::reserve(&store, "embed:test", 10, 0, BUDGET, None, || 101).unwrap()
    else {
        panic!("daily budget should admit");
    };
    let dispatch = usage::mark_dispatched(
        &store,
        &held,
        None,
        || 101,
        |_, _| {
            Ok(usage::Limits {
                budget: BUDGET,
                rpm: Some(1),
                tpm: None,
            })
        },
    )
    .unwrap();
    assert!(matches!(dispatch, usage::Dispatch::RateLimited { .. }));
    usage::release(&store, &held).unwrap();
    assert!(matches!(
        run(&store, &work, 2000).await,
        Err(Error::Refused(usage::Dispatch::RateLimited { .. }))
    ));
    assert_eq!(work.calls.load(Ordering::SeqCst), 1);
    assert_eq!(count(&store, "reserved"), 0);
}

#[tokio::test]
async fn policy_refusal_releases_unsent_and_timeout_retains_uncertain_dispatch() {
    let store = store();
    let denied = query(10, false, false);
    assert!(matches!(
        run(&store, &denied, 2000).await,
        Err(Error::Refused(usage::Dispatch::Superseded))
    ));
    assert_eq!(denied.calls.load(Ordering::SeqCst), 0);
    assert_eq!(count(&store, "reserved"), 0);
    let pending = query(10, true, true);
    assert!(matches!(
        run(&store, &pending, 80).await,
        Err(Error::Deadline)
    ));
    assert_eq!(pending.calls.load(Ordering::SeqCst), 1);
    assert_eq!(count(&store, "dispatched"), 1);
    let metadata: (String, i64) = store
        .writer()
        .unwrap()
        .query_row(
            "SELECT endpoint_url,sent_units FROM usage_reservations WHERE state='dispatched'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(metadata, ("https://example.com/v1/embeddings".into(), 1));
}

#[tokio::test]
async fn deadline_while_waiting_for_writer_never_dispatches_after_unlock() {
    let store = store();
    let work = query(10, true, false);
    let db = Arc::clone(&store);
    let (ready, acquired) = tokio::sync::oneshot::channel();
    let (release, released) = std::sync::mpsc::channel();
    let holder = tokio::task::spawn_blocking(move || {
        let _lock = db.writer().unwrap();
        ready.send(()).unwrap();
        released.recv().unwrap();
    });
    acquired.await.unwrap();
    // A current-thread runtime proves the blocked writer does not block the timer.
    assert!(matches!(run(&store, &work, 50).await, Err(Error::Deadline)));
    release.send(()).unwrap();
    holder.await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(work.calls.load(Ordering::SeqCst), 0);
    assert_eq!(count(&store, "reserved"), 0);
    assert_eq!(count(&store, "dispatched"), 0);
}

#[tokio::test]
async fn abort_during_policy_resolution_releases_the_owned_reservation() {
    let store = store();
    let fence = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
    let work = Arc::new(Query {
        calls: AtomicUsize::new(0),
        rpm: 10,
        allowed: true,
        pending: false,
        fence: Some(Arc::clone(&fence)),
        entered: AtomicBool::new(false),
    });
    let db = Arc::clone(&store);
    let prepared = Arc::clone(&work);
    let task = tokio::spawn(async move { run(&db, &prepared, 5000).await });
    tokio::time::timeout(Duration::from_secs(2), async {
        while !work.entered.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    *fence.0.lock().unwrap() = true;
    fence.1.notify_all();
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let db = Arc::clone(&store);
            if tokio::task::spawn_blocking(move || count(&db, "released"))
                .await
                .unwrap()
                == 1
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(work.calls.load(Ordering::SeqCst), 0);
    assert_eq!(count(&store, "dispatched"), 0);
    assert_eq!(count(&store, "reserved"), 0);
}

fn file_store() -> (tempfile::TempDir, Arc<Store>) {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(
        store::open(
            dir.path().join("code.db"),
            StoreLimits {
                wal_max_bytes: 8 * 1024 * 1024,
                ..Default::default()
            },
        )
        .unwrap(),
    );
    store.writer().unwrap().execute_batch("CREATE TABLE pressure(id INTEGER PRIMARY KEY,data BLOB); INSERT INTO pressure VALUES(1,zeroblob(65536))").unwrap();
    store::wal::checkpoint(&store).unwrap();
    (dir, store)
}
fn pin_and_fill(store: &Store) -> rusqlite::Connection {
    let path = store.writer().unwrap().path().unwrap().to_owned();
    let reader =
        rusqlite::Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .unwrap();
    reader
        .execute_batch("BEGIN; SELECT length(data) FROM pressure")
        .unwrap();
    for _ in 0..100 {
        let result: Result<(), store::Error> =
            store::admission::transaction(store, store::admission::Kind::Growth, |tx| {
                tx.execute("UPDATE pressure SET data=randomblob(65536)", [])?;
                tx.commit()?;
                Ok(())
            });
        match result {
            Ok(()) => {}
            Err(store::Error::WalCapacity { .. }) => return reader,
            Err(error) => panic!("unexpected fixture failure: {error}"),
        }
    }
    panic!("fixture did not reach WAL backpressure")
}
#[test]
fn rolled_back_reserve_and_dispatch_recover_after_pinned_reader_finishes() {
    for deny in [false, true] {
        let (_dir, store) = file_store();
        let mut reader = Some(pin_and_fill(&store));
        let stopped = AtomicBool::new(false);
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut tries = 0;
        let reserved = admit_with_backpressure(&stopped, deadline, || {
            tries += 1;
            let result = usage::reserve(&store, "embed:test", 10, 0, BUDGET, None, || 100);
            if tries == 1 {
                assert!(matches!(
                    result,
                    Err(queue::Error::Store(store::Error::WalCapacity { .. }))
                ));
                assert_eq!(count(&store, "reserved"), 0);
                reader.take().unwrap().execute_batch("COMMIT").unwrap();
            }
            result
        })
        .unwrap();
        assert_eq!(tries, 2);
        let usage::Admission::Reserved(reservation) = reserved else {
            panic!("reservation missing")
        };
        let mut reader = Some(pin_and_fill(&store));
        let mut tries = 0;
        let mut policy_checks = 0;
        let mut allowed = true;
        let dispatch = admit_with_backpressure(&stopped, deadline, || {
            tries += 1;
            let result = usage::mark_dispatched_checked(
                &store,
                &reservation,
                None,
                || 100,
                |_, _| {
                    policy_checks += 1;
                    Ok(allowed.then_some(usage::Limits {
                        budget: BUDGET,
                        rpm: None,
                        tpm: None,
                    }))
                },
            );
            if tries == 1 {
                assert!(matches!(
                    result,
                    Err(queue::Error::Store(store::Error::WalCapacity { .. }))
                ));
                assert_eq!(count(&store, "reserved"), 1);
                allowed = !deny;
                reader.take().unwrap().execute_batch("COMMIT").unwrap();
            }
            result
        })
        .unwrap();
        assert_eq!(tries, 2);
        assert_eq!(policy_checks, 1);
        if deny {
            assert!(matches!(dispatch, usage::Dispatch::Superseded));
            assert_eq!(count(&store, "released"), 1)
        } else {
            assert!(matches!(dispatch, usage::Dispatch::Recorded));
            assert_eq!(count(&store, "dispatched"), 1)
        }
        assert!(store::wal::wal_bytes(&store.writer().unwrap()).unwrap() <= store.wal_max_bytes);
    }
}
#[tokio::test]
async fn persistent_reader_pressure_stays_bounded_visible_and_unsent() {
    let (_dir, store) = file_store();
    let reader = pin_and_fill(&store);
    let mut tries = 0;
    let result = admit_with_backpressure(
        &AtomicBool::new(false),
        Instant::now() + Duration::from_secs(2),
        || {
            tries += 1;
            usage::reserve(&store, "embed:test", 10, 0, BUDGET, None, || 100)
        },
    );
    assert!(matches!(
        result,
        Err(Error::Accounting(queue::Error::Store(
            store::Error::WalCapacity { .. }
        )))
    ));
    assert_eq!(tries, 8);
    assert_eq!(count(&store, "reserved"), 0);
    let work = query(10, true, false);
    assert!(matches!(
        run(&store, &work, 2000).await,
        Err(Error::Accounting(queue::Error::Store(
            store::Error::WalCapacity { .. }
        )))
    ));
    assert_eq!(work.calls.load(Ordering::SeqCst), 0);
    assert_eq!(count(&store, "reserved"), 0);
    assert!(store::wal::wal_bytes(&store.writer().unwrap()).unwrap() <= store.wal_max_bytes);
    let stopped = AtomicBool::new(false);
    let mut cancelled_tries = 0;
    let result = admit_with_backpressure(&stopped, Instant::now() + Duration::from_secs(2), || {
        cancelled_tries += 1;
        let result = usage::reserve(&store, "embed:test", 10, 0, BUDGET, None, || 100);
        assert!(matches!(
            result,
            Err(queue::Error::Store(store::Error::WalCapacity { .. }))
        ));
        stopped.store(true, Ordering::SeqCst);
        reader.execute_batch("COMMIT").unwrap();
        result
    });
    assert!(matches!(result, Err(Error::Deadline)));
    assert_eq!(cancelled_tries, 1);
    assert_eq!(count(&store, "reserved"), 0);
    assert_eq!(run(&store, &work, 2000).await.unwrap(), 42);
    assert_eq!(work.calls.load(Ordering::SeqCst), 1);
    assert_eq!(count(&store, "reconciled"), 1);
}
struct OverflowDispatch {
    checks: AtomicUsize,
    calls: AtomicUsize,
}
#[async_trait]
impl Work for OverflowDispatch {
    type Output = i64;
    fn tokens(&self) -> i64 {
        10
    }
    fn endpoint(&self) -> Option<usage::Endpoint> {
        None
    }
    fn limits(&self, tx: &Transaction<'_>, _: &str) -> Result<Option<usage::Limits>, queue::Error> {
        if self.checks.fetch_add(1, Ordering::SeqCst) == 0 {
            tx.execute("UPDATE pressure SET data=zeroblob(5242880)", [])?;
        }
        Ok(Some(usage::Limits {
            budget: BUDGET,
            rpm: None,
            tpm: None,
        }))
    }
    async fn call(&self) -> Result<Response<i64>, AttemptError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(Response {
            output: 42,
            tokens: Some(3),
        })
    }
}
#[tokio::test]
async fn dispatch_rollback_retries_sql_without_replaying_model_or_reserving_twice() {
    let (_dir, store) = file_store();
    let work = Arc::new(OverflowDispatch {
        checks: AtomicUsize::new(0),
        calls: AtomicUsize::new(0),
    });
    assert_eq!(
        execute(
            Arc::clone(&store),
            "embed:test".into(),
            BUDGET,
            Instant::now() + Duration::from_secs(2),
            Arc::clone(&work),
            || 100
        )
        .await
        .unwrap(),
        42
    );
    assert_eq!(work.checks.load(Ordering::SeqCst), 2);
    assert_eq!(work.calls.load(Ordering::SeqCst), 1);
    assert_eq!(count(&store, "reconciled"), 1);
    assert_eq!(count(&store, "reserved"), 0);
    let size: i64 = store
        .writer()
        .unwrap()
        .query_row("SELECT length(data) FROM pressure", [], |r| r.get(0))
        .unwrap();
    assert_eq!(size, 65536);
    assert!(store::wal::wal_bytes(&store.writer().unwrap()).unwrap() <= store.wal_max_bytes);
}
