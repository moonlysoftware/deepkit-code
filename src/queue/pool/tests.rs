//! Provider-pass concurrency, deferral and cancellation regressions for v1 §8.
use super::*;
use crate::{
    models::attempt::AttemptError,
    queue::{self, CurrentInput, worker::Work},
    store::{self, StoreLimits},
};
use async_trait::async_trait;
use rusqlite::{Transaction, params};
use std::sync::atomic::{AtomicI64, AtomicUsize};

#[derive(Default)]
struct Observed {
    active: AtomicUsize,
    peak: AtomicUsize,
    calls: AtomicUsize,
    prepared: AtomicUsize,
    clock: AtomicI64,
}
struct Build {
    seen: Arc<Observed>,
    mode: u8,
}
#[derive(Clone)]
struct Job {
    seen: Arc<Observed>,
    id: i64,
    mode: u8,
}
struct Active(Arc<Observed>);
impl Drop for Active {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::SeqCst);
    }
}
impl Factory for Build {
    type Job = Job;
    fn prepare(&self, _: &Store, lease: &Lease) -> Result<Job, Error> {
        self.seen.prepared.fetch_add(1, Ordering::SeqCst);
        if self.mode == 3 && lease.id == 1 {
            return Err(Error::Invalid("injected preparation error"));
        }
        Ok(Job {
            seen: Arc::clone(&self.seen),
            id: lease.id,
            mode: self.mode,
        })
    }
    fn prepare_with_work(
        &self,
        store: &Store,
        lease: &Lease,
        _: crate::sync::l1::Work<'_>,
    ) -> Result<Job, Error> {
        if self.mode != 5 {
            return self.prepare(store, lease);
        }
        self.seen.prepared.fetch_add(1, Ordering::SeqCst);
        self.seen.clock.store(119, Ordering::SeqCst);
        while store.writer()?.query_row(
            "SELECT leased_until FROM jobs WHERE id=?1",
            [lease.id],
            |row| row.get::<_, i64>(0),
        )? < 239
        {
            std::thread::sleep(Duration::from_millis(1));
        }
        self.seen.clock.store(121, Ordering::SeqCst);
        Ok(Job {
            seen: Arc::clone(&self.seen),
            id: lease.id,
            mode: self.mode,
        })
    }
}
#[async_trait]
impl Work for Job {
    type Output = i64;
    fn current(&self, _: &Transaction<'_>, _: &Lease) -> Result<Option<CurrentInput>, Error> {
        Ok(Some(CurrentInput {
            target_hash: [1; 32],
            desired_hash: [2; 32],
        }))
    }
    fn limits(&self, _: &Transaction<'_>, _: &str) -> Result<usage::Limits, Error> {
        Ok(usage::Limits {
            budget: usage::Budget {
                calls: None,
                tokens: None,
            },
            rpm: None,
            tpm: None,
        })
    }
    async fn call(&self) -> Result<worker::Response<i64>, AttemptError> {
        let active = self.seen.active.fetch_add(1, Ordering::SeqCst) + 1;
        let _active = Active(Arc::clone(&self.seen));
        self.seen.peak.fetch_max(active, Ordering::SeqCst);
        self.seen.calls.fetch_add(1, Ordering::SeqCst);
        if self.mode == 1 {
            return std::future::pending().await;
        }
        if self.mode == 0 {
            // First pair must overlap, proving the runner actually uses both slots.
            while self.seen.calls.load(Ordering::SeqCst) < 2 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        if self.mode == 2 && self.id == 1 {
            return Err(AttemptError::Retryable {
                status: Some(429),
                retry_after: Some(Duration::from_secs(60)),
            });
        }
        if self.mode == 2 && self.id == 2 {
            return Err(AttemptError::Rejected { status: 401 });
        }
        Ok(worker::Response {
            output: self.id,
            tokens: Some(1),
        })
    }
    fn publish(&self, tx: &Transaction<'_>, id: i64) -> Result<(), Error> {
        tx.execute("INSERT INTO output VALUES(?1)", [id])?;
        Ok(())
    }
}
fn setup(count: i64) -> Arc<Store> {
    let store = store::open(":memory:", StoreLimits::default()).unwrap();
    store
        .writer()
        .unwrap()
        .execute_batch("CREATE TABLE output(id INTEGER PRIMARY KEY)")
        .unwrap();
    for id in 1..=count {
        queue::enqueue(
            &store,
            &queue::Enqueue {
                kind: queue::Kind::Embed,
                target: queue::Target::Unit,
                target_id: id,
                target_hash: &[1; 32],
                desired_hash: &[2; 32],
                provider: "embed:test",
                priority: 1.0,
                tokens: 10,
            },
            0,
        )
        .unwrap();
    }
    Arc::new(store)
}
fn settings() -> Settings {
    Settings {
        provider: "embed:test".into(),
        concurrency: 2,
        max_jobs: 100,
        max_claim_tokens: 100,
        budget: usage::Budget {
            calls: None,
            tokens: None,
        },
        max_output_tokens: 0,
        timeout: Duration::from_secs(2),
        heartbeat: Duration::from_millis(20),
    }
}
fn states(store: &Store, state: &str) -> i64 {
    store
        .writer()
        .unwrap()
        .query_row("SELECT COUNT(*) FROM jobs WHERE state=?1", [state], |r| {
            r.get(0)
        })
        .unwrap()
}

#[tokio::test]
async fn provider_pass_uses_two_slots_and_respects_job_bound() {
    let store = setup(10);
    let seen = Arc::new(Observed::default());
    let factory = Arc::new(Build {
        seen: Arc::clone(&seen),
        mode: 0,
    });
    let mut cfg = settings();
    cfg.max_jobs = 6;
    cfg.max_claim_tokens = 10; // Fill both slots even when one claim batch fits only one job.
    let result = run_ready(Arc::clone(&store), Arc::new(cfg), factory, || 0)
        .await
        .unwrap();
    assert_eq!((result.claimed, result.published), (6, 6));
    assert!(result.errors.is_empty());
    assert_eq!(seen.peak.load(Ordering::SeqCst), 2);
    assert_eq!(seen.calls.load(Ordering::SeqCst), 6);
    assert_eq!(states(&store, "queued"), 4);
    assert_eq!(states(&store, "leased"), 0);
}

#[tokio::test]
async fn deferred_and_failed_jobs_do_not_stall_other_eligible_work() {
    for mode in [2, 3] {
        let store = setup(5);
        let seen = Arc::new(Observed::default());
        let result = run_ready(
            Arc::clone(&store),
            Arc::new(settings()),
            Arc::new(Build { seen, mode }),
            || 0,
        )
        .await
        .unwrap();
        assert_eq!(result.claimed, 5);
        if mode == 2 {
            assert_eq!((result.published, result.deferred, result.dead), (3, 1, 1));
            assert!(result.errors.is_empty());
            assert_eq!(states(&store, "failed"), 1);
        } else {
            assert_eq!(result.published, 4);
            assert_eq!(result.errors.len(), 1);
            assert_eq!(result.errors[0].0, 1);
            assert_eq!(states(&store, "leased"), 1);
        }
    }
}

#[tokio::test]
async fn cancellation_aborts_active_jobs_without_prefetching_waiting_leases() {
    let store = setup(10);
    let seen = Arc::new(Observed::default());
    let mut run = Box::pin(run_ready(
        Arc::clone(&store),
        Arc::new(settings()),
        Arc::new(Build {
            seen: Arc::clone(&seen),
            mode: 1,
        }),
        || 0,
    ));
    tokio::time::timeout(Duration::from_secs(1),async {
        tokio::select! {result=&mut run=>panic!("pool ended early: {result:?}"), _=async {while seen.active.load(Ordering::SeqCst)<2 {tokio::time::sleep(Duration::from_millis(1)).await;}}=>{}}
    }).await.unwrap();
    assert_eq!(states(&store, "leased"), 2);
    assert_eq!(states(&store, "queued"), 8);
    drop(run);
    tokio::time::timeout(Duration::from_secs(1), async {
        while seen.active.load(Ordering::SeqCst) > 0 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(seen.calls.load(Ordering::SeqCst), 2);
    assert_eq!(seen.prepared.load(Ordering::SeqCst), 2);
    let charged: i64 = store
        .writer()
        .unwrap()
        .query_row(
            "SELECT COUNT(*) FROM usage_reservations WHERE state='dispatched'",
            params![],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(charged, 2);
}

#[tokio::test]
async fn empty_wrong_provider_and_invalid_settings_never_prepare_work() {
    let store = setup(2);
    let seen = Arc::new(Observed::default());
    let factory = Arc::new(Build {
        seen: Arc::clone(&seen),
        mode: 0,
    });
    let mut cfg = settings();
    cfg.provider = "other".into();
    assert_eq!(
        run_ready(
            Arc::clone(&store),
            Arc::new(cfg),
            Arc::clone(&factory),
            || 0
        )
        .await
        .unwrap()
        .claimed,
        0
    );
    let mut cfg = settings();
    cfg.concurrency = 0;
    assert!(
        run_ready(Arc::clone(&store), Arc::new(cfg), factory, || 0)
            .await
            .is_err()
    );
    assert_eq!(seen.prepared.load(Ordering::SeqCst), 0);
    assert_eq!(states(&store, "queued"), 2);
}

#[tokio::test]
async fn preparation_renews_a_live_lease_before_input_building_finishes() {
    let store = setup(1);
    let seen = Arc::new(Observed::default());
    let clock = Arc::clone(&seen);
    let result = tokio::time::timeout(
        Duration::from_secs(1),
        run_ready(
            Arc::clone(&store),
            Arc::new(settings()),
            Arc::new(Build {
                seen: Arc::clone(&seen),
                mode: 5,
            }),
            move || clock.clock.load(Ordering::SeqCst),
        ),
    )
    .await
    .expect("preparation heartbeat must renew the lease")
    .unwrap();
    assert_eq!(
        (result.claimed, result.published, result.lease_lost),
        (1, 1, 0)
    );
    assert_eq!(seen.calls.load(Ordering::SeqCst), 1);
    assert_eq!(states(&store, "done"), 1);
}

#[tokio::test]
async fn independent_runners_share_claims_without_duplicate_publication() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let first = Arc::new(store::open(file.path(), StoreLimits::default()).unwrap());
    first
        .writer()
        .unwrap()
        .execute_batch("CREATE TABLE output(id INTEGER PRIMARY KEY)")
        .unwrap();
    for id in 1..=7 {
        queue::enqueue(
            &first,
            &queue::Enqueue {
                kind: queue::Kind::Embed,
                target: queue::Target::Unit,
                target_id: id,
                target_hash: &[1; 32],
                desired_hash: &[2; 32],
                provider: "embed:test",
                priority: 1.0,
                tokens: 10,
            },
            0,
        )
        .unwrap();
    }
    let second = Arc::new(store::open(file.path(), StoreLimits::default()).unwrap());
    let factory = Arc::new(Build {
        seen: Arc::new(Observed::default()),
        mode: 4,
    });
    let (a, b) = tokio::join!(
        run_ready(
            Arc::clone(&first),
            Arc::new(settings()),
            Arc::clone(&factory),
            || 0
        ),
        run_ready(second, Arc::new(settings()), Arc::clone(&factory), || 0)
    );
    let (a, b) = (a.unwrap(), b.unwrap());
    assert_eq!(a.claimed + b.claimed, 7);
    assert_eq!(a.published + b.published, 7);
    assert!(a.errors.is_empty() && b.errors.is_empty());
    assert_eq!(states(&first, "done"), 7);
    assert_eq!(factory.seen.calls.load(Ordering::SeqCst), 7);
}

#[test]
fn delayed_claim_samples_current_time_and_can_be_cancelled_before_lock_acquisition() {
    use std::sync::atomic::AtomicI64;
    for final_time in [120, -1] {
        let file = tempfile::NamedTempFile::new().unwrap();
        let first = store::open(file.path(), StoreLimits::default()).unwrap();
        let second = store::open(file.path(), StoreLimits::default()).unwrap();
        queue::enqueue(
            &first,
            &queue::Enqueue {
                kind: queue::Kind::Embed,
                target: queue::Target::Unit,
                target_id: 1,
                target_hash: &[1; 32],
                desired_hash: &[2; 32],
                provider: "embed:test",
                priority: 1.0,
                tokens: 10,
            },
            0,
        )
        .unwrap();
        let owner = Owner::new(&first).unwrap();
        let now = AtomicI64::new(0);
        let started = std::sync::Barrier::new(2);
        let mut writer = first.writer().unwrap();
        let tx = writer
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .unwrap();
        std::thread::scope(|scope| {
            let claim = scope.spawn(|| {
                started.wait();
                claim_with_clock(
                    &second,
                    "embed:test",
                    &owner,
                    || now.load(Ordering::SeqCst),
                    1,
                    10,
                )
            });
            started.wait();
            std::thread::sleep(Duration::from_millis(20));
            now.store(final_time, Ordering::SeqCst);
            tx.commit().unwrap();
            let result = claim.join().unwrap();
            if final_time < 0 {
                assert!(result.is_err());
            } else {
                assert_eq!(result.unwrap().len(), 1);
            }
        });
        drop(writer);
        if final_time < 0 {
            assert_eq!(states(&first, "queued"), 1);
        } else {
            let until: i64 = first
                .writer()
                .unwrap()
                .query_row("SELECT leased_until FROM jobs", [], |r| r.get(0))
                .unwrap();
            assert_eq!(until, 240);
        }
    }
}
