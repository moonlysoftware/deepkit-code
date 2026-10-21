//! Bounded provider passes over the persistent queue; v1 §8.2.
use super::{Completion, Error, Lease, Owner, claim_with_clock, retry, usage, worker};
use crate::store::Store;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tokio::task::JoinSet;

/// Builds one immutable job on a blocking thread. Preparation must be bounded,
/// avoid network calls, and leave current-input/policy validation to Work resolvers.
pub(crate) trait Factory: Send + Sync + 'static {
    type Job: worker::Work + Clone + 'static;
    /// Resolves claimed identity into canonical inputs and transaction-aware writers.
    fn prepare(&self, store: &Store, lease: &Lease) -> Result<Self::Job, Error>;
    /// Allows expensive input builders to stop when their owning pass is dropped.
    fn prepare_with_work(
        &self,
        store: &Store,
        lease: &Lease,
        work: crate::sync::l1::Work<'_>,
    ) -> Result<Self::Job, Error> {
        super::source::check_work(work)?;
        self.prepare(store, lease)
    }
}

/// Limits one provider runner in one process. The host starts one runner per
/// provider; shared SQLite admission remains the cross-process rate authority.
pub(crate) struct Settings {
    pub(crate) provider: String,
    pub(crate) concurrency: usize,
    pub(crate) max_jobs: usize,
    pub(crate) max_claim_tokens: i64,
    pub(crate) budget: usage::Budget,
    pub(crate) max_output_tokens: i64,
    pub(crate) timeout: Duration,
    pub(crate) heartbeat: Duration,
}

/// Counts a finite pass without confusing discarded work with published coverage.
#[derive(Debug, Default)]
pub(crate) struct Report {
    pub(crate) claimed: usize,
    pub(crate) published: usize,
    pub(crate) discarded: usize,
    pub(crate) deferred: usize,
    pub(crate) dead: usize,
    pub(crate) lease_lost: usize,
    pub(crate) errors: Vec<(i64, Error)>,
}
impl Report {
    pub(crate) fn record(&mut self, id: i64, result: Result<worker::Finished, Error>) {
        match result {
            Ok(worker::Finished::Completed(Completion::Published)) => self.published += 1,
            Ok(worker::Finished::Completed(Completion::Discarded)) => self.discarded += 1,
            Ok(worker::Finished::Scheduled(retry::Scheduled::RetryAt(_))) => self.deferred += 1,
            Ok(worker::Finished::Scheduled(retry::Scheduled::Dead)) => self.dead += 1,
            Ok(_) => self.lease_lost += 1,
            Err(error) => self.errors.push((id, error)),
        }
    }
}
struct Stop(Arc<AtomicBool>);
impl Drop for Stop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// Runs currently eligible jobs with no more claimed/active tasks than free slots.
/// Stops at max_jobs or when no eligible/running work remains; future retries are
/// left for a later pass. Dropping this future aborts async workers and fences queued
/// claims; already-claimed leases and dispatched charges recover normally. Job-level
/// failures are reported while other jobs finish; a task panic aborts the pass.
pub(crate) async fn run_ready<F, C>(
    store: Arc<Store>,
    settings: Arc<Settings>,
    factory: Arc<F>,
    clock: C,
) -> Result<Report, Error>
where
    F: Factory,
    C: Fn() -> i64 + Send + Sync + 'static,
{
    if settings.provider.is_empty()
        || settings.concurrency == 0
        || settings.max_jobs == 0
        || settings.max_claim_tokens < 0
        || settings.max_output_tokens < 0
        || settings.budget.calls.is_some_and(|v| v < 0)
        || settings.budget.tokens.is_some_and(|v| v < 0)
        || settings.timeout.is_zero()
        || settings.heartbeat.is_zero()
        || settings.heartbeat >= Duration::from_secs(120)
        || tokio::time::Instant::now()
            .checked_add(settings.timeout)
            .is_none()
    {
        return Err(Error::Invalid("provider pool settings"));
    }
    let clock = Arc::new(clock);
    let stopped = Arc::new(AtomicBool::new(false));
    let _stop = Stop(Arc::clone(&stopped));
    let owner = {
        let store = Arc::clone(&store);
        tokio::task::spawn_blocking(move || Owner::new(&store))
            .await
            .map_err(|_| Error::WorkerTask)??
    };
    let mut running = JoinSet::new();
    let mut report = Report::default();
    loop {
        while running.len() < settings.concurrency && report.claimed < settings.max_jobs {
            let available =
                (settings.concurrency - running.len()).min(settings.max_jobs - report.claimed);
            let batch = {
                let store = Arc::clone(&store);
                let settings = Arc::clone(&settings);
                let owner = owner.clone();
                let clock = Arc::clone(&clock);
                let stopped = Arc::clone(&stopped);
                tokio::task::spawn_blocking(move || {
                    claim_with_clock(
                        &store,
                        &settings.provider,
                        &owner,
                        || {
                            if stopped.load(Ordering::SeqCst) {
                                -1
                            } else {
                                clock()
                            }
                        },
                        available,
                        settings.max_claim_tokens,
                    )
                })
                .await
                .map_err(|_| Error::WorkerTask)??
            };
            if batch.is_empty() {
                break;
            }
            report.claimed += batch.len();
            for lease in batch {
                let store = Arc::clone(&store);
                let settings = Arc::clone(&settings);
                let factory = Arc::clone(&factory);
                let clock = Arc::clone(&clock);
                let stopped = Arc::clone(&stopped);
                running.spawn(async move {
                    let id = lease.id;
                    let result = async {
                        let Some(job) = prepare_with_renewal(
                            Arc::clone(&store),
                            Arc::clone(&factory),
                            lease.clone(),
                            Arc::clone(&clock),
                            Arc::clone(&stopped),
                            settings.heartbeat,
                        )
                        .await?
                        else {
                            return Ok(worker::Finished::LeaseLost);
                        };
                        if stopped.load(Ordering::SeqCst) {
                            return Err(Error::WorkerStopped);
                        }
                        let config = worker::Settings {
                            provider: &settings.provider,
                            budget: settings.budget,
                            max_output_tokens: settings.max_output_tokens,
                            timeout: settings.timeout,
                            heartbeat: settings.heartbeat,
                        };
                        let worker_clock = Arc::clone(&clock);
                        worker::execute(&store, &lease, &config, &job, move || worker_clock()).await
                    }
                    .await;
                    let deferred = result.as_ref().err().and_then(|error| {
                        if matches!(error, Error::InputsUnavailable) {
                            Some(retry::Outcome::InputsUnavailable)
                        } else if matches!(error, Error::InputPreparation) {
                            Some(retry::Outcome::InputPreparation)
                        } else {
                            retry::Outcome::from_capacity(error)
                        }
                    });
                    let result = if let Some(outcome) = deferred {
                        tokio::task::spawn_blocking(move || {
                            retry::schedule(&store, &lease, outcome, || {
                                if stopped.load(Ordering::SeqCst) {
                                    -1
                                } else {
                                    clock()
                                }
                            })
                            .map(worker::Finished::Scheduled)
                        })
                        .await
                        .map_err(|_| Error::WorkerTask)?
                    } else {
                        result
                    };
                    Ok::<_, Error>((id, result))
                });
            }
        }
        let Some(result) = running.join_next().await else {
            break;
        };
        let (id, result) = result.map_err(|_| Error::WorkerTask)??;
        report.record(id, result);
    }
    Ok(report)
}

/// Builds claimed inputs while extending the fenced lease at the configured
/// heartbeat. Some canonical source builders can take longer than one lease,
/// and preparation happens before the worker's request-time renewal loop.
pub(crate) async fn prepare_with_renewal<F, C>(
    store: Arc<Store>,
    factory: Arc<F>,
    lease: Lease,
    clock: Arc<C>,
    stopped: Arc<AtomicBool>,
    heartbeat: Duration,
) -> Result<Option<F::Job>, Error>
where
    F: Factory,
    C: Fn() -> i64 + Send + Sync + 'static,
{
    let mut preparation = Box::pin(tokio::task::spawn_blocking({
        let store = Arc::clone(&store);
        let lease = lease.clone();
        let stopped = Arc::clone(&stopped);
        move || {
            if stopped.load(Ordering::SeqCst) {
                return Err(Error::WorkerStopped);
            }
            factory.prepare_with_work(
                &store,
                &lease,
                crate::sync::l1::Work {
                    deadline: None,
                    cancelled: Some(&stopped),
                },
            )
        }
    }));
    let renewal_stopped = Arc::new(AtomicBool::new(false));
    let _renewal_guard = Stop(Arc::clone(&renewal_stopped));
    let heartbeat_interval = heartbeat;
    let heartbeat_at = tokio::time::Instant::now()
        .checked_add(heartbeat_interval)
        .ok_or(Error::Invalid("preparation heartbeat deadline"))?;
    let heartbeat = tokio::time::sleep_until(heartbeat_at);
    tokio::pin!(heartbeat);
    let mut renewal = None;
    loop {
        tokio::select! {
            biased;
            result = &mut preparation => {
                return result.map_err(|_| Error::WorkerTask)?.map(Some);
            }
            result = async { match renewal.as_mut() {
                Some(task) => task.await,
                None => std::future::pending::<Result<Result<bool, Error>, tokio::task::JoinError>>().await,
            }} => {
                renewal = None;
                if !result.map_err(|_| Error::WorkerTask)?? {
                    return Ok(None);
                }
                let next = tokio::time::Instant::now()
                    .checked_add(heartbeat_interval)
                    .ok_or(Error::Invalid("preparation heartbeat deadline"))?;
                heartbeat.as_mut().reset(next);
            }
            _ = &mut heartbeat, if renewal.is_none() => {
                let store = Arc::clone(&store);
                let lease = lease.clone();
                let clock = Arc::clone(&clock);
                let stopped = Arc::clone(&stopped);
                let renewal_stopped = Arc::clone(&renewal_stopped);
                renewal = Some(tokio::task::spawn_blocking(move || {
                    super::renew_with_clock(&store, &lease, || {
                        if stopped.load(Ordering::SeqCst) || renewal_stopped.load(Ordering::SeqCst) {
                            -1
                        } else {
                            clock()
                        }
                    })
                }));
            }
        }
    }
}

#[cfg(test)]
mod tests;
