//! One admitted worker attempt with renewal, deadlines and fenced publication; v1 §8.
use super::{Completion, CurrentInput, Error, Lease, complete, renew_with_clock, retry, usage};
use crate::{models::attempt::AttemptError, store::Store};
use async_trait::async_trait;
use rusqlite::Transaction;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

/// The provider result plus optional reported input/output token usage.
pub(crate) struct Response<T> {
    pub(crate) output: T,
    pub(crate) tokens: Option<i64>,
}

/// A prepared immutable job. Implementations resolve authoritative source, context,
/// configuration and outbound policy through the supplied transaction (None denies
/// publication/dispatch). They never acquire the writer or perform HTTP in resolvers.
/// Actual canonical input construction and L2/L3 writers belong to the pipeline.
#[async_trait]
pub(crate) trait Work: Send + Sync {
    type Output: Send + 'static;
    /// Returns current desired inputs only when source and outbound policy permit work.
    fn current(&self, tx: &Transaction<'_>, lease: &Lease) -> Result<Option<CurrentInput>, Error>;
    /// Revalidates with the worker shutdown fence inside expensive source traversal.
    fn current_with_work(
        &self,
        tx: &Transaction<'_>,
        lease: &Lease,
        work: crate::sync::l1::Work<'_>,
    ) -> Result<Option<CurrentInput>, Error> {
        super::source::check_work(work)?;
        self.current(tx, lease)
    }
    /// Revalidates with the worker clock so capacity and lease expiry use the same time.
    fn current_at(
        &self,
        tx: &Transaction<'_>,
        lease: &Lease,
        work: crate::sync::l1::Work<'_>,
        _now: i64,
    ) -> Result<Option<CurrentInput>, Error> {
        self.current_with_work(tx, lease, work)
    }
    /// Reads current effective rate and daily limits in the admission transaction.
    fn limits(&self, tx: &Transaction<'_>, provider: &str) -> Result<usage::Limits, Error>;
    /// Describes this attempt without source text. None means local or unavailable
    /// historical metadata, never permission to infer an endpoint from current config.
    fn outbound(&self) -> Result<Option<usage::Endpoint>, Error> {
        Ok(None)
    }
    /// Overrides the legacy host ceiling with this immutable job's prepared
    /// output bound. None preserves existing embedding and direct-worker behavior.
    fn max_output_tokens(&self) -> Option<i64> {
        None
    }
    /// Performs exactly one provider attempt; no retries, split batches or redirects.
    /// Dropping the future must stop further local work and discard its result.
    /// Non-interruptible native work must retain its capacity/resources until it
    /// exits; replacement attempts must not overlap it or queue blocked threads.
    async fn call(&self) -> Result<Response<Self::Output>, AttemptError>;
    /// Writes results using the completion transaction without acquiring the writer.
    fn publish(&self, tx: &Transaction<'_>, output: Self::Output) -> Result<(), Error>;
    /// Carries shutdown checks into expensive completion-side derived writes.
    fn publish_with_work(
        &self,
        tx: &Transaction<'_>,
        output: Self::Output,
        work: crate::sync::l1::Work<'_>,
    ) -> Result<(), Error> {
        super::source::check_work(work)?;
        self.publish(tx, output)
    }
}

/// Bounded settings for one already-claimed job, not a concurrency scheduler.
pub(crate) struct Settings<'a> {
    pub(crate) provider: &'a str,
    pub(crate) budget: usage::Budget,
    pub(crate) max_output_tokens: i64,
    pub(crate) timeout: Duration,
    pub(crate) heartbeat: Duration,
}

/// A worker returns after publication or durable scheduling; it never waits for quota.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Finished {
    Completed(Completion),
    Scheduled(retry::Scheduled),
    LeaseLost,
}

struct Context<W, C> {
    store: Arc<Store>,
    lease: Lease,
    work: W,
    clock: C,
    stopped: Arc<AtomicBool>,
}
impl<W, C: Fn() -> i64> Context<W, C> {
    fn now(&self) -> i64 {
        if self.stopped.load(Ordering::SeqCst) {
            -1
        } else {
            (self.clock)()
        }
    }
}
struct CancelOnDrop(Arc<AtomicBool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}
fn spawn_database<W, C, T>(
    context: &Arc<Context<W, C>>,
    operation: impl FnOnce(&Context<W, C>) -> Result<T, Error> + Send + 'static,
) -> tokio::task::JoinHandle<Result<T, Error>>
where
    W: Send + Sync + 'static,
    C: Send + Sync + 'static,
    T: Send + 'static,
{
    let context = Arc::clone(context);
    tokio::task::spawn_blocking(move || {
        if context.stopped.load(Ordering::SeqCst) {
            return Err(Error::WorkerStopped);
        }
        operation(&context)
    })
}
async fn database<W, C, T>(
    context: &Arc<Context<W, C>>,
    operation: impl FnOnce(&Context<W, C>) -> Result<T, Error> + Send + 'static,
) -> Result<T, Error>
where
    W: Send + Sync + 'static,
    C: Send + Sync + 'static,
    T: Send + 'static,
{
    spawn_database(context, operation)
        .await
        .map_err(|_| Error::WorkerTask)?
}

/// Runs one provider attempt while SQLite waits execute on blocking threads.
/// Prepared work must own its data (cheap Arc-backed clones are recommended).
/// Cancellation cannot forcibly stop a running SQLite operation: the cancellation
/// clock rejects writes after lock acquisition, and all original lease/input fences
/// remain enforced. Dispatched charges survive cancellation conservatively.
pub(crate) async fn execute<W, C>(
    store: &Arc<Store>,
    lease: &Lease,
    settings: &Settings<'_>,
    work: &W,
    clock: C,
) -> Result<Finished, Error>
where
    W: Work + Clone + 'static,
    C: Fn() -> i64 + Send + Sync + 'static,
{
    if settings.timeout.is_zero()
        || settings.heartbeat.is_zero()
        || settings.heartbeat >= Duration::from_secs(120)
        || tokio::time::Instant::now()
            .checked_add(settings.timeout)
            .is_none()
    {
        return Err(Error::Invalid("worker deadline or heartbeat"));
    }
    let stopped = Arc::new(AtomicBool::new(false));
    let _cancel = CancelOnDrop(Arc::clone(&stopped));
    let context = Arc::new(Context {
        store: Arc::clone(store),
        lease: lease.clone(),
        work: work.clone(),
        clock,
        stopped,
    });
    let provider = settings.provider.to_owned();
    let output_tokens = work
        .max_output_tokens()
        .unwrap_or(settings.max_output_tokens);
    let budget = settings.budget;
    let reservation = match database(&context, move |c| {
        usage::reserve(
            &c.store,
            &provider,
            c.lease.tokens,
            output_tokens,
            budget,
            Some(&c.lease),
            || c.now(),
        )
    })
    .await?
    {
        usage::Admission::Reserved(r) => r,
        usage::Admission::LeaseLost => return Ok(Finished::LeaseLost),
        usage::Admission::BudgetExhausted { retry_at } => {
            return Ok(Finished::Scheduled(
                database(&context, move |c| {
                    retry::schedule(
                        &c.store,
                        &c.lease,
                        retry::Outcome::Budget { retry_at },
                        || c.now(),
                    )
                })
                .await?,
            ));
        }
    };
    let reservation = Arc::new(reservation);
    let held = Arc::clone(&reservation);
    let admitted = database(&context, move |c| {
        usage::mark_dispatched_checked(
            &c.store,
            &held,
            Some(&c.lease),
            || c.now(),
            |tx, provider| {
                if !c
                    .work
                    .current_at(
                        tx,
                        &c.lease,
                        crate::sync::l1::Work {
                            deadline: None,
                            cancelled: Some(&c.stopped),
                        },
                        c.now(),
                    )?
                    .is_some_and(|input| {
                        input.target_hash.as_slice() == c.lease.target_hash
                            && input.desired_hash.as_slice() == c.lease.desired_hash
                    })
                {
                    return Ok(None);
                }
                let limits = c.work.limits(tx, provider)?;
                if let Some(endpoint) = c.work.outbound()? {
                    usage::record_endpoint(tx, &held, &endpoint)?;
                }
                Ok(Some(limits))
            },
        )
    })
    .await?;
    let defer = match admitted {
        usage::Dispatch::Recorded => None,
        usage::Dispatch::Superseded => return Ok(Finished::Completed(Completion::Discarded)),
        usage::Dispatch::LeaseLost | usage::Dispatch::Unavailable => {
            return Ok(Finished::LeaseLost);
        }
        usage::Dispatch::BudgetExhausted { retry_at } => Some(retry::Outcome::Budget { retry_at }),
        usage::Dispatch::RateLimited { retry_at } => Some(retry::Outcome::Rate { retry_at }),
        usage::Dispatch::AttemptsExhausted | usage::Dispatch::RequestTooLarge => {
            Some(retry::Outcome::Permanent)
        }
        usage::Dispatch::DayExpired => Some(retry::Outcome::Rate {
            retry_at: context
                .now()
                .checked_add(1)
                .ok_or(Error::Invalid("worker retry timestamp"))?,
        }),
    };
    if let Some(outcome) = defer {
        return Ok(Finished::Scheduled(
            database(&context, move |c| {
                retry::schedule(&c.store, &c.lease, outcome, || c.now())
            })
            .await?,
        ));
    }

    let renewal_stopped = Arc::new(AtomicBool::new(false));
    let renewal_guard = CancelOnDrop(Arc::clone(&renewal_stopped));
    let response = {
        let request = context.work.call();
        tokio::pin!(request);
        let deadline = tokio::time::sleep(settings.timeout);
        tokio::pin!(deadline);
        let heartbeat = tokio::time::sleep(settings.heartbeat);
        tokio::pin!(heartbeat);
        let mut renewal: Option<tokio::task::JoinHandle<Result<bool, Error>>> = None;
        loop {
            tokio::select! {
                biased;
                _=&mut deadline=>break Err(AttemptError::Retryable{status:None,retry_after:None}),
                result=async {match renewal.as_mut() {Some(task)=>task.await,None=>std::future::pending().await}}=> {
                    renewal=None;
                    if !result.map_err(|_|Error::WorkerTask)?? {return Ok(Finished::LeaseLost);}
                    heartbeat.as_mut().reset(tokio::time::Instant::now()+settings.heartbeat);
                },
                _=&mut heartbeat, if renewal.is_none()=> {
                    let stopped=Arc::clone(&renewal_stopped);
                    renewal=Some(spawn_database(&context,move|c|renew_with_clock(&c.store,&c.lease,||if stopped.load(Ordering::SeqCst) {-1} else {c.now()})));
                },
                result=&mut request=>break result,
            }
        }
    }; // Drop the HTTP future before waiting for retry/settlement database locks.
    drop(renewal_guard);
    match response {
        Err(error) => Ok(Finished::Scheduled(
            database(&context, move |c| {
                if let Some(tokens) = error.reported_tokens() {
                    usage::reconcile(&c.store, &reservation, Some(tokens))?;
                }
                retry::schedule(
                    &c.store,
                    &c.lease,
                    retry::Outcome::from_provider_attempt(&error),
                    || c.now(),
                )
            })
            .await?,
        )),
        Ok(response) => {
            // Payload-free boundary for diagnosing response-to-publication crashes.
            tracing::trace!(job_id = context.lease.id, "queue provider response decoded");
            database(&context, move |c| {
                usage::reconcile(&c.store, &reservation, response.tokens)?;
                // A response can arrive just before the next heartbeat. Renew
                // the still-valid lease before bounded local publication, whose
                // writer transaction prevents background heartbeats. Completion
                // still checks expiry after lock acquisition and after writing.
                if !renew_with_clock(&c.store, &c.lease, || c.now())? {
                    return Ok(Finished::LeaseLost);
                }
                Ok(Finished::Completed(complete(
                    &c.store,
                    &c.lease,
                    || c.now(),
                    |tx, lease| {
                        c.work.current_at(
                            tx,
                            lease,
                            crate::sync::l1::Work {
                                deadline: None,
                                cancelled: Some(&c.stopped),
                            },
                            c.now(),
                        )
                    },
                    |tx| {
                        c.work.publish_with_work(
                            tx,
                            response.output,
                            crate::sync::l1::Work {
                                deadline: None,
                                cancelled: Some(&c.stopped),
                            },
                        )
                    },
                )?))
            })
            .await
        }
    }
}

#[cfg(test)]
mod tests;
