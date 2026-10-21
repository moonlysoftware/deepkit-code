//! Deadline-bounded query attempts sharing durable provider admission; v1 §§8.2, 9.
#![expect(
    dead_code,
    reason = "dense orchestration connects query admission in T1.7"
)]

use crate::{
    models::attempt::AttemptError,
    queue::{self, usage, worker::Response},
    store::Store,
};
use async_trait::async_trait;
use rusqlite::Transaction;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::time::Instant;

/// Immutable prepared query input. Implementations must bind token estimates and
/// endpoint identity to the exact bytes sent, and resolve current policy in SQL.
#[async_trait]
pub(crate) trait Work: Send + Sync + 'static {
    type Output: Send + 'static;
    /// Input plus maximum output tokens held before this one attempt.
    fn tokens(&self) -> i64;
    /// Source-free destination metadata for potential-dispatch accounting; local
    /// in-process work returns None and remains in the shared usage ledger.
    fn endpoint(&self) -> Option<usage::Endpoint>;
    /// None denies outbound work after a configuration or policy change. Never
    /// acquire the writer again or perform network I/O inside this callback.
    fn limits(
        &self,
        tx: &Transaction<'_>,
        provider: &str,
    ) -> Result<Option<usage::Limits>, queue::Error>;
    /// Sends exactly one request with no implicit retries or redirects.
    async fn call(&self) -> Result<Response<Self::Output>, AttemptError>;
}

/// A refusal leaves the caller free to degrade to another retrieval leg.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("query deadline exceeded")]
    Deadline,
    #[error("query daily budget exhausted until {retry_at}")]
    Budget { retry_at: i64 },
    #[error("query dispatch refused: {0:?}")]
    Refused(usage::Dispatch),
    #[error(transparent)]
    Accounting(#[from] queue::Error),
    #[error(transparent)]
    Provider(#[from] AttemptError),
    #[error("query accounting task failed")]
    Task,
}

struct Stop(Arc<AtomicBool>);
impl Drop for Stop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

// Only rolled-back SQL admission is retried. These short waits let concurrent
// lexical snapshots finish, permitting the existing nonwaiting WAL checkpoint.
// No cap is raised, no provider call is replayed, and policy is checked each time.
fn admit_with_backpressure<T>(
    stopped: &AtomicBool,
    deadline: Instant,
    mut action: impl FnMut() -> Result<T, queue::Error>,
) -> Result<T, Error> {
    for attempt in 0..8 {
        if stopped.load(Ordering::SeqCst) || Instant::now() >= deadline {
            return Err(Error::Deadline);
        }
        let error = match action() {
            Ok(value) => return Ok(value),
            Err(error) => error,
        };
        if !matches!(
            error,
            queue::Error::Store(crate::store::Error::WalCapacity { .. })
        ) || attempt == 7
        {
            return Err(error.into());
        }
        let delay = std::time::Duration::from_millis(1 << attempt.min(4));
        if Instant::now()
            .checked_add(delay)
            .is_none_or(|next| next >= deadline)
        {
            return Err(error.into());
        }
        // This helper runs only on the owning blocking task. It retains any
        // reservation ID while waiting and rechecks cancellation before new SQL.
        std::thread::sleep(delay);
    }
    Err(Error::Deadline)
}

/// Executes one accounted attempt within an absolute end-to-end deadline. The
/// caller owns retry policy; every retry must enter this function again. SQLite
/// waits run off the runtime, with cancellation rechecked after lock acquisition.
/// Unsent reservations are released by their owning blocking task even if its
/// result is abandoned. Potential dispatches survive timeout conservatively.
pub(crate) async fn execute<W, C>(
    store: Arc<Store>,
    provider: String,
    budget: usage::Budget,
    deadline: Instant,
    work: Arc<W>,
    clock: C,
) -> Result<W::Output, Error>
where
    W: Work,
    C: Fn() -> i64 + Send + Sync + 'static,
{
    let stopped = Arc::new(AtomicBool::new(false));
    let _stop = Stop(Arc::clone(&stopped));
    let attempt = async {
        if Instant::now() >= deadline {
            return Err(Error::Deadline);
        }
        let db = Arc::clone(&store);
        let prepared = Arc::clone(&work);
        let cancelled = Arc::clone(&stopped);
        let reservation = tokio::task::spawn_blocking(move || {
            let now = || {
                if cancelled.load(Ordering::SeqCst) || Instant::now() >= deadline {
                    -1
                } else {
                    clock()
                }
            };
            let reservation = match admit_with_backpressure(&cancelled, deadline, || {
                usage::reserve(&db, &provider, prepared.tokens(), 0, budget, None, now)
            })? {
                usage::Admission::Reserved(r) => r,
                usage::Admission::BudgetExhausted { retry_at } => {
                    return Err(Error::Budget { retry_at });
                }
                usage::Admission::LeaseLost => {
                    return Err(Error::Refused(usage::Dispatch::LeaseLost));
                }
            };
            let dispatch = admit_with_backpressure(&cancelled, deadline, || {
                usage::mark_dispatched_checked(&db, &reservation, None, now, |tx, provider| {
                    let limits = prepared.limits(tx, provider)?;
                    if limits.is_some()
                        && let Some(endpoint) = prepared.endpoint()
                    {
                        usage::record_endpoint(tx, &reservation, &endpoint)?;
                    }
                    Ok(limits)
                })
            });
            match dispatch {
                Ok(usage::Dispatch::Recorded) => Ok(reservation),
                other => {
                    // This task owns the reservation across both transactions;
                    // cancellation cannot lose an undispatched reservation ID.
                    usage::release(&db, &reservation)?;
                    match other {
                        Ok(refused) => Err(Error::Refused(refused)),
                        Err(error) => Err(error),
                    }
                }
            }
        })
        .await
        .map_err(|_| Error::Task)??;
        // Dispatch may have committed immediately before cancellation/deadline.
        // Keep its uncertain charge but never start late HTTP.
        if stopped.load(Ordering::SeqCst) || Instant::now() >= deadline {
            return Err(Error::Deadline);
        }
        let response = work.call().await;
        if let Err(error) = response {
            if let Some(tokens) = error.reported_tokens() {
                tokio::task::spawn_blocking(move || {
                    usage::reconcile(&store, &reservation, Some(tokens))
                })
                .await
                .map_err(|_| Error::Task)??;
            }
            return Err(error.into());
        }
        let response = response?;
        // Once a valid result arrives, settlement may finish after cancellation.
        // It contains only usage and never publishes query results or starts I/O.
        tokio::task::spawn_blocking(move || {
            usage::reconcile(&store, &reservation, response.tokens)
        })
        .await
        .map_err(|_| Error::Task)??;
        Ok(response.output)
    };
    tokio::time::timeout_at(deadline, attempt)
        .await
        .map_err(|_| Error::Deadline)?
}

#[cfg(test)]
mod tests;
