//! Query-owned retries under one absolute deadline; v1 §§8.2, 9.
use super::admission::Error;
use crate::models::attempt::AttemptError;
use backon::{BackoffBuilder, ExponentialBuilder};
use std::{future::Future, time::Duration};
use tokio::time::Instant;

/// Calls an independently admitted single-attempt operation at most eight times.
/// Quota/policy/permanent failures return immediately. Transient backoff has
/// jitter, capped at 60 seconds before applying Retry-After as a lower bound.
/// Dimensions negotiation can retry immediately, but still consumes an attempt.
/// If a retry cannot start before the deadline, return the last provider failure
/// immediately so other retrieval legs can use the remaining budget.
pub(crate) async fn execute<T, F, Fut>(deadline: Instant, mut call: F) -> Result<T, Error>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, Error>>,
{
    let mut backoff = ExponentialBuilder::default()
        .with_min_delay(Duration::from_secs(1))
        .with_max_delay(Duration::from_secs(60))
        .with_max_times(7)
        .with_jitter()
        .build();
    let mut attempts = 0;
    loop {
        if Instant::now() >= deadline {
            return Err(Error::Deadline);
        }
        let result = tokio::time::timeout_at(deadline, call())
            .await
            .map_err(|_| Error::Deadline)?;
        if Instant::now() >= deadline {
            return Err(Error::Deadline);
        }
        let error = match result {
            Ok(output) => return Ok(output),
            Err(error) => error,
        };
        attempts += 1;
        if attempts >= 8 {
            return Err(error);
        }
        let delay = match &error {
            Error::Provider(AttemptError::DimensionsRejected) => Duration::ZERO,
            Error::Provider(AttemptError::Retryable { retry_after, .. }) => {
                let Some(backoff) = backoff.next() else {
                    return Err(error);
                };
                backoff
                    .min(Duration::from_secs(60))
                    .max(retry_after.unwrap_or(Duration::ZERO))
            }
            _ => return Err(error),
        };
        let Some(next) = Instant::now()
            .checked_add(delay)
            .filter(|next| *next < deadline)
        else {
            return Err(error);
        };
        tokio::time::sleep_until(next).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn negotiation_is_bounded_and_permanent_failures_are_not_retried() {
        let calls = AtomicUsize::new(0);
        let deadline = Instant::now() + Duration::from_secs(2);
        let result = execute::<(), _, _>(deadline, || async {
            calls.fetch_add(1, Ordering::SeqCst);
            Err(Error::Provider(AttemptError::DimensionsRejected))
        })
        .await;
        assert!(matches!(
            result,
            Err(Error::Provider(AttemptError::DimensionsRejected))
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 8);
        for error in [
            AttemptError::InvalidResponse,
            AttemptError::InvalidInput,
            AttemptError::Rejected { status: 401 },
        ] {
            let mut error = Some(error);
            let result = execute::<(), _, _>(deadline, || {
                calls.fetch_add(1, Ordering::SeqCst);
                std::future::ready(Err(Error::Provider(error.take().expect("no retry"))))
            })
            .await;
            assert!(matches!(result, Err(Error::Provider(_))));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 11);
    }

    #[tokio::test]
    async fn retry_after_cannot_extend_deadline_or_force_a_wait() {
        let calls = AtomicUsize::new(0);
        let result = tokio::time::timeout(
            Duration::from_secs(1),
            execute::<(), _, _>(Instant::now() + Duration::from_secs(2), || async {
                calls.fetch_add(1, Ordering::SeqCst);
                Err(Error::Provider(AttemptError::Retryable {
                    status: Some(429),
                    retry_after: Some(Duration::from_secs(60)),
                }))
            }),
        )
        .await
        .unwrap();
        assert!(matches!(
            result,
            Err(Error::Provider(AttemptError::Retryable {
                status: Some(429),
                ..
            }))
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
