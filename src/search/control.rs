//! Request-owned deadlines and cancellation across async/blocking search; v1 §9.
use super::Error;
use std::{
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::watch;

/// Sole cancellation owner; dropping an invocation wakes async work and fences reads.
pub(crate) struct Owner {
    pub(crate) deadline: Instant,
    pub(crate) cancelled: Arc<AtomicBool>,
    wake: watch::Sender<bool>,
}
impl Owner {
    /// Starts one absolute budget, including work done before provider selection.
    pub(crate) fn new(started: Instant, timeout_ms: u64) -> Result<Self, Error> {
        let deadline = started
            .checked_add(Duration::from_millis(timeout_ms))
            .ok_or(Error::InvalidQuery {
                reason: "query timeout exceeds the supported time range",
            })?;
        let (wake, _) = watch::channel(false);
        Ok(Self {
            deadline,
            cancelled: Arc::new(AtomicBool::new(false)),
            wake,
        })
    }
    /// Borrows invocation authority without extending its cancellation ownership.
    pub(crate) fn control(&self) -> Control {
        Control {
            deadline: tokio::time::Instant::from_std(self.deadline),
            cancelled: Arc::clone(&self.cancelled),
            wake: self.wake.subscribe(),
        }
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        self.wake.send_replace(true);
    }
}

/// Cloneable authority for a stage; keeping this alive cannot keep its owner alive.
#[derive(Clone)]
pub(crate) struct Control {
    deadline: tokio::time::Instant,
    cancelled: Arc<AtomicBool>,
    wake: watch::Receiver<bool>,
}
impl Control {
    /// Supplies the unchanged deadline to provider APIs with their own drop guards.
    pub(crate) fn deadline(&self) -> tokio::time::Instant {
        self.deadline
    }
    /// Fences queued blocking work before and after acquiring database authority.
    pub(crate) fn check(&self) -> Result<(), Error> {
        if self.cancelled.load(Ordering::Acquire) || tokio::time::Instant::now() >= self.deadline {
            Err(Error::Deadline)
        } else {
            Ok(())
        }
    }
    /// Drops unfinished provider futures promptly on cancellation or budget expiry.
    pub(crate) async fn run<T>(&self, operation: impl Future<Output = T>) -> Result<T, Error> {
        self.check()?;
        let mut wake = self.wake.clone();
        tokio::select! {
            biased;
            _ = wake.wait_for(|cancelled| *cancelled) => Err(Error::Deadline),
            _ = tokio::time::sleep_until(self.deadline) => Err(Error::Deadline),
            output = operation => { self.check()?; Ok(output) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn cancellation_wakes_and_drops_pending_work_without_waiting_for_deadline() {
        struct Dropped(Arc<AtomicBool>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        let owner = Owner::new(Instant::now(), 60_000).unwrap();
        let control = owner.control();
        let dropped = Arc::new(AtomicBool::new(false));
        let guard = Dropped(Arc::clone(&dropped));
        let (entered, ready) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            control
                .run(async move {
                    let _guard = guard;
                    entered.send(()).unwrap();
                    std::future::pending::<()>().await;
                })
                .await
        });
        ready.await.unwrap();
        drop(owner);
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(1), task)
                .await
                .unwrap()
                .unwrap(),
            Err(Error::Deadline)
        ));
        assert!(dropped.load(Ordering::Acquire));
        let retry = Owner::new(Instant::now(), 1_000).unwrap();
        assert_eq!(retry.control().run(async { 42 }).await.unwrap(), 42);
    }
    #[tokio::test]
    async fn expired_authority_does_not_poll_or_refresh_its_budget() {
        let owner = Owner::new(Instant::now() - Duration::from_secs(1), 1).unwrap();
        let control = owner.control();
        assert_eq!(control.deadline().into_std(), owner.deadline);
        let polled = AtomicBool::new(false);
        assert!(matches!(
            control
                .run(async {
                    polled.store(true, Ordering::Release);
                })
                .await,
            Err(Error::Deadline)
        ));
        assert!(!polled.load(Ordering::Acquire));
    }
}
