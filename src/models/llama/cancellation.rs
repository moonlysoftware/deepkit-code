//! Bounded native work lifetime across dropped embedding futures; v1 §7.2/§8.2.
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::Semaphore;

/// A cooperative stop signal checked between native operations. In-flight decode
/// keeps its context/model alive until it returns; no raw native pointer escapes.
#[derive(Clone, Default)]
pub(super) struct Cancellation(Arc<AtomicBool>);

impl Cancellation {
    /// Rejects further work after the owning async call has been dropped.
    pub(super) fn check(&self) -> Result<(), super::Error> {
        if self.0.load(Ordering::Acquire) {
            Err(super::Error::Cancelled)
        } else {
            Ok(())
        }
    }
}

struct CancelOnDrop(Cancellation);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.0.store(true, Ordering::Release);
    }
}

/// Acquires native capacity asynchronously, so cancelled waiters never become
/// blocked threads. The blocking task owns the permit even after its future is
/// dropped; a replacement cannot overlap unfinished decode or model destruction.
pub(super) async fn run<T: Send + 'static>(
    gate: Arc<Semaphore>,
    action: impl FnOnce(Cancellation) -> crate::models::Result<T> + Send + 'static,
) -> crate::models::Result<T> {
    let permit = gate
        .acquire_owned()
        .await
        .map_err(|_| crate::models::Error::LocalWorker)?;
    let stopped = Cancellation::default();
    let _cancel = CancelOnDrop(stopped.clone());
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        stopped.check()?;
        action(stopped)
    })
    .await
    .map_err(|_| crate::models::Error::LocalWorker)?
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        future::Future,
        task::{Context, Waker},
    };

    #[tokio::test]
    async fn cancelled_native_work_retains_capacity_and_discards_followup_work()
    -> Result<(), Box<dyn std::error::Error>> {
        let gate = Arc::new(Semaphore::new(1));
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let (finished_tx, finished_rx) = tokio::sync::oneshot::channel();
        let continued = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&continued);
        let active = tokio::spawn(run(Arc::clone(&gate), move |stopped| {
            let _ = started_tx.send(stopped.clone());
            // Simulate a native operation which cannot be forcibly interrupted.
            let _ = release_rx.recv();
            let result = stopped.check();
            if result.is_ok() {
                observed.store(true, Ordering::Release);
            }
            let _ = finished_tx.send(result.is_err());
            result?;
            Ok(())
        }));
        let stopped = started_rx.await?;
        active.abort();
        assert!(active.await.unwrap_err().is_cancelled());
        assert!(stopped.check().is_err());
        assert_eq!(gate.available_permits(), 0);
        // A queued request can be polled then cancelled without spawning work.
        let entered = Arc::new(AtomicBool::new(false));
        let entered_worker = Arc::clone(&entered);
        let mut waiting = Box::pin(run(Arc::clone(&gate), move |_| {
            entered_worker.store(true, Ordering::Release);
            Ok(())
        }));
        assert!(
            waiting
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        drop(waiting);
        assert!(!entered.load(Ordering::Acquire));
        release_tx.send(())?;
        assert!(finished_rx.await?);
        run(Arc::clone(&gate), |_| Ok(())).await?;
        assert!(!continued.load(Ordering::Acquire));
        assert!(!entered.load(Ordering::Acquire));
        assert_eq!(gate.available_permits(), 1);
        Ok(())
    }
}
