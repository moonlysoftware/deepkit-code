//! Process-local background L1 progress for zero-setup serve; v1 §4.2, §10.2.
use crate::sync::index::Progress;
use serde::Serialize;
use std::sync::{
    Arc, Mutex, MutexGuard,
    atomic::{AtomicBool, Ordering},
};

/// Shares worker activity with request repair without writing progress to SQLite.
/// The supervisor must call `begin` before spawning and `fail` on errors or panics.
#[derive(Clone, Default)]
pub(super) struct Status {
    active: Arc<AtomicBool>,
    suppressed: Arc<AtomicBool>,
    snapshot: Arc<Mutex<Option<Snapshot>>>,
}

/// A bounded summary of this process's background L1 attempt, not index freshness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(super) struct Snapshot {
    /// Whether this attempt is building, complete, capped/partial, or failed.
    pub(super) state: State,
    /// Last pipeline stage observed; retained on failure to identify stalled work.
    pub(super) stage: Stage,
    /// File completion percentage, capped at 99 until the whole attempt completes.
    /// Discovery has no known denominator and reports zero. This is not an ETA.
    pub(super) percentage: u8,
    /// Counts are absent until the runner yields its pending L1 work list.
    pub(super) processed: Option<usize>,
    /// Pending L1 file count reported by the runner, not all repository files.
    pub(super) total: Option<usize>,
}

/// Terminal states describe the attempt, without retaining arbitrary error text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum State {
    Building,
    Complete,
    Partial,
    Failed,
}

/// Distinguishes discovery and post-file work when the file count cannot advance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Stage {
    Discovering,
    Files,
    Graph,
    Completing,
    Finished,
}

impl Status {
    /// Exposes the same flag to Context's repair admission; readers use Acquire.
    /// Only this status owner should mutate it, to keep snapshots synchronized.
    pub(super) fn active(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.active)
    }

    /// Admits one worker and resets only this process's previous attempt summary.
    /// Call synchronously before spawning so incoming tools see building state.
    pub(super) fn begin(&self) -> bool {
        let mut snapshot = self.lock();
        if self.active.load(Ordering::Acquire) {
            return false;
        }
        *snapshot = Some(Snapshot {
            state: State::Building,
            stage: Stage::Discovering,
            percentage: 0,
            processed: None,
            total: None,
        });
        self.active.store(true, Ordering::Release);
        true
    }

    /// Records ordered observations from the admitted worker; ignores terminal updates.
    /// The supervisor must join the old worker before admitting a new attempt.
    /// Finished clears activity only after the terminal snapshot is available.
    pub(super) fn observe(&self, progress: Progress) {
        let mut snapshot = self.lock();
        let Some(current) = snapshot
            .as_mut()
            .filter(|value| value.state == State::Building)
        else {
            return;
        };
        match progress {
            Progress::Discovering => current.stage = Stage::Discovering,
            Progress::Files { processed, total } => {
                current.stage = Stage::Files;
                current.processed = Some(processed);
                current.total = Some(total);
                // u128 avoids multiplication overflow for every supported usize.
                current.percentage = if total == 0 {
                    0
                } else {
                    ((processed as u128 * 100 / total as u128).min(99)) as u8
                };
            }
            Progress::Graph => current.stage = Stage::Graph,
            Progress::Completing => current.stage = Stage::Completing,
            Progress::Finished { complete } => {
                current.stage = Stage::Finished;
                current.state = if complete {
                    State::Complete
                } else {
                    State::Partial
                };
                if complete {
                    current.percentage = 100;
                }
                self.active.store(false, Ordering::Release);
            }
        }
    }

    /// Releases repair admission after an error or join failure, retaining counters.
    /// Takes no error argument so source paths or credentials cannot enter status.
    pub(super) fn fail(&self) {
        let mut snapshot = self.lock();
        if let Some(current) = snapshot
            .as_mut()
            .filter(|value| value.state == State::Building)
        {
            current.state = State::Failed;
            self.active.store(false, Ordering::Release);
        }
    }

    /// Permanently hides this attempt after its effective configuration is replaced.
    /// The original worker retains activity ownership until it actually finishes.
    pub(super) fn suppress(&self) {
        self.suppressed.store(true, Ordering::Release);
    }

    /// Returns no claim before startup or after configuration invalidates the attempt.
    pub(super) fn snapshot(&self) -> Option<Snapshot> {
        let snapshot = *self.lock();
        if self.suppressed.load(Ordering::Acquire) {
            None
        } else {
            snapshot
        }
    }

    fn lock(&self) -> MutexGuard<'_, Option<Snapshot>> {
        self.snapshot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_activity_and_stage_progress_complete_only_on_finished() {
        let status = Status::default();
        let shared = status.clone();
        let active = shared.active();
        assert_eq!(status.snapshot(), None);
        assert!(!active.load(Ordering::Acquire));
        assert!(status.begin());
        assert!(!shared.begin());
        assert!(active.load(Ordering::Acquire));
        assert_eq!(shared.snapshot().unwrap().stage, Stage::Discovering);
        status.observe(Progress::Files {
            processed: 1,
            total: 4,
        });
        assert_eq!(shared.snapshot().unwrap().percentage, 25);
        status.observe(Progress::Files {
            processed: 4,
            total: 4,
        });
        for event in [Progress::Graph, Progress::Completing] {
            status.observe(event);
            let current = shared.snapshot().unwrap();
            assert_eq!(current.state, State::Building);
            assert_eq!(current.percentage, 99);
            assert!(active.load(Ordering::Acquire));
        }
        status.observe(Progress::Finished { complete: true });
        let current = shared.snapshot().unwrap();
        assert_eq!(current.state, State::Complete);
        assert_eq!(current.stage, Stage::Finished);
        assert_eq!(current.percentage, 100);
        assert_eq!((current.processed, current.total), (Some(4), Some(4)));
        assert!(!active.load(Ordering::Acquire));
        status.fail();
        status.observe(Progress::Graph);
        assert_eq!(shared.snapshot(), Some(current));
    }

    #[test]
    fn partial_and_failed_keep_honest_counts_and_allow_a_new_attempt() {
        let status = Status::default();
        status.fail();
        status.observe(Progress::Finished { complete: true });
        assert_eq!(status.snapshot(), None);
        assert!(status.begin());
        status.observe(Progress::Files {
            processed: 2,
            total: 5,
        });
        status.observe(Progress::Finished { complete: false });
        let partial = status.snapshot().unwrap();
        assert_eq!(partial.state, State::Partial);
        assert_eq!(partial.percentage, 40);
        assert!(!status.active().load(Ordering::Acquire));
        assert!(status.begin());
        assert_eq!(status.snapshot().unwrap().total, None);
        status.observe(Progress::Files {
            processed: 1,
            total: 3,
        });
        status.fail();
        let failed = status.snapshot().unwrap();
        assert_eq!(failed.state, State::Failed);
        assert_eq!(failed.stage, Stage::Files);
        assert_eq!(failed.percentage, 33);
        assert_eq!(failed.processed, Some(1));
        assert!(!status.active().load(Ordering::Acquire));
    }

    #[test]
    fn empty_and_large_file_counts_do_not_divide_by_zero_or_overflow() {
        for (processed, total, percentage) in [(0, 0, 0), (usize::MAX, usize::MAX, 99), (2, 1, 99)]
        {
            let status = Status::default();
            assert!(status.begin());
            status.observe(Progress::Files { processed, total });
            assert_eq!(status.snapshot().unwrap().percentage, percentage);
            status.observe(Progress::Finished { complete: false });
            assert!(status.snapshot().unwrap().percentage < 100);
        }
    }
    #[test]
    fn suppressed_attempt_never_reappears_but_worker_keeps_activity_ownership() {
        let status = Status::default();
        let observer = status.clone();
        assert!(status.begin());
        status.suppress();
        assert_eq!(observer.snapshot(), None);
        assert!(observer.active().load(Ordering::Acquire));
        status.observe(Progress::Finished { complete: true });
        assert!(!observer.active().load(Ordering::Acquire));
        assert_eq!(observer.snapshot(), None);
        assert!(status.begin());
        assert_eq!(observer.snapshot(), None);
        status.fail();
        assert!(!observer.active().load(Ordering::Acquire));
    }
}
