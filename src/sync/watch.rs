//! Debounced filesystem hints with bounded delivery; v1.md §2.1 and §4.5.
mod filter;
mod poll;

use notify_debouncer_full::{
    DebounceEventResult, Debouncer, NoCache, new_debouncer_opt,
    notify::{self, RecommendedWatcher, RecursiveMode},
};
use std::{
    collections::BTreeSet,
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

use filter::SourceWatcher;
/// Keeps OS backend failures typed at the transport boundary.
pub(crate) use notify::Error as BackendError;
use poll::MetadataWatcher;

const DEBOUNCE: Duration = Duration::from_millis(500);
const MAX_PENDING_PATHS: usize = 4096;

/// One coalesced batch. Overflow or backend loss requires a full reconciliation;
/// dropping individual event paths would otherwise leave an undetectable gap.
#[derive(Default, Debug)]
pub(crate) struct Batch {
    pub(crate) paths: BTreeSet<String>,
    pub(crate) rescan: bool,
    pub(crate) backend_failed: bool,
}

impl Batch {
    /// Indicates that no path or recovery scan is outstanding.
    pub(crate) fn empty(&self) -> bool {
        self.paths.is_empty() && !self.rescan
    }

    /// Retains failed work alongside new arrivals without an unbounded backlog.
    pub(crate) fn merge(&mut self, other: Self) {
        self.rescan |= other.rescan;
        self.backend_failed |= other.backend_failed;
        self.paths.extend(other.paths);
        self.rescan |= self.paths.len() > MAX_PENDING_PATHS;
        if self.rescan {
            self.paths.clear();
        }
    }
}

#[derive(Default)]
struct Pending {
    batch: Mutex<Batch>,
    ready: tokio::sync::Notify,
}

/// Owns backend and debounce threads for exactly one process lifetime. No file-ID
/// cache is built; polling reads metadata only, leaving hashing behind ignore checks.
pub(crate) struct Watcher {
    root: PathBuf,
    pending: Arc<Pending>,
    native: Option<Debouncer<SourceWatcher<RecommendedWatcher>, NoCache>>,
    _polling: Option<Debouncer<SourceWatcher<MetadataWatcher>, NoCache>>,
}

impl Watcher {
    /// Registers before initial indexing so changes during bootstrap remain queued.
    /// Native registration failure falls back to polling; --poll selects it directly.
    pub(crate) fn start(root: &Path, poll: bool) -> Result<Self, notify::Error> {
        let pending = Arc::new(Pending::default());
        if !poll {
            match backend::<RecommendedWatcher>(root, Arc::clone(&pending)) {
                Ok(native) => {
                    return Ok(Self {
                        root: root.into(),
                        pending,
                        native: Some(native),
                        _polling: None,
                    });
                }
                Err(_) => tracing::warn!("Native watcher unavailable; using metadata polling"),
            }
        }
        let polling = backend::<MetadataWatcher>(root, Arc::clone(&pending))?;
        Ok(Self {
            root: root.into(),
            pending,
            native: None,
            _polling: Some(polling),
        })
    }

    /// Exposes backend choice without source paths or OS diagnostics.
    pub(crate) fn backend(&self) -> &'static str {
        if self.native.is_some() {
            "native"
        } else {
            "polling"
        }
    }

    /// Returns the captured root so a supervisor cannot apply old-root events to a
    /// newly configured index. A root change requires a new registration and scan.
    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    /// Waits without blocking the async runtime; arrivals during publication remain
    /// queued for the next batch. The set is bounded independently of event volume.
    pub(crate) async fn next(&self) -> Batch {
        loop {
            let ready = self.pending.ready.notified();
            {
                let mut batch = self.pending.batch.lock().unwrap_or_else(|e| e.into_inner());
                if !batch.empty() {
                    return std::mem::take(&mut *batch);
                }
            }
            ready.await;
        }
    }
}

fn backend<T: notify::Watcher>(
    root: &Path,
    pending: Arc<Pending>,
) -> Result<Debouncer<SourceWatcher<T>, NoCache>, notify::Error> {
    let observed_root = root.to_path_buf();
    let mut watcher = new_debouncer_opt::<_, SourceWatcher<T>, NoCache>(
        DEBOUNCE,
        None,
        move |events: DebounceEventResult| receive(&observed_root, &pending, events),
        NoCache,
        notify::Config::default()
            .with_follow_symlinks(false)
            .with_poll_interval(Duration::from_millis(250))
            .with_compare_contents(false),
    )?;
    watcher.watch(root, RecursiveMode::Recursive)?;
    Ok(watcher)
}

fn receive(root: &Path, pending: &Pending, events: DebounceEventResult) {
    let mut batch = pending.batch.lock().unwrap_or_else(|e| e.into_inner());
    match events {
        Err(_) => {
            batch.rescan = true;
            batch.backend_failed = true;
        }
        Ok(events) => {
            for event in events {
                if event.kind.is_access() && !event.need_rescan() {
                    continue;
                }
                if event.need_rescan() {
                    batch.rescan = true;
                }
                for path in &event.paths {
                    let Ok(relative) = path.strip_prefix(root) else {
                        continue;
                    };
                    if relative
                        .components()
                        .any(|part| !matches!(part, Component::Normal(_)))
                    {
                        continue;
                    }
                    if relative
                        .components()
                        .any(|part| part.as_os_str() == ".deepkit")
                    {
                        continue;
                    }
                    if relative.starts_with(".git") {
                        if relative == Path::new(".git/info/exclude") {
                            batch.rescan = true;
                        }
                        continue;
                    }
                    if matches!(
                        relative.file_name().and_then(|s| s.to_str()),
                        Some(".gitignore" | ".deepkitignore" | ".ignore")
                    ) {
                        batch.rescan = true;
                        continue;
                    }
                    let Some(path) = relative.to_str() else {
                        batch.rescan = true;
                        continue;
                    };
                    if path.is_empty() {
                        batch.rescan = true;
                        continue;
                    }
                    if !batch.rescan {
                        batch.paths.insert(super::normalize_path(path));
                        if batch.paths.len() > MAX_PENDING_PATHS {
                            batch.rescan = true;
                        }
                    }
                }
            }
        }
    }
    if batch.rescan {
        batch.paths.clear();
    }
    if !batch.empty() {
        pending.ready.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use notify_debouncer_full::{
        DebouncedEvent,
        notify::{
            Event, EventKind,
            event::{AccessKind, CreateKind, ModifyKind, RenameMode},
        },
    };

    fn event(kind: EventKind, paths: &[&str]) -> DebouncedEvent {
        let mut event = Event::new(kind);
        event.paths = paths.iter().map(PathBuf::from).collect();
        DebouncedEvent::new(event, std::time::Instant::now())
    }

    #[test]
    fn watcher_delivery_coalesces_rename_pairs_and_excludes_internal_access_events() {
        let pending = Pending::default();
        let root = Path::new("/repo");
        let mut events = (0..1000)
            .map(|_| event(EventKind::Create(CreateKind::File), &["/repo/src/file.rs"]))
            .collect::<Vec<_>>();
        events.push(event(
            EventKind::Modify(ModifyKind::Name(RenameMode::Both)),
            &["/repo/src/old.rs", "/repo/src/new.rs"],
        ));
        for path in [
            "/repo/.deepkit/code.db-wal",
            "/repo/.git/index",
            "/outside/other.rs",
            "/repo/../other.rs",
        ] {
            events.push(event(EventKind::Create(CreateKind::File), &[path]));
        }
        events.push(event(
            EventKind::Access(AccessKind::Any),
            &["/repo/src/read.rs"],
        ));
        receive(root, &pending, Ok(events));
        let batch = pending.batch.lock().unwrap();
        assert_eq!(
            batch.paths,
            ["src/file.rs", "src/new.rs", "src/old.rs"]
                .map(String::from)
                .into()
        );
        assert!(!batch.rescan);
    }

    #[test]
    fn watcher_overflow_errors_and_ignore_changes_require_reconciliation() {
        for scenario in ["overflow", "backend", "ignore"] {
            let pending = Pending::default();
            let events = match scenario {
                "overflow" => Ok((0..=MAX_PENDING_PATHS)
                    .map(|n| {
                        event(
                            EventKind::Create(CreateKind::File),
                            &[&format!("/repo/{n}.rs")],
                        )
                    })
                    .collect()),
                "backend" => Err(vec![notify::Error::generic("backend lost events")]),
                _ => Ok(vec![event(
                    EventKind::Create(CreateKind::File),
                    &["/repo/nested/.gitignore"],
                )]),
            };
            receive(Path::new("/repo"), &pending, events);
            let batch = pending.batch.lock().unwrap();
            assert!(batch.rescan);
            assert!(batch.paths.is_empty());
            assert_eq!(batch.backend_failed, scenario == "backend");
        }
    }
}
