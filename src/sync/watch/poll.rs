//! Ignore-aware metadata polling with subsecond change detection; v1 §4.1/4.5.
// notify 8.2's PollWatcher truncates mtime to seconds unless it hashes content.
// Keep full timestamps instead; source hashing belongs to the bounded L1 reader.

use notify_debouncer_full::notify::{
    self, Event, EventHandler, EventKind, RecursiveMode, WatcherKind,
    event::{CreateKind, ModifyKind, RemoveKind},
};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, SystemTime},
};

#[derive(PartialEq, Eq)]
struct Stamp {
    modified: SystemTime,
    size: u64,
    #[cfg(unix)]
    identity: (u64, u64, i64, i64),
}
struct Root {
    path: PathBuf,
    recursive: RecursiveMode,
    files: BTreeMap<PathBuf, Stamp>,
}
#[derive(Default)]
struct Shared {
    root: Mutex<Option<Root>>,
    stopped: AtomicBool,
}

/// Supplies metadata-only events to the full debouncer without whole-second
/// timestamp loss, symlink traversal or reading ignored/oversized source bytes.
pub(super) struct MetadataWatcher(Arc<Shared>);

impl notify::Watcher for MetadataWatcher {
    fn new<F: EventHandler>(mut handler: F, config: notify::Config) -> notify::Result<Self> {
        let shared = Arc::new(Shared::default());
        let worker = Arc::clone(&shared);
        let interval = config.poll_interval().unwrap_or(Duration::from_millis(250));
        std::thread::Builder::new()
            .name("dkc-metadata-watch".into())
            .spawn(move || {
                while !worker.stopped.load(Ordering::Acquire) {
                    {
                        let mut root = worker.root.lock().unwrap_or_else(|e| e.into_inner());
                        if let Some(root) = root.as_mut() {
                            match snapshot(&root.path, root.recursive, &worker.stopped) {
                                Ok(files) if !worker.stopped.load(Ordering::Acquire) => {
                                    for (path, stamp) in &files {
                                        let kind = match root.files.get(path) {
                                            None => Some(EventKind::Create(CreateKind::File)),
                                            Some(old) if old != stamp => {
                                                Some(EventKind::Modify(ModifyKind::Any))
                                            }
                                            _ => None,
                                        };
                                        if let Some(kind) = kind {
                                            handler.handle_event(Ok(
                                                Event::new(kind).add_path(path.clone())
                                            ));
                                        }
                                    }
                                    for path in
                                        root.files.keys().filter(|path| !files.contains_key(*path))
                                    {
                                        handler.handle_event(Ok(Event::new(EventKind::Remove(
                                            RemoveKind::File,
                                        ))
                                        .add_path(path.clone())));
                                    }
                                    root.files = files;
                                }
                                Err(error) => handler.handle_event(Err(error)),
                                _ => {}
                            }
                        }
                    }
                    std::thread::sleep(interval);
                }
            })
            .map_err(notify::Error::io)?;
        Ok(Self(shared))
    }

    fn watch(&mut self, path: &Path, recursive: RecursiveMode) -> notify::Result<()> {
        let files = snapshot(path, recursive, &self.0.stopped)?;
        *self.0.root.lock().unwrap_or_else(|e| e.into_inner()) = Some(Root {
            path: path.into(),
            recursive,
            files,
        });
        Ok(())
    }

    fn unwatch(&mut self, path: &Path) -> notify::Result<()> {
        let mut root = self.0.root.lock().unwrap_or_else(|e| e.into_inner());
        if root.as_ref().is_some_and(|root| root.path == path) {
            *root = None;
            Ok(())
        } else {
            Err(notify::Error::watch_not_found())
        }
    }

    fn kind() -> WatcherKind {
        WatcherKind::PollWatcher
    }
}

impl Drop for MetadataWatcher {
    fn drop(&mut self) {
        self.0.stopped.store(true, Ordering::Release);
    }
}

fn snapshot(
    root: &Path,
    recursive: RecursiveMode,
    stopped: &AtomicBool,
) -> notify::Result<BTreeMap<PathBuf, Stamp>> {
    let mut builder = crate::sync::discovery_builder(root);
    if recursive == RecursiveMode::NonRecursive {
        builder.max_depth(Some(1));
    }
    let mut files = BTreeMap::new();
    for entry in builder.build() {
        if stopped.load(Ordering::Acquire) {
            break;
        }
        let entry = entry.map_err(|_| notify::Error::generic("watch discovery failed"))?;
        if entry.error().is_some() {
            return Err(notify::Error::generic(
                "watch ignore policy could not be read",
            ));
        }
        if entry.path_is_symlink() || !entry.file_type().is_some_and(|kind| kind.is_file()) {
            continue;
        }
        let metadata = match fs::symlink_metadata(entry.path()) {
            Ok(metadata) if metadata.is_file() => metadata,
            Ok(_) => continue,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(notify::Error::io(error)),
        };
        let stamp = Stamp {
            modified: metadata.modified().map_err(notify::Error::io)?,
            size: metadata.len(),
            #[cfg(unix)]
            identity: {
                use std::os::unix::fs::MetadataExt;
                (
                    metadata.dev(),
                    metadata.ino(),
                    metadata.ctime(),
                    metadata.ctime_nsec(),
                )
            },
        };
        files.insert(entry.into_path(), stamp);
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use notify::Watcher;

    #[test]
    fn watcher_polling_detects_equal_size_subsecond_edits_without_reading_ignored_bytes() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("source.rs");
        fs::write(&source, "pub fn before() {}\n").unwrap();
        fs::write(root.path().join(".deepkitignore"), "ignored.rs\n").unwrap();
        fs::write(root.path().join("ignored.rs"), b"never poll these bytes").unwrap();
        let set_time = |nanos| {
            fs::File::open(&source)
                .unwrap()
                .set_times(
                    fs::FileTimes::new()
                        .set_modified(SystemTime::UNIX_EPOCH + Duration::new(1_700_000_000, nanos)),
                )
                .unwrap()
        };
        set_time(100);
        let (send, receive) = std::sync::mpsc::channel();
        let mut watcher = MetadataWatcher::new(
            send,
            notify::Config::default().with_poll_interval(Duration::from_millis(20)),
        )
        .unwrap();
        watcher
            .watch(root.path(), RecursiveMode::Recursive)
            .unwrap();
        fs::write(&source, "pub fn after_() {}\n").unwrap();
        set_time(200);
        let event = receive
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        assert!(event.kind.is_modify());
        assert_eq!(event.paths.as_slice(), std::slice::from_ref(&source));
        let observed = snapshot(
            root.path(),
            RecursiveMode::Recursive,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(observed.keys().collect::<Vec<_>>(), [&source]);
        drop(watcher);
    }
}
