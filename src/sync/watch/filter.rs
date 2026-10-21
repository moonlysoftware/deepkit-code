//! Preserve source mutation hints through debounce queues; see v1.md §4.5.
use notify_debouncer_full::notify::{
    self, EventHandler, EventKind, RecursiveMode, event::ModifyKind,
};
use std::path::Path;

/// Access events may sit between writes in a full-debouncer queue. Filtering
/// them only after debounce lets an unexpired access event split a write burst.
/// Queries can publish intermediate filesystem states before debounce completes,
/// so create/delete cancellation and rename folding must not erase affected paths.
pub(super) struct SourceWatcher<T>(pub(super) T);

impl<T: notify::Watcher> notify::Watcher for SourceWatcher<T> {
    fn new<F: EventHandler>(mut handler: F, config: notify::Config) -> notify::Result<Self> {
        T::new(
            move |event: notify::Result<notify::Event>| {
                if event
                    .as_ref()
                    .is_ok_and(|event| event.kind.is_access() && !event.need_rescan())
                {
                    return;
                }
                match event {
                    Ok(event)
                        if !event.need_rescan()
                            && !event.kind.is_access()
                            && !matches!(event.kind, EventKind::Other) =>
                    {
                        if event.paths.is_empty() {
                            handler.handle_event(Ok(event));
                            return;
                        }
                        // Consumers need paths, not operation history. Independent
                        // modify hints retain both rename sides and cannot cancel
                        // when a later event removes a just-created destination.
                        // One shared kind also lets the full debouncer collapse all
                        // write metadata and content hints for the same path.
                        for path in &event.paths {
                            let mut hint = event.clone();
                            hint.kind = EventKind::Modify(ModifyKind::Any);
                            hint.paths = vec![path.clone()];
                            handler.handle_event(Ok(hint));
                        }
                    }
                    other => handler.handle_event(other),
                }
            },
            config,
        )
        .map(Self)
    }

    fn watch(&mut self, path: &Path, mode: RecursiveMode) -> notify::Result<()> {
        self.0.watch(path, mode)
    }

    fn unwatch(&mut self, path: &Path) -> notify::Result<()> {
        self.0.unwatch(path)
    }

    fn configure(&mut self, config: notify::Config) -> notify::Result<bool> {
        self.0.configure(config)
    }

    fn kind() -> notify::WatcherKind {
        T::kind()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use notify::Watcher;

    struct Manual(Box<dyn EventHandler>);
    impl Watcher for Manual {
        fn new<F: EventHandler>(handler: F, _: notify::Config) -> notify::Result<Self> {
            Ok(Self(Box::new(handler)))
        }
        fn watch(&mut self, _: &Path, _: RecursiveMode) -> notify::Result<()> {
            Ok(())
        }
        fn unwatch(&mut self, _: &Path) -> notify::Result<()> {
            Ok(())
        }
        fn kind() -> notify::WatcherKind {
            notify::WatcherKind::PollWatcher
        }
    }

    /// Replays a burst before any debounce tick, independently of the host OS.
    struct Burst(Box<dyn EventHandler>);
    impl Watcher for Burst {
        fn new<F: EventHandler>(handler: F, _: notify::Config) -> notify::Result<Self> {
            Ok(Self(Box::new(handler)))
        }
        fn watch(&mut self, root: &Path, _: RecursiveMode) -> notify::Result<()> {
            use notify::{
                Event, EventKind,
                event::{CreateKind, ModifyKind, RemoveKind, RenameMode},
            };
            for (kind, path) in [
                (EventKind::Create(CreateKind::File), "transient.rs"),
                (EventKind::Remove(RemoveKind::File), "transient.rs"),
                (
                    EventKind::Modify(ModifyKind::Name(RenameMode::From)),
                    "old.rs",
                ),
                (
                    EventKind::Modify(ModifyKind::Name(RenameMode::To)),
                    "new.rs",
                ),
                (EventKind::Remove(RemoveKind::File), "new.rs"),
                (EventKind::Modify(ModifyKind::Any), "survivor.rs"),
            ] {
                self.0.handle_event(Ok(Event::new(kind)
                    .add_path(root.join(path))
                    .set_tracker(42)));
            }
            Ok(())
        }
        fn unwatch(&mut self, _: &Path) -> notify::Result<()> {
            Ok(())
        }
        fn kind() -> notify::WatcherKind {
            notify::WatcherKind::PollWatcher
        }
    }

    #[test]
    fn debounce_retains_paths_that_queries_may_publish_inside_a_mutation_burst() {
        use notify_debouncer_full::{NoCache, new_debouncer_opt};
        use std::{collections::BTreeSet, time::Duration};
        let (send, receive) = std::sync::mpsc::channel();
        let mut watcher = new_debouncer_opt::<_, SourceWatcher<Burst>, NoCache>(
            Duration::from_millis(20),
            None,
            send,
            NoCache,
            Default::default(),
        )
        .unwrap();
        let root = Path::new("/fixture");
        watcher.watch(root, RecursiveMode::Recursive).unwrap();
        let events = receive
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap();
        let paths = events
            .into_iter()
            .flat_map(|event| event.paths.clone())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            paths,
            ["transient.rs", "old.rs", "new.rs", "survivor.rs"]
                .map(|p| root.join(p))
                .into()
        );
    }

    #[test]
    fn access_interleaving_never_reaches_debounce_but_mutations_and_loss_do() {
        use notify::{
            Event, EventKind,
            event::{AccessKind, Flag, ModifyKind},
        };
        let (send, receive) = std::sync::mpsc::channel();
        let mut watcher = SourceWatcher::<Manual>::new(send, Default::default()).unwrap();
        for _ in 0..1000 {
            watcher
                .0
                .0
                .handle_event(Ok(Event::new(EventKind::Access(AccessKind::Any))));
            watcher
                .0
                .0
                .handle_event(Ok(Event::new(EventKind::Modify(ModifyKind::Any))));
        }
        watcher.0.0.handle_event(Ok(
            Event::new(EventKind::Access(AccessKind::Any)).set_flag(Flag::Rescan)
        ));
        watcher
            .0
            .0
            .handle_event(Err(notify::Error::generic("fixture backend loss")));
        let events = receive.try_iter().collect::<Vec<_>>();
        assert_eq!(events.len(), 1002);
        assert!(
            events[..1000]
                .iter()
                .all(|e| e.as_ref().unwrap().kind.is_modify())
        );
        assert!(events[1000].as_ref().unwrap().need_rescan());
        assert!(events[1001].is_err());
    }
}
