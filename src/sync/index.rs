//! Shared full L1 discovery, publication and graph completion; v1 §4.2–4.6.
use crate::{
    config::{self, Config},
    store::{self, GraphEdgeKind, GraphFileRank, GraphSymbolRank, Store},
    sync::{self, SyncLimits, l1::Work},
    treat::graph,
};
use std::{collections::BTreeSet, path::Path};

/// Typed failures preserve source chains for CLI capacity diagnoses and serve fallback.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("{0}")]
    Sync(#[from] sync::Error),
    #[error("{0}")]
    Store(#[from] store::Error),
    #[error(transparent)]
    Config(#[from] config::Error),
    #[error("missing L1 fingerprint for {0}")]
    MissingFingerprint(String),
    #[error("failed to publish L1 for {path}")]
    File {
        path: String,
        #[source]
        source: Box<sync::Error>,
    },
    #[error("L1 configuration changed during indexing; index again")]
    ConfigurationChanged,
    #[error("Git HEAD changed during indexing; index again before claiming completion")]
    HeadChanged,
    #[error("{0} is too large for this platform")]
    InvalidLimit(&'static str),
    #[error("failed to start L1 preparation worker: {0}")]
    PreparationThread(std::io::Error),
    #[error("L1 preparation worker panicked")]
    PreparationWorkerPanicked,
    #[error(transparent)]
    History(#[from] sync::history::Error),
    #[error(transparent)]
    SymbolHistory(#[from] sync::symbol_history::Error),
    #[error(transparent)]
    Modules(#[from] sync::modules::Error),
    #[error(transparent)]
    Linker(#[from] sync::linker::Error),
    #[error(transparent)]
    Roles(#[from] crate::queue::Error),
}

/// Explicit invocation inputs avoid process cwd and preserve one cancellation owner.
pub(crate) struct Input<'a> {
    pub(crate) config_root: &'a Path,
    pub(crate) root: &'a Path,
    pub(crate) settings: &'a Config,
    pub(crate) store: &'a Store,
    pub(crate) full: bool,
    pub(crate) work: Work<'a>,
}

/// Bounded progress observations; finishing an incomplete scan is not full L1 completion.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Progress {
    Discovering,
    Files { processed: usize, total: usize },
    Graph,
    Completing,
    Finished { complete: bool },
}

/// Retains original CLI counters and scan diagnostics without re-reading authority.
pub(crate) struct Report {
    pub(crate) publication: sync::l1::Publication,
    pub(crate) scan: sync::Scan,
    pub(crate) unit_count: usize,
    pub(crate) truncated_files: usize,
    pub(crate) history: sync::history::Report,
    pub(crate) prs: sync::prs::Report,
    pub(crate) histories: sync::symbol_history::Report,
    pub(crate) modules: sync::modules::Report,
    pub(crate) linker: sync::linker::Report,
}

/// Runs the established L1 pipeline with fenced per-file commits and final HEAD publication.
/// Discovery polls cancellation between entries; individual filesystem/parser calls
/// remain cooperative. Call from a blocking worker when serving async tools.
pub(crate) fn run(input: Input<'_>, progress: impl FnMut(Progress)) -> Result<Report, Error> {
    run_inner(input, None, None, progress)
}

/// Runs the same L1 and graph pipeline for a watcher event scope. Absence outside
/// that scope has no deletion authority, and a partial observation cannot advance HEAD.
pub(crate) fn run_paths(
    input: Input<'_>,
    paths: &BTreeSet<String>,
    progress: impl FnMut(Progress),
) -> Result<Report, Error> {
    run_inner(input, Some(paths), None, progress)
}

/// Reconciles a module's repository-wide provenance under the query's existing
/// L1 session and deadline. It never starts providers or bypasses index caps.
pub(crate) fn run_in_session(
    input: Input<'_>,
    session: &store::l1_session::Session,
) -> Result<Report, Error> {
    run_inner(input, None, Some(session), |_| {})
}

fn run_inner(
    input: Input<'_>,
    affected: Option<&BTreeSet<String>>,
    session: Option<&store::l1_session::Session>,
    mut progress: impl FnMut(Progress),
) -> Result<Report, Error> {
    let Input {
        config_root,
        root,
        settings,
        store,
        full,
        work,
    } = input;
    let started = std::time::Instant::now();
    let mut progress = |event| {
        tracing::debug!(
            ?event,
            elapsed_us = started.elapsed().as_micros() as u64,
            "L1 progress"
        );
        progress(event);
    };
    let check_store = || work.check().map_err(|_| store::Error::L1InputsChanged);
    work.check()?;
    let _session = if session.is_none() {
        Some(store.l1_session(root, check_store)?)
    } else {
        None
    };
    let configuration = ConfigurationFence::new(config_root, root, settings, work);
    progress(Progress::Discovering);
    let generation = store.l1_generation()?;
    let snapshot = store.index_snapshot()?;
    let git = sync::inspect_git(root, store.get_meta("indexed_head")?.as_deref())?;
    let limits = limits(settings)?;
    let scan = if let Some(paths) = affected {
        sync::paths::scan(root, paths, limits, &snapshot, work)?
    } else {
        sync::scan_checked(root, limits, git.as_ref(), &snapshot, work)?
    };
    // A query-owned module refresh must not turn the 256 KiB targeted repair
    // ceiling into an unrestricted full index. Refuse before catalogue writes;
    // the watcher or an explicit index can complete larger source inventories.
    if session.is_some() && scan.files().any(|file| file.size() >= 256 * 1024) {
        return Err(store::Error::L1InputsChanged.into());
    }
    configuration.check()?;
    work.check()?;
    let publication_input = sync::l1::PublishInput {
        work,
        scan: &scan,
        git: git.as_ref(),
        snapshot: &snapshot,
        generation,
        root,
        settings,
        full,
    };
    let publication = if let Some(paths) = affected {
        sync::l1::publish_paths(store, publication_input, paths, || {
            configuration
                .check()
                .map_err(|_| store::Error::L1InputsChanged)
        })?
    } else {
        sync::l1::publish(store, publication_input)?
    };
    let indexed = store.index_snapshot()?;
    let paths = store
        .pending_l1_paths()?
        .into_iter()
        .filter(|path| affected.is_none() || scan.file(path).is_some())
        .collect::<BTreeSet<_>>();
    // An interrupted prior graph publication has no trustworthy old topology;
    // recover with a full pass. Ordinary edits use the old and new two-hop sets.
    let rank_recovery = store.get_meta("graph_pending")?.as_deref() != Some("0");
    let prior_graph = if !full && !rank_recovery && !paths.is_empty() {
        Some(store.graph_snapshot()?)
    } else {
        None
    };
    let changed_files = indexed
        .files
        .iter()
        .filter(|file| paths.contains(&file.path))
        .map(|file| file.id)
        .collect::<BTreeSet<_>>();
    let processor = sync::l1::Processor::new(root, settings)?;
    let mut unit_count = 0_usize;
    let mut truncated_files = 0_usize;
    let total = paths.len();
    let mut processed = 0;
    progress(Progress::Files { processed, total });
    let width = if affected.is_some() {
        1
    } else {
        preparation_width(settings, total)
    };
    let paths = paths.into_iter().collect::<Vec<_>>();
    let mut inputs = Vec::with_capacity(paths.len());
    for path in &paths {
        work.check()?;
        let Some(file) = scan.file(path) else {
            continue;
        };
        let Some(record) = indexed.files.iter().find(|record| record.path == *path) else {
            continue;
        };
        let fingerprint = publication
            .fingerprints
            .get(path)
            .ok_or_else(|| Error::MissingFingerprint(path.clone()))?;
        inputs.push(sync::l1::FileInput {
            file,
            work,
            fence: store::l1::Fence {
                generation: publication.generation,
                file_id: record.id,
                path,
                content_hash: &record.blake3,
                fingerprint,
            },
        });
    }
    let validate = || {
        configuration
            .check()
            .map_err(|_| store::Error::L1InputsChanged)
    };
    let mut published = |outcome: sync::l1::FileOutcome| -> Result<(), Error> {
        unit_count += outcome.units;
        truncated_files += usize::from(outcome.truncated);
        processed += 1;
        progress(Progress::Files { processed, total });
        work.check()?;
        Ok(())
    };
    if affected.is_none() && width > 1 && inputs.len() > 1 {
        prepare_files(&processor, store, &inputs, width, |input, prepared| {
            let outcome = processor
                .publish_prepared(store, prepared, validate)
                .map_err(|source| Error::File {
                    path: input.file.path.clone(),
                    source: Box::new(source),
                })?;
            published(outcome)
        })?;
    } else {
        for input in inputs {
            let outcome = if affected.is_some() {
                processor.process_target(store, input, validate)
            } else {
                processor.process_file(store, input, validate)
            }
            .map_err(|source| Error::File {
                path: input.file.path.clone(),
                source: Box::new(source),
            })?;
            published(outcome)?;
        }
    }
    work.check()?;
    let linker = sync::linker::ingest_index(root, settings, store, work, &|| {
        configuration
            .check()
            .map_err(|_| store::Error::L1InputsChanged)
    })?;
    progress(Progress::Graph);
    #[cfg(test)]
    store::publication_observed(store::PublicationPoint::BeforeGraphRank);
    if full || store.get_meta("graph_pending")?.as_deref() != Some("0") {
        let graph_snapshot = store.graph_snapshot()?;
        let (symbols, edges) = rank_inputs(&graph_snapshot);
        let rank = if let Some(prior) = prior_graph {
            let (previous_symbols, previous_edges) = rank_inputs(&prior);
            graph::rank_changed_files(
                &symbols,
                &edges,
                &previous_symbols,
                &previous_edges,
                &changed_files,
            )
        } else {
            graph::rank_full(&symbols, &edges)
        };
        work.check()?;
        store.update_graph_ranks_checked(
            publication.generation,
            &rank
                .symbols
                .iter()
                .map(|symbol| GraphSymbolRank {
                    symbol_id: symbol.symbol_id,
                    rank: symbol.rank,
                })
                .collect::<Vec<_>>(),
            &rank
                .files
                .iter()
                .map(|file| GraphFileRank {
                    file_id: file.file_id,
                    rank: file.rank,
                })
                .collect::<Vec<_>>(),
            (work.deadline.is_some() || work.cancelled.is_some())
                .then_some(&check_store as &dyn Fn() -> Result<(), store::Error>),
        )?;
    }
    configuration.check()?;
    let current_git = sync::inspect_git(root, None)?;
    let captured_head = git
        .as_ref()
        .and_then(|g| g.head.as_ref())
        .map(|h| h.id.as_str());
    let current_head = current_git
        .as_ref()
        .and_then(|g| g.head.as_ref())
        .map(|h| h.id.as_str());
    if captured_head != current_head {
        return Err(Error::HeadChanged);
    }
    let history = sync::history::ingest(root, settings, store, work, &|| {
        configuration
            .check()
            .map_err(|_| store::Error::L1InputsChanged)
    })?;
    let prs = sync::prs::ingest(root, settings, store, work, &|| {
        configuration
            .check()
            .map_err(|_| store::Error::L1InputsChanged)
    })?;
    let histories = sync::symbol_history::ingest(root, settings, store, work, &|| {
        configuration
            .check()
            .map_err(|_| store::Error::L1InputsChanged)
    })?;
    let modules = sync::modules::ingest(root, settings, store, work, &|| {
        configuration
            .check()
            .map_err(|_| store::Error::L1InputsChanged)
    })?;
    crate::queue::roles::reconcile(store, root, config_root, settings, work)?;
    progress(Progress::Completing);
    work.check()?;
    if affected.is_none() && scan.is_complete() {
        store.complete_l1_scan(
            (work.deadline.is_some() || work.cancelled.is_some())
                .then_some(&check_store as &dyn Fn() -> Result<(), store::Error>),
            publication.generation,
            captured_head.unwrap_or(""),
            &root.to_string_lossy(),
            super::unix_timestamp(),
            || {
                configuration
                    .check()
                    .map_err(|_| store::Error::L1InputsChanged)?;
                let latest =
                    sync::inspect_git(root, None).map_err(|_| store::Error::L1InputsChanged)?;
                if latest
                    .as_ref()
                    .and_then(|g| g.head.as_ref())
                    .map(|h| h.id.as_str())
                    != captured_head
                {
                    return Err(store::Error::L1InputsChanged);
                }
                Ok(())
            },
        )?;
    }
    work.check()?;
    progress(Progress::Finished {
        complete: scan.is_complete(),
    });
    Ok(Report {
        publication,
        scan,
        unit_count,
        truncated_files,
        history,
        prs,
        histories,
        modules,
        linker,
    })
}

fn rank_inputs(snapshot: &store::GraphSnapshot) -> (Vec<graph::RankSymbol>, Vec<graph::RankEdge>) {
    let symbols = snapshot
        .symbols
        .iter()
        .filter(|symbol| symbol.kind != "doc")
        .map(|symbol| graph::RankSymbol {
            id: symbol.id,
            file_id: symbol.file_id,
            name: symbol.name.clone(),
            pagerank: symbol.pagerank,
        })
        .collect();
    let edges = snapshot
        .edges
        .iter()
        .filter_map(|edge| {
            let kind = match edge.kind {
                GraphEdgeKind::Ref | GraphEdgeKind::TestOf => graph::EdgeKind::Ref,
                GraphEdgeKind::Import => graph::EdgeKind::Import,
                GraphEdgeKind::DocMention => return None,
            };
            Some(graph::RankEdge {
                src_symbol: edge.src_symbol,
                dst_name: edge.dst_name.clone(),
                dst_symbol: edge.dst_symbol,
                kind,
            })
        })
        .collect();
    (symbols, edges)
}

/// Respects the configured ceiling and bounds retained parser output even when
/// automatic discovery uses a larger machine-wide parallelism value.
fn preparation_width(settings: &Config, total: usize) -> usize {
    preparation_width_for(
        settings.limits.index_threads,
        std::thread::available_parallelism().map_or(1, usize::from),
        total,
    )
}

/// Automatic indexing leaves one core available; explicit limits include the
/// caller that publishes while workers parse. One-core hosts still make progress.
fn preparation_width_for(configured: u64, available: usize, total: usize) -> usize {
    let requested = if configured == 0 {
        available.saturating_sub(1).max(1)
    } else {
        usize::try_from(configured).unwrap_or(usize::MAX)
    };
    requested.clamp(1, 8).min(total.max(1))
}

/// Keeps a bounded lookahead while one writer publishes in original path order.
/// Workers own no SQLite connection; database reads and publication checks stay on the caller.
fn prepare_files<'a>(
    processor: &sync::l1::Processor<'_>,
    store: &Store,
    inputs: &[sync::l1::FileInput<'a>],
    width: usize,
    mut publish: impl FnMut(sync::l1::FileInput<'a>, sync::l1::PreparedFile<'a>) -> Result<(), Error>,
) -> Result<(), Error> {
    use std::{
        collections::VecDeque,
        sync::{Mutex, mpsc},
    };
    type Prepared<'a> = Result<sync::l1::PreparedFile<'a>, sync::Error>;
    type Job<'a> = (
        sync::l1::FileInput<'a>,
        BTreeSet<store::l1::OccurrenceKey>,
        mpsc::SyncSender<Prepared<'a>>,
    );
    let width = width.clamp(1, 8).min(inputs.len().max(1));
    let retained = |input: sync::l1::FileInput<'a>| -> Result<_, Error> {
        input.work.check()?;
        Ok(store.file_occurrence_keys_checked(input.fence, &|| {
            input
                .work
                .check()
                .map_err(|_| store::Error::L1InputsChanged)
        })?)
    };
    if width == 1 {
        for input in inputs.iter().copied() {
            let prepared = processor
                .prepare_file(input, retained(input)?)
                .map_err(|source| Error::File {
                    path: input.file.path.clone(),
                    source: Box::new(source),
                })?;
            publish(input, prepared)?;
        }
        return Ok(());
    }
    let (send, receive) = mpsc::channel::<Job<'a>>();
    let receive = Mutex::new(receive);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut result = Ok(());
        // Reserve one indexing thread for the active publisher.
        for _ in 0..width - 1 {
            let receive = &receive;
            match std::thread::Builder::new()
                .name("dkc-prepare".into())
                .spawn_scoped(scope, move || {
                    loop {
                        let job = receive
                            .lock()
                            .map_err(|_| Error::PreparationWorkerPanicked)?
                            .recv();
                        let Ok((input, existing, reply)) = job else {
                            break;
                        };
                        // Release the queue lock before parsing. Each job owns one
                        // single-slot reply, so an abandoned caller cannot strand a sender.
                        if reply.send(processor.prepare_file(input, existing)).is_err() {
                            break;
                        }
                    }
                    Ok::<_, Error>(())
                }) {
                Ok(handle) => handles.push(handle),
                Err(error) => {
                    result = Err(Error::PreparationThread(error));
                    break;
                }
            }
        }
        if result.is_ok() {
            result = (|| {
                let enqueue = |input: sync::l1::FileInput<'a>| {
                    let existing = retained(input)?;
                    let (reply, ready) = mpsc::sync_channel(1);
                    send.send((input, existing, reply))
                        .map_err(|_| Error::PreparationWorkerPanicked)?;
                    Ok::<_, Error>((input, ready))
                };
                let mut next = inputs.iter().copied();
                let mut pending = VecDeque::with_capacity(width);
                for input in next.by_ref().take(width) {
                    pending.push_back(enqueue(input)?);
                }
                while let Some((input, ready)) = pending.pop_front() {
                    let prepared = ready
                        .recv()
                        .map_err(|_| Error::PreparationWorkerPanicked)?
                        .map_err(|source| Error::File {
                            path: input.file.path.clone(),
                            source: Box::new(source),
                        })?;
                    publish(input, prepared)?;
                    // Refill only after publication drops its owned parser output:
                    // prepared + in-flight + publishing files never exceed width.
                    if let Some(input) = next.next() {
                        pending.push_back(enqueue(input)?);
                    }
                }
                Ok(())
            })();
        }
        // Close input before joining, including every early/error path. Dropped
        // reply receivers let queued workers finish without blocked delivery.
        drop(send);
        for handle in handles {
            let joined = handle
                .join()
                .unwrap_or(Err(Error::PreparationWorkerPanicked));
            if result.is_ok() {
                result = joined;
            }
        }
        result
    })
}

#[cfg(test)]
fn prepare_batch<'a>(
    processor: &sync::l1::Processor<'_>,
    store: &Store,
    inputs: &[sync::l1::FileInput<'a>],
) -> Result<Vec<sync::l1::PreparedFile<'a>>, Error> {
    let mut ready = Vec::new();
    prepare_files(processor, store, inputs, inputs.len().max(1), |_, file| {
        ready.push(file);
        Ok(())
    })?;
    Ok(ready)
}

/// Rechecks the complete effective L1 configuration before publication.
/// Reuses validation only while the current captured layers remain identical.
/// Root resolution is repeated even when the configuration bytes are unchanged.
struct ConfigurationFence<'a> {
    config_root: &'a Path,
    expected_root: &'a Path,
    expected: &'a Config,
    work: Work<'a>,
    cached: std::cell::RefCell<Option<(config::reload::Snapshot, std::path::PathBuf)>>,
}
impl<'a> ConfigurationFence<'a> {
    fn new(
        config_root: &'a Path,
        expected_root: &'a Path,
        expected: &'a Config,
        work: Work<'a>,
    ) -> Self {
        Self {
            config_root,
            expected_root,
            expected,
            work,
            cached: Default::default(),
        }
    }

    fn check(&self) -> Result<(), Error> {
        self.work.check()?;
        let started = std::time::Instant::now();
        let inputs = config::reload::Snapshot::capture(self.config_root)?;
        let known = self
            .cached
            .borrow()
            .as_ref()
            .and_then(|(old, path)| (old == &inputs).then(|| path.clone()));
        let configured = if let Some(path) = known {
            path
        } else {
            let current = inputs.load()?;
            if sync::l1::settings_fingerprint(&current)
                != sync::l1::settings_fingerprint(self.expected)
                || current.enrich.top_symbols != self.expected.enrich.top_symbols
                || current.commits != self.expected.commits
                || current.limits.max_commits != self.expected.limits.max_commits
                || current.limits.max_prs != self.expected.limits.max_prs
                || current.limits.max_pr_commits != self.expected.limits.max_pr_commits
            {
                return Err(Error::ConfigurationChanged);
            }
            let configured = std::path::PathBuf::from(&current.index.root);
            if configured.is_absolute() {
                configured
            } else {
                self.config_root.join(configured)
            }
        };
        let resolved =
            std::fs::canonicalize(&configured).map_err(|_| Error::ConfigurationChanged)?;
        let expected_root = crate::store::path::system_path(self.expected_root);
        if resolved != expected_root {
            return Err(Error::ConfigurationChanged);
        }
        self.work.check()?;
        *self.cached.borrow_mut() = Some((inputs, configured));
        tracing::debug!(
            elapsed_us = started.elapsed().as_micros() as u64,
            "Verified L1 configuration"
        );
        Ok(())
    }
}

#[cfg(test)]
fn verify_configuration(
    config_root: &Path,
    expected_root: &Path,
    expected: &Config,
    work: Work<'_>,
) -> Result<(), Error> {
    ConfigurationFence::new(config_root, expected_root, expected, work).check()
}

/// Converts discovery limits without weakening platform or configured caps.
pub(crate) fn limits(settings: &Config) -> Result<SyncLimits, Error> {
    Ok(SyncLimits {
        max_files: usize::try_from(settings.limits.max_files)
            .map_err(|_| Error::InvalidLimit("max_files"))?,
        max_file_bytes: settings.limits.max_file_bytes()?.0,
        threads: usize::try_from(settings.limits.index_threads)
            .map_err(|_| Error::InvalidLimit("index_threads"))?,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn configuration_fence_rejects_changed_and_malformed_inputs_then_recovers() {
        let (root, settings, _store) = fixture(None);
        let fence = super::ConfigurationFence::new(
            root.path(),
            root.path(),
            &settings,
            super::Work::default(),
        );
        fence.check().unwrap();
        fence.check().unwrap();
        let path = root.path().join(".deepkit/config.toml");
        let original = std::fs::read(&path).unwrap();
        let mut changed = settings.clone();
        changed.chunk.max_tokens += 1;
        std::fs::write(&path, changed.dump().unwrap()).unwrap();
        assert!(matches!(
            fence.check(),
            Err(super::Error::ConfigurationChanged)
        ));
        std::fs::write(&path, "invalid = [").unwrap();
        assert!(fence.check().is_err());
        std::fs::write(&path, original).unwrap();
        fence.check().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn configuration_fence_rechecks_root_symlink_without_configuration_edits() {
        let (root, mut settings, _store) = fixture(None);
        let first = root.path().join("first");
        let second = root.path().join("second");
        std::fs::create_dir(&first).unwrap();
        std::fs::create_dir(&second).unwrap();
        let link = root.path().join("linked");
        std::os::unix::fs::symlink(&first, &link).unwrap();
        settings.index.root = "linked".into();
        std::fs::write(
            root.path().join(".deepkit/config.toml"),
            settings.dump().unwrap(),
        )
        .unwrap();
        let fence =
            super::ConfigurationFence::new(root.path(), &first, &settings, super::Work::default());
        fence.check().unwrap();
        std::fs::remove_file(&link).unwrap();
        std::os::unix::fs::symlink(&second, &link).unwrap();
        assert!(matches!(
            fence.check(),
            Err(super::Error::ConfigurationChanged)
        ));
    }
    #[cfg(unix)]
    #[test]
    fn configuration_fence_rejects_symlink_substituted_for_expected_root() {
        let (root, mut settings, _store) = fixture(None);
        let expected = root.path().join("first");
        let replacement = root.path().join("second");
        std::fs::create_dir(&expected).unwrap();
        std::fs::create_dir(&replacement).unwrap();
        settings.index.root = "first".into();
        std::fs::write(
            root.path().join(".deepkit/config.toml"),
            settings.dump().unwrap(),
        )
        .unwrap();
        let fence = super::ConfigurationFence::new(
            root.path(),
            &expected,
            &settings,
            super::Work::default(),
        );
        fence.check().unwrap();
        std::fs::rename(&expected, root.path().join("retired")).unwrap();
        std::os::unix::fs::symlink(&replacement, &expected).unwrap();
        assert!(matches!(
            fence.check(),
            Err(super::Error::ConfigurationChanged)
        ));
    }

    use super::*;
    use std::{
        fs,
        sync::atomic::{AtomicBool, Ordering},
    };

    fn fixture(max_files: Option<u64>) -> (tempfile::TempDir, Config, Store) {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join(".deepkit")).unwrap();
        let mut settings = Config::default();
        settings.embedder.profile = "local".into();
        settings.llm.enabled = false;
        if let Some(max_files) = max_files {
            settings.limits.max_files = max_files;
        }
        fs::write(
            root.path().join(".deepkit/config.toml"),
            settings.dump().unwrap(),
        )
        .unwrap();
        for name in ["alpha", "beta", "gamma"] {
            fs::write(
                root.path().join(format!("{name}.rs")),
                format!("pub fn {name}() -> bool {{ true }}\n"),
            )
            .unwrap();
        }
        let settings = config::load(root.path(), None).unwrap().config;
        let store = store::open(
            root.path().join(".deepkit/code.db"),
            store::StoreLimits::default(),
        )
        .unwrap();
        (root, settings, store)
    }

    #[test]
    fn parallel_preparation_preserves_order_and_publication_fences() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let (root, settings, store) = fixture(None);
        let scan = sync::scan(root.path(), Default::default()).unwrap();
        let published = sync::l1::publish(
            &store,
            sync::l1::PublishInput {
                work: Work::default(),
                scan: &scan,
                git: None,
                snapshot: &store.index_snapshot().unwrap(),
                generation: store.l1_generation().unwrap(),
                root: root.path(),
                settings: &settings,
                full: false,
            },
        )
        .unwrap();
        let snapshot = store.index_snapshot().unwrap();
        let cancelled = AtomicBool::new(false);
        let inputs = snapshot
            .files
            .iter()
            .map(|record| sync::l1::FileInput {
                file: scan.file(&record.path).unwrap(),
                work: Work {
                    deadline: None,
                    cancelled: Some(&cancelled),
                },
                fence: store::l1::Fence {
                    generation: published.generation,
                    file_id: record.id,
                    path: &record.path,
                    content_hash: &record.blake3,
                    fingerprint: &published.fingerprints[&record.path],
                },
            })
            .collect::<Vec<_>>();
        assert_eq!(
            inputs
                .iter()
                .map(|i| i.file.path.as_str())
                .collect::<Vec<_>>(),
            ["alpha.rs", "beta.rs", "gamma.rs"]
        );
        let processor = sync::l1::Processor::new(root.path(), &settings).unwrap();
        let mut attempted = 0;
        assert!(matches!(
            prepare_files(&processor, &store, &inputs, 2, |_, _| {
                attempted += 1;
                Err(Error::ConfigurationChanged)
            }),
            Err(Error::ConfigurationChanged)
        ));
        assert_eq!(attempted, 1, "refusal must stop ordered publication");
        // Refusal joins the already-started workers; a fresh call can proceed.
        let mut ready = prepare_batch(&processor, &store, &inputs)
            .unwrap()
            .into_iter();
        let units = || {
            store
                .writer()
                .unwrap()
                .query_row("SELECT count(*) FROM units", [], |r| r.get::<_, i64>(0))
                .unwrap()
        };
        assert_eq!(units(), 0, "preparation must not publish");
        fs::write(
            root.path().join("alpha.rs"),
            "pub fn changed_after_preparation() {}\n",
        )
        .unwrap();
        assert!(
            processor
                .publish_prepared(&store, ready.next().unwrap(), || Ok(()))
                .is_err()
        );
        assert_eq!(units(), 0);
        cancelled.store(true, Ordering::Release);
        assert!(matches!(
            processor.publish_prepared(&store, ready.next().unwrap(), || Ok(())),
            Err(sync::Error::WorkStopped)
        ));
        cancelled.store(false, Ordering::Release);
        fs::write(root.path().join(".deepkit/config.toml"), "invalid = [").unwrap();
        assert!(
            processor
                .publish_prepared(&store, ready.next().unwrap(), || {
                    verify_configuration(root.path(), root.path(), &settings, Work::default())
                        .map_err(|_| store::Error::L1InputsChanged)
                })
                .is_err()
        );
        assert_eq!(units(), 0);
        assert!(ready.next().is_none());
        fs::write(
            root.path().join(".deepkit/config.toml"),
            settings.dump().unwrap(),
        )
        .unwrap();
        assert!(
            run(
                Input {
                    config_root: root.path(),
                    root: root.path(),
                    settings: &settings,
                    store: &store,
                    full: false,
                    work: Work::default()
                },
                |_| {}
            )
            .unwrap()
            .unit_count
                > 0
        );
        assert!(store.pending_l1_paths().unwrap().is_empty());
    }

    #[test]
    fn serial_and_parallel_preparation_publish_identical_l1() {
        let mut snapshots = Vec::new();
        for threads in [1, 4] {
            let (root, mut settings, store) = fixture(None);
            settings.limits.index_threads = threads;
            settings.index.repo_name = "parallel-fixture".into();
            fs::write(
                root.path().join(".deepkit/config.toml"),
                settings.dump().unwrap(),
            )
            .unwrap();
            for number in 0..17 {
                fs::write(
                    root.path().join(format!("extra{number:02}.rs")),
                    format!("pub fn extra_{number}() -> bool {{ true }}\n"),
                )
                .unwrap();
            }
            assert_eq!(preparation_width(&settings, 20), threads as usize);
            run(
                Input {
                    config_root: root.path(),
                    root: root.path(),
                    settings: &settings,
                    store: &store,
                    full: false,
                    work: Work::default(),
                },
                |_| {},
            )
            .unwrap();
            let writer = store.writer().unwrap();
            let mut snapshot = Vec::<Vec<rusqlite::types::Value>>::new();
            for query in [
                "SELECT id,hash,content_hash,source_key,duplicate_ordinal,file_id,kind,line_start,line_end,scope_path,header FROM units ORDER BY id",
                "SELECT * FROM symbols ORDER BY id",
                "SELECT * FROM edges ORDER BY rowid",
                "SELECT rowid,* FROM fts_names ORDER BY rowid",
            ] {
                let mut statement = writer.prepare(query).unwrap();
                let columns = statement.column_count();
                snapshot.extend(
                    statement
                        .query_map([], |r| (0..columns).map(|i| r.get(i)).collect())
                        .unwrap()
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap(),
                );
            }
            snapshots.push(snapshot);
        }
        assert_eq!(snapshots[0], snapshots[1]);
        let mut settings = Config::default();
        settings.limits.index_threads = 99;
        assert_eq!(preparation_width(&settings, 100), 8);
        assert_eq!(preparation_width(&settings, 2), 2);
        // The automatic contract is cores minus one, including small machines;
        // explicit limits remain authoritative even on an oversubscribed host.
        assert_eq!(preparation_width_for(0, 1, 20), 1);
        assert_eq!(preparation_width_for(0, 2, 20), 1);
        assert_eq!(preparation_width_for(0, 8, 20), 7);
        assert_eq!(preparation_width_for(0, 64, 20), 8);
        assert_eq!(preparation_width_for(4, 2, 20), 4);
    }

    #[test]
    fn same_blob_checkout_retags_without_rechunking_even_after_a_stat_change() {
        let (root, settings, store) = fixture(None);
        let git = |args: &[&str]| {
            let output = std::process::Command::new("git")
                .arg("-C")
                .arg(root.path())
                .args([
                    "-c",
                    "user.name=Fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "-c",
                    "commit.gpgsign=false",
                ])
                .args(args)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            String::from_utf8(output.stdout).unwrap().trim().to_owned()
        };
        git(&["init", "--initial-branch=main"]);
        git(&["add", "alpha.rs", "beta.rs", "gamma.rs"]);
        git(&["commit", "-m", "original"]);
        std::thread::sleep(std::time::Duration::from_millis(10));
        git(&["update-index", "--refresh"]);
        let input = || Input {
            config_root: root.path(),
            root: root.path(),
            settings: &settings,
            store: &store,
            full: false,
            work: Work::default(),
        };
        assert!(run(input(), |_| {}).unwrap().unit_count > 0);
        let ids = || {
            store
                .writer()
                .unwrap()
                .prepare("SELECT id FROM units WHERE file_id IS NOT NULL ORDER BY id")
                .unwrap()
                .query_map([], |r| r.get::<_, i64>(0))
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        };
        let original_ids = ids();
        store.writer().unwrap().execute_batch("CREATE TRIGGER reject_rechunk BEFORE INSERT ON units WHEN NEW.file_id IS NOT NULL BEGIN SELECT RAISE(ABORT,'unexpected unit rewrite'); END;").unwrap();
        git(&["checkout", "-b", "alternate"]);
        git(&["commit", "--allow-empty", "-m", "same blobs"]);
        let next_head = git(&["rev-parse", "HEAD"]);
        let path = root.path().join("alpha.rs");
        fs::write(&path, fs::read(&path).unwrap()).unwrap();
        let report = run(input(), |_| {}).unwrap();
        assert_eq!(report.unit_count, 0);
        assert_eq!(report.scan.status.hashes_reused_from_git, 2);
        assert_eq!(ids(), original_ids);
        assert_eq!(
            store.get_meta("indexed_head").unwrap().as_deref(),
            Some(next_head.as_str())
        );
        assert!(store.pending_l1_paths().unwrap().is_empty());
    }

    #[test]
    fn cancelled_file_pass_retains_completed_work_and_resumes_to_completion() {
        let (root, settings, store) = fixture(None);
        let cancelled = AtomicBool::new(false);
        let first = run(
            Input {
                config_root: root.path(),
                root: root.path(),
                settings: &settings,
                store: &store,
                full: false,
                work: Work {
                    cancelled: Some(&cancelled),
                    deadline: None,
                },
            },
            |progress| {
                if matches!(progress, Progress::Files { processed: 1, .. }) {
                    cancelled.store(true, Ordering::Release);
                }
            },
        );
        assert!(matches!(first, Err(Error::Sync(sync::Error::WorkStopped))));
        assert!(!store.pending_l1_paths().unwrap().is_empty());
        assert!(store.counts().unwrap().units > 0);
        assert_eq!(
            store
                .get_meta("l1_completed_generation")
                .unwrap()
                .as_deref(),
            Some("0")
        );
        let mut completed = false;
        let resumed = run(
            Input {
                config_root: root.path(),
                root: root.path(),
                settings: &settings,
                store: &store,
                full: false,
                work: Work::default(),
            },
            |progress| {
                if let Progress::Finished { complete } = progress {
                    completed = complete;
                }
            },
        )
        .unwrap();
        assert!(completed);
        assert!(resumed.scan.is_complete());
        assert!(store.pending_l1_paths().unwrap().is_empty());
        assert_eq!(
            store.get_meta("l1_completed_generation").unwrap(),
            Some(resumed.publication.generation.to_string())
        );
        assert_eq!(store.counts().unwrap().files, 3);
        assert_eq!(
            store.get_meta("graph_pending").unwrap().as_deref(),
            Some("0")
        );
    }

    #[test]
    fn discovery_cancellation_and_capped_scan_never_claim_completion() {
        let (root, settings, store) = fixture(Some(1));
        let generation = store.l1_generation().unwrap();
        let cancelled = AtomicBool::new(false);
        let result = run(
            Input {
                config_root: root.path(),
                root: root.path(),
                settings: &settings,
                store: &store,
                full: false,
                work: Work {
                    cancelled: Some(&cancelled),
                    deadline: None,
                },
            },
            |progress| {
                if matches!(progress, Progress::Discovering) {
                    cancelled.store(true, Ordering::Release);
                }
            },
        );
        assert!(matches!(result, Err(Error::Sync(sync::Error::WorkStopped))));
        assert_eq!(store.l1_generation().unwrap(), generation);
        assert_eq!(store.counts().unwrap().files, 0);
        let mut completed = None;
        let report = run(
            Input {
                config_root: root.path(),
                root: root.path(),
                settings: &settings,
                store: &store,
                full: false,
                work: Work::default(),
            },
            |progress| {
                if let Progress::Finished { complete } = progress {
                    completed = Some(complete);
                }
            },
        )
        .unwrap();
        assert_eq!(completed, Some(false));
        assert!(!report.scan.is_complete());
        assert_eq!(
            store
                .get_meta("l1_completed_generation")
                .unwrap()
                .as_deref(),
            Some("0")
        );
    }
}
