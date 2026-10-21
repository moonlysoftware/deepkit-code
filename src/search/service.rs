//! Shared CLI/MCP repository tools and explicit fallback; see `v1.md` §9–10.1.
mod links;
pub(crate) mod recent;
mod response_limits;
mod symbol_history;

use super::{navigation, pipeline};
use crate::{
    config::{self, Config},
    queue,
    store::{self, Store, StoreLimits},
    sync,
};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use std::sync::atomic::{AtomicBool, Ordering};

type Result<T> = std::result::Result<T, Error>;

/// Repository tool failures retain typed causes without exposing them to clients.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("Git history is not indexed")]
    HistoryNotIndexed,
    #[error(transparent)]
    Config(#[from] config::Error),
    #[error(transparent)]
    Store(#[from] store::Error),
    #[error(transparent)]
    Search(#[from] super::Error),
    #[error(transparent)]
    Sync(#[from] sync::Error),
    #[error(transparent)]
    Queue(#[from] queue::Error),
    #[error(transparent)]
    Status(#[from] super::status::Error),
    #[error("could not resolve repository directory")]
    Root(#[source] std::io::Error),
    #[error("could not serialize tool response")]
    Json(#[from] serde_json::Error),
    #[error("system clock is outside the supported Unix timestamp range")]
    Clock,
    #[error("configured open limit exceeds this platform's address range")]
    OpenLimit,
}

impl Error {
    /// Stable application code, independent of provider messages or filesystem paths.
    pub(crate) fn code(&self) -> &'static str {
        match self {
            Self::HistoryNotIndexed => "history_not_indexed",
            Self::Search(super::Error::NotFound { .. }) => "not_found",
            Self::Search(super::Error::AmbiguousSymbol { .. }) => "ambiguous_symbol",
            Self::Search(super::Error::Deadline) => "timeout",
            Self::Search(super::Error::OpenSourceUnavailable) => "source_unavailable",
            Self::Search(
                super::Error::InvalidQuery { .. }
                | super::Error::InvalidK { .. }
                | super::Error::InvalidReferenceLimit
                | super::Error::InvalidMapBudget,
            ) => "invalid_arguments",
            Self::Config(_)
            | Self::OpenLimit
            | Self::Status(super::status::Error::Config(_) | super::status::Error::Profile) => {
                "invalid_configuration"
            }
            Self::Root(_) | Self::Sync(_) => "repository_unavailable",
            _ => "internal_error",
        }
    }

    /// Safe client message that never copies source text, credentials or backend errors.
    pub(crate) fn message(&self) -> &'static str {
        match self.code() {
            "history_not_indexed" => "Git commits are not indexed yet.",
            "not_found" => "No indexed result matches the requested target.",
            "ambiguous_symbol" => "The symbol identifier matches more than one location.",
            "timeout" => "The query deadline was exceeded.",
            "source_unavailable" => {
                "The source cannot be verified against its completed index generation."
            }
            "invalid_arguments" => "The tool arguments are invalid.",
            "invalid_configuration" => "The repository configuration is invalid.",
            "repository_unavailable" => "The repository could not be inspected.",
            _ => "The repository tool could not complete.",
        }
    }

    /// Actionable bounded guidance suitable for CLI JSON and MCP error responses.
    pub(crate) fn hint(&self) -> &'static str {
        match self.code() {
            "not_found" => "Check the target and index the repository before retrying.",
            "ambiguous_symbol" => "Use a repository-relative path and line range.",
            "timeout" => "Narrow the query or inspect the configured query timeout.",
            "source_unavailable" => {
                "Index the repository to repair changed, missing or pending sources."
            }
            "invalid_arguments" => "Check the tool schema and configured result limits.",
            "invalid_configuration" => {
                "Inspect the effective configuration and correct invalid settings."
            }
            "repository_unavailable" => {
                "Check that the configured repository directory is accessible."
            }
            _ => "Inspect index status and local diagnostics before retrying.",
        }
    }
}

impl From<rusqlite::Error> for Error {
    fn from(error: rusqlite::Error) -> Self {
        Self::Store(store::Error::from(error))
    }
}

/// One invocation's effective configuration and shared database, without model defaults.
#[derive(Clone)]
pub(crate) struct Context {
    pub(crate) config_root: PathBuf,
    pub(crate) root: PathBuf,
    pub(crate) config: Config,
    pub(crate) store: Arc<Store>,
    query_host: Arc<std::sync::Mutex<Option<Arc<super::host::Host>>>>,
    pub(crate) query_latencies: Arc<super::latency::History>,
    /// Shared process-local bootstrap activity; prevents competing lazy publication.
    pub(crate) background_l1: Arc<AtomicBool>,
}

// The invocation owns cancellation; pipeline stages borrow the same authority.
type QueryWork = super::control::Owner;

impl QueryWork {
    async fn blocking<T: Send + 'static>(
        &self,
        operation: impl FnOnce(sync::l1::Work<'_>) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        check_query_work(sync::l1::Work {
            deadline: Some(self.deadline),
            cancelled: Some(&self.cancelled),
        })?;
        let deadline = self.deadline;
        let cancelled = Arc::clone(&self.cancelled);
        tokio::task::spawn_blocking(move || {
            let work = sync::l1::Work {
                deadline: Some(deadline),
                cancelled: Some(&cancelled),
            };
            check_query_work(work)?;
            let result = operation(work);
            check_query_work(work)?;
            result
        })
        .await
        .map_err(|_| Error::Search(super::Error::QueryTask))?
    }
}

fn check_query_work(work: sync::l1::Work<'_>) -> Result<()> {
    work.check()
        .map_err(|_| Error::Search(super::Error::Deadline))
}

/// Presents verified histories and repairs stale provenance once on the lazy path.
/// Healthy requests stay read-only; the final response fence never writes.
fn populate_symbol_histories(
    context: &Context,
    symbols: &mut [navigation::SymbolResult],
    selected: &std::collections::BTreeSet<i64>,
    work: sync::l1::Work<'_>,
    repair: bool,
) -> Result<()> {
    match symbol_history::populate_many(context, symbols, selected, work) {
        Ok(()) => Ok(()),
        Err(Error::Search(super::Error::OpenSourceUnavailable)) if repair => {
            sync::symbol_history::repair_selected(
                &context.root,
                &context.config,
                &context.store,
                work,
                &|| {
                    super::repair::validate_configuration(context, work)
                        .map_err(|_| store::Error::L1InputsChanged)
                },
                selected,
            )
            .map_err(|_| super::Error::OpenSourceUnavailable)?;
            symbol_history::populate_many(context, symbols, selected, work)
        }
        Err(error) => Err(error),
    }
}

/// Checks retained hidden contributors against canonical current L1 settings.
/// Uses indexed hashes only: hidden files do not expand filesystem repair scope.
fn query_hidden_publications(
    context: &Context,
    selection: &super::Selection,
    work: sync::l1::Work<'_>,
) -> Result<Option<std::collections::BTreeSet<i64>>> {
    let mut incomplete = std::collections::BTreeSet::new();
    check_query_work(work)?;
    if selection.publications.is_empty() {
        return Ok(selection.incomplete_hidden_hits.clone());
    }
    let router = crate::treat::Router::new(&context.config.treatments, context.config.llm.enabled)
        .map_err(sync::Error::from)?;
    let name = sync::l1::repo_name(&context.root, &context.config);
    for (file, publication) in &selection.publications {
        check_query_work(work)?;
        let current = publication.as_ref().is_some_and(|source| {
            let Ok(hash) = <&[u8; 32]>::try_from(source.hash.as_slice()) else {
                return false;
            };
            let Some(kind) = store::FileKind::from_db(&source.kind) else {
                return false;
            };
            let treatment = router.route(&source.path, kind);
            let expected = sync::l1::file_fingerprint(
                &context.config,
                &name,
                sync::l1::FingerprintFile {
                    path: &source.path,
                    content_hash: hash,
                    language: source.language.as_deref(),
                    kind,
                    capped: false,
                },
                &treatment,
            );
            source.is_complete() && source.desired.as_deref() == Some(expected.as_slice())
        });
        if !current {
            incomplete.insert(*file);
        }
    }
    check_query_work(work)?;
    if incomplete.is_empty() {
        return Ok(selection.incomplete_hidden_hits.clone());
    }
    let mut affected = selection.incomplete_hidden_hits.clone().unwrap_or_default();
    // Selection retains original match identities before source suppression.
    for hit in selection.selected.as_array().into_iter().flatten() {
        check_query_work(work)?;
        for entry in hit["matches"].as_array().into_iter().flatten() {
            check_query_work(work)?;
            if let Some(links) = entry["unit"]
                .as_i64()
                .and_then(|id| selection.links.get(&id))
            {
                for link in links {
                    check_query_work(work)?;
                    if link.file_id.is_some_and(|file| incomplete.contains(&file))
                        && let Some(id) = hit["unit"].as_i64()
                    {
                        affected.insert(id);
                    }
                }
            }
        }
    }
    check_query_work(work)?;
    Ok(Some(affected))
}

fn query_generation(store: &Store, work: sync::l1::Work<'_>) -> Result<i64> {
    let result =
        store.l1_generation_checked(&|| work.check().map_err(|_| store::Error::L1InputsChanged));
    check_query_work(work)?;
    Ok(result?)
}

fn query_occurrences(
    store: &Store,
    occurrences: &[store::StoredUnit],
    work: sync::l1::Work<'_>,
) -> Result<Vec<bool>> {
    let check = || work.check().map_err(|_| store::Error::L1InputsChanged);
    let result = store.checked_read_snapshot(&check, |snapshot| {
        occurrences
            .iter()
            .map(|occurrence| {
                check()?;
                store::occurrence_matches_in(snapshot, occurrence)
            })
            .collect::<std::result::Result<Vec<_>, store::Error>>()
    });
    check_query_work(work)?;
    Ok(result??)
}

#[derive(PartialEq)]
enum HitSource {
    File(store::OpenSource),
    History {
        content: [u8; 32],
        policy: [u8; 32],
        context: crate::treat::context::EmbeddingInput,
        max_level: i64,
        protected: bool,
    },
}

fn query_hit_source(
    context: &Context,
    occurrence: &store::StoredUnit,
    hit: &Value,
    work: sync::l1::Work<'_>,
) -> Result<std::result::Result<(Option<u64>, HitSource), &'static str>> {
    if !matches!(
        occurrence.kind.as_str(),
        "commit" | "pr" | "history" | "module"
    ) || occurrence.path.is_some()
    {
        return Ok(navigation::hit_authority(
            &context.store,
            &context.root,
            hit,
            context.config.limits.max_file_bytes()?.0,
            work,
        )?
        .map(|(age, source)| (age, HitSource::File(source))));
    }
    let result = context.store.checked_read_snapshot(
        &|| work.check().map_err(|_| store::Error::L1InputsChanged),
        |tx| {
            queue::source::resolve_freshness(
                tx,
                &context.root,
                &context.config_root,
                &context.config,
                occurrence.id,
                work,
            )
        },
    );
    check_query_work(work)?;
    Ok(match result?? {
        Some(input) => Ok((
            None,
            HitSource::History {
                content: input.occurrence.content_hash,
                policy: input.policy.fingerprint,
                context: input.context,
                max_level: input.policy.max_level,
                protected: input.policy.never_send,
            },
        )),
        None => Err("history_source_unavailable"),
    })
}

// All labels in one observation share graph/source reads. Recreate this batch
// after the response's final observation boundary; it never caches freshness.
fn map_contains_module(mapped: &super::map::MapResult, path: &str) -> bool {
    mapped.nodes.iter().any(|node| {
        path.is_empty()
            || node
                .path
                .strip_prefix(path)
                .is_some_and(|tail| tail.starts_with('/'))
    })
}

fn query_module_sources(
    context: &Context,
    units: &[i64],
    work: sync::l1::Work<'_>,
    cache: &mut Option<(store::modules::Revision, queue::source::ModuleSnapshotCache)>,
) -> Result<std::collections::BTreeMap<i64, HitSource>> {
    let check = || work.check().map_err(|_| store::Error::L1InputsChanged);
    let before = context.store.revision_checked(&check)?;
    let mut snapshots = cache
        .take()
        .and_then(|(revision, snapshots)| (revision == before).then_some(snapshots));
    let result = context.store.checked_read_snapshot(&check, |tx| {
        queue::source::resolve_module_freshness_batch_with_cache(
            tx,
            &context.root,
            &context.config_root,
            &context.config,
            units,
            work,
            &mut snapshots,
        )
    });
    check_query_work(work)?;
    let inputs = result??;
    let after = context.store.revision_checked(&check)?;
    *cache = if before == after {
        snapshots.map(|snapshots| (after, snapshots))
    } else {
        None
    };
    Ok(inputs
        .into_iter()
        .map(|(id, input)| {
            (
                id,
                HitSource::History {
                    content: input.occurrence.content_hash,
                    policy: input.policy.fingerprint,
                    context: input.context,
                    max_level: input.policy.max_level,
                    protected: input.policy.never_send,
                },
            )
        })
        .collect())
}

#[cfg(test)]
fn query_l2(
    context: &Context,
    occurrence: &store::StoredUnit,
    work: sync::l1::Work<'_>,
) -> Result<Option<&'static str>> {
    Ok(query_l2_many(context, &[occurrence], work)?.remove(0))
}

fn query_l2_many(
    context: &Context,
    occurrences: &[&store::StoredUnit],
    work: sync::l1::Work<'_>,
) -> Result<Vec<Option<&'static str>>> {
    use queue::destination::FileFreshness;
    use store::embedding_freshness::State;
    let result = queue::destination::search_occurrences_freshness(
        &context.store,
        &context.root,
        &context.config_root,
        &context.config,
        occurrences,
        work,
    );
    // Queue reader admission uses its own typed cancellation. Preserve the
    // public query timeout contract instead of exposing an internal error.
    check_query_work(work)?;
    Ok(result?
        .into_iter()
        .map(|state| {
            state.map(|state| match state {
                FileFreshness::Ineligible => "none",
                FileFreshness::Representation(State::Missing | State::Stale) => "pending",
                FileFreshness::Representation(State::Current) => "current",
            })
        })
        .collect())
}

enum SearchEvidence {
    Lexical(Box<super::Selection>),
    Pipeline(Box<pipeline::Results>),
}
impl SearchEvidence {
    fn selection(&self) -> &super::Selection {
        match self {
            Self::Lexical(selection) => selection,
            Self::Pipeline(results) => &results.selection,
        }
    }
    fn is_current(
        &self,
        store: &Store,
        parsed: &super::query::Parsed,
        request: &pipeline::Request,
        settings: &config::Search,
        check: &dyn Fn() -> std::result::Result<(), super::Error>,
    ) -> std::result::Result<bool, super::Error> {
        match self {
            Self::Lexical(selection) => super::lexical_with_selection(
                store,
                parsed,
                request.k,
                request.scope.as_ref(),
                settings,
                &request.exclude_ids,
                check,
            )
            .map(|(_, current)| current == **selection),
            Self::Pipeline(results) => {
                results.selection_is_current(store, request, settings, check)
            }
        }
    }
    fn dense_state(&self, requested: bool) -> &'static str {
        if !requested {
            return "not_requested";
        }
        match self {
            Self::Lexical(_) => "unavailable",
            Self::Pipeline(results) => match results.dense {
                pipeline::DenseState::Complete => "complete",
                pipeline::DenseState::NotRequested => "not_requested",
                pipeline::DenseState::Partial(_) => "partial",
                _ => "unavailable",
            },
        }
    }
    fn rerank_state(&self, requested: bool, config: &Config) -> &'static str {
        if !requested || config.rerank.mode == "off" {
            return "off";
        }
        match self {
            Self::Lexical(_) => "unavailable",
            Self::Pipeline(results) => match results.rerank {
                super::rerank::State::Complete => "complete",
                super::rerank::State::Preparing => "preparing",
                super::rerank::State::Privacy => "privacy",
                super::rerank::State::Timeout => "timeout",
                super::rerank::State::NotRequested if results.hits.is_empty() => "not_requested",
                _ => "unavailable",
            },
        }
    }
    fn degraded(&self, requested: bool, config: &Config) -> bool {
        matches!(self, Self::Pipeline(results) if results.degraded())
            || matches!(
                self.rerank_state(requested, config),
                "unavailable" | "preparing" | "privacy" | "timeout"
            )
    }
    fn lexical_failed(&self) -> bool {
        matches!(self, Self::Pipeline(results) if matches!(results.lexical, pipeline::LexicalState::Failed(_)))
    }
}

struct Inspection {
    head: Option<String>,
    indexed_head: Option<String>,
    pending: i64,
}

impl Inspection {
    fn freshness(&self) -> Value {
        let changed = sync::head::changed(self.indexed_head.as_deref(), self.head.as_deref());
        json!({
            "state": if changed || self.pending > 0 { "stale" } else { "unknown" },
            "reason": if changed { "head_changed" } else if self.pending > 0 { "l1_pending" } else { "source_not_verified" },
        })
    }

    fn annotate(&self, value: &mut Value) {
        // Preserve stronger occurrence-specific evidence from lower-level tools.
        if value["fresh"]["state"]
            .as_str()
            .is_some_and(|state| state != "fresh")
        {
            return;
        }
        value["fresh"] = if let Some(reason) = value["l1_deferred_reason"].as_str() {
            json!({"state": "capped", "reason": reason})
        } else {
            self.freshness()
        };
    }

    fn index(&self) -> Value {
        json!({"head": self.head, "indexed_head": self.indexed_head,
            "dirty_files": null, "dirty_files_state": "not_checked",
            "l1_pending_files": self.pending, "fresh": self.freshness()})
    }
}

impl Context {
    /// Resolves roots and database limits using the same rules as ordinary CLI tools.
    /// Create a new context per invocation so effective configuration is not cached.
    pub(crate) fn load(config_root: &Path) -> Result<Self> {
        let config = config::load(config_root, None)?.config;
        Self::load_with_config(config_root, config)
    }

    /// Opens a context with captured settings and a checked repository database.
    /// Request source/configuration fences still validate those settings normally.
    pub(crate) fn load_with_config(config_root: &Path, config: Config) -> Result<Self> {
        let configured = PathBuf::from(&config.index.root);
        let root = fs::canonicalize(if configured.is_absolute() {
            configured
        } else {
            config_root.join(configured)
        })
        .map_err(Error::Root)?;
        let store = Arc::new(store::open_repository(
            &root,
            Path::new(&config.store.path),
            StoreLimits {
                db_max_bytes: config.limits.db_max_bytes()?.0,
                wal_max_bytes: config.limits.wal_max_bytes()?.0,
                max_l2_units: config.limits.max_l2_units,
                max_l3_units: config.limits.max_l3_units,
            },
        )?);
        Ok(Self {
            config_root: config_root.to_path_buf(),
            root,
            config,
            store,
            query_host: Default::default(),
            query_latencies: Default::default(),
            background_l1: Default::default(),
        })
    }

    fn inspect(&self) -> Result<Inspection> {
        // Status has no QueryWork budget and retains ordinary reader wait policy.
        let indexed_head = self.store.get_meta("indexed_head")?;
        let head = sync::head::read(&self.root, Default::default())?.map(|head| head.id);
        let pending = self.store.counts()?.l1_pending_files;
        Ok(Inspection {
            head,
            indexed_head,
            pending,
        })
    }

    fn inspect_with_work(&self, work: sync::l1::Work<'_>) -> Result<Inspection> {
        check_query_work(work)?;
        let result = self
            .store
            .inspection_checked(&|| work.check().map_err(|_| store::Error::L1InputsChanged));
        check_query_work(work)?;
        let (indexed_head, pending) = result?;
        let head = sync::head::read(&self.root, work)?.map(|head| head.id);
        check_query_work(work)?;
        Ok(Inspection {
            head,
            indexed_head,
            pending,
        })
    }

    // Refresh inspection after every attempted branch pass: a refused pass can
    // still have committed a catalogue or individual files. Never reuse its
    // pre-publication generation for subsequent retrieval.
    fn reconcile_branch(
        &self,
        inspection: &mut Inspection,
        work: sync::l1::Work<'_>,
    ) -> Result<Value> {
        check_query_work(work)?;
        if !sync::head::changed(
            inspection.indexed_head.as_deref(),
            inspection.head.as_deref(),
        ) {
            return Ok(json!({"state": "not_needed"}));
        }
        if self.background_l1.load(Ordering::Acquire) {
            return Ok(json!({"state": "incomplete", "reason": "background_l1"}));
        }
        let outcome = super::repair::branch(self, work);
        check_query_work(work)?;
        *inspection = self.inspect_with_work(work)?;
        Ok(match outcome {
            Ok(Some((progress, completed))) => json!({
                "state": if completed { "completed" } else { "incomplete" },
                "processed_files": progress.processed_files,
                "already_settled_files": progress.already_settled_files,
                "truncated_files": progress.truncated_files,
                "content_capped_files": progress.content_capped_files,
                "discovery_incomplete": progress.discovery_incomplete,
            }),
            Ok(None) => json!({"state": "disabled"}),
            Err(_) => json!({"state": "refused"}),
        })
    }

    // Observe only completed public outcomes. Counts contain no query/source text
    // and follow the existing process-local latency status lifetime.
    fn observe_response_caps(&self, tool: &'static str, result: &Result<Value>) {
        use super::latency::ResponseCap;
        match result {
            Err(Error::Search(super::Error::Deadline)) => self
                .query_latencies
                .record_cap(ResponseCap::QueryTimeout, tool),
            Ok(value) => {
                if value["index"]["reason"] == "query_timeout" {
                    self.query_latencies
                        .record_cap(ResponseCap::QueryTimeout, tool);
                }
                if value["index"]["rerank"] == "timeout"
                    && value["index"]["reason"] != "query_timeout"
                {
                    self.query_latencies
                        .record_cap(ResponseCap::RerankTimeout, tool);
                }
                if value["result_budget"]["truncated"] == true {
                    self.query_latencies
                        .record_cap(ResponseCap::ResultTokens, tool);
                }
                if tool == "open" && value["continuation_id"].is_string() {
                    self.query_latencies
                        .record_cap(ResponseCap::OpenBytes, tool);
                }
            }
            _ => {}
        }
    }

    /// Uses shared retrieval without reranking for explicit diagnostic/benchmark callers.
    pub(crate) async fn search(&self, request: pipeline::Request, verbose: bool) -> Result<Value> {
        self.search_with_rerank(request, verbose, false).await
    }

    /// Runs configured retrieval and optional reranking under one invocation budget.
    /// CLI/MCP pass their per-call rerank choice here, before any provider dispatch.
    pub(crate) async fn search_with_rerank(
        &self,
        request: pipeline::Request,
        verbose: bool,
        rerank: bool,
    ) -> Result<Value> {
        // Public CLI/MCP inputs obey v1 §10.1 even when diagnostics raise max_k.
        // Invalid arguments must not be hidden by an exhausted query deadline.
        let max = self.config.search.max_k.min(50);
        if request.k == 0 || request.k as u128 > u128::from(max) {
            return Err(Error::Search(super::Error::InvalidK { max }));
        }
        let _measurement = self.query_latencies.measure();
        let started = Instant::now();
        let work = QueryWork::new(started, self.config.limits.query_timeout_ms)?;
        let query = request.query.clone();
        let result = match tokio::time::timeout_at(
            tokio::time::Instant::from_std(work.deadline),
            self.search_within_options(request, verbose, started, &work, rerank, || {}),
        )
        .await
        {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(Error::Search(super::Error::Deadline))) | Err(_) => {
                // No completed verification is available. Unknown counters must not
                // look like a complete zero-hit result or a verified HEAD snapshot.
                Ok(json!({
                    "query": query, "hits": [], "took_ms": started.elapsed().as_millis(),
                    "index": {
                        "partial": true, "degraded": true, "reason": "query_timeout",
                        "head": null, "indexed_head": null, "l1_pending_files": null,
                        "dirty_files": null, "dirty_files_state": "not_checked",
                        "suppressed_hits": null, "freshness_refusals": {},
                        "fresh": {"state": "unknown", "reason": "query_timeout"},
                        "lexical": "incomplete", "dense": "unavailable", "reranked": false,
                        "rerank": if rerank && self.config.rerank.mode != "off" { "timeout" } else { "off" },
                        "took_ms": started.elapsed().as_millis()
                    }
                }))
            }
            Ok(Err(error)) => Err(error),
        };
        self.observe_response_caps("search", &result);
        result
    }

    #[cfg(test)]
    async fn search_within_checked(
        &self,
        request: pipeline::Request,
        verbose: bool,
        started: Instant,
        work: &QueryWork,
        after_verification: impl FnOnce() + Send + 'static,
    ) -> Result<Value> {
        self.search_within_options(request, verbose, started, work, false, after_verification)
            .await
    }

    async fn search_within_options(
        &self,
        request: pipeline::Request,
        verbose: bool,
        started: Instant,
        work: &QueryWork,
        rerank: bool,
        after_verification: impl FnOnce() + Send + 'static,
    ) -> Result<Value> {
        tracing::debug!(stage = "started", "search observation");
        let lexical_only = request.mode == super::fusion::Mode::Lexical && !rerank;
        let _session = self.query_session(work).await?;
        tracing::debug!(
            stage = "session",
            elapsed_us = started.elapsed().as_micros(),
            "search observation"
        );
        let context = self.clone();
        let mut inspection = work
            .blocking(move |signal| context.inspect_with_work(signal))
            .await?;
        let context = self.clone();
        let (mut inspection, branch, mut generation) = work
            .blocking(move |signal| {
                let branch = context.reconcile_branch(&mut inspection, signal)?;
                let generation = query_generation(&context.store, signal)?;
                Ok((inspection, branch, generation))
            })
            .await?;
        let results = self.retrieve(&request, rerank, work).await?;
        tracing::debug!(
            stage = "retrieved",
            elapsed_us = started.elapsed().as_micros(),
            "search observation"
        );
        let context = self.clone();
        let store = Arc::clone(&self.store);
        let settings = self.config.search.clone();
        let root = self.root.clone();
        let max_file_bytes = self.config.limits.max_file_bytes()?.0;
        work.blocking(move |signal| {
            let parsed = super::query::parse(&request.query, request.mode)?;
            let retrieve = || -> Result<(Value, Vec<store::StoredUnit>, super::Selection)> {
                check_query_work(signal)?;
                let (hits, selection) = super::lexical_with_selection(
                    &store,
                    &parsed,
                    request.k,
                    request.scope.as_ref(),
                    &settings,
                    &request.exclude_ids,
                    &|| {
                        if signal.check().is_ok() {
                            Ok(())
                        } else {
                            Err(super::Error::Deadline)
                        }
                    },
                )?;
                check_query_work(signal)?;
                let response =
                    super::format_response_verbose(&store, &request.query, &hits, verbose)?;
                Ok((
                    response,
                    hits.into_iter().map(|hit| hit.unit).collect(),
                    selection,
                ))
            };
            // Keep the original model snapshot through source verification. A repair
            // replaces it with local evidence, never another outbound request.
            let (mut response, mut occurrences, mut evidence) = match results {
                Some(results) => (
                    super::format_response_verbose(&store, &request.query, &results.hits, verbose)?,
                    results.hits.iter().map(|hit| hit.unit.clone()).collect(),
                    SearchEvidence::Pipeline(Box::new(results)),
                ),
                None => {
                    let (response, occurrences, selection) = retrieve()?;
                    (response, occurrences, SearchEvidence::Lexical(Box::new(selection)))
                }
            };
            let mut targets = std::collections::BTreeSet::new();
            if settings.lazy_resync
                && let Some(hits) = response["hits"].as_array()
            {
                for hit in hits {
                    check_query_work(signal)?;
                    if matches!(
                        navigation::verify_hit(&store, &root, hit, max_file_bytes, signal)?,
                        Err("source_changed"
                            | "l1_pending_or_unavailable"
                            | "invalid_source_range")
                    ) && let Some(path) = hit["path"].as_str()
                    {
                        targets.insert(path.to_owned());
                    }
                }
            }
            let mut published = 0_usize;
            let mut refused = 0_usize;
            for path in &targets {
                check_query_work(signal)?;
                match super::repair::target(&context, path, signal) {
                    Ok(true) => published += 1,
                    Ok(false) | Err(_) => refused += 1,
                }
                check_query_work(signal)?;
            }
            if !targets.is_empty() {
                // Even a refused completion can have committed a pending catalogue.
                // Never reuse pre-repair IDs, snippets, ranks or pending counters.
                inspection = context.inspect_with_work(signal)?;
                generation = query_generation(&store, signal)?;
                let (new_response, new_occurrences, selection) = retrieve()?;
                response = new_response;
                occurrences = new_occurrences;
                evidence = SearchEvidence::Lexical(Box::new(selection));
            }
            let mut suppressed = std::collections::BTreeMap::<&str, usize>::new();
            let mut retained = Vec::new();
            let mut sources = Vec::new();
            let incomplete_hidden_hits = query_hidden_publications(&context, evidence.selection(), signal)?;
            tracing::debug!(stage = "hidden_sources", elapsed_us = started.elapsed().as_micros(), "search observation");
            if incomplete_hidden_hits.is_some() {
                // Equal initial/final records cannot prove completion when the
                // initial hidden evidence was already pending or unavailable.
                suppressed.insert("contributor_l1_incomplete", 0);
            }
            if let Some(hits) = response["hits"].as_array_mut() {
                let mut verified = Vec::with_capacity(hits.len());
                for (mut hit, occurrence) in std::mem::take(hits).into_iter().zip(occurrences) {
                    check_query_work(signal)?;
                    if incomplete_hidden_hits
                        .as_ref()
                        .is_some_and(|ids| ids.contains(&occurrence.id))
                    {
                        *suppressed.entry("contributor_l1_incomplete").or_default() += 1;
                        continue;
                    }
                    match query_hit_source(&context, &occurrence, &hit, signal)? {
                        Ok((changed_ago_s, source)) => {
                            inspection.annotate(&mut hit);
                            if hit["l1_deferred_reason"].is_null() {
                                hit["fresh"] = json!({"state":"fresh", "l1":"current",
                                    "reason":"source_verified", "changed_ago_s": changed_ago_s});
                            }
                            sources.push(source);
                            retained.push(occurrence);
                            verified.push(hit);
                        }
                        Err(reason) => *suppressed.entry(reason).or_default() += 1,
                    }
                }
                *hits = verified;
            }
            tracing::debug!(stage = "hit_sources", elapsed_us = started.elapsed().as_micros(), "search observation");
            let summary_sources = super::summaries::observe_many(&context, &retained.iter().collect::<Vec<_>>(), signal)?;
            after_verification();
            tracing::debug!(stage = "initial_sources", elapsed_us = started.elapsed().as_micros(), "search observation");
            let settings = if lexical_only {
                super::repair::validate_lexical_configuration(&context, signal).map(|_| ())
            } else {
                super::repair::validate_configuration(&context, signal)
            };
            check_query_work(signal)?;
            if settings.is_err() {
                let removed = response["hits"].as_array().map_or(0, Vec::len);
                response["hits"] = json!([]);
                suppressed.insert("settings_changed", removed);
            }
            let retrieved_head = inspection.head.clone();
            inspection = context.inspect_with_work(signal)?;
            let head_changed = inspection.head != retrieved_head;
            let identities = query_occurrences(&store, &retained, signal)?;
            let occurrence_changed = identities.iter().any(|current| !current);
            if let Some(hits) = response["hits"].as_array_mut() {
                let mut verified = Vec::with_capacity(hits.len());
                let mut verified_occurrences = Vec::with_capacity(hits.len());
                let mut verified_summaries = Vec::with_capacity(hits.len());
                let mut verified_sources = Vec::with_capacity(hits.len());
                for ((((mut hit, current), before), occurrence), summary) in std::mem::take(hits)
                    .into_iter()
                    .zip(identities)
                    .zip(sources)
                    .zip(&retained)
                    .zip(summary_sources)
                {
                    check_query_work(signal)?;
                    if !current {
                        *suppressed.entry("occurrence_changed").or_default() += 1;
                        continue;
                    }
                    // Rehash under the original request budget and compare the
                    // retained completion, without starting another repair pass.
                    match query_hit_source(&context, occurrence, &hit, signal)? {
                        Ok((age, after)) if after == before => {
                            // Publication may progress or be invalidated without a
                            // catalogue generation change. Evaluate L2 here once,
                            // after the final source pass; repeating canonical input
                            // preparation earlier adds work without stronger evidence.
                            // Valid lexical evidence does not require a vector.
                            hit["fresh"]["changed_ago_s"] = json!(age);
                            verified_occurrences.push(occurrence);
                            verified_summaries.push(summary);
                            verified_sources.push(after);
                            verified.push(hit);
                        }
                        Ok(_) => *suppressed.entry("source_publication_changed").or_default() += 1,
                        Err("timeout") => return Err(super::Error::Deadline.into()),
                        Err(reason) => *suppressed.entry(reason).or_default() += 1,
                    }
                }
                tracing::debug!(stage = "final_hit_sources", elapsed_us = started.elapsed().as_micros(), "search observation");
                let states = query_l2_many(&context, &verified_occurrences, signal)?;
                tracing::debug!(stage = "l2_sources", elapsed_us = started.elapsed().as_micros(), "search observation");
                let current_summaries = super::summaries::observe_many(&context, &verified_occurrences, signal)?;
                tracing::debug!(stage = "final_summaries", elapsed_us = started.elapsed().as_micros(), "search observation");
                let provider_transition = lexical_only
                    && matches!(
                        super::repair::validate_lexical_configuration(&context, signal),
                        Ok(true)
                    );
                check_query_work(signal)?;
                for (((((mut hit, state), before), after), occurrence), source) in verified
                    .into_iter()
                    .zip(states)
                    .zip(verified_summaries)
                    .zip(current_summaries)
                    .zip(verified_occurrences)
                    .zip(verified_sources)
                {
                    if before != after {
                        *suppressed.entry("summary_source_changed").or_default() += 1;
                        continue;
                    }
                    if let Some(l2) = state {
                        hit["fresh"]["l2"] = json!(l2);
                        if let Some(summary) = after { hit["summary"] = json!(summary.text); }
                        hits.push(hit);
                    } else if provider_transition {
                        // An embedding-only transition does not invalidate a
                        // lexical result. Recheck L1 bytes after the refused L2
                        // observation before returning it without an L2 claim.
                        match query_hit_source(&context, occurrence, &hit, signal)? {
                            Ok((_, current)) if current == source => {
                                // A selected-space vector is not proven current;
                                // use the existing conservative L2 state.
                                hit["fresh"]["l2"] = json!("pending");
                                hits.push(hit);
                            }
                            Ok(_) => *suppressed.entry("source_publication_changed").or_default() += 1,
                            Err("timeout") => return Err(super::Error::Deadline.into()),
                            Err(reason) => *suppressed.entry(reason).or_default() += 1,
                        }
                    } else {
                        *suppressed.entry("l2_authority_unavailable").or_default() += 1;
                    }
                }
            }
            // Validate retained L1 inputs without replaying model calls or comparing
            // hybrid/reranked scores against a lexical-only result.
            let current = evidence.is_current(&store, &parsed, &request, &context.config.search, &|| {
                signal.check().map_err(|_| super::Error::Deadline)
            });
            check_query_work(signal)?;
            if !matches!(current, Ok(true)) {
                let removed = response["hits"].as_array().map_or(0, Vec::len);
                response["hits"] = json!([]);
                suppressed.insert("selection_changed", removed);
            }
            // Cover empty selections too, and settings changes during the
            // repeated source/L2 reads. Never restart repair at this boundary.
            let settings = if lexical_only {
                super::repair::validate_lexical_configuration(&context, signal).map(|_| ())
            } else {
                super::repair::validate_configuration(&context, signal)
            };
            check_query_work(signal)?;
            if settings.is_err() {
                let removed = response["hits"].as_array().map_or(0, Vec::len);
                response["hits"] = json!([]);
                *suppressed.entry("settings_changed").or_default() += removed;
            }
            let generation_changed = query_generation(&store, signal)? != generation;
            if generation_changed {
                let removed = response["hits"].as_array().map_or(0, Vec::len);
                response["hits"] = json!([]);
                // Even an empty retrieval is incomplete after a competing scan.
                // A new call may retry; this call keeps its original work budget.
                suppressed.insert("index_generation_changed", removed);
            }
            if head_changed {
                // A checkout detected after verification invalidates that source
                // evidence, even when no competing index writer changed rows.
                let removed = response["hits"].as_array().map_or(0, Vec::len);
                response["hits"] = json!([]);
                suppressed.insert("head_changed", removed);
            }
            let branch_incomplete = head_changed
                || matches!(
                    branch["state"].as_str(),
                    Some("disabled" | "refused" | "incomplete")
                )
                || sync::head::changed(
                    inspection.indexed_head.as_deref(),
                    inspection.head.as_deref(),
                );
            // Pending catalogue entries may have no searchable occurrence. The
            // final inspection still makes overall coverage incomplete; keep
            // independently verified hits and do not expand the repair scope.
            let pending_incomplete = inspection.pending > 0;
            let mut index = inspection.index();
            index["branch_resync"] = branch;
            index["partial"] = json!(
                branch_incomplete || pending_incomplete || !suppressed.is_empty() || refused > 0
            );
            index["lazy_resync"] = json!({"attempted_files": targets.len(),
                "published_files": published, "refused_files": refused});
            index["suppressed_hits"] = json!(suppressed.values().sum::<usize>());
            index["freshness_refusals"] = json!(suppressed);
            if !suppressed.is_empty() || refused > 0 {
                index["fresh"] = json!({"state":"stale", "reason":"unverified_sources"});
            }
            index["degraded"] = json!(
                branch_incomplete
                    || pending_incomplete
                    || (parsed.dense.is_some() && evidence.dense_state(true) != "complete")
                    || evidence.degraded(rerank, &context.config)
                    || !suppressed.is_empty()
                    || refused > 0
            );
            index["dense"] = json!(evidence.dense_state(parsed.dense.is_some()));
            index["rerank"] = json!(evidence.rerank_state(rerank, &context.config));
            index["lexical"] = json!(if parsed.lexical.is_some() {
                if branch_incomplete
                    || pending_incomplete
                    || evidence.lexical_failed()
                    || generation_changed
                    || occurrence_changed
                    || !suppressed.is_empty()
                {
                    "incomplete"
                } else {
                    "complete"
                }
            } else {
                "not_requested"
            });
            index["reranked"] = json!(matches!(&evidence, SearchEvidence::Pipeline(results) if matches!(results.rerank, super::rerank::State::Complete)));
            index["took_ms"] = json!(started.elapsed().as_millis());
            response["index"] = index;
            if let Some(hits)=response["hits"].as_array_mut() {
                let report=response_limits::search(hits,context.config.limits.max_result_tokens,signal)?;
                response["result_budget"]=serde_json::to_value(report)?;
            }
            response["took_ms"] = json!(started.elapsed().as_millis());
            tracing::debug!(stage = "complete", elapsed_us = started.elapsed().as_micros(), "search observation");
            Ok(response)
        })
        .await
    }

    // Keep repair and its final retrieval proof in one L1 session. Bootstrap
    // retains its immediate incomplete-response policy for zero-setup serving.
    async fn query_session(&self, work: &QueryWork) -> Result<Option<store::l1_session::Session>> {
        let context = self.clone();
        work.blocking(move |signal| context.query_session_checked(signal))
            .await
    }

    fn query_session_checked(
        &self,
        work: sync::l1::Work<'_>,
    ) -> Result<Option<store::l1_session::Session>> {
        if self.background_l1.load(Ordering::Acquire) {
            return Ok(None);
        }
        let session = self.store.l1_session(&self.root, || {
            work.check().map_err(|_| store::Error::L1InputsChanged)
        });
        check_query_work(work)?;
        Ok(Some(session?))
    }

    fn query_host(&self) -> Result<Arc<super::host::Host>> {
        let mut slot = self
            .query_host
            .lock()
            .map_err(|_| super::Error::QueryTask)?;
        if let Some(host) = slot
            .as_ref()
            .filter(|host| host.matches(&self.store, &self.config))
        {
            return Ok(Arc::clone(host));
        }
        let host = Arc::new(super::host::Host::new(
            Arc::clone(&self.store),
            self.config_root.clone(),
            self.config.clone(),
        ));
        *slot = Some(Arc::clone(&host));
        Ok(host)
    }

    async fn retrieve(
        &self,
        request: &pipeline::Request,
        rerank: bool,
        work: &QueryWork,
    ) -> Result<Option<pipeline::Results>> {
        let parsed = super::query::parse(&request.query, request.mode)?;
        let host = self.query_host()?;
        let control = work.control();
        let providers = if parsed.dense.is_some() {
            let scope = super::dense::Scope::from_search(
                request.scope.as_ref(),
                request.exclude_ids.clone(),
            );
            host.providers(&scope, &control).await.ok()
        } else {
            None
        };
        let reranker = if rerank && self.config.rerank.mode != "off" {
            host.reranker(&control).await.ok()
        } else {
            None
        };
        // Preserve the established full lexical window when no model work can
        // contribute. Rerank candidate limits must not truncate plain fallback.
        if reranker.is_none()
            && providers
                .as_ref()
                .is_none_or(|providers| providers.factories.is_empty())
        {
            return Ok(None);
        }
        let clock = || {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_secs().min(i64::MAX as u64) as i64)
        };
        let mut candidates = match providers {
            Some(providers) => {
                pipeline::candidates_routed_with_control(
                    providers,
                    request.clone(),
                    reranker.as_deref(),
                    control,
                    clock,
                )
                .await?
            }
            None => {
                pipeline::candidates_with_control(
                    Arc::clone(&self.store),
                    self.config.clone(),
                    request.clone(),
                    None,
                    reranker.as_deref(),
                    control,
                    clock,
                )
                .await?
            }
        };
        if !candidates.hits.is_empty()
            && let Some(reranker) = reranker
        {
            candidates = candidates.rerank_lazy(&reranker, clock).await;
        }
        Ok(Some(candidates.finish(request.k, &self.config.search)?))
    }

    /// Opens source-verified pages under one request-owned deadline and cancellation flag.
    pub(crate) async fn open(&self, target: &str, context: u64) -> Result<Value> {
        self.open_checked(target, context, |_| {}, || {}).await
    }

    async fn open_checked(
        &self,
        target: &str,
        context: u64,
        before_source: impl FnOnce(sync::l1::Work<'_>) + Send + 'static,
        after_source: impl FnOnce() + Send + 'static,
    ) -> Result<Value> {
        let started = Instant::now();
        let owner = QueryWork::new(started, self.config.limits.query_timeout_ms)?;
        let service = self.clone();
        let target = target.to_owned();
        let operation = owner.blocking(move |work| {
            let _session = service.query_session_checked(work)?;
            let mut inspection = service.inspect_with_work(work)?;
            let branch = service.reconcile_branch(&mut inspection, work)?;
            if matches!(
                branch["state"].as_str(),
                Some("disabled" | "refused" | "incomplete")
            ) {
                return Err(super::Error::OpenSourceUnavailable.into());
            }
            // Resolve the original locator only after branch publication. Old
            // occurrence IDs and continuation digests retain their normal refusal.
            let retrieved_head = inspection.head.clone();
            let mut generation = query_generation(&service.store, work)?;
            let max_bytes = usize::try_from(service.config.limits.max_open_bytes()?.0)
                .map_err(|_| Error::OpenLimit)?;
            before_source(work);
            check_query_work(work)?;
            let mut opened = navigation::open_source(
                &service.store,
                (&service.root, &service.config_root),
                &service.config,
                &target,
                context,
                (max_bytes, service.config.limits.max_file_bytes()?.0),
                work,
            );
            check_query_work(work)?;
            if matches!(opened, Err(super::Error::OpenSourceUnavailable))
                && service.config.search.lazy_resync
            {
                let path = navigation::repair_path(
                    &service.store,
                    &service.root,
                    &target,
                    context,
                    &service.config,
                    work,
                );
                check_query_work(work)?;
                let Some(path) = path? else {
                    return Err(super::Error::OpenSourceUnavailable.into());
                };
                let repaired = super::repair::target(&service, &path, work);
                check_query_work(work)?;
                if !matches!(repaired, Ok(true)) {
                    return Err(super::Error::OpenSourceUnavailable.into());
                }
                generation = query_generation(&service.store, work)?;
                // Reresolve the original locator: old occurrence IDs and cursor
                // digests cannot inherit authority from the repaired path.
                opened = navigation::open_source(
                    &service.store,
                    (&service.root, &service.config_root),
                    &service.config,
                    &target,
                    context,
                    (max_bytes, service.config.limits.max_file_bytes()?.0),
                    work,
                );
            }
            // Map cooperative storage/source cancellation to the public timeout.
            check_query_work(work)?;
            let opened = opened?;
            after_source();
            // Re-resolve the original locator against current source bytes and
            // publication evidence. A changed page is refused, never substituted
            // into this response or repaired a second time under a new deadline.
            let settings = super::repair::validate_configuration(&service, work);
            check_query_work(work)?;
            settings.map_err(|_| super::Error::OpenSourceUnavailable)?;
            let verified = navigation::open_source(
                &service.store,
                (&service.root, &service.config_root),
                &service.config,
                &target,
                context,
                (max_bytes, service.config.limits.max_file_bytes()?.0),
                work,
            );
            check_query_work(work)?;
            let mut verified = verified.map_err(|_| super::Error::OpenSourceUnavailable)?;
            // Background embedding can advance the ladder without changing the
            // locator, rendered page or completed L1/source identity.
            verified.occurrence.ladder_level = opened.occurrence.ladder_level;
            if verified != opened
                || query_occurrences(
                    &service.store,
                    std::slice::from_ref(&opened.occurrence),
                    work,
                )? != [true]
            {
                return Err(super::Error::OpenSourceUnavailable.into());
            }
            let settings = super::repair::validate_configuration(&service, work);
            check_query_work(work)?;
            settings.map_err(|_| super::Error::OpenSourceUnavailable)?;
            inspection = service.inspect_with_work(work)?;
            if query_generation(&service.store, work)? != generation
                || inspection.head != retrieved_head
                || sync::head::changed(
                    inspection.indexed_head.as_deref(),
                    inspection.head.as_deref(),
                )
            {
                return Err(super::Error::OpenSourceUnavailable.into());
            }
            let mut response = serde_json::to_value(opened)?;
            inspection.annotate(&mut response);
            response["index"] = inspection.index();
            response["index"]["branch_resync"] = branch;
            response["took_ms"] = json!(started.elapsed().as_millis());
            Ok(response)
        });
        let result =
            tokio::time::timeout_at(tokio::time::Instant::from_std(owner.deadline), operation)
                .await
                .unwrap_or_else(|_| Err(Error::Search(super::Error::Deadline)));
        self.observe_response_caps("open", &result);
        result
    }

    /// Returns the specified symbol array only when every selected source verifies.
    /// Refusal and timeout use explicit errors because the array has no partial envelope.
    pub(crate) async fn symbol(&self, target: &str) -> Result<Value> {
        self.symbol_checked(target, || {}).await
    }

    async fn symbol_checked(
        &self,
        target: &str,
        after_verification: impl FnOnce() + Send + 'static,
    ) -> Result<Value> {
        let owner = QueryWork::new(Instant::now(), self.config.limits.query_timeout_ms)?;
        let context = self.clone();
        let target = target.to_owned();
        let operation = owner.blocking(move |work| {
            let _session = context.query_session_checked(work)?;
            let mut inspection = context.inspect_with_work(work)?;
            let branch = context.reconcile_branch(&mut inspection, work)?;
            // An array has nowhere to report incomplete discovery, even if the
            // original lookup would return zero matches or only valid sources.
            if matches!(
                branch["state"].as_str(),
                Some("disabled" | "refused" | "incomplete")
            ) {
                return Err(super::Error::OpenSourceUnavailable.into());
            }
            links::refresh(&context, work)?;
            let retrieved_head = inspection.head.clone();
            let mut generation = query_generation(&context.store, work)?;
            let mut symbols = navigation::symbol_checked(&context.store, &target, &|| {
                work.check().map_err(|_| store::Error::L1InputsChanged)
            })?;
            let (_, mut response_truncated) = symbol_history::truncate_to_candidates(
                &mut symbols,
                context.config.limits.max_result_tokens,
                work,
            )?;
            let max_bytes = context.config.limits.max_file_bytes()?.0;
            let l1 = navigation::L1Verifier::new(&context.root, &context.config)
                .ok_or(super::Error::OpenSourceUnavailable)?;
            let mut targets = std::collections::BTreeSet::new();
            if context.config.search.lazy_resync {
                for symbol in &mut symbols {
                    check_query_work(work)?;
                    match navigation::verify_symbol_with(
                        &context.store,
                        &context.root,
                        symbol,
                        max_bytes,
                        work,
                        &l1,
                    ) {
                        Ok(()) => {}
                        Err(super::Error::OpenSourceUnavailable) => {
                            targets.insert(symbol.path.clone());
                        }
                        Err(error) => return Err(error.into()),
                    }
                }
            }
            for path in &targets {
                check_query_work(work)?;
                // The array cannot report partial repair. A refused attempt must
                // remain an explicit whole-call error, even if other files publish.
                let repaired = super::repair::target(&context, path, work);
                check_query_work(work)?;
                if !matches!(repaired, Ok(true)) {
                    return Err(super::Error::OpenSourceUnavailable.into());
                }
            }
            if !targets.is_empty() {
                links::refresh(&context, work)?;
                generation = query_generation(&context.store, work)?;
                symbols = navigation::symbol_checked(&context.store, &target, &|| {
                    work.check().map_err(|_| store::Error::L1InputsChanged)
                })?;
                let (_, truncated) = symbol_history::truncate_to_candidates(
                    &mut symbols,
                    context.config.limits.max_result_tokens,
                    work,
                )?;
                response_truncated |= truncated;
            }
            if links::repair_sources(&context, &symbols, work)? {
                generation = query_generation(&context.store, work)?;
                symbols = navigation::symbol_checked(&context.store, &target, &|| {
                    work.check().map_err(|_| store::Error::L1InputsChanged)
                })?;
                let (_, truncated) = symbol_history::truncate_to_candidates(
                    &mut symbols,
                    context.config.limits.max_result_tokens,
                    work,
                )?;
                response_truncated |= truncated;
            }
            for symbol in &mut symbols {
                navigation::verify_symbol_with(
                    &context.store,
                    &context.root,
                    symbol,
                    max_bytes,
                    work,
                    &l1,
                )?;
            }
            for symbol in &mut symbols {
                super::summaries::populate_symbol(&context, symbol, work)?;
            }
            links::populate(&context, &mut symbols, work)?;
            let (history_candidates, truncated) = symbol_history::truncate_to_candidates(
                &mut symbols,
                context.config.limits.max_result_tokens,
                work,
            )?;
            response_truncated |= truncated;
            populate_symbol_histories(
                &context,
                &mut symbols,
                &history_candidates,
                work,
                context.config.search.lazy_resync,
            )?;
            check_query_work(work)?;
            after_verification();
            let settings = super::repair::validate_configuration(&context, work);
            check_query_work(work)?;
            settings.map_err(|_| super::Error::OpenSourceUnavailable)?;
            // Repeat the original selection, including empty arrays. Late
            // changes refuse this response without a second repair attempt.
            let selected = navigation::symbol_checked(&context.store, &target, &|| {
                work.check().map_err(|_| store::Error::L1InputsChanged)
            });
            check_query_work(work)?;
            let mut verified = selected.map_err(|_| super::Error::OpenSourceUnavailable)?;
            let (_, truncated) = symbol_history::truncate_to_candidates(
                &mut verified,
                context.config.limits.max_result_tokens,
                work,
            )?;
            response_truncated |= truncated;
            for symbol in &mut verified {
                let result = navigation::verify_symbol_with(
                    &context.store,
                    &context.root,
                    symbol,
                    max_bytes,
                    work,
                    &l1,
                );
                check_query_work(work)?;
                result.map_err(|_| super::Error::OpenSourceUnavailable)?;
                super::summaries::populate_symbol(&context, symbol, work)?;
            }
            links::populate(&context, &mut verified, work)?;
            let (history_candidates, truncated) = symbol_history::truncate_to_candidates(
                &mut verified,
                context.config.limits.max_result_tokens,
                work,
            )?;
            response_truncated |= truncated;
            populate_symbol_histories(&context, &mut verified, &history_candidates, work, false)?;
            for (original, current) in symbols.iter().zip(&mut verified) {
                if let (Some((before, _)), Some((after, _))) =
                    (&original.evidence, &mut current.evidence)
                {
                    // Background L2 publication need not invalidate L1 authority.
                    after.ladder_level = before.ladder_level;
                    if query_occurrences(&context.store, std::slice::from_ref(before), work)?
                        != [true]
                    {
                        return Err(super::Error::OpenSourceUnavailable.into());
                    }
                }
            }
            if verified != symbols {
                return Err(super::Error::OpenSourceUnavailable.into());
            }
            let settings = super::repair::validate_configuration(&context, work);
            check_query_work(work)?;
            settings.map_err(|_| super::Error::OpenSourceUnavailable)?;
            inspection = context.inspect_with_work(work)?;
            if query_generation(&context.store, work)? != generation
                || inspection.head != retrieved_head
                || inspection.pending > 0
                || sync::head::changed(
                    inspection.indexed_head.as_deref(),
                    inspection.head.as_deref(),
                )
            {
                return Err(super::Error::OpenSourceUnavailable.into());
            }
            let mut response = serde_json::to_value(symbols)?;
            if let Some(symbols) = response.as_array_mut() {
                for symbol in symbols {
                    inspection.annotate(symbol);
                }
            }
            let (response, final_truncated) =
                symbol_history::limit(response, context.config.limits.max_result_tokens, work)?;
            if response_truncated || final_truncated {
                context
                    .query_latencies
                    .record_cap(super::latency::ResponseCap::ResultTokens, "symbol");
            }
            Ok(response)
        });
        let result =
            tokio::time::timeout_at(tokio::time::Instant::from_std(owner.deadline), operation)
                .await
                .unwrap_or_else(|_| Err(Error::Search(super::Error::Deadline)));
        self.observe_response_caps("symbol", &result);
        result
    }

    /// Suppresses unverifiable reference locations under one request-owned deadline.
    /// Relationship inference remains approximate even when source ranges verify.
    pub(crate) async fn refs(
        &self,
        target: &str,
        k: Option<usize>,
        tests_only: bool,
    ) -> Result<Value> {
        self.refs_checked(target, k, tests_only, || {}).await
    }

    async fn refs_checked(
        &self,
        target: &str,
        k: Option<usize>,
        tests_only: bool,
        after_verification: impl FnOnce() + Send + 'static,
    ) -> Result<Value> {
        if k == Some(0) {
            return Err(super::Error::InvalidReferenceLimit.into());
        }
        let started = Instant::now();
        let owner = QueryWork::new(started, self.config.limits.query_timeout_ms)?;
        let context = self.clone();
        let target = target.to_owned();
        let operation = owner.blocking(move |work| {
            let _session = context.query_session_checked(work)?;
            let mut inspection = context.inspect_with_work(work)?;
            let branch = context.reconcile_branch(&mut inspection, work)?;
            links::refresh(&context, work)?;
            let retrieved_head = inspection.head.clone();
            let mut generation = query_generation(&context.store, work)?;
            let mut selection_cache = None;
            let mut references = navigation::refs_checked_reuse(
                &context.store,
                &mut selection_cache,
                &target,
                k,
                tests_only,
                &|| work.check().map_err(|_| store::Error::L1InputsChanged),
            )?;
            let max_bytes = context.config.limits.max_file_bytes()?.0;
            let l1 = navigation::L1Verifier::new(&context.root, &context.config)
                .ok_or(super::Error::OpenSourceUnavailable)?;
            let proof_revision_before = context
                .store
                .revision_checked(&|| work.check().map_err(|_| store::Error::L1InputsChanged))?;
            let mut targets = std::collections::BTreeSet::new();
            // The lazy pass already proves unchanged groups. Retain those
            // request-local observations for the first output pass; the
            // independent final pass still re-reads live source and storage.
            let mut lazy_verified = Vec::new();
            if context.config.search.lazy_resync {
                for group in &references.groups {
                    check_query_work(work)?;
                    match navigation::reference_authority_with(
                        &context.store,
                        &context.root,
                        group,
                        max_bytes,
                        work,
                        &l1,
                    )? {
                        Ok(authority) => lazy_verified.push(Some(authority)),
                        Err(reason) => {
                            lazy_verified.push(None);
                            if matches!(
                                reason,
                                "source_changed"
                                    | "l1_pending_or_unavailable"
                                    | "invalid_source_range"
                                    | "l1_config_stale"
                            ) {
                                targets.insert(group.path.clone());
                            }
                        }
                    }
                }
            }
            let mut published = 0usize;
            let mut refused = 0usize;
            for path in &targets {
                check_query_work(work)?;
                match super::repair::target(&context, path, work) {
                    Ok(true) => published += 1,
                    Ok(false) | Err(_) => refused += 1,
                }
                check_query_work(work)?;
            }
            if !targets.is_empty() {
                links::refresh(&context, work)?;
                // Refused completion can still publish a pending catalogue. Reload
                // relationships, counts and generation using the original filters.
                generation = query_generation(&context.store, work)?;
                references = navigation::refs_checked_reuse(
                    &context.store,
                    &mut selection_cache,
                    &target,
                    k,
                    tests_only,
                    &|| work.check().map_err(|_| store::Error::L1InputsChanged),
                )?;
            }
            check_query_work(work)?;
            let selected = references.clone();
            let indexed_total = references.total;
            let mut groups = Vec::new();
            let mut evidence = Vec::new();
            let mut suppressed = 0_usize;
            let mut refusals = std::collections::BTreeMap::<&str, usize>::new();
            for (index, group) in references.groups.into_iter().enumerate() {
                check_query_work(work)?;
                let cached = if targets.is_empty() {
                    lazy_verified.get_mut(index).and_then(Option::take)
                } else {
                    None
                };
                let verified = if let Some(authority) = cached {
                    Ok(authority)
                } else {
                    navigation::reference_authority_with(
                        &context.store,
                        &context.root,
                        &group,
                        max_bytes,
                        work,
                        &l1,
                    )?
                };
                match verified {
                    Ok(authority) => {
                        evidence.push(authority);
                        groups.push(group);
                    }
                    Err(reason) => {
                        suppressed += group.lines.len();
                        *refusals.entry(reason).or_default() += 1;
                    }
                }
            }
            let proof_revision_after = context
                .store
                .revision_checked(&|| work.check().map_err(|_| store::Error::L1InputsChanged))?;
            let proof_revision =
                (proof_revision_before == proof_revision_after).then_some(proof_revision_after);
            check_query_work(work)?;
            after_verification();
            // Recheck the original filters and pre-limit total, even for empty
            // selections. Same-generation relationship updates can change both.
            // Late refusal never starts another repair or returns unchecked groups.
            let settings = super::repair::validate_configuration(&context, work);
            check_query_work(work)?;
            let mut settings_changed = settings.is_err();
            let mut selection_changed = false;
            let mut current_contributors = None;
            if !settings_changed {
                let current = navigation::refs_checked_reuse(
                    &context.store,
                    &mut selection_cache,
                    &target,
                    k,
                    tests_only,
                    &|| work.check().map_err(|_| store::Error::L1InputsChanged),
                );
                check_query_work(work)?;
                selection_changed = current.as_ref().map_or(true, |value| {
                    value.groups != selected.groups || value.total != selected.total
                });
                current_contributors = current.ok().map(|value| value.contributors);
                let settings = super::repair::validate_configuration(&context, work);
                check_query_work(work)?;
                settings_changed = settings.is_err();
            }
            let mut source_changed = false;
            if !settings_changed && !selection_changed {
                let mut verified_groups = Vec::new();
                for (group, (before, authority)) in groups.into_iter().zip(evidence) {
                    check_query_work(work)?;
                    let cached = if let Some(revision) = proof_revision {
                        navigation::reference_authority_cached_with(
                            &context.store,
                            &context.root,
                            &group,
                            navigation::ReferenceProof {
                                unit: &before,
                                authority: &authority,
                                revision,
                            },
                            max_bytes,
                            work,
                            &l1,
                        )?
                    } else {
                        None
                    };
                    let current = match cached {
                        Some(verified) => verified,
                        None => navigation::reference_authority_with(
                            &context.store,
                            &context.root,
                            &group,
                            max_bytes,
                            work,
                            &l1,
                        )?,
                    };
                    check_query_work(work)?;
                    let key = (group.path.clone(), group.kind);
                    let original_contributors = selected.contributors.get(&key);
                    let contributors_changed = current_contributors
                        .as_ref()
                        .and_then(|values| values.get(&key))
                        != original_contributors;
                    let occurrences = original_contributors
                        .into_iter()
                        .flat_map(|value| value.occurrences.values().cloned())
                        .collect::<Vec<_>>();
                    let occurrences_current =
                        query_occurrences(&context.store, &occurrences, work)?
                            .into_iter()
                            .all(|current| current);
                    let reason = match current {
                        Ok(_) if contributors_changed || !occurrences_current => {
                            Some("occurrence_changed")
                        }
                        Ok((mut after, current_authority)) => {
                            // L2 can advance independently of completed L1.
                            after.ladder_level = before.ladder_level;
                            if after != before
                                || current_authority != authority
                                || query_occurrences(
                                    &context.store,
                                    std::slice::from_ref(&before),
                                    work,
                                )? != [true]
                            {
                                Some("occurrence_changed")
                            } else {
                                None
                            }
                        }
                        Err(reason) => Some(reason),
                    };
                    if let Some(reason) = reason {
                        source_changed = true;
                        suppressed += group.lines.len();
                        *refusals.entry(reason).or_default() += 1;
                    } else {
                        verified_groups.push(group);
                    }
                }
                groups = verified_groups;
                let settings = super::repair::validate_configuration(&context, work);
                check_query_work(work)?;
                settings_changed = settings.is_err();
            }
            let authority_changed = settings_changed || selection_changed || source_changed;
            if settings_changed || selection_changed {
                suppressed += groups.iter().map(|group| group.lines.len()).sum::<usize>();
                refusals.insert(
                    if settings_changed {
                        "settings_changed"
                    } else {
                        "selection_changed"
                    },
                    groups.len(),
                );
                groups.clear();
            }
            // Catalogue and HEAD fences also cover the repeated lookup.
            inspection = context.inspect_with_work(work)?;
            let head_changed = inspection.head != retrieved_head;
            let generation_changed = query_generation(&context.store, work)? != generation;
            if generation_changed || head_changed {
                suppressed += groups.iter().map(|group| group.lines.len()).sum::<usize>();
                refusals.insert(
                    if head_changed {
                        "head_changed"
                    } else {
                        "index_changed"
                    },
                    groups.len(),
                );
                groups.clear();
            }
            let mut groups = groups
                .into_iter()
                .map(serde_json::to_value)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let result_budget =
                response_limits::array(&mut groups, context.config.limits.max_result_tokens, work)?;
            let returned_total = groups
                .iter()
                .map(|group| group["lines"].as_array().map_or(0, Vec::len))
                .sum::<usize>();
            let mut index = inspection.index();
            let partial = authority_changed
                || suppressed > 0
                || refused > 0
                || generation_changed
                || head_changed
                || inspection.pending > 0
                || matches!(
                    branch["state"].as_str(),
                    Some("disabled" | "refused" | "incomplete")
                )
                || sync::head::changed(
                    inspection.indexed_head.as_deref(),
                    inspection.head.as_deref(),
                );
            index["partial"] = json!(partial);
            index["degraded"] = json!(partial);
            index["branch_resync"] = branch;
            if authority_changed || generation_changed || head_changed {
                index["fresh"] = json!({"state":"stale", "reason":"unverified_sources"});
            }
            index["lazy_resync"] = json!({"attempted_files": targets.len(),
                "published_files": published, "refused_files": refused});
            index["suppressed_references"] = json!(suppressed);
            index["freshness_refusals"] = json!(refusals);
            Ok(json!({
                "groups": groups, "total": if partial { None } else { Some(indexed_total) },
                "indexed_total": indexed_total, "returned_total": returned_total,
                "approximate": true, "index": index, "result_budget": result_budget,
                "took_ms": started.elapsed().as_millis()
            }))
        });
        let result = match tokio::time::timeout_at(
            tokio::time::Instant::from_std(owner.deadline),
            operation,
        )
        .await
        {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(Error::Search(super::Error::Deadline))) | Err(_) => Ok(json!({
                "groups": [], "total": null, "indexed_total": null, "returned_total": 0,
                "approximate": true, "took_ms": started.elapsed().as_millis(),
                "index": {"partial": true, "degraded": true, "reason": "query_timeout",
                    "head": null, "indexed_head": null, "l1_pending_files": null,
                        "dirty_files": null, "dirty_files_state": "not_checked",
                    "suppressed_references": null, "freshness_refusals": {},
                    "fresh": {"state": "unknown", "reason": "query_timeout"}}
            })),
            Ok(Err(error)) => Err(error),
        };
        self.observe_response_caps("refs", &result);
        result
    }

    /// Repairs and verifies selected map locations under one request deadline.
    pub(crate) async fn map(&self, budget: usize, focus: &[String]) -> Result<Value> {
        self.map_checked(budget, focus, || {}).await
    }

    async fn map_checked(
        &self,
        budget: usize,
        focus: &[String],
        after_verification: impl FnOnce() + Send + 'static,
    ) -> Result<Value> {
        if budget == 0 {
            return Err(super::Error::InvalidMapBudget.into());
        }
        let started = Instant::now();
        let owner = QueryWork::new(started, self.config.limits.query_timeout_ms)?;
        let context = self.clone();
        let focus = focus.to_vec();
        let operation = owner.blocking(move |work| {
            let _session = context.query_session_checked(work)?;
            let mut inspection = context.inspect_with_work(work)?;
            let branch = context.reconcile_branch(&mut inspection, work)?;
            let retrieved_head = inspection.head.clone();
            let mut generation = query_generation(&context.store, work)?;
            let mut suppressed = 0usize;
            let mut refusals = std::collections::BTreeMap::<&str, usize>::new();
            let max_bytes = context.config.limits.max_file_bytes()?.0;
            let mut targets = std::collections::BTreeSet::new();
            let mut map_cache = None;
            if context.config.search.lazy_resync {
                super::map::map_checked_reuse(
                    &context.store,
                    &mut map_cache,
                    budget,
                    &focus,
                    &|| work.check().map_err(|_| store::Error::L1InputsChanged),
                    |path, symbols| {
                        match navigation::verify_map_file(
                            &context.store,
                            &context.root,
                            &context.config,
                            path,
                            symbols,
                            max_bytes,
                            work,
                        )? {
                            Err(
                                "source_changed"
                                | "l1_pending_or_unavailable"
                                | "l1_config_stale"
                                | "invalid_source_range",
                            ) => {
                                targets.insert(path.to_owned());
                            }
                            Err("timeout") => return Err(super::Error::Deadline),
                            _ => {}
                        }
                        Ok(true)
                    },
                )?;
            }
            let mut published = 0usize;
            let mut refused = 0usize;
            for path in &targets {
                check_query_work(work)?;
                match super::repair::target(&context, path, work) {
                    Ok(true) => published += 1,
                    Ok(false) | Err(_) => refused += 1,
                }
                check_query_work(work)?;
            }
            if !targets.is_empty() {
                // Even refused completion may publish pending catalogue state.
                // Reselect with the original focus and budget, using new metadata.
                generation = query_generation(&context.store, work)?;
            }
            let mut observed_module_sources = None;
            let mut module_snapshot_cache = None;
            // Module labels depend on more than the focused file. Repair stale
            // module provenance under this same request/session before capturing
            // the final file/rank selection; the standard index enforces all caps.
            if context.config.search.lazy_resync
                && let Some(session) = _session.as_ref()
            {
                let selection = super::map::map_checked_reuse(
                    &context.store,
                    &mut map_cache,
                    budget,
                    &focus,
                    &|| work.check().map_err(|_| store::Error::L1InputsChanged),
                    |_, _| Ok(true),
                )?;
                let modules = super::map::modules(&context.store, &selection, &|| {
                    work.check().map_err(|_| store::Error::L1InputsChanged)
                })?;
                let ids = modules.iter().map(|m| m.unit.id).collect::<Vec<_>>();
                let sources =
                    query_module_sources(&context, &ids, work, &mut module_snapshot_cache)?;
                let stale_module = modules.iter().any(|m| !sources.contains_key(&m.unit.id));
                if stale_module {
                    let result = sync::index::run_in_session(
                        sync::index::Input {
                            config_root: &context.config_root,
                            root: &context.root,
                            settings: &context.config,
                            store: &context.store,
                            full: false,
                            work,
                        },
                        session,
                    );
                    check_query_work(work)?;
                    if result.is_err() {
                        refused += 1;
                    }
                    generation = query_generation(&context.store, work)?;
                } else {
                    observed_module_sources = Some(sources);
                }
            }
            let mut evidence = std::collections::BTreeMap::new();
            let mut mapped = super::map::map_checked_reuse(
                &context.store,
                &mut map_cache,
                budget,
                &focus,
                &|| work.check().map_err(|_| store::Error::L1InputsChanged),
                |path, symbols| match navigation::map_authority(
                    &context.store,
                    &context.root,
                    &context.config,
                    path,
                    symbols,
                    max_bytes,
                    work,
                )? {
                    Ok(authority) => {
                        evidence.insert(path.to_owned(), (symbols.to_vec(), authority));
                        Ok(true)
                    }
                    Err("timeout") => Err(super::Error::Deadline),
                    Err(reason) => {
                        suppressed += 1;
                        *refusals.entry(reason).or_default() += 1;
                        Ok(false)
                    }
                },
            )?;
            let mut module_evidence = Vec::new();
            let mut labels = std::collections::BTreeMap::new();
            let mut modules = super::map::modules(&context.store, &mapped, &|| {
                work.check().map_err(|_| store::Error::L1InputsChanged)
            })?;
            super::summaries::populate_modules(&context, &mut modules, work)?;
            let ids = modules.iter().map(|m| m.unit.id).collect::<Vec<_>>();
            // Without a repair, these sources were just verified in this same
            // observation. Retain that evidence for the independently fresh final
            // batch below; new selections and repaired modules need a new batch.
            let mut sources = match observed_module_sources
                .filter(|sources| ids.iter().all(|id| sources.contains_key(id)))
            {
                Some(sources) => sources,
                None => query_module_sources(&context, &ids, work, &mut module_snapshot_cache)?,
            };
            for module in modules {
                match sources.remove(&module.unit.id) {
                    Some(source) => {
                        labels.insert(module.path.clone(), module.label());
                        module_evidence.push((module, source));
                    }
                    None => {
                        suppressed += 1;
                        *refusals.entry("history_source_unavailable").or_default() += 1;
                    }
                }
            }
            super::map::label_modules(&mut mapped, labels, budget, &|| {
                work.check().map_err(|_| store::Error::L1InputsChanged)
            })?;
            check_query_work(work)?;
            after_verification();
            // Repeat the original focus/budget lookup without repairing or
            // rendering any newly selected source into the response. Private
            // evidence includes ranks and ownership hidden by compact output.
            let settings = super::repair::validate_configuration(&context, work);
            check_query_work(work)?;
            let mut settings_changed = settings.is_err();
            let mut selection_changed = false;
            if !settings_changed {
                let current = super::map::map_checked_reuse(
                    &context.store,
                    &mut map_cache,
                    budget,
                    &focus,
                    &|| work.check().map_err(|_| store::Error::L1InputsChanged),
                    |_, _| Ok(true),
                );
                check_query_work(work)?;
                selection_changed = current
                    .as_ref()
                    .map_or(true, |value| value.selection != mapped.selection);
                let settings = super::repair::validate_configuration(&context, work);
                check_query_work(work)?;
                settings_changed = settings.is_err();
            }
            let mut source_changed = false;
            if !settings_changed && !selection_changed {
                let mut current_modules = super::map::modules(&context.store, &mapped, &|| {
                    work.check().map_err(|_| store::Error::L1InputsChanged)
                })?;
                super::summaries::populate_modules(&context, &mut current_modules, work)?;
                let ids = module_evidence
                    .iter()
                    .filter(|(m, _)| map_contains_module(&mapped, &m.path))
                    .map(|(m, _)| m.unit.id)
                    .collect::<Vec<_>>();
                let sources =
                    query_module_sources(&context, &ids, work, &mut module_snapshot_cache)?;
                for (module, before) in &module_evidence {
                    // A label whose entire subtree was budget-trimmed contributes
                    // nothing. Retained labels need both source and presentation proof.
                    let current = current_modules.iter().find(|m| m.path == module.path);
                    if !map_contains_module(&mapped, &module.path) {
                        continue;
                    }
                    let source = sources.get(&module.unit.id);
                    if current != Some(module) || source != Some(before) {
                        mapped.module_labels.remove(&module.path);
                        source_changed = true;
                        suppressed += 1;
                        *refusals.entry("module_source_changed").or_default() += 1;
                    }
                }
                if source_changed {
                    let labels = mapped.module_labels.clone();
                    super::map::label_modules(&mut mapped, labels, budget, &|| {
                        work.check().map_err(|_| store::Error::L1InputsChanged)
                    })?;
                }
                let mut retained = std::collections::BTreeSet::new();
                for node in &mapped.nodes {
                    check_query_work(work)?;
                    let reason = if let Some((symbols, before)) = evidence.get(&node.path) {
                        let current = navigation::map_authority(
                            &context.store,
                            &context.root,
                            &context.config,
                            &node.path,
                            symbols,
                            max_bytes,
                            work,
                        )?;
                        check_query_work(work)?;
                        match current {
                            Ok(after) if &after == before => None,
                            Ok(_) => Some("occurrence_changed"),
                            Err("timeout") => return Err(super::Error::Deadline.into()),
                            Err(reason) => Some(reason),
                        }
                    } else {
                        Some("l1_pending_or_unavailable")
                    };
                    if let Some(reason) = reason {
                        source_changed = true;
                        suppressed += 1;
                        *refusals.entry(reason).or_default() += 1;
                    } else {
                        retained.insert(node.path.clone());
                    }
                }
                if source_changed {
                    super::map::retain_checked(&mut mapped, &retained, budget, &|| {
                        work.check().map_err(|_| store::Error::L1InputsChanged)
                    })?;
                }
                let settings = super::repair::validate_configuration(&context, work);
                check_query_work(work)?;
                settings_changed = settings.is_err();
            }
            let authority_changed = settings_changed || selection_changed;
            if authority_changed {
                suppressed += mapped.nodes.len();
                refusals.insert(
                    if settings_changed {
                        "settings_changed"
                    } else {
                        "selection_changed"
                    },
                    mapped.nodes.len(),
                );
                mapped.nodes.clear();
                mapped.tree.clear();
            }
            inspection = context.inspect_with_work(work)?;
            let head_changed = inspection.head != retrieved_head;
            let generation_changed = query_generation(&context.store, work)? != generation;
            if generation_changed || head_changed {
                suppressed += mapped.nodes.len();
                refusals.insert(
                    if head_changed {
                        "head_changed"
                    } else {
                        "index_changed"
                    },
                    mapped.nodes.len(),
                );
                mapped.nodes.clear();
                mapped.tree.clear();
            }
            let result_budget = response_limits::map(
                &mut mapped,
                context.config.limits.max_result_tokens,
                budget,
                work,
            )?;
            let mut response = serde_json::to_value(mapped)?;
            response["result_budget"] = serde_json::to_value(result_budget)?;
            response["index"] = inspection.index();
            let partial = source_changed
                || authority_changed
                || suppressed > 0
                || refused > 0
                || generation_changed
                || head_changed
                || inspection.pending > 0
                || matches!(
                    branch["state"].as_str(),
                    Some("disabled" | "refused" | "incomplete")
                )
                || sync::head::changed(
                    inspection.indexed_head.as_deref(),
                    inspection.head.as_deref(),
                );
            response["index"]["partial"] = json!(partial);
            response["index"]["degraded"] = json!(partial);
            response["index"]["branch_resync"] = branch;
            if source_changed || authority_changed || generation_changed || head_changed {
                response["index"]["fresh"] =
                    json!({"state":"stale", "reason":"unverified_sources"});
            }
            response["index"]["lazy_resync"] = json!({"attempted_files": targets.len(),
                "published_files": published, "refused_files": refused});
            response["index"]["suppressed_files"] = json!(suppressed);
            response["index"]["freshness_refusals"] = json!(refusals);
            response["took_ms"] = json!(started.elapsed().as_millis());
            Ok(response)
        });
        let result = match tokio::time::timeout_at(
            tokio::time::Instant::from_std(owner.deadline),
            operation,
        )
        .await
        {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(Error::Search(super::Error::Deadline))) | Err(_) => Ok(json!({
                "tree": "", "nodes": [], "took_ms": started.elapsed().as_millis(),
                "index": {"partial": true, "degraded": true, "reason": "query_timeout", "head": null,
                    "indexed_head": null, "l1_pending_files": null, "suppressed_files": null,
                    "freshness_refusals": {}, "fresh": {"state": "unknown", "reason": "query_timeout"}}
            })),
            Ok(Err(error)) => Err(error),
        };
        self.observe_response_caps("map", &result);
        result
    }

    fn file_capacity_reached(&self, files: i64) -> Result<bool> {
        let discovery = self
            .store
            .get_meta("discovery_status")?
            .map(|value| serde_json::from_str::<Value>(&value))
            .transpose()?;
        Ok(
            u64::try_from(files).unwrap_or(u64::MAX) >= self.config.limits.max_files
                || discovery.is_some_and(|value| {
                    value["max_files"].as_u64() == Some(self.config.limits.max_files)
                        && value["max_files_reached"] == true
                }),
        )
    }

    /// Preserves the CLI's readable cap, queue, model and reranker status summaries.
    pub(crate) fn status_text(&self) -> Result<String> {
        let _inspection = self.inspect()?;
        let counts = self.store.counts()?;
        let queue = queue::status::snapshot(&self.store, unix_seconds()?)?;
        let embedding = queue::inspection::read_configured(&self.store, &self.config)?;
        let rerank_auto = super::rerank::auto::read_status(&self.store, &self.config)?;
        let health_summary = crate::models::rerank::health::read(&self.store, &self.config)?
            .as_ref()
            .map(|report| report.summary())
            .unwrap_or_default();
        let rerank_summary = rerank_auto.message.map(str::to_owned).unwrap_or_else(|| {
            format!(
                "rerank: {} candidates (mode {})",
                rerank_auto.effective_candidates, self.config.rerank.mode
            )
        });
        let preparation = self
            .store
            .get_meta("model_preparation")?
            .map(|value| serde_json::from_str::<Value>(&value))
            .transpose()?
            .unwrap_or(Value::Null);
        let model_status = preparation["state"].as_str().unwrap_or("not attempted");
        let model_detail = preparation["detail"].as_str().unwrap_or("");
        let model_summary = format!("{model_status} {model_detail}");
        let history = self
            .store
            .get_meta("history_status")?
            .map(|value| serde_json::from_str::<Value>(&value))
            .transpose()?
            .unwrap_or(Value::Null);
        let history_count = self.store.commit_count(&|| Ok(()))?;
        let history_state = if self.config.commits.enabled {
            history["state"].as_str().unwrap_or("not indexed")
        } else {
            "disabled"
        };
        let prs = self
            .store
            .get_meta("pr_status")?
            .map(|value| serde_json::from_str::<Value>(&value))
            .transpose()?
            .unwrap_or(Value::Null);
        let pr_summary = format!(
            "PRs: {} retained; {}; deferrals: {}",
            prs["retained"],
            if self.config.commits.enabled && self.config.commits.prs {
                prs["state"].as_str().unwrap_or("not indexed")
            } else {
                "disabled"
            },
            prs["deferred"]
        );
        let histories = self
            .store
            .get_meta("symbol_history_status")?
            .map(|value| serde_json::from_str::<Value>(&value))
            .transpose()?
            .unwrap_or(Value::Null);
        let histories_summary = format!(
            "symbol histories: {}; {} selected; {} deferred",
            histories["state"].as_str().unwrap_or("not indexed"),
            histories["selected"],
            histories["deferred"]
        );
        let modules = self
            .store
            .get_meta("modules_status")?
            .map(|value| serde_json::from_str::<Value>(&value))
            .transpose()?
            .unwrap_or(Value::Null);
        let modules_summary = format!(
            "modules: {}; {} selected; {} deferred",
            modules["state"].as_str().unwrap_or("not indexed"),
            modules["selected"],
            modules["deferred"]
        );
        let linker = self
            .store
            .get_meta("linker_status")?
            .map(|value| serde_json::from_str::<Value>(&value))
            .transpose()?
            .unwrap_or(Value::Null);
        let linker_summary = format!(
            "documentation/test links: {}; deferred: {}; reason: {}",
            linker["state"].as_str().unwrap_or("not indexed"),
            linker["deferred"],
            linker["reason"]
        );
        let history_summary = format!(
            "history: {history_count} commits; {history_state}; max_count={}, max_commits={}",
            self.config.commits.max_count, self.config.limits.max_commits
        );

        let priority = self
            .store
            .get_meta("index_priority")?
            .map(|value| serde_json::from_str::<Value>(&value))
            .transpose()?
            .unwrap_or(Value::Null);
        let shrink = self
            .store
            .get_meta("last_shrink")?
            .map(|value| serde_json::from_str::<Value>(&value))
            .transpose()?
            .unwrap_or(Value::Null);
        let deferred_files = self.store.l1_deferred_files()?;
        let cap_reasons = self
            .store
            .source_cap_counts(self.config.limits.max_file_bytes()?.0)?;
        let source_caps = if cap_reasons.is_empty() {
            "none".to_owned()
        } else {
            cap_reasons
                .iter()
                .map(|(reason, count)| format!("{reason}: {count}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let enrichment = self.store.enrichment_status()?;
        let resource_summary = format!(
            "index priority: {}; last shrink: {}\n{}\nL1 caps: {}{}",
            priority["state"].as_str().unwrap_or("not attempted"),
            shrink["state"].as_str().unwrap_or("not requested"),
            enrichment.summary(),
            source_caps,
            if self.file_capacity_reached(counts.files)? {
                format!("\n{}", crate::sync::MAX_FILES_MESSAGE)
            } else {
                String::new()
            }
        );

        Ok(format!(
            "{} files, {} units, {} truncated, {} L1 pending, {} files deferred by caps; model preparation: {}\n{}\n{}\n{}",
            counts.files,
            counts.units,
            counts.truncated_files,
            counts.l1_pending_files,
            deferred_files.len(),
            model_summary.trim(),
            embedding.summary(),
            queue.summary(),
            format!("{rerank_summary}\n{health_summary}\n{history_summary}\n{pr_summary}\n{histories_summary}\n{modules_summary}\n{linker_summary}\n{resource_summary}")
                .trim_end()
        ))
    }

    /// Reports existing CLI status fields from shared storage and configuration sources.
    pub(crate) fn status(&self) -> Result<Value> {
        let started = Instant::now();
        let inspection = self.inspect()?;
        let counts = self.store.counts()?;
        let footprint = self.store.footprint()?;
        let queue = queue::status::snapshot(&self.store, unix_seconds()?)?;
        let embedding = queue::inspection::read_configured(&self.store, &self.config)?;
        let rerank_auto = super::rerank::auto::read_status(&self.store, &self.config)?;
        let rerank_health = crate::models::rerank::health::read(&self.store, &self.config)?;
        let details = super::status::read(&self.store, &self.config)?;
        let mut result = json!({
            "query_latency_ms": self.query_latencies.snapshot(),
            "ladder": details.ladder,
            "profile": details.profile,
            "models": details.models,
            "budgets": details.budgets,
            "rerank_auto": rerank_auto,
            "last_local_reranker_check": rerank_health,
            "l2_pending": embedding.l2.pending_jobs,
            "l2": embedding.l2,
            "endpoints": embedding.endpoints,
            "endpoint_accounting": "potential_dispatches_including_retries",
            "unknown_endpoint_dispatches": embedding.unknown_endpoint_dispatches,
            "queue": queue.jobs,
            "queue_status": queue,
            "root": self.root,
            "head": inspection.head,
            "indexed_head": inspection.indexed_head,
            // Pending catalogue work is not a repository-wide dirty-file count.
            "dirty_files": null,
            "dirty_files_state": "not_checked",
            "model_preparation": self.store.get_meta("model_preparation")?.map(|value| serde_json::from_str::<Value>(&value)).transpose()?,
            "history": self.store.get_meta("history_status")?.map(|value| serde_json::from_str::<Value>(&value)).transpose()?,
            "histories": self.store.get_meta("symbol_history_status")?.map(|value| serde_json::from_str::<Value>(&value)).transpose()?,
            "prs": self.store.get_meta("pr_status")?.map(|value| serde_json::from_str::<Value>(&value)).transpose()?,
            "commits": self.store.commit_count(&||Ok(()))?,
            "history_enabled": self.config.commits.enabled,
            "l1_scan_generation": self.store.l1_generation()?,
            "l1_completed_generation": self.store.get_meta("l1_completed_generation")?,
            "files": counts.files,
            "units": counts.units,
            "symbols": counts.symbols,
            "edges": counts.edges,
            "truncated_files": counts.truncated_files,
            "l1_pending_files": counts.l1_pending_files,
            "graph_pending": self.store.get_meta("graph_pending")?.as_deref() != Some("0"),
            "l1_deferred": self.store.l1_deferred_files()?.iter().map(|file| json!({
                "path":file.path,"reason":file.reason,"effective_level":0,"retained_snapshot":file.retained_snapshot,
            })).collect::<Vec<_>>(),
            "caps": {
                "max_files": self.config.limits.max_files,
                "file_capacity_reached": self.file_capacity_reached(counts.files)?,
                "max_units_per_file": self.config.limits.max_units_per_file,
                "max_open_bytes": self.config.limits.max_open_bytes,
                "truncated_files": counts.truncated_files,
            },
            "footprint_bytes": footprint.bytes,
            "tables": footprint.tables.iter().map(|table| json!({"name":table.name,"bytes":table.bytes})).collect::<Vec<_>>(),
            "fresh": inspection.freshness(),
            "took_ms": started.elapsed().as_millis(),
        });
        result["discovery"] = self
            .store
            .get_meta("discovery_status")?
            .map(|value| serde_json::from_str::<Value>(&value))
            .transpose()?
            .unwrap_or(Value::Null);
        result["l3"] = serde_json::to_value(self.store.enrichment_status()?)?;
        result["limits"] = serde_json::to_value(&self.config.limits)?;
        result["source_caps"] = serde_json::to_value(
            self.store
                .source_cap_counts(self.config.limits.max_file_bytes()?.0)?,
        )?;
        result["caps"]["file_capacity_message"] = if self.file_capacity_reached(counts.files)? {
            json!(crate::sync::MAX_FILES_MESSAGE)
        } else {
            Value::Null
        };
        for key in ["index_priority", "last_shrink"] {
            result[key] = self
                .store
                .get_meta(key)?
                .map(|value| serde_json::from_str::<Value>(&value))
                .transpose()?
                .unwrap_or(Value::Null);
        }
        result["response_caps"] = self.query_latencies.cap_snapshot();
        result["modules"] = self
            .store
            .get_meta("modules_status")?
            .map(|value| serde_json::from_str::<Value>(&value))
            .transpose()?
            .unwrap_or(Value::Null);
        result["linker"] = self
            .store
            .get_meta("linker_status")?
            .map(|value| serde_json::from_str::<Value>(&value))
            .transpose()?
            .unwrap_or(Value::Null);
        Ok(result)
    }
}

fn unix_seconds() -> Result<i64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| Error::Clock)?;
    i64::try_from(duration.as_secs()).map_err(|_| Error::Clock)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::{fusion::Mode, tests::add};
    use std::process::Command;

    fn persist_fixture_config(context: &Context) {
        fs::create_dir_all(context.config_root.join(".deepkit")).unwrap();
        fs::write(
            context.config_root.join(".deepkit/config.toml"),
            toml::to_string(&context.config).unwrap(),
        )
        .unwrap();
    }

    fn complete_fixture_source(context: &Context, path: &str) {
        let bytes = fs::read(context.root.join(path)).unwrap();
        let hash = blake3::hash(&bytes);
        let (kind, language): (String, Option<String>) = context
            .store
            .writer()
            .unwrap()
            .query_row("SELECT kind,lang FROM files WHERE path=?1", [path], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        let kind = store::FileKind::from_db(&kind).unwrap();
        let treatment =
            crate::treat::Router::new(&context.config.treatments, context.config.llm.enabled)
                .unwrap()
                .route(path, kind);
        let fingerprint = sync::l1::file_fingerprint(
            &context.config,
            &sync::l1::repo_name(&context.root, &context.config),
            sync::l1::FingerprintFile {
                path,
                content_hash: hash.as_bytes(),
                language: language.as_deref(),
                kind,
                capped: false,
            },
            &treatment,
        );
        let writer = context.store.writer().unwrap();
        writer
            .execute(
                "UPDATE files SET blake3=?1,l1_desired=?2,l1_fingerprint=?2 WHERE path=?3",
                rusqlite::params![hash.as_bytes().as_slice(), fingerprint.as_slice(), path],
            )
            .unwrap();
        writer
            .execute("UPDATE files SET l1_pending=0 WHERE path=?1", [path])
            .unwrap();
    }

    fn fixture() -> (tempfile::TempDir, Context) {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join(".deepkit")).unwrap();
        fs::write(
            root.path().join(".deepkit/config.toml"),
            "[embedder]\nprofile='local'\n",
        )
        .unwrap();
        let context = Context::load(root.path()).unwrap();
        add(
            &context.store,
            "src/retry.rs",
            "should_retry",
            "retry payment",
            None,
        )
        .unwrap();
        fs::create_dir(root.path().join("src")).unwrap();
        fs::write(root.path().join("src/retry.rs"), "retry payment").unwrap();
        complete_fixture_source(&context, "src/retry.rs");
        (root, context)
    }

    #[test]
    fn cap_observer_does_not_invent_a_rerank_deadline_from_query_timeout_envelopes() {
        let (_root, context) = fixture();
        context.observe_response_caps(
            "search",
            &Ok(json!({"index":{"reason":"query_timeout","rerank":"timeout"}})),
        );
        let first = context.query_latencies.cap_snapshot();
        assert_eq!(first["events"]["query_timeout_ms"]["count"], 1);
        assert!(first["events"]["rerank_timeout_ms"].is_null());
        context.observe_response_caps("search", &Ok(json!({"index":{"rerank":"timeout"}})));
        let second = context.query_latencies.cap_snapshot();
        assert_eq!(second["events"]["rerank_timeout_ms"]["count"], 1);
        assert_eq!(second["events"]["query_timeout_ms"]["count"], 1);
    }

    #[test]
    fn inspection_bounds_reader_admission_and_retries_without_status_scans() {
        use std::time::Duration;
        let (_root, mut context) = fixture();
        context.store = Arc::new(store::open(":memory:", Default::default()).unwrap());
        for cancel in [false, true] {
            let cancelled = AtomicBool::new(false);
            let held = context.store.writer().unwrap();
            std::thread::scope(|scope| {
                let (send, receive) = std::sync::mpsc::channel();
                let context = &context;
                let cancelled = &cancelled;
                scope.spawn(move || {
                    let work = sync::l1::Work {
                        deadline: Some(
                            Instant::now() + Duration::from_millis(if cancel { 5_000 } else { 20 }),
                        ),
                        cancelled: Some(cancelled),
                    };
                    send.send(matches!(
                        context.inspect_with_work(work),
                        Err(Error::Search(super::super::Error::Deadline))
                    ))
                    .unwrap();
                });
                if cancel {
                    std::thread::sleep(Duration::from_millis(20));
                    cancelled.store(true, Ordering::Release);
                }
                let result = receive.recv_timeout(Duration::from_millis(500));
                drop(held);
                assert!(
                    result.unwrap(),
                    "inspection must stop before reader release"
                );
            });
            assert_eq!(
                context
                    .inspect_with_work(Default::default())
                    .unwrap()
                    .pending,
                0
            );
        }
        // Unrelated status tables are deliberately unavailable in this fixture.
        // Request inspection needs only meta/files; ordinary status still diagnoses it.
        context
            .store
            .writer()
            .unwrap()
            .execute_batch("ALTER TABLE symbols RENAME TO unavailable_symbols;")
            .unwrap();
        assert!(context.store.counts().is_err());
        assert!(context.inspect().is_err());
        assert_eq!(
            context
                .inspect_with_work(Default::default())
                .unwrap()
                .pending,
            0
        );
    }

    fn request(mode: Mode) -> pipeline::Request {
        pipeline::Request {
            query: "retry payment".into(),
            scope: None,
            k: 10,
            mode,
            exclude_ids: vec![],
        }
    }

    fn insert_fixture_vector(context: &Context, unit_id: i64) -> (String, [u8; 32]) {
        use store::vectors;
        let (prepared, destination) = context
            .store
            .read_snapshot(|tx| {
                queue::destination::prepare_file(
                    tx,
                    &context.root,
                    &context.config_root,
                    &context.config,
                    unit_id,
                )
            })
            .unwrap()
            .unwrap()
            .unwrap();
        let binding = destination.binding(&context.config);
        let slug = binding.destination_slug();
        context
            .store
            .ensure_vector_table(vectors::VectorTableSpec {
                slug: &slug,
                model: &binding.model,
                profile: &binding.profile,
                dims: binding.dims as usize,
                created_at: 0,
                active: true,
            })
            .unwrap();
        context.store.bind_embedding_table(&slug, &binding).unwrap();
        context
            .store
            .insert_vector(vectors::VectorInsert {
                table_slug: &slug,
                unit_id,
                ctx_hash: prepared.context_hash(),
                embedding_hash: &prepared.current().target_hash,
                embedding: &vec![1.0; binding.dims as usize],
            })
            .unwrap();
        (slug, prepared.current().desired_hash)
    }

    fn publish_fixture_vector(context: &Context, unit_id: i64) {
        let (slug, desired) = insert_fixture_vector(context, unit_id);
        context.store.writer().unwrap().execute(
            "INSERT INTO embedding_publications(unit_id,table_slug,desired_hash) VALUES(?1,?2,?3) ON CONFLICT(unit_id) DO UPDATE SET table_slug=excluded.table_slug,desired_hash=excluded.desired_hash",
            rusqlite::params![unit_id, slug, desired.as_slice()],
        ).unwrap();
    }

    async fn assert_fixture_l2(context: &Context, expected: &[(&str, &str)]) -> Value {
        let changes = context.store.writer().unwrap().total_changes();
        let response = context.search(request(Mode::Lexical), false).await.unwrap();
        let hits = response["hits"].as_array().unwrap();
        assert_eq!(hits.len(), expected.len(), "{response}");
        for (path, state) in expected {
            let hit = hits.iter().find(|hit| hit["path"] == *path).unwrap();
            assert_eq!(hit["fresh"]["l2"], *state, "{response}");
            assert_eq!(hit["fresh"]["l1"], "current", "{response}");
            assert_eq!(hit["fresh"]["state"], "fresh", "{response}");
        }
        assert_eq!(response["index"]["partial"], false, "{response}");
        assert_eq!(context.store.writer().unwrap().total_changes(), changes);
        response
    }

    #[tokio::test]
    async fn lexical_search_keeps_verified_l1_during_provider_only_transition() {
        let (_root, mut context) = fixture();
        let captured = context.clone();
        let baseline = captured
            .search(request(Mode::Lexical), false)
            .await
            .unwrap();
        assert_eq!(baseline["hits"].as_array().unwrap().len(), 1, "{baseline}");

        context.config.embedder.profile = "ollama".into();
        persist_fixture_config(&context);
        let switched = captured
            .search(request(Mode::Lexical), false)
            .await
            .unwrap();
        assert_eq!(switched["hits"][0]["id"], baseline["hits"][0]["id"]);
        assert_eq!(switched["hits"][0]["fresh"]["l1"], "current");
        assert_eq!(switched["hits"][0]["fresh"]["l2"], "pending");
        assert_eq!(switched["index"]["partial"], false, "{switched}");

        // A search-policy edit is not a provider-only transition.
        context.config.search.max_hits_per_file = 1;
        persist_fixture_config(&context);
        let refused = captured
            .search(request(Mode::Lexical), false)
            .await
            .unwrap();
        assert_eq!(refused["hits"], json!([]), "{refused}");
        assert_eq!(refused["index"]["partial"], true);
    }

    #[tokio::test]
    async fn search_l2_uses_publication_evidence_and_current_policy_without_writes() {
        let (_root, mut context) = fixture();
        context
            .store
            .writer()
            .unwrap()
            .execute("UPDATE files SET max_level=3,l1_pending=0", [])
            .unwrap();
        complete_fixture_source(&context, "src/retry.rs");
        let occurrence = context.store.unit_by_id(1).unwrap().unwrap();
        let read = |context: &Context| {
            let context = context.clone();
            async move { context.search(request(Mode::Lexical), false).await.unwrap() }
        };
        assert_eq!(read(&context).await["hits"][0]["fresh"]["l2"], "pending");
        let (slug, desired) = insert_fixture_vector(&context, occurrence.id);
        assert_eq!(read(&context).await["hits"][0]["fresh"]["l2"], "pending");
        context.store.writer().unwrap().execute(
            "INSERT INTO embedding_publications(unit_id,table_slug,desired_hash) VALUES(?1,?2,?3)",
            rusqlite::params![occurrence.id, slug, desired.as_slice()],
        ).unwrap();
        let changes = context.store.writer().unwrap().total_changes();
        let current = read(&context).await;
        assert_eq!(current["hits"][0]["fresh"]["l2"], "current", "{current}");
        assert_eq!(current["index"]["partial"], false);
        assert_eq!(context.store.writer().unwrap().total_changes(), changes);
        context
            .store
            .writer()
            .unwrap()
            .execute(
                "UPDATE embedding_publications SET desired_hash=zeroblob(32)",
                [],
            )
            .unwrap();
        assert_eq!(read(&context).await["hits"][0]["fresh"]["l2"], "pending");
        // A changed policy is refused until L1 is completed under that policy.
        let treatment = context
            .config
            .treatments
            .iter_mut()
            .find(|t| t.name == "source")
            .unwrap();
        treatment.max_level = 1;
        persist_fixture_config(&context);
        let refused = read(&context).await;
        assert_eq!(refused["hits"], json!([]));
        assert_eq!(
            refused["index"]["freshness_refusals"]["l2_authority_unavailable"],
            1
        );
        complete_fixture_source(&context, "src/retry.rs");
        let ineligible = read(&context).await;
        assert_eq!(ineligible["hits"][0]["fresh"]["l2"], "none", "{ineligible}");
        assert_eq!(ineligible["index"]["partial"], false);
    }

    #[tokio::test]
    async fn search_l2_routes_configured_tiers_and_refuses_changed_provider_spaces() {
        let server = wiremock::MockServer::start().await;
        let (_root, mut context) = fixture();
        context.config.embedder.profile = "remote".into();
        context.config.embedder.dims = 2;
        context.config.embedder.remote.url = format!("{}/v1", server.uri());
        context.config.embedder.ollama.url = server.uri();
        persist_fixture_config(&context);
        context
            .store
            .writer()
            .unwrap()
            .execute("UPDATE files SET max_level=3", [])
            .unwrap();
        complete_fixture_source(&context, "src/retry.rs");
        let path = "retries.md";
        let body = "retry payment";
        fs::write(context.root.join(path), body).unwrap();
        let hash = blake3::hash(body.as_bytes());
        let file_id = context
            .store
            .upsert_file(store::FileUpsert {
                path,
                blake3: hash.as_bytes(),
                size: body.len() as i64,
                mtime_ns: 1,
                lang: Some("markdown"),
                kind: store::FileKind::Doc,
                git_blob: None,
                max_level: 3,
                treatment: "docs",
                indexed_at: Some(1),
                needs_vlm: false,
            })
            .unwrap();
        let doc = context
            .store
            .insert_indexed_unit(
                store::UnitInsert {
                    hash: hash.as_bytes(),
                    source_key: None,
                    duplicate_ordinal: 0,
                    file_id: Some(file_id),
                    kind: store::UnitKind::DocSection,
                    byte_start: Some(0),
                    byte_end: Some(body.len() as i64),
                    line_start: Some(1),
                    line_end: Some(1),
                    scope_path: Some("Retries"),
                    header: "Retries",
                    text: body,
                    tokens: 2,
                    ladder_level: 1,
                    parent_unit: None,
                    created_at: 1,
                    updated_at: 1,
                    fts: store::FtsColumns {
                        header: "Retries",
                        idents: "",
                        body,
                        prose: body,
                    },
                },
                &[],
                &[],
            )
            .unwrap()
            .unit_id;
        complete_fixture_source(&context, path);
        publish_fixture_vector(&context, 1);
        publish_fixture_vector(&context, doc);
        let both_current = [("src/retry.rs", "current"), (path, "current")];
        let initial = assert_fixture_l2(&context, &both_current).await;
        context.config.embedder.docs = Some(config::DocsEmbedder {
            profile: "ollama".into(),
            kinds: vec!["doc_section".into()],
        });
        persist_fixture_config(&context);
        // A docs-only route edit preserves the unchanged code publication;
        // the selected docs destination still requires its own publication.
        assert_fixture_l2(&context, &[("src/retry.rs", "current"), (path, "pending")]).await;
        // Old-tier bytes are retained but cannot prove selected-tier freshness.
        assert_eq!(context.store.count_vectors().unwrap(), 2);
        publish_fixture_vector(&context, doc);
        assert_eq!(context.store.count_vectors().unwrap(), 3);
        assert_fixture_l2(&context, &both_current).await;
        let configured = context.config.clone();
        for change in [
            "docs_endpoint",
            "docs_model",
            "code_endpoint",
            "code_model",
            "dimensions",
            "kinds",
            "disable",
        ] {
            context.config = configured.clone();
            match change {
                "docs_endpoint" => {
                    context.config.embedder.ollama.url = format!("{}/replacement", server.uri());
                }
                "docs_model" => {
                    context.config.embedder.ollama.model = "replacement-model".into();
                }
                "code_endpoint" => {
                    context.config.embedder.remote.url = format!("{}/replacement", server.uri());
                }
                "code_model" => {
                    context.config.embedder.remote.model = "replacement-model".into();
                }
                "dimensions" => {
                    context.config.embedder.dims = 3;
                }
                "kinds" => {
                    context.config.embedder.docs.as_mut().unwrap().kinds = vec!["symbol".into()];
                }
                "disable" => {
                    context.config.embedder.docs = None;
                }
                _ => unreachable!(),
            };
            persist_fixture_config(&context);
            let code_state = if change == "disable" {
                "current"
            } else {
                "pending"
            };
            let changed =
                assert_fixture_l2(&context, &[("src/retry.rs", code_state), (path, "pending")])
                    .await;
            for before in initial["hits"].as_array().unwrap() {
                let after = changed["hits"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|hit| hit["path"] == before["path"])
                    .unwrap();
                assert_eq!(after["id"], before["id"], "{change}");
                assert_eq!(after["lines"], before["lines"], "{change}");
            }
            assert_eq!(context.store.count_vectors().unwrap(), 3, "{change}");
            context.config = configured.clone();
            persist_fixture_config(&context);
            assert_fixture_l2(&context, &both_current).await;
        }
        assert!(server.received_requests().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn search_l2_context_changes_preserve_occurrence_and_require_republication() {
        let server = wiremock::MockServer::start().await;
        let (_root, mut context) = fixture();
        context.config.embedder.profile = "remote".into();
        context.config.embedder.remote.url = server.uri();
        context.config.embedder.dims = 2;
        persist_fixture_config(&context);
        context
            .store
            .writer()
            .unwrap()
            .execute("UPDATE files SET max_level=3", [])
            .unwrap();
        complete_fixture_source(&context, "src/retry.rs");
        publish_fixture_vector(&context, 1);
        let initial = assert_fixture_l2(&context, &[("src/retry.rs", "current")]).await;
        let configured = context.config.clone();
        for change in ["breadcrumb", "kind_line", "text_cap", "header", "repo_name"] {
            context.config = configured.clone();
            match change {
                "breadcrumb" => context.config.embedder.breadcrumb = false,
                "kind_line" => context.config.embedder.kind_line = false,
                "text_cap" => context.config.embedder.max_doc_tokens = 1,
                "header" => {
                    context
                        .store
                        .writer()
                        .unwrap()
                        .execute("UPDATE units SET header='new context' WHERE id=1", [])
                        .unwrap();
                }
                "repo_name" => context.config.index.repo_name = "renamed-repository".into(),
                _ => unreachable!(),
            }
            persist_fixture_config(&context);
            if change == "repo_name" {
                let changes = context.store.writer().unwrap().total_changes();
                let refused = context.search(request(Mode::Lexical), false).await.unwrap();
                assert_eq!(refused["hits"], json!([]), "{refused}");
                assert_eq!(
                    refused["index"]["freshness_refusals"]["l2_authority_unavailable"],
                    1
                );
                assert_eq!(context.store.writer().unwrap().total_changes(), changes);
                complete_fixture_source(&context, "src/retry.rs");
            }
            let pending = assert_fixture_l2(&context, &[("src/retry.rs", "pending")]).await;
            assert_eq!(
                pending["hits"][0]["id"], initial["hits"][0]["id"],
                "{change}"
            );
            assert_eq!(
                pending["hits"][0]["lines"], initial["hits"][0]["lines"],
                "{change}"
            );
            assert_eq!(context.store.count_vectors().unwrap(), 1);
            publish_fixture_vector(&context, 1);
            assert_fixture_l2(&context, &[("src/retry.rs", "current")]).await;
            context.config = configured.clone();
            persist_fixture_config(&context);
            context
                .store
                .writer()
                .unwrap()
                .execute("UPDATE units SET header='should_retry' WHERE id=1", [])
                .unwrap();
            complete_fixture_source(&context, "src/retry.rs");
            assert_fixture_l2(&context, &[("src/retry.rs", "pending")]).await;
            publish_fixture_vector(&context, 1);
            assert_fixture_l2(&context, &[("src/retry.rs", "current")]).await;
        }
        assert!(server.received_requests().await.unwrap().is_empty());
    }

    #[test]
    fn query_l2_preserves_refusal_deadline_and_cancellation() {
        let (_root, context) = fixture();
        let occurrence = context.store.unit_by_id(1).unwrap().unwrap();
        assert!(
            query_l2(&context, &occurrence, Default::default())
                .unwrap()
                .is_some()
        );
        let cancelled = AtomicBool::new(true);
        for work in [
            sync::l1::Work {
                cancelled: Some(&cancelled),
                ..Default::default()
            },
            sync::l1::Work {
                deadline: Some(Instant::now()),
                ..Default::default()
            },
        ] {
            assert_eq!(
                query_l2(&context, &occurrence, work).unwrap_err().code(),
                "timeout"
            );
        }
        fs::write(context.root.join("src/retry.rs"), "changed source").unwrap();
        assert_eq!(
            query_l2(&context, &occurrence, Default::default()).unwrap(),
            None
        );
        fs::write(context.root.join("src/retry.rs"), "retry payment").unwrap();
        assert!(
            query_l2(&context, &occurrence, Default::default())
                .unwrap()
                .is_some()
        );
    }

    #[tokio::test]
    async fn search_final_selection_retains_graph_associations_and_ranks() {
        for direction in ["incoming", "outgoing"] {
            for mutation in [
                "rank",
                "precise_rank",
                "masked_rank",
                "source",
                "candidate",
                "edge",
                "identity",
                "duplicate",
            ] {
                let (_root, context) = fixture();
                add(
                    &context.store,
                    "src/hidden.rs",
                    "hidden",
                    "unrelated",
                    Some("should_retry"),
                )
                .unwrap();
                fs::write(context.root.join("src/hidden.rs"), "unrelated").unwrap();
                complete_fixture_source(&context, "src/hidden.rs");
                {
                    let writer = context.store.writer().unwrap();
                    writer
                        .execute("UPDATE symbols SET pagerank=0.1 WHERE name='hidden'", [])
                        .unwrap();
                    if mutation == "masked_rank" {
                        writer.execute("INSERT INTO symbols(id,file_id,unit_id,name,qualified,kind,signature,line_start,line_end,pagerank)
                        SELECT id+1000,file_id,unit_id,name,qualified,kind,signature,line_start,line_end,0.8 FROM symbols WHERE name='hidden'", []).unwrap();
                        writer.execute("INSERT INTO edges(src_symbol,dst_name,dst_symbol,kind,file_id,line)
                        SELECT src_symbol+1000,dst_name,dst_symbol,kind,file_id,line FROM edges", []).unwrap();
                    }
                    if direction == "outgoing" {
                        writer.execute("UPDATE edges SET src_symbol=(SELECT id FROM symbols WHERE name='should_retry'),dst_name='hidden'", []).unwrap();
                    }
                }
                let make_request = || {
                    let mut query = request(Mode::Lexical);
                    query.k = 1;
                    query
                };
                let original = context.search(make_request(), true).await.unwrap();
                assert_eq!(original["index"]["partial"], false);
                assert_eq!(original["hits"].as_array().unwrap().len(), 1);
                let worker = context.clone();
                let started = Instant::now();
                let work = QueryWork::new(started, 5000).unwrap();
                let result = context
                .search_within_checked(make_request(), true, started, &work, move || {
                    let writer = worker.store.writer().unwrap();
                    writer.execute(match mutation {
                    "rank" | "masked_rank" => "UPDATE symbols SET pagerank=0.2 WHERE name='hidden' AND pagerank<0.8",
                    "precise_rank" => "UPDATE symbols SET pagerank=0.10000000000000002 WHERE name='hidden'",
                    "source" => "UPDATE symbols SET qualified='changed' WHERE name='should_retry'",
                    "candidate" => "UPDATE symbols SET signature='changed' WHERE name='hidden'",
                    "edge" => "UPDATE edges SET kind='import',line=2",
                    "identity" => "UPDATE edges SET rowid=rowid+1000",
                    "duplicate" => "INSERT INTO edges SELECT * FROM edges",
                    _ => unreachable!(),
                }, []).unwrap();
                })
                .await
                .unwrap();
                assert_eq!(result["index"]["partial"], true, "{mutation}: {result}");
                assert_eq!(result["hits"], json!([]));
                assert_eq!(
                    result["index"]["freshness_refusals"]["selection_changed"],
                    1
                );
                assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
                let retry = context.search(make_request(), true).await.unwrap();
                assert_eq!(retry["index"]["partial"], false);
                assert_eq!(retry["hits"], original["hits"]);
            }
        }
    }

    #[tokio::test]
    async fn search_final_selection_retains_exact_symbol_associations() {
        for mutation in ["qualified", "signature", "range", "identity"] {
            let (_root, context) = fixture();
            let make_request = || {
                let mut query = request(Mode::Lexical);
                query.query = "should_retry".into();
                query.k = 1;
                query
            };
            let original = context.search(make_request(), true).await.unwrap();
            assert_eq!(original["index"]["partial"], false);
            let worker = context.clone();
            let started = Instant::now();
            let work = QueryWork::new(started, 5000).unwrap();
            let result = context
                .search_within_checked(make_request(), true, started, &work, move || {
                    let writer = worker.store.writer().unwrap();
                    writer
                        .execute(
                            match mutation {
                                "qualified" => {
                                    "UPDATE symbols SET qualified='changed::should_retry'"
                                }
                                "signature" => "UPDATE symbols SET signature='changed signature'",
                                "range" => "UPDATE symbols SET line_end=line_end+1",
                                "identity" => "INSERT INTO symbols(id,file_id,unit_id,name,qualified,kind,signature,line_start,line_end)
                                    SELECT id+1000,file_id,unit_id,name,qualified,kind,signature,line_start,line_end FROM symbols",
                                _ => unreachable!(),
                            },
                            [],
                        )
                        .unwrap();
                    if mutation == "identity" {
                        writer.execute("UPDATE fts_names SET ref_id=ref_id+1000 WHERE kind='symbol'", []).unwrap();
                    }
                })
                .await
                .unwrap();
            assert_eq!(result["index"]["partial"], true, "{mutation}: {result}");
            assert_eq!(result["hits"], json!([]));
            assert_eq!(
                result["index"]["freshness_refusals"]["selection_changed"],
                1
            );
            assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
            let retry = context.search(make_request(), true).await.unwrap();
            assert_eq!(retry["index"]["partial"], false);
            assert_eq!(retry["hits"], original["hits"]);
        }
    }

    #[tokio::test]
    async fn search_final_selection_retains_chunk_and_parent_byte_bounds() {
        for through_summary in [false, true] {
            for mutation in [
                "chunk_start",
                "chunk_end",
                "parent_start",
                "parent_end",
                "ladder",
            ] {
                let (_root, context) = fixture();
                let chunk = add(
                    &context.store,
                    "src/chunk.rs",
                    "chunk",
                    "retry payment",
                    None,
                )
                .unwrap();
                fs::write(context.root.join("src/chunk.rs"), "retry payment").unwrap();
                complete_fixture_source(&context, "src/chunk.rs");
                {
                    let writer = context.store.writer().unwrap();
                    writer.execute("UPDATE units SET kind='chunk',file_id=(SELECT file_id FROM units WHERE id=1),byte_start=2,byte_end=4 WHERE id=?1", [chunk]).unwrap();
                    if through_summary {
                        // Only the summary is retrieved; neither the chunk nor its
                        // selected containing symbol is an original candidate.
                        writer
                            .execute("DELETE FROM fts_units WHERE rowid IN (1,?1)", [chunk])
                            .unwrap();
                    }
                }
                if through_summary {
                    let summary = add(
                        &context.store,
                        "src/summary.rs",
                        "summary",
                        "retry payment",
                        None,
                    )
                    .unwrap();
                    fs::write(context.root.join("src/summary.rs"), "retry payment").unwrap();
                    complete_fixture_source(&context, "src/summary.rs");
                    context
                        .store
                        .writer()
                        .unwrap()
                        .execute(
                            "UPDATE units SET kind='summary',parent_unit=?1 WHERE id=?2",
                            [chunk, summary],
                        )
                        .unwrap();
                }
                let make_request = || {
                    let mut query = request(Mode::Lexical);
                    query.k = 1;
                    query
                };
                let original = context.search(make_request(), true).await.unwrap();
                assert_eq!(original["index"]["partial"], false);
                assert_eq!(
                    original["hits"][0]["matches"].as_array().unwrap().len(),
                    if through_summary { 1 } else { 2 }
                );
                let generation = context.store.l1_generation().unwrap();
                let worker = context.clone();
                let started = Instant::now();
                let work = QueryWork::new(started, 5000).unwrap();
                let result = context
                    .search_within_checked(make_request(), true, started, &work, move || {
                        let writer = worker.store.writer().unwrap();
                        let (sql, id) = match mutation {
                            "chunk_start" => ("UPDATE units SET byte_start=1 WHERE id=?1", chunk),
                            "chunk_end" => ("UPDATE units SET byte_end=5 WHERE id=?1", chunk),
                            "parent_start" => ("UPDATE units SET byte_start=1 WHERE id=?1", 1),
                            "parent_end" => ("UPDATE units SET byte_end=12 WHERE id=?1", 1),
                            "ladder" => ("UPDATE units SET ladder_level=2 WHERE id=?1", chunk),
                            _ => unreachable!(),
                        };
                        writer.execute(sql, [id]).unwrap();
                    })
                    .await
                    .unwrap();
                assert_eq!(context.store.l1_generation().unwrap(), generation);
                assert_eq!(
                    result["index"]["partial"],
                    mutation != "ladder",
                    "summary={through_summary}/{mutation}: {result}"
                );
                assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
                if mutation != "ladder" {
                    assert_eq!(result["hits"], json!([]));
                    assert_eq!(
                        result["index"]["freshness_refusals"]["selection_changed"],
                        1
                    );
                }
                let retry = context.search(make_request(), true).await.unwrap();
                assert_eq!(retry["index"]["partial"], false);
                assert_eq!(retry["hits"], original["hits"]);
            }
        }
    }

    #[tokio::test]
    async fn search_final_selection_retains_intermediate_summary_links() {
        for mutation in ["shortcut", "alternate", "ladder", "unrelated"] {
            let (_root, context) = fixture();
            let mut chain = Vec::new();
            for name in ["matched", "middle", "tail", "alternate"] {
                let path = format!("src/{name}.rs");
                let id = add(
                    &context.store,
                    &path,
                    name,
                    "unrelated",
                    (name == "matched").then_some("should_retry"),
                )
                .unwrap();
                fs::write(context.root.join(&path), "unrelated").unwrap();
                complete_fixture_source(&context, &path);
                chain.push(id);
            }
            {
                let writer = context.store.writer().unwrap();
                for (id, parent) in [
                    (chain[0], chain[1]),
                    (chain[1], chain[2]),
                    (chain[2], 1),
                    (chain[3], 1),
                ] {
                    writer
                        .execute(
                            "UPDATE units SET kind='summary',parent_unit=?1 WHERE id=?2",
                            [parent, id],
                        )
                        .unwrap();
                }
            }
            let make_request = || {
                let mut query = request(Mode::Lexical);
                query.k = 1;
                query
            };
            let original = context.search(make_request(), true).await.unwrap();
            assert_eq!(original["index"]["partial"], false);
            assert_eq!(original["hits"][0]["matches"].as_array().unwrap().len(), 2);
            let generation = context.store.l1_generation().unwrap();
            let worker = context.clone();
            let started = Instant::now();
            let work = QueryWork::new(started, 5000).unwrap();
            let result = context
                .search_within_checked(make_request(), true, started, &work, move || {
                    let writer = worker.store.writer().unwrap();
                    match mutation {
                        "shortcut" | "alternate" => {
                            let parent = if mutation == "shortcut" { 1 } else { chain[3] };
                            writer
                                .execute(
                                    "UPDATE units SET parent_unit=?1 WHERE id=?2",
                                    [parent, chain[1]],
                                )
                                .unwrap();
                        }
                        "ladder" => {
                            writer
                                .execute("UPDATE units SET ladder_level=2 WHERE id=?1", [chain[1]])
                                .unwrap();
                        }
                        "unrelated" => {
                            writer
                                .execute(
                                    "UPDATE units SET parent_unit=?1 WHERE id=?2",
                                    [chain[2], chain[3]],
                                )
                                .unwrap();
                        }
                        _ => unreachable!(),
                    }
                })
                .await
                .unwrap();
            assert_eq!(context.store.l1_generation().unwrap(), generation);
            assert_eq!(
                result["index"]["partial"],
                matches!(mutation, "shortcut" | "alternate"),
                "{mutation}: {result}"
            );
            assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
            if matches!(mutation, "shortcut" | "alternate") {
                assert_eq!(result["hits"], json!([]));
                assert_eq!(
                    result["index"]["freshness_refusals"]["selection_changed"],
                    1
                );
            }
            let retry = context.search(make_request(), true).await.unwrap();
            assert_eq!(retry["index"]["partial"], false);
            assert_eq!(retry["hits"], original["hits"]);
        }
    }

    #[test]
    fn search_hidden_configuration_check_honors_stopped_work_and_retry() {
        let (_root, mut context) = fixture();
        add(
            &context.store,
            "src/hidden.rs",
            "hidden",
            "unrelated",
            Some("should_retry"),
        )
        .unwrap();
        fs::write(context.root.join("src/hidden.rs"), "unrelated").unwrap();
        complete_fixture_source(&context, "src/hidden.rs");
        let parsed = super::super::query::parse("retry payment", Mode::Lexical).unwrap();
        let (_, selection) = super::super::lexical_with_selection(
            &context.store,
            &parsed,
            1,
            None,
            &context.config.search,
            &[],
            &|| Ok(()),
        )
        .unwrap();
        assert!(!selection.publications.is_empty());
        let active = sync::l1::Work {
            deadline: None,
            cancelled: None,
        };
        assert_eq!(
            query_hidden_publications(&context, &selection, active).unwrap(),
            None
        );
        context.config.tokenize.split_subtokens = !context.config.tokenize.split_subtokens;
        let writes = context.store.writer().unwrap().total_changes();
        let cancelled = AtomicBool::new(true);
        for work in [
            sync::l1::Work {
                deadline: None,
                cancelled: Some(&cancelled),
            },
            sync::l1::Work {
                deadline: Some(Instant::now()),
                cancelled: None,
            },
        ] {
            assert!(matches!(
                query_hidden_publications(&context, &selection, work),
                Err(Error::Search(super::super::Error::Deadline))
            ));
        }
        cancelled.store(false, Ordering::Release);
        let retry = sync::l1::Work {
            deadline: None,
            cancelled: Some(&cancelled),
        };
        assert_eq!(
            query_hidden_publications(&context, &selection, retry).unwrap(),
            Some(std::collections::BTreeSet::new()),
        );
        assert_eq!(context.store.writer().unwrap().total_changes(), writes);
    }

    #[tokio::test]
    async fn search_initial_hidden_publication_refuses_incomplete_evidence_and_recovers() {
        for route in ["hidden", "collapsed", "intermediate"] {
            for mutation in [
                "config",
                "config_chunk",
                "config_repo",
                "config_policy",
                "pending",
                "desired",
                "completion",
                "both_missing",
                "mismatch",
                "malformed",
                "deferred",
            ] {
                let (_root, mut context) = fixture();
                let hidden = add(
                    &context.store,
                    "src/hidden.rs",
                    "hidden",
                    "unrelated",
                    Some("should_retry"),
                )
                .unwrap();
                fs::write(context.root.join("src/hidden.rs"), "unrelated").unwrap();
                complete_fixture_source(&context, "src/hidden.rs");
                let mut changed = hidden;
                if route == "intermediate" {
                    changed =
                        add(&context.store, "src/middle.rs", "middle", "middle", None).unwrap();
                    fs::write(context.root.join("src/middle.rs"), "middle").unwrap();
                    complete_fixture_source(&context, "src/middle.rs");
                    context
                        .store
                        .writer()
                        .unwrap()
                        .execute(
                            "UPDATE units SET kind='summary',parent_unit=1 WHERE id=?1",
                            [changed],
                        )
                        .unwrap();
                }
                if route != "hidden" {
                    let parent = if route == "collapsed" { 1 } else { changed };
                    context
                        .store
                        .writer()
                        .unwrap()
                        .execute(
                            "UPDATE units SET kind='summary',parent_unit=?1 WHERE id=?2",
                            [parent, hidden],
                        )
                        .unwrap();
                }
                add(
                    &context.store,
                    "src/other.rs",
                    "other",
                    "retry payment",
                    None,
                )
                .unwrap();
                fs::write(context.root.join("src/other.rs"), "retry payment").unwrap();
                complete_fixture_source(&context, "src/other.rs");
                let path = if route == "intermediate" {
                    "src/middle.rs"
                } else {
                    "src/hidden.rs"
                };
                let make_request = || {
                    let mut query = request(Mode::Lexical);
                    query.k = if route == "hidden" { 1 } else { 2 };
                    query
                };
                let original = context.search(make_request(), true).await.unwrap();
                assert_eq!(original["hits"][0]["path"], "src/retry.rs");
                assert_eq!(original["index"]["partial"], false);
                let generation = context.store.l1_generation().unwrap();
                if mutation.starts_with("config") {
                    match mutation {
                        "config" => {
                            context.config.tokenize.split_subtokens =
                                !context.config.tokenize.split_subtokens
                        }
                        "config_chunk" => context.config.chunk.target_tokens += 1,
                        "config_repo" => context.config.index.repo_name = "changed-context".into(),
                        "config_policy" => context.config.treatments.insert(
                            0,
                            crate::config::Treatment {
                                name: "hidden-local".into(),
                                patterns: vec![path.into()],
                                chunker: None,
                                max_level: 1,
                                enrich: vec![],
                                never_send: true,
                            },
                        ),
                        _ => unreachable!(),
                    }
                    persist_fixture_config(&context);
                    for source in [
                        "src/retry.rs",
                        "src/other.rs",
                        "src/hidden.rs",
                        "src/middle.rs",
                    ] {
                        if source != path && context.root.join(source).exists() {
                            complete_fixture_source(&context, source);
                        }
                    }
                }
                context.store.writer().unwrap().execute(match mutation {
                    "config" | "config_chunk" | "config_repo" | "config_policy" => "UPDATE files SET l1_pending=l1_pending WHERE path=?1",
                    "pending" => "UPDATE files SET l1_pending=1 WHERE path=?1",
                    "desired" => "UPDATE files SET l1_desired=NULL WHERE path=?1",
                    "completion" => "UPDATE files SET l1_fingerprint=NULL WHERE path=?1",
                    "both_missing" => "UPDATE files SET l1_desired=NULL,l1_fingerprint=NULL WHERE path=?1",
                    "mismatch" => "UPDATE files SET l1_desired=zeroblob(32) WHERE path=?1",
                    "malformed" => "UPDATE files SET l1_desired=x'00',l1_fingerprint=x'00' WHERE path=?1",
                    "deferred" => "UPDATE files SET l1_deferred_reason='max_units_per_file' WHERE path=?1",
                    _ => unreachable!(),
                }, [path]).unwrap();
                let writes = context.store.writer().unwrap().total_changes();
                let result = context.search(make_request(), true).await.unwrap();
                assert_eq!(context.store.writer().unwrap().total_changes(), writes);
                assert_eq!(
                    result["index"]["partial"], true,
                    "{route}/{mutation}: {result}"
                );
                assert_eq!(result["index"]["lexical"], "incomplete");
                assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
                assert_eq!(context.store.l1_generation().unwrap(), generation);
                let removed = usize::from(route != "hidden");
                assert_eq!(
                    result["index"]["freshness_refusals"]["contributor_l1_incomplete"],
                    removed
                );
                assert_eq!(result["index"]["suppressed_hits"], removed);
                if route == "hidden" {
                    assert_eq!(result["hits"], original["hits"]);
                } else {
                    assert_eq!(result["hits"], json!([original["hits"][1].clone()]));
                    assert_eq!(result["hits"][0]["path"], "src/other.rs");
                }
                context
                    .store
                    .writer()
                    .unwrap()
                    .execute(
                        "UPDATE files SET l1_deferred_reason=NULL WHERE path=?1",
                        [path],
                    )
                    .unwrap();
                complete_fixture_source(&context, path);
                let retry = context.search(make_request(), true).await.unwrap();
                assert_eq!(retry["index"]["partial"], false);
                assert_eq!(retry["hits"], original["hits"]);
            }
        }
    }

    #[tokio::test]
    async fn search_final_selection_retains_hidden_source_publication() {
        for route in ["hidden", "collapsed", "intermediate"] {
            for mutation in [
                "pending",
                "desired",
                "completion",
                "hash",
                "language",
                "kind",
                "deferred",
                "ladder",
                "filesystem",
                "unrelated",
            ] {
                let (_root, context) = fixture();
                let hidden = add(
                    &context.store,
                    "src/hidden.rs",
                    "hidden",
                    "unrelated",
                    Some("should_retry"),
                )
                .unwrap();
                fs::write(context.root.join("src/hidden.rs"), "unrelated").unwrap();
                complete_fixture_source(&context, "src/hidden.rs");
                let mut changed = hidden;
                if route == "intermediate" {
                    changed =
                        add(&context.store, "src/middle.rs", "middle", "middle", None).unwrap();
                    fs::write(context.root.join("src/middle.rs"), "middle").unwrap();
                    complete_fixture_source(&context, "src/middle.rs");
                    context
                        .store
                        .writer()
                        .unwrap()
                        .execute(
                            "UPDATE units SET kind='summary',parent_unit=1 WHERE id=?1",
                            [changed],
                        )
                        .unwrap();
                }
                if route != "hidden" {
                    let parent = if route == "collapsed" { 1 } else { changed };
                    context
                        .store
                        .writer()
                        .unwrap()
                        .execute(
                            "UPDATE units SET kind='summary',parent_unit=?1 WHERE id=?2",
                            [parent, hidden],
                        )
                        .unwrap();
                }
                add(&context.store, "src/other.rs", "other", "opaque", None).unwrap();
                fs::write(context.root.join("src/other.rs"), "opaque").unwrap();
                complete_fixture_source(&context, "src/other.rs");
                let path = if route == "intermediate" {
                    "src/middle.rs"
                } else {
                    "src/hidden.rs"
                };
                let make_request = || {
                    let mut query = request(Mode::Lexical);
                    query.k = 1;
                    query
                };
                let original = context.search(make_request(), true).await.unwrap();
                assert_eq!(original["hits"][0]["path"], "src/retry.rs");
                assert_eq!(original["index"]["partial"], false);
                let generation = context.store.l1_generation().unwrap();
                let worker = context.clone();
                let started = Instant::now();
                let work = QueryWork::new(started, 5000).unwrap();
                let result = context.search_within_checked(make_request(), true, started, &work, move || {
                    if mutation == "filesystem" {
                        fs::write(worker.root.join(path), "changed bytes outside returned files").unwrap();
                        return;
                    }
                    let writer = worker.store.writer().unwrap();
                    if mutation == "unrelated" {
                        writer.execute("UPDATE files SET l1_pending=1 WHERE path='src/other.rs'", []).unwrap();
                        return;
                    }
                    writer.execute(match mutation {
                        "pending" => "UPDATE files SET l1_pending=1 WHERE id=(SELECT file_id FROM units WHERE id=?1)",
                        "desired" => "UPDATE files SET l1_desired=zeroblob(32) WHERE id=(SELECT file_id FROM units WHERE id=?1)",
                        "completion" => "UPDATE files SET l1_fingerprint=NULL WHERE id=(SELECT file_id FROM units WHERE id=?1)",
                        "hash" => "UPDATE files SET blake3=zeroblob(32) WHERE id=(SELECT file_id FROM units WHERE id=?1)",
                        "language" => "UPDATE files SET lang='python' WHERE id=(SELECT file_id FROM units WHERE id=?1)",
                        "kind" => "UPDATE files SET kind='doc' WHERE id=(SELECT file_id FROM units WHERE id=?1)",
                        "deferred" => "UPDATE files SET l1_deferred_reason='max_units_per_file' WHERE id=(SELECT file_id FROM units WHERE id=?1)",
                        "ladder" => "UPDATE files SET ladder_level=2 WHERE id=(SELECT file_id FROM units WHERE id=?1)",
                        _ => unreachable!(),
                    }, [changed]).unwrap();
                }).await.unwrap();
                let refused = !matches!(mutation, "ladder" | "filesystem" | "unrelated");
                assert_eq!(context.store.l1_generation().unwrap(), generation);
                assert_eq!(
                    result["index"]["partial"],
                    refused || mutation == "unrelated",
                    "{route}/{mutation}: {result}"
                );
                assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
                if refused {
                    assert_eq!(result["hits"], json!([]));
                    assert_eq!(
                        result["index"]["freshness_refusals"]["selection_changed"],
                        1
                    );
                } else {
                    assert_eq!(result["hits"], original["hits"]);
                }
                // Restore the completed source without changing occurrence rows.
                context.store.writer().unwrap().execute("UPDATE files SET lang='rust',kind='code',l1_deferred_reason=NULL WHERE path=?1", [path]).unwrap();
                fs::write(
                    context.root.join(path),
                    if route == "intermediate" {
                        "middle"
                    } else {
                        "unrelated"
                    },
                )
                .unwrap();
                complete_fixture_source(&context, path);
                // Unrelated pending work preserves these hits but makes global
                // coverage incomplete until that file is completed too.
                complete_fixture_source(&context, "src/other.rs");
                let retry = context.search(make_request(), true).await.unwrap();
                assert_eq!(retry["index"]["partial"], false);
                assert_eq!(retry["hits"], original["hits"]);
            }
        }
    }

    #[tokio::test]
    async fn search_final_selection_retains_hidden_occurrence_rows() {
        for collapsed in ["none", "winner", "hidden_parent"] {
            for mutation in ["header", "text", "range", "ladder"] {
                let (_root, context) = fixture();
                let hidden = add(
                    &context.store,
                    "src/hidden.rs",
                    "hidden",
                    "unrelated",
                    Some("should_retry"),
                )
                .unwrap();
                fs::write(context.root.join("src/hidden.rs"), "unrelated").unwrap();
                complete_fixture_source(&context, "src/hidden.rs");
                let changed = if collapsed == "hidden_parent" {
                    let parent = add(
                        &context.store,
                        "src/parent.rs",
                        "parent",
                        "unrelated parent",
                        None,
                    )
                    .unwrap();
                    fs::write(context.root.join("src/parent.rs"), "unrelated parent").unwrap();
                    complete_fixture_source(&context, "src/parent.rs");
                    context
                        .store
                        .writer()
                        .unwrap()
                        .execute(
                            "UPDATE units SET kind='summary',parent_unit=?1 WHERE id=?2",
                            [parent, hidden],
                        )
                        .unwrap();
                    parent
                } else {
                    hidden
                };
                if collapsed == "winner" {
                    context
                        .store
                        .writer()
                        .unwrap()
                        .execute(
                            "UPDATE units SET kind='summary',parent_unit=1 WHERE id=?1",
                            [hidden],
                        )
                        .unwrap();
                }
                let make_request = || {
                    let mut query = request(Mode::Lexical);
                    query.k = 1;
                    query
                };
                let original = context.search(make_request(), true).await.unwrap();
                assert_eq!(original["hits"][0]["path"], "src/retry.rs");
                assert_eq!(original["index"]["partial"], false);
                if collapsed == "winner" {
                    assert_eq!(original["hits"][0]["matches"].as_array().unwrap().len(), 2);
                }
                let generation = context.store.l1_generation().unwrap();
                let worker = context.clone();
                let started = Instant::now();
                let work = QueryWork::new(started, 5000).unwrap();
                let result = context.search_within_checked(make_request(), true, started, &work, move || {
                    let writer = worker.store.writer().unwrap();
                    writer.execute(match mutation {
                        "header" => "UPDATE units SET header='changed' WHERE id=?1",
                        "text" => "UPDATE unit_contents SET text='changed' WHERE hash=(SELECT content_hash FROM units WHERE id=?1)",
                        "range" => "UPDATE units SET line_end=2 WHERE id=?1",
                        "ladder" => "UPDATE units SET ladder_level=2 WHERE id=?1",
                        _ => unreachable!(),
                    }, [changed]).unwrap();
                }).await.unwrap();
                assert_eq!(context.store.l1_generation().unwrap(), generation);
                assert_eq!(
                    result["index"]["partial"],
                    mutation != "ladder",
                    "{collapsed}/{mutation}: {result}"
                );
                assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
                if mutation != "ladder" {
                    assert_eq!(result["hits"], json!([]));
                    assert_eq!(
                        result["index"]["freshness_refusals"]["selection_changed"],
                        1
                    );
                }
                let retry = context.search(make_request(), true).await.unwrap();
                assert_eq!(retry["index"]["partial"], false);
                assert_eq!(retry["hits"], original["hits"]);
            }
        }
    }

    #[tokio::test]
    async fn search_final_selection_refuses_same_generation_rank_and_exact_changes() {
        for mutation in ["exact", "edge", "rank"] {
            let (_root, context) = fixture();
            for (path, name) in [("src/a.rs", "alpha"), ("src/b.rs", "beta")] {
                add(
                    &context.store,
                    path,
                    name,
                    "unrelated body",
                    Some("should_retry"),
                )
                .unwrap();
                fs::write(context.root.join(path), "unrelated body").unwrap();
                complete_fixture_source(&context, path);
            }
            let make_request = || {
                let mut query = request(Mode::Lexical);
                query.query = "should_retry".into();
                query.k = 1;
                query
            };
            let original = context.search(make_request(), false).await.unwrap();
            assert_eq!(original["hits"].as_array().unwrap().len(), 1);
            assert_eq!(original["index"]["partial"], false);
            let generation = context.store.l1_generation().unwrap();
            let worker = context.clone();
            let started = Instant::now();
            let work = QueryWork::new(started, 5000).unwrap();
            let result =
                context
                    .search_within_checked(make_request(), false, started, &work, move || {
                        let writer = worker.store.writer().unwrap();
                        writer.execute(match mutation {
                    "exact" => "UPDATE symbols SET name='renamed' WHERE name='should_retry'",
                    "edge" => "DELETE FROM edges",
                    "rank" => "UPDATE symbols SET pagerank=10 WHERE name='beta'",
                    _ => unreachable!(),
                }, []).unwrap();
                    })
                    .await
                    .unwrap();
            assert_eq!(context.store.l1_generation().unwrap(), generation);
            assert_eq!(result["hits"], json!([]), "{mutation}: {result}");
            assert_eq!(result["index"]["partial"], true);
            assert_eq!(result["index"]["lexical"], "incomplete");
            assert_eq!(
                result["index"]["freshness_refusals"]["selection_changed"],
                1
            );
            assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
            let retry = context.search(make_request(), false).await.unwrap();
            assert_eq!(retry["index"]["partial"], false);
            assert_eq!(retry["hits"].as_array().unwrap().len(), 1);
            if mutation != "exact" {
                // The bounded public winner is unchanged; hidden graph candidates
                // still participated in the original selection.
                assert_eq!(retry["hits"], original["hits"]);
            }
        }
    }

    #[tokio::test]
    async fn search_final_selection_checks_empty_results_and_repeats_scope_exclusions() {
        for scope in [false, true] {
            for excluded in [false, true] {
                let (_root, context) = fixture();
                let make_request = || {
                    let mut query = request(Mode::Lexical);
                    query.query = "absent_name".into();
                    query.scope = scope.then(|| store::SearchScope::from("src/retry.rs"));
                    query.exclude_ids = if excluded { vec![2] } else { vec![] };
                    query
                };
                let worker = context.clone();
                let generation = context.store.l1_generation().unwrap();
                let started = Instant::now();
                let work = QueryWork::new(started, 5000).unwrap();
                let result = context
                    .search_within_checked(make_request(), true, started, &work, move || {
                        assert_eq!(
                            add(&worker.store, "src/new.rs", "absent_name", "new body", None)
                                .unwrap(),
                            2
                        );
                        fs::write(worker.root.join("src/new.rs"), "new body").unwrap();
                        complete_fixture_source(&worker, "src/new.rs");
                    })
                    .await
                    .unwrap();
                assert_eq!(context.store.l1_generation().unwrap(), generation);
                assert_eq!(result["hits"], json!([]));
                assert_eq!(
                    result["index"]["partial"],
                    !scope && !excluded,
                    "{scope}/{excluded}: {result}"
                );
                assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
                let retry = context.search(make_request(), true).await.unwrap();
                assert_eq!(retry["index"]["partial"], false);
                assert_eq!(
                    retry["hits"].as_array().unwrap().len(),
                    usize::from(!scope && !excluded)
                );
            }
        }
    }

    #[tokio::test]
    async fn search_final_settings_refuses_late_changes_even_when_empty() {
        for empty in [false, true] {
            for mutation in ["root", "name", "limit", "ranking", "invalid"] {
                let (_root, context) = fixture();
                let make_request = || {
                    let mut request = request(Mode::Lexical);
                    if empty {
                        request.query = "absentterm".into();
                    }
                    request
                };
                let original = context.search(make_request(), true).await.unwrap();
                assert_eq!(original["index"]["partial"], false);
                let worker = context.clone();
                let writes = context.store.writer().unwrap().total_changes();
                let started = Instant::now();
                let work = QueryWork::new(started, 5000).unwrap();
                let result = context
                    .search_within_checked(make_request(), true, started, &work, move || {
                        if mutation == "invalid" {
                            fs::write(worker.config_root.join(".deepkit/config.toml"), "[invalid")
                                .unwrap();
                        } else {
                            let mut changed = worker;
                            match mutation {
                                "root" => changed.config.index.root = "src".into(),
                                "name" => changed.config.index.repo_name = "changed".into(),
                                "limit" => changed.config.limits.max_file_bytes = "1 B".into(),
                                "ranking" => changed.config.search.graph_prior_weight = 0.0,
                                _ => unreachable!(),
                            }
                            persist_fixture_config(&changed);
                        }
                    })
                    .await
                    .unwrap();
                assert_eq!(result["hits"], json!([]), "{empty}/{mutation}: {result}");
                assert_eq!(result["index"]["partial"], true);
                assert_eq!(result["index"]["degraded"], true);
                assert_eq!(result["index"]["lexical"], "incomplete");
                assert_eq!(
                    result["index"]["freshness_refusals"]["settings_changed"],
                    usize::from(!empty)
                );
                assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
                assert_eq!(context.store.writer().unwrap().total_changes(), writes);
                persist_fixture_config(&context);
                let retry = context.search(make_request(), true).await.unwrap();
                assert_eq!(retry["hits"], original["hits"]);
                assert_eq!(retry["index"]["partial"], false);
            }
        }
    }

    #[tokio::test]
    async fn search_final_l2_refreshes_publication_state_without_repair() {
        for mutation in ["delete", "supersede", "vector", "advance", "ladder"] {
            let (_root, context) = fixture();
            context
                .store
                .writer()
                .unwrap()
                .execute("UPDATE files SET max_level=3,l1_pending=0", [])
                .unwrap();
            complete_fixture_source(&context, "src/retry.rs");
            if mutation != "advance" {
                publish_fixture_vector(&context, 1);
            }
            let original = context.search(request(Mode::Lexical), true).await.unwrap();
            assert_eq!(
                original["hits"][0]["fresh"]["l2"],
                if mutation == "advance" {
                    "pending"
                } else {
                    "current"
                }
            );
            let worker = context.clone();
            let writes = Arc::new(std::sync::atomic::AtomicU64::new(0));
            let observed = Arc::clone(&writes);
            let started = Instant::now();
            let work = QueryWork::new(started, 5000).unwrap();
            let result = context
                .search_within_checked(request(Mode::Lexical), true, started, &work, move || {
                    match mutation {
                        "delete" => {
                            worker
                                .store
                                .writer()
                                .unwrap()
                                .execute("DELETE FROM embedding_publications", [])
                                .unwrap();
                        }
                        "supersede" => {
                            worker
                                .store
                                .writer()
                                .unwrap()
                                .execute(
                                    "UPDATE embedding_publications SET desired_hash=zeroblob(32)",
                                    [],
                                )
                                .unwrap();
                        }
                        "vector" => {
                            let slug: String = worker
                                .store
                                .writer()
                                .unwrap()
                                .query_row(
                                    "SELECT table_slug FROM embedding_publications",
                                    [],
                                    |row| row.get(0),
                                )
                                .unwrap();
                            worker.store.delete_vector(&slug, 1).unwrap();
                        }
                        "advance" => publish_fixture_vector(&worker, 1),
                        "ladder" => {
                            worker
                                .store
                                .writer()
                                .unwrap()
                                .execute("UPDATE units SET ladder_level=2", [])
                                .unwrap();
                        }
                        _ => unreachable!(),
                    }
                    observed.store(
                        worker.store.writer().unwrap().total_changes(),
                        Ordering::Release,
                    );
                })
                .await
                .unwrap();
            assert_eq!(
                result["hits"].as_array().unwrap().len(),
                1,
                "{mutation}: {result}"
            );
            assert_eq!(result["hits"][0]["id"], original["hits"][0]["id"]);
            assert_eq!(result["hits"][0]["snippet"], original["hits"][0]["snippet"]);
            assert_eq!(
                result["hits"][0]["fresh"]["l2"],
                if matches!(mutation, "advance" | "ladder") {
                    "current"
                } else {
                    "pending"
                },
                "{mutation}: {result}"
            );
            assert_eq!(result["index"]["partial"], false);
            assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
            assert_eq!(
                context.store.writer().unwrap().total_changes(),
                writes.load(Ordering::Acquire)
            );
            publish_fixture_vector(&context, 1);
            assert_fixture_l2(&context, &[("src/retry.rs", "current")]).await;
        }
    }

    #[tokio::test]
    async fn search_final_source_fence_refuses_late_changes_and_preserves_other_files() {
        for change in ["edit", "delete", "replace", "pending", "completion"] {
            let (_root, context) = fixture();
            add(
                &context.store,
                "src/other.rs",
                "other_retry",
                "retry payment",
                None,
            )
            .unwrap();
            fs::write(context.root.join("src/other.rs"), "retry payment").unwrap();
            complete_fixture_source(&context, "src/other.rs");
            let mutation = context.clone();
            let generation = context.store.l1_generation().unwrap();
            let started = Instant::now();
            let work = QueryWork::new(started, 5000).unwrap();
            let response = context.search_within_checked(
                request(Mode::Lexical), true, started, &work, move || {
                    let path = mutation.root.join("src/retry.rs");
                    match change {
                        "edit" => fs::write(&path, "changed source").unwrap(),
                        "delete" => fs::remove_file(&path).unwrap(),
                        "replace" => {
                            fs::write(mutation.root.join("replacement"), "changed source").unwrap();
                            fs::rename(mutation.root.join("replacement"), &path).unwrap();
                        }
                        "pending" => { mutation.store.writer().unwrap().execute(
                            "UPDATE files SET l1_pending=1 WHERE path='src/retry.rs'", []).unwrap(); }
                        "completion" => { mutation.store.writer().unwrap().execute(
                            "UPDATE files SET l1_fingerprint=x'1234',l1_desired=x'1234' WHERE path='src/retry.rs'", []).unwrap(); }
                        _ => unreachable!(),
                    }
                }
            ).await.unwrap();
            assert_eq!(context.store.l1_generation().unwrap(), generation);
            assert_eq!(
                response["hits"].as_array().unwrap().len(),
                1,
                "{change}: {response}"
            );
            assert_eq!(response["hits"][0]["path"], "src/other.rs");
            assert_eq!(response["index"]["partial"], true);
            assert_eq!(response["index"]["degraded"], true);
            assert_eq!(response["index"]["lexical"], "incomplete");
            assert_eq!(response["index"]["suppressed_hits"], 1);
            assert_eq!(response["index"]["lazy_resync"]["attempted_files"], 0);
            fs::write(context.root.join("src/retry.rs"), "retry payment").unwrap();
            complete_fixture_source(&context, "src/retry.rs");
            let retry = context.search(request(Mode::Lexical), true).await.unwrap();
            assert_eq!(retry["hits"].as_array().unwrap().len(), 2);
            assert_eq!(retry["index"]["partial"], false);
        }
    }

    #[tokio::test]
    async fn search_final_source_fence_keeps_original_deadline_cancellation_and_retry() {
        for cancel in [false, true] {
            let (_root, context) = fixture();
            let started = Instant::now();
            let work = QueryWork::new(started, 1000).unwrap();
            let cancelled = Arc::clone(&work.cancelled);
            let entered = Arc::new(AtomicBool::new(false));
            let observed = Arc::clone(&entered);
            let deadline = work.deadline;
            let error = context
                .search_within_checked(request(Mode::Lexical), true, started, &work, move || {
                    observed.store(true, Ordering::Release);
                    if cancel {
                        cancelled.store(true, Ordering::Release);
                    } else {
                        std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
                    }
                })
                .await
                .unwrap_err();
            assert!(entered.load(Ordering::Acquire));
            assert_eq!(error.code(), "timeout");
            let retry = context.search(request(Mode::Lexical), true).await.unwrap();
            assert_eq!(retry["hits"].as_array().unwrap().len(), 1);
            assert_eq!(retry["index"]["partial"], false);
        }
    }

    #[tokio::test]
    async fn search_pending_catalogue_marks_empty_and_verified_results_incomplete() {
        for lazy in [false, true] {
            for late in [false, true] {
                for has_hits in [false, true] {
                    let (root, mut context) = fixture();
                    context.config.search.lazy_resync = lazy;
                    persist_fixture_config(&context);
                    // A catalogued empty file has no occurrence to select or repair.
                    context.store.writer().unwrap().execute(
                        "INSERT INTO files(path,blake3,size,mtime_ns,lang,kind,max_level,treatment)
                         VALUES('src/pending.rs',zeroblob(32),0,1,'rust','code',1,'source')",
                        [],
                    ).unwrap();
                    fs::write(root.path().join("src/pending.rs"), "").unwrap();
                    complete_fixture_source(&context, "src/pending.rs");
                    let make_request = || {
                        let mut request = request(Mode::Lexical);
                        if !has_hits {
                            request.query = "absentterm".into();
                        }
                        request
                    };
                    let initial = context.search(make_request(), true).await.unwrap();
                    assert_eq!(initial["index"]["partial"], false);
                    let generation = context.store.l1_generation().unwrap();
                    let store = Arc::clone(&context.store);
                    let mark_pending = move || {
                        store
                            .writer()
                            .unwrap()
                            .execute(
                                "UPDATE files SET l1_pending=1 WHERE path='src/pending.rs'",
                                [],
                            )
                            .unwrap();
                    };
                    if !late {
                        mark_pending();
                    }
                    let writes = context.store.writer().unwrap().total_changes();
                    let started = Instant::now();
                    let work = QueryWork::new(started, 5000).unwrap();
                    let response = context
                        .search_within_checked(make_request(), true, started, &work, move || {
                            if late {
                                mark_pending();
                            }
                        })
                        .await
                        .unwrap();
                    assert_eq!(response["hits"], initial["hits"]);
                    assert_eq!(
                        response["index"]["partial"], true,
                        "{lazy}/{late}/{has_hits}: {response}"
                    );
                    assert_eq!(response["index"]["degraded"], true);
                    assert_eq!(response["index"]["lexical"], "incomplete");
                    assert_eq!(response["index"]["fresh"]["reason"], "l1_pending");
                    assert_eq!(response["index"]["l1_pending_files"], 1);
                    assert_eq!(response["index"]["suppressed_hits"], 0);
                    assert_eq!(response["index"]["freshness_refusals"], json!({}));
                    assert_eq!(response["index"]["lazy_resync"]["attempted_files"], 0);
                    assert_eq!(context.store.l1_generation().unwrap(), generation);
                    assert_eq!(
                        context.store.writer().unwrap().total_changes(),
                        writes + u64::from(late)
                    );
                    store_pending_clear(&context);
                    let retry = context.search(make_request(), true).await.unwrap();
                    assert_eq!(retry["hits"], initial["hits"]);
                    assert_eq!(retry["index"]["partial"], false);
                    assert_eq!(retry["index"]["lexical"], "complete");
                }
            }
        }
    }

    #[tokio::test]
    async fn search_generation_change_refuses_verified_and_empty_results() {
        for has_hits in [true, false] {
            let (_root, context) = fixture();
            let make_request = || {
                let mut request = request(Mode::Lexical);
                if !has_hits {
                    request.query = "absentterm".into();
                }
                request
            };
            let started = Instant::now();
            let work = QueryWork::new(started, 5000).unwrap();
            let store = Arc::clone(&context.store);
            let before = store.l1_generation().unwrap();
            let response = context
                .search_within_checked(make_request(), true, started, &work, move || {
                    store
                        .set_meta("l1_scan_generation", &(before + 1).to_string())
                        .unwrap();
                })
                .await
                .unwrap();
            assert_eq!(response["hits"], json!([]));
            assert_eq!(response["index"]["partial"], true);
            assert_eq!(response["index"]["degraded"], true);
            assert_eq!(response["index"]["lexical"], "incomplete");
            assert_eq!(response["index"]["fresh"]["state"], "stale");
            assert_eq!(response["index"]["suppressed_hits"], usize::from(has_hits));
            assert_eq!(
                response["index"]["freshness_refusals"]["index_generation_changed"],
                usize::from(has_hits)
            );
            let retry = context.search(make_request(), true).await.unwrap();
            assert_eq!(retry["index"]["partial"], false);
            assert_eq!(retry["index"]["lexical"], "complete");
            assert_eq!(
                retry["hits"].as_array().unwrap().len(),
                usize::from(has_hits)
            );
        }
    }

    #[tokio::test]
    async fn search_retains_occurrences_and_refuses_same_generation_replacement() {
        for sql in [
            "UPDATE units SET header='replacement'",
            "UPDATE units SET line_end=line_end+1",
            "DELETE FROM units",
        ] {
            let (_root, context) = fixture();
            let store = Arc::clone(&context.store);
            let before = store.l1_generation().unwrap();
            let started = Instant::now();
            let work = QueryWork::new(started, 5000).unwrap();
            let response = context
                .search_within_checked(request(Mode::Lexical), true, started, &work, move || {
                    store.writer().unwrap().execute(sql, []).unwrap();
                })
                .await
                .unwrap();
            assert_eq!(context.store.l1_generation().unwrap(), before);
            assert_eq!(response["hits"], json!([]), "{sql}");
            assert_eq!(response["index"]["partial"], true);
            assert_eq!(response["index"]["degraded"], true);
            assert_eq!(response["index"]["lexical"], "incomplete");
            assert_eq!(response["index"]["suppressed_hits"], 1);
            assert_eq!(
                response["index"]["freshness_refusals"]["occurrence_changed"],
                1
            );
        }
    }

    #[tokio::test]
    async fn search_occurrence_fence_allows_ladder_advancement_and_retry() {
        let (_root, context) = fixture();
        let store = Arc::clone(&context.store);
        let started = Instant::now();
        let work = QueryWork::new(started, 5000).unwrap();
        let response = context
            .search_within_checked(request(Mode::Lexical), true, started, &work, move || {
                store
                    .writer()
                    .unwrap()
                    .execute("UPDATE units SET ladder_level=2", [])
                    .unwrap();
            })
            .await
            .unwrap();
        assert_eq!(response["hits"].as_array().unwrap().len(), 1);
        assert_eq!(response["index"]["partial"], false);
        let retry = context.search(request(Mode::Lexical), true).await.unwrap();
        assert_eq!(response["hits"][0]["id"], retry["hits"][0]["id"]);
        assert_eq!(retry["index"]["partial"], false);
    }

    #[test]
    fn refs_verification_reader_respects_deadline_cancellation_and_retry() {
        use std::{
            sync::atomic::{AtomicBool, Ordering},
            time::Duration,
        };
        let (_root, context) = fixture();
        let symbols = navigation::symbol(&context.store, "should_retry").unwrap();
        let group = navigation::ReferenceGroup {
            path: symbols[0].path.clone(),
            lines: [symbols[0].lines[0]].into(),
            kind: "ref",
        };
        let memory = store::open(Path::new(":memory:"), StoreLimits::default()).unwrap();
        for cancel in [false, true] {
            let held = memory.writer().unwrap();
            let started = Instant::now();
            let cancelled = AtomicBool::new(false);
            std::thread::scope(|scope| {
                if cancel {
                    scope.spawn(|| {
                        std::thread::sleep(Duration::from_millis(15));
                        cancelled.store(true, Ordering::Release);
                    });
                }
                let work = sync::l1::Work {
                    deadline: (!cancel).then_some(started + Duration::from_millis(15)),
                    cancelled: Some(&cancelled),
                };
                assert!(
                    navigation::verify_reference_group(
                        &memory,
                        &context.root,
                        &context.config,
                        &group,
                        1024,
                        work
                    )
                    .is_err()
                );
            });
            assert!(started.elapsed() < Duration::from_secs(1));
            drop(held);
            assert_eq!(
                navigation::verify_reference_group(
                    &context.store,
                    &context.root,
                    &context.config,
                    &group,
                    1024,
                    Default::default()
                )
                .unwrap(),
                Ok(())
            );
            assert_eq!(
                navigation::verify_reference_group(
                    &memory,
                    &context.root,
                    &context.config,
                    &group,
                    1024,
                    Default::default()
                )
                .unwrap(),
                Err("l1_pending_or_unavailable")
            );
        }
    }

    #[test]
    fn map_verification_reader_respects_deadline_and_retry() {
        use std::time::Duration;
        let (_root, context) = fixture();
        let symbols = context.store.tool_symbols().unwrap();
        let memory = store::open(Path::new(":memory:"), StoreLimits::default()).unwrap();
        let held = memory.writer().unwrap();
        let started = Instant::now();
        assert!(
            navigation::verify_map_file(
                &memory,
                &context.root,
                &context.config,
                "src/retry.rs",
                &symbols,
                1024,
                sync::l1::Work {
                    deadline: Some(started + Duration::from_millis(15)),
                    cancelled: None
                }
            )
            .is_err()
        );
        assert!(started.elapsed() < Duration::from_secs(1));
        drop(held);
        assert_eq!(
            navigation::verify_map_file(
                &context.store,
                &context.root,
                &context.config,
                "src/retry.rs",
                &symbols,
                1024,
                Default::default()
            )
            .unwrap(),
            Ok(())
        );
    }

    #[test]
    fn symbol_verification_reader_respects_deadline_and_retry() {
        use std::time::Duration;
        let (_root, context) = fixture();
        let mut symbols = navigation::symbol(&context.store, "should_retry").unwrap();
        let memory = store::open(Path::new(":memory:"), StoreLimits::default()).unwrap();
        let held = memory.writer().unwrap();
        let started = Instant::now();
        let work = sync::l1::Work {
            deadline: Some(started + Duration::from_millis(15)),
            cancelled: None,
        };
        assert!(
            navigation::verify_symbol(
                &memory,
                &context.root,
                &context.config,
                &mut symbols[0],
                1024,
                work
            )
            .is_err()
        );
        assert!(started.elapsed() < Duration::from_secs(1));
        drop(held);
        navigation::verify_symbol(
            &context.store,
            &context.root,
            &context.config,
            &mut symbols[0],
            1024,
            Default::default(),
        )
        .unwrap();
        let hit = serde_json::json!({"id": "u_missing", "path": "src/retry.rs"});
        let held = memory.writer().unwrap();
        let started = Instant::now();
        assert!(
            navigation::verify_hit(
                &memory,
                &context.root,
                &hit,
                1024,
                sync::l1::Work {
                    deadline: Some(started + Duration::from_millis(15)),
                    cancelled: None,
                }
            )
            .is_err()
        );
        assert!(started.elapsed() < Duration::from_secs(1));
        drop(held);
    }

    #[test]
    fn search_occurrence_reader_respects_deadline_cancellation_and_retry() {
        use std::time::Duration;
        let (_root, context) = fixture();
        let store = &context.store;
        let parsed = super::super::query::parse("retry payment", Mode::Lexical).unwrap();
        let occurrences = super::super::lexical_with_exclusions(
            store,
            &parsed,
            10,
            None,
            &context.config.search,
            &[],
        )
        .unwrap()
        .into_iter()
        .map(|hit| hit.unit)
        .collect::<Vec<_>>();
        assert_eq!(occurrences.len(), 1);
        let memory = store::open(Path::new(":memory:"), StoreLimits::default()).unwrap();
        let held = memory.writer().unwrap();
        let started = Instant::now();
        assert!(matches!(
            query_occurrences(
                &memory,
                &occurrences,
                sync::l1::Work {
                    deadline: Some(started + Duration::from_millis(15)),
                    cancelled: None,
                }
            ),
            Err(Error::Search(super::super::Error::Deadline))
        ));
        assert!(started.elapsed() < Duration::from_secs(1));
        drop(held);
        let cancelled = AtomicBool::new(true);
        assert!(matches!(
            query_occurrences(
                store,
                &occurrences,
                sync::l1::Work {
                    deadline: None,
                    cancelled: Some(&cancelled),
                }
            ),
            Err(Error::Search(super::super::Error::Deadline))
        ));
        assert_eq!(
            query_occurrences(store, &occurrences, Default::default()).unwrap(),
            vec![true]
        );
    }

    #[test]
    fn search_generation_observation_respects_contention_cancellation_and_retry() {
        use std::time::Duration;
        let store = store::open(Path::new(":memory:"), StoreLimits::default()).unwrap();
        let held = store.writer().unwrap();
        let started = Instant::now();
        let work = sync::l1::Work {
            deadline: Some(started + Duration::from_millis(15)),
            cancelled: None,
        };
        assert!(matches!(
            query_generation(&store, work),
            Err(Error::Search(super::super::Error::Deadline))
        ));
        assert!(started.elapsed() < Duration::from_secs(1));
        drop(held);
        let cancelled = AtomicBool::new(true);
        assert!(matches!(
            query_generation(
                &store,
                sync::l1::Work {
                    deadline: None,
                    cancelled: Some(&cancelled)
                }
            ),
            Err(Error::Search(super::super::Error::Deadline))
        ));
        assert_eq!(
            query_generation(&store, Default::default()).unwrap(),
            store.l1_generation().unwrap()
        );
    }

    #[tokio::test]
    async fn search_reports_verified_source_age_and_suppresses_changed_bytes() {
        use std::time::{Duration, SystemTime};
        let (root, mut context) = fixture();
        context.config.search.lazy_resync = false;
        persist_fixture_config(&context);
        let path = root.path().join("src/retry.rs");
        let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_modified(SystemTime::now() - Duration::from_secs(90))
            .unwrap();
        let response = context.search(request(Mode::Lexical), false).await.unwrap();
        let fresh = &response["hits"][0]["fresh"];
        assert_eq!(fresh["state"], "fresh");
        assert_eq!(fresh["l1"], "current");
        assert!((90..120).contains(&fresh["changed_ago_s"].as_u64().unwrap()));
        file.set_modified(SystemTime::now() + Duration::from_secs(3600))
            .unwrap();
        let future = context.search(request(Mode::Lexical), false).await.unwrap();
        assert_eq!(future["hits"][0]["fresh"]["changed_ago_s"], 0);
        // Equal mtime never substitutes for verifying the returned source bytes.
        let modified = file.metadata().unwrap().modified().unwrap();
        fs::write(&path, "other payment").unwrap();
        file.set_modified(modified).unwrap();
        let changed = context.search(request(Mode::Lexical), false).await.unwrap();
        assert_eq!(changed["hits"], json!([]));
        assert_eq!(changed["index"]["freshness_refusals"]["source_changed"], 1);
    }

    #[tokio::test]
    async fn timeout_preserves_explicit_rerank_choice_without_dispatch() {
        let (_root, mut context) = fixture();
        context.config.limits.query_timeout_ms = 0;
        for enabled in [false, true] {
            let response = context
                .search_with_rerank(request(Mode::Hybrid), false, enabled)
                .await
                .unwrap();
            assert_eq!(
                response["index"]["rerank"],
                if enabled { "timeout" } else { "off" }
            );
            assert_eq!(response["index"]["reranked"], false);
            assert_eq!(response["index"]["reason"], "query_timeout");
        }
    }

    #[tokio::test]
    async fn public_search_bounds_precede_deadlines_in_every_mode() {
        let (_root, mut context) = fixture();
        for configured in [1, 50, 100] {
            context.config.search.max_k = configured;
            let maximum = configured.min(50);
            for timeout in [0, 5000] {
                context.config.limits.query_timeout_ms = timeout;
                for mode in [Mode::Lexical, Mode::Dense, Mode::Hybrid] {
                    for k in [0, maximum as usize + 1, 101] {
                        let mut input = request(mode);
                        input.k = k;
                        assert!(matches!(
                            context.search(input, false).await,
                            Err(Error::Search(super::super::Error::InvalidK { max }))
                                if max == maximum
                        ));
                    }
                }
            }
        }
        // The public guard does not narrow internal diagnostic retrieval.
        let parsed = super::super::query::parse("retry payment", Mode::Lexical).unwrap();
        assert!(
            super::super::lexical_with_selection(
                &context.store,
                &parsed,
                100,
                None,
                &context.config.search,
                &[],
                &|| Ok(()),
            )
            .is_ok()
        );
    }

    #[tokio::test]
    async fn shared_search_preserves_lexical_evidence_and_reports_unavailable_dense() {
        let (_root, context) = fixture();
        let lexical = context.search(request(Mode::Lexical), true).await.unwrap();
        let hybrid = context.search(request(Mode::Hybrid), true).await.unwrap();
        let without_age = |value: &Value| {
            let mut hits = value["hits"].clone();
            for hit in hits.as_array_mut().unwrap() {
                hit["fresh"]
                    .as_object_mut()
                    .unwrap()
                    .remove("changed_ago_s");
            }
            hits
        };
        assert_eq!(without_age(&lexical), without_age(&hybrid));
        assert_eq!(lexical["index"]["degraded"], false);
        assert_eq!(lexical["index"]["dense"], "not_requested");
        assert_eq!(hybrid["index"]["dense"], "unavailable");
        assert_eq!(hybrid["index"]["degraded"], true);
        assert_eq!(lexical["hits"][0]["path"], "src/retry.rs");
        assert_eq!(lexical["hits"][0]["fresh"]["state"], "fresh");
        assert_eq!(context.status().unwrap()["queue"]["queued"], 0);
    }

    #[tokio::test]
    async fn lexical_fallback_preserves_legacy_admission_typed_routing_and_exclusions() {
        let (_root, mut context) = fixture();
        add(
            &context.store,
            "docs/guide.md",
            "payment_help",
            "retry payment",
            None,
        )
        .unwrap();
        fs::create_dir(_root.path().join("docs")).unwrap();
        fs::write(_root.path().join("docs/guide.md"), "retry payment").unwrap();
        context.config.rerank.candidates = 1;
        persist_fixture_config(&context);
        complete_fixture_source(&context, "src/retry.rs");
        complete_fixture_source(&context, "docs/guide.md");
        let expected = crate::search::lexical(
            &context.store,
            "retry payment",
            10,
            None,
            &context.config.search,
        )
        .unwrap();
        assert_eq!(expected.len(), 2);
        let result = context.search(request(Mode::Hybrid), false).await.unwrap();
        let ids = result["hits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|hit| hit["id"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            expected
                .iter()
                .map(|hit| navigation::public_id(&hit.unit))
                .collect::<Vec<_>>(),
            "{result}"
        );
        let mut excluded = request(Mode::Lexical);
        excluded.exclude_ids.push(expected[0].unit.id);
        let result = context.search(excluded, false).await.unwrap();
        assert_eq!(result["hits"].as_array().unwrap().len(), 1);
        assert_eq!(
            result["hits"][0]["id"],
            navigation::public_id(&expected[1].unit)
        );
        let mut scoped = request(Mode::Hybrid);
        scoped.scope = Some("docs/*".into());
        let result = context.search(scoped, false).await.unwrap();
        assert_eq!(result["hits"].as_array().unwrap().len(), 1);
        assert_eq!(result["hits"][0]["path"], "docs/guide.md");
        let mut dense_only = request(Mode::Lexical);
        dense_only.query = "vec:retry payment".into();
        let result = context.search(dense_only, false).await.unwrap();
        assert_eq!(result["hits"], json!([]));
        assert_eq!(result["index"]["lexical"], "not_requested");
        assert_eq!(result["index"]["dense"], "unavailable");
        let mut mixed = request(Mode::Dense);
        mixed.query = "lex:retry payment vec:unrelated".into();
        let result = context.search(mixed, false).await.unwrap();
        assert_eq!(result["hits"].as_array().unwrap().len(), 2);
        let mut phrase = request(Mode::Hybrid);
        phrase.query = "lex:\"retry absent\" vec:payment".into();
        assert_eq!(
            context.search(phrase, false).await.unwrap()["hits"],
            json!([])
        );
    }

    #[tokio::test]
    async fn search_suppresses_changed_missing_pending_and_out_of_range_sources() {
        let (root, mut context) = fixture();
        context.config.search.lazy_resync = false;
        persist_fixture_config(&context);
        let run = |context: &Context| {
            let context = context.clone();
            async move { context.search(request(Mode::Lexical), false).await.unwrap() }
        };
        let good = run(&context).await;
        assert_eq!(good["hits"].as_array().unwrap().len(), 1);
        assert_eq!(good["index"]["partial"], false);
        for contents in [Some("different text"), None, Some("retry payment")] {
            let path = root.path().join("src/retry.rs");
            match contents {
                Some(text) => fs::write(&path, text).unwrap(),
                None => fs::remove_file(&path).unwrap(),
            }
            if contents == Some("retry payment") {
                context
                    .store
                    .writer()
                    .unwrap()
                    .execute("UPDATE files SET l1_pending=1", [])
                    .unwrap();
            }
            let result = run(&context).await;
            assert_eq!(result["hits"], json!([]));
            assert_eq!(result["index"]["partial"], true);
            assert_eq!(result["index"]["suppressed_hits"], 1);
            assert_eq!(result["index"]["fresh"]["state"], "stale");
        }
        context
            .store
            .writer()
            .unwrap()
            .execute("UPDATE files SET l1_pending=0", [])
            .unwrap();
        assert_eq!(run(&context).await["hits"].as_array().unwrap().len(), 1);
        let hit = &good["hits"][0];
        assert_eq!(
            navigation::verify_hit(
                &context.store,
                &context.root,
                hit,
                100,
                sync::l1::Work {
                    deadline: Some(Instant::now()),
                    ..Default::default()
                }
            )
            .unwrap(),
            Err("timeout")
        );
        let cancelled = AtomicBool::new(true);
        assert_eq!(
            navigation::verify_hit(
                &context.store,
                &context.root,
                hit,
                100,
                sync::l1::Work {
                    cancelled: Some(&cancelled),
                    ..Default::default()
                },
            )
            .unwrap(),
            Err("timeout")
        );
        let mut invalid = hit.clone();
        invalid["lines"] = json!([1, 2]);
        assert_eq!(
            navigation::verify_hit(
                &context.store,
                &context.root,
                &invalid,
                100,
                sync::l1::Work {
                    deadline: Some(Instant::now() + std::time::Duration::from_secs(1)),
                    ..Default::default()
                }
            )
            .unwrap(),
            Err("invalid_source_range")
        );
        context.config.limits.max_file_bytes = "1 B".into();
        let capped = run(&context).await;
        assert_eq!(capped["hits"], json!([]));
        assert_eq!(capped["index"]["freshness_refusals"]["source_bytes"], 1);
    }

    #[tokio::test]
    async fn references_suppress_changed_missing_pending_and_invalid_ranges_then_recover() {
        let (root, mut context) = fixture();
        context.config.search.lazy_resync = false;
        persist_fixture_config(&context);
        context.store.writer().unwrap().execute("INSERT INTO edges(src_symbol,dst_name,kind,file_id,line) SELECT id,'payment','ref',file_id,1 FROM symbols", []).unwrap();
        let path = root.path().join("src/retry.rs");
        let original = fs::read(&path).unwrap();
        let initial = context.refs("payment", None, false).await.unwrap();
        assert_eq!(initial["total"], 1);
        assert_eq!(initial["returned_total"], 1);
        assert_eq!(initial["index"]["partial"], false);
        for mutation in [
            "edit",
            "delete",
            "rename",
            "pending",
            "range",
            "fingerprint",
        ] {
            match mutation {
                "edit" => fs::write(&path, "changed").unwrap(),
                "delete" => fs::remove_file(&path).unwrap(),
                "rename" => fs::rename(&path, root.path().join("moved.rs")).unwrap(),
                "pending" => {
                    context
                        .store
                        .writer()
                        .unwrap()
                        .execute("UPDATE files SET l1_pending=1", [])
                        .unwrap();
                }
                "range" => {
                    context
                        .store
                        .writer()
                        .unwrap()
                        .execute("UPDATE edges SET line=100", [])
                        .unwrap();
                }
                "fingerprint" => {
                    context
                        .store
                        .writer()
                        .unwrap()
                        .execute("UPDATE files SET l1_fingerprint=X'02'", [])
                        .unwrap();
                }
                _ => unreachable!(),
            }
            let result = context.refs("payment", None, false).await.unwrap();
            assert_eq!(result["groups"], json!([]), "{mutation}: {result}");
            assert_eq!(result["index"]["partial"], true);
            assert_eq!(result["index"]["suppressed_references"], 1);
            assert_eq!(result["indexed_total"], 1);
            assert!(result["total"].is_null());
            if mutation == "rename" {
                fs::remove_file(root.path().join("moved.rs")).unwrap();
            }
            fs::write(&path, &original).unwrap();
            context
                .store
                .writer()
                .unwrap()
                .execute(
                    "UPDATE files SET l1_pending=0, l1_fingerprint=l1_desired",
                    [],
                )
                .unwrap();
            context
                .store
                .writer()
                .unwrap()
                .execute("UPDATE edges SET line=1", [])
                .unwrap();
            assert_eq!(
                context.refs("payment", None, false).await.unwrap()["groups"],
                initial["groups"]
            );
        }
        context.config.limits.max_file_bytes = "1 B".into();
        persist_fixture_config(&context);
        let capped = context.refs("payment", None, false).await.unwrap();
        assert_eq!(capped["index"]["freshness_refusals"]["source_bytes"], 1);
        context.config.limits.query_timeout_ms = 0;
        let expired = context.refs("payment", None, false).await.unwrap();
        assert_eq!(expired["index"]["reason"], "query_timeout");
        assert!(expired["total"].is_null());
        assert!(expired["index"]["suppressed_references"].is_null());
        assert!(context.refs("payment", Some(0), false).await.is_err());
    }

    #[tokio::test]
    async fn symbols_refuse_changed_missing_pending_and_invalid_ranges_then_recover() {
        let (root, mut context) = fixture();
        context.config.search.lazy_resync = false;
        persist_fixture_config(&context);
        let path = root.path().join("src/retry.rs");
        let original = fs::read(&path).unwrap();
        let initial = context.symbol("should_retry").await.unwrap();
        assert_eq!(initial.as_array().unwrap().len(), 1);
        for mutation in [
            "edit",
            "delete",
            "rename",
            "pending",
            "range",
            "fingerprint",
        ] {
            match mutation {
                "edit" => fs::write(&path, "changed").unwrap(),
                "delete" => fs::remove_file(&path).unwrap(),
                "rename" => fs::rename(&path, root.path().join("moved.rs")).unwrap(),
                "pending" => {
                    context
                        .store
                        .writer()
                        .unwrap()
                        .execute("UPDATE files SET l1_pending=1", [])
                        .unwrap();
                }
                "range" => {
                    context
                        .store
                        .writer()
                        .unwrap()
                        .execute("UPDATE symbols SET line_end=100", [])
                        .unwrap();
                }
                "fingerprint" => {
                    context
                        .store
                        .writer()
                        .unwrap()
                        .execute("UPDATE files SET l1_fingerprint=X'02'", [])
                        .unwrap();
                }
                _ => unreachable!(),
            }
            assert_eq!(
                context.symbol("should_retry").await.unwrap_err().code(),
                "source_unavailable",
                "{mutation}"
            );
            if mutation == "rename" {
                fs::remove_file(root.path().join("moved.rs")).unwrap();
            }
            fs::write(&path, &original).unwrap();
            context
                .store
                .writer()
                .unwrap()
                .execute(
                    "UPDATE files SET l1_pending=0, l1_fingerprint=l1_desired",
                    [],
                )
                .unwrap();
            context
                .store
                .writer()
                .unwrap()
                .execute("UPDATE symbols SET line_end=1", [])
                .unwrap();
            assert_eq!(context.symbol("should_retry").await.unwrap(), initial);
        }
        context.config.limits.max_file_bytes = "1 B".into();
        persist_fixture_config(&context);
        assert_eq!(
            context.symbol("should_retry").await.unwrap_err().code(),
            "source_unavailable"
        );
        context.config.limits.query_timeout_ms = 0;
        assert_eq!(
            context.symbol("should_retry").await.unwrap_err().code(),
            "timeout"
        );
    }

    #[tokio::test]
    async fn maps_omit_invalid_files_from_tree_and_nodes_and_recover() {
        let (root, mut context) = fixture();
        context.config.search.lazy_resync = false;
        persist_fixture_config(&context);
        let path = root.path().join("src/retry.rs");
        let original = fs::read(&path).unwrap();
        let initial = context.map(2000, &[]).await.unwrap();
        assert_eq!(initial["nodes"].as_array().unwrap().len(), 1);
        for mutation in [
            "edit",
            "delete",
            "rename",
            "pending",
            "range",
            "fingerprint",
        ] {
            match mutation {
                "edit" => fs::write(&path, "changed").unwrap(),
                "delete" => fs::remove_file(&path).unwrap(),
                "rename" => fs::rename(&path, root.path().join("moved.rs")).unwrap(),
                "pending" => {
                    context
                        .store
                        .writer()
                        .unwrap()
                        .execute("UPDATE files SET l1_pending=1", [])
                        .unwrap();
                }
                "range" => {
                    context
                        .store
                        .writer()
                        .unwrap()
                        .execute("UPDATE symbols SET line_end=100", [])
                        .unwrap();
                }
                "fingerprint" => {
                    context
                        .store
                        .writer()
                        .unwrap()
                        .execute("UPDATE files SET l1_fingerprint=X'02'", [])
                        .unwrap();
                }
                _ => unreachable!(),
            }
            let mapped = context.map(2000, &[]).await.unwrap();
            assert_eq!(mapped["nodes"], json!([]), "{mutation}");
            assert_eq!(mapped["tree"], "");
            assert_eq!(mapped["index"]["partial"], true);
            assert_eq!(mapped["index"]["suppressed_files"], 1);
            if mutation == "rename" {
                fs::remove_file(root.path().join("moved.rs")).unwrap();
            }
            fs::write(&path, &original).unwrap();
            context
                .store
                .writer()
                .unwrap()
                .execute(
                    "UPDATE files SET l1_pending=0, l1_fingerprint=l1_desired",
                    [],
                )
                .unwrap();
            context
                .store
                .writer()
                .unwrap()
                .execute("UPDATE symbols SET line_end=1", [])
                .unwrap();
            let restored = context.map(2000, &[]).await.unwrap();
            assert_eq!(restored["nodes"], initial["nodes"]);
            assert_eq!(restored["tree"], initial["tree"]);
            assert_eq!(restored["index"]["partial"], false);
        }
        context.config.limits.max_file_bytes = "1 B".into();
        persist_fixture_config(&context);
        assert_eq!(
            context.map(2000, &[]).await.unwrap()["index"]["freshness_refusals"]["source_bytes"],
            1
        );
        context.config.limits.query_timeout_ms = 0;
        let timed = context.map(2000, &[]).await.unwrap();
        assert_eq!(timed["index"]["reason"], "query_timeout");
        assert_eq!(timed["nodes"], json!([]));
        assert!(timed["index"]["suppressed_files"].is_null());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn maps_without_signatures_verify_paths_without_claiming_content_freshness() {
        let (root, context) = fixture();
        context
            .store
            .writer()
            .unwrap()
            .execute("DELETE FROM symbols", [])
            .unwrap();
        let path = root.path().join("src/retry.rs");
        fs::write(&path, "").unwrap();
        let mapped = context.map(2000, &[]).await.unwrap();
        assert_eq!(mapped["nodes"][0]["path"], "src/retry.rs");
        assert_eq!(mapped["nodes"][0]["symbols"], json!([]));
        assert_ne!(mapped["index"]["fresh"]["state"], "fresh");
        for kind in ["missing", "directory", "symlink", "fifo"] {
            fs::remove_file(&path).unwrap();
            match kind {
                "missing" => {}
                "directory" => fs::create_dir(&path).unwrap(),
                "symlink" => std::os::unix::fs::symlink("/dev/null", &path).unwrap(),
                "fifo" => assert!(
                    Command::new("mkfifo")
                        .arg(&path)
                        .status()
                        .unwrap()
                        .success()
                ),
                _ => unreachable!(),
            }
            let refused = context.map(2000, &[]).await.unwrap();
            assert_eq!(refused["nodes"], json!([]), "{kind}");
            assert_eq!(refused["tree"], "");
            assert_eq!(refused["index"]["partial"], true);
            match kind {
                "directory" => fs::remove_dir(&path).unwrap(),
                "missing" => {}
                _ => fs::remove_file(&path).unwrap(),
            }
            fs::write(&path, "").unwrap();
        }
        assert_eq!(
            context.map(2000, &[]).await.unwrap()["nodes"],
            mapped["nodes"]
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn references_refuse_identical_external_symlinks_and_special_files() {
        let (root, context) = fixture();
        context.store.writer().unwrap().execute("INSERT INTO edges(src_symbol,dst_name,kind,file_id,line) SELECT id,'payment','ref',file_id,1 FROM symbols", []).unwrap();
        let path = root.path().join("src/retry.rs");
        let outside = tempfile::NamedTempFile::new().unwrap();
        fs::copy(&path, outside.path()).unwrap();
        fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink(outside.path(), &path).unwrap();
        let mapped = context.map(2000, &[]).await.unwrap();
        assert_eq!(mapped["nodes"], json!([]));
        assert_eq!(mapped["index"]["partial"], true);
        let redirected = context.refs("payment", None, false).await.unwrap();
        assert_eq!(redirected["groups"], json!([]));
        assert_eq!(
            redirected["index"]["freshness_refusals"]["source_unavailable"],
            1
        );
        assert_eq!(
            context.symbol("should_retry").await.unwrap_err().code(),
            "source_unavailable"
        );
        fs::remove_file(&path).unwrap();
        assert!(
            Command::new("mkfifo")
                .arg(&path)
                .status()
                .unwrap()
                .success()
        );
        assert_eq!(
            context.symbol("should_retry").await.unwrap_err().code(),
            "source_unavailable"
        );
        let fifo = context.refs("payment", None, false).await.unwrap();
        assert_eq!(fifo["groups"], json!([]));
        assert_eq!(fifo["index"]["freshness_refusals"]["source_unavailable"], 1);
    }

    #[tokio::test]
    async fn query_work_preserves_deadline_results_and_cancels_on_scope_exit() {
        let started = Instant::now();
        let owner = QueryWork::new(started, 2_000).unwrap();
        let flag = Arc::clone(&owner.cancelled);
        let deadline = owner.deadline;
        assert_eq!(deadline, started + std::time::Duration::from_secs(2));
        assert_eq!(
            owner
                .blocking(move |work| {
                    assert_eq!(work.deadline, Some(deadline));
                    assert!(work.cancelled.is_some());
                    Ok(42)
                })
                .await
                .unwrap(),
            42
        );
        assert!(matches!(
            owner.blocking(|_| Err::<(), _>(Error::Clock)).await,
            Err(Error::Clock)
        ));
        assert!(!flag.load(Ordering::Acquire));
        drop(owner);
        assert!(flag.load(Ordering::Acquire));

        let expired = QueryWork::new(Instant::now(), 0).unwrap();
        let invoked = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&invoked);
        assert!(matches!(
            expired
                .blocking(move |_| {
                    observed.store(true, Ordering::Release);
                    Ok(())
                })
                .await,
            Err(Error::Search(super::super::Error::Deadline))
        ));
        assert!(!invoked.load(Ordering::Acquire));
    }

    // Gate an already-running blocking stage so cancellation is observed before
    // it can perform its next write, independent of thread scheduling delays.
    async fn cancelled_query_stage(timeout: bool) {
        let store = Arc::new(store::open(":memory:", StoreLimits::default()).unwrap());
        let worker_store = Arc::clone(&store);
        let (entered, ready) = tokio::sync::oneshot::channel();
        let (release, released) = std::sync::mpsc::channel();
        let (finished, done) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            let owner = QueryWork::new(Instant::now(), 10_000).unwrap();
            let flag = Arc::clone(&owner.cancelled);
            let stage = owner.blocking(move |work| {
                entered.send(flag).unwrap();
                released
                    .recv_timeout(std::time::Duration::from_secs(2))
                    .unwrap();
                let allowed = check_query_work(work);
                if allowed.is_ok() {
                    worker_store
                        .set_meta("cancelled_query_write", "unexpected")
                        .unwrap();
                }
                finished.send(allowed.is_err()).unwrap();
                allowed
            });
            if timeout {
                assert!(
                    tokio::time::timeout(std::time::Duration::from_millis(50), stage)
                        .await
                        .is_err()
                );
            } else {
                stage.await.unwrap();
            }
        });
        let flag = ready.await.unwrap();
        if timeout {
            task.await.unwrap();
        } else {
            task.abort();
            assert!(task.await.unwrap_err().is_cancelled());
        }
        assert!(flag.load(Ordering::Acquire));
        release.send(()).unwrap();
        assert!(done.await.unwrap());
        assert!(store.get_meta("cancelled_query_write").unwrap().is_none());
        // Cancellation belongs to one request; a later request can use the store.
        let retry = QueryWork::new(Instant::now(), 2_000).unwrap();
        retry
            .blocking(move |_| Ok(store.set_meta("retry", "ok")?))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn query_timeout_cancels_detached_blocking_stage_before_next_write() {
        cancelled_query_stage(true).await;
    }

    #[tokio::test]
    async fn dropped_query_cancels_detached_blocking_stage_before_next_write() {
        cancelled_query_stage(false).await;
    }

    #[tokio::test]
    async fn a_held_inspection_writer_obeys_the_query_deadline_without_blocking_the_runtime() {
        let root = tempfile::tempdir().unwrap();
        let store = Arc::new(store::open(":memory:", StoreLimits::default()).unwrap());
        let mut config = Config::default();
        config.limits.query_timeout_ms = 50;
        let context = Context {
            config_root: root.path().into(),
            root: root.path().into(),
            config,
            store: Arc::clone(&store),
            query_host: Default::default(),
            query_latencies: Default::default(),
            background_l1: Default::default(),
        };
        let (ready, acquired) = tokio::sync::oneshot::channel();
        let (release, released) = std::sync::mpsc::channel();
        let holder = tokio::task::spawn_blocking(move || {
            let _lock = store.writer().unwrap();
            ready.send(()).unwrap();
            // A broken synchronous implementation must fail, not hang the test runtime.
            let _ = released.recv_timeout(std::time::Duration::from_secs(2));
        });
        acquired.await.unwrap();
        let started = Instant::now();
        let heartbeat = tokio::spawn(async {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            Instant::now()
        });
        let response = context.search(request(Mode::Lexical), false).await;
        let refs = context.refs("payment", None, false).await.unwrap();
        assert_eq!(refs["index"]["reason"], "query_timeout");
        assert_eq!(refs["groups"], json!([]));
        assert_eq!(
            context.symbol("payment").await.unwrap_err().code(),
            "timeout"
        );
        assert_eq!(
            context.open("missing.rs", 0).await.unwrap_err().code(),
            "timeout"
        );
        let mapped = context.map(2000, &[]).await.unwrap();
        assert_eq!(mapped["index"]["reason"], "query_timeout");
        assert_eq!(mapped["nodes"], json!([]));
        let elapsed = started.elapsed();
        let heartbeat_at = heartbeat.await.unwrap();
        let _ = release.send(());
        holder.await.unwrap();
        let response = response.unwrap();
        assert_eq!(response["hits"], json!([]));
        assert_eq!(response["index"]["partial"], true);
        assert_eq!(response["index"]["reason"], "query_timeout");
        assert_eq!(response["index"]["fresh"]["state"], "unknown");
        assert!(response["index"]["head"].is_null());
        assert!(response["index"]["l1_pending_files"].is_null());
        assert!(response["index"]["suppressed_hits"].is_null());
        // Releasing the lock lets a later call complete normally.
        let mut resumed = context.clone();
        resumed.config.limits.query_timeout_ms = 2_000;
        persist_fixture_config(&resumed);
        let resumed = resumed.search(request(Mode::Lexical), false).await.unwrap();
        assert_eq!(resumed["index"]["partial"], false);
        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "{elapsed:?}"
        );
        assert!(heartbeat_at.duration_since(started) < std::time::Duration::from_millis(200));
    }

    #[tokio::test]
    async fn open_cancellation_discards_source_work_and_retry_preserves_pages() {
        for timeout in [false, true] {
            let (_root, mut context) = fixture();
            context.config.limits.query_timeout_ms = if timeout { 50 } else { 2000 };
            let (entered, ready) = tokio::sync::oneshot::channel();
            let (release, released) = std::sync::mpsc::channel();
            let (finished, done) = tokio::sync::oneshot::channel();
            let worker = context.clone();
            let task = tokio::spawn(async move {
                worker
                    .open_checked(
                        "src/retry.rs:1-1",
                        0,
                        move |work| {
                            entered.send(()).unwrap();
                            released
                                .recv_timeout(std::time::Duration::from_secs(2))
                                .unwrap();
                            finished.send(work.check().is_err()).unwrap();
                        },
                        || {},
                    )
                    .await
            });
            ready.await.unwrap();
            if timeout {
                assert_eq!(task.await.unwrap().unwrap_err().code(), "timeout");
            } else {
                task.abort();
                assert!(task.await.unwrap_err().is_cancelled());
            }
            release.send(()).unwrap();
            assert!(done.await.unwrap());
            context.config.limits.query_timeout_ms = 2000;
            let page = context.open("src/retry.rs:1-1", 0).await.unwrap();
            assert_eq!(page["text"], "retry payment");
            assert_eq!(page["lines"], json!([1, 1]));
        }
    }

    #[tokio::test]
    async fn shared_query_latency_counts_search_once_including_timeouts_without_writes() {
        let (root, mut context) = fixture();
        let observer = context.clone();
        let writes = context.store.writer().unwrap().total_changes();
        assert_eq!(context.status().unwrap()["query_latency_ms"]["count"], 0);
        context.search(request(Mode::Lexical), false).await.unwrap();
        assert_eq!(observer.status().unwrap()["query_latency_ms"]["count"], 1);
        context.config.limits.query_timeout_ms = 0;
        let response = context.search(request(Mode::Hybrid), false).await.unwrap();
        assert_eq!(response["index"]["reason"], "query_timeout");
        assert!(response["index"]["dirty_files"].is_null());
        assert_eq!(response["index"]["dirty_files_state"], "not_checked");
        let status = observer.status().unwrap();
        assert!(status["dirty_files"].is_null());
        assert_eq!(status["dirty_files_state"], "not_checked");
        assert_eq!(status["query_latency_ms"]["count"], 2);
        assert_eq!(status["query_latency_ms"]["lifetime"], "process");
        assert!(
            status["query_latency_ms"]["p95"]
                .as_f64()
                .unwrap()
                .is_finite()
        );
        assert_eq!(context.store.writer().unwrap().total_changes(), writes);
        // A separate owner has no samples from previous CLI/server lifetimes.
        assert_eq!(
            Context::load(root.path()).unwrap().status().unwrap()["query_latency_ms"]["count"],
            0
        );
    }

    #[tokio::test]
    async fn navigation_and_status_preserve_shapes_without_false_freshness() {
        let (root, context) = fixture();
        assert_eq!(context.config_root, root.path());
        let opened = context.open("src/retry.rs:1-1", 0).await.unwrap();
        assert_eq!(opened["text"], "retry payment");
        assert_ne!(opened["fresh"]["state"], "fresh");
        let symbols = context.symbol("should_retry").await.unwrap();
        assert_eq!(symbols[0]["approximate"], true);
        assert_ne!(symbols[0]["fresh"]["state"], "fresh");
        assert_eq!(
            context.refs("should_retry", None, false).await.unwrap()["approximate"],
            true
        );
        assert!(context.map(2000, &[]).await.unwrap()["tree"].is_string());
        let status = context.status().unwrap();
        for field in [
            "rerank_auto",
            "last_local_reranker_check",
            "l2_pending",
            "l2",
            "endpoints",
            "endpoint_accounting",
            "unknown_endpoint_dispatches",
            "queue",
            "queue_status",
            "root",
            "head",
            "indexed_head",
            "model_preparation",
            "l1_scan_generation",
            "l1_completed_generation",
            "files",
            "units",
            "symbols",
            "edges",
            "truncated_files",
            "l1_pending_files",
            "graph_pending",
            "l1_deferred",
            "caps",
            "footprint_bytes",
            "tables",
        ] {
            assert!(status.get(field).is_some(), "missing {field}");
        }
        let text = context.status_text().unwrap();
        assert!(text.contains("1 files, 1 units"));
        assert!(text.contains("model preparation: not attempted"));
        assert!(
            text.contains(
                &queue::status::snapshot(&context.store, unix_seconds().unwrap())
                    .unwrap()
                    .summary()
            )
        );
        assert!(
            text.contains(
                &queue::inspection::read_configured(&context.store, &context.config)
                    .unwrap()
                    .summary()
            )
        );
        assert_eq!(status["units"], 1);
        assert_eq!(
            status["endpoint_accounting"],
            "potential_dispatches_including_retries"
        );
        assert_ne!(status["fresh"]["state"], "fresh");
        let error = context.open("missing.rs", 0).await.unwrap_err();
        assert_eq!(error.code(), "not_found");
        assert!(!error.message().contains("missing.rs"));
        assert!(!error.hint().is_empty());
    }

    #[test]
    fn head_freshness_includes_missing_commit_transitions() {
        for (indexed_head, head, changed) in [
            (Some("old"), None, true),
            (Some(""), Some("new"), true),
            (None, Some("new"), true),
            (Some("old"), Some("new"), true),
            (Some("same"), Some("same"), false),
            (Some(""), None, false),
            (None, None, false),
        ] {
            let inspection = Inspection {
                head: head.map(str::to_owned),
                indexed_head: indexed_head.map(str::to_owned),
                pending: 0,
            };
            assert_eq!(
                inspection.index()["fresh"]["state"],
                if changed { "stale" } else { "unknown" }
            );
            assert_eq!(
                inspection.freshness()["reason"],
                if changed {
                    "head_changed"
                } else {
                    "source_not_verified"
                }
            );
        }
    }

    #[test]
    fn source_annotations_preserve_existing_cap_and_deferred_evidence() {
        let inspection = Inspection {
            head: Some("new".into()),
            indexed_head: Some("old".into()),
            pending: 1,
        };
        let mut capped =
            json!({"fresh":{"state":"capped","reason":"source_bytes","retained_snapshot":true}});
        let prior = capped.clone();
        inspection.annotate(&mut capped);
        assert_eq!(capped, prior);
        let mut deferred = json!({"l1_deferred_reason":"source_bytes"});
        inspection.annotate(&mut deferred);
        assert_eq!(
            deferred["fresh"],
            json!({"state":"capped","reason":"source_bytes"})
        );
        assert_eq!(deferred["l1_deferred_reason"], "source_bytes");
        let mut claimed_fresh = json!({"fresh":{"state":"fresh"}});
        inspection.annotate(&mut claimed_fresh);
        assert_eq!(claimed_fresh["fresh"]["reason"], "head_changed");
    }

    fn git(root: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap().trim().to_owned()
    }

    #[tokio::test]
    async fn search_rechecks_head_after_empty_retrieval_and_retries_with_new_work() {
        for query_text in ["missing_identifier", "retry payment"] {
            let (root, context) = fixture();
            let mut query = request(Mode::Lexical);
            query.query = query_text.into();
            let started = Instant::now();
            let work = QueryWork::new(started, 2000).unwrap();
            let path = root.path().to_owned();
            let result = context
                .search_within_checked(query, false, started, &work, move || {
                    git(&path, &["init", "--quiet"]);
                    git(
                        &path,
                        &[
                            "-c",
                            "user.name=Search Test",
                            "-c",
                            "user.email=search@example.invalid",
                            "-c",
                            "commit.gpgsign=false",
                            "commit",
                            "--quiet",
                            "--allow-empty",
                            "-m",
                            "late",
                        ],
                    );
                })
                .await
                .unwrap();
            assert_eq!(result["hits"], json!([]));
            assert_eq!(result["index"]["branch_resync"]["state"], "not_needed");
            assert_eq!(result["index"]["partial"], true);
            assert_eq!(result["index"]["degraded"], true);
            assert_eq!(result["index"]["lexical"], "incomplete");
            assert_eq!(
                result["index"]["freshness_refusals"]["head_changed"],
                if query_text == "retry payment" { 1 } else { 0 }
            );
            let retry = context.search(request(Mode::Lexical), false).await.unwrap();
            assert_eq!(
                retry["index"]["branch_resync"]["state"], "completed",
                "{retry}"
            );
            assert_eq!(retry["index"]["partial"], false);
        }
    }

    fn configure_reference_tests(context: &mut Context) {
        context.config.tests.globs = vec!["src/**".into()];
        persist_fixture_config(context);
        complete_fixture_source(context, "src/retry.rs");
        add(
            &context.store,
            "app/payment.rs",
            "payment",
            "fn payment() {}",
            None,
        )
        .unwrap();
        fs::create_dir_all(context.root.join("app")).unwrap();
        fs::write(context.root.join("app/payment.rs"), "fn payment() {}").unwrap();
        complete_fixture_source(context, "app/payment.rs");
    }

    fn duplicate_reference_contributor(context: &Context, graph: bool) -> i64 {
        let writer = context.store.writer().unwrap();
        writer.execute_batch("INSERT INTO units(hash,content_hash,source_key,duplicate_ordinal,file_id,kind,header,line_start,line_end,created_at,updated_at)
            SELECT randomblob(32),content_hash,source_key,1,file_id,kind,header,1,1,0,0 FROM units ORDER BY id LIMIT 1;").unwrap();
        let unit = writer.last_insert_rowid();
        writer
            .execute_batch(
                "DELETE FROM fts_units;
            INSERT INTO fts_units(rowid,header,idents,body,prose)
            SELECT id,'','payment','payment','' FROM units;",
            )
            .unwrap();
        if graph {
            writer
                .execute(
                    "INSERT INTO symbols(file_id,unit_id,name,qualified,kind,line_start,line_end)
                SELECT file_id,?1,'second','second',kind,1,1 FROM symbols ORDER BY id LIMIT 1",
                    [unit],
                )
                .unwrap();
            writer
                .execute_batch(
                    "INSERT INTO edges(src_symbol,dst_name,kind,file_id,line)
                SELECT id,'payment','test_of',file_id,1 FROM symbols WHERE file_id=(SELECT id FROM files WHERE path='src/retry.rs');",
                )
                .unwrap();
        }
        unit
    }

    #[tokio::test]
    async fn refs_contributors_refuse_hidden_non_anchor_changes_and_recover() {
        for graph in [false, true] {
            for mutation in ["header", "delete", "association"] {
                if !graph && mutation == "association" {
                    continue;
                }
                let (_root, mut context) = fixture();
                if graph {
                    configure_reference_tests(&mut context);
                }
                let unit = duplicate_reference_contributor(&context, graph);
                let initial = context.refs("payment", Some(1), graph).await.unwrap();
                assert_eq!(initial["returned_total"], 1);
                assert_eq!(initial["index"]["partial"], false);
                let selected = navigation::refs(&context.store, "payment", Some(1), graph).unwrap();
                let generation = context.store.l1_generation().unwrap();
                let store = Arc::clone(&context.store);
                let result = context.refs_checked("payment", Some(1), graph, move || {
                    let writer = store.writer().unwrap();
                    match mutation {
                        "header" => { writer.execute("UPDATE units SET header='changed' WHERE id=?1", [unit]).unwrap(); }
                        "delete" => { writer.execute("DELETE FROM units WHERE id=?1", [unit]).unwrap(); }
                        "association" => { writer.execute("UPDATE symbols SET unit_id=(SELECT min(id) FROM units) WHERE unit_id=?1", [unit]).unwrap(); }
                        _ => unreachable!(),
                    }
                }).await.unwrap();
                let current = navigation::refs(&context.store, "payment", Some(1), graph).unwrap();
                assert_eq!(current.groups, selected.groups, "{graph} {mutation}");
                assert_eq!(current.total, selected.total);
                assert_eq!(context.store.l1_generation().unwrap(), generation);
                assert_eq!(result["groups"], json!([]), "{graph} {mutation}: {result}");
                assert_eq!(
                    result["index"]["freshness_refusals"]["occurrence_changed"],
                    1
                );
                assert!(result["total"].is_null());
                assert_eq!(result["index"]["partial"], true);
                assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
                // A new call observes the new relationships without stale retained evidence.
                let retry = context.refs("payment", Some(1), graph).await.unwrap();
                assert_eq!(retry["groups"], initial["groups"]);
                assert_eq!(retry["index"]["partial"], false);
                assert!(
                    serde_json::to_value(current)
                        .unwrap()
                        .get("contributors")
                        .is_none()
                );
            }
        }
    }

    #[tokio::test]
    async fn refs_contributors_preserve_other_groups_filters_and_ladder_progress() {
        for tests_only in [false, true] {
            let (root, mut context) = fixture();
            configure_reference_tests(&mut context);
            let unit = duplicate_reference_contributor(&context, true);
            add(
                &context.store,
                "src/other.rs",
                "other",
                "retry payment",
                Some("payment"),
            )
            .unwrap();
            fs::write(root.path().join("src/other.rs"), "retry payment").unwrap();
            complete_fixture_source(&context, "src/other.rs");
            context
                .store
                .writer()
                .unwrap()
                .execute("UPDATE edges SET kind='test_of'", [])
                .unwrap();
            let initial = context.refs("payment", None, tests_only).await.unwrap();
            let store = Arc::clone(&context.store);
            let result = context
                .refs_checked("payment", None, tests_only, move || {
                    let writer = store.writer().unwrap();
                    writer
                        .execute("UPDATE units SET ladder_level=2", [])
                        .unwrap();
                    writer
                        .execute("UPDATE units SET header='changed' WHERE id=?1", [unit])
                        .unwrap();
                })
                .await
                .unwrap();
            assert_eq!(initial["returned_total"], 2);
            assert_eq!(result["returned_total"], 1);
            assert_eq!(result["groups"][0]["path"], "src/other.rs");
            assert_eq!(result["index"]["partial"], true);
            assert!(result["total"].is_null());
            let store = Arc::clone(&context.store);
            let recovered = context
                .refs_checked("payment", Some(1), tests_only, move || {
                    store
                        .writer()
                        .unwrap()
                        .execute("UPDATE units SET ladder_level=3", [])
                        .unwrap();
                })
                .await
                .unwrap();
            assert_eq!(recovered["index"]["partial"], false);
            assert_eq!(recovered["returned_total"], 1);
        }
    }

    #[tokio::test]
    async fn refs_preserve_verified_group_when_unrelated_source_becomes_pending() {
        let (root, context) = fixture();
        context.store.writer().unwrap().execute("INSERT INTO edges(src_symbol,dst_name,kind,file_id,line) SELECT id,'payment','ref',file_id,1 FROM symbols", []).unwrap();
        add(
            &context.store,
            "src/unrelated.rs",
            "unrelated",
            "unrelated source",
            None,
        )
        .unwrap();
        fs::write(root.path().join("src/unrelated.rs"), "unrelated source").unwrap();
        complete_fixture_source(&context, "src/unrelated.rs");
        let original = context.refs("payment", None, false).await.unwrap();
        assert_eq!(original["index"]["partial"], false);
        assert_eq!(original["returned_total"], 1);
        let store = Arc::clone(&context.store);
        let result = context
            .refs_checked("payment", None, false, move || {
                store
                    .writer()
                    .unwrap()
                    .execute(
                        "UPDATE files SET l1_pending=1 WHERE path='src/unrelated.rs'",
                        [],
                    )
                    .unwrap();
            })
            .await
            .unwrap();
        assert_eq!(result["groups"], original["groups"]);
        assert_eq!(result["returned_total"], 1);
        assert!(result["total"].is_null());
        assert_eq!(result["index"]["partial"], true);
        assert_eq!(result["index"]["degraded"], true);
        assert_eq!(result["index"]["l1_pending_files"], 1);
        assert_eq!(result["index"]["suppressed_references"], 0);
        store_pending_clear(&context);
        let recovered = context.refs("payment", None, false).await.unwrap();
        assert_eq!(recovered["index"]["partial"], false);
        assert_eq!(recovered["groups"], original["groups"]);
        assert_eq!(recovered["total"], original["total"]);
    }

    #[tokio::test]
    async fn refs_final_source_fence_refuses_late_edits_deletes_and_replacements() {
        for mutation in ["edit", "delete", "replace"] {
            let (root, context) = fixture();
            context.store.writer().unwrap().execute("INSERT INTO edges(src_symbol,dst_name,kind,file_id,line) SELECT id,'payment','ref',file_id,1 FROM symbols", []).unwrap();
            let original = context.refs("payment", None, false).await.unwrap();
            assert_eq!(original["returned_total"], 1);
            let writes = context.store.writer().unwrap().total_changes();
            let path = root.path().join("src/retry.rs");
            let result = context
                .refs_checked("payment", None, false, move || match mutation {
                    "edit" => fs::write(&path, "changed bytes").unwrap(),
                    "delete" => fs::remove_file(&path).unwrap(),
                    "replace" => {
                        let replacement = path.with_extension("replacement");
                        fs::write(&replacement, "replacement bytes").unwrap();
                        fs::rename(replacement, &path).unwrap();
                    }
                    _ => unreachable!(),
                })
                .await
                .unwrap();
            assert_eq!(result["groups"], json!([]), "{mutation}: {result}");
            assert!(result["total"].is_null());
            assert_eq!(result["index"]["partial"], true);
            assert_eq!(result["index"]["degraded"], true);
            assert_eq!(result["index"]["fresh"]["state"], "stale");
            assert_eq!(result["index"]["suppressed_references"], 1);
            assert_eq!(
                result["index"]["freshness_refusals"][if mutation == "delete" {
                    "source_unavailable"
                } else {
                    "source_changed"
                }],
                1
            );
            assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
            assert_eq!(context.store.writer().unwrap().total_changes(), writes);
            fs::write(root.path().join("src/retry.rs"), "retry payment").unwrap();
            let retry = context.refs("payment", None, false).await.unwrap();
            assert_eq!(retry["groups"], original["groups"]);
            assert_eq!(retry["index"]["partial"], false);
            assert!(retry["groups"][0].get("evidence").is_none());
        }
    }

    #[tokio::test]
    async fn refs_final_source_fence_retains_anchor_identity_with_equal_groups() {
        for (sql, restore) in [
            (
                "UPDATE units SET header=header || 'changed'",
                "UPDATE units SET header=substr(header,1,length(header)-7)",
            ),
            (
                "UPDATE units SET scope_path='changed'",
                "UPDATE units SET scope_path=NULL",
            ),
            ("UPDATE units SET line_end=2", "UPDATE units SET line_end=1"),
        ] {
            let (_root, context) = fixture();
            context.store.writer().unwrap().execute("INSERT INTO edges(src_symbol,dst_name,kind,file_id,line) SELECT id,'payment','ref',file_id,1 FROM symbols", []).unwrap();
            let original = context.refs("payment", None, false).await.unwrap();
            let selected = navigation::refs(&context.store, "payment", None, false).unwrap();
            let generation = context.store.l1_generation().unwrap();
            let store = Arc::clone(&context.store);
            let result = context
                .refs_checked("payment", None, false, move || {
                    store.writer().unwrap().execute(sql, []).unwrap();
                })
                .await
                .unwrap();
            assert_eq!(
                navigation::refs(&context.store, "payment", None, false)
                    .unwrap()
                    .groups,
                selected.groups
            );
            assert_eq!(context.store.l1_generation().unwrap(), generation);
            assert_eq!(result["groups"], json!([]), "{sql}: {result}");
            assert!(result["total"].is_null());
            assert_eq!(result["index"]["partial"], true);
            assert_eq!(
                result["index"]["freshness_refusals"]["occurrence_changed"],
                1
            );
            assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
            context
                .store
                .writer()
                .unwrap()
                .execute(restore, [])
                .unwrap();
            let retry = context.refs("payment", None, false).await.unwrap();
            assert_eq!(retry["groups"], original["groups"]);
            assert_eq!(retry["index"]["partial"], false);
        }
    }

    #[tokio::test]
    async fn refs_cached_final_proof_rechecks_external_same_generation_unit_change() {
        let (root, context) = fixture();
        context.store.writer().unwrap().execute("INSERT INTO edges(src_symbol,dst_name,kind,file_id,line) SELECT id,'payment','ref',file_id,1 FROM symbols", []).unwrap();
        let original = context.refs("payment", None, false).await.unwrap();
        assert_eq!(original["returned_total"], 1);
        let generation = context.store.l1_generation().unwrap();
        let database = root.path().join(".deepkit/code.db");
        assert!(database.exists());
        let external = rusqlite::Connection::open(database).unwrap();
        let result = context
            .refs_checked("payment", None, false, move || {
                external
                    .execute("UPDATE units SET header=header || 'external'", [])
                    .unwrap();
            })
            .await
            .unwrap();
        assert_eq!(context.store.l1_generation().unwrap(), generation);
        assert_eq!(result["groups"], json!([]));
        assert_eq!(
            result["index"]["freshness_refusals"]["occurrence_changed"],
            1
        );
        assert_eq!(result["index"]["partial"], true);
    }

    #[tokio::test]
    async fn refs_final_source_fence_preserves_other_groups_and_ladder_progress() {
        let (root, context) = fixture();
        add(
            &context.store,
            "src/other.rs",
            "other",
            "retry payment",
            None,
        )
        .unwrap();
        fs::write(root.path().join("src/other.rs"), "retry payment").unwrap();
        complete_fixture_source(&context, "src/other.rs");
        context.store.writer().unwrap().execute("INSERT INTO edges(src_symbol,dst_name,kind,file_id,line) SELECT id,'payment','ref',file_id,1 FROM symbols", []).unwrap();
        let original = context.refs("payment", None, false).await.unwrap();
        assert_eq!(original["returned_total"], 2);
        let path = root.path().join("src/retry.rs");
        let store = Arc::clone(&context.store);
        let result = context
            .refs_checked("payment", None, false, move || {
                fs::write(path, "changed bytes").unwrap();
                store
                    .writer()
                    .unwrap()
                    .execute("UPDATE units SET ladder_level=2", [])
                    .unwrap();
            })
            .await
            .unwrap();
        assert_eq!(result["groups"].as_array().unwrap().len(), 1);
        assert_eq!(result["groups"][0]["path"], "src/other.rs");
        assert_eq!(result["returned_total"], 1);
        assert_eq!(result["indexed_total"], 2);
        assert!(result["total"].is_null());
        assert_eq!(result["index"]["freshness_refusals"]["source_changed"], 1);
        assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
        fs::write(root.path().join("src/retry.rs"), "retry payment").unwrap();
        let retry = context.refs("payment", None, false).await.unwrap();
        assert_eq!(retry["groups"], original["groups"]);
        assert_eq!(retry["index"]["partial"], false);
    }

    #[tokio::test]
    async fn refs_final_selection_refuses_late_settings_even_when_empty() {
        for empty in [false, true] {
            for mutation in ["root", "name", "limit", "invalid"] {
                let (_root, context) = fixture();
                context.store.writer().unwrap().execute("INSERT INTO edges(src_symbol,dst_name,kind,file_id,line) SELECT id,'payment','ref',file_id,1 FROM symbols", []).unwrap();
                let target = if empty { "absent" } else { "payment" };
                let original = context.refs(target, Some(1), false).await.unwrap();
                assert_eq!(original["index"]["partial"], false);
                assert_eq!(original["returned_total"], usize::from(!empty));
                let worker = context.clone();
                let writes = context.store.writer().unwrap().total_changes();
                let result = context
                    .refs_checked(target, Some(1), false, move || {
                        if mutation == "invalid" {
                            fs::write(worker.config_root.join(".deepkit/config.toml"), "[invalid")
                                .unwrap();
                        } else {
                            let mut changed = worker;
                            match mutation {
                                "root" => changed.config.index.root = "src".into(),
                                "name" => changed.config.index.repo_name = "changed".into(),
                                "limit" => changed.config.limits.max_file_bytes = "1 B".into(),
                                _ => unreachable!(),
                            }
                            persist_fixture_config(&changed);
                        }
                    })
                    .await
                    .unwrap();
                assert_eq!(result["groups"], json!([]), "{empty}/{mutation}: {result}");
                assert!(result["total"].is_null());
                assert_eq!(result["index"]["partial"], true);
                assert_eq!(result["index"]["degraded"], true);
                assert_eq!(result["index"]["fresh"]["state"], "stale");
                assert_eq!(
                    result["index"]["freshness_refusals"]["settings_changed"],
                    usize::from(!empty)
                );
                assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
                assert_eq!(context.store.writer().unwrap().total_changes(), writes);
                persist_fixture_config(&context);
                let retry = context.refs(target, Some(1), false).await.unwrap();
                assert_eq!(retry["groups"], original["groups"]);
                assert_eq!(retry["index"]["partial"], false);
            }
        }
    }

    #[tokio::test]
    async fn refs_final_selection_repeats_filters_and_prelimit_total() {
        for (target, k, tests_only, sql) in [
            (
                "absent",
                None,
                false,
                "INSERT INTO edges(src_symbol,dst_name,kind,file_id,line) SELECT id,'absent','ref',file_id,1 FROM symbols",
            ),
            ("payment", Some(1), false, "UPDATE edges SET kind='import'"),
            // k keeps the same first group, but the pre-limit count changes.
            (
                "payment",
                Some(1),
                false,
                "INSERT INTO edges(src_symbol,dst_name,kind,file_id,line) SELECT id,'payment','test_of',file_id,1 FROM symbols",
            ),
            (
                "payment",
                Some(1),
                true,
                "INSERT INTO edges(src_symbol,dst_name,kind,file_id,line) SELECT id,'payment','test_of',file_id,1 FROM symbols",
            ),
        ] {
            let (_root, context) = fixture();
            context.store.writer().unwrap().execute("INSERT INTO edges(src_symbol,dst_name,kind,file_id,line) SELECT id,'payment','ref',file_id,1 FROM symbols", []).unwrap();
            let original = context.refs(target, k, tests_only).await.unwrap();
            let generation = context.store.l1_generation().unwrap();
            let store = Arc::clone(&context.store);
            let result = context
                .refs_checked(target, k, tests_only, move || {
                    store.writer().unwrap().execute(sql, []).unwrap();
                })
                .await
                .unwrap();
            assert_eq!(
                result["groups"],
                json!([]),
                "{target}/{tests_only}/{sql}: {result}"
            );
            assert!(result["total"].is_null());
            assert_eq!(result["index"]["partial"], true);
            assert_eq!(result["index"]["degraded"], true);
            assert!(
                result["index"]["freshness_refusals"]
                    .get("selection_changed")
                    .is_some()
            );
            assert_eq!(result["indexed_total"], original["indexed_total"]);
            assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
            assert_eq!(context.store.l1_generation().unwrap(), generation);
            let retry = context.refs(target, k, tests_only).await.unwrap();
            assert_eq!(retry["index"]["partial"], false, "{sql}: {retry}");
            assert!(retry["total"].is_number());
        }
    }

    #[tokio::test]
    async fn refs_final_selection_allows_irrelevant_updates_and_keeps_deadline() {
        let (_root, context) = fixture();
        context.store.writer().unwrap().execute("INSERT INTO edges(src_symbol,dst_name,kind,file_id,line) SELECT id,'payment','ref',file_id,1 FROM symbols", []).unwrap();
        for tests_only in [false, true] {
            let original = context.refs("payment", Some(1), tests_only).await.unwrap();
            let store = Arc::clone(&context.store);
            let result = context.refs_checked("payment", Some(1), tests_only, move || {
                let writer = store.writer().unwrap();
                writer.execute("UPDATE units SET ladder_level=2", []).unwrap();
                writer.execute("INSERT INTO edges(src_symbol,dst_name,kind,file_id,line) SELECT id,'unrelated','ref',file_id,1 FROM symbols", []).unwrap();
            }).await.unwrap();
            assert_eq!(result["groups"], original["groups"]);
            assert_eq!(result["total"], original["total"]);
            assert_eq!(result["index"]["partial"], false);
        }
        let mut short = context.clone();
        short.config.limits.query_timeout_ms = 100;
        persist_fixture_config(&short);
        let result = short
            .refs_checked("absent", Some(1), true, || {
                std::thread::sleep(std::time::Duration::from_millis(150));
            })
            .await
            .unwrap();
        assert_eq!(result["index"]["reason"], "query_timeout");
        assert!(result["total"].is_null());
        assert_eq!(result["groups"], json!([]));
        persist_fixture_config(&context);
        assert_eq!(
            context.refs("payment", Some(1), false).await.unwrap()["index"]["partial"],
            false
        );
    }

    #[tokio::test]
    async fn maps_final_source_preserves_verified_files_and_budget() {
        for mutation in ["edit", "delete", "replace"] {
            let (root, context) = fixture();
            add(
                &context.store,
                "src/other.rs",
                "other",
                "retry payment",
                None,
            )
            .unwrap();
            fs::write(root.path().join("src/other.rs"), "retry payment").unwrap();
            complete_fixture_source(&context, "src/other.rs");
            let original = context.map(2000, &[]).await.unwrap();
            assert_eq!(original["nodes"].as_array().unwrap().len(), 2);
            let path = root.path().join("src/retry.rs");
            let store = Arc::clone(&context.store);
            let result = context
                .map_checked(2000, &[], move || {
                    match mutation {
                        "edit" => fs::write(&path, "changed bytes").unwrap(),
                        "delete" => fs::remove_file(&path).unwrap(),
                        "replace" => {
                            let replacement = path.with_extension("replacement");
                            fs::write(&replacement, "replacement bytes").unwrap();
                            fs::rename(replacement, &path).unwrap();
                        }
                        _ => unreachable!(),
                    }
                    store
                        .writer()
                        .unwrap()
                        .execute("UPDATE units SET ladder_level=2", [])
                        .unwrap();
                })
                .await
                .unwrap();
            assert_eq!(
                result["nodes"].as_array().unwrap().len(),
                1,
                "{mutation}: {result}"
            );
            assert_eq!(result["nodes"][0]["path"], "src/other.rs");
            assert!(!result["tree"].as_str().unwrap().contains("retry.rs"));
            assert!(result["tree"].as_str().unwrap().contains("other.rs"));
            assert!(
                tiktoken_rs::cl100k_base_singleton()
                    .encode_with_special_tokens(result["tree"].as_str().unwrap())
                    .len()
                    <= 2000
            );
            assert_eq!(result["index"]["partial"], true);
            assert_eq!(result["index"]["fresh"]["state"], "stale");
            assert_eq!(result["index"]["suppressed_files"], 1);
            assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
            fs::write(root.path().join("src/retry.rs"), "retry payment").unwrap();
            let retry = context.map(2000, &[]).await.unwrap();
            assert_eq!(retry["nodes"], original["nodes"]);
            assert_eq!(retry["tree"], original["tree"]);
            assert_eq!(retry["index"]["partial"], false);
        }
    }

    #[tokio::test]
    async fn maps_final_source_retains_hidden_occurrences_without_signatures() {
        for budget in [10, 2000] {
            for (sql, undo) in [
                (
                    "UPDATE units SET header=header || 'changed'",
                    "UPDATE units SET header=substr(header,1,length(header)-7)",
                ),
                (
                    "UPDATE units SET scope_path='changed'",
                    "UPDATE units SET scope_path=NULL",
                ),
                ("UPDATE units SET line_end=2", "UPDATE units SET line_end=1"),
            ] {
                let (_root, context) = fixture();
                let original = context.map(budget, &[]).await.unwrap();
                assert_eq!(original["nodes"].as_array().unwrap().len(), 1);
                let store = Arc::clone(&context.store);
                let generation = context.store.l1_generation().unwrap();
                let result = context
                    .map_checked(budget, &[], move || {
                        store.writer().unwrap().execute(sql, []).unwrap();
                    })
                    .await
                    .unwrap();
                assert_eq!(result["nodes"], json!([]), "{budget}/{sql}: {result}");
                assert_eq!(result["tree"], "");
                assert_eq!(
                    result["index"]["freshness_refusals"]["occurrence_changed"],
                    1
                );
                assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
                assert_eq!(context.store.l1_generation().unwrap(), generation);
                context.store.writer().unwrap().execute(undo, []).unwrap();
                let retry = context.map(budget, &[]).await.unwrap();
                assert_eq!(retry["nodes"], original["nodes"]);
                assert_eq!(retry["index"]["partial"], false);
            }
        }
    }

    #[tokio::test]
    async fn maps_final_source_retains_nonanchor_symbol_occurrence() {
        for field in ["header", "scope_path"] {
            let (_root, context) = fixture();
            let unit = duplicate_reference_contributor(&context, true);
            let original = context.map(2000, &[]).await.unwrap();
            assert_eq!(original["nodes"][0]["symbols"].as_array().unwrap().len(), 2);
            let store = Arc::clone(&context.store);
            let result = context
                .map_checked(2000, &[], move || {
                    store
                        .writer()
                        .unwrap()
                        .execute(
                            &format!("UPDATE units SET {field}='changed' WHERE id=?1"),
                            [unit],
                        )
                        .unwrap();
                })
                .await
                .unwrap();
            assert_eq!(result["nodes"], json!([]), "{field}: {result}");
            assert_eq!(
                result["index"]["freshness_refusals"]["occurrence_changed"],
                1
            );
            assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
            let retry = context.map(2000, &[]).await.unwrap();
            assert_eq!(retry["nodes"], original["nodes"]);
            assert_eq!(retry["index"]["partial"], false);
        }
    }

    #[tokio::test]
    async fn maps_final_source_rechecks_empty_and_l0_catalogue_authority() {
        for l0 in [false, true] {
            for mutation in ["source", "metadata", "pending"] {
                let (root, mut context) = fixture();
                let path = "src/retry.rs";
                let bytes = if l0 { "catalogue only" } else { "" };
                fs::write(root.path().join(path), bytes).unwrap();
                if l0 {
                    context.config.treatments.insert(
                        0,
                        crate::config::Treatment {
                            name: "catalogue-only".into(),
                            patterns: vec!["**/*.rs".into()],
                            chunker: None,
                            max_level: 0,
                            enrich: vec![],
                            never_send: false,
                        },
                    );
                }
                persist_fixture_config(&context);
                assert!(
                    super::super::repair::target(&context, path, sync::l1::Work::default())
                        .unwrap()
                );
                let original = context.map(2000, &[]).await.unwrap();
                assert_eq!(original["nodes"].as_array().unwrap().len(), 1);
                let worker = context.clone();
                let result = context
                    .map_checked(2000, &[], move || match mutation {
                        "source" => fs::write(worker.root.join(path), "changed").unwrap(),
                        "metadata" => {
                            worker
                                .store
                                .writer()
                                .unwrap()
                                .execute("UPDATE files SET mtime_ns=mtime_ns+1", [])
                                .unwrap();
                        }
                        "pending" => {
                            worker
                                .store
                                .writer()
                                .unwrap()
                                .execute("UPDATE files SET l1_pending=1", [])
                                .unwrap();
                        }
                        _ => unreachable!(),
                    })
                    .await
                    .unwrap();
                assert_eq!(result["nodes"], json!([]), "{l0}/{mutation}: {result}");
                assert_eq!(result["tree"], "");
                assert_eq!(result["index"]["partial"], true);
                assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
                fs::write(root.path().join(path), bytes).unwrap();
                assert!(
                    super::super::repair::target(&context, path, sync::l1::Work::default())
                        .unwrap()
                );
                let retry = context.map(2000, &[]).await.unwrap();
                assert_eq!(retry["nodes"], original["nodes"]);
                assert_eq!(retry["index"]["partial"], false);
            }
        }
    }

    #[tokio::test]
    async fn maps_focus_contributors_detect_hidden_changes_and_recover() {
        for budget in [1, 10, 2000] {
            for mutation in [
                "lexical_header",
                "lexical_remove",
                "symbol_name",
                "symbol_owner",
            ] {
                let (_root, context) = fixture();
                let unit = duplicate_reference_contributor(&context, true);
                let focus = vec!["payment".to_owned(), "second".to_owned()];
                let original = context.map(budget, &focus).await.unwrap();
                assert_eq!(original["index"]["partial"], false);
                let before = super::super::map::map(&context.store, budget, &focus).unwrap();
                let generation = context.store.l1_generation().unwrap();
                let store = Arc::clone(&context.store);
                let result = context.map_checked(budget, &focus, move || {
                    let writer = store.writer().unwrap();
                    match mutation {
                        "lexical_header" => { writer.execute("UPDATE units SET header='changed' WHERE id=?1", [unit]).unwrap(); }
                        "lexical_remove" => { writer.execute("DELETE FROM fts_units WHERE rowid=?1", [unit]).unwrap(); }
                        "symbol_name" => { writer.execute("UPDATE symbols SET name='second_changed',qualified='second_changed' WHERE unit_id=?1", [unit]).unwrap(); }
                        "symbol_owner" => { writer.execute("UPDATE symbols SET unit_id=(SELECT min(id) FROM units) WHERE unit_id=?1", [unit]).unwrap(); }
                        _ => unreachable!(),
                    }
                }).await.unwrap();
                let after = super::super::map::map(&context.store, budget, &focus).unwrap();
                // Lexical changes remain hidden even when signatures are displayed;
                // symbol changes remain hidden when the budget omits signatures.
                if mutation.starts_with("lexical") || budget <= 10 {
                    assert_eq!(
                        serde_json::to_value(before).unwrap(),
                        serde_json::to_value(after).unwrap()
                    );
                }
                assert_eq!(context.store.l1_generation().unwrap(), generation);
                assert_eq!(result["nodes"], json!([]), "{budget}/{mutation}: {result}");
                assert_eq!(result["tree"], "");
                assert_eq!(
                    result["index"]["partial"], true,
                    "{budget}/{mutation}: {result}"
                );
                assert!(
                    result["index"]["freshness_refusals"]
                        .get("selection_changed")
                        .is_some()
                );
                assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
                let retry = context.map(budget, &focus).await.unwrap();
                assert_eq!(retry["index"]["partial"], false);
            }
        }
    }

    #[tokio::test]
    async fn maps_focus_contributors_retain_symbol_only_occurrences() {
        for budget in [1, 10] {
            let (_root, context) = fixture();
            let unit = duplicate_reference_contributor(&context, true);
            let focus = vec!["second".to_owned()];
            let original = context.map(budget, &focus).await.unwrap();
            assert_eq!(original["index"]["partial"], false);
            let store = Arc::clone(&context.store);
            let result = context
                .map_checked(budget, &focus, move || {
                    store
                        .writer()
                        .unwrap()
                        .execute("UPDATE units SET header='changed' WHERE id=?1", [unit])
                        .unwrap();
                })
                .await
                .unwrap();
            assert_eq!(result["nodes"], json!([]));
            assert_eq!(result["index"]["partial"], true);
            assert!(
                result["index"]["freshness_refusals"]
                    .get("selection_changed")
                    .is_some()
            );
            assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
            let retry = context.map(budget, &focus).await.unwrap();
            assert_eq!(retry["nodes"], original["nodes"]);
            assert_eq!(retry["index"]["partial"], false);
        }
    }

    #[tokio::test]
    async fn maps_focus_contributors_allow_l2_and_ignore_unmatched_symbols() {
        for budget in [1, 10, 2000] {
            let (_root, context) = fixture();
            duplicate_reference_contributor(&context, true);
            let focus = vec!["payment".to_owned()];
            let original = context.map(budget, &focus).await.unwrap();
            let store = Arc::clone(&context.store);
            let result = context
                .map_checked(budget, &focus, move || {
                    store
                        .writer()
                        .unwrap()
                        .execute("UPDATE units SET ladder_level=2", [])
                        .unwrap();
                })
                .await
                .unwrap();
            assert_eq!(result["nodes"], original["nodes"]);
            assert_eq!(result["tree"], original["tree"]);
            assert_eq!(result["index"]["partial"], false);
            assert!(result.get("selection").is_none());
        }
        let (_root, context) = fixture();
        duplicate_reference_contributor(&context, true);
        let focus = vec!["should_retry".to_owned()];
        let original = context.map(10, &focus).await.unwrap();
        let store = Arc::clone(&context.store);
        let result = context.map_checked(10, &focus, move || {
            store.writer().unwrap().execute("UPDATE symbols SET name='unrelated',qualified='unrelated' WHERE name='second'", []).unwrap();
        }).await.unwrap();
        assert_eq!(result["nodes"], original["nodes"]);
        assert_eq!(result["tree"], original["tree"]);
        assert_eq!(result["index"]["partial"], false);
    }

    #[tokio::test]
    async fn maps_final_selection_refuses_settings_even_when_empty() {
        for empty in [false, true] {
            for mutation in ["root", "name", "limit", "invalid"] {
                let (_root, context) = fixture();
                let focus = if empty { vec!["absent".into()] } else { vec![] };
                let original = context.map(2000, &focus).await.unwrap();
                let worker = context.clone();
                let writes = context.store.writer().unwrap().total_changes();
                let result = context
                    .map_checked(2000, &focus, move || {
                        if mutation == "invalid" {
                            fs::write(worker.config_root.join(".deepkit/config.toml"), "[invalid")
                                .unwrap();
                        } else {
                            let mut changed = worker;
                            match mutation {
                                "root" => changed.config.index.root = "src".into(),
                                "name" => changed.config.index.repo_name = "changed".into(),
                                "limit" => changed.config.limits.max_file_bytes = "1 B".into(),
                                _ => unreachable!(),
                            }
                            persist_fixture_config(&changed);
                        }
                    })
                    .await
                    .unwrap();
                assert_eq!(result["nodes"], json!([]), "{mutation}/{empty}: {result}");
                assert_eq!(result["tree"], "");
                assert_eq!(result["index"]["partial"], true);
                assert_eq!(result["index"]["degraded"], true);
                assert_eq!(result["index"]["fresh"]["state"], "stale");
                assert_eq!(
                    result["index"]["freshness_refusals"]["settings_changed"],
                    usize::from(!empty)
                );
                assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
                assert_eq!(context.store.writer().unwrap().total_changes(), writes);
                persist_fixture_config(&context);
                let retry = context.map(2000, &focus).await.unwrap();
                assert_eq!(retry["tree"], original["tree"]);
                assert_eq!(retry["nodes"], original["nodes"]);
                assert_eq!(retry["index"]["partial"], false);
            }
        }
    }

    #[tokio::test]
    async fn maps_final_selection_checks_hidden_ranks_ownership_and_original_focus() {
        for (mutation, undo, focus) in [
            (
                "UPDATE files SET rank=rank+1",
                "UPDATE files SET rank=rank-1",
                "",
            ),
            (
                "UPDATE symbols SET pagerank=pagerank+1",
                "UPDATE symbols SET pagerank=pagerank-1",
                "",
            ),
            ("UPDATE symbols SET signature='changed'", "", ""),
            ("UPDATE symbols SET unit_id=NULL", "", ""),
            ("UPDATE symbols SET name='absent'", "", "absent"),
            ("UPDATE symbols SET name='other'", "", "should_retry"),
        ] {
            let (_root, context) = fixture();
            let focus = if focus.is_empty() {
                vec![]
            } else {
                vec![focus.to_owned()]
            };
            let original = context.map(2000, &focus).await.unwrap();
            let symbols = context.store.tool_symbols_checked(&|| Ok(())).unwrap();
            let store = Arc::clone(&context.store);
            let generation = query_generation(&context.store, sync::l1::Work::default()).unwrap();
            let result = context
                .map_checked(2000, &focus, move || {
                    store.writer().unwrap().execute(mutation, []).unwrap();
                })
                .await
                .unwrap();
            assert_eq!(result["nodes"], json!([]), "{mutation}: {result}");
            assert_eq!(result["tree"], "");
            assert_eq!(result["index"]["partial"], true);
            assert_eq!(result["index"]["fresh"]["state"], "stale");
            assert!(
                result["index"]["freshness_refusals"]
                    .get("selection_changed")
                    .is_some()
            );
            assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 0);
            assert_eq!(
                query_generation(&context.store, sync::l1::Work::default()).unwrap(),
                generation
            );
            {
                let writer = context.store.writer().unwrap();
                if !undo.is_empty() {
                    writer.execute(undo, []).unwrap();
                }
                for symbol in symbols {
                    writer.execute("UPDATE symbols SET signature=?1,unit_id=?2,name=?3,pagerank=?4 WHERE id=?5",
                        rusqlite::params![symbol.signature,symbol.unit_id,symbol.name,symbol.pagerank,symbol.id]).unwrap();
                }
            }
            let retry = context.map(2000, &focus).await.unwrap();
            assert_eq!(retry["tree"], original["tree"]);
            assert_eq!(retry["nodes"], original["nodes"]);
            assert_eq!(retry["index"]["partial"], false);
        }
    }

    #[tokio::test]
    async fn maps_final_selection_allows_l2_and_obeys_original_deadline() {
        let (_root, context) = fixture();
        let original = context.map(2000, &[]).await.unwrap();
        let store = Arc::clone(&context.store);
        let result = context
            .map_checked(2000, &[], move || {
                store
                    .writer()
                    .unwrap()
                    .execute("UPDATE units SET ladder_level=2", [])
                    .unwrap();
            })
            .await
            .unwrap();
        assert_eq!(result["tree"], original["tree"]);
        assert_eq!(result["nodes"], original["nodes"]);
        assert_eq!(result["index"]["partial"], false);
        assert!(result.get("selection").is_none());
        let mut short = context.clone();
        short.config.limits.query_timeout_ms = 100;
        persist_fixture_config(&short);
        let result = short
            .map_checked(2000, &["absent".into()], || {
                std::thread::sleep(std::time::Duration::from_millis(150));
            })
            .await
            .unwrap();
        assert_eq!(result["index"]["reason"], "query_timeout");
        assert_eq!(result["tree"], "");
        assert_eq!(result["nodes"], json!([]));
        persist_fixture_config(&context);
        assert_eq!(
            context.map(2000, &[]).await.unwrap()["index"]["partial"],
            false
        );
    }

    #[tokio::test]
    async fn maps_catalogue_only_checks_completion_source_and_config_then_repairs() {
        for l0 in [false, true] {
            let (root, mut context) = fixture();
            let path = "src/retry.rs";
            fs::write(
                root.path().join(path),
                if l0 { "catalogue only" } else { "" },
            )
            .unwrap();
            if l0 {
                context.config.treatments.insert(
                    0,
                    crate::config::Treatment {
                        name: "catalogue-only".into(),
                        patterns: vec!["**/*.rs".into()],
                        chunker: None,
                        max_level: 0,
                        enrich: vec![],
                        never_send: false,
                    },
                );
            }
            persist_fixture_config(&context);
            assert!(
                super::super::repair::target(&context, path, sync::l1::Work::default()).unwrap()
            );
            assert!(
                !context
                    .store
                    .catalogue_source_checked(path, &|| Ok(()))
                    .unwrap()
                    .unwrap()
                    .has_occurrences
            );
            let focus = [path.into()];
            let initial = context.map(2000, &focus).await.unwrap();
            assert_eq!(initial["nodes"].as_array().unwrap().len(), 1, "{initial}");
            assert_eq!(initial["index"]["partial"], false);
            let head = context.store.get_meta("indexed_head").unwrap();
            context.config.search.lazy_resync = false;
            context.config.tokenize.split_subtokens = false;
            persist_fixture_config(&context);
            let refused = Context::load(root.path())
                .unwrap()
                .map(2000, &focus)
                .await
                .unwrap();
            assert_eq!(refused["nodes"], json!([]), "{refused}");
            assert_eq!(refused["index"]["freshness_refusals"]["l1_config_stale"], 1);
            context.config.search.lazy_resync = true;
            persist_fixture_config(&context);
            let repaired = context.map(2000, &focus).await.unwrap();
            assert_eq!(repaired["index"]["lazy_resync"]["attempted_files"], 1);
            assert_eq!(repaired["index"]["partial"], false, "{repaired}");
            assert_eq!(repaired["nodes"], initial["nodes"]);
            assert_eq!(repaired["tree"], initial["tree"]);
            assert_eq!(
                context.map(2000, &focus).await.unwrap()["index"]["lazy_resync"]["attempted_files"],
                0
            );
            context.config.search.lazy_resync = false;
            persist_fixture_config(&context);
            for sql in [
                "UPDATE files SET l1_pending=1",
                "UPDATE files SET l1_fingerprint=NULL",
                "UPDATE files SET l1_desired=zeroblob(32)",
                "UPDATE files SET l1_deferred_reason='storage_full'",
            ] {
                context.store.writer().unwrap().execute(sql, []).unwrap();
                let refused = context.map(2000, &focus).await.unwrap();
                assert_eq!(refused["nodes"], json!([]), "{sql}: {refused}");
                assert_eq!(refused["index"]["partial"], true);
                assert_eq!(
                    refused["index"]["freshness_refusals"]["l1_pending_or_unavailable"],
                    1
                );
                context.config.search.lazy_resync = true;
                persist_fixture_config(&context);
                assert!(
                    super::super::repair::target(&context, path, sync::l1::Work::default())
                        .unwrap()
                );
                context.config.search.lazy_resync = false;
                persist_fixture_config(&context);
            }
            fs::write(root.path().join(path), "pub fn newly_present() {}\n").unwrap();
            let changed = context.map(2000, &focus).await.unwrap();
            assert_eq!(changed["nodes"], json!([]));
            assert_eq!(changed["index"]["freshness_refusals"]["source_changed"], 1);
            context.config.search.lazy_resync = true;
            persist_fixture_config(&context);
            let repaired = context.map(2000, &focus).await.unwrap();
            assert_eq!(repaired["index"]["partial"], false, "{repaired}");
            assert_eq!(repaired["nodes"].as_array().unwrap().len(), 1);
            assert_eq!(
                repaired["nodes"][0]["symbols"]
                    .as_array()
                    .unwrap()
                    .is_empty(),
                l0
            );
            fs::remove_file(root.path().join(path)).unwrap();
            let deleted = context.map(2000, &focus).await.unwrap();
            assert_eq!(deleted["nodes"], json!([]));
            assert_eq!(deleted["tree"], "");
            assert_eq!(deleted["index"]["partial"], true);
            assert_eq!(context.store.get_meta("indexed_head").unwrap(), head);
        }
    }

    #[tokio::test]
    async fn maps_reloaded_config_refuses_old_l1_and_repairs_unchanged_sources() {
        for change in ["tokenize", "chunk", "repo", "protected", "cap"] {
            let (root, mut context) = fixture();
            fs::write(
                root.path().join("src/retry.rs"),
                "pub fn should_retry() {}\n",
            )
            .unwrap();
            complete_fixture_source(&context, "src/retry.rs");
            let bytes = fs::read(root.path().join("src/retry.rs")).unwrap();
            let head = context.store.get_meta("indexed_head").unwrap();
            match change {
                "tokenize" => context.config.tokenize.split_subtokens = false,
                "chunk" => context.config.chunk.target_tokens += 1,
                "repo" => context.config.index.repo_name = "new-name".into(),
                "protected" => context.config.treatments.insert(
                    0,
                    crate::config::Treatment {
                        name: "private-local".into(),
                        patterns: vec!["**/*.rs".into()],
                        chunker: None,
                        max_level: 1,
                        enrich: vec![],
                        never_send: true,
                    },
                ),
                "cap" => context.config.limits.max_units_per_file = 0,
                _ => unreachable!(),
            }
            context.config.search.lazy_resync = false;
            persist_fixture_config(&context);
            let reloaded = Context::load(root.path()).unwrap();
            let generation = reloaded.store.l1_generation().unwrap();
            let focus = ["src/retry.rs".into()];
            let refused = reloaded.map(2000, &focus).await.unwrap();
            assert_eq!(refused["nodes"], json!([]), "{change}: {refused}");
            assert_eq!(refused["tree"], "");
            assert_eq!(refused["index"]["partial"], true);
            assert_eq!(refused["index"]["freshness_refusals"]["l1_config_stale"], 1);
            assert_eq!(reloaded.store.l1_generation().unwrap(), generation);
            context.config.search.lazy_resync = true;
            persist_fixture_config(&context);
            let reloaded = Context::load(root.path()).unwrap();
            let result = reloaded.map(2000, &focus).await.unwrap();
            assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 1);
            if change == "cap" {
                assert_eq!(result["nodes"], json!([]));
                assert_eq!(result["tree"], "");
                assert_eq!(result["index"]["partial"], true);
            } else {
                assert_eq!(
                    result["nodes"].as_array().unwrap().len(),
                    1,
                    "{change}: {result}"
                );
                assert_eq!(result["nodes"][0]["symbols"][0]["name"], "should_retry");
                assert_eq!(result["index"]["partial"], false, "{change}: {result}");
                let generation = reloaded.store.l1_generation().unwrap();
                let retry = reloaded.map(2000, &focus).await.unwrap();
                assert_eq!(retry["nodes"], result["nodes"]);
                assert_eq!(retry["tree"], result["tree"]);
                assert_eq!(retry["index"]["lazy_resync"]["attempted_files"], 0);
                assert_eq!(reloaded.store.l1_generation().unwrap(), generation);
            }
            assert_eq!(fs::read(root.path().join("src/retry.rs")).unwrap(), bytes);
            assert_eq!(reloaded.store.get_meta("indexed_head").unwrap(), head);
        }
    }

    #[tokio::test]
    async fn maps_config_authority_applies_without_signatures_and_reselects_original_focus() {
        for without_symbols in [false, true] {
            let (root, mut context) = fixture();
            fs::write(
                root.path().join("src/retry.rs"),
                "pub fn should_retry() {}\n",
            )
            .unwrap();
            complete_fixture_source(&context, "src/retry.rs");
            if without_symbols {
                context
                    .store
                    .writer()
                    .unwrap()
                    .execute("DELETE FROM symbols", [])
                    .unwrap();
            }
            // Pick a real budget that admits the path but no signature.
            let budget = (1..100)
                .find(|budget| {
                    let mapped = super::super::map::map(&context.store, *budget, &[]).unwrap();
                    mapped.nodes.len() == 1 && mapped.nodes[0].symbols.is_empty()
                })
                .unwrap();
            context.config.tokenize.split_subtokens = false;
            context.config.search.lazy_resync = false;
            persist_fixture_config(&context);
            let reloaded = Context::load(root.path()).unwrap();
            let refused = reloaded.map(budget, &[]).await.unwrap();
            assert_eq!(refused["nodes"], json!([]));
            assert_eq!(refused["tree"], "");
            assert_eq!(refused["index"]["freshness_refusals"]["l1_config_stale"], 1);
            context.config.search.lazy_resync = true;
            persist_fixture_config(&context);
            let reloaded = Context::load(root.path()).unwrap();
            let repaired = reloaded
                .map(budget, &["src/retry.rs".into()])
                .await
                .unwrap();
            assert_eq!(repaired["nodes"][0]["path"], "src/retry.rs");
            assert_eq!(repaired["nodes"][0]["symbols"], json!([]));
            assert_eq!(repaired["index"]["lazy_resync"]["attempted_files"], 1);
            assert_eq!(repaired["index"]["partial"], false);
            assert!(
                tiktoken_rs::cl100k_base_singleton()
                    .encode_with_special_tokens(repaired["tree"].as_str().unwrap())
                    .len()
                    <= budget
            );
        }
        // A lexical focus must be rerun after a repair removes its old match.
        let (root, mut context) = fixture();
        fs::write(
            root.path().join("src/retry.rs"),
            "pub fn should_retry() {}\n",
        )
        .unwrap();
        complete_fixture_source(&context, "src/retry.rs");
        assert_eq!(
            context.map(2000, &["payment".into()]).await.unwrap()["nodes"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        context.config.tokenize.split_subtokens = false;
        persist_fixture_config(&context);
        let result = Context::load(root.path())
            .unwrap()
            .map(2000, &["payment".into()])
            .await
            .unwrap();
        assert_eq!(result["nodes"], json!([]));
        assert_eq!(result["tree"], "");
        assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 1);
    }

    #[tokio::test]
    async fn refs_reloaded_config_refuses_old_l1_and_repairs_unchanged_sources() {
        for change in ["tokenize", "chunk", "repo", "protected", "cap"] {
            let (root, mut context) = fixture();
            fs::write(
                root.path().join("src/retry.rs"),
                "pub fn should_retry() {}\npub fn worker() { should_retry(); }\n",
            )
            .unwrap();
            complete_fixture_source(&context, "src/retry.rs");
            context.store.writer().unwrap().execute("INSERT INTO edges(src_symbol,dst_name,kind,file_id,line) SELECT id,'should_retry','ref',file_id,2 FROM symbols", []).unwrap();
            assert_eq!(
                context.refs("should_retry", Some(1), false).await.unwrap()["returned_total"],
                1
            );
            let bytes = fs::read(root.path().join("src/retry.rs")).unwrap();
            let head = context.store.get_meta("indexed_head").unwrap();
            match change {
                "tokenize" => context.config.tokenize.split_subtokens = false,
                "chunk" => context.config.chunk.target_tokens += 1,
                "repo" => context.config.index.repo_name = "new-name".into(),
                "protected" => context.config.treatments.insert(
                    0,
                    crate::config::Treatment {
                        name: "private-local".into(),
                        patterns: vec!["**/*.rs".into()],
                        chunker: None,
                        max_level: 1,
                        enrich: vec![],
                        never_send: true,
                    },
                ),
                "cap" => context.config.limits.max_units_per_file = 0,
                _ => unreachable!(),
            }
            context.config.search.lazy_resync = false;
            persist_fixture_config(&context);
            let reloaded = Context::load(root.path()).unwrap();
            let generation = reloaded.store.l1_generation().unwrap();
            let refused = reloaded.refs("should_retry", Some(1), false).await.unwrap();
            assert_eq!(refused["groups"], json!([]), "{change}: {refused}");
            assert!(refused["total"].is_null());
            assert_eq!(refused["index"]["partial"], true);
            assert_eq!(refused["index"]["freshness_refusals"]["l1_config_stale"], 1);
            assert_eq!(reloaded.store.l1_generation().unwrap(), generation);
            context.config.search.lazy_resync = true;
            persist_fixture_config(&context);
            let reloaded = Context::load(root.path()).unwrap();
            let result = reloaded.refs("should_retry", Some(1), false).await.unwrap();
            assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 1);
            if change == "cap" {
                assert_eq!(result["groups"], json!([]));
                assert!(result["total"].is_null());
            } else {
                assert_eq!(
                    result["groups"].as_array().unwrap().len(),
                    1,
                    "{change}: {result}"
                );
                assert_eq!(result["index"]["partial"], false, "{change}: {result}");
                let generation = reloaded.store.l1_generation().unwrap();
                let retry = reloaded.refs("should_retry", Some(1), false).await.unwrap();
                assert_eq!(retry["groups"], result["groups"]);
                assert_eq!(retry["index"]["lazy_resync"]["attempted_files"], 0);
                assert_eq!(reloaded.store.l1_generation().unwrap(), generation);
            }
            assert_eq!(fs::read(root.path().join("src/retry.rs")).unwrap(), bytes);
            assert_eq!(reloaded.store.get_meta("indexed_head").unwrap(), head);
        }
    }

    #[tokio::test]
    async fn symbol_reloaded_config_refuses_old_l1_and_repairs_unchanged_sources() {
        for change in ["tokenize", "chunk", "repo", "protected", "cap"] {
            let (root, mut context) = fixture();
            fs::write(
                root.path().join("src/retry.rs"),
                "pub fn should_retry() {}\n",
            )
            .unwrap();
            complete_fixture_source(&context, "src/retry.rs");
            let bytes = fs::read(root.path().join("src/retry.rs")).unwrap();
            let head = context.store.get_meta("indexed_head").unwrap();
            match change {
                "tokenize" => context.config.tokenize.split_subtokens = false,
                "chunk" => context.config.chunk.target_tokens += 1,
                "repo" => context.config.index.repo_name = "new-name".into(),
                "protected" => context.config.treatments.insert(
                    0,
                    crate::config::Treatment {
                        name: "private-local".into(),
                        patterns: vec!["**/*.rs".into()],
                        chunker: None,
                        max_level: 1,
                        enrich: vec![],
                        never_send: true,
                    },
                ),
                "cap" => context.config.limits.max_units_per_file = 0,
                _ => unreachable!(),
            }
            context.config.search.lazy_resync = false;
            persist_fixture_config(&context);
            let reloaded = Context::load(root.path()).unwrap();
            let generation = reloaded.store.l1_generation().unwrap();
            assert_eq!(
                reloaded.symbol("should_retry").await.unwrap_err().code(),
                "source_unavailable",
                "{change}"
            );
            assert_eq!(reloaded.store.l1_generation().unwrap(), generation);
            context.config.search.lazy_resync = true;
            persist_fixture_config(&context);
            let reloaded = Context::load(root.path()).unwrap();
            let result = reloaded.symbol("should_retry").await;
            if change == "cap" {
                assert!(result.is_err());
            } else {
                assert_eq!(result.unwrap().as_array().unwrap().len(), 1, "{change}");
                let generation = reloaded.store.l1_generation().unwrap();
                assert!(reloaded.symbol("should_retry").await.is_ok());
                assert_eq!(reloaded.store.l1_generation().unwrap(), generation);
            }
            assert_eq!(fs::read(root.path().join("src/retry.rs")).unwrap(), bytes);
            assert_eq!(reloaded.store.get_meta("indexed_head").unwrap(), head);
        }
    }

    #[tokio::test]
    async fn open_reloaded_config_refuses_old_l1_and_repairs_unchanged_sources() {
        for change in ["tokenize", "chunk", "repo", "protected", "cap"] {
            let (root, mut context) = fixture();
            let bytes = fs::read(root.path().join("src/retry.rs")).unwrap();
            let head = context.store.get_meta("indexed_head").unwrap();
            match change {
                "tokenize" => context.config.tokenize.split_subtokens = false,
                "chunk" => context.config.chunk.target_tokens += 1,
                "repo" => context.config.index.repo_name = "new-name".into(),
                "protected" => context.config.treatments.insert(
                    0,
                    crate::config::Treatment {
                        name: "private-local".into(),
                        patterns: vec!["**/*.rs".into()],
                        chunker: None,
                        max_level: 1,
                        enrich: vec![],
                        never_send: true,
                    },
                ),
                "cap" => context.config.limits.max_units_per_file = 0,
                _ => unreachable!(),
            }
            context.config.search.lazy_resync = false;
            persist_fixture_config(&context);
            let reloaded = Context::load(root.path()).unwrap();
            let generation = reloaded.store.l1_generation().unwrap();
            assert_eq!(
                reloaded.open("src/retry.rs", 0).await.unwrap_err().code(),
                "source_unavailable",
                "{change}"
            );
            assert_eq!(reloaded.store.l1_generation().unwrap(), generation);
            context.config.search.lazy_resync = true;
            persist_fixture_config(&context);
            let reloaded = Context::load(root.path()).unwrap();
            let result = reloaded.open("src/retry.rs", 0).await;
            if change == "cap" {
                assert!(result.is_err());
            } else {
                assert!(result.is_ok(), "{change}: {result:?}");
                let generation = reloaded.store.l1_generation().unwrap();
                assert!(reloaded.open("src/retry.rs", 0).await.is_ok());
                assert_eq!(reloaded.store.l1_generation().unwrap(), generation);
            }
            assert_eq!(fs::read(root.path().join("src/retry.rs")).unwrap(), bytes);
            assert_eq!(reloaded.store.get_meta("indexed_head").unwrap(), head);
        }
    }

    #[tokio::test]
    async fn open_repairs_same_head_sources_and_preserves_locator_refusals() {
        for locator in ["path", "id", "cursor"] {
            let (root, mut context) = fixture();
            context.config.limits.max_open_bytes = "4 B".into();
            persist_fixture_config(&context);
            let original = context.open("src/retry.rs", 0).await.unwrap();
            let target = match locator {
                "id" => original["id"].as_str().unwrap(),
                "cursor" => original["continuation_id"].as_str().unwrap(),
                _ => "src/retry.rs",
            };
            let head = context.store.get_meta("indexed_head").unwrap();
            fs::write(root.path().join("src/retry.rs"), "pub fn repaired() {}\n").unwrap();
            let result = context.open(target, 0).await;
            if locator == "path" {
                assert_eq!(result.unwrap()["text"], "pub ");
            } else {
                assert!(result.is_err(), "old {locator} must be refused");
            }
            assert_eq!(context.store.get_meta("indexed_head").unwrap(), head);
            assert_eq!(
                context.open("src/retry.rs", 0).await.unwrap()["text"],
                "pub "
            );
            fs::remove_file(root.path().join("src/retry.rs")).unwrap();
            assert_eq!(
                context.open("src/retry.rs", 0).await.unwrap_err().code(),
                "source_unavailable"
            );
            fs::write(root.path().join("src/retry.rs"), "pub fn recovered() {}\n").unwrap();
            assert_eq!(
                context.open("src/retry.rs", 0).await.unwrap()["text"],
                "pub "
            );
        }
    }

    #[tokio::test]
    async fn open_config_repair_never_revives_previous_cursor() {
        let (root, mut context) = fixture();
        context.config.limits.max_open_bytes = "4 B".into();
        persist_fixture_config(&context);
        let original = context.open("src/retry.rs", 0).await.unwrap();
        let cursor = original["continuation_id"].as_str().unwrap();
        context.config.index.repo_name = "changed-context".into();
        persist_fixture_config(&context);
        let reloaded = Context::load(root.path()).unwrap();
        let generation = reloaded.store.l1_generation().unwrap();
        assert!(reloaded.open(cursor, 0).await.is_err());
        assert!(reloaded.store.l1_generation().unwrap() > generation);
        let generation = reloaded.store.l1_generation().unwrap();
        assert!(reloaded.open(cursor, 0).await.is_err());
        assert_eq!(reloaded.store.l1_generation().unwrap(), generation);
        let page = reloaded.open("src/retry.rs", 0).await.unwrap();
        assert!(
            page["breadcrumb"]
                .as_str()
                .unwrap()
                .starts_with("changed-context")
        );
        assert_eq!(page["text"], "retr");
    }

    #[tokio::test]
    async fn open_cursor_digest_refusal_does_not_repair_current_source() {
        let (_root, mut context) = fixture();
        context.config.limits.max_open_bytes = "4 B".into();
        persist_fixture_config(&context);
        let original = context.open("src/retry.rs", 0).await.unwrap();
        let cursor = original["continuation_id"].as_str().unwrap();
        use base64::Engine;
        let codec = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let mut decoded: Value =
            serde_json::from_slice(&codec.decode(cursor.strip_prefix("o1_").unwrap()).unwrap())
                .unwrap();
        decoded["digest"] = json!("obsolete-page-digest");
        let cursor = format!("o1_{}", codec.encode(serde_json::to_vec(&decoded).unwrap()));
        let generation = context.store.l1_generation().unwrap();
        assert_eq!(
            context.open(&cursor, 0).await.unwrap_err().code(),
            "source_unavailable"
        );
        assert_eq!(context.store.l1_generation().unwrap(), generation);
    }

    #[tokio::test]
    async fn open_small_repair_refuses_policy_and_size_then_retries() {
        for refusal in ["disabled", "large", "ignored", "pending"] {
            let (root, mut context) = fixture();
            let path = root.path().join("src/retry.rs");
            fs::write(&path, "pub fn changed() {}\n").unwrap();
            match refusal {
                "disabled" => context.config.search.lazy_resync = false,
                "large" => fs::write(&path, "x".repeat(256 * 1024)).unwrap(),
                "ignored" => fs::write(root.path().join(".ignore"), "src/retry.rs\n").unwrap(),
                "pending" => {
                    context
                        .store
                        .writer()
                        .unwrap()
                        .execute("UPDATE files SET l1_pending=1", [])
                        .unwrap();
                }
                _ => unreachable!(),
            }
            persist_fixture_config(&context);
            let result = context.open("src/retry.rs", 0).await;
            if refusal == "pending" {
                assert_eq!(result.unwrap()["text"], "pub fn changed() {}\n");
            } else {
                assert_eq!(
                    result.unwrap_err().code(),
                    "source_unavailable",
                    "{refusal}"
                );
            }
            context.config.search.lazy_resync = true;
            persist_fixture_config(&context);
            if refusal == "ignored" {
                fs::remove_file(root.path().join(".ignore")).unwrap();
            }
            fs::write(&path, "pub fn recovered() {}\n").unwrap();
            assert_eq!(
                context.open("src/retry.rs", 0).await.unwrap()["text"],
                "pub fn recovered() {}\n"
            );
        }
    }

    #[tokio::test]
    async fn open_final_fence_refuses_late_source_and_effective_settings_changes() {
        for mutation in [
            "edit",
            "delete",
            "outside_page",
            "root",
            "name",
            "limit",
            "invalid",
        ] {
            let (root, mut context) = fixture();
            context.config.limits.max_open_bytes = "4 B".into();
            persist_fixture_config(&context);
            let original = context.open("src/retry.rs", 0).await.unwrap();
            let path = root.path().join("src/retry.rs");
            let configuration = root.path().join(".deepkit/config.toml");
            let worker = context.clone();
            let store = Arc::clone(&context.store);
            let writes = store.writer().unwrap().total_changes();
            let result = context
                .open_checked(
                    "src/retry.rs",
                    0,
                    |_| {},
                    move || match mutation {
                        "edit" => fs::write(&path, "changed bytes").unwrap(),
                        "delete" => fs::remove_file(&path).unwrap(),
                        "outside_page" => fs::write(&path, "retry paymenX").unwrap(),
                        "invalid" => fs::write(&configuration, "[invalid").unwrap(),
                        _ => {
                            let mut changed = worker;
                            match mutation {
                                "root" => changed.config.index.root = "src".into(),
                                "name" => changed.config.index.repo_name = "changed".into(),
                                "limit" => changed.config.limits.max_open_bytes = "1 B".into(),
                                _ => unreachable!(),
                            }
                            persist_fixture_config(&changed);
                        }
                    },
                )
                .await;
            assert_eq!(
                result.unwrap_err().code(),
                "source_unavailable",
                "{mutation}"
            );
            assert_eq!(
                store.writer().unwrap().total_changes(),
                writes,
                "{mutation}"
            );
            fs::write(root.path().join("src/retry.rs"), "retry payment").unwrap();
            persist_fixture_config(&context);
            let retry = context.open("src/retry.rs", 0).await.unwrap();
            for key in ["id", "text", "lines", "continuation_id", "byte_range"] {
                assert_eq!(retry[key], original[key], "{mutation}/{key}");
            }
            assert!(retry.get("occurrence").is_none());
            assert!(retry.get("source").is_none());
        }
    }

    #[tokio::test]
    async fn open_final_fence_refuses_same_generation_publication_and_allows_ladder_progress() {
        for sql in [
            "UPDATE units SET header='replacement'",
            "UPDATE units SET line_end=line_end+1",
            "UPDATE units SET scope_path='replacement'",
            "UPDATE unit_contents SET text='retry paymenX'",
            "UPDATE files SET l1_pending=1",
            "UPDATE files SET l1_fingerprint=X'01',l1_desired=X'01'",
            "DELETE FROM units",
            "UPDATE units SET ladder_level=2",
        ] {
            let (_root, mut context) = fixture();
            context.config.limits.max_open_bytes = "4 B".into();
            persist_fixture_config(&context);
            let original = context.open("src/retry.rs", 0).await.unwrap();
            let generation = context.store.l1_generation().unwrap();
            let store = Arc::clone(&context.store);
            let (sent, writes) = tokio::sync::oneshot::channel();
            let result = context
                .open_checked(
                    "src/retry.rs",
                    0,
                    |_| {},
                    move || {
                        let writer = store.writer().unwrap();
                        writer.execute(sql, []).unwrap();
                        // Make a distinct completed publication while retaining the scan generation.
                        if sql.contains("l1_fingerprint") {
                            writer.execute("UPDATE files SET l1_pending=0", []).unwrap();
                        }
                        sent.send(writer.total_changes()).unwrap();
                    },
                )
                .await;
            assert_eq!(context.store.l1_generation().unwrap(), generation);
            let writes = writes.await.unwrap();
            assert_eq!(context.store.writer().unwrap().total_changes(), writes);
            if sql.contains("ladder_level") {
                let result = result.unwrap();
                assert_eq!(result["text"], original["text"]);
                assert_eq!(result["continuation_id"], original["continuation_id"]);
            } else {
                assert_eq!(result.unwrap_err().code(), "source_unavailable", "{sql}");
                // Restore through the real per-file Processor, including deleted occurrences.
                context
                    .store
                    .writer()
                    .unwrap()
                    .execute("UPDATE files SET l1_pending=1", [])
                    .unwrap();
                assert!(
                    crate::search::repair::target(&context, "src/retry.rs", Default::default())
                        .unwrap()
                );
                let retry = context.open("src/retry.rs", 0).await.unwrap();
                assert_eq!(retry["text"], "retr", "{sql}");
                assert_eq!(retry["lines"], serde_json::json!([1, 1]));
            }
        }
    }

    #[tokio::test]
    async fn open_final_fence_retains_source_identity_for_identical_pages_and_cursors() {
        for locator in ["path", "id", "cursor"] {
            let (root, mut context) = fixture();
            context.config.limits.max_open_bytes = "4 B".into();
            persist_fixture_config(&context);
            let original = context.open("src/retry.rs", 0).await.unwrap();
            let target = match locator {
                "id" => original["id"].as_str().unwrap(),
                "cursor" => original["continuation_id"].as_str().unwrap(),
                _ => "src/retry.rs",
            };
            let expected = context.open(target, 0).await.unwrap();
            let worker = context.clone();
            let generation = context.store.l1_generation().unwrap();
            let result = context
                .open_checked(
                    target,
                    0,
                    |_| {},
                    move || {
                        fs::write(worker.root.join("src/retry.rs"), "retry paymenX").unwrap();
                        complete_fixture_source(&worker, "src/retry.rs");
                    },
                )
                .await;
            assert_eq!(
                result.unwrap_err().code(),
                "source_unavailable",
                "{locator}"
            );
            assert_eq!(context.store.l1_generation().unwrap(), generation);
            // Both the first and second page still contain identical text. The
            // original source digest must nevertheless refuse the prepared page.
            let changed = context.open("src/retry.rs", 0).await.unwrap();
            assert_eq!(changed["text"], original["text"]);
            assert_ne!(changed["continuation_id"], original["continuation_id"]);
            fs::write(root.path().join("src/retry.rs"), "retry payment").unwrap();
            complete_fixture_source(&context, "src/retry.rs");
            let retry = context.open(target, 0).await.unwrap();
            for key in ["id", "text", "lines", "continuation_id", "byte_range"] {
                assert_eq!(retry[key], expected[key], "{locator}/{key}");
            }
        }
    }

    #[tokio::test]
    async fn open_final_fence_obeys_original_deadline_and_drop_with_retry() {
        for timeout in [false, true] {
            let (_root, mut context) = fixture();
            context.config.limits.query_timeout_ms = if timeout { 100 } else { 2000 };
            persist_fixture_config(&context);
            let (entered, ready) = tokio::sync::oneshot::channel();
            let (release, released) = std::sync::mpsc::channel();
            let (finished, done) = tokio::sync::oneshot::channel();
            let worker = context.clone();
            let writes = context.store.writer().unwrap().total_changes();
            let task = tokio::spawn(async move {
                worker
                    .open_checked(
                        "src/retry.rs",
                        0,
                        |_| {},
                        move || {
                            entered.send(()).unwrap();
                            released
                                .recv_timeout(std::time::Duration::from_secs(2))
                                .unwrap();
                            finished.send(()).unwrap();
                        },
                    )
                    .await
            });
            ready.await.unwrap();
            if timeout {
                assert_eq!(task.await.unwrap().unwrap_err().code(), "timeout");
            } else {
                task.abort();
                assert!(task.await.unwrap_err().is_cancelled());
            }
            release.send(()).unwrap();
            done.await.unwrap();
            context.config.limits.query_timeout_ms = 2000;
            persist_fixture_config(&context);
            let retry = context.open("src/retry.rs", 0).await.unwrap();
            assert_eq!(retry["text"], "retry payment");
            assert_eq!(context.store.writer().unwrap().total_changes(), writes);
        }
    }

    #[tokio::test]
    async fn open_rechecks_head_and_generation_after_source_and_retries() {
        for mutation in ["head", "generation"] {
            let (root, context) = fixture();
            let path = root.path().to_owned();
            let store = Arc::clone(&context.store);
            let result = context
                .open_checked(
                    "src/retry.rs:1-1",
                    0,
                    |_| {},
                    move || {
                        if mutation == "head" {
                            git(&path, &["init", "--quiet"]);
                            git(
                                &path,
                                &[
                                    "-c",
                                    "user.name=Open Test",
                                    "-c",
                                    "user.email=open@example.invalid",
                                    "-c",
                                    "commit.gpgsign=false",
                                    "commit",
                                    "--quiet",
                                    "--allow-empty",
                                    "-m",
                                    "late",
                                ],
                            );
                        } else {
                            store
                                .set_meta(
                                    "l1_scan_generation",
                                    &(store.l1_generation().unwrap() + 1).to_string(),
                                )
                                .unwrap();
                        }
                    },
                )
                .await;
            assert_eq!(
                result.unwrap_err().code(),
                "source_unavailable",
                "{mutation}"
            );
            let result = context.open("src/retry.rs:1-1", 0).await.unwrap();
            assert_eq!(result["text"], "retry payment");
            assert_eq!(
                result["index"]["branch_resync"]["state"],
                if mutation == "head" {
                    "completed"
                } else {
                    "not_needed"
                }
            );
        }
    }

    #[tokio::test]
    async fn symbol_final_fence_refuses_source_and_settings_changes_without_repair() {
        for mutation in ["edit", "delete", "root", "name", "limit", "invalid"] {
            let (root, context) = fixture();
            let path = root.path().join("src/retry.rs");
            let original = context.symbol("should_retry").await.unwrap();
            let worker = context.clone();
            let writes = context.store.writer().unwrap().total_changes();
            let result = context
                .symbol_checked("should_retry", move || match mutation {
                    "edit" => fs::write(&path, "changed bytes").unwrap(),
                    "delete" => fs::remove_file(&path).unwrap(),
                    "invalid" => {
                        fs::write(worker.config_root.join(".deepkit/config.toml"), "[invalid")
                            .unwrap()
                    }
                    _ => {
                        let mut changed = worker;
                        match mutation {
                            "root" => changed.config.index.root = "src".into(),
                            "name" => changed.config.index.repo_name = "changed".into(),
                            "limit" => changed.config.limits.max_file_bytes = "1 B".into(),
                            _ => unreachable!(),
                        }
                        persist_fixture_config(&changed);
                    }
                })
                .await;
            assert_eq!(
                result.unwrap_err().code(),
                "source_unavailable",
                "{mutation}"
            );
            assert_eq!(
                context.store.writer().unwrap().total_changes(),
                writes,
                "{mutation}"
            );
            fs::write(root.path().join("src/retry.rs"), "retry payment").unwrap();
            persist_fixture_config(&context);
            assert_eq!(
                context.symbol("should_retry").await.unwrap(),
                original,
                "{mutation}"
            );
            assert!(original[0].get("evidence").is_none());
            assert!(original[0].get("selected").is_none());
        }
    }

    #[tokio::test]
    async fn symbol_budget_excludes_discarded_suffix_from_source_fence() {
        let (root, mut context) = fixture();
        add(
            &context.store,
            "src/discarded.rs",
            "should_retry",
            "retry payment",
            None,
        )
        .unwrap();
        fs::write(root.path().join("src/discarded.rs"), "retry payment").unwrap();
        complete_fixture_source(&context, "src/discarded.rs");
        let oversized_signature = format!("fn should_retry({})", "argument: usize, ".repeat(600));
        context
            .store
            .writer()
            .unwrap()
            .execute(
                "UPDATE symbols SET signature=?1,pagerank=0 WHERE file_id=(SELECT id FROM files WHERE path='src/discarded.rs')",
                [oversized_signature],
            )
            .unwrap();
        context
            .store
            .writer()
            .unwrap()
            .execute(
                "UPDATE symbols SET pagerank=1 WHERE file_id=(SELECT id FROM files WHERE path='src/retry.rs')",
                [],
            )
            .unwrap();
        context.config.limits.max_result_tokens = 50_000;
        persist_fixture_config(&context);
        let full = context.symbol("should_retry").await.unwrap();
        assert_eq!(full.as_array().unwrap().len(), 2);
        context.config.limits.max_result_tokens =
            crate::treat::count_tokens(&serde_json::to_string(&json!([full[0].clone()])).unwrap())
                as u64;
        persist_fixture_config(&context);
        let discarded = root.path().join("src/discarded.rs");
        let capped = context
            .symbol_checked("should_retry", move || {
                fs::remove_file(discarded).unwrap();
            })
            .await
            .unwrap();
        assert_eq!(capped.as_array().unwrap().len(), 1);
        assert_eq!(capped[0]["path"], "src/retry.rs");
        assert_eq!(
            context.query_latencies.cap_snapshot()["events"]["max_result_tokens"]["count"],
            1
        );
    }

    #[tokio::test]
    async fn symbol_final_fence_rechecks_empty_selection_and_occurrence_authority() {
        for (target, mutation, sql, restore) in [
            (
                "absent",
                "selection",
                "UPDATE symbols SET name='absent',qualified='absent'",
                "UPDATE symbols SET name='should_retry',qualified='should_retry'",
            ),
            ("absent", "settings", "", ""),
            (
                "should_retry",
                "association",
                "UPDATE symbols SET unit_id=NULL",
                "UPDATE symbols SET unit_id=(SELECT min(id) FROM units)",
            ),
            (
                "should_retry",
                "signature",
                "UPDATE symbols SET signature='changed'",
                "UPDATE symbols SET signature='fn should_retry()'",
            ),
            (
                "should_retry",
                "occurrence",
                "UPDATE units SET header=header || 'changed'",
                "UPDATE units SET header=substr(header,1,length(header)-7)",
            ),
            (
                "should_retry",
                "scope",
                "UPDATE units SET scope_path='changed'",
                "UPDATE units SET scope_path=NULL",
            ),
        ] {
            let (_root, context) = fixture();
            let original = context.symbol(target).await.unwrap();
            let generation = context.store.l1_generation().unwrap();
            let worker = context.clone();
            let result = context
                .symbol_checked(target, move || {
                    if mutation == "settings" {
                        let mut changed = worker;
                        changed.config.index.repo_name = "changed".into();
                        persist_fixture_config(&changed);
                    } else {
                        worker.store.writer().unwrap().execute(sql, []).unwrap();
                    }
                })
                .await;
            assert_eq!(
                result.unwrap_err().code(),
                "source_unavailable",
                "{mutation}"
            );
            assert_eq!(context.store.l1_generation().unwrap(), generation);
            if !restore.is_empty() {
                context
                    .store
                    .writer()
                    .unwrap()
                    .execute(restore, [])
                    .unwrap();
            }
            persist_fixture_config(&context);
            let retry = context.symbol(target).await.unwrap();
            assert_eq!(
                retry.as_array().unwrap().len(),
                original.as_array().unwrap().len(),
                "{mutation}"
            );
        }
    }

    #[tokio::test]
    async fn symbol_final_fence_allows_ladder_advance_and_retains_deadline() {
        let (_root, context) = fixture();
        let original = context.symbol("should_retry").await.unwrap();
        let store = Arc::clone(&context.store);
        let result = context
            .symbol_checked("should_retry", move || {
                store
                    .writer()
                    .unwrap()
                    .execute("UPDATE units SET ladder_level=2", [])
                    .unwrap();
            })
            .await
            .unwrap();
        assert_eq!(result, original);
        let mut short = context.clone();
        short.config.limits.query_timeout_ms = 100;
        persist_fixture_config(&short);
        let result = short
            .symbol_checked("absent", || {
                std::thread::sleep(std::time::Duration::from_millis(150));
            })
            .await;
        assert_eq!(result.unwrap_err().code(), "timeout");
        persist_fixture_config(&context);
        assert_eq!(context.symbol("should_retry").await.unwrap(), original);
    }

    #[tokio::test]
    async fn symbols_recheck_head_and_generation_even_after_empty_lookup() {
        for target in ["should_retry", "absent"] {
            for mutation in ["head", "generation"] {
                let (root, context) = fixture();
                fs::write(
                    root.path().join("src/retry.rs"),
                    "pub fn should_retry() {}\n",
                )
                .unwrap();
                complete_fixture_source(&context, "src/retry.rs");
                let store = Arc::clone(&context.store);
                let path = root.path().to_owned();
                let result = context
                    .symbol_checked(target, move || {
                        if mutation == "head" {
                            git(&path, &["init", "--quiet"]);
                            git(
                                &path,
                                &[
                                    "-c",
                                    "user.name=Symbol Test",
                                    "-c",
                                    "user.email=symbol@example.invalid",
                                    "-c",
                                    "commit.gpgsign=false",
                                    "commit",
                                    "--quiet",
                                    "--allow-empty",
                                    "-m",
                                    "late",
                                ],
                            );
                        } else {
                            store
                                .set_meta(
                                    "l1_scan_generation",
                                    &(store.l1_generation().unwrap() + 1).to_string(),
                                )
                                .unwrap();
                        }
                    })
                    .await;
                assert_eq!(
                    result.unwrap_err().code(),
                    "source_unavailable",
                    "{target}/{mutation}"
                );
                let retry = context.symbol(target).await.unwrap();
                assert_eq!(
                    retry.as_array().unwrap().len(),
                    usize::from(target == "should_retry")
                );
            }
        }
    }

    #[tokio::test]
    async fn refs_and_maps_recheck_late_head_generation_and_pending_even_when_empty() {
        for tool in ["refs", "map"] {
            for empty in [false, true] {
                for mutation in ["head", "generation", "pending"] {
                    let (root, context) = fixture();
                    context.store.writer().unwrap().execute("INSERT INTO edges(src_symbol,dst_name,kind,file_id,line) SELECT id,'payment','ref',file_id,1 FROM symbols", []).unwrap();
                    let path = root.path().to_owned();
                    let store = Arc::clone(&context.store);
                    let mutate = move || match mutation {
                        "head" => {
                            git(&path, &["init", "--quiet"]);
                            git(
                                &path,
                                &[
                                    "-c",
                                    "user.name=Navigation Test",
                                    "-c",
                                    "user.email=navigation@example.invalid",
                                    "-c",
                                    "commit.gpgsign=false",
                                    "commit",
                                    "--quiet",
                                    "--allow-empty",
                                    "-m",
                                    "late",
                                ],
                            );
                        }
                        "generation" => store
                            .set_meta(
                                "l1_scan_generation",
                                &(store.l1_generation().unwrap() + 1).to_string(),
                            )
                            .unwrap(),
                        "pending" => {
                            store
                                .writer()
                                .unwrap()
                                .execute("UPDATE files SET l1_pending=1", [])
                                .unwrap();
                        }
                        _ => unreachable!(),
                    };
                    let response = if tool == "refs" {
                        context
                            .refs_checked(
                                if empty { "absent" } else { "payment" },
                                None,
                                false,
                                mutate,
                            )
                            .await
                            .unwrap()
                    } else {
                        context
                            .map_checked(
                                2000,
                                &if empty { vec!["absent".into()] } else { vec![] },
                                mutate,
                            )
                            .await
                            .unwrap()
                    };
                    assert_eq!(
                        response["index"]["partial"], true,
                        "{tool}/{empty}/{mutation}: {response}"
                    );
                    assert_eq!(response["index"]["degraded"], true);
                    if tool == "refs" {
                        assert!(response["total"].is_null());
                    }
                    if mutation != "pending" {
                        assert_eq!(
                            response[if tool == "refs" { "groups" } else { "nodes" }],
                            json!([])
                        );
                        if tool == "map" {
                            assert_eq!(response["tree"], "");
                        }
                    }
                    // Clear pending state or reconcile the new HEAD with fresh Work.
                    store_pending_clear(&context);
                    let retry = if tool == "refs" {
                        context.refs("absent", None, false).await.unwrap()
                    } else {
                        context.map(2000, &["absent".into()]).await.unwrap()
                    };
                    assert_eq!(retry["index"]["partial"], false, "{retry}");
                }
            }
        }
    }

    fn store_pending_clear(context: &Context) {
        context
            .store
            .writer()
            .unwrap()
            .execute("UPDATE files SET l1_pending=0", [])
            .unwrap();
    }

    #[tokio::test]
    async fn an_existing_context_observes_head_changes_before_each_navigation_call() {
        let (root, mut context) = fixture();
        // This test exercises observation with repair explicitly disabled.
        context.config.search.lazy_resync = false;
        fs::write(
            root.path().join(".deepkit/config.local.toml"),
            "[search]\nlazy_resync=false\n",
        )
        .unwrap();
        git(root.path(), &["init", "--quiet"]);
        let commit = [
            "-c",
            "user.name=Service Test",
            "-c",
            "user.email=service@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "first",
        ];
        git(root.path(), &commit);
        let first = git(root.path(), &["rev-parse", "HEAD"]);
        context.store.set_meta("indexed_head", &first).unwrap();
        assert_eq!(context.status().unwrap()["head"], first);
        let mut next = commit;
        next[10] = "second";
        git(root.path(), &next);
        let second = git(root.path(), &["rev-parse", "HEAD"]);
        assert_ne!(first, second);
        // Query freshness observes HEAD only. Even a truncated Git index must
        // not enter the full-index loader while serving the existing catalogue.
        fs::write(root.path().join(".git/index"), b"invalid index").unwrap();
        assert_eq!(
            context.search(request(Mode::Lexical), false).await.unwrap()["index"]["head"],
            second
        );
        assert_eq!(
            context
                .open("src/retry.rs:1-1", 0)
                .await
                .unwrap_err()
                .code(),
            "source_unavailable"
        );
        assert_eq!(
            context.map(2000, &[]).await.unwrap()["index"]["head"],
            second
        );
        assert_eq!(
            context.refs("should_retry", None, false).await.unwrap()["index"]["head"],
            second
        );
        assert_eq!(
            context.symbol("should_retry").await.unwrap_err().code(),
            "source_unavailable"
        );
    }
    #[test]
    fn background_l1_defers_branch_and_target_repairs_without_writes() {
        let (root, context) = fixture();
        context.background_l1.store(true, Ordering::Release);
        let mut inspection = Inspection {
            head: Some("new-head".into()),
            indexed_head: Some("old-head".into()),
            pending: 1,
        };
        let writes = context.store.writer().unwrap().total_changes();
        let branch = context
            .reconcile_branch(&mut inspection, Default::default())
            .unwrap();
        assert_eq!(
            branch,
            json!({"state":"incomplete", "reason":"background_l1"})
        );
        assert_eq!(inspection.head.as_deref(), Some("new-head"));
        fs::write(root.path().join("src/retry.rs"), "changed source").unwrap();
        assert!(
            !super::super::repair::target(&context, "src/retry.rs", Default::default()).unwrap()
        );
        assert_eq!(context.store.writer().unwrap().total_changes(), writes);
        inspection.indexed_head = inspection.head.clone();
        assert_eq!(
            context
                .reconcile_branch(&mut inspection, Default::default())
                .unwrap(),
            json!({"state":"not_needed"})
        );
    }

    #[test]
    fn load_with_config_uses_captured_settings_without_reopening_configuration() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join(".deepkit")).unwrap();
        fs::write(root.path().join(".deepkit/config.toml"), "invalid TOML [").unwrap();
        let mut settings = Config::default();
        settings.search.default_k = 7;
        let context = Context::load_with_config(root.path(), settings.clone()).unwrap();
        assert_eq!(context.config, settings);
        assert!(!context.background_l1.load(Ordering::Acquire));
        assert!(Context::load(root.path()).is_err());
    }
}
