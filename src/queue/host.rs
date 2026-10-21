//! Shared index/watch/serve embedding host orchestration; v1 §8.2–8.3.
pub(crate) mod group;
mod retirement;

use super::{Error, destination, http_embedding, pool, priority::Recency, usage};
#[cfg(test)]
use crate::store::vectors::embedding_table_slug as table_slug;
use crate::{
    config::Config,
    store::{Store, UnitKind, vectors::VectorTableSpec},
};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

/// Finite execution bounds and an explicit provider-wide daily budget.
#[derive(Clone, Copy)]
pub(crate) struct Bounds {
    pub(crate) budget: usage::Budget,
    pub(crate) timeout: Duration,
    pub(crate) heartbeat: Duration,
    pub(crate) max_schedule: usize,
    pub(crate) max_jobs: usize,
}
impl Bounds {
    /// Selects the canonical provider budget before preparing any host resources.
    pub(crate) fn for_provider(
        settings: &crate::config::Queue,
        provider: &str,
    ) -> Result<Self, Error> {
        let budget = settings
            .budget(provider)
            .map_err(|_| Error::Invalid("embedding endpoint budgets"))?;
        Self::from_config(
            settings,
            usage::Budget {
                calls: budget.and_then(|value| value.daily_calls),
                tokens: budget.and_then(|value| value.daily_tokens),
            },
        )
    }

    /// Maps validated queue settings while preserving unlimited and zero budgets.
    /// Endpoint budget selection remains the caller's responsibility.
    pub(crate) fn from_config(
        settings: &crate::config::Queue,
        budget: usage::Budget,
    ) -> Result<Self, Error> {
        let execution = settings
            .execution()
            .map_err(|_| Error::Invalid("embedding queue settings"))?;
        let bounds = Self {
            budget,
            timeout: execution.timeout,
            heartbeat: execution.heartbeat,
            max_schedule: execution.max_schedule,
            max_jobs: execution.max_jobs,
        };
        bounds.validate(0)?;
        Ok(bounds)
    }

    /// Validates finite worker settings before any provider/table preparation.
    pub(crate) fn validate(&self, now: i64) -> Result<(), Error> {
        if self.max_schedule == 0
            || self.max_jobs == 0
            || self.timeout.is_zero()
            || self.heartbeat.is_zero()
            || self.heartbeat >= Duration::from_secs(120)
            || now < 0
            || self.budget.calls.is_some_and(|value| value < 0)
            || self.budget.tokens.is_some_and(|value| value < 0)
        {
            return Err(Error::Invalid("embedding host bounds"));
        }
        Ok(())
    }
}

/// Immutable inputs for one selected destination; configuration remains rechecked
/// by every job's source and publication resolvers.
pub(crate) struct Settings {
    pub(crate) root: PathBuf,
    pub(crate) config_root: PathBuf,
    pub(crate) config: Config,
    pub(crate) kind: UnitKind,
    pub(crate) local_cache: PathBuf,
    pub(crate) bounds: Bounds,
}
/// File occurrence priority supplied by the authoritative catalogue scheduler.
#[derive(Clone, Copy)]
pub(crate) struct Target {
    pub(crate) unit: i64,
    pub(crate) rank_pct: f64,
    pub(crate) recency: Recency,
}
/// Scheduling progress counts accepted existing jobs as well as newly queued jobs.
#[derive(Debug, Default)]
pub(crate) struct Report {
    pub(crate) visited: usize,
    pub(crate) scheduled_or_existing: usize,
    pub(crate) skipped: usize,
    pub(crate) execution: pool::Report,
    pub(crate) retirement: retirement::Report,
    pub(crate) transition: Option<crate::store::transitions::State>,
    pub(crate) transition_error: Option<Error>,
}
/// Why a finite drain stopped; Empty describes observed queued/leased work,
/// not a claim that dead jobs succeeded or that future producers cannot enqueue.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum DrainStop {
    Empty,
    Waiting,
    PassLimit,
    Failed,
}
/// Completed pass reports and the most recent pending-work observation.
#[derive(Debug)]
pub(crate) struct DrainReport {
    pub(crate) passes: Vec<Report>,
    pub(crate) pending: Option<i64>,
    pub(crate) stopped: DrainStop,
}
/// Retains completed-pass evidence when a later drain operation fails.
#[derive(Debug, thiserror::Error)]
#[error("embedding drain stopped: {source}")]
pub(crate) struct DrainFailure {
    pub(crate) progress: DrainReport,
    #[source]
    pub(crate) source: Error,
}

/// A failed pass retains progress from committed scheduling transactions.
#[derive(Debug, thiserror::Error)]
#[error(
    "embedding host stopped after {visited} visited targets ({scheduled_or_existing} scheduled or existing): {source}"
)]
pub(crate) struct Failure {
    pub(crate) visited: usize,
    pub(crate) scheduled_or_existing: usize,
    #[source]
    pub(crate) source: Error,
}

/// One selected provider/destination. Preparation preserves old coverage; passes
/// retire obsolete empty tables while protecting every configured destination.
pub(crate) struct Host {
    store: Arc<Store>,
    provider: String,
    factory: Arc<http_embedding::Factory>,
    bounds: Bounds,
    pub(crate) table_slug: String,
    pub(crate) binding_state: crate::store::embedding_bindings::State,
    pub(crate) transition: crate::store::transitions::State,
    retirement: retirement::Policy,
    refresh_context: bool,
}

struct CancelOnDrop(Arc<AtomicBool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

impl Host {
    /// Prepares the selected table/provider without enqueueing or contacting an
    /// embedding endpoint. Local weights must already exist in the supplied cache.
    pub(crate) async fn prepare(
        store: Arc<Store>,
        settings: Settings,
        now: i64,
    ) -> Result<Self, Error> {
        let Settings {
            root,
            config_root,
            config,
            kind,
            local_cache,
            bounds,
        } = settings;
        bounds.validate(now)?;
        let selected = destination::resolve(&config, kind)?;
        selected.limits(bounds.budget)?;
        if config
            .limits
            .wal_max_bytes()
            .map_err(|_| Error::Invalid("embedding WAL cap"))?
            .0
            != store.wal_max_bytes
        {
            return Err(Error::Invalid(
                "embedding WAL cap differs from open store; reopen with current limits",
            ));
        }
        if config
            .limits
            .db_max_bytes()
            .map_err(|_| Error::Invalid("embedding database cap"))?
            .0
            != store.db_max_bytes
        {
            return Err(Error::Invalid(
                "embedding database cap differs from open store; reopen with current limits",
            ));
        }
        let profile = if selected.identity.provider == "embed:local" {
            "local"
        } else if selected.identity.provider.starts_with("embed:remote:") {
            "remote"
        } else {
            "ollama"
        };
        let binding = selected.binding(&config);
        let model = selected.identity.model;
        let provider = selected.identity.provider;
        let dims = usize::try_from(selected.identity.dims)
            .map_err(|_| Error::Invalid("embedding dimensions overflow"))?;
        let setup_store = Arc::clone(&store);
        let route_config = config.clone();
        let route_root = config_root.clone();
        let (slug, binding_state, transition) = tokio::task::spawn_blocking(move || {
            let bindings = UnitKind::ALL
                .into_iter()
                .map(|kind| {
                    destination::resolve(&route_config, kind)
                        .map(|resolved| (kind.as_str().to_owned(), resolved.binding(&route_config)))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let transition = setup_store.prepare_transition(|tx| {
                let current = crate::config::load(&route_root, None)
                    .map_err(|_| Error::Invalid("embedding routing configuration reload"))?
                    .config;
                if current != route_config {
                    return Err(Error::Invalid("embedding routing configuration changed"));
                }
                bindings
                    .iter()
                    .map(|(kind, binding)| {
                        crate::store::embedding_bindings::select(tx, binding)
                            .map(|table| (kind.clone(), table))
                            .map_err(Error::from)
                    })
                    .collect::<Result<crate::store::transitions::Routes, Error>>()
            })?;
            let table = transition
                .desired
                .get(kind.as_str())
                .cloned()
                .ok_or(Error::Invalid("missing embedding kind route"))?;
            // Existing matching active tables require no maintenance reservation.
            let current = setup_store
                .vector_tables()?
                .into_iter()
                .find(|entry| entry.slug == table);
            if !current.is_some_and(|entry| {
                entry.active
                    && entry.model == model
                    && entry.profile == profile
                    && entry.dims == dims
            }) {
                setup_store.ensure_vector_table(VectorTableSpec {
                    slug: &table,
                    model: &model,
                    profile,
                    dims,
                    created_at: now,
                    active: true,
                })?;
            }
            let state = setup_store.bind_embedding_table(&table, &binding)?;
            Ok::<_, Error>((table, state, transition))
        })
        .await
        .map_err(|_| Error::WorkerTask)??;
        let retirement = retirement::Policy {
            root: root.clone(),
            config_root: config_root.clone(),
            config: config.clone(),
        };
        let settings = http_embedding::Settings {
            root,
            config_root,
            config,
            kind,
            table_slug: slug.clone(),
            budget: bounds.budget,
            timeout: bounds.timeout,
        };
        let factory = if profile == "local" {
            http_embedding::Factory::new_local(Arc::clone(&store), settings, local_cache).await?
        } else {
            http_embedding::Factory::new(Arc::clone(&store), settings)?
        };
        Ok(Self {
            store,
            provider,
            factory,
            bounds,
            table_slug: slug,
            binding_state,
            transition,
            retirement,
            refresh_context: false,
        })
    }

    /// Enables an explicit context refresh for this finite host lifetime only.
    /// It does not mutate persisted configuration or bypass worker admission.
    pub(crate) fn with_context_refresh(mut self, enabled: bool) -> Self {
        self.refresh_context = enabled;
        self
    }

    async fn pending(&self) -> Result<i64, Error> {
        let store = Arc::clone(&self.store);
        let provider = self.provider.clone();
        tokio::task::spawn_blocking(move || {
            store.read_snapshot(|tx| tx.query_row("SELECT count(*) FROM jobs WHERE provider=?1 AND state IN ('queued','leased','failed')",[provider],|r|r.get(0)))?.map_err(Error::from)
        }).await.map_err(|_|Error::WorkerTask)?
    }

    /// Drains ready work through at most max_passes finite provider passes. It
    /// never sleeps through delays or treats deferred/leased work as completed.
    /// Errors retain prior completed-pass counts; a failed in-flight pass may
    /// have committed additional work not represented by those counts.
    pub(crate) async fn drain<C>(
        &self,
        max_passes: usize,
        clock: C,
    ) -> Result<DrainReport, DrainFailure>
    where
        C: Fn() -> i64 + Send + Sync + 'static,
    {
        let mut progress = DrainReport {
            passes: Vec::new(),
            pending: None,
            stopped: DrainStop::PassLimit,
        };
        let fail = |mut progress: DrainReport, source| {
            progress.stopped = DrainStop::Failed;
            DrainFailure { progress, source }
        };
        if max_passes == 0 || max_passes.checked_mul(self.bounds.max_jobs).is_none() {
            return Err(fail(progress, Error::Invalid("embedding drain pass bound")));
        }
        let clock = Arc::new(clock);
        for _ in 0..max_passes {
            progress.pending = Some(match self.pending().await {
                Ok(n) => n,
                Err(error) => return Err(fail(progress, error)),
            });
            if progress.pending == Some(0) {
                progress.stopped = DrainStop::Empty;
                return Ok(progress);
            }
            let pass_clock = Arc::clone(&clock);
            let report = match self.run(Vec::new(), move || pass_clock()).await {
                Ok(report) => report,
                Err(error) => return Err(fail(progress, error.source)),
            };
            let claimed = report.execution.claimed;
            let failed = !report.execution.errors.is_empty() || report.retirement.error.is_some();
            progress.passes.push(report);
            progress.pending = Some(match self.pending().await {
                Ok(n) => n,
                Err(error) => return Err(fail(progress, error)),
            });
            if failed {
                progress.stopped = DrainStop::Failed;
                return Ok(progress);
            }
            if progress.pending == Some(0) {
                progress.stopped = DrainStop::Empty;
                return Ok(progress);
            }
            if claimed == 0 {
                progress.stopped = DrainStop::Waiting;
                return Ok(progress);
            }
        }
        Ok(progress)
    }

    /// Starts independent catalogue progress for this provider without blocking
    /// the async runtime. A superseded scan must restart from a fresh cursor.
    pub(crate) async fn start_catalogue(&self) -> Result<super::catalogue::Cursor, Error> {
        let store = Arc::clone(&self.store);
        tokio::task::spawn_blocking(move || super::catalogue::Cursor::start(&store))
            .await
            .map_err(|_| Error::WorkerTask)?
    }

    /// Selects and runs one bounded catalogue page. Each provider owns its cursor.
    /// The caller's cursor advances only after the host returns successfully;
    /// errors or cancellation leave it replayable, including committed prefixes.
    /// Recency must be derived from caller-owned run/history metadata without I/O.
    pub(crate) async fn run_catalogue<C, R>(
        &self,
        cursor: &mut super::catalogue::Cursor,
        recency: R,
        clock: C,
    ) -> Result<Report, Failure>
    where
        C: Fn() -> i64 + Send + Sync + 'static,
        R: Fn(&super::catalogue::Candidate) -> Result<Recency, Error> + Send + 'static,
    {
        let store = Arc::clone(&self.store);
        let mut next = cursor.clone();
        let limit = self.bounds.max_schedule;
        let fail = |source| Failure {
            visited: 0,
            scheduled_or_existing: 0,
            source,
        };
        let (next, targets) = tokio::task::spawn_blocking(move || {
            let page = next.next(&store, limit)?;
            let targets = page
                .iter()
                .map(|candidate| Ok(candidate.target(recency(candidate)?)))
                .collect::<Result<Vec<_>, Error>>()?;
            Ok::<_, Error>((next, targets))
        })
        .await
        .map_err(|_| fail(Error::WorkerTask))?
        .map_err(fail)?;
        let report = self.run(targets, clock).await?;
        *cursor = next;
        Ok(report)
    }

    /// Schedules a finite caller-owned target list and runs one provider pass.
    /// Cancellation stops later scheduling and drops the runner; completed writes
    /// remain durable. A failed target is not counted as visited, allowing resume.
    pub(crate) async fn run<C>(&self, targets: Vec<Target>, clock: C) -> Result<Report, Failure>
    where
        C: Fn() -> i64 + Send + Sync + 'static,
    {
        let mut report = Report::default();
        let fail = |report: &Report, source| Failure {
            visited: report.visited,
            scheduled_or_existing: report.scheduled_or_existing,
            source,
        };
        if targets.len() > self.bounds.max_schedule {
            return Err(fail(
                &report,
                Error::Invalid("embedding scheduling pass exceeds bound"),
            ));
        }
        let stopped = Arc::new(AtomicBool::new(false));
        let _cancel = CancelOnDrop(Arc::clone(&stopped));
        let clock = Arc::new(clock);
        for target in targets {
            let factory = Arc::clone(&self.factory);
            let cancelled = Arc::clone(&stopped);
            let schedule_clock = Arc::clone(&clock);
            let refresh_context = self.refresh_context;
            let result = tokio::task::spawn_blocking(move || {
                let now = || {
                    if cancelled.load(Ordering::Acquire) {
                        -1
                    } else {
                        schedule_clock()
                    }
                };
                if refresh_context {
                    factory.schedule_context(
                        target.unit,
                        target.rank_pct,
                        target.recency,
                        true,
                        now,
                    )
                } else {
                    factory.schedule_file(target.unit, target.rank_pct, target.recency, now)
                }
            })
            .await
            .map_err(|_| fail(&report, Error::WorkerTask))?
            .map_err(|error| fail(&report, error))?;
            report.visited += 1;
            if result.is_some() {
                report.scheduled_or_existing += 1;
            } else {
                report.skipped += 1;
            }
        }
        report.execution = Arc::clone(&self.factory)
            .run_ready(
                http_embedding::RunLimits {
                    max_jobs: self.bounds.max_jobs,
                    heartbeat: self.bounds.heartbeat,
                },
                move || clock(),
            )
            .await
            .map_err(|error| fail(&report, error))?;
        let store = Arc::clone(&self.store);
        let policy = self.retirement.clone();
        report.retirement = tokio::task::spawn_blocking(move || policy.run(&store, &stopped))
            .await
            .unwrap_or_else(|_| retirement::Report {
                error: Some(Error::WorkerTask),
                ..Default::default()
            });
        let store = Arc::clone(&self.store);
        let generation = self.transition.generation;
        match tokio::task::spawn_blocking(move || store.finish_transition(generation)).await {
            Ok(Ok(state)) => report.transition = Some(state),
            Ok(Err(error)) => report.transition_error = Some(error.into()),
            Err(_) => report.transition_error = Some(Error::WorkerTask),
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests;
