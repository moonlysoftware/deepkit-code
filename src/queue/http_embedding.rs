//! File-backed local and HTTP embedding execution; see `v1.md` §7.2 and §8.2.

use super::{
    CurrentInput, Error, Lease, Owner, claim_with_clock, destination, embedding::Prepared, pool,
    remote_dimensions, retry, usage, worker,
};
use crate::{
    config::{self, Config},
    models::{
        Embedder,
        attempt::{AttemptError, EmbeddingBatch},
        llama::LocalProvider,
        ollama::{OllamaConfig, OllamaEmbedder},
        openai_compat::{OpenAiConfig, OpenAiEmbedder},
    },
    store::{
        Store, UnitKind,
        vectors::{self, VectorInsert},
    },
};
use async_trait::async_trait;
use rusqlite::{OptionalExtension, Transaction, params};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tokio::task::JoinSet;

/// Explicit host settings; budgets/deadlines have no invented configuration defaults.
pub(crate) struct Settings {
    pub(crate) root: PathBuf,
    pub(crate) config_root: PathBuf,
    pub(crate) config: Config,
    pub(crate) kind: UnitKind,
    pub(crate) table_slug: String,
    pub(crate) budget: usage::Budget,
    pub(crate) timeout: Duration,
}
/// Explicit bounds for one finite provider pass; future deferrals wait for another pass.
pub(crate) struct RunLimits {
    pub(crate) max_jobs: usize,
    pub(crate) heartbeat: Duration,
}

struct BatchStop(Arc<AtomicBool>);
impl Drop for BatchStop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[derive(Clone, Copy)]
enum BatchSchedule {
    Budget(i64),
    Rate(i64),
    Transient(Option<Duration>),
    Permanent,
    InputsUnavailable,
}
impl BatchSchedule {
    fn outcome(self) -> retry::Outcome {
        match self {
            Self::Budget(retry_at) => retry::Outcome::Budget { retry_at },
            Self::Rate(retry_at) => retry::Outcome::Rate { retry_at },
            Self::Transient(retry_after) => retry::Outcome::Transient { retry_after },
            Self::Permanent => retry::Outcome::Permanent,
            Self::InputsUnavailable => retry::Outcome::InputsUnavailable,
        }
    }
}

enum Provider {
    Remote(OpenAiEmbedder),
    Ollama(OllamaEmbedder),
    Local(LocalProvider),
}

/// One provider's file-job factory. Client construction does not send a request.
pub(crate) struct Factory {
    store: Arc<Store>,
    settings: Settings,
    provider: Provider,
    provider_key: String,
    model: String,
    profile: &'static str,
    dimensions_capability: Option<String>,
}
impl Factory {
    /// Builds only the selected HTTP adapter, resolving remote credentials privately.
    /// The host creates the matching vector table before starting the runner.
    pub(crate) fn new(store: Arc<Store>, settings: Settings) -> Result<Arc<Self>, Error> {
        let destination = Self::validate(&store, &settings)?;
        let embedder = &settings.config.embedder;
        let (provider, profile) = if destination.identity.provider.starts_with("embed:remote:") {
            let config =
                OpenAiConfig::from_remote(&embedder.remote, embedder.dims, settings.timeout)
                    .map_err(|_| Error::Invalid("remote embedding settings"))?;
            (
                Provider::Remote(
                    OpenAiEmbedder::from_config(config)
                        .map_err(|_| Error::Invalid("remote embedding client or credentials"))?,
                ),
                "remote",
            )
        } else if destination.identity.provider.starts_with("embed:ollama:") {
            let config =
                OllamaConfig::from_profile(&embedder.ollama, embedder.dims, settings.timeout)
                    .map_err(|_| Error::Invalid("Ollama embedding settings"))?;
            (
                Provider::Ollama(
                    OllamaEmbedder::from_config(config)
                        .map_err(|_| Error::Invalid("Ollama embedding client"))?,
                ),
                "ollama",
            )
        } else {
            return Err(Error::Invalid(
                "HTTP embedding factory requires remote or Ollama",
            ));
        };
        Self::from_http_provider(store, settings, destination, provider, profile)
    }

    fn from_http_provider(
        store: Arc<Store>,
        settings: Settings,
        destination: destination::Resolved,
        provider: Provider,
        profile: &'static str,
    ) -> Result<Arc<Self>, Error> {
        let dimensions_capability = if let Provider::Remote(remote) = &provider {
            Some(remote_dimensions::restore(
                &store,
                &destination.identity,
                remote,
            )?)
        } else {
            None
        };
        Ok(Arc::new(Self {
            store,
            settings,
            provider,
            profile,
            provider_key: destination.identity.provider,
            model: destination.identity.model,
            dimensions_capability,
        }))
    }

    fn validate(store: &Store, settings: &Settings) -> Result<destination::Resolved, Error> {
        if settings.timeout.is_zero() {
            return Err(Error::Invalid("zero embedding timeout"));
        }
        if settings
            .config
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
        vectors::vector_table_name(&settings.table_slug)?;
        let destination = destination::resolve(&settings.config, settings.kind)?;
        destination.limits(settings.budget)?;
        Ok(destination)
    }

    /// Loads a verified local adapter without downloading weights. Cache preparation
    /// belongs to the host; all jobs retain the shared dispatch/publication fences.
    pub(crate) async fn new_local(
        store: Arc<Store>,
        settings: Settings,
        cache: PathBuf,
    ) -> Result<Arc<Self>, Error> {
        let destination = Self::validate(&store, &settings)?;
        if destination.identity.provider != "embed:local" {
            return Err(Error::Invalid("local factory requires a local destination"));
        }
        let idle = settings
            .config
            .models
            .unload_after_idle()
            .map_err(|_| Error::Invalid("local idle timeout"))?;
        let provider = LocalProvider::load_registered(
            cache,
            settings.config.embedder.local.clone(),
            settings.config.embedder.dims,
            idle,
        )
        .await
        .map_err(|_| Error::Invalid("local embedding artifact or model initialization"))?;
        Ok(Arc::new(Self {
            store,
            settings,
            provider: Provider::Local(provider),
            profile: "local",
            provider_key: destination.identity.provider,
            model: destination.identity.model,
            dimensions_capability: None,
        }))
    }

    /// Runs currently eligible jobs using this provider's configured concurrency.
    /// The host calls this once per provider, with explicit pass/heartbeat bounds.
    /// Dropping the future cancels further work; in-flight native work retains its
    /// inference slot until exit. Leases/uncertain charges remain
    /// recoverable. Deferred jobs are reported without sleeping until their due time.
    pub(crate) async fn run_ready<C>(
        self: Arc<Self>,
        limits: RunLimits,
        clock: C,
    ) -> Result<pool::Report, Error>
    where
        C: Fn() -> i64 + Send + Sync + 'static,
    {
        let destination = destination::resolve(&self.settings.config, self.settings.kind)?;
        let settings = Arc::new(pool::Settings {
            provider: self.provider_key.clone(),
            concurrency: destination.concurrency,
            max_jobs: limits.max_jobs,
            // Local execution consumes one lease per native call. Remote paths
            // use their configured batch token bound below. Oversized saved
            // estimates must still reach current-input capacity checks so they
            // become visible deferrals instead of staying ready forever.
            max_claim_tokens: i64::MAX,
            budget: self.settings.budget,
            max_output_tokens: 0,
            timeout: self.settings.timeout,
            heartbeat: limits.heartbeat,
        });
        if limits.max_jobs == 0
            || limits.heartbeat.is_zero()
            || limits.heartbeat >= Duration::from_secs(120)
        {
            return Err(Error::Invalid("provider pass bounds"));
        }
        let clock = Arc::new(clock);
        let resume_clock = Arc::clone(&clock);
        let store = Arc::clone(&self.store);
        let provider = self.provider_key.clone();
        let slug = self.settings.table_slug.clone();
        let cap = self.settings.config.limits.max_l2_units;
        let token_cap = destination.max_doc_tokens.min(destination.max_batch_tokens);
        tokio::task::spawn_blocking(move || {
            super::capacity::resume_embedding(&store, &provider, &slug, cap, token_cap, || {
                resume_clock()
            })
        })
        .await
        .map_err(|_| Error::WorkerTask)??;
        if matches!(&self.provider, Provider::Local(_)) {
            pool::run_ready(
                Arc::clone(&self.store),
                settings,
                Arc::new(self),
                move || clock(),
            )
            .await
        } else {
            self.run_ready_batched(
                limits,
                destination.max_batch_items,
                destination.max_request_tokens(),
                move || clock(),
            )
            .await
        }
    }

    async fn run_ready_batched<C>(
        self: Arc<Self>,
        limits: RunLimits,
        max_batch_items: usize,
        max_batch_tokens: i64,
        clock: C,
    ) -> Result<pool::Report, Error>
    where
        C: Fn() -> i64 + Send + Sync + 'static,
    {
        let destination = destination::resolve(&self.settings.config, self.settings.kind)?;
        if limits.max_jobs == 0
            || limits.heartbeat.is_zero()
            || limits.heartbeat >= Duration::from_secs(120)
            || max_batch_items == 0
            || max_batch_tokens <= 0
            || destination.concurrency == 0
        {
            return Err(Error::Invalid("embedding batch runner bounds"));
        }
        let clock = Arc::new(clock);
        let stopped = Arc::new(AtomicBool::new(false));
        let _stop = BatchStop(Arc::clone(&stopped));
        let owner = {
            let store = Arc::clone(&self.store);
            tokio::task::spawn_blocking(move || Owner::new(&store))
                .await
                .map_err(|_| Error::WorkerTask)??
        };
        let mut running = JoinSet::new();
        let mut report = pool::Report::default();
        loop {
            while running.len() < destination.concurrency && report.claimed < limits.max_jobs {
                let available = max_batch_items.min(limits.max_jobs - report.claimed);
                let batch = {
                    let store = Arc::clone(&self.store);
                    let provider = self.provider_key.clone();
                    let owner = owner.clone();
                    let clock = Arc::clone(&clock);
                    let stopped = Arc::clone(&stopped);
                    tokio::task::spawn_blocking(move || {
                        let mut batch = claim_with_clock(
                            &store,
                            &provider,
                            &owner,
                            || {
                                if stopped.load(Ordering::SeqCst) {
                                    -1
                                } else {
                                    clock()
                                }
                            },
                            available,
                            max_batch_tokens,
                        )?;
                        if batch.is_empty() {
                            batch = claim_with_clock(
                                &store,
                                &provider,
                                &owner,
                                || {
                                    if stopped.load(Ordering::SeqCst) {
                                        -1
                                    } else {
                                        clock()
                                    }
                                },
                                1,
                                i64::MAX,
                            )?;
                        }
                        Ok::<_, Error>(batch)
                    })
                    .await
                    .map_err(|_| Error::WorkerTask)??
                };
                if batch.is_empty() {
                    break;
                }
                report.claimed += batch.len();
                let factory = Arc::clone(&self);
                let clock = Arc::clone(&clock);
                let stopped = Arc::clone(&stopped);
                let heartbeat = limits.heartbeat;
                running.spawn(async move {
                    factory
                        .execute_batch(batch, clock, stopped, heartbeat)
                        .await
                });
            }
            let Some(result) = running.join_next().await else {
                break;
            };
            let next = result.map_err(|_| Error::WorkerTask)??;
            report.published += next.published;
            report.discarded += next.discarded;
            report.deferred += next.deferred;
            report.dead += next.dead;
            report.lease_lost += next.lease_lost;
            report.errors.extend(next.errors);
        }
        Ok(report)
    }

    async fn execute_batch<C>(
        self: Arc<Self>,
        leases: Vec<Lease>,
        clock: Arc<C>,
        stopped: Arc<AtomicBool>,
        heartbeat: Duration,
    ) -> Result<pool::Report, Error>
    where
        C: Fn() -> i64 + Send + Sync + 'static,
    {
        let mut report = pool::Report::default();
        let mut prepared = Vec::with_capacity(leases.len());
        for lease in leases {
            let id = lease.id;
            let factory = Arc::new(Arc::clone(&self));
            match pool::prepare_with_renewal(
                Arc::clone(&self.store),
                factory,
                lease.clone(),
                Arc::clone(&clock),
                Arc::clone(&stopped),
                heartbeat,
            )
            .await
            {
                Ok(Some(job)) if job.prepared.is_some() => prepared.push((lease, job)),
                Ok(Some(_)) => {
                    let store = Arc::clone(&self.store);
                    let clock = Arc::clone(&clock);
                    let stopped = Arc::clone(&stopped);
                    let discarded = tokio::task::spawn_blocking(move || {
                        Ok(worker::Finished::Completed(super::complete(
                            &store,
                            &lease,
                            || {
                                if stopped.load(Ordering::SeqCst) {
                                    -1
                                } else {
                                    clock()
                                }
                            },
                            |_, _| Ok(None),
                            |_| Ok(()),
                        )?))
                    })
                    .await
                    .map_err(|_| Error::WorkerTask)?;
                    report.record(id, discarded);
                }
                Ok(None) => report.record(id, Ok(worker::Finished::LeaseLost)),
                Err(error) => {
                    let outcome = if matches!(error, Error::InputsUnavailable) {
                        Some(retry::Outcome::InputsUnavailable)
                    } else if matches!(error, Error::InputPreparation) {
                        Some(retry::Outcome::InputPreparation)
                    } else {
                        retry::Outcome::from_capacity(&error)
                    };
                    if let Some(outcome) = outcome {
                        let store = Arc::clone(&self.store);
                        let attempt = lease.clone();
                        let clock = Arc::clone(&clock);
                        let stopped = Arc::clone(&stopped);
                        let scheduled = tokio::task::spawn_blocking(move || {
                            retry::schedule(&store, &attempt, outcome, || {
                                if stopped.load(Ordering::SeqCst) {
                                    -1
                                } else {
                                    clock()
                                }
                            })
                            .map(worker::Finished::Scheduled)
                        })
                        .await
                        .map_err(|_| Error::WorkerTask)?;
                        report.record(id, scheduled);
                    } else {
                        report.record(id, Err(error));
                    }
                }
            }
        }
        if prepared.is_empty() {
            return Ok(report);
        }
        let batch_leases = prepared
            .iter()
            .map(|(lease, _)| lease.clone())
            .collect::<Vec<_>>();
        let input_tokens = batch_leases.iter().try_fold(0_i64, |sum, lease| {
            sum.checked_add(lease.tokens)
                .ok_or(Error::Invalid("embedding batch token overflow"))
        })?;
        let leader = batch_leases
            .first()
            .cloned()
            .ok_or(Error::Invalid("empty prepared embedding batch"))?;
        let leader_id = leader.id;
        let reservation = {
            let store = Arc::clone(&self.store);
            let provider = self.provider_key.clone();
            let budget = self.settings.budget;
            let clock = Arc::clone(&clock);
            let stopped = Arc::clone(&stopped);
            tokio::task::spawn_blocking(move || {
                usage::reserve(
                    &store,
                    &provider,
                    input_tokens,
                    0,
                    budget,
                    Some(&leader),
                    || {
                        if stopped.load(Ordering::SeqCst) {
                            -1
                        } else {
                            clock()
                        }
                    },
                )
            })
            .await
            .map_err(|_| Error::WorkerTask)??
        };
        let reservation = match reservation {
            usage::Admission::Reserved(reservation) => reservation,
            usage::Admission::LeaseLost => {
                for (lease, _) in prepared {
                    report.record(lease.id, Ok(worker::Finished::LeaseLost));
                }
                return Ok(report);
            }
            usage::Admission::BudgetExhausted { retry_at } => {
                self.schedule_batch(
                    &mut report,
                    prepared.into_iter().map(|(lease, _)| lease).collect(),
                    BatchSchedule::Budget(retry_at),
                    &clock,
                    &stopped,
                )
                .await?;
                return Ok(report);
            }
        };
        let dispatch = {
            let store = Arc::clone(&self.store);
            let held = Arc::new(reservation);
            let reservation = Arc::clone(&held);
            let leases = batch_leases.clone();
            let prepared = prepared.clone();
            let clock = Arc::clone(&clock);
            let stopped = Arc::clone(&stopped);
            let count = i64::try_from(prepared.len())
                .map_err(|_| Error::Invalid("embedding batch item overflow"))?;
            let result = tokio::task::spawn_blocking(move || {
                usage::mark_dispatched_batch_checked(
                    &store,
                    &reservation,
                    &leases,
                    || {
                        if stopped.load(Ordering::SeqCst) {
                            -1
                        } else {
                            clock()
                        }
                    },
                    |tx, provider| {
                        for (lease, job) in &prepared {
                            let current = worker::Work::current_at(
                                job,
                                tx,
                                lease,
                                crate::sync::l1::Work {
                                    deadline: None,
                                    cancelled: Some(&stopped),
                                },
                                clock(),
                            )?;
                            if !current.is_some_and(|input| {
                                input.target_hash.as_slice() == lease.target_hash
                                    && input.desired_hash.as_slice() == lease.desired_hash
                            }) {
                                return Err(Error::InputsUnavailable);
                            }
                        }
                        let (_, first) = prepared
                            .first()
                            .ok_or(Error::Invalid("empty embedding dispatch batch"))?;
                        let limits = worker::Work::limits(first, tx, provider)?;
                        let mut endpoint = worker::Work::outbound(first)?
                            .ok_or(Error::Invalid("HTTP embedding endpoint unavailable"))?;
                        endpoint.units = count;
                        usage::record_endpoint(tx, &reservation, &endpoint)?;
                        Ok(Some(limits))
                    },
                )
            })
            .await
            .map_err(|_| Error::WorkerTask)?;
            (held, result)
        };
        let (reservation, dispatch) = dispatch;
        let schedule = match dispatch {
            Ok(usage::Dispatch::Recorded) => None,
            Ok(usage::Dispatch::BudgetExhausted { retry_at }) => {
                Some(BatchSchedule::Budget(retry_at))
            }
            Ok(usage::Dispatch::RateLimited { retry_at }) => Some(BatchSchedule::Rate(retry_at)),
            Ok(usage::Dispatch::AttemptsExhausted | usage::Dispatch::RequestTooLarge) => {
                Some(BatchSchedule::Permanent)
            }
            Ok(
                usage::Dispatch::LeaseLost
                | usage::Dispatch::Unavailable
                | usage::Dispatch::Superseded,
            ) => Some(BatchSchedule::InputsUnavailable),
            Ok(usage::Dispatch::DayExpired) => Some(BatchSchedule::Rate(
                clock()
                    .checked_add(1)
                    .ok_or(Error::Invalid("embedding batch retry timestamp"))?,
            )),
            Err(Error::InputsUnavailable) => {
                usage::release(&self.store, &reservation)?;
                Some(BatchSchedule::InputsUnavailable)
            }
            Err(error) => return Err(error),
        };
        if let Some(schedule) = schedule {
            self.schedule_batch(
                &mut report,
                prepared.into_iter().map(|(lease, _)| lease).collect(),
                schedule,
                &clock,
                &stopped,
            )
            .await?;
            return Ok(report);
        }

        let response = self
            .call_batch_with_renewal(&prepared, &clock, &stopped, heartbeat)
            .await;
        match response {
            Err(error) => {
                if let Some(tokens) = error.reported_tokens() {
                    usage::reconcile(&self.store, &reservation, Some(tokens))?;
                }
                if matches!(error, AttemptError::DimensionsRejected) {
                    let key = self.dimensions_capability.as_deref().ok_or(Error::Invalid(
                        "dimensions rejection from non-remote provider",
                    ))?;
                    remote_dimensions::remember(&self.store, key)?;
                }
                let schedule = match &error {
                    AttemptError::Retryable { retry_after, .. } => {
                        BatchSchedule::Transient(*retry_after)
                    }
                    AttemptError::DimensionsRejected => BatchSchedule::Transient(None),
                    _ => BatchSchedule::Permanent,
                };
                self.schedule_batch(
                    &mut report,
                    prepared.into_iter().map(|(lease, _)| lease).collect(),
                    schedule,
                    &clock,
                    &stopped,
                )
                .await?;
            }
            Ok(response) => {
                // Payload-free boundary for diagnosing response-to-publication crashes.
                tracing::trace!(
                    target: "dkc::queue::worker",
                    job_id = leader_id,
                    batch_items = prepared.len(),
                    "queue provider response decoded"
                );
                usage::reconcile(&self.store, &reservation, response.tokens)?;
                if response.embeddings.len() != prepared.len() {
                    return Err(Error::Invalid("embedding batch response count"));
                }
                for ((lease, job), output) in prepared.into_iter().zip(response.embeddings) {
                    let id = lease.id;
                    let retry_lease = lease.clone();
                    let store = Arc::clone(&self.store);
                    let completion_clock = Arc::clone(&clock);
                    let completion_stopped = Arc::clone(&stopped);
                    let completed = tokio::task::spawn_blocking(move || {
                        if !super::renew_with_clock(&store, &lease, || {
                            if completion_stopped.load(Ordering::SeqCst) {
                                -1
                            } else {
                                completion_clock()
                            }
                        })? {
                            return Ok(worker::Finished::LeaseLost);
                        }
                        Ok(worker::Finished::Completed(super::complete(
                            &store,
                            &lease,
                            || {
                                if completion_stopped.load(Ordering::SeqCst) {
                                    -1
                                } else {
                                    completion_clock()
                                }
                            },
                            |tx, lease| {
                                worker::Work::current_at(
                                    &job,
                                    tx,
                                    lease,
                                    crate::sync::l1::Work {
                                        deadline: None,
                                        cancelled: Some(&completion_stopped),
                                    },
                                    completion_clock(),
                                )
                            },
                            |tx| {
                                worker::Work::publish_with_work(
                                    &job,
                                    tx,
                                    output,
                                    crate::sync::l1::Work {
                                        deadline: None,
                                        cancelled: Some(&completion_stopped),
                                    },
                                )
                            },
                        )?))
                    })
                    .await
                    .map_err(|_| Error::WorkerTask)?;
                    let outcome = completed.as_ref().err().and_then(|error| {
                        if matches!(error, Error::InputsUnavailable) {
                            Some(retry::Outcome::InputsUnavailable)
                        } else if matches!(error, Error::InputPreparation) {
                            Some(retry::Outcome::InputPreparation)
                        } else {
                            retry::Outcome::from_capacity(error)
                        }
                    });
                    if let Some(outcome) = outcome {
                        let store = Arc::clone(&self.store);
                        let clock = Arc::clone(&clock);
                        let stopped = Arc::clone(&stopped);
                        let scheduled = tokio::task::spawn_blocking(move || {
                            retry::schedule(&store, &retry_lease, outcome, || {
                                if stopped.load(Ordering::SeqCst) {
                                    -1
                                } else {
                                    clock()
                                }
                            })
                            .map(worker::Finished::Scheduled)
                        })
                        .await
                        .map_err(|_| Error::WorkerTask)?;
                        report.record(id, scheduled);
                    } else {
                        report.record(id, completed);
                    }
                }
            }
        }
        Ok(report)
    }

    async fn schedule_batch<C>(
        &self,
        report: &mut pool::Report,
        leases: Vec<Lease>,
        schedule: BatchSchedule,
        clock: &Arc<C>,
        stopped: &Arc<AtomicBool>,
    ) -> Result<(), Error>
    where
        C: Fn() -> i64 + Send + Sync + 'static,
    {
        for lease in leases {
            let id = lease.id;
            let store = Arc::clone(&self.store);
            let clock = Arc::clone(clock);
            let stopped = Arc::clone(stopped);
            let result = tokio::task::spawn_blocking(move || {
                retry::schedule(&store, &lease, schedule.outcome(), || {
                    if stopped.load(Ordering::SeqCst) {
                        -1
                    } else {
                        clock()
                    }
                })
                .map(worker::Finished::Scheduled)
            })
            .await
            .map_err(|_| Error::WorkerTask)?;
            report.record(id, result);
        }
        Ok(())
    }

    async fn call_batch_with_renewal<C>(
        &self,
        prepared: &[(Lease, Job)],
        clock: &Arc<C>,
        stopped: &Arc<AtomicBool>,
        heartbeat: Duration,
    ) -> Result<EmbeddingBatch, AttemptError>
    where
        C: Fn() -> i64 + Send + Sync + 'static,
    {
        let documents = prepared
            .iter()
            .filter_map(|(_, job)| job.prepared.as_deref().map(Prepared::input))
            .collect::<Vec<_>>();
        if documents.len() != prepared.len() {
            return Err(AttemptError::InvalidInput);
        }
        let request = async {
            match &self.provider {
                Provider::Remote(provider) => {
                    provider.embed_documents_with_usage_once(&documents).await
                }
                Provider::Ollama(provider) => {
                    provider.embed_documents_with_usage_once(&documents).await
                }
                Provider::Local(_) => Err(AttemptError::InvalidInput),
            }
        };
        tokio::pin!(request);
        let deadline = tokio::time::sleep(self.settings.timeout);
        tokio::pin!(deadline);
        let heartbeat_timer = tokio::time::sleep(heartbeat);
        tokio::pin!(heartbeat_timer);
        loop {
            tokio::select! {
                biased;
                _ = &mut deadline => return Err(AttemptError::Retryable { status: None, retry_after: None }),
                _ = &mut heartbeat_timer => {
                    let store = Arc::clone(&self.store);
                    let leases = prepared.iter().map(|(lease, _)| lease.clone()).collect::<Vec<_>>();
                    let clock = Arc::clone(clock);
                    let stopped = Arc::clone(stopped);
                    let renewed = tokio::task::spawn_blocking(move || {
                        for lease in &leases {
                            if !super::renew_with_clock(&store, lease, || {
                                if stopped.load(Ordering::SeqCst) { -1 } else { clock() }
                            })? {
                                return Ok(false);
                            }
                        }
                        Ok::<_, Error>(true)
                    }).await.map_err(|_| AttemptError::InvalidResponse)
                        .and_then(|result| result.map_err(|_| AttemptError::InvalidResponse))?;
                    if !renewed {
                        return Err(AttemptError::Retryable { status: None, retry_after: None });
                    }
                    heartbeat_timer.as_mut().reset(tokio::time::Instant::now() + heartbeat);
                }
                result = &mut request => return result,
            }
        }
    }

    /// Reads context freshness after resolving current source/configuration policy.
    /// None means the source is not presently eligible; stale vectors are retained.
    pub(crate) fn context_state(&self, unit: i64) -> Result<Option<vectors::ContextState>, Error> {
        self.store.read_snapshot(|tx| {
            let Some((prepared, _)) = self.resolve_current(tx, unit)? else {
                return Ok(None);
            };
            Ok::<_, Error>(Some(vectors::context_state_in(
                tx,
                &self.settings.table_slug,
                unit,
                prepared.context_hash(),
            )?))
        })?
    }

    /// Schedules a current file occurrence with rank/recency priority and cap checks.
    /// New capped generations are persisted paused, visible without a worker lease.
    /// Stale/protected sources and retained impossible-growth pauses return None;
    /// existing job state is preserved without claiming fresh source evidence. This
    /// blocking API performs no HTTP, and does not reserve future DB/WAL growth.
    pub(crate) fn schedule_file(
        &self,
        unit: i64,
        rank_pct: f64,
        recency: super::priority::Recency,
        clock: impl FnOnce() -> i64,
    ) -> Result<Option<i64>, Error> {
        self.schedule_context(unit, rank_pct, recency, false, clock)
    }

    /// Schedules an occurrence, optionally refreshing retained role-only coverage.
    /// Explicit refresh still requires current source proof and normal cap admission.
    pub(crate) fn schedule_context(
        &self,
        unit: i64,
        rank_pct: f64,
        recency: super::priority::Recency,
        refresh_context: bool,
        clock: impl FnOnce() -> i64,
    ) -> Result<Option<i64>, Error> {
        super::storage::transaction(&self.store, super::storage::Kind::Growth, |tx| {
            let now = clock();
            if now < 0 || now == super::capacity::PAUSED {
                return Err(Error::Invalid("scheduling timestamp"));
            }
            self.check_database_authority()?;
            let retained_pause: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM jobs WHERE kind='embed' AND target_kind='unit'
                 AND target_id=?1 AND provider=?2 AND state='queued' AND lease_owner IS NULL
                 AND attempts<8 AND not_before=?3 AND last_error=?4)",
                rusqlite::params![
                    unit,
                    self.provider_key,
                    super::capacity::PAUSED,
                    super::capacity::Reason::Vectors.message()
                ],
                |r| r.get(0),
            )?;
            if retained_pause
                && super::capacity::vector_growth_blocked(
                    &self.store,
                    &tx,
                    &self.settings.table_slug,
                    unit,
                    self.settings.config.limits.max_l2_units,
                )?
            {
                // Do not prepare another generation that cannot grow. Retain
                // the visible pause without claiming its source is current.
                // Once capacity changes, normal preparation runs from scratch.
                return Ok(None);
            }
            let Some((prepared, destination)) = self.resolve_current(&tx, unit)? else {
                return Ok(None);
            };
            crate::store::unit_contexts::record(
                &tx,
                unit,
                &prepared.current().target_hash,
                prepared.context_hash(),
                &self.settings.config,
            )?;
            let mut input = prepared.enqueue_ranked(rank_pct, recency)?;
            let context_stale = vectors::context_state_in(
                &tx,
                &self.settings.table_slug,
                unit,
                prepared.context_hash(),
            )? == vectors::ContextState::Stale;
            if context_stale {
                // Only a matching neutral generation proves a role-only change.
                // For legacy rows the old full generation can prove the original
                // role-free input; NULL is never treated as unconditional proof.
                let role_only: bool = tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM embedding_publications WHERE unit_id=?1 AND table_slug=?2 AND COALESCE(role_free_hash,desired_hash)=?3)",
                    rusqlite::params![unit, self.settings.table_slug, prepared.role_free_hash()],
                    |r| r.get(0),
                )?;
                if role_only
                    && !refresh_context
                    && !self.settings.config.embedder.reembed_on_role_change
                {
                    tx.commit()?;
                    return Ok(None);
                }
                // Existing coverage is still served. Context-only replacements
                // use the lowest ordinary score instead of competing as new L2.
                input.priority = 0.0;
            }
            let paused = match self.check_capacity(&tx, unit, &prepared, &destination) {
                Ok(()) => None,
                Err(error) => Some(super::capacity::reason(&error).ok_or(error)?),
            };
            let id = super::enqueue_in(&tx, &input, now, paused)?;
            if context_stale
                || !vectors::publication_matches_in(
                    &tx,
                    &self.settings.table_slug,
                    unit,
                    &prepared.current().desired_hash,
                )?
            {
                // Historical completion is not current coverage. Reuse identity,
                // but acquire a new lease generation and admission on dispatch.
                // Failed/dead jobs, active leases and delayed retries are untouched.
                tx.execute("UPDATE jobs SET state='queued',attempts=0,not_before=?2,lease_owner=NULL,leased_until=NULL,last_error=?3,updated_at=?4 WHERE id=?1 AND state='done'",
                    rusqlite::params![id, paused.map_or(0, |_| super::capacity::PAUSED), paused.map(super::capacity::Reason::message), now])?;
            }
            tx.commit()?;
            Ok(Some(id))
        })
    }

    fn prepare_current(&self, tx: &Transaction<'_>, unit: i64) -> Result<Option<Prepared>, Error> {
        self.check_database_authority()?;
        let Some((prepared, destination)) = self.resolve_current(tx, unit)? else {
            self.check_database_authority()?;
            return Ok(None);
        };
        self.check_capacity(tx, unit, &prepared, &destination)?;
        self.check_database_authority()?;
        Ok(Some(prepared))
    }

    fn check_database_authority(&self) -> Result<(), Error> {
        if !self.store.checked_database_is_current() {
            return Err(Error::Invalid("embedding usage database changed"));
        }
        Ok(())
    }

    fn resolve_current(
        &self,
        tx: &Transaction<'_>,
        unit: i64,
    ) -> Result<Option<(Prepared, destination::Resolved)>, Error> {
        let Some((prepared, destination)) = destination::prepare_file(
            tx,
            &self.settings.root,
            &self.settings.config_root,
            &self.settings.config,
            unit,
        )?
        else {
            return Ok(None);
        };
        if destination.identity.provider != self.provider_key
            || destination.identity.model != self.model
        {
            return Ok(None);
        }
        if matches!(
            prepared.input().kind,
            crate::store::UnitKind::PullRequest | crate::store::UnitKind::Module
        ) && super::summary_source::replaces_parent(
            tx,
            &self.settings.root,
            &self.settings.config_root,
            &self.settings.config,
            unit,
        )? {
            return Ok(None);
        }
        Ok(Some((prepared, destination)))
    }

    fn check_capacity(
        &self,
        tx: &Transaction<'_>,
        unit: i64,
        prepared: &Prepared,
        destination: &destination::Resolved,
    ) -> Result<(), Error> {
        let cap = self
            .settings
            .config
            .limits
            .db_max_bytes()
            .map_err(|_| Error::Invalid("embedding database cap"))?
            .0
            .min(self.store.db_max_bytes);
        let allocated = crate::store::capacity::limit_main_growth(
            tx,
            crate::store::capacity::growth_limit(cap),
        )?;
        let tokens = prepared.enqueue(0.0).tokens;
        if tokens > destination.max_doc_tokens || tokens > destination.max_batch_tokens {
            return Err(super::capacity::Reason::Tokens.into());
        }
        let matching: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM embedding_tables WHERE slug=?1 AND model=?2 AND profile=?3 AND dims=?4 AND active=1)",
            params![self.settings.table_slug,self.model,self.profile,i64::try_from(destination.identity.dims).map_err(|_| Error::Invalid("embedding dimensions overflow"))?], |r| r.get(0))?;
        if !matching {
            return Err(Error::Invalid(
                "embedding table does not match current destination",
            ));
        }
        if let Some(binding) =
            crate::store::embedding_bindings::read(tx, &self.settings.table_slug)?
            && !binding.same_space(&destination.binding(&self.settings.config))
        {
            return Err(Error::Invalid(
                "embedding provider differs from table binding",
            ));
        }
        if let Some(route) = crate::store::transitions::read(tx)?
            && route.desired.get(prepared.input().kind.as_str()) != Some(&self.settings.table_slug)
        {
            return Err(Error::Invalid(
                "embedding table is not the current routing destination",
            ));
        }
        let replacement = vectors::plan_replacement(
            &self.store,
            tx,
            &self.settings.table_slug,
            unit,
            self.settings.config.limits.max_l2_units,
        )?;
        if replacement.grows_bytes(destination.identity.dims)
            && !super::capacity::database_growth_allowed(
                allocated,
                cap,
                super::capacity::GrowthLevel::Dense,
            )
        {
            return Err(super::capacity::Reason::Database.into());
        }
        Ok(())
    }
}

/// Owned immutable job; provider calls cannot acquire the database writer lock.
#[derive(Clone)]
pub(crate) struct Job {
    factory: Arc<Factory>,
    unit: i64,
    prepared: Option<Arc<Prepared>>,
}

impl pool::Factory for Arc<Factory> {
    type Job = Job;
    fn prepare(&self, store: &Store, lease: &Lease) -> Result<Job, Error> {
        // Factory and worker publication must target the same Store instance.
        if !std::ptr::eq(store, self.store.as_ref()) {
            return Err(Error::Invalid("embedding factory store mismatch"));
        }
        super::storage::transaction(store, super::storage::Kind::Control, |tx| {
            let unit: Option<i64> = tx.query_row("SELECT target_id FROM jobs WHERE id=?1 AND kind='embed' AND target_kind='unit' AND provider=?2",
            params![lease.id,self.provider_key],|r|r.get(0)).optional()?;
            let unit = unit.ok_or(Error::Invalid("unsupported embedding job target"))?;
            let prepared = self.prepare_current(&tx, unit)?.map(Arc::new);
            Ok(Job {
                factory: Arc::clone(self),
                unit,
                prepared,
            })
        })
    }
}

#[async_trait]
impl worker::Work for Job {
    type Output = Vec<f32>;
    fn current(&self, tx: &Transaction<'_>, _lease: &Lease) -> Result<Option<CurrentInput>, Error> {
        if self.prepared.is_none() {
            return Ok(None);
        }
        Ok(self
            .factory
            .prepare_current(tx, self.unit)?
            .map(|input| input.current()))
    }
    fn limits(&self, _tx: &Transaction<'_>, provider: &str) -> Result<usage::Limits, Error> {
        self.factory.check_database_authority()?;
        let current = config::load(&self.factory.settings.config_root, None)
            .map_err(|_| Error::Invalid("embedding limits reload"))?
            .config;
        if current != self.factory.settings.config {
            return Err(Error::Invalid(
                "embedding configuration changed before admission",
            ));
        }
        let kind = self
            .prepared
            .as_ref()
            .ok_or(Error::Invalid("missing embedding input"))?
            .input()
            .kind;
        let destination = destination::resolve(&current, kind)?;
        if destination.identity.provider != provider {
            return Err(Error::Invalid(
                "embedding provider changed before admission",
            ));
        }
        let limits = destination.limits(self.factory.settings.budget)?;
        self.factory.check_database_authority()?;
        Ok(limits)
    }
    fn outbound(&self) -> Result<Option<usage::Endpoint>, Error> {
        if self.factory.provider_key == "embed:local" {
            return Ok(None);
        }
        let kind = self
            .prepared
            .as_ref()
            .ok_or(Error::Invalid("missing outbound input"))?
            .input()
            .kind;
        let resolved = destination::resolve(&self.factory.settings.config, kind)?;
        Ok(Some(usage::Endpoint {
            kind: "embed",
            url: resolved.identity.endpoint,
            units: 1,
        }))
    }
    async fn call(&self) -> Result<worker::Response<Vec<f32>>, AttemptError> {
        let input = self
            .prepared
            .as_ref()
            .ok_or(AttemptError::InvalidInput)?
            .input();
        let mut batch = match &self.factory.provider {
            Provider::Remote(provider) => {
                provider.embed_documents_with_usage_once(&[input]).await?
            }
            Provider::Ollama(provider) => {
                provider.embed_documents_with_usage_once(&[input]).await?
            }
            Provider::Local(provider) => EmbeddingBatch {
                embeddings: provider.embed_documents(&[input]).await.map_err(
                    |error| match error {
                        crate::models::Error::Local(
                            crate::models::llama::Error::InputTooLong { .. },
                        ) => AttemptError::InvalidInput,
                        _ => AttemptError::InvalidResponse,
                    },
                )?,
                // Native token usage is not yet returned by the provider API.
                tokens: None,
            },
        };
        if batch.embeddings.len() != 1 {
            return Err(AttemptError::InvalidResponse);
        }
        let output = batch
            .embeddings
            .pop()
            .ok_or(AttemptError::InvalidResponse)?;
        // The queue reconciles provider totals; absent usage remains explicitly estimated.
        Ok(worker::Response {
            output,
            tokens: batch.tokens,
        })
    }
    fn publish(&self, tx: &Transaction<'_>, output: Vec<f32>) -> Result<(), Error> {
        let prepared = self
            .prepared
            .as_ref()
            .ok_or(Error::Invalid("missing embedding publication input"))?;
        self.factory.store.replace_vector_in(
            tx,
            VectorInsert {
                table_slug: &self.factory.settings.table_slug,
                unit_id: self.unit,
                ctx_hash: prepared.context_hash(),
                embedding_hash: &prepared.current().target_hash,
                embedding: &output,
            },
        )?;
        crate::store::unit_contexts::record(
            tx,
            self.unit,
            &prepared.current().target_hash,
            prepared.context_hash(),
            &self.factory.settings.config,
        )?;
        tx.execute(
            "INSERT INTO embedding_publications(unit_id,table_slug,desired_hash,role_free_hash) VALUES(?1,?2,?3,?4)",
            rusqlite::params![
                self.unit,
                self.factory.settings.table_slug,
                prepared.current().desired_hash,
                prepared.role_free_hash(),
            ],
        )?;
        tx.execute(
            "UPDATE units SET ladder_level=MAX(ladder_level,2) WHERE id=?1",
            [self.unit],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        queue::{self, pool::Factory as _, worker::Work as _},
        store::vectors::VectorTableSpec,
    };
    use std::{error::Error as StdError, fs};
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    async fn setup(
        server: &MockServer,
    ) -> Result<(tempfile::TempDir, Arc<Store>, Arc<Factory>, Lease, Job), Box<dyn StdError>> {
        let (root, store, _) = queue::source::tests::fixture()?;
        fs::write(
            root.path().join(".deepkit/config.toml"),
            format!(
                "[index]\nrepo_name='fixture'\n[embedder]\nprofile='ollama'\ndims=2\n[embedder.ollama]\nurl='{}'\n",
                server.uri()
            ),
        )?;
        let config = config::load(root.path(), None)?.config;
        let store = Arc::new(store);
        store.ensure_vector_table(VectorTableSpec {
            slug: "ollama_fixture_2",
            model: &config.embedder.ollama.model,
            profile: "ollama",
            dims: 2,
            created_at: 0,
            active: true,
        })?;
        let prepared = {
            let mut writer = store.writer()?;
            let tx = writer.transaction()?;
            destination::prepare_file(&tx, root.path(), root.path(), &config, 1)?
                .ok_or("missing input")?
                .0
        };
        queue::enqueue(&store, &prepared.enqueue(1.0), 0)?;
        let lease = queue::claim(
            &store,
            prepared.enqueue(1.0).provider,
            &queue::Owner::new(&store)?,
            0,
            1,
            10000,
        )?
        .remove(0);
        let factory = Factory::new(
            Arc::clone(&store),
            Settings {
                root: root.path().into(),
                config_root: root.path().into(),
                config,
                kind: UnitKind::Chunk,
                table_slug: "ollama_fixture_2".into(),
                budget: usage::Budget {
                    calls: None,
                    tokens: None,
                },
                timeout: Duration::from_secs(2),
            },
        )?;
        let job = factory.prepare(&store, &lease)?;
        Ok((root, store, factory, lease, job))
    }
    fn settings(factory: &Factory) -> worker::Settings<'_> {
        worker::Settings {
            provider: &factory.provider_key,
            budget: factory.settings.budget,
            max_output_tokens: 0,
            timeout: Duration::from_secs(2),
            heartbeat: Duration::from_secs(30),
        }
    }
    #[tokio::test]
    async fn checked_ledger_replacement_after_preparation_refuses_dispatch_and_publication()
    -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start().await;
        // A usable response ensures refusal cannot be attributed to a missing mock.
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]]})),
            )
            .expect(0)
            .mount(&server)
            .await;
        let (root, initial, original, lease, original_job) = setup(&server).await?;
        let database = root.path().join(".deepkit/code.db");
        initial
            .writer()?
            .execute("VACUUM INTO ?1", [database.to_string_lossy().as_ref()])?;
        let checked = Arc::new(crate::store::open_existing_for_usage(
            root.path(),
            &database,
            Default::default(),
        )?);
        let factory = Factory::new(
            Arc::clone(&checked),
            Settings {
                root: root.path().into(),
                config_root: root.path().into(),
                config: original.settings.config.clone(),
                kind: UnitKind::Chunk,
                table_slug: "ollama_fixture_2".into(),
                budget: usage::Budget {
                    calls: None,
                    tokens: None,
                },
                timeout: Duration::from_secs(2),
            },
        )?;
        let job = factory.prepare(&checked, &lease)?;
        checked.read_snapshot(|tx| {
            assert!(job.current(tx, &lease)?.is_some());
            job.limits(tx, &factory.provider_key)?;
            Ok::<_, Error>(())
        })??;
        drop(original_job);
        drop(original);
        drop(initial);

        let retired = root.path().join(".deepkit/retired");
        fs::create_dir(&retired)?;
        for name in ["code.db", "code.db-wal", "code.db-shm"] {
            let path = root.path().join(".deepkit").join(name);
            if path.exists() {
                fs::rename(path, retired.join(name))?;
            }
        }
        let replacement = crate::store::open(&database, Default::default())?;
        checked.read_snapshot(|tx| {
            assert!(job.current(tx, &lease).is_err());
            assert!(job.limits(tx, &factory.provider_key).is_err());
        })?;
        assert!(factory.prepare(&checked, &lease).is_err());
        assert!(
            worker::execute(&checked, &lease, &settings(&factory), &job, || 1)
                .await
                .is_err()
        );
        assert!(server.received_requests().await.unwrap().is_empty());
        replacement.read_snapshot(|tx| {
            for table in ["usage_reservations", "embedding_publications"] {
                let count: i64 = tx
                    .query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
                    .unwrap();
                assert_eq!(count, 0, "replacement {table}");
            }
            assert_eq!(vectors::count_vectors_in_transaction(tx).unwrap(), 0);
        })?;
        // Retired pooled handles still remember the original WAL pathname. Close
        // every owner before auditing the moved database together with its sidecars.
        drop(job);
        drop(factory);
        assert_eq!(Arc::strong_count(&checked), 1);
        drop(checked);
        let retired_ledger = rusqlite::Connection::open_with_flags(
            fs::canonicalize(&retired)?.join("code.db"),
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NOFOLLOW,
        )?;
        for sql in [
            "SELECT count(*) FROM usage_reservations WHERE state IN ('dispatched','reconciled')",
            "SELECT count(*) FROM embedding_publications",
            "SELECT count(*) FROM vec_ollama_fixture_2",
            "SELECT count(*) FROM jobs WHERE state='done'",
        ] {
            let count: i64 = retired_ledger.query_row(sql, [], |r| r.get(0))?;
            assert_eq!(count, 0, "retired ledger: {sql}");
        }
        Ok(())
    }
    #[tokio::test]
    async fn completed_current_schedule_is_write_free_until_priority_changes()
    -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]]})),
            )
            .expect(1)
            .mount(&server)
            .await;
        let (root, initial, original, lease, job) = setup(&server).await?;
        assert_eq!(
            worker::execute(&initial, &lease, &settings(&original), &job, || 1).await?,
            worker::Finished::Completed(queue::Completion::Published)
        );
        let db = root.path().join(".deepkit/scheduling.db");
        initial
            .writer()?
            .execute("VACUUM INTO ?1", [db.to_string_lossy().as_ref()])?;
        let store = Arc::new(crate::store::open(&db, Default::default())?);
        let factory = Factory::new(
            Arc::clone(&store),
            Settings {
                root: root.path().into(),
                config_root: root.path().into(),
                config: original.settings.config.clone(),
                kind: UnitKind::Chunk,
                table_slug: "ollama_fixture_2".into(),
                budget: usage::Budget {
                    calls: None,
                    tokens: None,
                },
                timeout: Duration::from_secs(2),
            },
        )?;
        let schedule = |rank, now| {
            factory.schedule_file(1, rank, queue::priority::Recency::ChangedFile, || now)
        };
        // Normalize scheduling metadata before observing the unchanged catalogue.
        assert_eq!(schedule(0.5, 2)?, Some(lease.id));
        type JobIdentity = (String, Vec<u8>, Vec<u8>, i64, i64);
        let identity = || -> Result<JobIdentity, Box<dyn StdError>> {
            // All generation/attempt fields survive reprioritization unchanged.
            Ok(store.writer()?.query_row(
                "SELECT state,target_hash,desired_hash,lease_generation,attempts FROM jobs WHERE id=?1",
                [lease.id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
            )?)
        };
        let before_identity = identity()?;
        assert_eq!(before_identity.0, "done");
        store.writer()?.execute_batch(
            "CREATE TRIGGER reject_equal_priority AFTER UPDATE OF priority ON jobs
             WHEN OLD.priority=NEW.priority
             BEGIN SELECT RAISE(ABORT,'redundant priority update'); END;",
        )?;
        assert!(!crate::store::wal::checkpoint(&store)?.busy);
        let changes = store.writer()?.total_changes();
        let wal = crate::store::wal::wal_bytes(&*store.writer()?)?;
        for now in 3..19 {
            assert_eq!(schedule(0.5, now)?, Some(lease.id));
        }
        assert_eq!(store.writer()?.total_changes(), changes);
        assert_eq!(crate::store::wal::wal_bytes(&*store.writer()?)?, wal);
        assert_eq!(identity()?, before_identity);

        // A genuine rank change remains observable exactly once, even for done jobs.
        assert_eq!(schedule(0.9, 19)?, Some(lease.id));
        assert_eq!(store.writer()?.total_changes(), changes + 1);
        let actual: f64 = store.writer()?.query_row(
            "SELECT priority FROM jobs WHERE id=?1",
            [lease.id],
            |r| r.get(0),
        )?;
        assert_eq!(
            actual,
            queue::priority::calculate(
                queue::Kind::Embed,
                0.9,
                queue::priority::Recency::ChangedFile,
            )?
        );
        assert_eq!(identity()?, before_identity);
        let changed_wal = crate::store::wal::wal_bytes(&*store.writer()?)?;
        assert_eq!(schedule(0.9, 20)?, Some(lease.id));
        assert_eq!(store.writer()?.total_changes(), changes + 1);
        assert_eq!(
            crate::store::wal::wal_bytes(&*store.writer()?)?,
            changed_wal
        );
        // A no-op candidate still resolves the full current configuration fence.
        let mut changed_config = factory.settings.config.clone();
        changed_config.embedder.max_doc_tokens /= 2;
        fs::write(
            root.path().join(".deepkit/config.toml"),
            toml::to_string(&changed_config)?,
        )?;
        assert_eq!(schedule(0.9, 21)?, None);
        assert_eq!(store.writer()?.total_changes(), changes + 1);
        assert_eq!(identity()?, before_identity);
        server.verify().await;
        Ok(())
    }

    #[tokio::test]
    async fn concrete_worker_reconciles_reported_usage_and_retains_malformed_uncertainty()
    -> Result<(), Box<dyn StdError>> {
        for count in [0, 42, -1] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/api/embed"))
                .respond_with(ResponseTemplate::new(200).set_body_json(
                    serde_json::json!({"embeddings":[[1.0,0.0]], "prompt_eval_count":count}),
                ))
                .expect(1)
                .mount(&server)
                .await;
            let (_root, store, factory, lease, job) = setup(&server).await?;
            let outcome = worker::execute(&store, &lease, &settings(&factory), &job, || 1).await?;
            let status = queue::status::snapshot(&store, 1)?;
            let json = serde_json::to_value(status)?;
            let provider = &json["providers"][&factory.provider_key];
            if count >= 0 {
                assert_eq!(
                    outcome,
                    worker::Finished::Completed(queue::Completion::Published)
                );
                assert_eq!(store.count_vectors()?, 1);
                assert_eq!(provider["calls_today"], 1);
                assert_eq!(provider["tokens_today"], count);
                assert_eq!(provider["estimated_tokens_today"], 0);
                assert_eq!(provider["uncertain_calls"], 0);
            } else {
                assert_eq!(
                    outcome,
                    worker::Finished::Scheduled(queue::retry::Scheduled::Dead)
                );
                assert_eq!(store.count_vectors()?, 0);
                assert_eq!(provider["calls_today"], 0);
                assert_eq!(provider["uncertain_calls"], 1);
                assert_eq!(json["jobs"]["dead"], 1);
            }
            server.verify().await;
        }
        Ok(())
    }

    #[tokio::test]
    async fn concrete_worker_sends_canonical_input_and_atomically_publishes_vector()
    -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]]})),
            )
            .expect(1)
            .mount(&server)
            .await;
        let (_root, store, factory, lease, job) = setup(&server).await?;
        let result = worker::execute(&store, &lease, &settings(&factory), &job, || 1).await?;
        assert_eq!(
            result,
            worker::Finished::Completed(queue::Completion::Published)
        );
        assert_eq!(store.count_vectors()?, 1);
        let (state,ladder):(String,i64)=store.writer()?.query_row("SELECT j.state,u.ladder_level FROM jobs j JOIN units u ON u.id=j.target_id WHERE j.id=?1",[lease.id],|r|Ok((r.get(0)?,r.get(1)?)))?;
        assert_eq!((state, ladder), ("done".into(), 2));
        let requests = server.received_requests().await.ok_or("missing requests")?;
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body)?;
        assert_eq!(
            body["input"][0],
            "fixture › a.rs\nfn example()\nrust chunk\nfn example() {}"
        );
        let (calls, tokens, estimated): (i64, i64, i64) = store.writer()?.query_row(
            "SELECT calls,tokens,estimated_tokens FROM usage",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )?;
        assert_eq!(calls, 1);
        assert!(tokens > 0);
        assert_eq!(tokens, estimated);
        Ok(())
    }
    #[tokio::test]
    async fn changed_endpoint_budget_before_dispatch_refuses_old_host_limits()
    -> Result<(), Box<dyn StdError>> {
        use worker::Work;
        let server = MockServer::start().await;
        let (root, store, factory, lease, job) = setup(&server).await?;
        let original = config::load(root.path(), None)?.config;
        let mut changed = original.clone();
        changed.queue.budgets.push(config::queue::EndpointBudget {
            profile: "ollama".into(),
            url: server.uri(),
            daily_calls: Some(0),
            daily_tokens: None,
        });
        fs::write(root.path().join(".deepkit/config.toml"), changed.dump()?)?;
        store.read_snapshot(|tx| {
            assert!(matches!(
                job.limits(tx, &factory.provider_key),
                Err(Error::Invalid(
                    "embedding configuration changed before admission"
                ))
            ));
            Ok::<_, rusqlite::Error>(())
        })??;
        // The actual leased worker also refuses stale configuration before HTTP.
        assert_eq!(
            worker::execute(&store, &lease, &settings(&factory), &job, || 1).await?,
            worker::Finished::Completed(queue::Completion::Discarded)
        );
        assert!(
            server
                .received_requests()
                .await
                .ok_or("requests")?
                .is_empty()
        );
        assert_eq!(store.count_vectors()?, 0);
        let reservations: i64 = store.writer()?.query_row(
            "SELECT count(*) FROM usage_reservations WHERE state != 'released'",
            [],
            |row| row.get(0),
        )?;
        assert_eq!(reservations, 0);
        let selected = destination::resolve(&changed, UnitKind::Chunk)?;
        assert_eq!(
            super::super::host::Bounds::for_provider(&changed.queue, &selected.identity.provider)?
                .budget
                .calls,
            Some(0)
        );
        fs::write(root.path().join(".deepkit/config.toml"), original.dump()?)?;
        store.read_snapshot(|tx| {
            assert!(job.limits(tx, &factory.provider_key).is_ok());
            Ok::<_, rusqlite::Error>(())
        })??;
        Ok(())
    }

    #[tokio::test]
    async fn changed_source_before_dispatch_sends_nothing() -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start().await;
        let (root, store, factory, lease, job) = setup(&server).await?;
        fs::write(root.path().join("a.rs"), "fn changed() {}")?;
        assert_eq!(
            worker::execute(&store, &lease, &settings(&factory), &job, || 1).await?,
            worker::Finished::Completed(queue::Completion::Discarded)
        );
        assert_eq!(server.received_requests().await.ok_or("requests")?.len(), 0);
        assert_eq!(store.count_vectors()?, 0);
        Ok(())
    }
    #[tokio::test]
    async fn source_change_during_http_discards_output_but_retains_usage()
    -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(Duration::from_millis(250))
                    .set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]]})),
            )
            .expect(1)
            .mount(&server)
            .await;
        let (root, store, factory, lease, job) = setup(&server).await?;
        let settings = settings(&factory);
        let edit = async {
            tokio::time::timeout(Duration::from_secs(1), async {
                loop {
                    if !server.received_requests().await.unwrap().is_empty() {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .unwrap();
            fs::write(root.path().join("a.rs"), "fn changed() {}").unwrap();
        };
        let (result, ()) =
            tokio::join!(worker::execute(&store, &lease, &settings, &job, || 1), edit);
        assert_eq!(
            result?,
            worker::Finished::Completed(queue::Completion::Discarded)
        );
        assert_eq!(store.count_vectors()?, 0);
        let calls: i64 = store
            .writer()?
            .query_row("SELECT calls FROM usage", [], |r| r.get(0))?;
        assert_eq!(calls, 1);
        Ok(())
    }
    #[tokio::test]
    async fn table_mismatch_is_rejected_before_http() -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start().await;
        let (_root, store, _factory, lease, job) = setup(&server).await?;
        store
            .writer()?
            .execute("UPDATE embedding_tables SET active=0", [])?;
        {
            let mut writer = store.writer()?;
            let tx = writer.transaction()?;
            assert!(job.current(&tx, &lease).is_err());
        }
        assert_eq!(server.received_requests().await.ok_or("requests")?.len(), 0);
        Ok(())
    }
    #[tokio::test]
    async fn configured_caps_stop_preparation_before_provider_calls()
    -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start().await;
        for extra in [
            "[limits]\nmax_l2_units=0\n",
            "[limits]\ndb_max_bytes='1 B'\n",
            "[embedder]\nprofile='ollama'\ndims=2\nmax_doc_tokens=1\n",
        ] {
            let (root, store, _factory, lease, _job) = setup(&server).await?;
            let base = if extra.starts_with("[embedder]") {
                ""
            } else {
                "[embedder]\nprofile='ollama'\ndims=2\n"
            };
            fs::write(
                root.path().join(".deepkit/config.toml"),
                format!(
                    "[index]\nrepo_name='fixture'\n{base}[embedder.ollama]\nurl='{}'\n{extra}",
                    server.uri()
                ),
            )?;
            let config = config::load(root.path(), None)?.config;
            let factory = Factory::new(
                Arc::clone(&store),
                Settings {
                    root: root.path().into(),
                    config_root: root.path().into(),
                    config,
                    kind: UnitKind::Chunk,
                    table_slug: "ollama_fixture_2".into(),
                    budget: usage::Budget {
                        calls: None,
                        tokens: None,
                    },
                    timeout: Duration::from_secs(2),
                },
            )?;
            assert!(factory.prepare(&store, &lease).is_err());
        }
        assert_eq!(server.received_requests().await.ok_or("requests")?.len(), 0);
        Ok(())
    }
}

#[cfg(test)]
mod run_tests {
    use super::*;
    use crate::{queue, store::vectors::VectorTableSpec};
    use std::{error::Error as StdError, fs};
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    async fn setup(
        server: &MockServer,
        calls: Option<i64>,
    ) -> Result<(tempfile::TempDir, Arc<Store>, Arc<Factory>), Box<dyn StdError>> {
        let (root, store, _) = queue::source::tests::fixture()?;
        fs::write(
            root.path().join(".deepkit/config.toml"),
            format!(
                "[index]\nrepo_name='fixture'\n[embedder]\nprofile='ollama'\ndims=2\n[embedder.ollama]\nurl='{}'\nconcurrency=2\n",
                server.uri()
            ),
        )?;
        let config = config::load(root.path(), None)?.config;
        let store = Arc::new(store);
        store.ensure_vector_table(VectorTableSpec {
            slug: "ollama_run_2",
            model: &config.embedder.ollama.model,
            profile: "ollama",
            dims: 2,
            created_at: 0,
            active: true,
        })?;
        for id in 1..=5 {
            if id > 1 {
                store.writer()?.execute("INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,created_at,updated_at)
                    SELECT ?1,randomblob(32),content_hash,'occurrence:' || ?1,file_id,kind,header,0,0 FROM units WHERE id=1",[id])?;
            }
            let input = {
                let mut writer = store.writer()?;
                let tx = writer.transaction()?;
                destination::prepare_file(&tx, root.path(), root.path(), &config, id)?
                    .ok_or("missing input")?
                    .0
            };
            queue::enqueue(&store, &input.enqueue(1.0), 0)?;
        }
        let factory = Factory::new(
            Arc::clone(&store),
            Settings {
                root: root.path().into(),
                config_root: root.path().into(),
                config,
                kind: UnitKind::Chunk,
                table_slug: "ollama_run_2".into(),
                budget: usage::Budget {
                    calls,
                    tokens: None,
                },
                timeout: Duration::from_secs(5),
            },
        )?;
        Ok((root, store, factory))
    }

    fn remote_factory(
        root: &std::path::Path,
        store: Arc<Store>,
    ) -> Result<Arc<Factory>, Box<dyn StdError>> {
        let config = config::load(root, None)?.config;
        let provider = OpenAiEmbedder::with_api_key(
            OpenAiConfig::from_remote(
                &config.embedder.remote,
                config.embedder.dims,
                Duration::from_secs(5),
            )?,
            "synthetic-test-key".into(),
        )?;
        let settings = Settings {
            root: root.into(),
            config_root: root.into(),
            config,
            kind: UnitKind::Chunk,
            table_slug: "remote_run_2".into(),
            budget: usage::Budget {
                calls: None,
                tokens: None,
            },
            timeout: Duration::from_secs(5),
        };
        let destination = Factory::validate(&store, &settings)?;
        Ok(Factory::from_http_provider(
            store,
            settings,
            destination,
            Provider::Remote(provider),
            "remote",
        )?)
    }

    async fn setup_remote(
        server: &MockServer,
    ) -> Result<(tempfile::TempDir, Arc<Store>, Arc<Factory>), Box<dyn StdError>> {
        let (root, store, _) = queue::source::tests::fixture()?;
        fs::write(
            root.path().join(".deepkit/config.toml"),
            format!(
                "[index]\nrepo_name='fixture'\n[embedder]\nprofile='remote'\ndims=2\n[embedder.remote]\nurl='{}'\nmodel='queue-restart-mrl-fixture'\napi_key_env='UNUSED_QUEUE_RESTART_KEY'\nmax_batch_items=1\nmax_batch_tokens=100\nconcurrency=1\n",
                server.uri()
            ),
        )?;
        let config = config::load(root.path(), None)?.config;
        let store = Arc::new(store);
        store.ensure_vector_table(VectorTableSpec {
            slug: "remote_run_2",
            model: &config.embedder.remote.model,
            profile: "remote",
            dims: 2,
            created_at: 0,
            active: true,
        })?;
        let input = {
            let mut writer = store.writer()?;
            let tx = writer.transaction()?;
            destination::prepare_file(&tx, root.path(), root.path(), &config, 1)?
                .ok_or("missing input")?
                .0
        };
        queue::enqueue(&store, &input.enqueue(1.0), 0)?;
        let factory = remote_factory(root.path(), Arc::clone(&store))?;
        Ok((root, store, factory))
    }
    fn limits(max_jobs: usize) -> RunLimits {
        RunLimits {
            max_jobs,
            heartbeat: Duration::from_secs(30),
        }
    }
    async fn wait_for_requests(server: &MockServer, count: usize) {
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if server.received_requests().await.unwrap().len() >= count {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
    }
    fn state_count(store: &Store, state: &str) -> Result<i64, Error> {
        Ok(store
            .writer()?
            .query_row("SELECT COUNT(*) FROM jobs WHERE state=?1", [state], |r| {
                r.get(0)
            })?)
    }

    fn batch_response(request: &wiremock::Request) -> ResponseTemplate {
        let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
        let count = body["input"].as_array().unwrap().len();
        ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "embeddings": vec![vec![1.0, 0.0]; count],
            "prompt_eval_count": count * 7,
        }))
    }

    #[tokio::test]
    async fn dimensions_rejection_survives_a_worker_process_restart()
    -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(|request: &wiremock::Request| {
                let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
                if body.get("dimensions").is_some() {
                    ResponseTemplate::new(400).set_body_raw(
                        include_str!("../../fixtures/openai_compat/matryoshka-rejection.json"),
                        "application/json",
                    )
                } else {
                    ResponseTemplate::new(200).set_body_raw(
                        include_str!("../../fixtures/openai_compat/embeddings-success.json"),
                        "application/json",
                    )
                }
            })
            .expect(2)
            .mount(&server)
            .await;
        let (root, store, first_factory) = setup_remote(&server).await?;
        let first = Arc::clone(&first_factory)
            .run_ready(limits(1), || 1)
            .await?;
        assert_eq!(first.claimed, 1);
        assert_eq!(first.published, 0);
        assert_eq!(first.deferred, 1);
        assert!(first.errors.is_empty());
        let capability = first_factory
            .dimensions_capability
            .as_deref()
            .ok_or("remote capability key")?;
        assert_eq!(store.get_meta(capability)?.as_deref(), Some("1"));
        match &first_factory.provider {
            Provider::Remote(provider) => provider.forget_dimensions_unsupported_for_test(),
            _ => return Err("remote provider expected".into()),
        }
        drop(first_factory);

        let restarted = remote_factory(root.path(), Arc::clone(&store))?;
        let second = restarted.run_ready(limits(1), || 1_000).await?;
        assert_eq!(second.claimed, 1);
        assert_eq!(second.published, 1);
        assert_eq!(second.deferred, 0);
        assert!(second.errors.is_empty());
        assert_eq!(store.count_vectors()?, 1);
        let requests = server.received_requests().await.ok_or("requests")?;
        assert_eq!(requests.len(), 2);
        let first_body: serde_json::Value = serde_json::from_slice(&requests[0].body)?;
        let second_body: serde_json::Value = serde_json::from_slice(&requests[1].body)?;
        assert_eq!(first_body["dimensions"], 2);
        assert!(second_body.get("dimensions").is_none());
        let (uncertain, reconciled, reserved_calls, settled_calls, attempts): (
            i64,
            i64,
            i64,
            i64,
            i64,
        ) = store.writer()?.query_row(
            "SELECT
                    (SELECT count(*) FROM usage_reservations WHERE state='dispatched'),
                    (SELECT count(*) FROM usage_reservations WHERE state='reconciled'),
                    (SELECT sum(calls) FROM usage_reservations),
                    (SELECT sum(calls) FROM usage),
                    (SELECT attempts FROM jobs WHERE target_id=1)",
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )?;
        assert_eq!(
            (
                uncertain,
                reconciled,
                reserved_calls,
                settled_calls,
                attempts
            ),
            (1, 1, 2, 1, 2)
        );
        server.verify().await;
        Ok(())
    }

    #[tokio::test]
    async fn hard_main_cap_rolls_back_vector_publication_and_pauses_without_losing_usage()
    -> Result<(), Box<dyn StdError>> {
        publication_capacity(false).await
    }

    #[tokio::test]
    async fn wal_reservation_rolls_back_http_publication_and_retains_actual_usage()
    -> Result<(), Box<dyn StdError>> {
        publication_capacity(true).await
    }

    async fn publication_capacity(wal: bool) -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(batch_response)
            .mount(&server)
            .await;
        let (root, store, _) = setup(&server, None).await?;
        let database_path = root.path().join(".deepkit/code.db");
        store
            .writer()?
            .execute("VACUUM INTO ?1", [database_path.to_string_lossy().as_ref()])?;
        let store = Arc::new(crate::store::open(
            &database_path,
            crate::store::StoreLimits::default(),
        )?);
        let hash: Vec<u8> =
            store
                .writer()?
                .query_row("SELECT content_hash FROM units WHERE id=1", [], |r| {
                    r.get(0)
                })?;
        store.insert_vector(VectorInsert {
            table_slug: "ollama_run_2",
            unit_id: 1,
            ctx_hash: &[0; 32],
            embedding_hash: &hash,
            embedding: &[0.0, 1.0],
        })?;
        let before: Vec<u8> = store.writer()?.query_row(
            "SELECT v FROM vec_ollama_run_2 WHERE unit_id=1",
            [],
            |r| r.get(0),
        )?;
        // Simulate unforeseen SQLite/index metadata growth after the vector write,
        // so a preflight-only implementation would publish beyond its byte cap.
        store.writer()?.execute_batch("CREATE TRIGGER force_growth AFTER UPDATE ON vec_ollama_run_2 BEGIN INSERT OR REPLACE INTO meta(key,value) VALUES('test_growth',zeroblob(16777216)); END;")?;
        let allocated: i64 = store.writer()?.query_row(
            "SELECT page_count*page_size FROM pragma_page_count(),pragma_page_size()",
            [],
            |r| r.get(0),
        )?;
        let cap = allocated * 100 / 90;
        let path = root.path().join(".deepkit/config.toml");
        let original = fs::read_to_string(&path)?;
        if !wal {
            fs::write(
                &path,
                format!("{original}\n[limits]\ndb_max_bytes='{cap} B'\n"),
            )?;
        }
        let factory = reload_factory(root.path(), Arc::clone(&store))?;
        let report = Arc::clone(&factory).run_ready(limits(1), || 1).await?;
        assert_eq!(report.deferred, 1);
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        assert_eq!(store.count_vectors()?, 1);
        assert_eq!(state_count(&store, "leased")?, 0);
        let after: Vec<u8> = store.writer()?.query_row(
            "SELECT v FROM vec_ollama_run_2 WHERE unit_id=1",
            [],
            |r| r.get(0),
        )?;
        assert_eq!(before, after);
        assert_eq!(
            store.writer()?.query_row(
                "SELECT COUNT(*) FROM meta WHERE key='test_growth'",
                [],
                |r| r.get::<_, i64>(0)
            )?,
            0
        );
        let snapshot = serde_json::to_value(queue::status::snapshot(&store, 1)?)?;
        let provider = &snapshot["providers"][&factory.provider_key];
        assert_eq!(provider["tokens_today"], 7);
        assert_eq!(provider["calls_today"], 1);
        assert_eq!(
            provider["deferrals"][if wal { "cap_wal" } else { "cap_storage" }]["jobs"],
            1
        );
        assert!(crate::store::wal::wal_bytes(&*store.writer()?)? <= store.wal_max_bytes);
        store
            .writer()?
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")?;
        assert!(fs::metadata(root.path().join(".deepkit/code.db"))?.len() <= cap as u64);
        store.writer()?.execute_batch("DROP TRIGGER force_growth")?;
        // Same cap now fits: next pass rechecks and publishes without resetting attempts.
        let report = factory.run_ready(limits(1), || 2).await?;
        assert_eq!(report.published, 1);
        assert!(report.errors.is_empty());
        let usage: (i64, i64) =
            store
                .writer()?
                .query_row("SELECT calls,tokens FROM usage", [], |r| {
                    Ok((r.get(0)?, r.get(1)?))
                })?;
        assert_eq!(usage, (2, 14));
        assert_eq!(
            store
                .writer()?
                .query_row("SELECT attempts FROM jobs WHERE target_id=1", [], |r| r
                    .get::<_, i64>(0))?,
            2
        );
        assert_eq!(server.received_requests().await.ok_or("requests")?.len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn changed_wal_configuration_requires_reopening_store_before_provider_work()
    -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start().await;
        let (root, store, _) = setup(&server, None).await?;
        let path = root.path().join(".deepkit/config.toml");
        let original = fs::read_to_string(&path)?;
        fs::write(
            &path,
            format!("{original}\n[limits]\nwal_max_bytes='32 MiB'\n"),
        )?;
        assert!(reload_factory(root.path(), store).is_err());
        assert_eq!(server.received_requests().await.ok_or("requests")?.len(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn configured_batch_and_pass_bound_apply_to_real_http_workers()
    -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &wiremock::Request| {
                batch_response(request).set_delay(Duration::from_millis(500))
            })
            .expect(1)
            .mount(&server)
            .await;
        let (_root, store, factory) = setup(&server, None).await?;
        let observer = async {
            wait_for_requests(&server, 1).await;
            assert_eq!(state_count(&store, "leased").unwrap(), 3);
            assert_eq!(state_count(&store, "queued").unwrap(), 2);
        };
        let (report, ()) = tokio::join!(factory.run_ready(limits(3), || 1), observer);
        let report = report?;
        assert_eq!(report.claimed, 3);
        assert_eq!(report.published, 3);
        assert!(report.errors.is_empty());
        assert_eq!(store.count_vectors()?, 3);
        assert_eq!(state_count(&store, "leased")?, 0);
        assert_eq!(state_count(&store, "queued")?, 2);
        Ok(())
    }

    #[tokio::test]
    async fn one_daily_call_budget_admits_one_multi_input_request() -> Result<(), Box<dyn StdError>>
    {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(batch_response)
            .expect(1)
            .mount(&server)
            .await;
        let (_root, store, factory) = setup(&server, Some(1)).await?;
        let report = factory.run_ready(limits(5), || 1).await?;
        assert_eq!(report.published, 5);
        assert_eq!(report.deferred, 0);
        assert!(report.errors.is_empty());
        assert_eq!(state_count(&store, "leased")?, 0);
        let (done, attempts): (i64, i64) = store.writer()?.query_row(
            "SELECT COUNT(*),SUM(attempts) FROM jobs WHERE state='done'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        assert_eq!((done, attempts), (5, 5));
        assert_eq!(store.count_vectors()?, 5);
        assert_eq!(server.received_requests().await.ok_or("requests")?.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn cancelled_http_pass_recovers_without_prefetch_or_duplicate_vectors()
    -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &wiremock::Request| {
                batch_response(request).set_delay(Duration::from_millis(500))
            })
            .mount(&server)
            .await;
        let (_root, store, factory) = setup(&server, None).await?;
        let task = tokio::spawn(Arc::clone(&factory).run_ready(limits(5), || 1));
        wait_for_requests(&server, 1).await;
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert_eq!(state_count(&store, "leased")?, 5);
        assert_eq!(state_count(&store, "queued")?, 0);
        assert_eq!(store.count_vectors()?, 0);
        let uncertain: i64 = store.writer()?.query_row(
            "SELECT COUNT(*) FROM usage_reservations WHERE state='dispatched'",
            [],
            |r| r.get(0),
        )?;
        assert_eq!(uncertain, 1);
        let report = factory.run_ready(limits(5), || 122).await?;
        assert_eq!(report.published, 5);
        assert!(report.errors.is_empty());
        assert_eq!(store.count_vectors()?, 5);
        assert_eq!(state_count(&store, "leased")?, 0);
        assert_eq!(server.received_requests().await.ok_or("requests")?.len(), 2);
        let uncertain: i64 = store.writer()?.query_row(
            "SELECT COUNT(*) FROM usage_reservations WHERE state='dispatched'",
            [],
            |r| r.get(0),
        )?;
        assert_eq!(uncertain, 1);
        Ok(())
    }
    #[tokio::test]
    async fn capped_pass_releases_claims_and_next_pass_rechecks_capacity()
    -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(batch_response)
            .expect(1)
            .mount(&server)
            .await;
        let (root, store, original) = setup(&server, None).await?;
        let path = root.path().join(".deepkit/config.toml");
        let original_text = fs::read_to_string(&path)?;
        fs::write(
            &path,
            format!("{original_text}\n[limits]\nmax_l2_units=0\n"),
        )?;
        let config = config::load(root.path(), None)?.config;
        let capped = Factory::new(
            Arc::clone(&store),
            Settings {
                root: root.path().into(),
                config_root: root.path().into(),
                config,
                kind: UnitKind::Chunk,
                table_slug: "ollama_run_2".into(),
                budget: usage::Budget {
                    calls: None,
                    tokens: None,
                },
                timeout: Duration::from_secs(5),
            },
        )?;
        for pass in 0..2 {
            let report = Arc::clone(&capped).run_ready(limits(5), || 1).await?;
            assert_eq!(report.deferred, if pass == 0 { 5 } else { 0 });
            assert_eq!(report.claimed, if pass == 0 { 5 } else { 0 });
            assert_eq!(
                store
                    .writer()?
                    .query_row("SELECT sum(lease_generation) FROM jobs", [], |r| r
                        .get::<_, i64>(0))?,
                5,
                "the second pass must not reacquire unchanged capacity pauses"
            );
            assert!(report.errors.is_empty());
            assert_eq!(state_count(&store, "leased")?, 0);
            assert_eq!(server.received_requests().await.ok_or("requests")?.len(), 0);
            let status = queue::status::snapshot(&store, 1)?;
            assert!(
                status
                    .summary()
                    .contains("max_l2_units cap; rechecked on the next provider run")
            );
            let json = serde_json::to_value(status)?;
            assert!(json["jobs"]["next_ready_at"].is_null());
            assert_eq!(
                json["providers"][&capped.provider_key]["deferrals"]["cap_vectors"]["jobs"],
                5
            );
            assert!(
                json["providers"][&capped.provider_key]["deferrals"]["cap_vectors"]["earliest_at"]
                    .is_null()
            );
        }
        fs::write(&path, original_text)?;
        let report = original.run_ready(limits(5), || 2).await?;
        assert_eq!(report.published, 5);
        assert!(report.errors.is_empty());
        assert_eq!(store.count_vectors()?, 5);
        assert_eq!(state_count(&store, "leased")?, 0);
        let attempts: i64 =
            store
                .writer()?
                .query_row("SELECT SUM(attempts) FROM jobs", [], |r| r.get(0))?;
        assert_eq!(attempts, 5);
        Ok(())
    }
    #[tokio::test]
    async fn capacity_reached_during_http_pauses_after_accounting_without_overfill()
    -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &wiremock::Request| {
                batch_response(request).set_delay(Duration::from_millis(250))
            })
            .expect(1)
            .mount(&server)
            .await;
        let (root, store, _original) = setup(&server, None).await?;
        let path = root.path().join(".deepkit/config.toml");
        let text = fs::read_to_string(&path)?;
        fs::write(&path, format!("{text}\n[limits]\nmax_l2_units=1\n"))?;
        let config = config::load(root.path(), None)?.config;
        let factory = Factory::new(
            Arc::clone(&store),
            Settings {
                root: root.path().into(),
                config_root: root.path().into(),
                config,
                kind: UnitKind::Chunk,
                table_slug: "ollama_run_2".into(),
                budget: usage::Budget {
                    calls: None,
                    tokens: None,
                },
                timeout: Duration::from_secs(5),
            },
        )?;
        let report = factory.run_ready(limits(5), || 1).await?;
        assert_eq!(report.published, 1);
        assert_eq!(report.deferred, 4);
        assert!(report.errors.is_empty());
        assert_eq!(store.count_vectors()?, 1);
        assert_eq!(state_count(&store, "leased")?, 0);
        let calls: i64 = store
            .writer()?
            .query_row("SELECT SUM(calls) FROM usage", [], |r| r.get(0))?;
        assert_eq!(calls, 1);
        let attempts: i64 =
            store
                .writer()?
                .query_row("SELECT SUM(attempts) FROM jobs", [], |r| r.get(0))?;
        assert_eq!(attempts, 5);
        Ok(())
    }
    #[tokio::test]
    async fn l2_growth_above_eighty_five_and_replacement_above_ninety_five_succeed()
    -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(batch_response)
            .expect(2)
            .mount(&server)
            .await;
        for replacement in [false, true] {
            let (root, store, _original) = setup(&server, None).await?;
            if replacement {
                let hash: Vec<u8> = store.writer()?.query_row(
                    "SELECT content_hash FROM units WHERE id=1",
                    [],
                    |r| r.get(0),
                )?;
                store.insert_vector(VectorInsert {
                    table_slug: "ollama_run_2",
                    unit_id: 1,
                    ctx_hash: &[0; 32],
                    embedding_hash: &hash,
                    embedding: &[0.0, 1.0],
                })?;
            }
            let allocated: i64 = store.writer()?.query_row(
                "SELECT page_count*page_size FROM pragma_page_count(),pragma_page_size()",
                [],
                |r| r.get(0),
            )?;
            let cap = allocated * 100 / if replacement { 96 } else { 90 };
            let path = root.path().join(".deepkit/config.toml");
            let text = fs::read_to_string(&path)?;
            fs::write(&path, format!("{text}\n[limits]\ndb_max_bytes='{cap} B'\n"))?;
            let config = config::load(root.path(), None)?.config;
            let factory = Factory::new(
                Arc::clone(&store),
                Settings {
                    root: root.path().into(),
                    config_root: root.path().into(),
                    config,
                    kind: UnitKind::Chunk,
                    table_slug: "ollama_run_2".into(),
                    budget: usage::Budget {
                        calls: None,
                        tokens: None,
                    },
                    timeout: Duration::from_secs(5),
                },
            )?;
            let report = factory.run_ready(limits(1), || 1).await?;
            assert_eq!(report.published, 1);
            assert!(report.errors.is_empty());
            assert_eq!(store.count_vectors()?, 1);
        }
        Ok(())
    }
    fn reload_factory(root: &std::path::Path, store: Arc<Store>) -> Result<Arc<Factory>, Error> {
        let config = config::load(root, None)
            .map_err(|_| Error::Invalid("test config"))?
            .config;
        Factory::new(
            store,
            Settings {
                root: root.into(),
                config_root: root.into(),
                config,
                kind: UnitKind::Chunk,
                table_slug: "ollama_run_2".into(),
                budget: usage::Budget {
                    calls: None,
                    tokens: None,
                },
                timeout: Duration::from_secs(5),
            },
        )
    }

    #[tokio::test]
    async fn role_only_scheduling_requires_proof_and_honors_explicit_refresh()
    -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start().await;
        for legacy in [false, true] {
            let (_root, store, factory) = setup(&server, None).await?;
            store.writer()?.execute("DELETE FROM jobs", [])?;
            let prepared = store
                .read_snapshot(|tx| factory.resolve_current(tx, 1))??
                .ok_or("current input")?
                .0;
            store.insert_vector(VectorInsert {
                table_slug: "ollama_run_2",
                unit_id: 1,
                ctx_hash: &[0; 32],
                embedding_hash: &prepared.current().target_hash,
                embedding: &[0.0, 1.0],
            })?;
            store.writer()?.execute(
                "INSERT INTO embedding_publications(unit_id,table_slug,desired_hash,role_free_hash) VALUES(1,'ollama_run_2',?1,?2)",
                rusqlite::params![prepared.role_free_hash(), (!legacy).then_some(prepared.role_free_hash())],
            )?;
            assert_eq!(
                factory.schedule_file(1, 0.5, queue::priority::Recency::ChangedFile, || 1)?,
                None
            );
            assert_eq!(store.count_vectors()?, 1);
            assert_eq!(state_count(&store, "queued")?, 0);
            let id = factory
                .schedule_context(1, 0.5, queue::priority::Recency::ChangedFile, true, || 2)?
                .ok_or("explicit refresh")?;
            let priority: f64 =
                store
                    .writer()?
                    .query_row("SELECT priority FROM jobs WHERE id=?1", [id], |r| r.get(0))?;
            assert_eq!(priority, 0.0);
            store.writer()?.execute("DELETE FROM jobs", [])?;
            store.writer()?.execute(
                "UPDATE embedding_publications SET desired_hash=?1,role_free_hash=NULL",
                [[9u8; 32].as_slice()],
            )?;
            assert!(
                factory
                    .schedule_file(1, 0.5, queue::priority::Recency::ChangedFile, || 3)?
                    .is_some()
            );
        }
        assert!(
            server
                .received_requests()
                .await
                .ok_or("requests")?
                .is_empty()
        );
        Ok(())
    }

    #[tokio::test]
    async fn oversized_saved_jobs_pause_without_dispatch_and_wait_for_a_larger_cap()
    -> Result<(), Box<dyn StdError>> {
        for batch_cap in [false, true] {
            let server = MockServer::start().await;
            let (root, store, _) = setup(&server, None).await?;
            let config_path = root.path().join(".deepkit/config.toml");
            let original = fs::read_to_string(&config_path)?;
            let limited = if batch_cap {
                original
                    .replace("dims=2", "dims=2\nmax_doc_tokens=1")
                    .replace("concurrency=2", "concurrency=2\nmax_batch_items=1")
            } else {
                original.replace("dims=2", "dims=2\nmax_doc_tokens=1")
            };
            fs::write(&config_path, limited)?;
            let factory = reload_factory(root.path(), Arc::clone(&store))?;
            let first = Arc::clone(&factory).run_ready(limits(10), || 1).await?;
            assert_eq!(
                first.claimed, 5,
                "oversized estimates must reach capacity checks"
            );
            assert_eq!(first.deferred, 5, "{first:?}");
            assert_eq!(first.published, 0);
            assert!(first.errors.is_empty());
            for now in 2..=3 {
                let next = Arc::clone(&factory)
                    .run_ready(limits(10), move || now)
                    .await?;
                assert_eq!(
                    next.claimed, 0,
                    "unchanged cap must not revive blocked jobs"
                );
                assert!(next.errors.is_empty());
            }
            assert_eq!(store.writer()?.query_row(
                "SELECT count(*) FROM jobs WHERE state='queued' AND not_before=?1 AND last_error=?2 AND attempts=0 AND lease_generation=1",
                rusqlite::params![queue::capacity::PAUSED, queue::capacity::Reason::Tokens.message()],
                |r| r.get::<_, i64>(0),
            )?, 5);
            assert_eq!(
                store
                    .writer()?
                    .query_row("SELECT count(*) FROM usage_reservations", [], |r| r
                        .get::<_, i64>(0))?,
                0
            );
            assert!(
                server
                    .received_requests()
                    .await
                    .ok_or("requests")?
                    .is_empty()
            );
            Mock::given(method("POST"))
                .and(path("/api/embed"))
                .respond_with(batch_response)
                .expect(1)
                .mount(&server)
                .await;
            fs::write(&config_path, original)?;
            let restored = reload_factory(root.path(), Arc::clone(&store))?;
            let final_pass = restored.run_ready(limits(10), || 4).await?;
            assert_eq!(final_pass.published, 5);
            assert!(final_pass.errors.is_empty());
            assert_eq!(state_count(&store, "done")?, 5);
            assert_eq!(store.count_vectors()?, 5);
        }
        Ok(())
    }

    #[tokio::test]
    async fn scheduling_caps_are_visible_before_any_lease_or_request()
    -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start().await;
        for (extra, reason) in [
            ("[limits]\nmax_l2_units=0\n", "cap_vectors"),
            ("[limits]\ndb_max_bytes='1 B'\n", "cap_database"),
            ("", "cap_tokens"),
        ] {
            let (root, store, _) = setup(&server, None).await?;
            store.writer()?.execute("DELETE FROM jobs", [])?;
            // A ready-then-pause implementation must fail this insertion guard.
            store.writer()?.execute_batch(&format!("CREATE TRIGGER admission_guard BEFORE INSERT ON jobs WHEN NEW.not_before != {} BEGIN SELECT RAISE(ABORT,'ready before admission'); END", queue::capacity::PAUSED))?;
            let path = root.path().join(".deepkit/config.toml");
            let text = fs::read_to_string(&path)?;
            let text = if reason == "cap_tokens" {
                text.replace("dims=2", "dims=2\nmax_doc_tokens=1")
            } else {
                format!("{text}\n{extra}")
            };
            fs::write(&path, text)?;
            let factory = reload_factory(root.path(), Arc::clone(&store))?;
            let id = factory
                .schedule_file(1, 0.5, queue::priority::Recency::ChangedFile, || 1)?
                .ok_or("not scheduled")?;
            assert_eq!(
                factory.schedule_file(1, 0.9, queue::priority::Recency::ChangedFile, || 2)?,
                if reason == "cap_vectors" {
                    None
                } else {
                    Some(id)
                }
            );
            assert_eq!(
                store
                    .writer()?
                    .query_row("SELECT count(*) FROM jobs WHERE id=?1", [id], |r| r
                        .get::<_, i64>(0))?,
                1,
                "skipping preparation must retain the original visible pause"
            );
            let status = serde_json::to_value(queue::status::snapshot(&store, 2)?)?;
            assert_eq!(
                status["providers"][&factory.provider_key]["deferrals"][reason]["jobs"],
                1
            );
            assert!(
                queue::claim(
                    &store,
                    &factory.provider_key,
                    &queue::Owner::new(&store)?,
                    2,
                    5,
                    10000
                )?
                .is_empty()
            );
            let untouched: i64 = store.writer()?.query_row("SELECT COUNT(*) FROM jobs WHERE attempts=0 AND lease_generation=0 AND lease_owner IS NULL", [], |r| r.get(0))?;
            assert_eq!(untouched, 1);
            let charges: i64 =
                store
                    .writer()?
                    .query_row("SELECT COUNT(*) FROM usage_reservations", [], |r| r.get(0))?;
            assert_eq!(charges, 0);
        }
        assert!(
            server
                .received_requests()
                .await
                .ok_or("requests")?
                .is_empty()
        );
        Ok(())
    }

    #[tokio::test]
    async fn scheduling_replacement_at_caps_runs_and_reuses_done_generation()
    -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]]})),
            )
            .expect(1)
            .mount(&server)
            .await;
        let (root, store, _) = setup(&server, None).await?;
        store.writer()?.execute("DELETE FROM jobs", [])?;
        let hash: Vec<u8> =
            store
                .writer()?
                .query_row("SELECT content_hash FROM units WHERE id=1", [], |r| {
                    r.get(0)
                })?;
        store.insert_vector(VectorInsert {
            table_slug: "ollama_run_2",
            unit_id: 1,
            ctx_hash: &[0; 32],
            embedding_hash: &hash,
            embedding: &[0.0, 1.0],
        })?;
        let path = root.path().join(".deepkit/config.toml");
        fs::write(
            &path,
            format!(
                "{}\n[limits]\nmax_l2_units=1\ndb_max_bytes='1 B'\n",
                fs::read_to_string(&path)?
            ),
        )?;
        let factory = reload_factory(root.path(), Arc::clone(&store))?;
        let id = factory
            .schedule_file(1, 0.5, queue::priority::Recency::ChangedFile, || 1)?
            .ok_or("not scheduled")?;
        let report = Arc::clone(&factory).run_ready(limits(1), || 2).await?;
        assert_eq!(report.published, 1);
        assert!(report.errors.is_empty());
        assert_eq!(
            factory.schedule_file(1, 1.0, queue::priority::Recency::ChangedFile, || 3)?,
            Some(id)
        );
        assert_eq!(state_count(&store, "done")?, 1);
        assert_eq!(store.count_vectors()?, 1);
        assert_eq!(factory.run_ready(limits(1), || 3).await?.claimed, 0);
        Ok(())
    }

    #[tokio::test]
    async fn scheduling_rejects_stale_sources_and_rolls_back_failed_insert()
    -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start().await;
        let (_root, store, factory) = setup(&server, None).await?;
        store.writer()?.execute("DELETE FROM jobs", [])?;
        assert!(
            factory
                .schedule_file(1, 0.5, queue::priority::Recency::ChangedFile, || -1)
                .is_err()
        );
        assert!(
            factory
                .schedule_file(1, f64::NAN, queue::priority::Recency::ChangedFile, || 1)
                .is_err()
        );
        store.writer()?.execute_batch("CREATE TRIGGER fail_insert AFTER INSERT ON jobs BEGIN SELECT RAISE(ABORT,'test failure'); END")?;
        assert!(
            factory
                .schedule_file(1, 0.5, queue::priority::Recency::ChangedFile, || 1)
                .is_err()
        );
        assert_eq!(state_count(&store, "queued")?, 0);
        store
            .writer()?
            .execute_batch("DROP TRIGGER fail_insert; UPDATE files SET l1_pending=1")?;
        assert_eq!(
            factory.schedule_file(1, 0.5, queue::priority::Recency::ChangedFile, || 1)?,
            None
        );
        assert_eq!(state_count(&store, "queued")?, 0);
        assert!(
            server
                .received_requests()
                .await
                .ok_or("requests")?
                .is_empty()
        );
        Ok(())
    }
}

#[cfg(test)]
mod local_tests;
