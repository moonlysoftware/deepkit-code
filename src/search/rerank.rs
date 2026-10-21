//! Accounted whole-set local and remote reranking and dispatch provenance; v1 §§5, 8.2, 9.
#![expect(
    dead_code,
    reason = "T1.8 pipeline and public host integration follows"
)]

use super::admission;
use crate::{
    config::{self, Config},
    models::{self, Reranker, attempt::AttemptError, llama::rerank::Local, rerank::remote::Remote},
    queue::{
        self,
        source::{self, RerankInput},
        usage,
        worker::Response,
    },
    store::{self, Store, StoredUnit},
};
use async_trait::async_trait;
use rusqlite::Transaction;
use std::{collections::BTreeSet, path::PathBuf, sync::Arc, time::Duration};
use tokio::time::Instant;

/// Explicit host limits avoid inventing undocumented daily or rerank rate defaults.
#[derive(Clone)]
pub(crate) struct Settings {
    pub(crate) config_root: PathBuf,
    pub(crate) config: Config,
    pub(crate) limits: usage::Limits,
}

/// A failed attempt leaves the caller's complete original RRF window untouched.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("reranking refused by current source or provenance policy")]
    Privacy,
    #[error("invalid reranking input or configuration")]
    Input,
    #[error("rerank deadline exceeded")]
    Deadline,
    #[error("rerank preparation task failed")]
    Task,
    #[error(transparent)]
    Download(#[from] models::download::Error),
    #[error(transparent)]
    Source(#[from] queue::Error),
    #[error(transparent)]
    Model(#[from] models::Error),
    #[error(transparent)]
    Admission(#[from] admission::Error),
}

/// Reusable scorer bound to a store, configuration and provider ledger.
/// No private inputs or credentials are included in Debug or serialized output.
pub(crate) struct Factory {
    store: Arc<Store>,
    settings: Settings,
    root: PathBuf,
    scorer: Scorer,
    cache: Option<PathBuf>,
    auto: Arc<auto::Control>,
    endpoint: String,
    provider: String,
}
enum Scorer {
    Remote(Remote),
    Local(Local),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Selection {
    Remote,
    Local,
}

fn selection(config: &Config) -> Result<Selection, Error> {
    if !matches!(config.rerank.mode.as_str(), "auto" | "on")
        || config.limits.rerank_timeout_ms == 0
        || config.rerank.candidates == 0
        || config.rerank.max_doc_chars == 0
    {
        return Err(Error::Input);
    }
    match config.rerank.provider.as_str() {
        "remote" => Ok(Selection::Remote),
        "local" => Ok(Selection::Local),
        "auto" => match config.embedder.profile.as_str() {
            "remote" => Ok(Selection::Remote),
            "local" | "ollama" => Ok(Selection::Local),
            _ => Err(Error::Input),
        },
        _ => Err(Error::Input),
    }
}

impl Factory {
    /// Hosts the configured scorer once, loading only verified cached local weights.
    /// Callers provide explicit admission limits and own any download workflow.
    pub(crate) async fn new_configured(
        store: Arc<Store>,
        settings: Settings,
        cache: PathBuf,
    ) -> Result<Arc<Self>, Error> {
        match selection(&settings.config)? {
            Selection::Local => Self::new_local(store, settings, cache).await,
            Selection::Remote => tokio::task::spawn_blocking(move || Self::new(store, settings))
                .await
                .map_err(|_| Error::Task)?,
        }
    }

    /// Reports the live adaptive bound and whether its reduction was saved.
    pub(crate) fn auto_status(&self) -> auto::Status {
        self.auto.snapshot()
    }

    /// Prevents a candidate window from using another store or effective configuration.
    pub(crate) fn matches(&self, store: &Arc<Store>, config: &Config) -> bool {
        Arc::ptr_eq(&self.store, store) && self.settings.config == *config
    }

    /// Prepares only a configured remote reranker; resolves keys through the shared
    /// environment/0600-file contract. Local hosting uses a separate preparation path.
    pub(crate) fn new(store: Arc<Store>, settings: Settings) -> Result<Arc<Self>, Error> {
        Self::build(store, settings, Remote::from_config)
    }

    fn build(
        store: Arc<Store>,
        settings: Settings,
        construct: impl FnOnce(&config::Rerank, Duration) -> models::Result<Remote>,
    ) -> Result<Arc<Self>, Error> {
        let config = &settings.config;
        if selection(config)? != Selection::Remote {
            return Err(Error::Input);
        }
        let base = reqwest::Url::parse(&config.rerank.remote.url).map_err(|_| Error::Input)?;
        if !matches!(base.scheme(), "http" | "https")
            || base.host_str().is_none()
            || !base.username().is_empty()
            || base.password().is_some()
            || base.query().is_some()
            || base.fragment().is_some()
        {
            return Err(Error::Input);
        }
        let endpoint = format!("{}/rerank", base.as_str().trim_end_matches('/'));
        let provider = format!(
            "rerank:remote:{}",
            blake3::hash(endpoint.as_bytes()).to_hex()
        );
        let auto = auto::Control::new(
            Arc::clone(&store),
            settings.config_root.clone(),
            config.clone(),
        )?;
        let remote = construct(
            &config.rerank,
            Duration::from_millis(config.limits.rerank_timeout_ms),
        )?;
        Ok(Arc::new(Self {
            root: settings.config_root.join(&config.index.root),
            settings,
            store,
            scorer: Scorer::Remote(remote),
            cache: None,
            auto,
            endpoint,
            provider,
        }))
    }

    /// Loads only the selected registered local Rank model from verified cache.
    /// Local and Ollama auto profiles use this path; unused remote settings and keys
    /// are not inspected. Native smoke, idle unload/reload and cancellation remain
    /// owned by the model adapter. Lazy downloads are supplied by the public host.
    pub(crate) async fn new_local(
        store: Arc<Store>,
        settings: Settings,
        cache: PathBuf,
    ) -> Result<Arc<Self>, Error> {
        if selection(&settings.config)? != Selection::Local {
            return Err(Error::Input);
        }
        let idle = settings
            .config
            .models
            .unload_after_idle()
            .map_err(|_| Error::Input)?;
        let loaded =
            Local::load_registered(cache.clone(), settings.config.rerank.clone(), idle).await;
        let local_cache = cache.clone();
        let health_store = Arc::clone(&store);
        let health_settings = settings.clone();
        let (state, detail) = match &loaded {
            Ok(_) => (
                models::rerank::health::State::Verified,
                "artifact verified; three-pair smoke passed".to_owned(),
            ),
            Err(models::Error::Rerank(models::rerank::Error::Smoke { file })) => (
                models::rerank::health::State::Disabled,
                format!("smoke failed; disabled model {file}"),
            ),
            Err(error) => (
                models::rerank::health::State::Unavailable,
                error.to_string(),
            ),
        };
        let saved = tokio::task::spawn_blocking(move || {
            let record = models::rerank::health::Record::new(
                &health_settings.config,
                &cache,
                state,
                detail,
            )?;
            models::rerank::health::write(
                &health_store,
                &health_settings.config_root,
                &health_settings.config,
                &record,
            )
        })
        .await;
        // Load failures retain their native diagnosis even when status storage is capped.
        if !matches!(saved, Ok(Ok(true))) {
            tracing::warn!("local reranker verification status could not be saved");
        }
        let local = loaded?;
        let control_store = Arc::clone(&store);
        let control_settings = settings.clone();
        let auto = tokio::task::spawn_blocking(move || {
            auto::Control::new(
                control_store,
                control_settings.config_root,
                control_settings.config,
            )
        })
        .await
        .map_err(|_| Error::Task)??;
        Ok(Arc::new(Self {
            root: settings.config_root.join(&settings.config.index.root),
            settings,
            store,
            scorer: Scorer::Local(local),
            cache: Some(local_cache),
            auto,
            endpoint: String::new(),
            provider: "rerank:local".into(),
        }))
    }

    /// Captures every selected occurrence and canonical context in one snapshot.
    /// Any unproven hit refuses the whole set, including lexical-only candidates.
    /// The caller retains its hits; this API cannot drop protected results or send
    /// a partially authorized set. All filesystem/SQL work runs off the async runtime.
    pub(crate) async fn prepare(
        self: &Arc<Self>,
        query: String,
        units: Vec<StoredUnit>,
        deadline: Instant,
    ) -> Result<Prepared, Error> {
        if query.trim().is_empty()
            || units.len() as u128 > u128::from(self.settings.config.rerank.candidates)
            || units
                .iter()
                .map(|unit| unit.id)
                .collect::<BTreeSet<_>>()
                .len()
                != units.len()
        {
            return Err(Error::Input);
        }
        let factory = Arc::clone(self);
        tokio::time::timeout_at(
            deadline,
            tokio::task::spawn_blocking(move || {
                let inputs = factory
                    .store
                    .read_snapshot(|tx| {
                        let mut inputs = Vec::with_capacity(units.len());
                        for unit in &units {
                            if Instant::now() >= deadline {
                                return Err(Error::Deadline);
                            }
                            inputs.push(factory.resolve(tx, unit)?.ok_or(Error::Privacy)?);
                        }
                        Ok::<_, Error>(inputs)
                    })
                    .map_err(queue::Error::from)??;
                if Instant::now() >= deadline {
                    return Err(Error::Deadline);
                }
                // Query is paired with every document by the cross encoder. Missing
                // provider usage remains an explicit estimate, never an actual count.
                let tokenizer = tiktoken_rs::cl100k_base_singleton();
                let query_tokens = tokenizer.encode_with_special_tokens(&query).len();
                let tokens = inputs.iter().try_fold(0_i64, |sum, input| {
                    let count = tokenizer
                        .encode_with_special_tokens(&input.document)
                        .len()
                        .checked_add(query_tokens)
                        .and_then(|count| count.checked_add(1))
                        .and_then(|count| i64::try_from(count).ok())
                        .ok_or(Error::Input)?;
                    sum.checked_add(count).ok_or(Error::Input)
                })?;
                if Instant::now() >= deadline {
                    return Err(Error::Deadline);
                }
                Ok(Prepared {
                    factory,
                    query,
                    units,
                    inputs,
                    tokens,
                    deadline,
                    started: Instant::now(),
                })
            }),
        )
        .await
        .map_err(|_| Error::Deadline)?
        .map_err(|_| Error::Task)?
    }

    fn resolve(
        &self,
        tx: &Transaction<'_>,
        unit: &StoredUnit,
    ) -> Result<Option<RerankInput>, queue::Error> {
        if store::unit_by_id_in(tx, unit.id)?.as_ref() != Some(unit) {
            return Ok(None);
        }
        source::resolve_rerank(
            tx,
            &self.root,
            &self.settings.config_root,
            &self.settings.config,
            unit.id,
        )
    }
}

/// Immutable complete candidate set. Its private proof and documents cannot be
/// replaced independently, or dispatched with a different factory or store.
pub(crate) struct Prepared {
    factory: Arc<Factory>,
    query: String,
    units: Vec<StoredUnit>,
    inputs: Vec<RerankInput>,
    tokens: i64,
    deadline: Instant,
    started: Instant,
}
impl Prepared {
    /// Makes one accounted call after revalidating every full context and source.
    /// A provider timeout, refusal or error leaves RRF fallback to the pipeline;
    /// uncertain calls retain their charge and are never replayed here.
    pub(crate) async fn score<C>(mut self, deadline: Instant, clock: C) -> Result<Vec<f32>, Error>
    where
        C: Fn() -> i64 + Send + Sync + 'static,
    {
        if Instant::now() >= deadline.min(self.deadline) {
            return Err(Error::Deadline);
        }
        if self.units.is_empty() {
            return Ok(Vec::new());
        }
        let started = Instant::now();
        let deadline = deadline.min(self.deadline).min(
            started
                .checked_add(Duration::from_millis(
                    self.factory.settings.config.limits.rerank_timeout_ms,
                ))
                .ok_or(Error::Input)?,
        );
        self.deadline = deadline;
        self.started = started;
        let result = admission::execute(
            Arc::clone(&self.factory.store),
            self.factory.provider.clone(),
            self.factory.settings.limits.budget,
            deadline,
            Arc::new(self),
            clock,
        )
        .await;
        match result {
            Err(admission::Error::Refused(usage::Dispatch::Superseded)) => {
                if Instant::now() >= deadline {
                    Err(Error::Deadline)
                } else {
                    Err(Error::Privacy)
                }
            }
            Err(admission::Error::Deadline)
            | Err(admission::Error::Provider(AttemptError::Rerank(
                models::rerank::Error::Timeout,
            ))) => Err(Error::Deadline),
            other => other.map_err(Error::Admission),
        }
    }
}

#[async_trait]
impl admission::Work for Prepared {
    type Output = Vec<f32>;
    fn tokens(&self) -> i64 {
        self.tokens
    }
    fn endpoint(&self) -> Option<usage::Endpoint> {
        if matches!(self.factory.scorer, Scorer::Local(_)) {
            return None;
        }
        Some(usage::Endpoint {
            kind: "rerank",
            url: self.factory.endpoint.clone(),
            units: self.units.len() as i64,
        })
    }
    fn limits(
        &self,
        tx: &Transaction<'_>,
        provider: &str,
    ) -> Result<Option<usage::Limits>, queue::Error> {
        if Instant::now() >= self.deadline
            || provider != self.factory.provider
            || !self.factory.store.checked_database_is_current()
        {
            return Ok(None);
        }
        for (unit, prepared) in self.units.iter().zip(&self.inputs) {
            if Instant::now() >= self.deadline {
                return Ok(None);
            }
            let Some(current) = self.factory.resolve(tx, unit)? else {
                return Ok(None);
            };
            let old = &prepared.file;
            let new = &current.file;
            if old.context != new.context
                || old.occurrence.id != new.occurrence.id
                || old.occurrence.content_hash != new.occurrence.content_hash
                || old.occurrence.kind != new.occurrence.kind
                || old.policy.fingerprint != new.policy.fingerprint
                || old.policy.max_level != new.policy.max_level
                || old.policy.never_send != new.policy.never_send
                || prepared.document != current.document
            {
                return Ok(None);
            }
        }
        if Instant::now() >= self.deadline {
            return Ok(None);
        }
        // A config edit while checking the last source must not authorize dispatch.
        let current = config::load(&self.factory.settings.config_root, None)
            .map_err(|_| queue::Error::Invalid("rerank configuration reload"))?
            .config;
        Ok((current == self.factory.settings.config
            && self.factory.store.checked_database_is_current())
        .then_some(self.factory.settings.limits))
    }
    async fn call(&self) -> Result<Response<Vec<f32>>, AttemptError> {
        let docs = self
            .inputs
            .iter()
            .map(|input| input.document.as_str())
            .collect::<Vec<_>>();
        let observation = self.factory.auto.start(docs.len(), self.started);
        let result = match &self.factory.scorer {
            Scorer::Remote(remote) => remote
                .score_once(&self.query, &docs)
                .await
                .map_err(AttemptError::Rerank),
            Scorer::Local(local) => {
                local
                    .score(&self.query, &docs)
                    .await
                    .map_err(|error| match error {
                        models::Error::Rerank(error) => AttemptError::Rerank(error),
                        other => AttemptError::LocalRerank(Box::new(other)),
                    })
            }
        };
        if matches!(&self.factory.scorer, Scorer::Local(_))
            && let Err(AttemptError::Rerank(models::rerank::Error::Smoke { file })) = &result
        {
            let factory = Arc::clone(&self.factory);
            let file = file.clone();
            tokio::task::spawn_blocking(move || {
                use models::rerank::health::{self, State};
                let saved = (|| -> Result<bool, queue::Error> {
                    let cache = factory
                        .cache
                        .as_deref()
                        .ok_or(queue::Error::Invalid("reranker health cache"))?;
                    let record = health::Record::new(
                        &factory.settings.config,
                        cache,
                        State::Disabled,
                        format!("reload smoke failed; disabled model {file}"),
                    )?;
                    health::write(
                        &factory.store,
                        &factory.settings.config_root,
                        &factory.settings.config,
                        &record,
                    )
                })();
                if !matches!(saved, Ok(true)) {
                    tracing::warn!("local reranker reload failure status could not be saved");
                }
            });
        }
        if let Some(observation) = observation {
            observation.finish(result.is_ok());
        }
        Ok(Response {
            output: result?,
            tokens: None,
        })
    }
}

#[cfg(test)]
mod tests;

/// Reranking outcome accompanies even empty final results, preserving degradation.
#[derive(Debug, Default)]
pub(crate) enum State {
    #[default]
    NotRequested,
    Complete,
    Privacy,
    Timeout,
    Preparing,
    PreparationFailed(Arc<Error>),
    Failed(Box<Error>),
}
impl State {
    /// Returns a stable transport reason only when requested reranking was lost.
    pub(crate) fn fallback(&self) -> Option<&'static str> {
        match self {
            Self::NotRequested | Self::Complete => None,
            Self::Privacy => Some("rerank_privacy"),
            Self::Timeout | Self::Preparing => Some("rerank_timeout"),
            Self::Failed(_) | Self::PreparationFailed(_) => Some("rerank_error"),
        }
    }
    /// Classifies lost reranking without discarding the typed diagnostic cause.
    pub(super) fn failure(error: Error) -> Self {
        match error {
            Error::Privacy => Self::Privacy,
            Error::Deadline => Self::Timeout,
            other => Self::Failed(Box::new(other)),
        }
    }
}

/// Rejects invalid ranking controls before any potentially billable scorer call.
pub(super) fn validate_ranking(
    rerank: &config::Rerank,
    search: &config::Search,
) -> Result<(), Error> {
    if !rerank.min_score.is_finite()
        || !(0.0..=1.0).contains(&rerank.min_score)
        || !search.diversity_margin.is_finite()
        || !(0.0..=1.0).contains(&search.diversity_margin)
    {
        return Err(Error::Input);
    }
    Ok(())
}

/// Applies complete finite probabilities atomically to the original RRF window.
/// Equal scores retain RRF order. Diversification greedily promotes the strongest
/// unseen file within the configured margin of the current leading candidate;
/// fileless units do not invent a shared file identity. Collapse and k follow later.
pub(super) fn apply_scores(
    hits: &mut Vec<super::RankedCandidate>,
    scores: &[f32],
    rerank: &config::Rerank,
    search: &config::Search,
) -> Result<(), Error> {
    validate_ranking(rerank, search)?;
    if scores.len() != hits.len()
        || scores
            .iter()
            .any(|score| !score.is_finite() || !(0.0..=1.0).contains(score))
    {
        return Err(Error::Input);
    }
    let mut ranked = std::mem::take(hits)
        .into_iter()
        .zip(scores)
        .filter_map(|(mut hit, &score)| {
            if f64::from(score) < rerank.min_score {
                return None;
            }
            hit.score = f64::from(score);
            hit.rerank_score = Some(score);
            for evidence in &mut hit.matches {
                evidence.rerank_score = Some(score);
                evidence.why.push("reranked");
            }
            Some(hit)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| {
        if a.score == b.score {
            std::cmp::Ordering::Equal
        } else {
            b.score.total_cmp(&a.score)
        }
    });
    let mut represented = BTreeSet::new();
    while !ranked.is_empty() {
        let best = &ranked[0];
        let promote = if best
            .unit
            .path
            .as_ref()
            .is_some_and(|path| represented.contains(path))
        {
            ranked
                .iter()
                .position(|hit| {
                    best.score - hit.score <= search.diversity_margin
                        && hit
                            .unit
                            .path
                            .as_ref()
                            .is_some_and(|path| !represented.contains(path))
                })
                .unwrap_or(0)
        } else {
            0
        };
        let hit = ranked.remove(promote);
        if let Some(path) = &hit.unit.path {
            represented.insert(path.clone());
        }
        hits.push(hit);
    }
    Ok(())
}

/// Adaptive candidate selection and matching persisted status; v1 §18.
pub(crate) mod auto;

/// Retained first-use model preparation; v1 §7.4.
pub(crate) mod lazy;
