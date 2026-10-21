//! Prepared local and HTTP query embeddings with shared admission; v1 §§7.2, 8.2, 9.
#![expect(
    dead_code,
    reason = "dense orchestration connects this adapter in T1.7"
)]

use super::{admission, retry};
use crate::{
    config::{self, Config},
    models::{
        attempt::{AttemptError, PreparedQuery},
        llama::LocalProvider,
        ollama::{OllamaConfig, OllamaEmbedder},
        openai_compat::{OpenAiConfig, OpenAiEmbedder},
    },
    queue::{
        self, destination, embedding::Destination, remote_dimensions, usage, worker::Response,
    },
    store::{Store, UnitKind},
};
use async_trait::async_trait;
use rusqlite::{Transaction, params};
use std::{path::PathBuf, sync::Arc, time::Duration};
use tokio::time::Instant;

/// Selected tier and authoritative configuration; daily budgets remain explicit
/// until the public queue configuration contract is settled.
pub(crate) struct Settings {
    pub(crate) config_root: PathBuf,
    pub(crate) config: Config,
    pub(crate) kind: UnitKind,
    pub(crate) budget: usage::Budget,
}

mod previous;

enum Provider {
    Remote(OpenAiEmbedder),
    Ollama(OllamaEmbedder),
    Local(LocalProvider),
}

/// Reusable provider for one configured query tier; no source text or secrets
/// are exposed through Debug. Model downloads belong to separate preparation.
pub(crate) struct Factory {
    store: Arc<Store>,
    settings: Settings,
    identity: Destination,
    provider: Provider,
    dimensions_capability: Option<String>,
    table: String,
    previous: Option<previous::Previous>,
}
impl Factory {
    /// Constructs only the selected HTTP tier, resolving remote credentials through
    /// the existing private credential loader. Run construction off the runtime.
    pub(crate) fn new(store: Arc<Store>, settings: Settings) -> Result<Arc<Self>, queue::Error> {
        Self::build(store, settings, OpenAiEmbedder::from_config)
    }

    /// Constructs a mock HTTP query provider without reading host credentials.
    #[cfg(test)]
    pub(crate) fn new_mock_remote(
        store: Arc<Store>,
        settings: Settings,
    ) -> Result<Arc<Self>, queue::Error> {
        Self::build(store, settings, |config| {
            OpenAiEmbedder::with_api_key(config, String::new())
        })
    }

    fn build(
        store: Arc<Store>,
        settings: Settings,
        remote: impl FnOnce(OpenAiConfig) -> crate::models::Result<OpenAiEmbedder>,
    ) -> Result<Arc<Self>, queue::Error> {
        let resolved = Self::validate(&store, &settings)?;
        let table = store.embedding_destination(&resolved.binding(&settings.config))?;
        let provider = Self::http_provider(&settings.config, &resolved, remote)?;
        let dimensions_capability = if let Provider::Remote(remote) = &provider {
            Some(remote_dimensions::restore(
                &store,
                &resolved.identity,
                remote,
            )?)
        } else {
            None
        };
        Ok(Arc::new(Self {
            store,
            settings,
            identity: resolved.identity,
            provider,
            dimensions_capability,
            table,
            previous: None,
        }))
    }

    fn http_provider(
        config: &Config,
        resolved: &destination::Resolved,
        remote: impl FnOnce(OpenAiConfig) -> crate::models::Result<OpenAiEmbedder>,
    ) -> Result<Provider, queue::Error> {
        let limits = &config.limits;
        let timeout = Duration::from_millis(limits.query_timeout_ms);
        let embedder = &config.embedder;
        let provider = if resolved.identity.provider.starts_with("embed:remote:") {
            let cfg = OpenAiConfig::from_remote(&embedder.remote, embedder.dims, timeout)
                .map_err(|_| queue::Error::Invalid("remote query settings"))?;
            Provider::Remote(
                remote(cfg)
                    .map_err(|_| queue::Error::Invalid("remote query client or credentials"))?,
            )
        } else if resolved.identity.provider.starts_with("embed:ollama:") {
            let cfg = OllamaConfig::from_profile(&embedder.ollama, embedder.dims, timeout)
                .map_err(|_| queue::Error::Invalid("Ollama query settings"))?;
            Provider::Ollama(
                OllamaEmbedder::from_config(cfg)
                    .map_err(|_| queue::Error::Invalid("Ollama query client"))?,
            )
        } else {
            return Err(queue::Error::Invalid(
                "HTTP queries require a remote or Ollama tier",
            ));
        };
        Ok(provider)
    }

    fn validate(store: &Store, settings: &Settings) -> Result<destination::Resolved, queue::Error> {
        let limits = &settings.config.limits;
        if limits.query_timeout_ms == 0 {
            return Err(queue::Error::Invalid("zero query timeout"));
        }
        if limits
            .db_max_bytes()
            .map_err(|_| queue::Error::Invalid("query database cap"))?
            .0
            != store.db_max_bytes
            || limits
                .wal_max_bytes()
                .map_err(|_| queue::Error::Invalid("query WAL cap"))?
                .0
                != store.wal_max_bytes
        {
            return Err(queue::Error::Invalid(
                "query storage caps differ from open store; reopen with current limits",
            ));
        }
        let resolved = destination::resolve(&settings.config, settings.kind)?;
        resolved.limits(settings.budget)?;
        Ok(resolved)
    }

    /// Loads only verified registry weights from the supplied cache. Construction
    /// is separate from query deadlines; native loading runs off the runtime.
    /// Each factory retains one shared cancellation/capacity gate for its queries.
    pub(crate) async fn new_local(
        store: Arc<Store>,
        settings: Settings,
        cache: PathBuf,
    ) -> Result<Arc<Self>, queue::Error> {
        let resolved = Self::validate(&store, &settings)?;
        if resolved.identity.provider != "embed:local" {
            return Err(queue::Error::Invalid(
                "local query factory requires a local tier",
            ));
        }
        let binding = resolved.binding(&settings.config);
        let selected_store = Arc::clone(&store);
        let table =
            tokio::task::spawn_blocking(move || selected_store.embedding_destination(&binding))
                .await
                .map_err(|_| queue::Error::WorkerTask)??;
        let idle = settings
            .config
            .models
            .unload_after_idle()
            .map_err(|_| queue::Error::Invalid("local query idle timeout"))?;
        let provider = LocalProvider::load_registered(
            cache,
            settings.config.embedder.local.clone(),
            settings.config.embedder.dims,
            idle,
        )
        .await
        .map_err(|_| queue::Error::Invalid("local query artifact or model initialization"))?;
        Ok(Arc::new(Self {
            store,
            settings,
            identity: resolved.identity,
            provider: Provider::Local(provider),
            dimensions_capability: None,
            table,
            previous: None,
        }))
    }

    /// Checks that a retrieval context uses the same effective configuration as
    /// this prepared provider, without exposing credentials or configuration values.
    pub(crate) fn matches_configuration(&self, config: &Config) -> bool {
        self.settings.config == *config
    }

    /// Returns the store owned by this provider so retrieval cannot scan another
    /// repository while charging this factory's admission ledger.
    pub(crate) fn store(&self) -> &Arc<Store> {
        &self.store
    }

    fn profile(&self) -> &'static str {
        match self.provider {
            Provider::Remote(_) => "remote",
            Provider::Ollama(_) => "ollama",
            Provider::Local(_) => "local",
        }
    }

    /// Returns the destination selected during provider preparation. Uses the same
    /// proven-space routing as the host; this accessor performs no database work.
    pub(crate) fn table_slug(&self) -> String {
        self.table.clone()
    }

    /// Checks active table identity in the caller's snapshot, without reading an
    /// unrelated vector table or assuming equal dimensions imply equal models.
    pub(crate) fn table_matches(
        &self,
        tx: &Transaction<'_>,
        slug: &str,
    ) -> Result<bool, crate::store::Error> {
        let profile = self.profile();
        if let Some(previous) = &self.previous {
            if slug != self.table || !previous.allows(tx, slug, self.settings.kind)? {
                return Ok(false);
            }
        } else if slug == self.table
            && let Some(route) = crate::store::transitions::read(tx)?
            && route.desired.get(self.settings.kind.as_str()) != Some(&self.table)
        {
            return Ok(false);
        }
        // Configured routing always requires its proven binding, including after
        // a table is removed/recreated. Explicit legacy/component tables remain
        // unproven and cannot serve transition fallback. Equal model/dimensions must
        // not authorize a query from another endpoint or local artifact.
        let binding = crate::store::embedding_bindings::read(tx, slug)?;
        if binding.is_none() && slug == self.table {
            return Ok(false);
        }
        if let Some(binding) = binding
            && (binding.profile != profile
                || binding.model != self.identity.model
                || binding.dims != self.identity.dims
                || binding.endpoint != self.identity.endpoint
                || binding.artifact_hash != self.identity.artifact_hash)
        {
            return Ok(false);
        }
        let dims = i64::try_from(self.identity.dims).map_err(|_| {
            crate::store::Error::InvalidVectorDimensions {
                slug: slug.to_owned(),
            }
        })?;
        Ok(tx.query_row("SELECT EXISTS(SELECT 1 FROM embedding_tables WHERE slug=?1 AND model=?2 AND profile=?3 AND dims=?4 AND active=1)",
            params![slug,self.identity.model,profile,dims], |r|r.get(0))?)
    }

    /// Rechecks the effective configuration before using an asynchronous result.
    pub(crate) fn current(&self) -> Result<bool, queue::Error> {
        Ok(config::load(&self.settings.config_root, None)
            .map_err(|_| queue::Error::Invalid("query configuration reload"))?
            .config
            == self.settings.config)
    }

    /// Caps an existing deadline at the configured total query timeout. Retrieval
    /// calls this before preflight, then passes the same deadline through embedding.
    pub(crate) fn deadline(&self, caller: Instant) -> Result<Instant, queue::Error> {
        let configured = Instant::now()
            .checked_add(Duration::from_millis(
                self.settings.config.limits.query_timeout_ms,
            ))
            .ok_or(queue::Error::Invalid("query deadline overflow"))?;
        Ok(caller.min(configured))
    }

    /// Embeds private query text once, including preparation and accounting in the
    /// caller's absolute deadline. The configured query cap can only shorten it.
    /// Retries must call this again, acquiring a new reservation each time.
    pub(crate) async fn embed_once<C>(
        self: &Arc<Self>,
        text: String,
        deadline: Instant,
        clock: C,
    ) -> Result<Vec<f32>, admission::Error>
    where
        C: Fn() -> i64 + Send + Sync + 'static,
    {
        self.execute(text, deadline, clock, false).await
    }

    /// Prepares once, then performs only separately admitted retries which fit the
    /// original deadline. No source text or uncertain charges are discarded by retry.
    pub(crate) async fn embed<C>(
        self: &Arc<Self>,
        text: String,
        deadline: Instant,
        clock: C,
    ) -> Result<Vec<f32>, admission::Error>
    where
        C: Fn() -> i64 + Send + Sync + 'static,
    {
        self.execute(text, deadline, clock, true).await
    }

    async fn execute<C>(
        self: &Arc<Self>,
        text: String,
        deadline: Instant,
        clock: C,
        retries: bool,
    ) -> Result<Vec<f32>, admission::Error>
    where
        C: Fn() -> i64 + Send + Sync + 'static,
    {
        let deadline = self.deadline(deadline)?;
        let factory = Arc::clone(self);
        let attempt = async {
            if Instant::now() >= deadline {
                return Err(admission::Error::Deadline);
            }
            let work = tokio::task::spawn_blocking(move || {
                if Instant::now() >= deadline {
                    return Err(admission::Error::Deadline);
                }
                let input = match &factory.provider {
                    Provider::Remote(provider) => provider.prepare_query(&text)?,
                    Provider::Ollama(provider) => provider.prepare_query(&text)?,
                    Provider::Local(provider) => provider.prepare_query(&text)?,
                };
                Ok(Arc::new(Query { factory, input }))
            })
            .await
            .map_err(|_| admission::Error::Task)??;
            let clock = Arc::new(clock);
            let attempt = || {
                let clock = Arc::clone(&clock);
                let factory = Arc::clone(self);
                let work = Arc::clone(&work);
                async move {
                    let result = admission::execute(
                        Arc::clone(&factory.store),
                        factory.identity.provider.clone(),
                        factory.settings.budget,
                        deadline,
                        work,
                        move || clock(),
                    )
                    .await;
                    if matches!(
                        result,
                        Err(admission::Error::Provider(AttemptError::DimensionsRejected))
                    ) {
                        let key = factory.dimensions_capability.clone().ok_or_else(|| {
                            admission::Error::Accounting(queue::Error::Invalid(
                                "dimensions rejection from non-remote query provider",
                            ))
                        })?;
                        let store = Arc::clone(&factory.store);
                        tokio::task::spawn_blocking(move || {
                            remote_dimensions::remember(&store, &key)
                        })
                        .await
                        .map_err(|_| admission::Error::Task)??;
                    }
                    result
                }
            };
            if retries {
                retry::execute(deadline, attempt).await
            } else {
                attempt().await
            }
        };
        tokio::time::timeout_at(deadline, attempt)
            .await
            .map_err(|_| admission::Error::Deadline)?
    }
}

struct Query {
    factory: Arc<Factory>,
    input: PreparedQuery,
}
#[async_trait]
impl admission::Work for Query {
    type Output = Vec<f32>;
    fn tokens(&self) -> i64 {
        self.input.tokens()
    }
    fn endpoint(&self) -> Option<usage::Endpoint> {
        if matches!(self.factory.provider, Provider::Local(_)) {
            return None;
        }
        Some(usage::Endpoint {
            kind: "embed",
            url: self.factory.identity.endpoint.clone(),
            units: 1,
        })
    }
    fn limits(
        &self,
        tx: &Transaction<'_>,
        provider: &str,
    ) -> Result<Option<usage::Limits>, queue::Error> {
        let settings = &self.factory.settings;
        let current = config::load(&settings.config_root, None)
            .map_err(|_| queue::Error::Invalid("query configuration reload"))?
            .config;
        if current != settings.config || provider != self.factory.identity.provider {
            return Ok(None);
        }
        if (self.factory.previous.is_some() || crate::store::transitions::read(tx)?.is_some())
            && !self.factory.table_matches(tx, &self.factory.table)?
        {
            return Ok(None);
        }
        let effective = if let Some(previous) = &self.factory.previous {
            previous.configuration(&current)?
        } else {
            current
        };
        destination::resolve(&effective, settings.kind)?
            .limits(settings.budget)
            .map(Some)
    }
    async fn call(&self) -> Result<Response<Vec<f32>>, AttemptError> {
        let mut batch = match &self.factory.provider {
            Provider::Remote(provider) => provider.embed_query_with_usage_once(&self.input).await?,
            Provider::Ollama(provider) => provider.embed_query_with_usage_once(&self.input).await?,
            Provider::Local(provider) => provider.embed_query_with_usage_once(&self.input).await?,
        };
        if batch.embeddings.len() != 1 {
            return Err(AttemptError::InvalidResponse);
        }
        let output = batch
            .embeddings
            .pop()
            .ok_or(AttemptError::InvalidResponse)?;
        Ok(Response {
            output,
            tokens: batch.tokens,
        })
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod local_tests;
