//! Fixed-input diagnostic embeddings through shared provider admission; v1 §§7.2, 8.2, 13.
use super::network::LedgerIdentity;
use crate::{
    config::{self, Config},
    models::{
        attempt::{AttemptError, PreparedQuery},
        llama::LocalProvider,
        ollama::{OllamaConfig, OllamaEmbedder},
        openai_compat::{OpenAiConfig, OpenAiEmbedder},
        registry,
    },
    queue::{self, destination, host::Bounds, usage, worker::Response},
    search::admission,
    store::{Store, UnitKind},
};
use async_trait::async_trait;
use rusqlite::Transaction;
use std::{path::PathBuf, sync::Arc, time::Duration};
use tokio::time::Instant;

const QUERY: &str = "Find the function that reads a configuration file.";

enum Provider {
    Remote(OpenAiEmbedder),
    Ollama(OllamaEmbedder),
    Local(LocalProvider),
}

#[cfg(test)]
type LocalExecution =
    Arc<dyn Fn() -> Result<crate::models::attempt::EmbeddingBatch, AttemptError> + Send + Sync>;

/// Private prepared public input: never binds vectors or accepts repository text.
pub(super) struct Embedding {
    ledger: LedgerIdentity,
    root: PathBuf,
    config: Config,
    kind: UnitKind,
    provider_id: String,
    endpoint: String,
    provider: Provider,
    input: PreparedQuery,
    budget: usage::Budget,
    #[cfg(test)]
    local_execution: Option<LocalExecution>,
}
impl Embedding {
    /// Resolves the selected tier and credentials without creating a vector binding.
    pub(super) fn new(
        root: PathBuf,
        config: Config,
        kind: UnitKind,
        ledger: LedgerIdentity,
    ) -> Result<Self, queue::Error> {
        Self::build(root, config, kind, ledger, OpenAiEmbedder::from_config)
    }
    fn build(
        root: PathBuf,
        config: Config,
        kind: UnitKind,
        ledger: LedgerIdentity,
        remote: impl FnOnce(OpenAiConfig) -> crate::models::Result<OpenAiEmbedder>,
    ) -> Result<Self, queue::Error> {
        let resolved = destination::resolve(&config, kind)?;
        let budget = Bounds::for_provider(&config.queue, &resolved.identity.provider)?.budget;
        resolved.limits(budget)?;
        let timeout = Duration::from_millis(config.limits.query_timeout_ms.min(15_000));
        if timeout.is_zero() {
            return Err(queue::Error::Invalid("zero diagnostic deadline"));
        }
        let provider = if resolved.identity.provider.starts_with("embed:remote:") {
            let settings =
                OpenAiConfig::from_remote(&config.embedder.remote, resolved.identity.dims, timeout)
                    .map_err(|_| queue::Error::Invalid("diagnostic embedding settings"))?;
            Provider::Remote(
                remote(settings)
                    .map_err(|_| queue::Error::Invalid("diagnostic credentials or client"))?,
            )
        } else if resolved.identity.provider.starts_with("embed:ollama:") {
            let settings = OllamaConfig::from_profile(
                &config.embedder.ollama,
                resolved.identity.dims,
                timeout,
            )
            .map_err(|_| queue::Error::Invalid("diagnostic Ollama settings"))?;
            Provider::Ollama(
                OllamaEmbedder::from_config(settings)
                    .map_err(|_| queue::Error::Invalid("diagnostic Ollama client"))?,
            )
        } else if resolved.identity.provider == "embed:local" {
            let cache = registry::cache_root()
                .map_err(|_| queue::Error::Invalid("diagnostic model cache unavailable"))?;
            let idle = config
                .models
                .unload_after_idle()
                .map_err(|_| queue::Error::Invalid("diagnostic local idle duration"))?;
            Provider::Local(
                LocalProvider::prepare_registered(
                    cache,
                    config.embedder.local.clone(),
                    resolved.identity.dims,
                    idle,
                )
                .map_err(|_| queue::Error::Invalid("diagnostic local embedding settings"))?,
            )
        } else {
            return Err(queue::Error::Invalid(
                "diagnostic embedding tier is not selected",
            ));
        };
        let input = match &provider {
            Provider::Remote(provider) => provider.prepare_query(QUERY),
            Provider::Ollama(provider) => provider.prepare_query(QUERY),
            Provider::Local(provider) => provider.prepare_query(QUERY),
        }
        .map_err(|_| queue::Error::Invalid("diagnostic input preparation"))?;
        Ok(Self {
            ledger,
            root,
            config,
            kind,
            provider_id: resolved.identity.provider,
            endpoint: resolved.identity.endpoint,
            provider,
            input,
            budget,
            #[cfg(test)]
            local_execution: None,
        })
    }

    /// Checks Ollama installation without pulling a model or invoking inference.
    /// Local returns true to skip listing; artifact verification remains inside
    /// the admitted call and is not attested to by this result.
    pub(super) async fn installed(&self) -> Result<bool, queue::Error> {
        match &self.provider {
            Provider::Remote(_) | Provider::Local(_) => Ok(true),
            Provider::Ollama(provider) => provider
                .has_model()
                .await
                .map_err(|_| queue::Error::Invalid("Ollama model listing unavailable")),
        }
    }

    /// Every attempt enters the ordinary ledger. Only dimensions rejection permits
    /// one separately admitted fallback within the original absolute deadline.
    /// Success reports dimensions and whether this run used that fallback.
    pub(super) async fn run<C>(
        self: Arc<Self>,
        store: Arc<Store>,
        deadline: Instant,
        clock: C,
    ) -> Result<(usize, bool), admission::Error>
    where
        C: Fn() -> i64 + Send + Sync + Clone + 'static,
    {
        for attempt in 0..2 {
            let result = admission::execute(
                Arc::clone(&store),
                self.provider_id.clone(),
                self.budget,
                deadline,
                Arc::clone(&self),
                clock.clone(),
            )
            .await;
            if attempt == 0
                && matches!(&self.provider, Provider::Remote(_))
                && matches!(
                    result,
                    Err(admission::Error::Provider(AttemptError::DimensionsRejected))
                )
            {
                continue;
            }
            return result.map(|dims| (dims, attempt == 1));
        }
        Err(admission::Error::Provider(AttemptError::InvalidResponse))
    }
}

#[async_trait]
impl admission::Work for Embedding {
    type Output = usize;
    fn tokens(&self) -> i64 {
        self.input.tokens()
    }
    fn endpoint(&self) -> Option<usage::Endpoint> {
        if matches!(&self.provider, Provider::Local(_)) {
            return None;
        }
        Some(usage::Endpoint {
            kind: "embed",
            url: self.endpoint.clone(),
            units: 1,
        })
    }
    fn limits(
        &self,
        _tx: &Transaction<'_>,
        provider: &str,
    ) -> Result<Option<usage::Limits>, queue::Error> {
        if !self.ledger.valid() {
            return Ok(None);
        }
        let current = config::load(&self.root, None)
            .map_err(|_| queue::Error::Invalid("diagnostic configuration reload"))?
            .config;
        if current != self.config || provider != self.provider_id {
            return Ok(None);
        }
        let resolved = destination::resolve(&current, self.kind)?;
        if resolved.identity.provider != self.provider_id {
            return Ok(None);
        }
        let budget = Bounds::for_provider(&current.queue, provider)?.budget;
        resolved.limits(budget).map(Some)
    }
    async fn call(&self) -> Result<Response<usize>, AttemptError> {
        let batch = match &self.provider {
            Provider::Remote(provider) => provider.embed_query_with_usage_once(&self.input).await?,
            Provider::Ollama(provider) => provider.embed_query_with_usage_once(&self.input).await?,
            Provider::Local(provider) => {
                #[cfg(test)]
                if let Some(execute) = &self.local_execution {
                    execute()?
                } else {
                    provider.embed_query_with_usage_once(&self.input).await?
                }
                #[cfg(not(test))]
                provider.embed_query_with_usage_once(&self.input).await?
            }
        };
        if batch.embeddings.len() != 1 {
            return Err(AttemptError::InvalidResponse);
        }
        let vector = &batch.embeddings[0];
        if vector.is_empty() || vector.iter().any(|value| !value.is_finite()) {
            return Err(AttemptError::InvalidResponse);
        }
        Ok(Response {
            output: vector.len(),
            tokens: batch.tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::params;
    use serde_json::{Value, json};
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    static MODEL_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    const NOW: i64 = 100;
    const PREFIX: &str = "Diagnostic docs prefix: ";

    fn fixture(
        server: &MockServer,
        calls: i64,
        docs: bool,
    ) -> (tempfile::TempDir, Arc<Store>, Arc<Embedding>) {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join(".deepkit")).unwrap();
        let tier = if docs { "local" } else { "remote" };
        let docs_config = if docs {
            "[embedder.docs]\nprofile='remote'\nkinds=['doc_section']\n"
        } else {
            ""
        };
        std::fs::write(root.path().join(".deepkit/config.toml"), format!(
            "[embedder]\nprofile='{tier}'\ndims=4\n[embedder.remote]\nurl='{}/v1'\nmodel='diagnostic-fixture-{}'\nquery_prefix='{PREFIX}'\n{docs_config}[[queue.budgets]]\nprofile='remote'\nurl='{}/v1'\ndaily_calls={calls}\n",
            server.uri(), MODEL_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed), server.uri(),
        )).unwrap();
        let config = config::load(root.path(), None).unwrap().config;
        let kind = if docs {
            UnitKind::DocSection
        } else {
            UnitKind::Chunk
        };
        let embedding = Embedding::build(
            root.path().to_owned(),
            config,
            kind,
            LedgerIdentity::UncheckedFixture,
            |settings| OpenAiEmbedder::with_api_key(settings, "fixture-placeholder".into()),
        )
        .unwrap();
        let store =
            crate::store::open(root.path().join(".deepkit/code.db"), Default::default()).unwrap();
        (root, Arc::new(store), Arc::new(embedding))
    }

    // Capture schema and all existing embedding metadata; ordinary usage writes
    // may change ledger rows, but diagnostics must not bind or publish a space.
    fn embedding_metadata(store: &Store) -> Vec<Vec<Vec<Value>>> {
        let connection = store.writer().unwrap();
        [
            "sqlite_schema",
            "meta",
            "embedding_tables",
            "embedding_bindings",
            "embedding_transition",
            "embedding_publications",
        ]
        .into_iter()
        .map(|table| {
            let mut query = connection
                .prepare(&format!("SELECT * FROM {table} ORDER BY rowid"))
                .unwrap();
            let columns = query.column_count();
            query
                .query_map([], |row| {
                    (0..columns)
                        .map(|column| {
                            use rusqlite::types::ValueRef;
                            Ok(match row.get_ref(column)? {
                                ValueRef::Null => Value::Null,
                                ValueRef::Integer(value) => json!(value),
                                ValueRef::Real(value) => json!(value),
                                ValueRef::Text(value) | ValueRef::Blob(value) => json!(value),
                            })
                        })
                        .collect::<rusqlite::Result<Vec<_>>>()
                })
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap()
        })
        .collect()
    }

    fn reservation_counts(store: &Store) -> (i64, i64, i64) {
        store.writer().unwrap().query_row(
            "SELECT COALESCE(SUM(state='reserved'),0),COALESCE(SUM(state='dispatched'),0),COALESCE(SUM(state='reconciled'),0) FROM usage_reservations", [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        ).unwrap()
    }

    fn success() -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"index": 0, "embedding": "AABAQAAAgEAAAAAAAAAAAA=="}],
            "usage": {"total_tokens": 7}
        }))
    }

    #[tokio::test]
    async fn dimensions_fallback_requires_each_daily_admission_and_never_binds_vectors() {
        for calls in [0, 1, 2] {
            let server = MockServer::start().await;
            Mock::given(method("POST")).and(path("/v1/embeddings"))
                .respond_with(|request: &wiremock::Request| {
                    let body: Value = serde_json::from_slice(&request.body).unwrap();
                    if body.get("dimensions").is_some() {
                        ResponseTemplate::new(400).set_body_json(json!({"error": {"message": "model does not support matryoshka representation"}}))
                    } else { success() }
                }).expect(calls as u64).mount(&server).await;
            let (_root, store, embedding) = fixture(&server, calls, false);
            let before = embedding_metadata(&store);
            let result = embedding
                .run(
                    Arc::clone(&store),
                    Instant::now() + Duration::from_secs(5),
                    || NOW,
                )
                .await;
            if calls < 2 {
                assert!(matches!(result, Err(admission::Error::Budget { .. })));
            } else {
                assert_eq!(result.unwrap(), (4, true));
            }
            assert_eq!(embedding_metadata(&store), before);
            assert_eq!(
                reservation_counts(&store),
                (0, i64::from(calls > 0), i64::from(calls == 2))
            );
            let requests = server.received_requests().await.unwrap();
            assert_eq!(requests.len(), calls as usize);
            if let Some(first) = requests.first() {
                let body: Value = serde_json::from_slice(&first.body).unwrap();
                assert_eq!(body["dimensions"], 4);
            }
            if calls == 2 {
                let second: Value = serde_json::from_slice(&requests[1].body).unwrap();
                assert!(second.get("dimensions").is_none());
            }
            server.verify().await;
        }
    }

    #[tokio::test]
    async fn docs_tier_uses_selected_prefix_dimensions_and_settles_actual_usage() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(success())
            .expect(1)
            .mount(&server)
            .await;
        let (_root, store, embedding) = fixture(&server, 1, true);
        let before = embedding_metadata(&store);
        let provider = embedding.provider_id.clone();
        assert_eq!(
            embedding
                .run(
                    Arc::clone(&store),
                    Instant::now() + Duration::from_secs(5),
                    || NOW
                )
                .await
                .unwrap(),
            (4, false)
        );
        assert_eq!(reservation_counts(&store), (0, 0, 1));
        let usage: (i64, i64) = store
            .writer()
            .unwrap()
            .query_row(
                "SELECT calls,tokens FROM usage WHERE provider=?1 AND day='1970-01-01'",
                params![provider],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(usage, (1, 7));
        let requests = server.received_requests().await.unwrap();
        let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["dimensions"], 4);
        assert_eq!(body["input"], json!([format!("{PREFIX}{QUERY}")]));
        assert_eq!(embedding_metadata(&store), before);
        server.verify().await;
    }

    #[tokio::test]
    async fn changed_configuration_refuses_dispatch_and_releases_reservation() {
        let server = MockServer::start().await;
        let (root, store, embedding) = fixture(&server, 2, false);
        let before = embedding_metadata(&store);
        let config_path = root.path().join(".deepkit/config.toml");
        let mut config = std::fs::read_to_string(&config_path).unwrap();
        config.push_str("\n[index]\nrepo_name='changed'\n");
        std::fs::write(config_path, config).unwrap();
        let result = embedding
            .run(
                Arc::clone(&store),
                Instant::now() + Duration::from_secs(5),
                || NOW,
            )
            .await;
        assert!(matches!(result, Err(admission::Error::Refused(_))));
        assert!(server.received_requests().await.unwrap().is_empty());
        assert_eq!(reservation_counts(&store), (0, 0, 0));
        let released: i64 = store
            .writer()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM usage_reservations WHERE state='released'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(released, 1);
        assert_eq!(embedding_metadata(&store), before);
    }

    #[tokio::test]
    async fn rate_limit_sends_once_and_retains_uncertain_charge() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("Retry-After", "1")
                    .set_body_string("private-error-fixture"),
            )
            .expect(1)
            .mount(&server)
            .await;
        let (_root, store, embedding) = fixture(&server, 2, false);
        let before = embedding_metadata(&store);
        let result = embedding
            .run(
                Arc::clone(&store),
                Instant::now() + Duration::from_secs(5),
                || NOW,
            )
            .await;
        assert!(matches!(
            result,
            Err(admission::Error::Provider(AttemptError::Retryable {
                status: Some(429),
                ..
            }))
        ));
        assert!(!format!("{result:?}").contains("private-error-fixture"));
        assert_eq!(reservation_counts(&store), (0, 1, 0));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
        assert_eq!(embedding_metadata(&store), before);
        server.verify().await;
    }
    #[tokio::test]
    async fn replaced_ledger_identity_refuses_dispatch_with_unchanged_config() {
        use crate::search::admission::Work;
        let server = MockServer::start().await;
        let (root, store, mut embedding) = fixture(&server, 2, false);
        let path = root.path().join(".deepkit/code.db");
        let guard = Arc::new(crate::store::path::prepare(root.path(), &path).unwrap());
        Arc::get_mut(&mut embedding).unwrap().ledger = LedgerIdentity::Checked(guard);
        let before = embedding_metadata(&store);
        {
            let mut writer = store.writer().unwrap();
            let transaction = writer.transaction().unwrap();
            assert!(
                embedding
                    .limits(&transaction, &embedding.provider_id)
                    .unwrap()
                    .is_some()
            );
            // Keep the old handle and identical effective configuration, just as
            // when a model listing yields while another process replaces the path.
            std::fs::rename(&path, path.with_extension("saved")).unwrap();
            std::fs::write(&path, b"replacement-marker").unwrap();
            assert!(
                embedding
                    .limits(&transaction, &embedding.provider_id)
                    .unwrap()
                    .is_none()
            );
            assert_eq!(
                config::load(root.path(), None).unwrap().config,
                embedding.config
            );
            assert_eq!(std::fs::read(&path).unwrap(), b"replacement-marker");
        }

        assert!(server.received_requests().await.unwrap().is_empty());
        // Restore before further SQLite access; the checked refusal itself did
        // not modify either database or dispatch an HTTP request.
        std::fs::remove_file(&path).unwrap();
        std::fs::rename(path.with_extension("saved"), &path).unwrap();
        assert_eq!(embedding_metadata(&store), before);
        assert_eq!(reservation_counts(&store), (0, 0, 0));
    }

    #[tokio::test]
    async fn absent_ledger_identity_cannot_dispatch_inference() {
        let server = MockServer::start().await;
        let (_root, store, mut embedding) = fixture(&server, 2, false);
        Arc::get_mut(&mut embedding).unwrap().ledger = LedgerIdentity::Missing;
        let result = embedding
            .run(
                Arc::clone(&store),
                Instant::now() + Duration::from_secs(5),
                || NOW,
            )
            .await;
        assert!(matches!(result, Err(admission::Error::Refused(_))));
        assert!(server.received_requests().await.unwrap().is_empty());
        assert_eq!(reservation_counts(&store), (0, 0, 0));
    }
    fn local_fixture(calls: i64) -> (tempfile::TempDir, Arc<Store>, Arc<Embedding>) {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join(".deepkit")).unwrap();
        std::fs::write(root.path().join(".deepkit/config.toml"), format!(
            "[embedder]\nprofile='local'\ndims=256\n[embedder.local]\nquery_prefix='{PREFIX}'\n[[queue.budgets]]\nprofile='local'\ndaily_calls={calls}\n"
        )).unwrap();
        let config = config::load(root.path(), None).unwrap().config;
        let embedding = Embedding::new(
            root.path().into(),
            config,
            UnitKind::Chunk,
            LedgerIdentity::UncheckedFixture,
        )
        .unwrap();
        let store =
            crate::store::open(root.path().join(".deepkit/code.db"), Default::default()).unwrap();
        (root, Arc::new(store), Arc::new(embedding))
    }

    fn assert_local_has_no_weights(embedding: &Embedding) {
        let Provider::Local(provider) = &embedding.provider else {
            panic!("expected local provider")
        };
        assert_eq!(provider.resident_for_test(), Some(false));
    }

    #[tokio::test]
    async fn local_zero_budget_refuses_without_loading_weights_or_binding_vectors() {
        use crate::search::admission::Work;
        let (_root, store, embedding) = local_fixture(0);
        let before = embedding_metadata(&store);
        assert!(embedding.installed().await.unwrap());
        assert_eq!(embedding.provider_id, "embed:local");
        assert!(embedding.endpoint().is_none());
        assert_local_has_no_weights(&embedding);
        let result = Arc::clone(&embedding)
            .run(
                Arc::clone(&store),
                Instant::now() + Duration::from_secs(5),
                || NOW,
            )
            .await;
        assert!(matches!(result, Err(admission::Error::Budget { .. })));
        assert_local_has_no_weights(&embedding);
        assert_eq!(reservation_counts(&store), (0, 0, 0));
        assert_eq!(embedding_metadata(&store), before);
    }

    #[tokio::test]
    async fn local_success_accounts_estimate_without_outbound_metadata_or_vector_binding() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let (_root, store, mut embedding) = local_fixture(1);
        let before = embedding_metadata(&store);
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&calls);
        Arc::get_mut(&mut embedding).unwrap().local_execution = Some(Arc::new(move || {
            observed.fetch_add(1, Ordering::SeqCst);
            Ok(crate::models::attempt::EmbeddingBatch {
                embeddings: vec![vec![1.0; 256]],
                tokens: None,
            })
        }));
        let estimate = embedding.input.tokens();
        assert_eq!(
            estimate as usize,
            tiktoken_rs::cl100k_base_singleton()
                .encode_with_special_tokens(&format!("{PREFIX}{QUERY}"))
                .len()
        );
        assert_eq!(
            Arc::clone(&embedding)
                .run(
                    Arc::clone(&store),
                    Instant::now() + Duration::from_secs(5),
                    || NOW
                )
                .await
                .unwrap(),
            (256, false)
        );
        let usage: (i64, i64, i64) = store.writer().unwrap().query_row(
            "SELECT calls,tokens,estimated_tokens FROM usage WHERE provider='embed:local' AND day='1970-01-01'", [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        ).unwrap();
        assert_eq!(usage, (1, estimate, estimate));
        let outbound: i64 = store.writer().unwrap().query_row(
            "SELECT COUNT(*) FROM usage_reservations WHERE endpoint_kind IS NOT NULL OR endpoint_url IS NOT NULL OR sent_units IS NOT NULL", [], |row| row.get(0),
        ).unwrap();
        assert_eq!(outbound, 0);
        // A second diagnostic shares the consumed local budget; no isolated probe allowance.
        let result = Arc::clone(&embedding)
            .run(
                Arc::clone(&store),
                Instant::now() + Duration::from_secs(5),
                || NOW,
            )
            .await;
        assert!(matches!(result, Err(admission::Error::Budget { .. })));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(reservation_counts(&store), (0, 0, 1));
        assert_local_has_no_weights(&embedding);
        assert_eq!(embedding_metadata(&store), before);
    }

    #[tokio::test]
    async fn local_config_drift_refuses_and_failure_is_static_without_retry() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        for drift in [true, false] {
            let (root, store, mut embedding) = local_fixture(2);
            let before = embedding_metadata(&store);
            let calls = Arc::new(AtomicUsize::new(0));
            let observed = Arc::clone(&calls);
            Arc::get_mut(&mut embedding).unwrap().local_execution = Some(Arc::new(move || {
                observed.fetch_add(1, Ordering::SeqCst);
                Err(AttemptError::InvalidResponse)
            }));
            if drift {
                let path = root.path().join(".deepkit/config.toml");
                let config = std::fs::read_to_string(&path).unwrap();
                std::fs::write(path, format!("{config}\n[index]\nrepo_name='changed'\n")).unwrap();
            }
            let result = Arc::clone(&embedding)
                .run(
                    Arc::clone(&store),
                    Instant::now() + Duration::from_secs(5),
                    || NOW,
                )
                .await;
            if drift {
                assert!(matches!(result, Err(admission::Error::Refused(_))));
            } else {
                assert!(matches!(
                    result,
                    Err(admission::Error::Provider(AttemptError::InvalidResponse))
                ));
                assert_eq!(
                    result.unwrap_err().to_string(),
                    "invalid embedding response; diagnosis required"
                );
            }
            assert_eq!(calls.load(Ordering::SeqCst), usize::from(!drift));
            assert_eq!(reservation_counts(&store), (0, i64::from(!drift), 0));
            assert_local_has_no_weights(&embedding);
            assert_eq!(embedding_metadata(&store), before);
        }
    }
}
