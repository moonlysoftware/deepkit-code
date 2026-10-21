//! Fixed public reranker smoke with shared usage accounting; v1 §§7.5, 8.2, 13.
use super::network::LedgerIdentity;
use crate::{
    config::{self, Config},
    models::{
        attempt::AttemptError,
        rerank::{remote::Remote, smoke},
    },
    queue::{self, usage, worker::Response},
    search::admission,
    store::Store,
};
use async_trait::async_trait;
use rusqlite::Transaction;
use std::{path::PathBuf, sync::Arc, time::Duration};
use tokio::time::Instant;

pub(super) struct Rerank {
    ledger: LedgerIdentity,
    root: PathBuf,
    config: Config,
    provider_id: String,
    endpoint: String,
    provider: Remote,
    documents: Vec<String>,
    tokens: i64,
}
impl Rerank {
    /// Constructs only the selected remote scorer; no source inputs or model downloads.
    pub(super) fn new(
        root: PathBuf,
        config: Config,
        ledger: LedgerIdentity,
    ) -> Result<Self, queue::Error> {
        Self::build(root, config, ledger, Remote::from_config)
    }
    fn build(
        root: PathBuf,
        config: Config,
        ledger: LedgerIdentity,
        construct: impl FnOnce(&config::Rerank, Duration) -> crate::models::Result<Remote>,
    ) -> Result<Self, queue::Error> {
        let endpoint = config::queue::embedding_endpoint(&config.rerank.remote.url, "rerank")
            .map_err(|_| queue::Error::Invalid("diagnostic rerank endpoint"))?;
        let provider_id = format!(
            "rerank:remote:{}",
            blake3::hash(endpoint.as_bytes()).to_hex()
        );
        let timeout = Duration::from_millis(config.limits.rerank_timeout_ms.min(15_000));
        if timeout.is_zero() || config.rerank.candidates < 3 || config.rerank.max_doc_chars == 0 {
            return Err(queue::Error::Invalid(
                "diagnostic rerank limits cannot fit smoke inputs",
            ));
        }
        let chars = usize::try_from(config.rerank.max_doc_chars)
            .map_err(|_| queue::Error::Invalid("diagnostic rerank document cap"))?;
        let documents: Vec<String> = smoke::DOCUMENTS
            .iter()
            .map(|document| document.chars().take(chars).collect())
            .collect();
        let tokenizer = tiktoken_rs::cl100k_base_singleton();
        let query_tokens = tokenizer.encode_with_special_tokens(smoke::QUERY).len();
        let tokens = documents.iter().try_fold(0_i64, |sum, document| {
            let count = tokenizer
                .encode_with_special_tokens(document)
                .len()
                .checked_add(query_tokens)
                .and_then(|count| count.checked_add(1))
                .and_then(|count| i64::try_from(count).ok())
                .ok_or(queue::Error::Invalid("diagnostic rerank token estimate"))?;
            sum.checked_add(count)
                .ok_or(queue::Error::Invalid("diagnostic rerank token estimate"))
        })?;
        let provider = construct(&config.rerank, timeout)
            .map_err(|_| queue::Error::Invalid("diagnostic rerank credentials or client"))?;
        Ok(Self {
            ledger,
            root,
            config,
            provider_id,
            endpoint,
            provider,
            documents,
            tokens,
        })
    }
    /// Settles a valid response before testing relevance. Bad relevance is still
    /// a charged provider call, never an excuse to erase usage or retry.
    pub(super) async fn run<C>(
        self: Arc<Self>,
        store: Arc<Store>,
        deadline: Instant,
        clock: C,
    ) -> Result<bool, admission::Error>
    where
        C: Fn() -> i64 + Send + Sync + 'static,
    {
        let scores = admission::execute(
            store,
            self.provider_id.clone(),
            budget(),
            deadline,
            self,
            clock,
        )
        .await?;
        Ok(smoke::validate(&scores, "remote reranker").is_ok())
    }
}
fn budget() -> usage::Budget {
    // Matches the public reranker Host: configured queue budgets are embedding-only.
    usage::Budget {
        calls: None,
        tokens: None,
    }
}
#[async_trait]
impl admission::Work for Rerank {
    type Output = Vec<f32>;
    fn tokens(&self) -> i64 {
        self.tokens
    }
    fn endpoint(&self) -> Option<usage::Endpoint> {
        Some(usage::Endpoint {
            kind: "rerank",
            url: self.endpoint.clone(),
            units: 3,
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
            .map_err(|_| queue::Error::Invalid("diagnostic rerank configuration reload"))?
            .config;
        if current != self.config || provider != self.provider_id {
            return Ok(None);
        }
        Ok(Some(usage::Limits {
            budget: budget(),
            rpm: None,
            tpm: None,
        }))
    }
    async fn call(&self) -> Result<Response<Vec<f32>>, AttemptError> {
        let documents: Vec<&str> = self.documents.iter().map(String::as_str).collect();
        let output = self.provider.score_once(smoke::QUERY, &documents).await?;
        Ok(Response {
            output,
            tokens: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{self, StoreLimits};
    use serde_json::{Value, json};
    use std::{
        fs,
        sync::atomic::{AtomicBool, Ordering},
    };
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{header, method, path},
    };

    fn fixture(server: &MockServer, characters: u64) -> (tempfile::TempDir, Config, Arc<Store>) {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join(".deepkit")).unwrap();
        fs::write(root.path().join("private.rs"), "repository-private-marker").unwrap();
        let mut config = Config::default();
        config.embedder.profile = "remote".into();
        config.embedder.remote.url = format!("{}/v1", server.uri());
        config.rerank.provider = "remote".into();
        config.rerank.mode = "on".into();
        config.rerank.remote.url = format!("{}/v1", server.uri());
        config.rerank.remote.model = "fixture".into();
        config.rerank.max_doc_chars = characters;
        config.limits.rerank_timeout_ms = 2000;
        // Zero embedding quota at this same gateway must not be cross-mapped
        // onto the distinct public rerank ledger, which has no configured cap.
        config.queue.budgets = vec![config::queue::EndpointBudget {
            profile: "remote".into(),
            url: config.embedder.remote.url.clone(),
            daily_calls: Some(0),
            daily_tokens: Some(0),
        }];
        fs::write(
            root.path().join(".deepkit/config.toml"),
            config.dump().unwrap(),
        )
        .unwrap();
        let config = config::load(root.path(), None).unwrap().config;
        let store = Arc::new(store::open(":memory:", StoreLimits::default()).unwrap());
        (root, config, store)
    }

    fn prepare(root: &std::path::Path, config: Config) -> Arc<Rerank> {
        Arc::new(
            Rerank::build(
                root.into(),
                config,
                LedgerIdentity::UncheckedFixture,
                |settings, timeout| {
                    Remote::with_api_key(settings, timeout, "fixture".into()).map_err(Into::into)
                },
            )
            .unwrap(),
        )
    }

    #[derive(Debug, PartialEq, Eq)]
    struct IndexSnapshot {
        schema: Vec<(String, Option<String>)>,
        metadata: Vec<(String, String)>,
        counts: Vec<i64>,
    }

    fn unchanged_index(store: &Store) -> IndexSnapshot {
        let connection = store.writer().unwrap();
        let schema = connection
            .prepare("SELECT name,sql FROM sqlite_schema ORDER BY name")
            .unwrap()
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let metadata = connection
            .prepare("SELECT key,value FROM meta ORDER BY key")
            .unwrap()
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let counts = [
            "files",
            "units",
            "embedding_tables",
            "embedding_bindings",
            "embedding_transition",
            "embedding_publications",
        ]
        .into_iter()
        .map(|table| {
            connection
                .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                    row.get(0)
                })
                .unwrap()
        })
        .collect();
        IndexSnapshot {
            schema,
            metadata,
            counts,
        }
    }

    async fn mock(server: &MockServer, status: u16, scores: [f32; 3]) {
        Mock::given(method("POST"))
            .and(path("/v1/rerank"))
            .and(header("authorization", "Bearer fixture"))
            .and(header(
                "user-agent",
                concat!("dkc/", env!("CARGO_PKG_VERSION")),
            ))
            .respond_with(
                ResponseTemplate::new(status).set_body_json(json!({"results":[
                {"index":2,"relevance_score":scores[2]},
                {"index":0,"relevance_score":scores[0]},
                {"index":1,"relevance_score":scores[1]}
            ],"error":"private-provider-marker"})),
            )
            .expect(1)
            .mount(server)
            .await;
    }

    #[tokio::test]
    async fn valid_and_bad_relevance_responses_settle_once_in_public_rerank_ledger() {
        for (scores, relevant) in [([0.9, 0.1, 0.4], true), ([0.5, 0.5, 0.5], false)] {
            let server = MockServer::start().await;
            let (root, config, store) = fixture(&server, 12000);
            let before = unchanged_index(&store);
            let work = prepare(root.path(), config);
            let expected_id = format!(
                "rerank:remote:{}",
                blake3::hash(format!("{}/v1/rerank", server.uri()).as_bytes()).to_hex()
            );
            assert_eq!(work.provider_id, expected_id);
            let estimate = work.tokens;
            mock(&server, 200, scores).await;
            assert_eq!(
                Arc::clone(&work)
                    .run(
                        Arc::clone(&store),
                        Instant::now() + Duration::from_secs(3),
                        || 86400
                    )
                    .await
                    .unwrap(),
                relevant
            );
            let calls = server.received_requests().await.unwrap();
            assert_eq!(calls.len(), 1);
            let body: Value = serde_json::from_slice(&calls[0].body).unwrap();
            assert_eq!(body["query"], smoke::QUERY);
            assert_eq!(body["documents"], json!(smoke::DOCUMENTS));
            assert_eq!(body["return_documents"], false);
            assert!(!String::from_utf8_lossy(&calls[0].body).contains("repository-private-marker"));
            let usage: (String, i64, i64, i64) = store
                .writer()
                .unwrap()
                .query_row(
                    "SELECT provider,calls,tokens,estimated_tokens FROM usage",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                )
                .unwrap();
            assert_eq!(usage, (expected_id, 1, estimate, estimate));
            let sent: (String, String, i64, String) = store
                .writer()
                .unwrap()
                .query_row(
                    "SELECT endpoint_kind,endpoint_url,sent_units,state FROM usage_reservations",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                )
                .unwrap();
            assert_eq!(
                sent,
                (
                    "rerank".into(),
                    format!("{}/v1/rerank", server.uri()),
                    3,
                    "reconciled".into()
                )
            );
            assert_eq!(unchanged_index(&store), before);
        }
    }

    #[tokio::test]
    async fn unauthorized_and_rate_limited_responses_keep_one_uncertain_charge_without_retry() {
        for status in [401, 429] {
            let server = MockServer::start().await;
            let (root, config, store) = fixture(&server, 12000);
            let before = unchanged_index(&store);
            let work = prepare(root.path(), config);
            mock(&server, status, [0.9, 0.1, 0.4]).await;
            let error = work
                .run(
                    Arc::clone(&store),
                    Instant::now() + Duration::from_secs(3),
                    || 86400,
                )
                .await
                .unwrap_err();
            assert!(
                matches!(error, admission::Error::Provider(AttemptError::Rerank(crate::models::rerank::Error::Http { status: code, .. })) if code == status)
            );
            assert!(!format!("{error:?}").contains("private-provider-marker"));
            assert_eq!(server.received_requests().await.unwrap().len(), 1);
            let state: (String, i64) = store
                .writer()
                .unwrap()
                .query_row("SELECT state,calls FROM usage_reservations", [], |row| {
                    Ok((row.get(0)?, row.get(1)?))
                })
                .unwrap();
            assert_eq!(state, ("dispatched".into(), 1));
            let settled: i64 = store
                .writer()
                .unwrap()
                .query_row("SELECT COUNT(*) FROM usage", [], |row| row.get(0))
                .unwrap();
            assert_eq!(settled, 0);
            assert_eq!(unchanged_index(&store), before);
        }
    }

    #[tokio::test]
    async fn policy_drift_releases_unsent_reservation_without_touching_vectors() {
        let server = MockServer::start().await;
        let (root, config, store) = fixture(&server, 12000);
        let before = unchanged_index(&store);
        let work = prepare(root.path(), config);
        fs::write(
            root.path().join(".deepkit/config.local.toml"),
            "[search]\ndefault_k=7\n",
        )
        .unwrap();
        let error = work
            .run(
                Arc::clone(&store),
                Instant::now() + Duration::from_secs(3),
                || 86400,
            )
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            admission::Error::Refused(usage::Dispatch::Superseded)
        ));
        assert!(server.received_requests().await.unwrap().is_empty());
        let state: String = store
            .writer()
            .unwrap()
            .query_row("SELECT state FROM usage_reservations", [], |row| row.get(0))
            .unwrap();
        assert_eq!(state, "released");
        assert_eq!(unchanged_index(&store), before);
    }

    #[tokio::test]
    async fn small_candidate_caps_refuse_before_client_and_character_caps_bind_estimate() {
        let server = MockServer::start().await;
        let (root, mut config, store) = fixture(&server, 6);
        let work = prepare(root.path(), config.clone());
        assert_eq!(work.documents, ["The ca", "Banana", "France"]);
        let tokenizer = tiktoken_rs::cl100k_base_singleton();
        let query = tokenizer.encode_with_special_tokens(smoke::QUERY).len();
        let expected: usize = ["The ca", "Banana", "France"]
            .into_iter()
            .map(|document| query + tokenizer.encode_with_special_tokens(document).len() + 1)
            .sum();
        assert_eq!(work.tokens, expected as i64);
        mock(&server, 200, [0.9, 0.1, 0.4]).await;
        work.run(
            Arc::clone(&store),
            Instant::now() + Duration::from_secs(3),
            || 86400,
        )
        .await
        .unwrap();
        let requests = server.received_requests().await.unwrap();
        let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["documents"], json!(["The ca", "Banana", "France"]));
        for candidates in [0, 1, 2] {
            config.rerank.candidates = candidates;
            let constructed = AtomicBool::new(false);
            assert!(
                Rerank::build(
                    root.path().into(),
                    config.clone(),
                    LedgerIdentity::UncheckedFixture,
                    |_, _| {
                        constructed.store(true, Ordering::SeqCst);
                        unreachable!("invalid smoke limits must refuse before client construction")
                    }
                )
                .is_err()
            );
            assert!(!constructed.load(Ordering::SeqCst));
        }
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }
}
