//! Fixed public chat smoke through shared durable admission; v1 §§7–8 and 13.
use super::network::LedgerIdentity;
use crate::{
    config::{self, Config},
    models::{
        attempt::AttemptError,
        openai_compat::chat::{self, Client, Prepared},
    },
    queue::{self, usage, worker::Response},
    search::admission,
    store::Store,
};
use async_trait::async_trait;
use rusqlite::Transaction;
use std::{path::PathBuf, sync::Arc, time::Duration};
use tokio::time::Instant;

pub(super) struct Chat {
    ledger: LedgerIdentity,
    root: PathBuf,
    config: Config,
    provider_id: String,
    endpoint: String,
    provider: Client,
    prepared: Prepared,
    tokens: i64,
}
impl Chat {
    /// Uses a fixed public text with no repository payload or configuration writes.
    pub(super) fn new(
        root: PathBuf,
        config: Config,
        ledger: LedgerIdentity,
    ) -> Result<Self, queue::Error> {
        Self::build(root, config, ledger, Client::from_config)
    }
    fn build(
        root: PathBuf,
        config: Config,
        ledger: LedgerIdentity,
        construct: impl FnOnce(&config::Llm, Duration, u64) -> crate::models::Result<Client>,
    ) -> Result<Self, queue::Error> {
        let endpoint = config::queue::embedding_endpoint(&config.llm.url, "chat/completions")
            .map_err(|_| queue::Error::Invalid("diagnostic chat endpoint"))?;
        let provider_id = chat::provider_id(&endpoint);
        let image_bytes = config
            .limits
            .max_image_bytes()
            .map_err(|_| queue::Error::Invalid("diagnostic image byte limit"))?
            .0;
        let provider = construct(&config.llm, Duration::from_secs(15), image_bytes)
            .map_err(|_| queue::Error::Invalid("diagnostic chat credentials or client"))?;
        let prepared = provider
            .prepare_text(
                "Reply with a short, plain English answer.",
                "What is two plus two?",
                120,
            )
            .map_err(|_| queue::Error::Invalid("diagnostic chat request"))?;
        let tokens = prepared
            .input_tokens
            .checked_add(prepared.max_output_tokens)
            .ok_or(queue::Error::Invalid("diagnostic chat token estimate"))?;
        Ok(Self {
            ledger,
            root,
            config,
            provider_id,
            endpoint,
            provider,
            prepared,
            tokens,
        })
    }
    /// Reports only successful completion, never the provider's arbitrary response text.
    pub(super) async fn run<C>(
        self: Arc<Self>,
        store: Arc<Store>,
        deadline: Instant,
        clock: C,
    ) -> Result<(), admission::Error>
    where
        C: Fn() -> i64 + Send + Sync + 'static,
    {
        let budget = chat::limits(&self.config.llm)?.budget;
        admission::execute(
            store,
            self.provider_id.clone(),
            budget,
            deadline,
            self,
            clock,
        )
        .await
    }
}
#[async_trait]
impl admission::Work for Chat {
    type Output = ();
    fn tokens(&self) -> i64 {
        self.tokens
    }
    fn endpoint(&self) -> Option<usage::Endpoint> {
        Some(usage::Endpoint {
            kind: "llm",
            url: self.endpoint.clone(),
            units: 1,
        })
    }
    fn limits(
        &self,
        _: &Transaction<'_>,
        provider: &str,
    ) -> Result<Option<usage::Limits>, queue::Error> {
        if !self.ledger.valid() {
            return Ok(None);
        }
        let current = config::load(&self.root, None)
            .map_err(|_| queue::Error::Invalid("diagnostic chat configuration reload"))?
            .config;
        if current != self.config || provider != self.provider_id {
            return Ok(None);
        }
        Ok(Some(chat::limits(&current.llm)?))
    }
    async fn call(&self) -> Result<Response<()>, AttemptError> {
        let reply = self.provider.post_once(&self.prepared).await?;
        Ok(Response {
            output: (),
            tokens: reply.tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{self, StoreLimits};
    use serde_json::json;
    use std::fs;
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    fn fixture(server: &MockServer) -> (tempfile::TempDir, Config, Arc<Store>) {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join(".deepkit")).unwrap();
        let mut config = Config::default();
        config.llm.enabled = true;
        config.llm.url = format!("{}/v1", server.uri());
        config.llm.max_calls_per_day = 1;
        fs::write(
            root.path().join(".deepkit/config.toml"),
            config.dump().unwrap(),
        )
        .unwrap();
        let config = config::load(root.path(), None).unwrap().config;
        let store = Arc::new(store::open(":memory:", StoreLimits::default()).unwrap());
        (root, config, store)
    }
    fn prepare(root: &std::path::Path, config: Config) -> Arc<Chat> {
        Arc::new(
            Chat::build(
                root.into(),
                config,
                LedgerIdentity::UncheckedFixture,
                |settings, timeout, cap| {
                    Client::with_key(settings, timeout, cap, "fixture".into()).map_err(Into::into)
                },
            )
            .unwrap(),
        )
    }
    async fn run(work: Arc<Chat>, store: &Arc<Store>) -> Result<(), admission::Error> {
        work.run(
            Arc::clone(store),
            Instant::now() + Duration::from_secs(3),
            || 86400,
        )
        .await
    }
    #[tokio::test]
    async fn chat_smoke_accounts_normal_and_empty_replies_and_shares_budget_across_models() {
        for content in ["Four.", ""] {
            let server = MockServer::start().await;
            let (root, mut config, store) = fixture(&server);
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                    "choices":[{"message":{"content":content}}],
                    "usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}
                })))
                .expect(1)
                .mount(&server)
                .await;
            let work = prepare(root.path(), config.clone());
            let id = work.provider_id.clone();
            let result = run(work, &store).await;
            if content.is_empty() {
                let error = result.unwrap_err();
                assert!(matches!(
                    error,
                    admission::Error::Provider(AttemptError::Chat(
                        chat::Error::ThinkingConsumedBudget { tokens: Some(30) }
                    ))
                ));
                assert!(
                    super::super::network::refusal("chat.probe", error)
                        .message
                        .contains("enable_thinking=false")
                );
            } else {
                result.unwrap();
            }
            let usage: (i64, i64, i64) = store
                .writer()
                .unwrap()
                .query_row("SELECT calls,tokens,estimated_tokens FROM usage", [], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?))
                })
                .unwrap();
            assert_eq!(usage, (1, 30, 0));
            config.llm.model = "another-text-model".into();
            config.llm.vlm_model = "another-image-model".into();
            fs::write(
                root.path().join(".deepkit/config.toml"),
                config.dump().unwrap(),
            )
            .unwrap();
            let next = prepare(root.path(), config);
            assert_eq!(next.provider_id, id);
            assert!(matches!(
                run(next, &store).await,
                Err(admission::Error::Budget { retry_at: 172800 })
            ));
            assert_eq!(server.received_requests().await.unwrap().len(), 1);
        }
    }
    #[tokio::test]
    async fn chat_smoke_refuses_zero_budget_and_configuration_drift_without_requests() {
        for drift in [false, true] {
            let server = MockServer::start().await;
            let (root, mut config, store) = fixture(&server);
            if !drift {
                config.llm.max_calls_per_day = 0;
                fs::write(
                    root.path().join(".deepkit/config.toml"),
                    config.dump().unwrap(),
                )
                .unwrap();
            }
            let work = prepare(root.path(), config);
            if drift {
                fs::write(
                    root.path().join(".deepkit/config.local.toml"),
                    "[llm]\nenabled=false\n",
                )
                .unwrap();
            }
            let error = run(work, &store).await.unwrap_err();
            if drift {
                assert!(matches!(
                    error,
                    admission::Error::Refused(usage::Dispatch::Superseded)
                ));
            } else {
                assert!(matches!(error, admission::Error::Budget { .. }));
            }
            assert!(server.received_requests().await.unwrap().is_empty());
        }
    }
}
