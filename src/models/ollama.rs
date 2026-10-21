//! Ollama embedding provider; see `v1.md` §7.2.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::{Client, Url, header::RETRY_AFTER};
use serde::{Deserialize, Serialize};

use crate::{
    config::OllamaEmbedder as OllamaProfile,
    models::{
        EmbedInput, Embedder as EmbedderTrait, Error, Result,
        attempt::{
            AttemptError, EmbeddingBatch, PreparedQuery, parse_retry_after, query_contract,
            token_count,
        },
        normalize_embedding,
    },
};

const USER_AGENT: &str = concat!("dkc/", env!("CARGO_PKG_VERSION"));

/// An Ollama embedding profile without any remote credentials.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OllamaConfig {
    /// Ollama's HTTP base URL, normally `http://127.0.0.1:11434`.
    pub(crate) url: String,
    /// Locally installed model name.
    pub(crate) model: String,
    /// Search-query instruction, intentionally excluded from documents.
    pub(crate) query_prefix: String,
    /// Maximum input count sent to `/api/embed` in one request.
    pub(crate) max_batch_items: usize,
    /// Dimensions retained for the Deepkit vector table.
    pub(crate) dims: usize,
    /// Total HTTP timeout for an Ollama request.
    pub(crate) timeout: Duration,
}

impl OllamaConfig {
    /// Builds provider settings from the configured Ollama profile.
    pub(crate) fn from_profile(
        profile: &OllamaProfile,
        dims: u64,
        timeout: Duration,
    ) -> Result<Self> {
        let max_batch_items =
            usize::try_from(profile.max_batch_items).map_err(|_| Error::Response {
                message: "Ollama max_batch_items does not fit on this platform".to_owned(),
            })?;
        let dims = usize::try_from(dims).map_err(|_| Error::Response {
            message: "embedding dimensions do not fit on this platform".to_owned(),
        })?;
        if max_batch_items == 0 || dims == 0 {
            return Err(Error::Response {
                message: "Ollama dimensions and max_batch_items must be non-zero".to_owned(),
            });
        }
        Ok(Self {
            url: profile.url.clone(),
            model: profile.model.clone(),
            query_prefix: profile.query_prefix.clone(),
            max_batch_items,
            dims,
            timeout,
        })
    }
}

/// A bounded batched client for Ollama's native `/api/embed` endpoint.
pub(crate) struct OllamaEmbedder {
    client: Client,
    embed_endpoint: Url,
    tags_endpoint: Url,
    id: String,
    model: String,
    query_prefix: String,
    max_batch_items: usize,
    dims: usize,
}

impl OllamaEmbedder {
    /// Builds an Ollama embedder from the configured local profile.
    pub(crate) fn from_config(config: OllamaConfig) -> Result<Self> {
        if config.max_batch_items == 0 || config.dims == 0 {
            return Err(Error::Response {
                message: "Ollama dimensions and max_batch_items must be non-zero".to_owned(),
            });
        }
        let client = Client::builder()
            .use_rustls_tls()
            .user_agent(USER_AGENT)
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .timeout(config.timeout)
            .build()
            .map_err(|_error| Error::Request {
                message: "HTTP client initialization failed".to_owned(),
            })?;
        Ok(Self {
            client,
            embed_endpoint: endpoint(&config.url, "api/embed")?,
            tags_endpoint: endpoint(&config.url, "api/tags")?,
            id: format!("ollama/{}", config.model),
            model: config.model,
            query_prefix: config.query_prefix,
            max_batch_items: config.max_batch_items,
            dims: config.dims,
        })
    }

    /// Reports whether the configured model appears in Ollama's installed-model list.
    ///
    /// This lets the future `doctor` integration detect a missing model without
    /// attempting a costly embedding.
    pub(crate) async fn has_model(&self) -> Result<bool> {
        let response = self
            .client
            .get(self.tags_endpoint.clone())
            .send()
            .await
            .map_err(request_error)?;
        let body = response_body(response, crate::models::http_body::LISTING_BYTES).await?;
        let tags =
            serde_json::from_slice::<TagsResponse>(&body).map_err(|error| Error::Response {
                message: format!("Ollama tags JSON decoding failed: {error}"),
            })?;
        Ok(tags
            .models
            .into_iter()
            .any(|tag| tag.name == self.model || tag.model.as_deref() == Some(self.model.as_str())))
    }

    async fn request_once(
        &self,
        inputs: &[String],
    ) -> std::result::Result<reqwest::Response, reqwest::Error> {
        self.client
            .post(self.embed_endpoint.clone())
            .json(&EmbedRequest {
                model: &self.model,
                input: inputs,
                truncate: false,
            })
            .send()
            .await
    }

    /// Renders the query instruction once and measures the exact admitted input.
    /// The preparation cannot be reused with another endpoint/model/instruction.
    pub(crate) fn prepare_query(
        &self,
        query: &str,
    ) -> std::result::Result<PreparedQuery, AttemptError> {
        PreparedQuery::new(self.query_contract(), &self.query_prefix, query, usize::MAX)
    }
    fn query_contract(&self) -> [u8; 32] {
        query_contract(
            self.embed_endpoint.as_str(),
            &self.model,
            &self.query_prefix,
            self.dims,
        )
    }
    /// Executes at most one admitted query request, retaining reported usage.
    /// Caller owns shared admission, deadline, and any separately charged retry.
    pub(crate) async fn embed_query_with_usage_once(
        &self,
        query: &PreparedQuery,
    ) -> std::result::Result<EmbeddingBatch, AttemptError> {
        let input = query.input(self.query_contract())?.to_owned();
        self.embed_inputs_once(&[input]).await
    }

    #[cfg(test)]
    pub(crate) async fn embed_documents_once(
        &self,
        documents: &[EmbedInput<'_>],
    ) -> std::result::Result<Vec<Vec<f32>>, AttemptError> {
        self.embed_documents_with_usage_once(documents)
            .await
            .map(|batch| batch.embeddings)
    }

    /// Embeds one admitted, nonempty document batch with at most one HTTP request.
    /// No implicit splitting, model lookup, redirects or transport retries occur.
    /// Callers enforce canonical per-document token caps before provider admission.
    pub(crate) async fn embed_documents_with_usage_once(
        &self,
        documents: &[EmbedInput<'_>],
    ) -> std::result::Result<EmbeddingBatch, AttemptError> {
        if documents.is_empty() || documents.len() > self.max_batch_items {
            return Err(AttemptError::InvalidInput);
        }
        let inputs = documents
            .iter()
            .map(|d| format!("{}\n{}", d.header, d.text))
            .collect::<Vec<_>>();
        self.embed_inputs_once(&inputs).await
    }

    async fn embed_inputs_once(
        &self,
        inputs: &[String],
    ) -> std::result::Result<EmbeddingBatch, AttemptError> {
        let response = self
            .request_once(inputs)
            .await
            .map_err(|_| AttemptError::Retryable {
                status: None,
                retry_after: None,
            })?;
        let status = response.status();
        if !status.is_success() {
            if status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
                return Err(AttemptError::Retryable {
                    status: Some(status.as_u16()),
                    retry_after: parse_retry_after(response.headers().get(RETRY_AFTER)),
                });
            }
            return Err(AttemptError::Rejected {
                status: status.as_u16(),
            });
        }
        let body =
            crate::models::http_body::read(response, crate::models::http_body::EMBEDDING_BYTES)
                .await
                .map_err(|error| match error {
                    crate::models::http_body::Error::TooLarge => AttemptError::InvalidResponse,
                    crate::models::http_body::Error::Transport => AttemptError::Retryable {
                        status: Some(status.as_u16()),
                        retry_after: None,
                    },
                })?;
        let embeddings = self
            .decode_embeddings(&body, inputs.len())
            .map_err(|_| AttemptError::InvalidResponse)?;
        let usage: UsageEnvelope =
            serde_json::from_slice(&body).map_err(|_| AttemptError::InvalidResponse)?;
        Ok(EmbeddingBatch {
            embeddings,
            tokens: token_count(usage.prompt_eval_count)?,
        })
    }

    async fn embed_texts(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let response = self.request_once(&inputs).await.map_err(request_error)?;
        let body = response_body(response, crate::models::http_body::EMBEDDING_BYTES).await?;
        self.decode_embeddings(&body, inputs.len())
    }

    fn decode_embeddings(&self, body: &[u8], expected: usize) -> Result<Vec<Vec<f32>>> {
        let response =
            serde_json::from_slice::<EmbedResponse>(body).map_err(|error| Error::Response {
                message: format!("Ollama embed JSON decoding failed: {error}"),
            })?;
        if response.embeddings.len() != expected {
            return Err(Error::Response {
                message: format!(
                    "Ollama returned {} embeddings for {} inputs",
                    response.embeddings.len(),
                    expected
                ),
            });
        }
        response
            .embeddings
            .into_iter()
            .map(|embedding| normalize_embedding(embedding, self.dims))
            .collect()
    }
}

#[async_trait]
impl EmbedderTrait for OllamaEmbedder {
    fn id(&self) -> &str {
        &self.id
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn max_batch_tokens(&self) -> usize {
        // Ollama has no request-level token batch setting. `embedder.max_doc_tokens`
        // bounds each canonical input before it reaches this provider.
        usize::MAX
    }

    fn max_batch_items(&self) -> usize {
        self.max_batch_items
    }

    async fn embed_documents(&self, documents: &[EmbedInput<'_>]) -> Result<Vec<Vec<f32>>> {
        let texts = documents
            .iter()
            .map(|document| {
                let _ = document.kind;
                format!("{}\n{}", document.header, document.text)
            })
            .collect::<Vec<_>>();
        let mut embeddings = Vec::with_capacity(texts.len());
        for batch in texts.chunks(self.max_batch_items) {
            embeddings.extend(self.embed_texts(batch.to_owned()).await?);
        }
        Ok(embeddings)
    }

    async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let input = format!("{}{}", self.query_prefix, query);
        let mut embeddings = self.embed_texts(vec![input]).await?;
        embeddings.pop().ok_or_else(|| Error::Response {
            message: "Ollama response had no query vector".to_owned(),
        })
    }
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
    truncate: bool,
}

#[derive(Deserialize)]
struct UsageEnvelope {
    prompt_eval_count: Option<u64>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct TagsResponse {
    models: Vec<ModelTag>,
}

#[derive(Deserialize)]
struct ModelTag {
    name: String,
    #[serde(default)]
    model: Option<String>,
}

async fn response_body(response: reqwest::Response, cap: usize) -> Result<Vec<u8>> {
    let status = response.status();
    if !status.is_success() {
        return Err(Error::Http {
            status: status.as_u16(),
        });
    }
    crate::models::http_body::read(response, cap)
        .await
        .map_err(|error| match error {
            crate::models::http_body::Error::TooLarge => Error::Response {
                message: "Provider response exceeds the supported size limit".into(),
            },
            crate::models::http_body::Error::Transport => Error::Request {
                message: "transport failure".into(),
            },
        })
}

fn request_error(_error: reqwest::Error) -> Error {
    Error::Request {
        message: "transport failure".to_owned(),
    }
}

fn endpoint(base: &str, path: &str) -> Result<Url> {
    Url::parse(&format!("{}/{}", base.trim_end_matches('/'), path))
        .map_err(|error| Error::Endpoint(error.to_string()))
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, Mutex},
        time::Duration,
    };

    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{header, method, path},
    };

    use super::{EmbedInput, EmbedderTrait, Error, OllamaConfig, OllamaEmbedder};
    use crate::store::UnitKind;

    const SUCCESS: &str = include_str!("../../fixtures/ollama/embed-success.json");
    const TAGS: &str = include_str!("../../fixtures/ollama/tags.json");
    const MALFORMED: &str = include_str!("../../fixtures/ollama/embed-malformed.json");

    fn provider(server: &MockServer, model: &str) -> OllamaEmbedder {
        OllamaEmbedder::from_config(OllamaConfig {
            url: server.uri(),
            model: model.to_owned(),
            query_prefix: "Instruct: test\nQuery: ".to_owned(),
            max_batch_items: 32,
            dims: 4,
            timeout: Duration::from_secs(2),
        })
        .unwrap()
    }

    #[tokio::test]
    async fn admitted_response_keeps_actual_usage_and_rejects_invalid_counts() {
        use serde_json::json;
        for (count, expected) in [
            (None, Ok(None)),
            (Some(json!(null)), Ok(None)),
            (Some(json!(0)), Ok(Some(0))),
            (Some(json!(42)), Ok(Some(42))),
            (Some(json!(-1)), Err(())),
            (Some(json!(1.5)), Err(())),
            (Some(json!("untrusted-usage-marker")), Err(())),
            (Some(json!(u64::MAX)), Err(())),
        ] {
            let server = MockServer::start().await;
            let mut body: serde_json::Value = serde_json::from_str(SUCCESS).unwrap();
            body.as_object_mut().unwrap().remove("prompt_eval_count");
            if let Some(count) = count {
                body["prompt_eval_count"] = count;
            }
            Mock::given(method("POST"))
                .and(path("/api/embed"))
                .respond_with(ResponseTemplate::new(200).set_body_json(body))
                .expect(1)
                .mount(&server)
                .await;
            let p = provider(&server, "usage-response");
            let result = p
                .embed_documents_with_usage_once(&[EmbedInput::from_context(
                    "context",
                    "body",
                    UnitKind::Symbol,
                )])
                .await;
            match (result, expected) {
                (Ok(batch), Ok(tokens)) => {
                    assert_eq!(batch.embeddings.len(), 1);
                    assert_eq!(batch.tokens, tokens);
                }
                (Err(error), Err(())) => {
                    assert!(matches!(
                        error,
                        crate::models::attempt::AttemptError::InvalidResponse
                    ));
                    assert!(!format!("{error:?}").contains("untrusted-usage-marker"));
                }
                (actual, expected) => {
                    panic!("unexpected usage result {actual:?}, expected {expected:?}")
                }
            }
            server.verify().await;
        }
    }

    #[tokio::test]
    async fn embeds_canonical_documents_and_prefixes_queries_only()
    -> Result<(), Box<dyn std::error::Error>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .and(header(
                "user-agent",
                concat!("dkc/", env!("CARGO_PKG_VERSION")),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_raw(SUCCESS, "application/json"))
            .mount(&server)
            .await;
        let provider = provider(&server, "qwen3-embedding:0.6b");
        let input = EmbedInput::from_context(
            "repo › src › lib.rs\nfn example\nrust function",
            "fn example() {}",
            UnitKind::Symbol,
        );

        let document = provider.embed_documents(&[input]).await?;
        let query = provider.embed_query("example").await?;

        assert_eq!(document[0], vec![0.6, 0.8, 0.0, 0.0]);
        assert_eq!(query, vec![0.6, 0.8, 0.0, 0.0]);
        let requests = server
            .received_requests()
            .await
            .ok_or("request capture unavailable")?;
        let document_request: serde_json::Value = serde_json::from_slice(&requests[0].body)?;
        let query_request: serde_json::Value = serde_json::from_slice(&requests[1].body)?;
        assert_eq!(
            document_request["input"][0],
            "repo › src › lib.rs\nfn example\nrust function\nfn example() {}"
        );
        assert_eq!(query_request["input"][0], "Instruct: test\nQuery: example");
        assert_eq!(document_request["truncate"], false);
        assert_eq!(query_request["truncate"], false);
        Ok(())
    }

    #[tokio::test]
    async fn batches_documents_at_the_configured_item_limit()
    -> Result<(), Box<dyn std::error::Error>> {
        let server = MockServer::start().await;
        let counts = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&counts);
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(move |request: &wiremock::Request| {
                let request = serde_json::from_slice::<serde_json::Value>(&request.body)
                    .unwrap_or(serde_json::Value::Null);
                let count = request["input"].as_array().map_or(0, Vec::len);
                if let Ok(mut counts) = seen.lock() {
                    counts.push(count);
                }
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "embeddings": (0..count).map(|_| vec![1.0_f32, 0.0, 0.0, 0.0]).collect::<Vec<_>>()
                }))
            })
            .mount(&server)
            .await;
        let provider = provider(&server, "batch-model");
        let texts = (0..33)
            .map(|index| format!("fn item_{index}() {{}}"))
            .collect::<Vec<_>>();
        let inputs = texts
            .iter()
            .map(|text| {
                EmbedInput::from_context(
                    "repo › items.rs\nfn item\nrust function",
                    text,
                    UnitKind::Chunk,
                )
            })
            .collect::<Vec<_>>();

        let embeddings = provider.embed_documents(&inputs).await?;

        assert_eq!(embeddings.len(), 33);
        assert_eq!(
            *counts.lock().map_err(|_| "batch capture poisoned")?,
            vec![32, 1]
        );
        Ok(())
    }

    #[tokio::test]
    async fn tags_reports_the_configured_model_and_invalid_embed_payloads_are_typed()
    -> Result<(), Box<dyn std::error::Error>> {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(TAGS, "application/json"))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(MALFORMED, "application/json"))
            .mount(&server)
            .await;
        let present = provider(&server, "qwen3-embedding:0.6b");
        let absent = provider(&server, "missing-model:latest");

        assert!(present.has_model().await?);
        assert!(!absent.has_model().await?);
        assert!(matches!(
            present.embed_query("broken").await,
            Err(Error::Response { .. })
        ));
        Ok(())
    }

    #[tokio::test]
    async fn refuses_invalid_vectors_counts_and_tags_without_retrying()
    -> Result<(), Box<dyn std::error::Error>> {
        for body in [
            r#"{"embeddings":[]}"#,
            r#"{"embeddings":[[0,0,0,0]]}"#,
            r#"{"embeddings":[[1,0]]}"#,
        ] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/api/embed"))
                .respond_with(ResponseTemplate::new(200).set_body_raw(body, "application/json"))
                .expect(1)
                .mount(&server)
                .await;
            assert!(
                provider(&server, "test")
                    .embed_query("query")
                    .await
                    .is_err()
            );
        }
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"error":"bad response"})),
            )
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(503))
            .expect(1)
            .mount(&server)
            .await;
        let client = provider(&server, "test");
        assert!(matches!(
            client.has_model().await,
            Err(Error::Response { .. })
        ));
        assert!(matches!(
            client.embed_query("query").await,
            Err(Error::Http { status: 503 })
        ));
        assert!(client.embed_documents(&[]).await?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn live_ollama_embedder_uses_the_configured_model()
    -> Result<(), Box<dyn std::error::Error>> {
        if std::env::var("DKC_LIVE_OLLAMA").as_deref() != Ok("1") {
            return Ok(());
        }
        let profile = crate::config::OllamaEmbedder::default();
        let config = OllamaConfig::from_profile(&profile, 1_024, Duration::from_secs(30))?;
        let provider = OllamaEmbedder::from_config(config)?;
        assert!(
            provider.has_model().await?,
            "configured Ollama model is not installed"
        );
        let first = "fn local_embedder() {}".to_owned();
        let second = "fn local_query() {}".to_owned();
        let inputs = [
            EmbedInput::from_context(
                "repo › lib.rs\nfn local_embedder\nrust function",
                &first,
                UnitKind::Chunk,
            ),
            EmbedInput::from_context(
                "repo › lib.rs\nfn local_query\nrust function",
                &second,
                UnitKind::Chunk,
            ),
        ];

        let embeddings = provider.embed_documents(&inputs).await?;

        assert_eq!(embeddings.len(), 2);
        assert!(embeddings.iter().all(|embedding| embedding.len() == 1_024));
        let query = provider.embed_query("local embedder").await?;
        for embedding in embeddings.iter().chain(std::iter::once(&query)) {
            assert!(embedding.iter().all(|value| value.is_finite()));
            let norm = embedding
                .iter()
                .map(|value| f64::from(*value).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!((norm - 1.0).abs() < 1e-5);
        }
        eprintln!(
            "Ollama live smoke: model={}, documents=2, query=1, dims={}, finite unit norms verified",
            provider.id(),
            provider.dims()
        );
        Ok(())
    }
    #[tokio::test]
    async fn queued_attempts_classify_errors_without_retaining_response_payloads() {
        use crate::models::attempt::AttemptError;
        for status in [401, 404, 429, 500, 503, 200] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/api/embed"))
                .and(header("user-agent", super::USER_AGENT))
                .respond_with(
                    ResponseTemplate::new(status)
                        .insert_header("Retry-After", "120")
                        .set_body_string("private-ollama-response-marker"),
                )
                .expect(1)
                .mount(&server)
                .await;
            let p = provider(&server, "queued-errors");
            let error = p
                .embed_documents_once(&[EmbedInput::from_context(
                    "context",
                    "body",
                    UnitKind::Symbol,
                )])
                .await
                .unwrap_err();
            assert!(!format!("{error:?} {error}").contains("private-ollama-response-marker"));
            match status {
                429 | 500 | 503 => assert!(
                    matches!(error,AttemptError::Retryable{retry_after:Some(d),..} if d==Duration::from_secs(120))
                ),
                200 => assert!(matches!(error, AttemptError::InvalidResponse)),
                _ => assert!(matches!(error,AttemptError::Rejected{status:s} if s==status)),
            }
            server.verify().await;
        }
        for body in [
            SUCCESS,
            "{\"embeddings\":[]}",
            "{\"embeddings\":[[0,0,0,0]]}",
            "{\"embeddings\":[[1,0]]}",
        ] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(200).set_body_raw(body, "application/json"))
                .expect(1)
                .mount(&server)
                .await;
            let p = provider(&server, "queued-vectors");
            let result = p
                .embed_documents_once(&[EmbedInput::from_context(
                    "context",
                    "body",
                    UnitKind::Symbol,
                )])
                .await;
            if body == SUCCESS {
                assert_eq!(result.unwrap(), vec![vec![0.6, 0.8, 0.0, 0.0]]);
            } else {
                assert!(matches!(result, Err(AttemptError::InvalidResponse)));
            }
            server.verify().await;
        }
    }

    #[tokio::test]
    async fn queued_batches_never_split_follow_redirects_or_lookup_models() {
        use crate::models::attempt::AttemptError;
        let server = MockServer::start().await;
        let p = provider(&server, "queued-bounds");
        assert!(matches!(
            p.embed_documents_once(&[]).await,
            Err(AttemptError::InvalidInput)
        ));
        let docs = (0..33)
            .map(|_| EmbedInput::from_context("context", "body", UnitKind::Symbol))
            .collect::<Vec<_>>();
        assert!(matches!(
            p.embed_documents_once(&docs).await,
            Err(AttemptError::InvalidInput)
        ));
        assert!(server.received_requests().await.unwrap().is_empty());
        Mock::given(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(307)
                    .insert_header("Location", format!("{}/redirect", server.uri())),
            )
            .expect(1)
            .mount(&server)
            .await;
        assert!(matches!(
            p.embed_documents_once(&docs[..1]).await,
            Err(AttemptError::Rejected { status: 307 })
        ));
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["input"][0], "context\nbody");
        assert_eq!(body["truncate"], false);
    }

    #[tokio::test]
    async fn queued_timeout_is_one_retryable_attempt() {
        use crate::models::attempt::AttemptError;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(SUCCESS, "application/json")
                    .set_delay(Duration::from_secs(3)),
            )
            .expect(1)
            .mount(&server)
            .await;
        let p = provider(&server, "queued-timeout");
        let docs = [EmbedInput::from_context(
            "context",
            "body",
            UnitKind::Symbol,
        )];
        let result = tokio::time::timeout(Duration::from_secs(5), p.embed_documents_once(&docs))
            .await
            .unwrap();
        assert!(matches!(
            result,
            Err(AttemptError::Retryable {
                status: None,
                retry_after: None
            })
        ));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn queued_ollama_http_retries_stop_at_eight_and_permanent_errors_stop_at_one() {
        use crate::{
            queue::{self, retry, usage},
            store::{self, StoreLimits},
        };
        for (status, expected) in [(429, 8), (404, 1)] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/api/embed"))
                .respond_with(ResponseTemplate::new(status).insert_header("Retry-After", "0"))
                .expect(expected as u64)
                .mount(&server)
                .await;
            let p = provider(&server, "queued-counted-retries");
            let store = store::open(":memory:", StoreLimits::default()).unwrap();
            queue::enqueue(
                &store,
                &queue::Enqueue {
                    kind: queue::Kind::Embed,
                    target: queue::Target::Unit,
                    target_id: 1,
                    target_hash: &[1; 32],
                    desired_hash: &[2; 32],
                    provider: "embed:ollama",
                    priority: 1.0,
                    tokens: 10,
                },
                0,
            )
            .unwrap();
            let owner = queue::Owner::new(&store).unwrap();
            let budget = usage::Budget {
                calls: None,
                tokens: None,
            };
            let mut now = 0;
            for attempt in 1..=expected {
                let lease = queue::claim(&store, "embed:ollama", &owner, now, 1, 10)
                    .unwrap()
                    .remove(0);
                let usage::Admission::Reserved(reservation) =
                    usage::reserve(&store, "embed:ollama", 10, 0, budget, Some(&lease), || now)
                        .unwrap()
                else {
                    panic!("reservation denied")
                };
                assert_eq!(
                    usage::mark_dispatched(
                        &store,
                        &reservation,
                        Some(&lease),
                        || now,
                        |_, _| Ok(usage::Limits {
                            budget,
                            rpm: None,
                            tpm: None
                        })
                    )
                    .unwrap(),
                    usage::Dispatch::Recorded
                );
                let error = p
                    .embed_documents_once(&[EmbedInput::from_context(
                        "context",
                        "body",
                        UnitKind::Symbol,
                    )])
                    .await
                    .unwrap_err();
                let outcome = retry::schedule(
                    &store,
                    &lease,
                    retry::Outcome::from_provider_attempt(&error),
                    || now,
                )
                .unwrap();
                if attempt < expected {
                    let retry::Scheduled::RetryAt(next) = outcome else {
                        panic!("retry missing")
                    };
                    now = next;
                } else {
                    assert_eq!(outcome, retry::Scheduled::Dead);
                }
            }
            assert!(
                queue::claim(&store, "embed:ollama", &owner, now + 1000, 1, 10)
                    .unwrap()
                    .is_empty()
            );
            assert_eq!(server.received_requests().await.unwrap().len(), expected);
            server.verify().await;
        }
    }
    #[tokio::test]
    async fn prepared_query_attempt_is_bound_and_preserves_usage_without_hidden_calls() {
        use crate::models::attempt::AttemptError;
        let server = MockServer::start().await;
        let mut body: serde_json::Value = serde_json::from_str(SUCCESS).unwrap();
        body["prompt_eval_count"] = serde_json::json!(17);
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .expect(1)
            .mount(&server)
            .await;
        let p = provider(&server, "prepared-query-ollama");
        let query = p.prepare_query("find retries").unwrap();
        assert!(matches!(
            provider(&server, "other-prepared-query")
                .embed_query_with_usage_once(&query)
                .await,
            Err(AttemptError::InvalidInput)
        ));
        assert!(p.prepare_query("   ").is_err());
        let response = p.embed_query_with_usage_once(&query).await.unwrap();
        assert_eq!(response.tokens, Some(17));
        assert!((response.embeddings[0][0] - 0.6).abs() < 1e-6);
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(
            body["input"],
            serde_json::json!(["Instruct: test\nQuery: find retries"])
        );
        assert_eq!(body["truncate"], false);
        assert_eq!(
            query.tokens() as usize,
            tiktoken_rs::cl100k_base_singleton()
                .encode_with_special_tokens(body["input"][0].as_str().unwrap())
                .len()
        );
    }
    #[tokio::test]
    async fn oversized_embedding_and_tags_bodies_are_refused_without_retrying() {
        use crate::models::{
            attempt::AttemptError,
            http_body::{self, tests::serve_once},
        };
        for tags in [false, true] {
            let cap = if tags {
                http_body::LISTING_BYTES
            } else {
                http_body::EMBEDDING_BYTES
            };
            let (url, task) =
                serve_once(200, format!("Content-Length: {}\r\n", cap + 1), Vec::new()).await;
            let config = OllamaConfig {
                url,
                model: "fixture".into(),
                query_prefix: String::new(),
                max_batch_items: 1,
                dims: 4,
                timeout: Duration::from_secs(2),
            };
            let provider = OllamaEmbedder::from_config(config).unwrap();
            if tags {
                let error = provider.has_model().await.err().unwrap();
                assert!(matches!(error, Error::Response { .. }));
                assert!(error.to_string().contains("size limit"));
            } else {
                let query = provider.prepare_query("fixed diagnostic query").unwrap();
                assert!(matches!(
                    provider.embed_query_with_usage_once(&query).await,
                    Err(AttemptError::InvalidResponse)
                ));
            }
            task.await.unwrap();
        }
    }
}
