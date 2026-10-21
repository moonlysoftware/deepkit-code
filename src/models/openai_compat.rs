//! OpenAI-compatible remote embedding provider; see `v1.md` §§7.1–7.2 and 11.
pub(crate) mod chat;

use std::{
    collections::HashSet,
    path::Path,
    sync::{Mutex, OnceLock},
    time::Duration,
};

use async_trait::async_trait;
use backon::{ExponentialBuilder, Retryable};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use reqwest::{Client, StatusCode, Url, header::RETRY_AFTER};
use serde::{Deserialize, Serialize};

use crate::{
    config::{self, RemoteEmbedder},
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
const RETRY_ATTEMPTS: usize = 3;

/// Models whose provider rejected `dimensions`; retained for the process lifetime.
static MRL_DIMENSIONS_UNSUPPORTED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

/// A remote OpenAI-compatible embedding profile without secret material.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OpenAiConfig {
    /// OpenAI-compatible API base URL, usually ending in `/v1`.
    pub(crate) url: String,
    /// Model sent in the embeddings request body.
    pub(crate) model: String,
    /// Environment variable to consult before the credentials file.
    pub(crate) api_key_env: String,
    /// Response encoding requested from the provider.
    pub(crate) encoding_format: String,
    /// Search-query instruction, intentionally excluded from documents.
    pub(crate) query_prefix: String,
    /// Maximum document count in one provider request.
    pub(crate) max_batch_items: usize,
    /// Maximum estimated tokens in one provider request.
    pub(crate) max_batch_tokens: usize,
    /// Number of dimensions retained for the local vector table.
    pub(crate) dims: usize,
    /// Total HTTP timeout for an individual provider attempt.
    pub(crate) timeout: Duration,
}

impl OpenAiConfig {
    /// Builds the provider settings from the configured remote embedding profile.
    pub(crate) fn from_remote(
        remote: &RemoteEmbedder,
        dims: u64,
        timeout: Duration,
    ) -> Result<Self> {
        let max_batch_items =
            usize::try_from(remote.max_batch_items).map_err(|_| Error::Response {
                message: "max_batch_items does not fit on this platform".to_owned(),
            })?;
        let max_batch_tokens =
            usize::try_from(remote.max_batch_tokens).map_err(|_| Error::Response {
                message: "max_batch_tokens does not fit on this platform".to_owned(),
            })?;
        let dims = usize::try_from(dims).map_err(|_| Error::Response {
            message: "embedding dimensions do not fit on this platform".to_owned(),
        })?;
        if max_batch_items == 0 || max_batch_tokens == 0 || dims == 0 {
            return Err(Error::Response {
                message: "embedding dimensions and batch limits must be non-zero".to_owned(),
            });
        }
        Ok(Self {
            url: remote.url.clone(),
            model: remote.model.clone(),
            api_key_env: remote.api_key_env.clone(),
            encoding_format: remote.encoding_format.clone(),
            query_prefix: remote.query_prefix.clone(),
            max_batch_items,
            max_batch_tokens,
            dims,
            timeout,
        })
    }
}

/// A remote provider that speaks the OpenAI embeddings wire format.
///
/// The bearer token is held only in the HTTP client path and this type deliberately
/// does not implement `Debug`, preventing accidental secret formatting.
pub(crate) struct OpenAiEmbedder {
    client: Client,
    endpoint: Url,
    api_key: String,
    id: String,
    dimensions_cache_key: String,
    model: String,
    query_prefix: String,
    max_batch_items: usize,
    max_batch_tokens: usize,
    dims: usize,
}

impl OpenAiEmbedder {
    /// Resolves the key and builds a provider from the repository configuration.
    pub(crate) fn from_config(config: OpenAiConfig) -> Result<Self> {
        let endpoint = embeddings_endpoint(&config.url)?;
        let api_key = resolve_api_key(
            &config.api_key_env,
            &endpoint,
            config::credentials_path().as_deref(),
        )?;
        Self::with_api_key(config, api_key)
    }

    /// Builds a provider with a caller-supplied key for injected credential sources.
    pub(crate) fn with_api_key(config: OpenAiConfig, api_key: String) -> Result<Self> {
        if !config.encoding_format.eq_ignore_ascii_case("base64") {
            return Err(Error::UnsupportedEncoding(config.encoding_format));
        }
        if config.max_batch_items == 0 || config.max_batch_tokens == 0 || config.dims == 0 {
            return Err(Error::Response {
                message: "embedding dimensions and batch limits must be non-zero".to_owned(),
            });
        }
        let endpoint = embeddings_endpoint(&config.url)?;
        let dimensions_cache_key = format!("{}|{}|{}", endpoint, config.model, config.dims);
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
            endpoint,
            api_key,
            id: format!("remote/{}", config.model),
            dimensions_cache_key,
            model: config.model,
            query_prefix: config.query_prefix,
            max_batch_items: config.max_batch_items,
            max_batch_tokens: config.max_batch_tokens,
            dims: config.dims,
        })
    }

    async fn embed_texts(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let send_dimensions = self.dimensions_supported();
        let response = match self.request_embeddings(&inputs, send_dimensions).await {
            Err(error) if send_dimensions && error.is_matryoshka_rejection() => {
                // Moonly's Qwen3-Embedding-8B deployment rejects the dimensions
                // parameter. The model is MRL, so retain the configured prefix and
                // normalize it client-side on this and all later requests.
                self.remember_dimensions_unsupported();
                self.request_embeddings(&inputs, false)
                    .await
                    .map_err(ProviderError::into_public)?
            }
            Err(error) => return Err(error.into_public()),
            Ok(response) => response,
        };
        decode_embeddings(&response, inputs.len(), self.dims)
    }

    fn dimensions_supported(&self) -> bool {
        let unsupported = unsupported_models()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        !unsupported.contains(&self.dimensions_cache_key)
    }

    /// Records a previously observed rejection so later requests omit `dimensions`.
    pub(crate) fn remember_dimensions_unsupported(&self) {
        let mut unsupported = unsupported_models()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        unsupported.insert(self.dimensions_cache_key.clone());
    }

    #[cfg(test)]
    pub(crate) fn forget_dimensions_unsupported_for_test(&self) {
        let mut unsupported = unsupported_models()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        unsupported.remove(&self.dimensions_cache_key);
    }

    async fn request_embeddings(
        &self,
        inputs: &[String],
        send_dimensions: bool,
    ) -> std::result::Result<Vec<u8>, ProviderError> {
        (|| self.request_once(inputs, send_dimensions))
            .retry(ExponentialBuilder::default().with_max_times(RETRY_ATTEMPTS))
            .sleep(tokio::time::sleep)
            .when(ProviderError::is_retryable)
            .adjust(|error, delay| error.retry_after.or(delay))
            .await
    }

    async fn request_once(
        &self,
        inputs: &[String],
        send_dimensions: bool,
    ) -> std::result::Result<Vec<u8>, ProviderError> {
        let request = EmbeddingRequest {
            input: inputs.to_owned(),
            model: self.model.clone(),
            encoding_format: "base64",
            dimensions: send_dimensions.then_some(self.dims),
        };
        let response = self
            .client
            .post(self.endpoint.clone())
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await
            .map_err(ProviderError::request)?;
        provider_response(response).await
    }

    /// Renders the query instruction once and measures the exact admitted input.
    /// The preparation cannot be reused with another endpoint/model/instruction.
    pub(crate) fn prepare_query(
        &self,
        query: &str,
    ) -> std::result::Result<PreparedQuery, AttemptError> {
        PreparedQuery::new(
            self.query_contract(),
            &self.query_prefix,
            query,
            self.max_batch_tokens,
        )
    }
    fn query_contract(&self) -> [u8; 32] {
        query_contract(
            self.endpoint.as_str(),
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
        if query.tokens() as u128 > self.max_batch_tokens as u128 {
            return Err(AttemptError::InvalidInput);
        }
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

    /// Executes at most one HTTP request for one already-admitted document batch.
    /// Never splits batches, sleeps, follows redirects or retries dimensions here.
    /// A dimensions rejection records the capability; the queue must admit and
    /// account for the next call separately. Invalid input performs no request.
    pub(crate) async fn embed_documents_with_usage_once(
        &self,
        documents: &[EmbedInput<'_>],
    ) -> std::result::Result<EmbeddingBatch, AttemptError> {
        if documents.is_empty() || documents.len() > self.max_batch_items {
            return Err(AttemptError::InvalidInput);
        }
        let mut batches = self
            .batches(documents)
            .map_err(|_| AttemptError::InvalidInput)?;
        if batches.len() != 1 {
            return Err(AttemptError::InvalidInput);
        }
        let inputs = batches.pop().ok_or(AttemptError::InvalidInput)?;
        self.embed_inputs_once(&inputs).await
    }

    async fn embed_inputs_once(
        &self,
        inputs: &[String],
    ) -> std::result::Result<EmbeddingBatch, AttemptError> {
        let dimensions = self.dimensions_supported();
        let response = match self.request_once(inputs, dimensions).await {
            Ok(body) => body,
            Err(error) if dimensions && error.is_matryoshka_rejection() => {
                self.remember_dimensions_unsupported();
                return Err(AttemptError::DimensionsRejected);
            }
            Err(error) if matches!(&error.error, Error::Response { .. }) => {
                return Err(AttemptError::InvalidResponse);
            }
            Err(error) if error.retryable => {
                return Err(AttemptError::Retryable {
                    status: error.status.map(|s| s.as_u16()),
                    retry_after: error.retry_after,
                });
            }
            Err(error) => {
                return Err(AttemptError::Rejected {
                    status: error.status.map_or(0, |s| s.as_u16()),
                });
            }
        };
        let embeddings = decode_embeddings(&response, inputs.len(), self.dims)
            .map_err(|_| AttemptError::InvalidResponse)?;
        let usage: UsageEnvelope =
            serde_json::from_slice(&response).map_err(|_| AttemptError::InvalidResponse)?;
        Ok(EmbeddingBatch {
            embeddings,
            tokens: token_count(usage.usage.and_then(|usage| usage.total_tokens))?,
        })
    }

    fn batches(&self, documents: &[EmbedInput<'_>]) -> Result<Vec<Vec<String>>> {
        let tokenizer = tiktoken_rs::cl100k_base_singleton();
        let mut batches = Vec::new();
        let mut batch = Vec::new();
        let mut batch_tokens = 0_usize;

        for document in documents {
            let _ = document.kind;
            let rendered = format!("{}\n{}", document.header, document.text);
            let tokens = tokenizer.encode_with_special_tokens(&rendered).len();
            if tokens > self.max_batch_tokens {
                return Err(Error::BatchTooLarge {
                    tokens,
                    max_tokens: self.max_batch_tokens,
                });
            }
            if !batch.is_empty()
                && (batch.len() == self.max_batch_items
                    || batch_tokens.saturating_add(tokens) > self.max_batch_tokens)
            {
                batches.push(std::mem::take(&mut batch));
                batch_tokens = 0;
            }
            batch_tokens = batch_tokens.saturating_add(tokens);
            batch.push(rendered);
        }
        if !batch.is_empty() {
            batches.push(batch);
        }
        Ok(batches)
    }
}

#[async_trait]
impl EmbedderTrait for OpenAiEmbedder {
    fn id(&self) -> &str {
        &self.id
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn max_batch_tokens(&self) -> usize {
        self.max_batch_tokens
    }

    fn max_batch_items(&self) -> usize {
        self.max_batch_items
    }

    async fn embed_documents(&self, documents: &[EmbedInput<'_>]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(documents.len());
        for batch in self.batches(documents)? {
            embeddings.extend(self.embed_texts(batch).await?);
        }
        Ok(embeddings)
    }

    async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let input = format!("{}{}", self.query_prefix, query);
        let mut embeddings = self.embed_texts(vec![input]).await?;
        embeddings.pop().ok_or_else(|| Error::Response {
            message: "embedding response had no query vector".to_owned(),
        })
    }
}

#[derive(Clone, Serialize)]
struct EmbeddingRequest {
    input: Vec<String>,
    model: String,
    encoding_format: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

#[derive(Deserialize)]
struct UsageEnvelope {
    usage: Option<EmbeddingUsage>,
}
#[derive(Deserialize)]
struct EmbeddingUsage {
    total_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingDatum>,
}

#[derive(Deserialize)]
struct EmbeddingDatum {
    index: usize,
    embedding: String,
}

struct ProviderError {
    error: Error,
    status: Option<StatusCode>,
    response_body: String,
    retry_after: Option<Duration>,
    retryable: bool,
}

impl ProviderError {
    fn request(error: reqwest::Error) -> Self {
        let _ = error;
        Self::body(crate::models::http_body::Error::Transport)
    }

    fn body(error: crate::models::http_body::Error) -> Self {
        let oversized = matches!(error, crate::models::http_body::Error::TooLarge);
        Self {
            error: if oversized {
                Error::Response {
                    message: "Provider response exceeds the supported size limit".into(),
                }
            } else {
                Error::Request {
                    message: "transport failure".into(),
                }
            },
            status: None,
            response_body: String::new(),
            retry_after: None,
            retryable: !oversized,
        }
    }

    fn is_retryable(&self) -> bool {
        self.retryable
    }

    fn is_matryoshka_rejection(&self) -> bool {
        self.status == Some(StatusCode::BAD_REQUEST)
            && self
                .response_body
                .to_ascii_lowercase()
                .contains("matryoshka")
    }

    fn into_public(self) -> Error {
        self.error
    }
}

async fn provider_response(
    response: reqwest::Response,
) -> std::result::Result<Vec<u8>, ProviderError> {
    let status = response.status();
    if status.is_success() {
        return crate::models::http_body::read(response, crate::models::http_body::EMBEDDING_BYTES)
            .await
            .map_err(ProviderError::body);
    }

    let retry_after = parse_retry_after(response.headers().get(RETRY_AFTER));
    let response_body =
        match crate::models::http_body::read(response, crate::models::http_body::ERROR_BYTES).await
        {
            Ok(body) => String::from_utf8_lossy(&body).into_owned(),
            Err(crate::models::http_body::Error::Transport) => String::new(),
            Err(error) => return Err(ProviderError::body(error)),
        };
    let error = if status == StatusCode::UNAUTHORIZED {
        Error::Unauthorized
    } else {
        Error::Http {
            status: status.as_u16(),
        }
    };
    Err(ProviderError {
        error,
        status: Some(status),
        response_body,
        retry_after,
        retryable: status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error(),
    })
}

fn decode_embeddings(body: &[u8], expected: usize, dims: usize) -> Result<Vec<Vec<f32>>> {
    let response =
        serde_json::from_slice::<EmbeddingResponse>(body).map_err(|error| Error::Response {
            message: format!("JSON decoding failed: {error}"),
        })?;
    if response.data.len() != expected {
        return Err(Error::Response {
            message: format!(
                "expected {expected} embeddings but provider returned {}",
                response.data.len()
            ),
        });
    }

    let mut ordered = vec![None; expected];
    for datum in response.data {
        if datum.index >= expected || ordered[datum.index].is_some() {
            return Err(Error::Response {
                message: "embedding response has invalid indices".to_owned(),
            });
        }
        ordered[datum.index] = Some(normalize_embedding(
            decode_base64_vector(&datum.embedding)?,
            dims,
        )?);
    }
    ordered
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| Error::Response {
            message: "embedding response omitted an index".to_owned(),
        })
}

fn decode_base64_vector(encoded: &str) -> Result<Vec<f32>> {
    let bytes = STANDARD.decode(encoded).map_err(|error| Error::Response {
        message: format!("base64 decoding failed: {error}"),
    })?;
    let (chunks, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(Error::Response {
            message: "base64 embedding byte length is not a multiple of four".to_owned(),
        });
    }
    Ok(chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

fn embeddings_endpoint(base: &str) -> Result<Url> {
    Url::parse(&format!("{}/embeddings", base.trim_end_matches('/')))
        .map_err(|error| Error::Endpoint(error.to_string()))
}

fn unsupported_models() -> &'static Mutex<HashSet<String>> {
    MRL_DIMENSIONS_UNSUPPORTED.get_or_init(|| Mutex::new(HashSet::new()))
}

#[derive(Deserialize)]
struct CredentialsFile {
    #[serde(default)]
    credentials: Vec<Credential>,
}

#[derive(Deserialize)]
struct Credential {
    host: String,
    api_key: String,
}

/// Resolves a provider key from its configured environment variable then credentials file.
///
/// The path parameter is injectable so tests do not need to access a developer's home
/// directory; production passes `config::credentials_path()`.
pub(crate) fn resolve_api_key(
    api_key_env: &str,
    endpoint: &Url,
    credentials_path: Option<&Path>,
) -> Result<String> {
    if let Ok(api_key) = std::env::var(api_key_env)
        && !api_key.is_empty()
    {
        return Ok(api_key);
    }

    let host = endpoint.host_str().unwrap_or("<missing host>").to_owned();
    let credentials = credentials_path
        .map(Path::to_path_buf)
        .unwrap_or_else(|| Path::new("~/.config/deepkit-code/credentials.toml").to_path_buf());
    if !credentials.exists() {
        return Err(Error::MissingApiKey {
            env: api_key_env.to_owned(),
            host,
            credentials: credentials.display().to_string(),
        });
    }
    let source = read_credentials(&credentials)?;
    let credentials_file =
        toml::from_str::<CredentialsFile>(&source).map_err(|_error| Error::Credentials {
            path: credentials.display().to_string(),
            message: "credentials file is not valid TOML".to_owned(),
        })?;
    credentials_file
        .credentials
        .into_iter()
        .find(|credential| credential.host == host && !credential.api_key.is_empty())
        .map(|credential| credential.api_key)
        .ok_or_else(|| Error::MissingApiKey {
            env: api_key_env.to_owned(),
            host,
            credentials: credentials.display().to_string(),
        })
}

fn read_credentials(path: &Path) -> Result<String> {
    use std::io::Read;

    let invalid = |message: &str| Error::Credentials {
        path: path.display().to_string(),
        message: message.to_owned(),
    };
    // NONBLOCK prevents a FIFO from hanging before its type can be checked.
    // Validate the opened descriptor, not a pathname that could be replaced.
    #[cfg(unix)]
    let file = {
        use rustix::fs::{Mode, OFlags, open};
        std::fs::File::from(
            open(
                path,
                OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .map_err(|_| invalid("could not securely open credentials file"))?,
        )
    };
    #[cfg(not(unix))]
    let file = std::fs::File::open(path).map_err(|_| invalid("could not open credentials file"))?;
    let metadata = file
        .metadata()
        .map_err(|_| invalid("could not inspect credentials file"))?;
    let limit = crate::config::credentials::MAX_FILE_BYTES;
    if !metadata.is_file() || metadata.len() > limit {
        return Err(invalid(
            "credentials must be a regular file no larger than 1 MiB",
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err(Error::InsecureCredentials {
                path: path.display().to_string(),
            });
        }
    }
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| invalid("could not read credentials file"))?;
    if bytes.len() as u64 > limit {
        return Err(invalid("credentials file exceeds 1 MiB"));
    }
    String::from_utf8(bytes).map_err(|_| invalid("credentials file is not valid UTF-8"))
}

#[cfg(test)]
mod tests {
    use super::{AttemptError, USER_AGENT};
    use std::{
        io::{self, Write},
        sync::{Arc, Mutex},
        time::Duration,
    };

    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{header, method, path},
    };

    use super::{EmbedInput, EmbedderTrait, Error, OpenAiConfig, OpenAiEmbedder, resolve_api_key};
    use crate::store::UnitKind;

    const SUCCESS: &str = include_str!("../../fixtures/openai_compat/embeddings-success.json");
    const MATRYOSHKA: &str = include_str!("../../fixtures/openai_compat/matryoshka-rejection.json");
    const MALFORMED: &str = include_str!("../../fixtures/openai_compat/malformed-base64.json");
    const UNAUTHORIZED: &str = include_str!("../../fixtures/openai_compat/unauthorized.json");
    const RATE_LIMITED: &str = include_str!("../../fixtures/openai_compat/rate-limited.json");

    fn provider(server: &MockServer, model: &str) -> OpenAiEmbedder {
        OpenAiEmbedder::with_api_key(
            OpenAiConfig {
                url: format!("{}/v1", server.uri()),
                model: model.to_owned(),
                api_key_env: "DKC_TEST_OPENAI_KEY".to_owned(),
                encoding_format: "base64".to_owned(),
                query_prefix: "Instruct: test\nQuery: ".to_owned(),
                max_batch_items: 2,
                max_batch_tokens: 100,
                dims: 4,
                timeout: Duration::from_secs(2),
            },
            "test-secret-never-log".to_owned(),
        )
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
            body.as_object_mut().unwrap().remove("usage");
            if let Some(count) = count {
                body["usage"] = json!({"total_tokens": count});
            }
            Mock::given(method("POST"))
                .and(path("/v1/embeddings"))
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
    async fn decodes_base64_and_sends_context_without_query_prefix()
    -> Result<(), Box<dyn std::error::Error>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .and(header(
                "user-agent",
                concat!("dkc/", env!("CARGO_PKG_VERSION")),
            ))
            .and(header("authorization", "Bearer test-secret-never-log"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(SUCCESS, "application/json"))
            .expect(1)
            .mount(&server)
            .await;
        let provider = provider(&server, "success-model");
        let input = EmbedInput::from_context(
            "repo › src › lib.rs\nrust fn example\nrust function",
            "fn example() {}",
            UnitKind::Symbol,
        );

        let embedding = provider.embed_documents(&[input]).await?;

        assert_eq!(embedding.len(), 1);
        assert_eq!(embedding[0].len(), 4);
        assert!((embedding[0][0] - 0.6).abs() < 1e-6);
        assert!((embedding[0][1] - 0.8).abs() < 1e-6);
        let requests = server
            .received_requests()
            .await
            .ok_or("request capture unavailable")?;
        let request: serde_json::Value = serde_json::from_slice(&requests[0].body)?;
        assert_eq!(request["dimensions"], 4);
        assert_eq!(request["encoding_format"], "base64");
        assert_eq!(
            request["input"][0],
            "repo › src › lib.rs\nrust fn example\nrust function\nfn example() {}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn remembers_matryoshka_rejection_and_truncates_locally()
    -> Result<(), Box<dyn std::error::Error>> {
        let server = MockServer::start().await;
        let calls = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));
        let observed = Arc::clone(&calls);
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&attempts);
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(move |request: &wiremock::Request| {
                let value = serde_json::from_slice::<serde_json::Value>(&request.body)
                    .unwrap_or(serde_json::Value::Null);
                if let Ok(mut captured) = observed.lock() {
                    captured.push(value);
                }
                if counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                    ResponseTemplate::new(400).set_body_raw(MATRYOSHKA, "application/json")
                } else {
                    ResponseTemplate::new(200).set_body_raw(
                        "{\"data\":[{\"index\":0,\"embedding\":\"AACAPwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=\"}]}",
                        "application/json",
                    )
                }
            })
            .mount(&server)
            .await;
        let provider = provider(&server, "matryoshka-model");

        let first = provider.embed_query("first").await?;
        let second = provider.embed_query("second").await?;

        assert_eq!(first, vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(second, vec![1.0, 0.0, 0.0, 0.0]);
        let calls = calls.lock().map_err(|_| "test request capture poisoned")?;
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0]["dimensions"], 4);
        assert_eq!(calls[0]["input"][0], "Instruct: test\nQuery: first");
        assert!(calls[1].get("dimensions").is_none());
        assert!(calls[2].get("dimensions").is_none());
        Ok(())
    }

    #[tokio::test]
    async fn retries_rate_limits_using_retry_after() -> Result<(), Box<dyn std::error::Error>> {
        let server = MockServer::start().await;
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&attempts);
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(move |_request: &wiremock::Request| {
                if counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                    ResponseTemplate::new(429)
                        .insert_header("Retry-After", "0")
                        .set_body_raw(RATE_LIMITED, "application/json")
                } else {
                    ResponseTemplate::new(200).set_body_raw(SUCCESS, "application/json")
                }
            })
            .mount(&server)
            .await;

        let embedding = provider(&server, "rate-limit-model")
            .embed_query("retry")
            .await?;

        assert_eq!(embedding.len(), 4);
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 2);
        Ok(())
    }

    #[tokio::test]
    async fn maps_unauthorized_and_malformed_base64_to_typed_errors()
    -> Result<(), Box<dyn std::error::Error>> {
        let unauthorized = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(401).set_body_raw(UNAUTHORIZED, "application/json"))
            .mount(&unauthorized)
            .await;
        assert!(matches!(
            provider(&unauthorized, "unauthorized-model")
                .embed_query("nope")
                .await,
            Err(Error::Unauthorized)
        ));

        let malformed = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(MALFORMED, "application/json"))
            .mount(&malformed)
            .await;
        assert!(matches!(
            provider(&malformed, "malformed-model")
                .embed_query("bad")
                .await,
            Err(Error::Response { .. })
        ));
        Ok(())
    }

    #[test]
    fn credentials_file_is_the_fallback_after_an_absent_environment_variable()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let credentials = directory.path().join("credentials.toml");
        std::fs::write(
            &credentials,
            "[[credentials]]\nhost = \"gateway.example\"\napi_key = \"file-secret\"\n",
        )?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&credentials, std::fs::Permissions::from_mode(0o600))?;
        }
        let endpoint = reqwest::Url::parse("https://gateway.example/v1/embeddings")?;

        let key = resolve_api_key("DKC_UNSET_TEST_KEY", &endpoint, Some(&credentials))?;

        assert_eq!(key, "file-secret");
        Ok(())
    }

    #[test]
    fn credential_reads_reject_oversized_files_before_parsing()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let credentials = directory.path().join("credentials.toml");
        let file = std::fs::File::create(&credentials)?;
        file.set_len(crate::config::credentials::MAX_FILE_BYTES + 1)?;
        let endpoint = reqwest::Url::parse("https://gateway.example/v1/embeddings")?;
        assert!(matches!(
            resolve_api_key("DKC_UNSET_TEST_KEY", &endpoint, Some(&credentials)),
            Err(Error::Credentials { .. })
        ));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn credential_reads_reject_fifos_symlinks_and_public_files()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::{PermissionsExt, symlink};

        let directory = tempfile::tempdir()?;
        let fifo = directory.path().join("fifo");
        rustix::fs::mkfifoat(
            rustix::fs::CWD,
            &fifo,
            rustix::fs::Mode::from_raw_mode(0o600),
        )?;
        assert!(matches!(
            super::read_credentials(&fifo),
            Err(Error::Credentials { .. })
        ));
        let file = directory.path().join("credentials.toml");
        std::fs::write(&file, "private-marker")?;
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o600))?;
        let alias = directory.path().join("alias");
        symlink(&file, &alias)?;
        let error = super::read_credentials(&alias).unwrap_err();
        assert!(!error.to_string().contains("private-marker"));
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o644))?;
        assert!(matches!(
            super::read_credentials(&file),
            Err(Error::InsecureCredentials { .. })
        ));
        assert!(matches!(
            super::read_credentials(directory.path()),
            Err(Error::Credentials { .. })
        ));
        Ok(())
    }

    #[tokio::test]
    async fn authorization_never_reaches_tracing_output() -> Result<(), Box<dyn std::error::Error>>
    {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(SUCCESS, "application/json"))
            .mount(&server)
            .await;
        let captured = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(CapturingWriter(Arc::clone(&captured)))
            .finish();
        let guard = tracing::subscriber::set_default(subscriber);

        let embedding = provider(&server, "tracing-model")
            .embed_query("safe")
            .await?;

        drop(guard);
        assert_eq!(embedding.len(), 4);
        let output = captured.lock().map_err(|_| "tracing capture poisoned")?;
        let output = String::from_utf8_lossy(&output);
        assert!(!output.contains("test-secret-never-log"));
        assert!(!output.contains("Authorization"));
        Ok(())
    }

    #[tokio::test]
    async fn batches_bound_items_and_estimated_tokens() -> Result<(), Box<dyn std::error::Error>> {
        let server = MockServer::start().await;
        let provider = provider(&server, "batch-model");
        let first = EmbedInput::from_context("one", "one", UnitKind::Chunk);
        let second = EmbedInput::from_context("two", "two", UnitKind::Chunk);
        let third = EmbedInput::from_context("three", "three", UnitKind::Chunk);

        let batches = provider.batches(&[first, second, third])?;

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn rejects_a_document_that_exceeds_the_token_batch_cap()
    -> Result<(), Box<dyn std::error::Error>> {
        let server = MockServer::start().await;
        let provider = provider(&server, "token-cap-model");
        let text = "token ".repeat(200);
        let input = EmbedInput::from_context("header", &text, UnitKind::Chunk);

        assert!(matches!(
            provider.batches(&[input]),
            Err(Error::BatchTooLarge {
                max_tokens: 100,
                ..
            })
        ));
        Ok(())
    }

    #[tokio::test]
    async fn live_moonly_embeddings_have_mrl_dimensions_and_a_single_64_item_batch()
    -> Result<(), Box<dyn std::error::Error>> {
        if std::env::var("DKC_LIVE_GATEWAY").as_deref() != Ok("1") {
            return Ok(());
        }

        for model in ["qwen3-embedding-8b", "qwen3-embedding-0.6b"] {
            let remote = crate::config::RemoteEmbedder {
                model: model.to_owned(),
                ..crate::config::RemoteEmbedder::default()
            };
            let config = OpenAiConfig::from_remote(&remote, 1_024, Duration::from_secs(30))?;
            let provider = OpenAiEmbedder::from_config(config)?;
            let texts = (0..64)
                .map(|index| format!("fn gateway_batch_{index}() -> usize {{ {index} }}"))
                .collect::<Vec<_>>();
            let inputs = texts
                .iter()
                .map(|text| {
                    EmbedInput::from_context(
                        "live › batch.rs\nfn gateway_batch\nrust function",
                        text,
                        UnitKind::Chunk,
                    )
                })
                .collect::<Vec<_>>();

            let embeddings = provider.embed_documents(&inputs).await?;

            assert_eq!(embeddings.len(), 64);
            assert!(embeddings.iter().all(|embedding| embedding.len() == 1_024));
            assert!(embeddings.iter().all(|embedding| {
                (embedding
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    .sqrt()
                    - 1.0)
                    .abs()
                    < 1e-4
            }));
        }
        Ok(())
    }

    #[derive(Clone)]
    struct CapturingWriter(Arc<Mutex<Vec<u8>>>);

    impl<'writer> tracing_subscriber::fmt::MakeWriter<'writer> for CapturingWriter {
        type Writer = Self;

        fn make_writer(&'writer self) -> Self::Writer {
            self.clone()
        }
    }

    impl Write for CapturingWriter {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            let mut captured = self
                .0
                .lock()
                .map_err(|_| io::Error::other("tracing capture poisoned"))?;
            captured.extend_from_slice(buffer);
            Ok(buffer.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    #[tokio::test]
    async fn queue_attempt_errors_are_single_request_and_payload_free() {
        for status in [401, 403, 429, 500, 503, 200] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(
                    ResponseTemplate::new(status)
                        .insert_header("Retry-After", "120")
                        .set_body_string("private-response-marker"),
                )
                .expect(1)
                .mount(&server)
                .await;
            let p = provider(&server, &format!("queue-status-{status}"));
            let doc = EmbedInput::from_context("context", "body", UnitKind::Symbol);
            let error = p.embed_documents_once(&[doc]).await.unwrap_err();
            assert!(!format!("{error:?} {error}").contains("private-response-marker"));
            assert!(!format!("{error:?}").contains("test-secret-never-log"));
            match status {
                429 | 500 | 503 => assert!(
                    matches!(error,AttemptError::Retryable{retry_after:Some(d),..} if d==Duration::from_secs(120))
                ),
                200 => assert!(matches!(error, AttemptError::InvalidResponse)),
                _ => assert!(matches!(error,AttemptError::Rejected{status:s} if s==status)),
            }
            server.verify().await;
        }
    }

    #[tokio::test]
    async fn queue_dimensions_fallback_requires_another_explicit_attempt() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(|r: &wiremock::Request| {
                let body: serde_json::Value = serde_json::from_slice(&r.body).unwrap();
                if body.get("dimensions").is_some() {
                    ResponseTemplate::new(400).set_body_raw(MATRYOSHKA, "application/json")
                } else {
                    ResponseTemplate::new(200).set_body_raw(SUCCESS, "application/json")
                }
            })
            .expect(2)
            .mount(&server)
            .await;
        let p = provider(&server, "queue-mrl-model");
        let doc = || EmbedInput::from_context("context", "body", UnitKind::Symbol);
        assert!(matches!(
            p.embed_documents_once(&[doc()]).await,
            Err(AttemptError::DimensionsRejected)
        ));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
        let result = p.embed_documents_once(&[doc()]).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 4);
        assert!((result[0][0] - 0.6).abs() < 1e-6);
        server.verify().await;
    }

    #[tokio::test]
    async fn queue_attempt_rejects_split_batches_and_does_not_follow_redirects() {
        let server = MockServer::start().await;
        let p = provider(&server, "queue-no-split");
        let doc = || EmbedInput::from_context("context", "body", UnitKind::Symbol);
        assert!(matches!(
            p.embed_documents_once(&[]).await,
            Err(AttemptError::InvalidInput)
        ));
        assert!(matches!(
            p.embed_documents_once(&[doc(), doc(), doc()]).await,
            Err(AttemptError::InvalidInput)
        ));
        let oversized = "long ".repeat(300);
        assert!(matches!(
            p.embed_documents_once(&[EmbedInput::from_context(
                "context",
                &oversized,
                UnitKind::Symbol
            )])
            .await,
            Err(AttemptError::InvalidInput)
        ));
        assert!(server.received_requests().await.unwrap().is_empty());
        Mock::given(path("/v1/embeddings"))
            .respond_with(
                ResponseTemplate::new(307)
                    .insert_header("Location", format!("{}/redirect", server.uri())),
            )
            .expect(1)
            .mount(&server)
            .await;
        assert!(matches!(
            p.embed_documents_once(&[doc()]).await,
            Err(AttemptError::Rejected { status: 307 })
        ));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn queue_retry_loop_makes_at_most_eight_actual_http_requests() {
        use crate::{
            queue::{self, retry, usage},
            store::{self, StoreLimits},
        };
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(header("user-agent", USER_AGENT))
            .respond_with(ResponseTemplate::new(429).insert_header("Retry-After", "0"))
            .expect(8)
            .mount(&server)
            .await;
        let p = provider(&server, "queue-eight-attempts");
        let store = store::open(":memory:", StoreLimits::default()).unwrap();
        queue::enqueue(
            &store,
            &queue::Enqueue {
                kind: queue::Kind::Embed,
                target: queue::Target::Unit,
                target_id: 1,
                target_hash: &[1; 32],
                desired_hash: &[2; 32],
                provider: "embed:remote",
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
        for attempt in 1..=8 {
            let lease = queue::claim(&store, "embed:remote", &owner, now, 1, 10)
                .unwrap()
                .remove(0);
            let usage::Admission::Reserved(reservation) =
                usage::reserve(&store, "embed:remote", 10, 0, budget, Some(&lease), || now)
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
            if attempt < 8 {
                let retry::Scheduled::RetryAt(next) = outcome else {
                    panic!("retry missing")
                };
                now = next;
            } else {
                assert_eq!(outcome, retry::Scheduled::Dead);
            }
        }
        assert!(
            queue::claim(&store, "embed:remote", &owner, now + 1000, 1, 10)
                .unwrap()
                .is_empty()
        );
        assert_eq!(server.received_requests().await.unwrap().len(), 8);
        server.verify().await;
    }
    #[tokio::test]
    async fn prepared_query_attempts_keep_instructions_usage_and_separate_dimension_retry() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(header("user-agent", USER_AGENT))
            .respond_with(|request: &wiremock::Request| {
                let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
                assert_eq!(
                    body["input"],
                    serde_json::json!(["Instruct: test\nQuery: find retries"])
                );
                if body.get("dimensions").is_some() {
                    ResponseTemplate::new(400).set_body_raw(MATRYOSHKA, "application/json")
                } else {
                    let mut response: serde_json::Value = serde_json::from_str(SUCCESS).unwrap();
                    response["usage"] = serde_json::json!({"total_tokens":42});
                    ResponseTemplate::new(200).set_body_json(response)
                }
            })
            .expect(2)
            .mount(&server)
            .await;
        let p = provider(&server, "prepared-query-mrl");
        let query = p.prepare_query("find retries").unwrap();
        assert!(matches!(
            p.embed_query_with_usage_once(&query).await,
            Err(AttemptError::DimensionsRejected)
        ));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
        let response = p.embed_query_with_usage_once(&query).await.unwrap();
        assert_eq!(response.tokens, Some(42));
        assert_eq!(response.embeddings.len(), 1);
        assert!((response.embeddings[0][0] - 0.6).abs() < 1e-6);
        let requests = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
        assert_eq!(
            query.tokens() as usize,
            tiktoken_rs::cl100k_base_singleton()
                .encode_with_special_tokens(body["input"][0].as_str().unwrap())
                .len()
        );
        assert!(
            provider(&server, "other-prepared-query-model")
                .embed_query_with_usage_once(&query)
                .await
                .is_err()
        );
        assert!(p.prepare_query("").is_err());
        assert!(p.prepare_query(&"word ".repeat(200)).is_err());
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
    }
    #[tokio::test]
    async fn prepared_query_transient_failure_never_retries_inside_the_provider() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(503).insert_header("Retry-After", "12"))
            .expect(1)
            .mount(&server)
            .await;
        let p = provider(&server, "prepared-query-unavailable");
        let query = p.prepare_query("private query").unwrap();
        assert!(
            matches!(p.embed_query_with_usage_once(&query).await,Err(AttemptError::Retryable {status:Some(503),retry_after:Some(delay)}) if delay==Duration::from_secs(12))
        );
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }
    #[tokio::test]
    async fn oversized_success_and_dimension_errors_are_permanent_single_attempt_failures() {
        use crate::models::http_body::{self, tests::serve_once};
        for (status, chunked) in [(200, false), (400, false), (400, true)] {
            let cap = if status == 200 {
                http_body::EMBEDDING_BYTES
            } else {
                http_body::ERROR_BYTES
            };
            let (headers, body) = if chunked {
                let private = format!("matryoshka private-placeholder{}", "x".repeat(cap));
                (
                    "Transfer-Encoding: chunked\r\n".to_owned(),
                    format!("{:x}\r\n{private}\r\n0\r\n\r\n", private.len()).into_bytes(),
                )
            } else {
                (format!("Content-Length: {}\r\n", cap + 1), Vec::new())
            };
            let (url, task) = serve_once(status, headers, body).await;
            let remote = crate::config::RemoteEmbedder {
                url,
                model: "doctor-oversized-body-fixture".into(),
                ..Default::default()
            };
            let provider = OpenAiEmbedder::with_api_key(
                OpenAiConfig::from_remote(&remote, 4, Duration::from_secs(2)).unwrap(),
                "fixture".into(),
            )
            .unwrap();
            let query = provider.prepare_query("fixed diagnostic query").unwrap();
            let error = provider
                .embed_query_with_usage_once(&query)
                .await
                .err()
                .unwrap();
            assert!(matches!(error, AttemptError::InvalidResponse));
            assert!(!format!("{error:?}").contains("private-placeholder"));
            // Oversized error text must not poison the process capability cache.
            assert!(provider.dimensions_supported());
            task.await.unwrap();
        }
    }

    #[tokio::test]
    async fn bounded_dimension_error_still_admits_the_existing_fallback_signal() {
        use crate::models::http_body::tests::serve_once;
        let body = b"matryoshka unsupported".to_vec();
        let (url, task) =
            serve_once(400, format!("Content-Length: {}\r\n", body.len()), body).await;
        let remote = crate::config::RemoteEmbedder {
            url,
            model: "doctor-bounded-dimension-fixture".into(),
            ..Default::default()
        };
        let provider = OpenAiEmbedder::with_api_key(
            OpenAiConfig::from_remote(&remote, 4, Duration::from_secs(2)).unwrap(),
            "fixture".into(),
        )
        .unwrap();
        let query = provider.prepare_query("fixed diagnostic query").unwrap();
        assert!(matches!(
            provider.embed_query_with_usage_once(&query).await,
            Err(AttemptError::DimensionsRejected)
        ));
        assert!(!provider.dimensions_supported());
        task.await.unwrap();
    }
}
