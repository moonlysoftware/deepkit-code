//! Model providers and profiles; see `v1.md` §7.

#![expect(
    dead_code,
    reason = "T1.6 and later queue/search tasks consume the provider traits introduced by T1.2"
)]

use async_trait::async_trait;

use crate::store::UnitKind;

/// A model-provider result with errors that are safe to show to users and logs.
pub(crate) type Result<T> = std::result::Result<T, Error>;

/// Errors returned by model providers.
///
/// Secret material is intentionally absent from every variant, so diagnostic
/// formatting cannot expose bearer tokens.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    /// Shared chat/VLM transport or response failure without private payloads.
    #[error(transparent)]
    Chat(#[from] openai_compat::chat::Error),
    /// Reranker transport, input or score validation failed without private payloads.
    #[error(transparent)]
    Rerank(#[from] rerank::Error),
    /// Native local model failure, retaining its typed cause.
    #[error(transparent)]
    Local(#[from] llama::Error),
    /// A native operation could not complete on its blocking worker.
    #[error("local model worker did not complete")]
    LocalWorker,
    /// The configured endpoint was not a valid HTTP URL.
    #[error("invalid model endpoint: {0}")]
    Endpoint(String),
    /// The configured remote encoding is not supported by the OpenAI-compatible client.
    #[error("unsupported embedding encoding format: {0}")]
    UnsupportedEncoding(String),
    /// The configured API key was unavailable from both permitted sources.
    #[error("API key is unavailable; set {env} or add credentials for {host} to {credentials}")]
    MissingApiKey {
        /// Environment variable configured for the provider.
        env: String,
        /// Endpoint host used to select a credentials-file entry.
        host: String,
        /// Credentials file consulted after the environment variable.
        credentials: String,
    },
    /// The credentials file could not be read or parsed safely.
    #[error("could not read credentials from {path}: {message}")]
    Credentials {
        /// Credentials file path.
        path: String,
        /// Safe description of the read or parse failure.
        message: String,
    },
    /// The credentials file is accessible to users other than its owner.
    #[error("credentials file {path} must have mode 0600")]
    InsecureCredentials {
        /// Credentials file path.
        path: String,
    },
    /// The remote server rejected the bearer token.
    #[error("remote model provider rejected the API key")]
    Unauthorized,
    /// The remote provider returned a non-retryable HTTP failure.
    #[error("remote model provider returned HTTP {status}")]
    Http {
        /// HTTP response status.
        status: u16,
    },
    /// The HTTP transport or response body failed before a valid payload arrived.
    #[error("remote model request failed: {message}")]
    Request {
        /// Safe transport failure text; request headers and body are never retained.
        message: String,
    },
    /// The provider response was not a well-formed embeddings payload.
    #[error("invalid embedding response: {message}")]
    Response {
        /// Safe response-shape or encoding diagnosis.
        message: String,
    },
    /// A document cannot be admitted to the configured remote embedding batch.
    #[error("embedding input has {tokens} tokens; remote batch cap is {max_tokens}")]
    BatchTooLarge {
        /// Estimated token count for the input.
        tokens: usize,
        /// Configured maximum token count per remote request.
        max_tokens: usize,
    },
    /// A provider vector could not be normalized for cosine retrieval.
    #[error("remote model returned a non-finite or zero-length embedding")]
    InvalidEmbedding,
}

/// The contextual document handed to an embedder.
///
/// `treat::context` alone constructs these values, keeping the breadcrumb,
/// structural header, and kind line identical for every provider.
pub(crate) struct EmbedInput<'a> {
    /// Rendered three-line context prefix.
    pub(crate) header: &'a str,
    /// Unit content after the context prefix.
    pub(crate) text: &'a str,
    /// Unit category for provider routing and future tier selection.
    pub(crate) kind: UnitKind,
}

impl<'a> EmbedInput<'a> {
    /// Builds an embedding input from the canonical treatment context.
    pub(crate) const fn from_context(header: &'a str, text: &'a str, kind: UnitKind) -> Self {
        Self { header, text, kind }
    }
}

/// One completed chat response.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Completion {
    /// The assistant text returned by the configured chat model.
    pub(crate) content: String,
}

/// Truncates a provider's MRL output and returns a finite unit-length vector.
///
/// Every embedding profile shares this final boundary before FLOAT16 storage,
/// so provider-specific wire formats cannot drift in their cosine semantics.
pub(crate) fn normalize_embedding(mut embedding: Vec<f32>, dims: usize) -> Result<Vec<f32>> {
    if embedding.len() < dims {
        return Err(Error::Response {
            message: format!(
                "provider returned {} dimensions but the configured table needs {dims}",
                embedding.len()
            ),
        });
    }
    embedding.truncate(dims);
    let squared_norm = embedding.iter().map(|value| value * value).sum::<f32>();
    if !squared_norm.is_finite() || squared_norm <= f32::EPSILON {
        return Err(Error::InvalidEmbedding);
    }
    let norm = squared_norm.sqrt();
    for value in &mut embedding {
        *value /= norm;
    }
    if embedding.iter().all(|value| value.is_finite()) {
        Ok(embedding)
    } else {
        Err(Error::InvalidEmbedding)
    }
}

/// Produces document and query vectors for a single embedding profile.
#[async_trait]
pub(crate) trait Embedder: Send + Sync {
    /// Returns the stable provider/model identifier used for vector table selection.
    fn id(&self) -> &str;
    /// Returns the dimensions stored for this profile after MRL truncation.
    fn dims(&self) -> usize;
    /// Returns the maximum estimated tokens sent in one document request.
    fn max_batch_tokens(&self) -> usize;
    /// Returns the maximum documents sent in one document request.
    fn max_batch_items(&self) -> usize;
    /// Embeds canonical document contexts without applying a query instruction.
    async fn embed_documents(&self, docs: &[EmbedInput<'_>]) -> Result<Vec<Vec<f32>>>;
    /// Embeds a search query with the configured model instruction prefix.
    async fn embed_query(&self, query: &str) -> Result<Vec<f32>>;
}

/// Scores candidate documents for a query after lexical and dense retrieval.
#[async_trait]
pub(crate) trait Reranker: Send + Sync {
    /// Returns the stable provider/model identifier.
    fn id(&self) -> &str;
    /// Returns the maximum source characters accepted per document.
    fn max_doc_chars(&self) -> usize;
    /// Scores documents from zero to one, where higher is more relevant.
    async fn score(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>>;
}

/// Completes a bounded enrichment prompt.
#[async_trait]
pub(crate) trait Llm: Send + Sync {
    /// Completes one prompt with an explicit output-token ceiling.
    async fn complete(&self, system: &str, user: &str, max_tokens: u32) -> Result<Completion>;
}

/// Describes an image for the two allowed image enrichment jobs.
#[async_trait]
pub(crate) trait Vlm: Send + Sync {
    /// Returns a textual description for image bytes and an explicit instruction.
    async fn describe(&self, image: &[u8], mime: &str, instruction: &str) -> Result<String>;
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "T1.6 wires the remote embedding provider into the queue"
    )
)]
pub(crate) mod attempt;
mod http_body;
pub(crate) mod openai_compat;

/// Local and remote candidate scorers; v1 §7.2.
pub(crate) mod rerank;

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "T1.6 wires the Ollama embedding provider into the queue"
    )
)]
pub(crate) mod ollama;

// T0.3 verifies this implementation through an ignored model-backed test; T1
// wires it into the local embedder provider and queue.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "T1 wires the verified local provider")
)]
pub(crate) mod llama;

/// Pinned artifacts and read-only model-cache verification; v1.md §7.4.
pub(crate) mod registry;

/// Resumable verified model downloads and cache admission; v1.md §7.4.
pub(crate) mod download;

/// First-index local artifact preparation; v1.md §7.4.
pub(crate) mod setup;

/// Bounded shared provider model discovery; v1.md §§7.2, 13.
pub(crate) mod discovery;
