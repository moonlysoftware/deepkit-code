//! Local llama.cpp embeddings for the `local` profile; see `v1.md` §7.2.

mod cancellation;

static RESIDENT_INSTANCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_resident_instance() -> u64 {
    RESIDENT_INSTANCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn residency_event(event: &'static str, role: &'static str, instance: u64, count: usize) {
    tracing::debug!(target: "dkc::models::residency", event, role, instance, count);
}

/// Verified local Rank reranking; v1 §7.2/§7.5.
pub(crate) mod rerank;
use super::attempt::{AttemptError, EmbeddingBatch, PreparedQuery, query_contract};
use cancellation::Cancellation;

use std::{
    num::NonZeroU32,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock, Weak},
    time::{Duration, Instant},
};

use llama_cpp_2::{
    context::params::{LlamaContextParams, LlamaPoolingType},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel, params::LlamaModelParams},
};

/// The local profile's pinned Qwen embedding GGUF filename.
pub(crate) const EMBEDDING_MODEL_FILE: &str = "Qwen3-Embedding-0.6B-Q8_0.gguf";
const MAX_BATCH_ITEMS: usize = 32;
/// Validated native limits, resolved before initializing llama.cpp.
#[derive(Clone, Debug)]
struct Settings {
    context_tokens: u32,
    batch_tokens: u32,
    threads: i32,
    gpu_layers: i32,
}

impl Settings {
    fn from_profile(profile: &crate::config::LocalEmbedder) -> Result<Self, Error> {
        let invalid = |message| Error::Configuration { message };
        let positive = |value, message| {
            u32::try_from(value)
                .ok()
                .filter(|value| *value > 0 && *value <= i32::MAX as u32)
                .ok_or_else(|| invalid(message))
        };
        let context_tokens = positive(profile.n_ctx, "local n_ctx must be in 1..=2147483647")?;
        let batch_tokens = positive(
            profile.batch_tokens,
            "local batch_tokens must be in 1..=2147483647",
        )?;
        let threads = if profile.threads == 0 {
            i32::try_from(num_cpus::get_physical().max(1))
                .map_err(|_| invalid("physical core count exceeds native limit"))?
        } else {
            positive(profile.threads, "local threads must be in 0..=2147483647")? as i32
        };
        let gpu_layers = i32::try_from(profile.gpu_layers)
            .ok()
            .filter(|value| *value >= -1)
            .ok_or_else(|| invalid("local gpu_layers must be in -1..=2147483647"))?;
        Ok(Self {
            context_tokens,
            batch_tokens,
            threads,
            gpu_layers,
        })
    }

    fn token_limit(&self) -> usize {
        self.context_tokens.min(self.batch_tokens) as usize
    }

    fn model_params(&self) -> LlamaModelParams {
        let params = LlamaModelParams::default();
        let accelerated = cfg!(any(
            feature = "cuda",
            feature = "metal",
            feature = "vulkan",
            all(target_os = "macos", target_arch = "aarch64")
        ));
        if !accelerated {
            params.with_n_gpu_layers(0)
        } else if self.gpu_layers == -1 {
            params
        } else {
            params.with_n_gpu_layers(self.gpu_layers as u32)
        }
    }
}

/// Errors returned by the local llama.cpp embedding implementation.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    /// A setting cannot be represented safely by the native embedding API.
    #[error("invalid local embedding configuration: {message}")]
    Configuration {
        /// Static diagnosis without model input content.
        message: &'static str,
    },
    /// The requested text does not fit within the configured local token budget.
    #[error("embedding input has {tokens} tokens; the local profile allows at most {max_tokens}")]
    InputTooLong {
        /// Number of model tokens in the input.
        tokens: usize,
        /// Effective minimum of context and batch capacity.
        max_tokens: usize,
    },
    /// llama.cpp rejected a model, context, tokenization, batch, or embedding operation.
    #[error("llama.cpp {operation} failed: {message}")]
    Llama {
        /// Operation being attempted when llama.cpp returned an error.
        operation: &'static str,
        /// Display form of the upstream error, retained for diagnosis.
        message: String,
    },
    /// The model emitted a vector that cannot be normalized for cosine search.
    #[error("llama.cpp returned a non-finite or zero-length embedding")]
    InvalidEmbedding,
    /// Cancellation was observed before starting further local work.
    #[error("local embedding was cancelled")]
    Cancelled,
    /// The cached model no longer matches its pinned registry artifact.
    #[error("local model artifact verification failed")]
    Artifact,
}

// llama.cpp has one process-wide backend. Keep its small initialization state alive
// while independently loading/unloading model weights and per-batch contexts.
static BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();

fn shared_backend() -> Result<&'static LlamaBackend, Error> {
    BACKEND
        .get_or_init(|| {
            let mut backend = LlamaBackend::init().map_err(|error| error.to_string())?;
            backend.void_logs();
            Ok(backend)
        })
        .as_ref()
        .map_err(|message| Error::Llama {
            operation: "backend initialization",
            message: message.clone(),
        })
}

/// Async local provider using canonical document context and query-only instructions.
/// Native loading and inference run on blocking workers so MCP can remain responsive.
/// Cancelled inference stops at native-operation boundaries; in-flight decode
/// retains the single provider slot until it exits, without accepting further batches.
pub(crate) struct LocalProvider {
    state: Arc<Mutex<ResidentModel>>,
    inference: Arc<tokio::sync::Semaphore>,
    idle: Duration,
    idle_started: OnceLock<()>,
    id: String,
    query_prefix: String,
    dims: usize,
    token_limit: usize,
}

struct ResidentModel {
    instance: u64,
    native: Option<LocalEmbedder>,
    path: PathBuf,
    profile: crate::config::LocalEmbedder,
    last_used: Instant,
    artifact: Option<(super::registry::Model, PathBuf)>,
}

type OwnedContext = (String, String);

impl Drop for ResidentModel {
    fn drop(&mut self) {
        self.unload();
    }
}

impl ResidentModel {
    // Both idle eviction and final owner destruction report actual release once.
    fn unload(&mut self) {
        if let Some(native) = self.native.take() {
            drop(native);
            residency_event("unloaded", "embedder", self.instance, 0);
        }
    }

    fn embed(
        &mut self,
        texts: &[String],
        dims: usize,
        stopped: &Cancellation,
    ) -> super::Result<Vec<Vec<f32>>> {
        stopped.check()?;
        if self.native.is_none() {
            let native = load_native(&self.path, &self.profile, self.artifact.as_ref())?;
            validate_dimensions(&native, dims)?;
            self.native = Some(native);
            residency_event("loaded", "embedder", self.instance, 0);
        }
        stopped.check()?;
        let native = self.native.as_ref().ok_or(super::Error::LocalWorker)?;
        let refs = texts.iter().map(String::as_str).collect::<Vec<_>>();
        let result: super::Result<Vec<Vec<f32>>> = native
            .embed_cancellable(&refs, stopped)?
            .into_iter()
            .map(|vector| super::normalize_embedding(vector, dims))
            .collect();
        self.last_used = Instant::now();
        if result.is_ok() && !texts.is_empty() {
            residency_event("inference_complete", "embedder", self.instance, texts.len());
        }
        result
    }

    fn embed_contexts(
        &mut self,
        contexts: &[OwnedContext],
        dims: usize,
        stopped: &Cancellation,
    ) -> super::Result<Vec<Vec<f32>>> {
        stopped.check()?;
        if self.native.is_none() {
            let native = load_native(&self.path, &self.profile, self.artifact.as_ref())?;
            validate_dimensions(&native, dims)?;
            self.native = Some(native);
            residency_event("loaded", "embedder", self.instance, 0);
        }
        stopped.check()?;
        let native = self.native.as_ref().ok_or(super::Error::LocalWorker)?;
        let refs = contexts
            .iter()
            .map(|(header, text)| (header.as_str(), text.as_str()))
            .collect::<Vec<_>>();
        let result: super::Result<Vec<Vec<f32>>> = native
            .embed_contexts_cancellable(&refs, stopped)?
            .into_iter()
            .map(|vector| super::normalize_embedding(vector, dims))
            .collect();
        self.last_used = Instant::now();
        if result.is_ok() && !contexts.is_empty() {
            residency_event(
                "inference_complete",
                "embedder",
                self.instance,
                contexts.len(),
            );
        }
        result
    }
}

// Hash before opening native weights, then verify that the opened path still has
// the same identity. Platforms without reusable identity rehash after loading.
fn load_native(
    path: &Path,
    profile: &crate::config::LocalEmbedder,
    artifact: Option<&(super::registry::Model, PathBuf)>,
) -> Result<LocalEmbedder, Error> {
    let proof = artifact
        .map(|(model, cache)| {
            super::registry::verify_with_identity(model, cache).map_err(|_| Error::Artifact)
        })
        .transpose()?
        .flatten();
    let mut native = LocalEmbedder::load_with_profile(path, profile)?;
    if let Some((model, cache)) = artifact {
        if let Some(proof) = &proof {
            if !proof.matches(model, cache) {
                return Err(Error::Artifact);
            }
        } else {
            super::registry::verify(model, cache).map_err(|_| Error::Artifact)?;
        }
    }
    native.artifact = artifact.map(|(model, cache)| (model.clone(), cache.clone(), proof));
    Ok(native)
}

fn validate_dimensions(native: &LocalEmbedder, dims: usize) -> Result<(), Error> {
    let width = usize::try_from(native.model.n_embd_out()).unwrap_or(0);
    if dims == 0 || dims > width {
        return Err(Error::Configuration {
            message: "local dimensions exceed model output width or are zero",
        });
    }
    Ok(())
}

// Only a weak reference lives across sleeps, so an idle timer cannot retain a provider.
async fn unload_idle(state: Weak<Mutex<ResidentModel>>, idle: Duration) {
    let mut remaining = idle;
    loop {
        tokio::time::sleep(remaining).await;
        let Some(resident) = state.upgrade() else {
            return;
        };
        let outcome = tokio::task::spawn_blocking(move || {
            let mut resident = resident.lock().ok()?;
            let elapsed = resident.last_used.elapsed();
            if elapsed >= idle {
                // Destruction happens on this worker, while holding the same lock as inference.
                resident.unload();
                Some(idle)
            } else {
                Some(idle - elapsed)
            }
        })
        .await;
        match outcome {
            Ok(Some(delay)) => remaining = delay,
            _ => return,
        }
    }
}

impl LocalProvider {
    /// Loads the configured model with the standard ten-minute idle unload setting.
    pub(crate) async fn load(
        path: PathBuf,
        profile: crate::config::LocalEmbedder,
        dims: u64,
    ) -> super::Result<Self> {
        Self::load_with_idle(path, profile, dims, Duration::from_secs(600)).await
    }

    /// Loads a local model and schedules weight unloading after the configured idle duration.
    /// Calls serialize per provider; a request reloads unloaded weights before inference.
    pub(crate) async fn load_with_idle(
        path: PathBuf,
        profile: crate::config::LocalEmbedder,
        dims: u64,
        idle: Duration,
    ) -> super::Result<Self> {
        Self::load_inner(path, profile, dims, idle, None).await
    }

    /// Loads only verified registry weights and re-verifies them after idle unload.
    /// The host prepares the cache separately; this method never downloads files.
    pub(crate) async fn load_registered(
        cache: PathBuf,
        profile: crate::config::LocalEmbedder,
        dims: u64,
        idle: Duration,
    ) -> super::Result<Self> {
        Self::prepare_registered(cache, profile, dims, idle)?
            .load_prepared()
            .await
    }

    /// Prepares a registered local provider without reading or loading weights.
    /// This needs no runtime, starts no tasks, and never downloads files. The first
    /// admitted nonempty inference verifies the artifact before native loading;
    /// successful preparation alone does not attest to the cached artifact.
    pub(crate) fn prepare_registered(
        cache: PathBuf,
        profile: crate::config::LocalEmbedder,
        dims: u64,
        idle: Duration,
    ) -> super::Result<Self> {
        let registry = super::registry::Registry::embedded().map_err(|_| Error::Artifact)?;
        let model = registry
            .get(&profile.model)
            .map_err(|_| Error::Artifact)?
            .clone();
        if model.pooling != "last" || dims > model.dims as u64 {
            return Err(Error::Artifact.into());
        }
        let path = cache.join(&model.file);
        Self::prepare_inner(path, profile, dims, idle, Some((model, cache)))
    }

    async fn load_inner(
        path: PathBuf,
        profile: crate::config::LocalEmbedder,
        dims: u64,
        idle: Duration,
        artifact: Option<(super::registry::Model, PathBuf)>,
    ) -> super::Result<Self> {
        Self::prepare_inner(path, profile, dims, idle, artifact)?
            .load_prepared()
            .await
    }

    fn prepare_inner(
        path: PathBuf,
        profile: crate::config::LocalEmbedder,
        dims: u64,
        idle: Duration,
        artifact: Option<(super::registry::Model, PathBuf)>,
    ) -> super::Result<Self> {
        let dims =
            usize::try_from(dims)
                .ok()
                .filter(|dims| *dims > 0)
                .ok_or(Error::Configuration {
                    message: "local dimensions must be nonzero and fit this platform",
                })?;
        if idle.is_zero() {
            return Err(Error::Configuration {
                message: "local idle unload duration must be positive",
            }
            .into());
        }
        let settings = Settings::from_profile(&profile)?;
        let id = format!("llama/{}", profile.model);
        let query_prefix = profile.query_prefix.clone();
        let state = Arc::new(Mutex::new(ResidentModel {
            instance: next_resident_instance(),
            native: None,
            path,
            profile,
            last_used: Instant::now(),
            artifact,
        }));
        Ok(Self {
            state,
            inference: Arc::new(tokio::sync::Semaphore::new(1)),
            idle,
            idle_started: OnceLock::new(),
            id,
            query_prefix,
            dims,
            token_limit: settings.token_limit(),
        })
    }

    async fn load_prepared(self) -> super::Result<Self> {
        let state = Arc::clone(&self.state);
        let dims = self.dims;
        tokio::task::spawn_blocking(move || {
            let mut resident = state.lock().map_err(|_| super::Error::LocalWorker)?;
            let native = load_native(
                &resident.path,
                &resident.profile,
                resident.artifact.as_ref(),
            )?;
            validate_dimensions(&native, dims)?;
            resident.native = Some(native);
            residency_event("loaded", "embedder", resident.instance, 0);
            resident.last_used = Instant::now();
            Ok::<_, super::Error>(())
        })
        .await
        .map_err(|_| super::Error::LocalWorker)??;
        self.start_idle_unloader();
        Ok(self)
    }

    fn start_idle_unloader(&self) {
        self.idle_started.get_or_init(|| {
            tokio::spawn(unload_idle(Arc::downgrade(&self.state), self.idle));
        });
    }

    /// Prepares the exact query instruction once. cl100k is a ledger estimate;
    /// native tokenization remains authoritative for the local context limit.
    pub(crate) fn prepare_query(&self, query: &str) -> Result<PreparedQuery, AttemptError> {
        PreparedQuery::new(self.query_contract(), &self.query_prefix, query, usize::MAX)
    }

    fn query_contract(&self) -> [u8; 32] {
        query_contract("local", &self.id, &self.query_prefix, self.dims)
    }

    /// Executes prepared bytes with the existing native cancellation/capacity gate.
    /// Native usage is unavailable, so settlement must retain an explicit estimate.
    pub(crate) async fn embed_query_with_usage_once(
        &self,
        query: &PreparedQuery,
    ) -> Result<EmbeddingBatch, AttemptError> {
        let text = query.input(self.query_contract())?.to_owned();
        let embeddings = self
            .embed_texts(vec![text])
            .await
            .map_err(|error| match error {
                super::Error::Local(Error::InputTooLong { .. }) => AttemptError::InvalidInput,
                _ => AttemptError::InvalidResponse,
            })?;
        Ok(EmbeddingBatch {
            embeddings,
            tokens: None,
        })
    }

    #[cfg(test)]
    /// Observes residency without blocking an active load, inference or unload.
    pub(crate) fn resident_for_test(&self) -> Option<bool> {
        self.state
            .try_lock()
            .ok()
            .map(|resident| resident.native.is_some())
    }

    async fn embed_texts(&self, texts: Vec<String>) -> super::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.start_idle_unloader();
        let state = Arc::clone(&self.state);
        let dims = self.dims;
        cancellation::run(Arc::clone(&self.inference), move |stopped| {
            stopped.check()?;
            state
                .lock()
                .map_err(|_| super::Error::LocalWorker)?
                .embed(&texts, dims, &stopped)
        })
        .await
    }

    async fn embed_contexts(&self, contexts: Vec<OwnedContext>) -> super::Result<Vec<Vec<f32>>> {
        if contexts.is_empty() {
            return Ok(Vec::new());
        }
        self.start_idle_unloader();
        let state = Arc::clone(&self.state);
        let dims = self.dims;
        cancellation::run(Arc::clone(&self.inference), move |stopped| {
            stopped.check()?;
            state
                .lock()
                .map_err(|_| super::Error::LocalWorker)?
                .embed_contexts(&contexts, dims, &stopped)
        })
        .await
    }
}

#[async_trait::async_trait]
impl super::Embedder for LocalProvider {
    fn id(&self) -> &str {
        &self.id
    }
    fn dims(&self) -> usize {
        self.dims
    }
    fn max_batch_tokens(&self) -> usize {
        self.token_limit
    }
    fn max_batch_items(&self) -> usize {
        MAX_BATCH_ITEMS
    }

    async fn embed_documents(
        &self,
        docs: &[super::EmbedInput<'_>],
    ) -> super::Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(docs.len());
        for batch in docs.chunks(MAX_BATCH_ITEMS) {
            let contexts = batch
                .iter()
                .map(|doc| (doc.header.to_owned(), doc.text.to_owned()))
                .collect();
            embeddings.extend(self.embed_contexts(contexts).await?);
        }
        Ok(embeddings)
    }

    async fn embed_query(&self, query: &str) -> super::Result<Vec<f32>> {
        self.embed_texts(vec![format!("{}{}", self.query_prefix, query)])
            .await?
            .pop()
            .ok_or_else(|| super::Error::Response {
                message: "local model returned no query vector".to_owned(),
            })
    }
}

/// A loaded Qwen local embedding model that produces normalized document vectors.
///
/// The backend and model remain loaded across calls, while contexts are per batch so
/// concurrent queue workers never share a mutable llama.cpp context.
pub(crate) struct LocalEmbedder {
    model: LlamaModel,
    backend: &'static LlamaBackend,
    settings: Settings,
    artifact: Option<(
        super::registry::Model,
        PathBuf,
        Option<super::registry::VerifiedArtifact>,
    )>,
}

impl LocalEmbedder {
    /// Loads the Qwen embedding model from `path` for the local profile.
    pub(crate) fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::load_with_profile(path, &crate::config::LocalEmbedder::default())
    }

    /// Loads a model with validated local limits and automatically detected CPU cores.
    pub(crate) fn load_with_profile(
        path: impl AsRef<Path>,
        profile: &crate::config::LocalEmbedder,
    ) -> Result<Self, Error> {
        let settings = Settings::from_profile(profile)?;
        let backend = shared_backend()?;

        let params = settings.model_params();
        let model =
            LlamaModel::load_from_file(backend, path, &params).map_err(|error| Error::Llama {
                operation: "model load",
                message: error.to_string(),
            })?;

        Ok(Self {
            model,
            backend,
            settings,
            artifact: None,
        })
    }

    /// Embeds documents in bounded batches using `Last` pooling and L2 normalization.
    pub(crate) fn embed_documents(&self, documents: &[&str]) -> Result<Vec<Vec<f32>>, Error> {
        self.embed_cancellable(documents, &Cancellation::default())
    }

    fn verify_artifact(&self) -> Result<(), Error> {
        if let Some((model, cache, proof)) = &self.artifact {
            if let Some(proof) = proof {
                if !proof.matches(model, cache) {
                    return Err(Error::Artifact);
                }
            } else {
                super::registry::verify(model, cache).map_err(|_| Error::Artifact)?;
            }
        }
        Ok(())
    }

    fn embed_cancellable(
        &self,
        documents: &[&str],
        stopped: &Cancellation,
    ) -> Result<Vec<Vec<f32>>, Error> {
        stopped.check()?;
        self.verify_artifact()?;
        self.embed_tokenized(
            documents.iter().map(|document| self.tokenize(document)),
            documents.len(),
            stopped,
        )
    }

    fn embed_contexts_cancellable(
        &self,
        documents: &[(&str, &str)],
        stopped: &Cancellation,
    ) -> Result<Vec<Vec<f32>>, Error> {
        stopped.check()?;
        self.verify_artifact()?;
        self.embed_tokenized(
            documents
                .iter()
                .map(|(header, text)| self.tokenize_context(header, text)),
            documents.len(),
            stopped,
        )
    }

    fn embed_tokenized(
        &self,
        documents: impl Iterator<Item = Result<Vec<llama_cpp_2::token::LlamaToken>, Error>>,
        document_count: usize,
        stopped: &Cancellation,
    ) -> Result<Vec<Vec<f32>>, Error> {
        let mut embeddings = Vec::with_capacity(document_count);
        let mut pending = Vec::with_capacity(MAX_BATCH_ITEMS);
        let mut pending_tokens = 0;
        for document in documents {
            stopped.check()?;
            let tokens = document?;
            if batch_is_full(
                pending.len(),
                pending_tokens,
                tokens.len(),
                self.settings.token_limit(),
            ) {
                embeddings.extend(self.embed_batch(&pending, stopped)?);
                pending.clear();
                pending_tokens = 0;
            }
            pending_tokens += tokens.len();
            pending.push(tokens);
        }
        if !pending.is_empty() {
            embeddings.extend(self.embed_batch(&pending, stopped)?);
        }

        stopped.check()?;
        self.verify_artifact()?;
        Ok(embeddings)
    }

    fn tokenize_context(
        &self,
        header: &str,
        text: &str,
    ) -> Result<Vec<llama_cpp_2::token::LlamaToken>, Error> {
        let rendered = format!("{header}\n{text}");
        match self.tokenize(&rendered) {
            Ok(tokens) => return Ok(tokens),
            Err(Error::InputTooLong { .. }) => {}
            Err(error) => return Err(error),
        }
        let header = format!("{header}\n");
        let header = self
            .model
            .str_to_token(&header, AddBos::Always)
            .map_err(|error| Error::Llama {
                operation: "context tokenization",
                message: error.to_string(),
            })?;
        let text = self
            .model
            .str_to_token(text, AddBos::Never)
            .map_err(|error| Error::Llama {
                operation: "document tokenization",
                message: error.to_string(),
            })?;
        let tokens = header.len().saturating_add(text.len());
        fit_context_tokens(header, text, self.settings.token_limit()).ok_or(Error::InputTooLong {
            tokens,
            max_tokens: self.settings.token_limit(),
        })
    }

    fn tokenize(&self, document: &str) -> Result<Vec<llama_cpp_2::token::LlamaToken>, Error> {
        let tokens = self
            .model
            .str_to_token(document, AddBos::Always)
            .map_err(|error| Error::Llama {
                operation: "tokenization",
                message: error.to_string(),
            })?;
        if tokens.len() > self.settings.token_limit() {
            return Err(Error::InputTooLong {
                tokens: tokens.len(),
                max_tokens: self.settings.token_limit(),
            });
        }
        Ok(tokens)
    }

    fn embed_batch(
        &self,
        documents: &[Vec<llama_cpp_2::token::LlamaToken>],
        stopped: &Cancellation,
    ) -> Result<Vec<Vec<f32>>, Error> {
        stopped.check()?;
        let token_count = documents.iter().map(Vec::len).sum::<usize>();
        let started = Instant::now();
        let context_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.settings.context_tokens))
            // Share the token budget; separate KV partitions constrain each input
            // to n_ctx / n_seq_max and reject otherwise-valid mixed-length batches.
            .with_kv_unified(true)
            .with_n_batch(self.settings.batch_tokens)
            .with_n_ubatch(self.settings.batch_tokens)
            .with_n_seq_max(u32::try_from(documents.len()).unwrap_or(u32::MAX))
            .with_n_threads(self.settings.threads)
            .with_n_threads_batch(self.settings.threads)
            .with_pooling_type(LlamaPoolingType::Last)
            .with_embeddings(true);
        let mut context = self
            .model
            .new_context(self.backend, context_params)
            .map_err(|error| Error::Llama {
                operation: "context initialization",
                message: error.to_string(),
            })?;
        stopped.check()?;
        let context_ready = Instant::now();
        let mut batch = LlamaBatch::new(
            token_count,
            i32::try_from(documents.len()).unwrap_or(i32::MAX),
        );

        for (sequence, tokens) in documents.iter().enumerate() {
            let sequence = i32::try_from(sequence).unwrap_or(i32::MAX);
            for (position, token) in tokens.iter().enumerate() {
                batch
                    .add(
                        *token,
                        i32::try_from(position).unwrap_or(i32::MAX),
                        &[sequence],
                        false,
                    )
                    .map_err(|error| Error::Llama {
                        operation: "batch construction",
                        message: error.to_string(),
                    })?;
            }
        }

        stopped.check()?;
        let batch_ready = Instant::now();
        context.decode(&mut batch).map_err(|error| Error::Llama {
            operation: "embedding decode",
            message: error.to_string(),
        })?;
        stopped.check()?;
        let decoded = Instant::now();

        let embeddings = documents
            .iter()
            .enumerate()
            .map(|(sequence, _)| {
                let sequence = i32::try_from(sequence).unwrap_or(i32::MAX);
                let embedding =
                    context
                        .embeddings_seq_ith(sequence)
                        .map_err(|error| Error::Llama {
                            operation: "embedding read",
                            message: error.to_string(),
                        })?;
                normalize(embedding)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let read = Instant::now();
        drop(context);
        tracing::debug!(
            documents = documents.len(),
            tokens = token_count,
            context_ms = context_ready.duration_since(started).as_secs_f64() * 1000.0,
            batch_ms = batch_ready.duration_since(context_ready).as_secs_f64() * 1000.0,
            decode_ms = decoded.duration_since(batch_ready).as_secs_f64() * 1000.0,
            read_ms = read.duration_since(decoded).as_secs_f64() * 1000.0,
            drop_ms = read.elapsed().as_secs_f64() * 1000.0,
            "local embedding batch timings"
        );
        Ok(embeddings)
    }
}

// Bound tokenized working memory to the pending context and next input, preserving order.
// A single input is checked by tokenize before reaching this admission check.
fn batch_is_full(items: usize, tokens: usize, next_tokens: usize, limit: usize) -> bool {
    items > 0 && (items >= MAX_BATCH_ITEMS || tokens + next_tokens > limit)
}

fn fit_context_tokens<T>(mut header: Vec<T>, mut text: Vec<T>, limit: usize) -> Option<Vec<T>> {
    if header.len() > limit {
        return None;
    }
    text.truncate(limit - header.len());
    header.extend(text);
    Some(header)
}

/// Resolves the standard local model path from the live-test environment.
pub(crate) fn model_path_from_env() -> Option<PathBuf> {
    std::env::var_os("DKC_MODELS_DIR")
        .map(|directory| PathBuf::from(directory).join(EMBEDDING_MODEL_FILE))
}

fn normalize(embedding: &[f32]) -> Result<Vec<f32>, Error> {
    let squared_norm = embedding.iter().map(|value| value * value).sum::<f32>();
    if !squared_norm.is_finite() || squared_norm <= f32::EPSILON {
        return Err(Error::InvalidEmbedding);
    }

    let norm = squared_norm.sqrt();
    let normalized = embedding
        .iter()
        .map(|value| value / norm)
        .collect::<Vec<_>>();
    if normalized.iter().all(|value| value.is_finite()) {
        Ok(normalized)
    } else {
        Err(Error::InvalidEmbedding)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io,
        time::{Duration, Instant},
    };

    use super::{LocalEmbedder, batch_is_full, fit_context_tokens, model_path_from_env};

    #[test]
    fn context_token_fit_preserves_header_and_drops_only_the_text_tail() {
        assert_eq!(
            fit_context_tokens(vec![1, 2], vec![3, 4, 5], 4),
            Some(vec![1, 2, 3, 4])
        );
        assert_eq!(
            fit_context_tokens(vec![1, 2], vec![3, 4], 4),
            Some(vec![1, 2, 3, 4])
        );
        assert_eq!(fit_context_tokens(vec![1, 2, 3], vec![4], 2), None);
    }

    #[test]
    fn token_batches_preserve_order_and_fill_context() {
        let lengths = [1_500, 500, 100, 1_900, 148, 2_048, 1];
        let mut batches = Vec::new();
        let mut pending = Vec::new();
        let mut total = 0;
        for length in lengths {
            if batch_is_full(pending.len(), total, length, 2_048) {
                batches.push(std::mem::take(&mut pending));
                total = 0;
            }
            total += length;
            pending.push(length);
        }
        batches.push(pending);
        assert_eq!(
            batches,
            vec![
                vec![1_500, 500],
                vec![100, 1_900],
                vec![148],
                vec![2_048],
                vec![1]
            ]
        );
        assert!(!batch_is_full(0, 0, 2_048, 2_048));
        assert!(!batch_is_full(31, 31, 1, 2_048));
        assert!(batch_is_full(32, 32, 1, 2_048));
        assert!(!batch_is_full(1, 1_900, 148, 2_048));
        assert!(batch_is_full(1, 1_900, 149, 2_048));
    }

    #[test]
    fn local_settings_validate_native_bounds_and_apply_overrides() -> Result<(), super::Error> {
        let defaults = crate::config::LocalEmbedder::default();
        let settings = super::Settings::from_profile(&defaults)?;
        assert_eq!(settings.threads as usize, num_cpus::get_physical().max(1));
        assert_eq!(settings.token_limit(), 2_048);
        let accelerated = cfg!(any(
            feature = "cuda",
            feature = "metal",
            feature = "vulkan",
            all(target_os = "macos", target_arch = "aarch64")
        ));
        assert_eq!(
            settings.model_params().n_gpu_layers(),
            if accelerated { -1 } else { 0 }
        );
        let custom = crate::config::LocalEmbedder {
            threads: 2,
            n_ctx: 512,
            batch_tokens: 256,
            gpu_layers: 3,
            ..defaults.clone()
        };
        let settings = super::Settings::from_profile(&custom)?;
        assert_eq!(settings.threads, 2);
        assert_eq!(settings.token_limit(), 256);
        assert_eq!(
            settings.model_params().n_gpu_layers(),
            if accelerated { 3 } else { 0 }
        );
        for invalid in [
            crate::config::LocalEmbedder {
                n_ctx: 0,
                ..defaults.clone()
            },
            crate::config::LocalEmbedder {
                batch_tokens: 0,
                ..defaults.clone()
            },
            crate::config::LocalEmbedder {
                threads: u64::MAX,
                ..defaults.clone()
            },
            crate::config::LocalEmbedder {
                n_ctx: u64::MAX,
                ..defaults.clone()
            },
            crate::config::LocalEmbedder {
                batch_tokens: u64::MAX,
                ..defaults.clone()
            },
            crate::config::LocalEmbedder {
                gpu_layers: -2,
                ..defaults.clone()
            },
            crate::config::LocalEmbedder {
                gpu_layers: i64::MAX,
                ..defaults
            },
        ] {
            // Validation runs before model-path access or global backend initialization.
            assert!(matches!(
                LocalEmbedder::load_with_profile("absent.gguf", &invalid),
                Err(super::Error::Configuration { .. })
            ));
        }
        Ok(())
    }

    #[tokio::test]
    async fn provider_rejects_invalid_settings_before_loading() {
        let result = super::LocalProvider::load("absent.gguf".into(), Default::default(), 0).await;
        assert!(matches!(
            result,
            Err(crate::models::Error::Local(
                super::Error::Configuration { .. }
            ))
        ));
        let result = super::LocalProvider::load_with_idle(
            "absent.gguf".into(),
            Default::default(),
            256,
            std::time::Duration::ZERO,
        )
        .await;
        assert!(matches!(
            result,
            Err(crate::models::Error::Local(
                super::Error::Configuration { .. }
            ))
        ));
    }

    #[test]
    fn registered_preparation_needs_neither_runtime_nor_weights() {
        let directory = tempfile::tempdir().unwrap();
        let cache = directory.path().join("absent-cache");
        let provider = super::LocalProvider::prepare_registered(
            cache.clone(),
            Default::default(),
            256,
            Duration::from_secs(60),
        )
        .unwrap();
        assert_eq!(provider.resident_for_test(), Some(false));
        assert!(provider.idle_started.get().is_none());
        assert!(!cache.exists());
        let resident = provider.state.lock().unwrap();
        let (model, stored_cache) = resident.artifact.as_ref().unwrap();
        assert_eq!(stored_cache, &cache);
        assert_eq!(resident.path, cache.join(&model.file));
        assert_eq!(model.pooling, "last");
        assert_eq!(provider.dims, 256);
        assert!(provider.prepare_query("find the parser").is_ok());
    }

    #[test]
    fn registered_preparation_rejects_invalid_recipes_without_cache_access() {
        let directory = tempfile::tempdir().unwrap();
        // A regular file cannot be a cache directory, but preparation must not open it.
        let cache = directory.path().join("cache");
        std::fs::write(&cache, b"unchanged").unwrap();
        let defaults = crate::config::LocalEmbedder::default();
        for (profile, dims, idle) in [
            (defaults.clone(), 0, Duration::from_secs(60)),
            (defaults.clone(), u64::MAX, Duration::from_secs(60)),
            (defaults.clone(), 256, Duration::ZERO),
            (
                crate::config::LocalEmbedder {
                    n_ctx: 0,
                    ..defaults.clone()
                },
                256,
                Duration::from_secs(60),
            ),
            (
                crate::config::LocalEmbedder {
                    model: "unregistered".into(),
                    ..defaults
                },
                256,
                Duration::from_secs(60),
            ),
        ] {
            assert!(
                super::LocalProvider::prepare_registered(cache.clone(), profile, dims, idle)
                    .is_err()
            );
        }
        assert_eq!(std::fs::read(cache).unwrap(), b"unchanged");
    }

    #[tokio::test]
    async fn lazy_registered_inference_rejects_corrupt_weights_before_native_load() {
        let directory = tempfile::tempdir().unwrap();
        let provider = super::LocalProvider::prepare_registered(
            directory.path().into(),
            Default::default(),
            256,
            Duration::from_secs(60),
        )
        .unwrap();
        let path = provider.state.lock().unwrap().path.clone();
        std::fs::write(&path, b"not registered weights").unwrap();
        assert!(provider.embed_texts(Vec::new()).await.unwrap().is_empty());
        assert!(provider.idle_started.get().is_none());
        let result = provider.embed_texts(vec!["find the parser".into()]).await;
        assert!(matches!(
            result,
            Err(crate::models::Error::Local(super::Error::Artifact))
        ));
        assert_eq!(provider.resident_for_test(), Some(false));
        assert_eq!(std::fs::read(path).unwrap(), b"not registered weights");
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    }

    #[tokio::test]
    async fn cancelled_lazy_query_never_enters_native_work_after_capacity_returns() {
        use std::{
            future::Future,
            task::{Context, Waker},
        };
        let directory = tempfile::tempdir().unwrap();
        let cache = directory.path().join("absent-cache");
        let provider = super::LocalProvider::prepare_registered(
            cache.clone(),
            Default::default(),
            256,
            Duration::from_secs(60),
        )
        .unwrap();
        let permit = provider.inference.clone().acquire_owned().await.unwrap();
        let query = provider.prepare_query("find the parser").unwrap();
        let mut pending = Box::pin(provider.embed_query_with_usage_once(&query));
        assert!(
            pending
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        drop(pending);
        drop(permit);
        // Immediate reacquisition proves the cancelled waiter owns no capacity.
        let permit = provider.inference.clone().try_acquire_owned().unwrap();
        assert_eq!(provider.resident_for_test(), Some(false));
        assert!(!cache.exists());
        drop(permit);
    }

    const SANITY_INPUT: &str = "fn should_retry(err: &ChargeError, attempt: u32) -> RetryDecision";

    #[cfg(feature = "cuda")]
    #[tokio::test]
    #[ignore = "requires cached registered Qwen3 embedding weights and an idle CUDA GPU"]
    async fn embed_cuda_one_second_idle_unloads_and_reloads_same_provider()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::models::{EmbedInput, Embedder};
        let cache = std::path::PathBuf::from(
            std::env::var_os("DKC_MODELS_DIR").ok_or("DKC_MODELS_DIR is not set")?,
        );
        let profile = crate::config::LocalEmbedder {
            gpu_layers: -1,
            ..Default::default()
        };
        // Registered loading verifies the cached SHA256 on first load and reload;
        // it never downloads. This test-only idle duration leaves defaults intact.
        let provider = tokio::time::timeout(
            Duration::from_secs(120),
            super::LocalProvider::load_registered(
                cache,
                profile.clone(),
                1024,
                Duration::from_secs(1),
            ),
        )
        .await??;
        let instance = {
            let state = provider.state.lock().map_err(|_| "poisoned model state")?;
            assert_eq!(state.profile.gpu_layers, -1);
            assert!(state.native.is_some());
            state.instance
        };
        assert_eq!(provider.id(), format!("llama/{}", profile.model));
        let input = [EmbedInput::from_context(
            "fixture > retry.rs\nfn should_retry\nkind: code",
            SANITY_INPUT,
            crate::store::UnitKind::Chunk,
        )];
        let documents =
            tokio::time::timeout(Duration::from_secs(30), provider.embed_documents(&input))
                .await??;
        let before =
            tokio::time::timeout(Duration::from_secs(30), provider.embed_query(SANITY_INPUT))
                .await??;
        assert_eq!(documents.len(), 1);
        for vector in [&documents[0], &before] {
            assert_eq!(vector.len(), 1024);
            assert!(vector.iter().all(|value| value.is_finite()));
            let norm: f32 = vector.iter().map(|value| value * value).sum();
            assert!((norm - 1.0).abs() < 1e-4);
        }
        assert_eq!(provider.resident_for_test(), Some(true));
        tokio::time::timeout(Duration::from_secs(10), async {
            while provider.resident_for_test() != Some(false) {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await?;
        {
            let state = provider.state.lock().map_err(|_| "poisoned model state")?;
            assert_eq!(state.instance, instance);
            assert!(
                state.native.is_none(),
                "timer did not release CUDA native model"
            );
        }
        let started = Instant::now();
        let after =
            tokio::time::timeout(Duration::from_secs(120), provider.embed_query(SANITY_INPUT))
                .await??;
        assert_eq!(after.len(), 1024);
        assert!(after.iter().all(|value| value.is_finite()));
        let norm: f32 = after.iter().map(|value| value * value).sum();
        assert!((norm - 1.0).abs() < 1e-4);
        let cosine: f32 = before.iter().zip(&after).map(|(a, b)| a * b).sum();
        assert!(
            cosine > 0.999,
            "CUDA reload changed query embedding: {cosine}"
        );
        {
            let state = provider.state.lock().map_err(|_| "poisoned model state")?;
            assert_eq!(state.instance, instance);
            assert_eq!(state.profile.gpu_layers, -1);
            assert!(state.native.is_some());
        }
        println!(
            "CUDA registered same-provider idle reload+query: {:.3}s; cosine={cosine}",
            started.elapsed().as_secs_f64()
        );
        Ok(())
    }
    #[test]
    #[ignore = "requires DKC_MODELS_DIR to contain Qwen3-Embedding-0.6B-Q8_0.gguf"]
    fn embed_sanity() -> Result<(), Box<dyn std::error::Error>> {
        let path = model_path_from_env().ok_or("DKC_MODELS_DIR is not set")?;
        let embedder = LocalEmbedder::load(&path)?;
        assert!(embedder.embed_documents(&[])?.is_empty());
        let vector = embedder.embed_documents(&[SANITY_INPUT])?;

        assert_eq!(vector.len(), 1);
        assert_eq!(vector[0].len(), 1_024);
        let norm = vector[0]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
        // Mixed lengths cross the context cap while retaining multi-document batches.
        let long = "fn retry() { return; }\n".repeat(130);
        let texts = [
            SANITY_INPUT,
            long.as_str(),
            "struct ChargeError;",
            long.as_str(),
        ];
        let together = embedder.embed_documents(&texts)?;
        assert_eq!(together.len(), texts.len());
        for (text, batched) in texts.iter().zip(together) {
            let alone = embedder.embed_documents(&[text])?;
            let cosine: f32 = alone[0].iter().zip(batched).map(|(a, b)| a * b).sum();
            assert!(cosine > 0.999, "batch changed embedding: {cosine}");
        }
        let oversized = "token ".repeat(2_049);
        assert!(matches!(
            embedder.embed_documents(&[&oversized]),
            Err(super::Error::InputTooLong { .. })
        ));
        drop(embedder);
        let profile = crate::config::LocalEmbedder {
            threads: 2,
            n_ctx: 512,
            batch_tokens: 256,
            gpu_layers: 0,
            ..crate::config::LocalEmbedder::default()
        };
        let reloaded = LocalEmbedder::load_with_profile(&path, &profile)?;
        assert!(matches!(
            reloaded.embed_documents(&[long.as_str()]),
            Err(super::Error::InputTooLong {
                max_tokens: 256,
                ..
            })
        ));
        let after_reload = reloaded.embed_documents(&[SANITY_INPUT])?;
        let cosine: f32 = vector[0]
            .iter()
            .zip(&after_reload[0])
            .map(|(a, b)| a * b)
            .sum();
        assert!(cosine > 0.999);
        drop(reloaded);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        runtime.block_on(async {
            use crate::models::{EmbedInput, Embedder};
            let profile = crate::config::LocalEmbedder {
                threads: 2,
                gpu_layers: 0,
                query_prefix: "Instruct: Find relevant code\nQuery: ".to_owned(),
                ..Default::default()
            };
            let models = crate::config::Models {
                unload_after_idle: "1s".to_owned(),
            };
            let provider = super::LocalProvider::load_with_idle(
                path,
                profile.clone(),
                256,
                models.unload_after_idle()?,
            )
            .await?;
            assert_eq!(provider.id(), format!("llama/{}", profile.model));
            assert_eq!(provider.dims(), 256);
            assert_eq!(provider.max_batch_tokens(), 2048);
            assert_eq!(provider.max_batch_items(), 32);
            assert!(provider.embed_documents(&[]).await?.is_empty());
            let header = "repository > src/retry.rs\nfn retry\nkind: code";
            let inputs = [EmbedInput::from_context(
                header,
                SANITY_INPUT,
                crate::store::UnitKind::Chunk,
            )];
            let actual = provider.embed_documents(&inputs).await?;
            let oversized_text = "token ".repeat(2_049);
            let oversized_document = [EmbedInput::from_context(
                header,
                &oversized_text,
                crate::store::UnitKind::History,
            )];
            let truncated = provider.embed_documents(&oversized_document).await?;
            assert_eq!(truncated.len(), 1);
            assert_eq!(truncated[0].len(), 256);
            let query = provider.embed_query(SANITY_INPUT).await?;
            let raw = LocalEmbedder::load_with_profile(
                model_path_from_env().ok_or("missing model path")?,
                &profile,
            )?;
            let expected = raw.embed_documents(&[
                &format!("{header}\n{SANITY_INPUT}"),
                &format!("{}{SANITY_INPUT}", profile.query_prefix),
            ])?;
            for (actual, expected) in [&actual[0], &query].into_iter().zip(expected) {
                let expected = crate::models::normalize_embedding(expected, 256)?;
                assert_eq!(actual.len(), 256);
                let norm: f32 = actual.iter().map(|value| value * value).sum();
                assert!((norm - 1.0).abs() < 1e-4);
                let cosine: f32 = actual.iter().zip(expected).map(|(a, b)| a * b).sum();
                assert!(cosine > 0.999);
            }
            assert!(matches!(
                provider.embed_query(&"token ".repeat(2049)).await,
                Err(crate::models::Error::Local(
                    super::Error::InputTooLong { .. }
                ))
            ));
            // The timer must actually release the native model while another one remains usable.
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                loop {
                    let unloaded = provider
                        .state
                        .lock()
                        .map_err(|_| "poisoned model state")?
                        .native
                        .is_none();
                    if unloaded {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                }
                Ok::<(), Box<dyn std::error::Error>>(())
            })
            .await??;
            assert_eq!(raw.embed_documents(&[SANITY_INPUT])?.len(), 1);
            let started = Instant::now();
            let after_idle = provider.embed_query(SANITY_INPUT).await?;
            println!(
                "local idle reload + query: {:.3}s",
                started.elapsed().as_secs_f64()
            );
            let cosine: f32 = query.iter().zip(after_idle).map(|(a, b)| a * b).sum();
            assert!(cosine > 0.999);
            assert!(
                provider
                    .state
                    .lock()
                    .map_err(|_| "poisoned model state")?
                    .native
                    .is_some()
            );
            let weak = std::sync::Arc::downgrade(&provider.state);
            drop(provider);
            assert!(
                weak.upgrade().is_none(),
                "idle timer retained the model owner"
            );
            assert!(matches!(
                super::LocalProvider::load(
                    model_path_from_env().ok_or("missing model path")?,
                    profile,
                    1025
                )
                .await,
                Err(crate::models::Error::Local(
                    super::Error::Configuration { .. }
                ))
            ));
            Ok::<(), Box<dyn std::error::Error>>(())
        })?;
        Ok(())
    }

    #[test]
    #[ignore = "requires DKC_RERANKER_GGUF pointing to a converted official artifact"]
    fn reranker_artifact_sanity() -> Result<(), Box<dyn std::error::Error>> {
        use std::{num::NonZeroU32, path::PathBuf};

        use super::{
            AddBos, LlamaBatch, LlamaContextParams, LlamaModel, LlamaModelParams, LlamaPoolingType,
        };

        let path = std::env::var_os("DKC_RERANKER_GGUF").ok_or("missing reranker artifact")?;
        let backend = super::shared_backend()?;
        let model = LlamaModel::load_from_file(
            backend,
            PathBuf::from(path),
            &LlamaModelParams::default().with_n_gpu_layers(0),
        )?;
        assert_eq!(model.meta_val_str("qwen3.pooling_type")?, "4");
        let template = model.meta_val_str("tokenizer.chat_template.rerank")?;
        assert!(template.contains("<Query>: {query}"));
        assert!(template.contains("<Document>: {document}"));
        let pairs = [
            (
                "What is the capital of France?",
                "The capital of France is Paris.",
            ),
            (
                "What is the capital of France?",
                "Bananas are yellow fruit.",
            ),
            (
                "What is the capital of France?",
                "France is a country in Europe.",
            ),
        ];
        let mut scores = Vec::new();
        for (query, document) in pairs {
            let prompt = template
                .replace("{query}", query)
                .replace("{document}", document);
            let tokens = model.str_to_token(&prompt, AddBos::Always)?;
            assert!(tokens.len() <= 2048);
            let params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(2048))
                .with_n_batch(2048)
                .with_n_ubatch(2048)
                .with_n_seq_max(1)
                .with_n_threads(8)
                .with_n_threads_batch(8)
                .with_pooling_type(LlamaPoolingType::Rank)
                .with_embeddings(true);
            let mut context = model.new_context(backend, params)?;
            let mut batch = LlamaBatch::new(tokens.len(), 1);
            for (position, token) in tokens.iter().enumerate() {
                batch.add(*token, i32::try_from(position)?, &[0], false)?;
            }
            context.decode(&mut batch)?;
            let probabilities = context.embeddings_seq_ith(0)?;
            assert_eq!(probabilities.len(), 2);
            assert!(
                probabilities
                    .iter()
                    .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
            );
            // The pinned Qwen3 Rank graph already applies softmax. A second
            // softmax would compress the scores and break the irrelevant gate.
            assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            scores.push(probabilities[0]); // Converted classifier labels: yes, no.
        }
        println!("reranker artifact P(yes): {scores:?}");
        assert!(scores[0] > 0.5);
        assert!(scores[1] < 0.2);
        let min = scores.iter().copied().fold(f32::INFINITY, f32::min);
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(max - min > 0.02);
        Ok(())
    }

    #[test]
    #[ignore = "requires DKC_MODELS_DIR and gateway credentials"]
    fn local_gateway_cosine() -> Result<(), Box<dyn std::error::Error>> {
        let path = model_path_from_env().ok_or("DKC_MODELS_DIR is not set")?;
        let embedder = LocalEmbedder::load(path)?;
        let local = embedder.embed_documents(&[SANITY_INPUT])?;
        let remote = gateway_embedding(SANITY_INPUT)?;

        assert_eq!(local[0].len(), remote.len());
        let cosine = local[0]
            .iter()
            .zip(&remote)
            .map(|(left, right)| left * right)
            .sum::<f32>();
        println!("local/gateway cosine: {cosine:.6}");
        assert!(cosine > 0.98, "local/gateway cosine was {cosine:.6}");
        Ok(())
    }

    #[test]
    #[ignore = "requires DKC_MODELS_DIR to contain Qwen3-Embedding-0.6B-Q8_0.gguf"]
    fn embed_throughput() -> Result<(), Box<dyn std::error::Error>> {
        // Opt-in component profiling emits counts/durations, never document text.
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_ansi(false)
            .without_time()
            .finish();
        let _trace = (std::env::var("DKC_BENCH_BATCH_TIMINGS").as_deref() == Ok("1"))
            .then(|| tracing::subscriber::set_default(subscriber));
        let path = model_path_from_env().ok_or("DKC_MODELS_DIR is not set")?;
        let threads = std::env::var("DKC_BENCH_CPU_THREADS")
            .ok()
            .map(|value| value.parse::<u64>())
            .transpose()?
            .unwrap_or(0);
        let repeats = std::env::var("DKC_BENCH_DOCUMENT_REPEATS")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()?
            .unwrap_or(1);
        let document_count = std::env::var("DKC_BENCH_DOCUMENT_COUNT")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()?
            .unwrap_or(256);
        if repeats == 0 || document_count == 0 {
            return Err(
                io::Error::other("benchmark document count and repeats must be positive").into(),
            );
        }
        let profile = crate::config::LocalEmbedder {
            threads,
            ..Default::default()
        };
        let embedder = LocalEmbedder::load_with_profile(path, &profile)?;
        let documents = (0..document_count)
            .map(|index| {
                format!(
                    "fn should_retry_{index}(err: &ChargeError, attempt: u32) -> RetryDecision\n"
                )
                .repeat(repeats)
            })
            .collect::<Vec<_>>();
        let document_refs = documents.iter().map(String::as_str).collect::<Vec<_>>();

        let token_counts = documents
            .iter()
            .map(|text| embedder.tokenize(text).map(|tokens| tokens.len()))
            .collect::<Result<Vec<_>, _>>()?;
        println!(
            "embedding benchmark: threads={}, documents={}, tokens_min={}, tokens_max={}, tokens_total={}, cuda={}, metal={}, requested_gpu_layers={}",
            embedder.settings.threads,
            documents.len(),
            token_counts.iter().min().unwrap_or(&0),
            token_counts.iter().max().unwrap_or(&0),
            token_counts.iter().sum::<usize>(),
            cfg!(feature = "cuda"),
            cfg!(feature = "metal"),
            embedder.settings.gpu_layers,
        );
        let started = Instant::now();
        let vectors = embedder.embed_documents(&document_refs)?;
        let elapsed = started.elapsed();
        let units_per_second = vectors.len() as f64 / elapsed.as_secs_f64();
        assert_eq!(vectors.len(), documents.len());
        println!(
            "embedding throughput: {units_per_second:.2} units/s ({} documents in {:.3}s)",
            vectors.len(),
            elapsed.as_secs_f64(),
        );
        Ok(())
    }

    fn gateway_embedding(input: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let endpoint = reqwest::Url::parse("https://gateway.moonly.ai/v1/embeddings")?;
        let api_key = crate::models::openai_compat::resolve_api_key(
            "MOONLY_GATEWAY_API_KEY",
            &endpoint,
            crate::config::credentials_path().as_deref(),
        )?;
        let client = reqwest::blocking::Client::builder()
            .user_agent(concat!("dkc/", env!("CARGO_PKG_VERSION")))
            .timeout(Duration::from_secs(60))
            .build()?;
        let response = client
            .post(endpoint)
            .bearer_auth(api_key)
            .json(&serde_json::json!({ "input": [input], "model": "qwen3-embedding-0.6b" }))
            .send()?
            .error_for_status()?
            .json::<serde_json::Value>()?;
        let embedding = response
            .pointer("/data/0/embedding")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| io::Error::other("gateway response has no data[0].embedding array"))?;
        embedding
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .map(|value| value as f32)
                    .ok_or_else(|| io::Error::other("gateway embedding contains a non-number"))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }
}
