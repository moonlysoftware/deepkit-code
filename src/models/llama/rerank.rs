//! Verified local Rank inference and smoke/idle lifecycle; v1 §§7.2, 7.4–7.5.
use super::{
    AddBos, Cancellation, LlamaBatch, LlamaContextParams, LlamaModel, LlamaPoolingType,
    MAX_BATCH_ITEMS, Settings, batch_is_full, cancellation, shared_backend,
};
use crate::{
    config,
    models::{
        self, Reranker, registry,
        rerank::{Error, ResponseReason, smoke},
    },
};
use std::{
    num::NonZeroU32,
    path::PathBuf,
    sync::{Arc, Mutex, Weak},
    time::{Duration, Instant},
};

struct Prompt {
    prefix: String,
    middle: String,
    suffix: String,
}
impl Prompt {
    fn new(template: &str, instruct: &str) -> Result<Self, Error> {
        let (prefix, tail) = template.split_once("{query}").ok_or(Error::Configuration)?;
        let (middle, suffix) = tail.split_once("{document}").ok_or(Error::Configuration)?;
        let original = "Given a web search query, retrieve relevant passages that answer the query";
        if template.contains('\0')
            || instruct.contains('\0')
            || !prefix.contains(original)
            || tail.contains("{query}")
            || suffix.contains("{document}")
        {
            return Err(Error::Configuration);
        }
        Ok(Self {
            prefix: prefix.replace(original, instruct),
            middle: middle.into(),
            suffix: suffix.into(),
        })
    }
    fn render(&self, query: &str, doc: &str) -> String {
        format!("{}{query}{}{doc}{}", self.prefix, self.middle, self.suffix)
    }
}
struct Recipe {
    artifact: registry::Model,
    cache: PathBuf,
    settings: Settings,
    prompt: Prompt,
}
/// Initial verified load and fixed-pair smoke evidence; times exclude query retrieval.
#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct LoadReport {
    pub(crate) scores: [f32; 3],
    pub(crate) preparation_ms: f64,
    pub(crate) smoke_ms: f64,
    pub(crate) threads: i32,
    pub(crate) requested_gpu_layers: i32,
}
struct Native {
    model: LlamaModel,
    proof: Option<registry::VerifiedArtifact>,
    report: LoadReport,
}
impl Native {
    fn load(recipe: &Recipe, stop: &Cancellation) -> models::Result<Self> {
        let started = Instant::now();
        stop.check()?;
        let proof = registry::verify_with_identity(&recipe.artifact, &recipe.cache)
            .map_err(|_| super::Error::Artifact)?;
        stop.check()?;
        let model = LlamaModel::load_from_file(
            shared_backend()?,
            recipe.cache.join(&recipe.artifact.file),
            &recipe.settings.model_params(),
        )
        .map_err(|_| Error::Native {
            operation: "model load",
        })?;
        verify(recipe, proof.as_ref())?;
        if model.n_cls_out() != 2
            || model.meta_val_str("qwen3.pooling_type").ok().as_deref() != Some("4")
            || model
                .meta_val_str("tokenizer.chat_template.rerank")
                .ok()
                .as_deref()
                != Some(recipe.artifact.instruct.as_str())
        {
            return Err(Error::Configuration.into());
        }
        let mut native = Self {
            model,
            proof,
            report: LoadReport {
                scores: [0.0; 3],
                preparation_ms: started.elapsed().as_secs_f64() * 1000.0,
                smoke_ms: 0.0,
                threads: recipe.settings.threads,
                requested_gpu_layers: recipe.settings.gpu_layers,
            },
        };
        let started = Instant::now();
        let scores = native.score(recipe, smoke::QUERY, &smoke::DOCUMENTS, stop)?;
        native.report.scores = smoke::validate(&scores, &recipe.artifact.file)?;
        native.report.smoke_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(native)
    }
    fn score(
        &self,
        recipe: &Recipe,
        query: &str,
        docs: &[&str],
        stop: &Cancellation,
    ) -> models::Result<Vec<f32>> {
        verify(recipe, self.proof.as_ref())?;
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let mut documents = Vec::with_capacity(docs.len());
        for doc in docs {
            stop.check()?;
            let prompt = recipe.prompt.render(query, doc);
            let tokens = self
                .model
                .str_to_token(&prompt, AddBos::Always)
                .map_err(|_| Error::Native {
                    operation: "tokenization",
                })?;
            if tokens.is_empty() || tokens.len() > recipe.settings.token_limit() {
                return Err(Error::Input.into());
            }
            documents.push(tokens);
        }
        let maximum_sequences = documents.len().min(MAX_BATCH_ITEMS);
        // Keep one context/batch only for this scoring call. Independent pairs
        // share each decode's token budget and sequence outputs, while the KV
        // cache is cleared between bounded batches.
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(recipe.settings.context_tokens))
            .with_kv_unified(true)
            .with_n_batch(recipe.settings.batch_tokens)
            .with_n_ubatch(recipe.settings.batch_tokens)
            .with_n_seq_max(u32::try_from(maximum_sequences).map_err(|_| Error::Input)?)
            .with_n_threads(recipe.settings.threads)
            .with_n_threads_batch(recipe.settings.threads)
            .with_pooling_type(LlamaPoolingType::Rank)
            .with_embeddings(true);
        stop.check()?;
        let mut context = self
            .model
            .new_context(shared_backend()?, params)
            .map_err(|_| Error::Native {
                operation: "context creation",
            })?;
        let mut batch = LlamaBatch::new(
            recipe.settings.token_limit(),
            i32::try_from(maximum_sequences).map_err(|_| Error::Input)?,
        );
        let mut scores = Vec::with_capacity(documents.len());
        let mut first = 0;
        while first < documents.len() {
            stop.check()?;
            let mut token_count = 0;
            let mut last = first;
            while let Some(tokens) = documents.get(last) {
                if batch_is_full(
                    last - first,
                    token_count,
                    tokens.len(),
                    recipe.settings.token_limit(),
                ) {
                    break;
                }
                token_count += tokens.len();
                last += 1;
            }
            context.clear_kv_cache();
            batch.clear();
            for (sequence, tokens) in documents[first..last].iter().enumerate() {
                let sequence = i32::try_from(sequence).map_err(|_| Error::Input)?;
                for (position, token) in tokens.iter().enumerate() {
                    batch
                        .add(
                            *token,
                            i32::try_from(position).map_err(|_| Error::Input)?,
                            &[sequence],
                            false,
                        )
                        .map_err(|_| Error::Native {
                            operation: "batch preparation",
                        })?;
                }
            }
            stop.check()?;
            context.decode(&mut batch).map_err(|_| Error::Native {
                operation: "Rank decode",
            })?;
            stop.check()?;
            for sequence in 0..(last - first) {
                let probabilities = context
                    .embeddings_seq_ith(i32::try_from(sequence).map_err(|_| Error::Input)?)
                    .map_err(|_| Error::Native {
                        operation: "Rank output",
                    })?;
                scores.push(probability(probabilities)?);
            }
            first = last;
        }
        stop.check()?;
        verify(recipe, self.proof.as_ref())?;
        Ok(scores)
    }
}
fn verify(recipe: &Recipe, proof: Option<&registry::VerifiedArtifact>) -> models::Result<()> {
    if let Some(proof) = proof {
        if !proof.matches(&recipe.artifact, &recipe.cache) {
            return Err(super::Error::Artifact.into());
        }
    } else {
        registry::verify(&recipe.artifact, &recipe.cache).map_err(|_| super::Error::Artifact)?;
    }
    Ok(())
}
fn probability(values: &[f32]) -> Result<f32, Error> {
    if values.len() != 2 {
        return Err(Error::Response(ResponseReason::Count));
    }
    if values
        .iter()
        .any(|s| !s.is_finite() || !(0.0..=1.0).contains(s))
        || (values.iter().sum::<f32>() - 1.0).abs() > 1e-4
    {
        return Err(Error::Response(ResponseReason::Score));
    }
    // The pinned converted classifier orders yes/no and already applies softmax.
    Ok(values[0])
}
struct Resident {
    instance: u64,
    native: Option<Native>,
    recipe: Recipe,
    last_used: Instant,
    disabled: bool,
}
impl Drop for Resident {
    fn drop(&mut self) {
        self.unload();
    }
}

impl Resident {
    // Both idle eviction and final owner destruction report actual release once.
    fn unload(&mut self) {
        if let Some(native) = self.native.take() {
            drop(native);
            super::residency_event("unloaded", "reranker", self.instance, 0);
        }
    }

    fn score(
        &mut self,
        query: &str,
        docs: &[String],
        stop: &Cancellation,
    ) -> models::Result<Vec<f32>> {
        stop.check()?;
        if self.disabled {
            return Err(Error::Smoke {
                file: self.recipe.artifact.file.clone(),
            }
            .into());
        }
        if self.native.is_none() {
            match Native::load(&self.recipe, stop) {
                Ok(native) => {
                    self.native = Some(native);
                    super::residency_event("loaded", "reranker", self.instance, 0);
                }
                Err(error) => {
                    self.disabled = matches!(error, models::Error::Rerank(Error::Smoke { .. }));
                    return Err(error);
                }
            }
        }
        let docs: Vec<_> = docs.iter().map(String::as_str).collect();
        let result = self
            .native
            .as_ref()
            .ok_or(models::Error::LocalWorker)?
            .score(&self.recipe, query, &docs, stop);
        self.last_used = Instant::now();
        if result.is_ok() && !docs.is_empty() {
            super::residency_event("inference_complete", "reranker", self.instance, docs.len());
        }
        result
    }
}
/// Verified reusable local reranker. Loading, smoke checks, reload and native
/// inference run off the async runtime and share one cancellation-safe capacity slot.
pub(crate) struct Local {
    initial_load: LoadReport,
    state: Arc<Mutex<Resident>>,
    gate: Arc<tokio::sync::Semaphore>,
    id: String,
    max_docs: usize,
    max_doc_chars: usize,
}
impl Local {
    /// Returns the successful initial load's smoke result without running it twice.
    pub(crate) fn initial_load_report(&self) -> &LoadReport {
        &self.initial_load
    }

    /// Loads only registered Rank weights, checks their identity/metadata, and
    /// refuses to publish a provider until its fixed-pair smoke gate passes.
    pub(crate) async fn load_registered(
        cache: PathBuf,
        config: config::Rerank,
        idle: Duration,
    ) -> models::Result<Self> {
        let registry = registry::Registry::embedded().map_err(|_| super::Error::Artifact)?;
        let artifact = registry
            .get(&config.local.model)
            .map_err(|_| super::Error::Artifact)?
            .clone();
        let max_docs = usize::try_from(config.candidates).map_err(|_| Error::Configuration)?;
        let max_doc_chars =
            usize::try_from(config.max_doc_chars).map_err(|_| Error::Configuration)?;
        if artifact.pooling != "rank"
            || artifact.dims != 2
            || max_docs == 0
            || max_doc_chars == 0
            || idle.is_zero()
        {
            return Err(Error::Configuration.into());
        }
        let profile = config::LocalEmbedder {
            gpu_layers: config.local.gpu_layers,
            n_ctx: u64::from(artifact.context),
            batch_tokens: u64::from(artifact.context),
            ..Default::default()
        };
        let recipe = Recipe {
            prompt: Prompt::new(&artifact.instruct, &config.instruct)?,
            settings: Settings::from_profile(&profile)?,
            artifact,
            cache,
        };
        let id = format!("llama/{}", config.local.model);
        let gate = Arc::new(tokio::sync::Semaphore::new(1));
        let (state, initial_load) = cancellation::run(Arc::clone(&gate), move |stop| {
            let native = Native::load(&recipe, &stop)?;
            stop.check()?;
            let report = native.report.clone();
            let instance = super::next_resident_instance();
            super::residency_event("loaded", "reranker", instance, 0);
            Ok((
                Arc::new(Mutex::new(Resident {
                    instance,
                    native: Some(native),
                    recipe,
                    last_used: Instant::now(),
                    disabled: false,
                })),
                report,
            ))
        })
        .await?;
        tokio::spawn(unload_idle(Arc::downgrade(&state), idle));
        Ok(Self {
            initial_load,
            state,
            gate,
            id,
            max_docs,
            max_doc_chars,
        })
    }
}
async fn unload_idle(state: Weak<Mutex<Resident>>, idle: Duration) {
    let mut remaining = idle;
    loop {
        tokio::time::sleep(remaining).await;
        let Some(state) = state.upgrade() else { return };
        let outcome = tokio::task::spawn_blocking(move || {
            let mut state = state.lock().ok()?;
            let elapsed = state.last_used.elapsed();
            if elapsed >= idle {
                state.unload();
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
#[async_trait::async_trait]
impl Reranker for Local {
    fn id(&self) -> &str {
        &self.id
    }
    fn max_doc_chars(&self) -> usize {
        self.max_doc_chars
    }
    async fn score(&self, query: &str, docs: &[&str]) -> models::Result<Vec<f32>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        if query.trim().is_empty() || query.contains('\0') || docs.len() > self.max_docs {
            return Err(Error::Input.into());
        }
        let query = query.to_owned();
        let docs = docs
            .iter()
            .map(|doc| doc.chars().take(self.max_doc_chars).collect::<String>())
            .collect::<Vec<_>>();
        // The native wrapper requires C strings. Refuse the entire effective
        // input before reload or inference, rather than scoring a prefix of it.
        if docs.iter().any(|doc| doc.contains('\0')) {
            return Err(Error::Input.into());
        }
        let state = Arc::clone(&self.state);
        cancellation::run(Arc::clone(&self.gate), move |stop| {
            state
                .lock()
                .map_err(|_| models::Error::LocalWorker)?
                .score(&query, &docs, &stop)
        })
        .await
    }
}
#[cfg(test)]
mod tests;
