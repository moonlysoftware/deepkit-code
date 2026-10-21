//! Local classifier contract and explicit verified CPU lifecycle exercise.
use super::*;
#[test]
fn prompt_substitution_never_reinterprets_private_placeholder_text() {
    let registry = registry::Registry::embedded().unwrap();
    let artifact = registry.get("qwen3-reranker-0.6b-q8_0").unwrap();
    let prompt = Prompt::new(&artifact.instruct, "literal {query} and {document}").unwrap();
    let text = prompt.render("query has {document}", "document has {query}");
    assert!(text.contains("<Instruct>: literal {query} and {document}\n<Query>: query has {document}\n<Document>: document has {query}<|im_end|>"));
    assert!(Prompt::new("{query} {query} {document}", "instruct").is_err());
}
#[test]
fn native_rank_probabilities_are_not_softmaxed_twice() {
    assert_eq!(probability(&[0.99, 0.01]).unwrap(), 0.99);
    assert_eq!(probability(&[0.0, 1.0]).unwrap(), 0.0);
    for values in [
        vec![],
        vec![0.9],
        vec![0.9, 0.9],
        vec![-1.0, 2.0],
        vec![f32::NAN, 0.1],
        vec![0.9, 0.1, 0.0],
    ] {
        let expected = if values.len() != 2 {
            ResponseReason::Count
        } else {
            ResponseReason::Score
        };
        assert!(matches!(probability(&values), Err(Error::Response(reason)) if reason == expected));
    }
}
#[tokio::test]
async fn incompatible_or_unverified_artifacts_refuse_before_native_load() {
    let cache = tempfile::tempdir().unwrap();
    let mut config = config::Rerank::default();
    config.local.model = "qwen3-embedding-0.6b-q8_0".into();
    assert!(matches!(
        Local::load_registered(cache.path().into(), config, Duration::from_secs(1)).await,
        Err(models::Error::Rerank(Error::Configuration))
    ));
    let mut config = config::Rerank::default();
    config.local.gpu_layers = 0;
    std::fs::write(
        cache.path().join("Qwen3-Reranker-0.6B-Q8_0.gguf"),
        b"broken",
    )
    .unwrap();
    assert!(matches!(
        Local::load_registered(cache.path().into(), config.clone(), Duration::from_secs(1)).await,
        Err(models::Error::Local(super::super::Error::Artifact))
    ));
    config.max_doc_chars = 0;
    assert!(matches!(
        Local::load_registered(cache.path().into(), config, Duration::from_secs(1)).await,
        Err(models::Error::Rerank(Error::Configuration))
    ));
}
#[tokio::test]
#[ignore = "requires DKC_RERANKER_CACHE with verified pinned weights; explicit CPU exercise"]
async fn verified_cpu_rank_smoke_unload_reload_and_context_refusal()
-> Result<(), Box<dyn std::error::Error>> {
    let cache = std::env::var_os("DKC_RERANKER_CACHE").ok_or("set DKC_RERANKER_CACHE")?;
    let mut config = config::Rerank::default();
    config.local.gpu_layers = 0;
    let provider = Local::load_registered(cache.into(), config, Duration::from_millis(100)).await?;
    let scores = smoke::run(&provider, "Qwen3-Reranker-0.6B-Q8_0.gguf").await?;
    assert_eq!(provider.id(), "llama/qwen3-reranker-0.6b-q8_0");
    assert_eq!(provider.max_doc_chars(), 12000);
    assert!(provider.score("", &[]).await?.is_empty());
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        if provider
            .state
            .try_lock()
            .is_ok_and(|state| state.native.is_none())
        {
            break;
        }
        if tokio::time::Instant::now() >= deadline {
            return Err("idle unload did not finish".into());
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let reloaded = smoke::run(&provider, "Qwen3-Reranker-0.6B-Q8_0.gguf").await?;
    assert!(
        scores
            .iter()
            .zip(reloaded)
            .all(|(before, after)| (*before - after).abs() < 0.002)
    );
    let too_long = "query ".repeat(4000);
    assert!(matches!(
        provider.score(&too_long, &["doc"]).await,
        Err(models::Error::Rerank(Error::Input))
    ));
    println!(
        "verified CPU Rank scores={scores:?}; reload={reloaded:?}; idle unload and oversized context refusal passed"
    );
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "metal"))]
async fn verified_accelerated_rank_fifty_candidate_timings(
    backend: &str,
    maximum_p95_ms: Option<f64>,
) -> Result<f64, Box<dyn std::error::Error>> {
    let cache = std::env::var_os("DKC_RERANKER_CACHE").ok_or("set DKC_RERANKER_CACHE")?;
    // Keep a fixed denominator and bounded total runtime. An external process timeout
    // is still required in case a native driver call cannot return for cancellation.
    tokio::time::timeout(Duration::from_secs(480), async move {
        let mut config = config::Rerank::default();
        config.local.gpu_layers = -1;
        config.candidates = 50;
        let provider = Local::load_registered(cache.into(), config, Duration::from_secs(600)).await?;
        let report = provider.initial_load_report();
        assert_eq!(report.requested_gpu_layers, -1);
        assert!(report.scores[0] > report.scores[1]);
        println!("{backend} Rank initial load: {}", serde_json::to_string(report)?);
        let documents = (0..50)
            .map(|index| smoke::DOCUMENTS[index % smoke::DOCUMENTS.len()])
            .collect::<Vec<_>>();
        let mut samples_ms = Vec::with_capacity(20);
        let mut warmup_ms = 0.0;
        let mut final_scores = Vec::new();
        for repetition in 0..=20 {
            let started = Instant::now();
            let scores = provider.score(smoke::QUERY, &documents).await?;
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(scores.len(), 50);
            assert!(scores.iter().all(|score| score.is_finite() && (0.0..=1.0).contains(score)));
            assert!(scores[0] > scores[1]);
            assert!(elapsed_ms.is_finite());
            if repetition == 0 {
                warmup_ms = elapsed_ms;
            } else {
                samples_ms.push(elapsed_ms);
            }
            // Preserve partial progress if the enclosing timeout ends the diagnostic.
            println!(
                "{backend} Rank repetition={repetition} candidates=50 elapsed_ms={elapsed_ms:.3}"
            );
            final_scores = scores;
        }
        let mut sorted = samples_ms.clone();
        sorted.sort_by(f64::total_cmp);
        let p95_ms = sorted[18];
        println!("{backend} Rank component diagnostic: {}", serde_json::json!({
            "backend": backend,
            "debug_assertions": cfg!(debug_assertions),
            "requested_gpu_layers": -1,
            "dataset": "fixed public smoke documents repeated cyclically to 50; synthetic, not corpus",
            "scope": "prepared Local::score including verification, tokenization and native contexts; no retrieval",
            "candidates": 50,
            "warmup_ms": warmup_ms,
            "samples_ms": samples_ms,
            "p95_nearest_rank_ms": p95_ms,
            "sample_count": 20,
            "final_scores": final_scores,
            "initial_load": report,
        }));
        if let Some(maximum) = maximum_p95_ms.filter(|maximum| p95_ms >= *maximum) {
            return Err(format!(
                "{backend} rerank p95 {p95_ms:.3} ms did not meet < {maximum:.3} ms"
            )
            .into());
        }
        Ok::<f64, Box<dyn std::error::Error>>(p95_ms)
    })
    .await?
}

/// Opt-in synthetic CUDA component diagnostic; retrieval is measured separately.
#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore = "requires DKC_LIVE_CUDA=1, a working GPU and DKC_RERANKER_CACHE"]
async fn verified_cuda_rank_smoke_and_fifty_candidate_timings()
-> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("DKC_LIVE_CUDA").as_deref() != Ok("1") {
        return Err("set DKC_LIVE_CUDA=1 to explicitly enable native GPU inference".into());
    }
    verified_accelerated_rank_fifty_candidate_timings("CUDA", None).await?;
    Ok(())
}

/// Release-gated Metal component measurement for exactly 50 candidates.
#[cfg(feature = "metal")]
#[tokio::test]
#[ignore = "requires DKC_LIVE_METAL=1 and verified DKC_RERANKER_CACHE"]
async fn verified_metal_rank_fifty_candidate_p95() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("DKC_LIVE_METAL").as_deref() != Ok("1") {
        return Err("set DKC_LIVE_METAL=1 to explicitly enable native Metal inference".into());
    }
    verified_accelerated_rank_fifty_candidate_timings("Metal", Some(400.0)).await?;
    Ok(())
}

#[cfg(feature = "metal")]
fn current_macos_rss_bytes() -> Result<u64, Box<dyn std::error::Error>> {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()?;
    if !output.status.success() {
        return Err("ps could not read the current test process RSS".into());
    }
    let kib = String::from_utf8(output.stdout)?.trim().parse::<u64>()?;
    Ok(kib.saturating_mul(1024))
}

/// Loads both local models through Metal, observes idle unload, and gates whole-process RSS.
#[cfg(feature = "metal")]
#[tokio::test]
#[ignore = "requires DKC_LIVE_METAL=1 and both verified models in DKC_RERANKER_CACHE"]
async fn verified_metal_idle_rss_after_both_models_unload() -> Result<(), Box<dyn std::error::Error>>
{
    use crate::models::Embedder;

    if std::env::var("DKC_LIVE_METAL").as_deref() != Ok("1") {
        return Err("set DKC_LIVE_METAL=1 to explicitly enable native Metal inference".into());
    }
    let cache = std::path::PathBuf::from(
        std::env::var_os("DKC_RERANKER_CACHE").ok_or("set DKC_RERANKER_CACHE")?,
    );
    let idle = Duration::from_secs(1);
    let embedder = super::super::LocalProvider::load_registered(
        cache.clone(),
        crate::config::LocalEmbedder {
            gpu_layers: -1,
            ..Default::default()
        },
        1_024,
        idle,
    )
    .await?;
    let mut rerank_config = config::Rerank::default();
    rerank_config.local.gpu_layers = -1;
    rerank_config.candidates = 50;
    let reranker = Local::load_registered(cache, rerank_config, idle).await?;
    assert_eq!(embedder.embed_query(smoke::QUERY).await?.len(), 1_024);
    assert_eq!(
        reranker.score(smoke::QUERY, &smoke::DOCUMENTS).await?.len(),
        3
    );
    let loaded_rss_bytes = current_macos_rss_bytes()?;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    loop {
        let embedding_unloaded = embedder.resident_for_test() == Some(false);
        let reranker_unloaded = reranker
            .state
            .try_lock()
            .is_ok_and(|state| state.native.is_none());
        if embedding_unloaded && reranker_unloaded {
            break;
        }
        if tokio::time::Instant::now() >= deadline {
            return Err("both Metal models did not unload before the RSS deadline".into());
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    tokio::time::sleep(Duration::from_millis(250)).await;
    let idle_rss_bytes = current_macos_rss_bytes()?;
    println!(
        "Metal idle RSS diagnostic: {}",
        serde_json::json!({
            "loaded_rss_bytes": loaded_rss_bytes,
            "idle_rss_bytes": idle_rss_bytes,
            "maximum_idle_rss_bytes": 300_000_000_u64,
            "embedding_unloaded": true,
            "reranker_unloaded": true,
        })
    );
    assert!(
        idle_rss_bytes < 300_000_000,
        "idle RSS {idle_rss_bytes} bytes did not meet <300 MB"
    );
    Ok(())
}

#[tokio::test]
#[ignore = "requires verified DKC_RERANKER_CACHE; explicit native context-reuse parity exercise"]
async fn reused_rank_context_matches_fresh_context_for_each_pair()
-> Result<(), Box<dyn std::error::Error>> {
    let cache = std::env::var_os("DKC_RERANKER_CACHE").ok_or("set DKC_RERANKER_CACHE")?;
    let mut config = config::Rerank::default();
    config.local.gpu_layers = if cfg!(feature = "cuda") { -1 } else { 0 };
    let provider = Local::load_registered(cache.into(), config, Duration::from_secs(600)).await?;
    let longer = format!("{} {}", smoke::DOCUMENTS[1], smoke::DOCUMENTS[1]);
    // Construct a valid prompt within two tokens of the actual context limit.
    // It verifies that a near-limit sequence does not contaminate its neighbors.
    let near_limit = {
        let state = provider
            .state
            .lock()
            .map_err(|_| "resident lock poisoned")?;
        let native = state.native.as_ref().ok_or("resident missing")?;
        let count = |doc: &str| -> Result<usize, Box<dyn std::error::Error>> {
            Ok(native
                .model
                .str_to_token(
                    &state.recipe.prompt.render(smoke::QUERY, doc),
                    AddBos::Always,
                )?
                .len())
        };
        let limit = state.recipe.settings.token_limit();
        let (mut low, mut high) = (0, limit * 2);
        while low + 1 < high {
            let middle = low + (high - low) / 2;
            if count(&" x".repeat(middle))? <= limit {
                low = middle;
            } else {
                high = middle;
            }
        }
        let document = " x".repeat(low);
        assert!(count(&document)? >= limit - 2);
        assert!(document.chars().count() <= provider.max_doc_chars());
        document
    };
    let docs = [
        smoke::DOCUMENTS[0],
        longer.as_str(),
        smoke::DOCUMENTS[2],
        near_limit.as_str(),
        smoke::DOCUMENTS[0],
        smoke::DOCUMENTS[2],
    ];
    let together = provider.score(smoke::QUERY, &docs).await?;
    assert_eq!(together.len(), docs.len());
    let mut fresh = Vec::new();
    for doc in docs {
        let score = provider.score(smoke::QUERY, &[doc]).await?;
        assert_eq!(score.len(), 1);
        fresh.push(score[0]);
    }
    let reversed_docs: Vec<_> = docs.iter().rev().copied().collect();
    let reversed = provider.score(smoke::QUERY, &reversed_docs).await?;
    assert_eq!(reversed.len(), docs.len());
    for ((reused, single), backwards) in together.iter().zip(&fresh).zip(reversed.iter().rev()) {
        assert!(reused.is_finite() && (0.0..=1.0).contains(reused));
        assert!((reused - single).abs() < 0.002);
        assert!((reused - backwards).abs() < 0.002);
    }
    assert!((together[0] - together[4]).abs() < 0.002);
    assert!((together[2] - together[5]).abs() < 0.002);
    for a in 0..docs.len() {
        for b in 0..docs.len() {
            // Scores indistinguishable at the probability tolerance may tie.
            if (fresh[a] - fresh[b]).abs() > 0.004 {
                assert_eq!(together[a] > together[b], fresh[a] > fresh[b]);
            }
        }
    }
    println!("Rank reused scores={together:?}; individual={fresh:?}; reversed={reversed:?}");
    println!(
        "Rank reused/fresh context parity: pairs={}, requested_cuda={}",
        docs.len(),
        cfg!(feature = "cuda")
    );
    Ok(())
}

#[tokio::test]
async fn nul_in_effective_rank_input_refuses_before_native_reload() {
    let cache = tempfile::tempdir().unwrap();
    let registry = registry::Registry::embedded().unwrap();
    let artifact = registry.get("qwen3-reranker-0.6b-q8_0").unwrap().clone();
    assert!(Prompt::new(&artifact.instruct, "bad\0instruction").is_err());
    let settings = Settings::from_profile(&config::LocalEmbedder {
        gpu_layers: 0,
        ..Default::default()
    })
    .unwrap();
    let provider = Local {
        initial_load: LoadReport {
            scores: [0.0; 3],
            preparation_ms: 0.0,
            smoke_ms: 0.0,
            threads: settings.threads,
            requested_gpu_layers: 0,
        },
        state: Arc::new(Mutex::new(Resident {
            instance: 0,
            native: None,
            last_used: Instant::now(),
            disabled: false,
            recipe: Recipe {
                prompt: Prompt::new(&artifact.instruct, "code search").unwrap(),
                artifact,
                settings,
                cache: cache.path().into(),
            },
        })),
        gate: Arc::new(tokio::sync::Semaphore::new(1)),
        id: "unloaded-test-provider".into(),
        max_docs: 50,
        max_doc_chars: 3,
    };
    for (query, docs) in [
        ("query\0suffix", vec!["ok"]),
        ("query", vec!["ok", "é\0x"]),
        ("query", {
            let mut docs = vec!["ok"; 49];
            docs.push("\0");
            docs
        }),
    ] {
        assert!(matches!(
            provider.score(query, &docs).await,
            Err(models::Error::Rerank(Error::Input))
        ));
    }
    // A NUL beyond the character cap is not part of the effective prompt.
    // This valid input reaches artifact verification, without any cached weights.
    assert!(matches!(
        provider.score("query", &["éab\0"]).await,
        Err(models::Error::Local(super::super::Error::Artifact))
    ));
    assert!(provider.state.lock().unwrap().native.is_none());
}
