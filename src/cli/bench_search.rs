//! Paired transport-neutral lexical/hybrid corpus measurements; v1 §9/§14.
use super::{CommitGolden, QualityMetrics, embedding, evaluate_golden_paths, goldens};
use crate::{
    cli,
    config::{self, Config},
    models,
    queue::{self, destination, usage},
    search::{
        dense::routed,
        embedding_query::{Factory, Settings},
        fusion::Mode,
        pipeline, rerank,
    },
    store::{Store, UnitKind, transitions},
};
use anyhow::{Result, bail};
use serde::Serialize;
use std::{
    collections::BTreeSet,
    fs,
    path::PathBuf,
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

/// Explicit query budgets and repeated diagnostic workload; no queue defaults.
#[derive(clap::Args)]
pub(crate) struct Args {
    #[arg(long)]
    root: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// Per-provider daily call ceiling, including usage already charged today.
    #[arg(long, value_parser=clap::value_parser!(i64).range(0..))]
    max_calls: i64,
    /// Per-provider daily token ceiling, including warmups and retries.
    #[arg(long, value_parser=clap::value_parser!(i64).range(0..))]
    max_tokens: i64,
    /// Measured workload repetitions after one full warmup, with alternating mode order.
    #[arg(long, default_value_t=5, value_parser=clap::value_parser!(u32).range(1..=20))]
    repeats: u32,
    /// Retain all bounded queries after degraded warmups; degraded runs still exit nonzero.
    #[arg(long)]
    continue_degraded: bool,
    /// Primary handwritten golden YAML; otherwise use commit-derived queries.
    #[arg(long)]
    golden: Option<PathBuf>,
    /// Keep missing historical expected files in diagnostic denominators and report their paths.
    #[arg(long, requires = "golden", conflicts_with = "golden_reviewed_blake3")]
    allow_missing_expected_files: bool,
    /// Additional held-out, scoped or no-answer cohorts; repeat this option.
    #[arg(long, requires = "golden")]
    extra_golden: Vec<PathBuf>,
    /// Leave unscoped cases unrestricted; explicit golden-case scopes stay intact.
    #[arg(long)]
    all_scopes: bool,
    /// Exclude occurrence kinds before candidate limits; repeat for composite ablations.
    /// Excluding summaries alone does not remove previously generated role contexts.
    #[arg(long, value_parser=["commit", "pr", "history", "module", "summary"])]
    exclude_kind: Vec<String>,
    /// Human-reviewed primary digest; must match the current primary bytes.
    #[arg(long, requires = "golden")]
    golden_reviewed_blake3: Option<String>,
    #[arg(long)]
    model_cache: Option<PathBuf>,
    /// Add a dense-only arm with the same queries, scopes, limits and accounting.
    #[arg(long)]
    dense_only: bool,
    /// Add a third hybrid-plus-rerank arm using the configured scorer.
    #[arg(long)]
    rerank: bool,
    /// Explicit reranker rolling-minute call ceiling, including prior usage.
    #[arg(long, requires="rerank", required_if_eq("rerank", "true"), value_parser=clap::value_parser!(i64).range(0..))]
    rerank_rpm: Option<i64>,
    /// Explicit reranker rolling-minute estimated-token ceiling.
    #[arg(long, requires="rerank", required_if_eq("rerank", "true"), value_parser=clap::value_parser!(i64).range(0..))]
    rerank_tpm: Option<i64>,
    /// Optional separate verified reranker cache; defaults to --model-cache.
    #[arg(long, requires = "rerank")]
    rerank_model_cache: Option<PathBuf>,
    #[arg(long)]
    json: bool,
}
#[derive(Serialize)]
struct Sample {
    elapsed_ms: f64,
    paths: Vec<String>,
    kinds: Vec<String>,
    degraded: bool,
    dense: String,
    lexical: String,
    error: Option<String>,
    evidence: Option<goldens::Evidence>,
    rerank: String,
    rerank_fallback: Option<&'static str>,
    candidate_count: Option<usize>,
    rerank_auto: Option<rerank::auto::Status>,
}
#[derive(Serialize)]
struct Pair {
    repetition: u32,
    golden: usize,
    lexical: Sample,
    hybrid: Sample,
    reranked: Option<Sample>,
    dense_only: Option<Sample>,
}

fn pair_degraded(pair: &Pair) -> bool {
    pair.lexical.degraded
        || pair.hybrid.degraded
        || pair.reranked.as_ref().is_some_and(|sample| sample.degraded)
        || pair
            .dense_only
            .as_ref()
            .is_some_and(|sample| sample.degraded)
}

/// Accepts only the documented first-full-call auto reduction during warmup.
/// Measured repetitions remain strict, and unrelated warmup degradation still
/// stops the benchmark before it can produce misleading scores.
fn expected_auto_warmup_reduction(pair: &Pair) -> bool {
    !pair.lexical.degraded
        && !pair.hybrid.degraded
        && !pair
            .dense_only
            .as_ref()
            .is_some_and(|sample| sample.degraded)
        && pair.reranked.as_ref().is_some_and(|sample| {
            sample.degraded
                && sample.error.is_none()
                && sample.rerank_fallback == Some("rerank_timeout")
                && sample.candidate_count == Some(50)
                && sample.rerank_auto.as_ref().is_some_and(|status| {
                    status.reduced
                        && status.configured_candidates == 50
                        && status.effective_candidates == 20
                })
        })
}
#[derive(Serialize)]
struct Score {
    lexical: Option<QualityMetrics>,
    hybrid: Option<QualityMetrics>,
    reranked: Option<QualityMetrics>,
    dense_only: Option<QualityMetrics>,
    handwritten: Option<HandwrittenScores>,
}
#[derive(Serialize)]
struct HandwrittenScores {
    lexical: goldens::Report,
    hybrid: goldens::Report,
    reranked: Option<goldens::Report>,
    dense_only: Option<goldens::Report>,
}
#[derive(Serialize)]
struct Report {
    format: &'static str,
    state: String,
    root: PathBuf,
    head: String,
    build: cli::bench_metadata::BuildInfo,
    config_blake3: String,
    default_scope: Option<&'static str>,
    exclude_kinds: Vec<String>,
    excluded_occurrences: usize,
    excluded_ids_blake3: Option<String>,
    max_calls: i64,
    max_tokens: i64,
    repeats: u32,
    rerank: bool,
    dense_only: bool,
    rerank_rpm: Option<i64>,
    rerank_tpm: Option<i64>,
    rerank_settings: Option<serde_json::Value>,
    rerank_before: Option<rerank::auto::Status>,
    rerank_after: Option<rerank::auto::Status>,
    warmup_repetitions: u32,
    continue_degraded: bool,
    method: &'static str,
    goldens: Vec<CommitGolden>,
    handwritten: Option<goldens::Dataset>,
    capacity_pauses: CapacityPauses,
    before: queue::inspection::Snapshot,
    after: Option<queue::inspection::Snapshot>,
    pairs: Vec<Pair>,
    scores: Vec<Score>,
    latency: Option<serde_json::Value>,
    error: Option<String>,
}
fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|v| i64::try_from(v.as_secs()).ok())
        .unwrap_or(-1)
}
fn authority(root: &std::path::Path, store: &Store, head: &str, generation: i64) -> Result<()> {
    if super::git_stdout(root, &["rev-parse", "HEAD"])?.trim() != head
        || store.get_meta("indexed_head")?.as_deref() != Some(head)
        || store.l1_generation()? != generation
        || store.counts()?.l1_pending_files != 0
        || store.get_meta("graph_pending")?.as_deref() != Some("0")
    {
        bail!("benchmark requires unchanged HEAD and completed L1 generation")
    }
    Ok(())
}
/// Runs the finite paired workload, retaining partial evidence on failure.
pub(super) fn run(args: Args) -> Result<()> {
    let config_root = fs::canonicalize(&args.root)?;
    let config = config::load(&config_root, None)?.config;
    let root = fs::canonicalize(config_root.join(&config.index.root))?;
    if !cli::database_path(&root, &config).is_file() {
        bail!("search benchmark requires an existing index")
    }
    let store = Arc::new(cli::open_store(&root, &config)?);
    let head = super::git_stdout(&root, &["rev-parse", "HEAD"])?
        .trim()
        .to_owned();
    let generation = store.l1_generation()?;
    authority(&root, &store, &head, generation)?;
    let before = queue::inspection::read(&store)?;
    let paused = capacity_pauses(&store)?;
    let capacity_degraded = paused.jobs != 0;
    let handwritten = args
        .golden
        .as_ref()
        .map(|primary| {
            goldens::load_with_missing_files(
                &root,
                primary,
                &args.extra_golden,
                args.golden_reviewed_blake3.as_deref(),
                args.allow_missing_expected_files,
            )
        })
        .transpose()?;
    if let Some(dataset) = &handwritten {
        goldens::validate_symbols(dataset, &store)?;
    }
    let goldens = if handwritten.is_some() {
        Vec::new()
    } else {
        super::commit_derived_goldens(&root, &config.tests.globs)?
    };
    if goldens.is_empty() && handwritten.is_none() {
        bail!("no commit-derived goldens; no quality denominator")
    }
    let mut report = Report {
        format: "dkc-paired-search-v1",
        state: "running".into(),
        root,
        head,
        build: cli::bench_metadata::build_info()?,
        config_blake3: blake3::hash(toml::to_string(&config)?.as_bytes())
            .to_hex()
            .to_string(),
        capacity_pauses: paused,
        default_scope: (!args.all_scopes).then_some("source"),
        exclude_kinds: args
            .exclude_kind
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        excluded_occurrences: 0,
        excluded_ids_blake3: None,
        max_calls: args.max_calls,
        max_tokens: args.max_tokens,
        repeats: args.repeats,
        rerank: args.rerank,
        dense_only: args.dense_only,
        rerank_rpm: args.rerank_rpm,
        rerank_tpm: args.rerank_tpm,
        rerank_settings: args.rerank.then(|| {
            serde_json::json!({
                "mode":config.rerank.mode,"provider":config.rerank.provider,
                "primary_embedder_profile":config.embedder.profile,
                "remote_model":config.rerank.remote.model,"local_model":config.rerank.local.model,
                "local_gpu_layers":config.rerank.local.gpu_layers,
                "candidates":config.rerank.candidates,"min_score":config.rerank.min_score,
                "max_doc_chars":config.rerank.max_doc_chars,
                "query_timeout_ms":config.limits.query_timeout_ms,
                "rerank_timeout_ms":config.limits.rerank_timeout_ms,
                "diversity_margin":config.search.diversity_margin,
            })
        }),
        rerank_before: None,
        rerank_after: None,
        warmup_repetitions: 1,
        continue_degraded: args.continue_degraded,
        method: if capacity_degraded {
            "in-process-capacity-degraded-diagnostic-v1"
        } else if args.allow_missing_expected_files {
            "in-process-missing-expected-files-diagnostic-v1"
        } else if args.dense_only && args.rerank {
            "in-process-source-scope-four-arm-diagnostic-v1"
        } else if args.dense_only {
            "in-process-dense-only-arm-diagnostic-v1"
        } else if args.all_scopes || !args.exclude_kind.is_empty() {
            "in-process-explicit-candidate-ablation-diagnostic-v1"
        } else if handwritten.is_some() {
            if args.rerank {
                "in-process-handwritten-three-arm-rerank-diagnostic-v1"
            } else {
                "in-process-handwritten-two-arm-diagnostic-v1"
            }
        } else if args.rerank {
            "in-process-source-scope-three-arm-rerank-diagnostic-v1"
        } else {
            "in-process-source-scope-no-rerank-diagnostic-v1"
        },
        goldens,
        handwritten,
        before,
        after: None,
        pairs: Vec::new(),
        scores: Vec::new(),
        latency: None,
        error: None,
    };
    embedding::save(&args.out, &report)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(measure(
        &args,
        &store,
        config_root,
        &config,
        generation,
        &mut report,
    ));
    report.after = Some(queue::inspection::read(&store)?);
    if let Err(error) = result {
        report.state = "failed".into();
        report.error = Some(error.to_string());
    }
    embedding::save(&args.out, &report)?;
    cli::emit(
        args.json,
        serde_json::to_value(&report)?,
        &format!(
            "paired search: {} ({} measured sets)",
            report.state,
            report.scores.len()
        ),
    )?;
    if report.state != "complete" {
        bail!("paired search benchmark {}; report retained", report.state)
    }
    Ok(())
}
async fn factories(
    args: &Args,
    store: &Arc<Store>,
    config_root: &std::path::Path,
    config: &Config,
    state: &transitions::State,
) -> Result<Vec<Arc<Factory>>> {
    let cache = match &args.model_cache {
        Some(p) => p.clone(),
        None => models::registry::cache_root()?,
    };
    let source = [
        UnitKind::Chunk,
        UnitKind::Symbol,
        UnitKind::ConfigKey,
        UnitKind::DocSection,
        UnitKind::Image,
    ];
    let mut seen = BTreeSet::new();
    let mut factories = Vec::new();
    let settings = |kind| Settings {
        config_root: config_root.into(),
        config: config.clone(),
        kind,
        budget: usage::Budget {
            calls: Some(args.max_calls),
            tokens: Some(args.max_tokens),
        },
    };
    for kind in source {
        let Some(table) = state.desired.get(kind.as_str()) else {
            bail!("embedding routes have not been prepared")
        };
        if !seen.insert(table.clone()) {
            continue;
        }
        let resolved = destination::resolve(config, kind)?;
        let query_settings = settings(kind);
        let query_store = Arc::clone(store);
        let factory = if resolved.identity.provider == "embed:local" {
            Factory::new_local(query_store, query_settings, cache.clone()).await?
        } else {
            tokio::task::spawn_blocking(move || Factory::new(query_store, query_settings)).await??
        };
        if factory.table_slug() != *table {
            bail!("configured provider does not match persisted routes")
        }
        factories.push(factory);
    }
    for previous in &state.previous {
        if seen.contains(&previous.table) {
            continue;
        }
        if let Some(kind) = source
            .into_iter()
            .find(|kind| previous.kinds.contains(kind.as_str()))
        {
            factories.push(
                Factory::new_previous(
                    Arc::clone(store),
                    settings(kind),
                    previous.table.clone(),
                    state.generation,
                    cache.clone(),
                )
                .await?,
            );
            seen.insert(previous.table.clone());
        }
    }
    Ok(factories)
}
async fn sample(
    providers: routed::Providers,
    query: &str,
    mode: Mode,
    scope: Option<&str>,
    capture: bool,
    exclude_ids: &[i64],
    scorer: Option<&Arc<rerank::Factory>>,
) -> Sample {
    let store = Arc::clone(&providers.store);
    let start = Instant::now();
    let request = pipeline::Request {
        query: query.into(),
        scope: scope.map(Into::into),
        k: 10,
        mode,
        exclude_ids: exclude_ids.to_vec(),
    };
    let mut candidate_count = None;
    let result = if let Some(scorer) = scorer {
        let settings = providers.config.search.clone();
        match pipeline::candidates_routed_for_rerank(providers, request, scorer, now).await {
            Ok(window) => {
                candidate_count = Some(window.hits.len());
                window
                    .rerank_with(Arc::clone(scorer), now)
                    .await
                    .finish(10, &settings)
            }
            Err(error) => Err(error),
        }
    } else {
        pipeline::search_routed_without_rerank(providers, request, now).await
    };
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let result = result.and_then(|result| {
        let evidence = if capture {
            let response = crate::search::format_response(&store, query, &result.hits)?;
            Some(goldens::Evidence::from_response(&response).map_err(|_| {
                crate::search::Error::InvalidQuery {
                    reason: "invalid golden response",
                }
            })?)
        } else {
            None
        };
        Ok((result, evidence))
    });
    match result {
        Ok((result, evidence)) => Sample {
            evidence,
            elapsed_ms,
            paths: result
                .hits
                .iter()
                .map(|h| h.unit.path.clone().unwrap_or_default())
                .collect(),
            kinds: result.hits.iter().map(|h| h.unit.kind.clone()).collect(),
            degraded: result.degraded(),
            dense: format!("{:?}", result.dense),
            lexical: format!("{:?}", result.lexical),
            error: None,
            rerank: format!("{:?}", result.rerank),
            rerank_fallback: result.rerank.fallback(),
            candidate_count,
            rerank_auto: result.rerank_auto,
        },
        Err(error) => Sample {
            evidence: None,
            elapsed_ms,
            paths: Vec::new(),
            kinds: Vec::new(),
            degraded: true,
            dense: "unknown".into(),
            lexical: "unknown".into(),
            error: Some(error.to_string()),
            rerank: "unknown".into(),
            rerank_fallback: None,
            candidate_count,
            rerank_auto: scorer.map(|host| host.auto_status()),
        },
    }
}
async fn measure(
    args: &Args,
    store: &Arc<Store>,
    config_root: PathBuf,
    config: &Config,
    generation: i64,
    report: &mut Report,
) -> Result<()> {
    let state = store
        .read_snapshot(transitions::read)??
        .ok_or_else(|| anyhow::anyhow!("embedding routes have not been prepared"))?;
    let excluded_ids = excluded_occurrences(store, &report.exclude_kinds)?;
    report.excluded_occurrences = excluded_ids.len();
    report.excluded_ids_blake3 = Some(
        blake3::hash(&serde_json::to_vec(&excluded_ids)?)
            .to_hex()
            .to_string(),
    );
    let prepared = factories(args, store, &config_root, config, &state).await?;
    let scorer = if args.rerank {
        let cache = args
            .rerank_model_cache
            .clone()
            .or_else(|| args.model_cache.clone())
            .map(Ok)
            .unwrap_or_else(models::registry::cache_root)?;
        Some(
            rerank::Factory::new_configured(
                Arc::clone(store),
                rerank::Settings {
                    config_root: config_root.clone(),
                    config: config.clone(),
                    limits: usage::Limits {
                        budget: usage::Budget {
                            calls: Some(args.max_calls),
                            tokens: Some(args.max_tokens),
                        },
                        rpm: Some(
                            args.rerank_rpm
                                .ok_or_else(|| anyhow::anyhow!("missing rerank rpm"))?,
                        ),
                        tpm: Some(
                            args.rerank_tpm
                                .ok_or_else(|| anyhow::anyhow!("missing rerank tpm"))?,
                        ),
                    },
                },
                cache,
            )
            .await?,
        )
    } else {
        None
    };
    report.rerank_before = scorer.as_ref().map(|host| host.auto_status());
    let providers = || routed::Providers {
        store: Arc::clone(store),
        config: config.clone(),
        config_root: config_root.clone(),
        factories: prepared.clone(),
    };
    let queries: Vec<_> = if let Some(dataset) = &report.handwritten {
        dataset
            .cases
            .iter()
            .map(|case| (case.query.clone(), case.scope.clone()))
            .collect()
    } else {
        report
            .goldens
            .iter()
            .map(|case| (case.query.clone(), None))
            .collect()
    };
    let mut measured_degraded = false;
    let mut unexpected_warmup_degradation = false;
    for repetition in 0..=args.repeats {
        let offset = report.pairs.len();
        for (index, (query, scope)) in queries.iter().enumerate() {
            authority(&report.root, store, &report.head, generation)?;
            // Rotate every enabled arm; default two/three-arm order stays unchanged.
            let mut arms = vec![0, 1];
            if scorer.is_some() {
                arms.push(2);
            }
            if args.dense_only {
                arms.push(3);
            }
            let mut samples = [None, None, None, None];
            for step in 0..arms.len() {
                let arm = arms[(index + repetition as usize + step) % arms.len()];
                samples[arm] = Some(
                    sample(
                        providers(),
                        query,
                        match arm {
                            0 => Mode::Lexical,
                            3 => Mode::Dense,
                            _ => Mode::Hybrid,
                        },
                        scope.as_deref().or(report.default_scope),
                        report.handwritten.is_some(),
                        &excluded_ids,
                        if arm == 2 { scorer.as_ref() } else { None },
                    )
                    .await,
                );
            }
            let [lexical, hybrid, reranked, dense_only] = samples;
            let lexical = lexical.ok_or_else(|| anyhow::anyhow!("missing lexical sample"))?;
            let hybrid = hybrid.ok_or_else(|| anyhow::anyhow!("missing hybrid sample"))?;
            let pair = Pair {
                repetition,
                golden: index,
                lexical,
                hybrid,
                reranked,
                dense_only,
            };
            let degraded = pair_degraded(&pair);
            if repetition == 0 {
                unexpected_warmup_degradation |= degraded && !expected_auto_warmup_reduction(&pair);
            } else {
                measured_degraded |= degraded;
            }
            report.pairs.push(pair);
            report.rerank_after = scorer.as_ref().map(|host| host.auto_status());
            embedding::save(&args.out, report)?;
            if repetition == 0 && unexpected_warmup_degradation && !args.continue_degraded {
                bail!("warmup degraded; inspect provider budget, deadline and queue state")
            }
        }
        if repetition == 0 {
            // Continued warmup failures still invalidate completion; admission is unchanged.
            if unexpected_warmup_degradation && !args.continue_degraded {
                bail!("warmup degraded; inspect provider budget, deadline and queue state")
            }
        } else if let Some(dataset) = &report.handwritten {
            let score_arm = |arm: usize| -> Result<goldens::Report> {
                let samples = report.pairs[offset..]
                    .iter()
                    .map(|pair| match arm {
                        0 => Some(&pair.lexical),
                        1 => Some(&pair.hybrid),
                        2 => pair.reranked.as_ref(),
                        _ => pair.dense_only.as_ref(),
                    })
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| anyhow::anyhow!("missing golden arm"))?;
                let responses = samples
                    .iter()
                    .map(|sample| sample.evidence.clone())
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| anyhow::anyhow!("missing golden evidence"))?;
                let values: Vec<_> = samples.iter().map(|sample| sample.elapsed_ms).collect();
                let mut latency = super::summarize_latencies(&values);
                latency.method = report.method.into();
                latency.sets_ms = vec![values];
                goldens::score(dataset, store, &responses, latency)
            };
            report.scores.push(Score {
                lexical: None,
                hybrid: None,
                reranked: None,
                dense_only: None,
                handwritten: Some(HandwrittenScores {
                    lexical: score_arm(0)?,
                    hybrid: score_arm(1)?,
                    reranked: args.rerank.then(|| score_arm(2)).transpose()?,
                    dense_only: args.dense_only.then(|| score_arm(3)).transpose()?,
                }),
            });
        } else {
            let reranked = if args.rerank {
                Some(evaluate_golden_paths(
                    &report.root,
                    &report.goldens,
                    config,
                    store,
                    |i, _| {
                        report.pairs[offset + i]
                            .reranked
                            .as_ref()
                            .map(|s| s.paths.clone())
                            .ok_or_else(|| anyhow::anyhow!("missing reranked sample"))
                    },
                )?)
            } else {
                None
            };
            let dense_only = if args.dense_only {
                Some(evaluate_golden_paths(
                    &report.root,
                    &report.goldens,
                    config,
                    store,
                    |i, _| {
                        report.pairs[offset + i]
                            .dense_only
                            .as_ref()
                            .map(|s| s.paths.clone())
                            .ok_or_else(|| anyhow::anyhow!("missing dense-only sample"))
                    },
                )?)
            } else {
                None
            };
            report.scores.push(Score {
                reranked,
                dense_only,
                handwritten: None,
                lexical: Some(evaluate_golden_paths(
                    &report.root,
                    &report.goldens,
                    config,
                    store,
                    |i, _| Ok(report.pairs[offset + i].lexical.paths.clone()),
                )?),
                hybrid: Some(evaluate_golden_paths(
                    &report.root,
                    &report.goldens,
                    config,
                    store,
                    |i, _| Ok(report.pairs[offset + i].hybrid.paths.clone()),
                )?),
            });
        }
        embedding::save(&args.out, report)?;
        eprintln!(
            "paired search repetition={repetition}/{} goldens={}",
            args.repeats,
            queries.len()
        );
    }
    authority(&report.root, store, &report.head, generation)?;
    if config::load(&config_root, None)?.config != *config
        || store.read_snapshot(transitions::read)??.as_ref() != Some(&state)
    {
        bail!("configuration or embedding routes changed during measurement")
    }
    if capacity_pauses(store)? != report.capacity_pauses {
        bail!("capacity pause cohort changed during measurement")
    }
    if excluded_occurrences(store, &report.exclude_kinds)? != excluded_ids {
        bail!("excluded candidate cohort changed during measurement")
    }
    let after = queue::inspection::read(store)?;
    if serde_json::to_value(&report.before.l2)? != serde_json::to_value(&after.l2)? {
        bail!("vector coverage changed during measurement")
    }
    let latency = |arm: usize| {
        let elapsed = |p: &Pair| match arm {
            0 => Some(p.lexical.elapsed_ms),
            1 => Some(p.hybrid.elapsed_ms),
            2 => p.reranked.as_ref().map(|s| s.elapsed_ms),
            _ => p.dense_only.as_ref().map(|s| s.elapsed_ms),
        };
        let sets = (1..=args.repeats)
            .map(|r| {
                report
                    .pairs
                    .iter()
                    .filter(|p| p.repetition == r)
                    .filter_map(elapsed)
                    .collect()
            })
            .collect();
        super::summarize_latency_sets(sets, None, 0, report.method)
    };
    report.latency = Some(serde_json::json!({"lexical":latency(0),"hybrid":latency(1),
        "reranked":args.rerank.then(||latency(2)),
        "dense_only":args.dense_only.then(||latency(3))}));
    report.state = if measured_degraded || unexpected_warmup_degradation {
        "degraded"
    } else {
        "complete"
    }
    .into();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Pair, Sample, expected_auto_warmup_reduction};
    use crate::search::rerank::auto::Status;

    fn sample(degraded: bool) -> Sample {
        Sample {
            elapsed_ms: 1.0,
            paths: Vec::new(),
            kinds: Vec::new(),
            degraded,
            dense: "Complete".into(),
            lexical: "Complete".into(),
            error: None,
            evidence: None,
            rerank: "Complete".into(),
            rerank_fallback: None,
            candidate_count: None,
            rerank_auto: None,
        }
    }

    fn adaptive_pair() -> Pair {
        let mut reranked = sample(true);
        reranked.rerank = "Timeout".into();
        reranked.rerank_fallback = Some("rerank_timeout");
        reranked.candidate_count = Some(50);
        reranked.rerank_auto = Some(Status {
            configured_candidates: 50,
            effective_candidates: 20,
            reduced: true,
            observed_ms: Some(1_501.0),
            incomplete_sample: true,
            persistence: "pending",
            message: Some(
                "rerank auto: subsequent queries use 20 candidates after a slow 50-candidate call",
            ),
        });
        Pair {
            repetition: 0,
            golden: 0,
            lexical: sample(false),
            hybrid: sample(false),
            reranked: Some(reranked),
            dense_only: Some(sample(false)),
        }
    }

    #[test]
    fn only_documented_auto_reduction_is_allowed_during_warmup() {
        let mut pair = adaptive_pair();
        assert!(expected_auto_warmup_reduction(&pair));
        pair.hybrid.degraded = true;
        assert!(!expected_auto_warmup_reduction(&pair));
        pair = adaptive_pair();
        pair.reranked.as_mut().unwrap().rerank_auto = None;
        assert!(!expected_auto_warmup_reduction(&pair));
    }
}

/// Resolves one immutable benchmark candidate cohort; runtime exclusions expand
/// collapsed chunks and summary descendants before both lexical/dense limits.
fn excluded_occurrences(store: &Store, kinds: &[String]) -> Result<Vec<i64>> {
    if kinds.is_empty() {
        return Ok(Vec::new());
    }
    Ok(store.read_snapshot(|tx| {
        let mut query = tx.prepare(
            "SELECT id FROM units WHERE kind IN (SELECT value FROM json_each(?1)) ORDER BY id",
        )?;
        let ids = query
            .query_map([serde_json::json!(kinds).to_string()], |row| row.get(0))?
            .collect::<Result<Vec<i64>, _>>()?;
        Ok::<_, crate::store::Error>(ids)
    })??)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct CapacityPauses {
    jobs: usize,
    identities_blake3: String,
    reasons: std::collections::BTreeMap<&'static str, usize>,
}

/// Allows only explicit immutable capacity pauses, preserving lower coverage in measurements.
/// Other unfinished states require normal draining/diagnosis before query dispatch.
fn capacity_pauses(store: &Store) -> Result<CapacityPauses> {
    store.read_snapshot(|tx| {
        let mut query = tx.prepare("SELECT id,provider,target_kind,target_id,hex(target_hash),hex(desired_hash),tokens,attempts,not_before,last_error,state FROM jobs WHERE kind='embed' AND state!='done' ORDER BY id")?;
        let mut rows = query.query([])?;
        let mut hasher = blake3::Hasher::new();
        let mut jobs = 0;
        let mut reasons = std::collections::BTreeMap::new();
        while let Some(row) = rows.next()? {
            let state: String = row.get(10)?;
            let not_before: i64 = row.get(8)?;
            if state != "queued" || not_before != i64::MAX {
                bail!("finish pending or failed embeddings before paired search measurement");
            }
            let reason: Option<String> = row.get(9)?;
            let code = match reason.as_deref() {
                Some("max_l2_units cap; waiting for capacity") => "max_l2_units",
                Some("max_l3_units cap; waiting for capacity") => "max_l3_units",
                Some("max_units_per_file cap; waiting for capacity") => "max_units_per_file",
                Some("database 85 percent soft cap; waiting for capacity") => "legacy_database_cap",
                Some("database 95 percent soft cap; waiting for capacity") => "database_cap",
                Some("embedding document or batch token cap; waiting for capacity") => "token_cap",
                Some("SQLite storage full; waiting for capacity") => "storage_full",
                Some("WAL capacity exhausted; waiting for checkpoint or capacity") => "wal_capacity",
                _ => bail!("unknown embedding pause requires diagnosis before paired search measurement"),
            };
            let identity = serde_json::to_vec(&(
                row.get::<_, i64>(0)?, row.get::<_, String>(1)?,
                row.get::<_, String>(2)?, row.get::<_, i64>(3)?,
                row.get::<_, String>(4)?, row.get::<_, String>(5)?,
                row.get::<_, i64>(6)?, row.get::<_, i64>(7)?, not_before, code,
            ))?;
            hasher.update(&(identity.len() as u64).to_le_bytes());
            hasher.update(&identity);
            jobs += 1;
            *reasons.entry(code).or_default() += 1;
        }
        Ok::<_, anyhow::Error>(CapacityPauses {
            jobs, identities_blake3: hasher.finalize().to_hex().to_string(), reasons,
        })
    })?
}
