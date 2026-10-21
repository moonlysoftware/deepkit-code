//! Benchmark harness and regression comparison; see `v1.md` §14.

use sha2::{Digest, Sha256};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    fs::File,
    path::{Path, PathBuf},
    process::{Command as ProcessCommand, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use clap::Subcommand;
use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{emit, timed_response};
#[path = "bench_embed.rs"]
mod embedding;
#[path = "bench_freshness.rs"]
mod freshness;
#[path = "bench_goldens.rs"]
mod goldens;
#[path = "bench_mcp.rs"]
mod mcp;
#[path = "bench_search.rs"]
mod paired_search;
#[path = "bench_rerank.rs"]
mod standalone_rerank;

#[path = "bench_pr_ranking.rs"]
mod pr_ranking;

const GOLDEN_LIMIT: usize = 200;
const TOUCH_FILE_LIMIT: usize = 20;
const PARSE_FAILURE_PATH_LIMIT: usize = 256;
const MAP_SAMPLE_COUNT: usize = 5;
const TOOL_SAMPLE_COUNT: usize = 20;
const PAIRED_SEARCH_REPEATS: u32 = 3;
const BENCH_MAX_CALLS: i64 = 100_000;
const BENCH_MAX_TOKENS: i64 = 100_000_000;
const BENCH_RERANK_RPM: i64 = 10_000;
const BENCH_RERANK_TPM: i64 = 100_000_000;
const MODEL_DRAIN_TIMEOUT: Duration = Duration::from_secs(10 * 60);
const MODEL_DRAIN_PASSES: u32 = 100;
const MODEL_DRAIN_INVOCATIONS: usize = 32;

#[derive(Clone, Copy)]
struct RunMode<'a> {
    dry_run: bool,
    profile: Option<&'a str>,
}

/// Benchmark subcommands available from `dkc bench`.
#[derive(Subcommand)]
pub(crate) enum BenchCommand {
    /// Internal model-free MCP child used by the benchmark harness.
    #[command(hide = true)]
    McpServer,
    /// Reports executing build identity without opening an index or invoking models.
    Metadata {
        /// Prints the structured build identity.
        #[arg(long)]
        json: bool,
    },
    /// Captures exact benchmark configuration on a disposable repository, without indexing.
    CaptureConfig {
        /// TOML corpus manifest containing the selected target.
        #[arg(long, default_value = "bench/corpus.toml")]
        corpus: PathBuf,
        /// One repository name in the corpus manifest.
        #[arg(long)]
        target: String,
        /// Matrix lane profile whose setup must be reproduced.
        #[arg(long, value_parser = ["l1", "local_cuda", "local_cpu", "moonly", "ollama", "dual_local_code_moonly_docs"])]
        profile: String,
        /// Prints the structured, nonsecret preparation result.
        #[arg(long)]
        json: bool,
    },
    /// Runs a bounded embedding exercise against an existing L1 index.
    Embed(embedding::Args),
    /// Compares lexical and hybrid search on commit-derived goldens.
    Search(paired_search::Args),
    /// Compares fixed PR rank weights on one shared direct lexical candidate pool.
    PrRanking(pr_ranking::Args),
    /// Measures the admitted scorer on exactly fifty fixed corpus candidates.
    Rerank(standalone_rerank::Args),
    /// Runs the deterministic lexical benchmark for selected manifest targets.
    Run {
        /// TOML corpus manifest containing `[[repo]]` entries.
        #[arg(long, default_value = "bench/corpus.toml")]
        corpus: PathBuf,
        /// Comma-separated repository names to run; omitted means every target.
        #[arg(long, value_delimiter = ',')]
        only: Vec<String>,
        /// Comma-separated manifest groups to run; combines with `--only`.
        #[arg(long, value_delimiter = ',')]
        groups: Vec<String>,
        /// Validates selection and lists targets without opening repositories.
        #[arg(long)]
        dry_run: bool,
        /// Model recipe for the higher-level pass; omitted keeps the L1-only baseline.
        #[arg(long, value_parser = ["local", "moonly", "cpu", "ollama", "dual"])]
        profile: Option<String>,
        /// Writes the JSON report here and a sibling Markdown table beside it.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Leaves each target's database in place for inspection.
        #[arg(long)]
        keep: bool,
        /// Prints a compact structured result instead of the Markdown table.
        #[arg(long)]
        json: bool,
    },
    /// Measures an existing index without cleaning or reindexing it (normal store migrations apply).
    Probe {
        /// Repository with an already populated index.
        #[arg(long)]
        root: PathBuf,
        /// Writes the structured probe report to this path.
        #[arg(long)]
        out: PathBuf,
        /// Prints the structured report.
        #[arg(long)]
        json: bool,
    },
    /// Compares two JSON reports against the documented regression thresholds.
    Diff {
        baseline: PathBuf,
        candidate: PathBuf,
        /// Exits non-zero when any documented threshold regresses.
        #[arg(long)]
        fail_on_regression: bool,
        /// Prints the structured comparison instead of a Markdown table.
        #[arg(long)]
        json: bool,
    },
}

/// Dispatches a benchmark command after Clap parsed the parent CLI.
pub(crate) fn run(command: BenchCommand) -> Result<()> {
    match command {
        BenchCommand::McpServer => {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            let result = runtime.block_on(crate::serve::benchmark_stdio(env::current_dir()?));
            runtime.shutdown_timeout(Duration::from_secs(1));
            result?;
            Ok(())
        }
        BenchCommand::Metadata { json } => {
            let build = super::bench_metadata::build_info()?;
            let summary = format!("Build {} ({})", build.commit, build.profile);
            emit(json, json!({"build": build}), &summary)?;
            Ok(())
        }
        BenchCommand::CaptureConfig {
            corpus,
            target,
            profile,
            json,
        } => capture_config(&corpus, &target, &profile, json),
        BenchCommand::Embed(args) => embedding::run(args),
        BenchCommand::Search(args) => paired_search::run(args),
        BenchCommand::PrRanking(args) => pr_ranking::run(args),
        BenchCommand::Rerank(args) => standalone_rerank::run(args),
        BenchCommand::Run {
            corpus,
            only,
            groups,
            dry_run,
            profile,
            out,
            keep,
            json,
        } => run_benchmark(
            &corpus,
            &only,
            &groups,
            RunMode {
                dry_run,
                profile: profile.as_deref(),
            },
            out.as_deref(),
            keep,
            json,
        ),
        BenchCommand::Probe { root, out, json } => probe_existing(&root, &out, json),
        BenchCommand::Diff {
            baseline,
            candidate,
            fail_on_regression,
            json,
        } => diff_benchmarks(&baseline, &candidate, fail_on_regression, json),
    }
}

#[derive(Debug, Deserialize)]
struct CorpusManifest {
    repo: Vec<CorpusTarget>,
}

#[derive(Clone, Debug, Deserialize)]
struct CorpusTarget {
    name: String,
    url: String,
    #[serde(default)]
    rev: String,
    #[serde(default)]
    config: Option<PathBuf>,
    #[serde(default)]
    golden: Option<PathBuf>,
    #[serde(default)]
    extra_goldens: Vec<PathBuf>,
    #[serde(default)]
    golden_reviewed_blake3: Option<String>,
    #[serde(default)]
    path: Option<PathBuf>,
    #[serde(default)]
    groups: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchReport {
    schema_version: u32,
    created_at: u64,
    dkc_commit: String,
    #[serde(default)]
    profile: String,
    #[serde(default)]
    build: Option<super::bench_metadata::BuildInfo>,
    machine: MachineInfo,
    targets: Vec<TargetReport>,
    #[serde(default)]
    failures: Vec<TargetFailure>,
}

/// A failed target keeps its stage and completed measurements without fabricated metrics.
#[derive(Debug, Serialize, Deserialize)]
struct TargetFailure {
    name: String,
    outcome: String,
    stage: String,
    error: String,
    exit_code: Option<i32>,
    panics: Option<u64>,
    partial: BTreeMap<String, Value>,
}

struct TargetProgress {
    stage: &'static str,
    measurements: BTreeMap<String, Value>,
}

#[derive(Debug, thiserror::Error)]
#[error("dkc {command} failed (exit {exit_code:?}): {stderr}")]
struct ChildFailure {
    command: String,
    exit_code: Option<i32>,
    stderr: String,
    panics: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct MachineInfo {
    os: String,
    arch: String,
    #[serde(default)]
    preflight: Option<super::bench_metadata::Preflight>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TargetReport {
    name: String,
    root: String,
    url: String,
    rev: String,
    config: Option<String>,
    golden: Option<String>,
    index: IndexMetrics,
    #[serde(default)]
    model_index: Option<ModelIndexMetrics>,
    #[serde(default)]
    paired_search: Option<Value>,
    #[serde(default)]
    composites: Option<CompositeMetrics>,
    size: SizeMetrics,
    quality: QualityMetrics,
    queries: QueryMetrics,
    #[serde(default)]
    freshness: Option<freshness::Report>,
    robustness: RobustnessMetrics,
    #[serde(default)]
    status: Option<Value>,
    #[serde(default)]
    provenance: Option<TargetProvenance>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelIndexMetrics {
    profile: String,
    requested_level: u8,
    elapsed_ms: f64,
    peak_rss_bytes: Option<u64>,
    published_units: u64,
    throughput_units_per_second: Option<f64>,
    embedding_state: String,
    pending_jobs: Option<i64>,
    #[serde(default)]
    terminal_provider_pending: BTreeMap<String, i64>,
    coverage: Value,
    index_result: Value,
    #[serde(default)]
    drain_results: Vec<Value>,
}

struct QueueState {
    unfinished: i64,
    ready: i64,
    leased: i64,
    next_ready_at: Option<i64>,
    observed_at: i64,
    embedding_pending: BTreeMap<String, i64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CompositeMetrics {
    pr_reconstruction: PrReconstructionMetrics,
}

#[derive(Debug, Serialize, Deserialize)]
struct PrReconstructionMetrics {
    detected: u64,
    reconstructed: u64,
    rate: Option<f64>,
    deferred: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct TargetProvenance {
    head: String,
    effective_config_blake3: String,
    config_files_blake3: BTreeMap<String, String>,
    indexed_sources_blake3: String,
    schema_version: Option<String>,
    page_size: i64,
    mmap_size: i64,
    model_usage: Value,
    pre_index: super::bench_metadata::Preflight,
}

#[derive(Debug, Serialize, Deserialize)]
struct IndexMetrics {
    cold_ms: f64,
    #[serde(default)]
    cold_index_result: Option<Value>,
    #[serde(default)]
    mtime_only_index_result: Option<Value>,
    incremental_ms: f64,
    peak_rss_bytes: Option<u64>,
    touched_files: usize,
    #[serde(default)]
    content_edit: Option<ContentEditMetrics>,
    files: u64,
    units: u64,
    symbols: u64,
    edges: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ContentEditMetrics {
    method: String,
    paths: Vec<String>,
    elapsed_ms: f64,
    peak_rss_bytes: Option<u64>,
    index_result: Value,
    restored: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct SizeMetrics {
    db_bytes: u64,
    checkout_bytes: u64,
    ratio: Option<f64>,
    #[serde(default)]
    accounting: Option<FootprintAccounting>,
}

#[derive(Debug, Serialize, Deserialize)]
struct FootprintAccounting {
    method: String,
    admitted_text_files: usize,
    walker_files: usize,
    walker_complete: bool,
    excluded: BTreeMap<String, usize>,
    wal_bytes: u64,
    total_db_bytes: u64,
}

struct EligibleText {
    paths: Vec<PathBuf>,
    bytes: u64,
    accounting: FootprintAccounting,
    parse_failures_by_language: BTreeMap<String, ParseLanguageMetrics>,
}

#[derive(Debug, Serialize, Deserialize)]
struct QualityMetrics {
    commit_derived: GoldenMetrics,
    #[serde(default)]
    eligibility: Option<GoldenEligibility>,
    #[serde(default)]
    handwritten: Option<goldens::Report>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GoldenEligibility {
    method: String,
    tests_globs: Vec<String>,
    eligible: GoldenMetrics,
    raw_expected_paths: usize,
    eligible_expected_paths: usize,
    rejected_paths: Vec<RejectedPath>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RejectedPath {
    revision: String,
    path: String,
    reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct GoldenMetrics {
    count: usize,
    hit_file_at_5: Option<f64>,
    recall_at_10: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct QueryMetrics {
    #[serde(default)]
    resident_memory: Option<ResidentMemory>,
    search: LatencyMetrics,
    map: LatencyMetrics,
    #[serde(default)]
    open: Option<ToolLatencyMetrics>,
    #[serde(default)]
    symbol: Option<ToolLatencyMetrics>,
    #[serde(default)]
    refs: Option<ToolLatencyMetrics>,
    #[serde(default)]
    recent: Option<ToolLatencyMetrics>,
    #[serde(default)]
    unavailable: BTreeMap<String, String>,
    #[serde(default)]
    cli_search: Option<LatencyMetrics>,
    #[serde(default)]
    cli_shared_results_match: Option<bool>,
    #[serde(default)]
    cli_median_overhead_ms: Option<f64>,
    #[serde(default)]
    mcp_search: Option<mcp::Report>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ToolLatencyMetrics {
    target: String,
    latency: LatencyMetrics,
}

/// Whole-process resident snapshots while the shared reader remains alive, not mmap allowances.
#[derive(Debug, Serialize, Deserialize)]
struct ResidentMemory {
    method: String,
    before_open_bytes: Option<u64>,
    after_open_bytes: Option<u64>,
    after_search_bytes: Option<u64>,
    after_map_bytes: Option<u64>,
    page_size: i64,
    mmap_size: i64,
    cache_size: i64,
}

#[derive(Debug, Serialize, Deserialize)]
struct LatencyMetrics {
    samples: usize,
    #[serde(default)]
    cold_ms: Option<f64>,
    p50_ms: Option<f64>,
    p95_ms: Option<f64>,
    #[serde(default)]
    method: String,
    #[serde(default)]
    warmups_per_set: usize,
    #[serde(default)]
    sets_ms: Vec<Vec<f64>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RobustnessMetrics {
    panics: u64,
    dead_jobs: u64,
    #[serde(default)]
    job_states: Option<BTreeMap<String, i64>>,
    #[serde(default)]
    cap_outcomes: Option<Value>,
    #[serde(default)]
    parse_failures_by_language: Option<BTreeMap<String, ParseLanguageMetrics>>,
    #[serde(default)]
    extraction_failures: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ParseLanguageMetrics {
    files: u64,
    failures: u64,
    failure_rate: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    failure_paths: Option<BTreeSet<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    failure_paths_omitted: Option<u64>,
}

impl ParseLanguageMetrics {
    fn observe(&mut self, path: &str, parse_error: bool) {
        self.files += 1;
        if !parse_error {
            return;
        }
        self.failures += 1;
        let paths = self.failure_paths.get_or_insert_with(BTreeSet::new);
        let candidate = path.to_owned();
        if paths.len() < PARSE_FAILURE_PATH_LIMIT {
            paths.insert(candidate);
        } else if paths
            .last()
            .is_some_and(|last| candidate.as_str() < last.as_str())
        {
            paths.pop_last();
            paths.insert(candidate);
        }
        self.failure_paths_omitted = Some(self.failures.saturating_sub(paths.len() as u64));
    }
}

#[derive(Debug, Serialize)]
struct CommitGolden {
    revision: String,
    query: String,
    expected_files: BTreeSet<String>,
}

fn run_benchmark(
    corpus_path: &Path,
    only: &[String],
    groups: &[String],
    mode: RunMode<'_>,
    out: Option<&Path>,
    keep: bool,
    json_output: bool,
) -> Result<()> {
    let RunMode { dry_run, profile } = mode;
    let corpus_path = canonical_or_current(corpus_path)?;
    let manifest_dir = corpus_path
        .parent()
        .ok_or_else(|| anyhow!("corpus manifest has no parent directory"))?;
    let manifest_text = fs::read_to_string(&corpus_path)
        .with_context(|| format!("failed to read corpus manifest {}", corpus_path.display()))?;
    let manifest: CorpusManifest = toml::from_str(&manifest_text)
        .with_context(|| format!("failed to parse corpus manifest {}", corpus_path.display()))?;
    if manifest.repo.is_empty() {
        bail!("corpus manifest contains no [[repo]] targets");
    }
    let names = manifest
        .repo
        .iter()
        .map(|target| target.name.as_str())
        .collect::<BTreeSet<_>>();
    if names.len() != manifest.repo.len() {
        bail!("corpus manifest contains duplicate target names");
    }
    let requested = only.iter().cloned().collect::<BTreeSet<_>>();
    let requested_groups = groups.iter().cloned().collect::<BTreeSet<_>>();
    let available_groups = manifest
        .repo
        .iter()
        .flat_map(|target| target.groups.iter().cloned())
        .collect::<BTreeSet<_>>();
    let missing_groups = requested_groups
        .difference(&available_groups)
        .cloned()
        .collect::<Vec<_>>();
    if !missing_groups.is_empty() {
        bail!(
            "--groups values absent from manifest: {}",
            missing_groups.join(", ")
        );
    }
    let selected = manifest
        .repo
        .into_iter()
        .filter(|target| {
            (requested.is_empty() || requested.contains(&target.name))
                && (requested_groups.is_empty()
                    || target
                        .groups
                        .iter()
                        .any(|group| requested_groups.contains(group)))
        })
        .collect::<Vec<_>>();
    if selected.is_empty() {
        bail!(
            "--only did not match any target in {}",
            corpus_path.display()
        );
    }
    let found = selected
        .iter()
        .map(|target| target.name.clone())
        .collect::<BTreeSet<_>>();
    let missing = requested.difference(&found).cloned().collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!(
            "--only targets absent from manifest: {}",
            missing.join(", ")
        );
    }
    if dry_run {
        let targets = selected
            .iter()
            .map(|target| {
                json!({
                    "name": target.name,
                    "url": target.url,
                    "rev": target.rev,
                    "groups": target.groups,
                    "config": target.config,
                })
            })
            .collect::<Vec<_>>();
        let text = targets
            .iter()
            .filter_map(|target| target["name"].as_str())
            .collect::<Vec<_>>()
            .join("\n");
        emit(
            json_output,
            json!({
                "dry_run": true,
                "profile": profile.unwrap_or("l1"),
                "target_count": targets.len(),
                "targets": targets,
            }),
            &text,
        )?;
        return Ok(());
    }

    let created_at = unix_timestamp();
    let machine = MachineInfo {
        os: env::consts::OS.to_owned(),
        arch: env::consts::ARCH.to_owned(),
        preflight: Some(super::bench_metadata::preflight(matches!(
            profile,
            Some("local" | "dual" | "ollama")
        ))),
    };
    let build = super::bench_metadata::build_info()?;
    let mut targets = Vec::new();
    let mut failures = Vec::new();
    for target in &selected {
        let mut progress = TargetProgress {
            stage: "resolve_target",
            measurements: BTreeMap::new(),
        };
        match run_target(target, manifest_dir, profile, keep, &mut progress) {
            Ok(report) => targets.push(report),
            Err(error) => {
                let child = error.downcast_ref::<ChildFailure>();
                failures.push(TargetFailure {
                    name: target.name.clone(),
                    outcome: "failed".to_owned(),
                    stage: progress.stage.to_owned(),
                    error: format!("{error:#}"),
                    exit_code: child.and_then(|child| child.exit_code),
                    panics: child.map(|child| child.panics),
                    partial: progress.measurements,
                });
            }
        }
    }
    let report = BenchReport {
        schema_version: 3,
        created_at,
        dkc_commit: build.commit.clone(),
        profile: profile.unwrap_or("l1").to_owned(),
        build: Some(build),
        machine,
        targets,
        failures,
    };
    let output_path = out.map_or_else(default_output_path, Path::to_path_buf);
    write_report(&output_path, &report)?;
    let markdown = render_markdown(&report);
    let markdown_path = output_path.with_extension("md");
    fs::write(&markdown_path, &markdown)
        .with_context(|| format!("failed to write {}", markdown_path.display()))?;
    emit(
        json_output,
        json!({"out": output_path, "markdown": markdown_path, "targets": selected.len(), "succeeded": report.targets.len(), "failed": report.failures.len()}),
        &markdown,
    )?;
    if !report.failures.is_empty() {
        bail!(
            "{} benchmark target(s) failed; partial report written to {}",
            report.failures.len(),
            output_path.display()
        );
    }
    Ok(())
}

fn queue_count(value: &Value, field: &str) -> Result<i64> {
    value[field]
        .as_i64()
        .filter(|count| *count >= 0)
        .ok_or_else(|| anyhow!("queue status has invalid {field} count"))
}

fn queue_state(status: &Value) -> Result<QueueState> {
    let jobs = status["jobs"]
        .as_object()
        .ok_or_else(|| anyhow!("queue status is missing job counts"))?;
    let jobs = Value::Object(jobs.clone());
    let queued = queue_count(&jobs, "queued")?;
    let leased = queue_count(&jobs, "leased")?;
    let failed = queue_count(&jobs, "failed")?;
    let unfinished = queued
        .checked_add(leased)
        .and_then(|count| count.checked_add(failed))
        .ok_or_else(|| anyhow!("queue status unfinished count overflow"))?;
    let ready = queue_count(&jobs, "ready")?;
    let observed_at = status["observed_at"]
        .as_i64()
        .filter(|value| *value >= 0)
        .ok_or_else(|| anyhow!("queue status has invalid observation time"))?;
    let next_ready_at = match jobs["next_ready_at"].as_i64() {
        Some(value) if value >= 0 => Some(value),
        Some(_) => return Err(anyhow!("queue status has invalid next-ready time")),
        None if jobs["next_ready_at"].is_null() => None,
        None => return Err(anyhow!("queue status has invalid next-ready time")),
    };
    let providers = status["providers"]
        .as_object()
        .ok_or_else(|| anyhow!("queue status is missing provider counts"))?;
    let mut embedding_pending = BTreeMap::new();
    for (provider, value) in providers {
        if !provider.starts_with("embed:") {
            continue;
        }
        let provider_jobs = &value["jobs"];
        let provider_queued = queue_count(provider_jobs, "queued")?;
        let provider_leased = queue_count(provider_jobs, "leased")?;
        let provider_failed = queue_count(provider_jobs, "failed")?;
        let pending = provider_queued
            .checked_add(provider_leased)
            .and_then(|count| count.checked_add(provider_failed))
            .ok_or_else(|| anyhow!("queue provider pending count overflow"))?;
        embedding_pending.insert(provider.clone(), pending);
    }
    Ok(QueueState {
        unfinished,
        ready,
        leased,
        next_ready_at,
        observed_at,
        embedding_pending,
    })
}

fn published_embedding_units(report: &Value) -> Result<u64> {
    report["providers"]
        .as_array()
        .ok_or_else(|| anyhow!("queue drain is missing provider progress"))?
        .iter()
        .filter(|provider| {
            provider["provider"]
                .as_str()
                .is_some_and(|name| name.starts_with("embed:"))
        })
        .try_fold(0_u64, |sum, provider| {
            let published = provider["published"]
                .as_u64()
                .ok_or_else(|| anyhow!("queue drain has invalid published count"))?;
            sum.checked_add(published)
                .ok_or_else(|| anyhow!("queue drain published count overflow"))
        })
}

fn finish_model_queue(
    root: &Path,
    started: Instant,
) -> Result<(Vec<Value>, QueueState, u64, Option<u64>)> {
    let deadline = started
        .checked_add(MODEL_DRAIN_TIMEOUT)
        .ok_or_else(|| anyhow!("model drain deadline overflow"))?;
    let mut drains = Vec::new();
    let mut published = 0_u64;
    let mut peak_rss = None;
    loop {
        let (status, _, _) = invoke_dkc_json(
            root,
            &["queue".to_owned(), "status".to_owned(), "--json".to_owned()],
        )?;
        let state = queue_state(&status)?;
        if state.unfinished == 0 || drains.len() >= MODEL_DRAIN_INVOCATIONS {
            return Ok((drains, state, published, peak_rss));
        }
        if state.ready == 0 {
            let wait_seconds = match state.next_ready_at {
                Some(next) => next.saturating_sub(state.observed_at).max(1),
                None if state.leased > 0 => 1,
                None => return Ok((drains, state, published, peak_rss)),
            };
            let wait = Duration::from_secs(u64::try_from(wait_seconds)?);
            let Some(wake) = Instant::now().checked_add(wait) else {
                return Ok((drains, state, published, peak_rss));
            };
            if wake > deadline {
                return Ok((drains, state, published, peak_rss));
            }
            thread::sleep(wait);
            continue;
        }
        if Instant::now() >= deadline {
            return Ok((drains, state, published, peak_rss));
        }
        let (drain, _, drain_peak) = invoke_dkc_json(
            root,
            &[
                "queue".to_owned(),
                "drain".to_owned(),
                "--max-passes".to_owned(),
                MODEL_DRAIN_PASSES.to_string(),
                "--json".to_owned(),
            ],
        )?;
        published = published
            .checked_add(published_embedding_units(&drain)?)
            .ok_or_else(|| anyhow!("model drain published count overflow"))?;
        peak_rss = peak_rss.max(drain_peak);
        drains.push(drain);
    }
}

struct PreparedConfig {
    settings: crate::config::Config,
    effective_config_blake3: String,
    config_files_blake3: BTreeMap<String, String>,
}

struct CaptureScratch(PathBuf);

impl CaptureScratch {
    fn create() -> Result<Self> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root = env::temp_dir().join(format!("dkc-bench-config-{}-{nonce}", std::process::id()));
        fs::create_dir(&root).with_context(|| format!("failed to create {}", root.display()))?;
        let scratch = Self(root);
        #[cfg(unix)]
        fs::set_permissions(&scratch.0, fs::Permissions::from_mode(0o700))?;
        Ok(scratch)
    }
}

impl Drop for CaptureScratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn ensure_capture_index_root(scratch: &Path, settings: &crate::config::Config) -> Result<()> {
    let index_root = Path::new(&settings.index.root);
    if index_root.is_absolute()
        || index_root
            .components()
            .any(|part| matches!(part, std::path::Component::ParentDir))
    {
        bail!("capture refuses an index root outside its disposable repository");
    }
    let safe_root = scratch.join(index_root);
    fs::create_dir_all(&safe_root)?;
    if !fs::canonicalize(&safe_root)?.starts_with(fs::canonicalize(scratch)?) {
        bail!("capture index root escapes its disposable repository");
    }
    Ok(())
}

/// Captures exact lane setup on a disposable repo while leaving corpus indexes intact.
fn capture_config(
    corpus_path: &Path,
    target_name: &str,
    lane_profile: &str,
    json_output: bool,
) -> Result<()> {
    let profile = match lane_profile {
        "l1" => None,
        "local_cuda" => Some("local"),
        "local_cpu" => Some("cpu"),
        "moonly" => Some("moonly"),
        "ollama" => Some("ollama"),
        "dual_local_code_moonly_docs" => Some("dual"),
        _ => bail!("unknown candidate lane profile {lane_profile}"),
    };
    let corpus_path = canonical_or_current(corpus_path)?;
    let manifest_dir = corpus_path
        .parent()
        .ok_or_else(|| anyhow!("corpus manifest has no parent"))?;
    let manifest: CorpusManifest = toml::from_str(&fs::read_to_string(&corpus_path)?)?;
    let mut selected = manifest.repo.iter().filter(|row| row.name == target_name);
    let target = selected
        .next()
        .ok_or_else(|| anyhow!("corpus target {target_name} is absent"))?;
    if selected.next().is_some() {
        bail!("corpus target {target_name} is duplicated");
    }
    let real_root = target_root(target, manifest_dir)?;
    if !real_root.join(".git").exists() {
        bail!(
            "benchmark target {} is not a full Git checkout",
            real_root.display()
        );
    }
    if real_root.join(".deepkit/bench-edits").exists()
        || real_root.join(".deepkit/bench-head.json").exists()
    {
        bail!("benchmark target {target_name} has pending benchmark recovery");
    }
    let head = git_stdout(&real_root, &["rev-parse", "HEAD"])?
        .trim()
        .to_owned();
    if !target.rev.is_empty() && head != target.rev {
        bail!("benchmark target {target_name} moved from its pinned revision");
    }
    if !git_stdout(
        &real_root,
        &["status", "--porcelain", "--untracked-files=no"],
    )?
    .trim()
    .is_empty()
    {
        bail!("benchmark target {target_name} has tracked changes");
    }

    let scratch = CaptureScratch::create()?;
    git_stdout(&scratch.0, &["init", "--quiet"])?;
    git_stdout(
        &scratch.0,
        &[
            "-c",
            "user.name=DKC Capture",
            "-c",
            "user.email=dkc-capture@example.invalid",
            "commit",
            "--allow-empty",
            "--quiet",
            "-m",
            "capture seed",
        ],
    )?;
    let mut starting_config_sha256 = BTreeMap::new();
    for name in ["config.toml", "config.local.toml"] {
        let source = real_root.join(".deepkit").join(name);
        if source.is_symlink() {
            bail!("benchmark target {target_name} has a linked {name}");
        }
        if source.is_file() {
            let destination = scratch.0.join(".deepkit").join(name);
            fs::create_dir_all(
                destination
                    .parent()
                    .ok_or_else(|| anyhow!("no config parent"))?,
            )?;
            let bytes = fs::read(&source)?;
            starting_config_sha256.insert(
                name.to_owned(),
                Sha256::digest(&bytes)
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>(),
            );
            fs::write(&destination, bytes)?;
        }
    }
    apply_target_config(target, manifest_dir, &scratch.0)?;
    let initial = crate::config::load(&scratch.0, None)?.config;
    ensure_capture_index_root(&scratch.0, &initial)?;
    let prepared = prepare_config(target, manifest_dir, &scratch.0, profile, None)?;
    ensure_capture_index_root(&scratch.0, &prepared.settings)?;
    let (status, _, _) = invoke_dkc_json(&scratch.0, &["status".to_owned(), "--json".to_owned()])?;
    let models = status
        .get("models")
        .ok_or_else(|| anyhow!("prepared status lacks model identity"))?;
    let settings = &prepared.settings;
    let model_usage = json!({
        "verified_models": if profile.is_some() { models.clone() } else { Value::Null },
        "configured_profile": settings.embedder.profile,
        "configured_embedding": {"remote": settings.embedder.remote.model, "local": settings.embedder.local.model, "ollama": settings.embedder.ollama.model},
        "configured_reranker": {"remote": settings.rerank.remote.model, "local": settings.rerank.local.model},
        "docs_tier_configured": settings.embedder.docs.is_some(),
    });
    let receipt = json!({
        "target_head": head,
        "effective_config": {
            "schema": "dkc.phase4.effective-config.v1", "target": target_name,
            "profile": lane_profile,
            "effective_config_blake3": prepared.effective_config_blake3,
            "config_files_blake3": prepared.config_files_blake3,
            "starting_config_sha256": starting_config_sha256,
        },
        "model_identity_template": {
            "schema": "dkc.phase4.model-identity.v1", "target": target_name,
            "profile": lane_profile, "model_usage": model_usage,
            "identities": [],
        },
        "model_evidence_complete": false,
    });
    emit(
        json_output,
        receipt,
        "Captured benchmark configuration; independent model evidence is still required",
    )?;
    Ok(())
}

/// Uses the same configuration steps for a real lane and a disposable capture.
fn prepare_config(
    target: &CorpusTarget,
    manifest_dir: &Path,
    root: &Path,
    profile: Option<&str>,
    mut progress: Option<&mut TargetProgress>,
) -> Result<PreparedConfig> {
    if let Some(state) = progress.as_mut() {
        state.stage = "configure";
    }
    apply_target_config(target, manifest_dir, root)?;
    if let Some(state) = progress.as_mut() {
        state
            .measurements
            .insert("configuration_applied".to_owned(), json!(true));
        state.stage = "clean";
    }
    let _ = invoke_dkc_json(root, &["clean".to_owned(), "--json".to_owned()])?;
    if let Some(state) = progress.as_mut() {
        state.stage = "init";
    }
    let mut init = vec!["init".to_owned()];
    if let Some(profile) = profile {
        let init_profile = match profile {
            "cpu" | "dual" => "local",
            other => other,
        };
        init.extend(["--profile".to_owned(), init_profile.to_owned()]);
        if profile == "dual" {
            init.extend(["--docs".to_owned(), "moonly".to_owned()]);
        }
        init.push("--yes".to_owned());
    }
    init.push("--json".to_owned());
    let _ = invoke_dkc_json(root, &init)?;
    if matches!(profile, Some("local" | "cpu" | "dual")) {
        let gpu_layers = if profile == Some("cpu") { "0" } else { "-1" };
        crate::config::set(root, "embedder.local.gpu_layers", gpu_layers, false)?;
        crate::config::set(root, "rerank.local.gpu_layers", gpu_layers, false)?;
    }
    if let Some(state) = progress.as_mut() {
        state
            .measurements
            .insert("initialized".to_owned(), json!(true));
        state.stage = "pre_index_metadata";
    }
    let settings = crate::config::load(root, None)?.config;
    match profile {
        Some("local")
            if settings.embedder.profile != "local"
                || settings.embedder.local.gpu_layers != -1
                || settings.rerank.local.gpu_layers != -1 =>
        {
            bail!("local benchmark profile was overridden after init")
        }
        Some("cpu")
            if settings.embedder.profile != "local"
                || settings.embedder.local.gpu_layers != 0
                || settings.rerank.local.gpu_layers != 0 =>
        {
            bail!("CPU benchmark profile was overridden after init")
        }
        Some("moonly") if settings.embedder.profile != "remote" => {
            bail!("Moonly benchmark profile was overridden after init")
        }
        Some("ollama") if settings.embedder.profile != "ollama" => {
            bail!("Ollama benchmark profile was overridden after init")
        }
        Some("dual")
            if settings.embedder.profile != "local"
                || settings.embedder.local.gpu_layers != -1
                || settings.rerank.local.gpu_layers != -1
                || settings
                    .embedder
                    .docs
                    .as_ref()
                    .is_none_or(|docs| docs.profile != "remote") =>
        {
            bail!("dual benchmark profile was overridden after init")
        }
        _ => {}
    }
    let effective_config_blake3 = blake3::hash(&serde_json::to_vec(&settings)?)
        .to_hex()
        .to_string();
    let config_files_blake3 = ["config.toml", "config.local.toml"]
        .into_iter()
        .filter_map(|name| {
            let path = root.join(".deepkit").join(name);
            path.exists().then(|| {
                fs::read(&path)
                    .map(|bytes| (name.to_owned(), blake3::hash(&bytes).to_hex().to_string()))
            })
        })
        .collect::<std::io::Result<BTreeMap<_, _>>>()?;
    Ok(PreparedConfig {
        settings,
        effective_config_blake3,
        config_files_blake3,
    })
}

fn run_target(
    target: &CorpusTarget,
    manifest_dir: &Path,
    profile: Option<&str>,
    keep: bool,
    progress: &mut TargetProgress,
) -> Result<TargetReport> {
    let root = target_root(target, manifest_dir)?;
    progress.measurements.insert("root".to_owned(), json!(root));
    progress.stage = "validate_target";
    if !root.is_dir() {
        bail!("benchmark target {} is not a directory", root.display());
    }
    if !root.join(".git").exists() {
        bail!(
            "benchmark target {} is not a full Git checkout",
            root.display()
        );
    }
    progress.stage = "recover_mutations";
    super::bench_edits::restore(&root)?;
    freshness::recover(&root)?;
    if !target.rev.is_empty() {
        let head = git_stdout(&root, &["rev-parse", "HEAD"])?;
        if head.trim() != target.rev {
            bail!(
                "benchmark target {} is at {}, not pinned revision {}",
                root.display(),
                head.trim(),
                target.rev
            );
        }
    }
    let PreparedConfig {
        settings,
        effective_config_blake3,
        config_files_blake3,
    } = prepare_config(target, manifest_dir, &root, profile, Some(&mut *progress))?;
    let pre_index =
        super::bench_metadata::preflight(matches!(profile, Some("local" | "dual" | "ollama")));
    progress
        .measurements
        .insert("pre_index".to_owned(), json!(pre_index));
    progress.measurements.insert(
        "effective_config_blake3".to_owned(),
        json!(effective_config_blake3),
    );
    progress.stage = "index_cold";
    let (cold_index_result, cold_duration, cold_peak_rss) = invoke_dkc_json(
        &root,
        &[
            "index".to_owned(),
            "--level".to_owned(),
            "1".to_owned(),
            "--full".to_owned(),
            "--json".to_owned(),
        ],
    )?;
    progress.measurements.insert(
        "cold_index_ms".to_owned(),
        json!(duration_ms(cold_duration)),
    );
    progress
        .measurements
        .insert("cold_index_result".to_owned(), cold_index_result.clone());
    progress
        .measurements
        .insert("cold_peak_rss_bytes".to_owned(), json!(cold_peak_rss));
    progress.stage = "eligibility";
    let eligible = eligible_text_files(&root, &settings)?;
    progress.measurements.insert(
        "eligibility".to_owned(),
        json!({
            "admitted_text_bytes": eligible.bytes, "accounting": eligible.accounting,
        }),
    );
    progress.stage = "touch";
    let touched_files = touch_regular_files(&eligible.paths, TOUCH_FILE_LIMIT)?;
    progress.stage = "index_incremental";
    let (mtime_only_index_result, incremental_duration, incremental_peak_rss) = invoke_dkc_json(
        &root,
        &[
            "index".to_owned(),
            "--level".to_owned(),
            "1".to_owned(),
            "--json".to_owned(),
        ],
    )?;
    progress.measurements.insert(
        "incremental_ms".to_owned(),
        json!(duration_ms(incremental_duration)),
    );
    progress.measurements.insert(
        "mtime_only_index_result".to_owned(),
        mtime_only_index_result.clone(),
    );
    progress.stage = "prepare_content_edits";
    let edited_paths = super::bench_edits::prepare(
        &root,
        &eligible.paths,
        settings.limits.max_file_bytes()?.0,
        TOUCH_FILE_LIMIT,
    )?;
    progress
        .measurements
        .insert("edited_paths".to_owned(), json!(edited_paths));
    let content_edit = if edited_paths.is_empty() {
        None
    } else {
        progress.stage = "index_content_edits";
        let measured = invoke_dkc_json(
            &root,
            &[
                "index".to_owned(),
                "--level".to_owned(),
                "1".to_owned(),
                "--json".to_owned(),
            ],
        );
        if let Ok((result, duration, rss)) = &measured {
            progress.measurements.insert("content_edit_index".to_owned(), json!({
                "index_result": result, "elapsed_ms": duration_ms(*duration), "peak_rss_bytes": rss,
            }));
        }
        // Restore on child failure too; interrupted runs recover before the next clean.
        super::bench_edits::restore(&root)?;
        let (index_result, duration, peak_rss_bytes) = measured?;
        progress.stage = "reindex_restored_sources";
        invoke_dkc_json(
            &root,
            &[
                "index".to_owned(),
                "--level".to_owned(),
                "1".to_owned(),
                "--json".to_owned(),
            ],
        )?;
        Some(ContentEditMetrics {
            method: "seeded-source-addition-v1".to_owned(),
            paths: edited_paths,
            elapsed_ms: duration_ms(duration),
            peak_rss_bytes,
            index_result,
            restored: true,
        })
    };
    progress
        .measurements
        .insert("content_edit".to_owned(), json!(content_edit));
    let mut model_index = if let Some(profile) = profile {
        progress.stage = "index_models";
        let requested_level = if matches!(profile, "moonly" | "dual") {
            3
        } else {
            2
        };
        let model_started = Instant::now();
        let (index_result, _, peak_rss_bytes) = invoke_dkc_json(
            &root,
            &[
                "index".to_owned(),
                "--level".to_owned(),
                requested_level.to_string(),
                "--json".to_owned(),
            ],
        )?;
        let initial_published_units = index_result["embedding"]["providers"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|provider| provider["published"].as_u64())
            .sum::<u64>();
        progress.stage = "drain_models";
        let (drain_results, terminal, drained_units, drain_peak_rss) =
            finish_model_queue(&root, model_started)?;
        let published_units = initial_published_units
            .checked_add(drained_units)
            .ok_or_else(|| anyhow!("model publication count overflow"))?;
        let pending_jobs = terminal
            .embedding_pending
            .values()
            .try_fold(0_i64, |sum, pending| sum.checked_add(*pending));
        let elapsed_ms = duration_ms(model_started.elapsed());
        let metrics = ModelIndexMetrics {
            profile: profile.to_owned(),
            requested_level,
            elapsed_ms,
            peak_rss_bytes: peak_rss_bytes.max(drain_peak_rss),
            published_units,
            throughput_units_per_second: (elapsed_ms > 0.0)
                .then(|| published_units as f64 * 1_000.0 / elapsed_ms),
            embedding_state: if pending_jobs == Some(0) {
                "quiescent".to_owned()
            } else {
                "pending".to_owned()
            },
            pending_jobs,
            terminal_provider_pending: terminal.embedding_pending,
            coverage: Value::Null,
            index_result,
            drain_results,
        };
        Some(metrics)
    } else {
        None
    };
    progress.stage = "status";
    let (status, _, _) = invoke_dkc_json(&root, &["status".to_owned(), "--json".to_owned()])?;
    if let Some(metrics) = &mut model_index {
        metrics.coverage = status["l2"].clone();
        progress
            .measurements
            .insert("model_index".to_owned(), json!(metrics));
    }
    progress
        .measurements
        .insert("status".to_owned(), status.clone());
    progress.stage = "indexed_metadata";
    let provenance = {
        let store = super::open_store(&root, &settings)?;
        let snapshot = store.index_snapshot()?;
        let source_inputs = snapshot
            .files
            .iter()
            .map(|file| (&file.path, &file.blake3, file.size))
            .collect::<Vec<_>>();
        TargetProvenance {
            head: git_stdout(&root, &["rev-parse", "HEAD"])?.trim().to_owned(),
            effective_config_blake3,
            config_files_blake3,
            indexed_sources_blake3: blake3::hash(&serde_json::to_vec(&source_inputs)?)
                .to_hex()
                .to_string(),
            schema_version: store.get_meta("schema_version")?,
            page_size: store.footprint()?.page_size,
            mmap_size: store.mmap_size()?,
            model_usage: json!({
                "state": if model_index.is_some() { "index_profile_invoked" } else { "not_invoked_l1_only" },
                "benchmark_profile": profile.unwrap_or("l1"),
                "model_index": model_index,
                "verified_models": if model_index.is_some() { status["models"].clone() } else { Value::Null },
                "configured_profile": settings.embedder.profile,
                "configured_embedding": {"remote": settings.embedder.remote.model, "local": settings.embedder.local.model, "ollama": settings.embedder.ollama.model},
                "configured_reranker": {"remote": settings.rerank.remote.model, "local": settings.rerank.local.model},
                "docs_tier_configured": settings.embedder.docs.is_some(),
            }),
            pre_index,
        }
    };
    progress
        .measurements
        .insert("provenance".to_owned(), json!(provenance));
    let job_states = super::open_store(&root, &settings)?.job_state_counts()?;
    let robustness = RobustnessMetrics {
        // Every child response is checked for panic diagnostics before becoming a successful target.
        panics: 0,
        dead_jobs: u64::try_from(job_states.get("dead").copied().unwrap_or_default())
            .context("invalid negative dead-job count")?,
        job_states: Some(job_states),
        cap_outcomes: Some(
            json!({ "caps": status["caps"], "l1_deferred": status["l1_deferred"],
            "l1_pending_files": status["l1_pending_files"], "graph_pending": status["graph_pending"] }),
        ),
        parse_failures_by_language: Some(eligible.parse_failures_by_language.clone()),
        // Document extraction telemetry is added with the document-profile corpus pass.
        extraction_failures: None,
    };
    progress
        .measurements
        .insert("robustness".to_owned(), json!(robustness));
    progress.stage = "derive_goldens";
    let goldens = commit_derived_goldens(&root, &settings.tests.globs)?;
    progress.stage = "search";
    let mut quality = evaluate_goldens(&root, &goldens, &settings)?;
    let paired_search = if profile.is_some() && goldens.is_empty() {
        Some(json!({
            "state": "skipped",
            "reason": "no_commit_derived_goldens",
            "goldens": 0,
            "repeats": PAIRED_SEARCH_REPEATS,
            "arms": ["lexical", "hybrid", "dense_only", "reranked"],
        }))
    } else if profile.is_some() {
        progress.stage = "paired_search";
        let report_path = root.join(".deepkit/bench-run-paired-search.json");
        match fs::remove_file(&report_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let arguments = vec![
            "bench".to_owned(),
            "search".to_owned(),
            "--root".to_owned(),
            root.to_string_lossy().to_string(),
            "--out".to_owned(),
            report_path.to_string_lossy().to_string(),
            "--max-calls".to_owned(),
            BENCH_MAX_CALLS.to_string(),
            "--max-tokens".to_owned(),
            BENCH_MAX_TOKENS.to_string(),
            "--repeats".to_owned(),
            PAIRED_SEARCH_REPEATS.to_string(),
            "--dense-only".to_owned(),
            "--rerank".to_owned(),
            "--rerank-rpm".to_owned(),
            BENCH_RERANK_RPM.to_string(),
            "--rerank-tpm".to_owned(),
            BENCH_RERANK_TPM.to_string(),
        ];
        let (report, child_error) = invoke_dkc_saved_json(&root, &arguments, &report_path)?;
        if let Some(report) = &report {
            progress
                .measurements
                .insert("paired_search".to_owned(), report.clone());
        }
        if let Some(error) = child_error {
            return Err(error);
        }
        let report = report.context("paired search succeeded without a saved report")?;
        fs::remove_file(&report_path).with_context(|| {
            format!(
                "failed to remove temporary paired-search report {}",
                report_path.display()
            )
        })?;
        Some(report)
    } else {
        None
    };
    let pr_report = &cold_index_result["prs"];
    let pr_detected = pr_report["detected"].as_u64().unwrap_or_default();
    let pr_reconstructed = ["inserted", "existing", "updated"]
        .into_iter()
        .filter_map(|key| pr_report[key].as_u64())
        .sum::<u64>();
    let composites = CompositeMetrics {
        pr_reconstruction: PrReconstructionMetrics {
            detected: pr_detected,
            reconstructed: pr_reconstructed,
            rate: (pr_detected > 0).then(|| pr_reconstructed as f64 / pr_detected as f64),
            deferred: pr_report["deferred"].clone(),
        },
    };
    progress.stage = "handwritten_goldens";
    if let Some(path) = &target.golden {
        quality.handwritten = Some(goldens::evaluate(
            &root,
            &settings,
            &resolve_manifest_path(manifest_dir, path),
            &target
                .extra_goldens
                .iter()
                .map(|path| resolve_manifest_path(manifest_dir, path))
                .collect::<Vec<_>>(),
            target.golden_reviewed_blake3.as_deref(),
        )?);
    } else if !target.extra_goldens.is_empty() {
        bail!("extra_goldens requires a primary golden file");
    }
    progress
        .measurements
        .insert("quality".to_owned(), json!(quality));
    progress.stage = "query_latency";
    let query_database = super::database_path(&root, &settings);
    let query_ledger_before = model_ledger(&query_database)?;
    let queries = measure_queries(&root, &goldens, &settings)?;
    let query_ledger_after = model_ledger(&query_database)?;
    if query_ledger_before != query_ledger_after {
        bail!(
            "lexical benchmark observed model work: before={query_ledger_before:?}, after={query_ledger_after:?}"
        );
    }
    progress.stage = "freshness";
    let freshness = Some(freshness::measure(
        &root,
        &eligible.paths,
        settings.limits.max_file_bytes()?.0,
        model_index
            .as_ref()
            .map_or(1, |metrics| metrics.requested_level),
    )?);
    progress
        .measurements
        .insert("freshness".to_owned(), json!(freshness));
    progress.stage = "footprint";
    let checkout_bytes = eligible.bytes;
    let database = super::database_path(&root, &settings);
    let db_bytes = fs::metadata(&database)?.len();
    let wal_path = PathBuf::from(format!("{}-wal", database.display()));
    let wal_bytes = match fs::metadata(&wal_path) {
        Ok(metadata) => metadata.len(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(error) => return Err(error.into()),
    };
    let accounting = FootprintAccounting {
        wal_bytes,
        total_db_bytes: db_bytes
            .checked_add(wal_bytes)
            .context("database size overflow")?,
        ..eligible.accounting
    };
    if !keep {
        progress.stage = "cleanup";
        let _ = invoke_dkc_json(&root, &["clean".to_owned(), "--json".to_owned()])?;
    }

    Ok(TargetReport {
        name: target.name.clone(),
        root: root.to_string_lossy().to_string(),
        url: target.url.clone(),
        rev: target.rev.clone(),
        config: target.config.as_ref().map(|path| {
            resolve_manifest_path(manifest_dir, path)
                .to_string_lossy()
                .to_string()
        }),
        golden: target.golden.as_ref().map(|path| {
            resolve_manifest_path(manifest_dir, path)
                .to_string_lossy()
                .to_string()
        }),
        index: IndexMetrics {
            cold_ms: duration_ms(cold_duration),
            cold_index_result: Some(cold_index_result),
            mtime_only_index_result: Some(mtime_only_index_result),
            incremental_ms: duration_ms(incremental_duration),
            peak_rss_bytes: [cold_peak_rss, incremental_peak_rss]
                .into_iter()
                .flatten()
                .max(),
            touched_files,
            content_edit,
            files: value_u64(&status, "files"),
            units: value_u64(&status, "units"),
            symbols: value_u64(&status, "symbols"),
            edges: value_u64(&status, "edges"),
        },
        model_index,
        paired_search,
        composites: Some(composites),
        size: SizeMetrics {
            db_bytes,
            checkout_bytes,
            ratio: (checkout_bytes > 0).then(|| db_bytes as f64 / checkout_bytes as f64),
            accounting: Some(accounting),
        },
        quality,
        queries,
        freshness,
        status: Some(status),
        provenance: Some(provenance),
        robustness,
    })
}

fn target_root(target: &CorpusTarget, manifest_dir: &Path) -> Result<PathBuf> {
    let configured = target
        .path
        .as_ref()
        .map(|path| resolve_manifest_path(manifest_dir, path));
    let root = configured.unwrap_or_else(|| {
        env::var_os("DKC_CORPUS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/data/dkc-corpus"))
            .join(&target.name)
    });
    fs::canonicalize(&root)
        .with_context(|| format!("failed to resolve benchmark target {}", root.display()))
}

fn apply_target_config(target: &CorpusTarget, manifest_dir: &Path, root: &Path) -> Result<()> {
    let Some(config) = &target.config else {
        return Ok(());
    };
    let source = resolve_manifest_path(manifest_dir, config);
    if !source.is_file() {
        bail!("benchmark config {} is not a file", source.display());
    }
    let destination = root.join(".deepkit/config.toml");
    let parent = destination
        .parent()
        .ok_or_else(|| anyhow!("benchmark config destination has no parent"))?;
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;
    fs::copy(&source, &destination).with_context(|| {
        format!(
            "failed to copy benchmark config {} to {}",
            source.display(),
            destination.display()
        )
    })?;
    Ok(())
}

/// Runs one CLI command and, for index work on Linux, samples its peak resident set.
fn invoke_dkc_json(root: &Path, arguments: &[String]) -> Result<(Value, Duration, Option<u64>)> {
    let executable = env::current_exe().context("failed to locate the dkc executable")?;
    let started = Instant::now();
    let child = ProcessCommand::new(executable)
        .args(arguments)
        .current_dir(root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to run dkc in {}", root.display()))?;
    let sampler = (arguments.first().is_some_and(|command| command == "index")
        || arguments
            .get(0..2)
            .is_some_and(|commands| commands == ["queue", "drain"]))
    .then(|| spawn_peak_rss_sampler(child.id()));
    let output = child
        .wait_with_output()
        .with_context(|| format!("failed to wait for dkc in {}", root.display()))?;
    let elapsed = started.elapsed();
    let peak_rss_bytes = sampler
        .and_then(|sampler| sampler.finish())
        .filter(|peak| *peak > 0);
    check_child_output(arguments, &output)?;
    let response = serde_json::from_slice(&output.stdout).with_context(|| {
        format!(
            "dkc {} did not return JSON in {}",
            arguments.join(" "),
            root.display()
        )
    })?;
    Ok((response, elapsed, peak_rss_bytes))
}

/// Runs a benchmark child that checkpoints a potentially large JSON report to disk.
fn invoke_dkc_saved_json(
    root: &Path,
    arguments: &[String],
    report_path: &Path,
) -> Result<(Option<Value>, Option<anyhow::Error>)> {
    let executable = env::current_exe().context("failed to locate the dkc executable")?;
    let output = ProcessCommand::new(executable)
        .args(arguments)
        .current_dir(root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("failed to run dkc in {}", root.display()))?;
    let report = match fs::read(report_path) {
        Ok(bytes) => Some(serde_json::from_slice(&bytes).with_context(|| {
            format!(
                "failed to parse saved benchmark report {}",
                report_path.display()
            )
        })?),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    let child_error = check_child_output(arguments, &output).err();
    Ok((report, child_error))
}

/// Panic diagnostics remain failures even when a caught worker panic leaves a zero process exit.
fn check_child_output(arguments: &[String], output: &std::process::Output) -> Result<()> {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    let panics = stderr.matches("panicked at").count() as u64;
    if !output.status.success() || panics > 0 {
        return Err(ChildFailure {
            command: arguments.join(" "),
            exit_code: output.status.code(),
            stderr,
            panics,
        }
        .into());
    }
    Ok(())
}

/// A joinable Linux `/proc` sampler for an index subprocess's peak RSS in bytes.
struct PeakRssSampler {
    done: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    join: thread::JoinHandle<()>,
}

impl PeakRssSampler {
    /// Stops sampling and returns the highest observed resident set size.
    fn finish(self) -> Option<u64> {
        self.done.store(true, Ordering::Release);
        let _ = self.join.join();
        let peak = self.peak.load(Ordering::Acquire);
        (peak > 0).then_some(peak)
    }
}

/// Samples `/proc/<pid>/status` while an index child runs; unsupported systems report no value.
fn spawn_peak_rss_sampler(pid: u32) -> PeakRssSampler {
    let done = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(0));
    let sampler_done = Arc::clone(&done);
    let sampler_peak = Arc::clone(&peak);
    let join = thread::spawn(move || {
        loop {
            if let Some(current) = proc_peak_rss_bytes(pid) {
                sampler_peak.fetch_max(current, Ordering::Relaxed);
            }
            if sampler_done.load(Ordering::Acquire) {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
    });
    PeakRssSampler { done, peak, join }
}

fn proc_resident_rss_bytes() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let line = status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))?;
    line.split_ascii_whitespace()
        .next()?
        .parse::<u64>()
        .ok()?
        .checked_mul(1024)
}

#[derive(Debug, PartialEq, Serialize)]
struct ModelLedger {
    jobs: i64,
    queued_jobs: i64,
    leased_jobs: i64,
    job_attempts: i64,
    publications: i64,
    reservations: i64,
    usage_calls: i64,
    usage_tokens: i64,
}

/// Reads model-work accounting without opening a writer or changing the index.
fn model_ledger(database: &Path) -> Result<ModelLedger> {
    let connection = Connection::open_with_flags(database, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    Ok(connection.query_row(
        "SELECT (SELECT COUNT(*) FROM jobs),
                (SELECT COUNT(*) FROM jobs WHERE state='queued'),
                (SELECT COUNT(*) FROM jobs WHERE state='leased'),
                (SELECT COALESCE(SUM(attempts),0) FROM jobs),
                (SELECT COUNT(*) FROM embedding_publications),
                (SELECT COUNT(*) FROM usage_reservations),
                (SELECT COALESCE(SUM(calls),0) FROM usage),
                (SELECT COALESCE(SUM(tokens),0) FROM usage)",
        [],
        |row| {
            Ok(ModelLedger {
                jobs: row.get(0)?,
                queued_jobs: row.get(1)?,
                leased_jobs: row.get(2)?,
                job_attempts: row.get(3)?,
                publications: row.get(4)?,
                reservations: row.get(5)?,
                usage_calls: row.get(6)?,
                usage_tokens: row.get(7)?,
            })
        },
    )?)
}

/// Exercises the normal shared query path against a retained page layout for memory comparisons.
fn probe_existing(root: &Path, out: &Path, json_output: bool) -> Result<()> {
    let root = fs::canonicalize(root)?;
    let settings = crate::config::load(&root, None)?.config;
    let database = super::database_path(&root, &settings);
    // Read-only SQLite can create WAL bookkeeping, so admit its destination too.
    crate::store::path::prepare(&root, &database)?.check()?;
    if !database.is_file() {
        bail!("probe requires an existing index; run dkc index first");
    }
    let build = super::bench_metadata::build_info()?;
    let preflight = super::bench_metadata::preflight(false);
    let goldens = commit_derived_goldens(&root, &settings.tests.globs)?;
    if goldens.is_empty() {
        bail!("probe requires commit-derived search queries");
    }
    let ledger_before = model_ledger(&database)?;
    let queries = measure_queries(&root, &goldens, &settings)?;
    let store = super::open_store(&root, &settings)?;
    let snapshot = store.index_snapshot()?;
    let sources = snapshot
        .files
        .iter()
        .map(|f| (&f.path, &f.blake3, f.size))
        .collect::<Vec<_>>();
    let footprint = store.footprint()?;
    let mut wal_path = super::database_path(&root, &settings).into_os_string();
    wal_path.push("-wal");
    let wal_bytes = match fs::metadata(&wal_path) {
        Ok(metadata) => metadata.len(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(error) => return Err(error).context("failed to measure probe WAL"),
    };
    let counts = store.counts()?;
    let ledger_after = model_ledger(&database)?;
    if ledger_before != ledger_after {
        bail!(
            "retained-index probe observed model work: before={ledger_before:?}, after={ledger_after:?}"
        );
    }
    let report = json!({
        "method": "existing-index-shared-read-probe-v2",
        "root": root, "build": build, "preflight": preflight,
        "head": git_stdout(&root, &["rev-parse", "HEAD"])?.trim(),
        "effective_config_blake3": blake3::hash(&serde_json::to_vec(&settings)?).to_hex().to_string(),
        "indexed_sources_blake3": blake3::hash(&serde_json::to_vec(&sources)?).to_hex().to_string(),
        "schema_version": store.get_meta("schema_version")?,
        "footprint": {"db_bytes": footprint.bytes, "page_size": footprint.page_size,
            "page_count": footprint.page_count, "freelist_count": footprint.freelist_count,
            "wal_bytes": wal_bytes}, "queries": queries,
        "index_state": {"files": counts.files, "units": counts.units,
            "symbols": counts.symbols, "edges": counts.edges,
            "l1_pending_files": counts.l1_pending_files,
            "graph_pending": store.get_meta("graph_pending")?},
        "query_inputs_blake3": blake3::hash(&serde_json::to_vec(&goldens)?).to_hex().to_string(),
        "model_usage": "not_invoked_l1_only",
        "model_ledger": ledger_after,
    });
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    emit(
        json_output,
        report,
        &format!("Existing-index probe written to {}", out.display()),
    )?;
    Ok(())
}

fn proc_peak_rss_bytes(pid: u32) -> Option<u64> {
    let status = fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    let kibibytes = status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))?
        .split_ascii_whitespace()
        .next()?
        .parse::<u64>()
        .ok()?;
    kibibytes.checked_mul(1024)
}

fn commit_derived_goldens(root: &Path, tests_globs: &[String]) -> Result<Vec<CommitGolden>> {
    let mut builder = globset::GlobSetBuilder::new();
    for pattern in tests_globs {
        builder.add(globset::Glob::new(pattern)?);
    }
    let tests = builder.build()?;
    if !root.join(".git").exists() {
        return Ok(Vec::new());
    }
    let revisions = git_stdout(root, &["rev-list", "--max-count=5000", "HEAD"])?;
    let mut goldens = Vec::new();
    for revision in revisions.lines() {
        if goldens.len() == GOLDEN_LIMIT {
            break;
        }
        let message = git_stdout(root, &["show", "-s", "--format=%B", revision])?;
        let message = message.trim();
        if message.chars().count() <= 40 {
            continue;
        }
        let changed = git_stdout(
            root,
            &[
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-z",
                "-r",
                "--first-parent",
                revision,
            ],
        )?;
        let expected_files = changed
            .split('\0')
            .filter(|path| !path.is_empty() && !tests.is_match(path))
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
        if !(1..=5).contains(&expected_files.len()) {
            continue;
        }
        goldens.push(CommitGolden {
            revision: revision.to_owned(),
            query: message.chars().take(300).collect(),
            expected_files,
        });
    }
    Ok(goldens)
}

fn git_stdout(root: &Path, arguments: &[&str]) -> Result<String> {
    let output = ProcessCommand::new("git")
        .args(arguments)
        .current_dir(root)
        .output()
        .with_context(|| format!("failed to run git {}", arguments.join(" ")))?;
    if !output.status.success() {
        bail!(
            "git {} failed in {}: {}",
            arguments.join(" "),
            root.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    String::from_utf8(output.stdout)
        .with_context(|| format!("git {} returned invalid UTF-8", arguments.join(" ")))
}

fn evaluate_goldens(
    root: &Path,
    goldens: &[CommitGolden],
    settings: &crate::config::Config,
) -> Result<QualityMetrics> {
    let context = crate::search::service::Context::load(root)?;
    if context.config != *settings {
        bail!("configuration changed before golden evaluation");
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    evaluate_golden_paths(root, goldens, settings, &context.store, |_, golden| {
        let response = shared_search(&context, &runtime, &golden.query)?;
        let hits = response["hits"]
            .as_array()
            .ok_or_else(|| anyhow!("search benchmark response did not contain an array of hits"))?;
        Ok(hits
            .iter()
            .map(|hit| hit["path"].as_str().unwrap_or_default().to_owned())
            .collect())
    })
}

fn evaluate_golden_paths(
    root: &Path,
    goldens: &[CommitGolden],
    settings: &crate::config::Config,
    store: &crate::store::Store,
    mut paths: impl FnMut(usize, &CommitGolden) -> Result<Vec<String>>,
) -> Result<QualityMetrics> {
    let snapshot = store.index_snapshot()?;
    let files = snapshot
        .files
        .iter()
        .map(|file| (file.path.as_str(), file))
        .collect::<BTreeMap<_, _>>();
    let searchable = store.searchable_source_paths()?;
    let max_bytes = settings.limits.max_file_bytes()?.0;
    let mut eligibility = GoldenEligibility {
        method: "source-raw-and-eligible-rank-preserving-v2".to_owned(),
        tests_globs: settings.tests.globs.clone(),
        eligible: GoldenMetrics {
            count: 0,
            hit_file_at_5: None,
            recall_at_10: None,
        },
        raw_expected_paths: 0,
        eligible_expected_paths: 0,
        rejected_paths: Vec::new(),
    };
    let mut eligible_hits = 0_usize;
    let mut eligible_recall = 0.0_f64;
    let mut hit_file_count = 0_usize;
    let mut recall_sum = 0.0_f64;
    for (index, golden) in goldens.iter().enumerate() {
        let mut admitted = BTreeSet::new();
        for path in &golden.expected_files {
            let reason = if !root.join(path).is_file() {
                Some("missing_at_pinned_tree")
            } else if let Some(file) = files.get(path.as_str()) {
                if file.size as u64 > max_bytes {
                    Some("byte_cap")
                } else if file.max_level < 1 {
                    Some("l0_treatment")
                } else if store
                    .file_truncation(file.id)?
                    .is_some_and(|value| value.truncated)
                {
                    Some("truncated_file")
                } else if !searchable.contains(path) {
                    Some("no_searchable_source_units")
                } else {
                    None
                }
            } else {
                Some("not_admitted_by_walker")
            };
            if let Some(reason) = reason {
                eligibility.rejected_paths.push(RejectedPath {
                    revision: golden.revision.clone(),
                    path: path.clone(),
                    reason: reason.to_owned(),
                });
            } else {
                admitted.insert(path.as_str());
            }
        }
        eligibility.raw_expected_paths += golden.expected_files.len();
        eligibility.eligible_expected_paths += admitted.len();
        let hits = paths(index, golden)?;
        let found_at_five = hits
            .iter()
            .take(5)
            .map(String::as_str)
            .any(|path| golden.expected_files.contains(path));
        hit_file_count += usize::from(found_at_five);
        let found_at_ten = hits
            .iter()
            .take(10)
            .map(String::as_str)
            .filter(|path| golden.expected_files.contains(*path))
            .collect::<BTreeSet<_>>();
        recall_sum += found_at_ten.len() as f64 / golden.expected_files.len() as f64;
        if !admitted.is_empty() {
            eligibility.eligible.count += 1;
            eligible_hits += usize::from(
                hits.iter()
                    .take(5)
                    .map(String::as_str)
                    .any(|path| admitted.contains(path)),
            );
            eligible_recall += found_at_ten
                .iter()
                .filter(|path| admitted.contains(**path))
                .count() as f64
                / admitted.len() as f64;
        }
    }
    let count = goldens.len();
    let denominator = (count > 0).then_some(count as f64);
    if eligibility.eligible.count > 0 {
        eligibility.eligible.hit_file_at_5 =
            Some(eligible_hits as f64 / eligibility.eligible.count as f64);
        eligibility.eligible.recall_at_10 =
            Some(eligible_recall / eligibility.eligible.count as f64);
    }
    Ok(QualityMetrics {
        commit_derived: GoldenMetrics {
            count,
            hit_file_at_5: denominator.map(|count| hit_file_count as f64 / count),
            recall_at_10: denominator.map(|count| recall_sum / count),
        },
        eligibility: Some(eligibility),
        handwritten: None,
    })
}

fn touch_regular_files(files: &[PathBuf], limit: usize) -> Result<usize> {
    let now = SystemTime::now();
    for path in files.iter().take(limit) {
        File::open(path)
            .with_context(|| format!("failed to open {} for touching", path.display()))?
            .set_times(fs::FileTimes::new().set_modified(now))
            .with_context(|| format!("failed to touch {}", path.display()))?;
    }
    Ok(files.len().min(limit))
}

/// Uses production discovery and treatment routing to define the text denominator.
fn eligible_text_files(root: &Path, settings: &crate::config::Config) -> Result<EligibleText> {
    let limits = super::sync_limits(settings)?;
    let scan = crate::sync::scan(root, limits)?;
    let router = crate::treat::Router::new(&settings.treatments, settings.llm.enabled)?;
    let mut paths = Vec::new();
    let mut bytes = 0_u64;
    let mut excluded = BTreeMap::<String, usize>::new();
    let mut parse_failures_by_language = BTreeMap::<String, ParseLanguageMetrics>::new();
    for file in scan.files() {
        let path = root.join(&file.path);
        let treatment = router.route(&file.path, file.kind);
        let reason = if file.content_capped {
            Some("byte_cap")
        } else if treatment.max_level < 1 {
            Some("l0_treatment")
        } else {
            let content = crate::sync::l1::read_source(
                root,
                file,
                limits.max_file_bytes,
                Default::default(),
            )?;
            if content.contains(&0) || std::str::from_utf8(&content).is_err() {
                Some("non_text")
            } else {
                if treatment.chunker == crate::treat::Chunker::Ast
                    && let Some(language) = file.lang.as_deref()
                    && let Some(graph) =
                        crate::treat::graph::extract_file_graph(language, &file.path, &content)?
                {
                    let metrics = parse_failures_by_language
                        .entry(language.to_owned())
                        .or_insert(ParseLanguageMetrics {
                            files: 0,
                            failures: 0,
                            failure_rate: 0.0,
                            failure_paths: Some(BTreeSet::new()),
                            failure_paths_omitted: Some(0),
                        });
                    metrics.observe(&file.path, graph.parse_error);
                }
                bytes = bytes
                    .checked_add(content.len() as u64)
                    .context("text byte count overflow")?;
                paths.push(path);
                None
            }
        };
        if let Some(reason) = reason {
            *excluded.entry(reason.to_owned()).or_default() += 1;
        }
    }
    // A fixed seed and relative path make the selection independent of checkout location.
    paths.sort_by_cached_key(|path| {
        let relative = path.strip_prefix(root).unwrap_or(path);
        blake3::hash(format!("dkc-bench-touch-v1:{}", relative.display()).as_bytes())
            .as_bytes()
            .to_owned()
    });
    let accounting = FootprintAccounting {
        method: "walker-treatment-text-v1".to_owned(),
        admitted_text_files: paths.len(),
        walker_files: scan.files().count(),
        walker_complete: scan.is_complete(),
        excluded,
        wal_bytes: 0,
        total_db_bytes: 0,
    };
    for metrics in parse_failures_by_language.values_mut() {
        metrics.failure_rate = metrics.failures as f64 / metrics.files as f64;
    }
    Ok(EligibleText {
        paths,
        bytes,
        accounting,
        parse_failures_by_language,
    })
}

fn write_report(path: &Path, report: &BenchReport) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("benchmark output has no parent directory"))?;
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;
    let rendered =
        serde_json::to_string_pretty(report).context("failed to serialize benchmark report")?;
    fs::write(path, rendered).with_context(|| format!("failed to write {}", path.display()))
}

fn render_markdown(report: &BenchReport) -> String {
    let mut table = format!("Profile: {}\n\n", report.profile);
    table.push_str(
        "| target | goldens | HitFile@5 | Recall@10 | L1 cold ms | incremental ms | L1 peak RSS bytes | db/check-out | files | units | symbols | edges | caps hit | search p95 ms |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |\n",
    );
    for target in &report.targets {
        let golden = &target.quality.commit_derived;
        table.push_str(&format!(
            "| {} | {} | {} | {} | {:.2} | {:.2} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            target.name,
            golden.count,
            decimal(golden.hit_file_at_5),
            decimal(golden.recall_at_10),
            target.index.cold_ms,
            target.index.incremental_ms,
            target
                .index
                .peak_rss_bytes
                .map_or_else(|| "—".to_owned(), |value| value.to_string()),
            decimal(target.size.ratio),
            target.index.files,
            target.index.units,
            target.index.symbols,
            target.index.edges,
            cap_summary(target.robustness.cap_outcomes.as_ref()),
            decimal(target.queries.search.p95_ms),
        ));
    }
    table.push_str(
        "\n| target | tool | cold ms | warm p95 ms | sample target |\n| --- | --- | ---: | ---: | --- |\n",
    );
    for target in &report.targets {
        for (tool, latency, sample_target) in [
            (
                "search",
                Some(&target.queries.search),
                "commit-derived queries",
            ),
            ("map", Some(&target.queries.map), "2,000 tokens"),
            (
                "open",
                target.queries.open.as_ref().map(|tool| &tool.latency),
                target
                    .queries
                    .open
                    .as_ref()
                    .map_or("unavailable", |tool| tool.target.as_str()),
            ),
            (
                "symbol",
                target.queries.symbol.as_ref().map(|tool| &tool.latency),
                target
                    .queries
                    .symbol
                    .as_ref()
                    .map_or("unavailable", |tool| tool.target.as_str()),
            ),
            (
                "refs",
                target.queries.refs.as_ref().map(|tool| &tool.latency),
                target
                    .queries
                    .refs
                    .as_ref()
                    .map_or("unavailable", |tool| tool.target.as_str()),
            ),
            (
                "recent",
                target.queries.recent.as_ref().map(|tool| &tool.latency),
                target
                    .queries
                    .recent
                    .as_ref()
                    .map_or("unavailable", |tool| tool.target.as_str()),
            ),
            (
                "search (MCP stdio)",
                target
                    .queries
                    .mcp_search
                    .as_ref()
                    .map(|report| &report.search),
                "20 round trips",
            ),
        ] {
            table.push_str(&format!(
                "| {} | {} | {} | {} | {} |\n",
                target.name,
                tool,
                decimal(latency.and_then(|value| value.cold_ms)),
                decimal(latency.and_then(|value| value.p95_ms)),
                sample_target.replace('|', "/"),
            ));
        }
    }
    if report
        .targets
        .iter()
        .any(|target| target.model_index.is_some())
    {
        table.push_str("\n| target | model level | published | units/s | peak RSS bytes | covered/indexed | pending |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for target in &report.targets {
            if let Some(model) = &target.model_index {
                let covered = model.coverage["covered_units"].as_i64();
                let indexed = model.coverage["indexed_units"].as_i64();
                table.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {} | {} |\n",
                    target.name,
                    model.requested_level,
                    model.published_units,
                    decimal(model.throughput_units_per_second),
                    model
                        .peak_rss_bytes
                        .map_or_else(|| "—".to_owned(), |value| value.to_string()),
                    covered.zip(indexed).map_or_else(
                        || "—".to_owned(),
                        |(covered, indexed)| format!("{covered}/{indexed}")
                    ),
                    model
                        .pending_jobs
                        .map_or_else(|| "—".to_owned(), |value| value.to_string()),
                ));
            }
        }
    }
    if report
        .targets
        .iter()
        .any(|target| target.paired_search.is_some())
    {
        table.push_str("\n| target | lexical HitFile@5 | lexical Recall@10 | hybrid HitFile@5 | hybrid Recall@10 | reranked HitFile@5 | reranked Recall@10 | rerank HitFile delta | rerank Recall delta | hybrid p95 ms | reranked p95 ms |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for target in &report.targets {
            if let Some(paired) = &target.paired_search {
                let lexical_hit = paired_score(paired, "lexical", "hit_file_at_5");
                let lexical_recall = paired_score(paired, "lexical", "recall_at_10");
                let hybrid_hit = paired_score(paired, "hybrid", "hit_file_at_5");
                let hybrid_recall = paired_score(paired, "hybrid", "recall_at_10");
                let reranked_hit = paired_score(paired, "reranked", "hit_file_at_5");
                let reranked_recall = paired_score(paired, "reranked", "recall_at_10");
                table.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                    target.name,
                    decimal(lexical_hit),
                    decimal(lexical_recall),
                    decimal(hybrid_hit),
                    decimal(hybrid_recall),
                    decimal(reranked_hit),
                    decimal(reranked_recall),
                    decimal(option_delta(hybrid_hit, reranked_hit)),
                    decimal(option_delta(hybrid_recall, reranked_recall)),
                    decimal(
                        paired
                            .pointer("/latency/hybrid/p95_ms")
                            .and_then(Value::as_f64)
                    ),
                    decimal(
                        paired
                            .pointer("/latency/reranked/p95_ms")
                            .and_then(Value::as_f64)
                    ),
                ));
            }
        }
    }
    if report
        .targets
        .iter()
        .any(|target| target.composites.is_some())
    {
        table.push_str("\n| target | PRs reconstructed/detected | PR reconstruction rate |\n| --- | ---: | ---: |\n");
        for target in &report.targets {
            if let Some(composites) = &target.composites {
                let prs = &composites.pr_reconstruction;
                table.push_str(&format!(
                    "| {} | {}/{} | {} |\n",
                    target.name,
                    prs.reconstructed,
                    prs.detected,
                    decimal(prs.rate),
                ));
            }
        }
    }
    if report.targets.iter().any(|target| {
        target
            .robustness
            .parse_failures_by_language
            .as_ref()
            .is_some_and(|languages| !languages.is_empty())
    }) {
        table.push_str("\n| target | language | parsed files | parse failures | failure rate |\n| --- | --- | ---: | ---: | ---: |\n");
        for target in &report.targets {
            if let Some(languages) = &target.robustness.parse_failures_by_language {
                for (language, metrics) in languages {
                    table.push_str(&format!(
                        "| {} | {} | {} | {} | {} |\n",
                        target.name,
                        language,
                        metrics.files,
                        metrics.failures,
                        decimal(Some(metrics.failure_rate)),
                    ));
                }
            }
        }
    }
    if let Some(build) = &report.build {
        table.push_str(&format!(
            "\nBuild: {} (source {}, dirty {}), {} opt={}, features={}; binary {}.\n",
            build.commit,
            build.source_blake3,
            build.source_dirty,
            build.profile,
            build.opt_level,
            build.features,
            build.binary_blake3
        ));
    }
    for target in &report.targets {
        if let Some(provenance) = &target.provenance {
            table.push_str(&format!("\n{} pre-index idle: {:?}; load: {:?}; schema {:?}, page {} bytes, mmap {} bytes. Model pass: {}.\n",
                target.name, provenance.pre_index.idle, provenance.pre_index.load_average,
                provenance.schema_version, provenance.page_size, provenance.mmap_size,
                if target.model_index.is_some() { "recorded" } else { "not invoked (L1 only)" }));
        }
    }
    for target in &report.targets {
        if let Some(eligibility) = &target.quality.eligibility {
            table.push_str(&format!(
                "\n{}: raw {} queries / {} expected paths; eligible {} queries / {} expected paths. Eligible HitFile@5 {}, Recall@10 {}; {} rejected paths (reasons in JSON).\n",
                target.name, target.quality.commit_derived.count, eligibility.raw_expected_paths,
                eligibility.eligible.count, eligibility.eligible_expected_paths,
                decimal(eligibility.eligible.hit_file_at_5), decimal(eligibility.eligible.recall_at_10),
                eligibility.rejected_paths.len(),
            ));
        }
    }
    for target in &report.targets {
        if let Some(report) = &target.quality.handwritten {
            table.push_str(&format!(
                "\n{} handwritten goldens: {}.\n",
                target.name, report.review_status
            ));
            for (cohort, score) in &report.cohorts {
                table.push_str(&format!("- {}: {} cases, HitFile@5 {}, Recall@10 {}, HitSymbol@5 {}, no-answer empty-response rate {}.\n",
                    cohort, score.count, decimal(score.hit_file_at_5), decimal(score.recall_at_10),
                    decimal(score.hit_symbol_at_5), decimal(score.no_answer_empty_rate)));
            }
        }
    }
    for failure in &report.failures {
        table.push_str(&format!(
            "\nFAIL {} at {}: {}\n",
            failure.name, failure.stage, failure.error
        ));
    }
    table
}

fn cap_summary(outcomes: Option<&Value>) -> String {
    let Some(outcomes) = outcomes else {
        return "—".to_owned();
    };
    let mut hits = Vec::new();
    if outcomes
        .pointer("/caps/file_capacity_reached")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        hits.push("file_capacity".to_owned());
    }
    if let Some(count) = outcomes
        .pointer("/caps/truncated_files")
        .and_then(Value::as_u64)
        .filter(|count| *count > 0)
    {
        hits.push(format!("truncated_files={count}"));
    }
    if outcomes["graph_pending"].as_bool().unwrap_or(false) {
        hits.push("graph_pending".to_owned());
    }
    if let Some(count) = outcomes["l1_pending_files"]
        .as_u64()
        .filter(|count| *count > 0)
    {
        hits.push(format!("l1_pending_files={count}"));
    }
    if let Some(count) = outcomes["l1_deferred"]
        .as_array()
        .map(Vec::len)
        .filter(|count| *count > 0)
    {
        hits.push(format!("l1_deferred={count}"));
    }
    if hits.is_empty() {
        "none".to_owned()
    } else {
        hits.join(", ")
    }
}

fn paired_score(report: &Value, arm: &str, metric: &str) -> Option<f64> {
    let pointer = format!("/{arm}/commit_derived/{metric}");
    let mut values = report["scores"]
        .as_array()?
        .iter()
        .filter_map(|score| score.pointer(&pointer).and_then(Value::as_f64))
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    percentile(&values, 0.5)
}

fn diff_benchmarks(
    baseline_path: &Path,
    candidate_path: &Path,
    fail_on_regression: bool,
    json_output: bool,
) -> Result<()> {
    let baseline = read_report(baseline_path)?;
    let candidate = read_report(candidate_path)?;
    let baseline_targets = baseline
        .targets
        .iter()
        .map(|target| (target.name.as_str(), target))
        .collect::<BTreeMap<_, _>>();
    let candidate_targets = candidate
        .targets
        .iter()
        .map(|target| (target.name.as_str(), target))
        .collect::<BTreeMap<_, _>>();
    let mut comparisons = Vec::new();
    let mut failures = Vec::new();
    match (&baseline.build, &candidate.build) {
        (Some(before), Some(after)) if before.comparable(after) => {}
        (None, None) => {}
        _ => failures.push("incompatible or missing build metadata".to_owned()),
    }

    if let (Some(a), Some(b)) = (&baseline.machine.preflight, &candidate.machine.preflight)
        && (a.cpu_model != b.cpu_model
            || a.logical_cpus != b.logical_cpus
            || a.memory_total_kib != b.memory_total_kib
            || a.gpu_inventory != b.gpu_inventory)
    {
        failures.push("incompatible machine hardware or GPU driver".to_owned());
    }
    if baseline.machine.os != candidate.machine.os
        || baseline.machine.arch != candidate.machine.arch
    {
        failures.push("incompatible operating system or architecture".to_owned());
    }
    let compatible_environment = failures.is_empty();
    for failure in &baseline.failures {
        failures.push(format!(
            "{}: baseline target failed at {}",
            failure.name, failure.stage
        ));
    }
    for failure in &candidate.failures {
        failures.push(format!(
            "{}: candidate target failed at {}",
            failure.name, failure.stage
        ));
    }

    for (name, before) in baseline_targets {
        if !compatible_environment {
            continue;
        }
        let Some(after) = candidate_targets.get(name) else {
            failures.push(format!("{name}: target missing from candidate"));
            continue;
        };
        if before.size.accounting.as_ref().map(|value| &value.method)
            != after.size.accounting.as_ref().map(|value| &value.method)
        {
            failures.push(format!("{name}: incompatible footprint accounting methods"));
            continue;
        }
        if before
            .quality
            .eligibility
            .as_ref()
            .map(|value| (&value.method, &value.tests_globs))
            != after
                .quality
                .eligibility
                .as_ref()
                .map(|value| (&value.method, &value.tests_globs))
        {
            failures.push(format!(
                "{name}: incompatible golden eligibility methods or test globs"
            ));
            continue;
        }
        if before.queries.search.method != after.queries.search.method
            || before.queries.map.method != after.queries.map.method
        {
            failures.push(format!("{name}: incompatible query timing methods"));
            continue;
        }
        match (&before.provenance, &after.provenance) {
            (Some(a), Some(b))
                if a.head == b.head
                    && a.effective_config_blake3 == b.effective_config_blake3
                    && a.indexed_sources_blake3 == b.indexed_sources_blake3
                    && a.page_size == b.page_size
                    && a.mmap_size == b.mmap_size => {}
            (None, None) => {}
            _ => {
                failures.push(format!(
                    "{name}: incompatible target inputs or storage layout"
                ));
                continue;
            }
        }
        if before
            .quality
            .handwritten
            .as_ref()
            .map(|value| &value.source_hashes)
            != after
                .quality
                .handwritten
                .as_ref()
                .map(|value| &value.source_hashes)
        {
            failures.push(format!("{name}: handwritten golden inputs differ"));
            continue;
        }
        compare_target(before, after, &mut comparisons, &mut failures);
    }
    let markdown = render_diff_markdown(&comparisons, &failures);
    let response = json!({
        "baseline": baseline_path,
        "candidate": candidate_path,
        "comparisons": comparisons,
        "failures": failures,
        "pass": failures.is_empty(),
    });
    emit(json_output, timed_response(response), &markdown)?;
    if fail_on_regression && !failures.is_empty() {
        bail!("benchmark regression: {}", failures.join("; "));
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct TargetComparison {
    target: String,
    hit_file_at_5_delta: Option<f64>,
    recall_at_10_delta: Option<f64>,
    paired_quality_deltas: BTreeMap<String, Option<f64>>,
    search_p95_ratio: Option<f64>,
    cold_index_ratio: Option<f64>,
    footprint_ratio: Option<f64>,
    pr_reconstruction_rate_delta: Option<f64>,
}

fn compare_target(
    before: &TargetReport,
    after: &TargetReport,
    comparisons: &mut Vec<TargetComparison>,
    failures: &mut Vec<String>,
) {
    let before_golden = &before.quality.commit_derived;
    let after_golden = &after.quality.commit_derived;
    let hit_delta = option_delta(before_golden.hit_file_at_5, after_golden.hit_file_at_5);
    let recall_delta = option_delta(before_golden.recall_at_10, after_golden.recall_at_10);
    let search_ratio = option_ratio(before.queries.search.p95_ms, after.queries.search.p95_ms);
    let index_ratio = ratio(before.index.cold_ms, after.index.cold_ms);
    let footprint_ratio = option_ratio(before.size.ratio, after.size.ratio);
    let pr_delta = option_delta(
        before
            .composites
            .as_ref()
            .and_then(|metrics| metrics.pr_reconstruction.rate),
        after
            .composites
            .as_ref()
            .and_then(|metrics| metrics.pr_reconstruction.rate),
    );
    if hit_delta.is_some_and(|delta| delta < -0.02) {
        failures.push(format!(
            "{}: HitFile@5 dropped by more than 2 points",
            before.name
        ));
    }
    if recall_delta.is_some_and(|delta| delta < -0.02) {
        failures.push(format!(
            "{}: Recall@10 dropped by more than 2 points",
            before.name
        ));
    }
    if search_ratio.is_some_and(|ratio| ratio > 1.25) {
        failures.push(format!("{}: search p95 rose by more than 25%", before.name));
    }
    if index_ratio.is_some_and(|ratio| ratio > 1.25) {
        failures.push(format!(
            "{}: L1 cold time rose by more than 25%",
            before.name
        ));
    }
    if footprint_ratio.is_some_and(|ratio| ratio > 1.15) {
        failures.push(format!(
            "{}: footprint ratio rose by more than 15%",
            before.name
        ));
    }
    if let (Some(a), Some(b)) = (&before.quality.handwritten, &after.quality.handwritten) {
        for (cohort, previous) in &a.cohorts {
            if let Some(current) = b.cohorts.get(cohort) {
                if option_delta(previous.hit_file_at_5, current.hit_file_at_5)
                    .is_some_and(|delta| delta < -0.02)
                {
                    failures.push(format!(
                        "{}: handwritten {cohort} HitFile@5 dropped by more than 2 points",
                        before.name
                    ));
                }
                if option_delta(previous.recall_at_10, current.recall_at_10)
                    .is_some_and(|delta| delta < -0.02)
                {
                    failures.push(format!(
                        "{}: handwritten {cohort} Recall@10 dropped by more than 2 points",
                        before.name
                    ));
                }
            }
        }
    }
    let mut paired_quality_deltas = BTreeMap::new();
    match (&before.paired_search, &after.paired_search) {
        (Some(previous), Some(current)) => {
            for arm in ["lexical", "hybrid", "reranked"] {
                for metric in ["hit_file_at_5", "recall_at_10"] {
                    for (kind, previous_value, current_value) in [
                        (
                            "commit_derived",
                            paired_score(previous, arm, metric),
                            paired_score(current, arm, metric),
                        ),
                        (
                            "handwritten",
                            paired_handwritten_score(previous, arm, metric),
                            paired_handwritten_score(current, arm, metric),
                        ),
                    ] {
                        let key = format!("{kind}.{arm}.{metric}");
                        let delta = option_delta(previous_value, current_value);
                        if delta.is_some_and(|value| value < -0.02) {
                            failures.push(format!(
                                "{}: {kind} {arm} {metric} dropped by more than 2 points",
                                before.name
                            ));
                        }
                        paired_quality_deltas.insert(key, delta);
                    }
                }
            }
        }
        (None, None) => {}
        _ => failures.push(format!(
            "{}: paired-search evidence is missing from one report",
            before.name
        )),
    }
    if before
        .composites
        .as_ref()
        .and_then(|metrics| metrics.pr_reconstruction.rate)
        .is_some_and(|rate| rate >= 0.95)
        && after
            .composites
            .as_ref()
            .and_then(|metrics| metrics.pr_reconstruction.rate)
            .is_some_and(|rate| rate < 0.95)
    {
        failures.push(format!(
            "{}: PR reconstruction dropped below 95%",
            before.name
        ));
    }
    if after.robustness.panics > 0 {
        failures.push(format!("{}: candidate reports panics", before.name));
    }
    if matches!(before.name.as_str(), "fixtures-docs" | "tiny") && after.robustness.dead_jobs > 0 {
        failures.push(format!("{}: candidate reports dead jobs", before.name));
    }
    comparisons.push(TargetComparison {
        target: before.name.clone(),
        hit_file_at_5_delta: hit_delta,
        recall_at_10_delta: recall_delta,
        paired_quality_deltas,
        search_p95_ratio: search_ratio,
        cold_index_ratio: index_ratio,
        footprint_ratio,
        pr_reconstruction_rate_delta: pr_delta,
    });
}

fn paired_handwritten_score(report: &Value, arm: &str, metric: &str) -> Option<f64> {
    let pointer = format!("/handwritten/{arm}/cohorts/positive/{metric}");
    let mut values = report["scores"]
        .as_array()?
        .iter()
        .filter_map(|score| score.pointer(&pointer).and_then(Value::as_f64))
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    percentile(&values, 0.5)
}

fn read_report(path: &Path) -> Result<BenchReport> {
    let input = fs::read_to_string(path)
        .with_context(|| format!("failed to read benchmark report {}", path.display()))?;
    serde_json::from_str(&input)
        .with_context(|| format!("failed to parse benchmark report {}", path.display()))
}

fn render_diff_markdown(comparisons: &[TargetComparison], failures: &[String]) -> String {
    let mut table = String::from(
        "| target | Δ HitFile@5 | Δ Recall@10 | search p95 ratio | cold index ratio | footprint ratio | Δ PR reconstruction |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    );
    for comparison in comparisons {
        table.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} |\n",
            comparison.target,
            decimal(comparison.hit_file_at_5_delta),
            decimal(comparison.recall_at_10_delta),
            decimal(comparison.search_p95_ratio),
            decimal(comparison.cold_index_ratio),
            decimal(comparison.footprint_ratio),
            decimal(comparison.pr_reconstruction_rate_delta),
        ));
    }
    if comparisons
        .iter()
        .any(|comparison| !comparison.paired_quality_deltas.is_empty())
    {
        table.push_str("\n| target | paired quality metric | delta |\n| --- | --- | ---: |\n");
        for comparison in comparisons {
            for (metric, delta) in &comparison.paired_quality_deltas {
                table.push_str(&format!(
                    "| {} | {} | {} |\n",
                    comparison.target,
                    metric,
                    decimal(*delta),
                ));
            }
        }
    }
    if failures.is_empty() {
        table.push_str("\nPASS\n");
    } else {
        table.push_str("\nFAIL\n");
        for failure in failures {
            table.push_str("- ");
            table.push_str(failure);
            table.push('\n');
        }
    }
    table
}

/// Measures the public shared backend, including HEAD inspection and JSON serialization.
/// Repository connections and the runtime are retained outside individual measurements.
fn shared_search(
    context: &crate::search::service::Context,
    runtime: &tokio::runtime::Runtime,
    query: &str,
) -> Result<Value> {
    let value = runtime.block_on(context.search(
        crate::search::pipeline::Request {
            query: query.to_owned(),
            scope: Some("source".into()),
            k: 10,
            mode: crate::search::fusion::Mode::Lexical,
            exclude_ids: Vec::new(),
        },
        false,
    ))?;
    if value["index"]["degraded"] == true {
        bail!("shared lexical benchmark degraded; inspect query deadline and index status");
    }
    std::hint::black_box(serde_json::to_vec(&value)?);
    Ok(value)
}

// A timed map is a correctness sample too. Refuse incomplete/missing evidence
// before recording latency, retaining only static fields and numeric counters.
fn consume_map_sample(value: &Value) -> Result<()> {
    let index = &value["index"];
    if index["partial"].as_bool() != Some(false) || index["degraded"].as_bool() != Some(false) {
        let evidence = json!({
            "partial": index["partial"].as_bool(),
            "degraded": index["degraded"].as_bool(),
            "query_timeout": index["reason"].as_str() == Some("query_timeout"),
            "l1_pending_files": index["l1_pending_files"].as_u64(),
            "suppressed_files": index["suppressed_files"].as_u64(),
            "invalid_source_range": index["freshness_refusals"]["invalid_source_range"].as_u64(),
            "attempted_files": index["lazy_resync"]["attempted_files"].as_u64(),
            "published_files": index["lazy_resync"]["published_files"].as_u64(),
            "refused_files": index["lazy_resync"]["refused_files"].as_u64(),
        });
        bail!("shared map benchmark incomplete; index evidence: {evidence}");
    }
    std::hint::black_box(serde_json::to_vec(value)?);
    Ok(())
}

fn measure_queries(
    root: &Path,
    goldens: &[CommitGolden],
    settings: &crate::config::Config,
) -> Result<QueryMetrics> {
    let before_open_bytes = proc_resident_rss_bytes();
    let context = crate::search::service::Context::load(root)?;
    if context.config != *settings {
        bail!("configuration changed before query measurement");
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let after_open_bytes = proc_resident_rss_bytes();
    let search = repeated_latency(goldens.len(), |index| {
        shared_search(&context, &runtime, &goldens[index].query)?;
        Ok(())
    })?;
    let after_search_bytes = proc_resident_rss_bytes();
    let map = repeated_latency(MAP_SAMPLE_COUNT, |_| {
        let result = runtime.block_on(context.map(2000, &[]))?;
        consume_map_sample(&result)
    })?;
    let mut unavailable = BTreeMap::new();
    let source_target = context.store.searchable_source_paths()?.into_iter().next();
    let open = if let Some(target) = source_target {
        Some(ToolLatencyMetrics {
            latency: repeated_latency(TOOL_SAMPLE_COUNT, |_| {
                let value = runtime.block_on(context.open(&target, 0))?;
                consume_tool_sample(&value)
            })?,
            target,
        })
    } else {
        unavailable.insert("open".to_owned(), "no searchable source units".to_owned());
        None
    };
    let symbol_target = context
        .store
        .tool_symbols()?
        .into_iter()
        .next()
        .map(|symbol| {
            symbol
                .qualified
                .filter(|qualified| !qualified.is_empty())
                .unwrap_or(symbol.name)
        });
    let (symbol, refs) = if let Some(target) = symbol_target {
        let symbol = ToolLatencyMetrics {
            latency: repeated_latency(TOOL_SAMPLE_COUNT, |_| {
                let value = runtime.block_on(context.symbol(&target))?;
                consume_tool_sample(&value)
            })?,
            target: target.clone(),
        };
        let refs = ToolLatencyMetrics {
            latency: repeated_latency(TOOL_SAMPLE_COUNT, |_| {
                let value = runtime.block_on(context.refs(&target, Some(10), false))?;
                consume_tool_sample(&value)
            })?,
            target,
        };
        (Some(symbol), Some(refs))
    } else {
        unavailable.insert("symbol".to_owned(), "no indexed symbols".to_owned());
        unavailable.insert("refs".to_owned(), "no indexed symbols".to_owned());
        (None, None)
    };
    let recent = if context.config.commits.enabled
        && context.store.get_meta("history_status")?.is_some()
    {
        Some(ToolLatencyMetrics {
            latency: repeated_latency(TOOL_SAMPLE_COUNT, |_| {
                let value =
                    runtime.block_on(context.recent(crate::search::service::recent::Request {
                        k: Some(10),
                        ..Default::default()
                    }))?;
                consume_tool_sample(&value)
            })?,
            target: "unfiltered:k=10".to_owned(),
        })
    } else {
        unavailable.insert("recent".to_owned(), "history is not indexed".to_owned());
        None
    };
    let resident_memory = Some(ResidentMemory {
        method: "linux-vmrss-live-reader-snapshots-v1".to_owned(),
        before_open_bytes,
        after_open_bytes,
        after_search_bytes,
        after_map_bytes: proc_resident_rss_bytes(),
        page_size: context.store.footprint()?.page_size,
        mmap_size: context.store.mmap_size()?,
        cache_size: context.store.cache_size()?,
    });
    let mut cli = Vec::new();
    if !goldens.is_empty() {
        for index in 0..20 {
            let query = &goldens[index % goldens.len()].query;
            let (value, elapsed, _) = invoke_dkc_json(
                root,
                &[
                    "search".to_owned(),
                    query.clone(),
                    "-k".to_owned(),
                    "10".to_owned(),
                    "--scope".to_owned(),
                    "source".to_owned(),
                    "--mode".to_owned(),
                    "lexical".to_owned(),
                    "--no-rerank".to_owned(),
                    "--json".to_owned(),
                ],
            )?;
            cli.push(duration_ms(elapsed));
            let shared = shared_search(&context, &runtime, query)?;
            if !search_results_match(&value, &shared) {
                bail!("CLI and shared search results differ");
            }
        }
    }
    let mcp_search = if goldens.is_empty() {
        unavailable.insert(
            "mcp_search".to_owned(),
            "no commit-derived search queries".to_owned(),
        );
        None
    } else {
        let queries = goldens
            .iter()
            .map(|golden| golden.query.as_str())
            .collect::<Vec<_>>();
        Some(mcp::measure(root, &queries, first_query_p50(&search, 20))?)
    };
    let cli_search = (!cli.is_empty()).then(|| summarize_latencies(&cli));
    let cli_median_overhead_ms = cli_search
        .as_ref()
        .and_then(|round_trip| round_trip.p50_ms.zip(first_query_p50(&search, 20)))
        .map(|(round_trip, in_process)| round_trip - in_process);
    Ok(QueryMetrics {
        resident_memory,
        search,
        map,
        open,
        symbol,
        refs,
        recent,
        unavailable,
        cli_shared_results_match: (!cli.is_empty()).then_some(true),
        cli_search,
        cli_median_overhead_ms,
        mcp_search,
    })
}

fn consume_tool_sample(value: &Value) -> Result<()> {
    std::hint::black_box(serde_json::to_vec(value)?);
    Ok(())
}

/// Compares retrieval and freshness state across calls made at different times.
/// Numeric source ages advance with wall time; preserve their presence and type,
/// including null for an unavailable timestamp, while comparing all other fields.
fn search_results_match(cli: &Value, shared: &Value) -> bool {
    fn stable_hits(response: &Value) -> Value {
        let mut hits = response["hits"].clone();
        if let Some(hits) = hits.as_array_mut() {
            for hit in hits {
                if let Some(age) = hit.pointer_mut("/fresh/changed_ago_s")
                    && age.as_u64().is_some()
                {
                    *age = json!(0);
                }
            }
        }
        hits
    }
    cli["query"] == shared["query"] && stable_hits(cli) == stable_hits(shared)
}

fn repeated_latency(
    samples: usize,
    mut query: impl FnMut(usize) -> Result<()>,
) -> Result<LatencyMetrics> {
    let mut sets = Vec::new();
    let mut cold_ms = None;
    if samples > 0 {
        let started = Instant::now();
        query(0)?;
        cold_ms = Some(duration_ms(started.elapsed()));
        for _ in 0..3 {
            for index in 0..5 {
                query(index % samples)?;
            }
            let mut set = Vec::with_capacity(samples);
            for index in 0..samples {
                let started = Instant::now();
                query(index)?;
                set.push(duration_ms(started.elapsed()));
            }
            sets.push(set);
        }
    }
    Ok(summarize_latency_sets(
        sets,
        cold_ms,
        5,
        "shared-service-head-aware-median-of-three-v2",
    ))
}

fn summarize_latency_sets(
    sets: Vec<Vec<f64>>,
    cold_ms: Option<f64>,
    warmups_per_set: usize,
    method: &str,
) -> LatencyMetrics {
    let mut p50 = Vec::new();
    let mut p95 = Vec::new();
    for set in &sets {
        let summary = summarize_latencies(set);
        p50.extend(summary.p50_ms);
        p95.extend(summary.p95_ms);
    }
    p50.sort_by(f64::total_cmp);
    p95.sort_by(f64::total_cmp);
    LatencyMetrics {
        samples: sets.iter().map(Vec::len).sum(),
        cold_ms,
        p50_ms: percentile(&p50, 0.5),
        p95_ms: percentile(&p95, 0.5),
        method: method.to_owned(),
        warmups_per_set,
        sets_ms: sets,
    }
}

fn summarize_latencies(samples: &[f64]) -> LatencyMetrics {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    LatencyMetrics {
        samples: sorted.len(),
        cold_ms: None,
        p50_ms: percentile(&sorted, 0.5),
        p95_ms: percentile(&sorted, 0.95),
        method: "cli-process-round-trip-v1".to_owned(),
        warmups_per_set: 0,
        sets_ms: vec![samples.to_vec()],
    }
}

// Transport probes cycle through the first 20 golden queries. Their baseline
// must use that same query mix, including fixtures with fewer than 20 goldens;
// the full-corpus median can represent a different mix.
fn first_query_p50(search: &LatencyMetrics, count: usize) -> Option<f64> {
    if count == 0 || search.sets_ms.is_empty() || search.sets_ms.iter().any(Vec::is_empty) {
        return None;
    }
    summarize_latency_sets(
        search
            .sets_ms
            .iter()
            .map(|set| (0..count).map(|index| set[index % set.len()]).collect())
            .collect(),
        None,
        search.warmups_per_set,
        "matched-query-transport-baseline-v1",
    )
    .p50_ms
}

fn percentile(sorted: &[f64], fraction: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let index = (sorted.len() as f64 * fraction).ceil() as usize;
    sorted.get(index.saturating_sub(1)).copied()
}

fn value_u64(value: &Value, key: &str) -> u64 {
    value[key].as_u64().unwrap_or_default()
}

fn option_delta(before: Option<f64>, after: Option<f64>) -> Option<f64> {
    before.zip(after).map(|(before, after)| after - before)
}

fn option_ratio(before: Option<f64>, after: Option<f64>) -> Option<f64> {
    before
        .zip(after)
        .and_then(|(before, after)| ratio(before, after))
}

fn ratio(before: f64, after: f64) -> Option<f64> {
    (before > 0.0).then(|| after / before)
}

fn decimal(value: Option<f64>) -> String {
    value.map_or_else(|| "—".to_owned(), |value| format!("{value:.3}"))
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn resolve_manifest_path(manifest_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        manifest_dir.join(path)
    }
}

fn canonical_or_current(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        env::current_dir()
            .context("failed to read current directory")
            .map(|current| current.join(path))
    }
}

fn default_output_path() -> PathBuf {
    PathBuf::from("notes/bench").join(format!("{}-lexical.json", unix_timestamp()))
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_capture_cannot_clean_outside_its_scratch_repo() -> Result<()> {
        let scratch = tempfile::tempdir()?;
        let mut settings = crate::config::Config::default();
        ensure_capture_index_root(scratch.path(), &settings)?;
        settings.index.root = "../other-repo".to_owned();
        assert!(ensure_capture_index_root(scratch.path(), &settings).is_err());
        settings.index.root = scratch.path().join("outside").display().to_string();
        assert!(ensure_capture_index_root(scratch.path(), &settings).is_err());
        Ok(())
    }

    #[test]
    fn parse_failure_paths_stay_bounded_sorted_and_old_reports_decode() -> Result<()> {
        let old: ParseLanguageMetrics = serde_json::from_value(json!({
            "files": 1, "failures": 1, "failure_rate": 1.0
        }))?;
        assert!(old.failure_paths.is_none());
        assert!(old.failure_paths_omitted.is_none());

        let mut current = ParseLanguageMetrics {
            files: 0,
            failures: 0,
            failure_rate: 0.0,
            failure_paths: Some(BTreeSet::new()),
            failure_paths_omitted: Some(0),
        };
        for number in (0..PARSE_FAILURE_PATH_LIMIT + 4).rev() {
            current.observe(&format!("file-{number:03}.rs"), true);
        }
        let paths = current
            .failure_paths
            .as_ref()
            .ok_or_else(|| anyhow!("paths"))?;
        assert_eq!(paths.len(), PARSE_FAILURE_PATH_LIMIT);
        assert_eq!(paths.first().map(String::as_str), Some("file-000.rs"));
        assert_eq!(paths.last().map(String::as_str), Some("file-255.rs"));
        assert_eq!(current.failure_paths_omitted, Some(4));
        assert_eq!(current.failures, (PARSE_FAILURE_PATH_LIMIT + 4) as u64);
        Ok(())
    }

    #[test]
    fn repeated_latency_uses_the_median_set_percentiles() {
        let metrics = summarize_latency_sets(
            vec![vec![1.0, 2.0], vec![100.0, 200.0], vec![10.0, 20.0]],
            Some(0.5),
            0,
            "fixture",
        );
        assert_eq!(metrics.samples, 6);
        assert_eq!(metrics.cold_ms, Some(0.5));
        assert_eq!(metrics.p50_ms, Some(10.0));
        assert_eq!(metrics.p95_ms, Some(20.0));
        assert_eq!(metrics.method, "fixture");
    }

    #[test]
    fn transport_baseline_uses_the_same_first_twenty_queries() {
        let mut sets = Vec::new();
        for first in [10.0, 11.0, 12.0] {
            let mut set = vec![first; 20];
            set.extend(vec![100.0; 180]);
            sets.push(set);
        }
        let search = summarize_latency_sets(sets, None, 5, "fixture");
        assert_eq!(search.p50_ms, Some(100.0));
        assert_eq!(first_query_p50(&search, 20), Some(11.0));
        assert_eq!(first_query_p50(&search, 201), Some(100.0));
        let short =
            summarize_latency_sets(vec![vec![10.0], vec![11.0], vec![12.0]], None, 5, "fixture");
        assert_eq!(first_query_p50(&short, 20), Some(11.0));
    }

    #[test]
    fn paired_score_reports_the_median_measured_repetition() {
        let report = json!({"scores": [
            {"hybrid":{"commit_derived":{"hit_file_at_5":0.2}}},
            {"hybrid":{"commit_derived":{"hit_file_at_5":0.9}}},
            {"hybrid":{"commit_derived":{"hit_file_at_5":0.5}}}
        ]});
        assert_eq!(paired_score(&report, "hybrid", "hit_file_at_5"), Some(0.5));
    }

    #[test]
    fn cap_summary_keeps_release_relevant_outcomes_visible() {
        let outcomes = json!({
            "caps":{"file_capacity_reached":true,"truncated_files":2},
            "graph_pending":true,
            "l1_pending_files":3,
            "l1_deferred":["max_units_per_file"],
        });
        assert_eq!(
            cap_summary(Some(&outcomes)),
            "file_capacity, truncated_files=2, graph_pending, l1_pending_files=3, l1_deferred=1"
        );
        assert_eq!(cap_summary(Some(&json!({}))), "none");
        assert_eq!(cap_summary(None), "—");
    }

    #[test]
    fn model_queue_state_separates_embedding_work_and_retry_time() -> Result<()> {
        let state = queue_state(&json!({
            "observed_at": 100,
            "jobs": {"queued": 3, "leased": 1, "failed": 2, "dead": 0,
                "ready": 4, "deferred": 2, "next_ready_at": 105},
            "providers": {
                "embed:remote:one": {"jobs": {"queued": 2, "leased": 1, "failed": 1}},
                "llm:remote:one": {"jobs": {"queued": 1, "leased": 0, "failed": 1}}
            }
        }))?;
        assert_eq!(state.unfinished, 6);
        assert_eq!(state.ready, 4);
        assert_eq!(state.leased, 1);
        assert_eq!(state.next_ready_at, Some(105));
        assert_eq!(
            state.embedding_pending,
            BTreeMap::from([("embed:remote:one".to_owned(), 4)])
        );
        Ok(())
    }

    #[test]
    fn model_drain_counts_only_embedding_publications() -> Result<()> {
        assert_eq!(
            published_embedding_units(&json!({"providers": [
                {"provider": "embed:remote:one", "published": 63},
                {"provider": "llm:remote:one", "published": 7}
            ]}))?,
            63
        );
        assert!(
            queue_state(&json!({
                "observed_at": 100,
                "jobs": {"queued": -1, "leased": 0, "failed": 0, "ready": 0,
                    "next_ready_at": null},
                "providers": {}
            }))
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn commit_metrics_keep_pathless_hits_in_their_actual_rank_positions()
    -> Result<(), Box<dyn std::error::Error>> {
        let (root, store, config) = crate::queue::source::tests::fixture()?;
        let goldens = vec![CommitGolden {
            revision: "fixture".to_owned(),
            query: "example".to_owned(),
            expected_files: BTreeSet::from(["a.rs".to_owned()]),
        }];
        for position in [1, 5, 6, 10, 11] {
            // Composite hits can have no file. Their ranks must remain occupied.
            let mut paths = vec![String::new(); position - 1];
            paths.push("a.rs".to_owned());
            let result = evaluate_golden_paths(root.path(), &goldens, &config, &store, |_, _| {
                Ok(paths.clone())
            })?;
            assert_eq!(result.commit_derived.count, 1);
            assert_eq!(
                result.commit_derived.hit_file_at_5,
                Some(if position <= 5 { 1.0 } else { 0.0 }),
                "rank {position}"
            );
            assert_eq!(
                result.commit_derived.recall_at_10,
                Some(if position <= 10 { 1.0 } else { 0.0 }),
                "rank {position}"
            );
        }
        Ok(())
    }

    #[test]
    fn map_measurement_refuses_incomplete_samples_and_retains_safe_index_evidence() {
        for (partial, degraded) in [(true, false), (false, true), (true, true)] {
            let mut calls = 0;
            let result = repeated_latency(3, |_| {
                calls += 1;
                let sample = if calls == 1 {
                    json!({"index":{"partial":false,"degraded":false}})
                } else {
                    json!({"nodes":[{"path":"private-source-marker","text":"private-body-marker"}],
                        "index":{"partial":partial,"degraded":degraded,
                            "l1_pending_files":0,"suppressed_files":2,
                            "freshness_refusals":{"invalid_source_range":2},
                            "lazy_resync":{"attempted_files":2,"published_files":2,"refused_files":0}}})
                };
                consume_map_sample(&sample)
            });
            assert_eq!(calls, 2, "an invalid warmup must stop the measurement");
            let error = match result {
                Err(error) => error.to_string(),
                Ok(_) => panic!("incomplete maps must not produce latency metrics"),
            };
            assert!(error.starts_with("shared map benchmark incomplete; index evidence:"));
            assert!(error.contains("\"invalid_source_range\":2"));
            assert!(error.contains("\"published_files\":2"));
            assert!(!error.contains("private-source-marker"));
            assert!(!error.contains("private-body-marker"));
        }
        let timeout = consume_map_sample(&json!({"index": {
            "partial": true, "degraded": true, "reason": "query_timeout"
        }}))
        .unwrap_err()
        .to_string();
        assert!(timeout.contains("\"query_timeout\":true"));
        for index in [
            json!({"reason":"private-reason-marker"}),
            json!({"partial":"private-marker","degraded":false}),
        ] {
            let error = consume_map_sample(&json!({"index":index}))
                .unwrap_err()
                .to_string();
            assert!(!error.contains("private-marker"));
            assert!(!error.contains("private-reason-marker"));
        }
        assert!(
            repeated_latency(1, |_| consume_map_sample(
                &json!({"index":{"partial":false,"degraded":false}})
            ))
            .is_ok()
        );
    }

    #[test]
    fn cross_call_parity_allows_age_ticks_but_preserves_retrieval_and_freshness() {
        let cli = json!({"query":"needle", "hits":[{
            "id":"u_one", "path":"live.rs", "lines":[1,2], "score":0.25,
            "fresh":{"changed_ago_s":0,"l1":"current","state":"fresh"}
        }, {"id":"u_two", "fresh":{"changed_ago_s":null}}]});
        let mut shared = cli.clone();
        shared["hits"][0]["fresh"]["changed_ago_s"] = json!(1);
        assert!(search_results_match(&cli, &shared));
        for (pointer, value) in [
            ("/query", json!("other")),
            ("/hits/0/id", json!("u_changed")),
            ("/hits/0/path", json!("other.rs")),
            ("/hits/0/lines", json!([2, 3])),
            ("/hits/0/score", json!(0.25000000000000006)),
            ("/hits/0/fresh/l1", json!("pending")),
            ("/hits/0/fresh/state", json!("stale")),
            ("/hits/0/fresh/changed_ago_s", Value::Null),
            ("/hits/0/fresh/changed_ago_s", json!("1")),
            ("/hits/0/fresh/changed_ago_s", json!(-1)),
            ("/hits/1/fresh/changed_ago_s", json!(0)),
        ] {
            let mut changed = shared.clone();
            *changed.pointer_mut(pointer).unwrap() = value;
            assert!(!search_results_match(&cli, &changed), "{pointer}");
        }
        let mut missing = shared.clone();
        missing["hits"][0]["fresh"]
            .as_object_mut()
            .unwrap()
            .remove("changed_ago_s");
        assert!(!search_results_match(&cli, &missing));
        shared["hits"].as_array_mut().unwrap().reverse();
        assert!(!search_results_match(&cli, &shared));
    }

    #[test]
    fn ranking_scores_survive_json_round_trips_exactly() -> Result<()> {
        for lexical in 0..200 {
            for graph in 0..100 {
                let score = 1.0 / (61.0 + lexical as f64) + 0.3 / (61.0 + graph as f64);
                let original = json!({ "score": score });
                let encoded = serde_json::to_vec(&original)?;
                let decoded: Value = serde_json::from_slice(&encoded)?;
                assert_eq!(original, decoded, "lexical={lexical}, graph={graph}");
            }
        }
        Ok(())
    }

    #[test]
    fn panic_diagnostics_fail_even_with_successful_child_exit() -> Result<()> {
        let mut output = ProcessCommand::new("rustc").arg("--version").output()?;
        assert!(output.status.success());
        check_child_output(&["index".to_owned()], &output)?;
        output.stderr = b"thread worker panicked at a fixture location".to_vec();
        let error =
            check_child_output(&["index".to_owned()], &output).expect_err("panic must fail");
        let child = error
            .downcast_ref::<ChildFailure>()
            .expect("typed child failure");
        assert_eq!(child.panics, 1);
        assert_eq!(child.exit_code, Some(0));
        Ok(())
    }
}

#[cfg(test)]
#[path = "bench_publication_audit.rs"]
mod publication_audit;

#[cfg(test)]
#[path = "bench_owner_sample_audit.rs"]
mod owner_sample_audit;
