//! Command-line interface and output; see v1.md §10.2.
mod priority;

mod bench;
mod bench_edits;
mod bench_metadata;
mod doctor;
mod hooks;
mod index_embedding;
mod index_enrichment;
mod init;
mod models;
#[cfg(test)]
mod publication_tests;
mod queue;
mod queue_drain;
mod recent;

use std::{
    cell::Cell,
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use serde_json::{Value, json};

use crate::{
    config::{self, Config},
    search,
    store::{self, Store, StoreLimits},
    sync::{self, SyncLimits},
};

thread_local! {
    static COMMAND_STARTED: Cell<Option<Instant>> = const { Cell::new(None) };
}

#[derive(Parser)]
#[command(
    name = "dkc",
    version,
    about = "Hybrid repository search for IDE agents"
)]
struct Cli {
    /// Emits structured diagnostic logs to stderr.
    #[arg(long, global = true)]
    log_json: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Lists recent PRs, with uncovered commits as fallback.
    Recent(recent::Options),
    /// Selects a model profile, verifies remote setup and writes repository configuration.
    Init(init::Options),
    /// Indexes L1 and executes bounded embedding work for higher levels.
    Index {
        /// Maximum requested treatment level; level 1 skips model preparation.
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u8).range(1..=3))]
        level: u8,
        /// Skips model preparation, embedding scheduling and queue execution.
        #[arg(long)]
        no_queue: bool,
        #[arg(long)]
        full: bool,
        /// Updates L1 only; never prepares models or runs the queue.
        #[arg(long, conflicts_with_all = ["full", "rebuild_fts", "compact", "shrink", "level"])]
        quick: bool,
        #[arg(long)]
        rebuild_fts: bool,
        /// Refreshes vectors whose folder descriptions changed, at low priority.
        #[arg(long, conflicts_with_all = ["quick", "no_queue", "shrink"])]
        refresh_context: bool,
        #[arg(long)]
        compact: bool,
        #[arg(long)]
        shrink: bool,
        #[arg(long)]
        json: bool,
    },
    /// Serves repository tools over MCP and watches source changes.
    Serve {
        #[arg(long, conflicts_with = "http")]
        stdio: bool,
        #[arg(long, num_args = 0..=1, default_missing_value = "")]
        http: Option<String>,
        /// Uses metadata polling when native filesystem events are unavailable.
        #[arg(long)]
        poll: bool,
    },
    /// Watches repository changes and runs the queue without serving MCP.
    Watch {
        /// Uses metadata polling instead of native filesystem events.
        #[arg(long)]
        poll: bool,
        /// Emits watcher readiness and completed batches as JSON lines.
        #[arg(long)]
        json: bool,
    },
    /// Installs or removes opt-in Git hooks without replacing existing scripts.
    Hooks {
        #[command(subcommand)]
        command: hooks::Command,
    },
    /// Searches indexed content with explicit fallback for unavailable dense work.
    Search {
        query: String,
        #[arg(short = 'k')]
        k: Option<usize>,
        #[arg(long, default_value = "hybrid", value_parser = ["lexical", "dense", "hybrid"])]
        mode: String,
        #[arg(long)]
        no_rerank: bool,
        #[arg(long)]
        scope: Vec<String>,
        #[arg(long)]
        json: bool,
        #[arg(long)]
        verbose: bool,
    },
    /// Opens a public unit id or a repository-relative path with an optional range.
    Open {
        target: String,
        /// Surrounding source lines to include around the containing symbol.
        #[arg(long, default_value_t = 0)]
        context: u64,
        #[arg(long)]
        json: bool,
    },
    /// Renders a PageRank-ordered repository tree with top symbol signatures.
    Map {
        #[arg(long, default_value_t = 2_000)]
        budget: usize,
        #[arg(long, value_delimiter = ',', num_args = 1..)]
        focus: Vec<String>,
        #[arg(long)]
        json: bool,
    },
    /// Looks up exact and trigram-matched symbols with approximate graph metadata.
    Symbol {
        target: String,
        #[arg(long)]
        json: bool,
    },
    /// Lists approximate incoming references grouped by source file.
    Refs {
        target: String,
        #[arg(short = 'k', long)]
        k: Option<usize>,
        #[arg(long)]
        tests_only: bool,
        #[arg(long)]
        json: bool,
    },
    /// Shows persistent index counts, cap state, and database footprint.
    Status {
        #[arg(long)]
        json: bool,
    },
    /// Removes only the Deepkit SQLite database and its explicit WAL sidecars.
    Clean {
        #[arg(long)]
        json: bool,
    },
    /// Checks the configured local reranker's verified artifact and relevance smoke test.
    Doctor {
        /// Also checks the installed nightly user timer, its last firing, and user lingering.
        #[arg(long)]
        bench: bool,
        #[arg(long)]
        json: bool,
    },
    /// Inspects pinned local models and verifies cached artifacts.
    Models {
        #[command(subcommand)]
        command: models::ModelsCommand,
    },
    /// Inspects persistent background jobs and usage.
    Queue {
        #[command(subcommand)]
        command: queue::QueueCommand,
    },
    /// Measures indexed repositories and compares stored benchmark reports.
    Bench {
        #[command(subcommand)]
        command: bench::BenchCommand,
    },
}

/// Parses and dispatches phase-0 commands from the complete process argument vector.
pub(crate) fn run(arguments: impl IntoIterator<Item = OsString>) -> Result<()> {
    let cli = match Cli::try_parse_from(arguments) {
        Ok(cli) => cli,
        Err(error) if error.exit_code() == 0 => {
            error.print()?;
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    COMMAND_STARTED.with(|started| started.set(Some(Instant::now())));
    initialize_logging(cli.log_json);
    match cli.command {
        Command::Recent(options) => recent::run(options),
        Command::Init(options) => init::run(options),
        Command::Index {
            level,
            full,
            quick,
            no_queue,
            rebuild_fts,
            refresh_context,
            compact,
            shrink,
            json,
        } => index(
            full,
            IndexQueue {
                disabled: no_queue || quick,
                refresh_context,
            },
            rebuild_fts,
            compact,
            shrink,
            if quick { 1 } else { level },
            json,
        ),
        Command::Hooks { command } => hooks::run(command),
        Command::Serve {
            stdio: _,
            http,
            poll,
        } => {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            let result = runtime.block_on(crate::serve::run(current_dir()?, http, poll));
            // A blocked stdin read or cooperative SQLite/native task must not
            // keep the process alive after its transport has been cancelled.
            runtime.shutdown_timeout(std::time::Duration::from_secs(1));
            result?;
            Ok(())
        }
        Command::Watch { poll, json } => {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            let result = runtime.block_on(crate::serve::watch(current_dir()?, poll, json));
            runtime.shutdown_timeout(std::time::Duration::from_secs(1));
            result?;
            Ok(())
        }
        Command::Search {
            query,
            k,
            mode,
            no_rerank,
            scope,
            json,
            verbose,
        } => search(&query, k, &mode, no_rerank, scope, json, verbose),
        Command::Open {
            target,
            context,
            json,
        } => open(&target, context, json),
        Command::Map {
            budget,
            focus,
            json,
        } => map(budget, &focus, json),
        Command::Symbol { target, json } => symbol(&target, json),
        Command::Refs {
            target,
            k,
            tests_only,
            json,
        } => refs(&target, k, tests_only, json),
        Command::Status { json } => status(json),
        Command::Clean { json } => clean(json),
        Command::Models { command } => models::run(command),
        Command::Doctor { bench, json } => doctor::run(json, bench),
        Command::Queue { command } => queue::run(command),
        Command::Bench { command } => bench::run(command),
    }
}

/// Reports capacity failures without retrying writes or fabricating coverage
/// counts. Nonzero exit and the original contextual stderr error are preserved.
#[derive(Clone, Copy)]
struct IndexQueue {
    disabled: bool,
    refresh_context: bool,
}

fn index(
    full: bool,
    queue: IndexQueue,
    rebuild_fts: bool,
    compact: bool,
    shrink: bool,
    level: u8,
    json_output: bool,
) -> Result<()> {
    let result = index_run(
        full,
        queue,
        rebuild_fts,
        compact,
        shrink,
        level,
        json_output,
    );
    if let Err(error) = &result
        && let Some((code, message, hint)) = index_capacity_error(error)
    {
        let mut output = json!({
            "index":{"state":"partial","degraded":true,"retry_required":true},
            "error":{"code":code,"message":message,"hint":hint},
        });
        let mut plain = format!("index partial: {message}; {hint}");
        if let Some(progress) = error.downcast_ref::<IndexProgress>() {
            output["shrink"] = progress.shrink.clone();
            output["index_priority"] = progress.priority.clone();
            plain.push_str(&format!(
                "; shrink: {} ({})",
                progress.shrink["state"].as_str().unwrap_or("unknown"),
                progress.shrink["reason"].as_str().unwrap_or("unknown")
            ));
        }
        emit(json_output, output, &plain)?;
    }
    result
}

/// Retains committed shrink progress even when later metadata or indexing is refused.
#[derive(Debug)]
struct IndexProgress {
    shrink: Value,
    priority: Value,
}
impl std::fmt::Display for IndexProgress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("index stopped after an explicit shrink pass")
    }
}

fn index_capacity_error(
    error: &anyhow::Error,
) -> Option<(&'static str, &'static str, &'static str)> {
    for source in error.chain() {
        let store_error = source.downcast_ref::<store::Error>().or_else(|| {
            match source.downcast_ref::<sync::Error>().or_else(|| {
                // Boxed per-file errors retain their concrete wrapper in source().
                // Inspect it before transparent sync errors forward past Store.
                source.downcast_ref::<Box<sync::Error>>().map(Box::as_ref)
            }) {
                Some(sync::Error::Store(error)) => Some(error),
                _ => None,
            }
        });
        if let Some(error) = store_error {
            match error {
                store::Error::FileCapacity { .. } => {
                    return Some((
                        "file_capacity",
                        "max_files stopped catalogue growth; committed data is retained",
                        "raise limits.max_files or review ignores, then index again",
                    ));
                }
                store::Error::WalCapacity { .. } => {
                    return Some((
                        "wal_capacity",
                        "WAL capacity stopped indexing; committed data is retained",
                        "allow readers to finish or raise limits.wal_max_bytes, then index again",
                    ));
                }
                store::Error::DatabaseGrowthLimit => {
                    return Some((
                        "database_limit",
                        "the main-file page limit prevented index initialization or admission",
                        "check limits.db_max_bytes and index again",
                    ));
                }
                store::Error::Sqlite(error)
                    if error.sqlite_error_code() == Some(rusqlite::ErrorCode::DiskFull) =>
                {
                    return Some((
                        "storage_full",
                        "storage exhaustion stopped indexing; committed data is retained",
                        "raise limits.db_max_bytes or free disk space, then index again",
                    ));
                }
                _ => {}
            }
        }
    }
    None
}

fn index_run(
    mut full: bool,
    queue: IndexQueue,
    rebuild_fts: bool,
    compact: bool,
    shrink: bool,
    level: u8,
    json_output: bool,
) -> Result<()> {
    let no_queue = queue.disabled;
    anyhow::ensure!(
        !queue.refresh_context || level > 1,
        "--refresh-context requires --level 2 or 3"
    );
    tracing::debug!(full, rebuild_fts, compact, shrink, "indexing repository");
    if rebuild_fts {
        full = true;
    }
    let (root, settings) = workspace()?;
    let index_priority = priority::apply(settings.limits.index_nice);
    let store = std::sync::Arc::new(open_store(&root, &settings)?);
    // Free permitted derived storage before catalogue/control writes can hit a
    // newly lowered cap. A second pass accounts for any L1 changes below.
    let mut shrink_report: Option<store::ShrinkReport> =
        if shrink { Some(store.shrink()?) } else { None };
    let result = (|| -> Result<()> {
        store.set_meta("index_priority", &serde_json::to_string(&index_priority)?)?;
        let previous_verification = store
            .get_meta("verified_local_model")?
            .and_then(|value| serde_json::from_str(&value).ok());
        let preparation = crate::models::setup::Preparation::start(
            &settings,
            if no_queue || shrink { 1 } else { level },
            previous_verification,
        );
        store.set_meta(
            "model_preparation",
            &serde_json::to_string(preparation.initial())?,
        )?;
        if !json_output && preparation.initial().state == "pending" {
            eprintln!("Preparing local model in parallel with L1 indexing");
        }
        let indexing = sync::index::run(
            sync::index::Input {
                config_root: &current_dir()?,
                root: &root,
                settings: &settings,
                store: &store,
                full,
                work: Default::default(),
            },
            |_| {},
        );
        let report = indexing?;
        let sync::index::Report {
            publication,
            scan,
            unit_count,
            truncated_files,
            history,
            prs,
            histories,
            modules,
            linker,
        } = report;
        let changes = &publication.changes;
        if compact {
            store.compact()?;
        }
        let enrichment = index_enrichment::run(
            std::sync::Arc::clone(&store),
            &root,
            &current_dir()?,
            &settings,
            level >= 3 && !no_queue && !shrink,
        )?;
        if !json_output && preparation.initial().state == "pending" {
            eprintln!("L1 published; waiting for local model preparation");
        }
        let mut model_preparation = if enrichment.state == "cancelled" {
            preparation.detach()
        } else {
            preparation.finish()
        };
        if no_queue {
            model_preparation.detail = Some("--no-queue skips model preparation".to_owned());
        }
        if let Some(verified) = &model_preparation.verified
            && store.l1_generation()? == publication.generation
        {
            store.set_meta("verified_local_model", &serde_json::to_string(verified)?)?;
        }
        store.set_meta(
            "model_preparation",
            &serde_json::to_string(&model_preparation)?,
        )?;
        let deferred = deferred_l1_json(&store)?;
        let deferred_count = store.l1_deferred_files()?.len();
        let embedding = index_embedding::run(
            std::sync::Arc::clone(&store),
            &root,
            &current_dir()?,
            &settings,
            level > 1
                && !no_queue
                && !shrink
                && model_preparation.state != "unavailable"
                && enrichment.state != "cancelled",
            queue.refresh_context,
        )?;
        if let Some(previous) = shrink_report.as_ref() {
            let mut final_report = store.shrink()?;
            final_report.before_bytes = previous.before_bytes;
            final_report.summaries_removed += previous.summaries_removed;
            final_report.page_sections_removed += previous.page_sections_removed;
            final_report.vectors_removed += previous.vectors_removed;
            shrink_report = Some(final_report);
            store.set_meta("last_shrink", &serde_json::to_string(&shrink_report)?)?;
        }
        // Cancellation must not start another database pass or wait on a worker
        // still leaving a blocking operation. The L1 report was captured above.
        let maintenance = if embedding.state == "cancelled" || enrichment.state == "cancelled" {
            None
        } else {
            Some(store.maintain()?)
        };
        let message = json!({
            "files_added": changes.added.len(),
            "files_modified": changes.modified.len(),
            "files_deleted": changes.deleted.len(),
            "files_renamed": changes.renamed.len(),
            "units_indexed": unit_count,
            "history": history,
            "prs": prs,
            "histories": histories, "modules": modules,
            "linker": linker,
            "maintenance": maintenance,
            "requested_level": level,
            "model_preparation": model_preparation,
            "embedding": embedding,
            "enrichment": enrichment,
            "higher_levels": "text and image summaries and scanned-page enrichment supported",
            "l1_deferred": deferred,
            "truncated_files": truncated_files,
            "status": scan.status.messages,
            "shrink": shrink_report,
            "index_priority": index_priority,
        });
        emit(
            json_output,
            message,
            &format!(
                "indexed {unit_count} L1 units; {} files deferred by caps; embedding: {}; summaries: {}; model preparation: {}{}{}",
                deferred_count,
                embedding.state,
                enrichment.state,
                model_preparation.state,
                model_preparation
                    .detail
                    .as_ref()
                    .map(|detail| format!(" ({detail})"))
                    .unwrap_or_default(),
                shrink_report
                    .as_ref()
                    .map_or_else(String::new, |report| format!(
                        "; shrink: {} ({})",
                        report.state, report.reason
                    )),
            ),
        )?;
        if enrichment.state == "failed" || enrichment.state == "cancelled" {
            bail!(
                "summary pass {}; committed output retained; inspect queue status",
                enrichment.state
            )
        }
        if embedding.state == "failed" {
            bail!("embedding pass failed; inspect dkc queue status and retry index")
        }
        if embedding.state == "cancelled" {
            bail!(
                "embedding pass interrupted; committed output retained; retry index after lease expiry"
            )
        }
        Ok(())
    })();
    result.map_err(|error| match &shrink_report {
        Some(report) => error.context(IndexProgress {
            shrink: json!(report),
            priority: json!(index_priority),
        }),
        None => error,
    })
}

fn search(
    query: &str,
    k: Option<usize>,
    mode: &str,
    no_rerank: bool,
    scope: Vec<String>,
    json_output: bool,
    verbose: bool,
) -> Result<()> {
    let context = search::service::Context::load(&current_dir()?)?;
    let mode = match mode {
        "lexical" => search::fusion::Mode::Lexical,
        "dense" => search::fusion::Mode::Dense,
        _ => search::fusion::Mode::Hybrid,
    };
    let request = search::pipeline::Request {
        query: query.into(),
        k: k.unwrap_or(context.config.search.default_k as usize),
        scope: match scope.as_slice() {
            [] => None,
            [one] => Some(one.clone().into()),
            _ => Some(crate::store::SearchScope::Globs(scope)),
        },
        mode,
        exclude_ids: Vec::new(),
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let response = runtime.block_on(context.search_with_rerank(request, verbose, !no_rerank))?;
    if json_output {
        outln!("{}", timed_response(response));
    } else if let Some(hits) = response["hits"].as_array() {
        if hits.is_empty() {
            if response["index"]["partial"] == true {
                outln!("no verified results (search incomplete)");
            } else {
                outln!("no results");
            }
        }
        for hit in hits {
            outln!(
                "{}:{} {}",
                hit["path"].as_str().unwrap_or("<repository>"),
                hit["lines"][0],
                hit["snippet"].as_str().unwrap_or("")
            );
            if verbose {
                outln!("{hit}");
            }
        }
        if response["index"]["reason"] == "query_timeout" {
            eprintln!("search incomplete: query deadline exceeded");
        } else if response["index"]["partial"] == true {
            eprintln!("search incomplete: unverifiable source locations suppressed");
        } else if response["index"]["degraded"].as_bool() == Some(true) {
            eprintln!("search degraded: requested model work is unavailable");
        }
    }
    Ok(())
}
fn open(target: &str, context: u64, json_output: bool) -> Result<()> {
    let service = search::service::Context::load(&current_dir()?)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let response = match runtime.block_on(service.open(target, context)) {
        Ok(response) => response,
        Err(error) => {
            if json_output {
                outln!(
                    "{}",
                    json!({"error": {"code": error.code(), "message": error.message(), "hint": error.hint()}})
                );
            }
            return Err(error.into());
        }
    };
    let plain = response["text"].as_str().unwrap_or("").to_owned();
    emit(json_output, response, &plain)?;
    Ok(())
}
fn map(budget: usize, focus: &[String], json_output: bool) -> Result<()> {
    let context = search::service::Context::load(&current_dir()?)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let response = runtime.block_on(context.map(budget, focus))?;
    if !json_output && response["index"]["partial"] == true {
        eprintln!("map incomplete: unverifiable files omitted or query deadline exceeded");
    }
    let plain = response["tree"].as_str().unwrap_or("").to_owned();
    emit(json_output, response, &plain)?;
    Ok(())
}
fn symbol(target: &str, json_output: bool) -> Result<()> {
    let context = search::service::Context::load(&current_dir()?)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let response = match runtime.block_on(context.symbol(target)) {
        Ok(response) => response,
        Err(error) => {
            if json_output {
                outln!(
                    "{}",
                    json!({"error": {
                        "code": error.code(), "message": error.message(), "hint": error.hint()
                    }})
                );
            }
            return Err(error.into());
        }
    };
    if json_output {
        outln!("{}", timed_response(response));
    } else if let Some(symbols) = response.as_array() {
        if symbols.is_empty() {
            outln!("no symbols");
        }
        for symbol in symbols {
            outln!(
                "{} {}:{} {}",
                symbol["id"].as_str().unwrap_or(""),
                symbol["path"].as_str().unwrap_or(""),
                symbol["lines"][0],
                symbol["signature"].as_str().unwrap_or("")
            );
        }
    }
    Ok(())
}
fn refs(target: &str, k: Option<usize>, tests_only: bool, json_output: bool) -> Result<()> {
    let context = search::service::Context::load(&current_dir()?)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let response = runtime.block_on(context.refs(target, k, tests_only))?;
    if json_output {
        outln!("{}", timed_response(response));
    } else {
        if response["index"]["partial"] == true {
            outln!(
                "reference results incomplete: {}",
                if response["index"]["reason"] == "query_timeout" {
                    "query deadline exceeded"
                } else {
                    "unverifiable source locations suppressed"
                }
            );
        } else if response["total"].as_u64() == Some(0) {
            outln!("no references");
        }
        if let Some(groups) = response["groups"].as_array() {
            for group in groups {
                outln!(
                    "{}:{} {}",
                    group["path"].as_str().unwrap_or(""),
                    group["lines"],
                    group["kind"].as_str().unwrap_or("")
                );
            }
        }
    }
    Ok(())
}

fn deferred_l1_json(store: &Store) -> Result<Value> {
    Ok(json!(store.l1_deferred_files()?.iter().map(|file| json!({
        "path": file.path, "reason": file.reason, "effective_level": 0, "retained_snapshot": file.retained_snapshot,
    })).collect::<Vec<_>>()))
}

fn status(json_output: bool) -> Result<()> {
    let context = search::service::Context::load(&current_dir()?)?;
    let response = context.status()?;
    let plain = context.status_text()?;
    emit(json_output, response, &plain)?;
    Ok(())
}

fn clean(json_output: bool) -> Result<()> {
    tracing::debug!("removing Deepkit database files");
    let (root, settings) = workspace()?;
    let removed = store::path::prepare(&root, Path::new(&settings.store.path))?.remove()?;
    emit(
        json_output,
        json!({"removed": removed}),
        "removed Deepkit database files",
    )?;
    Ok(())
}

fn workspace() -> Result<(PathBuf, Config)> {
    let current = current_dir()?;
    let loaded = config::load(&current, None)?;
    let configured = PathBuf::from(&loaded.config.index.root);
    let root = if configured.is_absolute() {
        configured
    } else {
        current.join(configured)
    };
    let root = fs::canonicalize(&root)
        .with_context(|| format!("failed to resolve repository root {}", root.display()))?;
    Ok((root, loaded.config))
}

fn current_dir() -> Result<PathBuf> {
    env::current_dir().context("failed to read current directory")
}

fn open_store(root: &Path, settings: &Config) -> Result<Store> {
    let limits = StoreLimits {
        db_max_bytes: settings.limits.db_max_bytes()?.0,
        wal_max_bytes: settings.limits.wal_max_bytes()?.0,
        max_l2_units: settings.limits.max_l2_units,
        max_l3_units: settings.limits.max_l3_units,
    };
    Ok(store::open_repository(
        root,
        Path::new(&settings.store.path),
        limits,
    )?)
}

fn database_path(root: &Path, settings: &Config) -> PathBuf {
    let configured = PathBuf::from(&settings.store.path);
    if configured.is_absolute() {
        configured
    } else {
        root.join(configured)
    }
}

fn sync_limits(settings: &Config) -> Result<SyncLimits> {
    Ok(sync::index::limits(settings)?)
}

fn emit(json_output: bool, value: serde_json::Value, plain: &str) -> std::io::Result<()> {
    if json_output {
        outln!("{}", timed_response(value));
    } else {
        outln!("{plain}");
    }
    Ok(())
}

fn initialize_logging(log_json: bool) {
    use tracing_subscriber::prelude::*;
    let filter = tracing_subscriber::EnvFilter::try_from_env("DKC_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    // rmcp logs full tool arguments at debug level, including submitted credentials.
    // An independent filter cannot be overridden by a more-specific DKC_LOG directive.
    let safe =
        tracing_subscriber::filter::filter_fn(|metadata| !metadata.target().starts_with("rmcp"));
    if log_json {
        let _ = tracing_subscriber::registry()
            .with(filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .json()
                    .with_writer(std::io::stderr)
                    .with_target(false)
                    .with_filter(safe),
            )
            .try_init();
    } else {
        let _ = tracing_subscriber::registry()
            .with(filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .with_writer(std::io::stderr)
                    .with_target(false)
                    .with_filter(safe),
            )
            .try_init();
    }
}

fn timed_response(mut value: Value) -> Value {
    if let Value::Object(object) = &mut value {
        object.insert("took_ms".to_owned(), json!(command_took_ms()));
    }
    value
}

fn command_took_ms() -> u128 {
    COMMAND_STARTED.with(|started| {
        started
            .get()
            .map_or(0, |started| started.elapsed().as_millis())
    })
}
