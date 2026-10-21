//! Finite, resumable corpus embedding preparation; v1 §8/§14 and T1.7.
use crate::{
    cli::{self, bench_metadata},
    config, models,
    queue::{self, host, priority::Recency, usage},
    store::{Store, UnitKind},
};
use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

/// Benchmark-only explicit bounds; this does not invent public queue defaults.
#[derive(clap::Args)]
pub(crate) struct Args {
    /// Configuration directory for an existing, completed L1 index.
    #[arg(long)]
    root: PathBuf,
    /// Atomically refreshed progress report; reruns preserve jobs and vectors.
    #[arg(long)]
    out: PathBuf,
    /// Per-provider daily ceiling, including existing charged usage.
    #[arg(long,value_parser=clap::value_parser!(i64).range(0..))]
    max_calls: i64,
    /// Per-provider daily token ceiling, including existing charged usage.
    #[arg(long,value_parser=clap::value_parser!(i64).range(0..))]
    max_tokens: i64,
    /// Timeout of each queued model attempt, independent of query deadlines.
    #[arg(long,value_parser=clap::value_parser!(u64).range(1..))]
    attempt_timeout_ms: u64,
    /// Lease heartbeat interval (must be below the 120-second lease).
    #[arg(long,value_parser=clap::value_parser!(u64).range(1..120000))]
    heartbeat_ms: u64,
    /// Maximum scheduling targets per provider pass.
    #[arg(long,value_parser=clap::value_parser!(u32).range(1..))]
    max_schedule: u32,
    /// Maximum job claims per provider pass.
    #[arg(long,value_parser=clap::value_parser!(u32).range(1..))]
    max_jobs: u32,
    /// Total catalogue/drain passes across all providers in this invocation.
    #[arg(long,value_parser=clap::value_parser!(u32).range(1..))]
    max_passes: u32,
    /// Existing verified model cache; otherwise uses DKC_MODELS_DIR/default cache.
    #[arg(long)]
    model_cache: Option<PathBuf>,
    #[arg(long)]
    json: bool,
}
#[derive(Serialize)]
struct Bounds {
    max_calls: i64,
    max_tokens: i64,
    attempt_timeout_ms: u64,
    heartbeat_ms: u64,
    max_schedule: u32,
    max_jobs: u32,
    max_passes: u32,
}
#[derive(Default, Serialize)]
struct Progress {
    provider: String,
    table: String,
    state: String,
    catalogue_finished: bool,
    passes: u32,
    visited: usize,
    scheduled_or_existing: usize,
    skipped: usize,
    claimed: usize,
    published: usize,
    discarded: usize,
    deferred: usize,
    dead: usize,
    lease_lost: usize,
    errors: usize,
}
impl Progress {
    fn record(&mut self, pass: &host::Report) -> bool {
        self.visited += pass.visited;
        self.scheduled_or_existing += pass.scheduled_or_existing;
        self.skipped += pass.skipped;
        let execution = &pass.execution;
        self.claimed += execution.claimed;
        self.published += execution.published;
        self.discarded += execution.discarded;
        self.deferred += execution.deferred;
        self.dead += execution.dead;
        self.lease_lost += execution.lease_lost;
        let errors = execution.errors.len()
            + usize::from(pass.retirement.error.is_some())
            + usize::from(pass.transition_error.is_some());
        self.errors += errors;
        errors > 0 || execution.dead > 0
    }
}
#[derive(Serialize)]
struct Report {
    format: &'static str,
    state: String,
    root: PathBuf,
    head: String,
    build: bench_metadata::BuildInfo,
    config_blake3: String,
    bounds: Bounds,
    started_at: u64,
    elapsed_ms: u128,
    passes: u32,
    before: queue::inspection::Snapshot,
    after: Option<queue::inspection::Snapshot>,
    transition: Option<crate::store::transitions::State>,
    jobs: std::collections::BTreeMap<String, i64>,
    providers: Vec<Progress>,
    error: Option<String>,
}
fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|v| i64::try_from(v.as_secs()).ok())
        .unwrap_or(-1)
}
/// Publishes a benchmark report atomically without clobbering foreign temporary files.
pub(super) fn save(path: &Path, report: &impl Serialize) -> Result<()> {
    let temp = path.with_extension(format!("{}.tmp", std::process::id()));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)
        .with_context(|| {
            format!(
                "failed to create progress temporary file {}",
                temp.display()
            )
        })?;
    let result = (|| {
        file.write_all(&serde_json::to_vec_pretty(report)?)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        fs::rename(&temp, path)?;
        Ok::<_, anyhow::Error>(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result.with_context(|| format!("failed to write embedding progress {}", path.display()))
}
fn checkpoint(args: &Args, store: &Store, report: &mut Report, started: Instant) -> Result<()> {
    report.after = Some(queue::inspection::read(store)?);
    report.jobs = store.job_state_counts()?;
    report.transition = store.read_snapshot(crate::store::transitions::read)??;
    report.elapsed_ms = started.elapsed().as_millis();
    save(&args.out, report)
}
fn authority(root: &Path, head: &str) -> Result<()> {
    if super::git_stdout(root, &["rev-parse", "HEAD"])?.trim() != head {
        bail!("HEAD changed during embedding exercise; re-index before resuming")
    }
    Ok(())
}

/// Prepares only existing L1 data with explicit bounds, retaining observable
/// progress after every finite pass. Non-quiescent outcomes exit nonzero.
pub(super) fn run(args: Args) -> Result<()> {
    let config_root = fs::canonicalize(&args.root)?;
    let settings = config::load(&config_root, None)?.config;
    let root = fs::canonicalize(config_root.join(&settings.index.root))?;
    if !cli::database_path(&root, &settings).is_file() {
        bail!("embedding exercise requires an existing L1 index; run dkc index first")
    }
    let store = Arc::new(cli::open_store(&root, &settings)?);
    if store.counts()?.l1_pending_files != 0
        || store.get_meta("graph_pending")?.as_deref() != Some("0")
    {
        bail!("L1 publication is pending; finish dkc index before embedding")
    }
    let head = super::git_stdout(&root, &["rev-parse", "HEAD"])?
        .trim()
        .to_owned();
    if store.get_meta("indexed_head")?.as_deref() != Some(head.as_str()) {
        bail!("indexed HEAD differs from current HEAD; run dkc index first")
    }
    args.bounds().validate(now())?;
    let before = queue::inspection::read(&store)?;
    let mut report = Report {
        format: "dkc-embedding-exercise-v1",
        state: "running".into(),
        root: root.clone(),
        head: head.clone(),
        build: bench_metadata::build_info()?,
        config_blake3: blake3::hash(toml::to_string(&settings)?.as_bytes())
            .to_hex()
            .to_string(),
        bounds: Bounds {
            max_calls: args.max_calls,
            max_tokens: args.max_tokens,
            attempt_timeout_ms: args.attempt_timeout_ms,
            heartbeat_ms: args.heartbeat_ms,
            max_schedule: args.max_schedule,
            max_jobs: args.max_jobs,
            max_passes: args.max_passes,
        },
        started_at: u64::try_from(now())?,
        elapsed_ms: 0,
        passes: 0,
        before,
        after: None,
        transition: None,
        jobs: store.job_state_counts()?,
        providers: Vec::new(),
        error: None,
    };
    let started = Instant::now();
    save(&args.out, &report)?;
    let result = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(execute(
            &args,
            &store,
            &config_root,
            settings,
            &mut report,
            started,
        ));
    if let Err(error) = result {
        report.state = "failed".into();
        if let Some(provider) = report.providers.last_mut() {
            provider.state = "failed".into();
        }
        report.error = Some(error.to_string());
    }
    checkpoint(&args, &store, &mut report, started)?;
    cli::emit(
        args.json,
        serde_json::to_value(&report)?,
        &format!(
            "embedding exercise: {} ({} passes); report {}",
            report.state,
            report.passes,
            args.out.display()
        ),
    )?;
    if report.state != "quiescent" {
        bail!(
            "embedding exercise stopped {}; progress retained in {}",
            report.state,
            args.out.display()
        )
    }
    Ok(())
}

impl Args {
    fn bounds(&self) -> host::Bounds {
        host::Bounds {
            budget: usage::Budget {
                calls: Some(self.max_calls),
                tokens: Some(self.max_tokens),
            },
            timeout: Duration::from_millis(self.attempt_timeout_ms),
            heartbeat: Duration::from_millis(self.heartbeat_ms),
            max_schedule: self.max_schedule as usize,
            max_jobs: self.max_jobs as usize,
        }
    }
}
fn eligible_kinds(store: &Store) -> Result<Vec<UnitKind>> {
    store.read_snapshot(|tx| -> Result<Vec<UnitKind>> {
        let mut query = tx.prepare("SELECT DISTINCT u.kind FROM units u JOIN files f ON f.id=u.file_id
            WHERE u.kind IN ('chunk','symbol','doc_section','config_key') AND u.ladder_level>=1 AND f.max_level>=2
            AND f.l1_pending=0 AND f.l1_fingerprint IS NOT NULL AND f.l1_fingerprint=f.l1_desired AND (f.l1_deferred_reason IS NULL OR f.l1_deferred_reason='max_pdf_pages') ORDER BY u.kind")?;
        let values = query.query_map([], |r| r.get::<_,String>(0))?.collect::<Result<Vec<_>,_>>()?;
        values.into_iter().map(|kind| match kind.as_str() {
            "chunk" => Ok(UnitKind::Chunk), "symbol" => Ok(UnitKind::Symbol),
            "doc_section" => Ok(UnitKind::DocSection), "config_key" => Ok(UnitKind::ConfigKey),
            _ => anyhow::bail!("unexpected catalogue kind"),
        }).collect()
    })?
}
async fn execute(
    args: &Args,
    store: &Arc<Store>,
    config_root: &Path,
    settings: config::Config,
    report: &mut Report,
    started: Instant,
) -> Result<()> {
    let cache = match &args.model_cache {
        Some(path) => path.clone(),
        None => models::registry::cache_root()?,
    };
    let kinds = eligible_kinds(store)?;
    let group = host::group::Group::prepare(
        Arc::clone(store),
        host::group::Settings {
            root: report.root.clone(),
            config_root: config_root.into(),
            config: settings,
            local_cache: cache,
        },
        &kinds,
        |_| Ok(args.bounds()),
        now(),
    )
    .await?;
    for entry in &group.entries {
        report.providers.push(Progress {
            provider: entry.provider.clone(),
            table: entry.host.table_slug.clone(),
            state: "running".into(),
            ..Default::default()
        });
        let index = report.providers.len() - 1;
        let mut cursor = entry.host.start_catalogue().await?;
        while !cursor.finished() && report.passes < args.max_passes {
            authority(&report.root, &report.head)?;
            report.passes += 1;
            report.providers[index].passes += 1;
            let pass = match entry
                .host
                .run_catalogue(&mut cursor, |_| Ok(Recency::ChangedFile), now)
                .await
            {
                Ok(pass) => pass,
                Err(error) => {
                    report.providers[index].visited += error.visited;
                    report.providers[index].scheduled_or_existing += error.scheduled_or_existing;
                    return Err(error.into());
                }
            };
            let failed = report.providers[index].record(&pass);
            report.providers[index].catalogue_finished = cursor.finished();
            checkpoint(args, store, report, started)?;
            eprintln!(
                "embedding pass={} provider={} published={} pending={}",
                report.passes,
                entry.provider,
                report.providers[index].published,
                report.after.as_ref().map_or(0, |s| s.l2.pending_jobs)
            );
            if failed {
                bail!("embedding pass failed; inspect dkc queue status and the progress report")
            }
        }
        if !cursor.finished() {
            report.providers[index].state = "bounded".into();
            report.state = "bounded".into();
            return Ok(());
        }
        report.providers[index].catalogue_finished = true;
        if report
            .after
            .as_ref()
            .is_some_and(|s| s.l2.pending_jobs == 0)
        {
            report.providers[index].state = "quiescent".into();
            continue;
        }
        while report.passes < args.max_passes {
            authority(&report.root, &report.head)?;
            report.passes += 1;
            report.providers[index].passes += 1;
            let drained = match entry.host.drain(1, now).await {
                Ok(drained) => drained,
                Err(failure) => {
                    for pass in &failure.progress.passes {
                        report.providers[index].record(pass);
                    }
                    return Err(failure.into());
                }
            };
            let mut failed = false;
            for pass in &drained.passes {
                failed |= report.providers[index].record(pass);
            }
            let state = if failed {
                "failed"
            } else {
                match drained.stopped {
                    host::DrainStop::Empty => "quiescent",
                    host::DrainStop::Waiting => "waiting",
                    host::DrainStop::PassLimit => "running",
                    host::DrainStop::Failed => "failed",
                }
            };
            report.providers[index].state = state.into();
            checkpoint(args, store, report, started)?;
            match state {
                "quiescent" => break,
                "waiting" | "failed" => {
                    report.state = state.into();
                    return Ok(());
                }
                _ => {}
            }
        }
        if report.providers[index].state != "quiescent" {
            report.providers[index].state = "bounded".into();
            report.state = "bounded".into();
            return Ok(());
        }
    }
    authority(&report.root, &report.head)?;
    report.state = "quiescent".into();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn report_writer_never_removes_an_existing_temporary_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("report.json");
        let temp = path.with_extension(format!("{}.tmp", std::process::id()));
        fs::write(&path, b"previous report").unwrap();
        fs::write(&temp, b"owned by another writer").unwrap();
        assert!(save(&path, &serde_json::json!({"state":"running"})).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"previous report");
        assert_eq!(fs::read(&temp).unwrap(), b"owned by another writer");
    }
}
