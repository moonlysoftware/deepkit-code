//! Fixed fifty-candidate admitted scorer measurement; v1 §§8.2, 14, 15.
use crate::{
    cli,
    config::{self, Config},
    models,
    queue::usage,
    search::rerank,
    store::{self, Store, StoredUnit},
};
use anyhow::{Result, bail};
use serde::Serialize;
use std::{
    collections::BTreeSet,
    fs,
    io::{Read, Seek, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::time::Instant;

/// Explicit fixed-corpus inputs and admission ceilings; never rewrites configuration.
#[derive(clap::Args)]
pub(crate) struct Args {
    #[arg(long)]
    root: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// UTF-8 query file; report retains only its digest.
    #[arg(long)]
    query_file: PathBuf,
    /// JSON array of exactly 50 distinct existing unit IDs, in scoring order.
    #[arg(long)]
    units_file: PathBuf,
    #[arg(long)]
    model_cache: Option<PathBuf>,
    /// Per-provider daily ceilings include previous usage and warmups.
    #[arg(long, value_parser=clap::value_parser!(i64).range(0..))]
    max_calls: i64,
    #[arg(long, value_parser=clap::value_parser!(i64).range(0..))]
    max_tokens: i64,
    #[arg(long, value_parser=clap::value_parser!(i64).range(0..))]
    rerank_rpm: i64,
    #[arg(long, value_parser=clap::value_parser!(i64).range(0..))]
    rerank_tpm: i64,
    #[arg(long, default_value_t=1, value_parser=clap::value_parser!(u32).range(1..=5))]
    warmups: u32,
    #[arg(long, default_value_t=20, value_parser=clap::value_parser!(u32).range(1..=100))]
    repeats: u32,
    #[arg(long)]
    json: bool,
}
/// Closed diagnostic vocabulary: never formats provider/source/config error values.
#[derive(Debug, Serialize, PartialEq, Eq)]
struct Failure {
    category: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    http_status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    validation_reason: Option<&'static str>,
}
fn failure(error: &rerank::Error) -> Failure {
    use crate::{
        models::{attempt::AttemptError, rerank::Error as Provider},
        search::admission,
    };
    let mut status = None;
    let mut validation_reason = None;
    let category = match error {
        rerank::Error::Privacy => "source_or_policy_refused",
        rerank::Error::Input => "input_or_configuration",
        rerank::Error::Deadline => "deadline",
        rerank::Error::Task => "preparation_task",
        rerank::Error::Source(_) => "source_or_store",
        rerank::Error::Download(_) => "model_artifact",
        rerank::Error::Model(_) => "model_preparation",
        rerank::Error::Admission(error) => match error {
            admission::Error::Deadline => "deadline",
            admission::Error::Budget { .. } => "daily_budget",
            admission::Error::Accounting(_) => "accounting_or_store",
            admission::Error::Task => "accounting_task",
            admission::Error::Refused(usage::Dispatch::RateLimited { .. }) => "rate_limit",
            admission::Error::Refused(usage::Dispatch::BudgetExhausted { .. }) => "daily_budget",
            admission::Error::Refused(_) => "admission_refused",
            admission::Error::Provider(AttemptError::Rerank(error)) => match error {
                Provider::Http { status: code, .. } => {
                    status = Some(*code);
                    "provider_http"
                }
                Provider::Response(reason) => {
                    validation_reason = Some(reason.code());
                    "provider_response_validation"
                }
                Provider::Transport => "provider_transport",
                Provider::Timeout => "provider_timeout",
                Provider::Input => "provider_input",
                Provider::Configuration => "provider_configuration",
                Provider::Client => "provider_client",
                Provider::Smoke { .. } => "native_smoke",
                Provider::Native { .. } => "native_inference",
            },
            admission::Error::Provider(AttemptError::LocalRerank(_)) => "native_inference",
            admission::Error::Provider(_) => "provider_other",
        },
    };
    Failure {
        category,
        http_status: status,
        validation_reason,
    }
}

#[derive(Serialize)]
struct Sample {
    phase: &'static str,
    ordinal: u32,
    candidate_count: usize,
    preparation_ms: f64,
    admitted_whole_set_score_ms: Option<f64>,
    outcome: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    failure: Option<Failure>,
}
#[derive(Serialize)]
struct Latency {
    samples: usize,
    p50_ms: f64,
    p95_ms: f64,
    max_ms: f64,
}
#[derive(Serialize)]
struct Accounting {
    reconciled: i64,
    dispatched: i64,
    released: i64,
    reserved: i64,
    sent_units: i64,
}
#[derive(Serialize)]
struct Report {
    format: &'static str,
    state: &'static str,
    timing_scope: &'static str,
    build: cli::bench_metadata::BuildInfo,
    head: String,
    generation: i64,
    config_blake3: String,
    query_blake3: String,
    ordered_units_blake3: String,
    candidate_count: usize,
    indexed_units: i64,
    physical_vectors: u64,
    factory_preparation_timeout_ms: u64,
    max_doc_chars: u64,
    timeout_ms: u64,
    configured_provider: String,
    primary_embedder_profile: String,
    remote_model_blake3: String,
    local_model_blake3: String,
    requested_gpu_layers: i64,
    actual_gpu_placement: &'static str,
    max_calls: i64,
    max_tokens: i64,
    rerank_rpm: i64,
    rerank_tpm: i64,
    warmups: u32,
    repeats: u32,
    factory_preparation_ms: Option<f64>,
    accounting_scope: &'static str,
    before: Accounting,
    after: Option<Accounting>,
    samples: Vec<Sample>,
    latency: Option<Latency>,
    error: Option<&'static str>,
}
// The report owns one new inode for its entire lifetime. Never rename over a
// caller-selected destination, even when that destination is the live database.
struct Output {
    file: fs::File,
    path: PathBuf,
    #[cfg(unix)]
    ancestry: Vec<(PathBuf, u64, u64)>,
}
impl Output {
    fn create(path: &Path) -> Result<Self> {
        #[cfg(unix)]
        {
            use rustix::fs::{Mode, OFlags, openat};
            use std::os::unix::fs::MetadataExt;
            let absolute = if path.is_absolute() {
                path.to_path_buf()
            } else {
                std::env::current_dir()?.join(path)
            };
            if absolute
                .components()
                .any(|part| matches!(part, std::path::Component::ParentDir))
            {
                bail!("report path must not contain parent traversal")
            }
            let parent = absolute
                .parent()
                .ok_or_else(|| anyhow::anyhow!("report parent is absent"))?;
            let name = absolute
                .file_name()
                .ok_or_else(|| anyhow::anyhow!("report filename is absent"))?;
            // macOS exposes `/var` through a system symlink. Rewrite that
            // system alias lexically so O_NOFOLLOW still sees and rejects any
            // repository-controlled symlink later in the path.
            let parent = crate::store::path::system_path(parent);
            let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
            let mut directory = fs::File::from(openat(rustix::fs::CWD, "/", flags, Mode::empty())?);
            let mut current = PathBuf::from("/");
            let meta = directory.metadata()?;
            let mut ancestry = vec![(current.clone(), meta.dev(), meta.ino())];
            for component in parent.components() {
                match component {
                    std::path::Component::RootDir | std::path::Component::CurDir => continue,
                    std::path::Component::Normal(part) => {
                        directory = fs::File::from(openat(&directory, part, flags, Mode::empty())?);
                        current.push(part);
                        let meta = directory.metadata()?;
                        ancestry.push((current.clone(), meta.dev(), meta.ino()));
                    }
                    _ => bail!("unsupported report path component"),
                }
            }
            let file = fs::File::from(openat(
                &directory,
                name,
                OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::RUSR | Mode::WUSR,
            )?);
            let output = Self {
                file,
                path: current.join(name),
                ancestry,
            };
            output.check()?;
            Ok(output)
        }
        #[cfg(not(unix))]
        {
            let _ = path;
            bail!("checked benchmark report creation is unavailable on this platform")
        }
    }
    fn check(&self) -> Result<()> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            for (path, device, inode) in &self.ancestry {
                let meta = fs::symlink_metadata(path)?;
                if !meta.is_dir() || meta.dev() != *device || meta.ino() != *inode {
                    bail!("benchmark report parent changed")
                }
            }
            let meta = fs::symlink_metadata(&self.path)?;
            let owned = self.file.metadata()?;
            if !meta.is_file()
                || meta.dev() != owned.dev()
                || meta.ino() != owned.ino()
                || meta.nlink() != 1
                || owned.nlink() != 1
            {
                bail!("benchmark report identity changed")
            }
            Ok(())
        }
        #[cfg(not(unix))]
        bail!("checked benchmark report writing is unavailable on this platform")
    }
    fn save(&mut self, report: &impl Serialize) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(report)?;
        self.check()?;
        self.file.rewind()?;
        self.file.set_len(0)?;
        self.file.write_all(&bytes)?;
        self.file.write_all(b"\n")?;
        self.file.sync_all()?;
        self.check()
    }
}

fn digest(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}
fn bounded(path: &Path) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(1_048_577)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 1_048_576 {
        bail!("benchmark input exceeds 1 MiB")
    }
    Ok(bytes)
}
fn inputs(args: &Args) -> Result<(String, Vec<i64>)> {
    let query = String::from_utf8(bounded(&args.query_file)?)
        .map_err(|_| anyhow::anyhow!("query must be UTF-8"))?;
    let ids: Vec<i64> = serde_json::from_slice(&bounded(&args.units_file)?)
        .map_err(|_| anyhow::anyhow!("unit selection must be a JSON integer array"))?;
    if query.trim().is_empty()
        || ids.len() != 50
        || ids.iter().any(|id| *id <= 0)
        || ids.iter().collect::<BTreeSet<_>>().len() != 50
    {
        bail!("benchmark requires a nonempty query and exactly 50 distinct positive unit IDs")
    }
    Ok((query, ids))
}
fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|value| i64::try_from(value.as_secs()).ok())
        .unwrap_or(-1)
}
fn accounting(store: &Store) -> Result<Accounting> {
    Ok(store.read_snapshot(|tx| {
        tx.query_row(
            "SELECT coalesce(sum(state='reconciled'),0),coalesce(sum(state='dispatched'),0),coalesce(sum(state='released'),0),coalesce(sum(state='reserved'),0),coalesce(sum(sent_units),0) FROM usage_reservations WHERE provider LIKE 'rerank:%'",
            [], |row| Ok(Accounting { reconciled:row.get(0)?, dispatched:row.get(1)?, released:row.get(2)?, reserved:row.get(3)?, sent_units:row.get(4)? }),
        )
    })??)
}
struct Authority {
    config_root: PathBuf,
    root: PathBuf,
    config: Config,
    store: Arc<Store>,
    head: String,
    generation: i64,
    database: Arc<store::path::Prepared>,
}
impl Authority {
    fn check(&self) -> Result<()> {
        self.database.check_existing_database()?;
        if config::load(&self.config_root, None)?.config != self.config
            || super::git_stdout(&self.root, &["rev-parse", "HEAD"])?.trim() != self.head
            || self.store.get_meta("indexed_head")?.as_deref() != Some(self.head.as_str())
            || self.store.l1_generation()? != self.generation
            || self.store.counts()?.l1_pending_files != 0
            || self.store.get_meta("graph_pending")?.as_deref() != Some("0")
        {
            bail!("benchmark authority changed or index preparation is incomplete")
        }
        Ok(())
    }
}
async fn check(authority: &Arc<Authority>) -> Result<()> {
    let authority = Arc::clone(authority);
    tokio::task::spawn_blocking(move || authority.check()).await??;
    Ok(())
}
fn latency(samples: &[Sample]) -> Option<Latency> {
    let mut values: Vec<f64> = samples
        .iter()
        .filter(|sample| sample.phase == "measured")
        .map(|sample| sample.admitted_whole_set_score_ms)
        .collect::<Option<Vec<_>>>()?;
    if values.is_empty() || samples.iter().any(|sample| sample.outcome != "passed") {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(Latency {
        samples: values.len(),
        p50_ms: values[(values.len() * 50).div_ceil(100) - 1],
        p95_ms: values[(values.len() * 95).div_ceil(100) - 1],
        max_ms: *values.last()?,
    })
}
/// Runs the fixed set through the shared provider and ledger; failures retain partial evidence.
pub(super) fn run(args: Args) -> Result<()> {
    let (query, ids) = inputs(&args)?;
    let config_root = fs::canonicalize(&args.root)?;
    let config = config::load(&config_root, None)?.config;
    if config.rerank.mode != "on"
        || config.rerank.candidates < 50
        || config.rerank.max_doc_chars == 0
        || config.limits.rerank_timeout_ms == 0
    {
        bail!(
            "configure rerank.mode='on', candidates>=50 and nonzero document/timeout caps before measuring"
        )
    }
    let root = fs::canonicalize(config_root.join(&config.index.root))?;
    let database = Arc::new(store::path::prepare(
        &root,
        &cli::database_path(&root, &config),
    )?);
    database.check_existing_database()?;
    let store = Arc::new(store::open_existing_for_usage(
        &root,
        &cli::database_path(&root, &config),
        store::StoreLimits {
            db_max_bytes: config.limits.db_max_bytes()?.0,
            wal_max_bytes: config.limits.wal_max_bytes()?.0,
            max_l2_units: config.limits.max_l2_units,
            max_l3_units: config.limits.max_l3_units,
        },
    )?);
    database.check_existing_database()?;
    let head = super::git_stdout(&root, &["rev-parse", "HEAD"])?
        .trim()
        .to_owned();
    let generation = store.l1_generation()?;
    let authority = Arc::new(Authority {
        config_root,
        root,
        config: config.clone(),
        store: Arc::clone(&store),
        head: head.clone(),
        generation,
        database,
    });
    authority.check()?;
    let units = store.read_snapshot(|tx| {
        ids.iter()
            .map(|id| {
                store::unit_by_id_in(tx, *id)?
                    .ok_or_else(|| anyhow::anyhow!("selected unit is absent"))
            })
            .collect::<Result<Vec<_>>>()
    })??;
    let mut identity = blake3::Hasher::new();
    for unit in &units {
        identity.update(&unit.id.to_le_bytes());
        identity.update(&unit.hash);
    }
    let mut output = Output::create(&args.out)?;
    let mut report = Report {
        format: "dkc-standalone-rerank-v1",
        state: "running",
        timing_scope: "admission-source-validation-score-settlement; preparation-excluded",
        build: cli::bench_metadata::build_info()?,
        head,
        generation,
        config_blake3: digest(toml::to_string(&config)?.as_bytes()),
        query_blake3: digest(query.as_bytes()),
        ordered_units_blake3: identity.finalize().to_hex().to_string(),
        candidate_count: 50,
        indexed_units: store.counts()?.units,
        physical_vectors: store.count_vectors()?,
        factory_preparation_timeout_ms: 120_000,
        max_doc_chars: config.rerank.max_doc_chars,
        timeout_ms: config.limits.rerank_timeout_ms,
        configured_provider: config.rerank.provider.clone(),
        primary_embedder_profile: config.embedder.profile.clone(),
        remote_model_blake3: digest(config.rerank.remote.model.as_bytes()),
        local_model_blake3: digest(config.rerank.local.model.as_bytes()),
        requested_gpu_layers: config.rerank.local.gpu_layers,
        actual_gpu_placement: "not_observed; retain native load evidence separately",
        max_calls: args.max_calls,
        max_tokens: args.max_tokens,
        rerank_rpm: args.rerank_rpm,
        rerank_tpm: args.rerank_tpm,
        warmups: args.warmups,
        repeats: args.repeats,
        factory_preparation_ms: None,
        accounting_scope: "lifetime reservations across all rerank providers; compare before/after on an idle index",
        before: accounting(&store)?,
        after: None,
        samples: Vec::new(),
        latency: None,
        error: None,
    };
    output.save(&report)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(measure(
        &args,
        &authority,
        query,
        units,
        &mut report,
        &mut output,
    ));
    report.after = accounting(&store).ok();
    if result.is_err() {
        report.state = "failed";
        report.error = Some("measurement refused or failed; inspect static sample outcome");
    } else if report.after.is_none() {
        report.state = "failed";
        report.error = Some("final accounting unavailable");
    } else {
        report.state = "complete";
        report.latency = latency(&report.samples);
    }
    output.save(&report)?;
    cli::emit(
        args.json,
        serde_json::to_value(&report)?,
        &format!("standalone rerank: {}", report.state),
    )?;
    if report.state != "complete" {
        bail!("standalone rerank failed; report retained")
    }
    Ok(())
}
async fn measure(
    args: &Args,
    authority: &Arc<Authority>,
    query: String,
    units: Vec<StoredUnit>,
    report: &mut Report,
    output: &mut Output,
) -> Result<()> {
    check(authority).await?;
    let cache = match &args.model_cache {
        Some(path) => path.clone(),
        None => models::registry::cache_root()?,
    };
    let started = Instant::now();
    let factory = tokio::time::timeout(
        Duration::from_secs(120),
        rerank::Factory::new_configured(
            Arc::clone(&authority.store),
            rerank::Settings {
                config_root: authority.config_root.clone(),
                config: authority.config.clone(),
                limits: usage::Limits {
                    budget: usage::Budget {
                        calls: Some(args.max_calls),
                        tokens: Some(args.max_tokens),
                    },
                    rpm: Some(args.rerank_rpm),
                    tpm: Some(args.rerank_tpm),
                },
            },
            cache,
        ),
    )
    .await;
    report.factory_preparation_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
    let factory = factory
        .map_err(|_| anyhow::anyhow!("configured scorer preparation timed out"))?
        .map_err(|_| anyhow::anyhow!("configured scorer preparation failed"))?;
    for index in 0..args.warmups + args.repeats {
        let phase = if index < args.warmups {
            "warmup"
        } else {
            "measured"
        };
        let mut sample = Sample {
            phase,
            ordinal: if phase == "warmup" {
                index
            } else {
                index - args.warmups
            },
            candidate_count: 50,
            preparation_ms: 0.0,
            admitted_whole_set_score_ms: None,
            outcome: "authority_refused",
            failure: None,
        };
        let result = async {
            check(authority).await?;
            let started = Instant::now();
            let deadline = started
                .checked_add(Duration::from_millis(
                    authority.config.limits.rerank_timeout_ms,
                ))
                .ok_or_else(|| anyhow::anyhow!("rerank deadline is out of range"))?;
            let prepared = factory
                .prepare(query.clone(), units.clone(), deadline)
                .await;
            sample.preparation_ms = started.elapsed().as_secs_f64() * 1000.0;
            sample.outcome = "preparation_refused";
            let prepared = prepared.map_err(|error| {
                sample.failure = Some(failure(&error));
                anyhow::anyhow!("candidate preparation refused")
            })?;
            let started = Instant::now();
            let scores = prepared.score(deadline, now).await;
            sample.admitted_whole_set_score_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
            sample.outcome = "score_refused_or_failed";
            let scores = scores.map_err(|error| {
                sample.failure = Some(failure(&error));
                anyhow::anyhow!("admitted scoring failed")
            })?;
            if scores.len() != 50 || scores.iter().any(|score| !score.is_finite()) {
                bail!("whole-set scores are invalid")
            }
            sample.outcome = "final_authority_refused";
            check(authority).await?;
            sample.outcome = "passed";
            Ok::<_, anyhow::Error>(())
        }
        .await;
        report.samples.push(sample);
        output.save(report)?;
        result?;
    }
    Ok(())
}

#[cfg(all(test, unix))]
mod output_tests {
    use super::*;
    #[test]
    fn replaced_report_refuses_update_without_touching_either_inode() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("report.json");
        let mut output = Output::create(&path).unwrap();
        output.save(&serde_json::json!({"first":true})).unwrap();
        let retained = root.path().join("retained.json");
        fs::rename(&path, &retained).unwrap();
        let before = fs::read(&retained).unwrap();
        fs::write(&path, "replacement").unwrap();
        assert!(output.save(&serde_json::json!({"second":true})).is_err());
        assert_eq!(fs::read(&retained).unwrap(), before);
        assert_eq!(fs::read(&path).unwrap(), b"replacement");
    }
    #[test]
    fn replaced_parent_refuses_update_to_retained_output() {
        let root = tempfile::tempdir().unwrap();
        let parent = root.path().join("output");
        fs::create_dir(&parent).unwrap();
        let mut output = Output::create(&parent.join("report.json")).unwrap();
        output.save(&serde_json::json!({"first":true})).unwrap();
        let retired = root.path().join("retired");
        fs::rename(&parent, &retired).unwrap();
        fs::create_dir(&parent).unwrap();
        let before = fs::read(retired.join("report.json")).unwrap();
        assert!(output.save(&serde_json::json!({"second":true})).is_err());
        assert_eq!(fs::read(retired.join("report.json")).unwrap(), before);
        assert!(!parent.join("report.json").exists());
    }
    #[test]
    fn failure_categories_retain_http_status_without_error_payloads() {
        use crate::{
            models::{attempt::AttemptError, rerank::Error as Provider},
            search::admission,
        };
        for status in [401, 403, 413, 422, 429, 500] {
            let error = rerank::Error::Admission(admission::Error::Provider(AttemptError::Rerank(
                Provider::Http {
                    status,
                    retry_after: Some(Duration::from_secs(17)),
                },
            )));
            assert_eq!(
                serde_json::to_value(failure(&error)).unwrap(),
                serde_json::json!({"category":"provider_http","http_status":status})
            );
        }
        for (reason, code) in [
            (
                crate::models::rerank::ResponseReason::BodyLimit,
                "body_limit",
            ),
            (crate::models::rerank::ResponseReason::Schema, "schema"),
            (crate::models::rerank::ResponseReason::Count, "count"),
            (crate::models::rerank::ResponseReason::Index, "index"),
            (crate::models::rerank::ResponseReason::Score, "score"),
        ] {
            let error = rerank::Error::Admission(admission::Error::Provider(AttemptError::Rerank(
                Provider::Response(reason),
            )));
            assert_eq!(
                serde_json::to_value(failure(&error)).unwrap(),
                serde_json::json!({"category":"provider_response_validation", "validation_reason":code})
            );
        }
        for (error, expected) in [
            (Provider::Transport, "provider_transport"),
            (
                Provider::Smoke {
                    file: "arbitrary-private-error-value".into(),
                },
                "native_smoke",
            ),
        ] {
            let error =
                rerank::Error::Admission(admission::Error::Provider(AttemptError::Rerank(error)));
            assert_eq!(
                serde_json::to_value(failure(&error)).unwrap(),
                serde_json::json!({"category":expected})
            );
        }
        assert_eq!(
            failure(&rerank::Error::Privacy).category,
            "source_or_policy_refused"
        );
        assert_eq!(
            failure(&rerank::Error::Admission(admission::Error::Budget {
                retry_at: 123
            }))
            .category,
            "daily_budget"
        );
    }
}
