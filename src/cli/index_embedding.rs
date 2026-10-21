//! Finite public index embedding passes after durable L1 publication; v1 §8/§10.2.
use crate::{config::Config, queue::host, store::Store};
use anyhow::{Result, bail};
use serde::Serialize;
use std::{
    path::Path,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[derive(Default, Serialize)]
pub(super) struct Report {
    pub(super) state: &'static str,
    providers: Vec<Progress>,
    // Only fixed diagnostics are exposed: provider errors may contain input data.
    error: Option<&'static str>,
}
#[derive(Default, Serialize)]
struct Progress {
    provider: String,
    catalogue_finished: bool,
    passes: usize,
    visited: usize,
    scheduled_or_existing: usize,
    skipped: usize,
    published: usize,
    discarded: usize,
    deferred: usize,
    dead: usize,
    lease_lost: usize,
    errors: usize,
    error_codes: std::collections::BTreeMap<String, usize>,
    pending: Option<i64>,
}
impl Progress {
    fn record(&mut self, pass: &host::Report) -> bool {
        self.passes += 1;
        self.visited += pass.visited;
        self.scheduled_or_existing += pass.scheduled_or_existing;
        self.skipped += pass.skipped;
        self.published += pass.execution.published;
        self.discarded += pass.execution.discarded;
        self.deferred += pass.execution.deferred;
        self.dead += pass.execution.dead;
        self.lease_lost += pass.execution.lease_lost;
        self.errors += pass.execution.errors.len()
            + usize::from(pass.retirement.error.is_some())
            + usize::from(pass.transition_error.is_some());
        for error in pass
            .execution
            .errors
            .iter()
            .map(|(_, error)| error)
            .chain(pass.retirement.error.iter())
            .chain(pass.transition_error.iter())
        {
            *self
                .error_codes
                .entry(crate::queue::diagnostics::code(error))
                .or_default() += 1;
        }
        self.errors > 0 || self.dead > 0
    }
}
fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|value| i64::try_from(value.as_secs()).ok())
        .unwrap_or(-1)
}
/// Runs one captured catalogue sweep and one final ready pass per selected
/// provider. Future retries and excess ready work remain durable for the next run.
/// A failed pass preserves committed counts and L1; no provider error is logged.
pub(super) fn run(
    store: Arc<Store>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    enabled: bool,
    refresh_context: bool,
) -> Result<Report> {
    let mut report = Report {
        state: "skipped",
        ..Default::default()
    };
    if !enabled {
        return Ok(report);
    }
    report.state = "running";
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let outcome = runtime.block_on(async {
        tokio::select! {
            // Register listeners before polling work; drop work first on shutdown
            // so the host/worker guards fence outstanding blocking operations.
            biased;
            signal = crate::serve::interrupted() => signal.map(|()| true).map_err(anyhow::Error::from),
            result = execute(&store, root, config_root, config, &mut report, refresh_context) => result.map(|()| false),
        }
    });
    // Native/SQLite calls cannot be forcibly cancelled. Bound runtime teardown
    // rather than waiting indefinitely for its blocking threads to return.
    runtime.shutdown_timeout(Duration::from_secs(1));
    match outcome {
        Ok(true) => {
            report.state = "cancelled";
            report.error = Some(
                "embedding work interrupted; committed output is retained; rerun index after outstanding leases expire",
            );
        }
        Ok(false) => {}
        Err(_) => {
            report.state = "failed";
            report.error = Some(
                "embedding work stopped; inspect queue status and rerun index with current settings",
            );
        }
    }
    Ok(report)
}

async fn execute(
    store: &Arc<Store>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    report: &mut Report,
    refresh_context: bool,
) -> Result<()> {
    let indexed_head = store.get_meta("indexed_head")?;
    let check = || -> Result<()> {
        let observed = crate::sync::head::read(root, Default::default())?.map(|head| head.id);
        if crate::sync::head::changed(indexed_head.as_deref(), observed.as_deref()) {
            bail!("HEAD changed after L1 publication")
        }
        Ok(())
    };
    check()?;
    let kinds = store.embedding_catalogue_kinds(
        config.commits.enabled,
        config.commits.enabled && config.commits.prs,
        config.enrich.top_symbols,
        config.enrich.modules_min_files,
    )?;
    let group = host::group::Group::prepare(
        Arc::clone(store),
        host::group::Settings {
            root: root.into(),
            config_root: config_root.into(),
            config: config.clone(),
            local_cache: crate::models::registry::cache_root()?,
        },
        &kinds,
        |provider| host::Bounds::for_provider(&config.queue, provider),
        now(),
    )
    .await?;
    for mut entry in group.entries {
        entry.host = entry.host.with_context_refresh(refresh_context);
        let mut cursor = entry.host.start_catalogue().await?;
        report.providers.push(Progress {
            provider: entry.provider,
            ..Default::default()
        });
        let progress = report
            .providers
            .last_mut()
            .ok_or_else(|| anyhow::anyhow!("missing provider progress"))?;
        while !cursor.finished() {
            check()?;
            let pass = match entry
                .host
                .run_catalogue(&mut cursor, |candidate| Ok(candidate.recency(now())), now)
                .await
            {
                Ok(pass) => pass,
                Err(error) => {
                    progress.visited += error.visited;
                    progress.scheduled_or_existing += error.scheduled_or_existing;
                    return Err(error.into());
                }
            };
            progress.catalogue_finished = cursor.finished();
            if progress.record(&pass) {
                bail!("embedding pass failed");
            }
        }
        check()?;
        let drain = match entry.host.drain(1, now).await {
            Ok(drain) => drain,
            Err(error) => {
                for pass in &error.progress.passes {
                    progress.record(pass);
                }
                progress.pending = error.progress.pending;
                return Err(error.source.into());
            }
        };
        for pass in &drain.passes {
            if progress.record(pass) {
                bail!("embedding drain failed");
            }
        }
        progress.pending = drain.pending;
    }
    check()?;
    report.state = if report.providers.iter().any(|p| p.pending != Some(0)) {
        "pending"
    } else {
        "quiescent"
    };
    Ok(())
}
