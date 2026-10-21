//! Bounded draining of persisted work without a catalogue sweep; v1 §8/§10.2.
use crate::{
    config,
    queue::{self, host},
    store::{self, Store, StoreLimits},
};
use anyhow::{Result, bail};
use serde::Serialize;
use std::{
    path::Path,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[derive(Default, Serialize)]
struct Progress {
    provider: String,
    passes: usize,
    claimed: usize,
    published: usize,
    discarded: usize,
    deferred: usize,
    dead: usize,
    lease_lost: usize,
    errors: usize,
    error_codes: std::collections::BTreeMap<String, usize>,
    pending: Option<i64>,
    stopped: &'static str,
}

impl Progress {
    fn record(&mut self, drain: &host::DrainReport) {
        self.pending = drain.pending;
        self.stopped = match drain.stopped {
            host::DrainStop::Empty => "empty",
            host::DrainStop::Waiting => "waiting",
            host::DrainStop::PassLimit => "pass_limit",
            host::DrainStop::Failed => "failed",
        };
        for pass in &drain.passes {
            self.passes += 1;
            self.claimed += pass.execution.claimed;
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
                    .entry(queue::diagnostics::code(error))
                    .or_default() += 1;
            }
        }
    }

    /// Retains chat-worker outcomes using the same source-free taxonomy as
    /// embedding drains, so a failed bounded resume remains actionable.
    fn record_chat(&mut self, pass: &queue::enrichment_host::Report) {
        self.passes += 1;
        self.claimed += pass.execution.claimed;
        self.published += pass.execution.published;
        self.discarded += pass.execution.discarded;
        self.deferred += pass.execution.deferred;
        self.dead += pass.execution.dead;
        self.lease_lost += pass.execution.lease_lost;
        self.errors += pass.execution.errors.len() + usize::from(pass.error.is_some());
        for error in pass
            .execution
            .errors
            .iter()
            .map(|(_, error)| error)
            .chain(pass.error.iter())
        {
            *self
                .error_codes
                .entry(queue::diagnostics::code(error))
                .or_default() += 1;
        }
    }
}

#[derive(Serialize)]
struct Report {
    state: &'static str,
    providers: Vec<Progress>,
    queue: Option<queue::status::Snapshot>,
    error: Option<&'static str>,
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|value| i64::try_from(value.as_secs()).ok())
        .unwrap_or(-1)
}

/// Drains a finite number of existing-job passes and retains committed progress
/// on interruption. Configured budgets, delays, leases and attempts remain owned
/// by the shared host; this command never retries or schedules catalogue targets.
pub(super) fn run(max_passes: u32, json: bool) -> Result<()> {
    let config_root = std::fs::canonicalize(super::current_dir()?)?;
    let config = config::load(&config_root, None)?.config;
    let root = std::fs::canonicalize(config_root.join(&config.index.root))?;
    let store = Arc::new(store::open_existing_for_usage(
        &root,
        &super::database_path(&root, &config),
        StoreLimits {
            db_max_bytes: config.limits.db_max_bytes()?.0,
            wal_max_bytes: config.limits.wal_max_bytes()?.0,
            max_l2_units: config.limits.max_l2_units,
            max_l3_units: config.limits.max_l3_units,
        },
    )?);
    let mut report = Report {
        state: "running",
        providers: Vec::new(),
        queue: None,
        error: None,
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let outcome = runtime.block_on(async {
        tokio::select! {
            biased;
            signal = crate::serve::interrupted() => signal.map(|()| true).map_err(anyhow::Error::from),
            result = execute(&store, &root, &config_root, &config, max_passes, &mut report) => result.map(|()| false),
        }
    });
    runtime.shutdown_timeout(Duration::from_secs(1));
    match outcome {
        Ok(true) => {
            report.state = "cancelled";
            report.error =
                Some("drain interrupted; committed work and outstanding leases are retained");
        }
        Err(_) => {
            report.state = "failed";
            report.error = Some(
                "drain stopped; inspect queue status and index current sources before resuming",
            );
        }
        Ok(false) => {}
    }
    match queue::status::snapshot_with_jobs(&store, now(), None) {
        Ok(snapshot) => {
            if !matches!(report.state, "failed" | "cancelled") {
                report.state = if snapshot.jobs.is_quiescent() {
                    "quiescent"
                } else {
                    "pending"
                };
            }
            report.queue = Some(snapshot);
        }
        Err(_) => {
            report.state = "failed";
            report.error =
                Some("drain accounting snapshot unavailable; committed work is retained");
        }
    }
    super::emit(
        json,
        serde_json::to_value(&report)?,
        &format!(
            "Queue drain {}; configured limits and delayed work retained.",
            report.state
        ),
    )?;
    if matches!(report.state, "failed" | "cancelled") {
        bail!("queue drain {}", report.state);
    }
    Ok(())
}

async fn execute(
    store: &Arc<Store>,
    root: &Path,
    config_root: &Path,
    config: &config::Config,
    max_passes: u32,
    report: &mut Report,
) -> Result<()> {
    let indexed = store.get_meta("indexed_head")?;
    let check = || -> Result<()> {
        let observed = crate::sync::head::read(root, Default::default())?.map(|head| head.id);
        if crate::sync::head::changed(indexed.as_deref(), observed.as_deref())
            || !store.pending_l1_paths()?.is_empty()
            || store.get_meta("graph_pending")?.as_deref() != Some("0")
            || config::load(config_root, None)?.config != *config
        {
            bail!("queue drain requires the current completed index and configuration");
        }
        Ok(())
    };
    check()?;
    if queue::status::snapshot_with_jobs(store, now(), None)?
        .jobs
        .is_quiescent()
    {
        return Ok(());
    }
    if let Some(mut chat) = queue::enrichment_host::Host::prepare_ready(
        Arc::clone(store),
        root.into(),
        config_root.into(),
        config.clone(),
        now(),
    )
    .await?
    {
        report.providers.push(Progress {
            provider: chat.provider().to_owned(),
            ..Default::default()
        });
        let progress = report
            .providers
            .last_mut()
            .ok_or_else(|| anyhow::anyhow!("missing chat drain progress"))?;
        for _ in 0..max_passes {
            check()?;
            let pass = chat.pass(now).await?;
            progress.record_chat(&pass);
            progress.pending = Some(chat.pending()?);
            progress.stopped = if progress.errors > 0 || progress.dead > 0 {
                "failed"
            } else if progress.pending == Some(0) {
                "empty"
            } else if pass.execution.claimed == 0 {
                "waiting"
            } else {
                "pass_limit"
            };
            if progress.stopped == "failed" {
                bail!("chat provider drain failed");
            }
            if matches!(progress.stopped, "empty" | "waiting") {
                break;
            }
        }
    }
    // Only persisted embedding jobs justify preparing their model resources.
    let embedding_pending: bool = store.read_snapshot(|tx| tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM jobs WHERE kind='embed' AND state IN ('queued','leased','failed'))", [], |r|r.get(0),
    ))??;
    if !embedding_pending {
        return Ok(());
    }
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
    for entry in group.entries {
        check()?;
        report.providers.push(Progress {
            provider: entry.provider,
            ..Default::default()
        });
        let progress = report
            .providers
            .last_mut()
            .ok_or_else(|| anyhow::anyhow!("missing drain provider progress"))?;
        for _ in 0..max_passes {
            check()?;
            let (drain, failed) = match entry.host.drain(1, now).await {
                Ok(drain) => (drain, false),
                Err(error) => (error.progress, true),
            };
            // Retain each completed pass before awaiting the next, so cancellation
            // cannot hide an earlier successful publication from this invocation.
            progress.record(&drain);
            if failed || progress.stopped == "failed" || progress.errors > 0 || progress.dead > 0 {
                bail!("queue provider drain failed");
            }
            if matches!(progress.stopped, "empty" | "waiting") {
                break;
            }
        }
    }
    check()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_progress_reports_source_free_worker_and_host_error_codes() {
        let mut progress = Progress::default();
        let pass = queue::enrichment_host::Report {
            error: Some(queue::Error::InputsUnavailable),
            execution: queue::pool::Report {
                errors: vec![(7, queue::Error::Invalid("synthetic queue failure"))],
                ..Default::default()
            },
            ..Default::default()
        };

        progress.record_chat(&pass);

        assert_eq!(progress.errors, 2);
        assert_eq!(
            progress.error_codes,
            std::collections::BTreeMap::from([
                ("inputs_unavailable".to_owned(), 1),
                ("invalid:synthetic queue failure".to_owned(), 1),
            ])
        );
    }
}
