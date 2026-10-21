//! Finite public L3 text passes independent of embedding readiness; v1 §6.8/§10.2.
use crate::{config::Config, queue::enrichment_host, store::Store};
use anyhow::{Result, bail};
use serde::Serialize;
use std::{
    path::Path,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

/// Fixed diagnostics and retained progress from already committed chat work.
#[derive(Default, Serialize)]
pub(super) struct Report {
    pub(super) state: &'static str,
    visited: usize,
    scheduled: usize,
    current: usize,
    skipped: usize,
    paused: usize,
    terminal: usize,
    published: usize,
    discarded: usize,
    deferred: usize,
    dead: usize,
    pending: Option<i64>,
    error: Option<&'static str>,
    error_codes: std::collections::BTreeMap<String, usize>,
}
fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|value| i64::try_from(value.as_secs()).ok())
        .unwrap_or(-1)
}
/// Runs after L1 publication, respecting explicit level/no-queue/shrink decisions.
pub(super) fn run(
    store: Arc<Store>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    enabled: bool,
) -> Result<Report> {
    let mut report = Report {
        state: "skipped",
        ..Default::default()
    };
    if !enabled || !config.llm.enabled {
        return Ok(report);
    }
    report.state = "running";
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(async {
        tokio::select! {
            biased;
            signal=crate::serve::interrupted()=>signal.map(|()|true).map_err(anyhow::Error::from),
            result=execute(store,root,config_root,config,&mut report)=>result.map(|()|false),
        }
    });
    runtime.shutdown_timeout(Duration::from_secs(1));
    match result {
        Ok(true) => {
            report.state = "cancelled";
            report.error =
                Some("summary work interrupted; committed output and usage are retained");
        }
        Err(error) => {
            if let Some(error) = error.downcast_ref::<crate::queue::Error>() {
                *report
                    .error_codes
                    .entry(crate::queue::diagnostics::code(error))
                    .or_default() += 1;
            }
            report.state = "failed";
            report.error = Some(
                "summary work stopped; check chat configuration, source eligibility and queue status",
            );
        }
        Ok(false) => {}
    }
    Ok(report)
}
async fn execute(
    store: Arc<Store>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    report: &mut Report,
) -> Result<()> {
    let Some(mut host) = enrichment_host::Host::prepare(
        store,
        root.into(),
        config_root.into(),
        config.clone(),
        now(),
    )
    .await?
    else {
        report.state = "skipped";
        return Ok(());
    };
    for _ in 0..host.catalogue_pass_bound() {
        let pass = host.pass(now).await?;
        report.visited += pass.visited;
        report.scheduled += pass.scheduled;
        report.current += pass.current;
        report.skipped += pass.skipped;
        report.paused += pass.paused;
        report.terminal += pass.terminal;
        report.published += pass.execution.published;
        report.discarded += pass.execution.discarded;
        report.deferred += pass.execution.deferred;
        report.dead += pass.execution.dead;
        for error in pass
            .execution
            .errors
            .iter()
            .map(|(_, error)| error)
            .chain(pass.error.iter())
        {
            *report
                .error_codes
                .entry(crate::queue::diagnostics::code(error))
                .or_default() += 1;
        }
        if pass.error.is_some() || !pass.execution.errors.is_empty() || pass.execution.dead > 0 {
            bail!("summary pass failed")
        }
        if host.catalogue_finished()? {
            break;
        }
    }
    report.pending = Some(host.pending()?);
    report.state = if report.terminal > 0 {
        report.error = Some(
            "selected summary generations cannot retry; inspect terminal jobs and their input or provider diagnostics",
        );
        "failed"
    } else if !host.catalogue_finished()? {
        "pass_limit"
    } else if report.pending != Some(0) || report.paused > 0 {
        "waiting"
    } else {
        "complete"
    };
    Ok(())
}
