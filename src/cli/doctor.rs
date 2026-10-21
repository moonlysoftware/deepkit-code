//! Independent local diagnostic collection and reranker smoke; v1 §§7.5, 13, 16.
mod artifacts;
mod bench;
mod chat;
mod globs;
mod ide;
mod inference;
mod local;
mod network;
mod report;
mod rerank;
mod tables;
use crate::{
    config::Config,
    models::{
        self,
        llama::rerank::{LoadReport, Local},
        registry,
    },
};
use anyhow::{Result, bail};
use serde::Serialize;
use std::{path::PathBuf, time::Duration};

#[derive(Debug, Serialize)]
struct Check {
    scope: &'static str,
    status: &'static str,
    model: String,
    file: Option<String>,
    message: String,
    initial_load: Option<LoadReport>,
    persistence: &'static str,
}
fn selected_local(config: &Config) -> bool {
    config.rerank.mode != "off"
        && (config.rerank.provider == "local"
            || (config.rerank.provider == "auto"
                && matches!(config.embedder.profile.as_str(), "local" | "ollama")))
}
impl Check {
    fn new(config: &Config) -> Self {
        // Only embedded registry identifiers may enter the report. An unknown
        // configured value can contain arbitrary private text, even when unused.
        let model = registry::Registry::embedded()
            .ok()
            .and_then(|registry| {
                registry
                    .get(&config.rerank.local.model)
                    .ok()
                    .map(|artifact| artifact.name.clone())
            })
            .unwrap_or_else(|| "unregistered".into());
        Self {
            scope: "local_reranker",
            status: "not_applicable",
            model,
            file: None,
            message: "local reranker smoke: not applicable to the configured reranker".into(),
            initial_load: None,
            persistence: "not_attempted",
        }
    }
    fn failed(mut self, _error: impl std::fmt::Display) -> Self {
        // Do not format provider, registry or filesystem error values: this
        // message is both emitted and retained by reranker health persistence.
        self.status = "failed";
        self.message = format!(
            "local reranker {}: verification failed; run dkc models pull",
            self.file.as_deref().unwrap_or(&self.model)
        );
        self
    }
    fn inference_failed(self, error: models::Error) -> Self {
        let smoke_failure = matches!(
            error,
            models::Error::Rerank(models::rerank::Error::Smoke { .. })
        );
        let mut result = self.failed(error);
        if smoke_failure {
            result.status = "disabled";
        }
        result
    }
}
async fn check(config: &Config, cache: PathBuf) -> Check {
    let mut result = Check::new(config);
    if !selected_local(config) {
        return result;
    }
    let registry = match registry::Registry::embedded() {
        Ok(value) => value,
        Err(error) => return result.failed(error),
    };
    let artifact = match registry.get(&config.rerank.local.model) {
        Ok(value) => value,
        Err(error) => return result.failed(error),
    };
    result.file = Some(artifact.file.clone());
    let path = cache.join(&artifact.file);
    match path.try_exists() {
        Ok(false) => {
            result.status = "failed";
            result.message = "local reranker: missing model; run dkc models pull".into();
            return result;
        }
        Err(error) => return result.failed(error),
        Ok(true) => {}
    }
    match Local::load_registered(cache, config.rerank.clone(), Duration::from_secs(60)).await {
        Ok(local) => {
            result.status = "passed";
            result.message = format!(
                "local reranker: {} verified; three-pair smoke passed",
                artifact.file
            );
            result.initial_load = Some(local.initial_load_report().clone());
            result
        }
        Err(error) => result.inference_failed(error),
    }
}
/// Runs the established smoke and preserves its fenced health persistence.
fn smoke(
    config_root: &std::path::Path,
    config: &Config,
    persistence: Option<&report::Persistence>,
) -> Result<Check> {
    let cache = registry::cache_root();
    let mut report = if selected_local(config) {
        match &cache {
            Ok(cache) => tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?
                .block_on(check(config, cache.clone())),
            Err(error) => Check::new(config).failed(error),
        }
    } else {
        Check::new(config)
    };
    if let Ok(cache) = &cache
        && selected_local(config)
        && report.model != "unregistered"
        && let Some(persistence) = persistence
    {
        use crate::models::rerank::health::{self, State};
        report.persistence = match persistence.open(config) {
            None => "deferred",
            Some(store) => {
                let saved = (|| -> Result<bool> {
                    let state = match report.status {
                        "passed" => State::Verified,
                        "disabled" => State::Disabled,
                        _ => State::Unavailable,
                    };
                    let record = health::Record::new(config, cache, state, report.message.clone())?;
                    Ok(health::write(&store, config_root, config, &record)?)
                })();
                if matches!(saved, Ok(true)) {
                    "saved"
                } else {
                    "deferred"
                }
            }
        };
    }
    Ok(report)
}

#[derive(Debug, Serialize)]
struct Finding {
    id: String,
    status: &'static str,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    evidence: Option<serde_json::Value>,
}
impl Finding {
    fn new(id: &str, status: &'static str, message: &str) -> Self {
        Self {
            id: id.into(),
            status,
            message: message.into(),
            evidence: None,
        }
    }
}

/// Collects independent diagnoses; absent or failed checks never imply health.
/// Provider calls use fixed public inputs and existing usage accounting.
pub(super) fn run(json: bool, bench_checks: bool) -> Result<()> {
    let root = std::env::current_dir()?;
    let local::Local {
        mut findings,
        config,
        explicit_treatments,
    } = local::collect(&root);
    let mut local_reranker = None;
    if let Some(config) = &config {
        let database = report::database(&root, config);
        let allow_persistence = database
            .iter()
            .any(|check| check.id == "database.schema" && check.status == "passed")
            && !database.iter().any(|check| check.status == "failed");
        let persistence = allow_persistence
            .then(|| report::Persistence::capture(&root, config))
            .flatten();
        findings.extend(database);
        findings.push(globs::check(&root, config, explicit_treatments));
        let cache = registry::cache_root().ok();
        findings.extend(artifacts::inspect(config, cache.as_deref()));
        match smoke(&root, config, persistence.as_ref()) {
            Ok(smoke) => {
                findings.push(Finding::new(
                    "models.reranker_smoke",
                    smoke.status,
                    &smoke.message,
                ));
                local_reranker = Some(smoke);
            }
            Err(_) => findings.push(Finding::new(
                "models.reranker_smoke",
                "failed",
                "Local reranker smoke runtime could not be initialized",
            )),
        }
        findings.extend(network::collect(&root, config));
    } else {
        findings.push(Finding::new(
            "database",
            "not_checked",
            "Database checks require valid effective settings",
        ));
        findings.push(Finding::new(
            "models",
            "not_checked",
            "Model checks require valid effective settings",
        ));
    }
    if !findings
        .iter()
        .any(|finding| finding.id == "vectors.tables")
    {
        findings.push(tables::unavailable());
    }
    if !findings.iter().any(|finding| finding.id == "queue.health") {
        findings.push(Finding::new(
            "queue.health",
            "not_checked",
            "Queue health requires a readable compatible database",
        ));
    }
    if !findings
        .iter()
        .any(|finding| finding.id == "config.treatment_globs")
    {
        findings.push(Finding::new(
            "config.treatment_globs",
            "not_checked",
            "Treatment glob lint requires valid effective configuration",
        ));
    }
    if let Ok(executable) = std::env::current_exe() {
        let path = std::env::var_os("PATH");
        for diagnosis in ide::diagnose(&root, &executable, path.as_deref()) {
            use ide::State;
            let status = match diagnosis.state {
                State::Ready => "passed",
                State::Absent | State::MissingEntry => "warning",
                State::Malformed
                | State::Conflicting
                | State::DifferentExecutable
                | State::DifferentWorkingDirectory => "failed",
                State::Unavailable | State::Unresolved => "not_checked",
            };
            findings.push(Finding::new(
                diagnosis.id,
                status,
                diagnosis.state.message(),
            ));
        }
    } else {
        findings.push(Finding::new(
            "mcp",
            "not_checked",
            "Current executable could not be resolved",
        ));
    }
    if bench_checks {
        findings.extend(bench::collect());
    }
    if config.is_none() {
        findings.push(Finding::new(
            "providers.probes",
            "not_checked",
            "Provider probes require valid effective settings",
        ));
    }
    let failed = findings
        .iter()
        .any(|check| matches!(check.status, "failed" | "disabled"));
    let complete = !findings
        .iter()
        .any(|check| matches!(check.status, "not_checked" | "not_implemented"));
    let status = if failed {
        "failed"
    } else if !complete {
        "incomplete"
    } else if findings.iter().any(|check| check.status == "warning") {
        "warning"
    } else {
        "passed"
    };
    let text = findings
        .iter()
        .map(|check| format!("{} [{}]: {}", check.id, check.status, check.message))
        .collect::<Vec<_>>()
        .join("\n");
    let report = report::Report {
        scope: "diagnostics",
        status,
        complete,
        checks: findings,
        local_reranker,
    };
    super::emit(json, serde_json::to_value(report)?, &text)?;
    if failed {
        bail!("doctor found failed checks; see diagnoses above");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn smoke_failure_disables_without_raw_filename_or_success_evidence() {
        let error = crate::models::rerank::smoke::validate(&[0.5; 3], "broken.gguf").unwrap_err();
        let report = Check::new(&Config::default()).inference_failed(error.into());
        assert_eq!(report.status, "disabled");
        assert!(!report.message.contains("broken.gguf"));
        assert!(report.message.contains("dkc models pull"));
        assert!(report.initial_load.is_none());
    }
    #[test]
    fn smoke_report_redacts_unknown_model_and_untrusted_error_even_when_unused() {
        struct PrivateError;
        impl std::fmt::Display for PrivateError {
            fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                panic!("doctor must not format backend errors");
            }
        }
        let marker = "private-placeholder";
        let mut config = Config::default();
        config.rerank.local.model = marker.into();
        config.rerank.mode = "off".into();
        let unused = Check::new(&config);
        assert_eq!(unused.model, "unregistered");
        assert!(!serde_json::to_string(&unused).unwrap().contains(marker));
        let failed = unused.failed(PrivateError);
        assert_eq!(failed.status, "failed");
        assert!(failed.message.contains("dkc models pull"));
        assert!(!serde_json::to_string(&failed).unwrap().contains(marker));
        assert!(failed.file.is_none());
        assert!(failed.initial_load.is_none());
    }

    #[test]
    fn smoke_report_keeps_only_registered_model_identifiers() {
        let config = Config::default();
        let registry = registry::Registry::embedded().unwrap();
        let expected = registry.get(&config.rerank.local.model).unwrap();
        let report = Check::new(&config);
        assert_eq!(report.model, expected.name);
        assert!(report.file.is_none());
    }
}
