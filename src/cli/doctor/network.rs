//! Independent provider diagnostics with public inputs and durable admission; v1 §13.
use super::{Finding, chat::Chat, inference::Embedding, rerank::Rerank};
use crate::{
    config::{self, Config},
    models::{discovery::Discovery, openai_compat::resolve_api_key},
    search::admission,
    store::{self, Store, StoreLimits, UnitKind},
};
use std::{
    path::Path,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::time::Instant;

fn clock() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|time| i64::try_from(time.as_secs()).ok())
        .unwrap_or(-1)
}
fn evidence(mut finding: Finding, started: std::time::Instant) -> Finding {
    finding.evidence = Some(serde_json::json!({"latency_ms":started.elapsed().as_millis()}));
    finding
}
fn unchanged(root: &Path, config: &Config) -> bool {
    config::load(root, None).is_ok_and(|current| current.config == *config)
}
/// A missing ledger permits model listing, but never inference dispatch.
#[derive(Clone)]
pub(super) enum LedgerIdentity {
    Missing,
    Checked(Arc<store::path::Prepared>),

    #[cfg(test)]
    UncheckedFixture,
}
impl LedgerIdentity {
    pub(super) fn valid(&self) -> bool {
        match self {
            Self::Missing => false,
            Self::Checked(guard) => guard.check_existing_database().is_ok(),
            #[cfg(test)]
            Self::UncheckedFixture => true,
        }
    }
}

fn usage_store(root: &Path, config: &Config) -> Option<(Arc<Store>, LedgerIdentity)> {
    let repository = root.join(&config.index.root).canonicalize().ok()?;
    let path = super::super::database_path(&repository, config);
    let limits = StoreLimits {
        db_max_bytes: config.limits.db_max_bytes().ok()?.0,
        wal_max_bytes: config.limits.wal_max_bytes().ok()?.0,
        max_l2_units: config.limits.max_l2_units,
        max_l3_units: config.limits.max_l3_units,
    };
    let guard = Arc::new(store::path::prepare(&repository, &path).ok()?);
    guard.check_existing_database().ok()?;
    let store = store::open_existing_for_usage(&repository, &path, limits).ok()?;
    guard.check_existing_database().ok()?;
    Some((Arc::new(store), LedgerIdentity::Checked(guard)))
}
pub(super) fn refusal(id: &str, error: admission::Error) -> Finding {
    let (status, message) = match error {
        admission::Error::Deadline => ("failed", "Provider diagnostic exceeded its deadline"),
        admission::Error::Budget { .. } => (
            "not_checked",
            "Configured daily budget prevents this diagnostic request",
        ),
        admission::Error::Refused(_) => (
            "not_checked",
            "Shared admission or changed policy prevents this diagnostic request",
        ),
        admission::Error::Accounting(_) | admission::Error::Task => (
            "not_checked",
            "Usage accounting could not safely admit or settle the diagnostic",
        ),
        admission::Error::Provider(crate::models::attempt::AttemptError::Chat(
            crate::models::openai_compat::chat::Error::ThinkingConsumedBudget { .. },
        )) => (
            "failed",
            "Chat used completion tokens but returned empty content; check llm.extra_body.chat_template_kwargs.enable_thinking=false; reported usage was retained and no blind retry was performed",
        ),
        admission::Error::Provider(_) => (
            "failed",
            "Provider diagnostic request failed; no blind retry was performed",
        ),
    };
    Finding::new(id, status, message)
}

/// Runs synchronous preparation outside any existing runtime; individual requests
/// are async/cancellable and have finite deadlines. No source text is accepted.
pub(super) fn collect(root: &Path, config: &Config) -> Vec<Finding> {
    let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    else {
        return vec![Finding::new(
            "providers.probes",
            "not_checked",
            "Provider diagnostic runtime is unavailable",
        )];
    };
    let mut findings = Vec::new();
    let discovery = Discovery::new();
    match super::super::init::probe::targets(config) {
        Ok(targets) => {
            if targets.is_empty() {
                findings.push(Finding::new(
                    "providers.discovery",
                    "not_applicable",
                    "No remote model listing is configured",
                ));
            }
            for (ordinal, target) in targets.into_iter().enumerate() {
                let id = format!("providers.discovery.{}", ordinal + 1);
                if !unchanged(root, config) {
                    findings.push(Finding::new(
                        &id,
                        "not_checked",
                        "Configuration changed before model discovery",
                    ));
                    continue;
                }
                let key = resolve_api_key(
                    &target.api_key_env,
                    &target.url,
                    config::credentials_path().as_deref(),
                );
                let started = std::time::Instant::now();
                let finding = match (&discovery, key) {
                    (Ok(client), Ok(key)) => match runtime.block_on(client.verify(
                        &target.url,
                        Some(&key),
                        &target.models,
                    )) {
                        Ok(()) => Finding::new(
                            &id,
                            "passed",
                            "Endpoint responded and lists every configured model",
                        ),
                        Err(error) => Finding::new(&id, "failed", &error.to_string()),
                    },
                    (_, Err(_)) => {
                        Finding::new(&id, "failed", &credential_message(&target.api_key_env))
                    }
                    (Err(_), _) => {
                        Finding::new(&id, "not_checked", "Model discovery client is unavailable")
                    }
                };
                findings.push(evidence(finding, started));
            }
        }
        Err(_) => findings.push(Finding::new(
            "providers.discovery",
            "failed",
            "Configured model discovery targets are invalid",
        )),
    }
    // Require an existing ledger even for unlimited budgets: a diagnostic must
    // not bypass accounting or silently initialize an index.
    let store = usage_store(root, config);
    for (id, kind, _profile) in tiers(config) {
        let Some(kind) = kind else {
            findings.push(Finding::new(
                id,
                "not_applicable",
                "No unit kinds are routed to this embedding tier",
            ));
            continue;
        };
        if !unchanged(root, config) {
            findings.push(Finding::new(
                id,
                "not_checked",
                "Configuration changed before embedding diagnosis",
            ));
            continue;
        }
        let provider = match Embedding::new(
            root.to_owned(),
            config.clone(),
            kind,
            store
                .as_ref()
                .map_or(LedgerIdentity::Missing, |(_, guard)| guard.clone()),
        ) {
            Ok(provider) => Arc::new(provider),
            Err(_) => {
                findings.push(Finding::new(
                    id,
                    "failed",
                    "Embedding diagnostic settings, credentials or client are unavailable",
                ));
                continue;
            }
        };
        let started = std::time::Instant::now();
        let deadline =
            Instant::now() + Duration::from_millis(config.limits.query_timeout_ms.min(15_000));
        let installed = runtime
            .block_on(async { tokio::time::timeout_at(deadline, provider.installed()).await });
        let finding = match (installed, &store) {
            (Ok(Ok(true)), None) => Finding::new(
                id,
                "not_checked",
                "Inference diagnosis requires an existing compatible usage ledger",
            ),
            (Ok(Ok(true)), Some((store, _))) => {
                match runtime.block_on(provider.run(Arc::clone(store), deadline, clock)) {
                    Ok((dimensions, fallback)) => {
                        let mut finding = evidence(
                            Finding::new(
                                id,
                                "passed",
                                if fallback {
                                    "Embedding succeeded; rejected dimensions falls back to local truncation"
                                } else {
                                    "Embedding succeeded with one fixed public input"
                                },
                            ),
                            started,
                        );
                        finding.evidence = Some(
                            serde_json::json!({"latency_ms":started.elapsed().as_millis(),"dimensions":dimensions,"dimensions_fallback":fallback}),
                        );
                        finding
                    }
                    Err(error) => evidence(refusal(id, error), started),
                }
            }
            (Ok(Ok(false)), _) => evidence(
                Finding::new(
                    id,
                    "failed",
                    "Configured Ollama model is not installed; no model was pulled",
                ),
                started,
            ),
            (Ok(Err(_)), _) => evidence(
                Finding::new(
                    id,
                    "failed",
                    "Ollama is unavailable or returned an invalid model listing",
                ),
                started,
            ),
            (Err(_), _) => evidence(
                Finding::new(
                    id,
                    "failed",
                    "Ollama model listing exceeded the diagnostic deadline",
                ),
                started,
            ),
        };
        findings.push(finding);
    }
    let id = "chat.probe";
    if !config.llm.enabled {
        findings.push(Finding::new(
            id,
            "not_applicable",
            "Chat enrichment is disabled",
        ));
    } else if let Some((store, guard)) = &store {
        let started = std::time::Instant::now();
        let finding = match Chat::new(root.to_owned(), config.clone(), guard.clone()) {
            Ok(provider) => match runtime.block_on(Arc::new(provider).run(
                Arc::clone(store),
                Instant::now() + Duration::from_secs(15),
                clock,
            )) {
                Ok(()) => Finding::new(
                    id,
                    "passed",
                    "Chat returned nonempty content for one fixed public input; usage accounted",
                ),
                Err(error) => refusal(id, error),
            },
            Err(_) => Finding::new(
                id,
                "failed",
                "Chat settings, credentials or diagnostic client are unavailable",
            ),
        };
        findings.push(evidence(finding, started));
    } else {
        findings.push(Finding::new(
            id,
            "not_checked",
            "Inference diagnosis requires an existing compatible usage ledger",
        ));
    }
    let id = "providers.rerank";
    let remote = config.rerank.provider == "remote"
        || (config.rerank.provider == "auto" && config.embedder.profile == "remote");
    if config.rerank.mode == "off" || !remote {
        findings.push(Finding::new(
            id,
            "not_applicable",
            "Remote reranking is not selected; local smoke is separate",
        ));
    } else if let Some((store, guard)) = store {
        match Rerank::new(root.to_owned(), config.clone(), guard) {
            Ok(provider) => {
                let started = std::time::Instant::now();
                let deadline = Instant::now()
                    + Duration::from_millis(config.limits.rerank_timeout_ms.min(15_000));
                let finding = match runtime.block_on(Arc::new(provider).run(store, deadline, clock))
                {
                    Ok(true) => Finding::new(
                        id,
                        "passed",
                        "Remote reranker passed the three fixed public relevance pairs",
                    ),
                    Ok(false) => Finding::new(
                        id,
                        "failed",
                        "Remote reranker returned scores that fail the relevance smoke; request remains accounted",
                    ),
                    Err(error) => refusal(id, error),
                };
                findings.push(evidence(finding, started));
            }
            Err(_) => findings.push(Finding::new(
                id,
                "failed",
                "Remote rerank settings, credentials or smoke capacity are unavailable",
            )),
        }
    } else {
        findings.push(Finding::new(
            id,
            "not_checked",
            "Inference diagnosis requires an existing compatible usage ledger",
        ));
    }
    findings
}

fn tiers(config: &Config) -> Vec<(&'static str, Option<UnitKind>, &str)> {
    let in_docs = |kind: &UnitKind| {
        config
            .embedder
            .docs
            .as_ref()
            .is_some_and(|docs| docs.kinds.iter().any(|value| value == kind.as_str()))
    };
    let mut tiers = vec![(
        "providers.embedding.code",
        UnitKind::ALL.into_iter().find(|kind| !in_docs(kind)),
        config.embedder.profile.as_str(),
    )];
    if let Some(docs) = &config.embedder.docs {
        tiers.push((
            "providers.embedding.docs",
            UnitKind::ALL.into_iter().find(in_docs),
            docs.profile.as_str(),
        ));
    }
    tiers
}

fn credential_message(name: &str) -> String {
    // A conventional bounded environment identifier is a reference, never its
    // resolved value. Unexpected configuration strings remain undisclosed.
    let conventional = name.len() <= 128
        && !name.is_empty()
        && name.bytes().enumerate().all(|(index, byte)| {
            byte == b'_' || byte.is_ascii_alphabetic() || (index > 0 && byte.is_ascii_digit())
        });
    let variable = if conventional {
        name
    } else {
        "the variable named by api_key_env"
    };
    format!(
        "Provider credential is missing or unreadable; check {variable} and ~/.config/deepkit-code/credentials.toml (or the configured user-config directory)"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn credential_diagnosis_names_locations_without_echoing_unexpected_configuration() {
        let normal = credential_message("MOONLY_GATEWAY_API_KEY");
        assert!(normal.contains("MOONLY_GATEWAY_API_KEY"));
        assert!(normal.contains("~/.config/deepkit-code/credentials.toml"));
        let marker = "private-marker\nnot-an-environment-reference";
        let redacted = credential_message(marker);
        assert!(!redacted.contains("private-marker"));
        assert!(!redacted.contains('\n'));
        assert!(redacted.contains("api_key_env"));
    }
    #[test]
    fn diagnostic_tiers_follow_configured_kinds_and_do_not_invent_a_second_tier() {
        let single = Config::default();
        assert_eq!(tiers(&single).len(), 1);
        let mut dual: Config =
            toml::from_str("[embedder.docs]\nprofile='remote'\nkinds=['image']\n").unwrap();
        let selected = tiers(&dual);
        assert_eq!(selected[1].1, Some(UnitKind::Image));
        assert_eq!(selected[1].2, "remote");
        assert_ne!(selected[0].1, Some(UnitKind::Image));
        dual.embedder.docs.as_mut().unwrap().kinds = UnitKind::ALL
            .iter()
            .map(|kind| kind.as_str().to_owned())
            .collect();
        assert_eq!(tiers(&dual)[0].1, None);
    }
}
