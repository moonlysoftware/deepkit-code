//! Read-only persisted ladder and configured model/budget status; v1 §§5, 10.1, 13.
use crate::{
    config::{self, Config},
    store::{self, Store},
};
use serde::Serialize;

/// Source-labelled status additions; usage and model health retain their existing surfaces.
#[derive(Debug, Serialize)]
pub(crate) struct Snapshot {
    pub(crate) ladder: Ladder,
    pub(crate) profile: Profile,
    pub(crate) models: Models,
    pub(crate) budgets: Budgets,
}
/// Exact highest-completed levels, not cumulative capabilities or fresh coverage.
/// L3 image/page enrichment can originate at L0, so it must not imply L2 completion.
#[derive(Debug, Serialize)]
pub(crate) struct Ladder {
    source: &'static str,
    counting: &'static str,
    l1: i64,
    l2: i64,
    l3: i64,
}
/// Effective configured routing; retained physical transition tables are reported elsewhere.
#[derive(Debug, Serialize)]
pub(crate) struct Profile {
    source: &'static str,
    mode: &'static str,
    primary: String,
    docs: Option<Docs>,
}
#[derive(Debug, Serialize)]
struct Docs {
    profile: String,
    kinds: Vec<String>,
}
/// Selected configuration recipes, with no claim of loaded, downloaded or healthy models.
#[derive(Debug, Serialize)]
pub(crate) struct Models {
    source: &'static str,
    embedder: Embedding,
    docs: Option<Embedding>,
    rerank: Rerank,
    llm: Llm,
}
#[derive(Debug, Serialize)]
struct Embedding {
    profile: String,
    model: String,
    dims: u64,
}
#[derive(Debug, Serialize)]
struct Rerank {
    mode: String,
    provider: String,
    selected_provider: Option<String>,
    model: Option<String>,
}
#[derive(Debug, Serialize)]
struct Llm {
    enabled: bool,
    model: Option<String>,
    vlm_model: Option<String>,
}
/// Configured embedding-provider ceilings only; daily usage stays in queue_status.
/// Null defaults mean unlimited; an explicit zero endpoint entry remains zero.
#[derive(Debug, Serialize)]
pub(crate) struct Budgets {
    source: &'static str,
    scope: &'static str,
    default_daily_calls: Option<i64>,
    default_daily_tokens: Option<i64>,
    endpoints: Vec<config::queue::EndpointBudget>,
}
/// Status must not hide invalid stored reads or silently guess an unknown profile.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error(transparent)]
    Store(#[from] store::Error),
    #[error(transparent)]
    Config(#[from] config::Error),
    #[error("unsupported configured model profile")]
    Profile,
}

/// Reads occurrence levels and copies explicitly labelled configuration only.
/// No writes, vector-table scans, credential resolution, cache checks or model work occur.
pub(crate) fn read(store: &Store, config: &Config) -> Result<Snapshot, Error> {
    config.queue.budget("")?;
    let ladder = store.read_snapshot(|tx| {
        tx.query_row("SELECT count(CASE WHEN ladder_level=1 THEN 1 END), count(CASE WHEN ladder_level=2 THEN 1 END), count(CASE WHEN ladder_level=3 THEN 1 END) FROM units", [], |row| {
            Ok(Ladder { source: "persisted_occurrences", counting: "exact_highest_completed_level",
                l1: row.get(0)?, l2: row.get(1)?, l3: row.get(2)? })
        }).map_err(store::Error::from)
    })??;
    let docs = config.embedder.docs.as_ref();
    Ok(Snapshot {
        ladder,
        profile: Profile {
            source: "configured",
            mode: if docs.is_some() { "dual" } else { "single" },
            primary: config.embedder.profile.clone(),
            docs: docs.map(|docs| Docs {
                profile: docs.profile.clone(),
                kinds: docs.kinds.clone(),
            }),
        },
        models: Models {
            source: "configured",
            embedder: embedding(config, &config.embedder.profile)?,
            docs: docs
                .map(|docs| embedding(config, &docs.profile))
                .transpose()?,
            rerank: rerank(config)?,
            llm: Llm {
                enabled: config.llm.enabled,
                model: config.llm.enabled.then(|| config.llm.model.clone()),
                vlm_model: config.llm.enabled.then(|| config.llm.vlm_model.clone()),
            },
        },
        budgets: Budgets {
            source: "configured",
            scope: "embedding_providers",
            default_daily_calls: None,
            default_daily_tokens: None,
            endpoints: config.queue.budgets.clone(),
        },
    })
}
fn embedding(config: &Config, profile: &str) -> Result<Embedding, Error> {
    let model = match profile {
        "remote" => &config.embedder.remote.model,
        "ollama" => &config.embedder.ollama.model,
        "local" => &config.embedder.local.model,
        _ => return Err(Error::Profile),
    };
    Ok(Embedding {
        profile: profile.into(),
        model: model.clone(),
        dims: config.embedder.dims,
    })
}
fn rerank(config: &Config) -> Result<Rerank, Error> {
    let configured = &config.rerank;
    let selected = if configured.mode == "off" {
        None
    } else {
        Some(match configured.provider.as_str() {
            "remote" => "remote",
            "local" => "local",
            "auto" => match config.embedder.profile.as_str() {
                "remote" => "remote",
                "local" | "ollama" => "local",
                _ => return Err(Error::Profile),
            },
            _ => return Err(Error::Profile),
        })
    };
    Ok(Rerank {
        mode: configured.mode.clone(),
        provider: configured.provider.clone(),
        selected_provider: selected.map(str::to_owned),
        model: selected.map(|profile| {
            if profile == "remote" {
                configured.remote.model.clone()
            } else {
                configured.local.model.clone()
            }
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn database() -> Store {
        store::open(":memory:", store::StoreLimits::default()).unwrap()
    }
    #[test]
    fn ladder_counts_occurrences_by_exact_persisted_level_without_writes() {
        let db = database();
        let config = Config::default();
        let empty = read(&db, &config).unwrap();
        assert_eq!(
            (empty.ladder.l1, empty.ladder.l2, empty.ladder.l3),
            (0, 0, 0)
        );
        for (index, level) in [1, 1, 2, 3].into_iter().enumerate() {
            let id =
                crate::search::tests::add(&db, &format!("{index}.rs"), "sample", "same text", None)
                    .unwrap();
            db.writer()
                .unwrap()
                .execute(
                    "UPDATE units SET ladder_level=?1 WHERE id=?2",
                    rusqlite::params![level, id],
                )
                .unwrap();
        }
        let before = db.writer().unwrap().total_changes();
        let status = read(&db, &config).unwrap();
        assert_eq!(
            (status.ladder.l1, status.ladder.l2, status.ladder.l3),
            (2, 1, 1)
        );
        assert_eq!(status.ladder.counting, "exact_highest_completed_level");
        assert_eq!(db.writer().unwrap().total_changes(), before);
    }
    #[test]
    fn configured_single_dual_and_inactive_models_need_no_provider_or_cache() {
        let db = database();
        let mut config = Config::default();
        config.embedder.profile = "local".into();
        config.embedder.local.model = "absent-unregistered-local-model".into();
        config.embedder.remote.api_key_env = "DKC_UNUSED_STATUS_CREDENTIAL".into();
        let before = db.writer().unwrap().total_changes();
        let single = serde_json::to_value(read(&db, &config).unwrap()).unwrap();
        assert_eq!(single["profile"]["mode"], "single");
        assert!(single["profile"]["docs"].is_null());
        assert!(single["models"]["docs"].is_null());
        assert_eq!(
            single["models"]["embedder"]["model"],
            "absent-unregistered-local-model"
        );
        assert_eq!(single["models"]["rerank"]["selected_provider"], "local");
        assert!(single["models"]["llm"]["model"].is_null());
        config.embedder.docs = Some(
            serde_json::from_value(json!({"profile":"remote", "kinds":["doc_section"]})).unwrap(),
        );
        config.llm.enabled = true;
        config.rerank.mode = "off".into();
        let dual = serde_json::to_value(read(&db, &config).unwrap()).unwrap();
        assert_eq!(dual["profile"]["mode"], "dual");
        assert_eq!(dual["profile"]["docs"]["kinds"], json!(["doc_section"]));
        assert_eq!(
            dual["models"]["docs"]["model"],
            config.embedder.remote.model
        );
        assert!(dual["models"]["rerank"]["model"].is_null());
        assert_eq!(dual["models"]["llm"]["model"], config.llm.model);
        assert!(!dual.to_string().contains("DKC_UNUSED_STATUS_CREDENTIAL"));
        assert_eq!(db.writer().unwrap().total_changes(), before);
    }
    #[test]
    fn budgets_distinguish_explicit_zero_from_unlimited_defaults() {
        let db = database();
        let mut config = Config::default();
        let unlimited = serde_json::to_value(read(&db, &config).unwrap()).unwrap();
        assert!(unlimited["budgets"]["default_daily_calls"].is_null());
        assert_eq!(unlimited["budgets"]["endpoints"], json!([]));
        config.queue.budgets = serde_json::from_value(json!([{
            "profile":"local", "daily_calls":0,"daily_tokens":123
        }]))
        .unwrap();
        let configured = serde_json::to_value(read(&db, &config).unwrap()).unwrap();
        assert_eq!(configured["budgets"]["endpoints"][0]["daily_calls"], 0);
        assert_eq!(configured["budgets"]["endpoints"][0]["daily_tokens"], 123);
        assert!(configured["budgets"]["default_daily_tokens"].is_null());
    }
}
