//! Matched PR ranking diagnostics, separate from production RRF; v1 §§9, 14.
use crate::{
    cli, config,
    search::{self, collapse, fusion, query},
    store::Store,
};
use anyhow::{Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
    time::Instant,
};

/// Frozen questions and rank order; this command never selects or tunes weights.
#[derive(clap::Args)]
pub(crate) struct Args {
    #[arg(long)]
    root: PathBuf,
    #[arg(long)]
    queries: PathBuf,
    #[arg(long)]
    ranks: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    json: bool,
}
#[derive(Deserialize)]
struct Question {
    rank: usize,
    merge: String,
    title: String,
}
#[derive(Deserialize)]
struct Questions {
    pin: String,
    original: Vec<Question>,
    #[serde(rename = "older_controls")]
    older: Vec<Question>,
}
#[derive(Deserialize)]
struct Rank {
    merge: String,
    rank: usize,
    candidate_score_multiplier: f64,
}
#[derive(Deserialize)]
struct Ranks {
    pin: String,
    prs: Vec<Rank>,
}

fn date_weight(rank: usize) -> f64 {
    match rank {
        1..=100 => 1.20,
        101..=200 => 1.15,
        201..=500 => 1.10,
        501..=1000 => 1.05,
        _ => 1.0,
    }
}
fn prs(store: &Store) -> Result<BTreeMap<i64, String>> {
    Ok(store.read_snapshot(|tx| {
        let mut statement =
            tx.prepare("SELECT unit_id,lower(hex(merge_sha)) FROM prs ORDER BY unit_id")?;
        statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<rusqlite::Result<BTreeMap<i64, String>>>()
    })??)
}
fn authority(root: &std::path::Path, store: &Store, head: &str, generation: i64) -> Result<()> {
    if super::git_stdout(root, &["rev-parse", "HEAD"])?.trim() != head
        || store.get_meta("indexed_head")?.as_deref() != Some(head)
        || store.l1_generation()? != generation
        || store.counts()?.l1_pending_files != 0
        || store.get_meta("graph_pending")?.as_deref() != Some("0")
    {
        bail!("PR diagnostic requires unchanged HEAD and completed L1/graph")
    }
    Ok(())
}
fn compare(
    store: &Store,
    q: &Question,
    settings: &config::Search,
    merges: &BTreeMap<i64, String>,
    ranks: &BTreeMap<String, usize>,
    timeout: std::time::Duration,
) -> Result<Value> {
    let parsed = query::parse(&q.title, fusion::Mode::Lexical)?;
    if parsed.dense.is_some() {
        bail!("PR diagnostic cannot accept a dense clause")
    }
    let start = Instant::now();
    let check = || {
        if start.elapsed() >= timeout {
            Err(search::Error::Deadline)
        } else {
            Ok(())
        }
    };
    let window = search::lexical_candidates(store, &parsed, 3, None, settings, &[], Some(&check))?;
    let retrieval_ms = start.elapsed().as_secs_f64() * 1000.0;
    let pool: Vec<_> = window
        .hits
        .iter()
        .map(|hit| {
            let projected = window.prepared.projected_unit(hit.unit.id);
            json!({"occurrence": hit.unit.id, "parent": projected.map(|u|u.id),
            "merge": projected.and_then(|u|merges.get(&u.id)), "score": hit.score})
        })
        .collect();
    let mut arms = Vec::new();
    for arm in ["baseline", "uniform_pr", "date_bands"] {
        let started = Instant::now();
        let mut hits = window.hits.clone();
        for hit in &mut hits {
            let pr = window
                .prepared
                .projected_unit(hit.unit.id)
                .filter(|u| u.kind == "pr");
            let multiplier = match (arm, pr) {
                ("uniform_pr", Some(_)) => 1.20,
                ("date_bands", Some(unit)) => merges
                    .get(&unit.id)
                    .and_then(|merge| ranks.get(merge))
                    .map_or(1.0, |rank| date_weight(*rank)),
                _ => 1.0,
            };
            hit.score *= multiplier;
        }
        // Stable ordering retains the original fused order for tied scores.
        // Baseline preserves its exact input order, including any future special ordering.
        if arm != "baseline" {
            hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        }
        let hits = collapse::finish(hits, 3, settings, &window.prepared)?;
        let selected: Vec<_> = hits.iter().filter_map(|h| merges.get(&h.unit.id)).collect();
        arms.push(json!({"arm":arm,"raw_hit_pr_at_3":selected.iter().any(|m|**m == q.merge),
            "selected_merges":selected, "response_before_service_limits":search::format_response(store,&q.title,&hits)?,
            "ranking_and_format_ms":started.elapsed().as_secs_f64()*1000.0}));
    }
    Ok(json!({"rank":q.rank,"merge":q.merge,"query":q.title,
        "expected_pr_stored":merges.values().any(|m|m == &q.merge),
        "expected_pr_in_pool":pool.iter().any(|p|p["merge"].as_str() == Some(q.merge.as_str())),
        "candidate_pool":pool,"shared_retrieval_ms":retrieval_ms,"arms":arms}))
}
/// Runs lexical-only diagnostics with fixed denominators and retained query failures.
/// Raw pre-service scores do not establish public freshness or delivered HitPR@3.
pub(crate) fn run(args: Args) -> Result<()> {
    if args.out.exists() {
        bail!("refusing to overwrite a PR diagnostic report")
    }
    let query_bytes = fs::read(&args.queries)?;
    let rank_bytes = fs::read(&args.ranks)?;
    let questions: Questions = serde_json::from_slice(&query_bytes)?;
    let frozen: Ranks = serde_json::from_slice(&rank_bytes)?;
    if questions.pin != frozen.pin || questions.original.is_empty() || questions.older.is_empty() {
        bail!("incompatible or empty frozen cohorts")
    }
    let mut ranks = BTreeMap::new();
    let mut rank_numbers = BTreeSet::new();
    for row in frozen.prs {
        if row.rank == 0
            || !rank_numbers.insert(row.rank)
            || row.candidate_score_multiplier != date_weight(row.rank)
            || ranks.insert(row.merge, row.rank).is_some()
        {
            bail!("rank table must use the fixed untuned date bands with unique merges")
        }
    }
    let mut targets = BTreeSet::new();
    for q in questions.original.iter().chain(&questions.older) {
        if !targets.insert(&q.merge) || ranks.get(&q.merge) != Some(&q.rank) {
            bail!("question rank disagrees with frozen rank order")
        }
    }
    let config_root = fs::canonicalize(&args.root)?;
    let config = config::load(&config_root, None)?.config;
    let root = fs::canonicalize(config_root.join(&config.index.root))?;
    if !cli::database_path(&root, &config).is_file() {
        bail!("existing index required")
    }
    let store = cli::open_store(&root, &config)?;
    let generation = store.l1_generation()?;
    authority(&root, &store, &questions.pin, generation)?;
    let merges = prs(&store)?;
    let mut report = json!({"state":"running","method":"shared-direct-lexical-pr-weights-v1",
        "head":questions.pin,"query_blake3":blake3::hash(&query_bytes).to_hex().to_string(),
        "ranks_blake3":blake3::hash(&rank_bytes).to_hex().to_string(),
        "uniform_pr_multiplier":1.20,"k":3,"scope":"all","scorer_invoked":false,
        "models_invoked":false,"production_ranking_changed":false,
        "current_public_authority_verified":false,"service_response_budget_applied":false,
        "native_performance_measured":false,"phase_complete":false,
        "recent_denominator":questions.original.len(),"older_denominator":questions.older.len(),"queries":[]});
    let mut rows = Vec::new();
    let mut failed = false;
    for (cohort, queries) in [("recent", &questions.original), ("older", &questions.older)] {
        for q in queries {
            authority(&root, &store, &questions.pin, generation)?;
            let mut row = match compare(
                &store,
                q,
                &config.search,
                &merges,
                &ranks,
                std::time::Duration::from_millis(config.limits.query_timeout_ms),
            ) {
                Ok(row) => row,
                Err(error) => {
                    failed = true;
                    json!({"rank":q.rank,"merge":q.merge,
                    "query":q.title,"error":error.to_string()})
                }
            };
            row["cohort"] = json!(cohort);
            rows.push(row);
            report["queries"] = json!(rows);
            super::embedding::save(&args.out, &report)?;
        }
    }
    authority(&root, &store, &questions.pin, generation)?;
    if config::load(&config_root, None)?.config != config || prs(&store)? != merges {
        bail!("configuration or PR identity changed during comparison")
    }
    report["state"] = json!(if failed {
        "completed_with_query_failures"
    } else {
        "diagnostic_complete"
    });
    super::embedding::save(&args.out, &report)?;
    cli::emit(
        args.json,
        report,
        "PR ranking diagnostic saved; public-response validation remains separate.",
    )?;
    if failed {
        bail!("one or more PR diagnostic queries failed")
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture_unit(
        store: &Store,
        n: i64,
        kind: crate::store::UnitKind,
        parent: Option<i64>,
    ) -> i64 {
        use crate::store::{FtsColumns, UnitInsert};
        let hash = blake3::hash(&n.to_le_bytes());
        store
            .insert_unit(UnitInsert {
                hash: hash.as_bytes(),
                source_key: Some(&format!("pr-diagnostic:{n}")),
                duplicate_ordinal: 0,
                file_id: None,
                kind,
                byte_start: None,
                byte_end: None,
                line_start: None,
                line_end: None,
                scope_path: None,
                header: "ranking probe",
                text: "ranking probe",
                tokens: 2,
                ladder_level: 1,
                parent_unit: parent,
                created_at: 1,
                updated_at: 1,
                fts: FtsColumns {
                    header: "",
                    idents: "",
                    body: "ranking probe",
                    prose: "",
                },
            })
            .unwrap()
    }
    #[test]
    fn matched_weights_promote_pr_parents_and_keep_missing_targets_as_misses() {
        use crate::store::{self, StoreLimits, UnitKind};
        let store = store::open(":memory:", StoreLimits::default()).unwrap();
        for n in 1..=3 {
            fixture_unit(&store, n, UnitKind::History, None);
        }
        let pr = fixture_unit(&store, 4, UnitKind::PullRequest, None);
        fixture_unit(&store, 5, UnitKind::Summary, Some(pr));
        let settings = config::Search {
            graph_prior_weight: 0.0,
            ..config::Search::default()
        };
        let merges = BTreeMap::from([(pr, "target".to_string())]);
        let ranks = BTreeMap::from([("target".to_string(), 1)]);
        let mut question = Question {
            rank: 1,
            merge: "target".into(),
            title: "ranking probe".into(),
        };
        let row = compare(
            &store,
            &question,
            &settings,
            &merges,
            &ranks,
            std::time::Duration::from_secs(10),
        )
        .unwrap();
        assert_eq!(row["candidate_pool"].as_array().unwrap().len(), 5);
        assert_eq!(row["expected_pr_in_pool"], true);
        assert_eq!(row["arms"][0]["raw_hit_pr_at_3"], false);
        for i in [1, 2] {
            assert_eq!(row["arms"][i]["raw_hit_pr_at_3"], true);
            assert_eq!(row["arms"][i]["selected_merges"], json!(["target"]));
        }
        let baseline = search::lexical(&store, &question.title, 3, None, &settings).unwrap();
        assert_eq!(
            row["arms"][0]["response_before_service_limits"],
            search::format_response(&store, &question.title, &baseline).unwrap()
        );
        question.merge = "absent".into();
        let missing = compare(
            &store,
            &question,
            &settings,
            &merges,
            &ranks,
            std::time::Duration::from_secs(10),
        )
        .unwrap();
        assert_eq!(missing["expected_pr_stored"], false);
        assert!(
            missing["arms"]
                .as_array()
                .unwrap()
                .iter()
                .all(|a| a["raw_hit_pr_at_3"] == false)
        );
    }
    #[test]
    fn frozen_bands_have_exact_boundaries_and_no_boost_outside_the_cohort() {
        for (rank, expected) in [
            (0, 1.0),
            (1, 1.20),
            (100, 1.20),
            (101, 1.15),
            (200, 1.15),
            (201, 1.10),
            (500, 1.10),
            (501, 1.05),
            (1000, 1.05),
            (1001, 1.0),
        ] {
            assert_eq!(date_weight(rank), expected);
        }
    }
}
