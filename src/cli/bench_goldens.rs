//! Typed handwritten golden evaluation with frozen denominators; v1 §14, plan §6.
use super::{LatencyMetrics, repeated_latency};
use crate::{config::Config, search, store::Store};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Case {
    id: String,
    pub(super) query: String,
    kind: String,
    expect_files: BTreeSet<String>,
    #[serde(default)]
    expect_symbols: BTreeSet<String>,
    pub(super) scope: Option<String>,
    cohort: Option<String>,
    #[serde(rename = "notes")]
    _notes: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct Report {
    pub review_status: String,
    pub source_hashes: Vec<String>,
    pub cohorts: BTreeMap<String, Score>,
    pub cases: Vec<Outcome>,
    pub latency: LatencyMetrics,
}
#[derive(Debug, Default, Serialize, Deserialize)]
pub(super) struct Score {
    pub count: usize,
    pub positive_count: usize,
    pub symbol_query_count: usize,
    pub no_answer_count: usize,
    pub hit_file_at_5: Option<f64>,
    pub recall_at_10: Option<f64>,
    pub hit_symbol_at_5: Option<f64>,
    pub no_answer_empty_rate: Option<f64>,
}
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct Outcome {
    id: String,
    query: String,
    scope: Option<String>,
    expected_symbols: BTreeSet<String>,
    kind: String,
    cohort: String,
    expected_files: BTreeSet<String>,
    returned_files: Vec<String>,
    hit_file_at_5: Option<bool>,
    recall_at_10: Option<f64>,
    hit_symbol_at_5: Option<bool>,
    no_answer_empty: Option<bool>,
    response_tokens: usize,
}

/// Validated input and review identity, independent of the retrieval implementation.
#[derive(Debug, Serialize)]
pub(super) struct Dataset {
    pub(super) cases: Vec<Case>,
    review_status: String,
    source_hashes: Vec<String>,
    missing_files_allowed: bool,
    missing_expected_files: BTreeMap<String, BTreeSet<String>>,
}

/// Source-free result locations plus the exact formatted response token count.
#[derive(Clone, Debug, Serialize)]
pub(super) struct Evidence {
    hits: Vec<Hit>,
    response_tokens: usize,
}
#[derive(Clone, Debug, Serialize)]
struct Hit {
    path: Option<String>,
    start: Option<i64>,
    end: Option<i64>,
}
impl Evidence {
    /// Captures only scoring fields; source text is never retained in raw evidence.
    pub(super) fn from_response(response: &serde_json::Value) -> Result<Self> {
        let hits = response["hits"]
            .as_array()
            .context("missing golden search hits")?;
        Ok(Self {
            hits: hits
                .iter()
                .map(|hit| Hit {
                    path: hit["path"].as_str().map(str::to_owned),
                    start: hit["lines"][0].as_i64(),
                    end: hit["lines"][1].as_i64(),
                })
                .collect(),
            response_tokens: tiktoken_rs::cl100k_base_singleton()
                .encode_ordinary(&serde_json::to_string(response)?)
                .len(),
        })
    }
}

/// Loads bounded golden files and checks frozen review identity before any search.
pub(super) fn load(
    root: &Path,
    primary: &Path,
    extras: &[PathBuf],
    reviewed_hash: Option<&str>,
) -> Result<Dataset> {
    load_with_missing_files(root, primary, extras, reviewed_hash, false)
}

/// Retains absent historical expectations as diagnostic misses without weakening path checks.
pub(super) fn load_with_missing_files(
    root: &Path,
    primary: &Path,
    extras: &[PathBuf],
    reviewed_hash: Option<&str>,
    allow_missing: bool,
) -> Result<Dataset> {
    if allow_missing && reviewed_hash.is_some() {
        bail!("missing-file diagnostics cannot claim reviewed goldens");
    }
    let mut cases = Vec::new();
    let mut source_hashes = Vec::new();
    let mut ids = BTreeSet::new();
    let mut missing_expected_files = BTreeMap::<String, BTreeSet<String>>::new();
    for (index, path) in std::iter::once(primary)
        .chain(extras.iter().map(PathBuf::as_path))
        .enumerate()
    {
        let mut bytes = Vec::new();
        fs::File::open(path)?
            .take(1_000_001)
            .read_to_end(&mut bytes)?;
        if bytes.len() > 1_000_000 {
            bail!("golden file exceeds 1 MB: {}", path.display());
        }
        source_hashes.push(blake3::hash(&bytes).to_hex().to_string());
        let loaded: Vec<Case> = serde_yaml_ng::from_slice(&bytes)
            .with_context(|| format!("invalid goldens {}", path.display()))?;
        if loaded.is_empty() {
            bail!("empty golden file: {}", path.display());
        }
        for mut case in loaded {
            if !ids.insert(case.id.clone()) || case.id.is_empty() || case.query.trim().is_empty() {
                bail!("duplicate or empty golden id/query: {}", case.id);
            }
            if !["vague", "ident", "doc", "cross"].contains(&case.kind.as_str()) {
                bail!("invalid golden kind: {}", case.kind);
            }
            let cohort = case.cohort.get_or_insert_with(|| "positive".to_owned());
            if (index == 0 && cohort != "positive")
                || (index > 0 && !["held_out", "scoped", "no_answer"].contains(&cohort.as_str()))
            {
                bail!("invalid primary/extra cohort for {}", case.id);
            }
            if (cohort == "no_answer") != case.expect_files.is_empty()
                || (cohort == "no_answer" && !case.expect_symbols.is_empty())
            {
                bail!("invalid expected answers for {}", case.id);
            }
            if cohort == "scoped" && case.scope.is_none() {
                bail!("scoped case lacks scope: {}", case.id);
            }
            for path in &case.expect_files {
                if Path::new(path)
                    .components()
                    .any(|component| !matches!(component, std::path::Component::Normal(_)))
                {
                    bail!(
                        "golden {} expects a path outside the repository: {path}",
                        case.id
                    );
                }
                if !root.join(path).is_file() {
                    if !allow_missing {
                        bail!("golden {} expects missing file {path}", case.id);
                    }
                    missing_expected_files
                        .entry(case.id.clone())
                        .or_default()
                        .insert(path.clone());
                }
            }
            cases.push(case);
        }
    }
    let review_status = match reviewed_hash {
        Some(hash) if source_hashes.first().is_some_and(|actual| actual == hash) => {
            "primary_reviewed_extras_unreviewed"
        }
        Some(_) => bail!("reviewed golden hash does not match current primary file"),
        None => "unreviewed_draft",
    }
    .to_owned();
    Ok(Dataset {
        cases,
        source_hashes,
        review_status,
        missing_files_allowed: allow_missing,
        missing_expected_files,
    })
}

/// Rejects missing expected symbols before a benchmark can spend provider budget.
pub(super) fn validate_symbols(dataset: &Dataset, store: &Store) -> Result<()> {
    let symbols = store.tool_symbols()?;
    for case in &dataset.cases {
        for name in &case.expect_symbols {
            if !symbols.iter().any(|symbol| {
                case.expect_files.contains(&symbol.path)
                    && (symbol.name == *name || symbol.qualified.as_deref() == Some(name))
            }) {
                bail!("golden {} expects missing symbol {name}", case.id);
            }
        }
    }
    Ok(())
}

/// Scores supplied retrieval evidence with shared file/symbol/cohort semantics.
pub(super) fn score(
    dataset: &Dataset,
    store: &Store,
    responses: &[Evidence],
    latency: LatencyMetrics,
) -> Result<Report> {
    if responses.len() != dataset.cases.len() {
        bail!("golden response count mismatch");
    }
    let symbols = store.tool_symbols()?;
    let mut outcomes = Vec::new();
    for (case, response) in dataset.cases.iter().zip(responses) {
        let hits = &response.hits;
        let paths = hits
            .iter()
            .take(10)
            .filter_map(|hit| hit.path.as_deref())
            .collect::<Vec<_>>();
        let found = paths
            .iter()
            .take(10)
            .filter(|path| case.expect_files.contains(**path))
            .copied()
            .collect::<BTreeSet<_>>();
        let mut expected_symbols = Vec::new();
        for name in &case.expect_symbols {
            let matches = symbols
                .iter()
                .filter(|symbol| {
                    case.expect_files.contains(&symbol.path)
                        && (symbol.name == *name || symbol.qualified.as_deref() == Some(name))
                })
                .collect::<Vec<_>>();
            if matches.is_empty() {
                bail!("golden {} expects missing symbol {name}", case.id);
            }
            expected_symbols.extend(matches);
        }
        let symbol_hit = hits.iter().take(5).any(|hit| {
            expected_symbols.iter().any(|symbol| {
                hit.path.as_deref() == Some(&symbol.path)
                    && hit.start.is_some_and(|start| start <= symbol.line_start)
                    && hit.end.is_some_and(|end| end >= symbol.line_end)
            })
        });
        outcomes.push(Outcome {
            id: case.id.clone(),
            query: case.query.clone(),
            scope: case.scope.clone(),
            expected_symbols: case.expect_symbols.clone(),
            kind: case.kind.clone(),
            cohort: case.cohort.clone().unwrap_or_default(),
            expected_files: case.expect_files.clone(),
            returned_files: paths.iter().map(|path| (*path).to_owned()).collect(),
            hit_file_at_5: (!case.expect_files.is_empty()).then(|| {
                hits.iter()
                    .take(5)
                    .filter_map(|hit| hit.path.as_deref())
                    .any(|path| case.expect_files.contains(path))
            }),
            recall_at_10: (!case.expect_files.is_empty())
                .then(|| found.len() as f64 / case.expect_files.len() as f64),
            hit_symbol_at_5: (!expected_symbols.is_empty()).then_some(symbol_hit),
            no_answer_empty: case.expect_files.is_empty().then_some(hits.is_empty()),
            response_tokens: response.response_tokens,
        });
    }
    let mut cohorts = BTreeMap::<String, Score>::new();
    for result in &outcomes {
        let score = cohorts.entry(result.cohort.clone()).or_default();
        score.count += 1;
        if let Some(hit) = result.hit_file_at_5 {
            score.positive_count += 1;
            *score.hit_file_at_5.get_or_insert(0.0) += f64::from(hit);
            *score.recall_at_10.get_or_insert(0.0) += result.recall_at_10.unwrap_or_default();
        }
        if let Some(hit) = result.hit_symbol_at_5 {
            score.symbol_query_count += 1;
            *score.hit_symbol_at_5.get_or_insert(0.0) += f64::from(hit);
        }
        if let Some(empty) = result.no_answer_empty {
            score.no_answer_count += 1;
            *score.no_answer_empty_rate.get_or_insert(0.0) += f64::from(empty);
        }
    }
    for score in cohorts.values_mut() {
        if score.positive_count > 0 {
            score.hit_file_at_5 = score
                .hit_file_at_5
                .map(|value| value / score.positive_count as f64);
            score.recall_at_10 = score
                .recall_at_10
                .map(|value| value / score.positive_count as f64);
        }
        score.hit_symbol_at_5 = score
            .hit_symbol_at_5
            .map(|value| value / score.symbol_query_count as f64);
        score.no_answer_empty_rate = score
            .no_answer_empty_rate
            .map(|value| value / score.no_answer_count as f64);
    }
    Ok(Report {
        review_status: dataset.review_status.clone(),
        source_hashes: dataset.source_hashes.clone(),
        cohorts,
        cases: outcomes,
        latency,
    })
}

/// Keeps the legacy lexical benchmark and its timing method on the shared scorer.
pub(super) fn evaluate(
    root: &Path,
    settings: &Config,
    primary: &Path,
    extras: &[PathBuf],
    reviewed_hash: Option<&str>,
) -> Result<Report> {
    let dataset = load(root, primary, extras, reviewed_hash)?;
    let store = super::super::open_store(root, settings)?;
    validate_symbols(&dataset, &store)?;
    let responses = dataset
        .cases
        .iter()
        .map(|case| Evidence::from_response(&run(&store, settings, case)?))
        .collect::<Result<Vec<_>>>()?;
    let latency = repeated_latency(dataset.cases.len(), |index| {
        run(&store, settings, &dataset.cases[index])?;
        Ok(())
    })?;
    score(&dataset, &store, &responses, latency)
}

fn run(store: &Store, settings: &Config, case: &Case) -> Result<serde_json::Value> {
    let hits = search::lexical(
        store,
        &case.query,
        10,
        case.scope.as_deref().or(Some("source")),
        &settings.search,
    )?;
    let result = search::format_response(store, &case.query, &hits)?;
    std::hint::black_box(serde_json::to_vec(&result)?);
    Ok(result)
}
