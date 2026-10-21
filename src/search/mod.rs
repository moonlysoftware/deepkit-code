//! Shared retrieval, ranking and result evidence; see `v1.md` §9.

pub(crate) mod service;
mod status;

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    config::Search,
    store::{self, Store, StoredUnit},
    treat::fts::build_match,
};

pub(crate) mod admission;
pub(crate) mod collapse;
pub(crate) mod control;
pub(crate) mod dense;
pub(crate) mod embedding_query;
pub(crate) mod fusion;
pub(crate) mod host;
pub(crate) mod latency;
pub(crate) mod map;
pub(crate) mod navigation;
pub(crate) mod pipeline;
pub(crate) mod query;
mod repair;
pub(crate) mod rerank;
mod retry;
pub(crate) mod summaries;

const LEXICAL_LIMIT: usize = 200;

/// Retrieval failures independent of CLI or MCP presentation.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("summary unit {unit_id} has a missing or cyclic parent")]
    InvalidSummaryParent { unit_id: i64 },
    #[error("query deadline exceeded")]
    Deadline,
    #[error("query database task failed")]
    QueryTask,
    #[error("invalid query: {reason}")]
    InvalidQuery { reason: &'static str },
    #[error("dense query retrieval is not connected yet")]
    DenseUnavailable,
    #[error("invalid reciprocal-rank fusion settings")]
    InvalidFusion,
    #[error(transparent)]
    Store(#[from] store::Error),
    #[error("k must be between 1 and {max}")]
    InvalidK { max: u64 },
    #[error("symbol id {target} is ambiguous; open by path and line range")]
    AmbiguousSymbol { target: String },
    #[error("no indexed unit matches {target}")]
    NotFound { target: String },
    #[error(
        "source cannot be verified against completed L1; index the repository before opening this source"
    )]
    OpenSourceUnavailable,
    #[error("-k must be greater than zero")]
    InvalidReferenceLimit,
    #[error("--budget must be greater than zero")]
    InvalidMapBudget,
}

/// One fused occurrence, with evidence retained for transport formatting.
#[derive(Clone)]
pub(crate) struct RankedCandidate {
    pub(crate) unit: StoredUnit,
    pub(crate) score: f64,
    pub(crate) rrf_score: f64,
    pub(crate) rerank_score: Option<f32>,
    pub(crate) rerank_fallback: Option<&'static str>,
    pub(crate) lexical: bool,
    pub(crate) graph: bool,
    pub(crate) exact_name: bool,
    pub(crate) dense: bool,
    pub(crate) scores: fusion::Scores,
    pub(crate) summary: bool,
    pub(crate) chunk: bool,
    pub(crate) matches: Vec<MatchEvidence>,
}

/// Original per-occurrence evidence retained when several hits share one result.
#[derive(Clone, serde::Serialize)]
pub(crate) struct MatchEvidence {
    #[serde(skip)]
    pub(crate) unit_id: i64,
    pub(crate) id: String,
    pub(crate) rrf_score: f64,
    pub(crate) rrf: fusion::Scores,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) rerank_score: Option<f32>,
    pub(crate) why: Vec<&'static str>,
}

/// Runs the shared lexical and graph pipeline with scoped, ranked candidate admission.
///
/// Relaxed matching replaces the strict list when fewer than k occurrences match;
/// exact symbol names lead that lexical list before graph seeding and RRF.
pub(crate) fn lexical(
    store: &Store,
    query: &str,
    k: usize,
    scope: Option<&str>,
    settings: &Search,
) -> Result<Vec<RankedCandidate>, Error> {
    if k == 0 || k as u128 > u128::from(settings.max_k) {
        return Err(Error::InvalidK {
            max: settings.max_k,
        });
    }
    let parsed = query::parse(query, fusion::Mode::Lexical)?;
    if parsed.dense.is_some() {
        return Err(Error::DenseUnavailable);
    }
    lexical_with_exclusions(
        store,
        &parsed,
        k,
        scope.map(crate::store::SearchScope::from).as_ref(),
        settings,
        &[],
    )
}

/// Retrieves an already-routed lexical leg with exclusions before admission limits.
/// Retains the established 200 lexical plus 100 graph candidates until collapse;
/// dense-only text contributes neither lexical matches nor identifier boosts.
pub(crate) fn lexical_with_exclusions(
    store: &Store,
    parsed: &query::Parsed,
    k: usize,
    scope: Option<&crate::store::SearchScope>,
    settings: &Search,
    exclude_ids: &[i64],
) -> Result<Vec<RankedCandidate>, Error> {
    lexical_selection(store, parsed, k, scope, settings, exclude_ids, None).map(|(hits, _)| hits)
}

/// Private candidate-order and collapse evidence, independent of mutable L2 labels.
#[derive(PartialEq)]
pub(crate) struct Selection {
    leg: LexicalLeg,
    occurrences: Vec<StoredUnit>,
    projections: Vec<StoredUnit>,
    links: BTreeMap<i64, Vec<collapse::ProjectionLink>>,
    publications: BTreeMap<i64, Option<store::projection::SourcePublication>>,
    incomplete_hidden_hits: Option<BTreeSet<i64>>,
    selected: serde_json::Value,
}

impl Selection {
    /// Captures final local evidence from retained retrieval snapshots, without
    /// rereading rows after reranking or hiding contributors removed by collapse.
    fn capture(
        leg: LexicalLeg,
        occurrences: Vec<StoredUnit>,
        projections: Vec<StoredUnit>,
        mut prepared: collapse::Prepared,
        hits: &[RankedCandidate],
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<Self, Error> {
        prepared.retain_hidden_publications(hits, check)?;
        let incomplete_hidden_hits = prepared.incomplete_hidden_hits(hits, check)?;
        check()?;
        let selected = serde_json::json!(
            hits.iter()
                .map(|hit| serde_json::json!({
                    "unit": hit.unit.id,
                    "score": hit.score,
                    "scores": hit.scores,
                    "why": reasons(hit),
                    "matches": hit.matches.iter().map(|entry| serde_json::json!({
                        "unit": entry.unit_id, "evidence": entry,
                    })).collect::<Vec<_>>(),
                }))
                .collect::<Vec<_>>()
        );
        check()?;
        Ok(Self {
            leg,
            occurrences,
            projections,
            links: prepared.links,
            publications: prepared.publications,
            incomplete_hidden_hits,
            selected,
        })
    }
}

/// Retains the original local selection for a final read-only comparison. Checks
/// bracket retrieval stages under the caller's existing budget; no model is invoked.
pub(crate) fn lexical_with_selection(
    store: &Store,
    parsed: &query::Parsed,
    k: usize,
    scope: Option<&crate::store::SearchScope>,
    settings: &Search,
    exclude_ids: &[i64],
    check: &dyn Fn() -> Result<(), Error>,
) -> Result<(Vec<RankedCandidate>, Selection), Error> {
    lexical_selection(store, parsed, k, scope, settings, exclude_ids, Some(check))
}

fn lexical_selection(
    store: &Store,
    parsed: &query::Parsed,
    k: usize,
    scope: Option<&crate::store::SearchScope>,
    settings: &Search,
    exclude_ids: &[i64],
    work: Option<&dyn Fn() -> Result<(), Error>>,
) -> Result<(Vec<RankedCandidate>, Selection), Error> {
    let window = lexical_candidates(store, parsed, k, scope, settings, exclude_ids, work)?;
    let check = work.unwrap_or(&|| Ok(()));
    let hits = collapse::finish(window.hits, k, settings, &window.prepared)?;
    let selection = Selection::capture(
        window.leg,
        window.occurrences,
        window.projections,
        window.prepared,
        &hits,
        check,
    )?;
    Ok((hits, selection))
}

/// Retains direct lexical admission before collapse for matched ranking diagnostics.
/// Candidate collection uses the requested k, including the strict/relaxed decision.
pub(crate) struct LexicalCandidates {
    pub(crate) hits: Vec<RankedCandidate>,
    pub(crate) prepared: collapse::Prepared,
    leg: LexicalLeg,
    occurrences: Vec<StoredUnit>,
    projections: Vec<StoredUnit>,
}

/// Shares the exact production lexical/graph pool; no reranker window is substituted.
pub(crate) fn lexical_candidates(
    store: &Store,
    parsed: &query::Parsed,
    k: usize,
    scope: Option<&crate::store::SearchScope>,
    settings: &Search,
    exclude_ids: &[i64],
    work: Option<&dyn Fn() -> Result<(), Error>>,
) -> Result<LexicalCandidates, Error> {
    let check = work.unwrap_or(&|| Ok(()));
    check()?;
    if k == 0 || k as u128 > u128::from(settings.max_k) {
        return Err(Error::InvalidK {
            max: settings.max_k,
        });
    }
    let excluded = exclude_ids.iter().copied().collect::<BTreeSet<_>>();
    let leg = lexical_leg_checked(store, parsed, k, scope, settings, exclude_ids, work)?;
    check()?;
    let fused = fusion::merge(
        fusion::Lists {
            lexical: &leg.ordered,
            dense: &[],
            graph: &leg.graph,
        },
        fusion::Settings {
            mode: fusion::Mode::Hybrid,
            rrf_k: settings.rrf_k,
            graph_weight: settings.graph_prior_weight,
            exclude: &excluded,
            limit: LEXICAL_LIMIT + 100,
        },
    )?;
    let ids = fused.iter().map(|hit| hit.id).collect::<Vec<_>>();
    let mut units: BTreeMap<i64, StoredUnit> = local_store_read(
        work,
        || store.units_by_ids(&ids),
        |check| store.units_by_ids_checked(&ids, check),
    )?
    .into_iter()
    .map(|unit| (unit.id, unit))
    .collect();
    check()?;
    let mut occurrences = Vec::new();
    if work.is_some() {
        for unit in units.values() {
            check()?;
            let mut unit: StoredUnit = unit.clone();
            unit.ladder_level = 0;
            occurrences.push(unit);
        }
    }
    let hits = materialize(fused, &leg.exact, &mut units);
    let prepared = collapse::prepare_with_work(
        store,
        &hits.iter().map(|hit| hit.unit.id).collect::<Vec<_>>(),
        &excluded,
        work,
    )?;
    let projections = if work.is_some() {
        prepared.occurrence_evidence(check)?
    } else {
        Vec::new()
    };
    Ok(LexicalCandidates {
        hits,
        prepared,
        leg,
        occurrences,
        projections,
    })
}

#[derive(Default, PartialEq)]
struct LexicalLeg {
    ordered: Vec<i64>,
    graph: Vec<i64>,
    exact: BTreeSet<i64>,
    exact_associations: Vec<Vec<String>>,
    graph_associations: Vec<String>,
}
// Preserve unrestricted callers' busy policy and the original query error when
// a checked snapshot performs its final callback after an earlier refusal.
fn local_store_read<T>(
    work: Option<&dyn Fn() -> Result<(), Error>>,
    ordinary: impl FnOnce() -> Result<T, store::Error>,
    checked: impl FnOnce(&dyn Fn() -> Result<(), store::Error>) -> Result<T, store::Error>,
) -> Result<T, Error> {
    let Some(work) = work else {
        return ordinary().map_err(Into::into);
    };
    let refused = std::cell::RefCell::new(None);
    let result = checked(&|| {
        if refused.borrow().is_some() {
            return Err(store::Error::L1InputsChanged);
        }
        if let Err(error) = work() {
            *refused.borrow_mut() = Some(error);
            return Err(store::Error::L1InputsChanged);
        }
        Ok(())
    });
    if let Some(error) = refused.into_inner() {
        return Err(error);
    }
    result.map_err(Into::into)
}

fn lexical_leg_checked(
    store: &Store,
    parsed: &query::Parsed,
    k: usize,
    scope: Option<&crate::store::SearchScope>,
    settings: &Search,
    exclude_ids: &[i64],
    work: Option<&dyn Fn() -> Result<(), Error>>,
) -> Result<LexicalLeg, Error> {
    let check = work.unwrap_or(&|| Ok(()));
    check()?;
    let Some(query) = parsed.lexical.as_deref() else {
        return Ok(LexicalLeg::default());
    };
    let fts = |query: &str| {
        local_store_read(
            work,
            || store.query_fts_scoped(query, LEXICAL_LIMIT, scope, exclude_ids),
            |check| store.query_fts_scoped_checked(query, LEXICAL_LIMIT, scope, exclude_ids, check),
        )
    };
    let mut lexical = fts(&build_match(query))?;
    check()?;
    if lexical.len() < k {
        lexical = fts(&parsed.relaxed_match())?;
    }
    check()?;
    let mut exact = BTreeSet::new();
    let mut ordered = Vec::new();
    let mut exact_associations = Vec::new();
    for term in &parsed.ident_terms {
        check()?;
        let (ids, associations) = local_store_read(
            work,
            || {
                store
                    .exact_name_units(term, LEXICAL_LIMIT, scope, exclude_ids)
                    .map(|ids| (ids, Vec::new()))
            },
            |check| {
                store.exact_name_selection_checked(term, LEXICAL_LIMIT, scope, exclude_ids, check)
            },
        )?;
        exact_associations.push(associations);
        for id in ids {
            check()?;
            if exact.insert(id) {
                ordered.push(id);
            }
        }
    }
    check()?;
    let mut seen = exact.clone();
    for hit in lexical {
        check()?;
        if seen.insert(hit.unit_id) {
            ordered.push(hit.unit_id);
        }
    }
    ordered.truncate(LEXICAL_LIMIT);
    // Preserve lexical rank, never the rowid ordering of the candidate map.
    let seed_units = local_store_read(
        work,
        || store.symbol_seeds(&ordered, 20),
        |check| store.symbol_seeds_checked(&ordered, 20, check),
    )?;
    check()?;
    let (graph_hits, graph_associations) = if settings.graph_prior_weight > 0.0 {
        local_store_read(
            work,
            || {
                store
                    .graph_neighbours(&seed_units, 100, scope, exclude_ids)
                    .map(|hits| (hits, Vec::new()))
            },
            |check| store.graph_selection_checked(&seed_units, 100, scope, exclude_ids, check),
        )?
    } else {
        (Vec::new(), Vec::new())
    };
    let graph = graph_hits.into_iter().map(|hit| hit.unit_id).collect();
    check()?;
    Ok(LexicalLeg {
        ordered,
        graph,
        exact,
        exact_associations,
        graph_associations,
    })
}

fn materialize(
    fused: Vec<fusion::Hit>,
    exact: &BTreeSet<i64>,
    units: &mut BTreeMap<i64, StoredUnit>,
) -> Vec<RankedCandidate> {
    fused
        .into_iter()
        .filter_map(|hit| {
            units.remove(&hit.id).map(|unit| {
                let mut candidate = RankedCandidate {
                    unit,
                    score: hit.score,
                    rrf_score: hit.score,
                    rerank_score: None,
                    rerank_fallback: None,
                    lexical: hit.scores.lexical.is_some(),
                    graph: hit.scores.graph.is_some(),
                    dense: hit.scores.dense.is_some(),
                    exact_name: exact.contains(&hit.id),
                    scores: hit.scores,
                    summary: false,
                    chunk: false,
                    matches: Vec::new(),
                };
                candidate.matches.push(MatchEvidence {
                    unit_id: candidate.unit.id,
                    id: navigation::public_id(&candidate.unit),
                    rrf_score: candidate.score,
                    rrf: candidate.scores.clone(),
                    rerank_score: None,
                    why: reasons(&candidate),
                });
                candidate
            })
        })
        .collect()
}

fn identifier_terms(query: &str) -> Vec<&str> {
    query
        .split(|c: char| !(c.is_alphanumeric() || matches!(c, '_' | '$' | '.')))
        .filter(|term| {
            let mut chars = term.chars();
            chars
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic() || matches!(c, '_' | '$'))
                && (term.contains(['_', '.']) || chars.any(|c| c.is_uppercase()))
        })
        .collect()
}

#[cfg(test)]
mod tests;

/// Formats retrieval evidence once for CLI, MCP and benchmark response parity.
pub(crate) fn format_response(
    store: &Store,
    query: &str,
    hits: &[RankedCandidate],
) -> Result<serde_json::Value, Error> {
    let deferred = store
        .l1_deferred_files()?
        .into_iter()
        .map(|file| (file.path, file.reason))
        .collect::<BTreeMap<_, _>>();
    let response = serde_json::json!({
        "query": query,
        "hits": hits.iter().map(|candidate| serde_json::json!({
            "id": navigation::public_id(&candidate.unit),
            "path": candidate.unit.path,
            "lines": [candidate.unit.line_start, candidate.unit.line_end],
            "kind": candidate.unit.kind,
            "header": candidate.unit.header,
            "snippet": snippet(&candidate.unit.text),
            "score": candidate.score,
            "why": reasons(candidate),
            "l1_deferred_reason": candidate.unit.path.as_ref().and_then(|path| deferred.get(path)),
        })).collect::<Vec<_>>(),
    });
    Ok(response)
}

/// Adds per-leg RRF positions/contributions for verbose CLI/MCP output while
/// retaining the shared compact response for ordinary calls and benchmarks.
pub(crate) fn format_response_verbose(
    store: &Store,
    query: &str,
    hits: &[RankedCandidate],
    verbose: bool,
) -> Result<serde_json::Value, Error> {
    let mut response = format_response(store, query, hits)?;
    if verbose && let Some(values) = response["hits"].as_array_mut() {
        for (value, candidate) in values.iter_mut().zip(hits) {
            value["rrf"] = serde_json::json!(candidate.scores);
            if let Some(score) = candidate.rerank_score {
                value["rrf_score"] = serde_json::json!(candidate.rrf_score);
                value["rerank_score"] = serde_json::json!(score);
            }
            if candidate.summary || candidate.chunk || candidate.matches.len() > 1 {
                value["matches"] = serde_json::json!(candidate.matches);
            }
        }
    }
    Ok(response)
}

fn reasons(candidate: &RankedCandidate) -> Vec<&'static str> {
    let mut reasons = Vec::new();
    if candidate.lexical {
        reasons.push("lexical");
    }
    if candidate.graph {
        reasons.push("graph");
    }
    if candidate.dense {
        reasons.push("dense");
    }
    if candidate.exact_name {
        reasons.push("exact_name");
    }
    if candidate.rerank_score.is_some() {
        reasons.push("reranked");
    }
    if let Some(reason) = candidate.rerank_fallback {
        reasons.push(reason);
    }
    if candidate.summary {
        reasons.push("summary");
    }
    if candidate.chunk {
        reasons.push("chunk");
    }
    reasons
}

fn snippet(text: &str) -> String {
    const MAX_TOKENS: usize = 60;
    let text = text.lines().take(4).collect::<Vec<_>>().join("\n");
    if crate::treat::count_tokens(&text) <= MAX_TOKENS {
        return text;
    }
    let boundaries = text.char_indices().map(|(at, _)| at).collect::<Vec<_>>();
    let mut low = 0;
    let mut high = boundaries.len();
    let mut retained = "…".to_owned();
    // Keep an actually measured feasible prefix. Token counts need not be
    // monotonic across BPE boundaries; no maximal-prefix claim is required.
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        let candidate = format!("{}…", &text[..boundaries[middle]]);
        if crate::treat::count_tokens(&candidate) <= MAX_TOKENS {
            low = middle;
            retained = candidate;
        } else {
            high = middle;
        }
    }
    retained
}
