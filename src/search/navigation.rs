//! Shared open, symbol and reference navigation; see `v1.md` §9–10.1.
mod continuation;
mod freshness;
use crate::sync::source;
pub(crate) use freshness::{
    L1Verifier, ReferenceProof, hit_authority, map_authority, reference_authority_cached_with,
    reference_authority_with, verify_hit, verify_map_file, verify_symbol_with,
};
#[cfg(test)]
pub(crate) use freshness::{verify_reference_group, verify_symbol};

use super::Error;
use crate::store::{self, GraphEdgeKind, Store, StoredUnit, ToolSymbol};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
    path::Path,
};
type Result<T> = std::result::Result<T, Error>;
/// Indexed occurrence text with consistent caps and deferred-state semantics.
#[derive(Debug, Serialize, PartialEq, Eq)]
pub(crate) struct OpenResult {
    /// Retained read evidence; never part of the public navigation payload.
    #[serde(skip)]
    pub(crate) occurrence: StoredUnit,
    #[serde(skip)]
    pub(crate) source: Option<store::OpenSource>,
    #[serde(skip)]
    pub(crate) history_source: Option<[u8; 32]>,
    #[serde(skip)]
    pub(crate) selected_symbol: Option<i64>,
    #[serde(skip)]
    pub(crate) summary_evidence: Option<super::summaries::Observation>,
    pub(crate) id: String,
    pub(crate) path: Option<String>,
    pub(crate) breadcrumb: String,
    pub(crate) lines: [Option<i64>; 2],
    /// Original indexed locator, not the source span of a normalized text page.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) indexed_lines: Option<[Option<i64>; 2]>,
    /// One-based lines in the extracted payload, independent of source layout.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) extracted_lines: Option<[usize; 2]>,
    pub(crate) kind: String,
    /// Immutable member locators for composites; each remains independently openable.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) sources: Vec<HistorySource>,
    pub(crate) header: String,
    pub(crate) text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) summary: Option<String>,
    pub(crate) l1_deferred_reason: Option<String>,
    pub(crate) truncated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) continuation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) byte_range: Option<[usize; 2]>,
}
/// One verified immutable contributor to the opened composite.
#[derive(Debug, Serialize, PartialEq, Eq)]
pub(crate) struct HistorySource {
    pub(crate) id: String,
    pub(crate) kind: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub(crate) sha: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) label: Option<String>,
}
/// Approximate symbol navigation data, independent of presentation.
#[derive(Debug, Serialize, PartialEq)]
pub(crate) struct SymbolResult {
    /// Original selected row, including its occurrence association.
    #[serde(skip)]
    pub(crate) selected: ToolSymbol,
    /// Private completed-source and occurrence authority retained for final validation.
    #[serde(skip)]
    pub(crate) evidence: Option<(StoredUnit, store::OpenSource)>,
    pub(crate) id: String,
    pub(crate) path: String,
    pub(crate) lines: [i64; 2],
    pub(crate) kind: String,
    pub(crate) signature: String,
    pub(crate) doc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) summary: Option<String>,
    #[serde(skip)]
    pub(crate) summary_evidence: Option<super::summaries::Observation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) history: Option<serde_json::Value>,
    pub(crate) tests: Vec<String>,
    pub(crate) documented_in: Vec<Documentation>,
    #[serde(skip)]
    pub(crate) links: Vec<LinkEvidence>,
    pub(crate) rank: f64,
    pub(crate) callers: usize,
    pub(crate) callees: usize,
    pub(crate) approximate: bool,
}
/// An independently openable documentation section mentioning this definition.
#[derive(Debug, Serialize, PartialEq, Eq)]
pub(crate) struct Documentation {
    pub(crate) id: String,
    pub(crate) header: String,
}
/// Private linkage and source evidence for final selection validation.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct LinkEvidence {
    pub(crate) edge: store::GraphEdge,
    pub(crate) unit: StoredUnit,
    pub(crate) source: store::OpenSource,
}
/// Reference locations grouped by path and edge kind, with the pre-limit total.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct RefsResult {
    /// Original contributors, retained independently of line deduplication and k.
    #[serde(skip)]
    pub(crate) contributors: BTreeMap<(String, &'static str), ReferenceContributors>,
    pub(crate) groups: Vec<ReferenceGroup>,
    pub(crate) total: usize,
    pub(crate) approximate: bool,
}
/// Private occurrence and graph-association evidence for one reference group.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ReferenceContributors {
    pub(crate) occurrences: BTreeMap<i64, StoredUnit>,
    pub(crate) edges: Vec<(store::GraphEdge, Option<i64>)>,
}

/// Sorted, deduplicated lines at a reference source.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct ReferenceGroup {
    pub(crate) path: String,
    pub(crate) lines: BTreeSet<i64>,
    pub(crate) kind: &'static str,
}
/// Opens an occurrence, symbol ID or path/range while enforcing the byte cap.
#[cfg(test)]
pub(crate) fn open(store: &Store, target: &str, max_bytes: usize) -> Result<OpenResult> {
    open_checked(store, target, max_bytes, &|| Ok(()))
}

/// Resolves a locator with caller-owned checks around storage and result boundaries.
fn open_checked(
    store: &Store,
    target: &str,
    max_bytes: usize,
    check: &dyn Fn() -> std::result::Result<(), store::Error>,
) -> Result<OpenResult> {
    check()?;
    let unit = if target.starts_with("u_") {
        store.unit_by_public_id_checked(target, check)?
    } else if target.starts_with("s_") {
        let mut symbols = store
            .tool_symbols_checked(check)?
            .into_iter()
            .filter(|symbol| public_symbol_id(symbol) == target);
        let symbol = symbols.next();
        if symbols.next().is_some() {
            return Err(Error::AmbiguousSymbol {
                target: target.to_owned(),
            });
        }
        match symbol.and_then(|symbol| symbol.unit_id) {
            Some(unit_id) => store.unit_by_id_checked(unit_id, check)?,
            None => None,
        }
    } else {
        let (path, start, end) = parse_path_range(target)?;
        // Blank lines between chunks still belong to the enclosing declaration.
        // Resolve that declaration before requiring an overlapping chunk.
        let containing = if let (Some(start), Some(end)) = (start, end) {
            store
                .tool_symbols_for_path_checked(path, check)?
                .into_iter()
                .filter(|symbol| {
                    symbol.path == path && symbol.line_start <= start && symbol.line_end >= end
                })
                .min_by_key(|symbol| {
                    (
                        symbol.line_end - symbol.line_start,
                        symbol.line_start,
                        symbol.id,
                    )
                })
                .and_then(|symbol| symbol.unit_id)
        } else {
            None
        };
        match containing {
            Some(id) => store.unit_by_id_checked(id, check)?,
            None => store.unit_by_path_checked(path, start, end, check)?,
        }
    }
    .ok_or_else(|| Error::NotFound {
        target: target.to_owned(),
    })?;
    let text = truncate_utf8(&unit.text, max_bytes);
    let deferred = store.open_deferred_reason(unit.path.as_deref(), check)?;
    check()?;
    Ok(OpenResult {
        occurrence: unit.clone(),
        source: None,
        sources: Vec::new(),
        history_source: None,
        selected_symbol: None,
        summary_evidence: None,
        summary: None,
        id: public_id(&unit),
        text: text.to_owned(),
        path: unit.path,
        breadcrumb: String::new(),
        lines: [unit.line_start, unit.line_end],
        kind: unit.kind,
        header: unit.header,
        l1_deferred_reason: deferred,
        truncated: text.len() < unit.text.len(),
        indexed_lines: None,
        extracted_lines: None,
        continuation_id: None,
        byte_range: None,
    })
}

pub(crate) use continuation::{open_source, repair_path};

/// Expands code ranges to the innermost containing symbol and adds surrounding
/// lines only after verifying the bounded source bytes against completed L1.
fn read_source(
    store: &Store,
    root: &Path,
    target: &str,
    context: u64,
    caps: (usize, u64),
    check: &dyn Fn() -> std::result::Result<(), store::Error>,
) -> Result<OpenResult> {
    let (max_bytes, max_file_bytes) = caps;
    check()?;
    let initial = open_checked(store, target, max_bytes, check)?;
    let unavailable = || Error::OpenSourceUnavailable;
    let before = store
        .open_source_checked(&initial.id, check)?
        .ok_or_else(unavailable)?;
    // Notebook cells are chunk units, but their lines describe extracted cell
    // text rather than the containing JSON. Only actual code files use ranges
    // from the source bytes; extracted documents retain their indexed payload.
    let code = before.kind == "code" && matches!(initial.kind.as_str(), "chunk" | "symbol");
    if !code && context != 0 {
        return Err(Error::InvalidQuery {
            reason: "context requires an indexed code source",
        });
    }
    // Resolve again inside the generation fence: the first lookup only finds the
    // file identity. Publication between that lookup and the fence is harmless.
    let mut result = open_checked(store, target, max_bytes, check)?;
    if result.id != initial.id || result.path != initial.path {
        return Err(unavailable());
    }
    let path = result.path.as_deref().ok_or_else(unavailable)?;
    let file = source::open_regular(root, Path::new(path)).map_err(|_| unavailable())?;
    check()?;
    let metadata = file.metadata().map_err(|_| unavailable())?;
    if !metadata.is_file() || metadata.len() > max_file_bytes {
        return Err(unavailable());
    }
    check()?;
    let mut bytes = Vec::new();
    file.take(max_file_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|_| unavailable())?;
    check()?;
    if bytes.len() as u128 > u128::from(max_file_bytes)
        || blake3::hash(&bytes).as_bytes().as_slice() != before.hash
    {
        return Err(unavailable());
    }
    if !code {
        // Extracted/configuration text need not occur verbatim in the source.
        // Keep its indexed locator distinct from the returned payload's lines.
        result.indexed_lines = Some(result.lines);
        result.lines = [None, None];
        if store.open_source_checked(&result.id, check)?.as_ref() != Some(&before) {
            return Err(unavailable());
        }
        return Ok(result);
    }
    let source = std::str::from_utf8(&bytes).map_err(|_| unavailable())?;
    let (_, requested_start, requested_end) = parse_path_range(target)?;
    let mut start = requested_start
        .or(result.lines[0])
        .ok_or_else(unavailable)?;
    let mut end = requested_end.or(result.lines[1]).ok_or_else(unavailable)?;
    let symbols = store.tool_symbols_for_path_checked(path, check)?;
    let symbol = if target.starts_with("s_") {
        symbols
            .iter()
            .find(|symbol| symbol.path == path && public_symbol_id(symbol) == target)
    } else {
        symbols
            .iter()
            .filter(|symbol| {
                symbol.path == path && symbol.line_start <= start && symbol.line_end >= end
            })
            .min_by_key(|symbol| {
                (
                    symbol.line_end - symbol.line_start,
                    symbol.line_start,
                    symbol.id,
                )
            })
    };
    if let Some(symbol) = symbol {
        result.selected_symbol = Some(symbol.id);
        start = symbol.line_start;
        end = symbol.line_end;
    }
    check()?;
    let lines = source.split_inclusive('\n').collect::<Vec<_>>();
    let start = usize::try_from(start).map_err(|_| unavailable())?;
    let end = usize::try_from(end).map_err(|_| unavailable())?;
    if start == 0 || end < start || end > lines.len() {
        return Err(Error::InvalidQuery {
            reason: "line range is outside the source",
        });
    }
    let context = usize::try_from(context).unwrap_or(usize::MAX);
    let first = start.saturating_sub(context).max(1);
    let last = end.saturating_add(context).min(lines.len());
    let text = lines[first - 1..last].concat();
    let bounded = truncate_utf8(&text, max_bytes);
    result.truncated = bounded.len() < text.len();
    result.text = bounded.to_owned();
    let returned_lines = bounded.split_inclusive('\n').count();
    result.lines = if returned_lines == 0 {
        [None, None]
    } else {
        [
            Some(first as i64),
            Some((first + returned_lines - 1) as i64),
        ]
    };
    if store.open_source_checked(&result.id, check)?.as_ref() != Some(&before) {
        return Err(unavailable());
    }
    check()?;
    Ok(result)
}
/// Resolves indexed symbol metadata without reading unverified repository files.
/// The public service verifies ranges and fills documentation before returning it.
#[cfg(test)]
pub(crate) fn symbol(store: &Store, target: &str) -> Result<Vec<SymbolResult>> {
    symbol_checked(store, target, &|| Ok(()))
}

/// Resolves symbols under original query checks, including matching and graph counts.
pub(crate) fn symbol_checked(
    store: &Store,
    target: &str,
    check: &dyn Fn() -> std::result::Result<(), store::Error>,
) -> Result<Vec<SymbolResult>> {
    let matches = matching_symbols_checked(store, target, check)?;
    let ids = matches.iter().map(|symbol| symbol.id).collect::<Vec<_>>();
    let names = matches
        .iter()
        .map(|symbol| symbol.name.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let edges = store.graph_edges_for_symbols_checked(&ids, &names, check)?;
    let mut response = Vec::new();
    for symbol in matches {
        check()?;
        let mut callers = 0;
        let mut callees = 0;
        for edge in &edges {
            check()?;
            if edge.kind == GraphEdgeKind::DocMention {
                continue;
            }
            callers +=
                usize::from(edge.dst_symbol == Some(symbol.id) || edge.dst_name == symbol.name);
            callees += usize::from(edge.src_symbol == symbol.id);
        }
        response.push(SymbolResult {
            selected: symbol.clone(),
            evidence: None,
            id: public_symbol_id(&symbol),
            path: symbol.path.clone(),
            lines: [symbol.line_start, symbol.line_end],
            kind: symbol.kind.clone(),
            signature: symbol
                .signature
                .as_deref()
                .unwrap_or(&symbol.name)
                .to_owned(),
            doc: String::new(),
            summary: None,
            summary_evidence: None,
            history: None,
            tests: Vec::new(),
            documented_in: Vec::new(),
            links: Vec::new(),
            rank: symbol.pagerank,
            callers,
            callees,
            approximate: true,
        });
    }
    check()?;
    Ok(response)
}
/// Collects approximate references with deterministic grouping and group limits.
#[cfg(test)]
pub(crate) fn refs(
    store: &Store,
    target: &str,
    k: Option<usize>,
    tests_only: bool,
) -> Result<RefsResult> {
    refs_checked(store, target, k, tests_only, &|| Ok(()))
}

/// Collects reference groups with query-owned reader admission and cancellation.
pub(crate) fn refs_checked(
    store: &Store,
    target: &str,
    k: Option<usize>,
    tests_only: bool,
    check: &dyn Fn() -> std::result::Result<(), store::Error>,
) -> Result<RefsResult> {
    check()?;
    if k == Some(0) {
        return Err(Error::InvalidReferenceLimit);
    }
    let matches = matching_symbols_checked(store, target, check)?;
    let mut target_ids = BTreeSet::new();
    let mut target_names = BTreeSet::new();
    let mut target_definition_units = BTreeSet::new();
    for symbol in &matches {
        check()?;
        target_ids.insert(symbol.id);
        target_names.insert(symbol.name.clone());
        if let Some(unit) = symbol.unit_id {
            target_definition_units.insert(unit);
        }
    }
    if !target.starts_with("s_") {
        target_names.insert(target.to_owned());
    }
    let target_ids = target_ids.into_iter().collect::<Vec<_>>();
    let target_names = target_names.into_iter().collect::<Vec<_>>();
    for name in &target_names {
        for symbol in store.tool_symbols_named_checked(name, check)? {
            check()?;
            if let Some(unit) = symbol.unit_id {
                target_definition_units.insert(unit);
            }
        }
    }
    let edges = store.graph_edges_for_symbols_checked(&target_ids, &target_names, check)?;
    let source_ids = edges
        .iter()
        .map(|edge| edge.src_symbol)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut symbols_by_id = BTreeMap::new();
    for symbol in store.tool_symbols_by_ids_checked(&source_ids, check)? {
        check()?;
        symbols_by_id.insert(symbol.id, symbol);
    }
    let mut selected_edges = Vec::new();
    for edge in &edges {
        check()?;
        let matches_target = edge
            .dst_symbol
            .is_some_and(|symbol_id| target_ids.contains(&symbol_id))
            || target_names.contains(&edge.dst_name);
        if !matches_target {
            continue;
        }
        let Some(kind) = reference_kind(edge.kind) else {
            continue;
        };
        if tests_only && kind != "test_of" {
            continue;
        }
        let Some(source) = symbols_by_id.get(&edge.src_symbol) else {
            continue;
        };
        selected_edges.push((edge, source, kind));
    }
    let edge_unit_ids = selected_edges
        .iter()
        .filter_map(|(_, source, _)| source.unit_id)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let edge_units = store
        .units_by_ids_checked(&edge_unit_ids, check)?
        .into_iter()
        .map(|mut unit| {
            unit.ladder_level = 0;
            (unit.id, unit)
        })
        .collect::<BTreeMap<_, _>>();
    let mut groups = BTreeMap::<(String, &'static str), BTreeSet<i64>>::new();
    let mut edge_source_units = BTreeSet::new();
    let mut contributors = BTreeMap::<_, ReferenceContributors>::new();
    for (edge, source, kind) in selected_edges {
        check()?;
        let evidence = contributors.entry((source.path.clone(), kind)).or_default();
        evidence.edges.push((edge.clone(), source.unit_id));
        if let Some(unit_id) = source.unit_id {
            edge_source_units.insert(unit_id);
            if let Some(unit) = edge_units.get(&unit_id) {
                // L2 publication does not invalidate completed L1 evidence.
                evidence
                    .occurrences
                    .entry(unit.id)
                    .or_insert_with(|| unit.clone());
            }
        }
        groups
            .entry((source.path.clone(), kind))
            .or_default()
            .insert(edge.line.unwrap_or(source.line_start));
    }
    if !tests_only {
        let mut fts_ids = Vec::new();
        for name in &target_names {
            check()?;
            let query = exact_identifier_match(name);
            for hit in store.query_fts_checked(&query, 200, check)? {
                check()?;
                fts_ids.push(hit.unit_id);
            }
        }
        let unique_fts_ids = fts_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let fts_units = store
            .units_by_ids_checked(&unique_fts_ids, check)?
            .into_iter()
            .map(|unit| (unit.id, unit))
            .collect::<BTreeMap<_, _>>();
        for unit_id in fts_ids {
            check()?;
            let Some(mut unit) = fts_units.get(&unit_id).cloned() else {
                continue;
            };
            let Some(path) = unit.path.clone() else {
                continue;
            };
            let Some(line) = unit.line_start else {
                continue;
            };
            let is_target_definition = target_definition_units.contains(&unit.id);
            if edge_source_units.contains(&unit.id) || is_target_definition {
                continue;
            }
            unit.ladder_level = 0;
            contributors
                .entry((path.clone(), "ref"))
                .or_default()
                .occurrences
                .insert(unit.id, unit);
            groups.entry((path, "ref")).or_default().insert(line);
        }
    }
    let mut total = 0;
    let mut response_groups = Vec::new();
    for ((path, kind), lines) in groups {
        check()?;
        total += lines.len();
        response_groups.push(ReferenceGroup { path, lines, kind });
    }
    if let Some(k) = k {
        response_groups.truncate(k);
    }
    check()?;
    Ok(RefsResult {
        contributors,
        groups: response_groups,
        total,
        approximate: true,
    })
}

/// Request-local reference selection, fenced against local and external writes.
/// Source files and final occurrence authority are still checked by the caller.
#[derive(Clone)]
pub(crate) struct RefsCache {
    revision: store::modules::Revision,
    target: String,
    k: Option<usize>,
    tests_only: bool,
    result: RefsResult,
}

/// Reuses a reference selection only while its inputs and writer revision match.
pub(crate) fn refs_checked_reuse(
    store: &Store,
    cache: &mut Option<RefsCache>,
    target: &str,
    k: Option<usize>,
    tests_only: bool,
    check: &dyn Fn() -> std::result::Result<(), store::Error>,
) -> Result<RefsResult> {
    check()?;
    let before = store.revision_checked(check)?;
    let result = match cache.as_ref() {
        Some(cached)
            if cached.revision == before
                && cached.target == target
                && cached.k == k
                && cached.tests_only == tests_only =>
        {
            cached.result.clone()
        }
        _ => refs_checked(store, target, k, tests_only, check)?,
    };
    let after = store.revision_checked(check)?;
    *cache = (before == after).then(|| RefsCache {
        revision: after,
        target: target.to_owned(),
        k,
        tests_only,
        result: result.clone(),
    });
    Ok(result)
}

/// Orders signatures by rank with deterministic source tie breaks.
pub(super) fn compare_map_symbols(left: &ToolSymbol, right: &ToolSymbol) -> std::cmp::Ordering {
    right
        .pagerank
        .total_cmp(&left.pagerank)
        .then_with(|| left.name.cmp(&right.name))
        .then_with(|| left.line_start.cmp(&right.line_start))
        .then_with(|| left.id.cmp(&right.id))
}

fn matching_symbols_checked(
    store: &Store,
    target: &str,
    check: &dyn Fn() -> std::result::Result<(), store::Error>,
) -> Result<Vec<ToolSymbol>> {
    let public = target.starts_with("s_");
    let mut ids = BTreeSet::new();
    if public {
        for symbol in store.tool_symbols_checked(check)? {
            check()?;
            if public_symbol_id(&symbol) == target {
                ids.insert(symbol.id);
            }
        }
    } else {
        for symbol in store.tool_symbols_named_checked(target, check)? {
            check()?;
            ids.insert(symbol.id);
        }
    }
    if !public && target.chars().count() >= 3 {
        for hit in store.query_fts_names_checked(&fts_phrase(target), 100, check)? {
            check()?;
            if hit.kind == store::FtsNameKind::Symbol {
                ids.insert(hit.ref_id);
            }
        }
    }
    let ids = ids.into_iter().collect::<Vec<_>>();
    let mut matched = Vec::new();
    for symbol in store.tool_symbols_by_ids_checked(&ids, check)? {
        check()?;
        if symbol.kind != "doc" {
            matched.push(symbol);
        }
    }
    matched.sort_by(|left, right| {
        (right.name == target)
            .cmp(&(left.name == target))
            .then_with(|| compare_map_symbols(left, right))
    });
    check()?;
    Ok(matched)
}

/// Formats the specified stable symbol identity for navigation consumers.
pub(crate) fn public_symbol_id(symbol: &ToolSymbol) -> String {
    let qualified = symbol.qualified.as_deref().unwrap_or(&symbol.name);
    let mut hasher = blake3::Hasher::new();
    for value in [&symbol.path, qualified, &symbol.kind] {
        hasher.update(value.as_bytes());
        hasher.update(b"|");
    }
    let hash = hasher.finalize().to_hex();
    format!("s_{}", &hash[..12])
}

fn exact_identifier_match(name: &str) -> String {
    format!("idents:{}", fts_phrase(name))
}

fn fts_phrase(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

fn reference_kind(kind: GraphEdgeKind) -> Option<&'static str> {
    match kind {
        GraphEdgeKind::Ref => Some("ref"),
        GraphEdgeKind::Import => Some("import"),
        GraphEdgeKind::TestOf => Some("test_of"),
        GraphEdgeKind::DocMention => None,
    }
}

fn symbol_doc_comment(source: &str, line_start: i64) -> String {
    let lines = source.lines().collect::<Vec<_>>();
    let Some(mut index) = usize::try_from(line_start.saturating_sub(1)).ok() else {
        return String::new();
    };
    index = index.min(lines.len());
    let mut comments = Vec::new();
    while index > 0 {
        index -= 1;
        let line = lines[index].trim();
        if line.is_empty() && comments.is_empty() {
            continue;
        }
        let comment = line
            .strip_prefix("///")
            .or_else(|| line.strip_prefix("//!"))
            .or_else(|| line.strip_prefix("/**"))
            .or_else(|| line.strip_prefix('*'));
        if let Some(comment) = comment {
            comments.push(comment.trim().trim_end_matches("*/").trim().to_owned());
        } else {
            break;
        }
    }
    comments.reverse();
    comments
        .into_iter()
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Formats an occurrence ID consistently across search and navigation.
pub(crate) fn public_id(unit: &StoredUnit) -> String {
    let prefix = unit
        .hash
        .iter()
        .take(6)
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("u_{prefix}")
}

fn parse_path_range(target: &str) -> Result<(&str, Option<i64>, Option<i64>)> {
    let Some((path, range)) = target.rsplit_once(':') else {
        return Ok((target, None, None));
    };
    let Some((start, end)) = range.split_once('-') else {
        return Ok((target, None, None));
    };
    match (start.parse(), end.parse()) {
        (Ok(start), Ok(end)) if start > 0 && end >= start => Ok((path, Some(start), Some(end))),
        _ => Err(Error::InvalidQuery {
            reason: "line bounds must be positive increasing integers",
        }),
    }
}

fn truncate_utf8(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    text.get(..max_bytes).unwrap_or_else(|| {
        text.char_indices()
            .take_while(|(index, _)| *index < max_bytes)
            .last()
            .map_or("", |(index, _)| &text[..index])
    })
}
