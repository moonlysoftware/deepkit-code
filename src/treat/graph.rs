//! Tags-derived code graph extraction and PageRank; see v1.md §6.4.

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
    sync::{Arc, OnceLock},
};

use petgraph::graph::{DiGraph, NodeIndex};
use tree_sitter::{Node, Parser};
use tree_sitter_tags::{TagsConfiguration, TagsContext};

use super::ast::{AstLanguage, is_import_kind};

const DAMPING: f64 = 0.85;
const MAX_ITERATIONS: usize = 50;
const CONVERGENCE: f64 = 1e-6;

/// The approximate relationship represented by one code-graph edge.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum EdgeKind {
    Ref,
    Import,
}

/// A definition extracted from a grammar tags query.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ExtractedSymbol {
    pub(crate) name: String,
    pub(crate) qualified: String,
    pub(crate) kind: String,
    pub(crate) signature: String,
    pub(crate) line_start: usize,
    pub(crate) line_end: usize,
}

/// A reference or import whose source is an index into the file graph symbols.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ExtractedEdge {
    pub(crate) source_index: usize,
    pub(crate) dst_name: String,
    pub(crate) kind: EdgeKind,
    pub(crate) line: usize,
    pub(crate) import_path: Option<String>,
}

/// The deterministic graph contribution parsed from one source file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FileGraph {
    pub(crate) symbols: Vec<ExtractedSymbol>,
    pub(crate) edges: Vec<ExtractedEdge>,
    pub(crate) parse_error: bool,
}

/// A persisted symbol candidate used to resolve name-only edges.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResolutionCandidate {
    pub(crate) id: i64,
    pub(crate) file_id: i64,
    pub(crate) path: String,
    pub(crate) name: String,
}

/// A graph edge with its best-effort persisted destination, if any.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResolvedEdge {
    pub(crate) source_index: usize,
    pub(crate) dst_name: String,
    pub(crate) dst_symbol: Option<i64>,
    pub(crate) kind: EdgeKind,
    pub(crate) line: usize,
}

/// A stored symbol participating in one PageRank rebuild.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RankSymbol {
    pub(crate) id: i64,
    pub(crate) file_id: i64,
    pub(crate) name: String,
    pub(crate) pagerank: f64,
}

/// A stored approximate edge participating in one PageRank rebuild.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RankEdge {
    pub(crate) src_symbol: i64,
    pub(crate) dst_name: String,
    pub(crate) dst_symbol: Option<i64>,
    pub(crate) kind: EdgeKind,
}

/// One symbol PageRank value ready to persist.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SymbolRank {
    pub(crate) symbol_id: i64,
    pub(crate) rank: f64,
}

/// One file rank, defined as the sum of its symbols' PageRank values.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FileRank {
    pub(crate) file_id: i64,
    pub(crate) rank: f64,
}

/// A completed rank pass and the number of iterations it required.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RankResult {
    pub(crate) symbols: Vec<SymbolRank>,
    pub(crate) files: Vec<FileRank>,
    pub(crate) iterations: usize,
}

/// Graph extraction failures for one code file.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("graph source was not valid UTF-8")]
    InvalidUtf8,
    #[error(transparent)]
    Tags(#[from] tree_sitter_tags::Error),
    #[error(transparent)]
    TagsConfiguration(Arc<tree_sitter_tags::Error>),
    #[error(transparent)]
    Language(#[from] tree_sitter::LanguageError),
    #[error("tree-sitter cancelled while parsing graph tags")]
    ParseCancelled,
    #[error("tree-sitter tag range was outside its source")]
    InvalidSourceRange,
}

/// Grammar queries are immutable and expensive to compile. Cache one result per
/// linked language, while keeping each file's parser/cursor/source state private.
pub(super) fn tags_configuration(
    language: AstLanguage,
) -> Result<&'static TagsConfiguration, Error> {
    type Cached = OnceLock<Result<TagsConfiguration, Arc<tree_sitter_tags::Error>>>;
    static CONFIGS: [Cached; AstLanguage::ALL.len()] =
        [const { OnceLock::new() }; AstLanguage::ALL.len()];
    let slot = language as usize;
    CONFIGS[slot]
        .get_or_init(|| {
            TagsConfiguration::new(language.grammar(), language.tags_query(), "").map_err(Arc::new)
        })
        .as_ref()
        .map_err(|error| Error::TagsConfiguration(Arc::clone(error)))
}

/// Extracts tags-query definitions, references, and imports for a supported language.
///
/// Unknown languages deliberately return None: they retain the phase-0 lexical
/// path without inventing an approximate graph from unparsed text.
pub(crate) fn extract_file_graph(
    language_id: &str,
    path: &str,
    source: &[u8],
) -> Result<Option<FileGraph>, Error> {
    extract(language_id, path, source, true)
}

/// Extracts the same historical definition ranges without constructing reference
/// edges or resolving their owners; history only needs hunk intersections.
pub(crate) fn extract_file_symbols(
    language_id: &str,
    path: &str,
    source: &[u8],
) -> Result<Option<Vec<ExtractedSymbol>>, Error> {
    Ok(extract(language_id, path, source, false)?.map(|graph| graph.symbols))
}

fn extract(
    language_id: &str,
    path: &str,
    source: &[u8],
    with_edges: bool,
) -> Result<Option<FileGraph>, Error> {
    let Some(language) =
        AstLanguage::for_path(language_id, path).filter(|language| language.is_code())
    else {
        return Ok(None);
    };
    let source = std::str::from_utf8(source).map_err(|_| Error::InvalidUtf8)?;
    let newlines = source
        .bytes()
        .enumerate()
        .filter_map(|(i, b)| (b == b'\n').then_some(i))
        .collect::<Vec<_>>();
    let end_line = |offset: usize| {
        newlines.partition_point(|i| *i < offset.min(source.len()).saturating_sub(1)) + 1
    };
    let config = tags_configuration(language)?;
    let mut context = TagsContext::new();
    let (tags, parse_error) = context.generate_tags(config, source.as_bytes(), None)?;
    let mut definitions = Vec::new();
    let mut references = Vec::new();

    for tag in tags {
        let tag = tag?;
        let name = source
            .get(tag.name_range.clone())
            .ok_or(Error::InvalidSourceRange)?
            .to_owned();
        if name.is_empty() {
            continue;
        }
        if tag.is_definition {
            let signature = source
                .get(tag.line_range)
                .map(str::trim)
                .unwrap_or_default()
                .to_owned();
            definitions.push(PositionedSymbol {
                range: tag.range.clone(),
                symbol: ExtractedSymbol {
                    name,
                    qualified: String::new(),
                    kind: config.syntax_type_name(tag.syntax_type_id).to_owned(),
                    signature,
                    line_start: tag.span.start.row + 1,
                    line_end: end_line(tag.range.end),
                },
            });
        } else {
            references.push(PositionedReference {
                range: tag.range,
                name,
                line: tag.span.start.row + 1,
            });
        }
    }

    definitions.sort_by(|left, right| {
        left.range
            .start
            .cmp(&right.range.start)
            .then_with(|| right.range.end.cmp(&left.range.end))
            .then_with(|| left.symbol.name.cmp(&right.symbol.name))
    });
    for index in 0..definitions.len() {
        let qualified = qualified_name(&definitions, index);
        definitions[index].symbol.qualified = qualified;
    }

    let imports = import_targets(language, source)?;
    let has_unowned_reference = references
        .iter()
        .any(|reference| owner_for_range(&definitions, &reference.range).is_none());
    let needs_file_module = !imports.is_empty() || has_unowned_reference;
    let module_index = definitions.len();
    if needs_file_module {
        definitions.push(PositionedSymbol {
            range: 0..source.len(),
            symbol: ExtractedSymbol {
                name: module_name(path),
                qualified: path.to_owned(),
                kind: "module".to_owned(),
                signature: path.to_owned(),
                line_start: 1,
                line_end: end_line(source.len()),
            },
        });
    }

    if !with_edges {
        return Ok(Some(FileGraph {
            symbols: definitions.into_iter().map(|d| d.symbol).collect(),
            edges: Vec::new(),
            parse_error,
        }));
    }

    let import_paths = imports
        .iter()
        .map(|target| (target.name.clone(), target.path.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut edges = references
        .into_iter()
        .filter_map(|reference| {
            let source_index = owner_for_range(&definitions, &reference.range)
                .or_else(|| needs_file_module.then_some(module_index))?;
            Some(ExtractedEdge {
                source_index,
                import_path: import_paths.get(&reference.name).cloned(),
                dst_name: reference.name,
                kind: EdgeKind::Ref,
                line: reference.line,
            })
        })
        .collect::<Vec<_>>();
    if needs_file_module {
        edges.extend(imports.into_iter().map(|target| ExtractedEdge {
            source_index: module_index,
            dst_name: target.name,
            kind: EdgeKind::Import,
            line: target.line,
            import_path: Some(target.path),
        }));
    }
    edges.sort_by(|left, right| {
        left.source_index
            .cmp(&right.source_index)
            .then_with(|| left.line.cmp(&right.line))
            .then_with(|| left.kind.cmp(&right.kind))
            .then_with(|| left.dst_name.cmp(&right.dst_name))
    });

    Ok(Some(FileGraph {
        symbols: definitions
            .into_iter()
            .map(|definition| definition.symbol)
            .collect(),
        edges,
        parse_error,
    }))
}

/// Resolves edges by the required unique-name, same-file, then import-suffix order.
pub(crate) fn resolve_edges(
    source_file_id: i64,
    edges: &[ExtractedEdge],
    candidates: &[ResolutionCandidate],
) -> Vec<ResolvedEdge> {
    edges
        .iter()
        .map(|edge| {
            let same_name = candidates
                .iter()
                .filter(|candidate| candidate.name == edge.dst_name)
                .collect::<Vec<_>>();
            let dst_symbol = if same_name.len() == 1 {
                Some(same_name[0].id)
            } else {
                let same_file = same_name
                    .iter()
                    .copied()
                    .filter(|candidate| candidate.file_id == source_file_id)
                    .collect::<Vec<_>>();
                if same_file.len() == 1 {
                    Some(same_file[0].id)
                } else if let Some(import_path) = &edge.import_path {
                    let imported = same_name
                        .iter()
                        .copied()
                        .filter(|candidate| path_suffix_matches(&candidate.path, import_path))
                        .collect::<Vec<_>>();
                    (imported.len() == 1).then(|| imported[0].id)
                } else {
                    None
                }
            };
            ResolvedEdge {
                source_index: edge.source_index,
                dst_name: edge.dst_name.clone(),
                dst_symbol,
                kind: edge.kind,
                line: edge.line,
            }
        })
        .collect()
}

/// Runs deterministic weighted PageRank over the entire approximate symbol graph.
pub(crate) fn rank_full(symbols: &[RankSymbol], edges: &[RankEdge]) -> RankResult {
    let graph = WeightedGraph::build(symbols, edges);
    graph.rank(None, None)
}

/// Recomputes a changed symbol's undirected two-hop neighbourhood.
///
/// Ranks outside the selected neighbourhood stay at their last persisted value and
/// act as fixed boundary contributions. A full index always calls rank_full.
pub(crate) fn rank_incremental(
    symbols: &[RankSymbol],
    edges: &[RankEdge],
    changed_symbols: &BTreeSet<i64>,
) -> RankResult {
    let graph = WeightedGraph::build(symbols, edges);
    let active = graph.two_hop_neighbourhood(changed_symbols);
    let previous = symbols
        .iter()
        .map(|symbol| (symbol.id, symbol.pagerank))
        .collect::<BTreeMap<_, _>>();
    graph.rank(Some(&active), Some(&previous))
}

/// Updates the union of old/new two-hop neighbourhoods for changed source files.
/// Old topology is needed when an edit removes a call or a symbol entirely.
pub(crate) fn rank_changed_files(
    symbols: &[RankSymbol],
    edges: &[RankEdge],
    previous_symbols: &[RankSymbol],
    previous_edges: &[RankEdge],
    files: &BTreeSet<i64>,
) -> RankResult {
    let old = WeightedGraph::build(previous_symbols, previous_edges);
    let old_seeds = previous_symbols
        .iter()
        .filter(|s| files.contains(&s.file_id))
        .map(|s| s.id)
        .collect();
    let old_affected = old
        .two_hop_neighbourhood(&old_seeds)
        .into_iter()
        .map(|node| old.graph[node]);
    let graph = WeightedGraph::build(symbols, edges);
    let seeds = symbols
        .iter()
        .filter(|s| files.contains(&s.file_id))
        .map(|s| s.id)
        .collect();
    let mut active = graph.two_hop_neighbourhood(&seeds);
    active.extend(old_affected.filter_map(|id| graph.ids.get(&id).copied()));
    let previous = symbols
        .iter()
        .map(|symbol| (symbol.id, symbol.pagerank))
        .collect();
    graph.rank(Some(&active), Some(&previous))
}

struct PositionedSymbol {
    range: Range<usize>,
    symbol: ExtractedSymbol,
}

struct PositionedReference {
    range: Range<usize>,
    name: String,
    line: usize,
}

struct ImportTarget {
    name: String,
    path: String,
    line: usize,
}

struct WeightedGraph {
    graph: DiGraph<i64, f64>,
    ids: BTreeMap<i64, NodeIndex>,
    files: BTreeMap<i64, i64>,
    edge_rows: Vec<(NodeIndex, NodeIndex, f64)>,
}

impl WeightedGraph {
    fn build(symbols: &[RankSymbol], edges: &[RankEdge]) -> Self {
        let mut graph = DiGraph::new();
        let mut ids = BTreeMap::new();
        let mut files = BTreeMap::new();
        let mut names = BTreeMap::<String, Vec<NodeIndex>>::new();
        let mut ordered_symbols = symbols.to_vec();
        ordered_symbols.sort_by_key(|symbol| symbol.id);
        for symbol in &ordered_symbols {
            let node = graph.add_node(symbol.id);
            ids.insert(symbol.id, node);
            files.insert(symbol.id, symbol.file_id);
            names.entry(symbol.name.clone()).or_default().push(node);
        }
        for candidates in names.values_mut() {
            candidates.sort_by_key(|node| graph[*node]);
        }

        let mut ordered_edges = edges.to_vec();
        ordered_edges.sort_by(|left, right| {
            left.src_symbol
                .cmp(&right.src_symbol)
                .then_with(|| left.dst_name.cmp(&right.dst_name))
                .then_with(|| left.kind.cmp(&right.kind))
                .then_with(|| left.dst_symbol.cmp(&right.dst_symbol))
        });
        let mut edge_rows = Vec::new();
        for edge in ordered_edges {
            let Some(source) = ids.get(&edge.src_symbol).copied() else {
                continue;
            };
            let targets = match edge.dst_symbol.and_then(|id| ids.get(&id).copied()) {
                Some(target) => vec![target],
                None => names.get(&edge.dst_name).cloned().unwrap_or_default(),
            };
            if targets.is_empty() {
                continue;
            }
            let total_weight = match edge.kind {
                EdgeKind::Ref if edge.dst_symbol.is_some() => 1.0,
                EdgeKind::Ref => 0.3,
                EdgeKind::Import => 0.5,
            };
            let weight = total_weight / targets.len() as f64;
            for target in targets {
                graph.add_edge(source, target, weight);
                edge_rows.push((source, target, weight));
            }
        }
        Self {
            graph,
            ids,
            files,
            edge_rows,
        }
    }

    fn two_hop_neighbourhood(&self, changed_symbols: &BTreeSet<i64>) -> BTreeSet<NodeIndex> {
        let mut active = changed_symbols
            .iter()
            .filter_map(|symbol_id| self.ids.get(symbol_id).copied())
            .collect::<BTreeSet<_>>();
        let mut frontier = active.clone();
        for _ in 0..2 {
            let mut next = BTreeSet::new();
            for (source, destination, _) in &self.edge_rows {
                if frontier.contains(source) {
                    next.insert(*destination);
                }
                if frontier.contains(destination) {
                    next.insert(*source);
                }
            }
            next.retain(|node| !active.contains(node));
            active.extend(&next);
            frontier = next;
        }
        active
    }

    fn rank(
        &self,
        active: Option<&BTreeSet<NodeIndex>>,
        previous: Option<&BTreeMap<i64, f64>>,
    ) -> RankResult {
        let count = self.graph.node_count();
        if count == 0 {
            return RankResult {
                symbols: Vec::new(),
                files: Vec::new(),
                iterations: 0,
            };
        }
        let all_nodes = self.graph.node_indices().collect::<Vec<_>>();
        let active = active
            .cloned()
            .unwrap_or_else(|| all_nodes.iter().copied().collect());
        // The graph is immutable during ranking and has dense node indices.
        // Preserve ordered accumulation while avoiding a tree lookup per edge
        // on each iteration, including the common all-nodes pass.
        let mut active_nodes = vec![false; count];
        for node in &active {
            active_nodes[node.index()] = true;
        }
        let mut ranks = vec![1.0 / count as f64; count];
        if let Some(previous) = previous {
            for node in &all_nodes {
                if let Some(rank) = previous.get(&self.graph[*node]) {
                    ranks[node.index()] = *rank;
                }
            }
        }
        let mut outgoing = vec![0.0; count];
        for (source, _, weight) in &self.edge_rows {
            outgoing[source.index()] += weight;
        }

        let mut iterations = 0;
        for iteration in 1..=MAX_ITERATIONS {
            let dangling = all_nodes
                .iter()
                .filter(|node| outgoing[node.index()] == 0.0)
                .map(|node| ranks[node.index()])
                .sum::<f64>();
            let base = (1.0 - DAMPING) / count as f64 + DAMPING * dangling / count as f64;
            let mut next = ranks.clone();
            for node in &active {
                next[node.index()] = base;
            }
            for (source, destination, weight) in &self.edge_rows {
                if active_nodes[destination.index()] {
                    next[destination.index()] +=
                        DAMPING * ranks[source.index()] * weight / outgoing[source.index()];
                }
            }
            let delta = active
                .iter()
                .map(|node| (next[node.index()] - ranks[node.index()]).abs())
                .fold(0.0, f64::max);
            ranks = next;
            iterations = iteration;
            if delta <= CONVERGENCE {
                break;
            }
        }

        let symbol_ranks = all_nodes
            .iter()
            .map(|node| SymbolRank {
                symbol_id: self.graph[*node],
                rank: ranks[node.index()],
            })
            .collect::<Vec<_>>();
        let mut file_ranks = BTreeMap::<i64, f64>::new();
        for symbol in &symbol_ranks {
            if let Some(file_id) = self.files.get(&symbol.symbol_id) {
                *file_ranks.entry(*file_id).or_default() += symbol.rank;
            }
        }
        RankResult {
            symbols: symbol_ranks,
            files: file_ranks
                .into_iter()
                .map(|(file_id, rank)| FileRank { file_id, rank })
                .collect(),
            iterations,
        }
    }
}

fn qualified_name(symbols: &[PositionedSymbol], index: usize) -> String {
    let child = &symbols[index];
    let mut scope = symbols
        .iter()
        .enumerate()
        .filter(|(parent_index, parent)| {
            *parent_index != index
                && parent.range.start <= child.range.start
                && parent.range.end >= child.range.end
                && (parent.range.start != child.range.start || parent.range.end != child.range.end)
        })
        .collect::<Vec<_>>();
    scope.sort_by(|(_, left), (_, right)| {
        left.range
            .start
            .cmp(&right.range.start)
            .then_with(|| right.range.end.cmp(&left.range.end))
    });
    scope.push((index, child));
    scope
        .into_iter()
        .map(|(_, symbol)| symbol.symbol.name.as_str())
        .collect::<Vec<_>>()
        .join(" › ")
}

fn owner_for_range(symbols: &[PositionedSymbol], range: &Range<usize>) -> Option<usize> {
    symbols
        .iter()
        .enumerate()
        .filter(|(_, symbol)| {
            symbol.range.start <= range.start
                && symbol.range.end >= range.end
                && (symbol.range.start != range.start || symbol.range.end != range.end)
        })
        .min_by_key(|(_, symbol)| symbol.range.end.saturating_sub(symbol.range.start))
        .map(|(index, _)| index)
}

fn import_targets(language: AstLanguage, source: &str) -> Result<Vec<ImportTarget>, Error> {
    let mut parser = Parser::new();
    parser.set_language(&language.grammar())?;
    let tree = parser.parse(source, None).ok_or(Error::ParseCancelled)?;
    let mut imports = Vec::new();
    collect_imports(tree.root_node(), language, source, &mut imports)?;
    Ok(imports)
}

fn collect_imports(
    node: Node<'_>,
    language: AstLanguage,
    source: &str,
    imports: &mut Vec<ImportTarget>,
) -> Result<(), Error> {
    if is_import_kind(language, node.kind()) {
        let text = source
            .get(node.start_byte()..node.end_byte())
            .ok_or(Error::InvalidSourceRange)?;
        if let Some((name, path)) = import_target(text) {
            imports.push(ImportTarget {
                name,
                path,
                line: node.start_position().row + 1,
            });
        }
        return Ok(());
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        collect_imports(child, language, source, imports)?;
    }
    Ok(())
}

fn import_target(text: &str) -> Option<(String, String)> {
    let identifiers = identifiers(text)
        .into_iter()
        .filter(|identifier| {
            !matches!(
                identifier.as_str(),
                "as" | "crate"
                    | "extern"
                    | "from"
                    | "import"
                    | "pub"
                    | "self"
                    | "super"
                    | "type"
                    | "use"
            )
        })
        .collect::<Vec<_>>();
    let path =
        quoted_path(text).or_else(|| (!identifiers.is_empty()).then(|| identifiers.join("/")))?;
    let name = identifiers
        .last()
        .cloned()
        .or_else(|| path.rsplit('/').next().map(str::to_owned))?;
    Some((name, normalize_path(&path)))
}

fn quoted_path(text: &str) -> Option<String> {
    let quote_start = text.find(['"', '\''])?;
    let quote = text.as_bytes().get(quote_start).copied()? as char;
    let remainder = text.get(quote_start + quote.len_utf8()..)?;
    let quote_end = remainder.find(quote)?;
    remainder.get(..quote_end).map(str::to_owned)
}

fn identifiers(text: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut start = None;
    for (offset, character) in text.char_indices() {
        if character.is_alphanumeric() || character == '_' || character == '$' {
            start.get_or_insert(offset);
        } else if let Some(start) = start.take()
            && let Some(value) = text.get(start..offset)
        {
            values.push(value.to_owned());
        }
    }
    if let Some(start) = start
        && let Some(value) = text.get(start..)
    {
        values.push(value.to_owned());
    }
    values
}

fn normalize_path(path: &str) -> String {
    path.split(['/', '\\', ':', '.'])
        .filter(|part| !part.is_empty() && *part != "_")
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join("/")
}

fn path_suffix_matches(path: &str, import_path: &str) -> bool {
    let path = normalize_path(path);
    let import_path = normalize_path(import_path);
    let path = ["/java", "/php", "/tsx", "/go", "/js", "/py", "/rs", "/ts"]
        .into_iter()
        .find_map(|suffix| path.strip_suffix(suffix))
        .unwrap_or(&path);
    path == import_path || path.ends_with(&format!("/{}", import_path))
}

fn module_name(path: &str) -> String {
    path.rsplit('/')
        .next()
        .unwrap_or(path)
        .split('.')
        .next()
        .filter(|name| !name.is_empty())
        .unwrap_or("module")
        .to_owned()
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, io, time::Instant};

    use super::{
        EdgeKind, ExtractedEdge, RankEdge, RankSymbol, ResolutionCandidate, rank_full,
        rank_incremental, resolve_edges,
    };

    fn symbol(id: i64, file_id: i64, name: &str, pagerank: f64) -> RankSymbol {
        RankSymbol {
            id,
            file_id,
            name: name.to_owned(),
            pagerank,
        }
    }

    #[test]
    fn incremental_edit_updates_old_neighbours_and_preserves_distant_ranks() {
        let mut symbols = (1..=6)
            .map(|id| symbol(id, id, &format!("symbol{id}"), 0.0))
            .collect::<Vec<_>>();
        let edges = vec![RankEdge {
            src_symbol: 1,
            dst_name: "symbol2".to_owned(),
            dst_symbol: Some(2),
            kind: EdgeKind::Ref,
        }];
        let previous = rank_full(&symbols, &edges);
        for (symbol, rank) in symbols.iter_mut().zip(&previous.symbols) {
            symbol.pagerank = rank.rank;
        }
        let changed =
            super::rank_changed_files(&symbols, &[], &symbols, &edges, &BTreeSet::from([1]));
        assert_ne!(changed.symbols[1].rank, previous.symbols[1].rank);
        for id in 2..6 {
            assert_eq!(changed.symbols[id].rank, previous.symbols[id].rank);
        }
    }

    #[test]
    fn rust_tags_yield_definitions_references_and_import_edges()
    -> Result<(), Box<dyn std::error::Error>> {
        let source = br#"
use crate::auth::User;

pub struct RetryPolicy;

impl RetryPolicy {
    pub fn should_retry(&self) {
        should_retry();
        User::new();
    }
}
"#;
        let graph = super::extract_file_graph("rust", "src/retry.rs", source)?
            .ok_or_else(|| io::Error::other("rust must be supported"))?;
        assert!(
            graph
                .symbols
                .iter()
                .any(|symbol| symbol.name == "RetryPolicy")
        );
        assert!(
            graph
                .symbols
                .iter()
                .any(|symbol| symbol.name == "should_retry"
                    && symbol.qualified.ends_with("should_retry"))
        );
        assert!(
            graph
                .edges
                .iter()
                .any(|edge| edge.kind == EdgeKind::Ref && edge.dst_name == "should_retry")
        );
        assert!(
            graph
                .edges
                .iter()
                .any(|edge| edge.kind == EdgeKind::Import && edge.dst_name == "User")
        );
        Ok(())
    }

    #[test]
    fn every_static_grammar_runs_its_tags_query() -> Result<(), Box<dyn std::error::Error>> {
        let examples: [(&str, &str, &[u8]); 8] = [
            ("rust", "src/example.rs", b"fn hello() { hello(); }"),
            ("python", "src/example.py", b"def hello():\n    hello()\n"),
            (
                "javascript",
                "src/example.js",
                b"function hello() { hello(); }",
            ),
            (
                "typescript",
                "src/example.ts",
                b"interface Example { hello(): void; }",
            ),
            (
                "tsx",
                "src/example.tsx",
                b"interface Example { hello(): void; }",
            ),
            ("go", "src/example.go", b"package example\nfunc Hello() {}"),
            (
                "java",
                "src/Example.java",
                b"class Example { void hello() {} }",
            ),
            (
                "php",
                "src/example.php",
                b"<?php function hello() { hello(); }",
            ),
        ];
        for (language, path, source) in examples {
            assert!(
                super::extract_file_graph(language, path, source)?.is_some(),
                "{language}"
            );
            assert_eq!(
                super::extract_file_symbols(language, path, source)?,
                super::extract_file_graph(language, path, source)?.map(|g| g.symbols),
                "{language}"
            );
        }
        Ok(())
    }

    #[test]
    fn resolver_uses_unique_same_file_then_import_suffix() {
        let candidates = [
            ResolutionCandidate {
                id: 1,
                file_id: 1,
                path: "src/retry.rs".to_owned(),
                name: "helper".to_owned(),
            },
            ResolutionCandidate {
                id: 2,
                file_id: 2,
                path: "src/auth/user.rs".to_owned(),
                name: "User".to_owned(),
            },
            ResolutionCandidate {
                id: 3,
                file_id: 3,
                path: "src/legacy/user.rs".to_owned(),
                name: "User".to_owned(),
            },
        ];
        let edges = [
            ExtractedEdge {
                source_index: 0,
                dst_name: "helper".to_owned(),
                kind: EdgeKind::Ref,
                line: 1,
                import_path: None,
            },
            ExtractedEdge {
                source_index: 0,
                dst_name: "User".to_owned(),
                kind: EdgeKind::Ref,
                line: 2,
                import_path: Some("auth/user".to_owned()),
            },
        ];
        let resolved = resolve_edges(1, &edges, &candidates);
        assert_eq!(resolved[0].dst_symbol, Some(1));
        assert_eq!(resolved[1].dst_symbol, Some(2));
    }

    #[test]
    fn rank_prefers_a_symbol_referenced_by_two_callers() {
        let symbols = [
            symbol(1, 1, "caller_one", 0.0),
            symbol(2, 1, "target", 0.0),
            symbol(3, 2, "caller_two", 0.0),
        ];
        let edges = [
            RankEdge {
                src_symbol: 1,
                dst_name: "target".to_owned(),
                dst_symbol: Some(2),
                kind: EdgeKind::Ref,
            },
            RankEdge {
                src_symbol: 3,
                dst_name: "target".to_owned(),
                dst_symbol: Some(2),
                kind: EdgeKind::Ref,
            },
        ];
        let result = rank_full(&symbols, &edges);
        let ranks = result
            .symbols
            .iter()
            .map(|rank| (rank.symbol_id, rank.rank))
            .collect::<std::collections::BTreeMap<_, _>>();
        assert!(ranks[&2] > ranks[&1]);
        assert!(ranks[&2] > ranks[&3]);
        assert!((ranks[&1] - 0.212_765_96).abs() < 1e-5);
        assert!((ranks[&2] - 0.574_468_09).abs() < 1e-5);
        assert!((result.files.iter().map(|file| file.rank).sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn incremental_two_hop_result_matches_full_graph_fixture() {
        let symbols = (1..=3)
            .map(|id| symbol(id, id, &id.to_string(), 1.0 / 3.0))
            .collect::<Vec<_>>();
        let edges = [
            RankEdge {
                src_symbol: 1,
                dst_name: "2".to_owned(),
                dst_symbol: Some(2),
                kind: EdgeKind::Ref,
            },
            RankEdge {
                src_symbol: 2,
                dst_name: "3".to_owned(),
                dst_symbol: Some(3),
                kind: EdgeKind::Ref,
            },
            RankEdge {
                src_symbol: 3,
                dst_name: "1".to_owned(),
                dst_symbol: Some(1),
                kind: EdgeKind::Ref,
            },
        ];
        let full = rank_full(&symbols, &edges);
        let incremental = rank_incremental(&symbols, &edges, &BTreeSet::from([1]));
        for (full, incremental) in full.symbols.iter().zip(&incremental.symbols) {
            assert!((full.rank - incremental.rank).abs() <= 1e-6);
        }
    }

    #[test]
    fn ranking_is_deterministic() {
        let symbols = (1..=4)
            .map(|id| symbol(id, id, &id.to_string(), 0.0))
            .collect::<Vec<_>>();
        let edges = [
            RankEdge {
                src_symbol: 1,
                dst_name: "2".to_owned(),
                dst_symbol: Some(2),
                kind: EdgeKind::Ref,
            },
            RankEdge {
                src_symbol: 2,
                dst_name: "3".to_owned(),
                dst_symbol: Some(3),
                kind: EdgeKind::Import,
            },
        ];
        assert_eq!(rank_full(&symbols, &edges), rank_full(&symbols, &edges));
    }

    #[test]
    fn hundred_thousand_edges_rank_within_two_seconds() {
        let symbols = (0..1_000)
            .map(|id| symbol(id, id, &id.to_string(), 0.0))
            .collect::<Vec<_>>();
        let mut edges = Vec::with_capacity(100_000);
        for source in 0..1_000 {
            for offset in 1..=100 {
                edges.push(RankEdge {
                    src_symbol: source,
                    dst_name: "resolved".to_owned(),
                    dst_symbol: Some((source + offset) % 1_000),
                    kind: EdgeKind::Ref,
                });
            }
        }
        let started = Instant::now();
        let result = rank_full(&symbols, &edges);
        assert_eq!(result.symbols.len(), 1_000);
        assert!(started.elapsed().as_secs_f64() < 2.0);
    }

    #[test]
    fn shared_queries_keep_concurrent_sources_and_languages_isolated() {
        std::thread::scope(|scope| {
            for worker in 0..8 {
                scope.spawn(move || {
                    for round in 0..8 {
                        let name = format!("unique_{worker}_{round}");
                        for (language, source) in [
                            ("rust", format!("fn {name}() {{}}")),
                            ("python", format!("def {name}():\n    pass\n")),
                        ] {
                            let graph =
                                super::extract_file_graph(language, "fixture", source.as_bytes())
                                    .unwrap()
                                    .unwrap();
                            assert!(!graph.parse_error);
                            assert_eq!(graph.symbols.len(), 1);
                            assert_eq!(graph.symbols[0].name, name);
                            assert!(graph.edges.is_empty());
                        }
                    }
                });
            }
        });
    }

    #[test]
    fn module_and_definition_ranges_are_inclusive_for_newlines_and_unicode_eof() {
        for newline in ["\n", "\r\n"] {
            for terminated in [false, true] {
                for (language, path, lines, definition_line) in [
                    (
                        "php",
                        "src/fixture.php",
                        vec![
                            "<?php",
                            "use Example\\Dependency;",
                            "function eofgraphmarker() { return 1; }",
                            "// λ",
                        ],
                        3,
                    ),
                    (
                        "python",
                        "src/fixture.py",
                        vec![
                            "import dependency",
                            "",
                            "def eofgraphmarker():",
                            "    return λ",
                        ],
                        4,
                    ),
                ] {
                    let mut source = lines.join(newline);
                    if terminated {
                        source.push_str(newline);
                    }
                    let graph = super::extract_file_graph(language, path, source.as_bytes())
                        .unwrap()
                        .unwrap();
                    assert!(!graph.parse_error, "{language}: {source}");
                    let module = graph
                        .symbols
                        .iter()
                        .find(|symbol| symbol.kind == "module" && symbol.qualified == path)
                        .unwrap();
                    assert_eq!(
                        (module.line_start, module.line_end),
                        (1, 4),
                        "{language} terminated={terminated}"
                    );
                    let definition = graph
                        .symbols
                        .iter()
                        .find(|symbol| symbol.name == "eofgraphmarker")
                        .unwrap();
                    assert_eq!(
                        definition.line_end, definition_line,
                        "{language} terminated={terminated}"
                    );
                    assert!(graph.symbols.iter().all(|symbol| symbol.line_start >= 1
                        && symbol.line_start <= symbol.line_end
                        && symbol.line_end <= source.lines().count()));
                }
            }
        }
    }
}
