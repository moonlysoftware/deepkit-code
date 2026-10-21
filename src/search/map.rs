//! Ranked repository maps with bounded token budgeting; see `v1.md` §9–10.1.
use super::{
    Error,
    navigation::{compare_map_symbols, public_symbol_id},
};
use crate::{
    store::{MapFile, Store, StoredUnit, ToolSymbol},
    treat::fts::build_match,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
type Check<'a> = &'a dyn Fn() -> std::result::Result<(), crate::store::Error>;
type Result<T> = std::result::Result<T, Error>;

/// Ranked tree and the exact admitted nodes, shared by CLI, MCP and benchmarks.
#[derive(Debug, Serialize)]
pub(crate) struct MapResult {
    pub(crate) tree: String,
    pub(crate) nodes: Vec<MapNode>,
    #[serde(skip)]
    pub(crate) selection: MapSelection,
    #[serde(skip)]
    pub(crate) module_labels: BTreeMap<String, String>,
}
/// One admitted map file and its highest-ranked signatures.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct MapNode {
    pub(crate) path: String,
    pub(crate) symbols: Vec<MapSymbol>,
}
/// Compact symbol metadata corresponding to one rendered signature.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct MapSymbol {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) signature: String,
    pub(crate) lines: [i64; 2],
    pub(crate) rank: f64,
}
#[derive(Clone, Debug, PartialEq)]
struct MapFileView {
    file: MapFile,
    symbols: Vec<ToolSymbol>,
}

/// Private pre-verification selection, including ranks and symbol ownership that
/// compact output can hide. Retained for final lookup with the original inputs.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MapSelection {
    files: Vec<MapFileView>,
    focus: FocusSelection,
}

/// Request-local database selection, valid only for its exact inputs/revision.
pub(crate) struct MapCache {
    revision: crate::store::modules::Revision,
    budget: usize,
    focus: Vec<String>,
    selection: MapSelection,
}

// Keep actual contributors before signature truncation and token admission.
// Multiple matches in the same file must not collapse their authority to one ID.
#[derive(Clone, Debug, Default, PartialEq)]
struct FocusSelection {
    file_ids: BTreeSet<i64>,
    symbols: BTreeMap<i64, ToolSymbol>,
    lexical: BTreeMap<i64, Option<StoredUnit>>,
    symbol_units: BTreeMap<i64, Option<StoredUnit>>,
}

#[derive(Default)]
struct MapTree<'view> {
    directories: BTreeMap<String, MapTree<'view>>,
    files: Vec<&'view MapFileView>,
    rank: f64,
}

enum MapTreeEntry<'view> {
    Directory(&'view str, &'view MapTree<'view>),
    File(&'view MapFileView),
}

/// Builds a deterministic map within the requested cl100k token budget.
/// Candidate admission accounts for new directory/file/signature lines once;
/// final exact validation handles token merges at line boundaries.
#[cfg(test)]
pub(crate) fn map(store: &Store, budget: usize, focus: &[String]) -> Result<MapResult> {
    map_checked(store, budget, focus, &|| Ok(()), |_, _| Ok(true))
}

/// Verifies selected files before rendering so tree and nodes omit the same sources.
/// Source verification runs only for budget-admitted files; removal is budgeted again.
/// Work checks cover reader admission, focus, selection and rendering. Individual
/// SQL statements, sorting and tokenizer calls remain cooperatively bounded.
pub(crate) fn map_checked(
    store: &Store,
    budget: usize,
    focus: &[String],
    work: Check<'_>,
    check: impl FnMut(&str, &[ToolSymbol]) -> Result<bool>,
) -> Result<MapResult> {
    if budget == 0 {
        return Err(Error::InvalidMapBudget);
    }
    work()?;
    let files = store.map_files_checked(work)?;
    let symbols = store
        .tool_symbols_checked(work)?
        .into_iter()
        .filter(|s| s.kind != "doc")
        .collect::<Vec<_>>();
    let focused = focus_selection(store, &files, &symbols, focus, work)?;
    let mut symbols_by_file = BTreeMap::<i64, Vec<ToolSymbol>>::new();
    for symbol in symbols {
        work()?;
        symbols_by_file
            .entry(symbol.file_id)
            .or_default()
            .push(symbol);
    }
    for file_symbols in symbols_by_file.values_mut() {
        work()?;
        file_symbols.sort_by(compare_map_symbols);
        file_symbols.truncate(3);
    }
    let mut candidates = Vec::new();
    for file in files {
        work()?;
        if focus.is_empty() || focused.file_ids.contains(&file.id) {
            candidates.push(MapFileView {
                symbols: symbols_by_file.remove(&file.id).unwrap_or_default(),
                file,
            });
        }
    }
    candidates.sort_by(|left, right| {
        right
            .file
            .rank
            .total_cmp(&left.file.rank)
            .then_with(|| left.file.path.cmp(&right.file.path))
    });

    work()?;
    let selected = select(candidates, budget, work)?;
    let selection = MapSelection {
        files: selected.clone(),
        focus: focused,
    };
    map_selected_checked(selection, budget, work, check)
}

/// Reuses only the database-derived selection while independently rechecking
/// every selected live source and enforcing the exact public token budget.
fn map_selected_checked(
    selection: MapSelection,
    budget: usize,
    work: Check<'_>,
    mut check: impl FnMut(&str, &[ToolSymbol]) -> Result<bool>,
) -> Result<MapResult> {
    let mut verified = Vec::new();
    for view in selection.files.iter().cloned() {
        work()?;
        if check(&view.file.path, &view.symbols)? {
            verified.push(view);
        }
    }
    let (selected, tree) = enforce_budget(verified, budget, work)?;
    work()?;
    let result = MapResult {
        module_labels: BTreeMap::new(),
        selection,
        tree,
        nodes: selected
            .into_iter()
            .map(|view| {
                work()?;
                Ok(MapNode {
                    path: view.file.path,
                    symbols: view
                        .symbols
                        .into_iter()
                        .map(|symbol| MapSymbol {
                            id: public_symbol_id(&symbol),
                            signature: symbol.signature.unwrap_or_else(|| symbol.name.clone()),
                            name: symbol.name,
                            lines: [symbol.line_start, symbol.line_end],
                            rank: symbol.pagerank,
                        })
                        .collect(),
                })
            })
            .collect::<Result<Vec<_>>>()?,
    };
    work()?;
    Ok(result)
}

/// Reuses a request-local map selection only while the writer-connection
/// revision proves that neither local nor external database writes intervened.
/// The callback and render still run on every pass, including the final fence.
pub(crate) fn map_checked_reuse(
    store: &Store,
    cache: &mut Option<MapCache>,
    budget: usize,
    focus: &[String],
    work: Check<'_>,
    check: impl FnMut(&str, &[ToolSymbol]) -> Result<bool>,
) -> Result<MapResult> {
    if budget == 0 {
        return Err(Error::InvalidMapBudget);
    }
    let before = store.revision_checked(work)?;
    let result = match cache.as_ref() {
        Some(cached)
            if cached.revision == before && cached.budget == budget && cached.focus == focus =>
        {
            map_selected_checked(cached.selection.clone(), budget, work, check)?
        }
        _ => map_checked(store, budget, focus, work, check)?,
    };
    let after = store.revision_checked(work)?;
    *cache = (before == after).then(|| MapCache {
        revision: after,
        budget,
        focus: focus.to_vec(),
        selection: result.selection.clone(),
    });
    Ok(result)
}

/// Rebuilds only retained response files from their original views. Removing a
/// source can alter token boundaries, so enforce the exact budget again.
pub(crate) fn retain_checked(
    mapped: &mut MapResult,
    paths: &BTreeSet<String>,
    budget: usize,
    work: Check<'_>,
) -> Result<()> {
    let mut views = Vec::new();
    for view in &mapped.selection.files {
        work()?;
        if paths.contains(&view.file.path) {
            views.push(view.clone());
        }
    }
    let (views, tree) = enforce_budget_labeled(views, budget, work, &mapped.module_labels)?;
    let retained: BTreeSet<_> = views.iter().map(|view| view.file.path.as_str()).collect();
    mapped
        .nodes
        .retain(|node| retained.contains(node.path.as_str()));
    mapped.tree = tree;
    work()?;
    Ok(())
}

/// Adds only verified labels, then re-applies the original exact map budget.
/// Retains a whole-file prefix; later response-budget trimming uses these labels.
pub(crate) fn label_modules(
    mapped: &mut MapResult,
    labels: BTreeMap<String, String>,
    budget: usize,
    work: Check<'_>,
) -> Result<()> {
    mapped.module_labels = labels;
    let paths = mapped.nodes.iter().map(|n| n.path.clone()).collect();
    retain_checked(mapped, &paths, budget, work)
}

/// Directory presentation plus exact owner/summary rows for final revalidation.
#[derive(Debug, PartialEq)]
pub(crate) struct Module {
    pub(crate) path: String,
    pub(crate) unit: StoredUnit,
    pub(crate) summary: Option<super::summaries::Observation>,
}
impl Module {
    /// Uses the verified summary's first line, otherwise the deterministic header.
    pub(crate) fn label(&self) -> String {
        self.summary
            .as_ref()
            .and_then(|s| s.text.lines().next())
            .filter(|line| !line.trim().is_empty())
            .unwrap_or(&self.unit.header)
            .to_owned()
    }
}

/// Finds modules only on ancestors of admitted files; source verification belongs
/// to the request owner and must be repeated after its final observation boundary.
pub(crate) fn modules(store: &Store, mapped: &MapResult, work: Check<'_>) -> Result<Vec<Module>> {
    let mut paths = BTreeSet::new();
    for node in &mapped.nodes {
        work()?;
        paths.insert(String::new());
        for (index, _) in node.path.match_indices('/') {
            paths.insert(node.path[..index].to_owned());
        }
    }
    let encoded =
        serde_json::to_string(&paths).map_err(|_| crate::store::Error::L1InputsChanged)?;
    store.checked_read_snapshot(work,|tx| {
        let rows=tx.prepare("SELECT d.path,d.module_unit FROM dirs d WHERE d.path IN (SELECT value FROM json_each(?1)) AND d.module_unit IS NOT NULL ORDER BY d.path")?.query_map([encoded],|r|Ok((r.get::<_,String>(0)?,r.get::<_,i64>(1)?)))?.collect::<std::result::Result<Vec<_>,_>>()?;
        let mut result=Vec::new();
        for (path,id) in rows {
            work()?;
            if let Some(mut unit)=crate::store::unit_by_id_in(tx,id)? {
                unit.ladder_level=0;
                result.push(Module{path,unit,summary:None});
            }
        }
        Ok::<_,crate::store::Error>(result)
    })?.map_err(Error::from)
}

fn select(
    candidates: Vec<MapFileView>,
    budget: usize,
    work: Check<'_>,
) -> Result<Vec<MapFileView>> {
    let mut directories = BTreeMap::<String, (usize, bool)>::new();
    let mut selected = Vec::new();
    let mut used = 0usize;
    for mut candidate in candidates {
        work()?;
        let mut cost = 0usize;
        let mut prefixes = Vec::new();
        let depth = candidate
            .file
            .path
            .bytes()
            .filter(|byte| *byte == b'/')
            .count();
        for (level, (index, _)) in candidate.file.path.match_indices('/').enumerate() {
            work()?;
            let prefix = &candidate.file.path[..index];
            let (tokens, admitted) = directories.entry(prefix.to_owned()).or_insert_with(|| {
                let name = prefix.rsplit('/').next().unwrap_or(prefix);
                (
                    line_cost(&format!(
                        "{space:indent$}{name}/",
                        space = "",
                        indent = level * 2
                    )),
                    false,
                )
            });
            if !*admitted {
                cost = cost.saturating_add(*tokens);
            }
            prefixes.push(prefix);
        }
        let name = candidate
            .file
            .path
            .rsplit('/')
            .next()
            .unwrap_or(&candidate.file.path);
        cost = cost.saturating_add(line_cost(&format!(
            "{space:indent$}{name}",
            space = "",
            indent = depth * 2
        )));
        if used.saturating_add(cost) > budget {
            continue;
        }
        let mut count = 0;
        for symbol in &candidate.symbols {
            work()?;
            let signature = symbol.signature.as_deref().unwrap_or(&symbol.name);
            let symbol_cost = line_cost(&format!(
                "{space:indent$}  {}: {signature}",
                symbol.name,
                space = "",
                indent = depth * 2
            ));
            if used.saturating_add(cost).saturating_add(symbol_cost) > budget {
                break;
            }
            cost = cost.saturating_add(symbol_cost);
            count += 1;
        }
        for prefix in prefixes {
            if let Some((_, admitted)) = directories.get_mut(prefix) {
                *admitted = true;
            }
        }
        candidate.symbols.truncate(count);
        used = used.saturating_add(cost);
        selected.push(candidate);
    }
    work()?;
    Ok(selected)
}

fn line_cost(line: &str) -> usize {
    token_count(&format!("{line}\n"))
}

// BPE is not assumed additive or monotonic across lines. Retain a measured
// feasible prefix if boundary effects exceed the budget, in O(log n) renders.
fn enforce_budget(
    selected: Vec<MapFileView>,
    budget: usize,
    work: Check<'_>,
) -> Result<(Vec<MapFileView>, String)> {
    enforce_budget_labeled(selected, budget, work, &BTreeMap::new())
}

fn enforce_budget_labeled(
    mut selected: Vec<MapFileView>,
    budget: usize,
    work: Check<'_>,
    labels: &BTreeMap<String, String>,
) -> Result<(Vec<MapFileView>, String)> {
    let mut tree = render_map_tree(&selected, work, labels)?;
    if token_count(&tree) <= budget {
        work()?;
        return Ok((selected, tree));
    }
    let mut low = 0;
    let mut high = selected.len();
    tree.clear();
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        let trial = render_map_tree(&selected[..middle], work, labels)?;
        if token_count(&trial) <= budget {
            low = middle;
            tree = trial;
        } else {
            high = middle;
        }
    }
    selected.truncate(low);
    work()?;
    Ok((selected, tree))
}

#[cfg(test)]
thread_local! {
    static TOKENIZED_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn token_count(text: &str) -> usize {
    #[cfg(test)]
    TOKENIZED_BYTES.with(|bytes| bytes.set(bytes.get().saturating_add(text.len())));
    tiktoken_rs::cl100k_base_singleton()
        .encode_with_special_tokens(text)
        .len()
}
fn focus_selection(
    store: &Store,
    files: &[MapFile],
    symbols: &[ToolSymbol],
    focus: &[String],
    work: Check<'_>,
) -> Result<FocusSelection> {
    let mut focused = FocusSelection::default();
    work()?;
    if focus.is_empty() {
        return Ok(focused);
    }
    let mut paths = BTreeMap::new();
    for file in files {
        work()?;
        paths.insert(file.path.as_str(), file.id);
    }
    for term in focus {
        work()?;
        let normalized = term.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            continue;
        }
        for file in files {
            work()?;
            if file.path.to_ascii_lowercase().contains(&normalized) {
                focused.file_ids.insert(file.id);
            }
        }
        for symbol in symbols {
            work()?;
            let qualified = symbol.qualified.as_deref().unwrap_or(&symbol.name);
            if symbol.name.to_ascii_lowercase().contains(&normalized)
                || qualified.to_ascii_lowercase().contains(&normalized)
            {
                focused.file_ids.insert(symbol.file_id);
                focused.symbols.insert(symbol.id, symbol.clone());
                if let Some(id) = symbol.unit_id {
                    focused
                        .symbol_units
                        .insert(id, focus_unit(store, id, work)?);
                }
            }
        }
        let query = build_match(term);
        if !query.is_empty() {
            for hit in store.query_fts_checked(&query, 200, work)? {
                work()?;
                let unit = focus_unit(store, hit.unit_id, work)?;
                if let Some(path) = unit.as_ref().and_then(|unit| unit.path.as_deref())
                    && let Some(id) = paths.get(path)
                {
                    focused.file_ids.insert(*id);
                }
                focused.lexical.insert(hit.unit_id, unit);
            }
        }
    }
    work()?;
    Ok(focused)
}

// Background embedding publication does not change L1 focus authority.
fn focus_unit(store: &Store, id: i64, work: Check<'_>) -> Result<Option<StoredUnit>> {
    let mut unit = store.unit_by_id_checked(id, work)?;
    if let Some(unit) = &mut unit {
        unit.ladder_level = 0;
    }
    Ok(unit)
}

fn render_map_tree(
    views: &[MapFileView],
    work: Check<'_>,
    labels: &BTreeMap<String, String>,
) -> Result<String> {
    work()?;
    let mut root = MapTree::default();
    for view in views {
        insert_map_file(&mut root, view, work)?;
    }
    let mut lines = Vec::new();
    if !views.is_empty()
        && let Some(label) = labels.get("")
    {
        lines.push(format!("./ — {label}"));
    }
    render_map_branch(&root, 0, &mut lines, work, "", labels)?;
    work()?;
    Ok(lines.join("\n"))
}

fn insert_map_file<'view>(
    tree: &mut MapTree<'view>,
    view: &'view MapFileView,
    work: Check<'_>,
) -> Result<()> {
    let mut segments = view.file.path.split('/').peekable();
    let mut current = tree;
    current.rank = current.rank.max(view.file.rank);
    while let Some(segment) = segments.next() {
        work()?;
        if segments.peek().is_some() {
            current = current.directories.entry(segment.to_owned()).or_default();
            current.rank = current.rank.max(view.file.rank);
        } else {
            current.files.push(view);
        }
    }
    Ok(())
}

fn render_map_branch(
    tree: &MapTree<'_>,
    indent: usize,
    lines: &mut Vec<String>,
    work: Check<'_>,
    prefix: &str,
    labels: &BTreeMap<String, String>,
) -> Result<()> {
    work()?;
    let mut entries = tree
        .directories
        .iter()
        .map(|(name, subtree)| MapTreeEntry::Directory(name, subtree))
        .chain(tree.files.iter().copied().map(MapTreeEntry::File))
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| {
        map_entry_rank(right)
            .total_cmp(&map_entry_rank(left))
            .then_with(|| map_entry_label(left).cmp(map_entry_label(right)))
    });
    for entry in entries {
        work()?;
        match entry {
            MapTreeEntry::Directory(name, subtree) => {
                let path = if prefix.is_empty() {
                    name.to_owned()
                } else {
                    format!("{prefix}/{name}")
                };
                let suffix = labels
                    .get(&path)
                    .map_or_else(String::new, |label| format!(" — {label}"));
                lines.push(format!("{space:indent$}{name}/{suffix}", space = ""));
                render_map_branch(subtree, indent + 2, lines, work, &path, labels)?;
            }
            MapTreeEntry::File(view) => {
                let name = view
                    .file
                    .path
                    .rsplit('/')
                    .next()
                    .map_or(view.file.path.as_str(), |name| name);
                lines.push(format!("{space:indent$}{name}", space = ""));
                for symbol in &view.symbols {
                    work()?;
                    let signature = symbol.signature.as_deref().unwrap_or(&symbol.name);
                    lines.push(format!(
                        "{space:indent$}  {}: {signature}",
                        symbol.name,
                        space = ""
                    ));
                }
            }
        }
    }
    work()?;
    Ok(())
}

fn map_entry_rank(entry: &MapTreeEntry<'_>) -> f64 {
    match entry {
        MapTreeEntry::Directory(_, subtree) => subtree.rank,
        MapTreeEntry::File(view) => view.file.rank,
    }
}

fn map_entry_label<'a>(entry: &'a MapTreeEntry<'_>) -> &'a str {
    match entry {
        MapTreeEntry::Directory(name, _) => name,
        MapTreeEntry::File(view) => &view.file.path,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        search::tests::add,
        store::{self, StoreLimits},
    };
    use std::error::Error;

    #[test]
    fn request_cache_rechecks_sources_and_reselects_after_local_or_external_writes()
    -> std::result::Result<(), Box<dyn Error>> {
        let file = tempfile::NamedTempFile::new()?;
        let store = store::open(file.path(), StoreLimits::default())?;
        add(&store, "src/a.rs", "First", "first", None)?;
        add(&store, "src/b.rs", "Second", "second", None)?;
        let mut cache = None;
        let mut checks = 0;
        let original = map_checked_reuse(&store, &mut cache, 2000, &[], &|| Ok(()), |_, _| {
            checks += 1;
            Ok(true)
        })?;
        let count = checks;
        let hidden = map_checked_reuse(&store, &mut cache, 2000, &[], &|| Ok(()), |path, _| {
            checks += 1;
            Ok(path != "src/a.rs")
        })?;
        assert_eq!(checks, count * 2);
        assert_eq!(hidden.selection, original.selection);
        assert!(hidden.nodes.iter().all(|node| node.path != "src/a.rs"));

        store
            .writer()?
            .execute("UPDATE files SET rank=rank+1 WHERE path='src/b.rs'", [])?;
        let local = map_checked_reuse(&store, &mut cache, 2000, &[], &|| Ok(()), |_, _| Ok(true))?;
        assert_eq!(local.selection, map(&store, 2000, &[])?.selection);
        assert_ne!(local.selection, original.selection);

        let external = rusqlite::Connection::open(file.path())?;
        external.execute("UPDATE files SET rank=rank+1 WHERE path='src/a.rs'", [])?;
        let updated =
            map_checked_reuse(&store, &mut cache, 2000, &[], &|| Ok(()), |_, _| Ok(true))?;
        assert_eq!(updated.selection, map(&store, 2000, &[])?.selection);
        assert_ne!(updated.selection, local.selection);
        Ok(())
    }

    #[test]
    fn shared_map_is_deterministic_focused_and_within_exact_budget()
    -> std::result::Result<(), Box<dyn Error>> {
        let first = tempfile::NamedTempFile::new()?;
        let second = tempfile::NamedTempFile::new()?;
        let stores = [
            store::open(first.path(), StoreLimits::default())?,
            store::open(second.path(), StoreLimits::default())?,
        ];
        let paths = [
            "src/a.rs",
            "src/nested/é.rs",
            "other/z.rs",
            "file with spaces.rs",
        ];
        for (index, store) in stores.iter().enumerate() {
            let order: Vec<_> = if index == 0 {
                paths.iter().collect()
            } else {
                paths.iter().rev().collect()
            };
            for path in order {
                add(store, path, "TargetName", "fn target() {}", None)?;
            }
        }
        for budget in [1, 2, 5, 10, 20, 40, 80, 2000] {
            let first = map(&stores[0], budget, &[])?;
            let second = map(&stores[1], budget, &[])?;
            assert_eq!(first.tree, second.tree);
            assert_eq!(
                serde_json::to_value(&first)?,
                serde_json::to_value(&second)?
            );
            assert!(
                tiktoken_rs::cl100k_base_singleton()
                    .encode_with_special_tokens(&first.tree)
                    .len()
                    <= budget
            );
            for node in &first.nodes {
                assert!(
                    first
                        .tree
                        .contains(node.path.rsplit('/').next().ok_or("empty path")?)
                );
                for symbol in &node.symbols {
                    assert_eq!(
                        super::super::navigation::open(&stores[0], &symbol.id, 100)?
                            .path
                            .as_deref(),
                        Some(node.path.as_str())
                    );
                }
            }
        }
        let filtered = map_checked(&stores[0], 80, &[], &|| Ok(()), |path, _| {
            Ok(path != "src/nested/é.rs")
        })?;
        assert!(!filtered.tree.contains("é.rs"));
        assert!(!filtered.tree.contains("nested/"));
        assert!(
            filtered
                .nodes
                .iter()
                .all(|node| node.path != "src/nested/é.rs")
        );
        assert!(token_count(&filtered.tree) <= 80);
        let focused = map(&stores[0], 2000, &["src/nested".to_owned()])?;
        assert_eq!(focused.nodes.len(), 1);
        assert_eq!(focused.nodes[0].path, "src/nested/é.rs");
        assert!(matches!(
            map(&stores[0], 0, &[]),
            Err(super::Error::InvalidMapBudget)
        ));
        Ok(())
    }

    #[test]
    fn map_reader_cancels_at_every_boundary_without_writes_and_retries() {
        use std::cell::Cell;
        let store = store::open(std::path::Path::new(":memory:"), StoreLimits::default()).unwrap();
        add(
            &store,
            "src/nested/a.rs",
            "TargetName",
            "lexicalneedle",
            None,
        )
        .unwrap();
        add(&store, "other/b.rs", "OtherName", "unrelated", None).unwrap();
        let changes = store.writer().unwrap().total_changes();
        for focus in [
            vec![],
            vec!["lexicalneedle".to_owned()],
            vec!["TargetName".to_owned()],
            vec!["missing".to_owned()],
        ] {
            for budget in [1, 30, 2000] {
                let calls = Cell::new(0);
                let expected = map_checked(
                    &store,
                    budget,
                    &focus,
                    &|| {
                        calls.set(calls.get() + 1);
                        Ok(())
                    },
                    |_, _| Ok(true),
                )
                .unwrap();
                if focus.first().is_some_and(|f| f == "lexicalneedle") && budget == 2000 {
                    assert_eq!(expected.nodes.len(), 1);
                    assert_eq!(expected.nodes[0].path, "src/nested/a.rs");
                }
                for stop in 1..=calls.get() {
                    let count = Cell::new(0);
                    let result = map_checked(
                        &store,
                        budget,
                        &focus,
                        &|| {
                            count.set(count.get() + 1);
                            if count.get() == stop {
                                Err(crate::store::Error::L1InputsChanged)
                            } else {
                                Ok(())
                            }
                        },
                        |_, _| Ok(true),
                    );
                    assert!(
                        matches!(
                            result,
                            Err(super::Error::Store(crate::store::Error::L1InputsChanged))
                        ),
                        "boundary {stop}"
                    );
                }
                assert_eq!(
                    serde_json::to_value(map(&store, budget, &focus).unwrap()).unwrap(),
                    serde_json::to_value(expected).unwrap()
                );
            }
        }
        assert_eq!(store.writer().unwrap().total_changes(), changes);
    }

    #[test]
    fn map_reader_admission_expires_and_recovers_without_writes() {
        use std::time::{Duration, Instant};
        let store = store::open(std::path::Path::new(":memory:"), StoreLimits::default()).unwrap();
        add(&store, "src/a.rs", "TargetName", "lexicalneedle", None).unwrap();
        let expected = serde_json::to_value(map(&store, 2000, &[]).unwrap()).unwrap();
        let held = store.writer().unwrap();
        let changes = held.total_changes();
        let started = Instant::now();
        let result = map_checked(
            &store,
            2000,
            &[],
            &|| {
                if started.elapsed() >= Duration::from_millis(15) {
                    Err(crate::store::Error::L1InputsChanged)
                } else {
                    Ok(())
                }
            },
            |_, _| Ok(true),
        );
        assert!(matches!(
            result,
            Err(super::Error::Store(crate::store::Error::L1InputsChanged))
        ));
        assert!(started.elapsed() < Duration::from_secs(1));
        assert_eq!(held.total_changes(), changes);
        drop(held);
        assert_eq!(
            serde_json::to_value(map(&store, 2000, &[]).unwrap()).unwrap(),
            expected
        );
    }

    #[test]
    fn map_budgeting_does_not_retokenize_the_selected_tree_per_candidate() {
        let candidates = (0..3000)
            .map(|i| MapFileView {
                file: MapFile {
                    id: i,
                    path: format!("src/shared/nested/file{i:04}.rs"),
                    rank: 1.0,
                },
                symbols: Vec::new(),
            })
            .collect();
        TOKENIZED_BYTES.with(|bytes| bytes.set(0));
        let selected = select(candidates, 2000, &|| Ok(())).unwrap();
        let (selected, tree) = enforce_budget(selected, 2000, &|| Ok(())).unwrap();
        let bytes = TOKENIZED_BYTES.with(std::cell::Cell::get);
        assert!(!selected.is_empty());
        assert!(token_count(&tree) <= 2000);
        // 3,000 short candidate lines plus one bounded final tree. The old
        // full-tree trial algorithm processes millions of bytes on this fixture.
        assert!(bytes < 150_000, "tokenized {bytes} bytes");
    }
}
