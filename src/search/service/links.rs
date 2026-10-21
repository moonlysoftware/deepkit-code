//! Verified documentation/test navigation additions; v1.md §6.5/§10.1.
use super::*;
use rusqlite::OptionalExtension;
use std::collections::{BTreeMap, BTreeSet};

/// Reconciles labels before selection, sharing full-index admission and fences.
pub(super) fn refresh(context: &Context, work: sync::l1::Work<'_>) -> Result<()> {
    if !context.config.search.lazy_resync {
        return Ok(());
    }
    let check = || work.check().map_err(|_| store::Error::L1InputsChanged);
    let pending = context.store.checked_read_snapshot(&check, |tx| {
        Ok::<_, store::Error>(
            tx.query_row(
                "SELECT value FROM meta WHERE key='graph_pending'",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .as_deref()
                != Some("0"),
        )
    })??;
    if !pending {
        return Ok(());
    }
    let result = sync::linker::ingest(
        &context.root,
        &context.config,
        &context.store,
        work,
        &|| {
            super::super::repair::validate_configuration(context, work)
                .map_err(|_| store::Error::L1InputsChanged)
        },
    );
    check_query_work(work)?;
    if result
        .map_err(|_| super::super::Error::OpenSourceUnavailable)?
        .deferred
    {
        return Err(super::super::Error::OpenSourceUnavailable.into());
    }
    Ok(())
}

/// Retains every contributing edge and source, even when public paths deduplicate.
/// The caller repeats this read after its final-verification hook and compares it.
pub(super) fn populate(
    context: &Context,
    symbols: &mut [navigation::SymbolResult],
    work: sync::l1::Work<'_>,
) -> Result<()> {
    if symbols.is_empty() {
        return Ok(());
    }
    let l1 = navigation::L1Verifier::new(&context.root, &context.config)
        .ok_or(super::super::Error::OpenSourceUnavailable)?;
    let check = || work.check().map_err(|_| store::Error::L1InputsChanged);
    let ids = symbols
        .iter()
        .map(|symbol| symbol.selected.id)
        .collect::<Vec<_>>();
    let names = symbol_names(symbols);
    let edges = context
        .store
        .graph_edges_for_symbols_checked(&ids, &names, &check)?;
    // Keep the graph's edge order, but discard rows that cannot contribute to
    // this selected response before loading any source metadata.
    let contributing = edges
        .iter()
        .filter(|edge| {
            matches!(
                edge.kind,
                store::GraphEdgeKind::DocMention | store::GraphEdgeKind::TestOf
            ) && symbols
                .iter()
                .any(|symbol| edge_targets_symbol(edge, symbol))
        })
        .collect::<Vec<_>>();
    let source_ids = contributing
        .iter()
        .map(|edge| edge.src_symbol)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let sources: BTreeMap<_, _> = context
        .store
        .tool_symbols_by_ids_checked(&source_ids, &check)?
        .into_iter()
        .map(|s| (s.id, s))
        .collect();
    let globs = crate::treat::linker::test_paths(&context.config.tests.globs)
        .map_err(|_| super::super::Error::OpenSourceUnavailable)?;
    let file_ids = contributing
        .iter()
        .map(|edge| {
            sources
                .get(&edge.src_symbol)
                .map(|source| source.file_id)
                .ok_or(super::super::Error::OpenSourceUnavailable)
        })
        .collect::<std::result::Result<BTreeSet<_>, _>>()?;
    // One checked reader covers all active-file metadata. The final response
    // pass performs this query again, independently of the first proof.
    let active = context.store.checked_read_snapshot(&check, |tx| {
        let mut statement = tx
            .prepare("SELECT max_level>=1 AND l1_deferred_reason IS NULL FROM files WHERE id=?1")?;
        let mut active = BTreeMap::new();
        for file_id in &file_ids {
            check()?;
            active.insert(
                *file_id,
                statement.query_row([file_id], |row| row.get::<_, bool>(0))?,
            );
        }
        Ok::<_, store::Error>(active)
    })??;
    let mut unit_ids = BTreeSet::new();
    for edge in &contributing {
        let source = sources
            .get(&edge.src_symbol)
            .ok_or(super::super::Error::OpenSourceUnavailable)?;
        if active
            .get(&source.file_id)
            .copied()
            .ok_or(super::super::Error::OpenSourceUnavailable)?
            && let Some(id) = source.unit_id
        {
            unit_ids.insert(id);
        }
    }
    let units = context
        .store
        .units_by_ids_checked(&unit_ids.into_iter().collect::<Vec<_>>(), &check)?
        .into_iter()
        .map(|unit| (unit.id, unit))
        .collect::<BTreeMap<_, _>>();
    let mut groups: BTreeMap<String, BTreeSet<i64>> = BTreeMap::new();
    for edge in &contributing {
        check()?;
        let source = sources
            .get(&edge.src_symbol)
            .ok_or(super::super::Error::OpenSourceUnavailable)?;
        if !active
            .get(&source.file_id)
            .copied()
            .ok_or(super::super::Error::OpenSourceUnavailable)?
        {
            continue;
        }
        let unit = source
            .unit_id
            .and_then(|id| units.get(&id))
            .ok_or(super::super::Error::OpenSourceUnavailable)?;
        if unit.path.as_deref() != Some(source.path.as_str()) || edge.file_id != source.file_id {
            return Err(super::super::Error::OpenSourceUnavailable.into());
        }
        groups
            .entry(source.path.clone())
            .or_default()
            .insert(edge.line.unwrap_or(source.line_start));
    }
    let max_file_bytes = context.config.limits.max_file_bytes()?.0;
    let mut authorities = BTreeMap::new();
    for (path, lines) in groups {
        check()?;
        let group = navigation::ReferenceGroup {
            path: path.clone(),
            lines,
            kind: "ref",
        };
        let (_, authority) = navigation::reference_authority_with(
            &context.store,
            &context.root,
            &group,
            max_file_bytes,
            work,
            &l1,
        )?
        .map_err(|_| super::super::Error::OpenSourceUnavailable)?;
        authorities.insert(path, authority);
    }
    for symbol in symbols {
        let mut tests = BTreeSet::new();
        let mut docs = BTreeMap::new();
        let mut evidence = Vec::new();
        for edge in &contributing {
            check()?;
            if !edge_targets_symbol(edge, symbol) {
                continue;
            }
            let source = sources
                .get(&edge.src_symbol)
                .ok_or(super::super::Error::OpenSourceUnavailable)?;
            if !active
                .get(&source.file_id)
                .copied()
                .ok_or(super::super::Error::OpenSourceUnavailable)?
            {
                continue;
            }
            let mut unit = source
                .unit_id
                .and_then(|id| units.get(&id))
                .cloned()
                .ok_or(super::super::Error::OpenSourceUnavailable)?;
            let authority = authorities
                .get(&source.path)
                .ok_or(super::super::Error::OpenSourceUnavailable)?
                .clone();
            if edge.kind == store::GraphEdgeKind::TestOf {
                if !globs.is_match(&source.path) || globs.is_match(&symbol.path) {
                    return Err(super::super::Error::OpenSourceUnavailable.into());
                }
                tests.insert(source.path.clone());
            } else {
                if source.kind != "doc" || unit.kind != "doc_section" {
                    return Err(super::super::Error::OpenSourceUnavailable.into());
                }
                docs.insert(navigation::public_id(&unit), unit.header.clone());
            }
            // Independent embedding completion does not alter L1 link authority.
            unit.ladder_level = 0;
            evidence.push(navigation::LinkEvidence {
                edge: (**edge).clone(),
                unit,
                source: authority,
            });
        }
        symbol.tests = tests.into_iter().collect();
        symbol.documented_in = docs
            .into_iter()
            .map(|(id, header)| navigation::Documentation { id, header })
            .collect();
        symbol.links = evidence;
    }
    Ok(())
}

/// Repairs only stale sources already contributing to the selected definitions.
/// Newly discovered relationships remain the watcher's/full index's responsibility.
pub(super) fn repair_sources(
    context: &Context,
    symbols: &[navigation::SymbolResult],
    work: sync::l1::Work<'_>,
) -> Result<bool> {
    if !context.config.search.lazy_resync {
        return Ok(false);
    }
    let l1 = navigation::L1Verifier::new(&context.root, &context.config)
        .ok_or(super::super::Error::OpenSourceUnavailable)?;
    let check = || work.check().map_err(|_| store::Error::L1InputsChanged);
    let ids = symbols
        .iter()
        .map(|symbol| symbol.selected.id)
        .collect::<Vec<_>>();
    let names = symbol_names(symbols);
    let edges = context
        .store
        .graph_edges_for_symbols_checked(&ids, &names, &check)?;
    let source_ids = edges
        .iter()
        .map(|edge| edge.src_symbol)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let sources: BTreeMap<_, _> = context
        .store
        .tool_symbols_by_ids_checked(&source_ids, &check)?
        .into_iter()
        .map(|s| (s.id, s))
        .collect();
    let mut groups: BTreeMap<String, BTreeSet<i64>> = BTreeMap::new();
    for edge in edges {
        check()?;
        if !matches!(
            edge.kind,
            store::GraphEdgeKind::DocMention | store::GraphEdgeKind::TestOf
        ) || !symbols.iter().any(|symbol| {
            edge.dst_symbol.map_or_else(
                || {
                    edge.dst_name == symbol.selected.name
                        || Some(&edge.dst_name) == symbol.selected.qualified.as_ref()
                },
                |id| id == symbol.selected.id,
            )
        }) {
            continue;
        }
        if let Some(source) = sources.get(&edge.src_symbol) {
            let active = context.store.checked_read_snapshot(&check, |tx| {
                tx.query_row(
                    "SELECT max_level>=1 AND l1_deferred_reason IS NULL FROM files WHERE id=?1",
                    [source.file_id],
                    |r| r.get::<_, bool>(0),
                )
            })??;
            if active {
                groups
                    .entry(source.path.clone())
                    .or_default()
                    .insert(edge.line.unwrap_or(source.line_start));
            }
        }
    }
    let mut repaired = false;
    for (path, lines) in groups {
        check()?;
        let group = navigation::ReferenceGroup {
            path: path.clone(),
            lines,
            kind: "ref",
        };
        if navigation::reference_authority_with(
            &context.store,
            &context.root,
            &group,
            context.config.limits.max_file_bytes()?.0,
            work,
            &l1,
        )?
        .is_err()
        {
            let result = super::super::repair::target(context, &path, work);
            check_query_work(work)?;
            if !matches!(result, Ok(true)) {
                return Err(super::super::Error::OpenSourceUnavailable.into());
            }
            repaired = true;
        }
    }
    Ok(repaired)
}

fn symbol_names(symbols: &[navigation::SymbolResult]) -> Vec<String> {
    let mut names = BTreeSet::new();
    for symbol in symbols {
        names.insert(symbol.selected.name.clone());
        if let Some(qualified) = &symbol.selected.qualified {
            names.insert(qualified.clone());
        }
    }
    names.into_iter().collect()
}

fn edge_targets_symbol(edge: &store::GraphEdge, symbol: &navigation::SymbolResult) -> bool {
    edge.dst_symbol.map_or_else(
        || {
            edge.dst_name == symbol.selected.name
                || Some(&edge.dst_name) == symbol.selected.qualified.as_ref()
        },
        |id| id == symbol.selected.id,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    fn fixture() -> (tempfile::TempDir, Context) {
        let root = tempfile::tempdir().unwrap();
        for (path, source) in [
            ("src/retry.rs", "pub fn should_retry() -> bool { true }\n"),
            (
                "tests/retry.rs",
                "fn test_retry() { should_retry(); }\nfn test_again() { should_retry(); }\n",
            ),
            (
                "README.md",
                "# Guide\nUse `should_retry`.\nAgain, `should_retry`.\n",
            ),
        ] {
            fs::create_dir_all(root.path().join(path).parent().unwrap()).unwrap();
            fs::write(root.path().join(path), source).unwrap();
        }
        fs::create_dir(root.path().join(".deepkit")).unwrap();
        fs::write(root.path().join(".deepkit/config.toml"),"[embedder]\nprofile='local'\n[commits]\nenabled=false\n[llm]\nenabled=false\n[limits]\nquery_timeout_ms=10000\n").unwrap();
        let context = Context::load(root.path()).unwrap();
        sync::index::run(
            sync::index::Input {
                config_root: &context.config_root,
                root: &context.root,
                settings: &context.config,
                store: &context.store,
                full: false,
                work: Default::default(),
            },
            |_| {},
        )
        .unwrap();
        (root, context)
    }
    #[tokio::test]
    async fn linker_final_fence_refuses_changed_sources_headers_and_hidden_edges() {
        for mutation in ["doc_source", "test_source", "header", "edge", "settings"] {
            let (_root, context) = fixture();
            let before = context.symbol("should_retry").await.unwrap();
            assert_eq!(before[0]["tests"], json!(["tests/retry.rs"]));
            assert_eq!(before[0]["documented_in"].as_array().unwrap().len(), 1);
            let worker = context.clone();
            let result=context.symbol_checked("should_retry",move ||match mutation {
                "doc_source"=>fs::write(worker.root.join("README.md"),"# Changed\nNo mention.\n").unwrap(),
                "test_source"=>fs::write(worker.root.join("tests/retry.rs"),"fn test_retry() {}\n").unwrap(),
                "header"=>{worker.store.writer().unwrap().execute("UPDATE units SET header='Changed' WHERE kind='doc_section'",[]).unwrap();},
                "edge"=>{worker.store.writer().unwrap().execute("INSERT INTO edges SELECT src_symbol,dst_name,dst_symbol,kind,file_id,line FROM edges WHERE kind='doc_mention'",[]).unwrap();},
                _=>fs::write(worker.root.join(".deepkit/config.local.toml"),"[tests]\nglobs=['elsewhere/**']\n").unwrap(),
            }).await;
            assert_eq!(
                result.unwrap_err().code(),
                "source_unavailable",
                "{mutation}"
            );
        }
    }

    #[tokio::test]
    async fn shared_test_source_keeps_deduplicated_navigation() {
        let (_root, context) = fixture();
        context
            .store
            .writer()
            .unwrap()
            .execute(
                "INSERT INTO edges SELECT src_symbol,dst_name,dst_symbol,kind,file_id,2 FROM edges WHERE kind='test_of' LIMIT 1",
                [],
            )
            .unwrap();
        let before = context.symbol("should_retry").await.unwrap();
        assert_eq!(before[0]["tests"], json!(["tests/retry.rs"]));
        let edges: i64 = context
            .store
            .writer()
            .unwrap()
            .query_row(
                "SELECT count(*) FROM edges WHERE kind='test_of'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(edges >= 2);
    }
}
