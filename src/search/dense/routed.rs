//! Snapshot-consistent dense lists across configured/transition spaces; v1 §7.3/§9.
use super::*;
use crate::{
    config::{self, Config},
    store::{Store, transitions},
};
use rusqlite::Transaction;
use std::{collections::BTreeMap, path::PathBuf};

/// Providers are prepared/reused by the host with explicit budgets and cache
/// policy. Missing providers become visible per-space failures, never guesses.
pub(crate) struct Providers {
    pub(crate) store: Arc<Store>,
    pub(crate) config: Config,
    pub(crate) config_root: PathBuf,
    pub(crate) factories: Vec<Arc<Factory>>,
}
/// Independent ranked list: distances must never be compared across spaces.
#[derive(Debug)]
pub(crate) struct SpaceResult {
    pub(crate) table: String,
    pub(crate) hits: Result<Vec<VectorHit>, Error>,
}
/// A single routing generation with successful lists and explicit partial errors.
#[derive(Debug)]
pub(crate) struct Results {
    pub(crate) generation: i64,
    pub(crate) spaces: Vec<SpaceResult>,
}
#[derive(Default)]
struct Space {
    kinds: Vec<UnitKind>,
    replaced_by: Vec<(UnitKind, String)>,
    incoming: bool,
}
fn selected(scope: &Scope, kind: UnitKind, summaries: bool) -> bool {
    scope.kinds.is_empty()
        || (kind == UnitKind::Summary && summaries)
        || scope.kinds.contains(&kind)
}
fn spaces(state: &transitions::State, scope: &Scope, summaries: bool) -> BTreeMap<String, Space> {
    let mut spaces = BTreeMap::<String, Space>::new();
    for kind in UnitKind::ALL
        .into_iter()
        .filter(|&k| selected(scope, k, summaries))
    {
        if let Some(table) = state.desired.get(kind.as_str()) {
            let space = spaces.entry(table.clone()).or_default();
            space.kinds.push(kind);
            space.incoming |= state
                .previous
                .iter()
                .any(|source| source.kinds.contains(kind.as_str()));
        }
    }
    for source in &state.previous {
        for kind in UnitKind::ALL
            .into_iter()
            .filter(|&k| selected(scope, k, summaries) && source.kinds.contains(k.as_str()))
        {
            let space = spaces.entry(source.table.clone()).or_default();
            if !space.kinds.contains(&kind) {
                space.kinds.push(kind);
            }
            if let Some(destination) = state.desired.get(kind.as_str()) {
                space.replaced_by.push((kind, destination.clone()));
            }
        }
    }
    spaces
}
fn current(providers: &Providers) -> Result<(), Error> {
    let config = config::load(&providers.config_root, None)
        .map_err(|_| queue::Error::Invalid("query configuration reload"))?
        .config;
    if config != providers.config {
        return Err(Error::ConfigurationChanged);
    }
    Ok(())
}
fn populated(tx: &Transaction<'_>, table: &str, space: &Space) -> Result<bool, Error> {
    let exists: bool = tx
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM embedding_tables WHERE slug=?1)",
            [table],
            |r| r.get(0),
        )
        .map_err(store::Error::from)?;
    if !exists {
        return Ok(false);
    }
    let table = vectors::vector_table_name(table)?;
    // Kind names are closed enum values, not user SQL. This read only determines
    // whether embedding is needed; actual retrieval applies every scope predicate.
    let kinds = space
        .kinds
        .iter()
        .map(|k| format!("'{}'", k.as_str()))
        .collect::<Vec<_>>()
        .join(",");
    Ok(tx.query_row(&format!("SELECT EXISTS(SELECT 1 FROM {table} v JOIN units u ON u.id=v.unit_id WHERE u.kind IN ({kinds}))"),[],|r|r.get(0)).map_err(store::Error::from)?)
}
fn matching(
    factory: &Factory,
    tx: &Transaction<'_>,
    table: &str,
    space: &Space,
) -> Result<bool, Error> {
    let permitted = factory.scan_kinds(&space.kinds).unwrap_or_default();
    Ok(permitted.len() == space.kinds.len()
        && factory.table_matches(tx, table)?
        && factory.current()?)
}

/// Embeds required spaces in parallel, then scans all surviving rows in one read
/// snapshot. Pending destinations are prepared even while empty, closing the
/// old-to-new movement gap. Every operation shares the original absolute deadline.
/// Returned lists retain separate ranks for one later RRF pass.
pub(crate) async fn retrieve<C>(
    providers: Providers,
    text: String,
    scope: Scope,
    deadline: Instant,
    clock: C,
) -> Result<Results, Error>
where
    C: Fn() -> i64 + Send + Sync + 'static,
{
    #[cfg(test)]
    let scan_observer = vectors::scan_observation::current();
    if providers.factories.len() > 4 || providers.config.limits.query_timeout_ms == 0 {
        return Err(Error::InvalidProviders);
    }
    let configured = Instant::now()
        .checked_add(std::time::Duration::from_millis(
            providers.config.limits.query_timeout_ms,
        ))
        .ok_or(Error::Deadline)?;
    let deadline = deadline.min(configured);
    let mut factories = BTreeMap::new();
    for factory in &providers.factories {
        if !Arc::ptr_eq(factory.store(), &providers.store)
            || !factory.matches_configuration(&providers.config)
            || factories
                .insert(factory.table_slug(), Arc::clone(factory))
                .is_some()
        {
            return Err(Error::InvalidProviders);
        }
    }
    let factories = Arc::new(factories);
    let providers = Arc::new(providers);
    let scope = Arc::new(scope);
    let stopped = Arc::new(AtomicBool::new(false));
    let _stop = Stop(Arc::clone(&stopped));
    let operation = async {
        let context = Arc::clone(&providers);
        let filter = Arc::clone(&scope);
        let cancelled = Arc::clone(&stopped);
        let available = Arc::clone(&factories);
        let (state, needs) = tokio::task::spawn_blocking(move || {
            check(deadline, &cancelled)?;
            current(&context)?;
            context.store.read_snapshot(|tx| -> Result<_, Error> {
                let state = transitions::read(tx)?.ok_or(Error::RoutingChanged)?;
                let summaries = store::projection::summaries_in_scope(tx, &filter.kinds)?;
                let needs = spaces(&state, &filter, summaries)
                    .into_iter()
                    .map(|(table, space)| {
                        let need = (|| {
                            if !populated(tx, &table, &space)? && !space.incoming {
                                return Ok(false);
                            }
                            let factory = available.get(&table).ok_or(Error::Unavailable)?;
                            if !matching(factory, tx, &table, &space)? {
                                return Err(Error::TableChanged);
                            }
                            Ok(true)
                        })();
                        (table, need)
                    })
                    .collect::<BTreeMap<_, _>>();
                Ok((state, needs))
            })?
        })
        .await
        .map_err(|_| Error::Task)??;
        let clock = Arc::new(clock);
        let mut tasks = tokio::task::JoinSet::new();
        let mut queries = BTreeMap::new();
        for (table, needed) in needs {
            match needed {
                Ok(false) => continue,
                Err(error) => {
                    queries.insert(table, Err(error));
                    continue;
                }
                Ok(true) => {}
            }
            if let Some(factory) = factories.get(&table) {
                let factory = Arc::clone(factory);
                let text = text.clone();
                let clock = Arc::clone(&clock);
                tasks.spawn(async move {
                    (
                        table,
                        factory
                            .embed(text, deadline, move || clock())
                            .await
                            .map_err(Error::from),
                    )
                });
            }
        }
        while let Some(result) = tasks.join_next().await {
            let (table, vector) = result.map_err(|_| Error::Task)?;
            queries.insert(table, vector);
        }
        let cancelled = Arc::clone(&stopped);
        tokio::task::spawn_blocking(move || {
            #[cfg(test)]
            let _scan_scope = vectors::scan_observation::enter(scan_observer);
            check(deadline, &cancelled)?;
            current(&providers)?;
            providers
                .store
                .vector_snapshot(|tx| -> Result<Results, Error> {
                    let now = transitions::read(tx)?.ok_or(Error::RoutingChanged)?;
                    if now.generation != state.generation || now.desired != state.desired {
                        return Err(Error::RoutingChanged);
                    }
                    let mut results = Vec::new();
                    let summaries = store::projection::summaries_in_scope(tx, &scope.kinds)?;
                    for (table, space) in spaces(&now, &scope, summaries) {
                        check(deadline, &cancelled)?;
                        let hits = (|| {
                            if !populated(tx, &table, &space)? {
                                return Ok(Vec::new());
                            }
                            let factory = factories.get(&table).ok_or(Error::Unavailable)?;
                            if !matching(factory, tx, &table, &space)? {
                                return Err(Error::TableChanged);
                            }
                            let vector = queries.remove(&table).ok_or(Error::Unprepared)??;
                            let predicate = VectorScanPredicate {
                                kinds: if space.kinds.len() == UnitKind::ALL.len() {
                                    &[]
                                } else {
                                    &space.kinds
                                },
                                scope_kinds: &scope.kinds,
                                path_globs: scope.path_globs.as_deref(),
                                exclude_ids: &scope.exclude_ids,
                                replaced_by: &space.replaced_by,
                            };
                            let mut hits =
                                vectors::scan_vectors_in(tx, &table, &vector, 400, predicate)?;
                            hits.truncate(200);
                            check(deadline, &cancelled)?;
                            Ok(hits)
                        })();
                        results.push(SpaceResult { table, hits });
                    }
                    Ok(Results {
                        generation: now.generation,
                        spaces: results,
                    })
                })?
        })
        .await
        .map_err(|_| Error::Task)?
    };
    tokio::time::timeout_at(deadline, operation)
        .await
        .map_err(|_| Error::Deadline)?
}

#[cfg(test)]
mod tests;
