//! Single-tier query embedding and scoped exact retrieval; v1 §9 step 4.
#![expect(dead_code, reason = "hybrid orchestration connects this leg in T1.7")]
use super::{admission, embedding_query::Factory};
use crate::{
    queue,
    store::{
        self, UnitKind,
        vectors::{self, VectorHit, VectorScanPredicate},
    },
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::time::Instant;

/// Owned query scope and exclusions, applied in SQL before exact top-k selection.
#[derive(Default)]
pub(crate) struct Scope {
    pub(crate) kinds: Vec<UnitKind>,
    pub(crate) path_globs: Option<Vec<String>>,
    pub(crate) exclude_ids: Vec<i64>,
}
impl Scope {
    /// Maps the same named groups accepted by lexical retrieval; other text is a
    /// SQLite path glob. An absent scope imposes no kind or path restriction.
    pub(crate) fn parse(scope: Option<&str>, exclude_ids: Vec<i64>) -> Self {
        use UnitKind::*;
        let kinds = match scope {
            Some("source") => vec![Chunk, Symbol, ConfigKey, DocSection, Image],
            Some("code") => vec![Chunk, Symbol, ConfigKey],
            Some("docs") => vec![DocSection, Image, Module],
            Some("images") => vec![Image],
            Some("commits") => vec![Commit],
            Some("prs") => vec![PullRequest, History],
            _ => Vec::new(),
        };
        let path_globs = scope
            .filter(|_| kinds.is_empty())
            .map(|s| vec![s.to_owned()]);
        Self {
            kinds,
            path_globs,
            exclude_ids,
        }
    }
    /// Preserve array entries as literal path globs, including reserved group names.
    pub(crate) fn from_search(scope: Option<&store::SearchScope>, exclude_ids: Vec<i64>) -> Self {
        match scope {
            Some(store::SearchScope::Globs(globs)) => Self {
                kinds: Vec::new(),
                path_globs: Some(globs.clone()),
                exclude_ids,
            },
            Some(store::SearchScope::One(one)) => Self::parse(Some(one), exclude_ids),
            None => Self::parse(None, exclude_ids),
        }
    }
    /// Summary vectors keep their stored routing kind; SQL checks their owner's
    /// scope before selecting candidates. Empty continues to mean every kind.
    pub(crate) fn retrieval_kinds(&self) -> Vec<UnitKind> {
        let mut kinds = self.kinds.clone();
        if !kinds.is_empty() && !kinds.contains(&UnitKind::Summary) {
            kinds.push(UnitKind::Summary);
        }
        kinds
    }
    fn predicate<'a>(&'a self, kinds: &'a [UnitKind]) -> VectorScanPredicate<'a> {
        VectorScanPredicate {
            kinds,
            scope_kinds: &self.kinds,
            path_globs: self.path_globs.as_deref(),
            exclude_ids: &self.exclude_ids,
            replaced_by: &[],
        }
    }
}

/// Dense-leg failures retain explicit degradation causes without query text.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("dense result projection failed: {0}")]
    Projection(#[source] super::Error),
    #[error(transparent)]
    Store(#[from] store::Error),
    #[error(transparent)]
    Query(#[from] admission::Error),
    #[error(transparent)]
    Policy(#[from] queue::Error),
    #[error("selected vector table is missing, inactive or belongs to another model")]
    TableChanged,
    #[error("query configuration changed before vector retrieval")]
    ConfigurationChanged,
    #[error("dense retrieval deadline exceeded")]
    Deadline,
    #[error("dense retrieval task failed")]
    Task,
    #[error("embedding routes changed or have not been prepared")]
    RoutingChanged,
    #[error("no matching prepared query provider for this vector space")]
    Unavailable,
    #[error("vector coverage appeared after empty-space preflight")]
    Unprepared,
    #[error("invalid or duplicate prepared query providers")]
    InvalidProviders,
}
struct Stop(Arc<AtomicBool>);
impl Drop for Stop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}
fn check(deadline: Instant, stopped: &AtomicBool) -> Result<(), Error> {
    if stopped.load(Ordering::SeqCst) || Instant::now() >= deadline {
        Err(Error::Deadline)
    } else {
        Ok(())
    }
}

/// Embeds once per selected tier (with separately admitted retries), exact-scans
/// top 400 eligible rows and keeps 200 occurrence IDs in distance order. Only the
/// explicitly selected table is read. Empty tables require no provider request.
/// A metadata/configuration change while HTTP runs discards the result. Native
/// scans already running retain their reader until completion after cancellation;
/// queued scans recheck cancellation after acquiring their reader snapshot.
pub(crate) async fn retrieve<C>(
    factory: Arc<Factory>,
    table: String,
    text: String,
    scope: Scope,
    deadline: Instant,
    clock: C,
) -> Result<Vec<VectorHit>, Error>
where
    C: Fn() -> i64 + Send + Sync + 'static,
{
    let deadline = factory.deadline(deadline)?;
    let Some(kinds) = factory.scan_kinds(&scope.retrieval_kinds()) else {
        return Ok(Vec::new());
    };
    let stopped = Arc::new(AtomicBool::new(false));
    let _stop = Stop(Arc::clone(&stopped));
    let operation = async {
        check(deadline, &stopped)?;
        let table_name = vectors::vector_table_name(&table)?;
        let provider = Arc::clone(&factory);
        let selected = table.clone();
        let cancelled = Arc::clone(&stopped);
        let populated = tokio::task::spawn_blocking(move || {
            provider.store().read_snapshot(|tx| {
                check(deadline, &cancelled)?;
                if !provider.table_matches(tx, &selected)? {
                    return Err(Error::TableChanged);
                }
                if !provider.current()? {
                    return Err(Error::ConfigurationChanged);
                }
                tx.query_row(
                    &format!("SELECT EXISTS(SELECT 1 FROM {table_name})"),
                    [],
                    |r| r.get::<_, bool>(0),
                )
                .map_err(store::Error::from)
                .map_err(Error::from)
            })?
        })
        .await
        .map_err(|_| Error::Task)??;
        if !populated {
            return Ok(Vec::new());
        }
        let vector = factory.embed(text, deadline, clock).await?;
        let cancelled = Arc::clone(&stopped);
        tokio::task::spawn_blocking(move || {
            let mut hits = factory.store().scan_vectors_checked(
                &table,
                &vector,
                400,
                scope.predicate(&kinds),
                |tx| {
                    check(deadline, &cancelled)?;
                    if !factory.table_matches(tx, &table)? {
                        return Err(Error::TableChanged);
                    }
                    if !factory.current()? {
                        return Err(Error::ConfigurationChanged);
                    }
                    Ok(())
                },
            )?;
            check(deadline, &cancelled)?;
            // A vector table has one row per occurrence; the set additionally
            // documents and preserves deduplication at this retrieval boundary.
            let mut seen = std::collections::BTreeSet::new();
            hits.retain(|hit| seen.insert(hit.unit_id));
            hits.truncate(200);
            Ok(hits)
        })
        .await
        .map_err(|_| Error::Task)?
    };
    let result = tokio::time::timeout_at(deadline, operation)
        .await
        .map_err(|_| Error::Deadline)?;
    check(deadline, &stopped)?;
    result
}

pub(crate) mod routed;

#[cfg(test)]
mod tests;
