//! Parent projection and result diversification after ranking; v1 §9 step 8.
use super::{Error, RankedCandidate};
use crate::{
    config::Search,
    store::{self, Store, StoredUnit},
};
use rusqlite::{OptionalExtension, Transaction};
use std::collections::{BTreeMap, BTreeSet};

/// Identity, ownership and containment fields used to resolve a parent chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ProjectionLink {
    id: i64,
    kind: String,
    parent: Option<i64>,
    pub(super) file_id: Option<i64>,
    byte_start: Option<i64>,
    byte_end: Option<i64>,
}
fn metadata(tx: &Transaction<'_>, id: i64) -> Result<Option<ProjectionLink>, Error> {
    tx.query_row(
        "SELECT kind,parent_unit,file_id,byte_start,byte_end FROM units WHERE id=?1",
        [id],
        |r| {
            Ok(ProjectionLink {
                id,
                kind: r.get(0)?,
                parent: r.get(1)?,
                file_id: r.get(2)?,
                byte_start: r.get(3)?,
                byte_end: r.get(4)?,
            })
        },
    )
    .optional()
    .map_err(store::Error::from)
    .map_err(Error::from)
}
struct Projection {
    unit: StoredUnit,
    summary: bool,
    chunk: bool,
}
fn project(
    tx: &Transaction<'_>,
    id: i64,
    exclude: &BTreeSet<i64>,
    check: &dyn Fn() -> Result<(), Error>,
    links: &mut Vec<ProjectionLink>,
) -> Result<Option<Projection>, Error> {
    check()?;
    let mut current = id;
    if exclude.contains(&current) {
        return Ok(None);
    }
    let Some(mut meta) = metadata(tx, current)? else {
        return Ok(None);
    };
    links.push(meta.clone());
    let mut visited = BTreeSet::from([current]);
    let mut summary = false;
    while meta.kind == "summary" {
        check()?;
        summary = true;
        current = meta
            .parent
            .ok_or(Error::InvalidSummaryParent { unit_id: id })?;
        if exclude.contains(&current) {
            return Ok(None);
        }
        if !visited.insert(current) {
            return Err(Error::InvalidSummaryParent { unit_id: id });
        }
        meta = metadata(tx, current)?.ok_or(Error::InvalidSummaryParent { unit_id: id })?;
        links.push(meta.clone());
    }
    check()?;
    let mut chunk = false;
    if meta.kind == "chunk"
        && let Some(parent) = store::projection::chunk_parent(tx, current)?
    {
        current = parent;
        chunk = true;
        // The same parent ID can survive changed containment bounds. Retain its
        // authority in this snapshot, including when later excluded or hidden.
        check()?;
        if let Some(meta) = metadata(tx, current)? {
            links.push(meta);
        }
    }
    if exclude.contains(&current) {
        return Ok(None);
    }
    check()?;
    Ok(store::unit_by_id_in(tx, current)?.map(|unit| Projection {
        unit,
        summary,
        chunk,
    }))
}

/// Collapses an already ranked candidate window, preserving its first/strongest
/// representative's order and scores rather than adding duplicate-match votes.
/// Summary chains resolve to a real parent; chunks choose the smallest containing
/// symbol byte range in the same file. Excluded originals or projected parents are
/// never reintroduced. Deleted occurrences disappear; malformed parent chains fail
/// explicitly. All projections use one read snapshot. Apply after reranking so
/// diversification order survives, then cap each real file and truncate to k.
#[cfg(test)]
pub(crate) fn apply(
    store: &Store,
    hits: Vec<RankedCandidate>,
    k: usize,
    settings: &Search,
    exclude: &BTreeSet<i64>,
) -> Result<Vec<RankedCandidate>, Error> {
    if k == 0 || k as u128 > u128::from(settings.max_k) {
        return Err(Error::InvalidK {
            max: settings.max_k,
        });
    }
    let prepared = prepare(
        store,
        &hits.iter().map(|hit| hit.unit.id).collect::<Vec<_>>(),
        exclude,
    )?;
    finish(hits, k, settings, &prepared)
}

/// Original occurrences and parent identities from one read snapshot. Retaining
/// both makes final collapse independent of database availability at the deadline.
#[derive(Default)]
pub(crate) struct Prepared {
    pub(crate) units: BTreeMap<i64, StoredUnit>,
    projections: BTreeMap<i64, Option<Projection>>,
    pub(crate) links: BTreeMap<i64, Vec<ProjectionLink>>,
    pub(crate) publications: BTreeMap<i64, Option<store::projection::SourcePublication>>,
}
impl Prepared {
    /// Resolves a retained occurrence to the same parent used by final collapse.
    /// Diagnostics use this so a PR summary receives its parent's rank weight.
    pub(crate) fn projected_unit(&self, id: i64) -> Option<&StoredUnit> {
        self.projections
            .get(&id)?
            .as_ref()
            .map(|projection| &projection.unit)
    }

    /// Identifies displayed results supported by incomplete hidden publications.
    /// Some(empty) still means incomplete coverage: the hidden candidate affected
    /// selection but contributed no displayed match. Independent hits survive.
    pub(crate) fn incomplete_hidden_hits(
        &self,
        hits: &[RankedCandidate],
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<Option<BTreeSet<i64>>, Error> {
        let mut incomplete = BTreeSet::new();
        for (file, publication) in &self.publications {
            check()?;
            if !publication
                .as_ref()
                .is_some_and(|source| source.is_complete())
            {
                incomplete.insert(*file);
            }
        }
        if incomplete.is_empty() {
            return Ok(None);
        }
        let mut affected = BTreeSet::new();
        for hit in hits {
            check()?;
            for entry in &hit.matches {
                check()?;
                if let Some(links) = self.links.get(&entry.unit_id) {
                    for link in links {
                        check()?;
                        if link.file_id.is_some_and(|file| incomplete.contains(&file)) {
                            affected.insert(hit.unit.id);
                        }
                    }
                }
            }
        }
        check()?;
        Ok(Some(affected))
    }

    /// Leaves returned files to their per-file source verifier, so a changed
    /// displayed source does not invalidate otherwise verified independent hits.
    pub(crate) fn retain_hidden_publications(
        &mut self,
        hits: &[RankedCandidate],
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<(), Error> {
        let displayed = hits.iter().map(|hit| hit.unit.id).collect::<BTreeSet<_>>();
        for link in self.links.values().flatten() {
            check()?;
            if displayed.contains(&link.id)
                && let Some(file_id) = link.file_id
            {
                self.publications.remove(&file_id);
            }
        }
        check()?;
        Ok(())
    }

    /// Retains the original and projected rows before file/k truncation so a
    /// final comparison cannot overlook a changed hidden or collapsed match.
    /// Ladder progress is independent of the L1 identity used for selection.
    pub(crate) fn occurrence_evidence(
        &self,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<Vec<StoredUnit>, Error> {
        let mut evidence = Vec::new();
        for unit in self.units.values().chain(
            self.projections
                .values()
                .filter_map(|projection| projection.as_ref().map(|projection| &projection.unit)),
        ) {
            check()?;
            let mut unit = unit.clone();
            unit.ladder_level = 0;
            evidence.push(unit);
        }
        check()?;
        Ok(evidence)
    }

    /// Merges another retrieval leg; its matching occurrence and projection replace
    /// the older pair together when the same ID appeared in both snapshots.
    pub(crate) fn extend(&mut self, other: Self) {
        for id in other.projections.keys() {
            self.units.remove(id);
        }
        self.units.extend(other.units);
        self.projections.extend(other.projections);
        self.links.extend(other.links);
        self.publications.extend(other.publications);
    }
}

/// Hydrates bounded retrieval IDs and collapse targets together. Empty legs avoid
/// a database read, allowing lexical-only and missing-tier queries to stay cheap.
#[cfg(test)]
pub(crate) fn prepare(
    store: &Store,
    ids: &[i64],
    exclude: &BTreeSet<i64>,
) -> Result<Prepared, Error> {
    prepare_with_work(store, ids, exclude, None)
}

/// Keeps hydration and each parent-chain step under the original query work.
/// Ordinary callers retain their configured reader wait policy.
pub(crate) fn prepare_with_work(
    store: &Store,
    ids: &[i64],
    exclude: &BTreeSet<i64>,
    work: Option<&dyn Fn() -> Result<(), Error>>,
) -> Result<Prepared, Error> {
    let check = work.unwrap_or(&|| Ok(()));
    check()?;
    if ids.is_empty() {
        return Ok(Prepared::default());
    }
    let read = |tx: &Transaction<'_>, check: &dyn Fn() -> Result<(), Error>| {
        let mut prepared = Prepared::default();
        for &id in ids {
            check()?;
            if prepared.projections.contains_key(&id) {
                continue;
            }
            let mut links = Vec::new();
            let projection = project(tx, id, exclude, check, &mut links)?;
            if work.is_some() {
                for link in &links {
                    check()?;
                    if let Some(file_id) = link.file_id
                        && let std::collections::btree_map::Entry::Vacant(entry) =
                            prepared.publications.entry(file_id)
                    {
                        entry.insert(store::projection::source_publication(tx, file_id)?);
                    }
                }
            }
            prepared.links.insert(id, links);
            check()?;
            if projection.is_some()
                && let Some(unit) = store::unit_by_id_in(tx, id)?
            {
                prepared.units.insert(id, unit);
            }
            prepared.projections.insert(id, projection);
        }
        check()?;
        Ok(prepared)
    };
    super::local_store_read(
        work,
        || store.read_snapshot(|tx| read(tx, &|| Ok(()))),
        |admission| {
            store.checked_read_snapshot(admission, |tx| {
                read(tx, &|| admission().map_err(Into::into))
            })
        },
    )?
}

/// Applies parent merging, file caps and k to an already ranked window using only
/// prepared data. Call after optional reranking; preserves scores and input order.
/// Freshness validation remains a separate stage, not implied by cached metadata.
pub(crate) fn finish(
    hits: Vec<RankedCandidate>,
    k: usize,
    settings: &Search,
    prepared: &Prepared,
) -> Result<Vec<RankedCandidate>, Error> {
    if k == 0 || k as u128 > u128::from(settings.max_k) {
        return Err(Error::InvalidK {
            max: settings.max_k,
        });
    }
    let mut positions = BTreeMap::new();
    let mut output: Vec<RankedCandidate> = Vec::new();
    for mut hit in hits {
        let Some(Some(projection)) = prepared.projections.get(&hit.unit.id) else {
            continue;
        };
        hit.unit = projection.unit.clone();
        hit.summary |= projection.summary;
        hit.chunk |= projection.chunk;
        if let Some(&position) = positions.get(&hit.unit.id) {
            let kept: &mut RankedCandidate = &mut output[position];
            kept.lexical |= hit.lexical;
            kept.graph |= hit.graph;
            kept.dense |= hit.dense;
            kept.exact_name |= hit.exact_name;
            kept.summary |= hit.summary;
            kept.chunk |= hit.chunk;
            for evidence in hit.matches {
                if !kept
                    .matches
                    .iter()
                    .any(|prior| prior.unit_id == evidence.unit_id)
                {
                    kept.matches.push(evidence);
                }
            }
        } else {
            positions.insert(hit.unit.id, output.len());
            output.push(hit);
        }
    }
    let mut per_file = BTreeMap::<String, u64>::new();
    output.retain(|hit| {
        let Some(path) = &hit.unit.path else {
            return true;
        };
        let count = per_file.entry(path.clone()).or_default();
        if *count >= settings.max_hits_per_file {
            return false;
        }
        *count += 1;
        true
    });
    output.truncate(k);
    Ok(output)
}

#[cfg(test)]
mod tests;
