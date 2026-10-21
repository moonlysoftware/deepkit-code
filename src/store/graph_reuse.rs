//! Retained per-occurrence graph contributions; see v1.md §4.3 and §6.4.
use rusqlite::{Transaction, params};

use super::{Error, GraphSymbol, SymbolInsert, UnitEdgeInsert};

/// Reuses identical graph topology and refreshes source positions atomically.
/// Retaining symbol IDs also retains incoming links, ranks and derived references.
pub(super) fn retain(
    tx: &Transaction<'_>,
    file_id: i64,
    unit_id: i64,
    path: &str,
    symbols: &[SymbolInsert<'_>],
    edges: &[UnitEdgeInsert<'_>],
) -> Result<Option<Vec<GraphSymbol>>, Error> {
    let mut previous = tx
        .prepare_cached(
            "SELECT id,name,qualified,kind,signature,line_start,line_end,pagerank
         FROM symbols WHERE file_id=?1 AND unit_id=?2 AND kind!='doc' ORDER BY id",
        )?
        .query_map(params![file_id, unit_id], |r| {
            Ok(GraphSymbol {
                id: r.get(0)?,
                file_id,
                path: path.to_owned(),
                name: r.get(1)?,
                qualified: r.get(2)?,
                kind: r.get(3)?,
                signature: r.get(4)?,
                line_start: r.get(5)?,
                line_end: r.get(6)?,
                pagerank: r.get(7)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    if previous.len() != symbols.len()
        || previous.iter().zip(symbols).any(|(old, new)| {
            old.name != new.name
                || old.qualified.as_deref() != new.qualified
                || old.kind != new.kind
        })
    {
        return Ok(None);
    }
    let previous_edges = tx
        .prepare_cached(
            "SELECT rowid,src_symbol,dst_name,dst_symbol,kind,line,file_id FROM edges
         WHERE src_symbol IN (SELECT id FROM symbols WHERE file_id=?1 AND unit_id=?2 AND kind!='doc')
         ORDER BY rowid",
        )?
        .query_map(params![file_id, unit_id], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, Option<i64>>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, Option<i64>>(5)?,
                r.get::<_, i64>(6)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    if previous_edges.len() != edges.len()
        || previous_edges.iter().zip(edges).any(|(old, new)| {
            old.1 != previous[new.src_symbol_index].id
                || old.2 != new.dst_name
                || old.3
                    != new
                        .dst_symbol_index
                        .map(|index| previous[index].id)
                        .or(new.dst_symbol)
                || (old.4 != new.kind.as_str()
                    && !(old.4 == "test_of" && new.kind.as_str() == "ref"))
                || old.6 != file_id
        })
    {
        return Ok(None);
    }
    for (old, new) in previous.iter_mut().zip(symbols) {
        if old.signature.as_deref() != new.signature
            || old.line_start != new.line_start
            || old.line_end != new.line_end
        {
            tx.prepare_cached(
                "UPDATE symbols SET signature=?1,line_start=?2,line_end=?3 WHERE id=?4",
            )?
            .execute(params![new.signature, new.line_start, new.line_end, old.id])?;
            old.signature = new.signature.map(str::to_owned);
            old.line_start = new.line_start;
            old.line_end = new.line_end;
        }
    }
    for (old, new) in previous_edges.iter().zip(edges) {
        if old.5 != new.line {
            tx.prepare_cached("UPDATE edges SET line=?1 WHERE rowid=?2")?
                .execute(params![new.line, old.0])?;
        }
    }
    Ok(Some(previous))
}

/// Removes one replaced contribution, or orphaned symbols after obsolete units
/// have been deleted. The file predicate prevents touching another source.
pub(super) fn remove(
    tx: &Transaction<'_>,
    file_id: i64,
    unit_id: Option<i64>,
) -> Result<(), Error> {
    tx.prepare_cached("DELETE FROM fts_names WHERE rowid IN (SELECT id FROM symbols WHERE file_id=?1 AND unit_id IS ?2)")?
        .execute(params![file_id,unit_id])?;
    tx.prepare_cached("DELETE FROM symbols WHERE file_id=?1 AND unit_id IS ?2")?
        .execute(params![file_id, unit_id])?;
    Ok(())
}

/// Removes obsolete definitions before their units, avoiding an intermediate
/// ON DELETE SET NULL update of every definition that will immediately be removed.
/// Both steps stay inside the caller's atomic file replacement transaction.
pub(super) fn remove_obsolete(tx: &Transaction<'_>, file_id: i64, ids: &str) -> Result<(), Error> {
    tx.prepare_cached(
        "DELETE FROM fts_names WHERE rowid IN (
         SELECT id FROM symbols WHERE file_id=?1 AND unit_id IN (SELECT value FROM json_each(?2)))",
    )?
    .execute(params![file_id, ids])?;
    tx.prepare_cached(
        "DELETE FROM symbols WHERE file_id=?1 AND unit_id IN (SELECT value FROM json_each(?2))",
    )?
    .execute(params![file_id, ids])?;
    Ok(())
}
