//! Explicit capacity eviction, retaining L1 and control state; v1.md §3.5.
use super::{Error, Store, admission, vectors::vector_table_name, wal};
use rusqlite::{Connection, OptionalExtension, params};

/// Physical and logical outcomes remain distinct when readers pin an old file.
#[derive(Debug, serde::Serialize)]
pub(crate) struct ShrinkReport {
    pub(crate) state: &'static str,
    pub(crate) reason: &'static str,
    pub(crate) target_bytes: u64,
    pub(crate) before_bytes: u64,
    pub(crate) allocated_bytes: u64,
    pub(crate) file_bytes: u64,
    pub(crate) summaries_removed: u64,
    pub(crate) page_sections_removed: u64,
    pub(crate) vectors_removed: u64,
    pub(crate) checkpoint_busy: bool,
}

struct Window {
    generation: i64,
    upper: i64,
    l3_units: u64,
    tables: Vec<String>,
}

fn generation(db: &Connection) -> Result<i64, Error> {
    Ok(db.query_row("SELECT coalesce((SELECT CAST(value AS INTEGER) FROM meta WHERE key='l1_scan_generation'),0)", [], |r| r.get(0))?)
}
fn check_generation(db: &Connection, window: &Window) -> Result<(), Error> {
    if generation(db)? != window.generation {
        return Err(Error::L1ScanSuperseded {
            generation: window.generation,
        });
    }
    Ok(())
}
fn table_names(db: &Connection) -> Result<Vec<String>, Error> {
    Ok(db
        .prepare("SELECT slug FROM embedding_tables ORDER BY slug")?
        .query_map([], |r| r.get(0))?
        .collect::<Result<_, _>>()?)
}
fn sizes(db: &Connection) -> Result<(u64, u64, u64), Error> {
    let page_size: u64 = u64::from(db.query_row("PRAGMA page_size", [], |r| r.get::<_, u32>(0))?);
    let pages: u64 = u64::from(db.query_row("PRAGMA page_count", [], |r| r.get::<_, u32>(0))?);
    let free: u64 = u64::from(db.query_row("PRAGMA freelist_count", [], |r| r.get::<_, u32>(0))?);
    let allocated = page_size
        .checked_mul(pages)
        .ok_or(Error::DatabaseGrowthLimit)?;
    let physical = match db.path().filter(|p| !p.is_empty()) {
        Some(path) => std::fs::metadata(path)
            .map_err(|source| Error::Io { source })?
            .len(),
        None => allocated,
    };
    Ok((allocated, physical, page_size * pages.saturating_sub(free)))
}
fn count_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<u64> {
    let value: i64 = row.get(0)?;
    u64::try_from(value).map_err(|_| rusqlite::Error::IntegralValueOutOfRange(0, value))
}
fn capture(store: &Store) -> Result<Window, Error> {
    store.read_snapshot(|tx| {
        Ok(Window {
            generation: generation(tx)?,
            upper: tx.query_row("SELECT coalesce(max(id),0) FROM units", [], |r| r.get(0))?,
            l3_units: tx.query_row(
                "SELECT count(*) FROM units WHERE kind='summary' OR (kind='doc_section' AND ladder_level=3)",
                [],
                count_row,
            )?,
            tables: table_names(tx)?,
        })
    })?
}

impl Store {
    /// Evicts only for explicit --shrink, in finite admitted passes. L3 summaries
    /// go first, then lowest-rank L2 occurrences across registered model tables.
    /// No lease, usage ledger, source occurrence or history record is discarded.
    pub(crate) fn shrink(&self) -> Result<ShrinkReport, Error> {
        let window = capture(self)?;
        let (allocated, physical, _) = sizes(&*self.writer()?)?;
        let mut report = ShrinkReport {
            state: "cannot_reach_target",
            reason: "remaining allocation cannot reach the target in this finite pass",
            target_bytes: (u128::from(self.db_max_bytes) * 70 / 100) as u64,
            before_bytes: allocated.max(physical),
            allocated_bytes: allocated,
            file_bytes: physical,
            summaries_removed: 0,
            page_sections_removed: 0,
            vectors_removed: 0,
            checkpoint_busy: false,
        };
        match self.shrink_pass(&window, &mut report) {
            Ok(()) => {}
            Err(Error::WalCapacity { .. }) => {
                report.state = "paused";
                report.reason = "WAL capacity or held reader prevented further reclamation";
            }
            Err(Error::L1ScanSuperseded { .. }) => {
                report.state = "paused";
                report.reason = "catalogue changed during shrink; run again";
            }
            Err(Error::Sqlite(error))
                if error.sqlite_error_code() == Some(rusqlite::ErrorCode::DiskFull) =>
            {
                report.state = "paused";
                report.reason = "SQLite main-file capacity prevented further reclamation";
            }
            Err(error) => return Err(error),
        }
        let (allocated, physical, _) = sizes(&*self.writer()?)?;
        report.allocated_bytes = allocated;
        report.file_bytes = physical;
        if allocated.max(physical) <= report.target_bytes {
            report.state = "target_reached";
            report.reason = "footprint is at or below 70 percent";
        }
        Ok(report)
    }
    fn shrink_pass(&self, window: &Window, report: &mut ShrinkReport) -> Result<(), Error> {
        if self.reclaim_for_shrink(report)? {
            return Ok(());
        }
        // The initial count prevents another producer from extending this loop.
        for count in 0..window.l3_units {
            let removed = admission::transaction(self, admission::Kind::Control, |tx| {
                check_generation(&tx, window)?;
                // Only leaves are removed: a summary may not cascade-delete L1.
                let id:Option<(i64,bool)>=tx.query_row("SELECT u.id,u.kind='doc_section' FROM units u WHERE u.id<=?1 AND (u.kind='summary' OR (u.kind='doc_section' AND u.ladder_level=3)) AND NOT EXISTS(SELECT 1 FROM units child WHERE child.parent_unit=u.id) ORDER BY u.id LIMIT 1",[window.upper],|r|Ok((r.get(0)?,r.get(1)?))).optional()?;
                let mut vectors = 0_u64;
                let removed = match id {
                    Some((id, _)) => {
                        for slug in table_names(&tx)? {
                            let table = vector_table_name(&slug)?;
                            vectors += tx.query_row(
                                &format!("SELECT count(*) FROM {table} WHERE unit_id=?1"),
                                [id],
                                count_row,
                            )?;
                        }
                        tx.execute("DELETE FROM units WHERE id=?1 AND (kind='summary' OR (kind='doc_section' AND ladder_level=3))", [id])?
                    }
                    None => 0,
                };
                tx.commit()?;
                Ok::<_, Error>((removed, vectors, id.is_some_and(|(_, page)| page)))
            })?;
            if removed.0 == 0 {
                break;
            }
            if removed.2 {
                report.page_sections_removed += removed.0 as u64;
            } else {
                report.summaries_removed += removed.0 as u64;
            }
            report.vectors_removed += removed.1;
            if (count % 32 == 31 || self.reclaimable_target(report.target_bytes)?)
                && self.reclaim_for_shrink(report)?
            {
                return Ok(());
            }
        }
        if self.reclaim_for_shrink(report)? {
            return Ok(());
        }
        let mut after = (-f64::MAX, 0_i64);
        loop {
            let candidates = self.shrink_candidates(window, after)?;
            if candidates.is_empty() {
                break;
            }
            for (id, rank) in candidates {
                after = (rank, id);
                for slug in &window.tables {
                    report.vectors_removed += self.shrink_vector(window, slug, id)? as u64;
                }
                if self.reclaimable_target(report.target_bytes)?
                    && self.reclaim_for_shrink(report)?
                {
                    return Ok(());
                }
            }
            if self.reclaim_for_shrink(report)? {
                return Ok(());
            }
        }
        self.reclaim_for_shrink(report)?;
        Ok(())
    }
    fn reclaimable_target(&self, target: u64) -> Result<bool, Error> {
        Ok(sizes(&*self.writer()?)?.2 <= target)
    }
    fn reclaim_for_shrink(&self, report: &mut ShrinkReport) -> Result<bool, Error> {
        self.maintain()?;
        let db = self.writer()?;
        if db.path().is_some_and(|p| !p.is_empty()) {
            report.checkpoint_busy = wal::checkpoint_in(&db)?.busy;
        }
        let (allocated, physical, _) = sizes(&db)?;
        if report.checkpoint_busy
            && allocated <= report.target_bytes
            && physical > report.target_bytes
        {
            report.state = "paused";
            report.reason = "held reader prevents physical reclamation";
            return Ok(true);
        }
        Ok(allocated.max(physical) <= report.target_bytes)
    }
    fn shrink_candidates(
        &self,
        window: &Window,
        after: (f64, i64),
    ) -> Result<Vec<(i64, f64)>, Error> {
        self.read_snapshot(|tx|{
            check_generation(tx,window)?;
            let current=table_names(tx)?;
            let selects=window.tables.iter().filter(|slug|current.contains(slug))
                .map(|slug|vector_table_name(slug).map(|table|format!("SELECT unit_id FROM {table}")))
                .collect::<Result<Vec<_>,_>>()?;
            if selects.is_empty(){return Ok(Vec::new());}
            let sql=format!("WITH represented AS ({}), ranked AS (
                SELECT u.id,coalesce((SELECT max(pagerank) FROM symbols WHERE unit_id=u.id),
                (SELECT rank FROM files WHERE id=u.file_id),
                (SELECT max(f.rank) FROM unit_sources us JOIN files f ON us.source_kind='file' AND f.id=us.source_id WHERE us.unit_id=u.id),
                (SELECT max(f.rank) FROM commits c JOIN json_each(c.files_touched) j JOIN files f ON f.path=j.value
                 WHERE c.unit_id=u.id OR c.id IN (SELECT source_id FROM unit_sources WHERE unit_id=u.id AND source_kind='commit')
                 OR c.pr_id IN (SELECT id FROM prs WHERE unit_id=u.id)),0.0) AS rank
                FROM represented v JOIN units u ON u.id=v.unit_id WHERE u.id<=?1 AND u.kind!='summary')
                SELECT id,rank FROM ranked WHERE rank>?2 OR (rank=?2 AND id>?3) ORDER BY rank,id LIMIT 128",selects.join(" UNION "));
            Ok(tx.prepare(&sql)?.query_map(params![window.upper,after.0,after.1],|r|Ok((r.get(0)?,r.get(1)?)))?.collect::<Result<_,_>>()?)
        })?
    }
    fn shrink_vector(&self, window: &Window, slug: &str, id: i64) -> Result<usize, Error> {
        let table = vector_table_name(slug)?;
        admission::transaction(self, admission::Kind::Control, |tx| {
            check_generation(&tx, window)?;
            let tables = table_names(&tx)?;
            if !tables.iter().any(|value| value == slug) {
                return Ok(0);
            }
            let removed = tx.execute(&format!("DELETE FROM {table} WHERE unit_id=?1"), [id])?;
            if removed > 0 {
                tx.execute(
                    "DELETE FROM embedding_publications WHERE unit_id=?1 AND table_slug=?2",
                    params![id, slug],
                )?;
                let mut represented = false;
                for slug in tables {
                    let table = vector_table_name(&slug)?;
                    represented |= tx.query_row(
                        &format!("SELECT EXISTS(SELECT 1 FROM {table} WHERE unit_id=?1)"),
                        [id],
                        |r| r.get::<_, bool>(0),
                    )?;
                }
                if !represented {
                    tx.execute(
                        "UPDATE units SET ladder_level=1 WHERE id=?1 AND ladder_level=2",
                        [id],
                    )?;
                }
            }
            tx.commit()?;
            Ok(removed)
        })
    }
}

#[cfg(test)]
mod tests;
