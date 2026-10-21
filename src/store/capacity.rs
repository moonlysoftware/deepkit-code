//! SQLite main-file growth enforcement; see `v1.md` §3.5 and plan contracts §6.

use rusqlite::{Connection, Transaction};

use super::Error;

/// Bounded main-file space kept outside ordinary text/vector growth so catalogue
/// and completion/accounting writes can still publish lower-level outcomes.
pub(crate) fn growth_limit(cap: u64) -> u64 {
    cap - (cap / 4).min(1024 * 1024)
}

/// Installs the hard main-file page ceiling on this writer before bounded writes.
/// Returns the current logical allocation, including committed WAL pages. A cap
/// below existing allocation freezes further page growth without deleting data.
/// The caller must hold an immediate write transaction and repeat this on every
/// writer/updated configuration: max_page_count is connection-local, not a WAL
/// reservation. Control/catalogue headroom is a separate admission policy.
pub(crate) fn limit_main_growth(tx: &Transaction<'_>, cap: u64) -> Result<u64, Error> {
    limit_connection_growth(tx, cap)
}

/// Installs the page ceiling before initial layout writes. Outside startup,
/// callers must hold an immediate transaction to observe an authoritative floor.
pub(super) fn limit_connection_growth(tx: &Connection, cap: u64) -> Result<u64, Error> {
    let page_size =
        u64::from(tx.query_row("PRAGMA main.page_size", [], |row| row.get::<_, u32>(0))?);
    let pages = u64::from(tx.query_row("PRAGMA main.page_count", [], |row| row.get::<_, u32>(0))?);
    let allocated = pages
        .checked_mul(page_size)
        .ok_or(Error::DatabaseGrowthLimit)?;
    if page_size == 0 || (pages == 0 && cap < page_size) {
        return Err(Error::DatabaseGrowthLimit);
    }
    // SQLite cannot lower the ceiling below the current page count. Explicitly
    // preserve that floor, and never send zero (which only queries the setting).
    // SQLite's maximum supported page number is 2^32-2.
    let requested = (cap / page_size).max(pages).min(4_294_967_294);
    let actual = u64::from(tx.pragma_update_and_check(
        None,
        "max_page_count",
        requested as u32,
        |row| row.get::<_, u32>(0),
    )?);
    if actual > requested || actual < pages {
        return Err(Error::DatabaseGrowthLimit);
    }
    Ok(allocated)
}

/// Observed main-file allocation and configured cap; inspecting never writes or
/// checkpoints. Free pages can still be reused when further page growth is frozen.
#[derive(Debug, serde::Serialize)]
pub(crate) struct Status {
    configured_limit_bytes: u64,
    effective_limit_bytes: u64,
    growth_limit_bytes: u64,
    control_headroom_bytes: u64,
    unallocated_control_bytes: u64,
    allocated_bytes: u64,
    file_bytes: Option<u64>,
    free_pages: u32,
    page_growth_available_bytes: u64,
    over_limit: bool,
}

impl Status {
    /// Describes allocation without promising that a particular write will fit.
    pub(crate) fn summary(&self) -> String {
        format!(
            "Database: {} / {} allocated bytes; main file {}; {}",
            self.allocated_bytes,
            self.effective_limit_bytes,
            self.file_bytes.map_or_else(
                || "memory-only".to_owned(),
                |bytes| format!("{bytes} bytes")
            ),
            if self.over_limit {
                "over cap; page growth frozen, data retained"
            } else if self.page_growth_available_bytes == 0 {
                "page limit reached; reusable pages may remain"
            } else {
                "within cap"
            },
        )
    }
}

/// Reports logical committed pages (including WAL) separately from the main file.
pub(crate) fn status(db: &Connection, store: &super::Store) -> Result<Status, Error> {
    let page_size = u64::from(db.query_row("PRAGMA page_size", [], |r| r.get::<_, u32>(0))?);
    let allocated_bytes = u64::from(db.query_row("PRAGMA page_count", [], |r| r.get::<_, u32>(0))?)
        .checked_mul(page_size)
        .ok_or(Error::DatabaseGrowthLimit)?;
    if page_size == 0 {
        return Err(Error::DatabaseGrowthLimit);
    }
    let effective_limit_bytes = (store.db_max_bytes / page_size).min(4_294_967_294) * page_size;
    let growth_limit_bytes =
        (growth_limit(store.db_max_bytes) / page_size).min(4_294_967_294) * page_size;
    let control_headroom_bytes = effective_limit_bytes.saturating_sub(growth_limit_bytes);
    let file_bytes = db
        .path()
        .filter(|path| !path.is_empty())
        .map(|path| std::fs::metadata(path).map(|metadata| metadata.len()))
        .transpose()
        .map_err(|source| Error::Io { source })?;
    Ok(Status {
        configured_limit_bytes: store.db_max_bytes,
        effective_limit_bytes,
        growth_limit_bytes,
        control_headroom_bytes,
        unallocated_control_bytes: effective_limit_bytes
            .saturating_sub(allocated_bytes)
            .min(control_headroom_bytes),
        allocated_bytes,
        file_bytes,
        free_pages: db.query_row("PRAGMA freelist_count", [], |r| r.get(0))?,
        page_growth_available_bytes: effective_limit_bytes.saturating_sub(allocated_bytes),
        over_limit: allocated_bytes.max(file_bytes.unwrap_or(0)) > effective_limit_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::{Connection, ErrorCode, TransactionBehavior};
    use std::error::Error as StdError;

    #[test]
    fn page_ceiling_rolls_back_whole_write_and_can_be_raised_without_losing_snapshot()
    -> Result<(), Box<dyn StdError>> {
        for page_size in [4096_u32, 65536] {
            let directory = tempfile::tempdir()?;
            let path = directory.path().join("code.db");
            crate::store::register_vector_extension()?;
            let mut db = Connection::open(&path)?;
            db.pragma_update(None, "page_size", page_size)?;
            // The process-wide auto-extension may create metadata at open. VACUUM
            // makes the requested layout real even in the parallel full suite.
            db.execute_batch("VACUUM")?;
            db.pragma_update(None, "journal_mode", "WAL")?;
            db.execute_batch("CREATE TABLE payload(id INTEGER PRIMARY KEY, value BLOB); INSERT INTO payload VALUES(1,'committed')")?;
            let pages: u32 = db.query_row("PRAGMA page_count", [], |r| r.get(0))?;
            let cap = u64::from((pages + 2) * page_size + page_size - 1);
            {
                let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
                assert_eq!(limit_main_growth(&tx, cap)?, u64::from(pages * page_size));
                let maximum: u32 = tx.query_row("PRAGMA max_page_count", [], |r| r.get(0))?;
                assert_eq!(maximum, pages + 2);
                tx.execute("UPDATE payload SET value='uncommitted' WHERE id=1", [])?;
                let error = tx
                    .execute(
                        "INSERT INTO payload VALUES(2,zeroblob(?1))",
                        [page_size * 16],
                    )
                    .unwrap_err();
                assert_eq!(error.sqlite_error_code(), Some(ErrorCode::DiskFull));
            }
            assert_eq!(
                db.query_row("SELECT value FROM payload WHERE id=1", [], |r| r
                    .get::<_, String>(0))?,
                "committed"
            );
            assert_eq!(
                db.query_row("SELECT COUNT(*) FROM payload", [], |r| r.get::<_, i64>(0))?,
                1
            );
            db.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")?;
            assert!(
                std::fs::metadata(&path)?.len()
                    <= cap / u64::from(page_size) * u64::from(page_size)
            );
            {
                let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
                limit_main_growth(&tx, u64::from(page_size * 64))?;
                tx.execute(
                    "INSERT INTO payload VALUES(2,zeroblob(?1))",
                    [page_size * 16],
                )?;
                tx.commit()?;
            }
            db.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")?;
            let retained = std::fs::metadata(&path)?.len();
            {
                let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
                assert_eq!(limit_main_growth(&tx, 1)?, retained);
                let error = tx
                    .execute(
                        "INSERT INTO payload VALUES(3,zeroblob(?1))",
                        [page_size * 16],
                    )
                    .unwrap_err();
                assert_eq!(error.sqlite_error_code(), Some(ErrorCode::DiskFull));
            }
            db.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")?;
            assert_eq!(std::fs::metadata(&path)?.len(), retained);
            assert_eq!(
                db.query_row("SELECT COUNT(*) FROM payload", [], |r| r.get::<_, i64>(0))?,
                2
            );
            assert_eq!(
                db.query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
                "ok"
            );
        }
        Ok(())
    }
    #[test]
    fn each_writer_reinstalls_ceiling_against_current_committed_allocation()
    -> Result<(), Box<dyn StdError>> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("code.db");
        crate::store::register_vector_extension()?;
        let mut first = Connection::open(&path)?;
        first.pragma_update(None, "journal_mode", "WAL")?;
        first.execute_batch("CREATE TABLE payload(value BLOB)")?;
        let size: u32 = first.query_row("PRAGMA page_size", [], |r| r.get(0))?;
        let pages: u32 = first.query_row("PRAGMA page_count", [], |r| r.get(0))?;
        let cap = u64::from((pages + 10) * size);
        {
            let tx = first.transaction_with_behavior(TransactionBehavior::Immediate)?;
            limit_main_growth(&tx, cap)?;
            tx.execute("INSERT INTO payload VALUES(zeroblob(?1))", [size * 4])?;
            tx.commit()?;
        }
        let mut second = Connection::open(&path)?;
        let before: u32 = second.query_row("PRAGMA max_page_count", [], |r| r.get(0))?;
        assert!(u64::from(before) > cap / u64::from(size));
        {
            let tx = second.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let allocated = limit_main_growth(&tx, cap)?;
            assert!(allocated > u64::from(pages * size));
            let error = tx
                .execute("INSERT INTO payload VALUES(zeroblob(?1))", [size * 16])
                .unwrap_err();
            assert_eq!(error.sqlite_error_code(), Some(ErrorCode::DiskFull));
        }
        first.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")?;
        assert!(std::fs::metadata(&path)?.len() <= cap);
        assert_eq!(
            second.query_row("SELECT COUNT(*) FROM payload", [], |r| r.get::<_, i64>(0))?,
            1
        );
        Ok(())
    }
}
