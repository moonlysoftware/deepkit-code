//! End-of-index incremental reclamation and planner maintenance; v1 §3.5.
use super::{Error, Store, admission, wal};

/// Committed page reclamation, distinct from physical truncation held by readers.
#[derive(Debug, serde::Serialize)]
pub(crate) struct Maintenance {
    pub(crate) pages_reclaimed: u64,
    pub(crate) optimization_ran: bool,
}

impl Store {
    /// Reclaims initially free pages in bounded transactions, then lets SQLite
    /// update planner statistics. Concurrent deletion cannot extend this pass
    /// indefinitely; a refused batch preserves all previously committed progress.
    pub(crate) fn maintain(&self) -> Result<super::Maintenance, Error> {
        let target = {
            let db = self.writer()?;
            let mode: i64 = db.query_row("PRAGMA auto_vacuum", [], |r| r.get(0))?;
            if mode == 2 {
                u64::from(db.query_row("PRAGMA freelist_count", [], |r| r.get::<_, u32>(0))?)
            } else {
                // Preserve legacy layouts; changing them requires a full rewrite.
                0
            }
        };
        let mut pages_reclaimed = 0;
        let mut batch_pages = 64_u64;
        while pages_reclaimed < target {
            let batch = batch_pages.min(target - pages_reclaimed);
            let result = admission::transaction(self, admission::Kind::Catalogue, |tx| {
                let before: u32 = tx.query_row("PRAGMA freelist_count", [], |r| r.get(0))?;
                tx.execute_batch(&format!("PRAGMA incremental_vacuum({batch})"))?;
                let after: u32 = tx.query_row("PRAGMA freelist_count", [], |r| r.get(0))?;
                tx.commit()?;
                Ok::<_, Error>(u64::from(before.saturating_sub(after)))
            });
            match result {
                Ok(0) => break,
                Ok(reclaimed) => pages_reclaimed += reclaimed,
                Err(Error::WalCapacity { .. }) if batch_pages > 1 => {
                    // Only the rolled-back transaction is retried with less work.
                    // Preflight refusal also terminates after at most six halvings.
                    batch_pages /= 2;
                }
                Err(error) => return Err(error),
            }
        }
        admission::transaction(self, admission::Kind::Catalogue, |tx| {
            tx.execute_batch("PRAGMA optimize")?;
            tx.commit()?;
            Ok::<_, Error>(())
        })?;
        if pages_reclaimed > 0 {
            wal::checkpoint(self)?;
        }
        Ok(Maintenance {
            pages_reclaimed,
            optimization_ran: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{self, StoreLimits};
    use rusqlite::Connection;
    use std::error::Error as StdError;

    #[test]
    fn maintenance_reclaims_in_batches_and_updates_planner_statistics()
    -> Result<(), Box<dyn StdError>> {
        for size in [4096, 65536] {
            let dir = tempfile::tempdir()?;
            let path = dir.path().join("code.db");
            store::register_vector_extension()?;
            let db = Connection::open(&path)?;
            db.pragma_update(None, "page_size", size)?;
            db.execute_batch("PRAGMA auto_vacuum=INCREMENTAL; CREATE TABLE payload(id INTEGER PRIMARY KEY,value BLOB); CREATE INDEX payload_value ON payload(value)")?;
            let store = store::open(&path, StoreLimits::default())?;
            store.writer()?.execute_batch("INSERT INTO payload VALUES(7,'retained'),(9,zeroblob(16777216)); DELETE FROM payload WHERE id=9; SELECT id FROM payload WHERE value='retained'")?;
            wal::checkpoint(&store)?;
            let before = store.footprint()?;
            assert!(before.freelist_count > 64);
            let maintenance = store.maintain()?;
            assert!(maintenance.optimization_ran);
            assert_eq!(maintenance.pages_reclaimed, before.freelist_count as u64);
            let after = store.footprint()?;
            assert_eq!(after.freelist_count, 0);
            assert!(after.bytes < before.bytes);
            assert_eq!(
                store
                    .writer()?
                    .query_row("SELECT id FROM payload", [], |r| r.get::<_, i64>(0))?,
                7
            );
            assert!(
                store.writer()?.query_row(
                    "SELECT count(*) FROM sqlite_stat1 WHERE idx='payload_value'",
                    [],
                    |r| r.get::<_, i64>(0)
                )? > 0
            );
            assert_eq!(store.maintain()?.pages_reclaimed, 0);
            assert_eq!(
                store
                    .writer()?
                    .query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
                "ok"
            );
        }
        Ok(())
    }

    #[test]
    fn reader_refusal_keeps_committed_reclamation_and_resumes() -> Result<(), Box<dyn StdError>> {
        for size in [4096, 65536] {
            let dir = tempfile::tempdir()?;
            let path = dir.path().join("code.db");
            store::register_vector_extension()?;
            let mut reader = Connection::open(&path)?;
            reader.pragma_update(None, "page_size", size)?;
            reader.execute_batch("PRAGMA auto_vacuum=INCREMENTAL; CREATE TABLE payload(id INTEGER PRIMARY KEY,value BLOB)")?;
            let initial = store::open(&path, StoreLimits::default())?;
            // Interleave live and deleted overflow pages so incremental vacuum
            // must move live pages, rather than merely truncate an empty tail.
            initial.writer()?.execute_batch("WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<512) INSERT INTO payload SELECT x,zeroblob(65536) FROM n; DELETE FROM payload WHERE id%2=0")?;
            wal::checkpoint(&initial)?;
            drop(initial);
            let store = store::open(
                &path,
                StoreLimits {
                    wal_max_bytes: 4 * 1024 * 1024,
                    ..Default::default()
                },
            )?;
            let snapshot = reader.transaction()?;
            assert_eq!(
                snapshot.query_row("SELECT count(*) FROM payload", [], |r| r.get::<_, i64>(0))?,
                256
            );
            let before = store.footprint()?.freelist_count;
            assert!(matches!(store.maintain(), Err(Error::WalCapacity { .. })));
            let after = store.footprint()?.freelist_count;
            assert!(
                after < before,
                "some batches should commit before backpressure"
            );
            assert!(after > 0);
            assert!(wal::wal_bytes(&*store.writer()?)? <= 3 * 1024 * 1024);
            assert_eq!(
                snapshot.query_row("SELECT count(*) FROM payload", [], |r| r.get::<_, i64>(0))?,
                256
            );
            snapshot.commit()?;
            store.maintain()?;
            assert_eq!(store.footprint()?.freelist_count, 0);
            assert_eq!(
                store
                    .writer()?
                    .query_row("SELECT count(*) FROM payload", [], |r| r.get::<_, i64>(0))?,
                256
            );
        }
        Ok(())
    }
}
