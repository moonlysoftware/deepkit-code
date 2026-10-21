//! Explicit compaction rollback, reader snapshots and cap preservation; v1 §3.5.
use super::*;
use crate::store::vectors::{VectorInsert, VectorScanPredicate, VectorTableSpec};
use crate::store::{self, StoreLimits};
use std::error::Error as StdError;

fn settings(store: &Store) -> Result<(i64, i64, i64, i64), Error> {
    let db = store.writer()?;
    Ok((
        native_limit(&db)?,
        db.query_row("PRAGMA temp_store", [], |r| r.get(0))?,
        db.query_row("PRAGMA cache_spill", [], |r| r.get(0))?,
        db.query_row("PRAGMA busy_timeout", [], |r| r.get(0))?,
    ))
}

#[test]
fn compact_overflow_rolls_back_and_retries_without_losing_fts_or_vectors()
-> Result<(), Box<dyn StdError>> {
    for size in [4096, 65536] {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("code.db");
        store::register_vector_extension()?;
        let observer = Connection::open(&path)?;
        observer.pragma_update(None, "page_size", size)?;
        observer.execute_batch("CREATE TABLE payload(id INTEGER PRIMARY KEY, value BLOB)")?;
        let initial = store::open(&path, StoreLimits::default())?;
        initial.writer()?.execute_batch(
            "INSERT INTO payload VALUES(7,zeroblob(8388608)),(99,zeroblob(8388608));
             DELETE FROM payload WHERE id=99;
             INSERT INTO unit_contents(hash,text,tokens) VALUES(zeroblob(32),'compact fixture',2);
             INSERT INTO units(id,hash,content_hash,source_key,kind,header,created_at,updated_at)
             VALUES(77,zeroblob(32),zeroblob(32),'compact:77','chunk','compact fixture',1,1);
             CREATE VIRTUAL TABLE compact_fts USING fts5(text,content='');
             INSERT INTO compact_fts(rowid,text) VALUES(77,'retained compaction snapshot');",
        )?;
        initial.ensure_vector_table(VectorTableSpec {
            slug: "compact_2",
            model: "fixture",
            profile: "fixture",
            dims: 2,
            created_at: 1,
            active: true,
        })?;
        initial.insert_vector(VectorInsert {
            table_slug: "compact_2",
            unit_id: 77,
            ctx_hash: &[1; 32],
            embedding_hash: &[0; 32],
            embedding: &[1.0, 0.0],
        })?;
        checkpoint(&initial)?;
        let original_bytes = fs::metadata(&path)?.len();
        drop(initial);
        let limits = StoreLimits {
            wal_max_bytes: 2 * 1024 * 1024,
            db_max_bytes: 1,
            ..Default::default()
        };
        let capped = store::open(&path, limits)?;
        let before = settings(&capped)?;
        let mut reader = Connection::open(&path)?;
        let snapshot = reader.transaction()?;
        assert_eq!(
            snapshot.query_row("SELECT length(value) FROM payload WHERE id=7", [], |r| r
                .get::<_, i64>(0))?,
            8388608
        );
        let failure = capped.compact();
        assert!(
            matches!(failure, Err(Error::WalCapacity { .. })),
            "{failure:?}"
        );
        assert!(wal_bytes(&*capped.writer()?)? <= 1024 * 1024 * 3 / 2);
        assert_eq!(settings(&capped)?, before);
        assert_eq!(fs::metadata(&path)?.len(), original_bytes);
        assert_eq!(
            snapshot.query_row("SELECT length(value) FROM payload WHERE id=7", [], |r| r
                .get::<_, i64>(0))?,
            8388608
        );
        snapshot.commit()?;
        // A cap below existing allocation must allow shrinking, without eviction.
        drop(capped);
        let resumed = store::open(
            &path,
            StoreLimits {
                db_max_bytes: 1,
                ..Default::default()
            },
        )?;
        let before = settings(&resumed)?;
        resumed.compact()?;
        assert_eq!(settings(&resumed)?, before);
        checkpoint(&resumed)?;
        assert!(fs::metadata(&path)?.len() < original_bytes);
        assert_eq!(
            resumed
                .writer()?
                .query_row("SELECT id FROM payload", [], |r| r.get::<_, i64>(0))?,
            7
        );
        assert_eq!(
            resumed.writer()?.query_row(
                "SELECT rowid FROM compact_fts WHERE compact_fts MATCH 'retained'",
                [],
                |r| r.get::<_, i64>(0)
            )?,
            77
        );
        let hits =
            resumed.scan_vectors("compact_2", &[1.0, 0.0], 1, VectorScanPredicate::default())?;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].unit_id, 77);
        assert_eq!(hits[0].distance, 0.0);
        resumed.set_meta("indexed_at", "2")?;
        assert_eq!(
            resumed
                .writer()?
                .query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
            "ok"
        );
        println!(
            "compact page_size={size}: before={original_bytes}, after={}, refusal_wal_cap={}",
            fs::metadata(&path)?.len(),
            limits.wal_max_bytes
        );
    }
    Ok(())
}

#[test]
fn compact_reader_backpressure_preserves_snapshot_and_resumes_after_checkpoint()
-> Result<(), Box<dyn StdError>> {
    for size in [4096, 65536] {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("code.db");
        store::register_vector_extension()?;
        let mut reader = Connection::open(&path)?;
        reader.pragma_update(None, "page_size", size)?;
        reader.execute_batch("CREATE TABLE payload(id INTEGER PRIMARY KEY,value BLOB); INSERT INTO payload VALUES(7,'original')")?;
        let store = store::open(
            &path,
            StoreLimits {
                wal_max_bytes: 8 * 1024 * 1024,
                ..Default::default()
            },
        )?;
        checkpoint(&store)?;
        let snapshot = reader.transaction()?;
        assert_eq!(
            snapshot.query_row("SELECT value FROM payload", [], |r| r.get::<_, String>(0))?,
            "original"
        );
        // Simulate another connection consuming the available WAL before VACUUM.
        store.writer()?.execute_batch(
            "UPDATE payload SET value=zeroblob(8388608); UPDATE payload SET value='latest'",
        )?;
        let physical = wal_bytes(&*store.writer()?)?;
        let before = settings(&store)?;
        assert!(checkpoint(&store)?.busy);
        assert!(matches!(store.compact(), Err(Error::WalCapacity { .. })));
        assert_eq!(settings(&store)?, before);
        assert_eq!(wal_bytes(&*store.writer()?)?, physical);
        assert_eq!(
            snapshot.query_row("SELECT value FROM payload", [], |r| r.get::<_, String>(0))?,
            "original"
        );
        snapshot.commit()?;
        store.compact()?;
        assert!(wal_bytes(&*store.writer()?)? <= 7 * 1024 * 1024);
        assert_eq!(
            store
                .writer()?
                .query_row("SELECT value FROM payload", [], |r| r.get::<_, String>(0))?,
            "latest"
        );
        assert_eq!(settings(&store)?, before);
    }
    let memory = store::open(":memory:", StoreLimits::default())?;
    memory.set_meta("compact_memory", "retained")?;
    memory.compact()?;
    assert_eq!(
        memory.get_meta("compact_memory")?.as_deref(),
        Some("retained")
    );
    Ok(())
}

#[test]
fn compact_below_free_page_threshold_skips_rewrite_even_without_wal_room()
-> Result<(), Box<dyn StdError>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("code.db");
    let initial = store::open(&path, StoreLimits::default())?;
    initial.set_meta("retained", "no rewrite")?;
    checkpoint(&initial)?;
    drop(initial);
    let store = store::open(
        &path,
        StoreLimits {
            wal_max_bytes: 0,
            ..Default::default()
        },
    )?;
    let schema = || -> Result<i64, Error> {
        Ok(store
            .writer()?
            .query_row("PRAGMA schema_version", [], |r| r.get(0))?)
    };
    let before = schema()?;
    let size = fs::metadata(&path)?.len();
    store.compact()?;
    assert_eq!(schema()?, before, "VACUUM increments the schema cookie");
    assert_eq!(fs::metadata(&path)?.len(), size);
    assert_eq!(wal_bytes(&*store.writer()?)?, 0);
    assert_eq!(store.get_meta("retained")?.as_deref(), Some("no rewrite"));
    Ok(())
}
