//! Standalone vector mutation WAL rollback and reader backpressure; v1 §3.5.
use super::{
    tests::{insert_unit, table},
    *,
};
use crate::store::{self, StoreLimits, wal};
use std::error::Error as StdError;

const CAP: u64 = 8 * 1024 * 1024;

fn fixture(size: u32) -> Result<(tempfile::TempDir, Store, [u8; 32]), Box<dyn StdError>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("code.db");
    store::register_vector_extension()?;
    {
        let db = Connection::open(&path)?;
        db.pragma_update(None, "page_size", size)?;
        db.execute_batch("VACUUM")?;
    }
    let store = store::open(
        &path,
        StoreLimits {
            wal_max_bytes: CAP,
            ..Default::default()
        },
    )?;
    store.ensure_vector_table(table("retained", 8))?;
    store.ensure_vector_table(table("other", 8))?;
    let hash = insert_unit(&store, 1)?;
    let vector = VectorInsert {
        table_slug: "retained",
        unit_id: 1,
        ctx_hash: &[1; 32],
        embedding_hash: &hash,
        embedding: &[1.; 8],
    };
    store.insert_vector(vector)?;
    store.insert_vector(VectorInsert {
        table_slug: "other",
        ..vector
    })?;
    store
        .writer()?
        .execute_batch("CREATE TABLE mutation_events(id INTEGER PRIMARY KEY,payload BLOB)")?;
    wal::checkpoint(&store)?;
    Ok((dir, store, hash))
}

#[test]
fn vector_mutation_overflow_retains_rows_and_table_metadata() -> Result<(), Box<dyn StdError>> {
    for size in [4096, 65536] {
        let (dir, store, hash) = fixture(size)?;
        let mut reader = Connection::open(dir.path().join("code.db"))?;
        let read = reader.transaction()?;
        let old: Vec<u8> = read.query_row("SELECT v FROM vec_retained", [], |r| r.get(0))?;
        for operation in ["insert", "delete", "drop"] {
            let trigger = match operation {
                "insert" => {
                    "CREATE TRIGGER refuse_mutation AFTER UPDATE ON vec_retained BEGIN INSERT INTO mutation_events(payload) VALUES(zeroblob(16777216)); END;"
                }
                "delete" => {
                    "CREATE TRIGGER refuse_mutation AFTER DELETE ON vec_retained BEGIN INSERT INTO mutation_events(payload) VALUES(zeroblob(16777216)); END;"
                }
                _ => {
                    "CREATE TRIGGER refuse_mutation AFTER DELETE ON embedding_tables BEGIN INSERT INTO mutation_events(payload) VALUES(zeroblob(16777216)); END;"
                }
            };
            store.writer()?.execute_batch(trigger)?;
            let error = match operation {
                "insert" => store.insert_vector(VectorInsert {
                    table_slug: "retained",
                    unit_id: 1,
                    ctx_hash: &[2; 32],
                    embedding_hash: &hash,
                    embedding: &[-1.; 8],
                }),
                "delete" => store.delete_vector("retained", 1),
                _ => store.drop_vector_table("retained"),
            }
            .unwrap_err();
            assert!(matches!(error, Error::WalCapacity { .. }));
            let db = store.writer()?;
            assert_eq!(
                db.query_row("SELECT v FROM vec_retained", [], |r| r.get::<_, Vec<u8>>(0))?,
                old
            );
            assert_eq!(
                db.query_row("SELECT ctx_hash FROM vec_retained", [], |r| r
                    .get::<_, Vec<u8>>(0))?,
                vec![1; 32]
            );
            assert_eq!(
                db.query_row("SELECT count(*) FROM mutation_events", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            assert_eq!(
                db.query_row("SELECT count(*) FROM embedding_tables", [], |r| r
                    .get::<_, i64>(0))?,
                2
            );
            assert!(wal::wal_bytes(&db)? <= CAP);
            db.execute_batch("DROP TRIGGER refuse_mutation")?;
        }
        assert_eq!(
            read.query_row("SELECT v FROM vec_retained", [], |r| r.get::<_, Vec<u8>>(0))?,
            old
        );
        read.commit()?;
        store.insert_vector(VectorInsert {
            table_slug: "retained",
            unit_id: 1,
            ctx_hash: &[2; 32],
            embedding_hash: &hash,
            embedding: &[-1.; 8],
        })?;
        assert_eq!(
            store.scan_vectors("retained", &[-1.; 8], 1, VectorScanPredicate::default())?[0]
                .unit_id,
            1
        );
        store.delete_vector("retained", 1)?;
        store.drop_vector_table("retained")?;
        assert_eq!(store.count_vectors()?, 1);
        assert_eq!(store.vector_tables()?[0].slug, "other");
        let db = store.writer()?;
        assert_eq!(
            db.query_row("SELECT count(*) FROM units", [], |r| r.get::<_, i64>(0))?,
            1
        );
        assert_eq!(
            db.query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
            "ok"
        );
        assert!(wal::wal_bytes(&db)? <= CAP);
        println!(
            "Standalone vector WAL: page_size={size}, three mutations refused atomically, resume=ok, cap={CAP}"
        );
    }
    Ok(())
}

#[test]
fn vector_growth_backpressure_leaves_control_space_and_resumes_after_reader_release()
-> Result<(), Box<dyn StdError>> {
    for size in [4096, 65536] {
        let (dir, store, hash) = fixture(size)?;
        store.writer()?.execute_batch("CREATE TRIGGER grow_vector AFTER UPDATE ON vec_retained BEGIN INSERT INTO mutation_events(payload) VALUES(zeroblob(262144)); END;")?;
        wal::checkpoint(&store)?;
        let mut reader = Connection::open(dir.path().join("code.db"))?;
        let read = reader.transaction()?;
        assert_eq!(
            read.query_row("SELECT count(*) FROM mutation_events", [], |r| r
                .get::<_, i64>(0))?,
            0
        );
        let mut completed = 0;
        loop {
            let result = store.insert_vector(VectorInsert {
                table_slug: "retained",
                unit_id: 1,
                ctx_hash: &[2; 32],
                embedding_hash: &hash,
                embedding: &[-1.; 8],
            });
            match result {
                Ok(()) => completed += 1,
                Err(Error::WalCapacity { .. }) => break,
                Err(error) => return Err(error.into()),
            }
            assert!(completed < 100, "writer did not stop under held reader");
            assert!(wal::wal_bytes(&*store.writer()?)? <= CAP);
        }
        assert!(completed > 0);
        assert_eq!(
            store
                .writer()?
                .query_row("SELECT count(*) FROM mutation_events", [], |r| r
                    .get::<_, i64>(0))?,
            completed
        );
        // A second writer sees the shared allocation policy through queue
        // status and can use control headroom after vector growth stops.
        let second = store::open(
            dir.path().join("code.db"),
            StoreLimits {
                wal_max_bytes: CAP,
                ..Default::default()
            },
        )?;
        let snapshot = serde_json::to_value(crate::queue::status::snapshot(&second, 0)?)?;
        assert_eq!(snapshot["wal"]["growth_paused"], true);
        assert_eq!(snapshot["wal"]["control_paused"], false);
        second.delete_vector("retained", 1)?;
        assert_eq!(
            read.query_row("SELECT count(*) FROM vec_retained", [], |r| r
                .get::<_, i64>(0))?,
            1
        );
        assert!(wal::checkpoint(&store)?.busy);
        read.commit()?;
        store.insert_vector(VectorInsert {
            table_slug: "retained",
            unit_id: 1,
            ctx_hash: &[3; 32],
            embedding_hash: &hash,
            embedding: &[1.; 8],
        })?;
        assert_eq!(store.count_vectors()?, 2);
        assert!(wal::wal_bytes(&*store.writer()?)? <= CAP);
        println!(
            "Standalone vector backpressure: page_size={size}, committed={completed}, control delete=ok, resumed=ok, cap={CAP}"
        );
    }
    Ok(())
}

#[test]
fn table_creation_commit_overflow_does_not_register_speculative_dimensions()
-> Result<(), Box<dyn StdError>> {
    for size in [4096, 65536] {
        let (dir, store, hash) = fixture(size)?;
        store.writer()?.execute_batch("CREATE TRIGGER expand_metadata AFTER INSERT ON embedding_tables
            WHEN NEW.slug='candidate' BEGIN INSERT INTO mutation_events(payload) VALUES(zeroblob(4194304)); END;")?;
        wal::checkpoint(&store)?;
        let mut reader = Connection::open(dir.path().join("code.db"))?;
        let read = reader.transaction()?;
        assert_eq!(
            read.query_row("SELECT count(*) FROM embedding_tables", [], |r| r
                .get::<_, i64>(0))?,
            2
        );
        let error = store
            .ensure_vector_table(table("candidate", 8))
            .unwrap_err();
        assert!(matches!(error, Error::WalCapacity { .. }));
        let db = store.writer()?;
        assert_eq!(
            db.query_row(
                "SELECT count(*) FROM sqlite_schema WHERE name='vec_candidate'",
                [],
                |r| r.get::<_, i64>(0)
            )?,
            0
        );
        assert_eq!(
            db.query_row("SELECT count(*) FROM embedding_tables", [], |r| r
                .get::<_, i64>(0))?,
            2
        );
        assert_eq!(
            db.query_row("SELECT count(*) FROM mutation_events", [], |r| r
                .get::<_, i64>(0))?,
            0
        );
        assert!(wal::wal_bytes(&db)? <= CAP);
        db.execute_batch("DROP TRIGGER expand_metadata")?;
        drop(db);
        read.commit()?;
        // If vector_init ran speculatively before the failed commit, the writer
        // would still remember eight dimensions and reject this valid retry.
        store.ensure_vector_table(table("candidate", 16))?;
        store.insert_vector(VectorInsert {
            table_slug: "candidate",
            unit_id: 1,
            ctx_hash: &[4; 32],
            embedding_hash: &hash,
            embedding: &[1.; 16],
        })?;
        for _ in 0..8 {
            assert_eq!(
                store.scan_vectors("candidate", &[1.; 16], 1, VectorScanPredicate::default())?[0]
                    .unit_id,
                1
            );
        }
        assert_eq!(store.count_vectors()?, 3);
        drop(store);
        let reopened = store::open(
            dir.path().join("code.db"),
            StoreLimits {
                wal_max_bytes: CAP,
                ..Default::default()
            },
        )?;
        assert_eq!(
            reopened
                .vector_tables()?
                .iter()
                .find(|t| t.slug == "candidate")
                .unwrap()
                .dims,
            16
        );
        assert_eq!(
            reopened.scan_vectors("candidate", &[1.; 16], 1, VectorScanPredicate::default())?[0]
                .unit_id,
            1
        );
        assert_eq!(
            reopened
                .writer()?
                .query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
            "ok"
        );
        println!(
            "Table creation WAL: page_size={size}, failed_dims=8, retried_dims=16, retained_vectors=3, cap={CAP}"
        );
    }
    Ok(())
}

#[test]
fn table_reactivation_overflow_preserves_metadata_and_invalid_dimensions_never_commit()
-> Result<(), Box<dyn StdError>> {
    for size in [4096, 65536] {
        let (_dir, store, _hash) = fixture(size)?;
        for dims in [0, i32::MAX as usize + 1, usize::MAX] {
            assert!(matches!(
                store.ensure_vector_table(table("invalid", dims)),
                Err(Error::InvalidVectorDimensions { .. })
            ));
        }
        assert_eq!(store.vector_tables()?.len(), 2);
        store.writer()?.execute_batch(
            "CREATE TRIGGER expand_reactivation AFTER UPDATE ON embedding_tables
            BEGIN INSERT INTO mutation_events(payload) VALUES(zeroblob(4194304)); END;",
        )?;
        wal::checkpoint(&store)?;
        let inactive = VectorTableSpec {
            active: false,
            ..table("retained", 8)
        };
        assert!(matches!(
            store.ensure_vector_table(inactive),
            Err(Error::WalCapacity { .. })
        ));
        assert!(
            store
                .vector_tables()?
                .iter()
                .find(|t| t.slug == "retained")
                .unwrap()
                .active
        );
        assert_eq!(store.count_vectors()?, 2);
        store
            .writer()?
            .execute_batch("DROP TRIGGER expand_reactivation")?;
        store.ensure_vector_table(inactive)?;
        let before = store.vector_tables()?;
        assert!(!before.iter().find(|t| t.slug == "retained").unwrap().active);
        assert!(matches!(
            store.ensure_vector_table(table("retained", 16)),
            Err(Error::VectorTableMismatch { .. })
        ));
        assert_eq!(store.vector_tables()?, before);
        assert!(wal::wal_bytes(&*store.writer()?)? <= CAP);
    }
    Ok(())
}

#[test]
fn native_table_limit_is_checked_before_durable_creation() -> Result<(), Box<dyn StdError>> {
    let store = store::open(":memory:", StoreLimits::default())?;
    for id in 0..MAX_VECTOR_TABLES {
        store.ensure_vector_table(table(&format!("model_{id}"), 8))?;
    }
    assert!(matches!(
        store.ensure_vector_table(table("overflow", 8)),
        Err(Error::VectorTableCapacity { max_tables: 128 })
    ));
    assert_eq!(store.vector_tables()?.len(), 128);
    assert_eq!(
        store.writer()?.query_row(
            "SELECT count(*) FROM sqlite_schema WHERE name='vec_overflow'",
            [],
            |r| r.get::<_, i64>(0)
        )?,
        0
    );
    store.ensure_vector_table(VectorTableSpec {
        active: false,
        ..table("model_0", 8)
    })?;
    Ok(())
}
