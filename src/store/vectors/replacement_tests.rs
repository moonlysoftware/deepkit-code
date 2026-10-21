//! Physical-cap profile replacement and rollback; v1 §7.3/§16, T1.6.
use super::{
    tests::{insert_unit, table},
    *,
};
use crate::store::{self, StoreLimits, wal};
use std::error::Error as StdError;

#[test]
fn composite_summary_replaces_parent_at_full_cap_and_rolls_back_rejection()
-> Result<(), Box<dyn StdError>> {
    for kind in ["pr", "module", "symbol"] {
        let store = store::open(
            ":memory:",
            StoreLimits {
                max_l2_units: 1,
                ..Default::default()
            },
        )?;
        store.ensure_vector_table(table("single", 2))?;
        let parent = insert_unit(&store, 1)?;
        let summary = insert_unit(&store, 2)?;
        {
            let db = store.writer()?;
            db.execute("UPDATE units SET kind=?1 WHERE id=1", [kind])?;
            if kind == "pr" {
                db.execute("INSERT INTO prs(id,number,title,merge_sha,strategy,commit_count,merged_at,unit_id,summary_unit) VALUES(1,1,'PR',randomblob(20),'squash',1,0,1,2)",[])?;
            } else if kind == "module" {
                db.execute(
                    "INSERT INTO dirs(path,merkle,file_count,module_unit,summary_unit) VALUES('',randomblob(32),1,1,2)",
                    [],
                )?;
            }
            db.execute(
                "UPDATE units SET kind='summary',parent_unit=1,source_key=?1 WHERE id=2",
                [format!("summary:{kind}:1")],
            )?;
        }
        let raw = VectorInsert {
            table_slug: "single",
            unit_id: 1,
            ctx_hash: &[1; 32],
            embedding_hash: &parent,
            embedding: &[1.0, 0.0],
        };
        store.insert_vector(raw)?;
        store.writer()?.execute("INSERT INTO embedding_publications(unit_id,table_slug,desired_hash) VALUES(1,'single',?1)",[[3u8;32].as_slice()])?;
        let output = VectorInsert {
            unit_id: 2,
            embedding_hash: &summary,
            ..raw
        };
        assert!(
            store
                .replace_vector(VectorInsert {
                    embedding: &[0.0, 0.0],
                    ..output
                })
                .is_err()
        );
        assert_eq!(
            store.writer()?.query_row(
                "SELECT count(*) FROM vec_single WHERE unit_id=1",
                [],
                |r| r.get::<_, i64>(0)
            )?,
            1
        );
        assert_eq!(
            store.writer()?.query_row(
                "SELECT count(*) FROM embedding_publications WHERE unit_id=1",
                [],
                |r| r.get::<_, i64>(0)
            )?,
            1
        );
        if kind == "symbol" {
            assert!(matches!(
                store.replace_vector(output),
                Err(Error::VectorCapReached { .. })
            ));
        } else {
            store.replace_vector(output)?;
            assert_eq!(
                store
                    .writer()?
                    .query_row("SELECT unit_id FROM vec_single", [], |r| r.get::<_, i64>(0))?,
                2
            );
            assert_eq!(
                store.writer()?.query_row(
                    "SELECT count(*) FROM embedding_publications WHERE unit_id=1",
                    [],
                    |r| r.get::<_, i64>(0)
                )?,
                0
            );
            assert_eq!(
                store
                    .writer()?
                    .query_row("SELECT count(*) FROM units WHERE id=1", [], |r| r
                        .get::<_, i64>(0))?,
                1
            );
        }
        assert_eq!(store.count_vectors()?, 1);
    }
    Ok(())
}

#[test]
fn replacement_at_cap_preserves_unrelated_rows_and_rejects_invalid_output()
-> Result<(), Box<dyn StdError>> {
    for size in [4096, 65536] {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("code.db");
        store::register_vector_extension()?;
        let mut reader = Connection::open(&path)?;
        reader.pragma_update(None, "page_size", size)?;
        reader.execute_batch("VACUUM")?;
        let store = store::open(
            &path,
            StoreLimits {
                max_l2_units: 2,
                ..Default::default()
            },
        )?;
        store.ensure_vector_table(table("old", 2))?;
        store.ensure_vector_table(table("new", 4))?;
        let hash = insert_unit(&store, 1)?;
        let other = insert_unit(&store, 2)?;
        let absent = insert_unit(&store, 3)?;
        let old = VectorInsert {
            table_slug: "old",
            unit_id: 1,
            ctx_hash: &[1; 32],
            embedding_hash: &hash,
            embedding: &[1.0, 0.0],
        };
        store.insert_vector(old)?;
        store.insert_vector(VectorInsert {
            unit_id: 2,
            embedding_hash: &other,
            ..old
        })?;
        wal::checkpoint(&store)?;
        let snapshot = reader.transaction()?;
        let old_bytes: Vec<u8> =
            snapshot.query_row("SELECT v FROM vec_old WHERE unit_id=1", [], |r| r.get(0))?;
        let new = VectorInsert {
            table_slug: "new",
            ctx_hash: &[2; 32],
            embedding: &[0.0, 1.0, 0.0, 0.0],
            ..old
        };
        assert!(
            store
                .replace_vector(VectorInsert {
                    embedding: &[0.0; 4],
                    ..new
                })
                .is_err()
        );
        assert!(
            store
                .replace_vector(VectorInsert {
                    embedding_hash: &[9; 32],
                    ..new
                })
                .is_err()
        );
        assert!(matches!(
            store.replace_vector(VectorInsert {
                unit_id: 3,
                embedding_hash: &absent,
                ..new
            }),
            Err(Error::VectorCapReached { .. })
        ));
        assert_eq!(store.count_vectors()?, 2);
        assert_eq!(
            store
                .writer()?
                .query_row("SELECT count(*) FROM vec_old", [], |r| r.get::<_, i64>(0))?,
            2
        );
        store.replace_vector(new)?;
        assert_eq!(store.count_vectors()?, 2);
        assert_eq!(
            store
                .writer()?
                .query_row("SELECT unit_id FROM vec_old", [], |r| r.get::<_, i64>(0))?,
            2
        );
        assert_eq!(
            store
                .writer()?
                .query_row("SELECT unit_id FROM vec_new", [], |r| r.get::<_, i64>(0))?,
            1
        );
        assert_eq!(
            snapshot.query_row("SELECT v FROM vec_old WHERE unit_id=1", [], |r| r
                .get::<_, Vec<u8>>(0))?,
            old_bytes
        );
        assert_eq!(
            snapshot.query_row("SELECT count(*) FROM vec_new", [], |r| r.get::<_, i64>(0))?,
            0
        );
        snapshot.commit()?;
        assert_eq!(
            store.scan_vectors(
                "new",
                &[0.0, 1.0, 0.0, 0.0],
                1,
                VectorScanPredicate::default()
            )?[0]
                .unit_id,
            1
        );
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
fn replacement_wal_overflow_rolls_back_old_deletion_and_retries() -> Result<(), Box<dyn StdError>> {
    for size in [4096, 65536] {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("code.db");
        store::register_vector_extension()?;
        let db = Connection::open(&path)?;
        db.pragma_update(None, "page_size", size)?;
        db.execute_batch("VACUUM")?;
        let store = store::open(
            &path,
            StoreLimits {
                max_l2_units: 1,
                wal_max_bytes: 8 * 1024 * 1024,
                ..Default::default()
            },
        )?;
        store.ensure_vector_table(table("old", 2))?;
        store.ensure_vector_table(table("new", 2))?;
        let hash = insert_unit(&store, 1)?;
        let vector = VectorInsert {
            table_slug: "old",
            unit_id: 1,
            ctx_hash: &[1; 32],
            embedding_hash: &hash,
            embedding: &[1.0, 0.0],
        };
        store.insert_vector(vector)?;
        store.writer()?.execute_batch("CREATE TABLE replacement_payload(value BLOB); CREATE TRIGGER overflow_replace AFTER INSERT ON vec_new BEGIN INSERT INTO replacement_payload VALUES(zeroblob(16777216)); END;")?;
        wal::checkpoint(&store)?;
        assert!(matches!(
            store.replace_vector(VectorInsert {
                table_slug: "new",
                ..vector
            }),
            Err(Error::WalCapacity { .. })
        ));
        assert!(wal::wal_bytes(&*store.writer()?)? <= 8 * 1024 * 1024);
        assert_eq!(
            store
                .writer()?
                .query_row("SELECT count(*) FROM vec_old", [], |r| r.get::<_, i64>(0))?,
            1
        );
        assert_eq!(
            store
                .writer()?
                .query_row("SELECT count(*) FROM vec_new", [], |r| r.get::<_, i64>(0))?,
            0
        );
        assert_eq!(
            store
                .writer()?
                .query_row("SELECT count(*) FROM replacement_payload", [], |r| r
                    .get::<_, i64>(0))?,
            0
        );
        wal::checkpoint(&store)?;
        store
            .writer()?
            .execute_batch("DROP TRIGGER overflow_replace")?;
        store.replace_vector(VectorInsert {
            table_slug: "new",
            ..vector
        })?;
        assert_eq!(store.count_vectors()?, 1);
        assert_eq!(
            store
                .writer()?
                .query_row("SELECT count(*) FROM vec_old", [], |r| r.get::<_, i64>(0))?,
            0
        );
    }
    Ok(())
}

#[test]
fn lowered_caps_permit_representation_replacement_but_no_new_coverage()
-> Result<(), Box<dyn StdError>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("code.db");
    let initial = store::open(&path, StoreLimits::default())?;
    for name in ["old", "duplicate", "new"] {
        initial.ensure_vector_table(table(name, 2))?;
    }
    let hash = insert_unit(&initial, 1)?;
    let second = insert_unit(&initial, 2)?;
    let vector = VectorInsert {
        table_slug: "old",
        unit_id: 1,
        ctx_hash: &[1; 32],
        embedding_hash: &hash,
        embedding: &[1.0, 0.0],
    };
    initial.insert_vector(vector)?;
    initial.insert_vector(VectorInsert {
        table_slug: "duplicate",
        ..vector
    })?;
    drop(initial);
    let store = store::open(
        &path,
        StoreLimits {
            max_l2_units: 0,
            ..Default::default()
        },
    )?;
    {
        let mut db = store.writer()?;
        let tx = db.transaction()?;
        let plan = plan_replacement(&store, &tx, "new", 1, 0)?;
        assert!(!plan.grows_bytes(2));
        assert!(plan.grows_bytes(8));
    }
    store.replace_vector(VectorInsert {
        table_slug: "new",
        ..vector
    })?;
    assert_eq!(store.count_vectors()?, 1);
    assert!(matches!(
        store.replace_vector(VectorInsert {
            table_slug: "new",
            unit_id: 2,
            embedding_hash: &second,
            ..vector
        }),
        Err(Error::VectorCapReached { .. })
    ));
    assert_eq!(store.count_vectors()?, 1);
    Ok(())
}
