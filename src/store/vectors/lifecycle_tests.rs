//! Completed-transition retirement and native slot reuse; v1 §7.3/§16.
use super::{
    tests::{insert_unit, table},
    *,
};
use crate::store::{self, StoreLimits};
use std::error::Error as StdError;

#[test]
fn retirement_preserves_coverage_and_rolls_back_failed_metadata_deletion()
-> Result<(), Box<dyn StdError>> {
    let store = store::open(":memory:", StoreLimits::default())?;
    store.ensure_vector_table(table("old", 2))?;
    store.ensure_vector_table(table("new", 2))?;
    let hash = insert_unit(&store, 1)?;
    store.insert_vector(VectorInsert {
        table_slug: "old",
        unit_id: 1,
        ctx_hash: &[1; 32],
        embedding_hash: &hash,
        embedding: &[1.0, 0.0],
    })?;
    assert!(!store.retire_empty_vector_table("old")?);
    assert_eq!(store.count_vectors()?, 1);
    store.replace_vector(VectorInsert {
        table_slug: "new",
        unit_id: 1,
        ctx_hash: &[1; 32],
        embedding_hash: &hash,
        embedding: &[0.0, 1.0],
    })?;
    store.writer()?.execute_batch("CREATE TRIGGER refuse_retirement BEFORE DELETE ON embedding_tables WHEN OLD.slug='old' BEGIN SELECT RAISE(ABORT,'hold retirement'); END;")?;
    assert!(store.retire_empty_vector_table("old").is_err());
    assert_eq!(store.vector_tables()?.len(), 2);
    assert!(
        store
            .scan_vectors("old", &[1.0, 0.0], 1, VectorScanPredicate::default())?
            .is_empty()
    );
    store
        .writer()?
        .execute_batch("DROP TRIGGER refuse_retirement")?;
    assert!(store.retire_empty_vector_table("old")?);
    assert!(store.retire_empty_vector_table("old")?);
    assert_eq!(store.vector_tables()?.len(), 1);
    assert_eq!(
        store.scan_vectors("new", &[0.0, 1.0], 1, VectorScanPredicate::default())?[0].unit_id,
        1
    );
    Ok(())
}

#[test]
fn retirement_reuses_native_slots_on_writer_and_every_reader() -> Result<(), Box<dyn StdError>> {
    let file = tempfile::NamedTempFile::new()?;
    let store = store::open(file.path(), StoreLimits::default())?;
    store.ensure_vector_table(table("retained", 2))?;
    let hash = insert_unit(&store, 1)?;
    store.insert_vector(VectorInsert {
        table_slug: "retained",
        unit_id: 1,
        ctx_hash: &[1; 32],
        embedding_hash: &hash,
        embedding: &[1.0, 0.0],
    })?;
    for i in 0..256 {
        let slug = format!("transition_{i}");
        store.ensure_vector_table(table(&slug, 2))?;
        assert!(store.retire_empty_vector_table(&slug)?);
        assert_eq!(
            store.scan_vectors("retained", &[1.0, 0.0], 1, VectorScanPredicate::default())?[0]
                .unit_id,
            1
        );
    }
    // Explicit drop uses the same lifecycle cleanup, including same-name/new-dims.
    store.ensure_vector_table(table("reused", 2))?;
    store.drop_vector_table("reused")?;
    store.ensure_vector_table(table("reused", 4))?;
    assert!(
        store
            .scan_vectors(
                "reused",
                &[1.0, 0.0, 0.0, 0.0],
                1,
                VectorScanPredicate::default()
            )?
            .is_empty()
    );
    for reader in &store.readers {
        assert_eq!(
            reader.lock().map_err(|_| "reader poisoned")?.query_row(
                "SELECT dkc_vector_prune()",
                [],
                |r| r.get::<_, i64>(0)
            )?,
            0
        );
    }
    assert_eq!(store.vector_tables()?.len(), 2);
    assert_eq!(store.count_vectors()?, 1);
    Ok(())
}

#[test]
fn retirement_late_policy_fence_rolls_back_drop() -> Result<(), Box<dyn StdError>> {
    let store = store::open(":memory:", StoreLimits::default())?;
    store.ensure_vector_table(table("obsolete", 2))?;
    let mut calls = 0;
    let result = store.retire_empty_vector_table_if("obsolete", |_| {
        calls += 1;
        if calls == 2 {
            Err(Error::VectorTableMissing {
                slug: "changed-policy".into(),
            })
        } else {
            Ok(())
        }
    });
    assert!(result.is_err());
    assert_eq!(calls, 2);
    assert_eq!(store.vector_tables()?.len(), 1);
    assert!(
        store
            .scan_vectors("obsolete", &[1.0, 0.0], 1, VectorScanPredicate::default())?
            .is_empty()
    );
    Ok(())
}
