//! Long-lived reader registration across committed table lifetimes; v1 §7.3/§8.2.
use super::{
    tests::{insert_unit, table},
    *,
};
use crate::store::{self, StoreLimits};
use std::{error::Error as StdError, process::Command};

#[test]
fn table_child() -> Result<(), Box<dyn StdError>> {
    let Some(path) = std::env::var_os("DKC_VECTOR_LIFETIME_DB") else {
        return Ok(());
    };
    let store = store::open(path, StoreLimits::default())?;
    if std::env::var("DKC_VECTOR_LIFETIME_ACTION")? == "create" {
        store.ensure_vector_table(table("future", 2))?;
        let hash = insert_unit(&store, 1)?;
        store.insert_vector(VectorInsert {
            table_slug: "future",
            unit_id: 1,
            ctx_hash: &[1; 32],
            embedding_hash: &hash,
            embedding: &[1.0, 0.0],
        })?;
    } else {
        store.drop_vector_table("future")?;
    }
    Ok(())
}
fn child(path: &std::path::Path, action: &str) -> Result<(), Box<dyn StdError>> {
    let output = Command::new(std::env::current_exe()?)
        .args([
            "--exact",
            "store::vectors::cross_process_tests::table_child",
            "--nocapture",
        ])
        .env("DKC_VECTOR_LIFETIME_DB", path)
        .env("DKC_VECTOR_LIFETIME_ACTION", action)
        .output()?;
    assert!(
        output.status.success(),
        "{} {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}
#[test]
fn child_created_tables_and_retired_snapshots_work_without_reopening_readers()
-> Result<(), Box<dyn StdError>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("code.db");
    let reader = store::open(&path, StoreLimits::default())?;
    let unregistered = store::open(&path, StoreLimits::default())?;
    child(&path, "create")?;
    // All pooled connections existed before the child's table creation.
    for _ in 0..8 {
        assert_eq!(
            reader.scan_vectors("future", &[1.0, 0.0], 1, VectorScanPredicate::default())?[0]
                .unit_id,
            1
        );
    }
    unregistered.read_snapshot(|snapshot| -> Result<(), Box<dyn StdError>> {
        assert_eq!(
            snapshot.query_row("SELECT count(*) FROM embedding_tables", [], |r| r
                .get::<_, i64>(0))?,
            1
        );
        child(&path, "drop")?;
        // Native initialization and the scan must still use the pinned old schema.
        assert_eq!(
            scan_vectors_in(
                snapshot,
                "future",
                &[1.0, 0.0],
                1,
                VectorScanPredicate::default()
            )?[0]
                .unit_id,
            1
        );
        Ok(())
    })??;
    assert!(matches!(
        reader.scan_vectors("future", &[1.0, 0.0], 1, VectorScanPredicate::default()),
        Err(Error::VectorTableMissing { .. })
    ));
    assert_eq!(
        reader
            .writer()?
            .query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
        "ok"
    );
    Ok(())
}
#[test]
fn foreign_transitions_reclaim_slots_on_every_long_lived_reader() -> Result<(), Box<dyn StdError>> {
    let file = tempfile::NamedTempFile::new()?;
    let writer = store::open(file.path(), StoreLimits::default())?;
    let reader = store::open(file.path(), StoreLimits::default())?;
    let hash = insert_unit(&writer, 1)?;
    let mut previous: Option<String> = None;
    for i in 0..140 {
        let slug = format!("foreign_{i}");
        writer.ensure_vector_table(table(&slug, 2))?;
        writer.replace_vector(VectorInsert {
            table_slug: &slug,
            unit_id: 1,
            ctx_hash: &[1; 32],
            embedding_hash: &hash,
            embedding: &[1.0, 0.0],
        })?;
        if let Some(previous) = previous {
            assert!(writer.retire_empty_vector_table(&previous)?);
        }
        for _ in 0..4 {
            assert_eq!(
                reader.scan_vectors(&slug, &[1.0, 0.0], 1, VectorScanPredicate::default())?[0]
                    .unit_id,
                1
            );
        }
        previous = Some(slug);
    }
    assert_eq!(reader.vector_tables()?.len(), 1);
    Ok(())
}
