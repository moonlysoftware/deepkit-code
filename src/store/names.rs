//! Direct trigram-name ownership and deletion; see v1.md §3.2 and §4.3.

use rusqlite::{Transaction, params};

use super::{Error, FtsNameKind};

/// Existing positive source IDs occupy disjoint rowid domains. These internal
/// FTS rowids are never public identifiers: paths are negative, symbols positive.
fn rowid(kind: FtsNameKind, ref_id: i64) -> Result<i64, Error> {
    if ref_id <= 0 {
        return Err(Error::InvalidNameIndex);
    }
    Ok(match kind {
        FtsNameKind::Symbol => ref_id,
        FtsNameKind::Path => -ref_id,
    })
}

/// Replaces one referent's name atomically without scanning unindexed metadata.
pub(super) fn insert(
    transaction: &Transaction<'_>,
    name: &str,
    kind: FtsNameKind,
    ref_id: i64,
) -> Result<(), Error> {
    let id = rowid(kind, ref_id)?;
    transaction
        .prepare_cached("DELETE FROM fts_names WHERE rowid=?1")?
        .execute([id])?;
    transaction
        .prepare_cached("INSERT INTO fts_names(rowid,name,kind,ref_id) VALUES (?1,?2,?3,?4)")?
        .execute(params![id, name, kind.as_str(), ref_id])?;
    Ok(())
}

/// Removes a path name using its existing file identity; no catalogue scan.
pub(super) fn remove_path(transaction: &Transaction<'_>, file_id: i64) -> Result<(), Error> {
    transaction
        .prepare_cached("DELETE FROM fts_names WHERE rowid=?1")?
        .execute([rowid(FtsNameKind::Path, file_id)?])?;
    Ok(())
}

/// Removes names before deleting their owning symbols. The indexed file lookup
/// supplies exact FTS rowids, so unrelated names are never visited.
pub(super) fn remove_symbols(transaction: &Transaction<'_>, file_id: i64) -> Result<(), Error> {
    transaction
        .prepare_cached(
            "DELETE FROM fts_names WHERE rowid IN (SELECT id FROM symbols WHERE file_id=?1)",
        )?
        .execute([file_id])?;
    Ok(())
}

/// Rebuilds names from canonical sources with the same rowid contract as updates.
pub(super) fn rebuild(transaction: &Transaction<'_>) -> Result<(), Error> {
    if transaction
        .prepare(
            "SELECT id FROM files WHERE id<=0 UNION ALL SELECT id FROM symbols WHERE id<=0 LIMIT 1",
        )?
        .exists([])?
    {
        return Err(Error::InvalidNameIndex);
    }
    transaction.execute("DELETE FROM fts_names", [])?;
    transaction.execute(
        "INSERT INTO fts_names(rowid,name,kind,ref_id) SELECT -id,path,'path',id FROM files",
        [],
    )?;
    transaction.execute(
        "INSERT INTO fts_names(rowid,name,kind,ref_id) SELECT id,name,'symbol',id FROM symbols",
        [],
    )?;
    Ok(())
}

/// Rekeys legacy names without changing payloads, referents, units or vectors.
/// Refuse invalid/ambiguous ownership before deleting any legacy FTS data.
pub(super) fn migrate(transaction: &Transaction<'_>) -> Result<(), Error> {
    let rows = transaction
        .prepare("SELECT name,kind,ref_id FROM fts_names ORDER BY rowid")?
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    let mut seen = std::collections::BTreeSet::new();
    for (_, kind, ref_id) in &rows {
        let kind = match kind.as_str() {
            "path" => FtsNameKind::Path,
            "symbol" => FtsNameKind::Symbol,
            _ => return Err(Error::InvalidNameIndex),
        };
        if !seen.insert(rowid(kind, *ref_id)?) {
            return Err(Error::InvalidNameIndex);
        }
    }
    transaction.execute("DELETE FROM fts_names", [])?;
    for (name, kind, ref_id) in rows {
        let id = if kind == "path" { -ref_id } else { ref_id };
        transaction
            .prepare_cached("INSERT INTO fts_names(rowid,name,kind,ref_id) VALUES (?1,?2,?3,?4)")?
            .execute(params![id, name, kind, ref_id])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{self, StoreLimits};
    use std::path::Path;

    fn legacy(path: &Path) -> rusqlite::Connection {
        let connection = rusqlite::Connection::open(path).unwrap();
        for migration in store::MIGRATIONS.iter().take(14) {
            connection.execute_batch(migration.sql).unwrap();
        }
        connection
            .execute("INSERT INTO meta VALUES('schema_version','14')", [])
            .unwrap();
        connection
            .execute_batch(
                "INSERT INTO fts_names(rowid,name,kind,ref_id) VALUES
             (101,'source/Original.rs','path',7), (102,'OriginalSymbol','symbol',7),
             (103,'UnrelatedSymbol','symbol',8);",
            )
            .unwrap();
        connection
    }

    fn rows(connection: &rusqlite::Connection) -> Vec<(i64, String, String, i64)> {
        connection
            .prepare("SELECT rowid,name,kind,ref_id FROM fts_names ORDER BY rowid")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    }

    #[test]
    fn name_domains_replace_independently_and_rollback_as_one_transaction() {
        let store = store::open(":memory:", StoreLimits::default()).unwrap();
        store
            .insert_fts_name("OriginalPath", FtsNameKind::Path, 7)
            .unwrap();
        store
            .insert_fts_name("OriginalSymbol", FtsNameKind::Symbol, 7)
            .unwrap();
        store
            .insert_fts_name("UnrelatedSymbol", FtsNameKind::Symbol, 8)
            .unwrap();
        store
            .insert_fts_name("ChangedSymbol", FtsNameKind::Symbol, 7)
            .unwrap();
        assert!(
            store
                .query_fts_names("OriginalSymbol", 10)
                .unwrap()
                .is_empty()
        );
        assert_eq!(store.query_fts_names("OriginalPath", 10).unwrap().len(), 1);
        assert_eq!(
            store.query_fts_names("ChangedSymbol", 10).unwrap()[0].ref_id,
            7
        );
        {
            let mut writer = store.writer().unwrap();
            let transaction = writer.transaction().unwrap();
            insert(&transaction, "RolledBackPath", FtsNameKind::Path, 7).unwrap();
            remove_path(&transaction, 7).unwrap();
            // No commit: the replacement and deletion must both roll back.
        }
        assert_eq!(store.query_fts_names("OriginalPath", 10).unwrap().len(), 1);
        {
            let mut writer = store.writer().unwrap();
            let transaction = writer.transaction().unwrap();
            remove_path(&transaction, 7).unwrap();
            transaction.commit().unwrap();
        }
        assert!(
            store
                .query_fts_names("OriginalPath", 10)
                .unwrap()
                .is_empty()
        );
        assert_eq!(store.query_fts_names("ChangedSymbol", 10).unwrap().len(), 1);
        assert_eq!(
            store.query_fts_names("UnrelatedSymbol", 10).unwrap().len(),
            1
        );
        for id in [0, -1, i64::MIN] {
            assert!(matches!(
                store.insert_fts_name("Invalid", FtsNameKind::Path, id),
                Err(Error::InvalidNameIndex)
            ));
        }
        store
            .insert_fts_name("LargestPath", FtsNameKind::Path, i64::MAX)
            .unwrap();
        store
            .insert_fts_name("LargestSymbol", FtsNameKind::Symbol, i64::MAX)
            .unwrap();
        assert_eq!(store.query_fts_names("Largest", 10).unwrap().len(), 2);
    }

    #[test]
    fn legacy_name_upgrade_preserves_payload_and_retries_after_atomic_failure() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("code.db");
        let legacy = legacy(&path);
        let before = rows(&legacy);
        legacy
            .execute_batch(
                "CREATE TRIGGER refuse_names BEFORE UPDATE ON meta
             WHEN NEW.key='schema_version' AND NEW.value='15'
             BEGIN SELECT RAISE(ABORT,'stop name migration'); END;",
            )
            .unwrap();
        assert!(store::open(&path, StoreLimits::default()).is_err());
        assert_eq!(rows(&legacy), before);
        assert_eq!(store::schema_version(&legacy).unwrap(), 14);
        assert!(
            !legacy
                .prepare("SELECT name FROM sqlite_schema WHERE name='units_parent'")
                .unwrap()
                .exists([])
                .unwrap()
        );
        legacy.execute_batch("DROP TRIGGER refuse_names").unwrap();
        let current = store::open(&path, StoreLimits::default()).unwrap();
        assert_eq!(
            current.get_meta("schema_version").unwrap(),
            Some(store::LATEST_SCHEMA_VERSION.to_string())
        );
        let migrated = rows(&legacy);
        assert_eq!(migrated.iter().map(|r| r.0).collect::<Vec<_>>(), [-7, 7, 8]);
        let payloads = |rows: Vec<(i64, String, String, i64)>| {
            let mut result = rows
                .into_iter()
                .map(|(_, name, kind, id)| (name, kind, id))
                .collect::<Vec<_>>();
            result.sort();
            result
        };
        assert_eq!(payloads(migrated), payloads(before));
        assert_eq!(current.query_fts_names("Original", 10).unwrap().len(), 2);
        drop(current);
        let reopened = store::open(&path, StoreLimits::default()).unwrap();
        assert_eq!(reopened.query_fts_names("Original", 10).unwrap().len(), 2);
    }

    #[test]
    fn invalid_legacy_name_ownership_is_preserved_and_refused() {
        for sql in [
            "INSERT INTO fts_names(name,kind,ref_id) VALUES('Alias','symbol',7)",
            "INSERT INTO fts_names(name,kind,ref_id) VALUES('Invalid','path',0)",
            "INSERT INTO fts_names(name,kind,ref_id) VALUES('Invalid','unknown',9)",
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("code.db");
            let legacy = legacy(&path);
            legacy.execute(sql, []).unwrap();
            let before = rows(&legacy);
            assert!(matches!(
                store::open(&path, StoreLimits::default()),
                Err(Error::InvalidNameIndex)
            ));
            assert_eq!(rows(&legacy), before);
            assert_eq!(store::schema_version(&legacy).unwrap(), 14);
        }
    }

    #[test]
    fn l1_reverse_references_have_indexed_lookup_plans() {
        let store = store::open(":memory:", StoreLimits::default()).unwrap();
        let writer = store.writer().unwrap();
        for (query, index) in [
            ("SELECT id FROM units WHERE parent_unit=1", "units_parent"),
            ("SELECT path FROM dirs WHERE module_unit=1", "dirs_module"),
            (
                "SELECT id FROM symbols WHERE summary_unit=1",
                "symbols_summary",
            ),
            (
                "SELECT id FROM symbols WHERE history_unit=1",
                "symbols_history",
            ),
            ("SELECT id FROM commits WHERE unit_id=1", "commits_unit"),
            ("SELECT id FROM prs WHERE unit_id=1", "prs_unit"),
            ("SELECT id FROM prs WHERE summary_unit=1", "prs_summary"),
            ("SELECT rowid FROM edges WHERE file_id=1", "edges_file"),
            (
                "SELECT unit_id FROM unit_sources WHERE source_kind='file' AND source_id=1",
                "unit_sources_source",
            ),
        ] {
            let plans = writer
                .prepare(&format!("EXPLAIN QUERY PLAN {query}"))
                .unwrap()
                .query_map([], |r| r.get::<_, String>(3))
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert!(
                plans
                    .iter()
                    .any(|p| p.contains("SEARCH") && p.contains(index)),
                "{query}: {plans:?}"
            );
            assert!(
                !plans.iter().any(|p| p.contains("SCAN")),
                "{query}: {plans:?}"
            );
        }
    }
}
