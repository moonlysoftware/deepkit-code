//! Migration admission regressions for v1 §3.5 and durable schema publication.
use super::*;
use std::{error::Error as StdError, sync::Arc};

const CAP: u64 = 8 * 1024 * 1024;

#[test]
fn context_upgrade_keeps_legacy_evidence_unknown() -> Result<(), Box<dyn StdError>> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("code.db");
    let store = open(&path, StoreLimits::default())?;
    store.writer()?.execute_batch("INSERT INTO unit_contents(hash,text,tokens) VALUES(zeroblob(32),'retained',1);
        INSERT INTO units(id,hash,content_hash,source_key,kind,header,created_at,updated_at) VALUES(1,x'01',zeroblob(32),'fixture','chunk','header',0,0);
        DROP TRIGGER units_invalidate_context; DROP TRIGGER files_invalidate_context; DROP TABLE unit_contexts; DROP INDEX edges_destination; UPDATE meta SET value='20' WHERE key='schema_version';")?;
    drop(store);
    let store = open(&path, StoreLimits::default())?;
    let db = store.writer()?;
    assert_eq!(
        db.query_row("SELECT count(*) FROM unit_contexts", [], |r| r
            .get::<_, i64>(0))?,
        0
    );
    assert_eq!(
        db.query_row("SELECT header FROM units WHERE id=1", [], |r| r
            .get::<_, String>(0))?,
        "header"
    );
    assert_eq!(schema_version(&db)?, LATEST_SCHEMA_VERSION);
    Ok(())
}

#[test]
fn role_free_upgrade_preserves_legacy_publication_and_vector_without_inventing_proof()
-> Result<(), Box<dyn StdError>> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("code.db");
    let store = open(&path, StoreLimits::default())?;
    store.writer()?.execute_batch("INSERT INTO unit_contents(hash,text,tokens) VALUES(x'01','retained',1);
        INSERT INTO units(id,hash,content_hash,source_key,kind,header,created_at,updated_at) VALUES(1,x'02',x'01','fixture','chunk','header',0,0);
        INSERT INTO embedding_tables(slug,model,profile,dims,created_at) VALUES('fixture','fixture','remote',2,0);
        CREATE TABLE vec_fixture(unit_id INTEGER PRIMARY KEY REFERENCES units(id),v BLOB,ctx_hash BLOB);
        INSERT INTO vec_fixture VALUES(1,x'003c0000',zeroblob(32));
        INSERT INTO embedding_publications(unit_id,table_slug,desired_hash) VALUES(1,'fixture',zeroblob(32));
        DROP TRIGGER IF EXISTS units_invalidate_context; DROP TRIGGER IF EXISTS files_invalidate_context; DROP TABLE IF EXISTS unit_contexts; ALTER TABLE embedding_publications DROP COLUMN role_free_hash;
        DROP INDEX edges_destination; UPDATE meta SET value='19' WHERE key='schema_version';")?;
    drop(store);
    let store = open(&path, StoreLimits::default())?;
    let db = store.writer()?;
    let retained: (Vec<u8>, Option<Vec<u8>>, Vec<u8>) = db.query_row(
        "SELECT p.desired_hash,p.role_free_hash,v.v FROM embedding_publications p JOIN vec_fixture v ON v.unit_id=p.unit_id WHERE p.unit_id=1", [],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
    assert_eq!(retained, (vec![0; 32], None, vec![0, 60, 0, 0]));
    db.execute(
        "UPDATE embedding_publications SET role_free_hash=zeroblob(32)",
        [],
    )?;
    assert!(
        db.execute(
            "UPDATE embedding_publications SET role_free_hash=zeroblob(31)",
            []
        )
        .is_err()
    );
    assert!(
        db.execute(
            "UPDATE embedding_publications SET role_free_hash=printf('%032d',0)",
            []
        )
        .is_err()
    );
    assert_eq!(schema_version(&db)?, LATEST_SCHEMA_VERSION);
    Ok(())
}

#[test]
fn page_requirements_upgrade_preserves_ordinal_and_active_lease() -> Result<(), Box<dyn StdError>> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("code.db");
    let store = open(&path, StoreLimits::default())?;
    store.writer()?.execute_batch("DROP TRIGGER IF EXISTS units_invalidate_context; DROP TRIGGER IF EXISTS files_invalidate_context; DROP TABLE IF EXISTS unit_contexts; ALTER TABLE embedding_publications DROP COLUMN role_free_hash;
        DROP INDEX units_page_anchor;
        ALTER TABLE jobs DROP COLUMN required_l3_units;
        DROP INDEX edges_destination; UPDATE meta SET value='18' WHERE key='schema_version';
        INSERT INTO jobs(id,kind,target_kind,target_id,target_hash,desired_hash,provider,priority,tokens,state,attempts,leased_until,lease_owner,lease_generation,page,created_at,updated_at)
        VALUES(1,'enrich:vlm_page','file',42,zeroblob(32),zeroblob(32),'chat:fixture',1,10,'leased',2,120,'retained-owner',7,3,0,1);")?;
    drop(store);
    let store = open(&path, StoreLimits::default())?;
    let db = store.writer()?;
    let retained:(String,i64,i64,String,i64,i64,i64)=db.query_row(
        "SELECT state,attempts,leased_until,lease_owner,lease_generation,page,required_l3_units FROM jobs WHERE id=1",
        [],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?)))?;
    assert_eq!(
        retained,
        ("leased".into(), 2, 120, "retained-owner".into(), 7, 3, 1)
    );
    for bad in [0_i64, -1, 4294967296] {
        assert!(
            db.execute("UPDATE jobs SET required_l3_units=?1 WHERE id=1", [bad])
                .is_err()
        );
    }
    db.execute(
        "UPDATE jobs SET required_l3_units=4294967295 WHERE id=1",
        [],
    )?;
    assert!(
        db.execute("UPDATE jobs SET kind='embed',page=NULL WHERE id=1", [])
            .is_err()
    );
    assert_eq!(schema_version(&db)?, LATEST_SCHEMA_VERSION);
    Ok(())
}

#[test]
fn ambiguous_git_oid_upgrade_retains_payloads_names_and_completed_l1()
-> Result<(), Box<dyn StdError>> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("code.db");
    let store = open(&path, StoreLimits::default())?;
    drop_module_summary(&*store.writer()?);
    store.writer()?.execute_batch(
        "INSERT INTO files(id,path,blake3,size,mtime_ns,kind,git_blob,max_level,treatment,l1_pending)
         VALUES(7,'source.rs',x'0102',19,1,'code',zeroblob(20),1,'default',0);
         INSERT INTO unit_contents(hash,text,tokens) VALUES(x'03','retainedmarker',1);
         INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,created_at,updated_at)
         VALUES(9,x'04',x'03','source.rs',7,'chunk','retainedmarker',1,1);
         INSERT INTO fts_units(rowid,header,idents,body,prose) VALUES(9,'retainedmarker','','','');
         INSERT INTO fts_names(rowid,name,kind,ref_id) VALUES(-7,'source.rs','path',7);
         UPDATE meta SET value='15' WHERE key='schema_version';"
    )?;
    drop(store);
    let store = open(&path, StoreLimits::default())?;
    let writer = store.writer()?;
    let record: (Vec<u8>, Option<Vec<u8>>, i64) = writer.query_row(
        "SELECT blake3,git_blob,l1_pending FROM files WHERE id=7",
        [],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    )?;
    assert_eq!(record, (vec![1, 2], None, 0));
    let text: String = writer.query_row(
        "SELECT c.text FROM units u JOIN unit_contents c ON c.hash=u.content_hash WHERE u.id=9",
        [],
        |r| r.get(0),
    )?;
    assert_eq!(text, "retainedmarker");
    assert_eq!(
        writer.query_row(
            "SELECT rowid FROM fts_units WHERE fts_units MATCH 'retainedmarker'",
            [],
            |r| r.get::<_, i64>(0)
        )?,
        9
    );
    assert_eq!(
        writer.query_row("SELECT ref_id FROM fts_names WHERE rowid=-7", [], |r| r
            .get::<_, i64>(0))?,
        7
    );
    assert_eq!(schema_version(&writer)?, LATEST_SCHEMA_VERSION);
    Ok(())
}

/// Removes version-15 indexes when a populated test database is downgraded to
/// exercise an older migration. Leave all historical payloads and vectors intact.
pub(crate) fn drop_l1_delete_indexes(connection: &Connection) {
    drop_module_summary(connection);
    for line in MIGRATIONS[14].sql.lines() {
        if let Some(definition) = line.strip_prefix("CREATE INDEX ") {
            let name = definition.split_whitespace().next().unwrap();
            connection
                .execute_batch(&format!("DROP INDEX {name}"))
                .unwrap();
        }
    }
}

/// Reconstructs pre-L3 directory schema for populated historical migration fixtures.
fn drop_module_summary(connection: &Connection) {
    connection
        .execute_batch("DROP INDEX IF EXISTS edges_destination;")
        .unwrap();
    connection
        .execute_batch("DROP TRIGGER IF EXISTS units_invalidate_context; DROP TRIGGER IF EXISTS files_invalidate_context; DROP TABLE IF EXISTS unit_contexts;")
        .unwrap();
    let role_column: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM pragma_table_info('embedding_publications') WHERE name='role_free_hash')", [], |r| r.get(0)
    ).unwrap();
    if role_column {
        connection
            .execute_batch("DROP TRIGGER IF EXISTS units_invalidate_context; DROP TRIGGER IF EXISTS files_invalidate_context; DROP TABLE IF EXISTS unit_contexts; ALTER TABLE embedding_publications DROP COLUMN role_free_hash;")
            .unwrap();
    }
    connection
        .execute_batch(
            "DROP INDEX units_page_anchor; ALTER TABLE jobs DROP COLUMN required_l3_units;
            DROP TABLE page_publications; ALTER TABLE jobs DROP COLUMN page;
            DROP INDEX dirs_summary; ALTER TABLE dirs DROP COLUMN summary_unit;",
        )
        .unwrap();
}

#[test]
fn module_summary_pointer_clears_when_parent_and_summary_are_retired()
-> Result<(), Box<dyn StdError>> {
    let store = open(":memory:", StoreLimits::default())?;
    store.writer()?.execute_batch("INSERT INTO unit_contents(hash,text,tokens) VALUES(x'01','module',1),(x'02','summary',1);
        INSERT INTO units(id,hash,content_hash,source_key,kind,header,created_at,updated_at) VALUES(1,x'01',x'01','module:fixture','module','module',0,0);
        INSERT INTO units(id,hash,content_hash,source_key,kind,header,parent_unit,ladder_level,created_at,updated_at) VALUES(2,x'02',x'02','summary:fixture','summary','summary',1,3,0,0);
        INSERT INTO dirs(path,merkle,file_count,module_unit,summary_unit) VALUES('fixture',x'03',5,1,2);")?;
    store.delete_unit(1)?;
    let pointers: (Option<i64>, Option<i64>) = store.writer()?.query_row(
        "SELECT module_unit,summary_unit FROM dirs WHERE path='fixture'",
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    assert_eq!(pointers, (None, None));
    assert!(store.unit_by_id(2)?.is_none());
    Ok(())
}

fn legacy(path: &Path, page_size: i64) -> Result<Connection, Box<dyn StdError>> {
    register_vector_extension()?;
    let connection = Connection::open(path)?;
    connection.pragma_update(None, "page_size", page_size)?;
    connection.execute_batch("VACUUM; PRAGMA journal_mode=WAL;")?;
    for migration in MIGRATIONS.iter().take(9) {
        connection.execute_batch(migration.sql)?;
    }
    connection.execute("INSERT INTO meta VALUES('schema_version','9')", [])?;
    Ok(connection)
}

#[test]
fn wal_refused_migration_preserves_schema_then_retries() -> Result<(), Box<dyn StdError>> {
    for page_size in [4096, 65536] {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("code.db");
        let db = legacy(&path, page_size)?;
        db.execute_batch(
            "CREATE TABLE payload(value BLOB);
            CREATE TRIGGER grow_migration AFTER UPDATE ON meta WHEN NEW.key='schema_version'
            BEGIN INSERT INTO payload VALUES(zeroblob(16777216)); END;
            PRAGMA wal_checkpoint(TRUNCATE);",
        )?;
        let limits = StoreLimits {
            wal_max_bytes: CAP,
            ..Default::default()
        };
        assert!(matches!(
            open(&path, limits),
            Err(Error::WalCapacity { .. })
        ));
        assert_eq!(schema_version(&db)?, 9);
        assert!(!db.prepare("SELECT name FROM pragma_table_info('usage_reservations') WHERE name='dispatched_at'")?.exists([])?);
        assert_eq!(
            db.query_row("SELECT count(*) FROM payload", [], |r| r.get::<_, i64>(0))?,
            0
        );
        assert!(wal::wal_bytes(&db)? <= CAP);
        assert_eq!(
            db.query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
            "ok"
        );
        db.execute_batch("DROP TRIGGER grow_migration")?;
        let store = open(&path, limits)?;
        assert_eq!(schema_version(&db)?, LATEST_SCHEMA_VERSION);
        assert_eq!(
            store
                .writer()?
                .query_row("PRAGMA foreign_keys", [], |r| r.get::<_, i64>(0))?,
            1
        );
        assert!(wal::wal_bytes(&db)? <= CAP);
    }
    Ok(())
}

#[test]
fn current_schema_opens_without_write_room_and_missing_defaults_are_atomic()
-> Result<(), Box<dyn StdError>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("code.db");
    let store = open(&path, StoreLimits::default())?;
    store.set_meta("retained", "original")?;
    wal::checkpoint(&store)?;
    let db = Connection::open(&path)?;
    db.execute_batch("BEGIN")?;
    assert_eq!(schema_version(&db)?, LATEST_SCHEMA_VERSION);
    store.set_meta("retained", "new")?;
    let before = wal::wal_bytes(&db)?;
    assert!(before > 1);
    let reopened = open(
        &path,
        StoreLimits {
            wal_max_bytes: 1,
            ..Default::default()
        },
    )?;
    assert_eq!(reopened.get_meta("retained")?.as_deref(), Some("new"));
    assert_eq!(wal::wal_bytes(&db)?, before);
    drop(reopened);
    db.execute_batch(
        "ROLLBACK;
        DELETE FROM meta WHERE key IN ('chunker_version','dims','indexed_at');
        CREATE TABLE payload(value BLOB);
        CREATE TRIGGER grow_defaults AFTER INSERT ON meta WHEN NEW.key='dims'
        BEGIN INSERT INTO payload VALUES(zeroblob(16777216)); END;
        PRAGMA wal_checkpoint(TRUNCATE);",
    )?;
    let limits = StoreLimits {
        wal_max_bytes: CAP,
        ..Default::default()
    };
    assert!(matches!(
        open(&path, limits),
        Err(Error::WalCapacity { .. })
    ));
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM meta WHERE key IN ('chunker_version','dims','indexed_at')",
            [],
            |r| r.get::<_, i64>(0)
        )?,
        0
    );
    assert_eq!(
        db.query_row("SELECT count(*) FROM payload", [], |r| r.get::<_, i64>(0))?,
        0
    );
    assert!(wal::wal_bytes(&db)? <= CAP);
    db.execute_batch("DROP TRIGGER grow_defaults")?;
    let reopened = open(&path, limits)?;
    assert_eq!(reopened.get_meta("retained")?.as_deref(), Some("new"));
    for (key, value) in META_DEFAULTS {
        assert_eq!(reopened.get_meta(key)?.as_deref(), Some(value));
    }
    Ok(())
}

#[test]
fn concurrent_openers_publish_each_migration_once() -> Result<(), Box<dyn StdError>> {
    for _ in 0..8 {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("code.db");
        let db = legacy(&path, 4096)?;
        db.execute_batch(
            "CREATE TABLE migration_audit(version TEXT);
            CREATE TRIGGER record_migration AFTER UPDATE ON meta WHEN NEW.key='schema_version'
            BEGIN INSERT INTO migration_audit VALUES(NEW.value); END;",
        )?;
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let handles = (0..2)
            .map(|_| {
                let path = path.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    open(path, StoreLimits::default())
                        .and_then(|store| store.get_meta("schema_version"))
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();
        for handle in handles {
            assert_eq!(
                handle.join().expect("opener thread")?.as_deref(),
                Some("23")
            );
        }
        let versions = db
            .prepare("SELECT version FROM migration_audit ORDER BY rowid")?
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            versions,
            vec![
                "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"
            ]
        );
    }
    Ok(())
}

#[test]
fn migration_adds_covering_indexes_to_existing_vector_tables() -> Result<(), Box<dyn StdError>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("code.db");
    let store = open(&path, StoreLimits::default())?;
    store.ensure_vector_table(super::vectors::VectorTableSpec {
        slug: "counted",
        model: "fixture",
        profile: "remote",
        dims: 2,
        created_at: 1,
        active: true,
    })?;
    let table = super::vectors::vector_table_name("counted")?;
    let index = format!("{table}_unit_id_count");
    store.writer()?.execute_batch(&format!(
        "DROP INDEX {index}; UPDATE meta SET value='22' WHERE key='schema_version'"
    ))?;
    drop(store);

    let reopened = open(&path, StoreLimits::default())?;
    assert_eq!(reopened.get_meta("schema_version")?.as_deref(), Some("23"));
    let plan = reopened
        .writer()?
        .prepare(&format!("EXPLAIN QUERY PLAN SELECT COUNT(*) FROM {table}"))?
        .query_map([], |row| row.get::<_, String>(3))?
        .collect::<Result<Vec<_>, _>>()?;
    assert!(
        plan.iter()
            .any(|detail| detail.contains(&format!("USING COVERING INDEX {index}"))),
        "unexpected vector count plan: {plan:?}"
    );
    Ok(())
}

#[test]
fn migration_uses_remaining_space_while_an_old_reader_pins_wal() -> Result<(), Box<dyn StdError>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("code.db");
    let reader = legacy(&path, 65536)?;
    reader.execute_batch(
        "CREATE TABLE payload(value BLOB); PRAGMA wal_checkpoint(TRUNCATE); BEGIN",
    )?;
    assert_eq!(schema_version(&reader)?, 9);
    let writer = Connection::open(&path)?;
    writer.execute("INSERT INTO payload VALUES(zeroblob(3145728))", [])?;
    let pinned = wal::wal_bytes(&reader)?;
    assert!(pinned > 3 * 1024 * 1024);
    let store = open(
        &path,
        StoreLimits {
            wal_max_bytes: CAP,
            ..Default::default()
        },
    )?;
    assert_eq!(store.get_meta("schema_version")?.as_deref(), Some("23"));
    assert_eq!(schema_version(&reader)?, 9);
    assert!(wal::checkpoint(&store)?.busy);
    assert!(wal::wal_bytes(&reader)? <= CAP);
    reader.execute_batch("ROLLBACK")?;
    assert_eq!(schema_version(&reader)?, LATEST_SCHEMA_VERSION);
    assert!(!wal::checkpoint(&store)?.busy);
    Ok(())
}

#[test]
fn automatic_extension_open_is_read_only_and_native_metadata_is_admitted()
-> Result<(), Box<dyn StdError>> {
    register_vector_extension()?;
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("code.db");
    let raw = Connection::open(&path)?;
    let has_native = |db: &Connection| -> rusqlite::Result<bool> {
        db.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_schema WHERE name='_sqliteai_vector')",
            [],
            |r| r.get(0),
        )
    };
    assert!(!has_native(&raw)?);
    assert_eq!(std::fs::metadata(&path)?.len(), 0);
    assert_eq!(
        raw.query_row("SELECT vector_version()", [], |r| r.get::<_, String>(0))?,
        "1.1.0"
    );
    let store = open(&path, StoreLimits::default())?;
    assert!(has_native(&raw)?);
    raw.execute_batch("DROP TABLE _sqliteai_vector; PRAGMA wal_checkpoint(TRUNCATE);")?;
    let before = wal::wal_bytes(&raw)?;
    let readonly = Connection::open_with_flags(&path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    assert!(!has_native(&readonly)?);
    assert_eq!(wal::wal_bytes(&raw)?, before);
    assert!(matches!(
        open(
            &path,
            StoreLimits {
                wal_max_bytes: 1,
                ..Default::default()
            }
        ),
        Err(Error::WalCapacity { .. })
    ));
    assert!(!has_native(&raw)?);
    assert_eq!(wal::wal_bytes(&raw)?, before);

    // Two schema pages cannot commit into a one-frame reservation. This proves
    // the native callback propagates write-limit failure and rolls back its DDL.
    let page_size = u64::from(raw.query_row("PRAGMA page_size", [], |r| r.get::<_, u32>(0))?);
    assert!(matches!(
        wal::write_reserved(&store, CAP, page_size + 56, 8, |tx| {
            tx.execute_batch("SELECT dkc_vector_metadata_init()")?;
            Ok(())
        }),
        Err(Error::WalCapacity { .. })
    ));
    assert!(!has_native(&raw)?);
    assert!(wal::wal_bytes(&raw)? <= CAP);
    let reopened = open(&path, StoreLimits::default())?;
    assert!(has_native(&raw)?);
    assert_eq!(reopened.get_meta("schema_version")?.as_deref(), Some("23"));
    assert_eq!(
        raw.query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
        "ok"
    );
    Ok(())
}

#[test]
fn old_dispatches_upgrade_without_inventing_endpoint_identity() -> Result<(), Box<dyn StdError>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("code.db");
    let db = legacy(&path, 4096)?;
    db.execute_batch(MIGRATIONS[9].sql)?;
    db.execute("UPDATE meta SET value='10' WHERE key='schema_version'", [])?;
    db.execute("INSERT INTO usage_reservations(id,provider,day,calls,tokens,state,created_at,dispatched_at) VALUES('old','embed:remote:old','1970-01-01',1,2,'dispatched',0,1)", [])?;
    drop(db);
    let store = open(&path, StoreLimits::default())?;
    let report = crate::queue::inspection::read(&store)?;
    assert!(report.endpoints.is_empty());
    assert_eq!(report.unknown_endpoint_dispatches, 1);
    assert_eq!(store.get_meta("schema_version")?.as_deref(), Some("23"));
    Ok(())
}

#[test]
fn destination_index_upgrade_preserves_resolved_and_unresolved_edges()
-> Result<(), Box<dyn StdError>> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("code.db");
    let store = open(&path, StoreLimits::default())?;
    store.writer()?.execute_batch(
        "DROP INDEX edges_destination;
         UPDATE meta SET value='21' WHERE key='schema_version';
         INSERT INTO files(id,path,blake3,size,mtime_ns,kind,max_level,treatment)
           VALUES(1,'source.rs',zeroblob(32),1,0,'code',3,'code');
         INSERT INTO symbols(id,file_id,name,kind,line_start,line_end)
           VALUES(1,1,'target','function',1,1),(2,1,'caller','function',2,2);
         INSERT INTO edges(src_symbol,dst_name,dst_symbol,kind,file_id)
           VALUES(2,'different',1,'ref',1),(2,'target',NULL,'ref',1),
                 (2,'target',NULL,'ref',1),(2,'qualified::target',NULL,'test_of',1),
                 (2,'unrelated',NULL,'ref',1),(2,'target',2,'ref',1);",
    )?;
    let query = "SELECT rowid FROM edges WHERE dst_symbol=?1 OR (dst_symbol IS NULL AND dst_name IN (?2,?3)) ORDER BY rowid";
    let selected = |db: &Connection| -> rusqlite::Result<Vec<i64>> {
        db.prepare(query)?
            .query_map(params![1, "target", "qualified::target"], |r| r.get(0))?
            .collect()
    };
    let before = {
        let db = store.writer()?;
        selected(&db)?
    };
    assert_eq!(before, [1, 2, 3, 4]);
    drop(store);
    let store = open(&path, StoreLimits::default())?;
    let db = store.writer()?;
    assert_eq!(schema_version(&db)?, LATEST_SCHEMA_VERSION);
    assert_eq!(selected(&db)?, before);
    assert_eq!(
        db.query_row("SELECT count(*) FROM edges", [], |r| r.get::<_, i64>(0))?,
        6
    );
    let plan = db
        .prepare(&format!("EXPLAIN QUERY PLAN {query}"))?
        .query_map(params![1, "target", "qualified::target"], |r| {
            r.get::<_, String>(3)
        })?
        .collect::<Result<Vec<_>, _>>()?;
    assert!(
        plan.iter().any(|step| step.contains("edges_destination")
            && step.contains("dst_symbol=? AND dst_name=?")),
        "{plan:?}"
    );
    assert_eq!(
        db.query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
        "ok"
    );
    Ok(())
}
