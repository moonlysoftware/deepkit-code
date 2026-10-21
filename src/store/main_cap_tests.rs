//! Main-file cap regressions for v1 §3.5, including startup and lowered caps.
use super::*;
use std::error::Error as StdError;

#[test]
fn startup_refuses_subpage_caps_and_resumes_partial_schema_with_a_larger_cap()
-> Result<(), Box<dyn StdError>> {
    let page_size = SQLITE_PAGE_SIZE as u64;
    for cap in [0, 1, page_size - 1, 65535, 131072] {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("code.db");
        let limits = StoreLimits {
            db_max_bytes: cap,
            ..Default::default()
        };
        let result = open(&path, limits);
        assert!(matches!(
            result,
            Err(Error::DatabaseGrowthLimit)
                | Err(Error::Sqlite(rusqlite::Error::SqliteFailure(
                    rusqlite::ffi::Error {
                        code: rusqlite::ErrorCode::DiskFull,
                        ..
                    },
                    _
                )))
        ));
        let bytes = std::fs::metadata(&path)?.len();
        assert!(
            bytes <= cap / page_size * page_size,
            "{bytes} bytes exceeds {cap} cap at {page_size}-byte pages"
        );
        let store = open(&path, StoreLimits::default())?;
        assert_eq!(store.get_meta("schema_version")?.as_deref(), Some("23"));
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
fn main_cap_fences_ordinary_writers_preserves_data_and_reports_lowered_limits()
-> Result<(), Box<dyn StdError>> {
    for page_size in [4096, 16384, 65536] {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("code.db");
        register_vector_extension()?;
        let observer = Connection::open(&path)?;
        observer.pragma_update(None, "page_size", page_size)?;
        observer.execute_batch("CREATE TABLE cap_payload(value BLOB)")?;
        let initial = open(&path, StoreLimits::default())?;
        initial.set_meta("retained", "original")?;
        observer.execute_batch("CREATE TRIGGER overflow_main AFTER UPDATE ON meta WHEN NEW.key='retained' BEGIN INSERT INTO cap_payload VALUES(zeroblob(16777216)); END;")?;
        wal::checkpoint(&initial)?;
        let allocated = initial.footprint()?.bytes as u64;
        let cap = allocated + page_size as u64 * 2 + 17;
        drop(initial);
        let limits = StoreLimits {
            db_max_bytes: cap,
            ..Default::default()
        };
        let first = open(&path, limits)?;
        let second = open(&path, limits)?;
        for store in [&first, &second] {
            let error = store.set_meta("retained", "changed").unwrap_err();
            assert!(
                matches!(error, Error::Sqlite(ref e) if e.sqlite_error_code() == Some(rusqlite::ErrorCode::DiskFull)),
                "{error:?}"
            );
            assert_eq!(store.get_meta("retained")?.as_deref(), Some("original"));
            assert_eq!(
                observer.query_row("SELECT count(*) FROM cap_payload", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            wal::checkpoint(store)?;
            assert!(std::fs::metadata(&path)?.len() <= cap / page_size as u64 * page_size as u64);
        }
        observer.execute_batch("DROP TRIGGER overflow_main")?;
        first.set_meta("retained", "success")?;
        wal::checkpoint(&first)?;
        let retained = std::fs::metadata(&path)?.len();
        let lowered = open(
            &path,
            StoreLimits {
                db_max_bytes: 1,
                ..Default::default()
            },
        )?;
        assert_eq!(lowered.get_meta("retained")?.as_deref(), Some("success"));
        let state = capacity::status(&*lowered.writer()?, &lowered)?;
        let json = serde_json::to_value(&state)?;
        assert_eq!(json["configured_limit_bytes"], 1);
        assert_eq!(json["effective_limit_bytes"], 0);
        assert_eq!(json["file_bytes"], retained);
        assert_eq!(json["over_limit"], true);
        assert_eq!(json["page_growth_available_bytes"], 0);
        assert!(state.summary().contains("data retained"));
        assert!(
            lowered
                .set_meta("new_large_value", &"x".repeat(4 * 1024 * 1024))
                .is_err()
        );
        wal::checkpoint(&lowered)?;
        assert_eq!(std::fs::metadata(&path)?.len(), retained);
        assert_eq!(lowered.get_meta("retained")?.as_deref(), Some("success"));
        let raised = open(&path, StoreLimits::default())?;
        raised.set_meta("new_large_value", &"x".repeat(262144))?;
        assert_eq!(
            raised.get_meta("new_large_value")?.map(|v| v.len()),
            Some(262144)
        );
        assert_eq!(
            observer.query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
            "ok"
        );
    }
    Ok(())
}

#[test]
fn ordinary_growth_leaves_space_for_control_publication() -> Result<(), Box<dyn StdError>> {
    for page_size in [4096, 16384, 65536] {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("code.db");
        let db = Connection::open(&path)?;
        db.pragma_update(None, "page_size", page_size)?;
        db.execute_batch("CREATE TABLE payload(value BLOB)")?;
        let original = open(&path, StoreLimits::default())?;
        wal::checkpoint(&original)?;
        let allocated = original.footprint()?.bytes as u64;
        drop(original);
        let cap = allocated + 1024 * 1024;
        let store = open(
            &path,
            StoreLimits {
                db_max_bytes: cap,
                ..Default::default()
            },
        )?;
        let growth = capacity::growth_limit(cap) / page_size as u64 * page_size as u64;
        let payload = growth.saturating_sub(allocated) + page_size as u64 * 4;
        let write = |kind| {
            admission::transaction::<_, Error>(&store, kind, |tx| {
                tx.execute("INSERT INTO payload VALUES(zeroblob(?1))", [payload as i64])?;
                tx.commit()?;
                Ok(())
            })
        };
        let error = write(admission::Kind::Growth).unwrap_err();
        assert!(
            matches!(error, Error::Sqlite(ref error) if error.sqlite_error_code() == Some(rusqlite::ErrorCode::DiskFull))
        );
        assert_eq!(
            db.query_row("SELECT count(*) FROM payload", [], |r| r.get::<_, i64>(0))?,
            0
        );
        write(admission::Kind::Control)?;
        wal::checkpoint(&store)?;
        let used = store.footprint()?.bytes as u64;
        assert!(used > growth && used <= cap);
        let status = serde_json::to_value(capacity::status(&db, &store)?)?;
        assert_eq!(status["growth_limit_bytes"], growth);
        assert!(
            status["unallocated_control_bytes"].as_u64().unwrap()
                < status["control_headroom_bytes"].as_u64().unwrap()
        );
        assert!(std::fs::metadata(&path)?.len() <= cap);
    }
    Ok(())
}

#[test]
fn failed_storage_deferral_preserves_pending_state_and_rechecks_inputs()
-> Result<(), Box<dyn StdError>> {
    for changed_inputs in [false, true] {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("code.db");
        let initial = open(&path, StoreLimits::default())?;
        let file = FileUpsert {
            path: "a.rs",
            blake3: &[1; 32],
            size: 0,
            mtime_ns: 1,
            lang: Some("rust"),
            kind: FileKind::Code,
            git_blob: None,
            max_level: 2,
            treatment: "default",
            indexed_at: None,
            needs_vlm: false,
        };
        let generation = initial.publish_catalogue(
            None,
            l1::Catalogue {
                discovery: None,
                max_files: u64::MAX,
                generation: initial.l1_generation()?,
                files: &[l1::Observation {
                    file,
                    fingerprint: &[2; 32],
                }],
                renames: &[],
                deleted: &[],
                dirs: &[],
                complete: true,
                full: false,
            },
            || Ok(()),
        )?;
        let id = initial.index_snapshot()?.files[0].id;
        initial.writer()?.execute_batch("CREATE TABLE payload(value BLOB);
            CREATE TRIGGER refuse_content AFTER UPDATE OF l1_fingerprint ON files BEGIN INSERT INTO payload VALUES(zeroblob(16777216)); END;")?;
        if !changed_inputs {
            initial.writer()?.execute_batch("CREATE TRIGGER refuse_deferral AFTER UPDATE OF l1_deferred_reason ON files WHEN NEW.l1_deferred_reason='storage_full' BEGIN INSERT INTO payload VALUES(zeroblob(16777216)); END;")?;
        }
        let cap = initial.footprint()?.bytes as u64 + 1024 * 1024;
        drop(initial);
        let store = open(
            &path,
            StoreLimits {
                db_max_bytes: cap,
                ..Default::default()
            },
        )?;
        let checks = std::cell::Cell::new(0);
        let result = store.publish_file_content(
            None,
            l1::Fence {
                generation,
                file_id: id,
                path: "a.rs",
                content_hash: &[1; 32],
                fingerprint: &[2; 32],
            },
            &[],
            false,
            || {
                checks.set(checks.get() + 1);
                if changed_inputs && checks.get() == 2 {
                    Err(Error::L1InputsChanged)
                } else {
                    Ok(())
                }
            },
        );
        assert_eq!(checks.get(), 2);
        if changed_inputs {
            assert!(matches!(result, Err(Error::L1InputsChanged)));
        } else {
            assert!(
                matches!(result, Err(Error::Sqlite(ref error)) if error.sqlite_error_code() == Some(rusqlite::ErrorCode::DiskFull))
            );
        }
        assert_eq!(store.pending_l1_paths()?, vec!["a.rs"]);
        assert!(store.l1_deferred_files()?.is_empty());
        assert_eq!(store.get_meta("indexed_head")?.as_deref(), Some(""));
        assert_eq!(
            store
                .writer()?
                .query_row("SELECT count(*) FROM payload", [], |r| r.get::<_, i64>(0))?,
            0
        );
    }
    Ok(())
}
