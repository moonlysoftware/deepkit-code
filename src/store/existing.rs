//! Existing-only usage-ledger opening for doctor inference; v1 §§8.2, 13.
use super::{Connection, Store, StoreLimits, diagnostics, path};
use rusqlite::OpenFlags;
use std::{
    path::Path,
    sync::{Mutex, atomic::AtomicUsize},
};

/// Static refusals never expose database content or arbitrary SQLite diagnostics.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("Usage database path is absent, changed or unsafe")]
    Path,
    #[error("Usage database is empty, corrupt or incompatible; index explicitly before probing")]
    Unavailable,
    #[error("Existing usage database could not be opened without creation")]
    Open,
    #[error("Existing usage database connection policy could not be configured")]
    Policy,
}

/// Opens an existing compatible ledger with normal caps, without creating a DB,
/// directories, migrations, defaults, catalogue metadata or vector bindings.
/// `root` must be canonical; the database must stay under its owned .deepkit.
/// SQLite may maintain WAL/SHM sidecars. Run off the async runtime; callers must
/// still revalidate effective settings inside each inference admission transaction.
pub(crate) fn open_existing_for_usage(
    root: &Path,
    database: &Path,
    limits: StoreLimits,
) -> Result<Store, Error> {
    open_checked(root, database, limits, || {})
}

fn open_checked(
    root: &Path,
    database: &Path,
    limits: StoreLimits,
    boundary: impl FnOnce(),
) -> Result<Store, Error> {
    let guard = path::prepare(root, database).map_err(|_| Error::Path)?;
    guard.check_existing_database().map_err(|_| Error::Path)?;
    let database = path::system_path(&root.join(database));
    if std::fs::symlink_metadata(&database)
        .map_err(|_| Error::Path)?
        .len()
        == 0
    {
        return Err(Error::Unavailable);
    }
    // This ordinary read-only diagnostic honors live WAL data. Incompatible
    // databases are rejected before any write-capable connection is opened.
    let report = diagnostics::inspect(&database).map_err(|_| Error::Unavailable)?;
    if !report.database_present
        || report.integrity.state != diagnostics::State::Passed
        || report.schema.check.state != diagnostics::State::Passed
    {
        return Err(Error::Unavailable);
    }
    guard.check_existing_database().map_err(|_| Error::Path)?;
    boundary();
    guard.check_existing_database().map_err(|_| Error::Path)?;
    let vfs = sqlite_vector_sys::register_wal_vfs().map_err(|_| Error::Open)?;
    let flags = OpenFlags::SQLITE_OPEN_READ_WRITE
        | OpenFlags::SQLITE_OPEN_NO_MUTEX
        | OpenFlags::SQLITE_OPEN_NOFOLLOW;
    let connect = || {
        guard.check_existing_database().map_err(|_| Error::Path)?;
        let connection =
            Connection::open_with_flags_and_vfs(&database, flags, vfs).map_err(|_| Error::Open)?;
        connection
            .busy_timeout(std::time::Duration::from_secs(2))
            .map_err(|_| Error::Policy)?;
        // Recheck the opened handle before journal-mode or other connection setup.
        validate(&connection)?;
        guard.check_existing_database().map_err(|_| Error::Path)?;
        super::configure_connection(&connection, limits, false).map_err(|_| Error::Policy)?;
        Ok::<_, Error>(connection)
    };
    let writer = connect()?;
    let mut readers = Vec::with_capacity(super::READ_POOL_SIZE);
    for _ in 0..super::READ_POOL_SIZE {
        readers.push(Mutex::new(connect()?));
    }
    guard.check_existing_database().map_err(|_| Error::Path)?;
    Ok(Store {
        database_authority: Some(guard),
        db_max_bytes: limits.db_max_bytes,
        wal_max_bytes: limits.wal_max_bytes,
        max_l2_units: limits.max_l2_units,
        max_l3_units: limits.max_l3_units,
        writer: Mutex::new(writer),
        readers,
        next_reader: AtomicUsize::new(0),
    })
}

fn validate(connection: &Connection) -> Result<(), Error> {
    let transaction = connection
        .unchecked_transaction()
        .map_err(|_| Error::Unavailable)?;
    if super::schema_version(&transaction).map_err(|_| Error::Unavailable)?
        != super::LATEST_SCHEMA_VERSION
    {
        return Err(Error::Unavailable);
    }
    for name in [
        "usage",
        "usage_reservations",
        "provider_admission",
        "meta",
        "jobs",
    ] {
        let ordinary: bool = transaction
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_schema WHERE type='table' AND name=?1)",
                [name],
                |row| row.get(0),
            )
            .map_err(|_| Error::Unavailable)?;
        if !ordinary {
            return Err(Error::Unavailable);
        }
    }
    // Preparing SELECTs verifies all columns touched by source-free usage
    // reserve/dispatch/reconcile, without changing data or initializing schema.
    for sql in [
        "SELECT day,provider,calls,tokens,estimated_tokens FROM usage LIMIT 0",
        "SELECT id,provider,day,job_id,lease_generation,calls,tokens,state,created_at,dispatched_at,endpoint_kind,endpoint_url,sent_units FROM usage_reservations LIMIT 0",
        "SELECT provider,request_window,requests,token_window,tokens FROM provider_admission LIMIT 0",
        "SELECT key,value FROM meta LIMIT 0",
        "SELECT id FROM jobs LIMIT 0",
    ] {
        transaction.prepare(sql).map_err(|_| Error::Unavailable)?;
    }
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::queue::usage;
    use std::fs;

    fn fixture() -> (tempfile::TempDir, std::path::PathBuf) {
        let root = tempfile::tempdir().unwrap();
        let database = root.path().join(".deepkit/code.db");
        drop(super::super::open(&database, StoreLimits::default()).unwrap());
        (root, database)
    }

    #[test]
    fn missing_empty_corrupt_and_incompatible_databases_are_unchanged() {
        let missing = tempfile::tempdir().unwrap();
        assert!(
            open_existing_for_usage(
                missing.path(),
                Path::new(".deepkit/code.db"),
                StoreLimits::default()
            )
            .is_err()
        );
        assert_eq!(fs::read_dir(missing.path()).unwrap().count(), 0);
        for contents in [b"".as_slice(), b"not sqlite".as_slice()] {
            let root = tempfile::tempdir().unwrap();
            fs::create_dir(root.path().join(".deepkit")).unwrap();
            let database = root.path().join(".deepkit/code.db");
            fs::write(&database, contents).unwrap();
            assert!(
                open_existing_for_usage(root.path(), &database, StoreLimits::default()).is_err()
            );
            assert_eq!(fs::read(&database).unwrap(), contents);
            assert_eq!(fs::read_dir(database.parent().unwrap()).unwrap().count(), 1);
        }
        for modification in [
            "UPDATE meta SET value='1' WHERE key='schema_version'",
            "DROP TABLE provider_admission",
        ] {
            let (root, database) = fixture();
            let connection = Connection::open(&database).unwrap();
            connection.execute_batch(modification).unwrap();
            drop(connection);
            let before = fs::read(&database).unwrap();
            assert!(
                open_existing_for_usage(root.path(), &database, StoreLimits::default()).is_err()
            );
            assert_eq!(fs::read(&database).unwrap(), before);
        }
    }

    #[test]
    fn changed_or_linked_database_is_refused_without_creation() {
        for replace in [false, true] {
            let (root, database) = fixture();
            let saved = database.with_extension("saved");
            let result = open_checked(root.path(), &database, StoreLimits::default(), || {
                fs::rename(&database, &saved).unwrap();
                if replace {
                    fs::write(&database, "replacement").unwrap();
                }
            });
            assert!(matches!(result, Err(Error::Path)));
            if replace {
                assert_eq!(fs::read_to_string(&database).unwrap(), "replacement");
            } else {
                assert!(!database.exists());
            }
        }
        for hard in [false, true] {
            let (root, database) = fixture();
            let saved = database.with_extension("saved");
            fs::rename(&database, &saved).unwrap();
            if hard {
                fs::hard_link(&saved, &database).unwrap();
            } else {
                std::os::unix::fs::symlink(&saved, &database).unwrap();
            }
            assert!(matches!(
                open_existing_for_usage(root.path(), &database, StoreLimits::default()),
                Err(Error::Path)
            ));
        }
    }

    #[test]
    fn existing_ledger_accounts_under_normal_caps_without_catalogue_setup() {
        let (root, database) = fixture();
        let before = fs::read(&database).unwrap();
        let limits = StoreLimits {
            db_max_bytes: 32 * 1024 * 1024,
            wal_max_bytes: 4 * 1024 * 1024,
            ..Default::default()
        };
        let store = open_existing_for_usage(root.path(), &database, limits).unwrap();
        assert_eq!(store.db_max_bytes, limits.db_max_bytes);
        assert_eq!(store.wal_max_bytes, limits.wal_max_bytes);
        assert_eq!(store.writer().unwrap().total_changes(), 0);
        assert_eq!(fs::read(&database).unwrap(), before);
        let schema: Vec<String> = store
            .with_reader(|connection| {
                connection
                    .prepare("SELECT name FROM sqlite_schema ORDER BY name")?
                    .query_map([], |row| row.get(0))?
                    .collect::<Result<_, _>>()
                    .map_err(super::super::Error::from)
            })
            .unwrap();
        let budget = usage::Budget {
            calls: Some(1),
            tokens: Some(100),
        };
        let reservation =
            match usage::reserve(&store, "embed:remote:fixture", 4, 0, budget, None, || 86400)
                .unwrap()
            {
                usage::Admission::Reserved(value) => value,
                _ => panic!("expected admitted usage"),
            };
        assert!(matches!(
            usage::reserve(&store, "embed:remote:fixture", 4, 0, budget, None, || 86400).unwrap(),
            usage::Admission::BudgetExhausted { .. }
        ));
        usage::release(&store, &reservation).unwrap();
        let after: Vec<String> = store
            .with_reader(|connection| {
                connection
                    .prepare("SELECT name FROM sqlite_schema ORDER BY name")?
                    .query_map([], |row| row.get(0))?
                    .collect::<Result<_, _>>()
                    .map_err(super::super::Error::from)
            })
            .unwrap();
        assert_eq!(schema, after);
        assert_eq!(store.counts().unwrap().files, 0);
    }
    #[test]
    fn schema_changed_after_read_only_preflight_is_rejected_without_migration() {
        let (root, database) = fixture();
        let result = open_checked(root.path(), &database, StoreLimits::default(), || {
            let connection = Connection::open(&database).unwrap();
            connection
                .execute("UPDATE meta SET value='1' WHERE key='schema_version'", [])
                .unwrap();
        });
        assert!(matches!(result, Err(Error::Unavailable)));
        let connection =
            Connection::open_with_flags(&database, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        assert_eq!(super::super::schema_version(&connection).unwrap(), 1);
    }

    #[test]
    fn lowered_main_cap_refuses_new_usage_growth_without_deleting_existing_data() {
        let (root, database) = fixture();
        let store = open_existing_for_usage(
            root.path(),
            &database,
            StoreLimits {
                db_max_bytes: 1,
                ..Default::default()
            },
        )
        .unwrap();
        // Existing pages may still accept a small row. Force allocation beyond
        // current pages to verify the lowered cap without forbidding in-place work.
        let provider = format!("embed:remote:{}", "x".repeat(2 * 1024 * 1024));
        assert!(
            usage::reserve(
                &store,
                &provider,
                4,
                0,
                usage::Budget {
                    calls: None,
                    tokens: None,
                },
                None,
                || 86400
            )
            .is_err()
        );
        let count: i64 = store
            .writer()
            .unwrap()
            .query_row("SELECT COUNT(*) FROM usage_reservations", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 0);
        assert_eq!(
            super::super::schema_version(&store.writer().unwrap()).unwrap(),
            super::super::LATEST_SCHEMA_VERSION
        );
    }
}
