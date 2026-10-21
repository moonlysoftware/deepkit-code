//! Read-only existing-database diagnostics without migration; v1 §13 and §16.
use rusqlite::{Connection, OpenFlags};
use serde::Serialize;
use std::{fs, path::Path, time::Duration};

/// Setup failures intentionally carry neither raw SQL errors nor database contents.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("Database destination could not be inspected")]
    Metadata,
    #[error("Database destination is not an ordinary regular file")]
    NotRegular,
    #[error("Existing database could not be opened read-only")]
    Open,
    #[error("Read-only database diagnostic connection could not be configured")]
    Connection,
}

/// No check is inferred healthy when its database could not be examined.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum State {
    Passed,
    Failed,
    NotChecked,
}

/// A static one-line diagnosis, safe for text, JSON and persistent logs.
#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct Check {
    pub(crate) state: State,
    pub(crate) message: &'static str,
}
impl Check {
    fn passed(message: &'static str) -> Self {
        Self {
            state: State::Passed,
            message,
        }
    }
    fn failed(message: &'static str) -> Self {
        Self {
            state: State::Failed,
            message,
        }
    }
    fn absent() -> Self {
        Self {
            state: State::NotChecked,
            message: "Database is absent; check not performed",
        }
    }
}

/// Schema compatibility is independent from SQLite page integrity.
#[derive(Debug, Serialize)]
pub(crate) struct Schema {
    pub(crate) check: Check,
    /// Numeric schema metadata only; malformed stored values are never copied here.
    pub(crate) observed: Option<i64>,
    pub(crate) supported: i64,
}

/// Filesystem sizes and SQLite page accounting are observations, not reservations.
/// Concurrent writers may change file sizes independently of the read transaction.
#[derive(Debug, Serialize)]
pub(crate) struct Storage {
    pub(crate) database_bytes: u64,
    pub(crate) database_allocated_bytes: Option<u64>,
    /// None means sidecar metadata was unavailable or not an ordinary regular file.
    pub(crate) wal_bytes: Option<u64>,
    pub(crate) wal_allocated_bytes: Option<u64>,
    pub(crate) page_count: Option<u64>,
    pub(crate) page_size: Option<u64>,
    pub(crate) freelist_count: Option<u64>,
}

/// Independent local checks of one read transaction; no repairs or index initialization.
#[derive(Debug, Serialize)]
pub(crate) struct Report {
    pub(crate) database_present: bool,
    pub(crate) storage: Option<Storage>,
    pub(crate) integrity: Check,
    pub(crate) schema: Schema,
    /// Availability in this diagnostic connection, not historical index configuration.
    pub(crate) sqlite_vector: Check,
    /// FTS5 module availability in this diagnostic connection.
    pub(crate) fts5: Check,
    /// Existing lexical virtual tables can prepare MATCH queries without recreation.
    pub(crate) lexical_tables: Check,
    /// Read-only connections may still participate in SQLite WAL shared-memory bookkeeping.
    pub(crate) wal_sidecars: &'static str,
}
impl Report {
    fn missing() -> Self {
        Self {
            database_present: false,
            storage: None,
            integrity: Check::absent(),
            schema: Schema {
                check: Check::absent(),
                observed: None,
                supported: super::LATEST_SCHEMA_VERSION,
            },
            sqlite_vector: Check::absent(),
            fts5: Check::absent(),
            lexical_tables: Check::absent(),
            wal_sidecars: "not_opened",
        }
    }
}

/// Inspects an existing file without CREATE flags, migrations, vector-table initialization,
/// schema repairs, checkpointing or application-data writes. Missing parents remain absent.
/// Uses ordinary read-only SQLite semantics so committed WAL data remains visible; SQLite
/// may create/use/update -shm bookkeeping (and requires suitable WAL-sidecar access). This
/// API does not promise unchanged sidecar bytes. Do not replace it with immutable=1: that
/// would ignore live WAL state. Concurrent writers may progress independently.
/// Integrity_check can scan the database; run off the async runtime if called from tools.
pub(crate) fn inspect(path: &Path) -> Result<Report, Error> {
    inspect_with(path, |_, _| ()).map(|(report, _)| report)
}

/// Adds a caller-owned read observation in the same transaction, only after
/// positive integrity and compatible schema checks. The callback never owns the
/// connection/transaction and must not write, change policy or commit it.
pub(crate) fn inspect_with<T>(
    path: &Path,
    observe: impl FnOnce(&rusqlite::Transaction<'_>, &Report) -> T,
) -> Result<(Report, Option<T>), Error> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(value) => value,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok((Report::missing(), None));
        }
        Err(_) => return Err(Error::Metadata),
    };
    if !metadata.file_type().is_file() {
        return Err(Error::NotRegular);
    }
    let registered = super::register_vector_extension().is_ok();
    let open_path = super::path::system_path(path);
    let connection = Connection::open_with_flags(
        &open_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY
            | OpenFlags::SQLITE_OPEN_NO_MUTEX
            | OpenFlags::SQLITE_OPEN_NOFOLLOW,
    )
    .map_err(|_| Error::Open)?;
    connection
        .busy_timeout(Duration::from_secs(2))
        .map_err(|_| Error::Connection)?;
    connection
        .pragma_update(None, "query_only", true)
        .map_err(|_| Error::Connection)?;
    // BEGIN is deferred/read-only; no writes or schema access are needed to begin.
    let snapshot = connection
        .unchecked_transaction()
        .map_err(|_| Error::Connection)?;
    let integrity = integrity(&snapshot);
    let schema = match super::schema_version(&snapshot) {
        Ok(observed) => Schema {
            observed: Some(observed),
            supported: super::LATEST_SCHEMA_VERSION,
            check: if observed == super::LATEST_SCHEMA_VERSION {
                Check::passed("Database schema matches this binary")
            } else if observed > super::LATEST_SCHEMA_VERSION {
                Check::failed("Database schema is newer than this binary")
            } else {
                Check::failed("Database schema is missing or requires migration")
            },
        },
        Err(_) => Schema {
            check: Check::failed("Database schema metadata is unreadable or invalid"),
            observed: None,
            supported: super::LATEST_SCHEMA_VERSION,
        },
    };
    let vector = registered
        && snapshot
            .query_row("SELECT length(vector_version()) > 0", [], |row| {
                row.get::<_, bool>(0)
            })
            .unwrap_or(false);
    let fts5 = snapshot
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_module_list WHERE name='fts5')",
            [],
            |row| row.get::<_, bool>(0),
        )
        .unwrap_or(false);
    let lexical = fts5
        && lexical_schema(&snapshot)
        && snapshot
            .prepare("SELECT rowid FROM fts_units WHERE fts_units MATCH ?1 LIMIT 0")
            .is_ok()
        && snapshot
            .prepare("SELECT rowid FROM fts_names WHERE fts_names MATCH ?1 LIMIT 0")
            .is_ok();
    let report = Report {
        database_present: true,
        storage: Some(storage(path, &metadata, &snapshot)),
        integrity,
        schema,
        sqlite_vector: if vector {
            Check::passed("sqlite-vector is available")
        } else {
            Check::failed("sqlite-vector is unavailable")
        },
        fts5: if fts5 {
            Check::passed("FTS5 is available")
        } else {
            Check::failed("FTS5 is unavailable")
        },
        lexical_tables: if lexical {
            Check::passed("Existing lexical FTS tables are available")
        } else {
            Check::failed("Existing lexical FTS tables are missing or unreadable")
        },
        wal_sidecars: "sqlite_read_only_shared_memory_bookkeeping_possible",
    };
    let extra = (report.integrity.state == State::Passed
        && report.schema.check.state == State::Passed)
        .then(|| observe(&snapshot, &report));
    Ok((report, extra))
}

// A prepared MATCH expression also accepts ordinary tables with a same-named
// column. Require real virtual tables and the FTS5 module in their declarations.
// Inspect only the declaration prefix before its first argument list; a column,
// option or quoted string mentioning FTS5 cannot masquerade as the module.
// SQL reduces schema text to a boolean; no raw DDL reaches the report.
fn lexical_schema(connection: &Connection) -> bool {
    connection
        .query_row(
            "WITH declarations AS (
            SELECT t.name,
                lower(replace(replace(replace(replace(replace(replace(replace(replace(replace(
                    substr(s.sql, 1, instr(s.sql, '(') - 1),
                    ' ', ''), char(9), ''), char(10), ''), char(13), ''),
                    char(34), ''), char(39), ''), '`', ''), '[', ''), ']', '')) AS prefix
            FROM pragma_table_list AS t JOIN main.sqlite_schema AS s ON s.name=t.name
            WHERE t.schema='main' AND t.type='virtual'
                AND t.name IN ('fts_units', 'fts_names') AND instr(s.sql, '(')>0
        ) SELECT count(*)=2 FROM declarations
          WHERE prefix='createvirtualtable' || name || 'usingfts5'",
            [],
            |row| row.get::<_, bool>(0),
        )
        .unwrap_or(false)
}

fn allocated(metadata: &fs::Metadata) -> Option<u64> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        metadata.blocks().checked_mul(512)
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        None
    }
}
fn storage(path: &Path, database: &fs::Metadata, connection: &Connection) -> Storage {
    let mut wal_path = path.as_os_str().to_os_string();
    wal_path.push("-wal");
    let (wal_bytes, wal_allocated_bytes) = match fs::symlink_metadata(Path::new(&wal_path)) {
        Ok(metadata) if metadata.file_type().is_file() => {
            (Some(metadata.len()), allocated(&metadata))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (Some(0), Some(0)),
        _ => (None, None),
    };
    let page = |name| {
        connection
            .pragma_query_value(None, name, |row| row.get::<_, i64>(0))
            .ok()
            .and_then(|value| u64::try_from(value).ok())
    };
    Storage {
        database_bytes: database.len(),
        database_allocated_bytes: allocated(database),
        wal_bytes,
        wal_allocated_bytes,
        page_count: page("page_count"),
        page_size: page("page_size"),
        freelist_count: page("freelist_count"),
    }
}

fn integrity(connection: &Connection) -> Check {
    // Limit diagnostics with the PRAGMA form: the table-valued form treats its
    // first argument as a table name, not the maximum error count. Compare bytes
    // locally; corruption messages can contain stored names and never escape.
    let checked = (|| -> rusqlite::Result<bool> {
        let mut statement = connection.prepare("PRAGMA integrity_check(1)")?;
        let mut rows = statement.query([])?;
        let Some(first) = rows.next()? else {
            return Ok(false);
        };
        let ok = matches!(first.get_ref(0)?, rusqlite::types::ValueRef::Text(b"ok"));
        Ok(ok && rows.next()?.is_none())
    })();
    match checked {
        Ok(true) => Check::passed("SQLite integrity check passed"),
        Ok(false) => Check::failed("SQLite integrity check found corruption"),
        Err(_) => Check::failed("SQLite integrity check could not read the database"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_database_does_not_create_its_parent_or_any_files() {
        let root = tempfile::tempdir().unwrap();
        let report = inspect(&root.path().join(".deepkit/code.db")).unwrap();
        assert!(!report.database_present);
        assert_eq!(report.integrity.state, State::NotChecked);
        assert_eq!(report.schema.observed, None);
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
    }

    #[test]
    fn corrupt_database_has_safe_failure_and_keeps_original_bytes() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("code.db");
        let contents = b"private-diagnostic-fixture not a SQLite database";
        fs::write(&path, contents).unwrap();
        let inspected = inspect(&path);
        let rendered = match inspected {
            Ok(report) => {
                assert_eq!(report.integrity.state, State::Failed);
                assert_eq!(report.schema.check.state, State::Failed);
                serde_json::to_string(&report).unwrap()
            }
            Err(error) => error.to_string(),
        };
        assert!(!rendered.contains("private-diagnostic-fixture"));
        assert_eq!(fs::read(&path).unwrap(), contents);
    }

    #[test]
    fn lexical_health_refuses_ordinary_and_other_virtual_tables_without_leaking_ddl() {
        for module in [None, Some("fts4"), Some("fts5")] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join("code.db");
            let connection = Connection::open(&path).unwrap();
            connection
                .execute_batch("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);")
                .unwrap();
            connection
                .execute(
                    "INSERT INTO meta VALUES('schema_version', ?1)",
                    [super::super::LATEST_SCHEMA_VERSION.to_string()],
                )
                .unwrap();
            for table in ["fts_units", "fts_names"] {
                let sql = match module {
                    None => {
                        format!("CREATE TABLE {table}({table} TEXT, private_diagnostic_ddl TEXT)")
                    }
                    Some(module) => format!(
                        "CREATE VIRTUAL TABLE {table} USING {module}(private_diagnostic_ddl)"
                    ),
                };
                connection.execute_batch(&sql).unwrap();
                // This is precisely the old false-positive probe.
                assert!(
                    connection
                        .prepare(&format!(
                            "SELECT rowid FROM {table} WHERE {table} MATCH ?1 LIMIT 0"
                        ))
                        .is_ok()
                );
            }
            drop(connection);
            let before = fs::read(&path).unwrap();
            let report = inspect(&path).unwrap();
            assert_eq!(report.integrity.state, State::Passed);
            assert_eq!(report.schema.check.state, State::Passed);
            assert_eq!(report.fts5.state, State::Passed);
            assert_eq!(report.fts5.message, "FTS5 is available");
            assert_eq!(
                report.lexical_tables.message,
                if module == Some("fts5") {
                    "Existing lexical FTS tables are available"
                } else {
                    "Existing lexical FTS tables are missing or unreadable"
                }
            );
            assert_eq!(
                report.lexical_tables.state,
                if module == Some("fts5") {
                    State::Passed
                } else {
                    State::Failed
                }
            );
            assert!(
                !serde_json::to_string(&report)
                    .unwrap()
                    .contains("private_diagnostic_ddl")
            );
            assert_eq!(fs::read(&path).unwrap(), before);
        }
    }

    #[cfg(unix)]
    #[test]
    fn observation_refuses_parent_symlinks_even_inside_the_repository() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("real")).unwrap();
        let database = root.path().join("real/code.db");
        let _store = super::super::open(&database, Default::default()).unwrap();
        std::os::unix::fs::symlink(root.path().join("real"), root.path().join("link")).unwrap();
        assert!(matches!(
            inspect(&root.path().join("link/code.db")),
            Err(Error::Open)
        ));
    }

    #[test]
    fn healthy_observation_preserves_database_and_sees_committed_wal_schema() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("code.db");
        let store = super::super::open(&path, Default::default()).unwrap();
        let before = fs::read(&path).unwrap();
        let wal = root.path().join("code.db-wal");
        let wal_before = fs::read(&wal).unwrap();
        let report = inspect(&path).unwrap();
        assert!(report.database_present);
        assert_eq!(report.integrity.state, State::Passed);
        assert_eq!(
            report.schema.observed,
            Some(super::super::LATEST_SCHEMA_VERSION)
        );
        assert_eq!(report.schema.check.state, State::Passed);
        assert_eq!(report.sqlite_vector.state, State::Passed);
        assert_eq!(report.fts5.state, State::Passed);
        assert_eq!(report.lexical_tables.state, State::Passed);
        let storage = report.storage.as_ref().unwrap();
        assert_eq!(storage.database_bytes, before.len() as u64);
        assert_eq!(storage.wal_bytes, Some(wal_before.len() as u64));
        assert!(storage.page_count.is_some_and(|count| count > 0));
        assert_eq!(
            storage.page_size,
            Some(super::super::SQLITE_PAGE_SIZE as u64)
        );
        assert!(storage.freelist_count.is_some());
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(fs::read(&wal).unwrap(), wal_before);
        // Metadata lives in an uncheckpointed WAL update. immutable=1 would be wrong.
        store.set_meta("schema_version", "1").unwrap();
        let report = inspect(&path).unwrap();
        assert_eq!(report.schema.observed, Some(1));
        assert_eq!(report.schema.check.state, State::Failed);
        assert_eq!(
            store.get_meta("schema_version").unwrap().as_deref(),
            Some("1")
        );
    }
}
