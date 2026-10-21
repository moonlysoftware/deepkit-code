//! Collected doctor findings with explicit unperformed checks; v1 §13/§16.
use super::{Check, Finding};
use crate::{config::Config, store::diagnostics};
use serde::Serialize;
use std::path::Path;

/// Coverage is independent of health: diagnosed failures may still be complete.
#[derive(Serialize)]
pub(super) struct Report {
    pub(super) scope: &'static str,
    pub(super) status: &'static str,
    pub(super) complete: bool,
    pub(super) checks: Vec<Finding>,
    pub(super) local_reranker: Option<Check>,
}

/// Converts only static diagnoses and numeric allocation evidence to user findings.
pub(super) fn database(root: &Path, config: &Config) -> Vec<Finding> {
    let Ok(repository) = std::fs::canonicalize(root.join(&config.index.root)) else {
        return vec![Finding::new(
            "database.open",
            "failed",
            "Configured repository root is unavailable",
        )];
    };
    let path = super::super::database_path(&repository, config);
    if crate::store::path::prepare(&repository, &path).is_err() {
        return vec![Finding::new(
            "database.open",
            "failed",
            "Database or sidecar path is outside .deepkit or unsafe",
        )];
    }
    let observed_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|elapsed| i64::try_from(elapsed.as_secs()).ok());
    let (report, observation) = match diagnostics::inspect_with(&path, |transaction, _| {
        let queue = observed_at
            .ok_or(crate::queue::Error::Invalid(
                "diagnostic timestamp unavailable",
            ))
            .and_then(|now| crate::queue::status::read_in(transaction, now, None))
            .and_then(|observation| observation.health());
        (queue, super::tables::inspect(transaction, config))
    }) {
        Ok(observation) => observation,
        Err(error) => return vec![Finding::new("database.open", "failed", &error.to_string())],
    };
    let mut findings = vec![Finding::new(
        "database.present",
        if report.database_present {
            "passed"
        } else {
            "warning"
        },
        if report.database_present {
            "Existing index database found"
        } else {
            "Index database is absent; run dkc index or serve"
        },
    )];
    for (id, check) in [
        ("database.integrity", report.integrity),
        ("database.schema", report.schema.check),
        ("database.sqlite_vector", report.sqlite_vector),
        ("database.fts5", report.fts5),
        ("database.lexical_tables", report.lexical_tables),
    ] {
        findings.push(Finding::new(
            id,
            match check.state {
                diagnostics::State::Passed => "passed",
                diagnostics::State::Failed => "failed",
                diagnostics::State::NotChecked => "not_checked",
            },
            check.message,
        ));
    }
    let (queue, tables) =
        observation.map_or((None, None), |(queue, tables)| (Some(queue), Some(tables)));
    findings.push(tables.unwrap_or_else(super::tables::unavailable));
    match queue {
        Some(Ok(health)) => {
            let attention = health.jobs.failed > 0
                || health.jobs.dead > 0
                || health.jobs.expired_leases > 0
                || !health.deferrals.is_empty()
                || health.uncertain_calls > 0
                || health.uncertain_tokens > 0;
            let mut finding = Finding::new(
                "queue.health",
                if attention { "warning" } else { "passed" },
                if attention {
                    "Queue contains failures, delays, expired leases or uncertain charges; inspect queue status"
                } else {
                    "Queue snapshot has no observed failures or expired leases; worker liveness is not inferred"
                },
            );
            finding.evidence = serde_json::to_value(health).ok();
            findings.push(finding);
        }
        _ => findings.push(Finding::new(
            "queue.health",
            "not_checked",
            "Queue health requires a readable compatible database and valid persisted counters",
        )),
    }
    if let Some(storage) = report.storage {
        let limits = config
            .limits
            .db_max_bytes()
            .and_then(|database| config.limits.wal_max_bytes().map(|wal| (database.0, wal.0)));
        if let Ok((database_limit, wal_limit)) = limits {
            let database_bytes = storage
                .page_count
                .zip(storage.page_size)
                .and_then(|(pages, size)| pages.checked_mul(size));
            let exceeded = database_bytes.is_some_and(|bytes| bytes > database_limit)
                || storage.database_bytes > database_limit
                || storage.wal_bytes.is_some_and(|bytes| bytes > wal_limit);
            let unknown = database_bytes.is_none() || storage.wal_bytes.is_none();
            let mut finding = Finding::new(
                "database.caps",
                if exceeded {
                    "warning"
                } else if unknown {
                    "not_checked"
                } else {
                    "passed"
                },
                if exceeded {
                    "Existing database or WAL exceeds configured cap; review limits or explicit shrink"
                } else if unknown {
                    "Some database or WAL allocation measurements are unavailable"
                } else {
                    "Observed database and WAL sizes fit configured caps"
                },
            );
            finding.evidence = Some(
                serde_json::json!({"storage": storage, "database_limit_bytes": database_limit, "wal_limit_bytes": wal_limit}),
            );
            findings.push(finding);
        }
    }
    findings
}

/// Retains destination identities across hashing/smoke before a separate health write.
/// The SQLite open still uses pathnames: this is a rechecked preflight, not atomic CAS.
pub(super) struct Persistence {
    root: std::path::PathBuf,
    path: std::path::PathBuf,
    destination: crate::store::path::Prepared,
}
impl Persistence {
    /// Captures only an existing destination after positive schema/integrity diagnosis.
    pub(super) fn capture(root: &Path, config: &Config) -> Option<Self> {
        let root = std::fs::canonicalize(root.join(&config.index.root)).ok()?;
        let path = super::super::database_path(&root, config);
        if !path.is_file() {
            return None;
        }
        let destination = crate::store::path::prepare(&root, &path).ok()?;
        Some(Self {
            root,
            path,
            destination,
        })
    }

    /// Refuses replacement, corruption or schema changes before the health writer opens.
    pub(super) fn open(&self, config: &Config) -> Option<crate::store::Store> {
        self.destination.check().ok()?;
        let current = diagnostics::inspect(&self.path).ok()?;
        if current.integrity.state != diagnostics::State::Passed
            || current.schema.check.state != diagnostics::State::Passed
        {
            return None;
        }
        self.destination.check().ok()?;
        super::super::open_store(&self.root, config).ok()
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::store;
    use std::fs;

    #[test]
    fn health_write_refuses_replaced_database_ancestor_or_sidecar() {
        for change in ["database", "ancestor", "sidecar"] {
            let root = tempfile::tempdir().unwrap();
            let config = Config::default();
            let path = root.path().join(".deepkit/code.db");
            let store = store::open(&path, Default::default()).unwrap();
            drop(store);
            let guard = Persistence::capture(root.path(), &config).unwrap();
            let bytes = fs::read(&path).unwrap();
            let outside = tempfile::tempdir().unwrap();
            let sentinel = outside.path().join("kept");
            fs::write(&sentinel, "untouched").unwrap();
            match change {
                "database" => {
                    fs::rename(&path, root.path().join("original.db")).unwrap();
                    fs::write(&path, &bytes).unwrap();
                }
                "ancestor" => {
                    fs::rename(root.path().join(".deepkit"), root.path().join("original")).unwrap();
                    fs::create_dir(root.path().join(".deepkit")).unwrap();
                    fs::write(&path, &bytes).unwrap();
                }
                _ => {
                    std::os::unix::fs::symlink(&sentinel, root.path().join(".deepkit/code.db-wal"))
                        .unwrap()
                }
            }
            assert!(guard.open(&config).is_none());
            assert_eq!(fs::read(&path).unwrap(), bytes);
            assert_eq!(fs::read(&sentinel).unwrap(), b"untouched");
        }
    }

    #[test]
    fn health_write_rechecks_schema_without_migrating_and_accepts_unchanged_identity() {
        let root = tempfile::tempdir().unwrap();
        let config = Config::default();
        let store = store::open(root.path().join(".deepkit/code.db"), Default::default()).unwrap();
        let guard = Persistence::capture(root.path(), &config).unwrap();
        assert!(guard.open(&config).is_some());
        store.set_meta("schema_version", "1").unwrap();
        assert!(guard.open(&config).is_none());
        assert_eq!(
            store.get_meta("schema_version").unwrap().as_deref(),
            Some("1")
        );
    }
}
