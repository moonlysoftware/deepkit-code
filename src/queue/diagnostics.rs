//! Source-free failure categories for finite queue reports; v1.md §8 and §10.2.
use super::Error;

/// Identifies actionable failures without formatting source-bearing error fields.
/// Only enum labels, numeric SQLite codes and compile-time request reasons escape.
pub(crate) fn code(error: &Error) -> String {
    if let Some(reason) = super::capacity::reason(error) {
        return format!("capacity:{reason:?}");
    }
    match error {
        Error::Sql(error) | Error::Store(crate::store::Error::Sqlite(error)) => match error {
            rusqlite::Error::SqliteFailure(code, _) => {
                format!("sqlite:{:?}:{}", code.code, code.extended_code)
            }
            _ => "sqlite_api_error".into(),
        },
        Error::Invalid(reason) => format!("invalid:{reason}"),
        Error::Store(error) => match error {
            crate::store::Error::L1InputsChanged => "source_changed",
            crate::store::Error::L1ScanSuperseded { .. } => "scan_superseded",
            crate::store::Error::L1Incomplete { .. } => "scan_incomplete",
            crate::store::Error::FileIndexSuperseded { .. } => "file_superseded",
            crate::store::Error::ContentHashConflict => "content_hash_conflict",
            crate::store::Error::VectorFenceMismatch { .. } => "vector_fence_mismatch",
            crate::store::Error::VectorDimensionMismatch { .. } => "vector_dimension_mismatch",
            crate::store::Error::VectorTableMismatch { .. } => "vector_table_mismatch",
            crate::store::Error::VectorTableMissing { .. } => "vector_table_missing",
            crate::store::Error::LockPoisoned { .. } => "store_lock_poisoned",
            _ => "store_error",
        }
        .into(),
        Error::InputsUnavailable => "inputs_unavailable".into(),
        Error::InputPreparation => "input_preparation".into(),
        Error::PagePreparation(_) => "page_preparation".into(),
        Error::IdentityConflict => "identity_conflict".into(),
        Error::BulkIncomplete { source, .. } | Error::RecoveryIncomplete { source, .. } => {
            code(source)
        }
        Error::WorkerTask => "worker_task".into(),
        Error::WorkerStopped => "worker_stopped".into(),
        Error::Capacity(_) | Error::PageCapacity { .. } => "capacity".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categories_preserve_sqlite_cause_without_source_bearing_messages() {
        let marker = "synthetic protected source text";
        let sql = rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
            Some(marker.into()),
        );
        let error = Error::RecoveryIncomplete {
            jobs: 2,
            reservations: 1,
            source: Box::new(Error::Store(crate::store::Error::Sqlite(sql))),
        };
        assert_eq!(code(&error), "sqlite:DatabaseBusy:5");
        let error = Error::Store(crate::store::Error::VectorTableMissing {
            slug: marker.into(),
        });
        assert_eq!(code(&error), "vector_table_missing");
        assert!(!code(&error).contains(marker));
        assert_eq!(
            code(&Error::Invalid("usage accounting")),
            "invalid:usage accounting"
        );
    }
}
