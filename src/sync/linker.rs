//! Shared full-index and lazy-repair relationship publication; v1.md §6.5.
use super::l1::Work;
use crate::{
    config::Config,
    store::{self, Store},
    treat,
};
use rusqlite::OptionalExtension;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error(transparent)]
    Store(#[from] store::Error),
    #[error(transparent)]
    Glob(#[from] globset::Error),
}
/// Distinguishes relationship deferral from completed per-file L1 publication.
#[derive(Debug, serde::Serialize)]
pub(crate) struct Report {
    pub(crate) state: &'static str,
    pub(crate) changed: bool,
    pub(crate) deferred: bool,
    pub(crate) reason: Option<&'static str>,
}
/// Publishes deterministic links using the request's configuration and HEAD fence.
pub(crate) fn ingest(
    root: &Path,
    settings: &Config,
    store: &Store,
    work: Work<'_>,
    validate: &dyn Fn() -> Result<(), store::Error>,
) -> Result<Report, Error> {
    let tests = treat::linker::test_paths(&settings.tests.globs)?;
    let head = super::head::read(root, work).map_err(|_| store::Error::L1InputsChanged)?;
    let poll = || work.check().map_err(|_| store::Error::L1InputsChanged);
    let result = store.link_documents(&tests, &poll, &|| {
        poll()?;
        validate()?;
        if super::head::read(root, work).map_err(|_| store::Error::L1InputsChanged)? != head {
            return Err(store::Error::L1InputsChanged);
        }
        Ok(())
    });
    match result {
        Ok(changed) => Ok(Report {
            state: "completed",
            changed,
            deferred: false,
            reason: None,
        }),
        Err(store::Error::DatabaseGrowthLimit) => Ok(Report {
            state: "capped",
            changed: false,
            deferred: true,
            reason: Some("db_max_bytes"),
        }),
        Err(store::Error::WalCapacity { .. }) => Ok(Report {
            state: "capped",
            changed: false,
            deferred: true,
            reason: Some("wal_max_bytes"),
        }),
        Err(store::Error::Sqlite(error))
            if error.sqlite_error_code() == Some(rusqlite::ErrorCode::DiskFull) =>
        {
            Ok(Report {
                state: "capped",
                changed: false,
                deferred: true,
                reason: Some("db_max_bytes"),
            })
        }
        Err(error) => Err(error.into()),
    }
}

/// Records whole-index availability without turning query subset work into progress.
/// Capacity rollback retains old links; status explains why new links are deferred.
pub(crate) fn ingest_index(
    root: &Path,
    settings: &Config,
    store: &Store,
    work: Work<'_>,
    validate: &dyn Fn() -> Result<(), store::Error>,
) -> Result<Report, Error> {
    let head = super::head::read(root, work).map_err(|_| store::Error::L1InputsChanged)?;
    let report = ingest(root, settings, store, work, validate)?;
    let check = || {
        if super::head::read(root, work).map_err(|_| store::Error::L1InputsChanged)? != head {
            return Err(store::Error::L1InputsChanged);
        }
        work.check().map_err(|_| store::Error::L1InputsChanged)?;
        validate()
    };
    let value =
        serde_json::json!({"state":report.state,"deferred":report.deferred,"reason":report.reason})
            .to_string();
    let old = store.checked_read_snapshot(&check, |tx| {
        Ok::<_, store::Error>(
            tx.query_row(
                "SELECT value FROM meta WHERE key='linker_status'",
                [],
                |r| r.get::<_, String>(0),
            )
            .optional()?,
        )
    })??;
    if old.as_ref() != Some(&value) {
        store::admission::transaction_checked(
            store,
            store::admission::Kind::Control,
            Some(&check),
            |tx| {
                tx.execute("INSERT INTO meta(key,value) VALUES ('linker_status',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[&value])?;
                check()?;
                tx.commit()?;
                Ok::<_, store::Error>(())
            },
        )?;
    }
    Ok(report)
}
