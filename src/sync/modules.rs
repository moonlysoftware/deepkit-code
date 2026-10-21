//! Publish deterministic module units after the graph is coherent; v1.md §6.6.
use super::l1::Work;
use crate::{
    config::Config,
    store::{self, FtsColumns, Store, UnitInsert, UnitKind},
    treat::{
        self, context,
        fts::{FtsInput, build_columns},
    },
};
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error(transparent)]
    Store(#[from] store::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

/// Counts published modules and explicit incomplete-source or capacity deferrals.
#[derive(Default, serde::Serialize)]
pub(crate) struct Report {
    pub(crate) state: String,
    pub(crate) selected: usize,
    pub(crate) inserted: usize,
    pub(crate) unchanged: usize,
    pub(crate) removed: usize,
    pub(crate) deferred: usize,
}

/// Uses the same publication path for full indexes and watcher/lazy source repair.
pub(crate) fn ingest(
    root: &Path,
    settings: &Config,
    store: &Store,
    work: Work<'_>,
    validate: &dyn Fn() -> Result<(), store::Error>,
) -> Result<Report, Error> {
    let head = super::head::read(root, work).map_err(|_| store::Error::L1InputsChanged)?;
    let poll = || work.check().map_err(|_| store::Error::L1InputsChanged);
    let check = || {
        poll()?;
        validate()?;
        if super::head::read(root, work).map_err(|_| store::Error::L1InputsChanged)? != head {
            return Err(store::Error::L1InputsChanged);
        }
        Ok(())
    };
    let mut report = Report {
        state: "completed".into(),
        ..Report::default()
    };
    match store.retire_changed_modules(settings.enrich.modules_min_files, &poll, &check) {
        Ok(count) => report.removed = count,
        Err(store::Error::DatabaseGrowthLimit | store::Error::WalCapacity { .. }) => {
            report.deferred += 1;
            report.state = "capped".into();
        }
        Err(error) => return Err(error.into()),
    }
    let mut batch = store.capture_modules(settings.enrich.modules_min_files, &poll, &check)?;
    let snapshots = std::mem::take(&mut batch.snapshots);
    report.selected = snapshots.len();
    let repo_name = super::l1::repo_name(root, settings);
    for (path, captured) in snapshots {
        let Some(captured) = captured else {
            report.deferred += 1;
            report.state = "incomplete".into();
            continue;
        };
        let collated = treat::module::collate(&captured.input);
        let breadcrumb = context::breadcrumb(
            context::ContextFile {
                repo_name: &repo_name,
                path: &path,
            },
            None,
            &[],
            true,
        );
        let columns = build_columns(
            FtsInput {
                breadcrumb: &breadcrumb,
                header: &collated.header,
                text: &collated.text,
                language: None,
                kind: store::FileKind::Doc,
                extra_idents: &[],
                has_code_fence: false,
            },
            &settings.tokenize,
        );
        let key = format!("module:{path}");
        let now = super::unix_timestamp();
        let unit = UnitInsert {
            hash: &collated.hash,
            source_key: Some(&key),
            duplicate_ordinal: 0,
            file_id: None,
            kind: UnitKind::Module,
            byte_start: None,
            byte_end: None,
            line_start: None,
            line_end: None,
            scope_path: Some(&breadcrumb),
            header: &collated.header,
            text: &collated.text,
            tokens: treat::count_tokens(&collated.text) as i64,
            ladder_level: 1,
            parent_unit: None,
            created_at: now,
            updated_at: now,
            fts: FtsColumns {
                header: &columns.header,
                idents: &columns.idents,
                body: &columns.body,
                prose: &columns.prose,
            },
        };
        match store.publish_module(
            &mut batch,
            &captured,
            settings.enrich.modules_min_files,
            unit,
            &poll,
            &check,
        ) {
            Ok(true) => report.inserted += 1,
            Ok(false) => report.unchanged += 1,
            Err(store::Error::DatabaseGrowthLimit | store::Error::WalCapacity { .. }) => {
                report.deferred += 1;
                report.state = "capped".into();
                break;
            }
            Err(error) => return Err(error.into()),
        }
    }
    let encoded = serde_json::to_string(&report)?;
    store::admission::transaction_checked(
        store,
        store::admission::Kind::Control,
        Some(&poll),
        |tx| {
            check()?;
            tx.execute("INSERT INTO meta(key,value) VALUES ('modules_status',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[encoded])?;
            check()?;
            tx.commit()?;
            Ok::<(), store::Error>(())
        },
    )?;
    Ok(report)
}
