//! Materialize ranked symbol histories after commits and PRs; v1.md §6.6.
use super::l1::Work;
use crate::{
    config::Config,
    store::{self, FtsColumns, Store, UnitInsert, UnitKind},
    treat::{
        self, context,
        fts::{FtsInput, LexicalColumns, build_columns},
    },
};
use std::{collections::BTreeSet, path::Path};
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error(transparent)]
    Store(#[from] store::Error),
    #[error(transparent)]
    Collation(#[from] treat::symbol_history::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
/// Counts actual publications and visible capacity deferrals, never inferred coverage.
#[derive(Default, serde::Serialize)]
pub(crate) struct Report {
    pub(crate) state: String,
    pub(crate) selected: usize,
    pub(crate) migration_selected: usize,
    pub(crate) inserted: usize,
    pub(crate) unchanged: usize,
    pub(crate) deferred: usize,
}
const PUBLICATION_BATCH: usize = 32;

struct PendingHistory<'a> {
    target: &'a store::symbol_history::Target,
    entries: &'a [store::symbol_history::Entry],
    hash: [u8; 32],
    key: String,
    breadcrumb: String,
    header: String,
    text: String,
    tokens: i64,
    columns: LexicalColumns,
    now: i64,
}

impl PendingHistory<'_> {
    fn publication(&self) -> store::symbol_history::Publication<'_> {
        store::symbol_history::Publication {
            target: self.target,
            entries: self.entries,
            unit: UnitInsert {
                hash: &self.hash,
                source_key: Some(&self.key),
                duplicate_ordinal: 0,
                file_id: None,
                kind: UnitKind::History,
                byte_start: None,
                byte_end: None,
                line_start: None,
                line_end: None,
                scope_path: Some(&self.breadcrumb),
                header: &self.header,
                text: &self.text,
                tokens: self.tokens,
                ladder_level: 1,
                parent_unit: Some(self.target.parent),
                created_at: self.now,
                updated_at: self.now,
                fts: FtsColumns {
                    header: &self.columns.header,
                    idents: &self.columns.idents,
                    body: &self.columns.body,
                    prose: &self.columns.prose,
                },
            },
        }
    }
}

fn publish_pending(
    pending: &mut Vec<PendingHistory<'_>>,
    store: &Store,
    report: &mut Report,
    poll: &dyn Fn() -> Result<(), store::Error>,
    validate: &dyn Fn() -> Result<(), store::Error>,
) -> Result<bool, Error> {
    if pending.is_empty() {
        return Ok(false);
    }
    let batch_result = {
        let publications = pending
            .iter()
            .map(PendingHistory::publication)
            .collect::<Vec<_>>();
        store.publish_symbol_histories(&publications, poll, validate)
    };
    match batch_result {
        Ok(()) => {
            report.inserted += pending.len();
            pending.clear();
            Ok(false)
        }
        Err(store::Error::DatabaseGrowthLimit | store::Error::WalCapacity { .. }) => {
            for history in pending.iter() {
                let publication = history.publication();
                match store.publish_symbol_history(
                    publication.target,
                    publication.entries,
                    publication.unit,
                    poll,
                    validate,
                ) {
                    Ok(true) => report.inserted += 1,
                    Ok(false) => report.unchanged += 1,
                    Err(store::Error::DatabaseGrowthLimit | store::Error::WalCapacity { .. }) => {
                        report.deferred += 1;
                        report.state = "capped".into();
                        return Ok(true);
                    }
                    Err(error) => return Err(error.into()),
                }
            }
            pending.clear();
            Ok(false)
        }
        Err(error) => Err(error.into()),
    }
}
/// Runs after graph/history publication, using the original HEAD/configuration fence.
pub(crate) fn ingest(
    root: &Path,
    settings: &Config,
    store: &Store,
    work: Work<'_>,
    validate: &dyn Fn() -> Result<(), store::Error>,
) -> Result<Report, Error> {
    ingest_selected(root, settings, store, work, validate, None)
}

/// Repairs only the queried definitions after lazy L1/branch reconciliation.
/// Uses the same source set, admission and precommit validation as full indexing;
/// does not overwrite whole-index progress with a query's subset counts.
pub(crate) fn repair_selected(
    root: &Path,
    settings: &Config,
    store: &Store,
    work: Work<'_>,
    validate: &dyn Fn() -> Result<(), store::Error>,
    selected: &BTreeSet<i64>,
) -> Result<(), Error> {
    ingest_selected(root, settings, store, work, validate, Some(selected))?;
    Ok(())
}

fn ingest_selected(
    root: &Path,
    settings: &Config,
    store: &Store,
    work: Work<'_>,
    validate: &dyn Fn() -> Result<(), store::Error>,
    selected: Option<&BTreeSet<i64>>,
) -> Result<Report, Error> {
    let mut report = Report::default();
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
    if !settings.commits.enabled {
        report.state = "disabled".into();
        if selected.is_none() {
            publish_status(store, &report, &check)?;
        }
        return Ok(report);
    }
    report.state = "completed".into();
    let (targets, entries, ranked) = store.checked_read_snapshot(&check, |tx| {
        let mut targets = store::symbol_history::targets(tx, settings.enrich.top_symbols)?;
        if let Some(selected) = selected {
            targets.retain(|target| selected.contains(&target.id));
        }
        let ranked = targets.len();
        if selected.is_none() {
            let ids = targets.iter().map(|t| t.id).collect::<BTreeSet<_>>();
            targets.extend(
                store::symbol_history::legacy_targets(tx, &poll)?
                    .into_iter()
                    .filter(|t| !ids.contains(&t.id)),
            );
        }
        let entries = store::symbol_history::inventory(tx, &targets, &|| {
            work.check().map_err(|_| store::Error::L1InputsChanged)
        })?;
        Ok::<_, store::Error>((targets, entries, ranked))
    })??;
    report.selected = ranked;
    report.migration_selected = targets.len() - ranked;
    if settings.enrich.top_symbols == 0 && targets.is_empty() {
        report.state = "disabled".into();
    }
    let repo_name = super::l1::repo_name(root, settings);
    let mut pending = Vec::with_capacity(PUBLICATION_BATCH);
    let mut capped = false;
    for (target, entries) in targets.iter().zip(&entries) {
        if entries.is_empty() && target.history.is_none() {
            continue;
        }
        poll()?;
        let collated = treat::symbol_history::collate(entries)?;
        let breadcrumb = context::breadcrumb(
            context::ContextFile {
                repo_name: &repo_name,
                path: &target.path,
            },
            None,
            &[],
            true,
        );
        let header = format!("history of {}", target.qualified);
        // Pure no-ops need no publication or FTS work. The enclosing capture and
        // final check retain HEAD/configuration fencing; actual writes below keep
        // their per-publication source and precommit checks.
        if store.checked_read_snapshot(&poll, |tx| {
            store::symbol_history::unchanged(
                tx,
                target,
                entries,
                &collated.hash,
                Some(&breadcrumb),
                &header,
            )
        })?? {
            report.unchanged += 1;
            continue;
        }
        let columns = build_columns(
            FtsInput {
                breadcrumb: &breadcrumb,
                header: &header,
                text: &collated.text,
                language: None,
                kind: store::FileKind::Doc,
                extra_idents: &[],
                has_code_fence: false,
            },
            &settings.tokenize,
        );
        let key = format!("history:{}", target.id);
        let now = super::unix_timestamp();
        pending.push(PendingHistory {
            target,
            entries,
            hash: collated.hash,
            key,
            breadcrumb,
            header,
            tokens: treat::count_tokens(&collated.text) as i64,
            text: collated.text,
            columns,
            now,
        });
        if pending.len() >= PUBLICATION_BATCH
            && publish_pending(&mut pending, store, &mut report, &poll, &check)?
        {
            capped = true;
            break;
        }
    }
    if !capped && publish_pending(&mut pending, store, &mut report, &poll, &check)? {
        capped = true;
    }
    if capped {
        report.state = "capped".into();
    }
    check()?;
    if selected.is_none() {
        publish_status(store, &report, &check)?;
    }
    Ok(report)
}

impl From<rusqlite::Error> for Error {
    fn from(error: rusqlite::Error) -> Self {
        Self::Store(error.into())
    }
}

fn publish_status(
    store: &Store,
    report: &Report,
    check: &dyn Fn() -> Result<(), store::Error>,
) -> Result<(), Error> {
    store::admission::transaction_checked(
        store,
        store::admission::Kind::Control,
        Some(check),
        |tx| {
            tx.execute("INSERT INTO meta(key,value) VALUES ('symbol_history_status',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[serde_json::to_string(report)?])?;
            check()?;
            tx.commit()?;
            Ok::<_, Error>(())
        },
    )?;
    Ok(())
}
