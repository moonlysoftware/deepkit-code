//! Verified directory roles for outbound/current context; v1 §6.8/§7.1.
use super::{
    Error,
    source::{self, FileInput},
    summary_source,
};
use crate::{config::Config, store::UnitKind, sync::l1::Work, treat::context};
use rusqlite::Transaction;
use std::path::Path;

struct Location {
    path: String,
    scope: Option<String>,
    directory: bool,
}

/// Refreshes file occurrence headers beneath a newly published module role in
/// the caller's publication transaction, replaying exact parser lexical columns.
/// Composite/generated occurrences are handled separately from parser output.
pub(super) fn refresh_module_files(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    module: i64,
    work: Work<'_>,
) -> Result<(), Error> {
    source::check_work(work)?;
    let path: String = tx.query_row(
        "SELECT path FROM dirs WHERE module_unit=?1",
        [module],
        |r| r.get(0),
    )?;
    refresh_path_files(tx, root, config_root, config, &path, work)
}

fn refresh_path_files(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    path: &str,
    work: Work<'_>,
) -> Result<(), Error> {
    refresh_path_files_checked(tx, root, config_root, config, path, None, work, || {})
}

#[allow(
    clippy::too_many_arguments,
    reason = "the checked replay keeps source, configuration, and transaction inputs explicit"
)]
fn refresh_path_files_checked(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    path: &str,
    observed_roles: Option<&[Role]>,
    work: Work<'_>,
    after_replay: impl FnOnce(),
) -> Result<(), Error> {
    let started = std::time::Instant::now();
    // Reconciliation has already source-verified every role in this transaction.
    // Reuse only that transaction-local observation; a full proof remains at the
    // completion fence below, so no role authority is cached across a boundary.
    let roles = match observed_roles {
        Some(roles) => roles_in_scope(roles, path),
        None => scope_roles(tx, root, config_root, config, path, work)?,
    };
    let mut statement = tx.prepare("SELECT id FROM files WHERE (?1='' OR substr(path,1,length(?1)+1)=?1||'/') AND l1_pending=0 AND l1_fingerprint=l1_desired AND EXISTS(SELECT 1 FROM units WHERE file_id=files.id AND parent_unit IS NULL AND kind!='image') ORDER BY id")?;
    let files = statement
        .query_map([path], |r| r.get::<_, i64>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    let processor = crate::sync::l1::Processor::new(root, config)
        .map_err(|_| Error::Invalid("role lexical preparation settings"))?;
    for file in files {
        refresh_file_with_roles(
            tx,
            root,
            config_root,
            config,
            &processor,
            file,
            &roles,
            work,
        )?;
    }
    let modules = refresh_module_composites(tx, root, config_root, config, path, work, &roles)?;
    after_replay();
    source::check_work(work)?;
    // These observations live only inside this write transaction. Re-prove every
    // role after replay so a filesystem/configuration change rolls back the batch.
    if !modules.verify(tx, root, config_root, config, work)?
        || scope_roles(tx, root, config_root, config, path, work)? != roles
    {
        return Err(Error::Invalid("role source changed during lexical replay"));
    }
    tracing::debug!(
        role_count = roles.len(),
        elapsed_us = started.elapsed().as_micros(),
        "folder role lexical replay completed"
    );
    source::check_work(work)
}

/// Replays one file's parser lexical columns with an already verified role scope.
/// Callers re-prove the role observation before their transaction commits.
#[allow(
    clippy::too_many_arguments,
    reason = "the replay needs each fenced source input"
)]
fn refresh_file_with_roles(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    processor: &crate::sync::l1::Processor,
    file: i64,
    roles: &[Role],
    work: Work<'_>,
) -> Result<(), Error> {
    source::check_work(work)?;
    let columns = processor
        .lexical_file_in(tx, file, work)
        .map_err(|error| match error {
            crate::sync::Error::WorkStopped => Error::WorkerStopped,
            _ => Error::Invalid("role lexical source changed"),
        })?;
    for (unit, mut columns) in columns {
        let current =
            source::resolve_freshness_without_roles(tx, root, config_root, config, unit, work)?
                .ok_or(Error::Invalid("role lexical source unavailable"))?;
        let current = apply_observed(tx, root, config, current, roles, work)?;
        crate::store::unit_contexts::record(
            tx,
            unit,
            &current.occurrence.content_hash,
            &super::embedding::context_hash(&current.context, current.occurrence.kind),
            config,
        )?;
        columns.header =
            crate::treat::fts::flatten_header(&current.context.breadcrumb, &current.context.header);
        tx.execute("DELETE FROM fts_units WHERE rowid=?1", [unit])?;
        tx.execute(
            "INSERT INTO fts_units(rowid,header,idents,body,prose) VALUES(?1,?2,?3,?4,?5)",
            rusqlite::params![
                unit,
                columns.header,
                columns.idents,
                columns.body,
                columns.prose
            ],
        )?;
        source::check_work(work)?;
    }
    Ok(())
}

/// Replays the original document/summary lexical recipe for path-owned derived
/// occurrences. Global PR/commit records have no single directory role.
fn refresh_module_composites(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    path: &str,
    work: Work<'_>,
    roles: &[Role],
) -> Result<ModuleReplay, Error> {
    let mut query = tx.prepare("WITH located(id,path,directory) AS (
        SELECT module_unit,path,1 FROM dirs WHERE module_unit IS NOT NULL
        UNION SELECT summary_unit,path,1 FROM dirs WHERE summary_unit IS NOT NULL
        UNION SELECT s.history_unit,f.path,0 FROM symbols s JOIN files f ON f.id=s.file_id WHERE s.history_unit IS NOT NULL
        UNION SELECT u.id,f.path,0 FROM units u JOIN files f ON f.id=u.file_id WHERE u.kind='doc_section' AND u.ladder_level=3
        UNION SELECT u.id,f.path,0 FROM units u JOIN units p ON p.id=u.parent_unit JOIN files f ON f.id=p.file_id WHERE u.kind='summary'
        ) SELECT DISTINCT u.id,u.kind FROM located l JOIN units u ON u.id=l.id
        WHERE ?1='' OR (l.directory=1 AND l.path=?1) OR substr(l.path,1,length(?1)+1)=?1||'/' ORDER BY u.id")?;
    let rows = query
        .query_map([path], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    let units = rows
        .iter()
        .filter(|(_, kind)| kind == "module")
        .map(|(unit, _)| *unit)
        .collect::<Vec<_>>();
    let mut modules =
        source::resolve_module_freshness_batch(tx, root, config_root, config, &units, work)?;
    let replay = ModuleReplay {
        units,
        observed: modules.clone(),
    };
    for (unit, kind) in rows {
        if kind == "module" {
            if location(tx, unit)?.is_none() {
                continue;
            }
            let current = modules
                .remove(&unit)
                .map(|input| apply_observed(tx, root, config, input, roles, work))
                .transpose()?;
            refresh_derived_columns(tx, config, unit, current, work)?;
        } else {
            refresh_derived_unit_observed(tx, root, config_root, config, unit, work, Some(roles))?;
        }
    }
    source::check_work(work)?;
    Ok(replay)
}

/// Shared module inputs survive only until this transaction's fresh final proof.
/// Retain absent inputs too: a newly available module must invalidate the replay.
struct ModuleReplay {
    units: Vec<i64>,
    observed: std::collections::BTreeMap<i64, FileInput>,
}

impl ModuleReplay {
    fn verify(
        &self,
        tx: &Transaction<'_>,
        root: &Path,
        config_root: &Path,
        config: &Config,
        work: Work<'_>,
    ) -> Result<bool, Error> {
        Ok(source::resolve_module_freshness_batch(
            tx,
            root,
            config_root,
            config,
            &self.units,
            work,
        )? == self.observed)
    }
}

/// Replays a derived occurrence's exact lexical recipe with current verified
/// roles, retaining its content identity, body positions and source provenance.
pub(super) fn refresh_derived_unit(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    unit: i64,
    work: Work<'_>,
) -> Result<(), Error> {
    refresh_derived_unit_observed(tx, root, config_root, config, unit, work, None)
}

fn refresh_derived_unit_observed(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    unit: i64,
    work: Work<'_>,
    roles: Option<&[Role]>,
) -> Result<(), Error> {
    source::check_work(work)?;
    if location(tx, unit)?.is_none() {
        return Ok(());
    }
    let current = match roles {
        None => source::resolve_freshness(tx, root, config_root, config, unit, work)?,
        Some(roles) => {
            source::resolve_freshness_without_roles(tx, root, config_root, config, unit, work)?
                .map(|input| apply_observed(tx, root, config, input, roles, work))
                .transpose()?
        }
    };
    refresh_derived_columns(tx, config, unit, current, work)
}

fn refresh_derived_columns(
    tx: &Transaction<'_>,
    config: &Config,
    unit: i64,
    current: Option<FileInput>,
    work: Work<'_>,
) -> Result<(), Error> {
    use crate::treat::fts::{FtsInput, LexicalColumns, build_columns, flatten_header};
    source::check_work(work)?;
    if let Some(current) = &current {
        crate::store::unit_contexts::record(
            tx,
            unit,
            &current.occurrence.content_hash,
            &super::embedding::context_hash(&current.context, current.occurrence.kind),
            config,
        )?;
    } else {
        // Retain stale content, but remove unsupported context terms. Source
        // authority still gates retrieval; fresh publication restores the header.
        tx.execute("DELETE FROM unit_contexts WHERE unit_id=?1", [unit])?;
    }
    let (breadcrumb, header) = current.as_ref().map_or(("", ""), |input| {
        (
            input.context.breadcrumb.as_str(),
            input.context.header.as_str(),
        )
    });
    let (kind, text): (String, String) = tx.query_row(
        "SELECT u.kind,c.text FROM units u JOIN unit_contents c ON c.hash=u.content_hash WHERE u.id=?1",
        [unit], |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    let columns = if kind == "summary" {
        LexicalColumns {
            header: flatten_header(breadcrumb, header),
            idents: String::new(),
            body: text.clone(),
            prose: text,
        }
    } else {
        build_columns(
            FtsInput {
                breadcrumb,
                header,
                text: &text,
                language: None,
                kind: crate::store::FileKind::Doc,
                extra_idents: &[],
                has_code_fence: kind == "doc_section" && crate::treat::contains_fenced_code(&text),
            },
            &config.tokenize,
        )
    };
    tx.execute("DELETE FROM fts_units WHERE rowid=?1", [unit])?;
    tx.execute(
        "INSERT INTO fts_units(rowid,header,idents,body,prose) VALUES(?1,?2,?3,?4,?5)",
        rusqlite::params![
            unit,
            columns.header,
            columns.idents,
            columns.body,
            columns.prose
        ],
    )?;
    source::check_work(work)
}

struct Reconciliation {
    roots: Vec<String>,
    dirty_keys: Vec<String>,
    receipt: Option<String>,
    roles: Vec<Role>,
    pending: bool,
}

/// A bounded replay item. File rows use parser columns; derived rows replay the
/// stored composite recipe with the same observed breadcrumb roles.
enum ReplayTarget {
    File(i64),
    Derived { unit: i64, kind: String },
}

fn replay_targets(tx: &Transaction<'_>, path: &str) -> Result<Vec<ReplayTarget>, Error> {
    let mut targets = tx
        .prepare("SELECT id FROM files WHERE (?1='' OR substr(path,1,length(?1)+1)=?1||'/') AND l1_pending=0 AND l1_fingerprint=l1_desired AND EXISTS(SELECT 1 FROM units WHERE file_id=files.id AND parent_unit IS NULL AND kind!='image') ORDER BY id")?
        .query_map([path], |row| row.get::<_, i64>(0))?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(ReplayTarget::File)
        .collect::<Vec<_>>();
    let mut derived = tx.prepare("WITH located(id,path,directory) AS (
        SELECT module_unit,path,1 FROM dirs WHERE module_unit IS NOT NULL
        UNION SELECT summary_unit,path,1 FROM dirs WHERE summary_unit IS NOT NULL
        UNION SELECT s.history_unit,f.path,0 FROM symbols s JOIN files f ON f.id=s.file_id WHERE s.history_unit IS NOT NULL
        UNION SELECT u.id,f.path,0 FROM units u JOIN files f ON f.id=u.file_id WHERE u.kind='doc_section' AND u.ladder_level=3
        UNION SELECT u.id,f.path,0 FROM units u JOIN units p ON p.id=u.parent_unit JOIN files f ON f.id=p.file_id WHERE u.kind='summary'
        ) SELECT DISTINCT u.id,u.kind FROM located l JOIN units u ON u.id=l.id
        WHERE ?1='' OR (l.directory=1 AND l.path=?1) OR substr(l.path,1,length(?1)+1)=?1||'/' ORDER BY u.id")?
        .query_map([path], |row| Ok(ReplayTarget::Derived { unit: row.get(0)?, kind: row.get(1)? }))?
        .collect::<Result<Vec<_>, _>>()?;
    targets.append(&mut derived);
    Ok(targets)
}

#[allow(
    clippy::too_many_arguments,
    reason = "the batch retains each fresh source proof explicitly"
)]
fn refresh_replay_batch(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    path: &str,
    targets: &[ReplayTarget],
    observed_roles: Option<&[Role]>,
    work: Work<'_>,
) -> Result<(), Error> {
    source::check_work(work)?;
    // A reconciliation plan observes every role before any bounded writer
    // transaction begins, then re-proves the complete plan before clearing its
    // durable marker. Reuse that observation within the replay so a large
    // cohort does not re-resolve every module summary for each small WAL batch.
    // Callers without a plan retain the per-batch authority proof.
    let roles = match observed_roles {
        Some(roles) => roles_in_scope(roles, path),
        None => scope_roles(tx, root, config_root, config, path, work)?,
    };
    let processor = crate::sync::l1::Processor::new(root, config)
        .map_err(|_| Error::Invalid("role lexical preparation settings"))?;
    for target in targets {
        match target {
            ReplayTarget::File(file) => {
                refresh_file_with_roles(
                    tx,
                    root,
                    config_root,
                    config,
                    &processor,
                    *file,
                    &roles,
                    work,
                )?;
            }
            ReplayTarget::Derived { unit, kind } if kind == "module" => {
                refresh_module_with_roles(tx, root, config_root, config, *unit, &roles, work)?;
            }
            ReplayTarget::Derived { unit, .. } => {
                refresh_derived_unit_observed(
                    tx,
                    root,
                    config_root,
                    config,
                    *unit,
                    work,
                    Some(&roles),
                )?;
            }
        }
        source::check_work(work)?;
    }
    if observed_roles.is_none() && scope_roles(tx, root, config_root, config, path, work)? != roles
    {
        return Err(Error::Invalid("role source changed during lexical replay"));
    }
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    reason = "the module replay retains the same fenced inputs as the batch"
)]
fn refresh_module_with_roles(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    unit: i64,
    roles: &[Role],
    work: Work<'_>,
) -> Result<(), Error> {
    let mut observed =
        source::resolve_module_freshness_batch(tx, root, config_root, config, &[unit], work)?;
    let input = observed.remove(&unit);
    let current = input
        .clone()
        .map(|input| apply_observed(tx, root, config, input, roles, work))
        .transpose()?;
    refresh_derived_columns(tx, config, unit, current, work)?;
    let mut verified =
        source::resolve_module_freshness_batch(tx, root, config_root, config, &[unit], work)?;
    if verified.remove(&unit) != input {
        return Err(Error::Invalid("role lexical source changed"));
    }
    Ok(())
}

fn plan_reconciliation(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    work: Work<'_>,
) -> Result<Reconciliation, Error> {
    use rusqlite::OptionalExtension;
    use std::collections::{BTreeMap, BTreeSet};
    source::check_work(work)?;
    let pending = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM meta WHERE key='lexical_role_reconcile_pending')",
        [],
        |row| row.get(0),
    )?;
    let stored: Option<String> = tx
        .query_row(
            "SELECT value FROM meta WHERE key='lexical_role_receipt_v1'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    let previous: BTreeMap<String, String> = stored
        .as_deref()
        .map(serde_json::from_str)
        .transpose()
        .map_err(|_| Error::Invalid("lexical role receipt"))?
        .unwrap_or_default();
    let mut query = tx.prepare(
        "SELECT path,summary_unit FROM dirs WHERE summary_unit IS NOT NULL ORDER BY path",
    )?;
    let candidates = query
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?
        .collect::<Result<Vec<_>, _>>()?;
    let roles = if config.embedder.breadcrumb && config.embedder.breadcrumb_roles {
        verified_roles(tx, root, config_root, config, candidates.clone(), work)?
    } else {
        Vec::new()
    };
    let current = roles
        .iter()
        .map(|role| {
            (
                role.path.clone(),
                blake3::hash(role.text.as_bytes()).to_hex().to_string(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut dirty: BTreeSet<String> = previous
        .keys()
        .chain(current.keys())
        .filter(|path| previous.get(*path) != current.get(*path))
        .cloned()
        .collect();
    // Pre-receipt databases may already contain role-bearing FTS headers.
    if stored.is_none() {
        dirty.extend(candidates.into_iter().map(|(path, _)| path));
    }
    let dirty_files = tx.prepare("SELECT m.key,f.path FROM meta m LEFT JOIN files f ON f.id=CAST(substr(m.key,20) AS INTEGER) AND m.key='lexical_role_dirty:'||f.id WHERE m.key GLOB 'lexical_role_dirty:*' ORDER BY m.key")?
        .query_map([], |r| Ok((r.get::<_,String>(0)?,r.get::<_,Option<String>>(1)?)))?
        .collect::<Result<Vec<_>,_>>()?;
    for (_, path) in &dirty_files {
        if let Some(path) = path
            && previous
                .keys()
                .chain(current.keys())
                .any(|role| role.is_empty() || path.starts_with(&format!("{role}/")))
        {
            dirty.insert(
                path.rsplit_once('/')
                    .map_or("", |(parent, _)| parent)
                    .to_owned(),
            );
        }
    }
    let dirty_keys = dirty_files.into_iter().map(|(key, _)| key).collect();
    let mut roots: Vec<String> = Vec::new();
    for path in dirty {
        if !roots.iter().any(|parent| {
            parent.is_empty() || path == *parent || path.starts_with(&format!("{parent}/"))
        }) {
            roots.push(path);
        }
    }
    let receipt = if stored.is_some() || !current.is_empty() || !roots.is_empty() {
        let value = serde_json::to_string(&current)
            .map_err(|_| Error::Invalid("lexical role receipt encoding"))?;
        (stored.as_deref() != Some(&value)).then_some(value)
    } else {
        None
    };
    source::check_work(work)?;
    Ok(Reconciliation {
        roots,
        dirty_keys,
        receipt,
        roles,
        pending,
    })
}

/// Reconciles applied lexical roles after source, summary or configuration changes.
/// The receipt only selects dirty scopes; each rewrite re-proves source authority.
pub(crate) fn reconcile_in(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    work: Work<'_>,
) -> Result<usize, Error> {
    let plan = plan_reconciliation(tx, root, config_root, config, work)?;
    for path in &plan.roots {
        refresh_path_files_checked(
            tx,
            root,
            config_root,
            config,
            path,
            Some(&plan.roles),
            work,
            || {},
        )?;
    }
    finish_reconciliation(tx, &plan)?;
    source::check_work(work)?;
    Ok(plan.roots.len())
}

/// Commits the role receipt only after every selected scope has been refreshed.
/// A pending marker stays durable when a bounded replay is interrupted or capped.
fn finish_reconciliation(tx: &Transaction<'_>, plan: &Reconciliation) -> Result<(), Error> {
    if let Some(value) = &plan.receipt {
        tx.execute("INSERT INTO meta(key,value) VALUES('lexical_role_receipt_v1',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value", [value])?;
    }
    for key in &plan.dirty_keys {
        tx.execute("DELETE FROM meta WHERE key=?1", [key])?;
    }
    // Enrichment records this durable marker in the publication transaction.
    // Clearing it only after the fresh receipt and affected FTS scopes commit
    // lets a later host pass recover a failed cohort reconciliation.
    tx.execute(
        "DELETE FROM meta WHERE key='lexical_role_reconcile_pending'",
        [],
    )?;
    Ok(())
}

/// Refreshes changed role scopes after an L1/graph pass, under normal WAL and
/// database growth admission. Unchanged repositories without a pending
/// reconciliation do not open a writer.
pub(crate) fn reconcile(
    store: &crate::store::Store,
    root: &Path,
    config_root: &Path,
    config: &Config,
    work: Work<'_>,
) -> Result<usize, Error> {
    // A single source-dense file can replace more FTS content than ordinary L1
    // admission reserves. Replay one target at a time so it can use the
    // checked atomic catalogue allowance and checkpoint before the next target.
    reconcile_with_batch_size(store, root, config_root, config, work, 1)
}

#[allow(
    clippy::too_many_arguments,
    reason = "tests exercise the transaction boundary at one replay item"
)]
fn reconcile_with_batch_size(
    store: &crate::store::Store,
    root: &Path,
    config_root: &Path,
    config: &Config,
    work: Work<'_>,
    batch_size: usize,
) -> Result<usize, Error> {
    if batch_size == 0 {
        return Err(Error::Invalid("role replay batch size"));
    }
    let needed: bool = store.read_snapshot(|tx| tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM dirs WHERE summary_unit IS NOT NULL) OR EXISTS(SELECT 1 FROM meta WHERE key IN ('lexical_role_receipt_v1','lexical_role_reconcile_pending'))", [], |r| r.get(0),
    ))??;
    if !needed {
        source::check_work(work)?;
        return Ok(0);
    }
    // Determine a true no-op through a reader before seeking capped writer
    // admission. Recompute inside any write transaction; this is no authority cache.
    let plan =
        store.read_snapshot(|tx| plan_reconciliation(tx, root, config_root, config, work))??;
    if plan.roots.is_empty()
        && plan.receipt.is_none()
        && plan.dirty_keys.is_empty()
        && !plan.pending
    {
        source::check_work(work)?;
        return Ok(0);
    }
    let check = || {
        work.check()
            .map_err(|_| crate::store::Error::L1InputsChanged)
    };
    if plan.roots.is_empty() {
        return crate::store::admission::transaction_checked(
            store,
            crate::store::admission::Kind::L1Growth,
            Some(&check),
            |tx| {
                let changed = reconcile_in(&tx, root, config_root, config, work)?;
                if crate::config::load(config_root, None)
                    .map_err(|_| Error::Invalid("role configuration reload"))?
                    .config
                    != *config
                {
                    return Err(Error::Invalid("role configuration changed"));
                }
                source::check_work(work)?;
                tx.commit()?;
                Ok(changed)
            },
        );
    }
    // Contentless FTS replacement can require far more WAL than the summary
    // publication that made a role current. Replay a bounded mixed list, with a
    // fresh role proof and transaction fence for each commit. The marker remains
    // until the final plan is re-proved and its receipt is committed.
    for path in &plan.roots {
        let targets = store.read_snapshot(|tx| replay_targets(tx, path))??;
        for targets in targets.chunks(batch_size) {
            crate::store::admission::transaction_checked(
                store,
                crate::store::admission::Kind::AtomicCatalogue,
                Some(&check),
                |tx| {
                    refresh_replay_batch(
                        &tx,
                        root,
                        config_root,
                        config,
                        path,
                        targets,
                        Some(&plan.roles),
                        work,
                    )?;
                    if crate::config::load(config_root, None)
                        .map_err(|_| Error::Invalid("role configuration reload"))?
                        .config
                        != *config
                    {
                        return Err(Error::Invalid("role configuration changed"));
                    }
                    source::check_work(work)?;
                    tx.commit()?;
                    Ok::<_, Error>(())
                },
            )?;
        }
    }
    crate::store::admission::transaction_checked(
        store,
        crate::store::admission::Kind::L1Growth,
        Some(&check),
        |tx| {
            let final_plan = plan_reconciliation(&tx, root, config_root, config, work)?;
            if final_plan.roots != plan.roots
                || final_plan.dirty_keys != plan.dirty_keys
                || final_plan.roles != plan.roles
            {
                return Err(Error::Invalid("role source changed during lexical replay"));
            }
            finish_reconciliation(&tx, &final_plan)?;
            if crate::config::load(config_root, None)
                .map_err(|_| Error::Invalid("role configuration reload"))?
                .config
                != *config
            {
                return Err(Error::Invalid("role configuration changed"));
            }
            source::check_work(work)?;
            tx.commit()?;
            Ok(plan.roots.len())
        },
    )
}

fn location(tx: &Transaction<'_>, unit: i64) -> Result<Option<Location>, Error> {
    let mut query = tx.prepare("SELECT f.path,u.scope_path,0 FROM units u JOIN files f ON f.id=u.file_id WHERE u.id=?1 AND u.kind!='summary'
        UNION ALL SELECT f.path,NULL,0 FROM units u JOIN units p ON p.id=u.parent_unit JOIN files f ON f.id=p.file_id WHERE u.id=?1 AND u.kind='summary' AND u.file_id IS NULL
        UNION ALL SELECT f.path,NULL,0 FROM symbols s JOIN files f ON f.id=s.file_id WHERE s.history_unit=?1
        UNION ALL SELECT d.path,NULL,1 FROM dirs d WHERE d.module_unit=?1 OR d.summary_unit=?1")?;
    let rows = query
        .query_map([unit], |r| {
            Ok(Location {
                path: r.get(0)?,
                scope: r.get(1)?,
                directory: r.get(2)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    match rows.len() {
        0 => Ok(None),
        1 => Ok(rows.into_iter().next()),
        _ => Err(Error::Invalid("ambiguous role context location")),
    }
}

fn ancestors(path: &str, directory: bool) -> Vec<String> {
    let parent = if directory {
        path
    } else {
        path.rsplit_once('/').map_or("", |(parent, _)| parent)
    };
    let mut result = vec![String::new()];
    let mut current = String::new();
    for segment in parent.split('/').filter(|segment| !segment.is_empty()) {
        if !current.is_empty() {
            current.push('/');
        }
        current.push_str(segment);
        result.push(current.clone());
    }
    result
}

/// Full role observation, retained only within one checked transaction.
#[derive(Clone, PartialEq, Eq)]
struct Role {
    path: String,
    summary: i64,
    content: [u8; 32],
    fingerprint: [u8; 32],
    text: String,
}

fn roles_in_scope(roles: &[Role], path: &str) -> Vec<Role> {
    let mut scoped = roles
        .iter()
        .filter(|role| {
            path.is_empty()
                || role.path.is_empty()
                || role.path == path
                || role.path.starts_with(&format!("{path}/"))
                || path.starts_with(&format!("{}/", role.path))
        })
        .cloned()
        .collect::<Vec<_>>();
    // `scope_roles` proves the final observation in ancestor order. The global
    // reconciliation plan is path ordered, so normalize its scoped subset
    // before comparing the two equivalent observations.
    scoped.sort_by(|left, right| {
        left.path
            .len()
            .cmp(&right.path.len())
            .then_with(|| left.path.cmp(&right.path))
    });
    scoped
}

/// Shared ancestor observations for one bounded read snapshot. Every observed
/// path, including absent roles, must be re-proved before using the batch.
#[derive(Default)]
pub(super) struct Batch {
    observed: std::collections::BTreeMap<String, Option<Role>>,
}

impl Batch {
    /// Adds individually verified ancestor roles, sharing only this pass's proofs.
    pub(super) fn apply(
        &mut self,
        tx: &Transaction<'_>,
        root: &Path,
        config_root: &Path,
        config: &Config,
        input: FileInput,
        work: Work<'_>,
    ) -> Result<FileInput, Error> {
        source::check_work(work)?;
        if !config.embedder.breadcrumb
            || !config.embedder.breadcrumb_roles
            || input.policy.never_send
        {
            return Ok(input);
        }
        let Some(location) = location(tx, input.occurrence.id)? else {
            return Ok(input);
        };
        let missing = ancestors(&location.path, location.directory)
            .into_iter()
            .filter(|path| !self.observed.contains_key(path))
            .collect::<Vec<_>>();
        let roles = path_roles(tx, root, config_root, config, &missing, work)?;
        for path in missing {
            self.observed.insert(path, None);
        }
        for role in roles {
            self.observed.insert(role.path.clone(), Some(role));
        }
        with_roles(
            root,
            config,
            input,
            &location,
            self.observed.values().filter_map(Option::as_ref),
        )
    }

    /// Re-observes every encountered path so source, policy and role changes fail closed.
    pub(super) fn verify(
        &self,
        tx: &Transaction<'_>,
        root: &Path,
        config_root: &Path,
        config: &Config,
        work: Work<'_>,
    ) -> Result<bool, Error> {
        let paths = self.observed.keys().cloned().collect::<Vec<_>>();
        let current = path_roles(tx, root, config_root, config, &paths, work)?;
        Ok(self
            .observed
            .values()
            .filter_map(Option::as_ref)
            .eq(current.iter()))
    }
}

fn path_roles(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    paths: &[String],
    work: Work<'_>,
) -> Result<Vec<Role>, Error> {
    source::check_work(work)?;
    if paths.is_empty() {
        return Ok(Vec::new());
    }
    let paths =
        serde_json::to_string(paths).map_err(|_| Error::Invalid("role ancestor encoding"))?;
    let mut query = tx.prepare(
        "SELECT d.path,d.summary_unit FROM dirs d JOIN json_each(?1) wanted
        ON wanted.value=d.path WHERE d.summary_unit IS NOT NULL ORDER BY d.path",
    )?;
    let candidates = query
        .query_map([paths], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    verified_roles(tx, root, config_root, config, candidates, work)
}

fn verified_roles(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    candidates: Vec<(String, i64)>,
    work: Work<'_>,
) -> Result<Vec<Role>, Error> {
    let started = std::time::Instant::now();
    let candidate_count = candidates.len();
    let mut roles = Vec::new();
    let mut symbols = super::enrichment_inputs::SymbolInputs::default();
    let summaries = serde_json::to_string(
        &candidates
            .iter()
            .map(|(_, summary)| *summary)
            .collect::<Vec<_>>(),
    )
    .map_err(|_| Error::Invalid("role summary selection"))?;
    let modules = tx
        .prepare(
            "SELECT module_unit FROM dirs WHERE summary_unit IN (SELECT value FROM json_each(?1)) ORDER BY module_unit",
        )?
        .query_map([summaries], |r| r.get::<_, i64>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    let parents = super::module_source::resolve_enrichment_batch(
        tx,
        root,
        config_root,
        config,
        &modules,
        work,
    )?;
    for (path, summary) in candidates {
        source::check_work(work)?;
        let Some(verified) = summary_source::resolve_with_role_caches(
            tx,
            root,
            config_root,
            config,
            summary,
            work,
            &mut symbols,
            &parents,
        )?
        else {
            continue;
        };
        if verified.policy.never_send {
            continue;
        }
        if let Some(line) = verified
            .context
            .text
            .lines()
            .next()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            roles.push(Role {
                path,
                summary,
                content: verified.occurrence.content_hash,
                fingerprint: verified.policy.fingerprint,
                text: line.to_owned(),
            });
        }
    }
    source::check_work(work)?;
    tracing::debug!(
        candidate_count,
        elapsed_us = started.elapsed().as_micros(),
        "breadcrumb role proof completed"
    );
    Ok(roles)
}

fn scope_roles(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    path: &str,
    work: Work<'_>,
) -> Result<Vec<Role>, Error> {
    source::check_work(work)?;
    if !config.embedder.breadcrumb || !config.embedder.breadcrumb_roles {
        return Ok(Vec::new());
    }
    let mut q = tx.prepare("SELECT path,summary_unit FROM dirs WHERE summary_unit IS NOT NULL AND
        (?1='' OR path='' OR path=?1 OR substr(path,1,length(?1)+1)=?1||'/' OR substr(?1,1,length(path)+1)=path||'/')
        ORDER BY length(path),path")?;
    let candidates = q
        .query_map([path], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    verified_roles(tx, root, config_root, config, candidates, work)
}

/// Adds only currently proved module first lines. No role/provenance cache
/// survives a call: dispatch, completion and response fences re-prove inputs.
/// Generative summary resolution itself remains role-free to avoid feedback.
pub(super) fn apply(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    input: FileInput,
    work: Work<'_>,
) -> Result<FileInput, Error> {
    source::check_work(work)?;
    if !config.embedder.breadcrumb || !config.embedder.breadcrumb_roles || input.policy.never_send {
        return Ok(input);
    }
    let available: bool = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM dirs WHERE summary_unit IS NOT NULL)",
        [],
        |r| r.get(0),
    )?;
    if !available {
        return Ok(input);
    }
    let Some(location) = location(tx, input.occurrence.id)? else {
        return Ok(input);
    };
    let paths = serde_json::to_string(&ancestors(&location.path, location.directory))
        .map_err(|_| Error::Invalid("role ancestor encoding"))?;
    let mut query = tx.prepare("SELECT d.path,d.summary_unit FROM dirs d JOIN json_each(?1) wanted ON wanted.value=d.path WHERE d.summary_unit IS NOT NULL ORDER BY length(d.path),d.path")?;
    let candidates = query
        .query_map([paths], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    let roles = verified_roles(tx, root, config_root, config, candidates, work)?;
    with_roles(root, config, input, &location, &roles)
}

fn apply_observed(
    tx: &Transaction<'_>,
    root: &Path,
    config: &Config,
    input: FileInput,
    roles: &[Role],
    work: Work<'_>,
) -> Result<FileInput, Error> {
    source::check_work(work)?;
    if !config.embedder.breadcrumb || !config.embedder.breadcrumb_roles || input.policy.never_send {
        return Ok(input);
    }
    let Some(location) = location(tx, input.occurrence.id)? else {
        return Ok(input);
    };
    with_roles(root, config, input, &location, roles)
}

fn with_roles<'a>(
    root: &Path,
    config: &Config,
    mut input: FileInput,
    location: &Location,
    roles: impl IntoIterator<Item = &'a Role>,
) -> Result<FileInput, Error> {
    let ancestors = ancestors(&location.path, location.directory);
    let dirs = roles
        .into_iter()
        .filter(|role| ancestors.contains(&role.path))
        .map(|role| context::DirectoryContext {
            path: &role.path,
            role: Some(&role.text),
        })
        .collect::<Vec<_>>();
    if dirs.is_empty() {
        return Ok(input);
    }
    let repo = crate::sync::l1::repo_name(root, config);
    let file = context::ContextFile {
        repo_name: &repo,
        path: &location.path,
    };
    let breadcrumb = if location.directory {
        context::directory_breadcrumb(file, &dirs, true)
    } else {
        context::breadcrumb(
            file,
            location
                .scope
                .as_deref()
                .filter(|_| input.occurrence.kind == UnitKind::DocSection),
            &dirs,
            true,
        )
    };
    input.context = input
        .context
        .with_breadcrumb(breadcrumb)
        .ok_or(Error::Invalid("role breadcrumb representation"))?;
    Ok(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        queue::{Kind, embedding, enrichment_inputs},
        store::{self, Store, enrichment::publication::Publication},
        sync,
    };
    use std::fs;

    #[test]
    fn scoped_plan_roles_use_the_final_ancestor_proof_order() {
        let role = |path: &str| Role {
            path: path.to_owned(),
            summary: 1,
            content: [1; 32],
            fingerprint: [2; 32],
            text: "role".to_owned(),
        };
        let scoped = roles_in_scope(&[role("alpha/deep"), role("z")], "");
        assert_eq!(
            scoped.into_iter().map(|role| role.path).collect::<Vec<_>>(),
            ["z", "alpha/deep"]
        );
    }

    #[test]
    fn serving_summary_observation_rejects_withdrawn_changed_and_private_inputs() {
        for case in ["withdraw", "source", "prompt", "privacy"] {
            let f = Fixture::new();
            f.publish(f.module, "Payments\nSecond line");
            let observe = || {
                f.store
                    .read_snapshot(|tx| {
                        crate::search::summaries::observe_in(
                            tx,
                            f.root.path(),
                            f.root.path(),
                            &f.config,
                            crate::queue::summary_source::Attachment::Parent(f.module),
                            Work::default(),
                        )
                    })
                    .unwrap()
                    .unwrap()
            };
            let before = observe().unwrap();
            assert_eq!(before.text, "Payments\nSecond line");
            match case {
                "withdraw" => {
                    f.store
                        .writer()
                        .unwrap()
                        .execute(
                            "UPDATE dirs SET summary_unit=NULL WHERE module_unit=?1",
                            [f.module],
                        )
                        .unwrap();
                }
                "source" => {
                    fs::write(
                        f.root.path().join("billing/b.rs"),
                        "pub fn changed_invoice() {}\n",
                    )
                    .unwrap();
                }
                "prompt" => {
                    fs::create_dir_all(f.root.path().join(".deepkit/prompts")).unwrap();
                    fs::write(
                        f.root.path().join(".deepkit/prompts/summarize_module.md"),
                        "Explain carefully.\n{{collation}}\n{{summaries}}\n",
                    )
                    .unwrap();
                }
                "privacy" => {
                    let mut config = f.config.clone();
                    for treatment in &mut config.treatments {
                        treatment.never_send = true;
                    }
                    fs::write(
                        f.root.path().join(".deepkit/config.toml"),
                        toml::to_string(&config).unwrap(),
                    )
                    .unwrap();
                }
                _ => unreachable!(),
            }
            assert!(observe().is_none(), "{case}");
            if case == "prompt" {
                f.publish(f.module, "Payments\nSecond line");
                let after = observe().unwrap();
                assert_eq!(before.text, after.text);
                assert_ne!(
                    before, after,
                    "equal text must retain different prompt evidence"
                );
                let mut old = blake3::Hasher::new();
                let mut new = blake3::Hasher::new();
                before.bind(&mut old);
                after.bind(&mut new);
                assert_ne!(old.finalize(), new.finalize());
            }
        }
    }

    #[test]
    fn search_role_batch_matches_independent_inputs_and_rejects_late_role_changes() {
        use std::sync::atomic::{AtomicBool, Ordering};
        for case in ["source", "policy", "withdraw", "publish", "cancel"] {
            let f = Fixture::new();
            let summary = f.publish(f.module, "Payments");
            let mut db = f.store.writer().unwrap();
            let tx = db.transaction().unwrap();
            if case == "publish" {
                tx.execute(
                    "UPDATE dirs SET summary_unit=NULL WHERE module_unit=?1",
                    [f.module],
                )
                .unwrap();
            }
            let units = tx
                .prepare("SELECT id FROM units WHERE file_id IS NOT NULL ORDER BY id")
                .unwrap()
                .query_map([], |r| r.get::<_, i64>(0))
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert!(units.len() >= 2);
            let cancelled = AtomicBool::new(false);
            let work = Work {
                deadline: None,
                cancelled: Some(&cancelled),
            };
            let mut batch = Batch::default();
            let before = tx.total_changes();
            for unit in units {
                let base = || {
                    source::resolve_freshness_without_roles(
                        &tx,
                        f.root.path(),
                        f.root.path(),
                        &f.config,
                        unit,
                        work,
                    )
                    .unwrap()
                    .unwrap()
                };
                let independent =
                    apply(&tx, f.root.path(), f.root.path(), &f.config, base(), work).unwrap();
                let shared = batch
                    .apply(&tx, f.root.path(), f.root.path(), &f.config, base(), work)
                    .unwrap();
                assert_eq!(shared.context, independent.context);
            }
            assert_eq!(batch.observed.len(), 2); // root and billing, shared by both files
            assert!(
                batch
                    .verify(&tx, f.root.path(), f.root.path(), &f.config, work)
                    .unwrap()
            );
            assert_eq!(tx.total_changes(), before);
            match case {
                "source" => fs::write(
                    f.root.path().join("billing/a.rs"),
                    "pub fn changed_retry() {}\n",
                )
                .unwrap(),
                "policy" => {
                    let mut config = f.config.clone();
                    for treatment in &mut config.treatments {
                        treatment.never_send = true;
                    }
                    fs::write(
                        f.root.path().join(".deepkit/config.toml"),
                        toml::to_string(&config).unwrap(),
                    )
                    .unwrap();
                }
                "withdraw" => {
                    tx.execute(
                        "UPDATE dirs SET summary_unit=NULL WHERE module_unit=?1",
                        [f.module],
                    )
                    .unwrap();
                }
                "publish" => {
                    tx.execute(
                        "UPDATE dirs SET summary_unit=?2 WHERE module_unit=?1",
                        rusqlite::params![f.module, summary],
                    )
                    .unwrap();
                }
                "cancel" => cancelled.store(true, Ordering::Release),
                _ => unreachable!(),
            }
            let verified = batch.verify(&tx, f.root.path(), f.root.path(), &f.config, work);
            if case == "cancel" {
                assert!(matches!(verified, Err(Error::WorkerStopped)));
            } else {
                assert!(!verified.unwrap(), "{case}");
            }
        }
    }

    #[test]
    fn batched_role_replay_rolls_back_on_late_source_policy_or_cancellation() {
        use std::sync::atomic::{AtomicBool, Ordering};
        for case in ["source", "policy", "cancel"] {
            let cancel = case == "cancel";
            let f = Fixture::new();
            f.publish(f.module, "Payments");
            reconcile(
                &f.store,
                f.root.path(),
                f.root.path(),
                &f.config,
                Work::default(),
            )
            .unwrap();
            f.publish(f.module, "Collections");
            let mut db = f.store.writer().unwrap();
            db.execute_batch("CREATE VIRTUAL TABLE temp.batch_role_vocab USING fts5vocab(main,fts_units,'instance')").unwrap();
            let terms = |db: &rusqlite::Connection| {
                db.prepare("SELECT doc,term,col,offset FROM temp.batch_role_vocab ORDER BY doc,col,offset,term").unwrap()
                    .query_map([], |r| Ok((r.get::<_,i64>(0)?,r.get::<_,String>(1)?,r.get::<_,String>(2)?,r.get::<_,i64>(3)?))).unwrap()
                    .collect::<Result<Vec<_>,_>>().unwrap()
            };
            let contexts = |db: &rusqlite::Connection| {
                db.prepare("SELECT unit_id,ctx_hash FROM unit_contexts ORDER BY unit_id")
                    .unwrap()
                    .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?)))
                    .unwrap()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap()
            };
            let before_terms = terms(&db);
            let before_contexts = contexts(&db);
            let cancelled = AtomicBool::new(false);
            let tx = db.transaction().unwrap();
            let result = refresh_path_files_checked(
                &tx,
                f.root.path(),
                f.root.path(),
                &f.config,
                "billing",
                None,
                Work {
                    deadline: None,
                    cancelled: Some(&cancelled),
                },
                || {
                    if cancel {
                        cancelled.store(true, Ordering::Release);
                    } else if case == "policy" {
                        let mut config = f.config.clone();
                        for treatment in &mut config.treatments {
                            treatment.never_send = true;
                        }
                        fs::write(
                            f.root.path().join(".deepkit/config.toml"),
                            toml::to_string(&config).unwrap(),
                        )
                        .unwrap();
                    } else {
                        fs::write(
                            f.root.path().join("billing/a.rs"),
                            "pub fn changed_retry() {}\n",
                        )
                        .unwrap();
                    }
                },
            );
            if cancel {
                assert!(matches!(result, Err(Error::WorkerStopped)));
            } else {
                assert!(matches!(
                    result,
                    Err(Error::Invalid("role source changed during lexical replay"))
                ));
            }
            tx.rollback().unwrap();
            assert_eq!(terms(&db), before_terms);
            assert_eq!(contexts(&db), before_contexts);
        }
    }

    #[tokio::test]
    async fn real_module_roles_refresh_only_affected_vectors_through_public_cli() {
        use std::collections::BTreeMap;
        use wiremock::{
            Mock, MockServer, ResponseTemplate,
            matchers::{method, path},
        };
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &wiremock::Request| {
                let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "embeddings": vec![vec![1.0, 0.0]; body["input"].as_array().unwrap().len()]
                }))
            })
            .mount(&server)
            .await;
        let mut f = Fixture::new();
        fs::write(f.root.path().join("control.rs"), "pub fn unaffected() {}\n").unwrap();
        f.config.embedder.profile = "ollama".into();
        f.config.embedder.dims = 2;
        f.config.embedder.ollama.url = server.uri();
        f.config.embedder.ollama.max_batch_items = 1;
        fs::write(
            f.root.path().join(".deepkit/config.toml"),
            toml::to_string(&f.config).unwrap(),
        )
        .unwrap();
        f.config = crate::config::load(f.root.path(), None).unwrap().config;
        async fn cli(root: &Path, refresh: bool) {
            let mut command = tokio::process::Command::new(assert_cmd::cargo::cargo_bin("dkc"));
            command
                .current_dir(root)
                .env("XDG_CONFIG_HOME", root.join("empty-config"))
                .env("DKC_MODELS_DIR", root.join("unused-models"))
                .args(["index", "--level", "2", "--json"]);
            if refresh {
                command.arg("--refresh-context");
            }
            command.kill_on_drop(true);
            let output = tokio::time::timeout(std::time::Duration::from_secs(30), command.output())
                .await
                .unwrap()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(value["embedding"]["state"], "quiescent", "{value}");
        }
        let vectors = || -> BTreeMap<i64, Vec<u8>> {
            let db = f.store.writer().unwrap();
            let slug: String = db
                .query_row("SELECT slug FROM embedding_tables", [], |r| r.get(0))
                .unwrap();
            assert!(slug.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'));
            db.prepare(&format!(
                "SELECT unit_id,ctx_hash FROM vec_{slug} ORDER BY unit_id"
            ))
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
        };
        cli(f.root.path(), false).await;
        // Establish all derived coverage before measuring a change to the role.
        f.publish(f.module, "Payments\nHandles invoices.");
        cli(f.root.path(), true).await;
        let baseline = vectors();
        let affected: Vec<i64> = f.store.read_snapshot(|tx| {
            let mut q = tx.prepare("SELECT u.id FROM units u LEFT JOIN files f ON f.id=u.file_id WHERE f.path GLOB 'billing/*' OR EXISTS(SELECT 1 FROM dirs d WHERE d.path='billing' AND (d.module_unit=u.id OR d.summary_unit=u.id)) ORDER BY u.id")?;
            q.query_map([], |r| r.get(0))?.collect::<Result<_, rusqlite::Error>>()
        }).unwrap().unwrap();
        let affected: Vec<i64> = affected
            .into_iter()
            .filter(|id| baseline.contains_key(id))
            .collect();
        assert!(affected.len() >= 2);
        assert!(
            baseline.len() > affected.len(),
            "needs unaffected control coverage"
        );
        let old_summary = f.publish(f.module, "Payments\nHandles invoices.");
        let new_summary = f.publish(f.module, "Collections\nHandles invoices.");
        assert_eq!(old_summary, new_summary);
        let calls_before = server.received_requests().await.unwrap().len();
        cli(f.root.path(), false).await;
        let retained = vectors();
        // Replacing the summary removes its coverage. The raw module is briefly
        // embedded as fallback until the new summary vector replaces it.
        // Every other covered unit retains its old context until explicit refresh.
        for (id, hash) in &baseline {
            if *id != new_summary {
                assert_eq!(retained.get(id), Some(hash), "unit {id}");
            }
        }
        let calls_default = server.received_requests().await.unwrap().len();
        assert_eq!(
            calls_default,
            calls_before + 2,
            "new requests: {:?}",
            server.received_requests().await.unwrap()[calls_before..]
                .iter()
                .map(|r| String::from_utf8_lossy(&r.body).into_owned())
                .collect::<Vec<_>>()
        );
        cli(f.root.path(), true).await;
        let refreshed = vectors();
        let changed: Vec<i64> = baseline
            .iter()
            .filter_map(|(id, old)| (refreshed.get(id) != Some(old)).then_some(*id))
            .collect();
        assert_eq!(changed, affected);
        let calls_refreshed = server.received_requests().await.unwrap().len();
        assert_eq!(calls_refreshed - calls_default, affected.len() - 1);
        cli(f.root.path(), true).await;
        assert_eq!(vectors(), refreshed);
        assert_eq!(
            server.received_requests().await.unwrap().len(),
            calls_refreshed
        );
        f.config.embedder.reembed_on_role_change = true;
        fs::write(
            f.root.path().join(".deepkit/config.toml"),
            toml::to_string(&f.config).unwrap(),
        )
        .unwrap();
        f.config = crate::config::load(f.root.path(), None).unwrap().config;
        f.publish(f.module, "Receivables\nHandles invoices.");
        cli(f.root.path(), false).await;
        let automatic = vectors();
        let changed: Vec<i64> = refreshed
            .iter()
            .filter_map(|(id, old)| (automatic.get(id) != Some(old)).then_some(*id))
            .collect();
        assert_eq!(changed, affected);
        assert_eq!(
            server.received_requests().await.unwrap().len() - calls_refreshed,
            affected.len() + 1
        );
        cli(f.root.path(), false).await;
        assert_eq!(vectors(), automatic);
        assert_eq!(
            server.received_requests().await.unwrap().len() - calls_refreshed,
            affected.len() + 1
        );
    }

    struct Fixture {
        root: tempfile::TempDir,
        store: std::sync::Arc<Store>,
        config: Config,
        module: i64,
        unit: i64,
    }
    impl Fixture {
        fn new() -> Self {
            Self::with_page(false)
        }
        fn with_page(page: bool) -> Self {
            Self::with_visual(page, false)
        }
        fn with_visual(page: bool, image: bool) -> Self {
            let root = tempfile::tempdir().unwrap();
            fs::create_dir(root.path().join(".deepkit")).unwrap();
            fs::create_dir(root.path().join("billing")).unwrap();
            fs::write(root.path().join("billing/a.rs"), "pub fn retry() {}\n").unwrap();
            fs::write(root.path().join("billing/b.rs"), "pub fn invoice() {}\n").unwrap();
            let mut config = Config::default();
            config.index.repo_name = "fixture".into();
            config.commits.enabled = false;
            config.llm.enabled = true;
            config.llm.url = "http://127.0.0.1:1/v1".into();
            config.llm.model = "fixture-text".into();
            config.enrich.modules_min_files = 2;
            config.enrich.top_symbols = 0;
            config.treatments = vec![crate::config::Treatment {
                name: "code".into(),
                patterns: vec!["**/*.rs".into()],
                max_level: 3,
                enrich: vec!["describe_symbol".into()],
                ..Default::default()
            }];
            if image {
                config.llm.vlm_model = "fixture-vision".into();
                config.enrich.images = true;
                config.treatments.push(crate::config::Treatment {
                    name: "image".into(),
                    patterns: vec!["**/*.png".into()],
                    max_level: 3,
                    enrich: vec!["describe_image".into()],
                    ..Default::default()
                });
                let mut bytes = std::io::Cursor::new(Vec::new());
                image::DynamicImage::new_rgb8(2, 2)
                    .write_to(&mut bytes, image::ImageFormat::Png)
                    .unwrap();
                fs::write(root.path().join("billing/diagram.png"), bytes.into_inner()).unwrap();
            }
            if page {
                config.enrich.scanned_pages = true;
                config.llm.vlm_model = "fixture-vision".into();
                config.treatments.push(crate::config::Treatment {
                    name: "scan".into(),
                    patterns: vec!["**/*.pdf".into()],
                    max_level: 3,
                    chunker: Some("pdf".into()),
                    enrich: vec!["vlm_page".into()],
                    ..Default::default()
                });
                fs::write(
                    root.path().join("billing/scan.pdf"),
                    include_bytes!("../../fixtures/docs/scanned-jpeg.pdf"),
                )
                .unwrap();
            }
            fs::write(
                root.path().join(".deepkit/config.toml"),
                toml::to_string(&config).unwrap(),
            )
            .unwrap();
            let config = crate::config::load(root.path(), None).unwrap().config;
            let store =
                store::open(root.path().join(".deepkit/code.db"), Default::default()).unwrap();
            sync::index::run(
                sync::index::Input {
                    config_root: root.path(),
                    root: root.path(),
                    settings: &config,
                    store: &store,
                    full: false,
                    work: Work::default(),
                },
                |_| {},
            )
            .unwrap();
            let (module, unit) = {
                let db = store.writer().unwrap();
                let module = db
                    .query_row(
                        "SELECT module_unit FROM dirs WHERE path='billing'",
                        [],
                        |r| r.get(0),
                    )
                    .unwrap();
                let unit = db.query_row("SELECT u.id FROM units u JOIN files f ON f.id=u.file_id WHERE f.path='billing/a.rs' AND u.kind IN ('chunk','symbol') ORDER BY u.id LIMIT 1", [], |r| r.get(0)).unwrap();
                (module, unit)
            };
            Self {
                root,
                store: std::sync::Arc::new(store),
                config,
                module,
                unit,
            }
        }
        fn desired(&self, module: i64) -> [u8; 32] {
            self.store
                .read_snapshot(|tx| {
                    let input = enrichment_inputs::module_with_work(
                        tx,
                        self.root.path(),
                        self.root.path(),
                        &self.config,
                        module,
                        Work::default(),
                    )?
                    .unwrap();
                    enrichment_inputs::desired(&self.config, Kind::SummarizeModule, &input)
                })
                .unwrap()
                .unwrap()
        }
        fn publish(&self, module: i64, text: &str) -> i64 {
            let mut db = self.store.writer().unwrap();
            let tx = db.transaction().unwrap();
            let input = enrichment_inputs::module_with_work(
                &tx,
                self.root.path(),
                self.root.path(),
                &self.config,
                module,
                Work::default(),
            )
            .unwrap()
            .unwrap();
            let desired =
                enrichment_inputs::desired(&self.config, Kind::SummarizeModule, &input).unwrap();
            let id = self
                .store
                .publish_summary_in(
                    &tx,
                    Publication {
                        owner: input.owner,
                        parent_unit: input.parent_unit,
                        parent_hash: &input.parent_hash,
                        desired_hash: &desired,
                        text,
                        sources: &input.sources,
                        now: 1,
                    },
                )
                .unwrap();
            tx.commit().unwrap();
            id
        }
        fn input(&self, unit: i64) -> FileInput {
            self.store
                .read_snapshot(|tx| {
                    source::resolve_with_work(
                        tx,
                        self.root.path(),
                        self.root.path(),
                        &self.config,
                        unit,
                        Work::default(),
                    )
                })
                .unwrap()
                .unwrap()
                .unwrap()
        }
        fn identity(&self, unit: i64) -> embedding::Prepared {
            let input = self.input(unit);
            embedding::Prepared::new(
                input.occurrence,
                input.context,
                embedding::Destination {
                    provider: "embed:remote".into(),
                    endpoint: "http://127.0.0.1:1/v1".into(),
                    model: "fixture".into(),
                    artifact_hash: None,
                    dims: 2,
                },
                input.policy,
            )
            .unwrap()
            .unwrap()
        }
    }

    #[test]
    fn directory_rename_and_removal_retire_old_roles_and_dirty_markers() {
        for remove in [false, true] {
            let fixture = Fixture::new();
            fixture.publish(fixture.module, "Payments");
            reconcile(
                &fixture.store,
                fixture.root.path(),
                fixture.root.path(),
                &fixture.config,
                Work::default(),
            )
            .unwrap();
            if remove {
                fs::remove_dir_all(fixture.root.path().join("billing")).unwrap();
            } else {
                fs::rename(
                    fixture.root.path().join("billing"),
                    fixture.root.path().join("ledger"),
                )
                .unwrap();
            }
            sync::index::run(
                sync::index::Input {
                    root: fixture.root.path(),
                    config_root: fixture.root.path(),
                    settings: &fixture.config,
                    store: &fixture.store,
                    full: false,
                    work: Work::default(),
                },
                |_| {},
            )
            .unwrap();
            let db = fixture.store.writer().unwrap();
            let stale: i64 = db
                .query_row(
                    "SELECT count(*) FROM fts_units WHERE fts_units MATCH 'header : Payments'",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(stale, 0, "old directory role survived; removed={remove}");
            let pending: i64 = db
                .query_row(
                    "SELECT count(*) FROM meta WHERE key GLOB 'lexical_role_dirty:*'",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(pending, 0);
            let receipt: String = db
                .query_row(
                    "SELECT value FROM meta WHERE key='lexical_role_receipt_v1'",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert!(
                !serde_json::from_str::<serde_json::Value>(&receipt)
                    .unwrap()
                    .as_object()
                    .unwrap()
                    .contains_key("billing")
            );
            let files: i64 = db
                .query_row(
                    "SELECT count(*) FROM files WHERE path GLOB 'ledger/*'",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(files, if remove { 0 } else { 2 });
        }
    }

    #[test]
    fn full_reindex_and_disabling_roles_update_all_retained_headers() {
        for disable in [false, true] {
            let fixture = Fixture::new();
            fixture.publish(fixture.module, "Payments");
            reconcile(
                &fixture.store,
                fixture.root.path(),
                fixture.root.path(),
                &fixture.config,
                Work::default(),
            )
            .unwrap();
            let mut config = fixture.config.clone();
            if disable {
                config.embedder.breadcrumb_roles = false;
                fs::write(
                    fixture.root.path().join(".deepkit/config.toml"),
                    toml::to_string(&config).unwrap(),
                )
                .unwrap();
                config = crate::config::load(fixture.root.path(), None)
                    .unwrap()
                    .config;
            }
            sync::index::run(
                sync::index::Input {
                    root: fixture.root.path(),
                    config_root: fixture.root.path(),
                    settings: &config,
                    store: &fixture.store,
                    full: true,
                    work: Work::default(),
                },
                |_| {},
            )
            .unwrap();
            let db = fixture.store.writer().unwrap();
            let current: i64 = db.query_row("SELECT count(*) FROM fts_units WHERE rowid=?1 AND fts_units MATCH 'header : Payments'",[fixture.unit],|r|r.get(0)).unwrap();
            assert_eq!(
                current,
                i64::from(!disable),
                "role header after full index; disabled={disable}"
            );
            if disable {
                let remaining: i64 = db
                    .query_row(
                        "SELECT count(*) FROM fts_units WHERE fts_units MATCH 'header : Payments'",
                        [],
                        |r| r.get(0),
                    )
                    .unwrap();
                assert_eq!(
                    remaining, 0,
                    "disabled roles must also disappear from retained composite headers"
                );
            }
        }
    }

    #[test]
    fn status_reports_role_context_mismatch_in_each_retained_vector_table() {
        let fixture = Fixture::new();
        let baseline = fixture.identity(fixture.unit);
        {
            let mut db = fixture.store.writer().unwrap();
            db.execute_batch("INSERT INTO embedding_tables(slug,model,profile,dims,created_at) VALUES('role_a','fixture','remote',2,0),('role_b','fixture','remote',2,0);
                CREATE TABLE vec_role_a(unit_id INTEGER PRIMARY KEY REFERENCES units(id),ctx_hash BLOB);
                CREATE TABLE vec_role_b(unit_id INTEGER PRIMARY KEY REFERENCES units(id),ctx_hash BLOB);").unwrap();
            for table in ["vec_role_a", "vec_role_b"] {
                db.execute(
                    &format!("INSERT INTO {table} VALUES(?1,?2)"),
                    rusqlite::params![fixture.unit, baseline.context_hash()],
                )
                .unwrap();
            }
            let tx = db.transaction().unwrap();
            crate::store::unit_contexts::record(
                &tx,
                fixture.unit,
                &baseline.current().target_hash,
                baseline.context_hash(),
                &fixture.config,
            )
            .unwrap();
            tx.commit().unwrap();
        }
        let counts = || {
            serde_json::to_value(
                crate::queue::inspection::read_configured(&fixture.store, &fixture.config).unwrap(),
            )
            .unwrap()
        };
        for table in counts()["l2"]["tables"].as_array().unwrap() {
            assert_eq!(table["context"]["current"], 1);
        }
        fixture.publish(fixture.module, "Payments");
        reconcile(
            &fixture.store,
            fixture.root.path(),
            fixture.root.path(),
            &fixture.config,
            Work::default(),
        )
        .unwrap();
        for table in counts()["l2"]["tables"].as_array().unwrap() {
            assert_eq!(table["context"]["stale"], 1);
            assert_eq!(table["context"]["unknown"], 0);
            assert_eq!(table["vectors"], 1);
        }
    }

    #[test]
    fn indexing_removes_applied_role_after_module_source_changes_without_repeated_replay() {
        let fixture = Fixture::new();
        fixture.publish(fixture.module, "Payments");
        assert!(
            reconcile(
                &fixture.store,
                fixture.root.path(),
                fixture.root.path(),
                &fixture.config,
                Work::default()
            )
            .unwrap()
                > 0
        );
        let hits = || {
            fixture.store.writer().unwrap().query_row(
            "SELECT count(*) FROM fts_units WHERE rowid=?1 AND fts_units MATCH 'header : Payments'",
            [fixture.unit], |r| r.get::<_, i64>(0),
        ).unwrap()
        };
        assert_eq!(hits(), 1);
        fs::write(
            fixture.root.path().join("billing/b.rs"),
            "pub fn changed_invoice() {}\n",
        )
        .unwrap();
        sync::index::run(
            sync::index::Input {
                root: fixture.root.path(),
                config_root: fixture.root.path(),
                settings: &fixture.config,
                store: &fixture.store,
                full: false,
                work: Work::default(),
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(hits(), 0);
        let before = fixture.store.writer().unwrap().total_changes();
        assert_eq!(
            reconcile(
                &fixture.store,
                fixture.root.path(),
                fixture.root.path(),
                &fixture.config,
                Work::default()
            )
            .unwrap(),
            0
        );
        assert_eq!(fixture.store.writer().unwrap().total_changes(), before);
        // Even an occupied writer must not block a proven unchanged receipt.
        let writer = fixture.store.writer().unwrap();
        assert_eq!(
            reconcile(
                &fixture.store,
                fixture.root.path(),
                fixture.root.path(),
                &fixture.config,
                Work {
                    deadline: Some(std::time::Instant::now() + std::time::Duration::from_secs(1)),
                    cancelled: None
                }
            )
            .unwrap(),
            0
        );
        assert_eq!(writer.total_changes(), before);
    }

    #[test]
    fn reconciliation_clears_pending_marker_without_changed_roles() {
        for receipt in [None, Some("{}")] {
            let fixture = Fixture::new();
            {
                let db = fixture.store.writer().unwrap();
                assert_eq!(
                    db.query_row(
                        "SELECT count(*) FROM dirs WHERE summary_unit IS NOT NULL",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap(),
                    0
                );
                db.execute("DELETE FROM meta WHERE key='lexical_role_receipt_v1'", [])
                    .unwrap();
                if let Some(receipt) = receipt {
                    db.execute(
                        "INSERT INTO meta(key,value) VALUES('lexical_role_receipt_v1',?1)",
                        [receipt],
                    )
                    .unwrap();
                }
                db.execute(
                    "INSERT INTO meta(key,value) VALUES('lexical_role_reconcile_pending','1')",
                    [],
                )
                .unwrap();
            }
            assert_eq!(
                reconcile(
                    &fixture.store,
                    fixture.root.path(),
                    fixture.root.path(),
                    &fixture.config,
                    Work::default(),
                )
                .unwrap(),
                0
            );
            let writer = fixture.store.writer().unwrap();
            assert!(
                !writer
                    .query_row(
                        "SELECT EXISTS(SELECT 1 FROM meta WHERE key='lexical_role_reconcile_pending')",
                        [],
                        |row| row.get::<_, bool>(0),
                    )
                    .unwrap()
            );
            let retained: Option<String> = writer
                .query_row(
                    "SELECT (SELECT value FROM meta WHERE key='lexical_role_receipt_v1')",
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(retained.as_deref(), receipt);
            let before = writer.total_changes();
            // Once the marker is cleared, the no-op must avoid writer admission.
            assert_eq!(
                reconcile(
                    &fixture.store,
                    fixture.root.path(),
                    fixture.root.path(),
                    &fixture.config,
                    Work {
                        deadline: Some(
                            std::time::Instant::now() + std::time::Duration::from_secs(1)
                        ),
                        cancelled: None,
                    },
                )
                .unwrap(),
                0
            );
            assert_eq!(writer.total_changes(), before);
        }
    }

    #[test]
    fn scanned_module_sources_still_reject_caps_pending_and_failed_extraction() {
        let fixture = Fixture::with_page(true);
        let mut db = fixture.store.writer().unwrap();
        for reason in [
            "needs_vlm",
            "no_renderer",
            "max_pdf_pages",
            "max_doc_bytes",
            "parse_error",
        ] {
            let tx = db.transaction().unwrap();
            tx.execute(
                "UPDATE files SET l1_deferred_reason=?1 WHERE path='billing/scan.pdf'",
                [reason],
            )
            .unwrap();
            let snapshot = store::modules::snapshot(&tx, "billing", 2, &|| Ok(())).unwrap();
            assert_eq!(
                snapshot.is_some(),
                matches!(reason, "needs_vlm" | "no_renderer"),
                "{reason}"
            );
            tx.execute(
                "UPDATE files SET l1_pending=1 WHERE path='billing/scan.pdf'",
                [],
            )
            .unwrap();
            assert!(
                store::modules::snapshot(&tx, "billing", 2, &|| Ok(()))
                    .unwrap()
                    .is_none()
            );
            tx.rollback().unwrap();
        }
    }

    #[test]
    fn completed_enrichment_cohort_rejects_stale_role_until_module_is_refreshed() {
        use crate::queue::{enrichment, worker::Work as _};
        use std::{sync::Arc, time::Duration};
        let fixture = Fixture::new();
        fixture.publish(fixture.module, "Payments");
        reconcile(
            &fixture.store,
            fixture.root.path(),
            fixture.root.path(),
            &fixture.config,
            Work::default(),
        )
        .unwrap();
        assert_eq!(fixture.store.writer().unwrap().query_row("SELECT count(*) FROM fts_units WHERE rowid=?1 AND fts_units MATCH 'header : Payments'", [fixture.unit], |r| r.get::<_,i64>(0)).unwrap(), 1);
        let client = crate::models::openai_compat::chat::Client::with_key(
            &fixture.config.llm,
            Duration::from_secs(3),
            1024 * 1024,
            "fixture".into(),
        )
        .unwrap();
        let env = Arc::new(enrichment::Context {
            root: fixture.root.path().into(),
            config_root: fixture.root.path().into(),
            store: fixture.store.clone(),
            config: Arc::new(fixture.config.clone()),
            client: Arc::new(client),
        });
        let symbol: i64 = env
            .store
            .writer()
            .unwrap()
            .query_row("SELECT id FROM symbols WHERE name='retry'", [], |r| {
                r.get(0)
            })
            .unwrap();
        let job = env
            .store
            .read_snapshot(|tx| env.prepare_job(tx, Kind::DescribeSymbol, symbol))
            .unwrap()
            .unwrap()
            .unwrap();
        let mut db = env.store.writer().unwrap();
        let tx = db.transaction().unwrap();
        job.publish_with_work(&tx, "Retries a failed request.".into(), Work::default())
            .unwrap();
        tx.commit().unwrap();
        assert!(
            db.query_row(
                "SELECT EXISTS(SELECT 1 FROM meta WHERE key='lexical_role_reconcile_pending')",
                [],
                |row| row.get::<_, bool>(0),
            )
            .unwrap(),
            "publication must leave a durable reconciliation receipt for the host"
        );
        // Enrichment publishes a finite cohort before the host reconciles
        // roles. That avoids replaying every derived row for each symbol while
        // retaining the same fresh-role fence before the cohort returns.
        drop(db);
        reconcile_with_batch_size(
            &env.store,
            &env.root,
            &env.config_root,
            &env.config,
            Work::default(),
            1,
        )
        .unwrap();
        let db = env.store.writer().unwrap();
        assert!(
            !db.query_row(
                "SELECT EXISTS(SELECT 1 FROM meta WHERE key='lexical_role_reconcile_pending')",
                [],
                |row| row.get::<_, bool>(0),
            )
            .unwrap(),
            "only a completed reconciliation clears the publication receipt"
        );
        let matches: i64 = db.query_row("SELECT count(*) FROM fts_units WHERE rowid=(SELECT summary_unit FROM symbols WHERE id=?1) AND fts_units MATCH 'header : Payments'", [symbol], |r| r.get(0)).unwrap();
        assert_eq!(
            matches, 0,
            "new symbol text invalidates the prior module prompt"
        );
        assert_eq!(db.query_row("SELECT count(*) FROM fts_units WHERE rowid=?1 AND fts_units MATCH 'header : Payments'", [fixture.unit], |r| r.get::<_,i64>(0)).unwrap(), 0, "sibling lexical headers must lose the stale role immediately");
        assert_eq!(
            db.query_row(
                "SELECT count(*) FROM fts_units WHERE fts_units MATCH 'header : Payments'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
            0,
            "retained stale summaries must lose unsupported role headers too"
        );
        drop(db);
        let module = env
            .store
            .read_snapshot(|tx| env.prepare_job(tx, Kind::SummarizeModule, fixture.module))
            .unwrap()
            .unwrap()
            .unwrap();
        let mut db = env.store.writer().unwrap();
        let tx = db.transaction().unwrap();
        module
            .publish_with_work(&tx, "Payments".into(), Work::default())
            .unwrap();
        tx.commit().unwrap();
        assert!(
            db.query_row(
                "SELECT EXISTS(SELECT 1 FROM meta WHERE key='lexical_role_reconcile_pending')",
                [],
                |row| row.get::<_, bool>(0),
            )
            .unwrap(),
            "module publication must defer the shared role replay to its host cohort"
        );
        drop(db);
        reconcile_with_batch_size(
            &env.store,
            &env.root,
            &env.config_root,
            &env.config,
            Work::default(),
            1,
        )
        .unwrap();
        let db = env.store.writer().unwrap();
        let matches: i64 = db.query_row("SELECT count(*) FROM fts_units WHERE rowid=(SELECT summary_unit FROM symbols WHERE id=?1) AND fts_units MATCH 'header : Payments'", [symbol], |r| r.get(0)).unwrap();
        assert_eq!(matches, 1);
    }

    #[test]
    fn image_description_roles_refresh_without_changing_body_or_parent_identity() {
        use crate::queue::{enrichment, worker::Work as _};
        use std::{sync::Arc, time::Duration};
        let fixture = Fixture::with_visual(false, true);
        fixture.publish(fixture.module, "Payments");
        let client = crate::models::openai_compat::chat::Client::with_key(
            &fixture.config.llm,
            Duration::from_secs(3),
            1024 * 1024,
            "fixture".into(),
        )
        .unwrap();
        let env = Arc::new(enrichment::Context {
            root: fixture.root.path().into(),
            config_root: fixture.root.path().into(),
            store: fixture.store.clone(),
            config: Arc::new(fixture.config.clone()),
            client: Arc::new(client),
        });
        let file: i64 = env
            .store
            .writer()
            .unwrap()
            .query_row(
                "SELECT id FROM files WHERE path='billing/diagram.png'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let job = env
            .store
            .read_snapshot(|tx| env.prepare_job(tx, Kind::DescribeImage, file))
            .unwrap()
            .unwrap()
            .unwrap();
        let mut db = env.store.writer().unwrap();
        let tx = db.transaction().unwrap();
        job.publish_with_work(
            &tx,
            "A quartz lantern and an invoice arrow.".into(),
            Work::default(),
        )
        .unwrap();
        tx.commit().unwrap();
        let (unit, parent): (i64, i64) = db.query_row(
            "SELECT u.id,p.id FROM units u JOIN units p ON p.id=u.parent_unit WHERE p.file_id=?1 AND u.kind='summary'",
            [file], |r| Ok((r.get(0)?,r.get(1)?)),
        ).unwrap();
        db.execute_batch(
            "CREATE VIRTUAL TABLE temp.image_role_vocab USING fts5vocab(main,fts_units,'instance')",
        )
        .unwrap();
        let terms = |db: &rusqlite::Connection| {
            db.prepare("SELECT term,col,offset FROM temp.image_role_vocab WHERE doc=?1 AND col!='header' ORDER BY col,offset,term").unwrap()
                .query_map([unit], |r| Ok((r.get::<_, String>(0)?,r.get::<_, String>(1)?,r.get::<_, i64>(2)?))).unwrap()
                .collect::<Result<Vec<_>,_>>().unwrap()
        };
        let role_hits = |db: &rusqlite::Connection, role: &str| -> i64 {
            db.query_row(
                "SELECT count(*) FROM fts_units WHERE rowid=?1 AND fts_units MATCH ?2",
                rusqlite::params![unit, format!("header : {role}")],
                |r| r.get(0),
            )
            .unwrap()
        };
        let before = terms(&db);
        assert!(!before.is_empty());
        assert_eq!(role_hits(&db, "Payments"), 1);
        drop(db);
        fixture.publish(fixture.module, "Invoicing");
        reconcile(
            &env.store,
            &env.root,
            &env.config_root,
            &env.config,
            Work::default(),
        )
        .unwrap();
        let db = env.store.writer().unwrap();
        assert_eq!(role_hits(&db, "Payments"), 0);
        assert_eq!(role_hits(&db, "Invoicing"), 1);
        assert_eq!(terms(&db), before);
        assert_eq!(
            db.query_row("SELECT parent_unit FROM units WHERE id=?1", [unit], |r| r
                .get::<_, i64>(
                0
            ))
            .unwrap(),
            parent
        );
        assert_eq!(
            db.query_row("SELECT kind FROM units WHERE id=?1", [parent], |r| r
                .get::<_, String>(0))
                .unwrap(),
            "image"
        );
    }

    #[test]
    fn generated_page_role_headers_preserve_fenced_body_and_refresh_after_publication() {
        use crate::queue::{enrichment, worker::Work as _};
        use std::{sync::Arc, time::Duration};
        let fixture = Fixture::with_page(true);
        fixture.publish(fixture.module, "Payments");
        let client = crate::models::openai_compat::chat::Client::with_key(
            &fixture.config.llm,
            Duration::from_secs(3),
            1024 * 1024,
            "fixture".into(),
        )
        .unwrap();
        let env = Arc::new(enrichment::Context {
            root: fixture.root.path().into(),
            config_root: fixture.root.path().into(),
            store: fixture.store.clone(),
            config: Arc::new(fixture.config.clone()),
            client: Arc::new(client),
        });
        let file: i64 = env
            .store
            .writer()
            .unwrap()
            .query_row(
                "SELECT id FROM files WHERE path='billing/scan.pdf'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let job = env
            .store
            .read_snapshot(|tx| env.prepare_page_with_work(tx, file, 1, Work::default()))
            .unwrap()
            .unwrap()
            .unwrap();
        let mut db = env.store.writer().unwrap();
        let tx = db.transaction().unwrap();
        job.publish_with_work(
            &tx,
            "# Guide\n\nPayment workflow.\n\n```rust\nfn retry_invoice() {}\n```".into(),
            Work::default(),
        )
        .unwrap();
        tx.commit().unwrap();
        db.execute_batch(
            "CREATE VIRTUAL TABLE temp.page_role_vocab USING fts5vocab(main,fts_units,'instance')",
        )
        .unwrap();
        let unit: i64 = db
            .query_row(
                "SELECT id FROM units WHERE file_id=?1 AND kind='doc_section' LIMIT 1",
                [file],
                |r| r.get(0),
            )
            .unwrap();
        let matches = |db: &rusqlite::Connection, word: &str| -> i64 {
            db.query_row(
                "SELECT count(*) FROM fts_units WHERE rowid=?1 AND fts_units MATCH ?2",
                rusqlite::params![unit, format!("header : {word}")],
                |r| r.get(0),
            )
            .unwrap()
        };
        let tokens = |db: &rusqlite::Connection| {
            db.prepare("SELECT term,col,offset FROM temp.page_role_vocab WHERE doc=?1 AND col!='header' ORDER BY col,offset,term").unwrap()
                .query_map([unit], |r| Ok((r.get::<_, String>(0)?,r.get::<_, String>(1)?,r.get::<_, i64>(2)?))).unwrap().collect::<Result<Vec<_>,_>>().unwrap()
        };
        assert_eq!(matches(&db, "Payments"), 1);
        let before = tokens(&db);
        assert!(
            before
                .iter()
                .any(|(term, col, _)| term == "retry" && col == "body")
        );
        drop(db);
        fixture.publish(fixture.module, "Invoicing");
        let mut db = env.store.writer().unwrap();
        let tx = db.transaction().unwrap();
        refresh_module_files(
            &tx,
            &env.root,
            &env.config_root,
            &env.config,
            fixture.module,
            Work::default(),
        )
        .unwrap();
        tx.commit().unwrap();
        assert_eq!(matches(&db, "Payments"), 0);
        assert_eq!(matches(&db, "Invoicing"), 1);
        assert_eq!(tokens(&db), before);
        // A source change after preparation must roll back even when lexical
        // resolution returns None instead of raising an error.
        fs::write(env.root.join("billing/scan.pdf"), b"changed source").unwrap();
        let tx = db.transaction().unwrap();
        assert!(
            job.publish_with_work(
                &tx,
                "# Replaced\n\nUntrusted replacement.".into(),
                Work::default()
            )
            .is_err()
        );
        tx.rollback().unwrap();
        assert_eq!(tokens(&db), before);
        assert_eq!(matches(&db, "Invoicing"), 1);
    }

    #[test]
    fn lexical_role_refresh_preserves_body_positions_and_occurrence_identity() {
        let fixture = Fixture::new();
        fixture.store.writer().unwrap().execute_batch("CREATE VIRTUAL TABLE temp.role_vocabulary USING fts5vocab(main,fts_units,'instance')").unwrap();
        let tokens = |unit: i64| {
            let db = fixture.store.writer().unwrap();
            let mut query = db.prepare("SELECT term,col,offset FROM temp.role_vocabulary WHERE doc=?1 AND col!='header' ORDER BY col,offset,term").unwrap();
            query
                .query_map([unit], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, i64>(2)?,
                    ))
                })
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        };
        let before = tokens(fixture.unit);
        assert!(!before.is_empty());
        let matches = |unit: i64, term: &str| {
            fixture
                .store
                .writer()
                .unwrap()
                .query_row(
                    "SELECT count(*) FROM fts_units WHERE fts_units MATCH ?1 AND rowid=?2",
                    rusqlite::params![format!("header : {term}"), unit],
                    |r| r.get::<_, i64>(0),
                )
                .unwrap()
        };
        for (role, previous) in [("Payments", "Absent"), ("Invoicing", "Payments")] {
            let summary = fixture.publish(fixture.module, &format!("{role}\n\nHandles requests."));
            let module_tokens = tokens(fixture.module);
            let summary_tokens = tokens(summary);
            let mut db = fixture.store.writer().unwrap();
            let tx = db.transaction().unwrap();
            refresh_module_files(
                &tx,
                fixture.root.path(),
                fixture.root.path(),
                &fixture.config,
                fixture.module,
                Work::default(),
            )
            .unwrap();
            tx.commit().unwrap();
            drop(db);
            assert_eq!(matches(fixture.unit, role), 1);
            assert_eq!(matches(fixture.module, role), 1);
            assert_eq!(matches(summary, role), 1);
            assert_eq!(tokens(fixture.module), module_tokens);
            assert_eq!(tokens(summary), summary_tokens);
            assert_eq!(matches(fixture.unit, previous), 0);
            assert_eq!(tokens(fixture.unit), before);
            assert_eq!(fixture.input(fixture.unit).occurrence.id, fixture.unit);
        }
    }

    #[test]
    fn current_role_changes_context_without_changing_its_own_summary_inputs() {
        let fixture = Fixture::new();
        let before = fixture.identity(fixture.unit);
        let desired = fixture.desired(fixture.module);
        let summary = fixture.publish(
            fixture.module,
            "Payment retries\n\nHandles failed payments.",
        );
        let input = fixture.input(fixture.unit);
        assert_eq!(
            input.context.breadcrumb,
            "fixture › billing (Payment retries) › a.rs"
        );
        let after = fixture.identity(fixture.unit);
        assert_eq!(before.role_free_hash(), after.role_free_hash());
        assert_ne!(before.context_hash(), after.context_hash());
        assert_eq!(fixture.desired(fixture.module), desired);
        assert_eq!(
            fixture.input(summary).context.breadcrumb,
            "fixture › billing (Payment retries)"
        );
        fixture.publish(
            fixture.module,
            "Payment retries\n\nChanged second paragraph.",
        );
        assert_eq!(
            fixture.identity(fixture.unit).current().desired_hash,
            after.current().desired_hash
        );
        fixture.publish(fixture.module, "Invoice processing\n\nChanged role.");
        let changed = fixture.identity(fixture.unit);
        assert_ne!(changed.context_hash(), after.context_hash());
        assert_eq!(changed.role_free_hash(), after.role_free_hash());
        assert_eq!(fixture.desired(fixture.module), desired);
    }

    #[test]
    fn root_and_directory_roles_cover_modules_and_stale_contributors_remove_roles() {
        let fixture = Fixture::new();
        let root: i64 = fixture
            .store
            .writer()
            .unwrap()
            .query_row("SELECT module_unit FROM dirs WHERE path=''", [], |r| {
                r.get(0)
            })
            .unwrap();
        fixture.publish(root, "Repository services");
        fixture.publish(fixture.module, "Payment retries");
        assert_eq!(
            fixture.input(fixture.module).context.breadcrumb,
            "fixture (Repository services) › billing (Payment retries)"
        );
        assert_eq!(
            fixture.input(fixture.unit).context.breadcrumb,
            "fixture (Repository services) › billing (Payment retries) › a.rs"
        );
        fs::write(
            fixture.root.path().join("billing/b.rs"),
            "pub fn changed_invoice() {}\n",
        )
        .unwrap();
        assert_eq!(
            fixture.input(fixture.unit).context.breadcrumb,
            "fixture › billing › a.rs"
        );
    }

    #[test]
    fn ancestors_include_root_and_the_module_itself_but_never_a_file() {
        assert_eq!(
            ancestors("billing/src/a.rs", false),
            ["", "billing", "billing/src"]
        );
        assert_eq!(
            ancestors("billing/src", true),
            ["", "billing", "billing/src"]
        );
        assert_eq!(ancestors("a.rs", false), [""]);
        assert_eq!(ancestors("", true), [""]);
    }
}
