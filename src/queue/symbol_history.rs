//! Verify symbol history ancestry, context and transitive source policy; v1 §6.6/§8.2.
use super::{
    Error,
    embedding::{Occurrence, Policy},
    source::{FileInput, check_work},
};
use crate::{
    config::Config,
    store::{self, UnitKind},
    sync::{self, l1::Work},
    treat::{self, context},
};
use rusqlite::{OptionalExtension, Transaction};
use std::path::Path;
/// Retained out-of-rank histories stay stored but are inactive under current settings.
#[allow(
    clippy::too_many_arguments,
    reason = "shared source resolver's explicit authority inputs"
)]
pub(super) fn resolve(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    current: &Config,
    unit: i64,
    minimum: i64,
    permit_protected: bool,
    work: Work<'_>,
) -> Result<Option<FileInput>, Error> {
    check_work(work)?;
    if !current.commits.enabled || current.enrich.top_symbols == 0 {
        return Ok(None);
    }
    let Some(target) = store::symbol_history::owner(tx, unit)? else {
        return Ok(None);
    };
    if !store::symbol_history::is_selected(tx, target.id, current.enrich.top_symbols)? {
        return Ok(None);
    }
    let check = || work.check().map_err(|_| store::Error::L1InputsChanged);
    let entries = store::symbol_history::entries(tx, &target, &check)?;
    if entries.is_empty() || !store::symbol_history::sources_match(tx, unit, &entries)? {
        return Ok(None);
    }
    // The parent supplies source/privacy authority, never its rendered context.
    // Roles are resolved once for the history itself by the outer source resolver;
    // rebuilding them here would duplicate the same ancestor proof at each fence.
    // owner() restricts the parent to completed code, excluding recursive composites.
    let Some(parent) = super::source::resolve_freshness_without_roles(
        tx,
        root,
        config_root,
        current,
        target.parent,
        work,
    )?
    else {
        return Ok(None);
    };
    let canonical =
        treat::symbol_history::collate(&entries).map_err(|_| Error::Invalid("history date"))?;
    let breadcrumb = context::breadcrumb(
        context::ContextFile {
            repo_name: &sync::l1::repo_name(root, current),
            path: &target.path,
        },
        None,
        &[],
        true,
    );
    let header = format!("history of {}", target.qualified);
    let valid=tx.query_row("SELECT u.content_hash=?2 AND u.scope_path=?3 AND u.header=?4 AND c.text=?5 AND u.parent_unit=?6 FROM units u JOIN unit_contents c ON c.hash=u.content_hash WHERE u.id=?1 AND u.kind='history' AND u.file_id IS NULL AND u.ladder_level>=1",rusqlite::params![unit,canonical.hash.as_slice(),breadcrumb,header,canonical.text,target.parent],|r|r.get::<_,bool>(0)).optional()?.unwrap_or(false);
    if !valid {
        return Ok(None);
    }
    let mut max_level = 0;
    let mut never_send = parent.policy.never_send;
    let mut fingerprint = blake3::Hasher::new_derive_key("dkc symbol history policy v1");
    fingerprint.update(&parent.policy.fingerprint);
    for entry in entries {
        check_work(work)?;
        let Some(source) = super::history_source::resolve(
            tx,
            root,
            config_root,
            current,
            entry.unit,
            0,
            true,
            work,
        )?
        else {
            return Ok(None);
        };
        max_level = max_level.max(source.policy.max_level);
        never_send |= source.policy.never_send;
        fingerprint.update(&entry.sha);
        fingerprint.update(&source.policy.fingerprint);
    }
    if max_level < minimum || (never_send && !permit_protected) {
        return Ok(None);
    }
    check_work(work)?;
    Ok(Some(FileInput {
        occurrence: Occurrence {
            id: unit,
            content_hash: canonical.hash,
            kind: UnitKind::History,
        },
        context: context::history_input(
            &breadcrumb,
            &header,
            "history",
            &canonical.text,
            context::Options::from(&current.embedder),
        ),
        policy: Policy {
            fingerprint: *fingerprint.finalize().as_bytes(),
            max_level,
            never_send,
        },
    }))
}
