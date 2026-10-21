//! Canonical PR authority through every immutable member and current policy; v1 §6.6/§8.2.
use super::{
    Error,
    embedding::{Occurrence, Policy},
    source::{FileInput, check_work},
};
use crate::{
    config::{Config, pr_patterns},
    store::{Store, UnitKind},
    sync::l1::Work,
    treat::{context, pr},
};
use rusqlite::{OptionalExtension, Transaction};
use std::{collections::BTreeSet, path::Path};

#[allow(
    clippy::too_many_arguments,
    reason = "shared resolver's explicit authority inputs"
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
    if !current.commits.enabled {
        return Ok(None);
    }
    type Row = (
        i64,
        Option<i64>,
        String,
        Vec<u8>,
        Option<Vec<u8>>,
        Vec<u8>,
        String,
        i64,
        i64,
        Vec<u8>,
        String,
        String,
        String,
    );
    let row:Option<Row>=tx.query_row("SELECT p.id,p.number,p.title,p.merge_sha,p.base_sha,p.head_sha,p.strategy,p.commit_count,p.merged_at,u.content_hash,u.scope_path,u.header,c.text FROM prs p JOIN units u ON u.id=p.unit_id JOIN unit_contents c ON c.hash=u.content_hash WHERE u.id=?1 AND u.kind='pr' AND u.file_id IS NULL AND u.ladder_level>=1",[unit],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?,r.get(7)?,r.get(8)?,r.get(9)?,r.get(10)?,r.get(11)?,r.get(12)?))).optional()?;
    let Some((
        id,
        number,
        title,
        merge,
        base,
        head,
        strategy,
        count,
        time,
        hash,
        breadcrumb,
        header,
        text,
    )) = row
    else {
        return super::symbol_history::resolve(
            tx,
            root,
            config_root,
            current,
            unit,
            minimum,
            permit_protected,
            work,
        );
    };
    if !current.commits.prs {
        return Ok(None);
    }
    let check = || {
        work.check()
            .map_err(|_| crate::store::Error::L1InputsChanged)
    };
    let Some(members) = Store::pr_members(
        tx,
        id,
        unit,
        usize::try_from(count).unwrap_or(usize::MAX),
        &check,
    )?
    else {
        return Ok(None);
    };
    let Ok(oid) = gix::ObjectId::try_from(merge.as_slice()) else {
        return Ok(None);
    };
    let repo = gix::discover(root).map_err(|_| Error::Invalid("PR Git repository"))?;
    let maximum = current
        .limits
        .max_file_bytes()
        .map_err(|_| Error::Invalid("PR metadata cap"))?
        .0;
    let Ok(object_header) = repo.find_header(oid) else {
        return Ok(None);
    };
    if object_header.size() > maximum {
        return Ok(None);
    }
    let Ok(raw) = repo.find_commit(oid) else {
        return Ok(None);
    };
    let Ok(decoded) = raw.decode() else {
        return Ok(None);
    };
    let Ok(message) = std::str::from_utf8(decoded.message) else {
        return Ok(None);
    };
    let (subject, body) = message.split_once('\n').unwrap_or((message, ""));
    let patterns = pr_patterns::compile(&current.commits.pr_patterns)
        .map_err(|_| Error::Invalid("PR patterns"))?;
    let parents = raw.parent_ids().map(|id| id.detach()).collect::<Vec<_>>();
    let Some(marker) = pr_patterns::detect(
        &patterns,
        parents.len(),
        subject.trim_end_matches('\r'),
        body.trim_matches(['\n', '\r']),
    ) else {
        return Ok(None);
    };
    let Ok(merged) = raw.time() else {
        return Ok(None);
    };
    if marker.number != number
        || marker.title != title
        || marker.strategy != strategy
        || merged.seconds != time
        || parents.first().map(|id| id.as_slice()) != base.as_deref()
        || (if strategy == "squash" {
            oid.as_slice()
        } else {
            parents[1].as_slice()
        }) != head
    {
        return Ok(None);
    }
    if strategy == "squash" && (members.len() != 1 || members[0].sha != merge) {
        return Ok(None);
    }
    let Ok(date) = merged.format(gix::date::time::format::SHORT) else {
        return Ok(None);
    };
    let Some((_, tail)) = text.rsplit_once("\nsymbols:\n") else {
        return Ok(None);
    };
    let Some((order, _)) = tail.rsplit_once('\n') else {
        return Ok(None);
    };
    let symbols = order.lines().map(str::to_owned).collect::<Vec<_>>();
    let union = members
        .iter()
        .flat_map(|m| m.symbols.iter())
        .collect::<BTreeSet<_>>();
    let selected = symbols.iter().collect::<BTreeSet<_>>();
    if symbols.len() != union.len().min(100)
        || selected.len() != symbols.len()
        || !selected.is_subset(&union)
    {
        return Ok(None);
    }
    let canonical = pr::collate(&marker, &date, &members, &symbols);
    if canonical.hash.as_slice() != hash
        || canonical.breadcrumb != breadcrumb
        || canonical.header != header
        || canonical.text != text
    {
        return Ok(None);
    }
    let mut max_level = 0;
    let mut never_send = false;
    let mut fingerprint = blake3::Hasher::new_derive_key("dkc PR embedding policy v1");
    fingerprint.update(current.commits.prs.to_string().as_bytes());
    for member in members {
        check_work(work)?;
        let Some(source) = super::history_source::resolve(
            tx,
            root,
            config_root,
            current,
            member.unit,
            0,
            true,
            work,
        )?
        else {
            return Ok(None);
        };
        max_level = max_level.max(source.policy.max_level);
        never_send |= source.policy.never_send;
        fingerprint.update(&member.sha);
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
            kind: UnitKind::PullRequest,
        },
        context: context::history_input(
            &breadcrumb,
            &header,
            "pr",
            &text,
            context::Options::from(&current.embedder),
        ),
        policy: Policy {
            fingerprint: *fingerprint.finalize().as_bytes(),
            max_level,
            never_send,
        },
    }))
}
