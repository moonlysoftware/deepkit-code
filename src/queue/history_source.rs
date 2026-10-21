//! Immutable commit embedding authority and inherited source policy; v1.md §7.1/§8.2.
use super::{
    Error,
    embedding::{Occurrence, Policy},
    source::{FileInput, check_work},
};
use crate::{
    config::{self, Config},
    store::UnitKind,
    sync::{self, l1::Work},
    treat::{Router, context},
};
use rusqlite::{OptionalExtension, Transaction};
use std::path::Path;

struct Row {
    id: i64,
    sha: Vec<u8>,
    author: String,
    time: i64,
    subject: String,
    body: String,
    parents: i64,
    files: String,
    symbols: String,
    hash: Vec<u8>,
    breadcrumb: String,
    header: String,
    text: String,
}

/// Verifies immutable Git identity, complete owned provenance, canonical text and
/// current source privacy before granting dispatch authority for a history unit.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the shared source resolver's explicit authority inputs"
)]
pub(super) fn resolve(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    current: &Config,
    unit: i64,
    minimum_level: i64,
    permit_protected: bool,
    work: Work<'_>,
) -> Result<Option<FileInput>, Error> {
    check_work(work)?;
    if !current.commits.enabled {
        return Ok(None);
    }
    let row=tx.query_row("SELECT c.id,c.sha,c.author,c.author_time,c.subject,c.body,c.parents,c.files_touched,c.symbols_touched,u.content_hash,u.scope_path,u.header,p.text FROM commits c JOIN units u ON u.id=c.unit_id JOIN unit_contents p ON p.hash=u.content_hash WHERE u.id=?1 AND u.kind='commit' AND u.file_id IS NULL AND u.ladder_level>=1",[unit],|r|Ok(Row {id:r.get(0)?,sha:r.get(1)?,author:r.get(2)?,time:r.get(3)?,subject:r.get(4)?,body:r.get(5)?,parents:r.get(6)?,files:r.get(7)?,symbols:r.get(8)?,hash:r.get(9)?,breadcrumb:r.get(10)?,header:r.get(11)?,text:r.get(12)?})).optional()?;
    let Some(row) = row else {
        return super::pr_source::resolve(
            tx,
            root,
            config_root,
            current,
            unit,
            minimum_level,
            permit_protected,
            work,
        );
    };
    let owned:bool=tx.query_row("SELECT count(*)=1 AND coalesce(sum(source_kind='commit' AND source_id=?2 AND role='content'),0)=1 FROM unit_sources WHERE unit_id=?1",rusqlite::params![unit,row.id],|r|r.get(0))?;
    if !owned {
        return Ok(None);
    }
    let Ok(oid) = gix::ObjectId::try_from(row.sha.as_slice()) else {
        return Ok(None);
    };
    let repo = gix::discover(root).map_err(|_| Error::Invalid("history Git repository"))?;
    let maximum = current
        .limits
        .max_file_bytes()
        .map_err(|_| Error::Invalid("history source cap"))?
        .0;
    let Ok(header) = repo.find_header(oid) else {
        return Ok(None);
    };
    if header.size() > maximum {
        return Ok(None);
    }
    let Ok(raw) = repo.find_commit(oid) else {
        return Ok(None);
    };
    let Ok(decoded) = raw.decode() else {
        return Ok(None);
    };
    let Ok(author) = decoded.author() else {
        return Ok(None);
    };
    let Ok(time) = author.time() else {
        return Ok(None);
    };
    let Ok(date) = time.format(gix::date::time::format::SHORT) else {
        return Ok(None);
    };
    let Ok(message) = std::str::from_utf8(decoded.message) else {
        return Ok(None);
    };
    let (subject, body) = message.split_once('\n').unwrap_or((message, ""));
    if row.subject != subject.trim_end_matches('\r')
        || row.body != body.trim_matches(['\n', '\r'])
        || row.author.as_bytes() != AsRef::<[u8]>::as_ref(author.name)
        || row.time != time.seconds
        || row.parents != raw.parent_ids().count() as i64
    {
        return Ok(None);
    }
    let files: Vec<String> =
        serde_json::from_str(&row.files).map_err(|_| Error::Invalid("history file provenance"))?;
    let symbols: Vec<String> = serde_json::from_str(&row.symbols)
        .map_err(|_| Error::Invalid("history symbol provenance"))?;
    let text = format!(
        "{}\nfiles:\n{}\nsymbols:\n{}",
        row.body,
        files.join("\n"),
        symbols.join("\n")
    );
    let breadcrumb = format!("commit › {date} › {}", row.author);
    let hash = context::unit_hash("commit", None, &text);
    if row.text != text
        || row.header != row.subject
        || row.breadcrumb != breadcrumb
        || row.hash.as_slice() != hash
    {
        return Ok(None);
    }
    let router = Router::new(&current.treatments, current.llm.enabled)
        .map_err(|_| Error::Invalid("history source policy"))?;
    // A verified metadata-only commit has no file treatment ceiling. Allow its
    // repository-level PR job; commits themselves still have no L3 job kind.
    let mut max_level = if files.is_empty() { 3 } else { 0 };
    let mut never_send = false;
    for path in files {
        check_work(work)?;
        let (kind, _) = sync::classify_path(&path);
        let policy = router.route(&path, kind);
        max_level = max_level.max(policy.max_level);
        never_send |= policy.never_send;
    }
    if max_level < minimum_level || (never_send && !permit_protected) {
        return Ok(None);
    }
    if config::load(config_root, None)
        .map_err(|_| Error::Invalid("history configuration reload"))?
        .config
        != *current
    {
        return Ok(None);
    }
    let mut embedding = serde_json::to_value(&current.embedder)
        .map_err(|_| Error::Invalid("history embedding policy"))?;
    embedding["docs"] = serde_json::Value::Null;
    // Scheduling policy does not change a retained history's embedding input.
    embedding["reembed_on_role_change"] = serde_json::Value::Bool(false);
    let mut policy = blake3::Hasher::new_derive_key("dkc history embedding policy v1");
    policy.update(&sync::l1::settings_fingerprint(current));
    policy.update(embedding.to_string().as_bytes());
    check_work(work)?;
    Ok(Some(FileInput {
        occurrence: Occurrence {
            id: unit,
            content_hash: hash,
            kind: UnitKind::Commit,
        },
        context: context::history_input(
            &breadcrumb,
            &row.subject,
            "commit",
            &text,
            context::Options::from(&current.embedder),
        ),
        policy: Policy {
            fingerprint: *policy.finalize().as_bytes(),
            max_level,
            never_send,
        },
    }))
}
