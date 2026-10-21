//! Stateless, source-generation-fenced open pages; `v1.md` §3.5 and §10.1.
use super::{Error, OpenResult, Result, Store, open_checked, read_source, truncate_utf8};
use crate::treat::context::{self, ContextFile};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Cursor {
    target: String,
    context: u64,
    offset: usize,
    digest: String,
}

fn invalid() -> Error {
    Error::InvalidQuery {
        reason: "invalid open continuation or byte cap too small for the next UTF-8 character",
    }
}

/// Returns a bounded source-verified page; continuation IDs retain the original range and
/// context and are reusable across processes only while source and L1 agree.
pub(crate) fn open_source(
    store: &Store,
    roots: (&Path, &Path),
    settings: &crate::config::Config,
    target: &str,
    context: u64,
    caps: (usize, u64),
    work: crate::sync::l1::Work<'_>,
) -> Result<OpenResult> {
    let (root, config_root) = roots;
    let (max_bytes, max_file_bytes) = caps;
    let check = || {
        work.check()
            .map_err(|_| crate::store::Error::L1InputsChanged)
    };
    check()?;
    let cursor = decode_cursor(target, context)?;
    let target = cursor
        .as_ref()
        .map_or(target, |cursor| cursor.target.as_str());
    let context = cursor.as_ref().map_or(context, |cursor| cursor.context);
    let initial = open_checked(store, target, max_bytes, &check)?;
    if matches!(
        initial.kind.as_str(),
        "commit" | "pr" | "history" | "module"
    ) && initial.path.is_none()
    {
        if context != 0 {
            return Err(Error::InvalidQuery {
                reason: "context requires an indexed code source",
            });
        }
        let mut result = open_checked(store, target, usize::MAX, &check)?;
        let (input, breadcrumb, sources, summary) = store
            .checked_read_snapshot(&check, |tx| {
                if !crate::store::occurrence_matches_in(tx, &result.occurrence)? {
                    return Ok::<_, crate::queue::Error>(None);
                }
                let Some(input) = crate::queue::source::resolve_freshness(
                    tx,
                    root,
                    config_root,
                    settings,
                    result.occurrence.id,
                    work,
                )?
                else {
                    return Ok(None);
                };
                let breadcrumb: String = tx.query_row(
                    "SELECT scope_path FROM units WHERE id=?1",
                    [result.occurrence.id],
                    |r| r.get(0),
                )?;
                let mut sources = Vec::new();
                if matches!(result.kind.as_str(), "pr" | "history") {
                    let rows = tx.prepare("SELECT c.unit_id,lower(hex(c.sha)) FROM unit_sources s JOIN commits c ON s.source_kind='commit' AND c.id=s.source_id WHERE s.unit_id=?1 ORDER BY c.author_time,c.sha LIMIT 501")?
                        .query_map([result.occurrence.id], |r| Ok((r.get::<_,i64>(0)?, r.get::<_,String>(1)?)))?
                        .collect::<std::result::Result<Vec<_>,_>>()?;
                    for (id,sha) in rows {
                        check()?;
                        let source = crate::store::unit_by_id_in(tx,id)?.ok_or(crate::queue::Error::Invalid("PR member occurrence"))?;
                        sources.push(super::HistorySource { id: super::public_id(&source), kind: "commit".into(), sha, label: None });
                    }
                }
                if result.kind == "module" {
                    let rows=tx.prepare("SELECT s.source_kind,s.source_id,coalesce(f.path,'') FROM unit_sources s LEFT JOIN files f ON s.source_kind='file' AND f.id=s.source_id WHERE s.unit_id=?1 ORDER BY s.source_kind,f.path,s.source_id")?.query_map([result.occurrence.id],|r|Ok((r.get::<_,String>(0)?,r.get::<_,i64>(1)?,r.get::<_,String>(2)?)))?.collect::<std::result::Result<Vec<_>,_>>()?;
                    for (kind,id,path) in rows {
                        check()?;
                        if kind=="file" {
                            sources.push(super::HistorySource{id:path.clone(),kind,sha:String::new(),label:Some(path)});
                        } else if kind=="unit" {
                            let source=crate::store::unit_by_id_in(tx,id)?.ok_or(crate::queue::Error::Invalid("module README occurrence"))?;
                            sources.push(super::HistorySource{id:super::public_id(&source),kind:source.kind.clone(),sha:String::new(),label:Some(source.header.clone())});
                        }
                    }
                }
                let summary = crate::search::summaries::observe_in(tx, root, config_root, settings,
                    crate::queue::summary_source::Attachment::Parent(result.occurrence.id), work)?;
                Ok(Some((input, breadcrumb, sources, summary)))
            })?
            .map_err(|_| Error::OpenSourceUnavailable)?
            .ok_or(Error::OpenSourceUnavailable)?;
        let mut identity = blake3::Hasher::new_derive_key("dkc history open authority v1");
        for bytes in [
            input.occurrence.content_hash.as_slice(),
            &input.policy.fingerprint,
            breadcrumb.as_bytes(),
            input.context.header.as_bytes(),
            input.context.text.as_bytes(),
            result.occurrence.hash.as_slice(),
            target.as_bytes(),
        ] {
            identity.update(&(bytes.len() as u64).to_le_bytes());
            identity.update(bytes);
        }
        if let Some(summary) = &summary {
            summary.bind(&mut identity);
            result.text.clone_from(&summary.text);
        }
        result.summary_evidence = summary;
        let identity = identity.finalize();
        result.history_source = Some(*identity.as_bytes());
        result.breadcrumb = breadcrumb;
        result.sources = sources;
        result.lines = [None, None];
        result.indexed_lines = Some([None, None]);
        check()?;
        return page(
            result,
            cursor.as_ref(),
            target,
            context,
            max_bytes,
            identity.to_hex().to_string(),
        );
    }
    let before = store
        .open_source_checked(&initial.id, &check)?
        .ok_or(Error::OpenSourceUnavailable)?;
    if !current_l1(root, settings, initial.path.as_deref(), &before) {
        return Err(Error::OpenSourceUnavailable);
    }
    check()?;
    // The full expansion is already bounded by max_file_bytes. Page slicing must
    // happen after expansion, or a mid-line cursor would expand the symbol again.
    let mut result = read_source(
        store,
        root,
        target,
        context,
        (usize::MAX, max_file_bytes),
        &check,
    )?;
    if store.open_source_checked(&result.id, &check)?.as_ref() != Some(&before) {
        return Err(Error::OpenSourceUnavailable);
    }
    result.breadcrumb = context::breadcrumb(
        ContextFile {
            repo_name: &crate::sync::l1::repo_name(root, settings),
            path: result.path.as_deref().ok_or(Error::OpenSourceUnavailable)?,
        },
        before
            .scope_path
            .as_deref()
            .filter(|_| result.kind == "doc_section"),
        &[],
        true,
    );
    let summary = store
        .checked_read_snapshot(&check, |tx| {
            if !crate::store::occurrence_matches_in(tx, &result.occurrence)? {
                return Ok::<_, crate::queue::Error>(None);
            }
            let target = result.selected_symbol.map_or(
                crate::queue::summary_source::Attachment::Parent(result.occurrence.id),
                crate::queue::summary_source::Attachment::Symbol,
            );
            crate::search::summaries::observe_in(tx, root, config_root, settings, target, work)
        })?
        .map_err(|_| Error::OpenSourceUnavailable)?;
    result.summary = summary.as_ref().map(|summary| summary.text.clone());
    let mut hash = blake3::Hasher::new();
    for bytes in [
        before.hash.as_slice(),
        &before.fingerprint,
        &before.occurrence,
        result.text.as_bytes(),
        result.breadcrumb.as_bytes(),
        target.as_bytes(),
        &context.to_le_bytes(),
    ] {
        hash.update(&(bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    if let Some(summary) = &summary {
        summary.bind(&mut hash);
    }
    result.summary_evidence = summary;
    let digest = hash.finalize().to_hex().to_string();
    result.source = Some(before);
    check()?;
    page(result, cursor.as_ref(), target, context, max_bytes, digest)
}

fn page(
    mut result: OpenResult,
    cursor: Option<&Cursor>,
    target: &str,
    context: u64,
    max_bytes: usize,
    digest: String,
) -> Result<OpenResult> {
    let offset = cursor.map_or(0, |cursor| cursor.offset);
    if cursor.is_some_and(|cursor| cursor.digest != digest) {
        return Err(Error::OpenSourceUnavailable);
    }
    let remainder = result.text.get(offset..).ok_or_else(invalid)?;
    if cursor.is_some() && remainder.is_empty() {
        return Err(invalid());
    }
    let page = truncate_utf8(remainder, max_bytes);
    if page.is_empty() && !remainder.is_empty() {
        return Err(invalid());
    }
    let end = offset + page.len();
    let skipped_lines = result.text[..offset]
        .bytes()
        .filter(|byte| *byte == b'\n')
        .count();
    let page_lines = page.split_inclusive('\n').count();
    if result.indexed_lines.is_some() {
        result.extracted_lines =
            (page_lines > 0).then(|| [1 + skipped_lines, skipped_lines + page_lines]);
    } else if let Some(start) = result.lines[0] {
        let first = start + skipped_lines as i64;
        result.lines = [Some(first), Some(first + page_lines as i64 - 1)];
    }
    result.truncated = end < result.text.len();
    result.byte_range = Some([offset, end]);
    if result.truncated {
        let bytes = serde_json::to_vec(&Cursor {
            target: target.to_owned(),
            context,
            offset: end,
            digest,
        })
        .map_err(|_| invalid())?;
        let encoded = URL_SAFE_NO_PAD.encode(bytes);
        if encoded.len() > 16_384 {
            return Err(invalid());
        }
        result.continuation_id = Some(format!("o1_{encoded}"));
    }
    result.text = page.to_owned();
    Ok(result)
}

fn decode_cursor(target: &str, context: u64) -> Result<Option<Cursor>> {
    if let Some(encoded) = target.strip_prefix("o1_") {
        if encoded.len() > 16_384 || context != 0 {
            return Err(invalid());
        }
        let bytes = URL_SAFE_NO_PAD.decode(encoded).map_err(|_| invalid())?;
        let cursor: Cursor = serde_json::from_slice(&bytes).map_err(|_| invalid())?;
        if cursor.target.starts_with("o1_") || cursor.offset == 0 {
            return Err(invalid());
        }
        Ok(Some(cursor))
    } else {
        Ok(None)
    }
}

/// Resolves only the indexed path to attempt repair. Cursor digests still must
/// validate through open_source after publication; decoding grants no authority.
pub(crate) fn repair_path(
    store: &Store,
    root: &Path,
    target: &str,
    context: u64,
    settings: &crate::config::Config,
    work: crate::sync::l1::Work<'_>,
) -> Result<Option<String>> {
    let cursor = decode_cursor(target, context)?;
    let target = cursor
        .as_ref()
        .map_or(target, |cursor| cursor.target.as_str());
    let check = || {
        work.check()
            .map_err(|_| crate::store::Error::L1InputsChanged)
    };
    let context = cursor.as_ref().map_or(context, |cursor| cursor.context);
    let max_file_bytes = settings
        .limits
        .max_file_bytes()
        .map_err(|_| Error::OpenSourceUnavailable)?
        .0;
    // A digest mismatch alone must not cause publication or revive an old page.
    match read_source(store, root, target, context, (0, max_file_bytes), &check) {
        Ok(result) => {
            let before = store
                .open_source_checked(&result.id, &check)?
                .ok_or(Error::OpenSourceUnavailable)?;
            let current = current_l1(root, settings, result.path.as_deref(), &before);
            check()?;
            Ok(if current { None } else { result.path })
        }
        Err(Error::OpenSourceUnavailable) => Ok(Some(
            open_checked(store, target, 0, &check)?
                .path
                .ok_or(Error::OpenSourceUnavailable)?,
        )),
        Err(error) => Err(error),
    }
}

/// Checks canonical L1 identity without applying outbound restrictions to local reads.
pub(super) fn current_l1(
    root: &Path,
    settings: &crate::config::Config,
    path: Option<&str>,
    source: &crate::store::OpenSource,
) -> bool {
    let (Ok(router), repo_name) = (
        crate::treat::Router::new(&settings.treatments, settings.llm.enabled),
        crate::sync::l1::repo_name(root, settings),
    ) else {
        return false;
    };
    current_l1_with(settings, path, source, &repo_name, &router)
}

/// Reuses one request's compiled treatment router and repository identity.
pub(super) fn current_l1_with(
    settings: &crate::config::Config,
    path: Option<&str>,
    source: &crate::store::OpenSource,
    repo_name: &str,
    router: &crate::treat::Router,
) -> bool {
    let (Some(path), Some(kind), Ok(hash)) = (
        path,
        crate::store::FileKind::from_db(&source.kind),
        <&[u8; 32]>::try_from(source.hash.as_slice()),
    ) else {
        return false;
    };
    let treatment = router.route(path, kind);
    treatment.max_level >= 1
        && crate::sync::l1::file_fingerprint(
            settings,
            repo_name,
            crate::sync::l1::FingerprintFile {
                path,
                content_hash: hash,
                language: source.language.as_deref(),
                kind,
                capped: false,
            },
            &treatment,
        )
        .as_slice()
            == source.fingerprint
}
