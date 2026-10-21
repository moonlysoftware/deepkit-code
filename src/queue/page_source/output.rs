//! Complete page generation authority for text embeddings and reranking; v1 §7/§8.
use super::*;
use crate::{
    queue::embedding::{Occurrence, Policy},
    store::{UnitKind, enrichment::publication::page},
    treat::context,
};

/// Re-proves the PDF, page generation, entire output and exact provenance before
/// exposing a generated section. A surviving child alone is never sufficient.
pub(crate) fn resolve(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    unit: i64,
    work: Work<'_>,
) -> Result<Option<source::FileInput>, Error> {
    let owner:Option<(i64,u32,i64)>=tx.query_row(
        "SELECT u.file_id,m.page,m.parent_unit FROM units u JOIN page_publications m ON m.parent_unit=u.parent_unit AND m.file_id=u.file_id
        WHERE u.id=?1 AND u.kind='doc_section' AND u.ladder_level=3",[unit],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()?;
    let Some((file, ordinal, parent)) = owner else {
        return Ok(None);
    };
    let Some(input) = super::prepare(tx, root, config_root, config, file, ordinal, work)? else {
        return Ok(None);
    };
    if page::matching(
        tx,
        file,
        &input.source.content_hash,
        ordinal,
        &input.desired,
    )? != Some(parent)
    {
        return Ok(None);
    }
    let provenance:bool=tx.query_row("SELECT count(*)=2 AND sum(source_kind='file' AND source_id=?2 AND role='content')=1 AND sum(source_kind='unit' AND source_id=?3 AND role='content')=1 FROM unit_sources WHERE unit_id=?1",params![unit,file,parent],|r|r.get(0))?;
    if !provenance {
        return Ok(None);
    }
    let (hash,header,scope,text):([u8;32],String,Option<String>,String)=tx.query_row(
        "SELECT u.content_hash,u.header,u.scope_path,c.text FROM units u JOIN unit_contents c ON c.hash=u.content_hash WHERE u.id=?1",[unit],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?)))?;
    let context = context::build_with_options(
        context::ContextUnit {
            scope_path: scope.as_deref(),
            header: &header,
            language: None,
            kind: "doc_section",
            text: &text,
            is_document: true,
        },
        context::ContextFile {
            repo_name: &crate::sync::l1::repo_name(root, config),
            path: &input.source.path,
        },
        &[],
        context::Options::from(&config.embedder),
    );
    let mut proof = blake3::Hasher::new_derive_key("dkc page outbound proof v1");
    proof.update(&input.desired);
    proof.update(&input.source.fingerprint);
    source::check_work(work)?;
    Ok(Some(source::FileInput {
        occurrence: Occurrence {
            id: unit,
            content_hash: hash,
            kind: UnitKind::DocSection,
        },
        context,
        policy: Policy {
            fingerprint: *proof.finalize().as_bytes(),
            max_level: 3,
            never_send: false,
        },
    }))
}
