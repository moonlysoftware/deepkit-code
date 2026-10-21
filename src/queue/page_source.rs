//! Reproducible scanned-page source and prompt authority; v1 §6.2/§6.8/§8.
use super::{Error, Kind, enrichment_inputs, source, visual_source};
use crate::{config::Config, models::openai_compat::chat, sync::l1::Work, treat::pdf::scan};
use rusqlite::{OptionalExtension, Transaction, params};
use std::{collections::BTreeMap, path::Path};

// Finite reservation and response ceiling. Incomplete output must not be published.
const MAX_OUTPUT_TOKENS: u32 = 8192;

mod output;
pub(super) use output::resolve as resolve_output;

mod selection;
pub(crate) use selection::select;

/// Binds a one-based page inside the enqueue transaction. Recovery never guesses
/// an ordinal, and repeated scheduling never changes metadata on an existing lease.
pub(crate) fn bind(tx: &Transaction<'_>, id: i64, page: u32) -> Result<(), Error> {
    if page == 0 {
        return Err(Error::Invalid("page ordinal"));
    }
    let row: Option<(Option<u32>, String)> = tx.query_row(
        "SELECT page,state FROM jobs WHERE id=?1 AND kind='enrich:vlm_page' AND target_kind='file'",
        [id], |row| Ok((row.get(0)?, row.get(1)?)),
    ).optional()?;
    match row {
        Some((Some(existing), _)) if existing == page => Ok(()),
        Some((None, state)) if state == "queued" => {
            let changed = tx.execute(
                "UPDATE jobs SET page=?2 WHERE id=?1 AND page IS NULL AND state='queued'",
                params![id, page],
            )?;
            if changed != 1 {
                return Err(Error::IdentityConflict);
            }
            Ok(())
        }
        _ => Err(Error::IdentityConflict),
    }
}

/// Source bytes and title deliberately have no Debug representation.
pub(crate) struct Prepared {
    pub(crate) source: visual_source::Input,
    pub(crate) page: scan::Page,
    pub(crate) prompt: String,
    pub(crate) desired: [u8; 32],
    pub(crate) wire_fingerprint: [u8; 32],
    pub(crate) max_output_tokens: u32,
}

/// Re-proves policy and exact PDF bytes before deriving one page's wire identity.
pub(crate) fn prepare(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    id: i64,
    page: u32,
    work: Work<'_>,
) -> Result<Option<Prepared>, Error> {
    let Some(input) = visual_source::resolve(
        tx,
        root,
        config_root,
        config,
        id,
        visual_source::Purpose::ScannedPage,
        work,
    )?
    else {
        return Ok(None);
    };
    let image_bytes = config
        .limits
        .max_image_bytes()
        .map_err(|_| Error::Invalid("page image cap"))?
        .0;
    let document_bytes = config
        .limits
        .max_doc_bytes()
        .map_err(|_| Error::Invalid("page document cap"))?
        .0
        .min(
            config
                .limits
                .max_file_bytes()
                .map_err(|_| Error::Invalid("page file cap"))?
                .0,
        );
    let page = scan::prepare(
        &input.bytes,
        page,
        scan::Limits {
            document_bytes,
            image_bytes,
            pages: config.limits.max_pdf_pages,
        },
        work,
    )
    .map_err(|error| match error {
        scan::Error::Stopped => Error::WorkerStopped,
        other => Error::PagePreparation(other),
    })?;
    let template = crate::treat::prompts::Template::load(
        config_root,
        &config.enrich.prompts_dir,
        Kind::VlmPage,
    )
    .map_err(|_| Error::Invalid("page prompt override"))?;
    let prompt = template
        .render(&BTreeMap::new(), 256 * 1024)
        .map_err(|_| Error::Invalid("page prompt capacity or fields"))?;
    let wire = chat::image_fingerprint(
        &config.llm,
        image_bytes,
        &page.bytes,
        page.mime,
        &prompt,
        MAX_OUTPUT_TOKENS,
    )
    .map_err(|_| Error::Invalid("page chat preparation"))?;
    let endpoint = chat::endpoint_identity(&config.llm.url)
        .map_err(|_| Error::Invalid("page endpoint identity"))?;
    let chunk =
        serde_json::to_vec(&config.chunk).map_err(|_| Error::Invalid("page chunk identity"))?;
    // Extracted metadata shapes section breadcrumbs and belongs to the output
    // generation alongside pixels, even for PDFs with identical page images.
    let metadata = serde_json::to_vec(&(page.number, page.total_pages, &page.title))
        .map_err(|_| Error::Invalid("page metadata identity"))?;
    let mut proof = blake3::Hasher::new_derive_key("dkc scanned page enrichment input v1");
    proof.update(&input.id.to_le_bytes());
    proof.update(&input.content_hash);
    proof.update(&input.fingerprint);
    proof.update(&metadata);
    proof.update(&template.fingerprint);
    proof.update(&chunk);
    if let Some(policy) = page.rendering_policy {
        proof.update(policy.as_bytes());
    }
    let desired = enrichment_inputs::generation(proof.finalize().as_bytes(), &wire, &endpoint);
    source::check_work(work)?;
    Ok(Some(Prepared {
        source: input,
        page,
        prompt,
        desired,
        wire_fingerprint: wire,
        max_output_tokens: MAX_OUTPUT_TOKENS,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        queue::{Enqueue, Target, enqueue_in},
        store::{self, StoreLimits},
    };

    #[test]
    fn page_binding_survives_restart_and_rejects_conflicting_or_active_legacy_jobs() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("code.db");
        let store = store::open(&path, StoreLimits::default()).unwrap();
        let id = {
            let mut writer = store.writer().unwrap();
            let tx = writer.transaction().unwrap();
            let input = Enqueue {
                kind: Kind::VlmPage,
                target: Target::File,
                target_id: 1,
                target_hash: &[1; 32],
                desired_hash: &[2; 32],
                provider: "fixture",
                priority: 0.0,
                tokens: 8192,
            };
            let id = enqueue_in(&tx, &input, 1, None).unwrap();
            bind(&tx, id, 7).unwrap();
            assert!(matches!(bind(&tx, id, 8), Err(Error::IdentityConflict)));
            tx.execute(
                "UPDATE jobs SET state='leased',attempts=2 WHERE id=?1",
                [id],
            )
            .unwrap();
            bind(&tx, id, 7).unwrap();
            assert!(matches!(bind(&tx, id, 8), Err(Error::IdentityConflict)));
            let legacy = Enqueue {
                desired_hash: &[3; 32],
                ..input
            };
            let legacy_id = enqueue_in(&tx, &legacy, 1, None).unwrap();
            tx.execute("UPDATE jobs SET state='leased' WHERE id=?1", [legacy_id])
                .unwrap();
            assert!(matches!(
                bind(&tx, legacy_id, 7),
                Err(Error::IdentityConflict)
            ));
            tx.commit().unwrap();
            id
        };
        drop(store);
        let reopened = store::open(&path, StoreLimits::default()).unwrap();
        let row: (u32, String, i64) = reopened
            .writer()
            .unwrap()
            .query_row(
                "SELECT page,state,attempts FROM jobs WHERE id=?1",
                [id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(row, (7, "leased".into(), 2));
    }

    #[test]
    fn page_binding_requires_page_file_kind_and_rolls_back_with_enqueue() {
        let store = store::open(":memory:", StoreLimits::default()).unwrap();
        let mut writer = store.writer().unwrap();
        {
            let tx = writer.transaction().unwrap();
            for (kind, target) in [
                (Kind::DescribeImage, Target::File),
                (Kind::VlmPage, Target::Unit),
            ] {
                let input = Enqueue {
                    kind,
                    target,
                    target_id: 1,
                    target_hash: &[1; 32],
                    desired_hash: &[2; 32],
                    provider: "fixture",
                    priority: 0.0,
                    tokens: 1,
                };
                let id = enqueue_in(&tx, &input, 1, None).unwrap();
                assert!(matches!(bind(&tx, id, 1), Err(Error::IdentityConflict)));
                assert!(
                    tx.execute("UPDATE jobs SET page=1 WHERE id=?1", [id])
                        .is_err()
                );
            }
            let input = Enqueue {
                kind: Kind::VlmPage,
                target: Target::File,
                target_id: 1,
                target_hash: &[1; 32],
                desired_hash: &[2; 32],
                provider: "fixture",
                priority: 0.0,
                tokens: 1,
            };
            let id = enqueue_in(&tx, &input, 1, None).unwrap();
            assert!(matches!(bind(&tx, id, 0), Err(Error::Invalid(_))));
            bind(&tx, id, 1).unwrap();
        }
        assert_eq!(
            writer
                .query_row("SELECT count(*) FROM jobs", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            0
        );
    }
}
