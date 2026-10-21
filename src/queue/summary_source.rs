//! Current summary generation proof for embeddings and serving; v1 §6.8/§7.1/§10.1.
use super::{
    Error, Kind,
    embedding::{Occurrence, Policy},
    enrichment_inputs,
    source::{self, FileInput},
    visual_source,
};
use crate::{config::Config, store::UnitKind, sync::l1::Work, treat::context};
use rusqlite::{OptionalExtension, Transaction};
use std::path::Path;

/// Suppresses raw composite embedding only after its current summary generation
/// has an actual vector publication in the configured summary destination.
pub(super) fn replaces_parent(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    parent: i64,
) -> Result<bool, Error> {
    let summary: Option<i64> = tx
        .query_row(
            "SELECT summary_unit FROM prs WHERE unit_id=?1 AND summary_unit IS NOT NULL
         UNION ALL SELECT summary_unit FROM dirs WHERE module_unit=?1 AND summary_unit IS NOT NULL",
            [parent],
            |r| r.get(0),
        )
        .optional()?;
    let Some(summary) = summary else {
        return Ok(false);
    };
    // Missing publication proof cannot suppress a raw composite. Avoid expensive
    // role/source preparation until there is evidence worth checking; a positive
    // result still goes through every configured destination and freshness fence.
    let has_proof: bool = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM embedding_publications WHERE unit_id=?1)",
        [summary],
        |r| r.get(0),
    )?;
    if !has_proof {
        return Ok(false);
    }
    let Some((prepared, destination)) =
        super::destination::prepare_file(tx, root, config_root, config, summary)?
    else {
        return Ok(false);
    };
    let slug = crate::store::embedding_bindings::select(tx, &destination.binding(config))?;
    let active: bool = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM embedding_tables WHERE slug=?1 AND active=1)",
        [&slug],
        |r| r.get(0),
    )?;
    if !active {
        return Ok(false);
    };
    Ok(crate::store::vectors::publication_matches_in(
        tx,
        &slug,
        summary,
        &prepared.current().desired_hash,
    )?)
}

/// Exact owner requested by a read-only tool. Shared parent occurrences do not
/// identify an individual symbol; ambiguous parent attachments are omitted.
#[derive(Clone, Copy)]
pub(crate) enum Attachment {
    Symbol(i64),
    Parent(i64),
}

/// Returns a currently owned, source-verified summary without loading a provider.
/// Callers retain its content/policy proof and repeat this lookup at their final
/// response boundary. Old children and ambiguous owners never supply a label.
pub(crate) fn attached(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    target: Attachment,
    work: Work<'_>,
) -> Result<Option<FileInput>, Error> {
    source::check_work(work)?;
    let candidates = match target {
        Attachment::Symbol(id) => tx.prepare(
            "SELECT u.id FROM symbols s JOIN units u ON u.id=s.summary_unit
             WHERE s.id=?1 AND u.kind='summary' AND u.parent_unit=s.unit_id
             AND u.source_key='summary:symbol:'||s.id"
        )?.query_map([id], |r|r.get::<_, i64>(0))?.collect::<Result<Vec<_>, _>>()?,
        Attachment::Parent(id) => tx.prepare(
            "SELECT u.id FROM units u WHERE u.parent_unit=?1 AND u.kind='summary' AND (
               EXISTS(SELECT 1 FROM symbols s WHERE s.unit_id=?1 AND s.summary_unit=u.id AND u.source_key='summary:symbol:'||s.id)
               OR EXISTS(SELECT 1 FROM prs p WHERE p.unit_id=?1 AND p.summary_unit=u.id AND u.source_key='summary:pr:'||p.id)
               OR EXISTS(SELECT 1 FROM dirs d WHERE d.module_unit=?1 AND d.summary_unit=u.id AND u.source_key='summary:module:'||d.module_unit)
               OR EXISTS(SELECT 1 FROM units p JOIN files f ON f.id=p.file_id WHERE p.id=?1
                 AND p.kind='image' AND p.ladder_level=0 AND p.parent_unit IS NULL AND f.kind='image'
                 AND u.source_key='summary:image:'||f.id)) ORDER BY u.id LIMIT 2"
        )?.query_map([id], |r|r.get::<_, i64>(0))?.collect::<Result<Vec<_>, _>>()?,
    };
    let [unit] = candidates.as_slice() else {
        return Ok(None);
    };
    Ok(resolve(tx, root, config_root, config, *unit, work)?
        .filter(|input| !input.policy.never_send))
}

/// Rebuilds all prompt inputs and the exact wire identity before returning stored
/// output. Owner pointers or live provenance IDs alone do not prove freshness.
pub(super) fn resolve(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    unit: i64,
    work: Work<'_>,
) -> Result<Option<FileInput>, Error> {
    resolve_with_caches(tx, root, config_root, config, unit, work, None, None)
}

/// Resolves a group of module-owned summaries with transaction-local symbol
/// preparations. The cache never crosses a source-proof boundary and every
/// module keeps its separate parent and generation verification.
pub(super) fn resolve_with_symbol_cache(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    unit: i64,
    work: Work<'_>,
    symbols: &mut enrichment_inputs::SymbolInputs,
) -> Result<Option<FileInput>, Error> {
    resolve_with_caches(
        tx,
        root,
        config_root,
        config,
        unit,
        work,
        Some(symbols),
        None,
    )
}

/// Extends the symbol cache with exact same-pass module parent observations.
/// The parent map is an enrichment-level batch proof, never a persisted cache.
#[allow(
    clippy::too_many_arguments,
    reason = "the transaction-scoped source proof inputs remain explicit at this boundary"
)]
pub(super) fn resolve_with_role_caches(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    unit: i64,
    work: Work<'_>,
    symbols: &mut enrichment_inputs::SymbolInputs,
    parents: &std::collections::BTreeMap<i64, FileInput>,
) -> Result<Option<FileInput>, Error> {
    resolve_with_caches(
        tx,
        root,
        config_root,
        config,
        unit,
        work,
        Some(symbols),
        Some(parents),
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "the source verification and optional same-pass proofs are explicit"
)]
fn resolve_with_caches(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    unit: i64,
    work: Work<'_>,
    symbols: Option<&mut enrichment_inputs::SymbolInputs>,
    parents: Option<&std::collections::BTreeMap<i64, FileInput>>,
) -> Result<Option<FileInput>, Error> {
    source::check_work(work)?;
    let owner: Option<(String,i64)> = tx.query_row(
        "SELECT 'symbol',s.id FROM symbols s JOIN units u ON u.id=s.summary_unit
         WHERE u.id=?1 AND u.kind='summary' AND u.parent_unit=s.unit_id AND u.source_key='summary:symbol:'||s.id
         UNION ALL
         SELECT 'pr',p.id FROM prs p JOIN units u ON u.id=p.summary_unit
         WHERE u.id=?1 AND u.kind='summary' AND u.parent_unit=p.unit_id AND u.source_key='summary:pr:'||p.id
         UNION ALL
         SELECT 'module',d.module_unit FROM dirs d JOIN units u ON u.id=d.summary_unit
         WHERE u.id=?1 AND u.kind='summary' AND u.parent_unit=d.module_unit AND u.source_key='summary:module:'||d.module_unit
         UNION ALL
         SELECT 'image',f.id FROM units u JOIN units p ON p.id=u.parent_unit JOIN files f ON f.id=p.file_id
         WHERE u.id=?1 AND u.kind='summary' AND p.kind='image' AND p.ladder_level=0 AND p.parent_unit IS NULL
         AND f.kind='image' AND u.source_key='summary:image:'||f.id",
        [unit], |r| Ok((r.get(0)?,r.get(1)?)),
    ).optional()?;
    let Some((kind, id)) = owner else {
        return Ok(None);
    };
    if kind == "image" {
        return image_summary(tx, root, config_root, config, unit, id, work);
    }
    if kind == "module" {
        let prepared = match (symbols, parents) {
            (Some(symbols), Some(parents)) => {
                let Some(parent) = parents.get(&id).cloned() else {
                    return Ok(None);
                };
                enrichment_inputs::module_with_parent_with_symbols_and_parent(
                    tx,
                    root,
                    config_root,
                    config,
                    id,
                    work,
                    symbols,
                    Some(parent),
                )?
            }
            (Some(symbols), None) => enrichment_inputs::module_with_parent_with_symbols(
                tx,
                root,
                config_root,
                config,
                id,
                work,
                symbols,
            )?,
            (None, _) => {
                enrichment_inputs::module_with_parent(tx, root, config_root, config, id, work)?
            }
        };
        let Some((input, parent)) = prepared else {
            return Ok(None);
        };
        let desired = enrichment_inputs::desired(config, Kind::SummarizeModule, &input)?;
        if input
            .owner
            .matching_summary(tx, input.parent_unit, &input.parent_hash, &desired)?
            != Some(unit)
        {
            return Ok(None);
        }
        return finish(tx, config, unit, &desired, parent, work);
    }
    let (kind, input) = match kind.as_str() {
        "symbol" => (
            Kind::DescribeSymbol,
            enrichment_inputs::symbol_with_parent(tx, root, config_root, config, id, work)?,
        ),
        "pr" => (
            Kind::SummarizePr,
            enrichment_inputs::pr_with_parent(tx, root, config_root, config, id, work)?,
        ),
        _ => return Ok(None),
    };
    source::check_work(work)?;
    let Some((input, parent)) = input else {
        return Ok(None);
    };
    let desired = enrichment_inputs::desired(config, kind, &input)?;
    if input
        .owner
        .matching_summary(tx, input.parent_unit, &input.parent_hash, &desired)?
        != Some(unit)
    {
        return Ok(None);
    }
    finish(tx, config, unit, &desired, parent, work)
}

/// An image's current verified bytes and captions authorize its description;
/// the L0 parent supplies a path, never raw image data for a text embedder.
fn image_summary(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    unit: i64,
    file: i64,
    work: Work<'_>,
) -> Result<Option<FileInput>, Error> {
    use crate::store::enrichment::publication::{Owner, image_parent};
    let Some(input) = visual_source::image(tx, root, config_root, config, file, work)? else {
        return Ok(None);
    };
    let Some(parent) = image_parent(tx, file, &input.source.content_hash)? else {
        return Ok(None);
    };
    if Owner::Image(file).matching_summary(
        tx,
        parent,
        &input.source.content_hash,
        &input.desired,
    )? != Some(unit)
    {
        return Ok(None);
    }
    let context = context::build_with_options(
        context::ContextUnit {
            scope_path: None,
            header: &input.source.path,
            language: None,
            kind: "image",
            text: "",
            is_document: true,
        },
        context::ContextFile {
            repo_name: &crate::sync::l1::repo_name(root, config),
            path: &input.source.path,
        },
        &[],
        context::Options::from(&config.embedder),
    );
    finish(
        tx,
        config,
        unit,
        &input.desired,
        FileInput {
            occurrence: Occurrence {
                id: parent,
                content_hash: input.source.content_hash,
                kind: UnitKind::Image,
            },
            context,
            // Image preparation already proved an unprotected L3-capable source.
            policy: Policy {
                fingerprint: input.source.fingerprint,
                max_level: 3,
                never_send: false,
            },
        },
        work,
    )
}

fn finish(
    tx: &Transaction<'_>,
    config: &Config,
    unit: i64,
    desired: &[u8; 32],
    parent: FileInput,
    work: Work<'_>,
) -> Result<Option<FileInput>, Error> {
    let (hash,text):(Vec<u8>,String)=tx.query_row(
        "SELECT u.content_hash,c.text FROM units u JOIN unit_contents c ON c.hash=u.content_hash WHERE u.id=?1",
        [unit],|r|Ok((r.get(0)?,r.get(1)?)),
    )?;
    let hash = hash
        .try_into()
        .map_err(|_| Error::Invalid("summary content hash"))?;
    let context = context::history_input(
        &parent.context.breadcrumb,
        &parent.context.header,
        "summary",
        &text,
        context::Options::from(&config.embedder),
    );
    let mut proof = blake3::Hasher::new_derive_key("dkc summary outbound proof v1");
    proof.update(desired);
    proof.update(&parent.policy.fingerprint);
    source::check_work(work)?;
    Ok(Some(FileInput {
        occurrence: Occurrence {
            id: unit,
            content_hash: hash,
            kind: UnitKind::Summary,
        },
        context,
        policy: Policy {
            fingerprint: *proof.finalize().as_bytes(),
            max_level: parent.policy.max_level,
            never_send: parent.policy.never_send,
        },
    }))
}
