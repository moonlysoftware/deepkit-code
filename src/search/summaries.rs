//! Verified summary presentation and response evidence; v1 §9/§10.1.
use crate::{config::Config, queue, sync::l1::Work};
use rusqlite::Transaction;
use std::path::Path;

/// Private response evidence. Only text is presented; identity and provenance
/// participate in final response equality and continuation validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Observation {
    pub(crate) text: String,
    unit: i64,
    content: [u8; 32],
    proof: [u8; 32],
}

impl Observation {
    /// Binds continuation bytes to the exact summary generation and content.
    pub(crate) fn bind(&self, hash: &mut blake3::Hasher) {
        hash.update(b"dkc presented summary v1");
        hash.update(&self.unit.to_le_bytes());
        hash.update(&self.content);
        hash.update(&self.proof);
        hash.update(&(self.text.len() as u64).to_le_bytes());
        hash.update(self.text.as_bytes());
    }
}

/// Observes a current owned summary in the caller's read snapshot. Returning
/// None preserves raw-source serving when no unambiguous verified output exists.
pub(crate) fn observe_in(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    target: queue::summary_source::Attachment,
    work: Work<'_>,
) -> Result<Option<Observation>, queue::Error> {
    Ok(
        queue::summary_source::attached(tx, root, config_root, config, target, work)?.map(
            |input| Observation {
                text: input.context.text,
                unit: input.occurrence.id,
                content: input.occurrence.content_hash,
                proof: input.policy.fingerprint,
            },
        ),
    )
}

/// Attaches an exact symbol owner's summary before response comparison and caps.
pub(crate) fn populate_symbol(
    context: &super::service::Context,
    symbol: &mut super::navigation::SymbolResult,
    work: Work<'_>,
) -> Result<(), super::service::Error> {
    let observed = context.store.checked_read_snapshot(
        &|| {
            work.check()
                .map_err(|_| crate::store::Error::L1InputsChanged)
        },
        |tx| {
            observe_in(
                tx,
                &context.root,
                &context.config_root,
                &context.config,
                queue::summary_source::Attachment::Symbol(symbol.selected.id),
                work,
            )
        },
    )??;
    symbol.summary = observed.as_ref().map(|summary| summary.text.clone());
    symbol.summary_evidence = observed;
    Ok(())
}

/// Verifies every admitted directory label in one read snapshot. Final map
/// comparison calls this again, including directories that currently lack a summary.
pub(crate) fn populate_modules(
    context: &super::service::Context,
    modules: &mut [super::map::Module],
    work: Work<'_>,
) -> Result<(), super::service::Error> {
    context.store.checked_read_snapshot(
        &|| {
            work.check()
                .map_err(|_| crate::store::Error::L1InputsChanged)
        },
        |tx| {
            for module in modules {
                module.summary = observe_in(
                    tx,
                    &context.root,
                    &context.config_root,
                    &context.config,
                    queue::summary_source::Attachment::Parent(module.unit.id),
                    work,
                )?;
            }
            Ok::<_, queue::Error>(())
        },
    )??;
    Ok(())
}

/// Observes optional attachments for a bounded returned-hit set, preserving
/// positions and absent-summary evidence for the independent final observation.
pub(crate) fn observe_many(
    context: &super::service::Context,
    occurrences: &[&crate::store::StoredUnit],
    work: Work<'_>,
) -> Result<Vec<Option<Observation>>, super::service::Error> {
    if occurrences.is_empty() {
        return Ok(Vec::new());
    }
    Ok(context.store.checked_read_snapshot(
        &|| {
            work.check()
                .map_err(|_| crate::store::Error::L1InputsChanged)
        },
        |tx| {
            occurrences
                .iter()
                .map(|unit| {
                    if !crate::store::occurrence_matches_in(tx, unit)? {
                        return Ok(None);
                    }
                    observe_in(
                        tx,
                        &context.root,
                        &context.config_root,
                        &context.config,
                        queue::summary_source::Attachment::Parent(unit.id),
                        work,
                    )
                })
                .collect::<Result<Vec<_>, queue::Error>>()
        },
    )??)
}
