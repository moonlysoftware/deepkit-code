//! Owned text/image output preparation and publication; v1 §5, §6.8 and §8.2.
use crate::{
    queue::{Error, capacity, enrichment_inputs, page_source, visual_source},
    store::enrichment::publication::{self, ImagePublication, Owner, Publication, Source},
};
use rusqlite::Transaction;

/// Different L0/L1 owners share the same lease, accounting and provider machinery.
pub(super) enum Input {
    Text(enrichment_inputs::Text),
    Image(visual_source::Image),
    Page(page_source::Prepared),
}
impl Input {
    /// Returns the source identity used by the durable Unit/Symbol/PR/File lease.
    pub(super) fn target_hash(&self) -> &[u8; 32] {
        match self {
            Self::Text(input) => &input.parent_hash,
            Self::Image(input) => &input.source.content_hash,
            Self::Page(input) => &input.source.content_hash,
        }
    }
    /// Compares complete prepared source evidence at dispatch and completion.
    pub(super) fn fingerprint(&self) -> &[u8; 32] {
        match self {
            Self::Text(input) => &input.fingerprint,
            Self::Image(input) => &input.desired,
            Self::Page(input) => &input.desired,
        }
    }
    /// Recognizes current owned output without inventing an L0 image occurrence.
    pub(super) fn matching_summary(
        &self,
        tx: &Transaction<'_>,
        desired: &[u8; 32],
    ) -> Result<Option<i64>, Error> {
        Ok(match self {
            Self::Text(input) => {
                input
                    .owner
                    .matching_summary(tx, input.parent_unit, &input.parent_hash, desired)?
            }
            Self::Image(input) => {
                let Some(parent) =
                    publication::image_parent(tx, input.source.id, &input.source.content_hash)?
                else {
                    return Ok(None);
                };
                Owner::Image(input.source.id).matching_summary(
                    tx,
                    parent,
                    &input.source.content_hash,
                    desired,
                )?
            }
            Self::Page(input) => publication::page::matching(
                tx,
                input.source.id,
                &input.source.content_hash,
                input.page.number,
                desired,
            )?,
        })
    }
    /// Supplies the durable one-based page identity for reconstruction and binding.
    pub(super) fn page(&self) -> Option<u32> {
        match self {
            Self::Page(input) => Some(input.page.number),
            _ => None,
        }
    }
    /// Existing image parents keep their allocated slot when a cap is lowered.
    pub(super) fn file_capacity(
        &self,
        tx: &Transaction<'_>,
        maximum: u64,
        required: u32,
    ) -> Result<Option<capacity::Reason>, Error> {
        let (file, needed) = match self {
            Self::Text(_) => return Ok(None),
            Self::Image(input) => {
                if publication::image_parent(tx, input.source.id, &input.source.content_hash)?
                    .is_some()
                {
                    return Ok(None);
                }
                (input.source.id, 1)
            }
            Self::Page(input) => {
                let parent = publication::page::parent(
                    tx,
                    input.source.id,
                    &input.source.content_hash,
                    input.page.number,
                )?;
                if let Some(parent) = parent {
                    let retained: i64 = tx.query_row("SELECT count(*) FROM units WHERE parent_unit=?1 AND file_id=?2 AND kind='doc_section' AND ladder_level=3", rusqlite::params![parent,input.source.id], |r| r.get(0))?;
                    if retained >= i64::from(required) {
                        return Ok(None);
                    }
                    let needed = u64::from(required).saturating_sub(retained.max(0) as u64);
                    (input.source.id, needed)
                } else {
                    (input.source.id, u64::from(required) + 1)
                }
            }
        };
        let count: i64 =
            tx.query_row("SELECT count(*) FROM units WHERE file_id=?1", [file], |r| {
                r.get(0)
            })?;
        let count = u64::try_from(count).map_err(|_| Error::Invalid("image source unit count"))?;
        Ok((count.saturating_add(needed) > maximum).then_some(capacity::Reason::FileUnits))
    }
    /// Uses the caller's completion transaction for either publication route.
    pub(super) fn publish(
        &self,
        context: &super::Context,
        tx: &Transaction<'_>,
        desired: &[u8; 32],
        text: &str,
        now: i64,
    ) -> Result<Vec<i64>, Error> {
        match self {
            Self::Text(input) => {
                let summary = context.store.publish_summary_in(
                    tx,
                    Publication {
                        owner: input.owner,
                        parent_unit: input.parent_unit,
                        parent_hash: &input.parent_hash,
                        desired_hash: desired,
                        text,
                        sources: &input.sources,
                        now,
                    },
                )?;
                Ok(vec![summary])
            }
            Self::Image(input) => {
                let sources = input
                    .captions
                    .files
                    .iter()
                    .copied()
                    .map(Source::File)
                    .collect::<Vec<_>>();
                let summary = context.store.publish_image_summary_in(
                    tx,
                    ImagePublication {
                        file_id: input.source.id,
                        file_hash: &input.source.content_hash,
                        desired_hash: desired,
                        text,
                        sources: &sources,
                        max_units_per_file: context.config.limits.max_units_per_file,
                        now,
                    },
                )?;
                Ok(vec![summary])
            }
            Self::Page(input) => {
                let maximum = context.config.limits.max_units_per_file;
                let sections = crate::treat::pdf::page_units(
                    text,
                    (input.page.number - 1) as usize,
                    input.page.title.as_deref(),
                    &context.config.chunk,
                    maximum,
                )
                .map_err(|_| Error::Invalid("page Markdown preparation"))?;
                let units = u32::try_from(sections.len())
                    .map_err(|_| Error::Invalid("page section count"))?;
                if units == 0 {
                    return Err(Error::Invalid("empty page transcription"));
                }
                if sections.len() as u64 > maximum
                    || sections.iter().any(|section| section.truncated)
                {
                    return Err(Error::PageCapacity {
                        reason: capacity::Reason::FileUnits,
                        units,
                    });
                }
                let parent = context
                    .store
                    .publish_page_in(
                        tx,
                        publication::page::Publication {
                            file: input.source.id,
                            file_hash: &input.source.content_hash,
                            page: input.page.number,
                            desired,
                            sections: &sections,
                            repo_name: &crate::sync::l1::repo_name(&context.root, &context.config),
                            tokenize: &context.config.tokenize,
                            max_units_per_file: maximum,
                            now,
                        },
                    )
                    .map_err(|error| {
                        let error = Error::Store(error);
                        match capacity::reason(&error) {
                            Some(reason) => Error::PageCapacity { reason, units },
                            None => error,
                        }
                    })?;
                let mut query = tx.prepare("SELECT id FROM units WHERE parent_unit=?1 AND kind='doc_section' AND ladder_level=3 ORDER BY id")?;
                Ok(query
                    .query_map([parent], |r| r.get(0))?
                    .collect::<Result<Vec<_>, _>>()?)
            }
        }
    }
}
