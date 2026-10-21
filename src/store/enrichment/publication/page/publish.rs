//! Atomic generated page sections, provenance and completeness; v1 §6.2/§6.8.
use super::{Error, matching, output_hash, parent, section_hash};
use crate::{
    config::Tokenize,
    store::{FileKind, FtsColumns, Store, UnitInsert, UnitKind, occurrences},
    treat::{
        PreparedUnit,
        context::{ContextFile, ContextUnit, build},
        fts::{FtsInput, build_columns},
    },
};
use rusqlite::{OptionalExtension, Transaction, params};

/// The caller has already verified source/privacy, lease and complete VLM output.
pub(crate) struct Publication<'a> {
    pub(crate) file: i64,
    pub(crate) file_hash: &'a [u8; 32],
    pub(crate) page: u32,
    pub(crate) desired: &'a [u8; 32],
    pub(crate) sections: &'a [PreparedUnit],
    pub(crate) repo_name: &'a str,
    pub(crate) tokenize: &'a Tokenize,
    pub(crate) max_units_per_file: u64,
    pub(crate) now: i64,
}

struct Old {
    id: i64,
    ordinal: u32,
    hash: Vec<u8>,
}

impl Store {
    /// Replaces only one PDF page's owned output in the completion transaction.
    /// Any error requires caller rollback, preserving prior sections and vectors.
    pub(crate) fn publish_page_in(
        &self,
        tx: &Transaction<'_>,
        input: Publication<'_>,
    ) -> Result<i64, Error> {
        if input.page == 0 || input.now < 0 || input.sections.is_empty() {
            return Err(Error::L1InputsChanged);
        }
        let path: Option<String> = tx.query_row(
            "SELECT path FROM files WHERE id=?1 AND kind='doc' AND blake3=?2 AND max_level>=3
             AND needs_vlm=1 AND l1_pending=0 AND l1_fingerprint IS NOT NULL AND l1_fingerprint=l1_desired
             AND l1_deferred_reason IN ('needs_vlm','max_pdf_pages','no_renderer')",
            params![input.file, input.file_hash.as_slice()], |r| r.get(0),
        ).optional()?;
        let path = path.ok_or(Error::L1InputsChanged)?;
        let page_header = format!("page {}", input.page);
        let mut hashes = Vec::with_capacity(input.sections.len());
        for section in input.sections {
            if section.kind != UnitKind::DocSection
                || section.truncated
                || section.text.trim().is_empty()
                || section.line_start == 0
                || section.line_end < section.line_start
                || !(section.header == page_header
                    || section.header.starts_with(&format!("{page_header} › ")))
            {
                return Err(Error::L1InputsChanged);
            }
            hashes.push(section_hash(
                input.desired,
                &section.header,
                &section.scope_path,
                i64::try_from(section.line_start).map_err(|_| Error::L1InputsChanged)?,
                i64::try_from(section.line_end).map_err(|_| Error::L1InputsChanged)?,
                &section.text,
            ));
        }
        let digest = output_hash(&hashes);
        let existing = parent(tx, input.file, input.file_hash, input.page)?;
        if let Some(anchor) = matching(tx, input.file, input.file_hash, input.page, input.desired)?
        {
            let same: bool = tx.query_row(
                "SELECT output_hash=?2 FROM page_publications WHERE parent_unit=?1",
                params![anchor, digest.as_slice()],
                |r| r.get(0),
            )?;
            if same {
                return Ok(anchor);
            }
        }
        let mut old = Vec::new();
        if let Some(anchor) = existing {
            let mut query = tx.prepare("SELECT id,duplicate_ordinal,content_hash,kind,ladder_level,file_id FROM units WHERE parent_unit=?1")?;
            let mut rows = query.query([anchor])?;
            while let Some(row) = rows.next()? {
                let kind: String = row.get(3)?;
                let level: i64 = row.get(4)?;
                let file: Option<i64> = row.get(5)?;
                if kind != "doc_section" || level != 3 || file != Some(input.file) {
                    return Err(Error::L1InputsChanged);
                }
                old.push(Old {
                    id: row.get(0)?,
                    ordinal: row.get(1)?,
                    hash: row.get(2)?,
                });
            }
        }
        let count: i64 = tx.query_row(
            "SELECT count(*) FROM units WHERE file_id=?1",
            [input.file],
            |r| r.get(0),
        )?;
        let count = u64::try_from(count).map_err(|_| Error::L1InputsChanged)?;
        let required = count
            .checked_sub(old.len() as u64)
            .and_then(|count| count.checked_add(input.sections.len() as u64))
            .and_then(|count| count.checked_add(u64::from(existing.is_none())))
            .ok_or(Error::L1InputsChanged)?;
        if required > count && required > input.max_units_per_file {
            return Err(Error::FileUnitCapReached {
                current: required,
                max_units_per_file: input.max_units_per_file,
            });
        }
        let before = self.enrichment_status_in(tx)?;
        if input.sections.len() > old.len() {
            before.admit_new()?;
        }
        // Remove changed sections first for bounded replacement headroom. Matching
        // occurrences keep their identifiers and already valid dense vectors.
        for previous in &old {
            let unchanged = hashes
                .get(previous.ordinal as usize)
                .is_some_and(|hash| previous.hash.as_slice() == hash);
            if !unchanged {
                tx.execute("DELETE FROM units WHERE id=?1", [previous.id])?;
            }
        }
        let anchor = if let Some(anchor) = existing {
            anchor
        } else {
            let collision: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM units WHERE hash=?1)",
                [occurrences::occurrence_hash(input.file_hash, &path, input.page - 1).as_slice()],
                |r| r.get(0),
            )?;
            if collision {
                return Err(Error::L1InputsChanged);
            }
            let id = occurrences::insert_unit_in_transaction(
                tx,
                UnitInsert {
                    hash: input.file_hash,
                    source_key: None,
                    duplicate_ordinal: input.page - 1,
                    file_id: Some(input.file),
                    kind: UnitKind::Image,
                    byte_start: None,
                    byte_end: None,
                    line_start: None,
                    line_end: None,
                    scope_path: None,
                    header: &page_header,
                    text: "",
                    tokens: 0,
                    ladder_level: 0,
                    parent_unit: None,
                    created_at: input.now,
                    updated_at: input.now,
                    fts: FtsColumns {
                        header: "",
                        idents: "",
                        body: "",
                        prose: "",
                    },
                },
                None,
            )?;
            tx.execute("DELETE FROM fts_units WHERE rowid=?1", [id])?;
            id
        };
        tx.execute("INSERT OR IGNORE INTO unit_sources(unit_id,source_kind,source_id,role) VALUES(?1,'file',?2,'content')", params![anchor, input.file])?;
        for (ordinal, (section, hash)) in input.sections.iter().zip(&hashes).enumerate() {
            let context = build(
                ContextUnit {
                    scope_path: Some(&section.scope_path),
                    header: &section.header,
                    language: None,
                    kind: "doc_section",
                    text: &section.text,
                    is_document: true,
                },
                ContextFile {
                    repo_name: input.repo_name,
                    path: &path,
                },
                &[],
            );
            let idents = section
                .idents
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            let columns = build_columns(
                FtsInput {
                    breadcrumb: &context.breadcrumb,
                    header: &context.header,
                    text: &context.text,
                    language: None,
                    kind: FileKind::Doc,
                    extra_idents: &idents,
                    has_code_fence: section.has_code_fence,
                },
                input.tokenize,
            );
            let id = occurrences::insert_unit_in_transaction(
                tx,
                UnitInsert {
                    hash,
                    source_key: None,
                    duplicate_ordinal: u32::try_from(ordinal)
                        .map_err(|_| Error::L1InputsChanged)?,
                    file_id: Some(input.file),
                    kind: UnitKind::DocSection,
                    byte_start: None,
                    byte_end: None,
                    line_start: Some(
                        i64::try_from(section.line_start).map_err(|_| Error::L1InputsChanged)?,
                    ),
                    line_end: Some(
                        i64::try_from(section.line_end).map_err(|_| Error::L1InputsChanged)?,
                    ),
                    scope_path: Some(&section.scope_path),
                    header: &section.header,
                    text: &section.text,
                    tokens: i64::try_from(crate::treat::count_tokens(&section.text))
                        .map_err(|_| Error::L1InputsChanged)?,
                    ladder_level: 3,
                    parent_unit: Some(anchor),
                    created_at: input.now,
                    updated_at: input.now,
                    fts: FtsColumns {
                        header: &columns.header,
                        idents: &columns.idents,
                        body: &columns.body,
                        prose: &columns.prose,
                    },
                },
                None,
            )?;
            tx.execute("DELETE FROM unit_sources WHERE unit_id=?1", [id])?;
            tx.execute("INSERT INTO unit_sources(unit_id,source_kind,source_id,role) VALUES(?1,'file',?2,'content'),(?1,'unit',?3,'content')", params![id, input.file, anchor])?;
        }
        tx.execute("INSERT INTO page_publications(file_id,page,parent_unit,desired_hash,sections,output_hash)
            VALUES(?1,?2,?3,?4,?5,?6) ON CONFLICT(file_id,page) DO UPDATE SET parent_unit=excluded.parent_unit,
            desired_hash=excluded.desired_hash,sections=excluded.sections,output_hash=excluded.output_hash",
            params![input.file, input.page, anchor, input.desired.as_slice(), input.sections.len() as i64, digest.as_slice()])?;
        before.check_growth(&self.enrichment_status_in(tx)?)?;
        let next = super::super::summary_generation_in(tx)?
            .checked_add(1)
            .ok_or(Error::SummaryGeneration)?;
        tx.execute("INSERT INTO meta(key,value) VALUES('summary_generation',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value", [next.to_string()])?;
        Ok(anchor)
    }
}
