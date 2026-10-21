//! Owner-bound summary replacement within worker completion; v1 §6.8 and §8.2.
use super::super::{Error, FtsColumns, Store, UnitInsert, UnitKind, occurrences};
use rusqlite::{OptionalExtension, Transaction, params};
mod image;
pub(crate) mod page;
pub(crate) use image::{ImagePublication, image_parent};

/// Logical job owner; shared source bytes never merge independently owned summaries.
#[derive(Clone, Copy)]
pub(crate) enum Owner {
    Symbol(i64),
    Pr(i64),
    Module(i64),
    Image(i64),
}
impl Owner {
    /// Finds this owner's output only when it proves the supplied current generation.
    /// The caller must first verify current source and policy inputs.
    pub(crate) fn matching_summary(
        self,
        tx: &Transaction<'_>,
        parent: i64,
        parent_hash: &[u8; 32],
        desired: &[u8; 32],
    ) -> Result<Option<i64>, Error> {
        let Some(Some(id)) = self.current(tx, parent)? else {
            return Ok(None);
        };
        let stored: Option<(Vec<u8>, String)> = tx.query_row(
            "SELECT u.content_hash,c.text FROM units u JOIN unit_contents c ON c.hash=u.content_hash
             WHERE u.id=?1 AND u.parent_unit=?2 AND u.kind='summary' AND u.source_key=?3",
            params![id,parent,self.key()], |r| Ok((r.get(0)?,r.get(1)?)),
        ).optional()?;
        Ok(stored
            .filter(|(hash, text)| hash.as_slice() == content_hash(parent_hash, desired, text))
            .map(|_| id))
    }
    fn key(self) -> String {
        match self {
            Self::Symbol(id) => format!("summary:symbol:{id}"),
            Self::Pr(id) => format!("summary:pr:{id}"),
            Self::Module(id) => format!("summary:module:{id}"),
            Self::Image(id) => format!("summary:image:{id}"),
        }
    }
    fn current(self, tx: &Transaction<'_>, parent: i64) -> Result<Option<Option<i64>>, Error> {
        let (sql, id) = match self {
            Self::Symbol(id) => (
                "SELECT s.summary_unit FROM symbols s JOIN units u ON u.id=s.unit_id WHERE s.id=?1 AND s.unit_id=?2 AND u.file_id=s.file_id AND u.kind IN ('symbol','chunk')",
                id,
            ),
            Self::Pr(id) => (
                "SELECT p.summary_unit FROM prs p JOIN units u ON u.id=p.unit_id WHERE p.id=?1 AND p.unit_id=?2 AND u.kind='pr' AND u.file_id IS NULL",
                id,
            ),
            Self::Module(id) => (
                "SELECT d.summary_unit FROM dirs d JOIN units u ON u.id=d.module_unit WHERE d.module_unit=?1 AND d.module_unit=?2 AND u.kind='module' AND u.file_id IS NULL",
                id,
            ),
            Self::Image(id) => (
                "SELECT (SELECT s.id FROM units s WHERE s.parent_unit=p.id AND s.kind='summary' AND s.source_key='summary:image:'||f.id)
                 FROM files f JOIN units p ON p.file_id=f.id WHERE f.id=?1 AND p.id=?2 AND f.kind='image' AND p.kind='image' AND p.ladder_level=0 AND p.parent_unit IS NULL",
                id,
            ),
        };
        Ok(tx
            .query_row(sql, params![id, parent], |r| r.get(0))
            .optional()?)
    }
    fn attach(self, tx: &Transaction<'_>, summary: i64) -> Result<(), Error> {
        if let Self::Image(file) = self {
            let owned: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM units s JOIN units p ON p.id=s.parent_unit
                 JOIN files f ON f.id=p.file_id WHERE s.id=?1 AND f.id=?2 AND f.kind='image'
                 AND p.kind='image' AND p.ladder_level=0 AND p.parent_unit IS NULL AND s.kind='summary' AND s.source_key='summary:image:'||f.id)",
                params![summary, file],
                |r| r.get(0),
            )?;
            return if owned {
                Ok(())
            } else {
                Err(Error::L1InputsChanged)
            };
        }
        let (sql, id) = match self {
            Self::Symbol(id) => ("UPDATE symbols SET summary_unit=?2 WHERE id=?1", id),
            Self::Pr(id) => ("UPDATE prs SET summary_unit=?2 WHERE id=?1", id),
            Self::Module(id) => ("UPDATE dirs SET summary_unit=?2 WHERE module_unit=?1", id),
            Self::Image(_) => return Err(Error::L1InputsChanged),
        };
        if tx.execute(sql, params![id, summary])? != 1 {
            return Err(Error::L1InputsChanged);
        }
        Ok(())
    }
}

/// Additional inputs already verified by the prepared worker at completion.
#[derive(Clone, Copy)]
pub(crate) enum Source {
    Unit(i64),
    File(i64),
    Symbol(i64),
    Commit(i64),
}
impl Source {
    fn exists(self, tx: &Transaction<'_>) -> Result<bool, Error> {
        let (sql, id) = match self {
            Self::Unit(id) => ("SELECT EXISTS(SELECT 1 FROM units WHERE id=?1)", id),
            Self::File(id) => ("SELECT EXISTS(SELECT 1 FROM files WHERE id=?1)", id),
            Self::Symbol(id) => ("SELECT EXISTS(SELECT 1 FROM symbols WHERE id=?1)", id),
            Self::Commit(id) => ("SELECT EXISTS(SELECT 1 FROM commits WHERE id=?1)", id),
        };
        Ok(tx.query_row(sql, [id], |r| r.get(0))?)
    }
    fn parts(self) -> (&'static str, i64) {
        match self {
            Self::Unit(id) => ("unit", id),
            Self::File(id) => ("file", id),
            Self::Symbol(id) => ("symbol", id),
            Self::Commit(id) => ("commit", id),
        }
    }
}

/// Complete prepared publication; source text deliberately has no Debug output.
pub(crate) struct Publication<'a> {
    pub(crate) owner: Owner,
    pub(crate) parent_unit: i64,
    pub(crate) parent_hash: &'a [u8; 32],
    pub(crate) desired_hash: &'a [u8; 32],
    pub(crate) text: &'a str,
    pub(crate) sources: &'a [Source],
    pub(crate) now: i64,
}

/// Recomputes a summary's content proof from verified current inputs and stored output.
pub(crate) fn content_hash(parent: &[u8; 32], desired: &[u8; 32], text: &str) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key("dkc enrichment summary content v1");
    for bytes in [parent.as_slice(), desired.as_slice(), text.as_bytes()] {
        hasher.update(&(bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    *hasher.finalize().as_bytes()
}

impl Store {
    /// Observes committed summary publications so idle workers can refresh their
    /// embedding catalogue without pretending that source files changed.
    pub(crate) fn summary_generation(&self) -> Result<i64, Error> {
        self.read_snapshot(summary_generation_in)?
    }
    /// Publishes one summary and all provenance in the caller's completion transaction.
    /// Caller must roll back on error; worker input/lease fences precede this call.
    /// Replacement deletes only this logical owner's prior summary and obeys caps.
    pub(crate) fn publish_summary_in(
        &self,
        tx: &Transaction<'_>,
        input: Publication<'_>,
    ) -> Result<i64, Error> {
        if input.text.trim().is_empty() || input.now < 0 {
            return Err(Error::L1InputsChanged);
        }
        let old = input
            .owner
            .current(tx, input.parent_unit)?
            .ok_or(Error::L1InputsChanged)?;
        let parent: Option<(Vec<u8>, String, Option<String>)> = tx
            .query_row(
                "SELECT content_hash,header,scope_path FROM units WHERE id=?1",
                [input.parent_unit],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;
        let (hash, header, scope) = parent.ok_or(Error::L1InputsChanged)?;
        if hash.as_slice() != input.parent_hash {
            return Err(Error::L1InputsChanged);
        }
        let key = input.owner.key();
        if let Some(old) = old {
            let owned:bool=tx.query_row("SELECT EXISTS(SELECT 1 FROM units WHERE id=?1 AND parent_unit=?2 AND kind='summary' AND source_key=?3)",params![old,input.parent_unit,key],|r|r.get(0))?;
            if !owned {
                return Err(Error::L1InputsChanged);
            }
        }
        for source in input.sources {
            if !source.exists(tx)? || matches!(source, Source::Unit(id) if Some(*id)==old) {
                return Err(Error::L1InputsChanged);
            }
        }
        let before = self.enrichment_status_in(tx)?;
        if old.is_none() {
            before.admit_new()?;
        }
        let content = content_hash(input.parent_hash, input.desired_hash, input.text);
        let mut changed = old.is_none();
        // Unchanged output/generation keeps the occurrence and any valid vectors.
        if let Some(old) = old {
            let same: bool = tx.query_row(
                "SELECT content_hash=?2 FROM units WHERE id=?1",
                params![old, content.as_slice()],
                |r| r.get(0),
            )?;
            if !same {
                changed = true;
                tx.execute("DELETE FROM units WHERE id=?1", [old])?;
            }
        }
        let id = occurrences::insert_unit_in_transaction(
            tx,
            UnitInsert {
                hash: &content,
                source_key: Some(&key),
                duplicate_ordinal: 0,
                file_id: None,
                kind: UnitKind::Summary,
                byte_start: None,
                byte_end: None,
                line_start: None,
                line_end: None,
                scope_path: scope.as_deref(),
                header: &header,
                text: input.text,
                tokens: crate::treat::count_tokens(input.text) as i64,
                ladder_level: 3,
                parent_unit: Some(input.parent_unit),
                created_at: input.now,
                updated_at: input.now,
                fts: FtsColumns {
                    header: &header,
                    idents: "",
                    body: input.text,
                    prose: input.text,
                },
            },
            None,
        )?;
        tx.execute("DELETE FROM unit_sources WHERE unit_id=?1", [id])?;
        for source in
            std::iter::once(Source::Unit(input.parent_unit)).chain(input.sources.iter().copied())
        {
            let (kind, source_id) = source.parts();
            tx.execute("INSERT OR IGNORE INTO unit_sources(unit_id,source_kind,source_id,role) VALUES(?1,?2,?3,'content')",params![id,kind,source_id])?;
        }
        input.owner.attach(tx, id)?;
        before.check_growth(&self.enrichment_status_in(tx)?)?;
        if changed {
            let next = summary_generation_in(tx)?
                .checked_add(1)
                .ok_or(Error::SummaryGeneration)?;
            tx.execute(
                "INSERT INTO meta(key,value) VALUES('summary_generation',?1)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                [next.to_string()],
            )?;
        }
        Ok(id)
    }
}

/// Reads the counter in the same snapshot as related owner/output state.
pub(crate) fn summary_generation_in(tx: &Transaction<'_>) -> Result<i64, Error> {
    let value: Option<String> = tx
        .query_row(
            "SELECT value FROM meta WHERE key='summary_generation'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    match value {
        None => Ok(0),
        Some(value) => value
            .parse::<i64>()
            .ok()
            .filter(|value| *value >= 0)
            .ok_or(Error::SummaryGeneration),
    }
}

#[cfg(test)]
mod tests;
