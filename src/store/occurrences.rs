//! Shared payloads, source occurrences and L1 replacement; see v1.md §3.2–3.4.

use std::collections::BTreeSet;

use rusqlite::{OptionalExtension, Transaction, params};

use super::{Error, Store, SymbolInsert, UnitEdgeInsert, UnitInsert, admission};

/// One prepared occurrence and its graph input, borrowed until atomic publication.
#[derive(Clone)]
pub(crate) struct IndexedUnitInput<'a> {
    pub(crate) unit: UnitInsert<'a>,
    pub(crate) symbols: Vec<SymbolInsert<'a>>,
    pub(crate) edges: Vec<UnitEdgeInsert<'a>>,
}

/// Computes a public occurrence identity independently of SQLite allocation order.
pub(super) fn occurrence_hash(content: &[u8], source: &str, ordinal: u32) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for part in [content, source.as_bytes(), &ordinal.to_le_bytes()] {
        hasher.update(&(part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    *hasher.finalize().as_bytes()
}

/// Whether the requested L1 snapshot fit or a visible lower-level outcome was retained.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PublicationOutcome {
    Indexed,
    StorageDeferred,
}

impl Store {
    /// Publishes L1 or records storage exhaustion after rollback, preserving old
    /// content. Recheck source/configuration again before committing the deferral.
    /// Preserve optional cheap admission checks across both writer acquisitions.
    pub(crate) fn publish_file_content(
        &self,
        admission_check: Option<&dyn Fn() -> Result<(), Error>>,
        fence: super::l1::Fence<'_>,
        units: &[IndexedUnitInput<'_>],
        completion: impl Into<super::l1::Completion>,
        validate_inputs: impl Fn() -> Result<(), Error>,
    ) -> Result<PublicationOutcome, Error> {
        match self.replace_file_content(admission_check, fence, units, completion, &validate_inputs)
        {
            Ok(()) => Ok(PublicationOutcome::Indexed),
            Err(Error::Sqlite(error))
                if error.sqlite_error_code() == Some(rusqlite::ErrorCode::DiskFull) =>
            {
                self.defer_file_l1(
                    admission_check,
                    fence,
                    super::l1::CapDeferral::Storage,
                    validate_inputs,
                )?;
                Ok(PublicationOutcome::StorageDeferred)
            }
            Err(error) => Err(error),
        }
    }

    /// Replaces one file's occurrences without removing unchanged vectors.
    ///
    /// All parsing/context work precedes this transaction. The captured source,
    /// desired configuration and catalogue generation must still match. The caller's
    /// repeatable input/work check runs after writer admission and just before commit.
    /// Optional cheap checks bound writer waits separately from source validation.
    pub(crate) fn replace_file_content(
        &self,
        admission_check: Option<&dyn Fn() -> Result<(), Error>>,
        fence: super::l1::Fence<'_>,
        units: &[IndexedUnitInput<'_>],
        completion: impl Into<super::l1::Completion>,
        validate_inputs: impl Fn() -> Result<(), Error>,
    ) -> Result<(), Error> {
        let completion = completion.into();
        admission::transaction_checked(
            self,
            admission::Kind::L1Growth,
            admission_check,
            |transaction| {
                validate_inputs()?;
                super::l1::check_file(&transaction, fence)?;
                let file_id = fence.file_id;
                let mut obsolete = transaction
                    // L1 replacement owns source units; summaries remain attached to
                    // retained parents and cascade only when those parents retire.
                    // L0 image anchors belong to visual publication, not the empty
                    // L1 chunker. Keep a canonical anchor while its image bytes
                    // remain current, including configuration/cap-only refreshes.
                    .prepare_cached("SELECT u.id FROM units u JOIN files f ON f.id=u.file_id
                        WHERE u.file_id=?1 AND u.kind!='summary' AND NOT (
                            u.kind='image' AND u.ladder_level=0 AND u.parent_unit IS NULL
                            AND u.content_hash=?2 AND u.source_key=f.path
                            AND ((f.kind='image' AND u.duplicate_ordinal=0) OR f.kind='doc'))
                        AND NOT (f.kind='doc' AND u.kind='doc_section' AND u.ladder_level=3
                            AND u.source_key=f.path AND EXISTS(SELECT 1 FROM units p
                                WHERE p.id=u.parent_unit AND p.file_id=f.id AND p.kind='image'
                                AND p.ladder_level=0 AND p.parent_unit IS NULL
                                AND p.content_hash=?2 AND p.source_key=f.path))")?
                    .query_map(params![file_id,fence.content_hash], |row| row.get(0))?
                    .collect::<Result<BTreeSet<i64>, _>>()?;
                let mut seen = BTreeSet::new();
                for input in units {
                    if input.unit.file_id != Some(file_id) {
                        return Err(Error::GraphUnitFileRequired);
                    }
                    let written = super::write_indexed_unit(
                        &transaction,
                        input.unit,
                        &input.symbols,
                        &input.edges,
                        Some(&obsolete),
                    )?;
                    if !seen.insert(written.unit_id) {
                        return Err(Error::DuplicateOccurrence {
                            unit_id: written.unit_id,
                        });
                    }
                    obsolete.remove(&written.unit_id);
                    #[cfg(test)]
                    publication_observed(PublicationPoint::OccurrenceWritten);
                }
                if !obsolete.is_empty() {
                    // Bind integer-only JSON instead of one placeholder per unit:
                    // configured caps may exceed SQLite's variable limit. One
                    // statement runs all existing FK/derived cleanup programs.
                    let ids = format!(
                        "[{}]",
                        obsolete
                            .iter()
                            .map(i64::to_string)
                            .collect::<Vec<_>>()
                            .join(",")
                    );
                    super::graph_reuse::remove_obsolete(&transaction, file_id, &ids)?;
                    transaction.prepare_cached(
                        "DELETE FROM units WHERE file_id=?1 AND id IN (SELECT value FROM json_each(?2))",
                    )?.execute(params![file_id, ids])?;
                }
                super::graph_reuse::remove(&transaction, file_id, None)?;
                #[cfg(test)]
                publication_observed(PublicationPoint::BeforeCompletion);
                transaction.execute(
            "UPDATE files SET truncated = ?1, l1_pending = 0, l1_fingerprint=?3, l1_deferred_reason=?4, needs_vlm=?5 WHERE id = ?2",
            params![completion.truncated, file_id, fence.fingerprint, completion.reason, completion.needs_vlm],
        )?;
                // A full reparse rewrites lexical headers even when content and
                // role receipts are unchanged. Reconciliation consumes this
                // marker atomically after re-proving the relevant folder roles.
                transaction.execute("INSERT INTO meta(key,value)
                    SELECT 'lexical_role_dirty:'||?1,'1' WHERE EXISTS(SELECT 1 FROM meta WHERE key='lexical_role_receipt_v1')
                    OR EXISTS(SELECT 1 FROM dirs WHERE summary_unit IS NOT NULL)
                    ON CONFLICT(key) DO NOTHING", [file_id])?;
                validate_inputs()?;
                if let Some(check) = admission_check {
                    check()?;
                }
                transaction.commit()?;
                Ok(())
            },
        )
    }
}

/// Test-only observation points inside the real publication transaction.
#[cfg(test)]
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum PublicationPoint {
    OccurrenceWritten,
    BeforeCompletion,
    BeforeGraphRank,
}

#[cfg(test)]
thread_local! {
    static PUBLICATION_OBSERVER: std::cell::Cell<Option<fn(PublicationPoint)>> = const { std::cell::Cell::new(None) };
}

/// Installs a thread-local observer used by real-process crash/concurrency tests.
#[cfg(test)]
pub(crate) fn observe_publication(observer: fn(PublicationPoint)) {
    PUBLICATION_OBSERVER.with(|slot| slot.set(Some(observer)));
}

#[cfg(test)]
pub(crate) fn publication_observed(point: PublicationPoint) {
    PUBLICATION_OBSERVER.with(|slot| {
        if let Some(observer) = slot.get() {
            observer(point);
        }
    });
}

/// Stores shared text once and updates one source occurrence and its FTS row.
pub(super) fn insert_unit_in_transaction(
    transaction: &Transaction<'_>,
    unit: UnitInsert<'_>,
    existing: Option<&BTreeSet<i64>>,
) -> Result<i64, Error> {
    let source = match unit.file_id {
        Some(file_id) => transaction
            .prepare_cached("SELECT path FROM files WHERE id = ?1")?
            .query_row([file_id], |row| row.get::<_, String>(0))
            .optional()?
            .ok_or(Error::GraphFileMissing { file_id })?,
        None => unit.source_key.ok_or(Error::UnitSourceRequired)?.to_owned(),
    };
    transaction
        .prepare_cached(
            "INSERT INTO unit_contents(hash, text, tokens) VALUES (?1, ?2, ?3)
         ON CONFLICT(hash) DO NOTHING",
        )?
        .execute(params![unit.hash, unit.text, unit.tokens])?;
    let same: bool = transaction
        .prepare_cached("SELECT text = ?2 FROM unit_contents WHERE hash = ?1")?
        .query_row(params![unit.hash, unit.text], |row| row.get(0))?;
    if !same {
        return Err(Error::ContentHashConflict);
    }
    let hash = occurrence_hash(unit.hash, &source, unit.duplicate_ordinal);
    let id = transaction.prepare_cached(
        "INSERT INTO units(hash, content_hash, source_key, duplicate_ordinal, file_id,
         kind, byte_start, byte_end, line_start, line_end, scope_path, header,
         ladder_level, parent_unit, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)
         ON CONFLICT(hash) DO UPDATE SET byte_start = excluded.byte_start,
         byte_end = excluded.byte_end, line_start = excluded.line_start, line_end = excluded.line_end,
         scope_path = excluded.scope_path, header = excluded.header,
         ladder_level = max(units.ladder_level, excluded.ladder_level),
         parent_unit = excluded.parent_unit, updated_at = excluded.updated_at
         RETURNING id")?.query_row(
        params![hash.as_slice(), unit.hash, source, unit.duplicate_ordinal, unit.file_id,
            unit.kind.as_str(), unit.byte_start, unit.byte_end, unit.line_start, unit.line_end,
            unit.scope_path, unit.header, unit.ladder_level, unit.parent_unit,
            unit.created_at, unit.updated_at], |row| row.get::<_, i64>(0),
    )?;
    // The caller's file-local set was read in this transaction, before any old
    // units were removed. A new ID therefore cannot have an existing FTS row.
    // Unknown callers and existing IDs still replace all lexical columns.
    if existing.is_none_or(|ids| ids.contains(&id)) {
        transaction
            .prepare_cached("DELETE FROM fts_units WHERE rowid = ?1")?
            .execute([id])?;
    }
    transaction
        .prepare_cached(
            "INSERT INTO fts_units(rowid, header, idents, body, prose) VALUES (?1, ?2, ?3, ?4, ?5)",
        )?
        .execute(params![
            id,
            unit.fts.header,
            unit.fts.idents,
            unit.fts.body,
            unit.fts.prose
        ])?;
    Ok(id)
}

/// Updates occurrence IDs on rename while preserving the rowids vectors reference.
pub(super) fn rename_occurrences(
    transaction: &Transaction<'_>,
    file_id: i64,
    path: &str,
) -> Result<(), Error> {
    let rows = transaction
        .prepare("SELECT id, content_hash, duplicate_ordinal FROM units WHERE file_id = ?1")?
        .query_map([file_id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, u32>(2)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    for (id, content, ordinal) in rows {
        let hash = occurrence_hash(&content, path, ordinal);
        transaction.execute(
            "UPDATE units SET hash = ?1, source_key = ?2 WHERE id = ?3",
            params![hash.as_slice(), path, id],
        )?;
    }
    Ok(())
}

/// Finishes SQL migration without inventing ranges or vectors for other sources.
pub(super) fn migrate_occurrences(transaction: &Transaction<'_>) -> Result<(), Error> {
    let rows = transaction
        .prepare("SELECT id, content_hash, source_key FROM units ORDER BY id")?
        .query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    // Move old content hashes out of the way before installing occurrence hashes.
    // All this is private to the migration transaction.
    transaction.execute(
        "UPDATE units SET hash = CAST('migrating:' || id AS BLOB)",
        [],
    )?;
    for (id, content, source) in rows {
        let hash = occurrence_hash(&content, &source, 0);
        transaction.execute(
            "UPDATE units SET hash = ?1 WHERE id = ?2",
            params![hash.as_slice(), id],
        )?;
    }
    let slugs = transaction
        .prepare("SELECT slug FROM embedding_tables")?
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    for slug in slugs {
        let table = super::vectors::vector_table_name(&slug)?;
        transaction.execute(
            &format!(
                "UPDATE {table} SET ctx_hash = zeroblob(32) WHERE unit_id IN (
             SELECT unit_id FROM unit_sources WHERE source_kind = 'file'
             GROUP BY unit_id HAVING count(*) > 1)"
            ),
            [],
        )?;
    }
    transaction.execute(
        "DELETE FROM unit_sources WHERE source_kind = 'file'
         AND source_id != (SELECT file_id FROM units WHERE id = unit_id)",
        [],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rusqlite::types::Value;

    use super::*;
    use crate::store::{
        FileKind, FileUpsert, FtsColumns, GraphEdgeKind, StoreLimits, UnitKind, open,
        vectors::{VectorInsert, VectorTableSpec},
    };

    fn input(
        file_id: Option<i64>,
        hash: &'static [u8; 32],
        text: &'static str,
    ) -> IndexedUnitInput<'static> {
        IndexedUnitInput {
            unit: UnitInsert {
                hash,
                source_key: Some("module:fixture"),
                duplicate_ordinal: 0,
                file_id,
                kind: UnitKind::Symbol,
                byte_start: Some(0),
                byte_end: Some(text.len() as i64),
                line_start: Some(1),
                line_end: Some(1),
                scope_path: None,
                header: text,
                text,
                tokens: 1,
                ladder_level: 1,
                parent_unit: None,
                created_at: 1,
                updated_at: 1,
                fts: FtsColumns {
                    header: text,
                    idents: text,
                    body: text,
                    prose: "",
                },
            },
            symbols: vec![SymbolInsert {
                unit_id: None,
                name: text,
                qualified: None,
                kind: "function",
                signature: None,
                line_start: 1,
                line_end: 1,
            }],
            edges: vec![UnitEdgeInsert {
                src_symbol_index: 0,
                dst_name: text,
                dst_symbol: None,
                dst_symbol_index: Some(0),
                kind: GraphEdgeKind::Ref,
                line: Some(1),
            }],
        }
    }

    fn vector_table(store: &Store) -> Result<(), Error> {
        store.ensure_vector_table(VectorTableSpec {
            slug: "occurrence_test",
            model: "test",
            profile: "test",
            dims: 2,
            created_at: 1,
            active: true,
        })
    }

    fn snapshot(store: &Store) -> Result<Vec<Vec<Value>>, Error> {
        let writer = store.writer()?;
        let mut rows = Vec::new();
        for table in [
            "files",
            "units",
            "unit_contents",
            "symbols",
            "edges",
            "unit_sources",
            "fts_names",
            "vec_occurrence_test",
        ] {
            let mut statement = writer.prepare(&format!("SELECT * FROM {table} ORDER BY rowid"))?;
            let columns = statement.column_count();
            rows.extend(
                statement
                    .query_map([], |row| (0..columns).map(|i| row.get(i)).collect())?
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        Ok(rows)
    }

    #[test]
    fn enrichment_summary_survives_retained_parent_and_cascades_with_changed_parent()
    -> Result<(), Box<dyn StdError>> {
        let store = open(":memory:", StoreLimits::default())?;
        let file = store.upsert_file(FileUpsert {
            path: "summary-owner.rs",
            blake3: &[1; 32],
            size: 20,
            mtime_ns: 1,
            lang: Some("rust"),
            kind: FileKind::Code,
            git_blob: None,
            max_level: 3,
            treatment: "default",
            indexed_at: Some(1),
            needs_vlm: false,
        })?;
        store.writer()?.execute(
            "UPDATE files SET l1_desired=?1 WHERE id=?2",
            params![[7_u8; 32].as_slice(), file],
        )?;
        let fence = crate::store::l1::Fence {
            generation: 0,
            file_id: file,
            path: "summary-owner.rs",
            content_hash: &[1; 32],
            fingerprint: &[7; 32],
        };
        let mut parent = input(Some(file), &[2; 32], "retainedfunction");
        store.replace_file_content(None, fence, &[parent.clone()], false, || Ok(()))?;
        let parent_id: i64 = store.writer()?.query_row(
            "SELECT unit_id FROM symbols WHERE name='retainedfunction'",
            [],
            |r| r.get(0),
        )?;
        let mut summary = input(Some(file), &[3; 32], "summarylexeme").unit;
        summary.kind = UnitKind::Summary;
        summary.parent_unit = Some(parent_id);
        summary.ladder_level = 3;
        let summary_id = store.insert_unit(summary)?;
        store.writer()?.execute(
            "UPDATE symbols SET summary_unit=?1 WHERE unit_id=?2",
            params![summary_id, parent_id],
        )?;
        parent.unit.line_end = Some(2);
        parent.symbols[0].line_end = 2;
        store.replace_file_content(None, fence, &[parent], false, || Ok(()))?;
        assert!(
            store.unit_by_id(summary_id)?.is_some(),
            "unchanged parent must retain its real summary child"
        );
        let matches: i64 = store.writer()?.query_row(
            "SELECT COUNT(*) FROM fts_units WHERE fts_units MATCH 'summarylexeme'",
            [],
            |r| r.get(0),
        )?;
        assert_eq!(matches, 1);
        store.replace_file_content(
            None,
            fence,
            &[input(Some(file), &[4; 32], "changedfunction")],
            false,
            || Ok(()),
        )?;
        assert!(store.unit_by_id(summary_id)?.is_none());
        let matches: i64 = store.writer()?.query_row(
            "SELECT COUNT(*) FROM fts_units WHERE fts_units MATCH 'summarylexeme'",
            [],
            |r| r.get(0),
        )?;
        assert_eq!(matches, 0);
        Ok(())
    }

    #[test]
    fn known_occurrences_replace_lexical_columns_and_new_ids_publish_atomically()
    -> Result<(), Box<dyn StdError>> {
        let store = open(":memory:", StoreLimits::default())?;
        let file_id = store.upsert_file(FileUpsert {
            path: "source.rs",
            blake3: &[1; 32],
            size: 1,
            mtime_ns: 1,
            lang: Some("rust"),
            kind: FileKind::Code,
            git_blob: None,
            max_level: 2,
            treatment: "default",
            indexed_at: Some(1),
            needs_vlm: false,
        })?;
        store.writer()?.execute(
            "UPDATE files SET l1_desired=?1 WHERE id=?2",
            params![[7_u8; 32].as_slice(), file_id],
        )?;
        let fence = crate::store::l1::Fence {
            generation: 0,
            file_id,
            path: "source.rs",
            content_hash: &[1; 32],
            fingerprint: &[7; 32],
        };
        let mut unit = input(Some(file_id), &[2; 32], "oldlexeme");
        store.replace_file_content(None, fence, &[unit.clone()], false, || Ok(()))?;
        let id: i64 =
            store
                .writer()?
                .query_row("SELECT id FROM units WHERE file_id=?1", [file_id], |r| {
                    r.get(0)
                })?;
        let matches = |term: &str| -> Result<Vec<i64>, Error> {
            Ok(store
                .writer()?
                .prepare("SELECT rowid FROM fts_units WHERE fts_units MATCH ?1")?
                .query_map([term], |r| r.get(0))?
                .collect::<Result<Vec<_>, _>>()?)
        };
        assert_eq!(matches("oldlexeme")?, [id]);
        // Same payload/occurrence, different lexical context: replace every FTS
        // column rather than assuming content identity means identical tokens.
        unit.unit.fts = FtsColumns {
            header: "newlexeme",
            idents: "",
            body: "",
            prose: "",
        };
        store.replace_file_content(None, fence, &[unit.clone()], false, || Ok(()))?;
        assert!(matches("oldlexeme")?.is_empty());
        assert_eq!(matches("newlexeme")?, [id]);
        let replacement = input(Some(file_id), &[3; 32], "replacementlexeme");
        let calls = std::cell::Cell::new(0);
        assert!(matches!(
            store.replace_file_content(
                None,
                fence,
                std::slice::from_ref(&replacement),
                false,
                || {
                    calls.set(calls.get() + 1);
                    if calls.get() == 2 {
                        Err(Error::L1InputsChanged)
                    } else {
                        Ok(())
                    }
                }
            ),
            Err(Error::L1InputsChanged)
        ));
        assert_eq!(matches("newlexeme")?, [id]);
        assert!(matches("replacementlexeme")?.is_empty());
        store.replace_file_content(None, fence, &[replacement], false, || Ok(()))?;
        assert!(matches("newlexeme")?.is_empty());
        let new_ids = matches("replacementlexeme")?;
        assert_eq!(new_ids.len(), 1);
        assert_ne!(new_ids[0], id);
        assert_eq!(
            store.writer()?.query_row(
                "SELECT count(*) FROM units WHERE file_id=?1",
                [file_id],
                |r| r.get::<_, i64>(0)
            )?,
            1
        );
        Ok(())
    }

    #[test]
    fn unchanged_graph_retains_ids_ranks_and_incoming_links_then_updates_atomically()
    -> Result<(), Box<dyn StdError>> {
        let store = open(":memory:", StoreLimits::default())?;
        let file = |path| {
            store.upsert_file(FileUpsert {
                path,
                blake3: &[1; 32],
                size: 1,
                mtime_ns: 1,
                lang: Some("rust"),
                kind: FileKind::Code,
                git_blob: None,
                max_level: 2,
                treatment: "default",
                indexed_at: Some(1),
                needs_vlm: false,
            })
        };
        let owner = file("owner.rs")?;
        let caller_file = file("caller.rs")?;
        store.writer()?.execute(
            "UPDATE files SET l1_desired=?1 WHERE id=?2",
            params![[7_u8; 32].as_slice(), owner],
        )?;
        let fence = crate::store::l1::Fence {
            generation: 0,
            file_id: owner,
            path: "owner.rs",
            content_hash: &[1; 32],
            fingerprint: &[7; 32],
        };
        let retained = input(Some(owner), &[2; 32], "retainednode");
        let replaced = input(Some(owner), &[3; 32], "oldnode");
        store.replace_file_content(
            None,
            fence,
            &[retained.clone(), replaced.clone()],
            false,
            || Ok(()),
        )?;
        let caller = input(Some(caller_file), &[4; 32], "callernode");
        let caller = store.insert_indexed_unit(caller.unit, &caller.symbols, &caller.edges)?;
        let retained_id: i64 = store.writer()?.query_row(
            "SELECT id FROM symbols WHERE name='retainednode'",
            [],
            |r| r.get(0),
        )?;
        let old_id: i64 =
            store
                .writer()?
                .query_row("SELECT id FROM symbols WHERE name='oldnode'", [], |r| {
                    r.get(0)
                })?;
        store.writer()?.execute("INSERT INTO edges(src_symbol,dst_name,dst_symbol,kind,file_id,line) VALUES(?1,'retainednode',?2,'ref',?3,1)", params![caller.symbols[0].id,retained_id,caller_file])?;
        store.writer()?.execute(
            "UPDATE symbols SET pagerank=0.75,summary_unit=?1 WHERE id=?2",
            params![caller.unit_id, retained_id],
        )?;
        store.set_meta("graph_pending", "0")?;
        vector_table(&store)?;
        let before = snapshot(&store)?;
        store.replace_file_content(
            None,
            fence,
            &[retained.clone(), replaced.clone()],
            false,
            || Ok(()),
        )?;
        assert_eq!(snapshot(&store)?, before);
        assert_eq!(store.get_meta("graph_pending")?.as_deref(), Some("0"));

        let mut moved = retained.clone();
        moved.unit.line_end = Some(2);
        moved.symbols[0].line_end = 2;
        let calls = std::cell::Cell::new(0);
        assert!(matches!(
            store.replace_file_content(
                None,
                fence,
                &[moved.clone(), replaced.clone()],
                false,
                || {
                    calls.set(calls.get() + 1);
                    if calls.get() == 2 {
                        Err(Error::L1InputsChanged)
                    } else {
                        Ok(())
                    }
                }
            ),
            Err(Error::L1InputsChanged)
        ));
        assert_eq!(snapshot(&store)?, before);
        store.replace_file_content(None, fence, &[moved.clone(), replaced], false, || Ok(()))?;
        assert_eq!(store.get_meta("graph_pending")?.as_deref(), Some("0"));
        let value: (i64, i64, f64, Option<i64>) = store.writer()?.query_row(
            "SELECT id,line_end,pagerank,summary_unit FROM symbols WHERE name='retainednode'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )?;
        assert_eq!(value, (retained_id, 2, 0.75, Some(caller.unit_id)));

        store.replace_file_content(
            None,
            fence,
            &[moved, input(Some(owner), &[5; 32], "newnode")],
            false,
            || Ok(()),
        )?;
        assert_eq!(store.get_meta("graph_pending")?.as_deref(), Some("1"));
        assert_eq!(
            store.writer()?.query_row(
                "SELECT COUNT(*) FROM fts_names WHERE kind='symbol' AND ref_id=?1",
                [old_id],
                |r| r.get::<_, i64>(0)
            )?,
            0
        );
        assert_eq!(
            store.writer()?.query_row(
                "SELECT dst_symbol FROM edges WHERE src_symbol=?1 AND dst_name='retainednode'",
                [caller.symbols[0].id],
                |r| r.get::<_, i64>(0)
            )?,
            retained_id
        );
        assert_eq!(
            store.writer()?.query_row(
                "SELECT COUNT(*) FROM symbols WHERE file_id=?1 AND unit_id IS NULL",
                [owner],
                |r| r.get::<_, i64>(0)
            )?,
            0
        );
        Ok(())
    }

    #[test]
    fn failed_file_replacement_retains_the_complete_previous_snapshot()
    -> Result<(), Box<dyn StdError>> {
        let file = tempfile::NamedTempFile::new()?;
        let store = open(file.path(), StoreLimits::default())?;
        let file_id = store.upsert_file(FileUpsert {
            path: "atomic.rs",
            blake3: &[1; 32],
            size: 1,
            mtime_ns: 1,
            lang: Some("rust"),
            kind: FileKind::Code,
            git_blob: None,
            max_level: 2,
            treatment: "default",
            indexed_at: Some(1),
            needs_vlm: false,
        })?;
        store.writer()?.execute(
            "UPDATE files SET l1_desired=?1 WHERE id=?2",
            params![[7_u8; 32].as_slice(), file_id],
        )?;
        let fence = crate::store::l1::Fence {
            generation: 0,
            file_id,
            path: "atomic.rs",
            content_hash: &[1; 32],
            fingerprint: &[7; 32],
        };
        store.replace_file_content(
            None,
            fence,
            &[input(Some(file_id), &[2; 32], "original")],
            true,
            || Ok(()),
        )?;
        let unit_id = store.query_fts("original", 10)?[0].unit_id;
        vector_table(&store)?;
        store.insert_vector(VectorInsert {
            table_slug: "occurrence_test",
            unit_id,
            ctx_hash: &[3; 32],
            embedding_hash: &[2; 32],
            embedding: &[1.0, 0.0],
        })?;
        store.mark_all_l1_pending()?;
        let before = snapshot(&store)?;
        let mut invalid = input(Some(file_id), &[5; 32], "invalid");
        invalid.edges[0].dst_symbol_index = None;
        invalid.edges[0].dst_symbol = Some(-1); // fail after writing the first new occurrence
        assert!(matches!(
            store.replace_file_content(
                None,
                fence,
                &[input(Some(file_id), &[4; 32], "replacement"), invalid,],
                false,
                || Ok(())
            ),
            Err(Error::Sqlite(_))
        ));
        assert_eq!(snapshot(&store)?, before);
        assert_eq!(store.query_fts("original", 10)?.len(), 1);
        assert!(store.query_fts("replacement OR invalid", 10)?.is_empty());
        assert!(matches!(
            store.replace_file_content(
                None,
                crate::store::l1::Fence {
                    content_hash: &[9; 32],
                    ..fence
                },
                &[],
                false,
                || Ok(())
            ),
            Err(Error::FileIndexSuperseded { .. })
        ));
        assert!(matches!(
            store.replace_file_content(
                None,
                fence,
                &[
                    input(Some(file_id), &[4; 32], "replacement"),
                    input(Some(file_id), &[4; 32], "replacement"),
                ],
                false,
                || Ok(())
            ),
            Err(Error::DuplicateOccurrence { .. })
        ));
        assert_eq!(snapshot(&store)?, before);
        Ok(())
    }

    #[test]
    fn equal_payloads_have_separate_context_vectors_and_count_toward_the_cap()
    -> Result<(), Box<dyn StdError>> {
        let file = tempfile::NamedTempFile::new()?;
        let store = open(
            file.path(),
            StoreLimits {
                max_l2_units: 1,
                ..StoreLimits::default()
            },
        )?;
        let first = input(None, &[1; 32], "shared").unit;
        let second = UnitInsert {
            source_key: Some("module:other"),
            ..first
        };
        let first_id = store.insert_unit(first)?;
        let second_id = store.insert_unit(second)?;
        assert_ne!(first_id, second_id);
        vector_table(&store)?;
        let first_vector = VectorInsert {
            table_slug: "occurrence_test",
            unit_id: first_id,
            ctx_hash: &[2; 32],
            embedding_hash: &[1; 32],
            embedding: &[1.0, 0.0],
        };
        let second_vector = VectorInsert {
            unit_id: second_id,
            ctx_hash: &[3; 32],
            embedding: &[0.0, 1.0],
            ..first_vector
        };
        store.insert_vector(first_vector)?;
        assert!(matches!(
            store.insert_vector(second_vector),
            Err(Error::VectorCapReached {
                current: 1,
                max_l2_units: 1
            })
        ));
        assert_eq!(store.count_vectors()?, 1);
        store.delete_vector("occurrence_test", first_id)?;
        store.insert_vector(second_vector)?;
        let (id, context): (i64, Vec<u8>) = store.writer()?.query_row(
            "SELECT unit_id, ctx_hash FROM vec_occurrence_test",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        assert_eq!((id, context), (second_id, vec![3; 32]));
        assert_eq!(
            store
                .writer()?
                .query_row("SELECT count(*) FROM unit_contents", [], |r| r
                    .get::<_, i64>(0))?,
            1
        );
        Ok(())
    }

    #[test]
    fn failed_occurrence_migration_rolls_back_and_restores_foreign_keys()
    -> Result<(), Box<dyn StdError>> {
        super::super::register_vector_extension()?;
        let connection = rusqlite::Connection::open_in_memory()?;
        for migration in super::super::MIGRATIONS.iter().take(4) {
            connection.execute_batch(migration.sql)?;
        }
        connection.execute_batch(
            "PRAGMA foreign_keys=OFF;
            INSERT INTO meta VALUES ('schema_version','4');
            INSERT INTO units(id,hash,file_id,kind,header,text,tokens,created_at,updated_at)
            VALUES (1,zeroblob(32),999,'chunk','orphan','original',1,1,1);
            PRAGMA foreign_keys=ON;",
        )?;
        let store = Store {
            database_authority: None,
            db_max_bytes: StoreLimits::default().db_max_bytes,
            writer: std::sync::Mutex::new(connection),
            readers: Vec::new(),
            next_reader: std::sync::atomic::AtomicUsize::new(0),
            wal_max_bytes: StoreLimits::default().wal_max_bytes,
            max_l2_units: StoreLimits::default().max_l2_units,
            max_l3_units: StoreLimits::default().max_l3_units,
        };
        assert!(matches!(
            super::super::migrate(&store),
            Err(Error::MigrationForeignKeys)
        ));
        let connection = store.writer()?;
        assert_eq!(
            connection.query_row(
                "SELECT value FROM meta WHERE key='schema_version'",
                [],
                |r| r.get::<_, String>(0)
            )?,
            "4"
        );
        assert_eq!(
            connection.query_row("SELECT text FROM units WHERE id=1", [], |r| r
                .get::<_, String>(0))?,
            "original"
        );
        assert!(
            !connection
                .prepare("SELECT name FROM sqlite_schema WHERE name='unit_contents'")?
                .exists([])?
        );
        assert!(connection.pragma_query_value(None, "foreign_keys", |r| r.get::<_, bool>(0))?);
        Ok(())
    }
}
