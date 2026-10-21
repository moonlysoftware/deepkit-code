//! Complete generated-page ownership and section proofs; v1 §6.2/§6.8/§8.
use crate::store::Error;
use rusqlite::{OptionalExtension, Transaction, params};
mod publish;
pub(crate) use publish::Publication;

/// Locates the exact L0 anchor for a one-based page of the current PDF bytes.
/// Image-file descriptions have different ownership and cannot satisfy this proof.
pub(crate) fn parent(
    tx: &Transaction<'_>,
    file: i64,
    file_hash: &[u8; 32],
    page: u32,
) -> Result<Option<i64>, Error> {
    if page == 0 {
        return Ok(None);
    }
    Ok(tx
        .query_row(
            "SELECT p.id FROM units p JOIN files f ON f.id=p.file_id
         WHERE f.id=?1 AND f.kind='doc' AND f.blake3=?2 AND p.kind='image'
         AND p.ladder_level=0 AND p.parent_unit IS NULL AND p.content_hash=?2
         AND p.source_key=f.path AND p.duplicate_ordinal=?3",
            params![file, file_hash.as_slice(), page - 1],
            |r| r.get(0),
        )
        .optional()?)
}

/// Hashes the prepared generation and every stored section locator/body field.
/// File/page/config/prompt/wire identity is already part of the desired generation.
pub(crate) fn section_hash(
    desired: &[u8; 32],
    header: &str,
    scope: &str,
    start: i64,
    end: i64,
    text: &str,
) -> [u8; 32] {
    let mut proof = blake3::Hasher::new_derive_key("dkc generated PDF section v1");
    proof.update(desired);
    proof.update(&start.to_le_bytes());
    proof.update(&end.to_le_bytes());
    for value in [header, scope, text] {
        proof.update(&(value.len() as u64).to_le_bytes());
        proof.update(value.as_bytes());
    }
    *proof.finalize().as_bytes()
}

/// Ordered fixed-width section identities prove the entire generated page.
pub(crate) fn output_hash(sections: &[[u8; 32]]) -> [u8; 32] {
    let mut proof = blake3::Hasher::new_derive_key("dkc generated PDF page manifest v1");
    for hash in sections {
        proof.update(hash);
    }
    *proof.finalize().as_bytes()
}

/// Verifies complete owned output, not merely a done job or a surviving child.
/// Callers must separately re-prove current file bytes, policy and desired input.
pub(crate) fn matching(
    tx: &Transaction<'_>,
    file: i64,
    file_hash: &[u8; 32],
    page: u32,
    desired: &[u8; 32],
) -> Result<Option<i64>, Error> {
    let Some(parent) = parent(tx, file, file_hash, page)? else {
        return Ok(None);
    };
    let manifest: Option<(i64, Vec<u8>)> = tx
        .query_row(
            "SELECT sections,output_hash FROM page_publications
         WHERE file_id=?1 AND page=?2 AND parent_unit=?3 AND desired_hash=?4",
            params![file, page, parent, desired.as_slice()],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?;
    let Some((count, expected)) = manifest else {
        return Ok(None);
    };
    if count <= 0 {
        return Ok(None);
    }
    let mut query = tx.prepare(
        "SELECT u.content_hash,u.header,u.scope_path,u.line_start,u.line_end,c.text,
                u.duplicate_ordinal,u.kind,u.ladder_level,u.file_id,u.source_key=f.path
         FROM units u LEFT JOIN unit_contents c ON c.hash=u.content_hash
         LEFT JOIN files f ON f.id=u.file_id WHERE u.parent_unit=?1 ORDER BY u.duplicate_ordinal,u.id",
    )?;
    let mut rows = query.query([parent])?;
    let mut proof = blake3::Hasher::new_derive_key("dkc generated PDF page manifest v1");
    let mut observed = 0_i64;
    while let Some(row) = rows.next()? {
        let hash: Vec<u8> = row.get(0)?;
        let header: String = row.get(1)?;
        let scope: Option<String> = row.get(2)?;
        let start: Option<i64> = row.get(3)?;
        let end: Option<i64> = row.get(4)?;
        let text: Option<String> = row.get(5)?;
        let ordinal: i64 = row.get(6)?;
        let kind: String = row.get(7)?;
        let level: i64 = row.get(8)?;
        let owner: Option<i64> = row.get(9)?;
        let source_matches: Option<bool> = row.get(10)?;
        let (Some(start), Some(end), Some(text)) = (start, end, text) else {
            return Ok(None);
        };
        if observed >= count
            || ordinal != observed
            || kind != "doc_section"
            || level != 3
            || owner != Some(file)
            || source_matches != Some(true)
            || start <= 0
            || end < start
            || text.trim().is_empty()
            || hash.as_slice()
                != section_hash(
                    desired,
                    &header,
                    scope.as_deref().unwrap_or(""),
                    start,
                    end,
                    &text,
                )
        {
            return Ok(None);
        }
        proof.update(&hash);
        observed += 1;
    }
    Ok((observed == count && expected.as_slice() == proof.finalize().as_bytes()).then_some(parent))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{self, FtsColumns, Store, StoreLimits, UnitInsert, UnitKind};

    fn fixture() -> (Store, i64) {
        let store = store::open(":memory:", StoreLimits::default()).unwrap();
        store.writer().unwrap().execute(
            "INSERT INTO files(id,path,blake3,size,mtime_ns,kind,max_level,treatment)
             VALUES(1,'scan.pdf',?1,100,0,'doc',3,'fixture'),(2,'other.pdf',?1,100,0,'doc',3,'fixture')",
            [[1_u8; 32].as_slice()],
        ).unwrap();
        store.writer().unwrap().execute_batch("UPDATE files SET l1_pending=0,
            l1_fingerprint=zeroblob(32),l1_desired=zeroblob(32),needs_vlm=1,l1_deferred_reason='needs_vlm'").unwrap();
        let anchor = store
            .insert_unit(UnitInsert {
                hash: &[1; 32],
                source_key: None,
                duplicate_ordinal: 0,
                file_id: Some(1),
                kind: UnitKind::Image,
                byte_start: None,
                byte_end: None,
                line_start: None,
                line_end: None,
                scope_path: None,
                header: "page 1",
                text: "",
                tokens: 0,
                ladder_level: 0,
                parent_unit: None,
                created_at: 0,
                updated_at: 0,
                fts: FtsColumns {
                    header: "",
                    idents: "",
                    body: "",
                    prose: "",
                },
            })
            .unwrap();
        let mut hashes = Vec::new();
        for (ordinal, text) in ["First section", "Second section"].into_iter().enumerate() {
            let hash = section_hash(&[2; 32], "page 1", "Title › page 1", 1, 1, text);
            store
                .insert_unit(UnitInsert {
                    hash: &hash,
                    source_key: None,
                    duplicate_ordinal: ordinal as u32,
                    file_id: Some(1),
                    kind: UnitKind::DocSection,
                    byte_start: None,
                    byte_end: None,
                    line_start: Some(1),
                    line_end: Some(1),
                    scope_path: Some("Title › page 1"),
                    header: "page 1",
                    text,
                    tokens: 2,
                    ladder_level: 3,
                    parent_unit: Some(anchor),
                    created_at: 0,
                    updated_at: 0,
                    fts: FtsColumns {
                        header: "page 1",
                        idents: "",
                        body: "",
                        prose: text,
                    },
                })
                .unwrap();
            hashes.push(hash);
        }
        store.writer().unwrap().execute(
            "INSERT INTO page_publications(file_id,page,parent_unit,desired_hash,sections,output_hash)
             VALUES(1,1,?1,?2,2,?3)",
            params![anchor, [2_u8; 32].as_slice(), output_hash(&hashes).as_slice()],
        ).unwrap();
        (store, anchor)
    }

    #[test]
    fn page_publication_preserves_unchanged_sections_and_rolls_back_cap_or_manifest_failure() {
        use crate::treat::{PreparedUnit, TextOrigin};
        let (mut store, anchor) = fixture();
        store.max_l3_units = 2;
        let sections = ["First section", "Replaced section"].map(|text| PreparedUnit {
            origin: TextOrigin::File,
            kind: UnitKind::DocSection,
            scope_path: "Title › page 1".into(),
            header: "page 1".into(),
            text: text.into(),
            line_start: 1,
            line_end: 1,
            has_code_fence: false,
            idents: Vec::new(),
            truncated: false,
        });
        let first: i64 = store
            .writer()
            .unwrap()
            .query_row(
                "SELECT id FROM units WHERE parent_unit=?1 AND duplicate_ordinal=0",
                [anchor],
                |r| r.get(0),
            )
            .unwrap();
        let publish = |sections: &[PreparedUnit]| {
            let mut writer = store.writer().unwrap();
            let tx = writer.transaction().unwrap();
            let result = store.publish_page_in(
                &tx,
                Publication {
                    file: 1,
                    file_hash: &[1; 32],
                    page: 1,
                    desired: &[2; 32],
                    sections,
                    repo_name: "fixture",
                    tokenize: &Default::default(),
                    max_units_per_file: 100,
                    now: 1,
                },
            )?;
            tx.commit()?;
            Ok::<_, Error>(result)
        };
        assert_eq!(publish(&sections).unwrap(), anchor);
        assert_eq!(
            store
                .writer()
                .unwrap()
                .query_row(
                    "SELECT id FROM units WHERE parent_unit=?1 AND duplicate_ordinal=0",
                    [anchor],
                    |r| r.get::<_, i64>(0),
                )
                .unwrap(),
            first
        );
        assert_eq!(store.query_fts("Replaced", 10).unwrap().len(), 1);
        let generation = store.summary_generation().unwrap();
        assert_eq!(publish(&sections).unwrap(), anchor);
        assert_eq!(store.summary_generation().unwrap(), generation);
        let mut extra = sections.to_vec();
        extra.push(sections[1].clone());
        assert!(matches!(
            publish(&extra),
            Err(Error::SummaryCapReached { .. })
        ));
        assert_eq!(store.enrichment_status().unwrap().l3_units, 2);
        store.writer().unwrap().execute_batch("CREATE TRIGGER refuse_page_manifest BEFORE INSERT ON page_publications BEGIN SELECT RAISE(ABORT,'fixture rollback'); END;").unwrap();
        let mut changed = sections.to_vec();
        changed[1].text = "Unpublished marker".into();
        assert!(publish(&changed).is_err());
        assert_eq!(store.query_fts("Unpublished", 10).unwrap().len(), 0);
        assert_eq!(store.query_fts("Replaced", 10).unwrap().len(), 1);
        assert_eq!(store.summary_generation().unwrap(), generation);
        assert_eq!(
            store
                .read_snapshot(|tx| matching(tx, 1, &[1; 32], 1, &[2; 32]))
                .unwrap()
                .unwrap(),
            Some(anchor)
        );
    }

    #[test]
    fn new_page_publication_admits_anchor_and_sections_together_and_keeps_other_pages() {
        let (store, old_anchor) = fixture();
        store.delete_unit(old_anchor).unwrap();
        let publish = |page: u32, cap| {
            let sections = crate::treat::pdf::page_units(
                "# Ledger\n\nBlue widgets: 42.\n",
                (page - 1) as usize,
                Some("Manual"),
                &Default::default(),
                100,
            )
            .unwrap();
            assert_eq!(sections.len(), 1);
            let mut writer = store.writer().unwrap();
            let tx = writer.transaction().unwrap();
            let anchor = store.publish_page_in(
                &tx,
                Publication {
                    file: 1,
                    file_hash: &[1; 32],
                    page,
                    desired: &[page as u8; 32],
                    sections: &sections,
                    repo_name: "fixture",
                    tokenize: &Default::default(),
                    max_units_per_file: cap,
                    now: 1,
                },
            )?;
            tx.commit()?;
            Ok::<_, Error>(anchor)
        };
        assert!(matches!(
            publish(1, 1),
            Err(Error::FileUnitCapReached { .. })
        ));
        assert_eq!(store.enrichment_status().unwrap().l3_units, 0);
        let first = publish(1, 2).unwrap();
        assert!(matches!(
            publish(2, 3),
            Err(Error::FileUnitCapReached { .. })
        ));
        let second = publish(2, 4).unwrap();
        assert_ne!(first, second);
        assert_eq!(store.enrichment_status().unwrap().l3_units, 2);
        store
            .read_snapshot(|tx| {
                assert_eq!(matching(tx, 1, &[1; 32], 1, &[1; 32])?, Some(first));
                assert_eq!(matching(tx, 1, &[1; 32], 2, &[2; 32])?, Some(second));
                assert_eq!(
                    tx.query_row("SELECT count(*) FROM units WHERE file_id=1", [], |r| r
                        .get::<_, i64>(0))?,
                    4
                );
                assert_eq!(
                    tx.query_row(
                        "SELECT count(*) FROM fts_units WHERE rowid IN (?1,?2)",
                        params![first, second],
                        |r| r.get::<_, i64>(0)
                    )?,
                    0
                );
                assert_eq!(
                    tx.query_row("SELECT count(*) FROM units WHERE kind='summary'", [], |r| r
                        .get::<_, i64>(0))?,
                    0
                );
                Ok::<_, Error>(())
            })
            .unwrap()
            .unwrap();
        // Lowered limits retain an already-complete page without deleting peers.
        assert_eq!(publish(1, 0).unwrap(), first);
        assert_eq!(store.enrichment_status().unwrap().l3_units, 2);
    }

    #[test]
    fn complete_page_manifest_refuses_missing_changed_or_foreign_sections() {
        let (store, anchor) = fixture();
        let mut writer = store.writer().unwrap();
        let tx = writer.transaction().unwrap();
        assert_eq!(
            matching(&tx, 1, &[1; 32], 1, &[2; 32]).unwrap(),
            Some(anchor)
        );
        for sql in [
            "DELETE FROM units WHERE parent_unit IS NOT NULL AND duplicate_ordinal=1",
            "UPDATE units SET duplicate_ordinal=2 WHERE parent_unit IS NOT NULL AND duplicate_ordinal=1",
            "UPDATE units SET header='changed' WHERE parent_unit IS NOT NULL",
            "UPDATE units SET scope_path='changed' WHERE parent_unit IS NOT NULL",
            "UPDATE units SET line_start=0 WHERE parent_unit IS NOT NULL",
            "UPDATE units SET kind='summary' WHERE parent_unit IS NOT NULL",
            "UPDATE units SET ladder_level=1 WHERE parent_unit IS NOT NULL",
            "UPDATE units SET file_id=2,source_key='other.pdf' WHERE parent_unit IS NOT NULL",
            "UPDATE page_publications SET output_hash=zeroblob(32)",
            "DELETE FROM page_publications",
        ] {
            tx.execute_batch("SAVEPOINT mutation").unwrap();
            tx.execute_batch(sql).unwrap();
            assert_eq!(
                matching(&tx, 1, &[1; 32], 1, &[2; 32]).unwrap(),
                None,
                "{sql}"
            );
            tx.execute_batch("ROLLBACK TO mutation; RELEASE mutation")
                .unwrap();
        }
        assert_eq!(
            matching(&tx, 1, &[1; 32], 1, &[2; 32]).unwrap(),
            Some(anchor)
        );
        for (file, hash, page, desired) in [
            (2, [1; 32], 1, [2; 32]),
            (1, [9; 32], 1, [2; 32]),
            (1, [1; 32], 0, [2; 32]),
            (1, [1; 32], 2, [2; 32]),
            (1, [1; 32], 1, [9; 32]),
        ] {
            assert_eq!(matching(&tx, file, &hash, page, &desired).unwrap(), None);
        }
    }
}
