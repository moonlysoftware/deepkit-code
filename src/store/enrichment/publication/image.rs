//! Atomic L0 image parent and L3 summary publication; v1 §5 and §6.8.
use super::{
    Error, FtsColumns, Owner, Publication, Source, Store, UnitInsert, UnitKind, occurrences,
};
use rusqlite::{OptionalExtension, Transaction, params};

/// Prepared image output and already verified caption provenance.
/// Worker source/lease/privacy fences must run before entering this transaction.
pub(crate) struct ImagePublication<'a> {
    pub(crate) file_id: i64,
    pub(crate) file_hash: &'a [u8; 32],
    pub(crate) desired_hash: &'a [u8; 32],
    pub(crate) text: &'a str,
    pub(crate) sources: &'a [Source],
    pub(crate) max_units_per_file: u64,
    pub(crate) now: i64,
}

/// Path-independent image-parent content identity; occurrence identity adds its path.
pub(crate) fn image_parent_hash(file_hash: &[u8; 32]) -> [u8; 32] {
    // The image itself is this parent's content; it has no invented text body.
    *file_hash
}

/// Locates only the current image parent's canonical occurrence, without writes.
pub(crate) fn image_parent(
    tx: &Transaction<'_>,
    file: i64,
    file_hash: &[u8; 32],
) -> Result<Option<i64>, Error> {
    Ok(tx.query_row(
        "SELECT u.id FROM units u JOIN files f ON f.id=u.file_id WHERE f.id=?1 AND f.kind='image'
         AND f.blake3=?2 AND u.kind='image' AND u.ladder_level=0 AND u.parent_unit IS NULL AND u.content_hash=?3 AND u.source_key=f.path AND u.duplicate_ordinal=0",
        params![file,file_hash.as_slice(),image_parent_hash(file_hash).as_slice()], |r|r.get(0),
    ).optional()?)
}

impl Store {
    /// Publishes the image parent and its owned description together. A failure
    /// must roll back the caller's transaction, retaining all earlier output.
    pub(crate) fn publish_image_summary_in(
        &self,
        tx: &Transaction<'_>,
        input: ImagePublication<'_>,
    ) -> Result<i64, Error> {
        if input.now < 0 || input.text.trim().is_empty() {
            return Err(Error::L1InputsChanged);
        }
        let path: Option<String> = tx
            .query_row(
                "SELECT path FROM files WHERE id=?1 AND kind='image' AND blake3=?2 AND max_level>=3
             AND l1_pending=0 AND l1_fingerprint IS NOT NULL AND l1_fingerprint=l1_desired
             AND l1_deferred_reason IS NULL",
                params![input.file_id, input.file_hash.as_slice()],
                |r| r.get(0),
            )
            .optional()?;
        let path = path.ok_or(Error::L1InputsChanged)?;
        let existing = image_parent(tx, input.file_id, input.file_hash)?;
        let owner = Owner::Image(input.file_id);
        let old = existing
            .map(|parent| owner.current(tx, parent))
            .transpose()?
            .flatten()
            .flatten();
        if old.is_none() {
            self.enrichment_status_in(tx)?.admit_new()?;
        }
        if existing.is_none() {
            let count: i64 = tx.query_row(
                "SELECT count(*) FROM units WHERE file_id=?1",
                [input.file_id],
                |r| r.get(0),
            )?;
            if count < 0 || count as u64 >= input.max_units_per_file {
                return Err(Error::FileUnitCapReached {
                    current: count.max(0) as u64,
                    max_units_per_file: input.max_units_per_file,
                });
            }
        }
        let hash = image_parent_hash(input.file_hash);
        let parent = occurrences::insert_unit_in_transaction(
            tx,
            UnitInsert {
                hash: &hash,
                source_key: None,
                duplicate_ordinal: 0,
                file_id: Some(input.file_id),
                kind: UnitKind::Image,
                byte_start: None,
                byte_end: None,
                line_start: None,
                line_end: None,
                scope_path: None,
                header: &path,
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
        // The anchor is ownership metadata, not another lexical document.
        tx.execute("DELETE FROM fts_units WHERE rowid=?1", [parent])?;
        tx.execute("INSERT OR IGNORE INTO unit_sources(unit_id,source_kind,source_id,role) VALUES(?1,'file',?2,'content')",params![parent,input.file_id])?;
        let mut sources = Vec::with_capacity(input.sources.len() + 1);
        sources.push(Source::File(input.file_id));
        sources.extend_from_slice(input.sources);
        let summary = self.publish_summary_in(
            tx,
            Publication {
                owner,
                parent_unit: parent,
                parent_hash: &hash,
                desired_hash: input.desired_hash,
                text: input.text,
                sources: &sources,
                now: input.now,
            },
        )?;
        Ok(summary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{self, StoreLimits};

    fn fixture() -> Store {
        let store = store::open(
            ":memory:",
            StoreLimits {
                max_l3_units: 1,
                ..Default::default()
            },
        )
        .unwrap();
        for (id, path, kind) in [
            (1, "diagram.png", "image"),
            (2, "README.md", "doc"),
            (3, "other.png", "image"),
        ] {
            store.writer().unwrap().execute("INSERT INTO files(id,path,blake3,size,mtime_ns,kind,max_level,treatment,l1_pending,l1_desired,l1_fingerprint)
                VALUES(?1,?2,?3,10,0,?4,3,'fixture',0,zeroblob(32),zeroblob(32))",
                params![id,path,[id as u8;32].as_slice(),kind]).unwrap();
        }
        store
    }
    fn input(text: &str) -> ImagePublication<'_> {
        ImagePublication {
            file_id: 1,
            file_hash: &[1; 32],
            desired_hash: &[9; 32],
            text,
            sources: &[Source::File(2)],
            max_units_per_file: 1,
            now: 10,
        }
    }
    #[test]
    fn l0_image_parent_summary_and_provenance_commit_together() {
        let store = fixture();
        let summary = {
            let mut db = store.writer().unwrap();
            let tx = db.transaction().unwrap();
            let summary = store
                .publish_image_summary_in(&tx, input("Network request diagram"))
                .unwrap();
            let parent = image_parent(&tx, 1, &[1; 32]).unwrap().unwrap();
            assert_eq!(
                Owner::Image(1)
                    .matching_summary(&tx, parent, &[1; 32], &[9; 32])
                    .unwrap(),
                Some(summary)
            );
            assert_eq!(
                tx.query_row(
                    "SELECT count(*) FROM fts_units WHERE rowid=?1",
                    [parent],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
                0
            );
            assert_eq!(
                tx.query_row(
                    "SELECT count(*) FROM unit_sources WHERE unit_id=?1 AND source_kind='file'",
                    [summary],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
                2
            );
            assert_eq!(
                tx.query_row(
                    "SELECT ladder_level FROM units WHERE id=?1",
                    [parent],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
                0
            );
            tx.commit().unwrap();
            summary
        };
        assert_eq!(store.summary_generation().unwrap(), 1);
        assert!(!store.query_fts("Network", 10).unwrap().is_empty());
        {
            let mut db = store.writer().unwrap();
            let tx = db.transaction().unwrap();
            assert_eq!(
                store
                    .publish_image_summary_in(&tx, input("Network request diagram"))
                    .unwrap(),
                summary
            );
            assert_eq!(super::super::summary_generation_in(&tx).unwrap(), 1);
            tx.commit().unwrap();
        }
        store.clear_file_content(1).unwrap();
        assert!(store.query_fts("Network", 10).unwrap().is_empty());
        assert_eq!(
            store
                .writer()
                .unwrap()
                .query_row("SELECT count(*) FROM units", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            0
        );
    }
    #[test]
    fn image_replacement_keeps_its_slot_and_cannot_borrow_another_images_slot() {
        let mut store = fixture();
        {
            let mut db = store.writer().unwrap();
            let tx = db.transaction().unwrap();
            store
                .publish_image_summary_in(&tx, input("Original description"))
                .unwrap();
            tx.commit().unwrap();
        }
        store.max_l3_units = 0;
        {
            let mut db = store.writer().unwrap();
            let tx = db.transaction().unwrap();
            store
                .publish_image_summary_in(
                    &tx,
                    ImagePublication {
                        desired_hash: &[8; 32],
                        max_units_per_file: 0,
                        ..input("Updated description")
                    },
                )
                .unwrap();
            tx.commit().unwrap();
        }
        assert!(store.query_fts("Original", 10).unwrap().is_empty());
        assert!(!store.query_fts("Updated", 10).unwrap().is_empty());
        {
            let mut db = store.writer().unwrap();
            let tx = db.transaction().unwrap();
            assert!(matches!(
                store.publish_image_summary_in(
                    &tx,
                    ImagePublication {
                        file_id: 3,
                        file_hash: &[3; 32],
                        ..input("Other image")
                    }
                ),
                Err(Error::SummaryCapReached { .. })
            ));
        }
        assert_eq!(store.enrichment_status().unwrap().summaries, 1);
        assert_eq!(
            store
                .writer()
                .unwrap()
                .query_row("SELECT count(*) FROM units WHERE file_id=3", [], |r| r
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }

    #[test]
    fn l1_configuration_refresh_retains_image_output_until_its_source_changes() {
        let store = fixture();
        let summary = {
            let mut db = store.writer().unwrap();
            let tx = db.transaction().unwrap();
            let id = store
                .publish_image_summary_in(&tx, input("Retained image description"))
                .unwrap();
            tx.commit().unwrap();
            id
        };
        let generation = store.l1_generation().unwrap();
        for (file_hash, fingerprint, retained) in
            [([1; 32], [4; 32], true), ([5; 32], [6; 32], false)]
        {
            store
                .writer()
                .unwrap()
                .execute(
                    "UPDATE files SET blake3=?1,l1_desired=?2,l1_pending=1 WHERE id=1",
                    params![file_hash.as_slice(), fingerprint.as_slice()],
                )
                .unwrap();
            store
                .replace_file_content(
                    None,
                    crate::store::l1::Fence {
                        generation,
                        file_id: 1,
                        path: "diagram.png",
                        content_hash: &file_hash,
                        fingerprint: &fingerprint,
                    },
                    &[],
                    crate::store::l1::Completion::default(),
                    || Ok(()),
                )
                .unwrap();
            let present = store
                .writer()
                .unwrap()
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM units WHERE id=?1)",
                    [summary],
                    |r| r.get::<_, bool>(0),
                )
                .unwrap();
            assert_eq!(present, retained);
        }
    }

    #[test]
    fn image_queue_claims_credit_only_the_matching_file_output() {
        use crate::queue;
        let store = fixture();
        let enqueue = |file, hash: &[u8; 32], desired: &[u8; 32], priority| {
            queue::enqueue(
                &store,
                &queue::Enqueue {
                    kind: queue::Kind::DescribeImage,
                    target: queue::Target::File,
                    target_id: file,
                    target_hash: hash,
                    desired_hash: desired,
                    provider: "images",
                    priority,
                    tokens: 1,
                },
                0,
            )
        };
        let growth = enqueue(3, &[3; 32], &[5; 32], 99.0).unwrap();
        {
            let mut db = store.writer().unwrap();
            let tx = db.transaction().unwrap();
            store
                .publish_image_summary_in(&tx, input("Existing description"))
                .unwrap();
            tx.commit().unwrap();
        }
        let replacement = enqueue(1, &[1; 32], &[6; 32], 0.1).unwrap();
        assert!(matches!(
            enqueue(1, &[2; 32], &[7; 32], 1.0),
            Err(queue::Error::Store(Error::SummaryCapReached { .. }))
        ));
        assert!(matches!(
            enqueue(3, &[3; 32], &[8; 32], 1.0),
            Err(queue::Error::Store(Error::SummaryCapReached { .. }))
        ));
        let owner = queue::Owner::new(&store).unwrap();
        let leases = queue::claim(&store, "images", &owner, 1, 10, 10).unwrap();
        assert_eq!(leases.len(), 1);
        assert_eq!(leases[0].id, replacement);
        assert_ne!(leases[0].id, growth);
        let status = serde_json::to_value(queue::status::snapshot(&store, 2).unwrap()).unwrap();
        assert_eq!(status["enrichment_claims"]["growth_leases"], 0);
    }
    #[test]
    fn image_source_unit_cap_and_failed_publication_leave_no_anchor() {
        let store = fixture();
        for (hash, cap) in [([2; 32], 1), ([1; 32], 0)] {
            let mut db = store.writer().unwrap();
            let tx = db.transaction().unwrap();
            let result = store.publish_image_summary_in(
                &tx,
                ImagePublication {
                    file_hash: &hash,
                    max_units_per_file: cap,
                    ..input("Must not publish")
                },
            );
            if cap == 0 {
                assert!(matches!(result, Err(Error::FileUnitCapReached { .. })));
            } else {
                assert!(matches!(result, Err(Error::L1InputsChanged)));
            }
        }
        {
            let mut db = store.writer().unwrap();
            let tx = db.transaction().unwrap();
            assert!(
                store
                    .publish_image_summary_in(
                        &tx,
                        ImagePublication {
                            sources: &[Source::File(999)],
                            ..input("Rollback anchor too")
                        }
                    )
                    .is_err()
            );
        }
        assert_eq!(
            store
                .writer()
                .unwrap()
                .query_row("SELECT count(*) FROM units", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(store.summary_generation().unwrap(), 0);
    }
}
