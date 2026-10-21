//! Shared L3 admission for scheduling and atomic publication; v1.md §3.5.
pub(crate) mod publication;
use super::{Error, Store};
use rusqlite::Connection;

/// Current summary/page coverage and the earlier L3 database soft threshold.
#[derive(Debug, serde::Serialize)]
pub(crate) struct Status {
    pub(crate) allocated_bytes: u64,
    pub(crate) limit_bytes: u64,
    pub(crate) summaries: u64,
    pub(crate) page_sections: u64,
    pub(crate) l3_units: u64,
    pub(crate) max_l3_units: u64,
    pub(crate) paused: bool,
    pub(crate) reason: Option<&'static str>,
}
impl Status {
    /// Refuses new L3 work without evicting existing results or touching leases.
    pub(crate) fn admit_new(&self) -> Result<(), Error> {
        if self.l3_units >= self.max_l3_units {
            return Err(Error::SummaryCapReached {
                current: self.l3_units,
                max_l3_units: self.max_l3_units,
            });
        }
        if u128::from(self.allocated_bytes) * 100 >= u128::from(self.limit_bytes) * 85 {
            return Err(Error::EnrichmentDatabasePaused {
                allocated: self.allocated_bytes,
                cap: self.limit_bytes,
            });
        }
        Ok(())
    }
    /// Allows retained coverage and same-count replacements after a cap is lowered.
    /// Call inside the same admitted transaction, before committing publication.
    pub(crate) fn check_growth(&self, after: &Self) -> Result<(), Error> {
        if after.l3_units <= self.l3_units {
            return Ok(());
        }
        self.admit_new()?;
        if after.l3_units > self.max_l3_units {
            return Err(Error::SummaryCapReached {
                current: after.l3_units,
                max_l3_units: self.max_l3_units,
            });
        }
        Ok(())
    }
    /// Shows the documented ladder pause from current storage, without writes.
    pub(crate) fn summary(&self) -> String {
        let percent = if self.limit_bytes == 0 {
            "n/a".to_owned()
        } else {
            format!(
                "{}%",
                u128::from(self.allocated_bytes) * 100 / u128::from(self.limit_bytes)
            )
        };
        format!(
            "footprint {} / {} ({}): L3 {}; units {} / {} ({} summaries, {} page sections){}",
            self.allocated_bytes,
            self.limit_bytes,
            percent,
            if self.paused { "paused" } else { "admissible" },
            self.l3_units,
            self.max_l3_units,
            self.summaries,
            self.page_sections,
            self.reason.map(|r| format!(" ({r})")).unwrap_or_default()
        )
    }
}
impl Store {
    /// Reads summary and footprint admission from one coherent snapshot.
    pub(crate) fn enrichment_status(&self) -> Result<Status, Error> {
        self.read_snapshot(|tx| self.enrichment_status_in(tx))?
    }
    /// Shares admission with callers already holding a publication/claim writer.
    pub(crate) fn enrichment_status_in(&self, db: &Connection) -> Result<Status, Error> {
        let db_pages: u32 = db.query_row("PRAGMA page_count", [], |r| r.get(0))?;
        let size: u32 = db.query_row("PRAGMA page_size", [], |r| r.get(0))?;
        let (count, pages): (i64, i64) = db.query_row(
            "SELECT (SELECT count(*) FROM units WHERE kind='summary'),
                    (SELECT count(*) FROM units WHERE kind='doc_section' AND ladder_level=3)",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        let summaries =
            u64::try_from(count).map_err(|_| rusqlite::Error::IntegralValueOutOfRange(0, count))?;
        let page_sections =
            u64::try_from(pages).map_err(|_| rusqlite::Error::IntegralValueOutOfRange(1, pages))?;
        let l3_units = summaries
            .checked_add(page_sections)
            .ok_or(Error::L1InputsChanged)?;
        let allocated_bytes = u64::from(db_pages) * u64::from(size);
        let reason = if l3_units >= self.max_l3_units {
            Some("max_l3_units")
        } else if u128::from(allocated_bytes) * 100 >= u128::from(self.db_max_bytes) * 85 {
            Some("db_max_bytes 85 percent")
        } else {
            None
        };
        Ok(Status {
            allocated_bytes,
            limit_bytes: self.db_max_bytes,
            summaries,
            page_sections,
            l3_units,
            max_l3_units: self.max_l3_units,
            paused: reason.is_some(),
            reason,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        queue,
        store::{self, FtsColumns, StoreLimits, UnitInsert, UnitKind},
    };
    use std::error::Error as StdError;
    fn summary(hash: &[u8; 32]) -> UnitInsert<'_> {
        UnitInsert {
            hash,
            source_key: Some("fixture:summary"),
            duplicate_ordinal: 0,
            file_id: None,
            kind: UnitKind::Summary,
            byte_start: None,
            byte_end: None,
            line_start: None,
            line_end: None,
            scope_path: None,
            header: "summary",
            text: "summary body",
            tokens: 3,
            ladder_level: 3,
            parent_unit: None,
            created_at: 0,
            updated_at: 0,
            fts: FtsColumns {
                header: "summary",
                idents: "",
                body: "summary body",
                prose: "",
            },
        }
    }
    fn job(id: i64, hash: &[u8; 32], priority: f64) -> queue::Enqueue<'_> {
        queue::Enqueue {
            kind: queue::Kind::SummarizeModule,
            target: queue::Target::Unit,
            target_id: id,
            target_hash: &[7; 32],
            desired_hash: hash,
            provider: "fixture",
            priority,
            tokens: 1,
        }
    }
    #[test]
    fn generated_page_sections_share_summary_capacity_without_counting_l1_or_anchors() {
        let store = store::open(
            ":memory:",
            StoreLimits {
                max_l3_units: 2,
                ..Default::default()
            },
        )
        .unwrap();
        store.insert_unit(summary(&[1; 32])).unwrap();
        let mut page = summary(&[2; 32]);
        page.kind = UnitKind::DocSection;
        let page_id = store.insert_unit(page).unwrap();
        let mut digital = summary(&[3; 32]);
        digital.kind = UnitKind::DocSection;
        digital.ladder_level = 1;
        store.insert_unit(digital).unwrap();
        let mut anchor = summary(&[4; 32]);
        anchor.kind = UnitKind::Image;
        anchor.ladder_level = 0;
        store.insert_unit(anchor).unwrap();
        let status = store.enrichment_status().unwrap();
        assert_eq!(
            (status.summaries, status.page_sections, status.l3_units),
            (1, 1, 2)
        );
        assert!(status.paused);
        for kind in [UnitKind::Summary, UnitKind::DocSection] {
            let mut input = summary(&[5; 32]);
            input.kind = kind;
            assert!(matches!(
                store.insert_unit(input),
                Err(Error::SummaryCapReached { .. })
            ));
        }
        let mut same = summary(&[2; 32]);
        same.kind = UnitKind::DocSection;
        assert_eq!(store.insert_unit(same).unwrap(), page_id);
        assert_eq!(store.enrichment_status().unwrap().l3_units, 2);
    }

    #[test]
    fn count_cap_preserves_duplicate_and_lowered_limit_coverage() -> Result<(), Box<dyn StdError>> {
        let mut store = store::open(
            ":memory:",
            StoreLimits {
                max_l3_units: 1,
                ..Default::default()
            },
        )?;
        let id = store.insert_unit(summary(&[1; 32]))?;
        assert_eq!(store.insert_unit(summary(&[1; 32]))?, id);
        assert!(matches!(
            store.insert_unit(summary(&[2; 32])),
            Err(Error::SummaryCapReached { .. })
        ));
        assert_eq!(store.enrichment_status()?.summaries, 1);
        store.max_l3_units = 0;
        assert_eq!(store.insert_unit(summary(&[1; 32]))?, id);
        store.insert_unit(UnitInsert {
            kind: UnitKind::Chunk,
            ladder_level: 1,
            ..summary(&[3; 32])
        })?;
        assert_eq!(store.enrichment_status()?.summaries, 1);
        assert_eq!(store.enrichment_status()?.reason, Some("max_l3_units"));
        assert!(store.enrichment_status()?.summary().contains("L3 paused"));
        Ok(())
    }
    #[test]
    fn eighty_five_percent_pauses_enrichment_but_keeps_embedding_jobs()
    -> Result<(), Box<dyn StdError>> {
        let mut store = store::open(":memory:", StoreLimits::default())?;
        let bytes = store.enrichment_status()?.allocated_bytes;
        store.db_max_bytes = bytes * 100 / 85;
        assert!(matches!(
            queue::enqueue(&store, &job(1, &[1; 32], 1.0), 0),
            Err(queue::Error::Store(Error::EnrichmentDatabasePaused { .. }))
        ));
        assert!(matches!(
            store.insert_unit(summary(&[1; 32])),
            Err(Error::EnrichmentDatabasePaused { .. })
        ));
        let mut embed = job(2, &[2; 32], 1.0);
        embed.kind = queue::Kind::Embed;
        queue::enqueue(&store, &embed, 0)?;
        let owner = queue::Owner::new(&store)?;
        assert_eq!(queue::claim(&store, "fixture", &owner, 0, 1, 1)?.len(), 1);
        assert_eq!(store.enrichment_status()?.summaries, 0);
        for (allocated, allowed) in [(84, true), (85, false), (86, false)] {
            let status = Status {
                allocated_bytes: allocated,
                limit_bytes: 100,
                summaries: 0,
                page_sections: 0,
                l3_units: 0,
                max_l3_units: 1,
                paused: !allowed,
                reason: None,
            };
            assert_eq!(status.admit_new().is_ok(), allowed);
        }
        Ok(())
    }
    #[test]
    fn ranked_jobs_pause_resume_and_over_cap_publication_rolls_back_atomically()
    -> Result<(), Box<dyn StdError>> {
        let mut store = store::open(
            ":memory:",
            StoreLimits {
                max_l3_units: 1,
                ..Default::default()
            },
        )?;
        let low = queue::enqueue(&store, &job(1, &[1; 32], 0.1), 0)?;
        let high = queue::enqueue(&store, &job(2, &[2; 32], 0.9), 0)?;
        let owner = queue::Owner::new(&store)?;
        let lease = queue::claim(&store, "fixture", &owner, 0, 1, 1)?.remove(0);
        assert_eq!(lease.id, high);
        queue::complete(
            &store,
            &lease,
            || 1,
            |_, _| {
                Ok(Some(queue::CurrentInput {
                    target_hash: [7; 32],
                    desired_hash: [2; 32],
                }))
            },
            |tx| {
                store::occurrences::insert_unit_in_transaction(tx, summary(&[2; 32]), None)?;
                Ok(())
            },
        )?;
        assert!(queue::claim(&store, "fixture", &owner, 2, 1, 1)?.is_empty());
        assert_eq!(queue::enqueue(&store, &job(1, &[1; 32], 0.1), 2)?, low);
        assert!(matches!(
            queue::enqueue(&store, &job(3, &[3; 32], 0.5), 2),
            Err(queue::Error::Store(Error::SummaryCapReached { .. }))
        ));
        // A blocked high-priority L3 must not starve a same-provider embed job.
        let mut embed = job(4, &[4; 32], 0.01);
        embed.kind = queue::Kind::Embed;
        let embedded = queue::enqueue(&store, &embed, 2)?;
        assert_eq!(
            queue::claim(&store, "fixture", &owner, 2, 1, 1)?
                .remove(0)
                .id,
            embedded
        );
        store.max_l3_units = 2;
        let lease = queue::claim(&store, "fixture", &owner, 3, 1, 1)?.remove(0);
        assert_eq!(lease.id, low);
        let result = queue::complete(
            &store,
            &lease,
            || 4,
            |_, _| {
                Ok(Some(queue::CurrentInput {
                    target_hash: [7; 32],
                    desired_hash: [1; 32],
                }))
            },
            |tx| {
                store::occurrences::insert_unit_in_transaction(tx, summary(&[1; 32]), None)?;
                store::occurrences::insert_unit_in_transaction(tx, summary(&[3; 32]), None)?;
                Ok(())
            },
        );
        assert!(matches!(
            result,
            Err(queue::Error::Store(Error::SummaryCapReached { .. }))
        ));
        assert_eq!(store.enrichment_status()?.summaries, 1);
        assert_eq!(
            store.writer()?.query_row(
                "SELECT count(*) FROM fts_units WHERE fts_units MATCH 'summary'",
                [],
                |r| r.get::<_, i64>(0)
            )?,
            1
        );
        queue::retry::schedule(
            &store,
            &lease,
            queue::retry::Outcome::Capacity(queue::capacity::Reason::Summaries),
            || 5,
        )?;
        let status = queue::status::snapshot(&store, 6)?;
        assert!(serde_json::to_string(&status)?.contains("cap_summaries"));
        assert_eq!(queue::capacity::resume(&store, "fixture", || 7)?, 1);
        let lease = queue::claim(&store, "fixture", &owner, 7, 1, 1)?.remove(0);
        queue::complete(
            &store,
            &lease,
            || 8,
            |_, _| {
                Ok(Some(queue::CurrentInput {
                    target_hash: [7; 32],
                    desired_hash: [1; 32],
                }))
            },
            |tx| {
                store::occurrences::insert_unit_in_transaction(tx, summary(&[1; 32]), None)?;
                Ok(())
            },
        )?;
        assert_eq!(store.enrichment_status()?.summaries, 2);
        Ok(())
    }
    #[test]
    fn independent_writers_share_the_last_summary_slot() -> Result<(), Box<dyn StdError>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("code.db");
        let limits = StoreLimits {
            max_l3_units: 1,
            ..Default::default()
        };
        let store = store::open(&path, limits)?;
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let mut threads = Vec::new();
        for id in [1, 2] {
            let other = store::open(&path, limits)?;
            let barrier = barrier.clone();
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                other.insert_unit(summary(&[id; 32]))
            }));
        }
        let outcomes: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(outcomes.iter().filter(|r| r.is_ok()).count(), 1);
        assert_eq!(
            outcomes
                .iter()
                .filter(|r| matches!(r, Err(Error::SummaryCapReached { .. })))
                .count(),
            1
        );
        assert_eq!(store.enrichment_status()?.summaries, 1);
        Ok(())
    }
    #[test]
    fn l3_claims_reserve_only_available_ranked_slots_across_connections_and_expiry()
    -> Result<(), Box<dyn StdError>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("code.db");
        let limits = StoreLimits {
            max_l3_units: 1,
            ..Default::default()
        };
        let mut store = store::open(&path, limits)?;
        let mut other = store::open(&path, limits)?;
        let low = queue::enqueue(&store, &job(1, &[1; 32], 0.1), 0)?;
        let high = queue::enqueue(&store, &job(2, &[2; 32], 0.9), 0)?;
        let mut second = job(3, &[3; 32], 0.01);
        second.provider = "second";
        let next = queue::enqueue(&other, &second, 0)?;
        let owner = queue::Owner::new(&store)?;
        let second_owner = queue::Owner::new(&other)?;
        let leases = queue::claim(&store, "fixture", &owner, 0, 8, 8)?;
        assert_eq!(leases.len(), 1);
        assert_eq!(leases[0].id, high);
        let snapshot = queue::status::snapshot(&other, 1)?;
        assert!(
            snapshot
                .summary()
                .contains("max_l3_units reserved by active leases")
        );
        let status = serde_json::to_value(snapshot)?;
        assert_eq!(status["enrichment_claims"]["active_leases"], 1);
        assert_eq!(status["enrichment_claims"]["available_slots"], 0);
        assert_eq!(status["enrichment_claims"]["paused"], true);
        assert!(queue::claim(&other, "second", &second_owner, 1, 8, 8)?.is_empty());
        queue::complete(
            &store,
            &leases[0],
            || 2,
            |_, _| {
                Ok(Some(queue::CurrentInput {
                    target_hash: [7; 32],
                    desired_hash: [2; 32],
                }))
            },
            |tx| {
                store::occurrences::insert_unit_in_transaction(tx, summary(&[2; 32]), None)?;
                Ok(())
            },
        )?;
        store.max_l3_units = 2;
        other.max_l3_units = 2;
        let low_lease = queue::claim(&store, "fixture", &owner, 3, 8, 8)?.remove(0);
        assert_eq!(low_lease.id, low);
        assert!(queue::claim(&other, "second", &second_owner, 4, 8, 8)?.is_empty());
        // The other provider need not mutate an expired foreign lease to reuse
        // its slot. That expired worker cannot publish late output.
        let resumed = queue::claim(&other, "second", &second_owner, 200, 8, 8)?;
        assert_eq!(resumed.len(), 1);
        assert_eq!(resumed[0].id, next);
        let late = queue::complete(
            &store,
            &low_lease,
            || 201,
            |_, _| {
                Ok(Some(queue::CurrentInput {
                    target_hash: [7; 32],
                    desired_hash: [1; 32],
                }))
            },
            |_| panic!("expired worker must not publish"),
        )?;
        assert_eq!(late, queue::Completion::LeaseLost);
        assert_eq!(store.enrichment_status()?.summaries, 1);
        Ok(())
    }
}
