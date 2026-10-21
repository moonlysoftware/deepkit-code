//! Capacity pauses retain jobs and prior output; see `v1.md` §3.5 and §8.2.
use super::Error;
use crate::store::Store;
use rusqlite::{OptionalExtension, params};

#[cfg(test)]
mod page_tests;

/// Sentinel means recheck on an explicit provider pass, never a promised UTC date.
pub(super) const PAUSED: i64 = i64::MAX;
/// Legacy mistaken L2 threshold; recognized until an explicit run rechecks it.
pub(super) const LEGACY_DATABASE_REASON: &str =
    "database 85 percent soft cap; waiting for capacity";

/// New L3 growth pauses before L2 growth, preserving the specified ladder order.
#[derive(Clone, Copy)]
pub(crate) enum GrowthLevel {
    Dense,
    Enrichment,
}

/// Checks the soft growth threshold exactly, without floating-point boundary drift.
/// This is not a reservation or permission to exceed the hard DB/WAL caps.
pub(crate) fn database_growth_allowed(allocated: u64, cap: u64, level: GrowthLevel) -> bool {
    let percent = match level {
        GrowthLevel::Dense => 95_u128,
        GrowthLevel::Enrichment => 85,
    };
    u128::from(allocated) * 100 < u128::from(cap) * percent
}

// Match only current, owner-bound summaries; another symbol sharing a chunk
// cannot donate its slot. This predicate is evaluated under the claim writer.
pub(super) const REPLACEMENT: &str = "(
 EXISTS(SELECT 1 FROM symbols s JOIN units p ON p.id=s.unit_id JOIN units u ON u.id=s.summary_unit
 WHERE jobs.kind='enrich:describe_symbol' AND jobs.target_kind='symbol' AND s.id=jobs.target_id
 AND p.kind IN ('symbol','chunk') AND p.file_id=s.file_id AND u.kind='summary'
 AND u.parent_unit=p.id AND u.source_key='summary:symbol:'||s.id)
 OR EXISTS(SELECT 1 FROM prs r JOIN units p ON p.id=r.unit_id JOIN units u ON u.id=r.summary_unit
 WHERE jobs.kind='enrich:summarize_pr' AND jobs.target_kind='pr' AND r.id=jobs.target_id
 AND p.kind='pr' AND p.file_id IS NULL AND u.kind='summary' AND u.parent_unit=p.id
 AND u.source_key='summary:pr:'||r.id)
 OR EXISTS(SELECT 1 FROM dirs d JOIN units p ON p.id=d.module_unit JOIN units u ON u.id=d.summary_unit
 WHERE jobs.kind='enrich:summarize_module' AND jobs.target_kind='unit' AND d.module_unit=jobs.target_id
 AND p.kind='module' AND p.file_id IS NULL AND u.kind='summary' AND u.parent_unit=p.id
 AND u.source_key='summary:module:'||d.module_unit)
 OR EXISTS(SELECT 1 FROM files f JOIN units p ON p.file_id=f.id JOIN units u ON u.parent_unit=p.id
 WHERE jobs.kind='enrich:describe_image' AND jobs.target_kind='file' AND f.id=jobs.target_id
 AND f.kind='image' AND p.kind='image' AND p.ladder_level=0 AND p.parent_unit IS NULL AND p.duplicate_ordinal=0
 AND p.source_key=f.path AND f.blake3=jobs.target_hash AND p.content_hash=jobs.target_hash
 AND u.kind='summary' AND u.source_key='summary:image:'||f.id)
 OR EXISTS(SELECT 1 FROM files f JOIN units p ON p.file_id=f.id JOIN units u ON u.parent_unit=p.id
 WHERE jobs.kind='enrich:vlm_page' AND jobs.target_kind='file' AND f.id=jobs.target_id
 AND f.kind='doc' AND p.kind='image' AND p.ladder_level=0 AND p.parent_unit IS NULL
 AND jobs.page>0 AND p.duplicate_ordinal=jobs.page-1 AND p.source_key=f.path
 AND f.blake3=jobs.target_hash AND p.content_hash=jobs.target_hash
 AND u.kind='doc_section' AND u.ladder_level=3 AND u.file_id=f.id AND u.source_key=f.path))";

const PAGE_UNITS: &str = "(SELECT count(*) FROM files f JOIN units p ON p.file_id=f.id JOIN units u ON u.parent_unit=p.id
 WHERE jobs.target_kind='file' AND f.id=jobs.target_id AND f.kind='doc'
 AND p.kind='image' AND p.ladder_level=0 AND p.parent_unit IS NULL
 AND jobs.page>0 AND p.duplicate_ordinal=jobs.page-1 AND p.source_key=f.path
 AND f.blake3=jobs.target_hash AND p.content_hash=jobs.target_hash
 AND u.kind='doc_section' AND u.ladder_level=3 AND u.file_id=f.id AND u.source_key=f.path)";

/// Known output width minus retained ownership, evaluated in the claim snapshot.
pub(super) fn growth_expression() -> String {
    format!(
        "max(0,jobs.required_l3_units-CASE WHEN jobs.kind='enrich:vlm_page' THEN {PAGE_UNITS} WHEN {REPLACEMENT} THEN 1 ELSE 0 END)"
    )
}

/// Recovers an observed output requirement without altering the desired identity.
pub(super) fn required_output(
    db: &rusqlite::Connection,
    input: &super::Enqueue<'_>,
) -> Result<u32, Error> {
    Ok(db.query_row("SELECT required_l3_units FROM jobs WHERE kind=?1 AND target_kind=?2 AND target_id=?3 AND desired_hash=?4",
        params![input.kind.as_str(), input.target.as_str(), input.target_id, input.desired_hash.as_slice()], |r| r.get(0)).optional()?.unwrap_or(1))
}

/// Measures growth for a prepared job before its row has been inserted.
pub(super) fn input_growth(
    db: &rusqlite::Connection,
    input: &super::Enqueue<'_>,
    page: Option<u32>,
    required: u32,
) -> Result<u64, Error> {
    let growth: i64 = db.query_row(&format!("SELECT {} FROM (SELECT ?1 AS kind,?2 AS target_kind,?3 AS target_id,?4 AS target_hash,?5 AS page,?6 AS required_l3_units) jobs", growth_expression()),
        params![input.kind.as_str(),input.target.as_str(),input.target_id,input.target_hash.as_slice(),page,required], |r| r.get(0))?;
    u64::try_from(growth).map_err(|_| Error::Invalid("L3 growth requirement"))
}

/// Recognizes an owned replacement before enqueue admission; it creates no slot.
pub(super) fn replacement_input(
    db: &rusqlite::Connection,
    input: &super::Enqueue<'_>,
) -> Result<bool, Error> {
    replacement_input_with_page(db, input, None)
}

/// Page replacement credit belongs only to the exact file/hash/ordinal owner.
pub(super) fn replacement_input_with_page(
    db: &rusqlite::Connection,
    input: &super::Enqueue<'_>,
    page: Option<u32>,
) -> Result<bool, Error> {
    Ok(db.query_row(
        &format!(
            "SELECT {REPLACEMENT} FROM (SELECT ?1 AS kind,?2 AS target_kind,?3 AS target_id,?4 AS target_hash,?5 AS page) jobs"
        ),
        params![input.kind.as_str(), input.target.as_str(), input.target_id,input.target_hash.as_slice(),page],
        |r| r.get(0),
    )?)
}

/// Shared claim admission and its read-only status evidence.
#[derive(Debug, serde::Serialize)]
pub(crate) struct EnrichmentClaims {
    summaries: u64,
    page_sections: u64,
    l3_units: u64,
    active_leases: u64,
    growth_leases: u64,
    growth_units: u64,
    max_l3_units: u64,
    available_slots: u64,
    pub(crate) paused: bool,
    reason: Option<&'static str>,
}
impl EnrichmentClaims {
    /// Remaining growth can be zero while same-coverage replacements still fit.
    pub(super) fn available_growth(&self) -> u64 {
        if self.paused { 0 } else { self.available_slots }
    }
    /// Detects lost replacement credit after explicit removal during an active lease.
    pub(super) fn exceeds_limit(&self) -> bool {
        self.l3_units.saturating_add(self.growth_units) > self.max_l3_units
    }
    /// Explains slot reservations without implying already-published summaries.
    pub(crate) fn summary(&self) -> String {
        format!(
            "L3 claims: {} units ({} summaries, {} page sections) + {} reserved units in {} growth leases / {}; {}{}",
            self.l3_units,
            self.summaries,
            self.page_sections,
            self.growth_units,
            self.growth_leases,
            self.max_l3_units,
            if self.paused { "paused" } else { "admissible" },
            self.reason
                .map(|reason| format!(" ({reason})"))
                .unwrap_or_default()
        )
    }
}

/// Active L3 leases reserve the remaining summary slots across providers and
/// processes. Count under the same writer as claim, excluding expired leases
/// whose publication fences can no longer authorize output.
pub(super) fn enrichment_claim_status(
    store: &Store,
    db: &rusqlite::Connection,
    now: i64,
) -> Result<EnrichmentClaims, Error> {
    let status = store.enrichment_status_in(db)?;
    let growth_sql = growth_expression();
    let (leased,growth,units):(i64,i64,i64) = db.query_row(
        &format!("SELECT count(*),coalesce(sum(({growth_sql})>0),0),coalesce(sum({growth_sql}),0) FROM jobs WHERE kind!='embed' AND state='leased' AND leased_until>?1"),
        [now], |r|Ok((r.get(0)?,r.get(1)?,r.get(2)?)),
    )?;
    let leased =
        u64::try_from(leased).map_err(|_| Error::Invalid("active enrichment lease count"))?;
    let growth =
        u64::try_from(growth).map_err(|_| Error::Invalid("enrichment growth lease count"))?;
    let units =
        u64::try_from(units).map_err(|_| Error::Invalid("enrichment reserved unit count"))?;
    let available_slots = status
        .max_l3_units
        .saturating_sub(status.l3_units.saturating_add(units));
    Ok(EnrichmentClaims {
        summaries: status.summaries,
        page_sections: status.page_sections,
        l3_units: status.l3_units,
        active_leases: leased,
        growth_leases: growth,
        growth_units: units,
        max_l3_units: status.max_l3_units,
        available_slots,
        paused: status.paused || available_slots == 0,
        reason: status
            .reason
            .or((available_slots == 0).then_some("max_l3_units reserved by active leases")),
    })
}

/// Only static capacity diagnostics are stored or displayed.
#[derive(Clone, Copy, Debug, thiserror::Error)]
pub(crate) enum Reason {
    #[error("max_l2_units cap; waiting for capacity")]
    Vectors,
    #[error("max_l3_units cap; waiting for capacity")]
    Summaries,
    #[error("max_units_per_file cap; waiting for capacity")]
    FileUnits,
    #[error("database 85 percent soft cap; waiting for capacity")]
    EnrichmentDatabase,
    #[error("database 95 percent soft cap; waiting for capacity")]
    Database,
    #[error("embedding document or batch token cap; waiting for capacity")]
    Tokens,
    #[error("SQLite storage full; waiting for capacity")]
    Storage,
    #[error("WAL capacity exhausted; waiting for checkpoint or capacity")]
    Wal,
}
impl Reason {
    /// Returns the stable persisted diagnostic used to identify resumable pauses.
    pub(super) fn message(self) -> &'static str {
        match self {
            Self::Vectors => "max_l2_units cap; waiting for capacity",
            Self::Summaries => "max_l3_units cap; waiting for capacity",
            Self::FileUnits => "max_units_per_file cap; waiting for capacity",
            Self::EnrichmentDatabase => LEGACY_DATABASE_REASON,
            Self::Database => "database 95 percent soft cap; waiting for capacity",
            Self::Tokens => "embedding document or batch token cap; waiting for capacity",
            Self::Storage => "SQLite storage full; waiting for capacity",
            Self::Wal => "WAL capacity exhausted; waiting for checkpoint or capacity",
        }
    }
}
/// Classifies only known cap failures, including the final transactional vector writer.
pub(super) fn reason(error: &Error) -> Option<Reason> {
    match error {
        Error::Capacity(reason) => Some(*reason),
        Error::PageCapacity { reason, .. } => Some(*reason),
        Error::Store(crate::store::Error::WalCapacity { .. }) => Some(Reason::Wal),
        Error::Store(crate::store::Error::VectorCapReached { .. }) => Some(Reason::Vectors),
        Error::Store(crate::store::Error::SummaryCapReached { .. }) => Some(Reason::Summaries),
        Error::Store(crate::store::Error::FileUnitCapReached { .. }) => Some(Reason::FileUnits),
        Error::Store(crate::store::Error::EnrichmentDatabasePaused { .. }) => {
            Some(Reason::EnrichmentDatabase)
        }
        Error::Sql(error) | Error::Store(crate::store::Error::Sqlite(error))
            if error.sqlite_error_code() == Some(rusqlite::ErrorCode::DiskFull) =>
        {
            Some(Reason::Storage)
        }
        _ => None,
    }
}
/// Makes this provider's capacity pauses eligible for one new explicit run.
/// Does not touch budget/rate delays, attempts, dead jobs, active leases or billing.
pub(crate) fn resume(
    store: &Store,
    provider: &str,
    clock: impl FnOnce() -> i64,
) -> Result<usize, Error> {
    resume_with_vectors(store, provider, None, clock)
}

/// Leaves dense jobs paused when their exact replacement still cannot fit.
/// Eligibility only permits a fresh claim; dispatch and publication retain all
/// current source, destination, privacy and capacity checks.
pub(crate) fn resume_embedding(
    store: &Store,
    provider: &str,
    slug: &str,
    max_l2_units: u64,
    max_input_tokens: i64,
    clock: impl FnOnce() -> i64,
) -> Result<usize, Error> {
    resume_with_vectors(
        store,
        provider,
        Some((slug, max_l2_units, max_input_tokens)),
        clock,
    )
}

/// A negative capacity observation only: false is never dispatch authority.
pub(super) fn vector_growth_blocked(
    store: &Store,
    tx: &rusqlite::Transaction<'_>,
    slug: &str,
    unit: i64,
    cap: u64,
) -> Result<bool, Error> {
    match crate::store::vectors::plan_replacement(store, tx, slug, unit, cap) {
        Ok(_) => Ok(false),
        Err(crate::store::Error::VectorCapReached { .. }) => Ok(true),
        Err(error) => Err(error.into()),
    }
}

fn resume_with_vectors(
    store: &Store,
    provider: &str,
    vectors: Option<(&str, u64, i64)>,
    clock: impl FnOnce() -> i64,
) -> Result<usize, Error> {
    if provider.is_empty() {
        return Err(Error::Invalid("capacity resume"));
    }
    // A previous guarded publication can have consumed the remaining WAL
    // allowance after its provider response was already reconciled. An explicit
    // resume is the safe boundary to reclaim unpinned frames before making that
    // retained work eligible for another outbound attempt.
    crate::store::wal::checkpoint(store)?;
    super::storage::job_batches(store, clock, |tx, window| {
        let ids = tx
            .prepare(
                "SELECT id FROM jobs WHERE id>?1 AND id<=?2 AND provider=?3
            AND state='queued' AND lease_owner IS NULL AND attempts<8 AND not_before=?4
            AND last_error IN (?5,?6,?7,?8,?9,?10,?11,?12) ORDER BY id LIMIT ?13",
            )?
            .query_map(
                params![
                    window.after,
                    window.through,
                    provider,
                    PAUSED,
                    Reason::Vectors.message(),
                    Reason::Database.message(),
                    Reason::Tokens.message(),
                    LEGACY_DATABASE_REASON,
                    Reason::Storage.message(),
                    Reason::Wal.message(),
                    Reason::Summaries.message(),
                    Reason::FileUnits.message(),
                    window.limit
                ],
                |r| r.get::<_, i64>(0),
            )?
            .collect::<Result<Vec<_>, _>>()?;
        let mut changed = 0;
        for id in &ids {
            if let Some((slug, cap, token_cap)) = vectors {
                // A saved token refusal cannot improve just because another
                // pass starts. New input generations are scheduled separately;
                // a larger current cap can make this same generation eligible.
                let token_blocked: bool = tx.query_row(
                    "SELECT kind='embed' AND last_error=?2 AND tokens>?3 FROM jobs WHERE id=?1",
                    params![id, Reason::Tokens.message(), token_cap],
                    |r| r.get(0),
                )?;
                if token_blocked {
                    continue;
                }
                let unit = tx
                    .query_row(
                        "SELECT target_id FROM jobs WHERE id=?1 AND kind='embed'
                     AND target_kind='unit' AND last_error=?2",
                        params![id, Reason::Vectors.message()],
                        |r| r.get::<_, i64>(0),
                    )
                    .optional()?;
                if let Some(unit) = unit
                    && vector_growth_blocked(store, tx, slug, unit, cap)?
                {
                    continue;
                }
            }
            let (page, growth): (bool, i64) = tx.query_row(
                &format!(
                    "SELECT kind='enrich:vlm_page',{} FROM jobs WHERE id=?1",
                    growth_expression()
                ),
                [id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )?;
            if page
                && u64::try_from(growth).map_err(|_| Error::Invalid("page resume growth"))?
                    > enrichment_claim_status(store, tx, window.now)?.available_growth()
            {
                continue;
            }
            changed+=tx.execute("UPDATE jobs SET not_before=?2,updated_at=?2,last_error=NULL
                WHERE id=?1 AND provider=?3 AND state='queued' AND lease_owner IS NULL AND attempts<8 AND not_before=?4
                AND last_error IN (?5,?6,?7,?8,?9,?10,?11,?12)",params![id,window.now,provider,PAUSED,Reason::Vectors.message(),Reason::Database.message(),Reason::Tokens.message(),LEGACY_DATABASE_REASON,Reason::Storage.message(),Reason::Wal.message(),Reason::Summaries.message(),Reason::FileUnits.message()])?;
        }
        Ok((ids, changed))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        queue::{self, retry, usage},
        store::{self, StoreLimits},
    };

    #[test]
    fn dense_resume_preserves_blocked_jobs_across_batches_and_credits_replacements()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::store::vectors::{VectorInsert, VectorTableSpec};
        let store = store::open(":memory:", StoreLimits::default())?;
        for slug in ["current", "retained"] {
            store.ensure_vector_table(VectorTableSpec {
                slug,
                model: slug,
                profile: "test",
                dims: 2,
                created_at: 0,
                active: true,
            })?;
        }
        for id in 1_i64..=131 {
            let hash = blake3::hash(&id.to_le_bytes());
            store.writer()?.execute(
                "INSERT INTO unit_contents(hash,text,tokens) VALUES(?1,'fixture',1)",
                [hash.as_bytes().as_slice()],
            )?;
            store.writer()?.execute(
                "INSERT INTO units(id,hash,content_hash,source_key,kind,header,created_at,updated_at)
                 VALUES(?1,?2,?2,'fixture:'||?1,'chunk','fixture',0,0)",
                params![id, hash.as_bytes().as_slice()],
            )?;
            queue::enqueue(&store, &queue::tests::input(id, &[1; 32]), 0)?;
            if id <= 2 {
                store.insert_vector(VectorInsert {
                    table_slug: if id == 1 { "current" } else { "retained" },
                    unit_id: id,
                    ctx_hash: &[2; 32],
                    embedding_hash: hash.as_bytes(),
                    embedding: &[1.0, 0.0],
                })?;
            }
        }
        // More than two control windows, including legacy vectors with no
        // embedding_publications receipts. The newest job has never been tried.
        store.writer()?.execute(
            "UPDATE jobs SET not_before=?1,last_error=?2,lease_generation=3,attempts=2 WHERE target_id<=130",
            params![PAUSED, Reason::Vectors.message()],
        )?;
        assert_eq!(
            resume_embedding(&store, "embed:local", "current", 0, i64::MAX, || 1)?,
            2
        );
        let owner = queue::Owner::new(&store)?;
        let leases = queue::claim(&store, "embed:local", &owner, 1, 131, 10_000)?;
        let mut targets = leases
            .iter()
            .map(|lease| {
                store
                    .writer()?
                    .query_row("SELECT target_id FROM jobs WHERE id=?1", [lease.id], |r| {
                        r.get::<_, i64>(0)
                    })
                    .map_err(store::Error::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        targets.sort_unstable();
        assert_eq!(targets, [1, 2, 131]);
        // Repeated provider passes cannot re-claim the other 128 blocked jobs.
        for now in 2..=4 {
            assert_eq!(
                resume_embedding(&store, "embed:local", "current", 0, i64::MAX, || now)?,
                0
            );
            assert!(queue::claim(&store, "embed:local", &owner, now, 131, 10_000)?.is_empty());
        }
        assert_eq!(
            store.writer()?.query_row(
                "SELECT COUNT(*) FROM jobs WHERE target_id BETWEEN 3 AND 130 AND lease_generation=3
             AND attempts=2 AND not_before=?1 AND last_error=?2",
                params![PAUSED, Reason::Vectors.message()],
                |r| r.get::<_, i64>(0),
            )?,
            128
        );
        assert_eq!(store.count_vectors()?, 2);
        // Current configuration supplies the cap; this Store was opened with
        // default limits. Raising it makes every retained pause eligible again.
        assert_eq!(
            resume_embedding(&store, "embed:local", "current", 3, i64::MAX, || 5)?,
            128
        );
        assert_eq!(
            store.writer()?.query_row(
                "SELECT COUNT(*) FROM jobs WHERE target_id BETWEEN 3 AND 130
             AND not_before=5 AND last_error IS NULL AND attempts=2 AND lease_generation=3",
                [],
                |r| r.get::<_, i64>(0),
            )?,
            128
        );
        assert_eq!(store.count_vectors()?, 2);
        Ok(())
    }

    #[test]
    fn cap_pause_releases_lease_and_unsent_reservations_but_preserves_attempts_and_uncertain_charges()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = store::open(":memory:", StoreLimits::default())?;
        queue::enqueue(&store, &queue::tests::input(1, &[1; 32]), 0)?;
        let owner = queue::Owner::new(&store)?;
        let lease = queue::claim(&store, "embed:local", &owner, 0, 1, 100)?.remove(0);
        let budget = usage::Budget {
            calls: None,
            tokens: None,
        };
        let held = usage::reserve(&store, "embed:local", 10, 0, budget, Some(&lease), || 1)?;
        assert!(matches!(held, usage::Admission::Reserved(_)));
        store.writer()?.execute("INSERT INTO usage_reservations(id,provider,day,job_id,lease_generation,calls,tokens,state,created_at,dispatched_at)
            VALUES('uncertain','embed:local','1970-01-01',?1,?2,1,10,'dispatched',1,1)",params![lease.id,lease.generation])?;
        store
            .writer()?
            .execute("UPDATE jobs SET attempts=3 WHERE id=?1", [lease.id])?;
        assert_eq!(
            retry::schedule(
                &store,
                &lease,
                retry::Outcome::Capacity(Reason::Vectors),
                || 2
            )?,
            retry::Scheduled::RetryAt(PAUSED)
        );
        let (state, attempts, until, owner): (String, i64, i64, Option<String>) =
            store.writer()?.query_row(
                "SELECT state,attempts,not_before,lease_owner FROM jobs WHERE id=?1",
                [lease.id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )?;
        assert_eq!(
            (state, attempts, until, owner),
            ("queued".into(), 3, PAUSED, None)
        );
        let (released, uncertain): (i64, i64) = store.writer()?.query_row(
            "SELECT SUM(state='released'),SUM(state='dispatched') FROM usage_reservations",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        assert_eq!((released, uncertain), (1, 1));
        assert!(
            queue::claim(
                &store,
                "embed:local",
                &queue::Owner::new(&store)?,
                3,
                1,
                100
            )?
            .is_empty()
        );
        assert_eq!(resume(&store, "other", || 3)?, 0);
        assert_eq!(resume(&store, "embed:local", || 3)?, 1);
        let newer = queue::claim(
            &store,
            "embed:local",
            &queue::Owner::new(&store)?,
            3,
            1,
            100,
        )?
        .remove(0);
        assert!(newer.generation > lease.generation);
        assert_eq!(
            retry::schedule(
                &store,
                &lease,
                retry::Outcome::Capacity(Reason::Database),
                || 4
            )?,
            retry::Scheduled::LeaseLost
        );
        Ok(())
    }

    #[test]
    fn capacity_resume_does_not_clear_budget_delays_or_exhausted_work()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = store::open(":memory:", StoreLimits::default())?;
        for id in 1..=3 {
            queue::enqueue(&store, &queue::tests::input(id, &[1; 32]), 0)?;
        }
        store.writer()?.execute(
            "UPDATE jobs SET not_before=?1,last_error=?2 WHERE target_id=1",
            params![PAUSED, Reason::Tokens.message()],
        )?;
        store.writer()?.execute(
            "UPDATE jobs SET not_before=86400,last_error=?1 WHERE target_id=2",
            [retry::BUDGET_REASON],
        )?;
        store.writer()?.execute(
            "UPDATE jobs SET attempts=8,not_before=?1,last_error=?2 WHERE target_id=3",
            params![PAUSED, Reason::Tokens.message()],
        )?;
        assert!(resume(&store, "embed:local", || -1).is_err());
        assert_eq!(resume(&store, "embed:local", || 1)?, 1);
        let unchanged:i64=store.writer()?.query_row("SELECT COUNT(*) FROM jobs WHERE (target_id=2 AND not_before=86400) OR (target_id=3 AND attempts=8 AND not_before=?1)",[PAUSED],|r|r.get(0))?;
        assert_eq!(unchanged, 2);
        Ok(())
    }

    #[test]
    fn capacity_resume_reclaims_unpinned_wal_before_releasing_wal_pauses()
    -> Result<(), Box<dyn std::error::Error>> {
        let file = tempfile::NamedTempFile::new()?;
        let store = store::open(
            file.path(),
            StoreLimits {
                wal_max_bytes: 8 * 1024 * 1024,
                ..Default::default()
            },
        )?;
        queue::enqueue(&store, &queue::tests::input(1, &[1; 32]), 0)?;
        store.writer()?.execute(
            "UPDATE jobs SET not_before=?1,last_error=?2 WHERE target_id=1",
            params![PAUSED, Reason::Wal.message()],
        )?;
        let writer = store.writer()?;
        writer.pragma_update(None, "wal_autocheckpoint", 0_i64)?;
        writer.execute_batch(
            "CREATE TABLE resume_wal_fixture(value BLOB);
             INSERT INTO resume_wal_fixture VALUES(zeroblob(2097152));",
        )?;
        let before = crate::store::wal::wal_bytes(&writer)?;
        assert!(before > 1024 * 1024);
        drop(writer);

        assert_eq!(resume(&store, "embed:local", || 1)?, 1);
        let after = crate::store::wal::wal_bytes(&*store.writer()?)?;
        assert!(after < before, "resume must reclaim the unpinned WAL");
        let state: (i64, Option<String>) = store.writer()?.query_row(
            "SELECT not_before,last_error FROM jobs WHERE target_id=1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        assert_eq!(state, (1, None));
        Ok(())
    }
}

#[cfg(test)]
mod threshold_tests {
    use super::*;
    #[test]
    fn ladder_thresholds_keep_l2_growth_until_ninety_five_percent() {
        for (used, dense, enrichment) in [
            (0, true, true),
            (8499, true, true),
            (8500, true, false),
            (9000, true, false),
            (9499, true, false),
            (9500, false, false),
            (10000, false, false),
        ] {
            assert_eq!(
                database_growth_allowed(used, 10000, GrowthLevel::Dense),
                dense
            );
            assert_eq!(
                database_growth_allowed(used, 10000, GrowthLevel::Enrichment),
                enrichment
            );
        }
        assert!(!database_growth_allowed(0, 0, GrowthLevel::Dense));
        assert!(!database_growth_allowed(
            u64::MAX,
            u64::MAX,
            GrowthLevel::Dense
        ));
        assert!(database_growth_allowed(
            u64::MAX / 2,
            u64::MAX,
            GrowthLevel::Dense
        ));
    }
    #[test]
    fn legacy_database_pauses_remain_visible_and_resumable()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::{
            queue,
            store::{self, StoreLimits},
        };
        let store = store::open(":memory:", StoreLimits::default())?;
        queue::enqueue(&store, &queue::tests::input(1, &[1; 32]), 0)?;
        store.writer()?.execute(
            "UPDATE jobs SET not_before=?1,last_error=?2",
            params![PAUSED, LEGACY_DATABASE_REASON],
        )?;
        let status = serde_json::to_value(queue::status::snapshot(&store, 1)?)?;
        assert_eq!(
            status["providers"]["embed:local"]["deferrals"]["cap_database"]["jobs"],
            1
        );
        assert_eq!(resume(&store, "embed:local", || 1)?, 1);
        assert_eq!(
            queue::claim(
                &store,
                "embed:local",
                &queue::Owner::new(&store)?,
                1,
                1,
                100
            )?
            .len(),
            1
        );
        Ok(())
    }
}
