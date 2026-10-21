//! Persistent desired-generation jobs and leases; see `v1.md` §8.
#![expect(
    dead_code,
    reason = "T1.5 staged queue core; dispatch and CLI wiring follow lease verification"
)]

use crate::store::Store;
use rusqlite::{OptionalExtension, Transaction, params};

pub(crate) mod capacity;
mod captions;
pub(crate) mod catalogue;
pub(crate) mod destination;
pub(crate) mod diagnostics;
pub(crate) mod embedding;
pub(crate) mod enrichment;
pub(crate) mod enrichment_catalogue;
pub(crate) mod enrichment_host;
pub(crate) mod enrichment_inputs;
mod history_source;
pub(crate) mod host;
pub(crate) mod http_embedding;
pub(crate) mod inspection;
mod module_source;
mod page_source;
pub(crate) mod pool;
mod pr_source;
pub(crate) mod priority;
pub(crate) mod remote_dimensions;
pub(crate) mod retry;
pub(crate) mod roles;
pub(crate) mod source;
pub(crate) mod status;
mod storage;
pub(crate) mod summary_source;
pub(crate) mod symbol_history;
pub(crate) mod usage;
mod visual_source;
pub(crate) mod worker;

const LEASE_SECONDS: i64 = 120;

/// Queue failures preserve typed storage causes without provider credentials.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error(transparent)]
    Capacity(#[from] capacity::Reason),
    #[error("{reason}")]
    PageCapacity {
        reason: capacity::Reason,
        units: u32,
    },
    #[error(transparent)]
    Store(#[from] crate::store::Error),
    #[error(transparent)]
    Sql(#[from] rusqlite::Error),
    #[error("invalid queue request: {0}")]
    Invalid(&'static str),
    #[error("current source inputs are unavailable; waiting for revalidation")]
    InputsUnavailable,
    #[error(transparent)]
    PagePreparation(#[from] crate::treat::pdf::scan::Error),
    /// Canonical input construction failed before a provider attempt could begin.
    #[error("input preparation failed; inspect configuration, prompts and source bounds")]
    InputPreparation,
    #[error("desired fingerprint reused with different job inputs")]
    IdentityConflict,
    #[error("bulk queue update stopped after {changed} committed rows: {source}")]
    BulkIncomplete {
        changed: usize,
        #[source]
        source: Box<Error>,
    },
    #[error(
        "recovery stopped after {jobs} jobs and {reservations} reservations committed: {source}"
    )]
    RecoveryIncomplete {
        jobs: usize,
        reservations: usize,
        #[source]
        source: Box<Error>,
    },
    #[error("queue database task did not complete")]
    WorkerTask,
    #[error("queue worker was cancelled")]
    WorkerStopped,
}

/// The one embedding operation and exactly five supported enrichment jobs.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Kind {
    Embed,
    DescribeSymbol,
    SummarizePr,
    SummarizeModule,
    DescribeImage,
    VlmPage,
}
impl Kind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::DescribeSymbol => "enrich:describe_symbol",
            Self::SummarizePr => "enrich:summarize_pr",
            Self::SummarizeModule => "enrich:summarize_module",
            Self::DescribeImage => "enrich:describe_image",
            Self::VlmPage => "enrich:vlm_page",
        }
    }
}

/// Includes L0 file targets so images/pages do not require pre-existing text units.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Target {
    Unit,
    File,
    Pr,
    Symbol,
}
impl Target {
    fn as_str(self) -> &'static str {
        match self {
            Self::Unit => "unit",
            Self::File => "file",
            Self::Pr => "pr",
            Self::Symbol => "symbol",
        }
    }
}

/// An already-rendered desired generation; callers hash all context/model/policy
/// inputs, excluding credentials. Scheduling caps are checked before this API.
pub(crate) struct Enqueue<'a> {
    pub(crate) kind: Kind,
    pub(crate) target: Target,
    pub(crate) target_id: i64,
    pub(crate) target_hash: &'a [u8; 32],
    pub(crate) desired_hash: &'a [u8; 32],
    pub(crate) provider: &'a str,
    pub(crate) priority: f64,
    pub(crate) tokens: i64,
}

/// A process-prefixed UUID owner, generated independently for each worker.
#[derive(Clone, Debug)]
pub(crate) struct Owner(String);
impl Owner {
    /// Uses SQLite's random source to avoid a second random-number dependency.
    pub(crate) fn new(store: &Store) -> Result<Self, Error> {
        let bytes: Vec<u8> = store
            .writer()?
            .query_row("SELECT randomblob(16)", [], |row| row.get(0))?;
        let mut bytes: [u8; 16] = bytes.try_into().map_err(|_| Error::Invalid("UUID bytes"))?;
        bytes[6] = (bytes[6] & 0x0f) | 0x40;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        Ok(Self(format!(
            "{}:{}-{}-{}-{}-{}",
            std::process::id(),
            &hex[..8],
            &hex[8..12],
            &hex[12..16],
            &hex[16..20],
            &hex[20..]
        )))
    }
}

/// A capability for one claim, retaining the source and desired generation fences.
#[derive(Clone, Debug)]
pub(crate) struct Lease {
    pub(crate) id: i64,
    pub(crate) target_hash: Vec<u8>,
    pub(crate) desired_hash: Vec<u8>,
    pub(crate) tokens: i64,
    generation: i64,
    owner: Owner,
}

/// Enqueues once per full desired identity, preserving done/dead/leased state on
/// repeated scheduling. New model/context fingerprints can reuse source bytes.
pub(crate) fn enqueue(store: &Store, input: &Enqueue<'_>, now: i64) -> Result<i64, Error> {
    storage::transaction(store, storage::Kind::Growth, |tx| {
        if !matches!(input.kind, Kind::Embed) {
            let existing:bool=tx.query_row("SELECT EXISTS(SELECT 1 FROM jobs WHERE kind=?1 AND target_kind=?2 AND target_id=?3 AND desired_hash=?4)",params![input.kind.as_str(),input.target.as_str(),input.target_id,input.desired_hash.as_slice()],|r|r.get(0))?;
            if !existing && !capacity::replacement_input(&tx, input)? {
                store.enrichment_status_in(&tx)?.admit_new()?;
            }
        }
        let id = enqueue_in(&tx, input, now, None)?;
        tx.commit()?;
        Ok(id)
    })
}

/// Inserts within the scheduler's admission transaction; never locks or commits.
/// An initial capacity pause applies only to new generations. Repeated scheduling
/// preserves leases, terminal states, quota delays and existing capacity pauses.
pub(super) fn enqueue_in(
    tx: &Transaction<'_>,
    input: &Enqueue<'_>,
    now: i64,
    paused: Option<capacity::Reason>,
) -> Result<i64, Error> {
    if now < 0 || input.tokens < 0 || !input.priority.is_finite() || input.provider.is_empty() {
        return Err(Error::Invalid("time, tokens, priority or provider"));
    }
    tx.execute("INSERT INTO jobs(kind,target_kind,target_id,target_hash,desired_hash,provider,priority,tokens,state,created_at,updated_at,not_before,last_error)
        VALUES(?1,?2,?3,?4,?5,?6,?7,?8,'queued',?9,?9,?10,?11)
        ON CONFLICT(kind,target_kind,target_id,desired_hash) DO NOTHING",
        params![input.kind.as_str(),input.target.as_str(),input.target_id,input.target_hash.as_slice(),input.desired_hash.as_slice(),input.provider,input.priority,input.tokens,now,paused.map_or(0, |_| capacity::PAUSED),paused.map(capacity::Reason::message)])?;
    let (id, hash, provider, tokens): (i64,Vec<u8>,String,i64) = tx.query_row(
        "SELECT id,target_hash,provider,tokens FROM jobs WHERE kind=?1 AND target_kind=?2 AND target_id=?3 AND desired_hash=?4",
        params![input.kind.as_str(),input.target.as_str(),input.target_id,input.desired_hash.as_slice()],
        |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?)))?;
    if hash != input.target_hash || provider != input.provider || tokens != input.tokens {
        return Err(Error::IdentityConflict);
    }
    // Priority is scheduling metadata, not part of the model input fingerprint.
    // Avoid rewriting its index for unchanged completed/current catalogue entries.
    tx.execute(
        "UPDATE jobs SET priority=?2 WHERE id=?1 AND priority<>?2",
        params![id, input.priority],
    )?;
    Ok(id)
}

/// Claims a highest-priority fitting batch under one immediate transaction.
/// Expiry is inclusive; claims increment fencing generations but do not count as
/// actual provider attempts. Dispatch/admission owns attempt accounting.
pub(crate) fn claim(
    store: &Store,
    provider: &str,
    owner: &Owner,
    now: i64,
    max_items: usize,
    max_tokens: i64,
) -> Result<Vec<Lease>, Error> {
    claim_with_clock(store, provider, owner, || now, max_items, max_tokens)
}

/// Recovers backlogs in control batches before atomically claiming at most 64
/// jobs (and possibly fewer under WAL pressure). Samples time/cancellation under
/// the writer lock on every claim attempt, including after recovery or rollback.
/// No lease commits unless every returned ownership change commits together.
pub(crate) fn claim_with_clock(
    store: &Store,
    provider: &str,
    owner: &Owner,
    mut clock: impl FnMut() -> i64,
    max_items: usize,
    max_tokens: i64,
) -> Result<Vec<Lease>, Error> {
    if provider.is_empty() || max_items == 0 || max_tokens < 0 {
        return Err(Error::Invalid("batch limits or provider"));
    }
    retry::recover(store, provider, &mut clock)?;
    storage::claim_batch(store, max_items, |tx, limit| {
        let now = clock();
        let until = now
            .checked_add(LEASE_SECONDS)
            .filter(|_| now >= 0)
            .ok_or(Error::Invalid("lease timestamp"))?;
        let mut result = Vec::new();
        let mut remaining = max_tokens;
        let mut allowed_growth = i64::MAX;
        let growth_sql = capacity::growth_expression();
        while result.len() < limit {
            let candidate = tx.query_row(&format!("SELECT id,target_hash,desired_hash,tokens,lease_generation,kind!='embed',{} FROM jobs
            WHERE provider=?1 AND tokens<=?2 AND attempts<8 AND (kind='embed' OR ({})<=?4) AND
              ((state IN ('queued','failed') AND not_before<=?3) OR (state='leased' AND leased_until<=?3))
            ORDER BY priority DESC,id LIMIT 1",growth_sql,growth_sql), params![provider,remaining,now,allowed_growth], |row| {
                Ok((Lease { id: row.get(0)?, target_hash: row.get(1)?, desired_hash: row.get(2)?, tokens: row.get(3)?, generation: row.get(4)?, owner: owner.clone() },row.get::<_,bool>(5)?,row.get::<_,i64>(6)?))
            }).optional()?;
            let Some((mut lease, enrichment, growth)) = candidate else {
                break;
            };
            if enrichment {
                let available =
                    capacity::enrichment_claim_status(store, &tx, now)?.available_growth();
                if u64::try_from(growth).map_err(|_| Error::Invalid("claim growth"))? > available {
                    allowed_growth = available.min(i64::MAX as u64) as i64;
                    continue;
                }
            }
            // This row may have expired after the recovery pass. Retire only
            // its proven-unsent old-generation allocations in this ownership
            // transaction; dispatched charges are never released by reclaiming.
            tx.execute(
                "UPDATE usage_reservations SET state='released' WHERE job_id=?1
                AND lease_generation=?2 AND state='reserved'",
                params![lease.id, lease.generation],
            )?;
            lease.generation = lease
                .generation
                .checked_add(1)
                .ok_or(Error::Invalid("lease generation overflow"))?;
            tx.execute("UPDATE jobs SET state='leased',lease_owner=?2,lease_generation=?3,leased_until=?4,updated_at=?5 WHERE id=?1",
            params![lease.id,owner.0,lease.generation,until,now])?;
            remaining -= lease.tokens;
            result.push(lease);
        }
        tx.commit()?;
        Ok(result)
    })
}

/// Extends only an unexpired matching lease. Expired or superseded workers cannot
/// revive ownership, even before another worker has reclaimed the row.
pub(crate) fn renew(store: &Store, lease: &Lease, now: i64) -> Result<bool, Error> {
    renew_with_clock(store, lease, || now)
}

/// Samples time after acquiring the writer so a waiting heartbeat cannot renew an
/// already-expired lease using a timestamp captured before lock contention.
pub(crate) fn renew_with_clock(
    store: &Store,
    lease: &Lease,
    clock: impl FnOnce() -> i64,
) -> Result<bool, Error> {
    storage::transaction(store, storage::Kind::Control, |tx| {
        let now = clock();
        let until = now
            .checked_add(LEASE_SECONDS)
            .filter(|_| now >= 0)
            .ok_or(Error::Invalid("lease timestamp"))?;
        let changed = tx.execute("UPDATE jobs SET leased_until=MAX(leased_until,?4),updated_at=?5
        WHERE id=?1 AND state='leased' AND lease_owner=?2 AND lease_generation=?3 AND leased_until>?5
        AND target_hash=?6 AND desired_hash=?7", params![lease.id,lease.owner.0,lease.generation,until,now,lease.target_hash,lease.desired_hash])? == 1;
        tx.commit()?;
        Ok(changed)
    })
}

/// Authoritative current inputs, recomputed from target/context/configuration
/// inside the publication transaction, rather than copied from the queued job.
pub(crate) struct CurrentInput {
    pub(crate) target_hash: [u8; 32],
    pub(crate) desired_hash: [u8; 32],
}

/// A completed job can retire obsolete work without publishing coverage.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Completion {
    Published,
    Discarded,
    LeaseLost,
}

/// Publishes database results and retires the job atomically under all fences.
/// `resolve` reads authoritative inputs (or returns None for a removed target).
/// Callbacks must use this transaction, never reacquire the store writer, change
/// job/input rows, or perform external work. Provider calls happen beforehand.
/// The clock is sampled after acquiring the lock, so time spent waiting cannot
/// authorize an expired lease. Once that check passes, the matching owner and
/// generation fence remains sufficient through SQL-only publication: this
/// transaction exclusively holds the writer, so no recovery can reclaim it.
pub(crate) fn complete(
    store: &Store,
    lease: &Lease,
    mut clock: impl FnMut() -> i64,
    resolve: impl FnOnce(&Transaction<'_>, &Lease) -> Result<Option<CurrentInput>, Error>,
    publish: impl FnOnce(&Transaction<'_>) -> Result<(), Error>,
) -> Result<Completion, Error> {
    storage::transaction(store, storage::Kind::Growth, |tx| {
        let started = clock();
        if started < 0 {
            return Err(Error::Invalid("completion timestamp"));
        }
        let enrichment: Option<bool> = tx
            .query_row(
                "SELECT kind!='embed' FROM jobs WHERE id=?1 AND state='leased'
         AND lease_owner=?2 AND lease_generation=?3 AND leased_until>?4
         AND target_hash=?5 AND desired_hash=?6",
                params![
                    lease.id,
                    lease.owner.0,
                    lease.generation,
                    started,
                    lease.target_hash,
                    lease.desired_hash
                ],
                |row| row.get(0),
            )
            .optional()?;
        let Some(enrichment) = enrichment else {
            return Ok(Completion::LeaseLost);
        };
        let current = resolve(&tx, lease)?;
        let matches = current.is_some_and(|input| {
            input.target_hash.as_slice() == lease.target_hash
                && input.desired_hash.as_slice() == lease.desired_hash
        });
        if matches {
            let before = enrichment
                .then(|| store.enrichment_status_in(&tx))
                .transpose()?;
            publish(&tx)?;
            if let Some(before) = before {
                before.check_growth(&store.enrichment_status_in(&tx)?)?;
            }
        }
        let finished = clock();
        if finished < started {
            return Err(Error::Invalid("completion clock moved backwards"));
        }
        let changed = tx.execute(
        "UPDATE jobs SET state='done',lease_owner=NULL,leased_until=NULL,updated_at=?4,last_error=?7
         WHERE id=?1 AND state='leased' AND lease_owner=?2 AND lease_generation=?3
         AND target_hash=?5 AND desired_hash=?6",
        params![lease.id, lease.owner.0, lease.generation, finished, lease.target_hash, lease.desired_hash,
            if matches { None } else { Some("superseded inputs; output discarded") }],
    )?;
        if changed != 1 {
            // Dropping the transaction also rolls back any publication writes.
            return Ok(Completion::LeaseLost);
        }
        tx.commit()?;
        Ok(if matches {
            Completion::Published
        } else {
            Completion::Discarded
        })
    })
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod vector_tests;
