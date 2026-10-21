//! Fenced retry scheduling, quota deferral and expired-work recovery; v1 §8.2.
use super::{Error, Lease};
pub(super) const BUDGET_REASON: &str = "daily budget exhausted; waiting until UTC midnight";
pub(super) const RATE_REASON: &str = "provider rate limit; waiting for capacity";
pub(super) const RETRY_REASON: &str = "transient provider error";

use crate::store::Store;
use backon::{BackoffBuilder, ExponentialBuilder};
use rusqlite::{OptionalExtension, params};
use std::time::Duration;

/// Only classified, credential-free reasons are persisted; adapters must classify
/// errors before this API instead of storing response bodies or arbitrary strings.
pub(crate) enum Outcome {
    Transient {
        retry_after: Option<Duration>,
    },
    Capacity(super::capacity::Reason),
    PageCapacity {
        reason: super::capacity::Reason,
        units: u32,
    },
    Permanent,
    PermanentChat(ChatFailure),
    EmptyContent,
    IncompleteContent,
    InputsUnavailable,
    /// Input construction failed locally; no provider attempt is consumed by this deferral.
    InputPreparation,
    Budget {
        retry_at: i64,
    },
    Rate {
        retry_at: i64,
    },
}

/// Source-free chat failure detail retained without changing retry policy.
pub(crate) enum ChatFailure {
    Http(u16, Option<crate::models::openai_compat::chat::HttpReason>),
    Input,
    Unauthorized,
    BodyLimit,
    Response,
}

impl ChatFailure {
    fn message(self) -> String {
        match self {
            Self::Http(status, reason) => match reason {
                Some(reason) => format!("chat provider HTTP {status}; {}", reason.code()),
                None => format!("chat provider HTTP {status}"),
            },
            Self::Input => "invalid chat request".into(),
            Self::Unauthorized => "chat provider unauthorized".into(),
            Self::BodyLimit => "chat response body limit".into(),
            Self::Response => "invalid chat response".into(),
        }
    }
}

/// Extracts only a valid numeric HTTP status, never arbitrary persisted text.
pub(super) fn chat_http_status(reason: &str) -> Option<u16> {
    chat_http_detail(reason).map(|(status, _)| status)
}

/// Strictly parses durable status and optional allowlisted provider code together.
pub(super) fn chat_http_detail(
    reason: &str,
) -> Option<(u16, Option<crate::models::openai_compat::chat::HttpReason>)> {
    let rest = reason.strip_prefix("chat provider HTTP ")?;
    let (status, detail) = match rest.split_once("; ") {
        Some((status, code)) => (
            status,
            Some(crate::models::openai_compat::chat::HttpReason::from_code(
                code,
            )?),
        ),
        None => (rest, None),
    };
    let status = status
        .parse::<u16>()
        .ok()
        .filter(|s| (100..=599).contains(s))?;
    Some((status, detail))
}

impl Outcome {
    /// Preserves a page's observed output width when publication rolls back.
    pub(super) fn from_capacity(error: &Error) -> Option<Self> {
        match error {
            Error::PageCapacity { reason, units } => Some(Self::PageCapacity {
                reason: *reason,
                units: *units,
            }),
            other => super::capacity::reason(other).map(Self::Capacity),
        }
    }
    /// Maps one HTTP embedding attempt into queue-owned retry scheduling without
    /// retaining provider payloads. Dimensions negotiation consumes another admission.
    pub(crate) fn from_provider_attempt(error: &crate::models::attempt::AttemptError) -> Self {
        use crate::models::attempt::AttemptError;
        match error {
            AttemptError::Chat(error) => {
                use crate::models::openai_compat::chat::Error;
                match error {
                    Error::ThinkingConsumedBudget { .. } | Error::EmptyContent { .. } => {
                        Self::EmptyContent
                    }
                    Error::IncompleteContent { .. } => Self::IncompleteContent,
                    Error::Transport => Self::Transient { retry_after: None },
                    Error::Http {
                        status,
                        retry_after,
                        ..
                    } if *status == 429 || *status >= 500 => Self::Transient {
                        retry_after: *retry_after,
                    },
                    Error::Http { status, reason, .. } => {
                        Self::PermanentChat(ChatFailure::Http(*status, *reason))
                    }
                    Error::Input => Self::PermanentChat(ChatFailure::Input),
                    Error::Unauthorized => Self::PermanentChat(ChatFailure::Unauthorized),
                    Error::BodyLimit => Self::PermanentChat(ChatFailure::BodyLimit),
                    Error::Response { .. } => Self::PermanentChat(ChatFailure::Response),
                }
            }
            AttemptError::Retryable { retry_after, .. } => Self::Transient {
                retry_after: *retry_after,
            },
            AttemptError::DimensionsRejected => Self::Transient { retry_after: None },
            AttemptError::Rejected { .. }
            | AttemptError::InvalidInput
            | AttemptError::InvalidResponse
            | AttemptError::Rerank(_)
            | AttemptError::LocalRerank(_) => Self::Permanent,
        }
    }
}

/// Result distinguishes ownership loss from a durable scheduling decision.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Scheduled {
    LeaseLost,
    RetryAt(i64),
    Dead,
}

fn retry_delay(attempts: i64, seed: u64) -> Result<i64, Error> {
    let index = attempts
        .checked_sub(1)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or(Error::Invalid("retry without dispatch"))?;
    let delay = ExponentialBuilder::default()
        .with_min_delay(Duration::from_secs(1))
        .with_max_delay(Duration::from_secs(60))
        .with_max_times(7)
        .with_jitter()
        .with_jitter_seed(seed)
        .build()
        .nth(index)
        .ok_or(Error::Invalid("retry attempt count"))?;
    // backon adds jitter after its cap. Bound the final delay, rounding up so
    // integer SQLite timestamps never schedule earlier than the requested delay.
    Ok((delay.as_secs() + u64::from(delay.subsec_nanos() != 0)).min(60) as i64)
}

/// Schedules a classified failure or quota wait only for a current unexpired lease.
/// No sleeping or attempt increment happens here; dispatch owns attempts. Release
/// the lease and its proven-unsent reservations together, retaining all dispatched
/// charges. Retry-After is a lower bound and can exceed the local 60-second cap.
pub(crate) fn schedule(
    store: &Store,
    lease: &Lease,
    outcome: Outcome,
    clock: impl FnOnce() -> i64,
) -> Result<Scheduled, Error> {
    super::storage::transaction(store, super::storage::Kind::Control, |tx| {
        let now = clock();
        if now < 0 {
            return Err(Error::Invalid("retry timestamp"));
        }
        let attempts: Option<i64> = tx.query_row("SELECT attempts FROM jobs WHERE id=?1 AND state='leased'
        AND lease_owner=?2 AND lease_generation=?3 AND leased_until>?4 AND target_hash=?5 AND desired_hash=?6",
        params![lease.id,lease.owner.0,lease.generation,now,lease.target_hash,lease.desired_hash],|r|r.get(0)).optional()?;
        let Some(attempts) = attempts else {
            return Ok(Scheduled::LeaseLost);
        };
        if let Outcome::PageCapacity { units, .. } = &outcome
            && (*units == 0 || tx.execute(
                "UPDATE jobs SET required_l3_units=?2 WHERE id=?1 AND kind='enrich:vlm_page' AND target_kind='file'",
                params![lease.id, units],
            )? != 1) {
            return Err(Error::Invalid("page output requirement"));
        }
        let permanent_chat_message;
        let (state, reason, not_before) = if attempts >= 8 {
            ("dead", "attempt limit reached", now)
        } else {
            match outcome {
                Outcome::InputPreparation => (
                    "queued",
                    "input preparation failed; inspect configuration, prompts and source bounds",
                    now.checked_add(60)
                        .ok_or(Error::Invalid("input retry timestamp overflow"))?,
                ),
                Outcome::InputsUnavailable => (
                    "queued",
                    "current source inputs unavailable; waiting for revalidation",
                    now.checked_add(60)
                        .ok_or(Error::Invalid("input retry timestamp overflow"))?,
                ),
                Outcome::Capacity(reason) => ("queued", reason.message(), super::capacity::PAUSED),
                Outcome::PageCapacity { reason, .. } => {
                    ("queued", reason.message(), super::capacity::PAUSED)
                }
                Outcome::Permanent => ("dead", "permanent provider error", now),
                Outcome::PermanentChat(reason) => {
                    permanent_chat_message = reason.message();
                    ("dead", permanent_chat_message.as_str(), now)
                }
                Outcome::EmptyContent => {
                    ("dead", "empty provider content; diagnosis required", now)
                }
                Outcome::IncompleteContent => (
                    "dead",
                    "incomplete page transcription; diagnosis required",
                    now,
                ),
                Outcome::Transient { retry_after } => {
                    let bytes: [u8; 8] = tx.query_row("SELECT randomblob(8)", [], |r| r.get(0))?;
                    let backoff = retry_delay(attempts, u64::from_le_bytes(bytes))?;
                    let requested = retry_after
                        .map(|d| {
                            d.as_secs()
                                .checked_add(u64::from(d.subsec_nanos() != 0))
                                .and_then(|s| i64::try_from(s).ok())
                                .ok_or(Error::Invalid("Retry-After overflow"))
                        })
                        .transpose()?
                        .unwrap_or(0);
                    let until = now
                        .checked_add(backoff.max(requested))
                        .ok_or(Error::Invalid("retry timestamp overflow"))?;
                    ("failed", RETRY_REASON, until)
                }
                Outcome::Budget { retry_at } => {
                    if retry_at <= 0 || retry_at % 86400 != 0 {
                        return Err(Error::Invalid("budget midnight"));
                    }
                    // Admission and retry publication use independent clock
                    // observations. An elapsed quota boundary means recheck now,
                    // not permission to dispatch without the normal budget gate.
                    ("queued", BUDGET_REASON, retry_at.max(now))
                }
                Outcome::Rate { retry_at } => {
                    if retry_at <= 0 {
                        return Err(Error::Invalid("rate retry time"));
                    }
                    ("queued", RATE_REASON, retry_at.max(now))
                }
            }
        };
        tx.execute("UPDATE usage_reservations SET state='released' WHERE job_id=?1 AND lease_generation=?2 AND state='reserved'",
        params![lease.id,lease.generation])?;
        tx.execute("UPDATE jobs SET state=?2,not_before=?3,last_error=?4,lease_owner=NULL,leased_until=NULL,updated_at=?5 WHERE id=?1",
        params![lease.id,state,not_before,reason,now])?;
        tx.commit()?;
        Ok(if state == "dead" {
            Scheduled::Dead
        } else {
            Scheduled::RetryAt(not_before)
        })
    })
}

/// Recovery counts expose what changed without treating uncertain billing as freed.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Recovered {
    pub(crate) jobs: usize,
    pub(crate) reservations: usize,
}

/// Explicitly requeues failed/dead work with attempts remaining. Preserves desired
/// identity, lease generation, Retry-After and all billing records; exhausted
/// generations require changed inputs, never an attempt-counter reset. Active,
/// completed and capacity/quota-paused jobs are left to their existing owners.
/// Batches commit independently; a later failure reports the committed row count.
pub(crate) fn request(store: &Store, job_id: Option<i64>, now: i64) -> Result<usize, Error> {
    if now < 0 || job_id.is_some_and(|id| id <= 0) {
        return Err(Error::Invalid("retry job id or timestamp"));
    }
    super::storage::job_batches(
        store,
        || now,
        |tx, window| {
            let ids = tx
                .prepare(
                    "SELECT id FROM jobs WHERE id>?1 AND id<=?2
            AND (?3 IS NULL OR id=?3) AND state IN ('failed','dead') AND attempts<8
            ORDER BY id LIMIT ?4",
                )?
                .query_map(
                    params![window.after, window.through, job_id, window.limit],
                    |r| r.get::<_, i64>(0),
                )?
                .collect::<Result<Vec<_>, _>>()?;
            let mut changed = 0;
            for id in &ids {
                changed += tx.execute(
                "UPDATE jobs SET state='queued',lease_owner=NULL,leased_until=NULL,updated_at=?2
                 WHERE id=?1 AND state IN ('failed','dead') AND attempts<8",
                params![id,window.now],
            )?;
            }
            Ok((ids, changed))
        },
    )
}

/// Explicitly requeues one completed text summary without deleting its visible
/// output. The worker repeats source, policy, desired-generation and owner
/// checks before a new provider call; publication replaces the output only when
/// those fences still hold. Attempts, lease generation and every usage record
/// are preserved, so this is suitable for a human-reviewed quality correction
/// but cannot revive exhausted, failed, image, page or embedding work.
pub(crate) fn request_regeneration(store: &Store, job_id: i64, now: i64) -> Result<bool, Error> {
    if job_id <= 0 || now < 0 {
        return Err(Error::Invalid("regeneration job id or timestamp"));
    }
    super::storage::transaction(store, super::storage::Kind::Control, |tx| {
        let changed = tx.execute(
            &format!(
                "UPDATE jobs SET state='queued',not_before=?2,last_error='manual regeneration requested; existing output retained',updated_at=?2
                 WHERE id=?1 AND state='done' AND attempts<8
                 AND (kind,target_kind) IN (
                    ('enrich:describe_symbol','symbol'),
                    ('enrich:summarize_pr','pr'),
                    ('enrich:summarize_module','unit')
                 )
                 AND {}",
                super::capacity::REPLACEMENT
            ),
            params![job_id, now],
        )?;
        tx.commit()?;
        Ok(changed == 1)
    })
}

/// Recovers a finite backlog in independently committed batches. Job retirement
/// precedes unsent-allocation cleanup; a failure between phases can conservatively
/// retain reservations but cannot free an uncertain dispatched charge. A later
/// failure reports both committed counts and is safe to resume with another pass.
pub(crate) fn recover(
    store: &Store,
    provider: &str,
    clock: impl FnOnce() -> i64,
) -> Result<Recovered, Error> {
    if provider.is_empty() {
        return Err(Error::Invalid("recovery provider"));
    }
    let mut sampled = None;
    let jobs = super::storage::job_batches(store, || {
        let now = clock();
        sampled = Some(now);
        now
    }, |tx, window| {
        let _: String = tx.query_row("SELECT date(?1,'unixepoch')", [window.now], |r| r.get(0))?;
        let ids = tx.prepare("SELECT id FROM jobs WHERE id>?1 AND id<=?2 AND provider=?3
            AND ((state='leased' AND leased_until<=?4) OR (state IN ('queued','failed') AND attempts>=8))
            ORDER BY id LIMIT ?5")?
            .query_map(params![window.after,window.through,provider,window.now,window.limit], |r| r.get::<_,i64>(0))?
            .collect::<Result<Vec<_>,_>>()?;
        let mut changed = 0;
        for id in &ids {
            changed += tx.execute("UPDATE jobs SET state=CASE WHEN attempts>=8 THEN 'dead' ELSE 'failed' END,
                last_error=CASE WHEN attempts>=8 THEN 'attempt limit reached during recovery' ELSE 'lease expired; outcome uncertain' END,
                lease_owner=NULL,leased_until=NULL,updated_at=?2
                WHERE id=?1 AND provider=?3 AND ((state='leased' AND leased_until<=?2)
                OR (state IN ('queued','failed') AND attempts>=8))", params![id,window.now,provider])?;
        }
        Ok((ids,changed))
    }).map_err(|error| recovery_error(error, 0, false))?;
    let now = sampled.ok_or(Error::Invalid("recovery clock not sampled"))?;
    let reservations = super::storage::row_batches::<String>(store, super::storage::Rows::Reservations, || now, |tx, window| {
        let ids = tx.prepare("SELECT id FROM usage_reservations
            WHERE id>?1 AND id<=?2 AND provider=?3 AND state='reserved' AND
            (day<date(?4,'unixepoch') OR (lease_generation IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM jobs j WHERE j.id=usage_reservations.job_id AND j.state='leased'
                AND j.lease_generation=usage_reservations.lease_generation AND j.leased_until>?4)))
            ORDER BY id LIMIT ?5")?
            .query_map(params![window.after,window.through,provider,window.now,window.limit], |r| r.get::<_,String>(0))?
            .collect::<Result<Vec<_>,_>>()?;
        let mut changed = 0;
        for id in &ids {
            changed += tx.execute("UPDATE usage_reservations SET state='released'
                WHERE id=?1 AND provider=?2 AND state='reserved' AND
                (day<date(?3,'unixepoch') OR (lease_generation IS NOT NULL AND NOT EXISTS (
                    SELECT 1 FROM jobs j WHERE j.id=usage_reservations.job_id AND j.state='leased'
                    AND j.lease_generation=usage_reservations.lease_generation AND j.leased_until>?3)))",
                params![id,provider,window.now])?;
        }
        Ok((ids,changed))
    }).map_err(|error| recovery_error(error, jobs, true))?;
    Ok(Recovered { jobs, reservations })
}

fn recovery_error(error: Error, jobs: usize, releasing: bool) -> Error {
    let (changed, source) = match error {
        Error::BulkIncomplete { changed, source } => (changed, source),
        error => (0, Box::new(error)),
    };
    let (jobs, reservations) = if releasing {
        (jobs, changed)
    } else {
        (changed, 0)
    };
    if jobs == 0 && reservations == 0 {
        *source
    } else {
        Error::RecoveryIncomplete {
            jobs,
            reservations,
            source,
        }
    }
}

/// Former unbounded recovery retained only as a regression workload.
#[cfg(test)]
pub(super) fn recover_in(
    tx: &rusqlite::Transaction<'_>,
    provider: &str,
    now: i64,
) -> Result<Recovered, Error> {
    if provider.is_empty() || now < 0 {
        return Err(Error::Invalid("recovery provider or timestamp"));
    }
    let today: String = tx.query_row("SELECT date(?1,'unixepoch')", [now], |r| r.get(0))?;
    let jobs=tx.execute("UPDATE jobs SET state=CASE WHEN attempts>=8 THEN 'dead' ELSE 'failed' END,
        last_error=CASE WHEN attempts>=8 THEN 'attempt limit reached during recovery' ELSE 'lease expired; outcome uncertain' END,
        lease_owner=NULL,leased_until=NULL,updated_at=?2
        WHERE provider=?1 AND ((state='leased' AND leased_until<=?2) OR (state IN ('queued','failed') AND attempts>=8))",
        params![provider,now])?;
    let reservations = tx.execute(
        "UPDATE usage_reservations SET state='released' WHERE provider=?1 AND state='reserved' AND
        (day<?3 OR (lease_generation IS NOT NULL AND NOT EXISTS (
         SELECT 1 FROM jobs j WHERE j.id=usage_reservations.job_id AND j.state='leased'
         AND j.lease_generation=usage_reservations.lease_generation AND j.leased_until>?2)))",
        params![provider, now, today],
    )?;
    Ok(Recovered { jobs, reservations })
}

#[cfg(test)]
mod tests;
