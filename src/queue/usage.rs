//! Durable daily reservations and crash-conservative usage accounting; v1 §8.2.
//! Rolling minute rates share the dispatch transaction; worker wiring follows.
use super::{Error, Lease};
use crate::store::Store;
use rusqlite::{OptionalExtension, Transaction, params};

/// Per-provider daily ceilings; None is unlimited, zero explicitly disables calls.
#[derive(Clone, Copy)]
pub(crate) struct Budget {
    pub(crate) calls: Option<i64>,
    pub(crate) tokens: Option<i64>,
}

/// Current effective provider limits, resolved inside the dispatch transaction.
/// None is unlimited; zero blocks requests (or nonzero tokens for tpm).
#[derive(Clone, Copy)]
pub(crate) struct Limits {
    pub(crate) budget: Budget,
    pub(crate) rpm: Option<i64>,
    pub(crate) tpm: Option<i64>,
}

/// Source-free metadata for one potential outbound attempt. Units count inputs,
/// including retries; the dispatch record cannot prove remote receipt.
pub(crate) struct Endpoint {
    pub(crate) kind: &'static str,
    pub(crate) url: String,
    pub(crate) units: i64,
}

/// Captures identity in the dispatch transaction; quota or policy rejection rolls
/// this back with the attempt. Credentials and URL query strings are forbidden.
pub(crate) fn record_endpoint(
    tx: &Transaction<'_>,
    reservation: &Reservation,
    endpoint: &Endpoint,
) -> Result<(), Error> {
    let url =
        reqwest::Url::parse(&endpoint.url).map_err(|_| Error::Invalid("outbound endpoint URL"))?;
    if !matches!(endpoint.kind, "embed" | "rerank" | "llm" | "vlm")
        || endpoint.units <= 0
        || !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(Error::Invalid("outbound endpoint metadata"));
    }
    let changed = tx.execute("UPDATE usage_reservations SET endpoint_kind=?2,endpoint_url=?3,sent_units=?4 WHERE id=?1 AND state='reserved'", params![reservation.0,endpoint.kind,url.as_str(),endpoint.units])?;
    if changed != 1 {
        return Err(Error::Invalid("outbound reservation unavailable"));
    }
    Ok(())
}

/// Opaque identity for one outbound request, including input and maximum output.
#[derive(Debug)]
pub(crate) struct Reservation(String);

/// Refusal gives the worker a durable scheduling boundary without waiting in SQL.
#[derive(Debug)]
pub(crate) enum Admission {
    Reserved(Reservation),
    BudgetExhausted { retry_at: i64 },
    LeaseLost,
}

/// A held reservation must be dispatched on its reserved UTC day and owned lease.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Dispatch {
    Recorded,
    Superseded,
    LeaseLost,
    DayExpired,
    Unavailable,
    AttemptsExhausted,
    BudgetExhausted { retry_at: i64 },
    RateLimited { retry_at: i64 },
    RequestTooLarge,
}

fn day(tx: &Transaction<'_>, now: i64) -> Result<String, Error> {
    if now < 0 {
        return Err(Error::Invalid("usage timestamp"));
    }
    // SQLite rejects dates outside its calendar range; never invent a day key.
    Ok(tx.query_row("SELECT date(?1,'unixepoch')", [now], |r| r.get(0))?)
}

fn owned(tx: &Transaction<'_>, lease: &Lease, provider: &str, now: i64) -> Result<bool, Error> {
    Ok(tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM jobs WHERE id=?1 AND state='leased'
        AND lease_owner=?2 AND lease_generation=?3 AND leased_until>?4
        AND target_hash=?5 AND desired_hash=?6 AND provider=?7)",
        params![
            lease.id,
            lease.owner.0,
            lease.generation,
            now,
            lease.target_hash,
            lease.desired_hash,
            provider
        ],
        |r| r.get(0),
    )?)
}

/// The lease was live when this writer transaction began. During its SQL-only
/// authoritative resolution, the same writer lock prevents recovery from
/// changing the row; keep every identity fence without requiring a second
/// wall-clock expiry check before committing the admission decision.
fn still_owned(tx: &Transaction<'_>, lease: &Lease, provider: &str) -> Result<bool, Error> {
    Ok(tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM jobs WHERE id=?1 AND state='leased'
        AND lease_owner=?2 AND lease_generation=?3
        AND target_hash=?4 AND desired_hash=?5 AND provider=?6)",
        params![
            lease.id,
            lease.owner.0,
            lease.generation,
            lease.target_hash,
            lease.desired_hash,
            provider
        ],
        |r| r.get(0),
    )?)
}

/// Holds one daily call and estimated input plus maximum output tokens atomically.
/// Queued calls require their current lease; query calls use None. Pending and
/// uncertain dispatched reservations both count alongside settled usage. This
/// API alone does not authorize HTTP: mark_dispatched must admit minute rates.
pub(crate) fn reserve(
    store: &Store,
    provider: &str,
    input_tokens: i64,
    max_output_tokens: i64,
    budget: Budget,
    lease: Option<&Lease>,
    clock: impl FnOnce() -> i64,
) -> Result<Admission, Error> {
    if provider.is_empty()
        || input_tokens < 0
        || max_output_tokens < 0
        || budget.calls.is_some_and(|v| v < 0)
        || budget.tokens.is_some_and(|v| v < 0)
    {
        return Err(Error::Invalid("provider, tokens or daily budget"));
    }
    let tokens = input_tokens
        .checked_add(max_output_tokens)
        .ok_or(Error::Invalid("token overflow"))?;
    super::storage::transaction(store, super::storage::Kind::Growth, |tx| {
        let now = clock();
        let today = day(&tx, now)?;
        if let Some(lease) = lease
            && !owned(&tx, lease, provider, now)?
        {
            return Ok(Admission::LeaseLost);
        }
        let (calls, used): (i64,i64) = tx.query_row(
        "SELECT COALESCE(SUM(calls),0),COALESCE(SUM(tokens),0) FROM (
         SELECT calls,tokens FROM usage WHERE day=?1 AND provider=?2 UNION ALL
         SELECT calls,tokens FROM usage_reservations WHERE day=?1 AND provider=?2 AND state IN ('reserved','dispatched'))",
        params![today,provider], |r|Ok((r.get(0)?,r.get(1)?)))?;
        let calls = calls
            .checked_add(1)
            .ok_or(Error::Invalid("call overflow"))?;
        let used = used
            .checked_add(tokens)
            .ok_or(Error::Invalid("usage overflow"))?;
        if budget.calls.is_some_and(|cap| calls > cap)
            || budget.tokens.is_some_and(|cap| used > cap)
        {
            return Ok(Admission::BudgetExhausted {
                retry_at: (now / 86400 + 1) * 86400,
            });
        }
        let id: String = tx.query_row("SELECT lower(hex(randomblob(16)))", [], |r| r.get(0))?;
        tx.execute("INSERT INTO usage_reservations(id,provider,day,job_id,lease_generation,calls,tokens,state,created_at)
        VALUES(?1,?2,?3,?4,?5,1,?6,'reserved',?7)",
        params![id,provider,today,lease.map(|l|l.id),lease.map(|l|l.generation),tokens,now])?;
        tx.commit()?;
        Ok(Admission::Reserved(Reservation(id)))
    })
}

/// Records potential dispatch before an external request. Never release this
/// reservation after an ambiguous network outcome. Duplicate dispatch is refused.
/// Current effective limits must be read by `resolve` using the supplied transaction;
/// it must not reacquire the writer or do external work. Rolling quota, daily-budget
/// revalidation and one queue attempt are committed together. A crash just before
/// HTTP conservatively consumes an attempt; at most eight requests can be authorized.
/// Callers still own current source/outbound-policy checks and bounded single-attempt
/// provider execution immediately after commit. Never sleep while holding this lock.
pub(crate) fn mark_dispatched(
    store: &Store,
    reservation: &Reservation,
    lease: Option<&Lease>,
    clock: impl FnMut() -> i64,
    resolve: impl FnOnce(&Transaction<'_>, &str) -> Result<Limits, Error>,
) -> Result<Dispatch, Error> {
    mark_dispatched_checked(store, reservation, lease, clock, |tx, provider| {
        resolve(tx, provider).map(Some)
    })
}

/// Checks authoritative inputs/policy in the admission transaction. A queued
/// lease must be live after acquiring the writer; its identity fence then spans
/// SQL-only resolution and its deadline is refreshed before an outbound call.
/// None retires obsolete owned work and releases unsent reservations without an
/// outbound call.
pub(crate) fn mark_dispatched_checked(
    store: &Store,
    reservation: &Reservation,
    lease: Option<&Lease>,
    clock: impl FnMut() -> i64,
    resolve: impl FnOnce(&Transaction<'_>, &str) -> Result<Option<Limits>, Error>,
) -> Result<Dispatch, Error> {
    match lease {
        Some(lease) => mark_dispatched_many_checked(
            store,
            reservation,
            std::slice::from_ref(lease),
            true,
            clock,
            resolve,
        ),
        None => mark_dispatched_many_checked(store, reservation, &[], true, clock, resolve),
    }
}

/// Admits one provider request that covers several independently fenced jobs.
/// The reservation is attached to the first lease for crash recovery, while the
/// dispatch transaction verifies and consumes one attempt from every lease.
pub(crate) fn mark_dispatched_batch_checked(
    store: &Store,
    reservation: &Reservation,
    leases: &[Lease],
    clock: impl FnMut() -> i64,
    resolve: impl FnOnce(&Transaction<'_>, &str) -> Result<Option<Limits>, Error>,
) -> Result<Dispatch, Error> {
    if leases.is_empty() {
        return Err(Error::Invalid("empty dispatch batch"));
    }
    mark_dispatched_many_checked(store, reservation, leases, false, clock, resolve)
}

fn mark_dispatched_many_checked(
    store: &Store,
    reservation: &Reservation,
    leases: &[Lease],
    retire_superseded: bool,
    mut clock: impl FnMut() -> i64,
    resolve: impl FnOnce(&Transaction<'_>, &str) -> Result<Option<Limits>, Error>,
) -> Result<Dispatch, Error> {
    super::storage::transaction(store, super::storage::Kind::Growth, |tx| {
        let now = clock();
        let today = day(&tx, now)?;
        let row = tx.query_row("SELECT provider,day,job_id,lease_generation,created_at,tokens FROM usage_reservations WHERE id=?1 AND state='reserved'",
        [&reservation.0], |r|Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?,r.get::<_,Option<i64>>(2)?,r.get::<_,Option<i64>>(3)?,r.get::<_,i64>(4)?,r.get::<_,i64>(5)?))).optional()?;
        let Some((provider, reserved_day, job_id, generation, created_at, tokens)) = row else {
            return Ok(Dispatch::Unavailable);
        };
        if now < created_at {
            return Err(Error::Invalid("dispatch clock moved backwards"));
        }
        if today != reserved_day {
            return Ok(Dispatch::DayExpired);
        }
        match (job_id, generation, leases.first()) {
            (None, None, None) => {}
            (Some(id), Some(g), Some(lease))
                if id == lease.id
                    && g == lease.generation
                    && leases
                        .iter()
                        .map(|lease| owned(&tx, lease, &provider, now))
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter()
                        .all(|owned| owned) => {}
            _ => return Ok(Dispatch::LeaseLost),
        }
        let limits = resolve(&tx, &provider)?;
        let resolved_at = clock();
        if resolved_at < now {
            return Err(Error::Invalid("dispatch clock moved backwards"));
        }
        let now = resolved_at;
        if day(&tx, now)? != today {
            return Ok(Dispatch::DayExpired);
        }
        if !leases
            .iter()
            .map(|lease| still_owned(&tx, lease, &provider))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .all(|owned| owned)
        {
            return Ok(Dispatch::LeaseLost);
        }
        let Some(limits) = limits else {
            if let Some(lease) = leases.first()
                && retire_superseded
            {
                let changed = tx.execute("UPDATE jobs SET state='done',lease_owner=NULL,leased_until=NULL,updated_at=?4,last_error='superseded inputs or outbound policy; no request sent'
                    WHERE id=?1 AND state='leased' AND lease_owner=?2 AND lease_generation=?3
                    AND target_hash=?5 AND desired_hash=?6 AND provider=?7",params![lease.id,lease.owner.0,lease.generation,now,lease.target_hash,lease.desired_hash,provider])?;
                if changed != 1 {
                    return Ok(Dispatch::LeaseLost);
                }
                tx.execute("UPDATE usage_reservations SET state='released' WHERE job_id=?1 AND lease_generation=?2 AND state='reserved'",params![lease.id,lease.generation])?;
            } else {
                tx.execute(
                    "UPDATE usage_reservations SET state='released' WHERE id=?1",
                    [&reservation.0],
                )?;
            }
            tx.commit()?;
            return Ok(Dispatch::Superseded);
        };
        if [
            limits.rpm,
            limits.tpm,
            limits.budget.calls,
            limits.budget.tokens,
        ]
        .into_iter()
        .flatten()
        .any(|v| v < 0)
        {
            return Err(Error::Invalid("dispatch limits"));
        }
        for lease in leases {
            let attempts: Option<i64> = tx
                .query_row(
                    "SELECT attempts FROM jobs WHERE id=?1 AND state='leased' AND lease_owner=?2
                 AND lease_generation=?3 AND target_hash=?4 AND desired_hash=?5 AND provider=?6",
                    params![
                        lease.id,
                        lease.owner.0,
                        lease.generation,
                        lease.target_hash,
                        lease.desired_hash,
                        provider
                    ],
                    |r| r.get(0),
                )
                .optional()?;
            let Some(attempts) = attempts else {
                return Ok(Dispatch::LeaseLost);
            };
            if attempts >= 8 {
                return Ok(Dispatch::AttemptsExhausted);
            }
        }
        let (calls, daily_tokens): (i64,i64) = tx.query_row(
        "SELECT COALESCE(SUM(calls),0),COALESCE(SUM(tokens),0) FROM (
         SELECT calls,tokens FROM usage WHERE day=?1 AND provider=?2 UNION ALL
         SELECT calls,tokens FROM usage_reservations WHERE day=?1 AND provider=?2 AND state IN ('reserved','dispatched'))",
        params![today,provider], |r|Ok((r.get(0)?,r.get(1)?)))?;
        if limits.budget.calls.is_some_and(|cap| calls > cap)
            || limits.budget.tokens.is_some_and(|cap| daily_tokens > cap)
        {
            return Ok(Dispatch::BudgetExhausted {
                retry_at: (now / 86400 + 1) * 86400,
            });
        }
        if limits.rpm == Some(0) || limits.tpm.is_some_and(|cap| tokens > cap) {
            return Ok(Dispatch::RequestTooLarge);
        }
        // Maximum estimates (or larger reported overruns) remain charged for the
        // minute; neither a quick response nor a crash refunds rate capacity.
        let (requests, charged, newest, oldest): (i64,i64,Option<i64>,Option<i64>) = tx.query_row(
        "SELECT COALESCE(SUM(calls),0),COALESCE(SUM(tokens),0),MAX(dispatched_at),MIN(dispatched_at)
         FROM usage_reservations WHERE provider=?1 AND dispatched_at>?2",
        params![provider,now-60], |r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?)))?;
        if newest.is_some_and(|t| t > now) {
            return Err(Error::Invalid("dispatch clock moved backwards"));
        }
        let requests = requests
            .checked_add(1)
            .ok_or(Error::Invalid("rate request overflow"))?;
        let charged = charged
            .checked_add(tokens)
            .ok_or(Error::Invalid("rate token overflow"))?;
        if limits.rpm.is_some_and(|cap| requests > cap)
            || limits.tpm.is_some_and(|cap| charged > cap)
        {
            return Ok(Dispatch::RateLimited {
                retry_at: oldest.unwrap_or(now) + 60,
            });
        }
        tx.execute("INSERT INTO provider_admission(provider,request_window,requests,token_window,tokens)
        VALUES(?1,?2,?3,?2,?4) ON CONFLICT(provider) DO UPDATE SET request_window=excluded.request_window,
        requests=excluded.requests,token_window=excluded.token_window,tokens=excluded.tokens",
        params![provider,now,requests,charged])?;
        for lease in leases {
            let until = now
                .checked_add(super::LEASE_SECONDS)
                .ok_or(Error::Invalid("lease timestamp"))?;
            let changed = tx.execute(
                "UPDATE jobs SET attempts=attempts+1,leased_until=MAX(leased_until,?4),updated_at=?5
                 WHERE id=?1 AND state='leased' AND lease_owner=?2 AND lease_generation=?3
                 AND target_hash=?6 AND desired_hash=?7 AND provider=?8",
                params![lease.id,lease.owner.0,lease.generation,until,now,lease.target_hash,lease.desired_hash,provider],
            )?;
            if changed != 1 {
                return Ok(Dispatch::LeaseLost);
            }
        }
        tx.execute(
            "UPDATE usage_reservations SET state='dispatched',dispatched_at=?2 WHERE id=?1",
            params![reservation.0, now],
        )?;
        tx.commit()?;
        Ok(Dispatch::Recorded)
    })
}

/// Releases only proven-undispatched work. A raced dispatch wins irrevocably;
/// an expired lease may release its own old reservation without touching jobs.
pub(crate) fn release(store: &Store, reservation: &Reservation) -> Result<bool, Error> {
    super::storage::transaction(store, super::storage::Kind::Control, |tx| {
        let changed = tx.execute(
            "UPDATE usage_reservations SET state='released' WHERE id=?1 AND state='reserved'",
            [&reservation.0],
        )? == 1;
        tx.commit()?;
        Ok(changed)
    })
}

/// Settles once on the reserved day, even if the response arrives after midnight.
/// None records the full reservation as an explicitly estimated token count.
/// Provider-reported overruns are recorded honestly and block further admission.
pub(crate) fn reconcile(
    store: &Store,
    reservation: &Reservation,
    actual_tokens: Option<i64>,
) -> Result<bool, Error> {
    if actual_tokens.is_some_and(|n| n < 0) {
        return Err(Error::Invalid("reported token usage"));
    }
    super::storage::transaction(store, super::storage::Kind::Control, |tx| {
        let row = tx
        .query_row(
            "SELECT provider,day,tokens FROM usage_reservations WHERE id=?1 AND state='dispatched'",
            [&reservation.0],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, i64>(2)?,
                ))
            },
        )
        .optional()?;
        let Some((provider, day, reserved)) = row else {
            return Ok(false);
        };
        let tokens = actual_tokens.unwrap_or(reserved);
        let estimated = if actual_tokens.is_none() { tokens } else { 0 };
        let prior: (i64, i64, i64) = tx
            .query_row(
                "SELECT calls,tokens,estimated_tokens FROM usage WHERE day=?1 AND provider=?2",
                params![day, provider],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?
            .unwrap_or((0, 0, 0));
        let add = |a: i64, b: i64| {
            a.checked_add(b)
                .ok_or(Error::Invalid("settled usage overflow"))
        };
        tx.execute("INSERT INTO usage(day,provider,calls,tokens,estimated_tokens) VALUES(?1,?2,?3,?4,?5)
        ON CONFLICT(day,provider) DO UPDATE SET calls=excluded.calls,tokens=excluded.tokens,estimated_tokens=excluded.estimated_tokens",
        params![day,provider,add(prior.0,1)?,add(prior.1,tokens)?,add(prior.2,estimated)?])?;
        tx.execute(
            "UPDATE usage_reservations SET state='reconciled',tokens=MAX(tokens,?2) WHERE id=?1",
            params![reservation.0, tokens],
        )?;
        tx.commit()?;
        Ok(true)
    })
}

#[cfg(test)]
mod tests;
