//! Consistent persisted queue and usage inspection; see `v1.md` §8 and §13.

use std::collections::BTreeMap;

use serde::Serialize;

use super::{Error, capacity, retry};
use crate::store::Store;

/// Stored states and scheduling hints; done includes discarded obsolete results.
#[derive(Debug, Default, Serialize)]
pub(crate) struct Counts {
    queued: i64,
    leased: i64,
    done: i64,
    failed: i64,
    dead: i64,
    ready: i64,
    deferred: i64,
    expired_leases: i64,
    next_ready_at: Option<i64>,
}

/// Settled usage today and unresolved reservations across all days.
#[derive(Debug, Default, Serialize)]
pub(crate) struct Provider {
    jobs: Counts,
    calls_today: i64,
    tokens_today: i64,
    estimated_tokens_today: i64,
    reserved_calls: i64,
    reserved_tokens: i64,
    uncertain_calls: i64,
    uncertain_tokens: i64,
    deferrals: BTreeMap<String, Deferral>,
}

/// A safe scheduling category with its earliest eligible time, not a provider error body.
#[derive(Debug, Serialize)]
struct Deferral {
    jobs: i64,
    earliest_at: Option<i64>,
    earliest_utc: Option<String>,
}

/// Bounded job details for selecting explicit retries without exposing source text
/// or stored provider error bodies. Ordering puts failed/dead work first.
#[derive(Debug, Serialize)]
struct JobEntry {
    id: i64,
    kind: String,
    target_kind: String,
    target_id: i64,
    state: String,
    attempts: i64,
    retry_allowed: bool,
    regeneration_allowed: bool,
    not_before: Option<i64>,
    reason: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    http_status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider_code: Option<&'static str>,
}

/// One database snapshot, without recovering leases or changing accounting.
#[derive(Debug, Serialize)]
pub(crate) struct Snapshot {
    pub(crate) jobs: Counts,
    #[serde(skip_serializing_if = "Option::is_none")]
    wal: Option<super::storage::Status>,
    database: crate::store::capacity::Status,
    enrichment_claims: capacity::EnrichmentClaims,
    observed_at: i64,
    usage_day: String,
    providers: BTreeMap<String, Provider>,
    #[serde(skip_serializing_if = "Option::is_none")]
    entries: Option<Vec<JobEntry>>,
}

/// Raw queue observation retained internally; use health() for safe doctor output.
/// Provider names and job error text are intentionally absent from the health type.
#[derive(Debug, Serialize)]
pub(crate) struct Observation {
    // Preserve invalid row evidence before reservation SUM can hide its sign.
    #[serde(skip)]
    negative_reservation_counter: bool,
    jobs: Counts,
    observed_at: i64,
    usage_day: String,
    providers: BTreeMap<String, Provider>,
    #[serde(skip_serializing_if = "Option::is_none")]
    entries: Option<Vec<JobEntry>>,
}

impl Counts {
    /// True only when no unfinished or dead jobs remain, including providers
    /// unavailable to the current drain host. Completed/discarded history stays.
    pub(crate) fn is_quiescent(&self) -> bool {
        self.queued == 0 && self.leased == 0 && self.failed == 0 && self.dead == 0
    }

    fn add(
        &mut self,
        state: &str,
        count: i64,
        ready: i64,
        deferred: i64,
        expired: i64,
        next: Option<i64>,
    ) {
        match state {
            "queued" => self.queued += count,
            "leased" => self.leased += count,
            "done" => self.done += count,
            "failed" => self.failed += count,
            "dead" => self.dead += count,
            _ => {}
        }
        self.ready += ready;
        self.deferred += deferred;
        self.expired_leases += expired;
        if let Some(next) = next {
            self.next_ready_at = Some(self.next_ready_at.map_or(next, |old| old.min(next)));
        }
    }
}

impl Snapshot {
    /// Renders scheduling and charge uncertainty for terminal users.
    pub(crate) fn summary(&self) -> String {
        let jobs = &self.jobs;
        let mut lines = vec![format!(
            "Queue: {} queued, {} leased, {} failed, {} dead, {} done (includes discarded results); {} ready, {} deferred, {} expired leases",
            jobs.queued,
            jobs.leased,
            jobs.failed,
            jobs.dead,
            jobs.done,
            jobs.ready,
            jobs.deferred,
            jobs.expired_leases
        )];
        lines.push(self.database.summary());
        lines.push(self.enrichment_claims.summary());
        if let Some(wal) = &self.wal {
            lines.push(wal.summary());
        }
        if let Some(next) = jobs.next_ready_at {
            lines.push(format!(
                "Next deferred work eligible at Unix timestamp {next}"
            ));
        }
        for (name, provider) in &self.providers {
            lines.push(format!(
                "{name}: today {} calls / {} tokens ({} estimated); reserved {} calls / {} tokens; uncertain {} calls / {} tokens across all days",
                provider.calls_today, provider.tokens_today, provider.estimated_tokens_today,
                provider.reserved_calls, provider.reserved_tokens,
                provider.uncertain_calls, provider.uncertain_tokens
            ));
            for (category, deferral) in &provider.deferrals {
                if category.starts_with("cap_") {
                    let reason = match category.as_str() {
                        "cap_vectors" => "max_l2_units cap",
                        "cap_summaries" => "max_l3_units cap",
                        "cap_database" => "database growth soft cap",
                        "cap_storage" => "SQLite storage full",
                        "cap_wal" => "WAL capacity exhausted",
                        _ => "embedding document or batch token cap",
                    };
                    lines.push(format!(
                        "{name}: {} jobs paused: {reason}; rechecked on the next provider run",
                        deferral.jobs
                    ));
                    continue;
                }
                let reason = match category.as_str() {
                    "daily_budget" => "daily budget exhausted",
                    "provider_rate" => "provider rate limit",
                    "retry" => "transient request retry",
                    _ => "deferred work",
                };
                let at = deferral.earliest_utc.clone().unwrap_or_else(|| {
                    format!(
                        "Unix timestamp {}",
                        deferral.earliest_at.unwrap_or_default()
                    )
                });
                lines.push(format!(
                    "{name}: {} jobs paused: {reason}; earliest eligible {at}",
                    deferral.jobs
                ));
            }
        }
        if let Some(entries) = &self.entries {
            let total = jobs.queued + jobs.leased + jobs.failed + jobs.dead + jobs.done;
            lines.push(format!(
                "Jobs: showing {} of {total} (failed/dead first)",
                entries.len()
            ));
            for entry in entries {
                lines.push(format!(
                    "{}: {} {} {}:{} · {} attempts · {} · explicit retry {} · regeneration {} · not_before {}",
                    entry.id,
                    entry.state,
                    entry.kind,
                    entry.target_kind,
                    entry.target_id,
                    entry.attempts,
                    entry.reason,
                    if entry.retry_allowed {
                        "allowed"
                    } else {
                        "unavailable"
                    },
                    if entry.regeneration_allowed {
                        "allowed"
                    } else {
                        "unavailable"
                    },
                    entry
                        .not_before
                        .map_or_else(|| "capacity pause".into(), |at| at.to_string())
                ));
            }
        }
        lines.join("\n")
    }
}

/// Reads one consistent snapshot; readiness excludes exhausted attempts and expired leases.
pub(crate) fn snapshot(store: &Store, now: i64) -> Result<Snapshot, Error> {
    snapshot_with_jobs(store, now, None)
}

/// Adds at most 1,000 job IDs to the same read transaction as the counters.
/// Omission preserves the compact status contract; inspection never recovers work.
pub(crate) fn snapshot_with_jobs(
    store: &Store,
    now: i64,
    limit: Option<u16>,
) -> Result<Snapshot, Error> {
    validate(now, limit)?;
    let mut connection = store.writer()?;
    let transaction = connection.transaction()?;
    let wal = super::storage::status(&transaction, store)?;
    let database = crate::store::capacity::status(&transaction, store)?;
    let enrichment_claims = capacity::enrichment_claim_status(store, &transaction, now)?;
    let Observation {
        negative_reservation_counter: _,
        jobs,
        observed_at,
        usage_day,
        providers,
        entries,
    } = read_in(&transaction, now, limit)?;
    transaction.commit()?;
    Ok(Snapshot {
        jobs,
        wal,
        database,
        enrichment_claims,
        observed_at,
        usage_day,
        providers,
        entries,
    })
}

fn validate(now: i64, limit: Option<u16>) -> Result<(), Error> {
    if limit.is_some_and(|limit| limit == 0 || limit > 1000) {
        return Err(Error::Invalid("job listing limit must be 1..=1000"));
    }
    if now < 0 {
        return Err(Error::Invalid("negative status timestamp"));
    }
    Ok(())
}

/// Reads queue evidence in a caller-owned transaction without opening, committing,
/// recovering leases or changing usage. Doctor supplies a read-only connection;
/// ordinary status supplies its existing transaction and preserves its JSON shape.
pub(crate) fn read_in(
    transaction: &rusqlite::Transaction<'_>,
    now: i64,
    limit: Option<u16>,
) -> Result<Observation, Error> {
    validate(now, limit)?;
    let day: Option<String> =
        transaction.query_row("SELECT date(?1, 'unixepoch')", [now], |row| row.get(0))?;
    let mut result = Observation {
        negative_reservation_counter: false,
        jobs: Counts::default(),
        observed_at: now,
        usage_day: day.ok_or(Error::Invalid("status timestamp out of range"))?,
        providers: BTreeMap::new(),
        entries: None,
    };
    {
        let mut query = transaction.prepare(
            "SELECT provider, state, COUNT(*),
             SUM(state IN ('queued','failed') AND attempts < 8 AND not_before <= ?1),
             SUM(state IN ('queued','failed') AND attempts < 8 AND not_before > ?1),
             SUM(state = 'leased' AND COALESCE(leased_until <= ?1, 0)),
             MIN(CASE WHEN state IN ('queued','failed') AND attempts < 8 AND not_before > ?1 AND not_before < 9223372036854775807 THEN not_before END)
             FROM jobs GROUP BY provider, state ORDER BY provider, state",
        )?;
        let mut rows = query.query([now])?;
        while let Some(row) = rows.next()? {
            let provider: String = row.get(0)?;
            let state: String = row.get(1)?;
            let (count, ready, deferred, expired, next) = (
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
                row.get(6)?,
            );
            result
                .jobs
                .add(&state, count, ready, deferred, expired, next);
            result
                .providers
                .entry(provider)
                .or_default()
                .jobs
                .add(&state, count, ready, deferred, expired, next);
        }
    }
    {
        let mut query = transaction.prepare(
            "SELECT provider, calls, tokens, estimated_tokens FROM usage WHERE day = ?1",
        )?;
        let mut rows = query.query([&result.usage_day])?;
        while let Some(row) = rows.next()? {
            let provider = result.providers.entry(row.get(0)?).or_default();
            provider.calls_today = row.get(1)?;
            provider.tokens_today = row.get(2)?;
            provider.estimated_tokens_today = row.get(3)?;
        }
    }
    {
        let mut query = transaction.prepare("SELECT provider, state, SUM(calls), SUM(tokens), MAX(calls < 0 OR tokens < 0) FROM usage_reservations WHERE state IN ('reserved','dispatched') GROUP BY provider, state")?;
        let mut rows = query.query([])?;
        while let Some(row) = rows.next()? {
            result.negative_reservation_counter |= row.get::<_, bool>(4)?;
            let provider = result.providers.entry(row.get(0)?).or_default();
            if row.get::<_, String>(1)? == "reserved" {
                provider.reserved_calls = row.get(2)?;
                provider.reserved_tokens = row.get(3)?;
            } else {
                provider.uncertain_calls = row.get(2)?;
                provider.uncertain_tokens = row.get(3)?;
            }
        }
    }
    {
        let mut query = transaction.prepare(
            "SELECT provider, CASE last_error WHEN ?2 THEN 'daily_budget' WHEN ?3 THEN 'provider_rate'
             WHEN ?4 THEN 'retry' WHEN ?5 THEN 'cap_vectors' WHEN ?6 THEN 'cap_database'
             WHEN ?7 THEN 'cap_tokens' WHEN ?8 THEN 'cap_database' WHEN ?9 THEN 'cap_storage' WHEN ?10 THEN 'cap_wal' WHEN ?11 THEN 'cap_summaries' ELSE 'other' END AS category, COUNT(*), NULLIF(MIN(not_before), 9223372036854775807),
             strftime('%Y-%m-%d %H:%M:%S UTC', MIN(not_before), 'unixepoch')
             FROM jobs WHERE state IN ('queued','failed') AND attempts < 8 AND not_before > ?1
             GROUP BY provider, category ORDER BY provider, category",
        )?;
        let mut rows = query.query(rusqlite::params![
            now,
            retry::BUDGET_REASON,
            retry::RATE_REASON,
            retry::RETRY_REASON,
            capacity::Reason::Vectors.message(),
            capacity::Reason::Database.message(),
            capacity::Reason::Tokens.message(),
            capacity::LEGACY_DATABASE_REASON,
            capacity::Reason::Storage.message(),
            capacity::Reason::Wal.message(),
            capacity::Reason::Summaries.message()
        ])?;
        while let Some(row) = rows.next()? {
            result
                .providers
                .entry(row.get(0)?)
                .or_default()
                .deferrals
                .insert(
                    row.get(1)?,
                    Deferral {
                        jobs: row.get(2)?,
                        earliest_at: row.get(3)?,
                        earliest_utc: row.get(4)?,
                    },
                );
        }
    }
    if let Some(limit) = limit {
        let mut query = transaction.prepare(&format!(
            "SELECT id,kind,target_kind,target_id,state,attempts,not_before,last_error,({}) FROM jobs
             ORDER BY CASE WHEN state IN ('failed','dead') THEN 0 ELSE 1 END, id LIMIT ?1",
            super::capacity::REPLACEMENT
        ))?;
        let entries = query
            .query_map([limit], |row| {
                let kind: String = row.get(1)?;
                let target_kind: String = row.get(2)?;
                let target_id: i64 = row.get(3)?;
                let state: String = row.get(4)?;
                let attempts: i64 = row.get(5)?;
                let at: i64 = row.get(6)?;
                let reason: Option<String> = row.get(7)?;
                let owner_bound: bool = row.get(8)?;
                Ok(JobEntry {
                    id: row.get(0)?,
                    regeneration_allowed: state == "done"
                        && attempts < 8
                        && matches!(
                            (kind.as_str(), target_kind.as_str()),
                            ("enrich:describe_symbol", "symbol")
                                | ("enrich:summarize_pr", "pr")
                                | ("enrich:summarize_module", "unit")
                        )
                        && owner_bound,
                    retry_allowed: matches!(state.as_str(), "failed" | "dead") && attempts < 8,
                    kind,
                    target_kind,
                    target_id,
                    state,
                    attempts,
                    not_before: (at != capacity::PAUSED).then_some(at),
                    http_status: reason.as_deref().and_then(retry::chat_http_status),
                    provider_code: reason
                        .as_deref()
                        .and_then(retry::chat_http_detail)
                        .and_then(|(_, detail)| detail.map(|detail| detail.code())),
                    reason: match reason.as_deref() {
                        None => "none",
                        Some(retry::BUDGET_REASON) => "daily_budget",
                        Some(retry::RATE_REASON) => "provider_rate",
                        Some(retry::RETRY_REASON) => "retry",
                        Some("permanent provider error") => "permanent_provider_error",
                        Some(reason) if retry::chat_http_status(reason).is_some() => {
                            "chat_http_error"
                        }
                        Some("invalid chat request") => "chat_invalid_request",
                        Some("chat provider unauthorized") => "chat_unauthorized",
                        Some("chat response body limit") => "chat_response_body_limit",
                        Some("invalid chat response") => "chat_invalid_response",
                        Some("empty provider content; diagnosis required") => "empty_content",
                        Some("attempt limit reached" | "attempt limit reached during recovery") => {
                            "attempt_limit"
                        }
                        Some("lease expired; outcome uncertain") => "expired_lease",
                        Some(reason) if reason == capacity::Reason::Vectors.message() => {
                            "cap_vectors"
                        }
                        Some(reason) if reason == capacity::Reason::Summaries.message() => {
                            "cap_summaries"
                        }
                        Some(reason) if reason == capacity::Reason::Tokens.message() => {
                            "cap_tokens"
                        }
                        Some(reason)
                            if reason == capacity::Reason::Database.message()
                                || reason == capacity::LEGACY_DATABASE_REASON =>
                        {
                            "cap_database"
                        }
                        Some(reason) if reason == capacity::Reason::Wal.message() => "cap_wal",
                        Some(reason) if reason == capacity::Reason::Storage.message() => {
                            "cap_storage"
                        }
                        Some(_) => "other",
                    },
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        result.entries = Some(entries);
    }
    Ok(result)
}

/// Numeric persisted job evidence; queued/leased counts do not prove worker liveness.
#[derive(Debug, Serialize)]
pub(crate) struct HealthCounts {
    pub(crate) queued: u64,
    pub(crate) leased: u64,
    pub(crate) done: u64,
    pub(crate) failed: u64,
    pub(crate) dead: u64,
    pub(crate) ready: u64,
    pub(crate) deferred: u64,
    pub(crate) expired_leases: u64,
    pub(crate) next_ready_at: Option<i64>,
}

/// Closed scheduling categories keep provider identifiers and stored errors private.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum HealthCategory {
    DailyBudget,
    ProviderRate,
    Retry,
    CapVectors,
    CapDatabase,
    CapTokens,
    CapStorage,
    CapWal,
    Other,
}
impl HealthCategory {
    fn from_normalized(category: &str) -> Self {
        match category {
            "daily_budget" => Self::DailyBudget,
            "provider_rate" => Self::ProviderRate,
            "retry" => Self::Retry,
            "cap_vectors" => Self::CapVectors,
            "cap_database" => Self::CapDatabase,
            "cap_tokens" => Self::CapTokens,
            "cap_storage" => Self::CapStorage,
            "cap_wal" => Self::CapWal,
            _ => Self::Other,
        }
    }
}

/// Aggregated deferrals; absent eligibility for cap sentinels is not a promised date.
#[derive(Debug, Default, Serialize)]
pub(crate) struct HealthDeferral {
    pub(crate) jobs: u64,
    pub(crate) earliest_at: Option<i64>,
}

/// Safe cross-provider health evidence; reservations/uncertainty are not asserted leaks.
#[derive(Debug, Serialize)]
pub(crate) struct Health {
    pub(crate) observed_at: i64,
    pub(crate) jobs: HealthCounts,
    pub(crate) calls_today: u64,
    pub(crate) tokens_today: u64,
    pub(crate) estimated_tokens_today: u64,
    pub(crate) reserved_calls: u64,
    pub(crate) reserved_tokens: u64,
    pub(crate) uncertain_calls: u64,
    pub(crate) uncertain_tokens: u64,
    pub(crate) deferrals: BTreeMap<HealthCategory, HealthDeferral>,
}

fn nonnegative(value: i64) -> Result<u64, Error> {
    u64::try_from(value).map_err(|_| Error::Invalid("negative queue health counter"))
}
fn add_health(total: &mut u64, value: i64) -> Result<(), Error> {
    *total = total
        .checked_add(nonnegative(value)?)
        .ok_or(Error::Invalid("queue health counter overflow"))?;
    Ok(())
}

impl Observation {
    /// Aggregates counts without serializing provider names, source text, job IDs or errors.
    /// Invalid persisted numeric counters refuse a health claim instead of wrapping totals.
    pub(crate) fn health(&self) -> Result<Health, Error> {
        if self.negative_reservation_counter {
            return Err(Error::Invalid("negative queue health counter"));
        }
        let jobs = &self.jobs;
        let mut result = Health {
            observed_at: self.observed_at,
            jobs: HealthCounts {
                queued: nonnegative(jobs.queued)?,
                leased: nonnegative(jobs.leased)?,
                done: nonnegative(jobs.done)?,
                failed: nonnegative(jobs.failed)?,
                dead: nonnegative(jobs.dead)?,
                ready: nonnegative(jobs.ready)?,
                deferred: nonnegative(jobs.deferred)?,
                expired_leases: nonnegative(jobs.expired_leases)?,
                next_ready_at: jobs.next_ready_at,
            },
            calls_today: 0,
            tokens_today: 0,
            estimated_tokens_today: 0,
            reserved_calls: 0,
            reserved_tokens: 0,
            uncertain_calls: 0,
            uncertain_tokens: 0,
            deferrals: BTreeMap::new(),
        };
        for provider in self.providers.values() {
            add_health(&mut result.calls_today, provider.calls_today)?;
            add_health(&mut result.tokens_today, provider.tokens_today)?;
            add_health(
                &mut result.estimated_tokens_today,
                provider.estimated_tokens_today,
            )?;
            add_health(&mut result.reserved_calls, provider.reserved_calls)?;
            add_health(&mut result.reserved_tokens, provider.reserved_tokens)?;
            add_health(&mut result.uncertain_calls, provider.uncertain_calls)?;
            add_health(&mut result.uncertain_tokens, provider.uncertain_tokens)?;
            for (category, deferral) in &provider.deferrals {
                let entry = result
                    .deferrals
                    .entry(HealthCategory::from_normalized(category))
                    .or_default();
                add_health(&mut entry.jobs, deferral.jobs)?;
                if let Some(earliest) = deferral.earliest_at {
                    entry.earliest_at =
                        Some(entry.earliest_at.map_or(earliest, |old| old.min(earliest)));
                }
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
