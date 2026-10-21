//! Bounded queue batches using shared writer admission; v1 §3.5/§8.2.
use super::Error;
use crate::store::Store;
pub(super) use crate::store::admission::{Kind, transaction};
pub(crate) use crate::store::admission::{Status, status};
use rusqlite::Transaction;

#[cfg(test)]
use crate::store::wal;
#[cfg(test)]
use rusqlite::Connection;

#[cfg(test)]
mod tests;

/// Fixed keyset window for a bulk update; eligibility is checked in each
/// transaction. Keys beyond the starting upper boundary wait for a later pass.
pub(super) struct Window<K = i64> {
    pub(super) after: K,
    pub(super) through: K,
    pub(super) limit: u32,
    pub(super) now: i64,
}

/// Commits bounded job batches and reports durable progress if a later batch
/// fails. Shrinks only a rolled-back batch that actually hit its WAL allowance;
/// preflight backpressure never replays the callback. Callbacks perform SQL only
/// and return selected IDs plus changed count, without committing. A short final
/// batch completes without requiring another write reservation for an empty scan.
pub(super) fn job_batches(
    store: &Store,
    clock: impl FnOnce() -> i64,
    update: impl FnMut(&Transaction<'_>, &Window) -> Result<(Vec<i64>, usize), Error>,
) -> Result<usize, Error> {
    row_batches(store, Rows::Jobs, clock, update)
}

/// Only these schema-owned row keys may define a finite bulk scan.
pub(super) enum Rows {
    Jobs,
    Reservations,
}

/// Applies bounded transactions using stable primary keys for either table.
pub(super) fn row_batches<K: Clone + Default + PartialOrd + rusqlite::types::FromSql>(
    store: &Store,
    rows: Rows,
    clock: impl FnOnce() -> i64,
    mut update: impl FnMut(&Transaction<'_>, &Window<K>) -> Result<(Vec<K>, usize), Error>,
) -> Result<usize, Error> {
    let mut clock = Some(clock);
    let mut boundary = None;
    let mut after = K::default();
    let mut limit = 64_u32;
    let mut changed = 0_usize;
    loop {
        let mut ran = false;
        let result = transaction(store, Kind::Control, |tx| {
            ran = true;
            let (through, now) = match boundary.clone() {
                Some(value) => value,
                None => {
                    let now = clock
                        .take()
                        .ok_or(Error::Invalid("bulk clock already sampled"))?(
                    );
                    if now < 0 || now == i64::MAX {
                        return Err(Error::Invalid("bulk timestamp"));
                    }
                    let through: K = tx.query_row(
                        match rows {
                            Rows::Jobs => "SELECT COALESCE(MAX(id),0) FROM jobs",
                            Rows::Reservations => {
                                "SELECT COALESCE(MAX(id),'') FROM usage_reservations"
                            }
                        },
                        [],
                        |r| r.get(0),
                    )?;
                    boundary = Some((through.clone(), now));
                    (through, now)
                }
            };
            let (ids, count) = update(
                &tx,
                &Window {
                    after: after.clone(),
                    through: through.clone(),
                    limit,
                    now,
                },
            )?;
            let last = ids.last().cloned();
            let finished = ids.len() < limit as usize || last.as_ref() == Some(&through);
            if last
                .as_ref()
                .is_some_and(|id| id <= &after || id > &through)
                || (last.is_none() && count != 0)
            {
                return Err(Error::Invalid("non-advancing bulk cursor"));
            }
            tx.commit()?;
            Ok((last, count, finished))
        });
        match result {
            Ok((last, count, finished)) => {
                changed = changed
                    .checked_add(count)
                    .ok_or(Error::Invalid("bulk count overflow"))?;
                if finished {
                    return Ok(changed);
                }
                match last {
                    Some(last) => after = last,
                    None => return Ok(changed),
                }
            }
            Err(Error::Store(crate::store::Error::WalCapacity { .. })) if ran && limit > 1 => {
                limit /= 2;
            }
            Err(error) if changed == 0 => return Err(error),
            Err(error) => {
                return Err(Error::BulkIncomplete {
                    changed,
                    source: Box::new(error),
                });
            }
        }
    }
}

/// Returns one atomic batch, shrinking only after a native WAL overflow rolled
/// the whole attempt back. The callback must contain SQL and fresh clock/fence
/// checks only; no network or externally visible work may be replayed.
pub(super) fn claim_batch<T>(
    store: &Store,
    max_items: usize,
    mut action: impl FnMut(Transaction<'_>, usize) -> Result<T, Error>,
) -> Result<T, Error> {
    let mut limit = max_items.min(64);
    loop {
        let mut ran = false;
        let result = transaction(store, Kind::Growth, |tx| {
            ran = true;
            action(tx, limit)
        });
        match result {
            Err(Error::Store(crate::store::Error::WalCapacity { .. })) if ran && limit > 1 => {
                limit /= 2;
            }
            result => return result,
        }
    }
}
