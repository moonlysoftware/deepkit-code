//! Verified symbol.history presentation including current PR pointers; v1.md §10.1.
use super::*;
use rusqlite::OptionalExtension;
use std::collections::{BTreeMap, BTreeSet};

/// Includes active histories after selecting the ranked cohort once per response.
pub(super) fn populate_many(
    context: &Context,
    symbols: &mut [navigation::SymbolResult],
    allowed: &BTreeSet<i64>,
    work: sync::l1::Work<'_>,
) -> Result<()> {
    for symbol in symbols.iter_mut() {
        symbol.history = None;
    }
    if !context.config.commits.enabled || context.config.enrich.top_symbols == 0 {
        return Ok(());
    }
    let check = || work.check().map_err(|_| store::Error::L1InputsChanged);
    context.store.checked_read_snapshot(&check, |tx| {
        let requested = symbols
            .iter()
            .map(|symbol| symbol.selected.id)
            .filter(|id| allowed.contains(id))
            .collect::<BTreeSet<_>>();
        if requested.is_empty() {
            return Ok::<_, Error>(());
        }
        // Most navigation responses have no collated history. Avoid walking
        // the full top-symbol rank prefix twice when none of the selected
        // definitions can contribute a history object.
        let requested_ids = serde_json::json!(requested).to_string();
        let has_history: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM symbols WHERE id IN \
             (SELECT value FROM json_each(?1)) AND history_unit IS NOT NULL)",
            [requested_ids],
            |row| row.get(0),
        )?;
        if !has_history {
            return Ok::<_, Error>(());
        }
        let targets = store::symbol_history::targets(tx, context.config.enrich.top_symbols)?
            .into_iter()
            .filter(|target| requested.contains(&target.id) && target.history.is_some())
            .collect::<Vec<_>>();
        let inventories = store::symbol_history::inventory(tx, &targets, &check)?;
        let selected = targets
            .iter()
            .zip(&inventories)
            .map(|(target, entries)| (target.id, (target, entries)))
            .collect::<BTreeMap<_, _>>();
        for symbol in symbols {
            check()?;
            let Some((target, entries)) = selected.get(&symbol.selected.id) else {
                continue;
            };
            let unit = target
                .history
                .ok_or(super::super::Error::OpenSourceUnavailable)?;
            if queue::source::resolve_freshness(
                tx,
                &context.root,
                &context.config_root,
                &context.config,
                unit,
                work,
            )?
            .is_none()
            {
                return Err(super::super::Error::OpenSourceUnavailable.into());
            }
            let mut commits = Vec::new();
            for entry in *entries {
                check()?;
                let sha = entry
                    .sha
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>();
                let date = gix::date::Time::new(entry.time, 0)
                    .format(gix::date::time::format::SHORT)
                    .map_err(|_| super::super::Error::OpenSourceUnavailable)?;
                let mut item = json!({"sha":sha,"date":date,"subject":entry.subject});
                if context.config.commits.prs {
                    let pr: Option<(i64, String)> = tx
                        .query_row(
                            "SELECT p.unit_id,p.title FROM commits c JOIN prs p ON p.id=c.pr_id WHERE c.id=?1 AND p.unit_id IS NOT NULL",
                            [entry.id],
                            |r| Ok((r.get(0)?, r.get(1)?)),
                        )
                        .optional()?;
                    if let Some((id, title)) = pr {
                        if queue::source::resolve_freshness(
                            tx,
                            &context.root,
                            &context.config_root,
                            &context.config,
                            id,
                            work,
                        )?
                        .is_none()
                        {
                            return Err(super::super::Error::OpenSourceUnavailable.into());
                        }
                        let pr = store::unit_by_id_in(tx, id)?
                            .ok_or(super::super::Error::OpenSourceUnavailable)?;
                        item["pr"] = json!({"id":navigation::public_id(&pr),"title":title});
                    }
                }
                commits.push(item);
            }
            symbol.history = Some(json!({"commits":commits}));
        }
        Ok::<_, Error>(())
    })??;
    Ok(())
}

/// Retains the rank prefix that fits before optional histories consume tokens.
///
/// Applying the response cap before source and graph enrichment prevents broad
/// names from doing work for symbols that cannot appear in the response. The
/// caller repeats this after enrichment because documentation and links can
/// make the retained objects larger.
pub(super) fn truncate_to_candidates(
    symbols: &mut Vec<navigation::SymbolResult>,
    maximum: u64,
    work: sync::l1::Work<'_>,
) -> Result<(BTreeSet<i64>, bool)> {
    let retained = rank_prefix_len(symbols, maximum, work)?;
    let truncated = retained < symbols.len();
    symbols.truncate(retained);
    Ok((
        symbols.iter().map(|symbol| symbol.selected.id).collect(),
        truncated,
    ))
}

fn rank_prefix_len<T: serde::Serialize>(
    symbols: &[T],
    maximum: u64,
    work: sync::l1::Work<'_>,
) -> Result<usize> {
    let fits = |length: usize| -> Result<bool> {
        check_query_work(work)?;
        Ok(
            crate::treat::count_tokens(&serde_json::to_string(&symbols[..length])?) as u64
                <= maximum,
        )
    };
    if symbols.is_empty() || fits(symbols.len())? {
        return Ok(symbols.len());
    }
    // A JSON object always adds bytes and structural tokens to this array, so
    // prefix token counts are monotonic. Find the largest fitting prefix in
    // O(log n) counts instead of serializing n, n-1, ... objects.
    let mut fitting = 0;
    let mut excluded = symbols.len();
    while fitting + 1 < excluded {
        let middle = fitting + (excluded - fitting) / 2;
        if fits(middle)? {
            fitting = middle;
        } else {
            excluded = middle;
        }
    }
    Ok(fitting)
}

/// Keep complete history entries and symbol objects within the response cap.
/// Drop oldest history entries first, making truncation explicit on that history.
pub(super) fn limit(
    mut response: Value,
    maximum: u64,
    work: sync::l1::Work<'_>,
) -> Result<(Value, bool)> {
    let mut truncated = false;
    while response.as_array().is_some_and(|rows| !rows.is_empty())
        && crate::treat::count_tokens(&response.to_string()) as u64 > maximum
    {
        check_query_work(work)?;
        truncated = true;
        let Some(rows) = response.as_array_mut() else {
            break;
        };
        let Some(last) = rows.last_mut() else { break };
        let count = last["history"]["commits"].as_array().map_or(0, Vec::len);
        if count > 0 {
            if last["history"]["total_commits"].is_null() {
                last["history"]["total_commits"] = json!(count);
            }
            last["history"]["truncated"] = json!(true);
            if let Some(commits) = last["history"]["commits"].as_array_mut() {
                commits.remove(0);
            }
        } else {
            rows.pop();
        }
    }
    Ok((response, truncated))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logarithmic_prefix_matches_exhaustive_token_caps() {
        let rows = (0..100)
            .map(|index| {
                json!({
                    "id": format!("s_{index:012}"),
                    "path": format!("src/module_{index}/implementation.rs"),
                    "signature": format!("fn shared_name_{index}(value: usize) -> Result<()>"),
                    "doc": "Unicode café and a deliberately uneven payload ".repeat(index % 7),
                    "rank": 100 - index,
                })
            })
            .collect::<Vec<_>>();
        let counts = (0..=rows.len())
            .map(|length| {
                crate::treat::count_tokens(&serde_json::to_string(&rows[..length]).unwrap()) as u64
            })
            .collect::<Vec<_>>();
        let mut limits = counts
            .iter()
            .flat_map(|count| [count.saturating_sub(1), *count, count.saturating_add(1)])
            .collect::<BTreeSet<_>>();
        limits.insert(0);
        for maximum in limits {
            let exhaustive = counts
                .iter()
                .enumerate()
                .filter(|(_, count)| **count <= maximum)
                .map(|(length, _)| length)
                .max()
                .unwrap_or(0);
            assert_eq!(
                rank_prefix_len(&rows, maximum, sync::l1::Work::default()).unwrap(),
                exhaustive,
                "maximum={maximum}"
            );
        }
    }
}
