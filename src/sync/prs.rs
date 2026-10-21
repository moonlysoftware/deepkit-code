//! Bounded marker detection and atomic PR collation; v1.md §4.4/§6.6.
mod ancestry;
use super::{
    history::{self, Error},
    l1::Work,
};
use crate::{
    config::{Config, pr_patterns},
    store::{self, FtsColumns, Store, UnitInsert, UnitKind},
    treat::{
        self,
        fts::{FtsInput, build_columns},
    },
};
use rusqlite::OptionalExtension;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

/// Retained progress distinguishes recognition, capacity and ownership fallback.
#[derive(Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub(crate) struct Report {
    pub(crate) state: String,
    pub(crate) visited: u64,
    pub(crate) detected: u64,
    pub(crate) inserted: u64,
    pub(crate) existing: u64,
    pub(crate) updated: u64,
    pub(crate) migration_visited: u64,
    pub(crate) retained: i64,
    pub(crate) deferred: BTreeMap<String, u64>,
}

/// Reconstructs latest markers after commit ingestion; no member is silently stolen
/// or fetched past the configured commit window/cap to inflate coverage.
pub(crate) fn ingest(
    root: &Path,
    settings: &Config,
    store: &Store,
    work: Work<'_>,
    validate: &dyn Fn() -> Result<(), store::Error>,
) -> Result<Report, Error> {
    let mut report = Report {
        state: "disabled".into(),
        ..Default::default()
    };
    if !settings.commits.enabled || !settings.commits.prs {
        return Ok(report);
    }
    let Some(repo) = super::head::discover(root)? else {
        report.state = "non_git".into();
        return Ok(report);
    };
    let Some(head) = super::head::resolve(&repo, root)? else {
        report.state = "unborn".into();
        return Ok(report);
    };
    let poll = || work.check().map_err(|_| store::Error::L1InputsChanged);
    let check = || {
        poll()?;
        validate()?;
        if super::head::read(root, work)
            .map_err(|_| store::Error::L1InputsChanged)?
            .map(|h| h.id)
            .as_deref()
            != Some(head.id.as_str())
        {
            return Err(store::Error::L1InputsChanged);
        }
        Ok(())
    };
    check()?;
    let key=blake3::hash(serde_json::json!({"head":head.id,"commits":settings.commits,"max_prs":settings.limits.max_prs,"max_pr_commits":settings.limits.max_pr_commits,"max_commits":settings.limits.max_commits,"max_file_bytes":settings.limits.max_file_bytes,"version":treat::pr::VERSION}).to_string().as_bytes()).to_hex().to_string();
    let (prior, status, count) = store.checked_read_snapshot(&check, |tx| {
        let get = |key: &str| {
            tx.query_row("SELECT value FROM meta WHERE key=?1", [key], |r| {
                r.get::<_, String>(0)
            })
            .optional()
        };
        Ok::<_, store::Error>((
            get("pr_completed")?,
            get("pr_status")?,
            tx.query_row("SELECT count(*) FROM prs", [], |r| r.get::<_, i64>(0))?,
        ))
    })??;
    if prior.as_deref() == Some(&key) {
        if let Some(status) = status {
            report = serde_json::from_str(&status)?;
        }
        report.state = if report.deferred.is_empty() {
            "not_needed"
        } else {
            "capped"
        }
        .into();
        report.visited = 0;
        report.inserted = 0;
        report.existing = 0;
        report.updated = 0;
        report.migration_visited = 0;
        return Ok(report);
    }
    report.retained = count;
    report.state = "pending".into();
    store.pr_progress(&serde_json::to_string(&report)?, None, &check)?;
    let patterns = pr_patterns::compile(&settings.commits.pr_patterns)
        .map_err(|_| Error::Git("compile PR patterns"))?;
    let id =
        gix::ObjectId::from_hex(head.id.as_bytes()).map_err(|_| Error::Git("decode PR HEAD"))?;
    let maximum = settings.limits.max_file_bytes()?.0;
    // Retained owners are independent of the current discovery window. A
    // collator version change revisits all of them before recording completion.
    let retained = store.checked_read_snapshot(&check, |tx| {
        Ok::<_, store::Error>(
            tx.prepare("SELECT merge_sha FROM prs ORDER BY id")?
                .query_map([], |r| r.get::<_, Vec<u8>>(0))?
                .collect::<Result<Vec<_>, _>>()?,
        )
    })??;
    // Immutable Git discovery and read-only SQLite snapshots only need bounded
    // work polling. Every publication and completion below still checks the
    // captured HEAD/configuration, then revalidates sources in its transaction.
    let mut seen = BTreeSet::new();
    let mut candidates = Vec::new();
    for sha in retained {
        poll()?;
        let id = gix::ObjectId::try_from(sha.as_slice())
            .map_err(|_| Error::Git("decode retained PR"))?;
        seen.insert(id);
        candidates.push((id, true));
    }
    for item in history::walk(&repo, id, false)?
        .take(settings.commits.max_count.min(usize::MAX as u64) as usize)
    {
        poll()?;
        report.visited += 1;
        let id = item.map_err(|_| Error::Git("walk PR candidates"))?.id;
        if seen.insert(id) {
            candidates.push((id, false));
        }
    }
    let mut migration_deferred = false;
    for (id, retained_candidate) in candidates {
        poll()?;
        report.migration_visited += u64::from(retained_candidate);
        let existing =
            store.checked_read_snapshot(&poll, |tx| store::prs::retained(tx, id.as_slice()))??;
        let header = repo
            .find_header(id)
            .map_err(|_| Error::Git("read PR marker header"))?;
        if header.size() > maximum {
            *report.deferred.entry("commit_metadata".into()).or_default() += 1;
            migration_deferred |= retained_candidate;
            continue;
        }
        let raw = repo
            .find_commit(id)
            .map_err(|_| Error::Git("read PR marker"))?;
        let decoded = raw.decode().map_err(|_| Error::Git("decode PR marker"))?;
        let message =
            std::str::from_utf8(decoded.message).map_err(|_| Error::Git("decode PR message"))?;
        let (subject, body) = message.split_once('\n').unwrap_or((message, ""));
        let body = body.trim_matches(['\n', '\r']);
        let parents = raw.parent_ids().map(|id| id.detach()).collect::<Vec<_>>();
        let Some(marker) = pr_patterns::detect(
            &patterns,
            parents.len(),
            subject.trim_end_matches('\r'),
            body,
        ) else {
            if retained_candidate {
                *report
                    .deferred
                    .entry("retained_marker_policy".into())
                    .or_default() += 1;
                migration_deferred = true;
            }
            continue;
        };
        report.detected += 1;
        if existing.is_none() && report.retained as u64 >= settings.limits.max_prs {
            *report.deferred.entry("max_prs".into()).or_default() += 1;
            continue;
        }
        let retained_members = if let Some(old) = &existing {
            let Some(unit) = old.unit else {
                return Err(store::Error::L1InputsChanged.into());
            };
            let members = store.checked_read_snapshot(&poll, |tx| {
                Store::pr_members(tx, old.id, unit, old.count as usize, &poll)
            })??;
            let Some(members) = members else {
                return Err(store::Error::L1InputsChanged.into());
            };
            Some(members)
        } else {
            None
        };
        let limit = settings.limits.max_pr_commits.min(500) as usize;
        let member_ids = if let Some(members) = &retained_members {
            Some(
                members
                    .iter()
                    .map(|m| {
                        gix::ObjectId::try_from(m.sha.as_slice())
                            .map_err(|_| Error::Git("decode retained member"))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else if marker.strategy == "squash" {
            if limit == 0 { None } else { Some(vec![id]) }
        } else {
            match ancestry::members(&repo, parents[0], parents[1], limit, maximum, work) {
                Err(Error::Bound(reason)) => {
                    *report.deferred.entry(reason.into()).or_default() += 1;
                    continue;
                }
                result => result?,
            }
        };
        let Some(member_ids) = member_ids else {
            *report.deferred.entry("max_pr_commits".into()).or_default() += 1;
            continue;
        };
        if member_ids.is_empty() {
            *report.deferred.entry("empty_members".into()).or_default() += 1;
            continue;
        }
        let members = store.checked_read_snapshot(&poll, |tx| {
            member_ids
                .iter()
                .map(|id| {
                    poll()?;
                    store::prs::member(tx, id.as_slice())
                })
                .collect::<Result<Option<Vec<_>>, store::Error>>()
        })??;
        let Some(members) = members else {
            *report
                .deferred
                .entry("members_not_retained".into())
                .or_default() += 1;
            continue;
        };
        if members
            .iter()
            .any(|m| m.owner.is_some() && m.owner != existing.as_ref().map(|old| old.id))
        {
            *report
                .deferred
                .entry("member_already_owned".into())
                .or_default() += 1;
            continue;
        }
        let mut symbols = members
            .iter()
            .flat_map(|m| m.symbols.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if let Some(old) = &existing {
            let text = store.checked_read_snapshot(&poll, |tx| tx.query_row("SELECT c.text FROM units u JOIN unit_contents c ON c.hash=u.content_hash WHERE u.id=?1", [old.unit], |r|r.get::<_,String>(0)).map_err(store::Error::from))??;
            let order = text
                .rsplit_once("\nsymbols:\n")
                .and_then(|(_, tail)| tail.rsplit_once('\n'))
                .map(|(order, _)| order)
                .ok_or(store::Error::L1InputsChanged)?;
            let preserved = order.lines().map(str::to_owned).collect::<Vec<_>>();
            let selected = preserved.iter().collect::<BTreeSet<_>>();
            let union = symbols.iter().collect::<BTreeSet<_>>();
            if preserved.len() != symbols.len().min(100)
                || selected.len() != preserved.len()
                || !selected.is_subset(&union)
            {
                return Err(store::Error::L1InputsChanged.into());
            }
            symbols = preserved;
        } else {
            let ranks = store.checked_read_snapshot(&poll, |tx| {
                let mut ranks = BTreeMap::new();
                let mut q = tx.prepare("SELECT coalesce(max(pagerank),0.0) FROM symbols WHERE coalesce(qualified,name)=?1")?;
                for symbol in &symbols {
                    poll()?;
                    ranks.insert(symbol.clone(), q.query_row([symbol], |r| r.get::<_, f64>(0))?);
                }
                Ok::<_, store::Error>(ranks)
            })??;
            symbols.sort_by(|a, b| ranks[b].total_cmp(&ranks[a]).then(a.cmp(b)));
            symbols.truncate(100);
        }
        let time = raw.time().map_err(|_| Error::Git("read PR merge time"))?;
        let date = time
            .format(gix::date::time::format::SHORT)
            .map_err(|_| Error::Git("format PR merge date"))?;
        if let Some(old) = &existing
            && (old.number != marker.number
                || old.title != marker.title
                || old.strategy != marker.strategy
                || old.time != time.seconds
                || old.base.as_deref() != parents.first().map(|p| p.as_slice())
                || old.head.as_slice()
                    != (if marker.strategy == "squash" {
                        id.as_slice()
                    } else {
                        parents[1].as_slice()
                    }))
        {
            return Err(store::Error::L1InputsChanged.into());
        }
        let collated = treat::pr::collate(&marker, &date, &members, &symbols);
        let columns = build_columns(
            FtsInput {
                breadcrumb: &collated.breadcrumb,
                header: &collated.header,
                text: &collated.text,
                language: None,
                kind: store::FileKind::Doc,
                extra_idents: &[],
                has_code_fence: false,
            },
            &settings.tokenize,
        );
        let source_key = format!("pr:{id}");
        let now = super::unix_timestamp();
        let unit = UnitInsert {
            hash: &collated.hash,
            source_key: Some(&source_key),
            duplicate_ordinal: 0,
            file_id: None,
            kind: UnitKind::PullRequest,
            byte_start: None,
            byte_end: None,
            line_start: None,
            line_end: None,
            scope_path: Some(&collated.breadcrumb),
            header: &collated.header,
            text: &collated.text,
            tokens: treat::count_tokens(&collated.text) as i64,
            ladder_level: 1,
            parent_unit: None,
            created_at: now,
            updated_at: now,
            fts: FtsColumns {
                header: &columns.header,
                idents: &columns.idents,
                body: &columns.body,
                prose: &columns.prose,
            },
        };
        let outcome = store.insert_pr(
            store::prs::Insert {
                marker: &marker,
                merge: id.as_slice(),
                base: parents.first().map(|id| id.as_slice()),
                head: if marker.strategy == "squash" {
                    id.as_slice()
                } else {
                    parents[1].as_slice()
                },
                time: time.seconds,
                members: &members,
                unit,
            },
            settings.limits.max_prs,
            &check,
        )?;
        match outcome {
            store::prs::Outcome::Inserted => {
                report.inserted += 1;
                report.retained += 1;
            }
            store::prs::Outcome::Existing => report.existing += 1,
            store::prs::Outcome::Updated => report.updated += 1,
            store::prs::Outcome::Capacity => {
                *report.deferred.entry("max_prs".into()).or_default() += 1
            }
            store::prs::Outcome::Owned => {
                *report
                    .deferred
                    .entry("member_already_owned".into())
                    .or_default() += 1
            }
        }
    }
    if report.retained as u64 >= settings.limits.max_prs {
        report.deferred.entry("max_prs".into()).or_default();
    }
    report.state = if report.deferred.is_empty() {
        "completed"
    } else {
        "capped"
    }
    .into();
    store.pr_progress(
        &serde_json::to_string(&report)?,
        if migration_deferred { None } else { Some(&key) },
        &check,
    )?;
    Ok(report)
}
