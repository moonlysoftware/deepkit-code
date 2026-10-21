//! Atomic PR ownership, provenance and bounded member reads; v1.md §3.2/§4.4.
use super::{Store, UnitInsert, admission, insert_unit_in_transaction};
use rusqlite::{OptionalExtension, Transaction, params};

/// Immutable published commit metadata used to collate PRs without another diff.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Member {
    pub(crate) id: i64,
    pub(crate) sha: Vec<u8>,
    pub(crate) unit: i64,
    pub(crate) owner: Option<i64>,
    pub(crate) time: i64,
    pub(crate) subject: String,
    pub(crate) body: String,
    pub(crate) files: Vec<String>,
    pub(crate) symbols: Vec<String>,
    pub(crate) additions: Option<i64>,
    pub(crate) deletions: Option<i64>,
}

/// Reads one retained member; malformed source metadata remains an explicit failure.
pub(crate) fn member(tx: &Transaction<'_>, sha: &[u8]) -> Result<Option<Member>, super::Error> {
    let row = tx.query_row("SELECT id,sha,unit_id,pr_id,author_time,subject,body,files_touched,coalesce(symbols_touched,'[]'),additions,deletions FROM commits WHERE sha=?1 AND unit_id IS NOT NULL AND EXISTS(SELECT 1 FROM units WHERE id=commits.unit_id AND kind='commit' AND file_id IS NULL)",[sha],|r| Ok((Member {id:r.get(0)?,sha:r.get(1)?,unit:r.get(2)?,owner:r.get(3)?,time:r.get(4)?,subject:r.get(5)?,body:r.get(6)?,files:Vec::new(),symbols:Vec::new(),additions:r.get(9)?,deletions:r.get(10)?},r.get::<_,String>(7)?,r.get::<_,String>(8)?))).optional()?;
    row.map(|(mut member, files, symbols)| {
        member.files = serde_json::from_str(&files).map_err(|_| super::Error::L1InputsChanged)?;
        member.symbols =
            serde_json::from_str(&symbols).map_err(|_| super::Error::L1InputsChanged)?;
        Ok(member)
    })
    .transpose()
}

pub(crate) struct Insert<'a> {
    pub(crate) marker: &'a crate::config::pr_patterns::Marker,
    pub(crate) merge: &'a [u8],
    pub(crate) base: Option<&'a [u8]>,
    pub(crate) head: &'a [u8],
    pub(crate) time: i64,
    pub(crate) members: &'a [Member],
    pub(crate) unit: UnitInsert<'a>,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    Inserted,
    Existing,
    Updated,
    Capacity,
    Owned,
}

/// Retained owner metadata, captured before replacing its occurrence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Retained {
    pub(crate) id: i64,
    pub(crate) unit: Option<i64>,
    pub(crate) number: Option<i64>,
    pub(crate) title: String,
    pub(crate) base: Option<Vec<u8>>,
    pub(crate) head: Vec<u8>,
    pub(crate) strategy: String,
    pub(crate) count: i64,
    pub(crate) time: i64,
}

/// Reads one owner independently of the current discovery window or PR cap.
pub(crate) fn retained(
    tx: &Transaction<'_>,
    merge: &[u8],
) -> Result<Option<Retained>, super::Error> {
    Ok(tx.query_row("SELECT id,unit_id,number,title,base_sha,head_sha,strategy,commit_count,merged_at FROM prs WHERE merge_sha=?1", [merge], |r|Ok(Retained {id:r.get(0)?,unit:r.get(1)?,number:r.get(2)?,title:r.get(3)?,base:r.get(4)?,head:r.get(5)?,strategy:r.get(6)?,count:r.get(7)?,time:r.get(8)?})).optional()?)
}

fn unchanged(
    tx: &Transaction<'_>,
    pr: &Insert<'_>,
    old: &Retained,
    check: &dyn Fn() -> Result<(), super::Error>,
) -> Result<bool, super::Error> {
    let Some(unit) = old.unit else {
        return Ok(false);
    };
    if old.number != pr.marker.number
        || old.title != pr.marker.title
        || old.base.as_deref() != pr.base
        || old.head != pr.head
        || old.strategy != pr.marker.strategy
        || old.count != pr.members.len() as i64
        || old.time != pr.time
    {
        return Ok(false);
    }
    let Some(members) = Store::pr_members(tx, old.id, unit, pr.members.len(), check)? else {
        return Ok(false);
    };
    if members.len() != pr.members.len() || !members.iter().all(|m| pr.members.contains(m)) {
        return Ok(false);
    }
    Ok(tx.query_row("SELECT kind='pr' AND file_id IS NULL AND content_hash=?2 AND source_key IS ?3 AND scope_path IS ?4 AND header=?5 AND parent_unit IS NULL FROM units WHERE id=?1", params![unit,pr.unit.hash,pr.unit.source_key,pr.unit.scope_path,pr.unit.header], |r|r.get::<_,bool>(0)).optional()?.unwrap_or(false))
}

impl Store {
    /// Saves PR completion and visible deferrals under the original publication fence.
    pub(crate) fn pr_progress(
        &self,
        status: &str,
        key: Option<&str>,
        check: &dyn Fn() -> Result<(), super::Error>,
    ) -> Result<(), super::Error> {
        admission::transaction_checked(self, admission::Kind::Control, Some(check), |tx| {
            check()?;
            tx.execute("INSERT INTO meta(key,value) VALUES ('pr_status',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[status])?;
            if let Some(key) = key {
                tx.execute("INSERT INTO meta(key,value) VALUES ('pr_completed',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[key])?;
            }
            check()?;
            tx.commit()?;
            Ok(())
        })
    }
    /// Commits a PR, one-owner assignments and all sources under one admitted writer.
    /// Capacity or a concurrent owner leaves every row untouched.
    pub(crate) fn insert_pr(
        &self,
        pr: Insert<'_>,
        maximum: u64,
        check: &dyn Fn() -> Result<(), super::Error>,
    ) -> Result<Outcome, super::Error> {
        check()?;
        let captured = self.checked_read_snapshot(check, |tx| retained(tx, pr.merge))??;
        if let Some(old) = &captured
            && self.checked_read_snapshot(check, |tx| {
                if retained(tx, pr.merge)?.as_ref() != Some(old) {
                    return Err(super::Error::L1InputsChanged);
                }
                unchanged(tx, &pr, old, check)
            })??
        {
            check()?;
            return Ok(Outcome::Existing);
        }
        admission::transaction_checked(self, admission::Kind::L1Growth, Some(check), |tx| {
            check()?;
            if retained(&tx, pr.merge)? != captured {
                return Err(super::Error::L1InputsChanged);
            }
            if captured.is_none() {
                let count: i64 = tx.query_row("SELECT count(*) FROM prs", [], |r| r.get(0))?;
                if count as u64 >= maximum {
                    return Ok(Outcome::Capacity);
                }
            }
            if pr.members.is_empty() || pr.members.len() > 500 {
                return Err(super::Error::L1InputsChanged);
            }
            let owner = captured.as_ref().map(|old| old.id);
            for expected in pr.members {
                check()?;
                let Some(actual) = member(&tx, &expected.sha)? else {
                    return Err(super::Error::L1InputsChanged);
                };
                if &actual != expected {
                    return Err(super::Error::L1InputsChanged);
                }
                if actual.owner.is_some() && actual.owner != owner {
                    return Ok(Outcome::Owned);
                }
            }
            if let Some(old) = &captured
                && let Some(unit) = old.unit
            {
                let same: bool = tx.query_row("SELECT content_hash=?2 AND source_key IS ?3 AND kind='pr' AND file_id IS NULL FROM units WHERE id=?1",params![unit,pr.unit.hash,pr.unit.source_key],|r|r.get(0))?;
                let same_members = Self::pr_members(&tx, old.id, unit, pr.members.len(), check)?
                    .is_some_and(|members| {
                        members.len() == pr.members.len()
                            && members.iter().all(|m| pr.members.contains(m))
                    });
                if same && same_members {
                    tx.execute("DELETE FROM unit_sources WHERE unit_id=?1", [unit])?;
                } else {
                    tx.execute("DELETE FROM units WHERE id=?1", [unit])?;
                }
            }
            let unit = insert_unit_in_transaction(&tx, pr.unit, None)?;
            let id = if let Some(old) = &captured {
                tx.execute("UPDATE prs SET number=?2,title=?3,base_sha=?4,head_sha=?5,strategy=?6,commit_count=?7,merged_at=?8,unit_id=?9 WHERE id=?1",params![old.id,pr.marker.number,pr.marker.title,pr.base,pr.head,pr.marker.strategy,pr.members.len() as i64,pr.time,unit])?;
                tx.execute("UPDATE commits SET pr_id=NULL WHERE pr_id=?1", [old.id])?;
                old.id
            } else {
                tx.execute("INSERT INTO prs(number,title,merge_sha,base_sha,head_sha,strategy,commit_count,merged_at,unit_id) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",params![pr.marker.number,pr.marker.title,pr.merge,pr.base,pr.head,pr.marker.strategy,pr.members.len() as i64,pr.time,unit])?;
                tx.last_insert_rowid()
            };
            for member in pr.members {
                check()?;
                tx.execute(
                    "UPDATE commits SET pr_id=?1 WHERE id=?2 AND pr_id IS NULL",
                    params![id, member.id],
                )?;
                tx.execute("INSERT INTO unit_sources(unit_id,source_kind,source_id,role) VALUES (?1,'commit',?2,'content')",params![unit,member.id])?;
            }
            check()?;
            tx.commit()?;
            Ok(if captured.is_some() {
                Outcome::Updated
            } else {
                Outcome::Inserted
            })
        })
    }

    /// Reads all members of one bounded PR, checking both directions of ownership.
    pub(crate) fn pr_members(
        tx: &Transaction<'_>,
        id: i64,
        unit: i64,
        count: usize,
        check: &dyn Fn() -> Result<(), super::Error>,
    ) -> Result<Option<Vec<Member>>, super::Error> {
        if count == 0 || count > 500 {
            return Ok(None);
        }
        let shas = tx
            .prepare("SELECT sha FROM commits WHERE pr_id=?1 ORDER BY author_time,sha LIMIT 501")?
            .query_map([id], |r| r.get::<_, Vec<u8>>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        let sources: i64 = tx.query_row(
            "SELECT count(*) FROM unit_sources WHERE unit_id=?1",
            [unit],
            |r| r.get(0),
        )?;
        if shas.len() != count || sources != count as i64 {
            return Ok(None);
        }
        let mut members = Vec::with_capacity(count);
        for sha in shas {
            check()?;
            let Some(member) = member(tx, &sha)? else {
                return Ok(None);
            };
            let owned:bool=tx.query_row("SELECT EXISTS(SELECT 1 FROM unit_sources WHERE unit_id=?1 AND source_kind='commit' AND source_id=?2 AND role='content')",params![unit,member.id],|r|r.get(0))?;
            if !owned {
                return Ok(None);
            }
            members.push(member);
        }
        Ok(Some(members))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::pr_patterns::Marker,
        store::{FtsColumns, UnitKind},
        treat,
    };
    use std::path::Path;

    fn unit<'a>(hash: &'a [u8; 32], source: &'a str, text: &'a str) -> UnitInsert<'a> {
        UnitInsert {
            hash,
            source_key: Some(source),
            duplicate_ordinal: 0,
            file_id: None,
            kind: UnitKind::PullRequest,
            byte_start: None,
            byte_end: None,
            line_start: None,
            line_end: None,
            scope_path: Some("PR"),
            header: "PR",
            text,
            tokens: 1,
            ladder_level: 1,
            parent_unit: None,
            created_at: 1,
            updated_at: 1,
            fts: FtsColumns {
                header: "PR",
                idents: "",
                body: text,
                prose: text,
            },
        }
    }
    fn publish(
        store: &Store,
        n: u8,
        members: &[Member],
        maximum: u64,
    ) -> Result<Outcome, crate::store::Error> {
        let marker = Marker {
            number: Some(i64::from(n)),
            title: format!("PR {n}"),
            strategy: "merge".into(),
        };
        let collated = treat::pr::collate(&marker, "2026-01-01", members, &[]);
        let source = format!("pr:{n}");
        store.insert_pr(
            Insert {
                marker: &marker,
                merge: &[n; 20],
                base: None,
                head: &[n; 20],
                time: 1,
                members,
                unit: unit(&collated.hash, &source, &collated.text),
            },
            maximum,
            &|| Ok(()),
        )
    }
    fn members(store: &Store, ids: &[u8]) -> Vec<Member> {
        store
            .checked_read_snapshot(&|| Ok(()), |tx| {
                ids.iter()
                    .map(|n| member(tx, &[*n; 20]).map(|m| m.unwrap()))
                    .collect::<Result<Vec<_>, _>>()
            })
            .unwrap()
            .unwrap()
    }
    fn identities(store: &Store) -> Vec<(i64, i64, Vec<u8>)> {
        store.checked_read_snapshot(&||Ok(()),|tx|tx.prepare("SELECT p.id,u.id,u.hash FROM prs p JOIN units u ON u.id=p.unit_id ORDER BY p.id").unwrap().query_map([],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).unwrap().collect::<Result<Vec<_>,_>>()).unwrap().unwrap()
    }
    #[test]
    fn member_set_replacement_changes_only_its_owner_and_fences_metadata_and_overlap() {
        let store = crate::store::open(Path::new(":memory:"), Default::default()).unwrap();
        {
            let mut writer = store.writer().unwrap();
            let tx = writer.transaction().unwrap();
            for n in 1u8..=3 {
                let source = format!("commit:{n}");
                let text = format!("commit {n}");
                let hash = [n; 32];
                let mut input = unit(&hash, &source, &text);
                input.kind = UnitKind::Commit;
                let id = insert_unit_in_transaction(&tx, input, None).unwrap();
                tx.execute("INSERT INTO commits(id,sha,author,author_time,subject,body,parents,files_touched,symbols_touched,unit_id,additions,deletions) VALUES (?1,?2,'fixture',1,?3,'',0,'[]','[]',?4,1,0)",params![n,[n;20].as_slice(),text,id]).unwrap();
            }
            tx.commit().unwrap();
        }
        assert_eq!(
            publish(&store, 10, &members(&store, &[1, 2]), 10).unwrap(),
            Outcome::Inserted
        );
        assert_eq!(
            publish(&store, 20, &members(&store, &[3]), 10).unwrap(),
            Outcome::Inserted
        );
        let before = identities(&store);
        let captured = members(&store, &[1]);
        store
            .writer()
            .unwrap()
            .execute(
                "UPDATE commits SET subject='changed metadata' WHERE id=1",
                [],
            )
            .unwrap();
        assert!(matches!(
            publish(&store, 10, &captured, 10),
            Err(crate::store::Error::L1InputsChanged)
        ));
        assert_eq!(identities(&store), before);
        store
            .writer()
            .unwrap()
            .execute("UPDATE commits SET subject='commit 1' WHERE id=1", [])
            .unwrap();
        assert_eq!(
            publish(&store, 10, &members(&store, &[1]), 0).unwrap(),
            Outcome::Updated
        );
        let updated = identities(&store);
        assert_eq!(updated[0].0, before[0].0);
        assert_ne!(updated[0].2, before[0].2);
        assert_eq!(updated[1], before[1]);
        assert_eq!(members(&store, &[2])[0].owner, None);
        assert_eq!(
            publish(&store, 10, &members(&store, &[1, 3]), 10).unwrap(),
            Outcome::Owned
        );
        assert_eq!(identities(&store), updated);
        assert_eq!(
            publish(&store, 10, &members(&store, &[1]), 0).unwrap(),
            Outcome::Existing
        );
        assert_eq!(identities(&store), updated);
    }
}
