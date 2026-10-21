//! Ranked symbol histories and atomic commit provenance; v1.md §3.2/§6.6.
use super::{Error, Store, UnitInsert, admission, insert_unit_in_transaction};
use rusqlite::{OptionalExtension, Row, Transaction, params};
use std::collections::{BTreeMap, BTreeSet};

/// A completed definition selected for history collation.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Target {
    pub(crate) id: i64,
    pub(crate) parent: i64,
    pub(crate) path: String,
    pub(crate) qualified: String,
    pub(crate) history: Option<i64>,
}
/// An immutable commit contribution; PR pointers are presentation metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Entry {
    pub(crate) id: i64,
    pub(crate) sha: Vec<u8>,
    pub(crate) unit: i64,
    pub(crate) author: String,
    pub(crate) time: i64,
    pub(crate) subject: String,
    pub(crate) additions: Option<i64>,
    pub(crate) deletions: Option<i64>,
}
/// One prepared symbol-history replacement. Text and FTS derivation are complete
/// before the bounded writer transaction starts.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Publication<'a> {
    pub(crate) target: &'a Target,
    pub(crate) entries: &'a [Entry],
    pub(crate) unit: UnitInsert<'a>,
}
const TARGET_COLUMNS: &str = "s.id,s.unit_id,f.path,coalesce(s.qualified,s.name),s.history_unit";
const TARGET_JOINS: &str =
    "FROM symbols s JOIN units u ON u.id=s.unit_id JOIN files f ON f.id=s.file_id";
// Walk the existing rank index before checking eligibility. Starting at units
// instead sorts all eligible symbols for every retained history's rank check.
const RANKED_TARGET_JOINS: &str = "FROM symbols s INDEXED BY symbols_rank CROSS JOIN units u ON u.id=s.unit_id CROSS JOIN files f ON f.id=s.file_id";
const TARGET_WHERE: &str = "WHERE u.file_id=f.id AND u.kind IN ('symbol','chunk') AND f.max_level>=1 AND f.l1_pending=0 AND f.l1_fingerprint=f.l1_desired AND f.l1_deferred_reason IS NULL";
const COMMITS: &str = "SELECT c.id,c.sha,c.unit_id,c.author,c.author_time,c.subject,c.additions,c.deletions,c.files_touched,coalesce(c.symbols_touched,'[]') FROM commits c JOIN units u ON u.id=c.unit_id WHERE u.kind='commit' AND u.file_id IS NULL AND u.ladder_level>=1";
fn target_row(r: &Row<'_>) -> rusqlite::Result<Target> {
    Ok(Target {
        id: r.get(0)?,
        parent: r.get(1)?,
        path: r.get(2)?,
        qualified: r.get(3)?,
        history: r.get(4)?,
    })
}
fn entry_row(r: &Row<'_>) -> rusqlite::Result<Entry> {
    Ok(Entry {
        id: r.get(0)?,
        sha: r.get(1)?,
        unit: r.get(2)?,
        author: r.get(3)?,
        time: r.get(4)?,
        subject: r.get(5)?,
        additions: r.get(6)?,
        deletions: r.get(7)?,
    })
}
/// Selects a stable rank prefix, excluding incomplete or non-code definitions.
pub(crate) fn targets(tx: &Transaction<'_>, maximum: u64) -> Result<Vec<Target>, Error> {
    Ok(tx
        .prepare(&format!(
            "SELECT {TARGET_COLUMNS} {RANKED_TARGET_JOINS} {TARGET_WHERE} ORDER BY s.pagerank DESC,s.id LIMIT ?1"
        ))?
        .query_map([maximum.min(i64::MAX as u64) as i64], target_row)?
        .collect::<Result<_, _>>()?)
}
/// Finds retained v1 occurrences even outside the current rank prefix. Recognition
/// uses their stored source set; current v2 histories outside the cap stay retained.
pub(crate) fn legacy_targets(
    tx: &Transaction<'_>,
    check: &dyn Fn() -> Result<(), Error>,
) -> Result<Vec<Target>, Error> {
    let targets = tx
        .prepare(&format!(
            "SELECT {TARGET_COLUMNS} {TARGET_JOINS} {TARGET_WHERE} AND s.history_unit IS NOT NULL ORDER BY s.id"
        ))?
        .query_map([], target_row)?
        .collect::<Result<Vec<_>, _>>()?;
    let mut legacy = Vec::new();
    for target in targets {
        check()?;
        let Some(unit) = target.history else {
            continue;
        };
        let hash: Vec<u8> = tx.query_row(
            "SELECT content_hash FROM units WHERE id=?1 AND kind='history'",
            [unit],
            |r| r.get(0),
        )?;
        let shas = tx.prepare("SELECT c.sha FROM unit_sources s JOIN commits c ON c.id=s.source_id WHERE s.unit_id=?1 AND s.source_kind='commit' AND s.role='content' ORDER BY c.sha")?
            .query_map([unit], |r|r.get::<_,Vec<u8>>(0))?.collect::<Result<Vec<_>,_>>()?;
        let mut old = blake3::Hasher::new();
        old.update(b"history|symbol-history-v1");
        for sha in shas {
            check()?;
            old.update(b"|");
            old.update(&sha);
        }
        if old.finalize().as_bytes().as_slice() == hash {
            legacy.push(target);
        }
    }
    Ok(legacy)
}
/// Checks current rank-prefix membership without materializing every target.
/// The total-symbol count is a conservative upper bound on eligible definitions;
/// when it fits the cap, only the selected owner's eligibility needs checking.
pub(crate) fn is_selected(tx: &Transaction<'_>, id: i64, maximum: u64) -> Result<bool, Error> {
    if maximum == 0 {
        return Ok(false);
    }
    let maximum = maximum.min(i64::MAX as u64) as i64;
    let total: i64 = tx.query_row("SELECT count(*) FROM symbols", [], |row| row.get(0))?;
    if maximum >= total {
        return Ok(tx.query_row(
            &format!("SELECT EXISTS(SELECT 1 {TARGET_JOINS} {TARGET_WHERE} AND s.id=?1)"),
            [id],
            |row| row.get(0),
        )?);
    }
    Ok(tx.query_row(
        &format!("SELECT EXISTS(SELECT 1 FROM (SELECT s.id {RANKED_TARGET_JOINS} {TARGET_WHERE} ORDER BY s.pagerank DESC,s.id LIMIT ?2) selected WHERE selected.id=?1)"),
        params![id, maximum],
        |row| row.get(0),
    )?)
}
/// Resolves the current definition owning a history, without guessing from names.
pub(crate) fn owner(tx: &Transaction<'_>, unit: i64) -> Result<Option<Target>, Error> {
    Ok(tx
        .query_row(
            &format!("SELECT {TARGET_COLUMNS} {TARGET_JOINS} {TARGET_WHERE} AND s.history_unit=?1"),
            [unit],
            target_row,
        )
        .optional()?)
}
/// Distributes one newest-first commit scan into capped symbol sets. File matches
/// prevent equal qualified names in different files from sharing unrelated history.
pub(crate) fn inventory(
    tx: &Transaction<'_>,
    selected: &[Target],
    check: &dyn Fn() -> Result<(), Error>,
) -> Result<Vec<Vec<Entry>>, Error> {
    let mut names: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, target) in selected.iter().enumerate() {
        names.entry(&target.qualified).or_default().push(i);
    }
    let mut result = vec![Vec::new(); selected.len()];
    if selected.is_empty() {
        return Ok(result);
    }
    let mut q = tx.prepare(&format!("{COMMITS} ORDER BY c.author_time DESC,c.sha DESC"))?;
    let mut rows = q.query([])?;
    while let Some(row) = rows.next()? {
        check()?;
        let entry = entry_row(row)?;
        let files: BTreeSet<String> =
            serde_json::from_str(&row.get::<_, String>(8)?).map_err(|_| Error::L1InputsChanged)?;
        let symbols: BTreeSet<String> =
            serde_json::from_str(&row.get::<_, String>(9)?).map_err(|_| Error::L1InputsChanged)?;
        for name in &symbols {
            if let Some(indices) = names.get(name.as_str()) {
                for &i in indices {
                    check()?;
                    if result[i].len() < 100 && files.contains(&selected[i].path) {
                        result[i].push(entry.clone());
                    }
                }
            }
        }
    }
    for entries in &mut result {
        entries.reverse();
    }
    Ok(result)
}
/// Recomputes the bounded source set for a single public verification.
pub(crate) fn entries(
    tx: &Transaction<'_>,
    target: &Target,
    check: &dyn Fn() -> Result<(), Error>,
) -> Result<Vec<Entry>, Error> {
    let mut q=tx.prepare(&format!("{COMMITS} AND EXISTS(SELECT 1 FROM json_each(c.files_touched) WHERE value=?1) AND EXISTS(SELECT 1 FROM json_each(c.symbols_touched) WHERE value=?2) ORDER BY c.author_time DESC,c.sha DESC LIMIT 100"))?;
    let mut rows = q.query(params![target.path, target.qualified])?;
    let mut result = Vec::new();
    while let Some(row) = rows.next()? {
        check()?;
        result.push(entry_row(row)?);
    }
    result.reverse();
    Ok(result)
}
/// Requires exactly the selected commit provenance, including roles and no extras.
pub(crate) fn sources_match(
    tx: &Transaction<'_>,
    unit: i64,
    entries: &[Entry],
) -> Result<bool, Error> {
    let actual=tx.prepare("SELECT source_kind,source_id,role FROM unit_sources WHERE unit_id=?1 ORDER BY source_id")?.query_map([unit],|r|Ok((r.get::<_,String>(0)?,r.get::<_,i64>(1)?,r.get::<_,Option<String>>(2)?)))?.collect::<Result<Vec<_>,_>>()?;
    let wanted = entries.iter().map(|e| e.id).collect::<BTreeSet<_>>();
    Ok(actual.len() == wanted.len()
        && actual.iter().all(|(kind, id, role)| {
            kind == "commit" && role.as_deref() == Some("content") && wanted.contains(id)
        }))
}
/// Checks payload, presentation and exact provenance before tokenizing a no-op.
pub(crate) fn unchanged(
    tx: &Transaction<'_>,
    target: &Target,
    entries: &[Entry],
    hash: &[u8],
    scope: Option<&str>,
    header: &str,
) -> Result<bool, Error> {
    let Some(id) = target.history else {
        return Ok(false);
    };
    let same=tx.query_row("SELECT kind='history' AND file_id IS NULL AND content_hash=?2 AND scope_path IS ?3 AND header=?4 AND parent_unit=?5 FROM units WHERE id=?1",params![id,hash,scope,header,target.parent],|r|r.get::<_,bool>(0)).optional()?.unwrap_or(false);
    Ok(same && sources_match(tx, id, entries)?)
}
impl Store {
    /// Atomically replaces a bounded batch of changed histories and their exact
    /// sources. The final validation fence rolls back the whole batch if HEAD or
    /// configuration changes during publication.
    pub(crate) fn publish_symbol_histories(
        &self,
        publications: &[Publication<'_>],
        check: &dyn Fn() -> Result<(), Error>,
        validate: &dyn Fn() -> Result<(), Error>,
    ) -> Result<(), Error> {
        if publications.is_empty() {
            return Ok(());
        }
        validate()?;
        admission::transaction_checked(self, admission::Kind::L1Growth, Some(check), |tx| {
            for publication in publications {
                let target = publication.target;
                let entries = publication.entries;
                let unit = publication.unit;
                check()?;
                let current = tx
                    .query_row(
                        &format!(
                            "SELECT {TARGET_COLUMNS} {TARGET_JOINS} {TARGET_WHERE} AND s.id=?1"
                        ),
                        [target.id],
                        target_row,
                    )
                    .optional()?;
                if current.as_ref() != Some(target) {
                    return Err(Error::L1InputsChanged);
                }
                for entry in entries {
                    check()?;
                    let current = tx
                        .query_row(&format!("{COMMITS} AND c.id=?1"), [entry.id], entry_row)
                        .optional()?;
                    if current.as_ref() != Some(entry) {
                        return Err(Error::L1InputsChanged);
                    }
                }
                if let Some(old) = target.history {
                    let same: bool = tx.query_row("SELECT content_hash=?2 AND source_key IS ?3 AND duplicate_ordinal=?4 AND kind='history' AND file_id IS NULL AND parent_unit=?5 FROM units WHERE id=?1", params![old,unit.hash,unit.source_key,unit.duplicate_ordinal,target.parent], |r|r.get(0))?;
                    if !entries.is_empty() && same && sources_match(&tx, old, entries)? {
                        tx.execute("DELETE FROM unit_sources WHERE unit_id=?1", [old])?;
                    } else {
                        tx.execute("DELETE FROM units WHERE id=?1", [old])?;
                    }
                }
                if !entries.is_empty() {
                    let id = insert_unit_in_transaction(&tx, unit, None)?;
                    for entry in entries {
                        check()?;
                        tx.execute("INSERT INTO unit_sources(unit_id,source_kind,source_id,role) VALUES (?1,'commit',?2,'content')",params![id,entry.id])?;
                    }
                    tx.execute(
                        "UPDATE symbols SET history_unit=?1 WHERE id=?2",
                        params![id, target.id],
                    )?;
                }
            }
            check()?;
            validate()?;
            tx.commit()?;
            Ok(())
        })
    }

    /// Atomically replaces only a changed history and its sources. No-op histories
    /// bypass growth admission; capacity failure rolls back the old unit and pointer.
    pub(crate) fn publish_symbol_history(
        &self,
        target: &Target,
        entries: &[Entry],
        unit: UnitInsert<'_>,
        check: &dyn Fn() -> Result<(), Error>,
        validate: &dyn Fn() -> Result<(), Error>,
    ) -> Result<bool, Error> {
        validate()?;
        if self.checked_read_snapshot(check, |tx| {
            unchanged(tx, target, entries, unit.hash, unit.scope_path, unit.header)
        })?? {
            validate()?;
            return Ok(false);
        }
        self.publish_symbol_histories(
            &[Publication {
                target,
                entries,
                unit,
            }],
            check,
            validate,
        )?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{FtsColumns, UnitKind};
    use std::{cell::Cell, path::Path};

    fn unit(hash: &'static [u8; 32], text: &'static str) -> UnitInsert<'static> {
        UnitInsert {
            hash,
            source_key: Some("history:1"),
            duplicate_ordinal: 0,
            file_id: None,
            kind: UnitKind::History,
            byte_start: None,
            byte_end: None,
            line_start: None,
            line_end: None,
            scope_path: Some("fixture › code.rs"),
            header: "history of item",
            text,
            tokens: 1,
            ladder_level: 1,
            parent_unit: None,
            created_at: 1,
            updated_at: 1,
            fts: FtsColumns {
                header: "history of item",
                idents: "item",
                body: text,
                prose: text,
            },
        }
    }

    #[test]
    fn rank_membership_respects_ties_caps_and_completed_code_in_both_paths() {
        let store = crate::store::open(Path::new(":memory:"), Default::default()).unwrap();
        let mut writer = store.writer().unwrap();
        let tx = writer.transaction().unwrap();
        for id in 1_i64..=9 {
            let hash = [id as u8; 32];
            tx.execute("INSERT INTO files(id,path,blake3,size,mtime_ns,kind,max_level,treatment,l1_pending,l1_desired,l1_fingerprint) VALUES (?1,?2,zeroblob(32),1,1,'code',3,'code',0,X'01',X'01')", params![id, format!("code{id}.rs")]).unwrap();
            tx.execute(
                "INSERT INTO unit_contents(hash,text,tokens) VALUES(?1,'code',1)",
                [hash.as_slice()],
            )
            .unwrap();
            tx.execute("INSERT INTO units(id,hash,content_hash,source_key,kind,header,file_id,ladder_level,created_at,updated_at) VALUES(?1,?2,?2,?3,'symbol','item',?1,1,1,1)", params![id,hash.as_slice(),format!("rank:{id}")]).unwrap();
            let rank = match id {
                1 => 1.0,
                2 | 3 => 3.0,
                _ => 100.0,
            };
            tx.execute("INSERT INTO symbols(id,file_id,unit_id,name,kind,line_start,line_end,pagerank) VALUES(?1,?1,?1,'item','function',1,1,?2)", params![id,rank]).unwrap();
        }
        tx.execute_batch(
            "UPDATE files SET max_level=0 WHERE id=4;
            UPDATE files SET l1_pending=1 WHERE id=5;
            UPDATE files SET l1_deferred_reason='cap' WHERE id=6;
            UPDATE files SET l1_fingerprint=X'02' WHERE id=7;
            UPDATE units SET kind='doc_section' WHERE id=8;
            UPDATE symbols SET file_id=1 WHERE id=9;",
        )
        .unwrap();
        for cap in [0, 1, 2, 3, 8, 9, u64::MAX] {
            for id in 1..=10 {
                let expected = match id {
                    2 => cap >= 1,
                    3 => cap >= 2,
                    1 => cap >= 3,
                    _ => false,
                };
                assert_eq!(
                    is_selected(&tx, id, cap).unwrap(),
                    expected,
                    "id={id}, cap={cap}"
                );
            }
        }
        tx.execute("UPDATE symbols SET pagerank=10 WHERE id=1", [])
            .unwrap();
        assert!(is_selected(&tx, 1, 1).unwrap());
        assert!(!is_selected(&tx, 2, 1).unwrap());
        tx.execute("UPDATE files SET max_level=0 WHERE id=1", [])
            .unwrap();
        assert!(!is_selected(&tx, 1, u64::MAX).unwrap());
        assert!(is_selected(&tx, 2, 1).unwrap());
    }

    #[test]
    fn late_publication_refusal_preserves_old_history_and_rolls_back_new_payload() {
        let store = crate::store::open(Path::new(":memory:"), Default::default()).unwrap();
        let (mut target, entries) = {
            let mut writer = store.writer().unwrap();
            let tx = writer.transaction().unwrap();
            tx.execute("INSERT INTO files(id,path,blake3,size,mtime_ns,kind,max_level,treatment,l1_pending,l1_desired,l1_fingerprint) VALUES (1,'code.rs',zeroblob(32),1,1,'code',3,'code',0,X'01',X'01')", []).unwrap();
            let mut parent = unit(&[1; 32], "fn item() {}");
            parent.file_id = Some(1);
            parent.kind = UnitKind::Symbol;
            let parent = insert_unit_in_transaction(&tx, parent, None).unwrap();
            let mut commit = unit(&[2; 32], "source commit");
            commit.source_key = Some("commit:fixture");
            commit.kind = UnitKind::Commit;
            let commit = insert_unit_in_transaction(&tx, commit, None).unwrap();
            tx.execute("INSERT INTO symbols(id,file_id,unit_id,name,kind,line_start,line_end) VALUES (1,1,?1,'item','function',1,1)", [parent]).unwrap();
            tx.execute("INSERT INTO commits(id,sha,author,author_time,subject,parents,files_touched,symbols_touched,unit_id) VALUES (1,zeroblob(20),'fixture',1,'source commit',0,'[\"code.rs\"]','[\"item\"]',?1)", [commit]).unwrap();
            let target = targets(&tx, 1).unwrap().remove(0);
            let entries = entries(&tx, &target, &|| Ok(())).unwrap();
            tx.commit().unwrap();
            (target, entries)
        };
        let mut first = unit(&[3; 32], "first history");
        first.parent_unit = Some(target.parent);
        assert!(
            store
                .publish_symbol_history(&target, &entries, first, &|| Ok(()), &|| Ok(()))
                .unwrap()
        );
        target = store
            .checked_read_snapshot(&|| Ok(()), |tx| targets(tx, 1))
            .unwrap()
            .unwrap()
            .remove(0);
        let old = target.history.unwrap();
        let mut context_only = unit(&[3; 32], "first history");
        context_only.parent_unit = Some(target.parent);
        context_only.scope_path = Some("renamed repository › code.rs");
        {
            let mut writer = store.writer().unwrap();
            let tx = writer.transaction().unwrap();
            let mut summary = unit(&[8; 32], "retained history role");
            summary.source_key = Some("summary:history");
            summary.kind = UnitKind::Summary;
            summary.parent_unit = Some(old);
            insert_unit_in_transaction(&tx, summary, None).unwrap();
            tx.commit().unwrap();
        }
        assert!(
            store
                .publish_symbol_history(&target, &entries, context_only, &|| Ok(()), &|| Ok(()))
                .unwrap()
        );
        {
            let writer = store.writer().unwrap();
            let retained: (i64, String) = writer.query_row("SELECT parent.id,parent.scope_path FROM units child JOIN units parent ON parent.id=child.parent_unit WHERE child.source_key='summary:history'", [], |r|Ok((r.get(0)?,r.get(1)?))).unwrap();
            assert_eq!(retained, (old, "renamed repository › code.rs".into()));
        }
        let checks = Cell::new(0);
        let mut replacement = unit(&[4; 32], "changed history");
        replacement.parent_unit = Some(target.parent);
        let result =
            store.publish_symbol_history(&target, &entries, replacement, &|| Ok(()), &|| {
                checks.set(checks.get() + 1);
                if checks.get() == 2 {
                    Err(Error::L1InputsChanged)
                } else {
                    Ok(())
                }
            });
        assert!(matches!(result, Err(Error::L1InputsChanged)));
        let writer = store.writer().unwrap();
        assert_eq!(
            writer
                .query_row("SELECT history_unit FROM symbols WHERE id=1", [], |r| r
                    .get::<_, i64>(0))
                .unwrap(),
            old
        );
        assert_eq!(
            writer
                .query_row("SELECT count(*) FROM units WHERE kind='history'", [], |r| r
                    .get::<_, i64>(0))
                .unwrap(),
            1
        );
        assert_eq!(
            writer
                .query_row(
                    "SELECT count(*) FROM unit_sources WHERE unit_id=?1",
                    [old],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
            1
        );
        assert_eq!(
            writer
                .query_row(
                    "SELECT count(*) FROM unit_contents WHERE hash=?1",
                    [&[4u8; 32][..]],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
            0
        );
    }
}
