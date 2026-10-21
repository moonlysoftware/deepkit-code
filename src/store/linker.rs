//! Atomic documentation mentions and test labels; v1.md §6.5.
use super::{Error, FtsNameKind, Store, admission, names};
use crate::treat::linker;
use rusqlite::{Transaction, params};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
struct File {
    id: i64,
    path: String,
    kind: String,
    hash: Vec<u8>,
    fingerprint: Option<Vec<u8>>,
    active: bool,
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct Code {
    id: i64,
    file: i64,
    name: String,
    qualified: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct Doc {
    id: i64,
    file: i64,
    header: String,
    text: String,
    start: i64,
    end: i64,
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct Reference {
    id: i64,
    src: i64,
    line: Option<i64>,
    file: i64,
    name: String,
    dst: Option<i64>,
    kind: String,
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Link {
    name: String,
    dst: Option<i64>,
    line: i64,
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct Synthetic {
    id: i64,
    unit: Option<i64>,
    file: i64,
    name: String,
    start: i64,
    end: i64,
    links: Vec<Link>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct Snapshot {
    generation: i64,
    files: Vec<File>,
    code: Vec<Code>,
    docs: Vec<Doc>,
    refs: Vec<Reference>,
    synthetic: Vec<Synthetic>,
}
const ELIGIBLE: &str = "f.max_level>=1 AND f.l1_pending=0 AND f.l1_fingerprint=f.l1_desired AND f.l1_deferred_reason IS NULL";
fn snapshot(
    tx: &Transaction<'_>,
    check: &dyn Fn() -> Result<(), Error>,
) -> Result<Snapshot, Error> {
    check()?;
    let generation = tx.query_row(
        "SELECT CAST(value AS INTEGER) FROM meta WHERE key='l1_scan_generation'",
        [],
        |r| r.get(0),
    )?;
    let files=tx.prepare(&format!("SELECT f.id,f.path,f.kind,f.blake3,f.l1_fingerprint,coalesce(({ELIGIBLE}),0) FROM files f ORDER BY f.id"))?.query_map([],|r|Ok(File{id:r.get(0)?,path:r.get(1)?,kind:r.get(2)?,hash:r.get(3)?,fingerprint:r.get(4)?,active:r.get(5)?}))?.collect::<Result<Vec<_>,_>>()?;
    check()?;
    let code=tx.prepare("SELECT s.id,s.file_id,s.name,s.qualified FROM symbols s JOIN files f ON f.id=s.file_id JOIN units u ON u.id=s.unit_id WHERE f.kind='code' AND u.kind IN ('symbol','chunk') AND s.kind!='doc' ORDER BY s.id")?.query_map([],|r|Ok(Code{id:r.get(0)?,file:r.get(1)?,name:r.get(2)?,qualified:r.get(3)?}))?.collect::<Result<Vec<_>,_>>()?;
    check()?;
    let docs=tx.prepare(&format!("SELECT u.id,u.file_id,u.header,c.text,coalesce(u.line_start,1),coalesce(u.line_end,1) FROM units u JOIN files f ON f.id=u.file_id JOIN unit_contents c ON c.hash=u.content_hash WHERE {ELIGIBLE} AND f.kind='doc' AND u.kind='doc_section' ORDER BY u.id"))?.query_map([],|r|Ok(Doc{id:r.get(0)?,file:r.get(1)?,header:r.get(2)?,text:r.get(3)?,start:r.get(4)?,end:r.get(5)?}))?.collect::<Result<Vec<_>,_>>()?;
    check()?;
    let refs=tx.prepare(&format!("SELECT e.rowid,e.file_id,e.dst_name,e.dst_symbol,e.kind,e.src_symbol,e.line FROM edges e JOIN files f ON f.id=e.file_id WHERE {ELIGIBLE} AND f.kind='code' AND e.kind IN ('ref','test_of') ORDER BY e.rowid"))?.query_map([],|r|Ok(Reference{id:r.get(0)?,file:r.get(1)?,name:r.get(2)?,dst:r.get(3)?,kind:r.get(4)?,src:r.get(5)?,line:r.get(6)?}))?.collect::<Result<Vec<_>,_>>()?;
    let mut synthetic=tx.prepare(&format!("SELECT s.id,s.unit_id,s.file_id,s.name,s.line_start,s.line_end FROM symbols s JOIN files f ON f.id=s.file_id WHERE {ELIGIBLE} AND s.kind='doc' ORDER BY s.id"))?.query_map([],|r|Ok(Synthetic{id:r.get(0)?,unit:r.get(1)?,file:r.get(2)?,name:r.get(3)?,start:r.get(4)?,end:r.get(5)?,links:Vec::new()}))?.collect::<Result<Vec<_>,_>>()?;
    let mut links: BTreeMap<i64, Vec<Link>> = BTreeMap::new();
    let mut query = tx.prepare(&format!("SELECT e.src_symbol,e.dst_name,e.dst_symbol,coalesce(e.line,1) FROM edges e JOIN symbols s ON s.id=e.src_symbol JOIN files f ON f.id=s.file_id WHERE {ELIGIBLE} AND s.kind='doc' AND e.kind='doc_mention' ORDER BY e.src_symbol,e.dst_name,e.dst_symbol,e.line"))?;
    let mut rows = query.query([])?;
    while let Some(row) = rows.next()? {
        check()?;
        links.entry(row.get(0)?).or_default().push(Link {
            name: row.get(1)?,
            dst: row.get(2)?,
            line: row.get(3)?,
        });
    }
    for doc in &mut synthetic {
        doc.links = links.remove(&doc.id).unwrap_or_default();
    }
    Ok(Snapshot {
        generation,
        files,
        code,
        docs,
        refs,
        synthetic,
    })
}

struct Desired {
    docs: Vec<Vec<Link>>,
    refs: Vec<&'static str>,
}
fn desired(
    input: &Snapshot,
    tests: &globset::GlobSet,
    check: &dyn Fn() -> Result<(), Error>,
) -> Result<Desired, Error> {
    let files: BTreeMap<_, _> = input.files.iter().map(|f| (f.id, f)).collect();
    let mut names: BTreeMap<&str, Vec<&Code>> = BTreeMap::new();
    let mut by_file: BTreeMap<i64, Vec<&Code>> = BTreeMap::new();
    let ids: BTreeMap<_, _> = input.code.iter().map(|s| (s.id, s)).collect();
    for s in &input.code {
        names.entry(&s.name).or_default().push(s);
        if let Some(q) = &s.qualified
            && q != &s.name
        {
            names.entry(q).or_default().push(s);
        }
        by_file.entry(s.file).or_default().push(s);
    }
    let code_files: BTreeMap<_, _> = input
        .files
        .iter()
        .filter(|f| f.kind == "code")
        .map(|f| (f.path.as_str(), f.id))
        .collect();
    let mut docs = Vec::new();
    for doc in &input.docs {
        check()?;
        let mut links = BTreeSet::new();
        for mention in linker::mentions(&doc.text) {
            check()?;
            let line = doc.start.saturating_add(mention.line as i64);
            if mention.code
                && let Some(targets) = names.get(mention.value.as_str())
            {
                for target in targets {
                    links.insert(Link {
                        name: mention.value.clone(),
                        dst: Some(target.id),
                        line,
                    });
                }
            }
            if mention.path {
                for path in linker::paths(&files[&doc.file].path, &mention.value) {
                    if let Some(file) = code_files.get(path.as_str()) {
                        if let Some(targets) = by_file.get(file) {
                            for target in targets {
                                links.insert(Link {
                                    name: path.clone(),
                                    dst: Some(target.id),
                                    line,
                                });
                            }
                        } else {
                            links.insert(Link {
                                name: path,
                                dst: None,
                                line,
                            });
                        }
                    }
                }
            }
        }
        docs.push(links.into_iter().collect());
    }
    let mut refs = Vec::new();
    for reference in &input.refs {
        check()?;
        let targets = match reference.dst {
            Some(id) => ids.get(&id).map(|s| vec![*s]).unwrap_or_default(),
            None => names
                .get(reference.name.as_str())
                .cloned()
                .unwrap_or_default(),
        };
        let is_test = tests.is_match(&files[&reference.file].path)
            && !targets.is_empty()
            && targets
                .iter()
                .all(|s| !tests.is_match(&files[&s.file].path));
        refs.push(if is_test { "test_of" } else { "ref" });
    }
    Ok(Desired { docs, refs })
}
fn same(old: &Synthetic, doc: &Doc, links: &[Link]) -> bool {
    old.unit == Some(doc.id)
        && old.file == doc.file
        && old.name == doc.header
        && old.start == doc.start
        && old.end == doc.end
        && old.links == links
}
impl Store {
    /// Reconciles current eligible links atomically. Retained capped files are
    /// untouched; unchanged relationships bypass growth admission and keep IDs.
    pub(crate) fn link_documents(
        &self,
        tests: &globset::GlobSet,
        check: &dyn Fn() -> Result<(), Error>,
        validate: &dyn Fn() -> Result<(), Error>,
    ) -> Result<bool, Error> {
        validate()?;
        let input = self.checked_read_snapshot(check, |tx| snapshot(tx, check))??;
        let wanted = desired(&input, tests, check)?;
        let orphaned: Vec<_> = input
            .synthetic
            .iter()
            .filter(|s| s.unit.is_none())
            .map(|s| s.id)
            .collect();
        let mut previous: BTreeMap<i64, Vec<&Synthetic>> = BTreeMap::new();
        for old in &input.synthetic {
            if let Some(unit) = old.unit {
                previous.entry(unit).or_default().push(old);
            }
        }
        let unchanged = orphaned.is_empty()
            && input.docs.iter().zip(&wanted.docs).all(|(doc, links)| {
                let current = previous.get(&doc.id).map(Vec::as_slice).unwrap_or(&[]);
                current.len() == 1 && same(current[0], doc, links)
            })
            && input
                .refs
                .iter()
                .zip(&wanted.refs)
                .all(|(r, kind)| r.kind == *kind);
        if unchanged {
            validate()?;
            return Ok(false);
        }
        admission::transaction_checked(self, admission::Kind::L1Growth, Some(check), |tx| {
            if snapshot(&tx, check)? != input {
                return Err(Error::L1InputsChanged);
            }
            for id in &orphaned {
                tx.execute("DELETE FROM fts_names WHERE rowid=?1", [id])?;
                tx.execute("DELETE FROM symbols WHERE id=?1", [id])?;
            }
            for (doc, links) in input.docs.iter().zip(&wanted.docs) {
                check()?;
                let old = previous.get(&doc.id).map(Vec::as_slice).unwrap_or(&[]);
                if old.len() == 1 && same(old[0], doc, links) {
                    continue;
                }
                let id = if let Some(first) = old.first() {
                    for extra in old.iter().skip(1) {
                        tx.execute("DELETE FROM fts_names WHERE rowid=?1", [extra.id])?;
                        tx.execute("DELETE FROM symbols WHERE id=?1", [extra.id])?;
                    }
                    tx.execute(
                        "UPDATE symbols SET name=?2,line_start=?3,line_end=?4 WHERE id=?1",
                        params![first.id, doc.header, doc.start, doc.end],
                    )?;
                    first.id
                } else {
                    tx.execute("INSERT INTO symbols(file_id,unit_id,name,kind,line_start,line_end) VALUES (?1,?2,?3,'doc',?4,?5)",params![doc.file,doc.id,doc.header,doc.start,doc.end])?;
                    tx.last_insert_rowid()
                };
                names::insert(&tx, &doc.header, FtsNameKind::Symbol, id)?;
                tx.execute(
                    "DELETE FROM edges WHERE src_symbol=?1 AND kind='doc_mention'",
                    [id],
                )?;
                for link in links {
                    check()?;
                    tx.execute("INSERT INTO edges(src_symbol,dst_name,dst_symbol,kind,file_id,line) VALUES (?1,?2,?3,'doc_mention',?4,?5)",params![id,link.name,link.dst,doc.file,link.line])?;
                }
            }
            for (reference, kind) in input.refs.iter().zip(&wanted.refs) {
                if reference.kind != *kind {
                    tx.execute(
                        "UPDATE edges SET kind=?2 WHERE rowid=?1",
                        params![reference.id, kind],
                    )?;
                }
            }
            // Rows owned by retained, inactive doc units are intentionally preserved.
            tx.execute("INSERT INTO meta(key,value) VALUES ('graph_pending','1') ON CONFLICT(key) DO UPDATE SET value='1'",[])?;
            validate()?;
            tx.commit()?;
            Ok(true)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{FtsColumns, UnitInsert, UnitKind, insert_unit_in_transaction};
    use std::{cell::Cell, path::Path};
    fn fixture() -> Store {
        let store = crate::store::open(Path::new(":memory:"), Default::default()).unwrap();
        {
            let mut writer = store.writer().unwrap();
            let tx = writer.transaction().unwrap();
            for (id, path, kind, text, unit_kind) in [
                (
                    1,
                    "src/code.rs",
                    "code",
                    "fn should_retry() {}",
                    UnitKind::Symbol,
                ),
                (
                    2,
                    "README.md",
                    "doc",
                    "Use `should_retry`.",
                    UnitKind::DocSection,
                ),
                (
                    3,
                    "tests/check.rs",
                    "code",
                    "fn check() { should_retry(); }",
                    UnitKind::Symbol,
                ),
            ] {
                tx.execute("INSERT INTO files(id,path,blake3,size,mtime_ns,kind,max_level,treatment,l1_pending,l1_desired,l1_fingerprint) VALUES (?1,?2,zeroblob(32),1,1,?3,3,?3,0,X'01',X'01')",params![id,path,kind]).unwrap();
                let hash = blake3::hash(text.as_bytes());
                let unit = insert_unit_in_transaction(
                    &tx,
                    UnitInsert {
                        hash: hash.as_bytes(),
                        source_key: Some(path),
                        duplicate_ordinal: 0,
                        file_id: Some(id),
                        kind: unit_kind,
                        byte_start: None,
                        byte_end: None,
                        line_start: Some(1),
                        line_end: Some(1),
                        scope_path: None,
                        header: path,
                        text,
                        tokens: 4,
                        ladder_level: 1,
                        parent_unit: None,
                        created_at: 1,
                        updated_at: 1,
                        fts: FtsColumns {
                            header: path,
                            idents: text,
                            body: text,
                            prose: text,
                        },
                    },
                    None,
                )
                .unwrap();
                if id != 2 {
                    tx.execute("INSERT INTO symbols(id,file_id,unit_id,name,kind,line_start,line_end) VALUES (?1,?1,?2,?3,'function',1,1)",params![id,unit,if id==1 {"should_retry"} else {"check"}]).unwrap();
                }
            }
            tx.execute("INSERT INTO edges(src_symbol,dst_name,kind,file_id,line) VALUES (3,'should_retry','ref',3,1)",[]).unwrap();
            tx.commit().unwrap();
        }
        store
    }
    fn read(store: &Store) -> Snapshot {
        store
            .checked_read_snapshot(&|| Ok(()), |tx| snapshot(tx, &|| Ok(())))
            .unwrap()
            .unwrap()
    }
    #[test]
    fn rollback_noop_and_cap_retention() {
        let store = fixture();
        let globs = linker::test_paths(&["tests/**".into()]).unwrap();
        let before = read(&store);
        let calls = Cell::new(0);
        assert!(matches!(
            store.link_documents(&globs, &|| Ok(()), &|| {
                calls.set(calls.get() + 1);
                if calls.get() == 2 {
                    Err(Error::L1InputsChanged)
                } else {
                    Ok(())
                }
            }),
            Err(Error::L1InputsChanged)
        ));
        assert_eq!(read(&store), before);
        assert!(
            store
                .link_documents(&globs, &|| Ok(()), &|| Ok(()))
                .unwrap()
        );
        let linked = read(&store);
        assert_eq!(linked.refs[0].kind, "test_of");
        assert_eq!(linked.synthetic[0].links.len(), 1);
        assert!(
            !store
                .link_documents(&globs, &|| Ok(()), &|| Ok(()))
                .unwrap()
        );
        assert_eq!(read(&store), linked);
        store
            .writer()
            .unwrap()
            .execute("UPDATE files SET max_level=0 WHERE id=1", [])
            .unwrap();
        assert!(
            !store
                .link_documents(&globs, &|| Ok(()), &|| Ok(()))
                .unwrap()
        );
        assert_eq!(read(&store).synthetic, linked.synthetic);
        store
            .writer()
            .unwrap()
            .execute("UPDATE files SET max_level=0 WHERE id=2", [])
            .unwrap();
        assert!(
            !store
                .link_documents(&globs, &|| Ok(()), &|| Ok(()))
                .unwrap()
        );
        assert_eq!(
            store
                .writer()
                .unwrap()
                .query_row("SELECT count(*) FROM symbols WHERE kind='doc'", [], |r| r
                    .get::<_, i64>(
                    0
                ))
                .unwrap(),
            1
        );
    }
    #[test]
    fn linker_growth_cap_reports_deferral_preserves_links_and_recovers() {
        let mut store = fixture();
        let root = tempfile::tempdir().unwrap();
        let settings = crate::config::Config::default();
        let run = |store: &Store| {
            crate::sync::linker::ingest_index(
                root.path(),
                &settings,
                store,
                Default::default(),
                &|| Ok(()),
            )
            .unwrap()
        };
        assert!(!run(&store).deferred);
        let before = read(&store).synthetic;
        store
            .writer()
            .unwrap()
            .execute(
                "UPDATE units SET header=?1 WHERE file_id=2",
                ["header ".repeat(80000)],
            )
            .unwrap();
        store.db_max_bytes = 0;
        let result = run(&store);
        assert!(result.deferred);
        assert_eq!(result.reason, Some("db_max_bytes"));
        assert_eq!(read(&store).synthetic, before);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(
                &store.get_meta("linker_status").unwrap().unwrap()
            )
            .unwrap()["state"],
            "capped"
        );
        store.db_max_bytes = crate::store::StoreLimits::default().db_max_bytes;
        assert!(!run(&store).deferred);
        assert_eq!(read(&store).synthetic[0].id, before[0].id);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(
                &store.get_meta("linker_status").unwrap().unwrap()
            )
            .unwrap()["state"],
            "completed"
        );
    }
}
