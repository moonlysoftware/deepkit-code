//! Directory composite inventories and atomic publication; v1.md §6.6–6.7.
use super::{Error, Store, UnitInsert, admission, insert_unit_in_transaction};
use crate::treat::module;
use rusqlite::{OptionalExtension, Transaction, params};
use std::collections::{BTreeSet, HashMap};

type Check<'a> = &'a dyn Fn() -> Result<(), Error>;

/// File identity and completed treatment generation captured with module text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Source {
    pub(crate) id: i64,
    pub(crate) path: String,
    pub(crate) hash: Vec<u8>,
    pub(crate) size: i64,
    pub(crate) language: Option<String>,
    pub(crate) kind: String,
    pub(crate) max_level: i64,
    pub(crate) fingerprint: Option<Vec<u8>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{FtsColumns, UnitKind};

    fn unit<'a>(text: &'a module::Collated, key: &'a str) -> UnitInsert<'a> {
        UnitInsert {
            hash: &text.hash,
            source_key: Some(key),
            duplicate_ordinal: 0,
            file_id: None,
            kind: UnitKind::Module,
            byte_start: None,
            byte_end: None,
            line_start: None,
            line_end: None,
            scope_path: Some("repository"),
            header: &text.header,
            text: &text.text,
            tokens: 1,
            ladder_level: 1,
            parent_unit: None,
            created_at: 1,
            updated_at: 1,
            fts: FtsColumns {
                header: &text.header,
                idents: "",
                body: &text.text,
                prose: &text.text,
            },
        }
    }

    #[test]
    fn batch_own_publication_remains_reusable_but_external_source_change_refuses_stale_payload() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("code.db");
        let store = crate::store::open(&path, Default::default()).unwrap();
        store.writer().unwrap().execute_batch("INSERT INTO files(id,path,blake3,size,mtime_ns,kind,max_level,treatment,l1_pending,l1_desired,l1_fingerprint) VALUES
            (1,'a/file.rs',zeroblob(32),1,1,'code',2,'code',0,X'01',X'01'),
            (2,'b/file.rs',zeroblob(32),1,1,'code',2,'code',0,X'01',X'01');
            INSERT INTO dirs(path,merkle,file_count) VALUES ('a',zeroblob(32),1),('b',zeroblob(32),1);
            INSERT INTO meta(key,value) VALUES ('graph_pending','0') ON CONFLICT(key) DO UPDATE SET value='0';").unwrap();
        let mut batch = store.capture_modules(1, &|| Ok(()), &|| Ok(())).unwrap();
        let a = batch.snapshots[0].1.clone().unwrap();
        let b = batch.snapshots[1].1.clone().unwrap();
        let text = module::collate(&a.input);
        assert!(
            store
                .publish_module(
                    &mut batch,
                    &a,
                    1,
                    unit(&text, "module:a"),
                    &|| Ok(()),
                    &|| Ok(())
                )
                .unwrap()
        );
        {
            let mut writer = store.writer().unwrap();
            let tx = writer.transaction().unwrap();
            assert!(batch.matches(&tx).unwrap());
        }
        let external = rusqlite::Connection::open(&path).unwrap();
        external
            .execute("UPDATE files SET path='b/renamed.rs' WHERE id=2", [])
            .unwrap();
        let text = module::collate(&b.input);
        let result = store.publish_module(
            &mut batch,
            &b,
            1,
            unit(&text, "module:b"),
            &|| Ok(()),
            &|| Ok(()),
        );
        assert!(matches!(result, Err(Error::L1InputsChanged)));
        assert_eq!(
            external
                .query_row("SELECT count(*) FROM units WHERE kind='module'", [], |r| {
                    r.get::<_, i64>(0)
                })
                .unwrap(),
            1
        );
        let pending = store.capture_modules(1, &|| Ok(()), &|| Ok(())).unwrap();
        assert!(pending.snapshots[1].1.is_none());
        assert_eq!(
            external
                .query_row("SELECT l1_pending FROM files WHERE id=2", [], |r| r
                    .get::<_, i64>(0))
                .unwrap(),
            1
        );
        // Complete the fixture's source repair before retrying collation.
        external
            .execute("UPDATE files SET l1_pending=0 WHERE id=2", [])
            .unwrap();
        let mut retry = store.capture_modules(1, &|| Ok(()), &|| Ok(())).unwrap();
        let b = retry.snapshots[1].1.clone().unwrap();
        let text = module::collate(&b.input);
        assert!(
            store
                .publish_module(
                    &mut retry,
                    &b,
                    1,
                    unit(&text, "module:b"),
                    &|| Ok(()),
                    &|| Ok(())
                )
                .unwrap()
        );
    }

    #[test]
    fn batch_revision_rejects_local_writes_and_external_commits() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("code.db");
        let store = crate::store::open(&path, Default::default()).unwrap();
        let mut batch = store.capture_modules(5, &|| Ok(()), &|| Ok(())).unwrap();
        {
            let mut writer = store.writer().unwrap();
            let tx = writer.transaction().unwrap();
            assert!(batch.matches(&tx).unwrap());
        }
        store
            .writer()
            .unwrap()
            .execute("INSERT INTO meta(key,value) VALUES ('batch-local','1')", [])
            .unwrap();
        {
            let mut writer = store.writer().unwrap();
            let tx = writer.transaction().unwrap();
            assert!(!batch.matches(&tx).unwrap());
        }
        let mut batch = store.capture_modules(5, &|| Ok(()), &|| Ok(())).unwrap();
        let before = store.writer().unwrap().total_changes();
        let external = rusqlite::Connection::open(&path).unwrap();
        external
            .execute(
                "INSERT INTO meta(key,value) VALUES ('batch-external','1')",
                [],
            )
            .unwrap();
        let mut writer = store.writer().unwrap();
        assert_eq!(writer.total_changes(), before);
        let tx = writer.transaction().unwrap();
        assert!(!batch.matches(&tx).unwrap());
    }

    #[test]
    fn root_batch_inventory_matches_individual_module_snapshots() {
        let root = tempfile::tempdir().unwrap();
        let store = crate::store::open(root.path().join("code.db"), Default::default()).unwrap();
        store.writer().unwrap().execute_batch("INSERT INTO files(id,path,blake3,size,mtime_ns,kind,max_level,treatment,l1_pending,l1_desired,l1_fingerprint) VALUES
            (1,'alpha/one.rs',zeroblob(32),1,1,'code',2,'code',0,X'01',X'01'),
            (2,'alpha/two.rs',zeroblob(32),1,1,'code',2,'code',0,X'01',X'01'),
            (3,'beta/three.rs',zeroblob(32),1,1,'code',2,'code',0,X'01',X'01');
            INSERT INTO dirs(path,merkle,file_count) VALUES
            ('',zeroblob(32),3),('alpha',zeroblob(32),2),('beta',zeroblob(32),1);
            INSERT INTO unit_contents(hash,text,tokens) VALUES (zeroblob(32),'symbol',1);
            INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,ladder_level,created_at,updated_at)
            VALUES (10,X'01',zeroblob(32),'alpha/one.rs',1,'symbol','one',1,1,1);
            INSERT INTO symbols(id,file_id,unit_id,name,qualified,kind,line_start,line_end,pagerank)
            VALUES (10,1,10,'one','alpha::one','function',1,1,0.5);
            INSERT INTO meta(key,value) VALUES ('graph_pending','0') ON CONFLICT(key) DO UPDATE SET value='0';").unwrap();
        let paths = ["", "alpha", "beta"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        assert!(
            store
                .read_snapshot(|tx| snapshot(tx, "", 1, &|| Ok(())))
                .unwrap()
                .unwrap()
                .is_some()
        );
        let batch = store
            .read_snapshot(|tx| snapshots(tx, &paths, 1, &|| Ok(())))
            .unwrap()
            .unwrap();
        assert_eq!(batch.len(), 3);
        for actual in batch {
            let expected = store
                .read_snapshot(|tx| snapshot(tx, &actual.path, 1, &|| Ok(())))
                .unwrap()
                .unwrap()
                .unwrap();
            assert_eq!(actual, expected, "{}", actual.path);
        }
    }
}

/// Complete current owner and sources; equality fences names, ranks and edges too.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Snapshot {
    pub(crate) path: String,
    pub(crate) unit: Option<i64>,
    pub(crate) sources: Vec<Source>,
    pub(crate) readme_unit: Option<i64>,
    pub(crate) input: module::Input,
}

/// Writer-connection revision for short-lived reuse of database-derived reads.
/// Local writes advance `changes`; commits from other connections advance
/// `external`, which must be read on this same long-lived writer connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Revision {
    changes: u64,
    external: i64,
}

impl Revision {
    fn read(connection: &rusqlite::Connection) -> Result<Self, Error> {
        Ok(Self {
            changes: connection.total_changes(),
            external: connection.query_row("PRAGMA data_version", [], |r| r.get(0))?,
        })
    }
}

/// Captured on the writer connection. Local changes and external commits both
/// invalidate reuse; only this batch's independent module publications advance it.
pub(crate) struct Batch {
    revision: Option<Revision>,
    pub(crate) snapshots: Vec<(String, Option<Snapshot>)>,
}

impl Batch {
    fn matches(&mut self, tx: &Transaction<'_>) -> Result<bool, Error> {
        let same = self.revision == Some(Revision::read(tx)?);
        if !same {
            self.revision = None;
        }
        Ok(same)
    }
}

/// Enumerates qualifying subtrees in stable order, including the repository root.
pub(crate) fn paths(tx: &Transaction<'_>, minimum: u64) -> Result<Vec<String>, Error> {
    Ok(tx
        .prepare("SELECT path FROM dirs WHERE file_count>=?1 ORDER BY path")?
        .query_map([minimum.min(i64::MAX as u64) as i64], |r| r.get(0))?
        .collect::<Result<_, _>>()?)
}

/// Tests subtree membership with a separator boundary, including the root.
pub(super) fn within(path: &str, directory: &str) -> bool {
    directory.is_empty()
        || path
            .strip_prefix(directory)
            .is_some_and(|tail| tail.starts_with('/'))
}

/// Names the containing directory; root-level files use the empty path.
pub(super) fn parent(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(parent, _)| parent)
}

// A root module already covers every file. Share its database inventory with
// selected descendants instead of rereading overlapping subtrees and symbols.
struct BatchInventory {
    files: Vec<(Source, bool, bool)>,
    by_path: HashMap<String, usize>,
    symbols: HashMap<i64, Vec<module::Symbol>>,
}

impl BatchInventory {
    fn load(tx: &Transaction<'_>, check: Check<'_>) -> Result<Self, Error> {
        let mut files = Vec::new();
        let mut by_path = HashMap::new();
        let mut statement = tx.prepare("SELECT id,path,blake3,size,lang,kind,max_level,l1_fingerprint,coalesce(l1_pending=0 AND (max_level<1 OR (l1_fingerprint=l1_desired AND (l1_deferred_reason IS NULL OR (kind='doc' AND needs_vlm=1 AND l1_deferred_reason IN ('needs_vlm','no_renderer'))))),0),coalesce(l1_pending=0 AND l1_fingerprint=l1_desired AND l1_deferred_reason IS NULL,0) FROM files ORDER BY path")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            check()?;
            let source = Source {
                id: row.get(0)?,
                path: row.get(1)?,
                hash: row.get(2)?,
                size: row.get(3)?,
                language: row.get(4)?,
                kind: row.get(5)?,
                max_level: row.get(6)?,
                fingerprint: row.get(7)?,
            };
            by_path.insert(source.path.clone(), files.len());
            files.push((source, row.get(8)?, row.get(9)?));
        }
        let mut symbols: HashMap<i64, Vec<module::Symbol>> = HashMap::new();
        let mut statement = tx.prepare("SELECT s.file_id,s.name,coalesce(s.qualified,s.name),s.pagerank,s.line_start FROM symbols s JOIN units u ON u.id=s.unit_id WHERE u.file_id=s.file_id AND u.kind IN ('symbol','chunk') AND s.kind!='doc' ORDER BY s.file_id,s.name,s.line_start,s.id")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            check()?;
            symbols
                .entry(row.get(0)?)
                .or_default()
                .push(module::Symbol {
                    name: row.get(1)?,
                    qualified: row.get(2)?,
                    rank: row.get(3)?,
                    line: row.get(4)?,
                });
        }
        Ok(Self {
            files,
            by_path,
            symbols,
        })
    }
}

/// Captures only coherent L1/graph inputs. Missing or incomplete contributors
/// refuse the whole module rather than silently publishing a smaller source set.
pub(crate) fn snapshot(
    tx: &Transaction<'_>,
    path: &str,
    minimum: u64,
    check: Check<'_>,
) -> Result<Option<Snapshot>, Error> {
    snapshot_with_graph(tx, path, minimum, check, None, None)
}

/// Captures the requested directories with one graph inventory in this transaction.
/// Each caller must capture anew for its final source/publication observation.
pub(crate) fn snapshots(
    tx: &Transaction<'_>,
    paths: &BTreeSet<String>,
    minimum: u64,
    check: Check<'_>,
) -> Result<Vec<Snapshot>, Error> {
    if paths.is_empty() {
        return Ok(Vec::new());
    }
    if paths.len() == 1
        && let Some(path) = paths.first()
    {
        return Ok(snapshot(tx, path, minimum, check)?.into_iter().collect());
    }
    let graph =
        super::module_graph::Graph::load(tx, &paths.iter().map(String::as_str).collect(), check)?;
    let inventory = paths
        .contains("")
        .then(|| BatchInventory::load(tx, check))
        .transpose()?;
    let mut result = Vec::new();
    for path in paths {
        check()?;
        if let Some(snapshot) =
            snapshot_with_graph(tx, path, minimum, check, Some(&graph), inventory.as_ref())?
        {
            result.push(snapshot);
        }
    }
    Ok(result)
}

fn snapshot_with_graph(
    tx: &Transaction<'_>,
    path: &str,
    minimum: u64,
    check: Check<'_>,
    graph: Option<&super::module_graph::Graph>,
    inventory: Option<&BatchInventory>,
) -> Result<Option<Snapshot>, Error> {
    check()?;
    let dir: Option<(i64, Option<i64>)> = tx
        .query_row(
            "SELECT file_count,module_unit FROM dirs WHERE path=?1",
            [path],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?;
    let Some((count, unit)) = dir else {
        return Ok(None);
    };
    if count < 0 || (count as u64) < minimum {
        return Ok(None);
    }
    let pending: Option<String> = tx
        .query_row(
            "SELECT value FROM meta WHERE key='graph_pending'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    if pending.as_deref() != Some("0") {
        return Ok(None);
    }
    let mut result = Snapshot {
        path: path.into(),
        unit,
        sources: Vec::new(),
        readme_unit: None,
        input: module::Input::default(),
    };
    // A completely probed scanned PDF contributes its filename/hash at L0.
    // Waiting for optional VLM output must not suppress its directory module;
    // capped, pending and failed parser sources remain incomplete.
    let file_rows = if let Some(inventory) = inventory {
        inventory
            .files
            .iter()
            .filter(|(source, _, _)| within(&source.path, path))
            .map(|(source, complete, _)| (source.clone(), *complete))
            .collect::<Vec<_>>()
    } else {
        let mut files = tx.prepare("SELECT id,path,blake3,size,lang,kind,max_level,l1_fingerprint,coalesce(l1_pending=0 AND (max_level<1 OR (l1_fingerprint=l1_desired AND (l1_deferred_reason IS NULL OR (kind='doc' AND needs_vlm=1 AND l1_deferred_reason IN ('needs_vlm','no_renderer'))))),0) FROM files WHERE ?1='' OR substr(path,1,length(?1)+1)=?1||'/' ORDER BY path")?;
        files
            .query_map([path], |r| {
                Ok((
                    Source {
                        id: r.get(0)?,
                        path: r.get(1)?,
                        hash: r.get(2)?,
                        size: r.get(3)?,
                        language: r.get(4)?,
                        kind: r.get(5)?,
                        max_level: r.get(6)?,
                        fingerprint: r.get(7)?,
                    },
                    r.get::<_, bool>(8)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?
    };
    let mut symbols_statement = tx.prepare_cached("SELECT s.name,coalesce(s.qualified,s.name),s.pagerank,s.line_start FROM symbols s JOIN units u ON u.id=s.unit_id WHERE s.file_id=?1 AND u.file_id=s.file_id AND u.kind IN ('symbol','chunk') AND s.kind!='doc' ORDER BY s.name,s.line_start,s.id")?;
    for (source, complete) in file_rows {
        check()?;
        if !complete || source.hash.len() != 32 {
            return Ok(None);
        }
        let symbols = if let Some(inventory) = inventory {
            inventory
                .symbols
                .get(&source.id)
                .cloned()
                .unwrap_or_default()
        } else {
            symbols_statement
                .query_map([source.id], |r| {
                    Ok(module::Symbol {
                        name: r.get(0)?,
                        qualified: r.get(1)?,
                        rank: r.get(2)?,
                        line: r.get(3)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?
        };
        let relative = if path.is_empty() {
            source.path.as_str()
        } else {
            &source.path[path.len() + 1..]
        };
        result.input.files.push(module::File {
            path: relative.into(),
            hash: source.hash.clone(),
            symbols,
        });
        // Choose a README in this directory, then its earliest completed section.
        let basename = source.path.rsplit('/').next().unwrap_or(&source.path);
        if result.readme_unit.is_none()
            && parent(&source.path) == path
            && (basename.eq_ignore_ascii_case("README")
                || basename.to_ascii_lowercase().starts_with("readme."))
        {
            let readme:Option<(i64,Vec<u8>,String)>=tx.query_row("SELECT u.id,u.content_hash,c.text FROM units u JOIN unit_contents c ON c.hash=u.content_hash WHERE u.file_id=?1 AND u.kind='doc_section' AND u.ladder_level>=1 ORDER BY u.byte_start,u.duplicate_ordinal,u.id LIMIT 1",[source.id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()?;
            if let Some((id, hash, text)) = readme {
                result.readme_unit = Some(id);
                result.input.readme = Some(module::Readme { hash, text });
            }
        }
        result.sources.push(source);
    }
    if result.sources.len() as i64 != count {
        return Ok(None);
    }
    // Match the rank graph: resolved destinations win; unresolved names expand
    // to known same-name symbols. Names without any candidate add no edge.
    let loaded;
    let graph = if let Some(graph) = graph {
        graph
    } else {
        loaded = super::module_graph::Graph::load(tx, &[path].into_iter().collect(), check)?;
        &loaded
    };
    let boundary = graph.boundary(path);
    if let Some(boundary) = boundary {
        result.input.depends_on.clone_from(&boundary.depends_on);
        result.input.used_by.clone_from(&boundary.used_by);
    }
    // Graph-derived directory labels also inherit the contributor's content and
    // policy. They are provenance, not additional files in the module's subtree.
    let mut source_statement = tx.prepare_cached("SELECT id,path,blake3,size,lang,kind,max_level,l1_fingerprint,coalesce(l1_pending=0 AND l1_fingerprint=l1_desired AND l1_deferred_reason IS NULL,0) FROM files WHERE path=?1")?;
    for path in boundary.into_iter().flat_map(|boundary| &boundary.sources) {
        check()?;
        let row = if let Some(inventory) = inventory {
            inventory.by_path.get(path).map(|&index| {
                let (source, _, complete) = &inventory.files[index];
                (source.clone(), *complete)
            })
        } else {
            source_statement
                .query_row([path], |r| {
                    Ok((
                        Source {
                            id: r.get(0)?,
                            path: r.get(1)?,
                            hash: r.get(2)?,
                            size: r.get(3)?,
                            language: r.get(4)?,
                            kind: r.get(5)?,
                            max_level: r.get(6)?,
                            fingerprint: r.get(7)?,
                        },
                        r.get::<_, bool>(8)?,
                    ))
                })
                .optional()?
        };
        let Some((source, true)) = row else {
            return Ok(None);
        };
        if source.hash.len() != 32 {
            return Ok(None);
        }
        result.input.graph_hashes.push(source.hash.clone());
        result.sources.push(source);
    }
    check()?;
    Ok(Some(result))
}

/// Verifies exact occurrence-specific provenance, rejecting hidden extra sources.
pub(crate) fn sources_match(
    tx: &Transaction<'_>,
    unit: i64,
    snapshot: &Snapshot,
) -> Result<bool, Error> {
    let actual = tx
        .prepare("SELECT source_kind,source_id,role FROM unit_sources WHERE unit_id=?1")?
        .query_map([unit], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, Option<String>>(2)?,
            ))
        })?
        .collect::<Result<BTreeSet<_>, _>>()?;
    let mut expected = snapshot
        .sources
        .iter()
        .map(|s| ("file".into(), s.id, Some("content".into())))
        .collect::<BTreeSet<_>>();
    if let Some(id) = snapshot.readme_unit {
        expected.insert(("unit".into(), id, Some("content".into())));
    }
    Ok(actual == expected)
}

fn unchanged(
    tx: &Transaction<'_>,
    snapshot: &Snapshot,
    unit: &UnitInsert<'_>,
) -> Result<bool, Error> {
    let Some(id) = snapshot.unit else {
        return Ok(false);
    };
    let same=tx.query_row("SELECT kind='module' AND file_id IS NULL AND content_hash=?2 AND scope_path IS ?3 AND header=?4 AND parent_unit IS NULL FROM units WHERE id=?1",params![id,unit.hash,unit.scope_path,unit.header],|r|r.get::<_,bool>(0)).optional()?.unwrap_or(false);
    Ok(same && sources_match(tx, id, snapshot)?)
}

impl Store {
    /// Reads a cooperative writer-connection revision so one request may reuse
    /// database-derived work only while both local and external writes are absent.
    pub(crate) fn revision_checked(&self, check: Check<'_>) -> Result<Revision, Error> {
        super::checked_writer::with_writer(self, check, |writer| Revision::read(writer))
    }

    /// Captures all directories with one graph traversal. BEGIN IMMEDIATE fences
    /// external writers during capture; no mutation or growth admission is needed.
    pub(crate) fn capture_modules(
        &self,
        minimum: u64,
        check: Check<'_>,
        validate: Check<'_>,
    ) -> Result<Batch, Error> {
        super::checked_writer::with_writer(self, check, |writer| {
            let tx = super::checked_writer::begin(writer, check)?;
            validate()?;
            let paths = paths(&tx, minimum)?;
            let graph = super::module_graph::Graph::load(
                &tx,
                &paths.iter().map(String::as_str).collect(),
                check,
            )?;
            let mut snapshots = Vec::new();
            for path in paths {
                let captured = snapshot_with_graph(&tx, &path, minimum, check, Some(&graph), None)?;
                snapshots.push((path, captured));
            }
            validate()?;
            Ok(Batch {
                revision: Some(Revision::read(&tx)?),
                snapshots,
            })
        })
    }

    /// Retires composites whose source changed after their subtree ceased to
    /// qualify. Raising only the selection threshold retains the old occurrence.
    pub(crate) fn retire_changed_modules(
        &self,
        minimum: u64,
        check: Check<'_>,
        validate: Check<'_>,
    ) -> Result<usize, Error> {
        let obsolete = |tx: &Transaction<'_>| -> Result<Vec<i64>, Error> {
            let candidates = tx.prepare("SELECT u.id,u.content_hash,d.path FROM units u LEFT JOIN dirs d ON d.module_unit=u.id WHERE u.kind='module' AND (d.path IS NULL OR d.file_count<?1)")?
                .query_map([minimum.min(i64::MAX as u64) as i64], |r| Ok((r.get::<_,i64>(0)?,r.get::<_,Vec<u8>>(1)?,r.get::<_,Option<String>>(2)?)))?
                .collect::<Result<Vec<_>,_>>()?;
            let mut ids = Vec::new();
            for (id, hash, path) in candidates {
                check()?;
                let changed = match path {
                    None => true,
                    Some(path) => snapshot(tx, &path, 0, check)?.is_some_and(|current| {
                        module::collate(&current.input).hash.as_slice() != hash
                    }),
                };
                if changed {
                    ids.push(id);
                }
            }
            Ok(ids)
        };
        validate()?;
        if self.checked_read_snapshot(check, obsolete)??.is_empty() {
            return Ok(0);
        }
        admission::transaction_checked(self, admission::Kind::L1Growth, Some(check), |tx| {
            validate()?;
            let ids = obsolete(&tx)?;
            for id in &ids {
                check()?;
                tx.execute("DELETE FROM units WHERE id=?1", [id])?;
            }
            validate()?;
            tx.commit()?;
            Ok(ids.len())
        })
    }

    /// Replaces changed module text and all derived rows in one admitted commit.
    /// No-op publication makes no writes; refusal retains the previous occurrence.
    pub(crate) fn publish_module(
        &self,
        batch: &mut Batch,
        captured: &Snapshot,
        minimum: u64,
        unit: UnitInsert<'_>,
        check: Check<'_>,
        validate: Check<'_>,
    ) -> Result<bool, Error> {
        validate()?;
        if super::checked_writer::with_writer(self, check, |writer| {
            let tx = super::checked_writer::begin(writer, check)?;
            if !batch.matches(&tx)?
                && snapshot(&tx, &captured.path, minimum, check)?.as_ref() != Some(captured)
            {
                return Err(Error::L1InputsChanged);
            }
            unchanged(&tx, captured, &unit)
        })? {
            validate()?;
            return Ok(false);
        }
        admission::transaction_checked(self, admission::Kind::L1Growth, Some(check), |tx| {
            let reusable = batch.matches(&tx)?;
            if !reusable
                && snapshot(&tx, &captured.path, minimum, check)?.as_ref() != Some(captured)
            {
                return Err(Error::L1InputsChanged);
            }
            validate()?;
            if let Some(old) = captured.unit {
                let same: bool = tx.query_row("SELECT content_hash=?2 AND source_key IS ?3 AND duplicate_ordinal=?4 AND kind='module' AND file_id IS NULL AND parent_unit IS NULL FROM units WHERE id=?1", params![old,unit.hash,unit.source_key,unit.duplicate_ordinal], |r|r.get(0))?;
                if same && sources_match(&tx, old, captured)? {
                    // Context changes refresh the occurrence/FTS through upsert;
                    // vectors and summaries remain attached to the same payload.
                    tx.execute("DELETE FROM unit_sources WHERE unit_id=?1", [old])?;
                } else {
                    tx.execute("DELETE FROM units WHERE id=?1", [old])?;
                }
            }
            let id = insert_unit_in_transaction(&tx, unit, None)?;
            for source in &captured.sources {
                check()?;
                tx.execute("INSERT INTO unit_sources(unit_id,source_kind,source_id,role) VALUES (?1,'file',?2,'content')",params![id,source.id])?;
            }
            if let Some(readme) = captured.readme_unit {
                tx.execute("INSERT INTO unit_sources(unit_id,source_kind,source_id,role) VALUES (?1,'unit',?2,'content')",params![id,readme])?;
            }
            tx.execute(
                "UPDATE dirs SET module_unit=?1 WHERE path=?2",
                params![id, captured.path],
            )?;
            check()?;
            validate()?;
            let revision = Revision::read(&tx)?;
            tx.commit()?;
            if reusable {
                batch.revision = Some(revision);
            }
            Ok(true)
        })
    }
}
