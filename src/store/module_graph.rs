//! Count module boundary edges without expanding same-name symbol pairs; v1 §6.6.
use super::Error;
use rusqlite::Transaction;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

#[derive(Default)]
pub(super) struct Boundary {
    pub(super) sources: BTreeSet<String>,
    pub(super) depends_on: BTreeMap<String, usize>,
    pub(super) used_by: BTreeMap<String, usize>,
}

/// A coherent graph inventory shared by every directory in one read transaction.
#[derive(Default)]
pub(super) struct Graph(BTreeMap<String, Boundary>);

#[derive(Default)]
struct BoundaryBuilder {
    sources: HashSet<i64>,
    depends_on: HashMap<usize, usize>,
    used_by: HashMap<usize, usize>,
}

fn record_boundary(
    boundaries: &mut [BoundaryBuilder],
    memberships: &HashMap<i64, Vec<usize>>,
    parents: &HashMap<i64, usize>,
    source: i64,
    destination: i64,
    count: usize,
    check: &dyn Fn() -> Result<(), Error>,
) -> Result<(), Error> {
    check()?;
    let source_dirs = &memberships[&source];
    let destination_dirs = &memberships[&destination];
    if source_dirs == destination_dirs {
        return Ok(());
    }
    for &directory in source_dirs {
        if !destination_dirs.contains(&directory) {
            let boundary = &mut boundaries[directory];
            boundary.sources.insert(destination);
            *boundary
                .depends_on
                .entry(parents[&destination])
                .or_default() += count;
        }
    }
    for &directory in destination_dirs {
        if !source_dirs.contains(&directory) {
            let boundary = &mut boundaries[directory];
            boundary.sources.insert(source);
            *boundary.used_by.entry(parents[&source]).or_default() += count;
        }
    }
    Ok(())
}

fn parents(path: &str) -> impl Iterator<Item = &str> {
    std::iter::successors(path.rsplit_once('/').map(|(p, _)| p), |p| {
        p.rsplit_once('/').map(|(p, _)| p)
    })
}

impl Graph {
    /// Groups duplicate destinations before multiplying their edge counts. This
    /// preserves resolved-destination precedence and unresolved name multiplicity.
    pub(super) fn load(
        tx: &Transaction<'_>,
        directories: &BTreeSet<&str>,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<Self, Error> {
        check()?;
        // Root membership is universal, so no edge can cross its boundary.
        if directories.iter().all(|directory| directory.is_empty()) {
            return Ok(Self::default());
        }
        let files = tx
            .prepare("SELECT id,path FROM files")?
            .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?
            .collect::<Result<HashMap<_, _>, _>>()?;
        // Equal memberships cannot cross any requested boundary. In particular,
        // avoid expanding same-name pairs wholly outside a small requested scope.
        let directory_paths = directories.iter().copied().collect::<Vec<_>>();
        let directory_ids = directory_paths
            .iter()
            .enumerate()
            .map(|(id, &path)| (path, id))
            .collect::<HashMap<_, _>>();
        let memberships: HashMap<i64, Vec<usize>> = files
            .iter()
            .map(|(&id, path)| {
                (
                    id,
                    parents(path)
                        .filter_map(|dir| directory_ids.get(dir).copied())
                        .collect(),
                )
            })
            .collect();
        let mut parent_paths = Vec::new();
        let mut parent_ids = HashMap::new();
        let parents_by_file = files
            .iter()
            .map(|(&id, path)| {
                let parent = super::modules::parent(path);
                let parent_id = *parent_ids.entry(parent).or_insert_with(|| {
                    let next = parent_paths.len();
                    parent_paths.push(parent);
                    next
                });
                (id, parent_id)
            })
            .collect::<HashMap<_, _>>();
        let mut boundaries = (0..directory_paths.len())
            .map(|_| BoundaryBuilder::default())
            .collect::<Vec<_>>();
        // These inventories only add integer multiplicities. Final directory
        // counts and contributor sets stay ordered for canonical collation.
        let mut symbols = HashMap::new();
        let mut names: HashMap<String, [HashMap<i64, usize>; 2]> = HashMap::new();
        let mut statement = tx.prepare("SELECT id,file_id,name FROM symbols")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            check()?;
            let id = row.get::<_, i64>(0)?;
            let file = row.get::<_, i64>(1)?;
            if files.contains_key(&file) {
                symbols.insert(id, file);
                let inside = usize::from(!memberships[&file].is_empty());
                *names.entry(row.get(2)?).or_default()[inside]
                    .entry(file)
                    .or_default() += 1;
            }
        }
        let mut unresolved: HashMap<(i64, String), usize> = HashMap::new();
        // A single-scope freshness proof only needs edges that can touch that
        // subtree. Preserve unresolved-name multiplicity and resolved precedence;
        // the normal inventory checks below still reject inconsistent sources.
        let directory = (directories.len() == 1)
            .then(|| directories.iter().next().copied())
            .flatten();
        let sql = if directory.is_some() {
            "WITH members AS MATERIALIZED (
                SELECT id FROM files WHERE substr(path,1,length(?1)+1)=?1||'/'
             ), destinations AS MATERIALIZED (
                SELECT id,name FROM symbols WHERE file_id IN (SELECT id FROM members)
             ) SELECT file_id,src_symbol,dst_symbol,dst_name FROM edges
             WHERE file_id IN (SELECT id FROM members)
                OR dst_symbol IN (SELECT id FROM destinations)
                OR (dst_symbol IS NULL AND dst_name IN (SELECT name FROM destinations))"
        } else {
            "SELECT file_id,src_symbol,dst_symbol,dst_name FROM edges"
        };
        let mut statement = tx.prepare(sql)?;
        let mut rows = match directory {
            Some(path) => statement.query([path])?,
            None => statement.query([])?,
        };
        while let Some(row) = rows.next()? {
            check()?;
            let file = row.get::<_, i64>(0)?;
            if symbols.get(&row.get::<_, i64>(1)?) != Some(&file) {
                continue;
            }
            if let Some(destination) = row.get::<_, Option<i64>>(2)? {
                if let Some(&destination) = symbols.get(&destination)
                    && memberships[&file] != memberships[&destination]
                {
                    record_boundary(
                        &mut boundaries,
                        &memberships,
                        &parents_by_file,
                        file,
                        destination,
                        1,
                        check,
                    )?;
                }
            } else if let Some(name) = row.get::<_, Option<String>>(3)? {
                *unresolved.entry((file, name)).or_default() += 1;
            }
        }
        for ((source, name), count) in unresolved {
            check()?;
            if let Some(groups) = names.get(&name) {
                let source_inside = usize::from(!memberships[&source].is_empty());
                for (inside, destinations) in groups.iter().enumerate() {
                    // For one scope, whole same-side name groups are irrelevant.
                    // Skip before expanding their file multiplicities.
                    if directory.is_some() && inside == source_inside {
                        continue;
                    }
                    for (&destination, &multiplicity) in destinations {
                        if memberships[&source] != memberships[&destination] {
                            record_boundary(
                                &mut boundaries,
                                &memberships,
                                &parents_by_file,
                                source,
                                destination,
                                count * multiplicity,
                                check,
                            )?;
                        }
                    }
                }
            }
        }
        let mut graph = Self::default();
        for (index, builder) in boundaries.into_iter().enumerate() {
            check()?;
            if builder.sources.is_empty() {
                continue;
            }
            graph.0.insert(
                directory_paths[index].to_owned(),
                Boundary {
                    sources: builder
                        .sources
                        .into_iter()
                        .filter_map(|id| files.get(&id).cloned())
                        .collect(),
                    depends_on: builder
                        .depends_on
                        .into_iter()
                        .map(|(parent, count)| (parent_paths[parent].to_owned(), count))
                        .collect(),
                    used_by: builder
                        .used_by
                        .into_iter()
                        .map(|(parent, count)| (parent_paths[parent].to_owned(), count))
                        .collect(),
                },
            );
        }
        Ok(graph)
    }

    /// Returns only edges crossing this directory's subtree boundary.
    pub(super) fn boundary(&self, directory: &str) -> Option<&Boundary> {
        self.0.get(directory)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregated_boundaries_match_expanded_join_with_duplicates_and_resolved_precedence() {
        let mut db = rusqlite::Connection::open_in_memory().unwrap();
        db.execute_batch("CREATE TABLE files(id INTEGER PRIMARY KEY,path TEXT);
            CREATE TABLE symbols(id INTEGER PRIMARY KEY,file_id INTEGER,name TEXT);
            CREATE TABLE edges(file_id INTEGER,src_symbol INTEGER,dst_symbol INTEGER,dst_name TEXT);
            INSERT INTO files VALUES(1,'a/inner/one.rs'),(2,'a/two.rs'),(3,'ab/three.rs'),(4,'root.rs');
            INSERT INTO symbols VALUES(1,1,'same'),(2,2,'same'),(3,2,'same'),(4,3,'same'),(5,4,'root');
            INSERT INTO edges VALUES(1,1,NULL,'same'),(1,1,NULL,'same'),(1,1,4,'same'),
                (1,1,999,'same'),(3,4,NULL,'same'),(4,5,NULL,'same'),(3,4,5,'root'),
                (2,1,NULL,'same'),(1,999,NULL,'same'),(1,1,NULL,'absent');").unwrap();
        let tx = db.transaction().unwrap();
        let graph = Graph::load(
            &tx,
            &["", "a", "a/inner", "ab", "missing"].into_iter().collect(),
            &|| Ok(()),
        )
        .unwrap();
        let expanded = tx.prepare("SELECT sf.path,df.path FROM edges e JOIN symbols s ON s.id=e.src_symbol JOIN files sf ON sf.id=s.file_id JOIN symbols d ON (d.id=e.dst_symbol OR (e.dst_symbol IS NULL AND d.name=e.dst_name)) JOIN files df ON df.id=d.file_id WHERE e.file_id=sf.id").unwrap()
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))).unwrap()
            .collect::<Result<Vec<_>, _>>().unwrap();
        for directory in ["", "a", "a/inner", "ab", "missing"] {
            let mut expected = Boundary::default();
            for (src, dst) in &expanded {
                match (
                    crate::store::modules::within(src, directory),
                    crate::store::modules::within(dst, directory),
                ) {
                    (true, false) => {
                        expected.sources.insert(dst.clone());
                        *expected
                            .depends_on
                            .entry(crate::store::modules::parent(dst).into())
                            .or_default() += 1;
                    }
                    (false, true) => {
                        expected.sources.insert(src.clone());
                        *expected
                            .used_by
                            .entry(crate::store::modules::parent(src).into())
                            .or_default() += 1;
                    }
                    _ => {}
                }
            }
            let single = Graph::load(&tx, &[directory].into_iter().collect(), &|| Ok(())).unwrap();
            let empty = Boundary::default();
            for observed in [&graph, &single] {
                let actual = observed.boundary(directory).unwrap_or(&empty);
                assert_eq!(actual.sources, expected.sources, "{directory}");
                assert_eq!(actual.depends_on, expected.depends_on, "{directory}");
                assert_eq!(actual.used_by, expected.used_by, "{directory}");
            }
        }
    }

    #[test]
    fn root_or_empty_scope_still_observes_cancellation() {
        let mut db = rusqlite::Connection::open_in_memory().unwrap();
        let tx = db.transaction().unwrap();
        for directories in [BTreeSet::new(), [""].into_iter().collect()] {
            assert!(matches!(
                Graph::load(&tx, &directories, &|| Err(Error::L1InputsChanged)),
                Err(Error::L1InputsChanged)
            ));
            assert!(
                Graph::load(&tx, &directories, &|| Ok(()))
                    .unwrap()
                    .boundary("")
                    .is_none()
            );
        }
    }
}
