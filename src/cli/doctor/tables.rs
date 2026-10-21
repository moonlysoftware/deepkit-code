//! Read-only physical vector diagnostics in doctor's checked snapshot; v1 §7.3/§13.
use super::Finding;
use crate::{
    config::Config,
    store::{UnitKind, embedding_bindings, transitions, vectors::vector_table_name},
};
use rusqlite::Transaction;
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// Makes missing snapshot coverage explicit without guessing table health.
pub(super) fn unavailable() -> Finding {
    Finding::new(
        "vectors.tables",
        "not_checked",
        "Vector table checks require a readable compatible database",
    )
}

/// Checks selected and retained tables without model loads, registration or writes.
/// Unassociated metadata is inventoried but never causes a second-table read in
/// steady single mode. Row shape is not a claim of numerical quality or freshness.
pub(super) fn inspect(tx: &Transaction<'_>, config: &Config) -> Finding {
    match inspect_in(tx, config) {
        Ok((status, evidence)) => {
            let mut finding = Finding::new(
                "vectors.tables",
                status,
                "Selected and retained vector table metadata and FLOAT16 row shapes checked; unassociated tables are not read",
            );
            finding.evidence = Some(evidence);
            finding
        }
        Err(_) => Finding::new(
            "vectors.tables",
            "failed",
            "Vector table inventory or routing metadata is unreadable or inconsistent",
        ),
    }
}

fn inspect_in(
    tx: &Transaction<'_>,
    config: &Config,
) -> Result<(&'static str, Value), anyhow::Error> {
    let mut desired = BTreeMap::new();
    for kind in UnitKind::ALL {
        let binding = crate::queue::destination::resolve(config, kind)?.binding(config);
        desired.insert(
            kind.as_str(),
            (embedding_bindings::select(tx, &binding)?, binding),
        );
    }
    let transition = transitions::read(tx)?;
    let metadata = tx
        .prepare(
            "SELECT slug,model,profile,dims,active FROM embedding_tables ORDER BY slug LIMIT 129",
        )?
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, i64>(4)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    anyhow::ensure!(metadata.len() <= 128, "vector metadata capacity exceeded");
    let mut failed = false;
    let mut warning = false;
    let mut tables = Vec::new();
    for (ordinal, (slug, model, profile, dims, active)) in metadata.iter().enumerate() {
        let selected: Vec<_> = desired
            .iter()
            .filter(|(_, (table, _))| table == slug)
            .collect();
        let retained: Vec<_> = transition.as_ref().map_or_else(Vec::new, |state| {
            state
                .previous
                .iter()
                .filter(|source| source.table == *slug)
                .flat_map(|source| source.kinds.iter().cloned())
                .collect()
        });
        let recorded: Vec<_> = transition.as_ref().map_or_else(Vec::new, |state| {
            state
                .desired
                .iter()
                .filter(|(_, table)| *table == slug)
                .map(|(kind, _)| kind.clone())
                .collect()
        });
        if selected.is_empty() && retained.is_empty() && recorded.is_empty() {
            warning = true;
            tables.push(json!({"ordinal":ordinal+1,"status":"not_checked","association":"unassociated","physical_rows_read":false}));
            continue;
        }
        let result = check_table(tx, slug, model, profile, *dims, *active);
        let (status, rows, provenance) = match result {
            Ok((rows, Some(binding)))
                if selected
                    .iter()
                    .all(|(_, (_, expected))| binding.same_space(expected)) =>
            {
                if *active == 0 && (!selected.is_empty() || !recorded.is_empty()) {
                    warning = true;
                    ("warning", Some(rows), "known")
                } else {
                    ("passed", Some(rows), "known")
                }
            }
            Ok((rows, None)) => {
                warning = true;
                ("warning", Some(rows), "unknown")
            }
            _ => {
                failed = true;
                ("failed", None, "inconsistent")
            }
        };
        tables.push(json!({"ordinal":ordinal+1,"status":status,"configured_kinds":selected.iter().map(|(kind,_)| **kind).collect::<Vec<_>>(),"retained_kinds":retained,"recorded_desired_kinds":recorded,"dimensions":dims,"active":*active == 1,"rows":rows,"provenance":provenance,"physical_rows_read":true}));
    }
    let missing_retained = transition.as_ref().map_or(0, |state| {
        state
            .previous
            .iter()
            .map(|source| &source.table)
            .filter(|slug| !metadata.iter().any(|row| &row.0 == *slug))
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    });
    failed |= missing_retained > 0;
    let pending_kinds: Vec<_> = desired
        .iter()
        .filter(|(_, (slug, _))| !metadata.iter().any(|row| &row.0 == slug))
        .map(|(kind, _)| *kind)
        .collect();
    warning |= !metadata.is_empty() && !pending_kinds.is_empty();
    Ok((
        if failed {
            "failed"
        } else if warning {
            "warning"
        } else {
            "passed"
        },
        json!({"tables":tables,"transition_generation":transition.map(|state| state.generation),"missing_retained_tables":missing_retained,"unmaterialized_configured_kinds":pending_kinds}),
    ))
}

fn check_table(
    tx: &Transaction<'_>,
    slug: &str,
    model: &str,
    profile: &str,
    dims: i64,
    active: i64,
) -> Result<(i64, Option<embedding_bindings::Binding>), anyhow::Error> {
    anyhow::ensure!(
        dims > 0 && dims <= i64::from(i32::MAX) && matches!(active, 0 | 1),
        "invalid metadata"
    );
    let table = vector_table_name(slug)?;
    let name = format!("vec_{slug}");
    let ordinary: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM pragma_table_list WHERE schema='main' AND name=?1 AND type='table')", [&name], |r| r.get(0))?;
    anyhow::ensure!(ordinary, "physical table missing or virtual");
    let columns = tx
        .prepare("SELECT name,type,pk,[notnull] FROM pragma_table_info(?1)")?
        .query_map([&name], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, bool>(3)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    anyhow::ensure!(
        columns.iter().any(|(name, kind, pk, _)| name == "unit_id"
            && kind.eq_ignore_ascii_case("INTEGER")
            && *pk == 1)
            && ["v", "ctx_hash"].iter().all(|required| columns.iter().any(
                |(name, kind, _, required_value)| name == required
                    && kind.eq_ignore_ascii_case("BLOB")
                    && *required_value
            )),
        "unexpected vector schema"
    );
    let binding = embedding_bindings::read(tx, slug)?;
    anyhow::ensure!(
        binding.as_ref().is_none_or(|binding| binding.model == model
            && binding.profile == profile
            && i64::try_from(binding.dims).ok() == Some(dims)),
        "binding mismatch"
    );
    let (rows, invalid): (i64, i64) = tx.query_row(&format!("SELECT count(*),coalesce(sum(CASE WHEN typeof(vectors.unit_id)!='integer' OR typeof(v)!='blob' OR length(v)!=?1 OR typeof(ctx_hash)!='blob' OR length(ctx_hash)!=32 OR NOT EXISTS(SELECT 1 FROM units u WHERE u.id=vectors.unit_id) THEN 1 ELSE 0 END),0) FROM {table} AS vectors"), [dims * 2], |r| Ok((r.get(0)?, r.get(1)?)))?;
    anyhow::ensure!(invalid == 0, "invalid row representation or dangling unit");
    Ok((rows, binding))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{self, Store, vectors::VectorTableSpec};

    fn fixture() -> (tempfile::TempDir, Store, Config, String) {
        let root = tempfile::tempdir().unwrap();
        let store = store::open(root.path().join(".deepkit/code.db"), Default::default()).unwrap();
        let config = Config::default();
        let binding = crate::queue::destination::resolve(&config, UnitKind::Chunk)
            .unwrap()
            .binding(&config);
        let slug = binding.destination_slug();
        store
            .ensure_vector_table(VectorTableSpec {
                slug: &slug,
                model: &binding.model,
                profile: &binding.profile,
                dims: binding.dims as usize,
                created_at: 1,
                active: true,
            })
            .unwrap();
        store.bind_embedding_table(&slug, &binding).unwrap();
        (root, store, config, slug)
    }

    fn diagnosis(store: &Store, config: &Config) -> Finding {
        store.read_snapshot(|tx| inspect(tx, config)).unwrap()
    }

    #[test]
    fn selected_table_is_checked_without_writes_or_provider_details() {
        let (_root, store, config, _) = fixture();
        let before = store.writer().unwrap().total_changes();
        let finding = diagnosis(&store, &config);
        assert_eq!(finding.status, "passed");
        let evidence = finding.evidence.unwrap();
        assert_eq!(evidence["tables"][0]["rows"], 0);
        assert_eq!(evidence["tables"][0]["provenance"], "known");
        assert!(!evidence.to_string().contains("https://"));
        assert_eq!(before, store.writer().unwrap().total_changes());
    }

    #[test]
    fn single_mode_inventories_but_never_reads_unrelated_physical_table() {
        let (_root, store, config, _) = fixture();
        store.writer().unwrap().execute("INSERT INTO embedding_tables(slug,model,profile,dims,created_at,active) VALUES('unrelated','private-model','remote',2,1,1)", []).unwrap();
        let finding = diagnosis(&store, &config);
        assert_eq!(finding.status, "warning");
        let evidence = finding.evidence.unwrap();
        let unrelated = evidence["tables"]
            .as_array()
            .unwrap()
            .iter()
            .find(|table| table["association"] == "unassociated")
            .unwrap();
        assert_eq!(unrelated["physical_rows_read"], false);
        assert!(!evidence.to_string().contains("private-model"));
    }

    #[test]
    fn planned_missing_destination_is_pending_but_missing_retained_source_fails() {
        let (_root, store, mut config, primary) = fixture();
        config.embedder.docs = Some(crate::config::DocsEmbedder {
            profile: "ollama".into(),
            kinds: vec!["pr".into()],
        });
        let docs = crate::queue::destination::resolve(&config, UnitKind::PullRequest)
            .unwrap()
            .binding(&config)
            .destination_slug();
        let desired: BTreeMap<_, _> = UnitKind::ALL
            .into_iter()
            .map(|kind| {
                (
                    kind.as_str(),
                    if kind == UnitKind::PullRequest {
                        docs.as_str()
                    } else {
                        primary.as_str()
                    },
                )
            })
            .collect();
        store.writer().unwrap().execute("INSERT INTO embedding_transition(singleton,generation,desired,previous) VALUES(1,1,?1,'[]')", [serde_json::to_string(&desired).unwrap()]).unwrap();
        let pending = diagnosis(&store, &config);
        assert_eq!(pending.status, "warning");
        assert_eq!(
            pending.evidence.unwrap()["unmaterialized_configured_kinds"],
            json!(["pr"])
        );
        let previous = vec![transitions::Source {
            table: "missing_retained".into(),
            kinds: std::collections::BTreeSet::from(["pr".into()]),
            known: false,
        }];
        store
            .writer()
            .unwrap()
            .execute(
                "UPDATE embedding_transition SET previous=?1",
                [serde_json::to_string(&previous).unwrap()],
            )
            .unwrap();
        assert_eq!(diagnosis(&store, &config).status, "failed");
    }

    #[test]
    fn inactive_selected_table_is_not_reported_healthy() {
        let (_root, store, config, _) = fixture();
        store
            .writer()
            .unwrap()
            .execute("UPDATE embedding_tables SET active=0", [])
            .unwrap();
        let finding = diagnosis(&store, &config);
        assert_eq!(finding.status, "warning");
        assert_eq!(finding.evidence.unwrap()["tables"][0]["active"], false);
    }

    #[test]
    fn malformed_empty_schema_cannot_pass_row_shape_check() {
        let (_root, store, config, slug) = fixture();
        let table = vector_table_name(&slug).unwrap();
        store
            .writer()
            .unwrap()
            .execute_batch(&format!(
                "DROP TABLE {table}; CREATE TABLE {table}(unit_id TEXT,v BLOB,ctx_hash BLOB);"
            ))
            .unwrap();
        assert_eq!(diagnosis(&store, &config).status, "failed");
    }

    #[test]
    fn selected_missing_table_and_bad_vector_shape_fail() {
        for missing in [true, false] {
            let (_root, store, config, slug) = fixture();
            let table = vector_table_name(&slug).unwrap();
            if missing {
                store
                    .writer()
                    .unwrap()
                    .execute_batch(&format!("DROP TABLE {table}"))
                    .unwrap();
            } else {
                store.writer().unwrap().execute_batch(&format!("INSERT INTO unit_contents(hash,text,tokens) VALUES(zeroblob(32),'fixture',1); INSERT INTO units(id,hash,content_hash,source_key,kind,header,created_at,updated_at) VALUES(1,zeroblob(32),zeroblob(32),'fixture','chunk','fixture',1,1); INSERT INTO {table}(unit_id,v,ctx_hash) VALUES(1,x'00',zeroblob(32));")).unwrap();
            }
            assert_eq!(diagnosis(&store, &config).status, "failed");
        }
    }

    fn populated_dual() -> (Store, Config, String, String) {
        use crate::store::vectors::VectorInsert;
        let store = store::open(":memory:", Default::default()).unwrap();
        let mut config = Config::default();
        config.embedder.profile = "local".into();
        config.embedder.dims = 2;
        config.embedder.docs = Some(crate::config::DocsEmbedder {
            profile: "remote".into(),
            kinds: vec!["pr".into()],
        });
        let mut slugs = Vec::new();
        for (id, kind) in [(1i64, UnitKind::Chunk), (2, UnitKind::PullRequest)] {
            let binding = crate::queue::destination::resolve(&config, kind)
                .unwrap()
                .binding(&config);
            let slug = binding.destination_slug();
            store
                .ensure_vector_table(VectorTableSpec {
                    slug: &slug,
                    model: &binding.model,
                    profile: &binding.profile,
                    dims: 2,
                    created_at: 1,
                    active: true,
                })
                .unwrap();
            store.bind_embedding_table(&slug, &binding).unwrap();
            let hash = blake3::hash(&id.to_le_bytes());
            store
                .writer()
                .unwrap()
                .execute(
                    "INSERT INTO unit_contents(hash,text,tokens) VALUES(?1,'fixture',1)",
                    [hash.as_bytes().as_slice()],
                )
                .unwrap();
            store.writer().unwrap().execute("INSERT INTO units(id,hash,content_hash,source_key,kind,header,created_at,updated_at) VALUES(?1,?2,?2,'fixture:' || ?1,?3,'fixture',1,1)",
                rusqlite::params![id, hash.as_bytes().as_slice(), kind.as_str()]).unwrap();
            store
                .insert_vector(VectorInsert {
                    table_slug: &slug,
                    unit_id: id,
                    ctx_hash: &[0; 32],
                    embedding_hash: hash.as_bytes(),
                    embedding: &[1.0, 0.0],
                })
                .unwrap();
            slugs.push(slug);
        }
        (store, config, slugs[0].clone(), slugs[1].clone())
    }

    #[test]
    fn healthy_local_and_remote_tables_report_independent_valid_row_counts() {
        let (store, config, primary, docs) = populated_dual();
        assert_ne!(primary, docs);
        let before = store.writer().unwrap().total_changes();
        let finding = diagnosis(&store, &config);
        assert_eq!(finding.status, "passed");
        let evidence = finding.evidence.unwrap();
        let tables = evidence["tables"].as_array().unwrap();
        assert_eq!(tables.len(), 2);
        for table in tables {
            assert_eq!(table["status"], "passed");
            assert_eq!(table["rows"], 1);
            assert_eq!(table["dimensions"], 2);
            assert_eq!(table["provenance"], "known");
            assert_eq!(table["physical_rows_read"], true);
        }
        assert!(
            tables
                .iter()
                .any(|table| table["configured_kinds"] == json!(["pr"]))
        );
        assert!(
            tables
                .iter()
                .any(|table| table["configured_kinds"].as_array().unwrap().len()
                    == UnitKind::ALL.len() - 1)
        );
        assert_eq!(evidence["unmaterialized_configured_kinds"], json!([]));
        assert_eq!(store.writer().unwrap().total_changes(), before);
    }

    #[test]
    fn corrupt_retained_row_or_orphan_fails_while_current_table_remains_healthy() {
        for orphan in [false, true] {
            let (store, mut config, primary, docs) = populated_dual();
            config.embedder.docs = None;
            let desired: BTreeMap<_, _> = UnitKind::ALL
                .into_iter()
                .map(|kind| (kind.as_str(), primary.as_str()))
                .collect();
            let previous = vec![transitions::Source {
                table: docs.clone(),
                kinds: std::collections::BTreeSet::from(["pr".into()]),
                known: true,
            }];
            store.writer().unwrap().execute("INSERT INTO embedding_transition(singleton,generation,desired,previous) VALUES(1,2,?1,?2)",
                rusqlite::params![serde_json::to_string(&desired).unwrap(), serde_json::to_string(&previous).unwrap()]).unwrap();
            let healthy = diagnosis(&store, &config);
            assert_eq!(healthy.status, "passed");
            let physical = vector_table_name(&docs).unwrap();
            {
                let writer = store.writer().unwrap();
                if orphan {
                    writer.pragma_update(None, "foreign_keys", false).unwrap();
                    writer
                        .execute(&format!("UPDATE {physical} SET unit_id=999"), [])
                        .unwrap();
                    writer.pragma_update(None, "foreign_keys", true).unwrap();
                } else {
                    writer
                        .execute(&format!("UPDATE {physical} SET v=x'00'"), [])
                        .unwrap();
                }
            }
            let before = store.writer().unwrap().total_changes();
            let finding = diagnosis(&store, &config);
            assert_eq!(finding.status, "failed");
            let evidence = finding.evidence.unwrap();
            let tables = evidence["tables"].as_array().unwrap();
            let retained = tables
                .iter()
                .find(|table| table["retained_kinds"] == json!(["pr"]))
                .unwrap();
            assert_eq!(retained["status"], "failed");
            assert_eq!(retained["physical_rows_read"], true);
            assert!(retained["rows"].is_null());
            let current = tables
                .iter()
                .find(|table| {
                    table["configured_kinds"]
                        .as_array()
                        .is_some_and(|kinds| kinds.len() == UnitKind::ALL.len())
                })
                .unwrap();
            assert_eq!(current["status"], "passed");
            assert_eq!(current["rows"], 1);
            assert_eq!(evidence["missing_retained_tables"], 0);
            assert_eq!(store.writer().unwrap().total_changes(), before);
        }
    }
}
