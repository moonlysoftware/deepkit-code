//! Persisted L2 coverage and potential outbound dispatches; v1 §7.1/§13.
use super::Error;
use crate::{
    config::Config,
    store::{Store, UnitKind, embedding_bindings, transitions, vectors::vector_table_name},
};
use serde::Serialize;

/// Physical table inventory; retained transition rows are counted explicitly.
#[derive(Debug, Serialize)]
pub(crate) struct Table {
    slug: String,
    model: String,
    profile: String,
    dims: i64,
    active: bool,
    vectors: i64,
    context: crate::store::unit_contexts::Counts,
    /// Persisted route metadata, not a claim of current configuration or freshness.
    recorded_desired_kinds: Vec<String>,
    retained_transition_kinds: Vec<String>,
    /// None means association was unavailable or not supplied (see Coverage);
    /// empty kinds mean configuration resolved but no proven table match exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    configured: Option<Configured>,
}
/// Current selected role/kinds, associated only through validated provider provenance.
#[derive(Debug, Default, Serialize)]
struct Configured {
    roles: Vec<&'static str>,
    kinds: Vec<String>,
}
/// Configuration-only failures must not suppress persisted inventory diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Association {
    NotSupplied,
    Available,
    Unavailable,
}
/// Persisted vector coverage, independent of current filesystem/policy eligibility.
#[derive(Debug, Serialize)]
pub(crate) struct Coverage {
    configured_association: Association,
    indexed_units: i64,
    covered_units: i64,
    physical_vectors: i64,
    pub(crate) pending_jobs: i64,
    tables: Vec<Table>,
    transition_generation: Option<i64>,
}
/// Inputs authorized for potential dispatch, including retries, not proof of
/// receipt. Counts survive model/configuration changes and publication failure.
#[derive(Debug, Serialize)]
pub(crate) struct Endpoint {
    kind: String,
    url: String,
    sent_units: i64,
    dispatches: i64,
    unresolved_dispatches: i64,
}
/// One read snapshot; older records with no endpoint identity stay unknown.
#[derive(Debug, Serialize)]
pub(crate) struct Snapshot {
    pub(crate) l2: Coverage,
    pub(crate) endpoints: Vec<Endpoint>,
    pub(crate) unknown_endpoint_dispatches: i64,
}
impl Snapshot {
    /// Formats inventory and conservative outbound accounting for terminal status.
    pub(crate) fn summary(&self) -> String {
        let mut lines = vec![format!(
            "L2: {} covered units, {} physical vectors, {} pending jobs",
            self.l2.covered_units, self.l2.physical_vectors, self.l2.pending_jobs
        )];
        if self.l2.configured_association == Association::Unavailable {
            lines.push(
                "L2 configured association unavailable: embedding recipe could not be resolved"
                    .into(),
            );
        }
        for table in &self.l2.tables {
            let configured = table.configured.as_ref().map_or_else(
                || match self.l2.configured_association {
                    Association::Unavailable => "unavailable".to_owned(),
                    _ => "not supplied".to_owned(),
                },
                |route| format!("{} [{}]", route.roles.join(","), route.kinds.join(",")),
            );
            lines.push(format!("L2 table {}: {} physical vectors; configured {}; recorded desired [{}]; retained transition [{}]",
                table.slug, table.vectors, configured, table.recorded_desired_kinds.join(","), table.retained_transition_kinds.join(",")));
        }
        for endpoint in &self.endpoints {
            lines.push(format!(
                "{} {}: {} input dispatches ({} unresolved attempts)",
                endpoint.kind, endpoint.url, endpoint.sent_units, endpoint.unresolved_dispatches
            ));
        }
        lines.push(format!("Outbound counts include potential dispatches/retries; {} dispatches lack historical endpoint identity",self.unknown_endpoint_dispatches));
        lines.join("\n")
    }
}

/// Reads actual table rows and dispatched reservations without resolving current
/// endpoints or causing model loads, provider calls, or queue maintenance.
pub(crate) fn read(store: &Store) -> Result<Snapshot, Error> {
    read_with_config(store, None)
}

/// Adds supplied effective configuration to the existing physical inventory.
/// Bindings and transition associations are read in that inventory's transaction;
/// annotations require no extra vector queries or provider/configuration writes.
pub(crate) fn read_configured(store: &Store, config: &Config) -> Result<Snapshot, Error> {
    read_with_config(store, Some(config))
}

fn read_with_config(store: &Store, config: Option<&Config>) -> Result<Snapshot, Error> {
    let configured = config
        .map(|config| {
            UnitKind::ALL
                .into_iter()
                .map(|kind| {
                    let role =
                        if config.embedder.docs.as_ref().is_some_and(|docs| {
                            docs.kinds.iter().any(|value| value == kind.as_str())
                        }) {
                            "docs"
                        } else {
                            "primary"
                        };
                    super::destination::resolve(config, kind)
                        .map(|resolved| (kind.as_str(), role, resolved.binding(config)))
                })
                .collect::<Result<Vec<_>, Error>>()
        })
        .transpose();
    let (configured, configured_association) = match configured {
        Ok(Some(routes)) => (Some(routes), Association::Available),
        Ok(None) => (None, Association::NotSupplied),
        Err(_) => (None, Association::Unavailable),
    };
    store.read_snapshot(|tx| {
        let transition = transitions::read(tx)?;
        let configured = configured.as_ref().map(|routes| routes.iter().map(|(kind, role, binding)| {
            embedding_bindings::select(tx, binding).map(|slug| (*kind, *role, binding, slug))
        }).collect::<Result<Vec<_>, _>>()).transpose()?;
        let metadata = tx.prepare("SELECT slug,model,profile,dims,active FROM embedding_tables ORDER BY slug")?
            .query_map([], |r| Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?,r.get::<_,String>(2)?,r.get::<_,i64>(3)?,r.get::<_,bool>(4)?)))?
            .collect::<Result<Vec<_>,_>>()?;
        let mut tables = Vec::new();
        let mut branches = Vec::new();
        let mut physical_vectors = 0_i64;
        for (slug, model, profile, dims, active) in metadata {
            let name = vector_table_name(&slug)?;
            let vectors: i64 = tx.query_row(&format!("SELECT count(*) FROM {name}"), [], |r|r.get(0))?;
            physical_vectors = physical_vectors.checked_add(vectors).ok_or(Error::Invalid("vector count overflow"))?;
            branches.push(format!("SELECT unit_id FROM {name}"));
            let binding = embedding_bindings::read(tx, &slug)?;
            if binding.as_ref().is_some_and(|binding| binding.model != model || binding.profile != profile || i64::try_from(binding.dims).ok() != Some(dims)) {
                return Err(Error::Invalid("vector binding metadata mismatch"));
            }
            let configured = configured.as_ref().map(|routes| {
                let mut result = Configured::default();
                for (kind, role, expected, selected) in routes {
                    if selected == &slug && binding.as_ref().is_some_and(|binding| binding.same_space(expected)) {
                        result.kinds.push((*kind).to_owned());
                        if !result.roles.contains(role) { result.roles.push(*role); }
                    }
                }
                result
            });
            let recorded_desired_kinds = transition.as_ref().map_or_else(Vec::new, |state| state.desired.iter().filter(|(_, table)| *table == &slug).map(|(kind, _)| kind.clone()).collect());
            let retained_transition_kinds = transition.as_ref().map_or_else(Vec::new, |state| state.previous.iter().filter(|source| source.table == slug).flat_map(|source| source.kinds.iter().cloned()).collect());
            let context = crate::store::unit_contexts::counts(tx, &slug, config)?;
            tables.push(Table { slug, model, profile, dims, active, vectors, context, recorded_desired_kinds, retained_transition_kinds, configured });
        }
        let covered_units = if branches.is_empty() { 0 } else {
            tx.query_row(&format!("SELECT count(*) FROM ({})", branches.join(" UNION ")), [], |r|r.get(0))?
        };
        let indexed_units = tx.query_row("SELECT count(*) FROM units", [], |r|r.get(0))?;
        let pending_jobs = tx.query_row("SELECT count(*) FROM jobs WHERE kind='embed' AND state IN ('queued','leased')", [], |r|r.get(0))?;
        let endpoints = tx.prepare("SELECT endpoint_kind,endpoint_url,SUM(sent_units),count(*),SUM(state='dispatched') FROM usage_reservations WHERE state IN ('dispatched','reconciled') AND endpoint_url IS NOT NULL AND endpoint_kind IS NOT NULL AND sent_units IS NOT NULL GROUP BY endpoint_kind,endpoint_url ORDER BY endpoint_kind,endpoint_url")?
            .query_map([], |r| Ok(Endpoint { kind:r.get(0)?,url:r.get(1)?,sent_units:r.get(2)?,dispatches:r.get(3)?,unresolved_dispatches:r.get(4)? }))?
            .collect::<Result<Vec<_>,_>>()?;
        let unknown_endpoint_dispatches = tx.query_row("SELECT count(*) FROM usage_reservations WHERE state IN ('dispatched','reconciled') AND provider NOT IN ('embed:local','rerank:local') AND (endpoint_url IS NULL OR endpoint_kind IS NULL OR sent_units IS NULL)", [], |r|r.get(0))?;
        Ok::<_, Error>(Snapshot { l2:Coverage { configured_association, indexed_units, covered_units, physical_vectors, pending_jobs, tables, transition_generation: transition.map(|state| state.generation) }, endpoints, unknown_endpoint_dispatches })
    })?
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        queue::usage,
        store::{self, StoreLimits},
    };
    use std::error::Error as StdError;

    #[test]
    fn local_inference_is_not_unknown_outbound_traffic() -> Result<(), Box<dyn StdError>> {
        let store = crate::store::open(":memory:", crate::store::StoreLimits::default())?;
        for provider in ["embed:local", "rerank:local", "old-unknown-provider"] {
            store.writer()?.execute("INSERT INTO usage_reservations(id,provider,day,calls,tokens,state,created_at) VALUES(?1,?1,'1970-01-01',1,10,'dispatched',0)", [provider])?;
        }
        let report = read(&store)?;
        assert!(report.endpoints.is_empty());
        assert_eq!(report.unknown_endpoint_dispatches, 1);
        Ok(())
    }

    #[test]
    fn endpoint_metadata_rolls_back_with_quota_refusal_and_counts_uncertainty()
    -> Result<(), Box<dyn StdError>> {
        let store = store::open(":memory:", StoreLimits::default())?;
        let budget = usage::Budget {
            calls: None,
            tokens: None,
        };
        let reservation =
            match usage::reserve(&store, "embed:remote:test", 3, 0, budget, None, || 0)? {
                usage::Admission::Reserved(r) => r,
                _ => return Err("no reservation".into()),
            };
        let endpoint = usage::Endpoint {
            kind: "embed",
            url: "https://example.invalid/v1/embeddings".into(),
            units: 2,
        };
        let outcome = usage::mark_dispatched(
            &store,
            &reservation,
            None,
            || 0,
            |tx, _| {
                usage::record_endpoint(tx, &reservation, &endpoint)?;
                Ok(usage::Limits {
                    budget,
                    rpm: Some(0),
                    tpm: None,
                })
            },
        )?;
        assert_eq!(outcome, usage::Dispatch::RequestTooLarge);
        assert_eq!(
            store.writer()?.query_row(
                "SELECT count(*) FROM usage_reservations WHERE endpoint_url IS NOT NULL",
                [],
                |r| r.get::<_, i64>(0)
            )?,
            0
        );
        assert!(read(&store)?.endpoints.is_empty());
        assert_eq!(
            usage::mark_dispatched(
                &store,
                &reservation,
                None,
                || 1,
                |tx, _| {
                    usage::record_endpoint(tx, &reservation, &endpoint)?;
                    Ok(usage::Limits {
                        budget,
                        rpm: None,
                        tpm: None,
                    })
                }
            )?,
            usage::Dispatch::Recorded
        );
        let report = read(&store)?;
        assert_eq!(report.endpoints[0].sent_units, 2);
        assert_eq!(report.endpoints[0].unresolved_dispatches, 1);
        assert_eq!(report.endpoints[0].dispatches, 1);
        assert_eq!(
            usage::mark_dispatched(
                &store,
                &reservation,
                None,
                || 2,
                |_, _| Ok(usage::Limits {
                    budget,
                    rpm: None,
                    tpm: None
                })
            )?,
            usage::Dispatch::Unavailable
        );
        // Missing historical metadata is unknown, never assigned the current URL.
        store.writer()?.execute(
            "UPDATE usage_reservations SET endpoint_url=NULL,endpoint_kind=NULL,sent_units=NULL",
            [],
        )?;
        let old = read(&store)?;
        assert!(old.endpoints.is_empty());
        assert_eq!(old.unknown_endpoint_dispatches, 1);
        Ok(())
    }
    #[test]
    fn coverage_counts_occurrences_once_across_retained_tables() -> Result<(), Box<dyn StdError>> {
        use crate::store::vectors::{VectorInsert, VectorTableSpec};
        let (_root, store, _config) = crate::queue::source::tests::fixture()?;
        let hash =
            store
                .writer()?
                .query_row("SELECT content_hash FROM units WHERE id=1", [], |r| {
                    r.get::<_, Vec<u8>>(0)
                })?;
        for (slug, active) in [("old", false), ("new", true)] {
            store.ensure_vector_table(VectorTableSpec {
                slug,
                model: slug,
                profile: "local",
                dims: 2,
                created_at: 0,
                active,
            })?;
            store.insert_vector(VectorInsert {
                table_slug: slug,
                unit_id: 1,
                ctx_hash: &[0; 32],
                embedding_hash: &hash,
                embedding: &[1.0, 0.0],
            })?;
        }
        let snapshot = read(&store)?;
        assert_eq!(snapshot.l2.indexed_units, 1);
        assert_eq!(snapshot.l2.covered_units, 1);
        assert_eq!(snapshot.l2.physical_vectors, 2);
        assert_eq!(snapshot.l2.tables.len(), 2);
        assert_eq!(snapshot.l2.pending_jobs, 0);
        assert!(snapshot.endpoints.is_empty());
        Ok(())
    }
    #[test]
    fn configured_roles_and_retained_kinds_share_physical_inventory_without_writes()
    -> Result<(), Box<dyn StdError>> {
        use crate::store::vectors::{VectorInsert, VectorTableSpec};
        use std::collections::{BTreeMap, BTreeSet};
        let (_root, store, mut config) = crate::queue::source::tests::fixture()?;
        config.embedder.profile = "remote".into();
        config.embedder.dims = 2;
        config.embedder.docs = Some(crate::config::DocsEmbedder {
            profile: "ollama".into(),
            kinds: vec!["image".into()],
        });
        let mut slugs = Vec::new();
        let hash: Vec<u8> =
            store
                .writer()?
                .query_row("SELECT content_hash FROM units WHERE id=1", [], |row| {
                    row.get(0)
                })?;
        for kind in [UnitKind::Symbol, UnitKind::Image] {
            let binding = super::super::destination::resolve(&config, kind)?.binding(&config);
            let slug = binding.destination_slug();
            store.ensure_vector_table(VectorTableSpec {
                slug: &slug,
                model: &binding.model,
                profile: &binding.profile,
                dims: 2,
                created_at: 0,
                active: true,
            })?;
            store.bind_embedding_table(&slug, &binding)?;
            store.insert_vector(VectorInsert {
                table_slug: &slug,
                unit_id: 1,
                ctx_hash: &[0; 32],
                embedding_hash: &hash,
                embedding: &[1.0, 0.0],
            })?;
            slugs.push(slug);
        }
        let desired: BTreeMap<String, String> = UnitKind::ALL
            .into_iter()
            .map(|kind| {
                (
                    kind.as_str().into(),
                    slugs[usize::from(kind == UnitKind::Image)].clone(),
                )
            })
            .collect();
        let previous = vec![transitions::Source {
            table: slugs[0].clone(),
            kinds: BTreeSet::from(["image".into()]),
            known: true,
        }];
        store.writer()?.execute("INSERT INTO embedding_transition(singleton,generation,desired,previous) VALUES(1,3,?1,?2)", rusqlite::params![serde_json::to_string(&desired)?, serde_json::to_string(&previous)?])?;
        let changes = store.writer()?.total_changes();
        let report = read_configured(&store, &config)?;
        assert_eq!(report.l2.covered_units, 1);
        assert_eq!(report.l2.physical_vectors, 2);
        assert_eq!(report.l2.transition_generation, Some(3));
        let primary = report
            .l2
            .tables
            .iter()
            .find(|table| table.slug == slugs[0])
            .ok_or("primary")?;
        let docs = report
            .l2
            .tables
            .iter()
            .find(|table| table.slug == slugs[1])
            .ok_or("docs")?;
        assert_eq!(
            primary.configured.as_ref().ok_or("configured")?.roles,
            ["primary"]
        );
        assert_eq!(
            primary.configured.as_ref().ok_or("configured")?.kinds.len(),
            UnitKind::ALL.len() - 1
        );
        assert_eq!(primary.retained_transition_kinds, ["image"]);
        assert_eq!(
            docs.configured.as_ref().ok_or("configured")?.roles,
            ["docs"]
        );
        assert_eq!(
            docs.configured.as_ref().ok_or("configured")?.kinds,
            ["image"]
        );
        assert_eq!(docs.recorded_desired_kinds, ["image"]);
        assert!(docs.retained_transition_kinds.is_empty());
        assert!(report.summary().contains("2 physical vectors"));
        // A current endpoint change cannot relabel historical provider provenance.
        config.embedder.remote.url = "https://changed.example.invalid/v1".into();
        let changed = read_configured(&store, &config)?;
        let primary = changed
            .l2
            .tables
            .iter()
            .find(|table| table.slug == slugs[0])
            .ok_or("primary")?;
        assert!(
            primary
                .configured
                .as_ref()
                .ok_or("configured")?
                .kinds
                .is_empty()
        );
        assert_eq!(primary.retained_transition_kinds, ["image"]);
        assert_eq!(store.writer()?.total_changes(), changes);
        Ok(())
    }

    #[test]
    fn unknown_legacy_table_is_never_assigned_configured_provenance()
    -> Result<(), Box<dyn StdError>> {
        use crate::store::vectors::VectorTableSpec;
        let store = store::open(":memory:", StoreLimits::default())?;
        let config = Config::default();
        let binding =
            super::super::destination::resolve(&config, UnitKind::Symbol)?.binding(&config);
        let slug = binding.destination_slug();
        store.ensure_vector_table(VectorTableSpec {
            slug: &slug,
            model: &binding.model,
            profile: &binding.profile,
            dims: binding.dims as usize,
            created_at: 0,
            active: true,
        })?;
        let report = read_configured(&store, &config)?;
        assert!(
            report.l2.tables[0]
                .configured
                .as_ref()
                .ok_or("configured")?
                .kinds
                .is_empty()
        );
        assert!(report.l2.transition_generation.is_none());
        assert!(read(&store)?.l2.tables[0].configured.is_none());
        store.bind_embedding_table(&slug, &binding)?;
        assert_eq!(
            read_configured(&store, &config)?.l2.tables[0]
                .configured
                .as_ref()
                .ok_or("configured")?
                .kinds
                .len(),
            UnitKind::ALL.len()
        );
        store.writer()?.execute(
            "UPDATE embedding_tables SET model='mismatched' WHERE slug=?1",
            [&slug],
        )?;
        assert!(matches!(
            read_configured(&store, &config),
            Err(Error::Invalid("vector binding metadata mismatch"))
        ));
        Ok(())
    }
    #[test]
    fn unresolvable_configuration_preserves_inventory_but_not_corrupt_metadata()
    -> Result<(), Box<dyn StdError>> {
        use crate::store::vectors::VectorTableSpec;
        let store = store::open(":memory:", StoreLimits::default())?;
        store.ensure_vector_table(VectorTableSpec {
            slug: "legacy",
            model: "fixture",
            profile: "local",
            dims: 2,
            created_at: 0,
            active: true,
        })?;
        let mut config = Config::default();
        config.embedder.profile = "local".into();
        config.embedder.local.model = "private-unresolvable-model-fixture".into();
        let changes = store.writer()?.total_changes();
        let report = read_configured(&store, &config)?;
        assert_eq!(report.l2.configured_association, Association::Unavailable);
        assert_eq!(report.l2.tables.len(), 1);
        assert_eq!(report.l2.physical_vectors, 0);
        assert!(report.l2.tables[0].configured.is_none());
        assert!(
            report
                .summary()
                .contains("configured association unavailable")
        );
        assert!(
            !report
                .summary()
                .contains("private-unresolvable-model-fixture")
        );
        let serialized = serde_json::to_string(&report)?;
        assert!(serialized.contains("\"configured_association\":\"unavailable\""));
        assert!(!serialized.contains("private-unresolvable-model-fixture"));
        assert_eq!(
            read(&store)?.l2.configured_association,
            Association::NotSupplied
        );
        assert_eq!(store.writer()?.total_changes(), changes);
        // Persisted routing corruption remains an error even when config cannot resolve.
        store.writer()?.execute("INSERT INTO embedding_transition(singleton,generation,desired,previous) VALUES(1,1,'{}','[]')", [])?;
        assert!(read_configured(&store, &config).is_err());
        Ok(())
    }
}
