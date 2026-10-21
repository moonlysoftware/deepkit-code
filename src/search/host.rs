//! Lazy canonical query-provider hosting for public search; v1 §§7.3–7.5 and 9.
use super::{Error, control::Control, dense, embedding_query, rerank};
use crate::{
    config::{self, Config},
    models, queue,
    store::{self, Store, UnitKind, embedding_bindings, transitions, vectors},
};
use std::{collections::BTreeMap, path::PathBuf, sync::Arc};
use tokio::sync::Mutex;

/// Shared across requests, but replaced when effective configuration or store changes.
/// Construction does not inspect credentials, open models or start network work.
#[derive(Clone)]
pub(crate) struct Host {
    store: Arc<Store>,
    root: PathBuf,
    config: Config,
    dense: Arc<Mutex<Prepared>>,
    reranker: Arc<Mutex<Option<Arc<rerank::lazy::Host>>>>,
}
#[derive(Default)]
struct Prepared {
    generation: Option<i64>,
    factories: BTreeMap<String, Arc<embedding_query::Factory>>,
}
struct Space {
    kind: UnitKind,
    previous: bool,
    provider: String,
}
impl Host {
    /// Allocates reusable lazy state; request controls govern all later preparation.
    pub(crate) fn new(store: Arc<Store>, root: PathBuf, config: Config) -> Self {
        Self {
            store,
            root,
            config,
            dense: Arc::new(Mutex::new(Prepared::default())),
            reranker: Arc::new(Mutex::new(None)),
        }
    }
    /// Refuses reuse across repositories or effective configuration changes.
    pub(crate) fn matches(&self, store: &Arc<Store>, config: &Config) -> bool {
        Arc::ptr_eq(&self.store, store) && self.config == *config
    }
    /// Prepares only scoped spaces with coverage (plus incoming transition targets).
    /// Failed providers remain absent so routed retrieval reports per-space fallback.
    /// Call only for a request which actually selects a dense leg.
    pub(crate) async fn providers(
        &self,
        scope: &dense::Scope,
        control: &Control,
    ) -> Result<dense::routed::Providers, Error> {
        control.run(self.prepare(scope, control)).await?
    }
    async fn prepare(
        &self,
        scope: &dense::Scope,
        control: &Control,
    ) -> Result<dense::routed::Providers, Error> {
        let mut prepared = self.dense.lock().await;
        control.check()?;
        let host = self.clone();
        let kinds = scope.kinds.clone();
        let check = control.clone();
        let (generation, spaces) = tokio::task::spawn_blocking(move || {
            check.check()?;
            host.store.checked_read_snapshot(
                &|| check.check().map_err(|_| store::Error::L1InputsChanged),
                |tx| {
                    check.check()?;
                    let result = host.spaces(tx, &kinds);
                    check.check()?;
                    result
                },
            )?
        })
        .await
        .map_err(|_| Error::QueryTask)??;
        if prepared.generation != generation {
            prepared.factories.clear();
            prepared.generation = generation;
        }
        let selected: Vec<_> = spaces.keys().cloned().collect();
        for (table, space) in spaces {
            control.check()?;
            if prepared.factories.contains_key(&table) {
                continue;
            }
            // A failed tier does not prevent an independent configured tier from serving.
            if let Ok(factory) = self.factory(&table, space, generation).await
                && factory.table_slug() == table
            {
                prepared.factories.insert(table, factory);
            }
        }
        Ok(dense::routed::Providers {
            store: Arc::clone(&self.store),
            config_root: self.root.clone(),
            config: self.config.clone(),
            factories: selected
                .iter()
                .filter_map(|table| prepared.factories.get(table).cloned())
                .collect(),
        })
    }
    fn spaces(
        &self,
        tx: &rusqlite::Transaction<'_>,
        kinds: &[UnitKind],
    ) -> Result<(Option<i64>, BTreeMap<String, Space>), Error> {
        let mut spaces = BTreeMap::new();
        let Some(state) = transitions::read(tx)? else {
            return Ok((None, spaces));
        };
        let summaries = store::projection::summaries_in_scope(tx, kinds)?;
        for kind in UnitKind::ALL.into_iter().filter(|kind| {
            kinds.is_empty() || kinds.contains(kind) || (*kind == UnitKind::Summary && summaries)
        }) {
            if let Some(table) = state.desired.get(kind.as_str()) {
                let incoming = state
                    .previous
                    .iter()
                    .any(|old| old.kinds.contains(kind.as_str()));
                if (incoming || populated(tx, table, kind)?)
                    && let Ok(resolved) = queue::destination::resolve(&self.config, kind)
                {
                    spaces.insert(
                        table.clone(),
                        Space {
                            kind,
                            previous: false,
                            provider: resolved.identity.provider,
                        },
                    );
                }
            }
            for previous in &state.previous {
                if !previous.kinds.contains(kind.as_str())
                    || spaces.contains_key(&previous.table)
                    || !populated(tx, &previous.table, kind)?
                {
                    continue;
                }
                if let Some(binding) = embedding_bindings::read(tx, &previous.table)? {
                    spaces.insert(
                        previous.table.clone(),
                        Space {
                            kind,
                            previous: true,
                            provider: config::queue::embedding_provider(
                                &binding.profile,
                                &binding.endpoint,
                            ),
                        },
                    );
                }
            }
        }
        Ok((Some(state.generation), spaces))
    }
    async fn factory(
        &self,
        table: &str,
        space: Space,
        generation: Option<i64>,
    ) -> Result<Arc<embedding_query::Factory>, queue::Error> {
        let budget = queue::host::Bounds::for_provider(&self.config.queue, &space.provider)?.budget;
        let settings = embedding_query::Settings {
            config_root: self.root.clone(),
            config: self.config.clone(),
            kind: space.kind,
            budget,
        };
        let store = Arc::clone(&self.store);
        if space.previous {
            embedding_query::Factory::new_previous(
                store,
                settings,
                table.into(),
                generation.ok_or(queue::Error::Invalid("missing query routing generation"))?,
                cache()?,
            )
            .await
        } else if space.provider == "embed:local" {
            embedding_query::Factory::new_local(store, settings, cache()?).await
        } else {
            tokio::task::spawn_blocking(move || embedding_query::Factory::new(store, settings))
                .await
                .map_err(|_| queue::Error::WorkerTask)?
        }
    }
    /// Reuses first-use reranker preparation and its adaptive state across requests.
    /// This allocates the existing lazy host; model work starts only through `get`.
    pub(crate) async fn reranker(
        &self,
        control: &Control,
    ) -> Result<Arc<rerank::lazy::Host>, Error> {
        control
            .run(async {
                let mut retained = self.reranker.lock().await;
                if let Some(host) = &*retained {
                    return Ok(Arc::clone(host));
                }
                let cache = cache().map_err(|_| Error::DenseUnavailable)?;
                // Rerank has no configured daily/rate caps. Embedding endpoint budgets
                // do not authorize inventing limits for a different provider ledger.
                let settings = rerank::Settings {
                    config_root: self.root.clone(),
                    config: self.config.clone(),
                    limits: queue::usage::Limits {
                        budget: queue::usage::Budget {
                            calls: None,
                            tokens: None,
                        },
                        rpm: None,
                        tpm: None,
                    },
                };
                let host = rerank::lazy::Host::new(Arc::clone(&self.store), settings, cache);
                *retained = Some(Arc::clone(&host));
                Ok(host)
            })
            .await?
    }
}
fn cache() -> Result<PathBuf, queue::Error> {
    models::registry::cache_root().map_err(|_| queue::Error::Invalid("query model cache root"))
}
fn populated(
    tx: &rusqlite::Transaction<'_>,
    table: &str,
    kind: UnitKind,
) -> Result<bool, store::Error> {
    let exists: bool = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM embedding_tables WHERE slug=?1 AND active=1)",
        [table],
        |row| row.get(0),
    )?;
    if !exists {
        return Ok(false);
    }
    let name = vectors::vector_table_name(table)?;
    Ok(tx.query_row(
        &format!(
            "SELECT EXISTS(SELECT 1 FROM {name} v JOIN units u ON u.id=v.unit_id WHERE u.kind=?1)"
        ),
        [kind.as_str()],
        |row| row.get(0),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::control::Owner;
    use std::time::Instant;

    fn host() -> (tempfile::TempDir, Host) {
        let root = tempfile::tempdir().unwrap();
        let mut config = Config::default();
        config.embedder.profile = "local".into();
        // An unusable model proves an empty index never reaches model preparation.
        config.embedder.local.model = "missing-model".into();
        let store = Arc::new(store::open(":memory:", store::StoreLimits::default()).unwrap());
        (root, Host::new(store, PathBuf::from("/unused"), config))
    }
    #[tokio::test]
    async fn no_vector_coverage_never_initializes_local_models_or_reranker() {
        let (_root, host) = host();
        let owner = Owner::new(Instant::now(), 5000).unwrap();
        let providers = host
            .providers(&dense::Scope::default(), &owner.control())
            .await
            .unwrap();
        assert!(providers.factories.is_empty());
        assert!(host.reranker.lock().await.is_none());
        let table = "empty_local";
        host.store
            .prepare_transition(|_| {
                Ok::<_, store::Error>(
                    UnitKind::ALL
                        .into_iter()
                        .map(|kind| (kind.as_str().into(), table.into()))
                        .collect(),
                )
            })
            .unwrap();
        let providers = host
            .providers(&dense::Scope::default(), &owner.control())
            .await
            .unwrap();
        assert!(providers.factories.is_empty());
        assert!(host.reranker.lock().await.is_none());
    }
    #[tokio::test]
    async fn warmed_provider_cache_does_not_supply_empty_current_spaces() {
        use crate::store::vectors::{VectorInsert, VectorTableSpec};
        let (_root, mut host) = host();
        host.config.embedder.profile = "ollama".into();
        host.config.embedder.dims = 2;
        let binding = queue::destination::resolve(&host.config, UnitKind::Symbol)
            .unwrap()
            .binding(&host.config);
        let table = binding.destination_slug();
        host.store
            .ensure_vector_table(VectorTableSpec {
                slug: &table,
                model: &binding.model,
                profile: &binding.profile,
                dims: 2,
                active: true,
                created_at: 1,
            })
            .unwrap();
        host.store.bind_embedding_table(&table, &binding).unwrap();
        host.store
            .prepare_transition(|_| {
                Ok::<_, store::Error>(
                    UnitKind::ALL
                        .into_iter()
                        .map(|kind| (kind.as_str().into(), table.clone()))
                        .collect(),
                )
            })
            .unwrap();
        let unit =
            crate::search::tests::add(&host.store, "code.rs", "cached", "cached", None).unwrap();
        let content_hash: Vec<u8> = host
            .store
            .read_snapshot(|tx| {
                tx.query_row(
                    "SELECT content_hash FROM units WHERE id=?1",
                    [unit],
                    |row| row.get(0),
                )
            })
            .unwrap()
            .unwrap();
        host.store
            .insert_vector(VectorInsert {
                table_slug: &table,
                unit_id: unit,
                ctx_hash: &[1; 32],
                embedding_hash: &content_hash,
                embedding: &[1.0, 0.0],
            })
            .unwrap();
        let owner = Owner::new(Instant::now(), 5000).unwrap();
        let warmed = host
            .providers(&dense::Scope::default(), &owner.control())
            .await
            .unwrap();
        assert_eq!(warmed.factories.len(), 1);
        let no_docs = host
            .providers(
                &dense::Scope::parse(Some("docs"), Vec::new()),
                &owner.control(),
            )
            .await
            .unwrap();
        assert!(no_docs.factories.is_empty());
        // Keep reusable hosting state, but never mistake it for current coverage.
        assert_eq!(host.dense.lock().await.factories.len(), 1);
        let reused = host
            .providers(&dense::Scope::default(), &owner.control())
            .await
            .unwrap();
        assert!(Arc::ptr_eq(&warmed.factories[0], &reused.factories[0]));
        host.store.delete_vector(&table, unit).unwrap();
        let empty = host
            .providers(&dense::Scope::default(), &owner.control())
            .await
            .unwrap();
        assert!(empty.factories.is_empty());
        assert_eq!(host.dense.lock().await.factories.len(), 1);
    }
    #[tokio::test]
    async fn expired_preparation_does_not_initialize_hosts() {
        let (_root, host) = host();
        let owner = Owner::new(Instant::now(), 0).unwrap();
        assert!(matches!(
            host.providers(&dense::Scope::default(), &owner.control())
                .await,
            Err(Error::Deadline)
        ));
        assert!(matches!(
            host.reranker(&owner.control()).await,
            Err(Error::Deadline)
        ));
        assert!(host.reranker.lock().await.is_none());
    }
    #[test]
    fn provider_budgets_preserve_unlimited_and_explicit_zero() {
        let queue: config::Queue = serde_json::from_value(serde_json::json!({
            "budgets": [{"profile": "ollama", "url": "http://localhost:11434/",
                         "daily_calls": 0, "daily_tokens": 123}]
        }))
        .unwrap();
        let provider =
            config::queue::embedding_provider("ollama", "http://localhost:11434/api/embed");
        let bounded = queue::host::Bounds::for_provider(&queue, &provider)
            .unwrap()
            .budget;
        assert_eq!(bounded.calls, Some(0));
        assert_eq!(bounded.tokens, Some(123));
        let unlimited = queue::host::Bounds::for_provider(&queue, "embed:local")
            .unwrap()
            .budget;
        assert_eq!(unlimited.calls, None);
        assert_eq!(unlimited.tokens, None);
    }
    #[tokio::test]
    async fn cloned_hosts_retain_one_idle_reranker() {
        let (_root, host) = host();
        let owner = Owner::new(Instant::now(), 5000).unwrap();
        let first = host.reranker(&owner.control()).await.unwrap();
        let second = host.clone().reranker(&owner.control()).await.unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert!(matches!(first.peek(), rerank::lazy::Outcome::Idle));
        assert!(host.matches(&host.store, &host.config));
        let mut changed = host.config.clone();
        changed.embedder.dims += 1;
        assert!(!host.matches(&host.store, &changed));
    }
}
