//! One host per selected provider, sharing rates and concurrency; v1 §7.4/§8.2.
use super::{Bounds, Host};
use crate::{
    config::Config,
    queue::{Error, destination, embedding::Destination},
    store::{Store, UnitKind},
};
use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

/// Configuration common to all selected providers; bounds are resolved explicitly
/// per provider before any table creation, model loading or credential resolution.
pub(crate) struct Settings {
    pub(crate) root: PathBuf,
    pub(crate) config_root: PathBuf,
    pub(crate) config: Config,
    pub(crate) local_cache: PathBuf,
}
/// A unique provider and its selected kinds. Callers keep independent catalogue
/// progress per entry; shared base/docs destinations use this one host.
pub(crate) struct Entry {
    pub(crate) provider: String,
    pub(crate) kinds: Vec<UnitKind>,
    pub(crate) host: Host,
}
/// Prepared hosts in deterministic provider order. Unselected profiles are never
/// instantiated; identical base/docs destinations share one native/client runner.
pub(crate) struct Group {
    pub(crate) entries: Vec<Entry>,
}
struct Selection {
    identity: Destination,
    kinds: Vec<UnitKind>,
    bounds: Bounds,
}
fn plan(
    config: &Config,
    kinds: &[UnitKind],
    bounds: impl Fn(&str) -> Result<Bounds, Error>,
    now: i64,
) -> Result<Vec<Selection>, Error> {
    let mut selected = BTreeMap::<String, (destination::Resolved, Vec<UnitKind>)>::new();
    for &kind in kinds {
        let resolved = destination::resolve(config, kind)?;
        if let Some((previous, kinds)) = selected.get_mut(&resolved.identity.provider) {
            // Config currently has one model/settings set per profile. Refuse an
            // incompatible future extension rather than split provider concurrency.
            if previous.identity.model != resolved.identity.model
                || previous.identity.endpoint != resolved.identity.endpoint
                || previous.identity.dims != resolved.identity.dims
                || previous.identity.artifact_hash != resolved.identity.artifact_hash
            {
                return Err(Error::Invalid(
                    "provider group has incompatible model destinations",
                ));
            }
            if !kinds.contains(&kind) {
                kinds.push(kind);
            }
        } else {
            selected.insert(resolved.identity.provider.clone(), (resolved, vec![kind]));
        }
    }
    selected
        .into_iter()
        .map(|(provider, (resolved, kinds))| {
            let bounds = bounds(&provider)?;
            bounds.validate(now)?;
            resolved.limits(bounds.budget)?;
            Ok(Selection {
                identity: resolved.identity,
                kinds,
                bounds,
            })
        })
        .collect()
}
impl Group {
    /// Preflights all provider bounds, then prepares each distinct destination once.
    /// Hosts may retain earlier empty tables on preparation failure, but no jobs
    /// are scheduled and no embedding calls are sent by this operation.
    pub(crate) async fn prepare(
        store: Arc<Store>,
        settings: Settings,
        kinds: &[UnitKind],
        bounds: impl Fn(&str) -> Result<Bounds, Error>,
        now: i64,
    ) -> Result<Self, Error> {
        let selected = plan(&settings.config, kinds, bounds, now)?;
        let mut entries = Vec::new();
        for selection in selected {
            let kind = selection
                .kinds
                .first()
                .copied()
                .ok_or(Error::Invalid("empty provider group"))?;
            let host = Host::prepare(
                Arc::clone(&store),
                super::Settings {
                    root: settings.root.clone(),
                    config_root: settings.config_root.clone(),
                    config: settings.config.clone(),
                    kind,
                    local_cache: settings.local_cache.clone(),
                    bounds: selection.bounds,
                },
                now,
            )
            .await?;
            entries.push(Entry {
                provider: selection.identity.provider,
                kinds: selection.kinds,
                host,
            });
        }
        Ok(Self { entries })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config, queue};
    use std::{error::Error as StdError, fs, time::Duration};
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };
    fn limits() -> Bounds {
        Bounds {
            budget: queue::usage::Budget {
                calls: None,
                tokens: None,
            },
            timeout: Duration::from_secs(2),
            heartbeat: Duration::from_secs(1),
            max_schedule: 8,
            max_jobs: 1,
        }
    }
    #[test]
    fn grouping_resolves_only_selected_kinds_and_coalesces_shared_profiles() -> Result<(), Error> {
        let mut config = Config::default();
        config.embedder.profile = "ollama".into();
        config.embedder.remote.url = "invalid unused URL".into();
        let shared = plan(
            &config,
            &[UnitKind::Chunk, UnitKind::DocSection, UnitKind::Chunk],
            |_| Ok(limits()),
            0,
        )?;
        assert_eq!(shared.len(), 1);
        assert_eq!(shared[0].kinds.len(), 2);
        config.embedder.docs = Some(config::DocsEmbedder {
            profile: "local".into(),
            kinds: vec!["doc_section".into()],
        });
        let distinct = plan(
            &config,
            &[UnitKind::Chunk, UnitKind::DocSection],
            |_| Ok(limits()),
            0,
        )?;
        assert_eq!(distinct.len(), 2);
        assert_ne!(distinct[0].identity.provider, distinct[1].identity.provider);
        config.embedder.local.model = "unused-invalid-model".into();
        assert_eq!(
            plan(&config, &[UnitKind::Chunk], |_| Ok(limits()), 0)?.len(),
            1
        );
        assert!(
            plan(
                &config,
                &[],
                |_| Err(Error::Invalid("unused budget must not be read")),
                0
            )?
            .is_empty()
        );
        Ok(())
    }
    #[tokio::test]
    async fn all_budgets_are_required_before_any_model_or_table_is_prepared()
    -> Result<(), Box<dyn StdError>> {
        let (root, store, mut config) = queue::source::tests::fixture()?;
        config.embedder.profile = "ollama".into();
        config.embedder.docs = Some(config::DocsEmbedder {
            profile: "local".into(),
            kinds: vec!["doc_section".into()],
        });
        let store = Arc::new(store);
        let result = Group::prepare(
            Arc::clone(&store),
            Settings {
                root: root.path().into(),
                config_root: root.path().into(),
                config,
                local_cache: root.path().join("missing-cache"),
            },
            &[UnitKind::Chunk, UnitKind::DocSection],
            |provider| {
                if provider == "embed:local" {
                    Ok(limits())
                } else {
                    Err(Error::Invalid("missing explicit provider budget"))
                }
            },
            0,
        )
        .await;
        assert!(matches!(
            result,
            Err(Error::Invalid("missing explicit provider budget"))
        ));
        assert!(store.vector_tables()?.is_empty());
        Ok(())
    }
    #[tokio::test]
    async fn shared_destination_prepares_one_host_and_executes_once()
    -> Result<(), Box<dyn StdError>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]]})),
            )
            .expect(1)
            .mount(&server)
            .await;
        let (root, store, _) = queue::source::tests::fixture()?;
        fs::write(
            root.path().join(".deepkit/config.toml"),
            format!(
                "[index]\nrepo_name='fixture'\n[embedder]\nprofile='ollama'\ndims=2\n[embedder.ollama]\nurl='{}'\n[embedder.docs]\nprofile='ollama'\nkinds=['doc_section']\n",
                server.uri()
            ),
        )?;
        let config = config::load(root.path(), None)?.config;
        let store = Arc::new(store);
        store.set_meta("graph_pending", "0")?;
        let group = Group::prepare(
            Arc::clone(&store),
            Settings {
                root: root.path().into(),
                config_root: root.path().into(),
                config,
                local_cache: root.path().join("unused-cache"),
            },
            &[UnitKind::Chunk, UnitKind::DocSection, UnitKind::Chunk],
            |_| Ok(limits()),
            0,
        )
        .await?;
        assert_eq!(group.entries.len(), 1);
        let entry = &group.entries[0];
        assert!(entry.provider.starts_with("embed:ollama:"));
        assert_eq!(entry.kinds.len(), 2);
        assert_eq!(store.vector_tables()?.len(), 1);
        let mut cursor = entry.host.start_catalogue().await?;
        assert_eq!(
            entry
                .host
                .run_catalogue(
                    &mut cursor,
                    |_| Ok(queue::priority::Recency::ChangedFile),
                    || 0
                )
                .await?
                .execution
                .published,
            1
        );
        assert_eq!(
            entry.host.drain(1, || 1).await?.stopped,
            super::super::DrainStop::Empty
        );
        assert_eq!(store.count_vectors()?, 1);
        Ok(())
    }
}
