//! Proven previous-provider query adapters during serialized migration; v1 §7.3/§9.
use super::*;
use crate::store::{self, embedding_bindings, transitions};

/// Immutable dispatch authority: stored provenance plus the routing generation
/// which still needs this source. A source's `known` hint alone is insufficient.
pub(super) struct Previous {
    generation: i64,
    binding: embedding_bindings::Binding,
    kinds: std::collections::BTreeSet<String>,
}
impl Previous {
    /// Rechecks source generation, affected kinds and exact active provenance
    /// inside the caller's dispatch or scan snapshot.
    pub(super) fn allows(
        &self,
        tx: &Transaction<'_>,
        table: &str,
        kind: UnitKind,
    ) -> Result<bool, store::Error> {
        let Some(route) = transitions::read(tx)? else {
            return Ok(false);
        };
        let dims = i64::try_from(self.binding.dims)
            .map_err(|_| store::Error::InvalidVectorDimensions { slug: table.into() })?;
        Ok(route.generation == self.generation
            && route
                .previous
                .iter()
                .any(|source| source.table == table && source.kinds == self.kinds && source.kinds.contains(kind.as_str()))
            && embedding_bindings::read(tx, table)?.as_ref() == Some(&self.binding)
            && tx.query_row("SELECT EXISTS(SELECT 1 FROM embedding_tables WHERE slug=?1 AND active=1 AND profile=?2 AND model=?3 AND dims=?4)",params![table,self.binding.profile,self.binding.model,dims], |r|r.get::<_,bool>(0))?)
    }

    /// Restores only the provider recipe. Budgets, rates, resource limits and the
    /// authoritative on-disk configuration remain those supplied by today's host.
    pub(super) fn configuration(&self, current: &Config) -> Result<Config, queue::Error> {
        let binding = &self.binding;
        let mut effective = current.clone();
        let embedder = &mut effective.embedder;
        embedder.docs = None;
        embedder.profile = binding.profile.clone();
        embedder.dims = binding.dims;
        match binding.profile.as_str() {
            "remote" => {
                embedder.remote.url = binding
                    .endpoint
                    .strip_suffix("/embeddings")
                    .ok_or(queue::Error::Invalid("previous remote endpoint"))?
                    .into();
                embedder.remote.model = binding.model.clone();
                embedder.remote.query_prefix = binding.query_prefix.clone();
                embedder.remote.api_key_env = binding
                    .api_key_env
                    .clone()
                    .ok_or(queue::Error::Invalid("previous credential reference"))?;
            }
            "ollama" => {
                embedder.ollama.url = binding
                    .endpoint
                    .strip_suffix("/api/embed")
                    .ok_or(queue::Error::Invalid("previous Ollama endpoint"))?
                    .into();
                embedder.ollama.model = binding.model.clone();
                embedder.ollama.query_prefix = binding.query_prefix.clone();
            }
            "local" => {
                embedder.local.model = binding.model.clone();
                embedder.local.query_prefix = binding.query_prefix.clone();
            }
            _ => return Err(queue::Error::Invalid("previous embedding profile")),
        }
        // Resolving through the ordinary registry also proves pinned local bytes.
        // A removed/changed registry entry must not load a different old model.
        let resolved = destination::resolve(&effective, UnitKind::Chunk)?;
        if resolved.binding(&effective) != *binding {
            return Err(queue::Error::Invalid(
                "previous provider identity cannot be reconstructed",
            ));
        }
        Ok(effective)
    }
}
impl Factory {
    /// Intersects a previous source's affected kinds with the requested scope.
    /// None means no eligible kind; an empty current-tier list still means all.
    pub(crate) fn scan_kinds(&self, requested: &[UnitKind]) -> Option<Vec<UnitKind>> {
        let Some(previous) = &self.previous else {
            return Some(requested.to_vec());
        };
        let kinds: Vec<_> = UnitKind::ALL
            .into_iter()
            .filter(|kind| {
                previous.kinds.contains(kind.as_str())
                    && (requested.is_empty() || requested.contains(kind))
            })
            .collect();
        (!kinds.is_empty()).then_some(kinds)
    }

    /// Builds one explicitly selected old source using persisted provenance and
    /// current limits. Does not download models or infer unknown legacy bindings.
    /// Callers retain the absolute query deadline around asynchronous preparation.
    pub(crate) async fn new_previous(
        store: Arc<Store>,
        settings: Settings,
        table: String,
        generation: i64,
        cache: PathBuf,
    ) -> Result<Arc<Self>, queue::Error> {
        Self::build_previous(
            store,
            settings,
            table,
            generation,
            cache,
            OpenAiEmbedder::from_config,
        )
        .await
    }

    /// Reconstructs a mock previous HTTP provider without reading host credentials.
    #[cfg(test)]
    pub(crate) async fn new_previous_mock_remote(
        store: Arc<Store>,
        settings: Settings,
        table: String,
        generation: i64,
        cache: PathBuf,
    ) -> Result<Arc<Self>, queue::Error> {
        Self::build_previous(store, settings, table, generation, cache, |config| {
            OpenAiEmbedder::with_api_key(config, String::new())
        })
        .await
    }

    async fn build_previous(
        store: Arc<Store>,
        settings: Settings,
        table: String,
        generation: i64,
        cache: PathBuf,
        remote: impl FnOnce(OpenAiConfig) -> crate::models::Result<OpenAiEmbedder> + Send + 'static,
    ) -> Result<Arc<Self>, queue::Error> {
        let (store, settings, table, previous, effective, resolved, http) =
            tokio::task::spawn_blocking(move || {
                if config::load(&settings.config_root, None)
                    .map_err(|_| queue::Error::Invalid("query configuration reload"))?
                    .config
                    != settings.config
                {
                    return Err(queue::Error::Invalid("query configuration changed"));
                }
                let previous = store.read_snapshot(|tx| -> Result<Previous, queue::Error> {
                    let binding = embedding_bindings::read(tx, &table)?.ok_or(
                        queue::Error::Invalid("previous provider provenance is unknown"),
                    )?;
                    let route = transitions::read(tx)?.ok_or(queue::Error::Invalid(
                        "previous query transition is missing",
                    ))?;
                    let kinds = route
                        .previous
                        .iter()
                        .find(|source| source.table == table)
                        .ok_or(queue::Error::Invalid("previous query source is missing"))?
                        .kinds
                        .clone();
                    let previous = Previous {
                        generation,
                        binding,
                        kinds,
                    };
                    if !previous.allows(tx, &table, settings.kind)? {
                        return Err(queue::Error::Invalid(
                            "previous query source is no longer authorized",
                        ));
                    }
                    Ok(previous)
                })??;
                let effective = previous.configuration(&settings.config)?;
                let resolved = Self::validate(
                    &store,
                    &Settings {
                        config_root: settings.config_root.clone(),
                        config: effective.clone(),
                        kind: settings.kind,
                        budget: settings.budget,
                    },
                )?;
                let http = if previous.binding.profile == "local" {
                    None
                } else {
                    Some(Self::http_provider(&effective, &resolved, remote)?)
                };
                Ok((store, settings, table, previous, effective, resolved, http))
            })
            .await
            .map_err(|_| queue::Error::WorkerTask)??;
        let provider = match http {
            Some(provider) => provider,
            None => {
                let idle = effective
                    .models
                    .unload_after_idle()
                    .map_err(|_| queue::Error::Invalid("local query idle timeout"))?;
                Provider::Local(
                    LocalProvider::load_registered(
                        cache,
                        effective.embedder.local,
                        effective.embedder.dims,
                        idle,
                    )
                    .await
                    .map_err(|_| {
                        queue::Error::Invalid("local query artifact or model initialization")
                    })?,
                )
            }
        };
        let dimensions_capability = if let Provider::Remote(remote) = &provider {
            Some(remote_dimensions::restore(
                &store,
                &resolved.identity,
                remote,
            )?)
        } else {
            None
        };
        Ok(Arc::new(Self {
            store,
            settings,
            table,
            identity: resolved.identity,
            provider,
            dimensions_capability,
            previous: Some(previous),
        }))
    }
}

#[cfg(test)]
mod tests;
