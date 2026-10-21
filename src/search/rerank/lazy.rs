//! Retained, deadline-aware first-use reranker preparation; v1 §§7.4–7.5 and 9.
use super::{Error, Factory, Selection, Settings, selection};
use crate::{config, models, queue, store::Store};
use std::{
    future::Future,
    path::PathBuf,
    pin::Pin,
    sync::{Arc, Mutex},
};
use tokio::{sync::watch, time::Instant};

type Work = Pin<Box<dyn Future<Output = Result<Arc<Factory>, Error>> + Send>>;
/// Only Ready exposes a usable factory; failure and pending preserve caller fallback.
#[derive(Clone)]
pub(crate) enum Outcome {
    Idle,
    Pending,
    Off,
    Ready(Arc<Factory>),
    Failed(Arc<Error>),
}
/// Source-free in-process status, independent of persisted historical verification.
#[derive(Clone, serde::Serialize)]
pub(crate) struct Status {
    pub(crate) state: &'static str,
    pub(crate) detail: Option<String>,
}
impl Outcome {
    /// Describes preparation without exposing a scorer, query or credentials.
    pub(crate) fn status(&self) -> Status {
        let (state, detail) = match self {
            Self::Idle => ("idle", None),
            Self::Pending => ("preparing", None),
            Self::Off => ("off", None),
            Self::Ready(_) => ("ready", None),
            Self::Failed(error) => ("failed", Some(error.to_string())),
        };
        Status { state, detail }
    }
}
/// A host owns one attempt; waiter cancellation never cancels or duplicates that attempt.
/// Recreate the host on configuration change or explicit retry; failures do not retry blindly.
pub(crate) struct Host {
    store: Arc<Store>,
    config: config::Config,
    work: Mutex<Option<Work>>,
    state: watch::Sender<Outcome>,
}
struct Completion {
    sender: watch::Sender<Outcome>,
    finished: bool,
}
impl Completion {
    fn finish(mut self, result: Result<Arc<Factory>, Error>) {
        self.sender.send_replace(match result {
            Ok(factory) => Outcome::Ready(factory),
            Err(error) => Outcome::Failed(Arc::new(error)),
        });
        self.finished = true;
        drop(self);
    }
}
impl Drop for Completion {
    fn drop(&mut self) {
        if !self.finished {
            self.sender
                .send_replace(Outcome::Failed(Arc::new(Error::Task)));
        }
    }
}
impl Host {
    /// Allocates lazy state without model, cache, credential, database or network work.
    pub(crate) fn new(store: Arc<Store>, settings: Settings, cache: PathBuf) -> Arc<Self> {
        if settings.config.rerank.mode == "off" {
            return Self::with_work(None, store, settings.config);
        }
        let work = Box::pin(prepare(Arc::clone(&store), settings.clone(), cache));
        Self::with_work(Some(work), store, settings.config)
    }
    pub(super) fn with_work(
        work: Option<Work>,
        store: Arc<Store>,
        config: config::Config,
    ) -> Arc<Self> {
        let initial = if work.is_some() {
            Outcome::Idle
        } else {
            Outcome::Off
        };
        Arc::new(Self {
            store,
            config,
            work: Mutex::new(work),
            state: watch::channel(initial).0,
        })
    }
    /// Refuses reuse across stores or effective configurations before starting work.
    pub(crate) fn matches(&self, store: &Arc<Store>, config: &config::Config) -> bool {
        Arc::ptr_eq(&self.store, store) && self.config == *config
    }
    /// Restores a saved adaptive bound before weights are ready, within the query deadline.
    /// Read failure must prevent scoring with an unknown bound; it never starts preparation.
    pub(crate) async fn selection_status(
        &self,
        deadline: Instant,
    ) -> Result<super::auto::Status, Error> {
        if let Outcome::Ready(factory) = self.peek() {
            return Ok(factory.auto_status());
        }
        if self.config.rerank.mode != "auto" || self.config.rerank.candidates != 50 {
            return Ok(super::auto::configured_status(&self.config));
        }
        if Instant::now() >= deadline {
            return Err(Error::Deadline);
        }
        let store = Arc::clone(&self.store);
        let config = self.config.clone();
        tokio::time::timeout_at(
            deadline,
            tokio::task::spawn_blocking(move || {
                if Instant::now() >= deadline {
                    return Err(Error::Deadline);
                }
                super::auto::read_status(&store, &config).map_err(Error::Source)
            }),
        )
        .await
        .map_err(|_| Error::Deadline)?
        .map_err(|_| Error::Task)?
    }
    /// Reads current readiness without starting preparation or performing I/O.
    pub(crate) fn peek(&self) -> Outcome {
        self.state.borrow().clone()
    }
    /// Starts at most once and waits only to this caller's original deadline.
    /// The runtime/host owner must outlive requests; this is not a detached OS service.
    pub(crate) async fn get(&self, deadline: Instant) -> Outcome {
        let mut receiver = self.state.subscribe();
        if Instant::now() >= deadline {
            return self.peek();
        }
        let work = match self.work.lock() {
            Ok(mut work) => work.take(),
            Err(_) => {
                self.state
                    .send_replace(Outcome::Failed(Arc::new(Error::Task)));
                None
            }
        };
        if let Some(work) = work {
            self.state.send_replace(Outcome::Pending);
            let completion = Completion {
                sender: self.state.clone(),
                finished: false,
            };
            tokio::spawn(async move {
                // A monitor turns a failed/panicked preparation task into a terminal state.
                let result = tokio::spawn(work).await.unwrap_or(Err(Error::Task));
                completion.finish(result);
            });
        }
        loop {
            let state = receiver.borrow_and_update().clone();
            if !matches!(state, Outcome::Idle | Outcome::Pending) {
                return state;
            }
            if !matches!(
                tokio::time::timeout_at(deadline, receiver.changed()).await,
                Ok(Ok(()))
            ) {
                return self.peek();
            }
        }
    }
}
async fn current(settings: &Settings) -> Result<(), Error> {
    let settings = settings.clone();
    tokio::task::spawn_blocking(move || {
        let current = config::load(&settings.config_root, None)
            .map_err(|_| Error::Input)?
            .config;
        if current == settings.config {
            Ok(())
        } else {
            Err(Error::Privacy)
        }
    })
    .await
    .map_err(|_| Error::Task)?
}
async fn artifact(settings: &Settings, cache: PathBuf) -> Result<(), Error> {
    let config = settings.config.clone();
    tokio::task::spawn_blocking(move || {
        let registry =
            models::registry::Registry::embedded().map_err(models::download::Error::from)?;
        let model = registry
            .get(&config.rerank.local.model)
            .map_err(models::download::Error::from)?;
        let cap = config
            .limits
            .models_cache_max_bytes()
            .map_err(|_| Error::Input)?
            .0;
        if model.pooling != "rank" || model.dims != 2 {
            return Err(Error::Input);
        }
        match std::fs::symlink_metadata(cache.join(&model.file)) {
            // Native loading performs the full verification; avoid hashing twice.
            Ok(metadata) if metadata.is_file() => Ok(()),
            Ok(_) => Err(models::download::Error::Invalid(
                "reranker cache entry is not a regular file".into(),
            )
            .into()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                models::download::pull_blocking(model, &cache, cap)?;
                Ok(())
            }
            Err(error) => Err(models::download::Error::Io(error).into()),
        }
    })
    .await
    .map_err(|_| Error::Task)?
}
async fn prepare(
    store: Arc<Store>,
    settings: Settings,
    cache: PathBuf,
) -> Result<Arc<Factory>, Error> {
    prepare_with(store, settings, cache, Factory::new_configured).await
}
async fn prepare_with<F, Fut>(
    store: Arc<Store>,
    settings: Settings,
    cache: PathBuf,
    load: F,
) -> Result<Arc<Factory>, Error>
where
    F: FnOnce(Arc<Store>, Settings, PathBuf) -> Fut + Send,
    Fut: Future<Output = Result<Arc<Factory>, Error>> + Send,
{
    current(&settings).await?;
    if selection(&settings.config)? == Selection::Local
        && let Err(error) = artifact(&settings, cache.clone()).await
    {
        let health_store = Arc::clone(&store);
        let health_settings = settings.clone();
        let detail = error.to_string();
        tokio::task::spawn_blocking(move || {
            use models::rerank::health::{self, State};
            let saved = (|| -> Result<bool, queue::Error> {
                let record = health::Record::new(
                    &health_settings.config,
                    &cache,
                    State::Unavailable,
                    detail,
                )?;
                health::write(
                    &health_store,
                    &health_settings.config_root,
                    &health_settings.config,
                    &record,
                )
            })();
            if !matches!(saved, Ok(true)) {
                tracing::warn!("reranker preparation failure status could not be saved");
            }
        })
        .await
        .map_err(|_| Error::Task)?;
        return Err(error);
    }
    current(&settings).await?;
    let factory = load(store, settings.clone(), cache).await?;
    current(&settings).await?;
    Ok(factory)
}

#[cfg(test)]
mod tests;
