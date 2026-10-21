//! Last local reranker verification, persisted for status; v1 §§7.5 and 13.
use crate::{
    config::{self, Config},
    queue,
    store::{self, Store},
};
use serde::{Deserialize, Serialize};
use std::path::Path;
const KEY: &str = "local_reranker_check_v1";
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum State {
    Verified,
    Unavailable,
    Disabled,
}
/// Historical verification evidence, never permission to bypass the native load gate.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Record {
    fingerprint: String,
    pub(crate) state: State,
    pub(crate) model: String,
    pub(crate) file: Option<String>,
    pub(crate) cache: String,
    pub(crate) detail: String,
}
fn fingerprint(config: &Config) -> Result<String, queue::Error> {
    let registry = super::super::registry::Registry::embedded()
        .map_err(|_| queue::Error::Invalid("reranker health registry"))?;
    let pin = registry
        .get(&config.rerank.local.model)
        .ok()
        .map(|model| (&model.sha256, model.size));
    let value = serde_json::json!({"cache_cap":config.limits.models_cache_max_bytes,"pin":pin,"rerank":config.rerank,"profile":config.embedder.profile,
        "version":env!("CARGO_PKG_VERSION"),"os":std::env::consts::OS,"arch":std::env::consts::ARCH,
        "cuda":cfg!(feature="cuda"),"metal":cfg!(feature="metal"),"vulkan":cfg!(feature="vulkan")});
    Ok(blake3::hash(
        &serde_json::to_vec(&value)
            .map_err(|_| queue::Error::Invalid("reranker health identity"))?,
    )
    .to_hex()
    .to_string())
}
impl Record {
    /// Builds bounded source-free evidence from a completed diagnostic or model load.
    pub(crate) fn new(
        config: &Config,
        cache: &Path,
        state: State,
        detail: String,
    ) -> Result<Self, queue::Error> {
        let file = super::super::registry::Registry::embedded()
            .ok()
            .and_then(|registry| {
                registry
                    .get(&config.rerank.local.model)
                    .ok()
                    .map(|model| model.file.clone())
            });
        Ok(Self {
            fingerprint: fingerprint(config)?,
            state,
            model: config.rerank.local.model.clone(),
            file,
            cache: cache.display().to_string(),
            detail: detail.chars().take(1024).collect(),
        })
    }
    /// Renders the last observation without implying an active model or worker.
    pub(crate) fn summary(&self) -> String {
        format!(
            "last local reranker check ({:?}): {}: {}",
            self.state,
            self.file.as_deref().unwrap_or(&self.model),
            self.detail
        )
    }
}
/// Writes only while the checked configuration is current, under database caps.
pub(crate) fn write(
    store: &Store,
    root: &Path,
    expected: &Config,
    record: &Record,
) -> Result<bool, queue::Error> {
    store::admission::transaction(store, store::admission::Kind::Control, |tx| {
        let current = config::load(root, None)
            .map_err(|_| queue::Error::Invalid("reranker health config reload"))?
            .config;
        if current != *expected || record.fingerprint != fingerprint(expected)? {
            return Ok(false);
        }
        let value = serde_json::to_string(record)
            .map_err(|_| queue::Error::Invalid("reranker health record"))?;
        tx.execute("INSERT INTO meta(key,value) VALUES(?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[KEY,&value])?;
        tx.commit()?;
        Ok(true)
    })
}
/// Reads matching last-check evidence; malformed or superseded records are ignored.
pub(crate) fn read(store: &Store, config: &Config) -> Result<Option<Record>, queue::Error> {
    let key = fingerprint(config)?;
    Ok(store
        .get_meta(KEY)?
        .and_then(|value| serde_json::from_str::<Record>(&value).ok())
        .filter(|record| record.fingerprint == key && config.rerank.mode != "off"))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn setup() -> (tempfile::TempDir, Store, Config) {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join(".deepkit")).unwrap();
        std::fs::write(
            root.path().join(".deepkit/config.toml"),
            "[embedder]\nprofile='local'\n",
        )
        .unwrap();
        let config = config::load(root.path(), None).unwrap().config;
        (
            root,
            store::open(":memory:", store::StoreLimits::default()).unwrap(),
            config,
        )
    }
    #[test]
    fn saved_failure_is_replaced_only_by_matching_current_verification() {
        let (root, store, config) = setup();
        let failed =
            Record::new(&config, root.path(), State::Disabled, "smoke failed".into()).unwrap();
        assert!(write(&store, root.path(), &config, &failed).unwrap());
        let saved = read(&store, &config).unwrap().unwrap();
        assert_eq!(saved.state, State::Disabled);
        assert!(saved.summary().contains("Qwen3-Reranker"));
        let good =
            Record::new(&config, root.path(), State::Verified, "smoke passed".into()).unwrap();
        assert!(write(&store, root.path(), &config, &good).unwrap());
        assert_eq!(
            read(&store, &config).unwrap().unwrap().state,
            State::Verified
        );
        let mut resized = config.clone();
        resized.limits.models_cache_max_bytes = "8 GiB".into();
        assert!(read(&store, &resized).unwrap().is_none());
        std::fs::write(
            root.path().join(".deepkit/config.toml"),
            "[rerank]\nmode='off'\n",
        )
        .unwrap();
        assert!(!write(&store, root.path(), &config, &failed).unwrap());
        let current = config::load(root.path(), None).unwrap().config;
        assert!(read(&store, &current).unwrap().is_none());
        assert_eq!(
            read(&store, &config).unwrap().unwrap().state,
            State::Verified
        );
    }
    #[test]
    fn failed_storage_preserves_previous_evidence_and_malformed_records_are_ignored() {
        let (root, store, config) = setup();
        let failed = Record::new(
            &config,
            root.path(),
            State::Unavailable,
            "load failed".into(),
        )
        .unwrap();
        write(&store, root.path(), &config, &failed).unwrap();
        store.writer().unwrap().execute_batch("CREATE TRIGGER refuse_health BEFORE UPDATE ON meta WHEN NEW.key='local_reranker_check_v1' BEGIN SELECT RAISE(ABORT,'fixture'); END;").unwrap();
        let good = Record::new(&config, root.path(), State::Verified, "verified".into()).unwrap();
        assert!(write(&store, root.path(), &config, &good).is_err());
        assert_eq!(
            read(&store, &config).unwrap().unwrap().state,
            State::Unavailable
        );
        store
            .writer()
            .unwrap()
            .execute_batch("DROP TRIGGER refuse_health")
            .unwrap();
        store.set_meta(KEY, "malformed").unwrap();
        assert!(read(&store, &config).unwrap().is_none());
    }
}
