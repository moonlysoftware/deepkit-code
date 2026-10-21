//! Selected local artifact verification without downloads; v1 §7.4/§13/§16.
use super::Finding;
use crate::{config::Config, models::registry};
use std::path::Path;

/// Hashes selected artifacts with the registry's bounded-memory verifier.
/// Provider-owned/Ollama models are not mistaken for local cache entries.
pub(super) fn inspect(config: &Config, cache: Option<&Path>) -> Vec<Finding> {
    let mut selected = Vec::new();
    if config.embedder.profile == "local"
        || config
            .embedder
            .docs
            .as_ref()
            .is_some_and(|docs| docs.profile == "local")
    {
        selected.push(("models.embedder", config.embedder.local.model.as_str()));
    }
    if super::selected_local(config) {
        selected.push(("models.reranker", config.rerank.local.model.as_str()));
    }
    if selected.is_empty() {
        return vec![Finding::new(
            "models.local",
            "not_applicable",
            "No selected provider uses local model artifacts",
        )];
    }
    let registry = registry::Registry::embedded();
    selected
        .into_iter()
        .map(|(id, name)| {
            let Some(cache) = cache else {
                return Finding::new(
                    id,
                    "failed",
                    "Model cache unavailable; set DKC_MODELS_DIR or HOME",
                );
            };
            let Ok(registry) = &registry else {
                return Finding::new(id, "failed", "Embedded model registry could not be read");
            };
            let Ok(artifact) = registry.get(name) else {
                return Finding::new(
                    id,
                    "failed",
                    "Selected local model is not in the pinned registry",
                );
            };
            verify(id, artifact, cache)
        })
        .collect()
}

fn verify(id: &str, artifact: &registry::Model, cache: &Path) -> Finding {
    let (status, message) = match registry::verify(artifact, cache) {
        Ok(()) => (
            "passed",
            "Selected local model size and SHA-256 match the pinned registry",
        ),
        Err(registry::Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => (
            "failed",
            "Selected local model is missing; run dkc models pull",
        ),
        Err(registry::Error::Mismatch { .. }) => (
            "failed",
            "Selected local model size or SHA-256 does not match; replace the artifact",
        ),
        Err(_) => (
            "failed",
            "Selected local model could not be safely verified",
        ),
    };
    Finding::new(id, status, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn hashes_fixture_without_loading_or_downloading_and_reports_corruption() {
        let cache = tempfile::tempdir().unwrap();
        let mut artifact = registry::Registry::embedded()
            .unwrap()
            .get("qwen3-embedding-0.6b-q8_0")
            .unwrap()
            .clone();
        let bytes = b"small test-only artifact";
        artifact.size = bytes.len() as u64;
        artifact.sha256 = Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let path = cache.path().join(&artifact.file);
        let missing = verify("test", &artifact, cache.path());
        assert_eq!(missing.status, "failed");
        assert_eq!(
            missing.message,
            "Selected local model is missing; run dkc models pull"
        );
        assert_eq!(std::fs::read_dir(cache.path()).unwrap().count(), 0);
        std::fs::write(&path, bytes).unwrap();
        let valid = verify("test", &artifact, cache.path());
        assert_eq!(valid.status, "passed");
        assert_eq!(
            valid.message,
            "Selected local model size and SHA-256 match the pinned registry"
        );
        std::fs::write(&path, b"bad").unwrap();
        let corrupt = verify("test", &artifact, cache.path());
        assert_eq!(corrupt.status, "failed");
        assert_eq!(
            corrupt.message,
            "Selected local model size or SHA-256 does not match; replace the artifact"
        );
        assert_eq!(std::fs::read(path).unwrap(), b"bad");
    }

    #[test]
    fn unused_artifacts_do_not_access_the_cache() {
        let mut config = Config::default();
        config.rerank.mode = "off".into();
        let report = inspect(&config, None);
        assert_eq!(report.len(), 1);
        assert_eq!(report[0].status, "not_applicable");
    }
}
