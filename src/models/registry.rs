//! Pinned local model artifacts and cache verification; see `v1.md` §7.4.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

/// Registry/cache failures retain a typed cause without storing model content.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    /// The embedded registry is malformed or unsafe to resolve into a cache path.
    #[error("invalid model registry: {0}")]
    Registry(String),
    /// Model name has no verified artifact entry.
    #[error("model is not in the pinned registry: {0}")]
    Unknown(String),
    /// No user cache root could be resolved.
    #[error("set DKC_MODELS_DIR or HOME to resolve the model cache")]
    CacheRoot,
    /// Reading the cache failed.
    #[error("model cache I/O failed: {0}")]
    Io(#[from] std::io::Error),
    /// Cached bytes are not exactly the pinned artifact.
    #[error(
        "model verification failed for {name}: expected {expected_size} bytes and SHA-256 {expected_hash}, got {size} bytes and SHA-256 {hash}"
    )]
    Mismatch {
        name: String,
        expected_size: u64,
        expected_hash: String,
        size: u64,
        hash: String,
    },
}

/// One immutable model artifact and the local inference settings associated with it.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Model {
    /// Configuration model identifier.
    pub(crate) name: String,
    /// HTTPS download URL pinned to a repository commit.
    pub(crate) url: String,
    /// Single cache filename, never a relative directory.
    pub(crate) file: String,
    /// Expected SHA-256 digest of the complete GGUF.
    pub(crate) sha256: String,
    /// Complete artifact size in bytes.
    pub(crate) size: u64,
    /// Recommended local context size.
    pub(crate) context: u32,
    /// llama.cpp pooling mode.
    pub(crate) pooling: String,
    /// Output width: embedding dimensions or the number of classifier scores.
    pub(crate) dims: usize,
    /// Query-only instruction template.
    pub(crate) instruct: String,
}

/// Validated registry embedded in the executable, independent of the working directory.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Registry {
    /// Available pinned artifacts, in display order.
    pub(crate) models: Vec<Model>,
}

impl Registry {
    /// Reads the built-in artifact list and checks path/hash/identity invariants.
    pub(crate) fn embedded() -> Result<Self, Error> {
        Self::parse(include_str!("models.toml"))
    }

    fn parse(source: &str) -> Result<Self, Error> {
        let registry: Self =
            toml::from_str(source).map_err(|error| Error::Registry(error.to_string()))?;
        let mut names = BTreeSet::new();
        let mut files = BTreeSet::new();
        for model in &registry.models {
            let url = reqwest::Url::parse(&model.url)
                .map_err(|error| Error::Registry(error.to_string()))?;
            let segments = url
                .path_segments()
                .map(|parts| parts.collect::<Vec<_>>())
                .unwrap_or_default();
            let pinned = segments.len() == 5
                && segments[2] == "resolve"
                && segments[3].len() == 40
                && segments[3].bytes().all(|byte| byte.is_ascii_hexdigit())
                && segments[4] == model.file;
            if model.name.is_empty()
                || !names.insert(&model.name)
                || !files.insert(&model.file)
                || model.file.is_empty()
                || model.file.contains(['/', '\\', ':'])
                || model.file == "."
                || model.file == ".."
                || model.sha256.len() != 64
                || !model
                    .sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                || model.size == 0
                || model.context == 0
                || model.dims == 0
                || !matches!(model.pooling.as_str(), "last" | "rank")
                || url.scheme() != "https"
                || url.host_str() != Some("huggingface.co")
                || !url.username().is_empty()
                || url.password().is_some()
                || url.query().is_some()
                || url.fragment().is_some()
                || !pinned
            {
                return Err(Error::Registry(format!(
                    "invalid or duplicate artifact {}",
                    model.name
                )));
            }
        }
        if registry.models.is_empty() {
            return Err(Error::Registry("empty model list".to_owned()));
        }
        Ok(registry)
    }

    /// Resolves a configured identifier without guessing an unverified download URL.
    pub(crate) fn get(&self, name: &str) -> Result<&Model, Error> {
        self.models
            .iter()
            .find(|model| model.name == name)
            .ok_or_else(|| Error::Unknown(name.to_owned()))
    }
}

/// Resolves the global model cache without creating or modifying any directory.
pub(crate) fn cache_root() -> Result<PathBuf, Error> {
    if let Some(value) = std::env::var_os("DKC_MODELS_DIR").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(value));
    }
    std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .map(|home| PathBuf::from(home).join(".cache/deepkit-code/models"))
        .ok_or(Error::CacheRoot)
}

/// File identity captured around successful hashing; used only by automatic preparation.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct VerifiedArtifact {
    version: u8,
    name: String,
    sha256: String,
    path: PathBuf,
    identity: FileIdentity,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct FileIdentity {
    device: u64,
    inode: u64,
    size: u64,
    modified: (i64, i64),
    changed: (i64, i64),
}

fn file_identity(metadata: &std::fs::Metadata) -> Option<FileIdentity> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Some(FileIdentity {
            device: metadata.dev(),
            inode: metadata.ino(),
            size: metadata.len(),
            modified: (metadata.mtime(), metadata.mtime_nsec()),
            changed: (metadata.ctime(), metadata.ctime_nsec()),
        })
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        None // Without reliable change identity, automatic preparation hashes every time.
    }
}

impl VerifiedArtifact {
    /// Checks current file identity and registry pin without reading model bytes.
    pub(crate) fn matches(&self, model: &Model, cache: &Path) -> bool {
        let path = cache.join(&model.file);
        let Ok(metadata) = std::fs::symlink_metadata(&path) else {
            return false;
        };
        self.version == 1
            && self.name == model.name
            && self.sha256 == model.sha256
            && self.identity.size == model.size
            && metadata.is_file()
            && path.canonicalize().is_ok_and(|path| path == self.path)
            && file_identity(&metadata).as_ref() == Some(&self.identity)
    }
}

/// Hashes the entire cache file with bounded memory and refuses size or digest mismatch.
pub(crate) fn verify(model: &Model, cache: &Path) -> Result<(), Error> {
    verify_with_identity(model, cache).map(|_| ())
}

/// Fully verifies bytes, refusing a changed file before issuing reusable identity metadata.
pub(crate) fn verify_with_identity(
    model: &Model,
    cache: &Path,
) -> Result<Option<VerifiedArtifact>, Error> {
    let path = cache.join(&model.file);
    let entry = std::fs::symlink_metadata(&path)?;
    if !entry.is_file() {
        return Err(std::io::Error::other("model cache entry is not a regular file").into());
    }
    let canonical = path.canonicalize()?;
    let mut file = File::open(&path)?;
    let before_metadata = file.metadata()?;
    let before = file_identity(&before_metadata);
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut size = 0u64;
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        size += count as u64;
        digest.update(&buffer[..count]);
    }
    let hash = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    if size != model.size || hash != model.sha256 {
        return Err(Error::Mismatch {
            name: model.name.clone(),
            expected_size: model.size,
            expected_hash: model.sha256.clone(),
            size,
            hash,
        });
    }
    let after_metadata = file.metadata()?;
    let after = file_identity(&after_metadata);
    let current = std::fs::symlink_metadata(&path)?;
    if !current.is_file()
        || before_metadata.len() != after_metadata.len()
        || before_metadata.modified().ok() != after_metadata.modified().ok()
        || after_metadata.len() != current.len()
        || after_metadata.modified().ok() != current.modified().ok()
        || before != after
        || after != file_identity(&current)
        || path.canonicalize()? != canonical
    {
        return Err(std::io::Error::other("model changed during verification").into());
    }
    Ok(after.map(|identity| VerifiedArtifact {
        version: 1,
        name: model.name.clone(),
        sha256: model.sha256.clone(),
        path: canonical,
        identity,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pins_are_unique_safe_and_complete() -> Result<(), Box<dyn std::error::Error>> {
        let registry = Registry::embedded()?;
        assert_eq!(registry.models.len(), 4);
        for name in ["qwen3-reranker-0.6b-q8_0", "qwen3-reranker-0.6b-f16"] {
            let model = registry.get(name)?;
            assert_eq!(model.pooling, "rank");
            assert_eq!(model.dims, 2);
            assert!(model.instruct.contains("<Query>: {query}"));
            assert!(model.instruct.contains("<Document>: {document}"));
            assert!(
                model
                    .url
                    .contains("bartscheffer/deepkit-code-reranker-gguf/resolve/")
            );
        }
        assert_eq!(registry.get("qwen3-embedding-0.6b-q8_0")?.size, 639_150_592);
        assert!(matches!(registry.get("absent"), Err(Error::Unknown(_))));
        for (from, to) in [
            ("Qwen3-Embedding-0.6B-Q8_0.gguf", "../escape.gguf"),
            ("370f27d7550e0def9b39c1f16d3fbaa13aa67728", "main"),
            ("https://", "http://"),
            ("pooling = \"last\"", "pooling = \"mean\""),
        ] {
            assert!(Registry::parse(&include_str!("models.toml").replace(from, to)).is_err());
        }
        Ok(())
    }

    #[test]
    fn verification_checks_digest_and_size_without_mutation()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let mut model = Registry::embedded()?.models.remove(0);
        model.size = 3;
        model.sha256 = Sha256::digest(b"abc")
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let path = root.path().join(&model.file);
        assert!(matches!(verify(&model, root.path()), Err(Error::Io(_))));
        std::fs::write(&path, b"abc")?;
        verify(&model, root.path())?;
        for bad in [b"abd".as_slice(), b"ab", b"abcd"] {
            std::fs::write(&path, bad)?;
            assert!(matches!(
                verify(&model, root.path()),
                Err(Error::Mismatch { .. })
            ));
            assert_eq!(std::fs::read(&path)?, bad);
        }
        Ok(())
    }
}
