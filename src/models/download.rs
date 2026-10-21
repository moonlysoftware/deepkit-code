//! Resumable verified GGUF downloads and cache admission; see `v1.md` §7.4 and §3.5.

use super::registry::{self, Model};
use std::{
    fs::{self, OpenOptions},
    io::{IsTerminal, Read, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

/// Download failures preserve existing artifacts and resumable partial bytes.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    /// Filesystem failure during admission or writing.
    #[error("model download I/O failed: {0}")]
    Io(#[from] std::io::Error),
    /// Verification refused an artifact.
    #[error(transparent)]
    Verify(#[from] registry::Error),
    /// Another process is changing the cache.
    #[error("model cache is busy: {0}")]
    Busy(String),
    /// The declared complete artifact would exceed the cache cap.
    #[error("model cache needs {required} bytes; configured cap is {cap}")]
    Cap { required: u64, cap: u64 },
    /// Invalid server or cache state; no final artifact was published.
    #[error("model download refused: {0}")]
    Invalid(String),
    /// HTTP transfer failed, retaining partial bytes for a future pull.
    #[error("model download transport failed: {0}")]
    Http(#[source] reqwest::Error),
    /// Blocking worker did not finish.
    #[error("model download worker did not complete")]
    Worker,
}

impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Self {
        // Redirected artifact URLs can carry temporary signed credentials.
        Self::Http(error.without_url())
    }
}

/// Pulls on a blocking worker so hashing, filesystem access and HTTP do not stall MCP.
pub(crate) async fn pull(model: Model, cache: PathBuf, cap: u64) -> Result<PathBuf, Error> {
    tokio::task::spawn_blocking(move || pull_blocking(&model, &cache, cap))
        .await
        .map_err(|_| Error::Worker)?
}

fn regular_size(path: &Path) -> Result<Option<u64>, Error> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() => Ok(Some(metadata.len())),
        Ok(_) => Err(Error::Invalid(format!(
            "cache entry is not a regular file: {}",
            path.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn cache_bytes(cache: &Path) -> Result<u64, Error> {
    let mut total = 0u64;
    for entry in fs::read_dir(cache)? {
        let path = entry?.path();
        let bytes = regular_size(&path)?.unwrap_or(0);
        total = total
            .checked_add(bytes)
            .ok_or_else(|| Error::Invalid("cache size overflow".to_owned()))?;
    }
    Ok(total)
}

struct CacheLock(std::fs::File);

impl Drop for CacheLock {
    fn drop(&mut self) {
        // Explicitly release flock even if a concurrently forked child temporarily
        // inherited the open file description before closing it at exec.
        let _ = self.0.unlock();
    }
}

fn lock_cache(cache: &Path) -> Result<CacheLock, Error> {
    let lock_path = cache.join(".download.lock");
    regular_size(&lock_path)?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)?;
    lock.try_lock()
        .map_err(|error| Error::Busy(error.to_string()))?;
    Ok(CacheLock(lock))
}

struct Progress<'a> {
    name: &'a str,
    total: u64,
    enabled: bool,
    last: Instant,
}

impl Progress<'_> {
    fn update(&mut self, written: u64) {
        if self.enabled
            && (self.last.elapsed() >= Duration::from_millis(100) || written == self.total)
        {
            let filled = ((written as f64 / self.total as f64) * 20.0) as usize;
            eprint!(
                "\r{} [{}{}] {written}/{} bytes",
                self.name,
                "=".repeat(filled.min(20)),
                " ".repeat(20 - filled.min(20)),
                self.total
            );
            self.last = Instant::now();
        }
    }
}

impl Drop for Progress<'_> {
    fn drop(&mut self) {
        if self.enabled {
            eprintln!();
        }
    }
}

/// Runs a verified pull on an already dedicated worker thread.
pub(crate) fn pull_blocking(model: &Model, cache: &Path, cap: u64) -> Result<PathBuf, Error> {
    fs::create_dir_all(cache)?;
    let lock = lock_cache(cache)?;
    let destination = cache.join(&model.file);
    if regular_size(&destination)?.is_some() {
        registry::verify(model, cache)?;
        return Ok(destination);
    }
    let mut partial_model = model.clone();
    partial_model.file = format!("{}.part", model.file);
    let partial = cache.join(&partial_model.file);
    let offset = regular_size(&partial)?.unwrap_or(0);
    if offset > model.size {
        return Err(Error::Invalid(
            "partial file exceeds pinned artifact size".to_owned(),
        ));
    }
    let required = cache_bytes(cache)?
        .checked_add(model.size - offset)
        .ok_or_else(|| Error::Invalid("cache size overflow".to_owned()))?;
    if required > cap {
        return Err(Error::Cap { required, cap });
    }
    if offset < model.size {
        let client = reqwest::blocking::Client::builder()
            .user_agent(concat!("dkc/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(1800))
            .build()?;
        let mut request = client
            .get(&model.url)
            .header(reqwest::header::ACCEPT_ENCODING, "identity");
        if offset > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
        }
        let mut response = request.send()?.error_for_status()?;
        let start = match response.status().as_u16() {
            200 => 0, // A server ignoring Range requires a bounded restart of the partial artifact.
            206 => {
                let expected = format!("bytes {offset}-{}/{}", model.size - 1, model.size);
                if response
                    .headers()
                    .get(reqwest::header::CONTENT_RANGE)
                    .and_then(|value| value.to_str().ok())
                    != Some(expected.as_str())
                {
                    return Err(Error::Invalid("unexpected Content-Range".to_owned()));
                }
                offset
            }
            status => return Err(Error::Invalid(format!("unexpected HTTP status {status}"))),
        };
        if response
            .content_length()
            .is_some_and(|length| length != model.size - start)
        {
            return Err(Error::Invalid(
                "response size differs from pinned artifact remainder".to_owned(),
            ));
        }
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(start == 0)
            .append(start > 0)
            .open(&partial)?;
        let mut written = start;
        let mut progress = Progress {
            name: &model.name,
            total: model.size,
            enabled: std::io::stderr().is_terminal(),
            last: Instant::now(),
        };
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let count = response.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            if count as u64 > model.size - written {
                return Err(Error::Invalid(
                    "response exceeds pinned artifact size".to_owned(),
                ));
            }
            file.write_all(&buffer[..count])?;
            written += count as u64;
            progress.update(written);
        }
        file.sync_all()?;
    }
    registry::verify(&partial_model, cache)?;
    // Hard-link publication refuses to overwrite a destination created by another writer.
    fs::hard_link(&partial, &destination)?;
    fs::remove_file(&partial)?;
    drop(lock);
    Ok(destination)
}

/// Readable result of cache pruning, with unknown files retained explicitly.
#[derive(Debug, serde::Serialize)]
pub(crate) struct PruneReport {
    /// Whether candidate files were retained for review.
    pub(crate) dry_run: bool,
    /// Known unreferenced artifacts and partials selected for removal.
    pub(crate) candidates: Vec<PathBuf>,
    /// Total bytes in the selected files before removal.
    pub(crate) reclaimable_bytes: u64,
    /// Files retained because their model is named in configuration.
    pub(crate) retained: Vec<PathBuf>,
    /// Unrecognized entries, never deleted by the model cache manager.
    pub(crate) unmanaged: Vec<PathBuf>,
}

/// Removes registered artifacts unreferenced by config, sharing download admission's lock.
/// Unknown files and configured partial downloads remain; dry-run only reports candidates.
pub(crate) fn prune(
    registry: &registry::Registry,
    cache: &Path,
    referenced: &std::collections::BTreeSet<String>,
    dry_run: bool,
) -> Result<PruneReport, Error> {
    let mut report = PruneReport {
        dry_run,
        candidates: Vec::new(),
        reclaimable_bytes: 0,
        retained: Vec::new(),
        unmanaged: Vec::new(),
    };
    if !cache.try_exists()? {
        return Ok(report);
    }
    let _lock = lock_cache(cache)?;
    let mut entries = fs::read_dir(cache)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort();
    for path in entries {
        if path
            .file_name()
            .is_some_and(|name| name == ".download.lock")
        {
            continue;
        }
        let owner = registry.models.iter().find(|model| {
            path.file_name().is_some_and(|name| {
                name == model.file.as_str() || name == format!("{}.part", model.file).as_str()
            })
        });
        let Some(owner) = owner else {
            report.unmanaged.push(path);
            continue;
        };
        if referenced.contains(&owner.name) {
            report.retained.push(path);
            continue;
        }
        let size = regular_size(&path)?.unwrap_or(0);
        report.reclaimable_bytes = report
            .reclaimable_bytes
            .checked_add(size)
            .ok_or_else(|| Error::Invalid("cache size overflow".to_owned()))?;
        report.candidates.push(path);
    }
    // Complete validation precedes deletion so malformed entries cannot cause a partial prune.
    if !dry_run {
        for path in &report.candidates {
            fs::remove_file(path)?;
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{header, method, path},
    };

    fn artifact(server: &MockServer) -> Model {
        let mut model = registry::Registry::embedded().unwrap().models.remove(0);
        model.url = format!("{}/model", server.uri());
        model.size = 6;
        model.sha256 = Sha256::digest(b"abcdef")
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        model
    }

    #[tokio::test]
    async fn resumes_verifies_and_reuses_without_network() -> Result<(), Box<dyn std::error::Error>>
    {
        let server = MockServer::start().await;
        let model = artifact(&server);
        let cache = tempfile::tempdir()?;
        fs::write(cache.path().join(format!("{}.part", model.file)), b"abc")?;
        Mock::given(method("GET"))
            .and(path("/model"))
            .and(header("range", "bytes=3-"))
            .and(header(
                "user-agent",
                concat!("dkc/", env!("CARGO_PKG_VERSION")),
            ))
            .respond_with(
                ResponseTemplate::new(206)
                    .insert_header("Content-Range", "bytes 3-5/6")
                    .set_body_bytes(b"def"),
            )
            .expect(1)
            .mount(&server)
            .await;
        let result = pull(model.clone(), cache.path().to_owned(), 6).await?;
        assert_eq!(fs::read(&result)?, b"abcdef");
        assert!(!cache.path().join(format!("{}.part", model.file)).exists());
        assert_eq!(pull(model, cache.path().to_owned(), 0).await?, result);
        Ok(())
    }

    #[tokio::test]
    async fn refuses_hash_mismatch_and_cap_without_evicting()
    -> Result<(), Box<dyn std::error::Error>> {
        let server = MockServer::start().await;
        let model = artifact(&server);
        let cache = tempfile::tempdir()?;
        fs::write(cache.path().join("retained.gguf"), b"x")?;
        assert!(matches!(
            pull(model.clone(), cache.path().to_owned(), 6).await,
            Err(Error::Cap { .. })
        ));
        assert!(server.received_requests().await.unwrap().is_empty());
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"abcdeg"))
            .expect(1)
            .mount(&server)
            .await;
        let mismatch = pull(model.clone(), cache.path().to_owned(), 7).await;
        assert!(
            matches!(
                mismatch,
                Err(Error::Verify(registry::Error::Mismatch { .. }))
            ),
            "unexpected result: {mismatch:?}"
        );
        assert!(!cache.path().join(&model.file).exists());
        assert_eq!(fs::read(cache.path().join("retained.gguf"))?, b"x");
        assert_eq!(
            fs::read(cache.path().join(format!("{}.part", model.file)))?,
            b"abcdeg"
        );
        Ok(())
    }

    #[tokio::test]
    async fn range_restart_and_malformed_range_preserve_bounds()
    -> Result<(), Box<dyn std::error::Error>> {
        for valid in [false, true] {
            let server = MockServer::start().await;
            let model = artifact(&server);
            let cache = tempfile::tempdir()?;
            let partial = cache.path().join(format!("{}.part", model.file));
            fs::write(&partial, b"abc")?;
            let response = if valid {
                ResponseTemplate::new(200).set_body_bytes(b"abcdef")
            } else {
                ResponseTemplate::new(206)
                    .insert_header("Content-Range", "bytes 0-2/6")
                    .set_body_bytes(b"def")
            };
            Mock::given(method("GET"))
                .respond_with(response)
                .mount(&server)
                .await;
            let result = pull(model.clone(), cache.path().to_owned(), 6).await;
            if valid {
                assert_eq!(fs::read(result?)?, b"abcdef");
            } else {
                assert!(matches!(result, Err(Error::Invalid(_))));
                assert_eq!(fs::read(partial)?, b"abc");
            }
        }
        Ok(())
    }
    #[tokio::test]
    async fn cache_lock_refuses_competing_writer_before_http()
    -> Result<(), Box<dyn std::error::Error>> {
        let server = MockServer::start().await;
        let model = artifact(&server);
        let cache = tempfile::tempdir()?;
        let lock = std::fs::File::create(cache.path().join(".download.lock"))?;
        lock.lock()?;
        assert!(matches!(
            pull(model, cache.path().to_owned(), 6).await,
            Err(Error::Busy(_))
        ));
        assert!(server.received_requests().await.unwrap().is_empty());
        Ok(())
    }
    #[test]
    fn prune_preserves_references_partials_unknowns_and_dry_run()
    -> Result<(), Box<dyn std::error::Error>> {
        let registry = registry::Registry::embedded()?;
        let cache = tempfile::tempdir()?;
        let active = &registry.models[0];
        let unused = &registry.models[1];
        for model in [&active, &unused] {
            fs::write(cache.path().join(&model.file), b"data")?;
            fs::write(cache.path().join(format!("{}.part", model.file)), b"part")?;
        }
        fs::write(cache.path().join("unmanaged.gguf"), b"keep")?;
        let references = [active.name.clone()].into_iter().collect();
        let dry = prune(&registry, cache.path(), &references, true)?;
        assert_eq!(dry.candidates.len(), 2);
        assert_eq!(dry.reclaimable_bytes, 8);
        assert_eq!(dry.retained.len(), 2);
        assert_eq!(dry.unmanaged.len(), 1);
        assert!(dry.candidates.iter().all(|path| path.exists()));
        let result = prune(&registry, cache.path(), &references, false)?;
        assert_eq!(result.candidates, dry.candidates);
        assert!(result.candidates.iter().all(|path| !path.exists()));
        assert!(
            result
                .retained
                .iter()
                .chain(&result.unmanaged)
                .all(|path| path.exists())
        );
        let lock = lock_cache(cache.path())?;
        assert!(matches!(
            prune(&registry, cache.path(), &references, false),
            Err(Error::Busy(_))
        ));
        drop(lock);
        assert!(
            prune(&registry, &cache.path().join("missing"), &references, false)?
                .candidates
                .is_empty()
        );
        assert!(!cache.path().join("missing").exists());
        Ok(())
    }
    #[cfg(unix)]
    #[test]
    fn prune_rejects_registered_symlink_before_deleting_candidates()
    -> Result<(), Box<dyn std::error::Error>> {
        let registry = registry::Registry::embedded()?;
        let cache = tempfile::tempdir()?;
        let first = cache.path().join(&registry.models[0].file);
        fs::write(&first, b"preserve")?;
        let external = tempfile::NamedTempFile::new()?;
        std::os::unix::fs::symlink(external.path(), cache.path().join(&registry.models[1].file))?;
        assert!(matches!(
            prune(&registry, cache.path(), &Default::default(), false),
            Err(Error::Invalid(_))
        ));
        assert_eq!(fs::read(first)?, b"preserve");
        assert!(external.path().exists());
        Ok(())
    }
}
