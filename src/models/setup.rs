//! First-index local model preparation independent of L1 work; see `v1.md` §7.4.

use super::registry::VerifiedArtifact;
use crate::config::Config;
use serde::Serialize;
use std::thread::{self, JoinHandle};

/// Last model-preparation outcome; failure degrades indexing to its completed L1 output.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct Report {
    /// Whether setup used a full hash or matched previously verified file identity.
    pub(crate) verification: Option<&'static str>,
    /// Reusable verification metadata; persisted separately from user-facing status.
    #[serde(skip)]
    pub(crate) verified: Option<VerifiedArtifact>,
    /// Pending, ready, unavailable, or skipped.
    pub(crate) state: &'static str,
    /// Configured local artifact, when needed by this request.
    pub(crate) model: Option<String>,
    /// Safe diagnostic explaining failure or skipped preparation.
    pub(crate) detail: Option<String>,
}

/// A background preparation task whose caller can publish L1 before joining it.
pub(crate) struct Preparation {
    initial: Report,
    worker: Option<JoinHandle<Report>>,
}

impl Preparation {
    /// Starts local preparation only when a requested level/tier needs local embeddings.
    pub(crate) fn start(config: &Config, level: u8, previous: Option<VerifiedArtifact>) -> Self {
        let local = config.embedder.profile == "local"
            || config
                .embedder
                .docs
                .as_ref()
                .is_some_and(|docs| docs.profile == "local");
        if level == 1 || !local {
            return Self {
                initial: Report {
                    verification: None,
                    verified: None,
                    state: "skipped",
                    model: None,
                    detail: Some(
                        if level == 1 {
                            "level 1 does not need models"
                        } else {
                            "embedding profiles do not need a local model"
                        }
                        .to_owned(),
                    ),
                },
                worker: None,
            };
        }
        let name = config.embedder.local.model.clone();
        let cap = config.limits.models_cache_max_bytes();
        Self::spawn(name.clone(), move || {
            let cap = cap.map_err(|error| error.to_string())?.0;
            let registry =
                super::registry::Registry::embedded().map_err(|error| error.to_string())?;
            let model = registry.get(&name).map_err(|error| error.to_string())?;
            let cache = super::registry::cache_root().map_err(|error| error.to_string())?;
            prepare_artifact(model, &cache, cap, previous).map_err(|error| error.to_string())
        })
    }

    fn spawn(
        name: String,
        work: impl FnOnce() -> Result<Prepared, String> + Send + 'static,
    ) -> Self {
        let mut initial = Report {
            verification: None,
            verified: None,
            state: "pending",
            model: Some(name.clone()),
            detail: None,
        };
        let worker = thread::Builder::new()
            .name("dkc-model-setup".to_owned())
            .spawn(move || {
                let result = work();
                let (verification, verified) = result
                    .as_ref()
                    .ok()
                    .map(|prepared| (Some(prepared.verification), prepared.verified.clone()))
                    .unwrap_or_default();
                Report {
                    verification,
                    verified,
                    state: if result.is_ok() {
                        "ready"
                    } else {
                        "unavailable"
                    },
                    model: Some(name),
                    detail: result.err(),
                }
            });
        let worker = match worker {
            Ok(worker) => Some(worker),
            Err(error) => {
                initial.state = "unavailable";
                initial.detail = Some(error.to_string());
                None
            }
        };
        Self { initial, worker }
    }

    /// Returns the status to publish before running independent L1 work.
    pub(crate) fn initial(&self) -> &Report {
        &self.initial
    }

    /// Leaves cache-only preparation detached when the caller is shutting down.
    /// The caller must not wait for a model download after an index interruption.
    pub(crate) fn detach(self) -> Report {
        let mut report = self.initial;
        if self.worker.is_some() {
            report.state = "pending";
            report.detail =
                Some("index interrupted before model preparation was joined".to_owned());
        }
        report
    }

    /// Waits only after L1 publication and converts worker failure into visible degradation.
    pub(crate) fn finish(self) -> Report {
        match self.worker {
            Some(worker) => worker.join().unwrap_or(Report {
                verification: None,
                verified: None,
                state: "unavailable",
                model: self.initial.model,
                detail: Some("model preparation worker did not complete".to_owned()),
            }),
            None => self.initial,
        }
    }
}

struct Prepared {
    verified: Option<VerifiedArtifact>,
    verification: &'static str,
}

fn prepare_artifact(
    model: &super::registry::Model,
    cache: &std::path::Path,
    cap: u64,
    previous: Option<VerifiedArtifact>,
) -> Result<Prepared, super::download::Error> {
    if previous
        .as_ref()
        .is_some_and(|proof| proof.matches(model, cache))
    {
        return Ok(Prepared {
            verified: previous,
            verification: "file_identity",
        });
    }
    if !cache.join(&model.file).try_exists()? {
        super::download::pull_blocking(model, cache, cap)?;
    }
    Ok(Prepared {
        verified: super::registry::verify_with_identity(model, cache)?,
        verification: "sha256",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preparation_returns_while_download_is_pending_and_preserves_failure() {
        let (release, wait) = std::sync::mpsc::channel();
        let task = Preparation::spawn("fixture".to_owned(), move || {
            wait.recv()
                .map(|()| Prepared {
                    verified: None,
                    verification: "sha256",
                })
                .map_err(|error| error.to_string())
        });
        assert_eq!(task.initial().state, "pending");
        assert!(!task.worker.as_ref().unwrap().is_finished());
        release.send(()).unwrap();
        assert_eq!(task.finish().state, "ready");
        let failed =
            Preparation::spawn("fixture".to_owned(), || Err("cap reached".to_owned())).finish();
        assert_eq!(failed.state, "unavailable");
        assert_eq!(failed.detail.as_deref(), Some("cap reached"));
    }

    #[test]
    fn lexical_and_external_profiles_skip_local_preparation() {
        let mut config = Config::default();
        for profile in ["local", "remote", "ollama"] {
            config.embedder.profile = profile.to_owned();
            assert_eq!(
                Preparation::start(&config, 1, None).finish().state,
                "skipped"
            );
            if profile != "local" {
                assert_eq!(
                    Preparation::start(&config, 3, None).finish().state,
                    "skipped"
                );
            }
        }
    }
    #[test]
    fn preparation_reuses_only_unchanged_verified_artifacts()
    -> Result<(), Box<dyn std::error::Error>> {
        use sha2::{Digest, Sha256};
        let root = tempfile::tempdir()?;
        let mut model = super::super::registry::Registry::embedded()?
            .models
            .remove(0);
        model.size = 3;
        model.sha256 = Sha256::digest(b"abc")
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let path = root.path().join(&model.file);
        std::fs::write(&path, b"abc")?;
        let first = prepare_artifact(&model, root.path(), 0, None)?;
        assert_eq!(first.verification, "sha256");
        let proof = first.verified.clone();
        let repeated = prepare_artifact(&model, root.path(), 0, proof.clone())?;
        assert_eq!(
            repeated.verification,
            if cfg!(unix) {
                "file_identity"
            } else {
                "sha256"
            }
        );
        if let Some(proof) = &proof {
            let mut changed_pin = model.clone();
            changed_pin.sha256 = "0".repeat(64);
            assert!(!proof.matches(&changed_pin, root.path()));
            assert!(!proof.matches(&model, &root.path().join("different-cache")));
            assert!(prepare_artifact(&changed_pin, root.path(), 0, Some(proof.clone())).is_err());
        }
        let old_time = std::fs::metadata(&path)?.modified()?;
        std::thread::sleep(std::time::Duration::from_millis(1100));
        std::fs::write(&path, b"abd")?;
        std::fs::File::options()
            .write(true)
            .open(&path)?
            .set_times(std::fs::FileTimes::new().set_modified(old_time))?;
        // Same inode, size and restored mtime still invalidate the Unix change timestamp.
        assert!(prepare_artifact(&model, root.path(), 0, proof.clone()).is_err());
        std::fs::remove_file(&path)?;
        if let Some(proof) = &proof {
            assert!(!proof.matches(&model, root.path()));
        }
        std::fs::write(&path, b"abc")?;
        assert_eq!(
            prepare_artifact(&model, root.path(), 0, proof)?.verification,
            "sha256"
        );
        Ok(())
    }
}
