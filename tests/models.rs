//! Model inspection must work outside a repository without creating cache files.
use assert_cmd::Command;
use serde_json::Value;
use std::{error::Error, fs};

#[test]
fn model_commands_are_read_only_and_refuse_corrupt_cache() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let cache = directory.path().join("models");
    let output = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(directory.path())
        .env("DKC_MODELS_DIR", &cache)
        .args(["models", "list", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let response: Value = serde_json::from_slice(&output)?;
    let rows = response["models"].as_array().ok_or("missing model list")?;
    assert_eq!(rows.len(), 4);
    assert!(
        rows.iter()
            .any(|row| row["model"]["name"] == "qwen3-reranker-0.6b-q8_0")
    );
    assert!(
        rows.iter()
            .any(|row| row["model"]["name"] == "qwen3-reranker-0.6b-f16")
    );
    assert!(rows.iter().all(|row| row["status"] == "missing"));
    assert!(!cache.exists());
    assert!(!directory.path().join(".deepkit").exists());
    fs::create_dir(&cache)?;
    let path = cache.join("Qwen3-Embedding-0.6B-Q8_0.gguf");
    fs::write(&path, b"corrupt")?;
    let output = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(directory.path())
        .env("DKC_MODELS_DIR", &cache)
        .args(["models", "list", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let response: Value = serde_json::from_slice(&output)?;
    assert_eq!(response["models"][0]["status"], "cached_unverified");
    Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(directory.path())
        .env("DKC_MODELS_DIR", &cache)
        .args(["models", "verify", "qwen3-embedding-0.6b-q8_0", "--json"])
        .assert()
        .failure();
    assert_eq!(fs::read(path)?, b"corrupt");
    Ok(())
}

#[test]
fn prune_uses_config_and_keeps_its_preview_read_only() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let cache = directory.path().join("models");
    fs::create_dir(&cache)?;
    fs::create_dir(directory.path().join(".deepkit"))?;
    fs::write(
        directory.path().join(".deepkit/config.toml"),
        "[embedder.local]\nmodel = 'qwen3-embedding-0.6b-f16'\n",
    )?;
    let active = cache.join("Qwen3-Embedding-0.6B-f16.gguf");
    let old = cache.join("Qwen3-Embedding-0.6B-Q8_0.gguf");
    fs::write(&active, b"configured")?;
    fs::write(&old, b"unused")?;
    for dry in [true, false] {
        let mut command = Command::new(assert_cmd::cargo::cargo_bin!("dkc"));
        command
            .current_dir(directory.path())
            .env("DKC_MODELS_DIR", &cache)
            .args(["models", "prune", "--json"]);
        if dry {
            command.arg("--dry-run");
        }
        let output = command.assert().success().get_output().stdout.clone();
        let result: Value = serde_json::from_slice(&output)?;
        assert_eq!(result["dry_run"], dry);
        assert_eq!(
            result["candidates"]
                .as_array()
                .ok_or("missing candidates")?
                .len(),
            1
        );
        assert_eq!(old.exists(), dry);
        assert_eq!(fs::read(&active)?, b"configured");
    }
    Ok(())
}
