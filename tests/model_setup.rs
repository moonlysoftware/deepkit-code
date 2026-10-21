//! First local index must retain searchable L1 when model preparation cannot fit.
use assert_cmd::Command;
use serde_json::Value;
use std::{error::Error, fs};

#[test]
fn capped_first_index_publishes_l1_and_reports_model_failure() -> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let cache = tempfile::tempdir()?;
    fs::create_dir(root.path().join(".deepkit"))?;
    fs::write(
        root.path().join(".deepkit/config.toml"),
        "[embedder]\nprofile = 'local'\n[limits]\nmodels_cache_max_bytes = '0 B'\n",
    )?;
    fs::write(
        root.path().join("retry.rs"),
        "pub fn deliberate_retry_policy() -> bool { true }\n",
    )?;
    let run = |args: &[&str]| -> Result<Value, Box<dyn Error>> {
        let output = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
            .current_dir(root.path())
            .env("DKC_MODELS_DIR", cache.path())
            .args(args)
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        Ok(serde_json::from_slice(&output)?)
    };
    let indexed = run(&["index", "--json"])?;
    assert!(indexed["units_indexed"].as_u64().ok_or("missing units")? > 0);
    assert_eq!(indexed["model_preparation"]["state"], "unavailable");
    assert!(
        indexed["model_preparation"]["detail"]
            .as_str()
            .ok_or("missing diagnostic")?
            .contains("configured cap is 0")
    );
    let status = run(&["status", "--json"])?;
    assert_eq!(status["l1_pending_files"], 0);
    assert_eq!(status["model_preparation"], indexed["model_preparation"]);
    let plain = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root.path())
        .env("DKC_MODELS_DIR", cache.path())
        .arg("status")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    assert!(String::from_utf8(plain)?.contains("model preparation: unavailable"));
    let found = run(&["search", "deliberate_retry_policy", "--json"])?;
    assert!(found.to_string().contains("retry.rs"));
    let lexical = run(&["index", "--level", "1", "--json"])?;
    assert_eq!(lexical["model_preparation"]["state"], "skipped");
    assert_eq!(lexical["units_indexed"], 0);
    assert_eq!(fs::read_dir(cache.path())?.count(), 1); // admission lock only; no transfer
    Ok(())
}
