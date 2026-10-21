//! Doctor retains separate local smoke evidence inside its collected diagnosis report.
use serde_json::Value;
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};
fn run(root: &Path, cache: &Path) -> Output {
    Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .env("DKC_MODELS_DIR", cache)
        .env("XDG_CONFIG_HOME", root.join("isolated-user"))
        .env_remove("MOONLY_GATEWAY_API_KEY")
        .env_remove("DOCTOR_UNUSED_FIXTURE_KEY")
        .args(["doctor", "--json"])
        .output()
        .unwrap()
}
fn config(root: &Path, text: &str) {
    fs::create_dir_all(root.join(".deepkit")).unwrap();
    fs::write(root.join(".deepkit/config.toml"), text).unwrap();
}
#[test]
fn missing_and_corrupt_artifacts_fail_without_index_or_download() {
    let root = tempfile::tempdir().unwrap();
    let cache = root.path().join("cache");
    config(
        root.path(),
        "[embedder]\nprofile='local'\n[rerank.local]\ngpu_layers=0\n",
    );
    let missing = run(root.path(), &cache);
    assert!(!missing.status.success());
    let missing: Value = serde_json::from_slice(&missing.stdout).unwrap();
    assert_eq!(missing["scope"], "diagnostics");
    assert_eq!(missing["status"], "failed");
    let missing = &missing["local_reranker"];
    assert_eq!(missing["status"], "failed");
    assert!(
        missing["message"]
            .as_str()
            .unwrap()
            .contains("missing model")
    );
    assert!(!cache.exists());
    fs::create_dir(&cache).unwrap();
    let file = missing["file"].as_str().unwrap();
    fs::write(cache.join(file), b"not a GGUF").unwrap();
    let broken = run(root.path(), &cache);
    assert!(!broken.status.success());
    let broken: Value = serde_json::from_slice(&broken.stdout).unwrap();
    assert_eq!(broken["status"], "failed");
    let broken = &broken["local_reranker"];
    assert!(broken["initial_load"].is_null());
    assert!(broken["message"].as_str().unwrap().contains(file));
    assert_eq!(fs::read(cache.join(file)).unwrap(), b"not a GGUF");
    assert_eq!(
        fs::read_dir(root.path().join(".deepkit")).unwrap().count(),
        1
    );
}
#[test]
fn unused_local_reranker_is_not_reported_healthy_or_loaded() {
    let root = tempfile::tempdir().unwrap();
    let cache = root.path().join("absent");
    for text in ["[rerank]\nmode='off'\n", "[rerank]\nprovider='remote'\n"] {
        config(
            root.path(),
            &format!(
                "[embedder]\nprofile='remote'\n[embedder.remote]\nurl='http://127.0.0.1:9/v1'\napi_key_env='DOCTOR_UNUSED_FIXTURE_KEY'\n{text}[rerank.remote]\nurl='http://127.0.0.1:9/v1'\napi_key_env='DOCTOR_UNUSED_FIXTURE_KEY'\n"
            ),
        );
        let result = run(root.path(), &cache);
        assert!(!result.status.success());
        let report: Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(report["status"], "failed");
        let report = &report["local_reranker"];
        assert_eq!(report["status"], "not_applicable");
        assert!(report["initial_load"].is_null());
        assert!(!cache.exists());
        assert!(!root.path().join(".deepkit/code.db").exists());
    }
}
#[test]
#[ignore = "requires DKC_RERANKER_CACHE with verified pinned weights; CPU only"]
fn verified_cpu_doctor_reports_one_initial_smoke_with_separate_timings() {
    let cache = std::env::var_os("DKC_RERANKER_CACHE").expect("verified model cache required");
    let root = tempfile::tempdir().unwrap();
    // A narrow query candidate cap must not shrink the fixed three-pair health gate.
    config(
        root.path(),
        "[embedder]\nprofile='local'\n[rerank]\ncandidates=1\n[rerank.local]\ngpu_layers=0\n",
    );
    let result = run(root.path(), Path::new(&cache));
    let report: Value = serde_json::from_slice(&result.stdout).unwrap();
    let report = &report["local_reranker"];
    assert_eq!(report["status"], "passed");
    let smoke = &report["initial_load"];
    assert!(smoke["scores"][0].as_f64().unwrap() > 0.5);
    assert!(smoke["scores"][1].as_f64().unwrap() < 0.2);
    assert!(smoke["preparation_ms"].as_f64().unwrap() > 0.0);
    assert!(smoke["smoke_ms"].as_f64().unwrap() > 0.0);
    assert!(smoke["threads"].as_i64().unwrap() > 0);
    assert_eq!(smoke["requested_gpu_layers"], 0);
    assert!(!root.path().join(".deepkit/code.db").exists());
    println!("{}", report);
}

#[test]
fn later_status_reports_saved_failure_and_ignores_changed_selection() {
    let root = tempfile::tempdir().unwrap();
    let git = Command::new("git")
        .current_dir(root.path())
        .args(["init", "--quiet"])
        .output()
        .unwrap();
    assert!(git.status.success());
    fs::write(root.path().join("sample.rs"), "pub fn sample() {}\n").unwrap();
    config(
        root.path(),
        "[embedder]\nprofile='local'\n[rerank.local]\ngpu_layers=0\n",
    );
    let index = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root.path())
        .args(["index", "--level", "1", "--json"])
        .output()
        .unwrap();
    assert!(
        index.status.success(),
        "{}",
        String::from_utf8_lossy(&index.stderr)
    );
    let result = run(root.path(), &root.path().join("absent-cache"));
    assert!(!result.status.success());
    let result: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(result["local_reranker"]["persistence"], "saved");
    let status = || {
        let result = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
            .current_dir(root.path())
            .args(["status", "--json"])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        serde_json::from_slice::<Value>(&result.stdout).unwrap()
    };
    let saved = status();
    assert_eq!(saved["last_local_reranker_check"]["state"], "unavailable");
    assert!(
        saved["last_local_reranker_check"]["detail"]
            .as_str()
            .unwrap()
            .contains("missing model")
    );
    config(root.path(), "[rerank]\nmode='off'\n");
    assert!(status()["last_local_reranker_check"].is_null());
}
