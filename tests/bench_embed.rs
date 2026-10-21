//! Explicitly bounded real CLI embedding preparation and durable resume reports.
use serde_json::Value;
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

#[tokio::test]
async fn missing_historical_expectations_are_explicit_misses_with_strict_default_validation() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(response())
        .mount(&server)
        .await;
    let root = fixture(&server);
    let prepared = run(root.path(), &root.path().join("embed.json"), "100", "100").await;
    assert!(
        prepared.status.success(),
        "{}",
        String::from_utf8_lossy(&prepared.stderr)
    );
    let golden = root.path().join("historical.yaml");
    let cases = serde_json::json!([
        {"id":"present","query":"retry_0","kind":"ident","expect_files":["source0.rs"]},
        {"id":"deleted","query":"retry_0","kind":"ident","expect_files":["deleted.rs"]},
        {"id":"partial","query":"retry_0","kind":"ident","expect_files":["source0.rs","deleted.rs"]}
    ]);
    fs::write(&golden, serde_json::to_vec(&cases).unwrap()).unwrap();
    let out = root.path().join("search.json");
    let base = vec!["--golden".into(), golden.display().to_string()];
    let before = server.received_requests().await.unwrap().len();
    let rejected = search_run_with_args(root.path(), &out, "100", false, base.clone()).await;
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("expects missing file"));
    assert_eq!(before, server.received_requests().await.unwrap().len());
    let mut allowed = base.clone();
    allowed.push("--allow-missing-expected-files".into());
    let accepted = search_run_with_args(root.path(), &out, "100", false, allowed.clone()).await;
    assert!(
        accepted.status.success(),
        "{}",
        String::from_utf8_lossy(&accepted.stderr)
    );
    let report: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(
        report["method"],
        "in-process-missing-expected-files-diagnostic-v1"
    );
    assert_eq!(report["handwritten"]["missing_files_allowed"], true);
    assert_eq!(
        report["handwritten"]["missing_expected_files"],
        serde_json::json!({"deleted":["deleted.rs"],"partial":["deleted.rs"]})
    );
    assert_eq!(report["handwritten"]["cases"].as_array().unwrap().len(), 3);
    for score in report["scores"].as_array().unwrap() {
        let arm = &score["handwritten"]["lexical"];
        assert_eq!(arm["cohorts"]["positive"]["positive_count"], 3);
        assert_eq!(arm["cases"][0]["hit_file_at_5"], true);
        assert_eq!(arm["cases"][1]["hit_file_at_5"], false);
        assert_eq!(arm["cases"][1]["recall_at_10"], 0.0);
        assert_eq!(arm["cases"][2]["recall_at_10"], 0.5);
        assert_eq!(arm["cohorts"]["positive"]["hit_file_at_5"], 2.0 / 3.0);
    }
    let before = server.received_requests().await.unwrap().len();
    let mut reviewed = allowed.clone();
    reviewed.extend(["--golden-reviewed-blake3".into(), "not-a-review".into()]);
    assert!(
        !search_run_with_args(root.path(), &out, "100", false, reviewed)
            .await
            .status
            .success()
    );
    for invalid in [
        serde_json::json!([{"id":"escape","query":"retry","kind":"ident","expect_files":["../outside.rs"]}]),
        serde_json::json!([{"id":"symbol","query":"retry","kind":"ident","expect_files":["deleted.rs"],"expect_symbols":["absent_symbol"]}]),
    ] {
        fs::write(&golden, serde_json::to_vec(&invalid).unwrap()).unwrap();
        assert!(
            !search_run_with_args(root.path(), &out, "100", false, allowed.clone())
                .await
                .status
                .success()
        );
    }
    assert_eq!(before, server.received_requests().await.unwrap().len());
}

fn git(root: &Path, args: &[&str]) {
    let output = Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
fn fixture(server: &MockServer) -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    for id in 0..3 {
        fs::write(
            root.path().join(format!("source{id}.rs")),
            format!("pub fn retry_{id}() -> bool {{ true }}\n"),
        )
        .unwrap();
    }
    git(root.path(), &["init", "--quiet"]);
    git(root.path(), &["add", "."]);
    git(
        root.path(),
        &[
            "-c",
            "user.name=Deepkit Test",
            "-c",
            "user.email=deepkit@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
    );
    fs::create_dir(root.path().join(".deepkit")).unwrap();
    fs::write(
        root.path().join(".deepkit/config.toml"),
        format!(
            "[embedder]\nprofile='ollama'\ndims=2\n[embedder.ollama]\nurl='{}'\nmodel='fixture'\n",
            server.uri()
        ),
    )
    .unwrap();
    let output = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root.path())
        .args(["index", "--level", "1", "--json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    root
}
async fn run(root: &Path, out: &Path, calls: &str, passes: &str) -> Output {
    let root = root.to_owned();
    let out = out.to_owned();
    let calls = calls.to_owned();
    let passes = passes.to_owned();
    tokio::task::spawn_blocking(move || {
        Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
            .current_dir(&root)
            .args(["bench", "embed", "--root"])
            .arg(&root)
            .arg("--out")
            .arg(out)
            .arg("--model-cache")
            .arg(root.join("absent-model-cache"))
            .args([
                "--max-calls",
                &calls,
                "--max-tokens",
                "100000",
                "--attempt-timeout-ms",
                "2000",
                "--heartbeat-ms",
                "500",
                "--max-schedule",
                "1",
                "--max-jobs",
                "1",
                "--max-passes",
                &passes,
                "--json",
            ])
            .output()
            .unwrap()
    })
    .await
    .unwrap()
}
fn report(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "{error}: {} / {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    })
}
fn response() -> ResponseTemplate {
    ResponseTemplate::new(200)
        .set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]],"prompt_eval_count":1}))
}
#[tokio::test]
async fn bounded_cli_embedding_resumes_without_repeating_done_outputs() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(response())
        .mount(&server)
        .await;
    let root = fixture(&server);
    let reports = tempfile::tempdir().unwrap();
    let out = reports.path().join("embed.json");
    let bounded = run(root.path(), &out, "100", "1").await;
    assert!(!bounded.status.success());
    let bounded = report(&bounded);
    assert_eq!(bounded["state"], "bounded");
    assert_eq!(bounded["passes"], 1);
    assert_eq!(bounded["providers"][0]["published"], 1);
    assert_eq!(bounded["after"]["l2"]["covered_units"], 1);
    let resumed = run(root.path(), &out, "100", "100").await;
    assert!(
        resumed.status.success(),
        "{}",
        String::from_utf8_lossy(&resumed.stderr)
    );
    let resumed = report(&resumed);
    assert_eq!(resumed["state"], "quiescent");
    let covered = resumed["after"]["l2"]["covered_units"].as_u64().unwrap();
    assert!(covered >= 3);
    assert_eq!(
        resumed["providers"][0]["published"].as_u64().unwrap(),
        covered - 1
    );
    assert_eq!(
        server.received_requests().await.unwrap().len() as u64,
        covered
    );
    let again = run(root.path(), &out, "100", "100").await;
    assert!(again.status.success());
    let again = report(&again);
    assert_eq!(again["providers"][0]["published"], 0);
    assert_eq!(
        server.received_requests().await.unwrap().len() as u64,
        covered
    );
    let saved: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    let mut comparable = again.clone();
    assert!(
        comparable
            .as_object_mut()
            .unwrap()
            .remove("took_ms")
            .is_some()
    );
    assert_eq!(saved, comparable);
    assert_eq!(again["providers"].as_array().unwrap().len(), 1);
    assert!(again["build"]["binary_blake3"].is_string());
    assert_eq!(again["bounds"]["max_calls"], 100);
    assert!(
        again["transition"]["previous"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}
#[tokio::test]
async fn zero_budget_and_provider_failure_are_visible_without_false_completion() {
    let server = MockServer::start().await;
    let root = fixture(&server);
    let reports = tempfile::tempdir().unwrap();
    let output = run(root.path(), &reports.path().join("zero.json"), "0", "100").await;
    assert!(!output.status.success());
    let value = report(&output);
    assert_eq!(value["state"], "waiting");
    assert!(value["after"]["l2"]["pending_jobs"].as_i64().unwrap() > 0);
    assert_eq!(value["after"]["l2"]["covered_units"], 0);
    assert!(server.received_requests().await.unwrap().is_empty());
    let second = fixture(&server);
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(400))
        .mount(&server)
        .await;
    let output = run(
        second.path(),
        &reports.path().join("failed.json"),
        "100",
        "100",
    )
    .await;
    assert!(!output.status.success());
    let value = report(&output);
    assert_eq!(value["state"], "failed");
    assert_eq!(value["providers"][0]["state"], "failed");
    assert_eq!(value["after"]["l2"]["covered_units"], 0);
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
    assert!(value["error"].is_string());
}
#[tokio::test]
async fn missing_bounds_or_changed_head_refuses_before_embedding() {
    let server = MockServer::start().await;
    let root = fixture(&server);
    let reports = tempfile::tempdir().unwrap();
    let output = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root.path())
        .args(["bench", "embed", "--root"])
        .arg(root.path())
        .arg("--out")
        .arg(reports.path().join("missing.json"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!reports.path().join("missing.json").exists());
    git(
        root.path(),
        &[
            "-c",
            "user.name=Deepkit Test",
            "-c",
            "user.email=deepkit@example.invalid",
            "commit",
            "--allow-empty",
            "--quiet",
            "-m",
            "changed HEAD",
        ],
    );
    let output = run(
        root.path(),
        &reports.path().join("changed.json"),
        "100",
        "100",
    )
    .await;
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("indexed HEAD differs"));
    assert!(!reports.path().join("changed.json").exists());
    assert!(server.received_requests().await.unwrap().is_empty());
}

async fn search_run(root: &Path, out: &Path, calls: &str) -> Output {
    search_run_with_policy(root, out, calls, false).await
}
async fn search_run_with_policy(
    root: &Path,
    out: &Path,
    calls: &str,
    continue_degraded: bool,
) -> Output {
    search_run_with_args(root, out, calls, continue_degraded, Vec::new()).await
}
async fn search_run_with_args(
    root: &Path,
    out: &Path,
    calls: &str,
    continue_degraded: bool,
    extra: Vec<String>,
) -> Output {
    let root = root.to_owned();
    let out = out.to_owned();
    let calls = calls.to_owned();
    tokio::task::spawn_blocking(move || {
        Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
            .current_dir(&root)
            .args(["bench", "search"])
            .args(extra)
            .args(continue_degraded.then_some("--continue-degraded"))
            .arg("--root")
            .arg(&root)
            .arg("--out")
            .arg(out)
            .args([
                "--max-calls",
                &calls,
                "--max-tokens",
                "100000",
                "--repeats",
                "2",
                "--json",
            ])
            .output()
            .unwrap()
    })
    .await
    .unwrap()
}
fn golden_commit(root: &Path) {
    fs::write(
        root.join("source0.rs"),
        "/// retry policy retries failed requests safely\npub fn retry_0() -> bool { false }\n",
    )
    .unwrap();
    git(root, &["add", "source0.rs"]);
    git(
        root,
        &[
            "-c",
            "user.name=Deepkit Test",
            "-c",
            "user.email=deepkit@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "retry policy retries failed requests safely in source zero",
        ],
    );
    let result = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .args(["index", "--level", "1", "--json"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}
#[tokio::test]
async fn dense_only_benchmark_uses_vectors_and_retains_separate_cost_quality_and_refusals() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(response())
        .mount(&server)
        .await;
    let root = fixture(&server);
    golden_commit(root.path());
    let prepared = run(root.path(), &root.path().join("embed.json"), "100", "100").await;
    assert!(
        prepared.status.success(),
        "{}",
        String::from_utf8_lossy(&prepared.stderr)
    );
    let before = server.received_requests().await.unwrap().len();
    let out = root.path().join("dense-search.json");
    let result =
        search_run_with_args(root.path(), &out, "100", false, vec!["--dense-only".into()]).await;
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(report["dense_only"], true);
    assert_eq!(report["method"], "in-process-dense-only-arm-diagnostic-v1");
    assert_eq!(report["pairs"].as_array().unwrap().len(), 3);
    for pair in report["pairs"].as_array().unwrap() {
        assert_eq!(pair["dense_only"]["dense"], "Complete");
        assert_eq!(pair["dense_only"]["lexical"], "NotRequested");
        assert_eq!(pair["dense_only"]["rerank"], "NotRequested");
        assert_eq!(pair["dense_only"]["degraded"], false);
        assert!(pair["reranked"].is_null());
    }
    for score in report["scores"].as_array().unwrap() {
        assert_eq!(score["dense_only"]["commit_derived"]["count"], 1);
        assert_eq!(score["dense_only"]["commit_derived"]["hit_file_at_5"], 1.0);
    }
    assert_eq!(report["latency"]["dense_only"]["samples"], 2);
    // One hybrid and one dense-only request per query, including the warmup.
    assert_eq!(server.received_requests().await.unwrap().len() - before, 6);

    let golden = root.path().join("dense.yaml");
    fs::write(
        &golden,
        "- id: semantic
  query: nonexistentxyz
  kind: vague
  expect_files: [source0.rs]
",
    )
    .unwrap();
    let args = vec![
        "--dense-only".into(),
        "--golden".into(),
        golden.display().to_string(),
    ];
    let result = search_run_with_args(root.path(), &out, "100", false, args.clone()).await;
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    for pair in report["pairs"].as_array().unwrap() {
        assert!(pair["lexical"]["paths"].as_array().unwrap().is_empty());
        assert!(
            !pair["dense_only"]["evidence"]["hits"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }
    for score in report["scores"].as_array().unwrap() {
        assert_eq!(
            score["handwritten"]["dense_only"]["cases"][0]["hit_file_at_5"],
            true
        );
        assert_eq!(
            score["handwritten"]["lexical"]["cases"][0]["hit_file_at_5"],
            false
        );
        assert_eq!(
            score["handwritten"]["dense_only"]["cohorts"]["positive"]["positive_count"],
            1
        );
    }
    let before = server.received_requests().await.unwrap().len();
    let refused = search_run_with_args(root.path(), &out, "0", false, args).await;
    assert!(!refused.status.success());
    let report: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(report["pairs"].as_array().unwrap().len(), 1);
    assert_eq!(report["pairs"][0]["dense_only"]["degraded"], true);
    assert!(report["scores"].as_array().unwrap().is_empty());
    assert_eq!(server.received_requests().await.unwrap().len(), before);
}

#[tokio::test]
async fn paired_search_reports_real_modes_denominators_repeats_and_budget_failure() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]],"prompt_eval_count":1})),
        )
        .mount(&server)
        .await;
    let root = fixture(&server);
    golden_commit(root.path());
    let embedding = run(root.path(), &root.path().join("embed.json"), "100", "100").await;
    assert!(
        embedding.status.success(),
        "{}",
        String::from_utf8_lossy(&embedding.stderr)
    );
    let before = server.received_requests().await.unwrap().len();
    let out = root.path().join("search.json");
    let result = search_run(root.path(), &out, "100").await;
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(report["state"], "complete");
    assert_eq!(report["dense_only"], false);
    assert_eq!(report["goldens"].as_array().unwrap().len(), 1);
    assert_eq!(report["pairs"].as_array().unwrap().len(), 3);
    assert_eq!(report["scores"].as_array().unwrap().len(), 2);
    for pair in report["pairs"].as_array().unwrap() {
        assert!(pair["dense_only"].is_null());
        assert_eq!(pair["lexical"]["dense"], "NotRequested");
        assert_eq!(pair["hybrid"]["dense"], "Complete");
        assert_eq!(pair["lexical"]["degraded"], false);
        assert_eq!(pair["hybrid"]["degraded"], false);
    }
    for score in report["scores"].as_array().unwrap() {
        for mode in ["lexical", "hybrid"] {
            assert_eq!(score[mode]["commit_derived"]["count"], 1);
            assert_eq!(score[mode]["commit_derived"]["hit_file_at_5"], 1.0);
            assert_eq!(score[mode]["eligibility"]["eligible"]["count"], 1);
            assert_eq!(score[mode]["eligibility"]["raw_expected_paths"], 1);
        }
    }
    assert_eq!(report["latency"]["hybrid"]["samples"], 2);
    assert_eq!(
        report["latency"]["hybrid"]["sets_ms"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(server.received_requests().await.unwrap().len(), before + 3);
    let refused = search_run(root.path(), &out, "0").await;
    assert!(!refused.status.success());
    let report: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(report["state"], "failed");
    assert_eq!(report["pairs"].as_array().unwrap().len(), 1);
    assert_eq!(report["pairs"][0]["hybrid"]["degraded"], true);
    assert_eq!(report["scores"].as_array().unwrap().len(), 0);
    assert!(report["latency"].is_null());
    assert_eq!(server.received_requests().await.unwrap().len(), before + 3);
    let continued = search_run_with_policy(root.path(), &out, "0", true).await;
    assert!(!continued.status.success());
    let report: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(report["state"], "degraded");
    assert_eq!(report["continue_degraded"], true);
    assert_eq!(report["pairs"].as_array().unwrap().len(), 3);
    assert_eq!(report["scores"].as_array().unwrap().len(), 2);
    assert_eq!(report["latency"]["hybrid"]["samples"], 2);
    for pair in report["pairs"].as_array().unwrap() {
        assert_eq!(pair["hybrid"]["degraded"], true);
    }
    // Continuing diagnostics neither widens admission nor dispatches after refusal.
    assert_eq!(server.received_requests().await.unwrap().len(), before + 3);
}

#[tokio::test]
async fn paired_search_refuses_unprepared_routes_without_model_calls() {
    let server = MockServer::start().await;
    let root = fixture(&server);
    golden_commit(root.path());
    let out = root.path().join("search.json");
    let result = search_run(root.path(), &out, "100").await;
    assert!(!result.status.success());
    let report: Value = serde_json::from_slice(&fs::read(out).unwrap()).unwrap();
    assert_eq!(report["state"], "failed");
    assert!(
        report["error"]
            .as_str()
            .unwrap()
            .contains("routes have not been prepared")
    );
    assert!(report["scores"].as_array().unwrap().is_empty());
    assert!(server.received_requests().await.unwrap().is_empty());
}

async fn rerank_search_run(root: &Path, out: &Path, rpm: &str) -> Output {
    rerank_search_args(root, out, rpm, Vec::new()).await
}
async fn rerank_search_args(root: &Path, out: &Path, rpm: &str, extra: Vec<String>) -> Output {
    let root = root.to_owned();
    let out = out.to_owned();
    let rpm = rpm.to_owned();
    tokio::task::spawn_blocking(move || {
        Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
            .current_dir(&root)
            .env("DKC_TEST_RERANK_KEY", "fixture-key")
            .args(["bench", "search", "--root"])
            .arg(&root)
            .arg("--out")
            .arg(out)
            .args([
                "--max-calls",
                "100",
                "--max-tokens",
                "100000",
                "--repeats",
                "2",
                "--rerank",
                "--rerank-rpm",
                &rpm,
                "--rerank-tpm",
                "100000",
                "--json",
            ])
            .args(extra)
            .output()
            .unwrap()
    })
    .await
    .unwrap()
}

#[tokio::test]
async fn rerank_benchmark_scores_separate_arm_and_retains_rate_refusal() {
    use wiremock::matchers::path;
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]],"prompt_eval_count":1})),
        )
        .mount(&server)
        .await;
    Mock::given(method("POST")).and(path("/rerank"))
        .respond_with(|request: &wiremock::Request| {
            let body: Value = request.body_json().unwrap();
            assert_eq!(body["return_documents"], false);
            let documents = body["documents"].as_array().unwrap();
            assert_eq!(body["top_n"].as_u64().unwrap() as usize, documents.len());
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"results":
                documents.iter().enumerate().map(|(index, doc)| serde_json::json!({
                    "index":index,"relevance_score":if doc.as_str().unwrap().contains("retry_0") {0.95} else {0.0}
                })).collect::<Vec<_>>() }))
        }).mount(&server).await;
    let root = fixture(&server);
    golden_commit(root.path());
    let config_path = root.path().join(".deepkit/config.toml");
    let mut config = fs::read_to_string(&config_path).unwrap();
    config.push_str(&format!("\n[rerank]\nmode='on'\nprovider='remote'\n[rerank.remote]\nurl='{}'\nmodel='fixture'\napi_key_env='DKC_TEST_RERANK_KEY'\n",server.uri()));
    fs::write(&config_path, config).unwrap();
    assert!(
        run(root.path(), &root.path().join("embed.json"), "100", "100")
            .await
            .status
            .success()
    );
    let out = root.path().join("rerank-search.json");
    let result = rerank_search_run(root.path(), &out, "100").await;
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(report["state"], "complete");
    assert_eq!(report["rerank"], true);
    assert_eq!(report["rerank_rpm"], 100);
    assert_eq!(report["rerank_before"]["effective_candidates"], 50);
    assert_eq!(report["pairs"].as_array().unwrap().len(), 3);
    for pair in report["pairs"].as_array().unwrap() {
        assert_eq!(pair["hybrid"]["rerank"], "NotRequested");
        assert_eq!(pair["reranked"]["rerank"], "Complete");
        assert_eq!(pair["reranked"]["degraded"], false);
        assert_eq!(pair["reranked"]["paths"], serde_json::json!(["source0.rs"]));
        assert!(pair["reranked"]["candidate_count"].as_u64().unwrap() > 1);
    }
    for score in report["scores"].as_array().unwrap() {
        assert_eq!(score["reranked"]["commit_derived"]["count"], 1);
        assert_eq!(score["reranked"]["commit_derived"]["hit_file_at_5"], 1.0);
    }
    assert_eq!(report["latency"]["reranked"]["samples"], 2);
    let requests = server.received_requests().await.unwrap();
    assert_eq!(
        requests
            .iter()
            .filter(|r| r.url.path() == "/rerank")
            .count(),
        3
    );
    // Distinct hybrid requests occur in both the unreranked and reranked arms.
    let embeds = requests
        .iter()
        .filter(|r| r.url.path() == "/api/embed")
        .count();
    let refused = rerank_search_run(root.path(), &out, "0").await;
    assert!(!refused.status.success());
    let report: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(report["state"], "failed");
    assert_eq!(report["pairs"][0]["hybrid"]["degraded"], false);
    assert_eq!(report["pairs"][0]["reranked"]["degraded"], true);
    assert!(
        report["pairs"][0]["reranked"]["rerank"]
            .as_str()
            .unwrap()
            .contains("RequestTooLarge"),
        "{}",
        report["pairs"][0]["reranked"]
    );
    assert_eq!(
        report["pairs"][0]["hybrid"]["paths"],
        report["pairs"][0]["reranked"]["paths"]
    );
    assert!(report["latency"].is_null());
    let requests = server.received_requests().await.unwrap();
    assert_eq!(
        requests
            .iter()
            .filter(|r| r.url.path() == "/rerank")
            .count(),
        3
    );
    assert_eq!(
        requests
            .iter()
            .filter(|r| r.url.path() == "/api/embed")
            .count(),
        embeds + 2
    );
    let primary = root.path().join("primary.yaml");
    let extras = root.path().join("extras.yaml");
    let input = "- id: primary
  query: retry policy
  kind: ident
  expect_files: [source0.rs]
  expect_symbols: [retry_0]
";
    fs::write(&primary, input).unwrap();
    fs::write(
        &extras,
        "- id: scoped
  query: retry
  kind: ident
  cohort: scoped
  scope: source1.rs
  expect_files: [source1.rs]
- id: absent
  query: nonexistentxyz
  kind: vague
  cohort: no_answer
  expect_files: []
",
    )
    .unwrap();
    let args = vec![
        "--dense-only".into(),
        "--golden".into(),
        primary.display().to_string(),
        "--extra-golden".into(),
        extras.display().to_string(),
    ];
    let handwritten = rerank_search_args(root.path(), &out, "100", args.clone()).await;
    assert!(
        handwritten.status.success(),
        "{}",
        String::from_utf8_lossy(&handwritten.stderr)
    );
    let report: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(report["state"], "complete");
    assert!(report["goldens"].as_array().unwrap().is_empty());
    assert_eq!(report["handwritten"]["review_status"], "unreviewed_draft");
    assert_eq!(report["pairs"].as_array().unwrap().len(), 9);
    for scores in report["scores"].as_array().unwrap() {
        assert!(scores["lexical"].is_null());
        for arm in ["lexical", "hybrid", "reranked", "dense_only"] {
            let score = &scores["handwritten"][arm];
            assert_eq!(score["cohorts"]["positive"]["positive_count"], 1);
            assert_eq!(score["cohorts"]["positive"]["symbol_query_count"], 1);
            assert_eq!(score["cohorts"]["positive"]["hit_symbol_at_5"], 1.0);
            assert_eq!(score["cohorts"]["scoped"]["positive_count"], 1);
            assert_eq!(score["cohorts"]["no_answer"]["positive_count"], 0);
            assert_eq!(score["cohorts"]["no_answer"]["no_answer_count"], 1);
            assert!(score["cohorts"]["no_answer"]["recall_at_10"].is_null());
            assert_eq!(score["latency"]["samples"], 3);
        }
    }
    assert_eq!(report["latency"]["reranked"]["samples"], 6);
    for pair in report["pairs"].as_array().unwrap() {
        for arm in ["lexical", "hybrid", "reranked", "dense_only"] {
            let evidence = &pair[arm]["evidence"];
            assert!(evidence["response_tokens"].as_u64().unwrap() > 0);
            assert!(!evidence.to_string().contains("snippet"));
            if pair["golden"] == 1 {
                assert!(
                    pair[arm]["paths"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .all(|p| p == "source1.rs")
                );
            }
        }
    }
    let refused = rerank_search_args(root.path(), &out, "0", args.clone()).await;
    assert!(!refused.status.success());
    let stopped: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(stopped["pairs"].as_array().unwrap().len(), 1);
    assert!(stopped["scores"].as_array().unwrap().is_empty());
    let before = server.received_requests().await.unwrap().len();
    let mut wrong = args;
    wrong.extend(["--golden-reviewed-blake3".into(), "wrong".into()]);
    let rejected = rerank_search_args(root.path(), &out, "100", wrong).await;
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("reviewed golden hash"));
    assert_eq!(server.received_requests().await.unwrap().len(), before);
}

#[test]
fn rerank_benchmark_requires_explicit_rates_before_any_setup() {
    let output = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .args([
            "bench",
            "search",
            "--root",
            "/absent",
            "--out",
            "/absent/report",
            "--max-calls",
            "1",
            "--max-tokens",
            "1",
            "--rerank",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let message = String::from_utf8_lossy(&output.stderr);
    assert!(message.contains("--rerank-rpm") && message.contains("--rerank-tpm"));
    assert!(message.contains("required arguments"));
}

#[tokio::test]
async fn candidate_ablation_preserves_explicit_scopes_and_excludes_pathless_commits() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(response())
        .mount(&server)
        .await;
    let root = fixture(&server);
    golden_commit(root.path());
    let output = run(root.path(), &root.path().join("embed.json"), "100", "100").await;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let out = root.path().join("search.json");
    let output =
        search_run_with_args(root.path(), &out, "100", false, vec!["--all-scopes".into()]).await;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let all: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert!(all["default_scope"].is_null());
    assert_eq!(
        all["method"],
        "in-process-explicit-candidate-ablation-diagnostic-v1"
    );
    assert!(all["pairs"].as_array().unwrap().iter().any(|p| {
        p["lexical"]["paths"]
            .as_array()
            .unwrap()
            .contains(&Value::String(String::new()))
    }));
    assert!(all["pairs"].as_array().unwrap().iter().any(|p| {
        p["lexical"]["kinds"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("commit"))
    }));
    let output = search_run_with_args(
        root.path(),
        &out,
        "100",
        false,
        vec![
            "--all-scopes".into(),
            "--exclude-kind".into(),
            "commit".into(),
            "--exclude-kind".into(),
            "commit".into(),
        ],
    )
    .await;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let excluded: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(excluded["exclude_kinds"], serde_json::json!(["commit"]));
    assert!(excluded["excluded_occurrences"].as_u64().unwrap() > 0);
    assert_ne!(excluded["excluded_ids_blake3"], all["excluded_ids_blake3"]);
    assert_eq!(excluded["goldens"], all["goldens"]);
    for p in excluded["pairs"].as_array().unwrap() {
        for arm in ["lexical", "hybrid"] {
            let paths = p[arm]["paths"].as_array().unwrap();
            assert!(paths.contains(&serde_json::json!("source0.rs")));
            assert!(
                !p[arm]["kinds"]
                    .as_array()
                    .unwrap()
                    .contains(&serde_json::json!("commit"))
            );
            assert_eq!(paths.len(), p[arm]["kinds"].as_array().unwrap().len());
        }
    }
    let golden = root.path().join("scoped.yaml");
    fs::write(&golden,"- id: scoped\n  query: retry\n  kind: ident\n  scope: source1.rs\n  expect_files: [source1.rs]\n").unwrap();
    let output = search_run_with_args(
        root.path(),
        &out,
        "100",
        false,
        vec![
            "--all-scopes".into(),
            "--golden".into(),
            golden.display().to_string(),
        ],
    )
    .await;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let scoped: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(scoped["handwritten"]["review_status"], "unreviewed_draft");
    for p in scoped["pairs"].as_array().unwrap() {
        for arm in ["lexical", "hybrid"] {
            assert_eq!(p[arm]["paths"], serde_json::json!(["source1.rs"]));
        }
    }
    let before = server.received_requests().await.unwrap().len();
    let output = search_run_with_args(
        root.path(),
        &out,
        "100",
        false,
        vec!["--exclude-kind".into(), "bogus".into()],
    )
    .await;
    assert!(!output.status.success());
    assert_eq!(server.received_requests().await.unwrap().len(), before);
}

#[tokio::test]
async fn paired_search_reports_capacity_pauses_and_refuses_other_work_or_changed_identities() {
    use rusqlite::{Connection, params};
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(response())
        .mount(&server)
        .await;
    let root = fixture(&server);
    golden_commit(root.path());
    let seed = run(root.path(), &root.path().join("seed.json"), "100", "1").await;
    assert!(
        report(&seed)["after"]["l2"]["covered_units"]
            .as_i64()
            .unwrap()
            > 0
    );
    let config_path = root.path().join(".deepkit/config.toml");
    let config = fs::read_to_string(&config_path).unwrap();
    assert!(config.contains("dims=2"));
    fs::write(
        &config_path,
        config.replace("dims=2", "dims=2\nmax_doc_tokens=1"),
    )
    .unwrap();
    let embed = run(root.path(), &root.path().join("embed.json"), "100", "100").await;
    // Inspect actual retained jobs: normal preparation must establish a real cap pause.
    let dbpath = root.path().join(".deepkit/code.db");
    let db = Connection::open(&dbpath).unwrap();
    let (total, paused): (i64, i64) = db.query_row(
        "SELECT count(*),sum(state='queued' AND not_before=9223372036854775807 AND last_error='embedding document or batch token cap; waiting for capacity') FROM jobs WHERE kind='embed'",
        [], |r| Ok((r.get(0)?,r.get(1)?)),
    ).unwrap();
    assert!(total > 0, "{}", String::from_utf8_lossy(&embed.stderr));
    assert!(paused > 0 && paused < total);
    let target: i64 = db
        .query_row(
            "SELECT min(id) FROM jobs WHERE kind='embed' AND state='queued'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let out = root.path().join("search.json");
    let output = search_run(root.path(), &out, "100").await;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(value["capacity_pauses"]["jobs"], paused);
    assert_eq!(
        value["method"],
        "in-process-capacity-degraded-diagnostic-v1"
    );
    assert_eq!(value["capacity_pauses"]["reasons"]["token_cap"], paused);
    assert!(value["before"]["l2"]["covered_units"].as_i64().unwrap() > 0);
    assert_eq!(value["before"]["l2"]["pending_jobs"], paused);
    assert_eq!(value["goldens"].as_array().unwrap().len(), 1);
    for pair in value["pairs"].as_array().unwrap() {
        assert!(
            pair["lexical"]["paths"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("source0.rs"))
        );
    }
    let original_reason = "embedding document or batch token cap; waiting for capacity";
    for (state, at, reason) in [
        ("queued", 1, original_reason),
        ("failed", 1, original_reason),
        ("leased", i64::MAX, original_reason),
        ("dead", 1, original_reason),
        ("queued", i64::MAX, "private-provider-marker"),
    ] {
        db.execute(
            "UPDATE jobs SET state=?2,not_before=?3,last_error=?4 WHERE id=?1",
            params![target, state, at, reason],
        )
        .unwrap();
        let before = server.received_requests().await.unwrap().len();
        let refused = search_run(root.path(), &out, "100").await;
        assert!(!refused.status.success(), "state={state}");
        assert_eq!(server.received_requests().await.unwrap().len(), before);
        assert!(!String::from_utf8_lossy(&refused.stderr).contains("private-provider-marker"));
        db.execute("UPDATE jobs SET state='queued',not_before=9223372036854775807,last_error=?2 WHERE id=?1",params![target,original_reason]).unwrap();
    }
    server.reset().await;
    let changed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let observed = changed.clone();
    Mock::given(method("POST"))
        .respond_with(move |_: &wiremock::Request| {
            if !changed.swap(true, std::sync::atomic::Ordering::SeqCst) {
                Connection::open(&dbpath)
                    .unwrap()
                    .execute("UPDATE jobs SET tokens=tokens+1 WHERE id=?1", [target])
                    .unwrap();
            }
            response()
        })
        .mount(&server)
        .await;
    let refused = search_run(root.path(), &out, "100").await;
    assert!(
        observed.load(std::sync::atomic::Ordering::SeqCst),
        "{}\n{}",
        String::from_utf8_lossy(&refused.stderr),
        String::from_utf8_lossy(&refused.stdout)
    );
    assert!(!refused.status.success());
    let value: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(value["state"], "failed");
    assert!(
        value["error"]
            .as_str()
            .unwrap()
            .contains("capacity pause cohort changed")
    );
    assert_eq!(
        db.query_row("SELECT count(*) FROM jobs WHERE kind='embed'", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        total
    );
}
