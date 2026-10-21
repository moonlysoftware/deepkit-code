//! Public standalone scorer exercises use indexed corpus units and an admitted mock endpoint.
use serde_json::{Value, json};
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

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
    for id in 0..50 {
        fs::write(
            root.path().join(format!("source{id}.rs")),
            format!("pub fn corpus_candidate_{id}() -> bool {{ true }}\n"),
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
            "corpus",
        ],
    );
    fs::create_dir(root.path().join(".deepkit")).unwrap();
    fs::write(root.path().join(".deepkit/config.toml"), format!(
        "[rerank]\nmode='on'\nprovider='remote'\ncandidates=50\n[rerank.remote]\nurl='{}'\nmodel='fixture'\napi_key_env='DKC_BENCH_RERANK_TEST_KEY'\n[limits]\nrerank_timeout_ms=10000\n", server.uri()
    )).unwrap();
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
    let db = rusqlite::Connection::open_with_flags(
        root.path().join(".deepkit/code.db"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .unwrap();
    let mut statement = db
        .prepare("SELECT id FROM units WHERE kind='symbol' ORDER BY id LIMIT 50")
        .unwrap();
    let ids: Vec<i64> = statement
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(ids.len(), 50);
    fs::write(
        root.path().join(".deepkit/units.json"),
        serde_json::to_vec(&ids).unwrap(),
    )
    .unwrap();
    fs::write(
        root.path().join(".deepkit/query.txt"),
        "FIXED_QUERY_CONTENT_MARKER corpus candidate",
    )
    .unwrap();
    root
}
async fn run(root: &Path, calls: &str) -> Output {
    run_to(root, calls, &root.join(".deepkit/measurement.json")).await
}
async fn run_to(root: &Path, calls: &str, out: &Path) -> Output {
    let out = out.to_owned();
    let root = root.to_owned();
    let calls = calls.to_owned();
    tokio::task::spawn_blocking(move || {
        Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
            .current_dir(&root)
            .env("DKC_BENCH_RERANK_TEST_KEY", "fixture-key")
            .args(["bench", "rerank", "--root"])
            .arg(&root)
            .arg("--out")
            .arg(out)
            .arg("--query-file")
            .arg(root.join(".deepkit/query.txt"))
            .arg("--units-file")
            .arg(root.join(".deepkit/units.json"))
            .arg("--model-cache")
            .arg(root.join(".deepkit/absent-model-cache"))
            .args([
                "--max-calls",
                &calls,
                "--max-tokens",
                "1000000",
                "--rerank-rpm",
                "100",
                "--rerank-tpm",
                "1000000",
                "--warmups",
                "1",
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
fn report(root: &Path, output: &Output) -> Value {
    let bytes = fs::read(root.join(".deepkit/measurement.json")).unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    let mut emitted: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(
        emitted
            .as_object_mut()
            .unwrap()
            .remove("took_ms")
            .unwrap()
            .is_number()
    );
    assert_eq!(emitted, value);
    for text in [&bytes[..], &output.stdout[..], &output.stderr[..]] {
        let text = String::from_utf8_lossy(text);
        for private in [
            "FIXED_QUERY_CONTENT_MARKER",
            "corpus_candidate_",
            "PROVIDER_ERROR_CONTENT_MARKER",
            "fixture-key",
        ] {
            assert!(
                !text.contains(private),
                "measurement leaked input or provider body"
            );
        }
    }
    value
}
async fn success(server: &MockServer, calls: u64) {
    Mock::given(method("POST")).and(path("/rerank"))
        .respond_with(|request: &wiremock::Request| {
            let body: Value = request.body_json().unwrap();
            let docs = body["documents"].as_array().unwrap();
            assert_eq!(docs.len(), 50);
            assert_eq!(body["top_n"], 50);
            assert!(docs.iter().all(|doc| doc.as_str().unwrap().contains("corpus_candidate_")));
            ResponseTemplate::new(200).set_body_json(json!({"results": (0..50).map(|index| json!({"index":index,"relevance_score":0.5})).collect::<Vec<_>>() }))
        }).expect(calls).mount(server).await;
}
#[tokio::test]
async fn standalone_cli_scores_fifty_for_warmup_and_each_repeat() {
    let server = MockServer::start().await;
    success(&server, 3).await;
    let root = fixture(&server);
    let output = run(root.path(), "3").await;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value = report(root.path(), &output);
    assert_eq!(value["state"], "complete");
    assert_eq!(value["candidate_count"], 50);
    assert_eq!(value["samples"].as_array().unwrap().len(), 3);
    assert_eq!(value["latency"]["samples"], 2);
    assert_eq!(
        value["after"]["reconciled"].as_i64().unwrap()
            - value["before"]["reconciled"].as_i64().unwrap(),
        3
    );
    assert_eq!(
        value["after"]["sent_units"].as_i64().unwrap()
            - value["before"]["sent_units"].as_i64().unwrap(),
        150
    );
    let requests = server.received_requests().await.unwrap();
    let first: Value = requests[0].body_json().unwrap();
    for request in requests {
        let body: Value = request.body_json().unwrap();
        assert_eq!(body["documents"], first["documents"]);
        assert_eq!(
            request.headers.get("user-agent").unwrap().to_str().unwrap(),
            format!("dkc/{}", env!("CARGO_PKG_VERSION"))
        );
    }
    server.verify().await;
}
#[tokio::test]
async fn standalone_cli_retains_failed_measured_sample_when_warmup_exhausts_budget() {
    let server = MockServer::start().await;
    success(&server, 1).await;
    let root = fixture(&server);
    let output = run(root.path(), "1").await;
    assert!(!output.status.success());
    let value = report(root.path(), &output);
    assert_eq!(value["state"], "failed");
    assert_eq!(value["samples"].as_array().unwrap().len(), 2);
    assert_eq!(value["samples"][0]["outcome"], "passed");
    assert_eq!(value["samples"][1]["outcome"], "score_refused_or_failed");
    assert!(value["latency"].is_null());
    server.verify().await;
}
#[tokio::test]
async fn standalone_cli_rate_limit_keeps_failure_without_retry_or_reduction() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/rerank"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "1")
                .set_body_string("PROVIDER_ERROR_CONTENT_MARKER"),
        )
        .expect(1)
        .mount(&server)
        .await;
    let root = fixture(&server);
    let output = run(root.path(), "3").await;
    assert!(!output.status.success());
    let value = report(root.path(), &output);
    assert_eq!(value["samples"].as_array().unwrap().len(), 1);
    assert!(value["latency"].is_null());
    assert_eq!(value["after"]["dispatched"], 1);
    assert_eq!(
        value["samples"][0]["failure"],
        serde_json::json!({"category":"provider_http", "http_status":429})
    );
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].body_json::<Value>().unwrap()["documents"]
            .as_array()
            .unwrap()
            .len(),
        50
    );
    server.verify().await;
}
#[tokio::test]
async fn standalone_cli_refuses_non_fifty_and_auto_before_any_request() {
    let server = MockServer::start().await;
    let root = fixture(&server);
    let file = root.path().join(".deepkit/units.json");
    let original = fs::read(&file).unwrap();
    let ids: Vec<i64> = serde_json::from_slice(&original).unwrap();
    for invalid in [
        ids[..49].to_vec(),
        vec![ids[0]; 50],
        [ids.clone(), vec![999999]].concat(),
    ] {
        fs::write(&file, serde_json::to_vec(&invalid).unwrap()).unwrap();
        assert!(!run(root.path(), "3").await.status.success());
    }
    fs::write(&file, original).unwrap();
    let config = root.path().join(".deepkit/config.toml");
    fs::write(
        &config,
        fs::read_to_string(&config)
            .unwrap()
            .replace("mode='on'", "mode='auto'"),
    )
    .unwrap();
    assert!(!run(root.path(), "3").await.status.success());
    assert!(server.received_requests().await.unwrap().is_empty());
    assert!(!root.path().join(".deepkit/measurement.json").exists());
}

#[tokio::test]
async fn standalone_cli_refuses_protected_whole_set_without_sending_a_subset() {
    let server = MockServer::start().await;
    let root = fixture(&server);
    let config_path = root.path().join(".deepkit/config.toml");
    let config = fs::read_to_string(&config_path).unwrap();
    fs::write(&config_path, format!("{config}\n[[treatments]]\nname='protected'\nmatch=['source0.rs']\nmax_level=1\nnever_send=true\n")).unwrap();
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
    // Refresh occurrence IDs after the explicit index, while retaining all50 paths.
    let db = rusqlite::Connection::open_with_flags(
        root.path().join(".deepkit/code.db"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .unwrap();
    let ids: Vec<i64> = db
        .prepare("SELECT id FROM units WHERE kind='symbol' ORDER BY id LIMIT 50")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(ids.len(), 50);
    fs::write(
        root.path().join(".deepkit/units.json"),
        serde_json::to_vec(&ids).unwrap(),
    )
    .unwrap();
    drop(db);
    let output = run(root.path(), "3").await;
    assert!(!output.status.success());
    let value = report(root.path(), &output);
    assert_eq!(value["samples"][0]["outcome"], "preparation_refused");
    assert_eq!(value["after"]["dispatched"], 0);
    assert_eq!(value["after"]["reconciled"], 0);
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[cfg(unix)]
#[tokio::test]
async fn standalone_cli_refuses_existing_and_linked_report_destinations() {
    use std::os::unix::fs::symlink;
    let server = MockServer::start().await;
    let root = fixture(&server);
    let old_report = root.path().join(".deepkit/existing.json");
    fs::write(&old_report, "previous evidence").unwrap();
    let model = root.path().join(".deepkit/fixture.gguf");
    fs::write(&model, "unchanged model bytes").unwrap();
    let link = root.path().join(".deepkit/report-link.json");
    symlink(&model, &link).unwrap();
    let hardlink = root.path().join(".deepkit/report-hardlink.json");
    fs::hard_link(&model, &hardlink).unwrap();
    for target in [
        root.path().join(".deepkit/code.db"),
        root.path().join(".deepkit/config.toml"),
        old_report,
        model.clone(),
        link,
        hardlink,
    ] {
        let before = fs::read(&target).unwrap();
        let output = run_to(root.path(), "3", &target).await;
        assert!(!output.status.success());
        assert_eq!(fs::read(&target).unwrap(), before);
        assert!(server.received_requests().await.unwrap().is_empty());
    }
    let alias = root.path().join("report-parent");
    symlink(root.path().join(".deepkit"), &alias).unwrap();
    assert!(
        !run_to(root.path(), "3", &alias.join("new.json"))
            .await
            .status
            .success()
    );
    assert!(!root.path().join(".deepkit/new.json").exists());
    assert_eq!(fs::read(&model).unwrap(), b"unchanged model bytes");
    assert!(server.received_requests().await.unwrap().is_empty());
}
