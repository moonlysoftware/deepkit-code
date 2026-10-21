//! Public doctor provider probes use fixed inputs, bounded requests and the existing ledger.
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Output,
    time::Duration,
};
use tokio::process::Command;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, method, path},
};

const KEY_ENV: &str = "DOCTOR_NETWORK_FIXTURE_KEY";
const KEY: &str = "synthetic-doctor-credential";
const SOURCE: &str = "pub fn synthetic_private_source_marker() -> bool { true }\n";
const WAIT: Duration = Duration::from_secs(25);

fn command(root: &Path, isolation: &Path) -> Command {
    let mut command = Command::new(assert_cmd::cargo::cargo_bin!("dkc"));
    command.current_dir(root).kill_on_drop(true);
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().starts_with("DKC_") {
            command.env_remove(name);
        }
    }
    command
        .env_remove("MOONLY_GATEWAY_API_KEY")
        .env("XDG_CONFIG_HOME", isolation.join("user"))
        .env("DKC_MODELS_DIR", isolation.join("absent-model-cache"))
        .env(KEY_ENV, KEY)
        .env("DKC_LOG", "trace");
    command
}
async fn cli(isolation: &Path, args: &[&str]) -> (Output, Value) {
    let output = tokio::time::timeout(
        WAIT,
        command(&isolation.join("repo"), isolation)
            .args(args)
            .output(),
    )
    .await
    .unwrap()
    .unwrap();
    let value = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "expected JSON: {error}; stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output, value)
}
async fn fixture(endpoint: &str, rerank: bool, indexed: bool) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("repo");
    fs::create_dir_all(root.join(".deepkit")).unwrap();
    fs::write(root.join("script.rs"), SOURCE).unwrap();
    let output = Command::new("git")
        .args(["init", "-q"])
        .current_dir(&root)
        .output()
        .await
        .unwrap();
    assert!(output.status.success());
    let mode = if rerank { "on" } else { "off" };
    fs::write(root.join(".deepkit/config.toml"), format!(
        "[embedder]\nprofile='remote'\ndims=2\n[embedder.remote]\nurl='{endpoint}/v1'\nmodel='embed'\napi_key_env='{KEY_ENV}'\nquery_prefix=''\n[rerank]\nmode='{mode}'\nprovider='remote'\n[rerank.remote]\nurl='{endpoint}/v1'\nmodel='rank'\napi_key_env='{KEY_ENV}'\n[limits]\nquery_timeout_ms=5000\nrerank_timeout_ms=5000\n"
    )).unwrap();
    if indexed {
        let (output, report) = cli(dir.path(), &["index", "--level", "1", "--json"]).await;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(report["units_indexed"].as_u64().is_some_and(|n| n > 0));
    }
    dir
}
fn check<'a>(report: &'a Value, id: &str) -> &'a Value {
    report["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["id"] == id)
        .unwrap_or_else(|| panic!("missing {id}: {report}"))
}
fn redacted(output: &Output) {
    for text in [&output.stdout, &output.stderr] {
        let text = String::from_utf8_lossy(text);
        for marker in [
            KEY,
            "synthetic_private_source_marker",
            "synthetic-provider-body",
        ] {
            assert!(
                !text.contains(marker),
                "private diagnostic content was emitted"
            );
        }
    }
}
async fn listing(server: &MockServer, status: u16) {
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("authorization", format!("Bearer {KEY}")))
        .and(header(
            "user-agent",
            concat!("dkc/", env!("CARGO_PKG_VERSION")),
        ))
        .respond_with(ResponseTemplate::new(status).set_body_json(json!({
            "data":[{"id":"embed"},{"id":"rank"}], "error":"synthetic-provider-body"
        })))
        .expect(1)
        .mount(server)
        .await;
}
async fn embedding(server: &MockServer, status: u16) {
    let encoded = STANDARD.encode(
        [1f32, 0f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(header("authorization", format!("Bearer {KEY}")))
        .and(header(
            "user-agent",
            concat!("dkc/", env!("CARGO_PKG_VERSION")),
        ))
        .respond_with(ResponseTemplate::new(status).set_body_json(json!({
            "data":[{"index":0,"embedding":encoded}], "usage":{"prompt_tokens":7,"total_tokens":7},
            "error":"synthetic-provider-body"
        })))
        .expect(1)
        .mount(server)
        .await;
}
fn usage(isolation: &Path, provider_prefix: &str) -> (i64, i64) {
    let db = rusqlite::Connection::open_with_flags(
        isolation.join("repo/.deepkit/code.db"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .unwrap();
    db.query_row(
        "SELECT coalesce(sum(calls),0),coalesce(sum(tokens),0) FROM usage WHERE provider LIKE ?1",
        [format!("{provider_prefix}%")],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )
    .unwrap()
}
fn snapshot(root: &Path) -> BTreeMap<PathBuf, Option<blake3::Hash>> {
    fn walk(root: &Path, path: &Path, out: &mut BTreeMap<PathBuf, Option<blake3::Hash>>) {
        for entry in fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            let relative = path.strip_prefix(root).unwrap().to_owned();
            if path.is_dir() {
                out.insert(relative, None);
                walk(root, &path, out);
            } else {
                out.insert(relative, Some(blake3::hash(&fs::read(path).unwrap())));
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}
async fn assert_public_requests(server: &MockServer) {
    for request in server.received_requests().await.unwrap() {
        let body = String::from_utf8_lossy(&request.body);
        assert!(!body.contains("synthetic_private_source_marker"));
        assert!(!body.contains(KEY));
        if request.url.path() == "/v1/embeddings" {
            let body: Value = request.body_json().unwrap();
            assert_eq!(
                body["input"],
                json!(["Find the function that reads a configuration file."])
            );
            assert_eq!(body["dimensions"], 2);
        }
        if request.url.path() == "/v1/rerank" {
            let body: Value = request.body_json().unwrap();
            assert_eq!(body["query"], "What is the capital of France?");
            assert_eq!(
                body["documents"],
                json!([
                    "The capital of France is Paris.",
                    "Bananas are yellow fruit.",
                    "France is a country in Europe."
                ])
            );
            assert_eq!(body["return_documents"], false);
        }
    }
}

#[tokio::test]
async fn discovery_and_embedding_succeed_and_settle_existing_usage() {
    let server = MockServer::start().await;
    listing(&server, 200).await;
    embedding(&server, 200).await;
    let fixture = fixture(&server.uri(), false, true).await;
    assert_eq!(usage(fixture.path(), "embed:remote:"), (0, 0));
    let source = fs::read(fixture.path().join("repo/script.rs")).unwrap();
    let config = fs::read(fixture.path().join("repo/.deepkit/config.toml")).unwrap();
    let (output, report) = cli(fixture.path(), &["doctor", "--json"]).await;
    assert_eq!(check(&report, "providers.discovery.1")["status"], "passed");
    assert_eq!(
        check(&report, "providers.embedding.code")["status"],
        "passed"
    );
    assert_eq!(
        check(&report, "providers.embedding.code")["evidence"]["dimensions"],
        2
    );
    assert_eq!(
        check(&report, "providers.rerank")["status"],
        "not_applicable"
    );
    assert_eq!(usage(fixture.path(), "embed:remote:"), (1, 7));
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
    assert_eq!(
        fs::read(fixture.path().join("repo/script.rs")).unwrap(),
        source
    );
    assert_eq!(
        fs::read(fixture.path().join("repo/.deepkit/config.toml")).unwrap(),
        config
    );
    assert!(!fixture.path().join("absent-model-cache").exists());
    redacted(&output);
    assert_public_requests(&server).await;
}

#[tokio::test]
async fn discovery_without_usage_ledger_does_not_dispatch_inference_or_create_files() {
    let server = MockServer::start().await;
    listing(&server, 200).await;
    let fixture = fixture(&server.uri(), true, false).await;
    let before = snapshot(fixture.path());
    let (output, report) = cli(fixture.path(), &["doctor", "--json"]).await;
    assert_eq!(check(&report, "providers.discovery.1")["status"], "passed");
    for id in ["providers.embedding.code", "providers.rerank"] {
        assert_eq!(check(&report, id)["status"], "not_checked");
        assert!(
            check(&report, id)["message"]
                .as_str()
                .unwrap()
                .contains("ledger")
        );
    }
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
    assert_eq!(snapshot(fixture.path()), before);
    redacted(&output);
}

#[tokio::test]
async fn listing_failure_does_not_suppress_independent_embedding_probe() {
    let server = MockServer::start().await;
    listing(&server, 503).await;
    embedding(&server, 200).await;
    let fixture = fixture(&server.uri(), false, true).await;
    let (output, report) = cli(fixture.path(), &["doctor", "--json"]).await;
    assert!(!output.status.success());
    assert_eq!(check(&report, "providers.discovery.1")["status"], "failed");
    assert_eq!(
        check(&report, "providers.embedding.code")["status"],
        "passed"
    );
    assert_eq!(usage(fixture.path(), "embed:remote:"), (1, 7));
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
    redacted(&output);
    assert_public_requests(&server).await;
}

#[tokio::test]
async fn remote_rerank_relevance_is_verified_and_bad_relevance_still_charged() {
    for (scores, expected) in [([0.9, 0.1, 0.4], "passed"), ([0.5, 0.5, 0.5], "failed")] {
        let server = MockServer::start().await;
        listing(&server, 200).await;
        embedding(&server, 503).await;
        Mock::given(method("POST")).and(path("/v1/rerank"))
            .and(header("authorization", format!("Bearer {KEY}")))
            .and(header("user-agent", concat!("dkc/", env!("CARGO_PKG_VERSION"))))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"results":
                scores.into_iter().enumerate().map(|(index, score)| json!({"index":index,"relevance_score":score})).collect::<Vec<_>>()
            }))).expect(1).mount(&server).await;
        let fixture = fixture(&server.uri(), true, true).await;
        let (output, report) = cli(fixture.path(), &["doctor", "--json"]).await;
        assert_eq!(
            check(&report, "providers.embedding.code")["status"],
            "failed"
        );
        assert_eq!(check(&report, "providers.rerank")["status"], expected);
        assert_eq!(usage(fixture.path(), "rerank:remote:").0, 1);
        assert_eq!(server.received_requests().await.unwrap().len(), 3);
        redacted(&output);
        assert_public_requests(&server).await;
    }
}

async fn ollama_fixture(endpoint: &str) -> tempfile::TempDir {
    let fixture = fixture(endpoint, false, false).await;
    fs::write(fixture.path().join("repo/.deepkit/config.toml"), format!(
        "[embedder]\nprofile='ollama'\ndims=2\n[embedder.ollama]\nurl='{endpoint}'\nmodel='synthetic-private-ollama-model:latest'\nquery_prefix=''\n[rerank]\nmode='off'\n[limits]\nquery_timeout_ms=2000\n"
    )).unwrap();
    fixture
}

#[tokio::test]
async fn ollama_down_is_reported_without_an_index_or_filesystem_mutation() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let fixture = ollama_fixture(&endpoint).await;
    let before = snapshot(fixture.path());
    let (output, report) = cli(fixture.path(), &["doctor", "--json"]).await;
    assert!(!output.status.success());
    let embedding = check(&report, "providers.embedding.code");
    assert_eq!(embedding["status"], "failed");
    assert_eq!(
        embedding["message"],
        "Ollama is unavailable or returned an invalid model listing"
    );
    assert_eq!(
        check(&report, "providers.discovery")["status"],
        "not_applicable"
    );
    assert_eq!(snapshot(fixture.path()), before);
    redacted(&output);
    for bytes in [&output.stdout, &output.stderr] {
        let text = String::from_utf8_lossy(bytes);
        assert!(!text.contains("synthetic-private-ollama-model"));
    }
    // Transport trace may identify the credential-free destination; the doctor
    // report itself contains only closed diagnoses and numeric evidence.
    assert!(!String::from_utf8_lossy(&output.stdout).contains(&endpoint));
}

#[tokio::test]
async fn ollama_missing_model_lists_once_without_embedding_pull_or_index_creation() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .and(header(
            "user-agent",
            concat!("dkc/", env!("CARGO_PKG_VERSION")),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models":[{"name":"synthetic-unrelated-installed-model:latest"}]
        })))
        .expect(1)
        .mount(&server)
        .await;
    let fixture = ollama_fixture(&server.uri()).await;
    let before = snapshot(fixture.path());
    let (output, report) = cli(fixture.path(), &["doctor", "--json"]).await;
    assert!(!output.status.success());
    let embedding = check(&report, "providers.embedding.code");
    assert_eq!(embedding["status"], "failed");
    assert_eq!(
        embedding["message"],
        "Configured Ollama model is not installed; no model was pulled"
    );
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].url.path(), "/api/tags");
    assert_eq!(snapshot(fixture.path()), before);
    redacted(&output);
    for bytes in [&output.stdout, &output.stderr] {
        let text = String::from_utf8_lossy(bytes);
        assert!(!text.contains("synthetic-private-ollama-model"));
        assert!(!text.contains("synthetic-unrelated-installed-model"));
    }
}

#[cfg(unix)]
#[tokio::test]
async fn ledger_replacement_during_ollama_listing_refuses_inference_without_charging_either_file() {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    let server = MockServer::start().await;
    let fixture = ollama_fixture(&server.uri()).await;
    let (output, indexed) = cli(fixture.path(), &["index", "--level", "1", "--json"]).await;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(indexed["units_indexed"].as_u64().is_some_and(|n| n > 0));
    let database = fixture.path().join("repo/.deepkit/code.db");
    let preserved = fixture.path().join("original-code.db");
    let staging = fixture.path().join("replacement-code.db");
    assert_eq!(usage(fixture.path(), "embed:ollama:"), (0, 0));
    // Prepare an ordinary, valid SQLite replacement before doctor starts. DELETE
    // mode makes its bytes independent of the old connection's open WAL handles.
    fs::copy(&database, &staging).unwrap();
    {
        let db = rusqlite::Connection::open(&staging).unwrap();
        db.pragma_update(None, "journal_mode", "DELETE").unwrap();
        db.execute_batch("CREATE TABLE replacement_marker(value TEXT); INSERT INTO replacement_marker VALUES('replacement');").unwrap();
    }
    let replacement_before = blake3::hash(&fs::read(&staging).unwrap());
    let source_before = fs::read(fixture.path().join("repo/script.rs")).unwrap();
    let config_before = fs::read(fixture.path().join("repo/.deepkit/config.toml")).unwrap();
    let swapped = Arc::new(AtomicBool::new(false));
    let responder_swapped = Arc::clone(&swapped);
    let responder_database = database.clone();
    let responder_preserved = preserved.clone();
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(move |_: &wiremock::Request| {
            assert!(
                !responder_swapped.swap(true, Ordering::SeqCst),
                "listing must not retry"
            );
            fs::rename(&responder_database, &responder_preserved).unwrap();
            // Preserve the original SQLite evidence under matching sidecar names;
            // the process still holds descriptors for these original files.
            for suffix in ["-wal", "-shm"] {
                let from = PathBuf::from(format!("{}{suffix}", responder_database.display()));
                let to = PathBuf::from(format!("{}{suffix}", responder_preserved.display()));
                if from.exists() {
                    fs::rename(from, to).unwrap();
                }
            }
            fs::rename(&staging, &responder_database).unwrap();
            ResponseTemplate::new(200).set_body_json(json!({
                "models":[{"name":"synthetic-private-ollama-model:latest"}]
            }))
        })
        .expect(1)
        .mount(&server)
        .await;
    let (output, report) = cli(fixture.path(), &["doctor", "--json"]).await;
    assert!(swapped.load(Ordering::SeqCst));
    let embedding = check(&report, "providers.embedding.code");
    assert_eq!(embedding["status"], "not_checked");
    assert_eq!(
        embedding["message"],
        "Shared admission or changed policy prevents this diagnostic request"
    );
    let requests = server.received_requests().await.unwrap();
    assert_eq!(
        requests.len(),
        1,
        "no embedding or pull may follow ledger replacement"
    );
    assert_eq!(requests[0].url.path(), "/api/tags");
    assert_eq!(
        blake3::hash(&fs::read(&database).unwrap()),
        replacement_before
    );
    assert_eq!(usage(fixture.path(), "embed:ollama:"), (0, 0));
    for path in [&database, &preserved] {
        let db =
            rusqlite::Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
                .unwrap();
        let calls: i64 = db
            .query_row("SELECT coalesce(sum(calls),0) FROM usage", [], |row| {
                row.get(0)
            })
            .unwrap();
        let reservations: i64 = db
            .query_row(
                "SELECT count(*) FROM usage_reservations WHERE state != 'released'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(calls, 0);
        assert_eq!(reservations, 0);
        // The original ledger keeps its released, unsent reservation for audit;
        // it is neither active nor a provider charge. The replacement stays empty.
        let released: i64 = db
            .query_row(
                "SELECT count(*) FROM usage_reservations WHERE state='released'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(released, i64::from(path == &preserved));
        let units: i64 = db
            .query_row("SELECT count(*) FROM units", [], |row| row.get(0))
            .unwrap();
        assert!(units > 0, "original index evidence must remain readable");
    }
    assert_eq!(
        fs::read(fixture.path().join("repo/script.rs")).unwrap(),
        source_before
    );
    assert_eq!(
        fs::read(fixture.path().join("repo/.deepkit/config.toml")).unwrap(),
        config_before
    );
    assert!(!fixture.path().join("absent-model-cache").exists());
    redacted(&output);
}

#[tokio::test]
async fn remote_discovery_failures_have_exact_safe_diagnoses() {
    for (status, expected) in [
        (
            401,
            "The provider rejected the API key (401); check the configured credential",
        ),
        (200, "A configured model is missing from provider discovery"),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(status).set_body_json(json!({
                "data":[], "error":"synthetic-provider-body"
            })))
            .expect(1)
            .mount(&server)
            .await;
        let fixture = fixture(&server.uri(), false, false).await;
        let before = snapshot(fixture.path());
        let (output, report) = cli(fixture.path(), &["doctor", "--json"]).await;
        assert!(!output.status.success());
        let discovery = check(&report, "providers.discovery.1");
        assert_eq!(discovery["status"], "failed");
        assert_eq!(discovery["message"], expected);
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
        assert_eq!(snapshot(fixture.path()), before);
        redacted(&output);
    }
}

#[tokio::test]
async fn unreachable_gateway_has_exact_safe_discovery_diagnosis() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let fixture = fixture(&endpoint, false, false).await;
    let before = snapshot(fixture.path());
    let (output, report) = cli(fixture.path(), &["doctor", "--json"]).await;
    assert!(!output.status.success());
    let discovery = check(&report, "providers.discovery.1");
    assert_eq!(discovery["status"], "failed");
    assert_eq!(
        discovery["message"],
        "Model discovery could not reach the configured provider"
    );
    assert_eq!(snapshot(fixture.path()), before);
    assert!(!String::from_utf8_lossy(&output.stdout).contains(&endpoint));
    redacted(&output);
}

#[tokio::test]
async fn absent_credential_has_exact_diagnosis_without_provider_dispatch() {
    let server = MockServer::start().await;
    let fixture = fixture(&server.uri(), false, false).await;
    let before = snapshot(fixture.path());
    let output = tokio::time::timeout(
        WAIT,
        command(&fixture.path().join("repo"), fixture.path())
            .env_remove(KEY_ENV)
            .args(["doctor", "--json"])
            .output(),
    )
    .await
    .unwrap()
    .unwrap();
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(!output.status.success());
    let discovery = check(&report, "providers.discovery.1");
    assert_eq!(discovery["status"], "failed");
    assert_eq!(
        discovery["message"],
        "Provider credential is missing or unreadable; check DOCTOR_NETWORK_FIXTURE_KEY and ~/.config/deepkit-code/credentials.toml (or the configured user-config directory)"
    );
    assert!(server.received_requests().await.unwrap().is_empty());
    assert_eq!(snapshot(fixture.path()), before);
    redacted(&output);
}

#[tokio::test]
async fn dimensions_rejection_reports_falls_back_with_two_separate_accounted_attempts() {
    let server = MockServer::start().await;
    listing(&server, 200).await;
    let encoded = STANDARD.encode(
        [1f32, 0f32, 0.5f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(move |request: &wiremock::Request| {
            let body: Value = request.body_json().unwrap();
            assert_eq!(
                body["input"],
                json!(["Find the function that reads a configuration file."])
            );
            if body.get("dimensions").is_some() {
                assert_eq!(body["dimensions"], 2);
                ResponseTemplate::new(400).set_body_json(json!({
                    "error":{"message":"model does not support matryoshka representation"}
                }))
            } else {
                ResponseTemplate::new(200).set_body_json(json!({
                    "data":[{"index":0,"embedding":encoded}], "usage":{"total_tokens":7}
                }))
            }
        })
        .expect(2)
        .mount(&server)
        .await;
    let fixture = fixture(&server.uri(), false, true).await;
    let (output, report) = cli(fixture.path(), &["doctor", "--json"]).await;
    let embedding = check(&report, "providers.embedding.code");
    assert_eq!(embedding["status"], "passed");
    assert_eq!(
        embedding["message"],
        "Embedding succeeded; rejected dimensions falls back to local truncation"
    );
    assert_eq!(embedding["evidence"]["dimensions"], 2);
    assert_eq!(embedding["evidence"]["dimensions_fallback"], true);
    assert!(embedding["evidence"]["latency_ms"].as_u64().is_some());
    let requests = server.received_requests().await.unwrap();
    let embeds: Vec<_> = requests
        .iter()
        .filter(|request| request.url.path() == "/v1/embeddings")
        .collect();
    assert_eq!(embeds.len(), 2);
    assert!(
        embeds[0]
            .body_json::<Value>()
            .unwrap()
            .get("dimensions")
            .is_some()
    );
    assert!(
        embeds[1]
            .body_json::<Value>()
            .unwrap()
            .get("dimensions")
            .is_none()
    );
    assert_eq!(requests.len(), 3);
    // The rejected first request remains conservatively charged as dispatched;
    // only the successful second response reconciles its actual token usage.
    assert_eq!(usage(fixture.path(), "embed:remote:"), (1, 7));
    let db = rusqlite::Connection::open_with_flags(
        fixture.path().join("repo/.deepkit/code.db"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .unwrap();
    let states: (i64, i64, i64) = db.query_row(
        "SELECT count(DISTINCT id),sum(state='dispatched'),sum(state='reconciled') FROM usage_reservations WHERE provider LIKE 'embed:remote:%'",
        [], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?))).unwrap();
    assert_eq!(states, (2, 1, 1));
    redacted(&output);
}

#[tokio::test]
async fn installed_ollama_model_embeds_once_with_usage_and_latency_without_pulling() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .and(header(
            "user-agent",
            concat!("dkc/", env!("CARGO_PKG_VERSION")),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models":[{"name":"synthetic-private-ollama-model:latest"}]
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .and(header(
            "user-agent",
            concat!("dkc/", env!("CARGO_PKG_VERSION")),
        ))
        .respond_with(|request: &wiremock::Request| {
            let body: Value = request.body_json().unwrap();
            assert_eq!(body["model"], "synthetic-private-ollama-model:latest");
            assert_eq!(
                body["input"],
                json!(["Find the function that reads a configuration file."])
            );
            assert!(
                !String::from_utf8_lossy(&request.body).contains("synthetic_private_source_marker")
            );
            ResponseTemplate::new(200)
                .set_body_json(json!({"embeddings":[[1.0,0.0]], "prompt_eval_count":11}))
        })
        .expect(1)
        .mount(&server)
        .await;
    let fixture = ollama_fixture(&server.uri()).await;
    let (indexed, _) = cli(fixture.path(), &["index", "--level", "1", "--json"]).await;
    assert!(indexed.status.success());
    let (output, report) = cli(fixture.path(), &["doctor", "--json"]).await;
    let embedding = check(&report, "providers.embedding.code");
    assert_eq!(embedding["status"], "passed");
    assert_eq!(
        embedding["message"],
        "Embedding succeeded with one fixed public input"
    );
    assert_eq!(embedding["evidence"]["dimensions"], 2);
    assert_eq!(embedding["evidence"]["dimensions_fallback"], false);
    assert!(embedding["evidence"]["latency_ms"].as_u64().is_some());
    assert_eq!(usage(fixture.path(), "embed:ollama:"), (1, 11));
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests
            .iter()
            .all(|request| matches!(request.url.path(), "/api/tags" | "/api/embed"))
    );
    assert!(!fixture.path().join("absent-model-cache").exists());
    assert_eq!(
        fs::read_to_string(fixture.path().join("repo/script.rs")).unwrap(),
        SOURCE
    );
    redacted(&output);
}
