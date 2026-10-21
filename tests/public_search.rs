//! Public CLI and MCP hybrid dispatch, fallback and source repair; v1 §§9–10.
use rmcp::{
    RoleClient, ServiceExt, model::CallToolRequestParams, service::RunningService,
    transport::StreamableHttpClientTransport,
};
use serde_json::{Value, json};
use std::{fs, net::TcpListener, path::Path, process::Stdio, time::Duration};
use tokio::{
    process::{Child, Command},
    time::timeout,
};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

type Client = RunningService<RoleClient, ()>;
const WAIT: Duration = Duration::from_secs(20);
const QUERY: &str = "needle_public";

fn command(root: &Path) -> Command {
    let mut command = Command::new(assert_cmd::cargo::cargo_bin!("dkc"));
    command
        .current_dir(root)
        .env("XDG_CONFIG_HOME", root.join("user-config"))
        .env("DKC_MODELS_DIR", root.join("unused-model-cache"))
        .env("DKC_PUBLIC_SEARCH_TEST_KEY", "fixture-key")
        .env_remove("DKC_SERVE__HTTP_TOKEN")
        .kill_on_drop(true);
    command
}
async fn cli(root: &Path, args: &[&str]) -> Value {
    let output = timeout(WAIT, command(root).args(args).output())
        .await
        .unwrap()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
async fn fixture() -> (tempfile::TempDir, MockServer) {
    fixture_policy(false).await
}
async fn fixture_policy(protected: bool) -> (tempfile::TempDir, MockServer) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(|request: &wiremock::Request| {
            let body: Value = request.body_json().unwrap();
            let embeddings = vec![vec![1.0, 0.0]; body["input"].as_array().unwrap().len()];
            ResponseTemplate::new(200)
                .set_body_json(json!({"embeddings": embeddings, "prompt_eval_count": 1}))
        })
        .mount(&server)
        .await;
    Mock::given(method("POST")).and(path("/rerank"))
        .respond_with(|request: &wiremock::Request| {
            let body: Value = request.body_json().unwrap();
            assert_eq!(body["return_documents"], false);
            let documents = body["documents"].as_array().unwrap();
            assert_eq!(body["top_n"].as_u64().unwrap() as usize, documents.len());
            ResponseTemplate::new(200).set_body_json(json!({"results": documents.iter().enumerate()
                .map(|(index, _)| json!({"index": index, "relevance_score": 0.95})).collect::<Vec<_>>()}))
        }).mount(&server).await;
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join(".deepkit")).unwrap();
    fs::create_dir(root.path().join("src")).unwrap();
    fs::create_dir(root.path().join("alt")).unwrap();
    fs::write(
        root.path().join("src/lib.rs"),
        "pub fn needle_public() -> bool { true }\n",
    )
    .unwrap();
    fs::write(
        root.path().join("alt/lib.rs"),
        "pub fn needle_public_alternative() -> bool { false }\n",
    )
    .unwrap();
    fs::write(root.path().join(".deepkit/config.toml"), format!(
        "[embedder]\nprofile='ollama'\ndims=2\n[embedder.ollama]\nurl='{}'\nquery_prefix='PUBLIC_QUERY: '\n[rerank]\nmode='on'\nprovider='remote'\n[rerank.remote]\nurl='{}'\nmodel='fixture'\napi_key_env='DKC_PUBLIC_SEARCH_TEST_KEY'\n[limits]\nquery_timeout_ms=10000\nrerank_timeout_ms=5000\n", server.uri(), server.uri())).unwrap();
    if protected {
        let path = root.path().join(".deepkit/config.toml");
        let config = fs::read_to_string(&path).unwrap();
        fs::write(path, format!("{config}\n[[treatments]]\nname='protected'\nmatch=['alt/**']\nmax_level=1\nnever_send=true\n")).unwrap();
    }
    for args in [
        vec!["init", "-q"],
        vec!["add", "src", "alt"],
        vec![
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
    ] {
        let output = Command::new("git")
            .current_dir(root.path())
            .args(args)
            .output()
            .await
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let indexed = cli(root.path(), &["index", "--level", "2", "--json"]).await;
    assert_eq!(indexed["embedding"]["state"], "quiescent", "{indexed}");
    assert!(counts(&server).await.0 == 0);
    (root, server)
}
async fn counts(server: &MockServer) -> (usize, usize) {
    let requests = server.received_requests().await.unwrap();
    let query = requests
        .iter()
        .filter(|request| {
            if request.url.path() != "/api/embed" {
                return false;
            }
            let body: Value = request.body_json().unwrap();
            body["input"].as_array().unwrap().iter().any(|value| {
                value
                    .as_str()
                    .is_some_and(|text| text.starts_with("PUBLIC_QUERY: "))
            })
        })
        .count();
    (
        query,
        requests
            .iter()
            .filter(|request| request.url.path() == "/rerank")
            .count(),
    )
}
fn hits(value: &Value) -> &[Value] {
    value["hits"].as_array().unwrap()
}
fn complete(value: &Value, reranked: bool) {
    assert!(!hits(value).is_empty(), "{value}");
    assert_eq!(value["index"]["dense"], "complete", "{value}");
    assert_eq!(value["index"]["reranked"], reranked, "{value}");
}
async fn connect(root: &Path, http: bool) -> (Child, Client) {
    if !http {
        let mut child = command(root)
            .args(["serve", "--stdio"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let client = timeout(
            WAIT,
            ().serve((child.stdout.take().unwrap(), child.stdin.take().unwrap())),
        )
        .await
        .unwrap()
        .unwrap();
        return (child, client);
    }
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let address = format!("127.0.0.1:{port}");
    let mut child = command(root)
        .args(["serve", "--http", &address])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    timeout(WAIT, async {
        loop {
            assert!(child.try_wait().unwrap().is_none());
            if tokio::net::TcpStream::connect(&address).await.is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let client = timeout(
        WAIT,
        ().serve(StreamableHttpClientTransport::from_uri(format!(
            "http://{address}/mcp"
        ))),
    )
    .await
    .unwrap()
    .unwrap();
    (child, client)
}
async fn search(client: &Client, args: Value) -> Value {
    let result = timeout(
        WAIT,
        client.call_tool(
            CallToolRequestParams::new("search").with_arguments(args.as_object().unwrap().clone()),
        ),
    )
    .await
    .unwrap()
    .unwrap();
    assert_ne!(result.is_error, Some(true), "{result:?}");
    result.structured_content.unwrap_or_else(|| {
        serde_json::from_str(
            &result
                .content
                .iter()
                .find_map(|content| content.as_text())
                .unwrap()
                .text,
        )
        .unwrap()
    })
}

#[tokio::test]
async fn cli_dispatches_dense_and_rerank_and_honors_explicit_lexical_opt_out() {
    let (root, server) = fixture().await;
    let lexical = cli(
        root.path(),
        &[
            "search",
            QUERY,
            "--mode",
            "lexical",
            "--no-rerank",
            "--json",
        ],
    )
    .await;
    assert!(!hits(&lexical).is_empty());
    assert_eq!(counts(&server).await, (0, 0));
    let hybrid = cli(
        root.path(),
        &["search", QUERY, "--mode", "hybrid", "--json"],
    )
    .await;
    complete(&hybrid, true);
    assert_eq!(hybrid["index"]["rerank"], "complete", "{hybrid}");
    assert_eq!(counts(&server).await, (1, 1));
    let dense = cli(
        root.path(),
        &[
            "search",
            QUERY,
            "--mode",
            "dense",
            "--no-rerank",
            "--scope",
            "src/*",
            "--json",
        ],
    )
    .await;
    complete(&dense, false);
    assert!(hits(&dense).iter().all(|hit| hit["path"] == "src/lib.rs"));
    assert_eq!(counts(&server).await, (2, 1));
    assert!(!root.path().join("unused-model-cache").exists());
}

#[tokio::test]
async fn stdio_and_http_preserve_scope_exclusions_and_repair_without_repeated_model_calls() {
    for http in [false, true] {
        let (root, server) = fixture().await;
        let (mut child, client) = connect(root.path(), http).await;
        let first = search(
            &client,
            json!({"query":QUERY,"mode":"hybrid","scope":["src/*"],"rerank":true}),
        )
        .await;
        complete(&first, true);
        assert_eq!(counts(&server).await, (1, 1));
        assert!(hits(&first).iter().all(|hit| hit["path"] == "src/lib.rs"));
        let ids: Vec<_> = hits(&first).iter().map(|hit| hit["id"].clone()).collect();
        let excluded = search(&client, json!({"query":QUERY,"mode":"lexical","scope":["src/*"],"rerank":false,"exclude_ids":ids})).await;
        assert!(hits(&excluded).is_empty(), "{excluded}");
        assert_eq!(counts(&server).await, (1, 1));
        fs::write(
            root.path().join("src/lib.rs"),
            "// changed source\npub fn needle_public() -> bool { false }\n",
        )
        .unwrap();
        let repaired = search(
            &client,
            json!({"query":QUERY,"mode":"hybrid","scope":["src/*"],"rerank":true}),
        )
        .await;
        assert!(!hits(&repaired).is_empty(), "{repaired}");
        assert!(
            hits(&repaired).iter().all(|hit| !ids.contains(&hit["id"])),
            "{repaired}"
        );
        let after = counts(&server).await;
        assert!(
            after.0 <= 2 && after.1 <= 2,
            "model replay during repair: {after:?}"
        );
        timeout(WAIT, client.cancel()).await.unwrap().unwrap();
        child.kill().await.ok();
        assert!(!root.path().join("unused-model-cache").exists());
    }
}

#[tokio::test]
async fn configured_zero_budget_and_provider_failure_keep_public_lexical_fallback() {
    let (root, server) = fixture().await;
    let config_path = root.path().join(".deepkit/config.toml");
    let original = fs::read_to_string(&config_path).unwrap();
    fs::write(
        &config_path,
        format!(
            "{original}\n[[queue.budgets]]\nprofile='ollama'\nurl='{}'\ndaily_calls=0\n",
            server.uri()
        ),
    )
    .unwrap();
    let denied = cli(
        root.path(),
        &["search", QUERY, "--mode", "hybrid", "--no-rerank", "--json"],
    )
    .await;
    assert!(!hits(&denied).is_empty(), "{denied}");
    assert_ne!(denied["index"]["dense"], "complete", "{denied}");
    assert_eq!(counts(&server).await, (0, 0));
    fs::write(&config_path, original).unwrap();
    server.reset().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(ResponseTemplate::new(401))
        .mount(&server)
        .await;
    let failed = cli(
        root.path(),
        &["search", QUERY, "--mode", "hybrid", "--no-rerank", "--json"],
    )
    .await;
    assert!(!hits(&failed).is_empty(), "{failed}");
    assert_ne!(failed["index"]["dense"], "complete", "{failed}");
    assert_eq!(counts(&server).await, (1, 0));
}

#[tokio::test]
async fn public_mixed_privacy_refusal_and_rerank_timeout_preserve_local_hits() {
    let (root, server) = fixture_policy(true).await;
    let private = cli(
        root.path(),
        &["search", QUERY, "--mode", "hybrid", "--json"],
    )
    .await;
    assert!(
        hits(&private).iter().any(|hit| hit["path"] == "alt/lib.rs"),
        "{private}"
    );
    assert_eq!(private["index"]["rerank"], "privacy", "{private}");
    assert_eq!(counts(&server).await, (1, 0));
    for request in server.received_requests().await.unwrap() {
        assert!(!String::from_utf8_lossy(&request.body).contains("needle_public_alternative"));
    }

    let (root, server) = fixture().await;
    // Exercise a response timeout after dispatch. A 50 ms budget can expire
    // during source verification on hosted runners, legitimately sending no POST.
    // Keep the fixture's 5 s stage budget and delay the mock beyond that budget.
    Mock::given(method("POST"))
        .and(path("/rerank"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(10))
                .set_body_json(json!({"results":[{"index":0,"relevance_score":0.9}]})),
        )
        .with_priority(1)
        .mount(&server)
        .await;
    let timed = cli(
        root.path(),
        &["search", QUERY, "--mode", "hybrid", "--json"],
    )
    .await;
    assert!(!hits(&timed).is_empty(), "{timed}");
    assert_eq!(timed["index"]["rerank"], "timeout", "{timed}");
    assert_eq!(timed["index"]["reranked"], false);
    assert_eq!(timed["index"]["dense"], "complete");
    assert_eq!(counts(&server).await, (1, 1));
}
