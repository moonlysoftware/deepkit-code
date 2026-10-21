//! Advertised MCP argument contracts over both transports; v1 §10.1, T1.9.
use rmcp::{
    RoleClient, ServiceExt,
    model::{CallToolRequestParams, CallToolResult},
    service::RunningService,
    transport::StreamableHttpClientTransport,
};
use serde_json::{Value, json};
use std::{fs, net::TcpListener, path::Path, process::Stdio, time::Duration};
use tokio::{
    process::{Child, Command},
    time::{Instant, timeout},
};

const DEADLINE: Duration = Duration::from_secs(10);
type Client = RunningService<RoleClient, ()>;

#[test]
fn cli_search_rejects_explicit_and_default_bounds_before_zero_deadline() {
    let root = fixture();
    let path = root.path().join(".deepkit/config.toml");
    let config = fs::read_to_string(&path)
        .unwrap()
        .replace("max_k=100", "max_k=100\ndefault_k=51");
    fs::write(&path, format!("{config}\n[limits]\nquery_timeout_ms=0\n")).unwrap();
    for mode in ["lexical", "dense", "hybrid"] {
        for k in [None, Some("0"), Some("51")] {
            let mut cmd = std::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"));
            cmd.current_dir(root.path())
                .env("XDG_CONFIG_HOME", root.path().join("user-config"))
                .args(["search", "should_retry", "--mode", mode, "--json"]);
            if let Some(k) = k {
                cmd.args(["-k", k]);
            }
            let output = cmd.output().unwrap();
            assert!(!output.status.success(), "{mode} {k:?}: {output:?}");
            assert!(
                String::from_utf8_lossy(&output.stderr).contains("k must be between 1 and 50"),
                "{output:?}"
            );
        }
    }
}

fn fixture() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join("src")).unwrap();
    fs::write(
        root.path().join("src/lib.rs"),
        "pub fn should_retry(attempt: u32) -> bool { attempt < 3 }\n",
    )
    .unwrap();
    fs::create_dir(root.path().join(".deepkit")).unwrap();
    fs::write(root.path().join(".deepkit/config.toml"), "[embedder]\nprofile='local'\n[rerank]\nmode='off'\n[search]\nmax_k=100\n[serve]\nagent_writable_config=true\n").unwrap();
    for args in [
        vec!["init", "--quiet"],
        vec!["add", "src"],
        vec![
            "-c",
            "user.name=MCP Test",
            "-c",
            "user.email=mcp@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
    ] {
        let output = std::process::Command::new("git")
            .current_dir(root.path())
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let output = std::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root.path())
        .env("XDG_CONFIG_HOME", root.path().join("user-config"))
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

fn command(root: &Path) -> Command {
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("dkc"));
    cmd.current_dir(root)
        .env("XDG_CONFIG_HOME", root.join("user-config"))
        .env("DKC_MODELS_DIR", root.join("absent-model-cache"))
        .env_remove("DKC_SERVE__HTTP_TOKEN")
        .stderr(Stdio::null())
        .kill_on_drop(true);
    cmd
}

async fn connect(root: &Path, http: bool) -> (Child, Client) {
    if !http {
        let mut child = command(root)
            .args(["serve", "--stdio"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let transport = (child.stdout.take().unwrap(), child.stdin.take().unwrap());
        let client = timeout(DEADLINE, ().serve(transport))
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
        .spawn()
        .unwrap();
    let deadline = Instant::now() + DEADLINE;
    loop {
        assert!(
            child.try_wait().unwrap().is_none(),
            "HTTP server exited before bind"
        );
        if tokio::net::TcpStream::connect(&address).await.is_ok() {
            break;
        }
        assert!(Instant::now() < deadline, "HTTP server did not bind");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let transport = StreamableHttpClientTransport::from_uri(format!("http://{address}/mcp"));
    let client = timeout(DEADLINE, ().serve(transport))
        .await
        .unwrap()
        .unwrap();
    (child, client)
}

async fn call(client: &Client, name: &'static str, args: Value) -> CallToolResult {
    let result = timeout(
        DEADLINE,
        client.call_tool(
            CallToolRequestParams::new(name).with_arguments(args.as_object().unwrap().clone()),
        ),
    )
    .await
    .unwrap()
    .unwrap();
    let elapsed = result
        .meta
        .as_ref()
        .unwrap()
        .get("took_ms")
        .unwrap()
        .as_f64()
        .unwrap();
    assert!(elapsed.is_finite() && elapsed >= 0.0, "{result:?}");
    result
}

fn payload(result: &CallToolResult) -> Value {
    result.structured_content.clone().unwrap_or_else(|| {
        serde_json::from_str(
            &result
                .content
                .iter()
                .find_map(|c| c.as_text())
                .unwrap()
                .text,
        )
        .unwrap()
    })
}

fn error(result: &CallToolResult, code: &str) {
    assert_eq!(result.is_error, Some(true), "{result:?}");
    let body = payload(result);
    assert_eq!(body["error"]["code"], code, "{body}");
    for field in ["message", "hint"] {
        assert!(body["error"][field].is_string(), "{body}");
    }
}

#[tokio::test]
async fn advertised_schemas_expose_enums_ranges_and_nullable_options() {
    for http in [false, true] {
        let root = fixture();
        let (mut child, client) = connect(root.path(), http).await;
        let tools = timeout(DEADLINE, client.list_all_tools())
            .await
            .unwrap()
            .unwrap();
        let schema = |name: &str| {
            tools
                .iter()
                .find(|tool| tool.name == name)
                .unwrap()
                .input_schema
                .get("properties")
                .unwrap()
        };
        assert_eq!(
            schema("search")["mode"]["enum"],
            json!(["lexical", "dense", "hybrid", null])
        );
        assert_eq!(
            schema("config_set")["layer"]["enum"],
            json!(["repo", "local", null])
        );
        assert_eq!(
            schema("recent")["granularity"]["enum"],
            json!(["pr", "commit", null])
        );
        assert_eq!(schema("search")["k"]["minimum"].as_f64(), Some(1.0));
        assert_eq!(schema("search")["k"]["maximum"].as_f64(), Some(50.0));
        for (name, field) in [("refs", "k"), ("recent", "k"), ("map", "budget_tokens")] {
            assert_eq!(schema(name)[field]["minimum"].as_f64(), Some(1.0));
            assert!(
                schema(name)[field].get("maximum").is_none(),
                "Unexpected search ceiling for {name}"
            );
        }
        let lines = &schema("open")["lines"];
        assert_eq!(lines["type"], json!(["array", "null"]));
        assert_eq!(lines["minItems"], 2);
        assert_eq!(lines["maxItems"], 2);
        assert_eq!(lines["items"]["minimum"], 1);
        assert_eq!(lines["items"]["maximum"], i64::MAX);
        timeout(DEADLINE, client.cancel()).await.unwrap().unwrap();
        if http {
            child.kill().await.unwrap();
        } else {
            assert!(
                timeout(DEADLINE, child.wait())
                    .await
                    .unwrap()
                    .unwrap()
                    .success()
            );
        }
    }
}

#[tokio::test]
async fn invalid_arguments_are_structured_and_valid_controls_keep_the_session_usable() {
    for http in [false, true] {
        let root = fixture();
        let (mut child, client) = connect(root.path(), http).await;
        for (name, args) in [
            ("search", json!({"query":"should_retry", "k":0})),
            ("search", json!({"query":"should_retry", "k":51})),
            (
                "search",
                json!({"query":"should_retry", "mode":"unsupported"}),
            ),
            ("refs", json!({"name":"should_retry", "k":0})),
            ("map", json!({"budget_tokens":0})),
            ("recent", json!({"granularity":"unsupported"})),
            ("recent", json!({"k":0})),
            (
                "config_set",
                json!({"key":"search.default_k", "value":5, "layer":"user"}),
            ),
            ("open", json!({"path":"src/lib.rs", "lines":[0,1]})),
            ("open", json!({"path":"src/lib.rs", "lines":[2,1]})),
            (
                "open",
                json!({"path":"src/lib.rs", "lines":[1,9223372036854775808u64]}),
            ),
            ("open", json!({"id":"irrelevant", "lines":[1,1]})),
        ] {
            error(&call(&client, name, args).await, "invalid_arguments");
        }
        let changed = call(
            &client,
            "config_set",
            json!({"key":"search.default_k", "value":51, "layer":"local"}),
        )
        .await;
        assert_ne!(changed.is_error, Some(true), "{changed:?}");
        error(
            &call(&client, "search", json!({"query":"should_retry"})).await,
            "invalid_arguments",
        );
        let restored = call(
            &client,
            "config_set",
            json!({"key":"search.default_k", "value":10, "layer":"local"}),
        )
        .await;
        assert_ne!(restored.is_error, Some(true), "{restored:?}");
        for k in [1, 50] {
            let result = call(
                &client,
                "search",
                json!({"query":"should_retry", "k":k, "mode":null, "scope":null}),
            )
            .await;
            assert_ne!(result.is_error, Some(true), "{result:?}");
            assert!(!payload(&result)["hits"].as_array().unwrap().is_empty());
        }
        let opened = call(
            &client,
            "open",
            json!({"path":"src/lib.rs", "lines":[1,1], "context":0}),
        )
        .await;
        assert_ne!(opened.is_error, Some(true), "{opened:?}");
        for granularity in [Value::Null, json!("pr"), json!("commit")] {
            let recent = call(&client, "recent", json!({"granularity":granularity, "k":1})).await;
            assert_ne!(recent.is_error, Some(true), "{recent:?}");
            let rows = payload(&recent);
            assert_eq!(rows.as_array().unwrap().len(), 1);
            assert_eq!(rows[0]["kind"], "commit");
            assert_eq!(rows[0]["title"], "fixture");
        }
        timeout(DEADLINE, client.cancel()).await.unwrap().unwrap();
        if http {
            child.kill().await.unwrap();
        } else {
            assert!(
                timeout(DEADLINE, child.wait())
                    .await
                    .unwrap()
                    .unwrap()
                    .success()
            );
        }
        assert!(!root.path().join("absent-model-cache").exists());
    }
}
