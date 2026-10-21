//! Public zero-setup serving, progress and restart acceptance; v1 §2.1/10.2.
#![cfg(unix)]
use rmcp::{
    RoleClient, ServiceExt,
    model::{CallToolRequestParams, CallToolResult},
    service::RunningService,
    transport::StreamableHttpClientTransport,
};
use serde_json::{Value, json};
use std::{
    env, fs,
    net::TcpListener,
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};
use tokio::{
    process::{Child, Command},
    time::{Instant, timeout, timeout_at},
};

type Client = RunningService<RoleClient, ()>;
static TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
const CALL: Duration = Duration::from_secs(10);
// Large enough for repeated L1 WAL admissions; assertions require observed
// building/interrupted state rather than assuming a minimum machine duration.
const INITIAL_FILES: usize = 500;
const COMPLETE: Duration = Duration::from_secs(120);
const PIN: &str = "3fce3b5bb0236da2df6d99672afb8a719642eca7";

fn command(root: &Path) -> Command {
    let mut command = Command::new(assert_cmd::cargo::cargo_bin!("dkc"));
    for (name, _) in env::vars_os() {
        if name.to_string_lossy().starts_with("DKC_") {
            command.env_remove(name);
        }
    }
    command
        .current_dir(root)
        .env("XDG_CONFIG_HOME", root.join("absent-user-config"))
        .env("DKC_MODELS_DIR", root.join("absent-model-cache"))
        // Keep CI freshness refusals diagnosable without source-bearing logs.
        .env("DKC_LOG", "warn,dkc::queue::destination=debug")
        .env_remove("MOONLY_GATEWAY_API_KEY")
        .stderr(Stdio::inherit())
        .kill_on_drop(true);
    command
}
fn git(root: &Path, args: &[&str]) -> Vec<u8> {
    let output = std::process::Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}
fn fixture(files: usize) -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join("src")).unwrap();
    for index in 0..files {
        let source = (0..8).map(|function| {
            format!("pub fn auto_probe_{index}_{function}(value: usize) -> usize {{ value + {function} }}\n")
        }).collect::<String>();
        fs::write(root.path().join(format!("src/file_{index:04}.rs")), source).unwrap();
    }
    git(root.path(), &["init", "-q"]);
    git(root.path(), &["add", "src"]);
    git(
        root.path(),
        &[
            "-c",
            "user.name=Auto Init",
            "-c",
            "user.email=auto@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
    );
    root
}
async fn cli(root: &Path, args: &[&str]) -> Value {
    let output = timeout(CALL, command(root).args(args).output())
        .await
        .unwrap()
        .unwrap();
    assert!(
        output.status.success(),
        "CLI operation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
async fn connect(root: &Path, http: bool) -> (Child, Client) {
    if !http {
        let mut command = command(root);
        command.args(["serve", "--stdio"]);
        return connect_stdio(command).await;
    }
    let address = format!(
        "127.0.0.1:{}",
        TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    );
    let mut child = command(root)
        .args(["serve", "--http", &address])
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    timeout(CALL, async {
        loop {
            assert!(
                child.try_wait().unwrap().is_none(),
                "HTTP server stopped during startup"
            );
            if tokio::net::TcpStream::connect(&address).await.is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let client = timeout(
        CALL,
        ().serve(StreamableHttpClientTransport::from_uri(format!(
            "http://{address}/mcp"
        ))),
    )
    .await
    .unwrap()
    .unwrap();
    (child, client)
}

async fn connect_stdio(mut command: Command) -> (Child, Client) {
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let client = timeout(
        CALL,
        ().serve((child.stdout.take().unwrap(), child.stdin.take().unwrap())),
    )
    .await
    .unwrap()
    .unwrap();
    (child, client)
}

// Execute the generated Codex arguments and cwd with the test build of dkc.
// The client starts elsewhere, so a missing/wrong cwd cannot pass by accident.
async fn connect_codex(root: &Path, client_root: &Path) -> (Child, Client) {
    let config: toml::Value =
        toml::from_str(&fs::read_to_string(root.join(".codex/config.toml")).unwrap()).unwrap();
    let server = &config["mcp_servers"]["deepkit-code"];
    assert_eq!(server["command"].as_str(), Some("dkc"));
    let cwd = Path::new(server["cwd"].as_str().unwrap());
    assert_eq!(cwd, root.canonicalize().unwrap());
    let args = server["args"]
        .as_array()
        .unwrap()
        .iter()
        .map(|arg| arg.as_str().unwrap());
    let mut command = command(client_root);
    command.current_dir(cwd).args(args);
    connect_stdio(command).await
}
async fn call(client: &Client, name: &str, args: Value) -> CallToolResult {
    timeout(
        CALL,
        client.call_tool(
            CallToolRequestParams::new(name.to_owned())
                .with_arguments(args.as_object().unwrap().clone()),
        ),
    )
    .await
    .unwrap()
    .unwrap()
}
fn success(result: CallToolResult) -> Value {
    assert_ne!(result.is_error, Some(true), "{result:?}");
    result.structured_content.unwrap_or_else(|| {
        let text = result
            .content
            .iter()
            .find_map(|part| part.as_text())
            .unwrap();
        serde_json::from_str(&text.text).unwrap()
    })
}
async fn search(client: &Client, query: &str) -> Value {
    success(
        call(
            client,
            "search",
            json!({"query":query,"mode":"lexical","rerank":false}),
        )
        .await,
    )
}
// Completed indexing does not remove the per-query deadline. Retry only the
// documented timeout envelope and retain every other failure with its response.
fn declared_query_timeout(value: &Value) -> bool {
    value["hits"].as_array().is_some_and(Vec::is_empty)
        && value["index"]["partial"] == true
        && value["index"]["degraded"] == true
        && value["index"]["reason"] == "query_timeout"
        && value["index"]["lexical"] == "incomplete"
}
async fn indexed_hit(client: &Client, query: &str, path: &str) {
    timeout(COMPLETE, async {
        loop {
            let response = search(client, query).await;
            let hits = response["hits"].as_array().expect("search hits array");
            if !hits.is_empty() {
                assert!(
                    hits.iter()
                        .any(|hit| hit["path"] == path && hit["fresh"]["state"] == "fresh"),
                    "expected verified {path}: {response}"
                );
                return;
            }
            assert!(
                declared_query_timeout(&response),
                "completed L1 produced an unexpected empty search: {response}"
            );
            eprintln!("post-bootstrap query deadline; retrying within readiness bound: {response}");
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("completed L1 never became searchable within readiness bound");
}
#[test]
fn only_explicit_query_timeout_is_retryable_during_readiness() {
    let timeout = json!({"hits":[],"index":{"partial":true,"degraded":true,"reason":"query_timeout","lexical":"incomplete"}});
    assert!(declared_query_timeout(&timeout));
    for bad in [
        json!({"hits":[],"index":{"partial":false}}),
        json!({"hits":[],"index":{"partial":true,"degraded":true,"reason":"source_changed","lexical":"incomplete"}}),
        json!({"hits":[],"index":{"partial":true,"degraded":true,"reason":"query_timeout","lexical":"complete"}}),
        json!({"hits":[],"index":{"partial":true,"degraded":false,"reason":"query_timeout","lexical":"incomplete"}}),
        json!({"hits":[],"index":{"partial":false,"degraded":true,"reason":"query_timeout","lexical":"incomplete"}}),
    ] {
        assert!(!declared_query_timeout(&bad), "{bad}");
    }
}

fn building(value: &Value) {
    assert_eq!(
        value["index"]["state"], "building",
        "initial work completed before observation or progress missing: {value}"
    );
    assert!(
        value["index"]["percentage"]
            .as_u64()
            .is_some_and(|number| number < 100),
        "{value}"
    );
}
async fn completed(client: &Client) -> Value {
    timeout(COMPLETE, async {
        loop {
            let status = success(call(client, "status", json!({})).await);
            match status["index"]["state"].as_str() {
                Some("complete") => {
                    assert_eq!(status["index"]["percentage"], 100);
                    assert_eq!(status["l1_pending_files"], 0);
                    assert!(status["l2_pending"].as_u64().is_some(), "{status}");
                    return status;
                }
                Some("building") => building(&status),
                _ => panic!("Initial L1 failed or lost its progress: {status}"),
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap()
}
async fn existing_index(client: &Client) {
    let status = success(call(client, "status", json!({})).await);
    // Progress belongs to this process's bootstrap attempt. Reopening a complete
    // index schedules no new attempt, so verify the persisted completion instead.
    assert!(
        status.get("index").is_none(),
        "unexpected bootstrap: {status}"
    );
    let completed = status["l1_completed_generation"]
        .as_str()
        .unwrap()
        .parse::<u64>()
        .unwrap();
    assert!(completed > 0, "{status}");
    assert_eq!(status["l1_scan_generation"], completed, "{status}");
    assert_eq!(status["l1_pending_files"], 0, "{status}");
    assert_eq!(status["graph_pending"], false, "{status}");
    assert!(
        status["units"].as_u64().is_some_and(|count| count > 0),
        "{status}"
    );
}
async fn stop(mut child: Child, client: Client) {
    // Both transports follow the same public graceful termination path.
    if child.try_wait().unwrap().is_none() {
        let pid = child.id().unwrap();
        let status = std::process::Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status()
            .unwrap();
        assert!(status.success());
    }
    client.cancel().await.unwrap();
    let status = timeout(CALL, child.wait()).await.unwrap().unwrap();
    assert!(status.success(), "serve did not stop normally: {status}");
}
fn owned_only(root: &Path) {
    let entries = fs::read_dir(root.join(".deepkit"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    assert!(
        entries.iter().all(|name| matches!(
            name.as_str(),
            "code.db" | "code.db-wal" | "code.db-shm" | "code.db-journal" | "code.db.l1-lock"
        )),
        "{entries:?}"
    );
    // T1.11 permits code.db* bookkeeping. T2.1 adds an empty coordination
    // sidecar so watching cannot supersede another process's L1 publication.
    let lock = fs::symlink_metadata(root.join(".deepkit/code.db.l1-lock")).unwrap();
    assert!(lock.is_file());
    assert_eq!(lock.len(), 0);
    for path in [
        ".deepkit/config.toml",
        ".deepkit/config.local.toml",
        ".cursor",
        ".vscode",
        ".gitignore",
        "absent-user-config",
        "absent-model-cache",
    ] {
        assert!(
            !root.join(path).exists(),
            "unexpected generated path: {path}"
        );
    }
    assert!(
        git(root, &["status", "--porcelain", "--untracked-files=all"]).is_empty(),
        "auto-init dirtied the worktree"
    );
    let excludes = fs::read_to_string(root.join(".git/info/exclude")).unwrap();
    assert_eq!(excludes.matches("# >>> deepkit-code >>>").count(), 1);
    assert_eq!(excludes.matches("# <<< deepkit-code <<<").count(), 1);
}

#[tokio::test]
async fn codex_stdio_keeps_concurrent_projects_and_restarts_isolated() {
    let _slot = TEST_LOCK.lock().await;
    let first = fixture(1);
    let second = fixture(1);
    let client_root = tempfile::tempdir().unwrap();
    for (root, name) in [
        (first.path(), "amberquartz"),
        (second.path(), "violetcobalt"),
    ] {
        fs::write(
            root.join("src/file_0000.rs"),
            format!("pub fn {name}() -> bool {{ true }}\n"),
        )
        .unwrap();
        cli(
            root,
            &["init", "--profile", "local", "--yes", "--codex", "--json"],
        )
        .await;
    }
    // Neither project has an index yet; their initial startup overlaps.
    let ((first_child, first_client), (second_child, second_client)) = tokio::join!(
        connect_codex(first.path(), client_root.path()),
        connect_codex(second.path(), client_root.path()),
    );
    tokio::join!(completed(&first_client), completed(&second_client));
    for (client, own, other) in [
        (&first_client, "amberquartz", "violetcobalt"),
        (&second_client, "violetcobalt", "amberquartz"),
    ] {
        indexed_hit(client, own, "src/file_0000.rs").await;
        let opened = success(call(client, "open", json!({"path":"src/file_0000.rs"})).await);
        let source = opened["text"].as_str().unwrap();
        assert!(source.contains(own), "{opened}");
        assert!(!source.contains(other), "{opened}");
    }
    stop(first_child, first_client).await;
    indexed_hit(&second_client, "violetcobalt", "src/file_0000.rs").await;
    let (first_child, first_client) = connect_codex(first.path(), client_root.path()).await;
    existing_index(&first_client).await;
    indexed_hit(&first_client, "amberquartz", "src/file_0000.rs").await;
    indexed_hit(&second_client, "violetcobalt", "src/file_0000.rs").await;
    tokio::join!(
        stop(first_child, first_client),
        stop(second_child, second_client)
    );
    assert!(!client_root.path().join(".deepkit").exists());
}

#[tokio::test]
async fn concurrent_stdio_clients_share_an_existing_project_and_stop_independently() {
    let _slot = TEST_LOCK.lock().await;
    let root = fixture(1);
    // This targets multiple clients of an established index, not concurrent schema creation.
    let (child, client) = connect(root.path(), false).await;
    completed(&client).await;
    stop(child, client).await;
    let ((first_child, first_client), (second_child, second_client)) =
        tokio::join!(connect(root.path(), false), connect(root.path(), false));
    tokio::join!(
        existing_index(&first_client),
        existing_index(&second_client)
    );
    indexed_hit(&first_client, "auto_probe_0_0", "src/file_0000.rs").await;
    indexed_hit(&second_client, "auto_probe_0_0", "src/file_0000.rs").await;
    stop(first_child, first_client).await;
    indexed_hit(&second_client, "auto_probe_0_0", "src/file_0000.rs").await;
    stop(second_child, second_client).await;
    owned_only(root.path());
}

#[tokio::test]
async fn both_transports_serve_during_initial_l1_without_repository_configuration() {
    let _slot = TEST_LOCK.lock().await;
    for http in [false, true] {
        let root = fixture(INITIAL_FILES);
        let started = Instant::now();
        let (child, client) = connect(root.path(), http).await;
        let first = search(&client, "auto_probe_0_0").await;
        assert!(first["hits"].is_array(), "{first}");
        building(&first);
        eprintln!(
            "auto-init transport={} first_response_ms={}",
            if http { "http" } else { "stdio" },
            started.elapsed().as_millis()
        );
        let status = completed(&client).await;
        assert_eq!(status["profile"]["primary"], "local");
        assert_eq!(status["models"]["llm"]["enabled"], false);
        indexed_hit(&client, "auto_probe_0_0", "src/file_0000.rs").await;
        stop(child, client).await;
        owned_only(root.path());
    }
}

#[tokio::test]
async fn observed_initial_l1_shutdown_resumes_on_both_transports() {
    let _slot = TEST_LOCK.lock().await;
    for http in [false, true] {
        let root = fixture(INITIAL_FILES);
        let (child, client) = connect(root.path(), http).await;
        let observed = search(&client, "auto_probe_0_0").await;
        building(&observed);
        // This stop follows an actual building response, not an assumed delay.
        stop(child, client).await;
        {
            let database = rusqlite::Connection::open_with_flags(
                root.path().join(".deepkit/code.db"),
                rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
            )
            .unwrap();
            let unfinished: bool = database.query_row(
                "SELECT CAST(COALESCE((SELECT value FROM meta WHERE key='l1_completed_generation'),'0') AS INTEGER)=0
                 OR (SELECT value FROM meta WHERE key='l1_completed_generation') != (SELECT value FROM meta WHERE key='l1_scan_generation')
                 OR EXISTS(SELECT 1 FROM files WHERE l1_pending=1)",
                [], |row| row.get(0),
            ).unwrap();
            assert!(
                unfinished,
                "L1 finished before shutdown; this run does not prove interrupted recovery"
            );
        }
        let (child, client) = connect(root.path(), http).await;
        let status = completed(&client).await;
        assert!(status["units"].as_u64().is_some_and(|count| count > 0));
        let recovered = search(&client, &format!("auto_probe_{}_0", INITIAL_FILES - 1)).await;
        assert!(
            !recovered["hits"].as_array().unwrap().is_empty(),
            "transport={http} completed={status} recovered={recovered}"
        );
        stop(child, client).await;
        owned_only(root.path());
    }
}

#[tokio::test]
async fn schema_only_database_without_auto_init_marker_is_recovered() {
    let _slot = TEST_LOCK.lock().await;
    let root = fixture(16);
    let status = cli(root.path(), &["status", "--json"]).await;
    assert_eq!(status["units"], 0);
    assert!(root.path().join(".deepkit/code.db").is_file());
    let (child, client) = connect(root.path(), false).await;
    let status = completed(&client).await;
    assert!(status["units"].as_u64().unwrap() > 0);
    assert!(
        !search(&client, "auto_probe_15_0").await["hits"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    stop(child, client).await;
    owned_only(root.path());
}

#[tokio::test]
async fn disabled_auto_init_and_unsafe_or_malformed_paths_make_no_outside_writes() {
    let _slot = TEST_LOCK.lock().await;
    for configured in [false, true] {
        let root = fixture(1);
        if configured {
            fs::create_dir(root.path().join(".deepkit")).unwrap();
            fs::write(
                root.path().join(".deepkit/config.toml"),
                "[serve]\nauto_init=false\n",
            )
            .unwrap();
        }
        let before = fs::read(root.path().join(".git/info/exclude")).unwrap();
        let mut cmd = command(root.path());
        if !configured {
            cmd.env("DKC_SERVE__AUTO_INIT", "false");
        }
        let output = timeout(
            CALL,
            cmd.args(["serve", "--stdio"]).stdin(Stdio::null()).output(),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(!output.status.success());
        assert!(!root.path().join(".deepkit/code.db").exists());
        if !configured {
            assert!(!root.path().join(".deepkit").exists());
        }
        assert_eq!(
            fs::read(root.path().join(".git/info/exclude")).unwrap(),
            before
        );
    }
    for kind in ["outside", "malformed", "symlink"] {
        let root = fixture(1);
        let outside = tempfile::tempdir().unwrap();
        let target = outside.path().join("database");
        fs::write(&target, b"outside must remain unchanged").unwrap();
        let before = fs::read(root.path().join(".git/info/exclude")).unwrap();
        let mut cmd = command(root.path());
        match kind {
            "outside" => {
                cmd.env("DKC_STORE__PATH", &target);
            }
            "malformed" => {
                fs::create_dir(root.path().join(".deepkit")).unwrap();
                fs::write(
                    root.path().join(".deepkit/code.db"),
                    b"not a sqlite database",
                )
                .unwrap();
            }
            _ => {
                fs::create_dir(root.path().join(".deepkit")).unwrap();
                std::os::unix::fs::symlink(&target, root.path().join(".deepkit/code.db")).unwrap();
            }
        }
        let output = timeout(
            CALL,
            cmd.args(["serve", "--stdio"]).stdin(Stdio::null()).output(),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(!output.status.success(), "{kind}");
        assert_eq!(fs::read(&target).unwrap(), b"outside must remain unchanged");
        assert_eq!(fs::read_dir(outside.path()).unwrap().count(), 1);
        assert_eq!(
            fs::read(root.path().join(".git/info/exclude")).unwrap(),
            before
        );
    }
}

#[tokio::test]
async fn pinned_ripgrep_first_building_response_is_under_five_seconds_when_corpus_exists() {
    let _slot = TEST_LOCK.lock().await;
    let corpus = env::var_os("DKC_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| "/data/dkc-corpus".into());
    let source = corpus.join("ripgrep");
    if !source.join(".git").exists() {
        eprintln!("SKIP: pinned ripgrep corpus absent; no T1.11 timing evidence measured");
        return;
    }
    let temporary = tempfile::tempdir().unwrap();
    let checkout = temporary.path().join("ripgrep");
    let output = std::process::Command::new("git")
        .args(["clone", "--shared", "--quiet"])
        .arg(&source)
        .arg(&checkout)
        .output()
        .unwrap();
    assert!(output.status.success(), "shared corpus clone failed");
    git(&checkout, &["checkout", "--detach", "--quiet", PIN]);
    assert_eq!(
        String::from_utf8(git(&checkout, &["rev-parse", "HEAD"]))
            .unwrap()
            .trim(),
        PIN
    );
    assert!(!checkout.join(".deepkit").exists());
    let started = Instant::now();
    let deadline = started + Duration::from_secs(5);
    let (child, client) = timeout_at(deadline, connect(&checkout, false))
        .await
        .unwrap();
    let first = timeout_at(deadline, search(&client, "ripgrep"))
        .await
        .unwrap();
    let first_ms = started.elapsed().as_millis();
    assert!(first["hits"].is_array(), "{first}");
    building(&first);
    assert!(first_ms < 5000, "first response took {first_ms}ms");
    let status = completed(&client).await;
    let completed_ms = started.elapsed().as_millis();
    assert!(status["units"].as_u64().unwrap() > 0);
    let useful = search(&client, "ripgrep").await;
    assert!(!useful["hits"].as_array().unwrap().is_empty());
    eprintln!(
        "T1.11 pinned_ripgrep first_response_ms={first_ms} completion_ms={completed_ms} first_hit_count={}",
        first["hits"].as_array().unwrap().len()
    );
    stop(child, client).await;
    // The pinned corpus already has .gitignore; auto-init must preserve it.
    assert!(!checkout.join(".deepkit/config.toml").exists());
    assert!(
        git(
            &checkout,
            &["status", "--porcelain", "--untracked-files=all"]
        )
        .is_empty()
    );
}
