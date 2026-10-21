//! Real watcher latency, batching and both-transport lifecycle; v1 §2.1/4.5.
#![cfg(unix)]
use rmcp::{
    RoleClient, ServiceExt, model::CallToolRequestParams, service::RunningService,
    transport::StreamableHttpClientTransport,
};
use serde_json::Value;
use std::{env, fs, net::TcpListener, path::Path, process::Stdio, time::Duration};
use tokio::{
    io::{AsyncBufReadExt, BufReader, Lines},
    process::{Child, ChildStdout, Command},
    time::{Instant, timeout},
};

const LATENCY: Duration = Duration::from_millis(1500);
const STARTUP: Duration = Duration::from_secs(15);
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
type Client = RunningService<RoleClient, ()>;
type Output = Lines<BufReader<ChildStdout>>;

fn command(root: &Path) -> Command {
    let mut command = Command::new(assert_cmd::cargo::cargo_bin!("dkc"));
    for (name, _) in env::vars_os() {
        if name.to_string_lossy().starts_with("DKC_") {
            command.env_remove(name);
        }
    }
    command
        .current_dir(root)
        .env("XDG_CONFIG_HOME", root.join(".deepkit/absent-config"))
        .env("DKC_MODELS_DIR", root.join(".deepkit/absent-models"))
        .env_remove("MOONLY_GATEWAY_API_KEY")
        .stderr(Stdio::inherit())
        .kill_on_drop(true);
    command
}

async fn cli(root: &Path, args: &[&str]) -> Value {
    let output = timeout(STARTUP, command(root).args(args).output())
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

async fn fixture() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join(".deepkit")).unwrap();
    fs::write(
        root.path().join(".deepkit/config.toml"),
        "[embedder]\nprofile='local'\n[llm]\nenabled=false\n[rerank]\nmode='off'\n",
    )
    .unwrap();
    fs::write(root.path().join("base.rs"), "pub fn base() {}\n").unwrap();
    fs::write(root.path().join(".deepkitignore"), "ignored.rs\n").unwrap();
    cli(root.path(), &["index", "--level", "1", "--json"]).await;
    root
}

fn rows(root: &Path) -> Vec<(i64, String, Vec<u8>)> {
    let connection = rusqlite::Connection::open_with_flags(
        root.join(".deepkit/code.db"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .unwrap();
    connection.busy_timeout(Duration::from_millis(100)).unwrap();
    connection
        .prepare("SELECT id,path,blake3 FROM files WHERE l1_pending=0 ORDER BY path")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap()
}

async fn wait_for(root: &Path, predicate: impl Fn(&[(i64, String, Vec<u8>)]) -> bool) {
    timeout(LATENCY, async {
        loop {
            if predicate(&rows(root)) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("watcher did not publish within 1.5 seconds");
}

async fn line(output: &mut Output) -> Value {
    let line = timeout(STARTUP, output.next_line())
        .await
        .unwrap()
        .unwrap()
        .expect("watch exited");
    serde_json::from_str(&line).unwrap()
}

async fn terminate(child: &mut Child) {
    let id = child.id().unwrap().to_string();
    assert!(
        std::process::Command::new("kill")
            .args(["-TERM", &id])
            .status()
            .unwrap()
            .success()
    );
    let status = timeout(Duration::from_secs(3), child.wait())
        .await
        .unwrap()
        .unwrap();
    assert!(status.success(), "{status}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_native_and_polling_publish_changes_coalesce_bursts_and_exclude_self_writes() {
    let _serial = SERIAL.lock().await;
    for poll in [false, true] {
        let root = fixture().await;
        let mut cmd = command(root.path());
        cmd.args(["watch", "--json"]).stdout(Stdio::piped());
        if poll {
            cmd.arg("--poll");
        }
        let mut child = cmd.spawn().unwrap();
        let mut output = BufReader::new(child.stdout.take().unwrap()).lines();
        let ready = line(&mut output).await;
        assert_eq!(ready["watch"]["state"], "watching");
        assert_eq!(
            ready["watch"]["backend"],
            if poll { "polling" } else { "native" }
        );
        // A literal backslash is a Unix filename byte, not a path separator.
        let source = root.path().join("created\\name.rs");
        let start = Instant::now();
        fs::write(&source, "pub fn created() {}\n").unwrap();
        wait_for(root.path(), |rows| {
            rows.iter().any(|r| r.1 == "created\\name.rs")
        })
        .await;
        let elapsed = start.elapsed();
        assert!(elapsed < LATENCY, "{elapsed:?}");
        let created = line(&mut output).await;
        assert_eq!(created["watch"]["added"], 1);
        assert_eq!(cli(root.path(), &["status", "--json"]).await["files"], 2);
        println!(
            "watch poll={poll} create_ms={}",
            elapsed.as_secs_f64() * 1000.0
        );

        let text = b"pub fn modified() {}\n";
        let digest = blake3::hash(text);
        let start = Instant::now();
        fs::write(&source, text).unwrap();
        wait_for(root.path(), |rows| {
            rows.iter()
                .any(|r| r.1 == "created\\name.rs" && r.2.as_slice() == digest.as_bytes())
        })
        .await;
        let elapsed = start.elapsed();
        assert!(elapsed < LATENCY, "{elapsed:?}");
        assert_eq!(line(&mut output).await["watch"]["modified"], 1);
        println!(
            "watch poll={poll} modify_ms={}",
            elapsed.as_secs_f64() * 1000.0
        );

        let id = rows(root.path())
            .into_iter()
            .find(|r| r.1 == "created\\name.rs")
            .unwrap()
            .0;
        let start = Instant::now();
        fs::rename(&source, root.path().join("renamed.rs")).unwrap();
        wait_for(root.path(), |rows| {
            rows.iter().any(|r| r.0 == id && r.1 == "renamed.rs")
                && !rows.iter().any(|r| r.1 == "created\\name.rs")
        })
        .await;
        let elapsed = start.elapsed();
        assert!(elapsed < LATENCY, "{elapsed:?}");
        assert_eq!(line(&mut output).await["watch"]["renamed"], 1);
        println!(
            "watch poll={poll} rename_ms={}",
            elapsed.as_secs_f64() * 1000.0
        );

        let start = Instant::now();
        fs::remove_file(root.path().join("renamed.rs")).unwrap();
        wait_for(root.path(), |rows| !rows.iter().any(|r| r.0 == id)).await;
        let elapsed = start.elapsed();
        assert!(elapsed < LATENCY, "{elapsed:?}");
        assert_eq!(line(&mut output).await["watch"]["deleted"], 1);
        assert_eq!(cli(root.path(), &["status", "--json"]).await["files"], 1);
        println!(
            "watch poll={poll} delete_ms={}",
            elapsed.as_secs_f64() * 1000.0
        );

        let before = cli(root.path(), &["status", "--json"]).await["l1_scan_generation"].clone();
        for n in 0..1000 {
            fs::write(root.path().join(".deepkit/watch-noise"), n.to_string()).unwrap();
        }
        assert!(
            timeout(Duration::from_millis(1100), output.next_line())
                .await
                .is_err(),
            "self-write caused an index pass"
        );
        assert_eq!(
            cli(root.path(), &["status", "--json"]).await["l1_scan_generation"],
            before
        );

        let start = Instant::now();
        for n in 0..1000 {
            fs::write(
                root.path().join("base.rs"),
                format!("pub fn burst_{n}() {{}}\n"),
            )
            .unwrap();
        }
        let writes_elapsed = start.elapsed();
        let final_hash = blake3::hash(b"pub fn burst_999() {}\n");
        wait_for(root.path(), |rows| {
            rows.iter()
                .any(|r| r.1 == "base.rs" && r.2.as_slice() == final_hash.as_bytes())
        })
        .await;
        let publication_elapsed = start.elapsed();
        let batch = line(&mut output).await;
        assert_eq!(batch["watch"]["state"], "updated", "{batch}");
        assert_eq!(batch["watch"]["scanned_files"], 1);
        assert_eq!(batch["watch"]["modified"], 1);
        assert_eq!(batch["watch"]["full_scan"], false);
        let extra = timeout(Duration::from_millis(1100), output.next_line()).await;
        assert!(
            extra.is_err(),
            "burst caused more than one rescan: writes={writes_elapsed:?}, publication={publication_elapsed:?}, first={batch}, extra={extra:?}"
        );
        println!(
            "watch poll={poll} burst_writes=1000 rescans=1 publication_ms={}",
            publication_elapsed.as_secs_f64() * 1000.0
        );
        terminate(&mut child).await;
    }
}

async fn connect(root: &Path, http: bool) -> (Child, Client) {
    if !http {
        let mut child = command(root)
            .args(["serve", "--stdio"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let client = timeout(
            STARTUP,
            ().serve((child.stdout.take().unwrap(), child.stdin.take().unwrap())),
        )
        .await
        .unwrap()
        .unwrap();
        return (child, client);
    }
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let mut child = command(root)
        .args(["serve", "--http", &address.to_string()])
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    timeout(STARTUP, async {
        loop {
            assert!(child.try_wait().unwrap().is_none());
            if tokio::net::TcpStream::connect(address).await.is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let client = timeout(
        STARTUP,
        ().serve(StreamableHttpClientTransport::from_uri(format!(
            "http://{address}/mcp"
        ))),
    )
    .await
    .unwrap()
    .unwrap();
    (child, client)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn both_mcp_transports_watch_without_a_search_or_repair_request() {
    let _serial = SERIAL.lock().await;
    for http in [false, true] {
        let root = fixture().await;
        let (mut child, client) = connect(root.path(), http).await;
        fs::write(
            root.path().join("automatic.rs"),
            "pub fn automatically_discovered() {}\n",
        )
        .unwrap();
        // Observe SQLite before any tool request can perform lazy repair.
        wait_for(root.path(), |rows| {
            rows.iter().any(|r| r.1 == "automatic.rs")
        })
        .await;
        let status = timeout(
            STARTUP,
            client.call_tool(CallToolRequestParams::new("status")),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(status.structured_content.as_ref().unwrap()["files"], 2);
        fs::remove_file(root.path().join("automatic.rs")).unwrap();
        wait_for(root.path(), |rows| {
            !rows.iter().any(|r| r.1 == "automatic.rs")
        })
        .await;
        let status = timeout(
            STARTUP,
            client.call_tool(CallToolRequestParams::new("status")),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(status.structured_content.as_ref().unwrap()["files"], 1);
        terminate(&mut child).await;
        let _ = client.cancel().await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_retains_edits_across_invalid_configuration_and_retargets_an_existing_root() {
    let _serial = SERIAL.lock().await;
    let first = fixture().await;
    let second = fixture().await;
    let mut child = command(first.path())
        .args(["watch", "--json"])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut output = BufReader::new(child.stdout.take().unwrap()).lines();
    assert_eq!(line(&mut output).await["watch"]["state"], "watching");
    let config = first.path().join(".deepkit/config.local.toml");
    fs::write(&config, "[invalid").unwrap();
    fs::write(first.path().join("base.rs"), "pub fn retained_edit() {}\n").unwrap();
    assert_eq!(
        line(&mut output).await["watch"]["state"],
        "waiting_configuration"
    );
    fs::remove_file(&config).unwrap();
    // No second source event: restoration itself must release retained work.
    let restored = line(&mut output).await;
    assert_eq!(restored["watch"]["state"], "updated", "{restored}");
    assert_eq!(restored["watch"]["full_scan"], true);
    let digest = blake3::hash(b"pub fn retained_edit() {}\n");
    assert!(
        rows(first.path())
            .iter()
            .any(|r| r.1 == "base.rs" && r.2.as_slice() == digest.as_bytes())
    );
    // Drain a possible already-debounced duplicate of the retained edit.
    if let Ok(extra) = timeout(Duration::from_millis(750), output.next_line()).await {
        let extra: Value = serde_json::from_str(&extra.unwrap().expect("watch exited")).unwrap();
        assert_eq!(extra["watch"]["state"], "updated");
        assert!(
            timeout(Duration::from_millis(750), output.next_line())
                .await
                .is_err(),
            "edit was not coalesced"
        );
    }
    let original = rows(first.path());
    fs::write(
        &config,
        format!("[index]\nroot={:?}\n", second.path().to_string_lossy()),
    )
    .unwrap();
    assert_eq!(line(&mut output).await["watch"]["state"], "watching");
    assert_eq!(line(&mut output).await["watch"]["state"], "updated");
    fs::write(first.path().join("old-root.rs"), "pub fn old_root() {}\n").unwrap();
    fs::write(second.path().join("new-root.rs"), "pub fn new_root() {}\n").unwrap();
    wait_for(second.path(), |rows| {
        rows.iter().any(|r| r.1 == "new-root.rs")
    })
    .await;
    assert_eq!(line(&mut output).await["watch"]["state"], "updated");
    assert_eq!(rows(first.path()), original);
    terminate(&mut child).await;
}
