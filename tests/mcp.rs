//! Real CLI MCP transports and tool contracts; v1 §10 and phase-1 T1.9.
#[path = "mcp/queue_wire.rs"]
mod queue_wire;
#[path = "mcp/response_caps.rs"]
mod response_caps;
#[path = "mcp/summary_serving.rs"]
mod summary_serving;
use rmcp::{
    RoleClient, ServiceExt,
    model::{CallToolRequestParams, CallToolResult, ErrorCode},
    service::{RunningService, ServiceError},
    transport::{
        StreamableHttpClientTransport, streamable_http_client::StreamableHttpClientTransportConfig,
    },
};
use serde_json::{Value, json};
use std::{fs, net::TcpListener, path::Path, process::Stdio, time::Duration};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    process::{Child, Command},
    time::{Instant, timeout, timeout_at},
};

const CALL_TIMEOUT: Duration = Duration::from_secs(10);
type Client = RunningService<RoleClient, ()>;

fn fixture(writable: bool) -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join("src")).unwrap();
    fs::write(root.path().join("src/lib.rs"), "pub fn should_retry(attempt: u32) -> bool { attempt < 3 }\npub fn worker() -> bool { should_retry(1) }\n").unwrap();
    fs::create_dir(root.path().join("alt")).unwrap();
    for path in ["src/scope.rs", "alt/scope.rs"] {
        fs::write(
            root.path().join(path),
            "pub fn scope_probe() -> bool { true }\n",
        )
        .unwrap();
    }
    let long = format!(
        "// before\npub fn expanded() {{\n{}    // end of expanded\n}}\n// after\n",
        "    let value = 1;\n".repeat(250)
    );
    fs::write(root.path().join("src/expanded.rs"), long).unwrap();
    fs::write(
        root.path().join("guide.md"),
        format!("# Guide\n\n{}", "Read café 🙂 carefully.\r\n".repeat(60)),
    )
    .unwrap();
    fs::create_dir(root.path().join(".deepkit")).unwrap();
    fs::write(root.path().join(".deepkit/config.toml"), format!("[embedder]\nprofile='local'\n[rerank]\nmode='off'\n[serve]\nagent_writable_config={writable}\n")).unwrap();
    for args in [
        vec!["init", "--quiet"],
        vec!["add", "src", "alt"],
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
    let mut command = Command::new(assert_cmd::cargo::cargo_bin!("dkc"));
    command
        .current_dir(root)
        .env("XDG_CONFIG_HOME", root.join("user-config"))
        .env("DKC_MODELS_DIR", root.join("absent-model-cache"))
        .env_remove("DKC_SERVE__HTTP_TOKEN")
        .env("DKC_MCP_CHAT_FIXTURE_KEY", "fixture")
        .env("DKC_LOG", "warn,dkc::search::service=debug")
        .stderr(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(root.join(".deepkit/mcp-server.stderr"))
                .unwrap(),
        )
        .kill_on_drop(true);
    command
}

async fn stdio(root: &Path) -> (Child, Client) {
    let mut child = command(root)
        .args(["serve", "--stdio"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let transport = (child.stdout.take().unwrap(), child.stdin.take().unwrap());
    let client = timeout(CALL_TIMEOUT, ().serve(transport))
        .await
        .unwrap()
        .unwrap();
    (child, client)
}

async fn call(client: &Client, name: &'static str, arguments: Value) -> CallToolResult {
    timeout(
        CALL_TIMEOUT,
        client.call_tool(
            CallToolRequestParams::new(name).with_arguments(arguments.as_object().unwrap().clone()),
        ),
    )
    .await
    .unwrap()
    .unwrap()
}

fn payload(result: &CallToolResult) -> Value {
    result.structured_content.clone().unwrap_or_else(|| {
        let text = result
            .content
            .iter()
            .find_map(|content| content.as_text())
            .unwrap();
        serde_json::from_str(&text.text).unwrap()
    })
}

fn success(result: &CallToolResult) -> Value {
    assert_ne!(result.is_error, Some(true), "{result:?}");
    assert!(
        result
            .meta
            .as_ref()
            .unwrap()
            .get("took_ms")
            .unwrap()
            .as_f64()
            .unwrap()
            .is_finite()
    );
    payload(result)
}

fn failure(result: &CallToolResult) -> Value {
    assert_eq!(result.is_error, Some(true), "{result:?}");
    let data = payload(result);
    for field in ["code", "message", "hint"] {
        assert!(
            data["error"][field].is_string(),
            "missing error {field}: {data}"
        );
    }
    data
}

/// Waits for automatic L1 publication without issuing a repairing tool call.
async fn wait_for_watcher_sources(root: &Path, paths: &[&str]) {
    use rusqlite::OptionalExtension;
    let expected = paths
        .iter()
        .map(|path| {
            let bytes = match fs::read(root.join(path)) {
                Ok(bytes) => Some(blake3::hash(&bytes)),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => panic!("{path}: {error}"),
            };
            (*path, bytes)
        })
        .collect::<Vec<_>>();
    let db = rusqlite::Connection::open_with_flags(
        root.join(".deepkit/code.db"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .unwrap();
    db.busy_timeout(CALL_TIMEOUT).unwrap();
    let mut last_observed = Vec::new();
    let mut last_graph = String::new();
    timeout(CALL_TIMEOUT, async {
        loop {
            let mut ready = true;
            last_observed.clear();
            for (path, expected) in &expected {
                let observed: Option<(Vec<u8>, i64)> = db
                    .query_row(
                        "SELECT blake3,l1_pending FROM files WHERE path=?1",
                        [path],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    )
                    .optional()
                    .unwrap();
                last_observed.push((*path, observed.clone()));
                ready &= match (expected, observed) {
                    (None, None) => true,
                    (Some(hash), Some((bytes, 0))) => hash.as_bytes().as_slice() == bytes,
                    _ => false,
                };
            }
            let graph: String = db
                .query_row(
                    "SELECT value FROM meta WHERE key='graph_pending'",
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            last_graph.clone_from(&graph);
            if ready && graph == "0" {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|error| {
        let stderr = fs::read_to_string(root.join(".deepkit/mcp-server.stderr"))
            .unwrap_or_default();
        panic!(
            "watcher publication {error}: expected={expected:?}, observed={last_observed:?}, graph={last_graph:?}\nserver stderr:\n{stderr}"
        );
    });
}

/// Observes completed checkout publication without giving a tool the repair work.
async fn wait_for_watcher_head(root: &Path, head: &str) {
    // An exclude-file notification requests a full watcher scan. Ordinary path
    // batches publish files but deliberately do not advance indexed_head.
    let exclude = root.join(".git/info/exclude");
    let mut excludes = fs::read_to_string(&exclude).unwrap();
    excludes.push('\n');
    fs::write(exclude, excludes).unwrap();
    wait_for_watcher_sources(
        root,
        &["src/added.rs", "src/moved.rs", "src/lib.rs", "alt/scope.rs"],
    )
    .await;
    let db = rusqlite::Connection::open_with_flags(
        root.join(".deepkit/code.db"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .unwrap();
    db.busy_timeout(CALL_TIMEOUT).unwrap();
    timeout(CALL_TIMEOUT, async {
        loop {
            let indexed: String = db
                .query_row("SELECT value FROM meta WHERE key='indexed_head'", [], |r| {
                    r.get(0)
                })
                .unwrap();
            if indexed == head {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
}

fn assert_mcp_branch_schedule(result: &Value, watch_first: bool) {
    let state = result["index"]["branch_resync"]["state"].as_str().unwrap();
    if watch_first {
        assert_eq!(state, "not_needed", "{result}");
    } else {
        // The request and automatic watcher share publication admission. Either
        // may finish first; the caller still checks the exact HEAD and contents.
        assert!(matches!(state, "completed" | "not_needed"), "{result}");
    }
}

fn assert_catalogue_absent(root: &Path, path: &str) {
    let db = rusqlite::Connection::open_with_flags(
        root.join(".deepkit/code.db"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .unwrap();
    let count: i64 = db
        .query_row("SELECT count(*) FROM files WHERE path=?1", [path], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(
        count, 0,
        "completed absence needs a reconciled catalogue: {path}"
    );
}

fn assert_at_most_one_query_repair(index: &Value, watch_first: bool) {
    let published = index["lazy_resync"]["published_files"].as_u64().unwrap();
    assert!(published <= 1, "{index}");
    if watch_first {
        assert_eq!(published, 0, "watcher already completed L1: {index}");
    }
}

async fn exercise_tools(client: &Client, writable: bool, root: &Path) {
    let tools = timeout(CALL_TIMEOUT, client.list_all_tools())
        .await
        .unwrap()
        .unwrap();
    let mut names = tools
        .iter()
        .map(|tool| tool.name.as_ref())
        .collect::<Vec<_>>();
    names.sort_unstable();
    assert_eq!(
        names,
        [
            "config_explain",
            "config_get",
            "config_set",
            "map",
            "open",
            "recent",
            "refs",
            "search",
            "status",
            "symbol"
        ]
    );
    for tool in &tools {
        assert!(!tool.description.as_deref().unwrap().trim().is_empty());
        assert_eq!(tool.input_schema.get("type"), Some(&json!("object")));
        let properties = tool
            .input_schema
            .get("properties")
            .and_then(Value::as_object);
        for key in match tool.name.as_ref() {
            "search" => &["query", "mode", "scope"][..],
            "open" => &["id", "path", "lines"][..],
            "symbol" | "refs" => &["name", "id"][..],
            "map" => &["budget_tokens", "focus"][..],
            "config_set" => &["key", "value", "layer"][..],
            "config_get" | "config_explain" => &["key"][..],
            _ => &[],
        } {
            assert!(
                properties.unwrap().contains_key(*key),
                "{} missing {key}",
                tool.name
            );
        }
    }
    let search = success(&call(client, "search", json!({"query":"should_retry", "k":5})).await);
    let hits = search["hits"].as_array().unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0]["path"], "src/lib.rs");
    assert!(
        search["index"]["degraded"].as_bool().unwrap(),
        "default hybrid must disclose unavailable dense leg: {search}"
    );
    let union = success(&call(client, "search", json!({"query":"scope_probe", "mode":"lexical", "scope":["src/*", "alt/*", "src/scope.rs"]})).await);
    let union_paths = union["hits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["path"].as_str().unwrap())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        union_paths,
        std::collections::BTreeSet::from(["src/scope.rs", "alt/scope.rs"])
    );
    let scoped = success(&call(client, "search", json!({"query":"should_retry", "mode":"lexical", "scope":["missing/*", "src/*", "src/lib.rs"], "k":5})).await);
    let scoped_hits = scoped["hits"].as_array().unwrap();
    assert!(!scoped_hits.is_empty());
    let ids = scoped_hits
        .iter()
        .map(|h| h["id"].as_str().unwrap())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(ids.len(), scoped_hits.len());
    for scope in [json!([]), json!(["missing/*"]), json!(["code"])] {
        let value = success(
            &call(
                client,
                "search",
                json!({"query":"should_retry", "mode":"lexical", "scope":scope}),
            )
            .await,
        );
        assert!(value["hits"].as_array().unwrap().is_empty());
    }
    let excluded = success(&call(client, "search", json!({"query":"should_retry", "mode":"lexical", "scope":["src/*", "missing/*"], "exclude_ids": ids})).await);
    assert!(excluded["hits"].as_array().unwrap().is_empty());
    let open = success(&call(client, "open", json!({"id": hits[0]["id"]})).await);
    assert_eq!(open["path"], "src/lib.rs");
    let repo_name = root.file_name().unwrap().to_str().unwrap();
    assert_eq!(open["breadcrumb"], format!("{repo_name} › src › lib.rs"));
    assert!(open["text"].as_str().unwrap().contains("should_retry"));
    let range = success(&call(client, "open", json!({"path":"src/lib.rs", "lines":[1,1]})).await);
    assert!(range["text"].as_str().unwrap().contains("should_retry"));
    let expanded = success(
        &call(
            client,
            "open",
            json!({"path":"src/expanded.rs", "lines":[125,125]}),
        )
        .await,
    );
    assert_eq!(expanded["lines"], json!([2, 254]));
    assert!(
        expanded["text"]
            .as_str()
            .unwrap()
            .starts_with("pub fn expanded()")
    );
    assert!(expanded["text"].as_str().unwrap().ends_with("}\n"));
    assert_eq!(expanded["truncated"], false);
    let surrounded = success(
        &call(
            client,
            "open",
            json!({"path":"src/expanded.rs", "lines":[125,125], "context":1}),
        )
        .await,
    );
    assert_eq!(surrounded["lines"], json!([1, 255]));
    assert!(surrounded["text"].as_str().unwrap().ends_with("// after\n"));
    let guide = success(&call(client, "open", json!({"path":"guide.md"})).await);
    assert_eq!(guide["lines"], json!([null, null]));
    assert_eq!(
        guide["breadcrumb"],
        format!("{repo_name} › guide.md › Guide")
    );
    let local_config = root.join(".deepkit/config.local.toml");
    let saved_config = fs::read(&local_config).ok();
    fs::write(&local_config, "[limits]\nmax_open_bytes='513 B'\n").unwrap();
    let mut page = success(
        &call(
            client,
            "open",
            json!({"path":"src/expanded.rs", "lines":[125,125], "context":1}),
        )
        .await,
    );
    // A fresh call owns its own deadline, including continuation requests.
    let saved_page_config = fs::read(&local_config).unwrap();
    fs::write(
        &local_config,
        "[limits]\nmax_open_bytes='513 B'\nquery_timeout_ms=0\n",
    )
    .unwrap();
    for args in [
        json!({"path":"src/expanded.rs"}),
        json!({"id":page["continuation_id"]}),
    ] {
        assert_eq!(
            failure(&call(client, "open", args).await)["error"]["code"],
            "timeout"
        );
    }
    fs::write(&local_config, saved_page_config).unwrap();
    let mut reconstructed = String::new();
    let first_token = page["continuation_id"].as_str().unwrap().to_owned();
    loop {
        assert_eq!(page["byte_range"][0], reconstructed.len());
        let text = page["text"].as_str().unwrap();
        assert!(!text.is_empty() && text.len() <= 513);
        reconstructed.push_str(text);
        assert_eq!(page["byte_range"][1], reconstructed.len());
        let Some(id) = page["continuation_id"].as_str() else {
            break;
        };
        page = success(&call(client, "open", json!({"id":id})).await);
    }
    assert_eq!(reconstructed, surrounded["text"].as_str().unwrap());
    let mut doc_page = success(&call(client, "open", json!({"path":"guide.md"})).await);
    let doc_token = doc_page["continuation_id"].as_str().unwrap().to_owned();
    let mut doc_text = String::new();
    loop {
        assert_eq!(doc_page["lines"], json!([null, null]));
        assert_eq!(doc_page["breadcrumb"], guide["breadcrumb"]);
        assert_eq!(doc_page["indexed_lines"], guide["indexed_lines"]);
        let text = doc_page["text"].as_str().unwrap();
        assert!(!text.is_empty() && text.len() <= 513);
        let skipped = doc_text.bytes().filter(|b| *b == b'\n').count();
        assert_eq!(
            doc_page["extracted_lines"],
            json!([1 + skipped, skipped + text.split_inclusive('\n').count()])
        );
        assert_eq!(doc_page["byte_range"][0], doc_text.len());
        doc_text.push_str(text);
        assert_eq!(doc_page["byte_range"][1], doc_text.len());
        let Some(id) = doc_page["continuation_id"].as_str() else {
            break;
        };
        doc_page = success(&call(client, "open", json!({"id":id})).await);
    }
    assert_eq!(doc_text, guide["text"]);
    let original_doc = fs::read(root.join("guide.md")).unwrap();
    fs::write(root.join("guide.md"), "changed").unwrap();
    for target in [guide["id"].as_str().unwrap(), &doc_token] {
        let error = failure(&call(client, "open", json!({"id":target})).await);
        assert!(matches!(
            error["error"]["code"].as_str(),
            Some("source_unavailable" | "not_found")
        ));
    }
    fs::remove_file(root.join("guide.md")).unwrap();
    failure(&call(client, "open", json!({"id":doc_token})).await);
    fs::write(root.join("guide.md"), original_doc).unwrap();
    match saved_config {
        Some(bytes) => fs::write(&local_config, bytes).unwrap(),
        None => fs::remove_file(&local_config).unwrap(),
    }
    let source_path = root.join("src/expanded.rs");
    let original = fs::read_to_string(&source_path).unwrap();
    fs::write(&source_path, original.replace("let value", "let other")).unwrap();
    let repaired = success(
        &call(
            client,
            "open",
            json!({"path":"src/expanded.rs", "lines":[125,125]}),
        )
        .await,
    );
    assert_eq!(
        failure(&call(client, "open", json!({"id":first_token})).await)["error"]["code"],
        "source_unavailable"
    );
    assert!(repaired["text"].as_str().unwrap().contains("let other"));
    assert!(!repaired["text"].as_str().unwrap().contains("let value"));
    fs::remove_file(&source_path).unwrap();
    // Make the catalogue state deterministic across both transports. The
    // watcher-free Context test covers source_unavailable before reconciliation.
    wait_for_watcher_sources(root, &["src/expanded.rs"]).await;
    let missing = failure(
        &call(
            client,
            "open",
            json!({"path":"src/expanded.rs", "context":1}),
        )
        .await,
    );
    assert_eq!(missing["error"]["code"], "not_found");
    assert_catalogue_absent(root, "src/expanded.rs");
    fs::write(&source_path, original).unwrap();
    // Each transport keeps the same server alive across source mutations.
    let scope_path = root.join("src/scope.rs");
    let scope_original = fs::read(&scope_path).unwrap();
    for (watch_first, mutation) in [false, true].into_iter().flat_map(|watch_first| {
        ["edit", "delete", "rename", "checkout"].map(move |mutation| (watch_first, mutation))
    }) {
        match mutation {
            "edit" => {
                fs::write(&scope_path, "\n\npub fn scope_probe() -> bool { false }\n").unwrap()
            }
            "delete" => fs::remove_file(&scope_path).unwrap(),
            "rename" => fs::rename(&scope_path, root.join("src/moved.rs")).unwrap(),
            "checkout" => {
                fs::write(&scope_path, "// replaced\n").unwrap();
                assert!(
                    std::process::Command::new("git")
                        .current_dir(root)
                        .args(["checkout", "--", "src/scope.rs"])
                        .status()
                        .unwrap()
                        .success()
                );
            }
            _ => unreachable!(),
        }
        // Exercise immediate calls and the deterministic watcher-first schedule.
        // Exact query-owned repair is covered by CLI/Context tests without a watcher.
        if watch_first {
            wait_for_watcher_sources(root, &["src/scope.rs", "src/moved.rs"]).await;
        }
        let result = success(
            &call(
                client,
                "search",
                json!({"query":"scope_probe",
            "mode":"lexical", "scope":["src/scope.rs"]}),
            )
            .await,
        );
        if matches!(mutation, "checkout" | "edit") {
            if mutation == "edit" {
                assert_eq!(result["hits"][0]["lines"][0], 3);
                assert_at_most_one_query_repair(&result["index"], watch_first);
            }
            assert_eq!(result["hits"].as_array().unwrap().len(), 1);
            assert_eq!(result["hits"][0]["fresh"]["state"], "fresh");
            assert!(result["hits"][0]["fresh"]["changed_ago_s"].is_u64());
            assert_eq!(result["hits"][0]["fresh"]["l2"], "pending");
            assert_eq!(result["index"]["partial"], false);
        } else {
            assert_eq!(result["hits"], json!([]));
            if result["index"]["partial"] == true {
                assert!(!watch_first, "{result}");
                assert_eq!(result["index"]["suppressed_hits"], 1);
            } else {
                assert_eq!(result["index"]["partial"], false);
                assert_eq!(result["index"]["suppressed_hits"], 0);
                assert_catalogue_absent(root, "src/scope.rs");
            }
        }
        if mutation == "rename" {
            fs::remove_file(root.join("src/moved.rs")).unwrap();
        }
        fs::write(&scope_path, &scope_original).unwrap();
        wait_for_watcher_sources(root, &["src/scope.rs", "src/moved.rs"]).await;
    }
    // A zero deadline deterministically exercises partial output over both
    // transports without machine-load assumptions or unverified locations.
    let config_path = root.join(".deepkit/config.toml");
    let original_config = fs::read_to_string(&config_path).unwrap();
    for protected in [true, false] {
        let config = if protected {
            format!(
                "{original_config}\n[[treatments]]\nname='protected_scope'\nmatch=['src/scope.rs']\nmax_level=1\nnever_send=true\n"
            )
        } else {
            original_config.clone()
        };
        fs::write(&config_path, config).unwrap();
        let output = std::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
            .current_dir(root)
            .env("XDG_CONFIG_HOME", root.join("user-config"))
            .args(["index", "--level", "1", "--json"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let result = success(
            &call(
                client,
                "search",
                json!({
                    "query":"scope_probe", "mode":"lexical", "scope":["src/scope.rs"]
                }),
            )
            .await,
        );
        assert_eq!(result["hits"].as_array().unwrap().len(), 1, "{result}");
        assert_eq!(
            result["hits"][0]["fresh"]["l2"],
            if protected { "none" } else { "pending" }
        );
        assert_eq!(result["index"]["partial"], false);
    }
    fs::write(
        &config_path,
        format!("{original_config}\n[limits]\nquery_timeout_ms=0\n"),
    )
    .unwrap();
    let timed_out = success(&call(client, "search", json!({"query":"should_retry"})).await);
    assert_eq!(timed_out["hits"], json!([]));
    assert_eq!(timed_out["index"]["partial"], true);
    assert_eq!(timed_out["index"]["reason"], "query_timeout");
    assert_eq!(timed_out["index"]["fresh"]["state"], "unknown");
    assert!(timed_out["index"]["head"].is_null());
    assert!(timed_out["index"]["suppressed_hits"].is_null());
    let timed_refs = success(&call(client, "refs", json!({"name":"should_retry"})).await);
    assert_eq!(timed_refs["groups"], json!([]));
    assert_eq!(timed_refs["index"]["reason"], "query_timeout");
    assert!(timed_refs["total"].is_null());
    let timed_map = success(&call(client, "map", json!({})).await);
    assert_eq!(timed_map["tree"], "");
    assert_eq!(timed_map["nodes"], json!([]));
    assert_eq!(timed_map["index"]["reason"], "query_timeout");
    let timed_symbol = failure(&call(client, "symbol", json!({"name":"should_retry"})).await);
    assert_eq!(timed_symbol["error"]["code"], "timeout");
    let timed_open = failure(&call(client, "open", json!({"path":"src/lib.rs"})).await);
    assert_eq!(timed_open["error"]["code"], "timeout");
    fs::write(&config_path, original_config).unwrap();
    let symbols = success(&call(client, "symbol", json!({"name":"should_retry"})).await);
    let symbols = symbols.as_array().unwrap();
    assert!(!symbols.is_empty());
    assert_eq!(symbols[0]["path"], "src/lib.rs");
    assert!(
        symbols[0]["signature"]
            .as_str()
            .unwrap()
            .contains("should_retry")
    );
    let refs = success(&call(client, "refs", json!({"name":"should_retry"})).await);
    assert_eq!(refs["approximate"], true);
    assert!(refs["groups"].is_array());
    assert!(refs["total"].as_u64().unwrap() > 0);
    let reference_path = root.join("src/lib.rs");
    let reference_original = fs::read(&reference_path).unwrap();
    for (watch_first, mutation) in [false, true].into_iter().flat_map(|watch_first| {
        ["edit", "delete", "rename", "checkout"].map(move |mutation| (watch_first, mutation))
    }) {
        match mutation {
            "edit" => {
                let mut shifted = b"\n\n".to_vec();
                shifted.extend_from_slice(&reference_original);
                fs::write(&reference_path, shifted).unwrap();
            }
            "delete" => fs::remove_file(&reference_path).unwrap(),
            "rename" => fs::rename(&reference_path, root.join("src/moved-reference.rs")).unwrap(),
            "checkout" => {
                fs::write(&reference_path, "// changed\n").unwrap();
                assert!(
                    std::process::Command::new("git")
                        .current_dir(root)
                        .args(["checkout", "--", "src/lib.rs"])
                        .status()
                        .unwrap()
                        .success()
                );
            }
            _ => unreachable!(),
        }
        if watch_first {
            wait_for_watcher_sources(root, &["src/lib.rs", "src/moved-reference.rs"]).await;
        }
        let mapped = success(&call(client, "map", json!({"focus":["src/lib.rs"]})).await);
        if matches!(mutation, "edit" | "checkout") {
            assert_at_most_one_query_repair(&mapped["index"], watch_first);
            assert_eq!(mapped["index"]["partial"], false);
            assert_eq!(mapped["nodes"][0]["path"], "src/lib.rs");
            let shift = if mutation == "edit" { 2 } else { 0 };
            let mapped_symbol = mapped["nodes"][0]["symbols"]
                .as_array()
                .unwrap()
                .iter()
                .find(|symbol| symbol["name"] == "should_retry")
                .unwrap();
            assert_eq!(
                mapped_symbol["lines"][0].as_u64().unwrap(),
                symbols[0]["lines"][0].as_u64().unwrap() + shift
            );
        } else {
            assert_eq!(mapped["nodes"], json!([]));
            assert_eq!(mapped["tree"], "");
            if mapped["index"]["partial"] == true {
                assert!(!watch_first, "{mapped}");
                assert!(mapped["index"]["suppressed_files"].as_u64().unwrap() > 0);
            } else {
                assert_eq!(mapped["index"]["partial"], false);
                assert_catalogue_absent(root, "src/lib.rs");
            }
        }
        let symbol_result = call(client, "symbol", json!({"name":"should_retry"})).await;
        if matches!(mutation, "edit" | "checkout") {
            let repaired = success(&symbol_result);
            assert_eq!(repaired[0]["path"], "src/lib.rs");
            let shift = if mutation == "edit" { 2 } else { 0 };
            assert_eq!(
                repaired[0]["lines"][0].as_u64().unwrap(),
                symbols[0]["lines"][0].as_u64().unwrap() + shift
            );
        } else if symbol_result.is_error == Some(true) {
            assert!(!watch_first);
            assert_eq!(
                failure(&symbol_result)["error"]["code"],
                "source_unavailable"
            );
        } else {
            let mut current = success(&symbol_result);
            assert_catalogue_absent(root, "src/lib.rs");
            if mutation == "delete" {
                assert_eq!(current, json!([]));
            } else {
                if current == json!([]) {
                    assert!(
                        !watch_first,
                        "settled rename lost its symbol: {symbol_result:?}"
                    );
                    // Deletion of the old catalogue path can precede admission
                    // of the new path. Targeted queries cannot discover new files.
                    // Keep the immediate observation, then require real watcher
                    // publication and the exact renamed symbol within CALL_TIMEOUT.
                    wait_for_watcher_sources(root, &["src/lib.rs", "src/moved-reference.rs"]).await;
                    current =
                        success(&call(client, "symbol", json!({"name":"should_retry"})).await);
                }
                assert_eq!(
                    current.as_array().unwrap().len(),
                    1,
                    "watch_first={watch_first}, initial={symbol_result:?}, current={current}"
                );
                assert_eq!(current[0]["path"], "src/moved-reference.rs");
                assert_eq!(current[0]["lines"], symbols[0]["lines"]);
            }
        }
        let result = success(&call(client, "refs", json!({"name":"should_retry"})).await);
        if mutation == "edit" {
            assert_eq!(result["index"]["lazy_resync"]["published_files"], 0);
            assert_eq!(result["index"]["partial"], false);
            assert_eq!(result["groups"][0]["lines"], json!([4]));
        } else if mutation == "checkout" {
            assert_eq!(result["index"]["lazy_resync"]["published_files"], 0);
            assert_eq!(result["groups"], refs["groups"]);
            assert_eq!(result["index"]["partial"], false);
        } else if result["index"]["partial"] == true {
            assert!(!watch_first, "{result}");
            assert!(result["total"].is_null(), "{result}");
            assert_eq!(result["index"]["degraded"], true, "{result}");
            if result["groups"] == json!([]) {
                assert!(
                    result["index"]["suppressed_references"].as_u64().unwrap() > 0,
                    "{result}"
                );
            } else {
                // Partial describes repository completeness, not the validity of
                // each retained group. A watcher may publish the renamed source
                // while unrelated L1 work is still pending. Only its exact fresh
                // location is acceptable; deletion never yields a retained group.
                assert_eq!(mutation, "rename", "{result}");
                let mut renamed = refs["groups"].clone();
                assert_eq!(renamed.as_array().unwrap().len(), 1);
                renamed[0]["path"] = json!("src/moved-reference.rs");
                assert_eq!(result["groups"], renamed, "{result}");
                assert_eq!(result["returned_total"], refs["returned_total"], "{result}");
            }
        } else {
            assert_eq!(result["index"]["partial"], false);
            assert_catalogue_absent(root, "src/lib.rs");
            if mutation == "delete" {
                assert_eq!(result["groups"], json!([]));
                assert_eq!(result["total"], 0);
            } else if result["groups"] == json!([]) {
                // A query can remove the stale old path before the watcher
                // discovers the new one, including after source_unavailable
                // from symbol. Empty means no verified indexed source here;
                // it must not claim fresh evidence for the undiscovered file.
                assert!(!watch_first, "settled rename lost its references: {result}");
                assert_eq!(result["total"], 0, "{result}");
                assert_eq!(result["index"]["fresh"]["state"], "unknown", "{result}");
            } else {
                assert_eq!(result["groups"].as_array().unwrap().len(), 1, "{result}");
                assert_eq!(result["groups"][0]["path"], "src/moved-reference.rs");
                assert_eq!(result["groups"][0]["lines"], refs["groups"][0]["lines"]);
                assert_eq!(result["total"], refs["total"]);
            }
        }
        if mutation == "rename" {
            // Both immediate schedules (partial and complete-empty) must
            // converge through the real watcher within the existing deadline.
            wait_for_watcher_sources(root, &["src/lib.rs", "src/moved-reference.rs"]).await;
            let settled = success(&call(client, "refs", json!({"name":"should_retry"})).await);
            let mut renamed = refs["groups"].clone();
            renamed[0]["path"] = json!("src/moved-reference.rs");
            assert_eq!(settled["groups"], renamed, "{settled}");
            assert_eq!(settled["total"], refs["total"], "{settled}");
            assert_eq!(settled["index"]["partial"], false, "{settled}");
            let settled_symbol =
                success(&call(client, "symbol", json!({"name":"should_retry"})).await);
            assert_eq!(
                settled_symbol.as_array().unwrap().len(),
                1,
                "{settled_symbol}"
            );
            assert_eq!(settled_symbol[0]["path"], "src/moved-reference.rs");
            assert_eq!(settled_symbol[0]["lines"], symbols[0]["lines"]);
            assert_catalogue_absent(root, "src/lib.rs");
            fs::remove_file(root.join("src/moved-reference.rs")).unwrap();
        }
        fs::write(&reference_path, &reference_original).unwrap();
        wait_for_watcher_sources(root, &["src/lib.rs", "src/moved-reference.rs"]).await;
    }
    let map_result = call(client, "map", json!({"budget_tokens":1000})).await;
    assert_eq!(
        map_result.meta.as_ref().unwrap().get("ttlMs"),
        Some(&json!(60_000))
    );
    let map = success(&map_result);
    assert!(map["tree"].as_str().unwrap().contains("lib.rs"));
    let status_result = call(client, "status", json!({})).await;
    assert_eq!(
        status_result.meta.as_ref().unwrap().get("ttlMs"),
        Some(&json!(5_000))
    );
    let status = success(&status_result);
    assert!(status["units"].as_u64().unwrap() > 0);
    assert!(status["dirty_files"].is_null());
    assert_eq!(status["dirty_files_state"], "not_checked");
    for field in ["ladder", "profile", "models", "budgets"] {
        assert!(status[field].is_object(), "missing {field}");
    }
    let samples = status["query_latency_ms"]["count"].as_u64().unwrap();
    assert!(samples > 0 && samples < 1000);
    assert_eq!(status["query_latency_ms"]["lifetime"], "process");
    assert!(
        status["query_latency_ms"]["p95"]
            .as_f64()
            .unwrap()
            .is_finite()
    );
    success(
        &call(
            client,
            "search",
            json!({"query":"should_retry", "mode":"lexical", "rerank":false}),
        )
        .await,
    );
    let after_search = success(&call(client, "status", json!({})).await);
    assert_eq!(after_search["query_latency_ms"]["count"], samples + 1);

    let recent = success(&call(client, "recent", json!({})).await);
    assert!(
        recent
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["kind"] == "commit" && item["title"] == "fixture")
    );
    let get = success(
        &call(
            client,
            "config_get",
            json!({"key":"search.graph_prior_weight"}),
        )
        .await,
    );
    assert_eq!(get["value"], 0.3);
    let explanation = success(
        &call(
            client,
            "config_explain",
            json!({"key":"search.graph_prior_weight"}),
        )
        .await,
    );
    assert_eq!(explanation["value"], 0.3);
    let set = call(
        client,
        "config_set",
        json!({"key":"search.graph_prior_weight", "value":0.25, "layer":"local"}),
    )
    .await;
    if writable {
        success(&set);
    } else {
        failure(&set);
    }
    let get = success(
        &call(
            client,
            "config_get",
            json!({"key":"search.graph_prior_weight"}),
        )
        .await,
    );
    assert_eq!(get["value"], if writable { 0.25 } else { 0.3 });
    // Reloading effective configuration retains the server's process window.
    let after_reload = success(&call(client, "status", json!({})).await);
    assert_eq!(after_reload["query_latency_ms"]["count"], samples + 1);

    if writable {
        // Configuration must remain repairable after selecting an unavailable index root.
        success(
            &call(
                client,
                "config_set",
                json!({"key":"index.root", "value":"missing-directory"}),
            )
            .await,
        );
        for tool in ["config_get", "config_explain"] {
            let response = success(&call(client, tool, json!({"key":"index.root"})).await);
            assert_eq!(response["value"], "missing-directory");
        }
        success(
            &call(
                client,
                "config_set",
                json!({"key":"index.root", "value":"."}),
            )
            .await,
        );
        let repaired = success(
            &call(
                client,
                "search",
                json!({"query":"should_retry", "mode":"lexical"}),
            )
            .await,
        );
        assert!(!repaired["hits"].as_array().unwrap().is_empty());
    }
    // Wrong argument types and unknown tools must not crash or end the session.
    for (name, args) in [
        ("search", json!({"query":17})),
        ("open", json!({})),
        ("unknown_tool", json!({})),
    ] {
        let result = timeout(
            CALL_TIMEOUT,
            client.call_tool(
                CallToolRequestParams::new(name).with_arguments(args.as_object().unwrap().clone()),
            ),
        )
        .await
        .unwrap();
        match result {
            Ok(result) if name == "search" && result.structured_content.is_none() => {
                // rmcp converts argument-deserialization failures into a textual tool
                // error before entering the application's structured-error adapter.
                assert_eq!(result.is_error, Some(true));
                let message = result
                    .content
                    .iter()
                    .find_map(|content| content.as_text())
                    .unwrap();
                assert!(
                    message
                        .text
                        .starts_with("failed to deserialize parameters:")
                );
                assert!(message.text.contains("expected a string"));
            }
            Ok(result) => {
                failure(&result);
            }
            Err(ServiceError::McpError(error)) => {
                assert_eq!(error.code, ErrorCode::INVALID_PARAMS);
            }
            Err(error) => panic!("malformed call must not break the transport: {error}"),
        }
    }
    // Keep this HEAD-observation contract read-only; checkout repair has its
    // own real-transport regression below.
    let override_path = root.join(".deepkit/config.local.toml");
    let previous_override = fs::read(&override_path).ok();
    fs::write(&override_path, "[search]\nlazy_resync=false\n").unwrap();
    let before = success(&call(client, "status", json!({})).await);
    let original_head = fs::read(root.join(".git/HEAD")).unwrap();
    // Move HEAD to an unborn branch without altering the source/index fixture.
    // A missing observed commit must invalidate the completed commit identity.
    assert!(
        std::process::Command::new("git")
            .current_dir(root)
            .args(["symbolic-ref", "HEAD", "refs/heads/mcp-unborn"])
            .status()
            .unwrap()
            .success()
    );
    let status = success(&call(client, "status", json!({})).await);
    assert!(status["head"].is_null());
    assert_eq!(status["indexed_head"], before["indexed_head"]);
    assert_eq!(status["fresh"]["reason"], "head_changed");
    for (tool, args) in [
        ("search", json!({"query":"should_retry", "mode":"lexical"})),
        ("map", json!({"budget_tokens":2000})),
        ("refs", json!({"name":"should_retry"})),
    ] {
        let result = success(&call(client, tool, args).await);
        assert_eq!(
            result["index"]["fresh"]["reason"], "head_changed",
            "{tool}: {result}"
        );
        assert!(result["index"]["head"].is_null());
    }
    fs::write(root.join(".git/HEAD"), original_head).unwrap();
    let restored = success(&call(client, "status", json!({})).await);
    assert_eq!(restored["head"], before["head"]);
    assert_ne!(restored["fresh"]["reason"], "head_changed");
    match previous_override {
        Some(bytes) => fs::write(override_path, bytes).unwrap(),
        None => fs::remove_file(override_path).unwrap(),
    }
}

#[tokio::test]
async fn stdio_client_lists_schemas_and_calls_every_tool() {
    let root = fixture(true);
    let (mut child, client) = stdio(root.path()).await;
    exercise_tools(&client, true, root.path()).await;
    timeout(CALL_TIMEOUT, client.cancel())
        .await
        .unwrap()
        .unwrap();
    timeout(CALL_TIMEOUT, child.wait()).await.unwrap().unwrap();
    assert!(!root.path().join("absent-model-cache").exists());
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

async fn http_child(root: &Path, bind: &str, token: Option<&str>) -> Child {
    http_child_with_user_config(root, bind, token, None).await
}

async fn http_child_with_user_config(
    root: &Path,
    bind: &str,
    token: Option<&str>,
    user_config: Option<&Path>,
) -> Child {
    let mut cmd = command(root);
    cmd.args(["serve", "--http", bind]).stdout(Stdio::null());
    if let Some(token) = token {
        cmd.env("DKC_SERVE__HTTP_TOKEN", token);
    }
    if let Some(user_config) = user_config {
        cmd.env("XDG_CONFIG_HOME", user_config);
    }
    let mut child = cmd.spawn().unwrap();
    let port = bind.rsplit(':').next().unwrap();
    let address = format!("127.0.0.1:{port}");
    let deadline = Instant::now() + CALL_TIMEOUT;
    loop {
        assert!(
            child.try_wait().unwrap().is_none(),
            "HTTP server exited before bind"
        );
        if tokio::net::TcpStream::connect(&address).await.is_ok() {
            return child;
        }
        assert!(Instant::now() < deadline, "HTTP server did not bind");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn http_client_calls_every_tool_and_refuses_config_mutation() {
    let root = fixture(false);
    let config_before = fs::read(root.path().join(".deepkit/config.toml")).unwrap();
    let address = format!("127.0.0.1:{}", free_port());
    let mut child = http_child(root.path(), &address, None).await;
    let transport = StreamableHttpClientTransport::from_uri(format!("http://{address}/mcp"));
    let client = timeout(CALL_TIMEOUT, ().serve(transport))
        .await
        .unwrap()
        .unwrap();
    exercise_tools(&client, false, root.path()).await;
    timeout(CALL_TIMEOUT, client.cancel())
        .await
        .unwrap()
        .unwrap();
    child.kill().await.unwrap();
    assert_eq!(
        fs::read(root.path().join(".deepkit/config.toml")).unwrap(),
        config_before
    );
    assert!(!root.path().join(".deepkit/config.local.toml").exists());
    assert!(!root.path().join("absent-model-cache").exists());
}

#[tokio::test]
async fn nonloopback_requires_token_and_authenticated_http_rejects_missing_or_wrong_bearer() {
    let root = fixture(false);
    let output = timeout(
        CALL_TIMEOUT,
        command(root.path())
            .args(["serve", "--http", "0.0.0.0:0"])
            .output(),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(!output.status.success());
    let token = blake3::hash(root.path().as_os_str().as_encoded_bytes())
        .to_hex()
        .to_string();
    let port = free_port();
    let mut child = http_child(root.path(), &format!("0.0.0.0:{port}"), Some(&token)).await;
    let url = format!("http://127.0.0.1:{port}/mcp");
    let http = reqwest::Client::builder()
        .timeout(CALL_TIMEOUT)
        .build()
        .unwrap();
    for supplied in [None, Some("invalid-runtime-bearer")] {
        let mut request = http.post(&url).header("Accept", "application/json, text/event-stream").json(&json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}));
        if let Some(supplied) = supplied {
            request = request.bearer_auth(supplied);
        }
        assert_eq!(request.send().await.unwrap().status(), 401);
    }
    let transport = StreamableHttpClientTransport::from_config(
        StreamableHttpClientTransportConfig::with_uri(url).auth_header(token),
    );
    let client = timeout(CALL_TIMEOUT, ().serve(transport))
        .await
        .unwrap()
        .unwrap();
    success(&call(&client, "status", json!({})).await);
    timeout(CALL_TIMEOUT, client.cancel())
        .await
        .unwrap()
        .unwrap();
    child.kill().await.unwrap();
}

#[tokio::test]
async fn trace_never_logs_submitted_credentials_even_with_explicit_sdk_trace_filter() {
    let root = fixture(true);
    let config_before = fs::read(root.path().join(".deepkit/config.toml")).unwrap();
    let token = blake3::hash(root.path().as_os_str().as_encoded_bytes())
        .to_hex()
        .to_string();
    let mut child = command(root.path())
        .args(["serve", "--stdio"])
        .env("DKC_LOG", "trace,rmcp::service=trace")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let stderr_task = tokio::spawn(async move {
        let mut bytes = Vec::new();
        stderr.read_to_end(&mut bytes).await.unwrap();
        bytes
    });
    let transport = (child.stdout.take().unwrap(), child.stdin.take().unwrap());
    let client = timeout(CALL_TIMEOUT, ().serve(transport))
        .await
        .unwrap()
        .unwrap();
    for layer in ["local", "repo"] {
        failure(
            &call(
                &client,
                "config_set",
                json!({"key":"serve.http_token", "value":token, "layer":layer}),
            )
            .await,
        );
        failure(
            &call(
                &client,
                "config_set",
                json!({"key":"serve", "value":{"http_token":token}, "layer":layer}),
            )
            .await,
        );
    }
    assert!(!root.path().join(".deepkit/config.local.toml").exists());
    assert_eq!(
        fs::read(root.path().join(".deepkit/config.toml")).unwrap(),
        config_before
    );
    failure(
        &call(
            &client,
            "config_set",
            json!({"key":"llm.api_key", "value":token}),
        )
        .await,
    );
    // serde's textual wrong-type error can echo the supplied string. That SDK
    // response must not be copied into process diagnostics either.
    let malformed = call(
        &client,
        "search",
        json!({"query":"should_retry", "verbose":token}),
    )
    .await;
    assert_eq!(malformed.is_error, Some(true));
    success(&call(&client, "status", json!({})).await);
    timeout(CALL_TIMEOUT, client.cancel())
        .await
        .unwrap()
        .unwrap();
    timeout(CALL_TIMEOUT, child.wait()).await.unwrap().unwrap();
    let stderr = timeout(CALL_TIMEOUT, stderr_task).await.unwrap().unwrap();
    assert!(
        !String::from_utf8_lossy(&stderr).contains(&token),
        "submitted credential appeared in stderr"
    );
}

#[tokio::test]
async fn http_token_rotation_applies_without_restart_and_lan_hosts_are_authorized() {
    let root = fixture(false);
    let user_config = tempfile::tempdir().unwrap();
    fs::create_dir(user_config.path().join("deepkit-code")).unwrap();
    let seed = root.path().as_os_str().as_encoded_bytes();
    let first = blake3::derive_key("dkc MCP test first runtime token", seed);
    let second = blake3::derive_key("dkc MCP test rotated runtime token", seed);
    let first = blake3::Hash::from(first).to_hex().to_string();
    let second = blake3::Hash::from(second).to_hex().to_string();
    let write_token = |token: &str| {
        // Runtime-only user configuration outside the repository; no token env override.
        use std::io::Write;
        let mut options = fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        options
            .open(user_config.path().join("deepkit-code/config.toml"))
            .unwrap()
            .write_all(format!("[serve]\nhttp_token={token:?}\n").as_bytes())
            .unwrap();
    };
    write_token(&first);
    let port = free_port();
    let mut child = http_child_with_user_config(
        root.path(),
        &format!("0.0.0.0:{port}"),
        None,
        Some(user_config.path()),
    )
    .await;
    let url = format!("http://127.0.0.1:{port}/mcp");
    let http = reqwest::Client::builder()
        .timeout(CALL_TIMEOUT)
        .build()
        .unwrap();
    let initialize = |bearer: Option<&str>| {
        let mut request = http.post(&url)
            .header("Host", "bench.example.invalid")
            .header("Accept", "application/json, text/event-stream")
            .json(&json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"rotation-test","version":"1"}}}));
        if let Some(bearer) = bearer {
            request = request.bearer_auth(bearer);
        }
        request
    };
    for token in [&first, &second] {
        write_token(token);
        let response = initialize(Some(token)).send().await.unwrap();
        assert_eq!(response.status(), 200, "authorized LAN Host must reach MCP");
        let response: Value = response.json().await.unwrap();
        assert_eq!(response["jsonrpc"], "2.0");
        assert!(response["result"]["serverInfo"].is_object());
        let wrong = if token == &first { &second } else { &first };
        assert_eq!(initialize(Some(wrong)).send().await.unwrap().status(), 401);
        assert_eq!(initialize(None).send().await.unwrap().status(), 401);
    }
    write_token("");
    for token in [None, Some(first.as_str()), Some(second.as_str())] {
        assert_eq!(
            initialize(token).send().await.unwrap().status(),
            401,
            "non-loopback token clearing must fail closed"
        );
    }
    assert!(
        child.try_wait().unwrap().is_none(),
        "rotation must not restart or stop the server"
    );
    child.kill().await.unwrap();
    assert!(!root.path().join(".deepkit/config.local.toml").exists());
}

#[tokio::test]
async fn trace_stdio_stdout_contains_only_jsonrpc_frames() {
    let root = fixture(false);
    let mut child = command(root.path())
        .args(["serve", "--stdio"])
        .env("DKC_LOG", "trace")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    for request in [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"trace-test","version":"1"}}}),
        json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}),
        json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"status","arguments":{}}}),
    ] {
        stdin
            .write_all(format!("{request}\n").as_bytes())
            .await
            .unwrap();
        stdin.flush().await.unwrap();
        if let Some(id) = request.get("id") {
            let deadline = Instant::now() + CALL_TIMEOUT;
            loop {
                let line = timeout_at(deadline, lines.next_line())
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap();
                let frame: Value =
                    serde_json::from_str(&line).expect("stdout must contain JSON, never tracing");
                assert_eq!(frame["jsonrpc"], "2.0");
                assert!(
                    frame.get("method").is_some()
                        || frame.get("result").is_some()
                        || frame.get("error").is_some()
                );
                if frame.get("id") == Some(id) {
                    assert!(frame.get("result").is_some(), "{frame}");
                    break;
                }
            }
        }
    }
    drop(stdin);
    timeout(CALL_TIMEOUT, child.wait()).await.unwrap().unwrap();
    while let Some(line) = timeout(CALL_TIMEOUT, lines.next_line())
        .await
        .unwrap()
        .unwrap()
    {
        let frame: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(frame["jsonrpc"], "2.0");
    }
}

#[test]
fn cli_repeated_scope_matches_the_mcp_union() {
    let root = fixture(false);
    let output = std::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root.path())
        .env("XDG_CONFIG_HOME", root.path().join("user-config"))
        .args([
            "search",
            "scope_probe",
            "--mode",
            "lexical",
            "--scope",
            "src/*",
            "--scope",
            "alt/*",
            "--scope",
            "src/scope.rs",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    let hits = value["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(
        hits.iter()
            .map(|h| h["path"].as_str().unwrap())
            .collect::<std::collections::BTreeSet<_>>(),
        std::collections::BTreeSet::from(["src/scope.rs", "alt/scope.rs"])
    );
}

#[test]
fn cli_search_repairs_changed_ranges_and_returns_current_openable_ids() {
    let root = fixture(false);
    let run = |args: &[&str]| {
        let output = std::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
            .current_dir(root.path())
            .env("XDG_CONFIG_HOME", root.path().join("user-config"))
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice::<Value>(&output.stdout).unwrap()
    };
    let args = [
        "search",
        "scope_probe",
        "--mode",
        "lexical",
        "--scope",
        "src/*",
        "--json",
    ];
    let old = run(&args);
    fs::write(
        root.path().join("src/scope.rs"),
        "\n\npub fn scope_probe() -> bool { false }\n",
    )
    .unwrap();
    let current = run(&args);
    assert_eq!(current["hits"].as_array().unwrap().len(), 1);
    assert_eq!(current["hits"][0]["path"], "src/scope.rs");
    assert_eq!(current["hits"][0]["lines"][0], 3);
    assert_ne!(current["hits"][0]["id"], old["hits"][0]["id"]);
    assert_eq!(current["index"]["partial"], false);
    assert_eq!(current["index"]["lazy_resync"]["published_files"], 1);
    let opened = run(&["open", current["hits"][0]["id"].as_str().unwrap(), "--json"]);
    assert!(opened["text"].as_str().unwrap().contains("false"));
    // A no-longer-matching replacement must disappear after reretrieval.
    fs::write(
        root.path().join("src/scope.rs"),
        "pub fn replacement() {}\n",
    )
    .unwrap();
    let removed = run(&args);
    assert_eq!(removed["hits"], json!([]));
    assert_eq!(removed["index"]["partial"], false);
    assert_eq!(removed["index"]["lazy_resync"]["published_files"], 1);
}

#[test]
fn cli_search_suppresses_deleted_hits_but_keeps_verified_files() {
    let root = fixture(false);
    fs::remove_file(root.path().join("src/scope.rs")).unwrap();
    let output = std::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root.path())
        .env("XDG_CONFIG_HOME", root.path().join("user-config"))
        .args([
            "search",
            "scope_probe",
            "--scope",
            "code",
            "--mode",
            "lexical",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["hits"].as_array().unwrap().len(), 1);
    assert_eq!(value["hits"][0]["path"], "alt/scope.rs");
    assert_eq!(value["hits"][0]["fresh"]["state"], "fresh");
    assert_eq!(value["index"]["partial"], true);
    assert_eq!(value["index"]["suppressed_hits"], 1);
    assert_eq!(
        value["index"]["freshness_refusals"]["source_unavailable"],
        1
    );
}

#[test]
fn cli_search_timeout_reports_partial_json_and_incomplete_text() {
    let root = fixture(false);
    let config_path = root.path().join(".deepkit/config.toml");
    let config = fs::read_to_string(&config_path).unwrap();
    fs::write(
        config_path,
        format!("{config}\n[limits]\nquery_timeout_ms=0\n"),
    )
    .unwrap();
    for json_output in [true, false] {
        let mut command = std::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"));
        command
            .current_dir(root.path())
            .env("XDG_CONFIG_HOME", root.path().join("user-config"))
            .args(["search", "should_retry", "--mode", "lexical"]);
        if json_output {
            command.arg("--json");
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        if json_output {
            let value: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(value["hits"], json!([]));
            assert_eq!(value["index"]["partial"], true);
            assert_eq!(value["index"]["reason"], "query_timeout");
            assert_eq!(value["index"]["fresh"]["state"], "unknown");
            assert!(value["index"]["head"].is_null());
        } else {
            assert!(String::from_utf8_lossy(&output.stdout).contains("search incomplete"));
            assert!(String::from_utf8_lossy(&output.stderr).contains("query deadline exceeded"));
            assert!(!String::from_utf8_lossy(&output.stderr).contains("model work"));
        }
    }
}

#[test]
fn cli_open_timeout_is_explicit_and_a_new_request_can_retry() {
    let root = fixture(false);
    let config_path = root.path().join(".deepkit/config.local.toml");
    let saved = fs::read(&config_path).ok();
    fs::write(&config_path, "[limits]\nquery_timeout_ms=0\n").unwrap();
    let run = || {
        std::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
            .current_dir(root.path())
            .env("XDG_CONFIG_HOME", root.path().join("user-config"))
            .args(["open", "src/lib.rs", "--json"])
            .output()
            .unwrap()
    };
    let failed = run();
    assert!(!failed.status.success());
    let value: Value = serde_json::from_slice(&failed.stdout).unwrap();
    assert_eq!(value["error"]["code"], "timeout");
    if let Some(saved) = saved {
        fs::write(&config_path, saved).unwrap();
    } else {
        fs::remove_file(&config_path).unwrap();
    }
    let retried = run();
    assert!(
        retried.status.success(),
        "{}",
        String::from_utf8_lossy(&retried.stderr)
    );
    let value: Value = serde_json::from_slice(&retried.stdout).unwrap();
    assert_eq!(value["path"], "src/lib.rs");
    assert!(!value["text"].as_str().unwrap().is_empty());
}

fn branch_git(root: &Path, args: &[&str]) -> String {
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
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

fn checkout_search_branch(root: &Path) -> String {
    let first = branch_git(root, &["rev-parse", "HEAD"]);
    fs::rename(root.join("src/lib.rs"), root.join("src/moved.rs")).unwrap();
    fs::remove_file(root.join("alt/scope.rs")).unwrap();
    fs::write(
        root.join("src/added.rs"),
        "pub fn checkout_discovery_probe() { should_retry(1); }\n",
    )
    .unwrap();
    branch_git(root, &["add", "--all", "src", "alt"]);
    branch_git(
        root,
        &[
            "-c",
            "user.name=MCP Test",
            "-c",
            "user.email=mcp@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "checkout",
        ],
    );
    let second = branch_git(root, &["rev-parse", "HEAD"]);
    branch_git(root, &["checkout", "--quiet", &first]);
    branch_git(root, &["checkout", "--quiet", &second]);
    second
}

fn assert_branch_search(result: &Value, head: &str) {
    assert_eq!(result["index"]["head"], head);
    assert_eq!(result["index"]["indexed_head"], head);
    assert_eq!(result["index"]["l1_pending_files"], 0);
    assert_eq!(result["index"]["partial"], false);
    assert_eq!(result["index"]["lexical"], "complete");
    assert_eq!(result["hits"][0]["path"], "src/added.rs");
    assert_eq!(result["hits"][0]["fresh"]["l1"], "current");
}

#[test]
fn cli_search_reconciles_checkout_before_retrieval() {
    let root = fixture(false);
    let head = checkout_search_branch(root.path());
    let output = std::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root.path())
        .env("XDG_CONFIG_HOME", root.path().join("user-config"))
        .args([
            "search",
            "checkout_discovery_probe",
            "--mode",
            "lexical",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["index"]["branch_resync"]["state"], "completed");
    assert_branch_search(&result, &head);
}

#[tokio::test]
async fn both_mcp_search_transports_reconcile_checkout_before_retrieval() {
    for (http, watch_first) in [false, true]
        .into_iter()
        .flat_map(|http| [false, true].map(move |watch_first| (http, watch_first)))
    {
        let root = fixture(false);
        let (mut child, client) = if http {
            let address = format!("127.0.0.1:{}", free_port());
            let child = http_child(root.path(), &address, None).await;
            let transport =
                StreamableHttpClientTransport::from_uri(format!("http://{address}/mcp"));
            (
                child,
                timeout(CALL_TIMEOUT, ().serve(transport))
                    .await
                    .unwrap()
                    .unwrap(),
            )
        } else {
            stdio(root.path()).await
        };
        let head = checkout_search_branch(root.path());
        if watch_first {
            wait_for_watcher_head(root.path(), &head).await;
        }
        let result = success(
            &call(
                &client,
                "search",
                json!({"query":"checkout_discovery_probe", "mode":"lexical"}),
            )
            .await,
        );
        assert_mcp_branch_schedule(&result, watch_first);
        assert_branch_search(&result, &head);
        let renamed = success(
            &call(
                &client,
                "search",
                json!({"query":"should_retry", "mode":"lexical"}),
            )
            .await,
        );
        assert_eq!(renamed["hits"][0]["path"], "src/moved.rs");
        let deleted = success(
            &call(
                &client,
                "search",
                json!({"query":"scope_probe", "mode":"lexical", "scope":["alt/*"]}),
            )
            .await,
        );
        assert_eq!(deleted["hits"], json!([]));
        assert_eq!(deleted["index"]["partial"], false);
        timeout(CALL_TIMEOUT, client.cancel())
            .await
            .unwrap()
            .unwrap();
        if http {
            child.kill().await.unwrap();
        } else {
            timeout(CALL_TIMEOUT, child.wait()).await.unwrap().unwrap();
        }
    }
}

#[test]
fn cli_symbol_branch_refusal_and_checkout_recovery() {
    let root = fixture(false);
    let head = checkout_search_branch(root.path());
    let override_path = root.path().join(".deepkit/config.local.toml");
    let invoke = |name: &str| {
        std::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
            .current_dir(root.path())
            .env("XDG_CONFIG_HOME", root.path().join("user-config"))
            .args(["symbol", name, "--json"])
            .output()
            .unwrap()
    };
    for config in ["[search]\nlazy_resync=false\n", "[limits]\nmax_files=1\n"] {
        fs::write(&override_path, config).unwrap();
        for name in ["checkout_discovery_probe", "absent_identifier"] {
            let output = invoke(name);
            assert!(!output.status.success());
            let error: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(error["error"]["code"], "source_unavailable", "{error}");
        }
    }
    fs::remove_file(override_path).unwrap();
    for (name, expected_paths) in [
        ("checkout_discovery_probe", vec!["src/added.rs"]),
        ("should_retry", vec!["src/moved.rs"]),
        ("scope_probe", vec!["src/scope.rs"]),
        ("absent_identifier", vec![]),
    ] {
        let output = invoke(name);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let result: Value = serde_json::from_slice(&output.stdout).unwrap();
        let paths: Vec<_> = result
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["path"].as_str().unwrap())
            .collect();
        assert_eq!(paths, expected_paths);
    }
    let db = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
    let indexed: String = db
        .query_row(
            "SELECT value FROM meta WHERE key='indexed_head'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(indexed, head);
}

#[tokio::test]
async fn both_mcp_symbol_transports_refuse_incomplete_checkout_and_recover() {
    for http in [false, true] {
        let root = fixture(false);
        let (mut child, client) = if http {
            let address = format!("127.0.0.1:{}", free_port());
            let child = http_child(root.path(), &address, None).await;
            let transport =
                StreamableHttpClientTransport::from_uri(format!("http://{address}/mcp"));
            (
                child,
                timeout(CALL_TIMEOUT, ().serve(transport))
                    .await
                    .unwrap()
                    .unwrap(),
            )
        } else {
            stdio(root.path()).await
        };
        let head = checkout_search_branch(root.path());
        let override_path = root.path().join(".deepkit/config.local.toml");
        for config in ["[search]\nlazy_resync=false\n", "[limits]\nmax_files=1\n"] {
            fs::write(&override_path, config).unwrap();
            for name in ["checkout_discovery_probe", "absent_identifier"] {
                let result = call(&client, "symbol", json!({"name":name})).await;
                assert_eq!(failure(&result)["error"]["code"], "source_unavailable");
            }
        }
        fs::remove_file(override_path).unwrap();
        for (name, expected_paths) in [
            ("checkout_discovery_probe", vec!["src/added.rs"]),
            ("should_retry", vec!["src/moved.rs"]),
            ("scope_probe", vec!["src/scope.rs"]),
            ("absent_identifier", vec![]),
        ] {
            let result = success(&call(&client, "symbol", json!({"name":name})).await);
            let paths: Vec<_> = result
                .as_array()
                .unwrap()
                .iter()
                .map(|row| row["path"].as_str().unwrap())
                .collect();
            assert_eq!(paths, expected_paths);
        }
        let status = success(&call(&client, "status", json!({})).await);
        assert_eq!(status["indexed_head"], head);
        assert_eq!(status["l1_pending_files"], 0);
        timeout(CALL_TIMEOUT, client.cancel())
            .await
            .unwrap()
            .unwrap();
        if http {
            child.kill().await.unwrap();
        } else {
            timeout(CALL_TIMEOUT, child.wait()).await.unwrap().unwrap();
        }
    }
}

fn assert_navigation_branch(result: &Value, tool: &str, head: &str) {
    assert_eq!(result["index"]["head"], head);
    assert_eq!(result["index"]["indexed_head"], head);
    assert_eq!(result["index"]["l1_pending_files"], 0);
    assert_eq!(result["index"]["partial"], false, "{result}");
    assert_eq!(result["index"]["degraded"], false);
    let rows = result[if tool == "refs" { "groups" } else { "nodes" }]
        .as_array()
        .unwrap();
    assert!(
        rows.iter().any(|row| row["path"] == "src/added.rs"),
        "{result}"
    );
    assert!(
        rows.iter().any(|row| row["path"] == "src/moved.rs"),
        "{result}"
    );
    assert!(
        !rows
            .iter()
            .any(|row| row["path"] == "src/lib.rs" || row["path"] == "alt/scope.rs")
    );
    if tool == "refs" {
        assert!(result["total"].as_u64().unwrap() >= 2);
    }
}

#[test]
fn cli_refs_and_map_branch_refusal_and_checkout_recovery() {
    for tool in ["refs", "map"] {
        let root = fixture(false);
        let head = checkout_search_branch(root.path());
        let override_path = root.path().join(".deepkit/config.local.toml");
        let invoke = |empty: bool| {
            let args = if tool == "refs" {
                vec![
                    "refs",
                    if empty {
                        "absent_identifier"
                    } else {
                        "should_retry"
                    },
                    "--json",
                ]
            } else if empty {
                vec!["map", "--focus", "absent_identifier", "--json"]
            } else {
                vec!["map", "--json"]
            };
            let output = std::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
                .current_dir(root.path())
                .env("XDG_CONFIG_HOME", root.path().join("user-config"))
                .args(args)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            serde_json::from_slice::<Value>(&output.stdout).unwrap()
        };
        for (config, state) in [
            ("[search]\nlazy_resync=false\n", "disabled"),
            ("[limits]\nmax_files=1\n", "incomplete"),
        ] {
            fs::write(&override_path, config).unwrap();
            for empty in [false, true] {
                let result = invoke(empty);
                assert_eq!(result["index"]["partial"], true, "{result}");
                assert_eq!(result["index"]["degraded"], true);
                assert_eq!(result["index"]["branch_resync"]["state"], state, "{result}");
                if tool == "refs" {
                    assert!(result["total"].is_null());
                }
            }
        }
        fs::remove_file(override_path).unwrap();
        let result = invoke(false);
        assert_eq!(result["index"]["branch_resync"]["state"], "completed");
        assert_navigation_branch(&result, tool, &head);
        let empty = invoke(true);
        assert_eq!(empty["index"]["partial"], false);
        assert_eq!(
            empty[if tool == "refs" { "groups" } else { "nodes" }],
            json!([])
        );
    }
}

#[tokio::test]
async fn both_mcp_refs_and_map_transports_reconcile_checkout_and_recover() {
    for (http, watch_first) in [false, true]
        .into_iter()
        .flat_map(|http| [false, true].map(move |watch_first| (http, watch_first)))
    {
        for tool in ["refs", "map"] {
            let root = fixture(false);
            let (mut child, client) = if http {
                let address = format!("127.0.0.1:{}", free_port());
                let child = http_child(root.path(), &address, None).await;
                let transport =
                    StreamableHttpClientTransport::from_uri(format!("http://{address}/mcp"));
                (
                    child,
                    timeout(CALL_TIMEOUT, ().serve(transport))
                        .await
                        .unwrap()
                        .unwrap(),
                )
            } else {
                stdio(root.path()).await
            };
            let head = checkout_search_branch(root.path());
            let override_path = root.path().join(".deepkit/config.local.toml");
            let args = |empty| {
                if tool == "refs" {
                    json!({"name":if empty { "absent_identifier" } else { "should_retry" }})
                } else if empty {
                    json!({"focus":["absent_identifier"]})
                } else {
                    json!({})
                }
            };
            for config in ["[search]\nlazy_resync=false\n", "[limits]\nmax_files=1\n"] {
                fs::write(&override_path, config).unwrap();
                for empty in [false, true] {
                    let result = success(&call(&client, tool, args(empty)).await);
                    assert_eq!(result["index"]["partial"], true, "{result}");
                    assert_eq!(result["index"]["degraded"], true);
                    if tool == "refs" {
                        assert!(result["total"].is_null());
                    }
                }
            }
            fs::remove_file(override_path).unwrap();
            if watch_first {
                wait_for_watcher_head(root.path(), &head).await;
            }
            let result = success(&call(&client, tool, args(false)).await);
            assert_mcp_branch_schedule(&result, watch_first);
            assert_navigation_branch(&result, tool, &head);
            let empty = success(&call(&client, tool, args(true)).await);
            assert_eq!(empty["index"]["partial"], false, "{empty}");
            assert_eq!(
                empty[if tool == "refs" { "groups" } else { "nodes" }],
                json!([])
            );
            timeout(CALL_TIMEOUT, client.cancel())
                .await
                .unwrap()
                .unwrap();
            if http {
                child.kill().await.unwrap();
            } else {
                timeout(CALL_TIMEOUT, child.wait()).await.unwrap().unwrap();
            }
        }
    }
}

#[test]
fn cli_open_branch_refusal_and_checkout_recovery() {
    let root = fixture(false);
    let invoke = |target: &str| {
        std::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
            .current_dir(root.path())
            .env("XDG_CONFIG_HOME", root.path().join("user-config"))
            .args(["open", target, "--json"])
            .output()
            .unwrap()
    };
    let override_path = root.path().join(".deepkit/config.local.toml");
    fs::write(&override_path, "[limits]\nmax_open_bytes='8 B'\n").unwrap();
    let original: Value = serde_json::from_slice(&invoke("src/lib.rs").stdout).unwrap();
    let old_cursor = original["continuation_id"].as_str().unwrap();
    fs::remove_file(&override_path).unwrap();
    let old_id = original["id"].as_str().unwrap();
    let head = checkout_search_branch(root.path());
    let override_path = root.path().join(".deepkit/config.local.toml");
    for config in ["[search]\nlazy_resync=false\n", "[limits]\nmax_files=1\n"] {
        fs::write(&override_path, config).unwrap();
        for target in ["src/added.rs", "src/moved.rs", old_id, "missing.rs"] {
            let output = invoke(target);
            assert!(!output.status.success());
            let error: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(error["error"]["code"], "source_unavailable", "{error}");
        }
    }
    fs::remove_file(override_path).unwrap();
    for (index, path) in ["src/added.rs", "src/moved.rs"].into_iter().enumerate() {
        let output = invoke(path);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let result: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["path"], path);
        assert_eq!(result["index"]["head"], head);
        assert_eq!(result["index"]["indexed_head"], head);
        assert_eq!(
            result["index"]["branch_resync"]["state"],
            if index == 0 {
                "completed"
            } else {
                "not_needed"
            }
        );
    }
    for target in ["src/lib.rs", "alt/scope.rs", old_id, old_cursor] {
        assert!(!invoke(target).status.success(), "{target}");
    }
}

#[tokio::test]
async fn both_mcp_open_transports_refuse_incomplete_checkout_and_recover() {
    for http in [false, true] {
        let root = fixture(false);
        let (mut child, client) = if http {
            let address = format!("127.0.0.1:{}", free_port());
            let child = http_child(root.path(), &address, None).await;
            let transport =
                StreamableHttpClientTransport::from_uri(format!("http://{address}/mcp"));
            (
                child,
                timeout(CALL_TIMEOUT, ().serve(transport))
                    .await
                    .unwrap()
                    .unwrap(),
            )
        } else {
            stdio(root.path()).await
        };
        let override_path = root.path().join(".deepkit/config.local.toml");
        fs::write(&override_path, "[limits]\nmax_open_bytes='8 B'\n").unwrap();
        let original = success(&call(&client, "open", json!({"path":"src/lib.rs"})).await);
        let old_cursor = original["continuation_id"].as_str().unwrap();
        fs::remove_file(&override_path).unwrap();
        let old_id = original["id"].as_str().unwrap();
        let head = checkout_search_branch(root.path());
        let override_path = root.path().join(".deepkit/config.local.toml");
        for config in ["[search]\nlazy_resync=false\n", "[limits]\nmax_files=1\n"] {
            fs::write(&override_path, config).unwrap();
            for target in ["src/added.rs", "src/moved.rs", "missing.rs"] {
                let result = call(&client, "open", json!({"path":target})).await;
                assert_eq!(failure(&result)["error"]["code"], "source_unavailable");
            }
        }
        fs::remove_file(override_path).unwrap();
        for (index, path) in ["src/added.rs", "src/moved.rs"].into_iter().enumerate() {
            let result = success(&call(&client, "open", json!({"path":path})).await);
            assert_eq!(result["path"], path);
            assert_eq!(result["index"]["head"], head);
            assert_eq!(result["index"]["indexed_head"], head);
            assert_eq!(
                result["index"]["branch_resync"]["state"],
                if index == 0 {
                    "completed"
                } else {
                    "not_needed"
                }
            );
        }
        for args in [
            json!({"path":"src/lib.rs"}),
            json!({"path":"alt/scope.rs"}),
            json!({"id":old_id}),
            json!({"id":old_cursor}),
        ] {
            failure(&call(&client, "open", args).await);
        }
        timeout(CALL_TIMEOUT, client.cancel())
            .await
            .unwrap()
            .unwrap();
        if http {
            child.kill().await.unwrap();
        } else {
            timeout(CALL_TIMEOUT, child.wait()).await.unwrap().unwrap();
        }
    }
}

#[tokio::test]
async fn cli_and_both_mcp_open_repair_same_head_edits_and_refuse_old_pages() {
    for transport in ["cli", "stdio", "http"] {
        let root = fixture(false);
        let (mut child, client) = match transport {
            "http" => {
                let address = format!("127.0.0.1:{}", free_port());
                let child = http_child(root.path(), &address, None).await;
                let wire = StreamableHttpClientTransport::from_uri(format!("http://{address}/mcp"));
                let client = timeout(CALL_TIMEOUT, ().serve(wire))
                    .await
                    .unwrap()
                    .unwrap();
                (Some(child), Some(client))
            }
            "stdio" => {
                let (child, client) = stdio(root.path()).await;
                (Some(child), Some(client))
            }
            _ => (None, None),
        };
        async fn invoke(root: &Path, client: Option<&Client>, target: &str) -> Value {
            if let Some(client) = client {
                let result = call(client, "open", json!({"id": target})).await;
                if result.is_error == Some(true) {
                    failure(&result)
                } else {
                    success(&result)
                }
            } else {
                let output = std::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
                    .current_dir(root)
                    .env("XDG_CONFIG_HOME", root.join("user-config"))
                    .args(["open", target, "--json"])
                    .output()
                    .unwrap();
                serde_json::from_slice(&output.stdout).unwrap()
            }
        }
        let config = root.path().join(".deepkit/config.local.toml");
        fs::write(&config, "[limits]\nmax_open_bytes='8 B'\n").unwrap();
        let original = invoke(root.path(), client.as_ref(), "src/lib.rs").await;
        let cursor = original["continuation_id"].as_str().unwrap();
        let id = original["id"].as_str().unwrap();
        let source = root.path().join("src/lib.rs");
        fs::write(&source, "pub fn replacement() -> bool { true }\n").unwrap();
        fs::write(&config, "[search]\nlazy_resync=false\n").unwrap();
        let refused = invoke(root.path(), client.as_ref(), "src/lib.rs").await;
        assert_eq!(
            refused["error"]["code"], "source_unavailable",
            "{transport}: {refused}"
        );
        fs::remove_file(&config).unwrap();
        let stale = invoke(root.path(), client.as_ref(), cursor).await;
        assert!(stale["error"].is_object(), "{transport}: {stale}");
        let stale = invoke(root.path(), client.as_ref(), id).await;
        assert!(stale["error"].is_object(), "{transport}: {stale}");
        let repaired = invoke(root.path(), client.as_ref(), "src/lib.rs").await;
        assert_eq!(
            repaired["text"], "pub fn replacement() -> bool { true }\n",
            "{transport}: {repaired}"
        );
        assert_eq!(repaired["index"]["branch_resync"]["state"], "not_needed");
        fs::remove_file(&source).unwrap();
        // MCP's automatic watcher can remove the catalogue row before open.
        // Select that completed state explicitly; CLI retains the old row.
        if client.is_some() {
            wait_for_watcher_sources(root.path(), &["src/lib.rs"]).await;
            assert_catalogue_absent(root.path(), "src/lib.rs");
        }
        let deleted = invoke(root.path(), client.as_ref(), "src/lib.rs").await;
        assert_eq!(
            deleted["error"]["code"],
            if client.is_some() {
                "not_found"
            } else {
                "source_unavailable"
            },
            "{transport}: {deleted}"
        );
        fs::write(&source, "pub fn recovered() {}\n").unwrap();
        if client.is_some() {
            wait_for_watcher_sources(root.path(), &["src/lib.rs"]).await;
        }
        assert_eq!(
            invoke(root.path(), client.as_ref(), "src/lib.rs").await["text"],
            "pub fn recovered() {}\n"
        );
        // Reload settings between public calls while keeping source and HEAD unchanged.
        fs::write(
            &config,
            "[search]\nlazy_resync=false\n[tokenize]\nsplit_subtokens=false\n",
        )
        .unwrap();
        let stale_config = invoke(root.path(), client.as_ref(), "src/lib.rs").await;
        assert_eq!(
            stale_config["error"]["code"], "source_unavailable",
            "{transport}: {stale_config}"
        );
        fs::write(&config, "[tokenize]\nsplit_subtokens=false\n").unwrap();
        let repaired_config = invoke(root.path(), client.as_ref(), "src/lib.rs").await;
        assert_eq!(
            repaired_config["text"], "pub fn recovered() {}\n",
            "{transport}: {repaired_config}"
        );
        if let Some(client) = client {
            timeout(CALL_TIMEOUT, client.cancel())
                .await
                .unwrap()
                .unwrap();
            let child = child.as_mut().unwrap();
            if transport == "http" {
                child.kill().await.unwrap();
            } else {
                timeout(CALL_TIMEOUT, child.wait()).await.unwrap().unwrap();
            }
        }
    }
}

#[tokio::test]
async fn cli_and_both_mcp_symbols_repair_config_only_staleness() {
    for transport in ["cli", "stdio", "http"] {
        let root = fixture(false);
        let (mut child, client) = match transport {
            "http" => {
                let address = format!("127.0.0.1:{}", free_port());
                let child = http_child(root.path(), &address, None).await;
                let wire = StreamableHttpClientTransport::from_uri(format!("http://{address}/mcp"));
                let client = timeout(CALL_TIMEOUT, ().serve(wire))
                    .await
                    .unwrap()
                    .unwrap();
                (Some(child), Some(client))
            }
            "stdio" => {
                let (child, client) = stdio(root.path()).await;
                (Some(child), Some(client))
            }
            _ => (None, None),
        };
        async fn invoke(root: &Path, client: Option<&Client>, target: &str) -> Value {
            if let Some(client) = client {
                let result = call(client, "symbol", json!({"name": target})).await;
                if result.is_error == Some(true) {
                    failure(&result)
                } else {
                    success(&result)
                }
            } else {
                let output = std::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
                    .current_dir(root)
                    .env("XDG_CONFIG_HOME", root.join("user-config"))
                    .args(["symbol", target, "--json"])
                    .output()
                    .unwrap();
                serde_json::from_slice(&output.stdout).unwrap()
            }
        }
        let config = root.path().join(".deepkit/config.local.toml");
        let source = fs::read(root.path().join("src/lib.rs")).unwrap();
        let original = invoke(root.path(), client.as_ref(), "should_retry").await;
        assert_eq!(
            original.as_array().unwrap().len(),
            1,
            "{transport}: {original}"
        );
        fs::write(
            &config,
            "[search]\nlazy_resync=false\n[tokenize]\nsplit_subtokens=false\n",
        )
        .unwrap();
        let refused = invoke(root.path(), client.as_ref(), "should_retry").await;
        assert_eq!(
            refused["error"]["code"], "source_unavailable",
            "{transport}: {refused}"
        );
        fs::write(&config, "[tokenize]\nsplit_subtokens=false\n").unwrap();
        let repaired = invoke(root.path(), client.as_ref(), "should_retry").await;
        assert_eq!(
            repaired.as_array().unwrap().len(),
            1,
            "{transport}: {repaired}"
        );
        assert_eq!(repaired[0]["id"], original[0]["id"]);
        assert_eq!(repaired[0]["lines"], original[0]["lines"]);
        let retry = invoke(root.path(), client.as_ref(), "should_retry").await;
        assert_eq!(retry[0]["id"], repaired[0]["id"]);
        assert_eq!(fs::read(root.path().join("src/lib.rs")).unwrap(), source);
        if let Some(client) = client {
            timeout(CALL_TIMEOUT, client.cancel())
                .await
                .unwrap()
                .unwrap();
            let child = child.as_mut().unwrap();
            if transport == "http" {
                child.kill().await.unwrap();
            } else {
                timeout(CALL_TIMEOUT, child.wait()).await.unwrap().unwrap();
            }
        }
    }
}

#[tokio::test]
async fn cli_and_both_mcp_refs_repair_config_only_staleness() {
    for transport in ["cli", "stdio", "http"] {
        let root = fixture(false);
        let (mut child, client) = match transport {
            "http" => {
                let address = format!("127.0.0.1:{}", free_port());
                let child = http_child(root.path(), &address, None).await;
                let wire = StreamableHttpClientTransport::from_uri(format!("http://{address}/mcp"));
                let client = timeout(CALL_TIMEOUT, ().serve(wire))
                    .await
                    .unwrap()
                    .unwrap();
                (Some(child), Some(client))
            }
            "stdio" => {
                let (child, client) = stdio(root.path()).await;
                (Some(child), Some(client))
            }
            _ => (None, None),
        };
        async fn invoke(root: &Path, client: Option<&Client>, target: &str) -> Value {
            if let Some(client) = client {
                let result = call(client, "refs", json!({"name": target})).await;
                if result.is_error == Some(true) {
                    failure(&result)
                } else {
                    success(&result)
                }
            } else {
                let output = std::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
                    .current_dir(root)
                    .env("XDG_CONFIG_HOME", root.join("user-config"))
                    .args(["refs", target, "--json"])
                    .output()
                    .unwrap();
                serde_json::from_slice(&output.stdout).unwrap()
            }
        }
        let config = root.path().join(".deepkit/config.local.toml");
        let source = fs::read(root.path().join("src/lib.rs")).unwrap();
        let original = invoke(root.path(), client.as_ref(), "should_retry").await;
        assert_eq!(
            original["groups"].as_array().unwrap().len(),
            1,
            "{transport}: {original}"
        );
        fs::write(
            &config,
            "[search]\nlazy_resync=false\n[tokenize]\nsplit_subtokens=false\n",
        )
        .unwrap();
        let refused = invoke(root.path(), client.as_ref(), "should_retry").await;
        assert_eq!(refused["groups"], json!([]), "{transport}: {refused}");
        assert!(refused["total"].is_null());
        assert_eq!(refused["index"]["partial"], true);
        assert_eq!(refused["index"]["freshness_refusals"]["l1_config_stale"], 1);
        fs::write(&config, "[tokenize]\nsplit_subtokens=false\n").unwrap();
        let repaired = invoke(root.path(), client.as_ref(), "should_retry").await;
        assert_eq!(
            repaired["groups"].as_array().unwrap().len(),
            1,
            "{transport}: {repaired}"
        );
        assert_eq!(repaired["groups"], original["groups"]);
        assert_eq!(repaired["index"]["partial"], false);
        assert_eq!(repaired["index"]["lazy_resync"]["attempted_files"], 1);
        let retry = invoke(root.path(), client.as_ref(), "should_retry").await;
        assert_eq!(retry["groups"], repaired["groups"]);
        assert_eq!(retry["index"]["lazy_resync"]["attempted_files"], 0);
        assert_eq!(fs::read(root.path().join("src/lib.rs")).unwrap(), source);
        if let Some(client) = client {
            timeout(CALL_TIMEOUT, client.cancel())
                .await
                .unwrap()
                .unwrap();
            let child = child.as_mut().unwrap();
            if transport == "http" {
                child.kill().await.unwrap();
            } else {
                timeout(CALL_TIMEOUT, child.wait()).await.unwrap().unwrap();
            }
        }
    }
}

#[tokio::test]
async fn cli_and_both_mcp_maps_repair_config_only_staleness() {
    for (transport, empty) in ["cli", "stdio", "http"]
        .into_iter()
        .flat_map(|transport| [false, true].map(|empty| (transport, empty)))
    {
        let root = fixture(false);
        if empty {
            fs::write(root.path().join("src/lib.rs"), "").unwrap();
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
        }
        let (mut child, client) = match transport {
            "http" => {
                let address = format!("127.0.0.1:{}", free_port());
                let child = http_child(root.path(), &address, None).await;
                let wire = StreamableHttpClientTransport::from_uri(format!("http://{address}/mcp"));
                let client = timeout(CALL_TIMEOUT, ().serve(wire))
                    .await
                    .unwrap()
                    .unwrap();
                (Some(child), Some(client))
            }
            "stdio" => {
                let (child, client) = stdio(root.path()).await;
                (Some(child), Some(client))
            }
            _ => (None, None),
        };
        async fn invoke(root: &Path, client: Option<&Client>, target: &str) -> Value {
            if let Some(client) = client {
                let result = call(client, "map", json!({"focus": [target]})).await;
                if result.is_error == Some(true) {
                    failure(&result)
                } else {
                    success(&result)
                }
            } else {
                let output = std::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
                    .current_dir(root)
                    .env("XDG_CONFIG_HOME", root.join("user-config"))
                    .args(["map", "--focus", target, "--json"])
                    .output()
                    .unwrap();
                serde_json::from_slice(&output.stdout).unwrap()
            }
        }
        let config = root.path().join(".deepkit/config.local.toml");
        let source = fs::read(root.path().join("src/lib.rs")).unwrap();
        let original = invoke(root.path(), client.as_ref(), "src/lib.rs").await;
        assert_eq!(
            original["nodes"].as_array().unwrap().len(),
            1,
            "{transport}: {original}"
        );
        fs::write(
            &config,
            "[search]\nlazy_resync=false\n[tokenize]\nsplit_subtokens=false\n",
        )
        .unwrap();
        let refused = invoke(root.path(), client.as_ref(), "src/lib.rs").await;
        assert_eq!(refused["nodes"], json!([]), "{transport}: {refused}");
        assert_eq!(refused["tree"], "");
        assert_eq!(refused["index"]["partial"], true);
        assert_eq!(refused["index"]["freshness_refusals"]["l1_config_stale"], 1);
        fs::write(&config, "[tokenize]\nsplit_subtokens=false\n").unwrap();
        let repaired = invoke(root.path(), client.as_ref(), "src/lib.rs").await;
        assert_eq!(
            repaired["nodes"].as_array().unwrap().len(),
            1,
            "{transport}: {repaired}"
        );
        // A tokenizer-only repair preserves graph topology, symbol IDs and ranks.
        // Verify the full node identity across CLI and both MCP transports.
        assert_eq!(repaired["nodes"], original["nodes"]);
        assert_eq!(repaired["tree"], original["tree"]);
        assert_eq!(repaired["index"]["partial"], false);
        assert_eq!(repaired["index"]["lazy_resync"]["attempted_files"], 1);
        let retry = invoke(root.path(), client.as_ref(), "src/lib.rs").await;
        assert_eq!(retry["nodes"], repaired["nodes"]);
        assert_eq!(retry["index"]["lazy_resync"]["attempted_files"], 0);
        assert_eq!(fs::read(root.path().join("src/lib.rs")).unwrap(), source);
        if let Some(client) = client {
            timeout(CALL_TIMEOUT, client.cancel())
                .await
                .unwrap()
                .unwrap();
            let child = child.as_mut().unwrap();
            if transport == "http" {
                child.kill().await.unwrap();
            } else {
                timeout(CALL_TIMEOUT, child.wait()).await.unwrap().unwrap();
            }
        }
    }
}

#[tokio::test]
async fn cli_and_both_mcp_tools_refuse_pending_l1_and_repair_independently() {
    async fn invoke(root: &Path, client: Option<&Client>, tool: &'static str) -> Value {
        let (args, cli): (Value, &[&str]) = match tool {
            "search" => (
                json!({"query":"should_retry", "mode":"lexical", "scope":["src/lib.rs"]}),
                &[
                    "search",
                    "should_retry",
                    "--mode",
                    "lexical",
                    "--scope",
                    "src/lib.rs",
                ],
            ),
            "search_empty" => (
                json!({"query":"absentterm", "mode":"lexical"}),
                &["search", "absentterm", "--mode", "lexical"],
            ),
            "open" => (json!({"path":"src/lib.rs"}), &["open", "src/lib.rs"]),
            "symbol" => (json!({"name":"should_retry"}), &["symbol", "should_retry"]),
            "refs" => (json!({"name":"should_retry"}), &["refs", "should_retry"]),
            "map" => (
                json!({"focus":["src/lib.rs"]}),
                &["map", "--focus", "src/lib.rs"],
            ),
            "status" => (json!({}), &["status"]),
            _ => unreachable!(),
        };
        if let Some(client) = client {
            let result = call(
                client,
                if tool == "search_empty" {
                    "search"
                } else {
                    tool
                },
                args,
            )
            .await;
            if result.is_error == Some(true) {
                failure(&result)
            } else {
                success(&result)
            }
        } else {
            let output = std::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
                .current_dir(root)
                .env("XDG_CONFIG_HOME", root.join("user-config"))
                .args(cli)
                .arg("--json")
                .output()
                .unwrap();
            serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
                panic!(
                    "{tool}: {error}: {}",
                    String::from_utf8_lossy(&output.stderr)
                )
            })
        }
    }
    for transport in ["cli", "stdio", "http"] {
        let root = fixture(false);
        let (mut child, client) = match transport {
            "stdio" => {
                let (child, client) = stdio(root.path()).await;
                (Some(child), Some(client))
            }
            "http" => {
                let address = format!("127.0.0.1:{}", free_port());
                let child = http_child(root.path(), &address, None).await;
                let wire = StreamableHttpClientTransport::from_uri(format!("http://{address}/mcp"));
                let client = timeout(CALL_TIMEOUT, ().serve(wire))
                    .await
                    .unwrap()
                    .unwrap();
                (Some(child), Some(client))
            }
            _ => (None, None),
        };
        let config = root.path().join(".deepkit/config.local.toml");
        let source = fs::read(root.path().join("src/lib.rs")).unwrap();
        let initial = invoke(root.path(), client.as_ref(), "status").await;
        let db = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
        db.busy_timeout(CALL_TIMEOUT).unwrap();
        for lazy in [false, true] {
            fs::write(&config, format!("[search]\nlazy_resync={lazy}\n")).unwrap();
            db.execute("UPDATE files SET l1_pending=1 WHERE path='src/lib.rs'", [])
                .unwrap();
            let empty = invoke(root.path(), client.as_ref(), "search_empty").await;
            assert_eq!(empty["hits"], json!([]));
            assert_eq!(
                empty["index"]["partial"], true,
                "{transport}/{lazy}: {empty}"
            );
            assert_eq!(empty["index"]["degraded"], true);
            assert_eq!(empty["index"]["lexical"], "incomplete");
            assert_eq!(empty["index"]["fresh"]["reason"], "l1_pending");
            assert_eq!(empty["index"]["l1_pending_files"], 1);
            assert_eq!(empty["index"]["suppressed_hits"], 0);
            assert_eq!(empty["index"]["lazy_resync"]["attempted_files"], 0);
            assert_eq!(empty["index"]["head"], initial["head"]);
            assert_eq!(empty["index"]["indexed_head"], initial["indexed_head"]);
            assert_eq!(
                invoke(root.path(), client.as_ref(), "status").await["l1_pending_files"],
                1
            );
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
            let retry = invoke(root.path(), client.as_ref(), "search_empty").await;
            assert_eq!(retry["hits"], json!([]));
            assert_eq!(retry["index"]["partial"], false, "{transport}: {retry}");
            assert_eq!(retry["index"]["lexical"], "complete");
            assert_eq!(fs::read(root.path().join("src/lib.rs")).unwrap(), source);
        }
        for tool in ["search", "open", "symbol", "refs", "map"] {
            // Reset before each tool: no earlier call may repair this one's input.
            assert_eq!(
                db.execute("UPDATE files SET l1_pending=1 WHERE path='src/lib.rs'", [])
                    .unwrap(),
                1
            );
            fs::write(&config, "[search]\nlazy_resync=false\n").unwrap();
            let status = invoke(root.path(), client.as_ref(), "status").await;
            assert_eq!(
                status["l1_pending_files"], 1,
                "{transport}/{tool}: {status}"
            );
            assert_eq!(status["fresh"]["reason"], "l1_pending");
            assert_eq!(status["head"], initial["head"]);
            let refused = invoke(root.path(), client.as_ref(), tool).await;
            match tool {
                "open" | "symbol" => assert_eq!(
                    refused["error"]["code"], "source_unavailable",
                    "{transport}/{tool}: {refused}"
                ),
                _ => {
                    assert_eq!(
                        refused["index"]["partial"], true,
                        "{transport}/{tool}: {refused}"
                    );
                    let field = match tool {
                        "search" => "hits",
                        "refs" => "groups",
                        _ => "nodes",
                    };
                    assert_eq!(refused[field], json!([]));
                    if tool == "refs" {
                        assert!(refused["total"].is_null());
                    }
                    if tool == "map" {
                        assert_eq!(refused["tree"], "");
                    }
                }
            }
            assert_eq!(
                invoke(root.path(), client.as_ref(), "status").await["l1_pending_files"],
                1
            );
            fs::write(&config, "[search]\nlazy_resync=true\n").unwrap();
            let repaired = invoke(root.path(), client.as_ref(), tool).await;
            assert!(
                repaired["error"].is_null(),
                "{transport}/{tool}: {repaired}"
            );
            match tool {
                "open" => assert_eq!(
                    repaired["text"],
                    "pub fn should_retry(attempt: u32) -> bool { attempt < 3 }\n"
                ),
                "symbol" => assert_eq!(repaired[0]["path"], "src/lib.rs"),
                _ => {
                    assert_eq!(
                        repaired["index"]["partial"], false,
                        "{transport}/{tool}: {repaired}"
                    );
                    let field = match tool {
                        "search" => "hits",
                        "refs" => "groups",
                        _ => "nodes",
                    };
                    assert!(!repaired[field].as_array().unwrap().is_empty());
                    assert_eq!(repaired["index"]["lazy_resync"]["published_files"], 1);
                }
            }
            let settled = invoke(root.path(), client.as_ref(), "status").await;
            assert_eq!(settled["l1_pending_files"], 0);
            assert_eq!(settled["indexed_head"], initial["indexed_head"]);
            assert_eq!(settled["head"], initial["head"]);
            assert_eq!(settled["units"], initial["units"]);
            assert_eq!(fs::read(root.path().join("src/lib.rs")).unwrap(), source);
        }
        if let Some(client) = client {
            timeout(CALL_TIMEOUT, client.cancel())
                .await
                .unwrap()
                .unwrap();
            let child = child.as_mut().unwrap();
            if transport == "http" {
                child.kill().await.unwrap();
            } else {
                timeout(CALL_TIMEOUT, child.wait()).await.unwrap().unwrap();
            }
        }
    }
}

#[tokio::test]
async fn both_mcp_config_recent_and_status_check_head_without_loading_git_index() {
    for http in [false, true] {
        let root = fixture(true);
        let root = root.path();
        let (mut child, client) = if http {
            let address = format!("127.0.0.1:{}", free_port());
            let child = http_child(root, &address, None).await;
            let transport =
                StreamableHttpClientTransport::from_uri(format!("http://{address}/mcp"));
            (
                child,
                timeout(CALL_TIMEOUT, ().serve(transport))
                    .await
                    .unwrap()
                    .unwrap(),
            )
        } else {
            stdio(root).await
        };
        let before = success(&call(&client, "status", json!({})).await);
        let index_path = root.join(".git/index");
        let index = fs::read(&index_path).unwrap();
        let head_path = root.join(".git/HEAD");
        let head = fs::read(&head_path).unwrap();
        let local_path = root.join(".deepkit/config.local.toml");
        // Neither a decodable index nor even a checksum-sized one is needed
        // for these operations. Exercise each guard on the existing server.
        for corrupt in [vec![0xff; 64], b"short".to_vec()] {
            fs::write(&index_path, &corrupt).unwrap();
            for tool in ["config_get", "config_explain"] {
                let value =
                    success(&call(&client, tool, json!({"key":"search.graph_prior_weight"})).await);
                assert_eq!(value["value"], 0.3);
            }
            success(
                &call(
                    &client,
                    "config_set",
                    json!({"key":"search.graph_prior_weight", "value":0.3}),
                )
                .await,
            );
            assert_eq!(
                success(&call(&client, "recent", json!({})).await)[0]["title"],
                "fixture"
            );
            let status = success(&call(&client, "status", json!({})).await);
            assert_eq!(status["head"], before["head"]);
            assert_eq!(status["indexed_head"], before["indexed_head"]);
            assert_eq!(fs::read(&index_path).unwrap(), corrupt);
        }
        // Changed and unborn HEAD remain observable despite the corrupt index.
        fs::write(&head_path, b"ref: refs/heads/head-probe\n").unwrap();
        let status = success(&call(&client, "status", json!({})).await);
        assert!(status["head"].is_null());
        assert_eq!(status["indexed_head"], before["indexed_head"]);
        assert_eq!(status["fresh"]["reason"], "head_changed");
        assert_eq!(
            failure(&call(&client, "recent", json!({})).await)["error"]["code"],
            "source_unavailable"
        );
        for tool in ["config_get", "config_explain"] {
            success(&call(&client, tool, json!({"key":"search.graph_prior_weight"})).await);
        }
        // A resolvable symbolic HEAD with an invalid target must refuse every
        // guard, especially config_set, without modifying configuration.
        let ref_path = root.join(".git/refs/heads/head-probe");
        fs::write(&ref_path, b"invalid object id\n").unwrap();
        let local_before = fs::read(&local_path).unwrap();
        for (tool, args) in [
            ("config_get", json!({"key":"search.graph_prior_weight"})),
            ("config_explain", json!({"key":"search.graph_prior_weight"})),
            (
                "config_set",
                json!({"key":"search.graph_prior_weight", "value":0.7}),
            ),
            ("recent", json!({})),
            ("status", json!({})),
        ] {
            assert_eq!(
                failure(&call(&client, tool, args).await)["error"]["code"],
                if tool == "recent" {
                    "repository_unavailable"
                } else {
                    "tool_failed"
                },
                "{tool}"
            );
            assert_eq!(fs::read(&local_path).unwrap(), local_before);
        }
        fs::remove_file(ref_path).unwrap();
        fs::write(&head_path, &head).unwrap();
        // Index-root mistakes stay repairable through config, including while
        // Git's index is corrupt; recent/status still need an available DB.
        success(
            &call(
                &client,
                "config_set",
                json!({"key":"index.root", "value":"missing-directory"}),
            )
            .await,
        );
        for tool in ["config_get", "config_explain"] {
            assert_eq!(
                success(&call(&client, tool, json!({"key":"index.root"})).await)["value"],
                "missing-directory"
            );
        }
        for tool in ["recent", "status"] {
            assert_eq!(
                failure(&call(&client, tool, json!({})).await)["error"]["code"],
                "index_unavailable"
            );
        }
        success(
            &call(
                &client,
                "config_set",
                json!({"key":"index.root", "value":"."}),
            )
            .await,
        );
        // The existing agent-write policy remains authoritative on both transports.
        let local = fs::read(&local_path).unwrap();
        let mut locked = local.clone();
        locked.extend_from_slice(b"\n[serve]\nagent_writable_config=false\n");
        fs::write(&local_path, &locked).unwrap();
        let denied = failure(
            &call(
                &client,
                "config_set",
                json!({"key":"search.graph_prior_weight", "value":0.7}),
            )
            .await,
        );
        assert_eq!(denied["error"]["code"], "config_refused");
        assert_eq!(fs::read(&local_path).unwrap(), locked);
        fs::write(&local_path, local).unwrap();
        fs::write(&index_path, index).unwrap();
        let restored = success(&call(&client, "status", json!({})).await);
        assert_eq!(restored["head"], before["head"]);
        assert_eq!(restored["indexed_head"], before["indexed_head"]);
        assert_eq!(restored["units"], before["units"]);
        assert_eq!(restored["l1_scan_generation"], before["l1_scan_generation"]);
        assert_eq!(
            success(&call(&client, "recent", json!({})).await)[0]["title"],
            "fixture"
        );
        success(
            &call(
                &client,
                "config_set",
                json!({"key":"search.graph_prior_weight", "value":0.4}),
            )
            .await,
        );
        assert_eq!(
            success(
                &call(
                    &client,
                    "config_get",
                    json!({"key":"search.graph_prior_weight"})
                )
                .await
            )["value"],
            0.4
        );
        timeout(CALL_TIMEOUT, client.cancel())
            .await
            .unwrap()
            .unwrap();
        if http {
            child.kill().await.unwrap();
        } else {
            timeout(CALL_TIMEOUT, child.wait()).await.unwrap().unwrap();
        }
    }
}

#[tokio::test]
async fn recent_pr_filters_and_open_sources_match_cli_on_both_transports() {
    for http in [false, true] {
        let root = fixture(true);
        let path = root.path();
        fs::write(
            path.join("src/lib.rs"),
            "pub fn billing() -> bool { true }\n",
        )
        .unwrap();
        for args in [
            vec!["add", "src/lib.rs"],
            vec![
                "-c",
                "user.name=MCP Test",
                "-c",
                "user.email=mcp@example.invalid",
                "commit",
                "-q",
                "-m",
                "Billing change (#321)",
            ],
        ] {
            assert!(
                std::process::Command::new("git")
                    .current_dir(path)
                    .args(args)
                    .output()
                    .unwrap()
                    .status
                    .success()
            );
        }
        let cli = std::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
            .current_dir(path)
            .env("XDG_CONFIG_HOME", path.join("user-config"))
            .args(["index", "--level", "1", "--json"])
            .output()
            .unwrap();
        assert!(
            cli.status.success(),
            "{}",
            String::from_utf8_lossy(&cli.stderr)
        );
        let cli = std::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
            .current_dir(path)
            .env("XDG_CONFIG_HOME", path.join("user-config"))
            .args(["recent", "--symbol", "billing", "--json"])
            .output()
            .unwrap();
        assert!(cli.status.success());
        let expected: Value = serde_json::from_slice(&cli.stdout).unwrap();
        assert_eq!(expected[0]["kind"], "pr");
        let (mut child, client) = if http {
            let address = format!("127.0.0.1:{}", free_port());
            let child = http_child(path, &address, None).await;
            let transport =
                StreamableHttpClientTransport::from_uri(format!("http://{address}/mcp"));
            (
                child,
                timeout(CALL_TIMEOUT, ().serve(transport))
                    .await
                    .unwrap()
                    .unwrap(),
            )
        } else {
            stdio(path).await
        };
        assert_eq!(
            success(&call(&client, "recent", json!({"symbol":"billing"})).await),
            expected
        );
        let cli_symbol = std::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
            .current_dir(path)
            .env("XDG_CONFIG_HOME", path.join("user-config"))
            .args(["symbol", "billing", "--json"])
            .output()
            .unwrap();
        assert!(cli_symbol.status.success());
        let cli_symbol: Value = serde_json::from_slice(&cli_symbol.stdout).unwrap();
        let symbol = success(&call(&client, "symbol", json!({"name":"billing"})).await);
        assert_eq!(symbol[0]["history"], cli_symbol[0]["history"]);
        assert_eq!(
            symbol[0]["history"]["commits"][0]["pr"]["id"],
            expected[0]["id"]
        );
        let opened = success(&call(&client, "open", json!({"id":expected[0]["id"]})).await);
        assert_eq!(opened["kind"], "pr");
        assert_eq!(opened["sources"].as_array().unwrap().len(), 1);
        assert_eq!(
            success(
                &call(
                    &client,
                    "recent",
                    json!({"symbol":"billing","granularity":"commit"})
                )
                .await
            )[0]["kind"],
            "commit"
        );
        assert_eq!(
            failure(&call(&client, "recent", json!({"since":"invalid"})).await)["error"]["code"],
            "invalid_arguments"
        );
        client.cancel().await.unwrap();
        if http {
            child.kill().await.unwrap();
        } else {
            timeout(CALL_TIMEOUT, child.wait()).await.unwrap().unwrap();
        }
    }
}

#[cfg(unix)]
#[tokio::test]
async fn serve_signals_stop_both_transports_with_open_clients_and_allow_restart() {
    use std::os::unix::process::ExitStatusExt;
    for http in [false, true] {
        for signal in [2, 15] {
            let root = fixture(true);
            let address = format!("127.0.0.1:{}", free_port());
            let (mut child, client) = if http {
                let child = http_child(root.path(), &address, None).await;
                let transport =
                    StreamableHttpClientTransport::from_uri(format!("http://{address}/mcp"));
                let client = timeout(CALL_TIMEOUT, ().serve(transport))
                    .await
                    .unwrap()
                    .unwrap();
                (child, client)
            } else {
                stdio(root.path()).await
            };
            let before = success(
                &call(
                    &client,
                    "search",
                    json!({"query":"should_retry", "mode":"lexical", "rerank":false}),
                )
                .await,
            );
            assert!(!before["hits"].as_array().unwrap().is_empty());
            // Keep stdin open, or an incomplete HTTP body pending, through exit.
            // Shutdown must not wait for either client to finish its input.
            let _unfinished = if http {
                let mut stream = tokio::net::TcpStream::connect(&address).await.unwrap();
                stream
                    .write_all(
                        b"POST /mcp HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100000\r\n\r\n{",
                    )
                    .await
                    .unwrap();
                Some(stream)
            } else {
                None
            };
            let result = std::process::Command::new("kill")
                .args([format!("-{signal}"), child.id().unwrap().to_string()])
                .status()
                .unwrap();
            assert!(result.success());
            let status = timeout(Duration::from_secs(4), child.wait())
                .await
                .expect("serve must bound shutdown while client input remains open")
                .unwrap();
            assert_eq!(
                status.signal(),
                None,
                "http={http} signal={signal}: {status}"
            );
            assert!(status.success(), "http={http} signal={signal}: {status}");
            let _ = timeout(CALL_TIMEOUT, client.cancel()).await;
            if http {
                assert!(tokio::net::TcpStream::connect(&address).await.is_err());
            }
            let (mut restarted, client) = stdio(root.path()).await;
            let after = success(
                &call(
                    &client,
                    "search",
                    json!({"query":"should_retry", "mode":"lexical", "rerank":false}),
                )
                .await,
            );
            let stable_hits = |mut response: Value| {
                for hit in response["hits"].as_array_mut().unwrap() {
                    hit["fresh"]
                        .as_object_mut()
                        .unwrap()
                        .remove("changed_ago_s");
                }
                response["hits"].clone()
            };
            assert_eq!(stable_hits(before), stable_hits(after));
            timeout(CALL_TIMEOUT, client.cancel())
                .await
                .unwrap()
                .unwrap();
            assert!(
                timeout(CALL_TIMEOUT, restarted.wait())
                    .await
                    .unwrap()
                    .unwrap()
                    .success()
            );
        }
    }
}

#[tokio::test]
async fn serve_worker_publishes_and_retries_ready_jobs_over_both_transports() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };
    for http in [false, true] {
        let root = fixture(true);
        let provider = MockServer::start().await;
        let requests = Arc::new(AtomicUsize::new(0));
        let calls = Arc::clone(&requests);
        Mock::given(method("POST")).and(path("/api/embed"))
            .respond_with(move |request: &wiremock::Request| {
                if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    return ResponseTemplate::new(429).insert_header("Retry-After", "1");
                }
                let body: Value = serde_json::from_slice(&request.body).unwrap();
                ResponseTemplate::new(200).set_body_json(json!({"embeddings": vec![vec![1.0,0.0]; body["input"].as_array().unwrap().len()]}))
            }).mount(&provider).await;
        fs::write(root.path().join(".deepkit/config.toml"), format!(
            "[embedder]\nprofile='ollama'\ndims=2\n[embedder.ollama]\nurl='{}'\n[rerank]\nmode='off'\n[queue]\nmax_schedule_per_pass=1\nmax_jobs_per_pass=1\nembedding_timeout='2s'\n", provider.uri())).unwrap();
        let output = command(root.path())
            .args(["index", "--level", "1", "--json"])
            .output()
            .await
            .unwrap();
        assert!(output.status.success());
        let address = format!("127.0.0.1:{}", free_port());
        let (mut child, client) = if http {
            let child = http_child(root.path(), &address, None).await;
            let client = ()
                .serve(StreamableHttpClientTransport::from_uri(format!(
                    "http://{address}/mcp"
                )))
                .await
                .unwrap();
            (child, client)
        } else {
            stdio(root.path()).await
        };
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            let status = success(&call(&client, "status", json!({})).await);
            assert!(Instant::now() < deadline, "worker did not finish: {status}");
            if status["embedding_worker"]["state"] == "waiting" {
                let db = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
                let (done, pending): (i64, i64) = db
                    .query_row(
                        "SELECT sum(state='done'),sum(state!='done') FROM jobs",
                        [],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    )
                    .unwrap();
                if done > 0 && pending == 0 {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(requests.load(Ordering::SeqCst) > 1);
        let hits = success(
            &call(
                &client,
                "search",
                json!({"query":"should_retry", "mode":"lexical", "rerank":false}),
            )
            .await,
        );
        assert!(!hits["hits"].as_array().unwrap().is_empty());
        let db = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
        let units: i64 = db
            .query_row(
                "SELECT count(*) FROM units WHERE ladder_level=2",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(units > 0);
        if http {
            child.kill().await.unwrap();
        } else {
            client.cancel().await.unwrap();
        }
        timeout(CALL_TIMEOUT, child.wait()).await.unwrap().unwrap();
    }
}

#[tokio::test]
async fn lexical_search_survives_local_ollama_moonly_profile_switches() {
    for http in [false, true] {
        let root = fixture(true);
        let config_path = root.path().join(".deepkit/config.toml");
        let config = fs::read_to_string(&config_path).unwrap();
        // The fixture isolates lexical continuity: no model is loaded or called.
        fs::write(
            &config_path,
            format!("{config}\n[limits]\nmax_l2_units=0\n"),
        )
        .unwrap();
        let address = format!("127.0.0.1:{}", free_port());
        let (mut child, client) = if http {
            let child = http_child(root.path(), &address, None).await;
            let client = ()
                .serve(StreamableHttpClientTransport::from_uri(format!(
                    "http://{address}/mcp"
                )))
                .await
                .unwrap();
            (child, client)
        } else {
            stdio(root.path()).await
        };
        let query = json!({"query":"should_retry", "mode":"lexical", "rerank":false, "scope":["src/lib.rs"], "k":5});
        let baseline = success(&call(&client, "search", query.clone()).await);
        let paths = baseline["hits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|hit| hit["path"].clone())
            .collect::<Vec<_>>();
        assert!(paths.contains(&json!("src/lib.rs")), "{baseline}");
        for profile in ["local", "ollama", "remote"] {
            let ((), effective) = tokio::join!(
                async {
                    for _ in 0..20 {
                        let response = success(&call(&client, "search", query.clone()).await);
                        assert_ne!(response["index"]["partial"], true, "{profile}: {response}");
                        assert!(
                            response["hits"].as_array().unwrap().iter().all(|hit| {
                                matches!(
                                    hit["fresh"]["l2"].as_str(),
                                    Some("current" | "pending" | "none")
                                )
                            }),
                            "{profile}: {response}"
                        );
                        let observed = response["hits"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|hit| hit["path"].clone())
                            .collect::<Vec<_>>();
                        assert_eq!(observed, paths, "{profile}: {response}");
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                },
                async {
                    // Keep lexical calls active before, during and after each write.
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    success(
                        &call(
                            &client,
                            "config_set",
                            json!({"key":"embedder.profile", "value":profile, "layer":"local"}),
                        )
                        .await,
                    );
                    success(&call(&client, "config_get", json!({"key":"embedder.profile"})).await)
                }
            );
            assert_eq!(effective["value"], profile, "{effective}");
        }
        if http {
            child.kill().await.unwrap();
        } else {
            client.cancel().await.unwrap();
        }
        timeout(CALL_TIMEOUT, child.wait()).await.unwrap().unwrap();
    }
}

#[tokio::test]
async fn serve_worker_rebuilds_after_config_repair_and_cap_changes() {
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };
    for http in [false, true] {
        let root = fixture(true);
        let provider = MockServer::start().await;
        Mock::given(method("POST")).and(path("/api/embed"))
            .respond_with(|request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap();
                ResponseTemplate::new(200).set_body_json(json!({"embeddings": vec![vec![1.0,0.0]; body["input"].as_array().unwrap().len()]}))
            }).mount(&provider).await;
        let config = format!(
            "[embedder]\nprofile='ollama'\ndims=2\n[embedder.ollama]\nurl='{}'\n[rerank]\nmode='off'\n[queue]\nmax_jobs_per_pass=1\nmax_schedule_per_pass=1\n[limits]\nmax_l2_units=0\n",
            provider.uri()
        );
        let config_path = root.path().join(".deepkit/config.toml");
        fs::write(&config_path, &config).unwrap();
        assert!(
            command(root.path())
                .args(["index", "--level", "1"])
                .output()
                .await
                .unwrap()
                .status
                .success()
        );
        let address = format!("127.0.0.1:{}", free_port());
        let (mut child, client) = if http {
            let child = http_child(root.path(), &address, None).await;
            let client = ()
                .serve(StreamableHttpClientTransport::from_uri(format!(
                    "http://{address}/mcp"
                )))
                .await
                .unwrap();
            (child, client)
        } else {
            stdio(root.path()).await
        };
        let db = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
        let deadline = Instant::now() + CALL_TIMEOUT;
        loop {
            let status = success(&call(&client, "status", json!({})).await);
            let queued: i64 = db
                .query_row("SELECT count(*) FROM jobs", [], |row| row.get(0))
                .unwrap();
            let units: i64 = db
                .query_row("SELECT count(*) FROM units", [], |row| row.get(0))
                .unwrap();
            if queued == units && queued > 0 {
                break;
            }
            assert!(Instant::now() < deadline, "missing capped jobs: {status}");
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(provider.received_requests().await.unwrap().is_empty());
        let jobs: i64 = db
            .query_row("SELECT count(*) FROM jobs", [], |row| row.get(0))
            .unwrap();
        // A valid but unusable profile stops preparation without breaking tools.
        fs::write(
            &config_path,
            config.replace("profile='ollama'", "profile='local'"),
        )
        .unwrap();
        let deadline = Instant::now() + CALL_TIMEOUT;
        loop {
            let status = success(&call(&client, "status", json!({})).await);
            if status["embedding_worker"]["state"] == "stopped" {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "missing stopped worker: {status}"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        success(&call(&client, "config_get", json!({"key":"embedder.profile"})).await);
        fs::write(&config_path, &config).unwrap();
        success(
            &call(
                &client,
                "config_set",
                json!({"key":"limits.max_l2_units", "value":1000, "layer":"local"}),
            )
            .await,
        );
        let deadline = Instant::now() + CALL_TIMEOUT;
        loop {
            let status = success(&call(&client, "status", json!({})).await);
            let done: i64 = db
                .query_row("SELECT count(*) FROM jobs WHERE state='done'", [], |row| {
                    row.get(0)
                })
                .unwrap();
            if done == jobs && status["embedding_worker"]["state"] == "waiting" {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "configuration did not rebuild worker: {status}"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let (total, attempts): (i64, i64) = db
            .query_row("SELECT count(*),sum(attempts) FROM jobs", [], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .unwrap();
        assert_eq!((total, attempts), (jobs, jobs));
        let requests = provider.received_requests().await.unwrap().len();
        assert!(requests > 0);
        // Another effective settings rebuild must reuse completed generations.
        success(
            &call(
                &client,
                "config_set",
                json!({"key":"queue.max_jobs_per_pass", "value":2, "layer":"local"}),
            )
            .await,
        );
        tokio::time::sleep(Duration::from_millis(2200)).await;
        assert_eq!(provider.received_requests().await.unwrap().len(), requests);
        let hits = success(
            &call(
                &client,
                "search",
                json!({"query":"should_retry", "mode":"lexical", "rerank":false}),
            )
            .await,
        );
        assert!(!hits["hits"].as_array().unwrap().is_empty());
        if http {
            child.kill().await.unwrap();
        } else {
            client.cancel().await.unwrap();
        }
        timeout(CALL_TIMEOUT, child.wait()).await.unwrap().unwrap();
    }
}

#[tokio::test]
async fn serve_worker_rebuilds_after_head_and_catalogue_generations() {
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };
    fn git(root: &Path, args: &[&str]) {
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
    }
    async fn settled(client: &Client, db: &rusqlite::Connection, expected: i64) {
        let deadline = Instant::now() + CALL_TIMEOUT;
        loop {
            let status = success(&call(client, "status", json!({})).await);
            let covered: i64 = db
                .query_row(
                    "SELECT count(*) FROM units WHERE ladder_level=2",
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            if covered == expected && status["embedding_worker"]["state"] == "waiting" {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "worker did not discover generation: {covered}/{expected}: {status}"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }
    fn completed(db: &rusqlite::Connection) -> Vec<(i64, i64, i64)> {
        db.prepare("SELECT id,attempts,lease_generation FROM jobs WHERE state='done' ORDER BY id")
            .unwrap()
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .unwrap()
            .map(Result::unwrap)
            .collect()
    }
    for http in [false, true] {
        let root = fixture(true);
        fs::remove_file(root.path().join("guide.md")).unwrap();
        let provider = MockServer::start().await;
        Mock::given(method("POST")).and(path("/api/embed"))
            .respond_with(|request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap();
                ResponseTemplate::new(200).set_body_json(json!({"embeddings": vec![vec![1.0,0.0]; body["input"].as_array().unwrap().len()]}))
            }).mount(&provider).await;
        fs::write(root.path().join(".deepkit/config.toml"), format!(
            "[embedder]\nprofile='ollama'\ndims=2\n[embedder.ollama]\nurl='{}'\n[rerank]\nmode='off'\n[queue]\nmax_jobs_per_pass=1\nmax_schedule_per_pass=1\n", provider.uri())).unwrap();
        assert!(
            command(root.path())
                .args(["index", "--level", "1"])
                .output()
                .await
                .unwrap()
                .status
                .success()
        );
        let address = format!("127.0.0.1:{}", free_port());
        let (mut child, client) = if http {
            let child = http_child(root.path(), &address, None).await;
            let client = ()
                .serve(StreamableHttpClientTransport::from_uri(format!(
                    "http://{address}/mcp"
                )))
                .await
                .unwrap();
            (child, client)
        } else {
            stdio(root.path()).await
        };
        let db = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
        db.busy_timeout(CALL_TIMEOUT).unwrap();
        let count = || {
            db.query_row("SELECT count(*) FROM units", [], |row| row.get::<_, i64>(0))
                .unwrap()
        };
        settled(&client, &db, count()).await;
        let original = completed(&db);
        assert!(!original.is_empty());
        // Same HEAD and settings, but a new kind and new catalogue generation.
        fs::write(
            root.path().join("new.md"),
            "# New guide\n\nNew document publication must be embedded.\n",
        )
        .unwrap();
        assert!(
            command(root.path())
                .args(["index", "--level", "1"])
                .output()
                .await
                .unwrap()
                .status
                .success()
        );
        assert!(count() > original.len() as i64);
        settled(&client, &db, count()).await;
        let docs: i64 = db
            .query_row(
                "SELECT count(*) FROM units WHERE kind='doc_section' AND ladder_level=2",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(docs > 0);
        // Commit changes HEAD before L1. The worker must stop and keep tools usable.
        // Capture before the edit: the always-on watcher may publish its file
        // while the worker is stopped waiting for completed HEAD authority.
        let before = count();
        fs::write(
            root.path().join("src/new.rs"),
            "pub fn later_generation() -> bool { true }\n",
        )
        .unwrap();
        git(root.path(), &["add", "src/new.rs"]);
        git(
            root.path(),
            &[
                "-c",
                "user.name=MCP Test",
                "-c",
                "user.email=mcp@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "new generation",
            ],
        );
        let deadline = Instant::now() + CALL_TIMEOUT;
        loop {
            let status = success(&call(&client, "status", json!({})).await);
            if status["embedding_worker"]["state"] == "stopped" {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "HEAD mismatch did not stop worker: {status}"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        success(&call(&client, "config_get", json!({"key":"embedder.profile"})).await);
        assert!(
            command(root.path())
                .args(["index", "--level", "1"])
                .output()
                .await
                .unwrap()
                .status
                .success()
        );
        assert!(count() > before);
        settled(&client, &db, count()).await;
        // Rebuilding reuses completed generations; detaching the same commit is harmless.
        let after = completed(&db);
        for job in original {
            assert!(after.contains(&job), "completed job changed: {job:?}");
        }
        let requests = provider.received_requests().await.unwrap().len();
        git(root.path(), &["checkout", "--quiet", "--detach", "HEAD"]);
        assert!(
            command(root.path())
                .args(["index", "--level", "1"])
                .output()
                .await
                .unwrap()
                .status
                .success()
        );
        tokio::time::sleep(Duration::from_millis(2200)).await;
        settled(&client, &db, count()).await;
        assert_eq!(completed(&db), after);
        assert_eq!(provider.received_requests().await.unwrap().len(), requests);
        if http {
            child.kill().await.unwrap();
        } else {
            client.cancel().await.unwrap();
        }
        timeout(CALL_TIMEOUT, child.wait()).await.unwrap().unwrap();
    }
}

#[cfg(unix)]
#[tokio::test]
async fn surviving_serve_recovers_only_expired_index_lease_and_retains_uncertain_charge() {
    concurrent_index_crash(false, Duration::ZERO).await;
}

#[cfg(unix)]
#[tokio::test]
async fn surviving_serve_recovers_index_crash_after_decoded_provider_response() {
    concurrent_index_crash(true, Duration::ZERO).await;
}

#[cfg(unix)]
#[tokio::test]
async fn surviving_serve_recovers_seeded_crash_timing_in_observed_response_windows() {
    // Fixed seeds make failures replayable. Sample within each observed window;
    // this does not sample the unguarded settlement/publication race.
    for seed in [0x19_u64, 0x20260907, 0xd1ce] {
        let hash = blake3::hash(&seed.to_le_bytes());
        for (slot, after_response) in [false, true].into_iter().enumerate() {
            let millis = 1 + u64::from(hash.as_bytes()[slot]) * 2;
            eprintln!("crash seed={seed:#x} after_response={after_response} delay_ms={millis}");
            concurrent_index_crash(after_response, Duration::from_millis(millis)).await;
        }
    }
}

#[cfg(unix)]
async fn concurrent_index_crash(after_response: bool, kill_delay: Duration) {
    concurrent_index_crash_window(after_response, kill_delay, false).await;
}

#[cfg(unix)]
#[tokio::test]
async fn surviving_serve_recovers_unguarded_settlement_and_publication_crashes() {
    let mut killed = 0;
    for seed in [0x19_u64, 0x20260907, 0xd1ce] {
        let hash = blake3::hash(&seed.to_le_bytes());
        let micros = u64::from(hash.as_bytes()[0]) * 10;
        eprintln!("unguarded crash seed={seed:#x} delay_us={micros}");
        killed += concurrent_index_crash_window(true, Duration::from_micros(micros), true).await;
    }
    // A later control admits retained publication or clean completion, logging
    // normal exits separately from actual SIGKILL evidence.
    killed += concurrent_index_crash_window(true, Duration::from_millis(50), true).await;
    assert!(
        killed > 0,
        "all samples exited normally; no SIGKILL evidence"
    );
}

#[cfg(unix)]
async fn concurrent_index_crash_window(
    after_response: bool,
    kill_delay: Duration,
    unguarded: bool,
) -> usize {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };
    let mut killed_count = 0;
    for http in [false, true] {
        let root = fixture(true);
        if unguarded {
            // Exactly one occurrence per owner keeps the observed outcome tied
            // to the decoded request, even if index finishes before the signal.
            fs::remove_dir_all(root.path().join("src")).unwrap();
            fs::remove_dir_all(root.path().join("alt")).unwrap();
            fs::remove_file(root.path().join("guide.md")).unwrap();
            fs::create_dir(root.path().join("src")).unwrap();
            fs::write(
                root.path().join("src/lib.rs"),
                "pub fn should_retry() -> bool { true }\n",
            )
            .unwrap();
            fs::write(
                root.path().join("src/other.rs"),
                "pub fn other() -> bool { false }\n",
            )
            .unwrap();
        }
        let provider = MockServer::start().await;
        let requests = Arc::new(AtomicUsize::new(0));
        let calls = Arc::clone(&requests);
        Mock::given(method("POST")).and(path("/api/embed"))
            .respond_with(move |request: &wiremock::Request| {
                let ordinal = calls.fetch_add(1, Ordering::SeqCst);
                let body: Value = serde_json::from_slice(&request.body).unwrap();
                let response = ResponseTemplate::new(200).set_body_json(json!({"embeddings": vec![vec![1.0,0.0]; body["input"].as_array().unwrap().len()]}));
                if ordinal < 2 { response.set_delay(Duration::from_secs(4)) } else { response }
            }).mount(&provider).await;
        fs::write(root.path().join(".deepkit/config.toml"), format!(
            "[embedder]\nprofile='ollama'\ndims=2\n[embedder.ollama]\nurl='{}'\nconcurrency=2\n[rerank]\nmode='off'\n[queue]\nembedding_timeout='10s'\nmax_jobs_per_pass=1\nmax_schedule_per_pass=1\n", provider.uri())).unwrap();
        if unguarded {
            // This test needs exactly two jobs, one per owner. Reset its prepared
            // fixture with history disabled; other crash cases retain history.
            let config_path = root.path().join(".deepkit/config.toml");
            fs::write(
                &config_path,
                format!(
                    "{}\n[commits]\nenabled=false\n",
                    fs::read_to_string(&config_path).unwrap()
                ),
            )
            .unwrap();
            assert!(
                command(root.path())
                    .args(["clean", "--json"])
                    .output()
                    .await
                    .unwrap()
                    .status
                    .success()
            );
        }
        assert!(
            command(root.path())
                .args(["index", "--level", "1"])
                .output()
                .await
                .unwrap()
                .status
                .success()
        );
        let address = format!("127.0.0.1:{}", free_port());
        let (mut child, client) = if http {
            let child = http_child(root.path(), &address, None).await;
            let client = ()
                .serve(StreamableHttpClientTransport::from_uri(format!(
                    "http://{address}/mcp"
                )))
                .await
                .unwrap();
            (child, client)
        } else {
            stdio(root.path()).await
        };
        let db = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
        db.busy_timeout(CALL_TIMEOUT).unwrap();
        let count = |sql: &str| db.query_row(sql, [], |row| row.get::<_, i64>(0)).unwrap();
        let initial_generation =
            count("SELECT CAST(value AS INTEGER) FROM meta WHERE key='l1_scan_generation'");
        let deadline = Instant::now() + CALL_TIMEOUT;
        while requests.load(Ordering::SeqCst) == 0 {
            assert!(Instant::now() < deadline, "serve made no request");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        // The index process republishes L1 while serve owns an active request.
        let response_log = tempfile::NamedTempFile::new().unwrap();
        let mut index = command(root.path())
            .args(["index", "--level", "2", "--json"])
            .stdout(Stdio::piped())
            .env("DKC_LOG", "dkc::queue::worker=trace")
            .stderr(response_log.reopen().unwrap())
            .spawn()
            .unwrap();
        let index_pid = index.id().unwrap();
        let serve_pid = child.id().unwrap();
        let deadline = Instant::now() + CALL_TIMEOUT;
        while requests.load(Ordering::SeqCst) < 2 {
            assert!(
                index.try_wait().unwrap().is_none(),
                "index exited before overlap"
            );
            assert!(Instant::now() < deadline, "second process made no request");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let owners: Vec<String> = db
            .prepare("SELECT lease_owner FROM jobs WHERE state='leased' ORDER BY id")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(owners.len(), 2, "{owners:?}");
        assert!(
            owners
                .iter()
                .any(|owner| owner.starts_with(&format!("{serve_pid}:"))),
            "{owners:?}"
        );
        assert!(
            owners
                .iter()
                .any(|owner| owner.starts_with(&format!("{index_pid}:"))),
            "{owners:?}"
        );
        assert!(
            count("SELECT CAST(value AS INTEGER) FROM meta WHERE key='l1_scan_generation'")
                > initial_generation
        );
        assert_eq!(
            count("SELECT sum(calls) FROM usage_reservations WHERE state='dispatched'"),
            2
        );

        let abandoned: (i64, i64, i64, String) = db.query_row(
            "SELECT id,lease_generation,attempts,lease_owner FROM jobs WHERE lease_owner LIKE ?1",
            [format!("{index_pid}:%")],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        ).unwrap();
        let reservation: (String, i64, i64) = db.query_row(
            "SELECT id,calls,tokens FROM usage_reservations WHERE job_id=?1 AND state='dispatched'",
            [abandoned.0],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        ).unwrap();
        if after_response {
            // Hold the writer before the delayed response arrives. The child-side
            // trace is emitted only after successful decoding, before settlement;
            // responder invocation alone cannot establish HTTP delivery.
            db.execute_batch("BEGIN IMMEDIATE").unwrap();
            let deadline = Instant::now() + CALL_TIMEOUT;
            loop {
                let log = fs::read_to_string(response_log.path()).unwrap();
                if log.contains("queue provider response decoded") {
                    break;
                }
                assert!(index.try_wait().unwrap().is_none(), "index exited: {log}");
                assert!(Instant::now() < deadline, "no decoded response: {log}");
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            assert_eq!(count("SELECT count(*) FROM jobs WHERE state='leased'"), 2);
            assert_eq!(
                count("SELECT count(*) FROM usage_reservations WHERE state='reconciled'"),
                0
            );
        }
        if unguarded {
            db.execute_batch("ROLLBACK").unwrap();
        }
        tokio::time::sleep(kill_delay).await;
        let log = fs::read_to_string(response_log.path()).unwrap();
        assert_eq!(
            log.contains("queue provider response decoded"),
            after_response
        );
        if !unguarded {
            assert_eq!(
                count("SELECT count(*) FROM usage_reservations WHERE state='reconciled'"),
                0
            );
        }
        use std::os::unix::process::ExitStatusExt;
        let killed = if let Some(status) = index.try_wait().unwrap() {
            assert!(
                unguarded && status.success(),
                "unexpected early exit: {status}"
            );
            false
        } else {
            index.kill().await.unwrap();
            let status = index.wait().await.unwrap();
            assert!(
                status.signal() == Some(9) || (unguarded && status.success()),
                "unexpected exit during kill: {status}"
            );
            status.signal() == Some(9)
        };
        assert_eq!(
            fs::read_to_string(response_log.path())
                .unwrap()
                .contains("queue provider response decoded"),
            after_response,
            "response boundary changed before confirmed death"
        );
        killed_count += usize::from(killed);
        if !unguarded {
            assert_eq!(count("SELECT count(*) FROM jobs WHERE state='leased'"), 2);
        }
        if after_response && !unguarded {
            db.execute_batch("ROLLBACK").unwrap();
        }
        let job_state: String = db
            .query_row("SELECT state FROM jobs WHERE id=?1", [abandoned.0], |r| {
                r.get(0)
            })
            .unwrap();
        assert!(
            matches!(job_state.as_str(), "leased" | "done"),
            "{job_state}"
        );
        let pending = i64::from(job_state == "leased");
        let charge_state: String = db
            .query_row(
                "SELECT state FROM usage_reservations WHERE id=?1",
                [&reservation.0],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            matches!(charge_state.as_str(), "dispatched" | "reconciled"),
            "{charge_state}"
        );
        if pending == 0 {
            assert_eq!(charge_state, "reconciled");
        }
        if !unguarded {
            assert_eq!(pending, 1);
            assert_eq!(charge_state, "dispatched");
        }
        eprintln!(
            "crash outcome http={http} unguarded={unguarded} delay_us={} killed={killed} job={job_state} charge={charge_state}",
            kill_delay.as_micros()
        );
        // Killing one owner must neither release its uncertain charge nor disturb
        // the other owner's still-live request.
        assert!(child.try_wait().unwrap().is_none());
        let total = count("SELECT count(*) FROM units");
        if unguarded {
            assert_eq!(total, 2);
        }
        let deadline = Instant::now() + CALL_TIMEOUT;
        loop {
            let status = success(&call(&client, "status", json!({})).await);
            if count("SELECT count(*) FROM jobs WHERE state='done'") == total - pending
                && status["embedding_worker"]["state"] == "waiting"
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "survivor did not settle: {status}"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert_eq!(requests.load(Ordering::SeqCst), total as usize);
        let lease = || {
            db.query_row(
                "SELECT id,lease_generation,attempts,lease_owner FROM jobs WHERE state='leased'",
                [],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                },
            )
            .unwrap()
        };
        if pending == 1 {
            assert_eq!(lease(), abandoned);
        }
        let charge = || {
            db.query_row(
                "SELECT state,calls,tokens FROM usage_reservations WHERE id=?1",
                [&reservation.0],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                },
            )
            .unwrap()
        };
        assert_eq!(
            charge(),
            (charge_state.clone(), reservation.1, reservation.2)
        );
        let hits = success(
            &call(
                &client,
                "search",
                json!({"query":"should_retry","mode":"lexical","rerank":false}),
            )
            .await,
        );
        assert!(!hits["hits"].as_array().unwrap().is_empty());
        let slug: String = db
            .query_row("SELECT slug FROM embedding_tables", [], |row| row.get(0))
            .unwrap();
        assert!(slug.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'));
        let snapshot = || {
            db.prepare(&format!(
                "SELECT unit_id,v,ctx_hash FROM vec_{slug} ORDER BY unit_id"
            ))
            .unwrap()
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                ))
            })
            .unwrap()
            .map(Result::unwrap)
            .collect::<Vec<_>>()
        };
        // Completion and its vector must agree, including a crash between the
        // separate accounting and completion transactions.
        assert_eq!(
            count(&format!(
                "SELECT count(*) FROM jobs j LEFT JOIN vec_{slug} v ON v.unit_id=j.target_id WHERE (j.state='done') != (v.unit_id IS NOT NULL)"
            )),
            0
        );
        let retained = snapshot();
        assert_eq!(retained.len(), (total - pending) as usize);
        // An immediate public restart also respects the abandoned live lease.
        let output = command(root.path())
            .args(["index", "--level", "2", "--json"])
            .output()
            .await
            .unwrap();
        assert!(output.status.success());
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            report["embedding"]["state"],
            if pending == 1 { "pending" } else { "quiescent" },
            "{report}"
        );
        if pending == 1 {
            assert_eq!(lease(), abandoned);
        }
        assert_eq!(snapshot(), retained);
        assert_eq!(requests.load(Ordering::SeqCst), total as usize);
        // Explicit fixture expiry advances ONLY dead ownership, without changing
        // production timeouts or waiting for the 120-second lease to elapse.
        assert_eq!(
            db.execute(
                "UPDATE jobs SET leased_until=0 WHERE id=?1 AND state='leased' AND lease_owner=?2",
                rusqlite::params![abandoned.0, abandoned.3],
            )
            .unwrap(),
            pending as usize
        );
        let deadline = Instant::now() + CALL_TIMEOUT;
        loop {
            let status = success(&call(&client, "status", json!({})).await);
            if count("SELECT count(*) FROM jobs WHERE state='done'") == total
                && status["embedding_worker"]["state"] == "waiting"
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "expired ownership not recovered: {status}"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(child.try_wait().unwrap().is_none(), "survivor exited");
        assert_eq!(requests.load(Ordering::SeqCst), (total + pending) as usize);
        assert_eq!(count("SELECT count(*) FROM jobs WHERE state!='done'"), 0);
        let recovered: (i64, i64) = db
            .query_row(
                "SELECT lease_generation,attempts FROM jobs WHERE id=?1",
                [abandoned.0],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(recovered, (abandoned.1 + pending, abandoned.2 + pending));
        assert_eq!(
            count("SELECT count(*) FROM jobs WHERE attempts!=1"),
            pending
        );
        assert_eq!(
            charge(),
            (charge_state.clone(), reservation.1, reservation.2)
        );
        assert_eq!(
            count("SELECT sum(calls) FROM usage_reservations WHERE state='reconciled'"),
            total
                + if charge_state == "reconciled" {
                    pending
                } else {
                    0
                }
        );
        assert_eq!(
            count("SELECT count(*) FROM usage_reservations WHERE state='reserved'"),
            0
        );
        assert_eq!(
            count("SELECT count(*) FROM usage_reservations WHERE state='dispatched'"),
            i64::from(charge_state == "dispatched")
        );
        assert_eq!(
            count("SELECT sum(calls) FROM usage"),
            total
                + if charge_state == "reconciled" {
                    pending
                } else {
                    0
                }
        );
        assert_eq!(
            count("SELECT sum(tokens) FROM usage"),
            count("SELECT sum(tokens) FROM usage_reservations WHERE state='reconciled'")
        );
        let final_vectors = snapshot();
        assert_eq!(final_vectors.len(), total as usize);
        for vector in retained {
            assert!(final_vectors.contains(&vector));
        }
        let output = command(root.path())
            .args(["index", "--level", "2", "--json"])
            .output()
            .await
            .unwrap();
        assert!(output.status.success());
        tokio::time::sleep(Duration::from_millis(2200)).await;
        assert_eq!(requests.load(Ordering::SeqCst), (total + pending) as usize);
        assert_eq!(snapshot(), final_vectors);
        assert_eq!(
            charge(),
            (charge_state.clone(), reservation.1, reservation.2)
        );
        if http {
            child.kill().await.unwrap();
        } else {
            client.cancel().await.unwrap();
        }
        timeout(CALL_TIMEOUT, child.wait()).await.unwrap().unwrap();
    }
    killed_count
}

#[tokio::test]
async fn public_index_and_serve_share_live_leases_budget_and_publications() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };
    for http in [false, true] {
        let root = fixture(true);
        let provider = MockServer::start().await;
        let requests = Arc::new(AtomicUsize::new(0));
        let calls = Arc::clone(&requests);
        Mock::given(method("POST")).and(path("/api/embed"))
            .respond_with(move |request: &wiremock::Request| {
                let ordinal = calls.fetch_add(1, Ordering::SeqCst);
                let body: Value = serde_json::from_slice(&request.body).unwrap();
                let response = ResponseTemplate::new(200).set_body_json(json!({"embeddings": vec![vec![1.0,0.0]; body["input"].as_array().unwrap().len()]}));
                if ordinal < 2 { response.set_delay(Duration::from_secs(4)) } else { response }
            }).mount(&provider).await;
        fs::write(root.path().join(".deepkit/config.toml"), format!(
            "[embedder]\nprofile='ollama'\ndims=2\n[embedder.ollama]\nurl='{}'\nconcurrency=2\n[rerank]\nmode='off'\n[queue]\nembedding_timeout='10s'\nmax_jobs_per_pass=1\nmax_schedule_per_pass=1\n[[queue.budgets]]\nprofile='ollama'\nurl='{}'\ndaily_calls=2\n", provider.uri(), provider.uri())).unwrap();
        assert!(
            command(root.path())
                .args(["index", "--level", "1"])
                .output()
                .await
                .unwrap()
                .status
                .success()
        );
        let address = format!("127.0.0.1:{}", free_port());
        let (mut child, client) = if http {
            let child = http_child(root.path(), &address, None).await;
            let client = ()
                .serve(StreamableHttpClientTransport::from_uri(format!(
                    "http://{address}/mcp"
                )))
                .await
                .unwrap();
            (child, client)
        } else {
            stdio(root.path()).await
        };
        let db = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
        db.busy_timeout(CALL_TIMEOUT).unwrap();
        let count = |sql: &str| db.query_row(sql, [], |row| row.get::<_, i64>(0)).unwrap();
        let initial_generation =
            count("SELECT CAST(value AS INTEGER) FROM meta WHERE key='l1_scan_generation'");
        let deadline = Instant::now() + CALL_TIMEOUT;
        while requests.load(Ordering::SeqCst) == 0 {
            assert!(Instant::now() < deadline, "serve made no request");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        // The index process republishes L1 while serve owns an active request.
        let mut index = command(root.path())
            .args(["index", "--level", "2", "--json"])
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let index_pid = index.id().unwrap();
        let serve_pid = child.id().unwrap();
        let deadline = Instant::now() + CALL_TIMEOUT;
        while requests.load(Ordering::SeqCst) < 2 {
            assert!(
                index.try_wait().unwrap().is_none(),
                "index exited before overlap"
            );
            assert!(Instant::now() < deadline, "second process made no request");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let owners: Vec<String> = db
            .prepare("SELECT lease_owner FROM jobs WHERE state='leased' ORDER BY id")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(owners.len(), 2, "{owners:?}");
        assert!(
            owners
                .iter()
                .any(|owner| owner.starts_with(&format!("{serve_pid}:"))),
            "{owners:?}"
        );
        assert!(
            owners
                .iter()
                .any(|owner| owner.starts_with(&format!("{index_pid}:"))),
            "{owners:?}"
        );
        assert!(
            count("SELECT CAST(value AS INTEGER) FROM meta WHERE key='l1_scan_generation'")
                > initial_generation
        );
        assert_eq!(
            count("SELECT sum(calls) FROM usage_reservations WHERE state='dispatched'"),
            2
        );
        let hits = success(
            &call(
                &client,
                "search",
                json!({"query":"should_retry", "mode":"lexical", "rerank":false}),
            )
            .await,
        );
        assert!(!hits["hits"].as_array().unwrap().is_empty());
        let output = timeout(Duration::from_secs(20), index.wait_with_output())
            .await
            .unwrap()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stdout)
        );
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["embedding"]["state"], "pending", "{report}");
        let deadline = Instant::now() + CALL_TIMEOUT;
        loop {
            let status = success(&call(&client, "status", json!({})).await);
            if count("SELECT count(*) FROM jobs WHERE state='done'") == 2
                && status["embedding_worker"]["state"] == "waiting"
            {
                break;
            }
            assert!(Instant::now() < deadline, "budget did not settle: {status}");
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        assert_eq!(count("SELECT count(*) FROM jobs WHERE state='leased'"), 0);
        assert_eq!(
            count("SELECT sum(calls) FROM usage_reservations WHERE state='reconciled'"),
            2
        );
        assert_eq!(
            count(
                "SELECT count(*) FROM jobs WHERE state='done' AND attempts=1 AND lease_generation=1"
            ),
            2
        );
        let slug: String = db
            .query_row("SELECT slug FROM embedding_tables", [], |row| row.get(0))
            .unwrap();
        assert!(slug.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'));
        let snapshot = || {
            db.prepare(&format!(
                "SELECT unit_id,v,ctx_hash FROM vec_{slug} ORDER BY unit_id"
            ))
            .unwrap()
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                ))
            })
            .unwrap()
            .map(Result::unwrap)
            .collect::<Vec<_>>()
        };
        let retained = snapshot();
        assert_eq!(retained.len(), 2);
        let delay = count("SELECT sum(not_before) FROM jobs WHERE state='queued'");
        assert!(delay > 0);
        success(
            &call(
                &client,
                "config_set",
                json!({"key":"queue.budgets", "value":[], "layer":"local"}),
            )
            .await,
        );
        tokio::time::sleep(Duration::from_millis(2200)).await;
        assert_eq!(
            count("SELECT sum(not_before) FROM jobs WHERE state='queued'"),
            delay
        );
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        // Explicit fixture clock advancement: changing budgets must not reset delays.
        db.execute("UPDATE jobs SET not_before=0 WHERE state='queued'", [])
            .unwrap();
        let total = count("SELECT count(*) FROM units");
        let deadline = Instant::now() + CALL_TIMEOUT;
        loop {
            let status = success(&call(&client, "status", json!({})).await);
            if count("SELECT count(*) FROM jobs WHERE state='done'") == total
                && status["embedding_worker"]["state"] == "waiting"
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "shared queue did not finish: {status}"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert_eq!(requests.load(Ordering::SeqCst), total as usize);
        assert_eq!(
            count("SELECT sum(calls) FROM usage_reservations WHERE state='reconciled'"),
            total
        );
        assert_eq!(
            count(
                "SELECT count(*) FROM usage_reservations WHERE state IN ('reserved','dispatched')"
            ),
            0
        );
        assert_eq!(
            count("SELECT count(*) FROM jobs WHERE state!='done' OR attempts!=1"),
            0
        );
        let final_vectors = snapshot();
        assert_eq!(final_vectors.len(), total as usize);
        for vector in retained {
            assert!(final_vectors.contains(&vector));
        }
        let output = command(root.path())
            .args(["index", "--level", "2", "--json"])
            .output()
            .await
            .unwrap();
        assert!(output.status.success());
        tokio::time::sleep(Duration::from_millis(2200)).await;
        assert_eq!(snapshot(), final_vectors);
        assert_eq!(requests.load(Ordering::SeqCst), total as usize);
        if http {
            child.kill().await.unwrap();
        } else {
            client.cancel().await.unwrap();
        }
        timeout(CALL_TIMEOUT, child.wait()).await.unwrap().unwrap();
    }
}

#[tokio::test]
async fn serve_worker_recovers_root_changed_during_response_without_losing_accounting() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };
    for http in [false, true] {
        let root = fixture(true);
        fs::create_dir(root.path().join("elsewhere")).unwrap();
        let provider = MockServer::start().await;
        let config = format!(
            "[embedder]\nprofile='ollama'\ndims=2\n[embedder.ollama]\nurl='{}'\n[rerank]\nmode='off'\n[queue]\nmax_jobs_per_pass=1\nmax_schedule_per_pass=1\n",
            provider.uri()
        );
        let config_path = root.path().join(".deepkit/config.toml");
        fs::write(&config_path, &config).unwrap();
        assert!(
            command(root.path())
                .args(["index", "--level", "1"])
                .output()
                .await
                .unwrap()
                .status
                .success()
        );
        let requests = Arc::new(AtomicUsize::new(0));
        let calls = Arc::clone(&requests);
        let changed = format!("{config}\n[index]\nroot='elsewhere'\n");
        Mock::given(method("POST")).and(path("/api/embed"))
            .respond_with(move |request: &wiremock::Request| {
                if calls.fetch_add(1, Ordering::SeqCst) == 1 {
                    fs::write(&config_path, &changed).unwrap();
                }
                let body: Value = serde_json::from_slice(&request.body).unwrap();
                ResponseTemplate::new(200).set_body_json(json!({"embeddings": vec![vec![1.0,0.0]; body["input"].as_array().unwrap().len()]}))
            }).mount(&provider).await;
        let address = format!("127.0.0.1:{}", free_port());
        let (mut child, client) = if http {
            let child = http_child(root.path(), &address, None).await;
            let client = ()
                .serve(StreamableHttpClientTransport::from_uri(format!(
                    "http://{address}/mcp"
                )))
                .await
                .unwrap();
            (child, client)
        } else {
            stdio(root.path()).await
        };
        timeout(CALL_TIMEOUT, async {
            while requests.load(Ordering::SeqCst) < 2 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        // Allow the failed-root rebuild to settle; config tools remain available
        // even though status cannot open that nonexistent index.
        tokio::time::sleep(Duration::from_millis(2200)).await;
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        success(&call(&client, "config_get", json!({"key":"index.root"})).await);
        let db = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
        let count = |sql: &str| db.query_row(sql, [], |row| row.get::<_, i64>(0)).unwrap();
        assert_eq!(count("SELECT count(*) FROM units WHERE ladder_level=2"), 1);
        assert_eq!(count("SELECT sum(calls) FROM usage"), 2);
        assert_eq!(count("SELECT count(*) FROM jobs WHERE state='leased'"), 0);
        let retained: (i64, String, Vec<u8>) = db
            .query_row(
                "SELECT unit_id,table_slug,desired_hash FROM embedding_publications",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        success(
            &call(
                &client,
                "config_set",
                json!({"key":"index.root", "value":".", "layer":"local"}),
            )
            .await,
        );
        let units = count("SELECT count(*) FROM units");
        let deadline = Instant::now() + CALL_TIMEOUT;
        loop {
            let status = success(&call(&client, "status", json!({})).await);
            if count("SELECT count(*) FROM units WHERE ladder_level=2") == units
                && status["embedding_worker"]["state"] == "waiting"
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "root repair did not resume: {status}"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert_eq!(requests.load(Ordering::SeqCst) as i64, units + 1);
        assert_eq!(count("SELECT sum(calls) FROM usage"), units + 1);
        assert_eq!(count("SELECT count(*) FROM jobs WHERE state='leased'"), 0);
        let after: (i64, String, Vec<u8>) = db.query_row("SELECT unit_id,table_slug,desired_hash FROM embedding_publications WHERE unit_id=?1", [retained.0], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?))).unwrap();
        assert_eq!(after, retained);
        if http {
            child.kill().await.unwrap();
        } else {
            client.cancel().await.unwrap();
        }
        timeout(CALL_TIMEOUT, child.wait()).await.unwrap().unwrap();
    }
}

#[tokio::test]
async fn serve_worker_preparation_failure_keeps_tools_available() {
    let root = fixture(true);
    let (mut child, client) = stdio(root.path()).await;
    let deadline = Instant::now() + CALL_TIMEOUT;
    loop {
        let status = success(&call(&client, "status", json!({})).await);
        if status["embedding_worker"]["state"] == "stopped" {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "missing worker failure: {status}"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    let hits = success(
        &call(
            &client,
            "search",
            json!({"query":"should_retry", "mode":"lexical", "rerank":false}),
        )
        .await,
    );
    assert!(!hits["hits"].as_array().unwrap().is_empty());
    success(&call(&client, "config_get", json!({"key":"embedder.profile"})).await);
    client.cancel().await.unwrap();
    assert!(
        timeout(CALL_TIMEOUT, child.wait())
            .await
            .unwrap()
            .unwrap()
            .success()
    );
}

#[cfg(unix)]
#[tokio::test]
async fn serve_worker_shutdown_cancels_requests_and_retains_committed_vectors() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };
    for (http, signal) in [(false, 2), (true, 15), (false, 0)] {
        let root = fixture(true);
        let provider = MockServer::start().await;
        let requests = Arc::new(AtomicUsize::new(0));
        let calls = Arc::clone(&requests);
        Mock::given(method("POST")).and(path("/api/embed"))
            .respond_with(move |request: &wiremock::Request| {
                let ordinal = calls.fetch_add(1, Ordering::SeqCst);
                let body: Value = serde_json::from_slice(&request.body).unwrap();
                let response = ResponseTemplate::new(200).set_body_json(json!({"embeddings": vec![vec![1.0,0.0]; body["input"].as_array().unwrap().len()]}));
                if ordinal == 0 { response } else { response.set_delay(Duration::from_secs(30)) }
            }).mount(&provider).await;
        fs::write(root.path().join(".deepkit/config.toml"), format!(
            "[embedder]\nprofile='ollama'\ndims=2\n[embedder.ollama]\nurl='{}'\n[rerank]\nmode='off'\n[queue]\nmax_schedule_per_pass=1\nmax_jobs_per_pass=1\nembedding_timeout='60s'\n", provider.uri())).unwrap();
        let output = command(root.path())
            .args(["index", "--level", "1", "--json"])
            .output()
            .await
            .unwrap();
        assert!(output.status.success());
        let address = format!("127.0.0.1:{}", free_port());
        let (mut child, client) = if http {
            let child = http_child(root.path(), &address, None).await;
            let client = ()
                .serve(StreamableHttpClientTransport::from_uri(format!(
                    "http://{address}/mcp"
                )))
                .await
                .unwrap();
            (child, client)
        } else {
            stdio(root.path()).await
        };
        timeout(CALL_TIMEOUT, async {
            while requests.load(Ordering::SeqCst) < 2 {
                assert!(child.try_wait().unwrap().is_none());
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        let status = success(&call(&client, "status", json!({})).await);
        assert_eq!(status["embedding_worker"]["state"], "running");
        assert_eq!(status["queue"]["done"], 1);
        let db = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
        let done = || {
            db.query_row("SELECT count(*) FROM jobs WHERE state='done'", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap()
        };
        let before = done();
        db.execute_batch("BEGIN IMMEDIATE").unwrap();
        if signal == 0 {
            client.cancel().await.unwrap();
        } else {
            assert!(
                std::process::Command::new("kill")
                    .args([format!("-{signal}"), child.id().unwrap().to_string()])
                    .status()
                    .unwrap()
                    .success()
            );
        }
        assert!(
            timeout(Duration::from_secs(4), child.wait())
                .await
                .expect("worker must not delay transport shutdown")
                .unwrap()
                .success()
        );
        db.execute_batch("ROLLBACK").unwrap();
        assert_eq!(done(), before);
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        let leased: i64 = db
            .query_row(
                "SELECT count(*) FROM jobs WHERE state='leased'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(leased, 1);
        let vectors: i64 = db
            .query_row(
                "SELECT count(*) FROM units WHERE ladder_level=2",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(vectors, 1);
    }
}

#[tokio::test]
async fn linker_documentation_and_test_references_match_cli_on_both_transports() {
    for http in [false, true] {
        let root = fixture(true);
        let path = root.path();
        fs::write(
            path.join("README.md"),
            "# Retry guide\nCall `should_retry`.\n",
        )
        .unwrap();
        fs::create_dir(path.join("tests")).unwrap();
        fs::write(
            path.join("tests/retry.rs"),
            "fn retry_test() { let result = should_retry(1); assert!(result); }\n",
        )
        .unwrap();
        let cli = |args: &[&str]| {
            let output = std::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
                .current_dir(path)
                .env("XDG_CONFIG_HOME", path.join("user-config"))
                .args(args)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            serde_json::from_slice::<Value>(&output.stdout).unwrap()
        };
        cli(&["index", "--level", "1", "--json"]);
        let expected_symbol = cli(&["symbol", "should_retry", "--json"]);
        let expected_refs = cli(&["refs", "should_retry", "--tests-only", "--json"]);
        assert_eq!(expected_symbol[0]["tests"], json!(["tests/retry.rs"]));
        assert_eq!(
            expected_symbol[0]["documented_in"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        let (mut child, client) = if http {
            let address = format!("127.0.0.1:{}", free_port());
            let child = http_child(path, &address, None).await;
            let client = timeout(
                CALL_TIMEOUT,
                ().serve(StreamableHttpClientTransport::from_uri(format!(
                    "http://{address}/mcp"
                ))),
            )
            .await
            .unwrap()
            .unwrap();
            (child, client)
        } else {
            stdio(path).await
        };
        let actual = success(&call(&client, "symbol", json!({"name":"should_retry"})).await);
        for key in ["tests", "documented_in"] {
            assert_eq!(actual[0][key], expected_symbol[0][key]);
        }
        let refs = success(
            &call(
                &client,
                "refs",
                json!({"name":"should_retry","tests_only":true}),
            )
            .await,
        );
        assert_eq!(refs["groups"], expected_refs["groups"]);
        timeout(CALL_TIMEOUT, client.cancel())
            .await
            .unwrap()
            .unwrap();
        if http {
            child.kill().await.unwrap();
        } else {
            timeout(CALL_TIMEOUT, child.wait()).await.unwrap().unwrap();
        }
    }
}

#[tokio::test]
async fn serve_chat_runs_without_embeddings_and_restarts_without_duplicate_calls() {
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };
    for (http, embeddings) in [(false, false), (true, false), (false, true), (true, true)] {
        let root = fixture(true);
        let provider = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices":[{"message":{"content":"Verified public transport summary."}}],
                "usage":{"prompt_tokens":30,"completion_tokens":10,"total_tokens":40}
            })))
            .mount(&provider)
            .await;
        Mock::given(method("POST")).and(path("/api/embed"))
            .respond_with(|request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap();
                ResponseTemplate::new(200).set_body_json(json!({"embeddings":vec![vec![1.0,0.0];body["input"].as_array().unwrap().len()]}))
            }).mount(&provider).await;
        let embedder = if embeddings {
            format!(
                "[embedder]\nprofile='ollama'\ndims=2\n[embedder.ollama]\nurl='{}'\n",
                provider.uri()
            )
        } else {
            "[embedder]\nprofile='local'\n[embedder.local]\nmodel='qwen3-embedding-0.6b-q8_0'\n"
                .into()
        };
        fs::write(root.path().join(".deepkit/config.toml"), format!(
            "{embedder}[rerank]\nmode='off'\n[llm]\nenabled=true\nurl='{}/v1'\nmodel='fixture-text'\napi_key_env='DKC_MCP_CHAT_FIXTURE_KEY'\n[enrich]\nmodules_min_files=1\n[queue]\nmax_schedule_per_pass=1\n[[treatments]]\nname='source'\nmatch=['**']\nmax_level=3\nenrich=['describe_symbol']\n", provider.uri())).unwrap();
        let output = command(root.path())
            .args(["index", "--level", "1", "--json"])
            .output()
            .await
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let db = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
        let mut calls = 0;
        for repeat in 0..2 {
            let address = format!("127.0.0.1:{}", free_port());
            let (mut child, client) = if http {
                let child = http_child(root.path(), &address, None).await;
                let client = ()
                    .serve(StreamableHttpClientTransport::from_uri(format!(
                        "http://{address}/mcp"
                    )))
                    .await
                    .unwrap();
                (child, client)
            } else {
                stdio(root.path()).await
            };
            let deadline = Instant::now() + Duration::from_secs(20);
            loop {
                let status = success(&call(&client, "status", json!({})).await);
                if Instant::now() >= deadline {
                    let retained = root.keep();
                    panic!(
                        "chat worker did not finish; retained {}: {status}",
                        retained.display()
                    );
                }
                if status["enrichment_worker"]["state"] == "waiting" {
                    let summaries: i64 = db
                        .query_row("SELECT count(*) FROM units WHERE kind='summary'", [], |r| {
                            r.get(0)
                        })
                        .unwrap();
                    let embedded: i64 = db.query_row("SELECT count(DISTINCT u.id) FROM units u JOIN embedding_publications p ON p.unit_id=u.id WHERE u.kind='summary'",[],|r|r.get(0)).unwrap();
                    if summaries > 0
                        && (!embeddings
                            || (embedded == summaries
                                && status["embedding_worker"]["state"] == "waiting"))
                    {
                        break;
                    }
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            let search_result = call(&client,"search",json!({
                "query":"Verified public transport summary", "mode":"lexical", "rerank":false, "verbose":true
            })).await;
            if search_result.is_error == Some(true) {
                let retained = root.keep();
                panic!(
                    "summary search failed; retained {}: {search_result:?}",
                    retained.display()
                );
            }
            let search = success(&search_result);
            eprintln!(
                "chat summary search: http={http} embeddings={embeddings} repeat={repeat} took_ms={} partial={}",
                search["took_ms"], search["index"]["partial"]
            );
            let hits = search["hits"].as_array().unwrap();
            if hits.is_empty() {
                let retained = root.keep();
                panic!(
                    "published summaries must be immediately searchable; http={http} embeddings={embeddings} repeat={repeat}; retained {}: {search}",
                    retained.display()
                );
            }
            assert!(hits.iter().any(|hit| {
                hit["why"]
                    .as_array()
                    .is_some_and(|why| why.contains(&json!("summary")))
            }));
            assert!(
                hits.iter().any(|hit| hit["summary"]
                    .as_str()
                    .is_some_and(|text| text.starts_with("Verified"))),
                "{search}"
            );
            let symbols = success(&call(&client, "symbol", json!({"name":"should_retry"})).await);
            assert_eq!(symbols[0]["summary"], "Verified public transport summary.");
            let opened = success(&call(&client, "open", json!({"id":symbols[0]["id"]})).await);
            assert_eq!(opened["summary"], "Verified public transport summary.");
            assert!(
                opened["text"]
                    .as_str()
                    .unwrap()
                    .contains("pub fn should_retry")
            );
            let mapped = success(&call(&client, "map", json!({"budget_tokens":2000})).await);
            assert!(
                mapped["tree"]
                    .as_str()
                    .unwrap()
                    .contains("Verified public transport summary."),
                "{mapped}"
            );
            assert!(
                hits.iter().all(|hit| matches!(
                    hit["kind"].as_str(),
                    Some("symbol" | "chunk" | "module" | "pr")
                )),
                "summary matches must report their parents: {search}"
            );
            let received = provider
                .received_requests()
                .await
                .unwrap()
                .into_iter()
                .filter(|request| request.url.path() == "/v1/chat/completions")
                .collect::<Vec<_>>();
            if repeat == 0 {
                assert!(!received.is_empty());
                calls = received.len();
            } else {
                assert_eq!(received.len(), calls);
            }
            let summaries: i64 = db
                .query_row("SELECT count(*) FROM units WHERE kind='summary'", [], |r| {
                    r.get(0)
                })
                .unwrap();
            assert_eq!(summaries as usize, calls);
            let vectors: i64 = db
                .query_row("SELECT count(*) FROM units WHERE ladder_level=2", [], |r| {
                    r.get(0)
                })
                .unwrap();
            assert_eq!(vectors > 0, embeddings);
            if repeat == 1 && http && embeddings {
                let summary_proofs = || {
                    db.prepare("SELECT source_key,content_hash FROM units WHERE kind='summary' ORDER BY source_key")
                        .unwrap().query_map([],|r|Ok((r.get::<_,String>(0)?,r.get::<_,Vec<u8>>(1)?)))
                        .unwrap().collect::<Result<Vec<_>,_>>().unwrap()
                };
                let before = summary_proofs();
                fs::create_dir_all(root.path().join(".deepkit/prompts")).unwrap();
                fs::write(
                    root.path().join(".deepkit/prompts/describe_symbol.md"),
                    "Revised description: {{signature}} {{code}} {{callers}} {{docs}}",
                )
                .unwrap();
                let deadline = Instant::now() + Duration::from_secs(20);
                loop {
                    let changed = provider
                        .received_requests()
                        .await
                        .unwrap()
                        .into_iter()
                        .filter(|request| request.url.path() == "/v1/chat/completions")
                        .count();
                    let status = success(&call(&client, "status", json!({})).await);
                    let after = summary_proofs();
                    let refreshed = before.len() == after.len()
                        && before
                            .iter()
                            .zip(&after)
                            .all(|((old_key, old_hash), (key, hash))| {
                                old_key == key && old_hash != hash
                            });
                    assert!(
                        Instant::now() < deadline,
                        "active prompt override was not published: {status}"
                    );
                    if changed > calls
                        && refreshed
                        && status["enrichment_worker"]["state"] == "waiting"
                        && status["embedding_worker"]["state"] == "waiting"
                    {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(25)).await;
                }
                let missing:i64=db.query_row("SELECT count(*) FROM units u WHERE u.kind='summary' AND NOT EXISTS(SELECT 1 FROM embedding_publications p WHERE p.unit_id=u.id)",[],|r|r.get(0)).unwrap();
                assert_eq!(
                    missing, 0,
                    "refreshed summaries must receive replacement vectors"
                );
            }
            if http {
                child.kill().await.unwrap();
            } else {
                client.cancel().await.unwrap();
            }
            timeout(CALL_TIMEOUT, child.wait()).await.unwrap().unwrap();
        }
    }
}
