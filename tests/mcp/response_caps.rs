//! Public whole-result caps and process-local status over CLI and both MCP transports.
use super::*;
fn limits(root: &Path, maximum: u64, deadline: u64) {
    fs::write(root.join(".deepkit/config.local.toml"),format!("[limits]\nmax_result_tokens={maximum}\nquery_timeout_ms={deadline}\nmax_open_bytes='513 B'\n")).unwrap();
}
fn tokens(value: &Value) -> u64 {
    tiktoken_rs::cl100k_base_singleton()
        .encode_with_special_tokens(&value.to_string())
        .len() as u64
}
// Wall-clock age is recomputed on every call. Keep every identity, location,
// content, score and freshness-state assertion exact without requiring two
// public requests to finish in the same second.
fn assert_same_hit_prefix(actual: &Value, expected: &[Value]) {
    let rows = actual.as_array().unwrap();
    assert_eq!(rows.len(), expected.len());
    let mut expected = expected.to_vec();
    for (actual, expected) in rows.iter().zip(&mut expected) {
        assert!(actual["fresh"]["changed_ago_s"].is_u64());
        expected["fresh"]["changed_ago_s"] = actual["fresh"]["changed_ago_s"].clone();
    }
    assert_eq!(actual, &json!(expected));
}
async fn exercise(client: &Client, root: &Path) {
    limits(root, 0, 5000);
    let search = success(
        &call(
            client,
            "search",
            json!({"query":"scope_probe","mode":"lexical","k":5}),
        )
        .await,
    );
    assert_eq!(search["hits"], json!([]));
    assert_eq!(search["result_budget"]["truncated"], true);
    let refs = success(&call(client, "refs", json!({"name":"should_retry"})).await);
    assert_eq!(refs["groups"], json!([]));
    assert_eq!(refs["returned_total"], 0);
    assert_eq!(refs["result_budget"]["truncated"], true);
    let map = success(&call(client, "map", json!({"budget_tokens":2000})).await);
    assert_eq!(map["nodes"], json!([]));
    assert_eq!(map["tree"], "");
    assert_eq!(map["result_budget"]["truncated"], true);
    assert_eq!(
        success(&call(client, "symbol", json!({"name":"should_retry"})).await),
        json!([])
    );
    assert_eq!(
        success(
            &call(
                client,
                "recent",
                json!({"granularity":"commit","since":"365d"})
            )
            .await
        ),
        json!([])
    );
    let page = success(&call(client, "open", json!({"path":"src/expanded.rs"})).await);
    assert!(page["continuation_id"].is_string());
    let status = success(&call(client, "status", json!({})).await);
    assert_eq!(status["response_caps"]["lifetime"], "process");
    assert_eq!(
        status["response_caps"]["events"]["max_result_tokens"]["count"],
        5
    );
    assert_eq!(
        status["response_caps"]["events"]["max_open_bytes"]["count"],
        1
    );
    limits(root, 10000, 0);
    for (tool, args) in [
        ("search", json!({"query":"scope_probe","mode":"lexical"})),
        ("refs", json!({"name":"should_retry"})),
        ("map", json!({})),
    ] {
        let result = success(&call(client, tool, args).await);
        assert_eq!(result["index"]["reason"], "query_timeout");
    }
    for (tool, args) in [
        ("open", json!({"path":"src/lib.rs"})),
        ("symbol", json!({"name":"should_retry"})),
        ("recent", json!({"granularity":"commit"})),
    ] {
        assert_eq!(
            failure(&call(client, tool, args).await)["error"]["code"],
            "timeout"
        );
    }
    let status = success(&call(client, "status", json!({})).await);
    assert_eq!(
        status["response_caps"]["events"]["query_timeout_ms"]["count"],
        6
    );
    assert!(status["response_caps"]["events"]["rerank_timeout_ms"].is_null());
    // The same server retains counters across configuration generations, while
    // raising the result cap permits the original complete ranked objects again.
    limits(root, 10000, 5000);
    let full = success(
        &call(
            client,
            "search",
            json!({"query":"scope_probe","mode":"lexical","k":5}),
        )
        .await,
    );
    let original = full["hits"].as_array().unwrap();
    assert!(original.len() > 1, "fixture needs competing ranked results");
    let maximum = tokens(&json!(&original[..1]));
    assert!(tokens(&json!(&original[..2])) > maximum);
    limits(root, maximum, 5000);
    let capped = success(
        &call(
            client,
            "search",
            json!({"query":"scope_probe","mode":"lexical","k":5}),
        )
        .await,
    );
    assert_same_hit_prefix(&capped["hits"], &original[..1]);
    assert!(tokens(&capped["hits"]) <= maximum);
    let status = success(&call(client, "status", json!({})).await);
    assert_eq!(
        status["response_caps"]["events"]["max_result_tokens"]["count"],
        6
    );
    assert!(!root.join("absent-model-cache").exists());
}
#[tokio::test]
async fn both_transports_enforce_whole_results_and_observe_caps_across_configuration_reload() {
    for http in [false, true] {
        let root = fixture(false);
        let (mut child, client) = if http {
            let address = format!("127.0.0.1:{}", free_port());
            let child = http_child(root.path(), &address, None).await;
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
            stdio(root.path()).await
        };
        exercise(&client, root.path()).await;
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
async fn cli_result_budget_keeps_rank_prefixes_and_zero_returns_empty_payloads() {
    let root = fixture(false);
    let root = root.path();
    limits(root, 10000, 5000);
    let query = ["search", "scope_probe", "--mode", "lexical", "--json"];
    let output = command(root).args(query).output().await.unwrap();
    assert!(output.status.success());
    let full: Value = serde_json::from_slice(&output.stdout).unwrap();
    let original = full["hits"].as_array().unwrap();
    assert!(original.len() > 1, "fixture needs competing ranked results");
    let maximum = tokens(&json!(&original[..1]));
    assert!(tokens(&json!(&original[..2])) > maximum);
    limits(root, maximum, 5000);
    let output = command(root).args(query).output().await.unwrap();
    assert!(output.status.success());
    let capped: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_same_hit_prefix(&capped["hits"], &original[..1]);
    assert!(tokens(&capped["hits"]) <= maximum);
    // A nonzero map cap must trim complete files and rebuild its text as well.
    limits(root, 10000, 5000);
    let output = command(root)
        .args(["map", "--json"])
        .output()
        .await
        .unwrap();
    assert!(output.status.success());
    let full_map: Value = serde_json::from_slice(&output.stdout).unwrap();
    let map_max = tokens(&json!({"tree":full_map["tree"],"nodes":full_map["nodes"]})) - 1;
    limits(root, map_max, 5000);
    let output = command(root)
        .args(["map", "--json"])
        .output()
        .await
        .unwrap();
    assert!(output.status.success());
    let map: Value = serde_json::from_slice(&output.stdout).unwrap();
    let nodes = map["nodes"].as_array().unwrap();
    assert!(!nodes.is_empty());
    assert!(nodes.len() < full_map["nodes"].as_array().unwrap().len());
    assert_eq!(
        map["nodes"],
        json!(&full_map["nodes"].as_array().unwrap()[..nodes.len()])
    );
    assert!(tokens(&json!({"tree":map["tree"],"nodes":map["nodes"]})) <= map_max);
    for node in nodes {
        assert!(
            map["tree"].as_str().unwrap().contains(
                Path::new(node["path"].as_str().unwrap())
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
            )
        );
    }
    assert_ne!(map["tree"], full_map["tree"]);
    limits(root, 0, 5000);
    for (args, key) in [
        (
            vec!["search", "scope_probe", "--mode", "lexical", "--json"],
            "hits",
        ),
        (vec!["refs", "should_retry", "--json"], "groups"),
        (vec!["map", "--json"], "nodes"),
    ] {
        let output = command(root).args(args).output().await.unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let result: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result[key], json!([]));
        assert_eq!(result["result_budget"]["truncated"], true);
    }
}
