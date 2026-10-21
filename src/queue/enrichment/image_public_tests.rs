//! Public image indexing, vector publication and parent search; v1 §6.8/§9.
use super::image_tests;
use serde_json::{Value, json};
use std::{fs, path::Path};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

pub(super) async fn run(root: &Path, args: &[&str]) -> Value {
    let output = tokio::process::Command::new(assert_cmd::cargo::cargo_bin("dkc"))
        .current_dir(root)
        .env("DKC_TEST_IMAGE_PUBLIC_KEY", "fixture")
        .args(args)
        .output()
        .await
        .unwrap();
    if !output.status.success() {
        let status = tokio::process::Command::new(assert_cmd::cargo::cargo_bin("dkc"))
            .current_dir(root)
            .env("DKC_TEST_IMAGE_PUBLIC_KEY", "fixture")
            .args(["queue", "status", "--json"])
            .output()
            .await
            .unwrap();
        panic!(
            "command {args:?}: {}\nstdout:{}\nqueue:{}",
            String::from_utf8_lossy(&output.stderr),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&status.stdout)
        );
    }
    serde_json::from_slice(&output.stdout).unwrap()
}

#[tokio::test]
async fn public_image_index_embeds_descriptions_and_search_projects_to_images() {
    let server = MockServer::start().await;
    Mock::given(method("POST")).and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices":[{"message":{"content":"Quartz lantern diagram connects the request handlers."}}],
            "usage":{"prompt_tokens":30,"completion_tokens":10,"total_tokens":40},
        }))).expect(2).mount(&server).await;
    Mock::given(method("POST")).and(path("/v1/embeddings"))
        .respond_with(|request: &wiremock::Request| {
            let body: Value = serde_json::from_slice(&request.body).unwrap();
            let count = body["input"].as_array().unwrap().len();
            ResponseTemplate::new(200).set_body_json(json!({
                "data": (0..count).map(|index| json!({"index":index,"embedding":"AACAPwAAAAA="})).collect::<Vec<_>>(),
                "usage":{"prompt_tokens":10,"total_tokens":10},
            }))
        }).mount(&server).await;
    let (root, _env) = image_tests::fixture(&server.uri(), |c| {
        // This checks projection and freshness, not latency. Leave room for
        // subprocess scheduling when the full suite shares a hosted runner.
        c.limits.query_timeout_ms = 10_000;
        c.llm.api_key_env = "DKC_TEST_IMAGE_PUBLIC_KEY".into();
        c.llm.concurrency = 1;
        c.enrich.top_symbols = 0;
        c.enrich.prs_last_n = 0;
        c.enrich.modules_min_files = 1_000_000;
        c.commits.enabled = false;
        c.embedder.profile = "remote".into();
        c.embedder.dims = 2;
        c.embedder.remote.url = format!("{}/v1", server.uri());
        c.embedder.remote.api_key_env = "DKC_TEST_IMAGE_PUBLIC_KEY".into();
        c.embedder.remote.encoding_format = "base64".into();
    });
    fs::write(root.path().join("a-unreferenced.png"), image_tests::png()).unwrap();
    fs::write(
        root.path().join("icon.svg"),
        "<svg xmlns=\"http://www.w3.org/2000/svg\"><text>quartz</text></svg>",
    )
    .unwrap();
    let mut first_calls = None;
    for repeat in 0..2 {
        let report = run(root.path(), &["index", "--level", "3", "--json"]).await;
        assert_eq!(report["enrichment"]["state"], "complete", "{report}");
        assert_eq!(
            report["enrichment"]["published"],
            if repeat == 0 { 2 } else { 0 },
            "{report}"
        );
        let count = server.received_requests().await.unwrap().len();
        if let Some(previous) = first_calls {
            assert_eq!(count, previous, "repeat repeated model work");
        }
        first_calls = Some(count);
    }
    let calls = server.received_requests().await.unwrap();
    let chats = calls
        .iter()
        .filter(|request| request.url.path().ends_with("chat/completions"))
        .collect::<Vec<_>>();
    assert_eq!(chats.len(), 2);
    assert!(String::from_utf8_lossy(&chats[0].body).contains("Request flow"));
    assert!(!String::from_utf8_lossy(&chats[1].body).contains("Request flow"));
    assert!(
        calls
            .iter()
            .any(|request| request.url.path().ends_with("embeddings")
                && String::from_utf8_lossy(&request.body).contains("Quartz lantern diagram"))
    );
    let found = run(
        root.path(),
        &[
            "search",
            "quartz lantern",
            "--mode",
            "lexical",
            "--no-rerank",
            "--json",
        ],
    )
    .await;
    let hits = found["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 2, "{found}");
    for hit in hits {
        assert!(
            matches!(
                hit["path"].as_str(),
                Some("diagram.png" | "a-unreferenced.png")
            ),
            "{hit}"
        );
        assert_eq!(hit["kind"], "image", "{hit}");
        assert_eq!(hit["fresh"]["l2"], "current", "{hit}");
        assert!(hit["why"].to_string().contains("summary"), "{hit}");
    }
    // A changed caption invalidates only the description that used it. Search
    // must preserve the independently described, unreferenced image.
    fs::write(
        root.path().join("README.md"),
        "# Updated caption\n\nA revised ![caption](diagram.png).\n",
    )
    .unwrap();
    let changed = run(
        root.path(),
        &[
            "search",
            "quartz lantern",
            "--mode",
            "lexical",
            "--no-rerank",
            "--json",
        ],
    )
    .await;
    let remaining = changed["hits"].as_array().unwrap();
    assert_eq!(remaining.len(), 1, "{changed}");
    assert_eq!(remaining[0]["path"], "a-unreferenced.png", "{changed}");
    let db = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
    assert_eq!(
        db.query_row("SELECT kind FROM files WHERE path='icon.svg'", [], |r| {
            r.get::<_, String>(0)
        })
        .unwrap(),
        "other"
    );
    assert_eq!(db.query_row("SELECT count(*) FROM embedding_publications e JOIN units u ON u.id=e.unit_id WHERE u.kind='summary'", [], |r| r.get::<_,i64>(0)).unwrap(), 2);
    assert_eq!(db.query_row("SELECT count(*) FROM embedding_publications e JOIN units u ON u.id=e.unit_id WHERE u.kind='image'", [], |r| r.get::<_,i64>(0)).unwrap(), 0);
}
