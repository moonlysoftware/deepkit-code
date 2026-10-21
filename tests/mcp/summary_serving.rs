//! Verified summary shapes, complete PR sources and generation-bound pages; v1 §10.1.
use super::*;
use std::collections::BTreeSet;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

#[tokio::test]
async fn pr_summary_pages_keep_every_source_and_refuse_changed_prompt_on_both_transports() {
    for http in [false, true] {
        let root = fixture(true);
        let git = |args: &[&str]| {
            let output = std::process::Command::new("git")
                .current_dir(root.path())
                .args([
                    "-c",
                    "user.name=MCP Test",
                    "-c",
                    "user.email=mcp@example.invalid",
                ])
                .args(args)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            String::from_utf8(output.stdout).unwrap().trim().to_owned()
        };
        let base = git(&["rev-parse", "HEAD"]);
        git(&["checkout", "-q", "-b", "summary-notes"]);
        for (body, message) in [
            ("First note.\n", "First notes"),
            ("Second note.\n", "More notes"),
        ] {
            fs::write(root.path().join("notes.md"), body).unwrap();
            for args in [
                vec!["add", "notes.md"],
                vec![
                    "-c",
                    "user.name=MCP Test",
                    "-c",
                    "user.email=mcp@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    message,
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
        }
        git(&["checkout", "-q", "--detach", &base]);
        git(&[
            "merge",
            "--no-ff",
            "summary-notes",
            "-m",
            "Merge pull request #321 from summary-notes",
        ]);
        let provider = MockServer::start().await;
        let summary = "Summary of the two commits. Café 🙂.\n".repeat(12);
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices":[{"message":{"content":summary}}],
                "usage":{"prompt_tokens":30,"completion_tokens":120,"total_tokens":150}
            })))
            .mount(&provider)
            .await;
        // This contract needs L1 sources and mocked PR summaries, not a model
        // download or native embedding work competing with deadline tests.
        let endpoint = provider.uri();
        fs::write(root.path().join(".deepkit/config.toml"), format!(
            "[embedder]\nprofile='remote'\ndims=2\n[embedder.remote]\nurl='{endpoint}/v1'\nmodel='fixture-embed'\napi_key_env='DKC_MCP_CHAT_FIXTURE_KEY'\n[rerank]\nmode='off'\n[llm]\nenabled=true\nurl='{endpoint}/v1'\nmodel='fixture-text'\napi_key_env='DKC_MCP_CHAT_FIXTURE_KEY'\n[enrich]\ntop_symbols=0\nmodules_min_files=9999\n[limits]\nmax_open_bytes='128 B'\nmax_l2_units=0\n[[treatments]]\nname='source'\nmatch=['**']\nmax_level=3\nenrich=['describe_symbol']\n")).unwrap();
        let output = command(root.path())
            .args(["index", "--level", "3", "--json"])
            .output()
            .await
            .unwrap();
        assert!(
            output.status.success(),
            "{} {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(provider.received_requests().await.unwrap().len(), 1);
        assert!(!root.path().join("absent-model-cache").exists());
        assert!(
            provider
                .received_requests()
                .await
                .unwrap()
                .iter()
                .all(|request| request.url.path() == "/v1/chat/completions")
        );
        let db = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
        let expected_sources = db.prepare("SELECT 'u_'||substr(lower(hex(u.hash)),1,12) FROM prs p JOIN unit_sources s ON s.unit_id=p.unit_id AND s.source_kind='commit' JOIN commits c ON c.id=s.source_id JOIN units u ON u.id=c.unit_id WHERE p.number=321")
            .unwrap().query_map([], |r|r.get::<_,String>(0)).unwrap().collect::<Result<BTreeSet<_>,_>>().unwrap();
        assert_eq!(expected_sources.len(), 2);
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
        let recent = success(&call(&client, "recent", json!({"granularity":"pr"})).await);
        let pr = recent
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["kind"] == "pr")
            .unwrap();
        assert_eq!(pr["summary"], summary);
        let mut page = success(&call(&client, "open", json!({"id":pr["id"]})).await);
        let cursor = page["continuation_id"].as_str().unwrap().to_owned();
        let mut full_text = String::new();
        loop {
            assert_eq!(page["id"], pr["id"]);
            assert_eq!(page["kind"], "pr");
            assert!(page["text"].as_str().unwrap().len() <= 128);
            let sources = page["sources"].as_array().unwrap();
            assert!(sources.iter().all(|source| source["kind"] == "commit"));
            assert_eq!(
                sources
                    .iter()
                    .map(|source| source["id"].as_str().unwrap().to_owned())
                    .collect::<BTreeSet<_>>(),
                expected_sources
            );
            full_text.push_str(page["text"].as_str().unwrap());
            let Some(next) = page["continuation_id"].as_str() else {
                break;
            };
            page = success(&call(&client, "open", json!({"id":next})).await);
        }
        assert_eq!(full_text, summary);
        for source in &expected_sources {
            let opened = success(&call(&client, "open", json!({"id":source})).await);
            assert_eq!(opened["kind"], "commit");
            assert!(opened.get("summary").is_none());
        }
        assert_eq!(provider.received_requests().await.unwrap().len(), 1);
        assert!(!root.path().join("absent-model-cache").exists());
        assert!(
            provider
                .received_requests()
                .await
                .unwrap()
                .iter()
                .all(|request| request.url.path() == "/v1/chat/completions")
        );
        // Even identical newly generated output must not revive an old cursor.
        fs::create_dir_all(root.path().join(".deepkit/prompts")).unwrap();
        fs::write(
            root.path().join(".deepkit/prompts/summarize_pr.md"),
            "Describe the purpose concisely.\n{{collation}}\n{{hunks}}\n",
        )
        .unwrap();
        assert_eq!(
            failure(&call(&client, "open", json!({"id":cursor})).await)["error"]["code"],
            "source_unavailable"
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
