//! Image-description scope inheritance through public search; v1 §§6.8 and 9.
use serde_json::{Value, json};
use std::{fs, path::Path, time::Duration};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

async fn cli(root: &Path, args: &[&str]) -> Value {
    let mut command = tokio::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"));
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("DKC_") {
            command.env_remove(key);
        }
    }
    let output = tokio::time::timeout(
        Duration::from_secs(30),
        command
            .current_dir(root)
            .env("XDG_CONFIG_HOME", root.join("absent-config"))
            .env("DKC_MODELS_DIR", root.join("unused-models"))
            .env("DKC_IMAGE_SEARCH_FIXTURE_KEY", "synthetic-fixture-only")
            .args(args)
            .kill_on_drop(true)
            .output(),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(
        output.status.success(),
        "{} {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[tokio::test]
async fn image_description_search_inherits_parent_scope_in_every_retrieval_mode() {
    let server = MockServer::start().await;
    Mock::given(method("POST")).and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices":[{"finish_reason":"stop","message":{"content":"AmberQuokkaCompass describes the green square."}}],
            "usage":{"prompt_tokens":20,"completion_tokens":10,"total_tokens":30}
        }))).mount(&server).await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(|request: &wiremock::Request| {
            let body: Value = request.body_json().unwrap();
            ResponseTemplate::new(200).set_body_json(
                json!({"embeddings":vec![vec![1.0,0.0];body["input"].as_array().unwrap().len()]}),
            )
        })
        .mount(&server)
        .await;
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join(".deepkit")).unwrap();
    fs::write(root.path().join(".deepkit/config.toml"),format!(
        "[embedder]\nprofile='ollama'\ndims=2\n[embedder.ollama]\nurl='{0}'\n[llm]\nenabled=true\nurl='{0}/v1'\nmodel='fixture-text'\nvlm_model='fixture-vision'\napi_key_env='DKC_IMAGE_SEARCH_FIXTURE_KEY'\n[rerank]\nmode='off'\n",server.uri())).unwrap();
    image::RgbImage::from_pixel(2, 2, image::Rgb([0u8, 240, 0]))
        .save(root.path().join("diagram.png"))
        .unwrap();
    let indexed = cli(root.path(), &["index", "--level", "3", "--json"]).await;
    assert_eq!(indexed["enrichment"]["published"], 1, "{indexed}");
    assert_eq!(indexed["embedding"]["state"], "quiescent", "{indexed}");
    let mut identity = None;
    for mode in ["lexical", "dense", "hybrid"] {
        for scope in [
            None,
            Some("diagram.png"),
            Some("images"),
            Some("docs"),
            Some("source"),
            Some("missing.png"),
            Some("code"),
        ] {
            let mut args = vec![
                "search",
                "AmberQuokkaCompass",
                "--mode",
                mode,
                "--no-rerank",
                "--json",
            ];
            if let Some(scope) = scope {
                args.extend(["--scope", scope]);
            }
            let calls_before = server.received_requests().await.unwrap().len();
            let response = cli(root.path(), &args).await;
            let hits = response["hits"].as_array().unwrap();
            if matches!(scope, Some("missing.png" | "code")) {
                assert!(hits.is_empty(), "{mode}/{scope:?}: {response}");
                if scope == Some("code") {
                    assert_eq!(
                        server.received_requests().await.unwrap().len(),
                        calls_before,
                        "unrelated image summaries must not prepare or query a model for code scope"
                    );
                }
            } else {
                if mode != "lexical" {
                    assert_eq!(
                        response["index"]["dense"], "complete",
                        "{mode}/{scope:?}: {response}"
                    );
                }
                assert_eq!(hits.len(), 1, "{mode}/{scope:?}: {response}");
                assert_eq!(hits[0]["path"], "diagram.png");
                assert_eq!(hits[0]["kind"], "image");
                if let Some(expected) = &identity {
                    assert_eq!(&hits[0]["id"], expected);
                } else {
                    identity = Some(hits[0]["id"].clone());
                }
            }
        }
    }
}
