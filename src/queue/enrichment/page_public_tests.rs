//! Real CLI page indexing, embedding and lexical search; v1 §6.2/§9.
use super::{image_public_tests::run, image_tests};
use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

#[tokio::test]
async fn public_scanned_page_index_embeds_sections_and_searches_without_repeat_calls() {
    page_flow(include_bytes!("../../../fixtures/docs/scanned-jpeg.pdf")).await;
}

#[cfg(feature = "pdfium")]
#[tokio::test]
#[ignore = "requires PDFium on the loader path and a pdfium-enabled dkc binary"]
async fn public_rendered_page_indexes_opens_and_retires_changed_source() {
    let generated = tempfile::tempdir().unwrap();
    let output = std::process::Command::new("python3")
        .arg(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("scripts/corpus/make-scan-fixtures.py"),
        )
        .arg("--output")
        .arg(generated.path())
        .args(["--case", "overlay"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    page_flow(&std::fs::read(generated.path().join("scanned-overlay.pdf")).unwrap()).await;
}

async fn page_flow(source: &[u8]) {
    let server = MockServer::start().await;
    Mock::given(method("POST")).and(path("/v1/chat/completions")).respond_with(ResponseTemplate::new(200).set_body_json(json!({
        "choices":[{"finish_reason":"stop","message":{"content":"# Amber manual\n\nAmberpage quartz maintenance instructions."}}],
        "usage":{"prompt_tokens":30,"completion_tokens":20,"total_tokens":50}
    }))).expect(1).mount(&server).await;
    Mock::given(method("POST")).and(path("/v1/embeddings")).respond_with(|request:&wiremock::Request| {
        let body:Value=serde_json::from_slice(&request.body).unwrap();
        let count=body["input"].as_array().unwrap().len();
        ResponseTemplate::new(200).set_body_json(json!({
            "data":(0..count).map(|index|json!({"index":index,"embedding":"AACAPwAAAAA="})).collect::<Vec<_>>(),
            "usage":{"prompt_tokens":10,"total_tokens":10}
        }))
    }).mount(&server).await;
    let (root, _env) = image_tests::fixture(&server.uri(), |c| {
        c.llm.api_key_env = "DKC_TEST_IMAGE_PUBLIC_KEY".into();
        c.llm.concurrency = 1;
        c.enrich.images = false;
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
    std::fs::write(root.path().join("scan.pdf"), source).unwrap();
    let mut previous = None;
    for repeat in 0..2 {
        let report = run(root.path(), &["index", "--level", "3", "--json"]).await;
        assert_eq!(report["enrichment"]["state"], "complete", "{report}");
        assert_eq!(
            report["enrichment"]["published"],
            if repeat == 0 { 1 } else { 0 },
            "{report}"
        );
        let calls = server.received_requests().await.unwrap().len();
        if let Some(old) = previous {
            assert_eq!(calls, old, "unchanged index repeated model work");
        }
        previous = Some(calls);
    }
    let found = run(
        root.path(),
        &[
            "search",
            "amberpage quartz",
            "--mode",
            "lexical",
            "--no-rerank",
            "--json",
        ],
    )
    .await;
    let hits = found["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 1, "{found}");
    assert_eq!(hits[0]["path"], "scan.pdf", "{found}");
    assert_eq!(hits[0]["fresh"]["l2"], "current", "{found}");
    let opened = run(
        root.path(),
        &["open", hits[0]["id"].as_str().unwrap(), "--json"],
    )
    .await;
    assert_eq!(opened["path"], "scan.pdf", "{opened}");
    assert!(
        opened["text"]
            .as_str()
            .unwrap()
            .contains("Amberpage quartz"),
        "{opened}"
    );
    assert_eq!(
        server.received_requests().await.unwrap().len(),
        previous.unwrap()
    );
    let db = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
    assert_eq!(db.query_row("SELECT count(*) FROM embedding_publications e JOIN units u ON u.id=e.unit_id WHERE u.kind='doc_section' AND u.ladder_level=3",[],|r|r.get::<_,i64>(0)).unwrap(),1);
    assert_eq!(db.query_row("SELECT count(*) FROM embedding_publications e JOIN units u ON u.id=e.unit_id WHERE u.kind='image'",[],|r|r.get::<_,i64>(0)).unwrap(),0);
    drop(db);
    std::fs::write(
        root.path().join("scan.pdf"),
        include_bytes!("../../../fixtures/docs/scanned.pdf"),
    )
    .unwrap();
    let changed = run(
        root.path(),
        &[
            "search",
            "amberpage quartz",
            "--mode",
            "lexical",
            "--no-rerank",
            "--json",
        ],
    )
    .await;
    assert!(
        changed["hits"].as_array().unwrap().is_empty(),
        "old transcription survived changed PDF: {changed}"
    );
}
