//! Remote rerank wire contract, private failures and explicit live diagnostic.
use super::*;
use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

fn config(server: &MockServer) -> config::Rerank {
    let mut config = config::Rerank {
        max_doc_chars: 2,
        candidates: 2,
        ..Default::default()
    };
    config.remote.url = format!("{}/v1/", server.uri());
    config.remote.model = "fixture-reranker".into();
    config
}
fn client(config: &config::Rerank) -> Remote {
    Remote::with_api_key(config, Duration::from_secs(2), "fixture-key".into()).unwrap()
}
#[tokio::test]
async fn sends_bounded_cohere_body_and_restores_original_candidate_order() {
    let server = MockServer::start().await;
    Mock::given(method("POST")).and(path("/v1/rerank"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"results":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":0.1}],"meta":{"api_version":{"version":"1"}}})))
        .expect(1).mount(&server).await;
    let provider = client(&config(&server));
    assert_eq!(provider.id(), "remote/fixture-reranker");
    assert_eq!(provider.max_doc_chars(), 2);
    assert_eq!(
        provider.score("query", &["é🙂TAIL", "abc"]).await.unwrap(),
        vec![0.1, 0.9]
    );
    let requests = server.received_requests().await.unwrap();
    let request = &requests[0];
    assert_eq!(
        request.headers["user-agent"],
        concat!("dkc/", env!("CARGO_PKG_VERSION"))
    );
    assert_eq!(request.headers["authorization"], "Bearer fixture-key");
    assert_eq!(
        request.body_json::<serde_json::Value>().unwrap(),
        json!({"model":"fixture-reranker","query":"query","documents":["é🙂","ab"],"top_n":2,"return_documents":false})
    );
    server.verify().await;
}
#[tokio::test]
async fn empty_or_invalid_input_never_dispatches() {
    let server = MockServer::start().await;
    let config = config(&server);
    let provider = client(&config);
    assert!(provider.score_once("", &[]).await.unwrap().is_empty());
    assert!(matches!(
        provider.score_once(" ", &["x"]).await,
        Err(Error::Input)
    ));
    assert!(matches!(
        provider.score_once("q", &["a", "b", "c"]).await,
        Err(Error::Input)
    ));
    assert!(server.received_requests().await.unwrap().is_empty());
    for url in [
        "not a URL",
        "file:///private",
        "https://name:password@example.invalid/v1",
        "https://example.invalid/v1?secret=value",
        "https://example.invalid/#fragment",
    ] {
        let mut config = config.clone();
        config.remote.url = url.into();
        assert!(matches!(
            Remote::with_api_key(&config, Duration::from_secs(1), "fixture".into()),
            Err(Error::Configuration)
        ));
    }
    assert!(matches!(
        Remote::with_api_key(&config, Duration::ZERO, "fixture".into()),
        Err(Error::Configuration)
    ));
    assert!(matches!(
        Remote::with_api_key(&config, Duration::from_secs(1), "bad\nheader".into()),
        Err(Error::Configuration)
    ));
    for (chars, docs) in [(0, 2), (2, 0)] {
        let mut config = config.clone();
        config.max_doc_chars = chars;
        config.candidates = docs;
        assert!(matches!(
            Remote::with_api_key(&config, Duration::from_secs(1), "fixture".into()),
            Err(Error::Configuration)
        ));
    }
}
#[test]
fn malformed_scores_never_invent_candidate_values() {
    for body in [
        json!({}),
        json!({"results":[]}),
        json!({"results":[{"index":0,"relevance_score":0.1}]}),
        json!({"results":[{"index":0,"relevance_score":0.1},{"index":0,"relevance_score":0.2}]}),
        json!({"results":[{"index":0,"relevance_score":0.1},{"index":2,"relevance_score":0.2}]}),
        json!({"results":[{"index":0,"relevance_score":-0.1},{"index":1,"relevance_score":0.2}]}),
        json!({"results":[{"index":0,"relevance_score":1.1},{"index":1,"relevance_score":0.2}]}),
        json!({"results":[{"index":-1,"relevance_score":0.1},{"index":1,"relevance_score":0.2}]}),
        json!({"results":[{"index":0.5,"relevance_score":0.1},{"index":1,"relevance_score":0.2}]}),
        json!({"results":[{"index":0,"relevance_score":"NaN"},{"index":1,"relevance_score":0.2}]}),
    ] {
        assert!(
            matches!(
                decode(&serde_json::to_vec(&body).unwrap(), 2),
                Err(Error::Response(_))
            ),
            "{body}"
        );
    }
    assert!(matches!(
        decode(b"private invalid body", 2),
        Err(Error::Response(_))
    ));
    assert!(matches!(
        decode(br#"{"results":[{"index":0,"relevance_score":1e999}]}"#, 1),
        Err(Error::Response(_))
    ));
    assert_eq!(
        decode(
            br#"{"results":[{"index":1,"relevance_score":1},{"index":0,"relevance_score":0}]}"#,
            2
        )
        .unwrap(),
        vec![0.0, 1.0]
    );
}
#[tokio::test]
async fn http_errors_and_redirects_do_not_retry_or_expose_private_bodies() {
    let server = MockServer::start().await;
    let elsewhere = MockServer::start().await;
    let provider = client(&config(&server));
    for status in [401, 403, 429, 500, 307] {
        server.reset().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(status)
                    .insert_header("Retry-After", "3")
                    .insert_header("Location", format!("{}/leak", elsewhere.uri()))
                    .set_body_string(
                        "private-response-marker fixture-key private-query private-document",
                    ),
            )
            .expect(1)
            .mount(&server)
            .await;
        let error = provider
            .score_once("private-query", &["private-document"])
            .await
            .unwrap_err();
        assert!(
            matches!(error,Error::Http{status:s,retry_after:Some(delay)} if s==status && delay==Duration::from_secs(3))
        );
        let rendered = format!("{error:?} {error}");
        for marker in [
            "private-response-marker",
            "fixture-key",
            "private-query",
            "private-document",
        ] {
            assert!(!rendered.contains(marker));
        }
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }
    assert!(elsewhere.received_requests().await.unwrap().is_empty());
}
#[tokio::test]
async fn response_bytes_and_request_deadline_are_bounded() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(
            json!({"results":[{"index":0,"relevance_score":0.9}],"padding":"x".repeat(32 * 1024)}),
        ))
        .expect(1)
        .mount(&server)
        .await;
    let provider = client(&config(&server));
    assert!(matches!(
        provider.score_once("q", &["d"]).await,
        Err(Error::Response(ResponseReason::BodyLimit))
    ));
    server.reset().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(250))
                .set_body_json(json!({"results":[{"index":0,"relevance_score":0.9}]})),
        )
        .expect(1)
        .mount(&server)
        .await;
    let provider = Remote::with_api_key(
        &config(&server),
        Duration::from_millis(40),
        "fixture-key".into(),
    )
    .unwrap();
    assert!(matches!(
        provider.score_once("private query", &["d"]).await,
        Err(Error::Timeout)
    ));
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}
#[tokio::test]
#[ignore = "requires DKC_LIVE_GATEWAY=1 and privately resolved Moonly credentials"]
async fn live_gateway_reranks_fixed_public_pairs()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    if std::env::var("DKC_LIVE_GATEWAY").as_deref() != Ok("1") {
        return Err("set DKC_LIVE_GATEWAY=1 for the explicit diagnostic".into());
    }
    let config = config::Rerank::default();
    let provider = Remote::from_config(&config, Duration::from_secs(15))?;
    let scores = provider
        .score(
            "What is the capital of France?",
            &[
                "The capital of France is Paris.",
                "Bananas are yellow fruit.",
                "France is a country in Europe.",
            ],
        )
        .await?;
    assert_eq!(scores.len(), 3);
    assert!(
        scores
            .iter()
            .all(|s| s.is_finite() && (0.0..=1.0).contains(s))
    );
    assert!(scores[0] > scores[1]);
    println!("gateway rerank public-pair scores: {scores:?}");
    Ok(())
}

#[test]
fn validation_reasons_are_closed_and_preserve_existing_rejection_order() {
    for (body, expected, reason) in [
        (r#"private-response-marker"#, 1, ResponseReason::Schema),
        (r#"{"results":[]}"#, 1, ResponseReason::Count),
        (
            r#"{"results":[{"index":0,"relevance_score":0.2},{"index":0,"relevance_score":0.3}]}"#,
            2,
            ResponseReason::Index,
        ),
        (
            r#"{"results":[{"index":2,"relevance_score":0.2}]}"#,
            1,
            ResponseReason::Index,
        ),
        (
            r#"{"results":[{"index":0,"relevance_score":-0.1}]}"#,
            1,
            ResponseReason::Score,
        ),
        (
            r#"{"results":[{"index":0,"relevance_score":1.1}]}"#,
            1,
            ResponseReason::Score,
        ),
        // Invalid JSON field types remain schema errors, not reparsed permissively.
        (
            r#"{"results":[{"index":-1,"relevance_score":0.2}]}"#,
            1,
            ResponseReason::Schema,
        ),
        (
            r#"{"results":[{"index":0,"relevance_score":"private-response-marker"}]}"#,
            1,
            ResponseReason::Schema,
        ),
        (
            r#"{"results":[{"index":0,"relevance_score":1e999}]}"#,
            1,
            ResponseReason::Schema,
        ),
    ] {
        let error = decode(body.as_bytes(), expected).unwrap_err();
        assert!(matches!(error, Error::Response(actual) if actual == reason));
        assert!(!format!("{error:?} {error}").contains("private-response-marker"));
    }
}

#[tokio::test]
async fn validation_failures_are_single_attempt_and_do_not_echo_private_inputs() {
    let server = MockServer::start().await;
    let provider = client(&config(&server));
    for (body, reason) in [
        (
            json!({"private-response-marker": true}),
            ResponseReason::Schema,
        ),
        (json!({"results": []}), ResponseReason::Count),
        (
            json!({"results": [{"index": 7, "relevance_score": 0.2}]}),
            ResponseReason::Index,
        ),
        (
            json!({"results": [{"index": 0, "relevance_score": 2.0}]}),
            ResponseReason::Score,
        ),
        (
            json!({"padding": "private-response-marker".repeat(2048)}),
            ResponseReason::BodyLimit,
        ),
    ] {
        server.reset().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .expect(1)
            .mount(&server)
            .await;
        let error = provider
            .score_once("private-query", &["private-document"])
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Response(actual) if actual == reason));
        let rendered = format!("{error:?} {error}");
        for marker in [
            "private-response-marker",
            "fixture-key",
            "private-query",
            "private-document",
        ] {
            assert!(!rendered.contains(marker));
        }
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
        server.verify().await;
    }
}

#[test]
fn echo_allowance_counts_exact_json_string_bytes() {
    // Quotes, backslashes and control characters expand; Unicode uses UTF-8 bytes.
    let documents = ["é🙂", "\"\\\n\u{0000}"];
    let encoded = documents
        .iter()
        .map(|doc| serde_json::to_string(doc).unwrap().len())
        .sum::<usize>();
    assert_eq!(
        response_cap(&documents).unwrap(),
        16 * 1024 + 2 * 256 + encoded
    );
    assert!(encoded > documents.iter().map(|doc| doc.len()).sum::<usize>());
    assert_eq!(response_cap(&[]).unwrap(), 16 * 1024);
}

#[tokio::test]
async fn valid_document_echo_above_old_limit_preserves_score_contract_and_wire_truncation() {
    let server = MockServer::start().await;
    let mut settings = config(&server);
    settings.max_doc_chars = 20000;
    let submitted = "é🙂\"\\\n".repeat(4000);
    let original = format!("{submitted}THIS_TAIL_MUST_NOT_BE_SENT");
    assert_eq!(submitted.chars().count(), settings.max_doc_chars as usize);
    let body = json!({"results":[{"index":0,"relevance_score":0.8,"document":{"text":submitted}}]});
    let bytes = serde_json::to_vec(&body).unwrap();
    assert!(bytes.len() > 16 * 1024 + 256);
    assert!(bytes.len() <= response_cap(&[&submitted]).unwrap());
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes))
        .expect(1)
        .mount(&server)
        .await;
    assert_eq!(
        client(&settings)
            .score_once("fixed query", &[&original])
            .await
            .unwrap(),
        vec![0.8]
    );
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let sent = requests[0].body_json::<serde_json::Value>().unwrap();
    assert_eq!(sent["return_documents"], false);
    assert_eq!(sent["documents"], json!([submitted]));
    server.verify().await;
}

#[tokio::test]
async fn echo_allowance_still_refuses_oversize_and_invalid_core_without_retry() {
    let server = MockServer::start().await;
    let mut settings = config(&server);
    settings.max_doc_chars = 20000;
    let document = "x".repeat(20000);
    let cap = response_cap(&[&document]).unwrap();
    let provider = client(&settings);
    let cases = [
        (
            json!({"results":[{"index":0,"relevance_score":0.8,"document":{"text":document}}],"padding":"x".repeat(cap)}),
            ResponseReason::BodyLimit,
        ),
        (
            json!({"results":[{"index":0,"document":{"text":document}}]}),
            ResponseReason::Schema,
        ),
        (
            json!({"results":[],"document":{"text":document}}),
            ResponseReason::Count,
        ),
        (
            json!({"results":[{"index":1,"relevance_score":0.8,"document":{"text":document}}]}),
            ResponseReason::Index,
        ),
        (
            json!({"results":[{"index":0,"relevance_score":1.2,"document":{"text":document}}]}),
            ResponseReason::Score,
        ),
    ];
    for (body, reason) in cases {
        server.reset().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .expect(1)
            .mount(&server)
            .await;
        assert!(
            matches!(provider.score_once("fixed query", &[&document]).await,
            Err(Error::Response(actual)) if actual == reason)
        );
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
        server.verify().await;
    }
}
