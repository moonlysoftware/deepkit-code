//! Fixed HTTP chat responses and request-contract regressions; v1 §§7–8.
use super::*;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, method, path},
};

#[tokio::test]
async fn configured_capacity_is_shared_by_text_and_image_and_released_on_cancellation() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    Mock::given(method("POST"))
        .respond_with(move |_: &wiremock::Request| {
            let response = ResponseTemplate::new(200)
                .set_body_json(json!({"choices":[{"message":{"content":"Done."}}]}));
            if observed.fetch_add(1, Ordering::SeqCst) == 0 {
                response.set_delay(Duration::from_secs(10))
            } else {
                response
            }
        })
        .expect(2)
        .mount(&server)
        .await;
    let config = crate::config::Llm {
        enabled: true,
        url: server.uri(),
        concurrency: 1,
        ..Default::default()
    };
    let client = Arc::new(
        Client::with_key(&config, Duration::from_secs(15), 1024, "fixture".into()).unwrap(),
    );
    let text = client
        .prepare_text("Summary.", "Public input.", 120)
        .unwrap();
    let first_client = Arc::clone(&client);
    let first = tokio::spawn(async move { first_client.post_once(&text).await });
    tokio::time::timeout(Duration::from_secs(3), async {
        while calls.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let mut encoded = Cursor::new(Vec::new());
    image::DynamicImage::new_rgb8(2, 2)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .unwrap();
    let image = client
        .prepare_image(encoded.get_ref(), "image/png", "Describe.", 200)
        .unwrap();
    let second = client.post_once(&image);
    tokio::pin!(second);
    assert!(
        tokio::time::timeout(Duration::from_millis(30), &mut second)
            .await
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    first.abort();
    assert!(matches!(first.await, Err(error) if error.is_cancelled()));
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(3), second)
            .await
            .unwrap()
            .unwrap()
            .completion
            .content,
        "Done."
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(client.slots.available_permits(), 1);
}

#[tokio::test]
async fn bounded_responses_and_redirects_fail_once_and_changed_contract_refuses_before_send() {
    for status in [200, 307] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(status)
                    .insert_header("Location", format!("{}/redirected", server.uri()))
                    .set_body_string("x".repeat(100_000)),
            )
            .expect(1)
            .mount(&server)
            .await;
        let client = provider(&server);
        let mut prepared = client.prepare_text("Summary.", "Public input.", 1).unwrap();
        let error = client.post_once(&prepared).await.err().unwrap();
        if status == 200 {
            assert!(matches!(error, Error::BodyLimit));
        } else {
            assert!(matches!(error, Error::Http { status: 307, .. }));
        }
        prepared.contract[0] ^= 1;
        assert!(matches!(
            client.post_once(&prepared).await,
            Err(Error::Input)
        ));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }
}

#[tokio::test]
async fn image_request_preserves_mime_model_and_bytes_and_refuses_invalid_inputs() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices":[{"message":{"content":"A small image."}}]
        })))
        .expect(1)
        .mount(&server)
        .await;
    let client = provider(&server);
    let mut encoded = Cursor::new(Vec::new());
    image::DynamicImage::new_rgb8(2, 2)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .unwrap();
    let bytes = encoded.into_inner();
    let prepared = client
        .prepare_image(&bytes, "image/png", "Describe this image.", 200)
        .unwrap();
    assert!(prepared.input_tokens >= 86);
    assert_eq!(prepared.max_output_tokens, 200);
    let reply = client.post_once(&prepared).await.unwrap();
    assert_eq!(reply.completion.content, "A small image.");
    let requests = server.received_requests().await.unwrap();
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["model"], "fixture-image");
    assert_eq!(body["max_tokens"], 200);
    let parts = &body["messages"][0]["content"];
    assert_eq!(
        parts[0],
        json!({"type":"text","text":"Describe this image."})
    );
    assert_eq!(parts[1]["type"], "image_url");
    let data = parts[1]["image_url"]["url"]
        .as_str()
        .unwrap()
        .strip_prefix("data:image/png;base64,")
        .unwrap();
    assert_eq!(STANDARD.decode(data).unwrap(), bytes);
    for (bytes, mime) in [
        (bytes.as_slice(), "image/jpeg"),
        (&[][..], "image/png"),
        (&[0; 1025][..], "image/png"),
    ] {
        assert!(client.prepare_image(bytes, mime, "Describe.", 200).is_err());
    }
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

fn provider(server: &MockServer) -> Client {
    let config = crate::config::Llm {
        enabled: true,
        url: format!("{}/v1", server.uri()),
        model: "fixture-text".into(),
        vlm_model: "fixture-image".into(),
        ..Default::default()
    };
    Client::with_key(&config, Duration::from_secs(2), 1024, "fixture-key".into()).unwrap()
}

#[tokio::test]
async fn normal_completion_sends_required_headers_options_and_accounts_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer fixture-key"))
        .and(header(
            "user-agent",
            concat!("dkc/", env!("CARGO_PKG_VERSION")),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices":[{"message":{"content":"A useful summary."}}],
            "usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let client = provider(&server);
    let prepared = client
        .prepare_text("Summarize code.", "Public fixture.", 120)
        .unwrap();
    let reply = client.post_once(&prepared).await.unwrap();
    assert_eq!(reply.completion.content, "A useful summary.");
    assert_eq!(reply.tokens, Some(15));
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["model"], "fixture-text");
    assert_eq!(body["messages"][1]["content"], "Public fixture.");
    assert_eq!(body["max_tokens"], 120);
    assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
}

#[tokio::test]
async fn empty_and_http_failures_make_one_attempt_without_exposing_provider_text() {
    for status in [200, 401, 429] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(status)
                .insert_header("Retry-After", "7")
                .set_body_json(json!({
                    "choices":[{"message":{"content":"","reasoning_content":"private-provider-marker"}}],
                    "error":{"message":"private-provider-marker"},
                    "usage":{"prompt_tokens":10,"completion_tokens":120,"total_tokens":130}
                })))
            .expect(1).mount(&server).await;
        let client = provider(&server);
        let prepared = client
            .prepare_text("Summary.", "Public fixture.", 120)
            .unwrap();
        let error = client.post_once(&prepared).await.err().unwrap();
        match status {
            200 => {
                assert!(matches!(error, Error::ThinkingConsumedBudget { .. }));
                assert_eq!(error.reported_tokens(), Some(130));
            }
            401 => assert!(matches!(error, Error::Unauthorized)),
            429 => assert!(
                matches!(error, Error::Http { status: 429, retry_after: Some(t), .. } if t == Duration::from_secs(7))
            ),
            _ => unreachable!(),
        }
        assert!(!format!("{error:?} {error}").contains("private-provider-marker"));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }
}

#[test]
fn image_generation_proof_matches_dispatch_and_tracks_image_prompt_and_model() {
    let config = crate::config::Llm {
        enabled: true,
        ..Default::default()
    };
    let client = Client::with_key(&config, Duration::from_secs(1), 1024, "fixture".into()).unwrap();
    let mut bytes = Cursor::new(Vec::new());
    image::DynamicImage::new_rgb8(2, 2)
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    let image = bytes.into_inner();
    let wire = client
        .prepare_image(&image, "image/png", "Describe.", 200)
        .unwrap();
    let proof = crate::models::openai_compat::chat::image_fingerprint(
        &config,
        1024,
        &image,
        "image/png",
        "Describe.",
        200,
    )
    .unwrap();
    assert_eq!(proof, wire.fingerprint);
    let mut changed = config.clone();
    changed.vlm_model = "another-vision-model".into();
    assert_ne!(
        proof,
        crate::models::openai_compat::chat::image_fingerprint(
            &changed,
            1024,
            &image,
            "image/png",
            "Describe.",
            200
        )
        .unwrap()
    );
    assert_ne!(
        proof,
        crate::models::openai_compat::chat::image_fingerprint(
            &config,
            1024,
            &image,
            "image/png",
            "New caption.",
            200
        )
        .unwrap()
    );
    assert!(
        crate::models::openai_compat::chat::image_fingerprint(
            &config,
            1,
            &image,
            "image/png",
            "Describe.",
            200
        )
        .is_err()
    );
    assert!(
        crate::models::openai_compat::chat::image_fingerprint(
            &config,
            1024,
            &image,
            "image/jpeg",
            "Describe.",
            200
        )
        .is_err()
    );
    let mut bytes = Cursor::new(Vec::new());
    image::DynamicImage::new_rgb8(3, 2)
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    assert_ne!(
        proof,
        crate::models::openai_compat::chat::image_fingerprint(
            &config,
            1024,
            bytes.get_ref(),
            "image/png",
            "Describe.",
            200
        )
        .unwrap()
    );
}

#[tokio::test]
async fn http_diagnostics_allowlist_codes_without_changing_retry_classification() {
    use crate::queue::retry::{ChatFailure, Outcome};
    for status in [400, 401, 429, 503] {
        for (body, recognized) in [
            (json!({"error":{"code":"context_length_exceeded","message":"secret-prompt-canary"}}).to_string(), true),
            (json!({"error":{"type":"context_length_exceeded","message":"secret-prompt-canary"}}).to_string(), true),
            (json!({"error":{"code":"secret-prompt-canary","message":"context_length_exceeded"}}).to_string(), false),
            ("secret-prompt-canary malformed".into(), false),
            (json!({"error":{"code":"context_length_exceeded","message":"secret-prompt-canary".repeat(1000)}}).to_string(), false),
        ] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(status).insert_header("Retry-After", "7").set_body_string(body))
                .expect(1).mount(&server).await;
            let client = provider(&server);
            let prepared = client.prepare_text("secret-prompt-canary", "secret-prompt-canary", 120).unwrap();
            let error = client.post_once(&prepared).await.err().unwrap();
            assert!(!format!("{error:?} {error}").contains("secret-prompt-canary"));
            if status == 401 {
                assert!(matches!(error, Error::Unauthorized));
            } else {
                assert!(matches!(&error, Error::Http {status: actual, retry_after: Some(delay), reason}
                    if *actual == status && *delay == Duration::from_secs(7) && reason.is_some() == recognized));
            }
            let outcome = Outcome::from_provider_attempt(&error.into());
            match status {
                400 => assert!(matches!(outcome, Outcome::PermanentChat(ChatFailure::Http(400, reason)) if reason.is_some()==recognized)),
                401 => assert!(matches!(outcome, Outcome::PermanentChat(ChatFailure::Unauthorized))),
                _ => assert!(matches!(outcome, Outcome::Transient { retry_after: Some(delay) } if delay == Duration::from_secs(7))),
            }
            assert_eq!(server.received_requests().await.unwrap().len(), 1);
        }
    }
}

#[tokio::test]
async fn stalled_error_body_retains_http_status_after_request_deadline() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (release, hold) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = [0; 4096];
        assert!(stream.read(&mut request).await.unwrap() > 0);
        stream
            .write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 100\r\n\r\n")
            .await
            .unwrap();
        let _ = hold.await;
    });
    let config = crate::config::Llm {
        enabled: true,
        url: format!("http://{address}/v1"),
        ..Default::default()
    };
    let client = Client::with_key(&config, Duration::from_secs(2), 1024, "fixture".into()).unwrap();
    let prepared = client
        .prepare_text("Summary", "Public fixture", 120)
        .unwrap();
    let result = tokio::time::timeout(Duration::from_secs(10), client.post_once(&prepared))
        .await
        .unwrap();
    assert!(matches!(
        result,
        Err(Error::Http {
            status: 400,
            reason: None,
            ..
        })
    ));
    release.send(()).unwrap();
    server.await.unwrap();
}
