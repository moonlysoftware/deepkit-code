//! Real HTTP plus durable ledger evidence for query dispatch; v1 §§7.2, 8.2, 9.
use super::*;
use crate::store::{self, StoreLimits};
use base64::Engine as _;
use std::{
    fs,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_FIXTURE_MODEL: AtomicU64 = AtomicU64::new(1);
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

fn setup(server: &MockServer, profile: &str, calls: i64) -> (tempfile::TempDir, Arc<Factory>) {
    let root = tempfile::tempdir().unwrap();
    // MockServer may recycle the same port; capability cache keys must identify
    // this fixture's independent model, not a previous pooled server occupant.
    let fixture_model = NEXT_FIXTURE_MODEL.fetch_add(1, Ordering::Relaxed);
    fs::create_dir(root.path().join(".deepkit")).unwrap();
    let mut config = format!(
        "[index]\nrepo_name='query-fixture'\n[embedder]\nprofile='{profile}'\ndims=2\n[embedder.{profile}]\nurl='{}'\nmodel='query-fixture-{fixture_model}-{}'\nquery_prefix='Represent: '\n",
        server.uri(),
        server.address().port()
    );
    if profile == "remote" {
        config.push_str("rpm=10\ntpm=10000\n");
    }
    fs::write(root.path().join(".deepkit/config.toml"), config).unwrap();
    let config = config::load(root.path(), None).unwrap().config;
    let store = Arc::new(store::open(":memory:", StoreLimits::default()).unwrap());
    let factory = Factory::build(
        store,
        Settings {
            config_root: root.path().into(),
            config,
            kind: UnitKind::Chunk,
            budget: usage::Budget {
                calls: Some(calls),
                tokens: Some(1000),
            },
        },
        |cfg| OpenAiEmbedder::with_api_key(cfg, String::new()),
    )
    .unwrap();
    (root, factory)
}
async fn embed(factory: &Arc<Factory>) -> Result<Vec<f32>, admission::Error> {
    factory
        .embed_once(
            "where is payment retried?".into(),
            Instant::now() + Duration::from_secs(2),
            || 100,
        )
        .await
}
fn counts(factory: &Factory) -> (i64, i64, i64) {
    factory.store.writer().unwrap().query_row("SELECT (SELECT count(*) FROM usage_reservations WHERE state='reserved'),(SELECT count(*) FROM usage_reservations WHERE state='dispatched'),(SELECT COALESCE(sum(tokens),0) FROM usage)", [], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?))).unwrap()
}

#[tokio::test]
async fn both_embedding_query_profiles_send_prepared_text_and_settle_actual_usage() {
    for profile in ["remote", "ollama"] {
        let server = MockServer::start().await;
        let (_root, factory) = setup(&server, profile, 1);
        let endpoint = if profile == "remote" {
            "/embeddings"
        } else {
            "/api/embed"
        };
        let body = if profile == "remote" {
            serde_json::json!({"data":[{"index":0,"embedding":encoded(&[3.0,4.0])}],"usage":{"total_tokens":7}})
        } else {
            serde_json::json!({"embeddings":[[3.0,4.0]],"prompt_eval_count":7})
        };
        Mock::given(method("POST"))
            .and(path(endpoint))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .expect(1)
            .mount(&server)
            .await;
        let vector = embed(&factory).await.unwrap();
        assert!((vector[0] - 0.6).abs() < 0.0001);
        assert!((vector[1] - 0.8).abs() < 0.0001);
        assert_eq!(counts(&factory), (0, 0, 7));
        assert!(matches!(
            embed(&factory).await,
            Err(admission::Error::Budget { .. })
        ));
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let input: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(
            input["input"],
            serde_json::json!(["Represent: where is payment retried?"])
        );
        assert_eq!(
            requests[0].headers.get("user-agent").unwrap(),
            concat!("dkc/", env!("CARGO_PKG_VERSION"))
        );
        if profile == "ollama" {
            assert_eq!(input["truncate"], false);
        }
        let identity: (String, String, i64) = factory
            .store
            .writer()
            .unwrap()
            .query_row(
                "SELECT provider,endpoint_url,sent_units FROM usage_reservations",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(
            identity,
            (
                factory.identity.provider.clone(),
                format!("{}{endpoint}", server.uri()),
                1
            )
        );
    }
}

#[tokio::test]
async fn changed_configuration_releases_unsent_query_without_contacting_either_endpoint() {
    let server = MockServer::start().await;
    let (root, factory) = setup(&server, "ollama", 5);
    let cfg_path = root.path().join(".deepkit/config.toml");
    let original = fs::read_to_string(&cfg_path).unwrap();
    fs::write(
        &cfg_path,
        original.replace("Represent: ", "New instruction: "),
    )
    .unwrap();
    assert!(matches!(
        embed(&factory).await,
        Err(admission::Error::Refused(usage::Dispatch::Superseded))
    ));
    assert_eq!(counts(&factory), (0, 0, 0));
    assert!(server.received_requests().await.unwrap().is_empty());
    let metadata: Option<String> = factory
        .store
        .writer()
        .unwrap()
        .query_row("SELECT endpoint_url FROM usage_reservations", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert!(metadata.is_none());
}

#[tokio::test]
async fn independent_query_fixtures_do_not_share_cached_dimensions_capability() {
    // Wiremock pools servers, so a port can be reused by a later fixture.
    // Force that reuse here instead of relying on test order or port allocation.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(400).set_body_json(
            serde_json::json!({"error":{"message":"model does not support matryoshka representation"}}),
        ))
        .expect(2)
        .mount(&server)
        .await;
    for _ in 0..2 {
        let (_root, factory) = setup(&server, "remote", 1);
        let result = embed(&factory).await;
        assert!(
            matches!(
                result,
                Err(admission::Error::Provider(AttemptError::DimensionsRejected))
            ),
            "{result:?}"
        );
        assert_eq!(counts(&factory), (0, 1, 0));
    }
    for request in server.received_requests().await.unwrap() {
        let input: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(input["dimensions"], 2);
    }
}

#[tokio::test]
async fn dimensions_fallback_is_a_separate_admitted_and_charged_request() {
    let server = MockServer::start().await;
    let (_root, factory) = setup(&server, "remote", 2);
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(
            ResponseTemplate::new(400).set_body_json(
                serde_json::json!({"error":{"message":"model does not support matryoshka representation"}}),
            ),
        )
        .expect(1)
        .mount(&server)
        .await;
    let rejection = embed(&factory).await;
    assert!(
        matches!(
            rejection,
            Err(admission::Error::Provider(AttemptError::DimensionsRejected))
        ),
        "{rejection:?}"
    );
    assert_eq!(counts(&factory), (0, 1, 0));
    let first = server.received_requests().await.unwrap();
    let input: serde_json::Value = serde_json::from_slice(&first[0].body).unwrap();
    assert_eq!(input["dimensions"], 2);
    server.reset().await;
    Mock::given(method("POST")).and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"data":[{"index":0,"embedding":encoded(&[3.0,4.0,8.0])}],"usage":{"total_tokens":9}})))
        .expect(1).mount(&server).await;
    assert_eq!(embed(&factory).await.unwrap().len(), 2);
    assert_eq!(counts(&factory), (0, 1, 9));
    assert!(matches!(
        embed(&factory).await,
        Err(admission::Error::Budget { .. })
    ));
    let second = server.received_requests().await.unwrap();
    assert_eq!(second.len(), 1);
    let input: serde_json::Value = serde_json::from_slice(&second[0].body).unwrap();
    assert!(input.get("dimensions").is_none());
}

#[tokio::test]
async fn dimensions_rejection_survives_a_query_process_restart() {
    let server = MockServer::start().await;
    let (root, factory) = setup(&server, "remote", 2);
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(400).set_body_json(
            serde_json::json!({"error":{"message":"model does not support matryoshka representation"}}),
        ))
        .expect(1)
        .mount(&server)
        .await;
    let rejection = embed(&factory).await;
    assert!(matches!(
        rejection,
        Err(admission::Error::Provider(AttemptError::DimensionsRejected))
    ));
    let capability = factory
        .dimensions_capability
        .as_deref()
        .expect("remote capability key");
    assert_eq!(
        factory.store.get_meta(capability).unwrap().as_deref(),
        Some("1")
    );
    match &factory.provider {
        Provider::Remote(provider) => provider.forget_dimensions_unsupported_for_test(),
        _ => panic!("remote provider expected"),
    }
    let store = Arc::clone(&factory.store);
    drop(factory);

    server.reset().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(
            serde_json::json!({"data":[{"index":0,"embedding":encoded(&[3.0,4.0,8.0])}],"usage":{"total_tokens":9}}),
        ))
        .expect(1)
        .mount(&server)
        .await;
    let restarted = Factory::build(
        store,
        Settings {
            config_root: root.path().into(),
            config: config::load(root.path(), None).unwrap().config,
            kind: UnitKind::Chunk,
            budget: usage::Budget {
                calls: Some(2),
                tokens: Some(1000),
            },
        },
        |config| OpenAiEmbedder::with_api_key(config, String::new()),
    )
    .unwrap();
    assert_eq!(embed(&restarted).await.unwrap().len(), 2);
    assert_eq!(counts(&restarted), (0, 1, 9));
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let input: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert!(input.get("dimensions").is_none());
}

#[tokio::test]
async fn http_timeout_keeps_uncertain_charge_and_empty_query_never_reserves() {
    let server = MockServer::start().await;
    let (root, initial) = setup(&server, "ollama", 1);
    let cfg_path = root.path().join(".deepkit/config.toml");
    let original = fs::read_to_string(&cfg_path).unwrap();
    fs::write(
        &cfg_path,
        // Keep the provider's transport timeout later than the caller deadline.
        // The test then exercises the outer deadline without racing reqwest's
        // independently configured timeout on a busy native runner.
        format!("{original}\n[limits]\nquery_timeout_ms=1500\n"),
    )
    .unwrap();
    let factory = Factory::new(
        Arc::clone(&initial.store),
        Settings {
            config_root: root.path().into(),
            config: config::load(root.path(), None).unwrap().config,
            kind: UnitKind::Chunk,
            budget: initial.settings.budget,
        },
    )
    .unwrap();
    assert!(matches!(
        factory
            .embed_once(" ".into(), Instant::now() + Duration::from_secs(2), || 100)
            .await,
        Err(admission::Error::Provider(AttemptError::InvalidInput))
    ));
    assert_eq!(counts(&factory), (0, 0, 0));
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(2))
                .set_body_json(serde_json::json!({"embeddings":[[3.0,4.0]],"prompt_eval_count":7})),
        )
        .expect(1)
        .mount(&server)
        .await;
    assert!(matches!(
        factory
            .embed_once(
                "slow query".into(),
                Instant::now() + Duration::from_secs(1),
                || 100
            )
            .await,
        Err(admission::Error::Deadline)
    ));
    assert_eq!(counts(&factory), (0, 1, 0));
    assert!(matches!(
        embed(&factory).await,
        Err(admission::Error::Budget { .. })
    ));
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

fn encoded(vector: &[f32]) -> String {
    base64::engine::general_purpose::STANDARD.encode(
        vector
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

#[tokio::test]
async fn retry_reuses_input_and_rechecks_configuration_before_the_next_dispatch() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    for changed in [false, true] {
        let server = MockServer::start().await;
        let (root, initial) = setup(&server, "ollama", 3);
        let cfg_path = root.path().join(".deepkit/config.toml");
        let config = format!(
            "{}\n[limits]\nquery_timeout_ms=6000\n",
            fs::read_to_string(&cfg_path).unwrap()
        );
        fs::write(&cfg_path, &config).unwrap();
        let factory = Factory::new(
            Arc::clone(&initial.store),
            Settings {
                config_root: root.path().into(),
                config: config::load(root.path(), None).unwrap().config,
                kind: UnitKind::Chunk,
                budget: initial.settings.budget,
            },
        )
        .unwrap();
        let calls = AtomicUsize::new(0);
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(move |_: &wiremock::Request| {
                if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    if changed {
                        fs::write(&cfg_path, config.replace("Represent: ", "Changed: ")).unwrap();
                    }
                    ResponseTemplate::new(503)
                } else {
                    ResponseTemplate::new(200).set_body_json(
                        serde_json::json!({"embeddings":[[3.0,4.0]],"prompt_eval_count":7}),
                    )
                }
            })
            .expect(if changed { 1 } else { 2 })
            .mount(&server)
            .await;
        let started = Instant::now();
        let result = factory
            .embed(
                "retry this query".into(),
                started + Duration::from_secs(6),
                || 100,
            )
            .await;
        assert!(started.elapsed() >= Duration::from_secs(1));
        if changed {
            assert!(matches!(
                result,
                Err(admission::Error::Refused(usage::Dispatch::Superseded))
            ));
            assert_eq!(counts(&factory), (0, 1, 0));
        } else {
            assert_eq!(result.unwrap().len(), 2);
            assert_eq!(counts(&factory), (0, 1, 7));
            let requests = server.received_requests().await.unwrap();
            assert_eq!(requests.len(), 2);
            assert_eq!(requests[0].body, requests[1].body);
        }
    }
}

#[tokio::test]
async fn automatic_dimensions_negotiation_keeps_both_attempts_in_the_ledger() {
    let server = MockServer::start().await;
    let (_root, factory) = setup(&server, "remote", 2);
    Mock::given(method("POST")).and(path("/embeddings"))
        .respond_with(|request: &wiremock::Request| {
            let input: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
            if input.get("dimensions").is_some() {
                ResponseTemplate::new(400).set_body_json(serde_json::json!({"error":{"message":"model does not support matryoshka representation"}}))
            } else {
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"data":[{"index":0,"embedding":encoded(&[3.0,4.0,8.0])}],"usage":{"total_tokens":9}}))
            }
        }).expect(2).mount(&server).await;
    assert_eq!(
        factory
            .embed(
                "negotiate".into(),
                Instant::now() + Duration::from_secs(2),
                || 100
            )
            .await
            .unwrap()
            .len(),
        2
    );
    assert_eq!(counts(&factory), (0, 1, 9));
    let dispatches: i64 = factory.store.writer().unwrap().query_row("SELECT sum(sent_units) FROM usage_reservations WHERE state IN ('dispatched','reconciled')",[],|r|r.get(0)).unwrap();
    assert_eq!(dispatches, 2);
}
