//! Real old-provider requests with current admission and transition fences.
use super::*;
use crate::store::vectors::{VectorInsert, VectorTableSpec};
use base64::Engine as _;
use std::{
    fs,
    sync::atomic::{AtomicUsize, Ordering},
};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

fn routes(table: &str) -> transitions::Routes {
    UnitKind::ALL
        .into_iter()
        .map(|kind| (kind.as_str().into(), table.into()))
        .collect()
}
fn fixture(server: &MockServer, profile: &str) -> (tempfile::TempDir, Arc<Store>, Config, i64) {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join(".deepkit")).unwrap();
    let store = store::open(":memory:", Default::default()).unwrap();
    let hash = blake3::hash(b"fixture");
    store
        .writer()
        .unwrap()
        .execute(
            "INSERT INTO unit_contents(hash,text,tokens) VALUES(?1,'fixture',1)",
            [hash.as_bytes().as_slice()],
        )
        .unwrap();
    store.writer().unwrap().execute("INSERT INTO units(id,hash,content_hash,source_key,kind,header,created_at,updated_at) VALUES(1,?1,?1,'fixture','chunk','fixture',0,0)",[hash.as_bytes().as_slice()]).unwrap();
    let mut old = Config::default();
    old.embedder.profile = profile.into();
    old.embedder.dims = 2;
    old.embedder.remote.url = format!("{}/old/v1", server.uri());
    old.embedder.remote.model = "old-model".into();
    old.embedder.remote.api_key_env = "DKC_OLD_QUERY_KEY".into();
    old.embedder.remote.query_prefix = "Old: ".into();
    old.embedder.ollama.url = format!("{}/old", server.uri());
    old.embedder.ollama.model = "old-model".into();
    old.embedder.ollama.query_prefix = "Old: ".into();
    let binding = destination::resolve(&old, UnitKind::Chunk)
        .unwrap()
        .binding(&old);
    store
        .ensure_vector_table(VectorTableSpec {
            slug: "old",
            model: &binding.model,
            profile: &binding.profile,
            dims: 2,
            created_at: 0,
            active: true,
        })
        .unwrap();
    store.bind_embedding_table("old", &binding).unwrap();
    store
        .prepare_transition(|_| Ok::<_, store::Error>(routes("old")))
        .unwrap();
    let hash: Vec<u8> = store
        .writer()
        .unwrap()
        .query_row("SELECT content_hash FROM units WHERE id=1", [], |r| {
            r.get(0)
        })
        .unwrap();
    store
        .insert_vector(VectorInsert {
            table_slug: "old",
            unit_id: 1,
            ctx_hash: &[0; 32],
            embedding_hash: &hash,
            embedding: &[1.0, 0.0],
        })
        .unwrap();
    let transition = store
        .prepare_transition(|_| Ok::<_, store::Error>(routes("new")))
        .unwrap();
    let mut current = old;
    current.embedder.profile = "local".into();
    current.embedder.remote.url = format!("{}/new/v1", server.uri());
    current.embedder.remote.model = "new-model".into();
    current.embedder.remote.query_prefix = "New: ".into();
    current.embedder.remote.api_key_env = "DKC_NEW_QUERY_KEY".into();
    current.embedder.remote.rpm = 1;
    current.embedder.ollama.url = format!("{}/new", server.uri());
    current.embedder.ollama.model = "new-model".into();
    current.embedder.ollama.query_prefix = "New: ".into();
    fs::write(
        root.path().join(".deepkit/config.toml"),
        toml::to_string(&current).unwrap(),
    )
    .unwrap();
    (root, Arc::new(store), current, transition.generation)
}
fn settings(root: &std::path::Path, config: Config, calls: i64) -> Settings {
    Settings {
        config_root: root.into(),
        config,
        kind: UnitKind::Chunk,
        budget: usage::Budget {
            calls: Some(calls),
            tokens: Some(1000),
        },
    }
}
async fn build(
    root: &std::path::Path,
    store: Arc<Store>,
    config: Config,
    generation: i64,
    calls: i64,
) -> Result<Arc<Factory>, queue::Error> {
    Factory::build_previous(
        store,
        settings(root, config, calls),
        "old".into(),
        generation,
        root.join("no-cache"),
        |cfg| {
            assert_eq!(cfg.api_key_env, "DKC_OLD_QUERY_KEY");
            OpenAiEmbedder::with_api_key(cfg, String::new())
        },
    )
    .await
}
async fn embed(factory: &Arc<Factory>, now: i64) -> Result<Vec<f32>, admission::Error> {
    factory
        .embed_once(
            "find retries".into(),
            Instant::now() + Duration::from_secs(2),
            move || now,
        )
        .await
}

#[tokio::test]
async fn previous_http_recipes_use_old_identity_and_current_admission() {
    for profile in ["remote", "ollama"] {
        let server = MockServer::start().await;
        let (root, store, config, generation) = fixture(&server, profile);
        let response = if profile == "remote" {
            serde_json::json!({"data":[{"index":0,"embedding":base64::engine::general_purpose::STANDARD.encode([1.0_f32,0.0].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>())}],"usage":{"total_tokens":7}})
        } else {
            serde_json::json!({"embeddings":[[1.0,0.0]],"prompt_eval_count":7})
        };
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(response))
            .expect(1)
            .mount(&server)
            .await;
        let factory = build(
            root.path(),
            Arc::clone(&store),
            config.clone(),
            generation,
            1,
        )
        .await
        .unwrap();
        assert!(factory.matches_configuration(&config));
        let hits = crate::search::dense::retrieve(
            Arc::clone(&factory),
            "old".into(),
            "find retries".into(),
            Default::default(),
            Instant::now() + Duration::from_secs(2),
            || 100,
        )
        .await
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].unit_id, 1);
        assert!(embed(&factory, 100).await.is_err()); // explicit current daily budget
        let requests = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["model"], "old-model");
        assert_eq!(body["input"], serde_json::json!(["Old: find retries"]));
        assert_eq!(
            requests[0].url.path(),
            if profile == "remote" {
                "/old/v1/embeddings"
            } else {
                "/old/api/embed"
            }
        );
        let usage: (String, i64) = store
            .writer()
            .unwrap()
            .query_row(
                "SELECT endpoint_url,tokens FROM usage_reservations WHERE state='reconciled'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert!(usage.0.ends_with(requests[0].url.path()));
        assert_eq!(usage.1, 7);
        if profile == "remote" {
            let larger = build(root.path(), Arc::clone(&store), config, generation, 10)
                .await
                .unwrap();
            assert!(matches!(
                embed(&larger, 100).await,
                Err(admission::Error::Refused(
                    usage::Dispatch::RateLimited { .. }
                ))
            ));
        }
    }
}

#[tokio::test]
async fn old_source_must_remain_authorized_at_dispatch_and_scan() {
    for change in ["generation", "binding", "kind", "inactive", "config"] {
        let server = MockServer::start().await;
        let (root, store, config, generation) = fixture(&server, "ollama");
        let factory = build(
            root.path(),
            Arc::clone(&store),
            config.clone(),
            generation,
            10,
        )
        .await
        .unwrap();
        match change {
            "generation" => {
                store
                    .writer()
                    .unwrap()
                    .execute(
                        "UPDATE embedding_transition SET generation=generation+1",
                        [],
                    )
                    .unwrap();
            }
            "binding" => {
                store
                    .writer()
                    .unwrap()
                    .execute(
                        "UPDATE embedding_bindings SET query_prefix='changed' WHERE slug='old'",
                        [],
                    )
                    .unwrap();
            }
            "kind" => {
                store.writer().unwrap().execute("UPDATE embedding_transition SET previous=json_set(previous,'$[0].kinds',json('[\"symbol\"]'))",[]).unwrap();
            }
            "inactive" => {
                store
                    .writer()
                    .unwrap()
                    .execute("UPDATE embedding_tables SET active=0 WHERE slug='old'", [])
                    .unwrap();
            }
            _ => {
                let mut changed = config;
                changed.embedder.breadcrumb = false;
                fs::write(
                    root.path().join(".deepkit/config.toml"),
                    toml::to_string(&changed).unwrap(),
                )
                .unwrap();
            }
        }
        assert!(embed(&factory, 100).await.is_err(), "{change}");
        assert!(
            crate::search::dense::retrieve(
                Arc::clone(&factory),
                "old".into(),
                "question".into(),
                Default::default(),
                Instant::now() + Duration::from_secs(2),
                || 100
            )
            .await
            .is_err(),
            "{change}"
        );
        assert!(server.received_requests().await.unwrap().is_empty());
        assert_eq!(store.writer().unwrap().query_row("SELECT COUNT(*) FROM usage_reservations WHERE state IN ('dispatched','reconciled')",[],|r|r.get::<_,i64>(0)).unwrap(),0);
    }
}

#[tokio::test]
async fn unknown_or_obsolete_source_is_rejected_before_credentials_or_model_loading() {
    for change in ["unknown", "generation", "kind", "inactive"] {
        let server = MockServer::start().await;
        let (root, store, config, generation) = fixture(&server, "remote");
        match change {
            "unknown" => {
                store
                    .writer()
                    .unwrap()
                    .execute("DELETE FROM embedding_bindings", [])
                    .unwrap();
            }
            "kind" => {
                store.writer().unwrap().execute("UPDATE embedding_transition SET previous=json_set(previous,'$[0].kinds',json('[\"symbol\"]'))",[]).unwrap();
            }
            "inactive" => {
                store
                    .writer()
                    .unwrap()
                    .execute("UPDATE embedding_tables SET active=0", [])
                    .unwrap();
            }
            _ => {}
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let called = Arc::clone(&calls);
        let result = Factory::build_previous(
            Arc::clone(&store),
            settings(root.path(), config, 10),
            "old".into(),
            generation + i64::from(change == "generation"),
            root.path().join("no-cache"),
            move |cfg| {
                called.fetch_add(1, Ordering::SeqCst);
                OpenAiEmbedder::with_api_key(cfg, String::new())
            },
        )
        .await;
        assert!(result.is_err(), "{change}");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(server.received_requests().await.unwrap().is_empty());
    }
}

#[tokio::test]
async fn previous_scan_is_limited_to_affected_kinds_before_top_k() {
    let server = MockServer::start().await;
    let (root, store, config, generation) = fixture(&server, "ollama");
    store.writer().unwrap().execute("UPDATE embedding_transition SET previous=json_set(previous,'$[0].kinds',json('[\"chunk\"]'))",[]).unwrap();
    store.writer().unwrap().execute("INSERT INTO units(id,hash,content_hash,source_key,kind,header,created_at,updated_at) SELECT 2,randomblob(32),content_hash,'symbol','symbol','fixture',0,0 FROM units WHERE id=1",[]).unwrap();
    let hash: Vec<u8> = store
        .writer()
        .unwrap()
        .query_row("SELECT content_hash FROM units WHERE id=2", [], |r| {
            r.get(0)
        })
        .unwrap();
    store
        .insert_vector(VectorInsert {
            table_slug: "old",
            unit_id: 2,
            ctx_hash: &[0; 32],
            embedding_hash: &hash,
            embedding: &[1.0, 0.0],
        })
        .unwrap();
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]]})),
        )
        .expect(1)
        .mount(&server)
        .await;
    let factory = build(root.path(), store, config, generation, 10)
        .await
        .unwrap();
    let hits = crate::search::dense::retrieve(
        Arc::clone(&factory),
        "old".into(),
        "question".into(),
        Default::default(),
        Instant::now() + Duration::from_secs(2),
        || 100,
    )
    .await
    .unwrap();
    assert_eq!(hits.iter().map(|h| h.unit_id).collect::<Vec<_>>(), vec![1]);
    let hits = crate::search::dense::retrieve(
        factory,
        "old".into(),
        "question".into(),
        crate::search::dense::Scope {
            kinds: vec![UnitKind::Symbol],
            ..Default::default()
        },
        Instant::now() + Duration::from_secs(2),
        || 100,
    )
    .await
    .unwrap();
    assert!(hits.is_empty());
}

#[tokio::test]
async fn previous_local_identity_must_match_registry_and_verified_cache() {
    let server = MockServer::start().await;
    let (root, store, config, generation) = fixture(&server, "local");
    let cache = root.path().join("cache");
    fs::create_dir(&cache).unwrap();
    fs::write(
        cache.join(crate::models::llama::EMBEDDING_MODEL_FILE),
        b"invalid model",
    )
    .unwrap();
    assert!(matches!(
        Factory::new_previous(
            Arc::clone(&store),
            settings(root.path(), config.clone(), 10),
            "old".into(),
            generation,
            cache.clone()
        )
        .await,
        Err(queue::Error::Invalid(
            "local query artifact or model initialization"
        ))
    ));
    store
        .writer()
        .unwrap()
        .execute(
            "UPDATE embedding_bindings SET artifact_hash=zeroblob(32)",
            [],
        )
        .unwrap();
    assert!(matches!(
        Factory::new_previous(
            Arc::clone(&store),
            settings(root.path(), config, 10),
            "old".into(),
            generation,
            cache
        )
        .await,
        Err(queue::Error::Invalid(
            "previous provider identity cannot be reconstructed"
        ))
    ));
    assert!(server.received_requests().await.unwrap().is_empty());
    assert_eq!(
        store
            .writer()
            .unwrap()
            .query_row("SELECT COUNT(*) FROM usage_reservations", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[tokio::test]
async fn completed_transition_during_http_discards_result_but_keeps_actual_usage() {
    let server = MockServer::start().await;
    let (root, store, config, generation) = fixture(&server, "ollama");
    let factory = build(root.path(), Arc::clone(&store), config, generation, 10)
        .await
        .unwrap();
    let changed = Arc::clone(&store);
    Mock::given(method("POST"))
        .respond_with(move |_: &wiremock::Request| {
            changed.delete_vector("old", 1).unwrap();
            assert!(
                changed
                    .finish_transition(generation)
                    .unwrap()
                    .previous
                    .is_empty()
            );
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]],"prompt_eval_count":7}))
        })
        .expect(1)
        .mount(&server)
        .await;
    assert!(matches!(
        crate::search::dense::retrieve(
            factory,
            "old".into(),
            "question".into(),
            Default::default(),
            Instant::now() + Duration::from_secs(2),
            || 100
        )
        .await,
        Err(crate::search::dense::Error::TableChanged)
    ));
    let usage: (i64, i64) = store
        .writer()
        .unwrap()
        .query_row(
            "SELECT COUNT(*),SUM(tokens) FROM usage_reservations WHERE state='reconciled'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(usage, (1, 7));
}

#[tokio::test]
#[ignore = "requires DKC_MODELS_DIR with verified local embedding weights; CPU only"]
async fn previous_local_query_uses_verified_old_space_and_local_accounting() {
    let server = MockServer::start().await;
    let (root, store, mut config, generation) = fixture(&server, "local");
    config.embedder.profile = "ollama".into();
    config.embedder.local.gpu_layers = 0;
    config.embedder.local.threads = 2;
    config.embedder.local.n_ctx = 512;
    config.embedder.local.batch_tokens = 512;
    config.embedder.local.query_prefix = "New instruction: ".into();
    config.limits.query_timeout_ms = 5000;
    fs::write(
        root.path().join(".deepkit/config.toml"),
        toml::to_string(&config).unwrap(),
    )
    .unwrap();
    let cache = std::env::var_os("DKC_MODELS_DIR").expect("verified model cache");
    let factory = Factory::new_previous(
        Arc::clone(&store),
        settings(root.path(), config, 10),
        "old".into(),
        generation,
        cache.into(),
    )
    .await
    .unwrap();
    let query = "where are failed payments retried?";
    let vector = factory
        .embed_once(
            query.into(),
            Instant::now() + Duration::from_secs(5),
            || 100,
        )
        .await
        .unwrap();
    let Provider::Local(provider) = &factory.provider else {
        panic!("local provider")
    };
    let reference = crate::models::Embedder::embed_query(provider, query)
        .await
        .unwrap();
    assert!(
        reference
            .iter()
            .zip(&vector)
            .all(|(a, b)| (a - b).abs() < 0.00001)
    );
    assert_eq!(vector.len(), 2);
    let hash: Vec<u8> = store
        .writer()
        .unwrap()
        .query_row("SELECT content_hash FROM units WHERE id=1", [], |r| {
            r.get(0)
        })
        .unwrap();
    store
        .insert_vector(VectorInsert {
            table_slug: "old",
            unit_id: 1,
            ctx_hash: &[0; 32],
            embedding_hash: &hash,
            embedding: &vector,
        })
        .unwrap();
    let hits = crate::search::dense::retrieve(
        factory,
        "old".into(),
        query.into(),
        Default::default(),
        Instant::now() + Duration::from_secs(5),
        || 100,
    )
    .await
    .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].unit_id, 1);
    assert!(hits[0].distance.abs() < 0.001);
    let usage: (i64, i64, i64) = store
        .writer()
        .unwrap()
        .query_row(
            "SELECT calls,tokens,estimated_tokens FROM usage WHERE provider='embed:local'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!(usage.0, 2);
    assert!(usage.1 > 0);
    assert_eq!(usage.1, usage.2);
    assert_eq!(
        store
            .writer()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM usage_reservations WHERE endpoint_url IS NOT NULL",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
    assert!(server.received_requests().await.unwrap().is_empty());
    println!(
        "CPU previous local query: dims=2, retrieved=1, calls={}, estimated_tokens={}, outbound_endpoints=0",
        usage.0, usage.2
    );
}
