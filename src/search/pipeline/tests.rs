//! Shared candidate modes, fusion, exclusions and fallback; v1 §9 steps 1–6.
use super::*;
use crate::{
    config,
    queue::usage,
    search::tests::add,
    store::{
        self, StoreLimits,
        vectors::{VectorInsert, VectorTableSpec},
    },
};
use std::fs;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

pub(super) fn request(query: &str, mode: fusion::Mode) -> Request {
    Request {
        query: query.into(),
        mode,
        scope: None,
        k: 1,
        exclude_ids: Vec::new(),
    }
}
pub(super) fn fixture(
    server: &MockServer,
) -> (
    tempfile::TempDir,
    Arc<Store>,
    Config,
    Arc<embedding_query::Factory>,
    i64,
    i64,
    i64,
) {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join(".deepkit")).unwrap();
    fs::write(root.path().join(".deepkit/config.toml"),format!("[index]\nrepo_name='pipeline'\n[embedder]\nprofile='ollama'\ndims=2\n[embedder.ollama]\nurl='{}'\nmodel='pipeline'\n[rerank]\ncandidates=2\n",server.uri())).unwrap();
    let config = config::load(root.path(), None).unwrap().config;
    let store = Arc::new(store::open(":memory:", StoreLimits::default()).unwrap());
    let a = add(
        &store,
        "src/a.rs",
        "payment_retry",
        "invoice",
        Some("linked"),
    )
    .unwrap();
    let b = add(&store, "src/b.rs", "network", "network failure", None).unwrap();
    let c = add(&store, "src/c.rs", "linked", "other text", None).unwrap();
    let slug = store::vectors::embedding_table_slug("ollama", "pipeline", 2);
    store
        .ensure_vector_table(VectorTableSpec {
            slug: &slug,
            model: "pipeline",
            profile: "ollama",
            dims: 2,
            created_at: 1,
            active: true,
        })
        .unwrap();
    let binding = crate::queue::destination::resolve(&config, store::UnitKind::Chunk)
        .unwrap()
        .binding(&config);
    store.bind_embedding_table(&slug, &binding).unwrap();
    for (id, path, vector) in [(a, "src/a.rs", [0.0, 1.0]), (b, "src/b.rs", [1.0, 0.0])] {
        let hash = *blake3::hash(path.as_bytes()).as_bytes();
        store
            .insert_vector(VectorInsert {
                table_slug: &slug,
                unit_id: id,
                ctx_hash: &hash,
                embedding_hash: &hash,
                embedding: &vector,
            })
            .unwrap();
    }
    let factory = embedding_query::Factory::new(
        Arc::clone(&store),
        embedding_query::Settings {
            config_root: root.path().into(),
            config: config.clone(),
            kind: store::UnitKind::Chunk,
            budget: usage::Budget {
                calls: Some(10),
                tokens: None,
            },
        },
    )
    .unwrap();
    (root, store, config, factory, a, b, c)
}
pub(super) fn tier(factory: &Arc<embedding_query::Factory>) -> Option<Tier> {
    Some(Tier {
        factory: Arc::clone(factory),
        table: factory.table_slug(),
    })
}
fn ids(output: &Candidates) -> Vec<i64> {
    output.hits.iter().map(|hit| hit.unit.id).collect()
}

#[tokio::test]
async fn modes_and_typed_clauses_share_fusion_and_the_configured_window() {
    use fusion::Mode::*;
    let server = MockServer::start().await;
    let (_root, store, config, factory, a, b, c) = fixture(&server);
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]],"prompt_eval_count":3})),
        )
        .expect(3)
        .mount(&server)
        .await;
    for (mode, text, expected, dense_requested) in [
        (Lexical, "invoice", vec![a, c], false),
        (Dense, "invoice", vec![b, a], true),
        (Hybrid, "invoice", vec![a, b], true),
        (Dense, "lex:invoice", vec![a, c], false),
        (Lexical, "vec:invoice", vec![b, a], true),
    ] {
        let output = candidates(
            Arc::clone(&store),
            config.clone(),
            request(text, mode),
            tier(&factory),
            || 100,
        )
        .await
        .unwrap();
        assert_eq!(ids(&output), expected);
        assert_eq!(output.hits.len(), 2); // candidate window, not the final k=1.
        assert_eq!(
            matches!(output.dense, DenseState::Complete),
            dense_requested
        );
        assert_eq!(
            matches!(output.lexical, LexicalState::Complete),
            mode == Hybrid || !dense_requested
        );
        if mode == Hybrid {
            let hit = &output.hits[0];
            assert!(hit.lexical && hit.dense);
            assert!((hit.score - (1.0 / 61.0 + 1.0 / 62.0)).abs() < 1e-12);
            assert_eq!(hit.scores.lexical.as_ref().unwrap().rank, 1);
            assert_eq!(hit.scores.dense.as_ref().unwrap().rank, 2);
        }
        if !dense_requested {
            assert!(output.hits[1].graph);
            assert!((output.hits[1].score - 0.3 / 61.0).abs() < 1e-12);
        }
    }
    assert_eq!(server.received_requests().await.unwrap().len(), 3);
}

#[tokio::test]
async fn dense_failure_and_missing_tier_leave_visible_lexical_fallback() {
    let server = MockServer::start().await;
    let (_root, store, config, factory, a, _b, c) = fixture(&server);
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(ResponseTemplate::new(503).insert_header("Retry-After", "60"))
        .expect(1)
        .mount(&server)
        .await;
    let output = candidates(
        Arc::clone(&store),
        config.clone(),
        request("invoice", fusion::Mode::Hybrid),
        tier(&factory),
        || 100,
    )
    .await
    .unwrap();
    assert_eq!(ids(&output), [a, c]);
    assert!(matches!(output.dense, DenseState::Failed(_)));
    assert!(output.hits.iter().all(|hit| !hit.dense));
    let output = candidates(
        store,
        config,
        request("invoice", fusion::Mode::Hybrid),
        None,
        || 100,
    )
    .await
    .unwrap();
    assert_eq!(ids(&output), [a, c]);
    assert!(matches!(output.dense, DenseState::Unavailable));
}

#[tokio::test]
async fn exclusions_precede_lexical_and_exact_name_candidate_limits() {
    let store = Arc::new(store::open(":memory:", StoreLimits::default()).unwrap());
    let mut units = Vec::new();
    for i in 0..251 {
        units.push(
            add(
                &store,
                &format!("src/file{i}.rs"),
                "should_retry",
                "should_retry",
                None,
            )
            .unwrap(),
        );
    }
    units.sort_unstable();
    let expected = units.pop().unwrap();
    let mut query = request("should_retry", fusion::Mode::Lexical);
    query.exclude_ids = units;
    let output = candidates(store, Config::default(), query, None, || 100)
        .await
        .unwrap();
    assert_eq!(ids(&output), [expected]);
    assert!(output.hits[0].exact_name);
}

#[tokio::test]
async fn leading_documents_do_not_consume_the_twenty_symbol_graph_seeds() {
    let store = Arc::new(store::open(":memory:", StoreLimits::default()).unwrap());
    for i in 0..20 {
        let id = add(&store, &format!("docs/{i}.md"), "doc", "topic", None).unwrap();
        store
            .writer()
            .unwrap()
            .execute("UPDATE units SET kind='doc_section' WHERE id=?1", [id])
            .unwrap();
    }
    add(
        &store,
        "src/seed.rs",
        "seed",
        &format!("topic {}", "padding ".repeat(100)),
        Some("neighbour"),
    )
    .unwrap();
    let neighbour = add(
        &store,
        "src/neighbour.rs",
        "neighbour",
        "different content",
        None,
    )
    .unwrap();
    let output = candidates(
        Arc::clone(&store),
        Config::default(),
        request("topic", fusion::Mode::Lexical),
        None,
        || 100,
    )
    .await
    .unwrap();
    assert!(
        output
            .hits
            .iter()
            .any(|hit| hit.unit.id == neighbour && hit.graph)
    );
    let mut excluded = request("topic", fusion::Mode::Lexical);
    excluded.exclude_ids = vec![neighbour];
    let output = candidates(store, Config::default(), excluded, None, || 100)
        .await
        .unwrap();
    assert!(!output.hits.iter().any(|hit| hit.unit.id == neighbour));
}

#[tokio::test]
async fn unused_tier_does_not_block_lexical_work_and_invalid_fusion_never_dispatches() {
    let server = MockServer::start().await;
    let (_root, store, mut config, factory, a, _b, _c) = fixture(&server);
    config.search.graph_prior_weight = 0.0;
    let output = candidates(
        Arc::clone(&store),
        config.clone(),
        request("invoice", fusion::Mode::Lexical),
        tier(&factory),
        || 100,
    )
    .await
    .unwrap();
    assert_eq!(ids(&output), [a]);
    assert!(matches!(output.dense, DenseState::NotRequested));
    config.search.rrf_k = 0;
    assert!(matches!(
        candidates(
            store,
            config,
            request("invoice", fusion::Mode::Hybrid),
            tier(&factory),
            || 100
        )
        .await,
        Err(Error::InvalidFusion)
    ));
    assert!(server.received_requests().await.unwrap().is_empty());
}

fn contain_b_in_a(store: &Store, a: i64, b: i64) {
    store.writer().unwrap().execute(
        "UPDATE units SET kind='chunk',file_id=(SELECT file_id FROM units WHERE id=?1),byte_start=1,byte_end=3 WHERE id=?2",
        rusqlite::params![a,b],
    ).unwrap();
}

#[tokio::test]
async fn final_rrf_search_collapses_combined_legs_after_the_candidate_window() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "embeddings": [[1.0,0.0]], "prompt_eval_count": 4
        })))
        .expect(1)
        .mount(&server)
        .await;
    let (_root, store, config, factory, a, b, _c) = fixture(&server);
    contain_b_in_a(&store, a, b);
    let output = search_without_rerank(
        store,
        config,
        request("invoice", fusion::Mode::Hybrid),
        Some(Arc::clone(&factory)),
        || 100,
    )
    .await
    .unwrap();
    assert!(matches!(output.dense, DenseState::Complete));
    assert_eq!(output.hits.len(), 1);
    let hit = &output.hits[0];
    assert_eq!(hit.unit.id, a);
    assert!(hit.lexical && hit.dense && hit.chunk);
    assert_eq!(
        hit.matches
            .iter()
            .map(|hit| hit.unit_id)
            .collect::<Vec<_>>(),
        [a, b]
    );
    assert!((hit.score - (1.0 / 61.0 + 1.0 / 62.0)).abs() < 1e-12);
}

#[tokio::test]
async fn final_collapse_uses_captured_parents_and_preserves_later_ranking() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"embeddings": [[1.0,0.0]]})),
        )
        .mount(&server)
        .await;
    let (_root, store, config, factory, a, b, _c) = fixture(&server);
    contain_b_in_a(&store, a, b);
    let mut output = candidates(
        Arc::clone(&store),
        config.clone(),
        request("invoice", fusion::Mode::Hybrid),
        tier(&factory),
        || 100,
    )
    .await
    .unwrap();
    assert_eq!(ids(&output), [a, b]);
    output.hits.reverse(); // Models the order supplied by the subsequent reranker.
    let winning_score = output.hits[0].score;
    // Parent metadata is captured with the original occurrence, not reread during
    // final grouping. Holding the only in-memory connection also proves no I/O.
    let writer = store.writer().unwrap();
    writer
        .execute("UPDATE units SET header='changed' WHERE id=?1", [a])
        .unwrap();
    writer
        .execute("UPDATE units SET byte_end=1000 WHERE id=?1", [b])
        .unwrap();
    let output = output.finish(1, &config.search).unwrap();
    assert_eq!(output.hits[0].unit.id, a);
    assert_eq!(output.hits[0].unit.header, "payment_retry");
    assert_eq!(output.hits[0].score, winning_score);
    assert_eq!(
        output.hits[0]
            .matches
            .iter()
            .map(|hit| hit.unit_id)
            .collect::<Vec<_>>(),
        [b, a]
    );
}

#[tokio::test]
async fn final_search_returns_prepared_lexical_results_when_embedding_times_out() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(3))
                .set_body_json(serde_json::json!({"embeddings": [[1.0,0.0]]})),
        )
        .expect(1)
        .mount(&server)
        .await;
    let (root, store, mut config, _factory, a, _b, _c) = fixture(&server);
    config.limits.query_timeout_ms = 1000;
    let path = root.path().join(".deepkit/config.toml");
    fs::write(
        &path,
        format!(
            "{}\n[limits]\nquery_timeout_ms=1000\n",
            fs::read_to_string(&path).unwrap()
        ),
    )
    .unwrap();
    let factory = embedding_query::Factory::new(
        Arc::clone(&store),
        embedding_query::Settings {
            config_root: root.path().into(),
            config: config.clone(),
            kind: store::UnitKind::Chunk,
            budget: usage::Budget {
                calls: Some(10),
                tokens: None,
            },
        },
    )
    .unwrap();
    let start = Instant::now();
    let output = search_without_rerank(
        store,
        config,
        request("invoice", fusion::Mode::Hybrid),
        Some(Arc::clone(&factory)),
        || 100,
    )
    .await
    .unwrap();
    assert!(start.elapsed() < Duration::from_millis(2500));
    assert!(matches!(output.dense, DenseState::Failed(_)));
    assert_eq!(output.hits.len(), 1);
    assert_eq!(output.hits[0].unit.id, a);
    assert!(output.hits[0].lexical && !output.hits[0].dense);
}

#[tokio::test]
async fn failed_lexical_read_keeps_dense_hits_and_reports_degradation() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"embeddings": [[1.0,0.0]]})),
        )
        .expect(1)
        .mount(&server)
        .await;
    let (_root, store, config, factory, _a, b, _c) = fixture(&server);
    // A real SQLite failure in the lexical leg, without breaking vector storage.
    store
        .writer()
        .unwrap()
        .execute("DROP TABLE fts_units", [])
        .unwrap();
    let output = search_without_rerank(
        store,
        config,
        request("invoice", fusion::Mode::Hybrid),
        Some(Arc::clone(&factory)),
        || 100,
    )
    .await
    .unwrap();
    assert!(output.degraded());
    assert!(matches!(
        output.lexical,
        LexicalState::Failed(Error::Store(_))
    ));
    assert!(matches!(output.dense, DenseState::Complete));
    assert_eq!(output.hits.len(), 1);
    assert_eq!(output.hits[0].unit.id, b);
    assert!(output.hits[0].dense && !output.hits[0].lexical && !output.hits[0].graph);
    assert!((output.hits[0].score - 1.0 / 61.0).abs() < 1e-12);
}

#[tokio::test]
async fn failed_or_unavailable_legs_are_distinct_from_healthy_empty_searches() {
    let server = MockServer::start().await;
    let (_root, store, config, factory, _a, _b, _c) = fixture(&server);
    let empty = search_without_rerank(
        Arc::clone(&store),
        config.clone(),
        request("nonexistentterm", fusion::Mode::Lexical),
        Some(Arc::clone(&factory)),
        || 100,
    )
    .await
    .unwrap();
    assert!(empty.hits.is_empty());
    assert!(!empty.degraded());
    assert!(matches!(empty.lexical, LexicalState::Complete));
    assert!(matches!(empty.dense, DenseState::NotRequested));
    store
        .writer()
        .unwrap()
        .execute("DROP TABLE fts_units", [])
        .unwrap();
    let output = search_without_rerank(
        Arc::clone(&store),
        config.clone(),
        request("invoice", fusion::Mode::Hybrid),
        None,
        || 100,
    )
    .await
    .unwrap();
    assert!(output.hits.is_empty() && output.degraded());
    assert!(matches!(
        output.lexical,
        LexicalState::Failed(Error::Store(_))
    ));
    assert!(matches!(output.dense, DenseState::Unavailable));
    let output = search_without_rerank(
        store,
        config,
        request("vec:invoice", fusion::Mode::Lexical),
        None,
        || 100,
    )
    .await
    .unwrap();
    assert!(output.hits.is_empty() && output.degraded());
    assert!(matches!(output.lexical, LexicalState::NotRequested));
    assert!(matches!(output.dense, DenseState::Unavailable));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn lexical_reader_wait_returns_explicit_deadline_without_waiting_for_the_lock() {
    let store = Arc::new(store::open(":memory:", StoreLimits::default()).unwrap());
    let mut config = Config::default();
    config.limits.query_timeout_ms = 50;
    let held_store = Arc::clone(&store);
    let (ready, acquired) = tokio::sync::oneshot::channel();
    let (release, released) = std::sync::mpsc::channel();
    let holder = tokio::task::spawn_blocking(move || {
        let _lock = held_store.writer().unwrap();
        ready.send(()).unwrap();
        released.recv().unwrap();
    });
    acquired.await.unwrap();
    let result = tokio::time::timeout(
        Duration::from_secs(1),
        search_without_rerank(
            store,
            config,
            request("invoice", fusion::Mode::Lexical),
            None,
            || 100,
        ),
    )
    .await;
    release.send(()).unwrap();
    holder.await.unwrap();
    let output = result.unwrap().unwrap();
    assert!(output.hits.is_empty() && output.degraded());
    assert!(matches!(
        output.lexical,
        LexicalState::Failed(Error::Deadline)
    ));
    assert!(matches!(output.dense, DenseState::NotRequested));
}

#[tokio::test]
async fn final_search_uses_the_host_table_and_leaves_an_unrelated_space_alone() {
    use crate::queue::host;
    let server = MockServer::start().await;
    let (root, store, config, factory, _a, b, _c) = fixture(&server);
    let host = host::Host::prepare(
        Arc::clone(&store),
        host::Settings {
            root: root.path().into(),
            config_root: root.path().into(),
            config: config.clone(),
            kind: store::UnitKind::Chunk,
            local_cache: root.path().join("unused-model-cache"),
            bounds: host::Bounds {
                budget: usage::Budget {
                    calls: Some(10),
                    tokens: None,
                },
                timeout: Duration::from_secs(1),
                heartbeat: Duration::from_millis(100),
                max_schedule: 10,
                max_jobs: 10,
            },
        },
        100,
    )
    .await
    .unwrap();
    assert_eq!(host.table_slug, factory.table_slug());
    assert!(server.received_requests().await.unwrap().is_empty());
    store
        .ensure_vector_table(VectorTableSpec {
            slug: "unrelated_space",
            model: "another-model",
            profile: "remote",
            dims: 3,
            created_at: 1,
            active: true,
        })
        .unwrap();
    let hash = *blake3::hash(b"src/b.rs").as_bytes();
    store
        .insert_vector(VectorInsert {
            table_slug: "unrelated_space",
            unit_id: b,
            ctx_hash: &hash,
            embedding_hash: &hash,
            embedding: &[1.0, 0.0, 0.0],
        })
        .unwrap();
    let before = store.vector_tables().unwrap();
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"embeddings": [[1.0,0.0]]})),
        )
        .expect(1)
        .mount(&server)
        .await;
    let output = search_without_rerank(
        Arc::clone(&store),
        config,
        request("invoice", fusion::Mode::Dense),
        Some(factory),
        || 100,
    )
    .await
    .unwrap();
    assert!(!output.degraded());
    assert_eq!(output.hits.len(), 1);
    assert_eq!(output.hits[0].unit.id, b);
    assert!(matches!(output.lexical, LexicalState::NotRequested));
    assert!(matches!(output.dense, DenseState::Complete));
    assert_eq!(before, store.vector_tables().unwrap());
}

#[tokio::test]
async fn canonical_table_refusal_never_probes_an_alias_or_dispatches_embedding() {
    let server = MockServer::start().await;
    let (_root, store, config, factory, a, b, _c) = fixture(&server);
    let slug = factory.table_slug();
    store
        .ensure_vector_table(VectorTableSpec {
            slug: "matching_alias",
            model: "pipeline",
            profile: "ollama",
            dims: 2,
            created_at: 1,
            active: true,
        })
        .unwrap();
    let hash = *blake3::hash(b"src/b.rs").as_bytes();
    store
        .insert_vector(VectorInsert {
            table_slug: "matching_alias",
            unit_id: b,
            ctx_hash: &hash,
            embedding_hash: &hash,
            embedding: &[1.0, 0.0],
        })
        .unwrap();
    for change in [
        "active=0",
        "model='different'",
        "profile='remote'",
        "dims=3",
    ] {
        store
            .writer()
            .unwrap()
            .execute(
                &format!("UPDATE embedding_tables SET {change} WHERE slug=?1"),
                [&slug],
            )
            .unwrap();
        let output = search_without_rerank(
            Arc::clone(&store),
            config.clone(),
            request("invoice", fusion::Mode::Hybrid),
            Some(Arc::clone(&factory)),
            || 100,
        )
        .await
        .unwrap();
        assert!(output.degraded());
        assert!(matches!(
            output.dense,
            DenseState::Failed(dense::Error::TableChanged)
        ));
        assert_eq!(output.hits[0].unit.id, a);
        store.writer().unwrap().execute(
            "UPDATE embedding_tables SET active=1,model='pipeline',profile='ollama',dims=2 WHERE slug=?1", [&slug],
        ).unwrap();
    }
    store.drop_vector_table(&slug).unwrap();
    let output = search_without_rerank(
        Arc::clone(&store),
        config.clone(),
        request("invoice", fusion::Mode::Hybrid),
        Some(Arc::clone(&factory)),
        || 100,
    )
    .await
    .unwrap();
    assert!(output.degraded());
    assert!(matches!(
        output.dense,
        DenseState::Failed(dense::Error::TableChanged)
    ));
    assert_eq!(output.hits[0].unit.id, a);
    assert_eq!(store.vector_tables().unwrap().len(), 1); // Alias remains untouched.
    store
        .ensure_vector_table(VectorTableSpec {
            slug: &slug,
            model: "pipeline",
            profile: "ollama",
            dims: 2,
            created_at: 2,
            active: true,
        })
        .unwrap();
    let binding = crate::queue::destination::resolve(&config, store::UnitKind::Chunk)
        .unwrap()
        .binding(&config);
    store.bind_embedding_table(&slug, &binding).unwrap();
    let output = search_without_rerank(
        Arc::clone(&store),
        config,
        request("invoice", fusion::Mode::Hybrid),
        Some(factory),
        || 100,
    )
    .await
    .unwrap();
    assert!(!output.degraded()); // A valid but empty table is a completed empty leg.
    assert!(matches!(output.dense, DenseState::Complete));
    assert_eq!(output.hits[0].unit.id, a);
    assert!(server.received_requests().await.unwrap().is_empty());
    let reservations: i64 = store
        .writer()
        .unwrap()
        .query_row("SELECT count(*) FROM usage_reservations", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(reservations, 0);
}

mod routed;
mod selection;

#[tokio::test]
async fn inherited_expired_deadline_refuses_dense_dispatch_without_a_new_budget() {
    let server = MockServer::start().await;
    let (_root, store, config, factory, _, _, _) = fixture(&server);
    let owner =
        crate::search::control::Owner::new(std::time::Instant::now() - Duration::from_secs(1), 1)
            .unwrap();
    let output = candidates_with_control(
        store,
        config,
        request("invoice", fusion::Mode::Hybrid),
        tier(&factory),
        None,
        owner.control(),
        || 100,
    )
    .await
    .unwrap();
    assert_eq!(output.control.deadline().into_std(), owner.deadline);
    assert!(matches!(
        output.lexical,
        LexicalState::Failed(Error::Deadline)
    ));
    assert!(matches!(
        output.dense,
        DenseState::Failed(dense::Error::Deadline)
    ));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn cancellation_returns_while_writer_is_held_and_a_new_request_can_retry() {
    let store = Arc::new(store::open(":memory:", StoreLimits::default()).unwrap());
    let owner = crate::search::control::Owner::new(std::time::Instant::now(), 60_000).unwrap();
    let control = owner.control();
    let held_store = Arc::clone(&store);
    let (ready, acquired) = tokio::sync::oneshot::channel();
    let (release, released) = std::sync::mpsc::channel();
    let holder = tokio::task::spawn_blocking(move || {
        let _lock = held_store.writer().unwrap();
        ready.send(()).unwrap();
        released.recv().unwrap();
    });
    acquired.await.unwrap();
    let search_store = Arc::clone(&store);
    let task = tokio::spawn(async move {
        candidates_with_control(
            search_store,
            Config::default(),
            request("invoice", fusion::Mode::Lexical),
            None,
            None,
            control,
            || 100,
        )
        .await
    });
    // The writer stays held through cancellation and the async response.
    tokio::task::yield_now().await;
    drop(owner);
    let result = tokio::time::timeout(Duration::from_secs(1), task).await;
    release.send(()).unwrap();
    holder.await.unwrap();
    let output = result.unwrap().unwrap().unwrap();
    assert!(matches!(
        output.lexical,
        LexicalState::Failed(Error::Deadline)
    ));
    let retry = search_without_rerank(
        store,
        Config::default(),
        request("invoice", fusion::Mode::Lexical),
        None,
        || 100,
    )
    .await
    .unwrap();
    assert!(matches!(retry.lexical, LexicalState::Complete));
}
