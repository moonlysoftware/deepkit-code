//! Routed lists retain scores, partial errors and collapse through final search.
use super::*;
use crate::{
    queue::destination,
    store::{UnitKind, transitions},
};

struct Fixture {
    root: tempfile::TempDir,
    store: Arc<Store>,
    config: Config,
    factories: Vec<Arc<embedding_query::Factory>>,
    old: String,
    new: String,
    a: i64,
    b: i64,
    c: i64,
}
fn routes(table: &str) -> transitions::Routes {
    UnitKind::ALL
        .into_iter()
        .map(|kind| (kind.as_str().into(), table.into()))
        .collect()
}
async fn setup(first: &MockServer, second: &MockServer) -> Fixture {
    let (root, store, mut config, old_factory, a, b, c) = fixture(first);
    let old = old_factory.table_slug();
    store
        .prepare_transition(|_| Ok::<_, store::Error>(routes(&old)))
        .unwrap();
    config.embedder.ollama.url = second.uri();
    config.embedder.ollama.model = "new-pipeline".into();
    config.rerank.candidates = 10;
    fs::write(
        root.path().join(".deepkit/config.toml"),
        toml::to_string(&config).unwrap(),
    )
    .unwrap();
    let binding = destination::resolve(&config, UnitKind::Chunk)
        .unwrap()
        .binding(&config);
    let new = binding.destination_slug();
    store
        .ensure_vector_table(VectorTableSpec {
            slug: &new,
            model: &binding.model,
            profile: &binding.profile,
            dims: 2,
            created_at: 0,
            active: true,
        })
        .unwrap();
    store.bind_embedding_table(&new, &binding).unwrap();
    let state = store
        .prepare_transition(|_| Ok::<_, store::Error>(routes(&new)))
        .unwrap();
    let settings = || embedding_query::Settings {
        config_root: root.path().into(),
        config: config.clone(),
        kind: UnitKind::Symbol,
        budget: usage::Budget {
            calls: Some(20),
            tokens: Some(10000),
        },
    };
    let current = embedding_query::Factory::new(Arc::clone(&store), settings()).unwrap();
    let previous = embedding_query::Factory::new_previous(
        Arc::clone(&store),
        settings(),
        old.clone(),
        state.generation,
        root.path().join("unused"),
    )
    .await
    .unwrap();
    let hash: Vec<u8> = store
        .writer()
        .unwrap()
        .query_row("SELECT content_hash FROM units WHERE id=?1", [a], |r| {
            r.get(0)
        })
        .unwrap();
    store
        .replace_vector(VectorInsert {
            table_slug: &new,
            unit_id: a,
            ctx_hash: &[0; 32],
            embedding_hash: &hash,
            embedding: &[1.0, 0.0],
        })
        .unwrap();
    Fixture {
        root,
        store,
        config,
        factories: vec![current, previous],
        old,
        new,
        a,
        b,
        c,
    }
}
impl Fixture {
    fn providers(&self) -> dense::routed::Providers {
        dense::routed::Providers {
            store: Arc::clone(&self.store),
            config: self.config.clone(),
            config_root: self.root.path().into(),
            factories: self.factories.clone(),
        }
    }
}
fn input(query: &str, mode: fusion::Mode) -> Request {
    Request {
        k: 3,
        ..request(query, mode)
    }
}
fn response() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]]}))
}
#[tokio::test]
async fn routed_final_search_fuses_original_ranks_and_retains_verbose_match_evidence() {
    let first = MockServer::start().await;
    let second = MockServer::start().await;
    let fixture = setup(&first, &second).await;
    for server in [&first, &second] {
        Mock::given(method("POST"))
            .respond_with(response())
            .expect(1)
            .mount(server)
            .await;
    }
    let candidates = candidates_routed(
        fixture.providers(),
        input("invoice", fusion::Mode::Hybrid),
        || 100,
    )
    .await
    .unwrap();
    assert!(matches!(candidates.dense, DenseState::Complete));
    // Final collapse consumes the already captured metadata, with no late read.
    fixture
        .store
        .writer()
        .unwrap()
        .execute("DELETE FROM units", [])
        .unwrap();
    let output = candidates.finish(3, &fixture.config.search).unwrap();
    assert_eq!(output.dense_selection.single_table, None);
    assert!(output.dense_selection.single.is_empty());
    assert_eq!(output.dense_selection.spaces.len(), 2);
    assert!(
        output
            .dense_selection
            .spaces
            .contains(&(fixture.new.clone(), vec![fixture.a]))
    );
    assert!(
        output
            .dense_selection
            .spaces
            .contains(&(fixture.old.clone(), vec![fixture.b]))
    );
    assert!(output.selection_legs[0].links.contains_key(&fixture.c));
    assert!(output.selection_legs[1].links.contains_key(&fixture.b));
    assert_eq!(output.selection.selected[0]["unit"], fixture.a);
    assert!(!output.degraded());
    assert_eq!(
        output.hits.iter().map(|h| h.unit.id).collect::<Vec<_>>(),
        vec![fixture.a, fixture.b, fixture.c]
    );
    let a = &output.hits[0];
    let b = &output.hits[1];
    assert!((a.score - 2.0 / 61.0).abs() < 1e-12);
    assert!((b.score - 1.0 / 61.0).abs() < 1e-12);
    assert_eq!(a.scores.dense_spaces[&fixture.new].rank, 1);
    assert_eq!(b.scores.dense_spaces[&fixture.old].rank, 1);
    assert_eq!(a.matches[0].rrf.dense_spaces[&fixture.new].rank, 1);
    let verbose = serde_json::to_value(&b.scores).unwrap();
    assert_eq!(verbose["dense_spaces"][&fixture.old]["rank"], 1);
}
#[tokio::test]
async fn routed_partial_failures_preserve_available_dense_and_lexical_hits() {
    for fail_new in [false, true] {
        let first = MockServer::start().await;
        let second = MockServer::start().await;
        let fixture = setup(&first, &second).await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(400))
            .expect(1)
            .mount(&first)
            .await;
        Mock::given(method("POST"))
            .respond_with(if fail_new {
                ResponseTemplate::new(400)
            } else {
                response()
            })
            .expect(1)
            .mount(&second)
            .await;
        let output = search_routed_without_rerank(
            fixture.providers(),
            input("invoice", fusion::Mode::Hybrid),
            || 100,
        )
        .await
        .unwrap();
        assert!(output.degraded());
        assert_eq!(
            output.hits.iter().map(|h| h.unit.id).collect::<Vec<_>>(),
            vec![fixture.a, fixture.c]
        );
        let DenseState::Partial(failures) = output.dense else {
            panic!("expected partial dense work")
        };
        assert_eq!(failures.len(), if fail_new { 2 } else { 1 });
        assert!(failures.iter().any(|f| f.table == fixture.old));
        assert!(
            failures
                .iter()
                .all(|f| matches!(f.error, dense::Error::Query(_)))
        );
        assert_eq!(output.hits[0].dense, !fail_new);
        assert_eq!(
            output.hits[0].scores.dense_spaces.len(),
            usize::from(!fail_new)
        );
    }
}
#[tokio::test]
async fn routed_modes_and_exclusions_use_the_shared_pipeline() {
    let first = MockServer::start().await;
    let second = MockServer::start().await;
    let fixture = setup(&first, &second).await;
    for server in [&first, &second] {
        Mock::given(method("POST"))
            .respond_with(response())
            .expect(1)
            .mount(server)
            .await;
    }
    let lexical = search_routed_without_rerank(
        fixture.providers(),
        input("lex:invoice", fusion::Mode::Dense),
        || 100,
    )
    .await
    .unwrap();
    assert!(matches!(lexical.dense, DenseState::NotRequested));
    assert!(first.received_requests().await.unwrap().is_empty());
    assert!(second.received_requests().await.unwrap().is_empty());
    let mut request = input("vec:network", fusion::Mode::Lexical);
    request.exclude_ids = vec![fixture.a];
    let dense = search_routed_without_rerank(fixture.providers(), request, || 100)
        .await
        .unwrap();
    assert!(matches!(dense.lexical, LexicalState::NotRequested));
    assert_eq!(
        dense.hits.iter().map(|h| h.unit.id).collect::<Vec<_>>(),
        vec![fixture.b]
    );
    assert_eq!(dense.hits[0].scores.dense_spaces[&fixture.old].rank, 1);
}
#[tokio::test]
async fn routed_dense_results_survive_lexical_failure() {
    let first = MockServer::start().await;
    let second = MockServer::start().await;
    let fixture = setup(&first, &second).await;
    fixture
        .store
        .writer()
        .unwrap()
        .execute_batch("DROP TABLE fts_units")
        .unwrap();
    for server in [&first, &second] {
        Mock::given(method("POST"))
            .respond_with(response())
            .expect(1)
            .mount(server)
            .await;
    }
    let output = search_routed_without_rerank(
        fixture.providers(),
        input("invoice", fusion::Mode::Hybrid),
        || 100,
    )
    .await
    .unwrap();
    assert!(output.degraded());
    assert!(matches!(output.lexical, LexicalState::Failed(_)));
    assert!(matches!(output.dense, DenseState::Complete));
    assert_eq!(
        output.hits.iter().map(|h| h.unit.id).collect::<Vec<_>>(),
        vec![fixture.a, fixture.b]
    );
    assert!(
        output
            .hits
            .iter()
            .all(|h| h.dense && !h.lexical && h.scores.dense_spaces.len() == 1)
    );
}

#[tokio::test]
async fn routed_spaces_borrow_cancellation_without_dispatching_after_owner_drop() {
    let first = MockServer::start().await;
    let second = MockServer::start().await;
    let fixture = setup(&first, &second).await;
    let owner = crate::search::control::Owner::new(std::time::Instant::now(), 60_000).unwrap();
    let control = owner.control();
    drop(owner);
    let output = candidates_routed_with_control(
        fixture.providers(),
        request("invoice", fusion::Mode::Hybrid),
        None,
        control,
        || 100,
    )
    .await
    .unwrap();
    assert!(matches!(
        output.dense,
        DenseState::Failed(dense::Error::Deadline)
    ));
    assert!(first.received_requests().await.unwrap().is_empty());
    assert!(second.received_requests().await.unwrap().is_empty());
}
