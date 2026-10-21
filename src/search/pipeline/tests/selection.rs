//! Retained selection authority across asynchronous ranking/collapse; v1 §9.
use super::*;

fn complete_sources(store: &Store) {
    store
        .writer()
        .unwrap()
        .execute(
            "UPDATE files SET l1_desired=blake3,l1_fingerprint=blake3,l1_pending=0",
            [],
        )
        .unwrap();
}

#[tokio::test]
async fn lexical_evidence_matches_checked_service_and_ignores_ladder_progress() {
    let server = MockServer::start().await;
    let (_root, store, config, _factory, a, _b, c) = fixture(&server);
    complete_sources(&store);
    let input = || request("payment_retry", fusion::Mode::Lexical);
    let retrieve =
        || search_without_rerank(Arc::clone(&store), config.clone(), input(), None, || 100);
    let before = retrieve().await.unwrap();
    let parsed = query::parse(&input().query, input().mode).unwrap();
    let (_, checked) = crate::search::lexical_with_selection(
        &store,
        &parsed,
        1,
        None,
        &config.search,
        &[],
        &|| Ok(()),
    )
    .unwrap();
    assert!(before.selection.leg == checked.leg);
    assert!(!before.selection.leg.exact_associations.is_empty());
    assert!(!before.selection.leg.graph_associations.is_empty());
    assert_eq!(before.selection.projections, checked.projections);
    assert_eq!(before.selection.links, checked.links);
    assert_eq!(before.selection.publications, checked.publications);
    assert_eq!(before.selection.selected, checked.selected);
    assert_eq!(before.selection.incomplete_hidden_hits, None);
    assert_eq!(before.dense_selection, DenseLists::default());
    assert!(before.selection_legs[1].publications.is_empty());
    assert!(before.selection.occurrences.iter().any(|unit| unit.id == c));
    assert_eq!(before.hits[0].unit.id, a);
    store
        .writer()
        .unwrap()
        .execute("UPDATE units SET ladder_level=2", [])
        .unwrap();
    let after = retrieve().await.unwrap();
    assert!(before.selection == after.selection);
    assert!(
        after
            .selection
            .occurrences
            .iter()
            .all(|unit| unit.ladder_level == 0)
    );
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn hidden_candidate_publication_is_captured_before_window_and_finish() {
    let server = MockServer::start().await;
    let (_root, store, mut config, _factory, a, _b, c) = fixture(&server);
    config.rerank.candidates = 1;
    complete_sources(&store);
    let retrieve = || {
        candidates(
            Arc::clone(&store),
            config.clone(),
            request("invoice", fusion::Mode::Lexical),
            None,
            || 100,
        )
    };
    let before = retrieve().await.unwrap();
    assert_eq!(ids(&before), [a]);
    let before = {
        let writer = store.writer().unwrap();
        writer
            .execute("UPDATE files SET l1_pending=1 WHERE path='src/c.rs'", [])
            .unwrap();
        // Finish must not acquire a reader or substitute this newly pending record.
        before.finish(1, &config.search).unwrap()
    };
    assert_eq!(before.selection.incomplete_hidden_hits, None);
    assert!(
        before
            .selection
            .publications
            .values()
            .all(|entry| entry.as_ref().unwrap().is_complete())
    );
    assert!(before.selection.projections.iter().any(|unit| unit.id == c));
    let after = retrieve().await.unwrap().finish(1, &config.search).unwrap();
    assert_eq!(after.hits[0].unit.id, a);
    assert_eq!(after.selection.selected, before.selection.selected);
    assert_eq!(
        after.selection.incomplete_hidden_hits,
        Some(BTreeSet::new())
    );
    assert_ne!(after.selection.publications, before.selection.publications);
}

#[tokio::test]
async fn collapsed_hidden_source_keeps_original_parent_and_incomplete_authority() {
    let server = MockServer::start().await;
    let (_root, store, config, _factory, a, b, c) = fixture(&server);
    complete_sources(&store);
    {
        let writer = store.writer().unwrap();
        writer
            .execute(
                "UPDATE units SET kind='summary',parent_unit=?1 WHERE id=?2",
                [a, b],
            )
            .unwrap();
        writer
            .execute("UPDATE files SET l1_pending=1 WHERE path='src/b.rs'", [])
            .unwrap();
    }
    let retrieve = || {
        candidates(
            Arc::clone(&store),
            config.clone(),
            request("network", fusion::Mode::Lexical),
            None,
            || 100,
        )
    };
    let before = retrieve().await.unwrap();
    let before = {
        let writer = store.writer().unwrap();
        writer
            .execute("UPDATE units SET parent_unit=?1 WHERE id=?2", [c, b])
            .unwrap();
        writer
            .execute("UPDATE units SET header='replacement' WHERE id=?1", [a])
            .unwrap();
        writer
            .execute("UPDATE files SET l1_pending=0 WHERE path='src/b.rs'", [])
            .unwrap();
        before.finish(1, &config.search).unwrap()
    };
    assert_eq!(before.hits[0].unit.id, a);
    assert_eq!(before.hits[0].unit.header, "payment_retry");
    assert_eq!(
        before.selection.incomplete_hidden_hits,
        Some(BTreeSet::from([a]))
    );
    assert_eq!(before.selection.selected[0]["matches"][0]["unit"], b);
    assert!(
        before
            .selection
            .projections
            .iter()
            .any(|unit| unit.id == b && unit.kind == "summary")
    );
    let after = retrieve().await.unwrap().finish(1, &config.search).unwrap();
    assert_eq!(after.hits[0].unit.id, c);
    assert_eq!(after.selection.incomplete_hidden_hits, None);
    assert_ne!(before.selection.links, after.selection.links);
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn hybrid_evidence_keeps_each_leg_and_pre_rerank_dense_order() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]]})),
        )
        .expect(1)
        .mount(&server)
        .await;
    let (_root, store, config, factory, a, b, c) = fixture(&server);
    complete_sources(&store);
    let mut output = candidates(
        Arc::clone(&store),
        config.clone(),
        request("invoice", fusion::Mode::Hybrid),
        tier(&factory),
        || 100,
    )
    .await
    .unwrap();
    output.hits.reverse();
    let output = {
        let writer = store.writer().unwrap();
        writer.execute("DELETE FROM units", []).unwrap();
        output.finish(1, &config.search).unwrap()
    };
    assert_eq!(output.hits[0].unit.id, b);
    assert_eq!(output.selection.selected[0]["unit"], b);
    assert_eq!(
        output.dense_selection.single_table,
        Some(factory.table_slug())
    );
    assert_eq!(output.dense_selection.single, [b, a]);
    assert!(output.dense_selection.spaces.is_empty());
    assert_eq!(output.selection.leg.ordered, [a]);
    assert_eq!(output.selection.leg.graph, [c]);
    assert_eq!(output.selection_legs[0].publications.len(), 2);
    assert_eq!(output.selection_legs[1].publications.len(), 2);
    for id in [a, b, c] {
        assert!(
            output
                .selection
                .occurrences
                .iter()
                .any(|unit| unit.id == id)
        );
    }
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}
