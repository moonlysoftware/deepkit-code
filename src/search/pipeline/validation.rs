//! Read-only retained selection validation across async model work; v1 §9.
use super::{Error, LexicalState, Request, Results, collapse, lexical_leg_checked, query};
use crate::{config::Search, store::Store};
use std::collections::BTreeSet;

impl Results {
    /// Revalidates sampled ranking's local inputs without repeating model calls.
    /// Returned files keep their independent source fence; vector progress does
    /// not invalidate L1 identity or replace the retained dense ranking.
    pub(crate) fn selection_is_current(
        &self,
        store: &Store,
        request: &Request,
        settings: &Search,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<bool, Error> {
        check()?;
        if matches!(self.lexical, LexicalState::Complete) {
            let parsed = query::parse(&request.query, request.mode)?;
            let current = lexical_leg_checked(
                store,
                &parsed,
                request.k,
                request.scope.as_ref(),
                settings,
                &request.exclude_ids,
                Some(check),
            )?;
            check()?;
            if current != self.selection.leg {
                return Ok(false);
            }
        }
        let excluded = request.exclude_ids.iter().copied().collect::<BTreeSet<_>>();
        for leg in &self.selection_legs {
            check()?;
            // Include IDs with missing/excluded projections, not just materialized
            // rows, so a newly available parent cannot silently change selection.
            let ids = leg.links.keys().copied().collect::<Vec<_>>();
            let mut current = collapse::prepare_with_work(store, &ids, &excluded, Some(check))?;
            if current.occurrence_evidence(check)? != leg.occurrences || current.links != leg.links
            {
                return Ok(false);
            }
            // Use the original displayed set even when the service subsequently
            // suppresses a hit. Displayed source changes must not discard other
            // verified files, while every originally hidden publication matters.
            let mut original = collapse::Prepared::default();
            original.links.clone_from(&leg.links);
            original.publications.clone_from(&leg.publications);
            original.retain_hidden_publications(&self.hits, check)?;
            current.retain_hidden_publications(&self.hits, check)?;
            check()?;
            if current.publications != original.publications {
                return Ok(false);
            }
        }
        check()?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{fixture, request, tier};
    use super::super::*;
    use crate::search::tests::add;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    #[tokio::test]
    async fn retained_hybrid_order_survives_l2_progress_without_another_dispatch() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]]})),
            )
            .expect(1)
            .mount(&server)
            .await;
        let (_root, store, config, factory, a, b, _c) = fixture(&server);
        let input = || request("invoice", fusion::Mode::Hybrid);
        let mut selected = candidates(
            Arc::clone(&store),
            config.clone(),
            input(),
            tier(&factory),
            || 100,
        )
        .await
        .unwrap();
        // Model ordering is retained evidence; validation must not reproduce it
        // with a lexical-only query or silently restore RRF order.
        selected.hits.reverse();
        let selected = selected.finish(1, &config.search).unwrap();
        assert_eq!(selected.hits[0].unit.id, b);
        assert_eq!(selected.dense_selection.single, [b, a]);
        assert!(
            selected
                .selection_is_current(&store, &input(), &config.search, &|| Ok(()))
                .unwrap()
        );
        store.delete_vector(&factory.table_slug(), b).unwrap();
        store
            .writer()
            .unwrap()
            .execute("UPDATE units SET ladder_level=2", [])
            .unwrap();
        assert!(
            selected
                .selection_is_current(&store, &input(), &config.search, &|| Ok(()))
                .unwrap()
        );
        assert_eq!(selected.hits[0].unit.id, b);
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
        store
            .writer()
            .unwrap()
            .execute("UPDATE units SET header='changed' WHERE id=?1", [a])
            .unwrap();
        assert!(
            !selected
                .selection_is_current(&store, &input(), &config.search, &|| Ok(()))
                .unwrap()
        );
    }

    #[tokio::test]
    async fn publications_keep_displayed_source_fences_and_independent_leg_authority() {
        let server = MockServer::start().await;
        let (_root, store, config, _factory, a, _b, c) = fixture(&server);
        let input = || request("invoice", fusion::Mode::Lexical);
        let retrieve =
            || search_without_rerank(Arc::clone(&store), config.clone(), input(), None, || 100);
        let earlier = retrieve().await.unwrap();
        assert_eq!(earlier.hits[0].unit.id, a);
        store
            .writer()
            .unwrap()
            .execute(
                "UPDATE files SET l1_desired=blake3,l1_fingerprint=blake3,l1_pending=0 WHERE path='src/a.rs'",
                [],
            )
            .unwrap();
        // Displayed publications are verified per file by the service.
        assert!(
            earlier
                .selection_is_current(&store, &input(), &config.search, &|| Ok(()))
                .unwrap()
        );
        store
            .writer()
            .unwrap()
            .execute(
                "UPDATE files SET l1_desired=blake3,l1_fingerprint=blake3,l1_pending=0 WHERE path='src/c.rs'",
                [],
            )
            .unwrap();
        assert!(
            !earlier
                .selection_is_current(&store, &input(), &config.search, &|| Ok(()))
                .unwrap()
        );
        let mut later = retrieve().await.unwrap();
        assert!(later.selection_legs[0].links.contains_key(&c));
        assert!(
            later
                .selection_is_current(&store, &input(), &config.search, &|| Ok(()))
                .unwrap()
        );
        // Model the lexical leg observing an earlier publication than a later
        // hydration. Current merged Selection must not replace that evidence.
        later.selection_legs[0] = earlier.selection_legs.into_iter().next().unwrap();
        assert!(
            !later
                .selection_is_current(&store, &input(), &config.search, &|| Ok(()))
                .unwrap()
        );
    }

    #[tokio::test]
    async fn empty_lexical_selection_rechecks_new_winners_scope_and_exclusions() {
        let server = MockServer::start().await;
        let (_root, store, config, _factory, _a, b, _c) = fixture(&server);
        let excluded = add(&store, "src/excluded.rs", "excluded", "brandnew", None).unwrap();
        let input = || {
            let mut input = request("brandnew", fusion::Mode::Lexical);
            input.scope = Some(crate::store::SearchScope::Globs(vec!["src/*.rs".into()]));
            input.exclude_ids.push(b);
            input.exclude_ids.push(excluded);
            input
        };
        let selected =
            search_without_rerank(Arc::clone(&store), config.clone(), input(), None, || 100)
                .await
                .unwrap();
        assert!(selected.hits.is_empty());
        add(&store, "docs/new.md", "outside", "brandnew", None).unwrap();
        assert!(
            selected
                .selection_is_current(&store, &input(), &config.search, &|| Ok(()))
                .unwrap()
        );
        add(&store, "src/new.rs", "inside", "brandnew", None).unwrap();
        assert!(
            !selected
                .selection_is_current(&store, &input(), &config.search, &|| Ok(()))
                .unwrap()
        );
        assert!(matches!(
            selected.selection_is_current(&store, &input(), &config.search, &|| {
                Err(Error::Deadline)
            }),
            Err(Error::Deadline)
        ));
        assert!(server.received_requests().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn retained_summary_projection_refuses_changed_parent() {
        let server = MockServer::start().await;
        let (_root, store, config, _factory, a, b, c) = fixture(&server);
        store
            .writer()
            .unwrap()
            .execute(
                "UPDATE units SET kind='summary',parent_unit=?1 WHERE id=?2",
                [a, b],
            )
            .unwrap();
        let input = || request("network", fusion::Mode::Lexical);
        let selected =
            search_without_rerank(Arc::clone(&store), config.clone(), input(), None, || 100)
                .await
                .unwrap();
        assert_eq!(selected.hits[0].unit.id, a);
        assert!(
            selected
                .selection_is_current(&store, &input(), &config.search, &|| Ok(()))
                .unwrap()
        );
        store
            .writer()
            .unwrap()
            .execute("UPDATE units SET parent_unit=?1 WHERE id=?2", [c, b])
            .unwrap();
        assert!(
            !selected
                .selection_is_current(&store, &input(), &config.search, &|| Ok(()))
                .unwrap()
        );
    }
}
