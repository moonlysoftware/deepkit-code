//! Real lexical candidates, recording HTTP proxy and durable rerank admission.
use super::*;
use crate::{
    store::StoreLimits,
    sync,
    treat::{Router, context},
};
use rusqlite::params;
use serde_json::json;
use std::fs;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

fn setup(server: &MockServer, protected: bool, calls: i64) -> (tempfile::TempDir, Arc<Factory>) {
    setup_count(server, protected, calls, 2)
}

fn setup_count(
    server: &MockServer,
    protected: bool,
    calls: i64,
    count: usize,
) -> (tempfile::TempDir, Arc<Factory>) {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join(".deepkit")).unwrap();
    let text = format!(
        "[index]\nrepo_name='fixture'\n[rerank]\nprovider='remote'\n[rerank.remote]\nurl='{}'\nmodel='fixture'\n[[treatments]]\nname='second'\nmatch=['b.rs']\nmax_level=1\nnever_send={protected}\n",
        server.uri()
    );
    let text = if count > 2 {
        format!("{text}\n[limits]\nquery_timeout_ms=10000\n")
    } else {
        text
    };
    fs::write(root.path().join(".deepkit/config.toml"), text).unwrap();
    let config = config::load(root.path(), None).unwrap().config;
    let store = Arc::new(store::open(":memory:", StoreLimits::default()).unwrap());
    for index in 0..count {
        let id = index as i64 + 1;
        let (path, text) = match index {
            0 => ("a.rs".to_owned(), "needle PUBLIC_CODE".to_owned()),
            1 => (
                "b.rs".to_owned(),
                "needle DISTINCTIVE_PRIVATE_CODE".to_owned(),
            ),
            _ => (
                format!("c{index}.rs"),
                format!("needle PUBLIC_CODE_{index}"),
            ),
        };
        fs::write(root.path().join(&path), &text).unwrap();
        let treatment = Router::new(&config.treatments, config.llm.enabled)
            .unwrap()
            .route(&path, store::FileKind::Code);
        let file_hash = *blake3::hash(text.as_bytes()).as_bytes();
        let fingerprint = sync::l1::file_fingerprint(
            &config,
            "fixture",
            sync::l1::FingerprintFile {
                path: &path,
                content_hash: &file_hash,
                language: Some("rust"),
                kind: store::FileKind::Code,
                capped: false,
            },
            &treatment,
        );
        let hash = context::unit_hash("chunk", None, &text);
        let mut writer = store.writer().unwrap();
        let tx = writer.transaction().unwrap();
        tx.execute("INSERT INTO files(id,path,blake3,size,mtime_ns,lang,kind,max_level,treatment,l1_pending,l1_desired,l1_fingerprint) VALUES(?1,?2,?3,?4,0,'rust','code',?5,?6,0,?7,?7)", params![id,path,file_hash.as_slice(),text.len() as i64,treatment.max_level,treatment.name,fingerprint.as_slice()]).unwrap();
        tx.execute(
            "INSERT INTO unit_contents(hash,text,tokens) VALUES(?1,?2,3)",
            params![hash.as_slice(), text],
        )
        .unwrap();
        tx.execute("INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,created_at,updated_at) VALUES(?1,?2,?2,?3,?1,'chunk','needle',0,0)", params![id,hash.as_slice(),path]).unwrap();
        tx.execute(
            "INSERT INTO unit_sources VALUES(?1,'file',?1,'content')",
            [id],
        )
        .unwrap();
        tx.execute(
            "INSERT INTO fts_units(rowid,header,body) VALUES(?1,'needle',?2)",
            params![id, text],
        )
        .unwrap();
        tx.commit().unwrap();
    }
    let factory = Factory::build(
        store,
        Settings {
            config_root: root.path().into(),
            config,
            limits: usage::Limits {
                budget: usage::Budget {
                    calls: Some(calls),
                    tokens: Some(100_000),
                },
                rpm: Some(10),
                tpm: Some(100_000),
            },
        },
        |config, timeout| Ok(Remote::with_api_key(config, timeout, "fixture-key".into())?),
    )
    .unwrap();
    (root, factory)
}
fn lexical(factory: &Factory, scope: Option<&str>) -> Vec<StoredUnit> {
    super::super::lexical(
        &factory.store,
        "needle",
        10,
        scope,
        &factory.settings.config.search,
    )
    .unwrap()
    .into_iter()
    .map(|hit| hit.unit)
    .collect()
}
fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(5)
}
fn states(factory: &Factory) -> (i64, i64, i64, i64) {
    factory.store.writer().unwrap().query_row(
        "SELECT SUM(state='reserved'),SUM(state='dispatched'),SUM(state='reconciled'),SUM(state='released') FROM usage_reservations", [],
        |row| Ok((row.get::<_,Option<i64>>(0)?.unwrap_or(0),row.get::<_,Option<i64>>(1)?.unwrap_or(0),row.get::<_,Option<i64>>(2)?.unwrap_or(0),row.get::<_,Option<i64>>(3)?.unwrap_or(0))),
    ).unwrap()
}
async fn respond(server: &MockServer, delay: Duration) {
    Mock::given(method("POST")).and(path("/rerank"))
        .respond_with(ResponseTemplate::new(200).set_delay(delay).set_body_json(json!({"results":[{"index":1,"relevance_score":0.2},{"index":0,"relevance_score":0.8}]})))
        .mount(server).await;
}

#[tokio::test]
async fn lexical_l1_candidates_send_canonical_whole_set_once_with_estimated_accounting() {
    let server = MockServer::start().await;
    respond(&server, Duration::ZERO).await;
    let (_root, factory) = setup(&server, false, 1);
    let units = lexical(&factory, None);
    assert_eq!(units.len(), 2);
    let prepared = factory
        .prepare("needle".into(), units.clone(), deadline())
        .await
        .unwrap();
    let expected = prepared
        .inputs
        .iter()
        .map(|input| input.document.clone())
        .collect::<Vec<_>>();
    let tokens = prepared.tokens;
    assert_eq!(
        prepared.score(deadline(), || 100).await.unwrap(),
        [0.8, 0.2]
    );
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let body = requests[0].body_json::<serde_json::Value>().unwrap();
    assert_eq!(body["documents"], json!(expected));
    assert_eq!(body["query"], "needle");
    assert_eq!(body["return_documents"], false);
    assert_eq!(body["top_n"], 2);
    assert_eq!(states(&factory), (0, 0, 1, 0));
    let counts: (i64, i64, i64) = factory
        .store
        .writer()
        .unwrap()
        .query_row("SELECT calls,tokens,estimated_tokens FROM usage", [], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .unwrap();
    assert_eq!(counts, (1, tokens, tokens));
    let endpoint: (String, i64) = factory
        .store
        .writer()
        .unwrap()
        .query_row(
            "SELECT endpoint_kind,sent_units FROM usage_reservations",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(endpoint, ("rerank".into(), 2));
    assert_eq!(lexical(&factory, None), units);
}

#[tokio::test]
async fn protected_lexical_and_mixed_sets_preserve_every_hit_and_send_nothing() {
    let server = MockServer::start().await;
    let (_root, factory) = setup(&server, true, 10);
    for scope in [Some("b.rs"), None] {
        let units = lexical(&factory, scope);
        assert_eq!(units.len(), if scope.is_some() { 1 } else { 2 });
        assert!(matches!(
            factory
                .prepare("needle".into(), units.clone(), deadline())
                .await,
            Err(Error::Privacy)
        ));
        assert_eq!(lexical(&factory, scope), units);
    }
    assert!(server.received_requests().await.unwrap().is_empty());
    assert_eq!(states(&factory), (0, 0, 0, 0));
}

#[tokio::test]
async fn dispatch_rechecks_policy_source_and_transitive_context_for_the_entire_set() {
    let server = MockServer::start().await;
    for mutation in 0..5 {
        let (root, factory) = setup(&server, mutation == 2, 10);
        let units = lexical(&factory, (mutation == 2).then_some("a.rs"));
        let prepared = factory
            .prepare("needle".into(), units.clone(), deadline())
            .await
            .unwrap();
        match mutation {
            0 => {
                let path = root.path().join(".deepkit/config.toml");
                fs::write(
                    &path,
                    fs::read_to_string(&path)
                        .unwrap()
                        .replace("never_send=false", "never_send=true"),
                )
                .unwrap();
            }
            1 => {
                fs::write(root.path().join("b.rs"), "needle changed source").unwrap();
            }
            2 => {
                factory
                    .store
                    .writer()
                    .unwrap()
                    .execute("INSERT INTO unit_sources VALUES(1,'unit',2,'role')", [])
                    .unwrap();
            }
            3 => {
                factory
                    .store
                    .writer()
                    .unwrap()
                    .execute("UPDATE units SET header='changed header' WHERE id=2", [])
                    .unwrap();
            }
            _ => {
                factory
                    .store
                    .writer()
                    .unwrap()
                    .execute("DELETE FROM unit_sources WHERE unit_id=2", [])
                    .unwrap();
            }
        }
        assert!(
            matches!(
                prepared.score(deadline(), || 100).await,
                Err(Error::Privacy)
            ),
            "mutation {mutation}"
        );
        assert_eq!(states(&factory), (0, 0, 0, 1));
        assert_eq!(units.len(), if mutation == 2 { 1 } else { 2 }); // Saved RRF results remain available.
    }
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn limits_empty_sets_and_deadlines_never_replay_or_leak_reservations() {
    let server = MockServer::start().await;
    respond(&server, Duration::from_secs(1)).await;
    let (_root, factory) = setup(&server, false, 0);
    let units = lexical(&factory, None);
    let empty = factory
        .prepare("needle".into(), vec![], deadline())
        .await
        .unwrap();
    assert!(empty.score(deadline(), || 100).await.unwrap().is_empty());
    assert!(matches!(
        factory.prepare(" ".into(), units.clone(), deadline()).await,
        Err(Error::Input)
    ));
    let prepared = factory
        .prepare("needle".into(), units, deadline())
        .await
        .unwrap();
    assert!(matches!(
        prepared.score(deadline(), || 100).await,
        Err(Error::Admission(admission::Error::Budget { .. }))
    ));
    assert_eq!(states(&factory), (0, 0, 0, 0));
    assert!(server.received_requests().await.unwrap().is_empty());
    let (_root, factory) = setup(&server, false, 10);
    let prepared = factory
        .prepare("needle".into(), lexical(&factory, None), deadline())
        .await
        .unwrap();
    assert!(matches!(
        prepared
            .score(Instant::now() + Duration::from_millis(200), || 100)
            .await,
        Err(Error::Deadline)
    ));
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
    assert_eq!(states(&factory), (0, 1, 0, 0));
}

#[tokio::test]
async fn expired_preparation_cannot_extend_its_deadline_or_start_dispatch_source_reads() {
    let server = MockServer::start().await;
    let (root, factory) = setup(&server, false, 10);
    let mut prepared = factory
        .prepare("needle".into(), lexical(&factory, None), deadline())
        .await
        .unwrap();
    // Advance this private preparation's deadline without a wall-clock sleep.
    prepared.deadline = Instant::now();
    fs::write(root.path().join(".deepkit/config.toml"), "invalid TOML [").unwrap();
    let refused = factory
        .store
        .read_snapshot(|tx| admission::Work::limits(&prepared, tx, &factory.provider))
        .unwrap()
        .unwrap();
    assert!(refused.is_none()); // Loading the now-invalid source config would fail.
    assert!(matches!(
        prepared.score(deadline(), || 100).await,
        Err(Error::Deadline)
    ));
    assert_eq!(states(&factory), (0, 0, 0, 0));
    assert!(server.received_requests().await.unwrap().is_empty());
}

fn ranked(entries: &[(i64, Option<&str>)]) -> Vec<super::super::RankedCandidate> {
    let mut units = entries
        .iter()
        .map(|&(id, path)| {
            (
                id,
                StoredUnit {
                    id,
                    hash: id.to_le_bytes().to_vec(),
                    path: path.map(str::to_owned),
                    kind: "chunk".into(),
                    line_start: None,
                    line_end: None,
                    header: format!("unit {id}"),
                    text: "body".into(),
                    ladder_level: 1,
                },
            )
        })
        .collect();
    let fused = entries
        .iter()
        .enumerate()
        .map(|(rank, &(id, _))| super::super::fusion::Hit {
            id,
            score: 1.0 / (61 + rank) as f64,
            scores: super::super::fusion::Scores::default(),
        })
        .collect();
    super::super::materialize(fused, &BTreeSet::new(), &mut units)
}

#[test]
fn score_threshold_and_diversification_preserve_rrf_evidence_without_chaining_promotions() {
    let mut hits = ranked(&[
        (1, Some("a")),
        (2, Some("a")),
        (3, Some("b")),
        (4, Some("c")),
        (5, Some("d")),
    ]);
    let original = hits
        .iter()
        .map(|hit| (hit.unit.id, hit.score))
        .collect::<std::collections::BTreeMap<_, _>>();
    apply_scores(
        &mut hits,
        &[0.9, 0.88, 0.86, 0.82, 0.01],
        &config::Rerank::default(),
        &config::Search::default(),
    )
    .unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.unit.id).collect::<Vec<_>>(),
        [1, 3, 2, 4]
    );
    for hit in &hits {
        assert_eq!(hit.rrf_score, original[&hit.unit.id]);
        assert_eq!(hit.matches[0].rrf_score, original[&hit.unit.id]);
        assert_eq!(hit.score, f64::from(hit.rerank_score.unwrap()));
        assert_eq!(hit.matches[0].rerank_score, hit.rerank_score);
    }
    let mut hits = ranked(&[(1, Some("a")), (2, Some("a")), (3, Some("b")), (4, None)]);
    let rerank = config::Rerank {
        min_score: 0.5,
        ..Default::default()
    };
    let search = config::Search {
        diversity_margin: 0.25,
        ..Default::default()
    };
    apply_scores(&mut hits, &[1.0, 0.75, 0.5, 0.49], &rerank, &search).unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.unit.id).collect::<Vec<_>>(),
        [1, 3, 2]
    );
    let mut hits = ranked(&[(1, None), (2, None), (3, None)]);
    apply_scores(
        &mut hits,
        &[-0.0, 0.0, -0.0],
        &config::Rerank {
            min_score: 0.0,
            ..Default::default()
        },
        &search,
    )
    .unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.unit.id).collect::<Vec<_>>(),
        [1, 2, 3]
    );
}

#[test]
fn malformed_score_sets_and_controls_leave_the_entire_rrf_window_unchanged() {
    for scores in [
        vec![0.9],
        vec![0.9, f32::NAN],
        vec![0.9, 1.1],
        vec![-0.1, 0.9],
    ] {
        let mut hits = ranked(&[(1, Some("a")), (2, Some("b"))]);
        let original = hits.iter().map(|hit| hit.score).collect::<Vec<_>>();
        assert!(
            apply_scores(
                &mut hits,
                &scores,
                &config::Rerank::default(),
                &config::Search::default()
            )
            .is_err()
        );
        assert_eq!(
            hits.iter().map(|hit| hit.score).collect::<Vec<_>>(),
            original
        );
        assert!(hits.iter().all(|hit| hit.rerank_score.is_none()));
    }
    for value in [f64::NAN, -0.1, 1.1] {
        assert!(
            validate_ranking(
                &config::Rerank {
                    min_score: value,
                    ..Default::default()
                },
                &config::Search::default()
            )
            .is_err()
        );
        assert!(
            validate_ranking(
                &config::Rerank::default(),
                &config::Search {
                    diversity_margin: value,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
}

async fn window(factory: &Arc<Factory>) -> super::super::pipeline::Candidates {
    super::super::pipeline::candidates(
        Arc::clone(&factory.store),
        factory.settings.config.clone(),
        super::super::pipeline::Request {
            query: "needle".into(),
            scope: None,
            k: 10,
            mode: super::super::fusion::Mode::Lexical,
            exclude_ids: vec![],
        },
        None,
        || 100,
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn pipeline_applies_scores_before_collapse_and_retains_verbose_rrf() {
    let server = MockServer::start().await;
    Mock::given(method("POST")).and(path("/rerank")).respond_with(ResponseTemplate::new(200).set_body_json(json!({"results":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":0.05}]}))).mount(&server).await;
    let (_root, factory) = setup(&server, false, 10);
    let candidates = window(&factory).await;
    let expected_id = candidates.hits[1].unit.id;
    let expected_rrf = candidates.hits[1].score;
    let candidates = candidates.rerank_with(Arc::clone(&factory), || 100).await;
    // Repeated orchestration cannot charge or rerank a previously handled window.
    let results = candidates
        .rerank_with(Arc::clone(&factory), || 100)
        .await
        .finish(10, &factory.settings.config.search)
        .unwrap();
    assert!(matches!(results.rerank, State::Complete));
    assert!(!results.degraded());
    assert_eq!(results.hits.len(), 1);
    assert_eq!(results.hits[0].unit.id, expected_id);
    let verbose =
        super::super::format_response_verbose(&factory.store, "needle", &results.hits, true)
            .unwrap();
    assert_eq!(verbose["hits"][0]["rrf_score"], json!(expected_rrf));
    assert_eq!(verbose["hits"][0]["rerank_score"], json!(0.9_f32));
    assert!(
        verbose["hits"][0]["why"]
            .as_array()
            .unwrap()
            .contains(&json!("reranked"))
    );
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn pipeline_privacy_timeout_and_provider_failure_keep_rrf_order_with_visible_reasons() {
    for (protected, delay, body, reason) in [
        (true, Duration::ZERO, json!({}), "rerank_privacy"),
        (
            false,
            Duration::from_secs(3),
            json!({"results":[]}),
            "rerank_timeout",
        ),
        (false, Duration::ZERO, json!({"results":[]}), "rerank_error"),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/rerank"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(delay)
                    .set_body_json(body),
            )
            .mount(&server)
            .await;
        let (_root, factory) = setup(&server, protected, 10);
        let candidates = window(&factory).await;
        let original = candidates
            .hits
            .iter()
            .map(|hit| (hit.unit.id, hit.score))
            .collect::<Vec<_>>();
        let results = candidates
            .rerank_with(Arc::clone(&factory), || 100)
            .await
            .finish(10, &factory.settings.config.search)
            .unwrap();
        assert!(results.degraded());
        assert_eq!(results.rerank.fallback(), Some(reason));
        assert_eq!(
            results
                .hits
                .iter()
                .map(|hit| (hit.unit.id, hit.score))
                .collect::<Vec<_>>(),
            original
        );
        let compact =
            super::super::format_response(&factory.store, "needle", &results.hits).unwrap();
        for hit in compact["hits"].as_array().unwrap() {
            assert!(hit["why"].as_array().unwrap().contains(&json!(reason)));
        }
        assert_eq!(
            server.received_requests().await.unwrap().len(),
            usize::from(!protected)
        );
    }
}

#[tokio::test]
async fn pipeline_rejects_a_foreign_factory_and_preserves_the_retrieval_deadline() {
    let server = MockServer::start().await;
    let (_first_root, first) = setup(&server, false, 10);
    let (_second_root, second) = setup(&server, false, 10);
    let results = window(&first)
        .await
        .rerank_with(Arc::clone(&second), || 100)
        .await
        .finish(10, &first.settings.config.search)
        .unwrap();
    assert_eq!(results.hits.len(), 2);
    assert_eq!(results.rerank.fallback(), Some("rerank_error"));
    assert_eq!(states(&first), (0, 0, 0, 0));
    assert_eq!(states(&second), (0, 0, 0, 0));

    let (root, mut factory) = setup(&server, false, 10);
    let path = root.path().join(".deepkit/config.toml");
    fs::write(
        &path,
        format!(
            "{}\n[limits]\nquery_timeout_ms=200\n",
            fs::read_to_string(&path).unwrap()
        ),
    )
    .unwrap();
    Arc::get_mut(&mut factory).unwrap().settings.config =
        config::load(root.path(), None).unwrap().config;
    let candidates = window(&factory).await;
    assert_eq!(candidates.hits.len(), 2);
    tokio::time::sleep(Duration::from_millis(250)).await;
    let results = candidates
        .rerank_with(Arc::clone(&factory), || 100)
        .await
        .finish(10, &factory.settings.config.search)
        .unwrap();
    assert_eq!(results.hits.len(), 2);
    assert_eq!(results.rerank.fallback(), Some("rerank_timeout"));
    assert_eq!(states(&factory), (0, 0, 0, 0));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[test]
fn profile_selection_is_explicit_and_does_not_follow_the_optional_docs_tier() {
    let mut config = Config::default();
    for (profile, expected) in [
        ("remote", Selection::Remote),
        ("local", Selection::Local),
        ("ollama", Selection::Local),
    ] {
        config.embedder.profile = profile.into();
        config.embedder.docs = Some(config::DocsEmbedder {
            profile: if profile == "remote" {
                "local"
            } else {
                "remote"
            }
            .into(),
            ..Default::default()
        });
        config.rerank.provider = "auto".into();
        assert_eq!(selection(&config).unwrap(), expected);
        config.rerank.provider = "local".into();
        assert_eq!(selection(&config).unwrap(), Selection::Local);
        config.rerank.provider = "remote".into();
        assert_eq!(selection(&config).unwrap(), Selection::Remote);
    }
    config.rerank.mode = "off".into();
    assert!(selection(&config).is_err());
    config.rerank.mode = "on".into();
    config.rerank.provider = "unknown".into();
    assert!(selection(&config).is_err());
}

#[tokio::test]
async fn local_preparation_refuses_missing_weights_without_reading_remote_settings_or_keys() {
    let root = tempfile::tempdir().unwrap();
    let store = Arc::new(store::open(":memory:", StoreLimits::default()).unwrap());
    for profile in ["local", "ollama"] {
        let mut config = Config::default();
        config.embedder.profile = profile.into();
        config.rerank.remote.url = "invalid unused URL".into();
        config.rerank.remote.api_key_env = "DKC_UNUSED_LOCAL_RERANK_KEY".into();
        let result = Factory::new_local(
            Arc::clone(&store),
            Settings {
                config_root: root.path().into(),
                config,
                limits: usage::Limits {
                    budget: usage::Budget {
                        calls: Some(10),
                        tokens: Some(10000),
                    },
                    rpm: None,
                    tpm: None,
                },
            },
            root.path().join("absent-cache"),
        )
        .await;
        assert!(matches!(
            result,
            Err(Error::Model(models::Error::Local(
                models::llama::Error::Artifact
            )))
        ));
    }
    assert_eq!(
        store
            .writer()
            .unwrap()
            .query_row("SELECT count(*) FROM usage_reservations", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[tokio::test]
#[ignore = "requires DKC_RERANKER_CACHE verified weights; explicit CPU host exercise"]
async fn verified_local_host_scores_through_admission_and_pipeline_without_outbound_counts() {
    let cache = std::env::var_os("DKC_RERANKER_CACHE").expect("set DKC_RERANKER_CACHE");
    let server = MockServer::start().await;
    let (root, remote) = setup(&server, false, 10);
    let path = root.path().join(".deepkit/config.toml");
    let text = fs::read_to_string(&path)
        .unwrap()
        .replace("provider='remote'", "provider='local'");
    fs::write(&path,format!("{text}\n[rerank.local]\ngpu_layers=0\n[limits]\nquery_timeout_ms=60000\nrerank_timeout_ms=60000\n")).unwrap();
    let config = config::load(root.path(), None).unwrap().config;
    let factory = Factory::new_local(
        Arc::clone(&remote.store),
        Settings {
            config_root: root.path().into(),
            config,
            limits: remote.settings.limits,
        },
        cache.into(),
    )
    .await
    .unwrap();
    assert_eq!(factory.provider, "rerank:local");
    let prepared = factory
        .prepare(
            "needle".into(),
            lexical(&factory, None),
            Instant::now() + Duration::from_secs(60),
        )
        .await
        .unwrap();
    assert!(admission::Work::endpoint(&prepared).is_none());
    let scores = prepared
        .score(Instant::now() + Duration::from_secs(60), || 100)
        .await
        .unwrap();
    assert_eq!(scores.len(), 2);
    assert!(
        scores
            .iter()
            .all(|score| score.is_finite() && (0.0..=1.0).contains(score))
    );
    let results = window(&factory)
        .await
        .rerank_with(Arc::clone(&factory), || 100)
        .await
        .finish(10, &factory.settings.config.search)
        .unwrap();
    assert!(matches!(results.rerank, State::Complete));
    assert!(!results.degraded());
    assert!(results.hits.iter().all(|hit| hit.rerank_score.is_some()));
    assert_eq!(states(&factory), (0, 0, 2, 0));
    let inspected = queue::inspection::read(&factory.store).unwrap();
    assert!(inspected.endpoints.is_empty());
    assert_eq!(inspected.unknown_endpoint_dispatches, 0);
    assert!(server.received_requests().await.unwrap().is_empty());
    // Policy is rechecked for native work too: never_send remains a ceiling of L1.
    let prepared = factory
        .prepare(
            "needle".into(),
            lexical(&factory, None),
            Instant::now() + Duration::from_secs(60),
        )
        .await
        .unwrap();
    fs::write(
        &path,
        fs::read_to_string(&path)
            .unwrap()
            .replace("never_send=false", "never_send=true"),
    )
    .unwrap();
    assert!(matches!(
        prepared
            .score(Instant::now() + Duration::from_secs(60), || 100)
            .await,
        Err(Error::Privacy)
    ));
    assert_eq!(states(&factory), (0, 0, 2, 1));
    println!(
        "verified CPU local host scores={scores:?}; two reconciled calls; no outbound endpoint; changed policy refused before native dispatch"
    );
}

#[tokio::test]
async fn configured_context_omissions_reach_the_remote_pipeline_without_extra_blank_lines() {
    let server = MockServer::start().await;
    respond(&server, Duration::ZERO).await;
    let (root, original) = setup(&server, false, 10);
    let path = root.path().join(".deepkit/config.toml");
    fs::write(
        &path,
        format!(
            "{}\n[embedder]\nbreadcrumb=false\nkind_line=false\n",
            fs::read_to_string(&path).unwrap()
        ),
    )
    .unwrap();
    let config = config::load(root.path(), None).unwrap().config;
    let factory = Factory::build(
        Arc::clone(&original.store),
        Settings {
            config_root: root.path().into(),
            config,
            limits: original.settings.limits,
        },
        |config, timeout| Ok(Remote::with_api_key(config, timeout, "fixture-key".into())?),
    )
    .unwrap();
    let result = window(&factory)
        .await
        .rerank_with(Arc::clone(&factory), || 100)
        .await
        .finish(10, &factory.settings.config.search)
        .unwrap();
    assert!(matches!(result.rerank, State::Complete));
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].body_json::<serde_json::Value>().unwrap()["documents"],
        json!([
            "needle\nneedle PUBLIC_CODE",
            "needle\nneedle DISTINCTIVE_PRIVATE_CODE"
        ])
    );
}

async fn adaptive_window(factory: &Arc<Factory>) -> super::super::pipeline::Candidates {
    super::super::pipeline::candidates_for_rerank(
        Arc::clone(&factory.store),
        factory.settings.config.clone(),
        super::super::pipeline::Request {
            query: "needle".into(),
            scope: None,
            k: 10,
            mode: super::super::fusion::Mode::Lexical,
            exclude_ids: vec![],
        },
        None,
        factory,
        || 100,
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn slow_full_call_reduces_future_selection_without_truncating_inflight_or_unsafe_sets() {
    let server = MockServer::start().await;
    Mock::given(method("POST")).and(path("/rerank")).respond_with(|request: &wiremock::Request| {
        let count = request.body_json::<serde_json::Value>().unwrap()["documents"].as_array().unwrap().len();
        ResponseTemplate::new(200).set_delay(if count==50 {Duration::from_millis(1700)} else {Duration::ZERO})
            .set_body_json(json!({"results":(0..count).map(|index|json!({"index":index,"relevance_score":0.8})).collect::<Vec<_>>()}))
    }).mount(&server).await;
    for (protected, calls) in [(true, 10), (false, 0)] {
        let (_root, factory) = setup_count(&server, protected, calls, 50);
        let result = adaptive_window(&factory)
            .await
            .rerank_with(Arc::clone(&factory), || 100)
            .await;
        assert_eq!(result.hits.len(), 50);
        assert!(!factory.auto_status().reduced);
        assert!(server.received_requests().await.unwrap().is_empty());
    }
    let (_root, factory) = setup_count(&server, false, 10, 50);
    let inflight = adaptive_window(&factory).await;
    assert_eq!(inflight.hits.len(), 50);
    let first = adaptive_window(&factory)
        .await
        .rerank_with(Arc::clone(&factory), || 100)
        .await;
    assert_eq!(first.rerank.fallback(), Some("rerank_timeout"));
    assert_eq!(first.hits.len(), 50);
    assert_eq!(factory.auto_status().effective_candidates, 20);
    let prepared = Arc::clone(&factory);
    let lazy = lazy::Host::with_work(
        Some(Box::pin(async move { Ok(prepared) })),
        Arc::clone(&factory.store),
        factory.settings.config.clone(),
    );
    assert!(matches!(
        lazy.get(Instant::now() + Duration::from_secs(1)).await,
        lazy::Outcome::Ready(_)
    ));
    assert_eq!(lazy_window(&factory, &lazy).await.hits.len(), 20);

    // A new retained host restores the saved bound before starting preparation.
    let until = Instant::now() + Duration::from_secs(3);
    while auto::read_status(&factory.store, &factory.settings.config)
        .unwrap()
        .persistence
        != "saved"
    {
        assert!(Instant::now() < until);
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let idle = lazy::Host::with_work(
        Some(Box::pin(async { panic!("selection started preparation") })),
        Arc::clone(&factory.store),
        factory.settings.config.clone(),
    );
    let selected = lazy_window(&factory, &idle).await;
    assert_eq!(selected.hits.len(), 20);
    assert_eq!(selected.rerank_auto.unwrap().persistence, "saved");
    let routed = super::super::pipeline::candidates_routed_for_lazy(
        super::super::dense::routed::Providers {
            store: Arc::clone(&factory.store),
            config: factory.settings.config.clone(),
            config_root: factory.settings.config_root.clone(),
            factories: vec![],
        },
        super::super::pipeline::Request {
            query: "needle".into(),
            scope: None,
            k: 10,
            mode: super::super::fusion::Mode::Lexical,
            exclude_ids: vec![],
        },
        &idle,
        || 100,
    )
    .await
    .unwrap();
    assert_eq!(routed.hits.len(), 20);
    assert_eq!(routed.rerank_auto.unwrap().effective_candidates, 20);
    assert!(matches!(idle.peek(), lazy::Outcome::Idle));

    // An already selected full window is never trimmed to get an authorized subset.
    let old = inflight.rerank_with(Arc::clone(&factory), || 100).await;
    assert_eq!(old.hits.len(), 50);
    let next = super::super::pipeline::candidates_routed_for_rerank(
        super::super::dense::routed::Providers {
            store: Arc::clone(&factory.store),
            config: factory.settings.config.clone(),
            config_root: factory.settings.config_root.clone(),
            factories: vec![],
        },
        super::super::pipeline::Request {
            query: "needle".into(),
            scope: None,
            k: 10,
            mode: super::super::fusion::Mode::Lexical,
            exclude_ids: vec![],
        },
        &factory,
        || 100,
    )
    .await
    .unwrap();
    assert_eq!(next.hits.len(), 20);
    let next = next.rerank_with(Arc::clone(&factory), || 100).await;
    assert!(matches!(next.rerank, State::Complete));
    let finished = next.finish(10, &factory.settings.config.search).unwrap();
    assert_eq!(finished.rerank_auto.unwrap().effective_candidates, 20);
    let requests = server.received_requests().await.unwrap();
    assert_eq!(
        requests
            .iter()
            .map(
                |request| request.body_json::<serde_json::Value>().unwrap()["documents"]
                    .as_array()
                    .unwrap()
                    .len()
            )
            .collect::<Vec<_>>(),
        [50, 50, 20]
    );
    let saved = auto::read_status(&factory.store, &factory.settings.config).unwrap();
    assert_eq!(saved.effective_candidates, 20);
    factory
        .store
        .writer()
        .unwrap()
        .execute("DELETE FROM unit_sources", [])
        .unwrap();
    let unsafe_set = adaptive_window(&factory)
        .await
        .rerank_with(Arc::clone(&factory), || 100)
        .await;
    assert_eq!(unsafe_set.hits.len(), 20);
    assert_eq!(unsafe_set.rerank.fallback(), Some("rerank_privacy"));
    assert_eq!(server.received_requests().await.unwrap().len(), 3);
}

async fn lazy_window(
    factory: &Arc<Factory>,
    host: &lazy::Host,
) -> super::super::pipeline::Candidates {
    super::super::pipeline::candidates_for_lazy(
        Arc::clone(&factory.store),
        factory.settings.config.clone(),
        super::super::pipeline::Request {
            query: "needle".into(),
            scope: None,
            k: 10,
            mode: super::super::fusion::Mode::Lexical,
            exclude_ids: vec![],
        },
        None,
        host,
        || 100,
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn lazy_pipeline_keeps_rrf_on_timeout_then_reuses_completed_preparation() {
    let server = MockServer::start().await;
    Mock::given(method("POST")).and(path("/rerank")).respond_with(ResponseTemplate::new(200)
        .set_body_json(json!({"results":[{"index":0,"relevance_score":0.8},{"index":1,"relevance_score":0.9}]})))
        .mount(&server).await;
    let (_root, factory) = setup(&server, false, 10);
    let prepared = Arc::clone(&factory);
    let (release, wait) = tokio::sync::oneshot::channel();
    let host = lazy::Host::with_work(
        Some(Box::pin(async move {
            wait.await.map_err(|_| Error::Task)?;
            Ok(prepared)
        })),
        Arc::clone(&factory.store),
        factory.settings.config.clone(),
    );
    let selected = lazy_window(&factory, &host).await;
    let original: Vec<_> = selected.hits.iter().map(|h| (h.unit.id, h.score)).collect();
    let result = selected.rerank_lazy(&host, || 100).await;
    assert!(matches!(result.rerank, State::Preparing));
    assert_eq!(
        result
            .hits
            .iter()
            .map(|h| (h.unit.id, h.score))
            .collect::<Vec<_>>(),
        original
    );
    assert!(
        result
            .hits
            .iter()
            .all(|h| h.rerank_fallback == Some("rerank_timeout"))
    );
    assert!(server.received_requests().await.unwrap().is_empty());
    release.send(()).unwrap();
    assert!(matches!(
        host.get(Instant::now() + Duration::from_secs(1)).await,
        lazy::Outcome::Ready(_)
    ));
    // The old timed-out window is not charged even once preparation has completed.
    let old = result
        .rerank_lazy(&host, || 100)
        .await
        .finish(10, &factory.settings.config.search)
        .unwrap();
    assert!(old.degraded());
    let next = lazy_window(&factory, &host)
        .await
        .rerank_lazy(&host, || 100)
        .await;
    assert!(matches!(next.rerank, State::Complete));
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn lazy_pipeline_failure_is_visible_and_foreign_binding_never_starts_work() {
    let server = MockServer::start().await;
    let (_root, factory) = setup(&server, false, 10);
    let failed = lazy::Host::with_work(
        Some(Box::pin(async { Err(Error::Input) })),
        Arc::clone(&factory.store),
        factory.settings.config.clone(),
    );
    let result = lazy_window(&factory, &failed)
        .await
        .rerank_lazy(&failed, || 100)
        .await;
    assert!(
        matches!(&result.rerank,State::PreparationFailed(error) if matches!(**error,Error::Input))
    );
    assert_eq!(result.hits.len(), 2);
    assert!(
        result
            .hits
            .iter()
            .all(|h| h.rerank_fallback == Some("rerank_error"))
    );
    assert!(
        result
            .finish(10, &factory.settings.config.search)
            .unwrap()
            .degraded()
    );
    let mut other = factory.settings.config.clone();
    other.rerank.max_doc_chars += 1;
    let foreign = lazy::Host::with_work(
        Some(Box::pin(async { panic!("foreign host started") })),
        Arc::clone(&factory.store),
        other,
    );
    let result = window(&factory).await.rerank_lazy(&foreign, || 100).await;
    assert!(matches!(result.rerank, State::Failed(_)));
    assert!(matches!(foreign.peek(), lazy::Outcome::Idle));
    let refused = super::super::pipeline::candidates_for_lazy(
        Arc::clone(&factory.store),
        factory.settings.config.clone(),
        super::super::pipeline::Request {
            query: "needle".into(),
            scope: None,
            k: 10,
            mode: super::super::fusion::Mode::Lexical,
            exclude_ids: vec![],
        },
        None,
        &foreign,
        || 100,
    )
    .await;
    assert!(refused.is_err());
    assert!(matches!(foreign.peek(), lazy::Outcome::Idle));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn lazy_preparation_privacy_failure_survives_empty_results() {
    let server = MockServer::start().await;
    let (_root, factory) = setup(&server, false, 10);
    let host = lazy::Host::with_work(
        Some(Box::pin(async { Err(Error::Privacy) })),
        Arc::clone(&factory.store),
        factory.settings.config.clone(),
    );
    let results = super::super::pipeline::candidates_for_lazy(
        Arc::clone(&factory.store),
        factory.settings.config.clone(),
        super::super::pipeline::Request {
            query: "absentword".into(),
            scope: None,
            k: 10,
            mode: super::super::fusion::Mode::Lexical,
            exclude_ids: vec![],
        },
        None,
        &host,
        || 100,
    )
    .await
    .unwrap()
    .rerank_lazy(&host, || 100)
    .await
    .finish(10, &factory.settings.config.search)
    .unwrap();
    assert!(results.hits.is_empty());
    assert!(matches!(results.rerank, State::Privacy));
    assert!(results.degraded());
    assert_eq!(results.rerank.fallback(), Some("rerank_privacy"));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn lazy_adaptive_read_failure_preserves_rrf_and_prevents_later_dispatch() {
    let server = MockServer::start().await;
    let (_root, factory) = setup(&server, false, 10);
    let baseline = window(&factory).await;
    let expected: Vec<_> = baseline
        .hits
        .iter()
        .map(|hit| (hit.unit.id, hit.score))
        .collect();
    let prepared = Arc::clone(&factory);
    let host = lazy::Host::with_work(
        Some(Box::pin(async move { Ok(prepared) })),
        Arc::clone(&factory.store),
        factory.settings.config.clone(),
    );
    factory
        .store
        .writer()
        .unwrap()
        .execute_batch("ALTER TABLE meta RENAME TO saved_meta")
        .unwrap();
    let selected = lazy_window(&factory, &host).await;
    factory
        .store
        .writer()
        .unwrap()
        .execute_batch("ALTER TABLE saved_meta RENAME TO meta")
        .unwrap();
    assert_eq!(selected.rerank.fallback(), Some("rerank_error"));
    assert_eq!(
        selected
            .hits
            .iter()
            .map(|hit| (hit.unit.id, hit.score))
            .collect::<Vec<_>>(),
        expected
    );
    assert!(
        selected
            .hits
            .iter()
            .all(|hit| hit.rerank_fallback == Some("rerank_error"))
    );
    assert!(matches!(host.peek(), lazy::Outcome::Idle));
    assert!(matches!(
        host.get(deadline()).await,
        lazy::Outcome::Ready(_)
    ));
    let finished = selected
        .rerank_lazy(&host, || 100)
        .await
        .finish(10, &factory.settings.config.search)
        .unwrap();
    assert!(finished.degraded());
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn routed_lazy_refuses_foreign_store_and_configuration_before_preparation() {
    let server = MockServer::start().await;
    let (_root, factory) = setup(&server, false, 10);
    for foreign_store in [false, true] {
        let mut config = factory.settings.config.clone();
        let store = if foreign_store {
            Arc::new(store::open(":memory:", StoreLimits::default()).unwrap())
        } else {
            config.rerank.max_doc_chars += 1;
            Arc::clone(&factory.store)
        };
        let host = lazy::Host::with_work(
            Some(Box::pin(async { panic!("foreign host preparation") })),
            store,
            config,
        );
        let result = super::super::pipeline::candidates_routed_for_lazy(
            super::super::dense::routed::Providers {
                store: Arc::clone(&factory.store),
                config: factory.settings.config.clone(),
                config_root: factory.settings.config_root.clone(),
                factories: vec![],
            },
            super::super::pipeline::Request {
                query: "needle".into(),
                scope: None,
                k: 10,
                mode: super::super::fusion::Mode::Lexical,
                exclude_ids: vec![],
            },
            &host,
            || 100,
        )
        .await;
        assert!(matches!(
            result,
            Err(super::super::Error::InvalidQuery { .. })
        ));
        assert!(matches!(host.peek(), lazy::Outcome::Idle));
    }
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn inherited_cancellation_during_rerank_keeps_original_rrf_order() {
    use super::super::{control::Owner, fusion::Mode, pipeline};
    let server = MockServer::start().await;
    let (_root, factory) = setup(&server, false, 10);
    let owner = Owner::new(std::time::Instant::now(), 60_000).unwrap();
    let candidates = pipeline::candidates_with_control(
        Arc::clone(&factory.store),
        factory.settings.config.clone(),
        pipeline::Request {
            query: "needle".into(),
            scope: None,
            k: 10,
            mode: Mode::Lexical,
            exclude_ids: vec![],
        },
        None,
        None,
        owner.control(),
        || 100,
    )
    .await
    .unwrap();
    let original: Vec<_> = candidates
        .hits
        .iter()
        .map(|hit| (hit.unit.id, hit.rrf_score))
        .collect();
    Mock::given(method("POST")).and(path("/rerank"))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(5))
            .set_body_json(json!({"results":[{"index":1,"relevance_score":0.99},{"index":0,"relevance_score":0.1}]})))
        .expect(1).mount(&server).await;
    let task = tokio::spawn(candidates.rerank_with(Arc::clone(&factory), || 100));
    tokio::time::timeout(Duration::from_secs(1), async {
        while server.received_requests().await.unwrap().is_empty() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    drop(owner);
    let output = tokio::time::timeout(Duration::from_secs(1), task)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(output.rerank, State::Timeout));
    assert_eq!(
        output
            .hits
            .iter()
            .map(|hit| (hit.unit.id, hit.rrf_score))
            .collect::<Vec<_>>(),
        original
    );
    assert!(output.hits.iter().all(|hit| hit.rerank_score.is_none()));
}

#[tokio::test]
async fn checked_ledger_replacement_after_preparation_refuses_dispatch() {
    use crate::search::admission::Work;
    let server = MockServer::start().await;
    let (root, original) = setup(&server, false, 5);
    let database = root.path().join(".deepkit/code.db");
    // Copy the existing authoritative fixture without synthesizing weaker source proofs.
    original
        .store
        .writer()
        .unwrap()
        .execute("VACUUM INTO ?1", [database.to_str().unwrap()])
        .unwrap();
    let checked = Arc::new(
        store::open_existing_for_usage(root.path(), &database, StoreLimits::default()).unwrap(),
    );
    let factory = Factory::build(
        Arc::clone(&checked),
        original.settings.clone(),
        |config, timeout| Ok(Remote::with_api_key(config, timeout, "fixture-key".into())?),
    )
    .unwrap();
    let prepared = factory
        .prepare("needle".into(), lexical(&factory, None), deadline())
        .await
        .unwrap();
    checked
        .read_snapshot(|tx| {
            assert!(prepared.limits(tx, &factory.provider).unwrap().is_some());
        })
        .unwrap();
    let retired = root.path().join(".deepkit/retired");
    fs::create_dir(&retired).unwrap();
    for name in ["code.db", "code.db-wal", "code.db-shm"] {
        let path = root.path().join(".deepkit").join(name);
        if path.exists() {
            fs::rename(path, retired.join(name)).unwrap();
        }
    }
    // A valid replacement must not become authority for the retained old handle.
    let replacement = store::open(&database, StoreLimits::default()).unwrap();
    checked
        .read_snapshot(|tx| {
            assert!(prepared.limits(tx, &factory.provider).unwrap().is_none());
        })
        .unwrap();
    assert!(prepared.score(deadline(), || 100).await.is_err());
    assert!(server.received_requests().await.unwrap().is_empty());
    replacement
        .read_snapshot(|tx| {
            let count: i64 = tx
                .query_row("SELECT count(*) FROM usage_reservations", [], |r| r.get(0))
                .unwrap();
            assert_eq!(count, 0);
        })
        .unwrap();
    // The original connections still carry the old WAL/SHM pathname. After
    // replacement, reading those connections is not a valid retired-ledger audit.
    // Release every original file-backed owner, then reopen the complete retired
    // database and its moved sidecars under their actual current pathname.
    drop(factory);
    assert_eq!(Arc::strong_count(&checked), 1);
    drop(checked);
    let retired_ledger = rusqlite::Connection::open_with_flags(
        fs::canonicalize(&retired).unwrap().join("code.db"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NOFOLLOW,
    )
    .unwrap();
    let count: i64 = retired_ledger
        .query_row(
            "SELECT count(*) FROM usage_reservations WHERE state IN ('dispatched','reconciled')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 0);
}
