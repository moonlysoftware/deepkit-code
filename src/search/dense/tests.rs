//! Real query-to-FLOAT16 retrieval and transition fences; v1 §9 step 4.
use super::*;
use crate::{
    config,
    queue::usage,
    search::embedding_query,
    store::{
        StoreLimits,
        vectors::{VectorInsert, VectorTableSpec},
    },
};
use std::{fs, time::Duration};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

fn setup(server: &MockServer, units: i64) -> (tempfile::TempDir, Arc<Factory>) {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join(".deepkit")).unwrap();
    fs::write(root.path().join(".deepkit/config.toml"),format!("[index]\nrepo_name='dense-fixture'\n[embedder]\nprofile='ollama'\ndims=2\n[embedder.ollama]\nurl='{}'\nmodel='dense-fixture'\n",server.uri())).unwrap();
    let config = config::load(root.path(), None).unwrap().config;
    let store = Arc::new(store::open(":memory:", StoreLimits::default()).unwrap());
    store
        .ensure_vector_table(VectorTableSpec {
            slug: "selected",
            model: "dense-fixture",
            profile: "ollama",
            dims: 2,
            created_at: 1,
            active: true,
        })
        .unwrap();
    for id in 1..=units {
        let hash = *blake3::hash(&id.to_le_bytes()).as_bytes();
        let docs = id > 450;
        {
            let mut writer = store.writer().unwrap();
            let tx = writer.transaction().unwrap();
            tx.execute("INSERT INTO files(id,path,blake3,size,mtime_ns,kind,treatment,max_level) VALUES(?1,?2,?3,1,1,'code','source',2)",rusqlite::params![id,format!("{}/{id}.rs",if docs {"docs"} else {"code"}),hash.as_slice()]).unwrap();
            tx.execute(
                "INSERT INTO unit_contents(hash,text,tokens) VALUES(?1,'fixture',1)",
                [hash.as_slice()],
            )
            .unwrap();
            tx.execute("INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,created_at,updated_at) VALUES(?1,?2,?2,'dense:' || ?1,?1,?3,'fixture',1,1)",rusqlite::params![id,hash.as_slice(),if docs {"doc_section"} else {"chunk"}]).unwrap();
            tx.commit().unwrap();
        }
        let vector = [1.0, if docs { (id - 450) as f32 / 100.0 } else { 0.0 }];
        store
            .insert_vector(VectorInsert {
                table_slug: "selected",
                unit_id: id,
                ctx_hash: &hash,
                embedding_hash: &hash,
                embedding: &vector,
            })
            .unwrap();
    }
    // A differently shaped table must never be inspected by single-tier retrieval.
    store
        .ensure_vector_table(VectorTableSpec {
            slug: "unused",
            model: "other-model",
            profile: "ollama",
            dims: 3,
            created_at: 1,
            active: true,
        })
        .unwrap();
    let factory = Factory::new(
        store,
        embedding_query::Settings {
            config_root: root.path().into(),
            config,
            kind: UnitKind::Chunk,
            budget: usage::Budget {
                calls: Some(20),
                tokens: None,
            },
        },
    )
    .unwrap();
    (root, factory)
}
fn response() -> ResponseTemplate {
    ResponseTemplate::new(200)
        .set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]],"prompt_eval_count":3}))
}
async fn run(factory: &Arc<Factory>, scope: Scope) -> Result<Vec<VectorHit>, Error> {
    retrieve(
        Arc::clone(factory),
        "selected".into(),
        "find the relevant implementation".into(),
        scope,
        Instant::now() + Duration::from_secs(2),
        || 100,
    )
    .await
}

#[tokio::test]
async fn dense_query_filters_before_top_k_and_returns_two_hundred_at_most() {
    let server = MockServer::start().await;
    let (_root, factory) = setup(&server, 500);
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(response())
        .expect(5)
        .mount(&server)
        .await;
    let unscoped = run(&factory, Scope::default()).await.unwrap();
    assert_eq!(unscoped.len(), 200);
    assert_eq!(unscoped.first().unwrap().unit_id, 1);
    assert_eq!(unscoped.last().unwrap().unit_id, 200);
    for scope in [
        Scope::parse(Some("docs"), Vec::new()),
        Scope::parse(None, (1..=450).collect()),
        Scope::parse(Some("docs/*"), (451..=480).collect()),
        Scope::from_search(
            Some(&store::SearchScope::Globs(vec![
                "absent/*".into(),
                "docs/*".into(),
                "docs/*".into(),
            ])),
            (451..=480).collect(),
        ),
    ] {
        let hits = run(&factory, scope).await.unwrap();
        assert!(!hits.is_empty());
        assert!(hits.iter().all(|hit| hit.unit_id > 450));
        assert_eq!(hits.last().unwrap().unit_id, 500);
        assert!(
            hits.windows(2)
                .all(|pair| pair[0].distance <= pair[1].distance
                    && pair[0].unit_id < pair[1].unit_id)
        );
    }
    let calls: i64 = factory
        .store()
        .writer()
        .unwrap()
        .query_row("SELECT sum(calls) FROM usage", [], |r| r.get(0))
        .unwrap();
    assert_eq!(calls, 5);
    assert_eq!(server.received_requests().await.unwrap().len(), 5);
}

#[tokio::test]
async fn empty_or_incompatible_tables_do_not_send_query_text() {
    let server = MockServer::start().await;
    let (_root, factory) = setup(&server, 0);
    assert!(run(&factory, Scope::default()).await.unwrap().is_empty());
    factory
        .store()
        .writer()
        .unwrap()
        .execute(
            "UPDATE embedding_tables SET model='another-model' WHERE slug='selected'",
            [],
        )
        .unwrap();
    assert!(matches!(
        run(&factory, Scope::default()).await,
        Err(Error::TableChanged)
    ));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn model_and_config_changes_during_http_discard_the_vector_before_scanning() {
    for config_change in [false, true] {
        let server = MockServer::start().await;
        let (root, factory) = setup(&server, 1);
        let store = Arc::clone(factory.store());
        let cfg_path = root.path().join(".deepkit/config.toml");
        Mock::given(method("POST")).and(path("/api/embed")).respond_with(move |_: &wiremock::Request| {
            if config_change {
                let cfg=fs::read_to_string(&cfg_path).unwrap();
                fs::write(&cfg_path,format!("{cfg}\nquery_prefix='new instruction'\n")).unwrap();
            } else {
                store.writer().unwrap().execute("UPDATE embedding_tables SET model='replaced-model' WHERE slug='selected'",[]).unwrap();
            }
            response()
        }).expect(1).mount(&server).await;
        let result = run(&factory, Scope::default()).await;
        assert!(if config_change {
            matches!(result, Err(Error::ConfigurationChanged))
        } else {
            matches!(result, Err(Error::TableChanged))
        });
        let calls: i64 = factory
            .store()
            .writer()
            .unwrap()
            .query_row("SELECT sum(calls) FROM usage", [], |r| r.get(0))
            .unwrap();
        assert_eq!(calls, 1);
    }
}

#[tokio::test]
async fn configured_deadline_covers_preflight_reader_wait_without_late_http() {
    let server = MockServer::start().await;
    let (root, initial) = setup(&server, 1);
    let cfg_path = root.path().join(".deepkit/config.toml");
    let cfg = fs::read_to_string(&cfg_path).unwrap();
    fs::write(&cfg_path, format!("{cfg}\n[limits]\nquery_timeout_ms=50\n")).unwrap();
    let factory = Factory::new(
        Arc::clone(initial.store()),
        embedding_query::Settings {
            config_root: root.path().into(),
            config: config::load(root.path(), None).unwrap().config,
            kind: UnitKind::Chunk,
            budget: usage::Budget {
                calls: Some(20),
                tokens: None,
            },
        },
    )
    .unwrap();
    let store = Arc::clone(factory.store());
    let (ready, acquired) = tokio::sync::oneshot::channel();
    let (release, released) = std::sync::mpsc::channel();
    let holder = tokio::task::spawn_blocking(move || {
        let _lock = store.writer().unwrap();
        ready.send(()).unwrap();
        released.recv().unwrap();
    });
    acquired.await.unwrap();
    let result =
        tokio::time::timeout(Duration::from_millis(500), run(&factory, Scope::default())).await;
    release.send(()).unwrap();
    holder.await.unwrap();
    assert!(matches!(result.unwrap(), Err(Error::Deadline)));
    // Wait for queued preflight work to observe cancellation after the reader opens.
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(server.received_requests().await.unwrap().is_empty());
    let held: i64 = factory
        .store()
        .writer()
        .unwrap()
        .query_row("SELECT count(*) FROM usage_reservations", [], |r| r.get(0))
        .unwrap();
    assert_eq!(held, 0);
}

#[test]
fn array_scope_keeps_literal_globs_and_empty_scope_restrictive() {
    for globs in [vec!["docs".into(), "src/*".into()], Vec::new()] {
        let scope = Scope::from_search(Some(&store::SearchScope::Globs(globs.clone())), vec![7]);
        assert!(scope.kinds.is_empty());
        assert_eq!(scope.path_globs, Some(globs));
        assert_eq!(scope.exclude_ids, [7]);
    }
    assert_eq!(
        Scope::parse(Some("images"), Vec::new()).kinds,
        [store::UnitKind::Image]
    );
}
