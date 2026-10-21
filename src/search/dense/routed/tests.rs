//! Concurrent replacement and per-space query evidence for v1 §7.3.
use super::*;
use crate::{
    queue::{destination, usage},
    search::embedding_query,
    store::vectors::{VectorInsert, VectorTableSpec},
};
use std::{fs, time::Duration};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

struct Fixture {
    root: tempfile::TempDir,
    store: Arc<Store>,
    config: Config,
    factories: Vec<Arc<Factory>>,
    old: String,
    new: String,
    generation: i64,
}
fn routes(table: &str) -> transitions::Routes {
    UnitKind::ALL
        .into_iter()
        .map(|kind| (kind.as_str().into(), table.into()))
        .collect()
}
fn table(store: &Store, config: &Config) -> String {
    let binding = destination::resolve(config, UnitKind::Chunk)
        .unwrap()
        .binding(config);
    let slug = binding.destination_slug();
    store
        .ensure_vector_table(VectorTableSpec {
            slug: &slug,
            model: &binding.model,
            profile: &binding.profile,
            dims: 2,
            created_at: 0,
            active: true,
        })
        .unwrap();
    store.bind_embedding_table(&slug, &binding).unwrap();
    slug
}
fn settings(root: &std::path::Path, config: Config) -> embedding_query::Settings {
    embedding_query::Settings {
        config_root: root.into(),
        config,
        kind: UnitKind::Chunk,
        budget: usage::Budget {
            calls: Some(20),
            tokens: Some(10000),
        },
    }
}
fn put(store: &Store, slug: &str, id: i64, replacement: bool) {
    let hash: Vec<u8> = store
        .writer()
        .unwrap()
        .query_row("SELECT content_hash FROM units WHERE id=?1", [id], |r| {
            r.get(0)
        })
        .unwrap();
    let vector = VectorInsert {
        table_slug: slug,
        unit_id: id,
        ctx_hash: &[0; 32],
        embedding_hash: &hash,
        embedding: &[1.0, 0.0],
    };
    if replacement {
        store.replace_vector(vector).unwrap()
    } else {
        store.insert_vector(vector).unwrap()
    }
}
async fn fixture(first: &MockServer, second: &MockServer, count: i64, cap: u64) -> Fixture {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join(".deepkit")).unwrap();
    let store = Arc::new(
        store::open(
            root.path().join(".deepkit/code.db"),
            store::StoreLimits {
                max_l2_units: cap,
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let mut config = Config::default();
    config.embedder.profile = "ollama".into();
    config.embedder.dims = 2;
    config.embedder.ollama.url = first.uri();
    config.embedder.ollama.model = "old-model".into();
    config.limits.max_l2_units = cap;
    config.limits.query_timeout_ms = 3000;
    let old = table(&store, &config);
    store
        .prepare_transition(|_| Ok::<_, store::Error>(routes(&old)))
        .unwrap();
    for id in 1..=count {
        let hash = blake3::hash(&id.to_le_bytes());
        store
            .writer()
            .unwrap()
            .execute(
                "INSERT INTO unit_contents(hash,text,tokens) VALUES(?1,'fixture',1)",
                [hash.as_bytes().as_slice()],
            )
            .unwrap();
        store.writer().unwrap().execute("INSERT INTO units(id,hash,content_hash,source_key,kind,header,created_at,updated_at) VALUES(?1,?2,?2,'fixture:' || ?1,'chunk','fixture',0,0)",rusqlite::params![id,hash.as_bytes().as_slice()]).unwrap();
        put(&store, &old, id, false);
    }
    config.embedder.ollama.url = second.uri();
    config.embedder.ollama.model = "new-model".into();
    let new = table(&store, &config);
    let generation = store
        .prepare_transition(|_| Ok::<_, store::Error>(routes(&new)))
        .unwrap()
        .generation;
    fs::write(
        root.path().join(".deepkit/config.toml"),
        toml::to_string(&config).unwrap(),
    )
    .unwrap();
    let current = Factory::new(Arc::clone(&store), settings(root.path(), config.clone())).unwrap();
    let previous = Factory::new_previous(
        Arc::clone(&store),
        settings(root.path(), config.clone()),
        old.clone(),
        generation,
        root.path().join("unused"),
    )
    .await
    .unwrap();
    Fixture {
        root,
        store,
        config,
        factories: vec![current, previous],
        old,
        new,
        generation,
    }
}
fn response() -> ResponseTemplate {
    ResponseTemplate::new(200)
        .set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]],"prompt_eval_count":3}))
}
impl Fixture {
    async fn run(&self) -> Result<Results, Error> {
        self.run_scoped(Scope::default()).await
    }
    async fn run_scoped(&self, scope: Scope) -> Result<Results, Error> {
        retrieve(
            Providers {
                store: Arc::clone(&self.store),
                config: self.config.clone(),
                config_root: self.root.path().into(),
                factories: self.factories.clone(),
            },
            "find implementation".into(),
            scope,
            Instant::now() + Duration::from_secs(3),
            || 100,
        )
        .await
    }
}
fn ids(results: &Results, table: &str) -> Vec<i64> {
    results
        .spaces
        .iter()
        .find(|s| s.table == table)
        .unwrap()
        .hits
        .as_ref()
        .unwrap()
        .iter()
        .map(|hit| hit.unit_id)
        .collect()
}

#[tokio::test]
async fn initially_empty_destination_is_queried_when_old_coverage_moves_during_http() {
    let first = MockServer::start().await;
    let second = MockServer::start().await;
    let fixture = fixture(&first, &second, 1, 1).await;
    let store = Arc::clone(&fixture.store);
    let new = fixture.new.clone();
    let old = fixture.old.clone();
    let generation = fixture.generation;
    Mock::given(method("POST"))
        .respond_with(move |_: &wiremock::Request| {
            put(&store, &new, 1, true);
            assert_eq!(store.count_vectors().unwrap(), 1);
            assert!(
                store
                    .finish_transition(generation)
                    .unwrap()
                    .previous
                    .is_empty()
            );
            assert!(store.retire_empty_vector_table(&old).unwrap());
            response()
        })
        .expect(1)
        .mount(&first)
        .await;
    Mock::given(method("POST"))
        .respond_with(response())
        .expect(1)
        .mount(&second)
        .await;
    let results = fixture.run().await.unwrap();
    assert_eq!(results.generation, generation);
    assert_eq!(results.spaces.len(), 1);
    assert_eq!(ids(&results, &fixture.new), vec![1]);
    assert_eq!(fixture.store.count_vectors().unwrap(), 1);
    assert_eq!(
        fixture
            .store
            .writer()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM usage_reservations WHERE state='reconciled'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        2
    );
}

#[tokio::test]
async fn steady_single_mode_ignores_an_unreadable_unrelated_vector_table() {
    let first = MockServer::start().await;
    let second = MockServer::start().await;
    let fixture = fixture(&first, &second, 1, 1).await;
    put(&fixture.store, &fixture.new, 1, true);
    fixture.store.finish_transition(fixture.generation).unwrap();
    let old = vectors::vector_table_name(&fixture.old).unwrap();
    fixture
        .store
        .writer()
        .unwrap()
        .execute_batch(&format!(
            "DROP TABLE {old}; CREATE VIEW {old} AS SELECT unit_id FROM forbidden_missing_table"
        ))
        .unwrap();
    Mock::given(method("POST"))
        .respond_with(response())
        .expect(1)
        .mount(&second)
        .await;
    let (results, scans) = vectors::scan_observation::capture(fixture.run()).await;
    let results = results.unwrap();
    assert_eq!(scans.len(), 1);
    assert_eq!(scans[0].table, fixture.new);
    assert!(scans[0].sql.contains("vector_full_scan"));
    assert_eq!(results.spaces.len(), 1);
    assert_eq!(ids(&results, &fixture.new), vec![1]);
    assert!(first.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn all_vector_scans_share_one_snapshot_while_another_connection_replaces_rows() {
    let first = MockServer::start().await;
    let second = MockServer::start().await;
    let fixture = fixture(&first, &second, 1, 1).await;
    let other = store::open(
        fixture.root.path().join(".deepkit/code.db"),
        store::StoreLimits {
            max_l2_units: 1,
            ..Default::default()
        },
    )
    .unwrap();
    fixture
        .store
        .vector_snapshot(|tx| {
            let old =
                vectors::scan_vectors_in(tx, &fixture.old, &[1.0, 0.0], 400, Default::default())
                    .unwrap();
            put(&other, &fixture.new, 1, true);
            let new =
                vectors::scan_vectors_in(tx, &fixture.new, &[1.0, 0.0], 400, Default::default())
                    .unwrap();
            assert_eq!(old.len(), 1);
            assert!(new.is_empty());
            assert_eq!(
                vectors::scan_vectors_in(tx, &fixture.old, &[1.0, 0.0], 400, Default::default())
                    .unwrap()
                    .len(),
                1
            );
        })
        .unwrap();
    Mock::given(method("POST"))
        .respond_with(response())
        .expect(1)
        .mount(&second)
        .await;
    let results = fixture.run().await.unwrap();
    assert!(ids(&results, &fixture.old).is_empty());
    assert_eq!(ids(&results, &fixture.new), vec![1]);
    assert!(first.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn already_replaced_rows_cannot_starve_unmigrated_top_k() {
    let first = MockServer::start().await;
    let second = MockServer::start().await;
    let fixture = fixture(&first, &second, 401, 802).await;
    // Legacy overlapping coverage: the first 400 old candidates also exist in
    // the destination, so filtering after top-k would lose occurrence 401.
    for id in 1..=400 {
        put(&fixture.store, &fixture.new, id, false);
    }
    for server in [&first, &second] {
        Mock::given(method("POST"))
            .respond_with(response())
            .expect(1)
            .mount(server)
            .await;
    }
    let results = fixture.run().await.unwrap();
    assert_eq!(ids(&results, &fixture.old), vec![401]);
    assert_eq!(ids(&results, &fixture.new), (1..=200).collect::<Vec<_>>());
}

#[tokio::test]
async fn failed_or_missing_old_provider_preserves_the_successful_new_list() {
    for missing in [false, true] {
        let first = MockServer::start().await;
        let second = MockServer::start().await;
        let mut fixture = fixture(&first, &second, 2, 2).await;
        put(&fixture.store, &fixture.new, 1, true);
        if missing {
            fixture.factories.pop();
        }
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(400))
            .expect(u64::from(!missing))
            .mount(&first)
            .await;
        Mock::given(method("POST"))
            .respond_with(response())
            .expect(1)
            .mount(&second)
            .await;
        let results = fixture.run().await.unwrap();
        assert_eq!(ids(&results, &fixture.new), vec![1]);
        let failed = &results
            .spaces
            .iter()
            .find(|s| s.table == fixture.old)
            .unwrap()
            .hits;
        assert!(if missing {
            matches!(failed, Err(Error::Unavailable))
        } else {
            matches!(failed, Err(Error::Query(_)))
        });
    }
}

#[tokio::test]
async fn a_new_routing_generation_during_embedding_is_not_reported_as_complete() {
    let first = MockServer::start().await;
    let second = MockServer::start().await;
    let fixture = fixture(&first, &second, 1, 1).await;
    let changed = Arc::clone(&fixture.store);
    Mock::given(method("POST"))
        .respond_with(move |_: &wiremock::Request| {
            changed
                .writer()
                .unwrap()
                .execute(
                    "UPDATE embedding_transition SET generation=generation+1",
                    [],
                )
                .unwrap();
            response()
        })
        .expect(1)
        .mount(&first)
        .await;
    Mock::given(method("POST"))
        .respond_with(response())
        .mount(&second)
        .await;
    assert!(matches!(fixture.run().await, Err(Error::RoutingChanged)));
}

#[tokio::test]
async fn glob_union_applies_to_transition_spaces_before_limits() {
    let first = MockServer::start().await;
    let second = MockServer::start().await;
    let fixture = fixture(&first, &second, 205, 205).await;
    fixture.store.writer().unwrap().execute_batch("INSERT INTO files(id,path,blake3,size,mtime_ns,kind,max_level,treatment) VALUES(1,'left/a.rs',zeroblob(32),0,0,'code',2,'default'),(2,'right/b.rs',zeroblob(32),0,0,'code',2,'default'); UPDATE units SET file_id=1 WHERE id=204; UPDATE units SET file_id=2 WHERE id=205;").unwrap();
    put(&fixture.store, &fixture.new, 205, true);
    for server in [&first, &second] {
        Mock::given(method("POST"))
            .respond_with(response())
            .expect(1)
            .mount(server)
            .await;
    }
    let scope = Scope::from_search(
        Some(&store::SearchScope::Globs(vec![
            "left/*".into(),
            "right/*".into(),
            "right/*".into(),
        ])),
        vec![],
    );
    let results = fixture.run_scoped(scope).await.unwrap();
    assert_eq!(ids(&results, &fixture.old), [204]);
    assert_eq!(ids(&results, &fixture.new), [205]);
}

// A genuine two-profile fixture; remote construction injects an empty mock key,
// avoiding process-global environment mutations and host credential files.
struct DualFixture {
    root: tempfile::TempDir,
    store: Arc<Store>,
    config: Config,
    code: String,
    docs: String,
    generation: i64,
}
fn table_for_kind(store: &Store, config: &Config, kind: UnitKind) -> String {
    let binding = destination::resolve(config, kind).unwrap().binding(config);
    let slug = binding.destination_slug();
    store
        .ensure_vector_table(VectorTableSpec {
            slug: &slug,
            model: &binding.model,
            profile: &binding.profile,
            dims: 2,
            created_at: 0,
            active: true,
        })
        .unwrap();
    store.bind_embedding_table(&slug, &binding).unwrap();
    slug
}
impl DualFixture {
    async fn new(code_server: &MockServer, docs_server: &MockServer) -> Self {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join(".deepkit")).unwrap();
        let store = Arc::new(
            store::open(
                root.path().join(".deepkit/code.db"),
                store::StoreLimits {
                    max_l2_units: 6,
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let mut config = Config::default();
        config.embedder.profile = "ollama".into();
        config.embedder.dims = 2;
        config.embedder.ollama.url = code_server.uri();
        config.embedder.ollama.model = "code-fixture".into();
        config.embedder.remote.url = docs_server.uri();
        config.embedder.remote.model = "docs-fixture".into();
        config.embedder.remote.api_key_env = "UNUSED_ROUTED_FIXTURE".into();
        config.limits.query_timeout_ms = 3000;
        config.limits.max_l2_units = 6;
        let code = table_for_kind(&store, &config, UnitKind::Chunk);
        store
            .prepare_transition(|_| Ok::<_, store::Error>(routes(&code)))
            .unwrap();
        for (id, kind) in [
            (1i64, UnitKind::Chunk),
            (2, UnitKind::PullRequest),
            (3, UnitKind::PullRequest),
        ] {
            let hash = blake3::hash(&id.to_le_bytes());
            store
                .writer()
                .unwrap()
                .execute(
                    "INSERT INTO unit_contents(hash,text,tokens) VALUES(?1,'fixture',1)",
                    [hash.as_bytes().as_slice()],
                )
                .unwrap();
            store.writer().unwrap().execute("INSERT INTO units(id,hash,content_hash,source_key,kind,header,created_at,updated_at) VALUES(?1,?2,?2,'fixture:' || ?1,?3,'fixture',0,0)",
                rusqlite::params![id, hash.as_bytes().as_slice(), kind.as_str()]).unwrap();
            put(&store, &code, id, false);
        }
        config.embedder.docs = Some(config::DocsEmbedder {
            profile: "remote".into(),
            kinds: vec!["pr".into()],
        });
        let docs = table_for_kind(&store, &config, UnitKind::PullRequest);
        let mut desired = routes(&code);
        desired.insert("pr".into(), docs.clone());
        let generation = store
            .prepare_transition(|_| Ok::<_, store::Error>(desired.clone()))
            .unwrap()
            .generation;
        fs::write(
            root.path().join(".deepkit/config.toml"),
            toml::to_string(&config).unwrap(),
        )
        .unwrap();
        Self {
            root,
            store,
            config,
            code,
            docs,
            generation,
        }
    }
    fn settings(&self, kind: UnitKind) -> embedding_query::Settings {
        let mut settings = settings(self.root.path(), self.config.clone());
        settings.kind = kind;
        settings
    }
    async fn query(&self) -> Result<Results, Error> {
        let code = Factory::new(Arc::clone(&self.store), self.settings(UnitKind::Chunk)).unwrap();
        let docs = if self.config.embedder.docs.is_some() {
            Factory::new_mock_remote(
                Arc::clone(&self.store),
                self.settings(UnitKind::PullRequest),
            )
            .unwrap()
        } else {
            Factory::new_previous_mock_remote(
                Arc::clone(&self.store),
                self.settings(UnitKind::PullRequest),
                self.docs.clone(),
                self.generation,
                self.root.path().join("unused-model-cache"),
            )
            .await
            .unwrap()
        };
        retrieve(
            Providers {
                store: Arc::clone(&self.store),
                config: self.config.clone(),
                config_root: self.root.path().into(),
                factories: vec![code, docs],
            },
            "find implementation".into(),
            Scope {
                kinds: vec![UnitKind::PullRequest],
                ..Default::default()
            },
            Instant::now() + Duration::from_secs(3),
            || 100,
        )
        .await
    }
}
async fn dual_servers() -> (MockServer, MockServer) {
    let code = MockServer::start().await;
    let docs = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(response())
        .mount(&code)
        .await;
    Mock::given(method("POST")).respond_with(ResponseTemplate::new(200).set_body_json(
        serde_json::json!({"data":[{"index":0,"embedding":"AACAPwAAAAA="}],"usage":{"total_tokens":3}})
    )).mount(&docs).await;
    (code, docs)
}
fn assert_pr_scan(event: &vectors::scan_observation::Event) {
    assert_eq!(event.kinds, vec![UnitKind::PullRequest]);
    assert!(event.sql.contains("vector_full_scan"));
    assert!(event.sql.contains("u.kind IN"));
}

#[tokio::test]
async fn pr_scope_executes_only_docs_scan_in_steady_dual_mode() {
    let (code, docs) = dual_servers().await;
    let fixture = DualFixture::new(&code, &docs).await;
    put(&fixture.store, &fixture.docs, 2, true);
    put(&fixture.store, &fixture.docs, 3, true);
    fixture.store.finish_transition(fixture.generation).unwrap();
    let (result, scans) = vectors::scan_observation::capture(fixture.query()).await;
    let result = result.unwrap();
    assert_eq!(scans.len(), 1);
    assert_eq!(scans[0].table, fixture.docs);
    assert_pr_scan(&scans[0]);
    assert!(scans[0].replaced_by.is_empty());
    assert_eq!(ids(&result, &fixture.docs), vec![2, 3]);
    assert!(code.received_requests().await.unwrap().is_empty());
    assert_eq!(docs.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn pr_scope_executes_old_and_docs_scans_during_enable_with_destination_duplicate_exclusion() {
    let (code, docs) = dual_servers().await;
    let fixture = DualFixture::new(&code, &docs).await;
    // Deliberate legacy overlap proves destination preference before old top-k.
    put(&fixture.store, &fixture.docs, 2, false);
    let (result, scans) = vectors::scan_observation::capture(fixture.query()).await;
    let result = result.unwrap();
    assert_eq!(scans.len(), 2);
    let by_table: BTreeMap<_, _> = scans
        .iter()
        .map(|scan| (scan.table.as_str(), scan))
        .collect();
    assert_eq!(by_table.len(), 2);
    for scan in &scans {
        assert_pr_scan(scan);
    }
    assert_eq!(
        by_table[fixture.code.as_str()].replaced_by,
        vec![(UnitKind::PullRequest, fixture.docs.clone())]
    );
    assert!(by_table[fixture.code.as_str()].sql.contains("NOT EXISTS"));
    assert!(by_table[fixture.docs.as_str()].replaced_by.is_empty());
    assert_eq!(ids(&result, &fixture.code), vec![3]);
    assert_eq!(ids(&result, &fixture.docs), vec![2]);
    assert_eq!(code.received_requests().await.unwrap().len(), 1);
    assert_eq!(docs.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn pr_scope_executes_old_docs_and_code_scans_during_disable() {
    let (code, docs) = dual_servers().await;
    let mut fixture = DualFixture::new(&code, &docs).await;
    put(&fixture.store, &fixture.docs, 2, true);
    put(&fixture.store, &fixture.docs, 3, true);
    fixture.store.finish_transition(fixture.generation).unwrap();
    fixture.config.embedder.docs = None;
    fs::write(
        fixture.root.path().join(".deepkit/config.toml"),
        toml::to_string(&fixture.config).unwrap(),
    )
    .unwrap();
    fixture.generation = fixture
        .store
        .prepare_transition(|_| Ok::<_, store::Error>(routes(&fixture.code)))
        .unwrap()
        .generation;
    put(&fixture.store, &fixture.code, 2, false);
    let (result, scans) = vectors::scan_observation::capture(fixture.query()).await;
    let result = result.unwrap();
    assert_eq!(scans.len(), 2);
    let by_table: BTreeMap<_, _> = scans
        .iter()
        .map(|scan| (scan.table.as_str(), scan))
        .collect();
    assert_eq!(by_table.len(), 2);
    for scan in &scans {
        assert_pr_scan(scan);
    }
    assert_eq!(
        by_table[fixture.docs.as_str()].replaced_by,
        vec![(UnitKind::PullRequest, fixture.code.clone())]
    );
    assert!(by_table[fixture.docs.as_str()].sql.contains("NOT EXISTS"));
    assert!(by_table[fixture.code.as_str()].replaced_by.is_empty());
    assert_eq!(ids(&result, &fixture.docs), vec![3]);
    assert_eq!(ids(&result, &fixture.code), vec![2]);
    assert_eq!(code.received_requests().await.unwrap().len(), 1);
    assert_eq!(docs.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn simultaneous_scan_observers_remain_scoped_to_their_own_query() {
    let (code, docs) = dual_servers().await;
    let fixture = DualFixture::new(&code, &docs).await;
    put(&fixture.store, &fixture.docs, 2, true);
    put(&fixture.store, &fixture.docs, 3, true);
    fixture.store.finish_transition(fixture.generation).unwrap();
    let (first, second) = tokio::join!(
        vectors::scan_observation::capture(fixture.query()),
        vectors::scan_observation::capture(fixture.query()),
    );
    for (result, scans) in [first, second] {
        assert_eq!(ids(&result.unwrap(), &fixture.docs), vec![2, 3]);
        assert_eq!(scans.len(), 1);
        assert_eq!(scans[0].table, fixture.docs);
    }
}
