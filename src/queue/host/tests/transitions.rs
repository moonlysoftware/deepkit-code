//! Bidirectional host publication at the full physical cap; v1 §§7.3, 16.
use super::*;
use crate::store::{
    self, FileKind, StoreLimits,
    vectors::{self, VectorScanPredicate},
};
use rusqlite::{Connection, Transaction, params};
use std::{
    collections::BTreeSet,
    path::Path,
    sync::{Condvar, Mutex},
};

// The remote key exists only in a child test process. No global environment
// mutation, user credential file or model cache is involved.
fn isolated(name: &str) -> Result<bool, Box<dyn StdError>> {
    if std::env::var_os("T113_TRANSITION_CHILD").is_some() {
        return Ok(false);
    }
    let home = tempfile::tempdir()?;
    let result = std::process::Command::new(std::env::current_exe()?)
        .args(["--exact", name, "--nocapture"])
        .env_clear()
        .env("HOME", home.path())
        .env("XDG_CONFIG_HOME", home.path())
        .env("T113_TRANSITION_CHILD", "1")
        .env("T113_FIXTURE_KEY", "fixture-placeholder")
        .output()?;
    assert!(
        result.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(true)
}

#[derive(Default)]
struct Gate {
    armed: AtomicBool,
    entered: tokio::sync::Notify,
    released: Mutex<bool>,
    ready: Condvar,
}
impl Gate {
    fn arm(&self) {
        *self.released.lock().unwrap() = false;
        self.armed.store(true, Ordering::SeqCst);
    }
    fn wait(&self) {
        if self.armed.swap(false, Ordering::SeqCst) {
            self.entered.notify_one();
            // The finite watchdog also releases a responder if an assertion fails.
            drop(
                self.ready
                    .wait_timeout_while(
                        self.released.lock().unwrap(),
                        Duration::from_secs(15),
                        |released| !*released,
                    )
                    .unwrap(),
            );
        }
    }
    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.ready.notify_all();
    }
}

async fn providers() -> (MockServer, MockServer, Arc<Gate>, Arc<Gate>) {
    let code = MockServer::start().await;
    let docs = MockServer::start().await;
    let code_gate = Arc::new(Gate::default());
    let docs_gate = Arc::new(Gate::default());
    let gate = Arc::clone(&code_gate);
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(move |_: &wiremock::Request| {
            gate.wait();
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]]}))
        })
        .mount(&code)
        .await;
    let gate = Arc::clone(&docs_gate);
    Mock::given(method("POST")).and(path("/v1/embeddings")).respond_with(move |_: &wiremock::Request| {
        gate.wait();
        ResponseTemplate::new(200).set_body_json(serde_json::json!({"data":[{"index":0,"embedding":"AAAAAAAAgD8="}],"usage":{"total_tokens":5}}))
    }).mount(&docs).await;
    (code, docs, code_gate, docs_gate)
}

fn persist(root: &Path, config: &Config) -> Result<(), Box<dyn StdError>> {
    fs::write(root.join(".deepkit/config.toml"), config.dump()?)?;
    Ok(())
}

async fn capped_fixture(
    code: &MockServer,
    docs: &MockServer,
) -> Result<(tempfile::TempDir, Arc<Store>, Config), Box<dyn StdError>> {
    let (root, initial, mut config) = fixture(code).await?;
    config.limits.max_l2_units = 2;
    config.limits.wal_max_bytes = "8 MiB".into();
    config.embedder.remote.url = format!("{}/v1", docs.uri());
    config.embedder.remote.model = "transition-docs-fixture".into();
    config.embedder.remote.api_key_env = "T113_FIXTURE_KEY".into();
    persist(root.path(), &config)?;
    let text = "Example documentation for the configuration parser.";
    fs::write(root.path().join("README.md"), text)?;
    let file_hash = *blake3::hash(text.as_bytes()).as_bytes();
    let hash = crate::treat::context::unit_hash("doc_section", None, text);
    let treatment = crate::treat::Router::new(&config.treatments, config.llm.enabled)?
        .route("README.md", FileKind::Doc);
    let fingerprint = crate::sync::l1::file_fingerprint(
        &config,
        "fixture",
        crate::sync::l1::FingerprintFile {
            path: "README.md",
            content_hash: &file_hash,
            language: None,
            kind: FileKind::Doc,
            capped: false,
        },
        &treatment,
    );
    initial.writer()?.execute("INSERT INTO files(id,path,blake3,size,mtime_ns,kind,max_level,treatment,l1_pending,l1_desired,l1_fingerprint) VALUES(2,'README.md',?1,?2,0,'doc',3,?3,0,?4,?4)", params![file_hash.as_slice(), text.len() as i64, treatment.name, fingerprint.as_slice()])?;
    initial.writer()?.execute(
        "INSERT INTO unit_contents(hash,text,tokens) VALUES(?1,?2,10)",
        params![hash.as_slice(), text],
    )?;
    initial.writer()?.execute("INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,created_at,updated_at) VALUES(2,?1,?1,'README.md',2,'doc_section','Example documentation',0,0)",[hash.as_slice()])?;
    initial.writer()?.execute_batch("INSERT INTO fts_units(rowid,header,idents,body,prose) VALUES(1,'example','','fn example() {}',''),(2,'Example documentation','','','Example documentation for the configuration parser.');")?;
    let path = root.path().join(".deepkit/transition.db");
    initial
        .writer()?
        .execute("VACUUM INTO ?1", [path.to_string_lossy().as_ref()])?;
    let store = Arc::new(store::open(
        &path,
        StoreLimits {
            max_l2_units: 2,
            wal_max_bytes: 8 * 1024 * 1024,
            ..Default::default()
        },
    )?);
    let runner = prepare(
        root.path(),
        Arc::clone(&store),
        config.clone(),
        UnitKind::Chunk,
    )
    .await?;
    for unit in [1, 2] {
        assert_eq!(
            runner
                .run(vec![target(unit)], || 0)
                .await?
                .execution
                .published,
            1
        );
    }
    assert_eq!(store.count_vectors()?, 2);
    Ok((root, store, config))
}

async fn prepare(
    root: &Path,
    store: Arc<Store>,
    config: Config,
    kind: UnitKind,
) -> Result<Arc<Host>, Error> {
    Host::prepare(
        store,
        Settings {
            root: root.into(),
            config_root: root.into(),
            config,
            kind,
            local_cache: root.join("unused-cache"),
            bounds: Bounds {
                timeout: Duration::from_secs(30),
                ..bounds()
            },
        },
        0,
    )
    .await
    .map(Arc::new)
}

// Counts and scans use one transaction so a read can see only a complete old or
// new publication. No claim is made about a cross-snapshot sum of table counts.
fn observe(tx: &Transaction<'_>) -> Result<(Vec<i64>, usize), store::Error> {
    let tables = tx
        .prepare("SELECT slug FROM embedding_tables ORDER BY slug")?
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut covered = BTreeSet::new();
    let mut dense = BTreeSet::new();
    let mut physical = 0;
    for table in tables {
        let ids = tx
            .prepare(&format!("SELECT unit_id FROM vec_{table}"))?
            .query_map([], |row| row.get::<_, i64>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        physical += ids.len();
        covered.extend(ids);
        dense.extend(
            vectors::scan_vectors_in(tx, &table, &[1.0, 0.0], 10, VectorScanPredicate::default())?
                .into_iter()
                .map(|hit| hit.unit_id),
        );
    }
    assert_eq!(covered, BTreeSet::from([1, 2]));
    assert_eq!(dense, covered);
    assert_eq!(physical, 2);
    let lexical = tx
        .prepare("SELECT rowid FROM fts_units WHERE fts_units MATCH 'example' ORDER BY rowid")?
        .query_map([], |row| row.get::<_, i64>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    assert_eq!(lexical, vec![1, 2]);
    Ok((covered.into_iter().collect(), physical))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_cap_enable_disable_preserves_concurrent_coverage_and_prior_snapshots()
-> Result<(), Box<dyn StdError>> {
    if isolated(
        "queue::host::tests::transitions::full_cap_enable_disable_preserves_concurrent_coverage_and_prior_snapshots",
    )? {
        return Ok(());
    }
    let (code, docs, code_gate, docs_gate) = providers().await;
    let (root, store, mut config) = capped_fixture(&code, &docs).await?;
    let base = store.vector_tables()?[0].slug.clone();
    let mut docs_table = None;
    for enabling in [true, false] {
        config.embedder.docs = enabling.then(|| config::DocsEmbedder {
            profile: "remote".into(),
            kinds: vec!["doc_section".into()],
        });
        persist(root.path(), &config)?;
        let runner = prepare(
            root.path(),
            Arc::clone(&store),
            config.clone(),
            UnitKind::DocSection,
        )
        .await?;
        if enabling {
            docs_table = Some(runner.table_slug.clone());
        } else {
            assert_eq!(runner.table_slug, base);
        }
        assert!(!runner.transition.previous.is_empty());
        store::wal::checkpoint(&store)?;
        let mut prior = Connection::open(root.path().join(".deepkit/transition.db"))?;
        let prior = prior.transaction()?;
        let before = observe(&prior)?;
        let route = store::transitions::read(&prior)?.ok_or("transition")?;
        let old_doc = if enabling {
            &base
        } else {
            docs_table.as_ref().ok_or("docs table")?
        };
        let old_bytes: Vec<u8> = prior.query_row(
            &format!("SELECT v FROM vec_{old_doc} WHERE unit_id=2"),
            [],
            |row| row.get(0),
        )?;
        let gate = if enabling { &docs_gate } else { &code_gate };
        gate.arm();
        let work = Arc::clone(&runner);
        let pending = tokio::spawn(async move { work.run(vec![target(2)], || 1).await });
        tokio::time::timeout(Duration::from_secs(10), gate.entered.notified()).await?;
        for _ in 0..8 {
            assert_eq!(store.read_snapshot(observe)??, before);
            assert_eq!(observe(&prior)?, before);
            tokio::task::yield_now().await;
        }
        gate.release();
        let report = tokio::time::timeout(Duration::from_secs(20), async {
            while !pending.is_finished() {
                store.read_snapshot(observe)??;
                tokio::task::yield_now().await;
            }
            Ok::<_, Box<dyn StdError>>(pending.await??)
        })
        .await??;
        assert_eq!(report.execution.published, 1, "{:?}", report.execution);
        assert!(report.execution.errors.is_empty());
        assert_eq!(observe(&prior)?, before);
        assert_eq!(
            store::transitions::read(&prior)?.ok_or("transition")?,
            route
        );
        assert_eq!(
            prior.query_row(
                &format!("SELECT v FROM vec_{old_doc} WHERE unit_id=2"),
                [],
                |row| row.get::<_, Vec<u8>>(0)
            )?,
            old_bytes
        );
        prior.commit()?;
        // Retirement may conservatively wait for a pinned reader. An explicit
        // empty pass must finish it after that reader is released, without calls.
        let final_pass = runner.run(Vec::new(), || 2).await?;
        assert_eq!(final_pass.execution.claimed, 0);
        assert!(final_pass.retirement.error.is_none());
        assert!(
            final_pass
                .transition
                .ok_or("completed transition")?
                .previous
                .is_empty()
        );
        store.read_snapshot(observe)??;
        assert_eq!(store.vector_tables()?.len(), if enabling { 2 } else { 1 });
        if !enabling {
            let old = docs_table.as_ref().ok_or("docs table")?;
            assert_eq!(
                store.writer()?.query_row(
                    "SELECT count(*) FROM sqlite_schema WHERE name=?1",
                    [format!("vec_{old}")],
                    |row| row.get::<_, i64>(0)
                )?,
                0
            );
        }
    }
    assert_eq!(
        code.received_requests().await.ok_or("code requests")?.len(),
        3
    );
    assert_eq!(
        docs.received_requests().await.ok_or("docs requests")?.len(),
        1
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_cap_transition_wal_failure_pauses_then_resumes_without_losing_old_rows()
-> Result<(), Box<dyn StdError>> {
    if isolated(
        "queue::host::tests::transitions::full_cap_transition_wal_failure_pauses_then_resumes_without_losing_old_rows",
    )? {
        return Ok(());
    }
    let (code, docs, _, _) = providers().await;
    let (root, store, mut config) = capped_fixture(&code, &docs).await?;
    let old = store.vector_tables()?[0].slug.clone();
    config.embedder.docs = Some(config::DocsEmbedder {
        profile: "remote".into(),
        kinds: vec!["doc_section".into()],
    });
    persist(root.path(), &config)?;
    let runner = prepare(
        root.path(),
        Arc::clone(&store),
        config,
        UnitKind::DocSection,
    )
    .await?;
    store.writer()?.execute_batch(&format!("CREATE TABLE transition_pressure(value BLOB); CREATE TRIGGER transition_overflow AFTER INSERT ON vec_{} BEGIN INSERT INTO transition_pressure VALUES(zeroblob(16777216)); END;",runner.table_slug))?;
    store::wal::checkpoint(&store)?;
    let before = store.read_snapshot(observe)??;
    let failed = runner.run(vec![target(2)], || 1).await?;
    assert_eq!(failed.execution.published, 0);
    assert_eq!(failed.execution.deferred, 1, "{:?}", failed.execution);
    assert!(failed.execution.errors.is_empty());
    assert_eq!(store.read_snapshot(observe)??, before);
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT count(*) FROM transition_pressure", [], |row| row
                .get::<_, i64>(0))?,
        0
    );
    assert_eq!(
        store
            .writer()?
            .query_row(&format!("SELECT count(*) FROM vec_{old}"), [], |row| row
                .get::<_, i64>(0))?,
        2
    );
    let (paused, reason): (i64, String) = store.writer()?.query_row(
        "SELECT not_before,last_error FROM jobs WHERE provider=?1",
        [&runner.provider],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    assert_eq!(paused, queue::capacity::PAUSED);
    assert_eq!(reason, queue::capacity::Reason::Wal.message());
    // A new Host::run is an explicit capacity resume. Ordinary claiming must
    // leave the paused job untouched until such a resume is requested.
    assert!(
        queue::claim(
            &store,
            &runner.provider,
            &queue::Owner::new(&store)?,
            2,
            1,
            100
        )?
        .is_empty()
    );
    assert_eq!(docs.received_requests().await.ok_or("requests")?.len(), 1);
    store::wal::checkpoint(&store)?;
    store
        .writer()?
        .execute_batch("DROP TRIGGER transition_overflow")?;
    assert_eq!(queue::capacity::resume(&store, &runner.provider, || 3)?, 1);
    let resumed = runner.run(Vec::new(), || 3).await?;
    assert_eq!(resumed.execution.published, 1);
    assert!(resumed.execution.errors.is_empty());
    assert!(resumed.transition.ok_or("transition")?.previous.is_empty());
    assert_eq!(store.read_snapshot(observe)??, before);
    assert!(store::wal::wal_bytes(&*store.writer()?)? <= 8 * 1024 * 1024);
    assert_eq!(docs.received_requests().await.ok_or("requests")?.len(), 2);
    Ok(())
}
