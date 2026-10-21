//! Host table preparation, finite execution and durable scheduling progress.
mod transitions;
use super::*;
use crate::{config, queue};
use std::{error::Error as StdError, fs};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

fn bounds() -> Bounds {
    Bounds {
        budget: usage::Budget {
            calls: None,
            tokens: None,
        },
        timeout: Duration::from_secs(2),
        heartbeat: Duration::from_secs(1),
        max_schedule: 8,
        max_jobs: 1,
    }
}

#[tokio::test]
async fn configured_host_bounds_preserve_unlimited_and_zero_daily_budgets()
-> Result<(), Box<dyn StdError>> {
    for (calls, tokens, expected) in [(None, None, 1), (Some(0), None, 0), (None, Some(0), 0)] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]]})),
            )
            .expect(expected)
            .mount(&server)
            .await;
        let (root, store, mut config) = fixture(&server).await?;
        config.queue.max_schedule_per_pass = 1;
        config.queue.max_jobs_per_pass = 1;
        config.queue.budgets.push(config::queue::EndpointBudget {
            profile: "ollama".into(),
            url: format!("{}/", server.uri()),
            daily_calls: calls,
            daily_tokens: tokens,
        });
        fs::write(root.path().join(".deepkit/config.toml"), config.dump()?)?;
        let config = config::load(root.path(), None)?.config;
        let provider = destination::resolve(&config, UnitKind::Chunk)?
            .identity
            .provider;
        let settings = Bounds::for_provider(&config.queue, &provider)?;
        assert_eq!(settings.budget.calls, calls);
        assert_eq!(settings.budget.tokens, tokens);
        assert_eq!(settings.timeout, Duration::from_secs(60));
        let host = Host::prepare(
            Arc::clone(&store),
            Settings {
                root: root.path().into(),
                config_root: root.path().into(),
                config,
                kind: UnitKind::Chunk,
                local_cache: root.path().join("unused-cache"),
                bounds: settings,
            },
            0,
        )
        .await?;
        // The configured scheduling bound is enforced before any job is inserted.
        assert!(host.run(vec![target(1), target(999)], || 0).await.is_err());
        assert_eq!(store.count_vectors()?, 0);
        assert!(
            server
                .received_requests()
                .await
                .ok_or("requests")?
                .is_empty()
        );
        let report = host.run(vec![target(1)], || 0).await?;
        assert_eq!(report.execution.published, expected as usize);
        assert_eq!(store.count_vectors()?, expected);
        assert_eq!(
            server.received_requests().await.ok_or("requests")?.len(),
            expected as usize
        );
    }
    for budget in [
        usage::Budget {
            calls: Some(-1),
            tokens: None,
        },
        usage::Budget {
            calls: None,
            tokens: Some(-1),
        },
    ] {
        assert!(Bounds::from_config(&config::Queue::default(), budget).is_err());
    }
    Ok(())
}
fn target(unit: i64) -> Target {
    Target {
        unit,
        rank_pct: 1.0,
        recency: Recency::ChangedFile,
    }
}
async fn fixture(
    server: &MockServer,
) -> Result<(tempfile::TempDir, Arc<Store>, Config), Box<dyn StdError>> {
    let (root, store, _) = queue::source::tests::fixture()?;
    fs::write(
        root.path().join(".deepkit/config.toml"),
        format!(
            "[index]\nrepo_name='fixture'\n[embedder]\nprofile='ollama'\ndims=2\n[embedder.ollama]\nurl='{}'\n",
            server.uri()
        ),
    )?;
    let config = config::load(root.path(), None)?.config;
    Ok((root, Arc::new(store), config))
}
async fn host(root: &std::path::Path, store: Arc<Store>, config: Config) -> Result<Host, Error> {
    Host::prepare(
        store,
        Settings {
            root: root.into(),
            config_root: root.into(),
            config,
            kind: UnitKind::Chunk,
            local_cache: root.join("unused-cache"),
            bounds: bounds(),
        },
        0,
    )
    .await
}
#[tokio::test]
async fn host_prepares_destination_and_publishes_once_without_touching_unused_tier()
-> Result<(), Box<dyn StdError>> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]]})),
        )
        .expect(1)
        .mount(&server)
        .await;
    let (root, store, config) = fixture(&server).await?;
    let host = host(root.path(), Arc::clone(&store), config.clone()).await?;
    assert_eq!(store.vector_tables()?.len(), 1);
    assert_eq!(store.vector_tables()?[0].slug, host.table_slug);
    assert_eq!(server.received_requests().await.ok_or("requests")?.len(), 0);
    let report = host.run(vec![target(1), target(999)], || 0).await?;
    assert_eq!(report.visited, 2);
    assert_eq!(report.scheduled_or_existing, 1);
    assert_eq!(report.skipped, 1);
    assert_eq!(report.execution.published, 1);
    assert_eq!(store.count_vectors()?, 1);
    let again = host.run(vec![target(1)], || 1).await?;
    assert_eq!(again.execution.claimed, 0);
    let rebuilt = Host::prepare(
        Arc::clone(&store),
        Settings {
            root: root.path().into(),
            config_root: root.path().into(),
            config,
            kind: UnitKind::Chunk,
            local_cache: root.path().join("unused-cache"),
            bounds: bounds(),
        },
        1,
    )
    .await?;
    assert_eq!(rebuilt.table_slug, host.table_slug);
    assert_eq!(store.vector_tables()?.len(), 1);
    let path = root.path().join(".deepkit/config.toml");
    let mut text = fs::read_to_string(&path)?;
    text.push_str("model='alternate-model'\n");
    fs::write(&path, text)?;
    let alternate = config::load(root.path(), None)?.config;
    let changed = Host::prepare(
        Arc::clone(&store),
        Settings {
            root: root.path().into(),
            config_root: root.path().into(),
            config: alternate,
            kind: UnitKind::Chunk,
            local_cache: root.path().join("unused-cache"),
            bounds: bounds(),
        },
        2,
    )
    .await?;
    assert_ne!(changed.table_slug, host.table_slug);
    assert_eq!(store.vector_tables()?.len(), 2);
    assert_eq!(
        store.count_vectors()?,
        1,
        "preparing a new model must retain old coverage"
    );

    Ok(())
}
#[tokio::test]
async fn host_rejects_unbounded_input_and_reports_committed_schedule_prefix()
-> Result<(), Box<dyn StdError>> {
    let server = MockServer::start().await;
    let (root, store, config) = fixture(&server).await?;
    let host = host(root.path(), Arc::clone(&store), config).await?;
    let error = host.run(vec![target(1); 9], || 0).await.unwrap_err();
    assert_eq!(error.visited, 0);
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT count(*) FROM jobs", [], |r| r.get::<_, i64>(0))?,
        0
    );
    let error = host
        .run(
            vec![
                target(1),
                Target {
                    rank_pct: f64::NAN,
                    ..target(1)
                },
            ],
            || 0,
        )
        .await
        .unwrap_err();
    assert_eq!(error.visited, 1);
    assert_eq!(error.scheduled_or_existing, 1);
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT state FROM jobs", [], |r| r.get::<_, String>(0))?,
        "queued"
    );
    assert_eq!(server.received_requests().await.ok_or("requests")?.len(), 0);
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]]})),
        )
        .expect(1)
        .mount(&server)
        .await;
    assert_eq!(host.run(Vec::new(), || 1).await?.execution.published, 1);
    Ok(())
}
#[test]
fn table_identity_distinguishes_sanitized_names_and_dimensions() {
    assert_ne!(
        table_slug("remote", "a/b", 1024),
        table_slug("remote", "a_b", 1024)
    );
    assert_ne!(
        table_slug("remote", "model", 1024),
        table_slug("remote", "model", 256)
    );
    assert_ne!(
        table_slug("remote", "model", 1024),
        table_slug("ollama", "model", 1024)
    );
    assert!(
        crate::store::vectors::vector_table_name(&table_slug(
            "remote",
            "模型; DROP TABLE files",
            1024
        ))
        .is_ok()
    );
}

#[tokio::test]
async fn changed_storage_limits_refuse_before_table_preparation() -> Result<(), Box<dyn StdError>> {
    let server = MockServer::start().await;
    let (root, store, _) = fixture(&server).await?;
    for limit in ["db_max_bytes", "wal_max_bytes"] {
        fs::write(
            root.path().join(".deepkit/config.toml"),
            format!("[embedder]\nprofile='ollama'\n[limits]\n{limit}='1 B'\n"),
        )?;
        let config = config::load(root.path(), None)?.config;
        assert!(host(root.path(), Arc::clone(&store), config).await.is_err());
        assert!(store.vector_tables()?.is_empty());
    }
    Ok(())
}

#[tokio::test]
async fn catalogue_page_replays_a_committed_prefix_after_scheduling_failure()
-> Result<(), Box<dyn StdError>> {
    let server = MockServer::start().await;
    let (root, store, config) = fixture(&server).await?;
    store.writer()?.execute_batch("INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,created_at,updated_at) SELECT 2,randomblob(32),content_hash,source_key,file_id,kind,header,0,0 FROM units WHERE id=1; UPDATE meta SET value='0' WHERE key='graph_pending'")?;
    let mut limits = bounds();
    limits.max_jobs = 2;
    let host = Host::prepare(
        Arc::clone(&store),
        Settings {
            root: root.path().into(),
            config_root: root.path().into(),
            config,
            kind: UnitKind::Chunk,
            local_cache: root.path().join("unused-cache"),
            bounds: limits,
        },
        0,
    )
    .await?;
    let mut cursor = host.start_catalogue().await?;
    let before = cursor.clone();
    let failure = host
        .run_catalogue(
            &mut cursor,
            |candidate| {
                Ok(if candidate.unit == 2 {
                    Recency::Commit { age_days: f64::NAN }
                } else {
                    Recency::ChangedFile
                })
            },
            || 0,
        )
        .await
        .unwrap_err();
    assert_eq!(failure.visited, 1);
    assert_eq!(failure.scheduled_or_existing, 1);
    assert_eq!(cursor, before);
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT count(*) FROM jobs", [], |r| r.get::<_, i64>(0))?,
        1
    );
    assert_eq!(server.received_requests().await.ok_or("requests")?.len(), 0);
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(|request: &wiremock::Request| {
            let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "embeddings": vec![vec![1.0, 0.0]; body["input"].as_array().unwrap().len()]
            }))
        })
        .expect(1)
        .mount(&server)
        .await;
    let report = host
        .run_catalogue(&mut cursor, |_| Ok(Recency::ChangedFile), || 1)
        .await?;
    assert_eq!(report.execution.published, 2);
    assert!(cursor.finished());
    assert_eq!(store.count_vectors()?, 2);
    assert_eq!(
        host.run_catalogue(&mut cursor, |_| Ok(Recency::ChangedFile), || 2)
            .await?
            .execution
            .claimed,
        0
    );
    Ok(())
}

#[tokio::test]
async fn cancelled_catalogue_pass_keeps_cursor_and_recovers_the_expired_lease()
-> Result<(), Box<dyn StdError>> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(5))
                .set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]]})),
        )
        .with_priority(2)
        .expect(1)
        .mount(&server)
        .await;
    let (root, store, config) = fixture(&server).await?;
    store.set_meta("graph_pending", "0")?;
    let host = host(root.path(), Arc::clone(&store), config).await?;
    let mut cursor = host.start_catalogue().await?;
    let before = cursor.clone();
    let mut pass = Box::pin(host.run_catalogue(&mut cursor, |_| Ok(Recency::ChangedFile), || 0));
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            tokio::select! {
                result = &mut pass => panic!("pass ended before cancellation: {result:?}"),
                _ = tokio::time::sleep(Duration::from_millis(5)) => {
                    if !server.received_requests().await.unwrap().is_empty() { break; }
                }
            }
        }
    })
    .await?;
    drop(pass);
    assert_eq!(cursor, before);
    assert_eq!(store.count_vectors()?, 0);
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"embeddings":[[0.0,1.0]]})),
        )
        .with_priority(1)
        .expect(1)
        .mount(&server)
        .await;
    let resumed = host
        .run_catalogue(&mut cursor, |_| Ok(Recency::ChangedFile), || 121)
        .await?;
    assert_eq!(resumed.execution.published, 1);
    assert!(cursor.finished());
    assert_eq!(store.count_vectors()?, 1);
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT count(*) FROM jobs WHERE state='leased'", [], |r| {
                r.get::<_, i64>(0)
            })?,
        0
    );
    Ok(())
}

#[tokio::test]
async fn drain_observes_pass_limits_and_deferred_work_then_resumes() -> Result<(), Box<dyn StdError>>
{
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]]})),
        )
        .expect(2)
        .mount(&server)
        .await;
    let (root, store, config) = fixture(&server).await?;
    store.writer()?.execute_batch("INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,created_at,updated_at) SELECT 2,randomblob(32),content_hash,source_key,file_id,kind,header,0,0 FROM units WHERE id=1")?;
    let host = host(root.path(), Arc::clone(&store), config).await?;
    for unit in [1, 2] {
        host.factory
            .schedule_file(unit, 1.0, Recency::ChangedFile, || 0)?;
    }
    assert!(host.drain(0, || 0).await.is_err());
    let first = host.drain(1, || 0).await?;
    assert_eq!(first.stopped, DrainStop::PassLimit);
    assert_eq!(first.pending, Some(1));
    assert_eq!(first.passes[0].execution.published, 1);
    store
        .writer()?
        .execute("UPDATE jobs SET not_before=100 WHERE state='queued'", [])?;
    let waiting = host.drain(5, || 1).await?;
    assert_eq!(waiting.stopped, DrainStop::Waiting);
    assert_eq!(waiting.pending, Some(1));
    assert_eq!(waiting.passes.len(), 1);
    assert_eq!(waiting.passes[0].execution.claimed, 0);
    let resumed = host.drain(5, || 100).await?;
    assert_eq!(resumed.stopped, DrainStop::Empty);
    assert_eq!(resumed.pending, Some(0));
    assert_eq!(resumed.passes.len(), 1);
    assert_eq!(resumed.passes[0].execution.published, 1);
    assert!(host.drain(1, || 101).await?.passes.is_empty());
    assert_eq!(store.count_vectors()?, 2);
    Ok(())
}

#[tokio::test]
async fn drain_keeps_completed_pass_evidence_when_later_publication_fails()
-> Result<(), Box<dyn StdError>> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]]})),
        )
        .expect(3)
        .mount(&server)
        .await;
    let (root, store, config) = fixture(&server).await?;
    store.writer()?.execute_batch("INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,created_at,updated_at) SELECT 2,randomblob(32),content_hash,source_key,file_id,kind,header,0,0 FROM units WHERE id=1")?;
    let host = host(root.path(), Arc::clone(&store), config).await?;
    for unit in [1, 2] {
        host.factory
            .schedule_file(unit, 1.0, Recency::ChangedFile, || 0)?;
    }
    store.writer()?.execute_batch(&format!("CREATE TRIGGER refuse_second BEFORE INSERT ON vec_{} WHEN NEW.unit_id=2 BEGIN SELECT RAISE(ABORT,'fixture publication failure'); END",host.table_slug))?;
    let failed = host.drain(5, || 0).await?;
    assert_eq!(failed.stopped, DrainStop::Failed);
    assert_eq!(failed.passes.len(), 2);
    assert_eq!(failed.passes[0].execution.published, 1);
    assert_eq!(failed.passes[1].execution.errors.len(), 1);
    assert_eq!(failed.pending, Some(1));
    assert_eq!(store.count_vectors()?, 1);
    store
        .writer()?
        .execute_batch("DROP TRIGGER refuse_second")?;
    let recovered = host.drain(5, || 121).await?;
    assert_eq!(recovered.stopped, DrainStop::Empty);
    assert_eq!(recovered.passes[0].execution.published, 1);
    assert_eq!(store.count_vectors()?, 2);
    Ok(())
}

#[tokio::test]
async fn profile_replacement_at_vector_cap_moves_coverage_without_changing_l1()
-> Result<(), Box<dyn StdError>> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"embeddings":[[0.0,1.0]]})),
        )
        .expect(1)
        .mount(&server)
        .await;
    let (root, initial, _) = fixture(&server).await?;
    let path = root.path().join(".deepkit/config.toml");
    let mut text = fs::read_to_string(&path)?;
    text.push_str("[limits]\nmax_l2_units=1\n");
    fs::write(path, text)?;
    let config = config::load(root.path(), None)?.config;
    let db_path = root.path().join("transition.db");
    initial
        .writer()?
        .execute("VACUUM INTO ?1", [db_path.to_string_lossy().as_ref()])?;
    let store = Arc::new(crate::store::open(
        &db_path,
        crate::store::StoreLimits {
            max_l2_units: 1,
            ..Default::default()
        },
    )?);
    store.ensure_vector_table(VectorTableSpec {
        slug: "old_model",
        model: "old",
        profile: "local",
        dims: 2,
        created_at: 0,
        active: true,
    })?;
    let hash: Vec<u8> =
        store
            .writer()?
            .query_row("SELECT content_hash FROM units WHERE id=1", [], |r| {
                r.get(0)
            })?;
    store.insert_vector(crate::store::vectors::VectorInsert {
        table_slug: "old_model",
        unit_id: 1,
        ctx_hash: &[1; 32],
        embedding_hash: &hash,
        embedding: &[1.0, 0.0],
    })?;
    store.writer()?.execute_batch("CREATE VIRTUAL TABLE transition_fts USING fts5(text,content=''); INSERT INTO transition_fts(rowid,text) VALUES(1,'retained lexical coverage')")?;
    let host = host(root.path(), Arc::clone(&store), config).await?;
    assert_eq!(store.count_vectors()?, 1);
    assert_eq!(
        store.writer()?.query_row(
            "SELECT rowid FROM transition_fts WHERE transition_fts MATCH 'retained'",
            [],
            |r| r.get::<_, i64>(0)
        )?,
        1
    );
    let result = host.run(vec![target(1)], || 0).await?;
    assert_eq!(result.execution.published, 1);
    assert_eq!(store.count_vectors()?, 1);
    assert!(result.retirement.error.is_none());
    assert_eq!(result.retirement.retired, vec!["old_model"]);
    assert_eq!(store.vector_tables()?.len(), 1);
    assert_eq!(
        store.writer()?.query_row(
            "SELECT count(*) FROM sqlite_schema WHERE name='vec_old_model'",
            [],
            |r| r.get::<_, i64>(0)
        )?,
        0
    );
    assert_eq!(
        store.writer()?.query_row(
            &format!("SELECT unit_id FROM vec_{}", host.table_slug),
            [],
            |r| r.get::<_, i64>(0)
        )?,
        1
    );
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT content_hash FROM units WHERE id=1", [], |r| r
                .get::<_, Vec<u8>>(0))?,
        hash
    );
    assert_eq!(
        store.writer()?.query_row(
            "SELECT rowid FROM transition_fts WHERE transition_fts MATCH 'retained'",
            [],
            |r| r.get::<_, i64>(0)
        )?,
        1
    );
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT ladder_level FROM units WHERE id=1", [], |r| r
                .get::<_, i64>(0))?,
        2
    );
    Ok(())
}

#[tokio::test]
async fn retirement_protects_unselected_docs_and_reports_partial_failure()
-> Result<(), Box<dyn StdError>> {
    let server = MockServer::start().await;
    let (root, store, _) = fixture(&server).await?;
    let path = root.path().join(".deepkit/config.toml");
    let text = fs::read_to_string(&path)?;
    fs::write(
        &path,
        format!("{text}\n[embedder.docs]\nprofile='remote'\nkinds=['doc_section']\n"),
    )?;
    let config = config::load(root.path(), None)?.config;
    let docs = destination::resolve(&config, UnitKind::DocSection)?;
    store.ensure_vector_table(VectorTableSpec {
        slug: "docs_keep",
        model: &docs.identity.model,
        profile: "remote",
        dims: docs.identity.dims as usize,
        created_at: 0,
        active: true,
    })?;
    for slug in ["a_obsolete", "b_obsolete"] {
        store.ensure_vector_table(VectorTableSpec {
            slug,
            model: "old-model",
            profile: "ollama",
            dims: 2,
            created_at: 0,
            active: true,
        })?;
    }
    let host = host(root.path(), Arc::clone(&store), config).await?;
    store.writer()?.execute_batch("CREATE TRIGGER refuse_retirement BEFORE DELETE ON embedding_tables WHEN OLD.slug='b_obsolete' BEGIN SELECT RAISE(ABORT,'hold'); END")?;
    let report = host.run(Vec::new(), || 0).await?;
    assert_eq!(report.retirement.retired, vec!["a_obsolete"]);
    assert!(report.retirement.error.is_some());
    assert_eq!(report.execution.published, 0);
    let names: Vec<_> = store.vector_tables()?.into_iter().map(|v| v.slug).collect();
    assert!(names.contains(&"docs_keep".into()));
    assert!(names.contains(&"b_obsolete".into()));
    assert!(names.contains(&host.table_slug));
    store
        .writer()?
        .execute_batch("DROP TRIGGER refuse_retirement")?;
    // Reloaded configuration no longer agrees with the prepared host: keep old tables.
    fs::write(&path, text)?;
    let stale = host.run(Vec::new(), || 0).await?;
    assert!(stale.retirement.retired.is_empty());
    assert!(stale.retirement.error.is_some());
    assert!(
        store
            .vector_tables()?
            .iter()
            .any(|t| t.slug == "b_obsolete")
    );
    Ok(())
}

#[tokio::test]
async fn indexed_rename_retains_dense_coverage_and_refreshes_context_at_low_priority()
-> Result<(), Box<dyn StdError>> {
    use crate::store::{
        self, StoreLimits,
        vectors::{ContextState, VectorScanPredicate},
    };
    use std::process::Command;
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]]})),
        )
        .mount(&server)
        .await;
    let (root, _, config) = fixture(&server).await?;
    assert!(
        Command::new("git")
            .args(["init", "-q"])
            .current_dir(root.path())
            .status()?
            .success()
    );
    let index = || -> Result<(), Box<dyn StdError>> {
        let output = Command::new(assert_cmd::cargo::cargo_bin("dkc"))
            .args(["index", "--level", "1", "--json"])
            .current_dir(root.path())
            .output()?;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(())
    };
    index()?;
    let store = Arc::new(store::open(
        root.path().join(".deepkit/code.db"),
        StoreLimits::default(),
    )?);
    let unit = store.writer()?.query_row(
        "SELECT id FROM units WHERE kind IN ('chunk','symbol') ORDER BY id LIMIT 1",
        [],
        |r| r.get::<_, i64>(0),
    )?;
    let host = host(root.path(), Arc::clone(&store), config).await?;
    assert_eq!(
        host.factory.context_state(unit)?,
        Some(ContextState::Missing)
    );
    assert_eq!(
        host.run(vec![target(unit)], || 0)
            .await?
            .execution
            .published,
        1
    );
    assert_eq!(
        host.factory.context_state(unit)?,
        Some(ContextState::Current)
    );
    assert_eq!(
        host.run(vec![target(unit)], || 1).await?.execution.claimed,
        0
    );
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT count(*) FROM jobs", [], |r| r.get::<_, i64>(0))?,
        1
    );
    fs::rename(root.path().join("a.rs"), root.path().join("moved.rs"))?;
    index()?;
    assert_eq!(store.count_vectors()?, 1);
    assert_eq!(host.factory.context_state(unit)?, Some(ContextState::Stale));
    let id = host
        .factory
        .schedule_file(unit, 1.0, Recency::ChangedFile, || 2)?
        .ok_or("missing refresh job")?;
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT priority FROM jobs WHERE id=?1", [id], |r| r
                .get::<_, f64>(0))?,
        0.0
    );
    assert_eq!(
        store.scan_vectors(
            &host.table_slug,
            &[1.0, 0.0],
            1,
            VectorScanPredicate::default()
        )?[0]
            .unit_id,
        unit
    );
    assert_eq!(host.run(Vec::new(), || 2).await?.execution.published, 1);
    assert_eq!(
        host.factory.context_state(unit)?,
        Some(ContextState::Current)
    );
    assert_eq!(store.count_vectors()?, 1);
    assert_eq!(
        host.run(vec![target(unit)], || 3).await?.execution.claimed,
        0
    );
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT count(*) FROM jobs", [], |r| r.get::<_, i64>(0))?,
        2
    );
    // A policy change must stop both stale observation and scheduling before HTTP.
    let path = root.path().join(".deepkit/config.toml");
    let text = fs::read_to_string(&path)?;
    fs::write(
        &path,
        format!(
            "{text}\n[[treatments]]\nname='protected'\nmatch=['*.rs']\nmax_level=3\nnever_send=true\n"
        ),
    )?;
    index()?;
    let protected_config = config::load(root.path(), None)?.config;
    let protected = Host::prepare(
        Arc::clone(&store),
        Settings {
            root: root.path().into(),
            config_root: root.path().into(),
            config: protected_config,
            kind: UnitKind::Chunk,
            local_cache: root.path().join("unused-cache"),
            bounds: bounds(),
        },
        4,
    )
    .await?;
    assert_eq!(protected.factory.context_state(unit)?, None);
    assert!(
        protected
            .factory
            .schedule_file(unit, 1.0, Recency::ChangedFile, || 4)?
            .is_none()
    );
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT count(*) FROM jobs", [], |r| r.get::<_, i64>(0))?,
        2
    );
    assert_eq!(
        server
            .received_requests()
            .await
            .ok_or("missing request log")?
            .len(),
        2
    );
    // Status must retain historical endpoint identity after configuration changes.
    fs::write(
        &path,
        fs::read_to_string(&path)?.replace(&server.uri(), "http://127.0.0.1:9"),
    )?;
    let output = Command::new(assert_cmd::cargo::cargo_bin("dkc"))
        .args(["status", "--json"])
        .current_dir(root.path())
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let status: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(
        status["endpoints"][0]["url"],
        format!("{}/api/embed", server.uri())
    );
    assert_eq!(status["endpoints"][0]["sent_units"], 2);
    assert_eq!(status["l2"]["physical_vectors"], 1);
    assert_eq!(status["l2"]["covered_units"], 1);
    assert_eq!(status["l2_pending"], 0);
    Ok(())
}

#[tokio::test]
async fn host_preserves_provider_identity_and_allocates_a_distinct_endpoint_destination()
-> Result<(), Box<dyn StdError>> {
    let first = MockServer::start().await;
    let second = MockServer::start().await;
    let (root, store, config) = fixture(&first).await?;
    let initial = host(root.path(), Arc::clone(&store), config.clone()).await?;
    assert_eq!(
        initial.binding_state,
        crate::store::embedding_bindings::State::Known
    );
    let prior = store
        .read_snapshot(|tx| crate::store::embedding_bindings::read(tx, &initial.table_slug))??
        .ok_or("binding")?;
    assert_eq!(prior.endpoint, format!("{}/api/embed", first.uri()));
    assert!(prior.api_key_env.is_none());
    let mut changed = config.clone();
    changed.embedder.ollama.url = second.uri();
    fs::write(
        root.path().join(".deepkit/config.toml"),
        toml::to_string(&changed)?,
    )?;
    let replacement = host(root.path(), Arc::clone(&store), changed.clone()).await?;
    assert_ne!(initial.table_slug, replacement.table_slug);
    assert_eq!(
        replacement.binding_state,
        crate::store::embedding_bindings::State::Known
    );
    // Direct queue factories must enforce the same provenance fence, even when
    // their caller bypasses Host preparation and current source/config is valid.
    fs::write(
        root.path().join(".deepkit/config.toml"),
        toml::to_string(&changed)?,
    )?;
    let direct = http_embedding::Factory::new(
        Arc::clone(&store),
        http_embedding::Settings {
            root: root.path().into(),
            config_root: root.path().into(),
            config: changed.clone(),
            kind: UnitKind::Chunk,
            table_slug: initial.table_slug.clone(),
            budget: usage::Budget {
                calls: Some(1),
                tokens: None,
            },
            timeout: Duration::from_secs(1),
        },
    )?;
    assert!(matches!(
        direct.schedule_file(1, 1.0, Recency::ChangedFile, || 0),
        Err(Error::Invalid(
            "embedding provider differs from table binding"
        ))
    ));
    let query = crate::search::embedding_query::Factory::new(
        Arc::clone(&store),
        crate::search::embedding_query::Settings {
            config_root: root.path().into(),
            config: changed,
            kind: UnitKind::Chunk,
            budget: usage::Budget {
                calls: Some(1),
                tokens: None,
            },
        },
    )?;
    assert!(!store.read_snapshot(|tx| query.table_matches(tx, &initial.table_slug))??);
    let current = store
        .read_snapshot(|tx| crate::store::embedding_bindings::read(tx, &initial.table_slug))??
        .ok_or("binding")?;
    assert!(prior == current);
    assert_eq!(store.vector_tables()?.len(), 2);
    assert!(
        first
            .received_requests()
            .await
            .ok_or("requests")?
            .is_empty()
    );
    assert!(
        second
            .received_requests()
            .await
            .ok_or("requests")?
            .is_empty()
    );
    Ok(())
}

#[tokio::test]
async fn endpoint_replacement_preserves_full_cap_coverage_and_retires_only_old_empty_space()
-> Result<(), Box<dyn StdError>> {
    let first = MockServer::start().await;
    let second = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]]})),
        )
        .expect(1)
        .mount(&first)
        .await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"embeddings":[[0.0,1.0]]})),
        )
        .expect(2)
        .mount(&second)
        .await;
    let (root, initial, mut config) = fixture(&first).await?;
    let db_path = root.path().join(".deepkit/endpoint.db");
    initial
        .writer()?
        .execute("VACUUM INTO ?1", [db_path.to_string_lossy().as_ref()])?;
    let store = Arc::new(crate::store::open(
        &db_path,
        crate::store::StoreLimits {
            max_l2_units: 1,
            ..Default::default()
        },
    )?);
    config.limits.max_l2_units = 1;
    fs::write(
        root.path().join(".deepkit/config.toml"),
        toml::to_string(&config)?,
    )?;
    store.writer()?.execute("INSERT INTO fts_units(rowid,header,idents,body,prose) VALUES(1,'example','','fn example() {}','')",[])?;
    let original = host(root.path(), Arc::clone(&store), config.clone()).await?;
    assert_eq!(
        original
            .run(vec![target(1)], || 0)
            .await?
            .execution
            .published,
        1
    );
    assert_eq!(store.count_vectors()?, 1);
    assert_eq!(store.query_fts("example", 10)?.len(), 1);
    config.embedder.ollama.url = second.uri();
    fs::write(
        root.path().join(".deepkit/config.toml"),
        toml::to_string(&config)?,
    )?;
    let replacement = host(root.path(), Arc::clone(&store), config.clone()).await?;
    assert_ne!(original.table_slug, replacement.table_slug);
    assert_eq!(store.vector_tables()?.len(), 2);
    assert_eq!(store.count_vectors()?, 1); // Old coverage survives empty destination preparation.
    assert_eq!(store.query_fts("example", 10)?.len(), 1);
    assert_eq!(replacement.transition.previous.len(), 1);
    let third = MockServer::start().await;
    let mut next = config.clone();
    next.embedder.ollama.url = third.uri();
    fs::write(
        root.path().join(".deepkit/config.toml"),
        toml::to_string(&next)?,
    )?;
    assert!(matches!(
        host(root.path(), Arc::clone(&store), next.clone()).await,
        Err(Error::Store(
            crate::store::Error::EmbeddingTransition { .. }
        ))
    ));
    assert_eq!(store.vector_tables()?.len(), 2); // Rejected before allocating a third table.
    assert_eq!(store.count_vectors()?, 1);
    fs::write(
        root.path().join(".deepkit/config.toml"),
        toml::to_string(&config)?,
    )?;
    let report = replacement.run(vec![target(1)], || 1).await?;
    assert!(report.transition_error.is_none());
    assert!(
        report
            .transition
            .as_ref()
            .ok_or("transition")?
            .previous
            .is_empty()
    );
    assert_eq!(report.execution.published, 1);
    assert!(report.retirement.error.is_none());
    assert_eq!(report.retirement.retired, vec![original.table_slug.clone()]);
    assert_eq!(store.vector_tables()?.len(), 1);
    assert_eq!(store.vector_tables()?[0].slug, replacement.table_slug);
    assert_eq!(store.count_vectors()?, 1);
    assert_eq!(store.query_fts("example", 10)?.len(), 1);
    assert!(
        store
            .read_snapshot(|tx| crate::store::embedding_bindings::read(tx, &original.table_slug))??
            .is_none()
    );
    let query = crate::search::embedding_query::Factory::new(
        Arc::clone(&store),
        crate::search::embedding_query::Settings {
            config_root: root.path().into(),
            config: config.clone(),
            kind: UnitKind::Chunk,
            budget: usage::Budget {
                calls: Some(10),
                tokens: None,
            },
        },
    )?;
    assert_eq!(query.table_slug(), replacement.table_slug);
    let output = crate::search::pipeline::search_without_rerank(
        Arc::clone(&store),
        config,
        crate::search::pipeline::Request {
            query: "example".into(),
            scope: None,
            k: 1,
            mode: crate::search::fusion::Mode::Dense,
            exclude_ids: Vec::new(),
        },
        Some(query),
        || 2,
    )
    .await?;
    assert!(!output.degraded());
    assert_eq!(output.hits[0].unit.id, 1);
    fs::write(
        root.path().join(".deepkit/config.toml"),
        toml::to_string(&next)?,
    )?;
    let accepted = host(root.path(), Arc::clone(&store), next).await?;
    assert!(accepted.transition.generation > replacement.transition.generation);
    assert_eq!(accepted.transition.previous.len(), 1);
    assert!(
        third
            .received_requests()
            .await
            .ok_or("requests")?
            .is_empty()
    );
    Ok(())
}

#[tokio::test]
async fn returning_to_completed_configuration_rebuilds_retired_output()
-> Result<(), Box<dyn StdError>> {
    let first = MockServer::start().await;
    let second = MockServer::start().await;
    for server in [&first, &second] {
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]]})),
            )
            .mount(server)
            .await;
    }
    let (root, initial, mut a) = fixture(&first).await?;
    let db_path = root.path().join(".deepkit/return.db");
    initial
        .writer()?
        .execute("VACUUM INTO ?1", [db_path.to_string_lossy().as_ref()])?;
    let store = Arc::new(crate::store::open(
        &db_path,
        crate::store::StoreLimits {
            max_l2_units: 1,
            ..Default::default()
        },
    )?);
    a.limits.max_l2_units = 1;
    let mut b = a.clone();
    b.embedder.ollama.url = second.uri();
    for (stage, config) in [a.clone(), b, a.clone()].into_iter().enumerate() {
        fs::write(
            root.path().join(".deepkit/config.toml"),
            toml::to_string(&config)?,
        )?;
        let runner = host(root.path(), Arc::clone(&store), config).await?;
        let report = runner.run(vec![target(1)], move || stage as i64).await?;
        assert_eq!(report.execution.published, 1, "stage {stage}");
        assert!(report.transition_error.is_none());
        assert!(report.transition.ok_or("transition")?.previous.is_empty());
        assert_eq!(store.count_vectors()?, 1);
        assert_eq!(store.vector_tables()?.len(), 1);
    }
    let again = host(root.path(), Arc::clone(&store), a).await?;
    assert_eq!(again.run(vec![target(1)], || 3).await?.execution.claimed, 0);
    assert_eq!(first.received_requests().await.ok_or("requests")?.len(), 2);
    assert_eq!(second.received_requests().await.ok_or("requests")?.len(), 1);
    let jobs: (i64, i64) =
        store
            .writer()?
            .query_row("SELECT COUNT(*),SUM(lease_generation) FROM jobs", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })?;
    assert_eq!(jobs, (2, 3));
    Ok(())
}

#[tokio::test]
async fn same_context_generations_require_exact_publication_and_manual_writes_clear_proof()
-> Result<(), Box<dyn StdError>> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]]})),
        )
        .expect(5)
        .mount(&server)
        .await;
    let (root, store, a) = fixture(&server).await?;
    let mut b = a.clone();
    b.embedder.max_doc_tokens /= 2;
    let mut identities = Vec::new();
    for (stage, config) in [a.clone(), b, a.clone()].into_iter().enumerate() {
        fs::write(
            root.path().join(".deepkit/config.toml"),
            toml::to_string(&config)?,
        )?;
        let runner = host(root.path(), Arc::clone(&store), config).await?;
        assert_eq!(
            runner
                .run(vec![target(1)], move || stage as i64)
                .await?
                .execution
                .published,
            1
        );
        let identity: (Vec<u8>, Vec<u8>) = store.writer()?.query_row(
            &format!("SELECT v.ctx_hash,p.desired_hash FROM vec_{} v JOIN embedding_publications p USING(unit_id)",runner.table_slug),
            [], |r| Ok((r.get(0)?,r.get(1)?)))?;
        identities.push(identity);
    }
    assert_eq!(identities[0].0, identities[1].0);
    assert_ne!(identities[0].1, identities[1].1);
    assert_eq!(identities[0], identities[2]);
    let runner = host(root.path(), Arc::clone(&store), a).await?;
    let hash: Vec<u8> =
        store
            .writer()?
            .query_row("SELECT content_hash FROM units WHERE id=1", [], |r| {
                r.get(0)
            })?;
    store.insert_vector(crate::store::vectors::VectorInsert {
        table_slug: &runner.table_slug,
        unit_id: 1,
        ctx_hash: &identities[0].0,
        embedding_hash: &hash,
        embedding: &[0.0, 1.0],
    })?;
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT COUNT(*) FROM embedding_publications", [], |r| r
                .get::<_, i64>(0))?,
        0
    );
    assert_eq!(
        runner.run(vec![target(1)], || 3).await?.execution.published,
        1
    );
    // Physical loss cannot be hidden by surviving publication metadata.
    store
        .writer()?
        .execute(&format!("DELETE FROM vec_{}", runner.table_slug), [])?;
    assert_eq!(
        runner.run(vec![target(1)], || 4).await?.execution.published,
        1
    );
    assert_eq!(
        runner.run(vec![target(1)], || 5).await?.execution.claimed,
        0
    );
    let usage: (i64, i64) = store.writer()?.query_row(
        "SELECT COUNT(*),COUNT(DISTINCT id) FROM usage_reservations WHERE state='reconciled'",
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    assert_eq!(usage, (5, 5));
    Ok(())
}

#[tokio::test]
async fn schema_thirteen_done_output_is_retained_then_readmitted_under_current_caps()
-> Result<(), Box<dyn StdError>> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]]})),
        )
        .expect(2)
        .mount(&server)
        .await;
    let (root, initial, config) = fixture(&server).await?;
    let db = root.path().join(".deepkit/upgrade.db");
    initial
        .writer()?
        .execute("VACUUM INTO ?1", [db.to_string_lossy().as_ref()])?;
    let store = Arc::new(crate::store::open(&db, Default::default())?);
    let runner = host(root.path(), Arc::clone(&store), config.clone()).await?;
    assert_eq!(
        runner.run(vec![target(1)], || 0).await?.execution.published,
        1
    );
    let slug = runner.table_slug.clone();
    drop(runner);
    store.writer()?.execute_batch(
        "DROP TABLE embedding_publications; UPDATE meta SET value='13' WHERE key='schema_version'",
    )?;
    crate::store::migration_tests::drop_l1_delete_indexes(&*store.writer()?);
    drop(store);
    let store = Arc::new(crate::store::open(&db, Default::default())?);
    assert_eq!(store.count_vectors()?, 1);
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT COUNT(*) FROM embedding_publications", [], |r| r
                .get::<_, i64>(0))?,
        0
    );
    // Losing the old vector under a zero L2 allowance revives into a capacity
    // pause, never an outbound attempt or a reset of historical accounting.
    store.delete_vector(&slug, 1)?;
    let mut paused = config.clone();
    paused.limits.max_l2_units = 0;
    fs::write(
        root.path().join(".deepkit/config.toml"),
        toml::to_string(&paused)?,
    )?;
    let runner = host(root.path(), Arc::clone(&store), paused).await?;
    runner
        .factory
        .schedule_file(1, 1.0, Recency::ChangedFile, || 1)?;
    let state: (String, i64, i64) = store.writer()?.query_row(
        "SELECT state,not_before,lease_generation FROM jobs",
        [],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    )?;
    assert_eq!(state, ("queued".into(), queue::capacity::PAUSED, 1));
    // Explicit passes keep an unchanged zero-cap pause without another claim.
    for _ in 0..3 {
        assert_eq!(
            runner.run(vec![target(1)], || 1).await?.execution.published,
            0
        );
    }
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT lease_generation FROM jobs", [], |r| r
                .get::<_, i64>(0))?,
        1
    );
    assert_eq!(server.received_requests().await.ok_or("requests")?.len(), 1);
    fs::write(
        root.path().join(".deepkit/config.toml"),
        toml::to_string(&config)?,
    )?;
    let runner = host(root.path(), Arc::clone(&store), config).await?;
    assert_eq!(
        runner.run(vec![target(1)], || 2).await?.execution.published,
        1
    );
    assert_eq!(store.count_vectors()?, 1);
    Ok(())
}

#[tokio::test]
async fn retained_vector_pause_never_authorizes_changed_or_protected_source_after_cap_recovery()
-> Result<(), Box<dyn StdError>> {
    for change in ["source", "privacy"] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500))
            .expect(0)
            .mount(&server)
            .await;
        let (root, store, config) = fixture(&server).await?;
        let config_path = root.path().join(".deepkit/config.toml");
        let original = toml::to_string(&config)?;
        let mut capped = config.clone();
        capped.limits.max_l2_units = 0;
        fs::write(&config_path, toml::to_string(&capped)?)?;
        let runner = host(root.path(), Arc::clone(&store), capped).await?;
        assert!(
            runner
                .factory
                .schedule_file(1, 1.0, Recency::ChangedFile, || 0)?
                .is_some()
        );
        if change == "source" {
            let file: String = store.writer()?.query_row(
                "SELECT f.path FROM units u JOIN files f ON f.id=u.file_id WHERE u.id=1",
                [],
                |r| r.get(0),
            )?;
            fs::write(root.path().join(file), "pub fn changed_after_pause() {}\n")?;
        }
        // The negative shortcut performs no preparation, lease or publication.
        for _ in 0..3 {
            let report = runner.run(vec![target(1)], || 1).await?;
            assert_eq!(
                (report.scheduled_or_existing, report.execution.claimed),
                (0, 0)
            );
        }
        let recovered = if change == "privacy" {
            format!(
                "{original}\n[[treatments]]\nname='protected'\nmatch=['*.rs']\nmax_level=3\nnever_send=true\n"
            )
        } else {
            original
        };
        fs::write(&config_path, recovered)?;
        let config = config::load(root.path(), None)?.config;
        let runner = host(root.path(), Arc::clone(&store), config).await?;
        assert_eq!(
            runner.run(vec![target(1)], || 2).await?.execution.published,
            0
        );
        assert_eq!(store.count_vectors()?, 0);
        assert!(
            server
                .received_requests()
                .await
                .ok_or("requests")?
                .is_empty()
        );
    }
    Ok(())
}
