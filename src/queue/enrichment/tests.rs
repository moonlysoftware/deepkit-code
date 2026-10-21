//! Real chat transport, worker fences and summary publication on verified sources.
use super::*;
use crate::{
    queue,
    store::FileKind,
    sync,
    treat::{Router, context},
};
use rusqlite::params;
use serde_json::json;
use std::{fs, time::Duration};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

fn text_input(job: &Job) -> &enrichment_inputs::Text {
    match job.input.as_ref() {
        Input::Text(input) => input,
        Input::Image(_) | Input::Page(_) => panic!("text fixture prepared a visual input"),
    }
}

fn fixture(url: &str) -> (tempfile::TempDir, Arc<Context>) {
    let (root, store, mut config) = queue::source::tests::fixture().unwrap();
    config.llm.enabled = true;
    config.llm.url = format!("{url}/v1");
    config.llm.model = "fixture-text".into();
    config.treatments = vec![
        crate::config::Treatment {
            name: "private".into(),
            patterns: vec!["private.rs".into()],
            max_level: 1,
            never_send: true,
            ..Default::default()
        },
        crate::config::Treatment {
            name: "source".into(),
            patterns: vec!["**".into()],
            max_level: 3,
            enrich: vec!["describe_symbol".into()],
            ..Default::default()
        },
    ];
    fs::write(
        root.path().join(".deepkit/config.toml"),
        toml::to_string(&config).unwrap(),
    )
    .unwrap();
    let config = crate::config::load(root.path(), None).unwrap().config;
    let treatment = Router::new(&config.treatments, true)
        .unwrap()
        .route("a.rs", FileKind::Code);
    let hash = *blake3::hash(b"fn example() {}").as_bytes();
    let fingerprint = sync::l1::file_fingerprint(
        &config,
        "fixture",
        sync::l1::FingerprintFile {
            path: "a.rs",
            content_hash: &hash,
            language: Some("rust"),
            kind: FileKind::Code,
            capped: false,
        },
        &treatment,
    );
    store
        .writer()
        .unwrap()
        .execute(
            "UPDATE files SET l1_desired=?1,l1_fingerprint=?1,l1_pending=0",
            [fingerprint.as_slice()],
        )
        .unwrap();
    store.writer().unwrap().execute_batch("INSERT INTO unit_sources(unit_id,source_kind,source_id,role) VALUES(1,'file',1,'content');
        INSERT INTO symbols(id,file_id,unit_id,name,signature,kind,line_start,line_end) VALUES(1,1,1,'example','fn example()','function',1,1);").unwrap();
    let client =
        chat::Client::with_key(&config.llm, Duration::from_secs(3), 1024, "fixture".into())
            .unwrap();
    let env = Arc::new(Context {
        store: Arc::new(store),
        root: root.path().to_owned(),
        config_root: root.path().to_owned(),
        config: Arc::new(config),
        client: Arc::new(client),
    });
    (root, env)
}
fn verify_summary_input(
    env: &Context,
    owner: crate::store::enrichment::publication::Owner,
    parent: i64,
    job: &Job,
) {
    let summary = env
        .store
        .read_snapshot(|tx| {
            owner.matching_summary(tx, parent, job.input.target_hash(), &job.desired)
        })
        .unwrap()
        .unwrap()
        .unwrap();
    let prepared = env
        .store
        .read_snapshot(|tx| {
            queue::destination::prepare_file(tx, &env.root, &env.config_root, &env.config, summary)
        })
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(prepared.0.enqueue(1.0).target_id, summary);
    let mut cursor = queue::catalogue::Cursor::start(&env.store).unwrap();
    let mut found = false;
    while !cursor.finished() {
        found |= cursor
            .next(&env.store, 100)
            .unwrap()
            .iter()
            .any(|candidate| {
                candidate.unit == summary && candidate.kind == crate::store::UnitKind::Summary
            });
    }
    assert!(found, "owned summary omitted from embedding catalogue");
    assert!(
        env.store
            .embedding_catalogue_kinds(
                env.config.commits.enabled,
                env.config.commits.enabled && env.config.commits.prs,
                env.config.enrich.top_symbols,
                env.config.enrich.modules_min_files
            )
            .unwrap()
            .contains(&crate::store::UnitKind::Summary),
        "provider inventory omitted the published summary destination"
    );

    let replaces = || {
        env.store
            .read_snapshot(|tx| {
                queue::summary_source::replaces_parent(
                    tx,
                    &env.root,
                    &env.config_root,
                    &env.config,
                    parent,
                )
            })
            .unwrap()
            .unwrap()
    };
    assert!(
        !replaces(),
        "summary text alone cannot retire raw vector coverage"
    );
    let binding = prepared.1.binding(&env.config);
    let slug = env.store.embedding_destination(&binding).unwrap();
    env.store
        .ensure_vector_table(crate::store::vectors::VectorTableSpec {
            slug: &slug,
            model: &binding.model,
            profile: &binding.profile,
            dims: binding.dims as usize,
            created_at: 0,
            active: true,
        })
        .unwrap();
    env.store.bind_embedding_table(&slug, &binding).unwrap();
    env.store
        .replace_vector(crate::store::vectors::VectorInsert {
            table_slug: &slug,
            unit_id: summary,
            ctx_hash: prepared.0.context_hash(),
            embedding_hash: &prepared.0.current().target_hash,
            embedding: &vec![1.0; binding.dims as usize],
        })
        .unwrap();
    assert!(
        !replaces(),
        "unattested standalone vector is not current publication"
    );
    env.store
        .writer()
        .unwrap()
        .execute(
            "INSERT INTO embedding_publications(unit_id,table_slug,desired_hash) VALUES(?1,?2,?3)",
            params![summary, slug, prepared.0.current().desired_hash.as_slice()],
        )
        .unwrap();
    let current = env
        .store
        .read_snapshot(|tx| {
            queue::destination::prepare_file(tx, &env.root, &env.config_root, &env.config, summary)
        })
        .unwrap()
        .unwrap();
    let current = current.expect("published summary lost source authority after vector insertion");
    assert_eq!(
        current.0.current().desired_hash,
        prepared.0.current().desired_hash,
        "summary embedding identity changed during publication"
    );
    if matches!(job.kind, Kind::DescribeSymbol) {
        assert!(!replaces(), "symbol summary must retain code embedding");
        return;
    }
    assert!(
        replaces(),
        "current summary publication did not suppress parent {parent}, summary {summary}, table {slug}"
    );
    let directory = env.root.join(".deepkit/prompts");
    fs::create_dir_all(&directory).unwrap();
    let path = directory.join(format!(
        "{}.md",
        crate::treat::prompts::name(job.kind).unwrap()
    ));
    let previous = fs::read(&path).ok();
    fs::write(&path, "A new summary instruction.").unwrap();
    assert!(
        !replaces(),
        "outdated summary generation cannot suppress raw embedding"
    );
    if let Some(previous) = previous {
        fs::write(path, previous).unwrap()
    } else {
        fs::remove_file(path).unwrap()
    }
}

fn add(env: &Context, id: i64, path: &str, text: &str, doc: bool) {
    fs::write(env.root.join(path), text).unwrap();
    let kind = if doc { FileKind::Doc } else { FileKind::Code };
    let unit_kind = if doc { "doc_section" } else { "chunk" };
    let language = if doc { None } else { Some("rust") };
    let treatment = Router::new(&env.config.treatments, true)
        .unwrap()
        .route(path, kind);
    let hash = *blake3::hash(text.as_bytes()).as_bytes();
    let fingerprint = sync::l1::file_fingerprint(
        &env.config,
        "fixture",
        sync::l1::FingerprintFile {
            path,
            content_hash: &hash,
            language,
            kind,
            capped: false,
        },
        &treatment,
    );
    let content = context::unit_hash(unit_kind, None, text);
    let db = env.store.writer().unwrap();
    db.execute("INSERT INTO files(id,path,blake3,size,mtime_ns,lang,kind,max_level,treatment,l1_pending,l1_desired,l1_fingerprint)
        VALUES(?1,?2,?3,?4,0,?5,?6,?7,?8,0,?9,?9)",params![id,path,hash.as_slice(),text.len() as i64,language,if doc {"doc"} else {"code"},treatment.max_level,treatment.name,fingerprint.as_slice()]).unwrap();
    db.execute(
        "INSERT INTO unit_contents(hash,text,tokens) VALUES(?1,?2,?3)",
        params![
            content.as_slice(),
            text,
            crate::treat::count_tokens(text) as i64
        ],
    )
    .unwrap();
    db.execute("INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,created_at,updated_at)
        VALUES(?1,?2,?2,?3,?1,?4,'context',0,0)",params![id,content.as_slice(),path,unit_kind]).unwrap();
    db.execute("INSERT INTO unit_sources(unit_id,source_kind,source_id,role) VALUES(?1,'file',?1,'content')",[id]).unwrap();
    db.execute("INSERT INTO symbols(id,file_id,unit_id,name,kind,line_start,line_end) VALUES(?1,?1,?1,?2,?3,1,1)",params![id,if path=="private.rs" {"private_caller".to_owned()} else {format!("caller_{id}")},if doc {"doc"} else {"function"}]).unwrap();
    db.execute("INSERT INTO edges(src_symbol,dst_symbol,dst_name,kind,file_id,line) VALUES(?1,1,'example',?2,?1,1)",params![id,if doc {"doc_mention"} else {"ref"}]).unwrap();
}
fn prepare(env: &Arc<Context>) -> Job {
    env.store
        .read_snapshot(|tx| env.symbol(tx, 1))
        .unwrap()
        .unwrap()
        .unwrap()
}
async fn execute(env: &Arc<Context>, job: &Job) -> worker::Finished {
    queue::enqueue(&env.store, &job.enqueue(1.0), 0).unwrap();
    let leases = queue::claim(
        &env.store,
        &job.provider,
        &queue::Owner::new(&env.store).unwrap(),
        0,
        1,
        job.wire.input_tokens,
    )
    .unwrap();
    assert_eq!(leases.len(), 1);
    worker::execute(
        &env.store,
        &leases[0],
        &worker::Settings {
            provider: &job.provider,
            budget: chat::limits(&env.config.llm).unwrap().budget,
            max_output_tokens: 777,
            timeout: Duration::from_secs(3),
            heartbeat: Duration::from_millis(20),
        },
        job,
        || 0,
    )
    .await
    .unwrap()
}

#[test]
fn shared_symbol_contexts_match_independent_proofs_and_reset_at_freshness_boundary() {
    let (_root, env) = fixture("http://127.0.0.1:1");
    add(
        &env,
        2,
        "private.rs",
        "fn private_caller() { example(); }",
        false,
    );
    add(&env, 3, "caller.rs", "fn caller_3() { example(); }", false);
    add(
        &env,
        4,
        "guide.md",
        "Shared documentation mentions example.",
        true,
    );
    // A second owner uses the same supplementary units, exercising actual reuse.
    add(&env, 5, "second.rs", "fn second() {}", false);
    env.store
        .writer()
        .unwrap()
        .execute_batch(
            "INSERT INTO edges(src_symbol,dst_symbol,dst_name,kind,file_id,line)
         SELECT src_symbol,5,'second',kind,file_id,line FROM edges
         WHERE dst_symbol=1 AND src_symbol IN (2,3,4);",
        )
        .unwrap();
    let prepare_pass = || {
        env.store
            .read_snapshot(|tx| -> Result<Vec<enrichment_inputs::Text>, Error> {
                let mut contexts = std::collections::BTreeMap::new();
                let mut inputs = Vec::new();
                for id in [1, 5] {
                    let (shared, _) = enrichment_inputs::symbol_with_contexts(
                        tx,
                        &env.root,
                        &env.config_root,
                        &env.config,
                        id,
                        crate::sync::l1::Work::default(),
                        &mut contexts,
                    )?
                    .unwrap();
                    let (independent, _) = enrichment_inputs::symbol_with_parent(
                        tx,
                        &env.root,
                        &env.config_root,
                        &env.config,
                        id,
                        crate::sync::l1::Work::default(),
                    )?
                    .unwrap();
                    assert_eq!(shared.prompt, independent.prompt);
                    assert_eq!(shared.fingerprint, independent.fingerprint);
                    let provenance = |input: &enrichment_inputs::Text| {
                        input
                            .sources
                            .iter()
                            .map(|source| match source {
                                crate::store::enrichment::publication::Source::Unit(id) => {
                                    ("unit", *id)
                                }
                                crate::store::enrichment::publication::Source::File(id) => {
                                    ("file", *id)
                                }
                                crate::store::enrichment::publication::Source::Symbol(id) => {
                                    ("symbol", *id)
                                }
                                crate::store::enrichment::publication::Source::Commit(id) => {
                                    ("commit", *id)
                                }
                            })
                            .collect::<Vec<_>>()
                    };
                    assert_eq!(provenance(&shared), provenance(&independent));
                    assert!(!shared.prompt.contains("private_caller"));
                    inputs.push(shared);
                }
                Ok(inputs)
            })
            .unwrap()
            .unwrap()
    };
    let before = prepare_pass();
    assert!(
        before
            .iter()
            .all(|input| input.prompt.contains("Shared documentation"))
    );
    assert!(before.iter().all(|input| {
        input.sources.iter().any(|source| {
            matches!(
                source,
                crate::store::enrichment::publication::Source::Unit(3)
            )
        })
    }));
    // A new publication/serving proof must reopen changed supplementary sources.
    fs::write(env.root.join("caller.rs"), "fn changed() {}").unwrap();
    fs::write(env.root.join("guide.md"), "Changed documentation.").unwrap();
    let after = prepare_pass();
    for (before, after) in before.iter().zip(&after) {
        assert_ne!(before.fingerprint, after.fingerprint);
        assert!(!after.prompt.contains("Shared documentation"));
        assert!(!after.sources.iter().any(|source| matches!(
            source,
            crate::store::enrichment::publication::Source::Unit(3)
        )));
    }
}

#[tokio::test]
async fn symbol_worker_sends_bounded_public_context_and_publishes_accounted_summary() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices":[{"message":{"content":"A verified symbol description."}}],
            "usage":{"prompt_tokens":30,"completion_tokens":10,"total_tokens":40}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let (_root, env) = fixture(&server.uri());
    add(
        &env,
        2,
        "private.rs",
        "fn private_caller() { example(); }",
        false,
    );
    for id in 3..=14 {
        add(
            &env,
            id,
            &format!("caller{id}.rs"),
            &format!("fn caller_{id}() {{ example(); }}"),
            false,
        );
    }
    for id in 20..=22 {
        add(
            &env,
            id,
            &format!("doc{id}.md"),
            &format!("Documentation section {id} mentions example."),
            true,
        );
    }
    let job = prepare(&env);
    assert_eq!(
        job.desired,
        enrichment_inputs::desired(&env.config, Kind::DescribeSymbol, text_input(&job)).unwrap()
    );
    assert!(!text_input(&job).prompt.contains("private_caller"));
    assert_eq!(text_input(&job).prompt.matches("caller_").count(), 9);
    assert_eq!(
        text_input(&job)
            .prompt
            .matches("Documentation section")
            .count(),
        2
    );
    assert_eq!(
        execute(&env, &job).await,
        worker::Finished::Completed(queue::Completion::Published)
    );
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["max_tokens"], 120);
    assert!(!body.to_string().contains("private_caller"));
    let db = env.store.writer().unwrap();
    let summary: i64 = db
        .query_row("SELECT summary_unit FROM symbols WHERE id=1", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(
        db.query_row(
            "SELECT parent_unit FROM units WHERE id=?1",
            [summary],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        1
    );
    assert_eq!(db.query_row("SELECT count(*) FROM unit_sources WHERE unit_id=?1 AND source_kind='symbol' AND source_id=2",[summary],|r|r.get::<_,i64>(0)).unwrap(),0);
    assert_eq!(
        db.query_row("SELECT sum(tokens) FROM usage", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        40
    );
    drop(db);
    assert!(!env.store.query_fts("verified", 10).unwrap().is_empty());
    let resolved = env
        .store
        .read_snapshot(|tx| {
            queue::source::resolve(tx, &env.root, &env.config_root, &env.config, summary)
        })
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(resolved.occurrence.kind, crate::store::UnitKind::Summary);
    assert_eq!(resolved.context.text, "A verified symbol description.");
    // Changing a supplementary source changes the prompt generation even though
    // the summary body and its parent code remain unchanged.
    fs::write(
        env.root.join("doc20.md"),
        "Changed supplementary documentation.",
    )
    .unwrap();
    assert!(
        env.store
            .read_snapshot(|tx| queue::source::resolve(
                tx,
                &env.root,
                &env.config_root,
                &env.config,
                summary
            ))
            .unwrap()
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn symbol_worker_discards_a_policy_change_after_enqueue_without_http() {
    let server = MockServer::start().await;
    let (_root, env) = fixture(&server.uri());
    let job = prepare(&env);
    queue::enqueue(&env.store, &job.enqueue(1.0), 0).unwrap();
    let mut config = (*env.config).clone();
    config.treatments[1].never_send = true;
    fs::write(
        env.root.join(".deepkit/config.toml"),
        toml::to_string(&config).unwrap(),
    )
    .unwrap();
    assert_eq!(
        execute(&env, &job).await,
        worker::Finished::Completed(queue::Completion::Discarded)
    );
    assert!(server.received_requests().await.unwrap().is_empty());
    assert_eq!(env.store.enrichment_status().unwrap().summaries, 0);
}

#[tokio::test]
async fn symbol_worker_discards_changed_source_after_http_but_records_reported_usage() {
    let server = MockServer::start().await;
    let (_root, env) = fixture(&server.uri());
    let path = env.root.join("a.rs");
    Mock::given(method("POST"))
        .respond_with(move |_: &wiremock::Request| {
            fs::write(&path, "fn changed_while_waiting() {}").unwrap();
            ResponseTemplate::new(200).set_body_json(json!({
                "choices":[{"message":{"content":"A response for the old source."}}],
                "usage":{"prompt_tokens":30,"completion_tokens":10,"total_tokens":40}
            }))
        })
        .expect(1)
        .mount(&server)
        .await;
    let job = prepare(&env);
    assert_eq!(
        execute(&env, &job).await,
        worker::Finished::Completed(queue::Completion::Discarded)
    );
    assert_eq!(env.store.enrichment_status().unwrap().summaries, 0);
    assert_eq!(
        env.store
            .writer()
            .unwrap()
            .query_row("SELECT sum(tokens) FROM usage", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        40
    );
}

#[test]
fn symbol_worker_override_changes_the_prepared_generation() {
    let (_root, env) = fixture("http://127.0.0.1:9");
    let original = prepare(&env);
    fs::create_dir_all(env.root.join(".deepkit/prompts")).unwrap();
    fs::write(
        env.root.join(".deepkit/prompts/describe_symbol.md"),
        "Custom description of {{signature}}: {{code}}",
    )
    .unwrap();
    let changed = prepare(&env);
    assert_ne!(original.desired, changed.desired);
    assert!(
        text_input(&changed)
            .prompt
            .starts_with("Custom description of fn example()")
    );
}

#[test]
fn summary_generation_distinguishes_endpoints_but_not_credentials() {
    let (_root, env) = fixture("http://127.0.0.1:9");
    let job = prepare(&env);
    let mut config = (*env.config).clone();
    assert_eq!(
        job.desired,
        enrichment_inputs::desired(&config, Kind::DescribeSymbol, text_input(&job)).unwrap()
    );
    config.llm.api_key_env = "UNSET_SUMMARY_PROOF_TEST_KEY".into();
    assert_eq!(
        job.desired,
        enrichment_inputs::desired(&config, Kind::DescribeSymbol, text_input(&job)).unwrap()
    );
    config.llm.url = "http://127.0.0.1:10/v1".into();
    assert_ne!(
        job.desired,
        enrichment_inputs::desired(&config, Kind::DescribeSymbol, text_input(&job)).unwrap()
    );
}

#[tokio::test]
async fn provider_pool_defers_unavailable_sources_and_resumes_the_same_generation() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices":[{"message":{"content":"Recovered source description."}}],
            "usage":{"prompt_tokens":30,"completion_tokens":10,"total_tokens":40}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let (_root, env) = fixture(&server.uri());
    let job = prepare(&env);
    let id = queue::enqueue(&env.store, &job.enqueue(1.0), 0).unwrap();
    let settings = Arc::new(pool::Settings {
        provider: job.provider.clone(),
        concurrency: 1,
        max_jobs: 10,
        max_claim_tokens: i64::MAX,
        budget: chat::limits(&env.config.llm).unwrap().budget,
        max_output_tokens: 777,
        timeout: Duration::from_secs(3),
        heartbeat: Duration::from_millis(20),
    });
    let factory = Arc::new(Arc::clone(&env));
    fs::write(env.root.join("a.rs"), "fn temporarily_changed() {}").unwrap();
    let report = pool::run_ready(
        Arc::clone(&env.store),
        Arc::clone(&settings),
        Arc::clone(&factory),
        || 0,
    )
    .await
    .unwrap();
    assert_eq!(
        (report.claimed, report.deferred, report.published),
        (1, 1, 0)
    );
    assert!(report.errors.is_empty());
    assert!(server.received_requests().await.unwrap().is_empty());
    let state: (String,i64,i64,i64) = env.store.writer().unwrap().query_row(
        "SELECT state,attempts,not_before,(SELECT count(*) FROM usage_reservations) FROM jobs WHERE id=?1",
        [id], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))).unwrap();
    assert_eq!(state, ("queued".into(), 0, 60, 0));
    fs::write(env.root.join("a.rs"), "fn example() {}").unwrap();
    assert_eq!(
        queue::enqueue(&env.store, &prepare(&env).enqueue(1.0), 1).unwrap(),
        id
    );
    let early = pool::run_ready(
        Arc::clone(&env.store),
        Arc::clone(&settings),
        Arc::clone(&factory),
        || 59,
    )
    .await
    .unwrap();
    assert_eq!(early.claimed, 0);
    let resumed = pool::run_ready(
        Arc::clone(&env.store),
        Arc::clone(&settings),
        Arc::clone(&factory),
        || 60,
    )
    .await
    .unwrap();
    assert_eq!((resumed.claimed, resumed.published), (1, 1));
    assert!(resumed.errors.is_empty());
    assert_eq!(
        queue::enqueue(&env.store, &prepare(&env).enqueue(1.0), 61).unwrap(),
        id
    );
    let repeated = pool::run_ready(Arc::clone(&env.store), settings, factory, || 61)
        .await
        .unwrap();
    assert_eq!(repeated.claimed, 0);
    assert_eq!(env.store.enrichment_status().unwrap().summaries, 1);
}

#[tokio::test]
async fn scheduling_recovers_discarded_output_without_resetting_attempts_or_repeating_success() {
    let server = MockServer::start().await;
    let (_root, env) = fixture(&server.uri());
    let path = env.root.join("a.rs");
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    Mock::given(method("POST"))
        .respond_with(move |_: &wiremock::Request| {
            if calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                fs::write(&path, "fn changed_during_first_call() {}").unwrap();
            }
            ResponseTemplate::new(200).set_body_json(json!({
                "choices":[{"message":{"content":"Recovered and verified output."}}],
                "usage":{"prompt_tokens":30,"completion_tokens":10,"total_tokens":40}
            }))
        })
        .expect(2)
        .mount(&server)
        .await;
    let original = prepare(&env);
    assert_eq!(
        execute(&env, &original).await,
        worker::Finished::Completed(queue::Completion::Discarded)
    );
    assert_eq!(original.schedule(1.0, 0).unwrap(), Scheduled::Superseded);
    fs::write(env.root.join("a.rs"), "fn example() {}").unwrap();
    let Scheduled::QueuedOrExisting(id) = prepare(&env).schedule(1.0, 0).unwrap() else {
        panic!("discarded generation was not recovered")
    };
    let attempts: i64 = env
        .store
        .writer()
        .unwrap()
        .query_row("SELECT attempts FROM jobs WHERE id=?1", [id], |r| r.get(0))
        .unwrap();
    assert_eq!(attempts, 1);
    assert_eq!(
        execute(&env, &original).await,
        worker::Finished::Completed(queue::Completion::Published)
    );
    let Scheduled::Current(summary) = prepare(&env).schedule(1.0, 0).unwrap() else {
        panic!("successful generation was repeated")
    };
    assert!(summary > 0);
    let db = env.store.writer().unwrap();
    let state: (String, i64, i64) = db
        .query_row(
            "SELECT state,attempts,(SELECT sum(tokens) FROM usage) FROM jobs WHERE id=?1",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!(state, ("done".into(), 2, 80));
    // Even a missing output never resets the per-generation dispatch ceiling.
    db.execute("DELETE FROM units WHERE id=?1", [summary])
        .unwrap();
    db.execute("UPDATE jobs SET attempts=8 WHERE id=?1", [id])
        .unwrap();
    drop(db);
    assert_eq!(original.schedule(1.0, 0).unwrap(), Scheduled::Terminal(id));
    let state: (String, i64) = env
        .store
        .writer()
        .unwrap()
        .query_row("SELECT state,attempts FROM jobs WHERE id=?1", [id], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(state, ("done".into(), 8));
}

#[tokio::test]
async fn pr_worker_uses_immutable_member_hunks_and_publishes_the_pr_summary() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices":[{"message":{"content":"Adds the billing changes from the selected PR."}}],
            "usage":{"prompt_tokens":30,"completion_tokens":10,"total_tokens":40}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let (_temp, env) = indexed_fixture(&server.uri());
    let pr = env
        .store
        .writer()
        .unwrap()
        .query_row("SELECT id FROM prs WHERE number=102", [], |r| {
            r.get::<_, i64>(0)
        })
        .unwrap();
    let job = env
        .store
        .read_snapshot(|tx| env.prepare_job(tx, Kind::SummarizePr, pr))
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(text_input(&job).prompt.contains("Billing squash"));
    let hunks = text_input(&job)
        .prompt
        .split("Selected member hunks:\n")
        .nth(1)
        .unwrap();
    assert!(hunks.contains("@@"));
    assert!(crate::treat::count_tokens(hunks) <= 3_000);
    assert_eq!(
        execute(&env, &job).await,
        worker::Finished::Completed(queue::Completion::Published)
    );
    verify_summary_input(
        &env,
        text_input(&job).owner,
        text_input(&job).parent_unit,
        &job,
    );
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(body["max_tokens"], 300);
    assert!(
        env.store
            .writer()
            .unwrap()
            .query_row(
                "SELECT summary_unit IS NOT NULL FROM prs WHERE id=?1",
                [pr],
                |r| r.get::<_, bool>(0)
            )
            .unwrap()
    );
}

fn indexed_fixture(url: &str) -> (tempfile::TempDir, Arc<Context>) {
    indexed_fixture_with_setup(url, |_| {})
}

fn indexed_fixture_with_setup(
    url: &str,
    setup: impl FnOnce(&std::path::Path),
) -> (tempfile::TempDir, Arc<Context>) {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("histories");
    let made = std::process::Command::new(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/fixtures/make-histories.sh"),
    )
    .arg(&output)
    .arg("squash")
    .output()
    .unwrap();
    assert!(
        made.status.success(),
        "{}",
        String::from_utf8_lossy(&made.stderr)
    );
    let root = output.join("squash");
    setup(&root);
    fs::create_dir_all(root.join(".deepkit")).unwrap();
    let mut config = Config::default();
    config.enrich.modules_min_files = 1;
    config.llm.enabled = true;
    config.llm.url = format!("{url}/v1");
    config.llm.model = "fixture-text".into();
    config.treatments = vec![crate::config::Treatment {
        name: "source".into(),
        patterns: vec!["**".into()],
        max_level: 3,
        enrich: vec!["describe_symbol".into()],
        ..Default::default()
    }];
    fs::write(
        root.join(".deepkit/config.toml"),
        toml::to_string(&config).unwrap(),
    )
    .unwrap();
    let indexed = std::process::Command::new(assert_cmd::cargo::cargo_bin("dkc"))
        .current_dir(&root)
        .args(["index", "--level", "1", "--no-queue", "--json"])
        .output()
        .unwrap();
    assert!(
        indexed.status.success(),
        "{}",
        String::from_utf8_lossy(&indexed.stderr)
    );
    let config = crate::config::load(&root, None).unwrap().config;
    let store = crate::store::open(
        root.join(".deepkit/code.db"),
        crate::store::StoreLimits::default(),
    )
    .unwrap();
    let client =
        chat::Client::with_key(&config.llm, Duration::from_secs(3), 1024, "fixture".into())
            .unwrap();
    let env = Arc::new(Context {
        store: Arc::new(store),
        root: root.clone(),
        config_root: root,
        config: Arc::new(config),
        client: Arc::new(client),
    });
    (temp, env)
}

#[tokio::test]
async fn metadata_only_pr_can_publish_without_inventing_changed_file_sources() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices":[{"message":{"content":"Records a release checkpoint without file changes."}}],
            "usage":{"prompt_tokens":30,"completion_tokens":10,"total_tokens":40}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let (_temp, env) = indexed_fixture_with_setup(&server.uri(), |root| {
        let committed = std::process::Command::new("git")
            .current_dir(root)
            .args([
                "commit",
                "--allow-empty",
                "-q",
                "-m",
                "Record release checkpoint without file changes (#9001)",
            ])
            .status()
            .unwrap();
        assert!(committed.success());
    });
    let (owner, parent, member, files): (i64, i64, i64, String) = env.store.writer().unwrap().query_row(
        "SELECT p.id,p.unit_id,c.id,c.files_touched FROM prs p JOIN commits c ON c.pr_id=p.id WHERE p.number=9001",
        [], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
    ).unwrap();
    assert_eq!(
        serde_json::from_str::<Vec<String>>(&files).unwrap(),
        Vec::<String>::new()
    );
    let job = env
        .store
        .read_snapshot(|tx| env.prepare_job(tx, Kind::SummarizePr, owner))
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(
        text_input(&job)
            .prompt
            .contains("Record release checkpoint")
    );
    assert_eq!(
        execute(&env, &job).await,
        worker::Finished::Completed(queue::Completion::Published)
    );
    let summary: i64 = env
        .store
        .writer()
        .unwrap()
        .query_row("SELECT summary_unit FROM prs WHERE id=?1", [owner], |r| {
            r.get(0)
        })
        .unwrap();
    let sources: Vec<(String, i64)> = env.store.writer().unwrap().prepare(
        "SELECT source_kind,source_id FROM unit_sources WHERE unit_id=?1 ORDER BY source_kind,source_id"
    ).unwrap().query_map([summary], |r| Ok((r.get(0)?, r.get(1)?))).unwrap()
        .collect::<Result<_, _>>().unwrap();
    assert!(sources.contains(&("commit".to_owned(), member)));
    assert!(sources.contains(&("unit".to_owned(), parent)));
    assert!(!sources.iter().any(|(kind, _)| kind == "file"));
}

#[tokio::test]
async fn large_pr_publishes_a_bounded_prompt_without_truncating_stored_collation() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices":[{"message":{"content":"Summarizes the bounded large pull request."}}],
            "usage":{"prompt_tokens":30,"completion_tokens":10,"total_tokens":40}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let (_temp, env) = indexed_fixture_with_setup(&server.uri(), |root| {
        let message = root.join(".git/dkc-large-pr-message");
        fs::write(
            &message,
            format!(
                "Large bounded prompt (#9002)\n\nBEGIN-LARGE-PR {} END-LARGE-PR\n",
                "repository evidence 文件🦀 ".repeat(20_000)
            ),
        )
        .unwrap();
        let committed = std::process::Command::new("git")
            .current_dir(root)
            .args(["commit", "--allow-empty", "-q", "-F"])
            .arg(&message)
            .status()
            .unwrap();
        assert!(committed.success());
    });
    let (owner, parent, full_text): (i64, i64, String) = env
        .store
        .writer()
        .unwrap()
        .query_row(
            "SELECT p.id,p.unit_id,c.text FROM prs p JOIN units u ON u.id=p.unit_id
             JOIN unit_contents c ON c.hash=u.content_hash WHERE p.number=9002",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert!(crate::treat::count_tokens(&full_text) > 7_500);
    let job = env
        .store
        .read_snapshot(|tx| env.prepare_job(tx, Kind::SummarizePr, owner))
        .unwrap()
        .unwrap()
        .unwrap();
    let prompt = &text_input(&job).prompt;
    assert!(crate::treat::count_tokens(prompt) <= 7_500);
    assert!(prompt.contains("BEGIN-LARGE-PR"));
    assert!(prompt.contains("END-LARGE-PR"));
    assert!(prompt.contains("omitted at PR prompt input limit"));
    assert_eq!(
        execute(&env, &job).await,
        worker::Finished::Completed(queue::Completion::Published)
    );
    let retained: String = env
        .store
        .writer()
        .unwrap()
        .query_row(
            "SELECT c.text FROM units u JOIN unit_contents c ON c.hash=u.content_hash WHERE u.id=?1",
            [parent],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(retained, full_text);
}

#[tokio::test]
async fn large_module_publishes_a_bounded_prompt_without_truncating_stored_collation() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices":[{"message":{"content":"Repository module with many public entry points."}}],
            "usage":{"prompt_tokens":30,"completion_tokens":10,"total_tokens":40}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let (_temp, env) = indexed_fixture_with_setup(&server.uri(), |root| {
        let generated = std::process::Command::new("python3")
            .arg(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("scripts/fixtures/make-large-module.py"),
            )
            .arg(root)
            .status()
            .unwrap();
        assert!(generated.success());
    });
    let (unit, full_text): (i64, String) = env
        .store
        .writer()
        .unwrap()
        .query_row(
            "SELECT d.module_unit,c.text FROM dirs d JOIN units u ON u.id=d.module_unit
         JOIN unit_contents c ON c.hash=u.content_hash WHERE d.path=''",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert!(full_text.len() > 256 * 1024, "{}", full_text.len());
    let job = env
        .store
        .read_snapshot(|tx| env.prepare_job(tx, Kind::SummarizeModule, unit))
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(crate::treat::count_tokens(&text_input(&job).prompt) <= 5_200);
    assert!(
        text_input(&job)
            .prompt
            .contains("omitted at module prompt input limit")
    );
    assert_eq!(
        execute(&env, &job).await,
        worker::Finished::Completed(queue::Completion::Published)
    );
    let requests = server.received_requests().await.unwrap();
    let sent: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(
        sent["messages"][1]["content"].as_str().unwrap(),
        text_input(&job).prompt
    );
    let retained: String = env.store.writer().unwrap().query_row(
        "SELECT c.text FROM units u JOIN unit_contents c ON c.hash=u.content_hash WHERE u.id=?1",
        [unit], |r| r.get(0),
    ).unwrap();
    assert_eq!(retained, full_text);
    fs::write(env.root.join("large_059.rs"), "pub fn changed() {}\n").unwrap();
    assert!(
        env.store
            .read_snapshot(|tx| env.prepare_job(tx, Kind::SummarizeModule, unit))
            .unwrap()
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn module_worker_uses_only_current_member_summary_generations() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices":[{"message":{"content":"Verified member responsibility."}}],
            "usage":{"prompt_tokens":30,"completion_tokens":10,"total_tokens":40}
        })))
        .expect(2)
        .mount(&server)
        .await;
    let (_temp, env) = indexed_fixture(&server.uri());
    let symbol = env
        .store
        .writer()
        .unwrap()
        .query_row(
            "SELECT id FROM symbols WHERE name='initial' LIMIT 1",
            [],
            |r| r.get::<_, i64>(0),
        )
        .unwrap();
    let job = env
        .store
        .read_snapshot(|tx| env.symbol(tx, symbol))
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(
        execute(&env, &job).await,
        worker::Finished::Completed(queue::Completion::Published)
    );
    let unit = env
        .store
        .writer()
        .unwrap()
        .query_row("SELECT module_unit FROM dirs WHERE path=''", [], |r| {
            r.get::<_, i64>(0)
        })
        .unwrap();
    let prepare_module = || {
        env.store
            .read_snapshot(|tx| env.prepare_job(tx, Kind::SummarizeModule, unit))
            .unwrap()
            .unwrap()
            .unwrap()
    };
    let module = prepare_module();
    assert!(
        text_input(&module)
            .prompt
            .contains("Verified member responsibility.")
    );
    fs::create_dir_all(env.root.join(".deepkit/prompts")).unwrap();
    let override_path = env.root.join(".deepkit/prompts/describe_symbol.md");
    fs::write(
        &override_path,
        "New description style: {{signature}} {{code}}",
    )
    .unwrap();
    let changed = prepare_module();
    assert!(
        !text_input(&changed)
            .prompt
            .contains("Verified member responsibility.")
    );
    assert_ne!(module.desired, changed.desired);
    assert!(
        env.store
            .writer()
            .unwrap()
            .query_row(
                "SELECT summary_unit IS NOT NULL FROM symbols WHERE id=?1",
                [symbol],
                |r| r.get::<_, bool>(0)
            )
            .unwrap()
    );
    fs::remove_file(override_path).unwrap();
    assert_eq!(
        execute(&env, &module).await,
        worker::Finished::Completed(queue::Completion::Published)
    );
    verify_summary_input(
        &env,
        text_input(&job).owner,
        text_input(&job).parent_unit,
        &job,
    );
    verify_summary_input(
        &env,
        text_input(&module).owner,
        text_input(&module).parent_unit,
        &module,
    );
    assert!(
        env.store
            .read_snapshot(|tx| {
                tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM meta WHERE key='lexical_role_reconcile_pending')",
                    [],
                    |row| row.get::<_, bool>(0),
                )
            })
            .unwrap()
            .unwrap(),
        "module publication must leave the cohort reconciliation marker durable"
    );
    crate::queue::roles::reconcile(
        &env.store,
        &env.root,
        &env.config_root,
        &env.config,
        Default::default(),
    )
    .unwrap();
    {
        let db = env.store.writer().unwrap();
        let files: i64 = db.query_row("SELECT count(*) FROM units WHERE file_id IS NOT NULL AND parent_unit IS NULL AND kind IN ('chunk','symbol')", [], |r| r.get(0)).unwrap();
        let updated: i64 = db.query_row("SELECT count(*) FROM fts_units JOIN units u ON u.id=fts_units.rowid WHERE fts_units MATCH 'header : responsibility' AND u.file_id IS NOT NULL AND u.parent_unit IS NULL AND u.kind IN ('chunk','symbol')", [], |r| r.get(0)).unwrap();
        assert!(files > 0);
        assert_eq!(
            updated, files,
            "cohort reconciliation must update file headers after module publication"
        );
    }
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(body["max_tokens"], 200);
    assert!(
        env.store
            .writer()
            .unwrap()
            .query_row(
                "SELECT summary_unit IS NOT NULL FROM dirs WHERE module_unit=?1",
                [unit],
                |r| r.get::<_, bool>(0)
            )
            .unwrap()
    );
}

#[tokio::test]
async fn chat_host_finishes_bounded_cohorts_without_embedding_resources_or_duplicate_calls() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices":[{"message":{"content":"A verified host description."}}],
            "usage":{"prompt_tokens":30,"completion_tokens":10,"total_tokens":40}
        })))
        .mount(&server)
        .await;
    let (_temp, original) = indexed_fixture(&server.uri());
    let mut config = (*original.config).clone();
    config.queue.max_schedule_per_pass = 1;
    config.queue.max_jobs_per_pass = 2;
    fs::write(
        original.root.join(".deepkit/config.toml"),
        toml::to_string(&config).unwrap(),
    )
    .unwrap();
    let env = Arc::new(Context {
        store: Arc::clone(&original.store),
        root: original.root.clone(),
        config_root: original.config_root.clone(),
        config: Arc::new(config),
        client: Arc::clone(&original.client),
    });
    assert!(env.store.vector_tables().unwrap().is_empty());
    for repeat in 0..2 {
        let mut host = queue::enrichment_host::Host::from_context(Arc::clone(&env), 0).unwrap();
        let mut published = 0;
        let mut passes = 0;
        while !host.catalogue_finished().unwrap() {
            passes += 1;
            assert!(passes < 100, "host never reached a stable finite cohort");
            let report = host.pass(|| 0).await.unwrap();
            assert!(report.error.is_none());
            assert!(report.execution.errors.is_empty());
            assert!(report.visited <= 1);
            assert!(
                !env.store
                    .read_snapshot(|tx| tx.query_row(
                        "SELECT EXISTS(SELECT 1 FROM meta WHERE key='lexical_role_reconcile_pending')",
                        [],
                        |row| row.get::<_, bool>(0),
                    ))
                    .unwrap()
                    .unwrap(),
                "each completed worker pass must reconcile its published roles"
            );
            published += report.execution.published;
        }
        assert!(passes > 1);
        if repeat == 0 {
            assert!(published > 0)
        } else {
            assert_eq!(published, 0)
        }
    }
    let calls = server.received_requests().await.unwrap();
    let summaries = env.store.enrichment_status().unwrap().summaries;
    assert_eq!(calls.len() as u64, summaries);
    assert!(calls.iter().any(|request| {
        let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
        body["max_tokens"] == 200 && body.to_string().contains("A verified host description.")
    }));
    assert!(env.store.vector_tables().unwrap().is_empty());
}

#[tokio::test]
async fn disabled_chat_host_does_not_read_roots_or_credentials() {
    let (_root, env) = fixture("http://127.0.0.1:9");
    let mut config = (*env.config).clone();
    config.llm.enabled = false;
    config.llm.url = "invalid unused URL".into();
    assert!(
        queue::enrichment_host::Host::prepare(
            Arc::clone(&env.store),
            "/missing-chat-host-root".into(),
            "/missing-chat-host-config".into(),
            config,
            0
        )
        .await
        .unwrap()
        .is_none()
    );
}

#[tokio::test]
async fn public_level_three_chat_survives_unavailable_embeddings_and_is_idempotent() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices":[{"message":{"content":"A public CLI summary."}}],
            "usage":{"prompt_tokens":30,"completion_tokens":10,"total_tokens":40}
        })))
        .mount(&server)
        .await;
    let (_temp, env) = indexed_fixture(&server.uri());
    let mut config = (*env.config).clone();
    config.llm.api_key_env = "DKC_TEST_CHAT_HOST_KEY".into();
    config.embedder.profile = "local".into();
    config.embedder.local.model = "unavailable-fixture-model".into();
    config.queue.max_schedule_per_pass = 1;
    fs::write(
        env.root.join(".deepkit/config.toml"),
        toml::to_string(&config).unwrap(),
    )
    .unwrap();
    // This source did not exist in the captured L1 index. Dispatch must follow
    // publication so the first public pass sees and describes this new symbol.
    fs::write(
        env.root.join("new.rs"),
        "pub fn newly_indexed() -> bool { true }\n",
    )
    .unwrap();
    let command = |args: &[&str]| {
        let mut command = tokio::process::Command::new(assert_cmd::cargo::cargo_bin("dkc"));
        command
            .current_dir(&env.root)
            .env("DKC_TEST_CHAT_HOST_KEY", "fixture")
            .args(args);
        command
    };
    for args in [
        vec!["index", "--level", "2", "--json"],
        vec!["index", "--level", "3", "--no-queue", "--json"],
    ] {
        let output = command(&args).output().await.unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["enrichment"]["state"], "skipped");
    }
    assert!(server.received_requests().await.unwrap().is_empty());
    fs::write(
        env.root.join("new.rs"),
        "pub fn newest_public_symbol() -> bool { true }\n",
    )
    .unwrap();
    let mut previous_calls = 0;
    for repeat in 0..2 {
        let output = command(&["index", "--level", "3", "--json"])
            .output()
            .await
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["enrichment"]["state"], "complete", "{report}");
        assert_eq!(report["model_preparation"]["state"], "unavailable");
        assert_eq!(report["embedding"]["state"], "skipped");
        let calls = server.received_requests().await.unwrap();
        if repeat == 0 {
            assert!(!calls.is_empty());
            assert!(calls.iter().any(|request| {
                String::from_utf8_lossy(&request.body).contains("newest_public_symbol")
            }));
            previous_calls = calls.len();
        } else {
            assert_eq!(calls.len(), previous_calls);
            assert_eq!(report["enrichment"]["published"], 0);
        }
    }
    assert!(env.store.vector_tables().unwrap().is_empty());
    assert_eq!(
        env.store.enrichment_status().unwrap().summaries as usize,
        previous_calls
    );
}

#[tokio::test]
async fn public_chat_drain_preserves_delays_and_never_schedules_new_targets() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices":[{"message":{"content":"One persisted job summary."}}],
            "usage":{"prompt_tokens":30,"completion_tokens":10,"total_tokens":40}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let (_temp, original) = indexed_fixture(&server.uri());
    let mut config = (*original.config).clone();
    config.llm.api_key_env = "DKC_TEST_CHAT_DRAIN_KEY".into();
    config.embedder.profile = "local".into();
    config.embedder.local.model = "unavailable-fixture-model".into();
    fs::write(
        original.root.join(".deepkit/config.toml"),
        toml::to_string(&config).unwrap(),
    )
    .unwrap();
    let command = |args: &[&str]| {
        let mut command = tokio::process::Command::new(assert_cmd::cargo::cargo_bin("dkc"));
        command
            .current_dir(&original.root)
            .env("DKC_TEST_CHAT_DRAIN_KEY", "fixture")
            .args(args);
        command
    };
    let indexed = command(&["index", "--level", "1", "--json"])
        .output()
        .await
        .unwrap();
    assert!(indexed.status.success());
    let env = Arc::new(Context {
        store: Arc::clone(&original.store),
        root: original.root.clone(),
        config_root: original.config_root.clone(),
        config: Arc::new(crate::config::load(&original.root, None).unwrap().config),
        client: Arc::clone(&original.client),
    });
    let selected = env
        .store
        .read_snapshot(|tx| queue::enrichment_catalogue::select(tx, &env.config))
        .unwrap()
        .unwrap();
    assert!(selected.len() > 1);
    let target = &selected[0];
    let job = env
        .store
        .read_snapshot(|tx| env.prepare_job(tx, target.kind, target.target_id))
        .unwrap()
        .unwrap()
        .unwrap();
    let Scheduled::QueuedOrExisting(id) = job.schedule(1.0, 0).unwrap() else {
        panic!("expected persisted job")
    };
    env.store
        .writer()
        .unwrap()
        .execute(
            "UPDATE jobs SET not_before=9223372036854775807 WHERE id=?1",
            [id],
        )
        .unwrap();
    for waiting in [true, false, false] {
        if !waiting {
            env.store
                .writer()
                .unwrap()
                .execute("UPDATE jobs SET not_before=0 WHERE id=?1", [id])
                .unwrap();
        }
        let output = command(&["queue", "drain", "--max-passes", "1", "--json"])
            .output()
            .await
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            report["state"],
            if waiting { "pending" } else { "quiescent" },
            "{report}"
        );
        assert_eq!(
            server.received_requests().await.unwrap().len(),
            usize::from(!waiting)
        );
        let jobs: i64 = env
            .store
            .writer()
            .unwrap()
            .query_row("SELECT count(*) FROM jobs", [], |r| r.get(0))
            .unwrap();
        assert_eq!(jobs, 1, "drain must not schedule other eligible targets");
    }
    assert!(env.store.vector_tables().unwrap().is_empty());
}

#[cfg(unix)]
#[tokio::test]
async fn public_chat_interrupt_retains_committed_summary_and_outstanding_lease() {
    use std::process::Stdio;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    Mock::given(method("POST"))
        .respond_with(move |_: &wiremock::Request| {
            let response = ResponseTemplate::new(200).set_body_json(json!({
                "choices":[{"message":{"content":"Retained committed description."}}],
                "usage":{"prompt_tokens":30,"completion_tokens":10,"total_tokens":40}
            }));
            if observed.fetch_add(1, Ordering::SeqCst) == 0 {
                response
            } else {
                response.set_delay(Duration::from_secs(30))
            }
        })
        .mount(&server)
        .await;
    let (_temp, env) = indexed_fixture(&server.uri());
    let mut config = (*env.config).clone();
    config.llm.api_key_env = "DKC_TEST_CHAT_INTERRUPT_KEY".into();
    config.llm.concurrency = 1;
    config.queue.max_jobs_per_pass = 1;
    config.queue.max_schedule_per_pass = 1;
    config.embedder.profile = "local".into();
    config.embedder.local.model = "unavailable-fixture-model".into();
    fs::write(
        env.root.join(".deepkit/config.toml"),
        toml::to_string(&config).unwrap(),
    )
    .unwrap();
    let mut child = tokio::process::Command::new(assert_cmd::cargo::cargo_bin("dkc"))
        .current_dir(&env.root)
        .env("DKC_TEST_CHAT_INTERRUPT_KEY", "fixture")
        .args(["index", "--level", "3", "--json"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    tokio::time::timeout(Duration::from_secs(10), async {
        while calls.load(Ordering::SeqCst) < 2 {
            assert!(child.try_wait().unwrap().is_none());
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let killed = std::process::Command::new("kill")
        .args(["-TERM", &child.id().unwrap().to_string()])
        .status()
        .unwrap();
    assert!(killed.success());
    let output = tokio::time::timeout(Duration::from_secs(5), child.wait_with_output())
        .await
        .unwrap()
        .unwrap();
    assert!(!output.status.success());
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["enrichment"]["state"], "cancelled");
    assert_eq!(report["enrichment"]["published"], 1);
    assert_eq!(env.store.enrichment_status().unwrap().summaries, 1);
    let (done, leased): (i64, i64) = env
        .store
        .writer()
        .unwrap()
        .query_row(
            "SELECT sum(state='done'),sum(state='leased') FROM jobs",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!((done, leased), (1, 1));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn public_chat_index_reports_matching_dead_generation_without_dispatch() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices":[{"message":{"content":"A corrected new generation."}}],
            "usage":{"prompt_tokens":30,"completion_tokens":10,"total_tokens":40}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let (_temp, original) = indexed_fixture(&server.uri());
    let mut config = (*original.config).clone();
    config.llm.api_key_env = "DKC_TEST_CHAT_TERMINAL_KEY".into();
    config.embedder.profile = "local".into();
    config.embedder.local.model = "unavailable-fixture-model".into();
    config.enrich.top_symbols = 1;
    config.enrich.prs_last_n = 0;
    config.enrich.modules_min_files = 99999;
    fs::write(
        original.root.join(".deepkit/config.toml"),
        toml::to_string(&config).unwrap(),
    )
    .unwrap();
    let command = |args: &[&str]| {
        let mut command = tokio::process::Command::new(assert_cmd::cargo::cargo_bin("dkc"));
        command
            .current_dir(&original.root)
            .env("DKC_TEST_CHAT_TERMINAL_KEY", "fixture")
            .args(args);
        command
    };
    assert!(
        command(&["index", "--level", "1", "--json"])
            .output()
            .await
            .unwrap()
            .status
            .success()
    );
    let env = Arc::new(Context {
        store: Arc::clone(&original.store),
        root: original.root.clone(),
        config_root: original.config_root.clone(),
        config: Arc::new(crate::config::load(&original.root, None).unwrap().config),
        client: Arc::clone(&original.client),
    });
    let selected = env
        .store
        .read_snapshot(|tx| queue::enrichment_catalogue::select(tx, &env.config))
        .unwrap()
        .unwrap();
    assert_eq!(selected.len(), 1);
    let target = &selected[0];
    let job = env
        .store
        .read_snapshot(|tx| env.prepare_job(tx, target.kind, target.target_id))
        .unwrap()
        .unwrap()
        .unwrap();
    let Scheduled::QueuedOrExisting(id) = job.schedule(1.0, 0).unwrap() else {
        panic!("expected persisted job")
    };
    env.store.writer().unwrap().execute("UPDATE jobs SET state='dead',attempts=8,last_error='attempt limit reached' WHERE id=?1",[id]).unwrap();
    let output = command(&["index", "--level", "3", "--json"])
        .output()
        .await
        .unwrap();
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(
        !output.status.success(),
        "matching terminal generation must not report success: {report}"
    );
    assert_eq!(report["enrichment"]["state"], "failed");
    assert_eq!(report["enrichment"]["terminal"], 1);
    assert!(server.received_requests().await.unwrap().is_empty());
    assert_eq!(env.store.enrichment_status().unwrap().summaries, 0);
    // An old dead generation must not block a newly configured, eligible input.
    fs::create_dir_all(env.root.join(".deepkit/prompts")).unwrap();
    fs::write(
        env.root.join(".deepkit/prompts/describe_symbol.md"),
        "New generation: {{signature}} {{code}}",
    )
    .unwrap();
    let output = command(&["index", "--level", "3", "--json"])
        .output()
        .await
        .unwrap();
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(
        output.status.success(),
        "historic terminal work blocked current generation: {report}"
    );
    assert_eq!(report["enrichment"]["state"], "complete");
    assert_eq!(report["enrichment"]["terminal"], 0);
    assert_eq!(report["enrichment"]["published"], 1);
    let old: (String, i64) = env
        .store
        .writer()
        .unwrap()
        .query_row("SELECT state,attempts FROM jobs WHERE id=?1", [id], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(old, ("dead".into(), 8));
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn provider_pool_diagnoses_invalid_preparation_and_resumes_without_spending_attempts() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices":[{"message":{"content":"Recovered source description."}}],
            "usage":{"prompt_tokens":30,"completion_tokens":10,"total_tokens":40}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let (_root, env) = fixture(&server.uri());
    let job = prepare(&env);
    let id = queue::enqueue(&env.store, &job.enqueue(1.0), 0).unwrap();
    let settings = Arc::new(pool::Settings {
        provider: job.provider.clone(),
        concurrency: 1,
        max_jobs: 10,
        max_claim_tokens: i64::MAX,
        budget: chat::limits(&env.config.llm).unwrap().budget,
        max_output_tokens: 777,
        timeout: Duration::from_secs(3),
        heartbeat: Duration::from_millis(20),
    });
    let factory = Arc::new(Arc::clone(&env));
    fs::create_dir_all(env.root.join(".deepkit/prompts")).unwrap();
    let bad_prompt = env.root.join(".deepkit/prompts/describe_symbol.md");
    fs::write(&bad_prompt, [0xff]).unwrap();
    let report = pool::run_ready(
        Arc::clone(&env.store),
        Arc::clone(&settings),
        Arc::clone(&factory),
        || 0,
    )
    .await
    .unwrap();
    assert_eq!(
        (report.claimed, report.deferred, report.published),
        (1, 1, 0)
    );
    assert!(report.errors.is_empty());
    assert!(server.received_requests().await.unwrap().is_empty());
    let state: (String,i64,i64,i64) = env.store.writer().unwrap().query_row(
        "SELECT state,attempts,not_before,(SELECT count(*) FROM usage_reservations) FROM jobs WHERE id=?1",
        [id], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))).unwrap();
    assert_eq!(state, ("queued".into(), 0, 60, 0));
    let reason: String = env
        .store
        .writer()
        .unwrap()
        .query_row("SELECT last_error FROM jobs WHERE id=?1", [id], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(
        reason,
        "input preparation failed; inspect configuration, prompts and source bounds"
    );
    fs::remove_file(bad_prompt).unwrap();
    assert_eq!(
        queue::enqueue(&env.store, &prepare(&env).enqueue(1.0), 1).unwrap(),
        id
    );
    let early = pool::run_ready(
        Arc::clone(&env.store),
        Arc::clone(&settings),
        Arc::clone(&factory),
        || 59,
    )
    .await
    .unwrap();
    assert_eq!(early.claimed, 0);
    let resumed = pool::run_ready(
        Arc::clone(&env.store),
        Arc::clone(&settings),
        Arc::clone(&factory),
        || 60,
    )
    .await
    .unwrap();
    assert_eq!((resumed.claimed, resumed.published), (1, 1));
    assert!(resumed.errors.is_empty());
    assert_eq!(
        queue::enqueue(&env.store, &prepare(&env).enqueue(1.0), 61).unwrap(),
        id
    );
    let repeated = pool::run_ready(Arc::clone(&env.store), settings, factory, || 61)
        .await
        .unwrap();
    assert_eq!(repeated.claimed, 0);
    assert_eq!(env.store.enrichment_status().unwrap().summaries, 1);
}

#[tokio::test]
#[ignore = "requires DKC_LIVE_MODULE_DIAGNOSTIC=1, isolated diagnostic root and output path; one admitted saved-job attempt"]
async fn diagnose_saved_module_once_with_accounting() -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(std::env::var("DKC_LIVE_MODULE_DIAGNOSTIC")?, "1");
    let root = fs::canonicalize(std::env::var("DKC_MODULE_DIAGNOSTIC_ROOT")?)?;
    assert_eq!(
        root,
        PathBuf::from("/data/dkc-corpus/t3-11-diagnostic-2/laravel-framework")
    );
    let output = PathBuf::from(std::env::var("DKC_MODULE_DIAGNOSTIC_OUT")?);
    assert!(!output.exists());
    let config = Arc::new(crate::config::load(&root, None)?.config);
    let store = Arc::new(crate::store::open(
        root.join(".deepkit/code.db"),
        crate::store::StoreLimits {
            db_max_bytes: config.limits.db_max_bytes()?.0,
            wal_max_bytes: config.limits.wal_max_bytes()?.0,
            max_l2_units: config.limits.max_l2_units,
            max_l3_units: config.limits.max_l3_units,
        },
    )?);
    let job_id = 35913;
    let before: (i64, String, i64, Vec<u8>, Vec<u8>) = store.writer()?.query_row(
        "SELECT target_id,state,attempts,target_hash,desired_hash FROM jobs WHERE id=?1 AND kind='enrich:summarize_module'",
        [job_id], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?)),
    )?;
    assert_eq!((before.0, before.1.as_str(), before.2), (37659, "dead", 1));
    let client = Arc::new(chat::Client::from_config(
        &config.llm,
        Duration::from_secs(120),
        config.limits.max_image_bytes()?.0,
    )?);
    let context = Arc::new(Context {
        store: Arc::clone(&store),
        root: root.clone(),
        config_root: root,
        config,
        client,
    });
    let job = store
        .read_snapshot(|tx| context.prepare_job(tx, Kind::SummarizeModule, before.0))??
        .ok_or("module inputs unavailable")?;
    assert_eq!(job.input.target_hash(), before.3.as_slice());
    assert_eq!(job.desired.as_slice(), before.4.as_slice());
    let now = || {
        i64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        )
        .unwrap()
    };
    assert_eq!(queue::retry::request(&store, Some(job_id), now())?, 1);
    let leases = queue::claim(
        &store,
        &job.provider,
        &queue::Owner::new(&store)?,
        now(),
        1,
        i64::MAX,
    )?;
    assert_eq!(leases.len(), 1);
    assert_eq!(
        leases[0].id, job_id,
        "refuse dispatch if another queued job was selected"
    );
    let started = std::time::Instant::now();
    let result = worker::execute(
        &store,
        &leases[0],
        &worker::Settings {
            provider: &job.provider,
            budget: chat::limits(&context.config.llm)?.budget,
            max_output_tokens: job.wire.max_output_tokens,
            timeout: Duration::from_secs(120),
            heartbeat: Duration::from_secs(1),
        },
        &job,
        now,
    )
    .await?;
    let after: (String, i64, Option<String>) = store.writer()?.query_row(
        "SELECT state,attempts,last_error FROM jobs WHERE id=?1",
        [job_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    let report = json!({"state":"diagnostic_attempt_finished","job_id":job_id,"target_id":before.0,
        "input_tokens":job.wire.input_tokens,"max_output_tokens":job.wire.max_output_tokens,
        "worker_result":format!("{result:?}"),"job_state":after.0,"attempts":after.1,
        "safe_error":after.2,"wall_seconds":started.elapsed().as_secs_f64(),
        "queue":queue::status::snapshot_with_jobs(&store,now(),Some(1000))?,
        "phase_complete":false,"recovery_complete":false});
    serde_json::to_writer_pretty(fs::File::create_new(output)?, &report)?;
    assert_eq!(
        after.1,
        before.2 + 1,
        "diagnosis requires exactly one admitted attempt"
    );
    Ok(())
}
