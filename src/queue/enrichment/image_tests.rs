//! Real shared image-job dispatch, accounting and source fences; v1 §6.8/§8.2.
use super::*;
use crate::{queue, sync, treat::Router};
use rusqlite::params;
use serde_json::json;
use std::{fs, io::Cursor, time::Duration};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

pub(super) fn register(env: &Context, id: i64, path: &str, bytes: &[u8]) {
    fs::write(env.root.join(path), bytes).unwrap();
    let (kind, language) = sync::classify_path(path);
    let treatment = Router::new(&env.config.treatments, true)
        .unwrap()
        .route(path, kind);
    let hash = *blake3::hash(bytes).as_bytes();
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
    env.store.writer().unwrap().execute("INSERT INTO files(id,path,blake3,size,mtime_ns,lang,kind,max_level,treatment,l1_pending,l1_desired,l1_fingerprint)
        VALUES(?1,?2,?3,?4,0,?5,?6,?7,?8,0,?9,?9)
        ON CONFLICT(id) DO UPDATE SET blake3=excluded.blake3,size=excluded.size,max_level=excluded.max_level,
        treatment=excluded.treatment,l1_pending=0,l1_desired=excluded.l1_desired,l1_fingerprint=excluded.l1_fingerprint",
        params![id,path,hash.as_slice(),bytes.len() as i64,language,kind.as_str(),treatment.max_level,treatment.name,fingerprint.as_slice()]).unwrap();
    // Updating treatment metadata correctly marks L1 pending. This fixture
    // represents the subsequent completed L1 pass, so settle it separately.
    env.store
        .writer()
        .unwrap()
        .execute("UPDATE files SET l1_pending=0 WHERE id=?1", [id])
        .unwrap();
}
pub(super) fn png() -> Vec<u8> {
    let mut bytes = Cursor::new(Vec::new());
    image::DynamicImage::new_rgb8(2, 2)
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    bytes.into_inner()
}
pub(super) fn fixture(
    url: &str,
    configure: impl FnOnce(&mut Config),
) -> (tempfile::TempDir, Arc<Context>) {
    let (root, store, mut config) = queue::source::tests::fixture().unwrap();
    config.llm.enabled = true;
    config.llm.url = format!("{url}/v1");
    config.llm.model = "fixture-text".into();
    config.llm.vlm_model = "fixture-vision".into();
    config.treatments.insert(
        0,
        crate::config::Treatment {
            name: "fixture-code".into(),
            patterns: vec!["**/*.rs".into()],
            max_level: 3,
            enrich: vec!["describe_symbol".into()],
            ..Default::default()
        },
    );
    configure(&mut config);
    fs::write(
        root.path().join(".deepkit/config.toml"),
        toml::to_string(&config).unwrap(),
    )
    .unwrap();
    let config = crate::config::load(root.path(), None).unwrap().config;
    let client = chat::Client::with_key(
        &config.llm,
        Duration::from_secs(3),
        1024 * 1024,
        "fixture".into(),
    )
    .unwrap();
    let env = Arc::new(Context {
        store: Arc::new(store),
        root: root.path().into(),
        config_root: root.path().into(),
        config: Arc::new(config),
        client: Arc::new(client),
    });
    register(&env, 1, "a.rs", b"fn example() {}");
    env.store.writer().unwrap().execute_batch("INSERT INTO unit_sources(unit_id,source_kind,source_id,role) VALUES(1,'file',1,'content');
        INSERT INTO symbols(id,file_id,unit_id,name,signature,kind,line_start,line_end) VALUES(1,1,1,'example','fn example()','function',1,1);").unwrap();
    register(&env, 2, "diagram.png", &png());
    register(
        &env,
        3,
        "README.md",
        b"# Architecture\n\nRequest flow uses ![connections](diagram.png).\n",
    );
    // The fixture contains one isolated symbol; its PageRank is one.
    env.store
        .update_graph_ranks(
            env.store.l1_identity().unwrap().scan,
            &[crate::store::GraphSymbolRank {
                symbol_id: 1,
                rank: 1.0,
            }],
            &[crate::store::GraphFileRank {
                file_id: 1,
                rank: 1.0,
            }],
        )
        .unwrap();
    (root, env)
}
fn prepare(env: &Arc<Context>, id: i64) -> Job {
    env.store
        .read_snapshot(|tx| env.prepare_job(tx, Kind::DescribeImage, id))
        .unwrap()
        .unwrap()
        .unwrap()
}
pub(super) async fn drain(env: &Arc<Context>, now: i64) -> pool::Report {
    let provider = chat::provider_id(env.client.endpoint());
    queue::capacity::resume(&env.store, &provider, || now).unwrap();
    pool::run_ready(
        Arc::clone(&env.store),
        Arc::new(pool::Settings {
            provider,
            concurrency: 1,
            max_jobs: 10,
            max_claim_tokens: i64::MAX,
            budget: chat::limits(&env.config.llm).unwrap().budget,
            max_output_tokens: 777,
            timeout: Duration::from_secs(3),
            heartbeat: Duration::from_millis(20),
        }),
        Arc::new(Arc::clone(env)),
        move || now,
    )
    .await
    .unwrap()
}
fn reply() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "choices":[{"message":{"content":"A verified fixture description."}}],
        "usage":{"prompt_tokens":30,"completion_tokens":10,"total_tokens":40},
    }))
}
#[tokio::test]
async fn image_jobs_publish_from_l0_with_captions_and_repeat_without_calls() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(reply())
        .expect(2)
        .mount(&server)
        .await;
    let (_root, env) = fixture(&server.uri(), |_| {});
    register(&env, 4, "unreferenced.png", &png());
    assert_eq!(
        env.store
            .writer()
            .unwrap()
            .query_row(
                "SELECT count(*) FROM units WHERE file_id IN (2,4)",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
    let referenced = prepare(&env, 2);
    let unreferenced = prepare(&env, 4);
    assert!(matches!(
        referenced.schedule(2.0, 0).unwrap(),
        Scheduled::QueuedOrExisting(_)
    ));
    assert!(matches!(
        unreferenced.schedule(1.0, 0).unwrap(),
        Scheduled::QueuedOrExisting(_)
    ));
    let report = drain(&env, 0).await;
    assert_eq!(report.published, 2);
    assert!(report.errors.is_empty());
    let requests = server.received_requests().await.unwrap();
    for (index, request) in requests.iter().enumerate() {
        let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["model"], "fixture-vision");
        assert_eq!(body["max_tokens"], 200);
        let instruction = body["messages"][0]["content"][0]["text"].as_str().unwrap();
        assert_eq!(instruction.contains("Request flow"), index == 0);
        assert!(
            body["messages"][0]["content"][1]["image_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,")
        );
    }
    {
        let db = env.store.writer().unwrap();
        assert_eq!(
            db.query_row("SELECT sum(tokens) FROM usage", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            80
        );
        assert_eq!(
            db.query_row("SELECT count(*) FROM units WHERE kind='summary'", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            2
        );
        assert_eq!(db.query_row("SELECT count(*) FROM unit_sources us JOIN units s ON s.id=us.unit_id WHERE s.source_key='summary:image:2' AND us.source_kind='file' AND us.source_id=3",[],|r|r.get::<_,i64>(0)).unwrap(),1);
        assert_eq!(db.query_row("SELECT count(*) FROM unit_sources us JOIN units s ON s.id=us.unit_id WHERE s.source_key='summary:image:4' AND us.source_kind='file' AND us.source_id=3",[],|r|r.get::<_,i64>(0)).unwrap(),0);
    }
    for id in [2, 4] {
        assert!(matches!(
            prepare(&env, id).schedule(1.0, 1).unwrap(),
            Scheduled::Current(_)
        ));
    }
    assert_eq!(drain(&env, 1).await.claimed, 0);
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn real_text_and_image_jobs_share_one_daily_call_budget() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(reply())
        .expect(1)
        .mount(&server)
        .await;
    let (_root, env) = fixture(&server.uri(), |c| c.llm.max_calls_per_day = 1);
    let text = env
        .store
        .read_snapshot(|tx| env.symbol(tx, 1))
        .unwrap()
        .unwrap()
        .unwrap();
    let image = prepare(&env, 2);
    assert_eq!(text.provider, image.provider);
    text.schedule(2.0, 0).unwrap();
    let id = match image.schedule(1.0, 0).unwrap() {
        Scheduled::QueuedOrExisting(id) => id,
        other => panic!("{other:?}"),
    };
    let report = drain(&env, 0).await;
    assert_eq!((report.published, report.deferred), (1, 1));
    assert!(report.errors.is_empty());
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["model"], "fixture-text");
    let db = env.store.writer().unwrap();
    let state: (String, i64) = db
        .query_row("SELECT state,attempts FROM jobs WHERE id=?1", [id], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(state, ("queued".into(), 0));
    assert_eq!(
        db.query_row("SELECT count(*) FROM units WHERE file_id=2", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[tokio::test]
async fn image_unit_cap_is_a_persisted_pause_without_dispatch_or_attempts() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(reply())
        .expect(0)
        .mount(&server)
        .await;
    let (_root, env) = fixture(&server.uri(), |c| c.limits.max_units_per_file = 0);
    let job = prepare(&env, 2);
    let id = match job.schedule(1.0, 0).unwrap() {
        Scheduled::Paused(id) => id,
        other => panic!("{other:?}"),
    };
    let report = drain(&env, 0).await;
    assert_eq!(report.published, 0);
    assert!(report.errors.is_empty());
    assert!(server.received_requests().await.unwrap().is_empty());
    let db = env.store.writer().unwrap();
    let state: (String, i64, i64, String) = db
        .query_row(
            "SELECT state,attempts,not_before,last_error FROM jobs WHERE id=?1",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .unwrap();
    assert_eq!(
        state,
        (
            "queued".into(),
            0,
            queue::capacity::PAUSED,
            queue::capacity::Reason::FileUnits.message().into()
        )
    );
    assert_eq!(
        db.query_row("SELECT count(*) FROM usage_reservations WHERE state IN ('reserved','dispatched','reconciled')", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[tokio::test]
async fn image_privacy_changes_before_dispatch_make_no_call() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(reply())
        .expect(0)
        .mount(&server)
        .await;
    let (_root, env) = fixture(&server.uri(), |_| {});
    let job = prepare(&env, 2);
    let id = match job.schedule(1.0, 0).unwrap() {
        Scheduled::QueuedOrExisting(id) => id,
        other => panic!("{other:?}"),
    };
    let mut changed = (*env.config).clone();
    changed.treatments.insert(
        0,
        crate::config::Treatment {
            name: "private-image".into(),
            patterns: vec!["diagram.png".into()],
            max_level: 1,
            never_send: true,
            ..Default::default()
        },
    );
    fs::write(
        env.config_root.join(".deepkit/config.toml"),
        toml::to_string(&changed).unwrap(),
    )
    .unwrap();
    let report = drain(&env, 0).await;
    assert_eq!((report.published, report.deferred), (0, 1));
    assert!(server.received_requests().await.unwrap().is_empty());
    assert_eq!(
        env.store
            .writer()
            .unwrap()
            .query_row("SELECT attempts FROM jobs WHERE id=?1", [id], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[tokio::test]
async fn changed_image_or_caption_after_response_discards_output_but_keeps_usage() {
    for caption in [true, false] {
        let server = MockServer::start().await;
        let (_root, env) = fixture(&server.uri(), |_| {});
        let job = prepare(&env, 2);
        job.schedule(1.0, 0).unwrap();
        let changed = env
            .root
            .join(if caption { "README.md" } else { "diagram.png" });
        Mock::given(method("POST"))
            .respond_with(move |_: &wiremock::Request| {
                fs::write(&changed, b"Changed source after dispatch").unwrap();
                reply()
            })
            .expect(1)
            .mount(&server)
            .await;
        let report = drain(&env, 0).await;
        assert_eq!((report.published, report.discarded), (0, 1));
        let db = env.store.writer().unwrap();
        assert_eq!(
            db.query_row("SELECT sum(tokens) FROM usage", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            40
        );
        assert_eq!(
            db.query_row(
                "SELECT count(*) FROM units WHERE kind IN ('image','summary')",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
            0
        );
    }
}

#[tokio::test]
async fn image_summary_enters_embedding_inventory_and_loses_authority_after_caption_change() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(reply())
        .expect(1)
        .mount(&server)
        .await;
    let (_root, env) = fixture(&server.uri(), |_| {});
    prepare(&env, 2).schedule(1.0, 0).unwrap();
    assert_eq!(drain(&env, 0).await.published, 1);
    let (summary, parent): (i64, i64) = env
        .store
        .writer()
        .unwrap()
        .query_row(
            "SELECT id,parent_unit FROM units WHERE source_key='summary:image:2'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    let prepare_summary = || {
        env.store
            .read_snapshot(|tx| {
                queue::destination::prepare_file(
                    tx,
                    &env.root,
                    &env.config_root,
                    &env.config,
                    summary,
                )
            })
            .unwrap()
            .unwrap()
    };
    let (prepared, _) = prepare_summary().expect("image summary must be embeddable");
    assert_eq!(prepared.enqueue(1.0).target_id, summary);
    assert!(
        env.store
            .embedding_catalogue_kinds(false, false, 0, u64::MAX)
            .unwrap()
            .contains(&crate::store::UnitKind::Summary)
    );
    let mut cursor = queue::catalogue::Cursor::start(&env.store).unwrap();
    let mut found = false;
    while !cursor.finished() {
        for candidate in cursor.next(&env.store, 100).unwrap() {
            assert_ne!(
                candidate.unit, parent,
                "L0 image is not a text embedding candidate"
            );
            found |= candidate.unit == summary;
        }
    }
    assert!(found);
    let source = env
        .store
        .read_snapshot(|tx| {
            queue::source::resolve_rerank_with_work(
                tx,
                &env.root,
                &env.config_root,
                &env.config,
                summary,
                Default::default(),
            )
        })
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(source.document.contains("A verified fixture description."));
    assert!(source.document.contains("diagram.png"));
    fs::write(env.root.join("README.md"), "# Changed reference context").unwrap();
    assert!(prepare_summary().is_none());
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn image_host_selects_referenced_cohort_first_and_repeat_is_current() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(reply())
        .expect(2)
        .mount(&server)
        .await;
    let (_root, env) = fixture(&server.uri(), |c| {
        c.enrich.top_symbols = 0;
        c.enrich.prs_last_n = 0;
        c.enrich.modules_min_files = 1_000_000;
    });
    // Its earlier path must not put an unreferenced image in the first cohort.
    register(&env, 4, "a-unreferenced.png", &png());
    let mut host = queue::enrichment_host::Host::from_context(Arc::clone(&env), 0).unwrap();
    let bound = host.catalogue_pass_bound();
    let first = host.pass(|| 0).await.unwrap();
    assert!(first.error.is_none());
    assert_eq!((first.visited, first.execution.published), (1, 1));
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
    let first_body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert!(
        first_body["messages"][0]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("Request flow")
    );
    let mut published = first.execution.published;
    for _ in 1..bound {
        let report = host.pass(|| 0).await.unwrap();
        assert!(report.error.is_none());
        published += report.execution.published;
        if host.catalogue_finished().unwrap() {
            break;
        }
    }
    assert_eq!(published, 2);
    assert!(host.catalogue_finished().unwrap());
    let mut repeat = queue::enrichment_host::Host::from_context(Arc::clone(&env), 1).unwrap();
    let mut current = 0;
    for _ in 0..repeat.catalogue_pass_bound() {
        let report = repeat.pass(|| 1).await.unwrap();
        assert!(report.error.is_none());
        current += report.current;
        assert_eq!(report.execution.claimed, 0);
        if repeat.catalogue_finished().unwrap() {
            break;
        }
    }
    assert_eq!(current, 2);
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[test]
fn disabled_image_enrichment_has_no_selection_or_dispatch_authority() {
    let (_root, env) = fixture("http://127.0.0.1:9", |c| c.enrich.images = false);
    assert!(
        env.store
            .read_snapshot(|tx| queue::visual_source::select_images(
                tx,
                &env.root,
                &env.config_root,
                &env.config,
                Default::default()
            ))
            .unwrap()
            .unwrap()
            .is_empty()
    );
    assert!(
        env.store
            .read_snapshot(|tx| env.prepare_job(tx, Kind::DescribeImage, 2))
            .unwrap()
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn capped_host_requires_current_discovery_and_finished_admitted_work() {
    for condition in [
        "ready",
        "non_unicode",
        "missing",
        "old_generation",
        "different_cap",
        "pending_file",
        "pending_graph",
        "unfinished_fingerprint",
    ] {
        let (_root, env) = fixture("http://127.0.0.1:9", |c| {
            c.enrich.top_symbols = 0;
            c.enrich.prs_last_n = 0;
            c.enrich.modules_min_files = 1_000_000;
        });
        let generation = env.store.l1_identity().unwrap().scan + 1;
        env.store
            .set_meta("l1_scan_generation", &generation.to_string())
            .unwrap();
        env.store
            .set_meta(
                "graph_pending",
                if condition == "pending_graph" {
                    "1"
                } else {
                    "0"
                },
            )
            .unwrap();
        if condition != "missing" {
            env.store.set_meta("discovery_status", &json!({
                "generation": if condition=="old_generation" {generation-1} else {generation},
                "max_files": env.config.limits.max_files + u64::from(condition=="different_cap"),
                "max_files_reached": condition!="non_unicode",
                "skipped_non_unicode_paths": usize::from(condition=="non_unicode"),
                "scope":"last_whole_repository_scan",
            }).to_string()).unwrap();
        }
        if condition == "pending_file" {
            env.store
                .writer()
                .unwrap()
                .execute("UPDATE files SET l1_pending=1 WHERE id=1", [])
                .unwrap();
        }
        if condition == "unfinished_fingerprint" {
            env.store
                .writer()
                .unwrap()
                .execute("UPDATE files SET l1_fingerprint=NULL WHERE id=1", [])
                .unwrap();
        }
        let host = queue::enrichment_host::Host::from_context(Arc::clone(&env), 0);
        let allowed = matches!(condition, "ready" | "non_unicode");
        assert_eq!(host.is_ok(), allowed, "{condition}");
        if let Ok(mut host) = host {
            // The exception is rechecked at dispatch, not cached as permission.
            env.store.set_meta("graph_pending", "1").unwrap();
            assert!(host.pass(|| 0).await.is_err());
        }
        assert_eq!(env.store.l1_identity().unwrap().scan, generation);
        assert_ne!(env.store.l1_identity().unwrap().completed, generation);
        assert_eq!(
            env.store
                .read_snapshot(|tx| tx.query_row(
                    "SELECT count(*) FROM usage_reservations",
                    [],
                    |r| r.get::<_, i64>(0)
                ))
                .unwrap()
                .unwrap(),
            0
        );
    }
}
