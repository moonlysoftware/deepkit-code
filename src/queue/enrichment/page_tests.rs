//! Shared page worker completion and durable source/output proofs; v1 §6.2/§8.
use super::*;
use serde_json::json;
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

fn fixture(url: &str) -> (tempfile::TempDir, Arc<Context>) {
    let (root, env) = image_tests::fixture(url, |config| {
        config.enrich.scanned_pages = true;
    });
    image_tests::register(
        &env,
        4,
        "scan.pdf",
        include_bytes!("../../../fixtures/docs/scanned-jpeg.pdf"),
    );
    env.store
        .writer()
        .unwrap()
        .execute(
            "UPDATE files SET needs_vlm=1,l1_deferred_reason='needs_vlm',l1_pending=0 WHERE id=4",
            [],
        )
        .unwrap();
    (root, env)
}
fn prepare(env: &Arc<Context>) -> Job {
    env.store
        .read_snapshot(|tx| env.prepare_page_with_work(tx, 4, 1, crate::sync::l1::Work::default()))
        .unwrap()
        .unwrap()
        .unwrap()
}

#[tokio::test]
async fn scanned_page_source_or_privacy_change_after_enqueue_prevents_dispatch() {
    for private in [false, true] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500))
            .expect(0)
            .mount(&server)
            .await;
        let (_root, env) = fixture(&server.uri());
        prepare(&env).schedule(1.0, 0).unwrap();
        if private {
            let mut config = (*env.config).clone();
            config.treatments.insert(
                0,
                crate::config::Treatment {
                    name: "private-page".into(),
                    patterns: vec!["**/*.pdf".into()],
                    never_send: true,
                    max_level: 3,
                    chunker: Some("pdf".into()),
                    enrich: vec!["vlm_page".into()],
                },
            );
            std::fs::write(
                env.root.join(".deepkit/config.toml"),
                toml::to_string(&config).unwrap(),
            )
            .unwrap();
        } else {
            std::fs::write(env.root.join("scan.pdf"), b"changed page source").unwrap();
        }
        image_tests::drain(&env, 0).await;
        assert!(server.received_requests().await.unwrap().is_empty());
        let db = env.store.writer().unwrap();
        assert_eq!(
            db.query_row("SELECT sum(attempts) FROM jobs", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(
            db.query_row("SELECT count(*) FROM units WHERE file_id=4", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }
}

#[tokio::test]
async fn scanned_page_output_cap_remembers_required_room_without_repeating_calls() {
    let server = MockServer::start().await;
    let content = format!(
        "# First\n\n{}\n\n# Second\n\n{}",
        "Firstsection words. ".repeat(800),
        "Secondsection words. ".repeat(800)
    );
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices":[{"finish_reason":"stop","message":{"content":content}}],
            "usage":{"prompt_tokens":30,"completion_tokens":20,"total_tokens":50}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let (_root, env) = image_tests::fixture(&server.uri(), |c| c.limits.max_units_per_file = 2);
    image_tests::register(
        &env,
        4,
        "scan.pdf",
        include_bytes!("../../../fixtures/docs/scanned-jpeg.pdf"),
    );
    env.store
        .writer()
        .unwrap()
        .execute(
            "UPDATE files SET needs_vlm=1,l1_deferred_reason='needs_vlm',l1_pending=0 WHERE id=4",
            [],
        )
        .unwrap();
    prepare(&env).schedule(1.0, 0).unwrap();
    for now in 0..3 {
        let report = image_tests::drain(&env, now).await;
        assert_eq!(report.published, 0);
        let db = env.store.writer().unwrap();
        let (required,attempts,paused):(u32,i64,bool)=db.query_row("SELECT required_l3_units,attempts,not_before=9223372036854775807 FROM jobs WHERE kind='enrich:vlm_page'",[],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).unwrap();
        assert!(required > 2);
        assert_eq!(attempts, 1);
        assert!(paused);
        assert_eq!(
            db.query_row("SELECT sum(tokens) FROM usage", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            50
        );
        assert_eq!(
            db.query_row("SELECT count(*) FROM units WHERE file_id=4", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }
}

#[tokio::test]
async fn scanned_page_host_selects_admitted_pages_in_order_and_repeats_without_calls() {
    let server = MockServer::start().await;
    Mock::given(method("POST")).respond_with(ResponseTemplate::new(200).set_body_json(json!({
        "choices":[{"finish_reason":"stop","message":{"content":"# Hostpage\n\nPublic page transcription."}}],
        "usage":{"prompt_tokens":30,"completion_tokens":20,"total_tokens":50}
    }))).expect(2).mount(&server).await;
    let (root, env) = image_tests::fixture(&server.uri(), |c| {
        c.enrich.images = false;
        c.enrich.top_symbols = 0;
        c.enrich.prs_last_n = 0;
        c.enrich.modules_min_files = 1_000_000;
        c.limits.max_pdf_pages = 2;
    });
    let generated = tempfile::tempdir().unwrap();
    let output = std::process::Command::new("python3")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/scripts/corpus/make-scan-fixtures.py"
        ))
        .arg("--output")
        .arg(generated.path())
        .args(["--case", "jpeg", "--pages", "3"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = std::fs::read(generated.path().join("scanned-jpeg.pdf")).unwrap();
    image_tests::register(&env, 4, "scan.pdf", &bytes);
    env.store.writer().unwrap().execute("UPDATE files SET needs_vlm=1,l1_deferred_reason='max_pdf_pages',l1_pending=0 WHERE id=4",[]).unwrap();
    let selection = env
        .store
        .read_snapshot(|tx| {
            crate::queue::page_source::select(
                tx,
                root.path(),
                root.path(),
                &env.config,
                Default::default(),
            )
        })
        .unwrap()
        .unwrap();
    assert_eq!(
        selection
            .iter()
            .map(|c| (c.file, c.page))
            .collect::<Vec<_>>(),
        [(4, 1), (4, 2)]
    );
    for round in 0..2 {
        let mut host =
            crate::queue::enrichment_host::Host::from_context(Arc::clone(&env), round).unwrap();
        let mut published = 0;
        let mut current = 0;
        for _ in 0..host.catalogue_pass_bound() {
            let report = host.pass(move || round).await.unwrap();
            assert!(report.error.is_none());
            assert!(report.execution.errors.is_empty());
            published += report.execution.published;
            current += report.current;
            if host.catalogue_finished().unwrap() {
                break;
            }
        }
        assert!(host.catalogue_finished().unwrap());
        assert_eq!(
            (published, current),
            if round == 0 { (2, 0) } else { (0, 2) }
        );
    }
    let db = env.store.writer().unwrap();
    let pages = db
        .prepare("SELECT page FROM jobs WHERE kind='enrich:vlm_page' ORDER BY id")
        .unwrap()
        .query_map([], |r| r.get::<_, u32>(0))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(pages, [1, 2]);
    assert_eq!(
        db.query_row("SELECT count(*) FROM page_publications", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        2
    );
}

#[tokio::test]
async fn scanned_page_host_keeps_no_renderer_visible_and_processes_later_pages() {
    let server = MockServer::start().await;
    Mock::given(method("POST")).respond_with(ResponseTemplate::new(200).set_body_json(json!({
        "choices":[{"finish_reason":"stop","message":{"content":"# Laterpage\n\nSecond page remains processable."}}],
        "usage":{"prompt_tokens":30,"completion_tokens":20,"total_tokens":50}
    }))).expect(1).mount(&server).await;
    let (_root, env) = image_tests::fixture(&server.uri(), |c| {
        c.enrich.images = false;
        c.enrich.top_symbols = 0;
        c.enrich.prs_last_n = 0;
        c.enrich.modules_min_files = 1_000_000;
    });
    let generated = tempfile::tempdir().unwrap();
    let output = std::process::Command::new("python3")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/scripts/corpus/make-scan-fixtures.py"
        ))
        .arg("--output")
        .arg(generated.path())
        .args(["--case", "first-overlay", "--pages", "2"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    image_tests::register(
        &env,
        4,
        "mixed.pdf",
        &std::fs::read(generated.path().join("scanned-first-overlay.pdf")).unwrap(),
    );
    env.store
        .writer()
        .unwrap()
        .execute(
            "UPDATE files SET needs_vlm=1,l1_deferred_reason='needs_vlm',l1_pending=0 WHERE id=4",
            [],
        )
        .unwrap();
    let mut host = crate::queue::enrichment_host::Host::from_context(Arc::clone(&env), 0).unwrap();
    let mut skipped = 0;
    let mut published = 0;
    for _ in 0..host.catalogue_pass_bound() {
        let report = host.pass(|| 0).await.unwrap();
        assert!(report.error.is_none());
        assert!(report.execution.errors.is_empty());
        skipped += report.skipped;
        published += report.execution.published;
        if host.catalogue_finished().unwrap() {
            break;
        }
    }
    assert_eq!((skipped, published), (1, 1));
    let db = env.store.writer().unwrap();
    assert_eq!(
        db.query_row("SELECT l1_deferred_reason FROM files WHERE id=4", [], |r| r
            .get::<_, String>(0))
            .unwrap(),
        "no_renderer"
    );
    assert_eq!(
        db.query_row(
            "SELECT page FROM page_publications WHERE file_id=4",
            [],
            |r| r.get::<_, u32>(0)
        )
        .unwrap(),
        2
    );
}

#[tokio::test]
async fn scanned_page_worker_publishes_complete_markdown_and_repeats_without_calls() {
    let server = MockServer::start().await;
    Mock::given(method("POST")).respond_with(ResponseTemplate::new(200).set_body_json(json!({
        "choices":[{"finish_reason":"stop","message":{"content":"# Scan title\n\nDurablepageword describes this scanned document.\n\n| Name | Value |\n| --- | --- |\n| Fixture | 42 |"}}],
        "usage":{"prompt_tokens":30,"completion_tokens":20,"total_tokens":50}
    }))).expect(1).mount(&server).await;
    let (_root, env) = fixture(&server.uri());
    let job = prepare(&env);
    assert!(matches!(
        job.schedule(1.0, 0).unwrap(),
        Scheduled::QueuedOrExisting(_)
    ));
    let report = image_tests::drain(&env, 0).await;
    assert_eq!(report.published, 1, "{:?}", report.errors);
    assert!(report.errors.is_empty());
    let anchor = match prepare(&env).schedule(1.0, 1).unwrap() {
        Scheduled::Current(id) => id,
        _ => panic!("complete page must be current"),
    };
    {
        let db = env.store.writer().unwrap();
        let (children,invalid):(i64,i64)=db.query_row("SELECT count(*),coalesce(sum(kind!='doc_section' OR ladder_level!=3 OR file_id!=4 OR line_start<1),0) FROM units WHERE parent_unit=?1",[anchor],|r|Ok((r.get(0)?,r.get(1)?))).unwrap();
        assert!(children > 0);
        assert_eq!(invalid, 0);
        assert_eq!(
            db.query_row(
                "SELECT count(*) FROM fts_units WHERE fts_units MATCH 'Durablepageword'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
            1
        );
        assert_eq!(
            db.query_row("SELECT sum(tokens) FROM usage", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            50
        );
        assert_eq!(
            db.query_row("SELECT count(*) FROM units WHERE kind='summary'", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }
    assert_eq!(image_tests::drain(&env, 1).await.published, 0);
    verify_page_source_and_refresh(&env, anchor);
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["model"], "fixture-vision");
    assert_eq!(body["max_tokens"], 8192);
    assert!(
        body["messages"][0]["content"][1]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/jpeg;base64,")
    );
}

fn verify_page_source_and_refresh(env: &Arc<Context>, anchor: i64) {
    let section: i64 = env
        .store
        .writer()
        .unwrap()
        .query_row(
            "SELECT id FROM units WHERE parent_unit=?1 ORDER BY id LIMIT 1",
            [anchor],
            |r| r.get(0),
        )
        .unwrap();
    let resolve = || {
        env.store
            .read_snapshot(|tx| {
                crate::queue::source::resolve(tx, &env.root, &env.config_root, &env.config, section)
            })
            .unwrap()
            .unwrap()
    };
    assert_eq!(
        resolve().unwrap().occurrence.kind,
        crate::store::UnitKind::DocSection
    );
    assert!(
        env.store
            .embedding_catalogue_kinds(false, false, 0, u64::MAX)
            .unwrap()
            .contains(&crate::store::UnitKind::DocSection)
    );
    let mut cursor = crate::queue::catalogue::Cursor::start(&env.store).unwrap();
    let mut found = false;
    while !cursor.finished() {
        for candidate in cursor.next(&env.store, 100).unwrap() {
            assert_ne!(
                candidate.unit, anchor,
                "PDF page pixels have no text vector"
            );
            found |= candidate.unit == section;
        }
    }
    assert!(found);
    let (raw, fp): ([u8; 32], [u8; 32]) = env
        .store
        .writer()
        .unwrap()
        .query_row("SELECT blake3,l1_desired FROM files WHERE id=4", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    let fence = crate::store::l1::Fence {
        generation: env.store.l1_identity().unwrap().scan,
        file_id: 4,
        path: "scan.pdf",
        content_hash: &raw,
        fingerprint: &fp,
    };
    let completion = crate::store::l1::Completion {
        truncated: false,
        needs_vlm: true,
        reason: Some("needs_vlm"),
    };
    env.store
        .replace_file_content(None, fence, &[], completion, || Ok(()))
        .unwrap();
    assert!(env.store.unit_by_id(section).unwrap().is_some());
    assert!(resolve().is_some());
    // Neither an extra private dependency nor a surviving partial manifest can
    // bypass the dedicated page resolver (including the generic PDF-cap path).
    env.store.writer().unwrap().execute("INSERT INTO unit_sources(unit_id,source_kind,source_id,role) VALUES(?1,'file',3,'content')",[section]).unwrap();
    assert!(resolve().is_none());
    env.store
        .writer()
        .unwrap()
        .execute(
            "DELETE FROM unit_sources WHERE unit_id=?1 AND source_kind='file' AND source_id=3",
            [section],
        )
        .unwrap();
    env.store
        .writer()
        .unwrap()
        .execute(
            "UPDATE page_publications SET sections=sections+1 WHERE parent_unit=?1",
            [anchor],
        )
        .unwrap();
    assert!(resolve().is_none());
    env.store
        .writer()
        .unwrap()
        .execute(
            "UPDATE files SET l1_deferred_reason='max_pdf_pages' WHERE id=4",
            [],
        )
        .unwrap();
    assert!(resolve().is_none());
    env.store
        .writer()
        .unwrap()
        .execute(
            "UPDATE page_publications SET sections=sections-1 WHERE parent_unit=?1",
            [anchor],
        )
        .unwrap();
    assert!(resolve().is_some());
    std::fs::write(env.root.join("scan.pdf"), b"changed source bytes").unwrap();
    assert!(resolve().is_none());
    env.store
        .writer()
        .unwrap()
        .execute(
            "UPDATE files SET blake3=?1,l1_desired=?2 WHERE id=4",
            rusqlite::params![[9_u8; 32].as_slice(), fp.as_slice()],
        )
        .unwrap();
    env.store
        .replace_file_content(
            None,
            crate::store::l1::Fence {
                content_hash: &[9; 32],
                ..fence
            },
            &[],
            completion,
            || Ok(()),
        )
        .unwrap();
    assert!(env.store.unit_by_id(anchor).unwrap().is_none());
    assert!(env.store.unit_by_id(section).unwrap().is_none());
    assert_eq!(
        env.store
            .writer()
            .unwrap()
            .query_row("SELECT count(*) FROM page_publications", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[tokio::test]
async fn scanned_page_worker_keeps_usage_but_never_publishes_incomplete_output() {
    for finish in [json!("length"), json!(null)] {
        let server = MockServer::start().await;
        Mock::given(method("POST")).respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices":[{"finish_reason":finish,"message":{"content":"Partialpageword must never be published."}}],
            "usage":{"prompt_tokens":30,"completion_tokens":20,"total_tokens":50}
        }))).expect(1).mount(&server).await;
        let (_root, env) = fixture(&server.uri());
        prepare(&env).schedule(1.0, 0).unwrap();
        let report = image_tests::drain(&env, 0).await;
        assert_eq!(report.published, 0);
        {
            let db = env.store.writer().unwrap();
            let (state, reason, attempts): (String, String, i64) = db
                .query_row(
                    "SELECT state,last_error,attempts FROM jobs WHERE kind='enrich:vlm_page'",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )
                .unwrap();
            assert_eq!(
                (state, reason, attempts),
                (
                    "dead".into(),
                    "incomplete page transcription; diagnosis required".into(),
                    1
                )
            );
            assert_eq!(
                db.query_row("SELECT count(*) FROM units WHERE file_id=4", [], |r| r
                    .get::<_, i64>(0))
                    .unwrap(),
                0
            );
            assert_eq!(
                db.query_row("SELECT sum(tokens) FROM usage", [], |r| r.get::<_, i64>(0))
                    .unwrap(),
                50
            );
        }
        assert_eq!(image_tests::drain(&env, 1).await.published, 0);
    }
}
