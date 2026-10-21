//! Recorded public CLI privacy boundaries; v1.md §5 and §7.3.
use base64::{Engine, engine::general_purpose::STANDARD};
use rusqlite::Connection;
use serde_json::{Value, json};
use std::{fs, io::Cursor, path::Path, time::Duration};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

const PRIVATE: &str = "SyntheticProtectedOrchid";
const PUBLIC: &str = "PublicSearchableMeadow";
const RETIRED_ROLE: &str = "WithdrawnOrchidRole";

async fn cli(root: &Path, args: &[&str]) -> Value {
    let mut command = tokio::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"));
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("DKC_") {
            command.env_remove(key);
        }
    }
    command
        .current_dir(root)
        .args(args)
        .env("XDG_CONFIG_HOME", root.join("absent-user-config"))
        .env("DKC_MODELS_DIR", root.join("unused-models"))
        .env("DKC_PRIVACY_FIXTURE_KEY", "synthetic-fixture-only")
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(60), command.output())
        .await
        .unwrap()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

async fn gateway() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST")).and(path("/api/embed"))
        .respond_with(|request: &wiremock::Request| {
            let body: Value = request.body_json().unwrap();
            ResponseTemplate::new(200).set_body_json(json!({
                "embeddings":vec![vec![1.0,0.0];body["input"].as_array().unwrap().len()], "prompt_eval_count":1
            }))
        }).mount(&server).await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(|request: &wiremock::Request| {
            let body: Value = request.body_json().unwrap();
            let content = if body.to_string().contains(PRIVATE) {
                format!("{RETIRED_ROLE}\n\nExplanation from the initially public fixture.")
            } else {
                "Public meadow\n\nVerified public source explanation.".to_owned()
            };
            ResponseTemplate::new(200).set_body_json(json!({
                "choices":[{"finish_reason":"stop","message":{"content":content}}],
                "usage":{"prompt_tokens":20,"completion_tokens":10,"total_tokens":30}
            }))
        })
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/rerank"))
        .respond_with(|request: &wiremock::Request| {
            let body: Value = request.body_json().unwrap();
            let results: Vec<_> = body["documents"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(index, _)| json!({"index":index,"relevance_score":0.9}))
                .collect();
            ResponseTemplate::new(200).set_body_json(json!({"results":results}))
        })
        .mount(&server)
        .await;
    server
}

fn config(root: &Path, url: &str, protected: bool, zero_budget: bool) {
    let calls = if zero_budget { 0 } else { 10000 };
    let policy = if protected {
        "\n[[treatments]]\nname='protected'\nmatch=['secrets/**']\nmax_level=3\nnever_send=true\n"
    } else {
        ""
    };
    fs::write(
        root.join(".deepkit/config.toml"),
        format!(
            r#"
[embedder]
profile='ollama'
dims=2
[embedder.ollama]
url='{url}'
[rerank]
mode='on'
provider='remote'
[rerank.remote]
url='{url}'
model='fixture'
api_key_env='DKC_PRIVACY_FIXTURE_KEY'
[llm]
enabled=true
url='{url}/v1'
model='fixture-text'
vlm_model='fixture-vision'
api_key_env='DKC_PRIVACY_FIXTURE_KEY'
rpm=10000
tpm=10000000
max_calls_per_day={calls}
max_tokens_per_day=10000000
[enrich]
modules_min_files=1
[queue]
max_schedule_per_pass=1000
max_jobs_per_pass=1000
embedding_timeout='5s'
[[queue.budgets]]
profile='ollama'
url='{url}'
daily_calls={calls}
[limits]
query_timeout_ms=10000
rerank_timeout_ms=5000
{policy}
[[treatments]]
name='fixture-code'
match=['**/*.rs']
max_level=3
enrich=['describe_symbol']
[[treatments]]
name='fixture-pages'
match=['**/*.pdf']
max_level=3
enrich=['vlm_page']
[[treatments]]
name='fixture-images'
match=['**/*.png']
max_level=3
enrich=['describe_image']
"#
        ),
    )
    .unwrap();
}

fn visual(root: &Path, directory: &str, private: bool) {
    let color = if private { [0, 0, 240] } else { [0, 240, 0] };
    let image =
        image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(32, 32, image::Rgb(color)));
    image
        .save(root.join(directory).join("diagram.png"))
        .unwrap();
    let mut jpeg = Cursor::new(Vec::new());
    image.write_to(&mut jpeg, image::ImageFormat::Jpeg).unwrap();
    let stream = b"q 32 0 0 32 0 0 cm /Scan Do Q\n";
    let mut image_object = format!("<< /Type /XObject /Subtype /Image /Width 32 /Height 32 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {} >>\nstream\n",jpeg.get_ref().len()).into_bytes();
    image_object.extend_from_slice(jpeg.get_ref());
    image_object.extend_from_slice(b"\nendstream");
    let objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 32 32] /Resources << /XObject << /Scan 5 0 R >> >> /Contents 4 0 R >>".to_vec(),
        [format!("<< /Length {} >>\nstream\n",stream.len()).as_bytes(),stream,b"endstream"].concat(),
        image_object,
    ];
    let mut pdf = b"%PDF-1.7\n".to_vec();
    let mut offsets = Vec::new();
    for (index, object) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{} 0 obj\n", index + 1).as_bytes());
        pdf.extend_from_slice(object);
        pdf.extend_from_slice(b"\nendobj\n");
    }
    let xref = pdf.len();
    pdf.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
    for offset in offsets {
        pdf.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    pdf.extend_from_slice(
        format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
    );
    fs::write(root.join(directory).join("scan.pdf"), pdf).unwrap();
}

fn assert_public_json(value: &Value, visual_inputs: &mut usize) {
    match value {
        Value::String(text) => {
            assert!(!text.contains(PRIVATE), "protected marker in outbound JSON");
            if text.starts_with("data:image/") {
                let (_, data) = text.split_once(";base64,").unwrap();
                let image = image::load_from_memory(&STANDARD.decode(data).unwrap())
                    .unwrap()
                    .to_rgb8();
                assert!(
                    !image.pixels().any(|p| p[2] > 180 && p[0] < 60 && p[1] < 60),
                    "protected blue pixels were sent"
                );
                assert!(
                    image.pixels().any(|p| p[1] > 180 && p[0] < 60 && p[2] < 60),
                    "public visual control missing"
                );
                *visual_inputs += 1;
            }
        }
        Value::Array(values) => {
            for item in values {
                assert_public_json(item, visual_inputs);
            }
        }
        Value::Object(values) => {
            for (key, item) in values {
                assert!(!key.contains(PRIVATE));
                assert_public_json(item, visual_inputs);
            }
        }
        _ => {}
    }
}

fn git(root: &Path, args: &[&str]) {
    let output = std::process::Command::new("git")
        .current_dir(root)
        .args([
            "-c",
            "user.name=Privacy Fixture",
            "-c",
            "user.email=privacy@example.invalid",
        ])
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn fixture(root: &Path) {
    for directory in [".deepkit", "public", "secrets", "control", "visual-control"] {
        fs::create_dir(root.join(directory)).unwrap();
    }
    fs::write(
        root.join("control/code.rs"),
        "pub fn isolated_control() -> bool { true }\n",
    )
    .unwrap();
    fs::write(
        root.join("control/README.md"),
        "# Isolated public control\n\nThe isolated helper returns a boolean.\n",
    )
    .unwrap();
    visual(root, "visual-control", false);
    fs::create_dir(root.join("secrets/roles")).unwrap();
    fs::write(
        root.join("secrets/roles/role.rs"),
        format!("pub fn {PRIVATE}_role() -> bool {{ true }}\n"),
    )
    .unwrap();
    fs::write(
        root.join("secrets/roles/README.md"),
        format!("# Role fixture\n\n{PRIVATE} directory responsibility.\n"),
    )
    .unwrap();
    for (directory, marker) in [("public", PUBLIC), ("secrets", PRIVATE)] {
        fs::write(
            root.join(directory).join("code.rs"),
            format!("/// {marker} documentation.\npub fn {marker}() -> bool {{ true }}\n"),
        )
        .unwrap();
        fs::write(
            root.join(directory).join("README.md"),
            format!("# {marker}\n\n{marker} documentation.\n\n![{marker} caption](diagram.png)\n"),
        )
        .unwrap();
        visual(root, directory, directory == "secrets");
    }
    git(root, &["init", "-q"]);
    git(
        root,
        &["add", "public", "secrets", "control", "visual-control"],
    );
    git(root, &["commit", "-qm", "Initial fixture"]);
    fs::write(
        root.join("public/extra.rs"),
        "pub fn public_extra() -> bool { true }\n",
    )
    .unwrap();
    git(root, &["add", "public"]);
    git(root, &["commit", "-qm", "Public change (#101)"]);
    fs::write(
        root.join("secrets/extra.rs"),
        format!("pub fn {PRIVATE}_caller() -> bool {{ crate::public::code::{PUBLIC}() }}\n"),
    )
    .unwrap();
    fs::write(root.join("public/mixed.rs"), "pub fn public_mixed() {}\n").unwrap();
    git(root, &["add", "public", "secrets"]);
    git(root, &["commit", "-qm", "Mixed change (#102)"]);
    // Untracked control has no mixed PR/history provenance and a body-only query.
    fs::write(
        root.join("query-control.md"),
        "# Search control\n\nMeadowRerankPositiveControl verifies the public route.\n",
    )
    .unwrap();
    fs::write(
        root.join("secrets/public-caption.md"),
        format!("# Caption\n\n![{PRIVATE} private caption](../public/diagram.png)\n"),
    )
    .unwrap();
}

#[tokio::test]
async fn never_send_filters_indexed_text_visuals_and_search_payloads() {
    let root = tempfile::tempdir().unwrap();
    fixture(root.path());
    let server = gateway().await;
    config(root.path(), &server.uri(), true, false);
    let indexed = cli(root.path(), &["index", "--level", "3", "--json"]).await;
    assert_eq!(indexed["enrichment"]["state"], "complete", "{indexed}");
    let db = Connection::open(root.path().join(".deepkit/code.db")).unwrap();
    let private_local: i64 = db
        .query_row(
            "SELECT count(*) FROM unit_contents WHERE instr(text,?1)>0",
            [PRIVATE],
            |r| r.get(0),
        )
        .unwrap();
    assert!(
        private_local > 0,
        "protected source must remain locally indexed"
    );
    let result = cli(
        root.path(),
        &[
            "search",
            "MeadowRerankPositiveControl",
            "--scope",
            "query-control.md",
            "--json",
        ],
    )
    .await;
    assert!(!result["hits"].as_array().unwrap().is_empty(), "{result}");
    let requests = server.received_requests().await.unwrap();
    let mut visuals = 0;
    for request in &requests {
        assert_public_json(&request.body_json::<Value>().unwrap(), &mut visuals);
    }
    assert!(
        visuals >= 2,
        "public image and scanned page must actually be sent"
    );
    for route in ["/api/embed", "/v1/chat/completions", "/rerank"] {
        assert!(
            requests.iter().any(|r| r.url.path() == route),
            "missing positive control: {route}"
        );
    }
    assert_job_kinds(&db, "done");
    assert_no_protected_dispatch_or_publication(&db);
    assert_accounting(root.path(), &requests).await;
}

fn assert_job_kinds(db: &Connection, state: &str) {
    for kind in [
        "describe_symbol",
        "summarize_pr",
        "summarize_module",
        "describe_image",
        "vlm_page",
    ] {
        let count: i64 = db
            .query_row(
                "SELECT count(*) FROM jobs WHERE kind=?1 AND state=?2",
                [format!("enrich:{kind}"), state.to_owned()],
                |r| r.get(0),
            )
            .unwrap();
        assert!(count > 0, "missing {state} positive job: {kind}");
    }
    if state == "done" {
        // A retired stale job may also be terminal. Require real nonempty
        // publications so retirement alone cannot satisfy positive controls.
        for kind in ["symbol", "pr", "module", "image"] {
            let count: i64 = db.query_row("SELECT count(*) FROM units s JOIN units p ON p.id=s.parent_unit JOIN unit_contents c ON c.hash=s.content_hash WHERE s.kind='summary' AND p.kind=?1 AND length(trim(c.text))>0",[kind],|r|r.get(0)).unwrap();
            assert!(count > 0, "missing positive summary publication: {kind}");
        }
        let pages: i64 = db
            .query_row(
                "SELECT count(*) FROM page_publications WHERE sections>0",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(pages > 0, "missing positive scanned-page publication");
    }
}

// Supplemental database proof for cases with no authorized private traffic.
// This deliberately does not run in the retained-role case: earlier authorized
// summaries and charges must remain stored after a later privacy change.
fn assert_no_protected_dispatch_or_publication(db: &Connection) {
    let protected = r#"WITH RECURSIVE
      private_files(id) AS (SELECT id FROM files WHERE path GLOB 'secrets/*'),
      private_symbols(id) AS (SELECT id FROM symbols WHERE file_id IN private_files),
      private_commits(id) AS (SELECT id FROM commits WHERE EXISTS(
        SELECT 1 FROM json_each(files_touched) WHERE value GLOB 'secrets/*')),
      private_units(id) AS (
        SELECT id FROM units WHERE file_id IN private_files
        UNION SELECT unit_id FROM commits WHERE id IN private_commits
        UNION SELECT unit_id FROM unit_sources WHERE
          (source_kind='file' AND source_id IN private_files) OR
          (source_kind='symbol' AND source_id IN private_symbols) OR
          (source_kind='commit' AND source_id IN private_commits)
        UNION SELECT s.unit_id FROM unit_sources s JOIN private_units p
          ON s.source_kind='unit' AND s.source_id=p.id
        UNION SELECT u.id FROM units u JOIN private_units p ON u.parent_unit=p.id
      ), private_jobs(id) AS (
        SELECT id FROM jobs WHERE
          (target_kind='file' AND target_id IN private_files) OR
          (target_kind='symbol' AND target_id IN private_symbols) OR
          (target_kind='unit' AND target_id IN private_units) OR
          (target_kind='pr' AND target_id IN
            (SELECT id FROM prs WHERE unit_id IN private_units))
      )"#;
    let counts: (i64, i64, i64, i64) = db
        .query_row(
            &format!(
                "{protected} SELECT
      (SELECT count(*) FROM private_files),
      (SELECT count(*) FROM usage_reservations WHERE job_id IN private_jobs
        AND state IN ('dispatched','reconciled')),
      (SELECT count(*) FROM units WHERE id IN private_units AND kind='summary' AND ladder_level>=3),
      (SELECT count(*) FROM page_publications WHERE file_id IN private_files)"
            ),
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .unwrap();
    assert!(counts.0 > 0, "private fixture must remain catalogued");
    assert_eq!(
        (counts.1, counts.2, counts.3),
        (0, 0, 0),
        "protected jobs dispatched or protected summaries/pages published"
    );
}

async fn assert_accounting(root: &Path, requests: &[wiremock::Request]) {
    let mut expected = std::collections::BTreeMap::<String, u64>::new();
    for request in requests {
        let body: Value = request.body_json().unwrap();
        let inputs = match request.url.path() {
            "/api/embed" => body["input"].as_array().unwrap().len() as u64,
            "/v1/chat/completions" => 1,
            "/rerank" => body["documents"].as_array().unwrap().len() as u64,
            route => panic!("unexpected outbound route: {route}"),
        };
        *expected.entry(request.url.path().to_owned()).or_default() += inputs;
    }
    let status = cli(root, &["status", "--json"]).await;
    let mut actual = std::collections::BTreeMap::<String, u64>::new();
    for endpoint in status["endpoints"].as_array().unwrap() {
        let url = reqwest::Url::parse(endpoint["url"].as_str().unwrap()).unwrap();
        *actual.entry(url.path().to_owned()).or_default() +=
            endpoint["sent_units"].as_u64().unwrap();
    }
    assert_eq!(
        actual, expected,
        "status must count only inputs seen by the gateway"
    );
}

#[tokio::test]
async fn queued_enrichment_rechecks_never_send_before_any_dispatch() {
    let root = tempfile::tempdir().unwrap();
    fixture(root.path());
    let server = gateway().await;
    config(root.path(), &server.uri(), false, true);
    cli(root.path(), &["index", "--level", "3", "--json"]).await;
    let db = Connection::open(root.path().join(".deepkit/code.db")).unwrap();
    assert_job_kinds(&db, "queued");
    assert!(server.received_requests().await.unwrap().is_empty());
    config(root.path(), &server.uri(), true, false);
    // Advance only fixture quota waiting times, preserving jobs and reservations.
    db.execute("UPDATE jobs SET not_before=0 WHERE state='queued'", [])
        .unwrap();
    // Drain may reject stale repository authority or retire old jobs. Capture its
    // public JSON either way; no original private input may reach the recorder.
    let mut command = tokio::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"));
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("DKC_") {
            command.env_remove(key);
        }
    }
    let output = tokio::time::timeout(
        Duration::from_secs(60),
        command
            .current_dir(root.path())
            .env("XDG_CONFIG_HOME", root.path().join("absent-user-config"))
            .env("DKC_MODELS_DIR", root.path().join("unused-models"))
            .env("DKC_PRIVACY_FIXTURE_KEY", "synthetic-fixture-only")
            .args(["queue", "drain", "--max-passes", "10", "--json"])
            .kill_on_drop(true)
            .output(),
    )
    .await
    .unwrap()
    .unwrap();
    let drained: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(drained["state"].is_string(), "{drained}");
    let mut visuals = 0;
    for request in server.received_requests().await.unwrap() {
        assert_public_json(&request.body_json::<Value>().unwrap(), &mut visuals);
    }
    // Refresh under the new policy and require actual public dispatches so a
    // conservative whole-repository invalidation cannot pass vacuously.
    cli(root.path(), &["index", "--level", "3", "--json"]).await;
    let requests = server.received_requests().await.unwrap();
    assert!(!requests.is_empty());
    for request in &requests {
        assert_public_json(&request.body_json::<Value>().unwrap(), &mut visuals);
    }
    assert_job_kinds(&db, "done");
    assert_no_protected_dispatch_or_publication(&db);
    assert_accounting(root.path(), &requests).await;
}

#[tokio::test]
async fn retained_summary_roles_are_withdrawn_from_outbound_context_after_privacy_change() {
    let root = tempfile::tempdir().unwrap();
    fixture(root.path());
    let server = gateway().await;
    config(root.path(), &server.uri(), false, false);
    cli(root.path(), &["index", "--level", "3", "--json"]).await;
    cli(
        root.path(),
        &["index", "--level", "2", "--refresh-context", "--json"],
    )
    .await;
    let before = server.received_requests().await.unwrap();
    assert!(
        before
            .iter()
            .any(|request| request.url.path() == "/api/embed"
                && request.body_json::<Value>().unwrap()["input"]
                    .to_string()
                    .contains(RETIRED_ROLE)),
        "the initially authorized generated context must actually reach embedding"
    );
    let db = Connection::open(root.path().join(".deepkit/code.db")).unwrap();
    let retained: i64 = db.query_row("SELECT count(*) FROM units u JOIN unit_contents c ON c.hash=u.content_hash WHERE u.kind='summary' AND instr(c.text,?1)>0",[RETIRED_ROLE],|r|r.get(0)).unwrap();
    assert!(
        retained > 0,
        "real summary publication required before changing policy"
    );
    let applied_roles: i64 = db.query_row("SELECT count(*) FROM fts_units JOIN units u ON u.id=fts_units.rowid JOIN files f ON f.id=u.file_id WHERE f.path LIKE 'secrets/roles/%' AND fts_units MATCH ?1",["header : \"withdrawn orchid role\""],|r|r.get(0)).unwrap();
    assert!(
        applied_roles > 0,
        "generated role must have been applied to original file headers"
    );
    let boundary = before.len();
    config(root.path(), &server.uri(), true, false);
    cli(
        root.path(),
        &["index", "--level", "3", "--refresh-context", "--json"],
    )
    .await;
    cli(
        root.path(),
        &[
            "search",
            "MeadowRerankPositiveControl",
            "--scope",
            "query-control.md",
            "--json",
        ],
    )
    .await;
    let requests = server.received_requests().await.unwrap();
    let after = &requests[boundary..];
    assert!(after.iter().any(|r| r.url.path() == "/api/embed"));
    assert!(after.iter().any(|r| r.url.path() == "/rerank"));
    let mut visuals = 0;
    for request in after {
        let value: Value = request.body_json().unwrap();
        assert_public_json(&value, &mut visuals);
        assert!(
            !value.to_string().contains(RETIRED_ROLE),
            "withdrawn summary leaked as generated context"
        );
    }
    // Accounting is cumulative, so compare the complete authorized-before and
    // filtered-after traffic together without deleting prior usage evidence.
    assert_accounting(root.path(), &requests).await;
}
