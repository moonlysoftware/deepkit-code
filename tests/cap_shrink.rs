//! Public shrink and index scheduling limits; v1.md §3.5 and §10.2.
use rusqlite::Connection;
use serde_json::Value;
use std::{error::Error, fs, path::Path, process::Command};

fn index(root: &Path, args: &[&str], nice: bool) -> Result<Value, Box<dyn Error>> {
    let output = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .args(args)
        .env("DKC_LIMITS__INDEX_NICE", nice.to_string())
        .output()?;
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}

#[tokio::test]
async fn completed_sources_under_discovery_cap_still_receive_image_and_page_enrichment() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};
    for scanned in [false, true] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices":[{"finish_reason":"stop","message":{"content":"# Verified public document\n\nA visible diagram and its explanation."}}],
                "usage":{"prompt_tokens":30,"completion_tokens":15,"total_tokens":45}
            })))
            .mount(&server).await;
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join(".deepkit")).unwrap();
        fs::write(root.path().join(".deepkit/config.toml"), format!(
            "[embedder]\nprofile='remote'\n[embedder.remote]\nurl='{0}/v1'\napi_key_env='DKC_CAP_FIXTURE_KEY'\n[llm]\nenabled=true\nurl='{0}/v1'\nmodel='fixture-text'\nvlm_model='fixture-vision'\napi_key_env='DKC_CAP_FIXTURE_KEY'\n[enrich]\ntop_symbols=0\nprs_last_n=0\nmodules_min_files=1000000\n[limits]\nmax_files=2\nmax_l2_units=0\n",server.uri())).unwrap();
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(2, 2)
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        for n in 0..4 {
            let (extension, bytes) = if scanned {
                (
                    "pdf",
                    include_bytes!("../fixtures/docs/scanned-jpeg.pdf").as_slice(),
                )
            } else {
                ("png", png.get_ref().as_slice())
            };
            fs::write(root.path().join(format!("source-{n}.{extension}")), bytes).unwrap();
        }
        let directory = root.path().to_owned();
        let output = tokio::task::spawn_blocking(move || {
            Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
                .current_dir(&directory)
                .env("XDG_CONFIG_HOME", directory.join("absent-config"))
                .env("DKC_CAP_FIXTURE_KEY", "fixture-only")
                .args(["index", "--level", "3", "--json"])
                .output()
                .unwrap()
        })
        .await
        .unwrap();
        assert!(
            output.status.success(),
            "scanned={scanned}: {} {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["files_added"], 2);
        assert_eq!(report["enrichment"]["state"], "complete", "{report}");
        let db = Connection::open(root.path().join(".deepkit/code.db")).unwrap();
        let query = if scanned {
            "SELECT count(DISTINCT file_id) FROM page_publications"
        } else {
            "SELECT count(DISTINCT p.file_id) FROM units s JOIN units p ON p.id=s.parent_unit WHERE s.kind='summary' AND p.kind='image'"
        };
        assert_eq!(db.query_row(query, [], |r| r.get::<_, i64>(0)).unwrap(), 2);
        let incomplete: bool = db.query_row("SELECT (SELECT value FROM meta WHERE key='l1_scan_generation') != (SELECT value FROM meta WHERE key='l1_completed_generation')", [], |r| r.get(0)).unwrap();
        assert!(incomplete, "a capped walk must not claim global completion");
        let requests = server.received_requests().await.unwrap();
        assert!(requests.len() >= 2);
        assert!(
            requests
                .iter()
                .all(|request| request.url.path() == "/v1/chat/completions")
        );
    }
}

#[cfg(unix)]
#[test]
fn discovery_caps_remain_visible_below_catalogue_capacity_and_clear_after_rescan()
-> Result<(), Box<dyn Error>> {
    use std::{ffi::OsString, os::unix::ffi::OsStringExt};
    let root = tempfile::tempdir()?;
    let root = root.path();
    fs::create_dir(root.join(".deepkit"))?;
    fs::write(root.join(".deepkit/config.toml"), "[limits]\nmax_files=2\n")?;
    fs::write(root.join("public.rs"), "pub fn public() {}\n")?;
    let mut non_unicode_paths = Vec::new();
    for n in 0..8 {
        let path = root.join(OsString::from_vec(vec![b'a' + n, 0xff]));
        match fs::write(&path, b"public fixture") {
            Ok(()) => non_unicode_paths.push(path),
            // APFS rejects the fixture. Verify ordinary discovery/status below
            // still works and does not invent a cap or skipped-path count.
            Err(error) if cfg!(target_os = "macos") && error.raw_os_error() == Some(92) => {
                assert!(non_unicode_paths.is_empty());
                break;
            }
            Err(error) => return Err(error.into()),
        }
    }
    index(root, &["index", "--level", "1", "--json"], false)?;
    let status = index(root, &["status", "--json"], false)?;
    assert!(status["files"].as_u64().unwrap() < 2);
    let cap_expected = !non_unicode_paths.is_empty();
    assert_eq!(status["caps"]["file_capacity_reached"], cap_expected);
    assert_eq!(
        status["discovery"]["skipped_non_unicode_paths"]
            .as_u64()
            .unwrap()
            > 0,
        cap_expected,
    );
    assert_eq!(status["discovery"]["scope"], "last_whole_repository_scan");
    let plain = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .arg("status")
        .output()?;
    assert!(plain.status.success());
    assert_eq!(
        String::from_utf8(plain.stdout)?.contains("max_files cap hit"),
        cap_expected
    );
    for path in non_unicode_paths {
        fs::remove_file(path)?;
    }
    index(root, &["index", "--level", "1", "--json"], false)?;
    let status = index(root, &["status", "--json"], false)?;
    assert_eq!(status["files"], 1);
    assert_eq!(status["caps"]["file_capacity_reached"], false);
    assert_eq!(status["discovery"]["skipped_non_unicode_paths"], 0);
    Ok(())
}

#[test]
fn shrink_reports_retained_l1_floor_without_preparing_or_refilling_models()
-> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    fs::write(root.path().join("source.rs"), "pub fn retained() {}\n")?;
    index(root.path(), &["index", "--level", "1", "--json"], false)?;
    let path = root.path().join(".deepkit/code.db");
    let identity = || -> Result<Vec<(String, String)>, rusqlite::Error> {
        Connection::open(&path)?
            .prepare("SELECT hex(hash),hex(content_hash) FROM units ORDER BY id")?
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect()
    };
    let before = identity()?;
    let bytes = fs::metadata(&path)?.len();
    fs::write(
        root.path().join(".deepkit/config.toml"),
        format!("[limits]\ndb_max_bytes='{} B'\n", bytes * 9 / 10),
    )?;
    // Default level 3 is intentional: shrink must not prepare models or refill L2.
    let shrunk = index(root.path(), &["index", "--shrink", "--json"], false)?;
    assert_eq!(shrunk["shrink"]["state"], "cannot_reach_target");
    assert_eq!(shrunk["shrink"]["summaries_removed"], 0);
    assert_eq!(shrunk["shrink"]["vectors_removed"], 0);
    assert_eq!(shrunk["model_preparation"]["state"], "skipped");
    assert_eq!(identity()?, before);
    assert_eq!(shrunk["maintenance"]["optimization_ran"], true);
    Ok(())
}

#[cfg(unix)]
#[test]
fn indexing_applies_or_disables_niceness_in_its_own_child_process() -> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    fs::write(root.path().join("source.rs"), "pub fn courtesy() {}\n")?;
    let before = rustix::process::getpriority_process(None)?;
    let applied = index(root.path(), &["index", "--quick", "--json"], true)?;
    assert_eq!(applied["index_priority"]["state"], "applied");
    assert!(applied["index_priority"]["after"].as_i64().unwrap() >= 10);
    assert!(
        applied["index_priority"]["after"].as_i64().unwrap()
            >= applied["index_priority"]["before"].as_i64().unwrap()
    );
    let disabled = index(root.path(), &["index", "--quick", "--json"], false)?;
    assert_eq!(disabled["index_priority"]["state"], "disabled");
    assert_eq!(rustix::process::getpriority_process(None)?, before);
    Ok(())
}

#[test]
fn public_status_names_file_and_l3_caps_and_retains_l1_after_lowering_limits()
-> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let root = root.path();
    fs::write(root.join("retained.rs"), "pub fn retained() {}\n")?;
    index(root, &["index", "--level", "1", "--json"], false)?;
    fs::write(root.join("new.rs"), "pub fn fresh_but_too_large() {}\n")?;
    fs::write(
        root.join(".deepkit/config.toml"),
        "[limits]\nmax_files=2\nmax_file_bytes='8 B'\nmax_l3_units=0\n",
    )?;
    index(root, &["index", "--level", "1", "--json"], false)?;
    let status = index(root, &["status", "--json"], false)?;
    assert_eq!(status["files"], 2);
    assert_eq!(status["units"], 1);
    assert_eq!(status["source_caps"]["max_file_bytes"], 2);
    assert_eq!(status["l3"]["paused"], true);
    assert_eq!(status["l3"]["reason"], "max_l3_units");
    assert_eq!(status["limits"]["max_l3_units"], 0);
    assert!(
        status["caps"]["file_capacity_message"]
            .as_str()
            .unwrap()
            .contains("cap hit, add ignores or treatments")
    );
    let plain = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .arg("status")
        .output()?;
    assert!(plain.status.success());
    let plain = String::from_utf8(plain.stdout)?;
    for message in [
        "max_file_bytes: 2",
        "L3 paused",
        "max_l3_units",
        "cap hit, add ignores or treatments",
    ] {
        assert!(plain.contains(message), "{plain}");
    }
    let db = Connection::open(root.join(".deepkit/code.db"))?;
    let pages: i64 = db.query_row("PRAGMA page_count", [], |r| r.get(0))?;
    let size: i64 = db.query_row("PRAGMA page_size", [], |r| r.get(0))?;
    drop(db);
    fs::write(
        root.join(".deepkit/config.toml"),
        format!("[limits]\ndb_max_bytes='{} B'\n", pages * size * 100 / 90),
    )?;
    let status = index(root, &["status", "--json"], false)?;
    assert_eq!(status["l3"]["paused"], true);
    assert_eq!(status["l3"]["reason"], "db_max_bytes 85 percent");
    assert_eq!(status["units"], 1);
    Ok(())
}

#[test]
fn shrink_progress_survives_a_later_catalogue_capacity_refusal_as_one_json_document()
-> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let root = root.path();
    fs::write(root.join("retained.rs"), "pub fn retained() {}\n")?;
    index(root, &["index", "--level", "1", "--json"], false)?;
    // No walk ordering can select only the retained file: discovery sees two new paths.
    fs::remove_file(root.join("retained.rs"))?;
    fs::write(root.join("new.rs"), "pub fn additional() {}\n")?;
    fs::write(root.join("other.rs"), "pub fn also_additional() {}\n")?;
    fs::write(root.join(".deepkit/config.toml"), "[limits]\nmax_files=1\n")?;
    let output = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .args(["index", "--shrink", "--json"])
        .output()?;
    assert!(!output.status.success());
    let value: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(value["index"]["state"], "partial");
    assert_eq!(value["error"]["code"], "file_capacity");
    assert_eq!(value["shrink"]["state"], "target_reached");
    assert_eq!(value["shrink"]["summaries_removed"], 0);
    assert_eq!(value["index_priority"]["requested"], true);
    Ok(())
}

#[test]
fn shrink_progress_is_reported_when_control_metadata_has_no_wal_reservation()
-> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let root = root.path();
    fs::write(root.join("retained.rs"), "pub fn retained() {}\n")?;
    index(root, &["index", "--level", "1", "--json"], false)?;
    // 16 KiB leaves less than one WAL frame in the control reservation, while
    // the catalogue maintenance reservation can still inspect the existing DB.
    fs::write(
        root.join(".deepkit/config.toml"),
        "[limits]\nwal_max_bytes='16 KiB'\n",
    )?;
    let output = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .args(["index", "--shrink", "--json"])
        .output()?;
    assert!(!output.status.success());
    let value: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(value["error"]["code"], "wal_capacity", "{value}");
    assert_eq!(value["shrink"]["state"], "target_reached", "{value}");
    assert_eq!(value["shrink"]["summaries_removed"], 0);
    let db = Connection::open(root.join(".deepkit/code.db"))?;
    assert_eq!(
        db.query_row("SELECT count(*) FROM units", [], |r| r.get::<_, i64>(0))?,
        1
    );
    Ok(())
}

#[test]
fn public_shrink_reaches_seventy_percent_on_disk_and_preserves_source_identity()
-> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let root = root.path();
    fs::write(root.join("retained.rs"), "pub fn retained() {}\n")?;
    index(root, &["index", "--level", "1", "--json"], false)?;
    let path = root.join(".deepkit/code.db");
    let identity: String;
    {
        let mut db = Connection::open(&path)?;
        identity = db.query_row("SELECT hex(hash) FROM units WHERE kind='symbol'", [], |r| {
            r.get(0)
        })?;
        let tx = db.transaction()?;
        // Derived rows must exceed 30% of the full file: retained schema/control
        // alone is about 1.2 MiB with the production page layout.
        for id in 1001_i64..=1032 {
            let mut hash = [0_u8; 32];
            hash[..8].copy_from_slice(&id.to_le_bytes());
            tx.execute(
                "INSERT INTO unit_contents(hash,text,tokens) VALUES(?1,?2,16384)",
                rusqlite::params![hash.as_slice(), "summary ".repeat(8192)],
            )?;
            tx.execute("INSERT INTO units(id,hash,content_hash,source_key,kind,header,ladder_level,created_at,updated_at) VALUES(?1,?2,?2,'fixture-summary:'||?1,'summary','summary',3,0,0)",rusqlite::params![id,hash.as_slice()])?;
            tx.execute(
                "INSERT INTO fts_units(rowid,header,body) VALUES(?1,'summary','summary')",
                [id],
            )?;
        }
        tx.commit()?;
    }
    let before = fs::metadata(&path)?.len();
    fs::write(
        root.join(".deepkit/config.toml"),
        format!("[limits]\ndb_max_bytes='{before} B'\n"),
    )?;
    let result = index(root, &["index", "--shrink", "--json"], false)?;
    assert_eq!(result["shrink"]["state"], "target_reached", "{result}");
    let target = result["shrink"]["target_bytes"].as_u64().unwrap();
    let actual = fs::metadata(&path)?.len();
    assert!(
        actual <= target,
        "actual={actual}, target={target}, result={result}"
    );
    assert_eq!(result["shrink"]["file_bytes"], actual);
    assert!(result["shrink"]["summaries_removed"].as_u64().unwrap() > 0);
    assert_eq!(result["embedding"]["state"], "skipped");
    let db = Connection::open(&path)?;
    assert_eq!(
        db.query_row("SELECT hex(hash) FROM units WHERE kind='symbol'", [], |r| r
            .get::<_, String>(0))?,
        identity
    );
    assert_eq!(
        db.query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
        "ok"
    );
    Ok(())
}
