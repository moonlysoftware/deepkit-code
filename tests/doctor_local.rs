//! Public local doctor diagnoses failures without creating or modifying repository state.
use assert_cmd::Command;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Output,
    time::Duration,
};

fn command(root: &Path, isolation: &Path) -> Command {
    let mut command = Command::new(assert_cmd::cargo::cargo_bin!("dkc"));
    command.current_dir(root).timeout(Duration::from_secs(20));
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().starts_with("DKC_") {
            command.env_remove(name);
        }
    }
    command
        .env_remove("MOONLY_GATEWAY_API_KEY")
        .env_remove("DOCTOR_LOCAL_UNUSED_KEY")
        .env("XDG_CONFIG_HOME", isolation.join("user"))
        .env("DKC_MODELS_DIR", isolation.join("absent-model-cache"))
        .env("DKC_LOG", "trace");
    command
}

fn git(root: &Path, args: &[&str]) {
    assert!(
        std::process::Command::new("git")
            // Fixture commits must finish all writes before read-only snapshots.
            // Git's detached auto-maintenance can otherwise remove its lock later.
            .args(["-c", "maintenance.auto=false", "-c", "gc.auto=0"])
            .arg("-C")
            .arg(root)
            .args(args)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .output()
            .unwrap()
            .status
            .success()
    );
}
fn fixture() -> tempfile::TempDir {
    let isolation = tempfile::tempdir().unwrap();
    let root = isolation.path().join("repo");
    fs::create_dir(&root).unwrap();
    git(&root, &["init", "-q"]);
    fs::write(
        root.join("script.rs"),
        "pub fn doctor_fixture() -> bool { true }\n",
    )
    .unwrap();
    git(&root, &["add", "script.rs"]);
    git(
        &root,
        &[
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        ],
    );
    fs::create_dir(root.join(".deepkit")).unwrap();
    // A remote profile with reranking disabled must not load local models.
    // Missing isolated credentials prevent requests to the loopback-only endpoint.
    fs::write(root.join(".deepkit/config.toml"),
        "[embedder]\nprofile='remote'\n[embedder.remote]\nurl='http://127.0.0.1:9/v1'\nmodel='fixture'\napi_key_env='DOCTOR_LOCAL_UNUSED_KEY'\n[rerank]\nmode='off'\n").unwrap();
    isolation
}

// Include directory entries as well as bytes to catch empty cache/config folders.
#[derive(Debug, PartialEq, Eq)]
enum Entry {
    Directory,
    File { length: usize, hash: blake3::Hash },
    Symlink(PathBuf),
}
fn snapshot(root: &Path) -> BTreeMap<PathBuf, Entry> {
    fn visit(root: &Path, dir: &Path, entries: &mut BTreeMap<PathBuf, Entry>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let relative = path.strip_prefix(root).unwrap().to_path_buf();
            let kind = fs::symlink_metadata(&path).unwrap().file_type();
            if kind.is_symlink() {
                entries.insert(relative, Entry::Symlink(fs::read_link(&path).unwrap()));
            } else if kind.is_dir() {
                entries.insert(relative, Entry::Directory);
                visit(root, &path, entries);
            } else {
                assert!(kind.is_file());
                let bytes = fs::read(&path).unwrap();
                entries.insert(
                    relative,
                    Entry::File {
                        length: bytes.len(),
                        hash: blake3::hash(&bytes),
                    },
                );
            }
        }
    }
    let mut entries = BTreeMap::new();
    visit(root, root, &mut entries);
    entries
}
fn doctor(isolation: &Path) -> (Output, Value) {
    let output = command(&isolation.join("repo"), isolation)
        .args(["doctor", "--json"])
        .output()
        .unwrap();
    let report: Value = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "doctor must emit JSON even on failure: {error}; stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    assert_eq!(report["scope"], "diagnostics");
    for finding in report["checks"].as_array().unwrap() {
        let message = finding["message"].as_str().unwrap();
        assert!(!message.is_empty());
        assert!(!message.contains(['\n', '\r']), "{finding}");
    }
    assert!(report["complete"].is_boolean());
    assert!(matches!(
        report["status"].as_str(),
        Some("failed" | "incomplete" | "warning" | "passed")
    ));
    (output, report)
}
fn check<'a>(report: &'a Value, id: &str) -> &'a Value {
    report["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["id"] == id)
        .unwrap_or_else(|| panic!("missing check {id}: {report}"))
}
fn message(report: &Value, id: &str, expected: &str) {
    assert_eq!(check(report, id)["message"], expected, "{id}: {report}");
}
fn no_leak(output: &Output, report: &Value, marker: &str) {
    assert!(!String::from_utf8_lossy(&output.stdout).contains(marker));
    assert!(!String::from_utf8_lossy(&output.stderr).contains(marker));
    assert!(!report.to_string().contains(marker));
}

#[test]
fn missing_index_is_reported_without_creating_anything() {
    let fixture = fixture();
    let before = snapshot(fixture.path());
    let (_, report) = doctor(fixture.path());
    assert_eq!(report["complete"], false);
    assert_eq!(check(&report, "database.present")["status"], "warning");
    message(
        &report,
        "database.present",
        "Index database is absent; run dkc index or serve",
    );
    message(&report, "mcp.cursor", "Project MCP configuration is absent");
    assert_eq!(check(&report, "queue.health")["status"], "not_checked");
    assert!(check(&report, "queue.health")["evidence"].is_null());
    assert_eq!(check(&report, "config.effective")["status"], "passed");
    assert_eq!(check(&report, "git.head")["status"], "passed");
    message(&report, "git.head", "Git HEAD resolves successfully");
    assert_eq!(snapshot(fixture.path()), before);
}

#[test]
fn corrupt_database_does_not_hide_other_diagnostics_or_change_bytes() {
    let fixture = fixture();
    fs::write(
        fixture.path().join("repo/.deepkit/code.db"),
        b"not a SQLite database\0\xff",
    )
    .unwrap();
    let before = snapshot(fixture.path());
    let (output, report) = doctor(fixture.path());
    assert_eq!(report["complete"], false);
    assert!(!output.status.success());
    assert_eq!(report["status"], "failed");
    assert_eq!(check(&report, "queue.health")["status"], "not_checked");
    assert!(check(&report, "queue.health")["evidence"].is_null());
    assert!(report["checks"].as_array().unwrap().iter().any(|entry| {
        matches!(
            entry["id"].as_str(),
            Some("database.open" | "database.integrity" | "database.schema")
        ) && entry["status"] == "failed"
    }));
    for diagnosis in report["checks"].as_array().unwrap() {
        let expected = match diagnosis["id"].as_str() {
            Some("database.open") => Some("Existing database could not be opened read-only"),
            Some("database.integrity") => {
                Some("SQLite integrity check could not read the database")
            }
            Some("database.schema") => Some("Database schema metadata is unreadable or invalid"),
            _ => None,
        };
        if let Some(expected) = expected {
            assert_eq!(diagnosis["message"], expected);
        }
    }

    assert_eq!(check(&report, "config.effective")["status"], "passed");
    assert_eq!(check(&report, "git.head")["status"], "passed");
    message(&report, "git.head", "Git HEAD resolves successfully");
    assert_eq!(snapshot(fixture.path()), before);
}

#[test]
fn invalid_config_still_emits_redacted_json_and_git_result() {
    let fixture = fixture();
    let marker = format!(
        "synthetic-{}",
        fixture.path().file_name().unwrap().to_string_lossy()
    );
    fs::write(
        fixture.path().join("repo/.deepkit/config.toml"),
        format!("broken='{marker}\n"),
    )
    .unwrap();
    let before = snapshot(fixture.path());
    let (output, report) = doctor(fixture.path());
    assert_eq!(report["complete"], false);
    assert!(!output.status.success());
    assert_eq!(check(&report, "config.repository")["status"], "failed");
    message(
        &report,
        "config.repository",
        "Repository configuration is not valid TOML",
    );
    assert_eq!(check(&report, "git.head")["status"], "passed");
    message(&report, "git.head", "Git HEAD resolves successfully");
    no_leak(&output, &report, &marker);
    assert_eq!(snapshot(fixture.path()), before);
}

#[test]
fn repository_secret_and_unknown_key_names_are_not_echoed() {
    let fixture = fixture();
    let marker = format!(
        "synthetic-{}",
        fixture.path().file_name().unwrap().to_string_lossy()
    );
    let path = fixture.path().join("repo/.deepkit/config.toml");
    let mut text = fs::read_to_string(&path).unwrap();
    text.push_str(&format!(
        "\n[custom]\n'{marker}'=true\npassword='{marker}'\n"
    ));
    fs::write(&path, text).unwrap();
    let before = snapshot(fixture.path());
    let (output, report) = doctor(fixture.path());
    assert!(!output.status.success());
    assert_eq!(
        check(&report, "config.repository.secrets")["status"],
        "failed"
    );
    assert_eq!(
        check(&report, "config.repository.unknown_keys")["status"],
        "warning"
    );
    assert_eq!(check(&report, "git.head")["status"], "passed");
    message(&report, "git.head", "Git HEAD resolves successfully");
    message(
        &report,
        "config.repository.secrets",
        "Repository configuration contains a secret field; move keys to external credentials",
    );
    message(
        &report,
        "config.repository.unknown_keys",
        "Unknown configuration keys found; review repository configuration",
    );
    no_leak(&output, &report, &marker);
    assert_eq!(snapshot(fixture.path()), before);
}

#[test]
fn healthy_index_reports_capabilities_without_application_writes() {
    let fixture = fixture();
    let output = command(&fixture.path().join("repo"), fixture.path())
        .args(["index", "--level", "1", "--json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let indexed: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(
        indexed["units_indexed"]
            .as_u64()
            .is_some_and(|count| count > 0)
    );
    let before = snapshot(fixture.path());
    let (output, report) = doctor(fixture.path());
    // Depends on aggregate coverage semantics: failed credentials were checked.
    assert_eq!(report["complete"], true);
    // Local checks pass independently of the deliberately unconfigured provider credential.
    assert!(!output.status.success());
    for id in [
        "database.integrity",
        "database.schema",
        "database.sqlite_vector",
        "database.fts5",
        "database.lexical_tables",
        "database.caps",
    ] {
        assert_eq!(check(&report, id)["status"], "passed", "{id}: {report}");
    }
    assert_eq!(check(&report, "git.head")["status"], "passed");
    message(&report, "git.head", "Git HEAD resolves successfully");
    for (id, expected) in [
        ("database.integrity", "SQLite integrity check passed"),
        ("database.schema", "Database schema matches this binary"),
        ("database.sqlite_vector", "sqlite-vector is available"),
        ("database.fts5", "FTS5 is available"),
        (
            "database.lexical_tables",
            "Existing lexical FTS tables are available",
        ),
        (
            "database.caps",
            "Observed database and WAL sizes fit configured caps",
        ),
        (
            "queue.health",
            "Queue snapshot has no observed failures or expired leases; worker liveness is not inferred",
        ),
    ] {
        message(&report, id, expected);
    }
    let queue = check(&report, "queue.health");
    assert_eq!(queue["status"], "passed");
    for counter in ["failed", "dead", "expired_leases"] {
        assert_eq!(queue["evidence"]["jobs"][counter], 0, "{counter}: {queue}");
    }
    assert_eq!(queue["evidence"]["uncertain_calls"], 0);
    assert_eq!(queue["evidence"]["deferrals"], serde_json::json!({}));
    // SQLite READ_ONLY sees committed WAL data and may create its bookkeeping
    // sidecars; closing the usage connection can remove empty sidecars again.
    // Any retained WAL must contain no application frames, and all other files
    // and repository entries must remain unchanged.
    let mut after = snapshot(fixture.path());
    for name in ["repo/.deepkit/code.db-wal", "repo/.deepkit/code.db-shm"] {
        assert!(!before.contains_key(Path::new(name)));
        let added = after.remove(Path::new(name));
        assert!(matches!(added, None | Some(Entry::File { .. })));
    }
    if let Ok(wal) = fs::metadata(fixture.path().join("repo/.deepkit/code.db-wal")) {
        assert_eq!(wal.len(), 0);
    }
    assert_eq!(after, before);
}

#[test]
fn explicitly_configured_unmatched_treatment_globs_warn_without_exposing_patterns() {
    let fixture = fixture();
    let marker = format!(
        "synthetic-{}",
        fixture.path().file_name().unwrap().to_string_lossy()
    );
    let path = fixture.path().join("repo/.deepkit/config.toml");
    let mut text = fs::read_to_string(&path).unwrap();
    text.push_str(&format!(
        "\n[[treatments]]\nname='{marker}'\nmatch=['script.rs', '{marker}/**']\nchunker='ast'\nmax_level=1\n"
    ));
    fs::write(&path, text).unwrap();
    let before = snapshot(fixture.path());
    let (output, report) = doctor(fixture.path());
    assert_eq!(check(&report, "config.effective")["status"], "passed");
    let globs = check(&report, "config.treatment_globs");
    assert_eq!(globs["status"], "warning");
    message(
        &report,
        "config.treatment_globs",
        "Configured treatment globs matched no observed eligible files",
    );
    assert_eq!(globs["evidence"]["patterns_checked"], 2);
    assert_eq!(globs["evidence"]["files_observed"], 1);
    assert_eq!(
        globs["evidence"]["unmatched"],
        serde_json::json!([{ "rule": 1, "pattern": 2 }])
    );
    no_leak(&output, &report, &marker);
    assert_eq!(check(&report, "git.head")["status"], "passed");
    message(&report, "git.head", "Git HEAD resolves successfully");
    assert_eq!(snapshot(fixture.path()), before);
}

#[test]
fn existing_over_cap_database_and_uncertain_queue_have_actionable_messages() {
    let fixture = fixture();
    let root = fixture.path().join("repo");
    let output = command(&root, fixture.path())
        .args(["index", "--level", "1", "--json"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let database = root.join(".deepkit/code.db");
    let connection = rusqlite::Connection::open(&database).unwrap();
    connection.execute("INSERT INTO usage_reservations(id,provider,day,calls,tokens,state,created_at,dispatched_at) VALUES('fixture','private-queue-provider','1970-01-01',1,1,'dispatched',0,0)", []).unwrap();
    drop(connection);
    let config = root.join(".deepkit/config.toml");
    let mut text = fs::read_to_string(&config).unwrap();
    text.push_str("\n[limits]\ndb_max_bytes='1 B'\n");
    fs::write(&config, text).unwrap();
    let before = fs::read(&database).unwrap();
    let (output, report) = doctor(fixture.path());
    assert_eq!(check(&report, "database.caps")["status"], "warning");
    message(
        &report,
        "database.caps",
        "Existing database or WAL exceeds configured cap; review limits or explicit shrink",
    );
    assert_eq!(check(&report, "queue.health")["status"], "warning");
    message(
        &report,
        "queue.health",
        "Queue contains failures, delays, expired leases or uncertain charges; inspect queue status",
    );
    assert_eq!(
        check(&report, "queue.health")["evidence"]["uncertain_calls"],
        1
    );
    no_leak(&output, &report, "private-queue-provider");
    assert_eq!(fs::read(&database).unwrap(), before);
}
