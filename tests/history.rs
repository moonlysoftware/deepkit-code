//! Public bounded commit ingestion and immutable historical symbols; v1.md §4.3.
use rusqlite::Connection;
use serde_json::Value;
use std::{fs, path::Path, process::Command};
fn git(root: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}
fn commit(root: &Path, day: u8, subject: &str) -> String {
    git(root, &["add", "code.rs"]);
    let output = Command::new("git")
        .current_dir(root)
        .env("GIT_AUTHOR_DATE", format!("2026-01-{day:02}T12:00:00Z"))
        .env("GIT_COMMITTER_DATE", format!("2026-01-{day:02}T12:00:00Z"))
        .args([
            "commit",
            "-q",
            "-m",
            subject,
            "-m",
            "A body line.\nSecond body line.",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    git(root, &["rev-parse", "HEAD"])
}
fn config(root: &Path, count: u64, cap: u64) {
    fs::create_dir_all(root.join(".deepkit")).unwrap();
    fs::write(root.join(".deepkit/config.toml"),format!("[embedder]\nprofile='local'\n[llm]\nenabled=false\n[rerank]\nmode='off'\n[commits]\nmax_count={count}\nprs=false\n[limits]\nmax_commits={cap}\n")).unwrap();
}
fn index(root: &Path) -> Value {
    let output = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .env("XDG_CONFIG_HOME", root.join("absent-user-config"))
        .args(["index", "--level", "1", "--json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
fn lexical_commits(root: &Path, query: &str) -> Value {
    let output = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .env("XDG_CONFIG_HOME", root.join("absent-user-config"))
        .args([
            "search", query, "--scope", "commits", "--mode", "lexical", "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
fn fixture() -> (tempfile::TempDir, Vec<String>) {
    let root = tempfile::tempdir().unwrap();
    let path = root.path();
    git(path, &["init", "-q", "-b", "main"]);
    git(path, &["config", "user.name", "History Fixture"]);
    git(path, &["config", "user.email", "history@example.invalid"]);
    fs::write(
        path.join("code.rs"),
        "pub fn historical() -> i32 {\n    1\n}\n",
    )
    .unwrap();
    let first = commit(path, 1, "initial definition");
    fs::write(
        path.join("code.rs"),
        "pub fn historical() -> i32 {\n    2\n}\n",
    )
    .unwrap();
    let second = commit(path, 2, "change old function");
    fs::write(
        path.join("code.rs"),
        format!(
            "{}pub fn current_name() -> i32 {{\n    3\n}}\n",
            "// moved later\n".repeat(50)
        ),
    )
    .unwrap();
    let third = commit(path, 3, "rename and move function");
    (root, vec![first, second, third])
}
fn script_fixture() -> (tempfile::TempDir, String, String) {
    let root = tempfile::tempdir().unwrap();
    let path = root.path();
    git(path, &["init", "-q", "-b", "main"]);
    git(path, &["config", "user.name", "History Fixture"]);
    git(path, &["config", "user.email", "history@example.invalid"]);
    fs::write(path.join("script"), "#!/usr/bin/env python3\nprint('ok')\n").unwrap();
    git(path, &["add", "script"]);
    git(
        path,
        &["commit", "-q", "-m", "unique history script marker"],
    );
    let sha = git(path, &["rev-parse", "HEAD"]);
    config(path, 2000, 2000);
    let config_path = path.join(".deepkit/config.toml");
    let original = fs::read_to_string(&config_path).unwrap();
    (root, sha, original)
}

fn script_treatment(config: &str, max_level: u8) -> String {
    format!(
        "{config}\n[[treatments]]\nname='script'\nmatch=['script']\nchunker='lines'\nmax_level={max_level}\n"
    )
}

#[test]
fn l0_only_commit_is_retained_but_not_published_and_policy_changes_reconcile() {
    let (root, sha, original) = script_fixture();
    let path = root.path();
    let config_path = path.join(".deepkit/config.toml");

    assert_eq!(index(path)["history"]["inserted"], 1);
    let db = Connection::open(path.join(".deepkit/code.db")).unwrap();
    let retained = || -> (i64, i64) {
        db.query_row(
            "SELECT c.unit_id,u.ladder_level FROM commits c JOIN units u ON u.id=c.unit_id WHERE lower(hex(c.sha))=?1",
            [&sha],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap()
    };
    let (unit, level) = retained();
    assert_eq!(level, 0);
    let result = lexical_commits(path, "unique history script marker");
    assert_eq!(result["index"]["partial"], false, "{result}");
    assert_eq!(result["hits"].as_array().unwrap().len(), 0, "{result}");

    fs::write(&config_path, script_treatment(&original, 2)).unwrap();
    assert_eq!(index(path)["history"]["inserted"], 0);
    assert_eq!(retained(), (unit, 1));
    let result = lexical_commits(path, "unique history script marker");
    assert_eq!(result["index"]["partial"], false, "{result}");
    assert!(
        result["hits"]
            .as_array()
            .unwrap()
            .iter()
            .any(|hit| hit["header"] == "unique history script marker"),
        "{result}"
    );

    fs::write(&config_path, original).unwrap();
    assert_eq!(index(path)["history"]["inserted"], 0);
    assert_eq!(retained(), (unit, 0));
    let result = lexical_commits(path, "unique history script marker");
    assert_eq!(result["index"]["partial"], false, "{result}");
    assert_eq!(result["hits"].as_array().unwrap().len(), 0, "{result}");
}
#[tokio::test]
async fn history_policy_downgrade_revokes_vector_and_reupgrade_reembeds() {
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(|request: &wiremock::Request| {
            let value: Value = serde_json::from_slice(&request.body).unwrap();
            let count = value["input"].as_array().unwrap().len();
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"embeddings":vec![vec![1.0,0.0];count]}))
        })
        .mount(&server)
        .await;
    let (root, sha, original) = script_fixture();
    let path = root.path();
    let config_path = path.join(".deepkit/config.toml");
    let remote = original.replace(
        "profile='local'",
        &format!(
            "profile='ollama'\ndims=2\n[embedder.ollama]\nurl='{}'",
            server.uri()
        ),
    );
    fs::write(&config_path, script_treatment(&remote, 2)).unwrap();
    let run_l2 = || async {
        let output = tokio::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
            .current_dir(path)
            .env("XDG_CONFIG_HOME", path.join("absent-user-config"))
            .args(["index", "--level", "2", "--json"])
            .output()
            .await
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    };
    run_l2().await;
    let db = Connection::open(path.join(".deepkit/code.db")).unwrap();
    let (unit, table): (i64, String) = (
        db.query_row(
            "SELECT unit_id FROM commits WHERE lower(hex(sha))=?1",
            [&sha],
            |r| r.get(0),
        )
        .unwrap(),
        db.query_row(
            "SELECT slug FROM embedding_tables WHERE active=1",
            [],
            |r| r.get(0),
        )
        .unwrap(),
    );
    let coverage = || -> i64 {
        db.query_row(
            &format!("SELECT count(*) FROM vec_{table} WHERE unit_id=?1"),
            [unit],
            |r| r.get(0),
        )
        .unwrap()
    };
    assert_eq!(coverage(), 1);
    let calls = server.received_requests().await.unwrap().len();

    fs::write(&config_path, script_treatment(&remote, 1)).unwrap();
    assert_eq!(index(path)["history"]["inserted"], 0);
    assert_eq!(coverage(), 0);
    assert_eq!(
        db.query_row("SELECT ladder_level FROM units WHERE id=?1", [unit], |r| {
            r.get::<_, i64>(0)
        })
        .unwrap(),
        1
    );
    let visible = lexical_commits(path, "unique history script marker");
    assert_eq!(visible["index"]["partial"], false);
    assert!(!visible["hits"].as_array().unwrap().is_empty());
    assert_eq!(server.received_requests().await.unwrap().len(), calls);

    fs::write(&config_path, &remote).unwrap();
    assert_eq!(index(path)["history"]["inserted"], 0);
    assert_eq!(coverage(), 0);
    assert_eq!(
        db.query_row("SELECT ladder_level FROM units WHERE id=?1", [unit], |r| {
            r.get::<_, i64>(0)
        })
        .unwrap(),
        0
    );

    fs::write(&config_path, script_treatment(&remote, 2)).unwrap();
    run_l2().await;
    assert_eq!(coverage(), 1);
    assert!(server.received_requests().await.unwrap().len() > calls);
}
#[test]
fn commit_metadata_units_and_hunks_use_the_historical_tree() {
    let (root, shas) = fixture();
    config(root.path(), 2000, 2000);
    let result = index(root.path());
    assert_eq!(result["history"]["inserted"], 3);
    let db = Connection::open(root.path().join(".deepkit/code.db")).unwrap();
    let (author,body,parents,files,symbols,additions,deletions):(String,String,i64,String,String,i64,i64)=db.query_row("SELECT author,body,parents,files_touched,symbols_touched,additions,deletions FROM commits WHERE lower(hex(sha))=?1",[&shas[1]],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?))).unwrap();
    assert_eq!(author, "History Fixture");
    assert_eq!(body, "A body line.\nSecond body line.");
    assert_eq!(parents, 1);
    assert_eq!(
        serde_json::from_str::<Value>(&files).unwrap(),
        serde_json::json!(["code.rs"])
    );
    let symbols: Vec<String> = serde_json::from_str(&symbols).unwrap();
    assert!(
        symbols.iter().any(|s| s.contains("historical")),
        "{symbols:?}"
    );
    assert!(!symbols.iter().any(|s| s.contains("current_name")));
    assert_eq!((additions, deletions), (1, 1));
    let (header,scope,text):(String,String,String)=db.query_row("SELECT u.header,u.scope_path,p.text FROM commits c JOIN units u ON u.id=c.unit_id JOIN unit_contents p ON p.hash=u.content_hash WHERE lower(hex(c.sha))=?1",[&shas[1]],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).unwrap();
    assert_eq!(header, "change old function");
    assert_eq!(scope, "commit › 2026-01-02 › History Fixture");
    assert!(text.contains("files:\ncode.rs\nsymbols:\n"));
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM unit_sources s JOIN units u ON u.id=s.unit_id WHERE s.source_kind='commit' AND u.kind='commit'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        3
    );
    assert_eq!(index(root.path())["history"]["state"], "not_needed");
}
#[test]
fn bounded_window_incremental_new_only_and_lowered_cap_retains_history() {
    let (root, shas) = fixture();
    config(root.path(), 2, 2);
    let first = index(root.path());
    assert_eq!(first["history"]["inserted"], 2);
    assert_eq!(first["history"]["max_count_reached"], true);
    let db = Connection::open(root.path().join(".deepkit/code.db")).unwrap();
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM commits WHERE lower(hex(sha))=?1",
            [&shas[0]],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
    fs::write(root.path().join("code.rs"), "pub fn latest() {}\n").unwrap();
    let fourth = commit(root.path(), 4, "new commit after cap");
    config(root.path(), 4, 1);
    let capped = index(root.path());
    assert_eq!(capped["history"]["state"], "capped");
    assert_eq!(capped["history"]["inserted"], 0);
    assert_eq!(
        db.query_row("SELECT count(*) FROM commits", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        2
    );
    assert_eq!(index(root.path())["history"]["max_commits_reached"], true);
    config(root.path(), 4, 4);
    let grown = index(root.path());
    assert_eq!(grown["history"]["inserted"], 2);
    assert_eq!(grown["history"]["existing"], 2);
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM commits WHERE lower(hex(sha))=?1",
            [fourth],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        1
    );
    assert_eq!(
        db.query_row("SELECT count(*) FROM commits", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        4
    );
}

#[test]
fn failed_provenance_insert_rolls_back_commit_unit_and_fts_before_retry() {
    let (root, _) = fixture();
    config(root.path(), 2000, 2000);
    let config_path = root.path().join(".deepkit/config.toml");
    let original = fs::read_to_string(&config_path).unwrap();
    fs::write(
        &config_path,
        original.replace("[commits]", "[commits]\nenabled=false"),
    )
    .unwrap();
    index(root.path());
    let db = Connection::open(root.path().join(".deepkit/code.db")).unwrap();
    db.execute_batch("CREATE TRIGGER refuse_history_sources BEFORE INSERT ON unit_sources WHEN new.source_kind='commit' BEGIN SELECT RAISE(ABORT,'fixture publication failure'); END;").unwrap();
    fs::write(&config_path, original).unwrap();
    let output = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root.path())
        .env("XDG_CONFIG_HOME", root.path().join("absent-user-config"))
        .args(["index", "--level", "1", "--json"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert_eq!(
        db.query_row("SELECT count(*) FROM commits", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        db.query_row("SELECT count(*) FROM units WHERE kind='commit'", [], |r| {
            r.get::<_, i64>(0)
        })
        .unwrap(),
        0
    );
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM fts_units WHERE rowid NOT IN (SELECT id FROM units)",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
    db.execute_batch("DROP TRIGGER refuse_history_sources;")
        .unwrap();
    assert_eq!(index(root.path())["history"]["inserted"], 3);
}

#[tokio::test]
async fn commit_units_embed_with_historical_context_and_recency() {
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(|request: &wiremock::Request| {
            let value: Value = serde_json::from_slice(&request.body).unwrap();
            let count = value["input"].as_array().unwrap().len();
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"embeddings":vec![vec![1.0,0.0];count]}))
        })
        .mount(&server)
        .await;
    let (root, _) = fixture();
    config(root.path(), 2000, 2000);
    let file = root.path().join(".deepkit/config.toml");
    let settings = fs::read_to_string(&file).unwrap().replace(
        "profile='local'",
        &format!(
            "profile='ollama'\ndims=2\n[embedder.ollama]\nurl='{}'",
            server.uri()
        ),
    );
    fs::write(&file, settings).unwrap();
    let output = tokio::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root.path())
        .env("XDG_CONFIG_HOME", root.path().join("absent-user-config"))
        .args(["index", "--level", "2", "--json"])
        .output()
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let requests = server.received_requests().await.unwrap();
    let inputs = requests
        .iter()
        .flat_map(|r| {
            serde_json::from_slice::<Value>(&r.body).unwrap()["input"]
                .as_array()
                .unwrap()
                .clone()
        })
        .filter_map(|v| v.as_str().map(str::to_owned))
        .collect::<Vec<_>>();
    assert!(
        inputs.iter().any(|s| s
            .starts_with("commit › 2026-01-02 › History Fixture\nchange old function\ncommit\n")),
        "{inputs:?}"
    );
    let db = Connection::open(root.path().join(".deepkit/code.db")).unwrap();
    let table: String = db
        .query_row(
            "SELECT slug FROM embedding_tables WHERE active=1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(table.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'));
    let count: i64 = db
        .query_row(
            &format!("SELECT count(*) FROM commits c JOIN vec_{table} v ON v.unit_id=c.unit_id"),
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(count, 3);
    let (older,newer):(f64,f64)=db.query_row("SELECT min(j.priority),max(j.priority) FROM jobs j JOIN commits c ON c.unit_id=j.target_id WHERE j.kind='embed'",[],|r|Ok((r.get(0)?,r.get(1)?))).unwrap();
    assert!(newer > older, "commit ages must affect priority");
    let calls = requests.len();
    let jobs: i64 = db
        .query_row("SELECT count(*) FROM jobs", [], |r| r.get(0))
        .unwrap();
    let output = tokio::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root.path())
        .env("XDG_CONFIG_HOME", root.path().join("absent-user-config"))
        .args(["index", "--level", "2", "--json"])
        .output()
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(server.received_requests().await.unwrap().len(), calls);
    assert_eq!(
        db.query_row("SELECT count(*) FROM jobs", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        jobs
    );
}

#[test]
fn merge_history_includes_side_branch_and_diff_is_against_first_parent() {
    let (root, shas) = fixture();
    let p = root.path();
    git(p, &["checkout", "-q", "-b", "side", &shas[0]]);
    fs::write(p.join("side.rs"), "pub fn only_on_side() {}\n").unwrap();
    git(p, &["add", "side.rs"]);
    let side = commit(p, 4, "side branch definition");
    git(p, &["checkout", "-q", "main"]);
    git(p, &["merge", "--no-ff", "-m", "merge side", "side"]);
    let merge = git(p, &["rev-parse", "HEAD"]);
    config(p, 2000, 2000);
    assert_eq!(index(p)["history"]["inserted"], 5);
    let db = Connection::open(p.join(".deepkit/code.db")).unwrap();
    let (parents, files, symbols, additions, deletions): (i64,String,String,i64,i64) = db.query_row("SELECT parents,files_touched,symbols_touched,additions,deletions FROM commits WHERE lower(hex(sha))=?1", [&merge], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?))).unwrap();
    assert_eq!(parents, 2);
    assert_eq!(
        serde_json::from_str::<Value>(&files).unwrap(),
        serde_json::json!(["side.rs"])
    );
    assert!(symbols.contains("only_on_side"));
    assert_eq!((additions, deletions), (1, 0));
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM commits WHERE lower(hex(sha))=?1",
            [&side],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        1
    );
}

#[test]
fn deleted_symbols_survive_and_binary_stats_remain_unknown() {
    let (root, _) = fixture();
    let p = root.path();
    fs::write(p.join("code.rs"), "// no definitions left\n").unwrap();
    let removed = commit(p, 4, "remove definition");
    fs::write(p.join("binary.dat"), b"a\0b\n").unwrap();
    git(p, &["add", "binary.dat"]);
    let binary = commit(p, 5, "add binary");
    config(p, 2000, 2000);
    assert_eq!(index(p)["history"]["incomplete_diffs"], 1);
    let db = Connection::open(p.join(".deepkit/code.db")).unwrap();
    let symbols: String = db
        .query_row(
            "SELECT symbols_touched FROM commits WHERE lower(hex(sha))=?1",
            [&removed],
            |r| r.get(0),
        )
        .unwrap();
    assert!(symbols.contains("current_name"));
    let stats: (Option<i64>, Option<i64>) = db
        .query_row(
            "SELECT additions,deletions FROM commits WHERE lower(hex(sha))=?1",
            [&binary],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(stats, (None, None));
    assert_eq!(index(p)["history"]["incomplete_diffs"], 1);
}

#[test]
fn history_search_verifies_immutable_commits_after_working_file_is_deleted() {
    let (root, _) = fixture();
    config(root.path(), 2000, 2000);
    index(root.path());
    let settings = root.path().join(".deepkit/config.toml");
    let original = fs::read_to_string(&settings).unwrap();
    fs::write(
        &settings,
        original.replace(
            "max_commits=2000",
            "max_commits=2000\nmax_open_bytes=\"32 B\"",
        ),
    )
    .unwrap();
    fs::remove_file(root.path().join("code.rs")).unwrap();
    let output = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root.path())
        .env("XDG_CONFIG_HOME", root.path().join("absent-user-config"))
        .args([
            "search",
            "change old function",
            "--scope",
            "commits",
            "--mode",
            "lexical",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    let hits = result["hits"].as_array().unwrap();
    assert!(!hits.is_empty(), "{result}");
    assert!(
        hits.iter()
            .all(|hit| hit["kind"] == "commit" && hit["fresh"]["l1"] == "current")
    );
    assert!(
        hits.iter()
            .any(|hit| hit["header"] == "change old function"),
        "{result}"
    );
    let selected = hits
        .iter()
        .find(|hit| hit["header"] == "change old function")
        .unwrap();
    let mut locator = selected["id"].as_str().unwrap().to_owned();
    let mut text = String::new();
    let mut old_page = None;
    for _ in 0..20 {
        let output = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
            .current_dir(root.path())
            .env("XDG_CONFIG_HOME", root.path().join("absent-user-config"))
            .args(["open", &locator, "--json"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let page: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(page["breadcrumb"], "commit › 2026-01-02 › History Fixture");
        assert!(page["text"].as_str().unwrap().len() <= 32);
        assert!(page["path"].is_null());
        text.push_str(page["text"].as_str().unwrap());
        let Some(next) = page["continuation_id"].as_str() else {
            break;
        };
        locator = next.to_owned();
        old_page = Some(locator.clone());
    }
    assert!(
        text.contains("A body line.\nSecond body line.\nfiles:\ncode.rs\nsymbols:"),
        "{text}"
    );
    let db = Connection::open(root.path().join(".deepkit/code.db")).unwrap();
    db.execute("DELETE FROM unit_sources WHERE source_kind='commit'", [])
        .unwrap();
    let output = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root.path())
        .env("XDG_CONFIG_HOME", root.path().join("absent-user-config"))
        .args(["open", &old_page.unwrap(), "--json"])
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "lost provenance must invalidate an old page"
    );
}

#[test]
fn oversized_commit_metadata_degrades_visibly_and_retries_after_cap_increase() {
    let (root, _) = fixture();
    config(root.path(), 2000, 2000);
    let path = root.path().join(".deepkit/config.toml");
    let original = fs::read_to_string(&path).unwrap();
    fs::write(
        &path,
        original.replace(
            "max_commits=2000",
            "max_commits=2000\nmax_file_bytes='128 B'",
        ),
    )
    .unwrap();
    let capped = index(root.path());
    assert_eq!(capped["history"]["state"], "capped");
    assert_eq!(capped["history"]["input_caps"]["max_file_bytes commit"], 3);
    assert_eq!(capped["history"]["inserted"], 0);
    assert_eq!(index(root.path())["history"]["visited"], 0);
    fs::write(&path, original).unwrap();
    assert_eq!(index(root.path())["history"]["inserted"], 3);
}

#[test]
fn history_gitlinks_do_not_require_the_submodule_object_database() {
    let (root, _) = fixture();
    git(
        root.path(),
        &[
            "update-index",
            "--add",
            "--cacheinfo",
            "160000,0123456789012345678901234567890123456789,submodule",
        ],
    );
    let sha = commit(root.path(), 4, "add unavailable submodule");
    config(root.path(), 2000, 2000);
    let result = index(root.path());
    assert_eq!(result["history"]["inserted"], 4);
    assert_eq!(result["history"]["incomplete_diffs"], 1);
    let db = Connection::open(root.path().join(".deepkit/code.db")).unwrap();
    let unknown: bool = db
        .query_row(
            "SELECT additions IS NULL AND deletions IS NULL FROM commits WHERE lower(hex(sha))=?1",
            [&sha],
            |r| r.get(0),
        )
        .unwrap();
    assert!(unknown);
}
