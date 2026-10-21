//! Real Git history ordering, capped provenance and public symbol/PR navigation; v1 §6.6.
use rusqlite::Connection;
use serde_json::Value;
use std::{fs, path::Path, process::Command};
fn git(root: &Path, args: &[&str]) -> String {
    let result = Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    String::from_utf8(result.stdout).unwrap().trim().to_owned()
}
fn command(root: &Path, args: &[&str]) -> std::process::Output {
    Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .env("XDG_CONFIG_HOME", root.join("absent-user-config"))
        .env("DKC_MODELS_DIR", root.join("absent-model-cache"))
        .args(args)
        .output()
        .unwrap()
}
fn success(root: &Path, args: &[&str]) -> Value {
    let output = command(root, args);
    assert!(
        output.status.success(),
        "{args:?}: {} {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
fn index(root: &Path) -> Value {
    success(root, &["index", "--level", "1", "--json"])
}
fn commit(root: &Path, n: usize) -> String {
    fs::write(
        root.join("code.rs"),
        format!("pub fn historical() -> i32 {{\n    {n}\n}}\n"),
    )
    .unwrap();
    git(root, &["add", "code.rs"]);
    let result = Command::new("git")
        .current_dir(root)
        .env("GIT_AUTHOR_DATE", format!("@{} +0000", 1767225600 + n * 60))
        .env(
            "GIT_COMMITTER_DATE",
            format!("@{} +0000", 1767225600 + n * 60),
        )
        .args(["commit", "-qm", &format!("change {n} (#{n})")])
        .output()
        .unwrap();
    assert!(result.status.success());
    git(root, &["rev-parse", "HEAD"])
}
fn fixture(count: usize) -> (tempfile::TempDir, Vec<String>) {
    let root = tempfile::tempdir().unwrap();
    git(root.path(), &["init", "-q", "-b", "main"]);
    git(root.path(), &["config", "user.name", "History Fixture"]);
    git(
        root.path(),
        &["config", "user.email", "history@example.invalid"],
    );
    let shas = (1..=count).map(|n| commit(root.path(), n)).collect();
    fs::create_dir(root.path().join(".deepkit")).unwrap();
    fs::write(root.path().join(".deepkit/config.toml"),"[index]\nrepo_name='fixture'\n[embedder]\nprofile='local'\n[llm]\nenabled=false\n[rerank]\nmode='off'\n[limits]\nquery_timeout_ms=10000\nmax_result_tokens=50000\n").unwrap();
    (root, shas)
}
fn history(root: &Path) -> (i64, String, String) {
    let db = Connection::open(root.join(".deepkit/code.db")).unwrap();
    db.query_row("SELECT u.id,'u_'||substr(lower(hex(u.hash)),1,12),c.text FROM symbols s JOIN units u ON u.id=s.history_unit JOIN unit_contents c ON c.hash=u.content_hash WHERE s.name='historical'",[],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).unwrap()
}
#[test]
fn chronological_latest_hundred_sources_pr_pointers_and_noop_identity() {
    let (root, shas) = fixture(105);
    let root = root.path();
    let indexed = index(root);
    assert!(indexed["histories"]["inserted"].as_u64().unwrap() > 0);
    let (unit, id, text) = history(root);
    let lines = text.lines().collect::<Vec<_>>();
    assert_eq!(lines.len(), 100);
    for (line, sha) in lines.iter().zip(&shas[5..]) {
        assert!(line.starts_with(&sha[..8]), "{line} {sha}");
        assert!(line.contains("History Fixture: change"));
    }
    let db = Connection::open(root.join(".deepkit/code.db")).unwrap();
    let sources=db.prepare("SELECT lower(hex(c.sha)) FROM unit_sources s JOIN commits c ON s.source_kind='commit' AND s.source_id=c.id WHERE s.unit_id=?1 ORDER BY c.author_time,c.sha").unwrap().query_map([unit],|r|r.get::<_,String>(0)).unwrap().collect::<Result<Vec<_>,_>>().unwrap();
    assert_eq!(sources, shas[5..]);
    let symbols = success(root, &["symbol", "historical", "--json"]);
    let commits = symbols[0]["history"]["commits"].as_array().unwrap();
    assert_eq!(commits.len(), 100);
    for (entry, sha) in commits.iter().zip(&shas[5..]) {
        assert_eq!(entry["sha"], *sha);
        assert!(entry["pr"]["id"].as_str().unwrap().starts_with("u_"));
    }
    let opened = success(root, &["open", &id, "--json"]);
    assert_eq!(opened["kind"], "history");
    assert_eq!(opened["breadcrumb"], "fixture › code.rs");
    assert_eq!(opened["sources"].as_array().unwrap().len(), 100);
    let rerun = index(root);
    assert_eq!(rerun["histories"]["inserted"], 0);
    assert_eq!(history(root).0, unit);
    assert_eq!(
        db.query_row("SELECT count(*) FROM jobs", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    let config = root.join(".deepkit/config.toml");
    fs::write(
        &config,
        fs::read_to_string(&config)
            .unwrap()
            .replace("max_result_tokens=50000", "max_result_tokens=700"),
    )
    .unwrap();
    let capped = success(root, &["symbol", "historical", "--json"]);
    assert_eq!(capped[0]["history"]["truncated"], true);
    assert!(capped[0]["history"]["commits"].as_array().unwrap().len() < 100);
    assert_eq!(
        capped[0]["history"]["commits"]
            .as_array()
            .unwrap()
            .last()
            .unwrap()["sha"],
        shas[104]
    );
}
#[test]
fn rank_cap_retains_history_and_new_commit_replaces_only_changed_sources() {
    let (root, _) = fixture(3);
    let root = root.path();
    index(root);
    let old = history(root);
    let config = root.join(".deepkit/config.toml");
    let original = fs::read_to_string(&config).unwrap();
    fs::write(&config, format!("{original}\n[enrich]\ntop_symbols=0\n")).unwrap();
    index(root);
    assert_eq!(history(root), old);
    let disabled = success(root, &["symbol", "historical", "--json"]);
    assert!(disabled[0].get("history").is_none());
    assert!(!command(root, &["open", &old.1, "--json"]).status.success());
    fs::write(&config, &original).unwrap();
    index(root);
    assert_eq!(history(root), old);
    commit(root, 4);
    index(root);
    let new = history(root);
    assert_ne!(old.1, new.1);
    assert_eq!(new.2.lines().count(), 4);
    assert!(!command(root, &["open", &old.1, "--json"]).status.success());
    let db = Connection::open(root.join(".deepkit/code.db")).unwrap();
    db.execute("DELETE FROM unit_sources WHERE unit_id=?1 AND source_id=(SELECT min(source_id) FROM unit_sources WHERE unit_id=?1)",[new.0]).unwrap();
    let local = root.join(".deepkit/config.local.toml");
    fs::write(&local, "[search]\nlazy_resync=false\n").unwrap();
    assert!(
        !command(root, &["symbol", "historical", "--json"])
            .status
            .success()
    );
    assert!(!command(root, &["open", &new.1, "--json"]).status.success());
    fs::remove_file(local).unwrap();
    // Lazy symbol repair regenerates verified provenance; stale open never
    // bypasses verification, and a later no-op index keeps the repair.
    assert_eq!(
        success(root, &["symbol", "historical", "--json"])[0]["history"]["commits"]
            .as_array()
            .unwrap()
            .len(),
        4
    );
    index(root);
    assert_eq!(
        success(root, &["symbol", "historical", "--json"])[0]["history"]["commits"]
            .as_array()
            .unwrap()
            .len(),
        4
    );
}

#[test]
fn rank_prefix_and_equal_names_in_different_files_keep_distinct_sources() {
    let (root, shas) = fixture(2);
    let root = root.path();
    fs::write(root.join("other.rs"), "pub fn historical() -> i32 { 99 }\n").unwrap();
    git(root, &["add", "other.rs"]);
    git(root, &["commit", "-qm", "isolated definition"]);
    let other = git(root, &["rev-parse", "HEAD"]);
    let config = root.join(".deepkit/config.toml");
    let base = fs::read_to_string(&config).unwrap();
    fs::write(&config, format!("{base}\n[enrich]\ntop_symbols=0\n")).unwrap();
    index(root);
    let db = Connection::open(root.join(".deepkit/code.db")).unwrap();
    db.execute("UPDATE symbols SET pagerank=CASE WHEN file_id=(SELECT id FROM files WHERE path='other.rs') THEN 1.0 ELSE 0.0 END",[]).unwrap();
    fs::write(&config, format!("{base}\n[enrich]\ntop_symbols=1\n")).unwrap();
    assert_eq!(index(root)["histories"]["selected"], 1);
    let result = success(root, &["symbol", "historical", "--json"]);
    let rows = result.as_array().unwrap();
    assert_eq!(rows.len(), 2);
    let selected = rows.iter().find(|r| r["path"] == "other.rs").unwrap();
    assert_eq!(selected["history"]["commits"].as_array().unwrap().len(), 1);
    assert_eq!(selected["history"]["commits"][0]["sha"], other);
    assert!(
        rows.iter()
            .find(|r| r["path"] == "code.rs")
            .unwrap()
            .get("history")
            .is_none()
    );
    fs::write(&config, base).unwrap();
    index(root);
    let result = success(root, &["symbol", "historical", "--json"]);
    let original = result
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["path"] == "code.rs")
        .unwrap();
    assert_eq!(
        original["history"]["commits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["sha"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>(),
        shas
    );
}
