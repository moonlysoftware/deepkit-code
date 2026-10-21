//! Public documentation/test navigation and deterministic updates; v1.md §6.5.
use rusqlite::Connection;
use serde_json::Value;
use std::{fs, path::Path, process::Command};
fn run(root: &Path, args: &[&str]) -> std::process::Output {
    Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .env("XDG_CONFIG_HOME", root.join("absent-config"))
        .env("DKC_MODELS_DIR", root.join("absent-models"))
        .args(args)
        .output()
        .unwrap()
}
fn success(root: &Path, args: &[&str]) -> Value {
    let result = run(root, args);
    assert!(
        result.status.success(),
        "{args:?}: {} {}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    serde_json::from_slice(&result.stdout).unwrap()
}
fn fixture() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    for path in [
        "src/retry.rs",
        "tests/retry.rs",
        "README.md",
        "docs/paths.md",
        "settings.toml",
    ] {
        fs::create_dir_all(root.path().join(path).parent().unwrap()).unwrap();
        fs::copy(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/linker")
                .join(path),
            root.path().join(path),
        )
        .unwrap();
    }
    fs::create_dir(root.path().join(".deepkit")).unwrap();
    fs::write(root.path().join(".deepkit/config.toml"),"[index]\nrepo_name='linker'\n[embedder]\nprofile='local'\n[llm]\nenabled=false\n[commits]\nenabled=false\n[rerank]\nmode='off'\n[limits]\nquery_timeout_ms=10000\n").unwrap();
    root
}
fn index(root: &Path) {
    success(root, &["index", "--level", "1", "--json"]);
}
type Edge = (i64, String, Option<i64>, String, i64, Option<i64>);
fn edges(db: &Connection) -> Vec<Edge> {
    db.prepare("SELECT src_symbol,dst_name,dst_symbol,kind,file_id,line FROM edges ORDER BY src_symbol,dst_name,dst_symbol,kind,line").unwrap().query_map([],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?))).unwrap().collect::<Result<_,_>>().unwrap()
}
#[test]
fn documents_tests_noop_and_configuration_reclassification() {
    let root = fixture();
    let root = root.path();
    index(root);
    let db = Connection::open(root.join(".deepkit/code.db")).unwrap();
    let before = edges(&db);
    assert!(
        before
            .iter()
            .any(|e| e.1 == "should_retry" && e.3 == "doc_mention")
    );
    assert!(
        before
            .iter()
            .any(|e| e.1 == "should_retry" && e.3 == "test_of")
    );
    assert!(before.iter().any(|e| e.1 == "helper" && e.3 == "ref"));
    assert!(
        !before
            .iter()
            .any(|e| e.1 == "prose_only" && e.3 == "doc_mention")
    );
    assert!(!before.iter().any(|e| e.1.contains("settings.toml")));
    let code = db
        .query_row(
            "SELECT id FROM symbols WHERE name='should_retry'",
            [],
            |r| r.get::<_, i64>(0),
        )
        .unwrap();
    assert!(db.execute("INSERT INTO edges(src_symbol,dst_name,kind,file_id) VALUES (?1,'configuration','config_of',(SELECT file_id FROM symbols WHERE id=?1))",[code]).is_err());
    let symbol = success(root, &["symbol", "should_retry", "--json"]);
    assert_eq!(symbol[0]["tests"], serde_json::json!(["tests/retry.rs"]));
    assert_eq!(symbol[0]["documented_in"].as_array().unwrap().len(), 2);
    for doc in symbol[0]["documented_in"].as_array().unwrap() {
        let opened = success(root, &["open", doc["id"].as_str().unwrap(), "--json"]);
        assert_eq!(opened["header"], doc["header"]);
    }
    let refs = success(root, &["refs", "should_retry", "--tests-only", "--json"]);
    assert_eq!(refs["groups"].as_array().unwrap().len(), 1, "{refs}");
    assert_eq!(refs["groups"][0]["path"], "tests/retry.rs");
    index(root);
    assert_eq!(edges(&db), before);
    success(root, &["index", "--level", "1", "--full", "--json"]);
    assert_eq!(edges(&db), before);
    fs::write(
        root.join(".deepkit/config.local.toml"),
        "[tests]\nglobs=['custom/**']\n",
    )
    .unwrap();
    index(root);
    assert!(edges(&db).iter().all(|e| e.3 != "test_of"));
    assert_eq!(
        success(root, &["symbol", "should_retry", "--json"])[0]["tests"],
        serde_json::json!([])
    );
}
#[test]
fn changed_document_refuses_stale_navigation_then_reindexes() {
    let root = fixture();
    let root = root.path();
    index(root);
    fs::write(
        root.join(".deepkit/config.local.toml"),
        "[search]\nlazy_resync=false\n",
    )
    .unwrap();
    fs::write(
        root.join("README.md"),
        "# Updated\nNo explicit identifiers remain.\n",
    )
    .unwrap();
    assert!(
        !run(root, &["symbol", "should_retry", "--json"])
            .status
            .success()
    );
    index(root);
    let symbol = success(root, &["symbol", "should_retry", "--json"]);
    assert_eq!(symbol[0]["documented_in"].as_array().unwrap().len(), 1);
}

#[test]
fn lazy_document_and_test_source_repair_updates_links() {
    let root = fixture();
    let root = root.path();
    index(root);
    fs::write(root.join("README.md"), "# Updated\nUse `prose_only`.\n").unwrap();
    fs::write(root.join("tests/retry.rs"), "fn test_should_retry() {}\n").unwrap();
    let symbol = success(root, &["symbol", "should_retry", "--json"]);
    assert_eq!(symbol[0]["tests"], serde_json::json!([]));
    assert_eq!(symbol[0]["documented_in"].as_array().unwrap().len(), 1);
    assert_eq!(
        success(root, &["symbol", "prose_only", "--json"])[0]["documented_in"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn configured_graph_weight_boosts_mentioned_document_sections() {
    let root = fixture();
    let root = root.path();
    index(root);
    let enabled = success(
        root,
        &[
            "search",
            "should_retry",
            "--mode",
            "lexical",
            "--verbose",
            "--json",
        ],
    );
    let enabled_doc = enabled["hits"]
        .as_array()
        .unwrap()
        .iter()
        .find(|h| h["path"] == "README.md")
        .unwrap();
    assert!(!enabled_doc["rrf"]["graph"].is_null(), "{enabled}");
    fs::write(
        root.join(".deepkit/config.local.toml"),
        "[search]\ngraph_prior_weight=0.0\n",
    )
    .unwrap();
    let disabled = success(
        root,
        &[
            "search",
            "should_retry",
            "--mode",
            "lexical",
            "--verbose",
            "--json",
        ],
    );
    let disabled_doc = disabled["hits"]
        .as_array()
        .unwrap()
        .iter()
        .find(|h| h["path"] == "README.md")
        .unwrap();
    assert!(disabled_doc["rrf"]["graph"].is_null(), "{disabled}");
    assert!(enabled_doc["score"].as_f64().unwrap() > disabled_doc["score"].as_f64().unwrap());
}
