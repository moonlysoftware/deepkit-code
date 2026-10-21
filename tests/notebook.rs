//! Public notebook source search and global cell limits; v1.md §6.2.
use rusqlite::Connection;
use serde_json::{Value, json};
use std::{error::Error, fs, path::Path, process::Command};
fn run(root: &Path, args: &[&str]) -> Result<Value, Box<dyn Error>> {
    let output = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .args(args)
        .output()?;
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}
fn identities(root: &Path) -> Result<Vec<String>, Box<dyn Error>> {
    let db = Connection::open(root.join(".deepkit/code.db"))?;
    Ok(db
        .prepare("SELECT hex(hash) FROM units ORDER BY hash")?
        .query_map([], |r| r.get(0))?
        .collect::<Result<_, _>>()?)
}
#[test]
fn notebook_search_and_open_use_sources_and_exclude_outputs() -> Result<(), Box<dyn Error>> {
    let temp = tempfile::tempdir()?;
    let root = temp.path();
    // Cell line numbers must not be validated against one-line JSON storage.
    let notebook: Value = serde_json::from_slice(&fs::read("fixtures/docs/walkthrough.ipynb")?)?;
    fs::write(
        root.join("walkthrough.ipynb"),
        serde_json::to_vec(&notebook)?,
    )?;
    run(root, &["index", "--level", "1", "--json"])?;
    for (query, locator) in [("normalize_name", "cell 2"), ("normalized name", "cell 5")] {
        let found = run(
            root,
            &[
                "search",
                query,
                "--mode",
                "lexical",
                "--no-rerank",
                "--json",
            ],
        )?;
        let hit = found["hits"]
            .as_array()
            .unwrap()
            .first()
            .ok_or("missing notebook hit")?;
        assert_eq!(hit["path"], "walkthrough.ipynb");
        assert_eq!(hit["fresh"]["state"], "fresh", "{hit}");
        let opened = run(root, &["open", hit["id"].as_str().unwrap(), "--json"])?;
        assert!(opened["text"].as_str().unwrap().contains(query), "{opened}");
        assert!(opened.to_string().contains(locator), "{opened}");
        assert_eq!(opened["lines"], json!([null, null]), "{opened}");
        assert_eq!(opened["indexed_lines"], json!([1, 3]), "{opened}");
        assert_eq!(opened["extracted_lines"], json!([1, 3]), "{opened}");
    }
    for sentinel in ["OUTPUT_SENTINEL", "ATTACHMENT_SENTINEL", "RAW_SENTINEL"] {
        let found = run(
            root,
            &[
                "search",
                sentinel,
                "--mode",
                "lexical",
                "--no-rerank",
                "--json",
            ],
        )?;
        assert!(found["hits"].as_array().unwrap().is_empty(), "{found}");
    }
    let db = Connection::open(root.join(".deepkit/code.db"))?;
    // Preserve code/prose weighting and Python's configured identifier rules.
    for (query, count) in [
        ("body: normalize_name", 1),
        ("body: return", 1),
        ("body: stable", 0),
        ("body: catalogue", 0),
        ("prose: \"stable lookup\"", 1),
    ] {
        assert_eq!(
            db.query_row(
                "SELECT count(*) FROM fts_units WHERE fts_units MATCH ?",
                [query],
                |r| r.get::<_, i64>(0)
            )?,
            count,
            "{query}"
        );
    }
    let before = identities(root)?;
    assert_eq!(
        run(root, &["index", "--level", "1", "--json"])?["units_indexed"],
        0
    );
    assert_eq!(identities(root)?, before);
    Ok(())
}
#[test]
fn two_thousand_cells_publish_only_admitted_prefix_and_retain_lowered_caps()
-> Result<(), Box<dyn Error>> {
    let temp = tempfile::tempdir()?;
    let root = temp.path();
    fs::create_dir(root.join(".deepkit"))?;
    let config = root.join(".deepkit/config.toml");
    fs::write(&config, "[limits]\nmax_units_per_file=500\n")?;
    let doc = json!({"nbformat":4,"cells":(0..2000).map(|n|json!({"cell_type":"markdown","source":format!("# Section {n}\n\nNotebook text.")})).collect::<Vec<_>>()});
    fs::write(root.join("large.ipynb"), serde_json::to_vec(&doc)?)?;
    run(root, &["index", "--level", "1", "--json"])?;
    let original = identities(root)?;
    assert_eq!(original.len(), 500);
    let db = Connection::open(root.join(".deepkit/code.db"))?;
    assert!(db.query_row(
        "SELECT truncated FROM files WHERE path='large.ipynb'",
        [],
        |r| r.get::<_, bool>(0)
    )?);
    fs::write(&config, "[limits]\nmax_units_per_file=1\n")?;
    run(root, &["index", "--level", "1", "--json"])?;
    assert_eq!(identities(root)?, original);
    let status = run(root, &["status", "--json"])?;
    assert!(
        status["l1_deferred"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f["path"] == "large.ipynb" && f["retained_snapshot"] == true)
    );
    fs::write(&config, "[limits]\nmax_units_per_file=2000\n")?;
    run(root, &["index", "--level", "1", "--json"])?;
    assert_eq!(identities(root)?.len(), 2000);
    assert!(!db.query_row(
        "SELECT truncated FROM files WHERE path='large.ipynb'",
        [],
        |r| r.get::<_, bool>(0)
    )?);
    assert_eq!(
        run(root, &["index", "--level", "1", "--json"])?["units_indexed"],
        0
    );
    Ok(())
}

#[test]
fn notebook_byte_cap_precedes_parsing_and_bad_input_keeps_a_reason() -> Result<(), Box<dyn Error>> {
    let temp = tempfile::tempdir()?;
    let root = temp.path();
    fs::create_dir(root.join(".deepkit"))?;
    let config = root.join(".deepkit/config.toml");
    fs::write(&config, "[limits]\nmax_doc_bytes='4 B'\n")?;
    fs::copy(
        "fixtures/docs/walkthrough.ipynb",
        root.join("walkthrough.ipynb"),
    )?;
    fs::write(root.join("broken.ipynb"), b"{broken notebook")?;
    run(root, &["index", "--level", "1", "--json"])?;
    assert!(identities(root)?.is_empty());
    let status = run(root, &["status", "--json"])?;
    let refused = status["l1_deferred"].as_array().unwrap();
    assert_eq!(refused.len(), 2);
    assert!(refused.iter().all(|f| f["reason"] == "max_doc_bytes"));
    fs::write(&config, "[limits]\nmax_doc_bytes='1 MiB'\n")?;
    for _ in 0..2 {
        run(root, &["index", "--level", "1", "--json"])?;
        assert!(!identities(root)?.is_empty());
        let status = run(root, &["status", "--json"])?;
        let refused = status["l1_deferred"].as_array().unwrap();
        assert_eq!(refused.len(), 1);
        assert_eq!(refused[0]["path"], "broken.ipynb");
        assert_eq!(refused[0]["reason"], "malformed_notebook");
    }
    Ok(())
}
