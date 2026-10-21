//! Public Office extraction, source opening and cap recovery; v1.md §6.2.
use rusqlite::Connection;
use serde_json::Value;
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
        .query_map([], |row| row.get(0))?
        .collect::<Result<_, _>>()?)
}

#[test]
fn office_formats_are_searchable_openable_and_reindex_is_a_noop() -> Result<(), Box<dyn Error>> {
    let temp = tempfile::tempdir()?;
    let root = temp.path();
    let cases = [
        ("handbook.docx", "interrupted update", "Recovery"),
        ("inventory.xlsx", "Refresh changed sources", "sheet 2"),
        ("workflow.pptx", "original source", "slide 2"),
    ];
    for (name, _, _) in cases {
        fs::copy(format!("fixtures/docs/{name}"), root.join(name))?;
    }
    run(root, &["index", "--level", "1", "--json"])?;
    let original = identities(root)?;
    assert_eq!(original.len(), 6);
    for (name, query, header) in cases {
        let result = run(
            root,
            &[
                "search",
                query,
                "--mode",
                "lexical",
                "--no-rerank",
                "--scope",
                name,
                "--json",
            ],
        )?;
        let hit = result["hits"]
            .as_array()
            .unwrap()
            .first()
            .ok_or("missing Office hit")?;
        assert_eq!(hit["path"], name);
        let opened = run(root, &["open", hit["id"].as_str().unwrap(), "--json"])?;
        assert!(opened["text"].as_str().unwrap().contains(query), "{opened}");
        assert!(opened.to_string().contains(header), "{opened}");
    }
    let repeated = run(root, &["index", "--level", "1", "--json"])?;
    assert_eq!(repeated["units_indexed"], 0);
    assert_eq!(identities(root)?, original);
    Ok(())
}

#[test]
fn byte_caps_precede_office_parsing_and_l0_reasons_survive_reindex() -> Result<(), Box<dyn Error>> {
    let temp = tempfile::tempdir()?;
    let root = temp.path();
    fs::create_dir(root.join(".deepkit"))?;
    let config = root.join(".deepkit/config.toml");
    fs::write(&config, "[limits]\nmax_doc_bytes='4 B'\n")?;
    for name in [
        "handbook.docx",
        "inventory.xlsx",
        "workflow.pptx",
        "encrypted.docx",
    ] {
        fs::copy(format!("fixtures/docs/{name}"), root.join(name))?;
    }
    for extension in ["docx", "xlsx", "pptx"] {
        fs::write(
            root.join(format!("corrupt.{extension}")),
            b"not an Office archive",
        )?;
    }
    run(root, &["index", "--level", "1", "--json"])?;
    assert!(identities(root)?.is_empty());
    let status = run(root, &["status", "--json"])?;
    let deferred = status["l1_deferred"].as_array().unwrap();
    assert_eq!(deferred.len(), 7);
    assert!(deferred.iter().all(|f| f["reason"] == "max_doc_bytes"));
    fs::write(&config, "[limits]\nmax_doc_bytes='1 MiB'\n")?;
    for _ in 0..2 {
        run(root, &["index", "--level", "1", "--json"])?;
        assert_eq!(identities(root)?.len(), 6);
        let status = run(root, &["status", "--json"])?;
        let deferred = status["l1_deferred"].as_array().unwrap();
        assert_eq!(deferred.len(), 4);
        for extension in ["docx", "xlsx", "pptx"] {
            assert!(
                deferred
                    .iter()
                    .any(|f| f["path"] == format!("corrupt.{extension}")
                        && f["reason"] == "malformed_office")
            );
        }
        assert!(
            deferred
                .iter()
                .any(|f| f["path"] == "encrypted.docx" && f["reason"] == "encrypted_office")
        );
    }
    Ok(())
}

#[test]
fn lowered_office_unit_cap_preserves_previous_units_and_raising_recovers()
-> Result<(), Box<dyn Error>> {
    let temp = tempfile::tempdir()?;
    let root = temp.path();
    fs::copy("fixtures/docs/inventory.xlsx", root.join("inventory.xlsx"))?;
    run(root, &["index", "--level", "1", "--json"])?;
    let original = identities(root)?;
    assert_eq!(original.len(), 2);
    let config = root.join(".deepkit/config.toml");
    fs::write(&config, "[limits]\nmax_units_per_file=1\n")?;
    run(root, &["index", "--level", "1", "--json"])?;
    assert_eq!(identities(root)?, original);
    let status = run(root, &["status", "--json"])?;
    assert!(
        status["l1_deferred"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f["path"] == "inventory.xlsx"
                && f["reason"] == "max_units_per_file"
                && f["retained_snapshot"] == true)
    );
    fs::write(&config, "[limits]\nmax_units_per_file=2\n")?;
    run(root, &["index", "--level", "1", "--json"])?;
    assert_eq!(identities(root)?, original);
    assert!(
        run(root, &["status", "--json"])?["l1_deferred"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        run(root, &["index", "--level", "1", "--json"])?["units_indexed"],
        0
    );
    Ok(())
}
