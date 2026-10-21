//! Public PDF extraction, searchable text and catalogue recovery; v1.md §6.2.
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
fn digital_pdf_is_searchable_and_lowered_caps_retain_then_recover() -> Result<(), Box<dyn Error>> {
    let temp = tempfile::tempdir()?;
    let root = temp.path();
    fs::copy("fixtures/docs/manual.pdf", root.join("manual.pdf"))?;
    run(root, &["index", "--level", "1", "--json"])?;
    let original = identities(root)?;
    assert_eq!(original.len(), 2);
    let hits = run(
        root,
        &[
            "search",
            "original context",
            "--mode",
            "lexical",
            "--no-rerank",
            "--json",
        ],
    )?;
    assert!(hits.to_string().contains("manual.pdf"), "{hits}");
    let config = root.join(".deepkit/config.toml");
    fs::write(&config, "[limits]\nmax_pdf_pages=1\n")?;
    run(root, &["index", "--level", "1", "--json"])?;
    assert_eq!(identities(root)?, original);
    let status = run(root, &["status", "--json"])?;
    assert!(
        status["l1_deferred"]
            .as_array()
            .unwrap()
            .iter()
            .any(|file| file["path"] == "manual.pdf"
                && file["reason"] == "max_pdf_pages"
                && file["retained_snapshot"] == true)
    );
    fs::write(&config, "[limits]\nmax_pdf_pages=2\n")?;
    run(root, &["index", "--level", "1", "--json"])?;
    assert_eq!(identities(root)?, original);
    let repeated = run(root, &["index", "--level", "1", "--json"])?;
    assert_eq!(repeated["units_indexed"], 0);
    Ok(())
}

#[test]
fn new_page_capped_pdf_publishes_prefix_and_l0_reasons_survive_reindex()
-> Result<(), Box<dyn Error>> {
    let temp = tempfile::tempdir()?;
    let root = temp.path();
    fs::create_dir(root.join(".deepkit"))?;
    fs::write(
        root.join(".deepkit/config.toml"),
        "[limits]\nmax_pdf_pages=1\n",
    )?;
    for name in ["manual", "scanned", "encrypted"] {
        fs::copy(
            format!("fixtures/docs/{name}.pdf"),
            root.join(format!("{name}.pdf")),
        )?;
    }
    fs::write(root.join("corrupt.pdf"), b"%PDF-1.7\ninvalid")?;
    for _ in 0..2 {
        run(root, &["index", "--level", "1", "--json"])?;
        let db = Connection::open(root.join(".deepkit/code.db"))?;
        let rows = db.prepare("SELECT f.path,f.needs_vlm,f.truncated,f.l1_deferred_reason,(SELECT count(*) FROM units WHERE file_id=f.id) FROM files f ORDER BY path")?
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, bool>(1)?, r.get::<_, bool>(2)?, r.get::<_, Option<String>>(3)?, r.get::<_, i64>(4)?)))?
            .collect::<Result<Vec<_>,_>>()?;
        assert_eq!(
            rows,
            vec![
                (
                    "corrupt.pdf".into(),
                    false,
                    false,
                    Some("malformed_pdf".into()),
                    0
                ),
                (
                    "encrypted.pdf".into(),
                    false,
                    false,
                    Some("encrypted_pdf".into()),
                    0
                ),
                (
                    "manual.pdf".into(),
                    false,
                    true,
                    Some("max_pdf_pages".into()),
                    1
                ),
                (
                    "scanned.pdf".into(),
                    true,
                    false,
                    Some("needs_vlm".into()),
                    0
                ),
            ]
        );
        let status = run(root, &["status", "--json"])?;
        for reason in [
            "malformed_pdf",
            "encrypted_pdf",
            "max_pdf_pages",
            "needs_vlm",
        ] {
            assert!(
                status["l1_deferred"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|f| f["reason"] == reason)
            );
        }
    }
    Ok(())
}

#[test]
fn document_bytes_refuse_pdf_before_parsing_and_recover_when_raised() -> Result<(), Box<dyn Error>>
{
    let temp = tempfile::tempdir()?;
    let root = temp.path();
    fs::create_dir(root.join(".deepkit"))?;
    let config = root.join(".deepkit/config.toml");
    fs::write(&config, "[limits]\nmax_doc_bytes='4 B'\n")?;
    fs::copy("fixtures/docs/invoice.pdf", root.join("invoice.pdf"))?;
    fs::write(root.join("corrupt.pdf"), b"not a valid PDF")?;
    run(root, &["index", "--level", "1", "--json"])?;
    assert!(identities(root)?.is_empty());
    let status = run(root, &["status", "--json"])?;
    let refused = status["l1_deferred"].as_array().unwrap();
    assert_eq!(refused.len(), 2);
    assert!(refused.iter().all(|file| file["reason"] == "max_doc_bytes"));
    fs::write(&config, "[limits]\nmax_doc_bytes='1 MiB'\n")?;
    run(root, &["index", "--level", "1", "--json"])?;
    assert_eq!(identities(root)?.len(), 1);
    let status = run(root, &["status", "--json"])?;
    assert!(
        status["l1_deferred"]
            .as_array()
            .unwrap()
            .iter()
            .any(|file| file["path"] == "corrupt.pdf" && file["reason"] == "malformed_pdf")
    );
    Ok(())
}
