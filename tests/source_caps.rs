//! Public format-byte admission and unchanged-source recovery; v1.md §3.5.
use rusqlite::Connection;
use serde_json::Value;
use std::{error::Error, fs, path::Path, process::Command};
fn run(root: &Path, args: &[&str]) -> Result<Value, Box<dyn Error>> {
    let out = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .args(args)
        .output()?;
    assert!(
        out.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    Ok(serde_json::from_slice(&out.stdout)?)
}
type Identity = (String, String, String);

fn units(root: &Path) -> Result<Vec<Identity>, Box<dyn Error>> {
    let db = Connection::open(root.join(".deepkit/code.db"))?;
    Ok(db.prepare("SELECT f.path,hex(u.hash),hex(u.content_hash) FROM units u JOIN files f ON f.id=u.file_id ORDER BY f.path,u.hash")?.query_map([],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?)))?.collect::<Result<_,_>>()?)
}
#[test]
fn format_caps_retain_old_documents_admit_only_l0_for_new_sources_and_resume()
-> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let root = root.path();
    fs::write(
        root.join("old.md"),
        "# Kept\n\nA previously indexed document.\n",
    )?;
    fs::write(root.join("code.rs"), "pub fn retained() {}\n")?;
    run(root, &["index", "--level", "1", "--json"])?;
    let before = units(root)?;
    assert!(before.iter().any(|u| u.0 == "old.md"));
    fs::write(
        root.join("new.md"),
        "# New\n\nA new document exceeding the byte cap.\n",
    )?;
    // Refusal happens before any image decoder; no valid raster is required here.
    fs::write(root.join("new.png"), b"larger than four bytes")?;
    let config = root.join(".deepkit/config.toml");
    fs::write(
        &config,
        "[limits]\nmax_doc_bytes='8 B'\nmax_image_bytes='4 B'\n[llm]\nenabled=true\n",
    )?;
    let capped = run(root, &["index", "--level", "1", "--json"])?;
    assert_eq!(capped["units_indexed"], 0);
    assert_eq!(units(root)?, before);
    let status = run(root, &["status", "--json"])?;
    for value in [&capped, &status] {
        let deferred = value["l1_deferred"].as_array().unwrap();
        for (path, reason, retained) in [
            ("old.md", "max_doc_bytes", true),
            ("new.md", "max_doc_bytes", false),
            ("new.png", "max_image_bytes", false),
        ] {
            let item = deferred.iter().find(|f| f["path"] == path).unwrap();
            assert_eq!(item["reason"], reason);
            assert_eq!(item["retained_snapshot"], retained);
        }
    }
    // Raising doc allowance repairs unchanged documents and retains code identities.
    fs::write(
        &config,
        "[limits]\nmax_doc_bytes='1024 B'\nmax_image_bytes='4 B'\n[llm]\nenabled=true\n",
    )?;
    let resumed = run(root, &["index", "--level", "1", "--json"])?;
    assert!(resumed["units_indexed"].as_u64().unwrap() > 0);
    let after = units(root)?;
    for old in before {
        assert!(after.contains(&old));
    }
    assert!(after.iter().any(|u| u.0 == "new.md"));
    let repeat = run(root, &["index", "--level", "1", "--json"])?;
    assert_eq!(repeat["units_indexed"], 0);
    assert_eq!(units(root)?, after);
    Ok(())
}
