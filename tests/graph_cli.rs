use std::{error::Error, fs, path::Path, process::Command};

use assert_cmd::Command as AssertCommand;
use serde_json::Value;
use tempfile::TempDir;

#[test]
fn symbol_and_refs_commands_keep_approximate_graph_results_disambiguated()
-> Result<(), Box<dyn Error>> {
    let repository = TempDir::new()?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/graph");
    copy_tree(&fixture, repository.path())?;
    // A path mention resolves to the definitions in this file. This keeps the
    // duplicate in `src/other.rs` outside the exact documentation edge and
    // catches cross-symbol leakage when link lookups are batched.
    fs::write(
        repository.path().join("docs/note.md"),
        "# API note\n\nSee `src/api.rs` for the duplicate shared increment behavior.\n",
    )?;
    let status = Command::new("git")
        .args(["init", "--quiet"])
        .current_dir(repository.path())
        .status()?;
    assert!(status.success());
    let status = Command::new("git")
        .args(["add", "src", "docs"])
        .current_dir(repository.path())
        .status()?;
    assert!(status.success());
    let status = Command::new("git")
        .args([
            "-c",
            "user.name=Deepkit Test",
            "-c",
            "user.email=deepkit@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "graph fixture",
        ])
        .current_dir(repository.path())
        .status()?;
    assert!(status.success());

    let _ = dkc(repository.path(), ["init", "--json"])?;
    let _ = dkc(repository.path(), ["index", "--level", "1", "--json"])?;

    let symbols = dkc(repository.path(), ["symbol", "duplicate", "--json"])?;
    let symbols = symbols.as_array().ok_or("symbol did not return an array")?;
    assert_eq!(symbols.len(), 2);
    let mut paths = symbols
        .iter()
        .filter_map(|symbol| symbol["path"].as_str())
        .collect::<Vec<_>>();
    paths.sort_unstable();
    assert_eq!(paths, vec!["src/api.rs", "src/other.rs"]);
    let api_symbol = symbols
        .iter()
        .find(|symbol| symbol["path"] == "src/api.rs")
        .ok_or("missing api duplicate")?;
    assert!(
        api_symbol["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("s_"))
    );
    assert!(
        api_symbol["doc"]
            .as_str()
            .is_some_and(|doc| doc.contains("incremented input"))
    );
    assert_eq!(api_symbol["approximate"], true);
    let other_symbol = symbols
        .iter()
        .find(|symbol| symbol["path"] == "src/other.rs")
        .ok_or("missing other duplicate")?;
    assert_eq!(other_symbol["tests"], serde_json::json!([]));
    assert_eq!(other_symbol["documented_in"], serde_json::json!([]));
    assert_eq!(other_symbol["approximate"], true);
    assert!(api_symbol.get("summary").is_none());
    assert_eq!(api_symbol["tests"], serde_json::json!([]));
    let docs = api_symbol["documented_in"]
        .as_array()
        .ok_or("missing documented_in")?;
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0]["header"], "API note");
    let documented = dkc(
        repository.path(),
        [
            "open",
            docs[0]["id"].as_str().ok_or("missing document id")?,
            "--json",
        ],
    )?;
    assert_eq!(documented["path"], "docs/note.md");

    let symbol_id = api_symbol["id"].as_str().ok_or("missing symbol id")?;
    let opened = dkc(repository.path(), ["open", symbol_id, "--json"])?;
    assert_eq!(opened["path"], "src/api.rs");
    assert!(
        opened["text"]
            .as_str()
            .is_some_and(|text| text.contains("duplicate"))
    );

    let refs = dkc(repository.path(), ["refs", "duplicate", "--json"])?;
    assert_eq!(refs["approximate"], true);
    assert_eq!(refs["total"], 2);
    assert_eq!(
        refs["groups"],
        serde_json::json!([
            {"path": "docs/note.md", "lines": [1], "kind": "ref"},
            {"path": "src/api.rs", "lines": [7], "kind": "ref"}
        ])
    );
    let api_path = repository.path().join("src/api.rs");
    let api_original = fs::read(&api_path)?;
    fs::write(
        repository.path().join(".deepkit/config.local.toml"),
        "[search]\nlazy_resync=false\n",
    )?;
    fs::write(&api_path, "// removed all declarations\n")?;
    // One invalid source refuses the complete symbol array, including the valid duplicate.
    let refused = AssertCommand::cargo_bin("dkc")?
        .current_dir(repository.path())
        .args(["symbol", "duplicate", "--json"])
        .output()?;
    assert!(!refused.status.success());
    let refused: Value = serde_json::from_slice(&refused.stdout)?;
    assert_eq!(refused["error"]["code"], "source_unavailable");
    let mapped = dkc(repository.path(), ["map", "--json"])?;
    assert_eq!(mapped["index"]["partial"], true);
    assert_eq!(mapped["index"]["suppressed_files"], 1);
    assert!(
        !mapped["tree"]
            .as_str()
            .ok_or("missing tree")?
            .contains("api.rs")
    );
    assert!(
        mapped["tree"]
            .as_str()
            .ok_or("missing tree")?
            .contains("other.rs")
    );
    assert!(
        mapped["nodes"]
            .as_array()
            .ok_or("missing nodes")?
            .iter()
            .all(|node| node["path"] != "src/api.rs")
    );
    let text_map = AssertCommand::cargo_bin("dkc")?
        .current_dir(repository.path())
        .args(["map"])
        .output()?;
    assert!(text_map.status.success());
    assert!(String::from_utf8(text_map.stderr)?.contains("map incomplete"));
    let partial = dkc(repository.path(), ["refs", "duplicate", "--json"])?;
    assert_eq!(partial["index"]["partial"], true);
    assert_eq!(partial["index"]["suppressed_references"], 1);
    assert_eq!(partial["indexed_total"], 2);
    assert_eq!(partial["returned_total"], 1);
    assert!(partial["total"].is_null());
    assert_eq!(partial["groups"][0]["path"], "docs/note.md");
    let text = AssertCommand::cargo_bin("dkc")?
        .current_dir(repository.path())
        .args(["refs", "duplicate"])
        .output()?;
    assert!(text.status.success());
    let text = String::from_utf8(text.stdout)?;
    assert!(text.contains("reference results incomplete"));
    assert!(text.contains("docs/note.md"));
    assert!(!text.contains("src/api.rs"));
    fs::write(&api_path, &api_original)?;
    assert_eq!(
        dkc(repository.path(), ["refs", "duplicate", "--json"])?["groups"],
        refs["groups"]
    );

    fs::remove_file(repository.path().join(".deepkit/config.local.toml"))?;
    let mut shifted = b"\n\n".to_vec();
    shifted.extend_from_slice(&api_original);
    fs::write(&api_path, shifted)?;
    let repaired_map = dkc(
        repository.path(),
        ["map", "--focus", "src/api.rs", "--json"],
    )?;
    assert_eq!(repaired_map["index"]["lazy_resync"]["published_files"], 1);
    assert_eq!(repaired_map["index"]["partial"], false);
    let mapped_api = repaired_map["nodes"][0]["symbols"]
        .as_array()
        .ok_or("missing map symbols")?
        .iter()
        .find(|symbol| symbol["name"] == "duplicate")
        .ok_or("missing duplicate")?;
    assert_eq!(
        mapped_api["lines"][0].as_u64(),
        api_symbol["lines"][0].as_u64().map(|line| line + 2)
    );
    let repaired_symbols = dkc(repository.path(), ["symbol", "duplicate", "--json"])?;
    let repaired_api = repaired_symbols
        .as_array()
        .ok_or("missing symbol array")?
        .iter()
        .find(|symbol| symbol["path"] == "src/api.rs")
        .ok_or("missing api")?;
    assert_eq!(repaired_api["id"], api_symbol["id"]);
    assert_eq!(repaired_api["doc"], api_symbol["doc"]);
    assert_eq!(
        repaired_api["lines"][0].as_u64(),
        api_symbol["lines"][0].as_u64().map(|line| line + 2)
    );
    let repaired = dkc(repository.path(), ["refs", "duplicate", "--json"])?;
    assert_eq!(repaired["index"]["partial"], false);
    assert_eq!(repaired["index"]["lazy_resync"]["published_files"], 0);
    assert_eq!(repaired["groups"][1]["lines"], serde_json::json!([9]));
    fs::write(&api_path, &api_original)?;
    assert_eq!(
        dkc(repository.path(), ["refs", "duplicate", "--json"])?["groups"],
        refs["groups"]
    );

    let restored_map = dkc(repository.path(), ["map", "--json"])?;
    assert_eq!(restored_map["index"]["partial"], false);
    assert!(
        restored_map["tree"]
            .as_str()
            .ok_or("missing tree")?
            .contains("api.rs")
    );

    let tests_only = dkc(
        repository.path(),
        ["refs", "duplicate", "--tests-only", "--json"],
    )?;
    assert_eq!(tests_only["groups"], serde_json::json!([]));
    assert_eq!(tests_only["total"], 0);

    let _ = dkc(repository.path(), ["clean", "--json"])?;
    let _ = dkc(repository.path(), ["index", "--level", "1", "--json"])?;
    let second_symbols = dkc(repository.path(), ["symbol", "duplicate", "--json"])?;
    let second_api_id = second_symbols
        .as_array()
        .and_then(|symbols| symbols.iter().find(|symbol| symbol["path"] == "src/api.rs"))
        .and_then(|symbol| symbol["id"].as_str())
        .ok_or("missing api duplicate after a fresh index")?;
    assert_eq!(second_api_id, symbol_id);
    fs::write(
        repository.path().join(".deepkit/config.local.toml"),
        "[limits]\nquery_timeout_ms=0\n",
    )?;
    let expired = AssertCommand::cargo_bin("dkc")?
        .current_dir(repository.path())
        .args(["symbol", "duplicate", "--json"])
        .output()?;
    assert!(!expired.status.success());
    let expired: Value = serde_json::from_slice(&expired.stdout)?;
    assert_eq!(expired["error"]["code"], "timeout");
    let expired_map = dkc(repository.path(), ["map", "--json"])?;
    assert_eq!(expired_map["tree"], "");
    assert_eq!(expired_map["index"]["reason"], "query_timeout");
    Ok(())
}

fn copy_tree(source: &Path, destination: &Path) -> Result<(), Box<dyn Error>> {
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let target = destination.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            fs::create_dir_all(&target)?;
            copy_tree(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

fn dkc<const N: usize>(repository: &Path, args: [&str; N]) -> Result<Value, Box<dyn Error>> {
    let output = AssertCommand::cargo_bin("dkc")?
        .args(args)
        .current_dir(repository)
        .output()?;
    assert!(
        output.status.success(),
        "dkc failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}
