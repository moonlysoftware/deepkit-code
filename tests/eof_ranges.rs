//! Public fallback chunks use inclusive source lines for exclusive byte ranges.
use assert_cmd::Command;
use serde_json::{Value, json};
use std::{fs, path::Path, time::Duration};

fn dkc(root: &Path, args: &[&str]) -> Value {
    let mut command = Command::new(assert_cmd::cargo::cargo_bin!("dkc"));
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().starts_with("DKC_") {
            command.env_remove(name);
        }
    }
    let output = command
        .current_dir(root)
        .env("XDG_CONFIG_HOME", root.join("absent-user-config"))
        .env("DKC_MODELS_DIR", root.join("absent-model-cache"))
        .env_remove("MOONLY_GATEWAY_API_KEY")
        .timeout(Duration::from_secs(30))
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn newline_terminated_blade_chunk_search_and_open_use_inclusive_eof_lines() {
    for newline in ["\n", "\r\n"] {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("src")).unwrap();
        fs::create_dir(root.path().join(".deepkit")).unwrap();
        fs::write(
            root.path().join(".deepkit/config.toml"),
            "[embedder]\nprofile='local'\n[llm]\nenabled=false\n[rerank]\nmode='off'\n",
        )
        .unwrap();
        // The PHP grammar exposes inline template text with an exclusive end
        // after the newline; exercise normal classification and L1 publication.
        let source = ["<x-input>", "    eofrangemarker", "</x-input>"].join(newline) + newline;
        fs::write(root.path().join("src/input.blade.php"), &source).unwrap();
        let indexed = dkc(root.path(), &["index", "--level", "1", "--json"]);
        assert_eq!(indexed["units_indexed"], 1);
        for _ in 0..2 {
            let response = dkc(
                root.path(),
                &[
                    "search",
                    "eofrangemarker",
                    "--mode",
                    "lexical",
                    "--scope",
                    "source",
                    "--no-rerank",
                    "--json",
                ],
            );
            assert_eq!(response["index"]["degraded"], false, "{response}");
            assert_eq!(response["index"]["partial"], false, "{response}");
            assert_eq!(response["index"]["lexical"], "complete");
            assert_eq!(response["index"]["dense"], "not_requested");
            assert_eq!(response["index"]["freshness_refusals"], json!({}));
            assert_eq!(response["index"]["lazy_resync"]["attempted_files"], 0);
            let hits = response["hits"].as_array().unwrap();
            assert_eq!(hits.len(), 1, "{response}");
            let hit = &hits[0];
            assert_eq!(hit["path"], "src/input.blade.php");
            assert_eq!(hit["kind"], "chunk");
            assert_eq!(hit["lines"], json!([1, 3]));
            let opened = dkc(
                root.path(),
                &[
                    "open",
                    hit["id"].as_str().unwrap(),
                    "--context",
                    "0",
                    "--json",
                ],
            );
            assert_eq!(opened["lines"], json!([1, 3]));
            assert_eq!(opened["text"], source);
            assert_eq!(opened["truncated"], false);
        }
        assert_eq!(
            fs::read_to_string(root.path().join("src/input.blade.php")).unwrap(),
            source
        );
        assert!(!root.path().join("absent-model-cache").exists());
        assert!(!root.path().join("absent-user-config").exists());
    }
}

#[test]
fn repeated_map_does_not_repair_valid_file_module_ranges_or_invalidate_graph_readiness() {
    for newline in ["\n", "\r\n"] {
        for terminated in [false, true] {
            let root = tempfile::tempdir().unwrap();
            fs::create_dir(root.path().join("src")).unwrap();
            fs::create_dir(root.path().join(".deepkit")).unwrap();
            fs::write(
                root.path().join(".deepkit/config.toml"),
                "[embedder]\nprofile='local'\n[llm]\nenabled=false\n[rerank]\nmode='off'\n",
            )
            .unwrap();
            let mut source = [
                "<?php",
                "use Example\\Dependency;",
                "function eofgraphmarker() { return 1; }",
                "// λ",
            ]
            .join(newline);
            if terminated {
                source.push_str(newline);
            }
            fs::write(root.path().join("src/fixture.php"), &source).unwrap();
            dkc(root.path(), &["index", "--level", "1", "--json"]);
            let db = rusqlite::Connection::open_with_flags(
                root.path().join(".deepkit/code.db"),
                rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
            )
            .unwrap();
            let state = || {
                let meta: Vec<(String, String)> = db.prepare("SELECT key,value FROM meta WHERE key IN ('graph_pending','l1_scan_generation','l1_completed_generation') ORDER BY key").unwrap()
                    .query_map([], |row| Ok((row.get(0)?, row.get(1)?))).unwrap().collect::<Result<_, _>>().unwrap();
                let units: Vec<(i64, Vec<u8>, Vec<u8>)> = db
                    .prepare("SELECT id,hash,content_hash FROM units ORDER BY id")
                    .unwrap()
                    .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
                    .unwrap()
                    .collect::<Result<_, _>>()
                    .unwrap();
                (meta, units)
            };
            let before = state();
            assert!(
                before
                    .0
                    .iter()
                    .any(|(key, value)| key == "graph_pending" && value == "0")
            );
            assert!(!before.1.is_empty());
            for _ in 0..2 {
                let map = dkc(
                    root.path(),
                    &[
                        "map",
                        "--budget",
                        "2000",
                        "--focus",
                        "src/fixture.php",
                        "--json",
                    ],
                );
                assert_eq!(map["index"]["partial"], false, "{map}");
                assert_eq!(map["index"]["lazy_resync"]["attempted_files"], 0, "{map}");
                assert_eq!(map["index"]["lazy_resync"]["published_files"], 0, "{map}");
                assert!(
                    map["nodes"]
                        .as_array()
                        .is_some_and(|nodes| !nodes.is_empty()),
                    "{map}"
                );
                assert_eq!(
                    state(),
                    before,
                    "map must preserve L1 generations, graph readiness and unit identities"
                );
            }
            assert_eq!(
                fs::read_to_string(root.path().join("src/fixture.php")).unwrap(),
                source
            );
            assert!(!root.path().join("absent-model-cache").exists());
        }
    }
}
