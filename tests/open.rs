//! Source-verified symbol and context navigation; v1 §10.1 and T1.9.
use assert_cmd::cargo::cargo_bin;
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};

fn command(root: &Path, args: &[&str]) -> std::process::Output {
    Command::new(cargo_bin!("dkc"))
        .current_dir(root)
        .env("XDG_CONFIG_HOME", root.join("user-config"))
        .args(args)
        .output()
        .unwrap()
}

fn dkc(root: &Path, args: &[&str]) -> Value {
    let output = command(root, args);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn open_expands_split_symbols_and_context_and_refuses_unverified_sources() {
    let root = tempfile::tempdir().unwrap();
    assert!(
        Command::new("git")
            .current_dir(root.path())
            .args(["init", "--quiet"])
            .status()
            .unwrap()
            .success()
    );
    fs::create_dir(root.path().join(".deepkit")).unwrap();
    fs::write(
        root.path().join(".deepkit/config.toml"),
        "[search]\nlazy_resync=false\n[chunk]\nmin_tokens=1\ntarget_tokens=30\nmax_tokens=60\n",
    )
    .unwrap();
    let body = format!(
        "// before\npub fn expanded() {{\n{}    // end of expanded\n}}\n// after\n",
        "    let value = \"é🙂\";\r\n".repeat(250)
    );
    fs::write(root.path().join("source.rs"), &body).unwrap();
    dkc(root.path(), &["index", "--level", "1", "--json"]);
    let db = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
    let count: i64 = db
        .query_row("SELECT count(*) FROM units", [], |row| row.get(0))
        .unwrap();
    assert!(count > 2, "must actually split the symbol: {count}");
    let result = dkc(root.path(), &["open", "source.rs:125-125", "--json"]);
    assert_eq!(result["lines"], json!([2, 254]));
    assert_eq!(
        result["breadcrumb"],
        format!(
            "{} › source.rs",
            root.path().file_name().unwrap().to_str().unwrap()
        )
    );
    assert_eq!(
        result["text"],
        body.strip_prefix("// before\n")
            .unwrap()
            .strip_suffix("// after\n")
            .unwrap()
    );
    assert_eq!(result["truncated"], false);
    let symbols = dkc(root.path(), &["symbol", "expanded", "--json"]);
    let symbol = symbols[0]["id"].as_str().unwrap();
    assert_eq!(
        dkc(root.path(), &["open", symbol, "--json"])["text"],
        result["text"]
    );
    let surrounded = dkc(
        root.path(),
        &["open", "source.rs:125-125", "--context", "1", "--json"],
    );
    assert_eq!(surrounded["text"], body);
    assert_eq!(surrounded["lines"], json!([1, 255]));
    assert_eq!(
        dkc(
            root.path(),
            &[
                "open",
                symbol,
                "--context",
                "18446744073709551615",
                "--json"
            ]
        )["text"],
        body
    );
    // With lazy repair disabled, unpublished bytes cannot validate old ranges.
    db.execute("UPDATE files SET l1_pending=1", []).unwrap();
    assert!(
        !command(root.path(), &["open", symbol, "--json"])
            .status
            .success()
    );
    db.execute("UPDATE files SET l1_pending=0", []).unwrap();
    fs::write(root.path().join("source.rs"), "changed source\n").unwrap();
    assert!(
        !command(root.path(), &["open", symbol, "--json"])
            .status
            .success()
    );
    fs::remove_file(root.path().join("source.rs")).unwrap();
    assert!(
        !command(root.path(), &["open", symbol, "--json"])
            .status
            .success()
    );
    fs::write(root.path().join("source.rs"), &body).unwrap();
    fs::write(
        root.path().join(".deepkit/config.local.toml"),
        "[limits]\nmax_open_bytes='35 B'\n",
    )
    .unwrap();
    let capped = dkc(root.path(), &["open", symbol, "--json"]);
    assert!(capped["text"].as_str().unwrap().len() <= 35);
    assert_eq!(capped["truncated"], true);
    assert_eq!(capped["lines"], json!([2, 3]));
    let token = capped["continuation_id"].as_str().unwrap();
    assert_eq!(
        capped["byte_range"],
        json!([0, capped["text"].as_str().unwrap().len()])
    );
    fs::write(
        root.path().join(".deepkit/config.local.toml"),
        "[limits]\nmax_open_bytes='513 B'\n",
    )
    .unwrap();
    let mut reconstructed = capped["text"].as_str().unwrap().to_owned();
    let mut next = Some(token.to_owned());
    while let Some(id) = next {
        let page = dkc(root.path(), &["open", &id, "--json"]);
        assert_eq!(page["breadcrumb"], result["breadcrumb"]);
        assert_eq!(page["byte_range"][0], reconstructed.len());
        let text = page["text"].as_str().unwrap();
        assert!(!text.is_empty() && text.len() <= 513);
        let first = 2 + reconstructed.bytes().filter(|byte| *byte == b'\n').count();
        assert_eq!(
            page["lines"],
            json!([first, first + text.split_inclusive('\n').count() - 1])
        );
        reconstructed.push_str(text);
        assert_eq!(page["byte_range"][1], reconstructed.len());
        next = page["continuation_id"].as_str().map(str::to_owned);
        assert_eq!(page["truncated"], next.is_some());
    }
    assert_eq!(reconstructed, result["text"].as_str().unwrap());
    // Reject offsets outside the expansion and inside a UTF-8 character.
    let decoded = URL_SAFE_NO_PAD
        .decode(token.strip_prefix("o1_").unwrap())
        .unwrap();
    let cursor: Value = serde_json::from_slice(&decoded).unwrap();
    let emoji = reconstructed.find('🙂').unwrap();
    for offset in [0, emoji + 1, reconstructed.len(), usize::MAX] {
        let mut bad = cursor.clone();
        bad["offset"] = json!(offset);
        let id = format!(
            "o1_{}",
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&bad).unwrap())
        );
        assert!(
            !command(root.path(), &["open", &id, "--json"])
                .status
                .success()
        );
    }
    // A cap smaller than the next character must fail, never return a looping cursor.
    let mut at_emoji = cursor.clone();
    at_emoji["offset"] = json!(emoji);
    let id = format!(
        "o1_{}",
        URL_SAFE_NO_PAD.encode(serde_json::to_vec(&at_emoji).unwrap())
    );
    fs::write(
        root.path().join(".deepkit/config.local.toml"),
        "[limits]\nmax_open_bytes='1 B'\n",
    )
    .unwrap();
    assert!(
        !command(root.path(), &["open", &id, "--json"])
            .status
            .success()
    );
    fs::write(
        root.path().join(".deepkit/config.local.toml"),
        "[limits]\nmax_open_bytes='4 B'\n",
    )
    .unwrap();
    assert_eq!(dkc(root.path(), &["open", &id, "--json"])["text"], "🙂");
    fs::write(
        root.path().join(".deepkit/config.local.toml"),
        "[limits]\nmax_open_bytes='513 B'\n",
    )
    .unwrap();
    db.execute("UPDATE files SET l1_pending=1", []).unwrap();
    assert!(
        !command(root.path(), &["open", token, "--json"])
            .status
            .success()
    );
    db.execute("UPDATE files SET l1_pending=0", []).unwrap();
    // A changed completed fingerprint invalidates the token even with identical bytes.
    db.execute(
        "UPDATE files SET l1_fingerprint=x'010203', l1_desired=x'010203'",
        [],
    )
    .unwrap();
    assert!(
        !command(root.path(), &["open", symbol, "--json"])
            .status
            .success()
    );
    assert!(
        !command(root.path(), &["open", token, "--json"])
            .status
            .success()
    );
    dkc(root.path(), &["index", "--level", "1", "--json"]);
    let fresh = dkc(
        root.path(),
        &["open", "source.rs:125-125", "--context", "1", "--json"],
    );
    let fresh_token = fresh["continuation_id"].as_str().unwrap();
    assert!(
        command(root.path(), &["open", fresh_token, "--json"])
            .status
            .success()
    );
    assert!(
        !command(
            root.path(),
            &["open", fresh_token, "--context", "1", "--json"]
        )
        .status
        .success()
    );
    fs::write(
        root.path().join("source.rs"),
        body.replace("let value", "let other"),
    )
    .unwrap();
    assert!(
        !command(root.path(), &["open", fresh_token, "--json"])
            .status
            .success()
    );
    fs::remove_file(root.path().join("source.rs")).unwrap();
    assert!(
        !command(root.path(), &["open", fresh_token, "--json"])
            .status
            .success()
    );
    for id in ["o1_!", "o1_e30", "o1_"] {
        assert!(
            !command(root.path(), &["open", id, "--json"])
                .status
                .success()
        );
    }
}

#[test]
fn extracted_pages_preserve_payload_and_refuse_stale_source_generations() {
    let root = tempfile::tempdir().unwrap();
    assert!(
        Command::new("git")
            .current_dir(root.path())
            .args(["init", "--quiet"])
            .status()
            .unwrap()
            .success()
    );
    fs::create_dir(root.path().join(".deepkit")).unwrap();
    fs::write(
        root.path().join(".deepkit/config.toml"),
        "[search]\nlazy_resync=false\n[index]\nrepo_name='named'\n[embedder]\nbreadcrumb=false\n[chunk]\nmin_tokens=1\nmarkdown_range=[100,800]\n",
    )
    .unwrap();
    let markdown = format!("# Guide\n\n{}\n", "Read café 🙂 carefully.\r\n".repeat(20));
    // JSON extraction normalizes layout; payload line numbers cannot be source lines.
    let config = format!(
        "{{\"message\":\"{}\"}}\n",
        "Read café 🙂 carefully. ".repeat(20)
    );
    fs::write(root.path().join("guide.md"), &markdown).unwrap();
    fs::write(root.path().join("settings.json"), &config).unwrap();
    dkc(root.path(), &["index", "--level", "1", "--json"]);
    let db = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
    for (path, source) in [("guide.md", markdown), ("settings.json", config)] {
        let full = dkc(root.path(), &["open", path, "--json"]);
        let id = full["id"].as_str().unwrap();
        let (payload, start, end): (String, i64, i64) = db.query_row(
            "SELECT c.text,u.line_start,u.line_end FROM units u JOIN unit_contents c ON c.hash=u.content_hash WHERE substr(lower(hex(u.hash)),1,12)=?1",
            [id.strip_prefix("u_").unwrap()], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?))).unwrap();
        assert_eq!(full["text"], payload);
        assert_eq!(
            full["breadcrumb"],
            if path == "guide.md" {
                "named › guide.md › Guide"
            } else {
                "named › settings.json"
            }
        );
        assert_eq!(full["lines"], json!([null, null]));
        assert_eq!(full["indexed_lines"], json!([start, end]));
        assert_eq!(
            full["extracted_lines"],
            json!([1, payload.split_inclusive('\n').count()])
        );
        fs::write(
            root.path().join(".deepkit/config.local.toml"),
            "[limits]\nmax_open_bytes='37 B'\n",
        )
        .unwrap();
        let mut page = dkc(root.path(), &["open", id, "--json"]);
        let token = page["continuation_id"].as_str().unwrap().to_owned();
        let mut reconstructed = String::new();
        loop {
            let text = page["text"].as_str().unwrap();
            assert!(!text.is_empty() && text.len() <= 37);
            assert_eq!(page["lines"], json!([null, null]));
            assert_eq!(page["breadcrumb"], full["breadcrumb"]);
            assert_eq!(page["indexed_lines"], full["indexed_lines"]);
            let skipped = reconstructed.bytes().filter(|b| *b == b'\n').count();
            assert_eq!(
                page["extracted_lines"],
                json!([1 + skipped, skipped + text.split_inclusive('\n').count()])
            );
            assert_eq!(page["byte_range"][0], reconstructed.len());
            reconstructed.push_str(text);
            assert_eq!(page["byte_range"][1], reconstructed.len());
            assert_eq!(page["truncated"], page["continuation_id"].is_string());
            let Some(next) = page["continuation_id"].as_str() else {
                break;
            };
            page = dkc(root.path(), &["open", next, "--json"]);
        }
        assert_eq!(reconstructed, payload);
        fs::write(
            root.path().join(".deepkit/config.local.toml"),
            "[index]\nrepo_name='renamed'\n[limits]\nmax_open_bytes='37 B'\n",
        )
        .unwrap();
        assert!(
            !command(root.path(), &["open", &token, "--json"])
                .status
                .success()
        );
        assert!(
            !command(root.path(), &["open", id, "--json"])
                .status
                .success()
        );
        dkc(root.path(), &["index", "--level", "1", "--json"]);
        assert!(
            dkc(root.path(), &["open", id, "--json"])["breadcrumb"]
                .as_str()
                .unwrap()
                .starts_with("renamed › ")
        );
        fs::write(
            root.path().join(".deepkit/config.local.toml"),
            "[limits]\nmax_open_bytes='37 B'\n",
        )
        .unwrap();
        assert!(
            !command(root.path(), &["open", id, "--context", "1", "--json"])
                .status
                .success()
        );
        db.execute("UPDATE files SET l1_pending=1 WHERE path=?1", [path])
            .unwrap();
        for target in [id, &token] {
            assert!(
                !command(root.path(), &["open", target, "--json"])
                    .status
                    .success()
            );
        }
        db.execute("UPDATE files SET l1_pending=0, l1_fingerprint=x'010203', l1_desired=x'010203' WHERE path=?1", [path]).unwrap();
        assert!(
            !command(root.path(), &["open", id, "--json"])
                .status
                .success()
        );
        assert!(
            !command(root.path(), &["open", &token, "--json"])
                .status
                .success()
        );
        dkc(root.path(), &["index", "--level", "1", "--json"]);
        let new = dkc(root.path(), &["open", id, "--json"]);
        let new_token = new["continuation_id"].as_str().unwrap();
        fs::write(root.path().join(path), source.replace("Read", "Edit")).unwrap();
        for target in [id, new_token] {
            assert!(
                !command(root.path(), &["open", target, "--json"])
                    .status
                    .success()
            );
        }
        fs::remove_file(root.path().join(path)).unwrap();
        assert!(
            !command(root.path(), &["open", new_token, "--json"])
                .status
                .success()
        );
        fs::write(root.path().join(path), source).unwrap();
        fs::remove_file(root.path().join(".deepkit/config.local.toml")).unwrap();
    }
}
