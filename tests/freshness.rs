//! Public checkout range and lazy-repair acceptance; v1.md §4.3–4.6 and §9.
use assert_cmd::Command;
use proptest::prelude::*;
use serde_json::Value;
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
fn git(root: &Path, args: &[&str]) -> String {
    let output = std::process::Command::new("git")
        .current_dir(root)
        .args([
            "-c",
            "commit.gpgsign=false",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
        ])
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
fn config(root: &Path) {
    fs::create_dir_all(root.join(".deepkit")).unwrap();
    fs::write(
        root.join(".deepkit/config.toml"),
        "[embedder]\nprofile='local'\n[llm]\nenabled=false\n[rerank]\nmode='off'\n",
    )
    .unwrap();
}
fn search(root: &Path) -> Value {
    dkc(
        root,
        &[
            "search",
            "checkoutmarker",
            "--mode",
            "lexical",
            "--scope",
            "source",
            "--no-rerank",
            "--json",
        ],
    )
}
fn valid_ranges(root: &Path, response: &Value) {
    let hits = response["hits"].as_array().unwrap();
    assert!(!hits.is_empty(), "{response}");
    for hit in hits {
        let text = fs::read_to_string(root.join(hit["path"].as_str().unwrap())).unwrap();
        let start = hit["lines"][0].as_u64().unwrap() as usize;
        let end = hit["lines"][1].as_u64().unwrap() as usize;
        assert!(
            start >= 1 && start <= end && end <= text.lines().count(),
            "{hit}: {text}"
        );
        assert_eq!(hit["fresh"]["state"], "fresh", "{hit}");
        assert!(
            text.lines()
                .skip(start - 1)
                .take(end - start + 1)
                .any(|line| line.contains("checkoutmarker")),
            "{hit}"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 8,
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Direct("tests/freshness.proptest-regressions"))),
        ..ProptestConfig::default()
    })]
    #[test]
    fn checkout_search_ranges_exist_at_every_generated_commit(
        leading in prop::collection::vec(0usize..12, 4..7),
        order in prop::collection::vec(any::<usize>(), 3..7),
    ) {
        let root = tempfile::tempdir().unwrap();
        git(root.path(), &["init", "-q", "--initial-branch=main"]);
        let mut commits = Vec::new();
        for (i, blanks) in leading.iter().enumerate() {
            let source = format!("{}pub fn checkoutmarker() -> usize {{\n    {i}\n}}\n", "\n".repeat(*blanks));
            fs::write(root.path().join("source.rs"), source).unwrap();
            let other = root.path().join("other.rs");
            if i % 2 == 0 {
                fs::write(&other, format!("pub fn checkoutmarker_other() -> usize {{ {i} }}\n")).unwrap();
            } else if other.exists() {
                fs::remove_file(other).unwrap();
            }
            git(root.path(), &["add", "-A"]);
            git(root.path(), &["commit", "-qm", &format!("generated {i}")]);
            commits.push(git(root.path(), &["rev-parse", "HEAD"]));
        }
        config(root.path());
        let indexed = dkc(root.path(), &["index", "--level", "1", "--json"]);
        prop_assert!(indexed["units_indexed"].as_u64().unwrap() > 0);
        // Visit every commit plus random transitions; empty responses cannot pass.
        for i in (0..commits.len()).chain(order.into_iter().map(|i| i % commits.len())) {
            git(root.path(), &["checkout", "--quiet", "--detach", &commits[i]]);
            let response = search(root.path());
            prop_assert_eq!(&response["index"]["partial"], &Value::Bool(false));
            prop_assert_eq!(response["index"]["indexed_head"].as_str(), Some(commits[i].as_str()));
            valid_ranges(root.path(), &response);
            for hit in response["hits"].as_array().unwrap() {
                prop_assert!(hit["snippet"].as_str().unwrap().contains(&i.to_string()));
            }
        }
    }
}

#[test]
fn next_search_repairs_dirty_text_and_reports_one_mib_refusal() {
    let root = tempfile::tempdir().unwrap();
    config(root.path());
    let source = root.path().join("source.rs");
    fs::write(
        &source,
        "pub fn checkoutmarker() { let old_payload = 1; }\n",
    )
    .unwrap();
    dkc(root.path(), &["index", "--level", "1", "--json"]);
    fs::write(
        &source,
        "\n\npub fn checkoutmarker() { let new_payload = 2; }\n",
    )
    .unwrap();
    let repaired = search(root.path());
    valid_ranges(root.path(), &repaired);
    assert_eq!(
        repaired["index"]["lazy_resync"]["published_files"], 1,
        "{repaired}"
    );
    for hit in repaired["hits"].as_array().unwrap() {
        let snippet = hit["snippet"].as_str().unwrap();
        assert!(
            snippet.contains("new_payload") && !snippet.contains("old_payload"),
            "{hit}"
        );
    }
    let connection = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
    let before: Vec<u8> = connection
        .query_row("SELECT blake3 FROM files WHERE path='source.rs'", [], |r| {
            r.get(0)
        })
        .unwrap();
    let generated = std::process::Command::new("python3")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/scripts/fixtures/make-large-rust.py"
        ))
        .arg((1024 * 1024).to_string())
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let large = generated.stdout;
    assert_eq!(large.len(), 1024 * 1024);
    fs::write(&source, &large).unwrap();
    let refused = search(root.path());
    assert_eq!(
        refused["index"]["lazy_resync"]["attempted_files"], 1,
        "{refused}"
    );
    assert_eq!(
        refused["index"]["lazy_resync"]["refused_files"], 1,
        "{refused}"
    );
    assert_eq!(
        refused["index"]["lazy_resync"]["published_files"], 0,
        "{refused}"
    );
    assert_eq!(refused["index"]["partial"], true, "{refused}");
    assert!(refused["hits"].as_array().unwrap().is_empty(), "{refused}");
    let after: Vec<u8> = connection
        .query_row("SELECT blake3 FROM files WHERE path='source.rs'", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(before, after);
    assert_eq!(fs::read(source).unwrap(), large);
}

#[test]
fn distant_fixture_has_one_thousand_reproducible_changed_files() {
    let roots = [tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap()];
    let mut revisions = Vec::new();
    for root in &roots {
        let output = std::process::Command::new("bash")
            .arg(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/scripts/fixtures/make-git-repo.sh"
            ))
            .arg("--distant")
            .arg(root.path())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            git(root.path(), &["diff", "--name-only", "main", "alternate"])
                .lines()
                .count(),
            1000
        );
        assert!(git(root.path(), &["status", "--porcelain"]).is_empty());
        revisions.push((
            git(root.path(), &["rev-parse", "main"]),
            git(root.path(), &["rev-parse", "alternate"]),
        ));
    }
    assert_eq!(revisions[0], revisions[1]);
}

#[test]
fn indexing_unterminated_comments_completes_and_keeps_other_source_searchable() {
    let root = tempfile::tempdir().unwrap();
    config(root.path());
    for (name, source) in [
        ("empty.rs", "/*"),
        ("short.rs", "/*a"),
        ("unicode.rs", "/*éx"),
    ] {
        fs::write(root.path().join(name), source).unwrap();
    }
    fs::write(
        root.path().join("source.rs"),
        "pub fn checkoutmarker() {}\n",
    )
    .unwrap();
    dkc(root.path(), &["index", "--level", "1", "--json"]);
    let status = dkc(root.path(), &["status", "--json"]);
    assert_eq!(status["files"], 4, "{status}");
    assert_eq!(status["l1_pending_files"], 0, "{status}");
    let response = search(root.path());
    assert_eq!(response["index"]["partial"], false, "{response}");
    valid_ranges(root.path(), &response);
}
