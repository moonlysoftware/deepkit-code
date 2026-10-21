use std::{error::Error, fs, process::Command};

use assert_cmd::Command as AssertCommand;
use serde_json::Value;
use tempfile::TempDir;

#[test]
fn help_and_version_succeed_without_opening_a_repository() -> Result<(), Box<dyn Error>> {
    let root = TempDir::new()?;
    for args in [
        &["--help"][..],
        &["-h"][..],
        &["--version"][..],
        &["-V"][..],
        &["search", "--help"][..],
    ] {
        let output = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
            .current_dir(root.path())
            .args(args)
            .output()?;
        assert!(output.status.success(), "{args:?}: {output:?}");
        assert!(output.stderr.is_empty(), "{args:?}: {output:?}");
        assert!(String::from_utf8_lossy(&output.stdout).contains("dkc"));
    }
    let invalid = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root.path())
        .arg("--not-a-real-option")
        .output()?;
    assert!(!invalid.status.success());
    assert!(invalid.stdout.is_empty());
    assert!(!invalid.stderr.is_empty());
    assert_eq!(fs::read_dir(root.path())?.count(), 0);
    Ok(())
}

#[test]
fn tiny_repository_runs_the_phase_zero_cli_flow() -> Result<(), Box<dyn Error>> {
    let repository = TempDir::new()?;
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/tiny");
    fs::copy(
        fixture.join("script.rs"),
        repository.path().join("script.rs"),
    )?;
    fs::copy(
        fixture.join("README.md"),
        repository.path().join("README.md"),
    )?;
    let status = Command::new("git")
        .args(["init", "--quiet"])
        .current_dir(repository.path())
        .status()?;
    assert!(status.success());
    let status = Command::new("git")
        .args(["add", "script.rs", "README.md"])
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
            "tiny baseline",
        ])
        .current_dir(repository.path())
        .status()?;
    assert!(status.success());

    let first = dkc(repository.path(), ["init", "--json"])?;
    assert!(first.get("config").is_some());
    let second = dkc(repository.path(), ["init", "--json"])?;
    assert!(second.get("exclude").is_some());
    assert!(
        fs::read_to_string(repository.path().join(".deepkit/config.toml"))?
            .contains("# max_files uses its documented default")
    );
    let exclude = fs::read_to_string(repository.path().join(".git/info/exclude"))?;
    assert_eq!(exclude.matches("# >>> deepkit-code >>>").count(), 1);
    assert!(exclude.contains(".deepkit/code.db*"));
    assert!(exclude.contains(".deepkit/config.local.toml"));

    let indexed = dkc(repository.path(), ["index", "--level", "1", "--json"])?;
    assert!(
        indexed["units_indexed"]
            .as_u64()
            .is_some_and(|count| count > 0)
    );
    let rebuilt = dkc(
        repository.path(),
        ["index", "--level", "1", "--rebuild-fts", "--json"],
    )?;
    assert!(
        rebuilt["units_indexed"]
            .as_u64()
            .is_some_and(|count| count > 0)
    );
    let search = dkc(repository.path(), ["search", "should_retry", "--json"])?;
    let hits = search["hits"]
        .as_array()
        .ok_or("search did not return hits")?;
    let first_hit = hits.first().ok_or("search returned no hits")?;
    assert_eq!(first_hit["path"], "script.rs");
    assert!(first_hit["fresh"]["changed_ago_s"].is_u64());
    assert_eq!(first_hit["fresh"]["l2"], "pending");
    let id = first_hit["id"].as_str().ok_or("hit did not have an id")?;

    let relaxed = dkc(
        repository.path(),
        [
            "search",
            "where does should_retry handle transient problems",
            "--json",
        ],
    )?;
    assert_eq!(relaxed["hits"][0]["path"], "script.rs");
    assert!(
        relaxed["hits"][0]["why"]
            .as_array()
            .is_some_and(|why| why.iter().any(|reason| reason == "exact_name"))
    );
    let scoped = dkc(
        repository.path(),
        ["search", "should_retry", "--scope", "*.rs", "--json"],
    )?;
    assert_eq!(scoped["hits"][0]["id"], first_hit["id"]);

    let opened = dkc(repository.path(), ["open", id, "--json"])?;
    assert!(
        opened["text"]
            .as_str()
            .is_some_and(|text| text.contains("should_retry"))
    );

    let map = dkc(repository.path(), ["map", "--json"])?;
    let golden = fs::read_to_string(fixture.join("map.golden"))?;
    assert_eq!(map["tree"].as_str(), Some(golden.trim_end()));
    let focused = dkc(repository.path(), ["map", "--focus", "script", "--json"])?;
    assert_eq!(focused["nodes"].as_array().map(Vec::len), Some(1));
    assert_eq!(focused["nodes"][0]["path"], "script.rs");
    let budgeted = dkc(repository.path(), ["map", "--budget", "20", "--json"])?;
    let budgeted_tree = budgeted["tree"]
        .as_str()
        .ok_or("map did not return a tree")?;
    let tokens = tiktoken_rs::cl100k_base_singleton()
        .encode_with_special_tokens(budgeted_tree)
        .len();
    assert!(
        tokens <= 21,
        "map used {tokens} tokens for a 20-token budget"
    );

    let status = dkc(repository.path(), ["status", "--json"])?;
    assert_eq!(
        status["query_latency_ms"],
        serde_json::json!({
            "count":0, "window":1000, "lifetime":"process", "p50":null, "p95":null,
        })
    );
    assert!(status["dirty_files"].is_null());
    assert_eq!(status["dirty_files_state"], "not_checked");
    for field in ["ladder", "profile", "models", "budgets"] {
        assert!(status[field].is_object(), "missing {field}");
    }
    assert!(status["ladder"]["l1"].as_i64().unwrap() > 0);

    assert_eq!(status["files"], 2);
    assert!(status["units"].as_i64().is_some_and(|count| count > 0));
    assert!(status["took_ms"].as_u64().is_some());

    let cleaned = dkc(repository.path(), ["clean", "--json"])?;
    assert!(
        cleaned["removed"]
            .as_array()
            .is_some_and(|removed| !removed.is_empty())
    );
    assert!(!repository.path().join(".deepkit/code.db").exists());
    let output = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(repository.path())
        .output()?;
    assert!(output.status.success());
    assert_eq!(String::from_utf8(output.stdout)?, "?? .deepkit/\n");

    let gitignore_repository = TempDir::new()?;
    let status = Command::new("git")
        .args(["init", "--quiet"])
        .current_dir(gitignore_repository.path())
        .status()?;
    assert!(status.success());
    let initialized = dkc(
        gitignore_repository.path(),
        ["init", "--gitignore", "--json"],
    )?;
    assert!(
        initialized["exclude"]
            .as_str()
            .is_some_and(|path| path.ends_with(".gitignore"))
    );
    let gitignore = fs::read_to_string(gitignore_repository.path().join(".gitignore"))?;
    assert_eq!(gitignore.matches("# >>> deepkit-code >>>").count(), 1);
    assert!(gitignore.contains(".deepkit/code.db*"));
    Ok(())
}

#[test]
fn cli_l2_freshness_tracks_completed_policy_changes() -> Result<(), Box<dyn Error>> {
    let root = TempDir::new()?;
    assert!(
        Command::new("git")
            .args(["init", "--quiet"])
            .current_dir(root.path())
            .status()?
            .success()
    );
    fs::create_dir(root.path().join("src"))?;
    fs::create_dir(root.path().join(".deepkit"))?;
    fs::write(root.path().join("src/retry.rs"), "pub fn policy_probe() {}")?;
    let config = root.path().join(".deepkit/config.toml");
    for (policy, state) in [(3, "pending"), (1, "none"), (3, "pending")] {
        fs::write(
            &config,
            format!("[[treatments]]\nname='policy'\nmatch=['src/**']\nmax_level={policy}\n"),
        )?;
        dkc(root.path(), ["index", "--level", "1", "--json"])?;
        let response = dkc(root.path(), ["search", "policy_probe", "--json"])?;
        assert_eq!(
            response["hits"].as_array().ok_or("missing hits")?.len(),
            1,
            "{response}"
        );
        assert_eq!(response["hits"][0]["fresh"]["l2"], state);
        assert_eq!(response["index"]["partial"], false);
    }
    Ok(())
}

fn dkc<const N: usize>(
    repository: &std::path::Path,
    args: [&str; N],
) -> Result<Value, Box<dyn Error>> {
    let mut command = AssertCommand::cargo_bin("dkc")?;
    command.args(args).current_dir(repository);
    if args.first() == Some(&"search") {
        // This fixture measures lexical/L1 behavior, independently of model setup.
        command.arg("--no-rerank");
    }
    let output = command.output()?;
    assert!(
        output.status.success(),
        "dkc failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}
