use std::{
    env,
    error::Error,
    path::{Path, PathBuf},
    process::Command,
};

use assert_cmd::Command as AssertCommand;
use serde_json::Value;
use tempfile::TempDir;

#[test]
fn ripgrep_lexical_queries_find_its_readme_when_the_corpus_is_available()
-> Result<(), Box<dyn Error>> {
    let corpus = env::var_os("DKC_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/data/dkc-corpus"));
    let source = corpus.join("ripgrep");
    if !source.join(".git").exists() {
        return Ok(());
    }

    let temporary = TempDir::new()?;
    let checkout = temporary.path().join("ripgrep");
    let cloned = Command::new("git")
        .args(["clone", "--shared", "--quiet"])
        .arg(&source)
        .arg(&checkout)
        .status()?;
    assert!(cloned.success(), "shared ripgrep checkout failed");

    let initialized = dkc(&checkout, ["init", "--json"])?;
    assert_eq!(initialized["profile"], "local");
    let indexed = dkc(&checkout, ["index", "--level", "1", "--full", "--json"])?;
    assert!(
        indexed["units_indexed"]
            .as_u64()
            .is_some_and(|count| count > 0)
    );

    for query in [
        "ripgrep",
        "fast",
        "command line",
        "oriented",
        "search",
        "tool",
        "regex",
        "recursive",
        "install",
        "windows",
    ] {
        let result = dkc(&checkout, ["search", query, "-k", "50", "--json"])?;
        assert_eq!(result["result_budget"]["truncated"], false, "{result}");
        let hits = result["hits"]
            .as_array()
            .ok_or("search did not return hits")?;
        assert!(
            hits.iter().any(|hit| hit["path"] == "README.md"),
            "{query} did not retrieve README.md: {result}"
        );
    }
    Ok(())
}

fn dkc<const N: usize>(repository: &Path, args: [&str; N]) -> Result<Value, Box<dyn Error>> {
    let mut command = AssertCommand::cargo_bin("dkc")?;
    // The README corpus smoke test measures source retrieval. Default history
    // ingestion and immutable-history search have separate public acceptance cases.
    command
        .args(args)
        .current_dir(repository)
        .env("DKC_COMMITS__ENABLED", "false")
        // README retrieval at k=50 needs its declared result population. Tiny
        // and default response caps have independent CLI/MCP acceptance tests.
        .env("DKC_LIMITS__MAX_RESULT_TOKENS", "64000");
    if args.first() == Some(&"search") {
        // This fixture measures lexical/L1 behavior, independently of model setup.
        command.args(["--no-rerank", "--mode", "lexical", "--scope", "source"]);
    }
    let output = command.output()?;
    assert!(
        output.status.success(),
        "dkc failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}
