use std::{error::Error, fs, path::Path, process::Command};

use assert_cmd::Command as AssertCommand;
use rusqlite::Connection;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

#[test]
fn bench_runs_a_commit_derived_fixture_and_diff_flags_a_regression() -> Result<(), Box<dyn Error>> {
    let workspace = TempDir::new()?;
    let repository = workspace.path().join("tiny");
    fs::create_dir_all(&repository)?;
    fs::write(
        repository.join("lib.rs"),
        "pub fn retry_limit() -> u32 { 3 }\n",
    )?;
    fs::write(repository.join("completion.zsh"), "#compdef tiny\n(((\n")?;
    git(&repository, ["init", "--quiet"])?;
    git(&repository, ["add", "lib.rs", "completion.zsh"])?;
    git(
        &repository,
        [
            "-c",
            "user.name=Deepkit Test",
            "-c",
            "user.email=deepkit@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "initial retry implementation for the benchmark fixture",
        ],
    )?;
    fs::write(
        repository.join("lib.rs"),
        "pub fn retry_limit() -> u32 { 4 }\n",
    )?;
    git(&repository, ["add", "lib.rs"])?;
    git(
        &repository,
        [
            "-c",
            "user.name=Deepkit Test",
            "-c",
            "user.email=deepkit@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "keep retry limit behavior local to the source module for callers",
        ],
    )?;
    let revision = git_stdout(&repository, ["rev-parse", "HEAD"])?;

    let corpus = workspace.path().join("corpus.toml");
    fs::write(
        workspace.path().join("profile.toml"),
        "[[treatments]]\nname='zsh-lines'\nmatch=['completion.zsh']\nchunker='lines'\nmax_level=1\n",
    )?;
    fs::write(
        &corpus,
        format!(
            "[[repo]]\nname = \"tiny\"\nurl = \"local\"\npath = \"{}\"\nrev = \"{}\"\nconfig = \"profile.toml\"\n",
            repository.display(),
            revision.trim()
        ),
    )?;
    let report_path = workspace.path().join("run.json");
    let result = dkc(
        workspace.path(),
        [
            "bench",
            "run",
            "--corpus",
            corpus.to_str().ok_or("non-UTF-8 corpus path")?,
            "--only",
            "tiny",
            "--keep",
            "--out",
            report_path.to_str().ok_or("non-UTF-8 report path")?,
            "--json",
        ],
    )?;
    assert_eq!(result["targets"], 1);
    assert!(report_path.is_file());
    assert!(report_path.with_extension("md").is_file());
    let markdown = fs::read_to_string(report_path.with_extension("md"))?;
    assert!(markdown.contains("PRs reconstructed/detected"));
    assert!(markdown.contains("parse failures"));
    let report: Value = serde_json::from_slice(&fs::read(&report_path)?)?;
    assert_eq!(report["targets"][0]["name"], "tiny");
    assert_eq!(report["dkc_commit"], env!("DKC_BUILD_COMMIT"));
    assert_eq!(
        report["build"]["source_blake3"],
        env!("DKC_BUILD_SOURCE_BLAKE3")
    );
    assert_eq!(report["build"]["features"], env!("DKC_BUILD_FEATURES"));
    assert!(report["machine"]["preflight"]["gpu_inventory"].is_null());
    assert!(report["machine"]["preflight"]["gpu_compute_processes"].is_null());
    let binary = fs::read(assert_cmd::cargo::cargo_bin("dkc"))?;
    assert_eq!(report["build"]["binary_bytes"], binary.len());
    assert_eq!(
        report["build"]["binary_blake3"],
        blake3::hash(&binary).to_hex().to_string()
    );
    let provenance = &report["targets"][0]["provenance"];
    assert!(provenance["pre_index"]["gpu_inventory"].is_null());
    assert!(provenance["pre_index"]["gpu_compute_processes"].is_null());
    assert_eq!(provenance["head"], revision.trim());
    assert_eq!(
        provenance["effective_config_blake3"]
            .as_str()
            .ok_or("config hash")?
            .len(),
        64
    );
    let database_before = fs::read(repository.join(".deepkit/code.db"))?;
    let capture = dkc(
        workspace.path(),
        [
            "bench",
            "capture-config",
            "--corpus",
            corpus.to_str().ok_or("non-UTF-8 corpus path")?,
            "--target",
            "tiny",
            "--profile",
            "l1",
            "--json",
        ],
    )?;
    assert_eq!(capture["model_evidence_complete"], false);
    assert_eq!(capture["target_head"], revision.trim());
    let starting_config = fs::read(repository.join(".deepkit/config.toml"))?;
    assert_eq!(
        capture["effective_config"]["starting_config_sha256"],
        json!({"config.toml": Sha256::digest(&starting_config)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()})
    );
    assert_eq!(
        capture["effective_config"]["effective_config_blake3"],
        provenance["effective_config_blake3"]
    );
    assert_eq!(
        capture["effective_config"]["config_files_blake3"],
        provenance["config_files_blake3"]
    );
    for field in [
        "configured_profile",
        "configured_embedding",
        "configured_reranker",
        "docs_tier_configured",
        "verified_models",
    ] {
        assert_eq!(
            capture["model_identity_template"]["model_usage"][field],
            provenance["model_usage"][field],
            "model field {field}"
        );
    }
    let cpu_capture = dkc(
        workspace.path(),
        [
            "bench",
            "capture-config",
            "--corpus",
            corpus.to_str().ok_or("non-UTF-8 corpus path")?,
            "--target",
            "tiny",
            "--profile",
            "local_cpu",
            "--json",
        ],
    )?;
    assert_eq!(cpu_capture["effective_config"]["profile"], "local_cpu");
    assert_eq!(
        cpu_capture["model_identity_template"]["model_usage"]["verified_models"]["embedder"]["profile"],
        "local"
    );
    assert_eq!(
        cpu_capture["model_identity_template"]["identities"],
        json!([])
    );
    assert_eq!(
        fs::read(repository.join(".deepkit/code.db"))?,
        database_before
    );
    assert_eq!(
        provenance["indexed_sources_blake3"]
            .as_str()
            .ok_or("source hash")?
            .len(),
        64
    );
    assert_eq!(provenance["page_size"], 16384);
    assert_eq!(provenance["model_usage"]["state"], "not_invoked_l1_only");
    assert!(provenance["model_usage"]["verified_models"].is_null());
    assert!(provenance["pre_index"]["captured_at"].as_u64() >= report["created_at"].as_u64());

    assert_eq!(
        report["targets"][0]["quality"]["commit_derived"]["count"],
        1
    );
    assert!(report["targets"][0]["index"]["cold_ms"].as_f64().is_some());
    if cfg!(target_os = "linux") {
        assert!(
            report["targets"][0]["index"]["peak_rss_bytes"]
                .as_u64()
                .is_some()
        );
    }
    assert!(report["targets"][0]["size"]["ratio"].as_f64().is_some());
    let queries = &report["targets"][0]["queries"];
    assert_eq!(queries["search"]["samples"], 3);
    assert!(queries["search"]["cold_ms"].as_f64().is_some());
    assert_eq!(queries["search"]["warmups_per_set"], 5);
    assert_eq!(queries["map"]["samples"], 15);
    assert!(queries["map"]["cold_ms"].as_f64().is_some());
    for tool in ["open", "symbol", "refs", "recent"] {
        assert_eq!(queries[tool]["latency"]["samples"], 60, "{tool}");
        assert!(
            queries[tool]["latency"]["cold_ms"].as_f64().is_some(),
            "{tool}"
        );
        assert!(queries[tool]["target"].as_str().is_some(), "{tool}");
    }
    assert_eq!(queries["unavailable"], json!({}));
    assert_eq!(queries["cli_search"]["samples"], 20);
    assert_eq!(queries["cli_shared_results_match"], true);
    assert!(queries["cli_median_overhead_ms"].as_f64().is_some());
    assert_eq!(queries["mcp_search"]["transport"], "stdio");
    assert_eq!(queries["mcp_search"]["search"]["samples"], 20);
    assert!(
        queries["mcp_search"]["search"]["cold_ms"]
            .as_f64()
            .is_some()
    );
    assert!(queries["mcp_search"]["startup_ms"].as_f64().is_some());
    assert!(queries["mcp_search"]["cold_start_ms"].as_f64().is_some());
    assert!(
        queries["mcp_search"]["median_overhead_ms"]
            .as_f64()
            .is_some()
    );
    let freshness = &report["targets"][0]["freshness"];
    assert_eq!(freshness["dirty_edit"]["matched"], true);
    assert_eq!(freshness["dirty_edit"]["stale_or_resynced"], true);
    assert_eq!(freshness["distant_checkout"]["invalid_source_ranges"], 0);
    assert!(
        freshness["distant_checkout"]["hits"]
            .as_u64()
            .is_some_and(|hits| hits > 0)
    );
    assert_eq!(
        git_stdout(&repository, ["rev-parse", "HEAD"])?.trim(),
        revision.trim()
    );
    assert!(
        git_stdout(
            &repository,
            ["status", "--porcelain", "--untracked-files=no"]
        )?
        .trim()
        .is_empty()
    );
    let memory = &queries["resident_memory"];
    assert_eq!(memory["page_size"], 16384);
    assert_eq!(memory["cache_size"], -65536);
    if cfg!(target_os = "linux") {
        for field in [
            "before_open_bytes",
            "after_open_bytes",
            "after_search_bytes",
            "after_map_bytes",
        ] {
            assert!(memory[field].as_u64().is_some_and(|n| n > 0));
        }
    }
    let database_path = repository.join(".deepkit/code.db");
    let database = Connection::open(&database_path)?;
    database.execute(
        "INSERT INTO jobs(kind,target_kind,target_id,target_hash,desired_hash,provider,priority,tokens,state,created_at,updated_at)
         SELECT 'embed','unit',id,hash,zeroblob(32),'embed:local',1,1,'queued',0,0
         FROM units ORDER BY id LIMIT 1",
        [],
    )?;
    let pending_job = database.last_insert_rowid();
    let before: (i64, i64, i64) = database.query_row(
        "SELECT (SELECT COUNT(*) FROM jobs),
                (SELECT COUNT(*) FROM embedding_publications),
                (SELECT COUNT(*) FROM usage_reservations)",
        [],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    drop(database);
    let probe_path = workspace.path().join("probe.json");
    let probe = dkc(
        workspace.path(),
        [
            "bench",
            "probe",
            "--root",
            repository.to_str().ok_or("root path")?,
            "--out",
            probe_path.to_str().ok_or("probe path")?,
            "--json",
        ],
    )?;
    assert_eq!(probe["queries"]["resident_memory"]["page_size"], 16384);
    assert_eq!(probe["queries"]["cli_shared_results_match"], true);
    assert_eq!(
        probe["indexed_sources_blake3"],
        provenance["indexed_sources_blake3"]
    );
    assert_eq!(probe["head"], revision.trim());
    let database = Connection::open(&database_path)?;
    let after: (i64, i64, i64) = database.query_row(
        "SELECT (SELECT COUNT(*) FROM jobs),
                (SELECT COUNT(*) FROM embedding_publications),
                (SELECT COUNT(*) FROM usage_reservations)",
        [],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    assert_eq!(
        before, after,
        "benchmark MCP requests must not start model work"
    );
    assert_eq!(
        database.query_row("SELECT state FROM jobs WHERE id=?1", [pending_job], |row| {
            row.get::<_, String>(0)
        })?,
        "queued"
    );
    assert_eq!(probe["model_ledger"]["jobs"], json!(before.0));
    assert_eq!(probe["model_ledger"]["publications"], json!(before.1));
    assert_eq!(probe["model_ledger"]["reservations"], json!(before.2));
    assert_eq!(
        fs::read_to_string(repository.join("lib.rs"))?,
        "pub fn retry_limit() -> u32 { 4 }\n"
    );

    assert_eq!(
        queries["search"]["method"],
        "shared-service-head-aware-median-of-three-v2"
    );
    assert_eq!(queries["cli_search"]["method"], "cli-process-round-trip-v1");
    let sets = queries["search"]["sets_ms"]
        .as_array()
        .ok_or("missing repeated timings")?;
    assert_eq!(sets.len(), 3);
    let mut p95s = sets
        .iter()
        .map(|set| set[0].as_f64().ok_or("missing sample"))
        .collect::<Result<Vec<_>, _>>()?;
    p95s.sort_by(f64::total_cmp);
    assert_eq!(queries["search"]["p95_ms"], json!(p95s[1]));
    let robust = &report["targets"][0]["robustness"];
    assert_eq!(robust["dead_jobs"], 0);
    assert_eq!(robust["job_states"], json!({}));
    assert_eq!(
        robust["parse_failures_by_language"],
        json!({"rust":{"files":1,"failures":0,"failure_rate":0.0,
            "failure_paths":[],"failure_paths_omitted":0}})
    );
    assert!(robust["extraction_failures"].is_null());
    let ladder = &report["targets"][0]["status"]["ladder"];
    assert_eq!(ladder["l1"], report["targets"][0]["index"]["units"]);
    assert_eq!(ladder["l2"], 0);
    assert_eq!(ladder["l3"], 0);
    assert_eq!(
        report["targets"][0]["index"]["cold_index_result"]["files_added"],
        2
    );
    let reconstruction = &report["targets"][0]["composites"]["pr_reconstruction"];
    assert_eq!(reconstruction["detected"], 0);
    assert_eq!(reconstruction["reconstructed"], 0);
    assert!(reconstruction["rate"].is_null());
    assert!(reconstruction.get("deferred").is_some());
    assert_eq!(
        report["targets"][0]["index"]["mtime_only_index_result"]["units_indexed"],
        0
    );
    let edits = &report["targets"][0]["index"]["content_edit"];
    assert_eq!(edits["paths"], json!(["lib.rs"]));
    assert_eq!(edits["restored"], true);
    assert_eq!(edits["index_result"]["files_modified"], 1);
    assert!(
        edits["index_result"]["units_indexed"]
            .as_u64()
            .is_some_and(|count| count >= 2)
    );
    assert_eq!(
        fs::read_to_string(repository.join("lib.rs"))?,
        "pub fn retry_limit() -> u32 { 4 }\n"
    );
    assert!(!repository.join(".deepkit/bench-edits").exists());

    let same = dkc(
        workspace.path(),
        [
            "bench",
            "diff",
            report_path.to_str().ok_or("non-UTF-8 report path")?,
            report_path.to_str().ok_or("non-UTF-8 report path")?,
            "--fail-on-regression",
            "--json",
        ],
    )?;
    assert_eq!(same["pass"], true);

    let baseline = workspace.path().join("baseline.json");
    let candidate = workspace.path().join("candidate.json");
    let mut baseline_report = report.clone();
    baseline_report["targets"][0]["quality"]["commit_derived"]["hit_file_at_5"] = json!(1.0);
    baseline_report["targets"][0]["quality"]["commit_derived"]["recall_at_10"] = json!(1.0);
    baseline_report["targets"][0]["paired_search"] = json!({
        "scores": [{
            "lexical": {"commit_derived": {"hit_file_at_5": 1.0, "recall_at_10": 1.0}},
            "hybrid": {"commit_derived": {"hit_file_at_5": 1.0, "recall_at_10": 1.0}},
            "reranked": {"commit_derived": {"hit_file_at_5": 1.0, "recall_at_10": 1.0}},
            "handwritten": {
                "lexical": {"cohorts": {"positive": {"hit_file_at_5": 1.0, "recall_at_10": 1.0}}},
                "hybrid": {"cohorts": {"positive": {"hit_file_at_5": 1.0, "recall_at_10": 1.0}}},
                "reranked": {"cohorts": {"positive": {"hit_file_at_5": 1.0, "recall_at_10": 1.0}}}
            }
        }]
    });
    baseline_report["targets"][0]["composites"]["pr_reconstruction"]["rate"] = json!(1.0);
    let mut candidate_report = baseline_report.clone();
    candidate_report["targets"][0]["quality"]["commit_derived"]["hit_file_at_5"] = json!(0.0);
    candidate_report["targets"][0]["quality"]["commit_derived"]["recall_at_10"] = json!(0.0);
    candidate_report["targets"][0]["paired_search"]["scores"][0]["reranked"] =
        json!({"commit_derived": {"hit_file_at_5": 0.97, "recall_at_10": 0.97}});
    candidate_report["targets"][0]["paired_search"]["scores"][0]["handwritten"]["reranked"]["cohorts"]
        ["positive"] = json!({"hit_file_at_5": 0.97, "recall_at_10": 0.97});
    candidate_report["targets"][0]["composites"]["pr_reconstruction"]["rate"] = json!(0.94);
    candidate_report["targets"][0]["robustness"]["dead_jobs"] = json!(1);
    fs::write(&baseline, serde_json::to_vec_pretty(&baseline_report)?)?;
    fs::write(&candidate, serde_json::to_vec_pretty(&candidate_report)?)?;
    let output = AssertCommand::cargo_bin("dkc")?
        .args([
            "bench",
            "diff",
            baseline.to_str().ok_or("non-UTF-8 baseline path")?,
            candidate.to_str().ok_or("non-UTF-8 candidate path")?,
            "--fail-on-regression",
            "--json",
        ])
        .current_dir(workspace.path())
        .output()?;
    assert!(!output.status.success());
    let failed: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(failed["pass"], false);
    assert!(
        failed["failures"]
            .as_array()
            .is_some_and(|failures| !failures.is_empty())
    );
    let failure_text = failed["failures"].to_string();
    assert!(failure_text.contains("commit_derived reranked hit_file_at_5"));
    assert!(failure_text.contains("handwritten reranked recall_at_10"));
    assert!(failure_text.contains("PR reconstruction dropped below 95%"));
    assert!(failure_text.contains("candidate reports dead jobs"));
    for (pointer, changed, reason) in [
        (
            "/targets/0/quality/eligibility/method",
            json!("source-raw-and-eligible-v1"),
            "golden eligibility methods",
        ),
        ("/build/opt_level", json!("different"), "build metadata"),
        ("/machine/arch", json!("different"), "architecture"),
        (
            "/targets/0/queries/search/method",
            json!("shared-formatted-median-of-three-v1"),
            "query timing methods",
        ),
        (
            "/targets/0/provenance/effective_config_blake3",
            json!("different"),
            "target inputs",
        ),
        (
            "/targets/0/provenance/page_size",
            json!(4096),
            "storage layout",
        ),
    ] {
        let mut incompatible = report.clone();
        *incompatible
            .pointer_mut(pointer)
            .ok_or("missing metadata field")? = changed;
        fs::write(&candidate, serde_json::to_vec(&incompatible)?)?;
        let output = AssertCommand::cargo_bin("dkc")?
            .args([
                "bench",
                "diff",
                report_path.to_str().ok_or("report path")?,
                candidate.to_str().ok_or("candidate path")?,
                "--fail-on-regression",
                "--json",
            ])
            .output()?;
        assert!(!output.status.success());
        let rejected: Value = serde_json::from_slice(&output.stdout)?;
        assert_eq!(rejected["comparisons"], json!([]));
        assert!(rejected["failures"].to_string().contains(reason));
    }
    Ok(())
}

fn git<const N: usize>(repository: &Path, arguments: [&str; N]) -> Result<(), Box<dyn Error>> {
    let status = Command::new("git")
        .args(arguments)
        .current_dir(repository)
        .status()?;
    assert!(status.success());
    Ok(())
}

fn git_stdout<const N: usize>(
    repository: &Path,
    arguments: [&str; N],
) -> Result<String, Box<dyn Error>> {
    let output = Command::new("git")
        .args(arguments)
        .current_dir(repository)
        .output()?;
    assert!(output.status.success());
    Ok(String::from_utf8(output.stdout)?)
}

fn dkc<const N: usize>(repository: &Path, args: [&str; N]) -> Result<Value, Box<dyn Error>> {
    let output = AssertCommand::cargo_bin("dkc")?
        .args(args)
        .current_dir(repository)
        .output()?;
    assert!(
        output.status.success(),
        "dkc failed: {}\n{}",
        String::from_utf8_lossy(&output.stderr),
        fs::read_to_string(repository.join("run.json")).unwrap_or_default()
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}

#[test]
fn phase_four_manifest_dry_run_lists_all_targets_and_filters_groups() -> Result<(), Box<dyn Error>>
{
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = root.join("bench/corpus.toml");
    let manifest_text = manifest.to_str().ok_or("non-UTF-8 corpus path")?;
    let all = dkc(
        root,
        [
            "bench",
            "run",
            "--corpus",
            manifest_text,
            "--dry-run",
            "--json",
        ],
    )?;
    assert_eq!(all["dry_run"], true);
    assert_eq!(all["profile"], "l1");
    assert_eq!(all["target_count"], 26);
    let names = all["targets"]
        .as_array()
        .ok_or("target array")?
        .iter()
        .filter_map(|target| target["name"].as_str())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(names.len(), 26);
    assert!(!names.contains("aurora"));
    for expected in ["ripgrep", "paddleocr", "fixtures-docs", "tiny"] {
        assert!(names.contains(expected));
    }

    let monsters = dkc(
        root,
        [
            "bench",
            "run",
            "--corpus",
            manifest_text,
            "--groups",
            "C",
            "--dry-run",
            "--json",
        ],
    )?;
    assert_eq!(monsters["target_count"], 4);
    let monster_names = monsters["targets"]
        .as_array()
        .ok_or("group C target array")?
        .iter()
        .filter_map(|target| target["name"].as_str())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        monster_names,
        ["kubernetes", "nextjs", "paddleocr", "vscode"]
            .into_iter()
            .collect()
    );

    let local = dkc(
        root,
        [
            "bench",
            "run",
            "--corpus",
            manifest_text,
            "--groups",
            "A",
            "--profile",
            "local",
            "--dry-run",
            "--json",
        ],
    )?;
    assert_eq!(local["profile"], "local");
    assert_eq!(local["target_count"], 11);
    for profile in ["cpu", "moonly", "ollama", "dual"] {
        let selected = dkc(
            root,
            [
                "bench",
                "run",
                "--corpus",
                manifest_text,
                "--only",
                "tiny",
                "--profile",
                profile,
                "--dry-run",
                "--json",
            ],
        )?;
        assert_eq!(selected["profile"], profile);
        assert_eq!(selected["target_count"], 1);
    }
    Ok(())
}

#[test]
fn failed_bench_targets_keep_structured_evidence_and_do_not_pass_diff() -> Result<(), Box<dyn Error>>
{
    let workspace = TempDir::new()?;
    for name in ["good", "bad"] {
        let root = workspace.path().join(name);
        fs::create_dir(&root)?;
        fs::write(root.join("lib.rs"), "pub fn benchmark_target() {}\n")?;
        git(&root, ["init", "--quiet"])?;
        git(&root, ["add", "lib.rs"])?;
        git(
            &root,
            [
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "initial source for structured benchmark outcome coverage",
            ],
        )?;
    }
    fs::write(workspace.path().join("bad.toml"), "[chunk]\nmax_tokens=0\n")?;
    let corpus = workspace.path().join("corpus.toml");
    fs::write(
        &corpus,
        "[[repo]]\nname='missing'\nurl='local'\npath='missing'\n[[repo]]\nname='bad'\nurl='local'\npath='bad'\nconfig='bad.toml'\n[[repo]]\nname='good'\nurl='local'\npath='good'\n",
    )?;
    let output = AssertCommand::cargo_bin("dkc")?
        .args([
            "bench",
            "run",
            "--corpus",
            "corpus.toml",
            "--out",
            "result.json",
            "--json",
        ])
        .current_dir(workspace.path())
        .output()?;
    assert!(!output.status.success());
    let summary: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(summary["targets"], 3);
    assert_eq!(summary["succeeded"], 1);
    assert_eq!(summary["failed"], 2);
    let report: Value = serde_json::from_slice(&fs::read(workspace.path().join("result.json"))?)?;
    assert_eq!(report["targets"][0]["name"], "good");
    let failures = report["failures"]
        .as_array()
        .ok_or("missing failure records")?;
    assert_eq!(failures.len(), 2);
    assert_eq!(failures[0]["stage"], "resolve_target");
    assert_eq!(failures[1]["stage"], "clean");
    assert_eq!(failures[1]["partial"]["configuration_applied"], true);
    assert!(
        failures[1]["exit_code"]
            .as_i64()
            .is_some_and(|code| code != 0)
    );
    assert_eq!(failures[1]["panics"], 0);
    assert!(failures[1].get("quality").is_none());
    assert!(failures[1]["partial"].get("cold_index_ms").is_none());
    assert!(fs::read_to_string(workspace.path().join("result.md"))?.contains("FAIL bad at clean"));
    let diff = AssertCommand::cargo_bin("dkc")?
        .args([
            "bench",
            "diff",
            "result.json",
            "result.json",
            "--fail-on-regression",
            "--json",
        ])
        .current_dir(workspace.path())
        .output()?;
    assert!(!diff.status.success());
    let compared: Value = serde_json::from_slice(&diff.stdout)?;
    assert_eq!(compared["pass"], false);
    Ok(())
}

#[test]
fn footprint_and_touch_use_admitted_text_including_tests() -> Result<(), Box<dyn Error>> {
    let workspace = TempDir::new()?;
    let repository = workspace.path().join("repo");
    fs::create_dir_all(repository.join("tests"))?;
    let admitted = [
        ("lib.rs", "pub fn eligible() {}\n"),
        ("tests/check.rs", "#[test] fn check() {}\n"),
        ("tests/broken.rs", "fn {\n"),
        (".gitignore", "ignored.txt\n"),
    ];
    for (path, text) in admitted {
        fs::write(repository.join(path), text)?;
    }
    fs::write(repository.join("ignored.txt"), "not eligible")?;
    fs::write(repository.join("oversized.txt"), "x".repeat(100))?;
    fs::write(repository.join("private.txt"), "explicit L0")?;
    fs::write(repository.join("binary.dat"), [0, 1, 2, 3])?;
    git(&repository, ["init", "--quiet"])?;
    git(&repository, ["add", "."])?;
    git(
        &repository,
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "initial",
        ],
    )?;
    let untouched = ["ignored.txt", "oversized.txt", "private.txt", "binary.dat"]
        .map(|path| Ok((path, fs::metadata(repository.join(path))?.modified()?)))
        .into_iter()
        .collect::<Result<Vec<_>, std::io::Error>>()?;
    fs::write(
        workspace.path().join("config.toml"),
        "[limits]\nmax_file_bytes='64 B'\n[[treatments]]\nname='private'\nmatch=['private.txt']\nmax_level=0\n",
    )?;
    fs::write(
        workspace.path().join("corpus.toml"),
        "[[repo]]\nname='fixture'\nurl='local'\npath='repo'\nconfig='config.toml'\n",
    )?;
    dkc(
        workspace.path(),
        [
            "bench",
            "run",
            "--corpus",
            "corpus.toml",
            "--out",
            "run.json",
            "--json",
        ],
    )?;
    let report: Value = serde_json::from_slice(&fs::read(workspace.path().join("run.json"))?)?;
    let target = &report["targets"][0];
    assert_eq!(
        target["size"]["checkout_bytes"],
        admitted[..3]
            .iter()
            .map(|(_, text)| text.len())
            .sum::<usize>()
    );
    assert_eq!(target["size"]["accounting"]["admitted_text_files"], 3);
    assert_eq!(target["size"]["accounting"]["walker_files"], 6);
    assert_eq!(target["size"]["accounting"]["excluded"]["byte_cap"], 1);
    assert_eq!(target["index"]["touched_files"], 3);
    assert_eq!(
        target["robustness"]["parse_failures_by_language"]["rust"],
        json!({"files":3,"failures":1,"failure_rate":1.0/3.0,
            "failure_paths":["tests/broken.rs"],"failure_paths_omitted":0})
    );
    assert_eq!(
        target["size"]["accounting"]["total_db_bytes"].as_u64(),
        Some(
            target["size"]["db_bytes"].as_u64().ok_or("db bytes")?
                + target["size"]["accounting"]["wal_bytes"]
                    .as_u64()
                    .ok_or("wal bytes")?
        )
    );
    for (path, modified) in untouched {
        assert_eq!(
            fs::metadata(repository.join(path))?.modified()?,
            modified,
            "{path}"
        );
    }
    Ok(())
}

#[test]
fn commit_goldens_keep_raw_denominators_and_diagnose_ineligible_paths() -> Result<(), Box<dyn Error>>
{
    let workspace = TempDir::new()?;
    let root = workspace.path().join("repo");
    fs::create_dir_all(root.join("tests"))?;
    fs::create_dir_all(root.join("specimens"))?;
    git(&root, ["init", "--quiet"])?;
    fs::write(root.join("seed.txt"), "seed")?;
    git(&root, ["add", "."])?;
    commit_fixture(&root, "initial")?;
    for (path, content) in [
        ("live.rs", "pub fn needle() {}\n".to_owned()),
        ("tests/check.rs", "pub fn needle() {}\n".to_owned()),
        ("specimens/case.rs", "pub fn needle() {}\n".to_owned()),
        ("removed.rs", "pub fn needle() {}\n".to_owned()),
        ("ignored.rs", "pub fn needle() {}\n".to_owned()),
        ("oversized.rs", "x".repeat(100)),
        ("empty.rs", String::new()),
        ("private.rs", "pub fn needle() {}\n".to_owned()),
    ] {
        fs::write(root.join(path), content)?;
        git(&root, ["add", path])?;
        commit_fixture(
            &root,
            &format!("locate needle behavior in the source module changed at {path}"),
        )?;
    }
    git(&root, ["rm", "--quiet", "removed.rs"])?;
    fs::write(root.join(".gitignore"), "ignored.rs\n")?;
    git(&root, ["add", ".gitignore"])?;
    commit_fixture(&root, "remove and ignore")?;
    fs::write(
        workspace.path().join("config.toml"),
        "[limits]\nmax_file_bytes='64 B'\n[tests]\nglobs=['specimens/**']\n[[treatments]]\nname='private'\nmatch=['private.rs']\nmax_level=0\n",
    )?;
    fs::write(
        workspace.path().join("corpus.toml"),
        "[[repo]]\nname='fixture'\nurl='local'\npath='repo'\nconfig='config.toml'\n",
    )?;
    dkc(
        workspace.path(),
        [
            "bench",
            "run",
            "--corpus",
            "corpus.toml",
            "--out",
            "run.json",
            "--json",
        ],
    )?;
    let report: Value = serde_json::from_slice(&fs::read(workspace.path().join("run.json"))?)?;
    let quality = &report["targets"][0]["quality"];
    assert_eq!(quality["commit_derived"]["count"], 7);
    let eligibility = &quality["eligibility"];
    assert_eq!(eligibility["eligible"]["count"], 2);
    assert_eq!(eligibility["raw_expected_paths"], 7);
    assert_eq!(eligibility["eligible_expected_paths"], 2);
    assert_eq!(eligibility["tests_globs"], json!(["specimens/**"]));
    let rejected = eligibility["rejected_paths"]
        .as_array()
        .ok_or("rejected paths")?;
    for (path, reason) in [
        ("removed.rs", "missing_at_pinned_tree"),
        ("ignored.rs", "not_admitted_by_walker"),
        ("oversized.rs", "byte_cap"),
        ("empty.rs", "no_searchable_source_units"),
        ("private.rs", "l0_treatment"),
    ] {
        assert!(
            rejected
                .iter()
                .any(|entry| entry["path"] == path && entry["reason"] == reason),
            "{path}: {rejected:?}"
        );
    }
    assert_eq!(rejected.len(), 5);
    Ok(())
}

fn commit_fixture(root: &Path, message: &str) -> Result<(), Box<dyn Error>> {
    git(
        root,
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--quiet",
            "-m",
            message,
        ],
    )
}

#[test]
fn handwritten_goldens_keep_review_and_cohort_denominators_explicit() -> Result<(), Box<dyn Error>>
{
    let workspace = TempDir::new()?;
    let root = workspace.path().join("repo");
    fs::create_dir(&root)?;
    fs::write(root.join("lib.rs"), "pub fn repeat_me() {}\n")?;
    fs::write(
        root.join("README.md"),
        "# Instructions\n\nUsage instructions for the library.\n",
    )?;
    git(&root, ["init", "--quiet"])?;
    git(&root, ["add", "."])?;
    commit_fixture(&root, "initial")?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/bench");
    fs::copy(
        fixture.join("goldens.yaml"),
        workspace.path().join("goldens.yaml"),
    )?;
    fs::copy(
        fixture.join("extras.yaml"),
        workspace.path().join("extras.yaml"),
    )?;
    let manifest = "[[repo]]\nname='fixture'\nurl='local'\npath='repo'\ngolden='goldens.yaml'\nextra_goldens=['extras.yaml']\n";
    fs::write(workspace.path().join("corpus.toml"), manifest)?;
    dkc(
        workspace.path(),
        [
            "bench",
            "run",
            "--corpus",
            "corpus.toml",
            "--out",
            "run.json",
            "--json",
        ],
    )?;
    let report: Value = serde_json::from_slice(&fs::read(workspace.path().join("run.json"))?)?;
    let handwritten = &report["targets"][0]["quality"]["handwritten"];
    assert_eq!(handwritten["review_status"], "unreviewed_draft");
    assert_eq!(handwritten["cohorts"]["positive"]["count"], 2);
    assert_eq!(handwritten["cohorts"]["positive"]["hit_file_at_5"], 1.0);
    assert_eq!(handwritten["cohorts"]["positive"]["hit_symbol_at_5"], 1.0);
    assert_eq!(handwritten["cohorts"]["held_out"]["count"], 1);
    assert_eq!(handwritten["cohorts"]["scoped"]["count"], 1);
    assert_eq!(handwritten["cohorts"]["no_answer"]["count"], 1);
    assert!(handwritten["cohorts"]["no_answer"]["hit_file_at_5"].is_null());
    assert_eq!(
        handwritten["cohorts"]["no_answer"]["no_answer_empty_rate"],
        1.0
    );
    assert_eq!(handwritten["latency"]["samples"], 15);
    assert!(
        handwritten["cases"][0]["response_tokens"]
            .as_u64()
            .is_some_and(|value| value > 0)
    );
    assert_eq!(
        report["targets"][0]["quality"]["commit_derived"]["count"],
        0
    );
    assert!(fs::read_to_string(workspace.path().join("run.md"))?.contains("unreviewed_draft"));

    // A changed review digest fails instead of silently claiming reviewed quality.
    fs::write(
        workspace.path().join("corpus.toml"),
        format!("{manifest}golden_reviewed_blake3='wrong'\n"),
    )?;
    let output = AssertCommand::cargo_bin("dkc")?
        .args([
            "bench",
            "run",
            "--corpus",
            "corpus.toml",
            "--out",
            "invalid.json",
            "--json",
        ])
        .current_dir(workspace.path())
        .output()?;
    assert!(!output.status.success());
    let invalid: Value = serde_json::from_slice(&fs::read(workspace.path().join("invalid.json"))?)?;
    assert!(
        invalid["failures"][0]["error"]
            .as_str()
            .is_some_and(|message| message.contains("reviewed golden hash"))
    );
    Ok(())
}

#[test]
fn benchmark_metadata_identifies_executable_without_loading_repository_configuration()
-> Result<(), Box<dyn Error>> {
    let root = TempDir::new()?;
    fs::create_dir(root.path().join(".deepkit"))?;
    let config = root.path().join(".deepkit/config.toml");
    fs::write(&config, "[intentionally malformed")?;
    let report = dkc(root.path(), ["bench", "metadata", "--json"])?;
    let executable = Path::new(env!("CARGO_BIN_EXE_dkc"));
    assert_eq!(
        report["build"]["binary_bytes"],
        fs::metadata(executable)?.len()
    );
    assert_eq!(
        report["build"]["binary_blake3"],
        blake3::hash(&fs::read(executable)?).to_hex().as_str()
    );
    assert_eq!(
        report["build"]["source_blake3"]
            .as_str()
            .ok_or("source hash")?
            .len(),
        64
    );
    assert_eq!(fs::read_to_string(config)?, "[intentionally malformed");
    assert_eq!(fs::read_dir(root.path().join(".deepkit"))?.count(), 1);
    assert!(!root.path().join(".git").exists());
    Ok(())
}
