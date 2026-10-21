//! Public finite embedding passes retain L1, honor queue opt-out and resume budgets.
use assert_cmd::Command;
use rusqlite::Connection;
use serde_json::{Value, json};
use std::{error::Error, fs, path::Path, time::Duration};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

#[cfg(unix)]
type VectorSnapshot = Vec<(i64, Vec<u8>, Vec<u8>)>;

#[cfg(unix)]
fn vectors(root: &Path) -> VectorSnapshot {
    let db = Connection::open(root.join(".deepkit/code.db")).unwrap();
    let slug: String = db
        .query_row("SELECT slug FROM embedding_tables", [], |row| row.get(0))
        .unwrap();
    assert!(slug.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'));
    db.prepare(&format!(
        "SELECT unit_id,v,ctx_hash FROM vec_{slug} ORDER BY unit_id"
    ))
    .unwrap()
    .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
    .unwrap()
    .collect::<Result<_, _>>()
    .unwrap()
}

#[cfg(unix)]
#[tokio::test]
async fn killed_public_index_preserves_commits_and_recovers_only_expired_leases()
-> Result<(), Box<dyn Error>> {
    interrupted_index_recovery(9).await
}

#[cfg(unix)]
#[tokio::test]
async fn graceful_public_index_shutdown_preserves_commits_and_recoverable_leases()
-> Result<(), Box<dyn Error>> {
    for signal in [2, 15] {
        interrupted_index_recovery(signal).await?;
    }
    Ok(())
}

#[cfg(unix)]
async fn interrupted_index_recovery(signal: i32) -> Result<(), Box<dyn Error>> {
    use std::process::Stdio;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    for stop_at in [1, 2] {
        let root = tempfile::tempdir()?;
        let server = MockServer::start().await;
        let calls = Arc::new(AtomicUsize::new(0));
        let requests = Arc::clone(&calls);
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(move |request: &wiremock::Request| {
                let ordinal = requests.fetch_add(1, Ordering::SeqCst) + 1;
                let body: Value = serde_json::from_slice(&request.body).unwrap();
                let embeddings = vec![vec![1.0, 0.0]; body["input"].as_array().unwrap().len()];
                let response =
                    ResponseTemplate::new(200).set_body_json(json!({"embeddings": embeddings}));
                if ordinal == stop_at {
                    response.set_delay(Duration::from_secs(30))
                } else {
                    response
                }
            })
            .mount(&server)
            .await;
        configure(root.path(), &server.uri(), "")?;
        let config_path = root.path().join(".deepkit/config.toml");
        fs::write(
            &config_path,
            fs::read_to_string(&config_path)?.replace("'2s'", "'60s'"),
        )?;
        for name in ["alpha", "beta", "gamma"] {
            fs::write(
                root.path().join(format!("{name}.rs")),
                format!("pub fn {name}_recovery() {{}}\n"),
            )?;
        }
        let mut child = tokio::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
            .current_dir(root.path())
            .env("XDG_CONFIG_HOME", root.path().join("empty-config"))
            .env("DKC_MODELS_DIR", root.path().join("unused-models"))
            .args(["index", "--level", "2", "--json"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()?;
        tokio::time::timeout(Duration::from_secs(10), async {
            while calls.load(Ordering::SeqCst) < stop_at {
                assert!(
                    child.try_wait().unwrap().is_none(),
                    "index exited before crash point"
                );
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await?;
        // Hold the writer through graceful exit: shutdown must not start
        // maintenance or wait for a fresh database admission after cancellation.
        let shutdown_lock = if signal != 9 {
            let db = Connection::open(root.path().join(".deepkit/code.db"))?;
            db.execute_batch("BEGIN IMMEDIATE")?;
            Some(db)
        } else {
            None
        };
        let pid = child.id().ok_or("missing child PID")?;
        assert!(
            std::process::Command::new("kill")
                .args([format!("-{signal}"), pid.to_string()])
                .status()?
                .success()
        );
        let output =
            tokio::time::timeout(Duration::from_secs(5), child.wait_with_output()).await??;
        use std::os::unix::process::ExitStatusExt;
        if signal == 9 {
            assert_eq!(output.status.signal(), Some(9));
        } else {
            assert_eq!(output.status.signal(), None);
            assert!(!output.status.success());
            let report: Value = serde_json::from_slice(&output.stdout)?;
            assert_eq!(report["embedding"]["state"], "cancelled", "{report}");
            assert_eq!(
                report["embedding"]["providers"][0]["published"],
                stop_at - 1
            );
            assert!(report["maintenance"].is_null());
        }
        assert_eq!(calls.load(Ordering::SeqCst), stop_at);
        if let Some(db) = shutdown_lock {
            db.execute_batch("ROLLBACK")?;
        }
        let committed = vectors(root.path());
        assert_eq!(committed.len(), stop_at - 1);
        assert_eq!(
            count(
                root.path(),
                "SELECT count(*) FROM jobs WHERE state='leased'"
            ),
            1
        );
        let db = Connection::open(root.path().join(".deepkit/code.db"))?;
        let (job, generation, attempts): (i64, i64, i64) = db.query_row(
            "SELECT id,lease_generation,attempts FROM jobs WHERE state='leased'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        let reservation: String = db.query_row(
            "SELECT id FROM usage_reservations WHERE job_id=?1 AND state='dispatched'",
            [job],
            |row| row.get(0),
        )?;
        let waiting = index(root.path(), &["index", "--level", "2", "--json"], true).await;
        assert_eq!(waiting["embedding"]["state"], "pending", "{waiting}");
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        assert_eq!(
            count(
                root.path(),
                "SELECT count(*) FROM jobs WHERE state='leased'"
            ),
            1
        );
        assert_eq!(vectors(root.path()).len(), 2);
        // Advance only the abandoned fixture lease; production lease duration
        // and heartbeat are unchanged, and an immediate restart above respected it.
        assert_eq!(
            db.execute(
                "UPDATE jobs SET leased_until=0 WHERE id=?1 AND state='leased'",
                [job]
            )?,
            1
        );
        let resumed = index(root.path(), &["index", "--level", "2", "--json"], true).await;
        assert_eq!(resumed["embedding"]["state"], "quiescent", "{resumed}");
        assert_eq!(calls.load(Ordering::SeqCst), 4);
        let recovered = vectors(root.path());
        assert_eq!(recovered.len(), 3);
        for vector in committed {
            assert!(
                recovered.contains(&vector),
                "committed vector changed after restart"
            );
        }
        assert_eq!(
            count(
                root.path(),
                "SELECT count(*) FROM units WHERE ladder_level<2"
            ),
            0
        );
        assert_eq!(
            count(root.path(), "SELECT count(*) FROM jobs WHERE state!='done'"),
            0
        );
        let recovered_job: (i64, i64) = db.query_row(
            "SELECT lease_generation,attempts FROM jobs WHERE id=?1",
            [job],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        assert_eq!(recovered_job, (generation + 1, attempts + 1));
        let accounting: String = db.query_row(
            "SELECT state FROM usage_reservations WHERE id=?1",
            [reservation],
            |row| row.get(0),
        )?;
        assert_eq!(
            accounting, "dispatched",
            "uncertain request accounting was erased"
        );
        let repeat = index(root.path(), &["index", "--level", "2", "--json"], true).await;
        assert_eq!(repeat["embedding"]["state"], "quiescent");
        assert_eq!(calls.load(Ordering::SeqCst), 4);
        assert_eq!(vectors(root.path()), recovered);
    }
    Ok(())
}

async fn index(root: &Path, args: &[&str], success: bool) -> Value {
    let root = root.to_owned();
    let args: Vec<String> = args.iter().map(|s| (*s).into()).collect();
    tokio::task::spawn_blocking(move || {
        let assertion = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
            .current_dir(&root)
            .env("XDG_CONFIG_HOME", root.join("empty-config"))
            .env("DKC_MODELS_DIR", root.join("unused-models"))
            .timeout(Duration::from_secs(20))
            .args(args)
            .assert();
        let assertion = if success {
            assertion.success()
        } else {
            assertion.failure()
        };
        serde_json::from_slice(&assertion.get_output().stdout).unwrap()
    })
    .await
    .unwrap()
}
fn configure(root: &Path, url: &str, extra: &str) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(root.join(".deepkit"))?;
    fs::write(
        root.join(".deepkit/config.toml"),
        format!(
            "[embedder]\nprofile='ollama'\ndims=2\n[embedder.ollama]\nurl='{url}'\n[queue]\nmax_schedule_per_pass=1\nmax_jobs_per_pass=1\nembedding_timeout='2s'\n{extra}"
        ),
    )?;
    Ok(())
}
#[cfg(unix)]
#[tokio::test]
async fn public_context_refresh_replaces_only_stale_coverage_and_repeats_without_calls()
-> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let server = provider(200).await;
    configure(root.path(), &server.uri(), "")?;
    for name in ["alpha", "beta"] {
        fs::write(
            root.path().join(format!("{name}.rs")),
            format!("pub fn {name}() {{}}\n"),
        )?;
    }
    index(root.path(), &["index", "--level", "2", "--json"], true).await;
    let original = vectors(root.path());
    assert!(original.len() >= 2);
    let calls = server.received_requests().await.ok_or("requests")?.len();
    // Isolate the public scheduling contract with one retained context mismatch.
    // The saved neutral publication proof and every source byte remain intact.
    let invalidate = || -> Result<(), Box<dyn Error>> {
        let db = Connection::open(root.path().join(".deepkit/code.db"))?;
        let slug: String = db.query_row("SELECT slug FROM embedding_tables", [], |r| r.get(0))?;
        assert!(slug.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'));
        db.execute(
            &format!("UPDATE vec_{slug} SET ctx_hash=zeroblob(32) WHERE unit_id=?1"),
            [original[0].0],
        )?;
        Ok(())
    };
    invalidate()?;
    let retained = vectors(root.path());
    index(root.path(), &["index", "--level", "2", "--json"], true).await;
    assert_eq!(vectors(root.path()), retained);
    assert_eq!(
        server.received_requests().await.ok_or("requests")?.len(),
        calls
    );
    for expected in [calls + 1, calls + 1] {
        let output = index(
            root.path(),
            &["index", "--level", "2", "--refresh-context", "--json"],
            true,
        )
        .await;
        assert_eq!(output["embedding"]["state"], "quiescent", "{output}");
        assert_eq!(
            server.received_requests().await.ok_or("requests")?.len(),
            expected
        );
        assert_eq!(vectors(root.path()), original);
    }
    invalidate()?;
    let config = root.path().join(".deepkit/config.toml");
    fs::write(
        &config,
        fs::read_to_string(&config)?
            .replace("[embedder]", "[embedder]\nreembed_on_role_change=true"),
    )?;
    index(root.path(), &["index", "--level", "2", "--json"], true).await;
    assert_eq!(
        server.received_requests().await.ok_or("requests")?.len(),
        calls + 2
    );
    assert_eq!(vectors(root.path()), original);
    Ok(())
}

async fn provider(status: u16) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(move |request: &wiremock::Request| {
            if status != 200 {
                return ResponseTemplate::new(status);
            }
            let body: Value = serde_json::from_slice(&request.body).unwrap();
            let embeddings = vec![vec![1.0, 0.0]; body["input"].as_array().unwrap().len()];
            ResponseTemplate::new(200).set_body_json(json!({"embeddings": embeddings}))
        })
        .mount(&server)
        .await;
    server
}
fn count(root: &Path, sql: &str) -> i64 {
    Connection::open(root.join(".deepkit/code.db"))
        .unwrap()
        .query_row(sql, [], |row| row.get(0))
        .unwrap()
}

#[cfg(unix)]
#[tokio::test]
async fn public_index_rechecks_config_during_response_and_recovers_without_losing_commits()
-> Result<(), Box<dyn Error>> {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    for change in ["endpoint", "budget", "policy", "root"] {
        let root = tempfile::tempdir()?;
        let server = MockServer::start().await;
        let alternate = provider(200).await;
        configure(root.path(), &server.uri(), "")?;
        let config_path = root.path().join(".deepkit/config.toml");
        let original = fs::read_to_string(&config_path)?;
        fs::create_dir(root.path().join("elsewhere"))?;
        let changed = match change {
            "endpoint" => original.replace(&server.uri(), &alternate.uri()),
            "budget" => format!(
                "{original}[[queue.budgets]]\nprofile='ollama'\nurl='{}'\ndaily_calls=0\n",
                server.uri()
            ),
            "policy" => format!(
                "{original}\n[[treatments]]\nname='protected'\nmatch=['**/*.rs']\nmax_level=2\nnever_send=true\n"
            ),
            "root" => format!("{original}\n[index]\nroot='elsewhere'\n"),
            _ => unreachable!(),
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let requests = Arc::clone(&calls);
        let config_file = config_path.clone();
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(move |request: &wiremock::Request| {
                if requests.fetch_add(1, Ordering::SeqCst) == 1 {
                    // The request was dispatched under the old settings. Change
                    // them before returning its response, without a timing race.
                    fs::write(&config_file, &changed).unwrap();
                }
                let body: Value = serde_json::from_slice(&request.body).unwrap();
                ResponseTemplate::new(200).set_body_json(json!({
                    "embeddings": vec![vec![1.0, 0.0]; body["input"].as_array().unwrap().len()]
                }))
            })
            .mount(&server)
            .await;
        for name in ["alpha", "beta", "gamma"] {
            fs::write(
                root.path().join(format!("{name}.rs")),
                format!("pub fn {name}_config() {{}}\n"),
            )?;
        }
        let failed = index(root.path(), &["index", "--level", "2", "--json"], false).await;
        assert_eq!(failed["embedding"]["state"], "failed", "{change}: {failed}");
        assert_eq!(calls.load(Ordering::SeqCst), 2, "{change}");
        assert!(alternate.received_requests().await.unwrap().is_empty());
        let retained = vectors(root.path());
        assert_eq!(retained.len(), 1, "{change}: stale response published");
        assert_eq!(count(root.path(), "SELECT count(*) FROM units"), 3);
        assert_eq!(
            count(
                root.path(),
                "SELECT count(*) FROM files WHERE l1_pending!=0"
            ),
            0
        );
        assert_eq!(
            count(
                root.path(),
                "SELECT count(*) FROM jobs WHERE state='leased'"
            ),
            0
        );
        assert_eq!(count(root.path(), "SELECT sum(calls) FROM usage"), 2);
        // Restoring the same effective settings must rebuild discarded output,
        // preserve the first commit, and never repeat a completed request.
        fs::write(&config_path, &original)?;
        let resumed = index(root.path(), &["index", "--level", "2", "--json"], true).await;
        assert_eq!(
            resumed["embedding"]["state"], "quiescent",
            "{change}: {resumed}"
        );
        let completed = vectors(root.path());
        assert_eq!(completed.len(), 3);
        assert!(completed.contains(&retained[0]));
        assert_eq!(calls.load(Ordering::SeqCst), 4);
        let repeat = index(root.path(), &["index", "--level", "2", "--json"], true).await;
        assert_eq!(repeat["embedding"]["state"], "quiescent");
        assert_eq!(calls.load(Ordering::SeqCst), 4);
        assert_eq!(vectors(root.path()), completed);
    }
    Ok(())
}

#[cfg(unix)]
fn git(root: &Path, args: &[&str]) -> String {
    let output = std::process::Command::new("git")
        .current_dir(root)
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

#[cfg(unix)]
#[tokio::test]
async fn public_index_checks_head_between_passes_without_advancing_completed_head()
-> Result<(), Box<dyn Error>> {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    for change in ["commit", "unborn", "invalid", "detach"] {
        let root = tempfile::tempdir()?;
        let server = MockServer::start().await;
        configure(root.path(), &server.uri(), "")?;
        for name in ["alpha", "beta", "gamma"] {
            fs::write(
                root.path().join(format!("{name}.rs")),
                format!("pub fn {name}_head() {{}}\n"),
            )?;
        }
        git(root.path(), &["init", "-q"]);
        git(root.path(), &["config", "user.name", "Fixture"]);
        git(
            root.path(),
            &["config", "user.email", "fixture@example.invalid"],
        );
        git(root.path(), &["add", "alpha.rs", "beta.rs", "gamma.rs"]);
        git(root.path(), &["commit", "-qm", "initial"]);
        let initial = git(root.path(), &["rev-parse", "HEAD"]);
        let head_file = fs::read(root.path().join(".git/HEAD"))?;
        let directory = root.path().to_owned();
        let calls = Arc::new(AtomicUsize::new(0));
        let requests = Arc::clone(&calls);
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(move |request: &wiremock::Request| {
                if requests.fetch_add(1, Ordering::SeqCst) == 1 {
                    match change {
                        "commit" => {
                            git(&directory, &["commit", "--allow-empty", "-qm", "next"]);
                        }
                        "unborn" => {
                            git(&directory, &["symbolic-ref", "HEAD", "refs/heads/unborn"]);
                        }
                        "invalid" => {
                            fs::write(directory.join(".git/HEAD"), "invalid HEAD\n").unwrap();
                        }
                        "detach" => {
                            git(&directory, &["checkout", "-q", "--detach"]);
                        }
                        _ => unreachable!(),
                    }
                }
                let body: Value = serde_json::from_slice(&request.body).unwrap();
                ResponseTemplate::new(200).set_body_json(json!({
                    "embeddings": vec![vec![1.0, 0.0]; body["input"].as_array().unwrap().len()]
                }))
            })
            .mount(&server)
            .await;
        let same = change == "detach";
        let result = index(root.path(), &["index", "--level", "2", "--json"], same).await;
        assert_eq!(
            result["embedding"]["state"],
            if same { "quiescent" } else { "failed" },
            "{change}: {result}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), if same { 7 } else { 2 });
        // Source bytes and effective configuration still authorize these vectors;
        // a changed commit must stop the sweep, without undoing valid publication.
        let retained = vectors(root.path());
        assert_eq!(retained.len(), if same { 7 } else { 2 });
        let db = Connection::open(root.path().join(".deepkit/code.db"))?;
        let indexed: String =
            db.query_row("SELECT value FROM meta WHERE key='indexed_head'", [], |r| {
                r.get(0)
            })?;
        assert_eq!(indexed, initial);
        assert_eq!(
            count(
                root.path(),
                "SELECT count(*) FROM jobs WHERE state='leased'"
            ),
            0
        );
        fs::write(root.path().join(".git/HEAD"), &head_file)?;
        let resumed = index(root.path(), &["index", "--level", "2", "--json"], true).await;
        assert_eq!(
            resumed["embedding"]["state"], "quiescent",
            "{change}: {resumed}"
        );
        // Three source files, their three histories, and one immutable commit
        // per observed HEAD. Histories add independently verified vectors.
        // A newly created commit is discovered only on the resumed L1 pass.
        let commits = if change == "commit" { 2 } else { 1 };
        let expected = 6 + commits;
        assert_eq!(calls.load(Ordering::SeqCst), expected);
        let completed = vectors(root.path());
        assert_eq!(completed.len(), expected);
        assert_eq!(
            count(
                root.path(),
                "SELECT count(*) FROM units WHERE file_id IS NOT NULL AND ladder_level=2"
            ),
            3
        );
        assert_eq!(
            count(root.path(), "SELECT count(*) FROM commits"),
            commits as i64
        );
        assert_eq!(
            count(
                root.path(),
                "SELECT count(*) FROM units WHERE kind='history' AND ladder_level=2"
            ),
            3
        );
        assert!(retained.iter().all(|v| completed.contains(v)));
        let indexed: String =
            db.query_row("SELECT value FROM meta WHERE key='indexed_head'", [], |r| {
                r.get(0)
            })?;
        assert_eq!(indexed, git(root.path(), &["rev-parse", "HEAD"]));
    }
    Ok(())
}
#[tokio::test]
async fn index_executes_all_catalogue_pages_and_unchanged_retry_makes_no_calls()
-> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let server = provider(200).await;
    configure(root.path(), &server.uri(), "")?;
    fs::write(
        root.path().join("a.rs"),
        "pub fn alpha_value() -> bool { true }\n",
    )?;
    fs::write(
        root.path().join("b.rs"),
        "pub fn beta_value() -> bool { false }\n",
    )?;
    let result = index(root.path(), &["index", "--level", "2", "--json"], true).await;
    assert_eq!(result["embedding"]["state"], "quiescent", "{result}");
    let calls = server.received_requests().await.unwrap().len();
    assert!(calls >= 2);
    assert_eq!(
        count(
            root.path(),
            "SELECT count(*) FROM jobs WHERE state='leased'"
        ),
        0
    );
    assert_eq!(
        count(
            root.path(),
            "SELECT count(*) FROM units WHERE ladder_level<2"
        ),
        0
    );
    assert_eq!(
        result["embedding"]["providers"][0]["catalogue_finished"],
        true
    );
    assert!(
        result["embedding"]["providers"][0]["passes"]
            .as_u64()
            .unwrap()
            >= 2
    );
    let repeat = index(root.path(), &["index", "--level", "2", "--json"], true).await;
    assert_eq!(repeat["embedding"]["state"], "quiescent", "{repeat}");
    assert_eq!(server.received_requests().await.unwrap().len(), calls);
    assert!(!root.path().join("unused-models").exists());
    Ok(())
}
#[tokio::test]
async fn opt_out_preserves_queue_and_zero_budget_defers_then_recovers() -> Result<(), Box<dyn Error>>
{
    let root = tempfile::tempdir()?;
    let server = provider(200).await;
    configure(
        root.path(),
        &server.uri(),
        &format!(
            "[[queue.budgets]]\nprofile='ollama'\nurl='{}'\ndaily_calls=0\n",
            server.uri()
        ),
    )?;
    fs::write(
        root.path().join("a.rs"),
        "pub fn retained_lexical_value() -> bool { true }\n",
    )?;
    for args in [
        vec!["index", "--no-queue", "--json"],
        vec!["index", "--level", "1", "--json"],
    ] {
        let skipped = index(root.path(), &args, true).await;
        assert_eq!(skipped["embedding"]["state"], "skipped");
        assert_eq!(count(root.path(), "SELECT count(*) FROM jobs"), 0);
        assert_eq!(
            count(root.path(), "SELECT count(*) FROM embedding_tables"),
            0
        );
    }
    let waiting = index(root.path(), &["index", "--level", "2", "--json"], true).await;
    assert_eq!(waiting["embedding"]["state"], "pending", "{waiting}");
    assert!(
        count(
            root.path(),
            "SELECT count(*) FROM jobs WHERE state='queued' AND not_before>0"
        ) > 0
    );
    let before = count(root.path(), "SELECT sum(not_before) FROM jobs");
    index(root.path(), &["index", "--no-queue", "--json"], true).await;
    assert_eq!(
        count(root.path(), "SELECT sum(not_before) FROM jobs"),
        before
    );
    assert!(server.received_requests().await.unwrap().is_empty());
    let found = index(
        root.path(),
        &["search", "retained_lexical_value", "--no-rerank", "--json"],
        true,
    )
    .await;
    assert!(!found["hits"].as_array().unwrap().is_empty());
    configure(root.path(), &server.uri(), "")?;
    // Budget changes preserve the recorded delay. Advance this fixture's job
    // eligibility explicitly to exercise the next eligible run without a day wait.
    Connection::open(root.path().join(".deepkit/code.db"))?
        .execute("UPDATE jobs SET not_before=0 WHERE state='queued'", [])?;
    let resumed = index(root.path(), &["index", "--level", "2", "--json"], true).await;
    assert_eq!(resumed["embedding"]["state"], "quiescent", "{resumed}");
    assert!(!server.received_requests().await.unwrap().is_empty());
    Ok(())
}
#[tokio::test]
async fn failed_provider_keeps_searchable_l1_and_reports_nonzero_json() -> Result<(), Box<dyn Error>>
{
    let root = tempfile::tempdir()?;
    let server = provider(401).await;
    configure(root.path(), &server.uri(), "")?;
    fs::write(
        root.path().join("a.rs"),
        "pub fn failed_embedding_retains_source() {}\n",
    )?;
    let result = index(root.path(), &["index", "--level", "2", "--json"], false).await;
    assert_eq!(result["embedding"]["state"], "failed", "{result}");
    assert!(count(root.path(), "SELECT count(*) FROM units") > 0);
    assert!(count(root.path(), "SELECT count(*) FROM jobs WHERE state='dead'") > 0);
    let found = index(
        root.path(),
        &[
            "search",
            "failed_embedding_retains_source",
            "--no-rerank",
            "--json",
        ],
        true,
    )
    .await;
    assert!(!found["hits"].as_array().unwrap().is_empty());
    Ok(())
}

#[tokio::test]
async fn local_opt_out_avoids_artifacts_and_vector_cap_pauses_without_dispatch()
-> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    fs::create_dir(root.path().join(".deepkit"))?;
    fs::write(
        root.path().join(".deepkit/config.toml"),
        "[embedder]\nprofile='local'\n[embedder.local]\nmodel='missing-artifact'\n",
    )?;
    fs::write(
        root.path().join("a.rs"),
        "pub fn capacity_preserves_source() {}\n",
    )?;
    let skipped = index(root.path(), &["index", "--no-queue", "--json"], true).await;
    assert_eq!(skipped["embedding"]["state"], "skipped");
    assert_eq!(skipped["model_preparation"]["state"], "skipped");
    assert!(!root.path().join("unused-models").exists());
    let server = provider(200).await;
    configure(root.path(), &server.uri(), "[limits]\nmax_l2_units=0\n")?;
    let capped = index(root.path(), &["index", "--level", "2", "--json"], true).await;
    assert_eq!(capped["embedding"]["state"], "pending", "{capped}");
    assert!(server.received_requests().await.unwrap().is_empty());
    assert_eq!(
        count(
            root.path(),
            "SELECT count(*) FROM units WHERE ladder_level=2"
        ),
        0
    );
    let status = index(root.path(), &["queue", "status", "--json"], true).await;
    let provider = status["providers"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap();
    assert_eq!(provider["deferrals"]["cap_vectors"]["jobs"], 1, "{status}");
    Ok(())
}
