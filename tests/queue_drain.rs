//! Public persisted-only drain contracts; v1 §8 and phase T1.5.
use assert_cmd::Command;
use rusqlite::Connection;
use serde_json::{Value, json};
use std::{
    error::Error,
    fs,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

async fn cli(root: &Path, args: &[&str], success: bool) -> Value {
    let root = root.to_owned();
    let args: Vec<String> = args.iter().map(|value| (*value).into()).collect();
    tokio::task::spawn_blocking(move || {
        let mut command = Command::new(assert_cmd::cargo::cargo_bin!("dkc"));
        for (name, _) in std::env::vars_os() {
            if name.to_string_lossy().starts_with("DKC_") {
                command.env_remove(name);
            }
        }
        let assertion = command
            .current_dir(&root)
            .env("XDG_CONFIG_HOME", root.join("unused-user-config"))
            .env("DKC_MODELS_DIR", root.join("unused-models"))
            .timeout(Duration::from_secs(20))
            .args(args)
            .assert();
        let assertion = if success {
            assertion.success()
        } else {
            assertion.failure()
        };
        serde_json::from_slice(&assertion.get_output().stdout).expect("public drain JSON")
    })
    .await
    .unwrap()
}

#[test]
fn drain_refuses_a_missing_index_without_creating_it() -> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root.path())
        .env("XDG_CONFIG_HOME", root.path().join("unused-user-config"))
        .args(["queue", "drain", "--json"])
        .assert()
        .failure();
    assert!(!root.path().join(".deepkit").exists());
    Ok(())
}

#[tokio::test]
async fn empty_drain_does_not_prepare_models_or_schedule_unembedded_units()
-> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    fs::write(root.path().join("a.rs"), "pub fn not_queued() {}\n")?;
    cli(root.path(), &["index", "--level", "1", "--json"], true).await;
    let report = cli(root.path(), &["queue", "drain", "--json"], true).await;
    assert_eq!(report["state"], "quiescent");
    assert_eq!(report["providers"], json!([]));
    let db = Connection::open(root.path().join(".deepkit/code.db"))?;
    assert!(count(&db, "SELECT count(*) FROM units") > 0);
    assert_eq!(count(&db, "SELECT count(*) FROM jobs"), 0);
    assert_eq!(count(&db, "SELECT count(*) FROM embedding_tables"), 0);
    assert!(!root.path().join("unused-models").exists());
    Ok(())
}

fn config(root: &Path, url: &str, zero: bool) {
    let budget = if zero {
        format!("\n[[queue.budgets]]\nprofile='ollama'\nurl='{url}'\ndaily_calls=0\n")
    } else {
        String::new()
    };
    fs::write(root.join(".deepkit/config.toml"), format!(
        "[embedder]\nprofile='ollama'\ndims=2\n[embedder.ollama]\nurl='{url}'\n[queue]\nmax_schedule_per_pass=1000\nmax_jobs_per_pass=1\nembedding_timeout='2s'\n{budget}"
    )).unwrap();
}
async fn provider(fail_second: bool) -> MockServer {
    let server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(move |request: &wiremock::Request| {
            let call = calls.fetch_add(1, Ordering::SeqCst);
            if fail_second && call > 0 {
                return ResponseTemplate::new(401);
            }
            let input: Value = serde_json::from_slice(&request.body).unwrap();
            ResponseTemplate::new(200).set_body_json(
                json!({"embeddings": vec![vec![1.0,0.0];input["input"].as_array().unwrap().len()]}),
            )
        })
        .mount(&server)
        .await;
    server
}
async fn seeded(root: &Path, server: &MockServer) -> Connection {
    fs::create_dir_all(root.join(".deepkit")).unwrap();
    for (file, symbol) in [("a.rs", "alpha"), ("b.rs", "bravo"), ("c.rs", "charlie")] {
        fs::write(
            root.join(file),
            format!("pub fn {symbol}() -> bool {{ true }}\n"),
        )
        .unwrap();
    }
    config(root, &server.uri(), true);
    let indexed = cli(root, &["index", "--level", "2", "--json"], true).await;
    assert_eq!(indexed["embedding"]["state"], "pending", "{indexed}");
    assert!(server.received_requests().await.unwrap().is_empty());
    let db = Connection::open(root.join(".deepkit/code.db")).unwrap();
    assert!(count(&db, "SELECT count(*) FROM jobs") >= 3);
    db
}
fn count(db: &Connection, sql: &str) -> i64 {
    db.query_row(sql, [], |row| row.get(0)).unwrap()
}
fn ready(root: &Path, server: &MockServer, db: &Connection) {
    config(root, &server.uri(), false);
    // Explicitly advance fixture eligibility instead of waiting for UTC midnight.
    // This is test preparation; drain itself must never clear a recorded delay.
    db.execute("UPDATE jobs SET not_before=0 WHERE state='queued'", [])
        .unwrap();
}
fn no_scheduling(db: &Connection) {
    db.execute_batch("CREATE TRIGGER no_new_jobs BEFORE INSERT ON jobs BEGIN SELECT RAISE(ABORT,'drain scheduled new work'); END;").unwrap();
}

#[tokio::test]
async fn drain_consumes_only_persisted_jobs_and_honors_each_pass_bound()
-> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let server = provider(false).await;
    let db = seeded(root.path(), &server).await;
    ready(root.path(), &server, &db);
    db.execute("DELETE FROM jobs WHERE id=(SELECT max(id) FROM jobs)", [])?;
    let total = count(&db, "SELECT count(*) FROM jobs");
    let units = count(&db, "SELECT count(*) FROM units");
    no_scheduling(&db);
    let first = cli(
        root.path(),
        &["queue", "drain", "--max-passes", "1", "--json"],
        true,
    )
    .await;
    assert_eq!(first["state"], "pending", "{first}");
    assert_eq!(first["providers"][0]["published"], 1);
    assert_eq!(first["providers"][0]["stopped"], "pass_limit");
    assert!(first["queue"].is_object());
    assert_eq!(
        count(&db, "SELECT count(*) FROM jobs WHERE state='done'"),
        1
    );
    assert_eq!(
        count(&db, "SELECT count(*) FROM jobs WHERE state='queued'"),
        total - 1
    );
    let final_report = cli(
        root.path(),
        &["queue", "drain", "--max-passes", "10", "--json"],
        true,
    )
    .await;
    assert_eq!(final_report["state"], "quiescent", "{final_report}");
    assert_eq!(count(&db, "SELECT count(*) FROM jobs"), total);
    assert_eq!(
        count(&db, "SELECT count(*) FROM jobs WHERE state='done'"),
        total
    );
    assert_eq!(count(&db, "SELECT count(*) FROM units"), units);
    assert_eq!(
        server.received_requests().await.unwrap().len(),
        usize::try_from(total)?
    );
    assert!(!root.path().join("unused-models").exists());
    Ok(())
}

#[tokio::test]
async fn drain_preserves_recorded_budget_delay_without_dispatch() -> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let server = provider(false).await;
    let db = seeded(root.path(), &server).await;
    // Freeze every target's fixture delay so no unvisited ready target affects
    // the preservation assertion. Raising the budget does not authorize retry.
    db.execute(
        "UPDATE jobs SET not_before=253402214400 WHERE state='queued'",
        [],
    )?;
    config(root.path(), &server.uri(), false);
    let before: Vec<(i64, i64, i64)> = db
        .prepare("SELECT id,not_before,attempts FROM jobs ORDER BY id")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<Result<_, _>>()?;
    no_scheduling(&db);
    let report = cli(root.path(), &["queue", "drain", "--json"], true).await;
    assert_eq!(report["state"], "pending", "{report}");
    let after: Vec<(i64, i64, i64)> = db
        .prepare("SELECT id,not_before,attempts FROM jobs ORDER BY id")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<Result<_, _>>()?;
    assert_eq!(after, before);
    assert!(server.received_requests().await.unwrap().is_empty());
    assert_eq!(
        count(
            &db,
            "SELECT count(*) FROM usage_reservations WHERE state='dispatched'"
        ),
        0
    );
    Ok(())
}

#[tokio::test]
async fn drain_rejects_unpublished_source_authority_without_dispatch_or_indexing()
-> Result<(), Box<dyn Error>> {
    for field in ["head", "l1", "graph"] {
        let root = tempfile::tempdir()?;
        let server = provider(false).await;
        let db = seeded(root.path(), &server).await;
        ready(root.path(), &server, &db);
        match field {
            "head" => {
                db.execute("UPDATE meta SET value='ffffffffffffffffffffffffffffffffffffffff' WHERE key='indexed_head'", [])?;
            }
            "l1" => {
                db.execute("UPDATE files SET l1_pending=1", [])?;
            }
            "graph" => {
                db.execute("UPDATE meta SET value='1' WHERE key='graph_pending'", [])?;
            }
            _ => unreachable!(),
        }
        no_scheduling(&db);
        let total = count(&db, "SELECT count(*) FROM jobs WHERE state='queued'");
        let report = cli(root.path(), &["queue", "drain", "--json"], false).await;
        assert_eq!(report["state"], "failed", "{field}: {report}");
        assert!(report["error"].is_string());
        assert_eq!(
            count(&db, "SELECT count(*) FROM jobs WHERE state='queued'"),
            total
        );
        assert!(server.received_requests().await.unwrap().is_empty());
    }
    Ok(())
}

#[tokio::test]
async fn drain_provider_failure_retains_committed_prefix_and_structured_report()
-> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let server = provider(true).await;
    let db = seeded(root.path(), &server).await;
    ready(root.path(), &server, &db);
    no_scheduling(&db);
    let report = cli(
        root.path(),
        &["queue", "drain", "--max-passes", "3", "--json"],
        false,
    )
    .await;
    assert_eq!(report["state"], "failed", "{report}");
    assert!(report["error"].is_string());
    assert_eq!(report["providers"][0]["published"], 1);
    assert!(report["queue"].is_object());
    assert_eq!(
        count(&db, "SELECT count(*) FROM jobs WHERE state='done'"),
        1
    );
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
    assert_eq!(
        count(
            &db,
            "SELECT sum(calls) FROM (SELECT calls FROM usage UNION ALL SELECT calls FROM usage_reservations WHERE state='dispatched')"
        ),
        2
    );
    Ok(())
}

#[tokio::test]
async fn drain_rechecks_changed_source_and_current_privacy_before_http()
-> Result<(), Box<dyn Error>> {
    for changed in ["source", "policy"] {
        let root = tempfile::tempdir()?;
        let server = provider(false).await;
        let db = seeded(root.path(), &server).await;
        ready(root.path(), &server, &db);
        if changed == "source" {
            for file in ["a.rs", "b.rs", "c.rs"] {
                fs::write(root.path().join(file), "pub fn changed_after_index() {}\n")?;
            }
        } else {
            let file = root.path().join(".deepkit/config.toml");
            let text = fs::read_to_string(&file)?;
            fs::write(
                file,
                format!(
                    "{text}\n[[treatments]]\nname='protected'\nmatch=['*.rs']\nmax_level=3\nnever_send=true\n"
                ),
            )?;
        }
        no_scheduling(&db);
        let directory = root.path().to_owned();
        // The CLI may reject the changed authority at preflight or the worker
        // may retire stale work. Both must produce JSON without sending content.
        let report: Value = tokio::task::spawn_blocking(move || {
            let mut command = Command::new(assert_cmd::cargo::cargo_bin!("dkc"));
            for (name, _) in std::env::vars_os() {
                if name.to_string_lossy().starts_with("DKC_") {
                    command.env_remove(name);
                }
            }
            let output = command
                .current_dir(&directory)
                .env("XDG_CONFIG_HOME", directory.join("unused-user-config"))
                .env("DKC_MODELS_DIR", directory.join("unused-models"))
                .timeout(Duration::from_secs(20))
                .args(["queue", "drain", "--max-passes", "10", "--json"])
                .output()
                .unwrap();
            serde_json::from_slice(&output.stdout).unwrap()
        })
        .await?;
        assert!(report["state"].is_string(), "{changed}: {report}");
        assert!(server.received_requests().await.unwrap().is_empty());
        assert_eq!(
            count(
                &db,
                "SELECT count(*) FROM jobs WHERE state='done' AND last_error IS NULL"
            ),
            0
        );
        assert_eq!(
            count(&db, "SELECT count(*) FROM units WHERE ladder_level>=2"),
            0
        );
        assert_eq!(
            count(
                &db,
                "SELECT count(*) FROM usage_reservations WHERE state IN ('dispatched','reconciled')"
            ),
            0
        );
    }
    Ok(())
}

#[tokio::test]
async fn drain_reports_storage_error_category_without_echoing_trigger_source_text() {
    let root = tempfile::tempdir().unwrap();
    let server = provider(false).await;
    let db = seeded(root.path(), &server).await;
    ready(root.path(), &server, &db);
    no_scheduling(&db);
    db.execute_batch(
        "CREATE TRIGGER reject_fixture_publication BEFORE UPDATE OF ladder_level ON units
        WHEN NEW.ladder_level>=2 BEGIN SELECT RAISE(ABORT,'synthetic protected source text'); END;",
    )
    .unwrap();
    let report = cli(root.path(), &["queue", "drain", "--json"], false).await;
    assert_eq!(report["state"], "failed", "{report}");
    assert_eq!(
        report["providers"][0]["error_codes"]["sqlite:ConstraintViolation:1811"], 1,
        "{report}"
    );
    assert!(
        !report
            .to_string()
            .contains("synthetic protected source text")
    );
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
    assert_eq!(
        count(&db, "SELECT count(*) FROM units WHERE ladder_level>=2"),
        0
    );
}

#[tokio::test]
async fn entirely_failed_jobs_preserve_future_delay_then_drain_due_without_scheduling()
-> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let server = provider(false).await;
    let db = seeded(root.path(), &server).await;
    ready(root.path(), &server, &db);
    db.execute("UPDATE jobs SET state='failed',attempts=1,not_before=253402214400,last_error='transient provider failure'", [])?;
    let total = count(&db, "SELECT count(*) FROM jobs");
    no_scheduling(&db);
    let waiting = cli(root.path(), &["queue", "drain", "--json"], true).await;
    assert_eq!(waiting["state"], "pending", "{waiting}");
    assert_eq!(
        count(
            &db,
            "SELECT count(*) FROM jobs WHERE state='failed' AND attempts=1 AND not_before=253402214400"
        ),
        total
    );
    assert!(server.received_requests().await.unwrap().is_empty());
    // Advance fixture time eligibility, without invoking public retry or changing
    // the desired generation. No queued job exists when drain starts.
    db.execute("UPDATE jobs SET not_before=0 WHERE state='failed'", [])?;
    assert_eq!(
        count(&db, "SELECT count(*) FROM jobs WHERE state='queued'"),
        0
    );
    let drained = cli(
        root.path(),
        &["queue", "drain", "--max-passes", "10", "--json"],
        true,
    )
    .await;
    assert_eq!(drained["state"], "quiescent", "{drained}");
    assert_eq!(count(&db, "SELECT count(*) FROM jobs"), total);
    assert_eq!(
        count(
            &db,
            "SELECT count(*) FROM jobs WHERE state='done' AND attempts=2 AND last_error IS NULL"
        ),
        total
    );
    assert_eq!(
        server.received_requests().await.unwrap().len(),
        usize::try_from(total)?
    );
    assert_eq!(drained["providers"][0]["published"], total);
    Ok(())
}

#[cfg(unix)]
#[tokio::test]
async fn cancelled_second_drain_pass_reports_first_publication_and_uncertain_charge()
-> Result<(), Box<dyn Error>> {
    use std::process::Stdio;
    let root = tempfile::tempdir()?;
    let server = provider(false).await;
    let db = seeded(root.path(), &server).await;
    ready(root.path(), &server, &db);
    let config_path = root.path().join(".deepkit/config.toml");
    fs::write(
        &config_path,
        fs::read_to_string(&config_path)?.replace("'2s'", "'60s'"),
    )?;
    no_scheduling(&db);
    server.reset().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(move |request: &wiremock::Request| {
            let ordinal = observed.fetch_add(1, Ordering::SeqCst) + 1;
            let input: Value = serde_json::from_slice(&request.body).unwrap();
            let response = ResponseTemplate::new(200).set_body_json(json!({
                "embeddings": vec![vec![1.0,0.0]; input["input"].as_array().unwrap().len()]
            }));
            if ordinal == 2 {
                response.set_delay(Duration::from_secs(30))
            } else {
                response
            }
        })
        .mount(&server)
        .await;
    let mut command = tokio::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"));
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().starts_with("DKC_") {
            command.env_remove(name);
        }
    }
    let mut child = command
        .current_dir(root.path())
        .env("XDG_CONFIG_HOME", root.path().join("unused-user-config"))
        .env("DKC_MODELS_DIR", root.path().join("unused-models"))
        .args(["queue", "drain", "--max-passes", "10", "--json"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()?;
    tokio::time::timeout(Duration::from_secs(10), async {
        while calls.load(Ordering::SeqCst) < 2 {
            assert!(
                child.try_wait().unwrap().is_none(),
                "drain exited before second request"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    assert_eq!(
        count(
            &db,
            "SELECT count(*) FROM jobs WHERE state='done' AND last_error IS NULL"
        ),
        1
    );
    let pid = child.id().ok_or("missing drain PID")?;
    assert!(
        std::process::Command::new("kill")
            .args(["-2", &pid.to_string()])
            .status()?
            .success()
    );
    let output = tokio::time::timeout(Duration::from_secs(10), child.wait_with_output()).await??;
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(report["state"], "cancelled", "{report}");
    assert_eq!(report["providers"][0]["published"], 1, "{report}");
    assert!(report["error"].is_string());
    assert!(report["queue"].is_object());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        count(
            &db,
            "SELECT count(*) FROM jobs WHERE state='done' AND last_error IS NULL"
        ),
        1
    );
    assert_eq!(
        count(&db, "SELECT count(*) FROM jobs WHERE state='leased'"),
        1
    );
    assert_eq!(
        count(
            &db,
            "SELECT count(*) FROM usage_reservations WHERE state='dispatched'"
        ),
        1
    );
    assert_eq!(
        count(
            &db,
            "SELECT sum(calls) FROM (SELECT calls FROM usage UNION ALL SELECT calls FROM usage_reservations WHERE state='dispatched')"
        ),
        2
    );
    Ok(())
}
