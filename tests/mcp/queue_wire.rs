//! Real queue HTTP retry/rate waits, colocated with process tests; v1 §8, T1.5.
use assert_cmd::Command;
use rusqlite::Connection;
use serde_json::{Value, json};
use std::{fs, path::Path, sync::Arc, time::Duration};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

// These two small SQL helpers are local so the wire tests do not import the
// queue_drain integration binary or duplicate its Ollama/CLI fixture.
fn count(db: &Connection, sql: &str) -> i64 {
    db.query_row(sql, [], |row| row.get(0)).unwrap()
}
fn no_scheduling(db: &Connection) {
    db.execute_batch("CREATE TRIGGER no_new_jobs BEFORE INSERT ON jobs BEGIN SELECT RAISE(ABORT,'drain scheduled new work'); END;").unwrap();
}

// These exercise real loopback HTTP and real
// wall-clock waits; no fake clock or retry-deadline mutation after first dispatch.
async fn wire_cli(root: &Path, args: &[&str], success: bool) -> Value {
    let root = root.to_owned();
    let args: Vec<String> = args.iter().map(|v| (*v).into()).collect();
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
            .env("QUEUE_WIRE_SYNTHETIC_KEY", "fixture-only")
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
fn wire_config(root: &Path, server: &MockServer, rpm: u64, calls: u64) {
    fs::write(root.join(".deepkit/config.toml"), format!(
        "[embedder]\nprofile='remote'\ndims=2\n[embedder.remote]\nurl='{}'\nmodel='wire-fixture'\napi_key_env='QUEUE_WIRE_SYNTHETIC_KEY'\nrpm={rpm}\ntpm=1000000\nmax_batch_items=1\n[queue]\nmax_jobs_per_pass=1\nembedding_timeout='2s'\n[[queue.budgets]]\nprofile='remote'\nurl='{}'\ndaily_calls={calls}\ndaily_tokens=1000000\n",
        server.uri(), server.uri(),
    )).unwrap();
}
async fn wire_seed(root: &Path, server: &MockServer, rpm: u64, jobs: i64) -> Connection {
    fs::create_dir(root.join(".deepkit")).unwrap();
    fs::write(root.join("a.rs"), "pub fn alpha() {}\n").unwrap();
    fs::write(root.join("b.rs"), "pub fn bravo() {}\n").unwrap();
    wire_config(root, server, rpm, 0);
    wire_cli(root, &["index", "--level", "2", "--json"], true).await;
    assert!(server.received_requests().await.unwrap().is_empty());
    let db = Connection::open(root.join(".deepkit/code.db")).unwrap();
    // Fixture construction only: preserve the requested number of valid seeded
    // jobs and remove midnight eligibility introduced by the zero setup budget.
    db.execute(
        "DELETE FROM jobs WHERE id NOT IN (SELECT id FROM jobs ORDER BY id LIMIT ?1)",
        [jobs],
    )
    .unwrap();
    db.execute("UPDATE jobs SET not_before=0 WHERE state='queued'", [])
        .unwrap();
    assert_eq!(count(&db, "SELECT count(*) FROM jobs"), jobs);
    assert_eq!(count(&db, "SELECT sum(attempts) FROM jobs"), 0);
    wire_config(root, server, rpm, 100);
    no_scheduling(&db);
    db
}
fn wire_seconds() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}
async fn wait_for_recorded_deadline(db: &Connection, until: i64, overall: std::time::Instant) {
    assert!(until <= wire_seconds() + 65, "unexpected retry delay");
    while wire_seconds() < until {
        assert!(overall.elapsed() < Duration::from_secs(420));
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    // No mutation of eligibility, attempts, rate windows or reservations occurs.
    assert_eq!(
        count(db, "SELECT min(not_before) FROM jobs WHERE state!='done'"),
        until
    );
}

#[tokio::test]
async fn persistent_transient_wire_failure_stops_after_eight_actual_posts() {
    let server = MockServer::start().await;
    let arrivals = Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed = Arc::clone(&arrivals);
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(move |_request: &wiremock::Request| {
            observed
                .lock()
                .unwrap()
                .push((std::time::Instant::now(), wire_seconds()));
            ResponseTemplate::new(503).insert_header("Retry-After", "2")
        })
        .mount(&server)
        .await;
    let root = tempfile::tempdir().unwrap();
    let db = wire_seed(root.path(), &server, 100, 1).await;
    let overall = std::time::Instant::now();
    for attempt in 1..=8_i64 {
        wire_cli(root.path(), &["queue", "drain", "--json"], attempt < 8).await;
        assert_eq!(
            server.received_requests().await.unwrap().len(),
            attempt as usize
        );
        let (state, attempts, due): (String, i64, i64) = db
            .query_row("SELECT state,attempts,not_before FROM jobs", [], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })
            .unwrap();
        assert_eq!(attempts, attempt);
        if attempt < 8 {
            assert_eq!(state, "failed");
            let received_second = arrivals.lock().unwrap().last().unwrap().1;
            assert!(
                due >= received_second + 2,
                "Retry-After must remain a lower bound in persisted UTC seconds"
            );
            wait_for_recorded_deadline(&db, due, overall).await;
        } else {
            assert_eq!(state, "dead");
        }
    }
    for _ in 0..2 {
        let pending = wire_cli(
            root.path(),
            &["queue", "drain", "--max-passes", "10", "--json"],
            true,
        )
        .await;
        assert_eq!(pending["state"], "pending");
        assert_eq!(pending["queue"]["jobs"]["dead"], 1);
    }
    assert_eq!(server.received_requests().await.unwrap().len(), 8);
    assert_eq!(count(&db, "SELECT sum(attempts) FROM jobs"), 8);
    assert_eq!(
        count(
            &db,
            "SELECT count(*) FROM usage_reservations WHERE state='dispatched'"
        ),
        8
    );
    assert_eq!(
        count(
            &db,
            "SELECT sum(calls) FROM usage_reservations WHERE state='dispatched'"
        ),
        8
    );
    assert_eq!(
        count(
            &db,
            "SELECT count(*) FROM usage_reservations WHERE state IN ('reserved','reconciled')"
        ),
        0
    );
    assert_eq!(count(&db, "SELECT count(*) FROM embedding_publications"), 0);
    let stamps = arrivals.lock().unwrap();
    assert_eq!(stamps.len(), 8);
    assert!(stamps.windows(2).all(|pair| pair[1].1 >= pair[0].1 + 2));
    eprintln!(
        "actual eight-post retry fixture: elapsed_seconds={:.3}, arrival_elapsed_seconds={:?}",
        overall.elapsed().as_secs_f64(),
        stamps
            .iter()
            .map(|x| x.0.duration_since(stamps[0].0).as_secs_f64())
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn rpm_one_blocks_actual_posts_across_processes_then_reopens_after_real_minute() {
    let server = MockServer::start().await;
    let arrivals = Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed = Arc::clone(&arrivals);
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(move |request: &wiremock::Request| {
            let body: Value = request.body_json().unwrap();
            assert_eq!(body["input"].as_array().unwrap().len(), 1);
            observed.lock().unwrap().push(std::time::Instant::now());
            ResponseTemplate::new(200)
                .set_body_json(json!({"data":[{"index":0,"embedding":"AACAPwAAAAA="}]}))
        })
        .mount(&server)
        .await;
    let root = tempfile::tempdir().unwrap();
    let db = wire_seed(root.path(), &server, 1, 2).await;
    wire_cli(root.path(), &["queue", "drain", "--json"], true).await;
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
    let first = arrivals.lock().unwrap()[0];
    // Another process attempts admission immediately and again mid-window.
    wire_cli(root.path(), &["queue", "drain", "--json"], true).await;
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
    assert_eq!(count(&db, "SELECT sum(attempts) FROM jobs"), 1);
    tokio::time::sleep_until(tokio::time::Instant::from_std(
        first + Duration::from_secs(30),
    ))
    .await;
    wire_cli(root.path(), &["queue", "drain", "--json"], true).await;
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
    assert_eq!(count(&db, "SELECT sum(attempts) FROM jobs"), 1);
    tokio::time::sleep_until(tokio::time::Instant::from_std(
        first + Duration::from_secs(61),
    ))
    .await;
    let done = wire_cli(root.path(), &["queue", "drain", "--json"], true).await;
    assert_eq!(done["state"], "quiescent");
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
    assert_eq!(
        count(
            &db,
            "SELECT count(*) FROM jobs WHERE state='done' AND attempts=1"
        ),
        2
    );
    assert_eq!(count(&db, "SELECT count(*) FROM embedding_publications"), 2);
    let stamps = arrivals.lock().unwrap();
    assert!(stamps[1].duration_since(stamps[0]) >= Duration::from_secs(60));
    eprintln!(
        "actual RPM1 wire separation_seconds={:.3}",
        stamps[1].duration_since(stamps[0]).as_secs_f64()
    );
}
