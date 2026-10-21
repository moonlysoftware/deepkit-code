//! Real child-process publication boundary tests for v1 §4.2–4.3 / T1.R1b.

use rusqlite::Connection;
use serde_json::{Value, json};
use std::{
    env,
    error::Error,
    fs,
    path::Path,
    process::{Child, Command},
    thread,
    time::{Duration, Instant},
};

use crate::store::{PublicationPoint, observe_publication};

struct IndexChild(Child);
impl Drop for IndexChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn observer(point: PublicationPoint) {
    let selected = env::var("DKC_TEST_L1_POINT").unwrap_or_default();
    let matches = matches!(
        (selected.as_str(), point),
        ("unit", PublicationPoint::OccurrenceWritten)
            | ("completion", PublicationPoint::BeforeCompletion)
            | ("rank", PublicationPoint::BeforeGraphRank)
    );
    let ready = Path::new(".deepkit/test-ready");
    if matches && !ready.exists() {
        fs::write(ready, "inside publication transaction").expect("write test marker");
        while !Path::new(".deepkit/test-release").exists() {
            thread::sleep(Duration::from_millis(5));
        }
    }
}

#[test]
fn child_index_publication() -> Result<(), Box<dyn Error>> {
    let Some(root) = env::var_os("DKC_TEST_L1_ROOT") else {
        return Ok(());
    };
    env::set_current_dir(root)?; // isolated child process, never the parallel parent test runner
    if env::var_os("DKC_TEST_L1_FRESH").is_some() {
        super::clean(true).map_err(|e| std::io::Error::other(e.to_string()))?;
    }
    if env::var_os("DKC_TEST_L1_POINT").is_some() {
        observe_publication(observer);
    }
    super::index(
        false,
        super::IndexQueue {
            disabled: false,
            refresh_context: false,
        },
        false,
        false,
        false,
        1,
        true,
    )
    .map_err(|e| std::io::Error::other(e.to_string()))?;
    Ok(())
}

fn spawn(root: &Path, point: Option<&str>, fresh: bool) -> Result<IndexChild, Box<dyn Error>> {
    for name in ["test-ready", "test-release"] {
        let _ = fs::remove_file(root.join(".deepkit").join(name));
    }
    let log = fs::File::create(root.join(".deepkit/test-child.log"))?;
    let mut command = Command::new(env::current_exe()?);
    command
        .args([
            "--exact",
            "cli::publication_tests::child_index_publication",
            "--nocapture",
        ])
        // These boundary fixtures assert file-only publication counts.
        .env("DKC_COMMITS__ENABLED", "false")
        .env("DKC_TEST_L1_ROOT", root)
        .env_remove("DKC_TEST_L1_POINT")
        .env_remove("DKC_TEST_L1_FRESH")
        .stdout(log.try_clone()?)
        .stderr(log);
    if let Some(point) = point {
        command.env("DKC_TEST_L1_POINT", point);
    }
    if fresh {
        command.env("DKC_TEST_L1_FRESH", "1");
    }
    Ok(IndexChild(command.spawn()?))
}

fn run(root: &Path, fresh: bool) -> Result<(), Box<dyn Error>> {
    assert!(
        spawn(root, None, fresh)?.0.wait()?.success(),
        "{}",
        fs::read_to_string(root.join(".deepkit/test-child.log"))?
    );
    Ok(())
}

fn wait_ready(root: &Path, child: &mut IndexChild) -> Result<(), Box<dyn Error>> {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !root.join(".deepkit/test-ready").exists() {
        assert!(
            Instant::now() < deadline,
            "publication boundary not observed"
        );
        assert!(
            child.0.try_wait()?.is_none(),
            "{}",
            fs::read_to_string(root.join(".deepkit/test-child.log"))?
        );
        thread::sleep(Duration::from_millis(5));
    }
    Ok(())
}

fn git(root: &Path, args: &[&str]) -> Result<String, Box<dyn Error>> {
    let result = Command::new("git").current_dir(root).args(args).output()?;
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(String::from_utf8(result.stdout)?.trim().to_owned())
}

fn fixture() -> Result<tempfile::TempDir, Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    fs::create_dir(root.path().join(".deepkit"))?;
    fs::write(
        root.path().join(".deepkit/config.local.toml"),
        "[index]\nrepo_name='publication-fixture'\n",
    )?;
    fs::write(root.path().join("a.rs"), "pub fn original() -> i32 { 1 }\n")?;
    fs::write(root.path().join("empty.rs"), "")?;
    fs::write(root.path().join("binary.bin"), [0_u8, 1, 2])?;
    git(root.path(), &["init", "--quiet"])?;
    git(root.path(), &["add", "a.rs", "empty.rs", "binary.bin"])?;
    git(
        root.path(),
        &[
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
    run(root.path(), false)?;
    Ok(root)
}

fn db(root: &Path) -> Result<Connection, rusqlite::Error> {
    Connection::open(root.join(".deepkit/code.db"))
}

fn snapshot(root: &Path) -> Result<Value, Box<dyn Error>> {
    let connection = db(root)?;
    let units = connection.prepare("SELECT f.path,hex(u.hash),u.line_start,u.line_end,c.text,u.header FROM units u JOIN files f ON f.id=u.file_id JOIN unit_contents c ON c.hash=u.content_hash ORDER BY f.path,u.line_start,u.hash")?
        .query_map([], |r| Ok(json!([r.get::<_, String>(0)?,r.get::<_, String>(1)?,r.get::<_, i64>(2)?,r.get::<_, i64>(3)?,r.get::<_, String>(4)?,r.get::<_, String>(5)?])))?.collect::<Result<Vec<_>,_>>()?;
    let symbols = connection.prepare("SELECT f.path,s.name,s.line_start,s.line_end FROM symbols s JOIN files f ON f.id=s.file_id ORDER BY f.path,s.line_start,s.name")?
        .query_map([], |r| Ok(json!([r.get::<_, String>(0)?,r.get::<_, String>(1)?,r.get::<_, i64>(2)?,r.get::<_, i64>(3)?])))?.collect::<Result<Vec<_>,_>>()?;
    let fts = connection.prepare("SELECT f.path,u.line_start FROM fts_units JOIN units u ON u.id=fts_units.rowid JOIN files f ON f.id=u.file_id WHERE fts_units MATCH 'original OR replacement OR concurrent' ORDER BY f.path,u.line_start")?
        .query_map([], |r| Ok(json!([r.get::<_, String>(0)?,r.get::<_, i64>(1)?])))?.collect::<Result<Vec<_>,_>>()?;
    Ok(json!({"units":units,"symbols":symbols,"fts":fts}))
}

fn assert_pending(root: &Path, head: &str) -> Result<(), Box<dyn Error>> {
    let connection = db(root)?;
    assert_eq!(
        connection.query_row("SELECT value FROM meta WHERE key='indexed_head'", [], |r| r
            .get::<_, String>(0))?,
        head
    );
    assert_eq!(
        connection.query_row("SELECT l1_pending FROM files WHERE path='a.rs'", [], |r| {
            r.get::<_, i64>(0)
        })?,
        1
    );
    Ok(())
}

#[test]
fn kill_during_replacement_and_before_completion_retains_snapshot_then_recovers()
-> Result<(), Box<dyn Error>> {
    for point in ["unit", "completion"] {
        let root = fixture()?;
        let before = snapshot(root.path())?;
        let head = git(root.path(), &["rev-parse", "HEAD"])?;
        fs::write(
            root.path().join("a.rs"),
            "pub fn replacement() -> i32 { 2 }\n",
        )?;
        let mut child = spawn(root.path(), Some(point), false)?;
        wait_ready(root.path(), &mut child)?;
        assert_eq!(snapshot(root.path())?, before); // readers see the whole old snapshot
        assert_pending(root.path(), &head)?;
        child.0.kill()?;
        assert!(!child.0.wait()?.success());
        assert_eq!(snapshot(root.path())?, before); // uncommitted rows and FTS rolled back
        run(root.path(), false)?;
        let recovered = snapshot(root.path())?;
        assert_ne!(recovered, before);
        assert_eq!(
            db(root.path())?.query_row("SELECT sum(l1_pending) FROM files", [], |r| r
                .get::<_, i64>(0))?,
            0
        );
        run(root.path(), true)?;
        assert_eq!(snapshot(root.path())?, recovered);
    }
    Ok(())
}

#[test]
fn source_or_config_change_inside_transaction_rejects_completion() -> Result<(), Box<dyn Error>> {
    for change in ["source", "config"] {
        let root = fixture()?;
        let before = snapshot(root.path())?;
        let head = git(root.path(), &["rev-parse", "HEAD"])?;
        fs::write(
            root.path().join("a.rs"),
            "pub fn replacement() -> i32 { 2 }\n",
        )?;
        let mut child = spawn(root.path(), Some("completion"), false)?;
        wait_ready(root.path(), &mut child)?;
        match change {
            "source" => fs::write(
                root.path().join("a.rs"),
                "pub fn concurrent() -> i32 { 3 }\n",
            )?,
            _ => fs::write(
                root.path().join(".deepkit/config.local.toml"),
                "[index]\nrepo_name='publication-fixture'\n[tokenize]\nsplit_subtokens=false\n",
            )?,
        }
        fs::write(root.path().join(".deepkit/test-release"), "continue")?;
        assert!(!child.0.wait()?.success());
        assert_eq!(snapshot(root.path())?, before);
        assert_pending(root.path(), &head)?;
        run(root.path(), false)?;
        let recovered = snapshot(root.path())?;
        run(root.path(), true)?;
        assert_eq!(snapshot(root.path())?, recovered);
    }
    Ok(())
}

#[test]
fn changed_head_is_not_published_as_a_completed_scan() -> Result<(), Box<dyn Error>> {
    let root = fixture()?;
    let old_head = git(root.path(), &["rev-parse", "HEAD"])?;
    fs::write(
        root.path().join("a.rs"),
        "pub fn replacement() -> i32 { 2 }\n",
    )?;
    let mut child = spawn(root.path(), Some("completion"), false)?;
    wait_ready(root.path(), &mut child)?;
    git(root.path(), &["add", "a.rs"])?;
    git(
        root.path(),
        &[
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "changed",
        ],
    )?;
    fs::write(root.path().join(".deepkit/test-release"), "continue")?;
    assert!(!child.0.wait()?.success());
    assert_eq!(
        db(root.path())?.query_row("SELECT value FROM meta WHERE key='indexed_head'", [], |r| r
            .get::<_, String>(0))?,
        old_head
    );
    run(root.path(), false)?;
    assert_eq!(
        db(root.path())?.query_row("SELECT value FROM meta WHERE key='indexed_head'", [], |r| r
            .get::<_, String>(0))?,
        git(root.path(), &["rev-parse", "HEAD"])?
    );
    Ok(())
}

#[test]
fn config_only_routing_reprocesses_affected_sources_and_refreshes_context()
-> Result<(), Box<dyn Error>> {
    let root = fixture()?;
    fs::write(
        root.path().join("b.rs"),
        "pub fn unaffected() -> i32 { 7 }\n",
    )?;
    run(root.path(), false)?;
    let connection = db(root.path())?;
    connection.execute("UPDATE units SET updated_at=123", [])?;
    let unchanged: Vec<u8> = connection.query_row(
        "SELECT l1_fingerprint FROM files WHERE path='b.rs'",
        [],
        |r| r.get(0),
    )?;
    drop(connection);
    fs::write(
        root.path().join(".deepkit/config.local.toml"),
        "[index]\nrepo_name='publication-fixture'\n[embedder.remote]\nmodel='different-provider-model'\n",
    )?;
    run(root.path(), false)?;
    assert_eq!(
        db(root.path())?.query_row("SELECT count(*) FROM units WHERE updated_at=123", [], |r| r
            .get::<_, i64>(0))?,
        2
    );
    fs::write(
        root.path().join(".deepkit/config.local.toml"),
        "[index]\nrepo_name='publication-fixture'\n[[treatments]]\nname='line-source'\nmatch=['a.rs']\nchunker='lines'\nmax_level=1\n",
    )?;
    run(root.path(), false)?;
    let connection = db(root.path())?;
    assert_eq!(
        connection.query_row(
            "SELECT u.kind FROM units u JOIN files f ON f.id=u.file_id WHERE f.path='a.rs'",
            [],
            |r| r.get::<_, String>(0)
        )?,
        "chunk"
    );
    assert_eq!(
        connection.query_row(
            "SELECT u.updated_at FROM units u JOIN files f ON f.id=u.file_id WHERE f.path='b.rs'",
            [],
            |r| r.get::<_, i64>(0)
        )?,
        123
    );
    assert_eq!(
        connection.query_row(
            "SELECT l1_fingerprint FROM files WHERE path='b.rs'",
            [],
            |r| r.get::<_, Vec<u8>>(0)
        )?,
        unchanged
    );
    drop(connection);
    fs::write(
        root.path().join(".deepkit/config.local.toml"),
        "[index]\nrepo_name='renamedcontext'\n",
    )?;
    run(root.path(), false)?;
    assert_eq!(
        db(root.path())?.query_row(
            "SELECT count(*) FROM fts_units WHERE fts_units MATCH 'header:renamedcontext'",
            [],
            |r| r.get::<_, i64>(0)
        )?,
        2
    );
    let recovered = snapshot(root.path())?;
    run(root.path(), true)?;
    assert_eq!(snapshot(root.path())?, recovered);
    Ok(())
}

#[test]
fn raising_source_cap_promotes_unchanged_tracked_l0_file() -> Result<(), Box<dyn Error>> {
    let root = fixture()?;
    // Start a fresh catalogue under a cap smaller than the tracked source.
    fs::write(
        root.path().join(".deepkit/config.local.toml"),
        "[index]\nrepo_name='publication-fixture'\n[limits]\nmax_file_bytes='8 B'\n",
    )?;
    run(root.path(), true)?;
    assert_eq!(
        db(root.path())?.query_row("SELECT count(*) FROM units", [], |r| r.get::<_, i64>(0))?,
        0
    );
    assert_eq!(
        db(root.path())?.query_row("SELECT sum(l1_pending) FROM files", [], |r| r
            .get::<_, i64>(0))?,
        0
    );
    fs::write(
        root.path().join(".deepkit/config.local.toml"),
        "[index]\nrepo_name='publication-fixture'\n",
    )?;
    run(root.path(), false)?;
    assert_eq!(
        db(root.path())?.query_row("SELECT count(*) FROM units", [], |r| r.get::<_, i64>(0))?,
        1
    );
    let promoted = snapshot(root.path())?;
    run(root.path(), true)?;
    assert_eq!(snapshot(root.path())?, promoted);
    Ok(())
}

#[test]
fn tighter_caps_retain_prior_units_and_vectors_without_endless_retry() -> Result<(), Box<dyn Error>>
{
    let root = fixture()?;
    fs::write(
        root.path().join("a.rs"),
        "pub fn original() -> i32 { 1 }\npub fn second() -> i32 { 2 }\n",
    )?;
    run(root.path(), false)?;
    db(root.path())?.execute_batch("CREATE TABLE vec_retained(unit_id INTEGER PRIMARY KEY REFERENCES units(id) ON DELETE CASCADE,v BLOB NOT NULL,ctx_hash BLOB NOT NULL);
        INSERT INTO embedding_tables VALUES ('retained','fixture','local',2,0,1);
        INSERT INTO vec_retained SELECT id,X'003c0000',hash FROM units;")?;
    let before = snapshot(root.path())?;
    for (setting, reason) in [
        ("max_units_per_file=1", "max_units_per_file"),
        ("max_file_bytes='8 B'", "max_file_bytes"),
    ] {
        fs::write(
            root.path().join(".deepkit/config.local.toml"),
            format!("[index]\nrepo_name='publication-fixture'\n[limits]\n{setting}\n"),
        )?;
        run(root.path(), false)?;
        assert_eq!(snapshot(root.path())?, before);
        let connection = db(root.path())?;
        assert_eq!(
            connection.query_row("SELECT count(*) FROM vec_retained", [], |r| r
                .get::<_, i64>(0))?,
            2
        );
        assert_eq!(
            connection.query_row(
                "SELECT l1_deferred_reason FROM files WHERE path='a.rs'",
                [],
                |r| r.get::<_, String>(0)
            )?,
            reason
        );
        assert_eq!(
            connection.query_row("SELECT sum(l1_pending) FROM files", [], |r| r
                .get::<_, i64>(0))?,
            0
        );
        assert!(connection.query_row(
            "SELECT l1_fingerprint IS NOT l1_desired FROM files WHERE path='a.rs'",
            [],
            |r| r.get::<_, bool>(0)
        )?);
        drop(connection);
        let store = crate::store::open(
            root.path().join(".deepkit/code.db"),
            crate::store::StoreLimits::default(),
        )?;
        assert_eq!(super::deferred_l1_json(&store)?[0]["reason"], reason);
        drop(store);
        run(root.path(), false)?;
        assert_eq!(snapshot(root.path())?, before);
        fs::write(
            root.path().join(".deepkit/config.local.toml"),
            "[index]\nrepo_name='publication-fixture'\n",
        )?;
        run(root.path(), false)?;
        assert_eq!(snapshot(root.path())?, before);
        assert_eq!(
            db(root.path())?.query_row(
                "SELECT count(*) FROM files WHERE l1_deferred_reason IS NOT NULL",
                [],
                |r| r.get::<_, i64>(0)
            )?,
            0
        );
        assert_eq!(
            db(root.path())?.query_row("SELECT count(*) FROM vec_retained", [], |r| r
                .get::<_, i64>(0))?,
            2
        );
    }
    Ok(())
}

#[test]
fn a_new_leading_chunk_at_capacity_cannot_evict_an_unchanged_occurrence()
-> Result<(), Box<dyn Error>> {
    let root = fixture()?;
    fs::write(
        root.path().join(".deepkit/config.local.toml"),
        "[index]\nrepo_name='publication-fixture'\n[limits]\nmax_units_per_file=1\n",
    )?;
    run(root.path(), false)?;
    let before = snapshot(root.path())?;
    fs::write(
        root.path().join("a.rs"),
        "pub fn replacement() -> i32 { 2 }\npub fn original() -> i32 { 1 }\n",
    )?;
    run(root.path(), false)?;
    assert_eq!(snapshot(root.path())?, before);
    assert_eq!(
        db(root.path())?.query_row(
            "SELECT l1_deferred_reason FROM files WHERE path='a.rs'",
            [],
            |r| r.get::<_, String>(0)
        )?,
        "max_units_per_file"
    );
    fs::write(
        root.path().join(".deepkit/config.local.toml"),
        "[index]\nrepo_name='publication-fixture'\n[limits]\nmax_units_per_file=2\n",
    )?;
    run(root.path(), false)?;
    assert_eq!(
        db(root.path())?.query_row("SELECT count(*) FROM units", [], |r| r.get::<_, i64>(0))?,
        2
    );
    let recovered = snapshot(root.path())?;
    run(root.path(), true)?;
    assert_eq!(snapshot(root.path())?, recovered);
    Ok(())
}

#[test]
fn config_only_chunk_sizes_and_tokenization_change_published_units() -> Result<(), Box<dyn Error>> {
    let root = fixture()?;
    let source = format!(
        "pub fn original() {{\n{}\n}}\n",
        "let item = readHTTPValue();\n".repeat(100)
    );
    let docs = format!(
        "# Guide\n\n{}",
        "A paragraph about application behaviour.\n\n".repeat(100)
    );
    fs::write(root.path().join("a.rs"), &source)?;
    fs::write(root.path().join("guide.md"), &docs)?;
    run(root.path(), false)?;
    let connection = db(root.path())?;
    let count = |path: &str| -> rusqlite::Result<i64> {
        connection.query_row(
            "SELECT count(*) FROM units u JOIN files f ON f.id=u.file_id WHERE f.path=?1",
            [path],
            |r| r.get(0),
        )
    };
    let before = (count("a.rs")?, count("guide.md")?);
    let http = || -> rusqlite::Result<i64> {
        connection.query_row(
            "SELECT count(*) FROM fts_units WHERE fts_units MATCH 'body:http'",
            [],
            |r| r.get(0),
        )
    };
    assert!(http()? > 0);
    fs::write(
        root.path().join(".deepkit/config.local.toml"),
        "[index]\nrepo_name='publication-fixture'\n[chunk]\nmin_tokens=5\ntarget_tokens=40\nmax_tokens=80\nmarkdown_range=[10,30]\n[tokenize]\nsplit_subtokens=false\n",
    )?;
    run(root.path(), false)?;
    assert!(count("a.rs")? > before.0);
    assert!(count("guide.md")? > before.1);
    assert_eq!(http()?, 0);
    let texts = connection.prepare("SELECT c.text FROM unit_contents c JOIN units u ON u.content_hash=c.hash JOIN files f ON f.id=u.file_id WHERE f.path='a.rs'")?
        .query_map([], |r| r.get::<_, String>(0))?.collect::<Result<Vec<_>, _>>()?;
    for text in texts {
        assert!(
            tiktoken_rs::cl100k_base_singleton()
                .encode_with_special_tokens(&text)
                .len()
                <= 80
        );
    }
    assert_eq!(fs::read_to_string(root.path().join("a.rs"))?, source);
    assert_eq!(fs::read_to_string(root.path().join("guide.md"))?, docs);
    let after = (count("a.rs")?, count("guide.md")?);
    run(root.path(), false)?;
    assert_eq!((count("a.rs")?, count("guide.md")?), after);
    fs::write(
        root.path().join(".deepkit/config.local.toml"),
        "[chunk]\nmax_tokens=0\n",
    )?;
    assert!(!spawn(root.path(), None, false)?.0.wait()?.success());
    assert_eq!((count("a.rs")?, count("guide.md")?), after);
    Ok(())
}

#[test]
fn noop_skips_rank_and_kill_after_l1_recovers_pending_graph() -> Result<(), Box<dyn Error>> {
    let root = fixture()?;
    let connection = db(root.path())?;
    connection.execute_batch("CREATE TRIGGER forbid_noop_rank BEFORE UPDATE ON meta WHEN NEW.key='graph_pending' BEGIN SELECT RAISE(ABORT, 'no-op touched graph rank'); END;")?;
    run(root.path(), false)?;
    connection.execute_batch("DROP TRIGGER forbid_noop_rank;")?;
    fs::write(
        root.path().join("a.rs"),
        "pub fn original() { target(); }\npub fn target() {}\n",
    )?;
    let mut child = spawn(root.path(), Some("rank"), false)?;
    wait_ready(root.path(), &mut child)?;
    assert_eq!(
        connection.query_row(
            "SELECT value FROM meta WHERE key='graph_pending'",
            [],
            |r| r.get::<_, String>(0)
        )?,
        "1"
    );
    assert_eq!(
        connection.query_row("SELECT count(*) FROM files WHERE l1_pending=1", [], |r| r
            .get::<_, i64>(
            0
        ))?,
        0
    );
    child.0.kill()?;
    assert!(!child.0.wait()?.success());
    run(root.path(), false)?;
    let ranks = |connection: &Connection| -> rusqlite::Result<Vec<(String, f64)>> {
        connection
            .prepare("SELECT name,pagerank FROM symbols ORDER BY name")?
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect()
    };
    let recovered = ranks(&connection)?;
    assert_eq!(
        connection.query_row(
            "SELECT value FROM meta WHERE key='graph_pending'",
            [],
            |r| r.get::<_, String>(0)
        )?,
        "0"
    );
    drop(connection);
    run(root.path(), true)?;
    assert_eq!(ranks(&db(root.path())?)?, recovered);
    fs::write(root.path().join("a.rs"), "// no symbols remain\n")?;
    run(root.path(), false)?;
    assert_eq!(
        db(root.path())?.query_row("SELECT rank FROM files WHERE path='a.rs'", [], |r| r
            .get::<_, f64>(0))?,
        0.0
    );
    Ok(())
}

#[test]
fn bounded_production_preserves_prior_occurrence_beyond_overflow_marker()
-> Result<(), Box<dyn Error>> {
    let root = fixture()?;
    let before = snapshot(root.path())?;
    fs::write(
        root.path().join(".deepkit/config.local.toml"),
        "[index]\nrepo_name='publication-fixture'\n[limits]\nmax_units_per_file=2\n",
    )?;
    let original = fs::read_to_string(root.path().join("a.rs"))?;
    let mut source = (0..20)
        .map(|i| format!("pub fn leading{i}() {{}}\n"))
        .collect::<String>();
    source.push_str(&original);
    fs::write(root.path().join("a.rs"), source)?;
    run(root.path(), false)?;
    let connection = db(root.path())?;
    assert_eq!(
        connection.query_row(
            "SELECT l1_deferred_reason FROM files WHERE path='a.rs'",
            [],
            |r| r.get::<_, String>(0)
        )?,
        "max_units_per_file"
    );
    assert_eq!(snapshot(root.path())?, before);
    Ok(())
}

#[test]
fn shared_index_errors_preserve_structured_capacity_diagnoses() {
    use crate::{store, sync};
    for (failure, expected) in [
        (
            sync::index::Error::Store(store::Error::DatabaseGrowthLimit),
            "database_limit",
        ),
        (
            sync::index::Error::Store(store::Error::WalCapacity {
                wal_max_bytes: 1,
                allocated_bytes: 0,
                reservation_bytes: 2,
            }),
            "wal_capacity",
        ),
        (
            sync::index::Error::Sync(sync::Error::Store(store::Error::FileCapacity { limit: 1 })),
            "file_capacity",
        ),
        (
            sync::index::Error::File {
                path: "fixture.rs".into(),
                source: Box::new(sync::Error::Store(store::Error::FileCapacity { limit: 1 })),
            },
            "file_capacity",
        ),
    ] {
        let error = anyhow::Error::new(failure);
        assert_eq!(
            super::index_capacity_error(&error).map(|value| value.0),
            Some(expected)
        );
    }
}
