//! Real-process recovery of catalogued and interrupted L1 work; v1 §4.3, T1.R1a.

use std::{
    error::Error,
    fs,
    path::Path,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

use assert_cmd::cargo::cargo_bin;
use rusqlite::{Connection, OpenFlags};
use serde_json::Value;
use tempfile::TempDir;

struct IndexChild(Child);

impl Drop for IndexChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn dkc(root: &Path, args: &[&str]) -> Result<Value, Box<dyn Error>> {
    let output = Command::new(cargo_bin("dkc"))
        .args(args)
        .current_dir(root)
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn database(root: &Path) -> Result<Connection, rusqlite::Error> {
    Connection::open_with_flags(
        root.join(".deepkit/code.db"),
        OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
}

fn interrupt_pending_index(root: &Path, rebuild: bool) -> Result<(), Box<dyn Error>> {
    let error_log = tempfile::NamedTempFile::new()?;
    let mut command = Command::new(cargo_bin("dkc"));
    command.arg("index").args(["--level", "1"]).arg("--json");
    if rebuild {
        command.arg("--rebuild-fts");
    }
    let mut child = IndexChild(
        command
            .current_dir(root)
            .stdout(Stdio::null())
            .stderr(error_log.reopen()?)
            .spawn()?,
    );
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let observed = database(root).and_then(|connection| {
            connection.query_row(
                "SELECT count(*), coalesce(sum(l1_pending), 0) FROM files",
                [],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
            )
        });
        if observed.is_ok_and(|(files, pending)| files == 122 && pending > 0) {
            child.0.kill()?;
            assert!(!child.0.wait()?.success());
            return Ok(());
        }
        assert!(
            Instant::now() < deadline,
            "L1 pending boundary was not observed"
        );
        assert!(
            child.0.try_wait()?.is_none(),
            "index finished before interruption: {}",
            fs::read_to_string(error_log.path())?
        );
        std::thread::yield_now();
    }
}

fn snapshot(root: &Path) -> Result<Value, Box<dyn Error>> {
    let connection = database(root)?;
    let units = connection
        .prepare(
            "SELECT f.path, hex(u.hash), u.line_start, u.line_end, c.text
         FROM units u JOIN unit_contents c ON c.hash = u.content_hash
         JOIN files f ON f.id = u.file_id ORDER BY f.path, u.hash",
        )?
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, String>(4)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    let symbols = connection
        .prepare(
            "SELECT f.path, s.name, s.line_start, s.line_end
         FROM symbols s JOIN files f ON f.id = s.file_id ORDER BY f.path, s.name",
        )?
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    let fts = connection
        .prepare(
            "SELECT f.path, hex(u.hash) FROM fts_units
         JOIN units u ON u.id = fts_units.rowid JOIN files f ON f.id = u.file_id
         WHERE fts_units MATCH 'retry' ORDER BY f.path, u.hash",
        )?
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    assert!(!units.is_empty());
    assert!(!symbols.is_empty());
    assert!(!fts.is_empty());
    Ok(serde_json::json!({"units": units, "symbols": symbols, "fts": fts}))
}

#[test]
fn normal_restart_recovers_catalogue_and_fts_rebuild_interruptions() -> Result<(), Box<dyn Error>> {
    let root = TempDir::new()?;
    assert!(
        Command::new("git")
            .args(["init", "--quiet"])
            .current_dir(root.path())
            .status()?
            .success()
    );
    for n in 0..120 {
        // The first file keeps chunking busy long enough to observe publication
        // without a product failpoint or a fixed timing sleep.
        let prefix = if n == 0 {
            "// recovery fixture context\n".repeat(8_000)
        } else {
            String::new()
        };
        fs::write(
            root.path().join(format!("file_{n:03}.rs")),
            format!("{prefix}pub fn retry_{n}(attempt: usize) -> bool {{ attempt < {n} }}\n"),
        )?;
    }
    fs::write(root.path().join("empty.rs"), "")?;
    fs::write(root.path().join("opaque.bin"), b"\0catalogue only")?;
    dkc(root.path(), &["init", "--json"])?;

    interrupt_pending_index(root.path(), false)?;
    assert!(
        dkc(root.path(), &["status", "--json"])?["l1_pending_files"]
            .as_u64()
            .is_some_and(|n| n > 0)
    );
    dkc(root.path(), &["index", "--level", "1", "--json"])?;
    assert_eq!(
        dkc(root.path(), &["status", "--json"])?["l1_pending_files"],
        0
    );
    let recovered = snapshot(root.path())?;
    assert_eq!(
        dkc(root.path(), &["index", "--level", "1", "--json"])?["units_indexed"],
        0
    );

    interrupt_pending_index(root.path(), true)?;
    dkc(root.path(), &["index", "--level", "1", "--json"])?;
    assert_eq!(
        dkc(root.path(), &["status", "--json"])?["l1_pending_files"],
        0
    );
    assert_eq!(snapshot(root.path())?, recovered);

    dkc(root.path(), &["clean", "--json"])?;
    dkc(root.path(), &["index", "--level", "1", "--json"])?;
    assert_eq!(snapshot(root.path())?, recovered);
    assert_eq!(
        dkc(root.path(), &["status", "--json"])?["l1_pending_files"],
        0
    );
    Ok(())
}

#[test]
fn wal_refused_publication_retains_search_snapshot_and_normal_restart_recovers()
-> Result<(), Box<dyn Error>> {
    for page_size in [4096, 65536] {
        let root = TempDir::new()?;
        assert!(
            Command::new("git")
                .args(["init", "--quiet"])
                .current_dir(root.path())
                .status()?
                .success()
        );
        fs::write(
            root.path().join("source.rs"),
            "pub fn retry_original() -> usize { 1 }\n",
        )?;
        dkc(root.path(), &["init", "--json"])?;
        let config_path = root.path().join(".deepkit/config.toml");
        let mut config: toml::Value = toml::from_str(&fs::read_to_string(&config_path)?)?;
        config["limits"]["wal_max_bytes"] = toml::Value::String("8 MiB".into());
        fs::write(config_path, toml::to_string(&config)?)?;
        let path = root.path().join(".deepkit/code.db");
        {
            let db = Connection::open(&path)?;
            db.execute_batch("PRAGMA journal_mode=DELETE")?;
            db.pragma_update(None, "page_size", page_size)?;
            db.execute_batch("VACUUM")?;
        }
        dkc(root.path(), &["index", "--level", "1", "--json"])?;
        let old = snapshot(root.path())?;
        let mut writer = Connection::open(&path)?;
        writer.execute_batch(
            "CREATE TABLE overflow_payload(value BLOB);
            CREATE TRIGGER fail_publication AFTER INSERT ON units BEGIN
            INSERT INTO overflow_payload VALUES(zeroblob(16777216)); END;
            PRAGMA wal_checkpoint(TRUNCATE);",
        )?;
        let old_head: String =
            writer.query_row("SELECT value FROM meta WHERE key='indexed_head'", [], |r| {
                r.get(0)
            })?;
        let read = writer.transaction()?;
        assert_eq!(
            read.query_row("SELECT count(*) FROM units", [], |r| r.get::<_, i64>(0))?,
            1
        );
        fs::write(
            root.path().join("source.rs"),
            "pub fn retry_replacement() -> usize { 222 }\n",
        )?;
        let output = Command::new(cargo_bin("dkc"))
            .args(["index", "--level", "1", "--json"])
            .current_dir(root.path())
            .output()?;
        assert!(!output.status.success());
        assert!(
            !output.stdout.is_empty(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let partial: Value = serde_json::from_slice(&output.stdout)?;
        assert_eq!(partial["index"]["state"], "partial");
        assert_eq!(partial["error"]["code"], "wal_capacity");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("WAL"),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(snapshot(root.path())?, old);
        assert_eq!(
            dkc(root.path(), &["status", "--json"])?["l1_pending_files"],
            1
        );
        assert_eq!(
            database(root.path())?.query_row(
                "SELECT value FROM meta WHERE key='indexed_head'",
                [],
                |r| r.get::<_, String>(0)
            )?,
            old_head
        );
        assert!(fs::metadata(root.path().join(".deepkit/code.db-wal"))?.len() <= 8 * 1024 * 1024);
        read.commit()?;
        writer.execute_batch("DROP TRIGGER fail_publication")?;
        dkc(root.path(), &["index", "--level", "1", "--json"])?;
        assert_eq!(
            dkc(root.path(), &["status", "--json"])?["l1_pending_files"],
            0
        );
        let recovered = snapshot(root.path())?;
        assert_ne!(recovered, old);
        assert_eq!(
            writer.query_row("SELECT count(*) FROM overflow_payload", [], |r| r
                .get::<_, i64>(0))?,
            0
        );
        assert_eq!(
            writer.query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
            "ok"
        );
        drop(writer);
        dkc(root.path(), &["clean", "--json"])?;
        dkc(root.path(), &["index", "--level", "1", "--json"])?;
        assert_eq!(snapshot(root.path())?, recovered);
    }
    Ok(())
}

#[test]
fn main_storage_exhaustion_defers_l1_preserves_old_hits_and_resumes_without_source_edits()
-> Result<(), Box<dyn Error>> {
    for page_size in [4096, 65536] {
        let root = TempDir::new()?;
        fs::create_dir(root.path().join(".deepkit"))?;
        let path = root.path().join(".deepkit/code.db");
        let db = Connection::open(&path)?;
        db.pragma_update(None, "page_size", page_size)?;
        db.execute_batch("CREATE TABLE overflow_payload(value BLOB)")?;
        fs::write(
            root.path().join("old.rs"),
            "pub fn storage_original() -> usize { 1 }\n",
        )?;
        dkc(root.path(), &["index", "--level", "1", "--json"])?;
        assert_eq!(
            db.query_row("SELECT count(*) FROM units", [], |r| r.get::<_, i64>(0))?,
            1
        );
        let allocated: i64 = db.query_row(
            "SELECT page_count*page_size FROM pragma_page_count(),pragma_page_size()",
            [],
            |r| r.get(0),
        )?;
        let cap = allocated + 2 * 1024 * 1024;
        fs::write(
            root.path().join(".deepkit/config.toml"),
            format!("[limits]\ndb_max_bytes='{cap} B'\n"),
        )?;
        db.execute_batch("CREATE TRIGGER storage_exhaustion AFTER INSERT ON unit_contents WHEN NEW.text LIKE '%storage_new_%' BEGIN INSERT INTO overflow_payload VALUES(zeroblob(16777216)); END;
            PRAGMA wal_checkpoint(TRUNCATE);")?;
        fs::write(
            root.path().join("old.rs"),
            "pub fn storage_new_replacement() -> usize { 2 }\n",
        )?;
        fs::write(
            root.path().join("new.rs"),
            "pub fn storage_new_file() -> usize { 3 }\n",
        )?;
        for _ in 0..2 {
            let indexed = dkc(root.path(), &["index", "--level", "1", "--json"])?;
            assert_eq!(indexed["units_indexed"], 0);
            let deferred = indexed["l1_deferred"]
                .as_array()
                .ok_or("missing deferred outcomes")?;
            assert_eq!(deferred.len(), 2);
            for file in deferred {
                assert_eq!(file["reason"], "storage_full");
                assert_eq!(file["effective_level"], 0);
                assert_eq!(file["retained_snapshot"], file["path"] == "old.rs");
            }
            let status = dkc(root.path(), &["status", "--json"])?;
            assert_eq!(status["l1_pending_files"], 0);
            assert_eq!(status["units"], 1);
            assert_eq!(status["l1_deferred"], indexed["l1_deferred"]);
            assert_eq!(
                db.query_row("SELECT count(*) FROM overflow_payload", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            let found = dkc(
                root.path(),
                &["search", "storage_original", "--no-rerank", "--json"],
            )?;
            assert!(found["hits"].as_array().ok_or("missing hits")?.is_empty());
            assert_eq!(found["index"]["partial"], true);
            assert_eq!(found["index"]["suppressed_hits"], 1);
            // Suppression must not evict the retained content or lexical posting.
            assert_eq!(
                db.query_row(
                    "SELECT c.text FROM units u JOIN unit_contents c ON c.hash=u.content_hash
                 JOIN fts_units f ON f.rowid=u.id WHERE fts_units MATCH 'storage_original'",
                    [],
                    |row| row.get::<_, String>(0),
                )?,
                "pub fn storage_original() -> usize { 1 }"
            );
            db.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")?;
            assert!(fs::metadata(&path)?.len() <= cap as u64);
        }
        db.execute_batch("DROP TRIGGER storage_exhaustion")?;
        let resumed = dkc(root.path(), &["index", "--level", "1", "--json"])?;
        assert_eq!(resumed["units_indexed"], 2);
        let repaired = dkc(
            root.path(),
            &["search", "storage_new_replacement", "--no-rerank", "--json"],
        )?;
        assert!(
            !repaired["hits"]
                .as_array()
                .ok_or("missing hits")?
                .is_empty()
        );
        assert_eq!(repaired["hits"][0]["fresh"]["state"], "fresh");
        assert_eq!(resumed["l1_deferred"], serde_json::json!([]));
        assert_eq!(dkc(root.path(), &["status", "--json"])?["units"], 2);
        assert_eq!(
            dkc(root.path(), &["index", "--level", "1", "--json"])?["units_indexed"],
            0
        );
        assert_eq!(
            db.query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
            "ok"
        );
    }
    Ok(())
}

#[test]
fn catalogue_and_control_capacity_failures_emit_partial_json_and_preserve_commits()
-> Result<(), Box<dyn Error>> {
    for stage in ["bootstrap", "catalogue", "control"] {
        let root = TempDir::new()?;
        fs::create_dir(root.path().join(".deepkit"))?;
        let config = root.path().join(".deepkit/config.toml");
        if stage == "bootstrap" {
            fs::write(&config, "[limits]\ndb_max_bytes='1 B'\n")?;
        } else {
            fs::write(
                root.path().join("source.rs"),
                "pub fn retained_on_failure() {}\n",
            )?;
            dkc(root.path(), &["index", "--level", "1", "--json"])?;
            let db = Connection::open(root.path().join(".deepkit/code.db"))?;
            let allocated: i64 = db.query_row(
                "SELECT page_count*page_size FROM pragma_page_count(),pragma_page_size()",
                [],
                |r| r.get(0),
            )?;
            fs::write(
                &config,
                format!("[limits]\ndb_max_bytes='{} B'\n", allocated + 1024 * 1024),
            )?;
            db.execute_batch("CREATE TABLE overflow_payload(value BLOB)")?;
            let event = if stage == "catalogue" {
                "AFTER INSERT ON files"
            } else {
                "AFTER UPDATE ON meta WHEN NEW.key='model_preparation'"
            };
            db.execute_batch(&format!("CREATE TRIGGER stop_capacity {event} BEGIN INSERT INTO overflow_payload VALUES(zeroblob(16777216)); END;"))?;
            fs::write(root.path().join("new.rs"), "pub fn not_admitted() {}\n")?;
        }
        let output = Command::new(cargo_bin("dkc"))
            .current_dir(root.path())
            .args(["index", "--level", "1", "--json"])
            .output()?;
        assert!(!output.status.success());
        assert!(
            !output.stdout.is_empty(),
            "{stage}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let partial: Value = serde_json::from_slice(&output.stdout)?;
        assert_eq!(partial["index"]["state"], "partial");
        assert_eq!(partial["index"]["degraded"], true);
        assert_eq!(partial["index"]["retry_required"], true);
        assert_eq!(
            partial["error"]["code"],
            if stage == "bootstrap" {
                "database_limit"
            } else {
                "storage_full"
            }
        );
        assert!(partial["took_ms"].is_number());
        assert!(partial.get("units_indexed").is_none());
        if stage == "bootstrap" {
            assert_eq!(fs::metadata(root.path().join(".deepkit/code.db"))?.len(), 0);
            fs::write(&config, "")?;
        } else {
            let db = Connection::open(root.path().join(".deepkit/code.db"))?;
            assert_eq!(
                db.query_row("SELECT count(*) FROM files", [], |r| r.get::<_, i64>(0))?,
                1
            );
            assert_eq!(
                db.query_row("SELECT count(*) FROM units", [], |r| r.get::<_, i64>(0))?,
                1
            );
            assert_eq!(
                db.query_row("SELECT count(*) FROM overflow_payload", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            db.execute_batch("DROP TRIGGER stop_capacity")?;
        }
        dkc(root.path(), &["index", "--level", "1", "--json"])?;
    }
    let root = TempDir::new()?;
    fs::create_dir(root.path().join(".deepkit"))?;
    fs::write(root.path().join(".deepkit/config.toml"), "invalid toml [")?;
    let output = Command::new(cargo_bin("dkc"))
        .current_dir(root.path())
        .args(["index", "--level", "1", "--json"])
        .output()?;
    assert!(!output.status.success());
    assert!(
        output.stdout.is_empty(),
        "configuration errors must not be mislabeled as capacity"
    );
    Ok(())
}

#[test]
fn ordinary_index_runs_incremental_maintenance_without_reindexing_unchanged_files()
-> Result<(), Box<dyn Error>> {
    let root = TempDir::new()?;
    fs::write(
        root.path().join("source.rs"),
        "pub fn maintenance_retained() {}\n",
    )?;
    let first = dkc(root.path(), &["index", "--level", "1", "--json"])?;
    assert_eq!(first["maintenance"]["optimization_ran"], true);
    let path = root.path().join(".deepkit/code.db");
    let db = Connection::open(&path)?;
    db.execute_batch("CREATE TABLE maintenance_payload(id INTEGER PRIMARY KEY,value BLOB); INSERT INTO maintenance_payload VALUES(7,'retained'),(9,zeroblob(8388608)); DELETE FROM maintenance_payload WHERE id=9; PRAGMA wal_checkpoint(TRUNCATE)")?;
    let free: i64 = db.query_row("PRAGMA freelist_count", [], |r| r.get(0))?;
    assert!(free > 64);
    let before = fs::metadata(&path)?.len();
    let indexed = dkc(root.path(), &["index", "--level", "1", "--json"])?;
    assert_eq!(indexed["units_indexed"], 0);
    assert!(
        indexed["maintenance"]["pages_reclaimed"]
            .as_u64()
            .unwrap_or(0)
            > 64
    );
    assert_eq!(indexed["maintenance"]["optimization_ran"], true);
    assert!(fs::metadata(&path)?.len() < before);
    assert_eq!(
        db.query_row("PRAGMA freelist_count", [], |r| r.get::<_, i64>(0))?,
        0
    );
    assert_eq!(
        db.query_row("SELECT value FROM maintenance_payload", [], |r| r
            .get::<_, String>(0))?,
        "retained"
    );
    assert_eq!(
        db.query_row("SELECT count(*) FROM units", [], |r| r.get::<_, i64>(0))?,
        1
    );
    let again = dkc(root.path(), &["index", "--level", "1", "--json"])?;
    assert_eq!(again["units_indexed"], 0);
    assert_eq!(again["maintenance"]["pages_reclaimed"], 0);
    Ok(())
}

#[test]
fn file_capacity_refusal_is_partial_retains_catalogue_and_resumes() -> Result<(), Box<dyn Error>> {
    let root = TempDir::new()?;
    fs::write(root.path().join("old.rs"), "pub fn retained() {}\n")?;
    dkc(root.path(), &["index", "--level", "1", "--json"])?;
    let db = database(root.path())?;
    let before: i64 = db.query_row(
        "SELECT CAST(value AS INTEGER) FROM meta WHERE key='l1_scan_generation'",
        [],
        |row| row.get(0),
    )?;
    // Capped discovery omits the old path, so absence cannot authorize its deletion.
    fs::remove_file(root.path().join("old.rs"))?;
    fs::write(root.path().join("a.rs"), "pub fn admitted_later() {}\n")?;
    fs::write(root.path().join("b.rs"), "pub fn also_later() {}\n")?;
    let config = root.path().join(".deepkit/config.local.toml");
    fs::write(&config, "[limits]\nmax_files=1\n")?;
    let output = Command::new(cargo_bin("dkc"))
        .current_dir(root.path())
        .args(["index", "--level", "1", "--json"])
        .output()?;
    assert!(!output.status.success());
    let partial: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(partial["index"]["state"], "partial");
    assert_eq!(partial["error"]["code"], "file_capacity");
    assert_eq!(
        db.query_row("SELECT path FROM files", [], |row| row.get::<_, String>(0))?,
        "old.rs"
    );
    assert_eq!(
        db.query_row(
            "SELECT CAST(value AS INTEGER) FROM meta WHERE key='l1_scan_generation'",
            [],
            |row| row.get::<_, i64>(0)
        )?,
        before
    );
    let status = dkc(root.path(), &["status", "--json"])?;
    assert_eq!(status["caps"]["file_capacity_reached"], true);
    fs::write(&config, "[limits]\nmax_files=3\n")?;
    dkc(root.path(), &["index", "--level", "1", "--json"])?;
    let status = dkc(root.path(), &["status", "--json"])?;
    assert_eq!(status["files"], 2);
    assert_eq!(status["caps"]["file_capacity_reached"], false);
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM files WHERE path='old.rs'",
            [],
            |row| row.get::<_, i64>(0)
        )?,
        0
    );
    Ok(())
}

#[test]
fn changed_grammar_fingerprint_refreshes_unchanged_files_and_retains_payload_identity()
-> Result<(), Box<dyn Error>> {
    let root = TempDir::new()?;
    fs::write(root.path().join("source.rs"), "pub fn retry() {}\n")?;
    fs::write(
        root.path().join("source.c"),
        "int retry(int value) { return value; }\n",
    )?;
    dkc(root.path(), &["index", "--level", "1", "--json"])?;
    let before = snapshot(root.path())?;
    {
        let db = Connection::open(root.path().join(".deepkit/code.db"))?;
        // A completed previous grammar generation, with no content or pending-work changes.
        db.execute(
            "UPDATE files SET l1_fingerprint=zeroblob(32),l1_desired=zeroblob(32),l1_pending=0",
            [],
        )?;
    }
    let upgraded = dkc(root.path(), &["index", "--level", "1", "--json"])?;
    assert!(upgraded["units_indexed"].as_u64().unwrap() >= 2);
    assert_eq!(snapshot(root.path())?, before);
    let repeated = dkc(root.path(), &["index", "--level", "1", "--json"])?;
    assert_eq!(repeated["units_indexed"], 0);
    assert_eq!(snapshot(root.path())?, before);
    Ok(())
}
