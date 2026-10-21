//! Queue inspection reports persisted states without consuming or recovering work.
use assert_cmd::Command;
use rusqlite::Connection;
use serde_json::Value;
use std::error::Error;

fn command(root: &std::path::Path, args: &[&str]) -> Value {
    let output = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .args(args)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    serde_json::from_slice(&output).expect("JSON command output")
}

#[test]
fn status_reports_delays_and_uncertain_charges_without_mutating_jobs() -> Result<(), Box<dyn Error>>
{
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    let empty = command(root, &["queue", "status", "--json"]);
    assert_eq!(empty["jobs"]["queued"], 0);
    assert_eq!(empty["providers"], serde_json::json!({}));
    let database = Connection::open(root.join(".deepkit/code.db"))?;
    database.execute_batch(
        "INSERT INTO jobs(kind,target_kind,target_id,target_hash,desired_hash,provider,priority,tokens,state,attempts,not_before,leased_until,created_at,updated_at)
         VALUES ('embed','unit',1,zeroblob(32),zeroblob(32),'mock',1,10,'queued',0,0,NULL,0,0),
                ('embed','unit',2,zeroblob(32),zeroblob(32),'mock',1,10,'failed',1,253402214400,NULL,0,0),
                ('embed','unit',3,zeroblob(32),zeroblob(32),'mock',1,10,'leased',1,0,1,0,0),
                ('embed','unit',4,zeroblob(32),zeroblob(32),'mock',1,10,'dead',8,0,NULL,0,0),
                ('embed','unit',5,zeroblob(32),zeroblob(32),'mock',1,10,'done',1,0,NULL,0,0),
                ('embed','unit',6,zeroblob(32),zeroblob(32),'mock',1,10,'failed',8,0,NULL,0,0);
         INSERT INTO usage(day,provider,calls,tokens,estimated_tokens)
         VALUES(date('now'),'mock',2,30,10), ('2000-01-01','mock',99,999,0),
               (date('now'),'query-only',1,7,0);
         INSERT INTO usage_reservations(id,provider,day,calls,tokens,state,created_at)
         VALUES ('r','mock',date('now'),1,20,'reserved',0),
                ('u','mock','2000-01-01',1,40,'dispatched',0),
                ('s','mock',date('now'),1,30,'reconciled',0),
                ('x','mock',date('now'),1,50,'released',0);
         CREATE TRIGGER no_job_update BEFORE UPDATE ON jobs BEGIN SELECT RAISE(ABORT,'inspection mutated jobs'); END;
         CREATE TRIGGER no_reservation_update BEFORE UPDATE ON usage_reservations BEGIN SELECT RAISE(ABORT,'inspection mutated accounting'); END;",
    )?;
    let status = command(root, &["queue", "status", "--json"]);
    assert_eq!(
        status["jobs"],
        serde_json::json!({
            "queued":1,"leased":1,"done":1,"failed":2,"dead":1,
            "ready":1,"deferred":1,"expired_leases":1,"next_ready_at":253402214400_i64
        })
    );
    let provider = &status["providers"]["mock"];
    for (field, expected) in [
        ("calls_today", 2),
        ("tokens_today", 30),
        ("estimated_tokens_today", 10),
        ("reserved_calls", 1),
        ("reserved_tokens", 20),
        ("uncertain_calls", 1),
        ("uncertain_tokens", 40),
    ] {
        assert_eq!(provider[field], expected, "{field}");
    }
    assert_eq!(status["providers"]["query-only"]["calls_today"], 1);
    let general = command(root, &["status", "--json"]);
    assert_eq!(general["queue"], status["jobs"]);
    assert_eq!(general["queue_status"]["providers"], status["providers"]);
    let text = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .args(["queue", "status"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    assert!(String::from_utf8(text)?.contains("uncertain 1 calls / 40 tokens"));
    assert_eq!(
        database.query_row("SELECT COUNT(*) FROM jobs", [], |row| row.get::<_, i64>(0))?,
        6
    );
    Ok(())
}

#[test]
fn cli_shows_safe_pause_categories_and_utc_resume_times() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    command(root, &["queue", "status", "--json"]);
    let database = Connection::open(root.join(".deepkit/code.db"))?;
    database.execute_batch(
        "INSERT INTO jobs(kind,target_kind,target_id,target_hash,desired_hash,provider,priority,tokens,state,attempts,not_before,last_error,created_at,updated_at)
         VALUES ('embed','unit',1,zeroblob(32),zeroblob(32),'mock',1,10,'queued',0,253402214400,'daily budget exhausted; waiting until UTC midnight',0,0),
                ('embed','unit',2,zeroblob(32),zeroblob(32),'mock',1,10,'queued',0,253402214401,'provider rate limit; waiting for capacity',0,0),
                ('embed','unit',3,zeroblob(32),zeroblob(32),'mock',1,10,'failed',1,253402214402,'transient provider error',0,0),
                ('embed','unit',4,zeroblob(32),zeroblob(32),'mock',1,10,'failed',1,253402214403,'untrusted arbitrary response text',0,0),
                ('embed','unit',5,zeroblob(32),zeroblob(32),'mock',1,10,'dead',8,253402214400,'daily budget exhausted; waiting until UTC midnight',0,0),
                ('embed','unit',6,zeroblob(32),zeroblob(32),'mock',1,10,'queued',0,0,'daily budget exhausted; waiting until UTC midnight',0,0);
         CREATE TRIGGER no_update BEFORE UPDATE ON jobs BEGIN SELECT RAISE(ABORT,'status changed a job'); END;"
    )?;
    let status = command(root, &["queue", "status", "--json"]);
    let deferrals = &status["providers"]["mock"]["deferrals"];
    for reason in ["daily_budget", "provider_rate", "retry", "other"] {
        assert_eq!(deferrals[reason]["jobs"], 1);
    }
    assert_eq!(
        deferrals["daily_budget"]["earliest_utc"],
        "9999-12-31 00:00:00 UTC"
    );
    assert_eq!(status["jobs"]["ready"], 1);
    assert_eq!(status["jobs"]["deferred"], 4);
    assert!(
        !status
            .to_string()
            .contains("untrusted arbitrary response text")
    );
    let general = command(root, &["status", "--json"]);
    assert_eq!(general["queue_status"]["providers"], status["providers"]);
    let output = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .args(["queue", "status"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(output)?;
    assert!(text.contains(
        "1 jobs paused: daily budget exhausted; earliest eligible 9999-12-31 00:00:00 UTC"
    ));
    assert!(!text.contains("untrusted arbitrary response text"));
    let listed = command(root, &["queue", "status", "--jobs", "4", "--json"]);
    assert!(
        !listed
            .to_string()
            .contains("untrusted arbitrary response text")
    );
    assert_eq!(listed["entries"][1]["id"], 4);
    assert_eq!(listed["entries"][1]["reason"], "other");
    let text = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .args(["queue", "status", "--jobs", "4"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(text)?;
    assert!(text.contains("Jobs: showing 4 of 6"));
    assert!(text.contains("4: failed"));
    assert!(!text.contains("untrusted arbitrary response text"));

    Ok(())
}

#[test]
fn explicit_retry_preserves_delays_attempt_limits_leases_and_accounting()
-> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    command(root, &["queue", "status", "--json"]);
    let database = Connection::open(root.join(".deepkit/code.db"))?;
    database.execute_batch(
        "INSERT INTO jobs(id,kind,target_kind,target_id,target_hash,desired_hash,provider,priority,tokens,state,attempts,not_before,lease_owner,leased_until,lease_generation,created_at,updated_at)
         VALUES (1,'embed','unit',1,zeroblob(32),zeroblob(32),'mock',1,10,'failed',2,253402214400,NULL,NULL,3,0,0),
                (2,'embed','unit',2,zeroblob(32),zeroblob(32),'mock',1,10,'dead',1,0,NULL,NULL,4,0,0),
                (3,'embed','unit',3,zeroblob(32),zeroblob(32),'mock',1,10,'dead',8,0,NULL,NULL,8,0,0),
                (4,'embed','unit',4,zeroblob(32),zeroblob(32),'mock',1,10,'leased',1,0,'active',253402214400,2,0,0),
                (5,'embed','unit',5,zeroblob(32),zeroblob(32),'mock',1,10,'done',1,0,NULL,NULL,1,0,0),
                (6,'embed','unit',6,zeroblob(32),zeroblob(32),'mock',1,10,'queued',0,253402214400,NULL,NULL,0,0,0),
                (7,'embed','unit',7,zeroblob(32),zeroblob(32),'mock',1,10,'failed',8,0,NULL,NULL,8,0,0);
         INSERT INTO usage(day,provider,calls,tokens,estimated_tokens) VALUES(date('now'),'mock',8,80,80);
         INSERT INTO usage_reservations(id,provider,day,job_id,lease_generation,calls,tokens,state,created_at)
         VALUES('uncertain','mock',date('now'),1,3,1,10,'dispatched',0);
         CREATE TRIGGER preserve_jobs BEFORE UPDATE ON jobs
         WHEN OLD.id NOT IN (1,2) OR NEW.attempts!=OLD.attempts
           OR NEW.not_before!=OLD.not_before OR NEW.desired_hash!=OLD.desired_hash
           OR NEW.target_hash!=OLD.target_hash OR NEW.lease_generation!=OLD.lease_generation
         BEGIN SELECT RAISE(ABORT,'retry violated job contract'); END;
         CREATE TRIGGER preserve_usage BEFORE UPDATE ON usage BEGIN SELECT RAISE(ABORT,'usage changed'); END;
         CREATE TRIGGER preserve_reservations BEFORE UPDATE ON usage_reservations BEGIN SELECT RAISE(ABORT,'reservation changed'); END;"
    )?;
    let listed = command(root, &["queue", "status", "--jobs", "2", "--json"]);
    let entries = listed["entries"].as_array().ok_or("job entries")?;
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0]["id"], 1);
    assert_eq!(entries[0]["not_before"], 253402214400_i64);
    assert_eq!(entries[0]["retry_allowed"], true);
    assert_eq!(entries[1]["id"], 2);
    assert_eq!(entries[1]["retry_allowed"], true);
    assert!(
        command(root, &["queue", "status", "--json"])
            .get("entries")
            .is_none()
    );
    for invalid in ["0", "1001"] {
        Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
            .current_dir(root)
            .args(["queue", "status", "--jobs", invalid])
            .assert()
            .failure();
    }
    assert_eq!(
        command(root, &["queue", "retry", "--job", "2", "--json"])["requeued"],
        1
    );
    assert_eq!(
        command(root, &["queue", "retry", "--job", "2", "--json"])["requeued"],
        0
    );
    assert_eq!(command(root, &["queue", "retry", "--json"])["requeued"], 1);
    let snapshot = command(root, &["queue", "status", "--json"]);
    assert_eq!(snapshot["jobs"]["ready"], 1);
    assert_eq!(snapshot["jobs"]["deferred"], 2);
    assert_eq!(snapshot["jobs"]["leased"], 1);
    assert_eq!(snapshot["jobs"]["done"], 1);
    assert_eq!(snapshot["providers"]["mock"]["calls_today"], 8);
    assert_eq!(snapshot["providers"]["mock"]["uncertain_calls"], 1);
    let listed = command(root, &["queue", "status", "--jobs", "1000", "--json"]);
    let entries = listed["entries"].as_array().ok_or("job entries")?;
    assert_eq!(entries.len(), 7);
    assert_eq!(entries[0]["id"], 3);
    assert_eq!(entries[1]["id"], 7);
    assert!(entries.iter().all(|entry| entry["retry_allowed"] == false));

    Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .args(["queue", "retry", "--job", "0"])
        .assert()
        .failure();
    Ok(())
}

#[test]
fn explicit_regeneration_requeues_only_completed_text_summaries_and_retains_output()
-> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    command(root, &["queue", "status", "--json"]);
    let database = Connection::open(root.join(".deepkit/code.db"))?;
    database.execute_batch(
        "INSERT INTO files(id,path,blake3,size,mtime_ns,kind,max_level,treatment)
         VALUES(1,'src/summary.rs',X'01',1,0,'code',3,'source');
         INSERT INTO unit_contents(hash,text,tokens)
         VALUES(X'02','pub struct Summary;',1),(X'03','existing owner-bound output',1);
         INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,created_at,updated_at)
         VALUES(10,X'02',X'02','src/summary.rs',1,'symbol','Summary',0,0),
               (11,X'03',X'03','summary:symbol:7',NULL,'summary','Summary',0,0);
         UPDATE units SET parent_unit=10 WHERE id=11;
         INSERT INTO symbols(id,file_id,unit_id,name,kind,line_start,line_end,summary_unit)
         VALUES(7,1,10,'Summary','struct',1,1,11);
         INSERT INTO jobs(id,kind,target_kind,target_id,target_hash,desired_hash,provider,priority,tokens,state,attempts,not_before,lease_generation,created_at,updated_at)
         VALUES(20,'enrich:describe_symbol','symbol',7,zeroblob(32),zeroblob(32),'llm:remote',1,10,'done',2,0,4,0,0),
               (21,'enrich:describe_image','file',1,zeroblob(32),zeroblob(32),'llm:remote',1,10,'done',2,0,5,0,0),
               (22,'enrich:summarize_pr','pr',8,zeroblob(32),zeroblob(32),'llm:remote',1,10,'done',8,0,6,0,0),
               (23,'enrich:describe_symbol','symbol',8,zeroblob(32),zeroblob(32),'llm:remote',1,10,'done',2,0,7,0,0);
         INSERT INTO usage(day,provider,calls,tokens,estimated_tokens)
         VALUES(date('now'),'llm:remote',2,20,20);
         INSERT INTO usage_reservations(id,provider,day,job_id,lease_generation,calls,tokens,state,created_at)
         VALUES('retained','llm:remote',date('now'),20,4,1,10,'reconciled',0);
         CREATE TRIGGER retain_summary BEFORE DELETE ON units WHEN OLD.id=11
         BEGIN SELECT RAISE(ABORT,'regeneration removed current summary'); END;
         CREATE TRIGGER retain_owner BEFORE UPDATE OF summary_unit ON symbols
         BEGIN SELECT RAISE(ABORT,'regeneration changed summary owner'); END;
         CREATE TRIGGER preserve_usage BEFORE UPDATE ON usage
         BEGIN SELECT RAISE(ABORT,'regeneration changed usage'); END;
         CREATE TRIGGER preserve_reservation BEFORE UPDATE ON usage_reservations
         BEGIN SELECT RAISE(ABORT,'regeneration changed reservation'); END;",
    )?;

    let listed = command(root, &["queue", "status", "--jobs", "1000", "--json"]);
    let entries = listed["entries"].as_array().ok_or("job entries")?;
    let summary = entries
        .iter()
        .find(|entry| entry["id"] == 20)
        .ok_or("summary job")?;
    assert_eq!(summary["kind"], "enrich:describe_symbol");
    assert_eq!(summary["target_kind"], "symbol");
    assert_eq!(summary["target_id"], 7);
    assert_eq!(summary["regeneration_allowed"], true);
    assert_eq!(
        entries
            .iter()
            .find(|entry| entry["id"] == 21)
            .ok_or("image job")?["regeneration_allowed"],
        false
    );
    assert_eq!(
        entries
            .iter()
            .find(|entry| entry["id"] == 22)
            .ok_or("exhausted job")?["regeneration_allowed"],
        false
    );
    assert_eq!(
        entries
            .iter()
            .find(|entry| entry["id"] == 23)
            .ok_or("unowned text job")?["regeneration_allowed"],
        false
    );

    let regenerated = command(root, &["queue", "regenerate", "--job", "20", "--json"]);
    assert_eq!(regenerated["requeued"], 1);
    assert_eq!(regenerated["output_retained"], true);
    assert_eq!(regenerated["attempts_preserved"], true);
    assert_eq!(
        command(root, &["queue", "regenerate", "--job", "20", "--json"])["requeued"],
        0
    );
    assert_eq!(
        command(root, &["queue", "regenerate", "--job", "21", "--json"])["requeued"],
        0
    );
    assert_eq!(
        command(root, &["queue", "regenerate", "--job", "22", "--json"])["requeued"],
        0
    );
    assert_eq!(
        command(root, &["queue", "regenerate", "--job", "23", "--json"])["requeued"],
        0
    );
    assert_eq!(
        database.query_row(
            "SELECT state,attempts,lease_generation,desired_hash,target_hash FROM jobs WHERE id=20",
            [],
            |row| Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, Vec<u8>>(3)?,
                row.get::<_, Vec<u8>>(4)?
            )),
        )?,
        ("queued".to_owned(), 2, 4, vec![0; 32], vec![0; 32])
    );
    assert_eq!(
        database.query_row("SELECT summary_unit FROM symbols WHERE id=7", [], |row| row
            .get::<_, i64>(0))?,
        11
    );
    assert_eq!(
        database.query_row("SELECT count(*) FROM units WHERE id=11", [], |row| row
            .get::<_, i64>(0))?,
        1
    );
    assert_eq!(
        database.query_row("SELECT state FROM jobs WHERE id=23", [], |row| row
            .get::<_, String>(0))?,
        "done"
    );
    assert_eq!(
        database.query_row(
            "SELECT count(*) FROM usage_reservations WHERE id='retained'",
            [],
            |row| row.get::<_, i64>(0)
        )?,
        1
    );
    Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .args(["queue", "regenerate", "--job", "0"])
        .assert()
        .failure();
    Ok(())
}

#[test]
fn retry_json_reports_committed_batches_when_a_later_batch_fails() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    command(root, &["queue", "status", "--json"]);
    let mut db = Connection::open(root.join(".deepkit/code.db"))?;
    {
        let tx = db.transaction()?;
        for id in 1..=130 {
            tx.execute("INSERT INTO jobs(kind,target_kind,target_id,target_hash,desired_hash,provider,priority,tokens,state,attempts,not_before,created_at,updated_at)
                VALUES('embed','unit',?1,zeroblob(32),zeroblob(32),'mock',1,10,'failed',2,253402214400,0,0)",[id])?;
        }
        tx.execute_batch("CREATE TRIGGER interrupt_bulk AFTER UPDATE ON jobs WHEN NEW.id=65 BEGIN SELECT RAISE(ABORT,'injected later-batch failure'); END;")?;
        tx.commit()?;
    }
    let output = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .args(["queue", "retry", "--json"])
        .assert()
        .failure()
        .get_output()
        .clone();
    let partial: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(partial["requeued"], 64);
    assert_eq!(partial["complete"], false);
    assert!(partial["took_ms"].is_number());
    assert!(String::from_utf8_lossy(&output.stderr).contains("64 committed rows"));
    assert_eq!(
        db.query_row("SELECT count(*) FROM jobs WHERE state='queued'", [], |r| {
            r.get::<_, i64>(0)
        })?,
        64
    );
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM jobs WHERE attempts=2 AND not_before=253402214400",
            [],
            |r| r.get::<_, i64>(0)
        )?,
        130
    );
    db.execute_batch("DROP TRIGGER interrupt_bulk")?;
    assert_eq!(command(root, &["queue", "retry", "--json"])["requeued"], 66);
    assert_eq!(command(root, &["queue", "retry", "--json"])["requeued"], 0);
    Ok(())
}

#[test]
fn cli_applies_database_cap_at_bootstrap_and_reports_lowered_cap_without_mutation()
-> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    std::fs::create_dir(root.join(".deepkit"))?;
    let config = root.join(".deepkit/config.toml");
    std::fs::write(&config, "[limits]\ndb_max_bytes='1 B'\n")?;
    Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .args(["queue", "status", "--json"])
        .assert()
        .failure();
    assert_eq!(std::fs::metadata(root.join(".deepkit/code.db"))?.len(), 0);
    std::fs::write(&config, "[limits]\ndb_max_bytes='16 MiB'\n")?;
    let initial = command(root, &["queue", "status", "--json"]);
    assert_eq!(
        initial["database"]["configured_limit_bytes"],
        16 * 1024 * 1024
    );
    assert_eq!(initial["database"]["over_limit"], false);
    let database = Connection::open(root.join(".deepkit/code.db"))?;
    database.execute_batch("CREATE TRIGGER deny_status_insert BEFORE INSERT ON meta BEGIN SELECT RAISE(ABORT,'status write'); END;
        CREATE TRIGGER deny_status_update BEFORE UPDATE ON meta BEGIN SELECT RAISE(ABORT,'status write'); END;
        PRAGMA wal_checkpoint(TRUNCATE);")?;
    let before = std::fs::read(root.join(".deepkit/code.db"))?;
    std::fs::write(&config, "[limits]\ndb_max_bytes='1 B'\n")?;
    let queue = command(root, &["queue", "status", "--json"]);
    let general = command(root, &["status", "--json"]);
    assert_eq!(queue["database"], general["queue_status"]["database"]);
    assert_eq!(queue["database"]["configured_limit_bytes"], 1);
    assert_eq!(queue["database"]["effective_limit_bytes"], 0);
    assert_eq!(queue["database"]["over_limit"], true);
    assert_eq!(queue["database"]["file_bytes"], before.len());
    let text = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .args(["status"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    assert!(String::from_utf8(text)?.contains("over cap; page growth frozen, data retained"));
    assert_eq!(std::fs::read(root.join(".deepkit/code.db"))?, before);
    Ok(())
}
