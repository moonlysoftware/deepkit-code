//! Real-process writer death and WAL reservation recovery; v1 §3.5/§8.2.

use super::*;
use crate::store::{self, StoreLimits};
use std::{
    error::Error as StdError,
    path::Path,
    process::{Child, Command},
    time::Instant,
};

struct Worker(Child);

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn wait_file(path: &Path) -> Result<(), Box<dyn StdError>> {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !path.exists() {
        if Instant::now() >= deadline {
            return Err(format!("timed out waiting for {}", path.display()).into());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    Ok(())
}

fn start(directory: &Path, name: &str, cap: u64) -> Result<Worker, Box<dyn StdError>> {
    let log = fs::File::create(directory.join(format!("{name}.log")))?;
    let child = Worker(
        Command::new(std::env::current_exe()?)
            .args([
                "--exact",
                "store::wal::process_tests::reservation_child",
                "--nocapture",
            ])
            .env("DKC_WAL_TEST_ROOT", directory)
            .env("DKC_WAL_TEST_WORKER", name)
            .env("DKC_WAL_TEST_CAP", cap.to_string())
            .stdout(log.try_clone()?)
            .stderr(log)
            .spawn()?,
    );
    if let Err(error) = wait_file(&directory.join(format!("{name}.ready"))) {
        return Err(format!(
            "{error}: {}",
            fs::read_to_string(directory.join(format!("{name}.log")))?
        )
        .into());
    }
    Ok(child)
}

#[test]
fn reservation_child() -> Result<(), Box<dyn StdError>> {
    let Some(root) = std::env::var_os("DKC_WAL_TEST_ROOT") else {
        return Ok(());
    };
    let root = PathBuf::from(root);
    let name = std::env::var("DKC_WAL_TEST_WORKER")?;
    let cap = std::env::var("DKC_WAL_TEST_CAP")?.parse()?;
    let store = store::open(root.join("code.db"), StoreLimits::default())?;
    fs::write(root.join(format!("{name}.ready")), b"ready")?;
    wait_file(&root.join(format!("{name}.start")))?;
    fs::write(root.join(format!("{name}.attempt")), b"attempt")?;
    write_reserved(&store, cap, cap, 0, |tx| {
        tx.execute("UPDATE wal_fixture SET value=?1 WHERE id=1", [&name])?;
        fs::write(root.join(format!("{name}.mutated")), b"mutated")
            .map_err(|source| Error::Io { source })?;
        if name == "uncommitted" {
            // Parent SIGKILLs this process while it owns BEGIN IMMEDIATE.
            std::thread::sleep(Duration::from_secs(30));
            return Err(Error::WalReservationUnsupported);
        }
        Ok(())
    })?;
    fs::write(root.join(format!("{name}.committed")), b"committed")?;
    // Keep the connection open so death cannot run SQLite's close checkpoint.
    std::thread::sleep(Duration::from_secs(30));
    Ok(())
}

#[test]
fn killed_writer_releases_reservation_and_committed_wal_survives_process_death()
-> Result<(), Box<dyn StdError>> {
    use std::os::unix::process::ExitStatusExt;

    for size in [4096_u32, 65536] {
        let directory = tempfile::tempdir()?;
        let root = directory.path();
        let path = root.join("code.db");
        store::register_vector_extension()?;
        {
            let db = Connection::open(&path)?;
            db.pragma_update(None, "page_size", size)?;
            db.execute_batch("VACUUM")?;
        }
        let cap = {
            let store = store::open(&path, StoreLimits::default())?;
            store.writer()?.execute_batch(
                "CREATE TABLE wal_fixture(id INTEGER PRIMARY KEY,value TEXT);
                 INSERT INTO wal_fixture VALUES(1,'original')",
            )?;
            8 * (u64::from(size) + 24) + 32
        };
        let mut uncommitted = start(root, "uncommitted", cap)?;
        let mut committed = start(root, "committed", cap)?;
        // Startup writes are outside this helper's scope. Establish the tested
        // empty WAL boundary only after both processes have opened their stores.
        let db = store::open(&path, StoreLimits::default())?;
        assert!(!checkpoint(&db)?.busy);
        drop(db);
        let mut reader =
            Connection::open_with_flags(&path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let read = reader.transaction()?;
        assert_eq!(
            read.query_row("SELECT value FROM wal_fixture", [], |r| r
                .get::<_, String>(0))?,
            "original"
        );
        fs::write(root.join("uncommitted.start"), b"start")?;
        wait_file(&root.join("uncommitted.mutated"))?;
        assert_eq!(wal_bytes(&read)?, 0, "uncommitted pages must not spill");
        fs::write(root.join("committed.start"), b"start")?;
        wait_file(&root.join("committed.attempt"))?;
        assert!(!root.join("committed.mutated").exists());
        uncommitted.0.kill()?;
        assert_eq!(uncommitted.0.wait()?.signal(), Some(9));
        wait_file(&root.join("committed.committed"))?;
        let physical = wal_bytes(&read)?;
        assert!(physical > 0 && physical <= cap);
        committed.0.kill()?;
        assert_eq!(committed.0.wait()?.signal(), Some(9));
        assert_eq!(
            read.query_row("SELECT value FROM wal_fixture", [], |r| r
                .get::<_, String>(0))?,
            "original"
        );
        read.commit()?;
        drop(reader);
        let recovered = store::open(&path, StoreLimits::default())?;
        assert_eq!(
            recovered
                .writer()?
                .query_row("SELECT value FROM wal_fixture", [], |r| r
                    .get::<_, String>(0))?,
            "committed"
        );
        assert_eq!(
            recovered
                .writer()?
                .query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
            "ok"
        );
        assert!(!checkpoint(&recovered)?.busy);
        assert_eq!(wal_bytes(&*recovered.writer()?)?, 0);
        write_reserved(&recovered, cap, cap, 0, |tx| {
            tx.execute("UPDATE wal_fixture SET value='resumed'", [])?;
            Ok(())
        })?;
        println!(
            "WAL process recovery: page_size={size}, cap={cap}, committed_wal={physical}, killed_writers=2, integrity=ok"
        );
    }
    Ok(())
}
