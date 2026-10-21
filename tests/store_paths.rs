//! Configured SQLite destinations stay inside the repository; v1 §2.1 and §3.
#![cfg(unix)]

use std::{fs, os::unix::fs::symlink, path::Path, process::Command};

const COMMANDS: &[&[&str]] = &[
    &["status", "--json"],
    &["index", "--level", "1", "--no-queue", "--json"],
    &[
        "search",
        "needle",
        "--mode",
        "lexical",
        "--no-rerank",
        "--json",
    ],
];

const PROBE: &[&str] = &[
    "bench",
    "probe",
    "--root",
    ".",
    "--out",
    "probe.json",
    "--json",
];

fn refusal_commands() -> impl Iterator<Item = &'static [&'static str]> {
    COMMANDS.iter().copied().chain(std::iter::once(PROBE))
}

fn config(root: &Path, database: &Path) {
    fs::create_dir_all(root.join(".deepkit")).unwrap();
    let value = toml::Value::String(database.to_string_lossy().into_owned());
    fs::write(
        root.join(".deepkit/config.toml"),
        format!("[store]\npath={value}\n"),
    )
    .unwrap();
}

fn run(root: &Path, args: &[&str]) -> std::process::Output {
    Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .args(args)
        .output()
        .unwrap()
}

fn assert_refused(root: &Path, args: &[&str]) {
    let output = run(root, args);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "{args:?} unexpectedly succeeded");
    assert!(stderr.contains("database"), "{args:?}: {stderr}");
    assert!(!root.join("probe.json").exists());
}

fn unrelated_database(path: &Path) -> Vec<u8> {
    let connection = rusqlite::Connection::open(path).unwrap();
    connection
        .execute_batch("CREATE TABLE unrelated(value TEXT); INSERT INTO unrelated VALUES ('keep');")
        .unwrap();
    drop(connection);
    fs::read(path).unwrap()
}

#[test]
fn public_commands_refuse_external_paths_before_creating_or_migrating_sqlite() {
    let base = tempfile::tempdir().unwrap();
    let root = base.path().join("repo");
    fs::create_dir(&root).unwrap();
    let outside = base.path().join("outside.db");
    for configured in [
        outside.clone(),
        "../outside.db".into(),
        ".deepkit/../../outside.db".into(),
    ] {
        config(&root, &configured);
        for args in refusal_commands() {
            assert_refused(&root, args);
            assert!(!outside.exists(), "{args:?} created an external database");
        }
        let before = unrelated_database(&outside);
        for args in refusal_commands() {
            assert_refused(&root, args);
            assert_eq!(fs::read(&outside).unwrap(), before, "{args:?}");
            assert!(!base.path().join("outside.db-wal").exists());
            assert!(!base.path().join("outside.db-shm").exists());
            assert!(!base.path().join("outside.db-journal").exists());
        }
        fs::remove_file(&outside).unwrap();
    }
    // Refuse an absent external parent before create_dir_all can create it.
    config(&root, &base.path().join("absent/outside.db"));
    for args in refusal_commands() {
        assert_refused(&root, args);
        assert!(!base.path().join("absent").exists());
    }
}

#[test]
fn public_commands_refuse_linked_ancestry_without_touching_external_sqlite() {
    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    config(root.path(), Path::new(".deepkit/linked/custom.db"));
    symlink(outside.path(), root.path().join(".deepkit/linked")).unwrap();
    let database = outside.path().join("custom.db");
    for args in refusal_commands() {
        assert_refused(root.path(), args);
        assert!(!database.exists());
    }
    let before = unrelated_database(&database);
    for args in refusal_commands() {
        assert_refused(root.path(), args);
        assert_eq!(fs::read(&database).unwrap(), before);
        assert_eq!(fs::read_dir(outside.path()).unwrap().count(), 1);
    }
}

#[test]
fn public_commands_refuse_linked_database_and_sidecars_before_sqlite_setup() {
    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let database = root.path().join(".deepkit/custom.db");
    let external = outside.path().join("unrelated.db");
    let before = unrelated_database(&external);
    config(root.path(), Path::new(".deepkit/custom.db"));

    for suffix in ["", "-wal", "-shm", "-journal"] {
        let destination = root.path().join(format!(".deepkit/custom.db{suffix}"));
        symlink(&external, &destination).unwrap();
        for args in refusal_commands() {
            assert_refused(root.path(), args);
            assert_eq!(fs::read(&external).unwrap(), before);
            if !suffix.is_empty() {
                assert!(!database.exists());
            }
        }
        fs::remove_file(&destination).unwrap();
        fs::hard_link(&external, &destination).unwrap();
        for args in refusal_commands() {
            assert_refused(root.path(), args);
            assert_eq!(fs::read(&external).unwrap(), before);
            if !suffix.is_empty() {
                assert!(!database.exists());
            }
        }
        fs::remove_file(&destination).unwrap();
    }
    assert_eq!(fs::read_dir(outside.path()).unwrap().count(), 1);
}

#[test]
fn public_commands_preserve_custom_database_paths_inside_deepkit() {
    let root = tempfile::tempdir().unwrap();
    for configured in [
        ".deepkit/nested/relative.db".into(),
        root.path().join(".deepkit/nested/absolute.db"),
    ] {
        config(root.path(), &configured);
        for args in COMMANDS {
            let output = run(root.path(), args);
            assert!(
                output.status.success(),
                "{args:?}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        assert!(root.path().join(configured).is_file());
    }
}
