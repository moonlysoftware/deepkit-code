//! Public cleanup stays inside its admitted index directory; v1 §2.1 and §10.2.
#![cfg(unix)]

use std::{fs, os::unix::fs::symlink, path::Path, process::Command};

fn config(root: &Path, database: &Path) {
    fs::create_dir_all(root.join(".deepkit")).unwrap();
    let value = toml::Value::String(database.to_string_lossy().into_owned());
    fs::write(
        root.join(".deepkit/config.toml"),
        format!("[store]\npath={value}\n"),
    )
    .unwrap();
}

fn clean(root: &Path) -> std::process::Output {
    Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .args(["clean", "--json"])
        .output()
        .unwrap()
}

#[test]
fn clean_refuses_external_absolute_and_parent_traversal_paths() {
    let base = tempfile::tempdir().unwrap();
    let root = base.path().join("repo");
    fs::create_dir(&root).unwrap();
    let outside = base.path().join("untouched");
    for path in [
        outside.clone(),
        "../untouched".into(),
        ".deepkit/../../untouched".into(),
    ] {
        fs::write(&outside, "keep").unwrap();
        config(&root, &path);
        let output = clean(&root);
        assert!(!output.status.success());
        assert_eq!(fs::read_to_string(&outside).unwrap(), "keep");
    }
}

#[test]
fn clean_refuses_linked_ancestry_and_sidecars_before_deleting_anything() {
    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let external = outside.path().join("custom.db");
    fs::write(&external, "keep").unwrap();
    config(root.path(), Path::new(".deepkit/linked/custom.db"));
    symlink(outside.path(), root.path().join(".deepkit/linked")).unwrap();
    assert!(!clean(root.path()).status.success());
    assert_eq!(fs::read_to_string(&external).unwrap(), "keep");

    config(root.path(), Path::new(".deepkit/custom.db"));
    let database = root.path().join(".deepkit/custom.db");
    fs::write(&database, "owned database").unwrap();
    let wal = root.path().join(".deepkit/custom.db-wal");
    symlink(&external, &wal).unwrap();
    assert!(!clean(root.path()).status.success());
    assert_eq!(fs::read_to_string(&database).unwrap(), "owned database");
    assert_eq!(fs::read_to_string(&external).unwrap(), "keep");
    fs::remove_file(&wal).unwrap();
    fs::hard_link(&external, &wal).unwrap();
    assert!(!clean(root.path()).status.success());
    assert_eq!(fs::read_to_string(&database).unwrap(), "owned database");
    assert_eq!(fs::read_to_string(&external).unwrap(), "keep");
}

#[test]
fn clean_removes_custom_owned_database_and_sidecars_without_creating_missing_paths() {
    let root = tempfile::tempdir().unwrap();
    config(root.path(), Path::new(".deepkit/nested/custom.db"));
    let output = clean(root.path());
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!root.path().join(".deepkit/nested").exists());
    fs::create_dir(root.path().join(".deepkit/nested")).unwrap();
    for suffix in ["", "-wal", "-shm"] {
        fs::write(
            root.path()
                .join(format!(".deepkit/nested/custom.db{suffix}")),
            "owned",
        )
        .unwrap();
    }
    // An absolute custom path inside the admitted directory is also valid.
    config(root.path(), &root.path().join(".deepkit/nested/custom.db"));
    let output = clean(root.path());
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let response: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(response["removed"].as_array().unwrap().len(), 3);
    assert!(root.path().join(".deepkit/config.toml").is_file());
    assert_eq!(
        fs::read_dir(root.path().join(".deepkit/nested"))
            .unwrap()
            .count(),
        0
    );
}
