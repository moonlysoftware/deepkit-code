//! Public opt-in hook ownership and quick L1 behavior; v1.md §4.5.
#![cfg(unix)]
use assert_cmd::Command;
use serde_json::Value;
use std::{
    fs,
    os::unix::fs::{PermissionsExt, symlink},
    path::Path,
};

const HOOKS: [&str; 4] = ["post-commit", "post-checkout", "post-merge", "post-rewrite"];
fn git(root: &Path, args: &[&str]) {
    assert!(
        std::process::Command::new("git")
            .current_dir(root)
            .args(args)
            .output()
            .unwrap()
            .status
            .success()
    );
}
fn repo() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    git(root.path(), &["init", "-q"]);
    root
}
fn dkc(root: &Path, args: &[&str], success: bool) -> Value {
    let mut command = Command::new(assert_cmd::cargo::cargo_bin!("dkc"));
    command
        .current_dir(root)
        .env("XDG_CONFIG_HOME", root.join("empty-config"))
        .args(args);
    let assertion = command.assert();
    let assertion = if success {
        assertion.success()
    } else {
        assertion.failure()
    };
    if success {
        serde_json::from_slice(&assertion.get_output().stdout).unwrap()
    } else {
        Value::Null
    }
}
#[test]
fn hooks_are_idempotent_executable_and_preserve_existing_files() {
    let root = repo();
    let root = root.path();
    let sample = fs::read(root.join(".git/hooks/pre-commit.sample")).unwrap();
    assert_eq!(
        dkc(root, &["hooks", "install", "--json"], true)["changed"]
            .as_array()
            .unwrap()
            .len(),
        4
    );
    assert_eq!(
        dkc(root, &["hooks", "install", "--json"], true)["changed"],
        serde_json::json!([])
    );
    let bin = tempfile::tempdir().unwrap();
    fs::write(
        bin.path().join("dkc"),
        "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$HOOK_CAPTURE\"\nexit 9\n",
    )
    .unwrap();
    fs::set_permissions(bin.path().join("dkc"), fs::Permissions::from_mode(0o755)).unwrap();
    let capture = bin.path().join("calls");
    for hook in HOOKS {
        let path = root.join(".git/hooks").join(hook);
        assert_ne!(fs::metadata(&path).unwrap().permissions().mode() & 0o111, 0);
        let output = std::process::Command::new(path)
            .current_dir(root)
            .env("PATH", bin.path())
            .env("HOOK_CAPTURE", &capture)
            .args(["git-argument", "ignored"])
            .output()
            .unwrap();
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("quick index failed"));
    }
    assert_eq!(
        fs::read_to_string(capture).unwrap(),
        "index --quick\n".repeat(4)
    );
    assert_eq!(
        dkc(root, &["hooks", "remove", "--json"], true)["changed"]
            .as_array()
            .unwrap()
            .len(),
        4
    );
    assert_eq!(
        dkc(root, &["hooks", "remove", "--json"], true)["changed"],
        serde_json::json!([])
    );
    assert_eq!(
        sample,
        fs::read(root.join(".git/hooks/pre-commit.sample")).unwrap()
    );
    fs::write(
        root.join(".git/hooks/post-rewrite"),
        "#!/bin/sh\n# human hook\n",
    )
    .unwrap();
    dkc(root, &["hooks", "install", "--json"], false);
    assert!(!root.join(".git/hooks/post-commit").exists());
    assert_eq!(
        fs::read_to_string(root.join(".git/hooks/post-rewrite")).unwrap(),
        "#!/bin/sh\n# human hook\n"
    );
}
#[test]
fn hook_managers_custom_paths_and_modified_owned_hooks_are_refused() {
    for marker in [
        ".husky",
        "lefthook.yml",
        ".pre-commit-config.yaml",
        "package.json",
    ] {
        let root = repo();
        fs::write(
            root.path().join(marker),
            if marker == "package.json" {
                "{\"husky\":{}}"
            } else {
                "managed"
            },
        )
        .unwrap();
        dkc(root.path(), &["hooks", "install", "--json"], false);
        assert!(!root.path().join(".git/hooks/post-commit").exists());
    }
    let root = repo();
    git(root.path(), &["config", "core.hooksPath", "custom-hooks"]);
    dkc(root.path(), &["hooks", "install", "--json"], false);
    assert!(!root.path().join("custom-hooks").exists());
    git(root.path(), &["config", "--unset", "core.hooksPath"]);
    dkc(root.path(), &["hooks", "install", "--json"], true);
    let changed = root.path().join(".git/hooks/post-rewrite");
    let text = fs::read_to_string(&changed).unwrap() + "# edited\n";
    fs::write(&changed, &text).unwrap();
    dkc(root.path(), &["hooks", "remove", "--json"], false);
    assert_eq!(fs::read_to_string(changed).unwrap(), text);
    assert!(root.path().join(".git/hooks/post-commit").exists());
}
#[test]
fn symlink_destinations_are_not_followed() {
    let root = repo();
    let outside = tempfile::tempdir().unwrap();
    let target = outside.path().join("hook");
    fs::write(&target, "untouched").unwrap();
    symlink(&target, root.path().join(".git/hooks/post-commit")).unwrap();
    dkc(root.path(), &["hooks", "install", "--json"], false);
    dkc(root.path(), &["hooks", "remove", "--json"], false);
    assert_eq!(fs::read_to_string(target).unwrap(), "untouched");
    fs::remove_file(root.path().join(".git/hooks/post-commit")).unwrap();
    fs::rename(
        root.path().join(".git/hooks"),
        root.path().join(".git/old-hooks"),
    )
    .unwrap();
    symlink(outside.path(), root.path().join(".git/hooks")).unwrap();
    dkc(root.path(), &["hooks", "install", "--json"], false);
    assert!(!outside.path().join("post-commit").exists());
}
#[tokio::test]
async fn quick_indexes_changes_without_provider_calls_or_queue_jobs() {
    let server = wiremock::MockServer::start().await;
    let root = repo();
    let root = root.path();
    fs::create_dir(root.join(".deepkit")).unwrap();
    fs::write(root.join(".deepkit/config.toml"),format!("[embedder]\nprofile='ollama'\ndims=2\n[embedder.ollama]\nurl='{}'\n[rerank]\nmode='off'\n",server.uri())).unwrap();
    fs::write(root.join("probe.rs"), "pub fn before_change() {}\n").unwrap();
    let first = dkc(root, &["index", "--quick", "--json"], true);
    assert_eq!(first["requested_level"], 1);
    assert_eq!(first["embedding"]["state"], "skipped");
    let unchanged = dkc(root, &["index", "--quick", "--json"], true);
    assert_eq!(unchanged["files_modified"], 0);
    fs::write(root.join("probe.rs"), "pub fn after_change() {}\n").unwrap();
    let changed = dkc(root, &["index", "--quick", "--json"], true);
    assert_eq!(changed["files_modified"], 1);
    let found = dkc(
        root,
        &[
            "search",
            "after_change",
            "--mode",
            "lexical",
            "--no-rerank",
            "--json",
        ],
        true,
    );
    assert!(!found["hits"].as_array().unwrap().is_empty());
    assert!(server.received_requests().await.unwrap().is_empty());
    let db = rusqlite::Connection::open(root.join(".deepkit/code.db")).unwrap();
    assert_eq!(
        db.query_row("SELECT count(*) FROM jobs", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    for args in [
        vec!["index", "--quick", "--full"],
        vec!["index", "--quick", "--level", "2"],
    ] {
        dkc(root, &args, false);
    }
}

#[test]
fn linked_worktree_uses_shared_hooks_and_missing_binary_does_not_fail_git() {
    let root = repo();
    git(
        root.path(),
        &[
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "base",
        ],
    );
    let parent = tempfile::tempdir().unwrap();
    let linked = parent.path().join("linked");
    git(
        root.path(),
        &["worktree", "add", "--detach", linked.to_str().unwrap()],
    );
    dkc(&linked, &["hooks", "install", "--json"], true);
    let empty = tempfile::tempdir().unwrap();
    for hook in HOOKS {
        let output = std::process::Command::new(root.path().join(".git/hooks").join(hook))
            .current_dir(&linked)
            .env("PATH", empty.path())
            .output()
            .unwrap();
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("executable missing"));
    }
    dkc(&linked, &["hooks", "remove", "--json"], true);
    assert!(!root.path().join(".git/hooks/post-commit").exists());
}
