//! Public init verifies a recording provider before any configuration/key publication.
#![cfg(unix)]
use assert_cmd::Command;
use serde_json::{Value, json};
use std::{fs, os::unix::fs::PermissionsExt, path::Path};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, method, path},
};

fn command(root: &Path, user: &Path) -> Command {
    let mut command = Command::new(assert_cmd::cargo::cargo_bin!("dkc"));
    command.current_dir(root).env("XDG_CONFIG_HOME", user);
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().starts_with("DKC_") {
            command.env_remove(name);
        }
    }
    command
        .env_remove("MOONLY_GATEWAY_API_KEY")
        .env("DKC_LOG", "trace");
    command
}

#[tokio::test]
async fn public_remote_init_verifies_then_stores_external_key_and_reuses_it_without_rewriting_config()
 {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("repo");
    let user = directory.path().join("user");
    fs::create_dir_all(root.join(".git/info")).unwrap();
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("authorization", "Bearer fixture"))
        .and(header(
            "user-agent",
            concat!("dkc/", env!("CARGO_PKG_VERSION")),
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"data":[{"id":"embed"},{"id":"rank"}]})),
        )
        .expect(2)
        .mount(&server)
        .await;
    let endpoint = format!("{}/v1", server.uri());
    let first_root = root.clone();
    let first_user = user.clone();
    let output = tokio::task::spawn_blocking(move || {
        command(&first_root, &first_user)
            .args([
                "init",
                "--profile",
                "remote",
                "--url",
                &endpoint,
                "--model",
                "embed",
                "--rerank-model",
                "rank",
                "--api-key-stdin",
                "--json",
            ])
            .write_stdin("fixture\n")
            .output()
            .unwrap()
    })
    .await
    .unwrap();
    assert!(!String::from_utf8_lossy(&output.stdout).contains("fixture"));
    assert!(!String::from_utf8_lossy(&output.stderr).contains("fixture"));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["verified_endpoints"], 1);
    assert_eq!(result["credentials_written"], 1);
    let config_path = root.join(".deepkit/config.toml");
    let config = fs::read(&config_path).unwrap();
    assert!(!String::from_utf8_lossy(&config).contains("fixture"));
    let credentials = user.join("deepkit-code/credentials.toml");
    assert_eq!(
        fs::metadata(&credentials).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert!(!root.join(".deepkit/code.db").exists());
    let output = tokio::task::spawn_blocking(move || {
        command(&root, &user)
            .args(["init", "--json"])
            .output()
            .unwrap()
    })
    .await
    .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["configuration_written"], false);
    assert_eq!(result["credentials_written"], 0);
    assert_eq!(fs::read(config_path).unwrap(), config);
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn rejected_public_init_preserves_config_credentials_and_excludes() {
    for status in [401, 503] {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("repo");
        let user = directory.path().join("user");
        fs::create_dir_all(root.join(".git/info")).unwrap();
        let exclude = root.join(".git/info/exclude");
        fs::write(&exclude, "# existing user excludes\n").unwrap();
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(
                ResponseTemplate::new(status)
                    .set_body_string("fixture must never appear in diagnostics"),
            )
            .expect(1)
            .mount(&server)
            .await;
        let endpoint = format!("{}/v1", server.uri());
        let operation_root = root.clone();
        let operation_user = user.clone();
        let output = tokio::task::spawn_blocking(move || {
            command(&operation_root, &operation_user)
                .args([
                    "init",
                    "--profile",
                    "remote",
                    "--url",
                    &endpoint,
                    "--model",
                    "embed",
                    "--cursor",
                    "--claude",
                    "--api-key-stdin",
                    "--json",
                ])
                .write_stdin("fixture\n")
                .output()
                .unwrap()
        })
        .await
        .unwrap();
        assert!(!output.status.success());
        assert!(!String::from_utf8_lossy(&output.stderr).contains("fixture"));
        assert!(!String::from_utf8_lossy(&output.stdout).contains("fixture"));
        assert_eq!(
            fs::read_to_string(exclude).unwrap(),
            "# existing user excludes\n"
        );
        assert!(!root.join(".cursor").exists());
        assert!(!root.join(".mcp.json").exists());
        assert!(!root.join(".deepkit").exists());
        assert!(!user.join("deepkit-code/credentials.toml").exists());
    }
}

#[test]
fn public_init_installs_only_requested_ide_entries_and_preserves_existing_servers() {
    for flags in [
        vec!["--cursor"],
        vec!["--claude"],
        vec!["--cursor", "--claude"],
    ] {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("repo");
        let user = directory.path().join("user");
        fs::create_dir_all(root.join(".git/info")).unwrap();
        for (flag, relative) in [("--cursor", ".cursor/mcp.json"), ("--claude", ".mcp.json")] {
            if flags.contains(&flag) {
                let path = root.join(relative);
                fs::create_dir_all(path.parent().unwrap()).unwrap();
                fs::write(path, br#"{"mcpServers":{"existing":{"command":"kept","args":["argument"]}},"setting":true}"#).unwrap();
            }
        }
        let mut args = vec!["init", "--yes", "--json"];
        args.extend(flags.iter().copied());
        let output = command(&root, &user).args(&args).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let result: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["mcp"].as_array().unwrap().len(), flags.len());
        let mut snapshots = Vec::new();
        for (flag, relative) in [("--cursor", ".cursor/mcp.json"), ("--claude", ".mcp.json")] {
            let path = root.join(relative);
            assert_eq!(path.exists(), flags.contains(&flag));
            if path.exists() {
                let bytes = fs::read(&path).unwrap();
                let value: Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(value["setting"], true);
                assert_eq!(
                    value["mcpServers"]["existing"],
                    json!({"command":"kept", "args":["argument"]})
                );
                assert_eq!(
                    value["mcpServers"]["deepkit-code"],
                    json!({"command":"dkc", "args":["serve", "--stdio"]})
                );
                snapshots.push((path, bytes));
            }
        }
        command(&root, &user).args(&args).assert().success();
        for (path, bytes) in snapshots {
            assert_eq!(fs::read(path).unwrap(), bytes);
        }
        assert!(!root.join(".deepkit/code.db").exists());
    }
}

#[test]
fn public_init_refuses_conflicting_ide_entry_before_any_setup_write() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("repo");
    let user = directory.path().join("user");
    fs::create_dir_all(root.join(".git/info")).unwrap();
    let original = br#"{"mcpServers":{"deepkit-code":{"command":"custom"}}}"#;
    fs::write(root.join(".mcp.json"), original).unwrap();
    let output = command(&root, &user)
        .args(["init", "--yes", "--cursor", "--claude", "--json"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!root.join(".cursor").exists());
    assert!(!root.join(".deepkit").exists());
    assert!(!root.join(".git/info/exclude").exists());
    assert_eq!(fs::read(root.join(".mcp.json")).unwrap(), original);
}

#[tokio::test]
async fn generated_ide_command_initializes_mcp_in_the_selected_repository() {
    use rmcp::{ServiceExt, model::CallToolRequestParams};
    use std::{process::Stdio, time::Duration};
    use tokio::time::timeout;
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("repo");
    let user = directory.path().join("user");
    fs::create_dir_all(&root).unwrap();
    assert!(
        std::process::Command::new("git")
            .current_dir(&root)
            .args(["init", "--quiet"])
            .status()
            .unwrap()
            .success()
    );
    fs::write(root.join("lib.rs"), "pub fn installed() -> bool { true }\n").unwrap();
    command(&root, &user)
        .args(["init", "--yes", "--cursor", "--claude"])
        .assert()
        .success();
    command(&root, &user)
        .args(["index", "--level", "1"])
        .assert()
        .success();
    let cursor: Value =
        serde_json::from_slice(&fs::read(root.join(".cursor/mcp.json")).unwrap()).unwrap();
    let claude: Value = serde_json::from_slice(&fs::read(root.join(".mcp.json")).unwrap()).unwrap();
    assert_eq!(cursor, claude);
    let entry = &cursor["mcpServers"]["deepkit-code"];
    let binary = Path::new(assert_cmd::cargo::cargo_bin!("dkc"));
    let mut paths = vec![binary.parent().unwrap().to_path_buf()];
    paths.extend(std::env::split_paths(
        &std::env::var_os("PATH").unwrap_or_default(),
    ));
    let mut launch = tokio::process::Command::new(entry["command"].as_str().unwrap());
    launch
        .args(
            entry["args"]
                .as_array()
                .unwrap()
                .iter()
                .map(|arg| arg.as_str().unwrap()),
        )
        .current_dir(&root)
        .env("PATH", std::env::join_paths(paths).unwrap())
        .env("XDG_CONFIG_HOME", &user)
        .env(
            "DKC_MODELS_DIR",
            directory.path().join("absent-model-cache"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().starts_with("DKC_") {
            launch.env_remove(name);
        }
    }
    launch.env(
        "DKC_MODELS_DIR",
        directory.path().join("absent-model-cache"),
    );
    let mut child = launch.spawn().unwrap();
    let transport = (child.stdout.take().unwrap(), child.stdin.take().unwrap());
    let client = timeout(Duration::from_secs(10), ().serve(transport))
        .await
        .unwrap()
        .unwrap();
    let response = timeout(
        Duration::from_secs(10),
        client.call_tool(CallToolRequestParams::new("status")),
    )
    .await
    .unwrap()
    .unwrap();
    assert_ne!(response.is_error, Some(true));
    client.cancel().await.unwrap();
    assert!(
        timeout(Duration::from_secs(5), child.wait())
            .await
            .unwrap()
            .unwrap()
            .success()
    );
}
