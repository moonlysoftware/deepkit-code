//! HTTP endpoint conflicts fail before automatic repository setup.
#![cfg(unix)]

use std::{env, fs, net::TcpListener, path::Path, time::Duration};
use tokio::{process::Command, time::timeout};

fn command(root: &Path, user_config: &Path) -> Command {
    let mut command = Command::new(assert_cmd::cargo::cargo_bin!("dkc"));
    for (name, _) in env::vars_os() {
        if name.to_string_lossy().starts_with("DKC_") {
            command.env_remove(name);
        }
    }
    command
        .current_dir(root)
        .env("XDG_CONFIG_HOME", user_config)
        .env("DKC_MODELS_DIR", root.join(".deepkit/absent-models"))
        .env_remove("MOONLY_GATEWAY_API_KEY")
        .kill_on_drop(true);
    command
}

#[tokio::test]
async fn occupied_configured_and_explicit_endpoints_do_not_initialize_repository() {
    let held = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = held.local_addr().unwrap().to_string();
    for explicit in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let user = tempfile::tempdir().unwrap();
        let git = Command::new("git")
            .args(["init", "--quiet"])
            .current_dir(root.path())
            .output()
            .await
            .unwrap();
        assert!(
            git.status.success(),
            "{}",
            String::from_utf8_lossy(&git.stderr)
        );
        let exclude = root.path().join(".git/info/exclude");
        let previous_exclude = fs::read(&exclude).unwrap();
        let configured = if explicit { "127.0.0.1:0" } else { &address };
        fs::create_dir(user.path().join("deepkit-code")).unwrap();
        fs::write(
            user.path().join("deepkit-code/config.toml"),
            format!(
                "[serve]\nhttp='{configured}'\n[embedder]\nprofile='local'\n[llm]\nenabled=false\n[rerank]\nmode='off'\n"
            ),
        )
        .unwrap();
        let mut cmd = command(root.path(), user.path());
        cmd.args(["serve", "--http"]);
        if explicit {
            cmd.arg(&address);
        }
        let output = timeout(Duration::from_secs(15), cmd.output())
            .await
            .expect("occupied endpoint did not fail promptly")
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{stderr}");
        assert!(stderr.contains(&address), "{stderr}");
        assert!(stderr.contains("already in use"), "{stderr}");
        assert!(stderr.contains("dkc serve --stdio"), "{stderr}");
        assert!(stderr.contains("different explicit"), "{stderr}");
        assert!(stderr.contains("client's MCP URL"), "{stderr}");
        assert!(output.stdout.is_empty());
        assert!(!root.path().join(".deepkit").exists());
        assert_eq!(fs::read(&exclude).unwrap(), previous_exclude);
    }
}
