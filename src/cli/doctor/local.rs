//! Bounded repository configuration and HEAD diagnostics; v1 §§13, 16.
use super::Finding;
use crate::{config, sync};
use std::path::Path;

/// Collected local findings plus usable effective settings for independent probes.
pub(super) struct Local {
    pub(super) findings: Vec<Finding>,
    pub(super) config: Option<config::Config>,
    pub(super) explicit_treatments: Option<bool>,
}

fn finding(id: &str, status: &'static str, message: &str) -> Finding {
    Finding {
        id: id.to_owned(),
        status,
        message: message.to_owned(),
        evidence: None,
    }
}

struct Layer {
    value: Option<toml::Value>,
    usable: bool,
}

fn layer(root: &Path, name: &'static str, id: &'static str, out: &mut Vec<Finding>) -> Layer {
    // No reopening through the ordinary repository config loader: these bounded,
    // descriptor-relative snapshots reject symlinks, FIFOs and unsafe ancestry.
    let text = match sync::setup::read_configuration(root, name) {
        Ok(text) => text,
        Err(_) => {
            out.push(finding(
                id,
                "failed",
                "Repository configuration could not be read safely",
            ));
            return Layer {
                value: None,
                usable: false,
            };
        }
    };
    let Some(text) = text else {
        out.push(finding(
            id,
            "passed",
            "Optional repository configuration is absent",
        ));
        return Layer {
            value: None,
            usable: true,
        };
    };
    let value: toml::Value = match toml::from_str(&text) {
        Ok(value) => value,
        Err(_) => {
            out.push(finding(
                id,
                "failed",
                "Repository configuration is not valid TOML",
            ));
            out.push(finding(
                &format!("{id}.secrets"),
                "not_checked",
                "Secret-field check requires valid TOML",
            ));
            out.push(finding(
                &format!("{id}.unknown_keys"),
                "not_checked",
                "Unknown-key check requires valid TOML",
            ));
            return Layer {
                value: None,
                usable: false,
            };
        }
    };
    out.push(finding(
        id,
        "passed",
        "Repository configuration TOML is readable",
    ));
    let secret = config::contains_secret(&value);
    out.push(if secret {
        finding(
            &format!("{id}.secrets"),
            "failed",
            "Repository configuration contains a secret field; move keys to external credentials",
        )
    } else {
        finding(
            &format!("{id}.secrets"),
            "passed",
            "No secret fields found in repository configuration",
        )
    });
    let lint_id = format!("{id}.unknown_keys");
    match config::lint_document(&value, &root.join(".deepkit").join(name)) {
        Ok(warnings) => {
            let mut item = if warnings.is_empty() {
                finding(&lint_id, "passed", "No unknown configuration keys found")
            } else {
                finding(
                    &lint_id,
                    "warning",
                    "Unknown configuration keys found; review repository configuration",
                )
            };
            // Keys and paths are user-controlled strings and can themselves
            // contain secrets. Emit only a count, never the raw lint records.
            item.evidence = Some(serde_json::json!({"count": warnings.len()}));
            out.push(item);
        }
        Err(_) => out.push(finding(
            &lint_id,
            "failed",
            "Configuration key lint could not complete",
        )),
    }
    Layer {
        value: Some(value),
        usable: !secret,
    }
}

/// Collects independent checks without exposing config, parser or Git error text.
pub(super) fn collect(root: &Path) -> Local {
    let mut findings = Vec::new();
    let mut explicit_treatments = None;
    let repository = layer(root, "config.toml", "config.repository", &mut findings);
    let local = layer(root, "config.local.toml", "config.local", &mut findings);
    let config = if repository.usable && local.usable {
        match config::load_init_snapshots(root, repository.value, local.value) {
            Ok(loaded) => {
                explicit_treatments = loaded
                    .explain("treatments")
                    .map(|layer| layer != config::Layer::Defaults);
                findings.push(finding(
                    "config.effective",
                    "passed",
                    "Effective configuration is valid",
                ));
                let mut lint = finding(
                    "config.effective.unknown_keys",
                    if loaded.warnings.is_empty() {
                        "passed"
                    } else {
                        "warning"
                    },
                    if loaded.warnings.is_empty() {
                        "No unknown keys found across effective configuration layers"
                    } else {
                        "Unknown keys found across effective configuration layers"
                    },
                );
                lint.evidence = Some(serde_json::json!({"count": loaded.warnings.len()}));
                findings.push(lint);
                Some(loaded.config)
            }
            Err(_) => {
                findings.push(finding(
                    "config.effective",
                    "failed",
                    "Effective configuration is invalid or unavailable",
                ));
                None
            }
        }
    } else {
        findings.push(finding(
            "config.effective",
            "not_checked",
            "Effective configuration requires safe, valid repository settings",
        ));
        None
    };
    // This helper never loads the Git index. It resolves HEAD identity; it does
    // not verify every reachable object, and a missing/unborn HEAD is explicit.
    let git_root = config.as_ref().map(|config| root.join(&config.index.root));
    let git_root = git_root.as_deref().unwrap_or(root);
    findings.push(
        match sync::head::read(git_root, sync::l1::Work::default()) {
            Ok(Some(_)) => finding("git.head", "passed", "Git HEAD resolves successfully"),
            Ok(None) => finding(
                "git.head",
                "warning",
                "No committed Git HEAD is available; repository may be unborn or non-Git",
            ),
            Err(_) => finding("git.head", "failed", "Git HEAD could not be read"),
        },
    );
    if let Some(head) = findings.last_mut() {
        head.evidence = Some(
            serde_json::json!({"scope": if config.is_some() { "configured_repository" } else { "invocation_repository_fallback" }}),
        );
    }
    Local {
        findings,
        config,
        explicit_treatments,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, process::Command};

    fn git(root: &Path, args: &[&str]) {
        assert!(
            Command::new("git")
                .arg("-C")
                .arg(root)
                .args(args)
                .env("GIT_CONFIG_NOSYSTEM", "1")
                .env("GIT_CONFIG_GLOBAL", "/dev/null")
                .output()
                .unwrap()
                .status
                .success()
        );
    }
    fn fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        git(dir.path(), &["init", "-q"]);
        git(
            dir.path(),
            &[
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--allow-empty",
                "-qm",
                "fixture",
            ],
        );
        fs::create_dir(dir.path().join(".deepkit")).unwrap();
        dir
    }
    fn item<'a>(local: &'a Local, id: &str) -> &'a Finding {
        local.findings.iter().find(|entry| entry.id == id).unwrap()
    }
    fn rendered(local: &Local) -> String {
        // Serialize precisely the externally visible fields, not effective config.
        local
            .findings
            .iter()
            .map(|entry| {
                format!(
                    "{} {} {} {:?}",
                    entry.id, entry.status, entry.message, entry.evidence
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn malformed_config_does_not_hide_git_or_other_layer_lint() {
        let dir = fixture();
        let marker = format!(
            "synthetic-{}",
            dir.path().file_name().unwrap().to_string_lossy()
        );
        fs::write(
            dir.path().join(".deepkit/config.toml"),
            format!("broken = '{marker}\n"),
        )
        .unwrap();
        fs::write(
            dir.path().join(".deepkit/config.local.toml"),
            "[limits]\nunknown_limit=true\n",
        )
        .unwrap();
        // A malformed index is irrelevant to HEAD-only diagnosis.
        fs::write(dir.path().join(".git/index"), "invalid index").unwrap();
        let result = collect(dir.path());
        assert_eq!(item(&result, "config.repository").status, "failed");
        assert_eq!(item(&result, "config.local.unknown_keys").status, "warning");
        assert_eq!(item(&result, "git.head").status, "passed");
        assert!(result.config.is_none());
        assert!(!rendered(&result).contains(&marker));
    }

    #[test]
    fn secret_values_and_unknown_key_names_are_redacted() {
        let dir = fixture();
        let marker = format!(
            "synthetic-{}",
            dir.path().file_name().unwrap().to_string_lossy()
        );
        fs::write(
            dir.path().join(".deepkit/config.toml"),
            format!("[custom]\n'{marker}'=true\npassword='{marker}'\n"),
        )
        .unwrap();
        let result = collect(dir.path());
        assert_eq!(item(&result, "config.repository.secrets").status, "failed");
        assert_eq!(
            item(&result, "config.repository.unknown_keys").status,
            "warning"
        );
        assert_eq!(item(&result, "git.head").status, "passed");
        assert!(result.config.is_none());
        assert!(!rendered(&result).contains(&marker));
        assert!(!rendered(&result).contains("password"));
    }

    #[cfg(unix)]
    #[test]
    fn unsafe_local_config_is_refused_without_reading_or_writing_target() {
        let dir = fixture();
        let outside = tempfile::NamedTempFile::new().unwrap();
        let marker = format!(
            "synthetic-{}",
            dir.path().file_name().unwrap().to_string_lossy()
        );
        fs::write(outside.path(), &marker).unwrap();
        std::os::unix::fs::symlink(
            outside.path(),
            dir.path().join(".deepkit/config.local.toml"),
        )
        .unwrap();
        let result = collect(dir.path());
        assert_eq!(item(&result, "config.local").status, "failed");
        assert_eq!(item(&result, "git.head").status, "passed");
        assert_eq!(fs::read_to_string(outside.path()).unwrap(), marker);
        assert!(!rendered(&result).contains(&marker));
    }
    #[test]
    fn git_diagnosis_uses_the_configured_repository() {
        let first = fixture();
        let second = fixture();
        fs::write(second.path().join(".git/HEAD"), "invalid HEAD").unwrap();
        let mut settings = config::Config::default();
        settings.index.root = second.path().to_string_lossy().into_owned();
        fs::write(
            first.path().join(".deepkit/config.toml"),
            settings.dump().unwrap(),
        )
        .unwrap();
        let result = collect(first.path());
        assert_eq!(item(&result, "config.effective").status, "passed");
        assert_eq!(item(&result, "git.head").status, "failed");
        assert_eq!(
            item(&result, "git.head").message,
            "Git HEAD could not be read"
        );
        assert_eq!(
            item(&result, "git.head").evidence.as_ref().unwrap()["scope"],
            "configured_repository"
        );
    }
}
