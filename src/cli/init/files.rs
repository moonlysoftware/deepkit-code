//! CLI setup facade keeps IDE configuration merging outside the shared filesystem core; v1 §10.2.
use crate::sync::setup;
use anyhow::Result;
use std::path::Path;

pub(super) use setup::read_configuration;

/// Captures configuration and explicit exclude destinations without writing.
pub(super) fn prepare(
    root: &Path,
    gitignore: bool,
    original_config: Option<&str>,
    proposed_config: Option<&str>,
) -> Result<Prepared> {
    Ok(Prepared(setup::prepare(
        root,
        gitignore,
        original_config,
        proposed_config,
    )?))
}

/// CLI-owned facade delegates all filesystem validation and publication.
pub(super) struct Prepared(setup::Prepared);
impl Prepared {
    /// Adds only selected IDE entries, preserving unrelated server configurations.
    pub(super) fn with_mcp(mut self, cursor: bool, claude: bool) -> Result<Self> {
        for (selected, target) in [
            (cursor, setup::ExtraFile::Cursor),
            (claude, setup::ExtraFile::Claude),
        ] {
            if selected {
                self.0 = self.0.with_file(target, super::ide::merge_mcp)?;
            }
        }
        self.0.check()?;
        Ok(self)
    }
    /// Adds the explicitly selected Codex entry with a canonical repository cwd.
    pub(super) fn with_codex(mut self, root: &Path, selected: bool) -> Result<Self> {
        if selected {
            self.0 = self.0.with_file(setup::ExtraFile::Codex, |existing| {
                super::codex::merge(existing, root)
            })?;
        }
        self.0.check()?;
        Ok(self)
    }
    /// Rechecks all destinations before provider credentials are stored.
    pub(super) fn check(&self) -> Result<()> {
        Ok(self.0.check()?)
    }
    /// Publishes each file through the shared guarded atomic writer.
    pub(super) fn publish(&self) -> Result<()> {
        Ok(self.0.publish()?)
    }
}
#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::{fs, os::unix::fs::symlink};

    #[cfg(unix)]
    use crate::sync::source::mkfifo;

    #[test]
    fn mcp_publication_creates_only_requested_targets() {
        for (cursor, claude) in [(false, false), (true, false), (false, true), (true, true)] {
            let root = tempfile::tempdir().unwrap();
            let prepared = prepare(root.path(), false, None, Some("proposed"))
                .unwrap()
                .with_mcp(cursor, claude)
                .unwrap();
            prepared.check().unwrap();
            assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
            prepared.publish().unwrap();
            assert_eq!(root.path().join(".cursor").exists(), cursor);
            for (relative, requested) in [(".cursor/mcp.json", cursor), (".mcp.json", claude)] {
                let target = root.path().join(relative);
                assert_eq!(target.exists(), requested);
                if requested {
                    let document: serde_json::Value =
                        serde_json::from_slice(&fs::read(target).unwrap()).unwrap();
                    assert_eq!(
                        document,
                        serde_json::json!({"mcpServers":{"deepkit-code":{
                            "command":"dkc","args":["serve","--stdio"]
                        }}})
                    );
                }
            }
        }
    }

    #[test]
    fn codex_publication_is_explicit_preserving_and_idempotent() {
        for selected in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let original = b"# preserve these preferences\nmodel = 'example'\n";
            if selected {
                fs::create_dir(root.path().join(".codex")).unwrap();
                fs::write(root.path().join(".codex/config.toml"), original).unwrap();
            }
            let prepared = prepare(root.path(), false, None, Some("proposed"))
                .unwrap()
                .with_codex(root.path(), selected)
                .unwrap();
            assert!(!root.path().join(".deepkit").exists());
            prepared.publish().unwrap();
            assert_eq!(root.path().join(".codex").exists(), selected);
            if selected {
                let file = root.path().join(".codex/config.toml");
                let first = fs::read(&file).unwrap();
                assert!(first.starts_with(original));
                prepare(root.path(), false, Some("proposed"), None)
                    .unwrap()
                    .with_codex(root.path(), true)
                    .unwrap()
                    .publish()
                    .unwrap();
                assert_eq!(fs::read(file).unwrap(), first);
            }
            assert!(!root.path().join(".cursor").exists());
            assert!(!root.path().join(".mcp.json").exists());
        }
    }

    #[test]
    fn mcp_preflight_refuses_hostile_targets_without_blocking_or_writing() {
        use std::{sync::mpsc, time::Duration};
        for relative in [".cursor/mcp.json", ".mcp.json", ".codex/config.toml"] {
            for kind in ["symlink", "hardlink", "fifo", "oversize", "directory"] {
                let root = tempfile::tempdir().unwrap();
                let outside = tempfile::tempdir().unwrap();
                let target = root.path().join(relative);
                fs::create_dir_all(target.parent().unwrap()).unwrap();
                let external = outside.path().join("untouched");
                fs::write(&external, "unchanged").unwrap();
                match kind {
                    "symlink" => symlink(&external, &target).unwrap(),
                    "hardlink" => fs::hard_link(&external, &target).unwrap(),
                    "fifo" => {
                        mkfifo(&target, rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR).unwrap()
                    }
                    "directory" => fs::create_dir(&target).unwrap(),
                    _ => fs::File::create(&target)
                        .unwrap()
                        .set_len(1024 * 1024 + 1)
                        .unwrap(),
                }
                let path = root.path().to_path_buf();
                let (sender, receiver) = mpsc::channel();
                let worker = std::thread::spawn(move || {
                    let rejected = prepare(&path, false, None, Some("proposed"))
                        .unwrap()
                        .with_mcp(relative.starts_with(".cursor/"), relative == ".mcp.json")
                        .and_then(|prepared| {
                            prepared.with_codex(&path, relative == ".codex/config.toml")
                        })
                        .is_err();
                    sender.send(rejected).unwrap();
                });
                assert!(
                    receiver.recv_timeout(Duration::from_secs(2)).unwrap(),
                    "{relative}/{kind}"
                );
                worker.join().unwrap();
                assert_eq!(fs::read_to_string(&external).unwrap(), "unchanged");
                assert!(!root.path().join(".deepkit").exists());
                assert!(!root.path().join(".git").exists());
                assert_eq!(fs::read_dir(root.path()).unwrap().count(), 1);
                if relative.starts_with(".cursor/") {
                    assert_eq!(
                        fs::read_dir(root.path().join(".cursor")).unwrap().count(),
                        1
                    );
                }
            }
        }
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        symlink(outside.path(), root.path().join(".cursor")).unwrap();
        assert!(
            prepare(root.path(), false, None, Some("proposed"))
                .unwrap()
                .with_mcp(true, false)
                .is_err()
        );
        assert_eq!(fs::read_dir(outside.path()).unwrap().count(), 0);
        assert!(!root.path().join(".deepkit").exists());
    }

    #[test]
    fn codex_changed_snapshot_or_ancestry_refuses_all_publication() {
        for change in ["file", "ancestry"] {
            let root = tempfile::tempdir().unwrap();
            let outside = tempfile::tempdir().unwrap();
            let directory = root.path().join(".codex");
            let file = directory.join("config.toml");
            fs::create_dir(&directory).unwrap();
            fs::write(&file, "# original\n").unwrap();
            let prepared = prepare(root.path(), false, None, Some("proposed"))
                .unwrap()
                .with_mcp(true, true)
                .unwrap()
                .with_codex(root.path(), true)
                .unwrap();
            if change == "file" {
                fs::write(&file, "# concurrent edit\n").unwrap();
            } else {
                fs::rename(&directory, root.path().join("saved-codex")).unwrap();
                symlink(outside.path(), &directory).unwrap();
            }
            assert!(prepared.check().is_err());
            assert!(prepared.publish().is_err());
            assert_eq!(
                prepare(root.path(), false, None, Some("proposed"))
                    .unwrap()
                    .with_codex(root.path(), true)
                    .is_ok(),
                change == "file"
            );
            assert_eq!(fs::read_dir(outside.path()).unwrap().count(), 0);
            assert!(!root.path().join(".deepkit").exists());
            assert!(!root.path().join(".git").exists());
            assert!(!root.path().join(".cursor").exists());
            assert!(!root.path().join(".mcp.json").exists());
            let preserved = if change == "file" {
                file
            } else {
                root.path().join("saved-codex/config.toml")
            };
            assert_eq!(
                fs::read_to_string(preserved).unwrap(),
                if change == "file" {
                    "# concurrent edit\n"
                } else {
                    "# original\n"
                }
            );
        }
    }

    #[test]
    fn mcp_changed_snapshot_refuses_all_publication() {
        for relative in [".cursor/mcp.json", ".mcp.json"] {
            for initially_present in [false, true] {
                let root = tempfile::tempdir().unwrap();
                let target = root.path().join(relative);
                fs::create_dir_all(target.parent().unwrap()).unwrap();
                if initially_present {
                    fs::write(&target, "{}").unwrap();
                }
                let prepared = prepare(root.path(), false, None, Some("proposed"))
                    .unwrap()
                    .with_mcp(true, true)
                    .unwrap();
                let concurrent = "{\"mcpServers\":{},\"concurrent\":true}";
                fs::write(&target, concurrent).unwrap();
                assert!(prepared.check().is_err());
                assert!(prepared.publish().is_err());
                assert_eq!(fs::read_to_string(&target).unwrap(), concurrent);
                assert!(!root.path().join(".deepkit").exists());
                assert!(!root.path().join(".git").exists());
                let other = if relative == ".mcp.json" {
                    ".cursor/mcp.json"
                } else {
                    ".mcp.json"
                };
                assert!(!root.path().join(other).exists());
            }
        }
    }

    #[test]
    fn mcp_changed_cursor_ancestry_is_not_followed() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join(".cursor")).unwrap();
        fs::write(root.path().join(".cursor/mcp.json"), "{}").unwrap();
        let prepared = prepare(root.path(), false, None, Some("proposed"))
            .unwrap()
            .with_mcp(true, true)
            .unwrap();
        fs::rename(
            root.path().join(".cursor"),
            root.path().join("saved-cursor"),
        )
        .unwrap();
        symlink(outside.path(), root.path().join(".cursor")).unwrap();
        assert!(prepared.check().is_err());
        assert!(prepared.publish().is_err());
        assert_eq!(fs::read_dir(outside.path()).unwrap().count(), 0);
        assert_eq!(
            fs::read_to_string(root.path().join("saved-cursor/mcp.json")).unwrap(),
            "{}"
        );
        assert!(!root.path().join(".deepkit").exists());
        assert!(!root.path().join(".git").exists());
        assert!(!root.path().join(".mcp.json").exists());
    }

    #[test]
    fn mcp_second_target_failure_preserves_first_and_other_destinations() {
        for invalid in [
            "not JSON",
            r#"{"mcpServers":{"deepkit-code":{"command":"other"}}}"#,
        ] {
            let root = tempfile::tempdir().unwrap();
            fs::create_dir_all(root.path().join(".git/info")).unwrap();
            fs::create_dir(root.path().join(".deepkit")).unwrap();
            fs::create_dir(root.path().join(".cursor")).unwrap();
            let config = "original configuration";
            let exclude = "existing excludes\n";
            let cursor = " { \"mcpServers\": {\"other\": {\"command\":\"other\"}} } \n";
            fs::write(root.path().join(".deepkit/config.toml"), config).unwrap();
            fs::write(root.path().join(".git/info/exclude"), exclude).unwrap();
            fs::write(root.path().join(".cursor/mcp.json"), cursor).unwrap();
            fs::write(root.path().join(".mcp.json"), invalid).unwrap();
            assert!(
                prepare(root.path(), false, Some(config), Some("proposed"))
                    .unwrap()
                    .with_mcp(true, true)
                    .is_err()
            );
            for (relative, original) in [
                (".deepkit/config.toml", config),
                (".git/info/exclude", exclude),
                (".cursor/mcp.json", cursor),
                (".mcp.json", invalid),
            ] {
                assert_eq!(
                    fs::read_to_string(root.path().join(relative)).unwrap(),
                    original
                );
            }
            assert_eq!(
                fs::read_dir(root.path().join(".cursor")).unwrap().count(),
                1
            );
            assert_eq!(
                fs::read_dir(root.path().join(".deepkit")).unwrap().count(),
                1
            );
        }
    }
}
