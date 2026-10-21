//! Read-only project MCP entry diagnosis; v1 §§10.2–10.3.
use crate::{
    cli::init::{codex, ide::parse_mcp},
    sync::setup,
};
use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
};

/// One fixed project configuration location, without its potentially secret values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Diagnosis {
    pub(super) id: &'static str,
    pub(super) state: State,
}

/// Diagnoses are intentionally static: configured commands and parser text stay private.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum State {
    Absent,
    Unavailable,
    Malformed,
    MissingEntry,
    Conflicting,
    DifferentWorkingDirectory,
    Unresolved,
    DifferentExecutable,
    Ready,
}
impl State {
    /// Safe finding text describes the check without copying input or credentials.
    pub(super) fn message(self) -> &'static str {
        match self {
            Self::Absent => "Project MCP configuration is absent",
            Self::Unavailable => "Project MCP configuration could not be read safely",
            Self::Malformed => "Project MCP configuration is malformed or has duplicate keys",
            Self::MissingEntry => "Project MCP configuration has no deepkit-code entry",
            Self::Conflicting => "The deepkit-code entry does not describe dkc serve --stdio",
            Self::DifferentWorkingDirectory => {
                "The Codex deepkit-code entry needs an absolute cwd for this repository; review .codex/config.toml"
            }
            Self::Unresolved => {
                "The deepkit-code executable could not be resolved in this process environment"
            }
            Self::DifferentExecutable => {
                "The deepkit-code entry resolves to a different executable"
            }
            Self::Ready => {
                "The deepkit-code entry resolves to this executable with serve --stdio arguments"
            }
        }
    }
}

/// Checks project files using bounded no-follow reads; never launches commands.
/// PATH and current executable are injected so tests need no process environment edits.
/// Resolution describes this process with repository cwd, not the editor's environment.
pub(super) fn diagnose(
    root: &Path,
    current_exe: &Path,
    search_path: Option<&OsStr>,
) -> [Diagnosis; 3] {
    [
        ("mcp.cursor", setup::ExtraFile::Cursor),
        ("mcp.claude", setup::ExtraFile::Claude),
        ("mcp.codex", setup::ExtraFile::Codex),
    ]
    .map(|(id, target)| {
        let state = match setup::read_extra(root, target) {
            Ok(None) => State::Absent,
            Err(_) => State::Unavailable,
            Ok(Some(bytes)) => match target {
                setup::ExtraFile::Codex => inspect_codex(&bytes, root, current_exe, search_path),
                _ => inspect(&bytes, root, current_exe, search_path),
            },
        };
        Diagnosis { id, state }
    })
}

fn inspect(bytes: &[u8], root: &Path, current_exe: &Path, search_path: Option<&OsStr>) -> State {
    let Ok(document) = parse_mcp(bytes) else {
        return State::Malformed;
    };
    let Some(document) = document.as_object() else {
        return State::Malformed;
    };
    let Some(servers) = document.get("mcpServers") else {
        return State::MissingEntry;
    };
    let Some(servers) = servers.as_object() else {
        return State::Malformed;
    };
    let Some(entry) = servers.get("deepkit-code") else {
        return State::MissingEntry;
    };
    let Some(entry) = entry.as_object() else {
        return State::Conflicting;
    };
    inspect_entry(entry, root, current_exe, search_path)
}

fn inspect_codex(
    bytes: &[u8],
    root: &Path,
    current_exe: &Path,
    search_path: Option<&OsStr>,
) -> State {
    let Ok(document) = codex::parse(bytes) else {
        return State::Malformed;
    };
    let Some(servers) = document.get("mcp_servers") else {
        return State::MissingEntry;
    };
    let Some(servers) = servers.as_table() else {
        return State::Malformed;
    };
    let Some(entry) = servers.get("deepkit-code") else {
        return State::MissingEntry;
    };
    let Some(entry) = entry.as_table() else {
        return State::Conflicting;
    };
    let Some(cwd) = entry.get("cwd").and_then(toml::Value::as_str) else {
        return State::DifferentWorkingDirectory;
    };
    if !Path::new(cwd).is_absolute() {
        return State::DifferentWorkingDirectory;
    }
    let Ok(repository) = std::fs::canonicalize(root) else {
        return State::Unresolved;
    };
    if std::fs::canonicalize(cwd).ok().as_ref() != Some(&repository) {
        return State::DifferentWorkingDirectory;
    }
    if entry
        .get("enabled")
        .is_some_and(|value| value.as_bool() != Some(true))
    {
        return State::Conflicting;
    }
    let Ok(serde_json::Value::Object(entry)) = serde_json::to_value(entry) else {
        return State::Malformed;
    };
    inspect_entry(&entry, &repository, current_exe, search_path)
}

fn inspect_entry(
    entry: &serde_json::Map<String, serde_json::Value>,
    root: &Path,
    current_exe: &Path,
    search_path: Option<&OsStr>,
) -> State {
    let Some(command) = entry
        .get("command")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
    else {
        return State::Conflicting;
    };
    if entry.get("args") != Some(&serde_json::json!(["serve", "--stdio"]))
        || entry.get("type").is_some_and(|value| value != "stdio")
        || entry.contains_key("url")
    {
        return State::Conflicting;
    }
    // Client interpolation and environment files cannot be reproduced honestly by
    // a local static check. Do not guess a different executable or read env files.
    if command.contains('$') || entry.contains_key("envFile") {
        return State::Unresolved;
    }
    let mut effective_path = search_path;
    if let Some(env) = entry.get("env") {
        let Some(env) = env.as_object() else {
            return State::Conflicting;
        };
        if env.values().any(|value| !value.is_string()) {
            return State::Conflicting;
        }
        if let Some(path) = env.get("PATH") {
            let Some(path) = path.as_str() else {
                return State::Conflicting;
            };
            if path.contains('$') {
                return State::Unresolved;
            }
            effective_path = Some(OsStr::new(path));
        }
    }
    let Some(resolved) = resolve(command, root, effective_path) else {
        return State::Unresolved;
    };
    let Ok(current) = std::fs::canonicalize(current_exe) else {
        return State::Unresolved;
    };
    if resolved == current || same_file(&resolved, &current) {
        State::Ready
    } else {
        State::DifferentExecutable
    }
}

fn resolve(command: &str, root: &Path, search_path: Option<&OsStr>) -> Option<PathBuf> {
    let command = Path::new(command);
    let candidates = if command.is_absolute() || command.components().count() > 1 {
        vec![root.join(command)]
    } else {
        std::env::split_paths(search_path?)
            .map(|directory| root.join(directory).join(command))
            .collect()
    };
    candidates.into_iter().find_map(|candidate| {
        if !std::fs::metadata(&candidate).ok()?.is_file() {
            return None;
        }
        #[cfg(unix)]
        if rustix::fs::access(&candidate, rustix::fs::Access::EXEC_OK).is_err() {
            return None;
        }
        #[cfg(not(unix))]
        {
            return None;
        }
        #[cfg(unix)]
        std::fs::canonicalize(candidate).ok()
    })
}

fn same_file(first: &Path, second: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        match (std::fs::metadata(first), std::fs::metadata(second)) {
            (Ok(first), Ok(second)) => first.dev() == second.dev() && first.ino() == second.ino(),
            _ => false,
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (first, second);
        false
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::{
        fs,
        os::unix::fs::{PermissionsExt, symlink},
    };

    #[cfg(unix)]
    use crate::sync::source::mkfifo;

    fn entry(command: &Path) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({"mcpServers":{"deepkit-code":{
            "command":command.to_str().unwrap(),"args":["serve","--stdio"]
        }}}))
        .unwrap()
    }

    #[test]
    fn absent_missing_malformed_and_conflicting_are_distinct_and_redacted() {
        let root = tempfile::tempdir().unwrap();
        let executable = std::env::current_exe().unwrap();
        assert_eq!(
            diagnose(root.path(), &executable, None).map(|value| value.state),
            [State::Absent; 3]
        );
        for (bytes, expected) in [
            (b"{}".as_slice(), State::MissingEntry),
            (br#"{"mcpServers":[]}"#.as_slice(), State::Malformed),
            (
                br#"{"private-placeholder":,"mcpServers":{}}"#.as_slice(),
                State::Malformed,
            ),
            (
                br#"{"mcpServers":{},"mcpServers":{}}"#.as_slice(),
                State::Malformed,
            ),
            (
                br#"{"mcpServers":{"deepkit-code":{"command":"private-placeholder","args":[]}}}"#
                    .as_slice(),
                State::Conflicting,
            ),
        ] {
            fs::write(root.path().join(".mcp.json"), bytes).unwrap();
            let diagnosis = diagnose(root.path(), &executable, None)[1];
            assert_eq!(diagnosis.state, expected);
            assert!(
                !format!("{diagnosis:?} {}", diagnosis.state.message())
                    .contains("private-placeholder")
            );
            assert_eq!(fs::read(root.path().join(".mcp.json")).unwrap(), bytes);
        }
    }

    #[test]
    fn resolves_absolute_path_and_path_alias_without_executing_anything() {
        let root = tempfile::tempdir().unwrap();
        let executable = std::env::current_exe().unwrap();
        fs::create_dir(root.path().join("bin")).unwrap();
        symlink(&executable, root.path().join("bin/dkc")).unwrap();
        let path = std::env::join_paths([root.path().join("bin")]).unwrap();
        for command in [&executable, Path::new("dkc")] {
            let ready = inspect(&entry(command), root.path(), &executable, Some(&path));
            assert_eq!(ready, State::Ready);
            assert_eq!(
                ready.message(),
                "The deepkit-code entry resolves to this executable with serve --stdio arguments"
            );
        }
        assert_eq!(
            inspect(
                &entry(Path::new("./bin/dkc")),
                root.path(),
                &executable,
                None
            ),
            State::Ready
        );
        let other = root.path().join("bin/other");
        fs::write(&other, "#!/bin/sh\ntouch must-never-exist\n").unwrap();
        fs::set_permissions(&other, fs::Permissions::from_mode(0o700)).unwrap();
        let stale = inspect(&entry(&other), root.path(), &executable, None);
        assert_eq!(stale, State::DifferentExecutable);
        assert_eq!(
            stale.message(),
            "The deepkit-code entry resolves to a different executable"
        );
        fs::create_dir(root.path().join("shadow")).unwrap();
        symlink(&other, root.path().join("shadow/dkc")).unwrap();
        let shadowed =
            std::env::join_paths([root.path().join("shadow"), root.path().join("bin")]).unwrap();
        assert_eq!(
            inspect(
                &entry(Path::new("dkc")),
                root.path(),
                &executable,
                Some(&shadowed)
            ),
            State::DifferentExecutable
        );
        fs::set_permissions(&other, fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(
            inspect(&entry(&other), root.path(), &executable, None),
            State::Unresolved
        );
        assert_eq!(
            inspect(
                &entry(Path::new("${workspaceFolder}/dkc")),
                root.path(),
                &executable,
                None
            ),
            State::Unresolved
        );
        assert!(!root.path().join("must-never-exist").exists());
        assert_eq!(
            inspect(
                &entry(Path::new("missing")),
                root.path(),
                &executable,
                Some(&path)
            ),
            State::Unresolved
        );
        let configured = serde_json::json!({"mcpServers":{"deepkit-code":{"command":"dkc","args":["serve","--stdio"],"env":{"PATH":root.path().join("bin")}}}});
        assert_eq!(
            inspect(
                &serde_json::to_vec(&configured).unwrap(),
                root.path(),
                &executable,
                None
            ),
            State::Ready
        );
    }

    #[test]
    fn codex_checks_repository_binding_before_resolving_the_command() {
        let root = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let executable = std::env::current_exe().unwrap();
        fs::create_dir(root.path().join("bin")).unwrap();
        symlink(&executable, root.path().join("bin/dkc")).unwrap();
        let path = std::env::join_paths([root.path().join("bin")]).unwrap();
        let generated = codex::merge(None, root.path()).unwrap();
        fs::create_dir(root.path().join(".codex")).unwrap();
        fs::write(root.path().join(".codex/config.toml"), &generated).unwrap();
        let diagnosis = diagnose(root.path(), &executable, Some(&path));
        assert_eq!(diagnosis[2].id, "mcp.codex");
        assert_eq!(diagnosis[2].state, State::Ready);
        assert_eq!(
            fs::read(root.path().join(".codex/config.toml")).unwrap(),
            generated
        );
        for cwd in [
            None,
            Some(toml::Value::String(".".into())),
            Some(toml::Value::String(other.path().to_str().unwrap().into())),
            Some(toml::Value::Integer(3)),
        ] {
            let mut document = codex::parse(&generated).unwrap();
            let entry = document
                .get_mut("mcp_servers")
                .unwrap()
                .get_mut("deepkit-code")
                .unwrap()
                .as_table_mut()
                .unwrap();
            match cwd {
                Some(cwd) => {
                    entry.insert("cwd".into(), cwd);
                }
                None => {
                    entry.remove("cwd");
                }
            }
            let bytes = toml::to_string(&document).unwrap();
            assert_eq!(
                inspect_codex(bytes.as_bytes(), root.path(), &executable, Some(&path)),
                State::DifferentWorkingDirectory
            );
        }
        for (source, state) in [
            ("# preferences only", State::MissingEntry),
            ("mcp_servers = 1", State::Malformed),
            ("private-placeholder = [", State::Malformed),
            (
                "[mcp_servers]\ndeepkit-code = 'private-placeholder'",
                State::Conflicting,
            ),
        ] {
            let actual = inspect_codex(source.as_bytes(), root.path(), &executable, Some(&path));
            assert_eq!(actual, state);
            assert!(!format!("{actual:?} {}", actual.message()).contains("private-placeholder"));
        }
    }

    #[test]
    fn project_readers_refuse_unsafe_files_without_writes_or_fifo_wait() {
        use std::{sync::mpsc, time::Duration};
        for (index, relative) in [".cursor/mcp.json", ".mcp.json", ".codex/config.toml"]
            .into_iter()
            .enumerate()
        {
            for kind in ["symlink", "hardlink", "fifo", "oversize"] {
                let root = tempfile::tempdir().unwrap();
                let outside = tempfile::tempdir().unwrap();
                let file = root.path().join(relative);
                fs::create_dir_all(file.parent().unwrap()).unwrap();
                let external = outside.path().join("kept");
                fs::write(&external, "unchanged").unwrap();
                match kind {
                    "symlink" => symlink(&external, &file).unwrap(),
                    "hardlink" => fs::hard_link(&external, &file).unwrap(),
                    "fifo" => {
                        mkfifo(&file, rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR).unwrap()
                    }
                    _ => fs::File::create(&file)
                        .unwrap()
                        .set_len(1024 * 1024 + 1)
                        .unwrap(),
                }
                let path = root.path().to_path_buf();
                let (sender, receiver) = mpsc::channel();
                let worker = std::thread::spawn(move || {
                    sender
                        .send(diagnose(&path, &std::env::current_exe().unwrap(), None))
                        .unwrap()
                });
                let results = receiver.recv_timeout(Duration::from_secs(2)).unwrap();
                worker.join().unwrap();
                assert_eq!(results[index].state, State::Unavailable);
                assert_eq!(fs::read_to_string(&external).unwrap(), "unchanged");
                assert!(!root.path().join(".deepkit").exists());
                assert!(!root.path().join(".git").exists());
            }
        }
    }
}
