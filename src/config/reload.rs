//! Coherent configuration inputs for repeated L1 publication checks; v1 §4.6/§11.
use super::{Config, Error, NamedSerialized, default_user_config_path, local_defaults};
use figment::{
    Figment, Metadata, Profile, Provider,
    providers::Env,
    value::{Dict, Map},
};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// Private, transient inputs: never debug-formatted, serialized or logged. Exact
/// file contents and effective environment values detect changes between checks.
#[derive(PartialEq)]
pub(crate) struct Snapshot {
    layers: Vec<(PathBuf, Option<String>)>,
    environment: Map<Profile, Dict>,
}

impl Snapshot {
    /// Reads all normal configuration layers, including their explicit absence.
    pub(crate) fn capture(root: &Path) -> Result<Self, Error> {
        Self::capture_with_user(root, default_user_config_path())
    }

    fn capture_with_user(root: &Path, user: Option<PathBuf>) -> Result<Self, Error> {
        let mut layers = Vec::new();
        for path in user.into_iter().chain([
            root.join(".deepkit/config.toml"),
            root.join(".deepkit/config.local.toml"),
        ]) {
            let contents = if path.exists() {
                Some(fs::read_to_string(&path).map_err(|source| Error::Io {
                    path: path.clone(),
                    source,
                })?)
            } else {
                None
            };
            layers.push((path, contents));
        }
        let environment = Env::prefixed("DKC_")
            .split("__")
            .data()
            .map_err(|error| Error::Figment(Box::new(error)))?;
        Ok(Self {
            layers,
            environment,
        })
    }

    /// Parses the captured inputs themselves, so concurrent edits cannot attach
    /// a parsed value to a different file-content identity. No source is reopened.
    pub(crate) fn load(&self) -> Result<Config, Error> {
        let mut figment =
            Figment::from(NamedSerialized::new(local_defaults(), "built-in defaults"));
        for (path, contents) in &self.layers {
            if let Some(contents) = contents {
                let value: toml::Value =
                    toml::from_str(contents).map_err(|source| Error::ParseToml {
                        path: path.clone(),
                        source,
                    })?;
                figment = figment.merge(NamedSerialized {
                    value,
                    name: "captured configuration",
                    source: Some(path.clone()),
                });
            }
        }
        let config: Config = figment
            .merge(CapturedEnvironment(&self.environment))
            .extract()
            .map_err(|error| Error::Figment(Box::new(error)))?;
        config.validate()?;
        Ok(config)
    }
}

/// Reloads current bytes and environment every time, reusing only their parsed
/// value. The single entry per thread retains no source, HEAD or policy authority;
/// callers still compare settings and perform their own source/final fences.
pub(crate) fn current(root: &Path) -> Result<Config, Error> {
    thread_local! {
        static PARSED: std::cell::RefCell<Parsed> = const {
            std::cell::RefCell::new(Parsed(None))
        };
    }
    let inputs = Snapshot::capture(root)?;
    PARSED.with(|parsed| match parsed.try_borrow_mut() {
        Ok(mut parsed) => parsed.load(inputs),
        Err(_) => inputs.load(),
    })
}

struct Parsed(Option<(Snapshot, Config)>);
impl Parsed {
    fn load(&mut self, inputs: Snapshot) -> Result<Config, Error> {
        if let Some((previous, config)) = &self.0
            && previous == &inputs
        {
            return Ok(config.clone());
        }
        let config = inputs.load()?;
        self.0 = Some((inputs, config.clone()));
        Ok(config)
    }
}

struct CapturedEnvironment<'a>(&'a Map<Profile, Dict>);
impl Provider for CapturedEnvironment<'_> {
    fn metadata(&self) -> Metadata {
        Metadata::named("captured environment variables")
    }
    fn data(&self) -> Result<Map<Profile, Dict>, figment::Error> {
        Ok(self.0.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_environment_matches_normal_loading() {
        const CHILD: &str = "DKC_RELOAD_TEST_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let root = tempfile::tempdir().unwrap();
            for value in ["17", "19"] {
                let result = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "config::reload::tests::captured_environment_matches_normal_loading",
                        "--nocapture",
                    ])
                    .env(CHILD, "1")
                    .env("DKC_LIMITS__MAX_FILES", value)
                    .env("XDG_CONFIG_HOME", root.path())
                    .output()
                    .unwrap();
                assert!(
                    result.status.success(),
                    "{} {}",
                    String::from_utf8_lossy(&result.stdout),
                    String::from_utf8_lossy(&result.stderr)
                );
            }
            return;
        }
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join(".deepkit")).unwrap();
        fs::write(
            root.path().join(".deepkit/config.toml"),
            "[limits]\nmax_files=3\n",
        )
        .unwrap();
        let captured = Snapshot::capture(root.path()).unwrap();
        let expected = super::super::load(root.path(), None).unwrap().config;
        assert_eq!(captured.load().unwrap(), expected);
        assert_eq!(
            current(root.path()).unwrap(),
            super::super::load(root.path(), None).unwrap().config
        );
        assert_eq!(
            expected.limits.max_files.to_string(),
            std::env::var("DKC_LIMITS__MAX_FILES").unwrap()
        );
    }

    #[test]
    fn captured_layers_match_normal_precedence_and_never_reopen_changed_files() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join(".deepkit")).unwrap();
        let user = root.path().join("user.toml");
        let repository = root.path().join(".deepkit/config.toml");
        let local = root.path().join(".deepkit/config.local.toml");
        fs::write(&user, "[index]\nrepo_name='user'\n[limits]\nmax_files=12\n").unwrap();
        fs::write(
            &repository,
            "[index]\nrepo_name='repo'\n[limits]\nmax_files=13\n",
        )
        .unwrap();
        fs::write(&local, "[index]\nrepo_name='local'\n").unwrap();
        let captured = Snapshot::capture_with_user(root.path(), Some(user.clone())).unwrap();
        let expected = super::super::load(root.path(), Some(&user)).unwrap().config;
        assert_eq!(captured.load().unwrap(), expected);
        let metadata = fs::metadata(&local).unwrap();
        fs::write(&local, "[index]\nrepo_name='other'\n").unwrap();
        fs::File::options()
            .write(true)
            .open(&local)
            .unwrap()
            .set_times(fs::FileTimes::new().set_modified(metadata.modified().unwrap()))
            .unwrap();
        let changed = Snapshot::capture_with_user(root.path(), Some(user.clone())).unwrap();
        assert!(
            captured != changed,
            "equal-length restored-mtime edits must invalidate reuse"
        );
        assert_eq!(captured.load().unwrap(), expected);
        assert_eq!(
            changed.load().unwrap(),
            super::super::load(root.path(), Some(&user)).unwrap().config
        );
        fs::write(&local, "invalid = [").unwrap();
        assert!(
            Snapshot::capture_with_user(root.path(), Some(user.clone()))
                .unwrap()
                .load()
                .is_err()
        );
        fs::remove_file(&local).unwrap();
        let absent = Snapshot::capture_with_user(root.path(), Some(user.clone())).unwrap();
        assert!(absent != changed);
        assert_eq!(
            absent.load().unwrap(),
            super::super::load(root.path(), Some(&user)).unwrap().config
        );
    }
    #[test]
    fn parsed_reuse_reads_equal_mtime_edits_absence_errors_and_distinct_roots() {
        let root = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join(".deepkit")).unwrap();
        let path = root.path().join(".deepkit/config.local.toml");
        let check = || {
            let actual = current(root.path()).unwrap();
            assert_eq!(
                actual,
                super::super::load(root.path(), None).unwrap().config
            );
            actual
        };
        let baseline = check();
        fs::write(&path, "[limits]\nmax_files=31\n").unwrap();
        assert_eq!(check().limits.max_files, 31);
        assert_eq!(check().limits.max_files, 31);
        let modified = fs::metadata(&path).unwrap().modified().unwrap();
        fs::write(&path, "[limits]\nmax_files=32\n").unwrap();
        fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(fs::FileTimes::new().set_modified(modified))
            .unwrap();
        assert_eq!(check().limits.max_files, 32);
        assert_eq!(
            current(other.path()).unwrap(),
            super::super::load(other.path(), None).unwrap().config
        );
        assert_eq!(check().limits.max_files, 32);
        fs::write(&path, "invalid = [").unwrap();
        assert!(current(root.path()).is_err());
        fs::write(&path, "[limits]\nmax_files=33\n").unwrap();
        assert_eq!(check().limits.max_files, 33);
        fs::remove_file(&path).unwrap();
        assert_eq!(check(), baseline);
    }
}
