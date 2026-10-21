//! Guarded MCP configuration reads and atomic layer writes; v1 §§10.1 and 11.
use super::{Config, DocsEmbedder, LoadedConfig, load, set_value, value_at};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Component, Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

static WRITES: Mutex<()> = Mutex::new(());
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

/// The only writable layers exposed to agents; omission selects the private local layer.
#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum WriteLayer {
    Repo,
    #[default]
    Local,
}

/// Effective value and winning provenance after applying layer precedence.
#[derive(Debug, Serialize)]
pub(crate) struct Response {
    pub(crate) key: String,
    pub(crate) value: serde_json::Value,
    pub(crate) source: &'static str,
}

/// Safe configuration errors never include a submitted value or raw file contents.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("agent configuration writes are disabled")]
    ReadOnly,
    #[error("unknown configuration key")]
    UnknownKey,
    #[error("configuration value has the wrong type or invalid settings")]
    InvalidValue,
    #[error(
        "Secrets must use environment variables (DKC_SERVE__HTTP_TOKEN for HTTP) or the user credentials file"
    )]
    Secret,
    #[error("configuration destination must remain inside the repository")]
    UnsafePath,
    #[error("configuration changed concurrently; read current values before retrying")]
    Changed,
    #[error("could not load configuration")]
    Load,
    #[error("could not access configuration files")]
    Io(#[source] std::io::Error),
    #[error("configuration writer lock is unavailable")]
    Lock,
}

/// Reads a known effective setting and its source, redacting HTTP bearer credentials.
pub(crate) fn get(root: &Path, key: &str) -> Result<Response, Error> {
    schema_value(key)?;
    response(&load(root, None).map_err(|_| Error::Load)?, key)
}

/// Explains the winning layer using the same value and redaction contract as get.
pub(crate) fn explain(root: &Path, key: &str) -> Result<Response, Error> {
    get(root, key)
}

fn response(loaded: &LoadedConfig, key: &str) -> Result<Response, Error> {
    let value = loaded.config.get(key).map_err(|_| Error::Load)?;
    if value.is_none() && key.starts_with("llm.extra_body.") {
        return Err(Error::UnknownKey);
    }
    let mut value = serde_json::to_value(value).map_err(|_| Error::Load)?;
    redact(key, &mut value);
    Ok(Response {
        key: key.into(),
        value,
        source: loaded
            .explain(key)
            .map_or("built-in defaults", super::Layer::name),
    })
}
fn redact(key: &str, value: &mut serde_json::Value) {
    if key.rsplit('.').next().is_some_and(secret_key) {
        *value = serde_json::Value::String("[redacted]".into());
    } else if key == "serve.http_token" {
        if value.as_str().is_some_and(|value| !value.is_empty()) {
            *value = serde_json::Value::String("[redacted]".into());
        }
    } else if let Some(table) = value.as_object_mut() {
        for (name, value) in table {
            redact(&format!("{key}.{name}"), value);
        }
    } else if let Some(values) = value.as_array_mut() {
        for value in values {
            redact(key, value);
        }
    }
}

fn template() -> Result<toml::Value, Error> {
    let mut config = Config::default();
    config.embedder.docs = Some(DocsEmbedder::default());
    config.queue.budgets.push(super::queue::EndpointBudget {
        profile: "remote".into(),
        url: String::new(),
        daily_calls: Some(0),
        daily_tokens: Some(0),
    });
    for treatment in &mut config.treatments {
        treatment.chunker = Some(String::new());
        treatment.enrich = vec![String::new()];
    }
    toml::Value::try_from(config).map_err(|_| Error::Load)
}
fn schema_value(key: &str) -> Result<Option<toml::Value>, Error> {
    if key.is_empty() || key.split('.').any(str::is_empty) {
        return Err(Error::UnknownKey);
    }
    if key.split('.').any(secret_key) {
        return Err(Error::Secret);
    }
    match value_at(&template()?, key).cloned() {
        Some(value) => Ok(Some(value)),
        None if key.starts_with("llm.extra_body.") => Ok(None),
        None => Err(Error::UnknownKey),
    }
}
/// Identifies fields that configuration publishers must never persist, regardless of value type.
pub(crate) fn secret_key(key: &str) -> bool {
    matches!(
        key.to_ascii_lowercase().as_str(),
        "api_key"
            | "apikey"
            | "authorization"
            | "access_token"
            | "refresh_token"
            | "password"
            | "secret"
            | "credentials"
            | "private_key"
    )
}
fn reject_secrets(path: &str, value: &serde_json::Value) -> Result<(), Error> {
    if path == "serve.http_token" && value.as_str().is_some_and(|value| !value.is_empty()) {
        return Err(Error::Secret);
    }
    match value {
        serde_json::Value::Object(values) => {
            for (key, value) in values {
                if secret_key(key) {
                    return Err(Error::Secret);
                }
                reject_secrets(&format!("{path}.{key}"), value)?;
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                reject_secrets(path, value)?;
            }
        }
        _ => {}
    }
    Ok(())
}
fn compatible(value: &toml::Value, template: &toml::Value, key: &str) -> bool {
    if key == "commits.pr_patterns" {
        return value
            .clone()
            .try_into::<Vec<super::pr_patterns::Pattern>>()
            .is_ok();
    }
    match (value, template) {
        (toml::Value::String(_), toml::Value::String(_))
        | (toml::Value::Integer(_), toml::Value::Integer(_))
        | (toml::Value::Float(_), toml::Value::Float(_))
        | (toml::Value::Boolean(_), toml::Value::Boolean(_)) => true,
        (toml::Value::Array(values), toml::Value::Array(known)) => values.iter().all(|value| {
            known
                .first()
                .is_some_and(|template| compatible(value, template, key))
        }),
        (toml::Value::Table(values), toml::Value::Table(known)) => {
            key == "llm.extra_body"
                || values.iter().all(|(name, value)| {
                    known.get(name).is_some_and(|template| {
                        compatible(value, template, &format!("{key}.{name}"))
                    })
                })
        }
        _ => false,
    }
}

fn read_optional(path: &Path) -> Result<Option<Vec<u8>>, Error> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(Error::Io(error)),
    }
}
fn refuse_symlink(path: &Path) -> Result<(), Error> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(Error::UnsafePath),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(Error::Io(error)),
    }
}
fn resolve_beneath(base: &Path, relative: &Path) -> Result<PathBuf, Error> {
    let mut resolved = base.to_path_buf();
    for component in relative.components() {
        match component {
            Component::CurDir => continue,
            Component::Normal(name) => resolved.push(name),
            _ => return Err(Error::UnsafePath),
        }
        match fs::symlink_metadata(&resolved) {
            Ok(_) => {
                resolved = fs::canonicalize(&resolved).map_err(Error::Io)?;
                if !resolved.starts_with(base) {
                    return Err(Error::UnsafePath);
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(Error::Io(error)),
        }
    }
    Ok(resolved)
}

fn validate_paths(root: &Path, proposed: &Config) -> Result<(), Error> {
    let root = fs::canonicalize(root).map_err(Error::Io)?;
    let indexed_root = resolve_beneath(&root, Path::new(&proposed.index.root))?;
    let components = Path::new(&proposed.store.path)
        .components()
        .filter(|component| *component != Component::CurDir)
        .collect::<Vec<_>>();
    if components.len() < 2
        || components.first() != Some(&Component::Normal(".deepkit".as_ref()))
        || components
            .iter()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(Error::UnsafePath);
    }
    let directory = indexed_root.join(".deepkit");
    refuse_symlink(&directory)?;
    let tail = components[1..]
        .iter()
        .map(|component| component.as_os_str())
        .collect::<PathBuf>();
    resolve_beneath(&directory, &tail)?;
    Ok(())
}

struct Temporary {
    path: PathBuf,
    remove: bool,
}
impl Drop for Temporary {
    fn drop(&mut self) {
        if self.remove {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Validates and writes one layer atomically, then returns the effective winning value.
/// Agent API writes serialize; byte comparisons detect observed external edits,
/// but are not a cross-process compare-and-swap transaction.
pub(crate) fn set(
    root: &Path,
    key: &str,
    value: serde_json::Value,
    layer: WriteLayer,
) -> Result<Response, Error> {
    let _lock = WRITES.lock().map_err(|_| Error::Lock)?;
    let schema = schema_value(key)?;
    reject_secrets(key, &value)?;
    let value = toml::Value::try_from(value).map_err(|_| Error::InvalidValue)?;
    if schema
        .as_ref()
        .is_some_and(|schema| !compatible(&value, schema, key))
    {
        return Err(Error::InvalidValue);
    }
    let directory = root.join(".deepkit");
    refuse_symlink(&directory)?;
    let repo = directory.join("config.toml");
    let local = directory.join("config.local.toml");
    refuse_symlink(&repo)?;
    refuse_symlink(&local)?;
    let before = [read_optional(&repo)?, read_optional(&local)?];
    let loaded = load(root, None).map_err(|_| Error::Load)?;
    if !loaded.config.serve.agent_writable_config {
        return Err(Error::ReadOnly);
    }
    let mut effective = toml::Value::try_from(&loaded.config).map_err(|_| Error::Load)?;
    set_value(&mut effective, key, value.clone()).map_err(|_| Error::InvalidValue)?;
    let proposed: Config = effective.try_into().map_err(|_| Error::InvalidValue)?;
    proposed.validate().map_err(|_| Error::InvalidValue)?;
    validate_paths(root, &proposed)?;
    let (path, bytes) = match layer {
        WriteLayer::Repo => (&repo, &before[0]),
        WriteLayer::Local => (&local, &before[1]),
    };
    let mut document = match bytes {
        Some(bytes) => toml::from_str(std::str::from_utf8(bytes).map_err(|_| Error::Load)?)
            .map_err(|_| Error::Load)?,
        None => toml::Value::Table(Default::default()),
    };
    set_value(&mut document, key, value).map_err(|_| Error::InvalidValue)?;
    let rendered = toml::to_string_pretty(&document).map_err(|_| Error::InvalidValue)?;
    fs::create_dir_all(&directory).map_err(Error::Io)?;
    let temporary_path = directory.join(format!(
        ".config-{}-{}.tmp",
        std::process::id(),
        TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temporary_path).map_err(Error::Io)?;
    let mut temporary = Temporary {
        path: temporary_path,
        remove: true,
    };
    file.write_all(rendered.as_bytes()).map_err(Error::Io)?;
    file.sync_all().map_err(Error::Io)?;
    drop(file);
    refuse_symlink(&directory)?;
    refuse_symlink(&repo)?;
    refuse_symlink(&local)?;
    let current = load(root, None).map_err(|_| Error::Load)?;
    if !current.config.serve.agent_writable_config {
        return Err(Error::ReadOnly);
    }
    if before != [read_optional(&repo)?, read_optional(&local)?] || current.config != loaded.config
    {
        return Err(Error::Changed);
    }
    validate_paths(root, &proposed)?;
    fs::rename(&temporary.path, path).map_err(Error::Io)?;
    temporary.remove = false;
    response(&load(root, None).map_err(|_| Error::Load)?, key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture(text: &str) -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join(".deepkit")).unwrap();
        fs::write(root.path().join(".deepkit/config.toml"), text).unwrap();
        root
    }

    #[test]
    fn writes_named_layer_and_reports_effective_winner_without_touching_other_bytes() {
        let root = fixture("# keep this comment\n[search]\nmax_k=40\n");
        let path = root.path().join(".deepkit/config.toml");
        let before = fs::read(&path).unwrap();
        let initial = get(root.path(), "search.max_k").unwrap();
        assert_eq!(initial.value, 40);
        assert_eq!(initial.source, "repository config");
        let changed = set(root.path(), "search.max_k", json!(30), WriteLayer::Local).unwrap();
        assert_eq!(changed.value, 30);
        assert_eq!(changed.source, "local repository config");
        assert_eq!(fs::read(&path).unwrap(), before);
        let masked = set(root.path(), "search.max_k", json!(20), WriteLayer::Repo).unwrap();
        assert_eq!(masked.value, 30);
        assert_eq!(
            explain(root.path(), "search.max_k").unwrap().source,
            "local repository config"
        );
        assert_eq!(
            fs::read_dir(root.path().join(".deepkit")).unwrap().count(),
            2
        );
    }

    #[test]
    fn readonly_is_reloaded_and_cannot_be_overridden_by_the_agent() {
        let root = fixture("[serve]\nagent_writable_config=false\n");
        let original = fs::read(root.path().join(".deepkit/config.toml")).unwrap();
        for (key, value) in [
            ("search.max_k", json!(20)),
            ("serve.agent_writable_config", json!(true)),
        ] {
            assert!(matches!(
                set(root.path(), key, value, WriteLayer::Repo),
                Err(Error::ReadOnly)
            ));
        }
        assert_eq!(
            fs::read(root.path().join(".deepkit/config.toml")).unwrap(),
            original
        );
        assert!(!root.path().join(".deepkit/config.local.toml").exists());
    }

    #[test]
    fn invalid_keys_types_and_semantics_leave_original_bytes_untouched() {
        let root = fixture("# unchanged\n[search]\nmax_k=40\n");
        let path = root.path().join(".deepkit/config.toml");
        let original = fs::read(&path).unwrap();
        for (key, value) in [
            ("search.typo", json!(20)),
            ("search.max_k", json!("20")),
            ("search.max_k", json!(-1)),
            ("chunk.min_tokens", json!(9000)),
            ("search", json!({"typo": 20})),
            ("tests.globs", json!([true])),
            ("embedder.docs", json!({"profile":"remote", "typo":true})),
            ("store.path", json!("../escaped.db")),
            ("store.path", json!("other.sqlite")),
            ("store.path", json!(".deepkit/../escaped.db")),
            ("store.path", json!("/tmp/escaped.db")),
            ("store.path", json!(".deepkit")),
            ("index.root", json!("../other-repository")),
            ("index.root", json!("/tmp")),
        ] {
            assert!(
                set(root.path(), key, value, WriteLayer::Repo).is_err(),
                "{key}"
            );
            assert_eq!(fs::read(&path).unwrap(), original);
        }
        assert_eq!(
            fs::read_dir(root.path().join(".deepkit")).unwrap().count(),
            1
        );
    }

    #[test]
    fn credentials_are_refused_while_env_names_and_legacy_token_redaction_work() {
        let root = fixture("");
        for (key, value) in [
            ("embedder.remote.api_key", json!("fixture-only")),
            (
                "llm.extra_body",
                json!({"nested":{"Authorization":"fixture-only"}}),
            ),
            ("llm", json!({"extra_body":{"api_key":"fixture-only"}})),
        ] {
            assert!(matches!(
                set(root.path(), key, value, WriteLayer::Local),
                Err(Error::Secret)
            ));
        }
        assert!(!root.path().join(".deepkit/config.local.toml").exists());
        let env = set(
            root.path(),
            "llm.api_key_env",
            json!("FIXTURE_API_KEY"),
            WriteLayer::Local,
        )
        .unwrap();
        assert_eq!(env.value, "FIXTURE_API_KEY");
        let synthetic = format!("fixture-bearer-{}", std::process::id());
        fs::write(
            root.path().join(".deepkit/config.toml"),
            format!("[serve]\nhttp_token={synthetic:?}\n"),
        )
        .unwrap();
        let token = get(root.path(), "serve.http_token").unwrap();
        assert_eq!(token.value, "[redacted]");
        assert_eq!(
            get(root.path(), "serve").unwrap().value["http_token"],
            "[redacted]"
        );
        assert_eq!(
            explain(root.path(), "serve.http_token").unwrap().value,
            "[redacted]"
        );
        assert!(!serde_json::to_string(&token).unwrap().contains(&synthetic));
    }

    #[test]
    fn http_token_writes_are_refused_in_both_layers_but_clearing_is_allowed() {
        let synthetic = format!("fixture-legacy-{}", std::process::id());
        let root = fixture(&format!(
            "# legacy token\n[serve]\nhttp_token={synthetic:?}\n"
        ));
        let repo = root.path().join(".deepkit/config.toml");
        let local = root.path().join(".deepkit/config.local.toml");
        fs::write(&local, "# local settings\n").unwrap();
        let original = [fs::read(&repo).unwrap(), fs::read(&local).unwrap()];
        for layer in [WriteLayer::Repo, WriteLayer::Local] {
            for (key, value) in [
                ("serve.http_token", json!("fixture-new-only")),
                ("serve", json!({"http_token":"fixture-new-only"})),
            ] {
                let error = set(root.path(), key, value, layer).unwrap_err();
                assert!(matches!(error, Error::Secret));
                assert!(error.to_string().contains("DKC_SERVE__HTTP_TOKEN"));
                assert_eq!(
                    [fs::read(&repo).unwrap(), fs::read(&local).unwrap()],
                    original
                );
            }
        }
        for layer in [WriteLayer::Repo, WriteLayer::Local] {
            assert_eq!(
                set(root.path(), "serve.http_token", json!(""), layer)
                    .unwrap()
                    .value,
                ""
            );
        }
        assert_eq!(
            fs::read_dir(root.path().join(".deepkit")).unwrap().count(),
            2
        );
    }

    #[test]
    fn reads_redact_existing_noncompliant_nested_credentials() {
        let root = fixture(
            "[llm.extra_body]
items=[{authorization='fixture-existing-only'}]
",
        );
        let response = get(root.path(), "llm.extra_body").unwrap();
        assert_eq!(response.value["items"][0]["authorization"], "[redacted]");
        assert!(
            !serde_json::to_string(&response)
                .unwrap()
                .contains("fixture-existing-only")
        );
    }

    #[test]
    fn optional_docs_and_arbitrary_extra_body_are_typed_and_persisted() {
        let root = fixture("");
        assert!(get(root.path(), "embedder.docs").unwrap().value.is_null());
        let docs = set(
            root.path(),
            "embedder.docs",
            json!({"profile":"local", "kinds":["doc_section"]}),
            WriteLayer::Local,
        )
        .unwrap();
        assert_eq!(docs.value["profile"], "local");
        let body = set(
            root.path(),
            "llm.extra_body",
            json!({"chat_template_kwargs":{"enable_thinking":false},"new_option":12}),
            WriteLayer::Local,
        )
        .unwrap();
        assert_eq!(body.value["new_option"], 12);
        let nested = set(
            root.path(),
            "llm.extra_body.custom.enabled",
            json!(true),
            WriteLayer::Local,
        )
        .unwrap();
        assert_eq!(nested.value, true);
        assert_eq!(
            get(root.path(), "llm.extra_body.custom.enabled")
                .unwrap()
                .value,
            true
        );
        assert!(matches!(
            get(root.path(), "llm.extra_body.missing"),
            Err(Error::UnknownKey)
        ));
        let treatments = set(
            root.path(),
            "treatments",
            json!([{
                "name":"docs", "match":["docs/**"], "chunker":"markdown",
                "max_level":3, "enrich":["vlm_page"], "never_send":false,
            }]),
            WriteLayer::Local,
        )
        .unwrap();
        assert_eq!(treatments.value[0]["enrich"], json!(["vlm_page"]));
    }

    #[test]
    fn structured_pr_patterns_are_typed_and_invalid_patterns_preserve_configuration() {
        let root = fixture("");
        let patterns = json!(["github-squash", {"regex":"^Merge commit '(?P<title>.+)'$", "strategy":"plain"}]);
        assert_eq!(
            set(
                root.path(),
                "commits.pr_patterns",
                patterns.clone(),
                WriteLayer::Local
            )
            .unwrap()
            .value,
            patterns
        );
        let path = root.path().join(".deepkit/config.local.toml");
        let before = fs::read(&path).unwrap();
        for invalid in [
            json!([{"regex":"[","strategy":"plain"}]),
            json!([{"regex":".*","strategy":"unknown"}]),
            json!([{"regex":".*","strategy":"plain","extra":true}]),
        ] {
            assert!(
                set(
                    root.path(),
                    "commits.pr_patterns",
                    invalid,
                    WriteLayer::Local
                )
                .is_err()
            );
            assert_eq!(fs::read(&path).unwrap(), before);
        }
    }

    #[cfg(unix)]
    #[test]
    fn symlink_destinations_do_not_write_outside_the_repository() {
        use std::os::unix::fs::symlink;
        let outside = tempfile::tempdir().unwrap();
        let external = outside.path().join("config.toml");
        fs::write(&external, "# outside\n").unwrap();
        let root = fixture("");
        symlink(&external, root.path().join(".deepkit/config.local.toml")).unwrap();
        assert!(matches!(
            set(root.path(), "search.max_k", json!(20), WriteLayer::Local),
            Err(Error::UnsafePath)
        ));
        assert_eq!(fs::read_to_string(&external).unwrap(), "# outside\n");
        let linked = tempfile::tempdir().unwrap();
        symlink(outside.path(), linked.path().join(".deepkit")).unwrap();
        assert!(matches!(
            set(linked.path(), "search.max_k", json!(20), WriteLayer::Repo),
            Err(Error::UnsafePath)
        ));
        assert_eq!(fs::read_to_string(&external).unwrap(), "# outside\n");
    }

    #[test]
    fn relative_missing_index_root_is_recoverable_without_creating_a_database() {
        let root = fixture("");
        assert_eq!(
            set(
                root.path(),
                "index.root",
                json!("missing-directory"),
                WriteLayer::Local
            )
            .unwrap()
            .value,
            "missing-directory"
        );
        assert!(!root.path().join("missing-directory").exists());
        assert_eq!(
            get(root.path(), "index.root").unwrap().value,
            "missing-directory"
        );
        set(root.path(), "index.root", json!("."), WriteLayer::Local).unwrap();
        set(
            root.path(),
            "store.path",
            json!("./.deepkit/nested/code.db"),
            WriteLayer::Local,
        )
        .unwrap();
        assert!(!root.path().join(".deepkit/nested").exists());
    }

    #[cfg(unix)]
    #[test]
    fn indexed_root_and_database_symlinks_cannot_redirect_writes() {
        use std::os::unix::fs::symlink;
        let root = fixture("");
        let outside = tempfile::tempdir().unwrap();
        symlink(outside.path(), root.path().join("external")).unwrap();
        assert!(matches!(
            set(
                root.path(),
                "index.root",
                json!("external"),
                WriteLayer::Local
            ),
            Err(Error::UnsafePath)
        ));
        symlink(outside.path(), root.path().join(".deepkit/external")).unwrap();
        assert!(matches!(
            set(
                root.path(),
                "store.path",
                json!(".deepkit/external/code.db"),
                WriteLayer::Local
            ),
            Err(Error::UnsafePath)
        ));
        fs::create_dir(root.path().join("source")).unwrap();
        symlink(
            root.path().join("source"),
            root.path().join(".deepkit/source"),
        )
        .unwrap();
        assert!(matches!(
            set(
                root.path(),
                "store.path",
                json!(".deepkit/source/code.db"),
                WriteLayer::Local
            ),
            Err(Error::UnsafePath)
        ));
        assert!(!root.path().join(".deepkit/config.local.toml").exists());
        assert!(!outside.path().join("code.db").exists());
    }

    #[test]
    fn same_process_writes_preserve_both_updates() {
        let root = fixture("");
        let path = root.path().to_path_buf();
        std::thread::scope(|scope| {
            for (key, value) in [("search.max_k", json!(20)), ("search.default_k", json!(5))] {
                let root = &path;
                scope.spawn(move || set(root, key, value, WriteLayer::Local).unwrap());
            }
        });
        assert_eq!(get(root.path(), "search.max_k").unwrap().value, 20);
        assert_eq!(get(root.path(), "search.default_k").unwrap().value, 5);
    }
}
