//! Injected init decisions and verified publication, with isolated child environments.
#![cfg(unix)]
use super::*;
use std::{
    cell::RefCell,
    collections::{BTreeSet, VecDeque},
    path::PathBuf,
};

const CHILD: &str = "DKC_INIT_TEST_CHILD";
const TEST_KEY_ENV: &str = "DKC_INIT_TEST_PROVIDER_KEY";

// A child environment prevents developer credentials/config from affecting prompt
// decisions, without mutating process globals while other Rust tests run.
fn isolated(name: &str) -> bool {
    let full = format!("cli::init::tests::{name}");
    if std::env::var(CHILD).as_deref() == Ok(full.as_str()) {
        return false;
    }
    let user = tempfile::tempdir().unwrap();
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().starts_with("DKC_") {
            command.env_remove(name);
        }
    }
    command
        .args(["--exact", &full, "--nocapture"])
        .env(CHILD, &full)
        .env("XDG_CONFIG_HOME", user.path())
        .env_remove("MOONLY_GATEWAY_API_KEY")
        .env_remove("DKC_REMOTE_API_KEY")
        .env_remove(TEST_KEY_ENV);
    if name == "environment_key_precedes_file_and_file_precedes_prompt" {
        command.env(TEST_KEY_ENV, synthetic("environment"));
    }
    if name == "inherited_http_token_is_never_published_into_repository_configuration" {
        command.env("DKC_SERVE__HTTP_TOKEN", synthetic("http-token"));
    }
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
    true
}
fn synthetic(label: &str) -> String {
    ["fixture", label].join("-")
}

struct Input {
    interactive: bool,
    selected: String,
    profile_calls: usize,
    texts: VecDeque<String>,
    text_calls: usize,
    secret_calls: Vec<bool>,
}
impl Input {
    fn new(interactive: bool, selected: &str) -> Self {
        Self {
            interactive,
            selected: selected.into(),
            profile_calls: 0,
            texts: VecDeque::new(),
            text_calls: 0,
            secret_calls: Vec::new(),
        }
    }
}
impl Terminal for Input {
    fn interactive(&self) -> bool {
        self.interactive
    }
    fn profile(&mut self) -> Result<String> {
        self.profile_calls += 1;
        Ok(self.selected.clone())
    }
    fn text(&mut self, _prompt: &str, _optional: bool) -> Result<String> {
        self.text_calls += 1;
        self.texts
            .pop_front()
            .ok_or_else(|| anyhow!("unexpected text prompt"))
    }
    fn secret(&mut self, _prompt: &str, stdin: bool) -> Result<String> {
        self.secret_calls.push(stdin);
        Ok(synthetic("prompt"))
    }
}
struct Seen {
    url: String,
    models: BTreeSet<String>,
}
#[derive(Default)]
struct RecordingProbe {
    seen: RefCell<Vec<Seen>>,
    expected_key: Option<String>,
    failure: Option<&'static str>,
}
impl probe::Probe for RecordingProbe {
    fn verify(&self, target: &probe::Target, key: &str) -> Result<()> {
        assert!(!key.is_empty());
        if let Some(expected) = &self.expected_key {
            assert!(
                key == expected,
                "credential precedence did not select expected source"
            );
        }
        self.seen.borrow_mut().push(Seen {
            url: target.url.to_string(),
            models: target.models.clone(),
        });
        if let Some(message) = self.failure {
            bail!("{message}");
        }
        Ok(())
    }
}
struct Fixture {
    repo: tempfile::TempDir,
    external: tempfile::TempDir,
}
impl Fixture {
    fn new() -> Self {
        let repo = tempfile::tempdir().unwrap();
        fs::create_dir_all(repo.path().join(".git/info")).unwrap();
        Self {
            repo,
            external: tempfile::tempdir().unwrap(),
        }
    }
    fn root(&self) -> &Path {
        self.repo.path()
    }
    fn credentials(&self) -> PathBuf {
        self.external.path().join("credentials.toml")
    }
    fn config(&self) -> PathBuf {
        self.root().join(".deepkit/config.toml")
    }
    fn exclude(&self) -> PathBuf {
        self.root().join(".git/info/exclude")
    }
    fn write_config(&self, source: &str) {
        fs::create_dir_all(self.root().join(".deepkit")).unwrap();
        fs::write(self.config(), source).unwrap();
    }
    fn settings(&self) -> Config {
        config::load(self.root(), None).unwrap().config
    }
    fn run(
        &self,
        options: &Options,
        input: &mut Input,
        probe: &RecordingProbe,
    ) -> Result<serde_json::Value> {
        execute(
            self.root(),
            options,
            input,
            probe,
            Some(&self.credentials()),
        )
    }
    fn snapshot(&self) -> [Option<Vec<u8>>; 3] {
        [self.config(), self.credentials(), self.exclude()].map(|path| fs::read(path).ok())
    }
    fn no_models_or_tempfiles(&self) {
        for directory in [
            self.root().join(".deepkit"),
            self.external.path().to_path_buf(),
        ] {
            if directory.exists() {
                assert!(fs::read_dir(directory).unwrap().all(|entry| {
                    let name = entry.unwrap().file_name();
                    let name = name.to_string_lossy();
                    !name.ends_with(".tmp") && !name.ends_with(".gguf") && name != "code.db"
                }));
            }
        }
    }
}
fn existing_remote(env: &str) -> String {
    format!(
        "# committed settings\n[index]\nrepo_name='preserved'\n[embedder]\nprofile='remote'\n[embedder.remote]\nurl='https://fixture.example/v1'\nmodel='embedding'\napi_key_env='{env}'\n[rerank]\nmode='off'\n[llm]\nenabled=false\n"
    )
}

#[test]
fn four_interactive_choices_select_expected_recipes_and_only_remote_probes() {
    if isolated("four_interactive_choices_select_expected_recipes_and_only_remote_probes") {
        return;
    }
    for profile in ["moonly", "local", "ollama", "remote"] {
        let fixture = Fixture::new();
        let mut input = Input::new(true, profile);
        if profile == "remote" {
            input.texts = [
                "https://fixture.example/v1",
                "custom-embed",
                "custom-rank",
                "custom-chat",
                "custom-vision",
            ]
            .map(str::to_owned)
            .into();
        }
        let probe = RecordingProbe::default();
        let value = fixture
            .run(&Options::default(), &mut input, &probe)
            .unwrap();
        let settings = fixture.settings();
        assert_eq!(input.profile_calls, 1);
        assert_eq!(
            settings.embedder.profile,
            if profile == "moonly" {
                "remote"
            } else {
                profile
            }
        );
        assert!(settings.embedder.docs.is_none());
        if matches!(profile, "local" | "ollama") {
            assert!(probe.seen.borrow().is_empty());
            assert!(input.secret_calls.is_empty());
            assert!(!fixture.credentials().exists());
            assert!(!settings.llm.enabled);
        } else {
            assert_eq!(probe.seen.borrow().len(), 1);
            assert_eq!(input.secret_calls, [false]);
            assert_eq!(value["credentials_written"], 1);
            assert!(settings.llm.enabled);
            let seen = probe.seen.borrow();
            let expected: BTreeSet<String> = if profile == "moonly" {
                ["qwen3-embedding-8b", "qwen3-reranker-4b", "Qwen3.5-35B-A3B"]
                    .map(str::to_owned)
                    .into()
            } else {
                [
                    "custom-embed",
                    "custom-rank",
                    "custom-chat",
                    "custom-vision",
                ]
                .map(str::to_owned)
                .into()
            };
            assert_eq!(seen[0].models, expected);
            assert!(seen[0].url.ends_with("/v1/models"));
        }
        assert_eq!(input.text_calls, if profile == "remote" { 5 } else { 0 });
        fixture.no_models_or_tempfiles();
    }
}

#[test]
fn noninteractive_and_yes_default_to_local_without_prompts_or_provider_work() {
    if isolated("noninteractive_and_yes_default_to_local_without_prompts_or_provider_work") {
        return;
    }
    for (interactive, yes) in [(false, false), (true, true)] {
        let fixture = Fixture::new();
        let mut input = Input::new(interactive, "moonly");
        let probe = RecordingProbe::default();
        let value = fixture
            .run(
                &Options {
                    yes,
                    ..Options::default()
                },
                &mut input,
                &probe,
            )
            .unwrap();
        assert_eq!(value["profile"], "local");
        assert_eq!(input.profile_calls, 0);
        assert_eq!(input.text_calls, 0);
        assert!(input.secret_calls.is_empty());
        assert!(probe.seen.borrow().is_empty());
        fixture.no_models_or_tempfiles();
    }
}

#[test]
fn codex_setup_reports_only_selected_file_without_model_work() {
    if isolated("codex_setup_reports_only_selected_file_without_model_work") {
        return;
    }
    let fixture = Fixture::new();
    let mut input = Input::new(false, "unused");
    let probe = RecordingProbe::default();
    let value = fixture
        .run(
            &Options {
                yes: true,
                codex: true,
                ..Options::default()
            },
            &mut input,
            &probe,
        )
        .unwrap();
    let path = fixture.root().join(".codex/config.toml");
    assert_eq!(value["mcp"], json!([path]));
    let configuration = codex::parse(&fs::read(path).unwrap()).unwrap();
    assert_eq!(
        configuration["mcp_servers"]["deepkit-code"]["cwd"].as_str(),
        fixture.root().canonicalize().unwrap().to_str()
    );
    assert_eq!(input.profile_calls, 0);
    assert!(probe.seen.borrow().is_empty());
    assert!(!fixture.root().join(".cursor").exists());
    assert!(!fixture.root().join(".mcp.json").exists());
    fixture.no_models_or_tempfiles();
}

#[test]
fn existing_gateway_only_requests_key_without_rewriting_committed_configuration() {
    if isolated("existing_gateway_only_requests_key_without_rewriting_committed_configuration") {
        return;
    }
    let fixture = Fixture::new();
    let mut settings = Config::default();
    presets::apply(&mut settings, "moonly", false, None).unwrap();
    settings.index.repo_name = "committed-name".into();
    settings.limits.query_timeout_ms = 3210;
    fixture.write_config(&format!(
        "# human formatting must survive\n{}",
        settings.dump().unwrap()
    ));
    let original = fs::read(fixture.config()).unwrap();
    let mut input = Input::new(true, "local");
    let probe = RecordingProbe::default();
    let value = fixture
        .run(&Options::default(), &mut input, &probe)
        .unwrap();
    assert_eq!(value["configuration_written"], false);
    assert_eq!(input.profile_calls, 0);
    assert_eq!(input.secret_calls, [false]);
    assert_eq!(fs::read(fixture.config()).unwrap(), original);
    assert_eq!(fixture.settings(), settings);
    assert_eq!(probe.seen.borrow().len(), 1);
    fixture.no_models_or_tempfiles();
}

#[test]
fn local_with_moonly_docs_preserves_limits_and_queries_only_selected_remote_roles() {
    if isolated("local_with_moonly_docs_preserves_limits_and_queries_only_selected_remote_roles") {
        return;
    }
    let fixture = Fixture::new();
    fixture.write_config("[index]\nrepo_name='kept'\n[embedder]\nprofile='local'\n[embedder.local]\ngpu_layers=0\n[limits]\nquery_timeout_ms=4321\n");
    let mut input = Input::new(true, "unused");
    let probe = RecordingProbe::default();
    fixture
        .run(
            &Options {
                profile: Some("local".into()),
                docs: Some("moonly".into()),
                ..Options::default()
            },
            &mut input,
            &probe,
        )
        .unwrap();
    let settings = fixture.settings();
    assert_eq!(settings.embedder.profile, "local");
    assert_eq!(settings.embedder.docs.unwrap().profile, "remote");
    assert_eq!(settings.embedder.local.gpu_layers, 0);
    assert_eq!(settings.index.repo_name, "kept");
    assert_eq!(settings.limits.query_timeout_ms, 4321);
    let seen = probe.seen.borrow();
    assert_eq!(seen.len(), 1);
    assert_eq!(
        seen[0].models,
        ["qwen3-embedding-8b", "Qwen3.5-35B-A3B"]
            .map(str::to_owned)
            .into()
    );
    assert_eq!(input.profile_calls, 0);
    fixture.no_models_or_tempfiles();
}

#[test]
fn environment_key_precedes_file_and_file_precedes_prompt() {
    if isolated("environment_key_precedes_file_and_file_precedes_prompt") {
        return;
    }
    for (env, expected) in [
        (TEST_KEY_ENV, synthetic("environment")),
        ("DKC_INIT_TEST_ABSENT_KEY", synthetic("stored")),
    ] {
        let fixture = Fixture::new();
        fixture.write_config(&existing_remote(env));
        config::credentials::upsert(
            fixture.root(),
            &fixture.credentials(),
            "fixture.example",
            &synthetic("stored"),
        )
        .unwrap();
        let before = fs::read(fixture.credentials()).unwrap();
        let mut input = Input::new(true, "local");
        let probe = RecordingProbe {
            expected_key: Some(expected),
            ..RecordingProbe::default()
        };
        let value = fixture
            .run(&Options::default(), &mut input, &probe)
            .unwrap();
        assert_eq!(value["credentials_written"], 0);
        assert_eq!(probe.seen.borrow().len(), 1);
        assert!(input.secret_calls.is_empty());
        assert_eq!(fs::read(fixture.credentials()).unwrap(), before);
        fixture.no_models_or_tempfiles();
    }
}

#[test]
fn missing_noninteractive_key_is_refused_but_explicit_stdin_is_used() {
    if isolated("missing_noninteractive_key_is_refused_but_explicit_stdin_is_used") {
        return;
    }
    let fixture = Fixture::new();
    fixture.write_config(&existing_remote("DKC_INIT_TEST_ABSENT_KEY"));
    let before = fixture.snapshot();
    let mut input = Input::new(false, "unused");
    let probe = RecordingProbe::default();
    let error = fixture
        .run(&Options::default(), &mut input, &probe)
        .unwrap_err();
    assert!(error.to_string().contains("API key is missing"));
    assert_eq!(fixture.snapshot(), before);
    assert!(input.secret_calls.is_empty());
    assert!(probe.seen.borrow().is_empty());
    let value = fixture
        .run(
            &Options {
                api_key_stdin: true,
                ..Options::default()
            },
            &mut input,
            &probe,
        )
        .unwrap();
    assert_eq!(input.secret_calls, [true]);
    assert_eq!(value["credentials_written"], 1);
    assert_eq!(value["configuration_written"], false);
    fixture.no_models_or_tempfiles();
}

#[test]
fn probe_failures_leave_configuration_credentials_and_excludes_unchanged() {
    if isolated("probe_failures_leave_configuration_credentials_and_excludes_unchanged") {
        return;
    }
    for message in [
        "The provider rejected the API key (401)",
        "Model discovery could not reach the configured provider",
        "Configured model is missing from provider discovery",
    ] {
        let fixture = Fixture::new();
        fixture.write_config("# original\n[embedder]\nprofile='local'\n");
        fs::write(fixture.exclude(), "# original excludes\n").unwrap();
        config::credentials::upsert(
            fixture.root(),
            &fixture.credentials(),
            "other.example",
            &synthetic("other"),
        )
        .unwrap();
        let before = fixture.snapshot();
        let mut input = Input::new(true, "unused");
        let probe = RecordingProbe {
            failure: Some(message),
            ..RecordingProbe::default()
        };
        let error = fixture
            .run(
                &Options {
                    profile: Some("moonly".into()),
                    ..Options::default()
                },
                &mut input,
                &probe,
            )
            .unwrap_err();
        assert_eq!(error.to_string(), message);
        assert!(!error.to_string().contains(&synthetic("prompt")));
        assert_eq!(input.secret_calls, [false]);
        assert_eq!(fixture.snapshot(), before);
        fixture.no_models_or_tempfiles();
    }
}

#[test]
fn repository_secret_fields_are_reported_without_the_literal_or_provider_activity() {
    if isolated("repository_secret_fields_are_reported_without_the_literal_or_provider_activity") {
        return;
    }
    for name in ["config.toml", "config.local.toml"] {
        let fixture = Fixture::new();
        fixture.write_config("[embedder]\nprofile='local'\n");
        let path = fixture.root().join(".deepkit").join(name);
        let key = synthetic("forbidden");
        let text = format!("[embedder.remote]\napi_key='{key}'\n");
        fs::write(&path, &text).unwrap();
        let mut input = Input::new(false, "unused");
        let probe = RecordingProbe::default();
        let error = fixture
            .run(&Options::default(), &mut input, &probe)
            .unwrap_err();
        assert!(error.to_string().contains("secret field"));
        assert!(!format!("{error:#}").contains(&key));
        assert!(probe.seen.borrow().is_empty());
        assert_eq!(fs::read_to_string(path).unwrap(), text);
        assert!(!fixture.credentials().exists());
        assert!(!fixture.exclude().exists());
    }
}

#[test]
fn inherited_http_token_is_never_published_into_repository_configuration() {
    if isolated("inherited_http_token_is_never_published_into_repository_configuration") {
        return;
    }
    let fixture = Fixture::new();
    let mut input = Input::new(false, "unused");
    let probe = RecordingProbe::default();
    fixture
        .run(&Options::default(), &mut input, &probe)
        .unwrap();
    let published = fs::read_to_string(fixture.config()).unwrap();
    assert!(!published.contains(&synthetic("http-token")));
    let parsed: toml::Value = toml::from_str(&published).unwrap();
    assert!(
        parsed
            .get("serve")
            .and_then(|serve| serve.get("http_token"))
            .and_then(toml::Value::as_str)
            .is_none_or(str::is_empty)
    );
    assert!(probe.seen.borrow().is_empty());
    assert!(!fixture.credentials().exists());
}

#[test]
fn conflicting_local_override_refuses_publication_without_changing_any_files() {
    if isolated("conflicting_local_override_refuses_publication_without_changing_any_files") {
        return;
    }
    let fixture = Fixture::new();
    fixture.write_config("[embedder]\nprofile='local'\n");
    let local = fixture.root().join(".deepkit/config.local.toml");
    let override_text = "[embedder]\nprofile='ollama'\n";
    fs::write(&local, override_text).unwrap();
    let before = fixture.snapshot();
    let mut input = Input::new(false, "unused");
    let probe = RecordingProbe::default();
    assert!(
        fixture
            .run(
                &Options {
                    profile: Some("local".into()),
                    ..Options::default()
                },
                &mut input,
                &probe
            )
            .is_err()
    );
    assert_eq!(fixture.snapshot(), before);
    assert_eq!(fs::read_to_string(local).unwrap(), override_text);
    assert!(probe.seen.borrow().is_empty());
    assert!(input.secret_calls.is_empty());
    fixture.no_models_or_tempfiles();
}

#[test]
fn ignored_model_flags_and_credential_bearing_ollama_urls_are_refused() {
    if isolated("ignored_model_flags_and_credential_bearing_ollama_urls_are_refused") {
        return;
    }
    let unsafe_url = format!("https://user:{}@fixture.example/api", synthetic("url"));
    let cases = [
        Options {
            profile: Some("local".into()),
            rerank_model: Some("ignored-ranker".into()),
            ..Options::default()
        },
        Options {
            profile: Some("moonly".into()),
            llm_model: Some("ignored-chat".into()),
            ..Options::default()
        },
        Options {
            profile: Some("ollama".into()),
            vlm_model: Some("ignored-vision".into()),
            ..Options::default()
        },
        Options {
            profile: Some("ollama".into()),
            url: Some(unsafe_url.clone()),
            ..Options::default()
        },
    ];
    for options in cases {
        let fixture = Fixture::new();
        let before = fixture.snapshot();
        let mut input = Input::new(false, "unused");
        let probe = RecordingProbe::default();
        let error = fixture.run(&options, &mut input, &probe).unwrap_err();
        assert!(!format!("{error:#}").contains(&unsafe_url));
        assert!(!format!("{error:#}").contains(&synthetic("url")));
        assert_eq!(fixture.snapshot(), before);
        assert!(probe.seen.borrow().is_empty());
        assert!(input.secret_calls.is_empty());
        fixture.no_models_or_tempfiles();
    }
}

#[test]
fn stdin_key_bound_keeps_overflow_and_multiline_evidence() {
    let maximum = "x".repeat(4096);
    for ending in ["", "\n", "\r\n"] {
        assert!(read_stdin_key(format!("{maximum}{ending}").as_bytes()).is_ok());
    }
    for value in [
        format!("{maximum}\nextra"),
        format!("{maximum}\r\nextra"),
        "x".repeat(4097),
        "first\nsecond".into(),
        "first\n\n".into(),
        String::new(),
    ] {
        assert!(read_stdin_key(value.as_bytes()).is_err());
    }
}

#[test]
fn publication_rejects_secret_field_types_and_preserves_multiline_payloads() {
    for key in [
        "api_key",
        "refresh_token",
        "private_key",
        "credentials",
        "authorization",
    ] {
        for value in [
            json!("fixture"),
            json!(["fixture"]),
            json!({"nested":"fixture"}),
            json!(42),
        ] {
            let body: toml::Value = serde_json::from_value(json!({(key):value})).unwrap();
            assert!(config::contains_secret(&body));
            let mut settings = Config::default();
            settings.llm.extra_body = body;
            assert!(publication(&settings).is_err());
        }
    }
    let mut settings = Config::default();
    let token = synthetic("external-http");
    settings.serve.http_token.clone_from(&token);
    let payload = "before\n[serve]\nhttp_token = retained text\nafter";
    settings
        .llm
        .extra_body
        .as_table_mut()
        .unwrap()
        .insert("payload".into(), toml::Value::String(payload.into()));
    let text = publication(&settings).unwrap();
    assert!(!text.contains(&token));
    let value: toml::Value = toml::from_str(&text).unwrap();
    assert!(value["serve"].get("http_token").is_none());
    assert_eq!(
        value["llm"]["extra_body"]["payload"].as_str(),
        Some(payload)
    );
}

#[test]
fn user_config_http_token_remains_external_and_effective() {
    if isolated("user_config_http_token_remains_external_and_effective") {
        return;
    }
    let user = PathBuf::from(std::env::var_os("XDG_CONFIG_HOME").unwrap()).join("deepkit-code");
    fs::create_dir_all(&user).unwrap();
    let token = synthetic("user-config-http");
    let data = toml::to_string(&serde_json::json!({"serve":{"http_token":token}})).unwrap();
    fs::write(user.join("config.toml"), data).unwrap();
    let fixture = Fixture::new();
    let result = fixture.run(
        &Options::default(),
        &mut Input::new(false, "local"),
        &RecordingProbe::default(),
    );
    assert!(result.is_ok());
    assert!(fixture.settings().serve.http_token == token);
    assert!(
        !fs::read_to_string(fixture.config())
            .unwrap()
            .contains(&token)
    );
}
