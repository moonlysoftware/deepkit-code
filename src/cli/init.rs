//! Interactive presets and verified external credentials for init; v1 §§7.2/10.2/11.
pub(super) mod codex;
mod files;
pub(super) mod ide;
mod presets;
pub(crate) mod probe;
#[cfg(test)]
mod tests;

use crate::{
    config::{self, Config},
    models::openai_compat::resolve_api_key,
};
use anyhow::{Result, anyhow, bail};
use clap::Args;
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    io::{self, IsTerminal, Read},
    path::Path,
};

/// Init accepts credential references or stdin, never a key command-line argument.
#[derive(Args, Default)]
pub(crate) struct Options {
    #[arg(long, value_parser = ["local", "moonly", "ollama", "remote"])]
    profile: Option<String>,
    /// Adds the project MCP entry to .cursor/mcp.json.
    #[arg(long)]
    cursor: bool,
    /// Adds the project MCP entry to .mcp.json for Claude Code.
    #[arg(long)]
    claude: bool,
    /// Adds a repository-bound stdio MCP entry to .codex/config.toml.
    #[arg(long)]
    codex: bool,
    #[arg(long, value_parser = ["moonly"])]
    docs: Option<String>,
    /// Uses noninteractive defaults and refuses prompts for missing remote settings.
    #[arg(long)]
    yes: bool,
    /// Reads a missing provider key from stdin instead of a hidden terminal prompt.
    #[arg(long)]
    api_key_stdin: bool,
    /// Base URL for a custom remote or Ollama provider.
    #[arg(long)]
    url: Option<String>,
    /// Embedding model name for a custom remote or Ollama provider.
    #[arg(long)]
    model: Option<String>,
    #[arg(long)]
    rerank_model: Option<String>,
    #[arg(long)]
    llm_model: Option<String>,
    #[arg(long)]
    vlm_model: Option<String>,
    #[arg(long)]
    gitignore: bool,
    #[arg(long)]
    json: bool,
}

trait Terminal {
    fn interactive(&self) -> bool;
    fn profile(&mut self) -> Result<String>;
    fn text(&mut self, prompt: &str, optional: bool) -> Result<String>;
    fn secret(&mut self, prompt: &str, stdin: bool) -> Result<String>;
}

struct Console;
impl Terminal for Console {
    fn interactive(&self) -> bool {
        io::stdin().is_terminal() && io::stderr().is_terminal()
    }
    fn profile(&mut self) -> Result<String> {
        let index = dialoguer::Select::new()
            .with_prompt("Choose how Deepkit Code runs models")
            .items([
                "Moonly gateway — remote embedding and enrichment",
                "Local — offline models, no API key",
                "Ollama — use an existing local embedding model",
                "Custom remote — configure an OpenAI-compatible provider",
            ])
            .default(0)
            .interact()?;
        ["moonly", "local", "ollama", "remote"]
            .get(index)
            .map(|profile| (*profile).into())
            .ok_or_else(|| anyhow!("Invalid profile selection"))
    }
    fn text(&mut self, prompt: &str, optional: bool) -> Result<String> {
        Ok(dialoguer::Input::<String>::new()
            .with_prompt(prompt)
            .allow_empty(optional)
            .interact_text()?)
    }
    fn secret(&mut self, prompt: &str, stdin: bool) -> Result<String> {
        let value = if stdin {
            if io::stdin().is_terminal() {
                bail!(
                    "--api-key-stdin expects piped input; omit it to use a hidden terminal prompt"
                );
            }
            read_stdin_key(io::stdin())?
        } else {
            dialoguer::Password::new()
                .with_prompt(prompt)
                .report(false)
                .interact()?
        };
        if value.is_empty() || value.len() > 4096 || value.chars().any(char::is_control) {
            bail!("API key must be a nonempty single line of at most 4096 bytes");
        }
        Ok(value)
    }
}

/// Runs the injected flow with the real terminal and bounded model-discovery client.
pub(crate) fn run(options: Options) -> Result<()> {
    let root = super::current_dir()?;
    let value = execute(
        &root,
        &options,
        &mut Console,
        &probe::HttpProbe,
        config::credentials_path().as_deref(),
    )?;
    super::emit(
        options.json,
        value,
        &format!(
            "initialized {}",
            root.join(".deepkit/config.toml").display()
        ),
    )?;
    Ok(())
}

fn execute(
    root: &Path,
    options: &Options,
    terminal: &mut dyn Terminal,
    probe: &dyn probe::Probe,
    credentials: Option<&Path>,
) -> Result<serde_json::Value> {
    validate_options(options)?;
    let layers = repository_settings(root)?;
    let original = layers.repository.clone();
    let mut settings = layers.load(root)?;
    let (profile, publish) = select_profile(options, original.is_some(), terminal)?;
    if let Some(profile) = &profile {
        let custom = if profile == "remote" {
            Some(remote_settings(options, terminal)?)
        } else {
            None
        };
        if profile == "ollama" {
            if let Some(url) = &options.url {
                settings.embedder.ollama.url.clone_from(url);
            }
            if let Some(model) = &options.model {
                settings.embedder.ollama.model.clone_from(model);
            }
        } else if profile != "remote" && (options.url.is_some() || options.model.is_some()) {
            bail!("--url and --model apply only to remote or Ollama profiles");
        }
        presets::apply(
            &mut settings,
            profile,
            options.docs.is_some(),
            custom.as_ref(),
        )?;
    } else if options.docs.is_some() {
        let profile = settings.embedder.profile.clone();
        presets::apply(&mut settings, &profile, true, None)?;
    }
    let proposed = (publish || options.docs.is_some())
        .then(|| publication(&settings))
        .transpose()?;
    validate_effective(root, &settings, proposed.as_deref())?;
    let prepared = files::prepare(
        root,
        options.gitignore,
        original.as_deref(),
        proposed.as_deref(),
    )?
    .with_mcp(options.cursor, options.claude)?
    .with_codex(root, options.codex)?;
    let targets = probe::targets(&settings)?;
    if !targets.is_empty()
        && let Some(path) = credentials
    {
        require_external(root, path)?;
    }
    // Verify every required endpoint before publishing any configuration or key.
    let mut pending = BTreeMap::<String, String>::new();
    let mut stdin_host = None::<String>;
    for target in &targets {
        let host = target
            .url
            .host_str()
            .ok_or_else(|| anyhow!("Provider URL has no host"))?;
        let key = match resolve_api_key(&target.api_key_env, &target.url, credentials) {
            Ok(key) => key,
            Err(crate::models::Error::MissingApiKey { .. }) => {
                if let Some(key) = pending.get(host) {
                    key.clone()
                } else {
                    if !options.api_key_stdin && (options.yes || !terminal.interactive()) {
                        bail!(
                            "API key is missing; set {} or use --api-key-stdin",
                            target.api_key_env
                        );
                    }
                    if options.api_key_stdin && stdin_host.as_deref().is_some_and(|old| old != host)
                    {
                        bail!("--api-key-stdin supports one missing provider host per invocation");
                    }
                    if credentials.is_none() {
                        bail!("Could not locate the external credentials file");
                    }
                    let key =
                        terminal.secret(&format!("API key for {host}"), options.api_key_stdin)?;
                    if options.api_key_stdin {
                        stdin_host = Some(host.into());
                    }
                    pending.insert(host.into(), key.clone());
                    key
                }
            }
            Err(error) => return Err(error.into()),
        };
        probe.verify(target, &key)?;
    }
    validate_effective(root, &settings, proposed.as_deref())?;
    prepared.check()?;
    for (host, key) in &pending {
        config::credentials::upsert(
            root,
            credentials.ok_or_else(|| anyhow!("Missing external credentials location"))?,
            host,
            key,
        )?;
    }
    prepared.publish()?;
    let exclude = if options.gitignore {
        root.join(".gitignore")
    } else {
        root.join(".git/info/exclude")
    };
    let mcp = [
        (options.cursor, ".cursor/mcp.json"),
        (options.claude, ".mcp.json"),
        (options.codex, ".codex/config.toml"),
    ]
    .into_iter()
    .filter(|(selected, _)| *selected)
    .map(|(_, path)| root.join(path))
    .collect::<Vec<_>>();
    Ok(
        json!({"mcp":mcp,"config":root.join(".deepkit/config.toml"), "exclude":exclude,
        "profile":settings.embedder.profile, "docs":settings.embedder.docs,
        "configuration_written":publish || options.docs.is_some(),
        "verified_endpoints":targets.len(), "credentials_written":pending.len()}),
    )
}

fn select_profile(
    options: &Options,
    existing: bool,
    terminal: &mut dyn Terminal,
) -> Result<(Option<String>, bool)> {
    if let Some(profile) = &options.profile {
        return Ok((Some(profile.clone()), true));
    }
    if existing {
        return Ok((None, false));
    }
    if options.yes || !terminal.interactive() {
        return Ok((Some("local".into()), true));
    }
    Ok((Some(terminal.profile()?), true))
}

fn remote_settings(options: &Options, terminal: &mut dyn Terminal) -> Result<presets::Remote> {
    let interactive = !options.yes && terminal.interactive();
    let mut value =
        |supplied: &Option<String>, prompt: &str, optional: bool| -> Result<Option<String>> {
            let value = match supplied {
                Some(value) => Some(value.clone()),
                None if interactive => Some(terminal.text(prompt, optional)?),
                None if optional => None,
                None => bail!("Custom remote needs --url and --model outside a terminal"),
            };
            Ok(value.filter(|value| !value.trim().is_empty()))
        };
    let url = value(&options.url, "Provider base URL", false)?
        .ok_or_else(|| anyhow!("Provider URL is required"))?;
    let embedding_model = value(&options.model, "Embedding model", false)?
        .ok_or_else(|| anyhow!("Embedding model is required"))?;
    let rerank_model = value(
        &options.rerank_model,
        "Rerank model (empty to disable)",
        true,
    )?;
    let llm_model = value(
        &options.llm_model,
        "Chat model (empty to disable enrichment)",
        true,
    )?;
    let vlm_model = if llm_model.is_some() {
        value(&options.vlm_model, "Vision model", false)?
    } else {
        options.vlm_model.clone()
    };
    Ok(presets::Remote {
        url,
        embedding_model,
        rerank_model,
        llm_model,
        vlm_model,
    })
}

struct Layers {
    repository: Option<String>,
    local: Option<String>,
}
impl Layers {
    fn load(&self, root: &Path) -> Result<Config> {
        let parse = |text: &str| {
            toml::from_str(text).map_err(|_| anyhow!("Repository configuration is not valid TOML"))
        };
        config::load_init_snapshots(
            root,
            self.repository.as_deref().map(parse).transpose()?,
            self.local.as_deref().map(parse).transpose()?,
        )
        .map(|loaded| loaded.config)
        .map_err(|_| anyhow!("Could not read effective init configuration"))
    }
}
fn repository_settings(root: &Path) -> Result<Layers> {
    let repository = files::read_configuration(root, "config.toml")?;
    let local = files::read_configuration(root, "config.local.toml")?;
    for source in [repository.as_deref(), local.as_deref()]
        .into_iter()
        .flatten()
    {
        let parsed: toml::Value = toml::from_str(source)
            .map_err(|_| anyhow!("Repository configuration is not valid TOML"))?;
        if config::contains_secret(&parsed) {
            bail!(
                "Repository configuration contains a secret field; move keys to the external credentials file"
            );
        }
    }
    Ok(Layers { repository, local })
}

fn read_stdin_key(input: impl Read) -> Result<String> {
    // One optional LF/CRLF plus an overflow byte; never trim away overflow evidence.
    let mut raw = String::new();
    input
        .take(4099)
        .read_to_string(&mut raw)
        .map_err(|_| anyhow!("Could not read API key from stdin"))?;
    if raw.len() > 4098 {
        bail!("API key input exceeds the single-line limit");
    }
    let key = raw
        .strip_suffix("\r\n")
        .or_else(|| raw.strip_suffix('\n'))
        .unwrap_or(&raw);
    if key.is_empty() || key.len() > 4096 || key.chars().any(char::is_control) {
        bail!("API key must be a nonempty single line of at most 4096 bytes");
    }
    Ok(key.into())
}

fn require_external(root: &Path, path: &Path) -> Result<()> {
    let root = fs::canonicalize(root)?;
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut ancestor = absolute.as_path();
    while !ancestor.exists() {
        ancestor = ancestor
            .parent()
            .ok_or_else(|| anyhow!("Invalid credentials location"))?;
    }
    if fs::canonicalize(ancestor)?.starts_with(&root) || absolute.starts_with(&root) {
        bail!("Credentials must be outside the repository");
    }
    Ok(())
}

fn validate_options(options: &Options) -> Result<()> {
    let custom = options.profile.as_deref() == Some("remote");
    let ollama = options.profile.as_deref() == Some("ollama");
    if (options.url.is_some() || options.model.is_some()) && !custom && !ollama {
        bail!("--url and --model require an explicit remote or Ollama profile");
    }
    if (options.rerank_model.is_some()
        || options.llm_model.is_some()
        || options.vlm_model.is_some())
        && !custom
    {
        bail!("Role model options require --profile remote");
    }
    if let Some(url) = &options.url {
        config::queue::embedding_endpoint(url, "models")?;
    }
    if options
        .model
        .as_ref()
        .is_some_and(|model| model.trim().is_empty())
    {
        bail!("Embedding model must not be empty");
    }
    Ok(())
}

fn publication(settings: &Config) -> Result<String> {
    let mut public = settings.clone();
    // HTTP authentication stays in the user's environment/configuration layer.
    public.serve.http_token.clear();
    for url in [
        &public.embedder.remote.url,
        &public.embedder.ollama.url,
        &public.rerank.remote.url,
        &public.llm.url,
    ] {
        config::queue::embedding_endpoint(url, "models")?;
    }
    // Structural omission preserves user-layer auth and arbitrary multiline values.
    let mut value = toml::Value::try_from(&public)?;
    if let Some(serve) = value.get_mut("serve").and_then(toml::Value::as_table_mut) {
        serve.remove("http_token");
    }
    if config::contains_secret(&value) {
        bail!(
            "Effective configuration contains secret fields that cannot be written into the repository"
        );
    }
    Ok(config::dump_commented_value(&value)?)
}

fn validate_effective(root: &Path, selected: &Config, proposed: Option<&str>) -> Result<()> {
    let mut layers = repository_settings(root)?;
    if let Some(text) = proposed {
        layers.repository = Some(text.into());
    }
    let effective = layers.load(root)?;
    if effective != *selected {
        bail!(
            "Local or environment configuration overrides the selected profile/settings; resolve the override before retrying init"
        );
    }
    Ok(())
}
