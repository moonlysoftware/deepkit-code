//! Layered configuration, its provenance, and the `config` command; see `v1.md` §11.

pub(crate) mod agent;
pub(crate) mod credentials;
pub(crate) mod pr_patterns;
pub(crate) mod queue;
pub(crate) mod reload;
pub(crate) use queue::Queue;

use std::{
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

use figment::{
    Figment, Metadata, Profile, Provider,
    providers::{Env, Format, Toml},
    value::{Dict, Map, Value},
};
use serde::{Deserialize, Serialize};

/// A binary byte quantity accepted from a human-readable configuration string.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub(crate) struct Bytes(pub(crate) u64);

/// Parses binary quantities such as `64 MiB` and `2 GiB`.
pub(crate) fn parse_bytes(input: &str) -> Result<Bytes, Error> {
    let (number, unit) = input
        .trim()
        .split_once(char::is_whitespace)
        .ok_or_else(|| Error::InvalidBytes(input.to_owned()))?;
    let number = number
        .parse::<u64>()
        .map_err(|_| Error::InvalidBytes(input.to_owned()))?;
    let multiplier = match unit.trim() {
        "B" => 1,
        "KiB" => 1024,
        "MiB" => 1024 * 1024,
        "GiB" => 1024 * 1024 * 1024,
        _ => return Err(Error::InvalidBytes(input.to_owned())),
    };
    number
        .checked_mul(multiplier)
        .map(Bytes)
        .ok_or_else(|| Error::InvalidBytes(input.to_owned()))
}

/// Parses the documented compact durations, including millisecond timeouts.
pub(crate) fn parse_duration(input: &str) -> Result<Duration, Error> {
    let input = input.trim();
    let (number, multiplier, milliseconds) = if let Some(number) = input.strip_suffix("ms") {
        (number, 1, true)
    } else {
        match input.chars().last() {
            Some('s') => (&input[..input.len() - 1], 1, false),
            Some('m') => (&input[..input.len() - 1], 60, false),
            Some('h') => (&input[..input.len() - 1], 60 * 60, false),
            _ => return Err(Error::InvalidDuration(input.to_owned())),
        }
    };
    let amount = number
        .parse::<u64>()
        .ok()
        .and_then(|number| number.checked_mul(multiplier))
        .ok_or_else(|| Error::InvalidDuration(input.to_owned()))?;
    Ok(if milliseconds {
        Duration::from_millis(amount)
    } else {
        Duration::from_secs(amount)
    })
}

macro_rules! section {
    ($name:ident { $($field:ident: $kind:ty = $value:expr),+ $(,)? }) => {
        #[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
        #[serde(default)]
        pub(crate) struct $name { $(pub(crate) $field: $kind),+ }

        impl Default for $name {
            fn default() -> Self { Self { $($field: $value),+ } }
        }
    };
}

section!(Index {
    root: String = ".".to_owned(),
    repo_name: String = String::new(),
    ignore_files: Vec<String> = vec![".gitignore".to_owned(), ".deepkitignore".to_owned()],
    languages: String = "auto".to_owned(),
    git_hooks: bool = false,
});

section!(Limits {
    db_max_bytes: String = "2 GiB".to_owned(),
    max_files: u64 = 250_000,
    max_file_bytes: String = "1 MiB".to_owned(),
    max_units_per_file: u64 = 400,
    max_l2_units: u64 = 300_000,
    max_l3_units: u64 = 20_000,
    max_commits: u64 = 2_000,
    max_prs: u64 = 1_000,
    max_pr_commits: u64 = 500,
    max_doc_bytes: String = "50 MiB".to_owned(),
    max_pdf_pages: u64 = 500,
    max_image_bytes: String = "5 MiB".to_owned(),
    max_result_tokens: u64 = 1_500,
    max_open_bytes: String = "256 KiB".to_owned(),
    query_timeout_ms: u64 = 2_000,
    rerank_timeout_ms: u64 = 1_500,
    index_threads: u64 = 0,
    index_nice: bool = true,
    wal_max_bytes: String = "64 MiB".to_owned(),
    models_cache_max_bytes: String = "4 GiB".to_owned(),
});

impl Limits {
    /// Returns the maximum database size as bytes.
    pub(crate) fn db_max_bytes(&self) -> Result<Bytes, Error> {
        parse_bytes(&self.db_max_bytes)
    }
    /// Returns the maximum source-file size as bytes.
    pub(crate) fn max_file_bytes(&self) -> Result<Bytes, Error> {
        parse_bytes(&self.max_file_bytes)
    }
    /// Returns the maximum document extraction input as bytes.
    pub(crate) fn max_doc_bytes(&self) -> Result<Bytes, Error> {
        parse_bytes(&self.max_doc_bytes)
    }
    /// Returns the maximum image extraction input as bytes.
    pub(crate) fn max_image_bytes(&self) -> Result<Bytes, Error> {
        parse_bytes(&self.max_image_bytes)
    }
    /// Returns the maximum `open` response as bytes.
    pub(crate) fn max_open_bytes(&self) -> Result<Bytes, Error> {
        parse_bytes(&self.max_open_bytes)
    }
    /// Returns the WAL size cap as bytes.
    pub(crate) fn wal_max_bytes(&self) -> Result<Bytes, Error> {
        parse_bytes(&self.wal_max_bytes)
    }
    /// Returns the local model cache cap as bytes.
    pub(crate) fn models_cache_max_bytes(&self) -> Result<Bytes, Error> {
        parse_bytes(&self.models_cache_max_bytes)
    }
}

section!(Models {
    unload_after_idle: String = "10m".to_owned()
});
impl Models {
    /// Returns the local-model idle unload timeout.
    pub(crate) fn unload_after_idle(&self) -> Result<Duration, Error> {
        parse_duration(&self.unload_after_idle)
    }
}

section!(Store {
    path: String = ".deepkit/code.db".to_owned(),
    vector_type: String = "float16".to_owned(),
});
section!(Chunk {
    target_tokens: u64 = 500,
    max_tokens: u64 = 1_500,
    min_tokens: u64 = 40,
    markdown_range: [u64; 2] = [300, 800],
});
section!(Tokenize {
    split_subtokens: bool = true,
    split_subtokens_off_for: Vec<String> = vec!["python".to_owned(), "ruby".to_owned(), "elixir".to_owned()],
});

section!(DocsEmbedder {
    profile: String = "remote".to_owned(),
    kinds: Vec<String> = vec!["doc_section".to_owned(), "pr".to_owned(), "module".to_owned(), "history".to_owned(), "summary".to_owned(), "image".to_owned()],
});
section!(RemoteEmbedder {
    url: String = "https://gateway.moonly.ai/v1".to_owned(),
    model: String = "qwen3-embedding-8b".to_owned(),
    api_key_env: String = "MOONLY_GATEWAY_API_KEY".to_owned(),
    encoding_format: String = "base64".to_owned(),
    dimensions: String = "auto".to_owned(),
    query_prefix: String = default_query_prefix(),
    max_batch_tokens: u64 = 32_000,
    max_batch_items: u64 = 64,
    rpm: u64 = 600,
    tpm: u64 = 2_000_000,
    concurrency: u64 = 2,
});
section!(OllamaEmbedder {
    url: String = "http://127.0.0.1:11434".to_owned(),
    model: String = "qwen3-embedding:0.6b".to_owned(),
    query_prefix: String = default_query_prefix(),
    max_batch_items: u64 = 32,
    concurrency: u64 = 1,
});
section!(LocalEmbedder {
    model: String = "qwen3-embedding-0.6b-q8_0".to_owned(),
    query_prefix: String = default_query_prefix(),
    gpu_layers: i64 = -1,
    threads: u64 = 0,
    n_ctx: u64 = 2_048,
    batch_tokens: u64 = 2_048,
});

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub(crate) struct Embedder {
    pub(crate) profile: String,
    pub(crate) dims: u64,
    pub(crate) max_doc_tokens: u64,
    pub(crate) breadcrumb: bool,
    pub(crate) breadcrumb_roles: bool,
    pub(crate) kind_line: bool,
    pub(crate) reembed_on_role_change: bool,
    pub(crate) docs: Option<DocsEmbedder>,
    pub(crate) remote: RemoteEmbedder,
    pub(crate) ollama: OllamaEmbedder,
    pub(crate) local: LocalEmbedder,
}
impl Default for Embedder {
    fn default() -> Self {
        Self {
            profile: "remote".to_owned(),
            dims: 1_024,
            max_doc_tokens: 2_048,
            breadcrumb: true,
            breadcrumb_roles: true,
            kind_line: true,
            reembed_on_role_change: false,
            docs: None,
            remote: RemoteEmbedder::default(),
            ollama: OllamaEmbedder::default(),
            local: LocalEmbedder::default(),
        }
    }
}

section!(RemoteRerank {
    url: String = "https://gateway.moonly.ai/v1".to_owned(),
    model: String = "qwen3-reranker-4b".to_owned(),
    api_key_env: String = "MOONLY_GATEWAY_API_KEY".to_owned(),
});
section!(LocalRerank {
    model: String = "qwen3-reranker-0.6b-q8_0".to_owned(),
    gpu_layers: i64 = -1,
});
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub(crate) struct Rerank {
    pub(crate) mode: String,
    pub(crate) provider: String,
    pub(crate) candidates: u64,
    pub(crate) min_score: f64,
    pub(crate) max_doc_chars: u64,
    pub(crate) instruct: String,
    pub(crate) remote: RemoteRerank,
    pub(crate) local: LocalRerank,
}
impl Default for Rerank {
    fn default() -> Self {
        Self {
            mode: "auto".to_owned(), provider: "auto".to_owned(), candidates: 50,
            min_score: 0.1, max_doc_chars: 12_000,
            instruct: "Given a code search query, retrieve relevant code, docs or changes that answer the query".to_owned(),
            remote: RemoteRerank::default(), local: LocalRerank::default(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub(crate) struct Llm {
    pub(crate) enabled: bool,
    pub(crate) url: String,
    pub(crate) model: String,
    pub(crate) vlm_model: String,
    pub(crate) api_key_env: String,
    pub(crate) extra_body: toml::Value,
    pub(crate) rpm: u64,
    pub(crate) tpm: u64,
    pub(crate) max_calls_per_day: u64,
    pub(crate) max_tokens_per_day: u64,
    pub(crate) concurrency: u64,
}
impl Default for Llm {
    fn default() -> Self {
        Self {
            enabled: false,
            url: "https://gateway.moonly.ai/v1".to_owned(),
            model: "Qwen3.5-35B-A3B".to_owned(),
            vlm_model: "Qwen3.5-35B-A3B".to_owned(),
            api_key_env: "MOONLY_GATEWAY_API_KEY".to_owned(),
            extra_body: default_extra_body(),
            rpm: 120,
            tpm: 400_000,
            max_calls_per_day: 2_000,
            max_tokens_per_day: 2_000_000,
            concurrency: 2,
        }
    }
}

section!(Enrich {
    top_symbols: u64 = 5_000,
    prs_last_n: u64 = 200,
    modules_min_files: u64 = 5,
    images: bool = true,
    scanned_pages: bool = true,
    prompts_dir: String = ".deepkit/prompts".to_owned(),
});
section!(Graph {
    enabled: bool = true,
    damping: f64 = 0.85,
    iterations: u64 = 50,
});
section!(Commits {
    enabled: bool = true,
    max_count: u64 = 2_000,
    prs: bool = true,
    pr_patterns: Vec<pr_patterns::Pattern> = ["github-merge", "github-squash", "gitlab-merge", "plain-merge"].into_iter().map(|s| pr_patterns::Pattern::Builtin(s.to_owned())).collect(),
});
section!(Tests {
    globs: Vec<String> = vec!["tests/**".to_owned(), "test/**".to_owned(), "**/*_test.*".to_owned(), "**/*.spec.*".to_owned(), "**/*.test.*".to_owned(), "**/test_*.py".to_owned(), "**/*Test.java".to_owned(), "**/*Tests.cs".to_owned()],
});
section!(Search {
    fusion: String = "rrf".to_owned(),
    rrf_k: u64 = 60,
    graph_prior_weight: f64 = 0.3,
    default_k: u64 = 10,
    max_k: u64 = 50,
    max_hits_per_file: u64 = 3,
    diversity_margin: f64 = 0.05,
    lazy_resync: bool = true,
    recent_granularity: String = "pr".to_owned(),
});
section!(Serve {
    mcp: bool = true,
    http: String = "127.0.0.1:7433".to_owned(),
    http_token: String = String::new(),
    agent_writable_config: bool = true,
    auto_init: bool = true,
});

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub(crate) struct Treatment {
    pub(crate) name: String,
    #[serde(rename = "match")]
    pub(crate) patterns: Vec<String>,
    pub(crate) chunker: Option<String>,
    pub(crate) max_level: u64,
    pub(crate) enrich: Vec<String>,
    pub(crate) never_send: bool,
}
/// Full §11 configuration. Nested types make unknown TOML entries non-fatal
/// while preserving every documented key for `dump --all`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub(crate) struct Config {
    pub(crate) index: Index,
    pub(crate) limits: Limits,
    pub(crate) models: Models,
    pub(crate) store: Store,
    pub(crate) chunk: Chunk,
    pub(crate) tokenize: Tokenize,
    pub(crate) embedder: Embedder,
    pub(crate) rerank: Rerank,
    pub(crate) llm: Llm,
    pub(crate) enrich: Enrich,
    pub(crate) graph: Graph,
    pub(crate) commits: Commits,
    pub(crate) tests: Tests,
    pub(crate) search: Search,
    pub(crate) serve: Serve,
    pub(crate) queue: Queue,
    pub(crate) treatments: Vec<Treatment>,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            index: Index::default(),
            limits: Limits::default(),
            models: Models::default(),
            store: Store::default(),
            chunk: Chunk::default(),
            tokenize: Tokenize::default(),
            embedder: Embedder::default(),
            rerank: Rerank::default(),
            llm: Llm::default(),
            enrich: Enrich::default(),
            graph: Graph::default(),
            commits: Commits::default(),
            tests: Tests::default(),
            search: Search::default(),
            serve: Serve::default(),
            queue: Queue::default(),
            treatments: default_treatments(),
        }
    }
}
impl Config {
    /// Renders the full effective configuration as TOML.
    pub(crate) fn dump(&self) -> Result<String, Error> {
        toml::to_string_pretty(self).map_err(Error::SerializeToml)
    }

    /// Renders the default TOML with a brief editable-default note for every setting.
    ///
    /// The init command uses this form so a new repository configuration doubles
    /// as discoverable documentation while preserving the exact Config defaults.
    pub(crate) fn dump_commented(&self) -> Result<String, Error> {
        Ok(commented(&self.dump()?))
    }

    /// Retrieves a configuration value by dotted key for `config get`.
    pub(crate) fn get(&self, key: &str) -> Result<Option<toml::Value>, Error> {
        let document: toml::Value = toml::from_str(&self.dump()?).map_err(Error::InlineToml)?;
        Ok(value_at(&document, key).cloned())
    }
    fn validate(&self) -> Result<(), Error> {
        self.queue.execution()?;
        pr_patterns::compile(&self.commits.pr_patterns).map_err(Error::InvalidPrPatterns)?;
        let chunk = &self.chunk;
        if chunk.min_tokens == 0
            || chunk.min_tokens > chunk.target_tokens
            || chunk.target_tokens > chunk.max_tokens
            || chunk.markdown_range[0] == 0
            || chunk.markdown_range[0] > chunk.markdown_range[1]
            || usize::try_from(chunk.max_tokens).is_err()
            || usize::try_from(chunk.markdown_range[1]).is_err()
        {
            return Err(Error::InvalidChunkSettings);
        }
        let _ = self.limits.db_max_bytes()?;
        let _ = self.limits.max_file_bytes()?;
        let _ = self.limits.max_doc_bytes()?;
        let _ = self.limits.max_image_bytes()?;
        let _ = self.limits.max_open_bytes()?;
        let _ = self.limits.wal_max_bytes()?;
        let _ = self.limits.models_cache_max_bytes()?;
        let _ = self.models.unload_after_idle()?;
        Ok(())
    }
}

/// Renders a structurally sanitized configuration value with the same setting notes.
/// Init can omit a private key without altering string contents that resemble TOML.
pub(crate) fn dump_commented_value(value: &toml::Value) -> Result<String, Error> {
    Ok(commented(
        &toml::to_string_pretty(value).map_err(Error::SerializeToml)?,
    ))
}

fn commented(rendered: &str) -> String {
    let mut documented = String::from(
        "# Deepkit Code repository configuration. Each value below is the built-in default.\n",
    );
    let mut inside_multiline_string = false;
    for line in rendered.lines() {
        let trimmed = line.trim();
        if inside_multiline_string {
            documented.push_str(line);
            documented.push(char::from(10));
            if line.matches("\"\"\"").count() % 2 == 1 {
                inside_multiline_string = false;
            }
        } else if trimmed.is_empty() {
            documented.push(char::from(10));
        } else if trimmed.starts_with('[') {
            documented.push_str("# Configuration section.\n");
            documented.push_str(line);
            documented.push(char::from(10));
        } else {
            let key = trimmed.split('=').next().unwrap_or("setting").trim();
            documented.push_str("# ");
            documented.push_str(key);
            documented.push_str(" uses its documented default; edit it to override.\n");
            documented.push_str(line);
            documented.push(char::from(10));
            if line.matches("\"\"\"").count() % 2 == 1 {
                inside_multiline_string = true;
            }
        }
    }
    documented
}

fn default_query_prefix() -> String {
    "Instruct: Given a code search query, retrieve relevant code or docs\nQuery: ".to_owned()
}
fn default_extra_body() -> toml::Value {
    let mut kwargs = toml::map::Map::new();
    kwargs.insert("enable_thinking".to_owned(), toml::Value::Boolean(false));
    let mut body = toml::map::Map::new();
    body.insert(
        "chat_template_kwargs".to_owned(),
        toml::Value::Table(kwargs),
    );
    toml::Value::Table(body)
}
fn default_treatments() -> Vec<Treatment> {
    vec![
        Treatment {
            name: "generated".to_owned(),
            patterns: vec![
                "**/vendor/**".to_owned(),
                "**/node_modules/**".to_owned(),
                "**/*.min.js".to_owned(),
                "**/zz_generated*".to_owned(),
                "**/*.pb.go".to_owned(),
                "**/dist/**".to_owned(),
            ],
            chunker: Some("none".to_owned()),
            max_level: 0,
            enrich: Vec::new(),
            never_send: false,
        },
        Treatment {
            name: "source".to_owned(),
            patterns: vec![
                "src/**".to_owned(),
                "lib/**".to_owned(),
                "app/**".to_owned(),
                "packages/*/src/**".to_owned(),
            ],
            chunker: Some("ast".to_owned()),
            max_level: 3,
            enrich: Vec::new(),
            never_send: false,
        },
        Treatment {
            name: "tests".to_owned(),
            patterns: vec![
                "tests/**".to_owned(),
                "test/**".to_owned(),
                "**/*_test.*".to_owned(),
                "**/*.spec.*".to_owned(),
                "**/*.test.*".to_owned(),
            ],
            chunker: Some("ast".to_owned()),
            max_level: 1,
            enrich: Vec::new(),
            never_send: false,
        },
        Treatment {
            name: "docs".to_owned(),
            patterns: vec![
                "docs/**".to_owned(),
                "**/*.md".to_owned(),
                "**/*.rst".to_owned(),
                "**/*.pdf".to_owned(),
                "**/*.docx".to_owned(),
                "**/*.xlsx".to_owned(),
                "**/*.pptx".to_owned(),
            ],
            chunker: None,
            max_level: 3,
            enrich: vec!["vlm_page".to_owned()],
            never_send: false,
        },
        Treatment {
            name: "images".to_owned(),
            patterns: vec![
                "**/*.png".to_owned(),
                "**/*.jpg".to_owned(),
                "**/*.jpeg".to_owned(),
                "**/*.webp".to_owned(),
                "**/*.gif".to_owned(),
            ],
            chunker: None,
            max_level: 3,
            enrich: vec!["describe_image".to_owned()],
            never_send: false,
        },
    ]
}

/// A configuration plus Figment tags needed to answer `config explain`.
#[derive(Clone, Debug)]
pub(crate) struct LoadedConfig {
    pub(crate) config: Config,
    figment: Figment,
    pub(crate) warnings: Vec<Warning>,
    user_config: Option<PathBuf>,
    repo_config: PathBuf,
    local_config: PathBuf,
}

impl LoadedConfig {
    /// Returns the winning source layer for a dotted configuration key.
    pub(crate) fn explain(&self, key: &str) -> Option<Layer> {
        let metadata = self.figment.find_metadata(key)?;
        if metadata.name.contains("environment variable") {
            return Some(Layer::Environment);
        }
        if metadata.name == "CLI flags" {
            return Some(Layer::Cli);
        }
        if metadata.name == "built-in defaults" {
            return Some(Layer::Defaults);
        }
        let source = metadata.source.as_ref()?.file_path()?;
        if source == self.local_config {
            Some(Layer::LocalFile)
        } else if source == self.repo_config {
            Some(Layer::RepoFile)
        } else if self
            .user_config
            .as_deref()
            .is_some_and(|path| source == path)
        {
            Some(Layer::UserFile)
        } else {
            None
        }
    }
}

/// Documented configuration layers, ordered from lowest to highest priority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Layer {
    Defaults,
    UserFile,
    RepoFile,
    LocalFile,
    Environment,
    Cli,
}

impl Layer {
    /// Gives `config explain` a stable user-facing source description.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Defaults => "built-in defaults",
            Self::UserFile => "user config",
            Self::RepoFile => "repository config",
            Self::LocalFile => "local repository config",
            Self::Environment => "environment",
            Self::Cli => "CLI flags",
        }
    }
}

/// A non-fatal configuration lint finding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Warning {
    pub(crate) path: PathBuf,
    pub(crate) key: String,
}

/// Loads all persisted layers and the process environment in §11 precedence.
pub(crate) fn load(repo_root: &Path, user_config: Option<&Path>) -> Result<LoadedConfig, Error> {
    load_with_cli(repo_root, user_config, None)
}

/// Loads configuration with optional command-line overrides as the final layer.
pub(crate) fn load_with_cli(
    repo_root: &Path,
    user_config: Option<&Path>,
    cli_overrides: Option<toml::Value>,
) -> Result<LoadedConfig, Error> {
    load_layers(
        repo_root,
        user_config,
        cli_overrides,
        LayerInput::File,
        LayerInput::File,
    )
}

/// Resolves already captured repository layers without reopening either file.
/// Explicit absence stays absent; user configuration and environment retain their
/// normal precedence, and snapshot lint/provenance use the original layer paths.
pub(crate) fn load_init_snapshots(
    repo_root: &Path,
    repository: Option<toml::Value>,
    local: Option<toml::Value>,
) -> Result<LoadedConfig, Error> {
    load_layers(
        repo_root,
        None,
        None,
        LayerInput::Snapshot(repository),
        LayerInput::Snapshot(local),
    )
}

enum LayerInput {
    File,
    Snapshot(Option<toml::Value>),
}
impl LayerInput {
    fn lint(&self, path: &Path) -> Result<Vec<Warning>, Error> {
        match self {
            Self::File if path.exists() => lint_file(path),
            Self::Snapshot(Some(value)) => lint_document(value, path),
            _ => Ok(Vec::new()),
        }
    }
    fn merge(self, figment: Figment, path: &Path) -> Figment {
        match self {
            Self::File if path.exists() => figment.merge(Toml::file(path)),
            Self::Snapshot(Some(value)) => figment.merge(NamedSerialized {
                value,
                name: "captured configuration",
                source: Some(path.to_path_buf()),
            }),
            _ => figment,
        }
    }
}

fn load_layers(
    repo_root: &Path,
    user_config: Option<&Path>,
    cli_overrides: Option<toml::Value>,
    repository: LayerInput,
    local: LayerInput,
) -> Result<LoadedConfig, Error> {
    let user_config = user_config
        .map(Path::to_path_buf)
        .or_else(default_user_config_path)
        .filter(|path| path.exists());
    let repo_config = repo_root.join(".deepkit/config.toml");
    let local_config = repo_root.join(".deepkit/config.local.toml");
    let mut warnings = Vec::new();
    if let Some(path) = user_config.as_deref() {
        warnings.extend(lint_file(path)?);
    }
    warnings.extend(repository.lint(&repo_config)?);
    warnings.extend(local.lint(&local_config)?);

    // Zero-setup and later reloads share a local baseline. Config::default remains
    // the canonical expanded Moonly recipe used by explicit init presets (§11).
    // Apply this before every external layer, without consulting database state
    // or reopening captured repository snapshots. Merely creating config.toml
    // must not opt an existing local repository into remote model calls.
    let defaults = local_defaults();
    let mut figment = Figment::from(NamedSerialized::new(defaults, "built-in defaults"));
    if let Some(path) = user_config.as_deref() {
        figment = figment.merge(Toml::file(path));
    }
    figment = repository.merge(figment, &repo_config);
    figment = local.merge(figment, &local_config);
    figment = figment.merge(Env::prefixed("DKC_").split("__"));
    if let Some(overrides) = cli_overrides {
        figment = figment.merge(NamedSerialized::new(overrides, "CLI flags"));
    }
    let config = figment
        .extract::<Config>()
        .map_err(|error| Error::Figment(Box::new(error)))?;
    config.validate()?;
    Ok(LoadedConfig {
        config,
        figment,
        warnings,
        user_config,
        repo_config,
        local_config,
    })
}

fn local_defaults() -> Config {
    let mut defaults = Config::default();
    defaults.embedder.profile = "local".to_owned();
    defaults.llm.enabled = false;
    defaults
}

/// Writes a key to local configuration by default and repository configuration
/// only when explicitly requested by `config set --repo`.
pub(crate) fn set(
    repo_root: &Path,
    key: &str,
    raw_value: &str,
    repo: bool,
) -> Result<PathBuf, Error> {
    let path = repo_root.join(if repo {
        ".deepkit/config.toml"
    } else {
        ".deepkit/config.local.toml"
    });
    let parent = path
        .parent()
        .ok_or_else(|| Error::ConfigPath(path.clone()))?;
    fs::create_dir_all(parent).map_err(|source| Error::Io {
        path: parent.to_path_buf(),
        source,
    })?;
    let mut document = if path.exists() {
        parse_file(&path)?
    } else {
        toml::Value::Table(toml::map::Map::new())
    };
    set_value(&mut document, key, parse_cli_value(raw_value))?;
    let rendered = toml::to_string_pretty(&document).map_err(Error::SerializeToml)?;
    fs::write(&path, rendered).map_err(|source| Error::Io {
        path: path.clone(),
        source,
    })?;
    Ok(path)
}

/// Opens local configuration by default and the repository file with `--repo`.
pub(crate) fn edit(repo_root: &Path, repo: bool) -> Result<(), Error> {
    let path = repo_root.join(if repo {
        ".deepkit/config.toml"
    } else {
        ".deepkit/config.local.toml"
    });
    let parent = path
        .parent()
        .ok_or_else(|| Error::ConfigPath(path.clone()))?;
    fs::create_dir_all(parent).map_err(|source| Error::Io {
        path: parent.to_path_buf(),
        source,
    })?;
    let editor = env::var_os("VISUAL")
        .or_else(|| env::var_os("EDITOR"))
        .ok_or(Error::EditorUnset)?;
    let status = Command::new(editor)
        .arg(&path)
        .status()
        .map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
    if status.success() {
        Ok(())
    } else {
        Err(Error::EditorFailed(status))
    }
}

/// Runs configuration subcommands until the T0.14 CLI owns command dispatch.
pub(crate) fn run_command(arguments: &[OsString]) -> Result<(), Error> {
    let root = env::current_dir().map_err(Error::CurrentDirectory)?;
    let arguments: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    match arguments.as_slice() {
        [command] if command == "dump" => out!("{}", load(&root, None)?.config.dump()?),
        [command, flag] if command == "dump" && flag == "--all" => {
            out!("{}", load(&root, None)?.config.dump()?)
        }
        [command, key] if command == "get" => {
            let value = load(&root, None)?
                .config
                .get(key)?
                .ok_or_else(|| Error::UnknownKey(key.clone()))?;
            outln!("{value}");
        }
        [command, key] if command == "explain" => {
            let layer = load(&root, None)?
                .explain(key)
                .ok_or_else(|| Error::UnknownKey(key.clone()))?;
            outln!("{}", layer.name());
        }
        [command, key, value] if command == "set" => {
            outln!("{}", set(&root, key, value, false)?.display())
        }
        [command, flag, key, value] if command == "set" && flag == "--repo" => {
            outln!("{}", set(&root, key, value, true)?.display())
        }
        [command] if command == "edit" => edit(&root, false)?,
        [command, flag] if command == "edit" && flag == "--repo" => edit(&root, true)?,
        _ => return Err(Error::Usage),
    }
    Ok(())
}

/// Configuration failures reported to the CLI and later to MCP callers.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("could not write stdout: {0}")]
    Stdout(#[from] std::io::Error),
    #[error("invalid commits.pr_patterns: {0}")]
    InvalidPrPatterns(String),
    #[error(
        "queue settings require positive representable attempt/pass bounds and a heartbeat below the 120-second lease"
    )]
    InvalidQueueSettings,
    #[error(
        "queue budgets require unique canonical embedding endpoints, local without a URL or remote/ollama with a credential-free HTTP base URL, and nonnegative integer daily limits"
    )]
    InvalidEndpointBudget,
    #[error(
        "chunk settings require 0 < min_tokens <= target_tokens <= max_tokens and 0 < markdown_range[0] <= markdown_range[1], within platform size limits"
    )]
    InvalidChunkSettings,
    #[error("invalid byte quantity: {0}")]
    InvalidBytes(String),
    #[error("invalid duration: {0}")]
    InvalidDuration(String),
    #[error("invalid TOML configuration in {path}: {source}")]
    ParseToml {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("invalid generated TOML: {0}")]
    InlineToml(#[source] toml::de::Error),
    #[error("could not serialize configuration: {0}")]
    SerializeToml(#[source] toml::ser::Error),
    #[error("could not access {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("configuration path has no parent: {0}")]
    ConfigPath(PathBuf),
    #[error("configuration key is nested under a scalar: {0}")]
    NonTable(String),
    #[error("unknown configuration key: {0}")]
    UnknownKey(String),
    #[error("set VISUAL or EDITOR before using `dkc config edit`")]
    EditorUnset,
    #[error("editor exited unsuccessfully: {0}")]
    EditorFailed(std::process::ExitStatus),
    #[error("could not determine the current repository: {0}")]
    CurrentDirectory(#[source] std::io::Error),
    #[error(
        "usage: dkc config <get KEY|set [--repo] KEY VALUE|explain KEY|edit [--repo]|dump [--all]>"
    )]
    Usage,
    #[error(transparent)]
    Figment(#[from] Box<figment::Error>),
}

#[derive(Clone)]
struct NamedSerialized<T> {
    value: T,
    name: &'static str,
    source: Option<PathBuf>,
}
impl<T> NamedSerialized<T> {
    fn new(value: T, name: &'static str) -> Self {
        Self {
            value,
            name,
            source: None,
        }
    }
}
impl<T: Clone + Serialize> Provider for NamedSerialized<T> {
    fn metadata(&self) -> Metadata {
        let metadata = Metadata::named(self.name);
        match &self.source {
            Some(path) => metadata.source(path.as_path()),
            None => metadata,
        }
    }
    fn data(&self) -> Result<Map<Profile, Dict>, figment::Error> {
        Value::serialize(self.value.clone())?
            .into_dict()
            .map(|dict| Profile::Default.collect(dict))
            .ok_or_else(|| figment::Error::from("configuration provider must produce a table"))
    }
}

fn default_user_config_path() -> Option<PathBuf> {
    env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
        .map(|directory| directory.join("deepkit-code/config.toml"))
}

/// Returns the user-owned credentials file permitted for remote provider secrets.
pub(crate) fn credentials_path() -> Option<PathBuf> {
    default_user_config_path().and_then(|path| {
        path.parent()
            .map(|directory| directory.join("credentials.toml"))
    })
}

fn parse_file(path: &Path) -> Result<toml::Value, Error> {
    let source = fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    toml::from_str(&source).map_err(|source| Error::ParseToml {
        path: path.to_path_buf(),
        source,
    })
}

fn lint_file(path: &Path) -> Result<Vec<Warning>, Error> {
    lint_document(&parse_file(path)?, path)
}

/// Detects structural secret fields without retaining names or values in diagnostics.
/// Shared by init refusal and read-only doctor inspection of repository settings.
pub(crate) fn contains_secret(value: &toml::Value) -> bool {
    match value {
        toml::Value::Table(table) => table.iter().any(|(key, value)| {
            let normalized = key.to_ascii_lowercase().replace('-', "_");
            let forbidden = if normalized == "http_token" {
                value.as_str().is_none_or(|text| !text.is_empty())
            } else {
                agent::secret_key(&normalized) || normalized == "token"
            };
            forbidden || contains_secret(value)
        }),
        toml::Value::Array(values) => values.iter().any(contains_secret),
        _ => false,
    }
}

/// Lints a parsed configuration snapshot without reopening its source file.
pub(crate) fn lint_document(value: &toml::Value, path: &Path) -> Result<Vec<Warning>, Error> {
    let mut template: toml::Value =
        toml::from_str(&Config::default().dump()?).map_err(Error::InlineToml)?;
    let mut docs = toml::map::Map::new();
    docs.insert(
        "profile".to_owned(),
        toml::Value::String("remote".to_owned()),
    );
    docs.insert(
        "kinds".to_owned(),
        toml::Value::Array(
            DocsEmbedder::default()
                .kinds
                .into_iter()
                .map(toml::Value::String)
                .collect(),
        ),
    );
    let docs = toml::Value::Table(docs);
    if let Some(embedder) = template
        .get_mut("embedder")
        .and_then(toml::Value::as_table_mut)
    {
        embedder.insert("docs".to_owned(), docs);
    }
    let mut warnings = Vec::new();
    lint_value(value, &template, "", path, &mut warnings);
    Ok(warnings)
}

fn lint_value(
    value: &toml::Value,
    template: &toml::Value,
    prefix: &str,
    path: &Path,
    warnings: &mut Vec<Warning>,
) {
    if prefix == "llm.extra_body" {
        return;
    }
    match (value, template) {
        (toml::Value::Table(values), toml::Value::Table(known)) => {
            for (key, child) in values {
                let dotted = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                match known.get(key) {
                    Some(template) => lint_value(child, template, &dotted, path, warnings),
                    None => warnings.push(Warning {
                        path: path.to_path_buf(),
                        key: dotted,
                    }),
                }
            }
        }
        (toml::Value::Array(values), toml::Value::Array(known)) => {
            if let Some(template) = known.first() {
                for child in values {
                    lint_value(child, template, prefix, path, warnings);
                }
            }
        }
        _ => {}
    }
}

fn parse_cli_value(raw: &str) -> toml::Value {
    let wrapped = format!("value = {raw}");
    match toml::from_str::<toml::Value>(&wrapped) {
        Ok(toml::Value::Table(mut table)) => table
            .remove("value")
            .unwrap_or_else(|| toml::Value::String(raw.to_owned())),
        Ok(_) | Err(_) => toml::Value::String(raw.to_owned()),
    }
}

fn set_value(document: &mut toml::Value, key: &str, value: toml::Value) -> Result<(), Error> {
    let mut segments = key.split('.').peekable();
    let mut current = document;
    while let Some(segment) = segments.next() {
        let table = current
            .as_table_mut()
            .ok_or_else(|| Error::NonTable(key.to_owned()))?;
        if segments.peek().is_none() {
            table.insert(segment.to_owned(), value);
            return Ok(());
        }
        current = table
            .entry(segment.to_owned())
            .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
    }
    Err(Error::UnknownKey(key.to_owned()))
}

fn value_at<'a>(document: &'a toml::Value, key: &str) -> Option<&'a toml::Value> {
    key.split('.')
        .try_fold(document, |value, segment| value.get(segment))
}

#[cfg(test)]
mod tests {
    use std::{env, fs, path::Path};

    use super::*;

    fn temporary_repository() -> tempfile::TempDir {
        tempfile::tempdir().unwrap_or_else(|error| panic!("temporary repository: {error}"))
    }

    fn write(root: &Path, relative: &str, content: &str) -> PathBuf {
        let path = root.join(relative);
        let parent = path
            .parent()
            .unwrap_or_else(|| panic!("test config has no parent: {}", path.display()));
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| panic!("create {}: {error}", parent.display()));
        fs::write(&path, content)
            .unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
        path
    }

    #[cfg(unix)]
    #[test]
    fn init_snapshots_never_reopen_repository_layers_and_preserve_precedence() {
        const CHILD: &str = "DKC_CONFIG_SNAPSHOT_CHILD";
        if env::var_os(CHILD).is_none() {
            let user = temporary_repository();
            write(
                user.path(),
                "deepkit-code/config.toml",
                "[limits]\nmax_files=10\nmax_l2_units=1000\nmax_commits=333\n[search]\ndefault_k=11\n",
            );
            let mut child = Command::new(env::current_exe().unwrap())
                .args(["--exact", "config::tests::init_snapshots_never_reopen_repository_layers_and_preserve_precedence", "--nocapture"])
                .env_clear().env(CHILD, "1").env("XDG_CONFIG_HOME", user.path())
                .env("DKC_LIMITS__MAX_FILES", "40")
                .stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null())
                .spawn().unwrap();
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            loop {
                if let Some(status) = child.try_wait().unwrap() {
                    assert!(status.success(), "snapshot child failed");
                    return;
                }
                if std::time::Instant::now() >= deadline {
                    child.kill().unwrap();
                    child.wait().unwrap();
                    panic!("snapshot loader blocked while repository FIFO must remain unopened");
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        let root = temporary_repository();
        fs::create_dir(root.path().join(".deepkit")).unwrap();
        let repo_path = root.path().join(".deepkit/config.toml");
        let local_path = root.path().join(".deepkit/config.local.toml");
        crate::sync::source::mkfifo(&repo_path, rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR)
            .unwrap();
        let elsewhere = temporary_repository();
        let malformed = write(elsewhere.path(), "malformed.toml", "invalid TOML [");
        std::os::unix::fs::symlink(&malformed, &local_path).unwrap();
        let repository = toml::from_str("[limits]\nmax_files=20\nmax_l2_units=2000\nsnapshot_unknown=true\n[index]\nrepo_name='captured'\n").unwrap();
        let local = toml::from_str("[limits]\nmax_files=30\n[search]\ndefault_k=12\n").unwrap();
        let loaded = load_init_snapshots(root.path(), Some(repository), Some(local)).unwrap();
        assert_eq!(loaded.config.limits.max_files, 40);
        assert_eq!(loaded.config.limits.max_l2_units, 2000);
        assert_eq!(loaded.config.limits.max_commits, 333);
        assert_eq!(loaded.config.search.default_k, 12);
        assert_eq!(loaded.config.index.repo_name, "captured");
        assert_eq!(loaded.explain("limits.max_files"), Some(Layer::Environment));
        assert_eq!(loaded.explain("limits.max_l2_units"), Some(Layer::RepoFile));
        assert_eq!(loaded.explain("limits.max_commits"), Some(Layer::UserFile));
        assert_eq!(loaded.explain("search.default_k"), Some(Layer::LocalFile));
        assert!(
            loaded.warnings.iter().any(
                |warning| warning.path == repo_path && warning.key == "limits.snapshot_unknown"
            )
        );
        let absent = load_init_snapshots(root.path(), None, None).unwrap();
        assert_eq!(absent.config.limits.max_files, 40);
        assert_eq!(absent.config.limits.max_l2_units, 1000);
        assert_eq!(absent.config.search.default_k, 11);
        assert_eq!(
            absent.config.index.repo_name,
            Config::default().index.repo_name
        );
        assert!(absent.warnings.is_empty());
        assert_eq!(fs::read_to_string(malformed).unwrap(), "invalid TOML [");
    }

    #[test]
    fn structural_comment_rendering_preserves_multiline_values_while_omitting_one_key() {
        let mut config = Config::default();
        let content = "Example configuration:\n[serve]\nhttp_token = 'literal example'\n# http_token is documented here\n[other]\nvalue = 42\n";
        config
            .llm
            .extra_body
            .as_table_mut()
            .unwrap()
            .insert("instruction".into(), toml::Value::String(content.into()));
        let mut value = toml::Value::try_from(&config).unwrap();
        value
            .get_mut("serve")
            .unwrap()
            .as_table_mut()
            .unwrap()
            .remove("http_token");
        let rendered = dump_commented_value(&value).unwrap();
        let parsed: toml::Value = toml::from_str(&rendered).unwrap();
        assert_eq!(parsed, value);
        assert!(parsed["serve"].get("http_token").is_none());
        assert_eq!(
            parsed["llm"]["extra_body"]["instruction"].as_str(),
            Some(content)
        );
        assert!(rendered.contains("# max_files uses its documented default"));
        assert!(rendered.contains("# Configuration section."));
        let ordinary: Config = toml::from_str(&config.dump_commented().unwrap()).unwrap();
        assert_eq!(ordinary, config);
    }

    #[test]
    fn parses_binary_bytes_and_compact_durations() {
        assert_eq!(
            parse_bytes("2 GiB").unwrap_or_else(|error| panic!("bytes: {error}")),
            Bytes(2 * 1024 * 1024 * 1024)
        );
        assert!(parse_bytes("1 MB").is_err());
        assert_eq!(
            parse_duration("10m").unwrap_or_else(|error| panic!("duration: {error}")),
            Duration::from_secs(600)
        );
        assert_eq!(
            parse_duration("250ms").unwrap_or_else(|error| panic!("duration: {error}")),
            Duration::from_millis(250)
        );
        assert!(parse_duration("10d").is_err());
    }

    #[test]
    fn precedence_uses_local_then_cli_over_user_and_repo() {
        let repository = temporary_repository();
        let user = write(repository.path(), "user.toml", "[limits]\nmax_files = 10\n");
        let _ = write(
            repository.path(),
            ".deepkit/config.toml",
            "[limits]\nmax_files = 20\n",
        );
        let _ = write(
            repository.path(),
            ".deepkit/config.local.toml",
            "[limits]\nmax_files = 30\n",
        );
        let cli: toml::Value = toml::from_str("[limits]\nmax_files = 40\n")
            .unwrap_or_else(|error| panic!("CLI overrides: {error}"));
        let loaded = load_with_cli(repository.path(), Some(&user), Some(cli))
            .unwrap_or_else(|error| panic!("layered config: {error}"));
        assert_eq!(loaded.config.limits.max_files, 40);
        assert_eq!(loaded.explain("limits.max_files"), Some(Layer::Cli));
    }

    #[test]
    fn environment_overrides_a_nested_key() {
        if env::var_os("DKC_CONFIG_TEST_CHILD").is_some() {
            let repository = temporary_repository();
            let absent = repository.path().join("absent.toml");
            let loaded = load(repository.path(), Some(&absent))
                .unwrap_or_else(|error| panic!("environment config: {error}"));
            assert_eq!(loaded.config.embedder.profile, "local");
            assert_eq!(loaded.explain("embedder.profile"), Some(Layer::Environment));
            return;
        }

        let executable =
            env::current_exe().unwrap_or_else(|error| panic!("test executable: {error}"));
        let status = Command::new(executable)
            .arg("--exact")
            .arg("config::tests::environment_overrides_a_nested_key")
            .arg("--nocapture")
            .env_clear()
            .env("DKC_CONFIG_TEST_CHILD", "1")
            .env("DKC_EMBEDDER__PROFILE", "local")
            .status()
            .unwrap_or_else(|error| panic!("environment child: {error}"));
        assert!(status.success());
    }

    #[test]
    fn explain_names_local_source_and_unknown_keys_are_warnings() {
        let repository = temporary_repository();
        let _ = write(
            repository.path(),
            ".deepkit/config.local.toml",
            "[limits]\nmax_files = 12\nunknown_limit = true\n",
        );
        let absent = repository.path().join("absent.toml");
        let loaded = load(repository.path(), Some(&absent))
            .unwrap_or_else(|error| panic!("local config: {error}"));
        assert_eq!(loaded.config.limits.max_files, 12);
        assert_eq!(loaded.explain("limits.max_files"), Some(Layer::LocalFile));
        assert_eq!(loaded.warnings.len(), 1);
        assert_eq!(loaded.warnings[0].key, "limits.unknown_limit");
    }

    #[test]
    fn limit_defaults_match_the_full_specification() {
        let limits = Limits::default();
        assert_eq!(limits.db_max_bytes, "2 GiB");
        assert_eq!(limits.max_files, 250_000);
        assert_eq!(limits.max_file_bytes, "1 MiB");
        assert_eq!(limits.max_units_per_file, 400);
        assert_eq!(limits.max_l2_units, 300_000);
        assert_eq!(limits.max_l3_units, 20_000);
        assert_eq!(limits.max_commits, 2_000);
        assert_eq!(limits.max_prs, 1_000);
        assert_eq!(limits.max_pr_commits, 500);
        assert_eq!(limits.max_doc_bytes, "50 MiB");
        assert_eq!(limits.max_pdf_pages, 500);
        assert_eq!(limits.max_image_bytes, "5 MiB");
        assert_eq!(limits.max_result_tokens, 1_500);
        assert_eq!(limits.max_open_bytes, "256 KiB");
        assert_eq!(limits.query_timeout_ms, 2_000);
        assert_eq!(limits.rerank_timeout_ms, 1_500);
        assert_eq!(limits.index_threads, 0);
        assert!(limits.index_nice);
        assert_eq!(limits.wal_max_bytes, "64 MiB");
        assert_eq!(limits.models_cache_max_bytes, "4 GiB");
    }

    #[test]
    fn full_default_dump_round_trips() {
        let dumped = Config::default()
            .dump()
            .unwrap_or_else(|error| panic!("dump defaults: {error}"));
        let parsed: Config =
            toml::from_str(&dumped).unwrap_or_else(|error| panic!("parse defaults: {error}"));
        assert_eq!(parsed, Config::default());
        for section in [
            "[index]",
            "[limits]",
            "[models]",
            "[store]",
            "[chunk]",
            "[tokenize]",
            "[embedder]",
            "[rerank]",
            "[llm]",
            "[enrich]",
            "[graph]",
            "[commits]",
            "[tests]",
            "[search]",
            "[serve]",
            "[[treatments]]",
        ] {
            assert!(dumped.contains(section), "missing {section}");
        }
    }

    #[test]
    fn commented_default_dump_round_trips_with_notes_for_every_setting() {
        let dumped = Config::default()
            .dump_commented()
            .unwrap_or_else(|error| panic!("commented defaults: {error}"));
        let parsed: Config =
            toml::from_str(&dumped).unwrap_or_else(|error| panic!("parse comments: {error}"));
        assert_eq!(parsed, Config::default());
        assert!(dumped.contains("# max_files uses its documented default"));
        assert!(dumped.contains("# Configuration section."));
    }

    #[test]
    fn set_defaults_to_local_and_repo_flag_writes_repo() {
        let repository = temporary_repository();
        let local = set(repository.path(), "embedder.profile", "local", false)
            .unwrap_or_else(|error| panic!("local set: {error}"));
        assert_eq!(local, repository.path().join(".deepkit/config.local.toml"));
        let repo = set(repository.path(), "limits.max_files", "123", true)
            .unwrap_or_else(|error| panic!("repo set: {error}"));
        assert_eq!(repo, repository.path().join(".deepkit/config.toml"));
        let absent = repository.path().join("absent.toml");
        let loaded = load(repository.path(), Some(&absent))
            .unwrap_or_else(|error| panic!("load written config: {error}"));
        assert_eq!(loaded.config.embedder.profile, "local");
        assert_eq!(loaded.config.limits.max_files, 123);
    }
    #[test]
    fn effective_local_defaults_survive_bootstrap_and_unrelated_configuration() {
        const CHILD: &str = "DKC_LOCAL_DEFAULTS_CHILD";
        if env::var_os(CHILD).is_none() {
            let output = Command::new(env::current_exe().unwrap())
                .args(["--exact", "config::tests::effective_local_defaults_survive_bootstrap_and_unrelated_configuration", "--nocapture"])
                .env_clear().env(CHILD, "1").output().unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        let repository = temporary_repository();
        let absent = repository.path().join("absent.toml");
        let assert_local = || {
            let loaded = load(repository.path(), Some(&absent)).unwrap();
            assert_eq!(loaded.config.embedder.profile, "local");
            assert!(!loaded.config.llm.enabled);
            assert_eq!(loaded.explain("embedder.profile"), Some(Layer::Defaults));
            assert_eq!(loaded.explain("llm.enabled"), Some(Layer::Defaults));
        };
        assert_local();
        assert!(!repository.path().join(".deepkit").exists());
        write(
            repository.path(),
            ".deepkit/code.db",
            "database state must not be inspected by config loading",
        );
        assert_local();
        write(
            repository.path(),
            ".deepkit/config.local.toml",
            "[limits]\nmax_files=17\n",
        );
        assert_local();
        write(
            repository.path(),
            ".deepkit/config.toml",
            "[index]\nrepo_name='kept local'\n",
        );
        assert_local();
        let user = write(
            repository.path(),
            "user.toml",
            "[embedder]\nprofile='remote'\n[llm]\nenabled=true\n",
        );
        let loaded = load(repository.path(), Some(&user)).unwrap();
        assert_eq!(loaded.config.embedder.profile, "remote");
        assert!(loaded.config.llm.enabled);
        assert_eq!(loaded.explain("embedder.profile"), Some(Layer::UserFile));
        write(
            repository.path(),
            ".deepkit/config.toml",
            "[embedder]\nprofile='ollama'\n",
        );
        let loaded = load(repository.path(), Some(&user)).unwrap();
        assert_eq!(loaded.config.embedder.profile, "ollama");
        assert_eq!(loaded.explain("embedder.profile"), Some(Layer::RepoFile));
        let loaded = load_with_cli(
            repository.path(),
            Some(&user),
            Some(toml::from_str("[embedder]\nprofile='local'\n[llm]\nenabled=false\n").unwrap()),
        )
        .unwrap();
        assert_eq!(loaded.config.embedder.profile, "local");
        assert!(!loaded.config.llm.enabled);
        assert_eq!(loaded.explain("embedder.profile"), Some(Layer::Cli));
    }
}
