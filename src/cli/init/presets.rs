//! Pure init profile construction without provider or credential activity; v1 §§7.2, 10.2, 11.
use crate::config::{Config, DocsEmbedder};
use anyhow::{Context, Result, bail};

/// Explicit custom endpoint recipes; this structure never contains credentials.
pub(crate) struct Remote {
    /// Credential-free HTTP(S) base URL shared by selected remote roles.
    pub(crate) url: String,
    /// Required embedding model identifier to verify through model discovery.
    pub(crate) embedding_model: String,
    /// Optional reranker identifier; absence disables reranking.
    pub(crate) rerank_model: Option<String>,
    /// Optional chat model identifier; absence disables enrichment.
    pub(crate) llm_model: Option<String>,
    /// Explicit multimodal identifier required whenever enrichment is enabled.
    pub(crate) vlm_model: Option<String>,
}

/// Applies only selected provider recipes, preserving unrelated repository settings.
/// Invalid input leaves configuration unchanged. Callers own prompts, model
/// discovery, credential storage and atomic publication of the returned settings.
pub(crate) fn apply(
    config: &mut Config,
    profile: &str,
    docs_moonly: bool,
    custom: Option<&Remote>,
) -> Result<()> {
    if docs_moonly && profile != "local" {
        bail!("--docs moonly requires --profile local");
    }
    if custom.is_some() && profile != "remote" {
        bail!("custom remote settings require --profile remote");
    }
    let mut proposed = config.clone();
    proposed.embedder.docs = None;
    proposed.rerank.mode = "auto".into();
    proposed.rerank.provider = "auto".into();
    match profile {
        "local" | "ollama" => {
            proposed.embedder.profile = profile.into();
            proposed.llm.enabled = false;
            if docs_moonly {
                moonly(&mut proposed)?;
                proposed.embedder.profile = "local".into();
                proposed.embedder.docs = Some(DocsEmbedder::default());
            }
        }
        "moonly" => moonly(&mut proposed)?,
        "remote" => remote(
            &mut proposed,
            custom.context("custom remote endpoint and embedding model are required")?,
        )?,
        _ => bail!("profile must be local, moonly, ollama or remote"),
    }
    *config = proposed;
    Ok(())
}
fn moonly(config: &mut Config) -> Result<()> {
    let defaults = Config::default();
    config.embedder.profile = "remote".into();
    config.embedder.dims = defaults.embedder.dims;
    let embedder = &mut config.embedder.remote;
    embedder.url = defaults.embedder.remote.url;
    embedder.model = defaults.embedder.remote.model;
    embedder.api_key_env = defaults.embedder.remote.api_key_env;
    embedder.encoding_format = defaults.embedder.remote.encoding_format;
    embedder.dimensions = defaults.embedder.remote.dimensions;
    embedder.query_prefix = defaults.embedder.remote.query_prefix;
    config.rerank.remote = defaults.rerank.remote;
    config.llm.enabled = true;
    config.llm.url = defaults.llm.url;
    config.llm.model = defaults.llm.model;
    config.llm.vlm_model = defaults.llm.vlm_model;
    config.llm.api_key_env = defaults.llm.api_key_env;
    let body = config
        .llm
        .extra_body
        .as_table_mut()
        .context("llm.extra_body must be a table")?;
    let kwargs = body
        .entry("chat_template_kwargs")
        .or_insert_with(|| toml::Value::Table(Default::default()));
    kwargs
        .as_table_mut()
        .context("llm.extra_body.chat_template_kwargs must be a table")?
        .insert("enable_thinking".into(), toml::Value::Boolean(false));
    Ok(())
}
fn model(name: &str, role: &str) -> Result<()> {
    if name.trim().is_empty() || name.chars().any(char::is_control) {
        bail!("a nonempty {role} model name is required");
    }
    Ok(())
}
fn endpoint(value: &str) -> Result<String> {
    // Fixed errors never echo an unvalidated URL which might contain credentials.
    if value != value.trim() || value.chars().any(char::is_control) || value.contains('\\') {
        bail!("remote URL must be a credential-free HTTP(S) base URL");
    }
    let url = reqwest::Url::parse(value)
        .map_err(|_| anyhow::anyhow!("remote URL must be a credential-free HTTP(S) base URL"))?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        bail!("remote URL must be a credential-free HTTP(S) base URL");
    }
    Ok(url.as_str().trim_end_matches('/').into())
}
fn remote(config: &mut Config, custom: &Remote) -> Result<()> {
    let url = endpoint(&custom.url)?;
    model(&custom.embedding_model, "embedding")?;
    if let Some(name) = &custom.rerank_model {
        model(name, "rerank")?;
    }
    if let Some(name) = &custom.llm_model {
        model(name, "LLM")?;
        model(
            custom
                .vlm_model
                .as_deref()
                .context("an explicit VLM model is required when LLM enrichment is enabled")?,
            "VLM",
        )?;
    } else if custom.vlm_model.is_some() {
        bail!(
            "a VLM model requires an LLM model because enrichment is enabled as one configuration"
        );
    }
    config.embedder.profile = "remote".into();
    config.embedder.remote.url = url.clone();
    config.embedder.remote.model = custom.embedding_model.clone();
    config.embedder.remote.api_key_env = "DKC_REMOTE_API_KEY".into();
    config.rerank.provider = "remote".into();
    config.rerank.mode = if custom.rerank_model.is_some() {
        "auto"
    } else {
        "off"
    }
    .into();
    config.rerank.remote.url = url.clone();
    config.rerank.remote.api_key_env = "DKC_REMOTE_API_KEY".into();
    config.rerank.remote.model = custom.rerank_model.clone().unwrap_or_default();
    config.llm.enabled = custom.llm_model.is_some();
    config.llm.url = url;
    config.llm.api_key_env = "DKC_REMOTE_API_KEY".into();
    config.llm.model = custom.llm_model.clone().unwrap_or_default();
    config.llm.vlm_model = custom.vlm_model.clone().unwrap_or_default();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn custom() -> Remote {
        Remote {
            url: "http://127.0.0.1:1234/v1/".into(),
            embedding_model: "custom-embedding".into(),
            rerank_model: Some("custom-rerank".into()),
            llm_model: Some("custom-chat".into()),
            vlm_model: Some("custom-vision".into()),
        }
    }
    #[test]
    fn moonly_restores_required_recipes_and_preserves_unrelated_options() {
        let mut config = Config::default();
        config.index.repo_name = "kept".into();
        config.search.max_k = 17;
        config.limits.query_timeout_ms = 987;
        config.embedder.remote.rpm = 7;
        config.rerank.candidates = 23;
        config.llm.max_calls_per_day = 9;
        config.llm.extra_body = toml::from_str(
            "temperature=0.2\n[chat_template_kwargs]\nenable_thinking=true\ncustom=true\n",
        )
        .unwrap();
        config.embedder.remote.url = "https://old.example".into();
        config.embedder.remote.encoding_format = "float".into();
        apply(&mut config, "moonly", false, None).unwrap();
        let defaults = Config::default();
        assert_eq!(config.embedder.profile, "remote");
        assert_eq!(config.embedder.remote.url, defaults.embedder.remote.url);
        assert_eq!(config.embedder.remote.model, defaults.embedder.remote.model);
        assert_eq!(config.embedder.remote.api_key_env, "MOONLY_GATEWAY_API_KEY");
        assert_eq!(config.embedder.remote.encoding_format, "base64");
        assert_eq!(config.embedder.remote.dimensions, "auto");
        assert_eq!(config.rerank.remote, defaults.rerank.remote);
        assert!(config.llm.enabled);
        assert_eq!(config.llm.model, defaults.llm.model);
        assert_eq!(config.llm.vlm_model, defaults.llm.vlm_model);
        assert_eq!(
            config.llm.extra_body["chat_template_kwargs"]["enable_thinking"].as_bool(),
            Some(false)
        );
        assert_eq!(
            config.llm.extra_body["chat_template_kwargs"]["custom"].as_bool(),
            Some(true)
        );
        assert_eq!(config.llm.extra_body["temperature"].as_float(), Some(0.2));
        assert_eq!(config.index.repo_name, "kept");
        assert_eq!(config.search.max_k, 17);
        assert_eq!(config.limits.query_timeout_ms, 987);
        assert_eq!(config.embedder.remote.rpm, 7);
        assert_eq!(config.rerank.candidates, 23);
        assert_eq!(config.llm.max_calls_per_day, 9);
    }
    #[test]
    fn offline_ollama_and_local_with_gateway_docs_select_only_requested_tiers() {
        for profile in ["local", "ollama"] {
            let mut config = Config::default();
            config.llm.enabled = true;
            config.embedder.docs = Some(DocsEmbedder::default());
            config.embedder.ollama.url = "http://localhost:4321".into();
            config.embedder.ollama.model = "chosen-ollama".into();
            let ollama = config.embedder.ollama.clone();
            apply(&mut config, profile, false, None).unwrap();
            assert_eq!(config.embedder.profile, profile);
            assert!(!config.llm.enabled);
            assert!(config.embedder.docs.is_none());
            assert_eq!(config.embedder.ollama, ollama);
            assert_eq!(config.rerank.provider, "auto");
        }
        let mut config = Config::default();
        config.embedder.local.gpu_layers = 0;
        apply(&mut config, "local", true, None).unwrap();
        assert_eq!(config.embedder.profile, "local");
        assert_eq!(config.embedder.local.gpu_layers, 0);
        assert_eq!(config.embedder.docs, Some(DocsEmbedder::default()));
        assert!(config.llm.enabled);
        assert_eq!(
            config.embedder.remote.url,
            Config::default().embedder.remote.url
        );
        assert_eq!(config.rerank.provider, "auto");
    }
    #[test]
    fn custom_roles_share_clean_endpoint_and_absent_roles_are_disabled() {
        let mut config = Config::default();
        let mut chosen = custom();
        apply(&mut config, "remote", false, Some(&chosen)).unwrap();
        assert_eq!(config.embedder.remote.url, "http://127.0.0.1:1234/v1");
        assert_eq!(config.embedder.remote.url, config.rerank.remote.url);
        assert_eq!(config.llm.url, config.embedder.remote.url);
        assert_eq!(config.embedder.remote.api_key_env, "DKC_REMOTE_API_KEY");
        assert_eq!(config.rerank.remote.api_key_env, "DKC_REMOTE_API_KEY");
        assert_eq!(config.llm.api_key_env, "DKC_REMOTE_API_KEY");
        assert_eq!(config.llm.vlm_model, "custom-vision");
        assert!(config.llm.enabled);
        chosen.rerank_model = None;
        chosen.llm_model = None;
        chosen.vlm_model = None;
        apply(&mut config, "remote", false, Some(&chosen)).unwrap();
        assert_eq!(config.rerank.mode, "off");
        assert!(!config.llm.enabled);
        assert!(config.rerank.remote.model.is_empty());
        assert!(config.llm.model.is_empty());
    }
    #[test]
    fn invalid_input_never_mutates_configuration_or_echoes_urls() {
        let original = Config::default();
        for url in [
            "file:///tmp/model",
            "https://user:password@example.org/v1",
            "https://example.org/v1?q=private",
            "https://example.org/#fragment",
            " https://example.org",
            "https://example.org\\bad",
        ] {
            let mut chosen = custom();
            chosen.url = url.into();
            let mut config = original.clone();
            let error = apply(&mut config, "remote", false, Some(&chosen)).unwrap_err();
            assert_eq!(config, original);
            assert!(!error.to_string().contains(url));
        }
        for missing in ["embedding", "rerank", "llm", "vlm", "absent_vlm"] {
            let mut chosen = custom();
            match missing {
                "embedding" => chosen.embedding_model.clear(),
                "rerank" => chosen.rerank_model = Some(" ".into()),
                "llm" => chosen.llm_model = Some("".into()),
                "vlm" => chosen.vlm_model = Some("".into()),
                _ => chosen.vlm_model = None,
            }
            let mut config = original.clone();
            assert!(apply(&mut config, "remote", false, Some(&chosen)).is_err());
            assert_eq!(config, original);
        }
        let mut config = original.clone();
        assert!(apply(&mut config, "ollama", true, None).is_err());
        assert!(apply(&mut config, "remote", false, None).is_err());
        assert_eq!(config, original);
    }
}
