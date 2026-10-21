//! Bounded model discovery before init publishes settings or credentials; v1 §§7.2/10.2.
use crate::config::{self, Config};
use anyhow::{Result, anyhow, bail};
use reqwest::Url;
use std::collections::BTreeSet;

/// One clean provider endpoint and its configured model requirements; never contains a key.
pub(crate) struct Target {
    pub(crate) url: Url,
    pub(crate) api_key_env: String,
    pub(crate) models: BTreeSet<String>,
}

/// Injectable discovery keeps init's state-machine tests independent of live providers.
pub(super) trait Probe {
    /// Checks every selected model without exposing provider bodies or credentials.
    fn verify(&self, target: &Target, key: &str) -> Result<()>;
}

pub(super) struct HttpProbe;

impl Probe for HttpProbe {
    fn verify(&self, target: &Target, key: &str) -> Result<()> {
        // Init is synchronous. Async callers use models::discovery directly;
        // this wrapper requires a synchronous or spawn_blocking caller and
        // does not detach an uncancellable background request.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|_| anyhow!("Could not initialize model discovery runtime"))?;
        runtime
            .block_on(async {
                crate::models::discovery::Discovery::new()?
                    .verify(&target.url, Some(key), &target.models)
                    .await
            })
            .map_err(anyhow::Error::from)
    }
}

/// Groups selected remote roles without probing inactive local or Ollama models.
pub(crate) fn targets(config: &Config) -> Result<Vec<Target>> {
    let mut targets = Vec::new();
    if config.embedder.profile == "remote"
        || config
            .embedder
            .docs
            .as_ref()
            .is_some_and(|docs| docs.profile == "remote")
    {
        let remote = &config.embedder.remote;
        add(
            &mut targets,
            &remote.url,
            &remote.api_key_env,
            &remote.model,
        )?;
    }
    if config.rerank.mode != "off"
        && (config.rerank.provider == "remote"
            || (config.rerank.provider == "auto" && config.embedder.profile == "remote"))
    {
        let remote = &config.rerank.remote;
        add(
            &mut targets,
            &remote.url,
            &remote.api_key_env,
            &remote.model,
        )?;
    }
    if config.llm.enabled {
        let llm = &config.llm;
        for model in [&llm.model, &llm.vlm_model] {
            add(&mut targets, &llm.url, &llm.api_key_env, model)?;
        }
    }
    Ok(targets)
}

fn add(targets: &mut Vec<Target>, base: &str, env: &str, model: &str) -> Result<()> {
    if model.trim().is_empty() || env.is_empty() {
        bail!("Remote model and credential environment name must be configured");
    }
    let url = Url::parse(&config::queue::embedding_endpoint(base, "models")?)
        .map_err(|_| anyhow!("Invalid model discovery endpoint"))?;
    if let Some(target) = targets
        .iter_mut()
        .find(|target| target.url == url && target.api_key_env == env)
    {
        target.models.insert(model.into());
    } else {
        targets.push(Target {
            url,
            api_key_env: env.into(),
            models: BTreeSet::from([model.into()]),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{header, method, path},
    };

    #[tokio::test]
    async fn discovery_verifies_models_user_agent_and_refuses_failed_or_redirected_responses() {
        for (status, models, expected) in [
            (200, vec!["embed", "rank"], true),
            (200, vec!["embed"], false),
            (401, vec![], false),
            (503, vec![], false),
            (302, vec![], false),
        ] {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/v1/models"))
                .and(header("user-agent", concat!("dkc/", env!("CARGO_PKG_VERSION"))))
                .and(header("authorization", "Bearer fixture"))
                .respond_with(ResponseTemplate::new(status).set_body_json(serde_json::json!({
                    "data": models.into_iter().map(|id| serde_json::json!({"id":id})).collect::<Vec<_>>()
                })).insert_header("location", format!("{}/should-not-follow", server.uri())))
                .expect(1)
                .mount(&server).await;
            let target = Target {
                url: Url::parse(&format!("{}/v1/models", server.uri())).unwrap(),
                api_key_env: "UNUSED".into(),
                models: BTreeSet::from(["embed".into(), "rank".into()]),
            };
            let result = tokio::task::spawn_blocking(move || HttpProbe.verify(&target, "fixture"))
                .await
                .unwrap();
            assert_eq!(result.is_ok(), expected);
            assert_eq!(server.received_requests().await.unwrap().len(), 1);
            if status == 401 {
                assert!(result.unwrap_err().to_string().contains("401"));
            }
        }
    }
}
