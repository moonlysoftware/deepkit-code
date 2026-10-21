//! Bounded embedded-worker configuration; v1 §§8.2 and 11, Q-007/PCR-014.
use super::{Error, parse_duration};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Finite embedding execution passes, independent of interactive query deadlines.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub(crate) struct Queue {
    pub(crate) budgets: Vec<EndpointBudget>,
    pub(crate) embedding_timeout: String,
    pub(crate) heartbeat: String,
    pub(crate) max_schedule_per_pass: u64,
    pub(crate) max_jobs_per_pass: u64,
}

impl Default for Queue {
    fn default() -> Self {
        Self {
            budgets: Vec::new(),
            embedding_timeout: "60s".into(),
            heartbeat: "30s".into(),
            max_schedule_per_pass: 1_000,
            max_jobs_per_pass: 1_000,
        }
    }
}

/// Optional daily limits shared by every embedding model/tier at one endpoint.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EndpointBudget {
    pub(crate) profile: String,
    #[serde(default)]
    pub(crate) url: String,
    pub(crate) daily_calls: Option<i64>,
    pub(crate) daily_tokens: Option<i64>,
}

/// Uses the provider adapters' base-path append semantics and URL normalization.
/// Rejects credential-bearing URLs without including their contents in errors.
pub(crate) fn embedding_endpoint(base: &str, suffix: &str) -> Result<String, Error> {
    let url = reqwest::Url::parse(base).map_err(|_| Error::InvalidEndpointBudget)?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(Error::InvalidEndpointBudget);
    }
    reqwest::Url::parse(&format!("{}/{suffix}", base.trim_end_matches('/')))
        .map(|url| url.to_string())
        .map_err(|_| Error::InvalidEndpointBudget)
}

/// Keeps admission identity independent of model, tier and credential references.
pub(crate) fn embedding_provider(profile: &str, endpoint: &str) -> String {
    if profile == "local" {
        "embed:local".into()
    } else {
        format!(
            "embed:{profile}:{}",
            blake3::hash(endpoint.as_bytes()).to_hex()
        )
    }
}

impl EndpointBudget {
    fn provider(&self) -> Result<String, Error> {
        if self.daily_calls.is_some_and(|n| n < 0) || self.daily_tokens.is_some_and(|n| n < 0) {
            return Err(Error::InvalidEndpointBudget);
        }
        let endpoint = match self.profile.as_str() {
            "remote" => embedding_endpoint(&self.url, "embeddings")?,
            "ollama" => embedding_endpoint(&self.url, "api/embed")?,
            "local" if self.url.is_empty() => String::new(),
            _ => return Err(Error::InvalidEndpointBudget),
        };
        Ok(embedding_provider(&self.profile, &endpoint))
    }
}

/// Checked runtime values shared by configuration loading and host preparation.
pub(crate) struct Execution {
    pub(crate) timeout: Duration,
    pub(crate) heartbeat: Duration,
    pub(crate) max_schedule: usize,
    pub(crate) max_jobs: usize,
}

impl Queue {
    /// Validates every entry before selecting a provider; duplicates are errors,
    /// including equivalent URLs, so ordering never silently chooses a ceiling.
    pub(crate) fn budget(&self, provider: &str) -> Result<Option<&EndpointBudget>, Error> {
        let mut seen = std::collections::BTreeSet::new();
        let mut selected = None;
        for budget in &self.budgets {
            let key = budget.provider()?;
            if !seen.insert(key.clone()) {
                return Err(Error::InvalidEndpointBudget);
            }
            if key == provider {
                selected = Some(budget);
            }
        }
        Ok(selected)
    }

    /// Rejects invalid bounds before any worker, model or database preparation.
    pub(crate) fn execution(&self) -> Result<Execution, Error> {
        self.budget("")?;
        let timeout =
            parse_duration(&self.embedding_timeout).map_err(|_| Error::InvalidQueueSettings)?;
        let heartbeat = parse_duration(&self.heartbeat).map_err(|_| Error::InvalidQueueSettings)?;
        let max_schedule =
            usize::try_from(self.max_schedule_per_pass).map_err(|_| Error::InvalidQueueSettings)?;
        let max_jobs =
            usize::try_from(self.max_jobs_per_pass).map_err(|_| Error::InvalidQueueSettings)?;
        if timeout.is_zero()
            || Instant::now().checked_add(timeout).is_none()
            || heartbeat.is_zero()
            || heartbeat >= Duration::from_secs(120)
            || max_schedule == 0
            || max_jobs == 0
        {
            return Err(Error::InvalidQueueSettings);
        }
        Ok(Execution {
            timeout,
            heartbeat,
            max_schedule,
            max_jobs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{self, Layer, agent};
    use std::{error::Error as StdError, fs};

    #[test]
    fn endpoint_budgets_share_canonical_identity_and_layer_as_one_list()
    -> Result<(), Box<dyn StdError>> {
        use crate::{queue::destination, store::UnitKind};
        let root = tempfile::tempdir()?;
        fs::create_dir(root.path().join(".deepkit"))?;
        let user = root.path().join("user.toml");
        fs::write(&user, "[[queue.budgets]]\nprofile='local'\ndaily_calls=0\n")?;
        let repo = root.path().join(".deepkit/config.toml");
        fs::write(
            &repo,
            "[embedder]\nprofile='remote'\n[embedder.remote]\nurl='https://example.org/v1/'\n[embedder.docs]\nprofile='remote'\nkinds=['doc_section']\n[[queue.budgets]]\nprofile='remote'\nurl='HTTPS://EXAMPLE.ORG:443/v1///'\ndaily_calls=0\ndaily_tokens=123\n",
        )?;
        let loaded = config::load(root.path(), Some(&user))?;
        assert!(loaded.warnings.is_empty());
        assert_eq!(loaded.explain("queue.budgets"), Some(Layer::RepoFile));
        assert_eq!(loaded.config.queue.budgets.len(), 1);
        let code = destination::resolve(&loaded.config, UnitKind::Chunk)?;
        let docs = destination::resolve(&loaded.config, UnitKind::DocSection)?;
        assert_eq!(code.identity.provider, docs.identity.provider);
        let selected = loaded
            .config
            .queue
            .budget(&code.identity.provider)?
            .ok_or("budget")?;
        assert_eq!(
            (selected.daily_calls, selected.daily_tokens),
            (Some(0), Some(123))
        );
        let mut changed = loaded.config.clone();
        changed.embedder.remote.model = "another-model".into();
        changed.embedder.remote.api_key_env = "ANOTHER_KEY_REFERENCE".into();
        assert_eq!(
            destination::resolve(&changed, UnitKind::Chunk)?
                .identity
                .provider,
            code.identity.provider
        );
        changed.embedder.remote.url = "https://example.org/other".into();
        assert!(
            changed
                .queue
                .budget(
                    &destination::resolve(&changed, UnitKind::Chunk)?
                        .identity
                        .provider
                )?
                .is_none()
        );
        assert!(changed.queue.budget("embed:local")?.is_none());
        let round_trip: config::Config = toml::from_str(&loaded.config.dump_commented()?)?;
        assert_eq!(round_trip, loaded.config);
        fs::write(
            root.path().join(".deepkit/config.local.toml"),
            "[queue]\nbudgets=[]\n",
        )?;
        assert!(
            config::load(root.path(), Some(&user))?
                .config
                .queue
                .budgets
                .is_empty()
        );
        Ok(())
    }

    #[test]
    fn endpoint_budgets_refuse_invalid_or_duplicate_entries_atomically()
    -> Result<(), Box<dyn StdError>> {
        let root = tempfile::tempdir()?;
        fs::create_dir(root.path().join(".deepkit"))?;
        let path = root.path().join(".deepkit/config.local.toml");
        let absent = root.path().join("absent.toml");
        for entries in [
            serde_json::json!([{"profile":"remote","url":"https://example.org","daily_calls":-1}]),
            serde_json::json!([{"profile":"local","daily_tokens":-1}]),
            serde_json::json!([{"profile":"local","daily_calls":1.5}]),
            serde_json::json!([{"profile":"unknown"}]),
            serde_json::json!([{"profile":"remote"}]),
            serde_json::json!([{"profile":"local","url":"https://example.org"}]),
            serde_json::json!([{"profile":"remote","url":"file:///tmp/provider"}]),
            serde_json::json!([{"profile":"remote","url":"https://example.org?query=value"}]),
            serde_json::json!([{"profile":"remote","url":"https://example.org#fragment"}]),
            serde_json::json!([{"profile":"local","daily_call":0}]),
            serde_json::json!([{"profile":"local"},{"profile":"local","daily_calls":0}]),
            serde_json::json!([{"profile":"remote","url":"https://example.org/v1","daily_calls":1},{"profile":"remote","url":"https://EXAMPLE.org:443/v1/","daily_calls":2}]),
        ] {
            let value: toml::Value =
                serde_json::from_value(serde_json::json!({"queue":{"budgets":entries}}))?;
            fs::write(&path, toml::to_string(&value)?)?;
            assert!(
                config::load(root.path(), Some(&absent)).is_err(),
                "{entries}"
            );
            fs::write(&path, "# retained\n")?;
            assert!(matches!(
                agent::set(
                    root.path(),
                    "queue.budgets",
                    entries,
                    agent::WriteLayer::Local
                ),
                Err(agent::Error::InvalidValue)
            ));
            assert_eq!(fs::read_to_string(&path)?, "# retained\n");
        }
        agent::set(
            root.path(),
            "queue.budgets",
            serde_json::json!([{"profile":"ollama","url":"http://localhost:11434/","daily_tokens":0}]),
            agent::WriteLayer::Local,
        )?;
        assert_eq!(
            config::load(root.path(), Some(&absent))?
                .config
                .queue
                .budgets[0]
                .daily_tokens,
            Some(0)
        );
        Ok(())
    }

    #[test]
    fn queue_layers_round_trip_and_explain_without_query_deadline_coupling()
    -> Result<(), Box<dyn StdError>> {
        let root = tempfile::tempdir()?;
        fs::create_dir(root.path().join(".deepkit"))?;
        let user = root.path().join("user.toml");
        fs::write(
            &user,
            "[queue]\nembedding_timeout='20s'\nmax_jobs_per_pass=7\n",
        )?;
        fs::write(
            root.path().join(".deepkit/config.toml"),
            "[queue]\nembedding_timeout='25s'\n[limits]\nquery_timeout_ms=0\n",
        )?;
        fs::write(
            root.path().join(".deepkit/config.local.toml"),
            "[queue]\nmax_schedule_per_pass=3\nheartbeat='250ms'\n",
        )?;
        let loaded = config::load(root.path(), Some(&user))?;
        assert!(loaded.warnings.is_empty());
        let execution = loaded.config.queue.execution()?;
        assert_eq!(execution.timeout, Duration::from_secs(25));
        assert_eq!(execution.heartbeat, Duration::from_millis(250));
        assert_eq!((execution.max_schedule, execution.max_jobs), (3, 7));
        assert_eq!(
            loaded.explain("queue.embedding_timeout"),
            Some(Layer::RepoFile)
        );
        assert_eq!(
            loaded.explain("queue.max_jobs_per_pass"),
            Some(Layer::UserFile)
        );
        assert_eq!(loaded.explain("queue.heartbeat"), Some(Layer::LocalFile));
        let round_trip: config::Config = toml::from_str(&loaded.config.dump_commented()?)?;
        assert_eq!(round_trip, loaded.config);
        let defaults = Queue::default().execution()?;
        assert_eq!(defaults.timeout, Duration::from_secs(60));
        assert_eq!(defaults.heartbeat, Duration::from_secs(30));
        assert_eq!((defaults.max_schedule, defaults.max_jobs), (1_000, 1_000));
        Ok(())
    }

    #[test]
    fn queue_invalid_bounds_fail_load_and_atomic_agent_write() -> Result<(), Box<dyn StdError>> {
        let root = tempfile::tempdir()?;
        fs::create_dir(root.path().join(".deepkit"))?;
        let path = root.path().join(".deepkit/config.local.toml");
        let absent = root.path().join("absent.toml");
        for (key, value) in [
            ("embedding_timeout", serde_json::json!("0s")),
            (
                "embedding_timeout",
                serde_json::json!("18446744073709551615s"),
            ),
            ("embedding_timeout", serde_json::json!("invalid")),
            ("heartbeat", serde_json::json!("0ms")),
            ("heartbeat", serde_json::json!("120s")),
            ("max_schedule_per_pass", serde_json::json!(0)),
            ("max_jobs_per_pass", serde_json::json!(0)),
            ("max_jobs_per_pass", serde_json::json!(-1)),
        ] {
            let document = format!("[queue]\n{key}={}\n", value);
            fs::write(&path, document)?;
            assert!(config::load(root.path(), Some(&absent)).is_err(), "{key}");
            fs::write(&path, "# preserve this file\n")?;
            assert!(
                matches!(
                    agent::set(
                        root.path(),
                        &format!("queue.{key}"),
                        value,
                        agent::WriteLayer::Local
                    ),
                    Err(agent::Error::InvalidValue)
                ),
                "{key}"
            );
            assert_eq!(fs::read_to_string(&path)?, "# preserve this file\n");
        }
        let response = agent::set(
            root.path(),
            "queue.max_jobs_per_pass",
            serde_json::json!(2),
            agent::WriteLayer::Local,
        )?;
        assert_eq!(response.value, serde_json::json!(2));
        assert_eq!(response.source, "local repository config");
        Ok(())
    }
}
