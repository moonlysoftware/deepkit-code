//! MCP schemas and adapters to shared repository operations; v1 §10.1.
use crate::{
    config,
    search::{fusion::Mode, pipeline, service},
    sync,
};
use rmcp::{
    ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, MetaObject, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Instant,
};

#[derive(Clone)]
pub(super) struct Server {
    root: PathBuf,
    pub(super) worker_status: Arc<super::worker::Status>,
    pub(super) enrichment_status: Arc<super::worker::Status>,
    pub(super) l1_status: Arc<super::progress::Status>,
    context: Arc<Mutex<Option<Arc<service::Context>>>>,
    latencies: Arc<crate::search::latency::History>,
    tool_router: ToolRouter<Self>,
}
struct Failure {
    code: &'static str,
    message: String,
    hint: &'static str,
}
impl From<service::Error> for Failure {
    fn from(error: service::Error) -> Self {
        Self {
            code: error.code(),
            message: error.message().into(),
            hint: error.hint(),
        }
    }
}
fn invalid(message: &str) -> Failure {
    Failure {
        code: "invalid_arguments",
        message: message.into(),
        hint: "Check the tool's input schema.",
    }
}
fn task_error() -> Failure {
    Failure {
        code: "tool_failed",
        message: "Repository operation failed".into(),
        hint: "Run dkc status and retry after resolving index errors.",
    }
}
fn target(name: Option<String>, id: Option<String>) -> Result<String, Failure> {
    match (name, id) {
        (Some(value), None) | (None, Some(value)) if !value.is_empty() => Ok(value),
        _ => Err(invalid("Supply exactly one name/path or id")),
    }
}
fn result(
    value: Result<Value, Failure>,
    start: Instant,
    ttl: Option<u64>,
    progress: Option<super::progress::Snapshot>,
) -> CallToolResult {
    let took_ms = start.elapsed().as_secs_f64() * 1000.0;
    let (mut value, error) = match value {
        Ok(value) => (value, false),
        Err(error) => (
            json!({"error":{"code":error.code,"message":error.message,"hint":error.hint}}),
            true,
        ),
    };
    if let Some(object) = value.as_object_mut() {
        object.insert("took_ms".into(), json!(took_ms));
    }
    let index = progress.map(|snapshot| annotate_progress(&mut value, snapshot));
    // Construct once after annotation so text content and structured content agree.
    let mut output = if error {
        CallToolResult::structured_error(value)
    } else {
        CallToolResult::structured(value)
    };
    let mut meta = MetaObject::new();
    meta.0.insert("took_ms".into(), json!(took_ms));
    if let Some(ttl) = ttl {
        meta.0.insert("ttlMs".into(), json!(ttl));
    }
    if let Some(index) = index {
        meta.0.insert("index".into(), index);
    }
    output.meta = Some(meta);
    output
}

// Preserve array results (notably symbol) and existing index details. Counts are
// copied only from this result; progress annotation never opens or scans SQLite.
fn annotate_progress(value: &mut Value, snapshot: super::progress::Snapshot) -> Value {
    let pending = value
        .get("index")
        .and_then(|index| index.get("l2_pending"))
        .filter(|value| value.as_u64().is_some())
        .or_else(|| {
            value
                .get("l2_pending")
                .filter(|value| value.as_u64().is_some())
        })
        .cloned();
    let mut index = value
        .get("index")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    index.insert("state".into(), json!(snapshot.state));
    index.insert("percentage".into(), json!(snapshot.percentage));
    index.insert("stage".into(), json!(snapshot.stage));
    index.insert("processed_files".into(), json!(snapshot.processed));
    index.insert("total_files".into(), json!(snapshot.total));
    index.insert("l2_pending".into(), pending.clone().unwrap_or(Value::Null));
    index.insert(
        "l2_pending_state".into(),
        json!(if pending.is_some() {
            "known"
        } else {
            "not_checked"
        }),
    );
    if snapshot.state != super::progress::State::Complete {
        index.insert("partial".into(), json!(true));
    }
    let index = Value::Object(index);
    if let Some(object) = value.as_object_mut() {
        object.insert("index".into(), index.clone());
    }
    index
}

impl Server {
    /// Shares repository connections and bounded latency history across HTTP sessions.
    pub(super) fn new(root: PathBuf) -> Self {
        Self {
            root,
            worker_status: Arc::new(super::worker::Status::default()),
            enrichment_status: Arc::new(super::worker::Status::default()),
            l1_status: Arc::new(super::progress::Status::default()),
            context: Arc::new(Mutex::new(None)),
            latencies: Arc::new(crate::search::latency::History::default()),
            tool_router: Self::tool_router(),
        }
    }
    /// Installs the already-open bootstrap context before accepting tool calls.
    /// Startup shares L1 activity; effective-config reloads retain only latency.
    pub(super) fn install_context(
        &self,
        mut context: service::Context,
    ) -> Result<Arc<service::Context>, super::Error> {
        context.query_latencies = Arc::clone(&self.latencies);
        context.background_l1 = self.l1_status.active();
        let context = Arc::new(context);
        let mut slot = self.context.lock().map_err(|_| super::Error::Worker)?;
        *slot = Some(Arc::clone(&context));
        Ok(context)
    }
    /// Reuses the tool connection owner for the embedded queue.
    pub(super) fn worker_context(&self) -> Result<Arc<service::Context>, super::Error> {
        self.context().map_err(|_| super::Error::Worker)
    }
    /// Reads effective settings even when a bad resource root prevents opening the index.
    pub(super) fn worker_config(&self) -> Result<config::Config, super::Error> {
        config::load(&self.root, None)
            .map(|loaded| loaded.config)
            .map_err(|_| super::Error::Worker)
    }
    fn context(&self) -> Result<Arc<service::Context>, Failure> {
        let loaded = config::load(&self.root, None).map_err(|_| {
            self.l1_status.suppress();
            task_error()
        })?;
        let mut slot = self.context.lock().map_err(|_| task_error())?;
        if let Some(context) = slot
            .as_ref()
            .filter(|context| context.config == loaded.config && context.config_root == self.root)
        {
            return Ok(Arc::clone(context));
        }
        // Even a failed reload invalidates the original attempt's public claim.
        // Keep its activity Arc alive only on the captured worker context.
        if slot.is_some() {
            self.l1_status.suppress();
        }
        if !super::existing_index(&self.root, &loaded.config) {
            return Err(Failure {
                code: "index_unavailable",
                message: "An existing index is required".into(),
                hint: "Run dkc index --level 1 or restore the prior configuration.",
            });
        }
        let mut context = service::Context::load_with_config(&self.root, loaded.config)?;
        context.query_latencies = Arc::clone(&self.latencies);
        if slot.is_none() {
            context.background_l1 = self.l1_status.active();
        }
        let context = Arc::new(context);
        *slot = Some(Arc::clone(&context));
        Ok(context)
    }
    async fn run_config(
        &self,
        operation: impl FnOnce(&std::path::Path) -> Result<Value, Failure> + Send + 'static,
    ) -> CallToolResult {
        let start = Instant::now();
        let root = self.root.clone();
        let server = self.clone();
        let value = tokio::task::spawn_blocking(move || {
            // Configuration repair must not require the configured index to be accessible.
            sync::head::read(&root, sync::l1::Work::default()).map_err(|_| task_error())?;
            let value = operation(&root);
            let unchanged = config::load(&root, None).ok().is_some_and(|loaded| {
                server.context.lock().ok().is_some_and(|slot| {
                    slot.as_ref()
                        .is_some_and(|context| context.config == loaded.config)
                })
            });
            if !unchanged {
                server.l1_status.suppress();
            }
            value
        })
        .await
        .unwrap_or_else(|_| Err(task_error()));
        result(value, start, None, self.l1_status.snapshot())
    }
    async fn run(
        &self,
        operation: impl FnOnce(&service::Context) -> Result<Value, Failure> + Send + 'static,
        ttl: Option<u64>,
    ) -> CallToolResult {
        let start = Instant::now();
        // Counts may precede bootstrap completion. Keep the progress observed
        // before reading them, rather than attaching a later complete claim.
        let progress = self.l1_status.snapshot();
        let server = self.clone();
        let value = tokio::task::spawn_blocking(move || {
            let context = server.context()?;
            // Status and recent need HEAD observation, not a full Git-index parse.
            sync::head::read(&context.root, sync::l1::Work::default()).map_err(|_| task_error())?;
            operation(&context)
        })
        .await
        .unwrap_or_else(|_| Err(task_error()));
        // A context reload can invalidate that earlier attempt during the call.
        let progress = progress.filter(|_| self.l1_status.snapshot().is_some());
        result(value, start, ttl, progress)
    }
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SearchArgs {
    query: String,
    scope: Option<crate::store::SearchScope>,
    #[schemars(range(min = 1, max = 50))]
    k: Option<usize>,
    #[schemars(extend("enum" = ["lexical", "dense", "hybrid", null]))]
    mode: Option<String>,
    rerank: Option<bool>,
    exclude_ids: Option<Vec<String>>,
    verbose: Option<bool>,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct OpenArgs {
    id: Option<String>,
    path: Option<String>,
    #[schemars(schema_with = "line_range_schema")]
    lines: Option<[u64; 2]>,
    context: Option<u64>,
}
// JSON Schema cannot compare endpoints. The adapter also checks the path/id
// relationship at runtime, preserving structured application errors.
fn line_range_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
    schemars::json_schema!({
        "type": ["array", "null"],
        "minItems": 2,
        "maxItems": 2,
        "items": {"type": "integer", "minimum": 1, "maximum": i64::MAX}
    })
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SymbolArgs {
    name: Option<String>,
    id: Option<String>,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct RefsArgs {
    name: Option<String>,
    id: Option<String>,
    #[schemars(range(min = 1))]
    k: Option<usize>,
    tests_only: Option<bool>,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct MapArgs {
    #[schemars(range(min = 1))]
    budget_tokens: Option<usize>,
    focus: Option<Vec<String>>,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct RecentArgs {
    path: Option<String>,
    symbol: Option<String>,
    since: Option<String>,
    #[schemars(extend("enum" = ["pr", "commit", null]))]
    granularity: Option<String>,
    #[schemars(range(min = 1))]
    k: Option<usize>,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct KeyArgs {
    key: String,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SetArgs {
    key: String,
    value: Value,
    #[schemars(extend("enum" = ["repo", "local", null]))]
    layer: Option<String>,
}
fn config_failure(error: config::agent::Error) -> Failure {
    Failure {
        code: "config_refused",
        message: error.to_string(),
        hint: "Inspect effective configuration with config_explain; credentials belong outside the repository.",
    }
}
#[tool_router]
impl Server {
    #[tool(
        description = "Search indexed code and documents. Results are compact; call open with an id for the full unit. Use lex:\"term\" for lexical matching. Unavailable model work is reported as degraded lexical fallback."
    )]
    async fn search(&self, Parameters(args): Parameters<SearchArgs>) -> CallToolResult {
        let start = Instant::now();
        let server = self.clone();
        let prepared = tokio::task::spawn_blocking(move || {
            let context = server.context()?;
            let mode = match args.mode.as_deref().unwrap_or("hybrid") {
                "lexical" => Mode::Lexical,
                "dense" => Mode::Dense,
                "hybrid" => Mode::Hybrid,
                _ => return Err(invalid("mode must be lexical, dense or hybrid")),
            };
            let k = args.k.unwrap_or(context.config.search.default_k as usize);
            let max = context.config.search.max_k.min(50);
            if k == 0 || k as u128 > u128::from(max) {
                return Err(invalid(&format!("k must be in 1..={max}")));
            }
            let scope = args.scope;
            let mut exclude_ids = Vec::new();
            for id in args.exclude_ids.unwrap_or_default() {
                let unit = context
                    .store
                    .unit_by_public_id(&id)
                    .map_err(|_| invalid("Invalid occurrence exclusion id"))?
                    .ok_or_else(|| invalid("Unknown occurrence exclusion id"))?;
                exclude_ids.push(unit.id);
            }
            let request = pipeline::Request {
                query: args.query,
                scope,
                k,
                mode,
                exclude_ids,
            };
            Ok((context, request, args.verbose.unwrap_or(false), args.rerank))
        })
        .await
        .unwrap_or_else(|_| Err(task_error()));
        let value = match prepared {
            Err(error) => Err(error),
            Ok((context, request, verbose, rerank)) => context
                .search_with_rerank(request, verbose, rerank != Some(false))
                .await
                .map_err(Failure::from),
        };
        result(value, start, None, self.l1_status.snapshot())
    }
    #[tool(
        description = "Open an indexed occurrence or symbol by id, or a repository-relative path and line range. Read this evidence before drawing conclusions from search snippets. For capped text, pass continuation_id as id to read the next page; it retains the original context. Non-code extracted_lines count payload lines; indexed_lines is the original locator, not the source span of the page. Freshness limitations are reported."
    )]
    async fn open(&self, Parameters(args): Parameters<OpenArgs>) -> CallToolResult {
        let start = Instant::now();
        let server = self.clone();
        let prepared = tokio::task::spawn_blocking(move || {
            let has_id = args.id.is_some();
            let mut target = target(args.path, args.id)?;
            if let Some([start, end]) = args.lines {
                if has_id || start == 0 || end < start || end > i64::MAX as u64 {
                    return Err(invalid(
                        "lines require a path and positive increasing bounds",
                    ));
                }
                target = format!("{target}:{start}-{end}");
            }
            Ok((server.context()?, target, args.context.unwrap_or(0)))
        })
        .await
        .unwrap_or_else(|_| Err(task_error()));
        let value = match prepared {
            Ok((context, target, lines)) => context.open(&target, lines).await.map_err(Into::into),
            Err(error) => Err(error),
        };
        result(value, start, None, self.l1_status.snapshot())
    }

    #[tool(
        description = "Find symbols by name or public symbol id. Returns signatures, verified locations and approximate relationship metadata; use open to read the source. If any selected source cannot verify, returns source_unavailable; deadlines return timeout. Retry after indexing."
    )]
    async fn symbol(&self, Parameters(args): Parameters<SymbolArgs>) -> CallToolResult {
        let start = Instant::now();
        let server = self.clone();
        let prepared = tokio::task::spawn_blocking(move || {
            Ok((server.context()?, target(args.name, args.id)?))
        })
        .await
        .unwrap_or_else(|_| Err(task_error()));
        let value = match prepared {
            Ok((context, target)) => context.symbol(&target).await.map_err(Into::into),
            Err(error) => Err(error),
        };
        result(value, start, None, self.l1_status.snapshot())
    }

    #[tool(
        description = "Find approximate references to a symbol by name or id, grouped by file. tests_only narrows to test relationships. Unverifiable locations are suppressed with index.partial and suppression counts; timeouts return partial empty groups. References are syntactic and may omit dynamic calls."
    )]
    async fn refs(&self, Parameters(args): Parameters<RefsArgs>) -> CallToolResult {
        let start = Instant::now();
        let server = self.clone();
        let prepared = tokio::task::spawn_blocking(move || {
            Ok((server.context()?, target(args.name, args.id)?))
        })
        .await
        .unwrap_or_else(|_| Err(task_error()));
        let value = match prepared {
            Ok((context, target)) => context
                .refs(&target, args.k, args.tests_only.unwrap_or(false))
                .await
                .map_err(Into::into),
            Err(error) => Err(error),
        };
        result(value, start, None, self.l1_status.snapshot())
    }
    #[tool(
        description = "Render a ranked repository map within a token budget. Focus on paths or symbols to inspect relevant structure before opening files. Unverifiable files are omitted from tree and nodes with index.partial and suppression counts; timeouts return an empty partial map."
    )]
    async fn map(&self, Parameters(args): Parameters<MapArgs>) -> CallToolResult {
        let start = Instant::now();
        let server = self.clone();
        let prepared = tokio::task::spawn_blocking(move || server.context())
            .await
            .unwrap_or_else(|_| Err(task_error()));
        let value = match prepared {
            Ok(context) => context
                .map(
                    args.budget_tokens.unwrap_or(2000),
                    &args.focus.unwrap_or_default(),
                )
                .await
                .map_err(Into::into),
            Err(error) => Err(error),
        };
        result(value, start, Some(60000), self.l1_status.snapshot())
    }
    #[tool(
        description = "Inspect recent PRs with uncovered commits as fallback. Filter by repository path, symbol or ISO/relative date; use commit granularity for individual commits. Changed or unverifiable history is refused."
    )]
    async fn recent(&self, Parameters(args): Parameters<RecentArgs>) -> CallToolResult {
        let start = Instant::now();
        let server = self.clone();
        let prepared = tokio::task::spawn_blocking(move || server.context())
            .await
            .unwrap_or_else(|_| Err(task_error()));
        let value = match prepared {
            Ok(context) => context
                .recent(service::recent::Request {
                    path: args.path,
                    symbol: args.symbol,
                    since: args.since,
                    granularity: args.granularity,
                    k: args.k,
                })
                .await
                .map_err(Into::into),
            Err(error) => Err(error),
        };
        result(value, start, None, self.l1_status.snapshot())
    }
    #[tool(
        description = "Inspect index coverage, current/indexed HEAD, queue states, provider usage, storage caps and recent query latency. Partial or unavailable work remains visible."
    )]
    async fn status(&self) -> CallToolResult {
        let worker = Arc::clone(&self.worker_status);
        let enrichment = Arc::clone(&self.enrichment_status);
        self.run(
            move |context| {
                let mut value = context.status()?;
                value["embedding_worker"] = json!({"state": worker.state()});
                value["enrichment_worker"] = json!({"state": enrichment.state()});
                Ok(value)
            },
            Some(5000),
        )
        .await
    }
    #[tool(
        description = "Read an effective configuration value by dotted key. Returns its source layer; sensitive values are redacted."
    )]
    async fn config_get(&self, Parameters(args): Parameters<KeyArgs>) -> CallToolResult {
        self.run_config(move |root| {
            Ok(json!(
                config::agent::get(root, &args.key).map_err(config_failure)?
            ))
        })
        .await
    }
    #[tool(
        description = "Explain which configuration layer supplies a dotted key. Use this before changing settings; credentials are never returned."
    )]
    async fn config_explain(&self, Parameters(args): Parameters<KeyArgs>) -> CallToolResult {
        self.run_config(move |root| {
            Ok(json!(
                config::agent::explain(root, &args.key).map_err(config_failure)?
            ))
        })
        .await
    }
    #[tool(
        description = "Set a validated configuration value in the local layer by default, or repo explicitly. Refused when serve.agent_writable_config is false. API keys must remain outside repository files."
    )]
    async fn config_set(&self, Parameters(args): Parameters<SetArgs>) -> CallToolResult {
        self.run_config(move |root| {
            let layer = match args.layer.as_deref().unwrap_or("local") {
                "local" => config::agent::WriteLayer::Local,
                "repo" => config::agent::WriteLayer::Repo,
                _ => return Err(invalid("layer must be repo or local")),
            };
            Ok(json!(
                config::agent::set(root, &args.key, args.value, layer).map_err(config_failure)?
            ))
        })
        .await
    }
}
#[tool_handler(router = self.tool_router)]
impl ServerHandler for Server {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.instructions = Some("Repository search and navigation. Call open for evidence. Model integration and per-file freshness remain incomplete; inspect status and result degradation.".into());
        info
    }
}

#[cfg(test)]
mod progress_tests {
    use super::*;
    use crate::serve::progress::{Snapshot, Stage, State};

    fn building() -> Snapshot {
        Snapshot {
            state: State::Building,
            stage: Stage::Files,
            percentage: 50,
            processed: Some(1),
            total: Some(2),
        }
    }

    fn assert_content_matches(result: &CallToolResult) {
        let text = result.content[0].as_text().unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&text.text).unwrap(),
            *result.structured_content.as_ref().unwrap()
        );
    }

    #[tokio::test]
    async fn status_counts_cannot_acquire_a_later_bootstrap_completion() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join(".deepkit")).unwrap();
        std::fs::write(
            root.path().join(".deepkit/config.toml"),
            "[embedder]\nprofile='local'\n",
        )
        .unwrap();
        let server = Server::new(root.path().into());
        assert!(server.l1_status.begin());
        server
            .install_context(service::Context::load(root.path()).unwrap())
            .unwrap();
        let status = server.l1_status.clone();
        let first = server
            .run(
                move |_| {
                    let counts = json!({"l1_pending_files":1,"l2_pending":0});
                    status.observe(sync::index::Progress::Finished { complete: true });
                    Ok(counts)
                },
                None,
            )
            .await;
        let body = first.structured_content.as_ref().unwrap();
        assert_eq!(body["l1_pending_files"], 1);
        assert_eq!(body["index"]["state"], "building");
        assert_eq!(body["index"]["partial"], true);
        assert_content_matches(&first);
        let second = server
            .run(|_| Ok(json!({"l1_pending_files":0})), None)
            .await;
        let body = second.structured_content.as_ref().unwrap();
        assert_eq!(body["index"]["state"], "complete");
        assert_eq!(body["index"]["percentage"], 100);
        assert_eq!(body["l1_pending_files"], 0);
        assert_content_matches(&second);
        let status = server.l1_status.clone();
        let suppressed = server
            .run(
                move |_| {
                    status.suppress();
                    Ok(json!({"l1_pending_files":0}))
                },
                None,
            )
            .await;
        assert!(
            suppressed
                .structured_content
                .as_ref()
                .unwrap()
                .get("index")
                .is_none()
        );
        assert!(!suppressed.meta.as_ref().unwrap().0.contains_key("index"));
        assert_content_matches(&suppressed);
    }

    #[test]
    fn progress_annotates_objects_errors_and_arrays_without_invented_counts() {
        let output = result(
            Ok(json!({"index":{"dense":"off","partial":false},"hits":[]})),
            Instant::now(),
            Some(60000),
            Some(building()),
        );
        let body = output.structured_content.as_ref().unwrap();
        assert_eq!(body["index"]["dense"], "off");
        assert_eq!(body["index"]["state"], "building");
        assert_eq!(body["index"]["partial"], true);
        assert_eq!(body["index"]["l2_pending"], Value::Null);
        assert_eq!(body["index"]["l2_pending_state"], "not_checked");
        assert_eq!(output.meta.as_ref().unwrap().0["ttlMs"], 60000);
        assert_content_matches(&output);
        let status = result(
            Ok(json!({"l2_pending":8})),
            Instant::now(),
            None,
            Some(building()),
        );
        assert_eq!(
            status.structured_content.as_ref().unwrap()["index"]["l2_pending"],
            8
        );
        let array = json!([{"name":"kept"}]);
        let symbols = result(Ok(array.clone()), Instant::now(), None, Some(building()));
        assert_eq!(symbols.structured_content.as_ref().unwrap(), &array);
        assert_eq!(symbols.meta.as_ref().unwrap().0["index"]["percentage"], 50);
        assert_content_matches(&symbols);
        let error = result(
            Err(invalid("invalid request")),
            Instant::now(),
            None,
            Some(building()),
        );
        assert_eq!(error.is_error, Some(true));
        assert_eq!(
            error.structured_content.as_ref().unwrap()["error"]["code"],
            "invalid_arguments"
        );
        assert_eq!(
            error.structured_content.as_ref().unwrap()["index"]["partial"],
            true
        );
        assert_content_matches(&error);
    }

    #[test]
    fn no_attempt_retains_original_shape_and_completion_keeps_source_partial() {
        let array = json!([{"name":"kept"}]);
        let output = result(Ok(array.clone()), Instant::now(), None, None);
        assert_eq!(output.structured_content, Some(array));
        assert!(!output.meta.as_ref().unwrap().0.contains_key("index"));
        let output = result(Ok(json!({"value":true})), Instant::now(), None, None);
        assert!(
            output
                .structured_content
                .as_ref()
                .unwrap()
                .get("index")
                .is_none()
        );
        let mut complete = building();
        complete.state = State::Complete;
        complete.stage = Stage::Finished;
        complete.percentage = 100;
        let output = result(
            Ok(json!({"index":{"partial":true,"reason":"changed_source"}})),
            Instant::now(),
            None,
            Some(complete),
        );
        assert_eq!(
            output.structured_content.as_ref().unwrap()["index"]["partial"],
            true
        );
        assert_eq!(
            output.structured_content.as_ref().unwrap()["index"]["reason"],
            "changed_source"
        );
    }

    #[test]
    fn reload_suppresses_old_progress_and_retains_only_latency_history() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join(".deepkit")).unwrap();
        std::fs::write(
            root.path().join(".deepkit/config.toml"),
            "[embedder]\nprofile='local'\n",
        )
        .unwrap();
        let server = Server::new(root.path().into());
        assert!(server.l1_status.begin());
        let installed = server
            .install_context(service::Context::load(root.path()).unwrap())
            .unwrap();
        assert!(Arc::ptr_eq(
            &installed.background_l1,
            &server.l1_status.active()
        ));
        assert!(Arc::ptr_eq(&installed.query_latencies, &server.latencies));
        let cached = server.context().ok().unwrap();
        assert!(Arc::ptr_eq(&cached, &installed));
        assert!(server.l1_status.snapshot().is_some());
        std::fs::write(
            root.path().join(".deepkit/config.local.toml"),
            "[search]\ndefault_k=7\n",
        )
        .unwrap();
        let reloaded = server.context().ok().unwrap();
        assert_eq!(reloaded.config.search.default_k, 7);
        assert!(!Arc::ptr_eq(
            &reloaded.background_l1,
            &installed.background_l1
        ));
        use std::sync::atomic::Ordering;
        assert!(installed.background_l1.load(Ordering::Acquire));
        assert!(!reloaded.background_l1.load(Ordering::Acquire));
        assert_eq!(server.l1_status.snapshot(), None);
        server
            .l1_status
            .observe(sync::index::Progress::Finished { complete: true });
        assert_eq!(server.l1_status.snapshot(), None);
        std::fs::remove_file(root.path().join(".deepkit/config.local.toml")).unwrap();
        let restored = server.context().ok().unwrap();
        assert!(!Arc::ptr_eq(
            &restored.background_l1,
            &installed.background_l1
        ));
        assert_eq!(server.l1_status.snapshot(), None);
        assert!(Arc::ptr_eq(
            &reloaded.query_latencies,
            &installed.query_latencies
        ));
    }
    #[test]
    fn failed_context_reload_also_suppresses_completed_bootstrap_claim() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join(".deepkit")).unwrap();
        std::fs::write(
            root.path().join(".deepkit/config.toml"),
            "[embedder]\nprofile='local'\n",
        )
        .unwrap();
        let server = Server::new(root.path().into());
        assert!(server.l1_status.begin());
        server
            .install_context(service::Context::load(root.path()).unwrap())
            .unwrap();
        server
            .l1_status
            .observe(sync::index::Progress::Finished { complete: true });
        assert!(server.l1_status.snapshot().is_some());
        std::fs::write(
            root.path().join(".deepkit/config.local.toml"),
            "[index]\nroot='missing-root'\n",
        )
        .unwrap();
        assert!(server.context().is_err());
        assert_eq!(server.l1_status.snapshot(), None);
        assert!(!root.path().join("missing-root").exists());
        std::fs::remove_file(root.path().join(".deepkit/config.local.toml")).unwrap();
        assert!(server.context().is_ok());
        assert_eq!(server.l1_status.snapshot(), None);
    }
}
