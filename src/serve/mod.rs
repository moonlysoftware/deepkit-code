//! MCP stdio and streamable HTTP transports; see `v1.md` §10.1/10.3.
mod bootstrap;
mod progress;
mod tools;
mod watcher;
mod worker;
use crate::config;
use rmcp::{
    ServiceExt,
    transport::{
        stdio,
        streamable_http_server::{
            StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
        },
    },
};
use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
};

/// Transport setup and shutdown failures, kept separate from structured tool errors.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error(transparent)]
    Config(#[from] config::Error),
    #[error("an existing index is required; run dkc index --level 1 first")]
    MissingIndex,
    #[error("Initial index setup failed: {0}")]
    Bootstrap(#[source] Box<bootstrap::Error>),
    #[error("benchmark index setup failed: {0}")]
    BenchmarkContext(#[source] Box<crate::search::service::Error>),
    #[error("embedding worker stopped; inspect queue/configuration and restart serve")]
    Worker,
    #[error("filesystem watcher failed: {0}")]
    Watch(#[from] crate::sync::watch::BackendError),
    #[error("serve.http_token is required for a non-loopback bind")]
    MissingToken,
    #[error("invalid serve.http address")]
    Address,
    #[error(
        "MCP HTTP address {address} is already in use; use `dkc serve --stdio` for a separate project connection, or choose a different explicit `dkc serve --http <address:port>` and update the client's MCP URL"
    )]
    HttpAddressInUse {
        address: SocketAddr,
        #[source]
        source: std::io::Error,
    },
    #[error("could not bind MCP HTTP server to {address}")]
    HttpBind {
        address: SocketAddr,
        #[source]
        source: std::io::Error,
    },
    #[error("MCP transport failed: {0}")]
    Transport(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
fn existing_index(root: &Path, settings: &config::Config) -> bool {
    let configured = PathBuf::from(&settings.index.root);
    let root = if configured.is_absolute() {
        configured
    } else {
        root.join(configured)
    };
    let database = PathBuf::from(&settings.store.path);
    (if database.is_absolute() {
        database
    } else {
        root.join(database)
    })
    .is_file()
}
/// Runs the selected transport in this process; HTTP shares backend state across requests.
pub(crate) async fn run(root: PathBuf, http: Option<String>, poll: bool) -> Result<(), Error> {
    run_mode(root, http, poll, None).await
}

/// Keeps the same watcher and queue worker alive without opening an MCP transport.
pub(crate) async fn watch(root: PathBuf, poll: bool, json: bool) -> Result<(), Error> {
    run_mode(root, None, poll, Some(json)).await
}

/// Serves real MCP stdio tools over an existing index for a model-free benchmark.
/// Workers, bootstrap and the watcher would change a retained index during timing.
pub(crate) async fn benchmark_stdio(root: PathBuf) -> Result<(), Error> {
    let settings = config::load(&root, None)?.config;
    if !existing_index(&root, &settings) {
        return Err(Error::MissingIndex);
    }
    let context = crate::search::service::Context::load_with_config(&root, settings)
        .map_err(|error| Error::BenchmarkContext(Box::new(error)))?;
    let server = tools::Server::new(root.clone());
    server.install_context(context)?;
    transport(root, None, server).await
}

async fn run_mode(
    root: PathBuf,
    http: Option<String>,
    poll: bool,
    watch_output: Option<bool>,
) -> Result<(), Error> {
    let settings = config::load(&root, None)?.config;
    let listener = match http_address(http, &settings.serve)? {
        Some(address) => Some(bind_http(address).await?),
        None => None,
    };
    // Reserve the requested endpoint before setup or background work can write
    // repository state. A failed bind must not start an otherwise unused server.
    let missing = !existing_index(&root, &settings);
    if missing && !settings.serve.auto_init {
        return Err(Error::MissingIndex);
    }
    let config_root = root.clone();
    let (context, scheduled) =
        tokio::task::spawn_blocking(move || bootstrap::prepare(&config_root, settings, missing))
            .await
            .map_err(|_| Error::Worker)?
            .map_err(|error| Error::Bootstrap(Box::new(error)))?;
    let server = tools::Server::new(root.clone());
    if scheduled {
        server.l1_status.begin();
    }
    let context = server.install_context(context)?;
    let watched_root = context.root.clone();
    let watcher = tokio::task::spawn_blocking(move || {
        crate::sync::watch::Watcher::start(&watched_root, poll)
    })
    .await
    .map_err(|_| Error::Worker)??;
    watcher::ready(watch_output, watcher.backend())?;
    let status = Arc::clone(&server.worker_status);
    // Poll signal registration before starting either transport. Dropping this
    // transport future cancels the stdio service and releases the HTTP listener.
    tokio::select! {
        biased;
        signal = interrupted() => signal.map_err(Error::from),
        result = async {
            if watch_output.is_some() { std::future::pending().await }
            else { transport(root, listener, server.clone()).await }
        } => result,
        () = worker::run(server.clone(), status) => Ok(()),
        () = worker::run_enrichment(server.clone(), Arc::clone(&server.enrichment_status)) => Ok(()),
        () = bootstrap::run(server.clone(), Arc::clone(&context), scheduled) => Ok(()),
        result = watcher::run(server.clone(), context, watcher, watch_output) => result,
    }
}

fn http_address(
    http: Option<String>,
    settings: &config::Serve,
) -> Result<Option<SocketAddr>, Error> {
    let address: Option<SocketAddr> = http
        .map(|address| {
            if address.is_empty() {
                settings.http.clone()
            } else {
                address
            }
        })
        .map(|address| address.parse().map_err(|_| Error::Address))
        .transpose()?;
    if address.is_some_and(|address| !address.ip().is_loopback()) && settings.http_token.is_empty()
    {
        return Err(Error::MissingToken);
    }
    Ok(address)
}

async fn bind_http(address: SocketAddr) -> Result<tokio::net::TcpListener, Error> {
    tokio::net::TcpListener::bind(address)
        .await
        .map_err(|source| match source.kind() {
            std::io::ErrorKind::AddrInUse => Error::HttpAddressInUse { address, source },
            _ => Error::HttpBind { address, source },
        })
}

/// Waits for process shutdown, including Unix service-manager termination.
pub(crate) async fn interrupted() -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut interrupt = signal(SignalKind::interrupt())?;
        let mut terminate = signal(SignalKind::terminate())?;
        tokio::select! {
            _ = interrupt.recv() => {},
            _ = terminate.recv() => {},
        }
        Ok(())
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await
}
async fn transport(
    root: PathBuf,
    listener: Option<tokio::net::TcpListener>,
    server: tools::Server,
) -> Result<(), Error> {
    match listener {
        None => {
            server
                .serve(stdio())
                .await
                .map_err(|error| Error::Transport(error.to_string()))?
                .waiting()
                .await
                .map_err(|error| Error::Transport(error.to_string()))?;
        }
        Some(listener) => {
            let address = listener.local_addr()?;
            let mut transport_config = StreamableHttpServerConfig::default()
                .with_json_response(true)
                .with_legacy_session_mode(false);
            // Non-loopback requests always require a bearer, including after config reload.
            // Permit LAN hostnames/IPs without imposing an undocumented hostname setting.
            if !address.ip().is_loopback() {
                transport_config = transport_config.disable_allowed_hosts();
            }
            let service = StreamableHttpService::new(
                move || Ok(server.clone()),
                Arc::new(LocalSessionManager::default()),
                transport_config,
            );
            let config_root = Arc::new(root);
            let app =
                axum::Router::new()
                    .nest_service("/mcp", service)
                    .layer(axum::middleware::from_fn(
                        move |request: axum::extract::Request, next: axum::middleware::Next| {
                            let config_root = Arc::clone(&config_root);
                            async move {
                                let expected = match tokio::task::spawn_blocking(move || {
                                    config::load(&config_root, None)
                                        .map(|loaded| loaded.config.serve.http_token)
                                })
                                .await
                                {
                                    Ok(Ok(token)) => token,
                                    _ => {
                                        return axum::response::IntoResponse::into_response(
                                            axum::http::StatusCode::SERVICE_UNAVAILABLE,
                                        );
                                    }
                                };
                                if (!address.ip().is_loopback() && expected.is_empty())
                                    || (!expected.is_empty()
                                        && request
                                            .headers()
                                            .get(axum::http::header::AUTHORIZATION)
                                            .and_then(|value| value.to_str().ok())
                                            .and_then(|value| value.strip_prefix("Bearer "))
                                            != Some(expected.as_str()))
                                {
                                    return axum::response::IntoResponse::into_response(
                                        axum::http::StatusCode::UNAUTHORIZED,
                                    );
                                }
                                next.run(request).await
                            }
                        },
                    ));
            tracing::info!(%address, "MCP listening");
            axum::serve(listener, app).await?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_address_preserves_stdio_defaults_and_explicit_selection() {
        let mut settings = config::Serve::default();
        assert_eq!(http_address(None, &settings).unwrap(), None);
        assert_eq!(
            http_address(Some(String::new()), &settings).unwrap(),
            Some("127.0.0.1:7433".parse().unwrap())
        );
        settings.http = "127.0.0.1:8433".into();
        assert_eq!(
            http_address(Some(String::new()), &settings).unwrap(),
            Some("127.0.0.1:8433".parse().unwrap())
        );
        assert_eq!(
            http_address(Some("127.0.0.1:9433".into()), &settings).unwrap(),
            Some("127.0.0.1:9433".parse().unwrap())
        );
        assert!(matches!(
            http_address(Some("invalid".into()), &settings),
            Err(Error::Address)
        ));
    }

    #[test]
    fn nonloopback_token_guard_precedes_http_binding() {
        let mut settings = config::Serve {
            http: "0.0.0.0:7433".into(),
            ..Default::default()
        };
        assert!(matches!(
            http_address(Some(String::new()), &settings),
            Err(Error::MissingToken)
        ));
        settings.http_token = "fixture".into();
        assert_eq!(
            http_address(Some(String::new()), &settings).unwrap(),
            Some("0.0.0.0:7433".parse().unwrap())
        );
    }

    #[tokio::test]
    async fn occupied_http_address_retains_os_error_and_actionable_remedies() {
        let held = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = held.local_addr().unwrap();
        let error = bind_http(address).await.unwrap_err();
        let message = error.to_string();
        assert!(message.contains(&address.to_string()));
        assert!(message.contains("dkc serve --stdio"));
        assert!(message.contains("dkc serve --http <address:port>"));
        assert!(message.contains("client's MCP URL"));
        match error {
            Error::HttpAddressInUse {
                address: reported,
                source,
            } => {
                assert_eq!(reported, address);
                assert_eq!(source.kind(), std::io::ErrorKind::AddrInUse);
            }
            other => panic!("unexpected HTTP bind error: {other}"),
        }
    }
}
