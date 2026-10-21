//! Transport-owned initial L1 work with resumable per-file publication; v1 §2.1/10.2.
use super::tools::Server;
use crate::sync::{self, l1::Work};
use std::{
    future::pending,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

/// Cancels cooperative blocking work when the transport or supervisor is dropped.
struct CancelOnDrop(Arc<AtomicBool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

/// Runs one admitted initial/recovery pass; completion must not stop the transport.
/// The startup path marks activity before serving the first request. A failed pass
/// is visible and retains pending work; it never starts an unbounded retry loop.
pub(super) async fn run(
    server: Server,
    context: Arc<crate::search::service::Context>,
    scheduled: bool,
) {
    if !scheduled {
        pending::<()>().await;
    }
    let cancelled = Arc::new(AtomicBool::new(false));
    let _cancel = CancelOnDrop(Arc::clone(&cancelled));
    let status = server.l1_status.clone();
    let observed = status.clone();
    let result = tokio::task::spawn_blocking(move || {
        // Retain the context admitted by startup. A configuration reload must
        // fail existing fences, never redirect this pass to an unguarded path.
        sync::index::run(
            sync::index::Input {
                config_root: &context.config_root,
                root: &context.root,
                settings: &context.config,
                store: &context.store,
                full: false,
                work: Work {
                    deadline: None,
                    cancelled: Some(&cancelled),
                },
            },
            |progress| observed.observe(progress),
        )
        .map(|_| ())
        .map_err(|error| failure_kind(&error))
    })
    .await;
    match result {
        Ok(Ok(())) => {}
        Ok(Err(stage)) => {
            status.fail();
            tracing::warn!(
                stage,
                "Initial L1 failed; run dkc index --level 1 to diagnose and resume"
            );
        }
        Err(_) => {
            status.fail();
            tracing::warn!(
                "Initial L1 worker stopped; run dkc index --level 1 to diagnose and resume"
            );
        }
    }
    pending::<()>().await;
}

/// Classifies failures without logging source paths, configuration or parser text.
pub(super) fn failure_kind(error: &sync::index::Error) -> &'static str {
    use crate::store;
    fn stored(error: &store::Error) -> &'static str {
        match error {
            store::Error::WalCapacity { .. } => "wal_capacity",
            store::Error::Sqlite(rusqlite::Error::SqliteFailure(code, _)) => match code.code {
                rusqlite::ErrorCode::DatabaseBusy => "sqlite_busy",
                rusqlite::ErrorCode::DatabaseLocked => "sqlite_locked",
                _ => "sqlite_failure",
            },
            store::Error::L1InputsChanged => "inputs_changed",
            _ => "store",
        }
    }
    fn synced(error: &sync::Error) -> &'static str {
        match error {
            sync::Error::Store(error) => stored(error),
            sync::Error::WorkStopped => "cancelled",
            sync::Error::SourceChanged { .. } => "source_changed",
            _ => "sync",
        }
    }
    match error {
        sync::index::Error::Store(error) => stored(error),
        sync::index::Error::Sync(error) => synced(error),
        sync::index::Error::File { source, .. } => synced(source),
        sync::index::Error::ConfigurationChanged => "configuration_changed",
        sync::index::Error::HeadChanged => "head_changed",
        _ => "indexing",
    }
}

/// Bootstrap failures remain typed and contain no repository configuration values.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("Initial index setup failed")]
    Setup(#[from] sync::setup::Error),
    #[error("Initial index path is unsafe")]
    Path(#[from] crate::store::path::Error),
    #[error("Initial index configuration could not be read")]
    Config(#[from] crate::config::Error),
    #[error("Initial index could not be opened")]
    Context(#[source] Box<crate::search::service::Error>),
    #[error("Initial index recovery state could not be recorded")]
    Store(#[source] Box<crate::store::Error>),
    #[error("Initial repository root could not be resolved")]
    Io(#[from] std::io::Error),
    #[error("Configuration changed during initial setup; retry serve")]
    Changed,
}

/// Creates only index data and the owned exclude block, or resumes marked L1 work.
/// Existing indexes are not rescanned merely because an MCP client reconnects.
pub(super) fn prepare(
    config_root: &std::path::Path,
    settings: crate::config::Config,
    missing: bool,
) -> Result<(crate::search::service::Context, bool), Error> {
    let configured = std::path::PathBuf::from(&settings.index.root);
    let root = std::fs::canonicalize(if configured.is_absolute() {
        configured
    } else {
        config_root.join(configured)
    })?;
    let database = crate::store::path::prepare(&root, std::path::Path::new(&settings.store.path))?;
    let excludes = missing
        .then(|| sync::setup::prepare_excludes(&root))
        .transpose()?;
    if crate::config::load(config_root, None)?.config != settings {
        return Err(Error::Changed);
    }
    if let Some(excludes) = excludes {
        excludes.publish()?;
    }
    database.check_for_open()?;
    let auto_init = settings.serve.auto_init;
    let context = crate::search::service::Context::load_with_config(config_root, settings)
        .map_err(|error| Error::Context(Box::new(error)))?;
    let marked = context
        .store
        .get_meta("serve_auto_init")
        .map_err(|error| Error::Store(Box::new(error)))?
        .as_deref()
        == Some("1");
    let identity = context
        .store
        .l1_identity()
        .map_err(|error| Error::Store(Box::new(error)))?;
    let pending = context
        .store
        .counts()
        .map_err(|error| Error::Store(Box::new(error)))?
        .l1_pending_files;
    // Database/schema creation may finish before the ownership marker is saved.
    // Identity zero must therefore resume even without that marker after a crash.
    let scheduled = auto_init
        && (missing
            || identity.completed == 0
            || (marked && (identity.completed != identity.scan || pending > 0)));
    // A prior status/schema-only open (or a crash) may leave a database without
    // the exclude block. Recovery must hide generated files before indexing too.
    if scheduled && !missing {
        sync::setup::prepare_excludes(&root)?.publish()?;
    }
    if scheduled && !marked {
        context
            .store
            .set_meta("serve_auto_init", "1")
            .map_err(|error| Error::Store(Box::new(error)))?;
    }
    Ok((context, scheduled))
}

#[cfg(test)]
mod captured_authority_tests {
    use super::*;
    use crate::{config, search::service::Context, serve::progress::State};
    use std::{fs, time::Duration};

    #[tokio::test]
    async fn startup_config_redirect_fails_without_touching_the_replacement_index() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        let mut settings = config::Config::default();
        settings.embedder.profile = "local".into();
        settings.llm.enabled = false;
        for root in [first.path(), second.path()] {
            fs::create_dir(root.join(".deepkit")).unwrap();
            fs::write(root.join(".deepkit/config.toml"), settings.dump().unwrap()).unwrap();
            fs::write(root.join("source.rs"), "pub fn captured() {}\n").unwrap();
        }
        // B has a real, openable database: the old worker_context reload would
        // accept it and publish there instead of refusing changed A authority.
        let replacement = Context::load(second.path()).unwrap();
        let replacement_before = replacement.store.l1_identity().unwrap();
        let replacement_changes = replacement.store.writer().unwrap().total_changes();
        let settings = config::load(first.path(), None).unwrap().config;
        let (context, scheduled) = prepare(first.path(), settings, true).unwrap();
        assert!(scheduled);
        let server = Server::new(first.path().into());
        assert!(server.l1_status.begin());
        let captured = server.install_context(context).unwrap();
        let captured_changes = captured.store.writer().unwrap().total_changes();
        let mut redirected = captured.config.clone();
        redirected.index.root = second.path().to_string_lossy().into_owned();
        fs::write(
            first.path().join(".deepkit/config.local.toml"),
            redirected.dump().unwrap(),
        )
        .unwrap();

        let supervisor = run(server.clone(), Arc::clone(&captured), true);
        tokio::pin!(supervisor);
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                tokio::select! {
                    () = &mut supervisor => panic!("bootstrap completion must not stop transport"),
                    () = tokio::time::sleep(Duration::from_millis(5)) => {
                        if server.l1_status.snapshot().is_some_and(|value| value.state == State::Failed) {
                            break;
                        }
                    }
                }
            }
        }).await.unwrap();
        assert_eq!(
            captured.store.writer().unwrap().total_changes(),
            captured_changes
        );
        assert_eq!(
            replacement.store.writer().unwrap().total_changes(),
            replacement_changes
        );
        assert_eq!(replacement.store.l1_identity().unwrap(), replacement_before);
        assert_eq!(replacement.store.counts().unwrap().files, 0);
        assert!(!second.path().join(".git/info/exclude").exists());
        assert!(!captured.background_l1.load(Ordering::Acquire));
    }
}
