//! Process-owned watcher publication and cancellation; v1.md §2.1/§4.5/§10.2.

use super::{Error, tools::Server};
use crate::{
    search::service::Context,
    sync::{
        self,
        watch::{Batch, Watcher},
    },
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

struct Cancel(Arc<AtomicBool>);
impl Drop for Cancel {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

/// Announces registration only on the standalone watch CLI, never MCP stdout.
pub(super) fn ready(output: Option<bool>, backend: &str) -> Result<(), Error> {
    match output {
        Some(true) => outln!(
            "{}",
            serde_json::json!({"watch": {"state": "watching", "backend": backend}})
        ),
        Some(false) => outln!("Watching repository changes ({backend})"),
        None => {}
    }
    Ok(())
}

/// Serializes event batches after bootstrap, sharing source/graph/queue fences.
/// Failed work is retained until another event or changed configuration; stable
/// parser/cap failures do not cause an endless full-repository retry loop.
pub(super) async fn run(
    server: Server,
    initial: Arc<Context>,
    mut watcher: Watcher,
    output: Option<bool>,
) -> Result<(), Error> {
    let cancelled = Arc::new(AtomicBool::new(false));
    let _cancel = Cancel(Arc::clone(&cancelled));
    let mut fingerprint = sync::l1::settings_fingerprint(&initial.config);
    let mut retained = Batch::default();
    let mut unavailable = false;
    loop {
        // Registration already owns events while initial indexing is active.
        while server.l1_status.active().load(Ordering::Acquire) {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let mut batch = tokio::select! {
            batch = watcher.next() => batch,
            () = tokio::time::sleep(Duration::from_secs(1)) => Batch::default(),
        };
        let copy = server.clone();
        let context = match tokio::task::spawn_blocking(move || copy.worker_context()).await {
            Ok(Ok(context)) => context,
            _ => {
                if !unavailable {
                    tracing::warn!("Watcher waiting for valid index configuration");
                    if output == Some(true) {
                        outln!(
                            "{}",
                            serde_json::json!({"watch": {"state": "waiting_configuration"}})
                        );
                    }
                }
                retained.merge(batch);
                unavailable = true;
                continue;
            }
        };
        let next_fingerprint = sync::l1::settings_fingerprint(&context.config);
        let changed =
            unavailable || next_fingerprint != fingerprint || watcher.root() != context.root;
        unavailable = false;
        if batch.empty() && !changed {
            continue;
        }
        batch.merge(std::mem::take(&mut retained));
        batch.rescan |= changed;
        if watcher.root() != context.root || batch.backend_failed {
            let root = context.root.clone();
            let poll = batch.backend_failed || watcher.backend() == "polling";
            watcher = tokio::task::spawn_blocking(move || Watcher::start(&root, poll))
                .await
                .map_err(|_| Error::Worker)??;
            batch.rescan = true;
            batch.backend_failed = false;
            ready(output, watcher.backend())?;
        }
        fingerprint = next_fingerprint;
        let full = changed || batch.rescan;
        // Watcher work shares durable L1 admission with query repair. Bootstrap
        // alone owns the fail-fast background flag and its public progress claim.
        let status = super::progress::Status::default();
        status.begin();
        let observed = status.clone();
        let work_cancelled = Arc::clone(&cancelled);
        let paths = batch.paths.clone();
        let result = tokio::task::spawn_blocking(move || {
            let input = sync::index::Input {
                config_root: &context.config_root,
                root: &context.root,
                settings: &context.config,
                store: &context.store,
                full: false,
                work: sync::l1::Work {
                    deadline: None,
                    cancelled: Some(&work_cancelled),
                },
            };
            if full {
                sync::index::run(input, |progress| observed.observe(progress))
            } else {
                sync::index::run_paths(input, &paths, |progress| observed.observe(progress))
            }
        })
        .await;
        match result {
            Ok(Ok(report)) => {
                if let Some(json) = output {
                    let changes = &report.publication.changes;
                    if json {
                        outln!(
                            "{}",
                            serde_json::json!({"watch": {
                                "state": if report.scan.is_complete() { "updated" } else { "partial" },
                                "backend": watcher.backend(), "full_scan": full,
                                "generation": report.publication.generation,
                                "scanned_files": report.scan.files().count(),
                                "added": changes.added.len(), "modified": changes.modified.len(),
                                "deleted": changes.deleted.len(), "renamed": changes.renamed.len()
                            }})
                        );
                    } else {
                        outln!(
                            "Updated index: {} added, {} modified, {} deleted, {} renamed",
                            changes.added.len(),
                            changes.modified.len(),
                            changes.deleted.len(),
                            changes.renamed.len()
                        );
                    }
                }
            }
            failure => {
                status.fail();
                let stage = match &failure {
                    Ok(Err(error)) => super::bootstrap::failure_kind(error),
                    _ => "worker",
                };
                tracing::warn!(
                    stage,
                    "Watcher update failed; retaining paths until the next source or configuration change"
                );
                if output == Some(true) {
                    outln!(
                        "{}",
                        serde_json::json!({"watch": {"state": "failed", "stage": stage}})
                    );
                }
                retained.merge(batch);
            }
        }
    }
}
