//! Transport-owned embedding catalogue and ready-job execution; v1 §8.2/§10.1.
use super::{Error, tools::Server};
use crate::{queue::host, search::service::Context};
use std::{
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

/// Process-local worker health; durable job/usage progress remains in SQLite.
#[derive(Default)]
pub(super) struct Status(AtomicU8);
impl Status {
    /// Reports health without exposing provider errors or repository input.
    pub(super) fn state(&self) -> &'static str {
        match self.0.load(Ordering::Acquire) {
            0 => "starting",
            1 => "running",
            2 => "waiting",
            _ => "stopped",
        }
    }
    fn set(&self, state: u8) {
        self.0.store(state, Ordering::Release);
    }
}
fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|time| i64::try_from(time.as_secs()).ok())
        .unwrap_or(-1)
}

/// Settings remain observable even when the configured index cannot be opened.
#[derive(PartialEq)]
struct Observation {
    config: Option<crate::config::Config>,
    source: Option<Source>,
}
#[derive(PartialEq)]
struct Source {
    head: Option<String>,
    index: crate::store::l1::Identity,
    summaries: Option<i64>,
    prompts: Option<[[u8; 32]; 3]>,
}

/// Keeps failures local to the worker so tools can inspect and repair settings.
/// The caller polls this future beside the transport, dropping both on shutdown.
pub(super) async fn run(server: Server, status: Arc<Status>) {
    loop {
        let observed = observation(&server).await;
        status.set(0);
        if let (Some(config), Some(source)) = (&observed.config, &observed.source) {
            let _ = execute(server.clone(), &status, config, source).await;
        }
        status.set(3);
        // Provider errors may contain input data. Emit only this fixed diagnostic.
        tracing::warn!(
            "Embedding worker stopped; awaiting changed configuration, HEAD or index generation"
        );
        // A new input observation rebuilds hosts and catalogue cursors. It never
        // resets persisted generations, retry delays, leases or usage charges.
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
            if observation(&server).await != observed {
                break;
            }
        }
    }
}

/// Runs independently of model readiness, retaining retry state between idle polls.
pub(super) async fn run_enrichment(server: Server, status: Arc<Status>) {
    loop {
        let observed = observe(&server, false).await;
        status.set(0);
        if let (Some(config), Some(source)) = (&observed.config, &observed.source) {
            let _ = execute_enrichment(server.clone(), &status, config, source).await;
        }
        status.set(3);
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
            if observe(&server, false).await != observed {
                break;
            }
        }
    }
}

async fn execute_enrichment(
    server: Server,
    status: &Status,
    config: &crate::config::Config,
    expected: &Source,
) -> Result<(), Error> {
    let context = tokio::task::spawn_blocking(move || server.worker_context())
        .await
        .map_err(|_| Error::Worker)??;
    if &context.config != config {
        return Err(Error::Worker);
    }
    check(&context, expected).await?;
    let Some(mut host) = crate::queue::enrichment_host::Host::prepare(
        Arc::clone(&context.store),
        context.root.clone(),
        context.config_root.clone(),
        config.clone(),
        now(),
    )
    .await
    .map_err(|_| Error::Worker)?
    else {
        return Ok(());
    };
    loop {
        check(&context, expected).await?;
        status.set(1);
        let pass = host.pass(now).await.map_err(|_| Error::Worker)?;
        if pass.error.is_some() || !pass.execution.errors.is_empty() {
            return Err(Error::Worker);
        }
        if pass.execution.claimed == 0 && host.catalogue_finished().map_err(|_| Error::Worker)? {
            status.set(2);
            tokio::time::sleep(Duration::from_secs(1)).await;
        } else {
            tokio::task::yield_now().await;
        }
    }
}

async fn observation(server: &Server) -> Observation {
    observe(server, true).await
}

async fn observe(server: &Server, include_summaries: bool) -> Observation {
    let server = server.clone();
    tokio::task::spawn_blocking(move || {
        let config = server.worker_config().ok();
        let source = config.as_ref().and_then(|config| {
            let context = server.worker_context().ok()?;
            (&context.config == config)
                .then(|| source(&context).ok())
                .flatten()
        });
        let source = source.map(|mut source| {
            if !include_summaries {
                source.summaries = None;
            }
            source
        });
        Observation { config, source }
    })
    .await
    .unwrap_or(Observation {
        config: None,
        source: None,
    })
}

fn source(context: &Context) -> Result<Source, Error> {
    let read_head = || {
        crate::sync::head::read(&context.root, Default::default())
            .map(|head| head.map(|head| head.id))
            .map_err(|_| Error::Worker)
    };
    let head = read_head()?;
    let (index, summaries) = context
        .store
        .read_snapshot(|tx| {
            Ok::<_, crate::store::Error>((
                crate::store::l1::identity_in(tx)?,
                crate::store::enrichment::publication::summary_generation_in(tx)?,
            ))
        })
        .map_err(|_| Error::Worker)?
        .map_err(|_| Error::Worker)?;
    let prompts = if context.config.llm.enabled {
        let mut fingerprints = [[0; 32]; 3];
        for (slot, kind) in fingerprints.iter_mut().zip([
            crate::queue::Kind::DescribeSymbol,
            crate::queue::Kind::SummarizePr,
            crate::queue::Kind::SummarizeModule,
        ]) {
            *slot = crate::treat::prompts::Template::load(
                &context.config_root,
                &context.config.enrich.prompts_dir,
                kind,
            )
            .map_err(|_| Error::Worker)?
            .fingerprint;
        }
        Some(fingerprints)
    } else {
        None
    };
    if read_head()? != head {
        return Err(Error::Worker);
    }
    Ok(Source {
        head,
        index,
        summaries: Some(summaries),
        prompts,
    })
}

async fn check(context: &Arc<Context>, expected: &Source) -> Result<(), Error> {
    let context = Arc::clone(context);
    let mut current = tokio::task::spawn_blocking(move || {
        let config = crate::config::load(&context.config_root, None)
            .map_err(|_| Error::Worker)?
            .config;
        if config != context.config {
            return Err(Error::Worker);
        }
        source(&context)
    })
    .await
    .map_err(|_| Error::Worker)??;
    // Chat retains its shared client and leases across its own publications.
    // Embedding hosts still restart their catalogue on each committed generation.
    if expected.summaries.is_none() {
        current.summaries = None;
    }
    if &current != expected
        || crate::sync::head::changed(
            current.index.indexed_head.as_deref(),
            current.head.as_deref(),
        )
    {
        return Err(Error::Worker);
    }
    Ok(())
}
fn valid(pass: &host::Report) -> Result<(), Error> {
    if !pass.execution.errors.is_empty()
        || pass.retirement.error.is_some()
        || pass.transition_error.is_some()
    {
        return Err(Error::Worker);
    }
    Ok(())
}
async fn execute(
    server: Server,
    status: &Status,
    config: &crate::config::Config,
    expected: &Source,
) -> Result<(), Error> {
    let context = tokio::task::spawn_blocking(move || server.worker_context())
        .await
        .map_err(|_| Error::Worker)??;
    // A settings change during context acquisition belongs to the next rebuild.
    if &context.config != config {
        return Err(Error::Worker);
    }
    check(&context, expected).await?;
    let source = Arc::clone(&context);
    let kinds = tokio::task::spawn_blocking(move || {
        source.store.embedding_catalogue_kinds(
            source.config.commits.enabled,
            source.config.commits.enabled && source.config.commits.prs,
            source.config.enrich.top_symbols,
            source.config.enrich.modules_min_files,
        )
    })
    .await
    .map_err(|_| Error::Worker)?
    .map_err(|_| Error::Worker)?;
    let group = host::group::Group::prepare(
        Arc::clone(&context.store),
        host::group::Settings {
            root: context.root.clone(),
            config_root: context.config_root.clone(),
            config: context.config.clone(),
            local_cache: crate::models::registry::cache_root().map_err(|_| Error::Worker)?,
        },
        &kinds,
        |provider| host::Bounds::for_provider(&context.config.queue, provider),
        now(),
    )
    .await
    .map_err(|_| Error::Worker)?;
    status.set(1);
    for entry in &group.entries {
        let mut cursor = entry
            .host
            .start_catalogue()
            .await
            .map_err(|_| Error::Worker)?;
        while !cursor.finished() {
            check(&context, expected).await?;
            let pass = entry
                .host
                .run_catalogue(&mut cursor, |candidate| Ok(candidate.recency(now())), now)
                .await
                .map_err(|_| Error::Worker)?;
            valid(&pass)?;
        }
    }
    loop {
        check(&context, expected).await?;
        status.set(1);
        let mut ready = false;
        for entry in &group.entries {
            check(&context, expected).await?;
            // Keep polling ready work and retry recovery across idle intervals;
            // the finite CLI drain instead returns when work must wait.
            let report = entry
                .host
                .run(Vec::new(), now)
                .await
                .map_err(|_| Error::Worker)?;
            valid(&report)?;
            ready |= report.execution.claimed > 0;
        }
        // Retry delays, quota waits and leases stay authoritative in the queue.
        // Yield between busy passes; poll waiting queues without spinning.
        if ready {
            tokio::task::yield_now().await;
        } else {
            status.set(2);
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::enrichment::publication::{Owner, Publication};

    #[tokio::test]
    async fn committed_summary_restarts_catalogue_observation_without_l1_edits() {
        let (root, store, _) = crate::queue::source::tests::fixture().unwrap();
        store.writer().unwrap().execute("INSERT INTO symbols(id,file_id,unit_id,name,kind,line_start,line_end) VALUES(1,1,1,'example','function',1,1)",[]).unwrap();
        let mut context = Context::load(root.path()).unwrap();
        context.store = Arc::new(store);
        let context = Arc::new(context);
        let before = source(&context).unwrap();
        check(&context, &before).await.unwrap();
        let publish = |commit: bool| {
            let mut db = context.store.writer().unwrap();
            let tx = db.transaction().unwrap();
            let hash: Vec<u8> = tx
                .query_row("SELECT content_hash FROM units WHERE id=1", [], |r| {
                    r.get(0)
                })
                .unwrap();
            let hash: [u8; 32] = hash.try_into().unwrap();
            context
                .store
                .publish_summary_in(
                    &tx,
                    Publication {
                        owner: Owner::Symbol(1),
                        parent_unit: 1,
                        parent_hash: &hash,
                        desired_hash: &[7; 32],
                        text: "A published summary.",
                        sources: &[],
                        now: 0,
                    },
                )
                .unwrap();
            if commit {
                tx.commit().unwrap();
            }
        };
        publish(false);
        check(&context, &before).await.unwrap();
        publish(true);
        let after = source(&context).unwrap();
        assert_eq!(before.head, after.head);
        assert_eq!(before.index, after.index);
        assert_eq!((before.summaries, after.summaries), (Some(0), Some(1)));
        let chat_observation = Source {
            head: before.head.clone(),
            index: context.store.l1_identity().unwrap(),
            summaries: None,
            prompts: before.prompts,
        };
        check(&context, &chat_observation).await.unwrap();
        assert!(check(&context, &before).await.is_err());
        check(&context, &after).await.unwrap();
        publish(true);
        check(&context, &after).await.unwrap();
    }
}
