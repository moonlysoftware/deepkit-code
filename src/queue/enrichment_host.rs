//! Finite chat-only catalogue passes independent of embedding providers; v1 §6.8/§8.
use super::{
    Error, Kind, capacity, catalogue,
    enrichment::{self, Context},
    enrichment_catalogue, host, pool, priority, retry, visual_source,
};
use crate::{config::Config, models::openai_compat::chat, store::Store, sync};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Clone)]
struct Target {
    kind: Kind,
    id: i64,
    priority: f64,
    referenced_image: bool,
    page: Option<u32>,
    source_hash: Option<[u8; 32]>,
}
/// Durable counts distinguish current coverage, new queue work and refused sources.
#[derive(Default)]
pub(crate) struct Report {
    pub(crate) visited: usize,
    pub(crate) scheduled: usize,
    pub(crate) current: usize,
    pub(crate) skipped: usize,
    pub(crate) paused: usize,
    pub(crate) terminal: usize,
    pub(crate) error: Option<Error>,
    pub(crate) execution: pool::Report,
}
/// One shared chat client and a captured finite cohort; no embedder/model cache is opened.
pub(crate) struct Host {
    context: Arc<Context>,
    settings: Arc<pool::Settings>,
    targets: Vec<Target>,
    position: usize,
    source: crate::store::l1::Identity,
    head: Option<String>,
    max_schedule: usize,
    modules_from: usize,
    observed_summaries: i64,
}
struct Stop(Arc<AtomicBool>);
impl Drop for Stop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}
impl Host {
    /// Resolves chat credentials only when enabled; disabled enrichment is a zero-work path.
    pub(crate) async fn prepare(
        store: Arc<Store>,
        root: PathBuf,
        config_root: PathBuf,
        config: Config,
        now: i64,
    ) -> Result<Option<Self>, Error> {
        Self::prepare_mode(store, root, config_root, config, now, true).await
    }
    /// Opens only persisted-job execution; queue drain must not enqueue a new cohort.
    pub(crate) async fn prepare_ready(
        store: Arc<Store>,
        root: PathBuf,
        config_root: PathBuf,
        config: Config,
        now: i64,
    ) -> Result<Option<Self>, Error> {
        Self::prepare_mode(store, root, config_root, config, now, false).await
    }
    async fn prepare_mode(
        store: Arc<Store>,
        root: PathBuf,
        config_root: PathBuf,
        config: Config,
        now: i64,
        catalogue: bool,
    ) -> Result<Option<Self>, Error> {
        if !config.llm.enabled {
            return Ok(None);
        }
        tokio::task::spawn_blocking(move || {
            let execution = config
                .queue
                .execution()
                .map_err(|_| Error::Invalid("chat queue settings"))?;
            let client = chat::Client::from_config(
                &config.llm,
                execution.timeout,
                config
                    .limits
                    .max_image_bytes()
                    .map_err(|_| Error::Invalid("chat image cap"))?
                    .0,
            )
            .map_err(|_| Error::Invalid("chat client configuration or credentials"))?;
            let context = Arc::new(Context {
                store,
                root,
                config_root,
                config: Arc::new(config),
                client: Arc::new(client),
            });
            if catalogue {
                Self::from_context(context, now)
            } else {
                Self::from_context_mode(context, now, false)
            }
            .map(Some)
        })
        .await
        .map_err(|_| Error::WorkerTask)?
    }
    /// Captures selection and the existing scheduling rank/recency rules without HTTP.
    pub(crate) fn from_context(context: Arc<Context>, now: i64) -> Result<Self, Error> {
        Self::from_context_mode(context, now, true)
    }
    fn from_context_mode(context: Arc<Context>, now: i64, catalogue: bool) -> Result<Self, Error> {
        let bounds = host::Bounds::from_config(
            &context.config.queue,
            chat::limits(&context.config.llm)?.budget,
        )?;
        bounds.validate(now)?;
        let source = context.store.l1_identity()?;
        let head = sync::head::read(&context.root, Default::default())
            .map_err(|_| Error::Invalid("chat repository HEAD"))?
            .map(|head| head.id);
        let selected = if catalogue {
            context
                .store
                .read_snapshot(|tx| enrichment_catalogue::select(tx, &context.config))??
        } else {
            Vec::new()
        };
        let wanted = selected
            .iter()
            .map(|target| target.parent_unit)
            .collect::<BTreeSet<_>>();
        let mut ranks = BTreeMap::new();
        if !wanted.is_empty() {
            let mut cursor = catalogue::Cursor::start(&context.store)?;
            while !cursor.finished() && ranks.len() < wanted.len() {
                for candidate in cursor.next(&context.store, bounds.max_schedule)? {
                    if wanted.contains(&candidate.unit) {
                        ranks.insert(candidate.unit, (candidate.rank_pct, candidate.recency(now)));
                    }
                }
            }
        }
        let mut targets = Vec::new();
        for candidate in selected {
            if let Some((rank, recency)) = ranks.get(&candidate.parent_unit) {
                targets.push(Target {
                    kind: candidate.kind,
                    id: candidate.target_id,
                    priority: priority::calculate(candidate.kind, *rank, *recency)?,
                    referenced_image: false,
                    page: None,
                    source_hash: None,
                });
            }
        }
        if catalogue {
            let images = context.store.read_snapshot(|tx| {
                visual_source::select_images(
                    tx,
                    &context.root,
                    &context.config_root,
                    &context.config,
                    Default::default(),
                )
            })??;
            let at = targets
                .iter()
                .position(|target| matches!(target.kind, Kind::SummarizeModule))
                .unwrap_or(targets.len());
            let images = images
                .into_iter()
                .map(|candidate| {
                    Ok(Target {
                        kind: Kind::DescribeImage,
                        id: candidate.id,
                        priority: priority::calculate(
                            Kind::DescribeImage,
                            candidate.rank_pct,
                            priority::Recency::ChangedFile,
                        )?,
                        referenced_image: candidate.referenced,
                        page: None,
                        source_hash: None,
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            targets.splice(at..at, images);
            let pages = context.store.read_snapshot(|tx| {
                super::page_source::select(
                    tx,
                    &context.root,
                    &context.config_root,
                    &context.config,
                    Default::default(),
                )
            })??;
            let at = targets
                .iter()
                .position(|target| matches!(target.kind, Kind::SummarizeModule))
                .unwrap_or(targets.len());
            let pages = pages
                .into_iter()
                .map(|candidate| {
                    Ok(Target {
                        kind: Kind::VlmPage,
                        id: candidate.file,
                        page: Some(candidate.page),
                        source_hash: Some(candidate.source_hash),
                        referenced_image: false,
                        priority: priority::calculate(
                            Kind::VlmPage,
                            candidate.rank,
                            priority::Recency::ChangedFile,
                        )?,
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            targets.splice(at..at, pages);
        }
        let settings = Arc::new(pool::Settings {
            provider: chat::provider_id(context.client.endpoint()),
            concurrency: usize::try_from(context.config.llm.concurrency)
                .map_err(|_| Error::Invalid("chat concurrency"))?,
            max_jobs: bounds.max_jobs,
            max_claim_tokens: i64::MAX,
            budget: bounds.budget,
            max_output_tokens: 0,
            timeout: bounds.timeout,
            heartbeat: bounds.heartbeat,
        });
        let modules_from = targets
            .iter()
            .position(|target| matches!(target.kind, Kind::SummarizeModule))
            .unwrap_or(targets.len());
        let observed_summaries = context.store.summary_generation()?;
        let host = Self {
            context,
            settings,
            targets,
            position: 0,
            source,
            head,
            max_schedule: bounds.max_schedule,
            modules_from,
            observed_summaries,
        };
        host.check()?;
        Ok(host)
    }
    fn check(&self) -> Result<(), Error> {
        check_source(&self.context, &self.source, &self.head)
    }
    async fn check_async(&self) -> Result<(), Error> {
        let context = Arc::clone(&self.context);
        let source = self.source.clone();
        let head = self.head.clone();
        tokio::task::spawn_blocking(move || check_source(&context, &source, &head))
            .await
            .map_err(|_| Error::WorkerTask)?
    }
    /// True means the captured selection was visited, not that deferred jobs succeeded.
    pub(crate) fn catalogue_finished(&self) -> Result<bool, Error> {
        Ok(self.position == self.targets.len()
            && self.observed_summaries == self.context.store.summary_generation()?)
    }
    /// Bounds a public sweep to its initial cohorts, one module refresh and a ready pass.
    pub(crate) fn catalogue_pass_bound(&self) -> usize {
        let mut passes = 1usize;
        let mut start = 0;
        while start < self.targets.len() {
            let kind = std::mem::discriminant(&self.targets[start].kind);
            let end = start
                + self.targets[start..]
                    .iter()
                    .take_while(|target| {
                        std::mem::discriminant(&target.kind) == kind
                            && target.referenced_image == self.targets[start].referenced_image
                    })
                    .count();
            passes = passes.saturating_add((end - start).div_ceil(self.max_schedule));
            start = end;
        }
        passes.saturating_add((self.targets.len() - self.modules_from).div_ceil(self.max_schedule))
    }
    /// Identifies the shared endpoint used by durable usage and queue status.
    pub(crate) fn provider(&self) -> &str {
        &self.settings.provider
    }
    /// Reports deferred/leased backlog without treating terminal failures as success.
    pub(crate) fn pending(&self) -> Result<i64, Error> {
        Ok(self.context.store.read_snapshot(|tx| {
            tx.query_row(
            "SELECT count(*) FROM jobs WHERE provider=?1 AND state IN ('queued','leased','failed')",
            [&self.settings.provider],|r|r.get(0))
        })??)
    }
    /// A cancelled/failed scheduling page remains replayable; committed work is idempotent.
    pub(crate) async fn pass<C>(&mut self, clock: C) -> Result<Report, Error>
    where
        C: Fn() -> i64 + Send + Sync + 'static,
    {
        let stopped = Arc::new(AtomicBool::new(false));
        let _stop = Stop(Arc::clone(&stopped));
        let clock = Arc::new(clock);
        let generation = self.context.store.summary_generation()?;
        if self.position == self.targets.len() && generation != self.observed_summaries {
            self.position = self.modules_from;
        }
        self.observed_summaries = generation;
        let mut end = self
            .position
            .saturating_add(self.max_schedule)
            .min(self.targets.len());
        // Flush each cohort before dependent module inputs or unreferenced images.
        if let Some(first) = self.targets.get(self.position)
            && let Some(boundary) = self.targets[self.position..end].iter().position(|target| {
                std::mem::discriminant(&target.kind) != std::mem::discriminant(&first.kind)
                    || target.referenced_image != first.referenced_image
            })
        {
            end = self.position + boundary;
        }
        let targets = self.targets[self.position..end].to_vec();
        let context = Arc::clone(&self.context);
        let scheduled_clock = Arc::clone(&clock);
        let cancelled = Arc::clone(&stopped);
        self.check_async().await?;
        let mut report = tokio::task::spawn_blocking(move || {
            let work = sync::l1::Work {
                deadline: None,
                cancelled: Some(&cancelled),
            };
            let mut report = Report::default();
            for target in targets {
                super::source::check_work(work)?;
                report.visited += 1;
                let result = context
                    .store
                    .read_snapshot(|tx| match target.page {
                        Some(page) => context.prepare_page_with_work(tx, target.id, page, work),
                        None => context.prepare_job_with_work(tx, target.kind, target.id, work),
                    })
                    .map_err(Error::from)
                    .and_then(|result| result);
                let result = result.and_then(|job| {
                    job.map(|job| job.schedule_with_work(target.priority, scheduled_clock(), work))
                        .transpose()
                });
                match result {
                    Ok(Some(enrichment::Scheduled::Current(_))) => report.current += 1,
                    Ok(Some(enrichment::Scheduled::QueuedOrExisting(_))) => report.scheduled += 1,
                    Ok(Some(enrichment::Scheduled::Terminal(_))) => report.terminal += 1,
                    Ok(Some(enrichment::Scheduled::Paused(_))) => report.paused += 1,
                    Ok(_) => report.skipped += 1,
                    Err(error) if capacity::reason(&error).is_some() => report.paused += 1,
                    Err(Error::PagePreparation(
                        crate::treat::pdf::scan::Error::ImageBytes
                        | crate::treat::pdf::scan::Error::DocumentBytes
                        | crate::treat::pdf::scan::Error::PageLimit,
                    )) => report.paused += 1,
                    Err(Error::PagePreparation(crate::treat::pdf::scan::Error::NoRenderer)) => {
                        if let Some(hash)=target.source_hash {
                            super::storage::transaction(&context.store,super::storage::Kind::Control,|tx| {
                                tx.execute("UPDATE files SET l1_deferred_reason='no_renderer' WHERE id=?1 AND blake3=?2 AND needs_vlm=1 AND l1_pending=0 AND l1_fingerprint=l1_desired AND l1_deferred_reason IN ('needs_vlm','max_pdf_pages','no_renderer')",rusqlite::params![target.id,hash.as_slice()])?;
                                tx.commit()?;
                                Ok::<_,Error>(())
                            })?;
                        }
                        report.skipped += 1;
                    },
                    Err(Error::PagePreparation(_)) => report.skipped += 1,
                    Err(error) => {
                        report.error = Some(error);
                        break;
                    }
                }
            }
            Ok::<_, Error>(report)
        })
        .await
        .map_err(|_| Error::WorkerTask)??;
        if let Err(error) = self.check_async().await {
            report.error = Some(error);
            return Ok(report);
        };
        let store = Arc::clone(&self.context.store);
        let provider = self.settings.provider.clone();
        let recovery_clock = Arc::clone(&clock);
        tokio::task::spawn_blocking(move || {
            retry::recover(&store, &provider, || recovery_clock())?;
            capacity::resume(&store, &provider, || recovery_clock())
        })
        .await
        .map_err(|_| Error::WorkerTask)??;
        let worker_clock = Arc::clone(&clock);
        report.execution = pool::run_ready(
            Arc::clone(&self.context.store),
            Arc::clone(&self.settings),
            Arc::new(Arc::clone(&self.context)),
            move || worker_clock(),
        )
        .await?;
        // A summary can make an ancestor breadcrumb role stale. Reconcile
        // after the finite worker cohort drains so all affected FTS rows are
        // refreshed from current source proofs once, rather than once per
        // published output.
        let roles_pending = self.context.store.read_snapshot(|tx| {
            tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM meta WHERE key='lexical_role_reconcile_pending')",
                [],
                |row| row.get::<_, bool>(0),
            )
        })??;
        if roles_pending {
            let store = Arc::clone(&self.context.store);
            let root = self.context.root.clone();
            let config_root = self.context.config_root.clone();
            let config = Arc::clone(&self.context.config);
            tokio::task::spawn_blocking(move || {
                super::roles::reconcile(&store, &root, &config_root, &config, Default::default())
            })
            .await
            .map_err(|_| Error::WorkerTask)??;
        }
        if report.error.is_none() {
            self.position = end;
        }
        Ok(report)
    }
}

fn check_source(
    context: &Context,
    source: &crate::store::l1::Identity,
    expected_head: &Option<String>,
) -> Result<(), Error> {
    let head = sync::head::read(&context.root, Default::default())
        .map_err(|_| Error::Invalid("chat repository HEAD"))?
        .map(|head| head.id);
    if head != *expected_head
        || (source.scan != source.completed && !retained_catalogue_ready(context, source.scan)?)
        || sync::head::changed(source.indexed_head.as_deref(), head.as_deref())
        || context.store.l1_identity()? != *source
        || crate::config::load(&context.config_root, None)
            .map_err(|_| Error::Invalid("chat configuration reload"))?
            .config
            != *context.config
    {
        return Err(Error::InputsUnavailable);
    }
    Ok(())
}

// A bounded walk lacks repository-wide absence authority, but its completed
// admitted files can still supply independently verified enrichment inputs.
// Never grant this exception to an interrupted scan or stale diagnostic record.
fn retained_catalogue_ready(context: &Context, generation: i64) -> Result<bool, Error> {
    Ok(context.store.read_snapshot(|tx| {
        tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM meta WHERE key='discovery_status'
                AND json_extract(value,'$.generation')=?1
                AND json_extract(value,'$.max_files')=?2
                AND (json_extract(value,'$.max_files_reached')=1
                     OR json_extract(value,'$.skipped_non_unicode_paths')>0))
             AND EXISTS(SELECT 1 FROM meta WHERE key='graph_pending' AND value='0')
             AND NOT EXISTS(SELECT 1 FROM files WHERE l1_pending=1
                 OR (l1_fingerprint IS NOT l1_desired AND l1_deferred_reason IS NULL))",
            rusqlite::params![
                generation,
                context.config.limits.max_files.min(i64::MAX as u64) as i64
            ],
            |row| row.get(0),
        )
    })??)
}
