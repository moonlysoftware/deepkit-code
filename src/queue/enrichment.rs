//! Fenced text enrichment through the shared accounted chat client; v1 §6.8/§8.2.
use super::{
    CurrentInput, Enqueue, Error, Kind, Lease, Target, enrichment_inputs, pool, usage, worker,
};
use crate::{
    config::Config,
    models::{attempt::AttemptError, openai_compat::chat},
    store::Store,
};
use rusqlite::{Transaction, params};
use std::{path::PathBuf, sync::Arc};
mod input;
use input::Input;

/// Resources shared across one host generation, including one chat semaphore.
pub(crate) struct Context {
    pub(crate) store: Arc<Store>,
    pub(crate) root: PathBuf,
    pub(crate) config_root: PathBuf,
    pub(crate) config: Arc<Config>,
    pub(crate) client: Arc<chat::Client>,
}
/// Immutable provider request and verified inputs; no source-bearing Debug output.
#[derive(Clone)]
pub(crate) struct Job {
    context: Arc<Context>,
    kind: Kind,
    target: Target,
    target_id: i64,
    input: Arc<Input>,
    wire: Arc<chat::Prepared>,
    desired: [u8; 32],
    provider: String,
}
/// A scheduling observation distinguishes existing coverage from rejected inputs.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Scheduled {
    Current(i64),
    QueuedOrExisting(i64),
    Terminal(i64),
    Paused(i64),
    Superseded,
}
impl Context {
    /// Prepares exactly one verified PDF page without dispatching or reserving it.
    pub(crate) fn prepare_page_with_work(
        self: &Arc<Self>,
        tx: &Transaction<'_>,
        id: i64,
        page: u32,
        work: crate::sync::l1::Work<'_>,
    ) -> Result<Option<Job>, Error> {
        let Some(input) = super::page_source::prepare(
            tx,
            &self.root,
            &self.config_root,
            &self.config,
            id,
            page,
            work,
        )?
        else {
            return Ok(None);
        };
        let wire = self
            .client
            .prepare_image(
                &input.page.bytes,
                input.page.mime,
                &input.prompt,
                input.max_output_tokens,
            )
            .map_err(|_| Error::Invalid("page chat preparation"))?;
        if wire.fingerprint != input.wire_fingerprint {
            return Err(Error::Invalid("page client configuration changed"));
        }
        let desired = input.desired;
        Ok(Some(Job {
            context: Arc::clone(self),
            kind: Kind::VlmPage,
            target: Target::File,
            target_id: id,
            input: Arc::new(Input::Page(input)),
            wire: Arc::new(wire),
            desired,
            provider: chat::provider_id(self.client.endpoint()),
        }))
    }
    /// Prepares a selected symbol without network calls; None is a source/policy refusal.
    pub(crate) fn symbol(
        self: &Arc<Self>,
        tx: &Transaction<'_>,
        id: i64,
    ) -> Result<Option<Job>, Error> {
        self.prepare_job(tx, Kind::DescribeSymbol, id)
    }
    /// Prepares a supported text or image owner using its current source proof.
    pub(crate) fn prepare_job(
        self: &Arc<Self>,
        tx: &Transaction<'_>,
        kind: Kind,
        id: i64,
    ) -> Result<Option<Job>, Error> {
        self.prepare_job_with_work(tx, kind, id, crate::sync::l1::Work::default())
    }
    /// Prepares sources with the active host or worker cancellation fence.
    pub(crate) fn prepare_job_with_work(
        self: &Arc<Self>,
        tx: &Transaction<'_>,
        kind: Kind,
        id: i64,
        work: crate::sync::l1::Work<'_>,
    ) -> Result<Option<Job>, Error> {
        if matches!(kind, Kind::DescribeImage) {
            let Some(input) = super::visual_source::image(
                tx,
                &self.root,
                &self.config_root,
                &self.config,
                id,
                work,
            )?
            else {
                return Ok(None);
            };
            let wire = self
                .client
                .prepare_image(
                    &input.source.bytes,
                    input.mime,
                    &input.prompt,
                    input.max_output_tokens,
                )
                .map_err(|_| Error::Invalid("image chat preparation"))?;
            if wire.fingerprint != input.wire_fingerprint {
                return Err(Error::Invalid("image client configuration changed"));
            }
            let desired = input.desired;
            return Ok(Some(Job {
                context: Arc::clone(self),
                kind,
                target: Target::File,
                target_id: id,
                input: Arc::new(Input::Image(input)),
                wire: Arc::new(wire),
                desired,
                provider: chat::provider_id(self.client.endpoint()),
            }));
        }
        let (target, input) = match kind {
            Kind::DescribeSymbol => (
                Target::Symbol,
                enrichment_inputs::symbol_with_work(
                    tx,
                    &self.root,
                    &self.config_root,
                    &self.config,
                    id,
                    work,
                )?,
            ),
            Kind::SummarizePr => (
                Target::Pr,
                enrichment_inputs::pr_with_work(
                    tx,
                    &self.root,
                    &self.config_root,
                    &self.config,
                    id,
                    work,
                )?,
            ),
            Kind::SummarizeModule => (
                Target::Unit,
                enrichment_inputs::module_with_work(
                    tx,
                    &self.root,
                    &self.config_root,
                    &self.config,
                    id,
                    work,
                )?,
            ),
            _ => return Err(Error::Invalid("unsupported enrichment job")),
        };
        let Some(input) = input else {
            return Ok(None);
        };
        let maximum = crate::treat::prompts::output_tokens(kind)
            .map_err(|_| Error::Invalid("enrichment output ceiling"))?
            .ok_or(Error::Invalid("text output ceiling"))?;
        let wire = self
            .client
            .prepare_text(
                "Use only the supplied repository evidence.",
                &input.prompt,
                maximum,
            )
            .map_err(|_| Error::Invalid("enrichment chat preparation"))?;
        let desired = enrichment_inputs::generation(
            &input.fingerprint,
            &wire.fingerprint,
            self.client.endpoint(),
        );
        Ok(Some(Job {
            context: Arc::clone(self),
            kind,
            target,
            target_id: id,
            input: Arc::new(Input::Text(input)),
            wire: Arc::new(wire),
            desired,
            provider: chat::provider_id(self.client.endpoint()),
        }))
    }
}
impl Job {
    fn prepare_current(
        &self,
        tx: &Transaction<'_>,
        work: crate::sync::l1::Work<'_>,
    ) -> Result<Option<Job>, Error> {
        match self.input.page() {
            Some(page) => self
                .context
                .prepare_page_with_work(tx, self.target_id, page, work),
            None => self
                .context
                .prepare_job_with_work(tx, self.kind, self.target_id, work),
        }
    }
    /// Revalidates before scheduling and repairs completed generations whose
    /// output is absent, preserving attempts, charges and all active leases.
    pub(crate) fn schedule(&self, priority: f64, now: i64) -> Result<Scheduled, Error> {
        self.schedule_with_work(priority, now, crate::sync::l1::Work::default())
    }
    /// Fences late cancellation after blocking preparation and before queue writes commit.
    pub(crate) fn schedule_with_work(
        &self,
        priority: f64,
        now: i64,
        work: crate::sync::l1::Work<'_>,
    ) -> Result<Scheduled, Error> {
        super::storage::transaction(&self.context.store, super::storage::Kind::Growth, |tx| {
            super::source::check_work(work)?;
            let Some(current) = self.prepare_current(&tx, work)? else {
                return Ok(Scheduled::Superseded);
            };
            super::source::check_work(work)?;
            if current.desired != self.desired {
                return Ok(Scheduled::Superseded);
            }
            if let Some(id) = self.input.matching_summary(&tx, &self.desired)? {
                return Ok(Scheduled::Current(id));
            }
            let input = self.enqueue(priority);
            let required = super::capacity::required_output(&tx, &input)?;
            // Reopening missing output consumes coverage just like a new job.
            // Replacement refreshes keep their existing slot even at the cap.
            if super::capacity::input_growth(&tx, &input, self.input.page(), required)? > 0 {
                self.context.store.enrichment_status_in(&tx)?.admit_new()?;
            }
            let paused = self.input.file_capacity(
                &tx,
                self.context.config.limits.max_units_per_file,
                required,
            )?;
            let id = super::enqueue_in(&tx, &input, now, paused)?;
            if let Some(page) = self.input.page() {
                super::page_source::bind(&tx, id, page)?;
            }
            tx.execute(
                "UPDATE jobs SET state='queued',not_before=?3,last_error=?4,updated_at=?2
                WHERE id=?1 AND state='done' AND attempts<8",
                params![
                    id,
                    now,
                    paused.map_or(now, |_| super::capacity::PAUSED),
                    paused.map(|reason| reason.message())
                ],
            )?;
            let terminal: bool = tx.query_row(
                "SELECT state='dead' OR (attempts>=8 AND (state!='leased' OR leased_until<=?2)) FROM jobs WHERE id=?1",
                params![id,now], |r|r.get(0),
            )?;
            super::source::check_work(work)?;
            tx.commit()?;
            Ok(if terminal {
                Scheduled::Terminal(id)
            } else if paused.is_some() {
                Scheduled::Paused(id)
            } else {
                Scheduled::QueuedOrExisting(id)
            })
        })
    }
    /// Schedules the exact prepared input estimate; worker admission adds its output ceiling.
    pub(crate) fn enqueue(&self, priority: f64) -> Enqueue<'_> {
        Enqueue {
            kind: self.kind,
            target: self.target,
            target_id: self.target_id,
            target_hash: self.input.target_hash(),
            desired_hash: &self.desired,
            provider: &self.provider,
            priority,
            tokens: self.wire.input_tokens,
        }
    }
}
impl pool::Factory for Arc<Context> {
    type Job = Job;
    fn prepare(&self, store: &Store, lease: &Lease) -> Result<Job, Error> {
        self.prepare_with_work(store, lease, crate::sync::l1::Work::default())
    }
    fn prepare_with_work(
        &self,
        store: &Store,
        lease: &Lease,
        work: crate::sync::l1::Work<'_>,
    ) -> Result<Job, Error> {
        super::source::check_work(work)?;
        store
            .read_snapshot(|tx| {
                let (kind, target, id, page): (String, String, i64, Option<u32>) = tx.query_row(
                    "SELECT kind,target_kind,target_id,page FROM jobs WHERE id=?1",
                    [lease.id],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
                )?;
                if kind == "enrich:vlm_page" && target == "file" {
                    let page = page
                        .filter(|page| *page > 0)
                        .ok_or(Error::Invalid("page job ordinal missing"))?;
                    return self
                        .prepare_page_with_work(tx, id, page, work)?
                        .ok_or(Error::InputsUnavailable);
                }
                let kind = match (kind.as_str(), target.as_str()) {
                    ("enrich:describe_symbol", "symbol") => Kind::DescribeSymbol,
                    ("enrich:summarize_pr", "pr") => Kind::SummarizePr,
                    ("enrich:summarize_module", "unit") => Kind::SummarizeModule,
                    ("enrich:describe_image", "file") => Kind::DescribeImage,
                    _ => return Err(Error::Invalid("unsupported enrichment job")),
                };
                self.prepare_job_with_work(tx, kind, id, work)?
                    .ok_or(Error::InputsUnavailable)
            })?
            .map_err(|error| match error {
                Error::Invalid(_) => Error::InputPreparation,
                error => error,
            })
    }
}
#[async_trait::async_trait]
impl worker::Work for Job {
    type Output = String;
    fn current(&self, tx: &Transaction<'_>, lease: &Lease) -> Result<Option<CurrentInput>, Error> {
        self.current_with_work(tx, lease, crate::sync::l1::Work::default())
    }
    fn current_with_work(
        &self,
        tx: &Transaction<'_>,
        lease: &Lease,
        work: crate::sync::l1::Work<'_>,
    ) -> Result<Option<CurrentInput>, Error> {
        let now: i64 = tx.query_row("SELECT unixepoch()", [], |r| r.get(0))?;
        self.current_at(tx, lease, work, now)
    }
    fn current_at(
        &self,
        tx: &Transaction<'_>,
        lease: &Lease,
        work: crate::sync::l1::Work<'_>,
        now: i64,
    ) -> Result<Option<CurrentInput>, Error> {
        let Some(current) = self.prepare_current(tx, work)? else {
            return Ok(None);
        };
        if current.desired != self.desired
            || current.input.fingerprint() != self.input.fingerprint()
        {
            return Ok(None);
        }
        let required: u32 = tx.query_row(
            "SELECT required_l3_units FROM jobs WHERE id=?1",
            [lease.id],
            |r| r.get(0),
        )?;
        if self.input.page().is_some() {
            let growth =
                super::capacity::input_growth(tx, &self.enqueue(0.0), self.input.page(), required)?;
            if growth > 0 {
                let status = self.context.store.enrichment_status_in(tx)?;
                status.admit_new()?;
                if status.l3_units.saturating_add(growth) > status.max_l3_units {
                    return Err(Error::Capacity(super::capacity::Reason::Summaries));
                }
            }
            if growth > 0
                && super::capacity::enrichment_claim_status(&self.context.store, tx, now)?
                    .exceeds_limit()
            {
                return Err(Error::Capacity(super::capacity::Reason::Summaries));
            }
        }
        if let Some(reason) = current.input.file_capacity(
            tx,
            self.context.config.limits.max_units_per_file,
            required,
        )? {
            return Err(Error::Capacity(reason));
        }
        Ok(Some(CurrentInput {
            target_hash: *self.input.target_hash(),
            desired_hash: self.desired,
        }))
    }
    fn limits(&self, _: &Transaction<'_>, provider: &str) -> Result<usage::Limits, Error> {
        if provider != self.provider {
            return Err(Error::Invalid("enrichment provider identity"));
        }
        chat::limits(&self.context.config.llm)
    }
    fn outbound(&self) -> Result<Option<usage::Endpoint>, Error> {
        Ok(Some(usage::Endpoint {
            kind: "llm",
            url: self.context.client.endpoint().to_owned(),
            units: 1,
        }))
    }
    fn max_output_tokens(&self) -> Option<i64> {
        Some(self.wire.max_output_tokens)
    }
    async fn call(&self) -> Result<worker::Response<String>, AttemptError> {
        let reply = self.context.client.post_once(&self.wire).await?;
        if matches!(self.kind, Kind::VlmPage) {
            reply.require_complete()?;
        }
        Ok(worker::Response {
            output: reply.completion.content,
            tokens: reply.tokens,
        })
    }
    fn publish(&self, tx: &Transaction<'_>, output: String) -> Result<(), Error> {
        self.publish_with_work(tx, output, Default::default())
    }
    fn publish_with_work(
        &self,
        tx: &Transaction<'_>,
        output: String,
        work: crate::sync::l1::Work<'_>,
    ) -> Result<(), Error> {
        super::source::check_work(work)?;
        let now = tx.query_row("SELECT unixepoch()", [], |r| r.get(0))?;
        let published = self
            .input
            .publish(&self.context, tx, &self.desired, &output, now)?;
        // Every summary can make a breadcrumb role stale. Keep its own derived
        // row and the completion fence atomic, then let the finite host cohort
        // refresh affected FTS scopes once. Replaying a large module scope in
        // this provider-response transaction can exceed its WAL allowance after
        // an otherwise successful remote call.
        if !published.is_empty() {
            tx.execute(
                "INSERT INTO meta(key,value) VALUES('lexical_role_reconcile_pending','1') ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                [],
            )?;
        }
        for unit in published {
            super::roles::refresh_derived_unit(
                tx,
                &self.context.root,
                &self.context.config_root,
                &self.context.config,
                unit,
                work,
            )?;
        }
        // Lexical replay can outlive the worker's original completion check.
        // Re-prove every source route before committing summary/page output.
        let current = self.prepare_current(tx, work)?.ok_or(Error::Invalid(
            "enrichment inputs changed during role refresh",
        ))?;
        if current.desired != self.desired
            || current.input.fingerprint() != self.input.fingerprint()
        {
            return Err(Error::Invalid(
                "enrichment inputs changed during role refresh",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod image_public_tests;
#[cfg(test)]
mod image_tests;

#[cfg(test)]
mod page_tests;

#[cfg(test)]
mod page_public_tests;
#[cfg(test)]
mod tests;
