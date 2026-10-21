//! Queue inspection, explicit retry and bounded draining; see `v1.md` §8 and §10.2.

use anyhow::Result;
use clap::Subcommand;

#[derive(Subcommand)]
pub(super) enum QueueCommand {
    /// Runs already queued work within configured limits; does not index new work.
    Drain {
        /// Maximum ready-work passes per provider; delayed jobs remain queued.
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
        max_passes: u32,
        #[arg(long)]
        json: bool,
    },
    /// Shows persisted jobs, scheduling delays, and usage reservations.
    Status {
        /// Lists up to this many job IDs, with failed/dead jobs first.
        #[arg(long, value_parser = clap::value_parser!(u16).range(1..=1000))]
        jobs: Option<u16>,
        #[arg(long)]
        json: bool,
    },
    /// Requeues failed/dead jobs with attempts remaining; keeps delays and usage.
    Retry {
        /// Restricts the request to one persisted job ID.
        #[arg(long, value_parser = clap::value_parser!(i64).range(1..))]
        job: Option<i64>,
        #[arg(long)]
        json: bool,
    },
    /// Requeues one completed text summary for a fresh, fenced provider attempt.
    Regenerate {
        /// The completed describe-symbol, summarize-pr or summarize-module job ID.
        #[arg(long, value_parser = clap::value_parser!(i64).range(1..))]
        job: i64,
        #[arg(long)]
        json: bool,
    },
}

pub(super) fn run(command: QueueCommand) -> Result<()> {
    match command {
        QueueCommand::Drain { max_passes, json } => {
            super::queue_drain::run(max_passes, json)?;
        }
        QueueCommand::Status { jobs, json } => {
            let (root, settings) = super::workspace()?;
            let store = super::open_store(&root, &settings)?;
            let now = i64::try_from(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs(),
            )?;
            let snapshot = crate::queue::status::snapshot_with_jobs(&store, now, jobs)?;
            super::emit(json, serde_json::to_value(&snapshot)?, &snapshot.summary())?;
        }
        QueueCommand::Retry { job, json } => {
            let (root, settings) = super::workspace()?;
            let store = super::open_store(&root, &settings)?;
            let now = i64::try_from(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs(),
            )?;
            let requeued = match crate::queue::retry::request(&store, job, now) {
                Ok(count) => count,
                Err(error @ crate::queue::Error::BulkIncomplete { changed, .. }) => {
                    super::emit(
                        json,
                        serde_json::json!({"requeued": changed, "complete": false}),
                        &format!(
                            "Requeued {changed} jobs before interruption; remaining jobs can be retried."
                        ),
                    )?;
                    return Err(error.into());
                }
                Err(error) => return Err(error.into()),
            };
            super::emit(
                json,
                serde_json::json!({"requeued": requeued}),
                &format!(
                    "Requeued {requeued} jobs; attempts, delays and usage preserved. No provider calls made."
                ),
            )?;
        }
        QueueCommand::Regenerate { job, json } => {
            let (root, settings) = super::workspace()?;
            let store = super::open_store(&root, &settings)?;
            let now = i64::try_from(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs(),
            )?;
            let requeued = crate::queue::retry::request_regeneration(&store, job, now)?;
            super::emit(
                json,
                serde_json::json!({
                    "requeued": if requeued { 1 } else { 0 },
                    "output_retained": true,
                    "attempts_preserved": true,
                }),
                if requeued {
                    "Requeued 1 completed text summary for regeneration; its current output and all accounting are retained until fenced replacement."
                } else {
                    "No eligible completed text summary was requeued; current output and all accounting are retained."
                },
            )?;
        }
    }
    Ok(())
}
