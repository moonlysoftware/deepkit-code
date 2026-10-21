//! Opt-in hook commands; see v1.md §4.5 and §10.2.

use anyhow::Result;
use clap::Subcommand;

#[derive(Subcommand)]
pub(super) enum Command {
    /// Adds four Deepkit post-hooks, refusing existing or managed hooks.
    Install {
        #[arg(long)]
        json: bool,
    },
    /// Removes only unchanged hooks installed by Deepkit.
    Remove {
        #[arg(long)]
        json: bool,
    },
}

/// Dispatches explicit hook changes and emits the affected hook names.
pub(super) fn run(command: Command) -> Result<()> {
    let (install, json) = match command {
        Command::Install { json } => (true, json),
        Command::Remove { json } => (false, json),
    };
    let report = crate::sync::hooks::change(&super::current_dir()?, install)?;
    super::emit(json, serde_json::to_value(report)?, "Git hooks updated")?;
    Ok(())
}
