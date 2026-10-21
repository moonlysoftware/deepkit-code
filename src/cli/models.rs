//! Model cache inspection commands; see `v1.md` §7.4 and §10.2.

use crate::models::registry::{self, Registry};
use anyhow::Result;
use clap::Subcommand;
use serde_json::json;

#[derive(Subcommand)]
pub(super) enum ModelsCommand {
    /// Lists pinned model artifacts and unverified cache presence.
    List {
        #[arg(long)]
        json: bool,
    },
    /// Downloads a pinned model with resume, verification and cache-cap admission.
    Pull {
        name: String,
        #[arg(long)]
        json: bool,
    },
    /// Removes unreferenced registered artifacts; preserves configured models and unknown files.
    Prune {
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        json: bool,
    },
    /// Checks a cached model's complete SHA-256 digest and byte size.
    Verify {
        name: String,
        #[arg(long)]
        json: bool,
    },
}

pub(super) fn run(command: ModelsCommand) -> Result<()> {
    let registry = Registry::embedded()?;
    let cache = registry::cache_root()?;
    match command {
        ModelsCommand::List { json } => {
            let mut rows = Vec::new();
            let mut lines = Vec::new();
            for model in &registry.models {
                let path = cache.join(&model.file);
                let present = path.try_exists()?;
                let status = if present {
                    "cached_unverified"
                } else {
                    "missing"
                };
                rows.push(json!({ "model": model, "path": path, "status": status }));
                lines.push(format!("{}: {} ({} bytes)", model.name, status, model.size));
            }
            super::emit(
                json,
                json!({ "cache": cache, "models": rows }),
                &lines.join("\n"),
            )?;
        }
        ModelsCommand::Pull { name, json } => {
            let model = registry.get(&name)?.clone();
            let config = crate::config::load(&std::env::current_dir()?, None)?.config;
            let cap = config.limits.models_cache_max_bytes()?.0;
            if !json {
                eprintln!(
                    "Pulling {} ({} bytes; existing partial downloads will resume)",
                    name, model.size
                );
            }
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            let path = runtime.block_on(crate::models::download::pull(model, cache, cap))?;
            super::emit(
                json,
                json!({ "name": name, "path": path, "verified": true }),
                &format!("{name}: cached and verified"),
            )?;
        }
        ModelsCommand::Prune { dry_run, json } => {
            let config = crate::config::load(&std::env::current_dir()?, None)?.config;
            // Retain explicitly named local models even while a different profile is active.
            let referenced = [config.embedder.local.model, config.rerank.local.model]
                .into_iter()
                .collect();
            let report = crate::models::download::prune(&registry, &cache, &referenced, dry_run)?;
            let action = if dry_run { "Would remove" } else { "Removed" };
            let mut lines = vec![format!(
                "{action} {} files ({} bytes); retained {} configured and {} unrecognized files",
                report.candidates.len(),
                report.reclaimable_bytes,
                report.retained.len(),
                report.unmanaged.len()
            )];
            lines.extend(
                report
                    .candidates
                    .iter()
                    .map(|path| path.display().to_string()),
            );
            super::emit(json, serde_json::to_value(report)?, &lines.join("\n"))?;
        }
        ModelsCommand::Verify { name, json } => {
            let model = registry.get(&name)?;
            registry::verify(model, &cache)?;
            super::emit(
                json,
                json!({ "name": name, "path": cache.join(&model.file), "verified": true, "sha256": model.sha256, "size": model.size }),
                &format!("{name}: verified SHA-256 and {} bytes", model.size),
            )?;
        }
    }
    Ok(())
}
