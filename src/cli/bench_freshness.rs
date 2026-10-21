//! Destructive-but-recoverable freshness probes for the Phase 4 benchmark; see `v1.md` §14.

use super::{duration_ms, git_stdout, invoke_dkc_json};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[cfg(test)]
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

const RECOVERY_PATH: &str = ".deepkit/bench-head.json";

#[derive(Debug, Serialize, Deserialize)]
struct Recovery {
    original_head: String,
    original_ref: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct Report {
    method: String,
    #[serde(default)]
    dirty_edit: Option<DirtyEdit>,
    #[serde(default)]
    distant_checkout: Option<DistantCheckout>,
    #[serde(default)]
    unavailable: BTreeMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DirtyEdit {
    path: String,
    elapsed_ms: f64,
    matched: bool,
    stale_or_resynced: bool,
    lazy_resync: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct DistantCheckout {
    original_head: String,
    distant_head: String,
    distance: usize,
    query: String,
    elapsed_ms: f64,
    hits: usize,
    invalid_source_ranges: usize,
    lazy_resync: Value,
}

/// Runs a dirty-source query and a 100-first-parent checkout query, restoring HEAD and index level.
pub(super) fn measure(
    root: &Path,
    eligible_paths: &[PathBuf],
    max_file_bytes: u64,
    restore_level: u8,
) -> Result<Report> {
    let dirty_edit = dirty_edit(root, eligible_paths, max_file_bytes, restore_level)?;
    let mut unavailable = BTreeMap::new();
    if dirty_edit.is_none() {
        unavailable.insert(
            "dirty_edit".to_owned(),
            "no admitted editable source contained a stable lexical probe within the file cap"
                .to_owned(),
        );
    }
    let distant_checkout = match distant_checkout(root, eligible_paths, restore_level)? {
        Some(report) => Some(report),
        None => {
            unavailable.insert(
                "distant_checkout".to_owned(),
                "repository has fewer than two first-parent commits with a searchable source"
                    .to_owned(),
            );
            None
        }
    };
    Ok(Report {
        method: "query-time-edit-and-first-parent-checkout-v1".to_owned(),
        dirty_edit,
        distant_checkout,
        unavailable,
    })
}

/// Restores a checkout left by an interrupted freshness probe before pin validation.
pub(super) fn recover(root: &Path) -> Result<bool> {
    let path = root.join(RECOVERY_PATH);
    if !path.exists() {
        return Ok(false);
    }
    let recovery: Recovery = serde_json::from_slice(
        &fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .context("invalid benchmark checkout recovery record")?;
    require_clean_tracked_worktree(root)?;
    restore_head(
        root,
        recovery.original_ref.as_deref(),
        &recovery.original_head,
    )?;
    remove_recovery(root)?;
    Ok(true)
}

fn search(root: &Path, query: &str) -> Result<(Value, std::time::Duration)> {
    let (response, elapsed, _) = invoke_dkc_json(
        root,
        &[
            "search".to_owned(),
            query.to_owned(),
            "-k".to_owned(),
            "10".to_owned(),
            "--scope".to_owned(),
            "source".to_owned(),
            "--mode".to_owned(),
            "lexical".to_owned(),
            "--no-rerank".to_owned(),
            "--json".to_owned(),
        ],
    )?;
    Ok((response, elapsed))
}

fn reindex(root: &Path, level: u8) -> Result<()> {
    invoke_dkc_json(
        root,
        &[
            "index".to_owned(),
            "--level".to_owned(),
            level.to_string(),
            "--json".to_owned(),
        ],
    )?;
    Ok(())
}

fn dirty_edit(
    root: &Path,
    eligible_paths: &[PathBuf],
    max_file_bytes: u64,
    restore_level: u8,
) -> Result<Option<DirtyEdit>> {
    let mut candidates = Vec::new();
    let mut queries = BTreeMap::new();
    for source in eligible_paths {
        let Some(query) = dirty_source_query(source)? else {
            continue;
        };
        let relative = source
            .strip_prefix(root)?
            .to_str()
            .context("non-UTF-8 freshness source path")?
            .to_owned();
        candidates.push(source.clone());
        queries.insert(relative, query);
    }
    let paths = super::super::bench_edits::prepare(root, &candidates, max_file_bytes, 1)?;
    let Some(path) = paths.first().cloned() else {
        return Ok(None);
    };
    let query = queries
        .get(&path)
        .context("freshness benchmark lost its lexical source probe")?;
    let measured = (|| {
        let (response, elapsed) = search(root, query)?;
        let hits = response["hits"]
            .as_array()
            .context("dirty-edit search omitted hits")?;
        let matched = hits
            .iter()
            .any(|hit| hit["path"].as_str() == Some(path.as_str()));
        let lazy_resync = response["index"]["lazy_resync"].clone();
        let stale = hits.iter().any(|hit| {
            hit["path"].as_str() == Some(path.as_str())
                && hit["fresh"]["state"].as_str() == Some("stale")
        });
        let published = lazy_resync["published_files"].as_u64().unwrap_or_default() > 0;
        if !matched || (!stale && !published) {
            bail!("dirty-edit search neither reported stale source nor resynced the edited file");
        }
        Ok(DirtyEdit {
            path: path.clone(),
            elapsed_ms: duration_ms(elapsed),
            matched,
            stale_or_resynced: true,
            lazy_resync,
        })
    })();
    let restored = super::super::bench_edits::restore(root)
        .context("failed to restore dirty-edit freshness source");
    let reindexed = restored.and_then(|_| reindex(root, restore_level));
    match (measured, reindexed) {
        (_, Err(error)) => Err(error.context("failed to recover after dirty-edit freshness probe")),
        (Err(error), Ok(())) => Err(error),
        (Ok(report), Ok(())) => Ok(Some(report)),
    }
}

fn dirty_source_query(path: &Path) -> Result<Option<String>> {
    if !matches!(
        path.extension().and_then(|value| value.to_str()),
        Some("rs" | "php" | "py" | "md" | "txt")
    ) {
        return Ok(None);
    }
    let bytes = fs::read(path)?;
    Ok(std::str::from_utf8(&bytes).ok().and_then(distinctive_token))
}

fn distant_checkout(
    root: &Path,
    eligible_paths: &[PathBuf],
    restore_level: u8,
) -> Result<Option<DistantCheckout>> {
    require_clean_tracked_worktree(root)?;
    let original_head = git_stdout(root, &["rev-parse", "HEAD"])?.trim().to_owned();
    let original_ref = symbolic_head(root)?;
    let revisions = git_stdout(
        root,
        &["rev-list", "--first-parent", "--max-count=101", "HEAD"],
    )?;
    let revisions = revisions.lines().collect::<Vec<_>>();
    let Some((distance, distant_head, query)) = distant_probe(root, &revisions, eligible_paths)?
    else {
        return Ok(None);
    };
    write_recovery(
        root,
        &Recovery {
            original_head: original_head.clone(),
            original_ref: original_ref.clone(),
        },
    )?;
    let measured = git(root, &["checkout", "--quiet", "--detach", &distant_head]).and_then(|()| {
        let (response, elapsed) = search(root, &query)?;
        let hits = response["hits"]
            .as_array()
            .context("distant-checkout search omitted hits")?;
        if hits.is_empty() {
            bail!("distant-checkout search returned no source evidence");
        }
        let invalid_source_ranges = hits.iter().filter(|hit| !valid_range(root, hit)).count();
        if invalid_source_ranges != 0 {
            bail!("distant-checkout search returned invalid source ranges");
        }
        Ok(DistantCheckout {
            original_head: original_head.clone(),
            distant_head: distant_head.to_owned(),
            distance,
            query: query.clone(),
            elapsed_ms: duration_ms(elapsed),
            hits: hits.len(),
            invalid_source_ranges,
            lazy_resync: response["index"]["lazy_resync"].clone(),
        })
    });
    let restored = restore_head(root, original_ref.as_deref(), &original_head)
        .and_then(|_| reindex(root, restore_level))
        .and_then(|_| remove_recovery(root));
    match (measured, restored) {
        (_, Err(error)) => Err(error.context("failed to recover after distant-checkout probe")),
        (Err(error), Ok(())) => Err(error),
        (Ok(report), Ok(())) => Ok(Some(report)),
    }
}

fn require_clean_tracked_worktree(root: &Path) -> Result<()> {
    if git_stdout(root, &["status", "--porcelain", "--untracked-files=no"])?
        .trim()
        .is_empty()
    {
        Ok(())
    } else {
        bail!("freshness checkout requires a clean tracked worktree")
    }
}

fn write_recovery(root: &Path, recovery: &Recovery) -> Result<()> {
    let path = root.join(RECOVERY_PATH);
    let parent = path
        .parent()
        .context("benchmark checkout recovery path has no parent")?;
    fs::create_dir_all(parent)?;
    if path.exists() {
        bail!(
            "benchmark checkout recovery already exists at {}",
            path.display()
        );
    }
    let temporary = path.with_extension("json.tmp");
    let mut file = fs::File::create(&temporary)?;
    file.write_all(&serde_json::to_vec(recovery)?)?;
    file.sync_all()?;
    fs::rename(&temporary, &path)?;
    Ok(())
}

fn remove_recovery(root: &Path) -> Result<()> {
    let path = root.join(RECOVERY_PATH);
    fs::remove_file(&path).with_context(|| format!("failed to remove {}", path.display()))
}

fn distant_probe(
    root: &Path,
    revisions: &[&str],
    eligible_paths: &[PathBuf],
) -> Result<Option<(usize, String, String)>> {
    for (distance, revision) in revisions.iter().enumerate().rev() {
        if distance == 0 {
            continue;
        }
        if let Some(query) = source_query(root, revision, eligible_paths)? {
            return Ok(Some((distance, (*revision).to_owned(), query)));
        }
    }
    Ok(None)
}

fn symbolic_head(root: &Path) -> Result<Option<String>> {
    let output = Command::new("git")
        .args(["symbolic-ref", "-q", "HEAD"])
        .current_dir(root)
        .output()?;
    if output.status.success() {
        return Ok(Some(String::from_utf8(output.stdout)?.trim().to_owned()));
    }
    if output.status.code() == Some(1) {
        return Ok(None);
    }
    bail!("git symbolic-ref failed during freshness benchmark")
}

fn restore_head(root: &Path, reference: Option<&str>, commit: &str) -> Result<()> {
    match reference {
        Some(reference) => {
            let branch = reference
                .strip_prefix("refs/heads/")
                .context("benchmark started on a non-branch symbolic HEAD")?;
            git(root, &["checkout", "--quiet", branch])?;
        }
        None => git(root, &["checkout", "--quiet", "--detach", commit])?,
    }
    let restored = git_stdout(root, &["rev-parse", "HEAD"])?;
    if restored.trim() != commit {
        bail!("freshness benchmark restored the wrong HEAD");
    }
    if symbolic_head(root)?.as_deref() != reference {
        bail!("freshness benchmark restored the wrong symbolic HEAD state");
    }
    Ok(())
}

fn git(root: &Path, arguments: &[&str]) -> Result<()> {
    let output = Command::new("git")
        .args(arguments)
        .current_dir(root)
        .output()?;
    if !output.status.success() {
        bail!(
            "git {} failed during freshness benchmark",
            arguments.join(" ")
        );
    }
    Ok(())
}

fn source_query(root: &Path, revision: &str, eligible_paths: &[PathBuf]) -> Result<Option<String>> {
    for path in eligible_paths.iter().take(1_000) {
        let relative = path
            .strip_prefix(root)?
            .to_str()
            .context("non-UTF-8 freshness source path")?;
        let object = format!("{revision}:{relative}");
        let output = Command::new("git")
            .args(["show", &object])
            .current_dir(root)
            .output()?;
        if !output.status.success() {
            continue;
        }
        let Ok(text) = std::str::from_utf8(&output.stdout) else {
            continue;
        };
        if let Some(token) = distinctive_token(text) {
            return Ok(Some(token));
        }
    }
    Ok(None)
}

fn distinctive_token(text: &str) -> Option<String> {
    text.split(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
        .filter(|token| {
            (8..=64).contains(&token.len())
                && token.as_bytes()[0].is_ascii_alphabetic()
                && token.bytes().any(|byte| byte.is_ascii_alphabetic())
        })
        .max_by_key(|token| token.len())
        .map(str::to_owned)
}

fn valid_range(root: &Path, hit: &Value) -> bool {
    let Some(path) = hit["path"].as_str() else {
        return false;
    };
    let Some(lines) = hit["lines"].as_array() else {
        return false;
    };
    let (Some(start), Some(end)) = (
        lines.first().and_then(Value::as_u64),
        lines.get(1).and_then(Value::as_u64),
    ) else {
        return false;
    };
    if start == 0 || end < start {
        return false;
    }
    let Ok(source) = fs::read_to_string(root.join(path)) else {
        return false;
    };
    let count = source.lines().count().max(1) as u64;
    end <= count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_ranges_require_existing_inclusive_lines() -> Result<()> {
        let root = tempfile::tempdir()?;
        fs::write(root.path().join("source.rs"), "one\ntwo\n")?;
        assert!(valid_range(
            root.path(),
            &json!({"path":"source.rs","lines":[1,2]})
        ));
        assert!(!valid_range(
            root.path(),
            &json!({"path":"source.rs","lines":[0,1]})
        ));
        assert!(!valid_range(
            root.path(),
            &json!({"path":"source.rs","lines":[2,3]})
        ));
        Ok(())
    }

    #[test]
    fn dirty_probe_uses_a_distinctive_token_already_in_the_index() -> Result<()> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("source.rs");
        fs::write(&path, "pub fn retry_limit() -> u32 { 4 }\n")?;
        assert_eq!(dirty_source_query(&path)?.as_deref(), Some("retry_limit"));
        Ok(())
    }

    #[test]
    fn interrupted_checkout_recovers_exact_branch_and_commit() -> Result<()> {
        let root = tempfile::tempdir()?;
        git(root.path(), &["init", "--quiet", "--initial-branch=main"])?;
        git(root.path(), &["config", "user.name", "Freshness Test"])?;
        git(
            root.path(),
            &["config", "user.email", "freshness@example.invalid"],
        )?;
        fs::write(root.path().join("source.rs"), "fn first() {}\n")?;
        git(root.path(), &["add", "source.rs"])?;
        git(root.path(), &["commit", "--quiet", "-m", "first"])?;
        let distant = git_stdout(root.path(), &["rev-parse", "HEAD"])?;
        fs::write(root.path().join("source.rs"), "fn second() {}\n")?;
        git(root.path(), &["add", "source.rs"])?;
        git(root.path(), &["commit", "--quiet", "-m", "second"])?;
        let original_head = git_stdout(root.path(), &["rev-parse", "HEAD"])?;
        let original_ref = symbolic_head(root.path())?;
        write_recovery(
            root.path(),
            &Recovery {
                original_head: original_head.trim().to_owned(),
                original_ref: original_ref.clone(),
            },
        )?;
        git(
            root.path(),
            &["checkout", "--quiet", "--detach", distant.trim()],
        )?;

        assert!(recover(root.path())?);
        assert_eq!(
            git_stdout(root.path(), &["rev-parse", "HEAD"])?.trim(),
            original_head.trim()
        );
        assert_eq!(symbolic_head(root.path())?, original_ref);
        assert!(!root.path().join(RECOVERY_PATH).exists());
        Ok(())
    }
}
