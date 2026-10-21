//! Recoverable real-content benchmark mutations; see `v1.md` §14 and plan §6.

use std::{
    fs,
    io::Write,
    path::{Component, Path, PathBuf},
    time::{Duration, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

const BACKUP: &str = ".deepkit/bench-edits";

#[derive(Serialize, Deserialize)]
struct Entry {
    path: String,
    original_hash: String,
    edited_hash: String,
    modified_seconds: u64,
    modified_nanos: u32,
}

/// Saves originals before editing a fixed-order sample; a later run can recover interrupted work.
pub(super) fn prepare(
    root: &Path,
    paths: &[PathBuf],
    max_bytes: u64,
    limit: usize,
) -> Result<Vec<String>> {
    let backup = root.join(BACKUP);
    let mut selected = Vec::new();
    for path in paths {
        if selected.len() == limit {
            break;
        }
        let original = fs::read(path)?;
        let Some(edited) = edit(path, &original) else {
            continue;
        };
        if edited.len() as u64 > max_bytes {
            continue;
        }
        let relative = path
            .strip_prefix(root)?
            .to_str()
            .context("non-UTF-8 benchmark path")?
            .to_owned();
        let modified = fs::metadata(path)?.modified()?.duration_since(UNIX_EPOCH)?;
        selected.push((
            Entry {
                path: relative,
                original_hash: blake3::hash(&original).to_hex().to_string(),
                edited_hash: blake3::hash(&edited).to_hex().to_string(),
                modified_seconds: modified.as_secs(),
                modified_nanos: modified.subsec_nanos(),
            },
            original,
            edited,
        ));
    }
    if selected.is_empty() {
        return Ok(Vec::new());
    }
    fs::create_dir(&backup).context("benchmark edit backup exists; recover it before editing")?;
    for (index, (_, original, _)) in selected.iter().enumerate() {
        durable_write(&backup.join(index.to_string()), original)?;
    }
    let entries = selected
        .iter()
        .map(|(entry, _, _)| entry)
        .collect::<Vec<_>>();
    durable_write(
        &backup.join("manifest.json"),
        &serde_json::to_vec(&entries)?,
    )?;
    let result = (|| {
        for (entry, _, edited) in &selected {
            let path = checked_path(root, &entry.path)?;
            if blake3::hash(&fs::read(&path)?).to_hex().as_str() != entry.original_hash {
                bail!("source changed before benchmark edit: {}", entry.path);
            }
            durable_write(&path, edited)?;
        }
        Ok(selected
            .iter()
            .map(|(entry, _, _)| entry.path.clone())
            .collect())
    })();
    if result.is_err() {
        restore(root)?;
    }
    result
}

/// Restores only originals or known benchmark edits, preserving conflicting user changes and backups.
pub(super) fn restore(root: &Path) -> Result<bool> {
    let backup = root.join(BACKUP);
    if !backup.exists() {
        return Ok(false);
    }
    let entries: Vec<Entry> = serde_json::from_slice(
        &fs::read(backup.join("manifest.json"))
            .context("incomplete benchmark backup; originals remain in .deepkit/bench-edits")?,
    )?;
    for (index, entry) in entries.iter().enumerate() {
        let path = checked_path(root, &entry.path)?;
        let original = fs::read(backup.join(index.to_string()))?;
        if blake3::hash(&original).to_hex().as_str() != entry.original_hash {
            bail!("benchmark backup hash mismatch for {}", entry.path);
        }
        let current = blake3::hash(&fs::read(&path)?).to_hex().to_string();
        if current != entry.original_hash && current != entry.edited_hash {
            bail!(
                "source changed during benchmark: {}; preserved source and backup in {}",
                entry.path,
                backup.display()
            );
        }
        if current != entry.original_hash {
            durable_write(&path, &original)?;
        }
        fs::File::open(path)?.set_times(fs::FileTimes::new().set_modified(
            UNIX_EPOCH + Duration::new(entry.modified_seconds, entry.modified_nanos),
        ))?;
    }
    fs::remove_dir_all(backup)?;
    Ok(true)
}

fn checked_path(root: &Path, relative: &str) -> Result<PathBuf> {
    let relative = Path::new(relative);
    if relative
        .components()
        .any(|part| !matches!(part, Component::Normal(_)))
    {
        bail!("invalid benchmark backup path");
    }
    let path = root.join(relative);
    if !fs::canonicalize(&path)?.starts_with(fs::canonicalize(root)?)
        || fs::symlink_metadata(&path)?.file_type().is_symlink()
    {
        bail!("benchmark source path escapes checkout or is a symlink");
    }
    Ok(path)
}

fn durable_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = fs::File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn edit(path: &Path, original: &[u8]) -> Option<Vec<u8>> {
    let text = std::str::from_utf8(original).ok()?;
    let extension = path.extension()?.to_str()?;
    let addition = match extension {
        "rs" => "\npub fn dkc_bench_edit_probe() -> u32 { 1 }\n",
        "php" => "\nfunction dkc_bench_edit_probe() { return 1; }\n",
        "py" => "\ndef dkc_bench_edit_probe():\n    return 1\n",
        "md" | "txt" => "\nBenchmark content edit probe.\n",
        _ => return None,
    };
    // PHP may close its code section; insert the function inside it.
    let offset = if extension == "php" {
        text.rfind("?>").unwrap_or(text.len())
    } else {
        text.len()
    };
    let mut edited = Vec::with_capacity(original.len() + addition.len());
    edited.extend_from_slice(&original[..offset]);
    edited.extend_from_slice(addition.as_bytes());
    edited.extend_from_slice(&original[offset..]);
    Some(edited)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abandoned_edits_restore_and_conflicts_keep_backups() -> Result<()> {
        let root = tempfile::tempdir()?;
        fs::create_dir(root.path().join(".deepkit"))?;
        let path = root.path().join("lib.rs");
        let original = b"pub fn original() {}\n";
        fs::write(&path, original)?;
        let modified = fs::metadata(&path)?.modified()?;
        assert_eq!(
            prepare(root.path(), std::slice::from_ref(&path), 1024, 20)?,
            ["lib.rs"]
        );
        assert_ne!(fs::read(&path)?, original);
        assert!(restore(root.path())?);
        assert_eq!(fs::read(&path)?, original);
        assert_eq!(fs::metadata(&path)?.modified()?, modified);
        prepare(root.path(), std::slice::from_ref(&path), 1024, 20)?;
        fs::write(&path, "user change")?;
        assert!(restore(root.path()).is_err());
        assert_eq!(fs::read_to_string(&path)?, "user change");
        assert!(root.path().join(BACKUP).join("0").exists());
        Ok(())
    }
}
