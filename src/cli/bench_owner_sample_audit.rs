//! Opt-in current-summary checks for a preselected human-review cohort.
use crate::{config, queue::summary_source, sync::l1::Work};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, io::Read, path::Path};

fn hash(path: &Path) -> Result<Option<String>> {
    let mut file = match fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0; 65536];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(Some(hasher.finalize().to_hex().to_string()))
}

#[test]
#[ignore = "requires BENCH_OWNER_SAMPLE_ROOT, BENCH_OWNER_SAMPLE_COHORT and BENCH_OWNER_SAMPLE_OUT; no model calls"]
fn audit_current_review_samples_without_models() -> Result<()> {
    let root = fs::canonicalize(std::env::var("BENCH_OWNER_SAMPLE_ROOT")?)?;
    let cohort = std::path::PathBuf::from(std::env::var("BENCH_OWNER_SAMPLE_COHORT")?);
    let out = std::path::PathBuf::from(std::env::var("BENCH_OWNER_SAMPLE_OUT")?);
    ensure!(!out.exists(), "refuse to replace audit evidence");
    let frozen: Value = serde_json::from_slice(&fs::read(&cohort)?)?;
    let samples = frozen["review_samples"]
        .as_array()
        .context("review_samples must be an array")?;
    ensure!(
        samples.len() == 20,
        "exactly twenty fixed review samples required"
    );
    let mut expected = BTreeMap::new();
    for sample in samples {
        let kind = sample["kind"]
            .as_str()
            .context("review sample kind must be a string")?;
        let owner = sample["owner_id"]
            .as_i64()
            .context("review sample owner_id must be an integer")?;
        let parent = sample["parent_unit"]
            .as_i64()
            .context("review sample parent_unit must be an integer")?;
        ensure!(
            matches!(kind, "symbol" | "pr" | "module"),
            "unsupported sample kind"
        );
        ensure!(
            expected.insert((kind.to_owned(), owner), parent).is_none(),
            "duplicate review owner"
        );
    }
    let expected_counts = expected
        .keys()
        .fold(BTreeMap::new(), |mut counts, (kind, _)| {
            *counts.entry(kind.clone()).or_insert(0usize) += 1;
            counts
        });
    ensure!(
        expected_counts
            == BTreeMap::from([
                ("module".to_owned(), 6usize),
                ("pr".to_owned(), 6usize),
                ("symbol".to_owned(), 8usize),
            ])
    );

    let settings = config::load(&root, None)?.config;
    let database = super::super::database_path(&root, &settings);
    ensure!(database.is_file(), "existing index required");
    let store = super::super::open_store(&root, &settings)?;
    let mut wal = database.as_os_str().to_os_string();
    wal.push("-wal");
    let tracked = [
        database.clone(),
        wal.into(),
        root.join(".deepkit/config.toml"),
        root.join(".git/HEAD"),
    ];
    let before = tracked
        .iter()
        .map(|path| hash(path))
        .collect::<Result<Vec<_>>>()?;
    let snapshot = store.index_snapshot()?;
    let mut counts = BTreeMap::<String, usize>::new();
    let mut receipts = Vec::new();
    for ((kind, owner), parent) in expected {
        let result = store.read_snapshot(|tx| -> Result<(i64, bool)> {
            let (summary, attachment): (i64, summary_source::Attachment) = match kind.as_str() {
                "symbol" => {
                    let summary = tx.query_row(
                        "SELECT s.summary_unit FROM symbols s JOIN units u ON u.id=s.summary_unit
                         WHERE s.id=?1 AND s.unit_id=?2 AND u.kind='summary'
                         AND u.parent_unit=s.unit_id AND u.source_key='summary:symbol:'||s.id",
                        [owner, parent],
                        |row| row.get(0),
                    )?;
                    (summary, summary_source::Attachment::Symbol(owner))
                }
                "pr" => {
                    let summary = tx.query_row(
                        "SELECT p.summary_unit FROM prs p JOIN units u ON u.id=p.summary_unit
                         WHERE p.id=?1 AND p.unit_id=?2 AND u.kind='summary'
                         AND u.parent_unit=p.unit_id AND u.source_key='summary:pr:'||p.id",
                        [owner, parent],
                        |row| row.get(0),
                    )?;
                    (summary, summary_source::Attachment::Parent(parent))
                }
                "module" => {
                    let summary = tx.query_row(
                        "SELECT d.summary_unit FROM dirs d JOIN units u ON u.id=d.summary_unit
                         WHERE d.module_unit=?1 AND d.module_unit=?2 AND u.kind='summary'
                         AND u.parent_unit=d.module_unit AND u.source_key='summary:module:'||d.module_unit",
                        [owner, parent],
                        |row| row.get(0),
                    )?;
                    (summary, summary_source::Attachment::Parent(parent))
                }
                _ => unreachable!("validated review sample kind"),
            };
            let observed = summary_source::attached(tx, &root, &root, &settings, attachment, Work::default())?;
            Ok((summary, observed.is_some_and(|input| input.occurrence.id == summary)))
        })??;
        ensure!(result.1, "stored review summary is not current");
        *counts.entry(kind.clone()).or_default() += 1;
        receipts.push(json!({
            "kind": kind,
            "owner_id": owner,
            "parent_unit": parent,
            "summary_unit": result.0,
            "state": "current_matching",
        }));
    }
    let after = tracked
        .iter()
        .map(|path| hash(path))
        .collect::<Result<Vec<_>>>()?;
    let unchanged = before == after
        && snapshot == store.index_snapshot()?
        && settings == config::load(&root, None)?.config;
    ensure!(
        unchanged,
        "audit changed source, configuration or catalogue state"
    );
    serde_json::to_writer_pretty(
        fs::File::create_new(out)?,
        &json!({
            "state": "complete",
            "scope": "fixed preselected owner summaries revalidated through production canonical source, prompt and owner-content checks; no model calls, queue writes or network",
            "root": root,
            "cohort": cohort,
            "samples": receipts,
            "counts": counts,
            "unchanged_database_config_head_and_catalogue": unchanged,
            "input_blake3": tracked.iter().zip(&before).map(|(path, digest)| json!({"path": path, "blake3": digest})).collect::<Vec<_>>(),
            "build": super::super::bench_metadata::build_info()?,
        }),
    )?;
    Ok(())
}
