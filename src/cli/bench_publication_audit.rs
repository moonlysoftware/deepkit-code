//! Opt-in read-only corpus publication evidence; v1 §§7.1, 9 and 14.
use crate::{
    config,
    queue::destination::{self, FileFreshness},
    store::embedding_freshness::State,
    sync::l1::Work,
};
use anyhow::{Result, ensure};
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::Path,
    time::{Duration, Instant},
};

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
#[ignore = "requires BENCH_PUBLICATION_ROOT and BENCH_PUBLICATION_OUT; no model calls"]
fn audit_existing_publications_without_models() -> Result<()> {
    let root = fs::canonicalize(std::env::var("BENCH_PUBLICATION_ROOT")?)?;
    let out = std::path::PathBuf::from(std::env::var("BENCH_PUBLICATION_OUT")?);
    ensure!(!out.exists(), "refuse to replace audit evidence");
    let settings = config::load(&root, None)?.config;
    let database = super::super::database_path(&root, &settings);
    ensure!(database.is_file(), "existing index required");
    let store = super::super::open_store(&root, &settings)?;
    let snapshot = store.index_snapshot()?;
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
        .map(|p| hash(p))
        .collect::<Result<Vec<_>>>()?;
    let ids: Vec<i64> = {
        let writer = store.writer()?;
        let mut statement = writer.prepare("SELECT id FROM units ORDER BY id")?;
        statement
            .query_map([], |row| row.get(0))?
            .collect::<std::result::Result<_, _>>()?
    };
    ensure!(
        !ids.is_empty() && ids.len() <= 100_000,
        "bounded nonempty corpus required"
    );
    let started = Instant::now();
    let work = Work {
        deadline: Some(started + Duration::from_secs(300)),
        cancelled: None,
    };
    let mut counts = BTreeMap::<&str, usize>::new();
    for id in &ids {
        let result = destination::file_freshness(&store, &root, &root, &settings, *id, work);
        let label = match result {
            Ok(Some(FileFreshness::Ineligible)) => "ineligible",
            Ok(Some(FileFreshness::Representation(State::Current))) => "current",
            Ok(Some(FileFreshness::Representation(State::Stale))) => "stale",
            Ok(Some(FileFreshness::Representation(State::Missing))) => "missing",
            Ok(None) => "source_refused",
            Err(_) => "inspection_error",
        };
        *counts.entry(label).or_default() += 1;
        if label == "inspection_error" {
            break;
        }
    }
    let after = tracked
        .iter()
        .map(|p| hash(p))
        .collect::<Result<Vec<_>>>()?;
    let unchanged = before == after
        && snapshot == store.index_snapshot()?
        && settings == config::load(&root, None)?.config;
    let complete = unchanged
        && counts.values().sum::<usize>() == ids.len()
        && !counts.contains_key("inspection_error");
    let report = json!({
        "state": if complete { "complete" } else { "incomplete" },
        "scope": "per-occurrence canonical source/policy and configured-space publication; no model calls, queue writes or vector scans",
        "root": root, "units": ids.len(), "counts": counts,
        "unchanged_database_config_head_and_catalogue": unchanged,
        "input_blake3": tracked.iter().zip(&before).map(|(p,h)| json!({"path":p,"blake3":h})).collect::<Vec<_>>(),
        "elapsed_ms": started.elapsed().as_millis(), "build": super::super::bench_metadata::build_info()?,
        "limitations": "Each observation uses the production source recheck. This is not an atomic filesystem snapshot or authorization for a future dispatch."
    });
    serde_json::to_writer_pretty(fs::File::create_new(out)?, &report)?;
    ensure!(complete, "audit incomplete; inspect retained report");
    Ok(())
}
