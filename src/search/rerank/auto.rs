//! First-full-call adaptive rerank bound and persisted status; v1 §§9, 11, 18.
use crate::{
    config::{self, Config},
    queue,
    store::{self, Store},
};
use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio::time::Instant;

const KEY: &str = "rerank_auto_v1";
const THRESHOLD: Duration = Duration::from_millis(1500);

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    fingerprint: String,
    elapsed_ns: u64,
    incomplete: bool,
}

fn fingerprint(config: &Config) -> Result<String, queue::Error> {
    let bytes = serde_json::to_vec(&serde_json::json!({
        "version":env!("CARGO_PKG_VERSION"), "os":std::env::consts::OS,"arch":std::env::consts::ARCH,
        "cuda":cfg!(feature="cuda"),"metal":cfg!(feature="metal"),"vulkan":cfg!(feature="vulkan"),
        "rerank":config.rerank,"profile":config.embedder.profile,
        "breadcrumb":config.embedder.breadcrumb,"roles":config.embedder.breadcrumb_roles,"kind_line":config.embedder.kind_line,
    })).map_err(|_|queue::Error::Invalid("rerank calibration identity"))?;
    let mut hash = blake3::Hasher::new_derive_key("dkc rerank auto configuration v1");
    hash.update(&bytes);
    Ok(hash.finalize().to_hex().to_string())
}
fn decode(value: Option<String>, key: &str) -> Option<Record> {
    value
        .and_then(|value| serde_json::from_str::<Record>(&value).ok())
        .filter(|record| {
            record.fingerprint == key && u128::from(record.elapsed_ns) > THRESHOLD.as_nanos()
        })
}

/// The effective bound is for subsequent candidate selection, never a post-hoc
/// privacy filter. Persistence state distinguishes a live decision from a saved one.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct Status {
    pub(crate) configured_candidates: u64,
    pub(crate) effective_candidates: u64,
    pub(crate) reduced: bool,
    pub(crate) observed_ms: Option<f64>,
    pub(crate) incomplete_sample: bool,
    pub(crate) persistence: &'static str,
    pub(crate) message: Option<&'static str>,
}
fn status(config: &Config, record: Option<&Record>, persistence: &'static str) -> Status {
    let reduced =
        config.rerank.mode == "auto" && config.rerank.candidates == 50 && record.is_some();
    Status {
        configured_candidates: config.rerank.candidates,
        effective_candidates: if reduced {
            20
        } else {
            config.rerank.candidates
        },
        reduced,
        observed_ms: record
            .filter(|_| reduced)
            .map(|record| record.elapsed_ns as f64 / 1_000_000.0),
        incomplete_sample: reduced && record.is_some_and(|record| record.incomplete),
        persistence: if reduced { persistence } else { "not_reduced" },
        message: reduced.then_some(
            "rerank auto: subsequent queries use 20 candidates after a slow 50-candidate call",
        ),
    }
}

/// Describes forced/off/custom bounds without unnecessary storage access.
pub(super) fn configured_status(config: &Config) -> Status {
    status(config, None, "not_reduced")
}

/// Reads matching cached calibration without loading a model or changing config.
pub(crate) fn read_status(store: &Store, config: &Config) -> Result<Status, queue::Error> {
    let key = fingerprint(config)?;
    let record = decode(store.get_meta(KEY)?, &key);
    Ok(status(config, record.as_ref(), "saved"))
}

/// One shared host's first eligible observation, with a bounded single metadata row.
pub(super) struct Control {
    store: Arc<Store>,
    root: PathBuf,
    config: Config,
    fingerprint: String,
    phase: AtomicU8, // 0 unmeasured, 1 observing, 2 fast, 3 reduced
    elapsed_ns: AtomicU64,
    incomplete: AtomicBool,
    persistence: AtomicU8, // 0 none, 1 pending, 2 saved, 3 deferred
}
impl Control {
    /// Restores only a matching saved reduction; no provider is loaded or called.
    pub(super) fn new(
        store: Arc<Store>,
        root: PathBuf,
        config: Config,
    ) -> Result<Arc<Self>, queue::Error> {
        let key = fingerprint(&config)?;
        let record = decode(store.get_meta(KEY)?, &key);
        let reduced =
            config.rerank.mode == "auto" && config.rerank.candidates == 50 && record.is_some();
        Ok(Arc::new(Self {
            store,
            root,
            config,
            fingerprint: key,
            phase: AtomicU8::new(if reduced { 3 } else { 0 }),
            elapsed_ns: AtomicU64::new(record.as_ref().map_or(0, |record| record.elapsed_ns)),
            incomplete: AtomicBool::new(record.is_some_and(|record| record.incomplete)),
            persistence: AtomicU8::new(if reduced { 2 } else { 0 }),
        }))
    }
    /// Takes a source-free status snapshot without database I/O.
    pub(super) fn snapshot(&self) -> Status {
        let record = (self.phase.load(Ordering::SeqCst) == 3).then(|| Record {
            fingerprint: self.fingerprint.clone(),
            elapsed_ns: self.elapsed_ns.load(Ordering::SeqCst),
            incomplete: self.incomplete.load(Ordering::SeqCst),
        });
        let persistence = match self.persistence.load(Ordering::SeqCst) {
            2 => "saved",
            3 => "deferred",
            _ => "pending",
        };
        status(&self.config, record.as_ref(), persistence)
    }
    /// Claims the first actual full-call sample, including its admission latency.
    pub(super) fn start(self: &Arc<Self>, count: usize, started: Instant) -> Option<Observation> {
        if self.config.rerank.mode != "auto"
            || self.config.rerank.candidates != 50
            || count != 50
            || self
                .phase
                .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
        {
            return None;
        }
        Some(Observation {
            control: Arc::clone(self),
            started,
            finished: false,
        })
    }
    fn observe(self: &Arc<Self>, elapsed: Duration, complete: bool) {
        if elapsed <= THRESHOLD {
            self.phase
                .store(if complete { 2 } else { 0 }, Ordering::SeqCst);
            return;
        }
        let record = Record {
            fingerprint: self.fingerprint.clone(),
            elapsed_ns: u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
            incomplete: !complete,
        };
        self.elapsed_ns.store(record.elapsed_ns, Ordering::SeqCst);
        self.incomplete.store(record.incomplete, Ordering::SeqCst);
        self.persistence.store(1, Ordering::SeqCst);
        self.phase.store(3, Ordering::SeqCst);
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            self.persistence.store(3, Ordering::SeqCst);
            return;
        };
        let control = Arc::clone(self);
        handle.spawn_blocking(move || {
            let saved = control.save(&record).unwrap_or(false);
            control
                .persistence
                .store(if saved { 2 } else { 3 }, Ordering::SeqCst);
        });
    }
    fn save(&self, record: &Record) -> Result<bool, queue::Error> {
        store::admission::transaction(&self.store, store::admission::Kind::Control, |tx| {
            let current = config::load(&self.root, None)
                .map_err(|_| queue::Error::Invalid("rerank calibration config reload"))?
                .config;
            if current != self.config {
                return Ok(false);
            }
            let old = tx
                .query_row("SELECT value FROM meta WHERE key=?1", [KEY], |row| {
                    row.get(0)
                })
                .optional()?;
            if decode(old, &self.fingerprint).is_none() {
                let value = serde_json::to_string(record)
                    .map_err(|_| queue::Error::Invalid("rerank calibration record"))?;
                tx.execute("INSERT INTO meta(key,value) VALUES(?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[KEY,&value])?;
            }
            tx.commit()?;
            Ok(true)
        })
    }
}

/// Dropping a slow incomplete call records a lower-bound latency observation;
/// cancellation before the threshold releases the sample for a later valid call.
pub(super) struct Observation {
    control: Arc<Control>,
    started: Instant,
    finished: bool,
}
impl Observation {
    /// Completes an observation; fast failures leave another sample available.
    pub(super) fn finish(mut self, complete: bool) {
        self.finished = true;
        self.control.observe(self.started.elapsed(), complete);
    }
}
impl Drop for Observation {
    fn drop(&mut self) {
        if !self.finished {
            self.control.observe(self.started.elapsed(), false);
        }
    }
}

#[cfg(test)]
mod tests;
