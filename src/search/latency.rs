//! Bounded process-local query latency shared by CLI and MCP; v1 §13.
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, VecDeque},
    sync::Mutex,
    time::{Duration, Instant},
};

const WINDOW: usize = 1000;

/// Numeric timings and closed cap labels; no query text or additional database writes.
#[derive(Default)]
pub(crate) struct History {
    samples: Mutex<VecDeque<f64>>,
    caps: Mutex<BTreeMap<&'static str, CapEvent>>,
}

/// A closed set keeps cap observations bounded and free of query/source text.
#[derive(Clone, Copy)]
pub(crate) enum ResponseCap {
    ResultTokens,
    QueryTimeout,
    RerankTimeout,
    OpenBytes,
}
impl ResponseCap {
    fn name(self) -> &'static str {
        match self {
            Self::ResultTokens => "max_result_tokens",
            Self::QueryTimeout => "query_timeout_ms",
            Self::RerankTimeout => "rerank_timeout_ms",
            Self::OpenBytes => "max_open_bytes",
        }
    }
}
#[derive(serde::Serialize)]
struct CapEvent {
    count: u64,
    last_tool: &'static str,
}

/// Records completed, failed and cancelled shared invocations exactly once.
pub(crate) struct Measurement<'a> {
    history: &'a History,
    started: Instant,
}

impl History {
    /// Begins timing at shared search admission, before retrieval or model work.
    pub(crate) fn measure(&self) -> Measurement<'_> {
        Measurement {
            history: self,
            started: Instant::now(),
        }
    }

    fn record(&self, elapsed: Duration) {
        let mut samples = self
            .samples
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if samples.len() == WINDOW {
            samples.pop_front();
        }
        samples.push_back(elapsed.as_secs_f64() * 1000.0);
    }

    /// Observes a cap in this serving context without consuming SQLite/WAL capacity.
    pub(crate) fn record_cap(&self, cap: ResponseCap, tool: &'static str) {
        let mut caps = self.caps.lock().unwrap_or_else(|error| error.into_inner());
        let event = caps.entry(cap.name()).or_insert(CapEvent {
            count: 0,
            last_tool: tool,
        });
        event.count = event.count.saturating_add(1);
        event.last_tool = tool;
    }
    /// Counters are process-local like latency history; a new CLI invocation starts empty.
    pub(crate) fn cap_snapshot(&self) -> Value {
        let caps = self.caps.lock().unwrap_or_else(|error| error.into_inner());
        json!({"lifetime":"process","events":&*caps})
    }

    /// Reports nearest-rank percentiles, with null values when this process has no samples.
    pub(crate) fn snapshot(&self) -> Value {
        let mut sorted = self
            .samples
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .copied()
            .collect::<Vec<_>>();
        sorted.sort_by(f64::total_cmp);
        let percentile = |percent: usize| {
            sorted
                .get((sorted.len() * percent).div_ceil(100).saturating_sub(1))
                .copied()
        };
        json!({
            "count": sorted.len(), "window": WINDOW, "lifetime": "process",
            "p50": percentile(50), "p95": percentile(95),
        })
    }
}

impl Drop for Measurement<'_> {
    fn drop(&mut self) {
        self.history.record(self.started.elapsed());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_window_uses_recent_samples_and_nearest_rank_percentiles() {
        let history = History::default();
        assert_eq!(
            history.snapshot(),
            json!({
                "count":0, "window":1000, "lifetime":"process", "p50":null, "p95":null,
            })
        );
        for millis in 1..=1100 {
            history.record(Duration::from_millis(millis));
        }
        assert_eq!(
            history.snapshot(),
            json!({
                "count":1000, "window":1000, "lifetime":"process", "p50":600.0, "p95":1050.0,
            })
        );
    }

    #[test]
    fn measurement_records_once_on_drop_without_inventing_zero_samples() {
        let history = History::default();
        let measurement = history.measure();
        assert_eq!(history.snapshot()["count"], 0);
        drop(measurement);
        let snapshot = history.snapshot();
        assert_eq!(snapshot["count"], 1);
        assert!(snapshot["p50"].as_f64().unwrap() >= 0.0);
        assert_eq!(snapshot["p50"], snapshot["p95"]);
    }
}
