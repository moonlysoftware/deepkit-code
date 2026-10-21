//! Stdio MCP round-trip measurement for the Phase 4 benchmark; see `v1.md` §14.

use super::{LatencyMetrics, duration_ms, summarize_latencies};
use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    env,
    io::{BufRead, BufReader, Write},
    path::Path,
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc::{Receiver, RecvTimeoutError, channel},
    thread,
    time::{Duration, Instant},
};

const RESPONSE_TIMEOUT: Duration = Duration::from_secs(30);
const CLEANUP_TIMEOUT: Duration = Duration::from_secs(2);
const CLEANUP_POLL: Duration = Duration::from_millis(10);

/// One initialized server plus exactly twenty measured search calls.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct Report {
    transport: String,
    startup_ms: f64,
    cold_start_ms: f64,
    median_overhead_ms: Option<f64>,
    pub(super) search: LatencyMetrics,
}

struct Client {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Receiver<std::result::Result<String, String>>,
    reader: Option<thread::JoinHandle<()>>,
}

impl Client {
    fn start(root: &Path) -> Result<(Self, Duration)> {
        let executable = env::current_exe().context("failed to locate the dkc executable")?;
        let started = Instant::now();
        let mut child = Command::new(executable)
            .args(["bench", "mcp-server"])
            .current_dir(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("failed to start MCP benchmark in {}", root.display()))?;
        let stdin = child.stdin.take().context("MCP child stdin unavailable")?;
        let stdout = child
            .stdout
            .take()
            .context("MCP child stdout unavailable")?;
        let (sender, lines) = channel();
        let reader = thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let terminal = line.is_err();
                let message = line.map_err(|error| error.to_string());
                if sender.send(message).is_err() || terminal {
                    break;
                }
            }
        });
        let mut client = Self {
            child,
            stdin: Some(stdin),
            lines,
            reader: Some(reader),
        };
        client.request(
            1,
            "initialize",
            json!({
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "dkc-bench", "version": env!("CARGO_PKG_VERSION")}
            }),
        )?;
        client.notify("notifications/initialized", json!({}))?;
        Ok((client, started.elapsed()))
    }

    fn write(&mut self, message: &Value) -> Result<()> {
        let stdin = self.stdin.as_mut().context("MCP benchmark stdin closed")?;
        serde_json::to_writer(&mut *stdin, message)?;
        stdin.write_all(b"\n")?;
        stdin.flush()?;
        Ok(())
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        self.write(&json!({"jsonrpc": "2.0", "method": method, "params": params}))
    }

    fn request(&mut self, id: u64, method: &str, params: Value) -> Result<Value> {
        self.write(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }))?;
        let deadline = Instant::now() + RESPONSE_TIMEOUT;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let line = match self.lines.recv_timeout(remaining) {
                Ok(Ok(line)) => line,
                Ok(Err(error)) => bail!("MCP benchmark output failed: {error}"),
                Err(RecvTimeoutError::Timeout) => bail!("MCP benchmark response timed out"),
                Err(RecvTimeoutError::Disconnected) => {
                    bail!("MCP benchmark server closed its output")
                }
            };
            let frame: Value = serde_json::from_str(&line)
                .context("MCP benchmark server returned a non-JSON frame")?;
            if frame.get("id").and_then(Value::as_u64) != Some(id) {
                continue;
            }
            if frame.get("error").is_some() {
                bail!("MCP benchmark request {method} returned a JSON-RPC error");
            }
            return frame
                .get("result")
                .cloned()
                .context("MCP benchmark response omitted its result");
        }
    }

    fn finish(&mut self) -> Result<()> {
        // Closing stdin first lets a responsive server shut down. A faulted
        // child or inherited output pipe must not hold the benchmark forever.
        self.stdin.take();
        if self.child.try_wait()?.is_none()
            && let Err(error) = self.child.kill()
            && self.child.try_wait()?.is_none()
        {
            return Err(error).context("failed to stop MCP benchmark server");
        }
        let deadline = Instant::now() + CLEANUP_TIMEOUT;
        while self.child.try_wait()?.is_none() {
            if Instant::now() >= deadline {
                bail!("MCP benchmark server did not exit after termination");
            }
            thread::sleep(CLEANUP_POLL);
        }
        if let Some(reader) = self.reader.take() {
            while !reader.is_finished() {
                if Instant::now() >= deadline {
                    bail!("MCP benchmark output pipe did not close after server exit");
                }
                thread::sleep(CLEANUP_POLL);
            }
            reader
                .join()
                .map_err(|_| anyhow!("MCP benchmark output reader panicked"))?;
        }
        Ok(())
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

/// Measures transport plus server handling while excluding process initialization.
pub(super) fn measure(
    root: &Path,
    queries: &[&str],
    in_process_p50_ms: Option<f64>,
) -> Result<Report> {
    if queries.is_empty() {
        bail!("MCP benchmark requires at least one query");
    }
    let (mut client, startup) = Client::start(root)?;
    let mut samples = Vec::with_capacity(20);
    for index in 0..20_u64 {
        let started = Instant::now();
        let result = client.request(
            index + 2,
            "tools/call",
            json!({
                "name": "search",
                "arguments": {
                    "query": queries[index as usize % queries.len()],
                    "scope": "source",
                    "k": 10,
                    "mode": "lexical",
                    "rerank": false
                }
            }),
        )?;
        if result["isError"].as_bool() == Some(true) {
            bail!("MCP search benchmark returned a tool error");
        }
        std::hint::black_box(serde_json::to_vec(&result)?);
        samples.push(duration_ms(started.elapsed()));
    }
    client.finish()?;
    let mut search = summarize_latencies(&samples);
    search.cold_ms = samples.first().copied();
    search.method = "mcp-stdio-round-trip-no-workers-20-calls-v2".to_owned();
    let startup_ms = duration_ms(startup);
    let cold_start_ms = startup_ms + samples.first().copied().unwrap_or_default();
    let median_overhead_ms = search
        .p50_ms
        .zip(in_process_p50_ms)
        .map(|(round_trip, in_process)| round_trip - in_process);
    Ok(Report {
        transport: "stdio".to_owned(),
        startup_ms,
        cold_start_ms,
        median_overhead_ms,
        search,
    })
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn cleanup_reports_an_output_pipe_held_by_a_descendant() -> Result<()> {
        let mut child = Command::new("sh")
            .args(["-c", "sleep 3 >&1 & exit 0"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let stdin = child.stdin.take().context("test child stdin missing")?;
        let stdout = child.stdout.take().context("test child stdout missing")?;
        child.wait()?;
        let reader = thread::spawn(move || {
            let mut line = String::new();
            let _ = BufReader::new(stdout).read_line(&mut line);
        });
        let (_, lines) = channel();
        let mut client = Client {
            child,
            stdin: Some(stdin),
            lines,
            reader: Some(reader),
        };
        let error = client
            .finish()
            .err()
            .context("cleanup unexpectedly passed")?;
        assert!(
            error
                .to_string()
                .contains("output pipe did not close after server exit")
        );
        Ok(())
    }
}
