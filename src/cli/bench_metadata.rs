//! Build, machine and effective-input provenance for reproducible benchmarks; v1 §14.
use anyhow::Result;
use serde::{Deserialize, Serialize};
#[cfg(feature = "cuda")]
use std::process::Command;
use std::{
    env, fs,
    io::Read,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct BuildInfo {
    pub commit: String,
    pub source_blake3: String,
    pub source_dirty: String,
    pub profile: String,
    pub opt_level: String,
    pub target: String,
    pub host: String,
    pub features: String,
    pub flags_blake3: String,
    pub rustc: String,
    pub binary_blake3: String,
    pub binary_bytes: u64,
}

impl BuildInfo {
    /// Compares compilation settings while allowing the source revision to change.
    pub(super) fn comparable(&self, other: &Self) -> bool {
        self.profile == other.profile
            && self.opt_level == other.opt_level
            && self.target == other.target
            && self.host == other.host
            && self.features == other.features
            && self.flags_blake3 == other.flags_blake3
            && self.rustc == other.rustc
    }
}

/// Captures the executing artifact's identity, without reading source-checkout HEAD at runtime.
pub(super) fn build_info() -> Result<BuildInfo> {
    let mut binary = fs::File::open(env::current_exe()?)?;
    let binary_bytes = binary.metadata()?.len();
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0; 65536];
    loop {
        let count = binary.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(BuildInfo {
        commit: env!("DKC_BUILD_COMMIT").to_owned(),
        source_blake3: env!("DKC_BUILD_SOURCE_BLAKE3").to_owned(),
        source_dirty: env!("DKC_BUILD_SOURCE_DIRTY").to_owned(),
        profile: env!("DKC_BUILD_PROFILE").to_owned(),
        opt_level: env!("DKC_BUILD_OPT_LEVEL").to_owned(),
        target: env!("DKC_BUILD_TARGET").to_owned(),
        host: env!("DKC_BUILD_HOST").to_owned(),
        features: env!("DKC_BUILD_FEATURES").to_owned(),
        flags_blake3: env!("DKC_BUILD_FLAGS_BLAKE3").to_owned(),
        rustc: env!("DKC_BUILD_RUSTC").to_owned(),
        binary_blake3: hasher.finalize().to_hex().to_string(),
        binary_bytes,
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct Preflight {
    pub captured_at: u64,
    pub cpu_model: Option<String>,
    pub logical_cpus: Option<usize>,
    pub memory_total_kib: Option<u64>,
    pub load_average: Option<Vec<f64>>,
    pub gpu_inventory: Option<String>,
    pub gpu_compute_processes: Option<String>,
    pub idle: Option<bool>,
}

/// Reads load before measured work; probes the device only for GPU-backed runs.
/// Missing probes stay unknown rather than reporting idle.
pub(super) fn preflight(probe_gpu: bool) -> Preflight {
    let cpu_model = fs::read_to_string("/proc/cpuinfo").ok().and_then(|text| {
        text.lines().find_map(|line| {
            line.strip_prefix("model name")
                .and_then(|part| part.strip_prefix("\t: "))
                .map(str::to_owned)
        })
    });
    let memory_total_kib = fs::read_to_string("/proc/meminfo").ok().and_then(|text| {
        text.lines().find_map(|line| {
            line.strip_prefix("MemTotal:")
                .and_then(|part| part.split_whitespace().next())
                .and_then(|part| part.parse().ok())
        })
    });
    let load_average = fs::read_to_string("/proc/loadavg").ok().and_then(|text| {
        text.split_whitespace()
            .take(3)
            .map(str::parse)
            .collect::<std::result::Result<Vec<f64>, _>>()
            .ok()
    });
    // L1/CPU/remote-only benchmarks must not depend on the NVIDIA driver.
    // GPU-backed CUDA runs still capture device identity and activity.
    #[cfg(feature = "cuda")]
    let (gpu_inventory, gpu_compute_processes) = if probe_gpu {
        (
            probe(&[
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]),
            probe(&[
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ]),
        )
    } else {
        (None, None)
    };
    #[cfg(not(feature = "cuda"))]
    let (gpu_inventory, gpu_compute_processes): (Option<String>, Option<String>) = {
        let _ = probe_gpu;
        (None, None)
    };
    let idle = load_average
        .as_ref()
        .and_then(|values| values.first())
        .zip(gpu_compute_processes.as_ref())
        .map(|(load, processes)| *load < 1.0 && processes.is_empty());
    Preflight {
        captured_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |value| value.as_secs()),
        cpu_model,
        logical_cpus: std::thread::available_parallelism().ok().map(usize::from),
        memory_total_kib,
        load_average,
        gpu_inventory,
        gpu_compute_processes,
        idle,
    }
}
#[cfg(feature = "cuda")]
fn probe(args: &[&str]) -> Option<String> {
    let result = Command::new("nvidia-smi").args(args).output().ok()?;
    probe_text(result.status.success(), &result.stdout, &result.stderr)
}

#[cfg(any(feature = "cuda", test))]
fn probe_text(success: bool, stdout: &[u8], stderr: &[u8]) -> Option<String> {
    // NVML can return exit 0 plus partial output when a GPU has fallen off the bus.
    (success && stderr.iter().all(u8::is_ascii_whitespace))
        .then(|| String::from_utf8_lossy(stdout).trim().to_owned())
}

#[cfg(test)]
mod tests {
    #[test]
    fn model_free_preflight_keeps_gpu_metadata_unknown() {
        let preflight = super::preflight(false);
        assert!(preflight.gpu_inventory.is_none());
        assert!(preflight.gpu_compute_processes.is_none());
        assert!(preflight.idle.is_none());
    }

    #[test]
    fn partial_gpu_probe_with_success_exit_is_unknown() {
        assert_eq!(super::probe_text(true, b"", b""), Some(String::new()));
        assert_eq!(
            super::probe_text(
                true,
                b"one surviving GPU",
                b"Unable to determine the device handle for GPU0"
            ),
            None
        );
        assert_eq!(super::probe_text(false, b"", b""), None);
    }
}
