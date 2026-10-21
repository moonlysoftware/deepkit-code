#!/usr/bin/env bash
# T1.4 component diagnostics; run under taskset and run-detached on the bench machine.
# Set DKC_BENCH_BATCH_TIMINGS=1 for per-batch context/decode/read/drop durations.
set -euo pipefail
: "${DKC_MODELS_DIR:?Set DKC_MODELS_DIR to the verified GGUF cache}"
export DKC_BENCH_CPU_THREADS=${DKC_BENCH_CPU_THREADS:-8}
python3 - <<'PY'
import json, os, platform, subprocess
print(json.dumps({
    'kind': 'local_embedding_component_diagnostic',
    'machine': platform.machine(),
    'kernel': platform.release(),
    'affinity': sorted(os.sched_getaffinity(0)),
    'load_average': os.getloadavg(),
    'threads': int(os.environ['DKC_BENCH_CPU_THREADS']),
    'batch_timings': os.environ.get('DKC_BENCH_BATCH_TIMINGS') == '1',
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True)),
    'rustc': subprocess.check_output(['rustc', '--version'], text=True).strip(),
    'cargo_profile': 'test (Rust unoptimized; native llama.cpp build as pinned)',
    'features': [],
    'scope': 'native embedding only; excludes context construction, queue and vector writes',
}), flush=True)
PY
for repeats in 1 16; do
    DKC_BENCH_DOCUMENT_REPEATS=$repeats cargo test models::llama::tests::embed_throughput -- --ignored --nocapture
done
