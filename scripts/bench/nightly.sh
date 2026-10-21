#!/usr/bin/env bash
# Full Phase 4 regression; plan/06-benchmarks.md and T4.5.
set -euo pipefail
exec python3 "$(dirname -- "${BASH_SOURCE[0]}")/nightly.py" "$@"
