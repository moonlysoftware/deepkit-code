#!/usr/bin/env bash
# Native Apple Silicon release verification for T4.10; see v1.md §§12 and 15.
set -euo pipefail
exec python3 "$(dirname -- "${BASH_SOURCE[0]}")/mac_verify.py" "$@"
