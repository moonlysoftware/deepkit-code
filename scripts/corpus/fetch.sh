#!/usr/bin/env bash
# Provision/verify pinned full corpus clones; plan/05-corpus.md §2–3.
set -euo pipefail
exec python3 -B "$(dirname -- "$0")/fetch.py" "$@"
