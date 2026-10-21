#!/usr/bin/env bash
# Reproducible document corpus; plan/05-corpus.md §4 and v1 §6/§16.
set -euo pipefail
exec python3 -B "$(dirname -- "$0")/make-fixtures-docs.py" "$@"
