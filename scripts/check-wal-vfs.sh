#!/usr/bin/env bash
# Native boundary/ASan/UBSan checks against the actual bundled SQLite source; v1 §3.5.
set -euo pipefail
if [[ $# != 1 || ! -f "$1/sqlite3.c" || ! -f "$1/sqlite3.h" ]]; then
  echo 'usage: scripts/check-wal-vfs.sh <bundled-sqlite-source-directory>' >&2
  exit 2
fi
scratch=$(mktemp -d /tmp/dkc-wal-vfs-XXXXXX)
trap 'rm -rf "$scratch"' EXIT
"${CC:-clang}" -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer \
  -I "$1" "$1/sqlite3.c" crates/sqlite-vector-sys/src/wal_vfs.c \
  fixtures/native/wal_vfs.c -ldl -lpthread -lm -o "$scratch/check"
ASAN_OPTIONS=detect_leaks=1 "$scratch/check" "$scratch/code.db"
