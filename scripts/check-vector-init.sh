#!/usr/bin/env bash
# Native startup/ASan/UBSan checks against the actual bundled SQLite; v1 §3.5.
set -euo pipefail
if [[ $# != 1 || ! -f "$1/sqlite3.c" || ! -f "$1/sqlite3.h" ]]; then
  echo 'usage: scripts/check-vector-init.sh <bundled-sqlite-source-directory>' >&2
  exit 2
fi
scratch=$(mktemp -d /tmp/dkc-vector-init-XXXXXX)
trap 'rm -rf "$scratch"' EXIT
source_dir=crates/sqlite-vector-sys/vendor/sqlite-vector/src
library_dir=crates/sqlite-vector-sys/vendor/sqlite-vector/libs
flags=(-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer -I "$1" -I "$source_dir" -I "$library_dir")
for unit in sqlite-vector distance-cpu distance-sse2 distance-neon distance-rvv; do
  "${CC:-clang}" "${flags[@]}" -c "$source_dir/$unit.c" -o "$scratch/$unit.o"
done
"${CC:-clang}" "${flags[@]}" -mavx2 -mfma -mf16c -c "$source_dir/distance-avx2.c" -o "$scratch/avx2.o"
"${CC:-clang}" "${flags[@]}" -mavx512f -mavx512bw -mavx512vl -mavx512dq -c "$source_dir/distance-avx512.c" -o "$scratch/avx512.o"
"${CC:-clang}" "${flags[@]}" "$1/sqlite3.c" fixtures/native/vector_init.c "$scratch/"*.o -ldl -lpthread -lm -o "$scratch/check"
ASAN_OPTIONS=detect_leaks=1 "$scratch/check"
