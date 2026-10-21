#!/usr/bin/env bash
# Use Zig for the target-specific C, C++ and Rust linker edges of a musl dist build.
set -euo pipefail

tool=${0##*/}
case "$tool" in
  dkc-zig-musl-x86_64-cc) arch=x86_64; driver=cc; role=compiler ;;
  dkc-zig-musl-x86_64-cxx) arch=x86_64; driver=c++; role=compiler ;;
  dkc-zig-musl-x86_64-linker) arch=x86_64; driver=c++; role=linker ;;
  dkc-zig-musl-aarch64-cc) arch=aarch64; driver=cc; role=compiler ;;
  dkc-zig-musl-aarch64-cxx) arch=aarch64; driver=c++; role=compiler ;;
  dkc-zig-musl-aarch64-linker) arch=aarch64; driver=c++; role=linker ;;
  *) echo "unknown Zig musl tool name: $tool" >&2; exit 2 ;;
esac

rust_target="${arch}-unknown-linux-musl"
zig_target="${arch}-linux-musl"
arguments=()
wide_vector=false
for argument in "$@"; do
  case "$argument" in
    "--target=$rust_target")
      if [[ "$role" == compiler ]]; then arguments+=("--target=$zig_target"); fi ;;
    -mavx512f)
      arguments+=("$argument")
      if [[ "$arch" == x86_64 && "$role" == compiler ]]; then wide_vector=true; fi ;;
    *self-contained*crt*.o|-lc)
      if [[ "$role" == compiler ]]; then arguments+=("$argument"); fi ;;
    -Wl,--fix-cortex-a53-843419)
      if [[ "$role" == compiler ]]; then arguments+=("$argument"); fi ;;
    -lgcc_s)
      if [[ "$role" == linker ]]; then arguments+=(-lunwind); else arguments+=("$argument"); fi ;;
    *) arguments+=("$argument") ;;
  esac
done

# Zig 0.15.1 needs this spelling when cc-rs has already requested AVX-512.
if "$wide_vector"; then arguments+=(-mavx10.1-512); fi
exec zig "$driver" -target "$zig_target" "${arguments[@]}"
