#!/usr/bin/env bash
# Native release-artifact smoke test for T4.7; see v1.md §12.
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: scripts/release/smoke.sh [/path/to/dkc]" >&2
  exit 2
fi

source_binary="${1:-${DKC_BIN:-}}"
if [[ -z "$source_binary" ]]; then
  source_binary="$(command -v dkc || true)"
fi
if [[ ! -f "$source_binary" || ! -x "$source_binary" ]]; then
  echo "dkc artifact is not an executable file: $source_binary" >&2
  exit 2
fi

root="$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
scratch="$(mktemp -d "${TMPDIR:-/tmp}/dkc-release-smoke.XXXXXX")"
trap 'rm -rf "$scratch"' EXIT

mkdir -p "$scratch/bin" "$scratch/home" "$scratch/repository"
cp "$source_binary" "$scratch/bin/dkc"
chmod 0755 "$scratch/bin/dkc"
cp -R "$root/fixtures/tiny/." "$scratch/repository/"

export HOME="$scratch/home"
export DKC_MODELS_DIR="${DKC_MODELS_DIR:-$scratch/models}"
git -C "$scratch/repository" init --quiet
git -C "$scratch/repository" add .
git -C "$scratch/repository" \
  -c user.name='Deepkit Release Smoke' \
  -c user.email='release-smoke@example.invalid' \
  commit --quiet -m 'release smoke fixture'

binary="$scratch/bin/dkc"
cd "$scratch/repository"
"$binary" init --profile local --yes --json >"$scratch/init.json"
"$binary" index --level 1 --json >"$scratch/index.json"
"$binary" search should_retry --mode lexical --no-rerank --json >"$scratch/search.json"
"$binary" status --json >"$scratch/status.json"

python3 - "$scratch" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
index = json.loads((root / "index.json").read_text())
search = json.loads((root / "search.json").read_text())
status = json.loads((root / "status.json").read_text())
hits = search.get("hits", [])
if not any(hit.get("path") == "script.rs" for hit in hits):
    raise SystemExit("release smoke search did not return script.rs")
if status.get("files", 0) < 2 or status.get("units", 0) < 1:
    raise SystemExit("release smoke status does not show an indexed fixture")
if index.get("embedding", {}).get("state") not in {"quiescent", "skipped"}:
    raise SystemExit("release smoke embedding pass did not complete")
if status.get("queue", {}).get("dead", 0) != 0:
    raise SystemExit("release smoke status reported dead jobs")
print(json.dumps({
    "binary": str(root / "bin" / "dkc"),
    "files": status["files"],
    "units": status["units"],
    "search_hits": len(hits),
    "script_hit": True,
}, sort_keys=True))
PY
