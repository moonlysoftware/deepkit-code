#!/usr/bin/env bash
# Builds reproducible small or 1,000-file-distant branches for freshness tests.
set -euo pipefail

repository=''
distant=false
while (($#)); do
  case "$1" in
    --distant) distant=true ;;
    -h|--help) printf 'Usage: %s [--distant] DESTINATION\n' "$0"; exit 0 ;;
    -*) printf 'Unknown fixture option: %s\n' "$1" >&2; exit 2 ;;
    *) if [[ -n "$repository" ]]; then printf 'Supply one destination\n' >&2; exit 2; fi
       repository=$1 ;;
  esac
  shift
done
[[ -n "$repository" ]] || { printf 'Missing fixture destination\n' >&2; exit 2; }
[[ ! -e "$repository/.git" ]] || { printf 'Fixture destination already contains Git history\n' >&2; exit 2; }
git init --initial-branch=main "$repository" >/dev/null
git -C "$repository" config user.email 'dkc-test@example.invalid'
git -C "$repository" config user.name 'Deepkit Code test'
git -C "$repository" config commit.gpgsign false
git -C "$repository" config core.autocrlf false

mkdir -p "$repository/src"
printf '%s\n' 'pub fn branch_name() -> &'"'"'static str { "main" }' >"$repository/src/lib.rs"
printf '%s\n' '# Fixture repository' >"$repository/README.md"
if $distant; then
  mkdir -p "$repository/src/generated"
  # Together with src/lib.rs, exactly 1,000 tracked files differ.
  for ((i=0; i<999; i++)); do
    printf -v stem '%04d' "$i"
    printf 'pub fn branch_probe_%s() -> usize { %d }\n' "$stem" "$i" >"$repository/src/generated/probe_$stem.rs"
  done
fi
git -C "$repository" add README.md src
GIT_AUTHOR_DATE=2000-01-01T00:00:00Z GIT_COMMITTER_DATE=2000-01-01T00:00:00Z git -C "$repository" commit --quiet -m 'main fixture'

git -C "$repository" checkout --quiet -b alternate
printf '%s\n' 'pub fn branch_name() -> &'"'"'static str { "alternate" }' >"$repository/src/lib.rs"
if $distant; then
  for ((i=0; i<999; i++)); do
    printf -v stem '%04d' "$i"
    printf '// alternate branch\npub fn branch_probe_%s() -> usize {\n    %d\n}\n' "$stem" "$((i + 1000))" >"$repository/src/generated/probe_$stem.rs"
  done
fi
git -C "$repository" add src
GIT_AUTHOR_DATE=2000-01-02T00:00:00Z GIT_COMMITTER_DATE=2000-01-02T00:00:00Z git -C "$repository" commit --quiet -m 'alternate fixture'
git -C "$repository" checkout --quiet main
