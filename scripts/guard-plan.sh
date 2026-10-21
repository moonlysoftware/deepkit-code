#!/usr/bin/env bash
# guard-plan.sh — refuse commits that touch human-owned files, leak secrets, or add large files.
#
# Human-owned (read-only for the agent): plan/, v1.md, research.md, overview.md, deepkit/, AGENTS.md.
# See plan/01-workflow.md §1 and §7.
#
# Usage:
#   scripts/guard-plan.sh              # pre-commit mode: inspects the staged changes
#   scripts/guard-plan.sh --ci [RANGE] # CI mode: inspects every commit in RANGE; RANGE defaults to $GUARD_RANGE, then
#                                      # origin/main..HEAD when that is non-empty, then HEAD~1..HEAD (HEAD for a root snapshot). On a push to main
#                                      # origin/main == HEAD, so CI must pass "<before>..<sha>" (checkout with fetch-depth 0).
#
# Bypass, humans only:
#   pre-commit:  PLAN_EDIT=1 git commit -m "docs(plan): … [plan]"
#   CI:          the commit message must contain "[plan]"
set -euo pipefail

PROTECTED_RE='^(plan/|v1\.md$|research\.md$|overview\.md$|deepkit/|AGENTS\.md$)'
FORBIDDEN_RE='(^|/)(\.deepkit/|credentials\.toml$)|\.gguf$'
MAX_BYTES=$((1024 * 1024))
# Conservative secret patterns: provider-style keys and explicit key assignments with long literals.
SECRET_RE='(sk-[A-Za-z0-9_-]{16,}|(api[_-]?key|secret|token|password)[[:space:]]*[:=][[:space:]]*["'"'"'][A-Za-z0-9_./+=-]{16,}["'"'"'])'

fail=0
say() { printf 'guard-plan: %s\n' "$*" >&2; }

# Note: these functions take their input as arguments, not via a pipe, so that `fail=1` survives (a piped function
# runs in a subshell and its variables are lost).
check_paths() {
  # $1 = label, $2 = newline-separated paths
  local label=$1 p
  while IFS= read -r p; do
    [ -n "$p" ] || continue
    if [[ "$p" =~ $PROTECTED_RE ]]; then
      say "$label touches human-owned file: $p"
      fail=1
    fi
    if [[ "$p" =~ $FORBIDDEN_RE ]]; then
      say "$label adds a forbidden file: $p"
      fail=1
    fi
  done <<<"$2"
}

check_secrets() {
  # $1 = label, $2 = added lines of a diff (without the leading +)
  local label=$1 hits
  hits=$(printf '%s\n' "$2" | grep -E -i -n "$SECRET_RE" || true)
  if [ -n "$hits" ]; then
    say "$label: possible secret in added lines:"
    printf '%s\n' "$hits" | sed 's/^/  /' >&2
    fail=1
  fi
}

added_lines() {
  # $@ = git diff command; prints the added lines without the leading +
  "$@" | grep -E '^\+' | grep -Ev '^\+\+\+' | sed 's/^+//' || true
}

if [ "${1:-}" = "--ci" ]; then
  range=${2:-${GUARD_RANGE:-}}
  # A push that creates a branch reports an all-zero "before" sha; fall back to the default range in that case.
  if [[ "$range" == 0000000000000000000000000000000000000000..* ]]; then range=""; fi
  if [ -z "$range" ]; then
    if git rev-parse --verify -q origin/main >/dev/null && [ "$(git rev-parse origin/main)" != "$(git rev-parse HEAD)" ]; then
      range="origin/main..HEAD"
    elif ! git rev-parse --verify -q HEAD^ >/dev/null && [ "$(git rev-parse --is-shallow-repository)" = "false" ]; then
      # A parentless initial snapshot establishes the baseline: diff-tree reports
      # no parent-relative changes. Later commits retain the same checks below.
      # A shallow checkout is not evidence of a parentless initial snapshot.
      range="HEAD"
    else
      range="HEAD~1..HEAD"
    fi
  fi
  if ! commits=$(git rev-list "$range" 2>/dev/null); then
    say "cannot resolve range '$range' (shallow checkout? use fetch-depth: 0)"
    exit 2
  fi
  [ -n "$commits" ] || say "no commits in range '$range'; nothing to check"
  for c in $commits; do
    label="commit ${c:0:7}"
    msg=$(git log -1 --format=%B "$c")
    if [[ "$msg" != *"[plan]"* ]]; then
      check_paths "$label" "$(git diff-tree --no-commit-id --name-only -r "$c")"
    fi
    check_secrets "$label" "$(added_lines git diff-tree --no-commit-id -r -p "$c")"
    while IFS=$'\t' read -r blob path; do
      [ -n "$blob" ] && [ "$blob" != "0000000000000000000000000000000000000000" ] || continue
      size=$(git cat-file -s "$blob")
      if [ "$size" -gt "$MAX_BYTES" ]; then say "$label adds a file over 1 MB: $path ($size bytes)"; fail=1; fi
    done < <(git diff-tree --no-commit-id -r --diff-filter=AM "$c" | awk '{print $4"\t"$6}')
  done
else
  staged=$(git diff --cached --name-only --diff-filter=ACMR)
  if [ "${PLAN_EDIT:-0}" != "1" ]; then
    check_paths "staged change" "$staged"
  fi
  check_secrets "staged change" "$(added_lines git diff --cached -p)"
  while IFS= read -r p; do
    [ -n "$p" ] && [ -f "$p" ] || continue
    size=$(wc -c <"$p" | tr -d ' ')
    if [ "$size" -gt "$MAX_BYTES" ]; then say "staged file over 1 MB: $p ($size bytes); generate it at test time instead"; fail=1; fi
  done <<<"$staged"
fi

if [ "$fail" -ne 0 ]; then
  say "refusing. Human-owned files change only via PLAN_EDIT=1 (local) / '[plan]' in the message (CI). Agents: write a PCR in notes/pcr/ instead."
  exit 1
fi
exit 0
