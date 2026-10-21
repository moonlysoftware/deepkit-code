#!/usr/bin/env bash
# run-detached.sh — run a long command detached from the session, with log, pid and exit code under notes/runs/.
#
#   scripts/run-detached.sh <name> -- <command…>
#
# Writes notes/runs/<name>.log (stdout+stderr), notes/runs/<name>.pid (while running) and notes/runs/<name>.exit
# (the exit code, written when the command finishes). Refuses to start when <name> is still running.
# A resumed agent session reads .exit and the log before deciding anything (plan/01-workflow.md §8).
set -euo pipefail

if [ $# -lt 3 ] || [ "$2" != "--" ]; then
  echo "usage: $0 <name> -- <command…>" >&2
  exit 2
fi
name=$1; shift 2
root=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
dir="$root/notes/runs"
mkdir -p "$dir"
log="$dir/$name.log"; pidf="$dir/$name.pid"; exitf="$dir/$name.exit"

if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
  echo "run-detached: '$name' is still running (pid $(cat "$pidf")); read $log or choose another name" >&2
  exit 1
fi
rm -f "$exitf"
{
  printf '# run-detached: %s\n# started: %s\n# cwd: %s\n# command: %s\n\n' "$name" "$(date -u +%FT%TZ)" "$PWD" "$*"
} >"$log"

# The wrapper subshell records the exit code; setsid detaches from the terminal where available (Linux).
starter=(bash -c 'cd "$1"; shift; "$@" >>"$LOG" 2>&1; ec=$?; echo "$ec" >"$EXITF"; printf "\n# finished: %s exit=%s\n" "$(date -u +%FT%TZ)" "$ec" >>"$LOG"; rm -f "$PIDF"' _ "$PWD" "$@")
if command -v setsid >/dev/null 2>&1; then
  LOG=$log EXITF=$exitf PIDF=$pidf setsid nohup "${starter[@]}" >/dev/null 2>&1 &
else
  LOG=$log EXITF=$exitf PIDF=$pidf nohup "${starter[@]}" >/dev/null 2>&1 &
fi
echo $! >"$pidf"
echo "run-detached: '$name' started (pid $(cat "$pidf")); log: $log; exit code will appear in $exitf"
