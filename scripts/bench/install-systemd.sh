#!/usr/bin/env bash
# Install the Phase 4 nightly as a systemd user timer on the bench machine.
set -euo pipefail

root="$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
destination="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
mkdir -p "$destination"

escaped_root="${root//&/\\&}"
sed "s|@ROOT@|${escaped_root}|g" \
  "$root/scripts/bench/systemd/dkc-nightly.service" \
  >"$destination/dkc-nightly.service"
install -m 0644 "$root/scripts/bench/systemd/dkc-nightly.timer" \
  "$destination/dkc-nightly.timer"

systemctl --user daemon-reload
systemctl --user enable --now dkc-nightly.timer
systemctl --user list-timers --all --no-pager dkc-nightly.timer
