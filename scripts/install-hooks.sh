#!/usr/bin/env bash
# install-hooks.sh — install scripts/guard-plan.sh as the repository's pre-commit hook. Idempotent.
set -euo pipefail
root=$(git rev-parse --show-toplevel)
hook="$root/.git/hooks/pre-commit"
cat >"$hook" <<'EOF'
#!/usr/bin/env bash
exec "$(git rev-parse --show-toplevel)/scripts/guard-plan.sh"
EOF
chmod +x "$hook" "$root/scripts/guard-plan.sh" "$root/scripts/run-detached.sh" 2>/dev/null || true
echo "installed $hook"
