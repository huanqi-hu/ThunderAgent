#!/usr/bin/env bash
# launch-thunderagent.sh — Start ThunderAgent routing proxy.
# Designed to be called via srun on the worker compute node.
#
# Usage:
#   bash launch-thunderagent.sh <run-root> <backend-url> [router-mode] [extra-args...]
#
# Arguments:
#   run-root     — Directory for ThunderAgent profiles/metrics output
#   backend-url  — SGLang backend URL (e.g., http://research-secure-23:8000/v1)
#   router-mode  — "default" or "tr" (default: "default")
#   extra-args   — Additional ThunderAgent CLI arguments
#
# For "tr" router, automatically adds --acting-token-weight 1.0 --use-acting-token-decay.

set -euo pipefail

# ─── Resolve repo root from this script's location ───────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/env_vars.sh"

# ─── Parse arguments ─────────────────────────────────────────────────
RUN_ROOT="${1:?Usage: $0 <run-root> <sglang-backend-url> [router-mode] [extra-args...]}"
BACKEND_URL="${2:?Usage: $0 <run-root> <sglang-backend-url> [router-mode] [extra-args...]}"
ROUTER_MODE="${3:-default}"
shift 3 2>/dev/null || shift $# 2>/dev/null
EXTRA_ARGS="$*"

# ─── Auto-add TR-specific flags ──────────────────────────────────────
if [[ "$ROUTER_MODE" == "tr" ]]; then
  # Only add if not already present in EXTRA_ARGS
  if [[ "$EXTRA_ARGS" != *"--acting-token-weight"* ]]; then
    EXTRA_ARGS="--acting-token-weight 1.0 --use-acting-token-decay ${EXTRA_ARGS}"
  fi
fi

mkdir -p "$RUN_ROOT/thunderagent_profiles"

# ─── Activate virtualenv ─────────────────────────────────────────────
source "${HARBOR_ENV_DIR}/bin/activate"

echo "[$(date)] Starting ThunderAgent (router=$ROUTER_MODE) -> $BACKEND_URL"
echo "[$(date)] Profile dir: $RUN_ROOT/thunderagent_profiles"
echo "[$(date)] Extra args: $EXTRA_ARGS"

exec python3 -m ThunderAgent \
  --host 0.0.0.0 --port "${TA_PORT}" \
  --router "$ROUTER_MODE" \
  --backend-type sglang \
  --backends "$BACKEND_URL" \
  --metrics --profile \
  --profile-dir "$RUN_ROOT/thunderagent_profiles" \
  --metrics-interval 5.0 \
  $EXTRA_ARGS
