#!/usr/bin/env bash
# stop-experiment.sh — Gracefully stop all experiment components.
#
# Usage:
#   bash stop-experiment.sh \
#     --worker-node research-secure-21 \
#     --sglang-node research-secure-23 \
#     --sglang-jobid 28283 \
#     --worker-jobid 28619
#
# Shutdown order:
#   1. Kill harbor worker (stop accepting new trials)
#   2. Kill ThunderAgent (stop routing requests)
#   3. Kill metrics collector
#   4. Stop SGLang Docker container

set -euo pipefail

# ─── Resolve repo root from this script's location ───────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/env_vars.sh"

# ─── Parse arguments ─────────────────────────────────────────────────
WORKER_NODE=""
SGLANG_NODE=""
SGLANG_JOBID=""
WORKER_JOBID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --worker-node)  WORKER_NODE="$2"; shift 2 ;;
    --sglang-node)  SGLANG_NODE="$2"; shift 2 ;;
    --sglang-jobid) SGLANG_JOBID="$2"; shift 2 ;;
    --worker-jobid) WORKER_JOBID="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

for var in WORKER_NODE SGLANG_NODE SGLANG_JOBID WORKER_JOBID; do
  if [[ -z "${!var}" ]]; then
    echo "ERROR: --$(echo "$var" | tr '[:upper:]' '[:lower:]' | tr '_' '-') is required"
    exit 1
  fi
done

echo "[$(date)] === Stopping experiment ==="
echo "[$(date)] Worker node:  $WORKER_NODE (jobid: $WORKER_JOBID)"
echo "[$(date)] SGLang node:  $SGLANG_NODE (jobid: $SGLANG_JOBID)"

# ─── 1. Kill harbor worker on worker-node ────────────────────────────
echo ""
echo "[$(date)] Step 1: Killing harbor worker on $WORKER_NODE..."
srun --jobid "$WORKER_JOBID" --overlap --gpus=0 \
  bash -c 'pkill -f "[h]arbor worker run" || true' 2>/dev/null || true
echo "[$(date)] Harbor worker stopped"

# ─── 2. Kill ThunderAgent on worker-node ─────────────────────────────
echo ""
echo "[$(date)] Step 2: Killing ThunderAgent on $WORKER_NODE..."
srun --jobid "$WORKER_JOBID" --overlap --gpus=0 \
  bash -c 'pkill -f "[T]hunderAgent" || true' 2>/dev/null || true
echo "[$(date)] ThunderAgent stopped"

# ─── 3. Kill metrics collector on sglang-node ────────────────────────
echo ""
echo "[$(date)] Step 3: Killing metrics collector on $SGLANG_NODE..."
srun --jobid "$SGLANG_JOBID" --overlap --gpus=0 \
  bash -c 'pkill -f "[c]ollect-metrics" || true' 2>/dev/null || true
echo "[$(date)] Metrics collector stopped"

# ─── 4. Stop SGLang on sglang-node (mode-aware) ───────────────────────
SGLANG_MODE="${SGLANG_MODE:-docker}"
echo ""
echo "[$(date)] Step 4: Stopping SGLang (mode=${SGLANG_MODE}) on $SGLANG_NODE..."
case "${SGLANG_MODE}" in
  docker|docker-mount)
    srun --jobid "$SGLANG_JOBID" --overlap --gpus=0 \
      bash -c "
        export XDG_RUNTIME_DIR=/tmp/xdg-\$USER
        export DOCKER_HOST=unix://\$XDG_RUNTIME_DIR/docker.sock
        docker stop sglang-m2.5 2>/dev/null || true
        docker rm -f sglang-m2.5 2>/dev/null || true
      " 2>/dev/null || true
    echo "[$(date)] SGLang container stopped"
    ;;
  conda|uv)
    srun --jobid "$SGLANG_JOBID" --overlap --gpus=0 \
      bash -c 'pkill -f "[s]glang.launch_server" || true' 2>/dev/null || true
    echo "[$(date)] SGLang process stopped"
    ;;
esac

# ─── 5. Kill harbor coordinator (runs on login node) ─────────────────
echo ""
echo "[$(date)] Step 5: Killing harbor coordinator (local)..."
pkill -f "[h]arbor run.*distributed.*manual-workers" || true
echo "[$(date)] Harbor coordinator stopped"

echo ""
echo "[$(date)] === All experiment components stopped ==="
