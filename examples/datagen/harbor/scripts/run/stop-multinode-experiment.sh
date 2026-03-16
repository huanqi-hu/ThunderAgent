#!/usr/bin/env bash
# stop-multinode-experiment.sh — Gracefully stop all multi-node experiment components.
#
# Usage:
#   # Default: stop everything except SGLang containers
#   bash stop-multinode-experiment.sh \
#     --sglang-nodes nodeA:jobid1,nodeB:jobid2 \
#     --worker-nodes nodeC:jobid3,nodeD:jobid4
#
#   # Flush SGLang KV cache (keep servers running, but clear radix tree):
#   bash stop-multinode-experiment.sh \
#     --sglang-nodes nodeA:jobid1,nodeB:jobid2 \
#     --worker-nodes nodeC:jobid3,nodeD:jobid4 \
#     --flush-cache
#
#   # Stop SGLang containers entirely:
#   bash stop-multinode-experiment.sh \
#     --sglang-nodes nodeA:jobid1,nodeB:jobid2 \
#     --worker-nodes nodeC:jobid3,nodeD:jobid4 \
#     --stop-sglang
#
#   # ThunderAgent on compute node:
#   bash stop-multinode-experiment.sh \
#     --sglang-nodes nodeA:jobid1,nodeB:jobid2 \
#     --worker-nodes nodeC:jobid3,nodeD:jobid4 \
#     --ta-node nodeC --ta-jobid jobid3 \
#     --flush-cache
#
# SGLang handling (mutually exclusive):
#   (default)       Leave SGLang servers as-is (fastest restart, keeps cached KV)
#   --flush-cache   Reset KV cache via SGLang admin API (server stays up, ~1s)
#   --stop-sglang   Stop and remove Docker containers (requires model reload, ~3-5min)
#
# Shutdown order:
#   1. Kill harbor workers on all worker-nodes (stop accepting new trials)
#   2. Kill ThunderAgent (local or remote)
#   3. Kill metrics collectors on all sglang-nodes
#   4. Handle SGLang servers (based on flag)
#   5. Kill harbor coordinator (local process)

set -euo pipefail

# ─── Resolve repo root from this script's location ───────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/env_vars.sh"

# ─── Parse arguments ─────────────────────────────────────────────────
SGLANG_NODES_RAW=""
WORKER_NODES_RAW=""
TA_NODE=""
TA_JOBID=""
SGLANG_ACTION="none"  # none | flush | stop

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sglang-nodes)  SGLANG_NODES_RAW="$2"; shift 2 ;;
    --worker-nodes)  WORKER_NODES_RAW="$2"; shift 2 ;;
    --ta-node)       TA_NODE="$2"; shift 2 ;;
    --ta-jobid)      TA_JOBID="$2"; shift 2 ;;
    --flush-cache)   SGLANG_ACTION="flush"; shift ;;
    --stop-sglang)   SGLANG_ACTION="stop"; shift ;;
    --reset-sglang)  SGLANG_ACTION="stop"; shift ;;  # backward compat
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

for var in SGLANG_NODES_RAW WORKER_NODES_RAW; do
  if [[ -z "${!var}" ]]; then
    echo "ERROR: --$(echo "$var" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | sed 's/-raw$//') is required"
    exit 1
  fi
done

# Determine if ThunderAgent is local (login node) or remote (compute node)
TA_LOCAL=true
if [[ -n "$TA_NODE" ]]; then
  TA_LOCAL=false
  if [[ -z "$TA_JOBID" ]]; then
    echo "ERROR: --ta-jobid is required when --ta-node is set"
    exit 1
  fi
fi

# ─── Parse node:jobid lists ──────────────────────────────────────────
IFS=',' read -ra SGLANG_PAIRS <<< "$SGLANG_NODES_RAW"
SGLANG_HOSTS=()
SGLANG_JIDS=()
for pair in "${SGLANG_PAIRS[@]}"; do
  SGLANG_HOSTS+=("${pair%%:*}")
  SGLANG_JIDS+=("${pair##*:}")
done

IFS=',' read -ra WORKER_PAIRS <<< "$WORKER_NODES_RAW"
WORKER_HOSTS=()
WORKER_JIDS=()
for pair in "${WORKER_PAIRS[@]}"; do
  WORKER_HOSTS+=("${pair%%:*}")
  WORKER_JIDS+=("${pair##*:}")
done

echo "[$(date)] === Stopping multi-node experiment ==="
echo "[$(date)] SGLang nodes: ${SGLANG_HOSTS[*]}"
echo "[$(date)] Worker nodes: ${WORKER_HOSTS[*]}"
echo "[$(date)] SGLang action: ${SGLANG_ACTION}"
if $TA_LOCAL; then
  echo "[$(date)] ThunderAgent: login node (local)"
else
  echo "[$(date)] ThunderAgent: $TA_NODE (jobid: $TA_JOBID)"
fi

# ─── 1. Kill harbor workers on all worker-nodes ──────────────────────
echo ""
echo "[$(date)] Step 1: Killing harbor workers on ${#WORKER_HOSTS[@]} nodes..."
for i in "${!WORKER_HOSTS[@]}"; do
  host="${WORKER_HOSTS[$i]}"
  jid="${WORKER_JIDS[$i]}"
  echo "[$(date)]   Killing worker on $host..."
  srun --jobid "$jid" --overlap --gpus=0 \
    bash -c 'pkill -f "[h]arbor worker run" || true' 2>/dev/null || true
done
echo "[$(date)] All harbor workers stopped"

# ─── 2. Kill ThunderAgent ─────────────────────────────────────────────
echo ""
if $TA_LOCAL; then
  echo "[$(date)] Step 2: Killing ThunderAgent on login node..."
  pkill -f "[T]hunderAgent" || true
else
  echo "[$(date)] Step 2: Killing ThunderAgent on $TA_NODE..."
  srun --jobid "$TA_JOBID" --overlap --gpus=0 \
    bash -c 'pkill -f "[T]hunderAgent" || true' 2>/dev/null || true
fi
echo "[$(date)] ThunderAgent stopped"

# ─── 3. Kill metrics collectors on all sglang-nodes ──────────────────
echo ""
echo "[$(date)] Step 3: Killing metrics collectors on ${#SGLANG_HOSTS[@]} nodes..."
for i in "${!SGLANG_HOSTS[@]}"; do
  host="${SGLANG_HOSTS[$i]}"
  jid="${SGLANG_JIDS[$i]}"
  echo "[$(date)]   Killing metrics on $host..."
  srun --jobid "$jid" --overlap --gpus=0 \
    bash -c 'pkill -f "[c]ollect-metrics" || true' 2>/dev/null || true
done
echo "[$(date)] All metrics collectors stopped"

# ─── 4. Handle SGLang servers ────────────────────────────────────────
echo ""
case "$SGLANG_ACTION" in
  flush)
    echo "[$(date)] Step 4: Flushing KV cache on ${#SGLANG_HOSTS[@]} SGLang servers (--flush-cache)..."
    ADMIN_KEY="${SGLANG_ADMIN_KEY:-}"
    for i in "${!SGLANG_HOSTS[@]}"; do
      host="${SGLANG_HOSTS[$i]}"
      url="http://${host}:${SGLANG_PORT}"
      echo -n "[$(date)]   Flushing cache on $host... "
      CURL_ARGS=(-s --max-time 10 -X POST "${url}/flush_cache")
      if [[ -n "$ADMIN_KEY" ]]; then
        CURL_ARGS+=(-H "X-API-Key: ${ADMIN_KEY}")
      fi
      if curl "${CURL_ARGS[@]}" >/dev/null 2>&1; then
        echo "OK"
      else
        echo "FAILED (server may be unreachable)"
      fi
    done
    echo "[$(date)] KV cache flushed on all SGLang servers (containers still running)"
    ;;
  stop)
    SGLANG_MODE="${SGLANG_MODE:-docker}"
    if [[ "$SGLANG_MODE" == "docker" || "$SGLANG_MODE" == "docker-mount" ]]; then
      echo "[$(date)] Step 4: Stopping SGLang containers on ${#SGLANG_HOSTS[@]} nodes (--stop-sglang)..."
      for i in "${!SGLANG_HOSTS[@]}"; do
        host="${SGLANG_HOSTS[$i]}"
        jid="${SGLANG_JIDS[$i]}"
        echo "[$(date)]   Stopping sglang-m2.5 on $host..."
        srun --jobid "$jid" --overlap --gpus=0 \
          bash -c "
            export XDG_RUNTIME_DIR=/tmp/xdg-\$USER
            export DOCKER_HOST=unix://\$XDG_RUNTIME_DIR/docker.sock
            docker stop sglang-m2.5 2>/dev/null || true
            docker rm -f sglang-m2.5 2>/dev/null || true
          " 2>/dev/null || true
      done
      echo "[$(date)] All SGLang containers stopped and removed"
    else
      echo "[$(date)] Step 4: Stopping SGLang processes on ${#SGLANG_HOSTS[@]} nodes (mode: ${SGLANG_MODE})..."
      for i in "${!SGLANG_HOSTS[@]}"; do
        host="${SGLANG_HOSTS[$i]}"
        jid="${SGLANG_JIDS[$i]}"
        echo "[$(date)]   Killing sglang on $host..."
        srun --jobid "$jid" --overlap --gpus=0 \
          bash -c 'pkill -f "[s]glang.launch_server" || true' 2>/dev/null || true
      done
      echo "[$(date)] All SGLang processes stopped"
    fi
    ;;
  none)
    echo "[$(date)] Step 4: Leaving SGLang servers as-is (no flag specified)"
    echo "[$(date)]   Use --flush-cache to clear KV cache, or --stop-sglang to stop containers"
    ;;
esac

# ─── 5. Kill harbor coordinator (local process) ──────────────────────
echo ""
echo "[$(date)] Step 5: Killing harbor coordinator (local)..."
pkill -f "[h]arbor run.*distributed.*manual-workers" || true
echo "[$(date)] Harbor coordinator stopped"

echo ""
echo "[$(date)] === All experiment components stopped ==="
