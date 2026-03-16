#!/usr/bin/env bash
# profile-multinode.sh — One-command wrapper to launch a multi-node profiling experiment.
#
# This script:
#   1. Validates arguments and SLURM jobs
#   2. Launches the multi-node experiment via run-multinode-experiment.sh (background)
#   3. Prints log file locations and monitoring commands
#   4. Waits for the user to manually stop via stop-multinode-experiment.sh
#
# Unlike run-multinode-experiment.sh which auto-stops after --duration minutes,
# this wrapper does NOT auto-stop. Use stop-multinode-experiment.sh to stop.
#
# Usage:
#   # Start a profiling run with 2 SGLang nodes + 2 worker nodes, TR router, bs=128
#   bash profile-multinode.sh \
#     --sglang-nodes nodeA:jobid1,nodeB:jobid2 \
#     --worker-nodes nodeC:jobid3,nodeD:jobid4 \
#     --bs 128 --router tr
#
#   # When done (from another terminal):
#   bash scripts/run/stop-multinode-experiment.sh \
#     --sglang-nodes nodeA:jobid1,nodeB:jobid2 \
#     --worker-nodes nodeC:jobid3,nodeD:jobid4
#
#   # Or stop and also reset SGLang containers:
#   bash scripts/run/stop-multinode-experiment.sh \
#     --sglang-nodes nodeA:jobid1,nodeB:jobid2 \
#     --worker-nodes nodeC:jobid3,nodeD:jobid4 \
#     --reset-sglang
#
# Arguments:
#   --sglang-nodes   Comma-separated node:jobid pairs for SGLang GPU servers
#   --worker-nodes   Comma-separated node:jobid pairs for Docker workers
#   --bs             Total concurrent rollout batch size (split across workers)
#   --router         Router mode: "tr" or "default"
#   --ta-node        (Optional) Node to run ThunderAgent on. Default: login node.
#   --ta-jobid       (Optional) SLURM jobid for ThunderAgent node.
#
# Outputs:
#   runs/<router>-<bs>-multinode/
#     thunderagent_profiles/step_profiles.csv
#     sglang-<nodeA>/sglang_metrics.csv
#     sglang-<nodeB>/sglang_metrics.csv
#     logs/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

# ─── Parse arguments (pass-through to run-multinode-experiment.sh) ────
SGLANG_NODES=""
WORKER_NODES=""
BS=""
ROUTER=""
TA_NODE=""
TA_JOBID=""

# Collect all arguments for pass-through
ALL_ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sglang-nodes)  SGLANG_NODES="$2"; shift 2 ;;
    --worker-nodes)  WORKER_NODES="$2"; shift 2 ;;
    --bs)            BS="$2"; shift 2 ;;
    --router)        ROUTER="$2"; shift 2 ;;
    --ta-node)       TA_NODE="$2"; shift 2 ;;
    --ta-jobid)      TA_JOBID="$2"; shift 2 ;;
    *) shift ;;
  esac
done

# ─── Validate ────────────────────────────────────────────────────────
for var in SGLANG_NODES WORKER_NODES BS ROUTER; do
  if [[ -z "${!var}" ]]; then
    echo "ERROR: --$(echo "$var" | tr '[:upper:]' '[:lower:]' | tr '_' '-') is required"
    echo ""
    echo "Usage: $0 \\"
    echo "  --sglang-nodes nodeA:jobid1,nodeB:jobid2 \\"
    echo "  --worker-nodes nodeC:jobid3,nodeD:jobid4 \\"
    echo "  --bs 128 --router {tr|default}"
    exit 1
  fi
done

RUN_NAME="${ROUTER}-${BS}-multinode"
RUN_DIR="${REPO_ROOT}/runs/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"

echo "============================================================"
echo " Multi-Node Profiling: ${RUN_NAME}"
echo "============================================================"
echo " Batch size:   ${BS}"
echo " Router:       ${ROUTER}"
echo " SGLang nodes: ${SGLANG_NODES}"
echo " Worker nodes: ${WORKER_NODES}"
echo " Output dir:   ${RUN_DIR}"
echo "============================================================"
echo ""

# ─── Set a very long duration (we stop manually) ─────────────────────
# Use 10000 minutes (~7 days) so the experiment effectively never auto-stops.
# The user stops it manually with stop-multinode-experiment.sh.
DURATION=10000

echo "[$(date)] Launching experiment (background, duration=${DURATION}m = effectively forever)..."
nohup bash "${REPO_ROOT}/scripts/run/run-multinode-experiment.sh" \
  "${ALL_ARGS[@]}" --duration "$DURATION" \
  > "${REPO_ROOT}/profile-${RUN_NAME}.log" 2>&1 &
MAIN_PID=$!

echo "[$(date)] Main orchestrator PID: $MAIN_PID"
echo "[$(date)] Main log: ${REPO_ROOT}/profile-${RUN_NAME}.log"
echo ""

# ─── Wait for the run directory to be created ────────────────────────
echo "[$(date)] Waiting for run directory to appear..."
for i in $(seq 1 300); do
  if [[ -d "$RUN_DIR" ]]; then
    break
  fi
  if ! kill -0 "$MAIN_PID" 2>/dev/null; then
    echo ""
    echo "[$(date)] ERROR: orchestrator exited unexpectedly. Check the log:"
    echo "  tail -50 ${REPO_ROOT}/profile-${RUN_NAME}.log"
    exit 1
  fi
  sleep 1
done

echo ""
echo "============================================================"
echo " Experiment launched successfully"
echo "============================================================"
echo ""
echo "Monitor progress:"
echo "  # Live monitor (auto-refreshes from CSV files)"
echo "  python3 ${REPO_ROOT}/monitor-multinode.py ${RUN_NAME}"
echo ""
echo "  # Tail the main orchestrator log"
echo "  tail -f ${REPO_ROOT}/profile-${RUN_NAME}.log"
echo ""
echo "  # Check ThunderAgent health"
echo "  curl -s http://localhost:8300/health | python3 -m json.tool"
echo ""
echo "Stop the experiment (3 levels):"
echo "  # (1) Keep SGLang as-is (fastest restart, keeps KV cache):"
echo "  bash ${REPO_ROOT}/scripts/run/stop-multinode-experiment.sh \\"
echo "    --sglang-nodes ${SGLANG_NODES} --worker-nodes ${WORKER_NODES}"
echo ""
echo "  # (2) Flush SGLang KV cache (server stays up, clean cache for next run):"
echo "  bash ${REPO_ROOT}/scripts/run/stop-multinode-experiment.sh \\"
echo "    --sglang-nodes ${SGLANG_NODES} --worker-nodes ${WORKER_NODES} --flush-cache"
echo ""
echo "  # (3) Stop SGLang containers entirely (requires model reload ~3-5min):"
echo "  bash ${REPO_ROOT}/scripts/run/stop-multinode-experiment.sh \\"
echo "    --sglang-nodes ${SGLANG_NODES} --worker-nodes ${WORKER_NODES} --stop-sglang"
echo ""
echo "Compute metrics after stopping:"
echo "  python3 ${REPO_ROOT}/scripts/analysis/compute_metrics_multinode.py --run-dir ${RUN_DIR}"
echo "============================================================"
