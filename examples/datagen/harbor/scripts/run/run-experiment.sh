#!/usr/bin/env bash
# run-experiment.sh — Main orchestrator for a TR-vs-default router experiment.
#
# Usage:
#   ./scripts/run/run-experiment.sh \
#     --bs 48 --router tr \
#     --sglang-node research-secure-23 --worker-node research-secure-21 \
#     --sglang-jobid 28283 --worker-jobid 28619 \
#     [--duration 120]
#
# Flow:
#   1. Source env_vars.sh, validate args, check SLURM jobs active
#   2. srun on sglang-node -> launch-sglang.sh
#   3. Wait for SGLang health
#   4. srun on worker-node -> launch-thunderagent.sh (background)
#   5. Wait for ThunderAgent health
#   6. srun on sglang-node -> collect-metrics.sh (background)
#   7. Run harbor coordinator (harbor run)
#   8. srun on worker-node -> launch-worker.sh
#   9. Sleep $DURATION minutes
#  10. Run stop-experiment.sh

set -euo pipefail

# ─── Resolve repo root from this script's location ───────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/env_vars.sh"

# ─── Parse arguments ─────────────────────────────────────────────────
BS=""
ROUTER=""
SGLANG_NODE=""
WORKER_NODE=""
SGLANG_JOBID=""
WORKER_JOBID=""
DURATION="${DEFAULT_DURATION:-120}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bs)           BS="$2"; shift 2 ;;
    --router)       ROUTER="$2"; shift 2 ;;
    --sglang-node)  SGLANG_NODE="$2"; shift 2 ;;
    --worker-node)  WORKER_NODE="$2"; shift 2 ;;
    --sglang-jobid) SGLANG_JOBID="$2"; shift 2 ;;
    --worker-jobid) WORKER_JOBID="$2"; shift 2 ;;
    --duration)     DURATION="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ─── Validate required arguments ─────────────────────────────────────
for var in BS ROUTER SGLANG_NODE WORKER_NODE SGLANG_JOBID WORKER_JOBID; do
  if [[ -z "${!var}" ]]; then
    echo "ERROR: --$(echo "$var" | tr '[:upper:]' '[:lower:]' | tr '_' '-') is required"
    echo "Usage: $0 --bs N --router {tr|default} --sglang-node NODE --worker-node NODE --sglang-jobid JOB --worker-jobid JOB [--duration MIN]"
    exit 1
  fi
done

if [[ "$ROUTER" != "tr" && "$ROUTER" != "default" ]]; then
  echo "ERROR: --router must be 'tr' or 'default', got '$ROUTER'"
  exit 1
fi

# ─── Derived paths ────────────────────────────────────────────────────
JOB_NAME="m2.5-nohicache-bs${BS}-${ROUTER}"
RUN_ROOT="${REPO_ROOT}/runs/${ROUTER}-${BS}"
JOB_DIR="${HARBOR_JOBS_DIR}/${JOB_NAME}"
SGLANG_URL="http://${SGLANG_NODE}:${SGLANG_PORT}"
TA_URL="http://${WORKER_NODE}:${TA_PORT}"
LOG_DIR="${RUN_ROOT}/logs"

mkdir -p "$RUN_ROOT" "$LOG_DIR" "$HARBOR_JOBS_DIR"

echo "============================================================"
echo " Experiment: ${JOB_NAME}"
echo "============================================================"
echo " BS:           ${BS}"
echo " Router:       ${ROUTER}"
echo " SGLang node:  ${SGLANG_NODE} (jobid: ${SGLANG_JOBID})"
echo " Worker node:  ${WORKER_NODE} (jobid: ${WORKER_JOBID})"
echo " Duration:     ${DURATION} minutes"
echo " Run root:     ${RUN_ROOT}"
echo " Job dir:      ${JOB_DIR}"
echo " SGLang URL:   ${SGLANG_URL}"
echo " TA URL:       ${TA_URL}"
echo "============================================================"
echo ""

# ─── 1. Check SLURM jobs are active ──────────────────────────────────
echo "[$(date)] Checking SLURM jobs..."
for jid in "$SGLANG_JOBID" "$WORKER_JOBID"; do
  STATE=$(squeue --job "$jid" --noheader -o "%T" 2>/dev/null || echo "UNKNOWN")
  if [[ "$STATE" != "RUNNING" ]]; then
    echo "ERROR: SLURM job $jid is not RUNNING (state: $STATE)"
    exit 1
  fi
done
echo "[$(date)] Both SLURM jobs are RUNNING"

# ─── 2. Launch SGLang on sglang-node ─────────────────────────────────
echo ""
echo "[$(date)] === Step 2: Launching SGLang on ${SGLANG_NODE} ==="
srun --jobid "$SGLANG_JOBID" --overlap --gpus=8 \
  bash "$SCRIPT_DIR/launch-sglang.sh" \
  > "$LOG_DIR/sglang.log" 2>&1 &
SGLANG_PID=$!
echo "[$(date)] SGLang launch PID: $SGLANG_PID"

# ─── 3. Wait for SGLang health ───────────────────────────────────────
echo ""
echo "[$(date)] === Step 3: Waiting for SGLang health at ${SGLANG_URL}/health ==="
HEALTH_TIMEOUT=600
for i in $(seq 1 "$HEALTH_TIMEOUT"); do
  if curl -sf --max-time 3 "${SGLANG_URL}/health" >/dev/null 2>&1; then
    echo "[$(date)] SGLang is healthy after ${i}s"
    break
  fi
  if [[ $((i % 30)) -eq 0 ]]; then
    echo "[$(date)] Still waiting for SGLang... (${i}/${HEALTH_TIMEOUT}s)"
  fi
  # Check if the srun process died
  if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
    echo "[$(date)] FATAL: SGLang srun process exited unexpectedly"
    tail -50 "$LOG_DIR/sglang.log"
    exit 1
  fi
  sleep 1
done

if ! curl -sf --max-time 3 "${SGLANG_URL}/health" >/dev/null 2>&1; then
  echo "[$(date)] FATAL: SGLang did not become healthy within ${HEALTH_TIMEOUT}s"
  tail -50 "$LOG_DIR/sglang.log"
  exit 1
fi

# ─── 4. Launch ThunderAgent on worker-node ────────────────────────────
echo ""
echo "[$(date)] === Step 4: Launching ThunderAgent on ${WORKER_NODE} ==="

TA_EXTRA_ARGS=""
if [[ "$ROUTER" == "tr" ]]; then
  TA_EXTRA_ARGS="--acting-token-weight 1.0 --use-acting-token-decay"
fi

srun --jobid "$WORKER_JOBID" --overlap --gpus=0 \
  bash "$SCRIPT_DIR/launch-thunderagent.sh" \
    "$RUN_ROOT" "${SGLANG_URL}/v1" "$ROUTER" $TA_EXTRA_ARGS \
  > "$LOG_DIR/thunderagent.log" 2>&1 &
TA_PID=$!
echo "[$(date)] ThunderAgent launch PID: $TA_PID"

# ─── 5. Wait for ThunderAgent health ─────────────────────────────────
echo ""
echo "[$(date)] === Step 5: Waiting for ThunderAgent health at ${TA_URL}/health ==="
TA_HEALTH_TIMEOUT=120
for i in $(seq 1 "$TA_HEALTH_TIMEOUT"); do
  if curl -sf --max-time 3 "${TA_URL}/health" >/dev/null 2>&1; then
    echo "[$(date)] ThunderAgent is healthy after ${i}s"
    break
  fi
  if [[ $((i % 15)) -eq 0 ]]; then
    echo "[$(date)] Still waiting for ThunderAgent... (${i}/${TA_HEALTH_TIMEOUT}s)"
  fi
  if ! kill -0 "$TA_PID" 2>/dev/null; then
    echo "[$(date)] FATAL: ThunderAgent srun process exited unexpectedly"
    tail -50 "$LOG_DIR/thunderagent.log"
    exit 1
  fi
  sleep 1
done

if ! curl -sf --max-time 3 "${TA_URL}/health" >/dev/null 2>&1; then
  echo "[$(date)] FATAL: ThunderAgent did not become healthy within ${TA_HEALTH_TIMEOUT}s"
  tail -50 "$LOG_DIR/thunderagent.log"
  exit 1
fi

# ─── 6. Start metrics collector on sglang-node (background) ──────────
echo ""
echo "[$(date)] === Step 6: Starting metrics collector on ${SGLANG_NODE} ==="
srun --jobid "$SGLANG_JOBID" --overlap --gpus=0 \
  bash "$SCRIPT_DIR/collect-metrics.sh" \
    "$RUN_ROOT" "$SGLANG_URL" 5 \
  > "$LOG_DIR/metrics-collector.log" 2>&1 &
METRICS_PID=$!
echo "[$(date)] Metrics collector PID: $METRICS_PID"

# ─── 7. Run harbor coordinator ───────────────────────────────────────
echo ""
echo "[$(date)] === Step 7: Starting harbor coordinator ==="

# Activate the harbor virtualenv
source "${HARBOR_ENV_DIR}/bin/activate"

harbor run \
    --distributed --manual-workers \
    --nodes "${WORKER_NODE}:${BS}" \
    --dataset "${DEFAULT_DATASET}" \
    --agent openhands \
    --model "openai/MiniMaxAI/MiniMax-M2.5" \
    --ak "api_base=${TA_URL}/v1" \
    --ak "local_path=${OPENHANDS_PATH}" \
    --ak max_iterations=100 \
    --prompt-template "${REPO_ROOT}/adapters/swebench/prompts/8phase.md.j2" \
    --network-mode "${NETWORK_MODE}" \
    --override-cpus "${OVERRIDE_CPUS}" \
    --override-memory-mb "${OVERRIDE_MEMORY_MB}" \
    --max-retries 3 \
    -v "${OPENHANDS_PATH}:/opt/openhands-src:ro" \
    --ae "DISABLE_STUCK_DETECTION=${AGENT_DISABLE_STUCK_DETECTION}" \
    --ae "ENABLE_DEFAULT_CONDENSER=${AGENT_ENABLE_DEFAULT_CONDENSER}" \
    --ae "LLM_API_KEY=${AGENT_LLM_API_KEY}" \
    --ae "LLM_NATIVE_TOOL_CALLING=${AGENT_LLM_NATIVE_TOOL_CALLING}" \
    --ae "LLM_TEMPERATURE=${AGENT_LLM_TEMPERATURE}" \
    --ae "LLM_TOP_P=${AGENT_LLM_TOP_P}" \
    --ae "LLM_TOP_K=${AGENT_LLM_TOP_K}" \
    --n-tasks "${DEFAULT_N_TASKS}" --n-attempts "${DEFAULT_N_ATTEMPTS}" \
    --jobs-dir "${HARBOR_JOBS_DIR}" \
    --job-name "${JOB_NAME}" \
    --quiet \
  > "$LOG_DIR/harbor-coordinator.log" 2>&1 &
HARBOR_PID=$!
echo "[$(date)] Harbor coordinator PID: $HARBOR_PID"

# Give the coordinator a moment to create the job directory
sleep 5

# ─── 8. Launch worker on worker-node ─────────────────────────────────
echo ""
echo "[$(date)] === Step 8: Launching harbor worker on ${WORKER_NODE} ==="
srun --jobid "$WORKER_JOBID" --overlap --gpus=0 \
  bash "$SCRIPT_DIR/launch-worker.sh" \
    "$JOB_DIR" "$BS" "$WORKER_NODE" \
  > "$LOG_DIR/worker.log" 2>&1 &
WORKER_PID=$!
echo "[$(date)] Worker PID: $WORKER_PID"

# ─── 9. Sleep for $DURATION minutes ──────────────────────────────────
echo ""
echo "[$(date)] === Step 9: Experiment running for ${DURATION} minutes ==="
echo "[$(date)] Log files:"
echo "  SGLang:         $LOG_DIR/sglang.log"
echo "  ThunderAgent:   $LOG_DIR/thunderagent.log"
echo "  Metrics:        $LOG_DIR/metrics-collector.log"
echo "  Coordinator:    $LOG_DIR/harbor-coordinator.log"
echo "  Worker:         $LOG_DIR/worker.log"
echo ""
echo "[$(date)] PIDs: sglang=$SGLANG_PID ta=$TA_PID metrics=$METRICS_PID harbor=$HARBOR_PID worker=$WORKER_PID"
echo ""

# Save PIDs for stop-experiment.sh
cat > "$RUN_ROOT/experiment.pids" <<EOF
SGLANG_PID=$SGLANG_PID
TA_PID=$TA_PID
METRICS_PID=$METRICS_PID
HARBOR_PID=$HARBOR_PID
WORKER_PID=$WORKER_PID
SGLANG_NODE=$SGLANG_NODE
WORKER_NODE=$WORKER_NODE
SGLANG_JOBID=$SGLANG_JOBID
WORKER_JOBID=$WORKER_JOBID
EOF

sleep "$((DURATION * 60))"

# ─── 10. Stop experiment ─────────────────────────────────────────────
echo ""
echo "[$(date)] === Step 10: Stopping experiment ==="
bash "$SCRIPT_DIR/stop-experiment.sh" \
  --worker-node "$WORKER_NODE" \
  --sglang-node "$SGLANG_NODE" \
  --sglang-jobid "$SGLANG_JOBID" \
  --worker-jobid "$WORKER_JOBID"

echo ""
echo "[$(date)] === Experiment ${JOB_NAME} complete ==="
echo "[$(date)] Results in: ${RUN_ROOT}"
echo "[$(date)] Job dir:    ${JOB_DIR}"
