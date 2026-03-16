#!/usr/bin/env bash
# run-multinode-experiment.sh — Orchestrator for multi-node SGLang + multi-worker experiments.
#
# Extends run-experiment.sh to support:
#   - Multiple SGLang server nodes (each serving the same model with tp=8 ep=8)
#   - Multiple Docker worker nodes (sharing one NFS work queue via harbor distributed)
#   - ThunderAgent connecting to all SGLang backends for capacity-aware routing
#
# Usage:
#   # Default: ThunderAgent on login node (no srun needed)
#   ./scripts/run/run-multinode-experiment.sh \
#     --sglang-nodes secure-11:28595,secure-06:28600 \
#     --worker-nodes secure-17:28584,secure-18:28307 \
#     --bs 128 --router tr --duration 600
#
#   # Optional: ThunderAgent on a compute node (via srun)
#   ./scripts/run/run-multinode-experiment.sh \
#     --sglang-nodes secure-11:28595,secure-06:28600 \
#     --worker-nodes secure-17:28584,secure-18:28307 \
#     --ta-node secure-17 --ta-jobid 28584 \
#     --bs 128 --router tr --duration 600
#
# Arguments:
#   --sglang-nodes   Comma-separated list of node:jobid pairs for SGLang servers
#   --worker-nodes   Comma-separated list of node:jobid pairs for Docker workers
#   --ta-node        (Optional) Node where ThunderAgent runs. Default: login node.
#   --ta-jobid       (Optional) SLURM jobid for the ThunderAgent node. Required if --ta-node is set.
#   --bs             Total concurrent rollout batch size (split evenly across workers)
#   --router         Router mode: "tr" or "default"
#   --duration       Experiment duration in minutes (default: from env_vars.sh)
#
# Flow:
#   1. Source env_vars.sh, validate args, check SLURM jobs active
#   2. Launch SGLang on each sglang-node in parallel (via srun)
#   3. Wait for ALL SGLang health checks
#   4. Launch ThunderAgent (login node by default, or ta-node via srun)
#   5. Wait for ThunderAgent health
#   6. Launch metrics collector on each sglang-node (background)
#   7. Run harbor coordinator (harbor run --distributed --manual-workers)
#   8. Launch harbor worker on each worker-node
#   9. Sleep for $DURATION minutes
#  10. Run stop-multinode-experiment.sh

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
BS=""
ROUTER=""
DURATION="${DEFAULT_DURATION:-120}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sglang-nodes)  SGLANG_NODES_RAW="$2"; shift 2 ;;
    --worker-nodes)  WORKER_NODES_RAW="$2"; shift 2 ;;
    --ta-node)       TA_NODE="$2"; shift 2 ;;
    --ta-jobid)      TA_JOBID="$2"; shift 2 ;;
    --bs)            BS="$2"; shift 2 ;;
    --router)        ROUTER="$2"; shift 2 ;;
    --duration)      DURATION="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ─── Validate required arguments ─────────────────────────────────────
for var in SGLANG_NODES_RAW WORKER_NODES_RAW BS ROUTER; do
  if [[ -z "${!var}" ]]; then
    echo "ERROR: --$(echo "$var" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | sed 's/-raw$//') is required"
    echo ""
    echo "Usage: $0 \\"
    echo "  --sglang-nodes node1:jobid1,node2:jobid2 \\"
    echo "  --worker-nodes node1:jobid1,node2:jobid2 \\"
    echo "  --bs N --router {tr|default} [--duration MIN] \\"
    echo "  [--ta-node NODE --ta-jobid JOBID]"
    exit 1
  fi
done

# If --ta-node is set, --ta-jobid is also required
if [[ -n "$TA_NODE" && -z "$TA_JOBID" ]]; then
  echo "ERROR: --ta-jobid is required when --ta-node is set"
  exit 1
fi

if [[ "$ROUTER" != "tr" && "$ROUTER" != "default" ]]; then
  echo "ERROR: --router must be 'tr' or 'default', got '$ROUTER'"
  exit 1
fi

# ─── Parse node:jobid lists ──────────────────────────────────────────
# Parse into parallel arrays: SGLANG_HOSTS[i], SGLANG_JIDS[i]
IFS=',' read -ra SGLANG_PAIRS <<< "$SGLANG_NODES_RAW"
SGLANG_HOSTS=()
SGLANG_JIDS=()
for pair in "${SGLANG_PAIRS[@]}"; do
  host="${pair%%:*}"
  jid="${pair##*:}"
  if [[ -z "$host" || -z "$jid" || "$host" == "$jid" ]]; then
    echo "ERROR: Invalid sglang-nodes format '$pair'. Expected 'hostname:jobid'"
    exit 1
  fi
  SGLANG_HOSTS+=("$host")
  SGLANG_JIDS+=("$jid")
done

IFS=',' read -ra WORKER_PAIRS <<< "$WORKER_NODES_RAW"
WORKER_HOSTS=()
WORKER_JIDS=()
for pair in "${WORKER_PAIRS[@]}"; do
  host="${pair%%:*}"
  jid="${pair##*:}"
  if [[ -z "$host" || -z "$jid" || "$host" == "$jid" ]]; then
    echo "ERROR: Invalid worker-nodes format '$pair'. Expected 'hostname:jobid'"
    exit 1
  fi
  WORKER_HOSTS+=("$host")
  WORKER_JIDS+=("$jid")
done

N_SGLANG=${#SGLANG_HOSTS[@]}
N_WORKERS=${#WORKER_HOSTS[@]}

# ─── Compute per-worker concurrency ──────────────────────────────────
# Split BS evenly across workers; remainder goes to the first workers
BS_PER_WORKER=$((BS / N_WORKERS))
BS_REMAINDER=$((BS % N_WORKERS))

# ─── Determine ThunderAgent location ──────────────────────────────────
# Default: login node (local). Override with --ta-node/--ta-jobid for compute node.
TA_LOCAL=true
if [[ -n "$TA_NODE" ]]; then
  TA_LOCAL=false
fi

# ─── Derived paths ────────────────────────────────────────────────────
JOB_NAME="m2.5-multinode-bs${BS}-${ROUTER}"
RUN_ROOT="${REPO_ROOT}/runs/${ROUTER}-${BS}-multinode"
JOB_DIR="${HARBOR_JOBS_DIR}/${JOB_NAME}"
LOG_DIR="${RUN_ROOT}/logs"

# ThunderAgent URL: use login node hostname for local, ta-node for remote.
# Docker containers on worker nodes reach ThunderAgent via this hostname.
if $TA_LOCAL; then
  TA_HOSTNAME="$(hostname -f)"
  TA_URL="http://${TA_HOSTNAME}:${TA_PORT}"
else
  TA_HOSTNAME="${TA_NODE}"
  TA_URL="http://${TA_NODE}:${TA_PORT}"
fi

# Build comma-separated backend URLs for ThunderAgent
BACKEND_URLS=""
for host in "${SGLANG_HOSTS[@]}"; do
  if [[ -n "$BACKEND_URLS" ]]; then
    BACKEND_URLS+=","
  fi
  BACKEND_URLS+="http://${host}:${SGLANG_PORT}/v1"
done

mkdir -p "$RUN_ROOT" "$LOG_DIR" "$HARBOR_JOBS_DIR"

# ─── Print experiment summary ─────────────────────────────────────────
echo "============================================================"
echo " Multi-Node Experiment: ${JOB_NAME}"
echo "============================================================"
echo " BS (total):     ${BS}"
echo " Router:         ${ROUTER}"
echo " Duration:       ${DURATION} minutes"
echo " Run root:       ${RUN_ROOT}"
echo " Job dir:        ${JOB_DIR}"
echo ""
echo " SGLang nodes (${N_SGLANG}):"
for i in "${!SGLANG_HOSTS[@]}"; do
  echo "   [${i}] ${SGLANG_HOSTS[$i]}:${SGLANG_PORT} (jobid: ${SGLANG_JIDS[$i]})"
done
echo ""
if $TA_LOCAL; then
  echo " ThunderAgent:   ${TA_HOSTNAME}:${TA_PORT} (login node, local)"
else
  echo " ThunderAgent:   ${TA_NODE}:${TA_PORT} (jobid: ${TA_JOBID}, compute node)"
fi
echo "   backends:     ${BACKEND_URLS}"
echo ""
echo " Worker nodes (${N_WORKERS}):"
for i in "${!WORKER_HOSTS[@]}"; do
  local_bs=$((BS_PER_WORKER + (i < BS_REMAINDER ? 1 : 0)))
  echo "   [${i}] ${WORKER_HOSTS[$i]} (jobid: ${WORKER_JIDS[$i]}, concurrent: ${local_bs})"
done
echo "============================================================"
echo ""

# ─── 1. Check SLURM jobs are active ──────────────────────────────────
echo "[$(date)] === Step 1: Checking SLURM jobs ==="
ALL_JIDS=("${SGLANG_JIDS[@]}" "${WORKER_JIDS[@]}")
if ! $TA_LOCAL; then
  ALL_JIDS+=("$TA_JOBID")
fi
# Deduplicate (ta-jobid may overlap with a worker-jobid)
UNIQUE_JIDS=($(printf '%s\n' "${ALL_JIDS[@]}" | sort -u))
for jid in "${UNIQUE_JIDS[@]}"; do
  STATE=$(squeue --job "$jid" --noheader -o "%T" 2>/dev/null || echo "UNKNOWN")
  if [[ "$STATE" != "RUNNING" ]]; then
    echo "ERROR: SLURM job $jid is not RUNNING (state: $STATE)"
    exit 1
  fi
done
echo "[$(date)] All ${#UNIQUE_JIDS[@]} SLURM jobs are RUNNING"

# ─── 2. Launch SGLang on each sglang-node ─────────────────────────────
echo ""
echo "[$(date)] === Step 2: Launching SGLang on ${N_SGLANG} nodes ==="
SGLANG_PIDS=()
for i in "${!SGLANG_HOSTS[@]}"; do
  host="${SGLANG_HOSTS[$i]}"
  jid="${SGLANG_JIDS[$i]}"
  log="$LOG_DIR/sglang-${host}.log"
  echo "[$(date)] Launching SGLang on ${host} (jobid: ${jid})..."
  srun --jobid "$jid" --overlap --gpus=8 \
    bash "$SCRIPT_DIR/launch-sglang.sh" \
    > "$log" 2>&1 &
  SGLANG_PIDS+=($!)
  echo "[$(date)]   PID: ${SGLANG_PIDS[$i]}, log: $log"
done

# ─── 3. Wait for ALL SGLang health checks ─────────────────────────────
echo ""
echo "[$(date)] === Step 3: Waiting for SGLang health on all ${N_SGLANG} nodes ==="
HEALTH_TIMEOUT=600
HEALTHY_FLAGS=()
for i in "${!SGLANG_HOSTS[@]}"; do
  HEALTHY_FLAGS+=("false")
done

for sec in $(seq 1 "$HEALTH_TIMEOUT"); do
  all_healthy=true
  for i in "${!SGLANG_HOSTS[@]}"; do
    if [[ "${HEALTHY_FLAGS[$i]}" == "true" ]]; then
      continue
    fi
    host="${SGLANG_HOSTS[$i]}"
    url="http://${host}:${SGLANG_PORT}"
    if curl -sf --max-time 3 "${url}/health" >/dev/null 2>&1; then
      HEALTHY_FLAGS[$i]="true"
      echo "[$(date)] SGLang on ${host} is healthy after ${sec}s"
    else
      all_healthy=false
    fi
  done

  if $all_healthy; then
    echo "[$(date)] All ${N_SGLANG} SGLang servers are healthy"
    break
  fi

  if [[ $((sec % 30)) -eq 0 ]]; then
    healthy_count=0
    for flag in "${HEALTHY_FLAGS[@]}"; do
      if [[ "$flag" == "true" ]]; then healthy_count=$((healthy_count + 1)); fi
    done
    echo "[$(date)] Still waiting... (${sec}/${HEALTH_TIMEOUT}s, ${healthy_count}/${N_SGLANG} healthy)"
  fi

  # Check if any srun process died
  for i in "${!SGLANG_PIDS[@]}"; do
    if [[ "${HEALTHY_FLAGS[$i]}" == "false" ]] && ! kill -0 "${SGLANG_PIDS[$i]}" 2>/dev/null; then
      echo "[$(date)] FATAL: SGLang srun process for ${SGLANG_HOSTS[$i]} exited unexpectedly"
      tail -50 "$LOG_DIR/sglang-${SGLANG_HOSTS[$i]}.log"
      exit 1
    fi
  done

  sleep 1
done

# Final check
for i in "${!SGLANG_HOSTS[@]}"; do
  if [[ "${HEALTHY_FLAGS[$i]}" != "true" ]]; then
    echo "[$(date)] FATAL: SGLang on ${SGLANG_HOSTS[$i]} did not become healthy within ${HEALTH_TIMEOUT}s"
    tail -50 "$LOG_DIR/sglang-${SGLANG_HOSTS[$i]}.log"
    exit 1
  fi
done

# ─── 4. Launch ThunderAgent ───────────────────────────────────────────
echo ""
if $TA_LOCAL; then
  echo "[$(date)] === Step 4: Launching ThunderAgent on login node (${TA_HOSTNAME}) ==="
else
  echo "[$(date)] === Step 4: Launching ThunderAgent on compute node ${TA_NODE} ==="
fi

TA_EXTRA_ARGS=""
if [[ "$ROUTER" == "tr" ]]; then
  TA_EXTRA_ARGS="--acting-token-weight 1.0 --use-acting-token-decay"
fi

if $TA_LOCAL; then
  # Launch ThunderAgent directly on login node (no srun)
  bash "$SCRIPT_DIR/launch-thunderagent.sh" \
    "$RUN_ROOT" "$BACKEND_URLS" "$ROUTER" $TA_EXTRA_ARGS \
    > "$LOG_DIR/thunderagent.log" 2>&1 &
else
  # Launch ThunderAgent on compute node via srun
  srun --jobid "$TA_JOBID" --overlap --gpus=0 \
    bash "$SCRIPT_DIR/launch-thunderagent.sh" \
      "$RUN_ROOT" "$BACKEND_URLS" "$ROUTER" $TA_EXTRA_ARGS \
    > "$LOG_DIR/thunderagent.log" 2>&1 &
fi
TA_PID=$!
echo "[$(date)] ThunderAgent PID: $TA_PID, log: $LOG_DIR/thunderagent.log"

# ─── 5. Wait for ThunderAgent health ─────────────────────────────────
echo ""
# For health check, use localhost when ThunderAgent is local
if $TA_LOCAL; then
  TA_HEALTH_URL="http://localhost:${TA_PORT}"
else
  TA_HEALTH_URL="${TA_URL}"
fi
echo "[$(date)] === Step 5: Waiting for ThunderAgent health at ${TA_HEALTH_URL}/health ==="
TA_HEALTH_TIMEOUT=120
for i in $(seq 1 "$TA_HEALTH_TIMEOUT"); do
  if curl -sf --max-time 3 "${TA_HEALTH_URL}/health" >/dev/null 2>&1; then
    echo "[$(date)] ThunderAgent is healthy after ${i}s"
    break
  fi
  if [[ $((i % 15)) -eq 0 ]]; then
    echo "[$(date)] Still waiting for ThunderAgent... (${i}/${TA_HEALTH_TIMEOUT}s)"
  fi
  if ! kill -0 "$TA_PID" 2>/dev/null; then
    echo "[$(date)] FATAL: ThunderAgent process exited unexpectedly"
    tail -50 "$LOG_DIR/thunderagent.log"
    exit 1
  fi
  sleep 1
done

if ! curl -sf --max-time 3 "${TA_HEALTH_URL}/health" >/dev/null 2>&1; then
  echo "[$(date)] FATAL: ThunderAgent did not become healthy within ${TA_HEALTH_TIMEOUT}s"
  tail -50 "$LOG_DIR/thunderagent.log"
  exit 1
fi

# ─── 6. Start metrics collectors on each sglang-node ──────────────────
echo ""
echo "[$(date)] === Step 6: Starting metrics collectors on ${N_SGLANG} SGLang nodes ==="
METRICS_PIDS=()
for i in "${!SGLANG_HOSTS[@]}"; do
  host="${SGLANG_HOSTS[$i]}"
  jid="${SGLANG_JIDS[$i]}"
  metrics_dir="${RUN_ROOT}/sglang-${host}"
  metrics_log="$LOG_DIR/metrics-${host}.log"
  mkdir -p "$metrics_dir"
  srun --jobid "$jid" --overlap --gpus=0 \
    bash "$SCRIPT_DIR/collect-metrics.sh" \
      "$metrics_dir" "http://${host}:${SGLANG_PORT}" 5 \
    > "$metrics_log" 2>&1 &
  METRICS_PIDS+=($!)
  echo "[$(date)]   ${host}: PID ${METRICS_PIDS[$i]}, dir: $metrics_dir"
done

# ─── 7. Run harbor coordinator ───────────────────────────────────────
echo ""
echo "[$(date)] === Step 7: Starting harbor coordinator ==="

# Activate the harbor virtualenv
source "${HARBOR_ENV_DIR}/bin/activate"

# Build --nodes flags for each worker
NODES_FLAGS=""
for i in "${!WORKER_HOSTS[@]}"; do
  local_bs=$((BS_PER_WORKER + (i < BS_REMAINDER ? 1 : 0)))
  NODES_FLAGS+=" --nodes ${WORKER_HOSTS[$i]}:${local_bs}"
done

harbor run \
    --distributed --manual-workers \
    $NODES_FLAGS \
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

# Give the coordinator a moment to create the job directory and queue
sleep 5

# ─── 8. Launch workers on each worker-node ────────────────────────────
echo ""
echo "[$(date)] === Step 8: Launching harbor workers on ${N_WORKERS} nodes ==="
WORKER_PIDS=()
for i in "${!WORKER_HOSTS[@]}"; do
  host="${WORKER_HOSTS[$i]}"
  jid="${WORKER_JIDS[$i]}"
  local_bs=$((BS_PER_WORKER + (i < BS_REMAINDER ? 1 : 0)))
  worker_log="$LOG_DIR/worker-${host}.log"
  echo "[$(date)] Launching worker on ${host} (jobid: ${jid}, concurrent: ${local_bs})..."
  srun --jobid "$jid" --overlap --gpus=0 \
    bash "$SCRIPT_DIR/launch-worker.sh" \
      "$JOB_DIR" "$local_bs" "$host" \
    > "$worker_log" 2>&1 &
  WORKER_PIDS+=($!)
  echo "[$(date)]   PID: ${WORKER_PIDS[$i]}, log: $worker_log"
done

# ─── 9. Sleep for $DURATION minutes ──────────────────────────────────
echo ""
echo "[$(date)] === Step 9: Experiment running for ${DURATION} minutes ==="
echo ""
echo "Log files:"
for i in "${!SGLANG_HOSTS[@]}"; do
  echo "  SGLang (${SGLANG_HOSTS[$i]}):   $LOG_DIR/sglang-${SGLANG_HOSTS[$i]}.log"
done
echo "  ThunderAgent:                 $LOG_DIR/thunderagent.log"
for i in "${!SGLANG_HOSTS[@]}"; do
  echo "  Metrics (${SGLANG_HOSTS[$i]}):  $LOG_DIR/metrics-${SGLANG_HOSTS[$i]}.log"
done
echo "  Coordinator:                  $LOG_DIR/harbor-coordinator.log"
for i in "${!WORKER_HOSTS[@]}"; do
  echo "  Worker (${WORKER_HOSTS[$i]}):   $LOG_DIR/worker-${WORKER_HOSTS[$i]}.log"
done
echo ""

# Save PIDs and node info for stop-multinode-experiment.sh
cat > "$RUN_ROOT/experiment.pids" <<EOF
# Auto-generated by run-multinode-experiment.sh at $(date)
SGLANG_NODES_RAW=$SGLANG_NODES_RAW
WORKER_NODES_RAW=$WORKER_NODES_RAW
TA_LOCAL=$TA_LOCAL
TA_NODE=${TA_NODE:-}
TA_JOBID=${TA_JOBID:-}
TA_HOSTNAME=$TA_HOSTNAME
TA_PID=$TA_PID
HARBOR_PID=$HARBOR_PID
EOF

# Save per-node PIDs
for i in "${!SGLANG_HOSTS[@]}"; do
  echo "SGLANG_PID_${i}=${SGLANG_PIDS[$i]}" >> "$RUN_ROOT/experiment.pids"
done
for i in "${!METRICS_PIDS[@]}"; do
  echo "METRICS_PID_${i}=${METRICS_PIDS[$i]}" >> "$RUN_ROOT/experiment.pids"
done
for i in "${!WORKER_HOSTS[@]}"; do
  echo "WORKER_PID_${i}=${WORKER_PIDS[$i]}" >> "$RUN_ROOT/experiment.pids"
done

echo "[$(date)] PID file saved to: $RUN_ROOT/experiment.pids"
echo ""

sleep "$((DURATION * 60))"

# ─── 10. Stop experiment ─────────────────────────────────────────────
echo ""
echo "[$(date)] === Step 10: Stopping experiment ==="
STOP_ARGS="--sglang-nodes $SGLANG_NODES_RAW --worker-nodes $WORKER_NODES_RAW"
if ! $TA_LOCAL; then
  STOP_ARGS+=" --ta-node $TA_NODE --ta-jobid $TA_JOBID"
fi
bash "$SCRIPT_DIR/stop-multinode-experiment.sh" $STOP_ARGS

echo ""
echo "[$(date)] === Experiment ${JOB_NAME} complete ==="
echo "[$(date)] Results in: ${RUN_ROOT}"
echo "[$(date)] Job dir:    ${JOB_DIR}"
