#!/usr/bin/env bash
# collect-metrics.sh — Collect SGLang + GPU metrics at regular intervals.
# Designed to be called via srun on the SGLang compute node.
#
# Usage:
#   bash collect-metrics.sh <metrics-dir> [sglang-url] [interval-seconds]
#
# Arguments:
#   metrics-dir  — Output directory for CSV files
#   sglang-url   — SGLang base URL (default: http://localhost:8000)
#   interval     — Collection interval in seconds (default: 5)
#
# Outputs:
#   metrics-dir/sglang_metrics.csv — SGLang Prometheus metrics (throughput, cache, retractions)
#   metrics-dir/gpu_metrics.csv    — nvidia-smi GPU stats (util, memory, temp, power)

# Only set -u (not -e) because curl failures are expected and handled
set -u

# ─── Resolve repo root from this script's location ───────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/env_vars.sh"

# ─── Parse arguments ─────────────────────────────────────────────────
METRICS_DIR="${1:?Usage: $0 <metrics-dir> [sglang-url] [interval-seconds]}"
SGLANG_URL="${2:-http://localhost:${SGLANG_PORT}}"
INTERVAL="${3:-5}"

mkdir -p "$METRICS_DIR"

SGLANG_CSV="$METRICS_DIR/sglang_metrics.csv"
GPU_CSV="$METRICS_DIR/gpu_metrics.csv"

# ─── CSV headers ─────────────────────────────────────────────────────
echo "timestamp,gen_throughput_total,num_running_reqs,num_waiting_reqs,num_queue_reqs,token_usage,cache_hit_rate,retracted_reqs,retracted_requests_total,retracted_input_tokens_total,retracted_output_tokens_total,evicted_tokens_total,prefill_cache_tokens_total,prefill_compute_tokens_total" > "$SGLANG_CSV"
echo "timestamp,gpu,util_pct,mem_used_mib,mem_total_mib,temp_c,power_w" > "$GPU_CSV"

echo "[$(date)] Metrics collector started (interval=${INTERVAL}s)"
echo "[$(date)] SGLang: $SGLANG_URL, output: $METRICS_DIR"

while true; do
  TS=$(date +%s)

  # ─── SGLang metrics (Prometheus format) ─────────────────────────────
  PROM=$(curl -s --max-time 3 "$SGLANG_URL/metrics" 2>/dev/null || echo "")
  if [ -n "$PROM" ]; then
    GEN_TP=$(echo "$PROM" | grep '^sglang:gen_throughput{' | awk '{print $2}' | head -1)
    RUNNING=$(echo "$PROM" | grep '^sglang:num_running_reqs{' | awk '{print $2}' | head -1)
    WAITING=$(echo "$PROM" | grep '^sglang:num_waiting_reqs{' | awk '{print $2}' | head -1)
    QUEUE=$(echo "$PROM" | grep '^sglang:num_queue_reqs{' | awk '{print $2}' | head -1)
    TOKEN_USAGE=$(echo "$PROM" | grep '^sglang:token_usage{' | awk '{print $2}' | head -1)
    CACHE_HIT=$(echo "$PROM" | grep '^sglang:cache_hit_rate{' | awk '{print $2}' | head -1)
    # Retraction metrics (key for tr vs default router comparison)
    RETRACTED_REQS=$(echo "$PROM" | grep '^sglang:num_retracted_reqs{' | awk '{print $2}' | head -1)
    RETRACTED_TOTAL=$(echo "$PROM" | grep '^sglang:num_retracted_requests_total{' | awk '{print $2}' | head -1)
    RETRACTED_IN_TOK=$(echo "$PROM" | grep '^sglang:num_retracted_input_tokens_total{' | awk '{print $2}' | head -1)
    RETRACTED_OUT_TOK=$(echo "$PROM" | grep '^sglang:num_retracted_output_tokens_total{' | awk '{print $2}' | head -1)
    # Radix tree eviction + cumulative prefill cache metrics
    EVICTED_TOK=$(echo "$PROM" | grep '^sglang:evicted_tokens_total{' | awk '{print $2}' | head -1)
    PREFILL_CACHE=$(echo "$PROM" | grep '^sglang:realtime_tokens_total{.*mode="prefill_cache"' | awk '{print $2}' | head -1)
    PREFILL_COMPUTE=$(echo "$PROM" | grep '^sglang:realtime_tokens_total{.*mode="prefill_compute"' | awk '{print $2}' | head -1)
    echo "$TS,${GEN_TP:-0},${RUNNING:-0},${WAITING:-0},${QUEUE:-0},${TOKEN_USAGE:-0},${CACHE_HIT:-0},${RETRACTED_REQS:-0},${RETRACTED_TOTAL:-0},${RETRACTED_IN_TOK:-0},${RETRACTED_OUT_TOK:-0},${EVICTED_TOK:-0},${PREFILL_CACHE:-0},${PREFILL_COMPUTE:-0}" >> "$SGLANG_CSV"
  fi

  # ─── GPU metrics ───────────────────────────────────────────────────
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | while IFS=, read -r idx util mem_used mem_total temp power; do
    echo "$TS,$idx,$util,$mem_used,$mem_total,$temp,$power" >> "$GPU_CSV"
  done

  sleep "$INTERVAL"
done
