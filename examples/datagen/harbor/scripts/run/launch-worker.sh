#!/usr/bin/env bash
# launch-worker.sh — Start a harbor worker that pulls and executes trials.
# Designed to be called via srun on the worker compute node.
#
# Usage:
#   bash launch-worker.sh <job-dir> <n-concurrent> <node-name>
#
# Arguments:
#   job-dir      — Harbor job directory (e.g., ~/harbor-jobs/m2.5-nohicache-bs48-tr)
#   n-concurrent — Number of concurrent trials (typically == batch size)
#   node-name    — Node identifier for the worker

set -euo pipefail

# ─── Resolve repo root from this script's location ───────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/env_vars.sh"

# ─── Parse arguments ─────────────────────────────────────────────────
JOB_DIR="${1:?Usage: $0 <job-dir> <n-concurrent> <node-name>}"
N_CONCURRENT="${2:?Usage: $0 <job-dir> <n-concurrent> <node-name>}"
NODE_NAME="${3:?Usage: $0 <job-dir> <n-concurrent> <node-name>}"

# ─── Rootless Docker environment ─────────────────────────────────────
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-$USER}"
export DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-/scratch/$USER/docker-rootless}"
export DOCKER_HOST="${DOCKER_HOST:-unix://$XDG_RUNTIME_DIR/docker.sock}"
SCRATCH="/scratch/$USER/scratch"
mkdir -p "$XDG_RUNTIME_DIR" "$DOCKER_DATA_ROOT" "$SCRATCH"

DOCKERD_LOG="$SCRATCH/dockerd-worker-${NODE_NAME}.log"

# ─── Start Docker daemon if not running ──────────────────────────────
if docker version >/dev/null 2>&1; then
  echo "[$(date)] Docker daemon already running"
else
  echo "[$(date)] Starting rootless Docker daemon..."
  rm -f "$XDG_RUNTIME_DIR/docker.sock" "$XDG_RUNTIME_DIR/docker.pid" || true
  rm -rf "$XDG_RUNTIME_DIR/docker-exec" || true
  rm -f "$XDG_RUNTIME_DIR/dockerd-rootless/lock" || true
  pkill -f '[r]ootlesskit.*dockerd' || true
  pkill -f '[d]ockerd.*docker-rootless' || true
  sleep 2

  dockerd-rootless.sh \
    --data-root "$DOCKER_DATA_ROOT" \
    --exec-root "$XDG_RUNTIME_DIR/docker-exec" \
    --pidfile "$XDG_RUNTIME_DIR/docker.pid" \
    --host "$DOCKER_HOST" \
    --storage-driver "${DOCKER_STORAGE_DRIVER:-overlay2}" \
    --exec-opt native.cgroupdriver=cgroupfs \
    > "$DOCKERD_LOG" 2>&1 &

  for i in $(seq 1 60); do
    docker version >/dev/null 2>&1 && break
    sleep 1
  done

  if ! docker version >/dev/null 2>&1; then
    echo "[$(date)] FATAL: Docker daemon failed to start"
    tail -30 "$DOCKERD_LOG"
    exit 1
  fi
  echo "[$(date)] Docker daemon ready"
fi

# ─── Docker Hub login for SWE-bench image pulls ──────────────────────
if [[ -n "${DOCKER_CREDENTIALS_SCRIPT:-}" && -f "${DOCKER_CREDENTIALS_SCRIPT}" ]]; then
  source "$DOCKER_CREDENTIALS_SCRIPT"
  echo "$DOCKER_TOKEN" | docker login -u "$DOCKER_USERNAME" --password-stdin 2>/dev/null || true
fi

echo "[$(date)] Image count: $(docker images -q | wc -l)"
echo "[$(date)] Starting Harbor worker: job=$JOB_DIR, concurrent=$N_CONCURRENT, node=$NODE_NAME"

# ─── Activate virtualenv and run worker ──────────────────────────────
source "${HARBOR_ENV_DIR}/bin/activate"

exec harbor worker run \
  --job-dir "$JOB_DIR" \
  --n-concurrent "$N_CONCURRENT" \
  --node-name "$NODE_NAME"
