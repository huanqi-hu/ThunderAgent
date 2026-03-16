#!/usr/bin/env bash
# setup-node.sh — Setup Docker prerequisites on a SLURM compute node.
# Run this via srun before launching SGLang or worker processes.
#
# Usage:
#   srun --jobid <JOBID> --overlap --gpus=0 bash scripts/setup/setup-node.sh
#
# What it does:
#   1. Creates XDG_RUNTIME_DIR and scratch directories
#   2. Starts rootless Docker daemon if not already running
#   3. Logs in to Docker Hub (if credentials script is configured)
#   4. Verifies Docker is operational
#
# All configuration comes from env_vars.sh.

set -euo pipefail

# ─── Resolve repo root from this script's location ───────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -f "$REPO_ROOT/env_vars.sh" ]]; then
  source "$REPO_ROOT/env_vars.sh"
else
  echo "WARNING: $REPO_ROOT/env_vars.sh not found, using defaults"
fi

NODE_NAME="$(hostname | cut -d. -f1)"

echo "============================================================"
echo " Setting up node: $NODE_NAME"
echo "============================================================"

# ─── 1. Create required directories ──────────────────────────────────
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-$USER}"
export DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-/scratch/$USER/docker-rootless}"
export DOCKER_HOST="${DOCKER_HOST:-unix://$XDG_RUNTIME_DIR/docker.sock}"
SCRATCH="/scratch/$USER/scratch"

echo "[$(date)] Creating directories..."
mkdir -p "$XDG_RUNTIME_DIR" "$DOCKER_DATA_ROOT" "$SCRATCH"
echo "  XDG_RUNTIME_DIR:  $XDG_RUNTIME_DIR"
echo "  DOCKER_DATA_ROOT: $DOCKER_DATA_ROOT"
echo "  DOCKER_HOST:      $DOCKER_HOST"
echo "  SCRATCH:          $SCRATCH"

DOCKERD_LOG="$SCRATCH/dockerd-setup-${NODE_NAME}.log"

# ─── 2. Start Docker daemon if not running ───────────────────────────
if docker version >/dev/null 2>&1; then
  echo ""
  echo "[$(date)] Docker daemon already running on $NODE_NAME"
  echo "  Driver: $(docker info --format '{{.Driver}}' 2>/dev/null || echo 'unknown')"
  echo "  Images: $(docker images -q 2>/dev/null | wc -l)"
else
  echo ""
  echo "[$(date)] === Starting rootless Docker daemon ==="

  # Clean stale state
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

  echo "[$(date)] Waiting for Docker daemon..."
  for i in $(seq 1 60); do
    docker version >/dev/null 2>&1 && break
    sleep 1
  done

  if ! docker version >/dev/null 2>&1; then
    echo "[$(date)] FATAL: Docker daemon failed to start on $NODE_NAME"
    echo ""
    echo "Last 30 lines of $DOCKERD_LOG:"
    tail -30 "$DOCKERD_LOG"
    exit 1
  fi

  echo "[$(date)] Docker daemon ready"
  echo "  Driver: $(docker info --format '{{.Driver}}')"
fi

# ─── 3. Docker Hub login ─────────────────────────────────────────────
if [[ -n "${DOCKER_CREDENTIALS_SCRIPT:-}" && -f "${DOCKER_CREDENTIALS_SCRIPT}" ]]; then
  echo ""
  echo "[$(date)] Logging in to Docker Hub..."
  source "$DOCKER_CREDENTIALS_SCRIPT"
  echo "$DOCKER_TOKEN" | docker login -u "$DOCKER_USERNAME" --password-stdin 2>/dev/null || true
  echo "[$(date)] Docker Hub login complete"
fi

# ─── 4. Verify ───────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Node setup complete: $NODE_NAME"
echo "============================================================"
echo " Docker version: $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo 'unknown')"
echo " Storage driver: $(docker info --format '{{.Driver}}' 2>/dev/null || echo 'unknown')"
echo " Images cached:  $(docker images -q 2>/dev/null | wc -l)"
echo " Disk usage:"
df -h /scratch/"$USER" 2>/dev/null | tail -1 || echo "  (could not check /scratch)"
echo "============================================================"
