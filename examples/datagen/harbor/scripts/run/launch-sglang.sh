#!/usr/bin/env bash
# launch-sglang.sh — Start SGLang server in one of 4 modes.
# Designed to be called via srun on the SGLang compute node.
# All configuration comes from env_vars.sh.
#
# Modes (set SGLANG_MODE in env_vars.sh):
#   docker        — Run inside SGLANG_IMAGE container (default)
#   docker-mount  — Same, but mount local sglang source into container
#   conda         — Native launch via conda environment
#   uv            — Native launch via uv + local sglang project
#
# Usage (called by run-experiment.sh):
#   srun --jobid $JOB --overlap --gpus=8 bash launch-sglang.sh

set -euo pipefail

# ─── Resolve repo root from this script's location ───────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/env_vars.sh"

NODE_NAME="$(hostname | cut -d. -f1)"
SGLANG_MODE="${SGLANG_MODE:-docker}"

# ─── Common SGLang launch_server arguments ───────────────────────────
# Model path differs between docker (container mount) and native (host path).
build_sglang_args() {
  local model_path="$1"
  SGLANG_ARGS=(
    --model-path "$model_path"
    --tp 8 --ep 8 --trust-remote-code
    --chunked-prefill-size 8192 --page-size 64
    --attention-backend fa3 --decode-attention-backend flashinfer
    --tool-call-parser minimax-m2
    --reasoning-parser minimax-append-think
    --served-model-name MiniMaxAI/MiniMax-M2.5
    --enable-metrics --enable-cache-report
    --admin-api-key "${SGLANG_ADMIN_KEY}"
    --host 0.0.0.0 --port "${SGLANG_PORT}"
    ${SGLANG_EXTRA_ARGS:-}
  )
}

# ─── Docker helpers ──────────────────────────────────────────────────
setup_rootless_docker() {
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-$USER}"
  export DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-/scratch/$USER/docker-rootless}"
  export DOCKER_HOST="${DOCKER_HOST:-unix://$XDG_RUNTIME_DIR/docker.sock}"
  SCRATCH="/scratch/$USER/scratch"
  mkdir -p "$XDG_RUNTIME_DIR" "$DOCKER_DATA_ROOT" "$SCRATCH"

  DOCKERD_LOG="$SCRATCH/dockerd-sglang-${NODE_NAME}.log"

  if docker version >/dev/null 2>&1; then
    echo "[$(date)] Docker daemon already running on $NODE_NAME"
  else
    echo "[$(date)] === Cleaning old Docker state on $NODE_NAME ==="
    rm -f "$XDG_RUNTIME_DIR/docker.sock" "$XDG_RUNTIME_DIR/docker.pid" || true
    rm -rf "$XDG_RUNTIME_DIR/docker-exec" || true
    rm -f "$XDG_RUNTIME_DIR/dockerd-rootless/lock" || true
    pkill -f '[r]ootlesskit.*dockerd' || true
    pkill -f '[d]ockerd.*docker-rootless' || true
    sleep 2

    echo "[$(date)] === Starting rootless Docker daemon ==="
    dockerd-rootless.sh \
      --data-root "$DOCKER_DATA_ROOT" \
      --exec-root "$XDG_RUNTIME_DIR/docker-exec" \
      --pidfile "$XDG_RUNTIME_DIR/docker.pid" \
      --host "$DOCKER_HOST" \
      --storage-driver "${DOCKER_STORAGE_DRIVER:-overlay2}" \
      --exec-opt native.cgroupdriver=cgroupfs \
      > "$DOCKERD_LOG" 2>&1 &

    echo "[$(date)] === Waiting for Docker daemon ==="
    for i in $(seq 1 60); do
      docker version >/dev/null 2>&1 && break
      sleep 1
    done

    if ! docker version >/dev/null 2>&1; then
      echo "[$(date)] FATAL: Docker daemon failed to start on $NODE_NAME"
      tail -50 "$DOCKERD_LOG"
      exit 1
    fi
    echo "[$(date)] Docker daemon ready (driver: $(docker info --format '{{.Driver}}'))"
  fi
}

pull_image_if_needed() {
  if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "${SGLANG_IMAGE}"; then
    echo "[$(date)] === Pulling ${SGLANG_IMAGE} ==="
    if [[ -n "${DOCKER_CREDENTIALS_SCRIPT:-}" && -f "${DOCKER_CREDENTIALS_SCRIPT}" ]]; then
      source "$DOCKER_CREDENTIALS_SCRIPT"
      echo "$DOCKER_TOKEN" | docker login -u "$DOCKER_USERNAME" --password-stdin 2>/dev/null || true
    fi
    docker pull "$SGLANG_IMAGE"
    echo "[$(date)] Image pulled"
  else
    echo "[$(date)] ${SGLANG_IMAGE} already present"
  fi
}

wait_for_uvicorn() {
  echo "[$(date)] === Container started, waiting for server ready ==="
  for i in $(seq 1 600); do
    if docker logs sglang-m2.5 2>&1 | tail -5 | grep -q 'Uvicorn running'; then
      echo "[$(date)] SGLang server is READY on ${NODE_NAME}:${SGLANG_PORT}"
      return 0
    fi
    if [[ $((i % 30)) -eq 0 ]]; then
      echo "[$(date)] Still loading... ($i seconds)"
    fi
    sleep 1
  done
  echo "[$(date)] WARNING: Timed out waiting for Uvicorn (may still be loading)"
}

# ─── Mode dispatch ───────────────────────────────────────────────────
echo "[$(date)] === launch-sglang.sh: mode=${SGLANG_MODE} on ${NODE_NAME} ==="

case "${SGLANG_MODE}" in

  # ── Mode 1: Standard Docker ──────────────────────────────────────
  docker)
    setup_rootless_docker
    pull_image_if_needed

    docker stop sglang-m2.5 2>/dev/null || true
    docker rm -f sglang-m2.5 2>/dev/null || true

    build_sglang_args "/models/MiniMaxAI/MiniMax-M2.5"

    echo "[$(date)] === Starting SGLang server (docker) on $NODE_NAME ==="
    docker run -d --init --name sglang-m2.5 \
      --gpus all --ipc=host --shm-size 32g -p "${SGLANG_PORT}:${SGLANG_PORT}" \
      -v "${MODEL_PATH}:/models/MiniMaxAI/MiniMax-M2.5:ro" \
      "${SGLANG_IMAGE}" \
      python3 -m sglang.launch_server "${SGLANG_ARGS[@]}"

    wait_for_uvicorn

    echo "[$(date)] === launch-sglang.sh entering wait ==="
    docker wait sglang-m2.5
    ;;

  # ── Mode 2: Docker + local source mount ──────────────────────────
  docker-mount)
    if [[ -z "${SGLANG_LOCAL_PATH:-}" ]]; then
      echo "FATAL: SGLANG_MODE=docker-mount requires SGLANG_LOCAL_PATH set in env_vars.sh"
      exit 1
    fi
    if [[ ! -d "${SGLANG_LOCAL_PATH}" ]]; then
      echo "FATAL: SGLANG_LOCAL_PATH=${SGLANG_LOCAL_PATH} does not exist"
      exit 1
    fi

    setup_rootless_docker
    pull_image_if_needed

    docker stop sglang-m2.5 2>/dev/null || true
    docker rm -f sglang-m2.5 2>/dev/null || true

    build_sglang_args "/models/MiniMaxAI/MiniMax-M2.5"

    echo "[$(date)] === Starting SGLang server (docker-mount) on $NODE_NAME ==="
    echo "[$(date)] Mounting local sglang from: ${SGLANG_LOCAL_PATH}"
    docker run -d --init --name sglang-m2.5 \
      --gpus all --ipc=host --shm-size 32g -p "${SGLANG_PORT}:${SGLANG_PORT}" \
      -v "${MODEL_PATH}:/models/MiniMaxAI/MiniMax-M2.5:ro" \
      -v "${SGLANG_LOCAL_PATH}:/sgl-workspace/sglang/python/sglang:ro" \
      "${SGLANG_IMAGE}" \
      python3 -m sglang.launch_server "${SGLANG_ARGS[@]}"

    wait_for_uvicorn

    echo "[$(date)] === launch-sglang.sh entering wait ==="
    docker wait sglang-m2.5
    ;;

  # ── Mode 3: Conda environment ───────────────────────────────────
  conda)
    SGLANG_CONDA_ENV="${SGLANG_CONDA_ENV:-sgl-local-editable}"

    echo "[$(date)] === Starting SGLang server (conda: ${SGLANG_CONDA_ENV}) on $NODE_NAME ==="

    # Initialize conda in this shell
    # Temporarily disable nounset — conda activate scripts may reference undefined vars
    set +u
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate "${SGLANG_CONDA_ENV}"
    set -u
    echo "[$(date)] Activated conda env: ${SGLANG_CONDA_ENV}"
    echo "[$(date)] Python: $(which python3)"
    echo "[$(date)] sglang location: $(python3 -c 'import sglang; print(sglang.__file__)' 2>/dev/null || echo 'not found')"

    build_sglang_args "${MODEL_PATH}"

    # Run in foreground — srun stays alive as long as python runs
    # PYTHONUNBUFFERED ensures real-time log output (no buffering to file)
    PYTHONUNBUFFERED=1 python3 -m sglang.launch_server "${SGLANG_ARGS[@]}"
    ;;

  # ── Mode 4: uv + local project ──────────────────────────────────
  uv)
    SGLANG_UV_PROJECT="${SGLANG_UV_PROJECT:-}"
    if [[ -z "${SGLANG_UV_PROJECT}" ]]; then
      echo "FATAL: SGLANG_MODE=uv requires SGLANG_UV_PROJECT set in env_vars.sh"
      exit 1
    fi
    if [[ ! -f "${SGLANG_UV_PROJECT}/pyproject.toml" ]]; then
      echo "FATAL: No pyproject.toml found at ${SGLANG_UV_PROJECT}/"
      exit 1
    fi

    echo "[$(date)] === Starting SGLang server (uv: ${SGLANG_UV_PROJECT}) on $NODE_NAME ==="

    build_sglang_args "${MODEL_PATH}"

    # Pin Python version if set (e.g. 3.12 — needed because some deps lack 3.13 wheels)
    UV_PYTHON_FLAG=""
    if [[ -n "${SGLANG_UV_PYTHON:-}" ]]; then
      UV_PYTHON_FLAG="--python ${SGLANG_UV_PYTHON}"
    fi

    # uv run auto-creates .venv if needed (first run installs deps, ~minutes)
    PYTHONUNBUFFERED=1 uv run --directory "${SGLANG_UV_PROJECT}" $UV_PYTHON_FLAG \
      python3 -m sglang.launch_server "${SGLANG_ARGS[@]}"
    ;;

  *)
    echo "FATAL: Unknown SGLANG_MODE='${SGLANG_MODE}'"
    echo "Valid modes: docker, docker-mount, conda, uv"
    exit 1
    ;;
esac
