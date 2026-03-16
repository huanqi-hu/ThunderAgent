#!/usr/bin/env bash
# install-deps.sh — Create virtualenv and install harbor + ThunderAgent + analysis tools.
#
# Usage:
#   bash scripts/setup/install-deps.sh
#
# This creates HARBOR_ENV_DIR (default: <repo_root>/harbor-env) with:
#   - harbor (editable install from repo root)
#   - ThunderAgent (from ThunderAgent repo root or THUNDERAGENT_PATH)
#   - pandas, matplotlib (for analysis scripts)

set -euo pipefail

# ─── Resolve repo root from this script's location ───────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source env_vars.sh if it exists; otherwise use defaults
if [[ -f "$REPO_ROOT/env_vars.sh" ]]; then
  source "$REPO_ROOT/env_vars.sh"
else
  echo "WARNING: $REPO_ROOT/env_vars.sh not found, using defaults"
fi

HARBOR_ENV_DIR="${HARBOR_ENV_DIR:-$REPO_ROOT/harbor-env}"  # should already be set by env_vars.sh

# ─── Detect ThunderAgent path ────────────────────────────────────────
if [[ -z "${THUNDERAGENT_PATH:-}" ]]; then
  # Auto-detect: this example lives at ThunderAgent/examples/datagen/harbor/
  # so the ThunderAgent repo root is 3 levels up.
  CANDIDATE="$(cd "$REPO_ROOT/../../.." && pwd)"
  if [[ -f "$CANDIDATE/pyproject.toml" ]] && grep -q 'name = "ThunderAgent"' "$CANDIDATE/pyproject.toml" 2>/dev/null; then
    THUNDERAGENT_PATH="$CANDIDATE"
  else
    echo "WARNING: ThunderAgent repo root not found at $CANDIDATE"
    echo "  Set THUNDERAGENT_PATH in env_vars.sh or install manually."
    THUNDERAGENT_PATH=""
  fi
fi

echo "============================================================"
echo " Installing dependencies"
echo "============================================================"
echo " Repo root:        $REPO_ROOT"
echo " Virtualenv:       $HARBOR_ENV_DIR"
echo " ThunderAgent:     ${THUNDERAGENT_PATH:-<not found>}"
echo "============================================================"
echo ""

# ─── 1. Create virtualenv ────────────────────────────────────────────
if [[ ! -d "$HARBOR_ENV_DIR" ]]; then
  echo "[$(date)] Creating virtualenv at $HARBOR_ENV_DIR..."
  uv venv "$HARBOR_ENV_DIR" --python 3.12
else
  echo "[$(date)] Virtualenv already exists at $HARBOR_ENV_DIR"
fi

PYTHON="$HARBOR_ENV_DIR/bin/python"
PIP="$HARBOR_ENV_DIR/bin/pip"

# ─── 2. Install harbor (editable) ────────────────────────────────────
echo ""
echo "[$(date)] Installing harbor (editable)..."
uv pip install -e "$REPO_ROOT" --python "$PYTHON"

# Verify harbor is available
if "$HARBOR_ENV_DIR/bin/harbor" --version >/dev/null 2>&1; then
  echo "[$(date)] harbor version: $("$HARBOR_ENV_DIR/bin/harbor" --version)"
else
  echo "[$(date)] WARNING: harbor CLI not found after install"
fi

# ─── 3. Install ThunderAgent ─────────────────────────────────────────
if [[ -n "$THUNDERAGENT_PATH" && -d "$THUNDERAGENT_PATH" ]]; then
  echo ""
  echo "[$(date)] Installing ThunderAgent from $THUNDERAGENT_PATH..."
  uv pip install -e "$THUNDERAGENT_PATH" --python "$PYTHON"
  echo "[$(date)] ThunderAgent installed"
else
  echo ""
  echo "[$(date)] Skipping ThunderAgent install (path not set)"
fi

# ─── 4. Install analysis dependencies ────────────────────────────────
echo ""
echo "[$(date)] Installing analysis dependencies (pandas, matplotlib)..."
uv pip install pandas matplotlib --python "$PYTHON"

# ─── 5. Summary ──────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Installation complete"
echo "============================================================"
echo " Activate with: source $HARBOR_ENV_DIR/bin/activate"
echo ""
echo " Installed packages:"
"$PIP" list --format=columns 2>/dev/null | grep -iE 'harbor|thunder|pandas|matplotlib' || true
echo "============================================================"
