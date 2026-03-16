#!/bin/bash
# Verify Docker and SSH access on all worker nodes.
# Run from the coordinator (research-dev-coder-009).
#
# Usage:
#   bash check-nodes.sh

set -euo pipefail

NODES=(
    research-dev-coder-002
    research-dev-coder-008
    research-dev-coder-011
    research-dev-coder-014
    research-dev-coder-015
)

echo "=== Checking ${#NODES[@]} worker nodes ==="
echo ""

ALL_OK=true
for node in "${NODES[@]}"; do
    printf "%-30s" "$node:"

    # SSH check
    if ! ssh -o ConnectTimeout=5 "$node" "true" 2>/dev/null; then
        echo "FAIL — SSH failed. Run: kinit"
        ALL_OK=false
        continue
    fi

    # Docker check
    DOCKER_OK=$(ssh -o ConnectTimeout=5 "$node" "docker info --format '{{.ContainersRunning}}' 2>/dev/null" 2>/dev/null)
    if [ -z "$DOCKER_OK" ]; then
        echo "FAIL — Docker permission denied. Run on that node: sudo usermod -aG docker $USER && newgrp docker"
        ALL_OK=false
        continue
    fi

    # Image count
    IMAGES=$(ssh -o ConnectTimeout=5 "$node" "docker images --format '{{.Repository}}' | grep -c 'sweb.eval\|hb__'" 2>/dev/null || echo "0")

    echo "OK (docker running=$DOCKER_OK, images=$IMAGES)"
done

echo ""
if [ "$ALL_OK" = true ]; then
    echo "All nodes ready."
else
    echo "Some nodes failed. Fix the issues above before running."
    exit 1
fi
