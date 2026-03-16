#!/bin/bash
# Warm up Docker images on all worker nodes.
# First time takes ~30-60 min. Subsequent runs are fast (images cached per-node).
# Docker images are shared across users on the same node — if someone already
# warmed a node, you can skip it.
#
# Usage:
#   bash warmup.sh           # all 5 nodes
#   bash warmup.sh 002 008   # specific nodes only

set -euo pipefail

HARBOR=~/workspace/envs/harbor/bin/harbor
ALL_NODES=(
    research-dev-coder-002
    research-dev-coder-008
    research-dev-coder-011
    research-dev-coder-014
    research-dev-coder-015
)

if [ $# -gt 0 ]; then
    NODES=()
    for n in "$@"; do
        NODES+=("research-dev-coder-$n")
    done
else
    NODES=("${ALL_NODES[@]}")
fi

NODE_ARGS=""
for node in "${NODES[@]}"; do
    NODE_ARGS="$NODE_ARGS --nodes $node"
done

echo "=== Warming up ${#NODES[@]} nodes ==="
echo "Nodes: ${NODES[*]}"
echo ""

$HARBOR warmup pull -d swebench-verified -n 16 $NODE_ARGS
