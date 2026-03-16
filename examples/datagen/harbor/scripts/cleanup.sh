#!/bin/bash
# Kill harbor processes and clean up Docker containers
echo "Killing harbor processes..."
pkill -f "harbor run" 2>/dev/null || true

echo "Stopping containers..."
docker ps -q 2>/dev/null | xargs -r docker rm -f 2>/dev/null | wc -l
echo "containers removed"
