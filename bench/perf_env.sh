#!/usr/bin/env bash
set -euo pipefail
source /home/chen/projects/dzyy-vllm-local/.venv-cu128/bin/activate
export PATH="/opt/nvidia/nsight-systems-cli/2026.4.1/bin:$PATH"
cd /mnt/d/Project/dzyy-vllm/learn-vllm
exec "$@"
