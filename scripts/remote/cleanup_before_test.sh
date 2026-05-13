#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-2335}"

echo "### gpu apps before cleanup"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
else
  echo "nvidia-smi not found"
fi

echo "### cleanup fixed distributed port ${PORT}"
if command -v lsof >/dev/null 2>&1; then
  pids="$(lsof -tiTCP:${PORT} -sTCP:LISTEN || true)"
  if [ -n "${pids}" ]; then
    echo "killing fixed-port listener pids: ${pids}"
    kill ${pids} || true
    sleep 2
  else
    echo "no listener on port ${PORT}"
  fi
else
  echo "lsof not found; skip port cleanup"
fi

echo "### gpu apps after cleanup"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
fi
