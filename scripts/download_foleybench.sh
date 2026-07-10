#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REPO_ID="${REPO_ID:-FoleyBench/foleybench}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/data/FoleyBench/foleybench}"
HF_CLI="${HF_CLI:-$ROOT_DIR/.venv/bin/hf}"
HF_HOME="${HF_HOME:-$ROOT_DIR/.hf_cache}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-8}"
PROXY_SCRIPT="${PROXY_SCRIPT:-/APP/u22/ai_x86/toolshs/setproxy.sh}"
PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-7890}"

if [[ -f "$PROXY_SCRIPT" ]]; then
  # shellcheck source=/dev/null
  source "$PROXY_SCRIPT" "$PROXY_HOST" "$PROXY_PORT" >/dev/null 2>&1 || true
fi
proxy_url="http://${PROXY_HOST}:${PROXY_PORT}"
export http_proxy="$proxy_url"
export https_proxy="$proxy_url"
export HTTP_PROXY="$proxy_url"
export HTTPS_PROXY="$proxy_url"
export all_proxy="$proxy_url"
export ALL_PROXY="$proxy_url"

if [[ ! -x "$HF_CLI" ]]; then
  cat >&2 <<EOF
Missing hf CLI: $HF_CLI

This project already has huggingface_hub in .venv. If the CLI is still missing, run:
  $ROOT_DIR/.venv/bin/python -m pip install -U huggingface_hub
EOF
  exit 1
fi

mkdir -p "$OUT_DIR" "$HF_HOME" "$HF_HUB_CACHE"
export HF_HOME HF_HUB_CACHE HF_HUB_DISABLE_XET

if [[ -z "${HF_TOKEN:-}" ]]; then
  if ! "$HF_CLI" auth whoami --quiet >/dev/null 2>&1; then
    cat >&2 <<EOF
No Hugging Face token is available.

FoleyBench/foleybench is gated. First approve access in the browser:
  https://huggingface.co/datasets/FoleyBench/foleybench

Then either log in for this project:
  cd "$ROOT_DIR"
  source "$PROXY_SCRIPT" "$PROXY_HOST" "$PROXY_PORT"
  HF_HOME="$HF_HOME" "$HF_CLI" auth login

Or run this script with a temporary token:
  HF_TOKEN=hf_xxx scripts/download_foleybench.sh
EOF
    exit 2
  fi
fi

args=(
  download "$REPO_ID"
  --repo-type dataset
  --local-dir "$OUT_DIR"
  --max-workers "$HF_MAX_WORKERS"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry-run)
fi

"$HF_CLI" "${args[@]}"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  echo
  echo "Downloaded to: $OUT_DIR"
  find "$OUT_DIR" -path "$OUT_DIR/.cache" -prune -o -type f -print | wc -l | awk '{print "Main file count: " $1}'
  du -sh "$OUT_DIR" 2>/dev/null | awk '{print "Disk usage: " $1}'
fi
