#!/bin/bash
# Single-SSH-session GPU job wrapper (validated pattern; see memory mmaudio-gpu-env-recipe).
#
#   scripts/run_on_node.sh <node> <command...>
#
# Stages .venv -> /dev/shm/foley_venv (Lustre cold-import is >400s; /dev/shm ~1s,
# but RemoveIPC wipes it when the last SSH session ends, so stage+run happens in
# ONE session). Exports the offline HF env. The command runs from the repo root
# with the staged venv's python first on PATH.
#
# Multi-GPU sharding example (8 workers in one session):
#   scripts/run_on_node.sh an17 'for i in $(seq 0 7); do \
#     CUDA_VISIBLE_DEVICES=$i python scripts/stage_m_micromap.py --shard $i/8 \
#       > logs/stage_m_shard$i.log 2>&1 & done; wait'
set -euo pipefail

NODE="$1"; shift
REPO="/XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions"
CMD="$*"

ssh -o BatchMode=yes "$NODE" bash -s <<REMOTE
set -euo pipefail
cd "$REPO"
echo "=== [\$(hostname)] staging venv -> /dev/shm/foley_venv ==="
mkdir -p /dev/shm/foley_venv
# flock: concurrent sessions on the same node must not race the --delete rsync
flock /dev/shm/foley_venv.lock rsync -a --delete .venv/ /dev/shm/foley_venv/
export PATH="/dev/shm/foley_venv/bin:\$PATH"
export HF_HOME="$REPO/.hf_cache" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
echo "=== [\$(hostname)] python=\$(command -v python) ==="
$CMD
echo "=== [\$(hostname)] DONE (exit 0) ==="
REMOTE
