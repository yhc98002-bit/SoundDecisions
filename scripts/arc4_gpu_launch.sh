#!/bin/bash
# Launch one isolated, guarded Arc-4 GPU job in a persistent remote tmux session.
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <an12|an21|an29> <session> <manifest-dir> <command>" >&2
  exit 2
fi

NODE="$1"
SESSION="$2"
MANIFEST_DIR="$3"
shift 3
COMMAND="$*"

case "$NODE" in
  an12|an21|an29) ;;
  *) echo "unauthorized Arc-4 GPU node: $NODE" >&2; exit 2 ;;
esac

REPO="/XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions-arc4-gpu"
PRIMARY="/XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions"
MIN_FREE_MIB=71680

# The query is deliberately scoped to the four authorized physical devices.
GPU_CSV="$(ssh -o BatchMode=yes "$NODE" \
  'nvidia-smi -i 4,5,6,7 --query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits')"
while IFS=',' read -r index uuid name total free util; do
  index="${index//[[:space:]]/}"
  free="${free//[[:space:]]/}"
  if [[ ! "$index" =~ ^(4|5|6|7)$ ]] || (( free < MIN_FREE_MIB )); then
    echo "GPU guard failed on $NODE: index=$index free_mib=$free" >&2
    exit 3
  fi
done <<< "$GPU_CSV"

mkdir -p "$MANIFEST_DIR"
LOG_DIR="$REPO/logs/arc4_gpu/$SESSION"
MANIFEST="$MANIFEST_DIR/${SESSION}.json"
GIT_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
PROTOCOL_SHA="$(sha256sum "$REPO/experiment/preregistered/B1_PROTOCOL.md" | awk '{print $1}')"
CONFIG_SHA="$(sha256sum "$REPO/configs/axes.json" "$REPO/configs/thresholds.json" | sha256sum | awk '{print $1}')"
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

jq -n \
  --arg node "$NODE" \
  --arg session "$SESSION" \
  --arg command "$COMMAND" \
  --arg git_commit "$GIT_COMMIT" \
  --arg protocol_sha256 "$PROTOCOL_SHA" \
  --arg config_sha256 "$CONFIG_SHA" \
  --arg started_utc "$STARTED" \
  --arg gpu_query "$GPU_CSV" \
  --arg log "$LOG_DIR/session.log" \
  '{node:$node, session:$session, command:$command, git_commit:$git_commit,
    protocol_sha256:$protocol_sha256, config_sha256:$config_sha256,
    seed:0, started_utc:$started_utc,
    placement:{physical_gpu_ids:[4,5,6,7], cuda_visible_devices:"4,5,6,7",
      tp_width:1, replica_count:4,
      rationale:"small_16k fits TP1; four independent replicas maximize throughput"},
    gpu_guard:{minimum_free_mib:71680, query:$gpu_query},
    weights:{source:"hf", offline:true}, log:$log, status:"LAUNCHED"}' \
  > "$MANIFEST"

COMMAND_B64="$(printf '%s' "$COMMAND" | base64 -w0)"
ssh -o BatchMode=yes "$NODE" bash -s -- "$SESSION" "$COMMAND_B64" "$LOG_DIR" <<'REMOTE'
set -euo pipefail
SESSION="$1"
COMMAND_B64="$2"
LOG_DIR="$3"
REPO="/XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions-arc4-gpu"
PRIMARY="/XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 4
fi
mkdir -p "$LOG_DIR"
printf '%s' "$COMMAND_B64" | base64 -d > "$LOG_DIR/command.sh"
chmod 700 "$LOG_DIR/command.sh"
cat > "$LOG_DIR/runner.sh" <<RUNNER
#!/bin/bash
set -euo pipefail
cd "$REPO"
export PATH="$PRIMARY/.venv/bin:\$PATH"
export PYTHONPATH="$PRIMARY/third_party/MMAudio:\${PYTHONPATH:-}"
export FOLEY_CW_MMAUDIO_ROOT="$PRIMARY/third_party/MMAudio"
export FOLEY_CW_WEIGHTS_SOURCE=hf
export HF_HOME="$PRIMARY/.hf_cache"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=4
echo "node=\$(hostname) session=$SESSION python=\$(command -v python) CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"
set +e
bash "$LOG_DIR/command.sh"
rc=\$?
set -e
printf '%s\n' "\$rc" > "$LOG_DIR/exit_code"
echo "session=$SESSION exit_code=\$rc"
exit "\$rc"
RUNNER
chmod 700 "$LOG_DIR/runner.sh"
tmux new-session -d -s "$SESSION" "bash '$LOG_DIR/runner.sh' > '$LOG_DIR/session.log' 2>&1"
echo "launched tmux=$SESSION log=$LOG_DIR/session.log"
REMOTE

printf '%s\n' "$GPU_CSV"
echo "manifest=$MANIFEST"
echo "tmux=$NODE:$SESSION"
echo "log=$LOG_DIR/session.log"
