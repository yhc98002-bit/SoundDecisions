#!/usr/bin/env bash
set -u

ROOT="/XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/claude_continue_watchdog_${RUN_ID}.log"
STATE="$LOG_DIR/claude_continue_watchdog_latest.state"
LOCK="$LOG_DIR/claude_continue_watchdog.lock"

TARGETS=("audio:0.0" "SoundDecisions:0.0")
TARGET_TIME="${TARGET_TIME:-01:11:00}"
IDLE_POLL_SECONDS=900
CHECK_SECONDS=300
POLL_SECONDS=15
MAX_ATTEMPTS=3

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z %z')" "$*" | tee -a "$LOG"
}

target_epoch() {
  local today_target
  today_target="$(date -d "$(date +%F) ${TARGET_TIME}" +%s)"
  local now
  now="$(date +%s)"
  if (( now < today_target )); then
    printf '%s\n' "$today_target"
  else
    date -d "tomorrow ${TARGET_TIME}" +%s
  fi
}

idle_until() {
  local wake="$1"
  local now
  local remaining
  local chunk

  while true; do
    now="$(date +%s)"
    remaining=$((wake - now))
    if (( remaining <= 0 )); then
      break
    fi

    chunk="$IDLE_POLL_SECONDS"
    if (( remaining < chunk )); then
      chunk="$remaining"
    fi

    log "idle poll: ${remaining}s until $(date -d "@$wake" '+%Y-%m-%d %H:%M:%S %Z %z')"
    sleep "$chunk"
  done

  log "wake time reached"
}

pane_hash() {
  local target="$1"
  tmux capture-pane -t "$target" -p -S -200 2>/dev/null | sha256sum | awk '{print $1}'
}

pane_tail() {
  local target="$1"
  tmux capture-pane -t "$target" -p -S -12 2>/dev/null | tail -12
}

send_continue() {
  local target="$1"
  log "sending continue to $target"
  tmux send-keys -t "$target" 'continue' C-m
}

wait_for_refresh() {
  local target="$1"
  local before="$2"
  local elapsed=0
  local after

  while (( elapsed < CHECK_SECONDS )); do
    sleep "$POLL_SECONDS"
    elapsed=$((elapsed + POLL_SECONDS))
    after="$(pane_hash "$target" || true)"
    if [[ -n "$after" && "$after" != "$before" ]]; then
      log "$target refreshed after ${elapsed}s"
      return 0
    fi
  done

  log "$target did not refresh after ${CHECK_SECONDS}s"
  return 1
}

main() {
  exec 9>"$LOCK"
  if ! flock -n 9; then
    log "another watchdog is already running; exiting"
    exit 0
  fi

  {
    echo "pid=$$"
    echo "log=$LOG"
    echo "target_time=$TARGET_TIME"
    echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %Z %z')"
  } > "$STATE"

  local wake
  wake="$(target_epoch)"
  local now
  now="$(date +%s)"
  local sleep_for=$((wake - now))
  if (( sleep_for < 0 )); then
    sleep_for=0
  fi

  log "watchdog armed for $(date -d "@$wake" '+%Y-%m-%d %H:%M:%S %Z %z')"
  idle_until "$wake"

  local target
  local active_targets=()
  for target in "${TARGETS[@]}"; do
    local pane_id
    local pane_cmd
    if ! pane_id="$(tmux display-message -p -t "$target" '#{pane_id}' 2>/dev/null)"; then
      log "$target missing; skipping"
    else
      pane_cmd="$(tmux display-message -p -t "$target" '#{pane_current_command}' 2>/dev/null || true)"
      log "$target found as $pane_id current_command=${pane_cmd:-unknown}"
      active_targets+=("$target")
    fi
  done

  if (( ${#active_targets[@]} == 0 )); then
    log "no active targets found; watchdog finished"
    return 0
  fi

  local pending=("${active_targets[@]}")
  local attempt=1
  while (( attempt <= MAX_ATTEMPTS && ${#pending[@]} > 0 )); do
    local -A post_send_hash=()
    log "attempt ${attempt}/${MAX_ATTEMPTS}; pending targets: ${pending[*]}"

    for target in "${pending[@]}"; do
      local before
      before="$(pane_hash "$target" || true)"
      log "$target before_hash=${before:-missing}"
      send_continue "$target"
    done

    sleep 5

    for target in "${pending[@]}"; do
      local post_send
      post_send="$(pane_hash "$target" || true)"
      post_send_hash["$target"]="$post_send"
      log "$target post_send_hash=${post_send:-missing}; now watching for a real refresh"
    done

    local elapsed=0
    local still_pending=("${pending[@]}")
    while (( elapsed < CHECK_SECONDS && ${#still_pending[@]} > 0 )); do
      sleep "$POLL_SECONDS"
      elapsed=$((elapsed + POLL_SECONDS))

      local next_pending=()
      for target in "${still_pending[@]}"; do
        local after
        after="$(pane_hash "$target" || true)"
        if [[ -n "$after" && "$after" != "${post_send_hash[$target]}" ]]; then
          log "$target refreshed after ${elapsed}s"
          log "$target tail after refresh:"
          pane_tail "$target" | sed 's/^/    /' | tee -a "$LOG" >/dev/null
        else
          next_pending+=("$target")
        fi
      done

      still_pending=("${next_pending[@]}")
    done

    pending=()
    for target in "${still_pending[@]}"; do
      log "$target did not refresh after ${CHECK_SECONDS}s"
      pending+=("$target")
    done

    if (( ${#pending[@]} > 0 )); then
      log "retrying targets because pane did not refresh: ${pending[*]}"
    fi
    attempt=$((attempt + 1))
  done

  if (( ${#pending[@]} > 0 )); then
    for target in "${pending[@]}"; do
      log "$target failed to show refresh after ${MAX_ATTEMPTS} attempts"
      log "$target final tail:"
      pane_tail "$target" | sed 's/^/    /' | tee -a "$LOG" >/dev/null
    done
  fi

  log "watchdog finished"
}

main "$@"
