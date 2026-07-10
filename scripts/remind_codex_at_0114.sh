#!/usr/bin/env bash
set -u

ROOT="/XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
REMINDER_NAME="${REMINDER_NAME:-codex_0114_reminder}"
LOG="$LOG_DIR/${REMINDER_NAME}_${RUN_ID}.log"
STATE="$LOG_DIR/${REMINDER_NAME}_latest.state"
LOCK="$LOG_DIR/${REMINDER_NAME}.lock"
DONE="$LOG_DIR/${REMINDER_NAME}.done"

TARGET_TIME="${TARGET_TIME:-01:14:00}"
POPUP_SECONDS="${POPUP_SECONDS:-300}"
MESSAGE_MS="${MESSAGE_MS:-600000}"
TARGET_CLIENT_SESSION="${TARGET_CLIENT_SESSION:-codex}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z %z')" "$*" | tee -a "$LOG"
}

target_epoch() {
  if [[ "${REMINDER_NOW:-0}" == "1" ]]; then
    date +%s
    return 0
  fi

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

notify_display_messages() {
  local message="$1"
  local client
  tmux list-clients -F '#{client_tty} #{client_session}' 2>/dev/null | awk -v target="$TARGET_CLIENT_SESSION" '$2==target{print $1}' | while IFS= read -r client; do
    if [[ -n "$client" ]]; then
      tmux display-message -c "$client" -d "$MESSAGE_MS" "$message" 2>/dev/null || true
      log "display-message sent to client $client"
    fi
  done
}

notify_codex_popup() {
  local message="$1"
  local codex_client
  codex_client="$(tmux list-clients -F '#{client_tty} #{client_session}' 2>/dev/null | awk -v target="$TARGET_CLIENT_SESSION" '$2==target{print $1; exit}')"
  if [[ -z "$codex_client" ]]; then
    log "no attached $TARGET_CLIENT_SESSION client found; popup skipped"
    return 0
  fi

  tmux display-popup \
    -c "$codex_client" \
    -w 90% \
    -h 40% \
    -T "Codex 01:14 Reminder" \
    "printf '%s\n\n' '$message'; printf '%s\n' 'Action: ask Codex to check logs/claude_continue_watchdog_latest.state and verify audio/SoundDecisions continued.'; printf '\nThis popup will close automatically in ${POPUP_SECONDS}s.\n'; sleep ${POPUP_SECONDS}" \
    2>/dev/null || true
  log "popup sent to codex client $codex_client"
}

main() {
  exec 9>"$LOCK"
  if ! flock -n 9; then
    log "another 01:14 reminder is already running; exiting"
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

  log "reminder armed for $(date -d "@$wake" '+%Y-%m-%d %H:%M:%S %Z %z')"
  log "sleeping ${sleep_for}s"
  sleep "$sleep_for"

  local message
  message="01:14 CST reminder: check audio and SoundDecisions now. Verify claude_continue_watchdog sent continue + Enter to both panes and double-check both panes really continued/refreshed."
  log "$message"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z %z')" "$message" > "$DONE"

  notify_display_messages "$message"
  notify_codex_popup "$message"

  log "01:14 reminder finished"
}

main "$@"
