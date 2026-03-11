#!/usr/bin/env bash
# Feishu Bot daemon - auto-restart on crash + auto-reload on code change
# Usage: ./run-bot.sh
# Stop:  kill $(cat /tmp/feishu-bot.pid)

set -u

PIDFILE="/tmp/feishu-bot.pid"
BOT_PIDFILE="/tmp/feishu-bot-child.pid"
LOGFILE="/var/log/feishu-bot.log"
WATCH_DIR="/workspace/feishu-mcp-server/src"
MAX_RESTARTS=1000
RESTART_DELAY=3

export FEISHU_APP_ID="${FEISHU_APP_ID:-cli_a93b9f3cabf8dcef}"
export FEISHU_APP_SECRET="${FEISHU_APP_SECRET:-uJLhFsa09Fwj7M0WEpOLQbEngjap55GI}"
export PYTHONPATH="/workspace/feishu-mcp-server/src"

# Kill existing instance
if [ -f "$PIDFILE" ]; then
    old_pid=$(cat "$PIDFILE")
    kill "$old_pid" 2>/dev/null && echo "Killed old daemon (PID $old_pid)"
    sleep 1
fi

echo $$ > "$PIDFILE"

cleanup() {
    # Kill bot child process
    if [ -f "$BOT_PIDFILE" ]; then
        kill $(cat "$BOT_PIDFILE") 2>/dev/null
        rm -f "$BOT_PIDFILE"
    fi
    # Kill file watcher
    [ -n "${WATCHER_PID:-}" ] && kill "$WATCHER_PID" 2>/dev/null
    rm -f "$PIDFILE"
    echo "$(date) Daemon stopped." | tee -a "$LOGFILE"
    exit 0
}
trap cleanup INT TERM

# --- File watcher: touches a trigger file when source changes ---
TRIGGER="/tmp/feishu-bot-reload"
rm -f "$TRIGGER"

python3 -c "
import sys, time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class Handler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            print(f'[WATCH] Changed: {event.src_path}', flush=True)
            open('$TRIGGER', 'w').close()

observer = Observer()
observer.schedule(Handler(), '$WATCH_DIR', recursive=True)
observer.start()
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
" >> "$LOGFILE" 2>&1 &
WATCHER_PID=$!
echo "$(date) File watcher started (PID $WATCHER_PID)" | tee -a "$LOGFILE"

# --- Main loop ---
echo "$(date) Feishu Bot daemon starting (PID $$)" | tee -a "$LOGFILE"

restarts=0
while [ $restarts -lt $MAX_RESTARTS ]; do
    echo "$(date) Starting bot (attempt $((restarts + 1)))" | tee -a "$LOGFILE"

    python3 -m feishu_mcp.bot >> "$LOGFILE" 2>&1 &
    bot_pid=$!
    echo $bot_pid > "$BOT_PIDFILE"

    # Wait for either: bot exits, or file change trigger
    while true; do
        # Check if bot is still alive
        if ! kill -0 "$bot_pid" 2>/dev/null; then
            wait "$bot_pid" 2>/dev/null
            exit_code=$?
            echo "$(date) Bot exited with code $exit_code" | tee -a "$LOGFILE"
            break
        fi

        # Check for file change trigger
        if [ -f "$TRIGGER" ]; then
            rm -f "$TRIGGER"
            echo "$(date) Code change detected, reloading bot..." | tee -a "$LOGFILE"
            kill "$bot_pid" 2>/dev/null
            wait "$bot_pid" 2>/dev/null
            break
        fi

        sleep 1
    done

    restarts=$((restarts + 1))
    echo "$(date) Restarting in ${RESTART_DELAY}s..." | tee -a "$LOGFILE"
    sleep $RESTART_DELAY
done

echo "$(date) Max restarts ($MAX_RESTARTS) reached." | tee -a "$LOGFILE"
cleanup
