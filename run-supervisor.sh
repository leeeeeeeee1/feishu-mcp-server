#!/usr/bin/env bash
# Supervisor Hub daemon - auto-restart on crash + auto-reload on code change
# Usage: ./run-supervisor.sh
# Stop:  kill $(cat /tmp/feishu-supervisor.pid)

set -u

PIDFILE="/tmp/feishu-supervisor.pid"
CHILD_PIDFILE="/tmp/feishu-supervisor-child.pid"
LOGFILE="/var/log/feishu-supervisor.log"
WATCH_DIR="/workspace/feishu-mcp-server/src"
MAX_RESTARTS=1000
RESTART_DELAY=3

export FEISHU_APP_ID="${FEISHU_APP_ID:-cli_a93b9f3cabf8dcef}"
export FEISHU_APP_SECRET="${FEISHU_APP_SECRET:-uJLhFsa09Fwj7M0WEpOLQbEngjap55GI}"
export PYTHONPATH="/workspace/feishu-mcp-server/src"

# Feishu domains bypass proxy for stable WebSocket connections
export NO_PROXY="open.feishu.cn,msg-frontier.feishu.cn,internal-api-lark-api.feishu.cn"
export no_proxy="$NO_PROXY"

# Kill existing instances (both daemon wrapper and any orphan python processes)
if [ -f "$PIDFILE" ]; then
    old_pid=$(cat "$PIDFILE")
    kill "$old_pid" 2>/dev/null && echo "Killed old daemon (PID $old_pid)"
fi
if [ -f "$CHILD_PIDFILE" ]; then
    old_child=$(cat "$CHILD_PIDFILE")
    kill "$old_child" 2>/dev/null && echo "Killed old child (PID $old_child)"
fi
# Kill any orphan supervisor.main processes
pkill -f "python3 -m supervisor.main" 2>/dev/null && echo "Killed orphan supervisor processes"
sleep 1

echo $$ > "$PIDFILE"

cleanup() {
    if [ -f "$CHILD_PIDFILE" ]; then
        kill $(cat "$CHILD_PIDFILE") 2>/dev/null
        rm -f "$CHILD_PIDFILE"
    fi
    [ -n "${WATCHER_PID:-}" ] && kill "$WATCHER_PID" 2>/dev/null
    rm -f "$PIDFILE"
    echo "$(date) Supervisor daemon stopped." | tee -a "$LOGFILE"
    exit 0
}
trap cleanup INT TERM

# --- File watcher: touches a trigger file when source changes ---
TRIGGER="/tmp/feishu-supervisor-reload"
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
echo "$(date) Supervisor Hub daemon starting (PID $$)" | tee -a "$LOGFILE"

restarts=0
while [ $restarts -lt $MAX_RESTARTS ]; do
    echo "$(date) Starting supervisor (attempt $((restarts + 1)))" | tee -a "$LOGFILE"

    python3 -m supervisor.main >> "$LOGFILE" 2>&1 &
    child_pid=$!
    echo $child_pid > "$CHILD_PIDFILE"

    while true; do
        if ! kill -0 "$child_pid" 2>/dev/null; then
            wait "$child_pid" 2>/dev/null
            exit_code=$?
            echo "$(date) Supervisor exited with code $exit_code" | tee -a "$LOGFILE"
            break
        fi

        if [ -f "$TRIGGER" ]; then
            rm -f "$TRIGGER"
            echo "$(date) Code change detected, reloading supervisor..." | tee -a "$LOGFILE"
            kill "$child_pid" 2>/dev/null
            wait "$child_pid" 2>/dev/null
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
