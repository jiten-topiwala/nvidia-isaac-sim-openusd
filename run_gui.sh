#!/bin/bash
# GUI (windowed) launch with the Carbonite BOOT-HANG watchdog. Kit sometimes deadlocks at
# "Passing the following args to the base kit application" (all threads in a futex, no kit log ever
# created) after a previous run's kill/exit. A retry boots fine. This wraps the normal windowed
# launch: it starts the sim, watches the log for "app ready", and if the boot hangs it kills and
# retries -- then, once booted, streams the sim's output so it behaves like a normal foreground run.
#
#   ./run_gui.sh                      # practical mode (default): the verified demo
#   GRIP_MODE=perfect ./run_gui.sh    # perfect mode: real pad friction, does not lift yet
#
# Any other environment variable can be passed on the front of the command as usual.
ISAAC="${ISAAC:-$HOME/isaac-sim/python.sh}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export OMPL_PYTHON="${OMPL_PYTHON:-$HERE/.venv/bin/python3}"
LOG="${MORPH_LOG:-$HERE/logs/live.log}"
mkdir -p "$(dirname "$LOG")"          # logs/ is gitignored; override with MORPH_LOG
for attempt in 1 2 3 4; do
  # sweep stale carb shm from dead owners (and the global sem) before booting
  rm -f /dev/shm/sem.carbonite-sharedmemory
  for f in /dev/shm/carb-* /dev/shm/sem.carb-*; do [ -e "$f" ] || continue
    pid=$(echo "$f" | grep -oE -- '-[0-9]+$' | tr -dc '0-9'); [ -z "$pid" ] && continue
    kill -0 "$pid" 2>/dev/null || rm -f "$f"; done
  : > "$LOG"
  # NOT setsid: keep it in this terminal's process group so Ctrl-C reaches it. Trap INT to kill the
  # sim (and the tail) on Ctrl-C -- the earlier setsid version detached it and Ctrl-C did nothing.
  env NAV_SEED="${NAV_SEED:-0}" PYTHONUNBUFFERED=1 "$ISAAC" "$HERE/play_isaac.py" >> "$LOG" 2>&1 < /dev/null &
  PID=$!
  trap 'kill -INT $PID 2>/dev/null; sleep 1; kill -9 $PID 2>/dev/null; exit 130' INT
  ok=0
  for i in $(seq 1 24); do
    sleep 5
    grep -q "app ready" "$LOG" 2>/dev/null && { ok=1; break; }
    kill -0 "$PID" 2>/dev/null || break        # process died on its own
  done
  if [ "$ok" = 1 ]; then
    echo ">>> booted on attempt $attempt -- streaming; Ctrl-C to stop"
    tail -n +1 -f "$LOG" --pid "$PID"
    exit 0
  fi
  echo ">>> attempt $attempt: boot hang at Kit init -- killing and retrying"
  pkill -9 -f "kit/python/bin/python3 $HERE/play_isaac.py" 2>/dev/null; sleep 8
done
echo ">>> FAILED to boot after 4 attempts"; exit 1
