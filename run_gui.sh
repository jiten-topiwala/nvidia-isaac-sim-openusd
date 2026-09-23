#!/bin/bash
# GUI (windowed) launch with the Carbonite BOOT-HANG watchdog. Kit sometimes deadlocks at
# "Passing the following args to the base kit application" (all threads in a futex, no kit log ever
# created) after a previous run's kill/exit. A retry boots fine. This wraps the normal windowed
# launch: it starts the sim, watches the log for "app ready", and if the boot hangs it kills and
# retries -- then, once booted, streams the sim's output so it behaves like a normal foreground run.
#
#   ./run_gui.sh                      # friction-only pick and place -- the only mode
#   ARM_PLANNER=0 ./run_gui.sh        # skip the planned reach (nothing is flown in its place)
#
# NO TUNING FLAGS ARE NEEDED. Every value the verified cycles were measured with is a default in
# morph/config.py (PROFILE), and drive mode is on by default. Verified 2026-09-06 on defaults alone: object 0 to
# slot 4, three pads 10.25/8.08/8.12 N, lift +249 mm, released 3.0 mm, placed 4 mm, ALL CHECKS PASS
# — identical to the long runbook command it replaces. `ARM_DRIVE=0` restores the kinematic path.
#
# Any other environment variable can still be passed on the front of the command; an explicit value
# always beats a default.
ISAAC="${ISAAC:-$HOME/isaac-sim/python.sh}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export OMPL_PYTHON="${OMPL_PYTHON:-$HERE/.venv/bin/python3}"
# RRTConnect resamples every run. Pinned here for the same reason NAV_SEED is: two runs of the same
# cycle should take the same route. Unset it for genuinely independent runs.
export ARM_SEED="${ARM_SEED:-1}"
LOG="${MORPH_LOG:-$HERE/logs/live.log}"
# play_isaac defaults this to logs/place_results.json and verify_place looks next to the LOG, so
# under MORPH_LOG the writer and the reader use different files. Pin both to this run's log dir.
export PLACE_RESULTS="${PLACE_RESULTS:-$(dirname "$LOG")/place_results.json}"
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
  env NAV_SEED="${NAV_SEED:-0}" PYTHONUNBUFFERED=1 \
    "$ISAAC" "$HERE/play_isaac.py" >> "$LOG" 2>&1 < /dev/null &
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
