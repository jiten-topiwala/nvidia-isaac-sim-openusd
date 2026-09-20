"""Planning a route for the arm: the OMPL child, its start pose, and the goal set it solves to."""
import json
import os
import subprocess
import threading
import time
from collections import namedtuple

import numpy as np

from morph.arm.obstacles import arm_blame, planned_obstacles
from morph.arm.model import ArmModel
from morph.arm.collision import MARGIN, CHASSIS_PAD_LINKS, CHASSIS_PAD_M, Q8_TOL_M, Q8_TOL_RAD
from morph.config import ARM_MODEL, ARM_PLAN, CLOSURE_LUT, VENV_PY

PLAN_TIMEOUT = 120     # seconds before the planner child is declared stuck (arm_plan caps its solve well under)
# The one child degrade this side must COUNT: it crosses the process boundary as text, so this must
# stay a substring of what `arm_plan._densify` writes.
TUNNEL_DEGRADE = "emitted segments can tunnel"


def plan_start(demo):
    """`(q8, source)`: the commanded pose under drives when it is within the same-pose band
    of the measured one (droop puts a cleared halt inside the margin), else measured."""
    meas = np.asarray(demo._arm_q8(), float)
    cmd = getattr(demo, "_drv_prev", None) if demo._drive_on() else None
    if cmd is None:
        return meas, "measured"
    cmd8 = np.array([float(cmd[demo.idx[n]]) for n in ArmModel.Q8])
    rm, rr = ArmModel.q8_resid(cmd8, meas)
    if rm <= Q8_TOL_M and rr <= Q8_TOL_RAD:
        print(f">>> plan start: COMMANDED pose (measured trails it by {rm * 1000:.2f}mm / "
              f"{rr * 1000:.2f}mrad, inside the {Q8_TOL_M * 1000:.0f}mm/{Q8_TOL_RAD * 1000:.0f}mrad "
              f"band)", flush=True)
        return cmd8, "commanded"
    print(f">>> plan start: MEASURED pose -- the commanded one is {rm * 1000:.1f}mm / "
          f"{rr * 1000:.1f}mrad away, outside the band, so it is not where the arm is",
          flush=True)
    return meas, "measured"


def try_budgets(solve_time, tries):
    """The solve budget each try gets: `solve_time`, then one more of it per retry."""
    return [solve_time * (1 + t) for t in range(tries)]


def child_env():
    """The planner child's environment. Isaac leaks its paths into children: scrubbed."""
    return {k: v for k, v in os.environ.items()
            if k not in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH")}


_WORKER = None
_SEQ = 0
# What `run_child` hands back: the two channels its callers read. A CompletedProcess would have to
# invent an argv and a return code the worker does not have.
Reply = namedtuple("Reply", "stdout stderr")
# CONTRACT with `arm_plan.answer`: its event record is the LAST stderr line of a reply, so it frames
# the answer. stdout goes unread until then, which holds only while the child's chatter is bounded.
EVENT_LINE = "[arm_plan] event "


def worker():
    """The one planner child, spawned on first use and after a crash or a timeout kill."""
    global _WORKER
    if _WORKER is None or _WORKER.poll() is not None:
        _WORKER = subprocess.Popen([VENV_PY, ARM_PLAN, "--loop"], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, bufsize=1, env=child_env())
        print(f">>> arm plan: planner worker spawned (pid {_WORKER.pid})", flush=True)
    return _WORKER


def drop_worker():
    """Kill the worker and forget it; the next `run_child` spawns a fresh one."""
    global _WORKER
    if _WORKER is not None:
        _WORKER.kill()
        _WORKER.wait()
        _WORKER = None


def _result_line(w):
    """The child's answer line. OMPL's C++ console shares that stdout and flushes on its own
    schedule, so lines are skipped until one starts as JSON; "" if the child died."""
    for line in iter(w.stdout.readline, ""):
        if line[:1] in ("[", "{"):
            return line
    return ""


def _read_frame(w):
    """(stderr, stdout) of one answer: the stderr lines up to the event record, then the result
    line. Empty stdout means the child died mid-request."""
    err = []
    for line in iter(w.stderr.readline, ""):
        err.append(line)
        if line.startswith(EVENT_LINE):
            return "".join(err), _result_line(w)
    return "".join(err), ""


def run_one_shot(req):
    """One planner child for this request alone, arm_plan.py's one-shot mode: what the offline
    benchmark wants, since it measures a COLD plan and runs its problems in parallel threads.
    None once it exceeds PLAN_TIMEOUT."""
    try:
        return subprocess.run([VENV_PY, ARM_PLAN], input=json.dumps(req), capture_output=True,
                              text=True, env=child_env(), timeout=PLAN_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None


def _unwrap(line, seq):
    """(result line as the one-shot child would have written it, does it answer `seq`?) out of the
    worker's `{"seq", "result"}` envelope."""
    try:
        env = json.loads(line)
        if env["seq"] == seq:
            return json.dumps(env["result"]), True
        got = env["seq"]
    except Exception as e:                     # noqa: BLE001 -- unreadable is itself the answer
        got = f"unreadable ({type(e).__name__})"
    return json.dumps({"error": f"planner protocol error: answer to request {got}, "
                                f"not {seq} -- worker dropped"}), False


def run_child(req):
    """One request to the persistent planner child over the JSON line contract; None once it
    exceeds PLAN_TIMEOUT, with the stuck worker killed so the next call gets a fresh one.

    The request carries a `seq` the child echoes: an answer that does not match it is a protocol
    error result and the worker is dropped, never a path paired with the wrong request."""
    global _SEQ
    _SEQ += 1
    seq, w = _SEQ, worker()
    # A watchdog rather than a read deadline: its kill turns the blocking read into the same EOF
    # a crashed child gives.
    stuck = []
    watchdog = threading.Timer(PLAN_TIMEOUT, lambda: (stuck.append(True), w.kill()))
    watchdog.start()
    try:
        w.stdin.write(json.dumps(dict(req, seq=seq)) + "\n")
        w.stdin.flush()
        err, out = _read_frame(w)
    except (ValueError, OSError) as e:
        err, out = f"[arm_plan] worker unreachable: {type(e).__name__}: {e}\n", ""
    finally:
        watchdog.cancel()
    if stuck:
        drop_worker()
        return None
    if not out:
        # The child closed its pipes or died: this answer is lost either way, and the next call
        # must not write into it.
        drop_worker()
        return Reply(out, err)
    out, matched = _unwrap(out, seq)
    if not matched:
        drop_worker()
    return Reply(out, err)


def child_result(stdout):
    """The child's answer, its last stdout line: a list is a path, a dict is the error."""
    try:
        return json.loads(stdout.strip().splitlines()[-1])
    except Exception:                          # noqa: BLE001 -- no JSON is itself the answer
        return {"error": "planner produced no JSON (its stderr, if any, is above)"}


def plan_arm(demo, q8_goal, held=None, solve_time=3.0, tries=2, grasp_names=(),
              fingers=None, start=None):
    """RRTConnect from where the arm is now to `q8_goal`; None if no valid path was found.

    `q8_goal` is one configuration OR a list of them for the same hand pose. Given a
    set, OMPL picks the reachable branch; the path's last waypoint is the chosen branch."""
    obs, paths = planned_obstacles(demo, grasp_names)
    bt, bR = demo._arm_base_world()
    # A 3-tuple is the old caller-supplied DELETION and must never reach this tier; a 4-tuple is the
    # producer's own pad, checked against the table. Unpacking an unexpected one would kill the cycle.
    _bad = [o for o in obs
            if len(o) not in (2, 4)
            or (len(o) == 4 and (float(o[2]), tuple(o[3]))
                != (CHASSIS_PAD_M, tuple(CHASSIS_PAD_LINKS)))]
    if _bad:
        print(f">>> plan: REFUSED -- {len(_bad)} obstacle(s) carry an exemption the planned "
              f"tier never grants", flush=True)
        demo._fallback("plan-bad-obstacle")
        return None

    lo, hi = demo._arm_bounds()
    _gg = np.asarray(q8_goal, float)
    _gg = _gg if _gg.ndim == 2 else _gg.reshape(1, -1)
    # The seed: `plan_start` (commanded under drives when it is where the arm is, else
    # measured), or an explicit `start` from a caller that already chose.
    _start8 = plan_start(demo)[0] if start is None else np.asarray(start, float)
    if not np.all(np.isfinite(_start8)):
        # The child's checker skips every box for a NaN pose, so it would validate the path.
        demo._fallback("plan-start-not-finite")
        print(">>> arm plan: start pose is not finite -- REFUSED, nothing planned", flush=True)
        return None
    req = {"start": _start8.tolist(),
           "goals": [[float(v) for v in g] for g in _gg],
           "model": ARM_MODEL, "lut": CLOSURE_LUT,
           "bounds": [lo.tolist(), hi.tolist()],
           "base": [bt.tolist(), bR.tolist()],
           # STILL EXACTLY THREE ELEMENTS: the child refuses any other length, because a positional entry of
           # the wrong size mis-parses silently. The pad travels as its own keyed side table below.
           "obstacles": [[o[0][0].tolist(), o[0][1].tolist(), o[1]] for o in obs],
           # (index, pad_m, links) per padded box. A child that does not know this key plans the deck at
           # full MARGIN and refuses MORE, never less -- the safe direction for a version skew.
           "obstacle_pads": [[i, float(o[2]), list(o[3])]
                             for i, o in enumerate(obs) if len(o) == 4],
           "held": None if held is None else np.asarray(held, float).tolist(),
           # Plain floats, like `held` above: an np.float64 raises in `json.dumps(req)`, which
           # nothing here handles, and kills the cycle -- the class the refusal above prevents.
           "fingers": None if fingers is None else {str(k): float(v)
                                                    for k, v in fingers.items()},
           "margin": MARGIN, "solve_time": solve_time,
           # ARM_SEED pins RRTConnect's sampling, as NAV_SEED pins the base planner.
           "seed": int(os.environ["ARM_SEED"]) if os.environ.get("ARM_SEED") else None}
    for t, budget in enumerate(try_budgets(solve_time, tries)):
        req["solve_time"] = budget
        _t0 = time.time()
        r = run_child(req)
        if r is None:
            # A slow plan is a planner FAILURE, counted and refused like any other no-path, not
            # an exception that kills the cycle; arm_plan bounds its solve, so this means stuck.
            demo._fallback("plan-timeout")
            print(f">>> arm plan: try {t + 1}/{tries} TIMED OUT after {PLAN_TIMEOUT}s -- "
                  f"{len(_gg)} goal candidate(s) at a {req['solve_time']:.1f}s budget "
                  f"-> no plan", flush=True)
            return None
        if not hasattr(demo, "_bench_latencies"):
            demo._bench_latencies = []
        # (seconds, the child answered with an event record): a refusal is a 2 ms plan through
        # the worker, so the latency alone cannot say whether the planner wrap really ran.
        demo._bench_latencies.append((time.time() - _t0, EVENT_LINE in r.stderr))
        # stdout is a strict JSON channel, so the child reports degrades on stderr and this is the only
        # place they reach the operator log. Reading it only on a parse failure hides them.
        for ln in r.stderr.splitlines():
            print(f">>>   {ln}", flush=True)
            if TUNNEL_DEGRADE in ln:
                # Counted wherever it appears, including on a densified candidate later rejected: this
                # closure has never amplified enough to reach a link's limit, so any appearance is a defect.
                demo._fallback("densify-depth-cap")
        out = child_result(r.stdout)
        if isinstance(out, list):
            # `o[0], o[1]`, not `for ab, tg in obs`: a padded box is a 4-tuple and would raise
            # here, where nothing handles it.
            _near = min((float(np.linalg.norm(
                np.maximum(np.maximum(o[0][0] - bt, bt - o[0][1]), 0.0)))
                for o in obs if o[1] == "world"), default=float("inf"))
            print(f">>> arm plan: {len(out)} waypoints over {len(obs)} obstacles, nearest world "
                  f"obstacle {_near:.2f}m (try {t + 1}/{tries})", flush=True)
            return [np.asarray(w, float) for w in out]
        err = out.get("error", "")
        print(f">>> arm plan: try {t + 1}/{tries} failed -- {err}", flush=True)
        if t == 0 and "invalid" in err:
            # `fingers` is the hand the CHECK used, not the swept union.
            arm_blame(demo, _start8 if "start" in err else _gg[0],
                      obs, paths, held, "start" if "start" in err else "goal", (bt, bR),
                      fingers=fingers)
    return None


def goal_ik(demo, m, pos, R, seed, hand, bounds):
    """The collision-repaired goal solve every `_object_relative_goal` branch runs; `pos` / `R`
    are in the ARM-ROOT frame.
    The same solve as `m.ik`, repaired only if it collides, against the WHOLE obstacle set --
    the target object included, since the planned reach ends at the baked HOVER frame, hundreds
    of mm above it. Returns bare None both for non-convergence and out-of-domain, which the
    caller reads as "use the baked frame's goal"."""
    # held=None: this is the PICK goal -- nothing is carried yet.
    return m.ik_repair(pos, R, obstacles=demo._arm_obstacles(), q_now=seed, margin=MARGIN,
                       held=None, base=demo._arm_base_world(), log=True,
                       link=hand, bounds=bounds)


def goal_set(demo, goal, grasp, held):
    """Return `goal` plus any collision-free IK branches at the same hand pose.

    The chosen goal stays FIRST, so with OBJ_GOAL_SET=0 -- the bare CODE default, though the
    profile sets it to 1 -- or when no sibling branch exists, this is exactly a
    single-goal request. Enabled, OMPL decides which branch to take by reachability instead of
    inheriting whichever one the damped-least-squares descent landed on."""
    goal = np.asarray(goal, float)
    if os.environ.get("OBJ_GOAL_SET", "0") != "1":
        return goal
    try:
        m = demo._arm_model()
        pos, R = m.fk(goal)[m.hand_link]
        lo, hi = demo._arm_bounds()
        cands = m.ik_candidates(pos, R, obstacles=demo._arm_obstacles(grasp), q_now=goal,
                                margin=MARGIN, held=held, base=demo._arm_base_world(),
                                bounds=(lo, hi))
        out = [goal]
        for (q8, _ep, _er), _clr, _s in cands:
            if all(float(np.max(np.abs(q8 - g))) > 1e-3 for g in out):
                out.append(np.asarray(q8, float))
        print(f">>> goal-set: {len(out)} goal candidate(s) at the same hand pose "
              f"({len(cands)} clear branches found)", flush=True)
        return out
    except Exception as _e:
        print(f">>> goal-set: unavailable ({_e}) -- planning to the single goal", flush=True)
        return goal
