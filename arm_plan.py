#!/usr/bin/env python3
"""OMPL RRTConnect over arm-1's eight actuated joints; JSON on stdin, JSON on stdout. A separate
process so a C++ segfault cannot take the simulator down with an object held by friction.
One request per run, or `--loop` for one request per line until EOF, the answer wrapped with the
request's `seq`: the simulator keeps one worker so spawn and import are paid once, not per plan.
State: [theta, h1, h2, a1, hb, wz, wy, wx] (ArmModel.Q8 order)."""
import json
import os
import sys
import time

import numpy as np
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from morph.arm.model import ArmModel  # noqa: E402

# Arm-vs-arm rejection inside the sampler, OFF by default. A VETO, not a counted degrade: `isValid`
# runs thousands of times per solve, so a counter here would measure the sampler, not the task.
SELF_COLLIDE = os.environ.get("SELF_COLLIDE_PLAN", "0") == "1"
DOF = 8
STEP = 0.02          # FLOOR on emitted waypoint spacing, per joint (rad or m) -- not the bound
MAX_DEPTH = 6        # bisections per STEP sub-segment: 64 pieces, 8x the worst measured need
RES = 0.005          # validity-check spacing, as a FRACTION of a subspace's maximum extent
# The eight joints by UNIT: Q8_LIN are METRES (both column carriages and the boom), Q8_ROT are
# RADIANS (turret, hand bearing, three wrists). They interleave, so this is not a 3/5 slice.
LIN, ROT = ArmModel.Q8_LIN, ArmModel.Q8_ROT


class Validity(ob.StateValidityChecker):
    def __init__(self, si, model, obstacles, held, margin, base, fingers=None):
        super().__init__(si)
        self.m, self.obs, self.held, self.margin = model, obstacles, held, margin
        # LAST, and defaulted: `tests/test_slab_gate.py` constructs this with six positionals.
        self.base, self.fingers = base, fingers

    def isValid(self, s):
        q = _q8(s)
        dh = q[2] - q[1]
        # Valid = inside the joint bounds and the LUT's (dh, a1) domain, every closure passive inside its
        # USD limit, nothing overlapping an obstacle past the margin. The cheap gates run before FK.
        if not self.m.in_lut_domain(dh, q[3]) or self.m.passive_violations(dh, q[3]):
            return False
        if SELF_COLLIDE and self.m.self_collides(q) is not None:
            return False
        return not self.m.collides(q, self.obs, self.margin, self.held, self.base,
                                   fingers=self.fingers)


def goal_budgets(solve_time, n):
    """Per-candidate solve budgets for `n` candidates, summing to at most 2 * `solve_time`: the
    first (preferred) candidate keeps the whole `solve_time`, the other n-1 share one more, so the
    sweep is bounded however many seeds were asked for."""
    n = int(n)
    if n <= 1:
        return [float(solve_time)] * max(0, n)
    return [float(solve_time)] + [float(solve_time) / (n - 1)] * (n - 1)


# One record per request, written as the last stderr line by `main`: the machine-readable form
# of the text lines, for the benchmark and any log reader. The text stays for the operator log.
EVENT = {}
# The seed applied to this process, so a worker does not re-seed per request.
_SEED = None


def _note(**kv):
    EVENT.update(kv)


def plan(req):
    # ARM_PLAN_DUMP=<dir>: dump each request verbatim, for offline replay against real
    # obstacles. Off unless set.
    _dump = os.environ.get("ARM_PLAN_DUMP")
    if _dump:
        try:
            os.makedirs(_dump, exist_ok=True)
            _n = len(os.listdir(_dump))
            with open(os.path.join(_dump, f"req_{_n:03d}.json"), "w") as _f:
                json.dump(req, _f)
        except Exception as _e:                    # noqa: BLE001 -- a probe must not kill a plan
            sys.stderr.write(f"[arm_plan] request dump failed: {_e}\n")
    m = ArmModel.load(req["model"], req["lut"])
    # Positional, so an entry of the wrong length parses cleanly with the extra dropped or the tag read
    # off the wrong index -- and the child would plan against a set the parent never sent. Refuse.
    obs = []
    for o in req.get("obstacles") or []:
        if not isinstance(o, (list, tuple)) or len(o) != 3:
            n = len(o) if isinstance(o, (list, tuple)) else type(o).__name__
            return {"error": f"malformed obstacle: expected (lo3, hi3, tag), got {n}"}
        obs.append((np.array([o[0], o[1]], float), o[2]))
    held = None if req.get("held") is None else np.array(req["held"], float)
    margin = float(req.get("margin", 0.02))
    # The COMMANDED hand, when the parent sends one. Absent, everything below runs exactly as
    # before on the swept union -- this whole surface is inert until a request carries the key.
    fingers = {str(k): float(v) for k, v in (req.get("fingers") or {}).items()} or None
    # Per-obstacle FLOORS, re-validated rather than trusted: parent and child must not read one request
    # differently. `pad` is clamped into [0, margin], so no encoding of a deletion crosses here.
    pads = []
    for e in req.get("obstacle_pads") or []:
        if not isinstance(e, (list, tuple)) or len(e) != 3:
            n = len(e) if isinstance(e, (list, tuple)) else type(e).__name__
            return {"error": f"malformed obstacle pad: expected (index, pad_m, links), got {n}"}
        i, pad, links = e
        if not isinstance(i, int) or not 0 <= i < len(obs):
            return {"error": f"obstacle pad indexes no obstacle: {i} of {len(obs)}"}
        if not links or not all(isinstance(s, str) for s in links):
            return {"error": "obstacle pad names no link"}
        if not 0.0 <= float(pad) <= margin:
            return {"error": f"obstacle pad {pad} outside [0, {margin}]"}
        obs[i] = (obs[i][0], obs[i][1], float(pad), tuple(links))
        pads.append(float(pad))
    # Each obstacle's TAG names the frame its box is axis-aligned in: "world" or "chassis" (the
    # base frame). `base` is (position, quaternion) and is what lets `collides` evaluate both.
    base = None if req.get("base") is None else (np.array(req["base"][0], float),
                                                 np.array(req["base"][1], float))
    # OMPL rejects seed 0 and silently uses 1. Re-seeding a process whose generation has started
    # is an `Error:` line per request in the operator log, so the same seed is applied once.
    global _SEED
    if req.get("seed") is not None and max(1, int(req["seed"])) != _SEED:
        _SEED = max(1, int(req["seed"]))
        ou.RNG.setSeed(_SEED)

    # Bounds arrive in the request, never from morph.config: that module prints on import, which
    # would corrupt this stdout protocol. The Isaac side owns the constants and sends them.
    lo, hi = m.bounds()
    # caller-supplied, tighter than the USD limits
    if req.get("bounds"):
        lo = np.array(req["bounds"][0], float)
        hi = np.array(req["bounds"][1], float)

    space = _space(lo, hi)

    si = ob.SpaceInformation(space)
    # The request has no schema check, so an unsent or misspelled `fingers` key is silently the
    # swept box again. This line is the only evidence at runtime of which hand was actually used.
    sys.stderr.write("[arm_plan] hand: %s\n" % ("21 boxes (fingers given)" if fingers
                                                else "12 (swept union)"))
    _note(hand="commanded" if fingers else "swept")
    checker = Validity(si, m, obs, held, margin, base, fingers)  # keep a ref: OMPL does not own it
    si.setStateValidityChecker(checker)
    # `CompoundStateSpace::setLongestValidSegmentFraction` PROPAGATES to the subspaces (measured in
    # tests/test_slab_gate.py), so this one call gives each unit its own step.
    si.setStateValidityCheckingResolution(RES)
    si.setup()

    _oob = _out_of_bounds(req["start"], lo, hi)
    if _oob is not None:
        return {"error": f"start state out of bounds: {_oob}"}
    start = _as_state(space, req["start"])
    # No nudging, unlike nav_plan: an arm goal is the exact joint vector the downstream servo
    # expects to start from, so an unreachable goal is an error the caller has to see.
    if not si.isValid(start):
        return {"error": "start state invalid (collision, outside the closure domain, "
                         "or a closure passive past its stop)"}

    # A goal SET, not one configuration: `ik` returns whichever branch its descent lands on, so handing
    # OMPL one decides by accident what OMPL can decide by reachability. `goal` still works.
    raw_goals = req.get("goals") or [req["goal"]]
    goal_states, goal_vecs, invalid, oob = [], [], 0, []
    for g in raw_goals:
        # Bounds FIRST: `si.isValid` never checks them, so an out-of-bounds candidate would be
        # planned toward -- measured, one was planned TO -- and the caller told "no path found".
        _g_oob = _out_of_bounds(g, lo, hi)
        if _g_oob is not None:
            oob.append(_g_oob)
            invalid += 1
            continue
        gs = _as_state(space, g)
        if si.isValid(gs):
            goal_states.append(gs)
            goal_vecs.append([float(x) for x in g])
        else:
            invalid += 1
    _note(goals=len(raw_goals), bad_goals=invalid)
    if not goal_states and oob:
        return {"error": f"goal state out of bounds: {oob[0]}"
                         + (f" (and {len(oob) - 1} more candidate(s))" if len(oob) > 1 else "")}
    if not goal_states:
        return {"error": f"goal state invalid (collision, outside the closure domain, or a "
                         f"closure passive past its stop); all {invalid} candidate(s) rejected"}
    if invalid:
        sys.stderr.write(f"[arm_plan] {invalid} of {len(raw_goals)} goal candidates invalid, "
                         f"planning to the remaining {len(goal_states)}\n")

    # `ob.GoalStates(si)` is unconstructible in this binding, `ob.GoalLazySamples` is not: the loop
    # is a CHOICE: candidates arrive best-first, so the preferred goal costs a single-goal request.
    last_err = "no path found"
    _t0 = time.perf_counter()
    _t_solve = 0.0
    budgets = goal_budgets(float(req.get("solve_time", 3.0)), len(goal_states))
    for gi, (gs, gvec) in enumerate(zip(goal_states, goal_vecs)):
        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(start, gs)
        planner = og.RRTConnect(si)
        # 0.15 in the compound metric (`_space`): d_metres + d_radians, between 1x and sqrt(2)x a
        # single space's hypot, so an extension is never longer than that bound.
        planner.setRange(0.15)
        planner.setProblemDefinition(pdef)
        planner.setup()
        _t_s = time.perf_counter()
        _solved = planner.solve(budgets[gi])
        _t_solve += time.perf_counter() - _t_s
        # `bool(PlannerStatus)` is TRUE for an APPROXIMATE solution, and accepting one is not a near miss:
        # `_densify` snaps the last point onto the goal, so the gap becomes a segment nothing validated.
        _exact = bool(_solved) and pdef.hasExactSolution()
        if not _exact:
            if _solved:
                sys.stderr.write(f"[arm_plan] goal {gi}: APPROXIMATE solution discarded -- the "
                                 f"planner stopped short and the snap would hide it\n")
                _note(approximate=EVENT.get("approximate", 0) + 1)
            last_err = (f"no path found to any of {len(goal_states)} goal candidate(s)"
                        if not _solved else
                        f"planner returned an APPROXIMATE solution for {len(goal_states)} goal "
                        f"candidate(s); it stops short and the goal snap would manufacture an "
                        f"unvalidated segment")
            continue
        if gi:
            sys.stderr.write(f"[arm_plan] goal candidate {gi} solved after {gi} unreachable\n")
        goal_vecs = [gvec]                      # densify snaps the endpoint to the goal REACHED
        _note(goal_index=gi)
        break
    else:
        return {"error": last_err}

    raw = pdef.getSolutionPath()
    raw_n = raw.getStateCount()
    simplified = og.PathGeometric(raw)
    # Post-processing dominates plan latency, not the solve. `reduce` is the default; `timed` is NOT a
    # budget, since `atLeastOnce` runs a full pass and one pass is the cost.
    _mode = os.environ.get("ARM_SIMPLIFY_MODE", "reduce")
    _ps = og.PathSimplifier(si)
    _t_simp = time.perf_counter()
    if _mode == "none":
        pass
    elif _mode == "reduce":
        _ps.reduceVertices(simplified)
    elif _mode == "timed":
        _ps.simplify(simplified, float(os.environ.get("ARM_SIMPLIFY_T", "1.0")), True)
    else:
        _ps.simplifyMax(simplified)
    _t_simp = time.perf_counter() - _t_simp

    # Densify from the SIMPLIFIED path only if every emitted waypoint survives the checker, else from
    # the raw one: OMPL validates at its own discretisation, so a shortcut can graze between checks.
    for cand in (simplified, raw):
        _t_dens = time.perf_counter()
        out = _densify(cand, req, goal_vecs, m, held, margin, fingers, pads)
        bad = [k for k, w in enumerate(out) if not checker.isValid(_as_state(space, w))]
        _t_dens = time.perf_counter() - _t_dens
        if not bad:
            # `solve_time` bounds the SOLVE only; simplify and densify are outside it.
            _t_all = time.perf_counter() - _t0
            sys.stderr.write(f"[arm_plan] {raw_n} -> {cand.getStateCount()} states, "
                             f"{len(out)} waypoints\n")
            sys.stderr.write(f"[arm_plan] plan {_t_all:.2f}s = solve {_t_solve:.2f} + simplify "
                             f"{_t_simp:.2f} + densify/revalidate {_t_dens:.2f} "
                             f"(solve_time budget {float(req.get('solve_time', 3.0)):.1f}s bounds "
                             f"the solve ONLY)\n")
            _note(states_raw=raw_n, states=cand.getStateCount(), waypoints=len(out),
                  plan_s=round(_t_all, 4), solve_s=round(_t_solve, 4), simplify_s=round(_t_simp, 4),
                  densify_s=round(_t_dens, 4))
            _report_self_collisions(m, out)
            return [w.tolist() for w in out]
        sys.stderr.write(f"[arm_plan] {len(bad)} of {len(out)} densified waypoints invalid "
                         f"({'simplified' if cand is simplified else 'raw'} path)\n")
        _note(densify_bad=EVENT.get("densify_bad", 0) + len(bad))
    return {"error": "no path whose densified waypoints are all valid"}


def _report_self_collisions(m, waypoints):
    """SHADOW metric: does the path we are about to emit fold the arm into itself?

    Measured on the FINAL densified waypoints -- the states the executor actually writes -- which
    is one pass per plan rather than the thousands `isValid` costs, and which counts the thing that
    matters instead of counting sampler rejections. Reports and returns; the veto is
    SELF_COLLIDE_PLAN=1. Nothing has ever measured this, so it is turned on before it is enforced.
    """
    hits = [(k, m.self_collides(np.asarray(w, float))) for k, w in enumerate(waypoints)]
    hits = [(k, pair) for k, pair in hits if pair is not None]
    _note(selfcol=len(hits), selfcol_first=hits[0][0] if hits else None)
    if not hits:
        sys.stderr.write(f"[arm_plan] self-collision: clear over {len(waypoints)} waypoints\n")
        return []
    worst = {}
    for _k, pair in hits:
        worst[pair] = worst.get(pair, 0) + 1
    top = ", ".join(f"{a} x {b} ({n})" for (a, b), n in
                    sorted(worst.items(), key=lambda kv: -kv[1])[:3])
    sys.stderr.write(f"[arm_plan] SELF-COLLISION on {len(hits)} of {len(waypoints)} emitted "
                     f"waypoints: {top}"
                     f"{'' if SELF_COLLIDE else '  (shadow only -- SELF_COLLIDE_PLAN=0)'}\n")
    # Waypoint 0 folding means the START was already folded -- the caller left the arm there,
    # not a route the planner chose. Different file to go fix, so say which.
    k0, pair0 = hits[0]
    sys.stderr.write(f"[arm_plan] first fold at waypoint {k0}/{len(waypoints)} "
                     f"({pair0[0]} x {pair0[1]}): "
                     f"q8={[round(float(x), 5) for x in np.asarray(waypoints[k0], float)]}"
                     f"{'  <-- THE START POSE IS ALREADY FOLDED' if k0 == 0 else ''}\n")
    return hits


def _out_of_bounds(q, lo, hi, tol=1e-9):
    """The joint that lies outside its bound, and by how much -- or None.

    `si.isValid` runs the validity CHECKER only: OMPL never calls satisfiesBounds for us, and
    `_as_state` does not clamp. An endpoint outside the bounds is therefore not refused; it is
    sampled toward until the budget is gone, or -- measured -- planned to outright. Either way the
    caller is told "no path found" and goes looking for an obstacle that was never there."""
    q = np.asarray(q, float)
    for i, (v, a, b) in enumerate(zip(q, np.asarray(lo, float), np.asarray(hi, float))):
        if v < a - tol or v > b + tol:
            name = ArmModel.Q8[i] if i < len(ArmModel.Q8) else f"joint {i}"
            past = (a - v) if v < a else (v - b)
            return f"{name} is {past:.4f} outside its bound [{a:.4f}, {b:.4f}] at {v:.4f}"
    return None


def _space(lo, hi):
    """The eight joints as R^3 METRES + R^5 RADIANS, not one mixed-unit R^8: the validity-checking
    resolution is a fraction of each subspace's OWN extent, so the prismatic step is fine without
    paying for it on the revolute joints (one R^8 would give 53mm of column travel per check)."""
    space = ob.CompoundStateSpace()
    for idx in (LIN, ROT):
        sub = ob.RealVectorStateSpace(len(idx))
        b = ob.RealVectorBounds(len(idx))
        for k, i in enumerate(idx):
            b.setLow(k, float(lo[i]))
            b.setHigh(k, float(hi[i]))
        sub.setBounds(b)
        space.addSubspace(sub, 1.0)
    return space


def _as_state(space, q):
    """A Q8-ordered vector into the compound's [metres, radians] layout."""
    s = space.allocState()
    for j, idx in enumerate((LIN, ROT)):
        for k, i in enumerate(idx):
            s[j][k] = float(q[i])
    return s


def _q8(s):
    """The inverse of `_as_state`: a compound state back to one Q8-ordered vector."""
    q = np.empty(DOF)
    for j, idx in enumerate((LIN, ROT)):
        for k, i in enumerate(idx):
            q[i] = s[j][k]
    return q


def _worst(m, lim, held, a, b, fingers=None):
    """(ratio, link) of the box closest to tunnelling between `a` and `b`: how far it moved as a
    fraction of `ArmModel.tunnel_limits`. >= 1.0 means an obstacle can pass through it unseen."""
    return max((d / lim[k], k) for k, d in m.corner_shifts(a, b, held, fingers).items())


def _densify(path, req, goals, m, held=None, margin=0.02, fingers=None, pads=()):
    """Straight-line interpolation from the caller's exact start to the REACHED goal, subdivided
    until every link stays inside its own `tunnel_limits` between consecutive waypoints (STEP is a
    floor). Endpoints are snapped BEFORE subdivision so the outer segments are refined as flown."""
    # `pads` because a pair held to a smaller floor tolerates a SHORTER step between checks: sizing
    # every limit from the global margin would let the deck pass through the boom between waypoints.
    lim = m.tunnel_limits(held, margin, fingers, pads)   # rigid boxes, computed once
    pts = np.array([_q8(path.getState(k)) for k in range(path.getStateCount())], float)
    if len(pts) < 2:
        pts = np.vstack([pts, pts])     # else `pts[0]` IS `pts[-1]` and the goal snap eats the start
    pts[0] = np.array(req["start"], float)
    _g = np.asarray(goals, float)
    pts[-1] = _g[int(np.argmin(np.linalg.norm(_g - pts[-1], axis=1)))]
    out = [pts[0]]

    capped = []

    def refine(a, b, depth):
        w = _worst(m, lim, held, a, b, fingers)
        if w[0] <= 1.0:
            out.append(b)
        elif depth < MAX_DEPTH:
            mid = 0.5 * (a + b)
            refine(a, mid, depth + 1)
            refine(mid, b, depth + 1)
        else:
            # over a link's limit at the cap: the degrade reported below
            capped.append(w)
            out.append(b)

    for a, c in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(c - a)) / STEP)))
        prev = a
        for t in range(1, n + 1):
            nxt = a + (c - a) * (t / n)
            refine(prev, nxt, 0)
            prev = nxt
    # `a + (c - a) * (n / n)` is `c` to within an ulp, not bit for bit
    out[-1] = pts[-1]
    # CONTRACT: `morph.arm.plan.TUNNEL_DEGRADE` matches this wording to relay the degrade. The
    # path is still emitted because `plan` validity-checks every waypoint below, and it is COUNTED.
    _note(tunnel=EVENT.get("tunnel", 0) + len(capped))
    if capped:
        r, link = max(capped)
        sys.stderr.write(f"[arm_plan] {len(capped)} of {len(out) - 1} emitted segments can tunnel "
                         f"a thin obstacle; worst {link} at {r:.2f}x its own limit "
                         f"(depth cap {MAX_DEPTH})\n")
    return out


def answer(text, envelope=False):
    """One request in; one JSON result line on stdout, the event record as the LAST stderr line
    before it -- the frame the parent reads. EVENT is module state, so it is cleared per request.

    The request's `seq`, when it carries one, is echoed in the event record and -- in loop mode,
    where the answer is wrapped as `{"seq", "result"}` -- on the result line too, so the parent
    cannot pair a stale stdout line with the wrong request. One-shot answers stay bare."""
    EVENT.clear()
    seq = None
    try:
        req = json.loads(text)
        seq = req.get("seq") if isinstance(req, dict) else None
        result = plan(req)
    except Exception as e:                     # noqa: BLE001 -- the parent reads one JSON answer
        result = {"error": f"{type(e).__name__}: {e}"}
    EVENT["ok"] = isinstance(result, list)
    if not EVENT["ok"]:
        EVENT["error"] = result.get("error", "")
    if seq is not None:
        EVENT["seq"] = seq
    # LAST stderr write of this request: it frames the answer, and the parent reads stderr only up
    # to it. Nothing below may write stderr before the result line goes out.
    sys.stderr.write("[arm_plan] event " + json.dumps(EVENT) + "\n")
    sys.stderr.flush()
    print(json.dumps({"seq": seq, "result": result} if envelope else result), flush=True)


def loop():
    """One request per stdin line for the life of the process, so the parent pays spawn and
    import once. EOF ends it; a request that raises is an error result."""
    for line in sys.stdin:
        if line.strip():
            answer(line, envelope=True)


if __name__ == "__main__":
    try:
        loop() if "--loop" in sys.argv[1:] else answer(sys.stdin.read())
    except BrokenPipeError:
        # The parent is gone: a normal exit would flush stdout into the closed pipe and raise
        # again, outside any handler.
        os._exit(0)
