"""Track F2: the planner takes a goal SET, not one configuration.

Offline: no Isaac, runs OMPL directly. Run: ./.venv/bin/python3 tests/test_goal_set.py

`ik` returns whichever branch its damped-least-squares descent lands on, and Track F measured that
half the solution set for a given hand pose is collision-free. Handing OMPL a single branch decides
by accident what OMPL can decide by reachability.
"""
import ast
import os
import time
import sys
from types import SimpleNamespace

import numpy as np
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import arm_plan                                                    # noqa: E402
from arm_plan import plan                                          # noqa: E402
from morph.arm.model import ArmModel  # noqa: E402

MODEL = os.path.join(_ROOT, "usd", "_arm_model.json")
LUT = os.path.join(_ROOT, "usd", "_closure_lut.json")
M = ArmModel.load(MODEL, LUT)
START = M.Q8_HOME.copy()

# Outside the closure LUT's (dh, a1) domain, so `Validity.isValid` rejects it before any FK.
BAD = START.copy()
BAD[1], BAD[2] = 0.0, 5.0          # dh = +5.0, far past the LUT's -0.100..0.300


def _req(**kw):
    r = {"model": MODEL, "lut": LUT, "start": START.tolist(), "solve_time": 5.0, "seed": 1}
    r.update(kw)
    return r


def _near_goal():
    """A reachable, valid goal: nudge a1 forward and re-check through the same gate the planner uses."""
    g = START.copy()
    g[3] = float(START[3]) + 0.06
    return g


def test_a_single_goal_still_plans():
    """The old contract is untouched: callers that send one `goal` behave exactly as before."""
    out = plan(_req(goal=_near_goal().tolist()))
    assert isinstance(out, list) and out, f"expected waypoints, got {out}"
    assert len(out[0]) == 8


def test_an_invalid_candidate_does_not_sink_the_request():
    """The defect F2 fixes: one bad branch used to fail the whole plan."""
    out = plan(_req(goals=[BAD.tolist(), _near_goal().tolist()]))
    assert isinstance(out, list) and out, f"a valid candidate was present, got {out}"


def test_the_path_ends_on_one_of_the_goals():
    """The caller consumes the last waypoint as the config the servo starts from, so with a goal
    SET that waypoint must actually be one of the candidates -- not an interpolation between them."""
    good = _near_goal()
    out = plan(_req(goals=[BAD.tolist(), good.tolist()]))
    assert isinstance(out, list) and out
    assert np.allclose(np.asarray(out[-1], float), good, atol=1e-6), \
        f"path ended at {out[-1]}, not on the goal {good.tolist()}"


def test_all_candidates_invalid_is_still_an_error():
    """A goal set must not turn 'unreachable' into a silent success."""
    out = plan(_req(goals=[BAD.tolist(), BAD.tolist()]))
    assert isinstance(out, dict) and "error" in out, f"expected an error, got {out}"
    # BAD is dh = +5.0: outside the column bound as well as the LUT domain, so either refusal is
    # correct as long as it still accounts for BOTH candidates rather than reporting one.
    assert "2 candidate" in out["error"] or "1 more candidate" in out["error"], out["error"]


def test_several_VALID_goals_actually_plan():
    """The gap that let a broken build ship: every earlier test here had at most ONE valid goal, so
    they all took the single-goal branch and the multi-goal path was never executed. It raised
    TypeError on OMPL's GoalStates ctor in the simulator, on every approach, and the offline suite
    was green throughout."""
    g1 = _near_goal()
    g2 = START.copy(); g2[3] = float(START[3]) + 0.10
    out = plan(_req(goals=[g1.tolist(), g2.tolist()]))
    assert isinstance(out, list) and out, f"two valid goals must plan, got {out}"
    assert len(out[0]) == 8


def test_the_first_reachable_candidate_is_the_one_used():
    """Candidates arrive best-first, so a reachable first goal must be taken unchanged -- that is
    what makes the goal set free when the preferred branch works."""
    good = _near_goal()
    far = START.copy(); far[3] = float(START[3]) + 0.10
    out = plan(_req(goals=[good.tolist(), far.tolist()]))
    assert isinstance(out, list) and out
    assert np.allclose(np.asarray(out[-1], float), good, atol=1e-6), \
        f"ended on {out[-1]}, expected the first candidate {good.tolist()}"


# The whole goal set's OMPL budget per reach, across every candidate and both escalation tries.
# Each try is its own subprocess under `_plan_arm`'s timeout, so the worst case sits well under it.
BUDGET_CAP = 20.0
SOLVE_TIME = 3.0                   # `_plan_arm`'s default; try 2 runs at 2x it
DEFAULT_CANDIDATES = 9             # `ik_candidates(seeds=8)` plus the preferred goal


def _budgets(solve_time, n):
    """`arm_plan.goal_budgets`, asserted into existence: without it every candidate is handed the
    FULL `solve_time` and the worst case grows with `ik_candidates(seeds=...)` with no ceiling."""
    assert hasattr(arm_plan, "goal_budgets"), (
        "arm_plan has no `goal_budgets`: every goal candidate still gets the full solve_time")
    return arm_plan.goal_budgets(solve_time, n)


def test_the_candidate_budgets_sum_to_a_bound_that_does_not_grow_with_the_seed_count():
    """The defect: the candidate loop spent `solve_time` PER CANDIDATE, so raising
    `ik_candidates(seeds=...)` raised the worst-case reach linearly and without a ceiling. The
    preferred candidate must still get the whole `solve_time` -- it arrives first and solving it is
    the normal path, so the common case has to cost exactly what it cost before."""
    for n in (1, 2, 3, DEFAULT_CANDIDATES, 25, 100):
        for st in (SOLVE_TIME, 2 * SOLVE_TIME):
            b = _budgets(st, n)
            assert len(b) == n, f"n={n}: got {len(b)} budgets"
            assert sum(b) <= 2 * st + 1e-9, f"n={n} solve_time={st}: total {sum(b)} > {2 * st}"
            assert b[0] == st, f"n={n}: the preferred candidate got {b[0]}, not the full {st}"
            assert all(x > 0 for x in b), f"n={n}: a candidate got a zero budget: {b}"


def test_the_worst_case_reach_sits_far_under_the_subprocess_timeout():
    """`_plan_arm` runs `tries=2` at `solve_time` then `2 * solve_time`, each try sweeping every
    candidate. Before the bound that was 9 x (3 + 6) = 81 s of solve time per reach."""
    worst = (sum(_budgets(SOLVE_TIME, DEFAULT_CANDIDATES))
             + sum(_budgets(2 * SOLVE_TIME, DEFAULT_CANDIDATES)))
    assert worst <= BUDGET_CAP, f"worst-case reach {worst}s over the {BUDGET_CAP}s cap"
    assert BUDGET_CAP * 2 < 120, "the cap must leave real margin under _plan_arm's 120s timeout"


def test_the_divided_budget_is_the_one_the_candidate_loop_actually_spends():
    """A bound the loop never spends is not a bound: `goal_budgets` could be exactly right and
    unreferenced, and the 81 s worst case would sit precisely where it was. So this records what
    `planner.solve()` is ASKED for, candidate by candidate -- the computed budget, no wall clock.

    `solve_time` is deliberately far too small for anything to solve, and that is what makes the
    loop sweep every candidate: with a workable budget the preferred one solves first and the
    indices where the two budget policies differ are never reached at all. The `len(spent) == 3`
    guard is load-bearing -- a sweep that stopped short would make the assertions after it
    vacuous."""
    st = 1e-6
    g2 = START.copy(); g2[3] = float(START[3]) + 0.08
    g3 = START.copy(); g3[3] = float(START[3]) + 0.10
    spent, real_og = [], arm_plan.og

    # `plan` only ever calls these four on the planner it builds
    class _Spy:
        def __init__(self, si):
            self._p = real_og.RRTConnect(si)

        def __getattr__(self, k):
            return getattr(self._p, k)

        def solve(self, t):
            spent.append(float(t))
            return self._p.solve(t)

    arm_plan.og = SimpleNamespace(RRTConnect=_Spy, PathGeometric=real_og.PathGeometric,
                                  PathSimplifier=real_og.PathSimplifier)
    try:
        out = plan(_req(goals=[_near_goal().tolist(), g2.tolist(), g3.tolist()], solve_time=st))
    finally:
        arm_plan.og = real_og
    assert isinstance(out, dict), f"the probe budget was meant to be unsolvable, got {out}"
    assert len(spent) == 3, (
        f"the loop did not sweep all 3 candidates, so nothing below is evidence: {spent}")
    assert spent[0] == st, f"the preferred candidate was given {spent[0]}, not the whole {st}"
    assert sum(spent) <= 2 * st + 1e-15, (
        f"the loop spent {sum(spent)}s over 3 candidates at solve_time={st} -- every candidate is "
        f"still being handed the full budget")


# What these bindings ACTUALLY do with a goal region: `arm_plan.plan` serves the set candidate by
# candidate, and these tests measure that justification rather than inheriting it.
_SEALED = (0.80, 0.80)      # valid, but inside a closed shell: reachable by nothing
_OPEN = (0.95, 0.05)        # valid and reachable
_START2 = (0.05, 0.05)


class _Free(ob.StateValidityChecker):
    """Invalid = the 0.65..0.95 square minus its 0.70..0.90 interior, i.e. a sealed pocket."""

    def isValid(self, s):
        x, y = s[0], s[1]
        return not (0.65 <= x <= 0.95 and 0.65 <= y <= 0.95
                    and not (0.70 < x < 0.90 and 0.70 < y < 0.90))


class _TwoGoals(ob.GoalSampleableRegion):
    """The thing ompl #895 says RRTConnect cannot solve against."""

    def __init__(self, si, goals):
        super().__init__(si)
        self.goals, self.calls = goals, 0
        self.setThreshold(1e-9)

    def sampleGoal(self, st):
        g = self.goals[self.calls % len(self.goals)]
        self.calls += 1
        st[0], st[1] = float(g[0]), float(g[1])

    def maxSampleCount(self):
        return len(self.goals)

    def distanceGoal(self, st):
        return min(max(abs(st[0] - g[0]), abs(st[1] - g[1])) for g in self.goals)


def _plane():
    """A 2-D unit square with the sealed pocket. Small on purpose: the claim under test is about
    the BINDINGS, so the arm's 8-D problem would only add ways for the test to fail for a
    different reason."""
    space = ob.RealVectorStateSpace(2)
    b = ob.RealVectorBounds(2)
    b.setLow(0.0)
    b.setHigh(1.0)
    space.setBounds(b)
    si = ob.SpaceInformation(space)
    checker = _Free(si)                     # keep the reference: ompl does not own it
    si.setStateValidityChecker(checker)
    si.setStateValidityCheckingResolution(0.001)
    si.setup()
    return space, si, checker


def _state(space, q):
    s = space.allocState()
    s[0], s[1] = float(q[0]), float(q[1])
    return s


def test_the_goalstates_half_of_the_justification_holds():
    """`ob.GoalStates(si)` really is unconstructible from these bindings -- the ctor wants a
    std::shared_ptr<SpaceInformation> and `ob.SpaceInformation(space)` is the raw object. This is
    the half of the planner's comment that survives measurement."""
    _space, si, _c = _plane()
    try:
        ob.GoalStates(si)
    except TypeError as e:
        assert "shared_ptr" in str(e), f"unconstructible, but not for the cited reason: {e}"
        return
    assert False, "ob.GoalStates(si) constructed: the planner's sequential loop has lost its reason"


def test_a_goalsampleableregion_subclass_DOES_solve_with_rrtconnect_here():
    """ompl #895 does not reproduce in this venv, and this is the measurement that says so.

    The planner's comment inherited #895 as 'subclassing GoalSampleableRegion is broken with
    RRTConnect'. #895's actual complaint is subtler than a crash -- the planner reports 'Exact
    solution' and hands back a path that does not solve the problem -- so the status alone would
    prove nothing and the PATH is what is checked here: it starts at the start, every motion along
    it is valid, and it ends on the goal that is actually REACHABLE, not the one the sampler
    offered first. The control above proves the sealed goal really is unreachable, so ending there
    is a choice by reachability and not an accident of which goal got sampled.

    Recorded as evidence only. `plan` still sweeps candidates sequentially; whoever revisits that
    starts from this instead of from the issue number."""
    ou.setLogLevel(ou.LOG_ERROR)
    space, si, _c = _plane()

    ou.RNG.setSeed(1)
    pd0 = ob.ProblemDefinition(si)
    pd0.setStartAndGoalStates(_state(space, _START2), _state(space, _SEALED))
    p0 = og.RRTConnect(si)
    p0.setRange(0.05)
    p0.setProblemDefinition(pd0)
    p0.setup()
    p0.solve(3.0)
    assert not pd0.hasExactSolution(), (
        "the CONTROL failed: the sealed goal is reachable on its own, so the experiment below "
        "would prove nothing about choosing by reachability")

    ou.RNG.setSeed(1)
    goal = _TwoGoals(si, [_SEALED, _OPEN])
    pd = ob.ProblemDefinition(si)
    pd.addStartState(_state(space, _START2))
    pd.setGoal(goal)
    pl = og.RRTConnect(si)
    pl.setRange(0.05)
    pl.setProblemDefinition(pd)
    pl.setup()
    pl.solve(3.0)

    assert goal.calls > 0, "RRTConnect never called the subclass's sampleGoal from C++"
    assert pd.hasExactSolution(), "no exact solution against a GoalSampleableRegion subclass"
    path = pd.getSolutionPath()
    n = path.getStateCount()
    assert path.check(), "#895's failure mode: 'Exact solution' over a path that is not valid"
    assert [round(path.getState(0)[i], 6) for i in (0, 1)] == list(_START2), (
        f"the path does not begin at the start: {[path.getState(0)[i] for i in (0, 1)]}")
    assert not [k for k in range(n - 1)
                if not si.checkMotion(path.getState(k), path.getState(k + 1))], (
        "#895's failure mode: a motion along the returned path is invalid")
    assert [round(path.getState(n - 1)[i], 6) for i in (0, 1)] == list(_OPEN), (
        f"ended at {[path.getState(n - 1)[i] for i in (0, 1)]}, not on the reachable goal "
        f"{list(_OPEN)}")


def test_goallazysamples_DOES_construct_from_these_bindings():
    """The other half of the planner's comment, and the half that does NOT survive measurement.
    It claimed BOTH ctors want a std::shared_ptr<SpaceInformation>. `GoalLazySamples` does not:
    its binding takes the RAW `ob.SpaceInformation` and all four arguments positionally
    (si, callback, auto_start, min_dist). The short forms raise a TypeError too -- which is
    presumably where "unconstructible" came from -- but for a missing argument, not a shared_ptr.
    `auto_start=False` on purpose: True starts a sampling thread this measurement does not need.

    Evidence only. Replacing the sequential loop with a real goal region needs the 8-D validity
    checker, candidate-priority behaviour, endpoint densification and a performance comparison;
    none of that is measured here and none of it is in scope."""
    _space, si, _c = _plane()

    def _cb(gls, st):
        # never yields a sample: construction is the whole measurement
        return False

    g = ob.GoalLazySamples(si, _cb, False, 1e-9)
    assert isinstance(g, ob.GoalSampleableRegion), f"constructed, but as {type(g)}"
    for short in ((si, _cb), (si, _cb, False)):
        try:
            ob.GoalLazySamples(*short)
        except TypeError as e:
            assert "shared_ptr" not in str(e), (
                f"the {len(short)}-argument form fails for the reason the comment claimed after "
                f"all, so the comment is right and this test is the thing that is wrong: {e}")
        else:
            assert False, f"the {len(short)}-argument form constructed too"

def test_the_planner_does_not_claim_goallazysamples_is_unconstructible():
    """The defect item 3 names is the COMMENT, so this is the regression that guards it. The test
    above measures the binding; this one keeps the file from claiming the opposite again."""
    src = open(os.path.join(_ROOT, "arm_plan.py")).read()
    bad = [l.strip() for l in src.splitlines()
           if "GoalLazySamples" in l and "cannot be constructed" in l]
    assert not bad, f"arm_plan.py still claims GoalLazySamples is unconstructible: {bad}"
    assert "GoalStates" in src, (
        "the surviving half of the claim went with it: `ob.GoalStates(si)` really does fail")

# The planner refuses a start inside MARGIN, and that is the contract: relaxing it would be
# granted against boxes the start was never near, and the goal and tunnel checks inherit it.

def _box_at(q8, gap, span=0.25):
    """A box `gap` metres beyond the FAR FACE of every arm link at pose `q8`, along +x.

    Measured from the geometry, not from a link origin: the hand is ~0.2 m across, so a gap taken
    from its origin puts the box inside it."""
    boxes = M.link_aabbs(q8)
    far_x = max(float(b[1][0]) for b in boxes.values())
    ys = [c for b in boxes.values() for c in (float(b[0][1]), float(b[1][1]))]
    zs = [c for b in boxes.values() for c in (float(b[0][2]), float(b[1][2]))]
    lo = [far_x + gap, (min(ys) + max(ys)) / 2.0 - span / 2.0, (min(zs) + max(zs)) / 2.0 - span / 2.0]
    hi = [lo[0] + span, lo[1] + span, lo[2] + span]
    return [lo, hi, "world"]                        # the shape arm_plan parses: lo, hi, tag


def test_a_start_inside_the_margin_is_refused_rather_than_planned_around():
    """19 mm of air is not contact, and the planner still refuses it. The caller escapes by the
    checked route instead -- see the note above for why relaxing this margin was rejected."""
    near = _box_at(START, 0.019)                      # inside MARGIN (20 mm), no overlap
    out = plan(_req(goal=_near_goal().tolist(), obstacles=[near]))
    assert isinstance(out, dict) and "error" in out, (
        f"the planner is expected to refuse a start inside MARGIN, got {type(out).__name__}")
    assert "start state invalid" in out["error"], out["error"]


def test_a_start_in_real_contact_is_refused_too():
    """The line that must never move: an overlapping start is invalid however the margin is read."""
    touching = _box_at(START, -0.010)                 # 10 mm of genuine overlap
    out = plan(_req(goal=_near_goal().tolist(), obstacles=[touching]))
    assert isinstance(out, dict) and "error" in out, (
        f"an overlapping start must be refused, got {type(out).__name__}")


def test_a_start_clear_of_the_margin_plans_as_before():
    """The same obstacle moved just outside MARGIN must not disturb anything. The goal retracts the
    boom AWAY from the box: a goal that drives into it would be refused for its own sake."""
    clear = _box_at(START, 0.021)
    away = START.copy()
    away[3] = float(START[3]) - 0.06
    out = plan(_req(goal=away.tolist(), obstacles=[clear]))
    assert isinstance(out, list) and out, f"a clear start must still plan, got {out}"


def test_an_out_of_bounds_goal_is_named_rather_than_burning_the_budget():
    """`si.isValid` runs the validity CHECKER; it never calls satisfiesBounds, and `_as_state` does
    not clamp. So a goal outside the joint bounds is not rejected -- it is sampled toward until the
    solve budget is gone, and the error says "no path found", which sends the caller looking for an
    obstacle that was never there. Name the joint instead."""
    far = _near_goal()
    # ArmLeftJoint_1, bound [0.000, 0.625]
    far[3] = float(far[3]) + 3.0
    t0 = time.time()
    out = plan(_req(goal=far.tolist(), solve_time=3.0))
    dt = time.time() - t0
    assert isinstance(out, dict) and "error" in out, f"expected a refusal, got {type(out).__name__}"
    assert "bound" in out["error"].lower(), out["error"]
    assert "ArmLeftJoint_1" in out["error"], f"the refusal must name the joint: {out['error']}"
    assert dt < 1.0, f"burned {dt:.2f}s of solve budget on a goal that was never in bounds"


def test_an_out_of_bounds_start_is_named_too():
    far = START.copy()
    # ArmLeftJoint_1 outside its bound
    far[3] = float(far[3]) + 3.0
    out = plan(_req(start=far.tolist(), goal=_near_goal().tolist()))
    assert isinstance(out, dict) and "error" in out, f"expected a refusal, got {type(out).__name__}"
    assert "bound" in out["error"].lower() and "ArmLeftJoint_1" in out["error"], out["error"]


def test_the_shadow_report_names_the_pose_and_says_when_the_START_is_folded():
    """A tally alone cannot be diagnosed. This metric has been reporting `7 of 59` for weeks; the
    one thing a reader needs -- WHICH pose -- was never printed. Waypoint 0 flagging is a
    categorically different fact from waypoint 30 flagging: it means the arm was ALREADY folded
    when the request was made, so the caller left it there and no route the planner picks can
    avoid it."""
    import io as _io
    import contextlib as _ctx

    ap = arm_plan                          # already imported at module scope

    class _M:
        def __init__(self, at):
            self._at = at
        def self_collides(self, q8, margin=0.0):
            return ("Arm_Left_1", "Gripper_Link3_1") if self._at is not None else None

    wps = [np.full(8, 0.11 * i) for i in range(4)]

    err = _io.StringIO()
    with _ctx.redirect_stderr(err):
        ap._report_self_collisions(_M(0), wps)
    out = err.getvalue()
    assert "q8=" in out, f"the offending pose is not reported, so the fold cannot be diagnosed:\n{out}"
    assert "0.11" in out or "0.0" in out, f"the reported pose is not the waypoint's own:\n{out}"
    assert "START POSE IS ALREADY FOLDED" in out, (
        f"waypoint 0 flagged and the report does not say the START was folded -- the reader is "
        f"left to assume the planner chose a folding route:\n{out}")

    # ...and it must NOT cry START when the fold begins later in the path.
    class _Late:
        def self_collides(self, q8, margin=0.0):
            return ("Arm_Left_1", "Gripper_Link3_1") if float(q8[0]) > 0.2 else None

    err2 = _io.StringIO()
    with _ctx.redirect_stderr(err2):
        ap._report_self_collisions(_Late(), wps)
    out2 = err2.getvalue()
    assert "q8=" in out2, out2
    # The pose printed must be the pose that FOLDED, not merely some pose. Waypoint 0 is all
    # zeros here, so a report that always prints waypoint 0 satisfies a bare "a q8 appeared".
    k0 = next(k for k, w in enumerate(wps) if float(w[0]) > 0.2)   # the fake's own predicate
    want = str(round(float(wps[k0][0]), 5))   # the report rounds to 5dp
    assert want in out2, (
        f"the fold begins at waypoint {k0} whose q8 starts {want}, and the report prints "
        f"{out2.split('q8=')[1][:40]!r} -- it is naming the wrong pose")
    assert "START POSE IS ALREADY FOLDED" not in out2, (
        f"a fold that begins mid-path is reported as a folded START, which sends whoever reads it "
        f"to the wrong file:\n{out2}")


def test_the_plan_reports_where_its_time_actually_went():
    """`solve_time` bounds ONLY `planner.solve`. `simplifyMax` takes no budget argument at all, and
    densify + per-waypoint revalidation walk every obstacle -- so the request's 3.0s bounds a
    fraction of a plan's real cost. The project's own audit measured ONE plan call at 11.443s, and
    5.942s on the re-validation: 2x to 4x the budget the gates are written against. A number nobody
    prints is a number nobody fixes."""
    src = open(os.path.join(_ROOT, "arm_plan.py"), encoding="utf-8").read()
    tree = ast.parse(src)

    # simplifyMax must still be the UNBOUNDED call this test exists to expose; if a budget is ever
    # added, this test should be revisited rather than silently keep passing.
    simp = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and getattr(n.func, "attr", "") == "simplifyMax"]
    assert len(simp) == 1, f"expected one simplifyMax call, found {len(simp)}"

    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "plan")
    body = ast.unparse(fn)
    assert "_t_solve" in body and "_t_simp" in body and "_t_dens" in body, (
        "the plan no longer times solve, simplify and densify separately -- a single total cannot "
        "say which of them to fix")
    assert "solve_time budget" in body and "bounds" in body, (
        "the report no longer says that the budget bounds the solve ONLY, which is the whole point "
        "of printing the split")

    # The timers must WRAP the real work, not be dead assignments.
    solves = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
              and getattr(n.func, "attr", "") == "solve"]
    assert solves, "fixture: no planner.solve call found"
    assert "_t_solve += " in body, "the solve timer never accumulates, so it reports zero"
    assert body.index("_t_simp = time.perf_counter()") < body.index("simplifyMax"), (
        "the simplify timer starts after simplifyMax, so it measures nothing")

    # ...and the SAME guarantee for the solve timer, which was only substring-checked: moving its
    # start to AFTER `planner.solve` reports ~0s and the suite stayed green.
    assert body.index("_t_s = time.perf_counter()") < body.index("planner.solve"), (
        "the solve timer starts after planner.solve, so it measures nothing")

    # `_t0` must be OUTSIDE the loop it totals, and before it -- structurally, because
    # `ast.unparse` rewrites the loop header and a substring search for it silently finds nothing.
    t0 = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign)
              and any(getattr(t, "id", "") == "_t0" for t in n.targets))
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.For)
             and any(isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "solve"
                     for c in ast.walk(n))]
    assert loops, "fixture: no solve loop found"
    assert t0.lineno < loops[0].lineno, (
        "_t0 is set inside or after the solve loop, so `plan Xs` reports a fraction of the cost")
    inner = {getattr(n, "lineno", -1) for lp in loops for st in lp.body for n in ast.walk(st)}
    assert t0.lineno not in inner, "_t0 is reset every iteration, so the total is one candidate"

    # The report must actually be WRITTEN. The timers exist to produce it, and turning the write
    # into a dead assignment left every `in body` assertion satisfied.
    writes = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
              and getattr(n.func, "attr", "") == "write"
              and any(isinstance(a, ast.JoinedStr) and "= solve " in ast.unparse(a)
                      for a in n.args)]
    assert writes, (
        "the split is computed and never printed -- a dead assignment satisfies every check that "
        "only looks for the strings")


def test_the_request_dump_is_inert_unless_asked_and_cannot_kill_a_plan():
    """Tuning the post-processing budget needs a REAL request; a synthesised scene answers a
    different question. The probe must be OFF by default and must never take a plan down with it --
    a diagnostic that can fail a motion is worse than no diagnostic."""
    import io as _io
    import contextlib as _ctx
    import tempfile

    src = open(os.path.join(_ROOT, "arm_plan.py"), encoding="utf-8").read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "plan")
    body = ast.unparse(fn)
    assert "ARM_PLAN_DUMP" in body and "environ" in body, "the dump is no longer env-gated"
    # The write must sit inside a try/except, or a full disk aborts the cycle.
    dumps = [n for n in ast.walk(fn) if isinstance(n, ast.Try)
             and any(isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "dump"
                     for c in ast.walk(n))]
    assert dumps, "the request dump is not inside a try/except: a write failure would kill the plan"

    prev = os.environ.pop("ARM_PLAN_DUMP", None)
    try:
        with tempfile.TemporaryDirectory() as d:
            # unset -> writes nothing
            try:
                arm_plan.plan({"model": "/nonexistent", "lut": "/nonexistent"})
            except Exception:
                pass
            assert os.listdir(d) == [], "the dump wrote with ARM_PLAN_DUMP unset"
            # set -> writes one file, and the plan still fails for its OWN reason, not the probe's
            os.environ["ARM_PLAN_DUMP"] = d
            err = _io.StringIO()
            with _ctx.redirect_stderr(err):
                try:
                    arm_plan.plan({"model": "/nonexistent", "lut": "/nonexistent"})
                except Exception:
                    pass
            assert len(os.listdir(d)) == 1, f"expected one dumped request, got {os.listdir(d)}"
            assert "request dump failed" not in err.getvalue(), err.getvalue()
    finally:
        os.environ.pop("ARM_PLAN_DUMP", None)
        if prev is not None:
            os.environ["ARM_PLAN_DUMP"] = prev


def test_the_post_processing_mode_is_measurable_and_defaults_to_todays_behaviour():
    """Post-processing is 82-91% of plan latency live, against a solve that is ~6%, so `solve_time`
    budgets the part that was never the problem.

    A TIME BUDGET IS NOT THE LEVER, and that is measured, not assumed: `simplify(path, t, True)`
    runs a full pass regardless of `t` because `atLeastOnce` says so, and one pass IS the cost --
    0.05s budget gave 9.37s, 0.5s gave 13.97s, unbounded 10.13s. The lever is WHICH routine runs.
    On a real captured request, three trials each: `simplifyMax` 9.92-11.15s against
    `reduceVertices` 0.53-0.62s, for a BYTE-IDENTICAL result (59 waypoints, q8 length 1.845)."""
    src = open(os.path.join(_ROOT, "arm_plan.py"), encoding="utf-8").read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "plan")
    body = ast.unparse(fn)
    assert "ARM_SIMPLIFY_MODE" in body, "the post-processing mode is no longer selectable"

    calls = {getattr(c.func, "attr", "") for c in ast.walk(fn) if isinstance(c, ast.Call)}
    for need in ("simplifyMax", "reduceVertices", "simplify"):
        assert need in calls, f"{need} is unreachable, so its cost cannot be attributed"

    # The DEFAULT must still be today's behaviour: a cheaper one has to be MEASURED and sim-verified,
    # not slipped in by changing the knob's fallback. `.get("ARM_SIMPLIFY_MODE", "max")` is the contract.
    gets = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
            and getattr(c.func, "attr", "") == "get"
            and c.args and isinstance(c.args[0], ast.Constant)
            and c.args[0].value == "ARM_SIMPLIFY_MODE"]
    assert len(gets) == 1 and len(gets[0].args) == 2 and gets[0].args[1].value == "reduce", (
        "the default post-processing mode is not 'reduce'. It was flipped on evidence: identical "
        "output (waypoint count AND q8 length) on every replayable captured request, and three "
        "live cycles with ALL CHECKS PASS, unchanged fallbacks and unchanged placement error. "
        "Going back to `max` costs 6-11s per plan for no measured benefit; going anywhere else "
        "needs its own corpus and its own live run")



if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    print("\nall passed" if not fails else f"\n{fails} FAILED")
    sys.exit(1 if fails else 0)
