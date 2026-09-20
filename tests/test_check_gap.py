"""F1b: the EXECUTED path must be subdivided on measured geometry, per link, not on a joint delta.

Offline: no Isaac, no simulator. Run: ./.venv/bin/python3 tests/test_check_gap.py

`_densify` used to emit until no JOINT moved more than `STEP`, and the executor cosine-eases
between consecutive waypoints while checking only the waypoints -- so one emitted segment is one
unchecked interval. The column closure turns a fixed joint step into anywhere from 26 mm to 353 mm
of corner travel (13x), so a joint delta bounds nothing geometric.

The criterion is PER LINK, because tunnelling is: a box passes clean through an obstacle unseen
only if it moves further than its OWN thinnest dimension plus the margin it is grown by on each
side -- `ArmModel.tunnel_limits`. One global number cannot serve both the 21 mm
`Gripper_Link1_1` and the 290 mm `Arm_1`: sized for the first it over-refines every segment on the
account of the second, sized for the second it leaves the first tunnelling. Measured over one
`STEP` segment, FIVE links overrun their own limit today, worst `Gripper_Link1_1` at 3.66x.
"""
import ast
import functools
import io
import json
import math
import os
import sys
from contextlib import redirect_stderr

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools"))
import arm_plan                                                          # noqa: E402
from morph.arm.collision import MARGIN                                    # noqa: E402
import measure_check_gap                                                 # noqa: E402
from arm_plan import MAX_DEPTH, STEP                                     # noqa: E402
from morph.arm.model import ArmModel  # noqa: E402

MODEL = os.path.join(_ROOT, "usd", "_arm_model.json")
M = ArmModel.load(MODEL, os.path.join(_ROOT, "usd", "_closure_lut.json"))
LIM = M.tunnel_limits(margin=MARGIN)
OBJH = np.array([0.0015, 0.0507, 0.1703])    # held-object centre in the hand frame, per test_armfk

# Found by sampling the differential-column direction over states that pass the planner's own
# validity gate: sub-segments that each move a collision-box corner further than the box is thick.
DIFF_START = np.array([-0.155453, 0.709718, 0.732852, 0.476701,
                       0.53777, 0.91126, 0.326273, 0.144675])
DIFF_END = DIFF_START + np.array([0.0, 0.06, -0.06, 0.0, 0.0, 0.0, 0.0, 0.0])
# Turret-only: the same STEP moves every box a fraction of its own limit, so nothing may be refined.
ROT_START = M.Q8_HOME.copy()
ROT_END = ROT_START + np.array([0.5, 0, 0, 0, 0, 0, 0, 0.0])


@functools.lru_cache(maxsize=1)
def _measured_step_shifts():
    """Seed-0 random and pure-differential samples, 4,000 valid STEP pairs apiece."""
    random = measure_check_gap.measure(M, "densify", samples=4000, seed=0)
    differential = measure_check_gap.measure(
        M, "densify", samples=4000, seed=0, direction=[0, 1, -1, 0, 0, 0, 0, 0])
    return random, differential


class _Path:
    """The two methods `_densify` uses off an `og.PathGeometric`, over plain Q8 vectors."""

    def __init__(self, pts):
        self.pts = [np.asarray(p, float) for p in pts]

    def getStateCount(self):
        return len(self.pts)

    def getState(self, k):
        q = self.pts[k]
        return [[q[i] for i in arm_plan.LIN], [q[i] for i in arm_plan.ROT]]


def _dense(pts, start=None, goals=None, held=None):
    """`arm_plan._densify` over a plain list of Q8 vectors, with its endpoint arguments."""
    req = {"start": [float(v) for v in (pts[0] if start is None else start)]}
    g = [[float(v) for v in pts[-1]]] if goals is None else goals
    return arm_plan._densify(_Path(pts), req, g, M, held, MARGIN)


def _worst_ratio(out, held=None):
    """The closest any box on the emitted path comes to its own limit; >= 1.0 can tunnel."""
    lim = M.tunnel_limits(held, MARGIN)
    return max(max(d / lim[k] for k, d in M.corner_shifts(a, b, held).items())
               for a, b in zip(out[:-1], out[1:]))


def _old_count(pts):
    """How many waypoints the per-joint `STEP` floor alone emits -- the pre-F1b behaviour."""
    return 1 + sum(max(1, int(math.ceil(np.max(np.abs(np.asarray(c) - np.asarray(a)) / STEP))))
                   for a, c in zip(pts[:-1], pts[1:]))


def test_the_limit_is_the_links_own_box_plus_both_margins_and_no_obstacle_term():
    """The limit is DERIVED from `link_aabb`, not a table of numbers: correcting a box moves it.
    `Gripper_Link3_1` is checked at its unioned FINGER_SWEEP size because that is the box
    `first_hit` tests -- shrink the sweep and this tightens by itself, which is the point.

    NO obstacle thickness is added. Adding one assumes an INFINITE SLAB -- that the body must
    cross the obstacle's full depth to pass through it. A finite shelf board is tunnelled by
    grazing its EDGE, where the depth to cross is zero, so an allowance for it bounds nothing."""
    for link, box in M.link_aabb.items():
        want = float((box[1] - box[0]).min()) + 2 * MARGIN
        assert abs(LIM[link] - want) < 1e-12, f"{link}: {LIM[link]} != {want}"
    assert abs(LIM["Gripper_Link1_1"] - 0.061) < 5e-5, LIM["Gripper_Link1_1"]
    assert abs(LIM["Arm_1"] - 0.330) < 5e-5, LIM["Arm_1"]
    assert abs(LIM["Gripper_Link3_1"] - 0.174) < 5e-5, (
        f"the palm limit is {LIM['Gripper_Link3_1']}, not the unioned 134 + 40 mm -- "
        f"tunnel_limits is not reading the box first_hit tests")


def test_the_limits_follow_the_model_rather_than_a_hardcoded_table():
    """Halve a link's thinnest dimension in a COPY of the model and its limit must move with it."""
    raw = json.load(open(MODEL))
    raw["link_aabb"]["Arm_Right_1"] = [[-0.5, -0.5, -0.0035], [0.5, 0.5, 0.0035]]
    alt = ArmModel(raw, M.lut)
    assert abs(alt.tunnel_limits(margin=MARGIN)["Arm_Right_1"] - (0.007 + 0.04)) < 1e-12, (
        "tunnel_limits ignored the model's box")


def test_five_links_overrun_their_own_limit_on_one_step_segment_today():
    """The defect, quantified against the current `thin_i + 2 * MARGIN` tunnel limits using
    4,000 seed-0 random valid STEP pairs plus 4,000 fixed pure-differential pairs. The observed
    maximum is pinned on BOTH sides: a drop changes the defect; a rise consumes cap headroom."""
    w, differential = _measured_step_shifts()
    over = {k: v / LIM[k] for k, v in w.items() if v > LIM[k]}
    assert set(over) == {"Gripper_Link1_1", "Hand_Bearing_1", "Gripper_Link2_1",
                         "Arm_Left_1", "Gripper_Link3_1"}, sorted(over)
    assert over["Gripper_Link1_1"] == max(over.values()), (
        f"the thinnest hand box is no longer the worst offender: {over}")
    assert over["Gripper_Link1_1"] > 3.0, f"only {over['Gripper_Link1_1']:.2f}x, expected ~3.7x"
    worst = max(v / LIM[k] for sample in (w, differential) for k, v in sample.items())
    assert worst > 4.30, f"only {worst:.3f}x, expected the pure-differential maximum near 4.36x"
    assert worst < 4.40, f"{worst:.3f}x exceeds the seed-0, 8,000-pair combined ceiling"


def test_a_differential_column_segment_is_subdivided_under_every_links_own_limit():
    """THE INVARIANT, adversarially. The differential column direction is where the closure
    amplifies hardest: one `STEP` here moves a corner further than the whole hand is thick, and
    the executor interpolates straight through it with no check."""
    third = DIFF_START + (DIFF_END - DIFF_START) / 3.0
    raw = max(d / LIM[k] for k, d in M.corner_shifts(DIFF_START, third).items())
    assert raw > 3.0, (f"the adversarial segment only reaches {raw:.2f}x a link's limit -- the "
                       f"geometry changed and this test no longer tests anything")
    out = _dense([DIFF_START, DIFF_END])
    got = _worst_ratio(out)
    assert got <= 1.0, (f"an emitted segment reaches {got:.2f}x a link's own tunnelling limit "
                        f"({len(out)} waypoints)")


def test_the_held_object_gets_its_own_limit_from_its_own_box():
    """The box that goes between the boards is the OBJECT, and `first_hit` tests it like a link.
    It is not in `link_aabb`, so it has to pick up a limit from the AABB the caller passes."""
    held = np.vstack([OBJH - [0.040, 0.090, 0.040], OBJH + [0.040, 0.090, 0.040]])
    assert abs(M.tunnel_limits(held, MARGIN)["__held__"] - (0.080 + 0.04)) < 1e-12
    out = _dense([DIFF_START, DIFF_END], held=held)
    lim = M.tunnel_limits(held, MARGIN)
    got = max(M.corner_shifts(a, b, held)["__held__"] / lim["__held__"]
              for a, b in zip(out[:-1], out[1:]))
    assert got <= 1.0, f"the held object reaches {got:.2f}x its own limit between waypoints"


def test_a_thin_link_binds_before_a_wide_one():
    """The reason for per-link. On the adversarial segment `Gripper_Link1_1` is what forces the
    subdivision; the palm, which moves FURTHER, is nowhere near its own limit."""
    out = _dense([DIFF_START, DIFF_END])
    a, b = out[0], out[1]
    s = M.corner_shifts(a, b)
    assert s["Gripper_Link3_1"] > s["Gripper_Link1_1"], "the palm no longer moves further"
    assert (s["Gripper_Link1_1"] / LIM["Gripper_Link1_1"]
            > s["Gripper_Link3_1"] / LIM["Gripper_Link3_1"]), (
        "the thin knuckle is not the binding link -- per-link buys nothing here")


# A thin board and one `STEP` sub-segment that walks straight through it, found by searching for
# a board every waypoint clears while the motion between them does not.
BOARD_A = np.array([-2.053798, 0.796959, 0.752012, 0.492145,
                    -1.260964, -2.746637, 1.058879, -0.49725])
BOARD_B = np.array([-2.065904, 0.814074, 0.732469, 0.492094,
                    -1.249612, -2.766637, 1.065053, -0.495635])
BOARD = [(np.array([[-0.072751, -0.385368, 1.539597],
                    [-0.022751, -0.335368, 1.549597]]), "world")]


def test_a_board_the_old_waypoints_walk_through_is_caught():
    """THE property this task exists to prevent, tested against an OBSTACLE rather than against
    the helper's own arithmetic -- the helper is the production rule and cannot also be its oracle.

    Both ends of this one `STEP` sub-segment are clear of a 10 mm board. 453 of 2001 densely
    sampled interior states are not. The old per-joint floor emitted exactly the two endpoints, so
    `plan` accepted a path that drives the palm through the board; the per-link refinement emits
    a waypoint inside it, and `plan`'s own validity sweep then throws the path out."""
    fine = [BOARD_A + (BOARD_B - BOARD_A) * t for t in np.linspace(0, 1, 2001)]
    hit = [i for i, q in enumerate(fine) if M.collides(q, BOARD, MARGIN)]
    assert M.first_hit(fine[hit[len(hit) // 2]], BOARD, MARGIN) == "Gripper_Link3_1", (
        "the board no longer intercepts the palm mid-segment")
    assert 300 < len(hit) < 700, f"{len(hit)}/2001 interior states collide, expected ~453"
    assert not M.collides(fine[0], BOARD, MARGIN), "the segment START already collides"
    assert not M.collides(fine[-1], BOARD, MARGIN), "the segment END already collides"

    old = [BOARD_A + (BOARD_B - BOARD_A) * (i / _n(BOARD_A, BOARD_B))
           for i in range(_n(BOARD_A, BOARD_B) + 1)]
    assert not any(M.collides(w, BOARD, MARGIN) for w in old), (
        "the old per-joint floor already caught this board -- it no longer shows the defect")

    out = _dense([BOARD_A, BOARD_B])
    assert any(M.collides(w, BOARD, MARGIN) for w in out), (
        f"the {len(out)} emitted waypoints all clear a board the path drives through: "
        f"plan() would accept a colliding path")


def test_refinement_catches_most_sampled_boards_the_coarse_path_walks_through():
    """Sample 200 thin boards in two hand links' swept AABBs and keep only boards a 101-state
    traversal hits while a real MAX_DEPTH=0 path misses. Refinement must catch a majority of that
    fixed seed-0 set. This proves an obstacle-level improvement over no refinement; it does NOT
    prove every board is caught or bound the swept volume of a rotating body."""
    fine = [DIFF_START + (DIFF_END - DIFF_START) * t for t in np.linspace(0, 1, 101)]
    depth = arm_plan.MAX_DEPTH
    try:
        arm_plan.MAX_DEPTH = 0
        with redirect_stderr(io.StringIO()):
            coarse = _dense([DIFF_START, DIFF_END])
    finally:
        arm_plan.MAX_DEPTH = depth
    refined = _dense([DIFF_START, DIFF_END])

    rng = np.random.default_rng(0)
    walked, caught = 0, 0
    for link in ("Gripper_Link3_1", "Gripper_Link1_1"):
        swept = np.array([M.link_aabbs(q)[link] for q in fine])
        lo, hi = swept[:, 0].min(axis=0), swept[:, 1].max(axis=0)
        for _ in range(100):
            centre = lo + rng.random(3) * (hi - lo)
            half = np.full(3, 0.025)
            half[rng.integers(3)] = 0.005
            board = [(np.vstack([centre - half, centre + half]), "world")]
            if any(M.collides(q, board, MARGIN) for q in coarse):
                continue
            if not any(M.collides(q, board, MARGIN) for q in fine):
                continue
            walked += 1
            caught += any(M.collides(q, board, MARGIN) for q in refined)

    assert walked >= 10, f"only {walked} sampled boards expose a coarse-path tunnel"
    assert caught * 4 >= walked * 3, (
        f"refinement caught only {caught} of {walked} sampled tunnels, under the pinned majority")


def _tunnel_degrade():
    """`morph.arm.plan.TUNNEL_DEGRADE`, by AST, so this file imports nothing from the package."""
    tree = ast.parse(open(os.path.join(_ROOT, "morph", "arm", "plan.py")).read())
    return next(n.value.value for n in tree.body
                if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
                and any(getattr(t, "id", None) == "TUNNEL_DEGRADE" for t in n.targets))


def _n(a, b):
    """The old per-joint sub-segment count for one OMPL segment."""
    return max(1, int(math.ceil(np.max(np.abs(np.asarray(b) - np.asarray(a))) / STEP)))


def test_a_thin_held_object_binds_before_every_link_does():
    """Decision: the held object IS bounded, on its OWN box, not left out. It is collision geometry
    `first_hit` tests, and its limit comes from the AABB the caller passes -- so a flat object gets
    a tighter limit than any link and forces the extra refinement by itself. For the three SHIPPED
    object sizes the 71 mm knuckle still binds first, so including it costs nothing today; it is
    what stops a thin payload tunnelling while every link sits inside its bound."""
    tray = np.vstack([OBJH - [0.12, 0.005, 0.12], OBJH + [0.12, 0.005, 0.12]])   # 10 mm thick
    lim = M.tunnel_limits(tray, MARGIN)
    assert abs(lim["__held__"] - 0.050) < 1e-12, lim["__held__"]
    assert lim["__held__"] < lim["Gripper_Link1_1"], "the tray is not the tightest box"
    third = DIFF_START + (DIFF_END - DIFF_START) / 3.0
    r = {k: d / lim[k] for k, d in M.corner_shifts(DIFF_START, third, tray).items()}
    assert r["__held__"] > max(v for k, v in r.items() if k != "__held__"), (
        f"the tray is not the binding box: {r}")
    bare, carried = _dense([DIFF_START, DIFF_END]), _dense([DIFF_START, DIFF_END], held=tray)
    assert len(carried) > len(bare), (
        f"carrying a 10 mm tray emitted {len(carried)} waypoints, no more than the {len(bare)} "
        f"emitted empty-handed -- the held box is not being refined against")
    assert _worst_ratio(carried, tray) <= 1.0, (
        f"the tray still reaches {_worst_ratio(carried, tray):.2f}x its own limit")


def test_a_snapped_endpoint_is_refined_and_not_snapped_over():
    """Snapping moves `out[0]` and `out[-1]` off the states OMPL handed over. Subdividing FIRST and
    snapping after would refine one path and emit a different one, leaving exactly the two outer
    segments unrefined -- so the snap happens first. Both endpoints here are perturbed by far more
    than a solve ever would, which is what makes the ordering visible."""
    start = DIFF_START + np.array([0, 0.02, -0.02, 0, 0, 0, 0, 0.0])
    goal = DIFF_END + np.array([0, 0.02, -0.02, 0, 0, 0, 0, 0.0])
    goals = [[float(v) for v in goal], [float(v) for v in ROT_START]]
    out = _dense([DIFF_START, DIFF_END], start=start, goals=goals)
    assert np.array_equal(out[0], np.array([float(v) for v in start])), "out[0] moved"
    assert np.array_equal(out[-1], np.array(goals[0], float)), "out[-1] is not the reached goal"
    got = _worst_ratio(out)
    assert got <= 1.0, (f"after snapping, an emitted segment reaches {got:.2f}x a link's limit -- "
                        f"the outer segments were refined against states that were then replaced")


def test_the_per_joint_step_floor_is_preserved():
    """`STEP` stays a FLOOR: subdivision may only ever make the spacing finer than it was."""
    for pts in ([DIFF_START, DIFF_END], [ROT_START, ROT_END], [ROT_START, DIFF_START, DIFF_END]):
        out = _dense(pts)
        worst = max(float(np.max(np.abs(b - a))) for a, b in zip(out[:-1], out[1:]))
        assert worst <= STEP + 1e-12, (f"a joint moves {worst:.5f} between waypoints, over the "
                                       f"{STEP} floor")


def _on(w, a, c, tol=1e-9):
    """Is `w` on the straight line from `a` to `c`, inside the segment?"""
    d = np.asarray(c) - np.asarray(a)
    t = float((np.asarray(w) - a) @ d / (d @ d))
    return -tol <= t <= 1 + tol and float(np.max(np.abs(a + d * t - w))) <= tol


def test_the_emitted_waypoints_stay_on_the_straight_line_between_ompl_states():
    """Bisection may only insert points ON the segment; a waypoint off the line is a different
    path from the one the planner validated."""
    out = _dense([ROT_START, DIFF_START, DIFF_END])
    for a, c in ((ROT_START, DIFF_START), (DIFF_START, DIFF_END)):
        assert any(_on(w, a, c) for w in out), f"no emitted waypoint lies on {a} -> {c}"
    for w in out:
        assert _on(w, ROT_START, DIFF_START) or _on(w, DIFF_START, DIFF_END), (
            f"waypoint {w} is off both segments of the path")


def test_the_endpoints_are_the_callers_start_and_the_reached_goal():
    """The endpoint contract outranks the limits: `out[0]` is the caller's exact start and
    `out[-1]` the goal candidate the path actually terminated on, bit for bit."""
    start = DIFF_START + 1e-7                       # a start that is NOT the path's first state
    goals = [[float(v) for v in DIFF_END + 1e-7], [float(v) for v in ROT_START]]
    out = _dense([DIFF_START, DIFF_END], start=start, goals=goals)
    assert np.array_equal(out[0], np.array([float(v) for v in start])), (
        f"out[0] is not the caller's start: {out[0]} vs {start}")
    assert np.array_equal(out[-1], np.array(goals[0], float)), (
        f"out[-1] is not the reached goal: {out[-1]} vs {goals[0]}")


def test_a_one_state_path_keeps_the_callers_start_instead_of_the_goal_snap_eating_it():
    """On a 1-state path `pts[0]` and `pts[-1]` are the SAME row, so the goal snap overwrites the
    caller's start and the emitted path begins where it should end. OMPL has never handed one over
    -- a solution carries both endpoints -- but the endpoint contract is not conditional on that."""
    goals = [[float(v) for v in DIFF_END]]
    out = _dense([DIFF_START], start=DIFF_START, goals=goals)
    assert np.array_equal(out[0], DIFF_START), (
        f"the goal snap ate the caller's start: {out[0]} vs {DIFF_START}")
    assert np.array_equal(out[-1], np.array(goals[0], float)), f"out[-1] is not the goal: {out[-1]}"
    assert _worst_ratio(out) <= 1.0, "the degenerate path skipped refinement"


def test_a_revolute_segment_is_not_refined_beyond_the_old_behaviour():
    """LAZINESS. A turret sweep moves every box well inside its own limit, so subdivision must
    cost it NOTHING -- a global bound sized for the 21 mm knuckle would refine this too."""
    pts = [ROT_START, ROT_END]
    out = _dense(pts)
    assert _worst_ratio(out) < 1.0, "the turret sweep is not actually inside the limits"
    assert len(out) == _old_count(pts), (f"a revolute-only segment grew from {_old_count(pts)} to "
                                         f"{len(out)} waypoints -- refinement is not adaptive")
    assert len(out) == 26, f"the turret sweep emits {len(out)} waypoints, not the pinned 26"


def test_the_depth_cap_terminates_and_is_reported_not_swallowed():
    """Bisection cannot run away on a locally stiff closure. A piece still over a link's limit at
    the cap is a real condition: counted and printed on the channel `arm_plan` already degrades on,
    never silently accepted. (`_fallback` is unreachable here -- it is a method on the Isaac-side
    Demo, and this file runs in the planner subprocess.)"""
    depth = arm_plan.MAX_DEPTH
    try:
        # no bisection at all: every piece is over
        arm_plan.MAX_DEPTH = 0
        err = io.StringIO()
        with redirect_stderr(err):
            out = _dense([DIFF_START, DIFF_END])
    finally:
        arm_plan.MAX_DEPTH = depth
    assert len(out) == _old_count([DIFF_START, DIFF_END]), (
        f"the cap did not stop subdivision: {len(out)} waypoints")
    # The capped path is still a COMPLETE, contract-satisfying path: refusing would fall back to a
    # baked trajectory with no collision checking at all. The cap reports; it does not withhold.
    assert np.array_equal(out[0], DIFF_START) and np.array_equal(out[-1], DIFF_END), (
        "the depth cap broke the endpoint contract")
    assert max(float(np.max(np.abs(b - a))) for a, b in zip(out[:-1], out[1:])) <= STEP + 1e-12, (
        "the depth cap dropped below the per-joint STEP floor")
    msg = err.getvalue()
    assert msg.startswith("[arm_plan] 3 of 3 emitted segments can tunnel"), (
        f"an over-limit segment was not reported: {msg!r}")
    assert "Gripper_Link1_1" in msg, f"the report does not name the offending link: {msg!r}"
    # The degrade crosses a PROCESS boundary as text: `_plan_arm` matches `TUNNEL_DEGRADE` in this
    # stderr to count it. Reword the message without that constant and the cap is silently accepted.
    assert _tunnel_degrade() in msg, (
        f"morph/arm/plan.py matches {_tunnel_degrade()!r}, which is not in {msg!r}")
    err2 = io.StringIO()
    with redirect_stderr(err2):
        _dense([ROT_START, ROT_END])
    assert not err2.getvalue(), f"a clean path reported a degrade: {err2.getvalue()!r}"


def test_the_degrade_names_no_obstacle_thickness_it_can_no_longer_bound():
    """The message may only claim what the criterion bounds. With no obstacle term in the limit, a
    segment over its limit is not known to pass a 10 mm board -- only some thinner one -- so naming
    a thickness there would over-state the degrade. The `TUNNEL_DEGRADE` substring, which crosses
    the process boundary into `morph/arm/plan.py`, has to survive that rewording."""
    depth = arm_plan.MAX_DEPTH
    try:
        arm_plan.MAX_DEPTH = 0
        err = io.StringIO()
        with redirect_stderr(err):
            _dense([DIFF_START, DIFF_END])
    finally:
        arm_plan.MAX_DEPTH = depth
    msg = err.getvalue()
    assert _tunnel_degrade() in msg, f"the cross-process contract broke: {msg!r}"
    assert "mm obstacle" not in msg, (
        f"the degrade still claims an obstacle thickness the limit no longer carries: {msg!r}")


def test_max_depth_has_headroom_over_the_worst_measured_segment():
    """The cap must not be reachable in normal operation, or it becomes a silent quality knob.
    Derive headroom from 4,000 seed-0 random and 4,000 pure-differential valid STEP pairs."""
    samples = _measured_step_shifts()
    worst = max(v / LIM[k] for sample in samples for k, v in sample.items())
    need = math.ceil(math.log2(worst))
    assert MAX_DEPTH >= need + 3, (f"MAX_DEPTH {MAX_DEPTH} leaves under 8x headroom over the "
                                   f"depth {need} the worst measured segment needs")


def test_corner_shift_is_the_boxes_first_hit_tests():
    """The helper is BOTH the production rule and the tool's measurement, so it cannot also be its
    own oracle. Rebuild the eight corners here from every `_links(q, held)` entry -- the iteration
    `first_hit` itself walks -- without calling the helper, and compare. A dropped link, a wrong
    corner correspondence or the wrong frame shows up here and nowhere else."""
    rng = np.random.default_rng(0)
    lo, hi = M.bounds()
    held = np.vstack([OBJH - [0.055, 0.070, 0.055], OBJH + [0.055, 0.070, 0.055]])
    for h in (None, held):
        for _ in range(15):
            qa, qb = lo + rng.random(8) * (hi - lo), lo + rng.random(8) * (hi - lo)
            got = M.corner_shifts(qa, qb, h)
            want = {}
            for (link, local, pa, Ra), (l2, _, pb, Rb) in zip(M._links(qa, h), M._links(qb, h)):
                assert link == l2, f"_links yielded {link} then {l2}: no corner correspondence"
                c = np.array([[x, y, z] for x in local[:, 0] for y in local[:, 1]
                              for z in local[:, 2]])
                want[link] = float(np.linalg.norm((pb + c @ Rb.T) - (pa + c @ Ra.T), axis=1).max())
            # exactly the boxes first_hit walks -- nothing dropped, nothing invented
            assert set(want) == set(M.link_aabb) | ({"__held__"} if h is not None else set())
            assert got.keys() == want.keys(), f"links differ: {got.keys()} vs {want.keys()}"
            for k in want:
                assert abs(got[k] - want[k]) < 1e-12, f"{k}: {got[k]} vs {want[k]}"


def test_the_measurement_tool_has_no_second_copy_of_the_corner_maths():
    """One helper, two consumers. A private copy in the tool is how the tool and the planner drift
    into disagreeing about the very quantity the criterion is stated in."""
    src = open(os.path.join(_ROOT, "tools", "measure_check_gap.py")).read()
    assert not hasattr(measure_check_gap, "_corners"), "the tool still defines its own _corners"
    assert "itertools" not in src, "the tool still enumerates box corners itself"
    assert "corner_shifts" in src, "the tool does not use ArmModel.corner_shifts"
    assert "tunnel_limits" in src, "the tool does not compare against ArmModel.tunnel_limits"
    assert "BOUND" not in src, "the tool still names arm_plan.BOUND, which exists nowhere"


def test_the_tools_copied_constants_still_match_the_planners():
    """The tool cannot import `arm_plan` -- that pulls in OMPL, and the tool is numpy-only -- so it
    hand-copies the two intervals it measures. This file imports both, so the drift that would
    silently invalidate `test_five_links_overrun_...` costs one assert to catch."""
    assert measure_check_gap.STEP == STEP, (
        f"the tool measures a {measure_check_gap.STEP} STEP, the planner emits at {STEP}")
    assert measure_check_gap.RES == arm_plan.RES, (
        f"the tool measures a {measure_check_gap.RES} RES, the planner checks at {arm_plan.RES}")


def _open_fingers():
    """The eleven finger joints at the open pose -- the hand a planned approach actually flies."""
    return {n: -0.55 if n.endswith("_joint_1_1") else 0.0
            for n in ("finger_a_joint_1_1", "finger_a_joint_2_1", "finger_a_joint_3_1",
                      "finger_b_joint_1_1", "finger_b_joint_2_1", "finger_b_joint_3_1",
                      "finger_c_joint_1_1", "finger_c_joint_2_1", "finger_c_joint_3_1",
                      "palm_finger_b_joint_1", "palm_finger_c_joint_1")}


def test_the_densifier_and_the_checker_measure_THE_SAME_HAND():
    """`_worst` does `lim[k]` over `corner_shifts`' keys, so a 21-link shift table against a
    12-link limit table raises `KeyError('finger_a_link_1_1')` -- which `arm_plan` wraps as
    `{"error": ...}`, whose text contains no "invalid", so `_arm_blame` never runs and the operator
    reads "no route found" for a crash.

    Mutation: drop `fingers` from `tunnel_limits`' body while keeping the parameter -- must raise.
    """
    f = _open_fingers()
    a, b = DIFF_START, DIFF_START + 0.01
    shifts = M.corner_shifts(a, b, None, f)
    # Anti-vacuity: a `corner_shifts` that IGNORED `fingers` would pass the subset check trivially.
    pads = [k for k in shifts if "finger" in k]
    assert len(pads) == 9, f"corner_shifts ignored `fingers`: {sorted(shifts)}"
    lim = M.tunnel_limits(margin=MARGIN, fingers=f)
    assert set(shifts) <= set(lim), f"the densifier cannot bound {set(shifts) - set(lim)}"
    # the KeyError path, exercised end to end
    arm_plan._worst(M, lim, None, a, b, f)


def test_the_hand_limit_follows_the_hand_the_checker_tests():
    """Without fingers the hand rides its swept union; with them, the palm. The limit must follow,
    because it is a bound on the box `first_hit` actually tests.

    Mutation: leave `tunnel_limits`' hand entry on `link_aabb` when `fingers` is given -- must fail.
    """
    swept = M.tunnel_limits(margin=MARGIN)[M.hand_link]
    palm = M.tunnel_limits(margin=MARGIN, fingers=_open_fingers())[M.hand_link]
    assert abs(swept - 0.174) < 1e-3, f"the swept hand limit moved: {swept}"
    assert palm < swept - 0.05, f"the palm limit did not tighten: {palm} vs {swept}"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("\nall passed" if not fails else f"\n{fails} FAILED")
    sys.exit(1 if fails else 0)
