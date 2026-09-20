"""F1: the validity check must be fine enough to see a slab thinner than its own step.

Offline: no Isaac, runs OMPL directly. Run: ./.venv/bin/python3 tests/test_slab_gate.py

The planner's state space mixes METRES (3 prismatic joints) and RADIANS (5 revolute), and
`setStateValidityCheckingResolution` is a FRACTION OF THE SPACE'S MAXIMUM EXTENT, not a length. One
unweighted `RealVectorStateSpace` over all eight therefore validates a motion once every
0.005 * 10.63 = 53 mm of column travel, and the closure turns column travel into ~5.5x as much hand
travel -- so a slab tens of millimetres thick fits entirely between two checks and OMPL calls the
motion valid. `verify_place.py`'s rack has a 10 mm board between the low shelf (underside above at
0.677) and the mid shelf (surface 0.687); this test is that board, thickened to 20 mm so it is
unambiguously THINNER than the old step and THICKER than the new one.

The motion below is prismatic-only: h1 down 41 mm, h2 up 33 mm, everything else held. Its length in
the planner's own metric is 0.0526 -- just under one old `longestValidSegmentLength`, so the old
space checks the two endpoints and NOTHING in between.
"""
import os
import sys

import numpy as np
from ompl import base as ob

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import arm_plan                                                    # noqa: E402
from arm_plan import DOF, Validity, plan                           # noqa: E402
from morph.arm.model import ArmModel  # noqa: E402

MODEL = os.path.join(_ROOT, "usd", "_arm_model.json")
LUT = os.path.join(_ROOT, "usd", "_closure_lut.json")
M = ArmModel.load(MODEL, LUT)
MARGIN = 0.02
THICK = 0.020                      # slab thickness, metres

# Wrists at +90 deg turn the hand box's 282 mm axis out of the vertical, leaving 134 mm of hand to
# clear -- without that the hand is taller than one step can carry it and nothing can tunnel.
START = M.Q8_HOME.copy()
START[6] = START[7] = np.pi / 2
GOAL = START.copy()
GOAL[1] -= 0.041
GOAL[2] += 0.033

# Slab: horizontal, at the mid-height of the hand's sweep, and starting at x = 0.61 -- beyond the
# 0.585 + margin every OTHER link reaches, so the only thing that can touch it is the hand.
_Z = float(np.mean([(M.link_aabbs(q)[M.hand_link][:, 2]).mean() for q in (START, GOAL)]))
SLAB = [[[0.61, -0.30, _Z - THICK / 2], [1.20, 0.60, _Z + THICK / 2], "world"]]
_OBS = [(np.array([o[0], o[1]], float), o[2]) for o in SLAB]


def _req(**kw):
    r = {"model": MODEL, "lut": LUT, "start": START.tolist(), "goal": GOAL.tolist(),
         "obstacles": SLAB, "margin": MARGIN, "solve_time": 5.0, "seed": 1}
    r.update(kw)
    return r


def _si(obstacles):
    """The planner's own space and checker, wired exactly as `plan` wires them."""
    lo, hi = M.bounds()
    space = arm_plan._space(lo, hi) if hasattr(arm_plan, "_space") else _legacy_space(lo, hi)
    si = ob.SpaceInformation(space)
    checker = Validity(si, M, obstacles, None, MARGIN, None)
    si.setStateValidityChecker(checker)
    si.setStateValidityCheckingResolution(0.005)
    si.setup()
    return space, si, checker


def _legacy_space(lo, hi):
    space = ob.RealVectorStateSpace(DOF)
    b = ob.RealVectorBounds(DOF)
    for i in range(DOF):
        b.setLow(i, float(lo[i]))
        b.setHigh(i, float(hi[i]))
    space.setBounds(b)
    return space


def _state(space, q):
    return arm_plan._as_state(space, q)


def _crosses(path, obstacles, margin=0.0):
    """First q on the emitted path with a link OVERLAPPING the slab, sampled 20x finer than `STEP`.

    Margin 0 on purpose: `MARGIN` is clearance held at the states the planner CHECKS, and the
    simplifier parks a path right on that boundary, so the continuum between two waypoints 0.02
    apart routinely grazes into the margin by a few mm. Sweeping THROUGH the board is the property
    this file is about; how much clearance the continuum keeps is `STEP`'s question, not the
    validity resolution's."""
    P = np.asarray(path, float)
    for a, c in zip(P[:-1], P[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(c - a)) / 0.001)))
        for k in range(n + 1):
            q = a + (c - a) * (k / n)
            if M.collides(q, obstacles, margin):
                return q
    return None


def test_the_premise_the_slab_is_thinner_than_the_old_step():
    """Guards the measurement, not the code: a slab THICKER than the old step would be caught by
    the old resolution too and the gate below would pass for the wrong reason."""
    lo, hi = M.bounds()
    old = _legacy_space(lo, hi)
    old.setLongestValidSegmentFraction(0.005)
    old.setup()
    step = old.getLongestValidSegmentLength()
    assert THICK < step, f"slab {THICK} is not thinner than the old step {step}"
    assert np.linalg.norm(GOAL - START) < step, (
        f"the motion is {np.linalg.norm(GOAL - START):.6f} long, which the old space already "
        f"splits -- it must fit inside ONE segment ({step:.6f}) for the endpoints to be all it saw")


def test_the_endpoints_clear_the_slab():
    """So a rejected motion is rejected for the slab in the middle, not for its endpoints."""
    _sp, si, _c = _si(_OBS)
    for name, q in (("start", START), ("goal", GOAL)):
        assert si.isValid(_state(_sp, q)), f"{name} is already invalid; the gate proves nothing"


def test_the_slab_really_is_in_the_way():
    """The straight line between them genuinely passes through the slab."""
    q = _crosses([START, GOAL], _OBS)
    assert q is not None, "the straight-line motion never enters the slab; nothing to tunnel"
    assert M.first_hit(q, _OBS, 0.0) == M.hand_link, (
        f"expected the hand to be what strikes the slab, got {M.first_hit(q, _OBS, 0.0)}")


def test_ompl_refuses_the_motion_through_the_slab():
    """THE GATE. `checkMotion` is the planner's own verdict on a straight-line motion. With one
    mixed-unit space it validates only the endpoints and calls this valid -- the planner reasons
    through the board instead of around it. It must say invalid."""
    _sp, si, _c = _si(_OBS)
    assert not si.checkMotion(_state(_sp, START), _state(_sp, GOAL)), (
        "OMPL reports the motion through the slab VALID: the validity check steps over the slab "
        "entirely, which is the defect")


def test_the_same_motion_is_valid_with_the_slab_removed():
    """The counterpart: the refusal above is the slab, not some unrelated invalidity."""
    _sp, si, _c = _si([])
    assert si.checkMotion(_state(_sp, START), _state(_sp, GOAL)), (
        "the motion is invalid even with no obstacle at all, so the gate above measures nothing")


def test_the_prismatic_subspace_is_validated_finer_than_the_revolute_one():
    """The mechanism: `CompoundStateSpace::validSegmentCount` takes the MAX over subspaces, so the
    metres and the radians get their own step. A single space structurally cannot do this."""
    space, _si_, _c = _si([])
    assert isinstance(space, ob.CompoundStateSpace), f"still one flat space: {type(space)}"
    lin, rot = space.getSubspace(0), space.getSubspace(1)
    assert lin.getDimension() == 3 and rot.getDimension() == 5, (
        f"expected 3 prismatic + 5 revolute, got {lin.getDimension()} + {rot.getDimension()}")
    assert abs(lin.getLongestValidSegmentLength() - 0.005 * lin.getMaximumExtent()) < 1e-9
    assert lin.getLongestValidSegmentLength() < 0.011, (
        f"prismatic step is {lin.getLongestValidSegmentLength() * 1000:.1f} mm of column, not the "
        f"~10.6 mm the split buys")
    assert rot.getLongestValidSegmentLength() > 0.05, (
        "the revolute step collapsed too, which buys nothing and costs collision calls")


def test_the_plan_routes_around_the_slab():
    """End to end: `plan` must still SOLVE the slab problem, and no link on the path it emits may
    overlap the slab.

    This one does NOT discriminate before from after, and is not the gate. `_densify` re-validates
    every emitted waypoint at `STEP` = 0.02 per joint, and 0.02 of column is ~120 mm of hand
    against a hand box no thinner than 134 mm, so that second net already caught this slab on the
    old space -- at the cost of throwing the shortcut away and emitting the raw path. What the
    resolution fix buys end to end is that the planner stops REASONING through the slab; the guard
    here is that it did not stop solving."""
    out = plan(_req())
    assert isinstance(out, list) and out, f"expected waypoints, got {out}"
    q = _crosses(out, _OBS)
    assert q is None, f"the emitted path puts {M.first_hit(q, _OBS, 0.0)} through the slab at {q}"


def test_the_q8_lin_rot_split_is_the_pinned_index_sets():
    """Pins `ArmModel.Q8_LIN`/`Q8_ROT` themselves, not just their self-consistency. `_as_state` and
    `_q8` are inverse permutations of EACH OTHER for any split that partitions all eight indices --
    a round trip is exact whether the split is the real one, some other in-family reorder, or a
    cross-family swap, so the round-trip test below cannot see this bug class by itself. The turret
    is joint 0, so this is not a 3/5 slice down the middle: pin the two SETS (order within a family
    is free -- that reorder is a different, harmless permutation)."""
    assert set(ArmModel.Q8_LIN) == {1, 2, 3}, f"Q8_LIN drifted to {ArmModel.Q8_LIN}"
    assert set(ArmModel.Q8_ROT) == {0, 4, 5, 6, 7}, f"Q8_ROT drifted to {ArmModel.Q8_ROT}"
    assert set(ArmModel.Q8_LIN) | set(ArmModel.Q8_ROT) == set(range(DOF)), (
        "the two families must partition all eight joints -- none dropped, none doubled")


def test_q8_pack_unpack_round_trips_and_lands_in_each_joints_own_bounds():
    """The reviewer's 1,000-case probe, kept as a gate. Every one of 1000 random Q8 vectors, each
    sampled inside the real USD bounds, must survive `_as_state` -> `_q8` unchanged, per joint, and
    every packed/unpacked value must sit inside ITS OWN joint's bounds -- not merely inside
    whichever subspace it happened to land in."""
    lo, hi = M.bounds()
    space = arm_plan._space(lo, hi)
    rng = np.random.default_rng(0)
    for _ in range(1000):
        q = rng.uniform(lo, hi)
        q2 = arm_plan._q8(arm_plan._as_state(space, q))
        assert np.allclose(q2, q, rtol=0, atol=1e-9), f"round trip changed a joint: {q} -> {q2}"
        assert np.all(q2 >= lo) and np.all(q2 <= hi), (
            f"a packed/unpacked joint landed outside its own Q8 bounds: {q2} not in [{lo}, {hi}]")


def test_an_APPROXIMATE_solution_is_never_accepted_as_a_path():
    """`bool(PlannerStatus)` is True for an APPROXIMATE solution -- OMPL's own "Adding approximate
    solution from planner RRTConnect". Accepting one matters because `_densify` then SNAPS the last
    point onto the goal, manufacturing a final segment the motion validator never saw, whose
    interior is bounded only by `tunnel_limits`. And the snap hides it: the path now ends AT the
    goal, so `_arrival` passes and `reach-no-arrival` can never fire.

    The fixture is the hand-only slab with a budget too small to connect the trees. Today the plan
    is rejected INCIDENTALLY -- a snapped waypoint happens to be invalid -- so the test pins the
    REASON, not merely that something failed.
    """
    goal = START.copy()
    goal[1] -= 0.30
    # the hand must travel down through the slab
    goal[2] -= 0.30
    _zm = float(np.mean([(M.link_aabbs(q)[M.hand_link][:, 2]).mean() for q in (START, goal)]))
    slab = [[[0.61, -0.30, _zm - 0.01], [1.20, 0.60, _zm + 0.01], "world"]]
    obs = [(np.array([slab[0][0], slab[0][1]], float), "world")]
    assert M.first_hit(START, obs, MARGIN) is None, "fixture: the start is not clear of the slab"
    assert M.first_hit(goal, obs, MARGIN) is None, "fixture: the goal is not clear of the slab"

    out = plan(_req(goal=goal.tolist(), obstacles=slab, solve_time=0.002, seed=3))
    assert isinstance(out, dict), (
        f"an approximate solution was returned as a usable path of {len(out)} waypoints; "
        f"`_densify` snapped its end onto a goal the planner never reached")
    assert "approximate" in out["error"].lower(), (
        f"the plan was refused for the wrong reason: {out['error']!r}. It must be refused BECAUSE "
        f"the solution is approximate -- being caught downstream by densification is luck, and a "
        f"snap through free space would ship")


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
