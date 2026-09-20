"""`_solve_lowest` must not return a pose the mechanism cannot reach.
Offline, no Isaac:  .venv/bin/python3 tests/test_solve_limits.py
"""
import ast
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from morph.arm.model import ArmModel  # noqa: E402

M = ArmModel.load(os.path.join(ROOT, "usd", "_arm_model.json"),
                  os.path.join(ROOT, "usd", "_closure_lut.json"))


def test_the_mirror_boom_is_what_actually_caps_a1():
    """The premise, and the number no constant can express. `ArmRightJoint_1` is a PRISMATIC passive
    driven by the closure, so a1's ceiling falls as the columns tilt. A fixed `A1_MAX` cannot be
    right at more than one tilt."""
    stops = {dh: next((a for a in np.arange(0.30, 0.63, 0.0005)
                       if M.passive_violations(dh, float(a))), None)
             for dh in (0.0, 0.10, 0.1294, 0.15)}
    assert all(v is not None for v in stops.values()), stops
    assert stops[0.0] > stops[0.10] > stops[0.1294] > stops[0.15], (
        f"a1's ceiling is not falling with tilt: {stops}")
    assert 0.455 < stops[0.1294] < 0.467, (
        f"the descent's tilt caps a1 near 0.4615, measured {stops[0.1294]:.4f}")
    # and the shipped grasp sits inside it, while A1_MAX does not
    assert not M.passive_violations(0.1294, 0.4367), "the shipped grasp is illegal"
    assert M.passive_violations(0.1294, 0.50), (
        "A1_MAX=0.50 is legal at the descent tilt -- this whole test is moot")


def _solver():
    """`_solve_lowest` on a bare mixin. kinematics.py imports one Isaac symbol at module level."""
    import types
    _m = types.ModuleType("isaacsim")
    _c = types.ModuleType("isaacsim.core")
    _p = types.ModuleType("isaacsim.core.prims")
    _p.SingleXFormPrim = object
    _c.prims = _p
    _m.core = _c
    sys.modules.setdefault("isaacsim", _m)
    sys.modules.setdefault("isaacsim.core", _c)
    sys.modules.setdefault("isaacsim.core.prims", _p)
    import json as _json
    from morph.kinematics import KinematicsMixin

    class _D(KinematicsMixin):
        clut = _json.load(open(os.path.join(ROOT, "usd", "_closure_lut.json")))

        def _arm_model(self):
            return M

    return _D()


# The solve predicts the pinch now, so it needs the pose it predicts AT: the turret and wrist it
# will fly, and the hand. `off_z` is gone -- there is no scalar left to derive or to get wrong.
def _descent_q8t():
    """The shipped descent's turret and wrist, ArmModel.Q8 order. h1/h2/a1 are per-candidate."""
    q = np.zeros(8)
    q[0] = DESCENT_TH
    return q


def _open_hand():
    return {n: (-0.55 if n.endswith("_joint_1_1") else 0.0) for n in
            ("finger_a_joint_1_1", "finger_a_joint_2_1", "finger_a_joint_3_1",
             "finger_b_joint_1_1", "finger_b_joint_2_1", "finger_b_joint_3_1",
             "finger_c_joint_1_1", "finger_c_joint_2_1", "finger_c_joint_3_1",
             "palm_finger_b_joint_1", "palm_finger_c_joint_1")}


# The shipped pick descent. The solve predicts the pinch per candidate now, so the logged h1 moved
# with it -- the older descent sat lower. That is staleness in the log, not a regression.
DESCENT_Z_TGT = 0.092
DESCENT_TH = -0.195          # turret angle through the descent, from the arm-body prints
DESCENT_DH_WANT = 0.13
RAMP_FOLD_A1 = 0.247      # what reach.py actually descends at; the solved a1 is discarded


def test_solve_lowest_never_returns_a_pose_that_puts_the_boom_in_the_chassis_body():
    """THE test the source-text one below could not be: a behavioural assertion on the RETURN.

    Adding the passive-limit filter refused the shipped tilt and fell through to a scan whose only
    objective is maximum reach, with no body check at all -- `_clears_chassis` is a grip-POINT test.
    It returned h1 0.0122 / dh 0.055 / reach 0.7215, which `_body_hit` reports as `Arm_Left_1`
    INSIDE the chassis body box. The simulator found that, not the suite: every offline test passed.

    Both halves: the pose must be legal for the closure AND clear of the body."""
    from morph.config import CHASSIS_PLATE_BOX
    d = _solver()
    sol = d._solve_lowest(DESCENT_Z_TGT, dh_want=DESCENT_DH_WANT, h1_min=0.0,
                          th=DESCENT_TH, q8_t=_descent_q8t(), fingers=_open_hand())
    assert sol is not None, "the descent no longer solves at all"
    h1, h2, a1, _reach = sol
    hit = d._body_hit(DESCENT_TH, h1, h2 - h1, a1)
    assert hit is None, (
        f"returned h1 {h1:.4f} dh {h2 - h1:+.4f} a1 {a1:.4f} -- {hit} is inside the chassis body")
    # And the constraint that actually binds, measured at the a1 the RAMP uses (the folded 0.247),
    # not the solver's proxy a1: the boom must keep its clearance to the deck it descends past.
    q = np.zeros(8)
    q[0], q[1], q[2], q[3] = DESCENT_TH, h1, h2, RAMP_FOLD_A1
    deck = np.asarray(CHASSIS_PLATE_BOX, float)
    assert M.first_hit(q, [(deck, "chassis")], 0.0, None, None, links=("Arm_Left_1",)) is None, (
        f"h1 {h1:.4f} puts Arm_Left_1 INSIDE the deck at the ramp's a1 -- filtering the proxy a1 "
        f"against the joint stop cost 32mm of column height and did exactly this")
    gap = float(M.clearance(q, [(deck, "chassis")], None, None))
    assert gap > 0.005, f"boom-to-deck down to {gap * 1000:.1f}mm"


def test_a_refused_tilt_degrades_to_the_nearest_tilt_not_another_pose_family():
    """`dh_want` is a REQUEST from the descent, which has already chosen its approach around that
    tilt. When the requested tilt is refused, falling through to "maximum reach at any tilt" answers
    a different question: it swapped dh 0.130/h1 0.198 for dh 0.055/h1 0.012, bottoming the columns
    186mm below the floor `_reach_lowest` enforces (morph/close/lowest.py, _h1_min = 0.20)."""
    d = _solver()
    # dh_want 0.13 is ACCEPTED outright and never reaches the fall-through, so it would pin nothing.
    # 0.02 is refused on the geometry itself and degrades to 0.035.
    assert abs(_tilt_of(d, DESCENT_DH_WANT) - DESCENT_DH_WANT) < 1e-6, (
        "0.13 stopped being accepted directly -- pick a new pair for this test")
    refused = 0.02
    got = _tilt_of(d, refused)
    assert abs(got - refused) > 1e-6, (
        f"{refused} is now accepted directly; this test is vacuous -- find a tilt the geometry "
        f"still refuses rather than loosening the assertion below")
    assert abs(got - refused) <= 0.03, (
        f"a refused tilt degraded from {refused:+.3f} to {got:+.3f} -- a different pose family, "
        f"which is what bottomed the columns 186mm below the floor the next stage enforces")


def _tilt_of(d, dh_want):
    sol = d._solve_lowest(DESCENT_Z_TGT, dh_want=dh_want, h1_min=0.0,
                          th=DESCENT_TH, q8_t=_descent_q8t(), fingers=_open_hand())
    assert sol is not None, f"no solve at dh_want {dh_want}"
    return sol[1] - sol[0]
# `_solve_reach_z` is the place-insert dual of `_solve_lowest`, and unlike it the returned a1 IS
# commanded -- insert.py interpolates it into the slide waypoints.
REACH_Z_VIOLATING = (0.45, 0.15)     # (reach_tgt, z_tgt) -> dh +0.185 a1 0.430, 15.3mm past


def test_solve_reach_z_never_returns_a_pose_the_closure_cannot_reach():
    """The sibling blind spot. `_solve_lowest`'s a1 is discarded by its caller; this one's is
    COMMANDED, so an unreachable return is driven into the slide, not just printed.

    Today's runs escape only because the shipped place always takes the UNTILTED branch and never
    calls this at all (5/5 in logs/bench_archive/p50_probe.log). That makes the filter a no-op on
    current trajectories -- and makes the top-shelf exception, where straight cannot reach and this
    IS called, the case nothing has ever checked."""
    d = _solver()
    reach, z = REACH_Z_VIOLATING
    sol = d._solve_reach_z(reach, z)
    if sol is None:
        # refusing is the other correct answer
        return
    h1, h2, a1 = sol[0], sol[1], sol[2]
    v = M.passive_violations(h2 - h1, a1)
    assert not v, (
        f"reach {reach} z {z} -> dh {h2 - h1:+.4f} a1 {a1:.4f}, which the closure cannot reach: {v}")


def test_the_shipped_place_never_needs_the_tilt_solve():
    """Pins WHY the filter above is safe to add. If a future change makes the shipped place take
    the tilt branch, this fails and the filter's effect stops being hypothetical."""
    import re
    log = os.path.join(ROOT, "logs", "bench_archive", "p50_probe.log")
    if not os.path.exists(log):
        return
    txt = open(log, encoding="utf-8", errors="replace").read()
    assert txt.count("insert[slide]: UNTILTED") >= 5, "the baseline stopped taking the untilted branch"
    assert "insert[slide]: tilt REQUIRED" not in txt, (
        "the shipped place now calls _solve_reach_z -- the filter is no longer a no-op, re-measure")


def test_a1_ceiling_agrees_with_the_rule_that_ENFORCES_it():
    """`a1_ceil` must invert the same channel `passive_violations` reads -- the bilinear LUT, not
    the closed form. The two disagree by up to 0.36mm BETWEEN grid nodes, always with the closed
    form HIGH, so a clamp built on the closed form lands past what the checker accepts and the pose
    is rejected anyway. Sampled on and between nodes, and at the tilts production actually holds."""
    # DENSE, on purpose: a sparse sample missed a 1-ULP disagreement affecting hundreds of states.
    # `passives` evaluates the same bilinear in a different order, so agreement needs the domain.
    for dh in np.linspace(M.lut["dh"][0], M.lut["dh"][-1], 801):
        c = M.a1_ceil(dh)
        assert not M.passive_violations(dh, c), (
            f"a1_ceil({dh:.5f}) = {c:.6f} is REJECTED by passive_violations -- the clamp does not "
            f"agree with the rule that enforces it")
        assert M.passive_violations(dh, c + 2e-4), (
            f"a1_ceil({dh:.5f}) = {c:.6f} is more than 0.2mm below the real ceiling -- it is "
            f"giving away reach")


def test_the_solver_keeps_out_of_the_passive_stop_it_now_knows_about():
    """Holding the iterate to the EXACT ceiling parks solutions on a hard stop: measured 19 of 484
    within 0.1mm, and `robot._apply` does not clip a pose that is exactly in range, so it executes
    with no headroom. The keep-out is the repo's measured model error, and the property that
    matters is that no solution comes nearer than it."""
    from morph.arm.model import PASSIVE_KEEPOUT_M
    from morph.config import A1_MAX, COLUMN_MAX
    lim = next(lo for n, lo, _ in M.passive_lims if n == "ArmRightJoint_1")
    lo, hi = M.bounds()
    lo, hi = lo.copy(), hi.copy()
    hi[1] = min(hi[1], COLUMN_MAX); hi[2] = min(hi[2], COLUMN_MAX); hi[3] = min(hi[3], A1_MAX)
    rng = np.random.default_rng(3)
    worst, n = 1.0, 0
    for _ in range(120):
        dh = rng.uniform(0.08, 0.16)
        h1 = rng.uniform(0.15, 0.45)
        q = np.array([0.0, h1, h1 + dh, rng.uniform(0.30, 0.46), 0, 0, 0, 0])
        if M.passive_violations(dh, q[3]) or not M.in_lut_domain(dh, q[3]):
            continue
        pos, R = M.fk(q)[M.hand_link]
        c = M.ik(pos + rng.normal(0, 0.03, 3), R, seed=q, bounds=(lo, hi), restarts=0)
        if c is None:
            continue
        n += 1
        worst = min(worst, M.passives(c[0][2] - c[0][1], c[0][3])["ArmRightJoint_1"] - lim)
    assert n > 50, f"only {n} solves -- this fixture no longer exercises the band"
    assert worst >= PASSIVE_KEEPOUT_M - 1e-4, (
        f"a solve came within {worst * 1000:.3f}mm of the passive stop, inside the "
        f"{PASSIVE_KEEPOUT_M * 1000:.0f}mm keep-out")


def test_a1_ceiling_falls_with_tilt_and_binds_before_the_slide_stop():
    """The whole point: a1's own USD stop (0.625) is never the binding limit, because the MIRROR
    boom reaches its stop first, and earlier the further the columns separate."""
    lo, hi = M.bounds()
    slide_stop = float(hi[ArmModel.Q8.index("ArmLeftJoint_1")])
    ceils = [M.a1_ceil(dh) for dh in (0.0, 0.05, 0.10, 0.1294, 0.20)]
    assert all(a > b for a, b in zip(ceils, ceils[1:])), f"ceiling not falling with tilt: {ceils}"
    assert max(ceils) < slide_stop, (
        f"the slide stop {slide_stop} is reachable at some tilt -- then it would be the real limit")


if __name__ == "__main__":
    fails = 0
    for name in sorted(n for n in dir() if n.startswith("test_")):
        try:
            globals()[name]()
            print(f"PASS {name}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL {name}: {e}")
    print("\nall passed" if not fails else f"\n{fails} failed")
    sys.exit(1 if fails else 0)