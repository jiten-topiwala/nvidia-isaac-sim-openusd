"""The place insert's boom-vs-chassis-body floor, and the `PLACE_DH_FLOOR` that clears it.
Offline, no Isaac:  .venv/bin/python3 tests/test_place_body.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morph.arm.model import ArmModel  # noqa: E402
from morph.config import ARM_MODEL, CLOSURE_LUT, PLACE_DH_FLOOR, A1_MAX      # noqa: E402

M = ArmModel.load(ARM_MODEL, CLOSURE_LUT)
MARGIN = 0.02                                     # morph/arm/collision.py MARGIN
# The battery twice: the visual union `KinematicsMixin.BODY_BOX` and the PhysX collider inside it.
BODY_BOX = np.array([[-0.3165, -0.2960, 0.1586], [-0.0219, +0.2809, +0.3310]])
COLLIDER = np.array([[-0.310, -0.290, 0.165], [-0.030, 0.270, 0.305]])
VIS = [(BODY_BOX, "chassis")]

# The shipped low-slot insert (perfect/drive, OBJ_HALF_H 0.09): slide, trim, 28 mm lower, settle.
TH, A1_SLIDE, A1_SEAT = 0.099, 0.456, 0.341
H1_SLIDE, H1_SEAT, H1_SETTLED = 0.077, 0.049, 0.043


def q8(th, h1, dh, a1):
    q = np.zeros(8)
    q[0], q[1], q[2], q[3] = th, h1, h1 + dh, a1
    return q


def lut_z(dh, a1):
    """The LUT grip z -- the invariant `morph/place/__init__.py`'s `_h1fp` line holds."""
    i, fi = M._cell(M.lut["dh"], dh)
    j, fj = M._cell(M.lut["a1"], a1)
    g = M.lut["grip"]
    return ((1 - fi) * (1 - fj) * g[i][j][2] + fi * (1 - fj) * g[i + 1][j][2]
            + (1 - fi) * fj * g[i][j + 1][2] + fi * fj * g[i + 1][j + 1][2])


def with_floor(th, h1, dh, a1):
    """The pose the pre-dock actually commands at `dh`: h1 re-solved to hold the grip height."""
    return q8(th, h1 + lut_z(0.0, a1) - lut_z(dh, a1), dh, a1)


def test_the_visual_box_contains_the_collider():
    """One obstacle is enough in `_body_hit` / `_body_clear` only if it bounds the other."""
    assert (BODY_BOX[0] <= COLLIDER[0]).all() and (BODY_BOX[1] >= COLLIDER[1]).all()


def test_the_shipped_low_slot_place_puts_the_boom_inside_the_battery():
    """The defect. Not a margin violation: `collides` at margin 0 is True at every phase."""
    for h1, a1 in ((H1_SLIDE, A1_SLIDE), (H1_SEAT, A1_SEAT), (H1_SETTLED, A1_SEAT)):
        q = q8(TH, h1, 0.0, a1)
        assert M.first_hit(q, VIS, 0.0) == "Arm_Left_1", (h1, a1)
        assert M.first_hit(q, [(COLLIDER, "chassis")], 0.0) == "Arm_Left_1", (h1, a1)
        assert M.clearance(q, VIS) == -1.0


def test_the_floor_clears_every_phase_of_the_insert():
    """`PLACE_DH_FLOOR`, through the grip-height invariant, clears the margin at every phase."""
    got = {}
    for tag, h1, a1 in (("slide", H1_SLIDE, A1_SLIDE), ("trim", H1_SLIDE, A1_SEAT),
                        ("seat", H1_SEAT, A1_SEAT), ("settle", H1_SETTLED, A1_SEAT)):
        q = with_floor(TH, h1, PLACE_DH_FLOOR, a1)
        assert not M.collides(q, VIS, MARGIN), (tag, M.first_hit(q, VIS, MARGIN))
        assert not M.passive_violations(PLACE_DH_FLOOR, a1) and M.in_lut_domain(PLACE_DH_FLOOR, a1)
        got[tag] = M.clearance(q, VIS)
    assert got["settle"] == min(got.values()), got
    assert got["settle"] >= MARGIN, got


def test_the_floor_is_the_bisected_one_not_a_rounded_guess():
    """Re-derive it: the constant must be at or above the smallest clearing dh, and within 2 mm."""
    lo, hi = 0.0, 0.045
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if M.collides(with_floor(TH, H1_SETTLED, mid, A1_SEAT), VIS, MARGIN):
            lo = mid
        else:
            hi = mid
    assert hi <= PLACE_DH_FLOOR <= hi + 0.002, (hi, PLACE_DH_FLOOR)


def test_mid_and_high_keep_the_level_boom():
    """The seat h1 tracks the slot height 1:1, so mid and high must not move with the fix."""
    for slot_z in (0.687, 1.190):
        q = q8(TH, H1_SETTLED + (slot_z - 0.247), 0.0, A1_SEAT)
        assert M.first_hit(q, VIS, MARGIN) is None, slot_z
        assert M.clearance(q, VIS) > 0.25
    # low hits
    assert M.first_hit(q8(TH, H1_SETTLED, 0.0, A1_SEAT), VIS, MARGIN) == "Arm_Left_1"


def test_the_reach_survives_the_floor():
    """`_reach_rot` at A1_MAX against the 0.638 m the low slots demand."""
    def reach(dh, a1):
        mx, my = M.lut["mount"][0], M.lut["mount"][1]
        i, fi = M._cell(M.lut["dh"], dh)
        j, fj = M._cell(M.lut["a1"], a1)
        g = M.lut["grip"]
        p = [((1 - fi) * (1 - fj) * g[i][j][k] + fi * (1 - fj) * g[i + 1][j][k]
              + (1 - fi) * fj * g[i][j + 1][k] + fi * fj * g[i + 1][j + 1][k]) for k in (0, 1)]
        return mx + np.cos(TH) * (p[0] - mx) - np.sin(TH) * (p[1] - my)

    assert reach(PLACE_DH_FLOOR, A1_MAX) >= 0.638 + 0.10
    assert reach(0.0, A1_MAX) - reach(PLACE_DH_FLOOR, A1_MAX) < 0.015
    assert not M.passive_violations(PLACE_DH_FLOOR, A1_MAX)


if __name__ == "__main__":
    fails = 0
    for name in sorted(n for n in dir() if n.startswith("test_")):
        try:
            globals()[name]()
            print(f"PASS {name}")
        except Exception as e:
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{'all passed' if not fails else str(fails) + ' failed'}")
    sys.exit(1 if fails else 0)
