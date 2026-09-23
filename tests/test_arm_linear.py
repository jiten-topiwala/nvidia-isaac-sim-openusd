"""The PREMISE the closed-form column ramp rests on, and the profile it flies.

The method itself is private to `morph.arm` and is exercised through `move_linear` in
tests/test_arm_api.py -- nothing outside the package may import it. What lives here needs only the
kinematic model, and is what must hold for the method to be valid at all.
    ./.venv/bin/python3 tests/test_arm_linear.py
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from morph.arm.model import ArmModel  # noqa: E402
from morph.config import A1_MAX, COLUMN_MAX                               # noqa: E402

M = ArmModel.load(os.path.join(ROOT, "usd", "_arm_model.json"),
                  os.path.join(ROOT, "usd", "_closure_lut.json"))
HAND = M.hand_link
IH1 = ArmModel.Q8.index("ColumnLeftBearingJoint_1")
IH2 = ArmModel.Q8.index("ColumnRightBearingJoint_1")
DT = 1.0 / 240.0


def _bounds():
    lo, hi = M.bounds()
    hi = hi.copy()
    hi[1] = min(hi[1], COLUMN_MAX)
    hi[2] = min(hi[2], COLUMN_MAX)
    hi[3] = min(hi[3], A1_MAX)
    return lo, hi


BOUNDS = _bounds()
GRASP = np.array([float(json.load(open(os.path.join(ROOT, "usd/_grasp_known.json")))
                        ["arm_joints"][n]) for n in ArmModel.Q8])


def _poses():
    """Several real column/boom combinations, not one: the closure is nonlinear in (dh, a1)."""
    out = [GRASP.copy()]
    for h1, dh, a1 in ((0.45, 0.00, 0.25), (0.80, 0.05, 0.40), (0.60, 0.11, 0.30)):
        q = GRASP.copy()
        q[IH1], q[IH2], q[3] = h1, h1 + dh, a1
        out.append(q)
    return out


def test_an_equal_column_move_is_an_EXACT_straight_line_in_the_hand_frame():
    """The premise the whole method rests on. Both column joints are parallel prismatics and `dh`
    is the closure's only input, so an equal increment translates everything above the bearings
    rigidly. If this ever stops holding, the closed form is invalid and must not be selected."""
    for q in _poses():
        p0, R0 = M.fk(q)[HAND]
        for dz in (0.10, -0.03, 0.001):
            q_to = q.copy()
            q_to[IH1] += dz
            q_to[IH2] += dz
            p1, R1 = M.fk(q_to)[HAND]
            d = p1 - p0
            assert abs(d[0]) < 1e-9 and abs(d[1]) < 1e-9, (
                f"an equal-column move moved the hand laterally by "
                f"({d[0] * 1000:.6f}, {d[1] * 1000:.6f})mm")
            assert abs(d[2] - dz) < 1e-9, f"asked {dz}, hand moved {d[2]}"
            assert float(np.abs(R1 - R0).max()) < 1e-12, (
                f"an equal-column move rotated the hand by {np.abs(R1 - R0).max():.3e}")


def test_the_ramp_flies_the_profile_the_insert_asked_for():
    """What replaced the cosine ramp on the CARRY, pinned on the real geometry rather than on a
    reimplementation of the loop it replaced.

    `_fly_ramp` checks an eased sequence and hands the executor the start pose plus that sequence;
    the executor collapses it (one straight line in joint space) and retimes it under the joint
    ceilings, with `secs` as a floor. So what the drives receive is a trapezoid of at least the
    asked duration, landing exactly on the endpoint, inside every ceiling."""
    from morph.arm.cartesian import _A_MAX, _V_MAX, q8_derate, retime_q8
    for secs, dz in ((1.0, 0.03), (1.0, 0.05), (1.2, 0.05)):
        q_to = GRASP.copy()
        q_to[IH1] += dz
        q_to[IH2] += dz
        n = max(1, int(secs / DT))
        eased = [GRASP + (0.5 - 0.5 * math.cos(math.pi * (k + 1) / n)) * (q_to - GRASP)
                 for k in range(n)]
        wps = np.array([GRASP] + eased)
        q = retime_q8(wps, DT, q8_derate(wps, secs))[1]

        flown = (len(q) - 1) * DT
        assert flown >= secs, f"dz {dz}, ask {secs}s: flown {flown:.4f}s is under the ask"
        assert flown < secs + 4 * DT, f"dz {dz}, ask {secs}s: flown {flown:.4f}s"
        assert np.allclose(q[0], GRASP) and np.allclose(q[-1], q_to), "the ramp missed an endpoint"

        d = np.diff(q, axis=0)
        assert float((np.abs(d) / DT / _V_MAX).max()) <= 1.0 + 1e-9, "past the velocity ceiling"
        assert float((np.abs(np.diff(d, axis=0)) / DT ** 2 / _A_MAX).max()) <= 1.0 + 1e-9, (
            "past the acceleration ceiling")
        assert float(np.abs(np.diff(q[:, IH2] - q[:, IH1])).max()) < 1e-12, (
            "the columns stopped moving together, so the hand tilts")

        # The shape, not just the bounds: a pure column delta under `_V_MAX**2/_A_MAX` is
        # acceleration-bound, so its peak rate is twice the mean. Above that it is a trapezoid.
        peak = float(np.abs(d[:, IH1]).max()) / DT
        assert abs(peak / (dz / flown) - 2.0) < 0.02, (
            f"dz {dz}, ask {secs}s: peak/mean {peak / (dz / flown):.3f}, neither a triangle (2.0) "
            f"nor the cosine it replaced (1.571)")


if __name__ == "__main__":
    fails = 0
    for name in sorted(n for n in dir() if n.startswith("test_")):
        try:
            globals()[name]()
            print(f"PASS {name}")
        except Exception as e:
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("\nall passed" if not fails else f"\n{fails} failed")
    sys.exit(1 if fails else 0)
