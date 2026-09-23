"""Straight-line METHOD choice: the closed-form column ramp, or nothing (the caller falls back).

PRIVATE to `morph.arm`. Everything outside the package goes through `api`.
"""
import numpy as np

from morph.arm.model import ArmModel

_IH1 = ArmModel.Q8.index("ColumnLeftBearingJoint_1")
_IH2 = ArmModel.Q8.index("ColumnRightBearingJoint_1")

# Lateral component under which a delta still counts as pure z, metres. One micron: far below the
# 0.005 arrival band, far above float noise on an FK round trip.
_LATERAL_TOL_M = 1e-6


def column_ramp(model, q8, delta, bounds):
    """`(q8_target, None)` for a base-frame `(0, 0, dz)` the columns can deliver alone (equal column
    increments translate the hand rigidly); `(None, None)` if the delta has no closed form -- not a
    refusal, the caller owes a Cartesian plan; `(None, reason)` if delivering it tilts the hand."""
    d = np.asarray(delta, float).ravel()
    if d.size != 3 or not np.all(np.isfinite(d)):
        return None, None
    if abs(d[0]) > _LATERAL_TOL_M or abs(d[1]) > _LATERAL_TOL_M:
        return None, None                       # not a pure-z ask: no closed form, not a refusal

    dz = float(d[2])
    q = np.asarray(q8, float).copy()
    lo, hi = bounds
    for i in (_IH1, _IH2):
        want = q[i] + dz
        if want < float(lo[i]) - 1e-12 or want > float(hi[i]) + 1e-12:
            return None, (f"the columns cannot deliver {dz * 1000:+.1f}mm: "
                          f"{ArmModel.Q8[i]} would reach {want:.4f}, outside "
                          f"[{float(lo[i]):.4f}, {float(hi[i]):.4f}]")
    q[_IH1] += dz
    q[_IH2] += dz
    return q, None
