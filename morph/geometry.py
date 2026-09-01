"""Pure geometry — quaternions, rotations, angle wrapping. Extracted from play_isaac.py.

The only module in the package with no Isaac dependency at all: it imports nothing but `math` and
`numpy`, so it is importable (and testable) without booting `SimulationApp`. Keep it that way —
anything needing the USD stage or the articulation belongs in `kinematics.py`, not here.

Quaternions are `[w, x, y, z]`, matching Isaac's convention.
"""
import math

import numpy as np


def quat_yaw(yaw):
    """Quaternion for a rotation of `yaw` radians about +Z."""
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def yaw_of(q):
    """Z-axis yaw angle of quaternion `q`, in radians."""
    w, x, y, z = [float(v) for v in q]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    """Wrap an angle to [-pi, pi).

    Note the half-open end: an exact +pi wraps to -pi, not +pi. Callers compare `abs(wrap(...))`
    against a tolerance, so the sign at the boundary does not matter to them.
    """
    return (a + math.pi) % (2 * math.pi) - math.pi


def quat_to_R(q):
    """Rotation matrix from quaternion `[w, x, y, z]`."""
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
                     [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    """Quaternion `[w, x, y, z]` from a rotation matrix.

    Branches on the largest diagonal term rather than always using the trace: the trace form loses
    precision as it approaches zero (a 180-degree rotation), where `s` collapses and the off-diagonal
    differences divide by ~0.
    """
    t = float(np.trace(R))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], float)
    return q / max(1e-12, float(np.linalg.norm(q)))


def _self_check():
    """Round-trip and identity checks. Run: python3 morph/geometry.py"""
    assert abs(wrap(0.5) - 0.5) < 1e-12
    assert abs(wrap(3 * math.pi) + math.pi) < 1e-12       # half-open: +pi lands on -pi
    assert abs(wrap(-3 * math.pi) + math.pi) < 1e-12
    assert abs(wrap(2 * math.pi + 0.25) - 0.25) < 1e-12
    for a in (-9.0, -1.0, 0.0, 1.0, 9.0, 100.0):
        assert -math.pi <= wrap(a) < math.pi, a

    for yaw in (0.0, 0.3, -1.2, 2.9, math.pi - 1e-6):
        assert abs(wrap(yaw_of(quat_yaw(yaw)) - yaw)) < 1e-9, yaw

    # quat -> R -> quat round-trip, including the near-180-degree case the branching exists for
    for q in ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
              [0.5, 0.5, 0.5, 0.5], [1e-8, 0.0, 0.0, 1.0]):
        q = np.array(q, float)
        q /= np.linalg.norm(q)
        R = quat_to_R(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9), q          # orthonormal
        assert abs(abs(float(np.linalg.det(R))) - 1.0) < 1e-9, q      # proper rotation
        back = R_to_quat(R)
        if float(back @ q) < 0:
            back = -back                                              # q and -q are the same rotation
        assert np.allclose(back, q, atol=1e-7), (q, back)

    print("[OK] morph.geometry self-check passed")


if __name__ == "__main__":
    _self_check()
