"""Pure geometry: quaternions, rotations, angle wrapping. No Isaac dependency -- keep it that way,
so it stays importable without `SimulationApp`. Quaternions are `[w, x, y, z]`, as in Isaac."""
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
    """Wrap an angle to [-pi, pi) -- half-open, so an exact +pi comes back as -pi."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def quat_to_R(q):
    """Rotation matrix from quaternion `[w, x, y, z]`."""
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
                     [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    """Quaternion `[w, x, y, z]` from a rotation matrix. Branches on the largest diagonal term
    because the trace form loses precision near a 180-degree rotation, where `s` collapses."""
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


def aabb_in_frame(aabb, p, R, fp, fR):
    """`aabb` (a body's own [min, max] in its own frame, world pose `(p, R)`) re-boxed in the frame
    at world `(fp, fR)`. Returns (2, 3) [min, max] RELATIVE to that frame -- the planner's `held` is
    posed at the hand link again, so an absolute box lands ~1 m off. Conservative: boxes 8 corners."""
    mn, mx = np.asarray(aabb[0], float), np.asarray(aabb[1], float)
    corners = np.array([[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1])
                        for z in (mn[2], mx[2])], float)
    v = (corners @ np.asarray(R, float).T + np.asarray(p, float)
         - np.asarray(fp, float)) @ np.asarray(fR, float)      # world -> frame: fR.T @ (w - fp)
    return np.vstack([v.min(axis=0), v.max(axis=0)])


def _self_check():
    """Round-trip and identity checks. Run: python3 morph/geometry.py"""
    assert abs(wrap(0.5) - 0.5) < 1e-12
    assert abs(wrap(3 * math.pi) + math.pi) < 1e-12
    assert abs(wrap(-3 * math.pi) + math.pi) < 1e-12
    assert abs(wrap(2 * math.pi + 0.25) - 0.25) < 1e-12
    for a in (-9.0, -1.0, 0.0, 1.0, 9.0, 100.0):
        assert -math.pi <= wrap(a) < math.pi, a

    box = np.array([[-0.04, -0.04, -0.09], [0.04, 0.04, 0.09]])
    fp, I = np.array([1.0, 2.0, 3.0]), np.eye(3)
    assert np.allclose(aabb_in_frame(box, fp, I, fp, I), box)
    Rz = quat_to_R(quat_yaw(math.pi / 2))
    got = aabb_in_frame(np.array([[-0.1, -0.02, -0.3], [0.1, 0.02, 0.3]]), fp, I, fp, Rz)
    assert np.allclose(got, [[-0.02, -0.1, -0.3], [0.02, 0.1, 0.3]], atol=1e-12), got

    for yaw in (0.0, 0.3, -1.2, 2.9, math.pi - 1e-6):
        assert abs(wrap(yaw_of(quat_yaw(yaw)) - yaw)) < 1e-9, yaw

    for q in ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
              [0.5, 0.5, 0.5, 0.5], [1e-8, 0.0, 0.0, 1.0]):
        q = np.array(q, float)
        q /= np.linalg.norm(q)
        R = quat_to_R(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9), q
        assert abs(abs(float(np.linalg.det(R))) - 1.0) < 1e-9, q
        back = R_to_quat(R)
        if float(back @ q) < 0:
            back = -back
        assert np.allclose(back, q, atol=1e-7), (q, back)

    print("[OK] morph.geometry self-check passed")


if __name__ == "__main__":
    _self_check()
