"""Time-parameterise a geometric joint path: trapezoidal, per-joint limits, no blending.

Pure numpy, offline. `retime` takes the limits as arguments, so the caller owns the derate.

WHY THIS EXISTS. `morph/arm/execute.py::execute_path` and `plan_cartesian` both time their paths
through this module. Spending a fixed wall-clock budget over however many waypoints OMPL returned
makes speed follow the planner's vertex count, which is not a physical quantity, and bounds
nothing.

TOTG is rejected on purpose: it needs a differentiable path and buys that by inserting circular
blends that leave the polyline. The safety argument here is that the polyline OMPL returned was
collision-checked and is re-validated at every executor checkpoint, so deviating from it
invalidates the check that makes the motion safe. Every sample sits on a straight segment between
two of the polyline's own waypoints.

Holding the path exactly forces zero velocity at every corner -- non-zero velocity through a
direction change is unbounded acceleration. That is a theorem, not a preference. Straight runs are
merged first (`_collapse`) so the arm only stops where the path actually turns -- the executor
calls this once per checkpoint interval, so it also rests where it re-validates.

Jerk is deliberately NOT bounded. The measured failure mode is a VELOCITY discontinuity: a 1.5 m/s
command step zeroes `_apply`'s feedforward and asks the drive for 4126 N against a 1500 N cap. A
trapezoid's velocity is continuous so that cannot happen, and every derived v_max stays under
`_apply`'s 5 mm jump mask. Acceleration is still discontinuous at every corner. "Will not ring"
(every arm drive is zeta >= 1 against its MEASURED inertia) is NOT "jerk is bounded": it says
nothing about force-rate limits, structural resonance or backlash. Settling that needs a
commanded-vs-delivered torque trace from a running simulator. Ruckig is the upgrade path."""
import numpy as np

__all__ = ["retime", "free_secs"]


def _collapse(path, tol=1e-9):
    """Drop repeated waypoints, and interior waypoints within `tol` (cosine similarity) of the
    straight run through their neighbours -- close to collinear, not exactly. So a dropped point's
    OWN coordinates are not necessarily on the emitted path: the deviation is of order sqrt(tol)
    times the segment length (about 0.01 mm on a 1 m segment at tol=1e-9, measured empirically,
    not derived), small enough that the collision check the polyline already passed still holds,
    but not zero. First and last, and any waypoint NOT within `tol`, still land exactly."""
    out = [path[0]]
    for p in path[1:]:
        d = p - out[-1]
        nd = float(np.linalg.norm(d))
        if nd == 0.0:
            # a repeated vertex is not a segment
            continue
        if len(out) >= 2:
            prev = out[-1] - out[-2]
            npv = float(np.linalg.norm(prev))
            if npv > 0.0 and float(prev @ d) >= (1.0 - tol) * npv * nd:
                # same heading: extend, do not stop here
                out[-1] = p
                continue
        out.append(p)
    return np.asarray(out, float)


def _segment(delta, v_max, a_max):
    """(duration, cruise rate, acceleration) for one straight segment, parameterised by s in [0,1].

    q(s) = a + s*delta, so joint j moves at s'*|delta_j| and accelerates at s''*|delta_j|. Taking
    the tightest joint gives one scalar profile that honours every per-joint bound at once, which
    is what Q8 needs -- it mixes metres and radians and has no meaningful scalar norm."""
    m = np.abs(delta) > 0.0
    sv = float(np.min(v_max[m] / np.abs(delta[m])))
    sa = float(np.min(a_max[m] / np.abs(delta[m])))
    # too short to reach the cruise rate
    if sv * sv >= sa:
        return 2.0 / np.sqrt(sa), np.sqrt(sa), sa
    return 1.0 / sv + sv / sa, sv, sa


def _s_of(tau, dur, sv, sa):
    """The trapezoid (or triangle) evaluated at times `tau` in [0, dur]. Returns s in [0, 1]."""
    ta = min(sv / sa, dur / 2.0)                       # ramp time; dur/2 is the triangular case
    cruise = 0.5 * sa * ta * ta + sv * (tau - ta)
    s = np.where(tau <= ta, 0.5 * sa * tau * tau,
                 np.where(tau >= dur - ta, 1.0 - 0.5 * sa * (dur - tau) ** 2, cruise))
    return np.clip(s, 0.0, 1.0)


def free_secs(path, v_max, a_max):
    """The trajectory's duration BEFORE `retime` rounds each segment up to whole `dt` steps.

    Scaling the limits by (k, k**2) scales THIS by exactly 1/k; it does not scale the rounded
    duration, so a caller sizing a derate off `retime`'s own `t[-1]` lands under its own ask."""
    way = _collapse(np.atleast_2d(np.asarray(path, float)))
    if len(way) < 2:
        return 0.0
    d = way.shape[1]
    v = np.broadcast_to(np.asarray(v_max, float), (d,))
    a = np.broadcast_to(np.asarray(a_max, float), (d,))
    return float(sum(_segment(b - p, v, a)[0] for p, b in zip(way[:-1], way[1:])))


def retime(path, v_max, a_max, dt=1.0 / 240.0):
    """Time-parameterise a geometric joint path under per-joint limits on dt grid.

    path: (n, d) waypoints. Emitted samples stay strictly on segment lines.
    v_max, a_max: (d,) or scalar per-joint velocity and acceleration limits (>0).
    dt: control period. Segment durations round up to dt multiples (conservative).
    Returns (t, q): t (N,) times from 0 spaced by dt, q (N, d) starting and ending at rest.
    q[0] and q[-1] match the endpoints bit-for-bit."""
    path = np.atleast_2d(np.asarray(path, float))
    if path.ndim != 2 or path.shape[0] < 1:
        raise ValueError(f"path must be (n, d) with n >= 1, got {path.shape}")
    d = path.shape[1]
    v_max = np.broadcast_to(np.asarray(v_max, float), (d,)).astype(float)
    a_max = np.broadcast_to(np.asarray(a_max, float), (d,)).astype(float)
    if not np.all(v_max > 0.0) or not np.all(a_max > 0.0):
        raise ValueError(f"v_max and a_max must be strictly positive, got {v_max}, {a_max}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    way = _collapse(path)
    # a point, or a path of repeated points
    if len(way) < 2:
        return np.zeros(1), path[:1].copy()

    qs = [way[0]]
    for a, b in zip(way[:-1], way[1:]):
        delta = b - a
        dur, sv, sa = _segment(delta, v_max, a_max)
        n = max(1, int(np.ceil(dur / dt - 1e-12)))     # whole steps, rounded UP: never faster
        tau = np.linspace(0.0, dur, n + 1)[1:]         # the segment's own clock, exact at both ends
        seg = a + _s_of(tau, dur, sv, sa)[:, None] * delta
        # exact, not a + 1.0 * (b - a)
        seg[-1] = b
        qs.append(seg)
    q = np.vstack([np.atleast_2d(qs[0])] + list(qs[1:]))
    return np.arange(len(q), dtype=float) * dt, q
