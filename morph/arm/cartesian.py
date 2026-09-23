"""Straight-line Cartesian path planning: move the hand along a given line, or say why not.
Pure numpy -- no Isaac imports, so it runs in the venv and in the OMPL planner subprocess."""
import numpy as np

from morph.arm.timing import free_secs, retime

# Waypoint spacing ceiling, m: nothing is checked BETWEEN waypoints; 0.02 bounds the flown bulge.
MAX_STEP = 0.02

# Q8 order, from the drive ceilings pinned in tests/test_timing.py. Acceleration is derated to 1%:
# those are isolated-drive values and this coupled arm spends the rest on coupled and load error.
_DT = 1.0 / 240.0
_V_MAX = np.array([0.111803, 0.363376, 0.363376, 0.122868,
                   0.583874, 0.583874, 0.583874, 0.583874])
_A_MAX = 0.01 * np.array([142.3380, 110.2118, 5616.2782, 32.0210,
                          623.8286, 1065.4637, 1065.7098, 1083.7652])
# Public through api as V_MAX: read-only, so no caller can retune the ceilings in place.
_V_MAX.setflags(write=False)

# Column-differential speed cap: keeps `RotationLeftJoint_1`'s brake under its 300 Nm at the
# (dh, a1) = (0, 0) corner (299.8 Nm); `tests/test_timing.py` recomputes it, FINDINGS derives it.
_DH_V_MAX = 0.0137
_DH_A_MAX = _A_MAX[1] + _A_MAX[2]


def _augment(path):
    """(path with the column differential as a ninth coordinate, its velocity ceilings, its
    acceleration ceilings). `_DH_V_MAX` bounds the DIFFERENTIAL, which no per-joint row does."""
    p = np.atleast_2d(np.asarray(path, float))
    return (np.column_stack((p, p[:, 2] - p[:, 1])),
            np.append(_V_MAX, _DH_V_MAX), np.append(_A_MAX, _DH_A_MAX))


def retime_q8(path, dt, derate=1.0):
    """Time-parameterise a Q8 path on the `dt` grid under the measured ceilings. `derate` below 1
    scales them (velocity by k, acceleration by k squared), which stretches the duration by
    exactly 1/k. Returns (t, q8), q8[0] and q8[-1] the endpoints bit-for-bit."""
    aug, v, a = _augment(path)
    k = float(derate)
    t, timed = retime(aug, v * k, a * k * k, dt)
    return t, timed[:, :8]


def q8_derate(path, secs):
    """The factor that makes `secs` a FLOOR for `retime_q8`. Sized off the UNROUNDED duration:
    `retime` only ever rounds a segment UP, so the flown time cannot then land under the ask."""
    if not secs:
        return 1.0
    aug, v, a = _augment(path)
    free = free_secs(aug, v, a)
    return min(1.0, free / float(secs)) if free > 0.0 else 1.0


def _arr(x, shape, what):
    """`x` as a finite float array of exactly `shape`, or ValueError naming what was wrong."""
    a = np.asarray(x, float)
    if a.shape != shape:
        raise ValueError(f"{what} must have shape {shape}, got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{what} must be finite, got {a.tolist()}")
    return a


def plan_cartesian(model, q_start, delta=None, target=None, link=None,
                   obstacles=(), margin=0.02, step=0.005, held=None, base=None,
                   bounds=None, keep_R=True, ik_restarts=0, diag_out=None,
                   fingers=None):
    """Interpolate `link` along a straight line and IK every waypoint. `base` = (t3, R3x3) of the
    base in world; given, `delta`/`target` and the check are WORLD, else all base-frame.
    Returns (waypoints, None), waypoints[0] == q_start, or (None, reason: bad-args|ik|blocked)."""
    try:
        q0 = _arr(q_start, (8,), "q_start").copy()
        if (delta is None) == (target is None):
            raise ValueError("pass exactly one of delta= or target=")
        line = _arr(delta, (3,), "delta") if target is None else _arr(target, (3,), "target")
        step = float(step)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError(f"step must be a positive distance in metres, got {step!r}")
        margin = float(margin)
        # a NaN margin makes every AABB comparison False: it silently disables the collision check
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError(f"margin must be a non-negative distance in metres, got {margin!r}")
        # (box, tag) or (box, tag, pad_m, links), the latter a FLOOR: those links are checked against THAT
        # box at `pad_m`, never skipped. Positional, so a wrong length is refused, not reshaped.
        obs2 = []
        for ob in obstacles:
            if len(ob) not in (2, 4):
                raise ValueError(f"obstacle must be (box, tag) or (box, tag, pad_m, links), "
                                 f"got {len(ob)} elements")
            obs2.append((_arr(ob[0], (2, 3), "obstacle"), ob[1]) if len(ob) == 2
                        else (_arr(ob[0], (2, 3), "obstacle"), ob[1], float(ob[2]), tuple(ob[3])))
        obstacles = obs2
        held = None if held is None else _arr(held, (2, 3), "held")
        bt, bR = (None, None) if base is None else (_arr(base[0], (3,), "base[0]"),
                                                    _arr(base[1], (3, 3), "base[1]"))
        lo, hi = model.bounds() if bounds is None else (_arr(bounds[0], (8,), "bounds[0]"),
                                                        _arr(bounds[1], (8,), "bounds[1]"))
        link = model.hand_link if link is None else link
        poses = model.fk(q0)
        if link not in poses:
            raise ValueError(f"unknown link {link!r}")
    except (TypeError, ValueError, IndexError, KeyError) as e:
        return None, f"bad-args: {e}"

    p0, R0 = poses[link]
    if target is None:
        goal = p0 + (line if bR is None else bR.T @ line)
    else:
        goal = line if bR is None else bR.T @ (line - bt)

    # `collides` is monotone in the margin, so a pose in real overlap is the only way to fail
    # at 0 -- testing `margin` first would only repeat this answer.
    if model.collides(q0, obstacles, margin=0.0, held=held, base=base, fingers=fingers):
        if diag_out is not None:
            diag_out.update({"q_blocked": q0, "where": "q_start"})
        return None, "blocked at q_start"

    span = float(np.linalg.norm(goal - p0))
    # A zero-length line checks no waypoint.
    if span < 1e-9:
        return [q0], None
    n = max(1, int(np.ceil(span / min(step, MAX_STEP))))

    out = [q0]
    q_prev = q0
    for k in range(1, n + 1):
        dist = span * (k / n)
        p_k = p0 + (goal - p0) * (k / n)
        sol = model.ik(p_k, R=R0 if keep_R else None, seed=q_prev, link=link,
                       bounds=(lo, hi), restarts=ik_restarts)
        where = f"{k}/{n} ({100.0 * k / n:.0f}% along, {dist * 1000:.0f}mm)"
        if sol is None:
            return None, f"ik failed at {where}"
        q_k = sol[0]
        if model.collides(q_k, obstacles, margin=margin, held=held, base=base, fingers=fingers):
            if diag_out is not None:
                diag_out.update({"q_blocked": q_k, "where": where})
                diag_out["prefix"] = (list(retime_q8(out, _DT)[1]) if len(out) > 1 else [])
            # `move_linear` EXECUTES this prefix, so measure it too.
            if diag_out is not None and diag_out.get("prefix"):
                _report_self_folds(model, diag_out["prefix"], None)
            return None, f"blocked at {where}"
        out.append(q_k)
        q_prev = q_k
    emitted = list(retime_q8(out, _DT)[1])
    _report_self_folds(model, emitted, diag_out)
    return emitted, None


def _report_self_folds(model, waypoints, diag_out=None):
    """Shadow metric: does this straight line fold the arm into itself?

    `collides` tests against OBSTACLES only. Measured, not enforced -- refusing here would refuse
    retreats that ship today. Returns the (index, pair) list.
    """
    hits = []
    try:
        for k, w in enumerate(waypoints):
            pair = model.self_collides(np.asarray(w, float)[:8])
            if pair is not None:
                hits.append((k, pair))
    except Exception as e:                         # noqa: BLE001 -- a metric must not kill a plan
        # ...but not QUIETLY: returning "clear" on failure is indistinguishable from clear.
        print(f">>> cartesian: SELF-FOLD metric FAILED ({type(e).__name__}: {e}) -- this path went "
              f"UNMEASURED", flush=True)
        return []
    if hits:
        # The SAME path judged against the palm's audited colliders: the self-check's hand box is the union
        # before the finger sweep, ~4x its metal, so only real geometry can say whether the path is clear.
        try:
            from morph.arm.collision import PALM_AABB
            _keep = model.selfcol_aabb.get(model.hand_link)
            if _keep is not None:
                model.selfcol_aabb[model.hand_link] = PALM_AABB
                model._selfcol_pairs = None
                tight = sum(1 for w in waypoints
                            if model.self_collides(np.asarray(w, float)[:8]) is not None)
                model.selfcol_aabb[model.hand_link] = _keep
                model._selfcol_pairs = None
                print(f">>> cartesian: the same path on the AUDITED PALM box folds on {tight} of "
                      f"{len(waypoints)} waypoints", flush=True)
        except Exception as _e:                    # noqa: BLE001 -- a metric must not kill a plan
            print(f">>> cartesian: palm-box comparison unavailable ({_e})", flush=True)
    if hits and diag_out is not None:
        diag_out["self_folds"] = hits
    if hits:
        k0, pair0 = hits[0]
        # The POSE, not just the index: a folding waypoint is somewhere the bake never went, so it cannot
        # be reconstructed offline from the corpus.
        _q0 = np.asarray(waypoints[k0], float)[:8]
        print(f">>> cartesian: SELF-FOLD on {len(hits)} of {len(waypoints)} waypoints, first at "
              f"{k0} ({pair0[0]} x {pair0[1]}) q8 {np.round(_q0, 4).tolist()}"
              f"{'  <-- the line STARTS folded' if k0 == 0 else ''}"
              f"  (shadow only -- nothing is refused)", flush=True)
    return hits
