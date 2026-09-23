import inspect
import math
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morph.arm.model import ArmModel  # noqa: E402
from morph.arm.cartesian import plan_cartesian                         # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# `load` is the path constructor; `ArmModel.__init__` takes already-parsed dicts.
M = ArmModel.load(os.path.join(_ROOT, "usd", "_arm_model.json"),
                  os.path.join(_ROOT, "usd", "_closure_lut.json"))
LINK = "Gripper_Link3_1"
START = M.Q8_HOME.copy()
# The operational caps every other planner tightens to, imported so this test moves with config.
# `cartesian.py` itself must NOT import config -- it runs in the OMPL planner subprocess.
from morph.config import A1_MAX, COLUMN_MAX                        # noqa: E402

DT = 1.0 / 240.0
V_MAX = np.array([0.111803, 0.363376, 0.363376, 0.122868,
                  0.583874, 0.583874, 0.583874, 0.583874])
A_MAX = 0.01 * np.array([142.3380, 110.2118, 5616.2782, 32.0210,
                         623.8286, 1065.4637, 1065.7098, 1083.7652])
DH_V_MAX = 0.0155


def _hand(q):
    return M.fk(q)[LINK][0]


def _base(yaw, t=(1.0, 2.0, 0.0)):
    """(translation3, rotation3x3) of the base link in world -- `ArmModel.collides`' contract."""
    c, s = math.cos(yaw), math.sin(yaw)
    return (np.array(t, float), np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))


def _hand_world(q, base):
    return base[0] + base[1] @ _hand(q)


def _tight_bounds():
    lo, hi = M.bounds()
    hi = hi.copy()
    hi[1] = min(hi[1], COLUMN_MAX)
    hi[2] = min(hi[2], COLUMN_MAX)
    hi[3] = min(hi[3], A1_MAX)
    return lo, hi


def test_straight_up_is_collinear():
    """Every waypoint's hand position must lie on the requested line, not merely end on it."""
    wps, err = plan_cartesian(M, START, delta=np.array([0.0, 0.0, 0.10]), link=LINK, step=0.005)
    assert err is None, err
    assert len(wps) >= 3, len(wps)
    p0, p1 = _hand(wps[0]), _hand(wps[-1])
    assert np.linalg.norm(p1 - p0 - np.array([0.0, 0.0, 0.10])) < 2e-3, (p0, p1)
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    for q in wps:
        d = _hand(q) - p0
        perp = d - u * float(d @ u)
        assert np.linalg.norm(perp) < 2e-3, np.linalg.norm(perp)


def _assert_monotonic(delta):
    """Progress along the REQUESTED direction must never go backwards."""
    wps, err = plan_cartesian(M, START, delta=np.asarray(delta, float), link=LINK, step=0.005)
    assert err is None, err
    p0, p1 = _hand(wps[0]), _hand(wps[-1])
    u = (p1 - p0) / np.linalg.norm(p1 - p0)      # derived, NOT hardcoded: a fixed +Z axis is
    s = [float((_hand(q) - p0) @ u) for q in wps]  # zero on a +X move and negative on a -Z one
    assert all(b >= a - 1e-6 for a, b in zip(s, s[1:])), s


def test_monotonic_progress():
    _assert_monotonic([0.0, 0.0, 0.10])


def test_monotonic_progress_downward():
    _assert_monotonic([0.0, 0.0, -0.10])


def test_monotonic_progress_sideways():
    _assert_monotonic([0.10, 0.0, 0.0])


def test_blocking_box_fails_with_a_reason():
    mid = _hand(START) + np.array([0.0, 0.0, 0.05])
    box = (np.array([mid - 0.03, mid + 0.03]), "world")
    wps, err = plan_cartesian(M, START, delta=np.array([0.0, 0.0, 0.10]), link=LINK,
                              obstacles=[box], step=0.005)
    assert wps is None and err is not None, (wps, err)
    assert "blocked" in err, err


def test_unreachable_delta_fails_at_ik_not_silently():
    wps, err = plan_cartesian(M, START, delta=np.array([5.0, 0.0, 0.0]), link=LINK, step=0.01)
    assert wps is None and "ik" in err.lower(), err


def test_delta_is_world_frame_when_base_is_given():
    """`base` names the world frame. `fk` seeds its root with (0, I), so an unconverted delta is
    consumed in the BASE frame -- 141 mm off at 90 deg base yaw."""
    base = _base(math.pi / 2)
    wps, err = plan_cartesian(M, START, delta=np.array([0.10, 0.0, 0.0]), link=LINK,
                              base=base, step=0.005)
    assert err is None, err
    moved = _hand_world(wps[-1], base) - _hand_world(wps[0], base)
    assert np.linalg.norm(moved - np.array([0.10, 0.0, 0.0])) < 2e-3, moved


def test_target_is_world_frame_when_base_is_given():
    base = _base(math.pi / 2)
    goal = _hand_world(START, base) + np.array([0.0, 0.0, 0.08])
    wps, err = plan_cartesian(M, START, target=goal, link=LINK, base=base, step=0.005)
    assert err is None, err
    assert np.linalg.norm(_hand_world(wps[-1], base) - goal) < 2e-3, _hand_world(wps[-1], base)


def test_delta_is_base_frame_when_base_is_absent():
    """With no `base` there is no world frame to speak of: the line is the model's own frame."""
    wps, err = plan_cartesian(M, START, delta=np.array([0.0, 0.0, 0.10]), link=LINK, step=0.005)
    assert err is None, err
    moved = _hand(wps[-1]) - _hand(wps[0])
    assert np.linalg.norm(moved - np.array([0.0, 0.0, 0.10])) < 2e-3, moved


def _slab(span=0.40):
    """A thin plate across the middle of a `span` vertical move, narrow enough that only the hand
    can reach it."""
    mid = _hand(START) + np.array([0.0, 0.0, span * 0.55])
    return (np.array([[mid[0] - 0.10, mid[1] - 0.10, mid[2] - 0.005],
                      [mid[0] + 0.10, mid[1] + 0.10, mid[2] + 0.005]]), "world")


def test_coarse_step_does_not_jump_the_obstacle():
    d = np.array([0.0, 0.0, 0.40])
    fine, err = plan_cartesian(M, START, delta=d, link=LINK, obstacles=[_slab()], step=0.005)
    assert fine is None and "blocked" in err, (fine, err)      # the truth this move must report
    for step in (0.40, 0.80, 0.10):
        wps, err = plan_cartesian(M, START, delta=d, link=LINK, obstacles=[_slab()], step=step)
        assert wps is None and err and "blocked" in err, (step, wps, err)


def test_non_positive_step_is_rejected_with_a_reason():
    for bad in (0.0, -0.005, float("nan")):
        wps, err = plan_cartesian(M, START, delta=np.array([0.0, 0.0, 0.10]), link=LINK, step=bad)
        assert wps is None and err and err.startswith("bad-args"), (bad, wps, err)


def test_start_pose_collision_is_reported():
    """Both a normal move and the span~0 path, where waypoints[0] is the only pose there is."""
    p0 = _hand(START)
    box = (np.array([p0 - 0.05, p0 + 0.05]), "world")
    wps, err = plan_cartesian(M, START, delta=np.array([0.0, 0.0, 0.10]), link=LINK,
                              obstacles=[box], step=0.005)
    assert wps is None and "blocked" in err, (wps, err)
    wps, err = plan_cartesian(M, START, delta=np.zeros(3), link=LINK, obstacles=[box], step=0.005)
    assert wps is None and "blocked" in err, (wps, err)


def test_bounds_argument_tightens_the_waypoints():
    """`model.bounds()` is the raw USD limits; OMPL plans against the operational caps, so a
    waypoint at a1 0.51 is a "success" the executor's own checkpoint would call invalid. 0.26 m
    overshoots A1_MAX by 10 mm while staying 15 mm inside the passive's -0.525 stop."""
    delta = np.array([0.26, 0.0, 0.0])
    raw, err = plan_cartesian(M, START, delta=delta, link=LINK, step=0.01)
    assert err is None and max(float(q[3]) for q in raw) > A1_MAX, "test no longer bites"
    lo, hi = _tight_bounds()
    wps, err = plan_cartesian(M, START, delta=delta, link=LINK, step=0.01, bounds=(lo, hi))
    if wps is None:
        assert err.startswith(("ik", "blocked")), err       # declining loudly is a valid outcome
    else:
        for q in wps[1:]:
            assert np.all(q <= hi + 1e-9) and np.all(q >= lo - 1e-9), q
    # the same bounds on a move that fits inside them still succeed
    wps, err = plan_cartesian(M, START, delta=np.array([0.20, 0.0, 0.0]), link=LINK, step=0.01,
                              bounds=(lo, hi))
    assert err is None, err
    for q in wps[1:]:
        assert np.all(q <= hi + 1e-9) and np.all(q >= lo - 1e-9), q


def test_bad_input_returns_a_reason_not_an_exception():
    """Track C branches on the reason text; a raise crashes it instead."""
    p0 = _hand(START)
    cases = {
        "neither": dict(),
        "both": dict(delta=np.zeros(3), target=p0),
        "nan delta": dict(delta=np.array([np.nan, 0.0, 0.0])),
        "inf target": dict(target=np.array([np.inf, 0.0, 0.0])),
        "short delta": dict(delta=np.array([0.0, 0.1])),
        "unknown link": dict(delta=np.array([0.0, 0.0, 0.1]), link="No_Such_Link_1"),
        "held tuple": dict(delta=np.array([0.0, 0.0, 0.1]),
                           held=(np.array([p0 - 0.03, p0 + 0.03]), "held")),
        "held ragged": dict(delta=np.array([0.0, 0.0, 0.1]), held=np.array([1.0, 2.0, 3.0])),
        "bad base": dict(delta=np.array([0.0, 0.0, 0.1]), base=(np.zeros(3),)),
        "bad obstacle": dict(delta=np.array([0.0, 0.0, 0.1]), obstacles=[np.zeros((2, 3))]),
        "nan margin": dict(delta=np.array([0.0, 0.0, 0.1]), margin=float("nan")),
        "negative margin": dict(delta=np.array([0.0, 0.0, 0.1]), margin=-0.02),
        "short q_start": dict(delta=np.array([0.0, 0.0, 0.1])),
    }
    for name, kw in cases.items():
        q = np.zeros(3) if name == "short q_start" else START
        kw.setdefault("link", LINK)
        try:
            wps, err = plan_cartesian(M, q, step=0.01, **kw)
        except Exception as e:                              # noqa: BLE001 -- that IS the defect
            raise AssertionError(f"{name} raised {type(e).__name__}: {e}")
        assert wps is None and err, (name, wps, err)
        assert err.startswith(("bad-args", "ik", "blocked")), (name, err)


def test_every_reason_starts_with_a_token_a_caller_can_branch_on():
    _wps, err = plan_cartesian(M, START, link=LINK)
    assert err.startswith("bad-args"), err


def test_link_defaults_to_the_model_hand_link():
    """The default must be `model.hand_link`, not the literal "Gripper_Link3_1". They coincide on
    this model, so only the signature proves it."""
    assert inspect.signature(plan_cartesian).parameters["link"].default is None
    a, err = plan_cartesian(M, START, delta=np.array([0.0, 0.0, 0.05]), step=0.01)
    assert err is None, err
    b, err = plan_cartesian(M, START, delta=np.array([0.0, 0.0, 0.05]), link=M.hand_link, step=0.01)
    assert err is None, err
    assert all(np.allclose(x, y) for x, y in zip(a, b))


def test_cartesian_samples_respect_actuated_velocity_limits():
    wps, err = plan_cartesian(M, START, delta=np.array([0.10, 0.0, 0.0]), step=0.01)
    assert err is None, err
    velocity = np.abs(np.diff(np.asarray(wps), axis=0)) / DT
    assert np.all(velocity <= V_MAX * (1.0 + 1e-9)), velocity.max(axis=0)


def test_cartesian_samples_respect_derated_actuated_acceleration_limits():
    wps, err = plan_cartesian(M, START, delta=np.array([0.10, 0.0, 0.0]), step=0.01)
    assert err is None, err
    velocity = np.diff(np.asarray(wps), axis=0) / DT
    acceleration = np.abs(np.diff(velocity, axis=0)) / DT
    assert np.all(acceleration <= A_MAX * (1.0 + 1e-9)), acceleration.max(axis=0)


def test_cartesian_samples_bound_the_closure_differential_rate():
    wps, err = plan_cartesian(M, START, delta=np.array([0.0, 0.0, 0.30]), step=0.02)
    assert err is None, err
    q = np.asarray(wps)
    dh_rate = np.abs(np.diff(q[:, 2] - q[:, 1])) / DT
    assert np.all(dh_rate <= DH_V_MAX * (1.0 + 1e-9)), dh_rate.max()


def test_a_start_inside_the_margin_refuses_at_the_first_waypoint():
    """A start clear at margin 0 but inside MARGIN was planned with a relaxed, growing margin
    (`a8b2d12`); that silent relaxation is deleted, so waypoint 1 is checked at the full margin
    and the line is refused there. The 14.9mm gap is what makes the row BIND: waypoint 1 rises
    5mm, leaving 19.9mm, so anything under the full 20mm clears it and a relaxation would pass."""
    p, R = M.fk(START)[LINK]
    link_box = M._xform_aabb(M.link_aabb[LINK], p, R)
    below = (np.array([link_box[0] - np.array([0.1, 0.1, 0.1]),
                       link_box[1] * np.array([1, 1, 0])
                       + np.array([0.1, 0.1, link_box[0][2] - 0.0149])]), "world")
    assert M.collides(START, [below], margin=0.02) and not M.collides(START, [below], margin=0.0)

    # Straight UP, away from `below`: under the relaxation this planned clean.
    wps, err = plan_cartesian(M, START, delta=np.array([0.0, 0.0, 0.10]), link=LINK,
                              obstacles=[below], step=0.005, margin=0.02)
    assert wps is None, wps
    assert err.startswith("blocked at 1/20"), err


def test_the_cartesian_tier_measures_whether_its_line_folds_the_arm():
    """`collides` tests the arm against OBSTACLES. `self_collides` was reachable from nowhere but
    the OMPL subprocess, so a straight line could drive the arm into a self-folded pose and the
    first thing to notice was the shadow report on the NEXT plan -- one motion too late. Measured
    here now, and not enforced: refusing would refuse retreats that ship today."""
    import contextlib as _ctx
    import io as _io
    from morph.arm.cartesian import _report_self_folds

    class _Folds:
        def __init__(self, at):
            self._at = at
        def self_collides(self, q8, margin=0.0):
            return ("Arm_Left_1", "Gripper_Link3_1") if self._at(q8) else None

    wps = [np.full(8, 0.1 * i) for i in range(5)]

    buf = _io.StringIO()
    with _ctx.redirect_stdout(buf):
        hits = _report_self_folds(_Folds(lambda q: True), wps)
    out = buf.getvalue()
    assert len(hits) == len(wps), f"only {len(hits)} of {len(wps)} folds were counted"
    assert "SELF-FOLD" in out and "the line STARTS folded" in out, (
        f"a line whose very first waypoint folds must say so -- it means the CALLER left the arm "
        f"folded, which is a different fix from a route the planner chose:\n{out}")

    buf2 = _io.StringIO()
    with _ctx.redirect_stdout(buf2):
        _report_self_folds(_Folds(lambda q: float(q[0]) > 0.25), wps)
    out2 = buf2.getvalue()
    assert "SELF-FOLD" in out2 and "STARTS folded" not in out2, (
        f"a fold beginning mid-line was reported as a folded start:\n{out2}")

    # A clear line must stay SILENT: a metric that prints on every motion is a metric nobody reads.
    buf3 = _io.StringIO()
    with _ctx.redirect_stdout(buf3):
        assert _report_self_folds(_Folds(lambda q: False), wps) == []
    assert buf3.getvalue() == "", f"a clear line printed anyway: {buf3.getvalue()!r}"

    # ...and a model that cannot answer must not kill the plan, but it must SAY SO: returning clear
    # silently makes a checker that raises indistinguishable from a path that never folds.
    class _Broken:
        def self_collides(self, q8, margin=0.0):
            raise RuntimeError("no geometry")
    buf4 = _io.StringIO()
    with _ctx.redirect_stdout(buf4):
        assert _report_self_folds(_Broken(), wps) == []
    assert "FAILED" in buf4.getvalue() and "UNMEASURED" in buf4.getvalue(), (
        f"a metric that cannot evaluate returned 'clear' in silence:\n{buf4.getvalue()!r}")


def test_the_self_fold_metric_gives_a_real_verdict_from_the_real_model():
    """Every other test here drives fakes that ignore the vector entirely, so a metric that throws
    on the REAL model -- wrong shape, wrong signature, anything -- passes them all. Pin one known
    folded pose against the shipped ArmModel.

    The pose is the boom through the wrist, not a box corner -- `tests/test_narrow_phase.py` pins
    that the wrist predicates agree with the boxes on this same q8. The live retreat waypoint this
    used to pin (wz -1.025, wy 0.801, wx 0.735) is 7 mm of air around the cylinder, and clear."""
    from morph.arm.cartesian import _report_self_folds
    folded = np.array([-1.3559, 0.9274, 0.9956, 0.183, -1.5653, 2.9748, -0.6333, -0.5844])
    truth = M.self_collides(folded)
    assert truth is not None, "fixture: this pose no longer self-collides -- pick another"
    hits = _report_self_folds(M, [folded, folded])
    assert len(hits) == 2 and hits[0][1] == truth, (
        f"the metric reported {hits!r} where the model itself says {truth!r}: it is not actually "
        f"evaluating the real geometry")


def test_the_self_fold_metric_is_wired_into_the_emitted_path():
    """Dead code is the failure mode for a metric: it can be correct and never called. This pins
    that `plan_cartesian` runs it on the waypoints it actually EMITS, after the retime."""
    import morph.arm.cartesian as _c

    seen = {}
    real = _c._report_self_folds

    def _spy(model, waypoints, diag_out=None):
        seen["n"] = len(waypoints)
        return real(model, waypoints, diag_out)

    _c._report_self_folds = _spy
    try:
        wps, err = plan_cartesian(M, START, delta=np.array([0.0, 0.0, 0.10]), link=LINK, step=0.005)
    finally:
        _c._report_self_folds = real
    assert err is None and wps, f"fixture: the straight line did not plan ({err})"
    assert seen.get("n") == len(wps), (
        f"the metric saw {seen.get('n')} waypoints but {len(wps)} were emitted -- it is either "
        f"not called at all, or called on a path that is not the one the executor writes")


def test_the_blocked_retreats_executed_prefix_is_measured_too():
    """When the line is blocked, `plan_cartesian` returns None -- but it hands back a PREFIX in
    `diag_out`, and `move_linear` EXECUTES that prefix. The metric ran only on
    the success path, so the motion most likely to leave the arm somewhere unshipped -- a retreat
    that could not finish -- was the one path never measured."""
    import morph.arm.cartesian as _c

    seen = []
    real = _c._report_self_folds
    _c._report_self_folds = lambda model, wps, diag=None: (seen.append(len(wps)),
                                                           real(model, wps, diag))[1]
    try:
        d = {}
        # Block with a real OBSTACLE, not an out-of-reach target: the prefix is only built on the COLLISION
        # branch, so an unreachable line exercises nothing.
        p0 = M.fk(START)[LINK][0]
        c = p0 + np.array([0.0, 0.0, 0.28])
        obs = [(np.array([c - 0.03, c + 0.03]), "world")]
        wps, err = plan_cartesian(M, START, delta=np.array([0.0, 0.0, 0.30]), link=LINK,
                                  step=0.005, obstacles=obs, diag_out=d)
    finally:
        _c._report_self_folds = real

    assert wps is None and err, f"fixture: expected a blocked line, got {len(wps or [])} waypoints"
    assert d.get("prefix"), "fixture: the blocked line built no prefix, so nothing is executed"
    assert seen, (
        "the line was blocked, a prefix was handed back for the caller to EXECUTE, and the "
        "self-fold metric never saw it")
    assert seen[-1] == len(d["prefix"]), (
        f"the metric saw {seen[-1]} waypoints but {len(d['prefix'])} are executed as the prefix")


def test_a_line_that_walks_a1_into_the_closure_stop_still_plans_CONTINUOUSLY():
    """The arm is 8-DOF against a 6-DOF task, so a hand pose near the closure stop is reachable by
    more than one configuration. The DLS iterates inside `[lo, hi]`, whose `a1` ceiling is a
    CONSTANT, so it walks a1 past the tilt-dependent stop, converges there and is rejected after the
    fact -- and `ik_restarts` then escapes to a distant configuration, moving a collision box
    hundreds of mm between two waypoints 5mm apart. Projecting each iterate onto the real ceiling
    lets the solver spend its redundancy staying holdable instead.

    Pins BOTH halves: the line plans, AND no adjacent pair moves a box further than that box may
    travel between two checks (`tunnel_limits`)."""
    import json as _json
    K = _json.load(open(os.path.join(_ROOT, "usd", "_grasp_known.json")))["arm_joints"]
    grasp = np.array([float(K[n]) for n in ArmModel.Q8], float)
    lo, hi = _tight_bounds()
    tun = M.tunnel_limits()

    wps, err = plan_cartesian(M, grasp, delta=np.array([0.15, 0.0, 0.0]), link=LINK,
                              step=0.005, bounds=(lo, hi), ik_restarts=0)
    assert err is None and wps, (
        f"the +x line from the shipped grasp pose still dies at the closure stop: {err}")
    worst, where = 0.0, -1
    for k in range(1, len(wps)):
        r = max(v / tun[n] for n, v in M.corner_shifts(np.asarray(wps[k - 1], float)[:8],
                                                       np.asarray(wps[k], float)[:8]).items()
                if tun.get(n))
        if r > worst:
            worst, where = r, k
    assert worst < 1.0, (
        f"waypoint {where} moves a collision box {worst:.2f}x its tunnel limit -- the line plans "
        f"by jumping through an interval nothing checks")


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
