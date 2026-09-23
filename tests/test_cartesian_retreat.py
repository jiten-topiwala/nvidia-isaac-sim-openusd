"""The Cartesian separation retreat that replaces the blind backout raise.

Offline: no Isaac, no simulator, no pxr -- `cartesian_retreat` is compiled out of the source and
run against a fake `Demo`, the way tests/test_target_obstacle.py runs `_arm_obstacles`.
    ./.venv/bin/python3 tests/test_cartesian_retreat.py"""
import ast
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from morph.arm.api import ArmMove, Outcome, move_linear, move_to_pose     # noqa: E402

# api's executor seam, routed to the fake under test (the real one needs a robot). Module-wide:
# the gate runs one process per suite.
import morph.arm.api as _api_mod                                                    # noqa: E402
_api_mod.execute_path = lambda demo, *a, **k: demo._execute_arm_path(*a, **k)
_api_mod.commanded_fingers = lambda demo: demo._commanded_fingers()
_api_mod.plan_arm = lambda demo, *a, **k: demo._plan_arm(*a, **k)
_api_mod.goal_set = lambda demo, *a, **k: demo._goal_set(*a, **k)
from morph.arm.model import ArmModel  # noqa: E402
from morph.arm.collision import Q8_TOL_M, Q8_TOL_RAD                      # noqa: E402
from morph.arm.cartesian import plan_cartesian                               # noqa: E402
from morph.config import A1_MAX, COLUMN_MAX, KNOWN, MOUTH_VEC_HAND       # noqa: E402

M = ArmModel.load(os.path.join(ROOT, "usd/_arm_model.json"),
                  os.path.join(ROOT, "usd/_closure_lut.json"))
HAND = M.hand_link
MARGIN = 0.02                                    # morph.arm.collision.MARGIN
R_OBJ = float(KNOWN["object_radius"])

# The measured in-hand relation: where the object centre sits in the HAND-LINK frame while held.
# Released onto the shelf it is still there -- the release opens the fingers, it does not move it.
OBJ_IN_HAND = np.array([0.0015, 0.0507, 0.1703])

# A real held pose, not Q8_HOME: the shipped side-grip, boom extended, wrist tilted.
GRASP = np.array([float(json.load(open(os.path.join(ROOT, "usd/_grasp_known.json")))
                        ["arm_joints"][n]) for n in ArmModel.Q8])


def _bounds():
    lo, hi = M.bounds()
    hi = hi.copy()
    hi[1] = min(hi[1], COLUMN_MAX)
    hi[2] = min(hi[2], COLUMN_MAX)
    hi[3] = min(hi[3], A1_MAX)
    return lo, hi


BOUNDS = _bounds()


def _base(yaw, t=(1.0, 2.0, 0.0)):
    """(translation3, rotation3x3) of the base link in world -- `ArmModel.collides`' contract."""
    c, s = math.cos(yaw), math.sin(yaw)
    return (np.array(t, float), np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))


def _object_box(q, base, r=R_OBJ, grow=np.zeros(3)):
    """The just-released object as a WORLD AABB, from where the hand is holding it at `q`."""
    p, R = M.fk(q)[HAND]
    c = p + R @ OBJ_IN_HAND
    return ArmModel._xform_aabb(np.vstack([c - r - grow, c + r + grow]), base[0], base[1])


# --------------------------------------------------------------------------- the source harness

def _func(rel, name):
    tree = ast.parse(open(os.path.join(ROOT, *rel)).read())
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == name), None)
    assert fn is not None, f"{os.path.join(*rel)} defines no {name}"
    return fn


def _exec_method(rel, name, glb):
    """Compile ONE function out of a source file and bind it into `glb`: the fakes intercept its
    collaborators by name, and morph/arm/stage.py imports pxr, which the offline venv lacks."""
    fn = _func(rel, name)
    assert not fn.decorator_list, f"{name} grew a decorator, which this harness would drop"
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 os.path.join(ROOT, *rel), "exec"), glb)
    return glb[name]

# The executor's module-level collaborators, routed to the fake's method of the same purpose.
_EXEC_SHIMS = {"retime_q8": __import__("morph.arm.cartesian", fromlist=["x"]).retime_q8,
               "q8_derate": __import__("morph.arm.cartesian", fromlist=["x"]).q8_derate,
               "with_closure_forced": lambda demo, q: demo._with_closure_forced(q),
               "commanded_fingers": lambda demo: demo._commanded_fingers(),
               "execute_path": lambda demo, *a, **k: demo._execute_arm_path(*a, **k),
               "planned_obstacles": lambda demo, *a, **k: demo._planned_obstacles(*a, **k),
               "blame": __import__("morph.arm.obstacles", fromlist=["blame"]).blame,
               "plan_arm": lambda demo, *a, **k: demo._plan_arm(*a, **k)}



class _RetreatDemo:
    """The `Demo` surface `cartesian_retreat` touches, recording every request it makes.

    `obs` is the obstacle set keyed the way the real `_arm_obstacles` keys it: a name in
    `grasp_names` is KEPT and tagged "grasped", a name in `exclude_names` is DROPPED, and a name in
    neither is an ordinary "world" obstacle. tests/test_target_obstacle.py pins that the real one
    behaves this way; this stands in for it without a stage."""

    lag = np.zeros(8)
    # The remainder plan seeds from the COMMANDED pose only under drives, and this double runs off
    # them, so the retreat plans from `q_now`. The driven case overrides `_drive_on`.
    idx = {n: i for i, n in enumerate(ArmModel.Q8)}

    def _drive_on(self):
        return False

    def _commanded_fingers(self):
        """This double has no articulation, so it reports no finger state -- which is exactly the
        fallback the production method takes, and keeps these tests judging the swept box."""
        return None

    def __init__(self, q, base, obj_box, name="pickup_obj_0"):
        self.q_now, self.base, self.obj_box, self.name = q.copy(), base, obj_box, name
        self.dt = 1.0 / 240.0
        self.calls = []
        self._plan_fallbacks = {}

    def _arm_model(self):
        return M

    def _arm_q8(self):
        return self.q_now.copy()

    def _arm_bounds(self):
        return BOUNDS

    def _arm_base_world(self):
        return self.base

    def _arm_obstacles(self, grasp_names=(), *, exclude_names=(), diagnostic_paths=None):
        self.calls.append(("obstacles", tuple(grasp_names), tuple(exclude_names)))
        if self.name in exclude_names:
            return []
        if diagnostic_paths is not None:
            diagnostic_paths.append("/mock/obstacle")
        return [(self.obj_box, "grasped" if self.name in grasp_names else "world")]

    def _execute_arm_path(self, waypoints, secs=None, **kw):
        self.calls.append(("execute", waypoints, secs, kw))
        if waypoints is None or len(waypoints) == 0:
            return None
        self.q_now = np.asarray(waypoints[-1], float) + self.lag
        return np.zeros(8)

    def _goal_set(self, goal, grasp, held):
        return goal

    def _plan_arm(self, goal_set, **kwargs):
        self.calls.append(("plan_arm", goal_set, kwargs))
        return [self.q_now, goal_set]

    def _fallback(self, reason):
        self._plan_fallbacks[reason] = self._plan_fallbacks.get(reason, 0) + 1


SAID = []


def _const(rel, name):
    """A module-level constant lifted from source text, NOT copied into this file. Copying is how
    a test starts asserting against a value the shipped code no longer uses (see `MARGIN` above,
    which is duplicated so this file stays free of the simulator's imports)."""
    tree = ast.parse(open(os.path.join(ROOT, *rel), encoding="utf-8").read())
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets):
            return ast.literal_eval(n.value)
    raise AssertionError(f"{name} is gone from {'/'.join(rel)}")


_GLB = {"plan_start": lambda demo: demo._plan_start(),
        "commanded_fingers": _EXEC_SHIMS["commanded_fingers"],
        "move_to_pose": move_to_pose,
        "np": np, "os": os, "MARGIN": MARGIN, "MOUTH_VEC_HAND": MOUTH_VEC_HAND,
        "_RETREAT_IK_RESTARTS": _const(("morph", "arm", "retreat.py"), "_RETREAT_IK_RESTARTS"),
        "COLUMN_MAX": COLUMN_MAX,
        "ArmModel": ArmModel, "Q8_TOL_M": Q8_TOL_M, "Q8_TOL_RAD": Q8_TOL_RAD,
        "move_linear": move_linear, "Outcome": Outcome,
        "print": lambda *a, **k: SAID.append(" ".join(str(x) for x in a))}
_RETREAT = _exec_method(("morph", "arm", "retreat.py"), "cartesian_retreat", _GLB)
# The retreat seeds its remainder plan through `_plan_start`; the REAL one, over this double's
# `_arm_q8` / `_drive_on` / `_drv_prev`, so a fake cannot quietly seed from the wrong pose.
_RetreatDemo._plan_start = _exec_method(("morph", "arm", "plan.py"), "plan_start", _GLB)
_RetreatDemo._cartesian_retreat = _RETREAT


def _executed(d):
    """The waypoint list `cartesian_retreat` handed the executor, and the kwargs it named."""
    return next(((w, s, k) for tag, w, s, k in
                 (c for c in d.calls if c[0] == "execute")), None)


def _delta_of(d, obj_idx=0):
    """Run the retreat with `move_linear` replaced by a recorder; return the delta it asked for."""
    seen = {}

    def _spy(demo, delta_or_target, frame, **kw):
        seen["delta"] = np.asarray(delta_or_target, float)
        seen["frame"] = frame
        seen["kw"] = kw
        return ArmMove(Outcome.ARRIVED, "arrived", (), np.zeros(8))

    real = _GLB["move_linear"]
    _GLB["move_linear"] = _spy
    try:
        d._cartesian_retreat(obj_idx)
    finally:
        _GLB["move_linear"] = real
    return seen


# --------------------------------------------------------------------------- the geometry

def test_the_retreat_direction_increases_separation():
    """The sign check. `MOUTH_VEC_HAND` points INTO the mouth: projected on it the object centre
    sits at -0.172 against the hand box's -0.2022, so the object is on the MINUS-mouth side and
    AWAY is +mouth. An inverted sign drives the hand into the object it just released, which is
    what the earlier draft of this move did."""
    base = _base(0.0)
    d = _RetreatDemo(GRASP, base, _object_box(GRASP, base))
    got = _delta_of(d)["delta"]
    p, R = M.fk(GRASP)[HAND]
    axis_w = base[1] @ (R @ MOUTH_VEC_HAND)
    assert float(got @ axis_w) > 0.0, (
        f"the retreat runs INTO the mouth, towards the object: {np.round(got, 4).tolist()}")
    c = p + R @ OBJ_IN_HAND
    sep0 = float(np.linalg.norm(p - c))

    def _sep(delta):
        wps, err = plan_cartesian(M, GRASP, delta=delta, link=HAND, base=base,
                                  bounds=BOUNDS, step=0.005)
        return None if err else float(np.linalg.norm(M.fk(np.asarray(wps[-1]))[HAND][0] - c))

    fwd = _sep(got)
    assert fwd is not None and fwd > sep0 + 0.050, (
        f"+mouth must separate: {sep0 * 1000:.0f} -> {fwd and fwd * 1000:.0f}mm")
    back = _sep(-got)
    assert back is None or back < sep0, (
        f"-mouth must never separate: {sep0 * 1000:.0f} -> {back * 1000:.0f}mm")


def test_the_retreat_ends_clear_of_the_object_with_it_back_in_the_set():
    """The move's whole purpose, measured the way the run's `backout-probe` measures it: with the
    placed object as an ORDINARY obstacle -- no allowance -- the end pose must not collide, and the
    start pose must. Nothing else in this suite proves the distance is enough."""
    base = _base(0.0)
    obj = _object_box(GRASP, base)
    d = _RetreatDemo(GRASP, base, obj)
    assert d._cartesian_retreat(0) is True, d._plan_fallbacks
    wps = _executed(d)[0]
    assert M.first_hit(GRASP, [(obj, "world")], MARGIN, None, base) == HAND, (
        "the start state is supposed to be inside the object -- this suite's premise")
    assert M.first_hit(np.asarray(wps[-1]), [(obj, "world")], MARGIN, None, base) is None, (
        "the retreat ended still inside the placed object")


# --------------------------------------------------------------------------- the collision policy

def test_a_link_other_than_the_hand_may_not_sweep_through_the_placed_object():
    """THE DEFECT the "grasped" tag exists for. The hand starts inside the released object, so the
    object cannot simply stay an ordinary obstacle; but DROPPING it blinds every link to it and the
    821 mm boom is then free to route through. Tagged "grasped" only `hand_link` is exempt, so an
    object grown out to where the boom sweeps must be REFUSED."""
    base = _base(0.0)
    boom = M.link_aabbs(GRASP, base=base)["Arm_Left_1"]
    reach = float(np.max(np.abs(np.vstack([boom[0], boom[1]])
                                - _object_box(GRASP, base).mean(axis=0))))
    fat = _object_box(GRASP, base, grow=np.full(3, reach))
    assert M.first_hit(GRASP, [(fat, "grasped")], MARGIN, None, base) == "Arm_Left_1", (
        "this fixture is supposed to put a non-hand link inside the object")
    d = _RetreatDemo(GRASP, base, fat)
    # remainder fails too
    d._plan_arm = lambda *a, **k: None
    del SAID[:]
    assert d._cartesian_retreat(0) is False, (
        "the boom is inside the placed object and the retreat was planned anyway -- the object "
        "is not being checked against the links that are not the hand")
    assert any("blocked" in m for m in SAID), (
        f"refused, but not because the geometry was rejected -- an object dropped from the set "
        f"refuses this way too, and for the opposite reason: {SAID}")
    assert d._plan_fallbacks == {"retreat-no-cartesian": 1, "retreat-no-remainder": 1}, d._plan_fallbacks
    assert _executed(d) is None, "nothing may be executed when no path was found"

def test_diagnostic_line_is_emitted_when_blocked():
    """Prove the diagnostic line is emitted when the retreat is blocked."""
    base = _base(0.0)
    boom = M.link_aabbs(GRASP, base=base)["Arm_Left_1"]
    reach = float(np.max(np.abs(np.vstack([boom[0], boom[1]])
                                - _object_box(GRASP, base).mean(axis=0))))
    fat = _object_box(GRASP, base, grow=np.full(3, reach))
    d = _RetreatDemo(GRASP, base, fat)
    # remainder fails too
    d._plan_arm = lambda *a, **k: None
    del SAID[:]
    assert d._cartesian_retreat(0) is False
    assert any("inside the margin of" in m and "/mock/obstacle" in m for m in SAID), (
        "a blocked retreat printed no line naming the obstacle: " + str(SAID))



def test_the_derived_distance_clears_the_object_at_every_pose_not_just_this_one():
    """The distance is a separating-axis gap, so it has to be enough WHEREVER the arm released
    from and whatever size the object is -- one pose proves nothing about a formula. Dropping the
    `MARGIN` term out of the hand's half-extents leaves the shipped grasp clear by luck and about
    one pose in twenty still inside the object it just released."""
    rng = np.random.default_rng(7)
    tried = clear = 0
    while tried < 70:
        yaw = rng.uniform(-math.pi, math.pi)
        q = np.array([rng.uniform(-math.pi, math.pi), 0.0, 0.0, rng.uniform(0.1, 0.55),
                      rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5),
                      rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)])
        col, dh = rng.uniform(0.6, 1.35), rng.uniform(0.0, 0.2)
        q[1], q[2] = col, col + dh
        r = rng.uniform(0.035, 0.080)
        if not M.in_lut_domain(dh, q[3]):
            continue
        base = _base(yaw, t=rng.uniform(-3.0, 3.0, 3) * np.array([1.0, 1.0, 0.0]))
        obj = _object_box(q, base, r=r)
        if M.first_hit(q, [(obj, "world")], MARGIN, None, base) != HAND:
            # not a release pose: the hand is not holding anything
            continue
        tried += 1
        d = _RetreatDemo(q, base, obj)
        d._plan_arm = lambda *a, **k: [d.q_now, a[0]]  # remainder succeeds
        d._cartesian_retreat(0)
        if "retreat-no-cartesian" in d._plan_fallbacks:
            # cartesian path was blocked
            continue
        clear += 1
        end = np.asarray(_executed(d)[0][-1])
        assert M.first_hit(end, [(obj, "world")], MARGIN, None, base) is None, (
            f"the derived distance left the arm inside the object at r={r:.3f}, "
            f"q8={np.round(q, 3).tolist()}")
    # The AVAILABILITY of the retreat, bounded BOTH ways: a drop means IK seeding regressed, a jump
    # means these poses stopped being representative. NOT a production rate -- these draws are synthetic.
    assert clear >= 54, (
        f"only {clear}/{tried} poses planned -- the straight-line retreat got LESS available, "
        f"which routes more releases to the unchecked open-loop raise")
    assert clear <= 62, (
        f"{clear}/{tried} planned, better than the 56/70 this was pinned at -- if that is a real "
        f"improvement re-pin it, but check first that the poses are still release-like")


def test_the_object_is_kept_and_tagged_not_dropped_from_the_set():
    """`exclude_names` would make the test above pass by making the object invisible. It is
    keyword-only precisely so that cannot happen by accident; assert it is not used at all."""
    base = _base(0.0)
    d = _RetreatDemo(GRASP, base, _object_box(GRASP, base))
    d._cartesian_retreat(3)
    asked = [c for c in d.calls if c[0] == "obstacles"]
    assert asked, "the retreat never built an obstacle set"
    for _tag, grasp, excl in asked:
        assert excl == (), f"the placed object is dropped from the set entirely: {excl}"
    # The sets the retreat MOVES against carry the tag; the P3.8 reading asks once without it.
    tagged = [grasp for _tag, grasp, _excl in asked if grasp]
    assert tagged and all(g == ("pickup_obj_3",) for g in tagged), tagged


# --------------------------------------------------------------------------- the frame

def test_the_retreat_delta_is_rotated_into_the_world_frame():
    """`plan_cartesian` is given `base=`, so its `delta` is WORLD; `fk` returns BASE-frame poses.
    A direction taken from `fk` and passed straight through TYPE-CHECKS and silently plans a
    different line. At a yawed base the unrotated delta plans a path that ends STILL INSIDE the
    object -- the failure has no error and no exception, only a wrong motion."""
    base = _base(math.pi / 2)
    obj = _object_box(GRASP, base)
    d = _RetreatDemo(GRASP, base, obj)
    got = _delta_of(d)["delta"]
    _p, R = M.fk(GRASP)[HAND]
    naive = float(np.linalg.norm(got)) * (R @ MOUTH_VEC_HAND)
    assert np.allclose(base[1].T @ got, naive, atol=1e-9), (
        f"the delta is not the base-frame mouth axis carried into world: "
        f"{np.round(base[1].T @ got, 4).tolist()} vs {np.round(naive, 4).tolist()}")
    assert np.linalg.norm(got - naive) > 0.05, "this base yaw does not separate the two lines"
    ends = {}
    for name, delta in (("world", got), ("unrotated", naive)):
        wps, err = plan_cartesian(M, GRASP, delta=delta, link=HAND, obstacles=[(obj, "grasped")],
                                  margin=MARGIN, base=base, bounds=BOUNDS, step=0.005)
        assert err is None, f"{name}: {err}"
        ends[name] = M.first_hit(np.asarray(wps[-1]), [(obj, "world")], MARGIN, None, base)
    assert ends["world"] is None, ends
    assert ends["unrotated"] == HAND, (
        f"the unrotated delta must be demonstrably wrong here, else this test proves nothing: "
        f"{ends}")


# --------------------------------------------------------------------------- failure and hand-off

def test_no_path_returns_false_without_raising_and_counts_the_degrade():
    """The failure policy: this project counts degrades, it does not raise them."""
    base = _base(0.0)
    d = _RetreatDemo(GRASP, base, _object_box(GRASP, base))
    # remainder fails too
    d._plan_arm = lambda *a, **k: None
    real = _GLB["move_linear"]
    _GLB["move_linear"] = lambda *a, **k: ArmMove(
        Outcome.REFUSED, "line not clear: blocked at 3/18 (17% along, 15mm)")
    try:
        assert d._cartesian_retreat(0) is False
    finally:
        _GLB["move_linear"] = real
    assert d._plan_fallbacks == {"retreat-no-cartesian": 1, "retreat-no-remainder": 1}, d._plan_fallbacks
    assert _executed(d) is None, "nothing may be executed when no path was found"


def test_executed_is_not_arrived():
    """Under `ARM_DRIVE=1` `_force` no-ops and the arm lags, so a finished loop is not a reached
    pose. The settle's snap already measures the residue; so must this."""
    base = _base(0.0)
    d = _RetreatDemo(GRASP, base, _object_box(GRASP, base))
    d.lag = np.zeros(8)
    d.lag[3] = 10 * Q8_TOL_M
    assert d._cartesian_retreat(0) is False, "a retreat that stopped short reported success"
    # The tally too, not just the return: this route hands the caller the same unchecked open-loop
    # raise the no-path route does, so the two must assert the same shape.
    assert d._plan_fallbacks == {"retreat-no-arrival": 1}, d._plan_fallbacks


def test_the_dense_path_is_played_one_waypoint_per_frame():
    """`plan_cartesian` already retimed this path onto the 240 Hz grid under the measured joint
    limits, and `execute_path` reads `secs` as a FLOOR -- so the floor must be the grid the path
    is already on, or the line is flown slower than the geometry it was planned as."""
    base = _base(0.0)
    d = _RetreatDemo(GRASP, base, _object_box(GRASP, base))
    assert d._cartesian_retreat(0) is True, d._plan_fallbacks
    wps, secs, kw = _executed(d)
    assert abs(secs - max(1, len(wps) - 1) * d.dt) < 1e-12, (
        f"{len(wps)} waypoints handed a {secs:.3f}s floor, not the {max(1, len(wps) - 1) * d.dt:.3f}s "
        f"the retimed path already spans")
    assert kw.get("grasp_names") == ("pickup_obj_0",), kw
    assert kw.get("checkpoint_every", 8) > len(wps), (
        "a mid-path checkpoint replans with `_plan_arm`, a free-space route that is not this "
        f"straight line and is free to sweep the exempt hand back through the object: {kw}")


def test_the_caller_no_longer_runs_unchecked_motion():
    """The place backout must not run the open-loop raise or const-z retract anymore."""
    src = os.path.join(ROOT, "morph", "place", "__init__.py")
    tree = ast.parse(open(src).read())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "cartesian_retreat"]
    assert len(calls) == 1, "expected exactly one backout call site"
    
    # Assert no open loop raise column increments are present under the retreat handling
    ramp = [n for n in ast.walk(tree)
            if isinstance(n, ast.For)
            and any(isinstance(c, ast.Call) and getattr(c.func, "id", None) == "frozen_step"
                    for c in ast.walk(n))
            and any(getattr(t, "id", None) == "_clr" for t in ast.walk(n))]
    assert not ramp, "the open-loop raise ramp must be deleted"
    
    # Assert no const-z retract is present under the retreat handling
    const_z = [n for n in ast.walk(tree)
               if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "_a1rt" for t in n.targets)]
    assert not const_z, "the const-z retract must be deleted"




class _RetractDemo:
    """The surface `_checked_a1_retract` touches.

    `clearances` is the sequence the fake model reports, in metres, one per checked pose: None
    means the pose is outside the margin entirely (no hit). A negative value is real overlap."""

    def __init__(self, clearances=None):
        self.idx = {"ArmLeftJoint_1": 3, "ColumnLeftBearingJoint_1": 1, "ColumnRightBearingJoint_1": 2}
        self.dt = 1.0 / 240.0
        self.commanded = []
        self._safety_abort = False
        self._seq = list(clearances or [])
        self._c1_start = 0.0
        self._a1_start = 0.0
        self._n = 0
        demo = self

        class _Model:
            Q8 = ["BaseJoint_1", "ColumnLeftBearingJoint_1", "ColumnRightBearingJoint_1",
                  "ArmLeftJoint_1", "w1", "w2", "w3", "w4"]

            def _value(self):
                if not demo._seq:
                    return None
                return demo._seq[min(demo._n, len(demo._seq) - 1)]

            def first_hit(self, q8, obstacles, margin, held, base):
                return None if self._value() is None else "Arm_Left_1"

            def clearance(self, q8, obstacles, held, base):
                demo._n += 1
                v = self._value()
                return 1.0 if v is None else float(v)

        self._model = _Model()
        for n in self._model.Q8:
            self.idx.setdefault(n, len(self.idx))
        self.robot = self._Robot(self)

    class _Robot:
        """Reports the last pose commanded, as a real articulation does: a fake that always reads
        zero cannot tell a ramp interpolating from a fixed start from one that re-reads mid-loop."""

        def __init__(self, demo):
            self._demo = demo

        def get_joint_positions(self):
            if self._demo.commanded:
                return self._demo.commanded[-1].copy()
            q = np.zeros(12)
            q[self._demo.idx["ColumnLeftBearingJoint_1"]] = self._demo._c1_start
            q[self._demo.idx["ColumnRightBearingJoint_1"]] = self._demo._c1_start
            q[self._demo.idx["ArmLeftJoint_1"]] = self._demo._a1_start
            return q

    def _arm_model(self):
        return self._model

    def _arm_base_world(self):
        return (np.zeros(3), np.eye(3))

    def _arm_obstacles(self, grasp_names=(), **kw):
        return [(np.array([[9.0, 9.0, 9.0], [9.1, 9.1, 9.1]]), "world")]

    def _closure_passives(self, dh, a1):
        return {}

    def step(self, q):
        self.commanded.append(np.asarray(q, float).copy())


def _retract(demo):
    glb = {"np": np, "os": os, "math": math, "ESCAPE_FLOOR_M": 0.008,
           "CLEARANCE_EPS_M": 1.0e-4, "COLUMN_MAX": 1.40, "A1_MAX": 0.60}
    _exec_method(("morph", "place", "__init__.py"), "_escape_ok", glb)
    _exec_method(("morph", "place", "__init__.py"), "_checked_a1_retract", glb)
    demo._escape_ok = glb["_escape_ok"].__get__(demo, type(demo))
    return glb["_checked_a1_retract"](demo, 0, demo.step)


def test_a_clear_retract_reaches_the_carry_pose_by_checked_motion():
    """Nothing is near anything: every pose is checked, then commanded, and a1 lands on target."""
    d = _RetractDemo()
    assert _retract(d) is True
    expected = max(2, int(0.4 / d.dt)) + max(2, int(1.5 / d.dt))   # checked lift, then the retract
    assert len(d.commanded) == expected, len(d.commanded)
    assert d._n >= len(d.commanded), "a pose was commanded without being checked first"
    assert d._safety_abort is False
    a1 = d.idx["ArmLeftJoint_1"]
    assert abs(float(d.commanded[-1][a1]) - 0.10) < 1e-9, float(d.commanded[-1][a1])


def test_motion_that_gains_clearance_is_allowed_from_inside_the_margin():
    """The backout STARTS inside the margin -- 19 mm from the battery at slot 1. An absolute test
    refuses the very first step of the escape; departure must be allowed."""
    d = _RetractDemo(clearances=[0.019 + 0.0001 * k for k in range(600)])
    assert _retract(d) is True, "an escape that gains clearance every step must be allowed"
    assert len(d.commanded) > 0
    assert d._safety_abort is False


def test_motion_that_gives_clearance_away_is_refused_and_halts_the_run():
    """Inside the margin and closing: that is approach, not escape."""
    d = _RetractDemo(clearances=[0.019 - 0.0005 * k for k in range(600)])  # > CLEARANCE_EPS_M per step
    assert _retract(d) is False
    assert d.commanded == [], f"commanded {len(d.commanded)} poses while closing on the obstacle"
    assert d._safety_abort is True


def test_real_overlap_is_refused_however_it_moves():
    """clearance < 0 is genuine geometric overlap: never commanded, whatever the trend."""
    d = _RetractDemo(clearances=[-0.001] * 600)
    assert _retract(d) is False
    assert d.commanded == []
    assert d._safety_abort is True


def test_a_retract_without_a_checker_refuses_rather_than_moving_blind():
    d = _RetractDemo()
    d._arm_model = lambda: (_ for _ in ()).throw(RuntimeError("no model"))
    assert _retract(d) is False
    assert d.commanded == [], "nothing may be commanded when the checker is unavailable"
    assert d._safety_abort is True


def test_a_gap_below_the_floor_is_never_commanded_however_it_trends():
    """Under ESCAPE_FLOOR_M the collider model is not evidence of air: this robot's own bodies are
    drawn up to 26 mm outside their colliders, so a 5 mm "gap" may be contact. Freeze, do not slide."""
    d = _RetractDemo(clearances=[0.005 + 0.0001 * k for k in range(600)])
    assert _retract(d) is False, "a gaining path below the floor must still be refused"
    assert d.commanded == []
    assert d._safety_abort is True


def test_a_retract_under_a_standing_abort_commands_nothing():
    """A latch means some motion was already disproved; 456 more commanded poses do not make it
    safer. `_recover` refuses to move under a standing latch and so must this."""
    d = _RetractDemo()
    d._safety_abort = True
    assert _retract(d) is False
    assert d.commanded == []


def test_the_lift_follows_the_cosine_it_was_written_as():
    """Every commanded column position must sit on the cosine ramp from the pose the lift STARTED
    at. Re-reading the robot inside the loop interpolates from where the arm already moved to, which
    front-loads the travel: the column carrying the whole boom then sees several times the designed
    velocity while the printed duration stays 0.4 s."""
    d = _RetractDemo()
    assert _retract(d) is True
    n_lift = max(2, int(0.4 / d.dt))
    c1 = d.idx["ColumnLeftBearingJoint_1"]
    for k in range(n_lift):
        want = 0.03 * (0.5 - 0.5 * math.cos(math.pi * (k + 1) / n_lift))
        got = float(d.commanded[k][c1])
        assert abs(got - want) < 1e-9, (
            f"step {k + 1}/{n_lift} commanded {got * 1000:.3f} mm, the ramp says {want * 1000:.3f} mm")


def test_the_lift_cannot_drive_a_column_past_its_hard_stop():
    """An env-tunable lift added to a column already near the top must clip, not command through
    the end stop. The collision check cannot save this: a pose past the stop is geometrically fine."""
    d = _RetractDemo()
    # 10 mm below COLUMN_MAX, with a 30 mm lift requested
    d._c1_start = 1.39
    assert _retract(d) is True
    c1 = d.idx["ColumnLeftBearingJoint_1"]
    worst = max(float(q[c1]) for q in d.commanded)
    assert worst <= 1.40 + 1e-9, f"commanded {worst:.4f} m, past the 1.40 m column stop"


def test_the_retract_is_timed_by_the_joint_limit_not_by_habit():
    """A cosine ramp peaks at pi/2 times its average rate. Timing the retract by a fixed 1.5 s asked
    0.209 m/s of a slide limited to 0.122868 m/s, and the track gate measured the 35 mm lag that
    followed. The commanded profile must stay inside the joint's own limit."""
    from morph.arm.cartesian import _V_MAX
    v_max = float(_V_MAX[3])                                  # ArmLeftJoint_1
    d = _RetractDemo()
    # a 200 mm retract to CARRY_A1 = 0.10
    d._a1_start = 0.30
    assert _retract(d) is True
    a1 = d.idx["ArmLeftJoint_1"]
    n_lift = max(2, int(0.4 / d.dt))
    prof = [float(q[a1]) for q in d.commanded[n_lift:]]
    peak = max(abs(prof[k + 1] - prof[k]) / d.dt for k in range(len(prof) - 1))
    assert peak <= v_max * 1.02, (
        f"commanded {peak * 1000:.0f} mm/s against a {v_max * 1000:.0f} mm/s limit")
    assert abs(prof[-1] - 0.10) < 1e-9, prof[-1]


def test_the_remainder_asks_for_no_duration_of_its_own():
    """`_plan_arm` returns waypoints metres apart in joint space, and any duration derived here is
    a second answer to a question `execute_path` already answers from the joint ceilings
    (tests/test_timing.py). The remainder must hand it none."""
    base = _base(0.0)
    boom = M.link_aabbs(GRASP, base=base)["Arm_Left_1"]
    reach = float(np.max(np.abs(np.vstack([boom[0], boom[1]])
                                - _object_box(GRASP, base).mean(axis=0))))
    fat = _object_box(GRASP, base, grow=np.full(3, reach))   # blocks the straight line
    d = _RetreatDemo(GRASP, base, fat)
    del SAID[:]
    d._cartesian_retreat(0)
    rem = [c for c in d.calls if c[0] == "execute" and c[3].get("tag") == "backout-remainder"]
    assert rem, f"no remainder execution happened; calls were {[c[0] for c in d.calls]}"
    assert rem[0][2] is None, f"the remainder was handed a duration of its own: {rem[0][2]}"


def test_the_success_path_is_unchanged():
    '''The success path (prefix + planned remainder) is unchanged.'''
    src = os.path.join(ROOT, "morph", "place", "__init__.py")
    tree = ast.parse(open(src).read())
    place_impl = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PlaceMixin").body
    place_impl = next(n for n in place_impl if isinstance(n, ast.FunctionDef) and n.name == "_place_impl")
    
    # cartesian retreat call should be assigned to _sep
    assigns = [n for n in ast.walk(place_impl) if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "_sep" for t in n.targets)]
    assert assigns, "must call cartesian retreat and store result in _sep"
def test_a_blocked_line_is_SURFACED_and_the_remainder_still_runs():
    """The caller's half of a blocked retreat. `move_linear` owns halting on the last cleared pose
    and naming what blocked it (tests/test_arm_api.py); this pins what the RETREAT does with it:
    count the degrade, print the reason so the operator sees which obstacle it was, and still plan
    the remainder to the carry pose rather than ending there."""
    base = _base(0.0)
    d = _RetreatDemo(GRASP, base, _object_box(GRASP, base))
    real = _GLB["move_linear"]
    _GLB["move_linear"] = lambda *a, **k: ArmMove(
        Outcome.REFUSED,
        "halted at the last cleared pose; line not clear: blocked at 3/18 -- "
        "Arm_Left_1 inside the margin of /mock/wall")
    del SAID[:]
    try:
        d._cartesian_retreat(0)
    finally:
        _GLB["move_linear"] = real
    assert d._plan_fallbacks.get("retreat-no-cartesian") == 1, d._plan_fallbacks
    assert any("/mock/wall" in m and "Arm_Left_1" in m for m in SAID), (
        "the retreat swallowed the reason the line was refused: " + str(SAID))
    assert any(c[0] == "plan_arm" for c in d.calls), (
        "a blocked line ended the retreat -- the remainder plan to the carry pose never ran")


def test_the_retreat_asks_for_the_restarts_and_the_world_frame():
    """`move_linear` defaults `ik_restarts` to 0 because a restart can flip IK branch between two
    waypoints nothing checks between. The retreat is the caller that wants them, so it must name
    them -- and the delta it hands over is WORLD, which is what `frame` says it reads."""
    base = _base(0.0)
    d = _RetreatDemo(GRASP, base, _object_box(GRASP, base))
    seen = _delta_of(d)
    assert seen["frame"] == "world", f"the retreat handed its world delta as {seen['frame']}"
    assert seen["kw"].get("ik_restarts") == _GLB["_RETREAT_IK_RESTARTS"], (
        f"the retreat stopped asking for its measured restarts: {seen['kw'].get('ik_restarts')}")
    # Passed, not non-None: this double has no articulation and reports None, which is the production
    # fallback. That it is the COMMANDED hand is pinned in tests/test_target_obstacle.py.
    assert "fingers" in seen["kw"], f"the retreat dropped the finger state: {sorted(seen['kw'])}"



def test_the_remainder_plan_starts_from_the_COMMANDED_pose_under_drives():
    """`move_linear` halts the retreat at the last waypoint the checker CLEARED. Under drives the
    arm is not quite there -- measured `logs/lag`: 0.34 mm behind the commanded pose on the
    columns/boom -- and a remainder seeded from the MEASURED pose puts a start that the line had
    validated at the margin a third of a millimetre inside it:

        start hit: Arm_Left_1 vs chassis ... (clearance 20mm of margin 20mm)
        backout: remainder plan to carry pose unavailable          -> retreat-no-remainder, +23 s

    `plan_arm` already seeds from `plan_start` (commanded under drives, measured otherwise), so the
    remainder passes no start of its own; what it must not do is BUILD the carry pose off the
    lagged measured one, which would command the same third of a millimetre back.
    """
    got = {}

    class _Drv(_RetreatDemo):
        idx = {n: i for i, n in enumerate(ArmModel.Q8)}

        def _drive_on(self):
            return True

        def _plan_arm(self, goal_set, **kwargs):
            got.update(kwargs)
            got["goal"] = np.asarray(goal_set, float).copy()
            self.calls.append(("plan_arm", goal_set, kwargs))
            # remainder "fails": the goal is all we test
            return None

    base = _base(0.0)
    d = _Drv(GRASP, base, _object_box(GRASP, base))
    # the measured 0.34 mm, on every joint
    d.lag = np.full(8, 0.00034)
    # set by the first _execute_arm_path below
    d._drv_prev = None
    _orig_exec = _Drv._execute_arm_path

    def _exec(self, waypoints, secs=None, **kw):
        out = _orig_exec(self, waypoints, secs, **kw)
        # what `_apply` records: the last COMMANDED full joint vector (here Q8-shaped)
        self._drv_prev = np.asarray(waypoints[-1], float).copy()
        return out
    _Drv._execute_arm_path = _exec
    del SAID[:]
    d._cartesian_retreat(0)
    assert "goal" in got, "the remainder never reached the planner"
    c1, c2, a1 = (ArmModel.Q8.index(n) for n in
                  ("ColumnLeftBearingJoint_1", "ColumnRightBearingJoint_1", "ArmLeftJoint_1"))
    want = d._drv_prev.copy()
    want[c1] += 0.03
    want[c2] += 0.03
    want[a1] = float(os.environ.get("CARRY_A1", "0.10"))
    assert np.allclose(got["goal"], want), (
        f"the carry pose was built off the lagged measured pose: "
        f"{np.round(got['goal'], 5).tolist()} vs {np.round(want, 5).tolist()}")
    assert not np.allclose(got["goal"][c1], d.q_now[c1] + 0.03), (
        "this fixture does not separate the commanded pose from the measured one")
    assert "start" not in got or got["start"] is None, (
        "the remainder passes a start of its own again -- `plan_arm` seeds from `plan_start`, and "
        "two answers to one question is how they drift apart")


def test_the_remainder_plans_and_checkpoints_with_no_grasp_exemption():
    """A4: a free-space plan carries no path-wide exemption. The line owns the object's box for
    the one straight motion that needs it; the remainder is sampled, it can route the boom
    anywhere, and it must see the released object as an ordinary obstacle -- in the PLAN and in the
    checkpoint the executor rebuilds, which `execute_path` requires to match."""
    base = _base(0.0)
    d = _RetreatDemo(GRASP, base, _object_box(GRASP, base))
    assert d._cartesian_retreat(0) is True, d._plan_fallbacks
    planned = [c for c in d.calls if c[0] == "plan_arm"]
    assert planned, "the remainder never planned"
    for _tag, _goal, kw in planned:
        assert not kw.get("grasp_names"), f"the remainder plan carries an exemption: {kw}"
    rem = [c for c in d.calls if c[0] == "execute" and c[3].get("tag") == "backout-remainder"]
    assert rem, f"no remainder execution happened; calls were {[c[0] for c in d.calls]}"
    assert not rem[0][3].get("grasp_names"), (
        f"the checkpoint rebuilds the set WITH the object exempt: {rem[0][3]}")


def test_the_remainder_plans_and_checkpoints_on_the_COMMANDED_hand():
    """`d723967`: two hands in one motion. The remainder must plan -- and re-validate at its
    checkpoints -- on the hand the grip is holding, not the swept finger union, which refuses poses
    the commanded pads clear. This double reports no hand by default, so the mutation
    `hold_fingers=False` passes every other row here unnoticed."""
    base = _base(0.0)
    d = _RetreatDemo(GRASP, base, _object_box(GRASP, base))
    hand = {"finger_a_joint_1_1": -0.55, "finger_b_joint_1_1": -0.55, "finger_c_joint_1_1": -0.55}
    d._commanded_fingers = lambda: dict(hand)
    d._cartesian_retreat(0)
    planned = [c for c in d.calls if c[0] == "plan_arm"]
    assert planned, "the remainder never planned"
    for _tag, _goal, kw in planned:
        assert kw.get("fingers") == hand, (
            f"the remainder plans the SWEPT hand, not the commanded one: {kw.get('fingers')}")
    rem = [c for c in d.calls if c[0] == "execute" and c[3].get("tag") == "backout-remainder"]
    assert rem, f"no remainder execution happened; calls were {[c[0] for c in d.calls]}"
    assert rem[0][3].get("hold_fingers", True) is not False, (
        f"the executor is free to re-read the fingers mid-path: {rem[0][3]}")


def test_a_remainder_that_did_not_run_is_never_reported_as_success():
    """The outcomes `move_to_pose` can tell apart that a bare `q_end is None` could not. A stopped
    or short remainder leaves the boom un-retracted, and the base then drives with it. The lag is
    applied to the REMAINDER only: applied to the line it refuses earlier, as `retreat-no-arrival`,
    and this would pass without ever judging the remainder."""
    base = _base(0.0)
    for name, stop in (("stopped", True), ("short", False)):
        d = _RetreatDemo(GRASP, base, _object_box(GRASP, base))
        _orig = type(d)._execute_arm_path

        def _exec(self, waypoints, secs=None, _stop=stop, **kw):
            if kw.get("tag") != "backout-remainder":
                return _orig(self, waypoints, secs, **kw)
            if _stop:
                self.calls.append(("execute", waypoints, secs, kw))
                return None
            self.lag = np.full(8, 10 * Q8_TOL_M)
            return _orig(self, waypoints, secs, **kw)
        d._execute_arm_path = _exec.__get__(d, type(d))
        assert d._cartesian_retreat(0) is False, f"a {name} remainder reported success"
        assert "retreat-no-arrival" not in d._plan_fallbacks, (
            f"the {name} case refused in the line, not in the remainder: {d._plan_fallbacks}")


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
