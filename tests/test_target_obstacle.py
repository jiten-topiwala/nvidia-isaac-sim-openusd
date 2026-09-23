"""The target object is an OBSTACLE, not a hole in the obstacle set.

`_arm_obstacles(exclude_names=...)` drops every collider under a named prim. The pick used to pass
the target object there, so for the whole planned approach the object was not an obstacle at all
and the 821 mm boom could route straight through it. Offline: no Isaac, no simulator, no pxr --
`_arm_obstacles` is compiled out of the source and run against a fake stage.
    ./.venv/bin/python3 tests/test_target_obstacle.py"""
import ast
import json
import math
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from morph.arm.model import ArmModel  # noqa: E402
from morph.arm.cartesian import plan_cartesian                                # noqa: E402

VENV = os.path.join(ROOT, ".venv/bin/python3")

M = ArmModel.load(os.path.join(ROOT, "usd/_arm_model.json"),
                  os.path.join(ROOT, "usd/_closure_lut.json"))
from morph.arm.collision import MARGIN  # noqa: E402
GRASP = np.array([float(json.load(open(os.path.join(ROOT, "usd/_grasp_known.json")))
                        ["arm_joints"][n]) for n in ArmModel.Q8])
# Mid-approach: the boom extended, the hand out in front -- `tests/test_armfk.py`'s REACH.
APPROACH = np.array([0.0, 0.45, 0.55, 0.40, 0.0, 0.0, 0.0, 0.0])
PARK = np.array([0.0, 1.2, 1.2, 0.1, 0.0, 0.0, 0.0, 0.0])
_BOOM_C = M.link_aabbs(APPROACH)["Arm_Left_1"].mean(axis=0)
# A 110 x 110 x 150 mm object sitting exactly where the boom sweeps.
ON_THE_BOOM = np.array([_BOOM_C - [0.055, 0.055, 0.075], _BOOM_C + [0.055, 0.055, 0.075]])


# --------------------------------------------------------------------------- the source harness

def _func(rel, name):
    tree = ast.parse(open(os.path.join(ROOT, *rel)).read())
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def _param_names(fn):
    a = fn.args
    return {p.arg for p in a.posonlyargs + a.args + a.kwonlyargs}


def _default(fn, name):
    """The literal default of `fn`'s keyword parameter `name`, by AST."""
    a = fn.args
    pairs = list(zip(a.args[len(a.args) - len(a.defaults):], a.defaults))
    pairs += [(k, d) for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
    return next(ast.literal_eval(d) for k, d in pairs if k.arg == name)


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


def _bind_executor(glb):
    """The real `execute_path` compiled into `glb`; its collaborators, the recursion included, are
    routed to the fake under test."""
    g = {**_EXEC_SHIMS, **glb}
    fn = _exec_method(("morph", "arm", "execute.py"), "execute_path", g)
    g["execute_path"] = _EXEC_SHIMS["execute_path"]
    return fn


# --------------------------------------------------------------------------- the fake USD stage

class _Rng:
    def __init__(self, box):
        self.b = np.asarray(box, float)

    def GetMin(self):
        return self.b[0]

    def GetMax(self):
        return self.b[1]

    def IsEmpty(self):
        return bool(np.any(self.b[1] < self.b[0]))

    def ComputeAlignedRange(self):
        return self


class _Mtx:
    """An identity transform that survives `a * b`, `.GetInverse()` and `m[i][j]`.

    Row indexing is not decoration: a real `Gf.Matrix4d` is subscriptable, and the obstacle builder
    reads the linear part to scale a sphere's authored radius into world units."""

    def __mul__(self, other):
        return self

    def GetInverse(self):
        return self

    def __getitem__(self, i):
        return [1.0 if i == j else 0.0 for j in range(4)]


class _Prim:
    def __init__(self, path, box, collider=True, enabled=True, radius=None):
        self._p, self.box, self.collider, self.enabled = path, box, collider, enabled
        # set -> this prim IS a UsdGeom.Sphere
        self.radius = radius

    def GetPath(self):
        return self._p

    def HasAPI(self, _api):
        return self.collider

    def IsA(self, schema):
        return schema is _FakeUsdGeom.Sphere and self.radius is not None


class _Stage:
    def __init__(self, prims):
        self.prims = prims

    def GetPseudoRoot(self):
        return self.prims

    def GetPrimAtPath(self, _p):
        return _Prim("/World/base", None)


class _FakeUsd:
    class TimeCode:
        @staticmethod
        def Default():
            return None

    @staticmethod
    def TraverseInstanceProxies():
        return None

    @staticmethod
    def PrimRange(root, _traverse):
        return list(root)


class _FakeUsdGeom:
    class BBoxCache:
        def __init__(self, *a, **k):
            pass

        @staticmethod
        def ComputeWorldBound(prim):
            class _BB:
                @staticmethod
                def GetRange():
                    return _Rng(prim.box)

                @staticmethod
                def GetMatrix():
                    return _Mtx()

                @staticmethod
                def ComputeAlignedRange():
                    return _Rng(prim.box)
            return _BB

    class Sphere:
        """A sphere prim reaches `_arm_obstacles` as its authored extent BOX carried through the
        transform, so on a rotated parent the bound is the AABB of a rotated cube, not the sphere.
        The builder rebuilds it as centre +- r; the double must model the API that lets it."""

        def __init__(self, prim):
            self._prim = prim

        def GetRadiusAttr(self):
            class _A:
                @staticmethod
                def Get():
                    return self._prim.radius
            return _A

    class Xformable:
        def __init__(self, _prim):
            pass

        @staticmethod
        def ComputeLocalToWorldTransform(_tc):
            return _Mtx()


class _FakeUsdPhysics:
    class CollisionAPI:
        def __init__(self, prim):
            self.prim = prim

        def GetCollisionEnabledAttr(self):
            class _A:
                @staticmethod
                def Get():
                    return self.prim.enabled
            return _A


class _FakeGf:
    @staticmethod
    def BBox3d(rng, _mtx):
        return rng


# The shipped stage in miniature: the arm (never an obstacle), a chassis collider, a shelf board,
# and the pick target sitting on the boom's path.
SHELF = np.array([[1.5, -0.5, 0.9], [2.5, 0.5, 0.91]])
BATTERY = np.array([[-0.31, -0.29, 0.165], [-0.03, 0.27, 0.305]])
PARKED_MAST = np.array([[0.40, -0.15, 0.00], [0.69, 0.14, 1.64]])   # the second arm's column
PARKED_TIP = np.array([[0.50, -0.02, 0.90], [0.54, 0.02, 0.94]])    # one of its fingertips
ROLLER_CENTRE = np.array([0.2225, 0.2045, 0.0454])
ROLLER_ROTATED_BOUND = np.vstack([ROLLER_CENTRE - [0.1385, 0.117, 0.1275],
                                  ROLLER_CENTRE + [0.1385, 0.117, 0.1275]])
# The deck plate, at its REAL path: `chassis_pad` matches this prim exactly and RAISES if it is
# absent, so any harness that plans or checkpoints must carry it. Far from every pose flown here.
_DECK = np.array([[40.0, 40.0, -1.0], [41.0, 41.0, -0.9]])
_PRIMS = [_Prim("/World/Arm_1/Arm_Left_1", np.zeros((2, 3))),
          _Prim("/World/Geometry/robot/base_footprint/base/Box", _DECK),
          _Prim("/World/base/battery", BATTERY),
          _Prim("/World/shelf/board", SHELF),
          _Prim("/World/pickup_obj_0", ON_THE_BOOM),
          _Prim("/World/decor", SHELF, collider=False),
          _Prim("/World/pickup_obj_1", SHELF, enabled=False),
          # A mecanum roller as it really arrives: r=80mm sphere whose bound is the AABB of a
          # ROTATED 160mm extent cube. 277 x 234 x 255mm is a live triple from logs/sphere/run3.log.
          _Prim("/World/base/wheel/roller_1/Sphere", ROLLER_ROTATED_BOUND, radius=0.08)]
_PATHS = {"Arm_1": "/World/Arm_1", "base": "/World/base",
          "pickup_obj_0": "/World/pickup_obj_0", "pickup_obj_1": "/World/pickup_obj_1"}


class _Demo:
    """The `Demo` surface `_arm_obstacles` touches, over the fake stage above."""

    def __init__(self):
        self.stage = _Stage(_PRIMS)
        self.__dict__["_armp"] = ("/World/Arm_1", "/World/base")


_OBS_GLB = {"np": np, "os": os, "MAX_SPAN": 50.0, "HULL_PARTS": frozenset(),
            "print": lambda *a, **k: None,
            "Usd": _FakeUsd, "UsdGeom": _FakeUsdGeom, "UsdPhysics": _FakeUsdPhysics,
            "Gf": _FakeGf, "find_path": lambda _s, n: _PATHS.get(n)}
_STAGE_SHIMS = {"arm_prim_paths": lambda demo: demo._arm_prim_paths(),
                "parked_arm_path": lambda demo: demo._parked_arm_path()}
_Demo._arm_obstacles = _exec_method(("morph", "arm", "stage.py"), "arm_obstacles", {**_OBS_GLB, **_STAGE_SHIMS})
_Demo._arm_prim_paths = lambda self: self._armp
_Demo._parked_arm_path = _exec_method(("morph", "arm", "stage.py"), "parked_arm_path", _OBS_GLB)


def _obstacles(*a, **kw):
    return _Demo()._arm_obstacles(*a, **kw)


def _same(a, b):
    """Two obstacle sets are the same set: same boxes, same tags, same order -- and the same pad,
    when an entry carries one (`(box, tag, pad_m, links)`)."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if len(x) != len(y) or x[1] != y[1] or not np.array_equal(x[0], y[0]):
            return False
        if len(x) == 4 and (abs(float(x[2]) - float(y[2])) > 1e-12 or tuple(x[3]) != tuple(y[3])):
            return False
    return True


# --------------------------------------------------------------------------- the approach policy

class _ApproachDemo:
    """`_plan_approach`'s `Demo` surface, recording every obstacle-set request it makes."""

    path = None

    def _commanded_fingers(self):
        return None

    def _arm_q8(self):
        # `move_to_pose` reads the measured pose to decide arrival; this fake's subject is the
        # OBSTACLE POLICY, so the arrival verdict is not load-bearing here.
        return np.zeros(8)

    def __init__(self):
        self.calls = []
        self._plan_fallbacks = {}

    def _arm_obstacles(self, *a, **k):
        self.calls.append((a, k))
        return []

    def _traj_q8(self, _tr, _i):
        return np.zeros(8)

    def _object_relative_goal(self, goal, _obj):
        return goal

    def _goal_set(self, goal, *a):
        return goal

    def _plan_arm(self, goal, **k):
        self.calls.append((("_plan_arm",), k))
        return self.path

    def _execute_arm_path(self, wps, secs=None, **k):
        self.calls.append((("_execute_arm_path",), k))
        return np.zeros(8)

    def _fallback(self, reason):
        self._plan_fallbacks[reason] = self._plan_fallbacks.get(reason, 0) + 1


from morph.arm.api import Outcome as _Outcome, move_to_pose as _move_to_pose  # noqa: E402

# api's executor seam, routed to the fake under test (the real one needs a robot). Module-wide:
# the gate runs one process per suite.
import morph.arm.api as _api_mod                                                    # noqa: E402
_api_mod.execute_path = lambda demo, *a, **k: demo._execute_arm_path(*a, **k)
_api_mod.commanded_fingers = lambda demo: demo._commanded_fingers()
_api_mod.plan_arm = lambda demo, *a, **k: demo._plan_arm(*a, **k)
_api_mod.goal_set = lambda demo, *a, **k: demo._goal_set(*a, **k)

_APPR_GLB = {"np": np, "os": os, "ARM_PLANNER": True, "print": lambda *a, **k: None,
             "move_to_pose": _move_to_pose, "Outcome": _Outcome}
_ApproachDemo._plan_approach = _exec_method(("morph", "pick", "approach.py"), "_plan_approach",
                                            _APPR_GLB)
# The REAL derivation, not a stub: `_plan_approach` delegates the goal to it, and a fake one would
# hand the planner a goal no production path produces -- which is the whole subject of this file.
_ApproachDemo._approach_goal = _exec_method(("morph", "pick", "approach.py"), "_approach_goal",
                                            _APPR_GLB)


def _approach_policy():
    """The obstacle policy `_plan_approach` hands the planner, as kwargs for `_arm_obstacles`."""
    d = _ApproachDemo()
    d._plan_approach({}, 0, 1.0, exclude_obj=0)
    plan = [k for tag, k in d.calls if tag and tag[0] == "_plan_arm"]
    assert plan, "the approach never reached the planner"
    return {k: v for k, v in plan[0].items() if k in ("exclude_names", "grasp_names")}


def test_the_boom_may_not_route_through_the_target_during_the_approach():
    """THE DEFECT. The obstacle set the approach plans against, built by the real `_arm_obstacles`
    over a stage where the target sits exactly on the boom's path: the boom must be rejected."""
    obs = _obstacles(**_approach_policy())
    assert M.first_hit(APPROACH, obs, MARGIN) == "Arm_Left_1", (
        "the 821 mm boom passes through the pick target during the planned approach: the target "
        f"is not in the obstacle set the approach plans against ({len(obs)} obstacles)")


def test_the_planner_and_the_checkpoint_share_one_obstacle_policy():
    """A path planned under one policy and re-validated under another is rejected at the first
    checkpoint. The approach names no policy to either, so both take the same default."""
    d = _ApproachDemo()
    d.path = [np.zeros(8)]
    assert d._plan_approach({}, 0, 1.0, exclude_obj=0) is True, d._plan_fallbacks
    named = [k for tag, k in d.calls if tag and tag[0] in ("_plan_arm", "_execute_arm_path")]
    assert len(named) == 2, named
    for k in named:
        assert not (set(k) & {"exclude_names", "grasp_names"}), (
            f"the approach names an obstacle policy the other consumer does not: {k}")
    got = {n: _default(_func(rel, n), "grasp_names")
           for rel, n in ((("morph", "arm", "plan.py"), "plan_arm"),
                          (("morph", "arm", "execute.py"), "execute_path"))}
    assert got == {"plan_arm": (), "execute_path": ()}, got


def test_goal_ik_repairs_against_the_same_obstacles_the_planner_plans_against():
    """A goal repaired against a set that has the target cut out of it can be a goal the planner
    then refuses -- and the approach's whole point is that the target is in the set."""
    got = {}
    m = SimpleNamespace(ik_repair=lambda *a, **k: got.update(k) or (np.zeros(8), 0.0, 0.0))
    d = _Demo()
    d._arm_base_world = lambda: (np.zeros(3), np.eye(3))
    _Demo._goal_ik = _exec_method(("morph", "arm", "plan.py"), "goal_ik",
                                  {"np": np, "MARGIN": MARGIN})
    d._goal_ik(m, np.zeros(3), np.eye(3), np.zeros(8), M.hand_link, M.bounds())
    assert _same(got["obstacles"], _obstacles(**_approach_policy())), (
        "_goal_ik repairs the goal against a different obstacle set than the planner uses")


def test_goal_set_filters_branches_against_the_same_obstacles():
    """`_goal_set` hands OMPL the branches it judged clear; judged against a set missing the
    target, it offers branches the planner's own `isValid` then throws away."""
    got = {}
    m = SimpleNamespace(fk=lambda q: {M.hand_link: (np.zeros(3), np.eye(3))},
                        hand_link=M.hand_link,
                        ik_candidates=lambda *a, **k: got.update(k) or [])
    d = _Demo()
    d._arm_model = lambda: m
    d._arm_bounds = lambda: M.bounds()
    d._arm_base_world = lambda: (np.zeros(3), np.eye(3))
    _Demo._goal_set = _exec_method(("morph", "arm", "plan.py"), "goal_set",
                                   {"np": np, "os": os, "MARGIN": MARGIN,
                                    "print": lambda *a, **k: None})
    os.environ["OBJ_GOAL_SET"] = "1"
    try:
        d._goal_set(np.zeros(8), (), None)
    finally:
        os.environ.pop("OBJ_GOAL_SET")
    assert _same(got["obstacles"], _obstacles(**_approach_policy())), (
        "_goal_set filters IK branches against a different obstacle set than the planner uses")


def test_the_target_survives_the_json_boundary_into_the_planner_subprocess():
    """The planner runs in another interpreter over JSON. A tag that does not survive that trip
    leaves the child planning against a set the parent never meant to send."""
    req = {"start": APPROACH.tolist(), "goal": APPROACH.tolist(),
           "model": os.path.join(ROOT, "usd/_arm_model.json"),
           "lut": os.path.join(ROOT, "usd/_closure_lut.json"),
           "obstacles": [[ab[0].tolist(), ab[1].tolist(), tg]
                         for ab, tg in _obstacles(**_approach_policy())],
           "held": None, "base": None, "margin": MARGIN, "solve_time": 1.0, "seed": 1}
    out = subprocess.run([VENV, os.path.join(ROOT, "arm_plan.py")], input=json.dumps(req),
                         capture_output=True, text=True, timeout=120)
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert isinstance(got, dict) and "invalid" in got.get("error", ""), (
        f"the child planned a path through the target: {str(got)[:200]}")


def test_the_planner_subprocess_refuses_a_malformed_obstacle_rather_than_dropping_it():
    """`arm_plan.py` reads `o[0], o[1], o[2]` POSITIONALLY, so an obstacle carrying a fourth element
    -- the shape a per-obstacle allowance takes -- parses cleanly with the fourth silently gone, and
    the child then plans without a permission the parent granted, or with one it did not. A planner
    that quietly drops part of its request is the failure this suite exists to prevent: refuse.

    Universal over LENGTH, not a single bad case: every entry that is not exactly (lo, hi, tag) must
    be rejected, because the drop is silent in both directions."""
    ok = [[ab[0].tolist(), ab[1].tolist(), tg] for ab, tg in _obstacles(**_approach_policy())]
    for bad, why in ((ok[0] + [["Arm_Left_1"]], "a fourth element is dropped"),
                     (ok[0][:2], "a missing tag makes every box a world box"),
                     (ok[0][:1], "a missing corner cannot be detected positionally")):
        req = {"start": APPROACH.tolist(), "goal": APPROACH.tolist(),
               "model": os.path.join(ROOT, "usd/_arm_model.json"),
               "lut": os.path.join(ROOT, "usd/_closure_lut.json"),
               "obstacles": [bad], "held": None, "base": None,
               "margin": MARGIN, "solve_time": 1.0, "seed": 1}
        out = subprocess.run([VENV, os.path.join(ROOT, "arm_plan.py")], input=json.dumps(req),
                             capture_output=True, text=True, timeout=120)
        line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "{}"
        got = json.loads(line)
        assert isinstance(got, dict) and "obstacle" in got.get("error", ""), (
            f"the child accepted a malformed obstacle ({why}): {str(got)[:200]}")


# `_plan_to_config` and its `SETTLE_PLAN` gate are DELETED: they carried an exemption into the
# planned tier, which A4 forbids, and never ran on a shipped cycle. P3.2 rebuilds this.


class _ExecDemo(_Demo):
    """`execute_path`'s `Demo` surface, over the same fake stage, recording every obstacle
    set the checkpoint actually re-validates against."""

    dt = 1.0
    _f_idx = []
    _finger_pin = None

    def __init__(self):
        _Demo.__init__(self)
        self.idx = {n: i for i, n in enumerate(ArmModel.Q8)}
        self.seen = []
        self.robot = SimpleNamespace(get_joint_positions=lambda: np.zeros(8))
        self.world = SimpleNamespace(step=lambda **k: None)
        # A SENTINEL, not a value: the checkpoint must re-validate against this exact object, or
        # the executor is judging a different hand than the planner was handed.
        self.fingers = {"finger_a_joint_1_1": -0.55}
        self.seen_fingers = []
        self.seen_checked = []

    def _commanded_fingers(self):
        return self.fingers

    def _arm_obstacles(self, *a, **k):
        obs = _Demo._arm_obstacles(self, *a, **k)
        self.seen.append(obs)
        return obs

    def _arm_model(self):
        # Records the hand every checkpoint judges, then defers to the real model.
        _demo = self

        class _Recording:
            def __getattr__(self, name):
                return getattr(M, name)

            def collides(self, q8, obs, margin, held=None, base=None, fingers=None):
                _demo.seen_fingers.append(fingers)
                # the set the check ACTUALLY judges
                _demo.seen_checked.append(list(obs))
                return M.collides(q8, obs, margin, held, base, fingers=fingers)

        return _Recording()

    def _arm_base_world(self):
        return np.zeros(3), np.eye(3)

    def base_ledger(self):
        return 0.0, 0.0, 0.0

    def set_base(self, *a):
        pass

    def _force(self, *a):
        pass

    def _apply(self, *a):
        pass

    def _with_closure_forced(self, q):
        return q

    def _plan_arm(self, goal, **k):
        return [np.asarray(goal, float)] * 2

    def _fallback(self, _r):
        pass


from morph.arm.obstacles import chassis_pad as _chassis_pad_real  # noqa: E402
from morph.arm.execute import commanded_fingers as _commanded_fingers_real  # noqa: E402
_ExecDemo._planned_obstacles = _exec_method(
    ("morph", "arm", "obstacles.py"), "planned_obstacles", {"chassis_pad": _chassis_pad_real})
_ExecDemo._execute_arm_path = _bind_executor({"np": np, "math": math, "os": os, "ArmModel": ArmModel, "MARGIN": MARGIN,
     "HEADLESS": True, "print": lambda *a, **k: None})


def test_the_checkpoint_revalidates_against_the_set_the_path_was_planned_with():
    """The checkpoint re-measures the base and re-checks the REMAINING waypoints. Built from a
    different policy than the plan, it rejects a path the planner accepted -- and a checkpoint
    rejection latches a safety abort that does NOT replay the bake."""
    d = _ExecDemo()
    d._execute_arm_path([PARK] * 3, 1.0, checkpoint_every=1, grasp_names=("pickup_obj_0",))
    assert d.seen, "no checkpoint ever built an obstacle set"
    for obs in d.seen:
        assert _same(obs, _obstacles(("pickup_obj_0",))), (
            "the checkpoint re-validates against a different obstacle set than the plan used")
    # ...and the set the check ACTUALLY JUDGES carries the planner's deck floor: judging at full
    # MARGIN while the planner floored the deck refused a waypoint at every checkpoint.
    from morph.arm.collision import CHASSIS_PAD_M, CHASSIS_PAD_LINKS, CHASSIS_PAD_PRIM
    assert d.seen_checked, "no checkpoint ever ran a collision check"
    for obs in d.seen_checked:
        floors = [o for o in obs if len(o) == 4 and abs(float(o[2]) - CHASSIS_PAD_M) < 1e-12
                  and tuple(o[3]) == tuple(CHASSIS_PAD_LINKS)]
        assert len(floors) == 1, (
            f"the checkpoint judged the deck ({CHASSIS_PAD_PRIM}) without the planner's "
            f"{CHASSIS_PAD_M * 1000:.0f}mm floor: {len(floors)} floored entries in {len(obs)}")
    # ...and against the same HAND: judging the swept union while the planner was handed the pads
    # rejects poses the plan cleared, and a checkpoint rejection latches an abort with the object held.
    assert d.seen_fingers, "no checkpoint ever ran a collision check"
    assert all(f is d.fingers for f in d.seen_fingers), (
        f"the checkpoint judged a different hand than the robot reports: {d.seen_fingers}")


# --------------------------------------------------------------------------- the allowance

def _grasped_target():
    """The object where the shipped grasp holds it, as a world-frame AABB."""
    hp, hR = M.fk(GRASP)[M.hand_link]
    obj_h = np.array([0.0015, 0.0507, 0.1703])                  # tests/test_armfk.py's OBJH
    r_obj = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    return ArmModel._xform_aabb(np.array([[-0.055, -0.055, -0.075], [0.055, 0.055, 0.075]]),
                                hp + hR @ obj_h, hR @ r_obj)


def test_only_the_hand_link_may_reach_a_grasped_object():
    tgt = _grasped_target()
    assert M.first_hit(GRASP, [(tgt, "world")], MARGIN) == M.hand_link, (
        "the premise: at the shipped grasp the hand is the only link on the object")
    assert M.first_hit(GRASP, [(tgt, "grasped")], MARGIN) is None
    # every OTHER link is still checked against it -- the boom above all
    assert M.first_hit(APPROACH, [(ON_THE_BOOM, "grasped")], MARGIN) == "Arm_Left_1"


def test_the_grasped_allowance_is_per_obstacle_not_a_world_exemption():
    """Correction 3: an allowance that discards obstacle identity silently disables collision
    against everything. The hand must still be blocked by every other obstacle in the same call."""
    tgt = _grasped_target()
    assert M.first_hit(GRASP, [(tgt, "grasped")], MARGIN) is None
    # THE SAME GEOMETRY under any other identity still blocks the same link, in the same call
    assert M.first_hit(GRASP, [(tgt, "grasped"), (tgt, "world")], MARGIN) == M.hand_link, (
        "one grasped object exempted the hand from every other world obstacle")
    assert M.first_hit(GRASP, [(tgt, "grasped"), (tgt, "chassis")], MARGIN) == M.hand_link


def test_the_allowance_cannot_be_narrowed_below_the_whole_gripper_box():
    """Correction 2: the model has NO finger links. `tools/extract_arm_model.py` folds every
    finger and palm collider into one Gripper_Link3_1 box, so permitting that link permits the
    palm AND every reachable finger pose. Nothing finer is representable -- do not read this
    allowance as finger-level permission."""
    assert not [k for k in M.link_aabb if "finger" in k.lower() or "palm" in k.lower()], (
        f"the model grew finger links: {list(M.link_aabb)}")
    extent = np.diff(M.link_aabb[M.hand_link], axis=0)[0]
    assert np.allclose(np.round(extent * 1000, 0), [282.0, 134.0, 179.0]), extent


def test_arm_blame_does_not_report_a_PER_OBSTACLE_allowance_as_a_fault():
    """`arm_blame` is a SECOND, hand-written copy of the exemption rules. Out of step with the one
    the check applies, it accuses the link of exactly the contact that was permitted -- and sends
    whoever reads the log after geometry that is behaving as designed."""
    out = []
    d = _Demo()
    d._arm_model = lambda: M
    _Demo._arm_blame = _exec_method(("morph", "arm", "obstacles.py"), "arm_blame",
                                    {"np": np, "MARGIN": MARGIN,
                                     "print": lambda *a, **k: out.append(" ".join(map(str, a)))})
    tgt = _grasped_target()
    base = (np.zeros(3), np.eye(3))
    d._arm_blame(GRASP, [(tgt, "world")], ["/World/target"], None, "goal", base)
    blamed = next(ln for ln in out if "hit:" in ln).split("hit: ")[1].split()[0]
    assert "1 link/obstacle overlap(s)" in out[-1], out
    out.clear()
    # A PAD, not a deletion: blame judges the padded pair at the pad, as `first_hit` does. The
    # invariant is AGREEMENT with the checker, which is what a second copy of the rules must preserve.
    padded = (tgt, "world", 0.0, (blamed,))
    d._arm_blame(GRASP, [padded], ["/World/target"], None, "goal", base)
    expect = sum(1 for lk in M.link_aabb
                 if M.first_hit(GRASP, [padded], MARGIN, None, base, links=(lk,)) is not None)
    assert f"{expect} link/obstacle overlap(s)" in out[-1], (
        f"blame disagrees with the checker on the padded pair: expected {expect}, got {out[-1]}")
    assert expect == 0 or M.first_hit(GRASP, [padded], 0.0, None, base) is not None, (
        "fixture: the padded pair neither clears nor overlaps, so this proves nothing")


def test_arm_blame_does_not_report_the_grasped_allowance_as_a_fault():
    """`arm_blame` duplicates the collision iteration; out of step with it, it names allowed
    contacts as the reason a plan failed and sends the reader after the wrong geometry."""
    out = []
    d = _Demo()
    d._arm_model = lambda: M
    _Demo._arm_blame = _exec_method(("morph", "arm", "obstacles.py"), "arm_blame",
                                    {"np": np, "MARGIN": MARGIN,
                                     "print": lambda *a, **k: out.append(" ".join(map(str, a)))})
    tgt = _grasped_target()
    d._arm_blame(GRASP, [(tgt, "grasped")], ["/World/target"], None, "goal",
                 (np.zeros(3), np.eye(3)))
    assert "0 link/obstacle overlap(s)" in out[-1], out
    out.clear()
    d._arm_blame(GRASP, [(tgt, "world")], ["/World/target"], None, "goal",
                 (np.zeros(3), np.eye(3)))
    assert "1 link/obstacle overlap(s)" in out[-1], out
    assert any(M.hand_link in ln for ln in out), out


# `first_hit` clears this pair comfortably because the boom's OBB is nowhere near it. Its world
# AABB is, which is the point: an AABB alone is nearly twice the boom's footprint at some yaws.
BLAME_Q8 = np.array([0.6, 0.45, 0.55, 0.40, 0.0, 0.0, 0.0, 0.0])
INSIDE_THE_BOOM_AABB = np.array([[0.448523, -0.017357, 0.62467],
                                 [0.498523, 0.032643, 0.67467]])


def test_arm_blame_uses_the_predicate_that_refused_not_a_broad_phase_copy():
    """`arm_blame` is the only account the operator gets of WHY a plan refused, and it re-derives
    the answer with its own AABB-vs-AABB test instead of the oriented one `collides` applied. It
    therefore names links the planner never refused for -- 189 of 196 probed turret/corner pairs,
    and here a boom with 120 mm of clearance. `retreat-probe` carried a wrong root cause for weeks
    for exactly this reason and was moved onto `first_hit`; this is its unfixed sibling.

    The error is one-directional: over 4000 random poses the broad phase never MISSED a real hit,
    so agreeing with `first_hit` can only ever withdraw a false accusation."""
    assert M.first_hit(BLAME_Q8, [(INSIDE_THE_BOOM_AABB, "world")], MARGIN, None,
                       (np.zeros(3), np.eye(3))) is None, "the pinned pose stopped being clear"
    out = []
    d = _Demo()
    d._arm_model = lambda: M
    _Demo._arm_blame = _exec_method(("morph", "arm", "obstacles.py"), "arm_blame",
                                    {"np": np, "MARGIN": MARGIN,
                                     "print": lambda *a, **k: out.append(" ".join(map(str, a)))})
    d._arm_blame(BLAME_Q8, [(INSIDE_THE_BOOM_AABB, "world")], ["/World/boom_aabb"], None,
                 "start", (np.zeros(3), np.eye(3)))
    assert "0 link/obstacle overlap(s)" in out[-1], (
        f"blame accuses a pose the checker cleared by 120mm: {out}")


# A carried box hanging clear of every arm link, and a world obstacle only IT reaches. `link_aabb`
# has no "__held__" key; `_links` appends one (morph/arm/model.py) so `link_aabbs`/`first_hit` do.
BLAME_HELD_Q8 = np.array([0.0, 0.45, 0.55, 0.40, 0.0, 0.0, 0.0, 0.0])
BLAME_HELD_BOX = np.array([[-0.04, 0.56, -0.04], [0.04, 0.64, 0.04]])
ONLY_THE_HELD_BOX = np.array([[0.537871, 0.725, 0.299516], [0.577871, 0.765, 0.339516]])


def test_arm_blame_can_blame_the_HELD_object_the_checker_refused_for():
    """Blaming per-link off `m.link_aabb` silently drops the carried box: it is a pseudo-link that
    only `_links` produces. `isValid` checks it (`collides(..., self.held, ...)`), so a carry plan
    refused BECAUSE of the object in the hand reported "0 link/obstacle overlap(s)" and sent the
    operator looking for a fault that was never in a link at all.

    Under-reporting is strictly worse than the broad-phase over-reporting this replaced -- that at
    least named something. `_plan_arm` passes `held` straight through, and the carry plans use it."""
    base = (np.zeros(3), np.eye(3))
    # The checker refuses, and it refuses for the held box specifically.
    assert M.first_hit(BLAME_HELD_Q8, [(ONLY_THE_HELD_BOX, "world")], MARGIN, BLAME_HELD_BOX,
                       base) == "__held__"
    out = []
    d = _Demo()
    d._arm_model = lambda: M
    _Demo._arm_blame = _exec_method(("morph", "arm", "obstacles.py"), "arm_blame",
                                    {"np": np, "MARGIN": MARGIN,
                                     "print": lambda *a, **k: out.append(" ".join(map(str, a)))})
    d._arm_blame(BLAME_HELD_Q8, [(ONLY_THE_HELD_BOX, "world")], ["/World/held_only"],
                 BLAME_HELD_BOX, "start", base)
    assert "1 link/obstacle overlap(s)" in out[-1], f"blame is blind to the held box: {out}"
    assert any("__held__" in ln for ln in out), out


def test_arm_blame_names_the_PRIM_of_the_obstacle_it_blames():
    """The report named a tag and eight box corners, so the operator matched a box by eye against
    the 124 the shipped set carries -- and the prim was sitting unused beside the call."""
    base = (np.zeros(3), np.eye(3))
    far = np.array([[9.0, 9.0, 9.0], [9.1, 9.1, 9.1]])
    obs = [(far, "world"), (_grasped_target(), "world")]
    paths = ["/World/far_away", "/World/Geometry/the_real_one"]
    # The fixture only proves anything if the SECOND entry is the one the checker refuses for.
    assert M.first_hit(GRASP, [obs[0]], MARGIN, None, base) is None
    assert M.first_hit(GRASP, [obs[1]], MARGIN, None, base) is not None
    out = []
    d = _Demo()
    d._arm_model = lambda: M
    _Demo._arm_blame = _exec_method(("morph", "arm", "obstacles.py"), "arm_blame",
                                    {"np": np, "MARGIN": MARGIN,
                                     "print": lambda *a, **k: out.append(" ".join(map(str, a)))})
    d._arm_blame(GRASP, obs, paths, None, "goal", base)
    hit = next(ln for ln in out if "hit:" in ln)
    assert "/World/Geometry/the_real_one" in hit, f"blame does not name the prim: {out}"
    assert "/World/far_away" not in hit, f"blame names the wrong prim: {out}"


def test_arm_blame_refuses_a_prim_list_that_does_not_match_its_obstacles():
    """`zip` truncates to the shorter of the two, so a short list silently drops every obstacle
    past the gap and a long one misnames nothing visibly. Both must raise, not report."""
    d = _Demo()
    d._arm_model = lambda: M
    _Demo._arm_blame = _exec_method(("morph", "arm", "obstacles.py"), "arm_blame",
                                    {"np": np, "MARGIN": MARGIN, "print": lambda *a, **k: None})
    obs = [(_grasped_target(), "world"), (_grasped_target(), "world")]
    for paths in ([], ["/World/one"], ["/a", "/b", "/c"]):
        try:
            d._arm_blame(GRASP, obs, paths, None, "goal", (np.zeros(3), np.eye(3)))
        except ValueError as e:
            assert "prim paths" in str(e), e
        else:
            raise AssertionError(f"blame accepted {len(paths)} prim paths for {len(obs)} obstacles")


def test_a_sphere_collider_is_boxed_as_the_sphere_not_its_rotated_extent_cube():
    """USD stores a Sphere's bound as its authored `extent` BOX, and BBoxCache carries that box
    through the transform -- so a roller on a hub that has spun arrives as the AABB of a ROTATED
    160mm cube, up to 160*sqrt(3) = 277mm across. Measured live: 277 x 234 x 255mm for an 80mm
    sphere (logs/sphere/run3.log).

    A sphere's true AABB is rotation-invariant. The phantom shell is 58.6mm per axis, against a
    MARGIN of 20mm, and it put a no-go slab on top of the deck the arm picks from while real roller
    metal stops below it. Rebuild the bound at the PRODUCER, so all nine consumers inherit it and
    the obstacle stays a plain (box, tag) 2-tuple -- the third slot already means an ACM entry."""
    got = next((b for b, _t in _obstacles() if _t == "chassis"
                and np.allclose(0.5 * (b[0] + b[1]), ROLLER_CENTRE)), None)
    assert got is not None, "the roller obstacle vanished from the set"
    extent = got[1] - got[0]
    assert np.allclose(extent, 0.16), (
        f"sphere emitted as {np.round(extent * 1000, 1).tolist()}mm, expected a 160mm cube")
    assert np.allclose(0.5 * (got[0] + got[1]), ROLLER_CENTRE), "the centre moved"
    # and it must SHRINK, never grow -- a tighter obstacle cannot invalidate a cleared path
    assert np.all(got[0] >= ROLLER_ROTATED_BOUND[0] - 1e-12)
    assert np.all(got[1] <= ROLLER_ROTATED_BOUND[1] + 1e-12)


# --------------------------------------------------------------------------- the other consumers

def test_the_hand_is_the_DRIVE_TARGET_while_the_grip_owns_it():
    """`_commanded_fingers` must report what the fingers are DRIVEN to, not where they are.

    While an object is held the grip stack owns the hand: `_apply` replaces those DOFs with
    `_finger_cmd` and `_force` excludes them, so the pads track a target with 42-82 mrad of lag.
    A measurement is then a sample of a hand in motion -- two samples either side of a
    `world.step` disagree, and the planner and the checkpoints end up judging different hands with
    the payload held. `fhold` does NOT prevent this; only reading the target does.

    Mutation: read `get_joint_positions()` unconditionally -- must fail.
    """
    _cf = _commanded_fingers_real

    class _Grip:
        idx = {"finger_a_joint_1_1": 3, "finger_b_joint_1_1": 4, "ArmLeftJoint_1": 0}
        _f_idx = np.array([3, 4])
        _finger_cmd = np.array([0.695, 0.695])        # the CLOSE target, constant while held
        robot = SimpleNamespace(                      # ...while the pads still lag behind it
            get_joint_positions=lambda: np.array([0.0, 0.0, 0.0, 0.613, 0.646]))

        def _arm_model(self):
            return SimpleNamespace(joints={"finger_a_joint_1_1": 1, "finger_b_joint_1_1": 1})

    got = _cf(_Grip())
    assert got == {"finger_a_joint_1_1": 0.695, "finger_b_joint_1_1": 0.695}, (
        f"the hand reported is the measured one, which drifts under load: {got}")


def test_without_a_grip_the_hand_is_the_MEASURED_one():
    """Nothing owns the fingers, so there is no target and the pose IS the command. Paired with
    the test above in one file so a stub returning a single source cannot satisfy both."""
    _cf = _commanded_fingers_real

    class _Free:
        idx = {"finger_a_joint_1_1": 3}
        _f_idx = np.array([3])
        _finger_cmd = None
        robot = SimpleNamespace(get_joint_positions=lambda: np.array([0.0, 0.0, 0.0, -0.55]))

        def _arm_model(self):
            return SimpleNamespace(joints={"finger_a_joint_1_1": 1})

    assert _cf(_Free()) == {"finger_a_joint_1_1": -0.55}


def test_every_cartesian_call_site_judges_the_hand_with_its_COMMANDED_fingers():
    """UNIVERSAL over call sites, because the last fix missed one. The hand's swept box spans finger
    configurations the hand never commands and its floor sits 39mm below where the fingers reach;
    a site that omits `fingers=` judges the hand with that sweep while its siblings use the pads.

    Covers `move_linear` call sites too: a motion routed through the primitive still owes it the
    finger state, and the sweep would otherwise stop guarding a site the moment it migrates."""
    import glob as _glob
    missing = []
    for path in _glob.glob(os.path.join(ROOT, "morph", "**", "*.py"), recursive=True):
        src = open(path, encoding="utf-8").read()
        rel = os.path.relpath(path, ROOT)
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            raise AssertionError(f"{rel}:{e.lineno} will not parse on this venv: {e.msg}")
        for c in ast.walk(tree):
            if not (isinstance(c, ast.Call)
                    and getattr(c.func, "id", "") in ("plan_cartesian", "move_linear")):
                continue
            if "fingers" not in {k.arg for k in c.keywords}:
                missing.append(f"{rel}:{c.lineno} ({getattr(c.func, 'id', '')})")
    assert not missing, (
        f"a straight line was planned without the commanded fingers at {missing} -- those sites "
        f"judge the hand with the swept box while their siblings use the pads")


def test_arm_obstacles_never_annotates_an_obstacle():
    """A per-obstacle allowance is an ACM entry and belongs to ONE motion. `_arm_obstacles` is the
    single source for nine consumers -- `_plan_arm` and `_goal_set` among them, the planned tier
    that A4 forbids any exemption on -- so annotating HERE grants it to all of them at once. The
    annotation is built at the `move_linear` call site instead.

    Universal over construction sites, not a spot check on one return value: a second `append` that
    annotates would be invisible to a test that only inspects the default call."""
    src = open(os.path.join(ROOT, "morph", "arm", "stage.py"), encoding="utf-8").read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "arm_obstacles")
    built = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "append"
             and getattr(getattr(n.func, "value", None), "id", "") == "out"]
    assert built, "no `out.append` in _arm_obstacles -- this test no longer guards anything"
    for call in built:
        arg = call.args[0]
        assert isinstance(arg, ast.Tuple) and len(arg.elts) == 2, (
            f"_arm_obstacles appends a {len(getattr(arg, 'elts', []))}-element obstacle at "
            f"line {call.lineno}; the allowance must be added by the caller, not here")
    for kw in ({}, {"grasp_names": ("pickup_obj_0",)}, {"exclude_names": ("pickup_obj_0",)},
               _approach_policy()):
        assert all(len(o) == 2 for o in _obstacles(**kw)), kw


def test_plan_cartesian_takes_an_annotated_obstacle_and_still_rejects_a_malformed_one():
    """`move_linear` is the one tier that may carry an allowance, and `plan_cartesian` is how it
    gets there -- so a 3-tuple must PARSE. A 4-tuple must still not: the elements are positional,
    so a wrong length is read as a different obstacle rather than as an error."""
    obs = _obstacles(("pickup_obj_0",))
    up = np.array([0.0, 0.0, 0.20])
    # parsed AND honoured: the target sits on the boom at this pose, so the reason is geometric
    assert plan_cartesian(M, APPROACH, delta=up, obstacles=obs, margin=MARGIN,
                          bounds=M.bounds())[1] == "blocked at q_start"
    clear = _obstacles(exclude_names=("pickup_obj_0",))
    assert plan_cartesian(M, APPROACH, delta=up, obstacles=clear, margin=MARGIN,
                          bounds=M.bounds())[1] is None

    # the SAME blocking set, PADDED. A pad of 0 holds the named link to must-not-overlap, so blame
    # must MOVE to the next link rather than clear the set -- which proves the pad reached the check.
    blamed = M.first_hit(APPROACH, obs, MARGIN)
    assert blamed is not None, "the fixture no longer blocks; what follows would be vacuous"
    overlapping = M.first_hit(APPROACH, obs, 0.0) == blamed
    padded = [(o[0], o[1], 0.0, (blamed,)) for o in obs]
    second = M.first_hit(APPROACH, padded, MARGIN)
    if overlapping:
        # The pad CANNOT excuse geometry that already intersects -- that is the floor's whole
        # point, and asserting the excuse here would be asserting the deletion this replaced.
        assert second == blamed, (
            f"a pad excused {blamed}, which OVERLAPS the box -- that is the deletion, not a floor")
    else:
        assert second is not None and second != blamed, (
            f"padding {blamed} cleared the whole set instead of moving the blame: {second}")
    assert plan_cartesian(M, APPROACH, delta=up, margin=MARGIN, bounds=M.bounds(),
                          obstacles=padded)[1] == "blocked at q_start", (
        "the pad lowered the margin for a link it did not name")
    # Pad EVERY link to 0: the set must then be judged exactly as an unpadded set at margin 0, which
    # is the strongest statement that the pad reached the check rather than being ignored.
    every = tuple(M.link_aabb)
    at_zero = M.first_hit(APPROACH, obs, 0.0)
    all_padded = M.first_hit(APPROACH, [(o[0], o[1], 0.0, every) for o in obs], MARGIN)
    assert all_padded == at_zero, (
        f"padding every link to 0 did not reproduce the margin-0 verdict: "
        f"{all_padded} vs {at_zero} -- the pad never reached the check")
    assert M.first_hit(APPROACH, obs, MARGIN) != at_zero or at_zero is not None, (
        "fixture: margin 0 and MARGIN agree here, so the comparison above proves nothing")

    # A 3-TUPLE is the old deletion form. It must now be refused outright, not reshaped: that is
    # what stops a caller-supplied exemption surviving the migration by accident.
    bad = plan_cartesian(M, APPROACH, delta=up,
                         obstacles=[(o[0], o[1], ()) for o in obs])
    assert bad[1].startswith("bad-args"), bad[1]
    worse = plan_cartesian(M, APPROACH, delta=up,
                           obstacles=[(o[0], o[1], 0.0, (), "extra") for o in obs])
    assert worse[1].startswith("bad-args"), worse[1]


def test_a_missing_prim_is_a_keyerror_not_a_silent_no_op():
    """`str(None)` caches the truthy "None" and matches no path: the request silently does nothing
    while reporting success, and every plan and clearance reading built on it is quietly wrong."""
    for kw in ({"grasp_names": ("pickup_obj_9",)}, {"exclude_names": ("pickup_obj_9",)}):
        try:
            _obstacles(**kw)
        except KeyError as e:
            assert "pickup_obj_9" in str(e), e
        else:
            raise AssertionError(f"no KeyError for a prim that is not on the stage: {kw}")


def test_dropping_an_obstacle_cannot_be_done_positionally():
    """The defect was an obstacle silently absent from the set. The positional argument is the one
    that KEEPS the object; dropping it has to be spelled out."""
    fn = _func(("morph", "arm", "stage.py"), "arm_obstacles")
    assert [a.arg for a in fn.args.args] == ["demo", "grasp_names"], _param_names(fn)
    assert [a.arg for a in fn.args.kwonlyargs] == ["exclude_names", "diagnostic_paths"], \
        _param_names(fn)
    obs = _obstacles(("pickup_obj_0",))
    assert [tg for _ab, tg in obs].count("grasped") == 1, obs


def test_the_parked_arm_is_an_obstacle_tagged_world_not_chassis():
    """Arm-2 never moves, so nothing plans FOR it -- but it is a solid body on the same chassis
    and arm-1 must not plan THROUGH it. Two things hid it, and both are exercised here:

      1. `_arm_obstacles` skipped prims whose `CollisionEnabledAttr` is False, and on the real
         stage arm-2's mast, columns and booms are all COL_OFF while only its FINGERS are enabled
         -- so the obstacle set held its fingertips and not its 1.64 m mast.
      2. Arm-2 lives under `base/`, so the tag expression called it "chassis", and
         `ArmModel.first_hit` EXEMPTS the column links from chassis boxes. Arm-1's columns would
         have swept through it even once it was present.

    Its own stage, deliberately: the shared `_PRIMS` fixture is consumed by the clearance and
    cartesian tests, and adding a 1.64 m mast to it changes what THEY measure."""
    prims = [_Prim("/World/Arm_1/Arm_Left_1", np.zeros((2, 3))),
             _Prim("/World/base/battery", BATTERY),
             _Prim("/World/base/Arm_2/mast", PARKED_MAST, enabled=False),   # as the real stage has it
             _Prim("/World/base/Arm_2/finger_a_1_2", PARKED_TIP),
             _Prim("/World/decor", SHELF, collider=False),
             _Prim("/World/junk", SHELF, enabled=False)]                    # disabled, NOT parked
    paths = dict(_PATHS, Arm_2="/World/base/Arm_2")
    d = _Demo()
    d.stage = _Stage(prims)
    _OBS_GLB["find_path"] = lambda _s, n: paths.get(n)
    # The feature ships OFF -- it makes the reach unplannable today (see `_arm_obstacles`) -- so
    # the test has to turn it on explicitly. Asserting the DEFAULT is a separate test below.
    os.environ["PARKED_ARM_OBSTACLE"] = "1"
    try:
        got = d._arm_obstacles()
    finally:
        os.environ.pop("PARKED_ARM_OBSTACLE", None)
        _OBS_GLB["find_path"] = lambda _s, n: _PATHS.get(n)

    mast = [tag for ob, tag in got if np.allclose(ob, PARKED_MAST)]
    tip = [tag for ob, tag in got if np.allclose(ob, PARKED_TIP)]
    assert mast, ("the parked arm's collision-DISABLED mast is not in the obstacle set -- arm-1 "
                  "can plan straight through the second arm's column")
    assert mast == ["world"], (
        f"the parked mast is tagged {mast[0]!r}; 'chassis' boxes are skipped for every link in "
        f"`column_group`, so arm-1's columns would not see the second arm at all")
    assert tip == ["world"], f"the parked arm's enabled fingertip is tagged {tip!r}, not 'world'"
    assert not [ob for ob, _ in got if np.allclose(ob, SHELF)], (
        "the exemption leaked: `/World/junk` is collision-disabled and OUTSIDE the parked arm, so "
        "it must still be skipped -- only the parked arm is exempt from that flag")
    assert [tag for ob, tag in got if np.allclose(ob, BATTERY)] == ["chassis"], (
        "the real chassis stopped being tagged 'chassis'; the column exemption exists for IT")


def test_the_parked_arm_obstacle_ships_off_with_its_reason_recorded():
    """It is a CORRECT change that currently makes things worse, and that combination is exactly
    what gets silently flipped back on by someone tidying up.

    Measured on a real cycle: with the parked arm in the obstacle set the reach cannot plan
    (`goal-ik-miss=1, no-plan=1`) and the attempt ends -- so it trades "does not know about
    arm-2" for "does not pick at all". Off until the reach plans
    with it on."""
    src = open(os.path.join(ROOT, "morph", "arm", "stage.py"), encoding="utf-8").read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "arm_obstacles")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and "PARKED_ARM_OBSTACLE" in ast.unparse(n)]
    assert calls, "`PARKED_ARM_OBSTACLE` is gone; the parked arm is now unconditional"
    default = [a.value for a in calls[0].args if isinstance(a, ast.Constant)][-1]
    assert default == "0", (
        f"`PARKED_ARM_OBSTACLE` now defaults to {default!r}. Turning it on makes the reach "
        f"unplannable and routes it to an unchecked bake -- re-measure a full cycle before "
        f"flipping this, and only flip it once the reach plans with it on.")



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
