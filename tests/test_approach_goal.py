"""`ApproachStage._object_relative_goal` under OBJ_GOAL=2, the shipped mode: the frame self-check
is a diagnostic and must not decide the goal. Real arm model, the captured reach request's base
and 6b goal, IK stubbed.  Offline:  .venv/bin/python3 tests/test_approach_goal.py"""
import ast
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from morph.arm.model import ArmModel  # noqa: E402
from morph.config import KNOWN, MOUTH_VEC_HAND  # noqa: E402

REQ = json.load(open(os.path.join(ROOT, "tools", "bench_problems", "reach.json")))
_SRC = os.path.join(ROOT, "morph", "pick", "approach.py")


def _compile_goal(glb):
    """`_object_relative_goal` compiled alone: `morph.pick` imports Isaac at module level."""
    tree = ast.parse(open(_SRC).read())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_object_relative_goal")
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), _SRC, "exec"), glb)
    return glb["_object_relative_goal"]


def _quat(R):
    """(w, x, y, z) of a rotation matrix, the layout `get_world_pose` returns."""
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = np.copysign(np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


class _Prim:
    def __init__(self, p, q):
        self.p, self.q = p, q

    def get_world_pose(self):
        return np.asarray(self.p, float), np.asarray(self.q, float)


class _Obj:
    def __init__(self, p):
        self.p = p

    def get_world_poses(self):
        return np.asarray([self.p], float), None


class _Demo:
    """The adapters mode 2 reads. `q8_fail` makes the live-q8 read raise; `hand_fail` makes the
    live hand pose unreadable."""

    def __init__(self, q8_fail=False, hand_fail=False):
        self.m = ArmModel.load(os.path.join(ROOT, "usd", "_arm_model.json"),
                               os.path.join(ROOT, "usd", "_closure_lut.json"))
        self.bp, self.bR = (np.asarray(x, float) for x in REQ["base"])
        self.goal = np.asarray(REQ["goals"][0], float)
        p_h, R_h = self.m.fk(self.goal)[self.m.hand_link]
        self.hw = self.bp + self.bR @ p_h
        self.hq = _quat(self.bR @ R_h)
        self.mouth = self.bR @ R_h @ np.asarray(MOUTH_VEC_HAND, float)
        self.objs = {0: _Obj(self.hw + 0.1 * self.mouth)}
        self.q8_fail, self.hand_fail = q8_fail, hand_fail
        self.fallbacks, self.ik_calls = {}, []

    def _arm_model(self):
        return self.m

    def _arm_base_world(self):
        return self.bp, self.bR

    def _arm_q8(self):
        if self.q8_fail:
            raise RuntimeError("injected: no live q8")
        return self.goal

    def _hand_prims(self):
        if self.hand_fail:
            raise RuntimeError("injected: hand prim gone")
        return {self.m.hand_link: _Prim(self.hw, self.hq)}

    def _pinch(self):
        return self.hw + 0.05 * self.mouth

    def base_pose(self):
        return float(self.bp[0]), float(self.bp[1]), 0.0

    def _arm_bounds(self):
        return np.asarray(REQ["bounds"][0], float), np.asarray(REQ["bounds"][1], float)

    def _fallback(self, tag):
        self.fallbacks[tag] = self.fallbacks.get(tag, 0) + 1


def _fake_goal_ik(demo, m, p_des, R_des, seed, hand, bounds):
    demo.ik_calls.append(np.asarray(p_des, float))
    q = np.asarray(seed, float).copy()
    q[0] += 0.05
    return q, 0.0, 0.0


def _run(demo):
    goal = _compile_goal({"np": np, "os": os, "math": math, "KNOWN": KNOWN, "ArmModel": ArmModel,
                          "MOUTH_VEC_HAND": MOUTH_VEC_HAND, "goal_ik": _fake_goal_ik,
                          "print": lambda *a, **k: None})
    prev = {k: os.environ.get(k) for k in ("OBJ_GOAL", "OBJ_GOAL_AIM")}
    os.environ["OBJ_GOAL"], os.environ["OBJ_GOAL_AIM"] = "2", "0"
    try:
        return goal(demo, demo.goal, 0)
    finally:
        for k, v in prev.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def test_the_healthy_path_builds_the_goal_from_the_live_object():
    demo = _Demo()
    got = _run(demo)
    assert demo.ik_calls and abs(got[0] - demo.goal[0] - 0.05) < 1e-12, (got, demo.fallbacks)
    assert demo.fallbacks == {}, demo.fallbacks


def test_a_failed_frame_check_does_not_downgrade_the_goal():
    demo = _Demo(q8_fail=True)
    got = _run(demo)
    assert "goal-exception" not in demo.fallbacks, demo.fallbacks
    assert demo.ik_calls and abs(got[0] - demo.goal[0] - 0.05) < 1e-12, (got, demo.fallbacks)


def test_an_unreadable_live_hand_is_a_counted_miss_to_the_baked_goal():
    demo = _Demo(hand_fail=True)
    got = _run(demo)
    assert demo.fallbacks.get("goal-exception") == 1 and not demo.ik_calls, demo.fallbacks
    assert np.array_equal(got, demo.goal)


if __name__ == "__main__":
    names = [n for n in list(globals()) if n.startswith("test_")]
    failed = 0
    for n in names:
        try:
            globals()[n]()
            print(f"PASS {n}")
        except Exception as e:                     # noqa: BLE001 -- a runner reports, it does not raise
            failed += 1
            print(f"FAIL {n}: {e}")
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
