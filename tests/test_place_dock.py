"""The place dock's standoff and its chassis-clearance floor: one definition each, and the
floor measured on the plate's nearest CORNER at the yaw each dock candidate parks at.
Offline, no Isaac:  .venv/bin/python3 tests/test_place_dock.py"""
import ast
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from morph.config import (CHASSIS_PLATE_X, CHASSIS_PLATE_Y,    # noqa: E402
                          DOCK_EDGE_M, RACK_FRONT_Y, dock_floor_y, plate_reach)

SQUARE_ON = -math.pi / 2             # facing the rack face
PARKED_SKEW = -math.pi / 2 - 0.195   # face_south + the descent turret: the absorbed-yaw candidate

SRC = os.path.join(ROOT, "morph", "place", "__init__.py")
with open(SRC) as _fh:
    TEXT = _fh.read()
TREE = ast.parse(TEXT)


def _assigned(name, tree=TREE):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name) and t.id == name]


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _calls(node, func):
    return [c for c in ast.walk(node) if isinstance(c, ast.Call)
            and isinstance(c.func, ast.Name) and c.func.id == func]


def _arg_src(call):
    """The call's single argument, as source text."""
    assert len(call.args) == 1 and not call.keywords, ast.dump(call)
    return ast.unparse(call.args[0])


def _fn(name):
    return next(n for n in ast.walk(TREE)
                if isinstance(n, ast.FunctionDef) and n.name == name)


def test_the_clearance_floor_is_written_once():
    """Two verbatim copies of the clamp shipped, the second re-testing what the first had just
    corrected. One assignment, or the second copy is back."""
    got = _assigned("_y_min")
    assert len(got) == 1, f"{len(got)} assignments to _y_min in morph/place/__init__.py"


def test_the_clearance_floor_prints_from_one_site():
    """The duplicate could never print -- the first clamp made its condition False -- so the log
    is no witness that it is gone. One print site is."""
    n = sum(1 for node in ast.walk(TREE) if isinstance(node, ast.Constant)
            and isinstance(node.value, str) and "floored at chassis clearance" in node.value)
    assert n == 1, f"{n} 'floored at chassis clearance' print sites"


def test_the_clamp_floors_on_the_yaw_the_base_will_park_at():
    """The name alone is not the fix: `dock_floor_y(0.0)` re-creates the 50 mm defect while still
    printing 100 mm. The clamp's argument must be the dock's own final yaw."""
    (node,) = _assigned("_y_min")
    assert "CHASSIS_PLATE_X" not in _names(node.value), ast.unparse(node.value)
    (call,) = _calls(node.value, "dock_floor_y")
    assert _arg_src(call) == "face_south", _arg_src(call)


def test_the_turret_probe_floors_each_candidate_at_its_own_yaw():
    """`_dock_for` compares two parked yaws; flooring both at the OUTER one reverts the whole
    effect, because it is the absorbed candidate's own floor that ends the absorption."""
    (call,) = _calls(_fn("_dock_for"), "dock_floor_y")
    assert _arg_src(call) == "_fs", _arg_src(call)


def test_the_readout_reads_the_same_projection_at_the_live_yaw():
    """The log's `chassis plate` figure and the floor must be one measurement, at one yaw, or the
    run reports a gap nothing enforced."""
    (node,) = _assigned("_plate")
    (call,) = _calls(node.value, "plate_reach")
    assert _arg_src(call) == "_byawd", _arg_src(call)
    assert "_byawd" in TEXT.split("dock clearance to the rack face", 1)[0], \
        "the readout still discards the base yaw"


def test_the_plate_reach_is_the_box_support_at_the_parked_yaw():
    """Square on it IS the half-length; at any other yaw the corner leads. Dropping either term
    reads the plate short, and the 51.5 mm between the two yaws is the whole defect."""
    assert abs(plate_reach(SQUARE_ON) - CHASSIS_PLATE_X) < 1e-12, plate_reach(SQUARE_ON)
    assert abs(plate_reach(0.0) - CHASSIS_PLATE_Y) < 1e-12, plate_reach(0.0)
    assert abs(plate_reach(PARKED_SKEW) - 0.401497) < 1e-6, plate_reach(PARKED_SKEW)
    assert abs(dock_floor_y(SQUARE_ON) - (-2.96)) < 1e-12, dock_floor_y(SQUARE_ON)
    assert abs(dock_floor_y(PARKED_SKEW) - (RACK_FRONT_Y + 0.401497 + DOCK_EDGE_M)) < 1e-6
    assert abs((dock_floor_y(PARKED_SKEW) - dock_floor_y(SQUARE_ON)) - 0.0515) < 1e-4, (
        dock_floor_y(PARKED_SKEW) - dock_floor_y(SQUARE_ON))


def test_the_standoff_is_written_once():
    """`stand` and the turret probe's copy of it were the same 0.20 twice."""
    got = _assigned("stand") + _assigned("_stand0")
    assert len(got) == 1, f"{len(got)} standoff assignments"


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
