"""The floor plan is one table. It was two, and two copies of a number drift.

`morph/config.py` inflates the racks for the spawn keep-outs and `nav_plan.py` inflates the same
rectangles for the base planner, so a rack edited in one file and not the other puts the planner
around an obstacle the spawner does not know about. Run:
    ./.venv/bin/python3 tests/test_floor_plan.py
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morph.rects import RACK_RECTS, WALL_RECTS                                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The values as they shipped before the two copies were merged. A rack really moving is a scene
# change and this list moves with it -- deliberately, in the same commit, having read the scene.
SHIPPED_RACKS = ((2.55, 4.34, -3.96, -3.41), (4.34, 7.96, -3.96, -3.41),
                 (2.54, 7.96, -4.55, -3.97), (0.72, 7.97, -7.96, -7.42),
                 (2.54, 7.96, -0.66, -0.04))
SHIPPED_WALLS = ((-0.1, 0.1, -8.0, 0.0), (7.9, 8.1, -8.0, 0.0), (0.0, 8.0, -0.1, 0.1))


def _sources():
    """Every source that runs the robot: the package and the entry points beside it. An allow-list,
    because the defect this scans for is a copy pasted anywhere a reader would call shipped code."""
    out = [f for f in sorted(os.listdir(ROOT)) if f.endswith(".py")]
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, "morph")):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        out += [os.path.relpath(os.path.join(dirpath, f), ROOT)
                for f in sorted(filenames) if f.endswith(".py")]
    assert len(out) >= 30, f"the scanned universe is {len(out)} files: too small to bind"
    return out


def _module(rel):
    return ast.parse(open(os.path.join(ROOT, rel), encoding="utf-8").read())


def _assigned(tree, name):
    """The top-level value bound to `name`, or None if the module does not bind it."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    return None


def test_the_tables_still_hold_the_shipped_geometry():
    assert tuple(RACK_RECTS) == SHIPPED_RACKS, f"the racks moved: {RACK_RECTS}"
    assert tuple(WALL_RECTS) == SHIPPED_WALLS, f"the walls moved: {WALL_RECTS}"


def test_the_planner_obstacles_are_the_racks_and_the_walls_and_nothing_else():
    """Read out of the planner's own module, so an edit there is caught, not assumed away."""
    sys.path.insert(0, ROOT)
    import nav_plan
    assert tuple(nav_plan.OBSTACLE_RECTS) == SHIPPED_RACKS + SHIPPED_WALLS, (
        f"the base planner's obstacle set no longer matches the floor plan: "
        f"{nav_plan.OBSTACLE_RECTS}")


def test_no_module_declares_a_second_copy_of_the_floor_plan():
    """The literals, not the names: a copy pasted under any name is the defect this test exists
    for, and it would pass a name check."""
    owned = {tuple(r) for r in SHIPPED_RACKS} | {tuple(r) for r in SHIPPED_WALLS}
    offenders = []
    for rel in _sources():
        if rel == os.path.join("morph", "rects.py"):
            continue
        if rel:
            try:
                tree = ast.parse(open(os.path.join(ROOT, rel), encoding="utf-8").read())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Tuple, ast.List)) or len(node.elts) != 4:
                    continue
                vals = []
                for e in node.elts:
                    if isinstance(e, ast.Constant) and isinstance(e.value, (int, float)):
                        vals.append(float(e.value))
                    elif (isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub)
                          and isinstance(e.operand, ast.Constant)):
                        vals.append(-float(e.operand.value))
                if len(vals) == 4 and tuple(vals) in owned:
                    offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        f"a floor-plan rectangle is written out again at {offenders}: import it from "
        f"morph/rects.py instead, or the two copies drift")


def test_rects_imports_nothing_and_prints_nothing():
    """It is read by a second interpreter that must not pick up this project's environment."""
    tree = _module(os.path.join("morph", "rects.py"))
    bad = [ast.dump(n) for n in ast.walk(tree)
           if isinstance(n, (ast.Import, ast.ImportFrom))
           or (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print")]
    assert not bad, f"morph/rects.py is no longer pure data: {bad}"


def test_config_reads_the_shared_table_rather_than_rebuilding_it():
    tree = _module(os.path.join("morph", "config.py"))
    assert _assigned(tree, "RACK_RECTS") is None, (
        "morph/config.py assigns RACK_RECTS again instead of importing it")
    assert any(isinstance(n, ast.ImportFrom) and n.module == "morph.rects"
               and any(a.name == "RACK_RECTS" for a in n.names) for n in tree.body), (
        "morph/config.py no longer imports RACK_RECTS from morph.rects")


if __name__ == "__main__":
    fails = 0
    for name in sorted(n for n in dir() if n.startswith("test_")):
        try:
            globals()[name]()
            print(f"PASS {name}")
        except Exception as e:
            fails += 1
            print(f"FAIL {name}: {e}")
    print("\nall passed" if not fails else f"\n{fails} failed")
    sys.exit(1 if fails else 0)
