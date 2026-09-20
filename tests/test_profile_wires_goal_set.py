"""The profile must wire the goal-set expansion ON.

Offline: reads morph/config.py's source and does not import it -- importing prints and applies
the profile to the live environment as a side effect. Run:
./.venv/bin/python3 tests/test_profile_wires_goal_set.py"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "morph", "config.py")


def _profile_entries():
    """Every string->string pair PROFILE ends up holding: the initial `PROFILE =
    {...}` literal plus every later `PROFILE.update({...})` call, merged in source order
    the same way the dict itself merges them at runtime."""
    tree = ast.parse(open(CONFIG).read())
    entries = {}
    for node in ast.walk(tree):
        d = None
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "PROFILE" for t in node.targets)
                and isinstance(node.value, ast.Dict)):
            d = node.value
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "PROFILE"
                and node.args and isinstance(node.args[0], ast.Dict)):
            d = node.args[0]
        if d is None:
            continue
        for k, v in zip(d.keys, d.values):
            if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                entries[k.value] = v.value
    return entries


def test_profile_wires_the_track_f2_goal_set_on():
    """The defect this closes: OBJ_GOAL_SET had zero occurrences outside the planner's own
    `os.environ.get(..., "0")`, so the capability the plan calls 'layer-1 DONE' never actually ran
    in production. §92's 5-cycle run placed 5/5 with it on; nothing measured says off is safer."""
    entries = _profile_entries()
    assert entries.get("OBJ_GOAL_SET") == "1", (
        f'PROFILE must set OBJ_GOAL_SET to "1", got {entries.get("OBJ_GOAL_SET")!r}')


def test_profile_arms_the_tracking_instrument_the_track_gate_reads():
    """`track_gate` (verify_place.py) keys on `TRACK BREACH`, printed by robot.py only
    under `TRACK_ERR=1` -- and that guard is the ONLY occurrence of the name in the
    tree, so with the profile off the gate cannot fail any shipped run. TRACK_TOL_MRAD=5.0 is
    measured (118 samples, every gated joint 0.00 mrad max and median, against 6 mrad at the obj-6
    failure), so the instrument belongs on by default."""
    entries = _profile_entries()
    assert entries.get("TRACK_ERR") == "1", (
        f'PROFILE must set TRACK_ERR to "1" or track_gate is vacuous, '
        f'got {entries.get("TRACK_ERR")!r}')


def _arm_drive_stage():
    """The shipped ARM_DRIVE_STAGE default: the `("ARM_DRIVE_STAGE", ...)` pair in the ARM_DRIVE=1
    setdefault list, read from source, as the tuple of stage names it names."""
    tree = ast.parse(open(CONFIG).read())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Tuple) and len(node.elts) == 2
                and all(isinstance(e, ast.Constant) for e in node.elts)
                and node.elts[0].value == "ARM_DRIVE_STAGE"):
            return tuple(s.strip() for s in node.elts[1].value.split(","))
    raise AssertionError("no ARM_DRIVE_STAGE default found in morph/config.py")


def test_arm_drive_stage_excludes_the_two_stages_that_execute_a_planned_path():
    """`execute_path` runs the planned reach (stage "reach"); the settle snaps to its pose. Neither is in this list, so `_drive_on` is False
    throughout every planned motion, `_track_tols` appends no kind="gate" row, and `TRACK BREACH`
    cannot print for it -- a green track gate says nothing about the planned path. Pin the scope so
    a stage added here is a deliberate widening, not a silent one."""
    stages = _arm_drive_stage()
    assert not {"reach", "settle", "all"} & set(stages), (      # "all" covers every stage
        f"reach/settle are ungated by design; ARM_DRIVE_STAGE now ships {stages}")


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
