"""`tools/extract_arm_model.py` must not silently delete what it cannot regenerate.
Offline, no Isaac:  .venv/bin/python3 tests/test_extract_model.py
"""
import ast
import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "tools", "extract_arm_model.py")


def _carry_forward():
    """`carry_forward` alone: the module imports Isaac at import time, so it is compiled from the
    source TEXT and only that function is executed."""
    tree = ast.parse(open(SRC, encoding="utf-8").read())
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "carry_forward"), None)
    assert fn is not None, "carry_forward is gone -- a re-extraction now deletes link_parts again"
    mod = types.ModuleType("_extract_under_test")
    mod.__dict__["print"] = lambda *a, **k: None
    exec(compile(ast.Module(body=[fn], type_ignores=[]), SRC, "exec"), mod.__dict__)
    return mod.carry_forward


def test_the_shipped_model_carries_what_the_extractor_cannot_write():
    """The premise. If the shipped model stops carrying these, this whole file is moot -- and if
    the extractor starts writing them, the carry-forward should go, not linger."""
    shipped = json.load(open(os.path.join(ROOT, "usd", "_arm_model.json")))
    assert shipped.get("link_parts"), "the shipped model has no link_parts"
    assert shipped.get("validated"), "the shipped model has no validation record"
    # what the extractor BUILDS, by AST: the dict literal assigned to `model`
    tree = ast.parse(open(SRC, encoding="utf-8").read())
    lit = next(n.value for n in ast.walk(tree)
               if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict)
               and any(isinstance(k, ast.Constant) and k.value == "link_aabb" for k in n.value.keys))
    keys = {k.value for k in lit.keys if isinstance(k, ast.Constant)}
    assert "link_parts" not in keys and "validated" not in keys, (
        f"the extractor now writes {keys & {'link_parts', 'validated'}} -- delete the "
        f"carry-forward instead of keeping two sources for one field")


def test_measured_geometry_survives_a_re_extraction():
    """`link_parts` is measured with audit_colliders and entered by hand. A plain overwrite deletes
    it and the arm silently returns to the union box it was decomposed to escape."""
    cf = _carry_forward()
    old = {"link_parts": {"Arm_1": {"rail_l": [[0, 0, 0], [1, 1, 1]]}}, "link_aabb": {"x": 1}}
    out = cf(old, {"link_aabb": {"x": 2}})
    assert out["link_parts"] == old["link_parts"], "a re-extraction deleted the measured parts"
    assert out["link_aabb"] == {"x": 2}, "the carry-forward overwrote what WAS re-measured"


def test_a_validation_record_never_rides_along_unmarked():
    """`validated` describes the model it was run on. Carrying it silently would let a re-extracted
    model claim a live FK check it never had."""
    cf = _carry_forward()
    was = {"date": "2026-09-04", "worst_link_error_mm": 0.07}
    out = cf({"validated": was}, {"link_aabb": {}})
    assert out["validated"]["stale"] is True, "a re-extracted model claims an old validation"
    assert out["validated"]["date"] == was["date"], "the record was replaced instead of marked"
    assert "stale_reason" in out["validated"], "stale without a reason is not a record"


def test_nothing_is_invented_when_there_was_no_previous_model():
    cf = _carry_forward()
    out = cf({}, {"link_aabb": {"x": 1}})
    assert out == {"link_aabb": {"x": 1}}, f"the carry-forward invented fields: {out}"


if __name__ == "__main__":
    fails = 0
    for name in sorted(n for n in dir() if n.startswith("test_")):
        try:
            globals()[name]()
            print(f"PASS {name}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL {name}: {e}")
    print("\nall passed" if not fails else f"\n{fails} failed")
    sys.exit(1 if fails else 0)
