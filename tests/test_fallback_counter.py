"""The per-cycle fallback counter's accounting and the log contract verify_place.py parses.
Offline: no Isaac, no simulator. Run: ./.venv/bin/python3 tests/test_fallback_counter.py"""
import ast
import contextlib
import csv
import io
import itertools
import json
import math
import os
import re
import sys
import tempfile
import types
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from morph.arm.model import ArmModel as _ARM_MODEL  # noqa: E402

LINE = re.compile(r">>> plan-fallbacks\[([^\]]+)\]: (\d+)")


def _fmt(tag, counts):
    """The exact line play_isaac.py must print. Kept here so the test owns the contract."""
    total = sum(counts.values())
    if not total:
        return f">>> plan-fallbacks[{tag}]: 0"
    parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    return f">>> plan-fallbacks[{tag}]: {total} ({parts})"


def test_zero_fallbacks_prints_a_bare_zero():
    assert _fmt("obj 0", {}) == ">>> plan-fallbacks[obj 0]: 0"


def test_the_total_is_the_sum_and_the_breakdown_is_sorted():
    got = _fmt("obj 3", {"ik-miss": 2, "aim-miss": 1})
    assert got == ">>> plan-fallbacks[obj 3]: 3 (aim-miss=1, ik-miss=2)"


def test_the_regex_verify_place_uses_reads_the_total():
    m = LINE.search(_fmt("obj 3", {"ik-miss": 2, "aim-miss": 1}))
    assert m and int(m.group(2)) == 3


def test_a_zero_line_is_still_matched_so_absence_and_zero_are_distinguishable():
    m = LINE.search(_fmt("obj 0", {}))
    assert m and int(m.group(2)) == 0


# Every silent-degrade site, tag by tag. A site that degrades without incrementing is invisible,
# so this registry is the tripwire: adding, removing or renaming one fails here until updated.
from verify_place import FALLBACK_TAGS


def _py_files():
    """play_isaac.py + every module under morph/ -- the tree the tags and the cycle flag live in."""
    files = [os.path.join(ROOT, "play_isaac.py")]
    for root, _, fs in os.walk(os.path.join(ROOT, "morph")):
        if "__pycache__" in root:
            continue
        files += [os.path.join(root, f) for f in fs if f.endswith(".py")]
    return files


def _fallback_sites():
    """(file, line, tag) for every `self._fallback(...)` in the tree, by AST not by string.

    A file the parser cannot read is the defect this guard exists to catch, so it fails here
    rather than being skipped."""
    sites = []
    for path in _py_files():
        src = open(path).read()
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            raise AssertionError(f"{path}:{e.lineno} will not parse on this venv: {e.msg}")
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_fallback":
                arg = n.args[0] if n.args else None
                assert isinstance(arg, ast.Constant) and isinstance(arg.value, str), (
                    f"{path}:{n.lineno} tags the fallback with a computed value; tags must be "
                    f"string literals so they stay greppable")
                sites.append((path, n.lineno, arg.value))
    return sites


def test_every_fallback_site_has_a_registered_tag():
    sites = _fallback_sites()
    found = {tag for _, _, tag in sites}
    reg = set(FALLBACK_TAGS.keys())
    assert found == reg, (
        f"unregistered: {sorted(found - reg)}; missing: {sorted(reg - found)}")
    for t, v in FALLBACK_TAGS.items():
        assert isinstance(v, tuple) and len(v) == 2, f"tag {t} missing class or justification"
        cls, just = v
        assert cls in ("SAFETY", "OUTCOME"), f"tag {t} has invalid class '{cls}'"
        assert just, f"tag {t} missing justification" 


def test_no_two_sites_share_a_tag():
    """Two sites on one tag merge into a single count, so a degrade hides behind its twin."""
    sites = _fallback_sites()
    tags = [tag for _, _, tag in sites]
    dupes = {t for t in tags if tags.count(t) > 1}
    assert not dupes, f"tags used at more than one site: {sorted(dupes)}"


def _func(rel, name):
    """The named FunctionDef in a source file under ROOT, by AST. ROOT is reassignable so a decoy
    tree can be substituted for the tracked repo without ever editing the tracked files."""
    tree = ast.parse(open(os.path.join(ROOT, *rel)).read())
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def _param_names(fn):
    """Every name `fn` (a FunctionDef) binds as a parameter -- positional, keyword-only, *args,
    **kwargs -- so a deleted keyword genuinely cannot hide as one of the others."""
    a = fn.args
    names = {p.arg for p in a.posonlyargs + a.args + a.kwonlyargs}
    if a.vararg:
        names.add(a.vararg.arg)
    if a.kwarg:
        names.add(a.kwarg.arg)
    return names


def _is_is_none(test, name):
    """`<name> is None` exactly -- the anchor these selectors hang on."""
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
            and test.left.id == name and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Is) and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None)


def _blocks(node):
    """Every statement list under `node` -- lets a selector ask where a statement actually SITS,
    rather than whether its shape exists somewhere in the subtree."""
    for n in ast.walk(node):
        for field in ("body", "orelse", "finalbody"):
            b = getattr(n, field, None)
            if isinstance(b, list):
                yield b


def _reason_in(test, values):
    """`self._plan_to_config_reason in (<values>)`, POSITIVELY. `!=` / `not in` name the same
    constants while admitting the opposite set -- the inverted-guard decoy this pins out."""
    return (isinstance(test, ast.Compare) and len(test.ops) == 1
            and isinstance(test.ops[0], ast.In)
            and isinstance(test.left, ast.Attribute) and test.left.attr == "_plan_to_config_reason"
            and len(test.comparators) == 1
            and {c.value for c in ast.walk(test.comparators[0])
                 if isinstance(c, ast.Constant)} == values)


def _sets_st_alive_false(node):
    """`st.alive = False` -- the one signal `pick()` ends an attempt on. Tuple form too: the stages
    write `st.alive, st.grip = ...`, so a target-only match would miss the real assignment."""
    for n in ast.walk(node):
        if not isinstance(n, ast.Assign):
            continue
        for t in n.targets:
            pairs = (list(zip(t.elts, n.value.elts))
                     if isinstance(t, ast.Tuple) and isinstance(n.value, ast.Tuple)
                     else [(t, n.value)])
            if any(isinstance(tg, ast.Attribute) and tg.attr == "alive"
                   and isinstance(tg.value, ast.Name) and tg.value.id == "st"
                   and isinstance(v, ast.Constant) and v.value is False for tg, v in pairs):
                return True
    return False


def _calls(node, attr):
    """Any `self.<attr>(...)` anywhere under `node` -- the motion a guarded branch must not make."""
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == attr for n in ast.walk(node))


def _aborts(node, rel, depth=2):
    """`st.alive = False` reached from `node`, directly or through up to `depth` `self.<helper>()`
    hops -- the abort paths delegate because the tag registry forbids two `_fallback` calls on one
    tag, so a selector that only looks at the branch body would miss the real assignment."""
    if _sets_st_alive_false(node):
        return True
    if depth <= 0:
        return False
    for x in ast.walk(node):
        if (isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)
                and isinstance(x.func.value, ast.Name) and x.func.value.id == "self"):
            try:
                if _aborts(_func(rel, x.func.attr), rel, depth - 1):
                    return True
            except StopIteration:
                pass
    return False


def _flag_guard(test, owner="self"):
    """`<owner>._safety_abort` referenced POSITIVELY -- bare, or as a direct `and` operand. `not`
    and `==`/`is` comparisons name the same flag while admitting exactly the case it must exclude.
    `owner` is "self" inside the pick path and "demo" in `main()`'s GUI loop."""
    ops = test.values if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And) else [test]
    return any(isinstance(o, ast.Attribute) and o.attr == "_safety_abort"
               and isinstance(o.value, ast.Name) and o.value.id == owner for o in ops)


def test_the_replan_failed_branch_returns_none_instead_of_falling_through():
    """The defect: `new is None` printed 'continuing' and fell through to execute the exact
    waypoints the checkpoint just proved unsafe. A count with no `return` regresses right back
    into that fall-through, and a `return <joint vector>` is indistinguishable from a completed
    path -- so the tripwire is `return None`, the explicit abort signal callers branch on. The
    selector is anchored to the unique `if new is None:` itself -- not "any If whose body
    carries the tag" -- so a decoy that relocates the tag+return into an unrelated `if False:`
    block elsewhere in the function, while restoring the unsafe real branch, cannot borrow
    this test's pass. It also ARMS the cycle-scoped `_safety_abort` flag: the only site that can,
    and without it every downstream guard is inert."""
    func = _func(("morph", "arm", "execute.py"), "execute_path")
    candidates = [n for n in ast.walk(func) if isinstance(n, ast.If) and _is_is_none(n.test, "new")]
    assert len(candidates) == 1, (
        f"expected exactly one `if new is None:` in execute_path, found {len(candidates)}")
    target = candidates[0]

    # Direct body statements only: walking the whole subtree (the old way) lets a decoy `If`
    # anywhere else in the function borrow this test's pass just by also carrying the tag.
    assert any(
            isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
            and isinstance(s.value.func, ast.Attribute) and s.value.func.attr == "_fallback"
            and s.value.args and isinstance(s.value.args[0], ast.Constant)
            and s.value.args[0].value == "execute-replan-failed"
            for s in ast.walk(target)), (
        "the `if new is None:` branch must directly call _fallback('execute-replan-failed')")
    assert any(isinstance(s, ast.Assign) and isinstance(s.value, ast.Constant)
               and s.value.value is True
               and any(isinstance(t, ast.Attribute) and t.attr == "_safety_abort"
                       and isinstance(t.value, ast.Name) and t.value.id == "demo"
                       for t in s.targets) for s in target.body), (
        "the `if new is None:` branch must ARM `demo._safety_abort` -- without it the flag is "
        "never set, `run_cycle`'s reset guards nothing, and the retry replays the bake again")
    rets = [s for s in target.body if isinstance(s, ast.Return)]
    assert rets, "the execute-replan-failed branch must return, not fall through to the waypoint loop"
    assert all(isinstance(r.value, ast.Constant) and r.value.value is None for r in rets), (
        "the abort must `return None`: a joint vector reads to every caller as a completed path")


def test_the_approach_reports_abort_as_none_not_as_a_fallback():
    """The defect, round 2: `_plan_approach` reported the abort as False -- and False is the
    caller's cue to REPLAY THE BAKE, which ends at the very goal the checkpoint rejected. So the
    contract is tri-state and the abort must be the third value: True arrived, False no plan
    (replay), None aborted (end the pick). Anchored on the name actually bound to the
    `execute_path` call, so a `return None` parked under an unrelated `if False:` cannot
    borrow the pass."""
    func = _func(("morph", "pick", "approach.py"), "_plan_approach")
    calls = [n for n in ast.walk(func) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "move_to_pose"]
    assert len(calls) == 1, (
        f"expected exactly one move_to_pose call in _plan_approach, found {len(calls)} -- a "
        f"second, unbound one would execute the path with its abort thrown away")
    binds = [n for n in ast.walk(func) if isinstance(n, ast.Assign)
             and isinstance(n.value, ast.Call) and getattr(n.value.func, "id", "") == "move_to_pose"]
    assert len(binds) == 1 and len(binds[0].targets) == 1 and isinstance(binds[0].targets[0], ast.Name), (
        "_plan_approach must bind move_to_pose's result to one name; discarding it IS the defect")
    nm = binds[0].targets[0].id
    # The abort arm tests the OUTCOME now, not `is None`: a returned pose was never the question,
    # and the primitive refuses to hand one back on an abort at all.
    guards = [n for n in ast.walk(func) if isinstance(n, ast.If)
              and any(isinstance(a, ast.Attribute) and getattr(a.value, "id", "") == "Outcome"
                      and a.attr in ("REFUSED", "PAYLOAD_LOST") for a in ast.walk(n.test))
              and any(isinstance(x, ast.Name) and x.id == nm for x in ast.walk(n.test))]
    assert len(guards) == 1, (
        f"expected exactly one abort guard on {nm}'s outcome in _plan_approach, found {len(guards)}")
    assert any(isinstance(s_, ast.Return) and isinstance(s_.value, ast.Constant)
               and s_.value.value is None for s_ in guards[0].body), (
        "the abort branch must `return None`: False is the caller's cue to replay the bake, and "
        "the bake ends at the goal the checkpoint just rejected")


def test_execute_path_and_plan_approach_do_not_accept_a_pin_kwarg():
    """`pin` was threaded three levels deep into `execute_path` and never read there -- the
    only "pin" in its body is `_pin_base`, an unrelated attribute. Dead parameter, deleted; this
    is the tripwire against it (or its pass-through in `_plan_approach`) coming back."""
    exec_names = _param_names(_func(("morph", "arm", "execute.py"), "execute_path"))
    plan_names = _param_names(_func(("morph", "pick", "approach.py"), "_plan_approach"))
    assert "pin" not in exec_names, "execute_path still accepts a dead `pin=` parameter"
    assert "pin" not in plan_names, "_plan_approach still accepts a dead `pin=` pass-through"


def test_execute_path_still_holds_fingers_and_base():
    """Tripwire, the other direction: `hold_fingers` (gates `fhold`) and `hold_base` (gates the
    per-step `set_base`) are LIVE, unlike `pin` -- neither may be deleted alongside it."""
    names = _param_names(_func(("morph", "arm", "execute.py"), "execute_path"))
    assert "hold_fingers" in names, "execute_path lost its live `hold_fingers` parameter"
    assert "hold_base" in names, "execute_path lost its live `hold_base` parameter"


def test_the_reach_ends_the_pick_on_an_abort_instead_of_running_on():
    """The defect: `_pick_reach` treated the approach's False as ordinary no-plan and ran on --
    re-commanding the configuration the checkpoint had just proven blocked. Every `_plan_approach`
    call must bind its result, test it `is None`, and that branch must abort the pick and return.
    The call is located by WHERE IT SITS (its own Assign, its own block), so the `if False:` decoy
    -- the whole guarded block parked as dead code beside a restored unguarded call -- cannot
    borrow the pass.

    ONE call now, not two. `_pick_reach` used to branch on `self.clut is not None` and carry a
    second, non-LUT approach. `play_isaac.py` loads the closure LUT with a bare
    `json.load(open(...))`, so `clut` can never BE None and that branch was unreachable; it was
    deleted with the guard. If a second call ever comes back, this number is the tripwire."""
    rel = ("morph", "pick", "reach.py")
    func = _func(rel, "_pick_reach")
    parents = {c: n for n in ast.walk(func) for c in ast.iter_child_nodes(n)}
    calls = [n for n in ast.walk(func) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "_plan_approach"]
    assert len(calls) == 1, (
        f"expected exactly 1 _plan_approach call in _pick_reach (the LUT path is the only one), "
        f"found "
        f"{len(calls)} -- an extra one executes the approach with its abort thrown away")
    for call in calls:
        n = call
        # dead code aborts nothing
        while n in parents:
            p = parents[n]
            assert not (isinstance(p, ast.If) and isinstance(p.test, ast.Constant)
                        and not p.test.value), (
                f"reach.py:{call.lineno}: the _plan_approach block sits under a constant-false "
                f"guard -- it is dead code, not an abort path")
            n = p
        assign = parents.get(call)
        assert (isinstance(assign, ast.Assign) and len(assign.targets) == 1
                and isinstance(assign.targets[0], ast.Name)), (
            f"reach.py:{call.lineno}: _plan_approach's result is discarded, so None (abort) and "
            f"False (no plan) read identically and the bake is replayed into the rejected goal")
        nm = assign.targets[0].id
        block = next(b for b in _blocks(func) if assign in b)
        guards = [s for s in block[block.index(assign):]
                  if isinstance(s, ast.If) and _is_is_none(s.test, nm)]
        assert len(guards) == 1, (
            f"reach.py:{call.lineno}: expected exactly one `if {nm} is None:` in the same block as "
            f"the call, found {len(guards)}")
        body = guards[0]
        assert any(isinstance(s, ast.Return) for s in body.body), (
            f"reach.py:{body.lineno}: the abort branch must return -- falling through runs the "
            f"rest of the reach from an intermediate pose")
        assert _aborts(body, rel), (
            f"reach.py:{body.lineno}: the abort branch must set `st.alive = False` -- that is the "
            f"only signal `pick()` ends an attempt on")
def test_the_latch_is_initialised_once_and_never_cleared_by_the_thing_it_guards():
    """The defect, round 5: `run_cycle` CLEARED the flag at cycle start, so a duplicate DEMO_CYCLES
    entry or a second GUI MOVE re-entered `pick()` with the rejection erased -- the guard erasing
    its own evidence. The latch is initialised exactly once, in `Demo.__init__`, and nothing
    auto-clears it: clearing is a deliberate act (today, restarting the process)."""
    clears = []
    for path in _py_files():
        src = open(path).read()
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            raise AssertionError(f"{path}:{e.lineno} will not parse on this venv: {e.msg}")
        clears += [(path, n) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and isinstance(n.value, ast.Constant) and n.value.value is False
                   and any(isinstance(t, ast.Attribute) and t.attr == "_safety_abort"
                           and isinstance(t.value, ast.Name) and t.value.id == "self"
                           for t in n.targets)]
    assert len(clears) == 1, (
        f"the safety latch must be cleared at exactly ONE site, found {len(clears)}: "
        f"{[f'{os.path.basename(p)}:{n.lineno}' for p, n in clears]} -- every extra site is a "
        f"place the latch is erased by something it was set to stop")
    assert os.path.basename(clears[0][0]) == "play_isaac.py", (
        f"the latch belongs to the RUN: initialising it in {os.path.basename(clears[0][0])} puts "
        f"it inside the pick path it guards")
    init = _func(("play_isaac.py",), "__init__")         # Demo.__init__ is the only one in the file
    assert clears[0][1].lineno in {getattr(n, "lineno", None) for n in ast.walk(init)}, (
        "the one clear site must be `Demo.__init__`: anywhere per-cycle or per-attempt and the "
        "rejection dies with the thing that saw it")
    func = _func(("play_isaac.py",), "run_cycle")
    inside = [n for n in ast.walk(func) if isinstance(n, ast.Assign)
              and isinstance(n.value, ast.Constant) and n.value.value is False
              and any(isinstance(t, ast.Attribute) and t.attr == "_safety_abort"
                      and isinstance(t.value, ast.Name) and t.value.id == "self"
                      for t in n.targets)]
    assert not inside, (
        f"run_cycle clears the latch at line(s) {[n.lineno for n in inside]} -- a latch cleared by "
        f"the thing it guards stops exactly one cycle and lets the next one re-command the "
        f"rejected configuration")
    picks = sorted(n.lineno for n in ast.walk(func) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and n.func.attr == "pick")
    assert len(picks) == 2, (
        f"run_cycle must still make the attempt+retry pair this latch exists for, found "
        f"{len(picks)} pick calls")


from morph.arm.api import Outcome as _Outcome, move_to_pose as _move_to_pose  # noqa: E402
# api's executor seam, routed to the fake under test (the real one needs a robot). Module-wide:
# the gate runs one process per suite.
import morph.arm.api as _api_mod                                                    # noqa: E402
_api_mod.execute_path = lambda demo, *a, **k: demo._execute_arm_path(*a, **k)
_api_mod.commanded_fingers = lambda demo: demo._commanded_fingers()
_api_mod.plan_arm = lambda demo, *a, **k: demo._plan_arm(*a, **k)
_api_mod.goal_set = lambda demo, *a, **k: demo._goal_set(*a, **k)


def _exec_method(rel, name, glb):
    """Compile ONE method out of a source file and bind it into `glb`. play_isaac.py imports Isaac
    at module level, so this is the only way to actually RUN `run_cycle` offline -- and running it
    is the point: every static selector in this file describes a shape, and four rounds of shapes
    were each satisfied by code that still retried."""
    path = os.path.join(ROOT, *rel)
    fn = _func(rel, name)
    assert not fn.decorator_list, f"{name} grew a decorator, which this harness would drop"
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), path, "exec"),
         glb)
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


class _PlanArmDemo:
    """The `Demo` surface `_plan_arm` touches, and nothing else. The geometry is arbitrary: the
    planner subprocess is stubbed, so nothing downstream of it is ever reached."""

    # Off drives unless a test says otherwise: measured == commanded, `_plan_start` seeds measured.
    idx = {n: i for i, n in enumerate(_ARM_MODEL.Q8)}

    def _drive_on(self):
        return False

    def __init__(self):
        self._plan_fallbacks = {}
        self.blamed = 0

    # The deck box among them, at the prim the chassis pad names: `_plan_arm` pads that box and RAISES
    # if it matches nothing, so a harness without it exercises the raise rather than the planner.
    obstacles = [((np.zeros(3), np.ones(3)), "world"),
                 ((np.zeros(3) - 5.0, np.ones(3) - 5.0), "chassis")]
    obstacle_paths = ["/World/elsewhere/Box",
                      __import__("morph.arm.collision", fromlist=["x"]).CHASSIS_PAD_PRIM]

    def _arm_obstacles(self, exclude_names=(), *, diagnostic_paths=None):
        if diagnostic_paths is not None:
            diagnostic_paths.extend(self.obstacle_paths)
        return self.obstacles

    def _arm_base_world(self):
        return np.zeros(3), np.eye(3)

    def _arm_bounds(self):
        return -np.ones(8), np.ones(8)

    def _arm_q8(self):
        return np.zeros(8)

    def _arm_blame(self, *a, **k):
        self.blamed += 1
        self.blamed_what = (np.asarray(a[0], float).copy(), a[4])
        self.blamed_obs, self.blamed_paths = a[1], list(a[2])


from morph.arm.obstacles import chassis_pad as _chassis_pad_real  # noqa: E402

# The REAL producer+pad on the planner harness too: `_plan_arm` builds its set through
# `self._planned_obstacles`, shared with the executor's checkpoint so the two cannot drift.
_PlanArmDemo._planned_obstacles = _exec_method(
    ("morph", "arm", "obstacles.py"), "planned_obstacles", {"chassis_pad": _chassis_pad_real})

def _const(rel, name):
    """A module-level string constant, by AST. Missing means the constant was renamed, which is the
    defect this exists to catch."""
    tree = ast.parse(open(os.path.join(ROOT, *rel)).read())
    return next(n.value.value for n in tree.body
                if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
                and any(getattr(t, "id", None) == name for t in n.targets))


import morph.arm.collision as _coll                                       # noqa: E402


def _run_plan_arm(run, goals=3, obstacles=None, paths=None):
    """The REAL `_plan_arm` with `run_child` replaced by `run`; (demo, result, printed lines).

    `run_child` is the boundary to the planner worker: everything it can report -- a path, an
    error result, a stderr degrade, a timeout as None -- has to leave `_plan_arm` as a counted
    outcome rather than an exception only `run_cycle`'s broad `except` stops."""
    out = []
    glb = {"np": np, "os": os, "json": json, "run_child": run,
           "VENV_PY": "/nonexistent/python3", "ARM_PLAN": "/nonexistent/arm_plan.py",
           "ARM_MODEL": "model.json", "CLOSURE_LUT": "lut.json", "MARGIN": 0.02,
           "PLAN_TIMEOUT": 120,
           "TUNNEL_DEGRADE": _const(("morph", "arm", "plan.py"), "TUNNEL_DEGRADE"),
           "EVENT_LINE": _const(("morph", "arm", "plan.py"), "EVENT_LINE"),
           "time": __import__("time"),
           "plan_start": lambda demo: demo._plan_start(),
           # The REAL constants: stubbing them would let `plan_arm` pass its own whitelist against
           # a table this harness invented.
           "planned_obstacles": lambda demo, *a, **k: demo._planned_obstacles(*a, **k),
           "arm_blame": lambda demo, *a, **k: demo._arm_blame(*a, **k),
           "CHASSIS_PAD_M": _coll.CHASSIS_PAD_M,
           "CHASSIS_PAD_LINKS": _coll.CHASSIS_PAD_LINKS,
           "print": lambda *a, **k: out.append(" ".join(str(x) for x in a))}
    # The child contract, the real thing: only the launch itself is doubled.
    for _fn in ("try_budgets", "child_result"):
        _exec_method(("morph", "arm", "plan.py"), _fn, glb)
    _PlanArmDemo._plan_arm = _exec_method(("morph", "arm", "plan.py"), "plan_arm", glb)
    _PlanArmDemo._plan_start = _exec_method(("morph", "arm", "plan.py"), "plan_start",
                                            {"np": np, "ArmModel": _ARM_MODEL, "print": glb.get("print", print),
                                             "Q8_TOL_M": 0.005, "Q8_TOL_RAD": 0.02})
    _PlanArmDemo._fallback = _exec_method(("morph", "arm", "surface.py"), "_fallback", {})
    demo = _PlanArmDemo()
    if obstacles is not None:
        demo.obstacles = obstacles
    if paths is not None:
        demo.obstacle_paths = paths
    return demo, demo._plan_arm(np.zeros((goals, 8))), out


def test_the_plan_arm_harness_reaches_the_subprocess_and_returns_its_path():
    """The control. Without it the timeout test below would pass just as well against a `_plan_arm`
    that never got as far as launching anything."""
    def _ok(*a, **k):
        return SimpleNamespace(stdout=json.dumps([[0.0] * 8, [0.1] * 8]), stderr="")
    demo, got, _out = _run_plan_arm(_ok)
    assert got is not None and len(got) == 2, f"expected the stub's 2 waypoints, got {got}"
    assert not demo._plan_fallbacks, f"a successful plan counted a degrade: {demo._plan_fallbacks}"


def test_an_annotated_obstacle_is_a_counted_refusal_at_the_planner_not_a_raised_exception():
    """`_plan_arm` serialises obstacles as `for ab, tg in obs` -- OUTSIDE its own `try`, whose only
    handler is `TimeoutExpired`. A per-obstacle allowance is a THREE-element tuple, so the day one
    reaches this call the unpack raises and `run_cycle`'s broad except kills the cycle.

    The planned tier must never carry an exemption at all (A4), so the right answer is not to
    serialise it -- it is to refuse it, counted, the way every other planner failure is counted."""
    def _never(*a, **k):
        raise AssertionError("the subprocess must not be launched with an annotated obstacle")

    # The deck box FIRST so `chassis_pad` finds its prim, then the caller's 3-tuple. Without the
    # deck the pad raises and this would test that raise instead of the A4 refusal below.
    demo, got, out = _run_plan_arm(
        _never, obstacles=[((np.zeros(3) - 5.0, np.ones(3) - 5.0), "chassis"),
                           ((np.zeros(3), np.ones(3)), "world", ("Arm_Left_1",))],
        paths=[_coll.CHASSIS_PAD_PRIM, "/World/elsewhere/Box"])
    assert got is None, f"an annotated obstacle must read as no-plan, got {got}"
    assert demo._plan_fallbacks == {"plan-bad-obstacle": 1}, (
        f"the refusal was not counted as one tagged degrade: {demo._plan_fallbacks}")
    assert "exemption" in " ".join(out), f"the refusal was counted but not explained: {out}"


def test_a_planner_worker_timeout_is_a_counted_no_plan_not_a_raised_exception():
    """A plan past PLAN_TIMEOUT reaches `_plan_arm` as None and must leave it as one counted
    `plan-timeout` and a no-plan, not as an exception only `run_cycle`'s broad `except` catches.
    The kill and the respawn behind that None are pinned in `tests/test_plan_worker.py`."""
    def _timeout(*a, **k):
        return None
    demo, got, out = _run_plan_arm(_timeout)
    assert got is None, f"a timed-out plan must read as no-plan, got {got}"
    assert demo._plan_fallbacks == {"plan-timeout": 1}, (
        f"the timeout was not counted as one tagged degrade: {demo._plan_fallbacks}")
    line = " ".join(out)
    assert "TIMED OUT" in line, f"the timeout was counted but not logged: {out}"
    assert "3 goal candidate" in line and "3.0s" in line, (
        f"the log names neither the candidates nor the budget that ran out: {out}")


def test_a_non_finite_start_is_a_counted_refusal_not_a_request_to_the_planner():
    """The child would validate a NaN start with a checker that skips every box for it."""
    calls = []
    def _run(*a, **k):
        calls.append(1)
        return SimpleNamespace(stdout=json.dumps([[0.0] * 8, [0.1] * 8]), stderr="")
    saved = _PlanArmDemo._arm_q8
    _PlanArmDemo._arm_q8 = lambda self: np.array([0.0, np.nan] + [0.0] * 6)
    try:
        demo, got, out = _run_plan_arm(_run)
    finally:
        _PlanArmDemo._arm_q8 = saved
    assert got is None and not calls, f"the planner was asked from a NaN start: {got} {calls}"
    assert demo._plan_fallbacks == {"plan-start-not-finite": 1}, demo._plan_fallbacks
    assert any("not finite" in line for line in out), out



# arm_plan.py runs in a separate interpreter and degrades by WRITING TO STDERR. `_plan_arm` must
# read that on the SUCCESS path too -- a degrade nobody can see is how a broken run looks green.
_CHILD_STDERR = ("[arm_plan] 2 of 3 goal candidates invalid, planning to the remaining 1\n"
                 "[arm_plan] 41 -> 6 states, 213 waypoints\n")
_CHILD_CAPPED = ("[arm_plan] 1 of 212 emitted segments can tunnel a thin obstacle; worst "
                 "Gripper_Link1_1 at 2.40x its own limit (depth cap 6)\n")


def test_an_invalid_first_try_is_blamed_once_and_never_planned_on():
    """The child refusing the start or goal as invalid is the one failure the operator can act on;
    `arm_blame` names the link and the box, on the first try only, and the plan reads as none."""
    def _run(*a, **k):
        return SimpleNamespace(stdout=json.dumps({"error": "start state invalid"}), stderr="")
    demo, got, out = _run_plan_arm(_run)
    assert got is None, f"an invalid start produced a path: {got}"
    assert demo.blamed == 1, f"blamed {demo.blamed} times; the first try, and only the first, is blamed"
    q, what = demo.blamed_what
    assert what == "start" and np.allclose(q, demo._arm_q8()), "the START was invalid; the blame names something else"
    # The prim list `arm_blame` names its obstacle from, index-aligned with the set it was given.
    # Handed the wrong list it reports the wrong prim and sends the reader after other geometry.
    _obs, _paths = demo._planned_obstacles(())
    assert demo.blamed_paths == list(_paths), (
        f"blame got {len(demo.blamed_paths)} prim paths for {len(demo.blamed_obs)} obstacles, "
        f"and not the producer's own order")
    assert any("start state invalid" in line for line in out), out


def _stub(stderr):
    def _run(*a, **k):
        return SimpleNamespace(stdout=json.dumps([[0.0] * 8, [0.1] * 8]), stderr=stderr)
    return _run


def test_the_planner_childs_stderr_reaches_the_log_on_the_success_path():
    """Every line, not just the parse-failure tail: these are the only report the child can make."""
    demo, got, out = _run_plan_arm(_stub(_CHILD_STDERR))
    assert got is not None and len(got) == 2, f"the stub's path was lost: {got}"
    line = " ".join(out)
    for want in ("2 of 3 goal candidates invalid", "41 -> 6 states"):
        assert want in line, f"the child's stderr never reached the log: {out}"
    assert not demo._plan_fallbacks, (
        f"an ordinary stderr line was counted as a degrade: {demo._plan_fallbacks}")


def test_a_depth_capped_densify_is_counted_and_not_merely_relayed():
    """A capped segment is an UNDER-REFINED emitted path: the executor interpolates through an
    interval nothing checked, which must be prevented. Visible is not enough
    -- it has to fail the run, so it goes through the tally verify_place.py gates on."""
    demo, got, out = _run_plan_arm(_stub(_CHILD_STDERR + _CHILD_CAPPED))
    assert got is not None and len(got) == 2, (
        f"a capped path must still be RETURNED -- withholding it replays the unchecked bake: {got}")
    assert demo._plan_fallbacks == {"densify-depth-cap": 1}, (
        f"the depth cap was not counted as one tagged degrade: {demo._plan_fallbacks}")
    assert any("can tunnel" in ln for ln in out), f"counted but not logged: {out}"


class _SuppressedCycle:
    """The `Demo` surface `run_cycle` touches, and nothing else. Attempt 1 arms the run-scoped
    latch `execute_path` arms on a checkpoint abort, then fails. Attempt 2 is the
    PLANNER-SUCCESS route: `_plan_approach` finds a path, EXECUTES it and returns True, so both
    `if not _appr and self._safety_abort` guards in reach.py are False and every level below
    `run_cycle` lets the rejected endpoint be re-commanded. Reaching it at all is the failure."""

    def __init__(self, out):
        self.attempts = 0
        self.placed = 0
        self.opened = 0
        self.out = out
        self._plan_fallbacks = {}
        # Demo.__init__ owns this: the ONE place it is cleared
        self._safety_abort = False
        self.world = SimpleNamespace(step=lambda *a, **k: None)

    def pick(self, obj_idx, status=None):
        self.attempts += 1
        if self.attempts == 1:
            # execute.py: a checkpoint rejected a configuration
            self._safety_abort = True
            return False
        # ...and the retry planned, executed and arrived
        return True

    def place(self, obj_idx, slot_idx, status=None):
        self.placed += 1
        # morph/place prints exactly one of these per placed cycle; results_gate requires the
        # count to equal the record count, which is why a refused cycle may stamp neither.
        self.out.append(f">>> PLACE obj {obj_idx} -> slot {slot_idx} ok=True -> precise_success")

    def _recover(self):
        pass

    def _open_place_result(self, *a, **k):
        self.opened += 1

    def _write_place_result(self, *a, **k):
        pass

    def pin_chassis(self, *a, **k):
        pass


def _run_suppressed_cycle(cycles=1):
    """`cycles` runs of the REAL `run_cycle` on one demo, the first of which hits a checkpoint
    safety abort. Returns (demo, printed lines, per-cycle return values).

    The cycle-entry instrumentation is left to raise into its own `try/except` -- that is what it
    does in production against a stub articulation -- but the pick/retry block is NOT allowed to,
    because a swallowed exception there would make every assertion below pass for the wrong
    reason."""
    out = []
    glb = {"os": os, "print": lambda *a, **k: out.append(" ".join(str(x) for x in a))}
    _SuppressedCycle.run_cycle = _exec_method(("play_isaac.py",), "run_cycle", glb)
    _SuppressedCycle._fallback_tally = _exec_method(("play_isaac.py",), "_fallback_tally", glb)
    _SuppressedCycle._fallback = _exec_method(("morph", "arm", "surface.py"), "_fallback", {})
    keep = os.environ.get("PICK_RETRY")
    # the retry must be ARMED, or this proves nothing
    os.environ["PICK_RETRY"] = "1"
    try:
        demo = _SuppressedCycle(out)
        rets = [demo.run_cycle(0, 0) for _ in range(cycles)]
    finally:
        if keep is None:
            os.environ.pop("PICK_RETRY", None)
        else:
            os.environ["PICK_RETRY"] = keep
    assert not any(l.startswith("!! cycle failed") for l in out), (
        f"run_cycle swallowed an exception out of the pick/retry block, so nothing below is "
        f"evidence: {out}")
    return demo, out, rets


def test_a_safety_abort_stops_the_retry_even_when_the_retrys_planner_would_succeed():
    """The defect, round 5: `run_cycle` recovers and retries on ANY False. reach.py's two guards
    are `if not _appr and self._safety_abort`, which is the NO-PLAN route only -- when the retry's
    planner FINDS a path, `_plan_approach` has already executed it by the time it returns True and
    the rejected endpoint has been re-commanded. `run_cycle` owns the decision to try again, so the
    rejection has to stop it there; a guard anywhere inside `pick()` is a guard the retry has
    already paid for."""
    demo, out, _ = _run_suppressed_cycle()
    assert demo.attempts == 1, (
        f"run_cycle made {demo.attempts} pick attempts after a checkpoint safety abort -- the "
        f"retry is identical, so it re-commands the configuration the checkpoint rejected")
    assert not demo.placed, "nothing was held, so the suppressed cycle must not place"
    reasons = [l for l in out if "retry" in l.lower() and "abort" in l.lower()
               and not LINE.search(l)]
    assert reasons, (
        f"the suppression printed no reason of its own -- a cycle that silently makes one attempt "
        f"instead of two is indistinguishable from one that never needed a retry: {out}")


def test_the_cycle_still_prints_its_tally_when_the_retry_was_suppressed():
    """`_fallback_tally(obj_idx, attempt=1)` is called INSIDE the retry branch, so suppressing the
    retry must not take the cycle's tally line with it: verify_place.py treats a graded log with no
    tally line ANYWHERE as pre-instrument and stops gating it. Exactly one line (the attempt-1 line belongs
    to a retry that no longer happens), and the suppression is counted into it -- an uncounted skip
    is invisible to the gate that reads this line."""
    demo, out, _ = _run_suppressed_cycle()
    assert demo.attempts == 1, "precondition: the retry must have been suppressed"
    tally = [l for l in out if LINE.search(l)]
    assert len(tally) == 1, (
        f"expected exactly one plan-fallbacks line for a single-attempt cycle, found "
        f"{len(tally)}: {tally}")
    m = LINE.search(tally[0])
    assert m.group(1) == "obj 0" and int(m.group(2)) >= 1, (
        f"the cycle-end tally must be printed and must carry the suppression: {tally[0]}")
    assert "retry-suppressed-after-abort=1" in tally[0], (
        f"the suppressed retry is not counted under its own tag, so the log cannot tell a "
        f"suppression from a clean cycle: {tally[0]}")


def test_a_second_run_cycle_after_an_abort_never_reaches_pick():
    """`run_cycle` must not clear the latch at cycle start: cycle 2 -- a
    duplicate DEMO_CYCLES entry, or a second GUI MOVE -- must not re-enter `pick()` with the rejection
    erased and place the object. Driven twice, the real `run_cycle` must make exactly the one
    attempt of cycle 1 and refuse cycle 2 outright, with an outcome its schedulers can test."""
    demo, out, rets = _run_suppressed_cycle(cycles=2)
    assert demo.attempts == 1, (
        f"{demo.attempts} pick attempts across two cycles after a checkpoint safety abort -- the "
        f"latch was cleared by the very thing it guards, and the rejected configuration is "
        f"commanded again")
    assert not demo.placed, "nothing was ever held, so no cycle may place"
    assert rets[0] is False and rets[1] is False, (
        f"run_cycle must return an explicit falsy outcome the schedulers can test, got {rets}")
    refusals = [l for l in out if "REFUS" in l.upper() and "SAFETY" in l.upper()]
    assert refusals, f"the refused cycle printed no reason of its own: {out}"


def test_a_refused_cycle_stamps_no_record_and_prints_no_grade_line():
    """`results_gate` requires len(records) == len(grade lines). A refusal placed AFTER
    `_open_place_result` stamps an attempt record for a cycle that then prints no grade line, so
    the gate fails every safety-aborted run for the wrong reason -- and a refusal that still
    reaches `place()` grades a cycle that never ran."""
    import verify_place
    demo, out, _ = _run_suppressed_cycle(cycles=2)
    assert demo.opened == 1, (
        f"{demo.opened} attempt records stamped across one run cycle and one refused cycle -- the "
        f"refusal must return BEFORE _open_place_result")
    grades = [l for l in out if verify_place.GRADE_LINE.search(l)]
    assert len(grades) == demo.opened - 1, (
        f"{len(grades)} grade line(s) for {demo.opened} record(s): the aborted cycle owns a record "
        f"and no grade line, and the refused cycle must own neither: {out}")


def test_both_schedulers_stop_once_the_latch_is_set():
    """The defect, round 6: `main()` scheduled every remaining DEMO_CYCLES entry regardless, and
    the GUI loop ran whatever MOVE put in `state["request"]`. The headless loop must TEST
    run_cycle's outcome and `break`; the GUI branch must sit in the else of a `_safety_abort`
    test, so the click is visibly refused rather than silently obeyed."""
    func = _func(("play_isaac.py",), "main")
    parents = {c: n for n in ast.walk(func) for c in ast.iter_child_nodes(n)}
    calls = [n for n in ast.walk(func) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "run_cycle"]
    assert len(calls) == 2, (
        f"expected the headless and GUI run_cycle calls in main(), found {len(calls)} -- an "
        f"unenumerated third scheduler is exactly how the last five rounds were bypassed")

    def _ancestors(n):
        while n in parents:
            n = parents[n]
            yield n

    tested = [c for c in calls if any(isinstance(a, ast.If) and c in set(ast.walk(a.test))
                                      and any(isinstance(x, ast.Break) for x in ast.walk(a))
                                      for a in _ancestors(c))]
    assert len(tested) == 1, (
        "the headless loop must branch on run_cycle's own outcome and break out of the remaining "
        "cycles; scheduling them anyway re-enters the pick with a rejection outstanding")
    guarded = [c for c in calls if any(isinstance(a, ast.If) and _flag_guard(a.test, "demo")
                                       and c in set(ast.walk(ast.Module(body=a.orelse,
                                                                        type_ignores=[])))
                                       for a in _ancestors(c))]
    assert len(guarded) == 1, (
        "the GUI's run_cycle must sit in the else of a positive `_safety_abort` test -- a MOVE "
        "click after a latched abort is a second cycle by another name")
    assert tested[0] is not guarded[0], "one scheduler is doing both jobs; the other is unguarded"
    brk = next(a for a in _ancestors(tested[0])
               if isinstance(a, ast.If) and any(isinstance(x, ast.Break) for x in ast.walk(a)))
    assert any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "print"
               for x in ast.walk(brk)), (
        "the headless stop must print which remaining cycles it skipped -- a run that silently "
        "ends early is indistinguishable from one that finished")


# ---------------------------------------------------------------------------------------------
# The headed GUI menu loop and tools/idle_probe.py, driven for real: both import Isaac at module
# scope, so EXECUTING the code is the only way to prove what they do. The AST shape says less.


class _GuiDemo:
    """The `Demo` surface `main()`'s headed menu loop touches. `run_cycle` latches on its single
    call exactly as a checkpoint abort does, then returns the falsy outcome. Every other attribute
    answers with a recorder, so an arm command that outlives the latch names ITSELF in the log
    instead of having to be anticipated by a selector."""

    def __init__(self, calls, jog=None, latch=True):
        self.calls = calls
        # not None -> the loop takes the `_jog_tick` branch
        self.jog = jog
        self.dt = 1.0 / 240.0
        self._latch = latch
        # Demo.__init__ owns this: the ONE place it is cleared
        self._safety_abort = False

    def run_cycle(self, obj_idx, slot_idx, status=None):
        self.calls.append("run_cycle")
        self._safety_abort = self._latch
        return not self._latch

    def __getattr__(self, name):
        def rec(*a, **k):
            self.calls.append(name)
        return rec


def _run_gui(jog=None, latch=True, ticks=8):
    """The REAL `play_isaac.main()` headed branch over a stub. HEADLESS is False so the menu loop
    is the branch taken; `build_menu` queues one MOVE, so iteration 1 runs a cycle; `is_running()`
    runs down so a loop that never halts terminates instead of hanging the suite. Returns (call
    log, printed lines)."""
    calls, out = [], []
    demo = _GuiDemo(calls, jog=jog, latch=latch)

    class _App:
        def __init__(self):
            self.ticks = ticks

        def is_running(self):
            self.ticks -= 1
            return self.ticks > 0

    glb = {"os": os, "HEADLESS": False, "Demo": lambda: demo, "simulation_app": _App(),
           "build_menu": lambda d, s: s.__setitem__("request", (0, 0)),
           "print": lambda *a, **k: out.append(" ".join(str(x) for x in a))}
    _exec_method(("play_isaac.py",), "main", glb)()
    return calls, out


def test_the_gui_menu_loop_stops_commanding_the_arm_once_a_checkpoint_rejects():
    """The MOVE click is refused, but the loop must not fall through later iterations
    to `demo._jog_tick()` or `demo.idle_step()`. Both command the arm -- the jog servo writes a
    cartesian target, idle_step writes q0 through `_force`/`_apply` -- so a run whose checkpoint
    rejected a configuration must not go on driving from it until the window is closed. Driven for real
    on BOTH branches: once the latch is set the loop itself stops, and says why once."""
    for jog in (None, object()):
        calls, out = _run_gui(jog=jog)
        assert calls.count("run_cycle") == 1, f"precondition: exactly one cycle ran: {calls}"
        after = calls[calls.index("run_cycle") + 1:]
        assert not [c for c in after if c in ("idle_step", "_jog_tick")], (
            f"{after} ran after a checkpoint SAFETY ABORT latched -- both the jog servo and the "
            f"idle step command the arm from the configuration the checkpoint rejected")
        halt = [l for l in out if "ABORT" in l.upper() and "STOP" in l.upper()]
        assert len(halt) == 1, (
            f"the menu loop must say why it stopped, exactly once -- a window that freezes in "
            f"silence is indistinguishable from a hung sim: {out}")


def test_the_gui_menu_loop_control_reaches_both_arm_writers_when_nothing_is_latched():
    """The positive control for the two absence assertions above. With no abort the loop must
    actually REACH `idle_step` and `_jog_tick`; without this, a renamed branch or a harness that
    never enters the loop would satisfy the halt test for the wrong reason."""
    idle, _ = _run_gui(jog=None, latch=False)
    assert "idle_step" in idle, f"the control never reached the idle branch: {idle}"
    jog, _ = _run_gui(jog=object(), latch=False)
    assert "_jog_tick" in jog, f"the control never reached the jog branch: {jog}"


def _dict_key(node, name):
    """The value bound to `name` in a dict literal, or None."""
    if not isinstance(node, ast.Dict):
        return None
    return next((v for k, v in zip(node.keys, node.values)
                 if isinstance(k, ast.Constant) and k.value == name), None)


# ---------------------------------------------------------------------------------------------
# The bench's OUTER variant scheduler, driven for real: an AST shape cannot see a scheduler that
# starts the next variant after a checkpoint rejection.

# Everything in `attempt()` past the abort guard that commands the arm, plus the scheduler's two
# arm writers. BY NAME against a call log, cross-checked against the AST so a rename cannot pass.
def test_no_pick_stage_runs_once_a_stage_has_aborted():
    """The pick sequence must check `st.alive` before every stage: a reach that
    aborted at a checkpoint must not still run the settle -- whose `_hold` force-writes the goal every step
    -- and then the close. Every stage from the settle on must sit under a POSITIVE `if st.alive:`,
    and the existing `if not st.alive:` exit must still `return False`, because that False is what
    the grasp candidate walk (and `run_cycle`'s recover+retry behind it) advances on.

    The sequence is `_pick_attempt`, not `pick`: `pick` walks candidates OVER it, so the guard
    belongs to the pass, and a walk of four unguarded passes is four times the defect."""
    func = _func(("morph", "pick", "__init__.py"), "_pick_attempt")
    parents = {c: n for n in ast.walk(func) for c in ast.iter_child_nodes(n)}
    for stage in ("_pick_settle", "_pick_descend", "_pick_close"):
        call = next((n for n in ast.walk(func) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == stage), None)
        assert call is not None, (
            f"no {stage} call in _pick_attempt() -- the selector has lost its target")
        n, guarded = call, False
        while n in parents:
            p = parents[n]
            if (isinstance(p, ast.If) and n in p.body and isinstance(p.test, ast.Attribute)
                    and p.test.attr == "alive" and isinstance(p.test.value, ast.Name)
                    and p.test.value.id == "st"):
                guarded = True
                break
            n = p
        assert guarded, (
            f"_pick_attempt() runs {stage} with no `if st.alive:` above it -- after an abort that "
            f"stage works from a pose nothing validated")
    exits = [n for n in ast.walk(func) if isinstance(n, ast.If)
             and isinstance(n.test, ast.UnaryOp) and isinstance(n.test.op, ast.Not)
             and isinstance(n.test.operand, ast.Attribute) and n.test.operand.attr == "alive"]
    assert any(isinstance(s, ast.Return) and isinstance(s.value, ast.Constant)
               and s.value.value is False for e in exits for s in e.body), (
        "_pick_attempt() must still `return False` on `if not st.alive:` -- the candidate walk, "
        "and run_cycle's recover+retry behind it, both hang off that False")


class _Cut(Exception):
    """Raised by the settle fake once the decision under test has been made."""


def test_the_settle_snap_is_bounded_by_the_residue_it_claims_to_be_closing():
    """`SETTLE_PLAN` and `_plan_to_config` are deleted (29fa282), so the snap below is reached on
    EVERY cycle with no counter at all. Its own comment calls it "the legitimate no-op it
    has always been" because the residue is ~0 after a normal reach -- which is true right up until
    the reach falls back or aborts, and then the same line teleports the arm a real distance into a
    pose nothing checked. Snap the no-op; refuse the teleport."""
    import types as _types
    from morph.arm.model import ArmModel as _AM
    from morph.arm.collision import Q8_TOL_M as _TM, Q8_TOL_RAD as _TR
    _AM_Q8 = _AM.Q8
    glb = {"np": np, "os": os, "ArmModel": _AM, "Q8_TOL_M": _TM, "Q8_TOL_RAD": _TR,
           "KNOWN": {"arm_joints": []}, "HEADLESS": True}
    settle = _exec_method(("morph", "pick", "descend.py"), "_pick_settle", glb)

    class _D:
        def __init__(self, residue):
            self.names = ["j0"]
            self.idx = {n: i for i, n in enumerate(_AM_Q8)}
            self.dt = 1.0 / 240.0
            self._stage = None
            self._gains_writer = None
            self._residue = float(residue)
            self.snapped = []
            self._plan_fallbacks = {}
            self._plan_to_config_reason = None
            self.robot = _types.SimpleNamespace(
                set_joint_positions=self.snapped.append,
                get_joint_positions=lambda: np.zeros(8))
        def _arm_q8(self):
            return np.zeros(8)
        def _hold(self, q, *a, **k):
            pass
        def _fallback(self, tag):
            self._plan_fallbacks[tag] = self._plan_fallbacks.get(tag, 0) + 1
        def _set_gains(self, *a, **k):
            # Everything after the snap decision -- gains, the settle loop, the body clear -- is a
            # different subject. Stop here so this test fails for its own reason and no other.
            raise _Cut
        def _arm_gain(self, n, kp):
            return kp, kp / 10.0, None
        def _wheel_gain(self, n):
            return 0.0, 0.0
        def _body_clear(self, *a, **k):
            pass
        def _finite(self, *a, **k):
            return True
        def _arm_obstacles(self, *a, **k):
            return []
        def _apply(self, *a, **k):
            pass
        def _force(self, *a, **k):
            pass

    def _run(demo, st):
        try:
            settle(demo, st)
        except _Cut:
            pass

    def _st(goal):
        return _types.SimpleNamespace(obj_idx=0, status=None, dock=None, qg=goal, obj=None,
                                      park=None, pin_bystanders=None, replay=True,
                                      alive=True, grip=None)

    near = _D(0.0)
    _run(near, _st(np.zeros(8)))
    assert len(near.snapped) == 1, "a residue of zero is the no-op snap and must still happen"

    far = _D(0.5)
    goal = np.zeros(8)
    # half a metre of boom slide away from where the arm stands
    goal[3] = 0.5
    _run(far, _st(goal))
    assert far.snapped == [], (
        f"teleported {0.5 * 1000:.0f}mm into a pose nothing checked; snapped={len(far.snapped)}")
    assert far._plan_fallbacks, "a refused teleport must be counted, not silent"


def test_a_multi_cycle_log_fails_when_any_cycle_fails():
    """The defect: findall/[-1] let one good final cycle mask eight failures."""
    import verify_place
    txt = ("=== cycle\n>>> PLACE obj 0 -> slot 0 [1,2,3] ok=True -> failed\n"
           ">>> PLACE obj 1 -> slot 4 [1,2,3] ok=True -> precise_success\n")
    assert verify_place.grade_gate(txt) is False, "a failed cycle must fail the run"


def test_a_multi_cycle_log_passes_when_every_cycle_passes():
    import verify_place
    txt = (">>> PLACE obj 0 -> slot 0 [1,2,3] ok=True -> precise_success\n"
           ">>> PLACE obj 1 -> slot 4 [1,2,3] ok=True -> precise_marginal\n")
    assert verify_place.grade_gate(txt) is True


def test_the_gate_fails_a_log_with_any_fallback():
    import verify_place
    assert verify_place.fallback_gate(">>> plan-fallbacks[obj 0]: 2 (goal-ik-miss=2)\n") is False, "fallback_gate must return False for a log containing SAFETY fallbacks"

def test_the_gate_passes_a_log_with_none():
    import verify_place
    assert verify_place.fallback_gate(">>> plan-fallbacks[obj 0]: 0\n") is True, "fallback_gate must return True for a log with zero fallbacks"


def test_a_graded_cycle_with_no_tally_fails_the_fallback_gate():
    """`grade_gate` (`verify_place.py::grade_gate`) and `drift_gate` require evidence via `bool(x) and all(...)`,
    and `fallback_gate` requires the same; an aborted run prints no
    tally, `all([])` is True, and a bare `all(...)` would pass the very run `grade_gate` fails because it
    aborted early enough to print nothing."""
    import verify_place
    graded_no_tally = ">>> PLACE obj 0 -> slot 0 ok=True -> precise_success\n"
    assert verify_place.grade_gate(graded_no_tally) is True, "the fixture is not a graded cycle"
    assert verify_place.fallback_gate(graded_no_tally) is False, (
        "a graded cycle that reported no plan-fallbacks tally passed the fallback gate on the "
        "empty `all([])`, while grade_gate fails the same log")


def test_a_log_that_predates_the_counter_still_passes_via_the_legacy_path():
    """The absence-pass `fallback_gate` used to grant itself is `check()`'s `pre_instrument` job,
    and has been since that flag existed -- one convention for pre-counter logs, not two. So the
    gate is free to require evidence like both its siblings, and a genuinely legacy log is still
    exempted where every other absence check is exempted."""
    import verify_place
    legacy = ">>> PLACE obj 0 -> slot 0 ok=True\n"
    assert verify_place.fallback_gate(legacy) is False, (
        "the gate itself must require evidence, like grade_gate and drift_gate")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        verify_place.check_text(legacy, "legacy", pre_instrument=True)
    assert "fallback-gate: FAIL" not in buf.getvalue(), (
        f"a log that predates the counter was failed for having no tally:\n{buf.getvalue()}")


def test_the_drift_gate_fails_an_unhealthy_cycle():
    import verify_place
    assert verify_place.drift_gate(">>> chassis released place-end (280 drift re-asserts, 1, 2)\n") is False


def test_the_drift_gate_passes_the_healthy_band():
    import verify_place
    assert verify_place.drift_gate(">>> chassis released place-end (3 drift re-asserts, 1, 2)\n") is True


def test_a_high_shelf_cycle_is_not_judged_against_a_low_shelf_cycle():
    """The arm-clear false failure: hand z from a HIGH-slot cycle compared against the LOW
    board of cycle 0. Splitting per cycle must make both cycles pass on their own geometry."""
    import verify_place
    txt = (">>> ===== CYCLE obj 0 -> slot 0 =====\n>>> PLACE obj 0 -> slot 0 ok=True -> precise_success\n"
           ">>> ===== CYCLE obj 7 -> slot 8 =====\n>>> PLACE obj 7 -> slot 8 ok=True -> precise_success\n")
    assert len(verify_place.split_cycles(txt)) == 2


def test_a_log_with_no_banner_is_one_segment():
    import verify_place
    assert len(verify_place.split_cycles(">>> PLACE obj 0 -> slot 0 ok=True\n")) == 1


def test_the_model_gate_fails_a_log_with_a_divergence():
    import verify_place
    assert verify_place.model_gate(">>> MODEL-PHYSICS DIVERGENCE[descent 200]: ...\n") is False


def test_the_model_gate_passes_a_clean_log():
    import verify_place
    assert verify_place.model_gate(">>> arm-body[descent 0]: dh +0.063\n") is True


def test_the_track_gate_fails_a_log_with_a_breach():
    import verify_place
    assert verify_place.track_gate(">>> TRACK BREACH[descent]: ArmLeftJoint_1 cmd +0.360 act +0.366 err 6.00mrad (tol 5.0) -- ...\n") is False


def test_the_track_gate_passes_a_clean_log():
    import verify_place
    assert verify_place.track_gate(">>> arm-body[descent 0]: dh +0.063\n") is True


# L5/L5b: `arm-clear` measured the hand FRAME ORIGIN, then the farthest corner over an
# UNCONSTRAINED wrist, which fails poses the place path never reaches.

# The place path levels and HOLDS the wrist through the slide, and the measured grasp puts
# object-up on the hand's -Y. So the world +z row of world<-hand is (0, -1, 0).
LEVEL_HOLD_R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def test_the_level_hold_fixture_is_a_real_rotation():
    """A sign check that runs. LEVEL_HOLD_R is the ONLY place a hand-frame axis claim enters these
    fixtures, and an inverted column would quietly move every expected number below."""
    assert np.allclose(LEVEL_HOLD_R @ LEVEL_HOLD_R.T, np.eye(3))
    assert abs(float(np.linalg.det(LEVEL_HOLD_R)) - 1.0) < 1e-12
    assert tuple(LEVEL_HOLD_R[2]) == (0.0, -1.0, 0.0)


def test_sweep_top_reproduces_the_box_its_own_max_z_under_identity():
    """The support function's own self-check: with no rotation the top of the box above the frame
    origin IS the box's +z face, so a transposed R or a dropped centre term shows up here."""
    from morph.arm.collision import FINGER_SWEEP_AABB, sweep_top
    assert abs(sweep_top(np.eye(3)) - float(FINGER_SWEEP_AABB[1][2])) < 1e-12


def test_sweep_top_on_the_held_wrist_is_far_under_the_unconstrained_corner():
    """`HAND_TOP = 0.2279` was the farthest corner over an unconstrained SO(3) wrist.
    The wrist the place path actually holds puts 0.067 m of hand above the frame origin -- 161 mm
    less -- so the constant failed level places with 60 mm of real clearance."""
    from morph.arm.collision import FINGER_SWEEP_AABB, sweep_top
    assert abs(sweep_top(LEVEL_HOLD_R) - 0.067) < 1e-9, sweep_top(LEVEL_HOLD_R)
    corner = max(math.dist((0, 0, 0), c) for c in itertools.product(*zip(*FINGER_SWEEP_AABB)))
    assert abs(corner - 0.2279) < 5e-5, corner
    assert sweep_top(LEVEL_HOLD_R) < corner - 0.15


def test_sweep_top_never_exceeds_the_farthest_corner_at_any_wrist_orientation():
    """The bound the deleted constant was reaching for, kept as a property of the formula: no
    rotation can lift the box higher than its farthest corner. Second assert: FINGER_SWEEP_AABB is
    only the WHOLE hand while it still contains the shipped `Gripper_Link3_1` geometry that
    `morph/arm/model.py::ArmModel.__init__` unions into it."""
    from morph.arm.collision import FINGER_SWEEP_AABB, sweep_top
    corner = max(math.dist((0, 0, 0), c) for c in itertools.product(*zip(*FINGER_SWEEP_AABB)))
    rng = np.random.default_rng(0)
    q = rng.normal(size=(4000, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    for w, x, y, z in q:
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        assert sweep_top(R) <= corner + 1e-12
    shipped = json.load(open(os.path.join(ROOT, "usd/_arm_model.json")))["link_aabb"]["Gripper_Link3_1"]
    assert max(math.dist((0, 0, 0), c) for c in itertools.product(*zip(*shipped))) <= corner


def test_insert_logs_the_world_space_hand_top_in_the_format_verify_place_parses():
    """The sample site is where the rotation is IN HAND, so it logs the measured
    world-space hand top rather than leaving the checker to bound it over every wrist it might
    have had. insert.py imports Isaac at module scope, so this reads the source."""
    from morph.arm.collision import sweep_top
    src = open(os.path.join(ROOT, "morph", "place", "insert.py")).read()
    assert "sweep_top(" in src, "the sample site does not compute the hand box's support at all"
    assert "hand top {" in src, "the sample line does not emit `hand top`"
    rendered = f">>> insert[slide] wp 1/6: ... hand z 0.550 hand top {0.550 + sweep_top(LEVEL_HOLD_R):.3f}"
    assert re.findall(r"hand top ([\d.]+)", rendered) == ["0.617"], rendered


def _place_log(hand_z=None, hand_top=None, fallbacks=True, heartbeat=True,
               hand_top_max=None, summary=True, instrument_failed=False,
               insert_end=1.0, released=1.0, settled=1.0):
    """One otherwise-green place cycle. Every other gate in `check_text` passes on it, so a FAIL
    here can only come from the check under test -- without that control these would be more
    gates passing on absence. `hand_top` defaults to the wrist the place path holds, and
    `hand_top_max` (the whole-slide maximum the gate actually reads) defaults to it: a flat slide
    is one whose samples and whose maximum agree."""
    lines = [">>> pre-dock height: object z -> 0.400 (slide 0.400, spans 0.310..0.490)"]
    if hand_z is not None:
        if hand_top is None:
            from morph.arm.collision import sweep_top
            hand_top = hand_z + sweep_top(LEVEL_HOLD_R)
        lines.append(f">>> insert[slide] wp 1/6: reach -> 0.300m (final 0.300, z off slide +0mm) "
                     f"| dh +0.100  a1 0.300  h1 0.400 -> 0.400  th +0.000 held "
                     f"hand z {hand_z:.3f} hand top {hand_top:.3f}")
        if summary:
            lines.append(f">>> insert[slide]: hand top max "
                         f"{hand_top if hand_top_max is None else hand_top_max:.3f} over 200 "
                         f"steps (every step; the wp lines above are 6 samples)")
    else:                                   # a run with no sample line still reports its waypoint
        lines.append(">>> insert[slide] wp 1/6: reach -> 0.300m (final 0.300, z off slide +0mm)")
    if instrument_failed:
        lines.append(">>> TRACK INSTRUMENT FAILED[insert-slide]: RuntimeError: physics view "
                     "dropped mid-readback -- the tracking check is not reporting, so nothing "
                     "downstream can tell a tracked run from an unmeasured one (printed once)")
    if heartbeat:
        lines.append(">>> track-err[insert-slide]: worst ArmLeftJoint_1 cmd +0.3600 "
                     "act +0.3600 err +0.00mm/mrad | p95 0.00")
    lines += [f">>> carry[{t}]: upright {d}deg"
              for t, d in (("insert-end", insert_end), ("released", released),
                           ("settled", settled)) if d is not None]
    if fallbacks:
        lines.append(">>> plan-fallbacks[obj 0]: 0")
    lines.append(">>> PLACE obj 0 -> slot 0 ok=True -> precise_success")
    return "".join(l + "\n" for l in lines)


def test_the_arm_clear_fixture_is_green_on_its_own():
    """The control. hand z 0.400 on the held wrist tops out at 0.467, 210mm under the low board,
    so if any test below goes red it is the check under test talking and not some other gate."""
    import verify_place
    assert verify_place.check_text(_place_log(hand_z=0.400), "ctrl") is True


def test_arm_clear_passes_a_level_place_that_really_does_clear_the_board():
    """Comparing `hand z` against the farthest corner of the hand box over an unconstrained wrist
    (0.2279) would falsely fail level places: hand z 0.550 -- a level place whose real hand top is
    0.617, a full 60mm under the 0.677 board -- was reported as -101mm and failed."""
    import verify_place
    assert verify_place.check_text(_place_log(hand_z=0.550), "level") is True


def test_arm_clear_fails_a_hand_top_that_really_does_breach_the_board():
    """The true negative, without which the fix above is an ungated pass. Same hand z, but a wrist
    rolled so the box's own +z face (0.166, `morph/arm/collision.py::FINGER_SWEEP_AABB`) points up: the hand tops out at 0.716,
    39mm INSIDE the low board."""
    import verify_place
    from morph.arm.collision import FINGER_SWEEP_AABB
    breach = 0.550 + float(FINGER_SWEEP_AABB[1][2])
    assert verify_place.check_text(_place_log(hand_z=0.550, hand_top=breach), "breach") is False


def test_arm_clear_fails_a_run_that_should_have_sampled_the_hand_and_did_not():
    """`no hand-z samples (older log) -- not gated` must not pass a run that placed and
    reported no samples. The hand-z instrument predates the fallback counter,
    so any log carrying a plan-fallbacks line was produced by a build that had it."""
    import verify_place
    assert verify_place.check_text(_place_log(hand_z=None), "gap") is False


def test_arm_clear_still_passes_a_log_that_predates_the_hand_z_instrument():
    """The other half: a log that predates the instruments outright is not gated on their
    absence -- the same `pre_instrument` rule the fallback gate uses in `verify_place.py::check_text`. It is
    not a grandfather clause: a log written AFTER the counter and missing this evidence fails."""
    import verify_place
    assert verify_place.check_text(_place_log(hand_z=None, fallbacks=False, heartbeat=False),
                                   "old", pre_instrument=True) is True


# --- L5b item 2: the track gate passed on ABSENCE, and its only instrument swallowed every
# exception, so a run whose readback raised on every step graded exactly like a clean one. ------

def test_the_track_gate_fails_a_placed_cycle_with_no_heartbeat():
    """`TRACK BREACH` is the gate's only evidence and it is printed ONLY on a breach, so a cycle
    that produced no tracking samples at all -- instrument off, or raising every step -- read as
    a clean one. A modern placed cycle must show at least one `track-err[...]` heartbeat."""
    import verify_place
    assert verify_place.check_text(_place_log(hand_z=0.400, heartbeat=False), "mute") is False


def test_the_track_gate_still_passes_a_log_that_predates_the_track_instrument():
    """The other half, and the same `pre_instrument` rule every other absence check here uses -- not a
    third convention."""
    import verify_place
    assert verify_place.check_text(_place_log(hand_z=0.400, fallbacks=False, heartbeat=False),
                                   "old", pre_instrument=True) is True


def _module_consts(rel, names):
    """Module-level constant assignments out of a file that imports Isaac and so cannot be
    imported. `_exec_method`'s equivalent for the constants a method reads from its module."""
    tree = ast.parse(open(os.path.join(ROOT, *rel)).read())
    body = [n for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    g = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), os.path.join(*rel), "exec"), g)
    assert names <= set(g), f"{sorted(names - set(g))} not found in {os.path.join(*rel)}"
    return {k: g[k] for k in names}


def test_the_track_readback_prints_its_heartbeat_when_it_works():
    """The positive control, and not garnish: the test below asserts that a FAILING readback says
    so, which passes just as happily against a harness that never reaches the readback at all."""
    _demo, out = _run_track(RUN19_CMD, RUN19_ACT, dt=0.5)
    beat = [l for l in out if re.search(r">>> track-err\[[^\]]+\]:", l)]
    assert beat, f"the working readback printed no heartbeat: {out}"


def test_the_track_readback_names_its_failure_instead_of_swallowing_it():
    """`except Exception: pass` around the readback made an instrument
    that raises on every step indistinguishable from one that is quiet because nothing is wrong --
    the readback must name its failure instead of swallowing it."""
    _demo, out = _run_track(RUN19_CMD, RUN19_ACT, dt=0.5, read_raises=True)
    said = [l for l in out if "TRACK INSTRUMENT FAILED" in l]
    assert said, (
        f"the tracking readback raised and printed nothing: track_gate now has no evidence and no "
        f"way to tell that from a clean run: {out}")


# `_recover`'s bad-base branch RE-HOMES the arm and then drives q0, AFTER the latch can already be
# armed in the same cycle. `world.reset()` makes the STATE clean, not the MOTION validated.

_RECOVER_ARM_WRITES = ("set_joint_positions", "_force", "_apply")


class _RecoverObj:
    """One entry of `self.objs`, and `self.world`. A recorder: the layout restore and the settle
    loop's `world.step` both run through these, and each call names itself in the log."""

    def __init__(self, calls):
        self.calls = calls

    def __getattr__(self, name):
        def rec(*a, **k):
            self.calls.append(name)
        return rec


class _RecoverRobot:
    """`self.robot`. `get_world_pose` is REAL because `_recover` reads THROUGH it -- `float(p[0])`
    for the bounds test on the way in, and again on the very last line. Everything else answers
    with a recorder, so an arm command names ITSELF instead of having to be anticipated."""

    def __init__(self, calls, base):
        self.calls = calls
        self.base = base

    def get_world_pose(self):
        self.calls.append("get_world_pose")
        return ([self.base[0], self.base[1], 0.10], [1.0, 0.0, 0.0, 0.0])

    def __getattr__(self, name):
        def rec(*a, **k):
            self.calls.append(name)
        return rec


class _RecoverDemo:
    """The `Demo` surface `morph/place/recover.py::_recover` touches.

    BASE is OUT OF BOUNDS (x > 7.8), so the sane-base early return in `morph/place/recover.py::RecoverStage._recover` is NOT taken
    and the branch under test is the one that re-homes. That is the single knob the control rests
    on: point BASE at a sane pose and `_recover` returns having commanded no joint at all, which
    is a harness that passes the no-arm-write test below without ever reaching the branch.

    No `__getattr__` here, deliberately, unlike the two recorders above: `self.*` is where a NEW
    arm write would be added, and an unanticipated one must raise AttributeError and go RED rather
    than be logged under a name no assertion happens to ban."""

    BASE = (12.0, -3.0)

    def __init__(self, calls, latch, n_objs):
        self.calls = calls
        # Demo.__init__ owns this: the ONE place it is cleared
        self._safety_abort = latch
        # -> the settle loop runs its 30 iterations, as in production
        self.dt = 1.0 / 60.0
        self.names = [f"j{i}" for i in range(6)]
        self.q0 = [0.0] * 6
        self.z0 = 0.10
        self.obj_half_h = 0.075
        self.placed = [(0, [1.0, -2.0, 0.90])]
        self.obj_xy = [(1.0 + i, -2.0 - i) for i in range(n_objs)]
        self.objs = [_RecoverObj(calls) for _ in range(n_objs)]
        self.robot = _RecoverRobot(calls, self.BASE)
        self.world = _RecoverObj(calls)

    def _force(self, q):
        self.calls.append("_force")

    def _apply(self, q):
        self.calls.append("_apply")


def _run_recover(latch):
    """The REAL `morph/place/recover.py::_recover` over that stub. Returns (call log, printed
    lines). recover.py imports Isaac at module level, so `_exec_method` is the only way to RUN it.

    Deliberately NOT wrapped in a try/except: an exception out of `_recover` must FAIL these
    tests. Swallowed, it leaves a short call log that reads exactly like a guard doing its job --
    which is how four assertions in this file went vacuous two rounds ago."""
    # the real loop counts, not a harness guess
    from morph.config import HEADLESS, NUM_OBJECTS
    from morph.geometry import quat_yaw
    calls, out = [], []
    glb = {"math": __import__("math"), "np": __import__("numpy"),
           "HEADLESS": HEADLESS, "NUM_OBJECTS": NUM_OBJECTS, "quat_yaw": quat_yaw,
           "scene": SimpleNamespace(set_arm_gains=lambda *a, **k: calls.append("set_arm_gains")),
           "print": lambda *a, **k: out.append(" ".join(str(x) for x in a))}
    _RecoverDemo._recover = _exec_method(("morph", "place", "recover.py"), "_recover", glb)
    _RecoverDemo(calls, latch, NUM_OBJECTS)._recover()
    return calls, out


def test_recover_does_not_re_home_the_arm_while_a_safety_abort_is_latched():
    """`run_cycle` calls `_recover()` in `play_isaac.py::Demo.run_cycle` after the latch
    can already be armed in the same cycle, and the bad-base branch writes q0 into the arm
    in `morph/place/recover.py::RecoverStage._recover` and then drives it for half a second. The `world.reset()` ahead of
    those commands no joint: it makes the STATE clean, which is not the MOTION validated."""
    calls, out = _run_recover(latch=True)
    ran = [n for n in _RECOVER_ARM_WRITES if n in calls]
    assert not ran, (
        f"{ran} ran with the safety latch set -- each commands the arm on a path no checkpoint "
        f"validated, and run_cycle reaches this call AFTER the abort: {calls}")
    said = [l for l in out if "ABORT" in l.upper() and "SKIP" in l.upper()]
    assert said, (
        f"the skip printed no reason of its own -- a recovery that silently does nothing is "
        f"indistinguishable from one that was never reached: {out}")


def test_recover_still_re_homes_a_bad_base_when_no_abort_is_latched():
    """The positive control, and not garnish: the test above asserts an ABSENCE, so it passes just
    as happily against a harness whose base pose never reaches the bad-base branch at all. Same
    stub, same bad base, latch clear -- every name that test forbids must APPEAR here, or its
    absence assertion measures nothing. Un-bricking a bad base is load-bearing when the run goes
    on; only the latched case has no next cycle to un-brick for."""
    calls, out = _run_recover(latch=False)
    blind = [n for n in _RECOVER_ARM_WRITES if n not in calls]
    assert not blind, (
        f"the control never reaches {blind}, so the latched test's absence assertion for those "
        f"names is vacuous -- and vacuous assertions read as coverage: {calls}")
    assert "reset" in calls, (
        f"the control never reached `world.reset()` -- it took the sane-base early return at "
        f"recover.py:39 instead of the bad-base branch under test: {calls}")
    assert not [l for l in out if "SKIP" in l.upper()], (
        f"a recovery with nothing latched must not report itself skipped: {out}")


def test_the_recover_control_is_not_truncated_before_the_end_of_recover():
    """A control is only as wide as the run it survives. `_recover`'s
    LAST statement is its own `RECOVER done` print, so that line is the proof the control ran the
    WHOLE branch -- a name absent past a break reads exactly like a name absent because a guard
    suppressed it. The AST check keeps the anchor honest if the tail is ever rewritten."""
    _tail = _func(("morph", "place", "recover.py"), "_recover").body[-1]
    assert isinstance(_tail, ast.Expr) and "RECOVER done" in ast.unparse(_tail), (
        "`_recover` no longer ENDS in its `RECOVER done` print -- the truncation check below is "
        "anchored on that line, and an anchor that is no longer last would pass on a partial run")
    calls, out = _run_recover(latch=False)
    assert [l for l in out if "RECOVER done" in l], (
        f"the control never reached the last statement of `_recover`, so everything it asserts "
        f"about the branch past the break is vacuous: {out} | {calls}")

# --- L10b item 1: L10 made `_apply`'s jump zeroing PER-JOINT. The recovery teleport is the one
# place the OLD global zeroing was accidentally load-bearing. ---------------------------------

class _RecoverDriveDemo(_RecoverDemo):
    """`_RecoverDriveDemo` with the REAL drive-path `_apply` and `_force` bolted on, so what the
    recovery teleport does to the velocity feedforward is read out of production code rather than
    modelled here. Drives are ON deliberately: that is the ONLY mode where `_force` returns BEFORE
    it nulls `_drv_prev` (`morph/robot.py::RobotMixin._force`), so it is the only mode where a stale command history
    can survive the teleport and reach `_apply` at all. `dt` is the production step, so the 0.005
    rad the per-joint jump mask lets through is 1.2 units/s of velocity the arm never has."""

    def __init__(self, calls, latch, n_objs, prev):
        super().__init__(calls, latch, n_objs)
        # -> 120 settle iterations, as in production
        self.dt = 1.0 / 240.0
        self._drv_prev = np.asarray(prev, float)
        self._finger_cmd = self._finger_pin = None
        # the recorder robot has no dof_properties
        self._q_lim = False
        self.applied = []
        self.ctrl = SimpleNamespace(apply_action=self.applied.append)

    def _drive_on(self):
        return True

    def _with_closure(self, q):
        return np.asarray(q, float)


def _recover_drive_demo(prev, latch=False):
    """A `_RecoverDriveDemo` carrying the REAL `_recover`, `_apply` and `_force`. recover.py and
    robot.py both import Isaac at module scope, so `_exec_method` is the only way to RUN them."""
    from morph.config import HEADLESS, NUM_OBJECTS
    from morph.geometry import quat_yaw
    calls, out = [], []
    glb = {"math": math, "np": np, "HEADLESS": HEADLESS, "NUM_OBJECTS": NUM_OBJECTS,
           "quat_yaw": quat_yaw,
           "scene": SimpleNamespace(set_arm_gains=lambda *a, **k: calls.append("set_arm_gains")),
           "print": lambda *a, **k: out.append(" ".join(str(x) for x in a))}
    _RecoverDriveDemo._recover = _exec_method(("morph", "place", "recover.py"), "_recover", glb)
    rglb = {"np": np, "math": math, "os": SimpleNamespace(environ={"ARM2_FREEZE": "0"}),
            "ArticulationAction": lambda **kw: SimpleNamespace(**{"joint_velocities": None, **kw}),
            "print": lambda *a, **k: out.append(" ".join(str(x) for x in a))}
    for _m in ("_apply", "_pin_fingers", "_force"):
        setattr(_RecoverDriveDemo, _m, _exec_method(("morph", "robot.py"), _m, rglb))
    return _RecoverDriveDemo(calls, latch, NUM_OBJECTS, prev), calls, out


# A stale `_drv_prev` from the pose the recovery is undoing: two joints sit under the jump
# threshold, which is exactly the case the per-joint mask lets through and global zeroing did not.
_STALE_PREV = (0.90, 0.004, -0.70, 0.003, 0.50, 0.00)


def test_the_recovery_teleport_does_not_leave_a_fictitious_velocity_behind():
    """`_recover` teleports the WHOLE articulation to `q0` after `world.reset()` and left
    `_drv_prev` pointing at the exploded pose. That was harmless while a jump zeroed the
    feedforward for EVERY joint; with per-joint jump zeroing, a joint whose stale
    delta happens to be under 0.005 is handed `delta/dt` -- up to 1.2 units/s of velocity on a
    joint that has just been TELEPORTED and is not moving at all. `_force` does not cover this:
    on the drives it returns before it nulls anything."""
    q0 = np.zeros(6)
    guard, _gc, _go = _recover_drive_demo(_STALE_PREV)
    guard._apply(q0.copy())
    vg = np.asarray(guard.applied[-1].joint_velocities, float)
    assert abs(float(vg[1]) - (-_STALE_PREV[1] / guard.dt)) < 1e-9, (
        f"the same stale history through the same `_apply` did NOT produce a fictitious velocity "
        f"on j1 ({float(vg[1]):+.3f}, expected {-_STALE_PREV[1] / guard.dt:+.3f}) -- this harness "
        f"cannot see the defect, so nothing below is evidence")

    demo, calls, out = _recover_drive_demo(_STALE_PREV)
    demo._recover()
    assert "set_joint_positions" in calls and demo.applied, (
        f"the recovery never reached the teleport or the settle loop, so nothing below is "
        f"evidence: {calls}")
    v = np.asarray(demo.applied[0].joint_velocities, float)
    hot = [(i, round(float(v[i]), 3)) for i in range(len(v)) if float(v[i]) != 0.0]
    assert not hot, (
        f"the first `_apply` after the recovery teleport commanded qd_target {hot} on joints the "
        f"teleport had just put at rest -- `_drv_prev` still describes the pose the recovery is "
        f"undoing")



# `TRACK INSTRUMENT FAILED` was made loud and then graded by nothing, and the hand top the
# arm-clear gate calls whole-slide was emitted a handful of times with no bound in between.

def test_the_track_gate_fails_a_cycle_whose_instrument_DIED_mid_run():
    """A valid heartbeat from BEFORE the readback died, the failure line, and a graded placement
    must fail: the breach branch has no `TRACK BREACH` to find, and the heartbeat branch is an `elif` that a present
    heartbeat skips -- so making the failure visible must also fail the gate."""
    import verify_place
    log = _place_log(hand_z=0.400, instrument_failed=True)
    assert re.search(r">>> track-err\[[^\]]+\]:", log), "the fixture lost its heartbeat"
    assert "TRACK INSTRUMENT FAILED" in log and "-> precise_success" in log, log
    assert verify_place.check_text(log, "died") is False, (
        "a run whose tracking instrument died still graded ALL CHECKS PASS")
    assert verify_place.track_gate(log) is False, (
        "`track_gate` itself still reports a clean chain, so every OTHER caller of it keeps "
        "grading a dead instrument as a tracked run")


def test_the_track_instrument_gate_does_not_fire_on_a_clean_run():
    """The true negative. Without it the gate above is satisfied by `return False`."""
    import verify_place
    assert verify_place.track_gate(_place_log(hand_z=0.400)) is True
    assert verify_place.check_text(_place_log(hand_z=0.400), "clean") is True


SLIDE_DT = 0.01
SLIDE_N = max(2, int(2.0 / SLIDE_DT))                          # insert.py's own `n`
SLIDE_SAMP = max(1, SLIDE_N // 6)                              # insert.py's own `_samp`
SLIDE_SAMPLES = [k for k in range(1, SLIDE_N + 1) if k % SLIDE_SAMP == 0]
PEAK_LO, PEAK_HI = SLIDE_SAMPLES[2] + 1, SLIDE_SAMPLES[3] - 1  # strictly between wp 3 and wp 4
SLIDE_BASE_Z = 0.400        # + the 0.166 hand box = 0.566, 111mm under the 0.677 low board
SLIDE_PEAK_Z = 0.520        # + the 0.166 hand box = 0.686, 9mm INSIDE it


class _SlideDemo:
    # `_insert_sequence` hands this to `move_linear` as `on_step`; these harnesses measure the
    # slide, not the contact watch, so it is a no-op here.
    def _contact_watch(self, tag):
        return None

    """The `Demo` surface `_insert_sequence` touches on its untilted, turret-track-off, no-drive
    path, and nothing else. The hand's z is a function of the STEP COUNT, not of how often it is
    read, so the six-sample form and the every-step form are shown the SAME physical slide."""

    @contextlib.contextmanager
    def _time_stage(self, stage_name, file_line=None):
        yield

    dt = SLIDE_DT

    def __init__(self):
        self.step = 0
        self.reads = []
        self.idx = {"ColumnLeftBearingJoint_1": 0, "ColumnRightBearingJoint_1": 1,
                    "ArmLeftJoint_1": 2, "BaseJoint_1": 3}
        self.clut = {"mount": (0.0, 0.0), "grip": None}
        self.q0 = np.array([0.400, 0.500, 0.200, 0.000])
        self.robot = SimpleNamespace(get_joint_positions=lambda: self.q0.copy())

    def hold(self, q):
        self.step += 1

    def _hand_frame(self):
        z = SLIDE_PEAK_Z if PEAK_LO <= self.step <= PEAK_HI else SLIDE_BASE_Z
        self.reads.append(self.step)
        return np.array([0.0, 0.0, z]), np.eye(3)

    def base_ledger(self):
        return 0.0, 0.0, 0.0

    def _grip_fk(self, h1, h2, a1, th):
        return np.zeros(3)

    def _lut_interp(self, table, dh, a1, n):
        return [0.5, 0.0, 0.0]                  # 0.5m of reach for a 0.3m goal: no tilt fallback

    def _lut_z(self, dh, a1):
        return 0.0

    def _bisect_a1(self, pred, lo, hi, iters=None):
        return hi

    def _closure_passives(self, dh, a1):
        return {}

    def _drive_on(self):
        return False

    def _ramp(self, q_from, q_to, secs, hold_fn, check=None):
        # `check` is honoured, not just accepted: a stub that swallowed it would let the ramps go
        # unchecked while every test here stayed green. Endpoint only -- this suite measures the slide.
        q = np.asarray(q_to, float).copy()
        if check is not None and check(q):
            return None
        return q

    def _body_clear(self, tag):
        pass

    def _arm_insert(self, *a, **k):
        return None


def _run_slide():
    """The REAL `morph/place/insert.py::_insert_sequence` over that stub. insert.py imports Isaac
    at module scope, so `_exec_method` is the only way to RUN it. Returns (demo, printed lines)."""
    from morph.arm.collision import sweep_top
    from morph.config import A1_MAX, COLUMN_MAX
    out = []
    glb = {"math": math, "np": np, "sweep_top": sweep_top,
           "COLUMN_MAX": COLUMN_MAX, "A1_MAX": A1_MAX, "move_linear": _stub_move_linear(),
           "os": SimpleNamespace(environ={"SLIDE_TH_TRACK": "0", "TRIM_BEFORE_LOWER": "0"}),
           "print": lambda *a, **k: out.append(" ".join(str(x) for x in a))}
    demo = _SlideDemo()
    obj = SimpleNamespace(get_world_poses=lambda: (np.array([[0.0, 0.0, 0.400]]),
                                                   np.array([[1.0, 0.0, 0.0, 0.0]])))
    seq = _exec_method(("morph", "place", "insert.py"), "_insert_sequence", glb)
    got = seq(demo, obj, np.array([0.300, 0.000, 0.350]), 0.050, demo.hold)
    assert got is not None, f"the slide bailed out, so nothing below is evidence: {out}"
    assert demo.step == SLIDE_N, f"the slide ran {demo.step} steps, not {SLIDE_N}: {out}"
    return demo, out


def test_the_synthetic_peak_really_does_fall_between_two_sample_points():
    """The control for the two tests below. If the peak overlapped a sample point, a six-sample
    reading would catch it by accident and neither test would measure anything."""
    assert len(SLIDE_SAMPLES) == 6, f"insert.py's formula gives {len(SLIDE_SAMPLES)} samples"
    assert PEAK_LO <= PEAK_HI, "the sample gap is empty: no peak can fall inside it"
    assert not [k for k in SLIDE_SAMPLES if PEAK_LO <= k <= PEAK_HI], (
        f"the peak {PEAK_LO}..{PEAK_HI} overlaps a sample point: {SLIDE_SAMPLES}")
    _demo, out = _run_slide()
    assert len([l for l in out if "insert[slide] wp " in l]) == 6, (
        f"the slide did not emit six waypoint samples: {out}")


def test_the_whole_slide_maximum_catches_a_peak_the_six_samples_miss():
    """Both readings are taken off ONE run of the real loop, so this is a measurement
    of the sampling gap and not an assertion about it: the six samples report the base height
    while the hand really reached 120mm higher between wp 3 and wp 4."""
    from morph.arm.collision import sweep_top
    demo, out = _run_slide()
    lift = sweep_top(np.eye(3))
    sampled = [float(m) for l in out for m in re.findall(r"hand top ([\d.]+)", l)]
    assert len(sampled) == 6, f"expected the six per-waypoint samples, got {sampled}"
    assert max(sampled) == round(SLIDE_BASE_Z + lift, 3), (
        f"the samples saw {max(sampled)}, so this run does not demonstrate a missed peak")
    assert demo.reads[-1] == SLIDE_N, (
        f"the hand was last read at step {demo.reads[-1]} of {SLIDE_N}, so the reading does not "
        f"reach the end of the slide")
    assert len(demo.reads) == SLIDE_N, (
        f"the hand was read {len(demo.reads)} times over {SLIDE_N} steps -- still a subset")
    m = re.findall(r"hand top max ([\d.]+) over (\d+) steps", "\n".join(out))
    assert m, f"the slide emitted no whole-slide hand-top maximum line: {out}"
    assert int(m[0][1]) == SLIDE_N, f"the maximum covers {m[0][1]} steps, not the slide's {SLIDE_N}"
    assert float(m[0][0]) == round(SLIDE_PEAK_Z + lift, 3), (
        f"the reported maximum is {m[0][0]}, but the hand reached "
        f"{SLIDE_PEAK_Z + lift:.3f} between samples -- it is still a sampled subset")


def test_the_arm_clear_gate_reads_the_whole_slide_maximum_and_not_the_samples():
    """The measurement above is only worth having if the GATE is what reads it. The same cycle
    twice: with the sampled value it clears the board by 111mm, with the real whole-slide maximum
    it is 9mm INSIDE it. A gate still reading the samples passes both."""
    import verify_place
    from morph.arm.collision import sweep_top
    seen = SLIDE_BASE_Z + sweep_top(np.eye(3))
    real = SLIDE_PEAK_Z + sweep_top(np.eye(3))
    assert verify_place.check_text(
        _place_log(hand_z=SLIDE_BASE_Z, hand_top=seen, hand_top_max=seen), "flat") is True, (
        "the control failed: a slide whose maximum IS its sample must still pass")
    assert verify_place.check_text(
        _place_log(hand_z=SLIDE_BASE_Z, hand_top=seen, hand_top_max=real), "peak") is False, (
        f"the hand reached {real:.3f} under the 0.677 board and the gate passed it: it is "
        f"reading the {seen:.3f} samples, not the whole-slide maximum")


def test_arm_clear_fails_a_modern_slide_that_reported_no_whole_slide_maximum():
    """Fail-closed, the same rule every other absence check here uses: a cycle that placed and
    emitted only the six samples has an UNMEASURED between-sample clearance, which is not the
    same thing as a clear one."""
    import verify_place
    from morph.arm.collision import sweep_top
    seen = SLIDE_BASE_Z + sweep_top(np.eye(3))
    assert verify_place.check_text(
        _place_log(hand_z=SLIDE_BASE_Z, hand_top=seen, summary=False), "gap") is False, (
        "a cycle that emitted only the six samples passed: the gate is still reading them")


def test_the_slide_accumulation_writes_nothing_the_robot_reads():
    """Item 2 is MEASUREMENT ONLY. The commanded joint vector must not depend on it, so the run
    above is compared against one whose hand never moves: same steps, same waypoint reach/dh/a1
    line, only the measured maximum differs."""
    global SLIDE_PEAK_Z
    _demo, peaked = _run_slide()
    was, SLIDE_PEAK_Z = SLIDE_PEAK_Z, SLIDE_BASE_Z
    try:
        _demo2, flat = _run_slide()
    finally:
        SLIDE_PEAK_Z = was
    strip = lambda ls: [re.sub(r"\s+hand (z|top) [\d.]+", "", l) for l in ls
                        if "hand top max" not in l]
    assert strip(peaked) == strip(flat), (
        "the hand-top accumulation changed what the slide commanded")


# ===== L9 group 3: three gates that pass on an empty collection (`all([])` is True) ============

def test_the_grade_gate_fails_a_run_that_produced_no_grade_line_at_all():
    """`all(... for g in grades)` over an EMPTY findall is True, so a run that crashed before any
    cycle could grade itself read exactly like a run whose every cycle passed. Same shape as
    results_gate's `bool(records) and all(...)` in `verify_place.py::results_gate`."""
    import verify_place
    assert verify_place.grade_gate("") is False, (
        "a log with no grade line at all passed the grade gate: `all([])` is True")
    assert verify_place.grade_gate(">>> something went wrong\n") is False


def test_the_grade_gate_still_passes_a_run_that_graded_itself_good():
    import verify_place
    assert verify_place.grade_gate("-> precise_success\n-> precise_marginal\n") is True


def test_the_drift_gate_fails_a_cycle_that_reported_no_drift_line_at_all():
    """Same shape: a cycle aborted before the release emits no `released ... (N drift` line, and
    the empty `all(...)` passed it."""
    import verify_place
    assert verify_place.drift_gate("") is False, (
        "a log with no drift line at all passed the drift gate: `all([])` is True")


def test_upright_fails_a_cycle_whose_carry_checkpoints_stop_before_the_shelf():
    """`after` is filtered out of `tilts`, so a cycle with carry checkpoints but none of the ones
    the gate judges left `after` empty -- and `all([])` PASSED it. The outer `if tilts:` does not
    cover this: the checkpoints exist, they are just all upstream."""
    import verify_place
    txt = _place_log(hand_z=0.400, insert_end=None, released=None, settled=None).replace(
        ">>> plan-fallbacks", ">>> carry[nav-end]: upright 0.5deg\n>>> plan-fallbacks")
    assert "carry[nav-end]" in txt and "carry[insert-end]" not in txt
    assert verify_place.check_text(txt, "aborted") is False, (
        "a cycle whose only carry checkpoint is upstream of the shelf passed the upright gate")


# ===== L9 group 2/4: the upright verdict must be the SETTLED attitude, not the released one ====

def test_upright_judges_the_settled_value_and_keeps_released_as_a_diagnostic():
    """run19: the object reads 5.6deg on the rim at `released` -- the hand is still there -- and
    lies flat 1.5s later once the backout has cleared. The shelf cares about where it ENDS UP, so
    the verdict is the settled reading; `released` stays in the log and is not the verdict."""
    import verify_place
    leaned = _place_log(hand_z=0.400, insert_end=0.5, released=5.6, settled=0.5)
    assert "carry[released]: upright 5.6deg" in leaned, "the fixture lost its released diagnostic"
    assert verify_place.check_text(leaned, "settles-flat") is True, (
        "a run that leaned at `released` and settled flat FAILED: the gate is still judging the "
        "instant the hand is in the way")


def test_upright_fails_an_object_still_tilted_after_the_backout():
    """The true negative: the same log with the tilt still there once the hand has gone."""
    import verify_place
    still = _place_log(hand_z=0.400, insert_end=0.5, released=5.6, settled=6.2)
    assert verify_place.check_text(still, "still-tilted") is False, (
        "an object left at 6.2deg after the backout passed the upright gate")


def test_upright_fails_a_released_cycle_that_never_reported_a_settled_reading():
    """Absence must not pass -- the same rule arm-clear and the track gate in `verify_place.py::check_text`
    already use. A cycle that got as far as `released` OWES a settled reading; without one
    its final resting attitude is unmeasured, which is not the same thing as upright."""
    import verify_place
    assert verify_place.check_text(
        _place_log(hand_z=0.400, settled=None), "owed") is False, (
        "a released cycle with no settled reading passed: its final attitude is unmeasured")


# ===== L9 group 1: the tracking gate measured the wrong joints against a mis-derived number =====
# A max-of-sparse-heartbeat-samples bound was applied per joint every step; most breaches were that.

def _class_consts(rel, cls, names):
    """Class-level constant assignments out of a file that imports Isaac and so cannot be
    imported. `_module_consts`' equivalent for the tables a method reads off `self`."""
    tree = ast.parse(open(os.path.join(ROOT, *rel)).read())
    cd = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
    body = [n for n in cd.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    g = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), os.path.join(*rel), "exec"), g)
    assert names <= set(g), f"{sorted(names - set(g))} not on {cls} in {os.path.join(*rel)}"
    return {k: g[k] for k in names}


TRACK_NAMES = ["ColumnLeftBearingJoint_1", "ColumnRightBearingJoint_1", "ArmLeftJoint_1",
               "ArmRightJoint_1", "RotationLeftJoint_1", "gripper_x_rotation_1",
               "gripper_y_rotation_1", "ContactCylinderJoint_1_1"]


class _TrackDemo:
    """The `Demo` surface `_apply` touches on its DRIVE path with everything else neutralised:
    `_q_lim` False skips the limit clip and an empty `idx` skips the arm-2 freeze. What is left is
    the command history, readback and tracking check."""

    def __init__(self, cmd, act, dt, drive=True, read_raises=False):
        self.names = list(TRACK_NAMES)
        self.dt = dt
        self.idx = {}
        self._finger_cmd = self._finger_pin = None
        self._q_lim = False
        self._drv_prev = None
        self._drive = drive
        self.clut = json.load(open(os.path.join(ROOT, "usd", "_closure_lut.json")))
        self.cmd = np.asarray(cmd, float)
        self.act = np.asarray(act, float)
        self.applied = []
        self._read_raises = read_raises
        # The two quantities the trace has no other source for. COUNTED: "the flag is off" has to
        # be evidence that no readback happened, not just that no file appeared.
        self.qd = np.arange(len(self.names), dtype=float) * 0.01
        self.eff = np.arange(len(self.names), dtype=float) * -2.0
        self.n_qd = self.n_eff = 0
        # What `_set_gains` leaves behind: the gains actually in force, which is what a
        # self-contained trace has to carry rather than re-deriving the rule at analysis time.
        self._gains = (np.arange(len(self.names), dtype=float) * 1.0e3 + 2.0e3,
                       np.arange(len(self.names), dtype=float) * 1.0e2 + 5.0e1,
                       np.arange(len(self.names), dtype=float) * 10.0 + 30.0)
        self.robot = SimpleNamespace(get_joint_positions=self._read,
                                     get_joint_velocities=self._read_qd,
                                     get_measured_joint_efforts=self._read_eff)
        self.ctrl = SimpleNamespace(apply_action=self.applied.append)

    def _read(self):
        if self._read_raises:
            raise RuntimeError("physics view dropped mid-readback")
        return self.act.copy()

    def _read_qd(self):
        self.n_qd += 1
        return self.qd.copy()

    def _read_eff(self):
        self.n_eff += 1
        return self.eff.copy()

    def _with_closure(self, q):
        return np.asarray(q, float)

    def _drive_on(self):
        return self._drive


def _run_track(cmd, act, dt=1.0 / 240.0, drive=True, stage="place", read_raises=False, env=None,
               previous="command"):
    """The REAL `morph/robot.py::_apply` and the REAL gain rule over that stub. robot.py imports
    Isaac at module scope, so `_exec_method` is the only way to RUN any of it. The gain rule comes
    from the same source file, not from a copy here: a tolerance derived from a duplicated table
    is a tolerance nothing keeps honest. `os.path`/`os.makedirs` are the REAL ones -- only the
    environment is faked, so a path the code computes is a path it would really compute."""
    out = []
    glb = {"np": np, "math": math, "json": json,
           "os": SimpleNamespace(environ=dict({"TRACK_ERR": "1", "ARM2_FREEZE": "0"}, **(env or {})),
                                 path=os.path, makedirs=os.makedirs),
           "ARM_MODEL": os.path.join(ROOT, "usd", "_arm_model.json"), "HERE": ROOT,
           "ArticulationAction": lambda **kw: SimpleNamespace(**{"joint_velocities": None, **kw}),
           "print": lambda *a, **k: out.append(" ".join(str(x) for x in a))}
    glb.update(_module_consts(("morph", "robot.py"),
                              {"TRACK_GATED_JOINTS", "TRACK_HEADROOM", "TRACK_STEP_CLIP_M",
                               "LUT_REPORT_TOL", "TRACK_TRACE_COLS"}))
    for _m in ("_apply", "_pin_fingers", "_track_tols", "_prismatic_names", "_arm_gain",
               "_wrist_gain", "_track_trace_row"):
        setattr(_TrackDemo, _m, _exec_method(("morph", "robot.py"), _m, glb))
    for _k, _v in _class_consts(("morph", "robot.py"), "RobotMixin",
                                {"_ARM_REFLECTED", "_ARM_LEAVES", "_WRIST", "_ARM_EFFORT"}).items():
        setattr(_TrackDemo, _k, _v)
    demo = _TrackDemo(cmd, act, dt, drive, read_raises)
    if isinstance(previous, str):
        assert previous == "command"
        demo._drv_prev = demo.cmd.copy()
    elif previous is not None:
        demo._drv_prev = np.asarray(previous, float)
    demo._stage = stage
    demo._apply(demo.cmd.copy())
    assert demo.applied, f"`_apply` never reached apply_action, so nothing below is evidence: {out}"
    return demo, out


def _breached(out):
    return sorted(re.findall(r"TRACK BREACH\[[^\]]+\]: (\S+)", "\n".join(out)))


# run19's nine breaches, as commanded/actual pairs, in TRACK_NAMES order.
RUN19_CMD = [0.5637, 0.5637, 0.4601, -0.4601, 0.5337, 0.6470, 0.5045, 0.0000]
RUN19_ACT = [0.5076, 0.5077, 0.4675, -0.4675, 0.5392, 0.6520, 0.5095, -0.0521]


def test_the_track_gate_no_longer_flags_the_undriven_leaf_or_the_lut_passives():
    """`ArmRightJoint_1` is in `_ARM_LEAVES`, so the gain rule hands it kp = 0.0 -- NO position
    drive at all -- and it and `RotationLeftJoint_1` are both four-bar passives written from the
    closure LUT. Their error is the LUT model disagreeing with physics, which is a real thing to
    know and is not command tracking. Both exclusions are derived from the gain rule and the LUT's
    own passive list at runtime, not from a second hardcoded list."""
    _d, out = _run_track(RUN19_CMD, RUN19_ACT)
    got = _breached(out)
    assert "ArmRightJoint_1" not in got, (
        f"a joint the gain rule gives kp = 0 was reported as a tracking failure: {got}")
    assert "RotationLeftJoint_1" not in got, (
        f"a four-bar LUT passive was reported as a tracking failure: {got}")


def test_a_gated_joint_with_no_position_drive_is_dropped_even_when_the_lut_would_not():
    """The kp = 0 rule ON ITS OWN. `_ARM_LEAVES` and the closure LUT's passive list happen to name
    the same joints today, so `ArmRightJoint_1` is dropped by either rule and the test above
    cannot say which one did it. Point `_ARM_LEAVES` at a joint the LUT does NOT own and the gain
    rule is the only thing that can drop it -- which is what stops a leaf added to `_ARM_LEAVES`
    tomorrow from being gated on a position drive it does not have."""
    demo, _out = _run_track(RUN19_CMD, RUN19_ACT)
    was = _TrackDemo._ARM_LEAVES
    assert "ColumnLeftBearingJoint_1" not in demo.clut["passives"], "the LUT owns the test joint"
    try:
        _TrackDemo._ARM_LEAVES = tuple(was) + ("ColumnLeftBearingJoint_1",)
        demo._track_tab = None
        gated = [n for n, _i, _t, _u, k in demo._track_tols() if k == "gate"]
    finally:
        _TrackDemo._ARM_LEAVES = was
    assert "ColumnLeftBearingJoint_1" not in gated, (
        f"a joint the gain rule now hands kp = 0 is still in the gated set: {gated}")
    assert "ColumnRightBearingJoint_1" in gated, (
        f"the control went with it -- this proves nothing about the kp rule: {gated}")


def test_the_track_gate_still_fails_the_genuine_column_breach():
    """The whole point. run19's two column readings are a REAL defect -- a hold pose captured
    before the pre-dock correction and applied after it -- and must survive the correction."""
    got = _breached(_run_track(RUN19_CMD, RUN19_ACT)[1])
    assert got == ["ColumnLeftBearingJoint_1", "ColumnRightBearingJoint_1"], (
        f"expected exactly the two genuine 56mm column breaches, got {got}")


# DELETED: two tests that excused a reading with `max_effort/kp`. That quantity is a SATURATION
# CLIFF, not a tolerance, and neither can be rewritten here. See the commit that removed them.


def test_smooth_wrist_motion_is_graded_against_the_previous_target():
    previous = np.asarray(RUN19_CMD, float)
    cmd = previous.copy()
    act = previous.copy()
    cmd[5] += 0.0045
    act[5] -= 0.0035
    assert abs(float(cmd[5] - act[5])) > 7.5e-3
    got = _breached(_run_track(cmd, act, previous=previous)[1])
    assert "gripper_x_rotation_1" not in got, (
        f"a wrist within 3.5 mrad of the target that had a physics interval false-failed: {got}")


def test_tracking_defers_when_no_valid_previous_target_exists():
    cmd = np.asarray(RUN19_CMD, float)
    act = np.asarray(RUN19_ACT, float)
    act[5] += 0.010
    invalid = (None, cmd[:-1], np.full(cmd.shape, np.nan))
    for previous in invalid:
        _demo, out = _run_track(cmd, act, dt=0.5, previous=previous)
        got = _breached(out)
        assert "gripper_x_rotation_1" not in got, (
            f"tracking graded readback without a valid previous target ({previous!r}): {got}")
        assert any("LUT-PHYSICS DISAGREE" in line for line in out), out
        assert any("gate awaiting previous target" in line for line in out), out
    with tempfile.TemporaryDirectory() as d:
        demo, _out = _run_track(cmd, act, previous=None, env=_trace_env(d))
        wrist = next(row for row in _trace_rows(d) if row["joint"] == "gripper_x_rotation_1")
        assert float(wrist["q_cmd"]) == float(demo.applied[-1].joint_positions[5]), wrist


def test_a_sustained_wrist_error_meaningfully_above_saturation_trips_the_gate():
    """The true negative, without which the previous-target rule merely disables the gate."""
    cmd = list(RUN19_CMD)
    act = list(RUN19_ACT)
    # 10 mrad: twice the drive's own floor
    act[5] = cmd[5] + 0.010
    got = _breached(_run_track(cmd, act, previous=cmd)[1])
    assert "gripper_x_rotation_1" in got, (
        f"a 10 mrad wrist error -- twice the drive's own saturation floor -- passed: {got}")


def test_the_per_joint_tolerances_come_out_of_each_drives_own_numbers():
    """The arithmetic, pinned. tol = 1.5 * (max_effort/kp + v*kd/kp for a prismatic), with
    kd = 2*sqrt(kp*m) from the reflected mass and v = 0.96 m/s:
      columns   1.5 * (1500/3e5 + 0.96*4127.95/3e5) = 1.5 * (5.000 + 13.210) = 27.31 mm
      a1        1.5 * ( 500/3e5 + 0.96*4069.40/3e5) = 1.5 * (1.667 + 13.022) = 22.03 mm
      wrist     1.5 * (  30/6e3)                    = 1.5 *  5.000           =  7.50 mrad
    One number for every joint is what made three of them wrong at once."""
    demo, _out = _run_track(RUN19_CMD, RUN19_ACT)
    tol = {n: t for n, _i, t, _u, _k in demo._track_tols() if _k == "gate"}
    assert set(tol) == {"ColumnLeftBearingJoint_1", "ColumnRightBearingJoint_1", "ArmLeftJoint_1",
                        "gripper_x_rotation_1", "gripper_y_rotation_1"}, sorted(tol)
    assert round(tol["ColumnLeftBearingJoint_1"] * 1000, 2) == 27.31, tol
    assert round(tol["ArmLeftJoint_1"] * 1000, 2) == 22.03, tol
    assert round(tol["gripper_x_rotation_1"] * 1000, 2) == 7.50, tol
    assert len(set(round(v, 6) for v in tol.values())) > 1, (
        "every joint got the same tolerance -- that is the single global constant again")


def test_the_breach_line_prints_metres_as_mm_and_radians_as_mrad():
    """`err ... mm/mrad` on every joint. A prismatic error is METRES; printing it under a unit
    that also names radians is how 56mm of column travel read as a 56 mrad angle for a whole
    review cycle."""
    cmd = list(RUN19_CMD)
    act = list(RUN19_ACT)
    act[5] = cmd[5] + 0.010
    _d, out = _run_track(cmd, act)
    col = [l for l in out if "TRACK BREACH" in l and "ColumnLeftBearingJoint_1" in l]
    wri = [l for l in out if "TRACK BREACH" in l and "gripper_x_rotation_1" in l]
    assert col and "56.10mm" in col[0] and "mrad" not in col[0], (
        f"the prismatic column breach is not printed in mm: {col}")
    assert wri and "10.00mrad" in wri[0] and "mm" not in wri[0].replace("mrad", ""), (
        f"the revolute wrist breach is not printed in mrad: {wri}")


def test_the_lut_physics_disagreement_is_kept_but_under_its_own_name():
    """It is a real signal -- `ContactCylinderJoint_1_1` sat 52mm off its LUT value for the whole
    place -- but it is the four-bar model disagreeing with physics, not the actuated chain failing
    to track, and calling it the latter is what put two passives in a command-tracking gate."""
    _d, out = _run_track(RUN19_CMD, RUN19_ACT)
    lut = "\n".join(l for l in out if "LUT-PHYSICS DISAGREE" in l)
    for n in ("RotationLeftJoint_1", "ArmRightJoint_1", "ContactCylinderJoint_1_1"):
        assert n in lut, f"{n}'s LUT-vs-physics disagreement went unreported: {out}"
    assert "TRACK BREACH" not in lut and "MODEL-PHYSICS DIVERGENCE" not in lut, (
        "the LUT report reuses another gate's marker, so verify_place would grade it as one")
    import verify_place
    assert verify_place.track_gate(lut + "\n") is True, (
        "the LUT diagnostic fails track_gate: it is a report, not a gate")


def test_the_heartbeat_names_the_gate_state_and_the_right_unit():
    """The heartbeat is verify_place's evidence that the check RAN (the heartbeat check in `verify_place.py::check_text`), so it
    has to say what the check is actually watching -- and the same unit bug was in it."""
    _d, out = _run_track(RUN19_CMD, RUN19_ACT, dt=0.5)
    beat = [l for l in out if re.search(r">>> track-err\[[^\]]+\]:", l)]
    assert beat, f"the working readback printed no heartbeat: {out}"
    assert "mm/mrad" not in beat[0], f"the heartbeat still prints both units at once: {beat[0]}"
    assert re.search(r"gate \d+ joints", beat[0]), (
        f"the heartbeat does not say how many joints the gate is watching: {beat[0]}")


def test_the_gate_says_so_when_no_joint_is_position_driven_at_this_stage():
    """Outside `ARM_DRIVE_STAGE` every arm write is a teleport (`_force` in `morph/robot.py::RobotMixin._force`) and the
    drive is a formality, so there is no command tracking to measure. That must be VISIBLE: a
    silent zero-joint gate reads exactly like a clean run, which is the absence-passes shape this
    file exists to stop."""
    _d, out = _run_track(RUN19_CMD, RUN19_ACT, dt=0.5, drive=False)
    assert not _breached(out), "the gate fired on a stage whose joints are teleported, not driven"
    beat = [l for l in out if re.search(r">>> track-err\[[^\]]+\]:", l)]
    assert beat and "gate off" in beat[0], (
        f"the heartbeat does not say the gate is off at this stage: {beat}")


# ===== L9 group 2: the object is graded at `released`, before the hand has left ================
# MEASUREMENT ONLY: a lean on the rim accounts for both errors, and both revert after the backout.

def _place_fn():
    """The IMPLEMENTATION. `place` itself is a thin wrapper whose only job is the `finally` that
    releases the chassis pin (see `test_every_exit_from_place_releases_the_chassis_pin`); every
    checkpoint test below is about the body it delegates to."""
    return _func(("morph", "place", "__init__.py"), "_place_impl")


def _call_names(node):
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            out.append(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return out


def _carry_calls(fn):
    """{tag: lineno} for every literal-tagged `_carry_check` in `place`."""
    return {c.args[1].value: c.lineno for c in ast.walk(fn)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
            and c.func.attr == "_carry_check" and len(c.args) > 1
            and isinstance(c.args[1], ast.Constant)}


def test_the_object_is_measured_again_after_the_backout_completes():
    """`_carry_check(obj, "released")` runs before the backout, so the graded attitude is the one
    the hand is still holding the object into. Add the reading that the shelf actually cares
    about, after every step of the backout has run."""
    fn = _place_fn()
    tags = _carry_calls(fn)
    assert "settled" in tags, (
        f"no `_carry_check(obj, \"settled\")` in `place`: the object's final resting attitude is "
        f"still unmeasured. Checkpoints present: {sorted(tags)}")
    last_step = max(c.lineno for c in ast.walk(fn) if isinstance(c, ast.Call)
                    and getattr(c.func, "id", "") == "frozen_step")
    assert tags["settled"] > last_step, (
        f"the settled reading is at line {tags['settled']}, before the backout's last step at "
        f"{last_step} -- it is measuring the same instant `released` already did")


def test_the_released_checkpoint_is_kept_as_a_diagnostic():
    """It is informative that the object leans while the hand is still there -- that is the whole
    of defect A. It stops being the VERDICT; it does not stop being logged."""
    tags = _carry_calls(_place_fn())
    assert "released" in tags, "the `released` diagnostic was deleted rather than demoted"
    assert tags["released"] < tags.get("settled", 10 ** 9)


def test_the_open_checkpoints_log_the_objects_velocity():
    """Review N could not determine what holds the object at a STATIC 5.6deg for 1.5s: a free
    cylinder at 5.6deg (tip-over 24deg) falls flat in 91ms. The leading hypothesis is PhysX body
    sleep freezing it mid-lean. The numbers settle it, so take them."""
    fn = _place_fn()
    watch = next((n for n in ast.walk(fn) if isinstance(n, ast.FunctionDef)
                  and n.name == "_open_watch"), None)
    assert watch is not None, "`_open_watch` is gone; the open+ checkpoints moved"
    assert "get_linear_velocities" in _call_names(watch), (
        "the open+ checkpoints still record no object velocity, so nothing can tell a frozen body "
        "from one that is genuinely balanced")


def test_the_open_checkpoint_observer_commands_nothing():
    """MEASUREMENT ONLY. `_open_watch` runs inside `open_gripper`'s step loop; a write there is
    a control change wearing an instrument's clothes."""
    fn = _place_fn()
    watch = next(n for n in ast.walk(fn) if isinstance(n, ast.FunctionDef)
                 and n.name == "_open_watch")
    wrote = sorted(set(_call_names(watch)) & {"frozen_step", "_apply", "_force", "set_base",
                                              "step", "open_gripper", "_seat_step"})
    assert not wrote, f"the open+ observer commands the robot: {wrote}"


# Every call in `place` that commands the robot or sets its timing, NAMED rather than counted: a
# census asserted with `==` froze the motion profile and forbade any future timing fix.
PLACE_MOTION_CALLS = frozenset({"frozen_step", "step", "_apply", "_force", "set_base",
                                "_wheel_move", "open_gripper", "_park_ramp", "_seat_step"})


def test_the_settled_measurement_commands_nothing():
    """MEASUREMENT ONLY, stated so it stays checkable. The settled reading is one `_carry_check`
    call; if `_carry_check` cannot command the robot then adding it cannot have changed a motion, a
    target or a timing -- and that stays true when the profile around it is legitimately retimed
    later, which the pinned census made impossible. Same shape as
    `test_the_open_checkpoint_observer_commands_nothing` above, applied to this checkpoint's own
    instrument. WHERE the reading is taken is
    `test_the_object_is_measured_again_after_the_backout_completes`'s job."""
    assert "settled" in _carry_calls(_place_fn()), (
        "the `settled` checkpoint is gone, so there is no settled measurement to judge")
    cc = _func(("morph", "align.py"), "_carry_check")
    wrote = sorted(set(_call_names(cc)) & PLACE_MOTION_CALLS)
    assert not wrote, (
        f"the settled reading's own instrument commands the robot: {wrote} -- a control change "
        f"wearing an instrument's clothes")


# ===== L10: three control defects the tracking instrument exposed ============================
# The feedforward is destroyed for base writes, one joint's jump zeroes every target, the hold is early.

class _BaseDemo:
    """The `Demo` surface `set_base` touches. The chassis write is RECORDED, never performed, so
    a refused write and a performed one are told apart by evidence rather than by reading the
    branch. `_bl = None` skips the cosmetic wheel spin."""

    def __init__(self, drive=True, brake=False, locked=None):
        self._drive = drive
        self._pin_brake = brake
        self._grasp_locked = locked
        # the sentinel the feedforward rides on
        self._drv_prev = np.arange(4, dtype=float)
        self._bl = None
        self.z0 = 0.0
        self.wrote = []
        self.robot = SimpleNamespace(
            get_world_pose=lambda: (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
            set_world_pose=lambda **kw: self.wrote.append(kw),
            set_velocities=lambda v: None)

    def _drive_on(self):
        return self._drive


def _run_set_base(demo, x, y, yaw, force=False):
    """The REAL `morph/robot.py::set_base` over that stub. robot.py imports Isaac at module scope,
    so `_exec_method` is the only way to RUN it; `wrap`/`quat_yaw` are the real ones."""
    sys.path.insert(0, ROOT)
    try:
        from morph.geometry import quat_yaw, wrap
    finally:
        sys.path.pop(0)
    glb = {"np": np, "math": math, "os": SimpleNamespace(environ={}),
           "wrap": wrap, "quat_yaw": quat_yaw,
           "print": lambda *a, **k: None}
    _BaseDemo.set_base = _exec_method(("morph", "robot.py"), "set_base", glb)
    demo.set_base(x, y, yaw, force=force)
    return demo


def test_a_chassis_write_the_grasp_lock_refuses_keeps_the_velocity_feedforward():
    """`set_base` nulls `_drv_prev` and only THEN asks the no-chassis-push rule whether the write
    may happen at all. A refused write does nothing to the chassis and still costs the next
    `_apply` its feedforward -- `v = zeros`, and the drive pays the full `v*kd/kp` velocity lag
    for a write that moved nothing."""
    demo = _run_set_base(_BaseDemo(locked=(0.0, 0.0, 0.0)), 0.5, 0.0, 0.0)
    assert sum(getattr(demo, "_base_block_n", {}).values()) == 1, (
        "the no-chassis-push rule did not refuse the write, so nothing below is evidence")
    assert not demo.wrote, "a REFUSED write still moved the chassis"
    assert demo._drv_prev is not None, (
        "a base write that was refused and did nothing still destroyed the drive feedforward")


def test_a_chassis_write_that_is_performed_reaches_the_chassis_and_the_base_ledger():
    """The true negative for the test above -- that the refusal branch SHORT-CIRCUITS rather than
    `set_base` being inert for everyone -- without pinning what a performed write does to the arm.

    `_drv_prev` was the wrong evidence for that. Nulling a JOINT-command history on a ROOT write is
    the defect review Q names (finding 3): a fine-align or guarded seat that commands the base
    silently destroys the arm's velocity feedforward, and asserting it here made the suite defend
    the thing that has to change. The discriminator that survives decoupling them is the work the
    refusal branch skips -- reaching `set_world_pose` and updating the base ledger `_bl`, which
    `base_pose()` reads while the stage pose is stale for a beat after any write."""
    demo = _run_set_base(_BaseDemo(locked=None), 0.5, 0.0, 0.25)
    assert demo.wrote, "the write never reached set_world_pose, so nothing below is evidence"
    assert demo._bl == (0.5, 0.0, 0.25), (
        f"a performed write left the base ledger at {demo._bl!r}: every reader of `base_pose()` in "
        f"the beat after a write gets the pose this ledger holds, not the stage's")
    refused = _run_set_base(_BaseDemo(locked=(0.0, 0.0, 0.0)), 0.5, 0.0, 0.25)
    assert not refused.wrote and refused._bl is None, (
        f"the no-chassis-push refusal did not short-circuit, so the assertions above say nothing "
        f"about the performed path: wrote={refused.wrote} _bl={refused._bl!r}")


def test_one_joints_jump_no_longer_zeroes_every_joints_velocity_target():
    """`v[:] = 0.0` on a jump commands `qd_target = 0` to joints that are genuinely MOVING. At
    kd 4128 with a column at 0.96 m/s that is a 3963 N braking demand against a 1500 N cap -- a
    full-cap brake pulse on the whole arm, 104 of them in run19. `_arm_ii` here deliberately does
    NOT start at 0: the per-joint mask is computed over the arm slice and has to be scattered back
    to full-vector indices, which a mask applied in slice coordinates would get wrong."""
    demo, _out = _run_track(RUN19_CMD, RUN19_ACT)
    prev = np.asarray(RUN19_CMD, float)
    q = prev.copy()
    # the jump (snap/sync) on ONE arm joint
    q[5] += 0.02
    # genuine motion on another arm joint, well under the 0.005 threshold
    q[2] += 0.001
    # ...and on one outside the arm indices entirely
    q[0] += 0.002
    demo._arm_ii = np.array([2, 3, 4, 5])
    demo._drv_prev = prev.copy()
    demo._drv_jump_n = 0
    demo.applied.clear()
    demo._apply(q)
    v = np.asarray(demo.applied[-1].joint_velocities, float)
    assert demo._drv_jump_n == 1, "the jump was not detected at all, so nothing below is evidence"
    assert float(v[5]) == 0.0, "the joint that actually jumped kept its feedforward"
    assert abs(float(v[2]) - 0.001 / demo.dt) < 1e-6, (
        f"an arm joint moving 1 mm/step was commanded qd_target = {float(v[2]):.2f} instead of "
        f"{0.001 / demo.dt:.2f} because ANOTHER joint jumped -- that is the brake pulse")
    assert abs(float(v[0]) - 0.002 / demo.dt) < 1e-6, (
        f"a joint outside the jump test's own index set was zeroed too: {float(v[0]):.2f}")


def _stmt_lists(node):
    """Every statement list inside `node`, so a block can be sliced out of a file that cannot be
    imported. Membership is DIRECT -- an ancestor's `ast.dump` contains its descendants."""
    for n in ast.walk(node):
        for f in ("body", "orelse", "finalbody"):
            b = getattr(n, f, None)
            if isinstance(b, list) and b and all(isinstance(s, ast.stmt) for s in b):
                yield b


def _predock_block():
    """`place`'s pre-dock statements, from the z-close correction to the `pre-dock-end` carry
    check: the region that captures the dock-leg hold pose AND moves the columns to the slide
    height. The end anchor is that carry check because the pre-dock statements sit directly in
    `if lifted:`, whose own body runs on to the release."""
    def _zc(s):
        return isinstance(s, ast.If) and "'PREDOCK_Z_CLOSE'" in ast.dump(s.test)

    def _cq(s):
        return isinstance(s, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "carry_q" for t in s.targets)

    def _end(s):
        return (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                and getattr(s.value.func, "attr", "") == "_carry_check"
                and any(isinstance(a, ast.Constant) and a.value == "pre-dock-end"
                        for a in s.value.args))

    cands = [b for b in _stmt_lists(_place_fn())
             if any(_zc(s) for s in b) and any(_cq(s) for s in b)]
    assert len(cands) == 1, f"expected ONE pre-dock block owning carry_q, found {len(cands)}"
    b = cands[0]
    starts = [i for i, s in enumerate(b) if _zc(s)]
    assert len(starts) == 1, f"expected ONE PREDOCK_Z_CLOSE gate to slice from, found {len(starts)}"
    ends = [i for i, s in enumerate(b) if _end(s)]
    assert len(ends) == 1 and ends[0] > starts[0], (
        f"expected ONE `_carry_check(obj, 'pre-dock-end')` after the z-close gate to slice to, "
        f"found {len(ends)} at {ends}")
    last_cq = max(i for i, s in enumerate(b) if _cq(s))
    assert starts[0] < last_cq < ends[0], (
        f"the dock-leg hold pose is captured at index {last_cq}, outside the pre-dock window "
        f"({starts[0]}, {ends[0]}) -- the slice no longer contains the capture it exists to run")
    return b[starts[0]:ends[0]]


class _PredockDemo:
    """The `Demo` surface the pre-dock block touches, with a column-height model: the COMMANDED
    columns are the object's height, so the z-close loop that moves them moves the reading too.
    Tracking is perfect on purpose -- the 56 mm is a stale capture, not a tracking error."""

    def __init__(self, q, ih1, ih2):
        self.q = np.asarray(q, float).copy()
        self.dt = 1.0 / 240.0
        self.obj_half_h = 0.035
        self._ih1 = ih1
        self.world = SimpleNamespace(step=lambda **kw: None)
        self.clut = None
        self.robot = SimpleNamespace(get_joint_positions=lambda: self.q.copy())

    def obj_z(self):
        return float(self.q[self._ih1])

    def set_base(self, *a):
        pass

    def _force(self, q):
        pass

    def _apply(self, q):
        self.q = np.asarray(q, float).copy()


def _predock_move_linear(demo, ih1, ih2):
    """`move_linear`'s CLOSED FORM for a base-frame (0,0,dz), as this slice would fly it.

    Stubbed, not real: the real primitive needs the planner, an obstacle set and a live stage,
    and this slice's subject is WHEN `carry_q` is captured relative to the correction -- not what
    the primitive checks. It applies the EQUAL increment `column_ramp` would to both columns,
    through the fake's own `_force`/`_apply`, so the drooping-plant variant still droops and the
    assertion on the final column height still binds.
    """
    def _mv(_self, delta, frame, *, secs=None, on_step=None, **kw):
        dz = float(np.asarray(delta, float)[2])
        q0 = np.asarray(demo.q, float).copy()
        n = max(2, int(float(secs or 0.25) / demo.dt))
        for k in range(n):
            f = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / n)
            qk = q0.copy()
            qk[ih1] = q0[ih1] + f * dz
            qk[ih2] = q0[ih2] + f * dz
            if on_step is not None:
                on_step()
            demo._force(qk)
            demo._apply(qk)
            demo.world.step(render=False)
        return SimpleNamespace(reason="arrived", q_end=np.asarray(demo.q, float).copy())
    return _mv


def _run_predock(slide_z=0.156, q=(0.100, 0.100, 0.300, 0.400), ih1=0, ih2=1, demo=None):
    """The REAL pre-dock statements out of `morph/place/__init__.py`, executed. `place` imports
    Isaac at module scope and is 300 lines long, so an AST slice is the only way to RUN this.
    `demo` swaps in a different plant -- the drooping one below -- over the same statements."""
    out = []
    demo = _PredockDemo(q, ih1, ih2) if demo is None else demo
    obj = SimpleNamespace(
        get_world_poses=lambda: (np.array([[0.0, 0.0, demo.obj_z()]]), None))
    ns = {"np": np, "math": math, "os": SimpleNamespace(environ={}), "HEADLESS": True,
          "print": lambda *a, **k: out.append(" ".join(str(x) for x in a)),
          "self": demo, "obj": obj, "_slide_z": float(slide_z), "obj_idx": 0,
          "_ih1p": ih1, "_ih2p": ih2, "_bx_ch": 0.0, "_by_ch": 0.0, "_byaw_ch": 0.0,
          "move_linear": _predock_move_linear(demo, ih1, ih2)}
    ns.update(_module_consts(("morph", "config.py"), {"COLUMN_MAX"}))
    body = _predock_block()
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 os.path.join("morph", "place", "__init__.py"), "exec"), ns)
    return demo, ns, out


def _assigns(stmt, name):
    return isinstance(stmt, ast.Assign) and any(
        isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store)
        for t in stmt.targets for n in ast.walk(t))


def test_the_pre_dock_height_ramp_is_unconditional():
    """P4.8: `PREDOCK_HEIGHT` shipped "1" and was set nowhere, and its "0" path could not run --
    `_pinned_pd` is assigned only inside the block and read unconditionally after it. The ramp is
    located by the statements it owns, so re-gating it under any name fails here."""
    src = open(os.path.join(ROOT, "morph", "place", "__init__.py"), encoding="utf-8").read()
    assert "PREDOCK_HEIGHT" not in src, "the pre-dock height ramp is gated again"
    # The whole ramp, not just the executable slice: from the base ledger it captures to the
    # `pre-dock-end` carry check.
    body = next(b_ for b_ in _stmt_lists(_place_fn())
                if any(_assigns(s_, "_bx_ch") for s_ in b_))
    i0 = next(i for i, s_ in enumerate(body) if _assigns(s_, "_bx_ch"))
    i1 = next(i for i, s_ in enumerate(body) if isinstance(s_, ast.Expr)
              and isinstance(s_.value, ast.Call)
              and getattr(s_.value.func, "attr", "") == "_carry_check"
              and any(isinstance(a, ast.Constant) and a.value == "pre-dock-end"
                      for a in s_.value.args))
    # Any env read in any branching form, not just `ast.If` spelling "environ": `os.getenv` and a
    # conditional expression both gate the ramp without either.
    def _env(node):
        t = ast.unparse(node)
        return "environ.get" in t or "environ[" in t or "getenv" in t

    gated = [ast.unparse(n.test if not isinstance(n, ast.BoolOp) else n)
             for s_ in body[i0:i1] for n in ast.walk(s_)
             if isinstance(n, (ast.If, ast.IfExp, ast.BoolOp))
             and _env(n.test if not isinstance(n, ast.BoolOp) else n)
             and "PREDOCK_Z_CLOSE" not in ast.unparse(n)
             and "PLACE_PIN_CHASSIS" not in ast.unparse(n)]
    assert gated == [], f"an environment read decides part of the pre-dock ramp again: {gated}"
    # The ramp's step count is a travel-derived expression, not a branch: collapsing it to 1 is a
    # one-frame teleport of the held object that no pose check would see.
    npd = [ast.unparse(s_.value) for s_ in body[i0:i1] if _assigns(s_, "_npd")]
    assert len(npd) == 1, f"expected ONE _npd assignment in the pre-dock ramp, found {npd}"
    assert npd[0].startswith("max(1, int(") and "self.dt" in npd[0] and "IfExp" not in str(
        [type(n).__name__ for n in ast.walk(ast.parse(npd[0], mode="eval"))]), (
        f"the pre-dock step count is `{npd[0]}`: it must stay derived from the travel and `self.dt`")


def test_the_settle_hold_is_not_kinematic():
    """P4.8: `KIN_SETTLE` shipped "0" (profile) against a code default of "1", so every run
    settled with `kin=False` -- the drive commands the pose and nothing force-writes it. The call
    is located by its own tag, so a re-added `kin=` or environment read fails here."""
    src = open(os.path.join(ROOT, "morph", "pick", "descend.py"), encoding="utf-8").read()
    assert "KIN_SETTLE" not in src and "environ" not in src, (
        "the settle reads the environment again")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and "settle_t" in ast.unparse(n))
    holds = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
             and getattr(c.func, "attr", "") == "_hold"]
    assert len(holds) == 1, f"expected ONE settle _hold, found {len(holds)}"
    kws = sorted(k.arg for k in holds[0].keywords)
    assert kws == ["on_step"], (
        f"the settle _hold is called with {kws}: `kin` pins the arm kinematically and the shipped "
        f"profile never asked for it")
    # `kin` is _hold's THIRD POSITIONAL, so keywords alone do not pin it.
    assert len(holds[0].args) == 2, (
        f"the settle _hold takes {len(holds[0].args)} positional arguments "
        f"({[ast.unparse(a) for a in holds[0].args]}): the third is `kin`")
    on_step = next(ast.unparse(k.value) for k in holds[0].keywords if k.arg == "on_step")
    assert on_step.isidentifier(), (
        f"the settle _hold's on_step is `{on_step}`: a callable built here can force-write the "
        f"pose every step, which is what `kin=True` did")


def test_the_dock_leg_hold_pose_is_the_arm_the_z_close_actually_left():
    """Column breaches from stale capture: `carry_q` is captured before the z-close block moves
    BOTH columns by `_err_z`, and the stale pose is handed to `_wheel_move` as the dock-leg hold
    pose, which `morph/navigation.py::NavigationMixin._wheel_move` applies as a STEP with no ramp -- a 56 mm command
    discontinuity, and the object rides that high for the whole 1.9 s dock leg."""
    demo, ns, _out = _run_predock()
    moved = float(demo.q[0])
    assert abs(moved - 0.156) < 1e-9, (
        f"the z-close correction did not put the columns at the slide height ({moved:.4f}); "
        f"nothing below is evidence")
    carry = np.asarray(ns["carry_q"], float)
    assert abs(float(carry[0]) - moved) < 1e-9 and abs(float(carry[1]) - moved) < 1e-9, (
        f"the dock-leg hold pose is the PRE-correction arm: columns {float(carry[0]):.4f} vs the "
        f"{moved:.4f} the z-close left them at -- a {abs(float(carry[0]) - moved) * 1000:.0f} mm "
        f"step commanded with no ramp")
    assert abs(float(carry[3]) - 0.400) < 1e-9, "the rest of the hold pose changed with it"
    # ...and the refreshed capture must still be the one the dock leg is handed: the producer is
    # only fixed if no later assignment re-stales it before the consumer.
    fn = _place_fn()
    dock = next(c.lineno for c in ast.walk(fn) if isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute) and c.func.attr == "_wheel_move"
                and any(isinstance(k.value, ast.Constant) and k.value.value == "dock-leg"
                        for k in c.keywords))
    zc_end = max(s.end_lineno for s in _predock_block()
                 if isinstance(s, ast.If) and "'PREDOCK_Z_CLOSE'" in ast.dump(s.test))
    last = max(a.lineno for a in ast.walk(fn) if isinstance(a, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "carry_q" for t in a.targets)
               and a.lineno < dock)
    assert zc_end < last < dock, (
        f"the last carry_q assignment before the dock leg is at :{last}, outside the window "
        f"({zc_end}, {dock}) -- the dock leg is handed a pose captured before the correction")



# --- L10b item 2: the same capture, under a plant that does NOT track perfectly. ---------------

class _DroopPredockDemo(_PredockDemo):
    """`_PredockDemo` with an IMPERFECT readback: joint `di` sits `droop` BELOW whatever it was
    last commanded, the way a loaded arm sits below its target. The model above cannot express
    this defect at all -- there measured == commanded, so a live capture and a commanded one are
    the SAME vector and the test above passes either way. `_apply` here is the REAL
    `morph/robot.py::_apply`, so `_drv_prev` -- the vector production code last commanded -- is
    written by production code, and the plant is driven from the action that reached
    `apply_action`, not from a second copy of the command kept by this harness."""

    def __init__(self, q, ih1, ih2, di, droop):
        super().__init__(q, ih1, ih2)
        self.di, self.droop = int(di), float(droop)
        self._finger_cmd = self._finger_pin = None
        self._q_lim = False
        self.applied = []
        self.ctrl = SimpleNamespace(apply_action=self.applied.append)
        # what the pre-dock ramp last commanded...
        self.cmd = np.asarray(q, float).copy()
        self._drv_prev = self.cmd.copy()
        self.q = self.cmd.copy()
        # ...and where the arm actually sits
        self.q[self.di] -= self.droop

    def _drive_on(self):
        return True

    def _with_closure(self, q):
        return np.asarray(q, float)

    def _apply(self, q):
        self._real_apply(q)
        self.cmd = np.asarray(self.applied[-1].joint_positions, float).copy()
        self.q = self.cmd.copy()
        self.q[self.di] -= self.droop


def _droop_predock_demo(di=3, droop=0.020, q=(0.100, 0.100, 0.300, 0.400), ih1=0, ih2=1):
    """That plant, carrying the REAL `_apply`. `di` is NOT a column: the z-close loop closes on
    the object height through the columns, so drooping one would change what the correction has
    to solve instead of what the capture reads."""
    assert di not in (ih1, ih2), "the drooping joint must not be a column"
    rglb = {"np": np, "math": math, "os": SimpleNamespace(environ={"ARM2_FREEZE": "0"}),
            "ArticulationAction": lambda **kw: SimpleNamespace(**{"joint_velocities": None, **kw}),
            "print": lambda *a, **k: None}
    _DroopPredockDemo._real_apply = _exec_method(("morph", "robot.py"), "_apply", rglb)
    _DroopPredockDemo._pin_fingers = _exec_method(("morph", "robot.py"), "_pin_fingers", rglb)
    return _DroopPredockDemo(q, ih1, ih2, di, droop)


def test_the_dock_leg_hold_pose_is_the_commanded_arm_not_the_drooped_readback():
    """Refreshing `carry_q` from a LIVE readback after the z-close correction fixes the
    two columns and also adopts load droop, and any transient, in every OTHER arm joint:
    `_wheel_move` (`morph/navigation.py::NavigationMixin._wheel_move`) then HOLDS that measured error as the arm's target for the
    whole 1.9 s dock leg, so the arm sags one more generation from there. The test above cannot
    see it -- its plant tracks perfectly."""
    demo = _droop_predock_demo()
    di, droop = demo.di, demo.droop
    demo, ns, _out = _run_predock(demo=demo)
    cmd = np.asarray(demo.cmd, float)
    live = np.asarray(demo.robot.get_joint_positions(), float)
    assert abs(float(live[di]) - (float(cmd[di]) - droop)) < 1e-12, (
        f"the plant is not drooping ({float(live[di]):.4f} read vs {float(cmd[di]):.4f} "
        f"commanded), so measured == commanded and nothing below is evidence")
    carry = np.asarray(ns["carry_q"], float)
    assert abs(float(carry[0]) - 0.156) < 1e-9 and abs(float(carry[1]) - 0.156) < 1e-9, (
        f"the z-close correction is no longer in the hold pose at all: columns "
        f"{float(carry[0]):.4f}/{float(carry[1]):.4f} vs the 0.156 slide height")
    assert abs(float(carry[di]) - float(cmd[di])) < 1e-9, (
        f"the dock-leg hold pose took joint {di} from the LIVE articulation "
        f"({float(carry[di]):.4f}) instead of the {float(cmd[di]):.4f} the block last COMMANDED "
        f"-- `_wheel_move` holds that {abs(float(carry[di]) - float(cmd[di])) * 1000:.0f} mm of "
        f"droop as the arm's target for the whole dock leg, and it sags again from there")



# ===== T1: the per-step trace the corrected tracking tolerance needs =========================
# Its floor, dwell and baseline are NOT derivable offline. This recorder moves and commands nothing.


def _trace_path(d):
    return os.path.join(d, "track_trace.csv")


def _trace_rows(d):
    with open(_trace_path(d), newline="") as f:
        return list(csv.DictReader(f))


def _trace_env(d, **extra):
    """The trace armed, writing BESIDE `PLACE_RESULTS` -- the path is not configured separately,
    so the file lands wherever this run's results already go and never in the repo."""
    return dict({"TRACK_TRACE": "1", "PLACE_RESULTS": os.path.join(d, "place_results.json")},
                **extra)


def test_the_trace_writes_and_reads_nothing_at_all_until_its_flag_is_set():
    """Off by default and FREE when off, with the on-case as its positive control: an "it wrote
    no file" assertion passes just as happily against a recorder that never runs at all. The two
    counters are the half that matters at 240 Hz -- velocities and measured efforts have no other
    source in `_apply`, so arming the trace is the only thing that may ever take them."""
    with tempfile.TemporaryDirectory() as d:
        off, _out = _run_track(RUN19_CMD, RUN19_ACT,
                               env={"PLACE_RESULTS": os.path.join(d, "place_results.json")})
        assert off.applied, "`_apply` never reached apply_action, so nothing below is evidence"
        assert not os.path.exists(_trace_path(d)), (
            f"a run with no TRACK_TRACE in its environment still wrote {_trace_path(d)}")
        assert (off.n_qd, off.n_eff) == (0, 0), (
            f"the unflagged run took {off.n_qd} velocity and {off.n_eff} effort readbacks -- an "
            f"instrument that is off is still perturbing the 240 Hz step it claims not to touch")
    with tempfile.TemporaryDirectory() as d:
        on, _out = _run_track(RUN19_CMD, RUN19_ACT, env=_trace_env(d))
        assert os.path.exists(_trace_path(d)), (
            f"TRACK_TRACE=1 produced no trace beside PLACE_RESULTS: {sorted(os.listdir(d))}")
        assert _trace_rows(d), "the trace file has a header and no rows"
        assert (on.n_qd, on.n_eff) == (1, 1), (
            f"the armed run took {on.n_qd} velocity and {on.n_eff} effort readbacks for one step "
            f"-- it must take exactly one of each, and it must reuse the tracking check's own "
            f"position read rather than adding a third")


def test_the_trace_carries_every_field_for_every_watched_joint_as_actually_written():
    """One row per watched joint per step, and each number is what `_apply` PUT IN THE ACTION,
    not a recomputation of it. `kp`/`kd`/`max_effort` come off the gain store `_set_gains` leaves
    behind, so the trace is self-contained and the analysis that finally settles `floor_j` does
    not have to re-derive the stage's gain rule to read it."""
    with tempfile.TemporaryDirectory() as d:
        demo, _out = _run_track(RUN19_CMD, RUN19_ACT, env=_trace_env(d))
        rows = _trace_rows(d)
        watched = demo._track_tols()
        assert len(rows) == len(watched) > 0, (
            f"{len(rows)} rows for {len(watched)} watched joints in one step")
        want = {"step", "stage", "joint", "kind", "q_cmd", "q_act", "qd_meas", "qd_target",
                "v_written", "effort_meas", "kp", "kd", "max_effort"}
        assert want <= set(rows[0]), f"missing from the trace: {sorted(want - set(rows[0]))}"
        act = demo.applied[-1]
        qa = np.asarray(act.joint_positions, float)
        qv = np.asarray(act.joint_velocities, float)
        by = {r["joint"]: r for r in rows}
        assert sorted(by) == sorted(n for n, _i, _t, _u, _k in watched), (
            f"the trace does not name the watched joints: {sorted(by)}")
        for n, i, _t, _u, kind in watched:
            r = by[n]
            assert r["stage"] == "place" and r["kind"] == kind and int(r["step"]) == 1, r
            assert abs(float(r["q_cmd"]) - float(qa[i])) < 1e-12, (
                f"{n}: trace q_cmd {r['q_cmd']} is not the {float(qa[i])} the action carried")
            assert abs(float(r["q_act"]) - float(demo.act[i])) < 1e-12, r
            assert abs(float(r["qd_meas"]) - float(demo.qd[i])) < 1e-12, r
            assert abs(float(r["v_written"]) - float(qv[i])) < 1e-12, (
                f"{n}: trace v_written {r['v_written']} is not the {float(qv[i])} feedforward the "
                f"action carried")
            assert abs(float(r["effort_meas"]) - float(demo.eff[i])) < 1e-12, r
            assert abs(float(r["kp"]) - float(demo._gains[0][i])) < 1e-9, r
            assert abs(float(r["kd"]) - float(demo._gains[1][i])) < 1e-9, r
            assert abs(float(r["max_effort"]) - float(demo._gains[2][i])) < 1e-9, r


def test_a_step_with_no_previous_target_records_qd_target_as_absent_not_as_zero():
    """The field the whole recorder exists for. `set_base` nulls `_drv_prev` in `morph/robot.py::RobotMixin.set_base`, so
    `_apply` falls to `v = zeros(len(q))` and writes a velocity target of 0.0 to every joint --
    an unbraked park ramp runs ~240 such steps. A trace that logs that as `qd_target = 0.0` says
    the joint was told to hold still, when in fact nobody told it anything, and the `(kd/kp)*|v|`
    term of the corrected tolerance cannot be told apart from a genuine hold. The genuine hold is
    the control: with a previous target present and the command unchanged, 0.0 is a real
    commanded zero and must be recorded as one."""
    with tempfile.TemporaryDirectory() as d:
        demo, _out = _run_track(RUN19_CMD, RUN19_ACT, env=_trace_env(d))
        assert demo._drv_prev is not None, "the first step left no previous target to hold still"
        # A genuine commanded zero: the same command again, differenced against a real history.
        demo.applied.clear()
        demo._apply(demo.cmd.copy())
        held = {r["joint"]: r for r in _trace_rows(d) if int(r["step"]) == 2}
        gated = [n for n, _i, _t, _u, k in demo._track_tols() if k == "gate"]
        assert gated, "no joint is gated, so nothing below is evidence"
        for n in gated:
            assert held[n]["qd_target"] == "0.0", (
                f"{n}: a REAL commanded zero was not recorded as 0.0 but as "
                f"{held[n]['qd_target']!r} -- if every step reads blank the field says nothing")
        # ...and now the feedforward the base write destroys.
        demo._drv_prev = None
        demo.applied.clear()
        demo._apply(demo.cmd.copy())
        blind = {r["joint"]: r for r in _trace_rows(d) if int(r["step"]) == 3}
        assert np.allclose(np.asarray(demo.applied[-1].joint_velocities, float), 0.0), (
            "the no-feedforward branch did not write zeros, so nothing below is evidence")
        for n in gated:
            assert blind[n]["qd_target"] == "", (
                f"{n}: a step with no previous target to difference against recorded "
                f"qd_target={blind[n]['qd_target']!r} -- indistinguishable from the commanded "
                f"zero above, which is the feedforward bug's entire signature")
            assert blind[n]["v_written"] == "0.0", (
                f"{n}: v_written must still say what the action really carried ("
                f"{blind[n]['v_written']!r}), or the trace loses the zeros the drive was fed")



# ---------------------------------------------------------------------------------------------
# The pick stage WALKS the grasp-candidate set instead of retrying one candidate twice, since the
# retry is identical. These pin that it offers a DIFFERENT configuration and stops at the latch.


class _WalkDemo:
    def _commanded_fingers(self):
        return None

    """The `Demo` surface `pick` touches once the per-attempt sequence is one call.

    `_pick_attempt` is stubbed: what is under test is the WALK -- which configuration each attempt
    is handed, in what order, and when it stops -- not the stage sequence an attempt runs."""

    def __init__(self, fails=1, latch_at=None, cands=(1.0, 2.0, 3.0, 4.0), usable=True):
        self.tried = []                  # the candidate commanded per attempt (None = the nominal)
        # this many attempts fail before one succeeds
        self.fails = fails
        # 1-based attempt that trips the checkpoint latch
        self.latch_at = latch_at
        self.cands = [np.full(8, c) for c in cands]
        self.fan_built = 0
        self.recovered = 0
        self.events = []                 # "attempt" / "recover" / "fan", in the order they happen
        self.tallied = []                # the `attempt=` label of every books-closing tally
        # what `_pick_attempt` leaves in `_cand_usable`
        self.usable = usable
        self._safety_abort = False
        self._plan_fallbacks = {}

    def _pick_attempt(self, obj_idx, status=None):
        self.events.append("attempt")
        self.tried.append(getattr(self, "_grasp_cand", None))
        # the real one sets this from `st.alive`
        self._cand_usable = self.usable
        n = len(self.tried)
        if self.latch_at == n:
            # execute.py: a checkpoint rejected this configuration
            self._safety_abort = True
        return n > self.fails

    def _grasp_candidates(self, obj_idx):
        self.events.append("fan")
        self.fan_built += 1
        return list(self.cands)

    def _recover(self):
        self.events.append("recover")
        self.recovered += 1

    def _fallback_tally(self, obj_idx, attempt=None):
        self.tallied.append(attempt)
        self._plan_fallbacks = {}        # the real one prints AND CLEARS


def _run_walk(fan="1", demo_cls=_WalkDemo, **kw):
    """The REAL `pick` over `demo_cls`; (demo, returned held, printed lines).

    `GRASP_FAN` is OFF in production, so the walk tests must turn it on explicitly. `fan=None`
    leaves it UNSET, which is what exercises the default -- passing an explicit "0" would pass
    whatever the default became."""
    out = []
    glb = {"os": os, "np": np,
           "print": lambda *a, **k: out.append(" ".join(str(x) for x in a))}
    demo_cls.pick = _exec_method(("morph", "pick", "__init__.py"), "pick", glb)
    demo = demo_cls(**kw)
    _prev = os.environ.get("GRASP_FAN")
    if fan is None:
        # exercise the DEFAULT, not an explicit "0"
        os.environ.pop("GRASP_FAN", None)
    else:
        os.environ["GRASP_FAN"] = fan
    try:
        held = demo.pick(0)
    finally:
        if _prev is None:
            os.environ.pop("GRASP_FAN", None)
        else:
            os.environ["GRASP_FAN"] = _prev
    return demo, held, out


def test_the_walk_is_off_by_default_so_a_stale_frame_cannot_be_commanded():
    """The fan is solved in the arm-root frame as it stands, but `_pick_attempt` re-docks and snaps
    the base before a candidate is commanded, so a base-relative q8 lands elsewhere in the world.
    Until that is fixed the walk must not run unless asked for."""
    _prev = os.environ.pop("GRASP_FAN", None)
    try:
        demo, held, out = _run_walk(fan=None, fails=2)
    finally:
        if _prev is not None:
            os.environ["GRASP_FAN"] = _prev
    assert held is False, f"the nominal failed and no candidate may be walked: {held}"
    assert len(demo.tried) == 1, (
        f"only the NOMINAL attempt may run with the fan off, got {len(demo.tried)}")
    assert demo.fan_built == 0, f"the fan must not even be built with the gate off: {demo.fan_built}"


def _commanded(demo):
    """What each attempt was actually handed, comparable -- the nominal reads as None."""
    return [None if c is None else tuple(np.round(np.asarray(c, float), 9)) for c in demo.tried]


def test_the_pick_stage_walks_the_candidate_set_when_the_nominal_grasp_fails():
    """A failed pick must not be re-attempted identically --
    the walk must hand each attempt the next candidate instead."""
    demo, held, _out = _run_walk(fails=2)
    assert held is True, f"the third attempt succeeded, so pick must report held: {held}"
    assert len(demo.tried) == 3, (
        f"expected the nominal plus two candidates, got {len(demo.tried)} attempt(s)")
    assert demo.tried[0] is None, "the first attempt must be the NOMINAL grasp, not a candidate"
    assert np.array_equal(demo.tried[1], demo.cands[0]), "candidates must be walked best-first"
    assert np.array_equal(demo.tried[2], demo.cands[1]), "candidates must be walked best-first"


def test_every_attempt_in_the_walk_commands_a_different_configuration():
    """`iterating candidates is NOT the same as retrying a rejected one` -- which is only true if
    the walk never re-offers a configuration it has already commanded. A walk that repeats one is
    the identical retry it replaces, wearing a loop."""
    demo, _held, _out = _run_walk(fails=99, cands=(1.0, 2.0, 3.0, 4.0))
    cmds = _commanded(demo)
    assert len(cmds) > 2, f"nothing was walked, so there is nothing to prove distinct: {cmds}"
    assert len(set(cmds)) == len(cmds), (
        f"the walk commanded the same configuration more than once: {cmds}")


def test_the_candidate_fan_is_never_built_when_the_nominal_grasp_succeeds():
    """CONSTRAINT 1, the laziness one. `grasp_candidates` multiplies `ik_candidates` by the fan
    size and its cost scales with the obstacle count (3.1 s offline with only two boxes). Built
    eagerly it would slow every SUCCESSFUL cycle to buy nothing."""
    demo, held, _out = _run_walk(fails=0)
    assert held is True and len(demo.tried) == 1, (
        f"the nominal grasp succeeded, so nothing else may run: {len(demo.tried)} attempt(s)")
    assert demo.fan_built == 0, (
        "the candidate fan was built on a SUCCESSFUL pick -- the sweep is paid for on every cycle "
        "and buys nothing")


def test_the_fan_is_built_once_the_nominal_grasp_has_failed():
    """The control for the test above, and the per-candidate-rebuild guard.

    Walk the WHOLE set: with one candidate walked `fan_built == 1` holds even if the fan is rebuilt
    inside the loop, so a rebuild-per-candidate mutation reads identical. Four walked candidates
    make it 4."""
    demo, _held, _out = _run_walk(fails=99, cands=(1.0, 2.0, 3.0, 4.0))
    assert len(demo.tried) == 5, (
        f"precondition: the nominal plus all four candidates must be walked, or one build and one "
        f"build-per-candidate are the same number; got {len(demo.tried)} attempt(s)")
    assert demo.fan_built == 1, (
        f"four candidates were walked and the fan was built {demo.fan_built} time(s) -- once is "
        f"the whole point, the sweep must not be repeated per candidate")


def test_a_checkpoint_safety_abort_on_the_nominal_grasp_offers_no_candidates_at_all():
    """The latch is RUN-scoped and halts the run; `run_cycle` already suppresses its retry for it.
    A walk that keeps going past it defeats exactly the mechanism that stopped the retry."""
    demo, held, _out = _run_walk(fails=99, latch_at=1)
    assert held is False, f"nothing was held: {held}"
    assert len(demo.tried) == 1, (
        f"{len(demo.tried)} attempts after a checkpoint SAFETY ABORT -- the latch halts the run, "
        f"so the walk must stop with it")
    assert demo.fan_built == 0, (
        "the fan was built after a checkpoint abort -- a sweep whose every candidate the latch "
        "forbids is pure cost")


def test_the_walk_stops_at_the_candidate_that_trips_the_safety_latch():
    """The latch can arm part-way THROUGH the walk. From that point on the run is halting, so the
    remaining candidates must not be commanded either."""
    demo, _held, _out = _run_walk(fails=99, latch_at=2)
    assert demo.fan_built == 1, "precondition: the nominal failed without the latch, so a fan ran"
    assert len(demo.tried) == 2, (
        f"the walk made {len(demo.tried)} attempts -- it continued past the candidate whose "
        f"checkpoint rejected a configuration")


def test_the_walk_reaches_a_success_that_sits_LAST_in_the_candidate_set():
    """The walk must not depend on the fan's ORDER being right.

    `ArmModel.grasp_candidates` ships no yaw-deviation term: it prices yaw only through the
    travel-from-`q_now` already in `ik_candidates`' score, and that was measured on ONE pose
    family. Where the turret already points along the orbit, a large deviation costs little travel
    and can outrank a small one -- so the ranking is a hint. The walk survives a wrong hint only by
    trying them ALL, which is exactly `until one reaches SUSTAINED TOUCH`. A `[:1]`, a top-N cap or
    a `break` on the first failure would each make the ordering load-bearing."""
    demo, held, _out = _run_walk(fails=4, cands=(1.0, 2.0, 3.0, 4.0))
    assert held is True, (
        "the grasp that worked was the LAST candidate and the walk never reached it -- the fan's "
        "ranking has been made load-bearing, and it carries no yaw term to bear it")
    assert len(demo.tried) == 5, (
        f"expected the nominal plus all four candidates, got {len(demo.tried)} attempt(s)")
    assert np.array_equal(demo.tried[-1], demo.cands[-1]), (
        f"the walk stopped short of the last candidate: {_commanded(demo)}")


def test_the_walk_reports_a_reason_of_its_own():
    """A pick that quietly makes four attempts instead of one is indistinguishable in the log from
    one that needed none -- the same defect the fallback counter exists for."""
    _demo, _held, out = _run_walk(fails=2)
    said = [l for l in out if "candidate" in l.lower()]
    assert said, f"the candidate walk printed nothing that names it: {out}"


def test_the_fan_is_built_from_the_pose_the_candidates_will_actually_face():
    """`_recover` teleports every unplaced object back to its spawn, upright. The failure this
    feature exists for is the one that SHOVED the object, so a fan built before the recover orbits
    the shoved position while every candidate attempt faces the reset one."""
    demo, _held, _out = _run_walk(fails=99)
    assert "fan" in demo.events and "recover" in demo.events, demo.events
    assert demo.events.index("recover") < demo.events.index("fan"), (
        f"the fan was built before the recover -- it orbits the pose the failed attempt left, not "
        f"the one the candidates start from: {demo.events}")


def test_every_attempt_after_a_failure_starts_from_a_recovered_pose():
    """The walk mirrors `run_cycle`'s retry, which recovers between attempts: a shoved object and a
    half-driven robot are the state a failed attempt leaves, and the next candidate would inherit
    it. Pinned as the whole sequence, so neither recover can be dropped."""
    demo, _held, _out = _run_walk(fails=99)
    assert demo.events == ["attempt", "recover", "fan"] + ["attempt", "recover"] * 4, (
        f"a failed attempt was not followed by a recover: {demo.events}")


def test_no_candidate_is_walked_when_the_reach_would_discard_it():
    """`_plan_approach` is the only reader of `_grasp_cand`, and only an attempt that reaches the
    reach runs it. After a refusal before the reach (no dock corridor, no baked trajectory) the
    walk would command four candidates nothing reads: four IDENTICAL extra attempts."""
    demo, held, out = _run_walk(fails=99, usable=False)
    assert held is False, held
    assert len(demo.tried) == 1, (
        f"{len(demo.tried)} attempts made with a candidate nothing can consume -- every one after "
        f"the first is the identical retry the walk exists to replace")
    assert demo.fan_built == 0, (
        "the fan was swept for candidates the reach will discard -- the most expensive thing in "
        "the pick, paid for nothing")
    assert [l for l in out if "candidate" in l.lower()], (
        f"the walk was skipped in silence, so the log cannot tell it from a pick that needed no "
        f"candidates: {out}")


def test_each_attempt_in_the_walk_closes_its_own_fallback_tally():
    """`_fallback_tally`'s contract: called between a failed attempt and the next, `so two attempts
    never merge into one number`. Five attempts inside one `pick()` merge five attempts' degrades
    into one line otherwise. Between attempts only -- the last one is closed by `run_cycle`'s
    cycle-end tally, exactly as the retry's second attempt is."""
    demo, _held, _out = _run_walk(fails=99)
    assert len(demo.tried) == 5, f"precondition: five attempts, got {len(demo.tried)}"
    assert len(demo.tallied) == 4, (
        f"{len(demo.tallied)} tally line(s) for 5 attempts -- each attempt but the last must close "
        f"its own books, or their degrades merge into one number: {demo.tallied}")
    assert len(set(map(str, demo.tallied))) == 4, (
        f"two attempts were tallied under the same label, so the log cannot tell them apart: "
        f"{demo.tallied}")


def test_the_successful_pick_tallies_nothing_of_its_own():
    """The no-regression: one attempt is one line, printed by `run_cycle` at cycle end. A tally
    from inside a pick that never walked would double every clean cycle's lines."""
    demo, held, _out = _run_walk(fails=0)
    assert held is True and not demo.tallied, (
        f"a pick that needed no candidate closed the books {len(demo.tallied)} time(s)")


# --- what ONE attempt clears on entry and leaves behind ----------------------------------------


class _St:
    """`_PickState` stands in: only the attribute carrier is used here, not its `__slots__`."""

    def __init__(self, obj_idx, status):
        self.obj_idx, self.status = obj_idx, status
        self.alive, self.held = True, False


class _AttemptDemo:
    """The `Demo` surface `_pick_attempt` touches up to the fallen-object early return.

    Everything past that return is stage code with tests of its own; what is under test here is the
    per-attempt state `_pick_attempt` owns -- what it clears on entry, and what it leaves for
    `pick` to read once it has failed."""

    @contextlib.contextmanager
    def _time_stage(self, stage_name, file_line=None):
        yield

    def __init__(self, refuse_at=None, goal=None):
        # "setup" (no corridor) or "pose" (no bake), else neither
        self.refuse_at = refuse_at
        self.restored, self.stages, self.releases = 0, [], []
        if goal is not None:
            self._grasp_goal = goal

    def _pick_setup(self, st):
        self.stages.append("setup")
        st.obj, st.alive = "obj", self.refuse_at != "setup"

    def _pick_pose(self, st):
        self.stages.append("pose")
        st.alive = self.refuse_at != "pose"
        st.restore_bystanders = self._restore

    def pin_chassis(self, on, why):
        self.releases.append(why)

    def _restore(self):
        self.restored += 1

    def _obj_state(self, st, tag):
        pass

    def _obj_R(self, obj):
        return np.diag([1.0, 1.0, 0.0])          # axis.z 0: lying on its side -> the early return


def _run_attempt(**kw):
    """The REAL `_pick_attempt` over the stubbed stages; (demo, returned held, printed lines)."""
    out = []
    glb = {"np": np, "os": os, "_PickState": _St,
           "print": lambda *a, **k: out.append(" ".join(str(x) for x in a))}
    _AttemptDemo._release_chassis = _exec_method(("morph", "pick", "__init__.py"), "_release_chassis", {})
    _AttemptDemo._pick_attempt = _exec_method(("morph", "pick", "__init__.py"),
                                              "_pick_attempt", glb)
    demo = _AttemptDemo(**kw)
    return demo, demo._pick_attempt(0), out


def test_an_attempt_that_plans_nothing_leaves_no_stale_goal_for_the_fan_to_orbit():
    """`_plan_approach` is the only writer of `_grasp_goal` and nothing cleared it between picks.
    An attempt that fails BEFORE the reach -- the fallen-object return here, or a refusal --
    then leaves the PREVIOUS object's grasp standing, and the fan orbits a hand pose belonging to
    something else: a full IK/collision sweep, then up to four wasted nav+pose attempts."""
    demo, held, _out = _run_attempt(goal=np.full(8, 5.0))     # the last object's planned grasp
    assert held is False and demo.restored == 1, "precondition: the fallen-object early return"
    assert getattr(demo, "_grasp_goal", None) is None, (
        f"the attempt planned nothing and left {getattr(demo, '_grasp_goal', None)} behind -- the "
        f"fan would orbit the grasp of whatever object was picked last")


def test_the_attempt_reports_whether_a_candidate_could_have_been_consumed_at_all():
    """Only the reach reads a candidate, so an attempt refused before it (no dock corridor at
    setup, no baked trajectory at pose) must report `_cand_usable` False and `pick` must not walk
    the fan for it."""
    for refuse_at in (None, "setup", "pose"):
        demo, held, _out = _run_attempt(refuse_at=refuse_at)
        assert held is False, "precondition: the fallen-object early return, or the refusal"
        assert demo._cand_usable is False, (refuse_at, demo._cand_usable)
    demo, _, _ = _run_attempt(refuse_at="setup")
    assert demo.stages == ["setup"] and demo.releases == [], "refused before the dock pin: no pose, nothing to release"
    demo, _, _ = _run_attempt(refuse_at="pose")
    assert demo.stages == ["setup", "pose"] and demo.releases == ["no-bake"], "refused past the pin: released"
    demo, _, _ = _run_attempt()
    assert demo.releases == ["fallen-object"], demo.releases


class _ConsumptionDemo(_WalkDemo):
    """Enough stage surface to run the real pick, attempt, reach, and approach in one chain."""

    def _bake_blocked(self, tr, upto, grasp_names=()):
        # this fake's subject is the candidate, not the geometry
        return None

    def _arm_q8(self):
        # `move_to_pose` reads the measured pose to decide arrival. This fake's subject is which
        # GOAL the approach consumes, so the arrival verdict is not load-bearing here.
        return np.zeros(8)

    def _reach_abort(self, st):
        # The reach ends the attempt on a refusal now that there is no bake to fall back to.
        st.alive = False

    @contextlib.contextmanager
    def _time_stage(self, stage_name, file_line=None):
        yield

    def __init__(self):
        super().__init__(fails=1, cands=(3.0,))
        self.cand = self.cands[0]
        self.planned = []
        self.obj_rel = 0
        # `names` as well as `frames`: the reach reads the baked turret out of it by name.
        self.traj = {"reach": {"frames": np.zeros((31, 1)), "idx": np.array([0]),
                               "names": ["BaseJoint_1"]}}
        # The LUT branch is the ONLY branch now -- the guard was always true and went with the dead path
        # it protected -- so this fake models the real one instead of dodging it.
        self.idx = {n: i for i, n in enumerate(_ARM_MODEL.Q8)}
        self.dt = 1.0 / 240.0
        self.obj_half_h = 0.09
        self.world = types.SimpleNamespace(step=lambda **kw: None)
        self.robot = types.SimpleNamespace(
            get_joint_positions=lambda: np.zeros(len(_ARM_MODEL.Q8)))

    def pin_chassis(self, on, tag=None):
        pass

    def base_ledger(self):
        return 0.0, 0.0, 0.0

    def _pinch(self):
        return np.array([0.0, 0.0, 0.5])

    def _a1_clearing_chassis(self, dh, margin=0.02, th=0.0):
        # no floor: the fold is whatever the goal carries
        return None

    def _solve_lowest(self, *a, **kw):
        # no solve -> the approach pose is flown as-is
        return None

    def _arm_base_world(self):
        return np.zeros(3), np.eye(3)

    def _descent_handoff(self, obj, tag):
        pass

    def _reach_lowest(self, obj, label=None):
        pass

    def _pick_setup(self, st):
        st.obj = types.SimpleNamespace(
            get_world_poses=lambda: (np.array([[1.0, 0.0, 0.09]]), None))

    def _pick_pose(self, st):
        st.O = st.dock = st.park = None
        st.qg = np.zeros(8)
        st.replay = True
        st.pin_bystanders = lambda: None
        st.restore_bystanders = lambda: None

    def _noop(self, *args, **kw):
        pass

    _obj_state = _noop
    _pick_settle = _noop
    _pick_descend = _noop
    _pick_seat_guarded = _noop
    _settle_hand = _noop
    _pick_close = _noop
    _pad_trace = _noop
    _pick_seat = _noop
    _pick_capture = _noop
    pin_chassis = _noop

    def _obj_R(self, obj):
        return np.eye(3)

    def _traj_q8(self, tr, frame_i):
        return np.full(8, 9.0)

    def _object_relative_goal(self, goal, obj_idx):
        self.obj_rel += 1
        return np.full(8, 7.0)

    def _goal_set(self, goal, excl, held):
        return goal

    def _plan_arm(self, goal, **kw):
        goal = np.asarray(goal, float)
        self.planned.append(goal.copy())
        return None if np.allclose(goal, 7.0) else [goal]

    def _execute_arm_path(self, wps, *args, **kw):
        return np.asarray(wps[-1], float)

    def _fallback(self, reason):
        self._plan_fallbacks[reason] = self._plan_fallbacks.get(reason, 0) + 1

    def _finite(self, tag):
        # TRUE now: the planned reach lives INSIDE this guard, so False would skip the `_plan_approach`
        # call this test exists to observe.
        return True

    def _pick_lift(self, st):
        st.held = getattr(self, "_grasp_cand", None) is not None

def _run_consumption():
    """The real candidate path through pick, attempt, reach, and approach."""
    out = []
    pr = lambda *a, **k: out.append(" ".join(str(x) for x in a))
    _ConsumptionDemo._release_chassis = _exec_method(("morph", "pick", "__init__.py"), "_release_chassis", {})
    _ConsumptionDemo._pick_attempt = _exec_method(
        ("morph", "pick", "__init__.py"), "_pick_attempt",
        {"np": np, "os": os, "_PickState": _St, "print": pr})
    _ConsumptionDemo._pick_reach = _exec_method(
        ("morph", "pick", "reach.py"), "_pick_reach",
        {"np": np, "os": os, "math": math, "print": pr, "commanded_fingers": _EXEC_SHIMS["commanded_fingers"],
         "KNOWN": {"object_radius": 0.040},
         "ArmModel": _ARM_MODEL, "A1_MAX": 0.50, "COLUMN_MAX": 1.27, "HEADLESS": True})
    _appr_glb = {"np": np, "os": os, "ARM_PLANNER": True, "print": pr,
                 "move_to_pose": _move_to_pose, "Outcome": _Outcome}
    _ConsumptionDemo._plan_approach = _exec_method(
        ("morph", "pick", "approach.py"), "_plan_approach", _appr_glb)
    # The real derivation too: the candidate this test follows is CONSUMED there, so a stub would
    # make the consumption it asserts unobservable.
    _ConsumptionDemo._approach_goal = _exec_method(
        ("morph", "pick", "approach.py"), "_approach_goal", _appr_glb)
    # OBJ_GOAL=2 is the SHIPPED profile and is what makes the goal the planner's own: any other value
    # writes the baked turret and wrist into it, rewriting the candidate this test follows.
    _prev = os.environ.get("OBJ_GOAL")
    os.environ["OBJ_GOAL"] = "2"
    try:
        demo, held, pick_out = _run_walk(fan="1", demo_cls=_ConsumptionDemo)
    finally:
        if _prev is None:
            os.environ.pop("OBJ_GOAL", None)
        else:
            os.environ["OBJ_GOAL"] = _prev
    return demo, held, out + pick_out


def test_pick_candidate_is_consumed_by_the_real_approach_path():
    """Run the real walk, attempt, reach, and approach with GRASP_FAN=1. The nominal solve must
    fail, then the candidate `pick` sets must be the next goal `_plan_approach` sends to planning;
    isolated writer and reader tests do not prove this handoff executes."""
    demo, held, _out = _run_consumption()
    assert held is True, "the candidate attempt never completed"
    assert len(demo.planned) == 2, f"expected nominal and candidate plans, got {demo.planned}"
    assert np.allclose(demo.planned[0], 7.0), f"the nominal plan changed: {demo.planned[0]}"
    assert np.array_equal(demo.planned[1], demo.cand), (
        f"pick set {demo.cand}, but the real approach planned {demo.planned[1]}")
    assert demo.obj_rel == 1, "the candidate was re-derived through the nominal object-relative solve"


# --- the fan itself: the orbit axis, and what it refuses to offer ------------------------------


class _FanDemo:
    """The `Demo` surface `_grasp_candidates` touches, doubling as its `ArmModel`.

    The base is TILTED on purpose. `grasp_candidates`' `axis` defaults to the IK frame's own +z,
    and on a tilted base that is NOT world-up -- the whole reason a caller has to pass it."""

    TILT = 0.30                                  # rad about x: enough that the two axes differ
    hand_link = "hand"

    def __init__(self, entries):
        c, s = math.cos(self.TILT), math.sin(self.TILT)
        self.bR = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], float)
        self.bp = np.array([1.0, 2.0, 0.5])
        self.o_w = np.array([1.5, 2.5, 0.9])
        self.entries = entries
        self.seen = {}
        self._plan_fallbacks = {}
        self._grasp_goal = np.zeros(8)
        self.objs = {0: SimpleNamespace(
            get_world_poses=lambda: (np.array([self.o_w]), None))}

    # --- the ArmModel surface ---
    def fk(self, q8):
        return {"hand": (np.array([0.4, 0.0, 0.3]), np.eye(3))}

    def grasp_candidates(self, pos, R, obj, **kw):
        self.seen = dict(kw, pos=pos, R=R, obj=obj)
        return list(self.entries)

    # --- the Demo surface ---
    def _arm_model(self):
        return self

    def _arm_base_world(self):
        return self.bp, self.bR

    def _arm_bounds(self):
        return -np.ones(8), np.ones(8)

    def _arm_q8(self):
        return np.zeros(8)

    def _arm_obstacles(self, exclude_names=()):
        return [((np.zeros(3), np.ones(3)), "world")]


def _entry(fill, yaw, score=0.0):
    """One `grasp_candidates` entry: ((q8, err_p, err_r), clearance, score, yaw)."""
    return ((np.full(8, float(fill)), 0.0, 0.0), 0.05, float(score), float(yaw))


def _run_fan(entries):
    """The REAL `_grasp_candidates`; (demo, the configurations it offers, printed lines)."""
    out = []
    glb = {"np": np, "os": os, "math": math, "MARGIN": 0.02,
           "print": lambda *a, **k: out.append(" ".join(str(x) for x in a))}
    _FanDemo._grasp_candidates = _exec_method(("morph", "pick", "approach.py"),
                                              "_grasp_candidates", glb)
    _FanDemo._fallback = _exec_method(("morph", "arm", "surface.py"), "_fallback", {})
    demo = _FanDemo(entries)
    return demo, demo._grasp_candidates(0), out


def test_the_fan_orbits_world_up_rotated_into_the_ik_frame_not_the_frames_own_z():
    """CONSTRAINT 2. `grasp_candidates`' `axis` defaults to the IK frame's +z. On a tilted base
    that orbits about the ARM ROOT's vertical instead of the OBJECT's, silently -- every candidate
    would be a pose near the object rather than a grasp of it."""
    demo, _cands, _out = _run_fan([_entry(1.0, 0.26), _entry(2.0, -0.26)])
    assert "axis" in demo.seen, (
        "`axis` was never passed, so the fan orbits the IK frame's own +z by default -- wrong on "
        "any tilted base, and wrong SILENTLY")
    want = demo.bR.T @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(demo.seen["axis"], want), (
        f"orbit axis {np.round(demo.seen['axis'], 4).tolist()}, expected world-up in the IK frame "
        f"{np.round(want, 4).tolist()}")
    assert not np.allclose(want, [0.0, 0.0, 1.0]), (
        "the stub base is not tilted, so passing the default would have passed this test too")


def test_the_fan_orbits_the_live_object_centre_in_the_ik_frame():
    """`pos`, `R`, `obj` and `axis` are all in the frame `ik` works in. A world-frame object centre
    here puts the orbit centre metres away and every candidate misses the object entirely."""
    demo, _cands, _out = _run_fan([_entry(1.0, 0.26)])
    want = demo.bR.T @ (demo.o_w - demo.bp)
    assert np.allclose(demo.seen["obj"], want), (
        f"orbit centre {np.round(demo.seen['obj'], 4).tolist()}, expected the live object in the "
        f"ARM-ROOT frame {np.round(want, 4).tolist()}")


def test_the_nominal_yaw_is_never_offered_as_a_candidate():
    """Yaw 0 IS the grasp that just failed -- `grasp_candidates`' set is a strict superset of the
    nominal solve and reproduces it entry for entry. Offering it back makes the walk the identical
    retry it exists to replace."""
    _demo, cands, _out = _run_fan([_entry(9.0, 0.0), _entry(1.0, 0.26), _entry(2.0, -0.26)])
    assert cands, "the fan offered nothing at all, so nothing below is evidence"
    assert not any(np.allclose(c, 9.0) for c in cands), (
        "the nominal-yaw candidate was offered back -- that is the configuration the attempt that "
        "just failed already commanded")
    assert len(cands) == 2, f"expected the two off-nominal yaws, got {len(cands)}"


def test_the_fan_offers_one_approach_per_yaw_not_every_ik_branch_of_each():
    """The fan is `ik_candidates` PER YAW, so one yaw can carry many branches. Walking branches
    costs a full pick attempt each to vary nothing about the APPROACH -- and choosing between
    branches at one hand pose is already `_goal_set`'s job, decided by reachability."""
    _demo, cands, _out = _run_fan([_entry(1.0, 0.26), _entry(2.0, 0.26), _entry(3.0, -0.26)])
    assert len(cands) == 2, (
        f"expected one candidate per distinct yaw, got {len(cands)} -- two of them differ only in "
        f"IK branch, which is not a different approach")


def test_the_fan_ranks_by_yaw_deviation_itself_rather_than_inheriting_the_scores_order():
    """`grasp_candidates` deliberately carries NO yaw penalty -- it argues turret travel already
    prices yaw, measured on one pose family. Where the turret already points along the orbit that
    breaks down, so the consumer ranks yaw itself: smallest orbit off the TUNED nominal grasp
    first, because the whole close/seat stack downstream is tuned for that approach.

    The fan here arrives WORST-yaw-first, so inheriting its order would be visible."""
    _demo, cands, _out = _run_fan([_entry(3.0, 0.52, score=9.0),      # 30 deg, best-scoring
                                   _entry(2.0, -0.26, score=1.0),     # -15 deg
                                   _entry(1.0, 0.26, score=0.0)])     # +15 deg, worst-scoring
    assert len(cands) == 3, f"expected all three yaws, got {len(cands)}"
    assert np.allclose(cands[-1], 3.0), (
        "the 30 deg candidate ranked ahead of both 15 deg ones -- the fan's score order was "
        "inherited, and that order contains no yaw term at all")
    assert np.allclose(cands[0], 2.0) and np.allclose(cands[1], 1.0), (
        f"equal |yaw| must keep the fan's own score order as the tiebreak: {cands}")


def test_the_fan_offers_every_distinct_yaw_it_is_given():
    """No cap and no top-N. A cap would make the walk depend on the ranking being right, and the
    ranking is precisely what `grasp_candidates` does not measure."""
    _demo, cands, _out = _run_fan([_entry(1.0, 0.26), _entry(2.0, -0.26),
                                   _entry(3.0, 0.52), _entry(4.0, -0.52)])
    assert len(cands) == 4, (
        f"the fan was given four distinct yaws and offered {len(cands)} -- a dropped candidate is "
        f"an approach the walk can never try")


def test_a_fan_with_no_goal_to_orbit_offers_nothing_and_sweeps_nothing():
    """`_grasp_candidates` orbits the goal the approach planned to. With none -- the planner was
    off, or the approach never got that far -- there is nothing to orbit, and paying for the sweep
    to discover that is the cost constraint 1 exists to avoid."""
    demo, _cands, _out = _run_fan([_entry(1.0, 0.26)])
    demo.seen = {}
    # nothing to orbit: the approach never planned one
    demo._grasp_goal = None
    assert demo._grasp_candidates(0) == [], "with no goal to orbit the fan must offer nothing"
    assert not demo.seen, "the sweep ran anyway, against a goal that does not exist"


# --- the wiring against the REAL ArmModel -----------------------------------------------------


class _RealFanDemo:
    """`_grasp_candidates` over the SHIPPED `ArmModel`, with nothing stubbed between the wiring and
    the IK. Identity base: the frame handoff is pinned by the tilted `_FanDemo` above, and what is
    left to prove here is that the real solver accepts what the wiring hands it."""

    def __init__(self, model, goal, obj, obstacles):
        self.model, self._grasp_goal, self.obstacles = model, goal, obstacles
        self._plan_fallbacks = {}
        self.objs = {0: SimpleNamespace(get_world_poses=lambda: (np.array([obj]), None))}

    def _arm_model(self):
        return self.model

    def _arm_base_world(self):
        return np.zeros(3), np.eye(3)

    def _arm_bounds(self):
        return self.model.bounds()

    def _arm_q8(self):
        return np.asarray(self._grasp_goal, float)

    def _arm_obstacles(self, exclude_names=()):
        return self.obstacles


def test_the_fan_wiring_solves_against_the_real_arm_model():
    """Every fan test above stubs `ArmModel.grasp_candidates`. The kwargs the wiring passes --
    `link=`, `bounds=`, `base=`, `held=None`, `margin=` -- are only STATICALLY compatible with
    `ik_candidates`/`ik` until something calls the pair together, and an incompatible one raises
    into `_grasp_candidates`' own `except`, where it becomes a counted degrade and an empty fan
    that reads exactly like a hard pick."""
    from morph.arm.model import ArmModel
    from morph.config import MOUTH_VEC_HAND
    model = ArmModel.load(os.path.join(ROOT, "usd/_arm_model.json"),
                          os.path.join(ROOT, "usd/_closure_lut.json"))
    goal = np.array([json.load(open(os.path.join(ROOT, "usd/_grasp_known.json")))
                     ["arm_joints"][n] for n in ArmModel.Q8], float)
    pos, R = model.fk(goal)[model.hand_link]
    mouth = R @ np.asarray(MOUTH_VEC_HAND, float)
    mouth[2] = 0.0
    obj = pos - 0.13 * mouth / float(np.linalg.norm(mouth))   # the object out along the mouth axis
    body = [(np.array([[-0.3165, -0.2960, 0.1586], [-0.0219, 0.2809, 0.3310]]), "chassis"),
            (np.array([[-0.350, -0.300, 0.139], [0.350, 0.300, 0.159]]), "chassis")]

    out = []
    glb = {"np": np, "os": os, "math": math, "MARGIN": 0.02,
           "print": lambda *a, **k: out.append(" ".join(str(x) for x in a))}
    _RealFanDemo._grasp_candidates = _exec_method(("morph", "pick", "approach.py"),
                                                  "_grasp_candidates", glb)
    _RealFanDemo._fallback = _exec_method(("morph", "arm", "surface.py"), "_fallback", {})
    demo = _RealFanDemo(model, goal, obj, body)
    cands = demo._grasp_candidates(0)

    assert not demo._plan_fallbacks, (
        f"the sweep degraded instead of solving -- the wiring hands `grasp_candidates` something "
        f"it cannot take: {demo._plan_fallbacks} / {out}")
    assert cands, f"the real model offered no approach at all from the shipped grasp: {out}"
    assert all(np.asarray(q).shape == (8,) for q in cands), (
        f"a candidate is not a q8, so nothing downstream can command it: "
        f"{[np.asarray(q).shape for q in cands]}")
    assert not any(np.allclose(q, goal, atol=1e-9) for q in cands), (
        "the nominal grasp came back as a candidate -- that is the configuration that just failed")
    for q in cands:
        assert not model.collides(np.asarray(q, float), body, 0.02, None,
                                  (np.zeros(3), np.eye(3))), (
            f"a candidate collides at the planner's own margin: {np.round(q, 4).tolist()}")


# --- the candidate reaching the planner -------------------------------------------------------


class _ApproachDemo:
    def _commanded_fingers(self):
        return None

    """The `Demo` surface `_plan_approach` touches. `_plan_arm` returns None so the method stops
    at the goal it chose, which is the only thing under test here."""

    def __init__(self, cand=None):
        if cand is not None:
            self._grasp_cand = cand
        self.obj_rel = 0
        self.planned = None
        self._plan_fallbacks = {}

    def _traj_q8(self, tr, frame_i):
        # the BAKED frame
        return np.full(8, 9.0)

    def _object_relative_goal(self, goal, obj_idx):
        self.obj_rel += 1
        # the nominal object-relative solve
        return np.full(8, 7.0)

    def _goal_set(self, goal, excl, held):
        return goal

    def _plan_arm(self, goal, held=None, exclude_names=(), fingers=None):
        self.planned = np.asarray(goal, float)
        return None

    def _fallback(self, reason):
        self._plan_fallbacks[reason] = self._plan_fallbacks.get(reason, 0) + 1


def _run_approach(cand=None):
    out = []
    glb = {"np": np, "os": os, "ARM_PLANNER": True,
           "move_to_pose": _move_to_pose, "Outcome": _Outcome,
           "print": lambda *a, **k: out.append(" ".join(str(x) for x in a))}
    _ApproachDemo._plan_approach = _exec_method(("morph", "pick", "approach.py"),
                                                "_plan_approach", glb)
    _ApproachDemo._approach_goal = _exec_method(("morph", "pick", "approach.py"),
                                                "_approach_goal", glb)
    demo = _ApproachDemo(cand)
    return demo, demo._plan_approach({}, 0, 1.0, exclude_obj=0), out


def test_without_a_candidate_the_approach_still_takes_the_object_relative_solve():
    """The control, and the no-regression: every successful cycle takes exactly this path."""
    demo, _r, _out = _run_approach()
    assert demo.obj_rel == 1, "the nominal approach must still be the object-relative solve"
    assert np.allclose(demo.planned, 7.0), (
        f"the planner was given {demo.planned}, not the object-relative goal")


def test_a_candidate_replaces_the_object_relative_solve_rather_than_reseeding_it():
    """A candidate is ALREADY solved, and already filtered through the same `collides` the planner
    calls. Re-deriving a goal from the bake here would discard the fan and re-command the approach
    that just failed, with the walk none the wiser."""
    demo, _r, _out = _run_approach(cand=np.full(8, 3.0))
    assert np.allclose(demo.planned, 3.0), (
        f"the planner was given {demo.planned}, not the candidate -- the walk changed nothing")
    assert demo.obj_rel == 0, (
        "the object-relative solve ran anyway, so the goal the candidate chose was recomputed")


def test_the_approach_stashes_the_goal_the_fan_has_to_orbit():
    """`_grasp_candidates` orbits the grasp that FAILED, not the bake -- so the approach has to
    leave behind the goal it actually planned to."""
    demo, _r, _out = _run_approach()
    assert np.allclose(getattr(demo, "_grasp_goal", None), 7.0), (
        f"the approach left no goal for the fan to orbit: {getattr(demo, '_grasp_goal', None)}")

# --- Recovery Tests ---

class _AbortRecoverDemo:
    def _commanded_fingers(self):
        return None

    # `_insert_sequence` hands this to `move_linear` as `on_step`; these harnesses measure the
    # slide, not the contact watch, so it is a no-op here.
    def _contact_watch(self, tag):
        return None

    def __init__(self, plan_returns=True, carry_slip=0.0):
        self._safety_abort = False
        self._abort_recovered = False
        self._plan_fallbacks = {}
        self._finger_pin = None
        self.q0 = np.zeros(8)
        self.idx = {n: i for i, n in enumerate(["Link1", "Link2", "Link3", "Link4", "Link5", "Link6", "Link7", "Fingers"])}
        self._f_idx = 7
        self.dt = 1.0/240.0
        self._focus_obj = "pickup_obj_0"
        self.objs = {0: "dummy_obj"}
        self._carry_slip = carry_slip
        self._plan_returns = plan_returns
        self._executed_qs = []
        self._collisions = []

        class World:
            def step(self, render):
                pass
        self.world = World()

        class Robot:
            def get_joint_positions(self):
                return np.zeros(8)
        self.robot = Robot()

    def _fallback(self, tag):
        self._plan_fallbacks[tag] = self._plan_fallbacks.get(tag, 0) + 1

    def _carry_check(self, obj, tag):
        return self._carry_slip

    def _plan_arm(self, q, held=None, grasp_names=(), fingers=None):
        if np.allclose(q, self.q0[:8]):
            if self._plan_returns:
                return [np.zeros(8), np.zeros(8)]
            return None
        return None

    def _execute_arm_path(self, *a, **k):
        # the recursive call for abort-return
        if getattr(self, "_recursed", False):
            return None
        self._recursed = True
        return np.zeros(8)

    def _arm_model(self):
        class Model:
            Q8 = ["Link1", "Link2", "Link3", "Link4", "Link5", "Link6", "Link7", "Fingers"]
            Q8_LIN = [1, 2, 3]              # the real ArmModel's prismatic indices; the taper
            Q8_ROT = [0, 4, 5, 6, 7]        # telemetry labels units from these
            def collides(self, q8, obs, margin, held, base):
                if getattr(self, "force_collision", False):
                    return True
                return False
        return Model()
    
    def _arm_obstacles(self, excl, diagnostic_paths=None, **kw):
        # The real stage ALWAYS carries the deck plate and the checkpoint pads it exactly as `_plan_arm`
        # does, so this fake carries it too -- far from every pose these harnesses fly.
        from morph.arm.collision import CHASSIS_PAD_PRIM
        if diagnostic_paths is not None:
            diagnostic_paths.append(CHASSIS_PAD_PRIM)
        return [(np.array([[50.0, 50.0, -1.0], [51.0, 51.0, -0.9]]), "chassis")]

    def _arm_base_world(self):
        return None

    def _with_closure_forced(self, q):
        return q

    def set_base(self, *a):
        pass

    def base_ledger(self):
        return 0, 0, 0

    def _force(self, q, qv):
        pass

    def _apply(self, q):
        self._executed_qs.append(q.copy())

# The REAL producer and pad, bound onto the abort harness: a fake one would let the checkpoint
# drift from the planner again, which is the defect it was written to end.
_AbortRecoverDemo._planned_obstacles = _exec_method(
    ("morph", "arm", "obstacles.py"), "planned_obstacles", {"chassis_pad": _chassis_pad_real})


def _run_execute_arm_path_abort(plan_returns=True, carry_slip=0.0, force_collision=False,
                                held=np.array([[-0.05, -0.05, 0.0], [0.05, 0.05, 0.10]]),
                                demo_cls=None):
    out = []
    glb = {"np": np, "os": os, "math": math, "MARGIN": 0.02, "HEADLESS": True, "ArmModel": type("ArmModel", (), {"Q8_LIN": [1, 2, 3], "Q8_ROT": [0, 4, 5, 6, 7], "Q8": ["Link1", "Link2", "Link3", "Link4", "Link5", "Link6", "Link7", "Fingers"]}),
           "print": lambda *a, **k: out.append(" ".join(str(x) for x in a))}
    _AbortRecoverDemo._execute_arm_path_tested = _bind_executor(glb)
    demo = (demo_cls or _AbortRecoverDemo)(plan_returns=plan_returns, carry_slip=carry_slip)
    # mock execution of failure
    waypoints = [np.zeros(8), np.ones(8) * 0.5, np.ones(8)]
    # set up last_step_delta
    demo._last_step_delta = np.ones(8) * 0.1
    # force checkpoint failure
    m = demo._arm_model()
    if force_collision:
        m.force_collision = True
    demo._arm_model = lambda: m
    
    def mock_collides(q8, obs, margin, held, base, fingers=None):
        if np.allclose(q8, waypoints[1][:8]) or np.allclose(q8, waypoints[2][:8]):
            return True
        if getattr(m, "force_collision", False):
            return True
        return False
    m.collides = mock_collides
    
    try:
        demo._execute_arm_path_tested(waypoints, 1.0, checkpoint_every=1, held=held)
    except Exception as e:
        print(f"Exception in _execute_arm_path: {e}")
        import traceback
        traceback.print_exc()
    
    return demo, out

def test_recovery_taper_and_success():
    # 1. Injected checkpoint abort WITH a payload -> taper runs, abort-recovered OUTCOME
    demo, out = _run_execute_arm_path_abort(plan_returns=True, carry_slip=0.0)
    assert demo._safety_abort is True
    assert demo._abort_recovered is True
    assert demo._plan_fallbacks.get("abort-recovered", 0) == 1
    assert demo._plan_fallbacks.get("execute-replan-failed", 0) == 0

def test_recovery_unroutable_park():
    # 2. _plan_arm -> None -> execute-replan-failed SAFETY, _abort_recovered stays False
    demo, out = _run_execute_arm_path_abort(plan_returns=False, carry_slip=0.0)
    assert demo._safety_abort is True
    assert demo._abort_recovered is False
    assert demo._plan_fallbacks.get("execute-replan-failed", 0) == 1
    assert demo._plan_fallbacks.get("abort-recovered", 0) == 0

def _run_hook_test(hook_value):
    out = []
    glb = {"np": np, "os": os, "math": math, "MARGIN": 0.02, "HEADLESS": True, "ArmModel": type("ArmModel", (), {"Q8_LIN": [1, 2, 3], "Q8_ROT": [0, 4, 5, 6, 7], "Q8": ["Link1", "Link2", "Link3", "Link4", "Link5", "Link6", "Link7", "Fingers"]}),
           "print": lambda *a, **k: out.append(" ".join(str(x) for x in a))}
    _AbortRecoverDemo._execute_arm_path_tested = _bind_executor(glb)
    
    # We want _plan_arm to SUCCEED when replanning, so the only reason it fails is the hook.
    class HookDemo(_AbortRecoverDemo):
        def _plan_arm(self, q, held=None, grasp_names=(), fingers=None):
            if np.allclose(q, self.q0[:8]):
                return [np.zeros(8), np.zeros(8)] # park succeeds
            return [np.zeros(8), np.ones(8)] # replan succeeds

    demo = HookDemo(plan_returns=True, carry_slip=0.0)
    demo._last_step_delta = np.ones(8) * 0.1
    waypoints = [np.zeros(8), np.zeros(8), np.ones(8)]
    
    m = demo._arm_model()
    def mock_collides(q8, obs, margin, held, base, fingers=None):
        if np.allclose(q8, waypoints[1][:8]) or np.allclose(q8, waypoints[2][:8]):
            return True
        return False
    m.collides = mock_collides
    demo._arm_model = lambda: m

    if hook_value is not None:
        os.environ["ARM_PLAN_TEST_FAIL_REPLAN"] = hook_value
    else:
        os.environ.pop("ARM_PLAN_TEST_FAIL_REPLAN", None)
    
    try:
        demo._execute_arm_path_tested(waypoints, 1.0, checkpoint_every=1)
    except Exception as e:
        print(f"Exception: {e}")
    finally:
        os.environ.pop("ARM_PLAN_TEST_FAIL_REPLAN", None)
        
    return demo, out

def test_recovery_hook_unset():
    # With the hook unset, the checkpoint replan succeeds and no abort is forced.
    demo, out = _run_hook_test(None)
    assert demo._safety_abort is False
    assert getattr(demo, "_arm_test_failed_replan", False) is False

def test_recovery_hook_set():
    # With the hook set, the abort path runs and recovery follows.
    demo, out = _run_hook_test("1")
    assert demo._safety_abort is True
    assert demo._abort_recovered is True
    assert demo._plan_fallbacks.get("abort-recovered", 0) == 1
    assert getattr(demo, "_arm_test_failed_replan", False) is True

def test_recovery_payload_lost():
    # 4. Payload slip beyond the band -> payload-lost emitted, PAYLOAD-LOST printed, no return-to-park
    demo, out = _run_execute_arm_path_abort(plan_returns=True, carry_slip=10.0)
    assert demo._plan_fallbacks.get("payload-lost", 0) == 1
    assert any("PAYLOAD-LOST" in l for l in out)
    # No return to park
    assert getattr(demo, "_recursed", False) is False

def test_no_payload_is_declared_so_no_payload_can_be_LOST():
    """The abort path measured `_carry_check` against `_focus_obj`, which is set at dock and never
    cleared -- so a motion running AFTER release measured the shelved object against a grip-time
    baseline. Measured over the logs: in-hand slip never exceeds 1.8mm, while `released` reads
    7.3mm and `settled` 350.7mm, both past the 5.0mm band. The backout REMAINDER runs
    `checkpoint_every=8` after release, so an abort there would have reported a dropped payload for
    an object sitting safely on the shelf, and `payload-lost` is OUTCOME-graded -- it fails the run.

    `held` is the declaration that there IS a payload (the deleted `_held_box`'s contract: "Call ONLY while an object is
    held"), and the remainder passes none."""
    demo, out = _run_execute_arm_path_abort(plan_returns=True, carry_slip=350.7, held=None)
    assert demo._plan_fallbacks.get("payload-lost", 0) == 0, (
        "a motion carrying nothing reported a lost payload: " + str(demo._plan_fallbacks))
    assert not any("PAYLOAD-LOST" in l for l in out), out


def test_recovery_collision_trip():
    # 3. Injected abort where the taper's per-step collision check trips -> the taper collapses
    demo, out = _run_execute_arm_path_abort(plan_returns=True, carry_slip=0.0, force_collision=True)
    # The taper's own count is the telemetry line's; what this pins is that the collapse added nothing
    # to the segment already flown. Recomputed, never a literal -- it moves with the joint ceilings.
    from morph.arm.cartesian import retime_q8
    flown = len(retime_q8([np.zeros(8), np.ones(8) * 0.5], 1.0 / 240.0)[1]) - 1
    assert len(demo._executed_qs) == flown, (
        f"{len(demo._executed_qs)} commands against the {flown} the first segment retimes to")
    assert demo._safety_abort is True


def _stub_move_linear(recorder=None, refuse=False):
    """`move_linear` for harnesses whose subject is the SLIDE, not the column ramps.

    The ramps are exercised for real in tests/test_arm_api.py and tests/test_arm_linear.py. Here
    they would drag the whole primitive's delegate surface into a fixture built to measure
    something else, so they are stubbed -- and the stub RECORDS the ask, so a test can still assert
    the ramp was requested with the delta and duration the caller meant."""
    from morph.arm.api import ArmMove, Outcome

    def _ml(demo, delta, frame, *, secs=None, tag="linear", **kw):
        if recorder is not None:
            recorder.append((tag, np.asarray(delta, float).copy(), secs))
        if refuse:
            return ArmMove(Outcome.REFUSED, f"stub refusal on {tag}")
        q = np.asarray(demo.robot.get_joint_positions(), float).copy()
        ih1, ih2 = demo.idx["ColumnLeftBearingJoint_1"], demo.idx["ColumnRightBearingJoint_1"]
        q[ih1] += float(delta[2])
        q[ih2] += float(delta[2])
        demo.robot.set_joint_positions(q) if hasattr(demo.robot, "set_joint_positions") else None
        return ArmMove(Outcome.ARRIVED, "arrived", (), q)

    return _ml


def _run_insert_abort():
    out = []
    glb = {"np": np, "os": os, "math": math, "A1_MAX": 0.5, "COLUMN_MAX": 0.5, "MARGIN": 0.02,
           "move_linear": _stub_move_linear(refuse=True),
           "print": lambda *a, **k: out.append(" ".join(str(x) for x in a))}
    _AbortRecoverDemo._insert_sequence = _exec_method(("morph", "place", "insert.py"), "_insert_sequence", glb)
    demo = _AbortRecoverDemo()
    demo._time_stage = lambda name: contextlib.nullcontext()
    # Force abort (q is None)
    demo._ramp = lambda *a, **k: None
    
    class Obj:
        def get_world_poses(self): return [[[0, 0, 0]], [[1, 0, 0, 0]]]
    
    demo.robot.get_joint_positions = lambda: np.zeros(30)
    demo.idx = {"ColumnLeftBearingJoint_1": 0, "ColumnRightBearingJoint_1": 1,
                "ArmLeftJoint_1": 2, "ArmLeftJoint_2": 3, "BaseJoint_1": 4}
    demo.clut = {"grip": np.zeros((10, 10, 3)), "mount": [0, 0]}
    
    demo._insert_sequence(Obj(), [1, 1, 1], 0.1, lambda q: None)
    return demo, out


def test_recovery_insert_phase_refused():
    # 5. Insert-phase abort is still refused: no taper, no payload check, no return-to-park, no abort-
    # recovered tag, and _abort_recovered stays False.
    demo, out = _run_insert_abort()
    assert demo._safety_abort is True
    assert getattr(demo, "_insert_refused", False) is True
    assert getattr(demo, "_abort_recovered", False) is False
    assert demo._plan_fallbacks.get("abort-recovered", 0) == 0
    # No return to park
    assert not getattr(demo, "_recursed", False)
    assert not any("PAYLOAD-LOST" in l for l in out)
    # No taper executed
    assert not demo._executed_qs


def test_abort_taper_telemetry_reports_steps_and_reason_completed():
    demo, out = _run_execute_arm_path_abort(plan_returns=True, carry_slip=0.0, force_collision=False)
    taper_lines = [l for l in out if ">>> abort-taper:" in l]
    assert len(taper_lines) == 1, "taper line must be printed exactly once"
    line = taper_lines[0]
    assert "ended by completion" in line, f"expected 'ended by completion', got: {line}"
    assert "ran 96/96 steps" in line, f"expected 'ran 96/96 steps', got: {line}"


def test_abort_taper_telemetry_reports_steps_and_reason_cut_short():
    demo, out = _run_execute_arm_path_abort(plan_returns=True, carry_slip=0.0, force_collision=True)
    taper_lines = [l for l in out if ">>> abort-taper:" in l]
    assert len(taper_lines) == 1, "taper line must be printed exactly once"
    line = taper_lines[0]
    assert "ended by collision check" in line, f"expected 'ended by collision check', got: {line}"
    assert "ran 0/96 steps" in line, f"expected 'ran 0/96 steps', got: {line}"
def test_a_replan_after_a_base_slip_retargets_in_world_not_in_joints():
    """P1.4. A checkpoint replan targeted `waypoints[-1]`, a JOINT vector. Obstacles are re-measured
    against the live base, so the replan is collision-correct -- but the arm is mounted ON the base,
    so after a slip those joints put the hand off in world BY the slip. Only the reach passes a
    world goal; the rest pass joint configurations and must be untouched."""
    import ast as _ast, os as _os
    src = _os.path.join(ROOT, "morph", "arm", "execute.py")
    tree = _ast.parse(open(src).read())
    fn = next(n for n in _ast.walk(tree)
              if isinstance(n, _ast.FunctionDef) and n.name == "execute_path")
    assert "regoal" in [a.arg for a in fn.args.args + fn.args.kwonlyargs], (
        "the executor cannot re-derive its goal, so a replan after a slip still targets the "
        "joint vector the plan began with")

    calls = [c for c in _ast.walk(fn) if isinstance(c, _ast.Call)
             and getattr(c.func, "id", "") == "plan_arm"]
    assert calls, "fixture: the executor replans somewhere"
    guard = [n for n in _ast.walk(fn) if isinstance(n, _ast.If)
             and any(getattr(x, "id", "") == "SLIP_RETARGET_M" for x in _ast.walk(n.test))]
    assert guard, "re-derivation is not gated on the base having actually moved"
    body = guard[0].body
    assert any(isinstance(a, _ast.Assign) and getattr(a.targets[0], "id", "") == "b0"
               for a in _ast.walk(guard[0])), (
        "the slip reference is never rebound, so the second checkpoint measures against a base "
        "position two corrections old")
    refuses = [n for n in _ast.walk(guard[0]) if isinstance(n, _ast.If)
               and any(isinstance(c, _ast.Constant) and c.value is None for c in _ast.walk(n.test))]
    assert refuses and any(isinstance(r, _ast.Return) for n in refuses for r in _ast.walk(n)), (
        "a goal that cannot be re-derived must refuse, not replan to the stale one")

    thresh = next(n for n in tree.body if isinstance(n, _ast.Assign)
                  and getattr(n.targets[0], "id", "") == "SLIP_RETARGET_M")
    assert 0.001 <= float(thresh.value.value) <= 0.02, float(thresh.value.value)


def test_the_refused_settle_holds_the_pose_it_reached_not_the_goal_it_refused():
    """`settle-snap-too-far` skips the one-frame teleport and then prints "holding the pose the arm
    actually reached" -- but `qg` is still the goal, and the `_hold` below commands its vector
    every step. Refusing the snap only spread the same unchecked motion over
    0.1s. Hold what was MEASURED, which is what the print already claims."""
    import types as _types
    from morph.arm.model import ArmModel as _AM
    from morph.arm.collision import Q8_TOL_M as _TM, Q8_TOL_RAD as _TR
    _AM_Q8 = _AM.Q8
    glb = {"np": np, "os": os, "ArmModel": _AM, "Q8_TOL_M": _TM, "Q8_TOL_RAD": _TR,
           "KNOWN": {"arm_joints": []}, "HEADLESS": True}
    settle = _exec_method(("morph", "pick", "descend.py"), "_pick_settle", glb)

    MEASURED = np.zeros(8)

    class _D:
        def __init__(self):
            self.names = ["j0"]
            self.idx = {n: i for i, n in enumerate(_AM_Q8)}
            self.dt = 1.0 / 240.0
            self._stage = None
            self._gains_writer = None
            self.snapped = []
            self.held = []
            self._plan_fallbacks = {}
            self._plan_to_config_reason = None
            self.robot = _types.SimpleNamespace(
                set_joint_positions=self.snapped.append,
                get_joint_positions=lambda: MEASURED.copy())
        def _arm_q8(self):
            return MEASURED.copy()
        def _hold(self, q, *a, **k):
            self.held.append(np.asarray(q, float).copy())
        def _fallback(self, tag):
            self._plan_fallbacks[tag] = self._plan_fallbacks.get(tag, 0) + 1
        def _set_gains(self, *a, **k):
            pass
        def _arm_gain(self, n, kp):
            return kp, kp / 10.0, None
        def _wheel_gain(self, n):
            return 0.0, 0.0
        def _body_clear(self, *a, **k):
            pass
        def _finite(self, *a, **k):
            # Everything past the hold -- the base-shove check, the grip bookkeeping -- is a
            # different subject. Stop here so this test fails for its own reason and no other.
            raise _Cut
        def _arm_obstacles(self, *a, **k):
            return []
        def _apply(self, *a, **k):
            pass
        def _force(self, *a, **k):
            pass

    goal = np.zeros(8)
    # half a metre of boom slide from where the arm stands
    goal[3] = 0.5
    st = _types.SimpleNamespace(obj_idx=0, status=None, dock=None, qg=goal, obj=None,
                                park=None, pin_bystanders=None, replay=True,
                                alive=True, grip=None)
    d = _D()
    try:
        settle(d, st)
    except _Cut:
        pass

    assert d.snapped == [], "the teleport itself must still be refused"
    assert d._plan_fallbacks.get("settle-snap-too-far"), "the refusal must be counted"
    # Nothing may be COMMANDED toward the refused goal -- not in one frame by the snap, and not
    # over 0.1s by the `_hold`, which commands its vector every step.
    for q in d.held:
        off = float(np.max(np.abs(np.asarray(q, float) - MEASURED)))
        assert off < 1e-9, (
            f"the refused settle commands the goal it just refused: the held vector is "
            f"{off * 1000:.0f}mm from the measured pose")
    assert st.alive is False, (
        "a residue too large to snap means the reach did not arrive, so every stage below would "
        "run from a pose that is not the grasp config -- the attempt must end")
    # And the pose handed back is the MEASURED one: `st.grip` is the only observable distinguishing
    # recorded-where-the-arm-is from recorded-the-goal-a-check-refused.
    assert st.grip is not None, "the attempt ended without recording a pose at all"
    off = float(np.max(np.abs(np.asarray(st.grip, float) - MEASURED)))
    assert off < 1e-9, (
        f"the refused attempt recorded the GOAL as the grip pose, {off * 1000:.0f}mm from where "
        f"the arm actually is -- anything that later reads it inherits the refused target")


def test_the_reach_has_no_baked_fallback_left():
    """P3.1's point: the reach no longer depends on a recorded trajectory for its ROUTE, and with
    `_play_traj` deleted there is no replay primitive left to depend on one. `self.traj["reach"]`
    survives for its frame COUNT and `_traj_q8`, neither of which commands the arm.

    The machinery that existed to make a route replay safe must be gone with it, or it rots: a
    checker with no caller is read as coverage that does not exist."""
    src = open(os.path.join(ROOT, "morph", "pick", "reach.py"), encoding="utf-8").read()
    whole = src + "".join(open(os.path.join(ROOT, "morph", *rel), encoding="utf-8").read()
                          for rel in (("arm", "api.py"), ("arm", "execute.py"), ("arm", "plan.py"),
                                      ("arm", "retreat.py"), ("arm", "surface.py"), ("pick", "approach.py")))
    for dead in ("_play_traj", "_bake_blocked", "_bake_refused", "_reach_no_bake"):
        assert dead not in whole, (
            f"{dead} survives with no caller -- it belonged to the route replay that is deleted")


def _reach_module():
    """`morph/pick/reach.py` compiled from source TEXT, never through the loader -- the bytecode
    cache is keyed on (mtime seconds, size), and mutating this file moves lines."""
    path = os.path.join(ROOT, "morph", "pick", "reach.py")
    mod = types.ModuleType("_reach_under_test")
    mod.__file__ = path
    exec(compile(open(path).read(), path, "exec"), mod.__dict__)
    return mod


from morph.arm.model import ArmModel  # noqa: E402


class _HandoffDemo:
    """The `self` the probe reads, backed by the REAL arm model.

    A stub that ignores `q8` and the obstacle box cannot see a probe frozen to one configuration:
    `q8 = np.zeros(8)` then reports a healthy-looking 29.8mm at every boundary of every descent
    forever. So the geometry here is the shipped `ArmModel`, and the numbers below are the live
    descent's own, reproduced offline.
    """

    _M = ArmModel.load(os.path.join(ROOT, "usd", "_arm_model.json"),
                       os.path.join(ROOT, "usd", "_closure_lut.json"))

    def __init__(self, th=0.0, h1=0.0, h2=0.0, a1=0.0):
        self.idx = {n: i for i, n in enumerate(ArmModel.Q8)}
        q = np.zeros(len(ArmModel.Q8))
        q[self.idx["BaseJoint_1"]] = th
        q[self.idx["ColumnLeftBearingJoint_1"]] = h1
        q[self.idx["ColumnRightBearingJoint_1"]] = h2
        q[self.idx["ArmLeftJoint_1"]] = a1
        self._q = q
        self._finger_cmd = None
        self.robot = types.SimpleNamespace(get_joint_positions=lambda: self._q)
        self.seen = []

    def _pinch(self):
        return np.array([0.1, 0.2, 0.3])

    def _arm_q8(self):
        return self._q.copy()

    def _arm_model(self):
        demo = self

        class _Watched:
            """The shipped model, with every `first_hit` call recorded."""

            def first_hit(self, q8, obstacles, margin, *a, **kw):
                demo.seen.append((np.asarray(q8, float).copy(), list(obstacles),
                                  float(margin), kw.get("links")))
                return demo._M.first_hit(q8, obstacles, margin, *a, **kw)

        return _Watched()


def _handoff_line(demo, tag="pre-reach-lowest"):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        _reach_module().ReachStage._descent_handoff(demo, None, tag)
    text = out.getvalue()
    assert "unavailable" not in text, f"the probe raised instead of measuring: {text.strip()}"
    return text


# The live descent, reproduced offline against the shipped model: `th` from the `arm-body` line two
# lines above each handoff in /tmp/bench/p36_check.log, the rest from the handoff line itself.
PRE_REACH_LOWEST = dict(th=-0.195, h1=0.1974, h2=0.3271, a1=0.2466)     # boom-deck 7.15mm
POST_REACH_LOWEST = dict(th=-0.195, h1=0.1973, h2=0.3267, a1=0.4367)    # 4.1mm INSIDE the deck


def test_the_descent_handoff_prints_what_the_close_stack_actually_reads():
    """A migration that lands the hand identically on a different IK BRANCH changes everything the
    close stack consumes, and the operator's own probe missed it by gating on hand pose alone:
    `_reach_lowest` uses the entry a1/dh, the guarded seat needs COLUMN-TRAVEL HEADROOM and force
    balance inherits standing joint error. RUN the probe and read its numbers."""
    from morph.config import A1_MAX, COLUMN_MAX
    p = PRE_REACH_LOWEST
    line = _handoff_line(_HandoffDemo(**p))

    assert f"a1 {p['a1']:.4f}" in line, f"a1, what _reach_lowest enters on, is not reported: {line}"
    assert f"head {A1_MAX - p['a1']:+.4f}" in line, f"a1 headroom is not reported: {line}"
    assert f"h1 {p['h1']:.4f} h2 {p['h2']:.4f}" in line, f"the columns are not reported: {line}"
    assert f"dh {p['h2'] - p['h1']:+.4f}" in line, f"dh is not reported: {line}"
    assert f"col head {COLUMN_MAX - p['h2']:+.4f}" in line, (
        f"column-travel headroom, which the guarded seat floors against, is missing: {line}")
    assert "pinch [0.1000 0.2000 0.3000]" in line, f"the hand pose is not reported: {line}"
    assert "fingers free" in line, f"finger ownership across the boundary is not reported: {line}"
    # The live descent's own entry clearance against the shipped model. Pinning the NUMBER catches a
    # probe measuring a frozen or wrong configuration -- zeros read healthy, a doubled box OVERLAPPING.
    m = re.search(r"boom-deck ([0-9.]+)mm", line)
    assert m and abs(float(m.group(1)) - 9.70) < 0.05, (
        f"boom-to-deck is not the descent's measured entry clearance (9.70mm): {line}")


def test_the_handoff_probe_reports_deck_saturation_and_overlap_distinctly():
    """Three states, three readings. A clamped bisection returns its own bound both when the boom is
    far and when the box or the link name is wrong, and returns 0 both when the boom touches the
    deck and when it is already inside it -- and the live descent IS inside it, so that second
    collapse is the one that would have hidden the finding this probe exists to make."""
    assert "boom-deck >=300mm" in _handoff_line(_HandoffDemo(h1=0.5, h2=0.5)), (
        "a boom well clear of the deck reads as a bisected number")
    assert "boom-deck OVERLAPPING" in _handoff_line(_HandoffDemo(**POST_REACH_LOWEST)), (
        "the descent's own end pose is 4mm INSIDE the deck box and must not read as clearance")


def test_the_handoff_probe_measures_the_boom_alone_against_the_deck():
    """Column links are exempt from the chassis by construction (armfk.py `column_group`), so an
    unfiltered check answers a different question than the one asked, and the box must be the deck
    the planner would carry -- not a box the probe invents."""
    from morph.config import CHASSIS_PLATE_BOX
    demo = _HandoffDemo(**PRE_REACH_LOWEST)
    _handoff_line(demo)
    assert demo.seen, "the probe never ran a collision query"
    for q8, obstacles, _margin, links in demo.seen:
        assert links == ("Arm_Left_1",), f"the deck check is not restricted to the boom: {links}"
        assert len(obstacles) == 1 and obstacles[0][1] == "chassis", (
            f"the deck check does not carry exactly the chassis deck: {obstacles}")
        assert np.array_equal(obstacles[0][0], CHASSIS_PLATE_BOX), (
            "the probe measures against a box other than the shipped chassis plate")
        assert abs(q8[ArmModel.Q8.index("ArmLeftJoint_1")] - PRE_REACH_LOWEST["a1"]) < 1e-9, (
            "the probe measures a configuration that is not the arm's current one")


def test_the_handoff_probe_cannot_end_a_cycle():
    """`play_isaac.py` ends the cycle on any exception out of the pick. A diagnostic that can do
    that is worse than no diagnostic, and ONE inner try is not enough -- the raise can come from
    the joint read or the print."""
    src = open(os.path.join(ROOT, "morph", "pick", "reach.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_descent_handoff"), None)
    assert fn is not None, "the descent handoff probe is gone -- step 2 has no baseline to compare"
    body = [n for n in fn.body if not (isinstance(n, ast.Expr)
                                       and isinstance(n.value, ast.Constant))]
    assert len(body) == 1 and isinstance(body[0], ast.Try), (
        "the probe body is not ONE try -- anything outside it can end the cycle")
    assert any(isinstance(h.type, ast.Name) and h.type.id == "Exception"
               for h in body[0].handlers), "the probe's guard does not catch Exception"

    # a probe that raises must SAY so, or a silent pass reads as a measurement
    demo = _HandoffDemo(**PRE_REACH_LOWEST)
    demo.robot = types.SimpleNamespace(
        get_joint_positions=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        _reach_module().ReachStage._descent_handoff(demo, None, "pre-reach-lowest")
    assert "unavailable" in out.getvalue(), (
        f"the probe swallowed a failure without reporting it: {out.getvalue()!r}")

    # BOTH boundaries, or old-vs-new cannot be compared where it matters.
    calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
             and getattr(c.func, "attr", "") == "_descent_handoff"]
    tags = {a.value for c in calls for a in c.args if isinstance(a, ast.Constant)}
    assert tags == {"pre-reach-lowest", "post-reach-lowest"}, (
        f"the probe runs at {sorted(tags)}; it must bracket `_reach_lowest`, because that is the "
        f"call whose entry configuration the migration would change")



def test_the_executor_checkpoint_pads_the_deck_exactly_as_the_planner_did():
    """`_plan_arm` floors `Gripper_Link2_1`/`Gripper_Link3_1` against the robot's own deck at
    `CHASSIS_PAD_M` instead of MARGIN (`chassis_pad`, 38bb14f). The checkpoint in
    `execute_path` re-validates the SAME waypoints, so it must apply the SAME floor -- or a
    waypoint the planner accepted at 11mm is refused at 20mm on every checkpoint and re-planned to
    the same route. Measured: seven replans per cycle, 5.8 s of an 8.0 s planner budget
    (`logs/walltime`), zero before the route ended near the deck (`logs/pad3`).

    BEHAVIOURAL, through the real `execute_path`: a fake model records the obstacle set the
    checkpoint hands `collides`, and the deck box must arrive as the 4-tuple floor. A structural
    check would pass with `chassis_pad` sitting in a dead branch.
    """
    from morph.arm.obstacles import chassis_pad as _pad
    from morph.arm.collision import CHASSIS_PAD_PRIM, CHASSIS_PAD_M, CHASSIS_PAD_LINKS
    seen = []
    deck = (np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.1]]), "chassis")

    class _PadDemo(_AbortRecoverDemo):
        def _arm_obstacles(self, names=(), diagnostic_paths=None, **kw):
            if diagnostic_paths is not None:
                diagnostic_paths.append(CHASSIS_PAD_PRIM)
            return [deck]

        def _arm_base_world(self):
            return np.zeros(3), np.eye(3)

        def base_ledger(self):
            return 0.0, 0.0, 0.0

        def set_base(self, *a, **k):
            pass

        def _force(self, *a, **k):
            pass

        def _with_closure_forced(self, q):
            return q

        def _arm_model(self):
            class Model:
                Q8 = ["Link1", "Link2", "Link3", "Link4", "Link5", "Link6", "Link7", "Fingers"]
                Q8_LIN = [1, 2, 3]
                Q8_ROT = [0, 4, 5, 6, 7]

                def collides(self, q8, obs, margin, held, base, fingers=None):
                    seen.append([tuple(o) if isinstance(o, (list, tuple)) else o for o in obs])
                    return False
            return Model()

    out = []
    glb = {"np": np, "os": os, "math": math, "MARGIN": 0.02, "HEADLESS": True,
           "chassis_pad": _pad,
           "ArmModel": type("ArmModel", (), {
               "Q8": ["Link1", "Link2", "Link3", "Link4", "Link5", "Link6", "Link7", "Fingers"],
               "Q8_LIN": [1, 2, 3], "Q8_ROT": [0, 4, 5, 6, 7]}),
           "print": lambda *a, **k: out.append(" ".join(str(x) for x in a))}
    _PadDemo._execute_arm_path_tested = _bind_executor(glb)
    demo = _PadDemo()
    demo._last_step_delta = np.zeros(8)
    wps = [np.full(8, 0.01 * k) for k in range(12)]      # > checkpoint_every, so one fires
    demo._execute_arm_path_tested(wps, 1.0, held=None, tag="pad-test", checkpoint_every=8)
    assert seen, "the checkpoint never called collides -- the harness did not reach it"
    last = seen[-1]
    floors = [o for o in last if len(o) == 4 and abs(float(o[2]) - CHASSIS_PAD_M) < 1e-12
              and tuple(o[3]) == tuple(CHASSIS_PAD_LINKS)]
    assert floors, (
        f"the executor's checkpoint validated against an UNPADDED deck {last} -- the planner "
        f"floored it at {CHASSIS_PAD_M * 1000:.0f}mm, so every waypoint inside MARGIN of the plate "
        f"is refused here and re-planned to the same route")


def test_a_checkpoint_that_cannot_build_its_obstacle_set_stops_the_motion_instead_of_raising():
    """`_planned_obstacles` RAISES when the deck prim matches no box -- by design, at plan time,
    before anything moves. The checkpoint now calls it MID-MOTION, and a raise there would unwind
    through `execute_path` with waypoints already flown and no stop of any kind. It must end
    the motion the way a failed replan does: latch the safety abort, count it, return None."""
    from morph.arm.obstacles import chassis_pad as _pad

    class _NoDeckDemo(_AbortRecoverDemo):
        def _arm_obstacles(self, excl, diagnostic_paths=None, **kw):
            return []                          # a stage with no deck: chassis_pad raises

    glb = {"np": np, "os": os, "math": math, "MARGIN": 0.02, "HEADLESS": True, "chassis_pad": _pad,
           "ArmModel": type("ArmModel", (), {
               "Q8": ["Link1", "Link2", "Link3", "Link4", "Link5", "Link6", "Link7", "Fingers"],
               "Q8_LIN": [1, 2, 3], "Q8_ROT": [0, 4, 5, 6, 7]}),
           "print": lambda *a, **k: None}
    _NoDeckDemo._execute_arm_path_tested = _bind_executor(glb)
    demo = _NoDeckDemo()
    demo._last_step_delta = np.zeros(8)
    wps = [np.full(8, 0.01 * k) for k in range(12)]
    try:
        out = demo._execute_arm_path_tested(wps, 1.0, held=None, tag="nodeck", checkpoint_every=8)
    except KeyError as e:
        raise AssertionError(f"the checkpoint let chassis_pad's KeyError escape mid-motion: {e}")
    assert out is None, f"the motion continued past a checkpoint it could not validate: {out!r}"
    assert demo._safety_abort is True, "no safety abort latched"
    assert demo._plan_fallbacks.get("checkpoint-obstacles-unavailable") == 1, demo._plan_fallbacks


def test_the_plan_seeds_from_the_commanded_pose_only_when_it_is_where_the_arm_is():
    """Under drives the arm trails its command (0.34 mm measured, `logs/lag`), and a line halts at
    the last waypoint the checker cleared -- so a start read from the ARM can sit inside a margin
    the COMMANDED pose clears, and the remainder plan is refused at "clearance 20mm of margin
    20mm". The commanded pose is the one that was validated, so it is the seed. But only inside
    the model's own same-pose band: a 22 mm breach is on record under `descend`, and a seed that
    far from the arm would plan an unchecked first segment. Off drives nothing changes."""
    out = []
    pr = lambda *a, **k: out.append(" ".join(str(x) for x in a))
    _PlanArmDemo._plan_start = _exec_method(("morph", "arm", "plan.py"), "plan_start",
                                            {"np": np, "ArmModel": _ARM_MODEL, "print": pr,
                                             "Q8_TOL_M": 0.005, "Q8_TOL_RAD": 0.02})
    meas = np.array([0.0, 0.2000, 0.3300, 0.4000, 0.0, 0.0, 0.0, 0.0])
    d = _PlanArmDemo()
    d._arm_q8 = lambda: meas.copy()
    # off drives: measured, even with a stale command on record
    d._drv_prev = meas + 0.0003
    q, src = d._plan_start()
    assert src == "measured" and np.allclose(q, meas), (src, q)
    # on drives, inside the band (0.34 mm on the columns): the commanded pose
    d._drive_on = lambda: True
    cmd = meas.copy(); cmd[1] += 0.00027; cmd[2] += 0.00034
    d._drv_prev = cmd.copy()
    q, src = d._plan_start()
    assert src == "commanded" and np.allclose(q, cmd), (src, q)
    assert any("COMMANDED" in m for m in out), out
    # on drives, outside the band (a 22 mm breach): the measured pose, and it says so
    cmd2 = meas.copy(); cmd2[3] += 0.022
    d._drv_prev = cmd2.copy()
    q, src = d._plan_start()
    assert src == "measured" and np.allclose(q, meas), (src, q)
    assert any("outside the band" in m for m in out), out

def test_recovery_refuses_when_the_declared_payload_cannot_be_found():
    """`held` declares a payload. When the recovery cannot resolve the object it cannot verify the
    grip, and flying back to park anyway is the silent pass this refuses."""
    class _NoObj(_AbortRecoverDemo):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.objs = {}                        # `pickup_obj_0` resolves to nothing
            self.carry_checks = 0

        def _carry_check(self, obj, tag):
            self.carry_checks += 1
            return 0.0

    demo, out = _run_execute_arm_path_abort(plan_returns=True, carry_slip=0.0, demo_cls=_NoObj)
    assert demo._safety_abort is True
    assert demo.carry_checks == 0, "nothing to check against"
    assert demo._plan_fallbacks.get("payload-unverifiable", 0) == 1, demo._plan_fallbacks
    assert demo._plan_fallbacks.get("abort-recovered", 0) == 0, "flew to park with an unverified payload"
    assert demo._plan_fallbacks.get("execute-replan-failed", 0) == 1, demo._plan_fallbacks
    assert demo._abort_recovered is False


class _VetoDemo:
    """The dock stage's surface with every corridor sample inside a rack and every dock valid, so
    the search ends with `best` set and `reach_ok` False. Everything else is a recorder, so an arm
    or base command after the veto names itself in `calls`."""

    def __init__(self):
        self.calls, self._plan_fallbacks = [], {}
        self.obj_xy = {0: np.array([4.0, -4.0]), 1: np.array([2.0, -2.0]), 2: np.array([6.0, -6.0])}
        # Standoff 1.5: corridor samples past the 0.35 m exemption around the object exist.
        self.local_obj, self.pick_standoff, self.placed = np.array([0.5, 0.0]), 1.5, []
        self.dt, self.idx, self.names = 1.0 / 240.0, {}, []
        self.robot = SimpleNamespace(get_joint_positions=lambda: np.zeros(8),
                                     set_joint_velocities=lambda v: self.calls.append("set_joint_velocities"))
        self.world = SimpleNamespace(step=lambda render: self.calls.append("world.step"))
        self.objs = {0: SimpleNamespace(get_world_poses=lambda: (np.array([[4.0, -4.0, 0.09]]), None))}

    def base_pose(self):
        return 3.0, -3.0, 0.0

    def base_ledger(self):
        return 3.0, -3.0, 0.0

    def _inside_rack(self, x, y, r):
        # corridor samples (0.30) blocked, docks (0.45) fine
        return r < 0.4

    def _fallback(self, tag):
        self._plan_fallbacks[tag] = self._plan_fallbacks.get(tag, 0) + 1

    def __getattr__(self, name):
        def rec(*a, **k):
            self.calls.append(name)
        return rec


def test_a_corridor_veto_ends_the_attempt_and_commands_nothing():
    """`dock-no-corridor` used to be counted and then IGNORED: the stage drove to the rejected dock
    and the pick went on into a reach that never happens and an unchecked jog. Under A2 a veto
    refuses the attempt."""
    out = []
    glb = {"np": np, "os": os, "math": math, "NUM_OBJECTS": 3, "HEADLESS": True, "wrap": lambda a: a,
           "scene": SimpleNamespace(set_arm_gains=lambda robot, names: None),
           "print": lambda *a, **k: out.append(" ".join(str(x) for x in a))}
    setup = _exec_method(("morph", "pick", "dock.py"), "_pick_setup", glb)
    demo = _VetoDemo()
    st = SimpleNamespace(obj_idx=0, status=None, alive=True)
    setup(demo, st)
    assert demo._plan_fallbacks == {"dock-no-corridor": 1}, demo._plan_fallbacks
    assert st.alive is False, "the veto must end the attempt"
    assert not demo.calls, f"commanded after the veto: {demo.calls}"
    clear = _VetoDemo()
    clear._inside_rack = lambda x, y, r: False
    st = SimpleNamespace(obj_idx=0, status=None, alive=True)
    setup(clear, st)
    assert st.alive and clear._plan_fallbacks == {} and "drive_to" in clear.calls, (clear._plan_fallbacks, clear.calls)


def test_the_executor_records_checkpoint_clearance_and_replans_for_the_run_metrics():
    """T3: at each checkpoint the arm's actual margin on the checker's own predicate, and every
    replan, land on the demo for the metrics file; a model without `clearance` costs the motion
    nothing."""
    class _WithClearance(_AbortRecoverDemo):
        def _arm_model(self):
            m = super()._arm_model()
            m.clearance = lambda q8, obs, held, base, fingers=None: 0.0123
            return m

    demo, _ = _run_execute_arm_path_abort(plan_returns=True, carry_slip=0.0, demo_cls=_WithClearance)
    assert demo._bench_clearance == [("arm", 1, 0.0123)], demo._bench_clearance
    demo, _ = _run_hook_test(None)
    assert demo._bench_replans >= 1, "the succeeded replan was not counted"
    assert not hasattr(demo, "_bench_clearance"), "a model without clearance records nothing, silently"


def test_the_carry_check_records_its_reading_for_the_run_metrics():
    out = []
    check = _exec_method(("morph", "align.py"), "_carry_check",
                         {"np": np, "math": math, "print": lambda *a, **k: out.append(" ".join(str(x) for x in a))})

    class _Obj:
        def get_world_poses(self):
            return np.array([[0.5, 0.0, 0.1]]), None

    class _Demo:
        _obj_local = np.array([0.5, 0.0, 0.1])
        _obj_local_R = None

        def _hand_frame(self):
            return np.zeros(3), np.eye(3)

        def _obj_R(self, obj):
            return np.eye(3)

    demo = _Demo()
    slip = check(demo, _Obj(), "released")
    assert slip == 0.0 and demo._bench_slip == [("released", 0.0, 0.0, None)], (slip, demo._bench_slip)


def test_the_grip_capture_reports_the_carry_attitude_and_the_held_box():
    """The payload box's headroom is read off this line, so the half-extents must be the
    CYLINDER's own support (`h|a| + r*sqrt(1-a^2)`), not the wider re-box of its object-frame
    AABB -- the two differ by 12.0mm per side at the second attitude below. The attitude is the
    GRASP's, `_obj_local_R[:, 2]`, and the centre is `_obj_local`, not the rotation beside it."""
    out = []
    cap = _exec_method(("morph", "align.py"), "_capture_grip_offset",
                       {"np": np, "math": math, "KNOWN": {"object_radius": 0.040},
                        "print": lambda *a, **k: out.append(" ".join(str(x) for x in a))})

    class _Obj:
        def get_world_poses(self):
            return np.array([[0.0, 0.0, 0.5]]), np.array([[1.0, 0.0, 0.0, 0.0]])

    class _Demo:
        obj_half_h = 0.090

        def __init__(self, R):
            self.R = R

        def _hand_frame(self):
            return np.zeros(3), np.eye(3)

        def _quat_to_R(self, q):
            return self.R

    # The shipped relation: the cylinder's axis lands on a hand axis, so the box is the cylinder.
    on_axis = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    cap(_Demo(on_axis), _Obj())
    # "hand, ", not the bare degrees: "0.0deg" is a SUFFIX of "90.0deg" and the assert could not fail.
    assert "hand, 0.0deg off the nearest hand axis" in out[-1], out[-1]
    assert "half-extents [40.0, 90.0, 40.0]mm" in out[-1], out[-1]
    # The CENTRE is `_obj_local`; printing `_obj_local_R` beside it reads as a plausible triple.
    assert "about centre [0.0, 0.0, 500.0]mm" in out[-1], out[-1]

    # Worst case: the axis on the body diagonal, where the box is fattest in every direction.
    r2, r3, r6 = math.sqrt(2), math.sqrt(3), math.sqrt(6)
    diag = np.array([[1 / r2, 1 / r6, 1 / r3], [-1 / r2, 1 / r6, 1 / r3], [0.0, -2 / r6, 1 / r3]])
    cap(_Demo(diag), _Obj())
    assert "hand, 54.7deg off the nearest hand axis" in out[-1], out[-1]
    assert "half-extents [84.6, 84.6, 84.6]mm" in out[-1], out[-1]

    # A rotation product can put |a_i| a float step past 1, where an unclipped 1 - a*a is
    # NEGATIVE and the whole line prints nan.
    _past = np.nextafter(1.0, 2.0)
    over = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -_past], [0.0, 1.0, 0.0]])
    cap(_Demo(over), _Obj())
    assert "nan" not in out[-1], out[-1]
    assert "half-extents [40.0, 90.0, 40.0]mm" in out[-1], out[-1]

    # A non-finite pose must not read as a confident upright grasp: `min(1.0, nan)` is 1.0.
    cap(_Demo(np.full((3, 3), np.nan)), _Obj())
    assert out[-1] == (">>> grip offset: attitude unavailable -- "
                       "a non-finite hand or object pose"), out[-1]


# The close's six refusals: each ends an attempt that `run_cycle` retries, so the count is the only
# record a run that recovered ever hit one. (tag, file, the print that names the refusal.)
CLOSE_REFUSALS = [
    ("close-gate-unreachable", ("morph", "close", "__init__.py"), "CLOSE GATE: pose unreachable"),
    ("close-missed-grasp", ("morph", "close", "__init__.py"), "force-close loaded NO finger"),
    ("close-partial-latch", ("morph", "close", "__init__.py"), "partial latch, ABORT"),
    ("lowest-tilt-diverged", ("morph", "close", "lowest.py"), "diverged on tilt it"),
    ("settle-base-shoved", ("morph", "pick", "descend.py"), "BASE SHOVED"),
    ("descend-align-displaced", ("morph", "pick", "descend.py"), "fine-align FAILED"),
    ("capture-grasp-rejected", ("morph", "pick", "grasp.py"), "capture: GRASP REJECTED"),
]


def _print_text(call):
    """Every literal fragment of a `print(...)` call, f-strings included, as one string."""
    return "".join(c.value for c in ast.walk(call) if isinstance(c, ast.Constant)
                   and isinstance(c.value, str))


def test_the_close_refusals_are_counted_where_they_end_the_attempt():
    """The tag must sit in the SAME statement list as the print that names the refusal -- one
    block up and it counts attempts that carried on, which is exactly what makes the tally a lie.
    The exact-set test covers the registry; this pins the site and the OUTCOME class (SAFETY here
    would fail a run that retried and succeeded)."""
    for tag, rel, marker in CLOSE_REFUSALS:
        tree = ast.parse(open(os.path.join(ROOT, *rel)).read())
        prints = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and getattr(n.func, "id", "") == "print" and marker in _print_text(n)]
        assert len(prints) == 1, f"{rel[-1]}: expected 1 print naming {marker!r}, found {len(prints)}"
        blocks = [b for b in _blocks(tree)
                  if any(p is s.value for s in b if isinstance(s, ast.Expr) for p in [prints[0]])]
        assert len(blocks) == 1, f"{rel[-1]}: {marker!r} is not a statement of exactly one block"
        block = blocks[0]
        calls = [i for i, s in enumerate(block) if isinstance(s, ast.Expr)
                 and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Attribute)
                 and s.value.func.attr == "_fallback" and s.value.args
                 and isinstance(s.value.args[0], ast.Constant) and s.value.args[0].value == tag]
        assert calls, (
            f"{rel[-1]}: _fallback({tag!r}) must sit beside the {marker!r} print, in its branch")
        # Beside is not enough: past the branch's own `return`/`break` the statement is
        # unreachable and the refusal goes uncounted exactly as before.
        leaves = [i for i, s in enumerate(block)
                  if isinstance(s, (ast.Return, ast.Break, ast.Continue, ast.Raise))]
        stop = min([block.index(next(s for s in block if isinstance(s, ast.Expr)
                                     and s.value is prints[0]))] + leaves)
        assert calls[0] < stop, (
            f"{rel[-1]}: _fallback({tag!r}) sits after the branch's print or its exit -- "
            f"unreachable or too late to count this refusal")
        assert FALLBACK_TAGS[tag][0] == "OUTCOME", (
            f"{tag} is not OUTCOME: SAFETY fails a run that retried this refusal and succeeded")


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
