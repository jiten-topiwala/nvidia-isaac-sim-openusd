"""Fast structural checks — no GPU, no Isaac Sim, no simulation.

Run after every refactor commit:

    python3 smoke_test.py

Isaac Sim cannot be imported without booting `SimulationApp` (which needs a GPU and takes minutes),
so nothing here imports the application. Every check is done by parsing the source. That is enough
to catch what a decomposition actually gets wrong: a method silently lost or duplicated while being
moved between modules, a name the internal probes import disappearing, or the bootstrap ordering
being broken so `isaacsim` gets imported before `SimulationApp` exists.

A full behavioural check means running a cycle; this is the cheap gate before that.
"""
import ast
import os
import sys
import tokenize
import re

# Every check here parses the shipped source, so it must run on the interpreter that source is
# written for. A checker that cannot parse the syntax it checks has no verdict to give.
_MIN = (3, 12)
if sys.version_info[:2] < _MIN:
    sys.stderr.write(
        f"smoke_test.py needs Python {_MIN[0]}.{_MIN[1]}+ to parse the shipped tree; this is "
        f"{sys.version.split()[0]}.\n"
        "The .venv is the 3.10 OMPL planner environment and cannot parse 3.12 syntax.\n"
        "Run it on the simulator's interpreter instead:\n"
        "    \"$HOME/isaac-sim/python.sh\" smoke_test.py\n")
    raise SystemExit(2)

ROOT = os.path.dirname(os.path.abspath(__file__))

# Every method of the composed `Demo` class, ONE-WAY: a missing name is a FAILURE, an extra one is
# a note. Do not read a clean run as "the class is exactly this set".
DEMO_METHODS = {
    "_R_to_quat", "_a1_clearing_chassis",
    "_arm_model", "_arm_q8", "_arm_bounds", "_arm_obstacles", "_commanded_fingers", "_arm_base_prim", "_arm_base_world",
    "_traj_q8", "_plan_approach",
    "_align_to_grasp_center",
    "_align_z_to_center", "_apply", "_apply_grip_material",
    "_arm_body_clearance", "_arm_insert", "_capture_grip_offset", "_carry_check",
    "_clears_chassis", "_close_force_balance", "_close_replay",
    "_closure_passives", "_drive_stamp", "_du_dv",
    "_finger_efforts", "_finger_gap_owner",
    "_finger_gap_parts", "_finger_hulls", "_finger_j1_force", "_finger_lever",
    "_finger_reaction", "_finger_surface_gap", "_force_closure_report", "_finger_world_pts", "_finite", "_fk_z",
    "_force", "_grasp_geometry_report", "_grip_fk", "_grip_frame", "_grip_gentle",
    "_grip_stiffen", "_hand_frame", "_hand_prims", "_hold",
    "_insert_sequence", "_inside_rack", "_seat_step", "_jog_tick", "_jog_write", "_knuckle_hulls",
    "_load_close_traj",
    "_load_trajs", "_lut_cell", "_lut_interp", "_lut_z", "_bisect_a1", "_park_ramp",
    "_mouth_depth", "_nav_discs",
    "_obj_R", "_obj_collision", "_obj_state", "_obj_watch", "_pad_force_step", "_pad_trace",
    "_phase_report", "_pinch",
    "_quat_to_R", "_ramp_arm", "_reach_lowest",
    "_recover", "_set_arm_gain", "_settle_hand", "_solve_lowest",
    "_solve_reach_z", "_spawn_keepout", "_spin_wheels_mecanum",
    "_sync_object_size", "_tilt_deg", "_tip_gaps", "_tip_prims",
    "_tip_report", "_with_closure", "_wrap_gap", "base_ledger",
    "base_pose", "drive_to", "grip_pos", "idle_step", "park", "pick", "pin_chassis", "place",
    "run_cycle", "scatter_objects", "set_base", "start_jog",
}

# The per-stage methods `pick` and `_close_replay` split into. Same rule as above: each must exist
# exactly once. Nothing here checks call order.
STAGE_METHODS = {
    "_pick_attempt",                                     # the per-attempt orchestrator `pick` walks
    "_pick_setup", "_pick_pose", "_pick_reach", "_pick_settle", "_pick_descend", "_pick_close",
    "_pick_seat", "_pick_capture", "_pick_lift",
    "_close_sync_arm",
}

# The module-level names external tooling imports from `play_isaac`. Keep them exported.
PUBLIC_API = {"KNOWN", "simulation_app", "OBJ_HALF_H", "LOCAL_OBJ", "Demo",
              "NUM_OBJECTS", "SHELF_SLOTS", "find_path"}

ISAAC_ROOTS = ("isaacsim", "pxr", "omni", "carb")
# Per-file ceiling: a ratchet, not a target. It exists so a module cannot quietly grow back into the
# monolith the package was decomposed out of.
MAX_LINES = 1250
MAX_DOCSTRING_LINES = 8
MAX_COMMENT_BLOCK = 10
MAX_TASK_IDS = 3

_failures = []


def _fail(msg):
    _failures.append(msg)
    print(f"[FAIL] {msg}")


def _ok(msg):
    print(f"[OK] {msg}")


def _sources():
    """Every shipped .py file: the package plus the entry points."""
    out = [os.path.join(ROOT, f) for f in ("play_isaac.py", "scene.py", "nav_plan.py")
           if os.path.exists(os.path.join(ROOT, f))]
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, "morph")):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        out += [os.path.join(dirpath, f) for f in sorted(filenames) if f.endswith(".py")]
    return out


def check_free_names(trees):
    """Every name a function reads as a global resolves at its module's scope (an import, a def,
    a class, an assignment) or is a builtin. A function no offline suite executes -- the pick's
    goal derivation is one -- would otherwise carry a NameError to the first live cycle."""
    import builtins
    import symtable
    bad = []
    for p, tree in trees.items():
        rel = os.path.relpath(p, ROOT)
        src = open(p, encoding="utf-8").read()
        defined = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                defined.update((a.asname or a.name).split(".")[0] for a in n.names)
        for n in ast.walk(tree):
            if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    for leaf in ast.walk(t):
                        if isinstance(leaf, ast.Name):
                            defined.add(leaf.id)
            elif isinstance(n, (ast.For, ast.With, ast.comprehension)):
                for leaf in ast.walk(n.target if not isinstance(n, ast.With) else ast.Module(body=[], type_ignores=[])):
                    if isinstance(leaf, ast.Name):
                        defined.add(leaf.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                defined.update((a.asname or a.name).split(".")[0] for a in n.names)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                defined.add(n.name)
            elif isinstance(n, ast.Global):
                defined.update(n.names)

        def walk(tab):
            if tab.get_type() == "function":
                for name in tab.get_globals():
                    if name not in defined:
                        bad.append(f"{rel}: {tab.get_name()}() reads `{name}`, defined nowhere in the module")
            for child in tab.get_children():
                walk(child)
        try:
            walk(symtable.symtable(src, rel, "exec"))
        except SyntaxError:
            continue
    for b in sorted(set(bad)):
        _fail(b)
    if not bad:
        _ok("every global a function reads is defined in its module")


def check_syntax(paths):
    trees = {}
    for p in paths:
        rel = os.path.relpath(p, ROOT)
        try:
            trees[p] = ast.parse(open(p).read(), filename=rel)
        except SyntaxError as e:
            _fail(f"syntax error in {rel}:{e.lineno}: {e.msg}")
    if len(trees) == len(paths):
        _ok(f"parsed {len(paths)} source files")
    return trees


def check_methods(trees):
    """Every Demo method exists exactly once across the class and any mixins it composes."""
    seen = {}
    for p, tree in trees.items():
        rel = os.path.relpath(p, ROOT)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            # `Demo` composes `*Mixin` classes, which in turn compose the per-stage `*Stage` classes
            # that hold the split-out bodies of `pick` and `_close_replay`.
            if node.name != "Demo" and not node.name.endswith(("Mixin", "Stage")):
                continue
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    seen.setdefault(m.name, []).append(f"{rel}:{node.name}")
                # class-level aliases count too: `_quat_to_R = staticmethod(quat_to_R)` delegates to
                # morph.geometry rather than keeping a second copy of the implementation
                elif isinstance(m, ast.Assign):
                    for t in m.targets:
                        if isinstance(t, ast.Name):
                            seen.setdefault(t.id, []).append(f"{rel}:{node.name}")

    want = DEMO_METHODS | STAGE_METHODS
    dupes = {n: w for n, w in seen.items() if len(w) > 1 and n in want}
    for n, w in sorted(dupes.items()):
        _fail(f"method defined more than once: {n} in {', '.join(w)}")

    missing = want - set(seen)
    for n in sorted(missing):
        _fail(f"method LOST in the refactor: {n}")

    extra = {n for n in set(seen) - want if not n.startswith("__")}
    if extra:
        print(f"[note] {len(extra)} method(s) not in the manifest (new work, or a rename): "
              f"{', '.join(sorted(extra)[:8])}{' ...' if len(extra) > 8 else ''}")

    if not dupes and not missing:
        _ok(f"all {len(want)} Demo/stage methods present exactly once")


def check_self_calls(trees):
    """Every `self.<name>(...)` call inside a Demo/mixin/stage method resolves to a method the
    composed class has, or to an attribute some method assigns. A method moved out of a mixin
    leaves its callers behind with an AttributeError the offline suites cannot reach."""
    has, assigned, calls = set(), set(), {}
    for p, tree in trees.items():
        rel = os.path.relpath(p, ROOT)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name != "Demo" and not node.name.endswith(("Mixin", "Stage")):
                continue
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    has.add(m.name)
                elif isinstance(m, ast.Assign):
                    has.update(t.id for t in m.targets if isinstance(t, ast.Name))
            for n in ast.walk(node):
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self":
                    if isinstance(getattr(n, "ctx", None), ast.Store):
                        assigned.add(n.attr)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                        and isinstance(n.func.value, ast.Name) and n.func.value.id == "self":
                    calls.setdefault(n.func.attr, set()).add(f"{rel}:{n.lineno}")
            # `self.x = ...` and `setattr`-free aliases
            for n in ast.walk(node):
                if isinstance(n, ast.Assign):
                    for t in n.targets:
                        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                            assigned.add(t.attr)
        # The arm package consumes the same surface through a `demo` parameter.
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or not node.args.args or node.args.args[0].arg != "demo":
                continue
            for n in ast.walk(node):
                if isinstance(n, ast.Assign):
                    for t in n.targets:
                        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "demo":
                            assigned.add(t.attr)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                        and isinstance(n.func.value, ast.Name) and n.func.value.id == "demo":
                    calls.setdefault(n.func.attr, set()).add(f"{rel}:{n.lineno}")
    unresolved = {n: w for n, w in calls.items() if n not in has and n not in assigned}
    for n, w in sorted(unresolved.items()):
        _fail(f"{n}() is called on the Demo at {', '.join(sorted(w)[:3])} but no Demo class defines it")
    if not unresolved:
        _ok(f"all {len(calls)} self/demo.<method>() names resolve on the composed Demo")


def check_public_api(trees):
    entry = os.path.join(ROOT, "play_isaac.py")
    tree = trees.get(entry)
    if tree is None:
        return _fail("play_isaac.py did not parse")
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            names |= {(a.asname or a.name) for a in node.names}
        elif isinstance(node, ast.Import):
            names |= {(a.asname or a.name.split(".")[0]) for a in node.names}
    missing = PUBLIC_API - names
    if missing:
        _fail(f"play_isaac.py no longer exposes {sorted(missing)} — the internal probes import these")
    else:
        _ok(f"public API intact ({len(PUBLIC_API)} names the probes rely on)")


def check_bootstrap(trees):
    """`SimulationApp(...)` must be constructed before any Isaac import in the entry module.

    Isaac's APIs come from runtime-loaded plugins, so importing them earlier fails. `morph/` modules
    are exempt: they are only ever imported by play_isaac.py, after the app exists.
    """
    entry = os.path.join(ROOT, "play_isaac.py")
    tree = trees.get(entry)
    if tree is None:
        return
    boot = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "SimulationApp"):
            boot = node.lineno if boot is None else min(boot, node.lineno)
    if boot is None:
        return _fail("play_isaac.py never constructs SimulationApp")

    early = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)) or node.lineno >= boot:
            continue
        mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                else [node.module or ""])
        for m in mods:
            if m.split(".")[0] in ISAAC_ROOTS and m != "isaacsim":
                early.append(f"line {node.lineno}: {m}")
    if early:
        _fail(f"Isaac imported before SimulationApp (line {boot}): {'; '.join(early)}")
    else:
        _ok(f"bootstrap ordering correct (SimulationApp at line {boot})")


def check_ratchets(paths, trees):
    task_id_pattern = re.compile(r'\b(F\d+\.\d+|G\d+\.\d+|Track\s+[A-Z][0-9]?|L\d{1,2}|C\d{1,2}|review\d+)\b')

    task_ids_count = 0
    longest_comment = 0
    longest_docstring = 0

    for p in paths:
        # Check comments
        try:
            tokens = list(tokenize.tokenize(open(p, 'rb').readline))
        except Exception:
            tokens = []

        for tok in tokens:
            if tok.type == tokenize.COMMENT:
                matches = task_id_pattern.findall(tok.string)
                task_ids_count += len(matches)

        # Contiguous comment block (pure comments)
        try:
            with open(p, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            current_len = 0
            last_line = -2
            for i, line in enumerate(lines):
                if line.strip().startswith('#'):
                    if i == last_line + 1:
                        current_len += 1
                    else:
                        current_len = 1
                    if current_len > longest_comment:
                        longest_comment = current_len
                    last_line = i
        except Exception:
            pass

    # Check docstrings
    for p, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node, clean=False)
                if docstring:
                    lines = docstring.split('\n')
                    if len(lines) > longest_docstring:
                        longest_docstring = len(lines)

    if longest_docstring > MAX_DOCSTRING_LINES:
        _fail(f"docstring ratchet: {longest_docstring} lines (ceiling {MAX_DOCSTRING_LINES})")
    else:
        _ok(f"docstring ratchet: {longest_docstring} lines (ceiling {MAX_DOCSTRING_LINES})")

    if longest_comment > MAX_COMMENT_BLOCK:
        _fail(f"comment block ratchet: {longest_comment} lines (ceiling {MAX_COMMENT_BLOCK})")
    else:
        _ok(f"comment block ratchet: {longest_comment} lines (ceiling {MAX_COMMENT_BLOCK})")

    if task_ids_count > MAX_TASK_IDS:
        _fail(f"task IDs ratchet: {task_ids_count} comments (ceiling {MAX_TASK_IDS})")
    else:
        _ok(f"task IDs ratchet: {task_ids_count} comments (ceiling {MAX_TASK_IDS})")

def check_sizes(paths):
    big = []
    for p in paths:
        n = sum(1 for _ in open(p))
        if n > MAX_LINES:
            big.append((os.path.relpath(p, ROOT), n))
    for rel, n in sorted(big, key=lambda t: -t[1]):
        _fail(f"{rel} is {n} lines (ceiling {MAX_LINES})")
    if not big:
        _ok(f"no shipped file exceeds {MAX_LINES} lines")


def main():
    paths = _sources()
    trees = check_syntax(paths)
    if len(trees) == len(paths):
        check_methods(trees)
        check_self_calls(trees)
        check_free_names(trees)
        check_public_api(trees)
        check_bootstrap(trees)
    check_ratchets(paths, trees)
    check_sizes(paths)

    print()
    if _failures:
        print(f"{len(_failures)} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
