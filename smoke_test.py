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

ROOT = os.path.dirname(os.path.abspath(__file__))

# Every method of the composed `Demo` class. As methods move between the `morph/` mixins this set
# must stay exactly the same — that is the point of the check.
DEMO_METHODS = {
    "CHASSIS_PLATE_X", "CHASSIS_PLATE_Y", "_R_to_quat", "_a1_clearing_chassis", "_align_height",
    "_align_recorded_relation", "_align_to_grasp_center", "_align_to_recorded",
    "_align_z_to_center", "_animate_fingers", "_apply", "_apply_grip_material",
    "_arm_body_clearance", "_arm_insert", "_capture_grip_offset", "_carry_check",
    "_clears_chassis", "_close_force_balance", "_close_replay", "_close_sym_squeeze",
    "_closure_passives", "_drive_stamp", "_du_dv", "_enclose_joint_servo", "_fine_align",
    "_finger_centroid_xy", "_finger_curl_targets", "_finger_efforts", "_finger_gap_owner",
    "_finger_gap_parts", "_finger_hulls", "_finger_j1_force", "_finger_lever",
    "_finger_reaction", "_finger_surface_gap", "_finger_world_pts", "_finite", "_fk_z",
    "_force", "_grasp_geometry_report", "_grip_fk", "_grip_frame", "_grip_gentle",
    "_grip_stiffen", "_hand_frame", "_hand_prims", "_hand_touching_object", "_hold",
    "_insert_sequence", "_inside_rack", "_jog_tick", "_jog_write", "_knuckle_hulls",
    "_level_gradient", "_level_wrist", "_lj_stop", "_load_close_traj",
    "_load_trajs", "_lut_cell", "_lut_interp", "_lut_z", "_bisect_a1", "_park_ramp",
    "_mouth_depth", "_mouth_offset", "_nav_discs",
    "_obj_R", "_obj_collision", "_obj_state", "_obj_watch", "_pad_force_step", "_pad_trace",
    "_phase_report", "_pick_palm_spread", "_pin_grip", "_pinch", "_play_traj",
    "_pre_close_open", "_quat_to_R", "_ramp", "_ramp_arm", "_reach_lowest", "_reach_polar",
    "_rec_target", "_recover", "_set_arm_gain", "_settle_hand", "_solve_lowest",
    "_solve_reach_z", "_spawn_keepout", "_spin_wheels", "_spin_wheels_mecanum",
    "_sync_object_size", "_tilt_deg", "_tilt_to_obj_z", "_tip_gaps", "_tip_prims",
    "_tip_report", "_unweld_grip", "_weld_grip", "_with_closure", "_wrap_gap", "base_ledger",
    "base_pose", "drive_to", "grip_pos", "idle_step", "park", "pick", "pin_chassis", "place",
    "run_cycle", "scatter_objects", "set_base", "start_jog",
}

# The per-stage methods `pick` and `_close_replay` were split into. Same rule as above: each must
# exist exactly once, and the orchestrators must call them in this order.
STAGE_METHODS = {
    "_pick_setup", "_pick_pose", "_pick_reach", "_pick_settle", "_pick_descend", "_pick_close",
    "_pick_seat", "_pick_capture", "_pick_lift",
    "_close_do_replay", "_close_enclose", "_close_align_eq", "_close_wrap", "_close_legal",
    "_close_seat_bc", "_close_grip_seat", "_seat_wrap", "_close_sync_arm",
}

# The module-level names external tooling imports from `play_isaac`. Keep them exported.
PUBLIC_API = {"KNOWN", "simulation_app", "OBJ_HALF_H", "LOCAL_OBJ", "Demo",
              "NUM_OBJECTS", "SHELF_SLOTS", "find_path"}

ISAAC_ROOTS = ("isaacsim", "pxr", "omni", "carb")
# Per-file ceiling: a ratchet, not a target. It exists so a module cannot quietly grow back into the
# monolith the package was decomposed out of.
MAX_LINES = 1650

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
        check_public_api(trees)
        check_bootstrap(trees)
    check_sizes(paths)

    print()
    if _failures:
        print(f"{len(_failures)} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
