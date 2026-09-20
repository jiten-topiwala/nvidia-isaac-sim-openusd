"""The grasp-side branches the shipped profile never selects, and what must survive their removal.
Each test pins the SURVIVING path, so an over-wide deletion fails here and not on the robot.
Offline: no Isaac, no simulator. Run: ./.venv/bin/python3 tests/test_grasp_flag_deletions.py"""
import ast
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(*rel):
    return open(os.path.join(ROOT, *rel), encoding="utf-8").read()


def _fn(src, name):
    tree = ast.parse(src)
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_the_lift_keeps_only_the_computed_column_ramp():
    fn = _fn(_src("morph", "pick", "grasp.py"), "_pick_lift")
    lift = ast.unparse(fn)
    assert "LIFT_MODE" not in lift, "the LIFT_MODE selector is back, and the replay branch with it"
    assert "_play_traj" not in lift, "_pick_lift replays a baked trajectory again"
    for j in ("ColumnLeftBearingJoint_1", "ColumnRightBearingJoint_1"):
        assert j in lift, f"the computed column ramp lost {j} -- nothing raises the object"
    # One route reaches the verdict, and it is the column ramp's: the replay branch ended
    # `self.lifted = self._finite("lift-replay")`.
    tags = [n.value.args[0].value for n in ast.walk(fn) if isinstance(n, ast.Assign)
            and any(ast.unparse(t) == "self.lifted" for t in n.targets)
            and isinstance(n.value, ast.Call) and getattr(n.value.func, "attr", "") == "_finite"]
    assert tags == ["lift-columns"], (
        f"_pick_lift's finite checks are {tags}: there must be exactly one and it must be the "
        f"column ramp's")


def test_the_lift_gate_is_held_alone():
    """`st.replay` is constant True, `_columns` is gone and LIFT_REPLAY had one reader and no
    writer -- it was the replay's flag and the replay is deleted. `held` is the whole condition.
    The gate is located by the ramp it owns, not by its own text, so an inversion cannot hide."""
    fn = _fn(_src("morph", "pick", "grasp.py"), "_pick_lift")
    gates = [n for n in ast.walk(fn)
             if isinstance(n, ast.If) and "self._finite('lift-columns')" in ast.unparse(n)]
    assert len(gates) == 1, f"expected one If to own the column ramp, found {len(gates)}"
    test = ast.unparse(gates[0].test)
    assert test == "held", (
        f"the lift gate is `{test}`: `held` alone decides whether the object is raised, so an "
        f"inversion or a re-added flag read is a lift that runs on the wrong attempts")
    assert not gates[0].orelse, "the lift gate grew an else branch -- a second lift path is back"


def test_play_traj_is_gone_but_the_bake_loader_is_not():
    for rel in (("morph", "kinematics.py"), ("morph", "pick", "grasp.py"),
                ("morph", "pick", "dock.py"), ("play_isaac.py",)):
        assert "_play_traj" not in _src(*rel), f"{'/'.join(rel)} still carries _play_traj"
        assert "_tighten_total" not in _src(*rel), (
            f"{'/'.join(rel)} still carries _tighten_total, whose only reader was _play_traj")
    assert "def _load_trajs(self):" in _src("morph", "kinematics.py"), (
        "`_load_trajs` went with `_play_traj`, but `morph/pick/reach.py` and `morph/place` still "
        "read `self.traj`")


def _morph_sources():
    for base, _d, files in os.walk(os.path.join(ROOT, "morph")):
        for f in sorted(files):
            if f.endswith(".py"):
                yield os.path.join(base, f), open(os.path.join(base, f), encoding="utf-8").read()


def _shipped_sources():
    """Everything a flag could reappear in: the package plus the entry points and tools beside it."""
    yield from _morph_sources()
    tools = sorted(f for f in os.listdir(os.path.join(ROOT, "tools")) if f.endswith(".py"))
    for rel in (["scene.py"], ["play_isaac.py"], ["grip_bench.py"], ["smoke_test.py"],
                *(["tools", f] for f in tools)):
        path = os.path.join(ROOT, *rel)
        # The research bench and most of `tools/` exist in the development tree only; a flag
        # cannot reappear in a file that is not there.
        if not os.path.exists(path):
            continue
        yield path, open(path, encoding="utf-8").read()


def test_the_wrist_level_probes_are_gone_from_the_pipeline():
    """`_level_wrist` and `_level_gradient` both return None when `_drive_on()`, and ARM_DRIVE
    ships "1" unconditionally (morph/config.py), so every caller -- the carry level on stage
    "lift", the pre-dock and place-dock levels on stage "place" -- got None."""
    for path, src in _morph_sources():
        for dead in ("_level_wrist", "_level_gradient", "WRIST_LEVEL"):
            assert dead not in src, f"{os.path.relpath(path, ROOT)} still carries {dead}"
    assert "_level_wrist" not in _src("smoke_test.py"), "DEMO_METHODS still lists _level_wrist"
    align = _src("morph", "align.py")
    for keep in ("def _tilt_deg(self, obj):", "def _carry_check(self, obj, tag):"):
        assert keep in align, f"morph/align.py lost `{keep}`, which the probes only USED"


def test_the_descent_keeps_its_own_wrist_compensation():
    """`morph/close/lowest.py`'s `_wl_i` is not a probe: it counter-rotates gripper_y against the
    LUT boom pitch every step of the extension, and nothing gates it on drives. WRIST_LEVEL ships
    unset, i.e. "1", so the compensation was live and stays live -- unconditional now."""
    fn = _fn(_src("morph", "close", "lowest.py"), "_reach_lowest")
    assigns = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "_wl_i" for t in n.targets)]
    assert len(assigns) == 1, f"expected one `_wl_i` assignment, found {len(assigns)}"
    value = ast.unparse(assigns[0].value)
    assert value == "self.idx.get('gripper_y_rotation_1')", (
        f"`_wl_i` is `{value}`: the shipped value of WRIST_LEVEL selects the joint, so anything "
        f"else either re-gates the compensation or deletes it")
    guards = [n for n in ast.walk(fn) if isinstance(n, ast.If) and "_wl_i" in ast.unparse(n.test)]
    tests = sorted({ast.unparse(g.test) for g in guards})
    assert len(guards) == 2 and tests == ["_wl_i is not None"], (
        f"the compensation's guards are {tests} ({len(guards)} of them): the capture and the "
        f"per-step write each test `_wl_i is not None`, and nothing else may gate them")
    writes = [n for g in guards for n in ast.walk(g)
              if isinstance(getattr(n, "ctx", None), ast.Store) and ast.unparse(n) == "_qc[_wl_i]"]
    assert writes, (
        "nothing inside the guard writes `_qc[_wl_i]` -- the descent stopped counter-rotating "
        "gripper_y against the LUT boom pitch, so the recorded wrist no longer matches the boom")


def _profile_keys():
    """Every key PROFILE ends up holding -- its dict literal plus every `PROFILE.update({...})`.
    Read from `morph/config.py` as source: importing it sets the whole environment as a side
    effect."""
    tree = ast.parse(_src("morph", "config.py"))
    keys = set()
    for n in ast.walk(tree):
        d = None
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "PROFILE" for t in n.targets):
            d = n.value
        elif (isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "update"
                and getattr(n.func.value, "id", "") == "PROFILE"):
            d = n.args[0]
        if isinstance(d, ast.Dict):
            keys |= {k.value for k in d.keys if isinstance(k, ast.Constant)}
    assert keys, "no PROFILE dict found in morph/config.py -- the reader is broken"
    return keys


def test_the_close_no_longer_carries_the_symmetric_squeeze():
    """`_close_sym_squeeze` opened with `if os.environ.get("CLOSE_SYM_SQUEEZE", "1") != "1":
    return` and the profile ships "0", so the guard fired and the whole body was dead. The stage
    that DOES load the pads is `_close_force_balance`, which Dm made unconditional."""
    for path, src in _morph_sources():
        for dead in ("CLOSE_SYM_SQUEEZE", "_close_sym_squeeze", "SQ_HOLD"):
            assert dead not in src, f"{os.path.relpath(path, ROOT)} still carries {dead}"
    for dead in ("_close_sym_squeeze", "CLOSE_SYM_SQUEEZE"):
        assert dead not in _src("smoke_test.py"), f"smoke_test.py still lists {dead}"
    assert "CLOSE_SYM_SQUEEZE" not in _profile_keys(), "the profile still sets CLOSE_SYM_SQUEEZE"
    close = _src("morph", "close", "__init__.py")
    assert "close_gripper(self, st)" in close, (
        "the force-balance close went with the squeeze -- it is the stage that loads the pads")
    for path, src in _morph_sources():
        assert not _reads_env(src, "GRIP_SQUEEZE"), (
            f"{os.path.relpath(path, ROOT)} reads GRIP_SQUEEZE again -- its last reader was the "
            f"seat, deleted with the chain")


def _reads_env(src, key):
    """Every way the tree reads an environment variable: `environ.get`, `getenv` and a direct
    `environ[...]`, either quote."""
    pat = rf"""(environ\.get|getenv)\(\s*['"]{key}['"]|environ\[\s*['"]{key}['"]\s*\]"""
    return re.search(pat, src) is not None


def test_the_close_no_longer_carries_the_wrap_stage():
    """`_close_wrap`'s whole body sat under `if os.environ.get("WRAP", "1") == "1"` and the
    profile ships "0", so the method reduced to a read-back write of a flag since deleted. The
    encompassing
    close it held is dead with it; `WRAP_CURL` is a DIFFERENT flag, read by the force servo."""
    assert not os.path.exists(os.path.join(ROOT, "morph", "close", "wrap.py")), (
        "morph/close/wrap.py is back")
    for path, src in _morph_sources():
        for dead in ("WrapStage", "_close_wrap"):
            assert not re.search(rf"\b{dead}\b", src), (
                f"{os.path.relpath(path, ROOT)} still carries {dead}")
        assert not _reads_env(src, "WRAP"), (
            f"{os.path.relpath(path, ROOT)} still reads the WRAP flag")
    assert "_close_wrap" not in _src("smoke_test.py"), "smoke_test.py still lists _close_wrap"
    keys = _profile_keys()
    assert "WRAP" not in keys, "the profile still sets WRAP"
    assert "WRAP_CURL" in keys, (
        "WRAP_CURL went with the wrap stage, but `_close_force_balance` reads it for the distal "
        "curl and the profile pins it at 0.0")
    assert "def _wrap_gap(self, f, oc, r, hh, R=None):" in _src("morph", "gripper", "fingers.py"), (
        "`_wrap_gap` went with the stage it is named after, but the enclose, align and force "
        "stages all measure with it")


def test_the_close_no_longer_carries_the_legal_stage():
    """`_close_legal` was double-dead: the seat chain skipped it whenever the force close held all
    three, and its own body gate LEGAL_CLOSE ships "0" with the only statement outside the gate
    writing back the value it read. `_lj_stop` was called from that body alone."""
    assert not os.path.exists(os.path.join(ROOT, "morph", "close", "legal.py")), (
        "morph/close/legal.py is back")
    for path, src in _morph_sources():
        for dead in ("LegalStage", "_close_legal", "_lj_stop", "lj_locked"):
            assert not re.search(rf"\b{dead}\b", src), (
                f"{os.path.relpath(path, ROOT)} still carries {dead}")
        assert not _reads_env(src, "LEGAL_CLOSE"), (
            f"{os.path.relpath(path, ROOT)} still reads the LEGAL_CLOSE flag")
    smoke = _src("smoke_test.py")
    for dead in ("_close_legal", "_lj_stop"):
        assert dead not in smoke, f"smoke_test.py still lists {dead}"


def test_the_close_keeps_the_geometry_report_but_stores_nothing_from_it():
    """`_grasp_geometry_report` is a printing measurement; its return had one reader and the legal
    stage held it. Pinned by POSITION in `_close_replay`'s own body, not by presence: a report
    moved across the close, or hidden under a flag, measures a different hand or no hand."""
    for path, src in _shipped_sources():
        # `.jaw_axis` and the slot string, not the bare name: the report's own local was called
        # that and has been deleted with the return.
        assert not re.search(r"\.jaw_axis\b|[\"']jaw_axis[\"']", src), (
            f"{os.path.relpath(path, ROOT)} carries jaw_axis again: the report's return has no "
            f"reader and must not be stored")
    body = _fn(_src("morph", "close", "__init__.py"), "_close_replay").body

    def _at(pred):
        return [k for k, st in enumerate(body) if pred(ast.unparse(st))]

    reports = {}
    for k, st in enumerate(body):
        if not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Call)
                and getattr(st.value.func, "attr", "") == "_grasp_geometry_report"):
            continue
        tag = next(a.value for a in st.value.keywords if a.arg == "tag")
        reports[tag.value] = k
    assert sorted(reports) == ["close-entry", "post-close"], (
        f"the close's UNNESTED geometry reports are {sorted(reports)}: both must be plain "
        f"statements of `_close_replay`, neither under an `if`/`try` and neither binding a return")
    closes = _at(lambda u: u == "close_gripper(self, st)")
    assert len(closes) == 1, f"{len(closes)} unnested `close_gripper` calls: the order pin is blind"
    assert reports["close-entry"] < closes[0] < reports["post-close"], (
        f"the reports sit at {reports} around the close at {closes[0]}: one measures the hand "
        f"BEFORE the fingers move and one measures the grasp they made")


def test_the_lift_never_re_tightens_the_fingers():
    """LIFT_TIGHTEN ships "0", so `"0" == "1"` was False and the mid-lift re-squeeze never ran.
    `_pick_lift` must command no finger at all now: that block was its only writer, and the
    profile's reason for "0" is that re-squeezing breaks the anchors the friction hold rides on."""
    src = _src("morph", "pick", "grasp.py")
    fn = _fn(src, "_pick_lift")
    body = ast.unparse(fn)
    assert "_tight" not in body, "the tighten bookkeeping survives its own block"
    # Every Store, not just Assign targets: `self._finger_cmd[pos] += 0.02` is an AugAssign.
    writes = [n for n in ast.walk(fn)
              if isinstance(getattr(n, "ctx", None), ast.Store)
              and ast.unparse(n).startswith("self._finger_cmd")]
    assert not writes, (
        f"_pick_lift commands the fingers again ({[ast.unparse(t) for t in writes]}) -- the lift "
        f"holds what the close latched, it does not add to it")
    for path, s2 in _morph_sources():
        assert not _reads_env(s2, "LIFT_TIGHTEN"), (
            f"{os.path.relpath(path, ROOT)} still reads LIFT_TIGHTEN")
    assert "LIFT_TIGHTEN" not in _profile_keys(), "the profile still sets LIFT_TIGHTEN"


FINGER_MOTIONS = ("_hold", "hold_gripper", "move_gripper", "open_gripper")

# The arguments that must SURVIVE at the call sites the `pin` deletion rewrote.
SURVIVING_KEYWORDS = {
    ("morph/pick/descend.py", "_hold"): [("on_step",)],
    ("morph/pick/grasp.py", "hold_gripper"): [("kin",), ("kin", "on_step")],
    # the capture's two holds reach `_hold` through this one forward
    ("morph/gripper/api.py", "_hold"): [("kin", "on_step")],
    ("morph/place/__init__.py", "_hold"): [("kin",)],
}


def _called_name(call):
    """The gripper verbs are module-level functions, `_hold` a method: one is a Name, one an
    Attribute, and matching only the latter would make every gripper call site invisible."""
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")


def _finger_motion_calls():
    """Every `ast.Call` to a FINGER_MOTIONS name under morph/, with the files that would not parse.

    AST, not text: two of these call sites spread their arguments over several lines, so a
    line-oriented search reads a keyword off a different line than the call it belongs to."""
    calls, unparsed = [], []
    for path, src in _morph_sources():
        rel = os.path.relpath(path, ROOT)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            unparsed.append(rel)
            continue
        calls += [(rel, n) for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and _called_name(n) in FINGER_MOTIONS]
    return calls, unparsed


def test_no_finger_motion_is_called_with_the_dead_pin_argument():
    """`pin` appeared in `_hold`'s and `move_gripper`'s signatures and in neither body: both
    pin through `_pin_brake`, `set_base` and the caller's `on_step`. Four call sites passed it."""
    calls, unparsed = _finger_motion_calls()
    assert unparsed == [], (
        f"{unparsed} does not parse on the 3.10 venv, so this sweep skips it in silence: every "
        f"shipped file must parse here")
    assert len(calls) >= 4, (
        f"only {len(calls)} calls to {FINGER_MOTIONS} found -- the universe is empty or the "
        f"walk is broken, and the assertion below cannot fail")
    passed = [f"{rel}:{n.lineno}" for rel, n in calls
              if any(kw.arg == "pin" for kw in n.keywords)]
    assert not passed, f"pin= is passed at {passed}; it is dead in both callees"
    for rel, name in (("morph/robot.py", "_hold"), ("morph/gripper/api.py", "move_gripper")):
        args = _fn(_src(*rel.split("/")), name).args
        declared = args.posonlyargs + args.args + args.kwonlyargs
        assert "pin" not in [a.arg for a in declared], f"{name} still declares `pin`"
        assert args.kwarg is None, (
            f"{name} grew a **kwargs, which swallows `pin=` at every call site and makes the "
            f"call-site assertion above unfalsifiable")
    survivors = {}
    for rel, n in calls:
        survivors.setdefault((rel, _called_name(n)), []).append(
            tuple(sorted(kw.arg for kw in n.keywords)))
    for key, kws in SURVIVING_KEYWORDS.items():
        assert sorted(survivors.get(key, [])) == kws, (
            f"{key[0]} calls {key[1]} with {sorted(survivors.get(key, []))}, not {kws}: the "
            f"`pin` deletion must leave the other arguments of these calls untouched")


def test_the_palm_spread_is_gone_from_the_pick():
    """`s = 0.0` was the only assignment to `s`, so the guard's third disjunct `s <= 0.0` made
    `_pick_palm_spread` return before the eight statements below it, two of them `_finger_cmd`
    writes. Shape test: the universe is guarded, so an empty file list cannot pass it."""
    srcs = {}
    for base, _, names in os.walk(os.path.join(ROOT, "morph")):
        for name in names:
            if name.endswith(".py"):
                path = os.path.join(base, name)
                srcs[os.path.relpath(path, ROOT)] = open(path, encoding="utf-8").read()
    assert len(srcs) >= 40, f"only {len(srcs)} files under morph/ -- the walk is broken"
    for dead in ("_pick_palm_spread", "morph.pick.prep"):
        hits = sorted(rel for rel, src in srcs.items() if dead in src)
        assert not hits, f"{dead} is back in {hits}"
    cls = next(n for n in ast.walk(ast.parse(srcs["morph/pick/__init__.py"]))
               if isinstance(n, ast.ClassDef) and n.name == "PickMixin")
    bases = [ast.unparse(b) for b in cls.bases]
    assert len(bases) >= 6, f"PickMixin has {len(bases)} bases: the stage mixins are gone, not prep"
    assert "PrepStage" not in bases, f"PickMixin still mixes in PrepStage: {bases}"


def test_the_one_mode_flags_are_not_read_again():
    """Dm resolves five switches to the values a clean shell already reads: the GRASP_PROFILE gate
    (PROFILE now always applies), FORCE_CLOSE, CLOSE_DIRECT, ALIGN_MODE and ENCLOSE_FIRST."""
    dead = ("GRASP_PROFILE", "FORCE_CLOSE", "CLOSE_DIRECT", "ALIGN_MODE", "ENCLOSE_FIRST")
    for path, src in _shipped_sources():
        for key in dead:
            assert not _reads_env(src, key), (
                f"{os.path.relpath(path, ROOT)} reads {key} again -- it is resolved, not optional")
    assert not (set(dead) & _profile_keys()), f"the profile sets one of {dead} again"
    cfg = ast.parse(_src("morph", "config.py"))
    setdefaults = [n for n in ast.walk(cfg) if isinstance(n, ast.Call)
                   and getattr(n.func, "attr", "") == "setdefault"
                   and any(isinstance(a, ast.Name) for a in n.args)]
    assert len(setdefaults) == 2, (
        f"{len(setdefaults)} loop setdefault calls in morph/config.py: PROFILE and the ARM_DRIVE "
        f"implied set are the two, and the ARM_DRIVE sub-gate must survive")
    nested = [n for n in ast.walk(cfg) if isinstance(n, ast.If)
              and _reads_env(ast.unparse(n.test), "ARM_DRIVE")
              and any(c in setdefaults for c in ast.walk(n))]
    assert len(nested) == 1, (
        f"{len(nested)} ARM_DRIVE-gated setdefault loops: the implied set must stay behind its "
        f"own gate, so ARM_DRIVE=0 still skips it")


def test_the_missed_grasp_abort_is_unconditional():
    """The constant Dm left behind decides behaviour, so it is pinned by the code it owns, not by
    its own text: an inversion or a re-added flag read cannot hide."""
    fn = _fn(_src("morph", "close", "__init__.py"), "_close_replay")
    abort = [n for n in ast.walk(fn) if isinstance(n, ast.If)
             and any("missed grasp, ABORT" in ast.unparse(c) for c in n.body)]
    assert len(abort) == 1, f"expected one missed-grasp abort, found {len(abort)}"
    assert ast.unparse(abort[0].test) == "len(st.fb_held) == 0", (
        f"the missed-grasp abort is gated on `{ast.unparse(abort[0].test)}`: it is unconditional, "
        f"so nothing but the no-finger condition itself may decide it")


DEAD_REPLAY_FILES = ("replay.py", "joint_servo.py", "enclose.py")
DEAD_REPLAY_NAMES = ("ReplayStage", "EncloseStage", "JointServoStage", "_close_do_replay",
                     "_close_enclose", "_close_align_eq", "_enclose_joint_servo", "_reach_polar",
                     "_tilt_to_obj_z", "_align_to_recorded", "_direct")
DEAD_REPLAY_FLAGS = ("ENCLOSE", "ENCLOSE_JOINT", "ENCLOSE_A1_DESCEND", "ENCLOSE_A1_MAX",
                     "ENCLOSE_DU", "ENCLOSE_DV_HOLD", "ENCLOSE_DV_TGT", "ENCLOSE_KEEP_INSERT",
                     "ENCLOSE_LOWEST", "ENCLOSE_OPEN_DISTAL", "ENCLOSE_OPEN_J1", "ENCLOSE_POLAR",
                     "ENCLOSE_QV", "ENCLOSE_STOP_ON", "ENCLOSE_STRAIGHTEN", "ENCLOSE_TH_CAP",
                     "ENCLOSE_Z_MODE", "ALIGN_EQ", "ALIGN_EQ_ARM", "EQ_ALL_LINKS", "TILT_FIRST",
                     "CLOSE_SLOW", "CLOSE_STANDOFF", "CLOSE_GEO_STOP")


def _deletion_sources():
    """The shipped sources plus the suites beside them, minus this file -- which names the dead
    strings on purpose and would otherwise refute itself."""
    yield from _shipped_sources()
    tdir = os.path.join(ROOT, "tests")
    for f in sorted(os.listdir(tdir)):
        path = os.path.join(tdir, f)
        if f.endswith(".py") and path != os.path.abspath(__file__):
            yield path, open(path, encoding="utf-8").read()


def test_the_recorded_replay_arc_is_gone():
    """Dm resolved `_direct` to True, so the `else` that ran the recorded replay, the enclose and
    the gap-equalisation was unreachable, and with it the joint servo and its two lowest-pose
    entries. Their flags have no read site left, so the profile may not set them either."""
    for f in DEAD_REPLAY_FILES:
        assert not os.path.exists(os.path.join(ROOT, "morph", "close", f)), (
            f"morph/close/{f} is back")
    for path, src in _deletion_sources():
        rel = os.path.relpath(path, ROOT)
        for dead in DEAD_REPLAY_NAMES:
            assert not re.search(rf"\b{dead}\b", src), f"{rel} still carries {dead}"
        for flag in DEAD_REPLAY_FLAGS:
            assert not _reads_env(src, flag), f"{rel} reads the dead {flag} flag again"
    assert not (set(DEAD_REPLAY_FLAGS) & _profile_keys()), (
        f"the profile sets one of {sorted(set(DEAD_REPLAY_FLAGS) & _profile_keys())} again")


def test_the_direct_close_is_the_only_path():
    """What Dc1b leaves must stay reachable: the two aligns, every surviving stage base, and a
    body no environment read can switch off -- the job the retired `_direct` pin used to do."""
    src = _src("morph", "close", "__init__.py")
    fn = _fn(src, "_close_replay")
    args = [a.arg for a in fn.args.args]
    assert args == ["self", "q_base", "obj", "status"], (
        f"`_close_replay` takes {args}: `secs` was the recorded replay's duration and has no "
        f"reader left, so it may not come back as a pass-through")
    body = ast.unparse(fn)
    for call in ("self._align_to_grasp_center(obj)", "self._align_z_to_center(obj)"):
        assert call in body, f"`{call}` is gone: the direct close has no other approach authority"
    parent = {c: n for n in ast.walk(fn) for c in ast.iter_child_nodes(n)}

    def env_gated(node):
        while node in parent:
            node = parent[node]
            if isinstance(node, ast.If) and "os.environ" in ast.unparse(node.test):
                return ast.unparse(node.test)
        return None
    for call in [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and ast.unparse(n).split("(")[0] in
                 ("self._align_to_grasp_center", "self._align_z_to_center",
                  "close_gripper")]:
        gate = env_gated(call)
        assert gate is None, (
            f"`{ast.unparse(call)}` sits under `{gate}`: the close is unconditional, so no "
            f"environment read may decide whether it runs")
    cls = next(n for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.ClassDef) and n.name == "CloseMixin")
    assert [ast.unparse(b) for b in cls.bases] == [
        "ApproachAlignStage", "LowestStage", "RampStage", "ForceBalanceStage"], (
        f"CloseMixin's bases are {[ast.unparse(b) for b in cls.bases]}: the seat chain is deleted "
        f"and every surviving stage still ships")


if __name__ == "__main__":
    test_the_lift_keeps_only_the_computed_column_ramp()
    test_the_lift_gate_is_held_alone()
    test_play_traj_is_gone_but_the_bake_loader_is_not()
    test_the_wrist_level_probes_are_gone_from_the_pipeline()
    test_the_descent_keeps_its_own_wrist_compensation()
    test_the_close_no_longer_carries_the_symmetric_squeeze()
    test_the_close_no_longer_carries_the_wrap_stage()
    test_the_close_no_longer_carries_the_legal_stage()
    test_the_close_keeps_the_geometry_report_but_stores_nothing_from_it()
    test_the_lift_never_re_tightens_the_fingers()
    test_no_finger_motion_is_called_with_the_dead_pin_argument()
    test_the_palm_spread_is_gone_from_the_pick()
    test_the_one_mode_flags_are_not_read_again()
    test_the_missed_grasp_abort_is_unconditional()
    test_the_recorded_replay_arc_is_gone()
    test_the_direct_close_is_the_only_path()
    print("all passed")
