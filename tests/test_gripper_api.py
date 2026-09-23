"""`morph/gripper/api.py` -- the contract of `move_gripper` and `open_gripper`.

House rule: observe the COMMAND channel, not the decision in front of it. `_apply` takes the finger
targets from `_finger_cmd`, so the per-step write to that channel IS the motion and every row below
reads the captured channel rather than the returned pose.
Most rows are PART E's, on the move's own contract. This file's own are the `on_step` row,
because nothing else fails when the callback is deleted and the release's open+ checkpoints ride
on it; the open's target-and-release row, because PART E judges no open; the empty-target refusal,
because `all()` over nothing is True; and the three rows pinning the curl close deleted, because
nothing else fails when it comes back.
Offline: no Isaac, no simulator. Run: ./.venv/bin/python3 tests/test_gripper_api.py"""
import ast
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import morph.gripper.api as _api                                                    # noqa: E402
from morph.arm.execute import commanded_fingers                                     # noqa: E402
from morph.gripper.api import GRIPPER_OPEN, move_gripper, open_gripper              # noqa: E402

FINGERS = "abc"
J1, J2, J3 = "finger_%s_joint_1_1", "finger_%s_joint_2_1", "finger_%s_joint_3_1"


def _names():
    """Arm one and arm two, with a non-finger joint between them: `_f_idx` spans both arms and is
    not contiguous, so a row that got the ordering wrong cannot pass by accident."""
    out = []
    for arm in ("1", "2"):
        for fl in FINGERS:
            if fl != "a":
                out.append(f"palm_finger_{fl}_joint_{arm}")
            out += [f"finger_{fl}_joint_{j}_{arm}" for j in (1, 2, 3)]
        out.append(f"ArmLeftJoint_{arm}")
    out.append("gripper_z_rotation_1")
    return out


class _Demo:
    """The surface `move_gripper` uses, capturing the finger channel on every `_apply`."""

    dt = 1.0 / 240.0

    def __init__(self):
        self.names = _names()
        self.idx = {n: i for i, n in enumerate(self.names)}
        self._f_idx = np.array([i for n, i in self.idx.items()
                                if n.startswith("finger_") or n.startswith("palm_finger")])
        self._f_names = [self.names[i] for i in self._f_idx]
        self._finger_cmd = None
        self._touch = {}
        self.chan = []            # `_finger_cmd` as each `_apply` saw it
        self.gains = []           # every gain/effort writer this motion called
        self.bases, self.steps, self.forces = [], 0, 0
        self.reads = []           # every pose the readback served, in order
        self.drives = True
        self.q_read = self.q_start()
        self.world = SimpleNamespace(step=self._step)
        self.robot = SimpleNamespace(get_joint_positions=self._readback)

    def _readback(self):
        self.reads.append(self.q_read.copy())
        return self.q_read

    def _drive_on(self):
        return self.drives

    def q_start(self):
        q = np.zeros(len(self.names))
        for n, i in self.idx.items():
            if n.endswith("_1") and ("finger" in n):
                q[i] = 0.6
            elif "finger" in n:
                # arm two: distinct, so a permutation is visible
                q[i] = 0.1 * (i + 1)
        return q

    def base_ledger(self):
        return (1.0, 2.0, 0.5)

    def set_base(self, bx, by, byaw):
        self.bases.append((bx, by, byaw))

    def _force(self, q):
        self.forces += 1

    def _apply(self, q):
        self.chan.append(None if self._finger_cmd is None
                         else np.asarray(self._finger_cmd, float).copy())
        # The drives lag their command, so a measured pose is never the commanded one.
        self.q_read = np.asarray(q, float) - 0.01

    def _step(self, render=False):
        self.steps += 1

    def _arm_model(self):
        return SimpleNamespace(joints=[n for n in self.names if n.endswith("_1")])

    def _set_arm_gain(self, *a):
        self.gains.append("_set_arm_gain")

    def _grip_stiffen(self, *a):
        self.gains.append("_grip_stiffen")

    def _grip_gentle(self, *a):
        self.gains.append("_grip_gentle")

    def set_gains(self, *a, **k):
        self.gains.append("set_gains")

    def set_max_efforts(self, *a, **k):
        self.gains.append("set_max_efforts")


OPEN = {"1": 0.0, "2": 0.05, "3": -0.10}


def _open_targets():
    t = {J1 % fl: OPEN["1"] for fl in FINGERS}
    t.update({J2 % fl: OPEN["2"] for fl in FINGERS})
    t.update({J3 % fl: OPEN["3"] for fl in FINGERS})
    t["palm_finger_b_joint_1"] = 0.0
    return t


def _run(**kw):
    d = _Demo()
    q = d.q_start()
    targets = _open_targets()
    out = move_gripper(d, q, targets, 0.5, **kw)
    return d, targets, out


def _series(d, jname):
    """The value commanded to one joint on every step, read off the finger channel."""
    pos = list(d._f_idx).index(d.idx[jname])
    return [float(c[pos]) for c in d.chan]


def _first_move(d, jname):
    s = _series(d, jname)
    return next((k for k, v in enumerate(s) if v != s[0]), len(s))


def test_the_open_vector_is_commanded_every_step_and_stays_latched():
    d, targets, _ = _run()
    assert len(d.chan) == 120 and all(c is not None for c in d.chan), (
        f"{len(d.chan)} applies, {sum(c is None for c in d.chan)} with no finger command")
    for jn, tgt in targets.items():
        assert abs(_series(d, jn)[-1] - tgt) < 1e-9, f"{jn} ends at {_series(d, jn)[-1]}, not {tgt}"
    assert d._finger_cmd is not None, (
        "`_finger_cmd` is cleared on return: `_apply` then falls back to the pose it is handed, "
        "which is the insert pose captured while the object was still held")
    latched = np.asarray(d._finger_cmd, float)
    for jn, tgt in targets.items():
        pos = list(d._f_idx).index(d.idx[jn])
        assert abs(float(latched[pos]) - tgt) < 1e-9, (
            f"`_finger_cmd` holds {latched[pos]} for {jn} on return, not the open target {tgt}: "
            f"the hand is re-commanded closed on the next `_apply`")


def test_the_motion_profile_is_a_windowed_smoothstep():
    d, targets, _ = _run(release=True)
    n = len(d.chan)
    for jn, tgt in targets.items():
        s = _series(d, jn)
        step = np.sign(tgt - s[0])
        diffs = np.diff(s) * step
        assert (diffs >= -1e-12).all(), f"{jn} reverses on the way to its target"
        assert len(set(np.round(s, 9))) >= 5, (
            f"{jn} takes {len(set(np.round(s, 9)))} distinct values over {n} steps: the hand jumps "
            f"to the target instead of ramping, which flicks the object off the slot")
    for fl in FINGERS:
        frac = _first_move(d, J3 % fl) / n
        assert frac >= 0.74, (
            f"finger {fl}'s distal joint starts moving at {frac:.2f} of the ramp; the phalanx "
            f"window holds it until 0.75 even on a release")


def test_release_zeroes_the_stagger():
    staggered = _run()[0]
    released = _run(release=True)[0]
    for jt in (J1, J2, J3):
        starts = {fl: _first_move(staggered, jt % fl) for fl in FINGERS}
        assert starts["a"] > starts["b"], (
            f"{jt % 'a'} and {jt % 'b'} start together at {starts}: nothing is staggered, so the "
            f"release cannot be shown to remove it")
        same = {_first_move(released, jt % fl) for fl in FINGERS}
        assert len(same) == 1, (
            f"{jt % 'x'} starts at {sorted(same)} across the fingers on a release: the "
            f"still-loaded finger pushes the object off the slot")


def test_on_step_is_invoked_on_every_step():
    d = _Demo()
    seen = []
    move_gripper(d, d.q_start(), _open_targets(), 0.5, on_step=lambda: seen.append(len(d.chan)))
    assert len(seen) == len(d.chan) == 120, (
        f"{len(seen)} on_step calls against {len(d.chan)} commands: the place release reads its "
        f"open+ checkpoints from this callback, and a missed step is a checkpoint never taken")
    assert seen == list(range(1, 121)), (
        f"on_step runs {seen[:3]}... relative to the commands: it must follow each `_apply`, or "
        f"it observes the step before the one it is reporting on")
    d2 = _Demo()
    move_gripper(d2, d2.q_start(), _open_targets(), 0.5, on_step=None)
    assert len(d2.chan) == 120, f"the motion needs a callback to run: {len(d2.chan)} commands"


def test_finger_cmd_stays_an_ndarray_parallel_to_f_idx():
    d, targets, _ = _run()
    assert isinstance(d._finger_cmd, np.ndarray), (
        f"`_finger_cmd` is {type(d._finger_cmd).__name__}; `_apply`'s fancy-index assignment and "
        f"grip_bench.save_state need an ndarray, and `commanded_fingers` hides the difference")
    assert len(d._finger_cmd) == len(d._f_idx), f"{len(d._finger_cmd)} values for {len(d._f_idx)} joints"
    got = commanded_fingers(d)
    for jn, tgt in targets.items():
        assert abs(got[jn] - tgt) < 1e-9, f"{jn} round-trips as {got[jn]}, not {tgt}"
    raw = dict(zip(d._f_names, np.asarray(d._finger_cmd, float)))
    q0 = d.q_start()
    for n2, v in raw.items():
        want = targets.get(n2, float(q0[d.idx[n2]]))
        assert abs(v - want) < 1e-9, (
            f"{n2} carries {v}, not {want}: the vector is not parallel to `_f_idx`. Arm two is "
            f"filtered out of `commanded_fingers`, so only this raw comparison can see it")


def test_the_release_path_touches_no_gains():
    witness = []
    # `scene.set_arm_gains` bypasses `_set_gains` entirely, so only a module-level stand-in can
    # witness it; the module names no `scene` today and the attribute is removed again below.
    had = hasattr(_api, "scene")
    prior = getattr(_api, "scene", None)
    _api.scene = SimpleNamespace(set_arm_gains=lambda *a, **k: witness.append("scene.set_arm_gains"))
    try:
        d, _, _ = _run()
        assert d.gains == [] and witness == [], (
            f"the release wrote gains: {d.gains + witness}. It runs with the object still in the "
            f"hand, so a stiffen here squeezes it on the way out")
    finally:
        if had:
            _api.scene = prior
        else:
            del _api.scene


def test_api_is_the_only_public_surface():
    """Shape test, deliberately: nothing outside morph/gripper/ may import anything under
    morph.gripper except `api`, and the package reaches nothing private in morph.arm."""
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True)
    files = out.stdout.split()
    assert out.returncode == 0 and files, "git ls-files gave nothing: the test would pass over an empty universe"
    assert any(f.startswith("morph/gripper/") for f in files), "morph/gripper/ is not tracked"
    bad, unparsed = [], []
    for rel in files:
        inside = rel.startswith("morph/gripper/")
        # tests exercise internals; nothing else is excused from the parse rule below
        if rel.startswith("tests/"):
            continue
        try:
            tree = ast.parse(open(os.path.join(ROOT, rel), encoding="utf-8").read())
        except SyntaxError as e:
            unparsed.append(f"{rel}:{e.lineno}: {e.msg}")
            continue
        for n in ast.walk(tree):
            mods = []
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods = [n.module]
                if n.module in ("morph.gripper", "morph.arm"):
                    mods = [f"{n.module}.{a.name}" for a in n.names]
            if inside:
                for mod in mods:
                    leaf = mod.rsplit(".", 1)[-1]
                    if mod.startswith("morph.arm") and leaf.startswith("_"):
                        bad.append(f"{rel}:{n.lineno} imports {mod}")
                continue
            for mod in mods:
                if mod.startswith("morph.gripper.") and mod != "morph.gripper.api":
                    bad.append(f"{rel}:{n.lineno} imports {mod}")
            if isinstance(n, ast.ImportFrom) and n.module == "morph.gripper.api":
                for a in n.names:
                    if a.name not in _api.__all__:
                        bad.append(f"{rel}:{n.lineno} imports {a.name}, not in api.__all__")
    assert not unparsed, (
        "shipped source the 3.10 venv cannot parse, so this test would skip it in silence: "
        + "; ".join(unparsed))
    assert not bad, "the gripper's private surface is reached from outside: " + "; ".join(bad)


def test_the_fingers_re_export_is_lazy_and_reaches_the_moved_module():
    """`api` must import offline, so `FingersMixin` is resolved on demand; the attribute error on
    an unknown name must stay an AttributeError, or `hasattr` on this module lies."""
    _expect(ModuleNotFoundError, lambda: _api.__getattr__("FingersMixin"),
            "the re-export does not reach morph.gripper.fingers, whose Isaac import must fail here")
    _expect(AttributeError, lambda: _api.__getattr__("Nope"),
            "an unknown attribute must raise AttributeError")


def _expect(exc, fn, why):
    try:
        fn()
    except exc:
        return
    except Exception as e:
        raise AssertionError(f"{why}: raised {type(e).__name__} instead of {exc.__name__}: {e}")
    raise AssertionError(why)


def test_q_cmd_is_the_command_and_q_meas_is_the_last_measured_pose_copied():
    d = _Demo()
    mv = move_gripper(d, d.q_start(), _open_targets(), 0.5)
    assert np.allclose(np.asarray(mv.q_cmd, float)[d._f_idx], np.asarray(d._finger_cmd, float)), (
        "`q_cmd` is not the commanded pose the loop ends on")
    assert not np.allclose(np.asarray(mv.q_cmd, float), np.asarray(mv.q_meas, float)), (
        "`q_cmd` and `q_meas` are the same vector: the drives lag their command, and a caller "
        "reading the measured pose off `q_cmd` never sees it")
    assert np.allclose(np.asarray(mv.q_meas, float), d.reads[-1]), (
        f"`q_meas` is not the LAST readback ({len(d.reads)} were served)")
    before = np.asarray(mv.q_meas, float).copy()
    d.q_read += 1.0
    assert np.allclose(np.asarray(mv.q_meas, float), before), (
        "`q_meas` aliases the articulation buffer: it changes under the caller on the next read")


def test_a_wrist_target_is_refused_before_anything_is_committed():
    """The package owns `finger_*`/`palm_finger_*` and nothing else. `q_base` still carries the
    wrist and `_apply` commands it every step -- the check covers the target dict, not `q_base`."""
    for extra, why in (("gripper_z_rotation_1", "a wrist joint the stub HAS"),
                       ("wrist_roll_9", "a joint absent from `idx`, which the move would drop")):
        d = _Demo()
        targets = _open_targets()
        targets[extra] = 0.0
        _expect(ValueError, lambda: move_gripper(d, d.q_start(), targets, 0.5),
                f"{extra} ({why}) was accepted as a finger target")
        assert d._finger_cmd is None and d.chan == [] and d.steps == 0, (
            f"the refusal of {extra} came after the hand was latched: "
            f"`_finger_cmd`={d._finger_cmd}, {len(d.chan)} commands, {d.steps} steps")


def test_a_target_set_that_drives_nothing_is_refused():
    """`all()` over no judged finger is True, so a close that moved not one joint would report
    arrival. The `idx` filter stays -- an articulation may genuinely lack a name."""
    d = _Demo()
    _expect(ValueError, lambda: move_gripper(d, d.q_start(), {"finger_z_joint_9_9": 0.0}, 0.5),
            "a target set the articulation carries none of ran the full motion")
    _expect(ValueError, lambda: move_gripper(d, d.q_start(), {}, 0.5),
            "an empty target set ran the full motion")
    assert d.steps == 0, f"the refusal came after {d.steps} steps of motion"
    d2 = _Demo()
    kept = dict(_open_targets())
    kept["finger_z_joint_9_9"] = 0.0
    move_gripper(d2, d2.q_start(), kept, 0.5)
    assert d2.steps == 120, (
        "one unknown name among eleven known ones was refused: the `idx` filter is what handles "
        "a joint this articulation does not carry")


def test_an_exception_propagates():
    """No exception-to-outcome conversion anywhere: the primitive raises exactly as `place` does."""
    d = _Demo()
    bad = _open_targets()
    bad[J1 % "a"] = object()
    _expect(TypeError, lambda: move_gripper(d, d.q_start(), bad, 0.5),
            "a target value that is not a number was swallowed while `moves` was built")

    d2 = _Demo()
    calls = {"k": 0}

    def _boom_once():
        calls["k"] += 1
        if calls["k"] == 5:
            raise RuntimeError("on_step")
    _expect(RuntimeError, lambda: move_gripper(d2, d2.q_start(), _open_targets(), 0.5,
                                               on_step=_boom_once),
            "a throw on ONE step of the loop was caught and the loop carried on")
    assert calls["k"] == 5, f"the loop ran on past the throw: {calls['k']} on_step calls"

    d3 = _Demo()
    d3.robot = SimpleNamespace(get_joint_positions=lambda: (_ for _ in ()).throw(
        RuntimeError("readback")))
    _expect(RuntimeError, lambda: move_gripper(d3, d3.q_start(), _open_targets(), 0.5),
            "the exit readback was wrapped: a target name missing from `idx` raises there")


def _ast_of(rel):
    return ast.parse(open(os.path.join(ROOT, rel), encoding="utf-8").read())


def _func(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def _repo_files(*patterns):
    out = subprocess.run(["git", "ls-files", *patterns], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, f"git ls-files {patterns} failed: {out.stderr}"
    return out.stdout.split()


def _morph_files():
    rels = _repo_files("morph/*.py", "morph/**/*.py")
    assert len(rels) >= 40, (
        f"git ls-files gave {len(rels)} files under morph/: every rule below would pass over an "
        f"empty universe")
    return rels


def _shipped_files():
    """Every source that ships on the robot: the packages, the entry points, the gate and the
    measurement tools. The bench and most of `tools/` exist only in the development tree, so they
    are checked WHEN PRESENT -- `morph/` is the universe the rules below have to bind on, and it
    is the same tree either way."""
    rels = (_morph_files() + ["play_isaac.py", "scene.py", "arm_plan.py", "verify_place.py"]
            + [r for r in ("grip_bench.py",) if os.path.exists(os.path.join(ROOT, r))]
            + sorted(_repo_files("tools/*.py")))
    return rels


def test_a_missing_close_asset_refuses_instead_of_curling():
    """The curl fallback closed the fingers end to end and gripped nothing -- j1 0.13, pads 0 N,
    SLIPPED (docs/FINDINGS.md PART F/15). Structure, not substring: the refusal must be the whole
    `else`, unconditional, tagged before it exits, and it must not reach the replay."""
    tree = _ast_of("morph/pick/close_anim.py")
    gates = [n for n in ast.walk(tree)
             if isinstance(n, ast.If) and ast.unparse(n.test) == "self.close_traj is not None"]
    assert len(gates) == 1, f"{len(gates)} close-asset gates in the stage, not one"
    body = gates[0].orelse
    assert body, "the gate has no `else`: a missing asset falls through into the seat"
    nested = [ast.unparse(n).split("\n")[0] for n in body
              if isinstance(n, (ast.If, ast.Try, ast.While, ast.For, ast.With))]
    assert not nested, (
        f"the refusal sits inside {nested}: a branch of its own (`if False:`, a swallowed "
        f"exception) leaves the tag unreached while this row stays green")
    steps = [ast.unparse(n) for n in body]
    assert "self._fallback('close-no-asset')" in steps, (
        f"the refusal is untagged ({steps}): the run has no record of why the attempt ended")
    assert "alive = False" in steps, (
        f"the refusal does not end the attempt ({steps}): `_pick_attempt` reads `alive` right "
        f"after this stage, and without it the pick seats, captures and lifts an unclosed hand")
    assert steps.index("self._fallback('close-no-asset')") < steps.index("alive = False"), (
        f"the exit precedes its tag ({steps})")
    assert "_close_replay" not in ast.unparse(ast.Module(body=body, type_ignores=[])), (
        "the refusal reaches the replay it exists because there is none of")


def test_no_finger_target_but_the_reference_open_is_commanded():
    """The curl close and `_finger_curl_targets` are deleted -- `close_gripper` is the force
    close, which drives no target dict; what binds is that nothing can
    command a curl again -- neither through the primitive's signature, nor by building the
    targets somewhere else in the package."""
    fn = _func(_ast_of("morph/gripper/api.py"), "move_gripper")
    a = fn.args
    names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    assert names == ["demo", "q_base", "targets", "secs", "stagger", "on_step", "release"], (
        f"move_gripper takes {names}: `stop_obj`/`delays` were the close's, and a re-added one "
        f"is the contact stop and the squeeze coming back with it")
    assert a.vararg is None and a.kwarg is None, (
        "move_gripper grew *args/**kwargs, which swallows `stop_obj=` at every call site and "
        "makes the signature above unfalsifiable")
    calls, unparsed = [], []
    for rel in _morph_files():
        try:
            tree = _ast_of(rel)
        except SyntaxError:
            unparsed.append(rel)
            continue
        calls += [(rel, n) for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "move_gripper"]
    assert unparsed == [], (
        f"{unparsed} does not parse on the 3.10 venv, so this sweep skips it in silence: every "
        f"shipped file must parse here")
    assert calls, "no move_gripper call under morph/: the rule below cannot fail"
    for rel, n in calls:
        got = ast.unparse(n.args[2]) if len(n.args) > 2 else ast.unparse(n)
        assert got == "dict(GRIPPER_OPEN)", (
            f"{rel}:{n.lineno} animates the fingers to {got}: the reference open is the only "
            f"finger target this package may command")
    for rel in [r for r in _morph_files() if r.startswith("morph/gripper/")]:
        tree = _ast_of(rel)
        # The one open pose is a named constant, not a profile: exempt it and nothing else.
        opens = [n.value for n in tree.body
                 if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "GRIPPER_OPEN"]
        for n in ast.walk(tree):
            if n in opens:
                continue
            if isinstance(n, ast.Dict):
                keys = [k for k in n.keys if k is not None]
            elif isinstance(n, ast.DictComp):
                keys = [n.key]
            elif isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store):
                keys = [n.slice]
            else:
                continue
            for k in keys:
                src = ast.unparse(k)
                assert not ("finger_" in src and "joint_" in src), (
                    f"{rel}:{n.lineno} builds finger joint targets keyed on {src}: that is the "
                    f"curl profile back under another name, whatever the function is called")


def test_the_refusal_tag_is_registered_as_an_outcome():
    sys.path.insert(0, ROOT)
    import verify_place
    assert verify_place.FALLBACK_TAGS.get("close-no-asset") == (
        "OUTCOME", "no recorded close trajectory: the close has no other close to run, so it "
                   "refuses and the attempt ends"), (
        f"close-no-asset reads {verify_place.FALLBACK_TAGS.get('close-no-asset')} in the "
        f"registry: an unregistered tag grades SAFETY and fails every run that refuses cleanly")


def test_the_open_takes_its_release_from_the_drive_state():
    """`release` collapses the stagger, and the open reads it off the drives rather than taking
    it from the caller: on the kinematic path the stagger keeps a loaded finger off the object."""
    on, off = _Demo(), _Demo()
    off.drives = False
    open_gripper(on, on.q_start(), 0.5)
    open_gripper(off, off.q_start(), 0.5)
    for n, tgt in GRIPPER_OPEN.items():
        assert abs(_series(on, n)[-1] - float(tgt)) < 1e-9, (
            f"{n} is commanded to {_series(on, n)[-1]}, not the open {tgt}: the targets are not "
            f"`GRIPPER_OPEN`")
    fl = [n for n in GRIPPER_OPEN if n.endswith("joint_1_1")]
    assert len({_first_move(on, n) for n in fl}) == 1, (
        "the fingers start apart with the drives on, where `release` should have zeroed the "
        "stagger: the open is not reading `demo._drive_on()`")
    assert len({_first_move(off, n) for n in fl}) > 1, (
        "the stagger is gone with the drives OFF too, so the open passes a literal, not the "
        "drive state -- and nothing here could see `release=True` hard-coded")


def test_gripper_open_pins_all_eleven_values_against_their_source():
    """Two sources, and the split is physical. j2, j3 and the two palm spreads are the recorded
    open geometry and must track the recording. j1 is the SIDE-grip angle: the recording is a
    TOP-DOWN grasp, thumb splayed to -1.040 and b/c half shut at -0.493, which is not an open at
    all for a side grip -- so all three j1 are the one symmetric -0.55 the release opens to."""
    from tests.fingers_recording import FINGERS_RECORDING as asset
    assert set(GRIPPER_OPEN) == set(asset), (
        f"GRIPPER_OPEN names {sorted(set(GRIPPER_OPEN) ^ set(asset))} that the asset does not, or "
        f"the other way: the open must command every recorded finger DOF and no invented one")
    assert len(GRIPPER_OPEN) == 11, f"GRIPPER_OPEN has {len(GRIPPER_OPEN)} joints, not 11"
    # Every value pinned, not just the three j1: this is a RECORDING of one capture, and a drift
    # in it silently moves the millimetres tests/test_armfk.py measures off it.
    assert asset == {
        "finger_a_joint_1_1": -1.04, "finger_b_joint_1_1": -0.493, "finger_c_joint_1_1": -0.493,
        "finger_a_joint_2_1": 0.0, "finger_b_joint_2_1": 0.0, "finger_c_joint_2_1": 0.0,
        "finger_a_joint_3_1": -0.0476, "finger_b_joint_3_1": -0.0476,
        "finger_c_joint_3_1": -0.0476,
        "palm_finger_b_joint_1": 0.161, "palm_finger_c_joint_1": 0.161}, (
        f"the recording is {asset}: it is what the capture measured, never a command, so no "
        f"value in it may be retuned -- change the open, which is GRIPPER_OPEN")
    for f in FINGERS:
        assert GRIPPER_OPEN[J1 % f] == -0.55, (
            f"{J1 % f} opens to {GRIPPER_OPEN[J1 % f]}, not -0.55: the release open is one "
            f"symmetric angle on a, b and c")
        assert float(asset[J1 % f]) != -0.55, (
            f"the recording now holds {J1 % f} at -0.55, so the row above would pass on the "
            f"recording alone and could no longer catch j1 falling back to the top-down splay")
    for n, v in GRIPPER_OPEN.items():
        if n.endswith("joint_1_1"):
            continue
        assert v == float(asset[n]), (
            f"{n} opens to {v}; the recording holds {asset[n]}. j2, j3 and the palm spreads are "
            f"the recorded geometry and carry no side-grip choice")


def _mutates_gripper_open(n):
    """A write THROUGH the name: `GRIPPER_OPEN[k] = v`, `+=`, or a mutating method call. The dict
    is a plain module constant, so a second open value can be installed at runtime without any of
    the text rules noticing."""
    if isinstance(n, (ast.Assign, ast.AugAssign)):
        tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
        return any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                   and t.value.id == "GRIPPER_OPEN" for t in tgts)
    return (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "GRIPPER_OPEN"
            and n.func.attr in ("update", "pop", "popitem", "setdefault", "clear"))


def test_no_shipped_source_carries_a_second_open_value():
    """One open, one definition. The asset key is retired to `tests/fingers_recording.py`, so no
    shipped source may name it, the five env flags that used to rewrite it are gone, and nothing
    writes through the constant at runtime."""
    asset = json.load(open(os.path.join(ROOT, "usd", "_grasp_known.json"), encoding="utf-8"))
    assert "fingers_open" not in asset, (
        "usd/_grasp_known.json carries `fingers_open` again: the recording lives in "
        "tests/fingers_recording.py, and an asset key nothing ships reads invites a second open")
    shipped = _shipped_files()
    for rel in shipped:
        src = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        assert '"fingers_open"' not in src and "'fingers_open'" not in src, (
            f"{rel} reads `fingers_open`: the key is retired from usd/_grasp_known.json -- it was "
            f"a recording, and the open the robot commands is GRIPPER_OPEN")
        # The recording moved to the suites, so the route back onto the robot is an import.
        for dead in ("FINGERS_RECORDING", "fingers_recording", "from tests", "import tests"):
            assert dead not in src, (
                f"{rel} names `{dead}`: the recording is test-side geometry, and a shipped source "
                f"reaching into tests/ is the retired open value coming back by another door")
        for flag in ("FINGERS_OPEN_SYM", "GRIPPER_OPEN_POS", "J1_WIDEST", "DESCENT_WIDE",
                     "SIDE_GRIP_SYMMETRIC"):
            assert flag not in src, (
                f"{rel} reads {flag}: each of these existed only to rewrite the open to some "
                f"other value, and every one of their readers is deleted")
        if rel == "morph/gripper/api.py":
            continue
        for n in ast.walk(ast.parse(src)):
            # Constants, not text: a comment about the four-bar's a1 0.48-0.55 is not an angle.
            hit = (isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub)
                   and isinstance(n.operand, ast.Constant) and n.operand.value == 0.55) or (
                   isinstance(n, ast.Constant) and n.value in (-0.55, "-0.55"))
            assert not hit, (
                f"{rel}:{n.lineno} writes the open angle -0.55: GRIPPER_OPEN in "
                f"morph/gripper/api.py is the one definition")
            assert not _mutates_gripper_open(n), (
                f"{rel}:{n.lineno} writes through GRIPPER_OPEN: the dict is mutable, so this "
                f"installs a second open value that every text rule above would miss")


def _fn_of(rel, name):
    return _func(_ast_of(rel), name)


def test_close_gripper_is_a_thin_wrapper_over_the_force_close():
    """The API name for closing may not grow a second close: it delegates to the servo mixed into
    `Demo` and does nothing else. Mutation caught: the servo body copied in here, or a step loop
    or an `_apply` of its own added beside the call."""
    fn = _fn_of("morph/gripper/api.py", "close_gripper")
    a = fn.args
    assert [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs] == ["demo", "st"], (
        f"close_gripper takes {[x.arg for x in a.args]}: the wrapper passes the close's state "
        f"through and adds no knob of its own")
    body = [n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    assert len(body) == 1 and isinstance(body[0], ast.Return), (
        f"close_gripper's body is {[ast.unparse(n) for n in body]}: one return, or the servo has "
        f"a second implementation here")
    assert ast.unparse(body[0]) == "return demo._close_force_balance(st)", (
        f"close_gripper runs `{ast.unparse(body[0])}`: it must call the force close and nothing else")


def test_the_close_runs_through_the_api_and_not_through_the_stage_method():
    """One way to close. `_close_replay` calls `close_gripper`; the only `_close_force_balance`
    call in the tree is the wrapper's, so there is no second entry into the servo."""
    close = _ast_of("morph/close/__init__.py")
    calls = [ast.unparse(n) for n in ast.walk(close) if isinstance(n, ast.Call)]
    assert "close_gripper(self, st)" in calls, "_close_replay does not close through the API"
    direct = []
    for rel in _shipped_files():
        for n in ast.walk(_ast_of(rel)):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "_close_force_balance"):
                direct.append(f"{rel}:{n.lineno}")
    assert direct == ["morph/gripper/api.py:%d" % _fn_of("morph/gripper/api.py",
                                                         "close_gripper").body[-1].lineno], (
        f"the servo is entered directly at {direct}: `close_gripper` is the one caller")


def test_the_force_close_moved_out_of_morph_close_completely():
    """A migration is not done until the old path is deleted. `morph/close/force_balance.py` is
    gone, `morph/gripper/force_close.py` ships, and no shipped source names the old module."""
    tracked = _repo_files("*.py")
    assert "morph/gripper/force_close.py" in tracked, "the moved module is not tracked"
    assert "morph/close/force_balance.py" not in tracked, "the old module is still tracked"
    left = [rel for rel in tracked
            if not rel.startswith("tests/")
            and any(t in open(os.path.join(ROOT, rel), encoding="utf-8").read()
                    for t in ("close.force_balance", "close/force_balance"))]
    assert left == [], f"{left} still reaches the old module: the path must be gone, not aliased"


def test_the_force_close_numbers_are_the_ones_that_shipped():
    """The move changes no number inside the servo. Every default the close servos on, pinned
    against the values at the move."""
    src = open(os.path.join(ROOT, "morph/gripper/force_close.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    got = {}
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and ast.unparse(n.func) == "os.environ.get"
                and len(n.args) == 2 and all(isinstance(a, ast.Constant) for a in n.args)):
            got[n.args[0].value] = n.args[1].value
    want = {
        "CLOSE_DRIVE": "0", "CLOSE_WATCH": "1", "FB_APPLY_AUDIT": "1", "FB_CMD_AUDIT": "1",
        "FB_DIAG": "0", "FB_EFFORT": "1", "FB_GAP_SYNC": "0", "FB_HI": "3.0",
        "FB_J1_MAX": "0.595", "FB_JUMP": "1", "FB_LATCH_ON": "pad", "FB_LEVEL": "0",
        "FB_OBJ_SETTLE": "1", "FB_OVERDRIVE": "0.06", "FB_RATE_MIN": "0.10", "FB_SETTLE": "1",
        "FB_STEP": "0.0006", "FB_SYNC_ON": "wrap", "FB_T": "8.0", "FB_TARGET": "1.0",
        "FB_TRUEFORCE": "0", "FB_TTC": "0", "FB_ZHOLD": "1", "HOLD_WHEELS": "1",
        "OBJ_DENSITY": "800.0", "PALM_BACK_STEP": "0.004", "PALM_GATE": "1",
        "PRELOAD_T": "0.8", "WRAP_CURL": "1.0",
    }
    assert got == want, f"the servo's defaults changed: {sorted(set(got.items()) ^ set(want.items()))}"
    assert len(got) == 29, f"{len(got)} environment reads in the servo, not the 29 that moved"
    # The pair above is blind to arithmetic AROUND the read -- `2.0 * float(os.environ.get(...))`
    # keeps it -- so every assignment that consumes one is pinned whole.
    stmts = sorted(ast.unparse(n) for n in ast.walk(tree)
                   if isinstance(n, (ast.Assign, ast.AnnAssign))
                   and "os.environ.get" in ast.unparse(n))
    assert stmts == [
        "J1_CURL_MAX = float(os.environ.get('FB_J1_MAX', '0.595'))",
        "_CLOSE_KIN = os.environ.get('CLOSE_DRIVE', '0') != '1'",
        "_gate_g = pad_gaps.get(f) if os.environ.get('FB_LATCH_ON', 'pad') == 'pad' else gaps.get(f)",
        "_mass = float(os.environ.get('OBJ_DENSITY', '800.0')) * math.pi * KNOWN['object_radius'] ** 2 * (2.0 * self.obj_half_h)",
        "_od = float(os.environ.get('FB_OVERDRIVE', '0.06'))",
        "_sync_on = os.environ.get('FB_SYNC_ON', 'wrap')",
        "_trueforce = os.environ.get('FB_TRUEFORCE', '0') == '1'",
        "_wc = float(os.environ.get('WRAP_CURL', '1.0'))",
        "d_j1 = float(os.environ.get('FB_STEP', '0.0006'))",
        "f_tgt = float(os.environ.get('FB_TARGET', '1.0'))",
        "gap_sync = os.environ.get('FB_GAP_SYNC', '0') == '1'",
        "hi_mult = float(os.environ.get('FB_HI', '3.0'))",
        "n_max = int(float(os.environ.get('FB_T', '8.0')) / self.dt)",
        "rate_floor = float(os.environ.get('FB_RATE_MIN', '0.10'))",
        "ttc_sync = os.environ.get('FB_TTC', '0') == '1'",
        "use_eff = os.environ.get('FB_EFFORT', '1') == '1'",
    ], f"an assignment around a servo read changed: {stmts}"
    assert "ema_a = 0.15" in src, "the force EMA is not 0.15 any more"
    assert '_CLOSE_KIN = os.environ.get("CLOSE_DRIVE", "0") != "1"' in src, (
        "CLOSE_DRIVE's module-level `_CLOSE_KIN` did not move byte-identical")




def test_hold_gripper_forwards_every_argument_and_the_step_count_is_secs_over_dt():
    """The wrapper names the intent; `_hold` stays the one path. Driven on the combination the
    shipped capture takes under CAPTURE_DRIVE=1 -- `kin=False` WITH a callback, which a wrapper
    that forwarded `kin` as a literal or gated `on_step` on it would still pass at kin=True -- and
    at a dt that is not the default, so a hard-coded 240 cannot stand in for `demo.dt`."""
    seen = {}
    demo = SimpleNamespace(dt=1.0 / 180.0)
    demo._hold = lambda cmd, steps, kin=False, on_step=None: seen.update(
        cmd=cmd, steps=steps, kin=kin, on_step=on_step)
    cb = lambda: None
    _api.hold_gripper(demo, "GRIP", 0.4, kin=False, on_step=cb)
    assert seen["cmd"] == "GRIP", f"the command vector is not forwarded: {seen['cmd']}"
    assert seen["steps"] == int(0.4 / demo.dt) == 72, (
        f"steps is {seen['steps']}, not int(secs/dt)=72: the hold runs for the wrong duration, or "
        f"the step rate is written here instead of read off `demo.dt` (a literal 240 gives 96)")
    assert seen["kin"] is False, (
        "kin arrives True where the caller passed False: on drives a kinematic arm fights the "
        "physically held root")
    assert seen["on_step"] is cb, "on_step is not forwarded: the bystander pin never runs"
    # kin=True as well, or a wrapper that dropped `kin` would ride `_hold`'s own False default
    _api.hold_gripper(demo, "GRIP", 0.4, kin=True)
    assert seen["kin"] is True and seen["on_step"] is None, (
        f"the second hold forwards kin={seen['kin']}, on_step={seen['on_step']}: `kin` is dropped "
        f"and takes `_hold`'s default, or `on_step` is not left unset")


def test_hold_gripper_is_a_wrapper_and_owns_no_step_loop():
    """A copied `_hold` body beside the original is the fork this phase exists to avoid."""
    fn = _func(_ast_of("morph/gripper/api.py"), "hold_gripper")
    loops = [n for n in ast.walk(fn) if isinstance(n, (ast.For, ast.While))]
    assert not loops, "hold_gripper carries its own step loop: `_hold`'s body has been copied"
    called = {n.func.attr for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert called == {"_hold"}, (
        f"hold_gripper calls {sorted(called)}: it must call `_hold` and nothing else -- an "
        f"`_apply`/`_force` of its own is the copied body coming back")


def test_the_capture_holds_go_through_hold_gripper_and_never_through_hold():
    """Both capture holds are the wrapper's, with the durations the shipped cycle runs."""
    fn = _func(_ast_of("morph/pick/grasp.py"), "_pick_capture")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    direct = [n for n in calls
              if isinstance(n.func, ast.Attribute) and n.func.attr == "_hold"]
    assert not direct, "_pick_capture still calls self._hold directly: two ways to hold"
    holds = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "hold_gripper"]
    assert len(holds) == 2, f"_pick_capture makes {len(holds)} hold_gripper calls, not 2"
    secs = [n.args[2].value for n in holds]
    assert secs == [0.3, 0.4], f"the hold durations are {secs}, not the shipped [0.3, 0.4]"
    kw = [sorted(k.arg for k in n.keywords) for n in holds]
    assert kw == [["kin", "on_step"], ["kin"]], (
        f"the keyword sets are {kw}: the first hold pins the bystanders, the second does not")


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
