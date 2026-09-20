"""The place insert's hand-top guard.

`_insert_sequence` measured the hand's top elevation every step, maxed it, printed it, and let the
run continue: the arm could drive its hand into the shelf board above and only `verify_place.py`'s
arm-clear gate ever said so, AFTER the fact. These tests pin the guard that turns that measurement
into a refusal: the pose about to be commanded is evaluated against the underside of the board above
the slot, and a breaching pose is never commanded at all.

Offline: no Isaac, no simulator. Run: ./.venv/bin/python3 tests/test_insert_guard.py
"""
import ast
import contextlib
import io
import math
import os
import sys
import types
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# The slide branches these select are not what is under test; the guard is.
os.environ["SLIDE_TH_TRACK"] = "0"
os.environ["TRIM_BEFORE_LOWER"] = "0"

from morph import config                                    # noqa: E402
from morph.arm.model import ArmModel  # noqa: E402
from morph.arm.collision import sweep_top


def _insert_module():
    """`morph/place/insert.py` loaded WITHOUT its package `__init__`, which pulls in Isaac.

    Compiled from the source TEXT, never through the loader: the bytecode cache is keyed on
    (mtime seconds, size), so an edit that keeps the size and lands in the same second -- a moved
    line, which is exactly what mutation-testing this file does -- silently re-runs the old code.
    """
    path = os.path.join(ROOT, "morph", "place", "insert.py")
    mod = types.ModuleType("_insert_under_test")
    mod.__file__ = path
    exec(compile(open(path).read(), path, "exec"), mod.__dict__)
    return mod


INSERT = _insert_module()


def _stub_column_ramp(demo, delta, frame, *, secs=None, tag="linear", **kw):
    """`move_linear` for fixtures whose subject is the SLIDE guard, not the column ramps.

    The ramps are exercised for real in tests/test_arm_api.py (behaviour) and
    tests/test_arm_linear.py (the premise and the bit-identity). Satisfying the primitive's whole
    delegate surface here would make these guard tests depend on machinery they were not written
    to measure. The stub moves the columns exactly as the ramp would and reports arrival."""
    from morph.arm.api import ArmMove, Outcome

    q = np.asarray(demo.robot.get_joint_positions(), float).copy()
    ih1 = demo.idx["ColumnLeftBearingJoint_1"]
    ih2 = demo.idx["ColumnRightBearingJoint_1"]
    q[ih1] += float(delta[2])
    q[ih2] += float(delta[2])
    if hasattr(demo, "q"):
        demo.q = q
    return ArmMove(Outcome.ARRIVED, "arrived", (), q)


INSERT.move_linear = _stub_column_ramp

DT = 0.01
N = max(2, int(2.0 / DT))          # the slide's own step count, from `_insert_sequence`
HALF_H = 0.09                      # object half-height: goal centre = slot surface + this
CLEAR = 0.03                       # the slide clearance `place` passes in
LOW_ROW, HIGH_ROW = 0, 7           # SHELF_SLOTS rows on the 0.247 and the 1.190 surface
LZ = 1.05                          # stub d(grip z)/d(dh): the closure's height-vs-tilt coupling
HAND_OFF = np.array([0.0, 0.0, 0.05])   # hand-frame origin above the grip point, in the HAND frame


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], float)


def _fake_hand(z0, h1, dh, tilt0, tilt_gain, dh0):
    """(position, rotation) of the stub's hand link. The grip rides the closure invariant
    `h1 + lut_z(dh)`, which the const-z slide holds fixed; the hand ORIGIN sits a fixed offset
    above it in the HAND frame, so its world height -- and the box's reach above it -- move with
    the boom attitude alone. That is the coupling `sweep_top` exists for."""
    R = _rot_x(tilt0 + tilt_gain * (dh - dh0))
    return np.array([0.0, 0.0, z0 + h1 + LZ * dh + float((R @ HAND_OFF)[2])]), R


class _FakeArm:
    """The `ArmModel` surface a predictive check touches: `Q8`, `hand_link`, `fk`. Geometry is the
    stub's, not the robot's -- what matters is that a pose can be evaluated WITHOUT commanding it."""

    Q8 = ArmModel.Q8
    hand_link = "Gripper_Link3_1"

    def __init__(self, demo):
        self.demo = demo

    def fk(self, q8):
        q8 = np.asarray(q8, float)
        h1, h2 = float(q8[1]), float(q8[2])
        d = self.demo
        return {self.hand_link: _fake_hand(d.z0, h1, h2 - h1, d.tilt0, d.tilt_gain, d.dh0)}


class _GuardDemo(INSERT.InsertStage):
    """The `Demo` surface `_insert_sequence` touches on its no-drive, turret-track-off path."""

    @contextlib.contextmanager
    def _time_stage(self, stage_name, file_line=None):
        yield

    dt = DT

    def __init__(self, row, h1, dh0, z0, tilt0=0.0, tilt_gain=0.0, sol=None, bias=0.0):
        self.slot = np.asarray(config.SHELF_SLOTS[row], float)
        self.goal = np.array([self.slot[0], self.slot[1], self.slot[2] + HALF_H])
        self.z0, self.dh0, self.tilt0, self.tilt_gain = z0, dh0, tilt0, tilt_gain
        self.sol, self.bias = sol, bias
        self._safety_abort = False
        self.held = []
        self.arm = _FakeArm(self)
        self.idx = {n: i for i, n in enumerate(ArmModel.Q8)}
        self.clut = {"mount": (0.0, 0.0), "grip": None}
        q = np.zeros(len(self.idx))
        q[1], q[2], q[3] = h1, h1 + dh0, 0.20
        self.q = q
        self.robot = SimpleNamespace(get_joint_positions=lambda: self.q.copy())
        # The slide's own arithmetic puts the object exactly at the slide height, so the raise is
        # a no-op and `off` is the object itself: the forward reach it must solve is `-bx`.
        _slide_z = float(self.goal[2]) + CLEAR
        self.obj = SimpleNamespace(get_world_poses=lambda: (
            np.array([[self.goal[0], self.goal[1], _slide_z]]),
            np.array([[1.0, 0.0, 0.0, 0.0]])))

    # -- the surface `_insert_sequence` calls ------------------------------------------------
    def hold(self, q):
        self.q = np.asarray(q, float).copy()
        self.held.append(self.q.copy())

    def _hand_frame(self):
        p, R = self.arm.fk(np.array([self.q[i] for i in range(8)]))[self.arm.hand_link]
        return p + np.array([0.0, 0.0, self.bias]), R

    def _arm_model(self):
        return self.arm

    def _arm_base_world(self):
        return np.zeros(3), np.eye(3)

    def base_ledger(self):
        return -0.30, 0.0, 0.0            # 0.30 m of forward reach for the slide to solve

    def _grip_fk(self, h1, h2, a1, th):
        return np.zeros(3)

    def _lut_interp(self, table, dh, a1, n):
        return [0.50, 0.0, 0.0]           # 0.50 m of reach available: no reach-driven tilt

    def _lut_z(self, dh, a1):
        return LZ * dh

    def _bisect_a1(self, pred, lo, hi, iters=40):
        return hi

    def _closure_passives(self, dh, a1):
        return {}

    def _solve_reach_z(self, *a, **k):
        return self.sol

    def _drive_on(self):
        return False

    def _body_clear(self, tag):
        pass

    def _arm_insert(self, *a, **k):
        return self.q.copy(), 0.0


def _run(demo):
    """Run the REAL `_insert_sequence` over the stub. Returns (result, printed text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        got = demo._insert_sequence(demo.obj, demo.goal, CLEAR, demo.hold)
    return got, buf.getvalue()


def _top(demo, h1, dh):
    """The hand top the stub's own geometry gives at (h1, dh) -- what the guard must predict."""
    p, R = _fake_hand(demo.z0, h1, dh, demo.tilt0, demo.tilt_gain, demo.dh0)
    return float(p[2]) + sweep_top(R)


def _bound(row):
    return config.board_above(config.SHELF_SLOTS[row]) - config.ARM_CLEAR_MARGIN


def _untilted(z0):
    """A low-slot insert: h1 + dh is far under the column stop, so the slide never tilts."""
    return _GuardDemo(LOW_ROW, h1=0.30, dh0=0.045, z0=z0)


# `_h1_f + dh0` must clear COLUMN_MAX - 0.01 for the slide to declare the headroom exhausted.
TILT_H1, TILT_DH0, TILT_DH_F = 1.35, 0.045, 0.345
TILT_SOL = (TILT_H1, TILT_H1 + TILT_DH_F, 0.50, 0.0, 0.0)


def _tilted(z0):
    """A top-slot insert: the frozen-dh solve runs out of column, so the slide falls through to
    the tilt, and the boom attitude -- and with it the hand's top -- moves through the slide."""
    return _GuardDemo(HIGH_ROW, h1=TILT_H1, dh0=TILT_DH0, z0=z0,
                      tilt0=0.0, tilt_gain=4.0, sol=TILT_SOL)


def _tilt_start_at(clearance):
    """A tilt fixture whose ramp STARTS `clearance` under the bound. Solved from the bound rather
    than written down, so it stays a fixture about the attitude ramp and not about my arithmetic."""
    z0 = _bound(HIGH_ROW) - clearance - _top(_tilted(0.0), TILT_H1, TILT_DH0)
    return _tilted(z0)


def _tilt_profile(demo):
    """The hand top along the whole dh ramp: h1 falls as the closure's `lut_z` rises, so the GRIP
    height never moves and only the attitude does."""
    return [_top(demo, TILT_H1 - LZ * (dh - TILT_DH0), dh)
            for dh in np.linspace(TILT_DH0, TILT_DH_F, 400)]


def _verify_place_literal(name):
    """The list `verify_place.check_text` assigns to `name`. It is a LOCAL, so it cannot be
    imported; reading it by AST pins the grader's rack without editing the grader."""
    tree = ast.parse(open(os.path.join(ROOT, "verify_place.py")).read())
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets):
            return ast.literal_eval(n.value)
    raise AssertionError(f"verify_place.py no longer assigns {name}")


# ===== the bound ==============================================================================

def test_the_board_above_a_slot_is_derived_from_the_slot_heights_in_config():
    """Each shelf's ceiling is the NEXT shelf surface up, less the board's own thickness -- so it
    follows SHELF_SLOTS, and a rack whose shelves move takes its clearances with it."""
    surfaces = sorted({float(z) for z in config.SHELF_SLOTS[:, 2]})
    for lo, hi in zip(surfaces, surfaces[1:]):
        row = next(r for r in config.SHELF_SLOTS if abs(float(r[2]) - lo) < 1e-9)
        assert abs(config.board_above(row) - (hi - config.SHELF_BOARD_T)) < 1e-9, (
            f"the board above the {lo:.3f} shelf is not the {hi:.3f} surface less the "
            f"{config.SHELF_BOARD_T * 1000:.0f}mm board: {config.board_above(row)}")


def test_the_top_slot_has_a_board_above_it_too():
    """The one level whose ceiling is not another slot. Left as None it would be the ONLY level
    the guard cannot judge -- and it is the level the tilt fires on."""
    top = max(float(z) for z in config.SHELF_SLOTS[:, 2])
    row = next(r for r in config.SHELF_SLOTS if abs(float(r[2]) - top) < 1e-9)
    assert config.board_above(row) is not None, "the top slot has no ceiling to check against"
    assert abs(config.board_above(row)
               - (config.RACK_TOP_SURFACE - config.SHELF_BOARD_T)) < 1e-9


def test_the_bound_moves_when_the_rack_moves():
    """The derivation is what is under test: a hardcoded 1.180 passes every check above and fails
    this one."""
    was = config.SHELF_SLOTS
    try:
        config.SHELF_SLOTS = was + np.array([0.0, 0.0, 0.25])
        moved = config.board_above(config.SHELF_SLOTS[LOW_ROW])
    finally:
        config.SHELF_SLOTS = was
    base = config.board_above(config.SHELF_SLOTS[LOW_ROW])
    assert abs(moved - (base + 0.25)) < 1e-9, (
        f"lifting every shelf 250mm moved the board above the low slot from {base} to {moved}")


def test_the_guard_and_the_grader_measure_against_the_same_boards():
    """A guard that stops on one rack while `verify_place` grades against another is two racks."""
    graded = _verify_place_literal("UNDERSIDES_ABOVE")
    surfaces = sorted({float(z) for z in config.SHELF_SLOTS[:, 2]})
    derived = [config.board_above(next(r for r in config.SHELF_SLOTS
                                       if abs(float(r[2]) - z) < 1e-9)) for z in surfaces]
    assert [round(v, 6) for v in derived] == [round(v, 6) for v in graded], (
        f"the guard would stop under {derived} while the arm-clear gate grades against {graded}")


def test_a_goal_that_is_not_a_slot_has_no_board_above_it():
    """The bound is a property of the RACK. Off the rack there is no board to derive one from,
    and the guard has nothing to say -- it does not invent a ceiling."""
    assert config.board_above(np.array([0.30, 0.0, 0.35])) is None


# ===== the guard ==============================================================================

def test_the_stub_reaches_the_untilted_slide_at_all():
    """The control for every untilted case below: without it a guard that never runs and a guard
    that always fires are indistinguishable."""
    demo, out = _untilted(z0=0.05), None
    got, out = _run(demo)
    assert "UNTILTED" in out, f"the stub did not take the untilted branch: {out}"
    assert got is not None and len(demo.held) == N, (
        f"the clear slide ran {len(demo.held)} of {N} steps: {out}")


def test_a_slide_whose_hand_stays_under_the_board_is_not_stopped():
    """The true negative. Without it every test below is satisfied by an unconditional abort."""
    demo = _untilted(z0=0.05)
    assert _top(demo, 0.30, 0.045) < _bound(LOW_ROW), "this fixture is not actually clear"
    got, out = _run(demo)
    assert got is not None, f"a clear slide was aborted: {out}"
    assert demo._safety_abort is False, "a clear slide latched the safety abort"
    assert len(demo.held) == N, f"a clear slide stopped after {len(demo.held)} of {N} steps"


def test_a_slide_that_would_put_the_hand_in_the_board_is_stopped():
    demo = _untilted(z0=0.15)
    assert _top(demo, 0.30, 0.045) > _bound(LOW_ROW), "this fixture is not actually breaching"
    got, out = _run(demo)
    assert got is None, f"the breaching slide returned {got} instead of aborting: {out}"
    assert demo._safety_abort is True, f"the abort did not latch the run-scoped flag: {out}"


def test_the_refused_pose_is_never_commanded():
    """The check has to be PREDICTIVE. A reactive one reads the hand only after `hold_fn` has
    already stepped physics with the arm at that pose -- by then the hand is in the board and the
    only thing left to decide is what to print."""
    demo = _untilted(z0=0.15)
    _got, out = _run(demo)
    over = [i for i, q in enumerate(demo.held)
            if _top(demo, float(q[1]), float(q[2]) - float(q[1])) > _bound(LOW_ROW)]
    assert not over, (
        f"{len(over)} commanded pose(s) were over the {_bound(LOW_ROW):.3f} bound (first at step "
        f"{over[0] if over else -1}) -- the check ran after the motion, not before it: {out}")


def test_the_abort_says_what_it_measured_and_what_it_stopped():
    _got, out = _run(_untilted(z0=0.15))
    assert "ABORT" in out.upper(), f"the abort printed nothing an operator can find: {out}"
    assert "hand top max" not in out, (
        f"a slide refused at its FIRST step reported a hand-top maximum it never measured; the "
        f"arm-clear gate reads that number: {out}")


# ===== the tilt path ==========================================================================

def test_the_stub_reaches_the_tilt_solve_at_all():
    """The control for the tilt cases: the slide must actually run out of column headroom and
    fall through, or these tests are measuring the untilted path twice."""
    _got, out = _run(_tilted(z0=0.0))
    assert "column headroom EXHAUSTED" in out, f"the stub never exhausted the headroom: {out}"
    assert "tilt REQUIRED" in out, f"the stub never reached the tilt solve: {out}"


def test_the_tilt_path_moves_the_hand_top_without_moving_the_grip():
    """Why the tilt path needs its own coverage: the slide holds the GRIP height invariant, so
    nothing about the commanded z changes -- the hand's top rises purely because the boom
    attitude does, which is exactly what `sweep_top` measures and a z bound would miss."""
    tops = _tilt_profile(_tilted(z0=0.0))
    assert max(tops) - tops[0] > 0.005, (
        f"the fixture's hand top only moves {(max(tops) - tops[0]) * 1000:.1f}mm through the "
        f"tilt, so it cannot demonstrate an attitude-driven breach")


def test_the_tilt_path_is_guarded_too():
    """The upper slot is the case that falls through to the tilt solve, and it is the case whose
    clearance nothing checked."""
    demo = _tilt_start_at(0.004)          # starts 4mm clear; only the attitude can breach it
    b = _bound(HIGH_ROW)
    prof = _tilt_profile(demo)
    start, peak = prof[0], max(prof)
    assert start < b < peak, (
        f"the fixture must START clear ({start:.3f}) and BREACH later ({peak:.3f}) against the "
        f"{b:.3f} bound, or it does not show a mid-slide stop")
    got, out = _run(demo)
    assert got is None and demo._safety_abort is True, (
        f"the tilt path drove the hand to {peak:.3f} under a {b:.3f} bound and was not stopped: "
        f"{out}")
    assert 0 < len(demo.held) < N, (
        f"the tilt slide held {len(demo.held)} of {N} steps -- it should stop PART WAY, after the "
        f"clear steps and before the breaching one")
    over = [i for i, q in enumerate(demo.held)
            if _top(demo, float(q[1]), float(q[2]) - float(q[1])) > b]
    assert not over, f"the tilt path commanded {len(over)} pose(s) over the bound: {out}"
    assert "hand top max" in out and "up to the ABORT" in out, (
        f"a slide stopped part way still owes the arm-clear gate the maximum it DID reach: {out}")


def test_a_tilt_that_stays_clear_is_not_stopped():
    """The tilt path's true negative."""
    demo = _tilt_start_at(0.030)          # 30mm of headroom: the attitude swing cannot spend it
    peak = max(_tilt_profile(demo))
    assert peak < _bound(HIGH_ROW), "this fixture is not actually clear"
    got, out = _run(demo)
    assert got is not None and demo._safety_abort is False, (
        f"a clear tilt was aborted: {out}")


# ===== the live measurement is what anchors the prediction ====================================

def test_the_prediction_is_anchored_on_the_measured_hand_and_not_only_on_the_model():
    """The measurement is the point: the model is a model, and the run this guard protects is the
    one where the real hand sits higher than the model says. A prediction that ignores the live
    reading passes the breaching pose straight through."""
    b = _bound(LOW_ROW)
    demo = _untilted(z0=0.05)
    modelled = _top(demo, 0.30, 0.045)
    assert modelled < b, "the model must read CLEAR, or this measures nothing"
    # the real hand is 10mm past the bound
    demo.bias = (b - modelled) + 0.010
    got, out = _run(demo)
    assert got is None and demo._safety_abort is True, (
        f"the model read {modelled:.3f} against a {b:.3f} bound and the measured hand was "
        f"{modelled + demo.bias:.3f}; the slide ran anyway: {out}")


def test_a_measured_hand_that_sits_lower_than_the_model_is_not_aborted_for_it():
    """The other direction of the same anchor: a model that reads high over a hand that is
    actually clear must not stop the insert."""
    b = _bound(LOW_ROW)
    demo = _untilted(z0=0.15)
    modelled = _top(demo, 0.30, 0.045)
    assert modelled > b, "the model must read BREACHING, or this measures nothing"
    # the real hand is 10mm under the bound
    demo.bias = (b - modelled) - 0.010
    got, out = _run(demo)
    assert got is not None and demo._safety_abort is False, (
        f"the measured hand was {modelled + demo.bias:.3f} under a {b:.3f} bound and the slide "
        f"was aborted anyway: {out}")


# ===== what the prediction means, against the REAL arm =======================================

class _ModelDemo:
    """`_hand_top_model`'s dependencies, over the shipped arm model rather than a stub."""

    _hand_top_model = INSERT.InsertStage._hand_top_model

    def __init__(self, base):
        from morph.config import ARM_MODEL, CLOSURE_LUT
        self.m = ArmModel.load(ARM_MODEL, CLOSURE_LUT)
        self.base = base
        self.idx = {n: i for i, n in enumerate(ArmModel.Q8)}

    def _arm_model(self):
        return self.m

    def _arm_base_world(self):
        return self.base


def test_the_predicted_hand_top_is_the_top_of_the_box_the_collision_code_tests():
    """`origin + sweep_top(R)` is not a proxy for the hand: on the shipped model it IS the top
    face of `Gripper_Link3_1`'s collision AABB -- the same box `collides` and `first_hit` use.
    Checked through the base transform too, so a yawed chassis is not silently dropped."""
    yaw = 0.7
    c, s = math.cos(yaw), math.sin(yaw)
    for base in ((np.zeros(3), np.eye(3)),
                 (np.array([2.0, -3.0, 0.12]),
                  np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))):
        demo = _ModelDemo(base)
        for q8 in ([0.0, 0.60, 0.65, 0.25, 0.0, 0.0, 0.0, 0.0],
                   [0.1, 1.00, 1.05, 0.40, 0.2, 0.3, -0.2, 0.1],
                   [-0.2, 1.20, 1.35, 0.50, -0.3, 0.1, 0.4, -0.2]):
            got = demo._hand_top_model(np.array(q8, float))
            want = float(demo.m.link_aabbs(np.array(q8, float),
                                           base=base)[demo.m.hand_link][1][2])
            assert got is not None and abs(got - want) < 1e-9, (
                f"at q8 {q8} the guard predicts {got} while the hand box the collision code "
                f"tests tops out at {want}")


def test_the_prediction_survives_an_arm_model_it_cannot_evaluate():
    """A model that will not evaluate must not take the insert down mid-slide; it degrades to
    an unpredicted step, which the slide's own coverage line then reports."""
    demo = _ModelDemo((np.zeros(3), np.eye(3)))
    demo._arm_model = lambda: (_ for _ in ()).throw(RuntimeError("no model"))
    assert demo._hand_top_model(np.zeros(8)) is None


# ===== the caller has to be able to see the abort =============================================

def test_the_column_ramps_are_collision_checked_and_a_hit_refuses_the_insert():
    """The raise and the lower move both columns together, which is ALREADY an exact straight line
    in the hand frame -- so they never needed a Cartesian planner. What they needed is this: a
    geometric check between the two endpoints. Before it, nothing looked at the geometry there, so
    "every motion is collision-checked" was simply not true of the insert.

    A blocked ramp must REFUSE the insert -- hold the last cleared pose, keep the object held,
    latch the run-scoped abort and the call-scoped refusal -- exactly as the slide guard does.
    Falling through would command the pose the check just disproved."""
    src_txt = open(os.path.join(ROOT, "morph", "place", "insert.py"), encoding="utf-8").read()
    tree = ast.parse(src_txt)
    seq = next(n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_insert_sequence")
    ramps = [n for n in ast.walk(seq) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "move_linear"]
    assert len(ramps) >= 2, (
        f"expected the raise and the lower to go through `move_linear`, found {len(ramps)}")
    for call in ramps[:2]:
        kw = {k.arg for k in call.keywords}
        assert "secs" in kw, (
            f"the ramp at line {call.lineno} names no `secs` -- without one the primitive retimes "
            f"it under its own limits instead of the duration the insert asked for")
        assert "allow_contact_with" in kw, (
            f"the ramp at line {call.lineno} passes no grasp names, so the object it is CARRYING "
            f"is not in the obstacle set it is checked against")

    # And the refusal is wired: a None from either ramp must latch both flags and return.
    src = ast.unparse(seq)
    assert src.count("_insert_refused = True") >= 2, (
        "a blocked column ramp does not set the call-scoped refusal, so `place` cannot tell it "
        "from a solve that simply did not run -- and will hand the corridor to the servo")
    assert src.count("_safety_abort = True") >= 2, (
        "a blocked column ramp does not latch the run-scoped abort")


def place_fn():
    """The IMPLEMENTATION. `place` itself is a thin wrapper whose only job is the `finally` that
    releases the chassis pin; every guard test here is about the body it delegates to."""
    tree = ast.parse(open(os.path.join(ROOT, "morph", "place", "__init__.py")).read())
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_place_impl")



def place_entry_fn():
    """The PUBLIC `place`, which is the thin wrapper. `place_fn` above returns the implementation
    it delegates to -- every guard test in this file is about the body, not the entry point."""
    tree = ast.parse(open(os.path.join(ROOT, "morph", "place", "__init__.py")).read())
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "place")


def test_every_exit_from_place_releases_the_chassis_pin():
    """`pin_chassis(True)` MONKEY-PATCHES `self.world.step` (`morph/close/__init__.py`). Left
    installed it freezes the chassis at the place dock for the life of the process -- through every
    later `world.step`, including `_recover`'s `world.reset()`, so the recovery cannot recover.

    The pin is engaged mid-method and released at the tail and on the abort return, but `place`
    calls `_arm_insert`, `_insert_sequence`, `cartesian_retreat` and `_arm_obstacles` in between,
    and `_arm_obstacles` raises KeyError BY DESIGN on a missing prim. An exception on any of those
    unwinds past every release. `run_cycle`'s broad `except` then reports a failed cycle and leaves
    the world patched.

    A release on the paths someone remembered is not a guarantee. `finally` is."""
    fn = place_entry_fn()
    tries = [n for n in fn.body if isinstance(n, ast.Try) and n.finalbody]
    assert tries, "`place` has no `try/finally` at its top level, so an exception can escape it"
    released = [c for t in tries for st in t.finalbody for c in ast.walk(st)
                if isinstance(c, ast.Call) and getattr(c.func, "attr", None) == "pin_chassis"
                and any(isinstance(a, ast.Constant) and a.value is False for a in c.args)]
    assert released, (
        "`place`'s `finally` does not call `pin_chassis(False)` -- an exception mid-method leaves "
        "`world.step` monkey-patched and the chassis frozen for the rest of the process")



def _branch_of(fn, node, name):
    """Which SIDE of an `if` mentioning `name` `node` sits on: 'body', 'orelse', or None.

    Which side is the whole question. "Enclosed by an if that mentions `_safety_abort`" is
    satisfied just as well by code that runs BECAUSE of the abort as by code the abort skips."""
    for n in ast.walk(fn):
        if not isinstance(n, ast.If):
            continue
        if not any(getattr(x, "attr", getattr(x, "id", None)) == name for x in ast.walk(n.test)):
            continue
        for side in ("body", "orelse"):
            for stmt in getattr(n, side):
                if any(m is node for m in ast.walk(stmt)):
                    return side
    return None


def test_an_aborted_insert_is_not_handed_straight_back_to_the_servo():
    """`_insert_sequence` returning None already means "fall through to `_arm_insert`", which
    would drive the SAME corridor the guard just refused -- the exact shape `_execute_arm_path`
    documents as the reason it returns instead of falling through."""
    fn = place_fn()
    slides = [n for n in ast.walk(fn)
              if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "_arm_insert"
              and any(isinstance(a, ast.Constant) and a.value == "slide" for a in n.args)]
    assert slides, "`place` no longer has the servo slide fallback this test guards"
    for call in slides:
        assert _branch_of(fn, call, "_insert_refused") == "orelse", (
            "the servo slide fallback is not on the side of the refusal test that the refusal "
            "SKIPS, so an aborted insert still re-drives the corridor it refused")
    # The BRANCH SIDE alone is not enough: a condition reading a pre-call snapshot of the RUN-scoped
    # flag puts the servo on the right side and still runs it when `pick` latched upstream.
    snaps = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
             and isinstance(n.value, ast.Attribute) and n.value.attr == "_safety_abort"]
    assert not snaps, (
        "`place` snapshots `_safety_abort` into a local -- a did-THIS-call test over a "
        "run-scoped flag, which reads False-positive whenever the flag was latched upstream "
        "and hands the refused corridor straight to the unchecked servo")


def test_an_aborted_insert_does_not_reach_the_release():
    """Opening the fingers after a refused insert drops the object between the boards -- worse
    than the strike the guard exists to prevent. The abort has to leave `place` first."""
    fn = place_fn()
    grip_none = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                 and any(getattr(t, "attr", None) == "grip" for t in n.targets)]
    assert grip_none, "`place` no longer releases the grip; this test is stale"
    rets = [n for n in ast.walk(fn) if isinstance(n, ast.Return)
            and _branch_of(fn, n, "_safety_abort") == "body"
            and n.lineno < min(g.lineno for g in grip_none)]
    assert rets, (
        "nothing returns out of `place` on a latched safety abort before the release, so "
        "an aborted insert still opens the hand")


def test_a_latched_abort_leaves_pick_before_it_can_report_a_grasp():
    """The guard above is only load-bearing if `place` is never entered with the run-scoped flag
    ALREADY latched. `pick` is the one caller that can do that: a settle abort does not clear
    `st.alive` (`descend.py` declines to teleport and counts a degrade), so `_pick_attempt` can
    close, seat, lift and report a grasp with `_safety_abort` standing. The abort test has to
    come BEFORE the success return, or the object reaches `place` under a latched flag."""
    src = os.path.join(ROOT, "morph", "pick", "__init__.py")
    fn = next(n for n in ast.walk(ast.parse(open(src).read()))
              if isinstance(n, ast.FunctionDef) and n.name == "pick")
    aborts = [n for n in ast.walk(fn) if isinstance(n, ast.If)
              and any(getattr(x, "attr", None) == "_safety_abort" for x in ast.walk(n.test))]
    assert aborts, "`pick` no longer tests `_safety_abort` at all"
    trues = [n for n in ast.walk(fn) if isinstance(n, ast.Return)
             and isinstance(n.value, ast.Constant) and n.value.value is True]
    assert trues, "`pick` no longer reports a grasp; this test is stale"
    # EVERY success return, not the first. THE RULE: walk OUT from the return and ask at each level
    # whether a `_safety_abort` test precedes it -- but an outer guard stops counting across a loop.
    parent = {}
    for n in ast.walk(fn):
        for f in ("body", "orelse", "finalbody"):
            b = getattr(n, f, None)
            if isinstance(b, list):
                for st in b:
                    parent[st] = (n, b)

    def _guarded(ret):
        node = ret
        while node in parent:
            owner, block = parent[node]
            if any(isinstance(st, ast.If)
                   and any(getattr(x, "attr", None) == "_safety_abort" for x in ast.walk(st.test))
                   for st in block[:block.index(node)]):
                return True
            if isinstance(owner, (ast.For, ast.While, ast.AsyncFor)):
                # an outer guard cannot speak for a later iteration
                return False
            node = owner
        return False

    for ret in trues:
        assert _guarded(ret), (
            f"the `return True` at line {ret.lineno} is reachable with no `_safety_abort` test "
            f"before it on that path (an outer guard does not count across a loop) -- a settle "
            f"abort that still produced a grasp hands the object to `place` with the run-scoped "
            f"flag already latched")


def test_a_checker_that_cannot_be_built_is_a_refusal_not_a_silent_pass():
    """With the object gripped between two shelf boards there is nothing left to check with, so
    the honest answer is to refuse. The check moved into `move_linear`; the property did not.

    It must REFUSE, not raise: the caller is mid-insert holding an object, and an exception would
    surface as some other failure entirely."""
    from morph.arm.api import Outcome, move_linear

    class _Broken:
        dt = DT
        stage = None
        idx = {n: i for i, n in enumerate(ArmModel.Q8)}
        _plan_fallbacks = {}

        def _arm_obstacles(self, *a, **k):
            raise RuntimeError("stage walk failed")

        def _arm_model(self):
            raise AssertionError("the model was reached, but the obstacle set never built")

        def _arm_q8(self):
            return np.zeros(8)

        def _arm_bounds(self):
            return (np.full(8, -10.0), np.full(8, 10.0))

        def _arm_base_world(self):
            return (np.zeros(3), np.eye(3))

    d = _Broken()
    r = move_linear(d, np.array([0.0, 0.0, -0.05]), "base", secs=1.0)
    assert r.outcome is Outcome.NO_PLAN, f"an unbuildable obstacle set was not refused: {r}"
    assert "could not be built" in r.reason, r.reason


def test_the_insert_refuses_when_a_column_ramp_does():
    """A refused ramp must stop the insert dead: hold the last cleared pose, keep the object held,
    latch the run-scoped abort AND the call-scoped refusal. Falling through would command the pose
    the check just disproved."""
    from morph.arm.api import ArmMove, Outcome

    src_txt = open(os.path.join(ROOT, "morph", "place", "insert.py"), encoding="utf-8").read()
    seq = next(n for n in ast.walk(ast.parse(src_txt))
               if isinstance(n, ast.FunctionDef) and n.name == "_insert_sequence")
    guards = [n for n in ast.walk(seq) if isinstance(n, ast.If)
              and "q is None" in ast.unparse(n.test)]
    assert len(guards) >= 2, (
        f"expected the raise and the lower each to guard on a refused ramp, found {len(guards)}")
    for g in guards[:2]:
        body = ast.unparse(g)
        assert "_safety_abort" in body and "_insert_refused" in body, (
            f"the guard at line {g.lineno} does not latch both flags: {body[:120]}")
        assert any(isinstance(r, ast.Return) for r in ast.walk(g)), (
            f"the guard at line {g.lineno} does not return -- the insert would carry on")


def test_the_servo_watches_the_proximal_links_and_nothing_else():
    """`_arm_insert` drives the object with the payload held between two boards and checked
    nothing. It still must not check everything: the field ranks a per-step FULL check last, and
    against real slip it refuses working placements. It watches the boom and the columns, which
    have no business touching anything during an insert."""
    import ast as _ast, math as _math, os as _os, numpy as _np
    from morph.arm.model import ArmModel as _AM
    src = _os.path.join(ROOT, "morph", "place", "insert.py")
    tree = _ast.parse(open(src).read())
    cls = next(n for n in _ast.walk(tree) if isinstance(n, _ast.ClassDef))
    watch = next(n for n in cls.body if isinstance(n, _ast.Assign)
                 and getattr(n.targets[0], "id", "") == "SERVO_WATCH")
    names = [e.value for e in watch.value.elts]
    assert "Arm_Left_1" in names, names
    for banned in ("Gripper_Link3_1", "Hand_Bearing_1", "finger_a_link_3_1"):
        assert banned not in names, f"{banned} holds the payload; watching it re-creates the "
    fn = next(n for n in cls.body if isinstance(n, _ast.FunctionDef) and n.name == "_servo_blocked")
    glb = {"np": _np, "os": _os, "math": _math, "ArmModel": _AM}
    exec(compile(_ast.fix_missing_locations(_ast.Module(body=[fn], type_ignores=[])), src, "exec"), glb)

    seen = {}

    class _D:
        SERVO_WATCH = tuple(names)
        idx = {n: i for i, n in enumerate(_AM.Q8)}
        def _arm_model(self):
            def first_hit(q8, obs, margin, held, base, links=None):
                seen["links"] = links
                seen["held"] = held
                return "Arm_Left_1"
            return SimpleNamespace(first_hit=first_hit)

    got = glb["_servo_blocked"](_D(), _np.zeros(len(_AM.Q8)), [("box", "world")], None)
    assert got == "Arm_Left_1", got
    assert seen["links"] == tuple(names), f"the watch did not restrict its links: {seen['links']}"
    assert seen["held"] is None, "the payload must not be passed as held: it is not what is watched"


def test_a_blocked_servo_step_is_never_commanded():
    """Ordering: the watch runs BEFORE hold_fn writes the pose, and a breach latches like every
    other insert refusal rather than falling through to the next step."""
    import ast as _ast, os as _os
    src = _os.path.join(ROOT, "morph", "place", "insert.py")
    tree = _ast.parse(open(src).read())
    fn = next(n for n in _ast.walk(tree)
              if isinstance(n, _ast.FunctionDef) and n.name == "_arm_insert")
    def _calls_any(node, attr):
        return any(isinstance(c, _ast.Call) and getattr(c.func, "attr", "") == attr
                   for c in _ast.walk(node))

    loops = [n for n in _ast.walk(fn) if isinstance(n, _ast.For) and _calls_any(n, "_servo_blocked")]
    assert loops, "no loop in the servo consults the proximal watch"
    body = loops[0].body
    def _calls(node, attr):
        return any(isinstance(c, _ast.Call) and getattr(c.func, "attr", "") == attr
                   for c in _ast.walk(node))

    guard_at = [i for i, st in enumerate(body) if _calls(st, "_servo_blocked")]
    holds_at = [i for i, st in enumerate(body)
                if any(getattr(c.func, "id", "") == "hold_fn" for c in _ast.walk(st)
                       if isinstance(c, _ast.Call))]
    assert guard_at, "the servo loop never consults the proximal watch"
    assert holds_at, "fixture: the loop should write with hold_fn"
    # The COMMANDED step is the loop's LAST hold_fn: the earlier one belongs to the divergence guard,
    # and re-commanding a pose the arm already held is not the motion this watch exists to stop.
    assert min(guard_at) < max(holds_at), "the commanded pose is written before the watch decides"
    guard_if = [st for st in body if isinstance(st, _ast.If)
                and any(isinstance(r, _ast.Return) for r in _ast.walk(st))
                and any(getattr(t, "attr", "") == "_insert_refused"
                        for a in _ast.walk(st) if isinstance(a, _ast.Assign) for t in a.targets)]
    assert guard_if, "a blocked step must return, not continue to the next one"
    # The CONDITION is pinned: `if False:` keeps the refusal in the tree while never running it,
    # which is the decoy this must reject -- the same one the reach's bake guard had to reject.
    t = guard_if[0].test
    assert (isinstance(t, _ast.Compare) and isinstance(t.ops[0], _ast.IsNot)
            and isinstance(t.comparators[0], _ast.Constant)
            and t.comparators[0].value is None), (
        "the servo guard must fire on `<result> is not None`")
    latched = {getattr(t, "attr", "") for st in guard_if for a in _ast.walk(st)
               if isinstance(a, _ast.Assign) for t in a.targets}
    assert "_safety_abort" in latched and "_insert_refused" in latched, latched


def test_a_joint_that_stops_following_its_command_aborts_the_insert():
    """The signal existed and was only GRADED after the run. During an insert it is a guarded move:
    ONE step past tolerance is the transient every new motion produces -- aborting on it broke a
    healthy placement at step 2 of 240 -- while a sustained breach means every pose downstream is
    computed from a command the arm is not at."""
    import ast as _ast, os as _os
    src = _os.path.join(ROOT, "morph", "place", "insert.py")
    tree = _ast.parse(open(src).read())
    fn = next(n for n in _ast.walk(tree)
              if isinstance(n, _ast.FunctionDef) and n.name == "_arm_insert")

    def _reads_over(node):
        return any(isinstance(c, _ast.Constant) and c.value == "_track_over" for c in _ast.walk(node))

    loops = [n for n in _ast.walk(fn) if isinstance(n, _ast.For) and _reads_over(n)]
    assert loops, "the servo loop never reads the following-error signal"
    body = loops[0].body
    reads = [i for i, st in enumerate(body) if _reads_over(st)]
    holds = [i for i, st in enumerate(body)
             if any(isinstance(c, _ast.Call) and getattr(c.func, "id", "") == "hold_fn"
                    for c in _ast.walk(st))]
    guards = [i for i, st in enumerate(body) if isinstance(st, _ast.If)
              and any(getattr(t, "id", "") == "_sv_run" for t in _ast.walk(st.test))]
    assert reads and holds and guards, (reads, holds, guards)
    assert min(reads) > max(holds), "the error must be read AFTER the write it describes"
    guard = body[min(guards)]
    assert any(isinstance(r, _ast.Return) for r in _ast.walk(guard)), "a sustained breach must stop"
    latched = {getattr(t, "attr", "") for a in _ast.walk(guard)
               if isinstance(a, _ast.Assign) for t in a.targets}
    assert {"_safety_abort", "_insert_refused"} <= latched, latched
    # A single step must NOT abort: the threshold is a run length, not a boolean.
    thresh = next(n for n in tree.body if isinstance(n, _ast.Assign)
                  and getattr(n.targets[0], "id", "") == "_SERVO_FOLLOW_STEPS")
    assert int(thresh.value.value) > 1, "one transient step would abort a healthy placement"


def _insert_mixin():
    return next(v for v in vars(INSERT).values()
                if isinstance(v, type) and hasattr(v, "_arm_insert"))


def test_no_insert_leg_runs_after_the_insert_has_already_refused():
    """A refusal means a check -- the servo watch, or a checked column ramp -- rejected THIS
    corridor. Every later leg of the insert drives the same corridor, so a refusal that only
    latches a flag and lets the next `_arm_insert` run has not stopped anything. `place` chains
    slide -> lower and the sequence chains trim-high -> lower; both discarded the refusal."""
    M = _insert_mixin()
    written = []

    class _Obj:
        @staticmethod
        def get_world_poses():
            return (np.array([[0.0, 0.0, 0.0]]), None)

    class _D:
        dt = 1.0 / 240.0
        _insert_refused = True                      # an EARLIER leg already refused
        _safety_abort = True
        idx = {"ColumnLeftBearingJoint_1": 0, "ColumnRightBearingJoint_1": 1,
               "ArmLeftJoint_1": 3, "BaseJoint_1": 4}
        robot = SimpleNamespace(get_joint_positions=lambda: np.zeros(8))
        def _arm_obstacles(self, *a, **k):
            return []
        def _arm_base_world(self):
            return np.eye(4)
        def _grip_fk(self, h1, h2, a1, th):
            return np.array([a1, th, 0.5 * (h1 + h2)])
        def _servo_blocked(self, *a, **k):
            return None
        def _fallback(self, tag):
            pass

    # A target the servo would chase for the full second if it ran at all.
    out = M._arm_insert(_D(), _Obj(), np.array([0.30, 0.0, 0.20]), 1.0, "lower",
                        lambda q: written.append(np.asarray(q, float).copy()))
    assert written == [], (
        f"{len(written)} poses were commanded AFTER the insert had already refused this corridor "
        f"-- the refusal latched a flag and the next leg drove through it anyway")
    assert out is not None and math.isnan(float(out[1])), (
        "a leg that did not run must report a NaN residual, the same shape the servo refusal "
        "returns, so no caller reads it as a completed placement")


def test_the_trim_before_the_lower_cannot_be_followed_by_the_lower_it_refused():
    """`trim-high` runs `_arm_insert` and its return is DISCARDED; the lower below it is a
    `move_linear`, not an `_arm_insert`, so the guard inside `_arm_insert` cannot stop it. The
    sequence itself has to end."""
    src = open(os.path.join(ROOT, "morph", "place", "insert.py"), encoding="utf-8").read()
    seq = next(n for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.FunctionDef) and n.name == "_insert_sequence")
    def _holds(node, pred):
        return any(pred(c) for c in ast.walk(node))

    trim = next(c for c in ast.walk(seq) if isinstance(c, ast.Call)
                and getattr(c.func, "attr", None) == "_arm_insert"
                and any(isinstance(a, ast.Constant) and a.value == "trim-high" for a in c.args))
    lower = min((c.lineno for c in ast.walk(seq) if isinstance(c, ast.Call)
                 and getattr(c.func, "id", None) == "move_linear" and c.lineno > trim.lineno),
                default=None)
    assert lower is not None, "no lower follows trim-high -- retarget this test"
    # POLARITY and REACHABILITY, not merely shape: a `return` nested behind an env check that never
    # fires also mentions the flag. The test must be the bare attribute and the `return` direct.
    gates = [n for n in ast.walk(seq) if isinstance(n, ast.If)
             and trim.lineno <= n.lineno < lower
             and isinstance(n.test, ast.Attribute) and n.test.attr == "_insert_refused"
             and any(isinstance(st, ast.Return) for st in n.body)]
    assert gates, (
        "nothing between the trim-high leg and the lower ramp checks `_insert_refused` and "
        "returns -- a trim that the servo watch refused is followed by the lower, which drives "
        "the same corridor")


def test_the_servo_watch_is_every_link_that_moves_and_not_three_chosen_names():
    """DERIVED, not chosen. The watch must cover every link the servo's own four joints move,
    less the distal hand and the payload -- those are excluded on purpose (a full check against a
    friction-held object refuses working placements, docs/FINDINGS.md). The first version named
    three links and covered 3 of 8; the five it omitted travel up to 37.7mm. Cost does not justify
    a short list: 379.4us for eight against 377.8us for three, because the cost is obstacle
    iteration, not link count."""
    import json as _json
    M = _insert_mixin()
    model = _json.load(open(os.path.join(ROOT, "usd", "_arm_model.json"), encoding="utf-8"))
    m = ArmModel.load(os.path.join(ROOT, "usd", "_arm_model.json"),
                      os.path.join(ROOT, "usd", "_closure_lut.json"))
    lo, hi = m.bounds()
    q0 = 0.5 * (np.asarray(lo, float) + np.asarray(hi, float))
    ref = m.link_aabbs(q0)

    servo = ["ColumnLeftBearingJoint_1", "ColumnRightBearingJoint_1", "ArmLeftJoint_1", "BaseJoint_1"]
    moving = set()
    for j in servo:
        i = ArmModel.Q8.index(j)
        q1 = q0.copy()
        q1[i] += 0.02 * (hi[i] - lo[i])
        for k, v in m.link_aabbs(q1).items():
            if float(np.max(np.abs(v - ref[k]))) > 1e-6:
                moving.add(k)

    # The hand is deliberately out of the watch -- but the two boom members are proximal structure
    # even though the model groups them with the hand, so they stay in.
    distal = set(model["hand_group"]) - {"Arm_Left_1", "Arm_Right_1"}
    want = moving - distal
    have = set(M.SERVO_WATCH)
    assert not (want - have), (
        f"the servo moves {sorted(want - have)} and the watch does not look at them -- the watch "
        f"covers {len(have & want)} of {len(want)} links the servo actually moves")
    assert not (have & distal), (
        f"the watch includes the distal hand {sorted(have & distal)}, which is what makes a check "
        f"against a friction-held payload refuse working placements")


def test_a_servo_watch_that_cannot_be_evaluated_is_a_refusal_not_a_pass():
    """The only geometric check on this motion is the watch. An exception inside it used to return
    None -- the same value as "clear" -- so any failure silently converted the check into a pass."""
    M = _insert_mixin()

    class _D:
        idx = {n: i for i, n in enumerate(ArmModel.Q8)}
        def _arm_model(self):
            raise RuntimeError("no stage")

    out = M._servo_blocked(_D(), np.zeros(8), [], None)
    assert out is False, (
        f"a watch that could not be evaluated returned {out!r}; None is what a CLEAR step returns, "
        f"so the servo reads a failed check as a pass and drives on unchecked")


def test_a_servo_watch_that_cannot_be_built_refuses_before_the_first_step():
    """Same rule the ramp checker already follows: asked for a check and could not build it is a
    refusal. Building the obstacle set walks the whole stage and can throw; that used to leave
    `_sv_obs = None`, which the loop read as "no check was asked for"."""
    M = _insert_mixin()
    written = []

    class _Obj:
        @staticmethod
        def get_world_poses():
            return (np.array([[0.0, 0.0, 0.0]]), None)

    class _D:
        dt = 1.0 / 240.0
        _insert_refused = False
        _safety_abort = False
        idx = {"ColumnLeftBearingJoint_1": 0, "ColumnRightBearingJoint_1": 1,
               "ArmLeftJoint_1": 3, "BaseJoint_1": 4}
        robot = SimpleNamespace(get_joint_positions=lambda: np.zeros(8))
        tags = []
        def _arm_obstacles(self, *a, **k):
            raise RuntimeError("stage walk failed")
        def _arm_base_world(self):
            return np.eye(4)
        def _grip_fk(self, h1, h2, a1, th):
            return np.array([a1, th, 0.5 * (h1 + h2)])
        def _servo_blocked(self, *a, **k):
            return None
        def _fallback(self, tag):
            self.tags.append(tag)

    d = _D()
    out = M._arm_insert(d, _Obj(), np.array([0.30, 0.0, 0.20]), 1.0, "lower",
                        lambda q: written.append(q), obj_idx=0)
    assert written == [], (
        f"{len(written)} poses were commanded with NO collision watch at all -- the watch failed "
        f"to build and the servo ran anyway")
    assert d._insert_refused and d._safety_abort, (
        "an insert that ran without its only check must refuse and latch, or `place` releases the "
        "object at the end of a motion nothing looked at")
    assert d.tags, "an unbuildable watch must be counted, not silent"
    assert math.isnan(float(out[1]))


def test_the_carried_object_is_not_an_obstacle_to_the_links_carrying_it():
    """`_arm_obstacles()` keeps the carried object as a world obstacle, and `held=None` omits only
    the SWEPT payload geometry -- so the prim itself stayed in the set the proximal links are
    checked against. A boom "hitting" the object the hand is holding is not a scene collision."""
    M = _insert_mixin()
    seen = {}

    class _Stop(Exception):
        pass

    class _D:
        dt = 1.0 / 240.0
        _insert_refused = False
        idx = {"ColumnLeftBearingJoint_1": 0, "ColumnRightBearingJoint_1": 1,
               "ArmLeftJoint_1": 3, "BaseJoint_1": 4}
        robot = SimpleNamespace(get_joint_positions=lambda: np.zeros(8))
        def _arm_obstacles(self, *a, **k):
            seen.update(k)
            seen["args"] = a
            # the obstacle call is the whole subject
            raise _Stop
        def _arm_base_world(self):
            return np.eye(4)

    try:
        M._arm_insert(_D(), None, np.zeros(3), 1.0, "lower", lambda q: None, obj_idx=7)
    except (_Stop, Exception):
        pass
    ex = tuple(seen.get("exclude_names", ()))
    assert "pickup_obj_7" in ex, (
        f"the servo's obstacle set was built with exclude_names={ex!r} -- the object being carried "
        f"is still an obstacle to the links carrying it")


def test_a_blocked_servo_step_is_never_commanded_behaviourally():
    """Behaviour, not shape: drive the REAL `_arm_insert` with a watch that blocks partway and
    assert nothing was written at or after the blocking step. The shape test above pins the
    ordering; this one proves the loop actually stops, which an AST walk cannot."""
    M = _insert_mixin()
    written = []
    BLOCK_AT = 6

    class _Obj:
        @staticmethod
        def get_world_poses():
            return (np.array([[0.0, 0.0, 0.0]]), None)

    class _D:
        dt = 1.0 / 240.0
        _insert_refused = False
        _safety_abort = False
        idx = {"ColumnLeftBearingJoint_1": 0, "ColumnRightBearingJoint_1": 1,
               "ArmLeftJoint_1": 3, "BaseJoint_1": 4}
        robot = SimpleNamespace(get_joint_positions=lambda: np.zeros(8))
        tags = []
        _track_over = 0.0
        def _arm_obstacles(self, *a, **k):
            return []
        def _arm_base_world(self):
            return np.eye(4)
        def _grip_fk(self, h1, h2, a1, th):
            return np.array([a1, th, 0.5 * (h1 + h2)])
        def _servo_blocked(self, q, obs, base):
            return "Arm_Left_1" if len(written) >= BLOCK_AT else None
        def _fallback(self, tag):
            self.tags.append(tag)

    d = _D()
    out = M._arm_insert(d, _Obj(), np.array([0.30, 0.0, 0.20]), 1.0, "lower",
                        lambda q: written.append(np.asarray(q, float).copy()), obj_idx=0)
    assert len(written) == BLOCK_AT, (
        f"the watch blocked at step {BLOCK_AT} but {len(written)} poses were commanded -- the "
        f"refused pose was written anyway, or the loop carried on to the next step")
    assert d._insert_refused and d._safety_abort, (
        "a blocked servo step must latch both the call-scoped refusal and the run-scoped abort, "
        "or `place` releases the object at the end of a motion a check rejected")
    assert d.tags, "a blocked servo step must be counted, not silent"
    assert math.isnan(float(out[1])), "a refused leg must report a NaN residual, never a distance"


def test_the_following_error_abort_is_a_distance_limited_guarded_move():
    """Behaviour, not shape. TWO claims, and the pair is the point -- a single step past tolerance
    is the transient every new motion produces, and aborting on it killed a healthy placement at
    step 2 of 240, while a joint that never catches up means every pose downstream is computed from
    a command the arm is not at, with the object held between two boards.

    So: a joint that is over tolerance but RECOVERS must not abort, and one that stays over for
    `_SERVO_FOLLOW_STEPS` consecutive steps must."""
    M = _insert_mixin()
    N = INSERT._SERVO_FOLLOW_STEPS
    # PINNED, not merely read: a test taking its threshold from the module under test passes at every
    # value, including the N=1 this guard rules out. The band comes from the physics.
    assert 8 <= N <= 60, (
        f"_SERVO_FOLLOW_STEPS = {N}: at {1 / 240 * 1000:.1f}ms a step that is "
        f"{N / 240 * 1000:.0f}ms, outside the 33..250ms a distance-limited guarded move means")

    class _Obj:
        @staticmethod
        def get_world_poses():
            return (np.array([[0.0, 0.0, 0.0]]), None)

    def _run(over_for):
        written = []

        class _D:
            dt = 1.0 / 240.0
            _insert_refused = False
            _safety_abort = False
            idx = {"ColumnLeftBearingJoint_1": 0, "ColumnRightBearingJoint_1": 1,
                   "ArmLeftJoint_1": 3, "BaseJoint_1": 4}
            robot = SimpleNamespace(get_joint_positions=lambda: np.zeros(8))
            tags = []
            def _arm_obstacles(self, *a, **k):
                return []
            def _arm_base_world(self):
                return np.eye(4)
            def _grip_fk(self, h1, h2, a1, th):
                return np.array([a1, th, 0.5 * (h1 + h2)])
            def _servo_blocked(self, *a, **k):
                return None
            def _fallback(self, tag):
                self.tags.append(tag)
            @property
            def _track_over(self):
                # over tolerance for the first `over_for` steps, then recovered
                return 2.0 if len(written) < over_for else 0.0

        d = _D()
        M._arm_insert(d, _Obj(), np.array([0.30, 0.0, 0.20]), 1.0, "lower",
                      lambda q: written.append(q), obj_idx=0)
        return d, written

    d, w = _run(N - 1)
    assert "servo-following-error" not in d.tags, (
        f"a joint over tolerance for {N - 1} steps and then recovered ABORTED the insert -- this is "
        f"the transient every new motion produces, and aborting on it killed a healthy placement "
        f"at step 2 of 240")
    assert not d._insert_refused

    d, w = _run(10_000)
    assert "servo-following-error" in d.tags, (
        f"a joint past tolerance for every one of {len(w)} steps never aborted -- every pose "
        f"downstream is computed from a command the arm is not at, with the object held between "
        f"two boards")
    assert d._insert_refused and d._safety_abort, "the abort must latch, like every other refusal"
    assert len(w) == N, (
        f"the guarded move ran {len(w)} steps before aborting, not {N} -- the distance limit is "
        f"what makes this a guarded move rather than a trip on the first transient")


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
