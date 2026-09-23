"""The finger freeze: the pin `execute_path` publishes for its motion, and the kinematic hold
`morph/robot.py::_apply` writes while the drives own the arm and nothing owns the fingers.

Offline: no Isaac. Run: ./.venv/bin/python3 tests/test_finger_freeze.py

WHY. The `palm_finger` spread joints carry the friction grasp's own compliance (kp 0.23), so a
driven free-space motion leaves them 150 mrad behind their target and the hand the planner
checked is not the hand that arrives (docs/FINDINGS.md, "Driving the reach fails on FINGER
tracking"). The fix pins them kinematically for the duration of ONE motion, which is what these
tests hold in place: the pin is scoped to the motion, it nests, and it never reaches a hand the
grip owns.
"""
import ast
import math
import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _exec_method(rel, name, glb):
    """Compile ONE method out of a source file and bind it into `glb`. Same helper as
    tests/test_timing.py -- both files it reads import Isaac at module scope."""
    path = os.path.join(ROOT, *rel)
    tree = ast.parse(open(path).read())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    assert not fn.decorator_list, f"{name} grew a decorator, which this harness would drop"
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), path, "exec"),
         glb)
    return glb[name]


# ------------------------------------------------------------------ the executor's side of it
_Q8 = ["h1", "h2", "a1", "a2", "w1", "w2", "w3", "w4"]
_ARM = type("ArmModel", (), {"Q8": _Q8, "Q8_LIN": [0, 1], "Q8_ROT": [2, 3, 4, 5, 6, 7]})
# Finger DOFs live OUTSIDE q8 and `_f_idx` carries BOTH arms'. Arm two sits BETWEEN arm one's, so
# the selection is not a prefix and a mutant that slices instead of masking cannot pass.
_NAMES = ["h1", "h2", "a1", "a2", "w1", "w2", "w3", "w4",
          "finger_a_joint_1_1", "finger_a_joint_1_2", "palm_finger_b_joint_1"]
_FING = np.array([8, 9, 10])
_ARM1 = ("finger_a_joint_1_1", "palm_finger_b_joint_1")
_F1 = np.array([8, 10])                  # ...and only these two are arm one's
_HAND = np.array([0.50, 0.30, -0.25])    # the hand a motion finds standing when it starts
_F1_HAND = _HAND[[0, 2]]                 # ...the half of it the freeze may touch
_SAG = 0.001                             # ...and what one step of a soft finger drive loses


class _PinDemo:
    """The `Demo` surface `execute_path` touches, recording `_finger_pin` at every `_apply`."""

    def __init__(self):
        self.idx = {n: i for i, n in enumerate(_Q8)}
        self._finger_pin = None
        self._f_idx = _FING
        self.dt = 1.0 / 240.0
        self.q0 = np.zeros(len(_NAMES))
        self._live = np.zeros(len(_NAMES))
        self._live[self._f_idx] = _HAND
        self.pins = []                   # `_finger_pin` as each `_apply` saw it
        # ...and what the recursion left behind on its way out
        self.nested_after = "never ran"
        self._plan_fallbacks = {}
        self._safety_abort = False
        self._abort_recovered = False
        self._focus_obj = None
        self.world = SimpleNamespace(step=lambda render: None)
        self.robot = SimpleNamespace(get_joint_positions=lambda: self._live.copy())
        # Only the far waypoint is blocked: the abort taper starts at the near one and must stay
        # clear, or it stops on its first step and never reaches the recovery.
        self._model = SimpleNamespace(
            collides=lambda q8, *a, **k: bool(np.max(np.asarray(q8, float)) > 0.9))

    def _fallback(self, tag):
        self._plan_fallbacks[tag] = self._plan_fallbacks.get(tag, 0) + 1

    def _arm_model(self):
        return self._model

    def _commanded_fingers(self):
        return None

    def _planned_obstacles(self, *a, **k):
        return [], []

    def _plan_arm(self, goal, **k):
        """Only the park is routable, so the blocked checkpoint aborts and the recovery runs. The
        park route TRAVELS: a path whose endpoints coincide is retimed to no motion at all, and
        then the recursion writes nothing for this to read."""
        return [np.full(8, 0.5), np.zeros(8)] if np.allclose(goal, 0.0) else None

    def _arm_base_world(self):
        return None

    def base_ledger(self):
        return 0.0, 0.0, 0.0

    def set_base(self, *a):
        pass

    def _with_closure_forced(self, q):
        return q

    def _force(self, q, qv=None):
        pass

    def _apply(self, q):
        self.pins.append(getattr(self, "_finger_pin", None))
        # A plant that sags: the recursion must capture ITS hand, not the outer motion's.
        self._live[self._f_idx] -= _SAG


def _bind_executor():
    """The real `execute_path`, with its collaborators and its own recursion routed to the fake."""
    out = []
    glb = {"np": np, "os": os, "math": math, "MARGIN": 0.02, "HEADLESS": True, "ArmModel": _ARM,
           "retime_q8": __import__("morph.arm.cartesian", fromlist=["x"]).retime_q8,
           "q8_derate": __import__("morph.arm.cartesian", fromlist=["x"]).q8_derate,
           "print": lambda *a, **k: out.append(" ".join(str(x) for x in a)),
           "with_closure_forced": lambda demo, q: demo._with_closure_forced(q),
           "commanded_fingers": lambda demo: demo._commanded_fingers(),
           "planned_obstacles": lambda demo, *a, **k: demo._planned_obstacles(*a, **k),
           "blame": lambda *a, **k: None,
           "plan_arm": lambda demo, *a, **k: demo._plan_arm(*a, **k)}
    fn = _exec_method(("morph", "arm", "execute.py"), "execute_path", glb)

    def _recursed(demo, *a, **k):
        q = fn(demo, *a, **k)
        demo.nested_after = getattr(demo, "_finger_pin", None)
        return q

    glb["execute_path"] = _recursed
    return fn, out


_SENTINEL = np.full(len(_FING), 9.0)     # a pin already standing when the motion starts


def test_the_motion_publishes_its_own_hand_as_the_pin():
    """Without this the freeze has nothing to hold: `_apply` cannot know the hand the planner
    checked, only the one physics has sagged to."""
    run, _out = _bind_executor()
    d = _PinDemo()
    hand = d._live[d._f_idx].copy()
    run(d, [np.zeros(8), np.full(8, 0.5)], 0.05)
    assert d.pins, "the motion commanded nothing, so this test proves nothing"
    assert all(p is not None and np.allclose(p, hand) for p in d.pins), (
        f"the pin was not the motion's own hand at every command: {d.pins[:3]} vs {hand}")


def test_the_pin_is_restored_rather_than_left_standing():
    """Scoped to the MOTION, not to the stage: left standing, the next stage's `_apply` would
    kinematically hold a hand captured by a motion that has already ended."""
    run, _out = _bind_executor()
    d = _PinDemo()
    d._finger_pin = _SENTINEL.copy()
    run(d, [np.zeros(8), np.full(8, 0.5)], 0.05)
    assert d._finger_pin is not None and np.allclose(d._finger_pin, _SENTINEL), (
        f"the motion left {d._finger_pin} behind instead of the {_SENTINEL} it found")


def test_hold_fingers_false_publishes_no_pin_at_all():
    """`hold_fingers=False` says the fingers may move during the motion, so it must also UNPIN
    for the duration: inheriting an outer motion's pin would freeze the hand this caller asked to
    leave free, at a pose from a motion that is not this one."""
    run, _out = _bind_executor()
    d = _PinDemo()
    d._finger_pin = _SENTINEL.copy()
    run(d, [np.zeros(8), np.full(8, 0.5)], 0.05, hold_fingers=False)
    assert d.pins, "the motion commanded nothing, so this test proves nothing"
    assert all(p is None for p in d.pins), f"a pin stood through the motion anyway: {d.pins[:3]}"
    assert d._finger_pin is not None and np.allclose(d._finger_pin, _SENTINEL), (
        f"the motion left {d._finger_pin} behind instead of the {_SENTINEL} it found")


def _run_abort():
    """A blocked checkpoint whose replan fails: taper, then the RECURSIVE park execution."""
    run, out = _bind_executor()
    d = _PinDemo()
    d._finger_pin = _SENTINEL.copy()
    outer = d._live[d._f_idx].copy()
    q = run(d, [np.zeros(8), np.full(8, 0.5), np.ones(8)], 0.05, checkpoint_every=1)
    assert q is None and d._safety_abort, "the harness did not abort, so the recovery never ran"
    assert d._abort_recovered, f"the recovery never reached the recursive park: {out[-1:]}"
    return d, outer


def test_the_recursion_captures_its_own_hand_and_gives_the_outer_one_back():
    """The abort recovery re-enters `execute_path`, which re-reads the hand. Clearing the pin on
    the way out of the inner motion would leave the outer one unpinned for the rest of its run."""
    d, outer = _run_abort()
    assert all(p is not None for p in d.pins), "the pin lapsed during the abort recovery"
    assert np.allclose(d.pins[0], outer), f"the outer motion's pin was {d.pins[0]}, not {outer}"
    inner = d.pins[-1]
    assert not np.allclose(inner, outer), (
        "the recursion re-used the outer hand -- this harness sags, so the two must differ")
    assert d.nested_after is not None and np.allclose(d.nested_after, outer), (
        f"the recursion left {d.nested_after} behind instead of the outer motion's {outer}")
    assert np.allclose(d._finger_pin, _SENTINEL), (
        f"the outer motion left {d._finger_pin} behind instead of the {_SENTINEL} it found")


# -------------------------------------------------------------------- the robot's side of it
def _apply_demo(drive=True, finger_cmd=None, pin=_HAND, known=_ARM1):
    """A fake carrying the REAL `_apply` and `_pin_fingers`: a freeze proven on a method nothing
    calls is a freeze in a dead branch. Every write lands in ONE ordered log, because their ORDER
    is what decides whether the drives keep their targets."""
    glb = {"np": np, "math": math, "os": SimpleNamespace(environ={"ARM2_FREEZE": "0"}),
           "ArticulationAction": lambda **kw: SimpleNamespace(**{"joint_velocities": None, **kw}),
           "print": lambda *a, **k: None}

    class _ApplyDemo:
        def __init__(self):
            self._finger_cmd = None if finger_cmd is None else np.array(finger_cmd, float)
            self._finger_pin = None if pin is None else np.array(pin, float)
            self._f1_mask = None
            self._f_idx = _FING
            self.names = _NAMES
            # The hand the planner models is arm ONE's, exactly as `commanded_fingers` filters it.
            self._armm = SimpleNamespace(joints={n: {} for n in known})
            self._roll_ii = None
            self._q_lim = False
            self._gains_writer = None
            self._drv_prev = None
            self.dt = 1.0 / 240.0
            self.log = []                # ("pos"/"vel", values, indices) / ("act", action, None)
            self.robot = SimpleNamespace(
                get_joint_positions=lambda: np.zeros(len(_NAMES)),
                set_joint_positions=lambda v, joint_indices=None: self._w("pos", v, joint_indices),
                set_joint_velocities=lambda v, joint_indices=None: self._w("vel", v, joint_indices))
            self.ctrl = SimpleNamespace(
                apply_action=lambda a: self.log.append(("act", a, None)))

        def _w(self, kind, v, ii):
            self.log.append((kind, np.asarray(v, float).copy(), ii))

        @property
        def kin(self):
            return [(v, i) for k, v, i in self.log if k == "pos"]

        @property
        def applied(self):
            return [a for k, a, _ in self.log if k == "act"]

        def _drive_on(self):
            return drive

        def _arm_model(self):
            return self._armm

        def _with_closure(self, q):
            return np.asarray(q, float)

    for _m in ("_pin_fingers", "_apply"):
        setattr(_ApplyDemo, _m, _exec_method(("morph", "robot.py"), _m, glb))
    return _ApplyDemo()


def test_the_freeze_holds_the_fingers_kinematically_while_the_drives_own_the_arm():
    """The drive target alone is not enough: it is ALREADY the pinned pose and kp 0.23 cannot
    hold it, which is the whole measurement behind this change."""
    d = _apply_demo()
    d._apply(np.zeros(len(_NAMES)))
    assert len(d.kin) == 1, f"the freeze made {len(d.kin)} kinematic writes, not one"
    got, ii = d.kin[0]
    assert np.allclose(got, _F1_HAND) and np.array_equal(ii, _F1), (
        f"the freeze wrote {got} to {ii}, not the pin to arm one's finger DOFs")
    cmd = np.asarray(d.applied[-1].joint_positions, float)
    assert np.allclose(cmd[_F1], _F1_HAND), (
        f"the drive target disagrees with the kinematic hold: {cmd[_F1]} vs {_F1_HAND}")


def test_the_freeze_re_asserts_the_pin_on_every_command():
    """The drives pull the joint back off the pin between steps -- that IS the 152.9 mrad this
    change exists for -- so a freeze that writes once and stops is the same sag with a tidier
    first frame, and every other test here passes while it happens."""
    d = _apply_demo()
    for _ in range(4):
        d._apply(np.zeros(len(_NAMES)))
    assert len(d.kin) == 4, (
        f"4 commands produced {len(d.kin)} kinematic writes; the fingers sag between the ones "
        f"that are missing")


def test_the_hold_is_a_position_and_a_velocity_written_before_the_action():
    """Two things at once. A subset `set_joint_positions` re-targets EVERY drive to its measured
    position -- Isaac assigns the subset into the full DOF array and then writes the state AND the
    targets -- so `apply_action` must come after to put the real targets back. And a position
    written without its velocity leaves the solver integrating the sag it just undid; `_force`
    pairs them for the same reason."""
    d = _apply_demo()
    d._apply(np.zeros(len(_NAMES)))
    kinds = [k for k, _, _ in d.log]
    assert kinds == ["pos", "vel", "act"], (
        f"`_apply` left its writes in the order {kinds}; the hold is a position AND a velocity, "
        f"both before the action")
    (_p, ipos), (vel, ivel) = [(v, i) for k, v, i in d.log if k in ("pos", "vel")]
    assert np.array_equal(ipos, ivel), f"position went to {ipos}, velocity to {ivel}"
    assert np.allclose(vel, 0.0), f"the pinned fingers were given a velocity of {vel}"


def test_the_pin_leaves_the_velocity_feedforward_alone():
    """`_apply` reads `_drv_prev` AFTER the freeze runs and derives the whole feedforward from it.
    Clearing it there -- which looks like hygiene after a kinematic write -- zeroes v_t on every
    joint every step, and the drives then lag by v*kd/kp with nothing saying so."""
    d = _apply_demo()
    d._drv_prev = np.zeros(len(_NAMES))
    q = np.zeros(len(_NAMES))
    # under `_apply`'s 0.005 jump threshold, so it is real motion
    q[0] = 0.001
    d._apply(q)
    v = np.asarray(d.applied[-1].joint_velocities, float)
    assert np.isclose(v[0], 0.001 / d.dt), (
        f"the feedforward for a moving joint came out {v[0]:.3f}, not {0.001 / d.dt:.3f} -- the "
        f"previous target was lost between the freeze and the velocity it feeds")


def test_a_second_motion_is_held_at_its_own_hand():
    """The arm-one MASK is cacheable; the pin is not. One `Demo` flies many motions -- the
    recovery park, the next cycle's reach -- and a pin resolved once freezes every later one at
    the first motion's hand."""
    d = _apply_demo()
    d._apply(np.zeros(len(_NAMES)))
    d._finger_pin = np.array([0.11, 0.22, 0.33])
    d._apply(np.zeros(len(_NAMES)))
    got, _ii = d.kin[-1]
    assert np.allclose(got, [0.11, 0.33]), f"the second motion was held at {got}, not its own hand"


def test_a_pin_with_no_arm_one_fingers_writes_nothing():
    """An empty selection still reaches Isaac, and an empty `set_joint_positions` still re-targets
    the WHOLE articulation to its measured position -- the expensive half of the write, for no
    joint at all."""
    d = _apply_demo(known=())
    d._apply(np.zeros(len(_NAMES)))
    assert d.kin == [], f"the freeze wrote an empty selection: {d.kin}"


def test_arm_twos_fingers_are_left_to_arm2_freeze():
    """`_f_idx` carries arm TWO's finger DOFs as well, and ARM2_FREEZE already parks those -- at
    the run's park pose, not at this motion's hand. Two owners writing one DOF at two values."""
    d = _apply_demo()
    d._apply(np.zeros(len(_NAMES)))
    _got, ii = d.kin[-1]
    assert 9 not in set(np.asarray(ii).ravel().tolist()), (
        f"the freeze wrote arm two's `finger_a_joint_1_2` (index 9): {ii}")


def test_the_grip_outranks_the_freeze():
    """`_finger_cmd` means the fingers own an object. A kinematic write there re-targets every
    drive to its measured position and wipes the contact forces the friction grasp lives on."""
    d = _apply_demo(finger_cmd=(0.11, 0.22, 0.33))
    d._apply(np.zeros(len(_NAMES)))
    assert d.kin == [], f"the freeze wrote over the grip's own hand: {d.kin}"
    cmd = np.asarray(d.applied[-1].joint_positions, float)
    assert np.allclose(cmd[_FING], d._finger_cmd), "the grip's targets did not reach the action"


def test_a_teleported_stage_is_left_alone():
    """Off the drives `_force` writes every joint kinematically already; a second writer here
    would put two hands in one motion."""
    d = _apply_demo(drive=False)
    d._apply(np.zeros(len(_NAMES)))
    assert d.kin == [], f"the freeze wrote on a stage that teleports: {d.kin}"


def test_no_pin_no_write():
    """Every `_apply` outside a pinning motion -- most of the cycle -- must reach the drives
    untouched, and must not need the attribute to exist at all."""
    d = _apply_demo(pin=None)
    d._apply(np.zeros(len(_NAMES)))
    assert d.kin == [], f"the freeze wrote with no pin standing: {d.kin}"


# no pytest in the venv? run direct.
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
