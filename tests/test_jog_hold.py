"""The jog holds the pose it started from; it does not chase the robot's own sag.

`_jog_write` commands four arm joints, the closure passives and the fingers. Every OTHER joint --
arm two, the wrist, the gripper links -- is carried through untouched, so where that vector comes
from decides whether they hold or drift. Built from a LIVE read they integrate their own sag: the
measurement becomes the next target, and a joint settling under gravity walks down one tick at a
time. Run: ./.venv/bin/python3 tests/test_jog_hold.py
"""
import ast
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOGGED = ("ColumnLeftBearingJoint_1", "ColumnRightBearingJoint_1", "ArmLeftJoint_1", "BaseJoint_1")
# arm two's shoulder and arm one's wrist: neither is commanded by the jog
PARKED = ("ColumnLeftBearingJoint_2", "gripper_y_rotation_1")


class _Sagging:
    """A robot whose uncommanded joints droop a little every time they are read."""

    def __init__(self, names, sag=0.01):
        self.names = list(names)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.q = np.zeros(len(self.names))
        self._sag = sag
        self.written = []
        self._f_idx = np.array([i for i, n in enumerate(self.names) if n.startswith("finger_")])
        self._finger_cmd = None
        self.dt = 1.0 / 240.0
        self.clut = {"passives": [], "grid": None}
        self.world = type("W", (), {"step": lambda *a, **k: None})()
        # the articulation view the real code reads through, so a reverted fix fails on the DRIFT
        # rather than on a missing attribute
        self.robot = self

    # the articulation: reading it shows the sag that has accumulated
    def get_joint_positions(self):
        for n in PARKED:
            self.q[self.idx[n]] -= self._sag
        return self.q.copy()

    def _closure_passives(self, dh, a1):
        return {}

    def base_pose(self):
        return (0.0, 0.0, 0.0)

    def set_base(self, x, y, yaw, force=False):
        pass

    def _force(self, q, qv=None):
        self.written.append(np.asarray(q, float).copy())

    def _apply(self, q, qv=None):
        pass


def _demo():
    from morph.align import AlignMixin
    names = list(JOGGED) + list(PARKED) + ["finger_a_joint_1_1"]
    d = _Sagging(names)
    d._jog_write = AlignMixin._jog_write.__get__(d)
    d.jog = {"anchor": (0.0, 0.0, 0.0), "fhold": np.zeros(len(d._f_idx)),
             "qhold": d.get_joint_positions().copy()}
    return d


def test_an_uncommanded_joint_holds_its_pose_over_a_long_jog():
    d = _demo()
    start = {n: float(d.jog["qhold"][d.idx[n]]) for n in PARKED}
    for _ in range(200):
        d._jog_write(np.array([0.3, 0.3, 0.2, 0.0]))
    assert d.written, "the jog never commanded anything"
    last = d.written[-1]
    for n in PARKED:
        moved = abs(float(last[d.idx[n]]) - start[n])
        assert moved < 1e-9, (
            f"{n} was commanded {moved * 1000:.0f}mm from where the jog started: the write is "
            f"following the measured sag instead of the pose it snapshotted")


def test_the_jogged_joints_still_follow_their_command():
    """The hold must not freeze the joints the jog exists to move."""
    d = _demo()
    d._jog_write(np.array([0.41, 0.42, 0.43, 0.44]))
    last = d.written[-1]
    for n, want in zip(JOGGED, (0.41, 0.42, 0.43, 0.44)):
        assert abs(float(last[d.idx[n]]) - want) < 1e-9, f"{n} did not take its jog value"


def test_arming_the_jog_does_not_move_the_arm():
    """The posture bias is anchored where the jog started, so with no target error and no slider
    touched, nothing is commanded to move. A fixed reference pulls the arm toward it the instant
    jog is armed, which is a lurch the operator did not ask for."""
    from morph.align import AlignMixin
    d = _demo()
    q0 = np.array([1.20, 1.2481, 0.10, 0.0])            # the park pose: columns at 1.2
    d.jog.update({"q": q0.copy(), "qref": q0.copy(),
                  "q0dh": float(q0[1] - q0[0])})
    # a hand whose position depends on the joints, so the Jacobian is real and err starts at zero
    d._grip_fk = lambda h1, h2, a1, th: np.array([0.5 + a1, 0.1 * th, h1])
    d.grip_pos = lambda: d._grip_fk(*d.jog["q"])
    d.jog["origin"] = d.grip_pos().copy()
    d.jog["target"] = d.jog["origin"].copy()
    d._jog_tick = AlignMixin._jog_tick.__get__(d)
    for _ in range(200):
        d._jog_tick()
    moved = float(np.max(np.abs(d.jog["q"] - q0)))
    assert moved < 1e-6, (
        f"arming the jog walked the arm {moved * 1000:.0f}mm with nothing asked of it: the "
        f"posture reference is not the pose the jog started from")


def test_the_write_is_built_from_the_snapshot_not_a_live_read():
    """The shape, so the behaviour above cannot be satisfied by a stale cache somewhere else."""
    src = open(os.path.join(ROOT, "morph", "align.py"), encoding="utf-8").read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_jog_write")
    reads = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
             and isinstance(c.func, ast.Attribute) and c.func.attr == "get_joint_positions"]
    assert not reads, (
        f"_jog_write calls get_joint_positions() at line {reads[0].lineno}: the command vector "
        f"must come from jog['qhold'], or every uncommanded joint chases its own sag")


def test_a_level_boom_is_not_tilted_by_arming_the_jog():
    """The park pose holds the boom level. A blanket pitch floor ramps it to 10 degrees on
    arming and walks h1 31mm while the servo holds the hand on target."""
    from morph.align import AlignMixin, JOG_DH_FLOOR
    d = _demo()
    level = np.array([1.20, 1.20, 0.10, 0.0])           # park: both columns equal, dh = 0
    d.jog.update({"q": level.copy(), "qref": level.copy(),
                  "q0dh": float(level[1] - level[0])})
    d._grip_fk = lambda h1, h2, a1, th: np.array([0.5 + a1, 0.1 * th, h1])
    d.grip_pos = lambda: d._grip_fk(*d.jog["q"])
    d.jog["origin"] = d.grip_pos().copy()
    d.jog["target"] = d.jog["origin"].copy()
    d._jog_tick = AlignMixin._jog_tick.__get__(d)
    for _ in range(40):
        d._jog_tick()
    dh = float(d.jog["q"][1] - d.jog["q"][0])
    assert abs(dh) < 1e-9, f"arming the jog pitched a level boom to {dh:.4f} rad"
    assert abs(float(d.jog["q"][0]) - 1.20) < 1e-9, (
        f"h1 moved {abs(float(d.jog['q'][0]) - 1.20) * 1000:.0f}mm with nothing asked of it")


def test_a_raised_boom_still_keeps_the_working_pitch_floor():
    """The floor is not removed: an arm already above it may not be jogged below it."""
    from morph.align import AlignMixin, JOG_DH_FLOOR
    d = _demo()
    q = np.array([1.20, 1.20 + 0.05, 0.10, 0.0])        # started well above the floor
    d.jog.update({"q": q.copy(), "qref": q.copy(), "q0dh": 0.05})
    d._grip_fk = lambda h1, h2, a1, th: np.array([0.5 + a1, 0.1 * th, h1])
    d.grip_pos = lambda: d._grip_fk(*d.jog["q"])
    d.jog["origin"] = d.grip_pos().copy()
    # ask for a hand far BELOW, so the servo drives the columns together
    d.jog["target"] = d.jog["origin"] - np.array([0.0, 0.0, 0.5])
    d._jog_tick = AlignMixin._jog_tick.__get__(d)
    for _ in range(300):
        d._jog_tick()
    dh = float(d.jog["q"][1] - d.jog["q"][0])
    assert dh >= JOG_DH_FLOOR - 1e-9, (
        f"the jog drove the boom to {dh:.4f} rad, under the {JOG_DH_FLOOR} working floor")


def test_the_gripper_slider_starts_where_the_hand_is_and_ends_at_a_full_curl():
    """0 must command the hand it was armed with -- a fixed OPEN there snaps the fingers 34 degrees
    on first touch -- and 1 must reach the reference curl, not the grip recorded on an object that
    is not in the hand."""
    from morph.gripper.api import GRIPPER_OPEN, gripper_fraction, J1_CURL_MAX

    class _Hand:
        def __init__(self):
            self._f_names = list(GRIPPER_OPEN)
            self._f_idx = np.arange(len(self._f_names))
            self.close_traj = None
            self.robot = self

        def get_joint_positions(self):
            return np.zeros(len(self._f_names))

    d, park = _Hand(), np.full(len(GRIPPER_OPEN), 0.0495)
    at_0 = gripper_fraction(d, 0.0, base=park)
    assert np.allclose(at_0, park), (
        f"slider 0 moved the hand by up to {np.max(np.abs(at_0 - park)):.3f} rad")

    at_1 = gripper_fraction(d, 1.0, base=park)
    knuckles = [k for k, n in enumerate(d._f_names) if n.endswith("_joint_1_1")]
    for k in knuckles:
        assert abs(at_1[k] - J1_CURL_MAX) < 1e-9, (
            f"{d._f_names[k]} closes to {at_1[k]:+.3f}, not the reference curl {J1_CURL_MAX:+.3f}")
    palms = [k for k, n in enumerate(d._f_names) if n.startswith("palm_")]
    assert np.allclose(at_1[palms], park[palms]), "the palm spread is not a curl and must not move"


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
