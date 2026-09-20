"""`finger_vector_line` -- the G2a close-entry measurement. One row: the printed DOFs must follow
`_f_idx`, since G2b reads that line as the open constant and a re-ordered list would mis-assign
every value. Delete with the diagnostic in G2b.
Offline: no Isaac, no simulator. Run: ./.venv/bin/python3 tests/test_finger_vector_line.py"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from morph.diagnostics import finger_vector_line                                    # noqa: E402


class _Demo:
    """Arm two's fingers sit BETWEEN arm one's, so `_f_idx` is not a prefix and a row that walked
    the names in source order, or "abc", cannot pass by accident."""

    def __init__(self):
        self.names = ["ArmLeftJoint_1", "finger_a_joint_1_1", "palm_finger_b_joint_1",
                      "finger_b_joint_1_1", "gripper_z_rotation_1", "finger_a_joint_1_2",
                      "finger_c_joint_3_1"]
        self.idx = {n: i for i, n in enumerate(self.names)}
        self._f_idx = np.array([i for n, i in self.idx.items()
                                if n.startswith("finger_") or n.startswith("palm_finger")])
        self._f_names = [self.names[i] for i in self._f_idx]
        self._finger_cmd = None


def _fields(line):
    return [seg.split() for seg in line.split(": ", 1)[1].split(" | ")]


def test_the_line_walks_f_idx_and_pairs_each_name_with_its_own_value():
    d = _Demo()
    q = np.arange(len(d.names), dtype=float) / 100.0
    d._finger_cmd = np.asarray([q[i] + 0.5 for i in d._f_idx])
    got = _fields(finger_vector_line(d, q, "close-entry fingers"))
    assert [f[0] for f in got] == [d.names[i] for i in d._f_idx], (
        f"the line prints {[f[0] for f in got]}, not `_f_idx` order "
        f"{[d.names[i] for i in d._f_idx]}: G2b would read every value against the wrong joint")
    for f, i in zip(got, d._f_idx):
        assert abs(float(f[2]) - q[i]) < 1e-9, f"{f[0]} reads {f[2]}, not its own {q[i]:+.4f}"
        assert abs(float(f[4]) - (q[i] + 0.5)) < 1e-9, \
            f"{f[0]} commands {f[4]}, not its own {q[i] + 0.5:+.4f}"


def test_the_commanded_half_is_absent_before_the_channel_is_set():
    d = _Demo()
    q = np.zeros(len(d.names))
    assert "cmd" not in finger_vector_line(d, q, "close-entry fingers"), \
        "an unset `_finger_cmd` prints a command anyway: the line invents a channel"


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
