"""The gripper's public surface: `move_gripper`, `open_gripper`, `close_gripper`, `hold_gripper`,
`GRIPPER_OPEN`,
`GripperMove`.

Owns the `finger_*` and `palm_finger_*` DOFs; it hands `_apply` the full vector every step, so it
commands the wrist too and merely leaves it where it found it. Force and gain policy stay in
`morph/robot.py`. `close_gripper` is the force close, whose servo is `force_close.py` beside this
module. No Isaac import -- `math`, `numpy`, `morph.config` and `morph.gripper.force_close`, which
imports no more than those -- so it imports offline; the running `demo` it is handed is what needs
a live stage."""
import dataclasses
import math

import numpy as np

from morph.config import HEADLESS
from morph.gripper.force_close import (ForceBalanceStage, J1_CURL_MAX, J2_PER_J1,
                                        J3_PER_J1)

__all__ = ["FingersMixin", "ForceBalanceStage", "GRIPPER_OPEN",
           "GripperMove", "close_gripper", "gripper_fraction",
           "hold_gripper", "move_gripper", "open_gripper"]

# The one open the robot commands: the RELEASE open. j1 is the side-grip angle, the rest the
# recorded geometry. The PRE-GRIP hand is the stage keyframe, written nowhere in Python.
GRIPPER_OPEN = {
    "finger_a_joint_1_1": -0.55, "finger_b_joint_1_1": -0.55, "finger_c_joint_1_1": -0.55,
    "finger_a_joint_2_1": 0.0, "finger_b_joint_2_1": 0.0, "finger_c_joint_2_1": 0.0,
    "finger_a_joint_3_1": -0.0476, "finger_b_joint_3_1": -0.0476, "finger_c_joint_3_1": -0.0476,
    "palm_finger_b_joint_1": 0.161, "palm_finger_c_joint_1": 0.161,
}


# The reference full curl, per phalanx, read from the force close rather than copied: `_1` is the
# knuckle, `_2` the middle, `_3` the distal, and the last index is the ARM.
_CURL = {"1": J1_CURL_MAX, "2": J1_CURL_MAX * J2_PER_J1, "3": J1_CURL_MAX * J3_PER_J1}


def gripper_fraction(demo, frac, base=None):
    """Finger vector in `_f_idx` order: 0 is `base` -- the hand as it stands, so the control moves
    nothing until it is asked to -- and 1 the reference full curl. Only arm one's hand is mapped;
    the palm spread is carried through, being a spread and not a curl."""
    live = np.asarray(demo.robot.get_joint_positions(), float)[demo._f_idx]
    out = (live if base is None else np.asarray(base, float)).copy()
    f = min(max(float(frac), 0.0), 1.0)
    for k, n in enumerate(demo._f_names):
        if n in GRIPPER_OPEN and n.startswith("finger_"):
            out[k] = (1.0 - f) * out[k] + f * _CURL[n.split("_")[3]]
    return out


@dataclasses.dataclass(frozen=True)
class GripperMove:
    """The pose a finger motion ends on. Nothing here judges the motion: the only caller that
    could, the friction close, drives the fingers from `morph/close/` and not through this."""

    # the COMMANDED pose the motion ends on
    q_cmd: object = None
    # the MEASURED pose at exit, copied. NOT `ArmMove.q_end`, which is a command to drive forward
    q_meas: object = None


def hold_gripper(demo, grip, secs, *, kin, on_step=None):
    """Hold the WHOLE robot still for `secs` -- base pin, arm and the latched command `grip`,
    which `_hold` writes in full every step -- so the fingers keep their load. The finger effort
    ceiling is inherited from the close, not re-stated here."""
    demo._hold(grip, int(secs / demo.dt), kin=kin, on_step=on_step)


def _finger_of(jname):
    return next((c for c in "abc" if f"_{c}_" in jname), "a")


def move_gripper(demo, q_base, targets, secs, stagger=0.15, on_step=None, release=False):
    """Animate the finger joints to `targets` at a FIXED arm config, with a per-finger start
    stagger."""
    # The package owns the finger DOFs and nothing else; refuse before the hand is latched, and
    # before the `idx` filter, or a wrist joint the stage does not carry is dropped in silence.
    alien = sorted(jn for jn in targets
                   if not (jn.startswith("finger_") or jn.startswith("palm_finger_")))
    if alien:
        raise ValueError(f"move_gripper drives finger joints only; refused {alien}")
    q = q_base.copy()
    n = max(2, int(secs / demo.dt))
    moves = []                                           # (joint idx, start, target, start-frac, window)
    for jn, tgt in targets.items():
        if jn in demo.idx:
            d = _finger_of(jn)
            d0 = {"a": 1.0, "b": 0.0, "c": 0.5}[d] * stagger / max(secs, 1e-6)
            # phalanx window within this finger's own close: j1 leads, j3 hooks last
            w0, w1 = {"1": (0.00, 0.55), "2": (0.45, 0.85),
                      "3": (0.75, 1.00)}.get(jn.split("joint_")[-1][0], (0.0, 1.0))
            if release:
                # RELEASE: all three together, or the still-loaded one pushes the object.
                # Phalanx order stays the close's — uncurling j2/j3 first extends into it.
                d0 = 0.0
            moves.append((demo.idx[jn], float(q[demo.idx[jn]]), float(tgt), d0, w0, w1))
    if not moves:
        raise ValueError(f"no target joint is on this articulation: {sorted(targets)}")
    bx, by, byaw = demo.base_ledger()
    demo._finger_cmd = q[demo._f_idx].copy()
    for k in range(n):
        for ji, s0, tgt, d0, w0, w1 in moves:
            s = float(np.clip(((k + 1) / n - d0) / max(1e-6, 1.0 - d0), 0.0, 1.0))
            # Underactuated wrap: proximal swings in first, then middle, then the distal hooks.
            s = float(np.clip((s - w0) / max(1e-6, w1 - w0), 0.0, 1.0))
            s = 0.5 - 0.5 * math.cos(math.pi * s)        # smoothstep, like MuJoCo's _set_gripper
            q[ji] = s0 + s * (tgt - s0)
        demo._finger_cmd = q[demo._f_idx].copy()
        demo.set_base(bx, by, byaw)
        demo._force(q)
        demo._apply(q)
        if on_step is not None:
            on_step()
        demo.world.step(render=not HEADLESS)
    return GripperMove(q_cmd=q, q_meas=np.array(demo.robot.get_joint_positions(), float))


def open_gripper(demo, q_base, secs, *, on_step=None):
    """Open to `GRIPPER_OPEN`. Unjudged: the caller's own post-settle re-measure is taken well
    after anything this can see."""
    # Off the drives the stagger keeps a still-loaded finger from pushing the object; on them
    # every finger must let go together.
    return move_gripper(demo, q_base, dict(GRIPPER_OPEN), secs,
                        on_step=on_step, release=demo._drive_on())


def close_gripper(demo, st):
    """Close on measured pad force until every pad reads FB_TARGET. The servo is
    `ForceBalanceStage`, mixed into `Demo`; this is the API's one name for closing."""
    return demo._close_force_balance(st)


def __getattr__(name):
    # `fingers` imports Isaac at module level; this module must stay importable offline.
    if name == "FingersMixin":
        from morph.gripper.fingers import FingersMixin
        return FingersMixin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
