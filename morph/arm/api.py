"""The two planned-motion primitives, their result type, and the open-loop ramp's check.

`move_to_pose` is sampling-based, always collision-checked, and carries NO exemption -- there is no
parameter for one. `move_linear` is a straight line, checked per waypoint, with an exemption scoped
to the single call; it never replans, so a line that cannot continue stops rather than becoming a
different line.
"""
import dataclasses
import enum
import math

import numpy as np

from morph.arm.cartesian import _V_MAX as V_MAX
from morph.arm.model import ArmModel
from morph.arm.collision import MARGIN, Q8_TOL_M, Q8_TOL_RAD, sweep_top
from morph.arm.execute import commanded_fingers, execute_path
from morph.arm.obstacles import blame
from morph.arm.plan import goal_ik, goal_set, plan_arm
from morph.arm.surface import ArmMixin

__all__ = ["ArmMove", "Outcome", "move_to_pose", "move_linear", "goal_ik", "cartesian_retreat", "ArmMixin",
           "ArmModel", "MARGIN", "Q8_TOL_M", "Q8_TOL_RAD", "ramp_blocked", "sweep_top", "V_MAX"]


class Outcome(str, enum.Enum):
    """What became of one motion. `str` mixin so a log line prints the value, not the repr."""

    ARRIVED = "arrived"            # executed, and inside the arrival band of the goal
    NO_PLAN = "no_plan"            # nothing executed; the arm is where it was
    REFUSED = "refused"            # a check disproved a pose; stopped at the last cleared one
    NO_ARRIVAL = "no_arrival"      # ran the whole path and stopped short
    PAYLOAD_LOST = "payload_lost"  # the stop broke the friction grasp


@dataclasses.dataclass(frozen=True)
class ArmMove:
    """One motion's result. Falsy unless ARRIVED, and a pose only when it did."""

    outcome: Outcome
    reason: str = ""
    tags: tuple = ()
    q_end: object = None

    def __post_init__(self):
        if self.outcome is not Outcome.ARRIVED and self.q_end is not None:
            raise ValueError("a motion that did not arrive carries no pose")

    def __bool__(self):
        return self.outcome is Outcome.ARRIVED


def _new_tags(demo, before):
    """Tags THIS call counted, by diffing the demo's counter. The facade must not count itself:
    the delegates already do, and a second tally doubles what the acceptance gate reads."""
    after = dict(getattr(demo, "_plan_fallbacks", {}) or {})
    out = []
    for k, v in after.items():
        out.extend([k] * (v - before.get(k, 0)))
    return tuple(out)


def _as_q8(demo, target):
    """`target` as an 8-vector, or None if it is neither a q8 nor a full joint vector."""
    try:
        t = np.asarray(target, float).ravel()
        if not np.all(np.isfinite(t)):
            return None
        if t.size == len(ArmModel.Q8):
            return t.copy()
        idx = demo.idx
        if t.size >= max(idx.values()) + 1:
            return np.array([t[idx[n]] for n in ArmModel.Q8], float)
    except Exception:                                  # noqa: BLE001
        return None
    return None


def _exemption(allow_contact_with):
    """The exemption as a tuple of prim names, or None if unusable. A bare string is refused:
    it is iterable, so one prim name would become one-character names that exempt nothing."""
    if allow_contact_with is None:
        return ()
    if isinstance(allow_contact_with, str):
        return None
    try:
        names = tuple(allow_contact_with)
    except TypeError:
        return None
    return names if all(isinstance(n, str) for n in names) else None


def ramp_blocked(demo, q_to):
    """Why an open-loop ramp to `q_to` must not run, or None. `("blocked", why)` names the first
    link inside MARGIN of a WORLD box; `("unbuildable", why)` means the check could not be made
    and refuses rather than passes.

    Blind to the chassis and to the grasp target, and neither is a claim the pair is safe: the
    hand is inside MARGIN of the object it reaches for at every seat, the boom is inside the deck
    box, and NOTHING judges that pair -- `_clears_chassis` is an XY grip-point test. See FINDINGS.
    """
    focus = getattr(demo, "_focus_obj", None)
    if not focus:
        return ("unbuildable", "no grasp target to exempt")
    q8 = _as_q8(demo, q_to)
    if q8 is None:
        return ("unbuildable", "the target is not a finite joint vector")
    # The SWEPT union is a different hand, not a safer one: at the seat it refuses a pose the
    # commanded pads clear comfortably.
    fingers = commanded_fingers(demo)
    if fingers is None:
        return ("unbuildable", "the commanded hand is unavailable")
    try:
        paths = []
        obs = demo._arm_obstacles((focus,), diagnostic_paths=paths)
        # `zip` would truncate to the shorter of the two and misname every prim past the gap.
        if len(paths) != len(obs):
            return ("unbuildable",
                    f"{len(paths)} prim paths for {len(obs)} obstacles")
        keep = [(o, p) for o, p in zip(obs, paths) if o[1] != "chassis"]
        hit = blame(demo._arm_model(), q8, [o for o, _ in keep], [p for _, p in keep], None,
                    demo._arm_base_world(), fingers)
    except Exception as exc:                 # noqa: BLE001 -- no checker means no checked motion
        return ("unbuildable", str(exc))
    return None if hit is None else (
        "blocked", f"{hit[0]} within {MARGIN * 1000:.0f}mm of {hit[1]}")


def _arrival(demo, goal8, tags, q_end):
    """Arrival against the model's own bands."""
    rm, rr = ArmModel.q8_resid(demo._arm_q8(), goal8)
    if rm > Q8_TOL_M or rr > Q8_TOL_RAD:
        return ArmMove(Outcome.NO_ARRIVAL,
                       f"stopped {rm * 1000:.1f}mm / {rr * 1000:.1f}mrad short", tags)
    return ArmMove(Outcome.ARRIVED, "arrived", tags, q_end)


def _stopped(tags, reason):
    """The result of a motion the executor did not finish. Only the delegates' own tag can say
    whether the stop broke the friction grasp, so the outcome is read back off `tags`."""
    out = Outcome.PAYLOAD_LOST if "payload-lost" in tags else Outcome.REFUSED
    return ArmMove(out, reason, tags)


def move_to_pose(demo, target, secs, *, held=None, tag="arm", regoal=None,
                 on_step=None, hold_fingers=True):
    """Plan a collision-checked route to `target` (a q8 or a full joint vector) and execute it.
    No exemption parameter: this primitive always plans against the whole obstacle set."""
    before = dict(getattr(demo, "_plan_fallbacks", {}) or {})
    goal8 = _as_q8(demo, target)
    if goal8 is None:
        return ArmMove(Outcome.NO_PLAN, "bad-target: not a q8 or a full joint vector",
                       ("bad-target",))

    # The hand this motion will FLY. `hold_fingers` is what holds it constant, so when it is False the
    # honest answer is None -- the swept box -- not a snapshot of a hand still free to move.
    wps = plan_arm(demo, goal_set(demo, goal8, (), held), held=held,
                   fingers=commanded_fingers(demo) if hold_fingers else None)
    if not wps:
        return ArmMove(Outcome.NO_PLAN, "no route found", _new_tags(demo, before))

    def _regoal():
        g = regoal()
        return None if g is None else goal_set(demo, g, (), held)
    q_end = execute_path(demo, wps, secs, held=held, tag=tag,
                         regoal=None if regoal is None else _regoal,
                         on_step=on_step, hold_fingers=hold_fingers)
    tags = _new_tags(demo, before)
    if q_end is None:
        return _stopped(tags, "aborted at a checkpoint after a failed replan")
    # Neither the request nor the caller's list: a goal SET has several IK branches and a replan rebinds
    # the executor's own. The executor must report where it actually drove.
    goal_reached = getattr(demo, "_exec_goal8", None)
    if goal_reached is None:
        goal_reached = np.asarray(wps[-1], float)
    return _arrival(demo, np.asarray(goal_reached, float), tags, q_end)



def _allowance(demo, names):
    """(obstacles, prim paths) for this motion, or (None, reason). Never cached."""
    paths = []
    try:
        obs = demo._arm_obstacles(names, diagnostic_paths=paths)
    except Exception as e:                             # noqa: BLE001 -- refuse, never raise
        # Nothing left to check against. The caller is mid-motion with a payload; an exception
        # here would leave it holding one and the failure would surface as something else.
        return None, f"bad-args: the obstacle set could not be built ({e})"
    return obs, paths



def _fly_ramp(demo, q8, q_to, secs, obs, paths, held, fingers,
              names, tag, before, on_step=None):
    """Fly a closed-form column ramp: every commanded pose checked, halting on the last cleared one.

    The eased sequence is the CHECK sequence: `n = max(1, int(secs/dt))` poses along one straight
    line in joint space, so the executor's own retimed samples lie between two of them. Checking it
    up front is equivalent to checking before each write, because the obstacle set is built once
    and the payload is rigid in the hand for the motion. `secs` reaches the executor as a floor,
    which is what keeps the ramp as slow as the insert asked for.
    """
    n = max(1, int(secs / demo.dt))
    wps = [q8 + (0.5 - 0.5 * math.cos(math.pi * (k + 1) / n)) * (q_to - q8) for k in range(n)]
    m = demo._arm_model()
    base = demo._arm_base_world()
    # ONE pass. The padded pair is judged inside `blame`'s own `first_hit`, at its floor, so a
    # second sweep for a bound would re-run the same test on the same poses.
    bad = next((i for i, q in enumerate(wps)
                if blame(m, q, obs, paths, held, base, fingers) is not None), None)
    if bad == 0:
        blamed = blame(m, wps[0], obs, paths, held, base, fingers)
        return ArmMove(Outcome.NO_PLAN,
                       f"ramp blocked at its first pose: {blamed[0]} inside the margin of "
                       f"{blamed[1]}" if blamed else "ramp blocked at its first pose",
                       _new_tags(demo, before))

    flown = wps if bad is None else wps[:bad]
    # `waypoints[0]` is the START pose, which the executor retimes FROM. Handed the eased sequence
    # alone the ramp would start from its own second pose and lose `secs/n` of the travel.
    q_end = execute_path(demo, [q8] + list(flown), secs * len(flown) / n, held=held,
                         grasp_names=names, tag=tag, checkpoint_every=len(flown) + 2,
                         on_step=on_step)
    tags = _new_tags(demo, before)
    if q_end is None:
        return _stopped(tags, "stopped inside the ramp")
    if bad is not None:
        blamed = blame(m, wps[bad], obs, paths, held, base, fingers)
        who = f"{blamed[0]} inside the margin of {blamed[1]}" if blamed else "the allowance bound"
        print(f">>> linear[{tag}]: ramp halted at {bad}/{n} -- {who}", flush=True)
        return ArmMove(Outcome.REFUSED, f"ramp halted at {bad}/{n}: {who}", tags)
    print(f">>> linear[{tag}]: column ramp, {n} steps", flush=True)
    return _arrival(demo, np.asarray(wps[-1], float), tags, q_end)


def move_linear(demo, delta_or_target, frame, *, kind="delta", allow_contact_with=None,
                link=None, held=None, tag="linear", fingers=None, ik_restarts=0, secs=None,
                on_step=None):
    """A straight line for the hand, checked at every waypoint; a blocked line halts at the last
    cleared pose, never replanning. `frame` world|base; `kind` delta|target; with `secs` a base-z
    delta is a column ramp; `ik_restarts` > 0 may hop IK branches."""
    from morph.arm.cartesian import plan_cartesian
    from morph.arm.linear import column_ramp

    before = dict(getattr(demo, "_plan_fallbacks", {}) or {})
    names = _exemption(allow_contact_with)
    if names is None or frame not in ("world", "base") or kind not in ("delta", "target"):
        return ArmMove(Outcome.NO_PLAN, "bad-args")
    obs, paths = _allowance(demo, names)
    if obs is None:
        return ArmMove(Outcome.NO_PLAN, paths)

    m = demo._arm_model()
    q8 = np.asarray(demo._arm_q8(), float)
    if not np.all(np.isfinite(q8)):
        # A NaN pose reads CLEAR: every box comparison is False, so `first_hit` skips them all.
        demo._fallback("linear-start-not-finite")
        return ArmMove(Outcome.REFUSED, "start pose is not finite", _new_tags(demo, before))
    # ALWAYS the real base: it carries links out to world for "world" boxes, and `plan_cartesian`
    # and `blame` share this one binding; withheld, they judged right only on the origin.
    base = demo._arm_base_world()

    # Base frame only: the columns are base-z, so a world-frame ask would need the chassis proven level
    # first. Without `secs` there is no step count to be exact about, so the planner retimes instead.
    if kind == "delta" and frame == "base" and secs:
        ramp_to, why = column_ramp(m, q8, np.asarray(delta_or_target, float), demo._arm_bounds())
        if why is not None:
            return ArmMove(Outcome.NO_PLAN, why, _new_tags(demo, before))
        if ramp_to is not None:
            return _fly_ramp(demo, q8, ramp_to, float(secs), obs, paths,
                             held, fingers, names, tag, before, on_step)

    # `plan_cartesian` takes ONE frame for the line and the check together, so the line is converted
    # here where `frame` is known. The round trip is exact.
    _vec = np.asarray(delta_or_target, float)
    if frame == "base":
        _bt, _bR = base
        _vec = _bR @ _vec if kind == "delta" else _bt + _bR @ _vec

    diag = {}
    wps, err = plan_cartesian(m, q8, link=(link or m.hand_link), obstacles=obs,
                              margin=MARGIN, base=base, bounds=demo._arm_bounds(), step=0.005,
                              held=held,
                              fingers=fingers, diag_out=diag, ik_restarts=ik_restarts,
                              **{kind: _vec})
    if not wps:
        # The last cleared pose is the end of the validated prefix, not q_start. Only the collision
        # branch builds one: an IK failure returns before it.
        blamed = (None if diag.get("q_blocked") is None
                  else blame(m, np.asarray(diag["q_blocked"], float), obs, paths, held, base,
                              fingers))
        why = (f"line not clear: {err}" if blamed is None else
               f"line not clear: {err} -- {blamed[0]} inside the margin of {blamed[1]}")
        prefix = diag.get("prefix") or []
        if not len(prefix):
            return ArmMove(Outcome.REFUSED, why, _new_tags(demo, before))
        # The prefix needs no separate bound: `plan_cartesian` validated it against the same
        # obstacle list, pad included, so its floor was enforced when the prefix was built.
        print(f">>> linear[{tag}]: blocked, executing the validated prefix of "
              f"{len(prefix)} waypoints -- {why}", flush=True)
        q_end = execute_path(demo, prefix, max(1, len(prefix) - 1) * demo.dt, held=held,
                             grasp_names=names, tag=tag, on_step=on_step,
                             checkpoint_every=len(prefix) + 1)
        tags = _new_tags(demo, before)
        if q_end is None:
            return _stopped(tags, f"stopped inside the prefix; {why}")
        return ArmMove(Outcome.REFUSED, f"halted at the last cleared pose; {why}", tags)

    # A checkpoint past the end of the path DISABLES checkpoint replanning. Without it the executor
    # re-plans a blocked checkpoint by design, and the straight line quietly becomes another line.
    print(f">>> linear[{tag}]: {len(wps)} waypoints", flush=True)
    q_end = execute_path(demo, wps, max(1, len(wps) - 1) * demo.dt, held=held,
                         grasp_names=names, tag=tag, on_step=on_step,
                         checkpoint_every=len(wps) + 1)
    tags = _new_tags(demo, before)
    if q_end is None:
        return _stopped(tags, "stopped at the last cleared waypoint")
    return _arrival(demo, np.asarray(wps[-1], float), tags, q_end)


def cartesian_retreat(demo, obj_idx):
    """Back the hand off the object it released along the mouth axis -- the one motion that
    exempts the hand from that object's box -- then plan on to the carry pose with the object as
    an ordinary obstacle; False if either part cannot run."""
    from morph.arm.retreat import cartesian_retreat as _retreat
    return _retreat(demo, obj_idx)
