"""`morph/arm/api.py` -- the two primitives' contract.

House rule: observe the COMMAND, not the decision in front of it, and be universal over the sites.
Every counting test asserts an exact sequence; none asserts that something was merely found.
"""
import ast
import inspect
import os
import subprocess
import sys
import types

import numpy as np
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from morph.arm.api import (ArmMove, Outcome, move_linear,                              # noqa: E402
                           move_to_pose, _exemption, _new_tags)
from morph.arm.obstacles import chassis_pad                                # noqa: E402
from morph.arm.model import ArmModel  # noqa: E402
from morph.arm.collision import MARGIN                                   # noqa: E402

# api's executor seam, routed to the fake under test (the real one needs a robot). Module-wide:
# the gate runs one process per suite.
import morph.arm.api as _api_mod                                                    # noqa: E402
_api_mod.execute_path = lambda demo, *a, **k: demo._execute_arm_path(*a, **k)
_api_mod.commanded_fingers = lambda demo: demo._commanded_fingers()
_api_mod.plan_arm = lambda demo, *a, **k: demo._plan_arm(*a, **k)
_api_mod.goal_set = lambda demo, *a, **k: demo._goal_set(*a, **k)

_MODEL = ArmModel.load(os.path.join(ROOT, "usd", "_arm_model.json"),
                       os.path.join(ROOT, "usd", "_closure_lut.json"))
# A shipped pose, so the straight line below starts somewhere the linkage actually reaches.
_START = np.array(
    [0.0, 0.30, 0.30, 0.20, 0.0, 0.0, 0.0, 0.0], float)


class _Demo:
    """The delegate surface both primitives use, recording every argument they pass."""

    dt = 1.0 / 240.0
    stage = None                  # `_annotate` resolves prim names through it; stubbed in tests

    def __init__(self, *, wps=None, q_end="ok", block_at=None, end_at_goal=True, moves=False):
        self.idx = {n: i for i, n in enumerate(ArmModel.Q8)}
        self._plan_fallbacks = {}
        self.seen_grasp = []          # every grasp_names any delegate received, in order
        self.seen_obs = []            # every obstacle list `_arm_obstacles` handed back
        self.checkpoints = []         # every checkpoint_every `execute_path` received
        self.seen_secs = []           # ...and every `secs`, which is now the executor's FLOOR
        self.seen_on_step = []        # every on_step `execute_path` received
        self.plan_arm_calls = 0
        self.commanded = []
        # The hand this demo reports. A dict here is a SENTINEL: every check the primitive makes
        # must carry this exact object, or the planner and the executor are judging two hands.
        self.fingers = None
        self.seen_fingers = []        # every `fingers` any delegate received, in order
        # Longer than `execute_path`'s default checkpoint_every, so the default CANNOT
        # satisfy the assertion below and deleting the explicit one is caught.
        self._wps = wps if wps is not None else [np.zeros(8) + i * 0.01 for i in range(20)]
        self._q_end = q_end
        self._block_at = block_at
        self._end_at_goal = end_at_goal
        self.moves = moves

    def _arm_q8(self):
        # `moves=True` makes the fake arm END WHERE IT WAS DRIVEN, which `_arrival` needs. Off by
        # default -- the tests that pin arrival SEMANTICS want to control both sides.
        if self.moves and self.commanded:
            return np.asarray(self.commanded[-1], float).copy()
        return _START.copy()

    def _fallback(self, reason):
        self._plan_fallbacks[reason] = self._plan_fallbacks.get(reason, 0) + 1

    def _arm_model(self):
        # the real model: `plan_cartesian` runs real FK and IK
        return _MODEL

    def _arm_base_world(self):
        return (np.zeros(3), np.eye(3))

    def _arm_bounds(self):
        return (np.full(8, -10.0), np.full(8, 10.0))

    # A real box the straight line below runs into, with a prim path, so an ACM entry has
    # something to name. Empty by default: most tests want an unobstructed line.
    obstacles = []
    obstacle_paths = []

    def _arm_obstacles(self, grasp_names=(), *, diagnostic_paths=None, **kw):
        self.seen_grasp.append(tuple(grasp_names))
        self.seen_obs.append([tuple(o) for o in self.obstacles])
        if diagnostic_paths is not None:
            diagnostic_paths.extend(self.obstacle_paths)
        return list(self.obstacles)

    def _goal_set(self, goal, grasp, held):
        self.seen_grasp.append(tuple(grasp))
        return goal

    def _planned_obstacles(self, grasp_names=()):
        # Mirrors `_plan_arm`'s producer+pad over this fake's stage (the executor harnesses in
        # test_fallback_counter run the real method); `chassis_pad` raises without a deck prim.
        paths = []
        obs = self._arm_obstacles(grasp_names, diagnostic_paths=paths)
        return chassis_pad(obs, paths), paths

    def _commanded_fingers(self):
        return self.fingers

    def _plan_arm(self, goal, held=None, grasp_names=(), **kw):
        self.plan_arm_calls += 1
        self.seen_grasp.append(tuple(grasp_names))
        self.seen_fingers.append(kw.get("fingers"))
        # A real path ENDS AT THE GOAL BRANCH OMPL REACHED, and that -- not the request -- is what
        # arrival is measured against. A fake whose path ends somewhere else tests nothing real.
        if self._end_at_goal:
            self._wps = self._wps[:-1] + [np.asarray(goal, float).ravel()[:8].copy()]
        return self._wps

    def _execute_arm_path(self, waypoints, secs=None, held=None, checkpoint_every=8,
                          grasp_names=(), tag="arm", **kw):
        self.seen_grasp.append(tuple(grasp_names))
        self.checkpoints.append(checkpoint_every)
        self.seen_secs.append(secs)
        self.seen_on_step.append(kw.get("on_step"))
        stop = len(waypoints) if self._block_at is None else self._block_at
        self.commanded.extend(waypoints[:stop])
        # Where a real executor leaves the arm: on its last executed waypoint. Returning zeros
        # instead made every arrival check read "stopped short" and no test could assert ARRIVED.
        return None if self._q_end is None else np.asarray(waypoints[stop - 1], float).copy()


def test_refused_is_not_readable_as_success():
    """Universal over the enum: every outcome but ARRIVED must be falsy AND carry no pose. A bare
    object is truthy, which is how a `return True` ahead of an abort check reached a caller."""
    for out in Outcome:
        if out is Outcome.ARRIVED:
            continue
        r = ArmMove(out, "x")
        assert bool(r) is False, f"{out} reads as success under `if move(...):`"
        assert r.q_end is None, f"{out} carries a pose, which is indistinguishable from success"
        try:
            ArmMove(out, "x", (), np.zeros(8))
        except ValueError:
            pass
        else:
            raise AssertionError(f"{out} was allowed to carry a pose at construction")
    assert bool(ArmMove(Outcome.ARRIVED, "ok", (), np.zeros(8))) is True


def test_move_to_pose_has_no_exemption_parameter_at_all():
    """Not a runtime guard -- the parameter does not exist, so Python itself refuses it and no
    future caller can pass one without changing the signature."""
    sig = inspect.signature(move_to_pose)
    bad = [n for n, p in sig.parameters.items()
           if any(w in n.lower() for w in ("grasp", "exempt", "allow", "contact"))]
    assert not bad, f"move_to_pose grew an exemption parameter: {bad}"
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()), (
        "move_to_pose takes **kwargs, so an exemption can be smuggled past the signature")
    try:
        move_to_pose(_Demo(), np.zeros(8), 1.0, allow_contact_with=("o",))
    except TypeError:
        pass
    else:
        raise AssertionError("move_to_pose accepted an exemption")


def test_move_to_pose_plans_the_hand_it_will_fly():
    """The planner must be handed the COMMANDED hand, not the swept union it falls back to.

    Without this, `first_hit` judges `link_aabb[hand]` -- the palm unioned with
    `FINGER_SWEEP_AABB` -- which at the live handoff pose PENETRATES the deck by 43.9mm where the
    real pads clear it by 11.9mm. A sentinel object is used rather than a value so the test proves
    the same hand travelled, not merely an equal-looking one.

    Mutation: drop `fingers=` from `move_to_pose`'s `_plan_arm` call -- must fail.
    """
    d = _Demo()
    d.fingers = {"finger_a_joint_1_1": -0.55}
    r = move_to_pose(d, _START, 1.0)
    assert d.plan_arm_calls >= 1, "no plan was made, so this proves nothing"
    assert d.seen_fingers and all(f is d.fingers for f in d.seen_fingers), (
        f"the planner was handed a different hand than the robot reports: {d.seen_fingers}")
    assert r.outcome is Outcome.ARRIVED


def test_a_hand_that_is_not_held_is_planned_as_the_SWEPT_box():
    """`hold_fingers=False` means the fingers may move during the motion, so a snapshot of them is
    a lie for part of the path. `None` is the honest answer and selects the conservative swept
    union -- the LARGER box, which is the safe direction.

    Mutation: make `move_to_pose` call `_commanded_fingers()` unconditionally -- must fail. Paired
    with the test above in one file so a stub that always returns `None` cannot satisfy both.
    """
    d = _Demo()
    d.fingers = {"finger_a_joint_1_1": -0.55}
    move_to_pose(d, _START, 1.0, hold_fingers=False)
    assert d.plan_arm_calls >= 1, "no plan was made, so this proves nothing"
    assert all(f is None for f in d.seen_fingers), (
        f"a hand that is free to move was planned as a fixed one: {d.seen_fingers}")


def test_fingers_None_is_conservative_AT_THE_HANDS_FLOWN_and_nowhere_else():
    """Both `None` defaults above rest on the swept box CONTAINING the pads. It does -- but only
    at the hands this robot actually flies, and the distinction is the whole safety argument.

    `_hits` grows each box by MARGIN in ITS OWN rotated axes, so raw box containment does not imply
    verdict containment and has to be measured after the growth, not before. Measured here:

        open hand (symmetric j1 -0.55)         grown pads escape by +0.00 mm
        seat hand (j1 +0.05)                   grown pads escape by +0.00 mm
        4000 hands over the commanded range    up to +32.6 mm  (finger_c_link_3_1)

    The two hands are built here, not read: `-0.55` is the release open's j1 (`GRIPPER_OPEN`),
    `+0.05` the pre-grip keyframe. `tests/fingers_recording.py` holds the RAW recording --
    asymmetric, a -1.04 and b/c -0.493 -- and the blame test further down this file reads it.

    So `fingers=None` is the conservative answer for the motions that exist, and is NOT a general
    upper bound. Anything that starts flying a new finger configuration must re-measure before
    relying on the fallback. An independent review asserted the escape was live at the carry hand;
    re-derived, it is 0.00 mm there -- but the general claim it was attacking was still wrong, so
    the number this pins is the one that was actually measured.
    """
    from morph.arm.model import ArmModel as _AM
    from morph.arm.collision import MARGIN as _MG
    _m = _AM.load(os.path.join(ROOT, "usd", "_arm_model.json"),
                  os.path.join(ROOT, "usd", "_closure_lut.json"))
    q = np.array([0.0, 0.35, 0.40, 0.25, 0.0, 0.0, 0.0, 0.0])
    names = ["finger_%s_joint_%d_1" % (f, j) for f in "abc" for j in (1, 2, 3)]
    names += ["palm_finger_b_joint_1", "palm_finger_c_joint_1"]

    def _grown(local, p, R):
        lo, hi = np.asarray(local[0]) - _MG, np.asarray(local[1]) + _MG
        c = np.array([[x, y, z] for x in (lo[0], hi[0])
                      for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        w = p + c @ R.T
        return w.min(axis=0), w.max(axis=0)

    def _escape(fing):
        parts = list(_m._links(q, fingers=fing))
        hand = [t for t in parts if t[0] == _m.hand_link][0]
        slo, shi = _grown(_m.link_aabb[_m.hand_link], hand[2], hand[3])
        return max(max((slo - _grown(l, p, R)[0]).max(), (_grown(l, p, R)[1] - shi).max())
                   for n, l, p, R in parts if "finger" in n)

    for label, j1 in (("open", -0.55), ("seat", 0.05)):
        e = _escape({n: (j1 if n.endswith("_joint_1_1") else 0.0) for n in names})
        assert e <= 1e-6, f"the {label} hand escapes the swept box by {e * 1000:.2f}mm"

    # ...and the general claim is FALSE, which is why the two above had to be measured rather
    # than assumed. Without this half, the test reads as "the sweep always contains the pads".
    rng = np.random.default_rng(0)
    worst = max(_escape({n: float(rng.uniform(-0.6, 0.75) if n.endswith("_joint_1_1") else
                                  rng.uniform(-0.1, 0.9)) for n in names})
                for _ in range(200))
    assert worst > 0.005, (
        "the pads no longer escape the sweep anywhere in the commanded range -- if that isgenuinely true, "
        "this test's warning is obsolete and the fallback is safe generally; re-measure and say so")


def test_move_to_pose_never_commands_an_exemption():
    """Universal over the delegates: EVERY grasp_names any of them received must be empty."""
    d = _Demo()
    r = move_to_pose(d, _START, 1.0)          # the pose the fake reports, so this ARRIVES
    assert d.seen_grasp, "no delegate was reached, so this proves nothing"
    assert all(g == () for g in d.seen_grasp), (
        f"move_to_pose passed an exemption to a delegate: {d.seen_grasp}")
    assert r.outcome is Outcome.ARRIVED


# An ACM entry: these links may overlap THIS prim's box, for THIS motion. Boxes in the base
# frame, sized off the real model at `_START`, so the line below actually runs into them.

# Clipping the TOP of the boom, which at `_START` reaches z 0.569: the wrist below it (z <= 0.545)
# is inside MARGIN of these boxes but never inside them, so the two cases differ in DEPTH alone.
_BOOM_SHALLOW = np.array([[-0.30, 0.10, 0.562], [0.45, 0.18, 0.65]])   # the boom overlaps ~7mm
_BOOM_DEEP = np.array([[-0.30, 0.10, 0.545], [0.45, 0.18, 0.65]])      # ~24mm


def _links_hitting(box):
    """Which links this box blocks at `_START`. Computed, never hardcoded: a list that drifts out
    of step with the geometry turns these tests into assertions about nothing."""
    return tuple(lk for lk in _MODEL.link_aabb
                 if _MODEL.first_hit(_START, [(box, "chassis")], MARGIN, links=(lk,)))
# The prim the pad actually names. A fixture that spells it differently makes every
# padded call raise, which is `chassis_pad` working -- but it tests the fixture.
from morph.arm.collision import CHASSIS_PAD_PRIM as _DECK_PATH
_UP = np.array([0.0, 0.0, 0.05])      # INTO the boxes above -- the case the depth bound is for
_DOWN = np.array([0.0, 0.0, -0.03])   # out of them, so scope can be tested without depth


def _blocked_demo(box=_BOOM_SHALLOW, **kw):
    kw.setdefault("moves", True)
    d = _Demo(**kw)
    d.obstacles = [(box, "chassis")]
    d.obstacle_paths = [_DECK_PATH]
    return d


def test_the_chassis_pad_lowers_the_margin_for_its_OWN_links_and_no_others():
    """`chassis_pad` is the whole caller-facing surface now: a flag, not a map. It names a fixed
    prim and a fixed link list, so a caller can ask for it or not and cannot shape it.

    Mutation: make `chassis_pad` return `obs` unchanged -- the padded call stops differing.
    """
    from morph.arm.obstacles import chassis_pad
    from morph.arm.collision import CHASSIS_PAD_M, CHASSIS_PAD_LINKS, CHASSIS_PAD_PRIM

    obs = [(_BOOM_SHALLOW, "chassis"), (_BOOM_DEEP, "chassis")]
    paths = [CHASSIS_PAD_PRIM, "/World/robot/base/Other"]
    out = chassis_pad(obs, paths)
    assert len(out[0]) == 4 and out[0][2] == CHASSIS_PAD_M, "the deck box was not padded"
    assert tuple(out[0][3]) == tuple(CHASSIS_PAD_LINKS), "the pad carries the wrong link list"
    assert len(out[1]) == 2, "a box that is NOT the deck was padded -- the pad is not scoped"


def test_a_pad_whose_prim_matches_nothing_RAISES_it_does_not_pass_quietly():
    """A padding entry that matches no box is a silent no-op: the motion plans at full MARGIN and
    reports success, and nothing distinguishes that from a pad that is working. This is exactly how
    `DECK_NEAR` stopped covering the fingers when `8df2c98` split them out, unnoticed for weeks.

    Mutation: return `obs` instead of raising when `hit != 1` -- must fail.
    """
    from morph.arm.obstacles import chassis_pad
    for paths in ([], ["/World/robot/base/Elsewhere"], ["/World/nope", "/World/also_nope"]):
        try:
            chassis_pad([(_BOOM_SHALLOW, "chassis")] * len(paths), paths)
        except KeyError as e:
            assert "silent no-op" in str(e), f"the raise does not say why it matters: {e}"
        else:
            raise AssertionError(f"a pad matching nothing passed quietly for paths={paths}")


def test_the_pad_matches_its_prim_EXACTLY_never_by_prefix():
    """`base` carries the deck, and the battery and the wheels hang under it with their own
    colliders and their own boxes. A prefix match would pad all of them -- and MARGIN is the only
    thing keeping the boom out of its own battery, with PhysX self-collision off.

    Mutation: match with `startswith` -- the battery box gets padded and this fails.
    """
    from morph.arm.obstacles import chassis_pad
    from morph.arm.collision import CHASSIS_PAD_PRIM

    under = CHASSIS_PAD_PRIM.rsplit("/", 1)[0]
    paths = [CHASSIS_PAD_PRIM, under + "/LifePo4_12V50Ah/Box", under + "/wheel/Sphere"]
    out = chassis_pad([(_BOOM_SHALLOW, "chassis")] * 3, paths)
    assert len(out[0]) == 4, "the deck itself was not padded"
    assert [len(o) for o in out[1:]] == [2, 2], (
        "a box UNDER the deck's parent inherited the pad -- the match is a prefix, not exact")


def test_neither_primitive_takes_the_pad_or_an_allowance():
    """The deck pad is applied by `planned_obstacles` for every planned motion, never asked for by
    a caller; and the deleted allowance must stay deleted: a caller-supplied ACM entry was scoped
    to one motion but named any prim and any links, which is the thing A4 exists to refuse."""
    for fn in (move_linear, move_to_pose):
        gone = [n for n in inspect.signature(fn).parameters
                if n in ("chassis_pad", "allow_contact_pairs", "allow_margin_m")]
        assert not gone, f"{fn.__name__} still carries a deleted caller knob: {gone}"


def test_move_to_pose_never_sees_an_annotated_obstacle():
    """Universal over the delegates, as the exemption test above is: `move_to_pose` carries no
    allowance of any kind, and that is structural -- it has no parameter to carry one."""
    sig = inspect.signature(move_to_pose).parameters
    assert not [p for p in sig if "allow" in p], (
        f"move_to_pose grew an allowance parameter: {list(sig)}")
    d = _blocked_demo()
    move_to_pose(d, _START, 1.0)
    assert d.seen_grasp, "no delegate was reached, so this proves nothing"
    for lst in d.seen_obs:
        assert all(len(o) == 2 for o in lst), f"move_to_pose received an annotated obstacle: {lst}"


def test_checkpoint_replanning_stays_disabled_at_every_executor_call_in_this_module():
    """The checkpoint rebuilds the obstacle set from `grasp_names` ALONE (`execute_path`), so
    it cannot see a per-motion allowance -- it would re-validate the exempted waypoint against the
    unexempted set and abort the motion the check accepted. Today the only thing preventing that is
    `checkpoint_every = len(wps) + 1`. Pin it universally over the call sites, not on one path."""
    src = open(os.path.join(ROOT, "morph", "arm", "api.py"), encoding="utf-8").read()
    calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "execute_path"]
    assert calls, "no executor call in api.py -- this test guards nothing"
    for c in calls:
        kw = {k.arg for k in c.keywords}
        assert "checkpoint_every" in kw or "regoal" in kw, (
            f"the executor call at line {c.lineno} takes the DEFAULT checkpoint_every; a straight "
            f"line that re-plans becomes a different line")


def test_no_allowance_rides_along_with_the_exemption_into_the_executor():
    """`execute_path` passes `grasp_names` on to an abort-recovery `_plan_arm` -- the PLANNED
    tier. An allowance threaded alongside it would escape the one motion it was scoped to, so the
    allowance must never reach the executor at all: it lives in the obstacle list, which the
    executor rebuilds for itself."""
    src = open(os.path.join(ROOT, "morph", "arm", "api.py"), encoding="utf-8").read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "move_linear")
    for c in [n for n in ast.walk(fn) if isinstance(n, ast.Call)
              and getattr(n.func, "id", "") == "execute_path"]:
        passed = {k.arg for k in c.keywords}
        assert not [k for k in passed if k and "allow" in k], (
            f"move_linear hands the executor {sorted(passed)}; an allowance among them reaches the "
            f"abort-recovery planner")




def test_move_linear_judges_the_hand_with_the_commanded_fingers_too():
    """UNIVERSAL over the calls `move_linear` makes. The swept box's floor sits 39mm below where the
    fingers actually reach, so a line judged on the sweep is permanently more pessimistic than one
    judged on the commanded hand -- and its BOUND then disagrees with the check it bounds."""
    src = open(os.path.join(ROOT, "morph", "arm", "api.py"), encoding="utf-8").read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "move_linear")
    assert "fingers" in {a.arg for a in fn.args.kwonlyargs}, (
        "move_linear takes no `fingers`: it cannot judge the commanded hand")
    # `_bound_broken` is deliberately absent: the pad is enforced INSIDE the check, so there is no
    # second pass to keep in step with the first. What remains must still judge the same hand.
    for name in ("plan_cartesian", "blame"):
        calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
                 and (getattr(c.func, "id", "") == name or getattr(c.func, "attr", "") == name)]
        assert calls, f"move_linear no longer calls {name}; this test guards nothing"
        for c in calls:
            passed = {k.arg for k in c.keywords} | {
                a.id for a in c.args if isinstance(a, ast.Name)}
            assert "fingers" in passed, (
                f"move_linear's {name} call at line {c.lineno} drops the finger state")


def test_move_linear_never_replans():
    """Three ways it could replan, all closed: no planner call, the executor's checkpointing
    disabled past the end of the path, and no retry after a block."""
    d = _Demo()
    move_linear(d, np.array([0.0, 0.0, 0.05]), "base")
    assert d.plan_arm_calls == 0, (
        f"move_linear called the sampling planner {d.plan_arm_calls} time(s) -- a straight line "
        f"that replans becomes a different line")
    assert d.checkpoints, "the executor was never reached"
    assert len(d._wps) > 8, "fixture: the path must exceed the executor's default checkpoint"
    for ce in d.checkpoints:
        assert ce > len(d.commanded), (
            f"checkpoint_every={ce} for a {len(d.commanded)}-waypoint path: the executor replans a "
            f"blocked checkpoint by design, so anything <= the path length re-enables it")


def test_the_exemption_does_not_survive_one_call():
    """The exact sequence every delegate received across three calls -- not "the flag was cleared"."""
    d = _Demo()
    move_linear(d, np.array([0.0, 0.0, 0.05]), "base", allow_contact_with=("pickup_obj_0",))
    move_to_pose(d, np.zeros(8), 1.0)
    move_linear(d, np.array([0.0, 0.0, 0.05]), "base")
    assert d.seen_grasp == [("pickup_obj_0",), ("pickup_obj_0",), (), (), (), (), ()], d.seen_grasp


def test_a_bare_string_exemption_is_refused():
    """A string is iterable: "pickup_obj_0" would become 13 one-character prim names that exempt
    nothing while looking like they worked."""
    assert _exemption("pickup_obj_0") is None
    assert _exemption(("pickup_obj_0",)) == ("pickup_obj_0",)
    assert _exemption(None) == ()
    r = move_linear(_Demo(), np.zeros(3), "base", allow_contact_with="pickup_obj_0")
    assert r.outcome is Outcome.NO_PLAN and not r


def test_every_tag_the_api_can_report_is_registered():
    """`_new_tags` reports whatever the delegates counted, so any tag literal spelled in api.py
    must exist in the acceptance gate's registry or it is invisible to it."""
    import verify_place
    src = open(os.path.join(ROOT, "morph", "arm", "api.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    # A tag is a string that reaches `_fallback`, or one this module TESTS for in its own tag
    # diff -- not every hyphenated literal, or the reason text counts as a tag.
    lits = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "_fallback":
            lits |= {a.value for a in n.args if isinstance(a, ast.Constant)}
        if isinstance(n, ast.Compare) and isinstance(n.left, ast.Constant) \
                and isinstance(n.left.value, str) and any(isinstance(o, ast.In) for o in n.ops):
            lits.add(n.left.value)
    unknown = {t for t in lits if t not in verify_place.FALLBACK_TAGS}
    assert not unknown, f"api.py names fallback tags the registry does not know: {sorted(unknown)}"
    assert "payload-lost" in lits, (
        "the tag the outcome mapping keys on is no longer detected -- retarget this test")





def test_api_is_the_only_public_surface():
    """Shape test, deliberately: nothing outside morph/arm/ may import anything under morph.arm
    except `api`, or there are two places to look for the contract."""
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.split(), "git ls-files gave nothing: the test would pass over an empty universe"
    bad, unparsed = [], []
    for rel in out.stdout.split():
        if rel.startswith("morph/arm/"):
            continue
        # tests exercise internals; nothing else is excused from the parse rule below
        if rel.startswith("tests/"):
            continue
        if rel.startswith("tools/") or rel in ("arm_plan.py", "sweep.py"):
            # measurement tools and the OMPL child are arm-side
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
            if isinstance(n, ast.ImportFrom) and n.module == "morph.arm":
                mods = [f"morph.arm.{a.name}" for a in n.names if a.name != "api"]
            if isinstance(n, ast.ImportFrom) and n.module == "morph" and any(a.name == "arm" for a in n.names):
                mods = ["morph.arm.<package>"]
            if isinstance(n, ast.Import) and any(a.name == "morph.arm" for a in n.names):
                mods = ["morph.arm.<package>"]
            for mod in mods:
                if mod.startswith("morph.arm.") and mod != "morph.arm.api":
                    bad.append(f"{rel}:{n.lineno} imports {mod}")
            if isinstance(n, ast.ImportFrom) and n.module == "morph.arm.api":
                for a in n.names:
                    if a.name not in _api_mod.__all__:
                        bad.append(f"{rel}:{n.lineno} imports {a.name}, not in api.__all__")
    assert not unparsed, (
        "shipped source the 3.10 venv cannot parse, so this test would skip it in silence: "
        + "; ".join(unparsed))
    assert not bad, "private modules imported from outside the package: " + "; ".join(bad)


def test_new_tags_reports_only_what_this_call_counted():
    d = _Demo()
    d._plan_fallbacks = {"reach-abort": 1}
    before = dict(d._plan_fallbacks)
    d._plan_fallbacks["reach-abort"] = 2
    d._plan_fallbacks["retreat-raised"] = 1
    assert sorted(_new_tags(d, before)) == ["reach-abort", "retreat-raised"]


def test_the_reach_maps_every_outcome_to_the_tri_state_its_callers_read():
    """`_plan_approach` is now `move_to_pose` plus goal derivation, but its callers still read a
    tri-state: True executes, False replays the BAKE, None ENDS the pick. The mapping is the whole
    risk of the migration -- sending an abort to False replays a bake that ends at the goal the
    checkpoint just rejected."""
    import morph.arm.api as _api
    src = open(os.path.join(ROOT, "morph", "pick", "approach.py"), encoding="utf-8").read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_plan_approach")
    body = ast.unparse(fn)

    assert "move_to_pose(" in body, (
        "_plan_approach no longer routes through the primitive -- the duplicated plan/execute "
        "chain is back")
    for gone in ("self._plan_arm(", "self._execute_arm_path("):
        assert gone not in body, (
            f"_plan_approach still calls {gone} directly, so there are two routes to the same "
            f"motion and only one of them carries the primitive's contract")

    # Names must appear in a BRANCH CONDITION, not merely somewhere in the function: a dead
    # `_unused = Outcome.PAYLOAD_LOST` satisfies "is mentioned" while the routing is deleted.
    tested = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.If):
            tested |= {a.attr for a in ast.walk(n.test) if isinstance(a, ast.Attribute)
                       and getattr(a.value, "id", "") == "Outcome"}
    # Universal over the ENUM, not over three names typed here: a sixth member added tomorrow must
    # fail this test rather than fall through to `return True`.
    from morph.arm.api import Outcome as _O
    # NO_ARRIVAL is NOT exempt: exempting it let `if False:` delete the counter with the whole suite
    # green. Only ARRIVED, the fall-through, may be absent from a branch condition.
    must = {o.name for o in _O} - {"ARRIVED"}
    missing = must - tested
    assert not missing, (
        f"{sorted(missing)} is never tested in a branch condition of _plan_approach, so it falls "
        f"through to `return True` and the caller proceeds as though the arm arrived")

    for name in ("REFUSED", "PAYLOAD_LOST"):
        aborts = [n for n in ast.walk(fn) if isinstance(n, ast.If)
                  and any(getattr(a, "attr", "") == name for a in ast.walk(n.test))]
        assert aborts, f"{name} is not in any branch condition"
        assert any(isinstance(st, ast.Return) for st in aborts[0].body), (
            f"the {name} branch does not return, so an aborted reach continues into the bake")


def test_the_reach_hands_the_executor_every_option_it_depends_on():
    """Not a substring search over source -- a comment reading `# passes on_step=on_step` satisfies
    that. Drive the REAL `_plan_approach` through a fake and read what `execute_path` was
    actually given. Each of these was dropped in a mutation and the whole suite stayed green:

      on_step       `pin_bystanders` stops running, so bystander objects -- colliders off for the
                    reach -- drift for the entire approach instead of being re-pinned each step
      regoal        the slip-retarget refusal is disabled; a base slip replans to a stale world
                    target instead of refusing
      held          the reach plans and collision-checks without the carried object's geometry
      hold_fingers  the fingers are no longer frozen for the motion
    """
    fn, _ag = _exec_plan_approach()
    seen = {}

    class _D:
        _grasp_cand = None
        _approach_goal = _ag
        idx = {n: i for i, n in enumerate(ArmModel.Q8)}
        _plan_fallbacks = {}
        def _traj_q8(self, tr, i):
            return np.zeros(8)
        def _object_relative_goal(self, baked, excl):
            return baked
        def _goal_set(self, g, grasp, held):
            return g
        def _commanded_fingers(self):
            return None
        def _plan_arm(self, goal, held=None, **kw):
            return [np.zeros(8), np.zeros(8)]
        def _arm_q8(self):
            return np.zeros(8)
        def _execute_arm_path(self, wps, secs=None, **kw):
            seen.update(kw)
            return np.zeros(8)
        def _fallback(self, t):
            self._plan_fallbacks[t] = self._plan_fallbacks.get(t, 0) + 1

    marker = object()
    d = _D()
    fn(d, {"idx": [], "frames": []}, 0, 2.5, exclude_obj=0, on_step=marker,
       hold_fingers=False, tag="reach", held=marker)

    assert seen.get("on_step") is marker, (
        f"on_step did not reach the executor (got {seen.get('on_step')!r}); pin_bystanders stops "
        f"running and bystander objects drift for the whole approach")
    assert seen.get("held") is marker, "held did not reach the executor: the reach checks without "\
                                       "the carried object's geometry"
    assert seen.get("hold_fingers") is False, (
        f"hold_fingers did not reach the executor (got {seen.get('hold_fingers')!r})")
    assert callable(seen.get("regoal")), (
        "regoal did not reach the executor, so a base slip replans to a stale world target "
        "instead of refusing")


def _exec_plan_approach():
    """`_plan_approach` AND the goal derivation it delegates to, as plain functions.

    Both, because the split is only a split: a fake carrying a stubbed `_approach_goal` would
    test the caller against a goal no production path produces.
    """
    import morph.arm.api as _api
    glb = {"np": np, "os": os, "ARM_PLANNER": True, "print": lambda *a, **k: None,
           "move_to_pose": _api.move_to_pose, "Outcome": _api.Outcome}
    src = open(os.path.join(ROOT, "morph", "pick", "approach.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    want = ("_plan_approach", "_approach_goal")
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in want]
    assert {n.name for n in fns} == set(want), f"missing {set(want) - {n.name for n in fns}}"
    mod = ast.Module(body=fns, type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, "<plan_approach>", "exec"), glb)
    return glb["_plan_approach"], glb["_approach_goal"]


def test_a_no_plan_refuses_and_a_bad_target_does_not_hide_inside_it():
    """`return True` on a no-plan survived the whole suite once: nothing executed, the arm never
    moved, and the caller played the descent from the un-moved pose. With the bake gone a no-plan
    must END the attempt -- there is no degraded path left to take."""
    fn, _ag = _exec_plan_approach()

    class _D:
        _grasp_cand = None
        _approach_goal = _ag
        idx = {n: i for i, n in enumerate(ArmModel.Q8)}
        def __init__(self, goal):
            self._plan_fallbacks = {}
            self._goal = goal
        def _traj_q8(self, tr, i):
            return self._goal
        def _object_relative_goal(self, baked, excl):
            return baked
        def _goal_set(self, g, grasp, held):
            return g
        def _commanded_fingers(self):
            return None
        def _plan_arm(self, goal, held=None, **kw):
            return None
        def _arm_q8(self):
            return np.zeros(8)
        def _execute_arm_path(self, *a, **k):
            raise AssertionError("nothing may execute when there is no plan")
        def _fallback(self, t):
            self._plan_fallbacks[t] = self._plan_fallbacks.get(t, 0) + 1

    d = _D(np.zeros(8))
    out = fn(d, {"idx": [], "frames": []}, 0, 2.5, exclude_obj=0, tag="reach")
    assert out is None, (
        f"a no-plan returned {out!r}. There is no bake to fall back to any more: a planned motion "
        f"that cannot be planned REFUSES, and None is the caller's cue to end the attempt. True "
        f"would make it play the baked descent from a pose the arm never left")
    assert d._plan_fallbacks == {"no-plan": 1}, d._plan_fallbacks

    # ...and a target that was never a joint vector is a DIFFERENT failure, counted separately.
    d2 = _D(np.full(8, np.nan))
    out2 = fn(d2, {"idx": [], "frames": []}, 0, 2.5, exclude_obj=0, tag="reach")
    assert out2 is None and d2._plan_fallbacks == {"bad-target": 1}, (
        f"{out2!r} {d2._plan_fallbacks} -- a goal that never parsed is reported as 'the planner "
        f"found no route', sending whoever reads it to the obstacle set")


def test_move_to_pose_maps_each_executor_result_to_the_right_outcome():
    """The outcome mapping had no behavioural test at all: deleting the arrival comparison, never
    distinguishing PAYLOAD_LOST, and reporting a bad target as ARRIVED all survived the suite."""
    d = _Demo()
    assert move_to_pose(d, _START, 1.0).outcome is Outcome.ARRIVED

    short = _Demo(end_at_goal=False)
    short._wps = [np.zeros(8), np.full(8, 0.9)]
    assert move_to_pose(short, _START, 1.0).outcome is Outcome.NO_ARRIVAL, (
        "a path ending far from the goal reports ARRIVED: the arrival comparison is gone")

    refused = _Demo(q_end=None)
    assert move_to_pose(refused, _START, 1.0).outcome is Outcome.REFUSED

    lost = _Demo(q_end=None)
    lost._plan_fallbacks = {}
    _real = lost._execute_arm_path
    def _exec(*a, **k):
        lost._plan_fallbacks["payload-lost"] = 1
        return _real(*a, **k)
    lost._execute_arm_path = _exec
    assert move_to_pose(lost, _START, 1.0).outcome is Outcome.PAYLOAD_LOST, (
        "a stop that broke the friction grasp is reported as an ordinary refusal")

    nan = move_to_pose(_Demo(), np.full(8, np.nan), 1.0)
    assert nan.outcome is Outcome.NO_PLAN and nan.q_end is None and not nan, (
        "a non-finite target is not reported as a failure to plan")


def test_arrival_is_measured_against_the_goal_the_executor_actually_reached():
    """A checkpoint replan REBINDS the executor's local `waypoints` to a spliced list; the caller's
    list object is never mutated. On the `regoal()` path that spliced tail ends somewhere
    DELIBERATELY different, because the base slipped and the world goal was re-derived.

    Measuring against the caller's list then reports a perfect slip-recovery as short -- and
    `SLIP_RETARGET_M` (5mm) EQUALS `Q8_TOL_M` (5mm), so every slip big enough to TRIGGER a retarget
    is by construction at or past the arrival band. That made a successful recovery count
    `reach-no-arrival`, which is SAFETY and zero-tolerance: the run fails on a good recovery."""
    RETARGET = np.full(8, 0.25)                # where the executor really went, post-slip

    class _Slip(_Demo):
        def _plan_arm(self, goal, held=None, grasp_names=(), **kw):
            self.plan_arm_calls += 1
            self.seen_grasp.append(tuple(grasp_names))
            return [np.zeros(8), np.full(8, 0.10)]      # the caller's list: ends at 0.10
        def _execute_arm_path(self, waypoints, secs=None, **kw):
            self.seen_grasp.append(tuple(kw.get("grasp_names", ())))
            self.checkpoints.append(kw.get("checkpoint_every", 8))
            # what the real executor does: rebind a LOCAL, then report where it ended
            waypoints = list(waypoints[:1]) + [RETARGET]
            self._exec_goal8 = np.asarray(waypoints[-1], float).copy()
            return RETARGET.copy()
        def _arm_q8(self):
            # the arm is exactly on the re-derived goal
            return RETARGET.copy()

    d = _Slip()
    r = move_to_pose(d, np.zeros(8), 1.0)
    assert r.outcome is Outcome.ARRIVED, (
        f"the arm ended EXACTLY on the re-derived goal and this reports {r.outcome.value} "
        f"({r.reason}). The caller then counts reach-no-arrival, which is SAFETY and fails the "
        f"whole run -- a successful slip recovery graded as a safety degrade")
    assert bool(r) is True and r.q_end is not None


def test_the_executor_publishes_the_goal_it_drove_to():
    """The fix depends on the executor reporting its own target; if it stops, `move_to_pose`
    silently falls back to the caller's list and the bug returns."""
    src = open(os.path.join(ROOT, "morph", "arm", "execute.py"), encoding="utf-8").read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "execute_path")
    sets = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
            and any(getattr(t, "attr", "") == "_exec_goal8" for t in n.targets)]
    assert sets, "execute_path no longer publishes the goal it drove to"
    # ...and it must be assigned from the (possibly spliced) waypoint list, not from a parameter
    src_set = ast.unparse(sets[0])
    assert "waypoints[-1]" in src_set, (
        f"the published goal is {src_set!r}: it must come from the executor's OWN final waypoint, "
        f"which is the spliced tail after a replan")


# A box the boom is clear of at `_START` and drives into over `_UP`: a line blocked at q_start
# builds no prefix, which is not the branch these tests are about.
_BOOM_ABOVE_GAP = np.array([[-0.30, 0.10, 0.600], [0.45, 0.18, 0.70]])


def test_a_blocked_line_executes_its_VALIDATED_PREFIX_and_stops_there():
    """A blocked line halts on the last cleared pose -- the end of the validated prefix, not
    `q_start`."""
    assert _MODEL.first_hit(_START, [(_BOOM_ABOVE_GAP, "chassis")], MARGIN) is None, (
        "fixture: the box already blocks at _START, so no prefix would be built and this test "
        "would pass on the wrong branch")

    d = _blocked_demo(box=_BOOM_ABOVE_GAP)
    r = move_linear(d, _UP, "base")
    assert r.outcome is Outcome.REFUSED, f"a blocked line was not refused: {r.outcome} {r.reason}"
    assert d.commanded, (
        "the line blocked partway and the arm was commanded NOTHING -- the validated prefix was "
        "discarded, so the arm sits at q_start and reports it stopped at the last cleared pose")
    assert d.checkpoints and d.checkpoints[-1] > len(d.commanded), (
        f"the prefix ran with checkpoint_every={d.checkpoints}: checkpoint replanning is enabled, "
        f"so the straight line can silently become a different line (A3)")


def test_the_refusal_of_a_blocked_line_NAMES_the_obstacle():
    """`plan_cartesian` reports an index and no geometry. The refusal must name the link AND the
    prim."""
    d = _blocked_demo(box=_BOOM_ABOVE_GAP)
    d.obstacle_paths = ["/World/some/specific/prim"]
    r = move_linear(d, _UP, "base")
    assert r.outcome is Outcome.REFUSED, f"the fixture no longer blocks: {r.outcome}"
    assert "/World/some/specific/prim" in r.reason, (
        f"the refusal does not name the obstacle it blames: {r.reason}")
    assert "Arm_Left_1" in r.reason, f"the refusal does not name the link: {r.reason}"


def test_the_refusal_names_a_pair_the_COMMANDED_hand_actually_hits():
    """`blame` must accuse a pair the refusing predicate agrees with. Judged on the swept finger
    union it can name a link and a prim the planner never refused for -- the decoy below sits where
    only the sweep reaches. The kwarg guard above cannot catch this: it only checks that `fingers=`
    is written at the call site."""
    from tests.fingers_recording import FINGERS_RECORDING
    fing = {k: float(v) for k, v in FINGERS_RECORDING.items()}
    p0, R0 = _MODEL.fk(_START)[_MODEL.hand_link]

    def _hand_box(lo, hi):
        return ArmModel._xform_aabb(np.vstack([lo, hi]), p0, R0)

    decoy = _hand_box([-0.01, -0.01, 0.120], [0.01, 0.01, 0.140])
    blocker = _hand_box([-0.02, -0.02, 0.010], [0.02, 0.02, 0.040])
    assert _MODEL.first_hit(_START, [(decoy, "chassis")], MARGIN) is not None, \
        "fixture: the decoy does not reach the swept box, so nothing could be misnamed"
    assert _MODEL.first_hit(_START, [(decoy, "chassis")], MARGIN, fingers=fing) is None, \
        "fixture: the commanded hand reaches the decoy too, so it is not swept-only"

    d = _Demo(moves=True)
    d.obstacles = [(decoy, "chassis"), (blocker, "chassis")]
    d.obstacle_paths = ["/World/decoy_swept_only", "/World/real_blocker"]
    r = move_linear(d, _DOWN, "base", fingers=fing)
    assert r.outcome is not Outcome.ARRIVED, "the fixture did not block at all"
    named_prim = r.reason.rsplit("inside the margin of ", 1)[-1].strip()
    named_link = r.reason.rsplit("-- ", 1)[-1].split(" inside", 1)[0].strip()
    box = {"/World/decoy_swept_only": decoy, "/World/real_blocker": blocker}[named_prim]
    assert _MODEL.first_hit(_START, [(box, "chassis")], MARGIN, links=(named_link,),
                            fingers=fing) is not None, (
        f"the refusal accused {named_link} x {named_prim}, which the COMMANDED hand clears")


def test_ik_restarts_is_the_callers_choice_and_defaults_to_OFF():
    """A restart solves from a RANDOM seed, so the waypoint it returns can sit on a different IK
    branch from the one before it, and nothing checks between waypoints. Measured from the shipped
    grasp pose: every line restarts rescued moved a collision box 346-395mm between two waypoints
    5mm apart, while lines that solve from their seed are identical either way. So the caller that
    wants the availability names it and owns the hazard; the default cannot hand it to anyone."""
    import morph.arm.cartesian as _c
    seen = []
    real = _c.plan_cartesian
    _c.plan_cartesian = lambda *a, **kw: (seen.append(kw.get("ik_restarts")), real(*a, **kw))[1]
    try:
        move_linear(_Demo(moves=True), _DOWN, "base")
        move_linear(_Demo(moves=True), _DOWN, "base", ik_restarts=4)
    finally:
        _c.plan_cartesian = real
    assert seen == [0, 4], f"move_linear passed ik_restarts={seen}, expected [0, 4]"


_IH1 = ArmModel.Q8.index("ColumnLeftBearingJoint_1")
_IH2 = ArmModel.Q8.index("ColumnRightBearingJoint_1")


def test_a_pure_z_ask_is_flown_as_a_column_ramp_and_invents_no_other_motion():
    """A3's method choice. Through `plan_cartesian` the same ask spends 8 DOF on a 6-DOF task and
    tilts the boom and rotates the wrist; the closed form moves the two columns and nothing else.
    `secs` is what the caller needs and what sets the step count -- without it there is nothing to
    be exact about and the Cartesian planner retimes instead. It must also REACH the executor,
    which reads it as a floor: dropped, this 1.2 s carry ramp flies in its free 0.43 s. And the
    checkpoint interval must stay past the path, or the ramp replans mid-line."""
    d = _Demo(moves=True)
    r = move_linear(d, np.array([0.0, 0.0, -0.03]), "base", secs=1.2)
    assert r.outcome is Outcome.ARRIVED, f"{r.outcome} {r.reason}"
    assert d.commanded, "the ramp commanded nothing"
    moved = [ArmModel.Q8[i] for i in range(8)
             if abs(np.asarray(d.commanded[-1], float)[i] - _START[i]) > 1e-12]
    assert moved == [ArmModel.Q8[_IH1], ArmModel.Q8[_IH2]], f"the ramp also moved {moved}"
    # The executor treats waypoints[0] as the START pose and commands waypoints[1:], so the list must
    # be the current pose PLUS the eased sequence or every commanded pose shifts by one.
    n = int(1.2 * 240)
    assert len(d.commanded) == n + 1, (
        f"the executor was handed {len(d.commanded)} waypoints; it needs the start pose plus "
        f"{n} eased poses")
    assert np.allclose(np.asarray(d.commanded[0], float), _START), (
        "the ramp did not hand the executor the pose the arm is actually in as waypoint 0")
    # Mutations: drop the `secs` (the carry runs 2.3x faster, every bound still satisfied);
    # hand a checkpoint interval inside the path (240 eased poses become 240 rest-to-rest legs).
    assert d.seen_secs == [1.2], (
        f"the executor was handed secs={d.seen_secs}, not the 1.2 s the caller asked for")
    assert d.checkpoints[-1] > n + 1, (
        f"checkpoint_every={d.checkpoints[-1]} sits inside a {n + 1}-waypoint ramp, so the line "
        f"can replan into a different line")


def test_an_ask_the_columns_cannot_DELIVER_is_refused_rather_than_clipped():
    """`insert.py` clips each column at COLUMN_MAX independently. When only one clips, `dh`
    changes, the equal-column property is gone and the hand TILTS -- with the object in it. The
    primitive must say it cannot deliver the ask."""
    d = _Demo(moves=True)
    d._arm_bounds = lambda: (np.zeros(8), np.array([10.0, _START[1] + 0.01, 10.0, 10.0,
                                                    10.0, 10.0, 10.0, 10.0]))
    r = move_linear(d, np.array([0.0, 0.0, 0.05]), "base", secs=1.0)
    assert r.outcome is Outcome.NO_PLAN and "cannot deliver" in r.reason, (
        f"a raise that clips ONE column was not refused: {r.outcome} {r.reason}")
    assert not d.commanded, "a refused ramp still commanded motion"


def test_an_ask_with_no_closed_form_still_gets_its_line():
    """The method choice is not a gate: a lateral ask has no column ramp and must fall through to
    the Cartesian planner, not be refused."""
    d = _Demo(moves=True)
    r = move_linear(d, np.array([0.03, 0.0, 0.0]), "base", secs=1.0)
    assert r.outcome is not Outcome.NO_PLAN or "cannot deliver" not in r.reason, (
        f"a lateral ask was refused by the ramp gate: {r.reason}")
    assert d.commanded, "a lateral ask commanded nothing at all"


def test_a_ramp_blocked_partway_halts_on_the_last_cleared_pose():
    """Same contract as the Cartesian branch, and the same one `_ramp` implements today: the
    cleared prefix is flown, the blocked pose is not, and the refusal names what blocked it."""
    d = _blocked_demo(box=_BOOM_ABOVE_GAP)
    d.obstacle_paths = ["/World/some/specific/prim"]
    r = move_linear(d, np.array([0.0, 0.0, 0.05]), "base", secs=1.0)
    assert r.outcome is Outcome.REFUSED, f"{r.outcome} {r.reason}"
    assert d.commanded, "a ramp blocked partway flew nothing at all"
    assert "/World/some/specific/prim" in r.reason and "Arm_Left_1" in r.reason, r.reason



def test_on_step_reaches_the_executor_on_EVERY_branch_that_drives():
    """`move_linear` accepts `on_step` and has three paths to the executor. A path that drops it
    silently disarms a caller's watch: both insert sites pass a contact watch and fall through to
    the Cartesian branch whenever the delta has no closed form."""
    marker = lambda *a, **k: None                                          # noqa: E731
    cases = {}

    d = _Demo(moves=True)                                                  # closed-form ramp
    move_linear(d, _UP, "base", secs=1.0, on_step=marker)
    cases["ramp"] = d

    d = _Demo(moves=True)                                                  # full Cartesian line
    move_linear(d, _UP, "base", on_step=marker)
    cases["cartesian"] = d

    d = _blocked_demo(box=_BOOM_ABOVE_GAP)                                 # validated prefix
    move_linear(d, _UP, "base", on_step=marker)
    cases["prefix"] = d

    for name, dm in cases.items():
        assert dm.seen_on_step, f"the {name} branch never reached the executor -- fixture is wrong"
        assert all(s is marker for s in dm.seen_on_step), (
            f"the {name} branch dropped on_step (executor got {dm.seen_on_step!r}): a caller's "
            f"watch is silently disarmed, with no error to say so")


def _world_box_on_the_arm(d):
    """A "world" box 5m away in BASE coordinates and exactly on the arm in WORLD. Only a check
    that carries links out to world can see it -- which is the whole point."""
    d.obstacles = [(_BOOM_SHALLOW + np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]), "world")]
    d.obstacle_paths = ["/World/shelf"]
    d._arm_base_world = lambda: (np.array([5.0, 0.0, 0.0]), np.eye(3))
    return d


def test_move_linear_judges_world_obstacles_at_the_robots_REAL_pose():
    """`base` carries links out to world for "world" boxes; `move_linear`'s Cartesian branch
    passed `base=None` for `frame="base"`, so it judged them against BASE-frame poses -- right only
    on the origin, and the robot docks 3.7-7.7m off it. The box must be "world"-tagged: "chassis"
    boxes are frame-invariant, which is why the rest of this file could not see it."""
    ctl = _world_box_on_the_arm(_Demo(moves=True))
    assert move_linear(ctl, _UP, "world").outcome is Outcome.REFUSED, (
        "fixture: the box does not block even in the frame that is judged correctly, so a refusal "
        "below would prove nothing")

    d = _world_box_on_the_arm(_Demo(moves=True))
    r = move_linear(d, _UP, "base")          # no `secs`: the ramp branch is skipped by design
    assert r.outcome is Outcome.REFUSED, (
        f"a world obstacle sitting on the arm was driven through: {r.outcome} {r.reason}. The "
        f"base-frame line judged it 5m away because it checked in the wrong frame")


def test_a_base_frame_line_still_travels_in_the_BASE_frame_under_yaw():
    """The base is now always passed, so the line is converted instead of the frame being withheld.
    If that conversion were dropped or inverted, a base-frame ask would run along a WORLD axis --
    the defect `13cd9ad` measured at 141mm on a 100mm move. Pinned under a real yaw, because at
    identity a missing rotation is invisible."""
    th = 0.9
    R = np.array([[np.cos(th), -np.sin(th), 0.0],
                  [np.sin(th), np.cos(th), 0.0],
                  [0.0, 0.0, 1.0]])
    d = _Demo(moves=True)
    d._arm_base_world = lambda: (np.array([3.0, -2.0, 0.0]), R)

    m = d._arm_model()
    link = m.hand_link
    p_before = m.fk(_START)[link][0].copy()
    # LATERAL, not the pure z the ramps use: a z-delta is invariant under a z-yaw, so a pure-z
    # fixture passes with the conversion deleted -- it cannot fail.
    delta = np.array([0.03, 0.02, 0.0])
    r = move_linear(d, delta, "base")
    assert r.outcome is Outcome.ARRIVED, f"the fixture no longer plans: {r.outcome} {r.reason}"

    # fk is base-frame, so the hand's own displacement IS the base-frame travel.
    moved = m.fk(np.asarray(d.commanded[-1], float))[link][0] - p_before
    assert np.linalg.norm(moved - delta) < 1e-3, (
        f"a base-frame delta {delta.tolist()} travelled {moved.tolist()} in the base frame: the "
        f"line was run in the wrong frame, which is a rotated direction, not a scaled one")


def test_an_ARRIVED_line_means_the_hand_reached_the_REQUESTED_pose():
    """`move_linear` judges arrival against `wps[-1]`, the path's OWN last waypoint, never against
    the Cartesian pose the caller asked for. That is honest ONLY because `plan_cartesian` cannot
    emit a waypoint off the requested line: `ArmModel.ik` returns a solution or None -- "never a
    near miss" -- and `plan_cartesian` keeps `sol[0]` while DISCARDING `err_p`/`err_r`. Relax that
    contract to return a best effort and the endpoint drifts off the ask while `_arrival` confirms
    the robot faithfully reached it.

    Nothing pinned this. The sibling guarantee -- that `column_ramp` refuses an undeliverable dz
    instead of clipping it -- is already covered by
    `test_an_ask_the_columns_cannot_DELIVER_is_refused_rather_than_clipped`.

    Run through the REAL model and the REAL planner: a structural check would pass with the
    guarantee sitting in a dead branch.
    """
    from morph.arm.cartesian import plan_cartesian

    q0 = np.array(_START, float)
    hand = _MODEL.hand_link
    p0 = np.asarray(_MODEL.fk(q0)[hand][0], float)

    delta = np.array([0.0, 0.0, -0.04], float)
    wps, why = plan_cartesian(_MODEL, q0, delta=delta, link=hand, bounds=_MODEL.bounds())
    assert wps, f"the fixture line did not plan ({why}) -- this test would pin nothing"
    got = np.asarray(_MODEL.fk(np.asarray(wps[-1], float))[hand][0], float)
    err = float(np.linalg.norm(got - (p0 + delta)))
    assert err <= 1e-4, (
        f"plan_cartesian's last waypoint sits {err * 1000:.3f}mm off the requested target -- "
        f"`move_linear` measures arrival against THAT waypoint, so it would report ARRIVED here")

    # The contract itself, at a pose far outside the envelope: a near-miss return is the only way
    # this comes back non-None.
    assert _MODEL.ik(p0 + np.array([50.0, 50.0, 50.0]), seed=q0,
                     bounds=_MODEL.bounds()) is None, (
        "ArmModel.ik returned something for an unreachable pose -- its contract is 'never a near "
        "miss', and plan_cartesian discards err_p/err_r on the strength of it")


def test_a_non_finite_start_pose_is_refused_before_anything_is_checked_or_commanded():
    """A NaN pose reads CLEAR: `_hits` skips every box whose comparison is False. Both branches
    must refuse on the measured start -- the ramp branch had no guard and flew 240 NaN poses."""
    bad = _START.copy()
    bad[1] = np.nan
    for secs in (1.0, None):
        d = _Demo(moves=True)
        d._arm_q8 = lambda: bad.copy()
        r = move_linear(d, np.array([0.0, 0.0, -0.03]), "base", secs=secs)
        assert r.outcome is Outcome.REFUSED and "finite" in r.reason, (
            f"secs={secs}: {r.outcome} {r.reason}")
        assert not d.commanded, f"secs={secs}: a non-finite start was commanded"
        assert d._plan_fallbacks == {"linear-start-not-finite": 1}, (
            f"secs={secs}: the refusal was not counted: {d._plan_fallbacks}")


def test_a_regoal_callback_returns_a_pose_and_the_executor_receives_its_goal_set():
    """The caller re-derives a q8; api turns it into the goal SET the executor replans to, so the
    caller never touches the planner's branch expansion."""
    seen = {}
    d = _Demo(moves=True)

    def _goal_set(goal, grasp, held):
        seen.update(goal=np.asarray(goal, float), grasp=grasp, held=held)
        return [np.asarray(goal, float)]
    d._goal_set = _goal_set
    d._execute_arm_path = lambda wps, secs, **kw: seen.update(regoal=kw.get("regoal")) or np.asarray(wps[-1], float)
    payload = np.array([[-0.04, -0.04, -0.09], [0.04, 0.04, 0.09]])
    move_to_pose(d, _START, 1.0, held=payload, regoal=lambda: _START + 0.01)
    assert callable(seen["regoal"]), "the executor was not handed a regoal callable"
    got = seen["regoal"]()
    assert isinstance(got, list) and np.allclose(got[0], _START + 0.01), got
    assert np.allclose(seen["goal"], _START + 0.01) and seen["grasp"] == () and seen["held"] is payload
    seen.clear()
    move_to_pose(d, _START, 1.0, regoal=lambda: None)
    assert seen["regoal"]() is None, "a caller that cannot re-derive must read as None, not a set"
    seen.clear()
    move_to_pose(d, _START, 1.0)
    assert seen["regoal"] is None, "no callback in means no callback out"


def test_the_facade_modules_import_offline_and_retreat_is_not_imported_at_module_level():
    """`cartesian_retreat` imports retreat.py at the call because retreat.py imports api at module
    level; a module-level import in api/execute/plan would be a cycle, and a defect in retreat.py
    would otherwise first surface after the object is released, inside the bay."""
    import importlib
    import pkgutil
    import morph.arm
    mods = sorted(f"morph.arm.{m.name}" for m in pkgutil.iter_modules(morph.arm.__path__) if m.name != "stage")
    assert len(mods) >= 11, mods
    # stage.py needs pxr; every other module imports offline
    for mod in mods:
        importlib.import_module(mod)
    for rel in ("api.py", "execute.py", "plan.py"):
        tree = ast.parse(open(os.path.join(ROOT, "morph", "arm", rel), encoding="utf-8").read())
        top = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "morph.arm.retreat"] \
            + [n for n in tree.body if isinstance(n, ast.Import) and any(a.name == "morph.arm.retreat" for a in n.names)]
        assert not top, f"morph/arm/{rel} imports retreat at module level: that is the cycle"


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
