"""`_ramp_arm`'s opt-in collision check: what it judges, what it refuses, what it still commands.
Offline: no Isaac, no simulator. Run: ./.venv/bin/python3 tests/test_ramp_check.py"""
import ast
import contextlib
import io
import json
import os
import re
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from morph.arm.api import ramp_blocked                                      # noqa: E402
from morph.arm.collision import MARGIN                                      # noqa: E402
from morph.arm.model import ArmModel                                        # noqa: E402

RAMP = os.path.join(ROOT, "morph", "close", "ramp.py")
LOWEST = os.path.join(ROOT, "morph", "close", "lowest.py")
REACH_JSON = os.path.join(ROOT, "tools", "bench_problems", "reach.json")


def _module(path):
    """A module from source TEXT: importing `morph.close` needs Isaac, and the bytecode cache is
    keyed on (mtime seconds, size), so a mutation can land in the same second at the same size."""
    mod = types.ModuleType("_" + os.path.basename(path).rsplit(".", 1)[0] + "_under_test")
    mod.__file__ = path
    exec(compile(open(path, encoding="utf-8").read(), path, "exec"), mod.__dict__)
    return mod


_RAMP = _module(RAMP)

FINGERS = ("finger_a_joint_1_1", "finger_b_joint_1_1")
# NOT identity and not from 0: the articulation carries far more joints than the eight, so a
# check that sliced `q_to[:8]` instead of mapping through `idx` would read different joints.
IDX = {n: 3 + i for i, n in enumerate(ArmModel.Q8)}
IDX.update({n: 11 + i for i, n in enumerate(FINGERS)})
NQ = max(IDX.values()) + 1


class _Model:
    """Blames `blamed` -- an index into the obstacle list it is HANDED -- and records every call."""

    joints = {n: {} for n in list(ArmModel.Q8) + list(FINGERS)}

    def __init__(self, blamed=None):
        self.blamed, self.seen, self._box = blamed, [], None

    def first_hit(self, q8, obs, margin, held=None, base=None, links=None, fingers=None):
        self.seen.append({"q8": np.asarray(q8, float), "obs": list(obs), "margin": margin,
                          "links": links, "fingers": fingers, "base": base})
        if self.blamed is None:
            return None
        # `blame`'s second pass, naming the prim
        if len(obs) == 1:
            return "Gripper_Link3_1" if obs[0] is self._box else None
        self._box = obs[self.blamed]
        return "Gripper_Link3_1"


class _Demo(_RAMP.RampStage):
    """The `self` `_ramp_arm` and `ramp_blocked` touch, and nothing else."""

    dt = 1.0 / 60.0

    def __init__(self, blamed=None, focus="pickup_obj_0", raises=False):
        self.model, self.raises = _Model(blamed), raises
        self._focus_obj, self._finger_cmd = focus, None
        self.idx = dict(IDX)
        self.forced, self.applied, self.steps, self.fallbacks = [], [], 0, []
        self.obs_asked = None
        self.world = types.SimpleNamespace(step=self._step)
        self.robot = types.SimpleNamespace(get_joint_positions=lambda: np.zeros(NQ) + 0.05)

    def _arm_model(self):
        return self.model

    def _arm_obstacles(self, grasp_names=(), *, exclude_names=(), diagnostic_paths=None):
        if self.raises:
            raise KeyError("no prim named 'pickup_obj_0' on the stage")
        self.obs_asked = tuple(grasp_names)
        grasp = "grasped" if "pickup_obj_0" in grasp_names else "world"
        out = [(np.zeros((2, 3)), "chassis"), (np.zeros((2, 3)), "world"),
               (np.zeros((2, 3)), grasp)]
        if diagnostic_paths is not None:
            diagnostic_paths += ["/World/Geometry/robot/base_footprint/base/Box",
                                 "/World/Geometry/shelf_2/board_0",
                                 "/World/Geometry/pickup_obj_0"]
        return out

    def _arm_base_world(self):
        return np.zeros(3), np.eye(3)

    def _fallback(self, tag):
        self.fallbacks.append(tag)

    def _force(self, q):
        self.forced.append(np.asarray(q, float).copy())

    def _apply(self, q):
        self.applied.append(np.asarray(q, float).copy())

    def _step(self, render=False):
        self.steps += 1

    def _pinch(self):
        return np.array([0.0, 0.0, 0.1])

    def base_ledger(self):
        return 0.0, 0.0, 0.0

    def set_base(self, *_a):
        pass


def _ramp(demo, q_to, check=True):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ok = demo._ramp_arm(np.zeros(NQ), q_to, check=check)
    return ok, out.getvalue()


def _target(**over):
    q = np.zeros(NQ)
    q[IDX["ColumnLeftBearingJoint_1"]] = 0.2
    q[IDX["ColumnRightBearingJoint_1"]] = 0.33
    q[IDX["ArmLeftJoint_1"]] = 0.44
    for k, v in over.items():
        q[IDX[k]] = v
    return q


Q_TO = _target()


def test_a_blamed_target_commands_nothing_and_counts_one_refusal():
    """Mutation caught: deleting the check, or placing it after the first `_force`/`_apply`. The
    point is that NO open-loop step runs once the target is inside MARGIN of a world box."""
    demo = _Demo(blamed=0)
    ok, log = _ramp(demo, Q_TO)
    assert ok is False, "a blamed target reported success"
    assert not demo.forced and not demo.applied and demo.steps == 0, (
        f"the refused ramp still commanded the arm: {len(demo.forced)} _force, "
        f"{len(demo.applied)} _apply, {demo.steps} steps")
    assert demo.fallbacks == ["ramp-blocked"], (
        f"a geometry refusal is not counted exactly once, under its own tag: {demo.fallbacks}")
    assert "Gripper_Link3_1" in log and "board_0" in log, (
        f"the refusal does not name the link and the prim it refused for: {log!r}")
    assert "chassis" in log and "grasp target" in log, (
        f"the refusal does not state what the check cannot see: {log!r}")


def test_an_unchecked_caller_never_reaches_the_check():
    """Mutation caught: making the check unconditional. Eleven of the eighteen `_ramp_arm` calls
    discard the return, and every one of them either restores a pose the arm just held or
    re-commands the live pose with a small increment; refusing one strands the arm where its
    caller was trying not to be, while the caller proceeds as though it moved."""
    demo = _Demo(blamed=0)
    ok, _ = _ramp(demo, Q_TO, check=False)
    assert ok is True and demo.model.seen == [], "an unchecked ramp consulted the checker anyway"
    assert len(demo.forced) == 110 and demo.fallbacks == [], (
        f"an unchecked ramp did not command: {len(demo.forced)} steps, {demo.fallbacks}")


def test_a_clear_target_commands_exactly_what_it_commanded_before():
    """Mutation caught: a check that refuses a clear pose, and any drift in the step arithmetic.

    Pinned from the shipped formula at dt 1/60 and `_dmax` 0.44: `n_steps = max(8, int(0.44/0.004))`
    = 110 and `n_settle = max(1, int(0.02/dt))` = 1, so 110 force, 110 apply, 110 world steps."""
    demo = _Demo(blamed=None)
    ok, _ = _ramp(demo, Q_TO)
    assert ok is True, "a clear target was refused"
    assert (len(demo.forced), len(demo.applied), demo.steps) == (110, 110, 110), (
        f"the cleared ramp commanded {len(demo.forced)} force / {len(demo.applied)} apply / "
        f"{demo.steps} steps, not the 110/110/110 the shipped arithmetic gives")
    assert demo.fallbacks == [], f"a cleared ramp counted a fallback: {demo.fallbacks}"


def test_the_check_drops_the_chassis_exempts_the_target_and_judges_the_commanded_hand():
    """Mutation caught: three wiring errors, each of which refuses every shipped seat.
    `test_the_shipped_seat_*` binds the same three against real geometry; this one binds the
    plumbing, so a lost `grasp_names` or a dropped `fingers=` fails here even if a box moves."""
    demo = _Demo(blamed=None)
    _ramp(demo, Q_TO)
    assert demo.model.seen, "the check never reached `first_hit`"
    call = demo.model.seen[0]
    assert demo.obs_asked == ("pickup_obj_0",), (
        f"the grasp target is not exempted: obstacles asked for with {demo.obs_asked}")
    tags = [o[1] for o in call["obs"]]
    assert "chassis" not in tags, f"the check still judges the chassis: {tags}"
    assert "grasped" in tags, f"the grasp target did not arrive exempt: {tags}"
    assert set(call["fingers"] or ()) == set(FINGERS), (
        f"the check judges the swept hand, not the commanded one: {call['fingers']}")
    assert call["base"] is not None, (
        "the check judges world boxes in the BASE frame: without `base` every world obstacle is "
        "carried to the wrong place and a pose inside the floor reads clear")


def test_the_check_maps_the_target_through_idx_rather_than_slicing_it():
    """Mutation caught: `q_to[:8]`. The articulation carries many more joints than the eight, so
    a slice reads whichever joints sit at 0..7 -- a pose that is not the one commanded."""
    demo = _Demo(blamed=None)
    q = _target(ArmLeftJoint_1=0.31, BaseJoint_1=-0.22)
    _ramp(demo, q)
    assert demo.model.seen, "the check never reached `first_hit`"
    got = demo.model.seen[0]["q8"]
    want = np.array([q[IDX[n]] for n in ArmModel.Q8])
    assert np.allclose(got, want), f"the check judges {got}, not the commanded pose {want}"
    assert not np.allclose(got, q[:8]), (
        "the fixture no longer distinguishes a mapped q8 from a sliced one")


def test_an_unbuildable_check_refuses_instead_of_passing():
    """Mutation caught: swallowing the failure and ramping anyway. A checker that could not be
    built has judged nothing, and the repo's rule for that is to end the motion, not fly it.

    The non-finite case matters most: every box comparison against NaN is False, so an unguarded
    `first_hit` reads a NaN pose as CLEAR."""
    cases = ((_Demo(raises=True), Q_TO, "the obstacle set raised"),
             (_Demo(focus=None), Q_TO, "there is no grasp target to exempt"),
             (_Demo(), _target(ArmLeftJoint_1=float("nan")), "the target is not finite"))
    for demo, q, why in cases:
        ok, log = _ramp(demo, q)
        assert ok is False and not demo.forced, f"{why} and the ramp ran anyway"
        assert demo.fallbacks == ["ramp-unbuildable"], (
            f"{why} and it was not counted as unbuildable: {demo.fallbacks}")
        assert "unbuildable" in log, f"{why} and the line does not say so: {log!r}"


def test_the_two_open_loop_ramps_ask_for_the_check_and_read_its_answer():
    """Mutation caught: dropping the check at either site, pinning it to a constant False, or
    going back to discarding the return at the columns-finish ramp. Two of the seven forward
    ramps are checked; these are those two."""
    tree = ast.parse(open(LOWEST, encoding="utf-8").read())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_reach_lowest")
    asked = {}
    for c in ast.walk(fn):
        if not (isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "_ramp_arm"):
            continue
        kw = next((k.value for k in c.keywords if k.arg == "check"), None)
        if kw is not None and not (isinstance(kw, ast.Constant) and kw.value is False):
            asked[c.args[0].id if isinstance(c.args[0], ast.Name) else "?"] = ast.unparse(kw)
    assert set(asked) == {"_q_c2", "q"}, (
        f"the ramps that ask for the check are {sorted(asked)}; the columns-finish ramp "
        f"(`_q_c2`) and the staged fallback's a1 retract (`q`) must both ask")
    assert asked["_q_c2"] == "_drop > 0.0", (
        f"the columns-finish ramp asks with {asked['_q_c2']!r}; a zero-drop ramp commands the "
        f"pose the arm already holds and must not be judged as though it moved")
    read = {node.test.operand.args[0].id for node in ast.walk(fn)
            if isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.operand, ast.Call)
            and getattr(node.test.operand.func, "attr", "") == "_ramp_arm"
            and isinstance(node.test.operand.args[0], ast.Name)}
    assert set(asked) <= read, (
        f"a ramp asks for the check and discards its answer: {sorted(set(asked) - read)}")


def test_the_staged_fallback_is_built_from_the_live_pose_not_the_entry_one():
    """Mutation caught: dropping the refresh. `q` is last read BEFORE the 800-step a1 extension,
    so without it the fallback's target -- and the check on it -- describe a pose the arm left."""
    tree = ast.parse(open(LOWEST, encoding="utf-8").read())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_reach_lowest")
    staged = next(n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
                  and getattr(n.func, "attr", "") == "_fallback"
                  and n.args and getattr(n.args[0], "value", None) == "lowest-staged")
    ramp = next(n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
                and getattr(n.func, "attr", "") == "_ramp_arm"
                and any(k.arg == "n_steps" and k.value.value == 12 for k in n.keywords))
    refresh = [n.lineno for n in ast.walk(fn)
               if isinstance(n, ast.Assign) and len(n.targets) == 1
               and getattr(n.targets[0], "id", None) == "q"
               and "get_joint_positions" in ast.unparse(n.value)]
    assert any(staged < ln < ramp for ln in refresh), (
        f"`q` is not re-read from the robot between the staged fallback opening (line {staged}) "
        f"and its a1 retract (line {ramp}); reads at {refresh}")


SHIPPED = ("morph", "tools", "tests")


def _shipped_sources():
    """Every shipped .py: the three source trees plus the scripts at the repo root."""
    for top in SHIPPED:
        for base, _dirs, names in os.walk(os.path.join(ROOT, top)):
            for name in sorted(n for n in names if n.endswith(".py")):
                yield os.path.join(base, name)
    for name in sorted(n for n in os.listdir(ROOT) if n.endswith(".py")):
        yield os.path.join(ROOT, name)


def test_ramp_arm_is_defined_once_and_only_in_ramp_py():
    """Mutation caught: a second `_ramp_arm` left behind by the move, or the definition living
    anywhere but `ramp.py` -- either way two stages could ramp by different code."""
    found = []
    for path in _shipped_sources():
        hits = re.findall(r"^\s*def _ramp_arm\b", open(path, encoding="utf-8").read(), re.M)
        found += [os.path.relpath(path, ROOT)] * len(hits)
    assert found == [os.path.join("morph", "close", "ramp.py")], (
        f"`_ramp_arm` is defined in {found}, not once in morph/close/ramp.py")


def test_ramp_stage_is_imported_and_mixed_into_close_mixin():
    """Mutation caught: the half of the move no offline check can see. Dropping `RampStage` from
    `CloseMixin`'s bases leaves every suite green while the composed `Demo` loses `_ramp_arm` at
    all 18 call sites."""
    tree = ast.parse(open(os.path.join(ROOT, "morph", "close", "__init__.py"),
                          encoding="utf-8").read())
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                and n.module == "morph.close.ramp" for a in n.names}
    assert "RampStage" in imported, f"morph.close does not import RampStage: {sorted(imported)}"
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CloseMixin")
    bases = [b.id for b in cls.bases if isinstance(b, ast.Name)]
    assert "RampStage" in bases, f"RampStage is not a CloseMixin base: {bases}"


# The shipped seat, `handoff[post-reach-lowest]` -- identical in 7 of 8 archived runs.
SEAT = dict(th=-0.1957, h1=0.1998, h2=0.3292, a1=0.4367)


def _seat_case():
    """(model, q8, base, fingers, obstacles) at the shipped seat, from the captured request whose
    single goal IS `handoff[pre-reach-lowest]` -- FK through its base reproduces the logged pinch
    to 0.3mm, so boxes, base and hand all belong to one run. h1 and h2 are the MEASURED pair, not
    h1 plus the goal's dh: the 0.6mm between them moves every pad number by about 0.7mm."""
    req = json.load(open(REACH_JSON, encoding="utf-8"))
    m = ArmModel.load(os.path.join(ROOT, req["model"]), os.path.join(ROOT, req["lut"]))
    obs = [(np.array([o[0], o[1]], float), o[2]) for o in req["obstacles"]]
    for i, pad, links in req["obstacle_pads"]:
        obs[i] = (obs[i][0], obs[i][1], float(pad), tuple(links))
    q8 = np.array(req["goals"][0], float)
    q8[0], q8[1], q8[2], q8[3] = SEAT["th"], SEAT["h1"], SEAT["h2"], SEAT["a1"]
    base = (np.array(req["base"][0], float), np.array(req["base"][1], float))
    return m, q8, base, req["fingers"], obs


def _target_box(obs):
    """The grasp target: the only 80x80x180mm world box at the logged seat XY."""
    seat = np.array([5.012, -6.551])
    for i, o in enumerate(obs):
        if len(o) > 2 or o[1] == "chassis":
            continue
        lo, hi = o[0]
        if np.max(hi - lo) < 0.30 and float(np.linalg.norm(0.5 * (lo + hi)[:2] - seat)) < 0.01:
            return i
    raise AssertionError("no grasp-target box at the logged seat XY in the captured request")


def _check_set(obs, t):
    return [(o[0], "grasped") if i == t else o for i, o in enumerate(obs)
            if not (len(o) > 2 or o[1] == "chassis")]


def test_the_shipped_seat_passes_the_check_and_every_wider_set_refuses_it():
    """Mutation caught: the one that breaks the machine rather than the code. Each wider set is a
    plausible simplification, and each refuses a seat eight archived runs completed.

    Per-(link, obstacle) margin-to-refusal at the seat: chassis kept -> `Arm_Left_1` OVERLAPPING
    the deck box; target not exempt -> `finger_a_link_3_1` at 1.70mm; swept hand ->
    `Gripper_Link3_1` 4.12mm from the ground. The shipped set clears by 40.59mm, its nearest pair
    `finger_b_link_3_1` against the ground plane."""
    m, q8, base, fingers, obs = _seat_case()
    world = _check_set(obs, _target_box(obs))
    assert m.first_hit(q8, world, MARGIN, None, base, fingers=fingers) is None, (
        "the check refuses the shipped seat: it would end every pick")
    assert m.first_hit(q8, obs, MARGIN, None, base, fingers=fingers) is not None, (
        "keeping the chassis boxes no longer refuses the seat -- the filter is a no-op and this "
        "test can no longer tell the two sets apart")
    unexempt = [o for o in obs if not (len(o) > 2 or o[1] == "chassis")]
    assert m.first_hit(q8, unexempt, MARGIN, None, base, fingers=fingers) is not None, (
        "the grasp target no longer needs its exemption -- re-derive before trusting the check")
    assert m.first_hit(q8, world, MARGIN, None, base, fingers=None) is not None, (
        "the swept hand no longer refuses the seat, so `fingers=` is untested here")


def test_the_check_leaves_measured_room_below_the_shipped_seat():
    """Mutation caught: a silent tightening -- a wider box, a bigger MARGIN, a lost exemption --
    that leaves the seat passing but the stage with no travel. The binding pair is a finger pad
    against the GROUND and it closes 1:1 with column travel, so the headroom IS the number.

    `_reach_lowest` floors the columns at `_h1_min` 0.20 and the seat sits at 0.1998, so the
    shipped stage cannot spend this; a second archived dock seats about 10mm lower."""
    m, q8, base, fingers, obs = _seat_case()
    world = _check_set(obs, _target_box(obs))
    room = 0.0
    while room < 0.10:
        q = q8.copy()
        q[1], q[2] = q8[1] - room, q8[2] - room
        if m.first_hit(q, world, MARGIN, None, base, fingers=fingers) is not None:
            break
        room += 0.001
    assert 0.015 <= room <= 0.030, (
        f"the check refuses {room * 1000:.0f}mm below the shipped seat, not the measured 21mm. "
        f"Below 15mm the stage has no room; above 30mm the geometry has moved and the 40.59mm "
        f"ground clearance this rests on must be re-derived")


class _RealDemo:
    """`ramp_blocked`'s `self`, wired to the REAL model, the REAL captured boxes, base and hand.
    The only fakes are the prim paths, which the captured request does not carry."""

    _finger_cmd = None

    def __init__(self, truncate_paths=False):
        self.m, _, self.base, fingers, self.obs = _seat_case()
        self.truncate, self._focus_obj = truncate_paths, "pickup_obj_0"
        t = _target_box(self.obs)
        self.paths = [f"/World/Geometry/shelf/board_{i}" for i in range(len(self.obs))]
        self.paths[t] = "/World/Geometry/pickup_obj_0"
        self.paths[91] = "/World/Geometry/GroundPlane/CollisionPlane"
        self.idx = {n: i for i, n in enumerate(ArmModel.Q8)}
        q = np.zeros(len(self.idx) + len(fingers))
        for k, (name, v) in enumerate(fingers.items()):
            self.idx[name] = len(ArmModel.Q8) + k
            q[self.idx[name]] = v
        self.robot = types.SimpleNamespace(get_joint_positions=lambda: q)

    def _arm_model(self):
        return self.m

    def _arm_base_world(self):
        return self.base

    def _arm_obstacles(self, grasp_names=(), *, exclude_names=(), diagnostic_paths=None):
        want = {f"/World/Geometry/{n}" for n in grasp_names}
        out = [((o[0], "grasped") + tuple(o[2:])) if p in want else o
               for o, p in zip(self.obs, self.paths)]
        if diagnostic_paths is not None:
            diagnostic_paths += self.paths[:-3] if self.truncate else list(self.paths)
        return out

    def q(self, h1, h2, a1, th=-0.1957):
        full = np.asarray(self.robot.get_joint_positions(), float).copy()
        full[:8] = np.array(_seat_case()[1], float)
        full[0], full[1], full[2], full[3] = th, h1, h2, a1
        return full


def test_the_real_check_clears_the_shipped_seat_and_refuses_the_floor():
    """Mutation caught: anything that makes the REAL `ramp_blocked` disagree with the geometry --
    a lost `base`, a lost exemption, a lost chassis filter, the swept hand, a sliced q8. Every
    other test here drives a fake model; this one drives the real one end to end.

    The shipped seat and both archived docks clear. Twenty-five millimetres of column below the
    shipped seat refuses, on a finger pad against the ground plane."""
    d = _RealDemo()
    for label, (h1, h2, a1) in {"shipped seat": (0.1998, 0.3292, 0.4367),
                                "accept3 seat": (0.1973, 0.3267, 0.4366),
                                "lower dock  ": (0.1655, 0.2948, 0.4067)}.items():
        assert ramp_blocked(d, d.q(h1, h2, a1)) is None, (
            f"the real check refuses the {label}: {ramp_blocked(d, d.q(h1, h2, a1))}")
    low = ramp_blocked(d, d.q(0.1748, 0.3042, 0.4367))
    assert low is not None and low[0] == "blocked", (
        f"25mm of column below the shipped seat is not refused: {low}")
    assert "finger" in low[1] and "GroundPlane" in low[1], (
        f"the refusal blames something other than a pad against the floor: {low}")


def test_the_real_check_refuses_when_the_prim_paths_do_not_match_the_boxes():
    """Mutation caught: dropping the length guard and letting `zip` truncate. Silently, every
    prim past the gap is misnamed and the tail of the obstacle set is never judged at all."""
    got = ramp_blocked(_RealDemo(truncate_paths=True), _RealDemo().q(0.1998, 0.3292, 0.4367))
    assert got is not None and got[0] == "unbuildable", (
        f"a truncated path list was judged instead of refused: {got}")


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
