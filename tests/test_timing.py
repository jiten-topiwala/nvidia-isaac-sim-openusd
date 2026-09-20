"""Measured arm limits, and `morph/arm/timing.py`.

Offline: no Isaac. Run: ./.venv/bin/python3 tests/test_timing.py

WHAT THIS FILE PINS.
  * the per-joint reflected inertia and static gravity load, recomputed here from
    `usd/_arm_model.json` + `usd/_link_masses.json`, and the velocity/acceleration limits that
    follow from them and from the REAL gain rule read out of `morph/robot.py` (via `ast`, since
    that file imports Isaac at module scope). Nothing here is copied from `_ARM_REFLECTED`: that
    table is wrong by 53x on the right column and is the thing these tests are here to outrank.
  * `morph/arm/timing.py::retime`, the trapezoidal parameterisation itself.
  * `morph/arm/execute.py::execute_path` flying a planned path inside those limits.
"""
import ast
import itertools
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from morph.arm.model import ArmModel, _AXIS_V, _quat_to_R  # noqa: E402
from morph.arm.timing import retime, _collapse                                # noqa: E402

DT = 1.0 / 240.0                     # scene.py:805, World(physics_dt=1/240)
G = 9.81
HELD_OFF = 0.12                      # object centre out along the hand link's own +Z


# ------------------------------------------------------------------------------- the measurement
def _class_consts(rel, cls, names):
    """Class-level constant assignments out of a file that imports Isaac and so cannot be
    imported. Same helper as tests/test_fallback_counter.py:2198 -- a derivation checked against a
    table retyped here would be a derivation nothing keeps honest."""
    tree = ast.parse(open(os.path.join(ROOT, *rel)).read())
    cd = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
    body = [n for n in cd.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    g = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), os.path.join(*rel), "exec"), g)
    assert names <= set(g), f"{sorted(names - set(g))} not on {cls} in {os.path.join(*rel)}"
    return {k: g[k] for k in names}


def _module_consts(rel, names):
    """Module-level constant assignments out of a file that imports Isaac and so cannot be
    imported. Same idea as `_class_consts`, one scope up -- `TRACK_STEP_CLIP_M` lives at module
    scope in `morph/robot.py`, not on a class."""
    tree = ast.parse(open(os.path.join(ROOT, *rel)).read())
    body = [n for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    g = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), os.path.join(*rel), "exec"), g)
    assert names <= set(g), f"{sorted(names - set(g))} not at module scope in {os.path.join(*rel)}"
    return {k: g[k] for k in names}


def _exec_method(rel, name, glb):
    """Compile ONE method out of a source file and bind it into `glb`. Same helper as
    tests/test_fallback_counter.py:456."""
    path = os.path.join(ROOT, *rel)
    tree = ast.parse(open(path).read())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    assert not fn.decorator_list, f"{name} grew a decorator, which this harness would drop"
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), path, "exec"),
         glb)
    return glb[name]


class _GainRule:
    """`morph/robot.py`'s REAL `_arm_gain` / `_wrist_gain` over the smallest surface they touch."""

    def __init__(self):
        for k, v in _class_consts(("morph", "robot.py"), "RobotMixin",
                                  {"_ARM_REFLECTED", "_ARM_EFFORT", "_ARM_LEAVES", "_WRIST"}).items():
            setattr(self, k, v)
        glb = {"os": type("os", (), {"environ": {}})()}
        for m in ("_arm_gain", "_wrist_gain"):
            setattr(type(self), m, _exec_method(("morph", "robot.py"), m, glb))

    def _drive_on(self):
        # the drive path is the only one that bounds effort at all
        return True


MODEL = ArmModel.load(os.path.join(ROOT, "usd", "_arm_model.json"),
                      os.path.join(ROOT, "usd", "_closure_lut.json"))
LINK_MASS = json.load(open(os.path.join(ROOT, "usd", "_link_masses.json")))
HELD = LINK_MASS["pickup_obj_0"]                       # 1.02542 kg, the transported load
_KIDS = {}
for _j in MODEL.m["joints"]:
    _KIDS.setdefault(_j["parent"], []).append(_j["child"])


def _subtree(link):
    out, st = [], [link]
    while st:
        n = st.pop()
        out.append(n)
        st.extend(_KIDS.get(n, []))
    return out


def _reflect(jname, poses):
    """(reflected inertia about the joint's own axis, |static gravity load| at the joint), with the
    transported object rigid in the hand. Prismatic -> (kg, N); revolute -> (kg m^2, N m).

    Composite-rigid-body, exact: the mass file gives principal moments plus `iquat`/`ipos` in each
    link's own frame and `ArmModel.fk` carries every link frame into the base frame. This is the
    number `_ARM_REFLECTED` claims to hold and does not."""
    j = MODEL.joints[jname]
    cp, cR = poses[j["child"]]
    o = cp + cR @ j["_p1"]                              # joint origin, inverted out of the child
    a = (cR @ j["_R1"]) @ _AXIS_V[j["axis"]]            # joint axis, base frame
    a = a / np.linalg.norm(a)
    st = _subtree(j["child"])
    bodies = []
    for link in st:
        e = LINK_MASS.get(link)
        if not e or e["mass"] <= 0.0:
            continue
        p, R = poses[link]
        Rp = R @ _quat_to_R(*e["iquat"])
        bodies.append((e["mass"], Rp @ np.diag(e["inertia"]) @ Rp.T,
                       p + R @ np.asarray(e["ipos"], float)))
    if MODEL.hand_link in st:
        p, R = poses[MODEL.hand_link]
        bodies.append((HELD["mass"], R @ np.diag(HELD["inertia"]) @ R.T,
                       p + R @ np.array([0.0, 0.0, HELD_OFF])))
    gv = np.array([0.0, 0.0, -G])
    if j["type"] == "prismatic":
        mass = sum(b[0] for b in bodies)
        return mass, abs(mass * float(a @ gv))
    return (sum(float(a @ b[1] @ a) + b[0] * float(np.cross(a, b[2] - o) @ np.cross(a, b[2] - o))
                for b in bodies),
            abs(sum(float(a @ np.cross(b[2] - o, b[0] * gv)) for b in bodies)))


# The worst case over the reachable workspace: heaviest inertia and largest gravity load need not
# coincide, so both are pinned. The three prismatic joints are configuration-INDEPENDENT.
WORST = {
    "BaseJoint_1":
        ([0.0, 0.0, 0.0, 0.46875, 0.0, 1.570796, 0.0, 0.0], 2.10765948,
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.00000000),
    "ColumnLeftBearingJoint_1":
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 12.49773035,
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 122.60273471),
    "ColumnRightBearingJoint_1":
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.26661509,
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2.61549401),
    "ArmLeftJoint_1":
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 12.09843502,
         [0.0, 0.0, 0.3, 0.0, -1.57, -3.141593, -1.5708, -1.5708], 112.59555047),
    "HandBearingJoint_1":
        ([0.0, 0.933291, 0.834038, 0.0, -0.785024, -3.14004, 0.01842, 0.0], 0.04438989,
         [0.0, 0.933333, 0.833333, 0.02, 0.785, -2.04159, 0.025, 0.0], 2.30831578),
    "gripper_z_rotation_1":
        ([0.0, 0.35, 0.3, 0.0, -1.169805, -3.12577, -0.001556, 1.570794], 0.02675069,
         [0.0, 0.0, 0.0, 0.0, 0.0, 1.570796, 1.5708, 0.0], 1.49811599),
    "gripper_y_rotation_1":
        ([0.0, 0.466671, 0.466669, 0.208333, 0.785003, 1.570805, 1.5708, 0.0], 0.02674427,
         [0.0, 0.933333, 0.833333, 0.000161, 1.552404, -3.141593, 0.7854, 0.0], 1.49837009),
    "gripper_x_rotation_1":
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.02629895,
         [0.0, 0.0, 0.0, 0.0, 0.0, -3.141541, -1.5708, 1.570785], 1.49811589),
}

# v_max = effort/kd is a BRAKING policy, not a speed ceiling: the speed a lost feedforward can
# still be damped from. a_max is ONE joint against its own inertia in isolation -- derate it.
LIMITS = {
    "BaseJoint_1":               (0.111803, 142.3380),
    "ColumnLeftBearingJoint_1":  (0.363376, 110.2118),
    "ColumnRightBearingJoint_1": (0.363376, 5616.2782),
    "ArmLeftJoint_1":            (0.122868, 32.0210),
    "HandBearingJoint_1":        (0.583874, 623.8286),
    "gripper_z_rotation_1":      (0.583874, 1065.4637),
    "gripper_y_rotation_1":      (0.583874, 1065.7098),
    "gripper_x_rotation_1":      (0.583874, 1083.7652),
}


def test_the_worst_case_inertia_and_gravity_load_are_reproducible():
    """The measurement itself: recomputed from the two JSON files, not read off a config."""
    bad = []
    for n, (qi, i_exp, qg, g_exp) in WORST.items():
        i_got = _reflect(n, MODEL.fk(np.asarray(qi, float)))[0]
        g_got = _reflect(n, MODEL.fk(np.asarray(qg, float)))[1]
        if abs(i_got - i_exp) > 1e-6 * max(1.0, i_exp):
            bad.append(f"{n}: inertia {i_got} != {i_exp}")
        if abs(g_got - g_exp) > 1e-6 * max(1.0, g_exp):
            bad.append(f"{n}: gravity {g_got} != {g_exp}")
    assert not bad, "; ".join(bad)


def test_the_prismatic_reflected_mass_is_the_subtree_and_nothing_else():
    """The claim `_ARM_REFLECTED` gets wrong. A prismatic joint translates its whole subtree
    rigidly, so its reflected mass is exactly the sum of the subtree masses in ANY pose -- 0.267 kg
    on the right column, because NO_LOOP=1 leaves the four-bar open and that subtree is three
    links. No FK, no configuration, no argument."""
    for n in ("ColumnLeftBearingJoint_1", "ColumnRightBearingJoint_1", "ArmLeftJoint_1"):
        st = _subtree(MODEL.joints[n]["child"])
        raw = sum(LINK_MASS.get(l, {}).get("mass", 0.0) for l in st)
        load = HELD["mass"] if MODEL.hand_link in st else 0.0   # the right column carries nothing
        _qi, i_exp, _qg, _g = WORST[n]
        assert abs(raw + load - i_exp) <= 1e-6, f"{n}: {raw} + {load} != {i_exp}"


def test_no_reachable_configuration_beats_the_documented_worst_case():
    """The pinned worst case above is a sweep RESULT, so it needs a guard: if the model or the mass
    file changes so that some pose is heavier than the table says, the limits derived from it are
    optimistic and this must say so rather than the arm finding out."""
    lo, hi = MODEL.bounds()
    hi = hi.copy()
    # config caps the columns at 1.40
    hi[1] = hi[2] = min(hi[1], 1.40)
    seen = 0
    for th, h1, a1, hb, wz, wy in itertools.product(
            np.linspace(lo[0], hi[0], 3), np.linspace(lo[1], hi[1], 3),
            np.linspace(lo[3], hi[3], 4), np.linspace(lo[4], hi[4], 4),
            np.linspace(-math.pi, math.pi, 4), np.linspace(lo[6], hi[6], 3)):
        for dh in (MODEL.lut["dh"][0], 0.0, MODEL.lut["dh"][-1]):
            h2 = h1 + dh
            if not (lo[2] <= h2 <= hi[2]) or not MODEL.in_lut_domain(dh, a1):
                continue
            if MODEL.passive_violations(dh, a1):
                continue
            poses = MODEL.fk(np.array([th, h1, h2, a1, hb, wz, 0.0, hi[7]]))
            seen += 1
            for n, (_qi, i_exp, _qg, g_exp) in WORST.items():
                i_got, g_got = _reflect(n, poses)
                assert i_got <= i_exp * 1.001, f"{n}: inertia {i_got} beats the documented {i_exp}"
                assert g_got <= g_exp * 1.001 + 1e-9, f"{n}: gravity {g_got} beats {g_exp}"
    assert seen > 200, f"the guard scan only reached {seen} valid poses, so it proves little"


def test_the_velocity_limit_follows_from_the_real_gain_rule():
    """v_max = effort / kd, with kd and the effort cap taken from `morph/robot.py`'s own
    `_arm_gain` -- not from a copy of `kd = 2*sqrt(kp*m)` typed out here."""
    rule = _GainRule()
    for n, (v_exp, _a) in LIMITS.items():
        kp, kd, eff = rule._arm_gain(n, 3.0e5, wrist_rule=True)
        assert kp > 0.0 and kd > 0.0, f"{n}: the gain rule gives no position drive at all"
        assert abs(v_exp - eff / kd) <= 1e-4 * v_exp, f"{n}: v_max {v_exp} != {eff / kd}"


def test_the_acceleration_limit_follows_from_the_measured_inertia():
    """a_max = (effort - |gravity|) / inertia. The effort cap is the real source; the inertia is
    the MEASURED one, which is why this is not just `_ARM_REFLECTED` rearranged."""
    rule = _GainRule()
    for n, (_v, a_exp) in LIMITS.items():
        _kp, _kd, eff = rule._arm_gain(n, 3.0e5, wrist_rule=True)
        _qi, inertia, _qg, grav = WORST[n]
        assert abs(a_exp - (eff - grav) / inertia) <= 1e-4 * a_exp, f"{n}: a_max {a_exp} wrong"


def test_every_joint_can_stop_from_its_velocity_limit_inside_one_control_step():
    """The closure that makes the pair usable: at v_max the momentum a single 1/240 s step has to
    absorb is inside the effort cap on every joint, so an abort or a checkpoint replan cannot
    demand a force the drive will not deliver. This is exactly what fails today -- 11.47 kg at the
    measured 1.5 m/s peak needs 4126 N against a 1500 N cap."""
    rule = _GainRule()
    for n, (v_max, _a) in LIMITS.items():
        _kp, _kd, eff = rule._arm_gain(n, 3.0e5, wrist_rule=True)
        _qi, inertia, _qg, grav = WORST[n]
        need = inertia * v_max / DT
        assert need <= eff - grav, f"{n}: one-step stop needs {need:.1f} of {eff - grav:.1f}"


def test_every_joint_is_critically_or_over_damped_against_the_measured_inertia():
    """Grounds the jerk justification in `morph/arm/timing.py`'s module docstring: a trapezoid's
    acceleration corner cannot make a drive RING only if zeta >= 1 against the joint's MEASURED
    inertia, not the shipped `_ARM_REFLECTED` table `kd = 2*sqrt(kp*m)` was actually tuned
    against. This is a narrower claim than "jerk is bounded" -- it says nothing about force-RATE
    limits -- but it is the one the evidence here actually supports."""
    rule = _GainRule()
    bad = []
    for n, (_qi, inertia, _qg, _g) in WORST.items():
        kp, kd, _eff = rule._arm_gain(n, 3.0e5, wrist_rule=True)
        zeta = kd / (2.0 * (kp * inertia) ** 0.5)
        if zeta < 1.0:
            bad.append(f"{n}: zeta {zeta:.4f} < 1 -- underdamped, CAN ring")
    assert not bad, "; ".join(bad)


def test_the_shipped_reflected_table_still_overstates_the_right_column():
    """A TRIPWIRE, not an endorsement. Every v_max above is `effort / kd` and kd comes from
    `_ARM_REFLECTED`, which is 53x heavy on the right column because NO_LOOP=1 leaves the four-bar
    open and its subtree is three links weighing 0.267 kg. When someone corrects that table the kd
    values drop, every v_max here rises, and this file's numbers must be re-derived -- so this test
    is here to fail at that moment rather than let the limits go quietly stale."""
    shipped = _class_consts(("morph", "robot.py"), "RobotMixin", {"_ARM_REFLECTED"})["_ARM_REFLECTED"]
    _qi, measured, _qg, _g = WORST["ColumnRightBearingJoint_1"]
    ratio = shipped["ColumnRightBearingJoint_1"] / measured
    assert ratio > 50.0, (
        f"_ARM_REFLECTED['ColumnRightBearingJoint_1'] is now {shipped['ColumnRightBearingJoint_1']} "
        f"against a measured {measured:.4f} kg ({ratio:.1f}x). If it was corrected, RE-DERIVE "
        f"LIMITS above: every v_max is effort/kd and kd = 2*sqrt(kp*m) reads that table.")


def test_the_continuous_rate_and_the_braking_speed_do_not_contradict():
    """`TRACK_STEP_CLIP_M/dt` (`morph/robot.py`) sits ABOVE `v_max` on the columns, and that is
    fine -- see the LIMITS comment above: different regimes, not a disagreement to resolve. 0.96
    m/s is a deliberate CONTINUOUS target rate whose 4 mm/step stays under `_apply`'s 5 mm jump
    mask, so the feedforward never drops out; v_max is the braking speed for the regime where it
    already has. Pinned here, reading the real constant via `ast`, so nobody re-opens this."""
    clip_m = _module_consts(("morph", "robot.py"), {"TRACK_STEP_CLIP_M"})["TRACK_STEP_CLIP_M"]
    clip_rate = clip_m / DT
    v_col = LIMITS["ColumnLeftBearingJoint_1"][0]
    assert abs(clip_rate - 0.96) <= 1e-6, f"TRACK_STEP_CLIP_M/dt is now {clip_rate}, not 0.96 m/s"
    assert clip_m <= 0.005, (
        f"TRACK_STEP_CLIP_M is now {clip_m}, no longer under `_apply`'s 5 mm jump mask -- the "
        f"'feedforward never drops out' half of the story no longer holds")
    assert clip_rate > v_col, (
        f"TRACK_STEP_CLIP_M/dt {clip_rate} is no longer above v_max {v_col} -- re-check whether "
        f"the two-regime story above still applies")


def test_bounding_the_eight_actuated_joints_does_not_bound_the_four_bar():
    """A measurement the WIRING task needs and would otherwise miss. `RotationLeftJoint_1` is a
    closure passive -- nothing plans it, `_with_closure` writes it from the LUT -- but it is in
    KNOWN["arm_joints"], so it gets a position drive with kd from `_ARM_REFLECTED` and a 300 Nm
    cap. Its rate is d(rot)/d(dh) * (h2dot - h1dot), and dh is a DIFFERENCE of two columns that
    `retime` bounds only INDIVIDUALLY. Two columns each inside v_max, moving apart, drive this
    joint far past its own SATURATION speed -- not past a physical wall it cannot cross, past the
    speed at which ITS zero-target brake (same `v_max = effort/kd` policy as the LIMITS comment
    above) would saturate. Bounding Q8 is therefore NOT bounding the arm.

    CONFIRMED three independent ways: this branch (below, grad and worst), a reviewer computing
    d(rot)/d(dh) = 9.870 rad/m from the closure LUT across its whole domain, another reproducing
    7.17 rad/s = 46.8x. Do not weaken this finding -- only its description as a hard physical limit
    was ever wrong.

    Also recorded here because it is new and confirmed: the peak d(rot)/d(dh) sits at dh = 0.0
    (the grid point between -0.02 and +0.02) across essentially the WHOLE valid a1 range,
    including a1's own lower bound a1 = 0.0 -- so it is on the operating envelope boundary, not
    in some unreachable corner. And `morph/pick/grasp.py`'s carry-level ramp (the cosine-ease
    boom-levelling pass run after every lift) drives dh at up to ~38x this same saturation-free
    threshold on EVERY LIFT -- taken from the review, not re-derived here, because `grasp.py` is
    OFF LIMITS for this task and re-deriving it needs a live run."""
    lut = json.load(open(os.path.join(ROOT, "usd", "_closure_lut.json")))
    k = lut["passives"].index("RotationLeftJoint_1")
    dh = np.asarray(lut["dh"], float)
    grad_by_dh = np.abs(np.gradient(np.asarray(lut["grid"], float)[:, :, k], dh, axis=0))
    grad = grad_by_dh.max()
    assert grad > 9.0, f"the LUT gradient is now {grad}, so this bound needs re-deriving"
    peak_dh = dh[np.unravel_index(np.argmax(grad_by_dh), grad_by_dh.shape)[0]]
    assert peak_dh == 0.0, f"the peak moved off dh=0.0 to {peak_dh}; the recorded location is stale"
    rule = _GainRule()
    _kp, kd, eff = rule._arm_gain("RotationLeftJoint_1", 3.0e5, wrist_rule=True)
    own = eff / kd                                       # its own saturation speed, rad/s
    worst = grad * 2.0 * LIMITS["ColumnLeftBearingJoint_1"][0]   # both columns apart at v_max
    # The differential (dh-rate) bound keeping THIS joint's own zero-target brake from saturating,
    # far tighter than either column's v_max alone. A braking policy, like v_max itself.
    diff_v_max = own / grad
    assert worst > 40.0 * own, (
        f"the coupling closed: {worst:.2f} rad/s driven against a {own:.4f} rad/s saturation speed")
    assert diff_v_max < LIMITS["ColumnLeftBearingJoint_1"][0] / 20.0, (
        f"the differential bound {diff_v_max:.4f} m/s is no longer well under the column's own "
        f"v_max -- the '~23x tighter' figure needs re-checking")



def _rot_left_gravity(model, masses, joints, links, h1, dh, a1):
    """|gravity moment| about RotationLeftJoint_1's own axis, Nm, at (h1, dh, a1).

    The four-bar is OPEN in the shipped articulation -- `play_isaac.py` sets NO_LOOP=1 and
    `scene.py` skips `close_arm_loop` on it -- so everything downstream of `Rotation_Link_Left_1`
    hangs off this hinge as a cantilever. Collision-box centres stand in for centres of mass,
    which is NOT conservative: a COM outboard of its box centre lengthens the lever."""
    q = np.zeros(8)
    q[1], q[2], q[3] = h1, h1 + dh, a1
    poses = model.fk(q)
    pp, pR = poses["Bearing_Column_Left_1"]
    jw = pp + pR @ np.asarray(joints["RotationLeftJoint_1"]["localPos0"], float)
    tot, mom = 0.0, np.zeros(3)
    for link, mass in links.items():
        lo, hi = np.asarray(model.link_aabb[link], float)
        p, R = poses[link]
        tot += mass
        mom += mass * (p + R @ ((lo + hi) / 2.0))
    r = mom / tot - jw
    axis = pR @ np.asarray(_AXIS_V["Y"], float)
    return abs(float(np.dot(np.cross(r, tot * np.array([0.0, 0.0, -9.81])), axis)))


def test_the_dh_cap_keeps_the_closure_joint_under_its_effort_ceiling_everywhere():
    """`_DH_V_MAX` is a braking policy: it bounds the column DIFFERENTIAL so that this joint's
    zero-target brake never saturates. The brake torque is `kd * v * d(rot)/d(dh)`, but that is
    not the only load -- the open four-bar leaves an 11 kg cantilever moment on the same axis,
    and the two ADD in whichever travel direction is the unfavourable one. The damping term
    flips sign with d(dh)/dt and the gravity term does not, so approach and retreat cannot both
    be favourable; since the cap is a magnitude limit `retime` applies to both, the binding case
    is the adding one.

    Both terms are pose-dependent and their worst corners COINCIDE at (dh, a1) = (0, 0) -- peak
    LUT sensitivity and peak lever. Gravity is nearly flat in dh but swings 2.9-34.8 Nm across
    a1, crossing near zero around a1 = 0.30 where the mass centroid passes over the joint axis,
    so any single-pose reading of it is meaningless.

    Recomputed here from the LUT, the link masses and the gain rule rather than pinned as a
    number, so that a new LUT bake, a mass change or a gain change fails this instead of
    silently overspeeding the joint."""
    from morph.arm.cartesian import _DH_V_MAX
    model = ArmModel.load(os.path.join(ROOT, "usd", "_arm_model.json"),
                          os.path.join(ROOT, "usd", "_closure_lut.json"))
    am = json.load(open(os.path.join(ROOT, "usd", "_arm_model.json")))
    joints = {e["name"]: e for e in am["joints"]}
    kids = {}
    for e in am["joints"]:
        kids.setdefault(e["parent"], []).append(e["child"])
    down, stack = set(), ["Rotation_Link_Left_1"]
    while stack:
        link = stack.pop()
        if link not in down:
            down.add(link)
            stack += kids.get(link, [])
    masses = json.load(open(os.path.join(ROOT, "usd", "_link_masses.json")))
    links = {n: masses[n]["mass"] for n in down if n in masses and n in model.link_aabb}
    assert 10.0 < sum(links.values()) < 12.0, f"the cantilevered mass moved: {sum(links.values())}"

    lut = json.load(open(os.path.join(ROOT, "usd", "_closure_lut.json")))
    k = lut["passives"].index("RotationLeftJoint_1")
    dh_ax, a1_ax = np.asarray(lut["dh"], float), np.asarray(lut["a1"], float)
    grad = np.abs(np.gradient(np.asarray(lut["grid"], float)[:, :, k], dh_ax, axis=0))
    _kp, kd, eff = _GainRule()._arm_gain("RotationLeftJoint_1", 3.0e5, wrist_rule=True)
    lo8, hi8 = (np.asarray(b, float) for b in model.bounds())

    worst, where = 0.0, None
    for i, dh in enumerate(dh_ax):
        # dh reachable by the two column bounds
        if not (0.0 <= dh <= hi8[2] - lo8[1]):
            continue
        for j, a1 in enumerate(a1_ax):
            if not (lo8[3] <= a1 <= hi8[3]):
                continue
            total = (kd * _DH_V_MAX * grad[i, j]
                     + _rot_left_gravity(model, masses, joints, links, 0.30, float(dh), float(a1)))
            if total > worst:
                worst, where = total, (float(dh), float(a1))
    assert worst <= eff, (
        f"at dh={where[0]:.3f} a1={where[1]:.3f} the brake and the cantilever together demand "
        f"{worst:.1f} Nm against a {eff:.0f} Nm ceiling -- _DH_V_MAX={_DH_V_MAX} is too fast; "
        f"the largest cap that fits is {_DH_V_MAX * (eff - (worst - kd * _DH_V_MAX * 9.870)) / (kd * _DH_V_MAX * 9.870):.4f}")
    # Not just "under": a cap far under the ceiling is retreat speed thrown away for nothing.
    assert worst > 0.9 * eff, (
        f"the worst corner only reaches {worst:.1f} of {eff:.0f} Nm -- _DH_V_MAX={_DH_V_MAX} is "
        f"leaving more than 10% of the joint's usable rate unused; re-derive it")


# --------------------------------------------------------------------------- morph/arm/timing.py
def _profile(t, q):
    """(max |velocity| per joint, max |acceleration| per joint) off the returned samples."""
    v = np.diff(q, axis=0) / np.diff(t)[:, None]
    a = np.diff(v, axis=0) / np.diff(t)[:-1, None]
    return np.abs(v).max(axis=0), (np.abs(a).max(axis=0) if len(a) else np.zeros(q.shape[1]))


def _dist_to_seg(p, a, b):
    """Distance from point `p` to the segment [a, b], any dimension (no 3-D-only cross product)."""
    d = b - a
    l2 = float(d @ d)
    if l2 == 0.0:
        return float(np.linalg.norm(p - a))
    t = min(1.0, max(0.0, float((p - a) @ d) / l2))
    return float(np.linalg.norm(p - (a + t * d)))


def test_the_output_never_leaves_a_segment_of_the_collapsed_waypoints():
    """The EXACT-PATH property `retime` actually has (module docstring, WHY TRAPEZOIDAL): unlike a
    TOTG-style blend, no sample is ever OFF the straight line between two consecutive waypoints of
    the (collapsed) input polyline. Checked by point-to-segment distance, not just by trusting the
    `a + s*delta` construction."""
    path = np.array([[0.0, 0.0], [0.4, -0.3], [0.9, 0.4], [0.2, 0.6], [-0.3, -0.1]])
    way = _collapse(path)
    _t, q = retime(path, [0.37, 0.29], [3.0, 2.0], DT)
    for row in q:
        best = min(_dist_to_seg(row, way[i], way[i + 1]) for i in range(len(way) - 1))
        assert best <= 1e-9, f"{row} is {best:.3e} off every segment of {way.tolist()}"


def test_a_proper_corner_waypoint_is_hit_bit_for_bit():
    """A real corner (not collinear with its neighbours) is NOT merged by `_collapse`, so it must
    appear exactly: a caller checkpointing against the planner's own polyline needs that exact
    vertex, not a nearby sample."""
    path = np.array([[0.0, 0.0], [0.4, 0.3], [1.1, -0.2]])          # a real corner at index 1
    _t, q = retime(path, [0.3, 0.3], [2.0, 2.0], DT)
    assert any(np.array_equal(row, path[1]) for row in q), "the corner never appears exactly"


def test_a_near_collinear_interior_waypoint_is_matched_only_within_the_collapse_tolerance():
    """The one caveat to the two tests above, and to the module docstring's EXACT-PATH claim:
    `_collapse` merges an interior point that is collinear with its neighbours to within a 1e-9
    cosine tolerance -- not exactly collinear, only close (empirically, about 1e-5 m of
    perpendicular offset on a 1 m segment already merges). So "never left" is not bit-for-bit for
    such a point: the retimed path passes NEAR it, not through its exact coordinates. Deliberately
    not asserting zero deviation here -- that would make the earlier claim, not this one."""
    a, c = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    b = np.array([0.5, 1e-6])                       # well inside the merge boundary (~1.1e-5)
    way = _collapse(np.array([a, b, c]))
    assert len(way) == 2, "this offset no longer merges; pick one further inside the boundary"
    _t, q = retime(np.array([a, b, c]), [1.0, 1.0], [4.0, 4.0], DT)
    closest = min(float(np.linalg.norm(row - b)) for row in q)
    assert 0.0 < closest < 1e-5, (
        f"b is {closest:.3e} from the nearest sample -- merged (not exact) but should still be "
        f"close, or the merge tolerance moved and this needs re-picking")


def test_a_single_waypoint_is_a_zero_length_trajectory():
    t, q = retime(np.array([[0.3, -0.2]]), [1.0, 1.0], [1.0, 1.0], DT)
    assert t.shape == (1,) and float(t[0]) == 0.0, t
    assert np.allclose(q, [[0.3, -0.2]]), q


def test_a_triangular_segment_matches_the_closed_form():
    """Short enough that the ramp never reaches v_max: T = 2*sqrt(L/a)."""
    t, _q = retime(np.array([[0.0], [0.01]]), [1.0], [1.0], DT)
    ideal = 2.0 * math.sqrt(0.01 / 1.0)
    assert ideal <= t[-1] <= ideal + DT, f"{t[-1]} not within one step of {ideal}"


def test_a_trapezoidal_segment_matches_the_closed_form():
    """Long enough to cruise: T = L/v + v/a."""
    t, _q = retime(np.array([[0.0], [1.0]]), [0.5], [1.0], DT)
    ideal = 1.0 / 0.5 + 0.5 / 1.0
    assert ideal <= t[-1] <= ideal + DT, f"{t[-1]} not within one step of {ideal}"


def test_no_joint_exceeds_its_velocity_limit():
    path = np.array([[0.0, 0.0], [0.4, -0.3], [0.9, 0.1], [0.2, 0.6], [1.1, 0.0]])
    vmax, amax = np.array([0.3634, 0.1118]), np.array([110.2, 142.3])
    _t, q = retime(path, vmax, amax, DT)
    v, _a = _profile(_t, q)
    assert np.all(v <= vmax * (1.0 + 1e-9)), f"velocity {v} over {vmax}"


def test_no_joint_exceeds_its_acceleration_limit():
    path = np.array([[0.0, 0.0], [0.4, -0.3], [0.9, 0.1], [0.2, 0.6], [1.1, 0.0]])
    vmax, amax = np.array([0.3634, 0.1118]), np.array([4.0, 3.0])
    t, q = retime(path, vmax, amax, DT)
    _v, a = _profile(t, q)
    assert np.all(a <= amax * (1.0 + 1e-9)), f"acceleration {a} over {amax}"


def test_the_duration_does_not_depend_on_the_waypoint_count():
    """THE DEFECT, stated as a property. `per = int(secs/dt/total)` makes commanded speed a
    function of how many vertices OMPL returned; subdividing a straight run must change nothing."""
    coarse = np.array([[0.0, 0.0], [1.0, 0.5]])
    fine = np.array([[0.0, 0.0], [0.25, 0.125], [0.5, 0.25], [0.75, 0.375], [1.0, 0.5]])
    a, b = retime(coarse, [0.3, 0.3], [1.0, 1.0], DT)[0][-1], retime(fine, [0.3, 0.3], [1.0, 1.0], DT)[0][-1]
    assert abs(a - b) <= 1e-12, f"5 collinear waypoints take {b} where 2 take {a}"


def test_a_longer_path_takes_longer():
    """The direct inverse of the measured symptom: today a longer path executes FASTER, because it
    is the same `secs` budget over more distance."""
    short = retime(np.array([[0.0], [1.0]]), [0.3], [1.0], DT)[0][-1]
    long_ = retime(np.array([[0.0], [2.0]]), [0.3], [1.0], DT)[0][-1]
    assert long_ > short + 1.0, f"2 m took {long_}, 1 m took {short}"


def test_raising_the_limits_shortens_the_trajectory():
    """Guards the trivial pass: a `retime` that just returns a very slow ramp satisfies every
    bound above."""
    slow = retime(np.array([[0.0], [1.0]]), [0.3], [1.0], DT)[0][-1]
    fast = retime(np.array([[0.0], [1.0]]), [1.2], [16.0], DT)[0][-1]
    assert fast < slow / 2.0, f"4x the limits gave {fast} against {slow}"


def test_the_endpoints_are_exact_and_the_ends_are_at_rest():
    """At rest means the FIRST and LAST step are reachable from zero velocity at a_max -- a sampled
    trajectory that truly started at rest still has a non-zero first difference."""
    path = np.array([[0.0, 0.0], [0.4, -0.3], [1.1, 0.2]])
    amax = np.array([2.0, 2.0])
    t, q = retime(path, [0.3, 0.3], amax, DT)
    assert np.allclose(q[0], path[0], atol=1e-12) and np.allclose(q[-1], path[-1], atol=1e-12)
    assert np.all(np.abs(q[1] - q[0]) / DT <= amax * DT * (1 + 1e-9)), f"starts moving: {q[1] - q[0]}"
    assert np.all(np.abs(q[-1] - q[-2]) / DT <= amax * DT * (1 + 1e-9)), "ends moving"


def test_the_last_sample_is_the_goal_bit_for_bit():
    """`a + 1.0 * (b - a)` is not always `b`. For a joint sitting near zero and moving under a
    millimetre -- what a fine IK correction looks like -- it lands one ulp short, and the caller
    grades arrival by comparing the achieved q8 against the goal. Snapped, not left to rounding.
    (Within the ordinary joint range the rounding happens to be exact, which is why this needs a
    pose picked to expose it rather than any old path.)"""
    a, b = -7.17531888904428e-06, 0.0009763492072862867
    assert a + (b - a) != b, "this pair no longer exposes the rounding; pick another"
    _t, q = retime(np.array([[a, 0.5], [b, 0.5]]), [0.3, 0.3], [2.0, 2.0], DT)
    assert q[-1][0] == b, f"{q[-1][0]!r} is not {b!r}"


def test_repeated_waypoints_do_not_stall_the_trajectory():
    """A planner that emits a duplicate vertex must not divide by zero or add dead time."""
    clean = np.array([[0.0], [1.0]])
    dupes = np.array([[0.0], [0.0], [1.0], [1.0]])
    assert abs(retime(clean, [0.3], [1.0], DT)[0][-1]
               - retime(dupes, [0.3], [1.0], DT)[0][-1]) <= 1e-12


def test_the_samples_land_on_the_control_grid():
    """The executor steps physics at a fixed dt; samples between steps cannot be commanded."""
    t, q = retime(np.array([[0.0, 0.0], [0.4, -0.3], [1.1, 0.2]]), [0.3, 0.3], [2.0, 2.0], DT)
    assert len(t) == len(q)
    assert np.allclose(np.diff(t), DT, atol=1e-12), np.diff(t)[:5]


def test_the_limits_are_applied_per_joint_and_not_as_one_scalar():
    """Q8 mixes metres and radians (armfk.py:89). A single scalar bound over all eight is the
    documented way to get this wrong."""
    path = np.array([[0.0, 0.0], [1.0, 1.0]])
    t, q = retime(path, [1.0, 0.05], [10.0, 10.0], DT)
    v, _a = _profile(t, q)
    assert v[1] <= 0.05 * (1 + 1e-9), f"the slow joint ran at {v[1]}"
    assert v[0] <= 0.05 * (1 + 1e-9) + 1e-9, "the fast joint left the slow one behind"
    assert t[-1] >= 1.0 / 0.05, f"the pair finished in {t[-1]}, faster than its slowest joint"


def test_a_non_positive_limit_is_refused():
    """A zero limit is an infinite duration or a division by zero, either of which reaches the arm
    as a silent stall rather than an error."""
    for bad in ([0.0, 1.0], [-1.0, 1.0]):
        for kw in ("v", "a"):
            try:
                retime(np.array([[0.0, 0.0], [1.0, 1.0]]),
                       bad if kw == "v" else [1.0, 1.0],
                       bad if kw == "a" else [1.0, 1.0], DT)
            except ValueError:
                continue
            raise AssertionError(f"{kw}_max={bad} was accepted")


# -------------------------------------------------------- morph/arm/execute.py::execute_path
_Q8N = ArmModel.Q8
# A sparse plan with a real corner: the columns separate over the first leg (so `_DH_V_MAX`
# binds) and the turret swings back over the second (so the path cannot collapse to one line).
_SPARSE = np.array([[0.00, 0.000, 0.000, 0.300, 0.0, 0.0, 0.0, 0.0],
                    [0.40, 0.050, 0.160, 0.250, 0.1, 0.0, 0.0, 0.0],
                    [0.10, 0.150, 0.260, 0.200, 0.2, 0.0, 0.0, 0.0]])


def _CART():
    """`morph.arm.cartesian` late, so a suite run against a tree without it still fails HERE."""
    return __import__("morph.arm.cartesian", fromlist=["x"])


class _ExecDemo:
    """The `Demo` surface `execute_path` touches, recording every q8 it commands."""

    def __init__(self, blocked=None, replan=None):
        self.idx = {n: i for i, n in enumerate(_Q8N)}
        self.dt = DT
        self._f_idx = np.array([], int)
        self._finger_pin = None
        self._plan_fallbacks = {}
        self._safety_abort = False
        self._abort_recovered = False
        self._focus_obj = None
        self.commanded = []
        self.checkpoints = []            # commands flown when each checkpoint re-fetched obstacles
        self.q0 = np.zeros(len(_Q8N))
        self._q = _SPARSE[0].copy()
        self._blocked, self._replan = blocked, replan
        self.world = SimpleNamespace(step=lambda render: None)
        self.robot = SimpleNamespace(get_joint_positions=lambda: self._q.copy())
        self._model = SimpleNamespace(
            collides=lambda q8, *a, **k: (self._blocked is not None and float(
                np.linalg.norm(np.asarray(q8, float) - self._blocked)) < 1e-9),
            clearance=lambda *a, **k: 1.0)

    def _fallback(self, tag):
        self._plan_fallbacks[tag] = self._plan_fallbacks.get(tag, 0) + 1

    def _arm_model(self):
        return self._model

    def _commanded_fingers(self):
        return None

    def _planned_obstacles(self, *a, **k):
        self.checkpoints.append(len(self.commanded))
        return [], []

    def _plan_arm(self, goal, **k):
        return self._replan

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
        self.commanded.append(np.asarray(q, float).copy())
        self._q = np.asarray(q, float).copy()


def _flown(waypoints, demo=None, **kw):
    """(demo, every q8 the real `execute_path` commanded, waypoints[0] first)."""
    demo = _ExecDemo() if demo is None else demo
    glb = {"np": np, "os": os, "math": math, "MARGIN": 0.02, "HEADLESS": True, "ArmModel": ArmModel,
           "print": lambda *a, **k: None,
           "retime_q8": lambda *a, **k: _CART().retime_q8(*a, **k),
           "q8_derate": lambda *a, **k: _CART().q8_derate(*a, **k),
           "with_closure_forced": lambda demo, q: demo._with_closure_forced(q),
           "commanded_fingers": lambda demo: demo._commanded_fingers(),
           "planned_obstacles": lambda demo, *a, **k: demo._planned_obstacles(*a, **k),
           "blame": lambda *a, **k: None,
           "plan_arm": lambda demo, *a, **k: demo._plan_arm(*a, **k)}
    fn = _exec_method(("morph", "arm", "execute.py"), "execute_path", glb)
    glb["execute_path"] = fn
    fn(demo, [np.asarray(w, float) for w in waypoints], hold_fingers=False, **kw)
    return demo, np.array([np.asarray(waypoints[0], float)] + list(demo.commanded))


def _peaks(q):
    """(v/v_max, a/a_max, dh rate/_DH_V_MAX) peaks over a commanded sequence."""
    from morph.arm.cartesian import _A_MAX, _DH_V_MAX, _V_MAX
    d = np.diff(q, axis=0)
    return (float((np.abs(d) / DT / _V_MAX).max()),
            float((np.abs(np.diff(d, axis=0)) / DT ** 2 / _A_MAX).max()) if len(d) > 1 else 0.0,
            float((np.abs(np.diff(q[:, 2] - q[:, 1])) / DT / _DH_V_MAX).max()))


def test_the_executor_commands_nothing_past_the_measured_joint_limits():
    """WHY THIS FILE EXISTS, stated as a behaviour: every pose `execute_path` writes is one the
    joints can actually reach in one control step, and one whose CHANGE the drives can deliver.

    Mutations: restore the `per = int(secs/dt/total)` cosine ease (velocity and acceleration both
    blow past 1.0); double the ceilings in the executor's call (velocity 2.0); drop the augmented
    `dh` row (the differential runs at ~5x its cap while both columns stay inside their own)."""
    _d, q = _flown(_SPARSE, secs=2.0)
    v, a, dh = _peaks(q)
    assert v <= 1.0 + 1e-9, f"commanded {v:.2f} x the velocity ceiling"
    assert a <= 1.0 + 1e-9, f"commanded {a:.2f} x the acceleration ceiling"
    assert dh <= 1.0 + 1e-9, f"the column differential ran at {dh:.2f} x its cap"


def test_the_bound_is_not_bought_by_flying_everything_slowly():
    """Guards the trivial pass: a retime that simply crawls satisfies every ceiling. Without a
    `secs` floor at least one of the three must actually BIND."""
    _d, q = _flown(_SPARSE)
    v, a, dh = _peaks(q)
    assert max(v, a, dh) > 0.99, f"nothing binds: v {v:.2f} a {a:.2f} dh {dh:.2f}"


def test_a_caller_supplied_secs_is_a_floor_and_never_a_ceiling():
    """`secs` buys a SLOWER motion for a caller that wants one (the insert ramps ask for 1.0 and
    1.2 s), and cannot buy a faster one than the joints allow.

    Mutation: let `secs` set the duration outright -- the short ask then finishes in 0.01 s."""
    _d, free = _flown(_SPARSE)
    _d, slow = _flown(_SPARSE, secs=4.0)
    _d, rushed = _flown(_SPARSE, secs=0.01)
    assert (len(slow) - 1) * DT >= 4.0, f"a 4.0 s ask ran in {(len(slow) - 1) * DT:.3f} s"
    assert len(rushed) == len(free), (
        f"a 0.01 s ask shortened the motion from {len(free)} to {len(rushed)} commands")
    v, a, dh = _peaks(slow)
    assert max(v, a, dh) <= 1.0 + 1e-9, f"the stretched motion still breached: {v} {a} {dh}"


def test_the_replan_splice_is_retimed_too():
    """A blocked checkpoint splices a fresh plan into the remainder mid-motion. Flying that tail
    on the timing derived before the splice is the branch a straightforward change misses.

    Mutations: keep the samples computed for the original path and write them after the splice;
    and drop the re-derate, which leaves every bound true and misses the floor instead."""
    blocked = _SPARSE[2]
    new = np.array([_SPARSE[1], _SPARSE[1] + np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])])
    demo = _ExecDemo(blocked=blocked, replan=[w.copy() for w in new])
    # 20.0 s: high enough that the floor BINDS on both paths, so the original's derate (0.5413)
    # and the spliced path's (0.8096) differ and dropping the re-derate is visible.
    _d, q = _flown(_SPARSE, demo=demo, secs=20.0, checkpoint_every=1)
    assert demo._plan_fallbacks == {}, demo._plan_fallbacks
    assert np.allclose(q[-1], new[-1]), "the spliced remainder was not flown"
    v, a, dh = _peaks(q)
    assert max(v, a, dh) <= 1.0 + 1e-9, f"the spliced tail breached: v {v} a {a} dh {dh}"
    # ...and the tail carries the derate the SPLICED path asks for, not the one the original did.
    # Dropping the re-derate leaves the bounds true and the floor missed, so bounds cannot see it.
    from morph.arm.cartesian import q8_derate, retime_q8
    spliced = np.array([_SPARSE[0], _SPARSE[1], new[1]])
    want = (len(retime_q8(_SPARSE[0:2], DT, q8_derate(_SPARSE, 20.0))[1]) - 1
            + len(retime_q8(spliced[1:3], DT, q8_derate(spliced, 20.0))[1]) - 1)
    assert len(q) - 1 == want, f"{len(q) - 1} commands against the {want} the re-derate owes"


# The shipped column ramp's own shape: both columns by `dz` and nothing else, so the `dh` row is
# slack and the columns' own acceleration binds -- a TRIANGLE. `_SPARSE` cannot see that half.
_RAMP = np.array([np.zeros(8), np.array([0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])])


def test_an_acceleration_bound_ramp_keeps_the_whole_duration_it_asked_for():
    """The insert raise and lower are pure column deltas asking for 1.0 s and 1.2 s with the
    payload in the hand, and their free profile is a TRIANGLE -- duration 2/sqrt(sa), where the
    derate enters as sqrt(k**2) and nothing else. `_SPARSE` is velocity-bound, where the duration
    is 1/sv and the acceleration derate cancels, so it cannot see this at all.

    Mutation: derate acceleration by `k` instead of `k**2` -- the 1.0 s ask flies in 0.654 s."""
    from morph.arm.cartesian import _A_MAX, _V_MAX
    dz = float(_RAMP[1][1])
    assert (_V_MAX[1] / dz) ** 2 >= _A_MAX[1] / dz, "the fixture stopped being a triangle"
    for secs in (1.0, 1.2):
        _d, q = _flown(_RAMP, secs=secs, checkpoint_every=len(_RAMP) + 1)
        flown = (len(q) - 1) * DT
        assert flown >= secs, f"a {secs}s ask flew in {flown:.4f}s"
        assert flown < secs + 4 * DT, f"a {secs}s ask flew in {flown:.4f}s, well over the ask"
        assert np.allclose(q[-1], _RAMP[-1]), "the ramp did not land on its endpoint"
        v, a, dh = _peaks(q)
        assert max(v, a, dh) <= 1.0 + 1e-9, f"{secs}s: v {v} a {a} dh {dh}"
    # The pre-dock trim asks 0.25 s for a delta this size and cannot have it: a floor never
    # shortens a motion, so it flies at its own 0.429 s. The operator's cycle reads that stage.
    _d, q = _flown(_RAMP, secs=0.25, checkpoint_every=len(_RAMP) + 1)
    assert 0.42 < (len(q) - 1) * DT < 0.44, f"the free duration moved to {(len(q) - 1) * DT:.4f}s"


def test_the_stretched_ramp_is_a_trapezoid_and_not_the_cosine_it_replaced():
    """The profile the drives actually receive on the carry. A cosine ease peaks at pi/2 times
    its mean rate; this triangle peaks at 2x. Restoring the cosine, or any blend of the two,
    moves that ratio -- which is the whole behaviour change on the insert ramps."""
    _d, q = _flown(_RAMP, secs=1.0, checkpoint_every=len(_RAMP) + 1)
    rate = np.abs(np.diff(q[:, 1])) / DT
    mean = float(np.abs(q[-1, 1] - q[0, 1])) / ((len(q) - 1) * DT)
    assert abs(rate.max() / mean - 2.0) < 0.02, (
        f"peak/mean {rate.max() / mean:.3f}; a triangle is 2.0, the deleted cosine 1.571")


def test_the_checkpoint_cadence_is_still_every_checkpoint_every_waypoints():
    """`i` advances a whole chunk at a time, so `if i:` IS the `checkpoint_every` cadence -- but
    only because the chunk end is `i + every`. Chunking one waypoint at a time leaves every
    assertion about bounds true and costs the reach 46 % of its wall clock and 8x its obstacle
    rebuilds.

    Mutation: end the chunk at `i + 1`."""
    wps = np.linspace(_SPARSE[0], _SPARSE[2], 18)
    demo, _q = _flown(wps, checkpoint_every=8)
    assert demo.checkpoints == [_n_between(wps, 0, 8), _n_between(wps, 0, 16)], (
        f"checkpoints fired after {demo.checkpoints} commands, not at waypoints 8 and 16")


def _n_between(wps, i, j, secs=None, every=8):
    """Commands the executor's own chunking owes between waypoints `i` and `j`, recomputed."""
    from morph.arm.cartesian import q8_derate, retime_q8
    k = q8_derate(wps, secs)
    n, c = 0, i
    while c < j:
        nxt = min(len(wps) - 1, c + every)
        n += len(retime_q8(wps[c:nxt + 1], DT, k)[1]) - 1
        c = nxt
    return n


def test_the_derate_is_sized_on_the_whole_path_not_on_the_first_interval():
    """`secs` is the whole motion's floor. Sizing the derate off one interval makes every later
    chunk carry a factor derived from a leg it is not on -- invisible on any 3-waypoint fixture,
    because there the first interval IS the path.

    Mutation: `q8_derate(waypoints[:every + 1], secs)`."""
    wps = np.linspace(_SPARSE[0], _SPARSE[2], 18)
    demo, q = _flown(wps, secs=40.0, checkpoint_every=8)
    assert len(q) - 1 == _n_between(wps, 0, len(wps) - 1, secs=40.0), (
        f"{len(q) - 1} commands against the {_n_between(wps, 0, len(wps) - 1, secs=40.0)} the "
        f"whole-path derate owes")
    assert (len(q) - 1) * DT >= 40.0, f"the floor was missed: {(len(q) - 1) * DT:.3f}s"


def test_an_interval_whose_endpoints_coincide_writes_nothing():
    """A semantics change worth naming: the deleted `per = max(1, ...)` guaranteed one write per
    segment, so a zero-delta ramp used to burn `secs` holding. It now moves nothing and steps the
    simulator zero times, and still reports the motion complete."""
    still = np.array([_SPARSE[0], _SPARSE[0].copy()])
    demo, q = _flown(still, secs=1.0)
    assert len(q) == 1 and demo.commanded == [], f"{len(demo.commanded)} commands for no motion"


def test_retiming_an_already_retimed_path_is_a_fixed_point():
    """`plan_cartesian` hands `execute_path` a path it has ALREADY retimed, and the executor
    retimes it again. Unless that is a fixed point, every straight line the retreat and the insert
    fly silently changes duration between the plan and the write."""
    from morph.arm.cartesian import retime_q8
    once = retime_q8(_SPARSE, DT)[1]
    twice = retime_q8(once, DT)[1]
    assert len(twice) == len(once), f"{len(once)} samples retimed to {len(twice)}"
    assert np.allclose(twice, once, atol=1e-9), np.abs(twice - once).max()


def test_one_control_step_at_the_velocity_ceiling_stays_under_the_jump_mask():
    """`_apply` reads a step over 5 mm as a snap and ZEROES the velocity feedforward on that
    joint, which is a lag, not an error anything reports (`morph/robot.py`). The executor never
    reaches it -- but only because the ceilings say so, so the implication is pinned here."""
    from morph.arm.cartesian import _V_MAX
    worst = float(_V_MAX[ArmModel.Q8_LIN].max()) * DT
    assert worst < 0.005, f"one step at the ceiling is {worst * 1000:.2f} mm, past the 5 mm mask"


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
