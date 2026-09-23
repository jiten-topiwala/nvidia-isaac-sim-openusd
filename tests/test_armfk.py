"""Model-level checks for `morph.arm.model`: no Isaac, no GPU, any Python with numpy. Self-consistency
only -- `tests/test_armfk_live.py` is what proves the model matches the real articulation.
    .venv/bin/python3 -m pytest tests/test_armfk.py -v"""
import ast
import inspect
import itertools
import json
import math
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.fingers_recording import FINGERS_RECORDING                              # noqa: E402
from morph.arm.model import ArmModel, HELD_LINK                                        # noqa: E402
from morph.arm.collision import (FINGER_SWEEP_AABB, Q8_TOL_M, Q8_TOL_RAD,          # noqa: E402
                                 _aabb_prunes, _obb_hits_aabb, _obb_hits_obb)
from morph.geometry import aabb_in_frame                                  # noqa: E402

M = ArmModel.load(os.path.join(ROOT, "usd/_arm_model.json"),
                  os.path.join(ROOT, "usd/_closure_lut.json"))


def test_passives_match_lut_grid_corner():
    lut = json.load(open(os.path.join(ROOT, "usd/_closure_lut.json")))
    dh0, a10 = lut["dh"][0], lut["a1"][0]
    p = M.passives(dh0, a10)
    for k, name in enumerate(lut["passives"]):
        assert abs(p[name] - lut["grid"][0][0][k]) < 1e-9


def test_fk_is_finite_and_hand_is_above_base_at_park():
    q = np.array([0.0, 1.2, 1.2, 0.1, 0.0, 0.0, 0.0, 0.0])
    poses = M.fk(q)
    assert set(M.link_aabb) <= set(poses)
    hand_z = poses["Gripper_Link3_1"][0][2]
    assert np.isfinite(hand_z) and hand_z > 1.0


def test_raising_columns_raises_hand_one_to_one():
    q = np.array([0.0, 0.5, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0])
    z0 = M.fk(q)["Gripper_Link3_1"][0][2]
    q[1] += 0.10
    # equal column move = rigid lift
    q[2] += 0.10
    z1 = M.fk(q)["Gripper_Link3_1"][0][2]
    assert abs((z1 - z0) - 0.10) < 1e-6


def test_turret_rotates_hand_about_mount_not_origin():
    q = np.array([0.0, 0.5, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0])
    p0 = M.fk(q)["Gripper_Link3_1"][0]
    q[0] = math.pi / 2
    p1 = M.fk(q)["Gripper_Link3_1"][0]
    mount = np.array([0.15, 0.15, 0.0])                         # BaseJoint_1 localPos0 xy
    r0, r1 = np.linalg.norm((p0 - mount)[:2]), np.linalg.norm((p1 - mount)[:2])
    assert abs(r0 - r1) < 1e-6 and abs(p0[2] - p1[2]) < 1e-9


def test_collision_against_a_blocking_slab():
    q = np.array([0.0, 0.5, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0])
    hand = M.fk(q)["Gripper_Link3_1"][0]
    slab = (np.array([[hand[0] - 0.05, -1.0, hand[2] - 0.05],
                      [hand[0] + 0.05, 1.0, hand[2] + 0.05]]), "world")
    assert M.collides(q, [slab]) is True
    far = (np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]]), "world")
    assert M.collides(q, [far]) is False


def test_column_group_ignores_chassis_tag():
    # Folded high, so the only link over the plate is the mast Arm_1 -- a column-group link,
    # bolted to the chassis and never a collision with it.
    q = np.array([0.0, 1.2, 1.2, 0.3, 0.0, 0.0, 0.0, 0.0])
    plate = np.array([[-0.35, -0.30, 0.0], [0.35, 0.30, 0.25]])
    over = [k for k, b in M.link_aabbs(q).items()
            if np.all(b[0] - 0.02 <= plate[1]) and np.all(b[1] + 0.02 >= plate[0])]
    assert over == ["Arm_1"], over
    assert M.collides(q, [(plate, "chassis")]) is False
    assert M.collides(q, [(plate, "world")]) is True


def test_lut_domain_gate():
    assert M.in_lut_domain(0.0, 0.3) is True
    # dh above the grid
    assert M.in_lut_domain(0.5, 0.3) is False


# `in_lut_domain` does not imply reachable: the domain is a superset of what the linkage adopts.


def test_only_the_prismatic_passive_has_a_checkable_limit():
    """Only `ArmRightJoint_1` is bounded ([-0.525, +0.100] m); the LUT grid runs to -0.8412."""
    assert [n for n, _, _ in M.passive_lims] == ["ArmRightJoint_1"]
    grid = np.array(M.lut["grid"], float)
    for k, n in enumerate(M.lut["passives"]):
        lo, hi = M.joints[n]["lower"], M.joints[n]["upper"]
        inside = bool(grid[:, :, k].min() >= lo and grid[:, :, k].max() <= hi)
        assert inside is (n != "ArmRightJoint_1"), n


def test_the_shipped_poses_all_survive_the_passive_rule():
    """The passive rule invalidates 9.6% of the in-LUT envelope, and nothing shipped."""
    K = json.load(open(os.path.join(ROOT, "usd/_grasp_known.json")))["arm_joints"]
    dh = K["ColumnRightBearingJoint_1"] - K["ColumnLeftBearingJoint_1"]
    assert M.in_lut_domain(dh, K["ArmLeftJoint_1"])
    assert M.passive_violations(dh, K["ArmLeftJoint_1"]) == []
    cols = ("ColumnLeftBearingJoint_1", "ColumnRightBearingJoint_1", "ArmLeftJoint_1")
    for name in ("reach", "lift", "place_low", "place_mid", "place_high"):
        tr = json.load(open(os.path.join(ROOT, f"usd/_traj_{name}.json")))
        ix = {n: tr["joints"].index(n) for n in cols}
        for k, fr in enumerate(tr["frames"]):
            dh = fr[ix[cols[1]]] - fr[ix[cols[0]]]
            a1 = fr[ix[cols[2]]]
            assert M.in_lut_domain(dh, a1), (name, k)
            assert M.passive_violations(dh, a1) == [], (name, k, dh, a1)


# `usd/_arm_model.json` bounds the fingers at the USD REST pose; a plan runs with the hand OPEN.

# Finger pad-box colliders in each link's own frame, from `usd/market_world_m1/payloads/base.usda`.
PAD_AABB = {
    "finger_a_link_1_1": [[0.00417, -0.02483, -0.0105], [0.04483, 0.00733, 0.0105]],
    "finger_a_link_2_1": [[0.00210, 0.00140, -0.0105], [0.03290, 0.00854, 0.0105]],
    "finger_a_link_3_1": [[-0.00696, 0.00072, -0.0091], [0.02096, 0.02098, 0.0091]],
    "finger_b_link_1_1": [[0.00417, -0.02273, -0.0091], [0.04483, 0.00733, 0.0091]],
    "finger_b_link_2_1": [[0.00210, 0.00140, -0.0091], [0.03290, 0.00854, 0.0091]],
    "finger_b_link_3_1": [[-0.00696, 0.00072, -0.0091], [0.02096, 0.02098, 0.0091]],
    "finger_c_link_1_1": [[0.00417, -0.02273, -0.0091], [0.04483, 0.00733, 0.0091]],
    "finger_c_link_2_1": [[0.00210, 0.00140, -0.0091], [0.03290, 0.00854, 0.0091]],
    "finger_c_link_3_1": [[-0.00696, 0.00072, -0.0091], [0.02096, 0.02098, 0.0091]],
}
FINGER_JOINTS = sorted(FINGERS_RECORDING)
# How far the finger MESH colliders exceed the pad boxes; their points live in a binary .usd.
MESH_ALLOWANCE = 0.0113


def _finger_envelope(fingers, q8=None):
    """[min, max] of every finger pad box in the HAND link's frame, at one finger configuration."""
    q8 = np.array([0.0, 0.60, 0.60, 0.25, 0.0, 0.0, 0.0, 0.0]) if q8 is None else q8
    dh = float(q8[2] - q8[1])
    pas = dict(M.passives(dh, float(q8[3])))
    pas.update(fingers)
    poses = M.fk(q8, passives=pas)
    hp, hR = poses[M.hand_link]
    pts = []
    for link, ab in PAD_AABB.items():
        p, R = poses[link]
        p, R = hR.T @ (p - hp), hR.T @ R
        corners = np.array([[x, y, z] for x in (ab[0][0], ab[1][0])
                            for y in (ab[0][1], ab[1][1]) for z in (ab[0][2], ab[1][2])], float)
        pts.append((R @ corners.T).T + p)
    a = np.vstack(pts)
    return np.vstack([a.min(axis=0), a.max(axis=0)])


def test_the_finger_envelope_does_not_depend_on_the_arm():
    """In the hand's own frame the fingers are where they are, whatever the arm does."""
    a = _finger_envelope(FINGERS_RECORDING)
    b = _finger_envelope(FINGERS_RECORDING,
                         np.array([1.1, 0.20, 0.45, 0.30, -0.6, 0.9, -0.4, 0.3]))
    assert np.abs(a - b).max() < 1e-6, (a, b)


def test_the_shipped_json_bounds_the_fingers_at_REST_and_only_at_REST():
    """Symmetric in x is the rest pose alone; the open pose leaves the shipped box by 36 mm."""
    raw = np.asarray(json.load(open(os.path.join(ROOT, "usd/_arm_model.json")))
                     ["link_aabb"]["Gripper_Link3_1"], float)
    assert abs(raw[0][0] + raw[1][0]) < 1e-5, raw
    rest = _finger_envelope({n: 0.0 for n in FINGER_JOINTS})
    assert np.all(rest[0] >= raw[0] - 1e-3) and np.all(rest[1][:2] <= raw[1][:2] + 1e-3), rest
    op = _finger_envelope(FINGERS_RECORDING)
    out_lo, out_hi = (raw[0] - op[0]) * 1000.0, (op[1] - raw[1]) * 1000.0
    assert round(float(out_hi[0]), 1) == 35.9 and round(float(out_lo[0]), 1) == 21.8, (out_lo, out_hi)
    assert max(float(out_hi[1]), float(out_hi[2]), float(out_lo[2])) < 0.0, (out_lo, out_hi)


def test_the_hand_box_now_bounds_the_fingers_at_every_pose_the_joints_allow():
    """`q8` has no finger DOF and the open pose is env-tunable, so the bound covers them all."""
    lim = [(n, M.joints[n]["lower"], M.joints[n]["upper"]) for n in FINGER_JOINTS]
    box = M.link_aabb[M.hand_link]
    rng = np.random.default_rng(0)
    worst = 0.0
    poses = [dict(zip(FINGER_JOINTS, c)) for c in itertools.product(*[(lo, hi) for _, lo, hi in lim])]
    poses += [{n: float(rng.uniform(lo, hi)) for n, lo, hi in lim} for _ in range(200)]
    for f in poses:
        e = _finger_envelope(f)
        assert np.all(e[0] - MESH_ALLOWANCE >= box[0] - 1e-9), (f, e[0], box[0])
        assert np.all(e[1] + MESH_ALLOWANCE <= box[1] + 1e-9), (f, e[1], box[1])
        worst = max(worst, float(np.max(np.maximum(box[0] - (e[0] - MESH_ALLOWANCE),
                                                   (e[1] + MESH_ALLOWANCE) - box[1]))))
    assert worst <= 0.0, worst


def test_the_finger_bound_is_the_only_thing_that_grew_the_hand_box():
    """It is a UNION, so no face may move inward: the palm's own geometry has to survive."""
    raw = np.asarray(json.load(open(os.path.join(ROOT, "usd/_arm_model.json")))
                     ["link_aabb"]["Gripper_Link3_1"], float)
    box = M.link_aabb[M.hand_link]
    assert np.all(box[0] <= raw[0]) and np.all(box[1] >= raw[1]), (raw, box)
    assert np.allclose(box[0], np.minimum(raw[0], FINGER_SWEEP_AABB[0]))
    assert np.allclose(box[1], np.maximum(raw[1], FINGER_SWEEP_AABB[1]))


def test_the_shipped_reach_still_clears_the_shelf_board_with_the_fingers_in():
    """Make-or-break: the shipped reach still clears the shelf board by 42.9 mm."""
    q8 = np.array([-0.572, 0.422, 0.480, 0.347, 0.020, -1.877, -0.431, 0.772])
    board = np.array([[-3.227, -2.594, 0.559], [3.903, -0.131, 0.572]])
    p, R = M.fk(q8)[M.hand_link]
    w = M._xform_aabb(M.link_aabb[M.hand_link], p, R)
    gap = float(np.max(np.maximum(board[0] - w[1], w[0] - board[1])))
    assert round(gap * 1000.0, 1) == 42.9, gap * 1000.0
    assert M.collides(q8, [(board, "world")], 0.02) is False


def test_the_finger_bound_changes_no_verdict_on_anything_the_robot_ships():
    """No verdict changes over the 406 shipped poses; worst hand clearance 88.2 -> 67.7 mm."""
    chassis = [(np.array([[-0.350, -0.300, 0.139], [0.350, 0.300, 0.159]]), "chassis"),
               (np.array([[-0.310, -0.290, 0.165], [-0.030, 0.270, 0.305]]), "chassis")]
    raw = np.asarray(json.load(open(os.path.join(ROOT, "usd/_arm_model.json")))
                     ["link_aabb"]["Gripper_Link3_1"], float)
    grown = M.link_aabb[M.hand_link].copy()
    cols = list(ArmModel.Q8)
    rows = [np.array([json.load(open(os.path.join(ROOT, "usd/_grasp_known.json")))
                      ["arm_joints"][n] for n in cols])]
    for name in ("reach", "lift", "place_low", "place_mid", "place_high"):
        tr = json.load(open(os.path.join(ROOT, f"usd/_traj_{name}.json")))
        ix = [tr["joints"].index(n) for n in cols]
        rows += [np.array([fr[i] for i in ix]) for fr in tr["frames"]]
    assert len(rows) == 406, len(rows)
    worst = [1e9, 1e9]
    try:
        for q in rows:
            for k, box in enumerate((raw, grown)):
                M.link_aabb[M.hand_link] = box
                p, R = M.fk(q)[M.hand_link]
                w = M._xform_aabb(box, p, R)
                for ob, _ in chassis:
                    worst[k] = min(worst[k], float(np.max(np.maximum(ob[0] - w[1], w[0] - ob[1]))))
            M.link_aabb[M.hand_link] = raw
            before = M.collides(q, chassis, 0.02)
            M.link_aabb[M.hand_link] = grown
            assert M.collides(q, chassis, 0.02) is before, q
    finally:
        M.link_aabb[M.hand_link] = grown
    assert [round(v * 1000.0, 1) for v in worst] == [88.2, 67.7], worst


def test_oriented_box_clears_what_the_axis_aligned_one_does_not():
    """The boom is a 0.82 m bar at turret -0.57 rad, so its AABB is far bigger than the bar."""
    q8 = np.array([-0.572, 0.422, 0.480, 0.347, 0.020, -1.877, -0.431, 0.772])
    board = (np.array([[-3.227, -2.594, 0.559], [3.903, -0.131, 0.572]]), "world")
    aabb = M.link_aabbs(q8)["Arm_Left_1"]
    lo, hi = aabb[0] - 0.02, aabb[1] + 0.02
    assert np.all(lo <= board[0][1]) and np.all(hi >= board[0][0])
    assert M.collides(q8, [board], 0.02) is False


# "world" boxes are tight in world, "chassis" boxes in the base frame; re-boxing inflates by yaw.

# The battery collider, [min, max] in the BASE LINK frame, 0.280 x 0.560 x 0.140 m, rigid to base.
BATTERY = np.array([[-0.310, -0.290, 0.165], [-0.030, 0.270, 0.305]])
# The real grasp configuration, in Q8 order -- what the pick replays.
_KNOWN = json.load(open(os.path.join(ROOT, "usd/_grasp_known.json")))
GRASP = np.array([float(_KNOWN["arm_joints"][n]) for n in ArmModel.Q8])
# Turret swung 30 deg, the size of an aim pass -- where the inflated box bites, unlike GRASP.
GRASP_AIMED = GRASP.copy()
GRASP_AIMED[0] -= math.radians(30)
BASE_T = np.array([5.0, -6.0, 0.30])                    # any pose; a rigid box does not care


def _base(yaw_deg, t=BASE_T):
    """(translation3, rotation3x3) of the base link, yawed. The form `_arm_base_world` returns."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.asarray(t, float), np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _world_aabb(box, base):
    """`box` re-boxed in world -- what `_arm_obstacles` used to hand over for a chassis collider."""
    t, R = base
    mn, mx = box
    corners = np.array([[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1])
                        for z in (mn[2], mx[2])], float)
    w = corners @ R.T + t
    return np.vstack([w.min(axis=0), w.max(axis=0)])


def test_world_aabb_of_a_chassis_box_is_inflated_by_the_base_yaw():
    """The premise the next test rests on: the re-boxing this fix removes is real and large."""
    got = [float(np.diff(_world_aabb(BATTERY, _base(d)), axis=0)[0][0]) for d in (0, 10, 20, 45)]
    # exact at yaw 0, by construction
    assert abs(got[0] - 0.280) < 1e-9, got
    assert got[1] > 0.37 and got[2] > 0.45 and got[3] > 0.59, got


def test_chassis_box_verdict_does_not_depend_on_the_base_yaw():
    """A box rigid to the base must give ONE verdict, whatever the base heading."""
    for name, q in (("grasp", GRASP), ("grasp+aim", GRASP_AIMED)):
        truth = M.collides(q, [(BATTERY, "chassis")], 0.02)     # base=None: the frames coincide
        for yaw in (0, 20, 45):
            got = M.collides(q, [(BATTERY, "chassis")], 0.02, None, _base(yaw))
            assert got is truth, f"{name} at yaw {yaw}: {got}, base frame says {truth}"


def test_q8_residue_separates_metres_from_radians():
    """Why the residue gate is two numbers: a single `max(abs(dq8))` mixes metres with radians,
    so a 49 mm column error read as "arrived" while being 159 mm of hand."""
    i1 = ArmModel.Q8.index("ColumnLeftBearingJoint_1")
    iw = ArmModel.Q8.index("gripper_x_rotation_1")
    col = GRASP.copy()
    col[i1] += 0.049
    # the old scalar: "arrived"
    assert float(np.max(np.abs(col - GRASP))) < 0.05
    rm, rr = ArmModel.q8_resid(col, GRASP)
    assert rm > Q8_TOL_M and rr == 0.0, (rm, rr)                 # the split: 49 mm of column
    hand = lambda q: M.fk(q)[M.hand_link][0]
    # ...and 159 mm of hand
    assert float(np.linalg.norm(hand(col) - hand(GRASP))) > 0.15
    # a pure wrist error is reported in radians and never as a length
    wr = GRASP.copy()
    wr[iw] += 0.10
    rm2, rr2 = ArmModel.q8_resid(wr, GRASP)
    assert rm2 == 0.0 and abs(rr2 - 0.10) < 1e-12, (rm2, rr2)
    # a matched pair: both thresholds are worth the same order of hand travel
    lin = GRASP.copy()
    lin[1] += Q8_TOL_M
    lin[2] += Q8_TOL_M
    rot = GRASP.copy()
    rot[0] += Q8_TOL_RAD
    d_lin = float(np.linalg.norm(hand(lin) - hand(GRASP)))
    d_rot = float(np.linalg.norm(hand(rot) - hand(GRASP)))
    assert 0.5 < d_rot / d_lin < 3.0, (d_lin, d_rot)


def _collides_pre_fix(q8, box, base, margin=0.02):
    """The pipeline this fix replaced: the chassis box re-boxed in WORLD, the links carried out
    to world. Written from FK and the SAT test because `collides` cannot give that verdict."""
    ob = _world_aabb(box, base)
    t, R = base
    for link, local, p, R_l in M._links(q8, None):
        if link in M.column_group:
            continue
        p_w, R_w = t + R @ p, R @ R_l
        b = M._xform_aabb(local, p_w, R_w)
        if not (np.all(b[0] - margin <= ob[1]) and np.all(b[1] + margin >= ob[0])):
            continue
        if _obb_hits_aabb(p_w + R_w @ ((local[0] + local[1]) * 0.5),
                          (local[1] - local[0]) * 0.5 + margin, R_w, ob):
            return True
    return False


def test_the_inflated_box_is_what_used_to_veto_the_aimed_grasp():
    """The verdict moved with the heading: the aimed neighbour is vetoed at yaw 45 alone."""
    assert M.collides(GRASP_AIMED, [(BATTERY, "chassis")], 0.02) is False
    assert [_collides_pre_fix(GRASP, BATTERY, _base(d)) for d in (0, 20, 45)] == [False] * 3
    assert [_collides_pre_fix(GRASP_AIMED, BATTERY, _base(d)) for d in (0, 20, 45)] == \
        [False, False, True]
    assert M.collides(GRASP_AIMED, [(BATTERY, "chassis")], 0.02, None, _base(45)) is False


def test_world_box_is_read_in_world_not_in_the_base_frame():
    """A "world" box must be tested where it is: same box, two placements, at a yawed base."""
    b = _base(40)
    t, R = b
    p_hand = M.fk(GRASP)[M.hand_link][0]
    blocking = np.vstack([t + R @ p_hand - 0.05, t + R @ p_hand + 0.05])   # around the WORLD hand
    assert M.collides(GRASP, [(blocking, "world")], 0.02, None, b) is True
    # the same box where the hand sits in the BASE frame: only a base-frame reading would hit
    elsewhere = np.vstack([p_hand - 0.05, p_hand + 0.05])
    assert M.collides(GRASP, [(elsewhere, "world")], 0.02, None, b) is False


def test_a_long_world_board_still_clears_the_mast_at_a_yawed_base():
    """Re-boxing this board into the base frame made a 5.4 m board a slab that hid the mast."""
    q8 = np.array([-0.572, 0.422, 0.480, 0.347, 0.020, -1.877, -0.431, 0.772])
    board = np.array([[-3.227, -2.594, 0.559], [3.903, -0.131, 0.572]])
    for yaw in (0, 20, 45):
        b = _base(yaw, t=np.zeros(3))
        assert M.collides(q8, [(board, "world")], 0.02, None, b) is False, yaw
        # the base-frame re-box really would have swallowed the arm, so the assert is not vacuous
        if yaw:
            t, R = b
            re_boxed = _world_aabb(board, (-R.T @ t, R.T))       # world -> base, re-aligned
            assert np.any(np.diff(re_boxed, axis=0)[0] > np.diff(board, axis=0)[0] + 0.5), yaw


IK_LO, IK_HI = M.bounds()
IK_LO[1], IK_HI[1] = 0.10, 1.30
IK_LO[3], IK_HI[3] = 0.05, 0.50
IK_LO[4:], IK_HI[4:] = -0.6, 0.6


def _reachable(rng):
    """Inside the working envelope AND adoptable -- the domain gate alone is not enough."""
    while True:
        q = rng.uniform(IK_LO, IK_HI)
        q[2] = min(max(q[1] + float(rng.uniform(M.lut["dh"][0], M.lut["dh"][-1])),
                       IK_LO[2]), IK_HI[2])
        if M.in_lut_domain(q[2] - q[1], q[3]) and not M.passive_violations(q[2] - q[1], q[3]):
            return q


def test_ik_round_trips_a_pose_the_arm_can_actually_reach():
    """IK from a PERTURBED seed -- seeding with the answer proves nothing."""
    rng = np.random.default_rng(0)
    solved, worst_p, worst_r = 0, 0.0, 0.0
    for _ in range(25):
        q = _reachable(rng)
        p, R = M.fk(q)[M.hand_link]
        seed = np.clip(q + rng.normal(0, 0.15, 8), IK_LO, IK_HI)
        got = M.ik(p, R, seed=seed, bounds=(IK_LO, IK_HI))
        if got is None:
            continue
        q2, _, er = got
        p2, _ = M.fk(q2)[M.hand_link]
        worst_p = max(worst_p, float(np.linalg.norm(p2 - p)))
        worst_r = max(worst_r, er)
        solved += 1
    assert solved >= 23, f"only {solved}/25 solved"
    assert worst_p < 0.001, f"worst position error {worst_p * 1000:.3f} mm"
    assert worst_r < 0.01, f"worst rotation error {math.degrees(worst_r):.3f} deg"


def test_ik_solutions_satisfy_the_closure():
    """Solutions must stay inside the LUT domain and must not ask a passive past its own stop."""
    rng = np.random.default_rng(1)
    for _ in range(10):
        q = _reachable(rng)
        p, R = M.fk(q)[M.hand_link]
        got = M.ik(p, R, bounds=(IK_LO, IK_HI))
        if got is not None:
            assert M.in_lut_domain(got[0][2] - got[0][1], got[0][3])
            assert M.passive_violations(got[0][2] - got[0][1], got[0][3]) == [], got[0]


def test_ik_reports_failure_instead_of_the_nearest_miss():
    assert M.ik(np.array([6.0, 6.0, 3.0]), bounds=(IK_LO, IK_HI)) is None


# The arm_plan.py subprocess: JSON in on stdin, JSON out on stdout, run by the OMPL venv.

VENV = os.environ.get("OMPL_PYTHON", os.path.join(ROOT, ".venv/bin/python3"))
PARK = [0.0, 1.2, 1.2, 0.1, 0.0, 0.0, 0.0, 0.0]
REACH = [0.0, 0.45, 0.55, 0.40, 0.0, 0.0, 0.0, 0.0]


def _plan(start, goal, obstacles=(), held=None, seed=0, solve_time=3.0, base=None,
          env=None, capture=False):
    req = {"start": list(start), "goal": list(goal),
           "model": os.path.join(ROOT, "usd/_arm_model.json"),
           "lut": os.path.join(ROOT, "usd/_closure_lut.json"),
           "obstacles": [[o[0].tolist(), o[1].tolist(), t] for o, t in obstacles],
           "held": None if held is None else [held[0].tolist(), held[1].tolist()],
           "base": None if base is None else [base[0].tolist(), base[1].tolist()],
           "margin": 0.02, "solve_time": solve_time, "seed": seed}
    _env = dict(os.environ)
    _env.update(env or {})
    out = subprocess.run([VENV, os.path.join(ROOT, "arm_plan.py")], input=json.dumps(req),
                         capture_output=True, text=True, timeout=120, env=_env)
    if not out.stdout.strip():
        raise AssertionError(f"planner produced nothing; stderr:\n{out.stderr[-2000:]}")
    parsed = json.loads(out.stdout.strip().splitlines()[-1])
    return (parsed, out.stderr) if capture else parsed


def test_plan_in_free_space_starts_and_ends_exactly():
    wp = _plan(PARK, REACH)
    assert isinstance(wp, list), wp
    assert len(wp) >= 2
    assert np.allclose(wp[0], PARK) and np.allclose(wp[-1], REACH)
    steps = np.max(np.abs(np.diff(np.array(wp), axis=0)), axis=1)
    # dense enough to execute directly
    assert steps.max() <= 0.02 + 1e-9


def test_every_waypoint_stays_in_lut_domain():
    for q in _plan(PARK, REACH):
        assert M.in_lut_domain(q[2] - q[1], q[3])


# In the LUT domain and inside the USD bounds, so only the passive rule can reject it.
PASSIVE_ONLY = [0.0, 0.300, 0.526, 0.44, 0.0, 0.0, 0.0, 0.0]


def test_every_waypoint_keeps_the_closure_inside_its_stops():
    for q in _plan(PARK, REACH):
        assert M.passive_violations(q[2] - q[1], q[3]) == [], q


def test_the_subprocess_applies_the_same_passive_rule_as_this_side():
    """The rule crosses an interpreter boundary as JSON, so both sides have to agree."""
    dh, a1 = PASSIVE_ONLY[2] - PASSIVE_ONLY[1], PASSIVE_ONLY[3]
    assert M.in_lut_domain(dh, a1)
    assert not M.collides(np.array(PASSIVE_ONLY), [])
    assert M.passive_violations(dh, a1)
    out = _plan(PARK, PASSIVE_ONLY)
    assert isinstance(out, dict) and "goal state invalid" in out.get("error", ""), out


def test_plan_routes_around_a_slab():
    # the slab sits at the mid-height the near-vertical PARK -> REACH travel passes through
    slab = (np.array([[0.62, -2.0, 0.78], [0.95, 2.0, 0.98]]), "world")
    assert not M.collides(PARK, [slab]) and not M.collides(REACH, [slab])
    line = [np.array(PARK) + (np.array(REACH) - np.array(PARK)) * t
            for t in np.linspace(0, 1, 101)]
    assert any(M.collides(q, [slab]) for q in line)
    wp = _plan(PARK, REACH, obstacles=[slab], solve_time=10.0)
    assert isinstance(wp, list), wp
    assert not any(M.collides(q, [slab]) for q in wp)


def test_the_planner_subprocess_reads_the_tags_in_the_same_frames():
    """Both cases use the SAME yawed base, so only the frame is doing the work."""
    b = _base(45, t=np.array([2.0, -3.0, 0.30]))
    assert M.collides(np.array(REACH), [(BATTERY, "chassis")], 0.02, None, b) is False
    wp = _plan(PARK, REACH, obstacles=[(BATTERY, "chassis")], base=b)
    # the real battery does not block it
    assert isinstance(wp, list), wp
    hand = M.fk(np.array(REACH))["Gripper_Link3_1"][0]         # base frame: a "chassis" box is too
    out = _plan(PARK, REACH, obstacles=[(np.array([hand - 0.05, hand + 0.05]), "chassis")],
                base=b, seed=1)
    assert isinstance(out, dict) and "error" in out, out


def test_blocked_goal_fails_cleanly():
    hand = M.fk(np.array(REACH))["Gripper_Link3_1"][0]
    box = (np.array([hand - 0.05, hand + 0.05]), "world")
    out = _plan(PARK, REACH, obstacles=[box], seed=1)
    assert isinstance(out, dict) and "error" in out, out


# `held` is an AABB in the HAND-LINK frame, posed at `hand_link` as the pseudo-link `__held__`;
# `morph.geometry.aabb_in_frame` produces it. OBJH / HR_MEAS are the measured in-hand relation.
OBJH = np.array([0.0015, 0.0507, 0.1703])
HR_MEAS = np.array([[-0.195, 0.016, 0.981],          # columns = the hand's x/y/z axes, in world
                    [-0.981, 0.004, -0.195],
                    [-0.007, -1.000, 0.015]])
# Object-local +z onto hand -Y -- the attitude HR_MEAS implies, pinned by the test below.
R_OBJ_IN_HAND = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _grasped(q8, r, half_h):
    """(local bound, world pose) of a held cylinder, plus the hand's world pose."""
    hp, hR = M.fk(q8)[M.hand_link]
    local = np.array([[-r, -r, -half_h], [r, r, half_h]])
    return (local, hp + hR @ OBJH, hR @ R_OBJ_IN_HAND), (hp, hR)


def test_measured_hand_rotation_puts_the_cylinder_axis_on_hand_minus_y():
    """What the logged hand rotation does to an upright cylinder."""
    axis_hand = HR_MEAS.T @ np.array([0.006, -0.0, 1.0])
    assert np.allclose(axis_hand, [-0.007, -1.0, 0.015], atol=0.01), axis_hand
    assert np.allclose(R_OBJ_IN_HAND[:, 2], [0.0, -1.0, 0.0])


def test_held_box_is_the_measured_relation_and_tracks_the_object_size():
    """Centre on the measured relation, extent from the LIVE object; both are run parameters."""
    for r, hh in ((0.040, 0.090), (0.055, 0.070), (0.075, 0.080)):
        (local, oc, oR), (hp, hR) = _grasped(GRASP, r, hh)
        held = aabb_in_frame(local, oc, oR, hp, hR)
        assert held.shape == (2, 3), held.shape
        assert np.allclose(held.mean(axis=0), OBJH, atol=1e-9), (r, hh, held.mean(axis=0))
        # y is the cylinder's axis in the hand frame, so y is the half-height and x/z the radius
        assert np.allclose(0.5 * np.diff(held, axis=0)[0], [r, hh, r], atol=1e-9)


def test_held_box_poses_back_onto_the_object_not_twice_the_hand():
    """`held` is HAND-LINK RELATIVE, so round-tripping it must land back on the object; an
    absolute-hand box lands 0.9 m off, and the (aabb, tag) tuple form must not be accepted."""
    (local, oc, oR), (hp, hR) = _grasped(GRASP, 0.040, 0.090)
    held = aabb_in_frame(local, oc, oR, hp, hR)
    got = M.link_aabbs(GRASP, held)["__held__"]
    assert np.allclose(got.mean(axis=0), oc, atol=1e-9), (got.mean(axis=0), oc)
    absolute = np.array([oc - [0.04, 0.04, 0.09], oc + [0.04, 0.04, 0.09]])
    bad = M.link_aabbs(GRASP, absolute)["__held__"].mean(axis=0)
    assert np.linalg.norm(bad - oc) > 0.5, bad
    try:
        M.link_aabbs(GRASP, (held, "held"))
        raise AssertionError("the (aabb, tag) tuple form must not be accepted")
    except (ValueError, TypeError):
        pass


def _tip_board(r=0.040, hh=0.090, half=0.010):
    """A small box at the far tip of the held cylinder, past the hand's own AABB in y and z."""
    (local, oc, oR), (hp, hR) = _grasped(GRASP, r, hh)
    held = aabb_in_frame(local, oc, oR, hp, hR)
    tip = oc + (hR @ np.array([0.0, 1.0, 0.0])) * (hh * 0.9)
    return held, (np.array([tip - half, tip + half]), "world")


def test_held_object_is_what_blocks_a_board_the_hand_clears():
    held, board = _tip_board()
    assert M.collides(GRASP, [board], 0.0) is False
    assert M.collides(GRASP, [board], 0.0, held) is True


def test_the_planner_subprocess_checks_the_held_box():
    """`held` must survive the JSON round trip into the OMPL venv."""
    held, board = _tip_board()
    ok = _plan(GRASP.tolist(), GRASP.tolist(), obstacles=[board], seed=1)
    assert isinstance(ok, list), ok
    bad = _plan(GRASP.tolist(), GRASP.tolist(), obstacles=[board], held=held, seed=1)
    assert isinstance(bad, dict) and "error" in bad, bad


def test_a_chassis_tagged_box_is_invisible_to_the_column_links():
    """The exemption that makes the parked arm's TAG load-bearing.

    `first_hit` skips "chassis" boxes for every link in `column_group`, because the columns
    legitimately overlap the real chassis and would otherwise veto every pose. That exemption is
    correct for the chassis and WRONG for anything else parked under `base/` -- notably the second
    arm, whose 1.64 m mast and 820 mm boom sit in exactly the volume arm-1's columns sweep.

    Pinning it as BEHAVIOUR: the identical box is invisible tagged "chassis" and solid tagged
    "world". If this ever stops being true the tag choice in `_arm_obstacles` stops mattering and
    someone will "simplify" it."""
    col = sorted(M.column_group)[0]
    lo, hi = np.asarray(M.link_aabb[col], float)
    poses = M.fk(GRASP)
    p, R = poses[col]
    centre = p + R @ ((lo + hi) / 2.0)
    half = (hi - lo) / 2.0
    box = np.vstack([centre - half * 0.5, centre + half * 0.5])   # squarely inside that column

    assert M.first_hit(GRASP, [(box, "chassis")]) != col, (
        f"{col} now SEES a chassis-tagged box -- the column exemption is gone, and the reason the "
        f"parked arm must be tagged 'world' has changed with it")
    assert M.first_hit(GRASP, [(box, "world")]) is not None, (
        "the same box tagged 'world' is also invisible, so this test proves nothing")


def test_the_parked_arm_is_an_obstacle_and_is_not_tagged_chassis():
    """Arm-2 never moves, so nothing plans FOR it -- but it is a solid body on the same chassis and
    arm-1 must not plan THROUGH it. Two ways that was silently untrue:

      1. `_arm_obstacles` skips prims whose `CollisionEnabledAttr` is False, and on the audited
         stage arm-2's mast, columns and booms are all COL_OFF while only its FINGERS are enabled.
         So the obstacle set held arm-2's fingertips and not its 1.64 m mast.
      2. Arm-2 lives under `base/`, so the tag expression would call it "chassis" -- which the
         test above shows is invisible to arm-1's columns.

    Structural, because `_arm_obstacles` walks a live USD stage and cannot run offline."""
    src = open(os.path.join(ROOT, "morph", "arm", "stage.py"), encoding="utf-8").read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "arm_obstacles")
    tags = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
            and any(getattr(t, "id", None) == "tag" for t in n.targets)]
    assert tags, "`_arm_obstacles` no longer computes a tag; this test is stale"
    src_tag = ast.dump(tags[0])
    assert "'world'" in src_tag.replace('"', "'"), "the tag expression lost its 'world' branch"
    # The parked-arm branch must come BEFORE the base_path test, or `base/Arm_2` lands on "chassis".
    body = ast.unparse(tags[0]) if hasattr(ast, "unparse") else src_tag
    assert body.index("world") < body.index("chassis"), (
        "the parked arm's 'world' branch no longer precedes the 'chassis' branch, so anything "
        "under `base/` -- the second arm included -- is tagged 'chassis' and the column links "
        "stop seeing it")
    guard = [n for n in ast.walk(fn) if isinstance(n, ast.If)
             and "GetCollisionEnabledAttr" in ast.unparse(n.test)]
    assert guard, "`_arm_obstacles` no longer tests CollisionEnabledAttr; this test is stale"
    assert "_parked" in ast.unparse(guard[0].test), (
        "the disabled-collider skip no longer exempts the parked arm, so arm-2's mast, columns "
        "and booms are back out of the obstacle set while its fingers stay in")



def _shipped_q8():
    """Every q8 in the baked trajectories -- the poses the robot actually performs."""
    import glob
    names8 = json.load(open(os.path.join(ROOT, "usd/_arm_model.json")))["actuated"]
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "usd/_traj_*.json"))) + \
            [os.path.join(ROOT, "usd/_grasp_close.json")]:
        d = json.load(open(f))
        if not (isinstance(d, dict) and "joints" in d and "frames" in d):
            continue
        idx = [d["joints"].index(n) for n in names8 if n in d["joints"]]
        if len(idx) != 8:
            continue
        for fr in d["frames"]:
            row = fr.get("q") if isinstance(fr, dict) else fr
            if row and len(row) >= len(d["joints"]):
                out.append(np.asarray([row[i] for i in idx], float))
    assert len(out) > 1000, f"only {len(out)} shipped poses found; this fixture is broken"
    return out


def test_the_obb_test_needs_its_edge_axes():
    """Two boxes can be separated by an EDGE-pair axis and by NO face normal. Six face axes alone
    would call this pair overlapping, which is why the test is 15 axes and not 6.

    The fixture is searched for, not guessed: the first version of this test used a hand-picked
    diagonal offset that was in fact separated by one of B's own face normals, so it passed with
    all nine edge axes deleted -- it named the property it did not test. The assertion below on
    the six face projections is what makes that impossible to repeat."""
    unit = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
    I = np.eye(3)
    pb = np.array([-1.167272, 1.108644, -0.196144])
    Rb = np.array([[0.905457, -0.400364, -0.140912],
                   [0.387542, 0.644467, 0.659146],
                   [-0.173085, -0.651438, 0.738695]])
    # Every one of the six FACE projections overlaps -- proved here, not asserted by comment.
    ha = hb = np.full(3, 0.5)
    d = pb
    R = Rb
    A = np.abs(R)
    assert not np.any(np.abs(d) > ha + A @ hb), "a face normal of A separates them; fixture is stale"
    assert not np.any(np.abs(R.T @ d) > hb + A.T @ ha), "a face normal of B separates them"
    assert not _obb_hits_obb(unit, np.zeros(3), I, unit, pb, Rb), (
        "a pair separated ONLY by an edge-pair axis was reported as overlapping -- the nine "
        "a_i x b_j axes are missing or wrong, and edge-on-edge crossings go unnoticed")
    assert _obb_hits_obb(unit, np.zeros(3), I, unit, pb * 0.2, Rb), (
        "two heavily overlapping boxes were reported clear; the test is broken")


def test_no_shipped_pose_self_collides():
    """The check must accept everything the robot already does, or it is unusable. 0 of 1228 with
    the tight `selfcol_aabb` boxes, a >3-hop adjacency mask, and the three permanent closure pairs
    ignored. A failure here means the mask, the boxes or the ignore list moved."""
    bad = [(q, M.self_collides(q)) for q in _shipped_q8()]
    hits = [(q, r) for q, r in bad if r is not None]
    assert not hits, (
        f"{len(hits)} of {len(bad)} SHIPPED poses are now flagged as self-collisions, e.g. "
        f"{hits[0][1]} at q8={np.round(hits[0][0], 3).tolist()} -- the planner would refuse "
        f"motions the robot performs every cycle")


def test_the_self_check_catches_the_hand_folding_onto_its_own_boom():
    """The point of the check. Nothing tested arm-vs-arm before: `collides` sees obstacles only and
    PhysX self-collision is off, so the sole protection was that every motion replayed a bake.
    `ik_repair`, `ik_candidates` and `_object_relative_goal` all hand the planner NOVEL goals.

    Bounded both ways on a uniform sample of admissible states: a rate near zero would mean the
    check does nothing, and a rate near 100% would mean it rejects the whole workspace."""
    lo, hi = (np.asarray(b, float) for b in M.bounds())
    rng = np.random.default_rng(0)
    tested = flagged = 0
    pairs = set()
    for _ in range(200000):
        if tested >= 2000:
            break
        q = lo + rng.random(8) * (hi - lo)
        if not M.in_lut_domain(q[2] - q[1], q[3]):
            continue
        tested += 1
        hit = M.self_collides(q)
        if hit:
            flagged += 1
            pairs.add(frozenset(hit))
    rate = flagged / tested
    # A BAND, not a target: it catches the finger sweep merged into the palm box, both primitives
    # boxed again, and both stubbed clear. It cannot see a single pair -- the per-pair test does.
    assert 0.75 < rate < 0.84, (
        f"{rate:.1%} of admissible states rejected -- outside the measured 78.2%. Too low means the "
        f"check stopped catching folds; too high means it started rejecting the workspace")
    assert frozenset(("Arm_Left_1", "Gripper_Link3_1")) in pairs, (
        "the dominant fold -- the hand doubling back onto its own boom -- is no longer caught")


def test_the_self_check_ignores_only_permanently_overlapping_pairs():
    """`SELFCOL_IGNORE` is a correctness mask, not a place to silence inconvenient pairs. Each
    entry must ACTUALLY overlap at essentially every shipped pose; a pair that is merely
    sometimes-colliding would be hidden by being listed here."""
    shipped = _shipped_q8()
    # 98%, with the one entry below 100% named: two links apart in the wrist, overlapping at all but
    # a handful of poses. The floor sits just under it so a merely-usual pair cannot be slipped in.
    for pair in M.SELFCOL_IGNORE:
        a, b = sorted(pair)
        n = sum(1 for q in shipped
                if _obb_hits_obb(M.selfcol_aabb[a], *M.fk(q)[M.selfcol_frame[a]],
                                 M.selfcol_aabb[b], *M.fk(q)[M.selfcol_frame[b]]))
        assert n >= 0.98 * len(shipped), (
            f"{a} x {b} is on the ignore list but overlaps in only {n}/{len(shipped)} shipped "
            f"poses -- it is not a permanent modelling artefact, so ignoring it hides a real pair")
    near_permanent = [tuple(sorted(p)) for p in M.SELFCOL_IGNORE
                      if sum(1 for q in shipped
                             if _obb_hits_obb(M.selfcol_aabb[sorted(p)[0]],
                                              *M.fk(q)[M.selfcol_frame[sorted(p)[0]]],
                                              M.selfcol_aabb[sorted(p)[1]],
                                              *M.fk(q)[M.selfcol_frame[sorted(p)[1]]]))
                      < len(shipped)]
    assert near_permanent == [("Gripper_Link3_1", "Hand_Bearing_1")], (
        f"the set of ignore entries that are NOT 100% permanent changed: {near_permanent}. "
        f"Every entry should be by-construction; this one exception is documented and measured.")


def test_the_mast_is_its_measured_sub_meshes_and_they_earn_their_place():
    """`Arm_1` is ONE 290x290x1640 mm box spanning the empty gap BETWEEN the twin rails, so the
    hand passes through its VOLUME in normal operation -- 19, 13 and 12 of 1228 shipped poses
    "collided" with it, all air. For the self-check it is replaced by the six sub-meshes it is
    made of, MEASURED off the live stage into `link_parts` and baked into the model.

    They cannot be had offline (binary `.usdc`, no `pxr` in either interpreter), and the obvious
    offline substitute was WRONG: deriving the rails from the column bearings' collider footprints
    put the left rail at y -74.1..-31.7 where the metal is at -94.37..-44.36, missing 20 mm of
    real rail. A plausible proxy is not a measurement, which is why this is data now.

    Both halves are asserted: the parts must be CLEAN on shipped poses, and they must CATCH
    something. Geometry that never fires would be decoration."""
    assert "Arm_1" not in M.selfcol_aabb, "the coarse mast box is back in the self-collision set"
    parts = [n for n in M.selfcol_aabb if n.startswith("Arm_1::")]
    assert len(parts) == 4, f"expected the two rails and their two feet, found {parts}"
    for r in parts:
        assert M.selfcol_frame[r] == "Arm_1", f"{r} does not ride the mast's frame"
        assert r.split("::")[1] not in M.SELFCOL_HULL_PARTS, f"{r} is a hull part; it must be out"
    # The caps are excluded for a MEASURED reason, so pin the measurement, not the exclusion.
    model_parts = json.load(open(os.path.join(ROOT, "usd/_arm_model.json")))["link_parts"]["Arm_1"]
    for cap in M.SELFCOL_HULL_PARTS:
        box = np.asarray(model_parts[cap], float)
        span = (box[1] - box[0])[:2]
        assert np.all(span > 0.28), (
            f"{cap} no longer spans the full cross-section ({span}) -- it was excluded BECAUSE it "
            f"is a plate hull the boom passes through; if the geometry changed, re-measure")
    lo, hi = (np.asarray(b, float) for b in M.bounds())
    rng = np.random.default_rng(0)
    tested = caught = 0
    while tested < 1200:
        q = lo + rng.random(8) * (hi - lo)
        if not M.in_lut_domain(q[2] - q[1], q[3]) or M.passive_violations(q[2] - q[1], q[3]):
            continue
        tested += 1
        hit = M.self_collides(q)
        if hit and any(n.startswith("Arm_1::") for n in hit):
            caught += 1
    assert caught / tested > 0.05, (
        f"the mast parts fire on only {caught}/{tested} admissible states -- the hand-into-mast "
        f"class they exist to catch has gone, so they are not earning the pairs they add")


def test_every_pair_in_the_list_is_one_the_check_actually_tests():
    """THE TEST THAT WAS MISSING, and its absence was found by review, not by me.

    The suite used to pass with 24 of the 25 pairs deleted from `_selfcol_pair_list`, because the
    only coverage was (a) membership in that list -- a shape -- and (b) one aggregate rejection
    rate that was almost entirely a single pair. Both survive gutting the list.

    So: assert per-pair BEHAVIOUR. Every pair that the shipped configuration fires on must still
    fire, individually. A pair silently dropped from the list fails here even if the aggregate
    rate barely moves.

    Counted over EVERY pair that fires, not the one `self_collides` returns first. That return is
    the first hit in pair-list order, so a pair can stop being reported because a neighbour two
    links up the wrist started winning the race -- which is a proxy for coverage, not coverage."""
    lo, hi = (np.asarray(b, float) for b in M.bounds())
    rng = np.random.default_rng(0)
    seen, tested = {}, 0
    while tested < 3000:
        q = lo + rng.random(8) * (hi - lo)
        if not M.in_lut_domain(q[2] - q[1], q[3]) or M.passive_violations(q[2] - q[1], q[3]):
            continue
        tested += 1
        poses = M.fk(q)
        for pair in M._selfcol_pair_list():
            if M._pair_hits(pair[0], pair[1], poses, 0.0):
                seen[pair] = seen.get(pair, 0) + 1
    # Measured on this seed. Each is a distinct physical fold, not a variant of one.
    must_fire = {
        ("Arm_Left_1", "Gripper_Link3_1"),      # the palm doubling back onto its own boom
        ("Arm_Left_1", "Gripper_Link1_1"),      # ... one link up the wrist
        ("Arm_Left_1", "Gripper_Link2_1"),      # ... two links up
        ("Arm_1::VerticalArm_3", "Gripper_Link3_1"),   # the hand into the near mast RAIL
    }
    # `Arm_1::VerticalArm_3 x Gripper_Link2_1` is NOT here, and that is a measurement: no sample this
    # size can witness it, so asserting it fires would assert a corner artefact.
    assert ("Arm_1::VerticalArm_3", "Gripper_Link2_1") in M._selfcol_pair_list(), (
        "the pair left the list. Nothing witnesses it firing -- the sphere never reaches the rail "
        "on 60000 admissible states -- so this membership is the only thing standing between it "
        "and a silent deletion, and a real sphere-into-rail would then go unchecked")
    missing = {p for p in must_fire if seen.get(p, 0) == 0}
    assert not missing, (
        f"these pairs no longer fire on ANY of {tested} admissible states: {sorted(missing)}. "
        f"A pair dropped from `_selfcol_pair_list` is invisible to an aggregate rate -- that is "
        f"exactly how a 24-of-25 deletion passed this suite before.")
    listed = {frozenset(p) for p in M._selfcol_pair_list()}
    for a, b in seen:
        assert frozenset((a, b)) in listed, f"{a} x {b} fired but is not in the pair list"


def test_the_check_does_not_tolerate_interpenetration():
    """The rejection-rate band alone is a weak guard: a NEGATIVE margin -- tolerating links
    already inside each other -- stayed green at -6 mm, which is 12 mm of accepted metal-in-metal.
    Assert the sign directly instead of hoping a rate notices."""
    lo, hi = (np.asarray(b, float) for b in M.bounds())
    rng = np.random.default_rng(4)
    q = None
    while q is None:
        c = lo + rng.random(8) * (hi - lo)
        if M.in_lut_domain(c[2] - c[1], c[3]) and not M.passive_violations(c[2] - c[1], c[3]) \
                and M.self_collides(c):
            q = c
    # A pose that collides at margin 0 must still collide at any NEGATIVE margin only if the
    # overlap exceeds it; the invariant that always holds is monotonicity in `margin`.
    assert M.self_collides(q, margin=0.0) is not None, "the fixture stopped colliding"
    assert M.self_collides(q, margin=0.01) is not None, (
        "growing every box by 10 mm made a colliding pose clear -- `margin` has the wrong sign")
    # The DEFAULT, explicitly: a tolerance allowing millimetres of link-into-link interpenetration
    # stayed green through the whole suite. Read the signature rather than trusting a rate.
    default = inspect.signature(ArmModel.self_collides).parameters["margin"].default
    assert default == 0.0, (
        f"`self_collides` defaults to margin={default}. Negative means links are allowed to "
        f"interpenetrate by twice that before the check fires; positive silently inflates every "
        f"box and will reject poses the robot performs.")


def test_the_broad_phase_never_prunes_a_narrow_phase_hit():
    """The broad phase must be a BOUND, not an approximation: if the oriented test would report a
    hit, the AABB test must not have pruned the pair first.

    Driven directly against `_aabb_prunes` with AXIS-ALIGNED boxes. That is the point -- with
    identity rotation the world AABB equals the OBB, so there is no slack to hide behind and the
    `2 * margin` factor is the entire answer. State sampling CANNOT test this: on this arm's real
    geometry the AABB-over-OBB slack swallows the difference, and 3,000 sampled admissible poses
    show zero disagreement with the factor either way. A guard that cannot fail is not a guard.

    Both call sites pass margin=0.0 today, so the defect is latent -- which is exactly why it
    needs pinning. It becomes a silent MISS the first time anyone passes a margin."""
    rng = np.random.default_rng(2)
    I = np.eye(3)
    checked = 0
    for _ in range(4000):
        margin = float(rng.choice([0.0, 0.002, 0.005, 0.02]))
        ha = rng.uniform(0.02, 0.2, 3)
        hb = rng.uniform(0.02, 0.2, 3)
        la = np.vstack([-ha, ha])
        lb = np.vstack([-hb, hb])
        # Place B so the gap straddles the interesting band [0, 2*margin] on one axis.
        pa = np.zeros(3)
        gap = rng.uniform(-0.01, max(0.05, 3.0 * margin))
        pb = np.array([ha[0] + hb[0] + gap, rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01)])
        narrow = _obb_hits_obb(la, pa, I, lb, pb, I, margin)
        lo_a, hi_a = pa - ha, pa + ha
        lo_b, hi_b = pb - hb, pb + hb
        pruned = bool(_aabb_prunes(lo_a, hi_a, lo_b, hi_b, margin))
        if narrow:
            checked += 1
            assert not pruned, (
                f"the broad phase pruned a pair the oriented test calls a HIT: gap={gap:.4f}, "
                f"margin={margin} -- it is not a conservative bound, so real collisions are "
                f"silently skipped whenever a non-zero margin is passed")
    assert checked > 200, f"only {checked} narrow-phase hits generated; this test proves little"



# A configuration that passes EVERY other gate -- joint bounds, LUT domain, passive limits, empty
# obstacle set -- and is a self-collision. Before `self_collides` the planner returned it happily.
FOLDED = np.array([-1.3559, 0.9274, 0.9956, 0.183, -1.5653, 2.9748, -0.6333, -0.5844])


def test_the_folded_pose_is_invalid_for_exactly_one_reason():
    """Guards the fixture. If a future bounds or LUT change made FOLDED invalid for some OTHER
    reason, the veto test below would pass without the veto doing anything."""
    dh = FOLDED[2] - FOLDED[1]
    assert M.in_lut_domain(dh, FOLDED[3]), "FOLDED left the LUT domain; pick a new fixture"
    assert not M.passive_violations(dh, FOLDED[3]), "FOLDED now violates a passive limit"
    assert not M.collides(FOLDED, [], 0.02, None, None), "FOLDED now hits an obstacle"
    # WHICH pair is reported depends on the list's order, and the list grew when the mask moved to
    # measured permanence. Assert the property that matters, not the particular tuple.
    hit = M.self_collides(FOLDED)
    assert hit is not None, "FOLDED no longer self-collides; pick a new fixture"
    assert "Arm_Left_1" in hit and hit[1].startswith("Gripper_"), hit


def test_the_planner_reports_self_collisions_on_the_path_it_emits():
    """SHADOW metric, always on. Measured on the FINAL densified waypoints -- the states the
    executor actually writes -- not inside `isValid`, which runs thousands of times per solve and
    would report sampler behaviour rather than anything a path did."""
    out = _plan(GRASP.tolist(), GRASP.tolist(), seed=1, capture=True)
    assert any("self-collision" in ln.lower() for ln in out[1].splitlines()), (
        f"the planner emitted a path without reporting whether it self-collides:\n{out[1][-800:]}")


def test_self_collide_plan_vetoes_the_fold_and_is_off_by_default():
    """The veto is OFF until a run measures the shadow metric -- turning it on blind could refuse
    motions the robot performs today. Both halves matter: that it is off, and that it works."""
    # NO env override on this half -- it must exercise the DEFAULT. Passing the flag here would let
    # the default flip with every test still green, which a mutation showed.
    assert "SELF_COLLIDE_PLAN" not in os.environ, (
        "SELF_COLLIDE_PLAN is set in this environment, so the default cannot be tested here")
    off = _plan(FOLDED.tolist(), FOLDED.tolist(), seed=1)
    assert isinstance(off, list), (
        f"a self-colliding start was refused BY DEFAULT -- the veto is on before any run has "
        f"measured the shadow metric, and it can refuse poses the robot performs today: {off}")
    on = _plan(FOLDED.tolist(), FOLDED.tolist(), seed=1, env={"SELF_COLLIDE_PLAN": "1"})
    assert isinstance(on, dict) and "error" in on, (
        f"SELF_COLLIDE_PLAN=1 planned from a self-colliding start anyway: {on}")



def test_first_hit_can_be_restricted_to_the_links_a_caller_cares_about():
    """The placement servo cannot be checked the way free space is: a per-step FULL check answers a
    mechanical question with geometry and, with millimetres of real slip against 10 mm boards, would
    refuse working placements (docs/FINDINGS.md). What it CAN check is the proximal links -- the
    boom and the columns -- which have no business touching anything during an insert. That needs a
    first_hit restricted to those links, with the hand and its payload left out entirely."""
    q = np.array([0.0, 0.5, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0])
    boxes = M.link_aabbs(q)
    palm = boxes["Gripper_Link3_1"]
    pt = np.array([palm[1][0] - 0.005, palm[1][1] - 0.005, palm[1][2] - 0.005])
    at_the_hand = [(np.array([pt - 0.001, pt + 0.001]), "world")]

    assert M.first_hit(q, at_the_hand, margin=0.0) == "Gripper_Link3_1", "fixture: the hand hits it"
    assert M.first_hit(q, at_the_hand, margin=0.0, links=("Arm_Left_1", "Arm_1")) is None, (
        "restricted to the proximal links, an obstacle at the HAND is not their business")

    boom = boxes["Arm_Left_1"]
    bpt = np.array([boom[1][0] - 0.005, boom[1][1] - 0.005, boom[1][2] - 0.005])
    at_the_boom = [(np.array([bpt - 0.001, bpt + 0.001]), "world")]
    assert M.first_hit(q, at_the_boom, margin=0.0, links=("Arm_Left_1", "Arm_1")) == "Arm_Left_1", (
        "an obstacle at the boom IS their business")


def test_the_grasped_exemption_covers_the_wrist_bearing_but_never_the_boom():
    """Every link that HOLDS the object may touch it; nothing above the wrist may. Rooting the
    subtree at the gripper left Hand_Bearing_1 out, and the insert ramp then refused its first step
    on 2 of 5 seeds -- the bearing was inside MARGIN of the object it was carrying."""
    assert "Hand_Bearing_1" in M.hand_subtree, sorted(M.hand_subtree)
    for link in ("Arm_Left_1", "Arm_Right_1", "Arm_1"):
        assert link not in M.hand_subtree, f"{link} is not a link that holds the object"
    q = np.array([0.0, 0.5, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0])
    b = M.link_aabbs(q)["Hand_Bearing_1"]
    pt = np.array([b[1][0] - 0.005, b[1][1] - 0.005, b[1][2] - 0.005])
    box = (np.array([pt - 0.001, pt + 0.001]), "grasped")
    assert M.first_hit(q, [box], margin=0.0) is None, "the bearing may overlap what it carries"
    world = (np.array([pt - 0.001, pt + 0.001]), "world")
    assert M.first_hit(q, [world], margin=0.0) == "Hand_Bearing_1", "but not world geometry"


def test_gripper_link_exempted_from_grasped_box():
    q = np.array([0.0, 0.5, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0])
    b = M.link_aabbs(q)["Gripper_Link2_1"]
    pt = np.array([b[1][0] - 0.005, b[1][1] - 0.005, b[1][2] - 0.005])
    tbox = (np.array([pt - 0.001, pt + 0.001]), "grasped")
    # A pose where a GRIPPER or FINGER link overlaps a "grasped" box is NOT reported as a collision.
    assert M.first_hit(q, [tbox], margin=0.0) is None

def test_gripper_link_not_exempted_from_world_box():
    q = np.array([0.0, 0.5, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0])
    b = M.link_aabbs(q)["Gripper_Link2_1"]
    pt = np.array([b[1][0] - 0.005, b[1][1] - 0.005, b[1][2] - 0.005])
    tbox = (np.array([pt - 0.001, pt + 0.001]), "world")
    # The same overlap against a "world" box IS still a collision.
    assert M.first_hit(q, [tbox], margin=0.0) == "Gripper_Link2_1"

def test_non_hand_link_not_exempted_from_grasped_box():
    q = np.array([0.0, 0.5, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0])
    b = M.link_aabbs(q)["Arm_1"]
    pt = np.array([b[1][0] - 0.005, b[1][1] - 0.005, b[1][2] - 0.005])
    tbox = (np.array([pt - 0.001, pt + 0.001]), "grasped")
    # A NON-hand link overlapping a "grasped" box IS still a collision.
    assert M.first_hit(q, [tbox], margin=0.0) == "Arm_1"


def test_the_held_box_carries_the_grasped_exemption_the_hand_subtree_has():
    """The payload attached to the hand IS the prim `allow_contact_with` tags "grasped", so a
    checker that exempts the hand but not `__held__` refuses every carry pose against the object
    the hand is holding. Per tag, never a blanket deletion: "world" still binds."""
    held, (tip, _) = _tip_board()
    for what, ob in (("2-tuple grasped", (tip, "grasped")),
                     ("4-tuple grasped floor", (tip, "grasped", 0.008, (HELD_LINK,)))):
        assert M.first_hit(GRASP, [ob], 0.0, held) is None, what
    assert M.first_hit(GRASP, [(tip, "world")], 0.0, held) == HELD_LINK
    # The object's own collider, where `arm_obstacles` puts it under `allow_contact_with`.
    (local, oc, _oR), _hand = _grasped(GRASP, 0.040, 0.090)
    own = np.array([oc + local[0], oc + local[1]])
    assert M.first_hit(GRASP, [(own, "grasped")], 0.0, held) is None


def test_the_finger_box_needs_no_ignore_entry_of_its_own():
    """An exemption is a check deleted, so the geometry is placed where none is needed.

    The first attempt put the floor at 20mm, where the slab reaches the wrist link the hand
    overlaps by construction -- 1185 of 1228 shipped poses flagged, and that was to be papered
    over with an ignore entry. It would not have survived this file's own rule that the ignore list
    holds only PERMANENT overlaps: the pair is clear in 43 of those poses. Raising the floor 5mm
    removes the overlap instead of excusing it."""
    fing = f"{M.hand_link}::fingers"
    assert fing in M.selfcol_aabb, (
        "there is no finger pseudo-link, so the memberships below prove nothing")
    assert M.selfcol_frame[fing] == M.hand_link, (
        f"the finger box rides {M.selfcol_frame[fing]!r}, not the hand -- it would be checked at "
        f"the wrong place in the world")
    for p in M.SELFCOL_IGNORE:
        assert fing not in p, (
            f"the finger box has acquired an ignore entry ({sorted(p)}). Move the geometry, not "
            f"the check: the measured window where it needs none is 22..29.7mm")
    pairs = {frozenset(q) for q in M._selfcol_pair_list()}
    for other in ("Gripper_Link1_1", "Hand_Bearing_1", "Arm_Left_1"):
        assert frozenset((fing, other)) in pairs, (
            f"{other} x the finger box is not tested at all -- either the pseudo-link is gone or "
            f"something is excusing it")


def test_the_finger_box_sits_above_the_palm_base_where_the_boom_passes():
    """z0 is the whole design. At z0 = 0 the box fills the corner at the palm base where the boom
    passes within 7.3mm and 646 of 1228 shipped poses flag; above 29.7mm the box can no longer
    reach the wrist links at all and the ignore-inheritance goes slack, so its own test would pass
    with the clause deleted. The shipped value must sit strictly inside that bracket."""
    fing = M.selfcol_aabb[f"{M.hand_link}::fingers"]
    z0 = float(fing[0][2])
    assert 0.022 <= z0 <= 0.0297, (
        f"the finger box floor is {z0 * 1000:.1f}mm, outside the measured [22.0, 29.7]mm window: "
        f"below it the slab reaches the wrist link the hand overlaps by construction and shipped "
        f"poses flag; above it the box stops reaching the wrist links and that coverage is lost")
    # x/y are the measured sweep, untouched: only the floor is raised.
    assert np.allclose(fing[0][:2], FINGER_SWEEP_AABB[0][:2]), fing[0]
    assert np.allclose(fing[1], FINGER_SWEEP_AABB[1]), fing[1]


def test_the_shipped_poses_keep_headroom_against_the_finger_box():
    """Not merely "does not flag" -- `test_no_shipped_pose_self_collides` already says that, and it
    would still pass with the boxes a hair from touching. Headroom, so a later tweak that eats the
    clearance fails here rather than in the simulator."""
    worst = None
    for q in _shipped_q8():
        r = M.self_collides(q, margin=0.001)
        if r is not None:
            worst = (q, r)
            break
    assert worst is None, (
        f"a shipped pose comes within 1.0mm of self-collision: {worst[1]} at "
        f"q8={np.round(worst[0], 3).tolist()}")


def test_margin_is_defined_once():
    """MARGIN is geometry; a second assignment under morph/ would be two constants with one name."""
    from morph.arm.collision import MARGIN
    assert MARGIN == 0.02, f"the clearance every link keeps changed to {MARGIN}"
    owners = []
    for dp, _, fs in os.walk(os.path.join(ROOT, "morph")):
        for f in fs:
            if not f.endswith(".py"):
                continue
            text = open(os.path.join(dp, f), encoding="utf-8").read()
            assigns = any(isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "MARGIN" for t in n.targets)
                          for n in ast.parse(text).body)
            if assigns:
                owners.append(os.path.relpath(os.path.join(dp, f), ROOT))
    assert owners == ["morph/arm/collision.py"], owners



# A pose where the fingers foul the mast while the PALM box is clear: the exact class palm-only
# geometry cannot see. Found by search, pinned as a literal so this test neither searches nor drifts.
_FINGER_WITNESS = np.array(
    [-3.051924, 0.898701, 1.134024, 0.320627, 0.709167, -1.71893, -0.947126, -0.43], float)


def test_the_finger_box_catches_a_fold_the_palm_box_cannot_see():
    """End to end, and the fixture proves itself: the palm box alone is CLEAR against the same
    partner at this pose, so a pass here can only come from the finger geometry."""
    fing = f"{M.hand_link}::fingers"
    hit = M.self_collides(_FINGER_WITNESS)
    assert hit is not None and fing in hit, (
        f"the witness pose reports {hit!r}: the fingers are folded onto the mast and nothing "
        f"sees it")
    other = hit[0] if hit[1] == fing else hit[1]

    poses = M.fk(_FINGER_WITNESS)
    pa, Ra = poses[M.selfcol_frame[M.hand_link]]
    pb, Rb = poses[M.selfcol_frame[other]]
    assert not _obb_hits_obb(M.selfcol_aabb[M.hand_link], pa, Ra,
                             M.selfcol_aabb[other], pb, Rb, 0.0), (
        f"the PALM box already overlaps {other} here, so this pose proves nothing about the "
        f"fingers -- find another witness")


def test_the_boom_keeps_measured_headroom_from_the_finger_box():
    """Aggregate clearance hides a pair that is one hair from touching. This is the pair the
    shadow metric actually reports folding: worst shipped clearance 8.79mm, so 3mm is headroom,
    not a coincidence."""
    fing = f"{M.hand_link}::fingers"
    worst, worst_q = 9.9, None
    for q in _shipped_q8():
        poses = M.fk(q)
        pa, Ra = poses[M.selfcol_frame["Arm_Left_1"]]
        pb, Rb = poses[M.selfcol_frame[fing]]
        if _obb_hits_obb(M.selfcol_aabb["Arm_Left_1"], pa, Ra,
                         M.selfcol_aabb[fing], pb, Rb, 0.003):
            worst, worst_q = 0.0, q
            break
    assert worst_q is None, (
        f"a shipped pose brings the boom within 3mm of the finger box: "
        f"q8={np.round(worst_q, 3).tolist()}")


def test_the_boom_self_checks_against_its_DRAWN_METAL_not_its_collision_proxy():
    """I made this mistake and it is worth a test rather than a comment.

    `Arm_Left_1` has one ENABLED collider (`Box`, 40.0mm wide in y) and two meshes whose colliders
    are authored-but-disabled (`HorizontalArm_0/1`), giving a 68.2mm union. Narrowing the
    self-check to the enabled collider looked right -- it made a reported fold disappear -- and it
    was wrong twice over:

    1. The justification was "MARGIN pays for visual-beyond-collider". It does not: every
       production caller of `self_collides` passes NO margin (`arm_plan.py`, `morph/arm/cartesian.py`),
       so the default 0.0 applies and nothing pays for anything.
    2. `tools/audit_colliders.py` reports this exact link as `VIS_OUT +24.9mm [col1/vis2]` -- one
       collider, two visuals, and the DRAWN METAL extends 24.9mm beyond the proxy.

    A self-collision check exists to stop the planner folding the arm into ITSELF, and "itself" is
    the metal, not the collision proxy PhysX happens to use. Under-approximating the arm here hides
    real folds. If this ever needs narrowing again, argue it from the MESHES."""
    box = M.selfcol_aabb["Arm_Left_1"]
    width = float(box[1][1] - box[0][1])
    assert width > 0.060, (
        f"the boom's self-check box is {width * 1000:.1f}mm wide in y. The enabled collider alone "
        f"is 40.0mm and the drawn metal is ~68mm (audit: VIS_OUT +24.9mm, col1/vis2). Anything "
        f"near 40mm means the check has been narrowed to the proxy and is ignoring real arm")

    # And the two ignore entries that go with it must stay: with the full-width box both pairs
    # overlap in 1228/1228 shipped poses, which is what an ignore entry is FOR.
    pairs = {frozenset(p) for p in M.SELFCOL_IGNORE}
    for a, b in (("Arm_1::VerticalArm_3", "Arm_Left_1"),
                 ("Arm_Left_1", "Bearing_Column_Left_1")):
        assert frozenset((a, b)) in pairs, (
            f"{a} x {b} is no longer ignored. With the boom at its drawn width these overlap in "
            f"every shipped pose, so removing the entry floods the check with permanent hits")


# An ACM entry in MoveIt's sense -- these links may touch THIS box, scoped to one motion. The
# `"grasped"` tag is the same thing special-cased for the hand.

def _deck_pose():
    """A q8 whose boom is INSIDE the chassis deck box -- the descent's own end pose, reproduced."""
    q = np.zeros(8)
    q[M.Q8.index("BaseJoint_1")] = -0.195
    q[M.Q8.index("ColumnLeftBearingJoint_1")] = 0.1973
    q[M.Q8.index("ColumnRightBearingJoint_1")] = 0.3267
    q[M.Q8.index("ArmLeftJoint_1")] = 0.4367
    return q


def _deck_gap(q, link, box, tag="chassis"):
    """Metres of clearance between `link` and `box` at `q`, bisected on `first_hit` itself so the
    number and the verdict cannot disagree. Negative means overlapping."""
    from morph.arm.collision import MARGIN
    if M.first_hit(q, [(box, tag)], 0.0, links=(link,)) is not None:
        lo, hi = -0.30, 0.0
    else:
        lo, hi = 0.0, MARGIN * 2
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if M.first_hit(q, [(box, tag)], mid, links=(link,)) is not None:
            hi = mid
        else:
            lo = mid
    return hi if hi <= 0.0 else lo


def test_a_pad_NEVER_excuses_a_pose_that_actually_overlaps():
    """The safety property that separates a FLOOR from the deletion it replaced. The old mechanism
    was `if link in allow: continue` -- the pair vanished from the check, so a penetrating pose
    passed. A pad is a smaller margin, and no margin excuses geometry that already intersects.

    Mutation: restore `continue` for a padded link -- must fail.
    """
    from morph.config import CHASSIS_PLATE_BOX
    q = _deck_pose()
    assert M.first_hit(q, [(CHASSIS_PLATE_BOX, "chassis")], 0.0, links=("Arm_Left_1",)) \
        == "Arm_Left_1", "the fixture is not the penetrating pose; this test would pass on absence"
    padded = (CHASSIS_PLATE_BOX, "chassis", 0.008, ("Arm_Left_1",))
    assert M.first_hit(q, [padded], 0.0, links=("Arm_Left_1",)) == "Arm_Left_1", (
        "a pad excused a pose that OVERLAPS the box -- that is the deletion this replaced")
    assert M.first_hit(q, [padded], 0.02, links=("Arm_Left_1",)) == "Arm_Left_1", (
        "a pad excused an overlap at the full margin too")


def test_a_pad_lowers_the_margin_for_the_links_it_names_and_no_others():
    """The pair is still CHECKED, at `pad` instead of `margin`. Self-calibrating: the gap is
    measured first, so the test pins behaviour rather than a fixture's happened-to-be distance.

    Mutation: apply `mg` to every link instead of `if link in links` -- the universal half fails.
    """
    from morph.arm.collision import MARGIN
    from morph.config import CHASSIS_PLATE_BOX
    q = _shipped_q8()[0]
    box, link = CHASSIS_PLATE_BOX, "Arm_Left_1"
    gap = _deck_gap(q, link, box)
    assert 0.0 < gap < MARGIN, (
        f"fixture: {link} must be CLEAR of the deck but inside MARGIN; gap {gap * 1000:.1f}mm")
    pad = gap * 0.5                       # under the real clearance, so the pad must pass it
    assert M.first_hit(q, [(box, "chassis")], MARGIN, links=(link,)) == link, (
        "the fixture is not inside MARGIN, so the pad has nothing to excuse")
    assert M.first_hit(q, [(box, "chassis", pad, (link,))], MARGIN, links=(link,)) is None, (
        "the pad did not lower the margin for the link it names")
    tight = gap * 1.5                     # OVER the real clearance: still refused, still a bound
    assert M.first_hit(q, [(box, "chassis", tight, (link,))], MARGIN, links=(link,)) == link, (
        "a pad LARGER than the real clearance passed -- the pad is not being enforced")
    # ...and every other link keeps MARGIN, universally.
    for other in M.link_aabb:
        if other == link or other in M.column_group:
            continue
        plain = M.first_hit(q, [(box, "chassis")], MARGIN, links=(other,))
        with_pad = M.first_hit(q, [(box, "chassis", pad, (link,))], MARGIN, links=(other,))
        assert plain == with_pad, (
            f"a pad naming {link} changed {other}'s verdict: {plain} -> {with_pad}")


def test_a_pad_does_not_leak_to_another_box_with_the_same_tag():
    """The pad names a BOX. A second box carrying the same tag must still hold the full margin --
    the whole difference between a per-pair floor and a tag-level rule."""
    from morph.arm.collision import MARGIN
    from morph.config import CHASSIS_PLATE_BOX
    q = _shipped_q8()[0]
    link = "Arm_Left_1"
    gap = _deck_gap(q, link, CHASSIS_PLATE_BOX)
    pad = gap * 0.5
    obs = [(CHASSIS_PLATE_BOX, "chassis", pad, (link,)), (CHASSIS_PLATE_BOX, "chassis")]
    assert M.first_hit(q, obs, MARGIN, links=(link,)) == link, (
        "a pad on one box lowered the margin against a DIFFERENT box with the same tag")
    assert M.first_hit(q, list(reversed(obs)), MARGIN, links=(link,)) == link, (
        "the pad is being read off the wrong box -- it leaked in the other direction")


def test_a_padded_box_keeps_the_frame_and_tag_rules_of_an_unpadded_one():
    """The pad changes a margin; it must not quietly change what the TAG means. Same box, same
    pose, same verdict for every link the pad does not name -- for all three tags."""
    from morph.config import CHASSIS_PLATE_BOX
    q = _deck_pose()
    tall = np.array([CHASSIS_PLATE_BOX[0], CHASSIS_PLATE_BOX[1] + [0.0, 0.0, 1.5]])
    # a box ON the hand, so the "grasped" tag's hand-subtree rule is actually exercised.
    hand = M.link_aabbs(q, None)[M.hand_link]
    covered = {tag: 0 for tag in ("chassis", "world", "grasped")}
    for tag, box in (("chassis", tall), ("world", tall), ("grasped", np.asarray(hand, float))):
        for link in M.link_aabb:
            plain = M.first_hit(q, [(box, tag)], 0.0, links=(link,))
            anno = M.first_hit(q, [(box, tag, 0.008, ("__nobody__",))], 0.0, links=(link,))
            assert plain == anno, (
                f"padding a {tag!r} box changed {link}'s verdict: {plain} -> {anno}")
            covered[tag] += plain is not None
    assert all(covered.values()), (
        f"a tag was never exercised by a real overlap, so its rule is untested: {covered}")


def test_an_unpadded_obstacle_set_is_decided_exactly_as_before():
    """The fast path. Every shipped pose, both margins, against the real deck: adding the mechanism
    must not move a single verdict when nobody uses it."""
    from morph.arm.collision import MARGIN
    from morph.config import CHASSIS_PLATE_BOX
    obs = [(CHASSIS_PLATE_BOX, "chassis")]
    n = 0
    for q in _shipped_q8():
        for mg in (0.0, MARGIN):
            before = M.first_hit(q, obs, mg)
            # a pad naming NOBODY: the soft branch runs, and every link keeps `margin`.
            after = M.first_hit(q, [(CHASSIS_PLATE_BOX, "chassis", 0.008, ())], mg)
            assert before == after, f"an empty pad changed a verdict: {before} -> {after}"
            n += 1
    assert n == 2456, f"the corpus moved: {n} checks, expected 2456"


# `FINGER_SWEEP_AABB` is honest over the whole finger JOINT RANGE, but the hand only commands
# open <-> closed, and the difference is slack in the direction that reaches the chassis deck.

def _descent_q8():
    """The shipped descent handoff, from the live probe."""
    q = np.zeros(8)
    q[M.Q8.index("BaseJoint_1")] = -0.1955
    q[M.Q8.index("ColumnLeftBearingJoint_1")] = 0.1973
    q[M.Q8.index("ColumnRightBearingJoint_1")] = 0.3270
    q[M.Q8.index("ArmLeftJoint_1")] = 0.2466
    return q


def _open_fingers():
    return dict(FINGERS_RECORDING)


def test_without_finger_values_every_verdict_is_byte_identical():
    """Monotonicity, and it is the whole safety argument: knowing nothing about the fingers must
    give exactly today's answer. Universal over the corpus and both margins, not a spot check."""
    from morph.arm.collision import MARGIN
    obs = [(np.array([[-0.35, -0.30, 0.1386], [0.35, 0.30, 0.1586]]), "chassis")]
    n = 0
    for q in _shipped_q8():
        for mg in (0.0, MARGIN):
            assert M.first_hit(q, obs, mg) == M.first_hit(q, obs, mg, fingers=None), q.tolist()
            n += 1
    assert n == 2456, f"the corpus moved: {n}"


def test_the_commanded_fingers_replace_the_swept_union_and_are_checked_THEMSELVES():
    """Both halves. The hand must stop carrying the sweep -- and the nine finger links must become
    checkable in its place, or this trades a false positive for a blind spot."""
    from morph.config import CHASSIS_PLATE_BOX
    q, obs = _descent_q8(), [(CHASSIS_PLATE_BOX, "chassis")]
    assert M.first_hit(q, obs, 0.0195, links=(M.hand_link,)) is not None, (
        "the fixture no longer has the hand inside MARGIN; the next assertion would be vacuous")
    assert M.first_hit(q, obs, 0.0195, links=(M.hand_link,), fingers=_open_fingers()) is None, (
        "the commanded fingers did not shrink the hand's box")
    seen = {lk for lk, _b, _p, _R in M._links(q, None, None, fingers=_open_fingers())}
    assert len([k for k in seen if "finger" in k]) == 9, (
        f"the finger links are not being checked in the union's place: {sorted(seen)}")


def test_the_finger_values_do_not_clobber_the_closure_passives():
    """`fk(q8, passives=)` REPLACES the LUT lookup, so routing finger values through it zeroes all
    four closure passives and silently moves the boom and columns. Merge, never replace."""
    q = _descent_q8()
    plain = M.fk(q)
    with_f = {}
    for lk, _b, p, _R in M._links(q, None, None, fingers=_open_fingers()):
        with_f[lk] = p
    for lk in ("Arm_Left_1", "Bearing_Column_Left_1", "Arm_1"):
        assert np.allclose(plain[lk][0], with_f[lk], atol=1e-9), (
            f"supplying finger values moved {lk} by "
            f"{np.linalg.norm(plain[lk][0] - with_f[lk]) * 1000:.1f}mm -- the LUT was clobbered")


# A real SELF-FOLD waypoint from a place retreat: one of the many that only the fat gripper union
# box rejects, and that the audited palm clears.
LIVE_FIRST_FOLD = np.array([0.0376, 0.1196, 0.1381, 0.3267, 0.2995, -1.0906, 0.6905, 0.7016])


def test_the_hand_self_box_is_the_audited_palm_not_the_gripper_union():
    """`selfcol_aabb` is copied from `link_aabb`, whose hand entry is the extractor's UNION of the
    palm AND every finger collider. The fingers then go in AGAIN as the `::fingers` pseudo-link, so
    the hand was double-counted and 4x its measured volume -- in the one place the module's own
    comment says fat is "not conservative but simply wrong", because an inflated box overlaps its
    neighbours permanently and reports a collision that is 100% air.

    Both halves asserted, as for the mast parts: the palm must be the AUDITED box and the fingers
    must still be covered by their own pseudo-link, shipped poses must stay clean, and the swap
    must actually change a verdict -- geometry that never fires would be decoration."""
    from morph.arm.collision import PALM_AABB
    box = M.selfcol_aabb[M.hand_link]
    assert np.allclose(box, PALM_AABB), (
        f"hand self box is {np.round((box[1] - box[0]) * 1000, 1).tolist()}mm, audited palm is "
        f"{np.round((PALM_AABB[1] - PALM_AABB[0]) * 1000, 1).tolist()}mm")
    # Swapping the palm must not drop the fingers -- they are covered by their own pseudo-link.
    assert f"{M.hand_link}::fingers" in M.selfcol_aabb, "the finger sweep left the self-check set"
    assert M.selfcol_frame[f"{M.hand_link}::fingers"] == M.hand_link
    # CATCHES something: the live run's first fold is air under the audited palm.
    assert M.self_collides(LIVE_FIRST_FOLD) is None, (
        f"the audited palm still folds at the live waypoint: {M.self_collides(LIVE_FIRST_FOLD)}")
    # CLEAN on shipped poses -- the swap only ever withdraws a rejection, never adds one.
    bad = [(q, M.self_collides(q)) for q in _shipped_q8()]
    assert not [b for b in bad if b[1]], f"shipped poses now self-collide: {[b[1] for b in bad][:3]}"

def test_every_finger_joint_reaches_the_dict_handed_to_the_CHECKER():
    """`palm_finger_b/c_joint_1` swing the whole b and c fingers and do NOT match
    `startswith("finger_")`. A filter that misses them leaves `fk` defaulting them to 0.0, so the
    checker judges a NARROWER hand than the robot has -- the direction that approves a pose which
    collides. Measured at the grip-end pose the robot holds through every carry: 19.1mm of real pad
    outside the hand the checker sees, against a 20mm MARGIN.

    A SHAPE test: it pins WHERE the one finger filter lives, not what it returns. It
    does not substring-match: it extracts each finger-dict comprehension's own `if` clause and
    EVALUATES it against the model's real joint names, so a predicate spelled differently but still
    correct passes. Mutation: revert the execute.py site to `startswith("finger_")` -- must fail.
    """
    import ast as _ast
    import json as _json

    with open(os.path.join(ROOT, "usd", "_arm_model.json"), encoding="utf-8") as fh:
        finger_joints = [j["name"] for j in _json.load(fh).get("joints", [])
                         if "finger" in j.get("name", "")]
    assert len(finger_joints) == 11, f"the model's finger-joint set moved: {sorted(finger_joints)}"
    assert [n for n in finger_joints if not n.startswith("finger_")], (
        "fixture: no palm_finger joint exists, so this test cannot catch the omission")

    checked = 0
    for rel in ("morph/arm/execute.py", "morph/pick/reach.py"):
        with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
            src = fh.read()
        for node in _ast.walk(_ast.parse(src)):
            if not isinstance(node, _ast.DictComp) or not node.generators:
                continue
            gen = node.generators[0]
            if not (isinstance(gen.target, _ast.Name) and len(gen.ifs) == 1):
                continue
            test = _ast.unparse(gen.ifs[0])
            if "finger" not in test:
                continue
            var = gen.target.id
            missed = [n for n in finger_joints if not eval(test, {}, {var: n})]  # noqa: S307
            checked += 1
            assert not missed, (
                f"{rel}: the finger-dict filter `{test}` omits {missed}; `fk` defaults those to "
                f"0.0 and the collision check then sees a hand narrower than the real one")
    assert checked >= 1, (
        f"only {checked} finger-dict filter(s) found -- the sites moved and this test guards "
        f"nothing")
    # `morph/pick/reach.py` must build NONE of its own: a second filter is a second thing to get
    # wrong, which is the defect this test exists for. It asks `_commanded_fingers()` instead.
    with open(os.path.join(ROOT, "morph", "pick", "reach.py"), encoding="utf-8") as fh:
        _rsrc = fh.read()
    _own = [_ast.unparse(n.generators[0].ifs[0]) for n in _ast.walk(_ast.parse(_rsrc))
            if isinstance(n, _ast.DictComp) and n.generators and len(n.generators[0].ifs) == 1
            and "finger" in _ast.unparse(n.generators[0].ifs[0])]
    assert not _own, (
        f"reach.py rolled its own finger dict again ({_own}); it must ask _commanded_fingers() so "
        f"the hand the checker sees and the hand the planner flies cannot diverge")
    assert "_commanded_fingers()" in _rsrc, (
        "reach.py neither builds a finger dict nor asks for one -- the descent would be checked "
        "against the SWEPT box, whose floor sits 39mm below where the fingers actually reach")


def test_every_CHASSIS_PAD_LINK_is_a_link_the_CHECKER_yields():
    """`first_hit` matches a padded link by `if link in links`, so a name that matches no link is a
    SILENT no-op -- indistinguishable from a pad that is working.

    Not hypothetical: `8df2c98` carved the nine finger pads out of `Gripper_Link3_1` and the
    descent's deck allowance stopped covering them without any entry changing. The list lives in
    `morph.arm.collision` now, beside the geometry, so this imports it directly instead of reading
    `morph/pick/reach.py` by AST -- the constant moved out of the module that needs a simulator.

    Mutation: rename any entry (`Hand_Bearing_1` -> `Hand_Bearing`) -- must fail.
    """
    from morph.arm.collision import CHASSIS_PAD_LINKS, CHASSIS_PAD_M, MARGIN
    assert CHASSIS_PAD_LINKS, "the pad names no link, so it can excuse nothing"
    assert 0.0 < CHASSIS_PAD_M < MARGIN, (
        f"a pad outside (0, MARGIN) either changes nothing or is not a floor: {CHASSIS_PAD_M}")
    q = np.array([0.0, 1.2, 1.2, 0.1, 0.0, 0.0, 0.0, 0.0])
    # WITH fingers: the pad names `finger_a_link_3_1`, which only exists as its own link when the
    # caller passes commanded fingers -- exactly the split that silently voided `DECK_NEAR`.
    checked = {t[0] for t in M._links(q, fingers={"finger_a_joint_1_1": 0.0})}
    missing = [n for n in CHASSIS_PAD_LINKS if n not in checked]
    assert not missing, f"the chassis pad names links the checker never yields: {missing}"


def _open_hand():
    return {n: (-0.55 if n.endswith("_joint_1_1") else 0.0) for n in
            ("finger_a_joint_1_1", "finger_a_joint_2_1", "finger_a_joint_3_1",
             "finger_b_joint_1_1", "finger_b_joint_2_1", "finger_b_joint_3_1",
             "finger_c_joint_1_1", "finger_c_joint_2_1", "finger_c_joint_3_1",
             "palm_finger_b_joint_1", "palm_finger_c_joint_1")}


def test_pinch_is_the_centroid_of_the_links_the_ROBOT_averages():
    """`morph/gripper/fingers.py`'s `_pinch()` is `0.5*(a + 0.5*(b+c))` over the three fingertip LINK
    ORIGINS. The prediction must be the same formula over the same links, or it predicts a
    different point than the robot measures -- which is the whole defect it exists to end.

    Mutation: average the three tips evenly, or swap a tip for the hand link -- must fail.
    """
    q = np.array([0.0, 0.1975, 0.3272, 0.2469, 0.0, 0.0, 0.0, 0.0])
    f = _open_hand()
    poses = dict((k, p) for k, _b, p, _R in M._links(q, fingers=f))
    t = [poses[n] for n in M.PINCH_TIPS]
    assert len(set(M.PINCH_TIPS)) == 3, "the tip list names fewer than three distinct links"
    want = 0.5 * (t[0] + 0.5 * (t[1] + t[2]))
    got = M.pinch(q, fingers=f)
    assert np.allclose(got, want, atol=1e-12), f"pinch is not that centroid: {got} vs {want}"


def test_every_PINCH_TIP_is_a_link_the_model_actually_yields():
    """A tip name that matches no link would raise -- or worse, silently pick up a stale pose. Same
    failure class as the `DECK_NEAR` names that stopped matching when `8df2c98` split the fingers
    out. The tips exist ONLY when commanded fingers are passed, which is the trap.

    Mutation: rename any entry of PINCH_TIPS -- must fail.
    """
    q = np.array([0.0, 0.1975, 0.3272, 0.2469, 0.0, 0.0, 0.0, 0.0])
    with_f = {t[0] for t in M._links(q, fingers=_open_hand())}
    missing = [n for n in M.PINCH_TIPS if n not in with_f]
    assert not missing, f"PINCH_TIPS names links the model never yields: {missing}"


def test_the_pinch_is_NOT_Gripper_Link1_and_the_gap_MOVES():
    """The LUT predicts `Gripper_Link1_1`; the robot measures the pinch. Treating one as the other
    is what `_fk_z`'s docstring does and what every `off_z` correction exists to paper over. The
    lever between them is not a constant -- so no constant can ever fix it, and anyone who
    "simplifies" this back to a fixed offset fails here.

    Measured across the shipped descent band, at fixed wrist: 53.2 -> 79.2mm.
    """
    f = _open_hand()
    levers = []
    for h1, h2, a1 in ((0.380, 0.443, 0.247), (0.297, 0.390, 0.247), (0.1975, 0.3272, 0.2469)):
        q = np.array([0.0, h1, h2, a1, 0.0, 0.0, 0.0, 0.0])
        link1 = dict((k, p) for k, _b, p, _R in M._links(q))["Gripper_Link1_1"]
        levers.append(float(link1[2] - M.pinch(q, fingers=f)[2]))
    assert min(levers) > 0.02, f"the pinch is not below Link1 at all: {levers}"
    assert max(levers) - min(levers) > 0.015, (
        f"the Link1->pinch lever looks CONSTANT ({levers}) -- if that is genuinely true a fixed "
        f"offset would fix `_fk_z`, and this test's premise is obsolete; re-measure and say so")


def test_pinch_takes_the_base_transform_like_every_other_consumer():
    """Base frame by default, world when handed `(t, R)` -- the same contract `_links`/`first_hit`
    use. A predictor that cannot be placed in world cannot be compared against a measurement.

    Mutation: ignore `base` -- must fail.
    """
    q = np.array([0.0, 0.1975, 0.3272, 0.2469, 0.0, 0.0, 0.0, 0.0])
    f = _open_hand()
    local = M.pinch(q, fingers=f)
    t = np.array([4.42, -6.30, 0.122])
    yaw = 0.7
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                  [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(M.pinch(q, fingers=f, base=(t, R)), t + R @ local, atol=1e-12)


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
