"""The narrow phase for the two wrist links that are not boxes -- `Gripper_Link1_1`
(a cylinder) and `Gripper_Link2_1` (a sphere). No Isaac, no GPU, any Python with numpy.
    .venv/bin/python3 tests/test_narrow_phase.py"""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from morph.arm.collision import (SELFCOL_PRIMITIVE, _obb_hits_obb, _seg_box_dist2,   # noqa: E402
                                 cylinder_hits_obb, sphere_hits_obb)
from morph.arm.model import ArmModel                                                 # noqa: E402

M = ArmModel.load(os.path.join(ROOT, "usd/_arm_model.json"),
                  os.path.join(ROOT, "usd/_closure_lut.json"))

I3 = np.eye(3)
UNIT = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

# The live `Gripper_Link1_1 x fingers` fold, from a logged retreat waypoint. Only wy and wx place
# this pair, and it is the only one whose boxes touch at the whole-arm pose.
LIVE_WRIST_FOLD = np.array([0.0376, 0.1196, 0.1381, 0.3267, 0.2995, -1.03, 0.81, 0.71])

# A pose that is a genuine fold for BOTH primitives: the boom through the wrist, not a box corner.
FOLDED = np.array([-1.3559, 0.9274, 0.9956, 0.183, -1.5653, 2.9748, -0.6333, -0.5844])

# A pose whose only box overlap is `Arm_Left_1 x Gripper_Link1_1` -- the primitive on side B of the
# pair, which is how 20 of the 23 primitive pairs sort. Found by search (seed 0), pinned literal.
PRIMITIVE_ON_SIDE_B = np.array(
    [-1.154712, 0.53869, 0.706705, 0.295292, 1.012544, -2.053403, 1.104228, 1.222228])


def _cyl_box(r, L):
    """The extracted AABB of a cylinder of radius `r` and half-height `L` about its local z."""
    return np.array([[-r, -r, -L], [r, r, L]])


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _cylinder_surface_gap(q8, prim, other):
    """Metres from the real cylinder's SURFACE to `other`'s oriented box, sampled 360 x 9 around
    the primitive -- the measurement the archive reports, recomputed rather than quoted."""
    poses = M.fk(q8)
    la, (pa, Ra) = M.selfcol_aabb[prim], poses[M.selfcol_frame[prim]]
    lb, (pb, Rb) = M.selfcol_aabb[other], poses[M.selfcol_frame[other]]
    half = (la[1] - la[0]) * 0.5
    r, L = float(max(half[0], half[1])), float(half[2])
    c = pa + Ra @ ((la[0] + la[1]) * 0.5)
    th = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
    z = np.linspace(-L, L, 9)
    T, Z = np.meshgrid(th, z, indexing="ij")
    local = np.stack([r * np.cos(T).ravel(), r * np.sin(T).ravel(), Z.ravel()], 1)
    q = (c + local @ Ra.T - pb) @ Rb
    d = np.maximum(np.maximum(lb[0] - q, q - lb[1]), 0.0)
    return float(np.linalg.norm(d, axis=1).min())


def test_segment_box_distance_matches_hand_computed_faces_edges_and_corners():
    """Every feature class of the box, computed by hand rather than by another implementation.
    The last case is the one that matters: the segment stays outside in z for its whole length and
    crosses the x face partway, so the minimum lies at the VERTEX of one piece's quadratic at
    t = 0.6 and not at any knot. p(t) = (4t, 0, 5 - 2t); on t in [0.25, 1] the active axes are x
    and z, d2 = (4t - 1)^2 + (4 - 2t)^2, minimised at t = 0.6 giving 1.96 + 7.84 = 9.8.

    Mutation: evaluate the knots only and drop the per-piece vertex -- that case returns 12.25.
    """
    cases = [
        ("over the +z face", (-5, 0, 2), (5, 0, 2), 1.0),
        ("endpoint off a corner", (2, 2, 2), (3, 3, 3), 3.0),
        ("parallel to an edge", (2, -5, 2), (2, 5, 2), 2.0),
        ("parallel to a face", (3, 0, 0), (3, 0, 5), 4.0),
        ("straight through", (-5, 0, 0), (5, 0, 0), 0.0),
        ("piece vertex, off every knot", (0, 0, 5), (4, 0, 3), 9.8),
    ]
    for name, s0, s1, want in cases:
        got = _seg_box_dist2(np.array(s0, float), np.array(s1, float), UNIT[0], UNIT[1])
        assert abs(got - want) < 1e-12, f"{name}: {got} != {want}"
        # A distance is symmetric in the segment's direction.
        rev = _seg_box_dist2(np.array(s1, float), np.array(s0, float), UNIT[0], UNIT[1])
        assert abs(rev - want) < 1e-12, f"{name} reversed: {rev} != {want}"


def test_segment_box_distance_never_over_reports_against_brute_force():
    """Over-reporting a DISTANCE is the one unsafe direction in this file: the capsule compares it
    against the radius, so a distance that is too large reads CLEAR on metal. Driven against dense
    sampling along the segment, which owes nothing to the piecewise-quadratic argument, and pinned
    BOTH ways -- never above the sample, and never meaningfully below it either.

    Mutation: build the knots from the `lo` faces only -- it over-reports on about a fifth of these
    and by up to 0.38 m, and every hand-computed case above still passes.
    """
    rng = np.random.default_rng(5)
    worst_under = 0.0
    for _ in range(400):
        lo = -rng.uniform(0.05, 0.5, 3)
        hi = rng.uniform(0.05, 0.5, 3)
        s0, s1 = rng.uniform(-1.0, 1.0, 3), rng.uniform(-1.0, 1.0, 3)
        p = s0 + np.linspace(0.0, 1.0, 4001)[:, None] * (s1 - s0)
        d = np.maximum(np.maximum(lo - p, p - hi), 0.0)
        brute = float((d * d).sum(axis=1).min())
        got = _seg_box_dist2(s0, s1, lo, hi)
        assert got <= brute + 1e-12, (
            f"{got} over-reports against a sampled {brute}: the capsule would read clear of a box "
            f"it reaches, which is the one direction this may not err in")
        worst_under = max(worst_under, brute - got)
    assert worst_under < 1e-6, (
        f"the exact answer sits {worst_under:.3e} below dense sampling -- too far to be "
        f"discretisation, so it is not finding the same minimum")


def test_the_sphere_predicate_flips_exactly_at_the_hand_computed_distance():
    """Face, edge and corner of the unit box, each with the closest distance written out by hand.
    Asserted either side of the flip so the predicate cannot pass by being constant.

    Mutations: compare against the distance to the box CENTRE instead of to the clamped closest
    point -- the face case then needs r > sqrt(3) x its true radius; take the radius from the box's
    SMALLEST half-extent -- the anisotropic case below then needs 1.67x its true radius.
    """
    for name, centre, want in (("face", (3.0, 0.0, 0.0), 2.0),
                               ("edge", (3.0, 3.0, 0.0), math.sqrt(8.0)),
                               ("corner", (3.0, 3.0, 3.0), math.sqrt(12.0))):
        c = np.array(centre, float)
        for r, hit in ((want * 0.999, False), (want * 1.001, True)):
            la = np.array([[-r, -r, -r], [r, r, r]])
            got = sphere_hits_obb(la, c, I3, UNIT, np.zeros(3), I3)
            assert got is hit, f"{name}: r={r} against a hand distance of {want} reported {got}"
    # A centre INSIDE the box hits at any radius, zero included.
    zero = np.zeros((2, 3))
    assert sphere_hits_obb(zero, np.array([0.5, 0.0, 0.0]), I3, UNIT, np.zeros(3), I3)
    assert not sphere_hits_obb(zero, np.array([1.5, 0.0, 0.0]), I3, UNIT, np.zeros(3), I3)
    # Off its link origin, so the `Ra @ box centre` term is exercised: a sphere whose box sits
    # 3.0 along local x, on a frame turned a quarter turn, is 2.0 from the box in world y.
    for r, hit in ((1.999, False), (2.001, True)):
        la = np.array([[-r + 3.0, -r, -r], [r + 3.0, r, r]])
        assert sphere_hits_obb(la, np.zeros(3), _rot_z(math.pi / 2), UNIT, np.zeros(3), I3) is hit, (
            f"r={r} against a hand distance of 2.0 with the box 3.0 off the link origin")

    # An extracted AABB is a cube only up to float noise, and the radius is read off it. Taking the
    # LARGEST half-extent is the conservative reading of an ambiguous box; the smallest under-reports.
    c = np.array([3.0, 0.0, 0.0])
    for r, hit in ((2.0 - 1e-9, False), (2.0 + 1e-9, True)):
        la = np.array([[-r, -0.8 * r, -0.6 * r], [r, 0.8 * r, 0.6 * r]])
        assert sphere_hits_obb(la, c, I3, UNIT, np.zeros(3), I3) is hit, (
            f"an anisotropic box with a largest half-extent of {r} against a gap of 2.0")


def test_the_sphere_and_box_grow_together_with_margin():
    """`_obb_hits_obb` grows BOTH boxes by `margin`, so the wrist predicates must too or a caller
    passing a margin silently gets a different clearance from the two halves of the check.

    Mutation: grow only the sphere -- the flip lands at 2.0 - margin instead of 2.0 - 2 * margin.
    """
    c = np.array([3.0, 0.0, 0.0])
    m = 0.25
    for r, hit in ((2.0 - 2.0 * m - 1e-9, False), (2.0 - 2.0 * m + 1e-9, True)):
        la = np.array([[-r, -r, -r], [r, r, r]])
        assert sphere_hits_obb(la, c, I3, UNIT, np.zeros(3), I3, m) is hit, (
            f"r={r}, margin={m}: the gap is 2.0 and both shapes grow by {m}")


def test_the_cylinder_predicate_refuses_the_phantom_corner_its_box_invents():
    """The live defect in synthetic form. A cylinder turned 45 degrees about its own axis presents
    a box CORNER at sqrt(2) x its radius while the metal is still a circle of radius r, so the
    oriented box reports a hit across a band of 0.414 r that is entirely air. Swept through the
    true contact to prove the predicate is not merely permissive.

    Mutation: return `_obb_hits_obb` -- the clear side of the sweep then reports a hit.
    """
    r, L = 0.4, 0.1
    la = _cyl_box(r, L)
    Ra = _rot_z(math.radians(45.0))
    for gap, exact in ((-1e-6, True), (1e-6, False), (0.2 * r, False), (0.5 * r, False)):
        pa = np.array([1.0 + r + gap, 0.0, 0.0])
        assert cylinder_hits_obb(la, pa, Ra, UNIT, np.zeros(3), I3) is exact, (
            f"gap {gap:+.3g} m from the cylinder SURFACE reported the wrong verdict")
    # ... and the box it replaces is wrong over that whole band, which is what this buys.
    for gap in (1e-6, 0.2 * r):
        pa = np.array([1.0 + r + gap, 0.0, 0.0])
        assert _obb_hits_obb(la, pa, Ra, UNIT, np.zeros(3), I3), (
            f"gap {gap:+.3g}: the box no longer over-reports, so this test proves nothing")

    # The same clearance with the cylinder sitting OFF its link origin. Both wrist boxes are
    # centred today, so this is the only thing that exercises the centre term at all.
    dx = 0.5
    off = np.array([[-r + dx, -r, -L], [r + dx, r, L]])
    pa = np.array([1.0 + r + 0.02, 0.0, 0.0]) - Ra @ np.array([dx, 0.0, 0.0])
    assert _obb_hits_obb(off, pa, Ra, UNIT, np.zeros(3), I3), "fixture: the boxes must overlap here"
    assert not cylinder_hits_obb(off, pa, Ra, UNIT, np.zeros(3), I3), (
        "the capsule was placed at the link origin rather than at its box centre")


def test_the_cylinder_is_its_capsule_intersected_with_its_own_box():
    """A capsule ALONE is not the cylinder: it adds a hemisphere of radius r to each flat end, and
    this wrist cylinder is squat (r 21 mm, half-height 10.5 mm), so a capsule triples its axial
    extent and is WORSE than the box it replaces on an axial approach. The box's z half-extent IS
    the cylinder's, so intersecting the two gives the solid cylinder exactly.

    Mutation: drop the `_obb_hits_obb` conjunction -- an approach onto the flat cap then reports a
    hit out to a full radius of clear air.
    """
    r, L = 0.5, 0.1
    la = _cyl_box(r, L)
    for gap, exact in ((-1e-3, True), (1e-3, False), (0.5 * r, False), (0.9 * r, False)):
        pa = np.array([0.0, 0.0, 1.0 + L + gap])
        assert cylinder_hits_obb(la, pa, I3, UNIT, np.zeros(3), I3) is exact, (
            f"gap {gap:+.3g} m above the flat cap reported the wrong verdict")
    # The capsule alone spans the whole band above, so the conjunction is doing real work.
    for gap in (1e-3, 0.5 * r, 0.9 * r):
        s = np.array([0.0, 0.0, 1.0 + gap])
        assert _seg_box_dist2(s, s + np.array([0.0, 0.0, 2.0 * L]), UNIT[0], UNIT[1]) <= r * r, (
            f"gap {gap:+.3g}: the capsule no longer over-reports, so this test proves nothing")


def test_neither_predicate_reports_clear_while_the_true_shape_overlaps():
    """The direction rule. Both predicates over-approximate -- a radius taken as the box's LARGEST
    half-extent, and a margin-grown cylinder whose rim rounds outwards -- so they may report a hit
    on air, never clear on metal. Driven against points sampled INSIDE the true primitive: a point
    in both solids is proof of overlap that owes nothing to either predicate.

    A deep overlap proves nothing -- the box would catch it too. The primitive is placed against
    the box's surface so the overlaps found are GRAZING, and the test asserts it reached that
    regime rather than assuming it.

    The primitive's box is ANISOTROPIC, with its largest half-extent on whichever axis the draw
    puts it. A cube exercises no choice of radius at all: `half.max()`, `half[0]` and
    `max(half[0], half[1])` are the same number, and a predicate reading any one of them passes.

    The truth sampled is the solid each predicate MODELS -- the ball of `half.max()` for the
    sphere, and for the cylinder the capsule of `max(half[0], half[1])` clipped back inside its own
    box, which is what the box conjunction leaves. Both are supersets of nothing the predicate may
    miss, so a sampled point inside the partner box is a real overlap either way.

    Mutations: sphere radius `half.max()` -> `half[0]`; cylinder `max(half[0], half[1])` ->
    `half[0]`; drop `margin` from either shape; return clear from either predicate.
    """
    rng = np.random.default_rng(11)
    u = rng.normal(size=(6000, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    ball = u * rng.random((6000, 1)) ** (1.0 / 3.0)
    # sqrt, not uniform: a uniform radius crowds the axis and under-samples the RIM, which is
    # exactly where a cylinder predicate is tight.
    ang = rng.random((6000, 1)) * 2.0 * math.pi
    disc = np.hstack([np.cos(ang), np.sin(ang)]) * np.sqrt(rng.random((6000, 1)))
    overlapping = grazing = 0
    for _ in range(3000):
        margin = float(rng.choice([0.0, 0.005, 0.02]))
        sphere = bool(rng.integers(2))
        half = rng.uniform(0.02, 0.2, 3)
        la = np.vstack([-half, half])
        r = float(half.max() if sphere else max(half[0], half[1]))
        L = float(half[2])
        lb = np.vstack([-rng.uniform(0.02, 0.3, 3), rng.uniform(0.02, 0.3, 3)])
        Ra, Rb = _rand_R(rng), _rand_R(rng)
        pb = rng.uniform(-0.35, 0.35, 3)
        # On the box's surface, offset along a random normal by about the primitive's own size:
        # the two solids then straddle contact instead of being far apart or deeply merged.
        face = rng.integers(3)
        side = 1.0 if rng.integers(2) else -1.0
        at = rng.uniform(lb[0], lb[1])
        at[face] = (lb[1] if side > 0 else lb[0])[face]
        n = np.zeros(3)
        n[face] = side
        reach = r if sphere else math.hypot(r, L)
        pa = pb + Rb @ (at + n * (rng.uniform(0.75, 1.05) * reach + margin))
        if sphere:
            local = ball * r
        else:
            local = np.hstack([disc * r, (rng.random((6000, 1)) * 2 - 1) * L])
            # the box conjunction clips the capsule back to its own lateral extents
            local = local[np.all(np.abs(local[:, :2]) <= half[:2], axis=1)]
        q = (pa + local @ Ra.T - pb) @ Rb
        # The TRUE solid against the grown box: any sampled point inside it is a real overlap.
        lo_b, hi_b = lb[0] - margin, lb[1] + margin
        within = np.all((q >= lo_b) & (q <= hi_b), axis=1)
        if not within.any():
            continue
        overlapping += 1
        # How far inside the deepest sampled point sits: small means the test reached contact.
        depth = float(np.minimum(q[within] - lo_b, hi_b - q[within]).min(axis=1).max())
        if depth < 0.001:
            grazing += 1
        fn = sphere_hits_obb if sphere else cylinder_hits_obb
        assert fn(la, pa, Ra, lb, pb, Rb, margin), (
            f"{'sphere' if sphere else 'cylinder'} reported CLEAR while a point of the true solid "
            f"is {depth * 1000:.3f} mm inside the box: r={r:.4f} L={L:.4f} margin={margin}")
    assert overlapping > 300, f"only {overlapping} overlapping configurations; this proves little"
    assert grazing > 50, (
        f"only {grazing} of {overlapping} overlaps are within 1 mm of contact -- the generator "
        f"never reaches the case that can break the predicate, so this test proves little")


def _rand_R(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], float)


def test_the_live_wrist_fold_is_clear_metal_and_a_hit_only_as_boxes():
    """The blocker, pinned BOTH ways so the test proves the difference rather than the verdict:
    at the wrist configuration three separate live retreats ended in, the oriented boxes report a
    hit while the cylinder's surface is 7 mm clear of the finger box.

    Mutation: route this pair back through `_obb_hits_obb` -- `self_collides` names it again.
    """
    prim, other = "Gripper_Link1_1", f"{M.hand_link}::fingers"
    assert (prim, other) in M._selfcol_pair_list(), "the pair left the self-collision list"
    poses = M.fk(LIVE_WRIST_FOLD)
    pa, Ra = poses[M.selfcol_frame[prim]]
    pb, Rb = poses[M.selfcol_frame[other]]
    la, lb = M.selfcol_aabb[prim], M.selfcol_aabb[other]

    assert _obb_hits_obb(la, pa, Ra, lb, pb, Rb, 0.0), (
        "the boxes no longer collide here, so this pose no longer demonstrates the defect")
    gap = _cylinder_surface_gap(LIVE_WRIST_FOLD, prim, other)
    assert gap > 0.006, f"the cylinder surface is only {gap * 1000:.2f} mm clear; archive says 7.0"
    assert not cylinder_hits_obb(la, pa, Ra, lb, pb, Rb, 0.0), (
        f"the cylinder predicate still reports a fold with {gap * 1000:.2f} mm of air around it")
    assert M.self_collides(LIVE_WRIST_FOLD) is None, (
        f"the live wrist fold is still refused: {M.self_collides(LIVE_WRIST_FOLD)}")


def test_a_genuine_fold_still_reads_hit_through_the_wrist_predicates():
    """The other half of the pin. Both primitives are inside the boom at this pose, not merely
    inside its box corner, so the wrist predicates must refuse it as the box test did.

    Mutation: return False from either predicate -- both assertions below fail.
    """
    poses = M.fk(FOLDED)
    for prim in ("Gripper_Link1_1", "Gripper_Link2_1"):
        pa, Ra = poses[M.selfcol_frame[prim]]
        pb, Rb = poses[M.selfcol_frame["Arm_Left_1"]]
        fn = SELFCOL_PRIMITIVE[prim]
        assert fn(M.selfcol_aabb[prim], pa, Ra, M.selfcol_aabb["Arm_Left_1"], pb, Rb, 0.0), (
            f"{prim} reads clear of the boom at a pose it is folded into")
    assert M.self_collides(FOLDED) is not None, "FOLDED stopped self-colliding"


def test_a_primitive_is_reached_whichever_side_of_the_pair_it_sorts_to():
    """`_selfcol_pair_list` sorts its names, so a primitive lands on side B of 20 of its 23 pairs
    and on side A of 3. `_pair_hits` swaps them before dispatch; without that swap the narrow
    phase reaches only those 3 and the rest silently keep their boxes.

    Mutation: delete the swap in `_pair_hits` -- this pose reads a fold again.
    """
    a, b = "Arm_Left_1", "Gripper_Link1_1"
    assert (a, b) in M._selfcol_pair_list() and b in SELFCOL_PRIMITIVE, "the pair stopped sorting this way"
    poses = M.fk(PRIMITIVE_ON_SIDE_B)
    pa, Ra = poses[M.selfcol_frame[a]]
    pb, Rb = poses[M.selfcol_frame[b]]
    assert _obb_hits_obb(M.selfcol_aabb[a], pa, Ra, M.selfcol_aabb[b], pb, Rb, 0.0), (
        "the boxes no longer overlap here, so this pose cannot show which predicate ran")
    assert not M._pair_hits(a, b, poses, 0.0), (
        f"{a} x {b} still reads the box verdict with the cylinder on side B")
    assert M.self_collides(PRIMITIVE_ON_SIDE_B) is None, (
        f"some pair still folds here: {M.self_collides(PRIMITIVE_ON_SIDE_B)}")


def test_the_cylinder_and_box_grow_together_with_margin():
    """The cylinder's `margin` arithmetic, isolated from its box test: turned 45 degrees about its
    own axis the box reaches sqrt(2) further than the capsule, so over this band the capsule alone
    decides and the flip sits at a centre distance of 1 + r + 2 * margin.

    Mutations: drop `+ margin` from the radius, or from the box B it measures against -- either
    moves the flip to 1 + r + margin; grow the radius by 2 * margin -- 1 + r + 3 * margin.
    """
    r, L, m = 0.4, 0.1, 0.05
    la = _cyl_box(r, L)
    Ra = _rot_z(math.radians(45.0))
    for d, hit in ((1.0 + r + 2.0 * m - 1e-4, True), (1.0 + r + 2.0 * m + 1e-4, False)):
        pa = np.array([d, 0.0, 0.0])
        assert cylinder_hits_obb(la, pa, Ra, UNIT, np.zeros(3), I3, m) is hit, (
            f"centre at {d:.4f} m, margin {m}: the flip belongs at {1.0 + r + 2.0 * m:.4f}")
        # ... and the box half of the conjunction is not what decided: it reaches further.
        assert _obb_hits_obb(la, pa, Ra, UNIT, np.zeros(3), I3, m), (
            "the grown boxes separate here too, so this test is not isolating the capsule")


def test_only_the_two_primitive_links_leave_the_box_test():
    """"Route those two links' pairs through the narrow phase; every other pair stays as it is."
    This is a DISPATCH check, not an independent computation: for a pair with no primitive on
    either side `_pair_hits` reduces to `_obb_hits_obb` on the same arguments, so the two agree by
    construction and the only thing that can move is which function the pair reaches.

    Mutation: route every pair through a primitive predicate -- the boom and the mast parts are
    not spheres and disagree immediately.
    """
    assert set(SELFCOL_PRIMITIVE) == {"Gripper_Link1_1", "Gripper_Link2_1"}, sorted(SELFCOL_PRIMITIVE)
    lo, hi = (np.asarray(b, float) for b in M.bounds())
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(400):
        q = lo + rng.random(8) * (hi - lo)
        if not M.in_lut_domain(q[2] - q[1], q[3]):
            continue
        poses = M.fk(q)
        for a, b in M._selfcol_pair_list():
            if a in SELFCOL_PRIMITIVE or b in SELFCOL_PRIMITIVE:
                continue
            pa, Ra = poses[M.selfcol_frame[a]]
            pb, Rb = poses[M.selfcol_frame[b]]
            box = _obb_hits_obb(M.selfcol_aabb[a], pa, Ra, M.selfcol_aabb[b], pb, Rb, 0.0)
            assert M._pair_hits(a, b, poses, 0.0) is box, f"{a} x {b} changed verdict"
            checked += 1
    assert checked > 5000, f"only {checked} non-primitive pair verdicts compared"


def test_the_primitive_dimensions_come_from_the_extracted_box():
    """The radius and half-height are read off `selfcol_aabb`, so the numbers have one definition
    and track `tools/extract_arm_model.py`. The largest half-extent is the box's INSPHERE, not its
    circumsphere, so it bounds the link only while the box really is that primitive's own AABB:
    a cube for the sphere, a square cross-section for the cylinder, both on the link origin. A
    genuine cube collider would pass the predicate while (sqrt(3) - 1) * r of corner metal goes
    untested, so the premise is pinned here rather than assumed.

    Mutation: point either entry at a link whose box is a genuine box -- the shape check fails.
    """
    sphere = M.selfcol_aabb["Gripper_Link2_1"]
    half = (sphere[1] - sphere[0]) * 0.5
    assert np.allclose(half, half[0], atol=1e-6), f"the sphere's box is not a cube: {half}"
    assert np.allclose((sphere[0] + sphere[1]) * 0.5, 0.0, atol=1e-6), "sphere box is off-centre"

    cyl = M.selfcol_aabb["Gripper_Link1_1"]
    half = (cyl[1] - cyl[0]) * 0.5
    assert abs(half[0] - half[1]) < 1e-6, f"the cylinder's cross-section is not square: {half}"
    assert half[2] < half[0], f"the cylinder's axis is not its box's z: {half}"
    assert np.allclose((cyl[0] + cyl[1]) * 0.5, 0.0, atol=1e-6), "cylinder box is off-centre"

    # `self_collides` prunes with the vectorised AABB of the BOX, so a primitive reaching outside its
    # own box could be pruned while overlapping. Both stay inscribed to within extraction noise.
    for link, reach in (("Gripper_Link2_1", np.full(3, (sphere[1] - sphere[0]).max() * 0.5)),
                        ("Gripper_Link1_1", np.array([max(half[0], half[1]), max(half[0], half[1]),
                                                      half[2]]))):
        box = M.selfcol_aabb[link]
        out = (reach - (box[1] - box[0]) * 0.5).max()
        assert out <= 1e-6, (
            f"the primitive derived for {link} reaches {out * 1e6:.3f} um outside its own box -- "
            f"the broad phase bounds the BOX, so it can prune an overlap this predicate reports")

    # No pair carries a primitive on BOTH sides, so `_pair_hits` orienting on the first match is
    # unambiguous. It would stay conservative if one ever did -- side b keeps its box.
    both = [p for p in M._selfcol_pair_list()
            if p[0] in SELFCOL_PRIMITIVE and p[1] in SELFCOL_PRIMITIVE]
    assert not both, both


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
