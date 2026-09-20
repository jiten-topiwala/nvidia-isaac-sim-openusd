"""The arm's collision geometry: the clearance and pad constants, the hand's boxes, and the
AABB/OBB predicates `ArmModel` checks with. Pure numpy; no dependency on the model."""
import numpy as np


# Metres of clearance every link keeps from every obstacle.
MARGIN = 0.02

# The arm vs its own deck is a fixed pair with no localisation error: a floor under MARGIN, not a
# deletion (FINDINGS: 8 mm under the planned goal's measured 11 mm on both links).
CHASSIS_PAD_PRIM = "/World/Geometry/robot/base_footprint/base/Box"

CHASSIS_PAD_M = 0.008

CHASSIS_PAD_LINKS = ("Gripper_Link2_1", "Gripper_Link3_1")

# "The arm is at this q8": one threshold per unit family, ~5-8 mm of hand.
Q8_TOL_M = 0.005
Q8_TOL_RAD = 0.02

# Every finger collider over the WHOLE allowed finger-joint range, in the hand-link frame; unioned
# into `Gripper_Link3_1` at load, whose shipped box covers the rest pose only.
FINGER_SWEEP_AABB = np.array([[-0.141, -0.067, -0.013], [0.141, 0.067, 0.166]])

# The palm alone, measured on the live stage; with the commanded finger values the check uses this
# plus the nine pads below instead of the sweep (FINDINGS: 554 cm3 against the sweep's 6764).
PALM_AABB = np.array([[-0.04409, -0.04556, -0.00089], [0.04999, 0.04556, 0.06363]])

# Finger pad colliders in their own link frames (usd/market_world_m1/payloads/base.usda), grown by
# the measured amount the meshes exceed them; finger b/c link 0 carry no collider.
FINGER_MESH_ALLOWANCE = 0.0113

FINGER_PAD_AABB = {
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

_FINGER_BOXES = {k: np.array([np.asarray(v[0], float) - FINGER_MESH_ALLOWANCE,
                              np.asarray(v[1], float) + FINGER_MESH_ALLOWANCE])
                 for k, v in FINGER_PAD_AABB.items()}

# Floor of the finger pseudo-link in `selfcol_aabb`, hand frame: inside the 22-29.7 mm window the
# fingers need no ignore entry (FINDINGS: the measured edges either side).
FINGER_SELFCOL_Z0 = 0.025


def sweep_top(R):
    """How far the hand box reaches above the hand frame's origin along world +z for world<-hand
    rotation `R`: the box's +z support function. Per sample, because a bound over an unconstrained
    wrist (0.2279) sits 161mm above the levelled wrist's real 0.067."""
    R = np.asarray(R, float)
    c = (FINGER_SWEEP_AABB[0] + FINGER_SWEEP_AABB[1]) / 2.0
    h = (FINGER_SWEEP_AABB[1] - FINGER_SWEEP_AABB[0]) / 2.0
    return float(R[2] @ c + np.abs(R[2]) @ h)


def _aabb_prunes(lo_a, hi_a, lo_b, hi_b, margin):
    """Broad phase: True where two world AABBs are far enough apart to skip the oriented test.
    `2 * margin` because `_obb_hits_obb` grows BOTH boxes by `margin`; pruning at `margin` would
    discard narrow-phase hits. Vectorised over leading axes; plain (3,) vectors work too."""
    return (np.any(hi_a + 2.0 * margin < lo_b, axis=-1)
            | np.any(hi_b + 2.0 * margin < lo_a, axis=-1))


def _obb_hits_obb(la, pa, Ra, lb, pb, Rb, margin=0.0, eps=1e-6):
    """Do two ORIENTED boxes overlap? `l*` are local [min,max] (2,3), `p*`/`R*` their frames. Full
    15-axis SAT (Ericson 4.4, B in A's frame once): face axes alone miss edge-on-edge, which here is
    the boom crossing a column. `eps` inflates near-parallel edge terms -- the conservative side."""
    ha = (la[1] - la[0]) * 0.5 + margin
    hb = (lb[1] - lb[0]) * 0.5 + margin
    d = (pb + Rb @ ((lb[0] + lb[1]) * 0.5)) - (pa + Ra @ ((la[0] + la[1]) * 0.5))
    R = Ra.T @ Rb                        # B's axes in A's frame
    t = Ra.T @ d                         # centre offset in A's frame
    A = np.abs(R) + eps

    # A's three face normals, then B's three.
    if np.any(np.abs(t) > ha + A @ hb):
        return False
    if np.any(np.abs(R.T @ t) > hb + A.T @ ha):
        return False
    # The nine edge-edge axes, a_i x b_j.
    for i in range(3):
        i1, i2 = (i + 1) % 3, (i + 2) % 3
        for j in range(3):
            j1, j2 = (j + 1) % 3, (j + 2) % 3
            ra = ha[i1] * A[i2, j] + ha[i2] * A[i1, j]
            rb = hb[j1] * A[i, j2] + hb[j2] * A[i, j1]
            if abs(t[i2] * R[i1, j] - t[i1] * R[i2, j]) > ra + rb:
                return False
    return True


def _seg_box_dist2(s0, s1, lo, hi):
    """Squared distance from segment [`s0`, `s1`] to the SOLID box [`lo`, `hi`], one frame, exact.
    Along the segment that distance is convex and piecewise quadratic, breaking only where a
    coordinate crosses a face plane, so the minimum is at a break, an end, or one piece's vertex.
    Evaluating the knots alone misses the vertex and over-reports."""
    s0 = np.asarray(s0, float)
    u = np.asarray(s1, float) - s0
    knots = [0.0, 1.0]
    for i in range(3):
        if abs(u[i]) > 1e-12:
            knots += [(lo[i] - s0[i]) / u[i], (hi[i] - s0[i]) / u[i]]
    knots = np.unique(np.clip(np.asarray(knots, float), 0.0, 1.0))
    cand = list(knots)
    for ta, tb in zip(knots[:-1], knots[1:]):
        p = s0 + 0.5 * (ta + tb) * u
        out = (p < lo) | (p > hi)
        if not out.any():
            return 0.0
        target = np.where(p < lo, lo, hi)
        a = float(u[out] @ u[out])
        if a > 0.0:
            b = float(u[out] @ (s0 - target)[out])
            cand.append(min(max(-b / a, ta), tb))
    pts = s0 + np.asarray(cand)[:, None] * u
    d = np.maximum(np.maximum(lo - pts, pts - hi), 0.0)
    return float((d * d).sum(axis=1).min())


def sphere_hits_obb(la, pa, Ra, lb, pb, Rb, margin=0.0):
    """Does the sphere inscribed in oriented box `la`/`pa`/`Ra` overlap oriented box `lb`/`pb`/`Rb`?
    Exact: the closest point of the box to the centre, against the radius. The radius is the box's
    LARGEST half-extent -- its INSPHERE, so this bounds the link only while the box is that
    sphere's own AABB. Both shapes grow by `margin`, as `_obb_hits_obb` grows its two."""
    half = (la[1] - la[0]) * 0.5
    r = float(half.max()) + margin
    q = Rb.T @ (pa + Ra @ ((la[0] + la[1]) * 0.5) - pb)
    d = q - np.clip(q, lb[0] - margin, lb[1] + margin)
    return bool(d @ d <= r * r)


def cylinder_hits_obb(la, pa, Ra, lb, pb, Rb, margin=0.0):
    """Does the cylinder inscribed in oriented box `la`/`pa`/`Ra` about its own local z overlap
    oriented box `lb`/`pb`/`Rb`? The cylinder IS its own box intersected with its capsule, but
    testing both against B is weaker than testing that intersection: B can meet each while
    missing the cylinder between them. So this BOUNDS the cylinder -- it over-reports, never the
    reverse, which is the only direction a self-check may err in. A capsule alone would be worse
    than the box it replaces: this wrist cylinder is squat and caps would triple its axial extent.
    `margin` > 0 rounds the rim outwards too, by at most sqrt(r^2 + 2*r*margin) - r. Exact needs a
    support-function narrow phase (GJK) over the true cylinder."""
    if not _obb_hits_obb(la, pa, Ra, lb, pb, Rb, margin):
        return False
    half = (la[1] - la[0]) * 0.5
    c = pa + Ra @ ((la[0] + la[1]) * 0.5) - pb
    e = Ra[:, 2] * half[2]
    r = float(max(half[0], half[1])) + margin
    return bool(_seg_box_dist2(Rb.T @ (c - e), Rb.T @ (c + e),
                               lb[0] - margin, lb[1] + margin) <= r * r)


# The two wrist links whose authored collider is a primitive, not a box: a cylinder about local z
# and a sphere. Dimensions come from each link's own `selfcol_aabb`, so they have one definition.
SELFCOL_PRIMITIVE = {"Gripper_Link1_1": cylinder_hits_obb, "Gripper_Link2_1": sphere_hits_obb}


def _obb_hits_aabb(c, h, R, ob, eps=1e-9):
    """Separating-axis test: oriented box (centre `c`, half-extents `h`, axes = columns of `R`)
    against axis-aligned `ob` = [[min],[max]]. True when no separating axis exists."""
    oc = (ob[0] + ob[1]) * 0.5
    oh = (ob[1] - ob[0]) * 0.5
    Rm = R.T                                  # Rm[i, j] = axis_i . e_j  (the AABB's axes are e_j)
    absR = np.abs(Rm) + eps
    t = R.T @ (oc - c)                        # centre offset in the OBB's frame

    if np.any(np.abs(t) > h + absR @ oh):
        return False
    t_w = oc - c
    if np.any(np.abs(t_w) > oh + absR.T @ h):
        return False
    for i in range(3):
        i1, i2 = (i + 1) % 3, (i + 2) % 3
        for j in range(3):
            j1, j2 = (j + 1) % 3, (j + 2) % 3
            ra = h[i1] * absR[i2, j] + h[i2] * absR[i1, j]
            rb = oh[j1] * absR[i, j2] + oh[j2] * absR[i, j1]
            if abs(t[i2] * Rm[i1, j] - t[i1] * Rm[i2, j]) > ra + rb:
                return False
    return True
