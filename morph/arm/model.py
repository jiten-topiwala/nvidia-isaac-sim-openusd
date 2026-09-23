"""Pure-numpy kinematics of arm-1: tree FK, the closure LUT for the passives, IK, and the link
boxes the collision predicates run over. Imported by BOTH interpreters (planner venv 3.10, Isaac
3.12): no Isaac imports, no dependency beyond numpy."""
import json
import math

import numpy as np

from morph.arm.collision import (FINGER_SELFCOL_Z0, FINGER_SWEEP_AABB, PALM_AABB, Q8_TOL_M,
                                 Q8_TOL_RAD, SELFCOL_PRIMITIVE, _FINGER_BOXES, _aabb_prunes,
                                 _obb_hits_aabb, _obb_hits_obb)


def _quat_to_R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], float)


def _axis_R(axis, ang):
    c, s = math.cos(ang), math.sin(ang)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)
    if axis == "Z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)
    raise ValueError(f"joint axis {axis!r} is not X/Y/Z")


def _axis_angle_R(axis, ang):
    """Rodrigues: rotation by `ang` about `axis`, which is normalised here. Exactly the identity
    at `ang` 0, which is what keeps the nominal grasp bit-identical to an unrotated solve."""
    k = np.asarray(axis, float)
    k = k / max(1e-9, float(np.linalg.norm(k)))
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], float)
    return np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)


_AXIS_V = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}


def _mm(c):
    """A `clearance` reading as a log string; -1.0 means "no margin", not "-1000 mm"."""
    return "OVERLAPPING" if c < 0.0 else f"{c * 1000:.1f} mm"

# `ik_repair` seed noise, a fraction of each joint's OWN range; keep it in [0.05, 0.30].
IK_SEED_SPREAD = 0.15
# `Q8_TOL`s of travel one clearance band is worth in `ik_repair`'s score. Measured band: [3, 10].
IK_REPAIR_TRAVEL_K = 5.0
# Approach-yaw fan, radians, NOMINAL FIRST so the tuned grasp is offered before any alternative.
# A calibration knob: widen it where the shelf leaves room beside the object, narrow it where not.
GRASP_YAWS = tuple(math.radians(d) for d in (0.0, 15.0, -15.0, 30.0, -30.0))


# The carried payload as `_links` names it: a pseudo-link posed at the hand, not a real link.
HELD_LINK = "__held__"

# Standoff from a closure passive's hard stop, metres. A pose exactly at the stop is in range, so
# nothing clips it and it executes with no headroom against model error -- which is what this is.
PASSIVE_KEEPOUT_M = 0.005

class ArmModel:
    """Arm-1's kinematic tree plus the closure LUT. All poses are in the BASE (chassis) frame."""

    # The canonical q8 order: [th, h1, h2, a1, hb, wz, wy, wx]. Every consumer uses it.
    Q8 = ["BaseJoint_1", "ColumnLeftBearingJoint_1", "ColumnRightBearingJoint_1", "ArmLeftJoint_1",
          "HandBearingJoint_1", "gripper_z_rotation_1", "gripper_y_rotation_1",
          "gripper_x_rotation_1"]
    # Q8 MIXES UNITS: never one scalar tolerance over all eight (a column is 3.24 m of hand per
    # metre; the largest revolute lever is 0.424 m/rad).
    Q8_LIN = [1, 2, 3]                  # ColumnLeft, ColumnRight, ArmLeft -- metres
    Q8_ROT = [0, 4, 5, 6, 7]            # BaseJoint, HandBearing, wz, wy, wx -- radians
    # Mid-envelope `ik` seed; starting on a joint limit stalls the descent.
    Q8_HOME = np.array([0.0, 0.60, 0.60, 0.25, 0.0, 0.0, 0.0, 0.0])

    @staticmethod
    def q8_resid(q8, goal8):
        """(prismatic residue METRES, revolute residue RADIANS); compare against Q8_TOL_M/_RAD."""
        d = np.abs(np.asarray(q8, float) - np.asarray(goal8, float))
        return float(d[ArmModel.Q8_LIN].max()), float(d[ArmModel.Q8_ROT].max())

    def __init__(self, model, lut):
        self.m, self.lut = model, lut
        self.joints = {j["name"]: j for j in model["joints"]}
        self.children = {}
        for j in model["joints"]:
            self.children.setdefault(j["parent"], []).append(j)
        # NOT every group link has geometry (Rotation_Link_Right_1 has none), so iterate link_aabb.
        self.link_aabb = {k: np.asarray(v, float) for k, v in model["link_aabb"].items()}
        self.hand_group = set(model["hand_group"])
        self.column_group = set(model["column_group"])
        self.hand_link = model["hand_link"]

        # The links that HOLD the object and may therefore touch it: everything downstream of the WRIST
        # BEARING, which is part of it. The boom and the columns are not, and stay checked.
        self.hand_subtree = set()
        stack = [self.joints["HandBearingJoint_1"]["child"]]
        while stack:
            _link = stack.pop()
            self.hand_subtree.add(_link)
            for _j in self.children.get(_link, []):
                stack.append(_j["child"])
        # What a "grasped" box may touch: the links that hold the object, and the object itself.
        self.payload_side = self.hand_subtree | {HELD_LINK}

        # Two geometries, because the uses want OPPOSITE errors: `link_aabb` is fat so a swinging finger
        # cannot tunnel a shelf board between checks; `selfcol_aabb` is tight, since fat self-overlaps.
        self.selfcol_aabb = {k: v.copy() for k, v in self.link_aabb.items()}
        # The mast's union box spans the empty gap BETWEEN its twin rails, which the hand passes through
        # in normal operation, so the self-check uses the six real sub-meshes instead.
        _parts = model.get("link_parts", {})
        for _link, _boxes in _parts.items():
            if _link not in self.selfcol_aabb:
                continue
            self.selfcol_aabb.pop(_link)
            for _nm, _b in _boxes.items():
                if _nm in self.SELFCOL_HULL_PARTS:
                    continue
                self.selfcol_aabb[f"{_link}::{_nm}"] = np.asarray(_b, float)
        # `link_aabb`'s hand entry already unions the fingers, which go in again below: for the
        # self-check that double-counts, so the palm is taken alone.
        self.selfcol_aabb[self.hand_link] = PALM_AABB.copy()
        self.selfcol_aabb[f"{self.hand_link}::fingers"] = np.array(
            [[FINGER_SWEEP_AABB[0][0], FINGER_SWEEP_AABB[0][1], FINGER_SELFCOL_Z0],
             FINGER_SWEEP_AABB[1]], float)
        # Pseudo-links ride the frame they were carved out of.
        self.selfcol_frame = {k: (k.split("::", 1)[0] if "::" in k else k)
                              for k in self.selfcol_aabb}
        # The fingers hang off the hand link, so their sweep rides with it: one box, not a link.
        h = self.link_aabb[self.hand_link]
        self.link_aabb[self.hand_link] = np.vstack([np.minimum(h[0], FINGER_SWEEP_AABB[0]),
                                                    np.maximum(h[1], FINGER_SWEEP_AABB[1])])
        self._selfcol_pairs = None
        self._selfcol_idx = None
        self._selfcol_ext = None
        # Bearing separation of the closed four-bar, from MODEL GEOMETRY and never a literal: the loop
        # spans twice this. tests/test_closure_form.py fails if a model edit breaks the agreement.
        self.D2 = 2 * abs(float(self.joints["RotationLeftJoint_1"]["localPos0"][0]))
        # Closure passives with a REAL limit: the width filter drops the +-pi "no limit" default and
        # the 0.0/0.0 pair, which kept would reject every pose.
        self.passive_lims = [(n, self.joints[n]["lower"], self.joints[n]["upper"])
                             for n in lut["passives"]
                             if 0.0 < self.joints[n]["upper"] - self.joints[n]["lower"]
                             < 2 * math.pi - 1e-6]
        for j in self.joints.values():
            j["_p0"] = np.asarray(j["localPos0"], float)
            j["_R0"] = _quat_to_R(*j["localRot0"])
            j["_p1"] = np.asarray(j["localPos1"], float)
            j["_R1"] = _quat_to_R(*j["localRot1"])

    @classmethod
    def load(cls, model_path, lut_path):
        with open(model_path) as fh:
            model = json.load(fh)
        with open(lut_path) as fh:
            lut = json.load(fh)
        return cls(model, lut)

    # Closure LUT: same semantics as kinematics._lut_cell/_lut_interp.
    @staticmethod
    def _cell(vals, v):
        v = min(max(v, vals[0]), vals[-1])
        for i in range(len(vals) - 2, -1, -1):
            if v >= vals[i]:
                return i, (v - vals[i]) / (vals[i + 1] - vals[i])
        return 0, 0.0

    def passives(self, dh, a1):
        """The 4 passive linkage joints at (h2 - h1, a1), bilinear over the LUT grid."""
        i, fi = self._cell(self.lut["dh"], dh)
        j, fj = self._cell(self.lut["a1"], a1)
        g = self.lut["grid"]
        vals = [(1 - fi) * (1 - fj) * g[i][j][k] + fi * (1 - fj) * g[i + 1][j][k]
                + (1 - fi) * fj * g[i][j + 1][k] + fi * fj * g[i + 1][j + 1][k]
                for k in range(4)]
        return dict(zip(self.lut["passives"], vals))

    def passives_closed_form(self, dh, a1):
        """The same 4 passives as `passives`, from the closed forms the LUT samples. NOT a drop-in:
        the shipped path was tuned against the bilinear LUT and the two disagree between grid
        points. `ContactCylinder`s are 0, as in every LUT cell (the bakes: Joint2_1 0.0445, Joint_1_1 0)."""
        return {"RotationLeftJoint_1": math.atan(dh / self.D2),
                "ArmRightJoint_1": self.D2 - math.hypot(self.D2, dh) - a1,
                "ContactCylinderJoint_1_1": 0.0,
                "ContactCylinderJoint2_1": 0.0}

    def passives_jacobian(self, dh, a1):
        """{passive: (d/d_dh, d/d_a1)} of `passives_closed_form`, exact -- the reason these forms
        are worth carrying. A finite difference of the LUT gives the SECANT of a bilinear cell,
        which is flat in dh across each cell and misses the true peak entirely.

        `RotationLeftJoint_1`'s gain in dh is `(1/D2) / (1 + (dh/D2)**2)`, maximal at dh = 0 where
        it is exactly `1/D2` = 10.0 rad/m. The repo's "measured peak 9.870 rad/m" is that same
        expression sampled off-grid, not a different number."""
        return {"RotationLeftJoint_1": ((1.0 / self.D2) / (1.0 + (dh / self.D2) ** 2), 0.0),
                "ArmRightJoint_1": (-dh / math.hypot(self.D2, dh), -1.0),
                "ContactCylinderJoint_1_1": (0.0, 0.0),
                "ContactCylinderJoint2_1": (0.0, 0.0)}

    def in_lut_domain(self, dh, a1):
        """False outside the grid, where `passives` clamps. True does NOT mean reachable --
        `passive_violations` is the other half of state validity."""
        return (self.lut["dh"][0] <= dh <= self.lut["dh"][-1]
                and self.lut["a1"][0] <= a1 <= self.lut["a1"][-1])

    def a1_ceil(self, dh):
        """Highest `a1` the closure holds at this `dh`, inverted on the same bilinear LUT row that
        `passive_violations` reads (the closed form runs up to 0.49mm high) and shaved 1e-9 so
        that check accepts it. Valid only inside the LUT's dh domain: `in_lut_domain` decides."""
        name = "ArmRightJoint_1"
        lim = next((lo for n, lo, _hi in self.passive_lims if n == name), None)
        if lim is None:
            return float(self.lut["a1"][-1])
        k = self.lut["passives"].index(name)
        i, fi = self._cell(self.lut["dh"], dh)
        g, nodes = self.lut["grid"], self.lut["a1"]
        row = [(1 - fi) * g[i][j][k] + fi * g[i + 1][j][k] for j in range(len(nodes))]
        # row falls with a1; the ceiling is where it meets the stop, linear inside the cell
        for j in range(len(nodes) - 1):
            if row[j] >= lim >= row[j + 1]:
                span = row[j] - row[j + 1]
                f = 0.0 if span <= 0 else (row[j] - lim) / span
                return float(nodes[j] + f * (nodes[j + 1] - nodes[j])) - 1e-9
        return float(nodes[-1] if row[-1] >= lim else nodes[0])

    def passive_violations(self, dh, a1):
        """[(name, value, lower, upper)] for closure passives past their USD stop; empty is valid.

        A pose exactly AT a stop is empty here and nothing downstream clips it -- `_with_closure`
        writes the LUT value and the drive clamp only engages out of range. Keep out by
        `PASSIVE_KEEPOUT_M` rather than relying on a clip that does not fire."""
        p = self.passives(dh, a1)
        return [(n, p[n], lo, hi) for n, lo, hi in self.passive_lims if not lo <= p[n] <= hi]

    def bounds(self):
        """(lo8, hi8) from the USD limits; callers tighten them (config caps columns at 1.40)."""
        lo = np.array([self.joints[n]["lower"] for n in self.Q8], float)
        hi = np.array([self.joints[n]["upper"] for n in self.Q8], float)
        return lo, hi

    def fk(self, q8, passives=None):
        """link -> (position3, rotation3x3), BASE frame. `passives` overrides the LUT lookup."""
        q = dict(zip(self.Q8, [float(v) for v in q8]))
        dh = q["ColumnRightBearingJoint_1"] - q["ColumnLeftBearingJoint_1"]
        q.update(self.passives(dh, q["ArmLeftJoint_1"]) if passives is None else passives)
        poses = {self.m["root"]: (np.zeros(3), np.eye(3))}
        stack = [self.m["root"]]
        while stack:
            parent = stack.pop()
            pp, pR = poses[parent]
            for j in self.children.get(parent, []):
                # Joint frame authored twice, coinciding at q=0:
                # T = T_par * (p0,R0) * M(q) * inv(p1,R1)
                Jp = pp + pR @ j["_p0"]
                JR = pR @ j["_R0"]
                v = q.get(j["name"], 0.0)
                if j["type"] == "revolute":
                    JR = JR @ _axis_R(j["axis"], v)
                elif j["type"] == "prismatic":
                    Jp = Jp + JR @ (_AXIS_V[j["axis"]] * v)
                cR = JR @ j["_R1"].T
                cp = Jp - cR @ j["_p1"]
                poses[j["child"]] = (cp, cR)
                stack.append(j["child"])
        return poses

    def ik(self, pos, R=None, seed=None, link=None, iters=200, tol_p=1e-4, tol_r=1e-3,
           damping=0.05, step_clip=0.10, bounds=None, restarts=4, rng=None):
        """DLS q8 putting `link` (default the hand) at `pos`, optionally at orientation `R`.
        Returns `(q8, err_p, err_r)`, or None -- never a near miss."""
        link = link or self.hand_link
        pos = np.asarray(pos, float)
        lo, hi = self.bounds() if bounds is None else (np.asarray(bounds[0], float),
                                                       np.asarray(bounds[1], float))
        rows = 3 if R is None else 6
        eps = 1e-5
        rng = rng or np.random.default_rng(0)
        starts = [self.Q8_HOME if seed is None else np.asarray(seed, float)]
        starts += [rng.uniform(lo, hi) for _ in range(max(0, restarts))]
        for q0 in starts:
            r = self._ik_once(np.clip(np.asarray(q0, float).copy(), lo, hi), pos, R, link,
                              lo, hi, rows, eps, iters, tol_p, tol_r, damping, step_clip)
            if r is not None:
                return r
        return None

    def _ik_once(self, q, pos, R, link, lo, hi, rows, eps, iters, tol_p, tol_r, damping, step_clip):
        for _ in range(iters):
            e = self._pose_err(q, link, pos, R)
            ep = float(np.linalg.norm(e[:3]))
            er = 0.0 if R is None else float(np.linalg.norm(e[3:]))
            if ep < tol_p and er < tol_r:
                dh, a1 = q[2] - q[1], q[3]
                # Same rule as the planner's `isValid`: the pose must be one the closure can hold.
                if not self.in_lut_domain(dh, a1) or self.passive_violations(dh, a1):
                    return None
                return q, ep, er
            # a1's ceiling moves with dh, so a constant `hi[3]` lets the iterate walk past the stop. Bind
            # the probe direction too, or the derivative crosses the clamp and stalls.
            hi_eff = hi.copy()
            hi_eff[3] = max(lo[3], min(hi[3], self.a1_ceil(q[2] - q[1]) - PASSIVE_KEEPOUT_M))
            J = np.zeros((rows, 8))
            for k in range(8):
                qk = q.copy()
                d = eps if qk[k] + eps <= hi_eff[k] else -eps
                qk[k] += d
                J[:, k] = (self._pose_err(qk, link, pos, R) - e) / d
            # J^T (J J^T + l^2 I)^-1 e -- damped, so it survives the turret/wrist singularities
            JT = J.T
            dq = -JT @ np.linalg.solve(J @ JT + (damping ** 2) * np.eye(rows), e)
            n = float(np.max(np.abs(dq)))
            if n > step_clip:
                dq *= step_clip / n
            q = np.clip(q + dq, lo, hi_eff)
        return None

    def _pose_err(self, q8, link, pos, R):
        """[position error; rotation error as an axis-angle vector] of `link` against the target."""
        p, Rl = self.fk(q8)[link]
        if R is None:
            return p - pos
        return np.concatenate([p - pos, _rot_vec(Rl @ np.asarray(R, float).T)])

    def limit_margin(self, q8, bounds=None):
        """(prismatic METRES, revolute RADIANS) to the nearest stop, PASSIVES included. Pass the
        caller's own `bounds` -- the raw USD limits over-report the margin by tens of mm."""
        lo, hi = self.bounds() if bounds is None else (np.asarray(bounds[0], float),
                                                       np.asarray(bounds[1], float))
        q = np.asarray(q8, float)
        d = np.minimum(q - lo, hi - q)
        lin, rot = float(d[self.Q8_LIN].min()), float(d[self.Q8_ROT].min())
        p = self.passives(q[2] - q[1], q[3])
        for n, plo, phi in self.passive_lims:
            m = min(p[n] - plo, phi - p[n])
            if self.joints[n]["type"] == "prismatic":
                lin = min(lin, m)
            else:
                rot = min(rot, m)
        return lin, rot

    def clearance(self, q8, obstacles, held=None, base=None, top=0.30, iters=12, fingers=None):
        """Metres of margin against `obstacles`, bisected on `collides` itself so the two agree;
        -1.0 if already overlapping, capped at `top`. An AABB gap is not a substitute."""
        if self.collides(q8, obstacles, 0.0, held, base, fingers=fingers):
            return -1.0
        lo, hi = 0.0, float(top)
        for _ in range(int(iters)):
            mid = 0.5 * (lo + hi)
            lo, hi = ((lo, mid) if self.collides(q8, obstacles, mid, held, base, fingers=fingers)
                      else (mid, hi))
        return lo

    def ik_candidates(self, pos, R, obstacles=(), q_now=None, seeds=8, margin=0.02,
                      held=None, base=None, rng=None, **ik_kw):
        """Every collision-free IK branch found for this hand pose, best first.

        `(q8, err_p, err_r), clearance, score` per entry, scored exactly as `ik_repair` scores:
        `min(clr-margin, margin)/margin - max(d_lin/Q8_TOL_M, d_rot/Q8_TOL_RAD)/K`. Track F measured
        that about half the solution set for a shipped pose is already clear, so a caller that can
        USE more than one -- the OMPL planner, which decides by reachability (Track F2) -- should be
        handed the set instead of the argmax. `ik_repair` is the single-answer consumer of this."""
        q0 = self.Q8_HOME if q_now is None else np.asarray(q_now, float)
        rng = rng or np.random.default_rng(0)
        b = ik_kw.get("bounds")
        lo, hi = self.bounds() if b is None else (np.asarray(b[0], float), np.asarray(b[1], float))
        band = max(float(margin), 1e-9)         # margin=0 is a legal ask; never divide by it
        out = []
        for _ in range(max(1, int(seeds))):
            seed = np.clip(q0 + rng.normal(0.0, IK_SEED_SPREAD, 8) * (hi - lo), lo, hi)
            cand = self.ik(pos, R, seed=seed, restarts=0, rng=rng, **ik_kw)
            if cand is None or self.collides(cand[0], obstacles, margin, held, base):
                continue
            clr = self.clearance(cand[0], obstacles, held, base)
            d_lin, d_rot = self.q8_resid(cand[0], q0)
            score = (min(clr - band, band) / band
                     - max(d_lin / Q8_TOL_M, d_rot / Q8_TOL_RAD) / IK_REPAIR_TRAVEL_K)
            out.append((cand, clr, score))
        out.sort(key=lambda e: e[2], reverse=True)
        return out

    def grasp_candidates(self, pos, R, obj, obstacles=(), q_now=None, yaws=GRASP_YAWS,
                         axis=(0.0, 0.0, 1.0), rng=None, **ik_kw):
        """Grasp-candidate set, best score first: `(q8, err_p, err_r), clearance, score, yaw` per
        entry, each clear of `obstacles` at `margin`, the hand orbited rigidly about `axis` through
        `obj` (all in `ik`'s frame). The score has no yaw term: treat the order as a hint."""
        pos, R, obj = np.asarray(pos, float), np.asarray(R, float), np.asarray(obj, float)
        rng = rng or np.random.default_rng(0)
        out = []
        for y in yaws:
            Ry = _axis_angle_R(axis, float(y))
            # `pos + (Ry-I)@d`, not `obj + Ry@d`: identical algebra, but exactly `pos` at yaw 0,
            # so the nominal slice is bit-identical to the unrotated solve.
            out += [(c, clr, s, float(y)) for c, clr, s in self.ik_candidates(
                pos + (Ry - np.eye(3)) @ (pos - obj), Ry @ R, obstacles=obstacles,
                q_now=q_now, rng=rng, **ik_kw)]
        out.sort(key=lambda e: e[2], reverse=True)
        return out

    def ik_repair(self, pos, R, obstacles=(), q_now=None, seeds=8, margin=0.02,
                  held=None, base=None, rng=None, log=False, **ik_kw):
        """`ik`, then a SEARCH ONLY IF that solution collides; same `(q8, err_p, err_r)` return.
        Candidates are `seeds` descents seeded from `q_now`, filtered by `collides` at `margin`,
        scored `min(clr-margin, margin)/margin - max(d_lin/Q8_TOL_M, d_rot/Q8_TOL_RAD)/K`. `R` is
        required positionally: with no orientation term the score wins with the hand rotated tens
        of degrees. Returns None rather than a colliding branch, so wire it only where None means
        "fall back to the baked trajectory". `log` off by default -- the OMPL subprocess imports
        this module and its stdout carries the plan as JSON."""
        sol = self.ik(pos, R, seed=q_now, **ik_kw)
        if sol is None:
            if log:
                print(">>> ik-repair: no IK solution at all -> caller falls back", flush=True)
            return None
        offender = self.first_hit(sol[0], obstacles, margin, held, base)
        if offender is None:
            # No `clearance` bisection here: 12 `collides` calls is ~94 ms on the real box set.
            if log:
                print(f">>> ik-repair: goal already clear at the {margin * 1000:.0f} mm margin, "
                      f"unchanged ({len(obstacles)} obstacles)", flush=True)
            return sol
        clr0 = self.clearance(sol[0], obstacles, held, base)
        cands = self.ik_candidates(pos, R, obstacles=obstacles, q_now=q_now, seeds=seeds,
                                   margin=margin, held=held, base=base, rng=rng, **ik_kw)
        best = cands[0][0] if cands else None
        best_clr = cands[0][1] if cands else -1.0
        q0 = self.Q8_HOME if q_now is None else np.asarray(q_now, float)
        if log:
            if best is None:
                print(f">>> ik-repair: NO CLEAR SOLUTION -- {offender} at {_mm(clr0)} "
                      f"(margin {margin * 1000:.0f} mm, {len(obstacles)} obstacles), "
                      f"{seeds} seeds tried -> caller falls back", flush=True)
            else:
                dl, dr = self.q8_resid(best[0], q0)
                print(f">>> ik-repair: REPAIRED -- {offender} {_mm(clr0)} -> {_mm(best_clr)} "
                      f"(margin {margin * 1000:.0f} mm, {len(obstacles)} obstacles); travel "
                      f"{dl * 1000:.0f} mm / {dr:.3f} rad", flush=True)
        return best

    @staticmethod
    def _box_corners(aabb):
        """The 8 corners of a local AABB, one per row. The only copy in `morph/`: the collision
        test and the check-gap measurement have to mean the same box by "corner". (`tools/` walks
        USD bounds, a different geometry source, and `tests/` keeps a deliberate oracle copy.)"""
        mn, mx = aabb
        return np.array([[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1])
                         for z in (mn[2], mx[2])], float)

    @staticmethod
    def _xform_aabb(aabb, p, R):
        """Local AABB -> axis-aligned AABB of its 8 rotated corners; never smaller than the box."""
        w = ArmModel._box_corners(aabb) @ R.T + p
        return np.vstack([w.min(axis=0), w.max(axis=0)])

    def _links(self, q8, held=None, base=None, fingers=None):
        """(link, local_aabb, p, R) per link, base frame -- or world if `base` is (t3, R3x3).

        `fingers` is {joint: value} for the COMMANDED hand. Given, the hand sheds the swept union
        and is checked as the palm plus the nine finger pads at that pose; absent, nothing changes.
        """
        if fingers is None:
            poses = self.fk(q8)
            items = list(self.link_aabb.items())
        else:
            # MERGE, never replace: `passives=` overrides the LUT lookup wholesale, so passing
            # finger values through it alone zeroes all four closure passives and moves the boom.
            q = dict(zip(self.Q8, [float(v) for v in q8]))
            merged = self.passives(q["ColumnRightBearingJoint_1"] - q["ColumnLeftBearingJoint_1"],
                                   q["ArmLeftJoint_1"])
            merged.update({k: float(v) for k, v in fingers.items()})
            poses = self.fk(q8, passives=merged)
            items = [(k, PALM_AABB if k == self.hand_link else v)
                     for k, v in self.link_aabb.items()]
            items += [(k, b) for k, b in _FINGER_BOXES.items() if k in poses]
        if held is not None:
            items.append((HELD_LINK, np.asarray(held, float)))
        for link, local in items:
            p, R = poses[self.hand_link if link == HELD_LINK else link]
            if base is not None:
                bt, bR = np.asarray(base[0], float), np.asarray(base[1], float)
                p, R = bt + bR @ p, bR @ R
            yield link, local, p, R

    # Sub-meshes that are convex hulls of a PLATE, not solid volume: the boom passes through them in
    # normal operation, so keeping them would veto poses the robot already performs.
    SELFCOL_HULL_PARTS = frozenset(("VerticalArm_0", "VerticalArm_1"))

    # Pairs that overlap at essentially EVERY shipped pose, MEASURED here rather than inferred: hop
    # distance on the joint graph was the old proxy and the two diverge badly.
    SELFCOL_IGNORE = frozenset(frozenset(p) for p in (
        # 1228/1228 -- permanent by construction.
        ("Arm_1::VerticalArm_3", "Arm_Left_1"),        # the boom rides the left rail
        ("Arm_Left_1", "Bearing_Column_Left_1"),
        ("Arm_Right_1", "Bearing_Column_Right_1"),
        ("Bearing_Column_Right_1", "Contact_Cylinder_1_1"),
        ("Bearing_Column_Right_1", "Contact_Cylinder_1_2"),
        ("Contact_Cylinder_1_1", "Contact_Cylinder_1_2"),   # ONE spherical joint, two branches
        ("Gripper_Link1_1", "Gripper_Link3_1"),
        ("Gripper_Link2_1", "Hand_Bearing_1"),
        # 1205/1228 (98.1%) -- two links apart in the wrist, overlapping at all but 23 poses.
        ("Gripper_Link3_1", "Hand_Bearing_1"),
    ))

    def _selfcol_pair_list(self):
        """Non-adjacent link pairs worth testing, built once from the joint graph."""
        if self._selfcol_pairs is None:
            near = {}
            for j in self.joints.values():
                near.setdefault(j["parent"], set()).add(j["child"])
                near.setdefault(j["child"], set()).add(j["parent"])

            def adjacent(a, b):
                """Directly jointed -- the only exclusion that is structural rather than measured.
                Pseudo-links inherit the adjacency of the frame they were carved from."""
                fa, fb = self.selfcol_frame[a], self.selfcol_frame[b]
                return fa == fb or fb in near.get(fa, ()) or fa in near.get(fb, ())

            names = sorted(self.selfcol_aabb)        # sorted: which pair is REPORTED must not
            # depend on the model file's key order
            self._selfcol_pairs = tuple(
                (a, b) for i, a in enumerate(names) for b in names[i + 1:]
                if not adjacent(a, b) and frozenset((a, b)) not in self.SELFCOL_IGNORE)
        return self._selfcol_pairs

    def self_collides(self, q8, margin=0.0):
        """("link_a", "link_b") of the first arm-vs-arm overlap at `q8`, or None if clear. A boolean
        test on the tight `selfcol_aabb` boxes with an adjacency mask, not a penetration budget:
        PhysX self-collision is off, and the planner consults this only under SELF_COLLIDE_PLAN=1."""
        poses = self.fk(q8)
        names, ia, ib = self._selfcol_index()
        # VECTORISED AABB broad phase: this runs in OMPL's inner loop, so the pair scan cannot be a Python
        # loop. Exact, not a heuristic -- non-overlapping AABBs prove the OBBs inside them cannot overlap.
        n = len(names)
        mid, half = self._selfcol_extents()
        lo = np.empty((n, 3))
        hi = np.empty((n, 3))
        for k, name in enumerate(names):
            p_, R_ = poses[self.selfcol_frame[name]]
            c = p_ + R_ @ mid[k]
            # World half-extent of an oriented box is |R| @ half -- the same bound `_xform_aabb` reaches by
            # transforming all 8 corners, at a fraction of the cost. This is the hot path.
            e = np.abs(R_) @ half[k]
            lo[k], hi[k] = c - e, c + e
        cand = np.flatnonzero(~_aabb_prunes(lo[ia], hi[ia], lo[ib], hi[ib], margin))
        for k in cand:
            a, b = names[ia[k]], names[ib[k]]
            if self._pair_hits(a, b, poses, margin):
                return (a, b)
        return None

    def _pair_hits(self, a, b, poses, margin):
        """Narrow phase for one self-collision pair. The two wrist links that are not boxes get
        their own predicate; every other pair is the oriented box test it always was."""
        if a not in SELFCOL_PRIMITIVE and b in SELFCOL_PRIMITIVE:
            a, b = b, a
        pa, Ra = poses[self.selfcol_frame[a]]
        pb, Rb = poses[self.selfcol_frame[b]]
        return SELFCOL_PRIMITIVE.get(a, _obb_hits_obb)(
            self.selfcol_aabb[a], pa, Ra, self.selfcol_aabb[b], pb, Rb, margin)

    def _selfcol_extents(self):
        """(mid, half) arrays for the self-collision links, in `_selfcol_index` order."""
        if getattr(self, "_selfcol_ext", None) is None:
            names = self._selfcol_index()[0]
            box = np.array([self.selfcol_aabb[n] for n in names], float)
            self._selfcol_ext = ((box[:, 0] + box[:, 1]) * 0.5, (box[:, 1] - box[:, 0]) * 0.5)
        return self._selfcol_ext

    def _selfcol_index(self):
        """(names, ia, ib) -- the pair list as index arrays, so the broad phase is pure numpy."""
        if getattr(self, "_selfcol_idx", None) is None:
            pairs = self._selfcol_pair_list()
            names = sorted({n for pr in pairs for n in pr})
            at = {n: i for i, n in enumerate(names)}
            self._selfcol_idx = (names,
                                 np.array([at[a] for a, _ in pairs], int),
                                 np.array([at[b] for _, b in pairs], int))
        return self._selfcol_idx

    # The three fingertip links whose origins `morph/gripper/fingers.py`'s `_pinch()` averages. Named here
    # so the prediction and the measurement cannot drift apart silently.
    PINCH_TIPS = ("finger_a_link_3_1", "finger_b_link_3_1", "finger_c_link_3_1")

    def pinch(self, q8, fingers=None, base=None):
        """The PREDICTED pinch point -- `0.5 * (a + 0.5 * (b + c))` over the three fingertip link
        origins from this FK; base frame, or world given `base=(t3, R3x3)`. Replaces `_fk_z` in the
        descent solve only (`morph/close/` keeps `_fk_z`); `fingers` moves the tips, pass it."""
        poses = self.fk(q8) if fingers is None else dict(
            (k, (p, R)) for k, _b, p, R in self._links(q8, fingers=fingers))
        if fingers is None:
            poses = {k: v for k, v in poses.items()}
        t = [np.asarray(poses[n][0], float) for n in self.PINCH_TIPS]
        p = 0.5 * (t[0] + 0.5 * (t[1] + t[2]))
        if base is None:
            return p
        bt, bR = np.asarray(base[0], float), np.asarray(base[1], float)
        return bt + bR @ p

    def link_aabbs(self, q8, held=None, base=None, fingers=None):
        """link -> (2,3) [min,max]. `held` is an AABB in the HAND-LINK frame, keyed `__held__`.

        `fingers` for the same reason `first_hit` takes it: without it the hand is the swept union,
        so a report built here describes a DIFFERENT hand from the one that refused -- which is how
        a rejected goal came with three geometrically impossible blame lines."""
        return {link: self._xform_aabb(local, p, R)
                for link, local, p, R in self._links(q8, held, base, fingers=fingers)}

    def corner_shifts(self, q_a, q_b, held=None, fingers=None):
        """{link: metres its furthest collision-box corner travels between `q_a` and `q_b`}.

        The boxes are the ones `first_hit` tests, so this is how far the CHECKED geometry moves
        over an interval nothing checks -- which a joint delta is not a proxy for: the column
        closure turns the same 0.02 into 26 mm of corner at one pose and 353 mm at another."""
        out = {}
        for (link, local, pa, Ra), (_, _, pb, Rb) in zip(self._links(q_a, held, fingers=fingers),
                                                         self._links(q_b, held, fingers=fingers)):
            c = self._box_corners(local)
            out[link] = float(np.linalg.norm((pb + c @ Rb.T) - (pa + c @ Ra.T), axis=1).max())
        return out

    def tunnel_limits(self, held=None, margin=0.02, fingers=None, pads=()):
        """{link: metres its box may travel between two checks without an obstacle passing through
        unseen}: the box's own thinnest dimension plus twice the margin it is grown by -- nothing for the
        obstacle's thickness, a finite board tunnels at its EDGE. Keys match `corner_shifts`."""
        # The SMALLEST margin ANY pair is checked at, because it bounds how far a box may travel between
        # checks: a pair held to a floor tolerates a shorter step than one at the global margin.
        margin = min([float(margin)] + [float(p) for p in pads])
        boxes = dict(self.link_aabb)
        if fingers is not None:
            boxes[self.hand_link] = PALM_AABB
            boxes.update(_FINGER_BOXES)
        if held is not None:
            boxes[HELD_LINK] = np.asarray(held, float)
        return {k: float((b[1] - b[0]).min()) + 2 * float(margin)
                for k, b in boxes.items()}

    def collides(self, q8, obstacles, margin=0.02, held=None, base=None, fingers=None):
        """`first_hit` as a bool; kept separate because this is OMPL's hot path."""
        return self.first_hit(q8, obstacles, margin, held, base, fingers=fingers) is not None

    def first_hit(self, q8, obstacles, margin=0.02, held=None, base=None, links=None,
                  fingers=None):
        """First link inside `margin` of an obstacle, or None. `obstacles`: `(aabb, tag)`, the tag
        naming the frame -- "world", "chassis" (base), "grasped" (world; the `hand_subtree` links
        and the carried box are exempt) -- or `(aabb, tag, pad_m, links)`, a FLOOR: those links are
        checked at min(pad, margin)."""
        # 4-tuples cannot ride the partition below, so they are split out and tested per link;
        # nobody pays for the split unless one is present -- `collides` is OMPL's hot path.
        soft = [ob for ob in obstacles if len(ob) > 2] or None
        if soft is not None:
            obstacles = [ob for ob in obstacles if len(ob) == 2]
        chassis = [ob for ob, tag in obstacles if tag == "chassis"]
        world = [ob for ob, tag in obstacles if tag not in ("chassis", "grasped")]
        grasped = [ob for ob, tag in obstacles if tag == "grasped"]
        every = world + grasped if grasped else world
        bt, bR = (None, None) if base is None else (np.asarray(base[0], float),
                                                    np.asarray(base[1], float))
        want = None if links is None else set(links)
        for link, local, p, R in self._links(q8, held, fingers=fingers):   # BASE frame, from fk
            if want is not None and link not in want:
                continue
            if chassis and link not in self.column_group:
                if self._hits(local, p, R, chassis, margin):
                    return link
            out = world if link in self.payload_side else every
            if out:
                pw, Rw = (p, R) if bt is None else (bt + bR @ p, bR @ R)
                if self._hits(local, pw, Rw, out, margin):
                    return link
            if soft is not None:
                for ob, tag, pad, links in soft:
                    # A FLOOR, not a deletion: the pair is still tested, at `pad` instead of `margin`. `min`
                    # because callers lower `margin` deliberately, and a fixed pad would hit at margin=0.
                    mg = min(pad, margin) if link in links else margin
                    if tag == "chassis":
                        if link in self.column_group:
                            continue
                        pp, RR = p, R
                    else:
                        if tag == "grasped" and link in self.payload_side:
                            # the same allowance the unannotated path gives
                            continue
                        pp, RR = (p, R) if bt is None else (bt + bR @ p, bR @ R)
                    if self._hits(local, pp, RR, [ob], mg):
                        return link
        return None

    def _hits(self, local, p, R, boxes, margin):
        """Does link box `local` at (p, R), grown by `margin`, overlap any of `boxes` (bare (2,3)
        [min,max], SAME frame)? AABB broad phase, then an ORIENTED test -- an AABB alone is nearly
        twice the boom's footprint at some turret angles."""
        box = self._xform_aabb(local, p, R)
        lo, hi = box[0] - margin, box[1] + margin
        c = p + R @ ((local[0] + local[1]) * 0.5)
        h = (local[1] - local[0]) * 0.5 + margin               # half-extents, grown by the margin
        for ob in boxes:
            if not (np.all(lo <= ob[1]) and np.all(hi >= ob[0])):
                continue
            if _obb_hits_aabb(c, h, R, ob):
                return True
        return False


def _rot_vec(Rerr, eps=1e-9):
    """Rotation matrix -> axis-angle vector; the residual an orientation servo drives to zero."""
    c = (np.trace(Rerr) - 1.0) * 0.5
    ang = math.acos(max(-1.0, min(1.0, c)))
    if ang < 1e-8:
        return np.zeros(3)
    axis = np.array([Rerr[2, 1] - Rerr[1, 2], Rerr[0, 2] - Rerr[2, 0], Rerr[1, 0] - Rerr[0, 1]])
    return axis * (ang / max(2.0 * math.sin(ang), eps))
