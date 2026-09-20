"""Frames, forward kinematics, the closure LUT and the baked-trajectory loaders.
Imports Isaac APIs at module level: importable only after SimulationApp exists."""
import json
import math
import os

import numpy as np
from isaacsim.core.prims import SingleXFormPrim

from morph.config import (HERE, KNOWN, COLUMN_MAX, A1_MAX,
                          CHASSIS_PLATE_X, CHASSIS_PLATE_Y)
from morph.usd_utils import find_path


class KinematicsMixin:
    """Mixed into Demo. Every `self.*` it touches is owned by Demo."""

    def grip_pos(self):
        return np.asarray(self._grip_prim.get_world_pose()[0], float)

    def _grip_frame(self):
        """World (pos, R) of the gripper body — R rotates a gripper-frame offset into world."""
        p, q = self._grip_prim.get_world_pose()
        p = np.asarray(p, float)
        w, x, y, z = [float(v) for v in q]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
                      [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]])
        return p, R

    def _hand_prims(self):
        """name -> SingleXFormPrim for every arm-1 hand/finger link (contact forensics)."""
        if getattr(self, "_hand_prims_cache", None) is None:
            out = {}
            for prim in self.stage.Traverse():
                nm = prim.GetName()
                pth = prim.GetPath().pathString
                if "Arm_1" not in pth:
                    continue
                if (nm.startswith("finger_") and "_link_" in nm and nm.endswith("_1")) \
                        or nm.startswith("Gripper_Link") and nm.endswith("_1"):
                    out[nm] = SingleXFormPrim(pth)
            self._hand_prims_cache = out
        return self._hand_prims_cache

    def _hand_frame(self):
        """World (pos, R) of Gripper_Link3_1, the FINGERS' parent -- NOT `_grip_frame`
        (Gripper_Link1_1): a held object pins to Link3, the baked FK and LUT stay in Link1."""
        if getattr(self, "_hand_prim", None) is None:
            self._hand_prim = SingleXFormPrim(find_path(self.stage, "Gripper_Link3_1"))
        p, q = self._hand_prim.get_world_pose()
        w, x, y, z = [float(v) for v in q]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
                      [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]])
        return np.asarray(p, float), R

    def _obj_R(self, obj):
        """World rotation of the object (None if unreadable); gap geometry needs its real frame."""
        try:
            return self._quat_to_R(np.asarray(obj.get_world_poses()[1][0], float))
        except Exception:
            return None

    @staticmethod
    def _lut_cell(vals, v):
        v = min(max(v, vals[0]), vals[-1])
        for i in range(len(vals) - 2, -1, -1):
            if v >= vals[i]:
                return i, (v - vals[i]) / (vals[i + 1] - vals[i])
        return 0, 0.0

    def _lut_interp(self, table, dh, a1, n):
        i, fi = self._lut_cell(self.clut["dh"], dh)
        j, fj = self._lut_cell(self.clut["a1"], a1)
        return [(1 - fi) * (1 - fj) * table[i][j][k] + fi * (1 - fj) * table[i + 1][j][k]
                + (1 - fi) * fj * table[i][j + 1][k] + fi * fj * table[i + 1][j + 1][k]
                for k in range(n)]

    def _closure_passives(self, dh, a1):
        """Closure LUT -> the 4 passive linkage joints for (h2-h1, a1)."""
        return dict(zip(self.clut["passives"], self._lut_interp(self.clut["grid"], dh, a1, 4)))

    def _grip_fk(self, h1, h2, a1, th):
        """World position of the gripper (Gripper_Link1_1) from the baked map: chassis-frame grip at
        (dh, a1), z-shifted 1:1 by (h1 - h1_ref), rotated about the mount by th, then base-placed."""
        p = np.array(self._lut_interp(self.clut["grip"], h2 - h1, a1, 3))
        p[2] += h1 - self.clut["h1_ref"]
        mx, my = self.clut["mount"][0], self.clut["mount"][1]
        c, s = math.cos(th), math.sin(th)
        px, py = p[0] - mx, p[1] - my
        p[0], p[1] = mx + c * px - s * py, my + s * px + c * py
        # base pose: the jog's own commanded pose while one is active, else the live ledger. The
        # ANCHOR alone was right only while the base could not move during a jog.
        if getattr(self, "jog", None):
            bx, by, byaw = self.jog["anchor"]
            _d = self.jog.get("base_delta", (0.0, 0.0, 0.0))
            bx, by, byaw = bx + _d[0], by + _d[1], byaw + _d[2]
        else:
            bx, by, byaw = self.base_ledger()
        c, s = math.cos(byaw), math.sin(byaw)
        return np.array([bx + c * p[0] - s * p[1], by + s * p[0] + c * p[1], p[2]])

    def _lut_z(self, dh, a1):
        """Raw LUT pinch z at (dh, a1) with h1 at the bake reference; `_fk_z` adds the live h1."""
        return float(self._lut_interp(self.clut["grip"], dh, a1, 3)[2])

    def _bisect_a1(self, pred, lo, hi, iters=40):
        """Smallest a1 in (lo, hi] where monotone `pred` holds; the caller must have checked pred(hi)."""
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if pred(mid):
                hi = mid
            else:
                lo = mid
        return float(hi)

    def _fk_z(self, h1, h2, a1):
        """Pinch HEIGHT from the baked closure map -- no physics, no settle, no transient. Read tilt
        gradients through this, never by moving the real joint; z is chassis-pose independent."""
        return self._lut_z(h2 - h1, a1) + h1 - self.clut["h1_ref"]

    # [min, max] in the BASE LINK frame, and NOT the battery despite the name: `tools/measure_body.py`
    # unions EVERY non-arm prim above the chassis plate. Never edit the six faces apart.
    BODY_BOX = np.array([[-0.3165, -0.2960, 0.1560],
                         [-0.0219, +0.2809, +0.3310]], float)

    def _body_hit(self, th, h1, dh, a1, margin=0.02):
        """Name of the link overlapping `BODY_BOX` at this column/boom pose, or None if clear. Wrist
        zeroed, no held box: only `Arm_Left_1` can reach the box over the place envelope."""
        q = np.zeros(8)                      # ArmModel.Q8 order: th, h1, h2, a1, hb, wz, wy, wx
        q[0], q[1], q[2], q[3] = float(th), float(h1), float(h1) + float(dh), float(a1)
        return self._arm_model().first_hit(q, [(self.BODY_BOX, "chassis")], margin)

    def _body_clear(self, tag):
        """Print, under `tag`, the arm's clearance from the chassis body at the CURRENT pose. Always
        prints a number, clear path included, so "clear" and "never ran" differ. Never blocks."""
        try:
            m, q = self._arm_model(), self._arm_q8()
            obs = [(self.BODY_BOX, "chassis")]
            c = m.clearance(q, obs)
            hit = m.first_hit(q, obs, 0.02)
            if not np.all(np.isfinite(q)):
                # A NaN pose walks through the OBB test as "no overlap" and would print CLEAR.
                verdict = "UNKNOWN -- the articulation pose is NaN"
            elif c < 0.0:
                verdict = f"OVERLAPPING it -- {m.first_hit(q, obs, 0.0)} is INSIDE the body"
            elif hit is not None:
                verdict = f"{c * 1000:.0f}mm -- INSIDE the 20mm margin, {hit}"
            else:
                verdict = f"{c * 1000:.0f}mm clear"
            print(f">>> body clear[{tag}]: h1 {q[1]:.3f} dh {q[2] - q[1]:+.4f} a1 {q[3]:.3f} "
                  f"th {q[0]:+.3f} -> boom vs chassis body {verdict}", flush=True)
        except Exception as _e:
            print(f">>> body clear[{tag}]: FAILED ({_e})", flush=True)

    def _clears_chassis(self, dh, a1, margin=0.02, th=0.0):
        """Is the gripper clear of the chassis collision plate at this (dh, a1, th)? Geometric --
        self-collision is off, so physics will not stop it. True (permissive) on a failed lookup."""
        try:
            # The LUT is baked at th=0: rotate xy about the mount by th.
            p = self._lut_interp(self.clut["grip"], dh, a1, 3)
            mx, my = self.clut["mount"][0], self.clut["mount"][1]
            c, s_ = math.cos(th), math.sin(th)
            px, py = float(p[0]) - mx, float(p[1]) - my
            x = mx + c * px - s_ * py
            y = my + s_ * px + c * py
            return (abs(x) >= CHASSIS_PLATE_X + margin
                    or abs(y) >= CHASSIS_PLATE_Y + margin)
        except Exception:
            # no table -> do not block motion
            return True

    def _a1_clearing_chassis(self, dh, margin=0.02, a1_max=A1_MAX, th=0.0):
        """Smallest a1 keeping the gripper clear of the plate at this tilt, None if none does.
        The safe bound is a (dh, a1) pair, not a tilt constant: deeper tilts need MORE a1."""
        lo, hi = 0.0, float(a1_max)
        if self._clears_chassis(dh, lo, margin, th=th):
            return 0.0
        if not self._clears_chassis(dh, hi, margin, th=th):
            return None
        return self._bisect_a1(lambda a: self._clears_chassis(dh, a, margin, th=th), lo, hi)

    def _pinch_z_ref(self, dh, a1, q8_t, fingers, base):
        """World z of the PREDICTED pinch at the LUT's reference height for this (dh, a1); the pinch
        translates 1:1 with the columns, so `pinch_z(h1) = this + (h1 - h1_ref)`. `q8_t` is the
        turret and wrist to fly: the Link1 -> pinch lever swings -30/+18mm with the wrist."""
        q = np.array(q8_t, float).copy()
        q[1] = float(self.clut["h1_ref"])
        q[2] = float(self.clut["h1_ref"]) + float(dh)
        q[3] = float(a1)
        return float(self._arm_model().pinch(q, fingers=fingers, base=base)[2])

    def _solve_lowest(self, z_tgt, a1_max=A1_MAX, h_max=COLUMN_MAX, dh_want=None,
                      h1_min=0.0, th=0.0, q8_t=None, fingers=None, base=None):
        """Solve (h1, h2) putting the PREDICTED pinch at `z_tgt` with the most forward reach; None
        if unreachable. The returned `a1` is a solver proxy and is deliberately NOT filtered against
        the mirror boom's stop: reach.py flies min(0.24 or the chassis floor, the goal's a1)."""
        # A requested tilt wins over maximum reach: tilt hard first, trade it back for reach after.
        if dh_want is not None:
            a1 = a1_max
            p = self._lut_interp(self.clut["grip"], dh_want, a1, 3)
            h1 = (z_tgt - self._pinch_z_ref(dh_want, a1, q8_t, fingers, base)
                  + self.clut["h1_ref"])
            if (h1_min <= h1 <= h_max and 0.0 <= h1 + dh_want <= h_max
                    and self._clears_chassis(dh_want, a1)
                    and self._body_hit(th, h1, dh_want, a1) is None):
                return (h1, h1 + dh_want, a1, float(p[0]))
            # requested tilt is past a column stop at this height -> fall through to max reach
        best = None
        dhs = self.clut["dh"]
        for i in range(81):
            dh = dhs[0] + (dhs[-1] - dhs[0]) * i / 80.0
            for j in range(51):
                a1 = a1_max * j / 50.0
                # `p` is still the LUT's, but ONLY for `p[0]` -- the forward reach this scan
                # scores on. The HEIGHT comes from the predicted pinch, per candidate.
                p = self._lut_interp(self.clut["grip"], dh, a1, 3)
                # pinch_z = pinch_z(h1_ref) + h1 - h1_ref  ->  h1 landing exactly on the target
                h1 = (z_tgt - self._pinch_z_ref(dh, a1, q8_t, fingers, base)
                      + self.clut["h1_ref"])
                h2 = h1 + dh
                if not (h1_min <= h1 <= h_max and 0.0 <= h2 <= h_max):
                    continue
                if not self._clears_chassis(dh, a1):
                    continue
                # `_clears_chassis` is a grip-POINT test: it cannot see the BOOM. Without this the
                # scan returned h1 0.0122 / dh 0.055 with Arm_Left_1 inside the chassis body box.
                if self._body_hit(th, h1, dh, a1) is not None:
                    continue
                # A refused `dh_want` must degrade to the NEAREST tilt, not to a different pose family:
                # maximum-reach-at-any-tilt answers a question the descent did not ask.
                key = (float(p[0]),) if dh_want is None else (-abs(dh - dh_want), float(p[0]))
                if best is None or key > best[0]:
                    best = (key, h1, h2, a1, float(p[0]))
        return None if best is None else (best[1], best[2], best[3], best[4])

    def _solve_reach_z(self, reach_tgt, z_tgt, a1_max=A1_MAX, h_max=COLUMN_MAX, h1_min=0.0,
                       dh_ref=None, th=0.0, a1_ref=None, h1_ref=None):
        """Solve (h1, h2, a1) putting the grip at `reach_tgt` forward AND `z_tgt` high -- the dual of
        `_solve_lowest`. Reach and height couple violently through the tilt, so (dh, a1) is chosen up
        front, nearest `dh_ref`/`a1_ref`/`h1_ref`. None if outside the arm's envelope."""
        best = None
        dhs = self.clut["dh"]
        for i in range(81):
            dh = dhs[0] + (dhs[-1] - dhs[0]) * i / 80.0
            for j in range(51):
                a1 = a1_max * j / 50.0
                p = self._lut_interp(self.clut["grip"], dh, a1, 3)
                h1 = z_tgt - float(p[2]) + self.clut["h1_ref"]
                h2 = h1 + dh
                if not (h1_min <= h1 <= h_max and 0.0 <= h2 <= h_max):
                    continue
                if not self._clears_chassis(dh, a1):
                    continue
                # Unlike `_solve_lowest`, whose a1 the caller discards, THIS a1 is COMMANDED --
                # `place/insert.py` interpolates it -- so an unreachable return is driven, not printed.
                if self._arm_model().passive_violations(dh, a1):
                    continue
                # The LUT stores the grip BEFORE the turret rotation: rotate by th before comparing.
                _mx, _my = self.clut["mount"][0], self.clut["mount"][1]
                _c, _s = math.cos(th), math.sin(th)
                _px, _py = float(p[0]) - _mx, float(p[1]) - _my
                reach = _mx + _c * _px - _s * _py
                err = abs(reach - reach_tgt)
                # (reach, z) is two constraints on three DOFs, so many triples hit the same point.
                _w = 0.5
                cost = err
                if dh_ref is not None:
                    cost += _w * abs(dh - dh_ref)
                if a1_ref is not None:
                    cost += _w * abs(a1 - a1_ref)
                if h1_ref is not None:
                    cost += _w * abs(h1 - h1_ref)
                if best is None or cost < best[0]:
                    best = (cost, err, h1, h2, a1, reach)
        if best is None:
            return None
        return (best[2], best[3], best[4], best[5], best[1])   # h1, h2, a1, reach, reach_err

    def _load_trajs(self):
        """Load the baked loop-consistent trajectories from `usd/_traj_*.json` -- every frame
        satisfies the linkage closure. None if any file or joint is missing, which the pick and
        the place both treat as a refusal."""
        out = {}
        for name in ("reach", "lift", "place_low", "place_mid", "place_high"):
            p = os.path.join(HERE, f"usd/_traj_{name}.json")
            if not os.path.exists(p):
                print(f">>> no {p} -> no baked trajectories (run gen_traj.py)", flush=True)
                return None
            t = json.load(open(p))
            missing = [j for j in t["joints"] if j not in self.idx]
            if missing:
                print(f">>> traj {name}: unknown joints {missing[:4]} -> unusable", flush=True)
                return None
            # Drop the arm-2 columns: the bake carries a low home pose for arm 2.
            keep = [i for i, j in enumerate(t["joints"]) if not j.endswith("_2")]
            out[name] = {"idx": np.array([self.idx[t["joints"][i]] for i in keep]),
                         "frames": np.array(t["frames"], float)[:, keep], "meta": t["meta"],
                         # kept-column joint NAMES: stages needing one baked joint look it up here
                         "names": [t["joints"][i] for i in keep]}
        print(f">>> loaded {len(out)} baked trajectories (loop-consistent)", flush=True)
        return out

    def _load_close_traj(self):
        """Finger trajectory of a real successful grasp, from `usd/_grasp_close.json`. `None` is a
        refusal: the close stage has no other close, and a curl profile never gripped."""
        p = os.path.join(HERE, "usd/_grasp_close.json")
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f">>> close-replay unavailable ({e}) -> the close will REFUSE", flush=True)
            return None
        ph = d.get("phase", [])
        # The `close` phase is thumb-only; b and c close in `post_close`, so a BOUNDED number of
        # those rows is added -- its tail frames re-OPEN the fingers. Only FINGER joints replayed.
        n_post = 200    # 0 = close-phase only
        rows = [i for i, s in enumerate(ph) if s == "close"]
        if n_post > 0:
            rows += [i for i, s in enumerate(ph) if s == "post_close"][:n_post]
        rows = rows or list(range(len(d["frames"])))
        names = [j for j in d["joints"] if "finger" in j]
        miss = [n for n in names if n not in self.idx]
        if miss:
            print(f">>> close-replay: joints missing in Isaac {miss[:3]} -> the close will REFUSE",
                  flush=True)
            return None
        cols = [d["joints"].index(n) for n in names]
        fr = np.array([[d["frames"][r][c] for c in cols] for r in rows], float)
        ob = d.get("obj_in_base")
        if ob:
            rec = np.array([float(ob[rows[0]][0]), float(ob[rows[0]][1])])
            # Clamp range, keep the recorded LATERAL offset: shorten x alone, never rescale.
            _dmax = 0.64
            if _dmax > 0 and float(np.linalg.norm(rec)) > _dmax and abs(rec[1]) < _dmax:
                _x_new = float(math.sqrt(_dmax * _dmax - float(rec[1]) * float(rec[1])))
                print(f">>> dock target: range {float(np.linalg.norm(rec)):.3f} -> {_dmax:.3f}m "
                      f"(the grasp distance); x {float(rec[0]):.4f} -> {_x_new:.4f}, "
                      f"lateral {float(rec[1]):+.4f} UNCHANGED", flush=True)
                rec = np.array([_x_new * (1.0 if rec[0] >= 0 else -1.0), float(rec[1])])
            self.pick_standoff = float(np.linalg.norm(rec))
            print(f">>> dock target: recorded grasp {np.round(rec, 4).tolist()} "
                  f"(was {np.round(self.local_obj, 4).tolist()}, "
                  f"{np.linalg.norm(rec - self.local_obj) * 1000:.0f}mm apart)", flush=True)
            self.local_obj = rec
        print(f">>> close-replay: {len(fr)} recorded grasp frames, {len(names)} finger joints "
              f"(grip pose j1 a/b/c = {fr[-1][names.index('finger_a_joint_1_1')]:+.2f}/"
              f"{fr[-1][names.index('finger_b_joint_1_1')]:+.2f}/"
              f"{fr[-1][names.index('finger_c_joint_1_1')]:+.2f})", flush=True)
        # FRAME PAIRING: `obj_in_weldbody` is in Gripper_Link1_1 (`_grip_frame`), `obj_in_gripper` in
        # Gripper_Link3_1 (`_hand_frame`) -- feed a servo the array for the frame it measures.
        key = "obj_in_weldbody" if os.environ.get("REC_FRAME", "weld") == "weld" else "obj_in_gripper"
        og = d.get(key) or d.get("obj_in_gripper")
        # The grip relation comes from the END of the `close` phase, not the last row replayed.
        grip_row = max([i for i, s in enumerate(ph) if s == "close"] or rows)
        if og:
            self._rec_obj_in_grip = np.array(og[grip_row][:3], float)
            # keep the LINK3 array too -- the frame the FINGERS live in
            _g3 = d.get("obj_in_gripper")
            if _g3:
                self._rec_obj_in_grip_L3 = np.array(_g3[grip_row][:3], float)
            print(f">>> recorded {key} at grasp (frame "
                  f"{d['meta'].get('weld_frame_body' if key == 'obj_in_weldbody' else 'gripper_frame_body')}): "
                  f"{np.round(self._rec_obj_in_grip, 4).tolist()}", flush=True)
        return {"names": names, "idx": np.array([self.idx[n] for n in names]), "frames": fr}

    def _sync_object_size(self):
        """Overwrite the json object radius/half-height with the stage cylinder's real size."""
        from pxr import UsdGeom as _UG
        # Author the radius on the prim, or OBJ_RADIUS is lost: this overwrites KNOWN from the stage.
        _ro = os.environ.get("OBJ_RADIUS")
        if _ro:
            _rv = float(_ro)
            _n_au = 0
            for prim in self.stage.Traverse():
                if "pickup_obj" in prim.GetPath().pathString and prim.GetTypeName() == "Cylinder":
                    _a = _UG.Cylinder(prim).GetRadiusAttr()
                    if _a and abs(float(_a.Get() or 0.0) - _rv) > 1e-6:
                        _a.Set(_rv)
                        _n_au += 1
            if _n_au:
                print(f">>> object size: AUTHORED radius {_rv:.3f} on {_n_au} pickup cylinders "
                      f"(OBJ_RADIUS)", flush=True)
        if os.environ.get("OBJ_HALF_H"):
            # explicit override wins
            return
        for prim in self.stage.Traverse():
            p = prim.GetPath().pathString
            if "pickup_obj_0" in p and prim.GetTypeName() == "Cylinder":
                cy = _UG.Cylinder(prim)
                h = float(cy.GetHeightAttr().Get() or 0.0)
                r = float(cy.GetRadiusAttr().Get() or 0.0)
                if h > 0:
                    if abs(h / 2.0 - self.obj_half_h) > 1e-4:
                        print(f">>> object size: measured half-height {h / 2.0:.3f} "
                              f"(json said {self.obj_half_h:.3f}) -> using the measurement", flush=True)
                    self.obj_half_h = h / 2.0
                if r > 0:
                    KNOWN["object_radius"] = r
                    print(f">>> object size: radius {r:.3f}, half-height {self.obj_half_h:.3f}",
                          flush=True)
                return
