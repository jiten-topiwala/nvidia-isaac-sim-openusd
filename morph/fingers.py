"""Finger geometry and contact sensing: hull/pad positions, surface gaps, joint reaction forces.

Extracted verbatim from play_isaac.py — the method bodies are unchanged.
"""
import math
import os

import numpy as np
from isaacsim.core.prims import SingleXFormPrim
from pxr import Usd, UsdPhysics

from morph.config import PERFECT, KNOWN
from morph.usd_utils import find_path


class FingersMixin:
    """Mixed into Demo. Every `self.*` it touches is owned by Demo."""

    def _finger_hulls(self):
        """Per-finger DISTAL COLLIDER vertices in local space, cached.

        The link_3 ORIGIN is a bad proxy for where the pad actually is: the distal geometry extends
        ~27mm inward of it, so an origin-based stop is wrong in BOTH directions — pad 0.030 left the
        tips visibly short of the object, while a smaller one buries the fingers inside it.
        Measuring the real collider vertices removes the magic offset entirely and works for any
        object size.
        """
        if hasattr(self, "_hulls"):
            return self._hulls
        from pxr import Usd, UsdGeom, UsdPhysics
        self._hulls = {}
        for f in "abc":
            got = []
            for prim in self.stage.Traverse(Usd.TraverseInstanceProxies()):
                path = prim.GetPath().pathString
                # EVERY collider of this finger, not just the distal link: this hand grips on the
                # mid-links too, so a link_3-only scan reports a finger tens of millimetres out when
                # its pads are on the surface.
                if f"finger_{f}_" not in path or "Gripper_Link3_1/" not in path:
                    continue
                # Pads only: keep colliders whose DEEPEST body is link_2 or link_3. link_1 is the
                # proximal segment off the palm, and with the hand open it hangs OVER the object and
                # becomes the closest vertex, which reads as a pad far from the object when its real
                # gripping surfaces are elsewhere.
                if os.environ.get("GAP_INCLUDE_LINK1", "0") != "1" and \
                        f"finger_{f}_link_2_1" not in path and f"finger_{f}_link_3_1" not in path:
                    continue
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                    continue
                t = prim.GetTypeName()
                if t == "Mesh":
                    v = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                elif t == "Cube":
                    e = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0) / 2.0
                    v = [(sx * e, sy * e, sz * e)
                         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
                else:
                    continue
                if v:
                    got.append((prim, np.asarray([[float(c) for c in q] for q in v], float)))
            self._hulls[f] = got
        # The palm is tracked as a fourth "finger", so that "is the gripper centre inside the
        # object?" has an answer.
        palm = []
        for prim in self.stage.Traverse(Usd.TraverseInstanceProxies()):
            path = prim.GetPath().pathString
            # Only the real palm. Everything under Gripper_Link3_1 without "finger" in its path
            # sweeps in a finger knuckle and a mounting bracket as well, and a knuckle vertex sits
            # well above and forward of the pads -- so the seat gets halted by geometry reported as
            # "palm", and every mouth-depth number taken off that hull measures the wrong thing.
            if "Gripper_Link3_1" not in path or "finger" in path:
                continue
            if os.environ.get("PALM_REAL_ONLY", "1") == "1" and "palm_real" not in path:
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                continue
            t = prim.GetTypeName()
            if t == "Mesh":
                v = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            elif t == "Cube":
                e = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0) / 2.0
                v = [(sx * e, sy * e, sz * e)
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
            else:
                continue
            if v:
                palm.append((prim, np.asarray([[float(c) for c in q] for q in v], float)))
        self._hulls["palm"] = palm
        print(">>> finger hulls: " + ", ".join(f"{f}:{sum(len(a) for _, a in v)}v"
                                               for f, v in self._hulls.items()), flush=True)
        return self._hulls

    def _finger_gap_parts(self, f, oc, radius, half_h, R=None):
        """Same measurement as _finger_surface_gap, but split into RADIAL and VERTICAL components
        for the closest vertex.  A 12mm gap that is purely vertical means the pad is above the
        cylinder's rim and the fix is descent DEPTH; a radial one means the object is out of the
        finger's reach and no depth change will help."""
        from pxr import Usd, UsdGeom
        best, parts = 1e9, (0.0, 0.0)
        for prim, loc in self._finger_hulls().get(f, ()):
            m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()), float)
            a = loc @ m[:3, :3] + m[3, :3]
            # Object frame: R is the carried object's world rotation, so once the pin tilts the
            # object with the hand, treating it as a world-Z cylinder mismeasures every gap (it
            # reported b opening 1.4 -> 9.9mm across a lift in which the finger joints never moved).
            p = (a - oc) @ R if R is not None else (a - oc)
            dr = np.hypot(p[:, 0], p[:, 1]) - radius
            dz = np.abs(p[:, 2]) - half_h
            d = np.hypot(np.maximum(dr, 0), np.maximum(dz, 0))
            j = int(np.argmin(d))
            if float(d[j]) < best:
                best = float(d[j])
                parts = (float(dr[j]), float(dz[j]))
        return best, parts

    def _finger_surface_gap(self, f, oc, radius, half_h, R=None):
        """Smallest distance from finger f's real distal colliders to the cylinder SURFACE.
        Negative = penetrating.  This is the number the contact-stop should use."""
        from pxr import Usd, UsdGeom
        best = 1e9
        for prim, loc in self._finger_hulls().get(f, ()):
            m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()), float)               # USD row-vector convention
            a = loc @ m[:3, :3] + m[3, :3]
            # Object frame: R is the carried object's world rotation, so once the pin tilts the
            # object with the hand, treating it as a world-Z cylinder mismeasures every gap (it
            # reported b opening 1.4 -> 9.9mm across a lift in which the finger joints never moved).
            p = (a - oc) @ R if R is not None else (a - oc)
            dr = np.hypot(p[:, 0], p[:, 1]) - radius
            dz = np.abs(p[:, 2]) - half_h
            inside = (dr < 0) & (dz < 0)
            if inside.any():
                d = -float(np.minimum(-dr[inside], -dz[inside]).max())
            else:
                d = float(np.hypot(np.maximum(dr, 0), np.maximum(dz, 0)).min())
            best = min(best, d)
        return best

    def _tip_prims(self):
        if not hasattr(self, "_tips"):
            self._tips = {f: SingleXFormPrim(find_path(self.stage, f"finger_{f}_link_3_1"))
                          for f in "abc"}
        return self._tips

    def _pinch(self):
        """Live finger-pinch centroid: the thumb against the b/c midpoint."""
        t = {f: np.asarray(p.get_world_pose()[0], float) for f, p in self._tip_prims().items()}
        return 0.5 * (t["a"] + 0.5 * (t["b"] + t["c"]))

    def _pad_trace(self, tag):
        """Per-pad force + command-vs-actual at one instant, for localising WHERE a grip is lost.

        The close ends with all three pads loaded (measured {a 2.34, b 2.81, c 2.12}N) and the grip
        check one stage later reports only `b`. Several things happen in between -- the hold gains go
        from kp 30 to kp 6e3, the seat runs, the capture runs -- and none of them were instrumented,
        so the loss could not be attributed. Print the same three numbers at each boundary and the
        stage that drops them is named rather than guessed.
        """
        try:
            eff = self._finger_efforts()
            qa = np.asarray(self.robot.get_joint_positions(), float)
            bits = []
            for f in "abc":
                i1 = self.idx.get(f"finger_{f}_joint_1_1")
                if i1 is None:
                    continue
                c = (float(self._finger_cmd[list(self._f_idx).index(i1)])
                     if (self._finger_cmd is not None and i1 in list(self._f_idx)) else float("nan"))
                bits.append(f"{f}: eff {float(eff.get(f, 0.0)):+.3f}Nm "
                            f"cmd{c:+.3f} act{float(qa[i1]):+.3f} err{c - float(qa[i1]):+.4f}")
            # What did the articulation actually receive? `_finger_cmd` is this module's
            # bookkeeping; the drive target is what physics acts on, and the two can disagree.
            try:
                _aa = self.robot.get_articulation_controller().get_applied_action()
                _tg = np.asarray(_aa.joint_positions, float).reshape(-1)
                _ab = []
                for f in "abc":
                    i1 = self.idx.get(f"finger_{f}_joint_1_1")
                    if i1 is None or i1 >= len(_tg) or _tg[i1] is None:
                        continue
                    _ab.append(f"{f}: drive_tgt {float(_tg[i1]):+.3f} act {float(qa[i1]):+.3f}")
                print(f">>> pad-trace[{tag}]: " + " | ".join(bits), flush=True)
                print(f">>>   applied-action[{tag}]: " + " | ".join(_ab), flush=True)
            except Exception as _e_aa:
                print(f">>> pad-trace[{tag}]: " + " | ".join(bits), flush=True)
                print(f">>>   applied-action[{tag}]: unavailable ({_e_aa})", flush=True)
        except Exception as e:
            print(f">>> pad-trace[{tag}]: failed ({e})", flush=True)

    def _du_dv(self, obj):
        """The object's position relative to the PINCH, in (mouth, jaw) coordinates — the same
        frame the geometry report prints.

        ONE definition, used by both, because two nearly-identical ones silently disagreed: the
        report builds the jaw axis from each finger's vertex CLOSEST TO THE OBJECT, an enclose servo
        built it from the MEAN of every link vertex, and the two `du` values differed enough
        (-107mm vs a distance of only 84mm) that the servo could not tell it was already inside the
        contact band and drove the hand 400mm the wrong way.

        Returns (du, dv) in metres, or None if a finger's geometry is unavailable.
        """
        oc = np.asarray(obj.get_world_poses()[0][0], float)
        pd = {}
        for f in "abc":
            pts = self._finger_world_pts(f)
            if pts is None or len(pts) == 0:
                return None
            pd[f] = pts[int(np.argmin(np.linalg.norm((pts - oc)[:, :2], axis=1)))]
        ax = (0.5 * (pd["b"] + pd["c"]) - pd["a"])[:2]
        n = float(np.linalg.norm(ax))
        if n < 1e-9:
            # Degenerate jaw axis: the thumb's closest vertex coincides with the b/c midpoint in XY.
            if getattr(self, "_ax_last", None) is None:
                if not getattr(self, "_ax_warned", False):
                    print(">>> _du_dv: degenerate jaw axis and no previous one -> None", flush=True)
                    self._ax_warned = True
                return None
            ax = self._ax_last
            if not getattr(self, "_ax_warned", False):
                print(">>> _du_dv: degenerate jaw axis -> reusing the last good one", flush=True)
                self._ax_warned = True
        else:
            ax = ax / n
            self._ax_last = ax
        mo = np.array([-ax[1], ax[0]])
        rel = (oc - self._pinch())[:2]
        return float(rel @ mo), float(rel @ ax)

    def _finger_curl_targets(self):
        """The reference model's close curl: size-aware close intensity
        i from ctrl = 0.20 - 4.0*radius (clamped [0.15, 0.20]) and the wrap profile j3 > j2 > j1 —
        the distal joint curls furthest, so the fingertips wrap around the object's sides."""
        ctrl = float(np.clip(0.20 - 4.0 * KNOWN["object_radius"], 0.15, 0.20))
        i = ctrl / 0.20
        if PERFECT:
            # command FULL curl and let the per-finger STALL-STOP halt at contact (MuJoCo's force-
            # close semantics).
            i = 1.0
        j1, j2, j3 = i * 0.70 * 0.85, i * 0.90 * 0.95, -0.052 - i * 1.10
        t = {}
        for f in "abc":
            t[f"finger_{f}_joint_1_1"], t[f"finger_{f}_joint_2_1"], t[f"finger_{f}_joint_3_1"] = j1, j2, j3
        return t

    def _tip_gaps(self, obj):
        """Per-finger tip -> object SURFACE distance (m).  MuJoCo staggers the close by these:
        the finger with the longest reach starts first so all pads LAND TOGETHER."""
        oc = np.asarray(obj.get_world_poses()[0][0], float)
        out = {}
        for f, prim in self._tip_prims().items():
            tp = np.asarray(prim.get_world_pose()[0], float)
            out[f] = max(0.0, math.hypot(tp[0] - oc[0], tp[1] - oc[1]) - KNOWN["object_radius"])
        return out

    def _finger_reaction(self):
        """Max |F| across the three finger joint_1 reactions — a strong, clean contact signal (the
        object's net contact force reads ~0 here, but a finger pressing the object spikes its reaction)."""
        return max(self._finger_j1_force(f) for f in "abc")

    def _finger_efforts(self):
        """Per-finger MEASURED drive torque (Nm) at joint_1 — get_measured_joint_efforts is the
        solver's applied-after-clamping torque, so a pad pressing a blocked joint reads its
        saturated effort.  Cleaner grip evidence than the 6D reaction wrench (which carries a
        150-1400N structural baseline) and independent of the contact-event stream."""
        try:
            e = np.asarray(self.robot.get_measured_joint_efforts(), float).ravel()
            out = {}
            for f in "abc":
                i = self.idx.get(f"finger_{f}_joint_1_1")
                if i is not None and i < len(e):
                    out[f] = round(float(e[i]), 2)
            return out
        except Exception:
            return {}

    def _finger_lever(self, fl, oc, radius, half_h, R=None):  # noqa: C901
        """MEASURED moment arm (m) from finger `fl`'s joint_1 axis to its contact point.

        The force target has to be converted to a drive torque, and assuming a lever is assuming
        the answer: tau = F * L, so a wrong L scales the commanded force by exactly that error.
        Take L as the distance from the joint_1 body origin to the pad vertex that is CLOSEST to the
        object — i.e. the point that will actually carry the contact."""
        try:
            from pxr import UsdGeom as _UG
            cache = _UG.XformCache()
            j1 = next(pr for pr in self.stage.Traverse(Usd.TraverseInstanceProxies())
                      if pr.GetName() == f"finger_{fl}_link_1_1")
            jp = np.asarray(cache.GetLocalToWorldTransform(j1).ExtractTranslation(), float)
            a = self._finger_world_pts(fl)
            if a is None or not len(a):
                return None
            p = (a - oc) @ R if R is not None else (a - oc)
            dr = np.hypot(p[:, 0], p[:, 1]) - radius
            dz = np.abs(p[:, 2]) - half_h
            k = int(np.argmin(np.maximum(dr, dz)))
            return float(np.linalg.norm(np.asarray(a[k], float) - jp))
        except Exception as e:
            print(f">>>   lever({fl}) measurement FAILED: {e}", flush=True)
            return None

    def _finger_gap_owner(self, f, oc, radius, half_h, R=None):
        """(gap, owning-collider-name) for finger `f` — same math as _finger_surface_gap, but it
        also reports WHICH prim held the closest vertex.

        `_finger_surface_gap` takes the MIN over link_2 + link_3 colliders, so its value can JUMP
        discontinuously when the closest vertex hops from one collider to another — the gap moves
        without the pad moving.  That is the leading explanation for the one measurement that
        breaks the geometry: 3-4mm of hand travel opening the thumb from 4.7mm to 18-62mm while the
        object moves 3-11mm.  If the owner changes across that jump, the metric is the artifact."""
        from pxr import Usd, UsdGeom
        best, owner = 1e9, None
        for prim, loc in self._finger_hulls().get(f, ()):
            m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()), float)
            a = loc @ m[:3, :3] + m[3, :3]
            p = (a - oc) @ R if R is not None else (a - oc)
            dr = np.hypot(p[:, 0], p[:, 1]) - radius
            dz = np.abs(p[:, 2]) - half_h
            inside = (dr < 0) & (dz < 0)
            if inside.any():
                d = -float(np.minimum(-dr[inside], -dz[inside]).max())
            else:
                d = float(np.hypot(np.maximum(dr, 0), np.maximum(dz, 0)).min())
            if d < best:
                best, owner = d, prim.GetPath().pathString.rsplit("/", 2)[-2:]
        return best, ("/".join(owner) if owner else "?")

    def _finger_world_pts(self, fl):
        """Finger `fl`'s pad vertices in WORLD space, as one (N,3) array.

        `_finger_hulls()` returns [(prim, LOCAL verts), ...] — a ragged list of tuples.  Feeding it
        straight to np.asarray raises "inhomogeneous shape after 2 dimensions", which is exactly
        what silently defeated both the lever measurement and the jaw-seat nudge (each fell back or
        was skipped, with the failure swallowed).  Transform per prim, same as _finger_surface_gap."""
        from pxr import Usd, UsdGeom
        out = []
        for prim, loc in self._finger_hulls().get(fl, ()):
            m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()), float)                  # USD row-vector convention
            out.append(loc @ m[:3, :3] + m[3, :3])
        return np.vstack(out) if out else None

    def _pad_force_step(self, fl):
        """THIS step's contact force (N) on finger `fl` — the tactile sensor for the preload loop.
        PhysX reports impulses; force = impulse / dt."""
        return float(self._fN_step.get(fl, 0.0)) / max(1e-9, self.dt)

    def _wrap_gap(self, f, oc, r, hh, R=None):
        """Minimum distance from ANY of finger `f`'s links to the cylinder surface.

        `_finger_surface_gap` measures the DISTAL PAD only.  That is the right metric for a PINCH
        and the wrong one for a WRAP, which contacts on link_1/link_2/link_3 as the fingers curl
        around the object.  Measured proof: `align-eq` balanced the PAD gaps to
        {a 7.9, b 8.1, c 6.3}mm and the wrap still came out one-sided
        ({a 0.8N, b 17.7N, c 1.8N}) — the fingers were balanced on surfaces that never touch."""
        # link_2 MATTERS MOST and was missing: the contact ledger shows `finger_b_link_2_1` and
        # `link_3` carrying the wrap, and with only link_1+link_3 this returned the PAD distance
        # unchanged (7.9/8.1/6.3 -> 7.8/8.2/6.3, i.e.
        if not hasattr(self, "_k2"):
            from pxr import Usd as U2, UsdGeom as G2, UsdPhysics as P2
            self._k2 = {}
            for f2 in "abc":
                acc = []
                for pr2 in self.stage.Traverse(U2.TraverseInstanceProxies()):
                    pp2 = pr2.GetPath().pathString
                    if f"finger_{f2}_link_2_1" not in pp2:
                        continue
                    if not pr2.HasAPI(P2.CollisionAPI) or pr2.GetTypeName() != "Mesh":
                        continue
                    if P2.CollisionAPI(pr2).GetCollisionEnabledAttr().Get() is False:
                        continue          # disabled != absent; see _knuckle_hulls
                    v2 = G2.Mesh(pr2).GetPointsAttr().Get()
                    if v2:
                        acc.append((pr2, np.asarray([[float(c) for c in q] for q in v2], float)))
                self._k2[f2] = acc
        pts = []
        for pr, loc in self._k2.get(f, ()):                  # link_2 (the wrap's main contact)
            from pxr import Usd, UsdGeom
            m = np.array(UsdGeom.Xformable(pr).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()), float)
            pts.append(loc @ m[:3, :3] + m[3, :3])
        for pr, loc in self._knuckle_hulls().get(f, ()):     # link_1
            from pxr import Usd, UsdGeom
            m = np.array(UsdGeom.Xformable(pr).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()), float)
            pts.append(loc @ m[:3, :3] + m[3, :3])
        w = self._finger_world_pts(f)                        # link_3 (distal pads)
        if w is not None and len(w):
            pts.append(w)
        if not pts:
            return 1e9
        P = np.vstack(pts)
        d = P - oc
        if R is not None:                                    # into the cylinder's own frame
            d = d @ R
        radial = np.linalg.norm(d[:, :2], axis=1) - r
        axial = np.abs(d[:, 2]) - hh
        g = np.where(axial > 0.0, np.hypot(np.maximum(radial, 0.0), axial), radial)
        return float(g.min())

    def _knuckle_hulls(self):
        """`finger_*_link_1` collider vertices (local), cached — the KNUCKLES.

        These, not the pads, are the narrow point of the jaw: measured span 44.9mm against a 188mm
        pad opening, and `finger_*_link_1` tops the contact ledger in every run at every object
        size.  Built lazily HERE rather than inside the close-entry diagnostic, which runs AFTER
        the seat — so the seat's throat servo found no hulls and silently never ran."""
        if hasattr(self, "_k1"):
            return self._k1
        from pxr import Usd, UsdGeom, UsdPhysics
        self._k1 = {}
        for f in "abc":
            pts = []
            for prim in self.stage.Traverse(Usd.TraverseInstanceProxies()):
                path = prim.GetPath().pathString
                if f"finger_{f}_link_1_1" not in path:
                    continue
                if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetTypeName() != "Mesh":
                    continue
                # ...and it must still be ENABLED. `HasAPI` stays True after
                # `CreateCollisionEnabledAttr(False)`, so with the proximal colliders switched off
                # this would keep measuring geometry the solver no longer collides.
                if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                    continue
                v = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                if v:
                    pts.append((prim, np.asarray([[float(c) for c in q] for q in v], float)))
            self._k1[f] = pts
        print(">>> knuckle hulls: " + ", ".join(f"{f}:{sum(len(a) for _, a in v)}v"
                                                for f, v in self._k1.items()), flush=True)
        return self._k1

    def _finger_j1_force(self, f):
        """|F| at one finger's joint_1 reaction (contact evidence for THAT finger)."""
        try:
            jf = np.asarray(self.robot.get_measured_joint_forces(), float)
            off = 0 if jf.shape[0] == len(self.names) else 1     # some builds prepend a root dof
            i = self.idx[f"finger_{f}_joint_1_1"] + off
            return float(np.linalg.norm(jf[i][:3])) if i < len(jf) else 0.0
        except Exception:
            return 0.0

    def _lj_stop(self, q, f, ix, gL, best, done, why):
        """Retire one finger from the legal close: command := ACTUAL on all three of its joints.
        COMMAND != ACTUAL on a force-limited drive — leaving the command ahead means the finger
        keeps pressing after we believe it stopped (this file's most repeated bug, 08-11bb)."""
        qa = np.asarray(self.robot.get_joint_positions(), float)
        for jn in (1, 2, 3):
            ji = self.idx.get(f"finger_{f}_joint_{jn}_1")
            if ji is not None:
                q[ji] = float(qa[ji])
        done.add(f)
        print(f">>>   legal close: {f} stopped at j1 {float(q[ix]):+.3f} gap {gL * 1000:+.1f}mm "
              f"(best {best[f] * 1000:+.1f}mm) — {why}", flush=True)
