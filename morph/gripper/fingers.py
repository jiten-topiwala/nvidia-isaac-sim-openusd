"""Finger geometry and contact sensing: hull/pad positions, surface gaps, joint reactions.
Fingers a/b/c (a = thumb, b/c = the opposing pair), links 1 (knuckle) to 3 (distal pad);
closing is +ve on joint_1 and joint_2, -ve on joint_3."""
import math

import numpy as np
from isaacsim.core.prims import SingleXFormPrim
from pxr import Usd, UsdPhysics

from morph.config import KNOWN
from morph.usd_utils import find_path


class FingersMixin:
    """Mixed into Demo. Every `self.*` it touches is owned by Demo."""

    def _finger_hulls(self):
        """Per-finger pad collider vertices in local space, cached. Measured off the real mesh:
        the link_3 origin sits ~27mm inward of the pad, so any origin-based stop is wrong."""
        if hasattr(self, "_hulls"):
            return self._hulls
        from pxr import Usd, UsdGeom, UsdPhysics
        self._hulls = {}
        for f in "abc":
            got = []
            for prim in self.stage.Traverse(Usd.TraverseInstanceProxies()):
                path = prim.GetPath().pathString
                # Every collider of this finger, not just the distal link: this hand grips on the
                # mid-links too, so a link_3-only scan reports a pad on the surface as far out.
                if f"finger_{f}_" not in path or "Gripper_Link3_1/" not in path:
                    continue
                # Pads only (link_2/link_3). link_1 is the proximal segment off the palm; with the
                # hand open it hangs OVER the object and would win as the closest vertex.
                if f"finger_{f}_link_2_1" not in path and f"finger_{f}_link_3_1" not in path:
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
        # The palm is a fourth "finger", so "is the gripper centre inside the object?" has an answer.
        palm = []
        for prim in self.stage.Traverse(Usd.TraverseInstanceProxies()):
            path = prim.GetPath().pathString
            # Real palm only: the wider match also sweeps in a knuckle and a mounting bracket, whose
            # vertices sit above and forward of the pads and would halt the seat as "palm" contact.
            if "Gripper_Link3_1" not in path or "finger" in path:
                continue
            if "palm_real" not in path:
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
        """_finger_surface_gap split into RADIAL and VERTICAL components for the closest vertex.
        Vertical means the pad is above the rim (fix descent depth); radial means out of reach."""
        from pxr import Usd, UsdGeom
        best, parts = 1e9, (0.0, 0.0)
        for prim, loc in self._finger_hulls().get(f, ()):
            m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()), float)
            a = loc @ m[:3, :3] + m[3, :3]
            # R is the object's world rotation: without it a tilted object is measured as a world-Z
            # cylinder and every gap is wrong, including for fingers that never moved.
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
            # R is the object's world rotation: without it a tilted object is measured as a world-Z
            # cylinder and every gap is wrong, including for fingers that never moved.
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
        """Per-pad force + command-vs-actual at one instant. Call it at each stage boundary to
        localise WHICH stage loses a grip."""
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
            # `_finger_cmd` is this module's bookkeeping; the drive target is what physics acts on.
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
        """The object's position relative to the PINCH, in (mouth, jaw) coordinates — the same frame
        the geometry report prints, and the only definition of it, since a second one built from a
        different vertex set disagreed by enough to send a servo the wrong way.

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

    def _tip_gaps(self, obj):
        """Per-finger tip -> object SURFACE distance (m). The close stagger is derived from these so
        that all three pads land together."""
        oc = np.asarray(obj.get_world_poses()[0][0], float)
        out = {}
        for f, prim in self._tip_prims().items():
            tp = np.asarray(prim.get_world_pose()[0], float)
            out[f] = max(0.0, math.hypot(tp[0] - oc[0], tp[1] - oc[1]) - KNOWN["object_radius"])
        return out

    def _finger_reaction(self):
        """Max |F| across the three finger joint_1 reactions — the object's net contact force reads
        ~0 here, but a finger pressing it spikes that finger's reaction."""
        return max(self._finger_j1_force(f) for f in "abc")

    def _finger_efforts(self):
        """Per-finger MEASURED drive torque (Nm) at joint_1: applied-after-clamping, so a pad
        pressing a blocked joint reads its saturated effort. Cleaner grip evidence than the 6D
        reaction wrench, which carries a large structural baseline."""
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
        """MEASURED moment arm (m) from finger `fl`'s joint_1 origin to the pad vertex closest to
        the object. tau = F * L, so a wrong L scales the commanded force by exactly that error."""
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
        """(gap, owning-collider-name) for finger `f` — _finger_surface_gap plus WHICH prim held the
        closest vertex. That min is over several colliders, so it can jump discontinuously when the
        owner changes; if the owner moved, the jump is a metric artifact, not pad motion."""
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
        """Finger `fl`'s pad vertices in WORLD space, as one (N,3) array. `_finger_hulls()` returns
        a ragged [(prim, local verts), ...] list that np.asarray cannot stack — callers that tried
        swallowed the exception and silently fell back — so transform per prim instead."""
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
        """Minimum distance from ANY of finger `f`'s links (1, 2 and 3) to the cylinder surface —
        the WRAP metric. `_finger_surface_gap` measures the distal pad only, which is right for a
        pinch and wrong here: balanced pad gaps still give a one-sided wrap."""
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
                        # disabled != absent; see _knuckle_hulls
                        continue
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
        # into the cylinder's own frame
        if R is not None:
            d = d @ R
        radial = np.linalg.norm(d[:, :2], axis=1) - r
        axial = np.abs(d[:, 2]) - hh
        g = np.where(axial > 0.0, np.hypot(np.maximum(radial, 0.0), axial), radial)
        return float(g.min())

    def _knuckle_hulls(self):
        """`finger_*_link_1` collider vertices (local), cached — the KNUCKLES, and the narrow point
        of the jaw (~45mm span against a 188mm pad opening). Built lazily here, not in the
        close-entry diagnostic, which runs after the seat and left the throat servo with no hulls."""
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
                # ...and still ENABLED: HasAPI stays True after CreateCollisionEnabledAttr(False),
                # which would leave this measuring geometry the solver no longer collides.
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
