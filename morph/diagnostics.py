"""Reporting only — measures and prints, never changes state.

Extracted verbatim from play_isaac.py — the method bodies are unchanged.
"""
import json
import math
import os

import numpy as np
from morph.config import HERE, KNOWN


class DiagnosticsMixin:
    """Mixed into Demo. Every `self.*` it touches is owned by Demo."""

    def _tip_report(self, tag, obj):
        """Where each fingertip stands relative to the object — the measurement that decides
        whether the cage is actually AROUND the object or reaching past it."""
        oc = np.asarray(obj.get_world_poses()[0][0], float)
        r = KNOWN["object_radius"]
        bx, by, byaw = self.base_ledger()
        c_b, s_b = math.cos(-byaw), math.sin(-byaw)
        dxb, dyb = float(oc[0]) - bx, float(oc[1]) - by
        obj_base = (c_b * dxb - s_b * dyb, s_b * dxb + c_b * dyb)
        rows = []
        for f, prim in self._tip_prims().items():
            tp = np.asarray(prim.get_world_pose()[0], float)
            dxt, dyt = float(tp[0]) - bx, float(tp[1]) - by
            tb = (c_b * dxt - s_b * dyt, s_b * dxt + c_b * dyt)   # tip in BASE frame: fwd, left
            rows.append(f"{f}: base=({tb[0]:.3f},{tb[1]:.3f},{tp[2]:.3f}) "
                        f"surface{(math.hypot(tp[0] - oc[0], tp[1] - oc[1]) - r) * 1000:+.0f}mm")
        print(f">>> tips@{tag}: " + " | ".join(rows), flush=True)
        q_f = np.asarray(self.robot.get_joint_positions(), float)
        jj = {f: [round(float(q_f[self.idx[f"finger_{f}_joint_{n}_1"]]), 2) for n in (1, 2, 3)]
              for f in "abc" if f"finger_{f}_joint_1_1" in self.idx}
        print(f">>>   finger joints [j1,j2,j3] {jj}  object in base frame "
              f"{np.round(obj_base, 3).tolist()} vs baked LOCAL_OBJ {np.round(self.local_obj, 3).tolist()}",
              flush=True)

    def _arm_body_clearance(self, tag=""):
        """Log each arm-chain link's signed clearance to the chassis plate (x +/-0.35, y +/-0.30 in
        the BASE frame) plus the joint angles. Self-collision is OFF (enabling it OOMs), so PhysX
        never reports arm<->body contact -- this GEOMETRIC readout is the only signal for the
        arm entering the body. A NEGATIVE clearance means the link is inside the plate footprint,
        i.e. overlapping the chassis. Called through the descent, so the tilt and pose at which a
        link first goes negative can be read straight from the log and used to calibrate the limit.
        """
        try:
            from isaacsim.core.prims import SingleXFormPrim
            from morph.usd_utils import find_path
            if getattr(self, "_arm_link_prims", None) is None:
                self._arm_link_prims = {}
                for nm in ("Bearing_Column_Left_1", "Rotation_Link_Left_1",
                           "Arm_Left_1", "Hand_Bearing_1"):
                    try:
                        self._arm_link_prims[nm] = SingleXFormPrim(find_path(self.stage, nm))
                    except Exception:
                        pass
            bx, by, byaw = self.base_ledger()
            c, s = math.cos(-byaw), math.sin(-byaw)
            q = np.asarray(self.robot.get_joint_positions(), float)
            h1 = float(q[self.idx["ColumnLeftBearingJoint_1"]])
            h2 = float(q[self.idx["ColumnRightBearingJoint_1"]])
            a1 = float(q[self.idx["ArmLeftJoint_1"]]); th = float(q[self.idx["BaseJoint_1"]])
            parts, worst = [], (1e9, "")
            for nm, p in self._arm_link_prims.items():
                w = np.asarray(p.get_world_pose()[0], float)
                dx, dy = w[0] - bx, w[1] - by
                px, py = c * dx - s * dy, s * dx + c * dy         # world -> base frame
                clr = max(abs(px) - self.CHASSIS_PLATE_X, abs(py) - self.CHASSIS_PLATE_Y)  # <0 = inside
                short = nm.replace("_Left_1", "").replace("_1", "")
                parts.append(f"{short}: base({px:+.3f},{py:+.3f}) z{w[2]:.3f} clr{clr * 1000:+.0f}")
                # MIN over the LOW, swinging links only (z < 0.35). The mount (Bearing_Column) and
                # shoulder (Rotation_Link) sit ON the chassis, high up (z 0.46-0.70) -- their XY is
                # always "inside" the footprint but they are ABOVE the plate, not colliding.
                if w[2] < 0.35 and clr < worst[0]:
                    worst = (clr, short)
            print(f">>> arm-body[{tag}]: dh {h2 - h1:+.3f} (h1 {h1:.3f} h2 {h2:.3f}) a1 {a1:.3f} "
                  f"th {th:+.3f} | MIN clr {worst[0] * 1000:+.0f}mm ({worst[1]}) | "
                  + " | ".join(parts), flush=True)
        except Exception as _e:
            print(f">>> arm-body[{tag}]: failed ({_e})", flush=True)

    def _finite(self, stage):
        """True if the articulation is still finite; prints the failing stage otherwise (the
        forced-linkage explosion is nondeterministic — knowing WHICH stage died is the only way
        to localize it in a headless run)."""
        ok = bool(np.all(np.isfinite(np.asarray(self.robot.get_joint_positions(), float))))
        if not ok:
            print(f">>> NONFINITE after {stage}", flush=True)
        return ok

    def _phase_report(self, tag, obj):
        """Per-phase ground truth: every finger joint against the RECORDED grasp reference, the
        real collider-to-surface gap per finger, and where the object sits in the gripper frame.

        Printing joint COMMAND against ACTUAL against REFERENCE is what distinguishes "the drive
        never got there" from "contact pushed it back" from "something reset the target" — three
        failures that look identical in the finger positions alone.
        """
        try:
            _op, _oq = obj.get_world_poses()
            oc = np.asarray(_op[0], float)
            oR = self._quat_to_R(np.asarray(_oq[0], float))       # the object may be TILTED in-hand
            r = KNOWN["object_radius"]
            qa = np.asarray(self.robot.get_joint_positions(), float)
            cmd = self._finger_cmd
            ref = None
            if self.close_traj is not None:
                ref = {n: float(v) for n, v in zip(self.close_traj["names"],
                                                   self.close_traj["frames"][-1])}
            gaps = {f: round(self._finger_surface_gap(f, oc, r, self.obj_half_h, oR) * 1000, 1)
                    for f in "abc"}
            split = {}
            for f in "abc":
                _b, (dr_, dz_) = self._finger_gap_parts(f, oc, r, self.obj_half_h, oR)
                split[f] = f"dr{dr_ * 1000:+.0f}/dz{dz_ * 1000:+.0f}"
            print(f">>> [{tag}] finger gaps(mm) {gaps}  radial/vertical {split}", flush=True)
            # PALM/gripper-centre clearance. Negative = the gripper body itself is inside the
            # cylinder — the palm entering the object, which no finger gap can reveal.
            gp_, gR_ = self._grip_frame()
            pgap, (pdr, pdz) = self._finger_gap_parts("palm", oc, r, self.obj_half_h, oR)
            print(f">>>   [{tag}] PALM shell gap {self._finger_surface_gap('palm', oc, r, self.obj_half_h, oR) * 1000:+.1f}mm "
                  f"(dr{pdr * 1000:+.0f}/dz{pdz * 1000:+.0f})  grip-origin radial "
                  f"{(math.hypot(gp_[0] - oc[0], gp_[1] - oc[1]) - r) * 1000:+.0f}mm "
                  f"z {(gp_[2] - oc[2]) * 1000:+.0f}mm", flush=True)
            # Reference check: the same palm shell against the object the grasp was RECORDED on, at
            # the raw recorded relation.
            raw = getattr(self, "_rec_obj_in_grip", None)
            if raw is not None:
                mj = json.load(open(os.path.join(HERE, "usd/_grasp_close.json")))["meta"] \
                    if not hasattr(self, "_mj_meta") else self._mj_meta
                self._mj_meta = mj
                ocv = gp_ + gR_ @ np.asarray(raw, float)
                print(f">>>   [{tag}] REF palm vs recorded-size obj at raw relation: "
                      f"{self._finger_surface_gap('palm', ocv, float(mj['obj_radius']), float(mj['obj_half_height'])) * 1000:+.1f}mm",
                      flush=True)
            for f in "abc":
                bits = []
                for jn in (f"finger_{f}_joint_1_1", f"finger_{f}_joint_2_1",
                           f"finger_{f}_joint_3_1"):
                    i = self.idx.get(jn)
                    if i is None:
                        continue
                    a = float(qa[i])
                    c = (float(cmd[list(self._f_idx).index(i)])
                         if cmd is not None and i in list(self._f_idx) else float("nan"))
                    rv = ref.get(jn, float("nan")) if ref else float("nan")
                    bits.append(f"j{jn.split('joint_')[1][0]} cmd{c:+.3f} act{a:+.3f} "
                                f"ref{rv:+.3f} d{a - rv:+.3f}")
                print(f">>>   [{tag}] {f}: " + "  ".join(bits), flush=True)
            # Hand-frame relation -- the one number that says whether the hand is still where the
            # align put it.
            hp_, hR_ = self._hand_frame()
            objH_ = hR_.T @ (oc - hp_)
            rec3_ = getattr(self, "_rec_obj_in_grip_L3", None)
            print(f">>>   [{tag}] obj quat {np.round(np.asarray(_oq[0], float), 4).tolist()} "
                  f"axis_world {np.round(oR[:, 2], 3).tolist()}", flush=True)
            print(f">>>   [{tag}] objH(L3) {np.round(objH_, 4).tolist()}"
                  + (f" recorded {np.round(np.asarray(rec3_, float), 4).tolist()}"
                     f" drift {np.linalg.norm(objH_ - np.asarray(rec3_, float)) * 1000:.0f}mm"
                     if rec3_ is not None else ""), flush=True)
            # Hand orientation. `objH(L3)` is a POSITION only, and the align nulls position only, so
            # the same relation can be reached with the hand rotated differently every cycle --
            # which matters because the fingers ARC: rotating the hand rotates the arc, and b/c flip
            # from closing to opening across their j1 minimum.
            print(f">>>   [{tag}] handR x {np.round(hR_[:, 0], 3).tolist()} "
                  f"y {np.round(hR_[:, 1], 3).tolist()} z {np.round(hR_[:, 2], 3).tolist()}",
                  flush=True)
            pg = np.asarray(self._pinch(), float)
            print(f">>>   [{tag}] obj {np.round(oc, 3).tolist()} pinch {np.round(pg, 3).tolist()} "
                  f"pinch->obj {np.linalg.norm(oc[:2] - pg[:2]) * 1000:.0f}mm", flush=True)
        except Exception as e:
            print(f">>> [{tag}] phase report failed: {e}", flush=True)

    def _grasp_geometry_report(self, obj, tag=""):
        """Measure and print the grasp geometry: jaw opposition, mouth depth, knuckle throat and
        per-finger pad heights. Returns the thumb->bc jaw axis (or None if it could not be read).

        Pure measurement — it writes nothing and steps nothing. It used to live inside the legal
        close, so turning that stage off (which every wrap config does) also turned off the only
        numbers that say WHERE the hand is relative to the object. Diagnostics must not be a
        side effect of a stage that mutates.
        """
        if os.environ.get("GRASP_DIAG", "1") != "1":
            return None
        # Pose dump, so a probe can be run at the pose the RUN actually reaches.
        if os.environ.get("GRASP_DUMP_POSE", "0") == "1" and tag:
            try:
                _oc_dp = np.asarray(obj.get_world_poses()[0][0], float)
                _bx_dp, _by_dp, _byaw_dp = self.base_ledger()
                json.dump({"tag": tag,
                           "joints": np.asarray(self.robot.get_joint_positions(),
                                                float).tolist(),
                           "names": list(self.names),
                           "object_xyz": _oc_dp.tolist(),
                           "base_xy_yaw": [_bx_dp, _by_dp, _byaw_dp]},
                          open(os.path.join(HERE, f"usd/_pose_{tag}.json"), "w"))
                print(f">>>   {_t}pose dumped -> usd/_pose_{tag}.json", flush=True)
            except Exception as _e_dp:
                print(f">>>   {_t}pose dump failed ({_e_dp})", flush=True)
        jaw_axis = None
        _t = f"[{tag}] " if tag else ""
        # Do the pads actually OPPOSE? `pinch->obj` is a SCALAR and cannot answer that -- it is
        # large both when all pads sit on one side and when the jaw is legitimately open with the
        # thumb far and b/c near.
        try:
            _oc_d = np.asarray(obj.get_world_poses()[0][0], float)
            _pd = {}
            for _f in "abc":
                _pts = self._finger_world_pts(_f)
                if _pts is None or len(_pts) == 0:
                    continue
                _d = _pts - _oc_d
                _pd[_f] = _pts[int(np.argmin(np.linalg.norm(_d[:, :2], axis=1)))]
            if len(_pd) == 3:
                _bcm = 0.5 * (_pd["b"] + _pd["c"])
                _ax = (_bcm - _pd["a"])[:2]
                _ax = _ax / max(1e-9, float(np.linalg.norm(_ax)))
                _pr = {k: float((v[:2] - _oc_d[:2]) @ _ax) * 1000.0 for k, v in _pd.items()}
                _opp = (_pr["a"] < 0.0) and (_pr["b"] > 0.0 or _pr["c"] > 0.0)
                # How deep is the object in the mouth? The same projection, on the MOUTH axis.
                # Negative means the pad is BEHIND the object centre, i.e.
                try:
                    _pm = self._finger_world_pts("palm")
                    _mo3d = np.array([-_ax[1], _ax[0], 0.0])
                    _palm_f = float(np.max(_pm @ _mo3d))       # most forward palm point
                    _pad_f = float(np.mean([np.max(self._finger_world_pts(_f_) @ _mo3d)
                                            for _f_ in "abc"]))
                    _mdep = _pad_f - _palm_f
                    # A negative depth is not a small capacity, it is no cavity at all: the pads sit
                    # BEHIND the palm face, so nothing can be enclosed at this pose and the palm is
                    # necessarily the first surface the object meets.
                    if _mdep <= 0:
                        print(f">>>   {_t}MOUTH GEOMETRY: palm face to pad line = "
                              f"{_mdep * 1000:+.1f}mm -> NO CAVITY at this pose (pads are BEHIND "
                              f"the palm face); the palm meets the object first whatever its size "
                              f"(this object: {KNOWN['object_radius'] * 2000:.0f}mm dia)",
                              flush=True)
                    else:
                        print(f">>>   {_t}MOUTH GEOMETRY: palm face to pad line = "
                              f"{_mdep * 1000:+.1f}mm -> centreline grip needs radius <= that, "
                              f"i.e. max graspable diameter ~{_mdep * 2000:.0f}mm (this object: "
                              f"{KNOWN['object_radius'] * 2000:.0f}mm)", flush=True)
                except Exception as _e_mg:
                    print(f">>>   {_t}MOUTH GEOMETRY: failed ({_e_mg})", flush=True)
                # Knuckle throat: the narrow point behind the pads. `finger_*_link_1` tops the
                # contact ranking in EVERY run at EVERY object size, which is what a throat narrower
                # than the pad opening looks like.
                try:
                    from pxr import Usd as _U, UsdGeom as _UG, UsdPhysics as _UP
                    if not hasattr(self, "_k1"):
                        self._k1 = {}
                        for _f2 in "abc":
                            _pts2 = []
                            for _prim in self.stage.Traverse(_U.TraverseInstanceProxies()):
                                _pp = _prim.GetPath().pathString
                                if f"finger_{_f2}_link_1_1" not in _pp:
                                    continue
                                if not _prim.HasAPI(_UP.CollisionAPI):
                                    continue
                                if _prim.GetTypeName() != "Mesh":
                                    continue
                                _v2 = _UG.Mesh(_prim).GetPointsAttr().Get()
                                if _v2:
                                    _pts2.append((_prim, np.asarray(
                                        [[float(c) for c in q] for q in _v2], float)))
                            self._k1[_f2] = _pts2
                    _kw = {}
                    for _f2, _lst in self._k1.items():
                        _acc = []
                        for _prim, _loc in _lst:
                            _m2 = np.array(_UG.Xformable(_prim).ComputeLocalToWorldTransform(
                                _U.TimeCode.Default()), float)
                            _acc.append(_loc @ _m2[:3, :3] + _m2[3, :3])
                        if _acc:
                            _kw[_f2] = np.vstack(_acc)
                    if len(_kw) == 3:
                        _pa2 = _kw["a"] @ np.array([_ax[0], _ax[1], 0.0])
                        _pbc = np.concatenate([_kw["b"], _kw["c"]]) @ np.array(
                            [_ax[0], _ax[1], 0.0])
                        _throat = float(_pbc.min() - _pa2.max())
                        print(f">>>   {_t}KNUCKLE THROAT: link_1 span across the jaw = "
                              f"{_throat * 1000:+.1f}mm  vs object diameter "
                              f"{KNOWN['object_radius'] * 2000:.0f}mm -> "
                              f"{'OBJECT FITS' if _throat > KNOWN['object_radius'] * 2 else 'TOO NARROW — object cannot enter'}",
                              flush=True)
                except Exception as _e_k:
                    print(f">>>   {_t}KNUCKLE THROAT: failed ({_e_k})", flush=True)
                # Per-finger z -- the dimension the horizontal numbers collapse away. They cannot
                # tell "further out" from "riding ABOVE the object", so print the closest pad
                # vertex's height against the object's band.
                try:
                    _top = _oc_d[2] + self.obj_half_h
                    _bot = _oc_d[2] - self.obj_half_h
                    _zs = {}
                    for _f3 in "abc":
                        _p3 = self._finger_world_pts(_f3)
                        _n3 = _p3[int(np.argmin(np.linalg.norm(_p3 - _oc_d, axis=1)))]
                        _zs[_f3] = (round(float(_n3[2]), 3),
                                    "ABOVE TOP" if _n3[2] > _top else
                                    "below base" if _n3[2] < _bot else "on body")
                    print(f">>>   {_t}PAD HEIGHTS: object band {_bot:.3f}..{_top:.3f}  -> "
                          f"{ {k: v for k, v in _zs.items()} }", flush=True)
                except Exception as _e_z:
                    print(f">>>   {_t}PAD HEIGHTS: failed ({_e_z})", flush=True)
                # The object in the probe's own coordinates: (du, dv, dz), the offset from the PINCH
                # along the mouth axis, the jaw axis and z.
                try:
                    _pin_d = self._pinch()
                    _d_pin = _oc_d - _pin_d
                    print(f">>>   {_t}OBJECT vs PINCH (probe coords): du "
                          f"{float(_d_pin[:2] @ np.array([-_ax[1], _ax[0]])) * 1000:+.1f}mm  dv "
                          f"{float(_d_pin[:2] @ _ax) * 1000:+.1f}mm  "
                          f"dz {float(_d_pin[2]) * 1000:+.1f}mm", flush=True)
                except Exception as _e_pc:
                    print(f">>>   {_t}OBJECT vs PINCH: failed ({_e_pc})", flush=True)
                # Log the axis, not just the depth. The depth is measured against `_ax`, and the
                # hand rotates through the close, so a depth that moves run to run may be the AXIS
                # moving rather than the seating.
                _mo2 = np.array([-_ax[1], _ax[0]])
                _pm = {k: float((v[:2] - _oc_d[:2]) @ _mo2) * 1000.0 for k, v in _pd.items()}
                print(f">>>   {_t}MOUTH AXIS: [{_mo2[0]:+.4f} {_mo2[1]:+.4f}] "
                      f"(bearing {math.degrees(math.atan2(_mo2[1], _mo2[0])):+.1f}deg)", flush=True)
                print(f">>>   {_t}MOUTH DEPTH: pad offsets along the mouth axis (mm, +ve = pad is "
                      f"in FRONT of the object centre): { {k: round(v, 1) for k, v in _pm.items()} }",
                      flush=True)
                jaw_axis = _ax                    # kept: the legal close decomposes its squeeze drift on it
                # The object's (du, dv): its offset from the pinch along the mouth and jaw axes, so
                # a run can be compared against the reachable window directly.
                try:
                    _dd_d = self._du_dv(obj)          # ONE definition, shared with the enclose servo
                    if _dd_d is None:
                        raise ValueError("finger geometry unavailable")
                    _du_d, _dv_d = _dd_d
                    _in_win = (-0.13 <= _du_d <= -0.07) and (abs(_dv_d) <= 0.03)
                    print(f">>>   {_t}OBJECT (du, dv) vs the pinch: du {_du_d * 1000:+.1f}mm "
                          f"dv {_dv_d * 1000:+.1f}mm  -> "
                          f"{'INSIDE' if _in_win else 'OUTSIDE'} the reachable window "
                          f"(du -70..-130mm, |dv| <= 30mm)", flush=True)
                except Exception as _e_dd:
                    print(f">>>   {_t}(du, dv) readout failed ({_e_dd})", flush=True)
                # Which grip-frame axis is the mouth? hover-align can only offset along grip X
                # (`standoff`) and grip Z (HOVER_STANDOFF_Z) -- gripper +Y is vertical here -- so
                # name the axis before turning a knob.
                _gp_d, _gR_d = self._grip_frame()
                # ...and in the HAND frame. This is the one to keep: the hand frame rotates with the
                # wrist but NOT with the fingers, whereas a pad-derived axis moves as the fingers
                # close -- so a "constant" grip-frame vector measured at close points elsewhere when
                # the align, with the fingers wide, uses it.
                _hp_d, _hR_d = self._hand_frame()
                print(f">>>   {_t}mouth axis in HAND frame "
                      f"{np.round(_hR_d.T @ np.array([-_ax[1], _ax[0], 0.0]), 4).tolist()}  "
                      f"jaw axis in HAND frame "
                      f"{np.round(_hR_d.T @ np.array([_ax[0], _ax[1], 0.0]), 4).tolist()}",
                      flush=True)
                _ax3 = np.array([_ax[0], _ax[1], 0.0])
                _mo3 = np.array([-_ax[1], _ax[0], 0.0])
                print(f">>>   {_t}jaw axis in GRIP frame {np.round(_gR_d.T @ _ax3, 3).tolist()}  "
                      f"mouth axis in GRIP frame {np.round(_gR_d.T @ _mo3, 3).tolist()}",
                      flush=True)
                print(f">>> {_t}JAW GEOMETRY: pad offsets along the thumb->bc axis (mm, signed, "
                      f"object centre = 0): { {k: round(v, 1) for k, v in _pr.items()} }  "
                      f"-> {'OPPOSED (object IS in the jaw)' if _opp else 'SAME SIDE (no pinch available)'}",
                      flush=True)
        except Exception as _e_d:
            print(f">>> {_t}JAW GEOMETRY: measurement failed: {_e_d}", flush=True)
        return jaw_axis
