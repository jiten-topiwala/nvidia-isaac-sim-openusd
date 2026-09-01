"""Alignment servos: bring the hand to the object without moving the object.

The recorded-relation align, the height align, the fine align and the jog servo that
drives the arm in Cartesian space.

Split out of play_isaac.py; the method bodies are unchanged. Imports Isaac APIs at module
level, so it is only importable after SimulationApp exists — see `morph/__init__.py`.
"""
import math
import os

import numpy as np
from pxr import Usd

from morph.config import PERFECT, HEADLESS, MOUTH_VEC_HAND


class AlignMixin:
    """Mixed into Demo. Every `self.*` it touches is owned by Demo."""

    def _rec_target(self):
        """The object's target position in the gripper frame: the recorded relation plus the
        size correction.

        There is one definition of this, deliberately. With two — one applying the size
        correction and one aiming at the raw recorded relation — the hand gets servoed onto a
        relation recorded for a differently-sized object, which drives the palm shell into the
        object and then holds that seat for the whole carry.
        """
        rec = getattr(self, "_rec_obj_in_grip", None)
        if rec is None:
            return None
        # ALIGN_TO_PINCH targets where the fingers ACTUALLY close, not where the object was
        # recorded. The recorded relation sits a hand's width from the pinch centroid, so the close
        # grips air.
        if PERFECT and os.environ.get("ALIGN_TO_PINCH", "0") == "1":
            try:
                gp_p, gR_p = self._grip_frame()
                return gR_p.T @ (self._pinch() - gp_p)
            except Exception as _e_ap:
                print(f">>> align-to-pinch unavailable ({_e_ap}) -> recorded relation", flush=True)
        # Seat the object where the pads actually converge, in world xy rotated into the grip frame.
        _xy_off = np.zeros(3)
        _gxy = os.environ.get("GRASP_XY_OFF", "")
        if _gxy:
            try:
                _dx, _dy = (float(v) for v in _gxy.split(",")[:2])
                _, _gR_xy = self._grip_frame()
                _xy_off = _gR_xy.T @ np.array([_dx, _dy, 0.0], float)
            except Exception as _e_xy:
                print(f">>> GRASP_XY_OFF ignored ({_e_xy})", flush=True)
        return _xy_off + np.asarray(rec, float) + np.array([
            # The x offset is a genuine size correction: +x is the thumb side, and a few millimetres
            # balances this object against the one the grasp was recorded on.
            float(os.environ.get("ALIGN_OFF_X", "0.003")),
            0.0,
            0.0], float) + self._mouth_offset()

    def _mouth_depth(self, obj):
        """(mean pad depth in m, mouth-axis unit vector in world XY).

        Depth = how far the pads sit IN FRONT of the object's centre along the mouth axis.  At the
        recorded relation it measures {a +40.9, b +56.4, c +25.3}mm — the pads grip the near FACE of
        the cylinder, not its diameter, so the normals point outward and the squeeze ejects it
        (-20.1mm out of the mouth vs -2.7mm along the jaw axis).  Depth ~0 = contacts on the
        diameter = the forces oppose.

        Mouth axis comes from the HAND frame (Gripper_Link3_1), NOT from pad positions: the
        pad-derived axis moves as the fingers close, so it is not the same direction at align time
        (fingers wide) and close time (half closed)."""
        oc = np.asarray(obj.get_world_poses()[0][0], float)
        _, hR = self._hand_frame()
        m = (hR @ MOUTH_VEC_HAND)[:2]
        m = m / max(1e-9, float(np.linalg.norm(m)))
        d = []
        for f in "abc":
            pts = self._finger_world_pts(f)
            if pts is None or len(pts) == 0:
                continue
            p = pts[int(np.argmin(np.linalg.norm(pts - oc, axis=1)))]
            d.append(float((p[:2] - oc[:2]) @ m))
        return (float(np.mean(d)) if d else 0.0), m

    def _mouth_offset(self):
        """How much deeper into the jaw the target relation should sit.

        At the recorded relation all three pads sit tens of millimetres IN FRONT of the object's
        centre along the mouth axis — they touch the near face of the cylinder rather than its
        diameter, so their normals point outward and the squeeze launches the object out of the
        mouth instead of holding it. A depth near zero puts the contacts on the diameter, where an
        opposed grasp actually holds.

        It belongs HERE, in the ONE relation every align stage targets.  Applying it as a base
        shift inside hover-align did nothing: the later base-align re-seats the hand on the baked
        relation and simply undid it (measured, a 40mm insert left the depth at {42.2, 44.1, 40.2},
        unchanged).  Same lesson as ALIGN_OFF_X: correct the shared definition, not one caller.

        The vector is the mouth axis in GRIP coordinates AT CLOSE TIME (measured
        [0.542, 0.543, -0.641]); do not recompute it at hover, the hand rotates through the descent
        and a hover-frame vector points somewhere else by the time it matters."""
        # Descent-only insert. The insert has to be ON for the descent and OFF for the advance, and
        # it is the same shared relation for both, which is why any single value trades one against
        # the other: with no insert the fingers and palm plough the object during the descent, and
        # with it the hand ends far out and the fingers exhaust their travel with no contact.
        ins = float(getattr(self, "_mouth_ins", None)
                    if getattr(self, "_mouth_ins", None) is not None
                    else os.environ.get("MOUTH_INSERT", "0.0"))
        if abs(ins) < 1e-9:
            return np.zeros(3)
        # The vector is fixed in the HAND frame (Gripper_Link3_1), measured at close:
        #     mouth [0.2207, -0.177, -0.9591]   jaw [-0.9752, -0.054, -0.2144]
        # so mouth is roughly hand -Z and jaw roughly hand -X, nearly orthogonal. The hand frame
        # rotates with the wrist but NOT with the fingers, so this direction holds whether the
        # fingers are wide or half closed -- a vector fixed in the GRIP frame does not, and leaks
        # into the jaw axis. Convert live, so it is correct at whatever pose the caller is in.
        v = MOUTH_VEC_HAND
        try:
            _, gR = self._grip_frame()
            _, hR = self._hand_frame()
            return ins * (gR.T @ (hR @ v))
        except Exception:
            return np.zeros(3)

    def _align_height(self, obj, tag="align"):
        """Close the gripper-frame Y residual by moving BOTH columns.

        _align_recorded_relation corrects world XY only, so on its own it converges to a tiny
        "residual" while the hand is still far off VERTICALLY — measured in a friction run,
        `hover-align done: residual 0.7mm` with `objG [-0.174, 0.233, 0.326]` against a target of
        `[-0.099, -0.015, 0.173]`, i.e. 248mm out in gripper y.  Gripper +Y is very nearly
        world-DOWN on this arm, so that residual IS a height error, and both columns are prismatic
        and move the gripper 1:1 in z, making the correction direct.

        This lived inside _fine_align, which only PIN mode calls — the same one-fix-two-paths split
        that caused the _rec_target bug.  Shared now."""
        rec_h = self._rec_target()
        # Same frame rule as the xy align: `rec_h` follows REC_FRAME, so the frame it is measured
        # against has to follow it too.
        _rh_l3 = os.environ.get("REC_FRAME", "weld") != "weld"
        if rec_h is None or os.environ.get("ALIGN_HEIGHT", "1") != "1":
            return 0.0
        gp_, gR_ = self._hand_frame() if _rh_l3 else self._grip_frame()
        oc_ = np.asarray(obj.get_world_poses()[0][0], float)
        # GRIP_DROP, in metres, positive to finish that much LOWER than the recorded relation.
        drop = 0.0
        # Size-dynamic by default. A fixed GRIP_DROP is a world-z constant and so is correct for
        # exactly one object -- change the half-height and the grasp lands somewhere else on the
        # body.
        if PERFECT and os.environ.get("GRIP_DROP_AUTO", "1") == "1":
            try:
                _pz = float(np.mean([self._finger_world_pts(_f)[:, 2].mean() for _f in "abc"]))
                # clip: this corrects a grasp-height offset, it is not a licence to drive the
                # columns anywhere. One object half-height is the most that can ever be needed.
                drop = float(np.clip(_pz - float(oc_[2]), -self.obj_half_h, self.obj_half_h))
            except Exception as _e_gd:
                print(f">>> {tag}-height: auto grip-drop unavailable ({_e_gd}) -> "
                      f"using GRIP_DROP {drop * 1000:+.0f}mm", flush=True)
        dy = float((gR_.T @ (oc_ - gp_))[1] - (rec_h[1] - drop))   # <0 -> hand too low
        up = float(np.clip(-dy, -0.12, 0.12))
        if abs(up) <= 0.002:
            return 0.0
        cl_, cr_ = self.idx["ColumnLeftBearingJoint_1"], self.idx["ColumnRightBearingJoint_1"]
        # Hand pitch: the one orientation freedom reachable without re-baking. The columns are
        # prismatic and their DIFFERENCE sets the boom pitch, so moving them differentially tilts
        # the hand where moving them together only raises it.
        pitch = 0.0
        qh = np.asarray(self.robot.get_joint_positions(), float).copy()
        c0h = np.array([qh[cl_], qh[cr_]])
        bxh, byh, byawh = self.base_ledger()
        nh = int(0.6 / self.dt)
        # Stop lowering once the hand is on the object. This move runs to a large clamp with the
        # palm above the object, so on a big residual it presses the object straight down -- a clean
        # descent followed by the object driven down and toppled before the close has begun.
        fh_prev = 0.0
        _dn = up < 0
        _stop_h = os.environ.get("ALIGN_H_CONTACT_STOP", "1") == "1" and PERFECT and _dn
        _focus_h = getattr(self, "_focus_obj", None)
        if _stop_h:
            self._ctc_any = {}
        for ih in range(nh):
            if _stop_h and ih % 12 == 0 and ih:
                _hit = [k for k in self._ctc_any
                        if "floor" not in k and (_focus_h is None or _focus_h in k)]
                if _hit:
                    print(f">>> {tag}-height: STOPPED at {fh_prev * up * 1000:+.0f}mm of "
                          f"{up * 1000:+.0f}mm -- {_hit[0]} is in contact; lowering further "
                          f"presses the object into the floor", flush=True)
                    break
                self._ctc_any = {}
            fh = 0.5 - 0.5 * math.cos(math.pi * (ih + 1) / nh)
            fh_prev = fh
            qh[cl_], qh[cr_] = (c0h + fh * up).tolist()
            if pitch:
                qh[cr_] = float(qh[cr_] + fh * pitch)
                _dh = qh[cr_] - qh[cl_]                      # keep the 4-bar on its manifold
                qh[cr_] = float(min(max(qh[cr_], qh[cl_] + 0.0176), qh[cl_] + 0.20, 1.42))
            self.set_base(bxh, byh, byawh)
            self._force(qh)
            self._apply(qh)
            self.world.step(render=not HEADLESS)
        print(f">>> {tag}-height: gripper-y residual {dy * 1000:+.0f}mm -> columns "
              f"{up * 1000:+.0f}mm" + (f"  (GRIP_DROP {drop * 1000:+.0f}mm)" if drop else "")
              + (f"  (GRIP_PITCH {pitch * 1000:+.0f}mm differential -> dh "
                 f"{float(qh[cr_] - qh[cl_]):.4f})" if pitch else ""),
              flush=True)
        return up

    def _hand_touching_object(self, obj, steps=2):
        """Is any hand link in contact with the focus object RIGHT NOW?

        `close_anim`'s post-descent aligns justify themselves with "nothing is in contact to
        bulldoze -- that is what wrecked the earlier post-descent align (object dragged 322mm, sunk
        70mm)". That precondition was asserted, never tested, and it is FALSE for this object: the
        descent ledger ends `finger_b_link_1_1: 2686, finger_c_link_1_1: 2320, Gripper_Link3_1:
        1815` contacts, and the documented failure is exactly what we get -- the object leaves the
        stage toppled (`axis.z -0.11`) before the close begins.

        Only answerable since the hand links carry `PhysxContactReportAPI` (scene.py); before that
        every ledger here required a pickup_obj on one side and could not see a thing.
        """
        try:
            _focus = getattr(self, "_focus_obj", None)
            self._ctc_any = {}
            for _ in range(int(steps)):
                self.world.step(render=not HEADLESS)
            for _pair in self._ctc_any:
                if "floor" in _pair:
                    continue
                if _focus is None or _focus in _pair:
                    return _pair
        except Exception:
            return None
        return None

    def _align_recorded_relation(self, obj, iters=None, standoff=0.0):
        """Move the BASE so the object sits at the RECORDED gripper-frame relation of the working
        grasp.  Must be called AT HOVER, before the descent.

        Two hard-won constraints:
        1. NOT after the descent.  At grasp height the hand is already against the object, so every
           base move drags it: measured, six iterations converged the relation to 10mm but pushed the
           cylinder 322mm across the floor and 70mm INTO it (z 0.150 -> 0.080).  This is the same
           thing probe_frame_match hit with dock_correct, which it solved by teleporting the object —
           not available here, because in friction the object is a real body.
        2. WORLD space, no least-squares.  The baked descent's last 30 frames are a columns-only
           drop, i.e. a pure world-z translation, so the gripper's ORIENTATION and its horizontal
           offset to the object are both preserved through it.  Aligning the world-xy offset at
           hover therefore lands the grasp-height relation exactly, and the correction is a plain
           subtraction instead of a rank-deficient solve in a steeply rotated frame."""
        # Size-aware offset on the recorded relation; see `_rec_target`. The recording was formed on
        # a different-radius cylinder, so docking this object to the raw relation buries one finger
        # while the others sit centimetres out and cannot reach -- driving them further makes their
        # gaps worse, because they are already past their closest approach.
        rec = self._rec_target()
        if rec is None:
            return None
        # Lateral standoff for the hover align. At the recorded grasp one finger sits only
        # millimetres off the cylinder, so ANY vertical descent into that pose grazes it and shoves
        # the object far enough to fail the fine-align gate.
        sd_z = 0.0
        # Seat the object deeper in the mouth before the descent. The jaws are balanced and the
        # object IS between them, but it sits at the mouth ENTRANCE, so pressing ejects it forward
        # like pinching a marble at the fingertips.
        if os.environ.get("ALIGN_CENTRE", "0") == "1":
            gp0, gR0 = self._grip_frame()
            rec = gR0.T @ (self._pinch() - gp0)
            print(f">>> hover-align: CENTRING the mouth on the object — target {np.round(rec, 4).tolist()} "
                  f"(recorded relation was {np.round(self._rec_target(), 4).tolist()})", flush=True)
        rec = rec + np.array([standoff, 0.0, sd_z])
        # 4 iterations was not enough: the grasp-height align kept hitting the cap at 16-24mm
        # residual, and the probe shows finger b sits 1.0mm off the surface at the true relation —
        # so a 20mm residual is the whole difference between a grip and zero contacts.
        iters = 24 if iters is None else iters
        tol = 0.002
        # Servo in the frame the target was recorded in. `_rec_target` returns whichever relation
        # REC_FRAME selected -- `obj_in_weldbody` is Link1, `obj_in_gripper` is Link3 -- so
        # measuring against a fixed frame here would feed one array into the other's servo, which
        # trades one error for another of the same size.
        _rec_l3 = os.environ.get("REC_FRAME", "weld") != "weld"
        _frame = self._hand_frame if _rec_l3 else self._grip_frame
        if _rec_l3:
            print(f">>> hover-align: servoing the LINK3 (hand) frame to match the "
                  f"obj_in_gripper target", flush=True)
        # Divergence guard. This servo moves the base by the full world error each iteration while
        # re-reading the object's LIVE position, so once the hand touches the object the loop is
        # positive feedback: the base moves, the hand shoves the object, the error grows.
        _best_e, _grew = None, 0
        for it in range(iters):
            gp, gR = _frame()
            op = np.asarray(obj.get_world_poses()[0][0], float)
            want = gR @ rec                                  # world offset gripper -> object
            err = (op - gp) - want                           # world; only xy is correctable
            _e_now = float(np.linalg.norm(err[:2]))
            if _e_now < tol:
                break
            if _best_e is not None and _e_now > _best_e + 0.002:
                _grew += 1
                if _grew >= 2:
                    print(f">>> hover-align: DIVERGING ({_best_e * 1000:.1f}mm -> "
                          f"{_e_now * 1000:.1f}mm over {_grew} iters) -- the hand is pushing the "
                          f"object it is aligning to. Stopping here.", flush=True)
                    break
            else:
                _grew = 0
            _best_e = _e_now if _best_e is None else min(_best_e, _e_now)
            # Step size is per-actuator. A step sized for a BASE move is far too large for the arm's
            # Cartesian jog -- handed a target that big it diverges rather than saturating.
            _cap = (0.05
                    if (os.environ.get("NO_CHASSIS_APPROACH", "1") == "1" and PERFECT) else 0.30)
            d = np.clip(err[:2], -_cap, _cap)
            n = int(0.5 / self.dt)
            q = np.asarray(self.robot.get_joint_positions(), float).copy()
            # Arm, not chassis. Once the base is docked it should not move again, and moving it here
            # is self-defeating anyway: with the hand touching, a base move drags the OBJECT and the
            # servo chases it.
            _arm_only = os.environ.get("NO_CHASSIS_APPROACH", "0") == "1" and PERFECT
            if _arm_only and os.environ.get("ALIGN_ARM_JOG", "0") != "1":
                print(f">>> hover-align[{it}] world err {np.linalg.norm(err[:2]) * 1000:.1f}mm "
                      f"-> SKIPPED (chassis parked at the dock; the enclose joint servo corrects "
                      f"this arm-only downstream)", flush=True)
                break
            if _arm_only:
                self.start_jog()
                _g0 = self.grip_pos()
                for i in range(n):
                    f = 0.5 - 0.5 * math.cos(math.pi * (i + 1) / n)
                    self.jog["target"] = np.array([_g0[0] + f * d[0], _g0[1] + f * d[1], _g0[2]])
                    self._jog_tick()
                    self.world.step(render=not HEADLESS)
                self.jog = None
                _moved = float(np.linalg.norm((self.grip_pos() - _g0)[:2]))
                # Sanity bound on the jog itself: even correctly sized, an IK jog can diverge, and
                # violently.
                if _moved > max(0.05, 4.0 * float(np.linalg.norm(d))):
                    print(f">>> hover-align: JOG DIVERGED (asked {np.linalg.norm(d) * 1000:.0f}mm, "
                          f"hand moved {_moved * 1000:.0f}mm) -- restoring and stopping", flush=True)
                    self._force(q)
                    self._apply(q)
                    for _ in range(int(0.1 / self.dt)):
                        self.world.step(render=not HEADLESS)
                    break
                print(f">>> hover-align[{it}] world err {np.linalg.norm(err[:2]) * 1000:.1f}mm "
                      f"-> ARM d {np.round(d, 4).tolist()} (hand moved {_moved * 1000:.1f}mm, "
                      f"chassis untouched)", flush=True)
            else:
                bx, by, byaw = self.base_ledger()
                for i in range(n):
                    f = 0.5 - 0.5 * math.cos(math.pi * (i + 1) / n)
                    self.set_base(float(bx + f * d[0]), float(by + f * d[1]), byaw)
                    self._force(q)
                    self._apply(q)
                    self.world.step(render=not HEADLESS)
                print(f">>> hover-align[{it}] world err {np.linalg.norm(err[:2]) * 1000:.1f}mm "
                      f"-> base d {np.round(d, 4).tolist()}", flush=True)
        gp, gR = _frame()
        op = np.asarray(obj.get_world_poses()[0][0], float)
        err = (op - gp) - gR @ rec
        # Print the HAND-frame relation next to the Link1 one. They describe the same placement only
        # when the wrist matches the recording, and it is the Link3 relation that decides where the
        # fingers are, since they are parented to it -- Link1 can converge to a fraction of a
        # millimetre with the fingers still centimetres off.
        hp, hR = self._hand_frame()
        raw3 = getattr(self, "_rec_obj_in_grip_L3", None)
        print(f">>> hover-align done: residual {np.linalg.norm(err[:2]) * 1000:.1f}mm  "
              f"objG {np.round(gR.T @ (op - gp), 4).tolist()} target {np.round(rec, 4).tolist()}",
              flush=True)
        print(f">>>   hand-frame(L3) objH {np.round(hR.T @ (op - hp), 4).tolist()}"
              + (f" recorded {np.round(np.asarray(raw3, float), 4).tolist()}" if raw3 is not None else ""),
              flush=True)
        return err

    def _finger_centroid_xy(self, oc):
        """World xy centroid of the three fingers' closest points to the object.

        Size-adaptive replacement for the recorded gripper-frame offset. The recording was formed
        on a smaller cylinder, and docking a larger object to that raw offset leaves the thumb
        hanging over the top face while b and c are on the wall — from there the hand cannot be
        lowered without burying the thumb, so no drop depth works. Putting the object's axis at the
        centroid of the three fingers straddles it by construction, whatever the radius.
        """
        from pxr import Usd, UsdGeom
        pts = []
        for f in "abc":
            best, bp = 1e9, None
            for prim, loc in self._finger_hulls().get(f, ()):
                m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()), float)
                a = loc @ m[:3, :3] + m[3, :3]
                d = np.hypot(a[:, 0] - oc[0], a[:, 1] - oc[1])
                j = int(np.argmin(d))
                if float(d[j]) < best:
                    best, bp = float(d[j]), a[j, :2].copy()
            if bp is not None:
                pts.append(bp)
        return np.mean(np.asarray(pts, float), axis=0) if pts else None

    def _capture_grip_offset(self, obj):
        """Record the object's position IN the gripper frame right now: local = Rᵀ·(obj − grip).

        The offset is small — the object is already seated between the fingers — so the hold that
        uses it is a precision constraint, not a transport.
        """
        gp, gR = self._hand_frame()          # HAND body (fingers' parent), not the wrist-side Link1
        op, oq = obj.get_world_poses()
        op = np.asarray(op[0], float)
        self._obj_local = gR.T @ (op - gp)
        # Record the orientation too. A rigid position hold with a frozen orientation is not a rigid
        # body: held axis-aligned while the hand tilts through the lift, the fingers rotate away
        # from the object and the gaps open by centimetres with the finger joints completely
        # unchanged.
        self._obj_local_R = gR.T @ self._quat_to_R(np.asarray(oq[0], float))

    def _tilt_deg(self, obj):
        """Object lean off world-vertical, degrees, or None if unreadable. Free -- one pose read."""
        R = self._obj_R(obj)
        if R is None:
            return None
        return math.degrees(math.acos(float(np.clip(R[:, 2][2], -1.0, 1.0))))

    def _level_gradient(self, obj, joint="HandBearingJoint_1", dp=0.05):
        """d(tilt)/d(joint) in deg/rad, probed once. The probe is trustworthy because the arm is
        kinematically forced (same argument as _level_wrist). Returns (index, gradient) or None."""
        i = self.idx.get(joint)
        t0 = self._tilt_deg(obj)
        if i is None or t0 is None:
            return None
        q0 = np.asarray(self.robot.get_joint_positions(), float).copy()
        qp = q0.copy()
        qp[i] = float(q0[i]) + dp
        # render=False: the probe physically jerks the wrist for two steps, and drawing those frames
        # shows as a twitch between chunks of the carry-level ramp. Measure invisibly.
        self._force(qp); self._apply(qp); self.world.step(render=False)
        tp = self._tilt_deg(obj)
        self._force(q0); self._apply(q0); self.world.step(render=False)
        if tp is None:
            return None
        g = (tp - t0) / dp
        return (i, g) if abs(g) > 5.0 else None

    def _level_wrist(self, obj, tag="level", tol_deg=2.0, max_delta=0.7, secs=0.35):
        """Bring the carried object back to vertical with ONE wrist DOF.

        The pick delivers the object level, and then every stage that changes the boom geometry
        tilts it, because the wrist is otherwise passive: it replays what the bake recorded and
        nothing corrects it. Left alone the tilt accumulates across the lift, the carry, the dock
        and the insert — the last of which tilts the boom to buy reach — so the object is carried
        and inserted leaning, and only the release snap makes it upright again.

        The correction has three rules:

          - ONE DOF, never several. Multi-DOF corrections on this arm are exactly what diverged
            in the insert servo (twice) and in the Cartesian jog before it.
          - Gradient by probe, not by assumption. Which wrist joint pitches the hand depends on the
            arm's pose, so measure d(tilt)/d(joint) instead of hardcoding a joint.
          - Refuse to act without authority: if no DOF moves the tilt meaningfully, say so and
            leave the pose alone rather than spending a large delta on a weak axis.

        The probe is safe to run live here because the arm is KINEMATICALLY FORCED: `_force`
        writes the pose exactly, so a perturbed step reads back the perturbed pose with no settle
        transient to contaminate the gradient.
        """
        if os.environ.get("WRIST_LEVEL", "1") != "1":
            return None
        names = ("HandBearingJoint_1", "gripper_y_rotation_1", "gripper_x_rotation_1")
        ii = [(n, self.idx[n]) for n in names if n in self.idx]
        if not ii:
            return None

        t0 = self._tilt_deg(obj)
        if t0 is None or t0 <= tol_deg:
            return t0
        q0 = np.asarray(self.robot.get_joint_positions(), float).copy()
        dp = 0.05
        best = None
        for n, i in ii:
            qp = q0.copy()
            qp[i] = float(q0[i]) + dp
            self._force(qp)
            self._apply(qp)
            self.world.step(render=False)                # probes are measurements -- never drawn
            tp = self._tilt_deg(obj)
            self._force(q0)                              # restore before scoring the next one
            self._apply(q0)
            self.world.step(render=False)
            if tp is None:
                continue
            g = (tp - t0) / dp                           # deg per rad
            if abs(g) < 20.0:
                continue                                 # too weak to be worth spending delta on
            if best is None or abs(g) > abs(best[2]):
                best = (n, i, g)
        if best is None:
            print(f">>> wrist-level[{tag}]: tilt {t0:.1f}deg but no wrist DOF has authority "
                  f"(all gradients under {os.environ.get('WRIST_MIN_GRAD', '20.0')} deg/rad) "
                  f"-- leaving the pose alone", flush=True)
            return t0
        n, i, g = best
        d = float(np.clip(-t0 / g, -max_delta, max_delta))
        qt = q0.copy()
        qt[i] = float(q0[i]) + d
        nstep = max(1, int(secs / self.dt))
        for k in range(nstep):
            f = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / nstep)
            qk = q0 + f * (qt - q0)
            self._force(qk)
            self._apply(qk)
            self.world.step(render=not HEADLESS)
        t1 = self._tilt_deg(obj)
        print(f">>> wrist-level[{tag}]: {t0:.1f} -> {t1:.1f}deg off-vertical via {n} "
              f"({d:+.3f} rad, gradient {g:+.0f} deg/rad)", flush=True)
        return t1

    def _carry_check(self, obj, tag):
        """How far the object has SLIPPED in the hand since capture, in the hand's own frame.

        The nearby drift readouts measure the offset from the RECORDED grasp relation, which is a
        constant of the grasp geometry and reads identically whether the hold is perfect or
        absent. They cannot answer "did the object move in the hand", which is the only question
        the carry cares about. This compares the LIVE hand-frame offset against the one
        `_capture_grip_offset` froze, so zero means the object has not moved relative to the
        fingers, whatever the hand itself did.
        """
        try:
            if self._obj_local is None:
                return None
            gp, gR = self._hand_frame()
            op = np.asarray(obj.get_world_poses()[0][0], float)
            slip = float(np.linalg.norm(gR.T @ (op - gp) - self._obj_local)) * 1000.0
            # Tilt too, not just position: a cylinder that stays put in the hand but rotates is
            # still a failed carry, because it cannot be stood upright on the shelf.
            up = rs = float("nan")
            R = self._obj_R(obj)
            if R is not None:
                up = math.degrees(math.acos(float(np.clip(R[:, 2][2], -1.0, 1.0))))
                if getattr(self, "_obj_local_R", None) is not None:
                    dR = (gR.T @ R) @ np.asarray(self._obj_local_R, float).T
                    rs = math.degrees(math.acos(float(np.clip(
                        (np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))))
            print(f">>> carry[{tag}]: object slip in hand {slip:.1f}mm  upright {up:.1f}deg "
                  f"off-vertical  rot-slip {rs:.1f}deg "
                  f"({'weld' if getattr(self, '_weld_path', None) else 'pose pin'})"
                  + ("   <-- TILTED" if up == up and up > 5.0 else ""), flush=True)
            return slip
        except Exception as _e_cc:
            print(f">>> carry[{tag}]: slip readout failed ({_e_cc})", flush=True)
            return None

    def _weld_grip(self, obj_idx):
        """Hold the carried object with a real PhysX constraint rather than a per-step pose write.

        A pose write is strictly weaker: it happens BETWEEN steps, so it is always one step of hand
        travel stale, and nothing the solver does can correct that lag — which reads as the object
        shifting inside the fingers during transport. A `UsdPhysics.FixedJoint` marked
        `excludeFromArticulation` lets the solver enforce the hold instead.

        The source model ships grasp welds of exactly this shape, and `scene.py` deletes them at
        load only because the converter imports them ACTIVE and snaps every object onto the gripper
        at frame 0. Creating one on demand is the same mechanism, switched on at the right moment.

        Pin mode's objects are already gravity-free and collision-free, so no gravcomp analogue is
        needed -- the weld only has to supply the attachment.
        """
        if PERFECT or os.environ.get("GRIP_WELD", "1") != "1":
            return False
        from pxr import Gf, Sdf, UsdPhysics
        self._unweld_grip()
        from morph.usd_utils import find_path
        # the body _hand_frame reads and the fingers are children of -- welding to Link1 instead
        # slides the object when the wrist turns, which is the bug kinematics.py:50 already
        # documents for the pose pin.
        hand = find_path(self.stage, "Gripper_Link3_1")
        objp = find_path(self.stage, f"pickup_obj_{obj_idx}")
        if hand is None or objp is None:
            print(">>> weld: hand or object prim not found -- falling back to the pose pin", flush=True)
            return False
        gp, gR = self._hand_frame()
        op, oq = self.objs[obj_idx].get_world_poses()
        loc = gR.T @ (np.asarray(op[0], float) - gp)          # object origin in the hand frame
        path = Sdf.Path(str(hand) + "/GraspWeld")
        j = UsdPhysics.FixedJoint.Define(self.stage, path)
        j.CreateBody0Rel().SetTargets([Sdf.Path(str(hand))])
        j.CreateBody1Rel().SetTargets([Sdf.Path(str(objp))])
        # The orientation must be the CAPTURED one, not identity. Identity local rotations do not
        # mean "hold the current relative pose" -- they constrain the object's axes to COINCIDE with
        # the hand's, so the solver rotates the object to get there the instant the joint engages,
        # and it rides the whole transport on its side while position slip still reads zero.
        from morph.geometry import R_to_quat
        _rq = R_to_quat(gR.T @ self._quat_to_R(np.asarray(oq[0], float)))
        j.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in loc]))
        j.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        j.CreateLocalRot0Attr().Set(Gf.Quatf(float(_rq[0]), Gf.Vec3f(float(_rq[1]),
                                                                     float(_rq[2]),
                                                                     float(_rq[3]))))
        j.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        j.CreateExcludeFromArticulationAttr().Set(True)       # maximal-coordinate constraint
        j.CreateCollisionEnabledAttr().Set(False)
        self._weld_path = path
        print(f">>> weld: FixedJoint hand<->pickup_obj_{obj_idx} at local "
              f"{np.round(loc, 4).tolist()} (excludeFromArticulation; GRIP_WELD=0 for the pose pin)",
              flush=True)
        return True

    def _unweld_grip(self):
        """Drop the grasp weld at release."""
        p = getattr(self, "_weld_path", None)
        if p is None:
            return
        try:
            self.stage.RemovePrim(p)
        except Exception as _e_uw:
            print(f">>> weld: remove failed ({_e_uw})", flush=True)
        self._weld_path = None

    def _pin_grip(self, obj):
        """Rigid gripper-frame pin: obj = grip_pos + R·local, rebuilt from the LIVE gripper every step, so
        the object rides WITH the hand (between the fingers) wherever the hand goes — never floats in the air.
        PERFECT mode: no-op — pad contact + friction carries the object, nothing else.
        WELD mode: also a no-op — the FixedJoint holds it and a pose write would fight the solver."""
        if getattr(self, "_weld_path", None) is not None:
            return
        if PERFECT:
            return
        gp, gR = self._hand_frame()          # must match the frame _capture_grip_offset used
        t = gp + gR @ self._obj_local
        rq = getattr(self, "_obj_local_R", None)
        oq = self._R_to_quat(gR @ rq) if rq is not None else np.array([1.0, 0, 0, 0])
        obj.set_world_poses(positions=np.array([t]), orientations=np.array([oq]))
        obj.set_velocities(np.zeros((1, 6)))

    def _fine_align(self, obj, park, secs=3.0, tol=0.008, aim_fwd=None, lock_th=False):
        """Drive the pinch centroid onto the object centre with the closed-loop cartesian jog,
        writing the LUT closure passives every tick.

        The HAND comes to the OBJECT; the object never moves. XY only — the open-finger pinch sits
        at palm height, so servoing its z onto the object's centre drags the whole palm down inside
        the object. The baked descent has already set the correct grasp height; hold it.

        Returns the final pinch-to-object xy error, in metres.
        """
        self.start_jog()
        if lock_th:
            self.jog["lock_th"] = True
        e = 1e9
        for _ in range(int(secs / self.dt)):
            # re-read the object EVERY tick: in perfect mode it is a real body — if the hand nudges
            # it, the servo must chase the live position, not a stale snapshot
            oc = np.asarray(obj.get_world_poses()[0][0], float)
            # Target the RECORDED relation, not the pinch centroid on the object's axis.
            rec_ = self._rec_target()
            if rec_ is not None and os.environ.get("ALIGN_RECORDED", "1") == "1":
                gp_, gR_ = self._grip_frame()
                d = oc - (gp_ + gR_ @ rec_)
                e = float(np.linalg.norm(d[:2]))
            else:
                d = oc - self._pinch()
                e = float(np.linalg.norm(d[:2]))
            if e < tol:
                break
            d[2] = 0.0
            if PERFECT and aim_fwd:
                # aim_fwd < 0 aims BEHIND the object along the TRUE a1 slide direction, taken by
                # finite difference.
                qj_ = self.jog["q"]
                f0_ = self._grip_fk(*qj_)
                qq_ = qj_.copy()
                qq_[2] += 1e-3
                fwd = (self._grip_fk(*qq_) - f0_)[:2]
                nf = float(np.linalg.norm(fwd))
                if nf > 1e-9:
                    d[:2] += fwd / nf * aim_fwd * 1e-3 / 1e-3
            self.jog["target"] = self.grip_pos() + d
            if not PERFECT:                                 # friction: it rests there by gravity
                obj.set_world_poses(positions=np.array([park]), orientations=np.array([[1, 0, 0, 0]]))
                obj.set_velocities(np.zeros((1, 6)))
            self._jog_tick()
        self.jog = None
        # Height term. `_fine_align` servos world XY only, so it cannot close the gripper-frame Y
        # residual -- and gripper +Y points nearly straight down here, so that residual IS a height
        # error.
        self._align_height(obj)
        gp_, gR_ = self._grip_frame()
        print(f">>> fine-align: err {e * 1000:.1f}mm  obj {np.round(oc, 3).tolist()} "
              f"objG {np.round(gR_.T @ (oc - gp_), 3).tolist()} "
              f"target {np.round(self._rec_target() if self._rec_target() is not None else np.zeros(3), 3).tolist()}",
              flush=True)
        return e

    def start_jog(self):
        """Begin cartesian jog from the CURRENT arm pose; target starts at the live gripper."""
        q = np.asarray(self.robot.get_joint_positions(), float)
        j = lambda n: float(q[self.idx[n]])
        self.jog = {"anchor": self.base_pose(),
                    "q": np.array([j("ColumnLeftBearingJoint_1"), j("ColumnRightBearingJoint_1"),
                                   j("ArmLeftJoint_1"), j("BaseJoint_1")]),
                    "origin": self.grip_pos().copy(),
                    # FIXED finger reference: rebuilding the command from MEASURED positions each
                    # tick let un-anchored fingers integrate a slow drift CLOSED during long jogs
                    # (the measured-feedback walk, same family as the idle base drift) — by the
                    # descent's end they had curled to +0.31 rad on their own
                    "fhold": q[self._f_idx].copy()}
        self.jog["target"] = self.jog["origin"].copy()
        pred = self._grip_fk(*self.jog["q"])
        print(f">>> jog start: measured {np.round(self.jog['origin'], 3).tolist()} "
              f"model {np.round(pred, 3).tolist()} "
              f"fkerr {np.linalg.norm(pred - self.jog['origin']) * 1000:.0f}mm", flush=True)

    def _jog_tick(self):
        """One kinematic servo tick toward jog['target'] (world).  Returns |error| in metres."""
        jg = self.jog
        err = np.asarray(jg["target"], float) - self.grip_pos()
        q = jg["q"].copy()
        f0 = self._grip_fk(*q)
        J = np.zeros((3, 4))
        for i in range(4):
            qq = q.copy()
            qq[i] += 1e-4
            J[:, i] = (self._grip_fk(*qq) - f0) / 1e-4
        if float(np.linalg.norm(f0 - self.grip_pos())) > 0.4:
            print(">>> jog: model/reality diverged >0.4m -> jog frozen (toggle jog off/on to reset)",
                  flush=True)
            jg["target"] = self.grip_pos()                   # hold in place; never chase a diverged model
            err = np.zeros(3)
        Jp = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), np.eye(3))
        # redundancy resolution: 4 joints for a 3D target — without a posture bias the nullspace
        # drifts the columns to their rails (h1 rose 1.26m for a 15cm x-move, then h2 exceeded its
        # 1.43 joint LIMIT and the limit-vs-kinematic-write fight went NaN) comfortable posture near
        # the box-clearing grasp (columns ~0.3, boom extended ~0.6); th free.
        QREF = np.array([0.30, 0.37, 0.55, q[3]])
        dq_task = np.clip(Jp @ err, -0.004, 0.004)           # rate limit ~0.24 m/s (the EMA-smooth analog)
        # posture bias clipped SEPARATELY and much smaller — sharing one clip let the bias saturate
        # it and cancel the task step (z stalled 0.3m low whenever h1 sat above the reference)
        dq_null = np.clip(0.02 * (np.eye(4) - Jp @ J) @ (QREF - q), -0.001, 0.001)
        if jg.get("lock_th"):
            # Cage lock: th rotation re-aims the whole finger cage (18.5deg swing observed when the
            # DLS used yaw to chase a radial target) — radial targets are a1's job
            dq_task[3] = 0.0
            dq_null[3] = 0.0
        dh_prev = q[1] - q[0]
        q += dq_task + dq_null
        q[0] = min(max(q[0], 0.0), 1.38)                     # h1 range: 1.43 joint limit minus pitch floor
        # Boom pitch floor (10 degrees), approached by RAMP: an instant clamp from the park pose
        # jumps the pitch through the whole floor in one step and destabilises the first tick.
        dh_floor = min(0.0176, max(dh_prev, 0.0) + 0.002)
        q[1] = min(max(q[1], q[0] + dh_floor), q[0] + 0.20)  # pitch cap 63°
        q[1] = min(q[1], 1.42)                               # NEVER beyond the 1.43 joint limit (NaN source)
        q[2] = min(max(q[2], 0.0), 0.625)                    # a1 boom range
        if self._grip_fk(*q)[2] < 0.12:
            # Floor guard: never command the hand below the floor -- a saturated-pitch transient
            # drove it under and the kinematic-vs-floor fight
            q = jg["q"]
            # NaN'd
        jg["q"] = q
        self._jog_write(q)
        return float(np.linalg.norm(err))

    def _jog_write(self, q):
        """Write a jog 4-vector [h1, h2, a1, th] to the robot: LUT closure passives + kinematic
        force + drive, base held at the jog anchor.  One physics step."""
        full = np.asarray(self.robot.get_joint_positions(), float)
        # `_finger_cmd` outranks `fhold`. `fhold` is a jog-start snapshot, which exists because
        # unanchored fingers integrate a slow drift closed during long jogs -- but it must not
        # outlive the grip that owns them: a `start_jog()` taken while the fingers are wide open
        # would otherwise restore that open vector over the close's grip.
        full[self._f_idx] = (self._finger_cmd if (PERFECT and self._finger_cmd is not None)
                             else self.jog["fhold"])
        vals = {"ColumnLeftBearingJoint_1": q[0], "ColumnRightBearingJoint_1": q[1],
                "ArmLeftJoint_1": q[2], "BaseJoint_1": q[3]}
        vals.update(self._closure_passives(q[1] - q[0], q[2]))
        for n, v in vals.items():
            if n in self.idx:
                full[self.idx[n]] = v
        bx, by, byaw = self.jog["anchor"]
        self.set_base(bx, by, byaw)
        self._force(full)
        self._apply(full)
        self.world.step(render=not HEADLESS)
