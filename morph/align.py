"""Alignment servos: bring the hand to the object without moving the object.
Imports Isaac APIs at module level, so importable only after SimulationApp exists.
"""
import math

import numpy as np

from morph.config import HEADLESS, KNOWN, MOUTH_VEC_HAND
from morph.geometry import wrap


# Chassis speed and turn rate the joystick commands, m/s and rad/s. A jog carries nothing, so
# neither is held to the wheel-drive legs' own caps.
BASE_JOG_MPS = 0.6
BASE_JOG_RADPS = 0.6
# Working boom pitch the jog keeps once the arm is above it, radians of column differential.
JOG_DH_FLOOR = 0.0176


class AlignMixin:
    """Mixed into Demo. Every `self.*` it touches is owned by Demo."""

    def _mouth_depth(self, obj):
        """(mean pad depth in m, mouth-axis unit vector in world XY). Depth = how far the pads
        sit in front of the object's centre along the mouth axis; ~0 puts the contacts on the
        diameter, where the squeeze opposes instead of ejecting. The axis comes from the HAND
        frame, not pad positions, which move as the fingers close."""
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

    def _capture_grip_offset(self, obj):
        """Freeze the object's pose in the HAND frame (local = Rᵀ·(obj − hand)) into
        `_obj_local` / `_obj_local_R`, which `_carry_check` measures slip against."""
        gp, gR = self._hand_frame()          # HAND body (fingers' parent), not the wrist-side Link1
        op, oq = obj.get_world_poses()
        op = np.asarray(op[0], float)
        self._obj_local = gR.T @ (op - gp)
        # Orientation too: a position-only hold lets the fingers rotate off the object on a tilt.
        self._obj_local_R = gR.T @ self._quat_to_R(np.asarray(oq[0], float))
        # The box a planner would attach to the hand, sized by the GRASP attitude. Read the tilt
        # against the hand's own axes, not against world-up: the two differ by the carry pose.
        axis = self._obj_local_R[:, 2]
        if not (np.all(np.isfinite(axis)) and np.all(np.isfinite(self._obj_local))):
            # `min(1.0, nan)` is 1.0, so without this the line reports a confident 0.0deg.
            print(">>> grip offset: attitude unavailable -- a non-finite hand or object pose",
                  flush=True)
            return
        half = self.obj_half_h * np.abs(axis) + KNOWN["object_radius"] * np.sqrt(
            np.clip(1.0 - axis * axis, 0.0, 1.0))
        print(f">>> grip offset: object axis {np.round(axis, 4).tolist()} in the hand, "
              f"{math.degrees(math.acos(min(1.0, float(np.abs(axis).max())))):.1f}deg off the "
              f"nearest hand axis; held-box half-extents "
              f"{np.round(half * 1000, 1).tolist()}mm about centre "
              f"{np.round(self._obj_local * 1000, 1).tolist()}mm", flush=True)

    def _tilt_deg(self, obj):
        """Object lean off world-vertical in degrees, or None if unreadable. One pose read."""
        R = self._obj_R(obj)
        if R is None:
            return None
        return math.degrees(math.acos(float(np.clip(R[:, 2][2], -1.0, 1.0))))

    def _carry_check(self, obj, tag):
        """Object slip in the hand since `_capture_grip_offset`, in mm, or None. Compares the
        LIVE hand-frame offset to the frozen one, so zero means no movement relative to the
        fingers -- unlike the drift readouts, which read the same with no hold at all."""
        try:
            if self._obj_local is None:
                return None
            gp, gR = self._hand_frame()
            op = np.asarray(obj.get_world_poses()[0][0], float)
            slip = float(np.linalg.norm(gR.T @ (op - gp) - self._obj_local)) * 1000.0
            # Tilt too: a cylinder that rotates in place cannot be stood upright on the shelf.
            up = rs = float("nan")
            R = self._obj_R(obj)
            if R is not None:
                up = math.degrees(math.acos(float(np.clip(R[:, 2][2], -1.0, 1.0))))
                if getattr(self, "_obj_local_R", None) is not None:
                    dR = (gR.T @ R) @ np.asarray(self._obj_local_R, float).T
                    rs = math.degrees(math.acos(float(np.clip(
                        (np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))))
            self._bench_slip = getattr(self, "_bench_slip", []) + [
                (tag, round(slip, 2), None if up != up else round(up, 2), None if rs != rs else round(rs, 2))]
            print(f">>> carry[{tag}]: object slip in hand {slip:.1f}mm  upright {up:.1f}deg "
                  f"off-vertical  rot-slip {rs:.1f}deg "
                  f"(friction only)"
                  + ("   <-- TILTED" if up == up and up > 5.0 else ""), flush=True)
            return slip
        except Exception as _e_cc:
            print(f">>> carry[{tag}]: slip readout failed ({_e_cc})", flush=True)
            return None




    def start_jog(self):
        """Begin cartesian jog from the CURRENT arm pose; target starts at the live gripper."""
        q = np.asarray(self.robot.get_joint_positions(), float)
        j = lambda n: float(q[self.idx[n]])
        self.jog = {"anchor": self.base_pose(),
                    "q": np.array([j("ColumnLeftBearingJoint_1"), j("ColumnRightBearingJoint_1"),
                                   j("ArmLeftJoint_1"), j("BaseJoint_1")]),
                    "origin": self.grip_pos().copy(),
                    # FIXED reference, and for the WHOLE robot: rebuilding from MEASURED positions
                    # each tick makes every joint the jog does not command chase its own sag.
                    "qhold": q.copy(),
                    "fhold": q[self._f_idx].copy()}
        self.jog["qref"] = self.jog["q"].copy()
        self.jog["q0dh"] = float(self.jog["q"][1] - self.jog["q"][0])
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
            # hold in place; never chase a diverged model
            jg["target"] = self.grip_pos()
            err = np.zeros(3)
        Jp = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), np.eye(3))
        # Redundancy: 4 joints, 3D target, so the nullspace needs a posture bias or the columns
        # drift into their limits. The reference is WHERE THE JOG STARTED; th is left free.
        QREF = np.array([*jg["qref"][:3], q[3]])
        dq_task = np.clip(Jp @ err, -0.004, 0.004)           # rate limit ~0.24 m/s
        # Posture bias clipped SEPARATELY and smaller: a shared clip lets it cancel the task step.
        dq_null = np.clip(0.02 * (np.eye(4) - Jp @ J) @ (QREF - q), -0.001, 0.001)
        q += dq_task + dq_null
        # h1 range: 1.43 joint limit minus pitch floor
        q[0] = min(max(q[0], 0.0), 1.38)
        # Never below the pitch the jog STARTED at, and never below the working floor once above
        # it. A blanket floor tilts a parked boom -- which is level -- by 10 degrees on arming.
        dh_floor = min(JOG_DH_FLOOR, max(float(jg["q0dh"]), 0.0))
        # pitch cap 63°
        q[1] = min(max(q[1], q[0] + dh_floor), q[0] + 0.20)
        # NEVER beyond the 1.43 joint limit (NaN source)
        q[1] = min(q[1], 1.42)
        # a1 boom range
        q[2] = min(max(q[2], 0.0), 0.625)
        if self._grip_fk(*q)[2] < 0.12:
            # Floor guard: below the floor, the kinematic-vs-floor fight goes NaN.
            q = jg["q"]
        jg["q"] = q
        self._jog_write(q)
        return float(np.linalg.norm(err))

    def _jog_write(self, q):
        """Write a jog 4-vector [h1, h2, a1, th] to the robot: LUT closure passives + kinematic
        force + drive, base held at the jog anchor.  One physics step."""
        full = np.asarray(self.jog["qhold"], float).copy()
        # `fcmd` is the jog slider, then `_finger_cmd`: a jog started with the fingers wide must
        # not restore `fhold` over a close's grip.
        full[self._f_idx] = (self.jog["fcmd"] if self.jog.get("fcmd") is not None
                             else self._finger_cmd if self._finger_cmd is not None
                             else self.jog["fhold"])
        vals = {"ColumnLeftBearingJoint_1": q[0], "ColumnRightBearingJoint_1": q[1],
                "ArmLeftJoint_1": q[2], "BaseJoint_1": q[3]}
        vals.update(self._closure_passives(q[1] - q[0], q[2]))
        for n, v in vals.items():
            if n in self.idx:
                full[self.idx[n]] = v
        bx, by, byaw = self.jog["anchor"]
        # A VELOCITY in the chassis frame, integrated one physics step at a time: mapping the
        # stick to a POSITION jumps the whole distance in one frame, wheels included.
        vx, vy = self.jog.get("base_vel", (0.0, 0.0))
        dx, dy, dyaw = self.jog.get("base_delta", (0.0, 0.0, 0.0))
        _yt = self.jog.get("yaw_target")
        if _yt is not None:
            # Slew, never snap: the ring names a heading, and a bounded turn rate is what lets
            # the wheels roll through it.
            _e = wrap(float(_yt) - dyaw)
            _step = BASE_JOG_RADPS * self.dt
            dyaw += max(-_step, min(_step, _e))
        if vx or vy or _yt is not None:
            _hd = byaw + dyaw
            dx += (math.cos(_hd) * vy + math.sin(_hd) * vx) * BASE_JOG_MPS * self.dt
            dy += (math.sin(_hd) * vy - math.cos(_hd) * vx) * BASE_JOG_MPS * self.dt
            self.jog["base_delta"] = (dx, dy, dyaw)
            # The hand rides the chassis: `target` and `origin` are WORLD points, so the servo
            # would otherwise chase the pose the hand held before the base moved.
            _p = self.jog.get("base_prev", (0.0, 0.0, 0.0))
            _c, _s = math.cos(dyaw - _p[2]), math.sin(dyaw - _p[2])
            _b0 = np.array([bx + _p[0], by + _p[1]])
            _b1 = np.array([bx + dx, by + dy])
            for _k in ("target", "origin"):
                _v = np.asarray(self.jog[_k], float).copy()
                _r = _v[:2] - _b0
                _v[:2] = _b1 + np.array([_c * _r[0] - _s * _r[1], _s * _r[0] + _c * _r[1]])
                self.jog[_k] = _v
        self.jog["base_prev"] = (dx, dy, dyaw)
        self.set_base(bx + dx, by + dy, byaw + dyaw)
        self._force(full)
        self._apply(full)
        self.world.step(render=not HEADLESS)
