"""Seat the object in the cage, capture the pin reference, verify the hold and lift.
Stage order and the shared state object: `morph/pick/__init__.py`.
Imports Isaac APIs at module level; only importable after SimulationApp."""
import math
import os

import numpy as np

from morph.config import HEADLESS, KNOWN, COLUMN_MAX
from morph.gripper.api import hold_gripper


class GraspStage:
    """One stage of `pick`. Mixed into `PickMixin`; all `self.*` belong to `Demo`."""

    def _pick_seat(self, st):
        self._stage = "seat"
        self._pad_trace("seat-entry")
        obj_idx, status = st.obj_idx, st.status
        dock, psi, obj, replay = st.dock, st.psi, st.obj, st.replay
        alive, grip = st.alive, st.grip
        restore_bystanders = st.restore_bystanders

        if status:
            status(f"gripping object {obj_idx}")
        # COMMAND against MEASUREMENT: cmd open means something re-commanded it; cmd closed while
        # act has sprung open means the drive is losing to the object.
        try:
            _qa_h = np.asarray(self.robot.get_joint_positions(), float)
            _rows = []
            for _fh in "abc":
                _ih = self.idx.get(f"finger_{_fh}_joint_1_1")
                if _ih is None:
                    continue
                _ci = list(self._f_idx).index(_ih) if _ih in list(self._f_idx) else None
                _cv = (float(self._finger_cmd[_ci]) if (self._finger_cmd is not None
                                                        and _ci is not None) else float("nan"))
                _rows.append(f"{_fh}: cmd {_cv:+.3f} act {float(_qa_h[_ih]):+.3f}")
            print(f">>> grip handoff: j1 {' | '.join(_rows)}  "
                  f"(_finger_cmd {'SET' if self._finger_cmd is not None else 'NONE'})", flush=True)
        except Exception as _e_h:
            print(f">>> grip handoff: readout failed ({_e_h})", flush=True)
        pinch = self._pinch()
        floor_pos = np.asarray(obj.get_world_poses()[0][0], float)
        # SEAT GUARANTEE: the object NEVER moves during the grasp -- fine-align brought the pinch onto
        # it, and it stays exactly where it stands on the floor.
        seat = floor_pos.copy()
        seat_gap = float(np.linalg.norm(pinch - floor_pos))
        st.seat, st.seat_gap, st.floor_pos, st.grip = seat, seat_gap, floor_pos, grip

    def _pick_capture(self, st):
        self._stage = "capture"
        if getattr(self, "_nonfinite_n", 0) and not self._finite("capture-entry"):
            print(">>> capture: articulation non-finite -> attempt failed clean", flush=True)
            st.held = False
            return
        if getattr(self, "_gate_blocked", False):
            self._gate_blocked = False
            print(">>> capture SKIPPED: the close gate refused this pose -- no squeeze, no lift; "
                  "fail clean and recover", flush=True)
            st.held = False
            return
        self._pad_trace("capture-entry")
        obj_idx, status = st.obj_idx, st.status
        dock, obj, replay, alive = st.dock, st.obj, st.replay, st.alive
        grip, seat_gap, floor_pos = st.grip, st.seat_gap, st.floor_pos
        pin_bystanders = st.pin_bystanders
        # CAPTURE_DRIVE=1: the capture holds run the ARM on drives. Kinematic arm with a teleported root,
        # or driven arm with a held root -- never mixed, or contact depenetration kicks the root away.
        _cdrv = os.environ.get("CAPTURE_DRIVE", "0") == "1"
        hold_gripper(self, grip, 0.3, kin=not _cdrv, on_step=pin_bystanders)
        # squeeze settle: let the force-limited drives load the pads, then read the contact
        # evidence. `_fN` accumulates impulse over the window, so avg N = sum / T.
        self._fN = {}
        self._ctc = {}
        hold_gripper(self, grip, 0.4, kin=not _cdrv)
        react = self._finger_reaction()
        oc_now = np.asarray(obj.get_world_poses()[0][0], float)
        drift = float(np.linalg.norm(oc_now[:2] - floor_pos[:2]))
        pads = {k: round(v / 0.4, 1) for k, v in getattr(self, "_fN", {}).items()}
        print(f">>> grip contacts: {dict(sorted(self._ctc.items(), key=lambda kv: -kv[1])[:6])}",
              flush=True)
        print(f">>> grip[friction]: reaction {react:.1f}N, obj drift {drift * 1000:.0f}mm, "
              f"pad loads (N): {pads}, drive torque (Nm): {self._finger_efforts()}, "
              f"pinch->obj {np.linalg.norm(oc_now[:2] - self._pinch()[:2]) * 1000:.0f}mm", flush=True)
        seat = oc_now
        print(f">>> grip: seat={np.round(seat,3).tolist()} pinch-gap={seat_gap:.3f} "
              f"(object {'UNMOVED' if replay else 'seated'})", flush=True)
        self.grip = grip
        # Pin reference: object seated and settled -- capture its gripper-frame offset HERE.

        # the seat the pin is about to freeze for the carry
        self._phase_report("pin-capture", obj)
        self._capture_grip_offset(obj)
        # `_capture_grip_offset` stays: it is the GRIP_WELD=0 fallback and `_obj_local` is read
        # elsewhere.

        # baseline: everything after is measured against it
        self._carry_check(obj, "grip")
        oc = np.asarray(obj.get_world_poses()[0][0], float)
        pbx, pby, _ = self.base_pose()
        # Grasp-quality gate (drive mode): a tilted or off-centre capture is carried all the way
        # to the shelf and only fails there. Fail HERE so the pick retry re-seats.
        if self._drive_on() and alive:
            try:
                _tilt = float(self._tilt_deg(obj) or 0.0)
                _xyoff = float(np.linalg.norm(np.asarray(self._pinch(), float)[:2] - oc[:2])) * 1000.0
                _tmax = float(os.environ.get("LIFT_TILT_MAX", "10"))
                _xmax = float(os.environ.get("LIFT_XYOFF_MAX", "60"))
                _nheld = len(getattr(self, "_fb_held", "abc"))   # pads carrying load
                _nmin = int(os.environ.get("LIFT_PADS_MIN", "3"))  # 1 pad passes tilt+offset, then tilts
                # Interpenetration gate: bound pad depth and palm penetration before the lift.
                _r_g = float(os.environ.get("OBJ_RADIUS", KNOWN["object_radius"]))
                _oR_g = self._obj_R(obj)
                _pad_min = min(self._finger_surface_gap(f, oc, _r_g, self.obj_half_h, _oR_g) for f in "abc") * 1000.0
                _palm_g = self._finger_surface_gap("palm", oc, _r_g, self.obj_half_h, _oR_g) * 1000.0
                _pad_lim = float(os.environ.get("LIFT_PAD_MIN_MM", "-12"))
                # The palm is a LOADED contact in this grasp, so gate its PENETRATION, not its
                # gap; the limit is a small measured allowance, not a hardware bound.
                _palm_lim = float(os.environ.get("LIFT_PALM_MIN_MM", "-2"))
                if (_tilt > _tmax or _xyoff > _xmax or _nheld < _nmin
                        or _pad_min < _pad_lim or _palm_g < _palm_lim):
                    self._fallback("capture-grasp-rejected")
                    print(f">>> capture: GRASP REJECTED -- object tilt {_tilt:.1f} deg (max {_tmax:.0f}) / "
                          f"pinch-centre offset {_xyoff:.0f} mm (max {_xmax:.0f}) / loaded pads {_nheld} "
                          f"(min {_nmin}) / deepest pad {_pad_min:+.1f} mm (min {_pad_lim:.0f}) / palm "
                          f"{_palm_g:+.1f} mm (min {_palm_lim:.0f}); no lift, retry", flush=True)
                    alive = False
                else:
                    print(f">>> capture: grasp gate OK -- tilt {_tilt:.1f} deg, offset {_xyoff:.0f} mm, "
                          f"pads {_nheld}, deepest pad {_pad_min:+.1f} mm, palm {_palm_g:+.1f} mm", flush=True)
            except Exception as _e_g:
                print(f">>> capture: grasp-quality gate unavailable ({_e_g})", flush=True)
        # held only if BOTH object and base are finite AND in-arena: a forced-linkage settle
        # explosion leaves them huge-but-finite, and place() on that nav-fails messily.
        held = bool(alive and np.all(np.isfinite(oc)) and np.all(np.abs(oc[:2]) < 20.0)
                    and math.isfinite(pbx) and 0.2 < pbx < 7.8 and -7.8 < pby < -0.2
                    and math.hypot(pbx - float(dock[0]), pby - float(dock[1])) < 0.2)
        #   ^ base must still be AT the dock: a slow in-bounds drift still flies the object around.
        st.held, st.seat, st.grip = held, seat, grip

    def _pick_lift(self, st):
        self._stage = "lift"
        self._pad_trace("lift-entry")
        obj_idx, status = st.obj_idx, st.status
        dock, obj, seat, held = st.dock, st.obj, st.seat, st.held
        pin_bystanders, restore_bystanders = st.pin_bystanders, st.restore_bystanders
        self.lifted = False
        if held:
            if status:
                status(f"lifting object {obj_idx}")
            seat_z = float(seat[2])
            lift_t = float(os.environ.get("LIFT_T", "2.5"))
            self._fN = {}
            # Lift from the pose the arm is actually in: a live read is path-independent,
            # so the ramp needs no bake and cannot start by teleporting the hand.
            _dz_l = 0.25
            _n_l = max(1, int(lift_t / self.dt))
            _q_l = np.asarray(self.robot.get_joint_positions(), float).copy()
            _ih1, _ih2 = self.idx["ColumnLeftBearingJoint_1"], self.idx["ColumnRightBearingJoint_1"]
            _h10, _h20 = float(_q_l[_ih1]), float(_q_l[_ih2])
            try:
                _bx_l, _by_l, _byaw_l = self.base_ledger()
            except Exception:
                _bx_l = None
            _prev_imp = dict(getattr(self, "_ctcN", {}))   # lift-trace baseline (see below)
            print(f">>> lift[columns]: +{_dz_l * 1000:.0f}mm over {lift_t:.1f}s from h1 {_h10:.3f} "
                  f"(a1/dh/th untouched)", flush=True)
            for _k in range(1, _n_l + 1):
                _f = 0.5 - 0.5 * math.cos(math.pi * _k / _n_l)          # smoothstep
                _qc = _q_l.copy()
                _qc[_ih1] = float(np.clip(_h10 + _f * _dz_l, 0.0, COLUMN_MAX))
                _qc[_ih2] = float(np.clip(_h20 + _f * _dz_l, 0.0, COLUMN_MAX))
                if _bx_l is not None:
                    # LIFT_PIN=0: no per-step root write while lifting on drives. `set_base` teleports the
                    # whole articulation, and PhysX drops the object's friction anchors on every one.
                    if os.environ.get("LIFT_PIN", "1") == "1":
                        self.set_base(_bx_l, _by_l, _byaw_l)
                _qv = np.zeros_like(_qc)
                _dz_step = _dz_l * (0.5 * math.pi / _n_l) * math.sin(math.pi * _k / _n_l)
                _qv[_ih1] = _qv[_ih2] = _dz_step / self.dt
                # LIFT_DRIVE=1: lift on the column DRIVES, since a teleported link invalidates the
                # anchor static friction holds by. Safe only because (dh, a1) stay unchanged.
                if os.environ.get("LIFT_DRIVE", "0") == "1":
                    self._apply(_qc)
                else:
                    self._force(_qc, _qv)
                    self._apply(_qc)
                if pin_bystanders is not None:
                    pin_bystanders()
                self.world.step(render=not HEADLESS)
                # The end state alone cannot tell a grip that NEVER FORMED from one that
                # formed and broke -- both finish with the hand up and the object down.
                if (os.environ.get("LIFT_TRACE", "1") == "1"
                        and _k % max(1, int(0.25 / self.dt)) == 0):
                    try:
                        _oc_t = np.asarray(obj.get_world_poses()[0][0], float)
                        _R_t = self._obj_R(obj)
                        _gaps = {f: round(self._finger_surface_gap(
                            f, _oc_t, float(KNOWN["object_radius"]), self.obj_half_h,
                            _R_t) * 1000, 1) for f in "abc"}
                        _now_imp = dict(getattr(self, "_ctcN", {}))
                        _d_imp = {k2: round(v2 - _prev_imp.get(k2, 0.0), 4)
                                  for k2, v2 in _now_imp.items()
                                  if v2 - _prev_imp.get(k2, 0.0) > 1e-6}
                        _prev_imp = _now_imp
                        _qa_t = np.asarray(self.robot.get_joint_positions(), float)
                        # SAG telemetry, mrad of (actual - target): under LIFT_DRIVE nothing
                        # forces a1/turret/wrist and a walking joint rewrites the grasp.
                        _sag = {}
                        for _sn, _sk in (("a1", "ArmLeftJoint_1"), ("th", "BaseJoint_1"),
                                         ("wr", "HandBearingJoint_1"),
                                         ("wy", "gripper_y_rotation_1")):
                            _si = self.idx.get(_sk)
                            if _si is not None:
                                _sag[_sn] = round((float(_qa_t[_si]) - float(_qc[_si])) * 1000, 1)
                        print(f">>>   lift[{_k:4d}/{_n_l}] h1 cmd {float(_qc[_ih1]):.3f} act "
                              f"{float(_qa_t[_ih1]):.3f} | obj z {float(_oc_t[2]):.3f} "
                              f"(rise {(float(_oc_t[2]) - self.obj_half_h) * 1000:+.0f}mm) | "
                              f"pinch z {float(self._pinch()[2]):.3f} | gaps(mm) {_gaps} | "
                              f"sag(mrad) {_sag} | "
                              f"obj-impulse {_d_imp if _d_imp else 'NONE'}", flush=True)
                        # A rising z alone cannot tell held from squeezed, riding a finger, or
                        # tumbling: a carry must be upright, xy-locked, z-tracking AND loaded.
                        _axz = float(_R_t[:, 2][2]) if _R_t is not None else float("nan")
                        _pin_t = np.asarray(self._pinch(), float)
                        _xy_d = float(np.linalg.norm(_oc_t[:2] - _pin_t[:2])) * 1000
                        _ok_carry = (_axz > 0.95 and _xy_d < 60.0
                                     and -0.05 < (float(_pin_t[2]) - float(_oc_t[2])) < 0.12
                                     and all(g is not None and g < 25.0 for g in _gaps.values()))
                        print(f">>>   carry-check[{_k:4d}] axis.z {_axz:+.3f} xy-off "
                              f"{_xy_d:.0f}mm -> "
                              f"{'CARRYING' if _ok_carry else 'MISBEHAVING (z-rise is not a lift)'}",
                              flush=True)
                    except Exception as _e_lt:
                        print(f">>>   lift trace failed ({_e_lt})", flush=True)
            self.grip = np.asarray(self.robot.get_joint_positions(), float).copy()
            self.lifted = self._finite("lift-columns")
            if self.lifted:
                pads2 = {k: round(v / lift_t, 1) for k, v in getattr(self, "_fN", {}).items()}
                print(f">>> lift pad loads (N): {pads2}", flush=True)
                oc2 = np.asarray(obj.get_world_poses()[0][0], float)
                # Rise is measured from the RESTING height, not the seat: `seat_z` is read after
                # the squeeze pressed the object into the floor, so the rebound alone reads as lift.
                rise = float(oc2[2]) - max(seat_z, self.obj_half_h)
                if seat_z < self.obj_half_h - 0.005:
                    print(f">>> lift[friction]: NOTE the squeeze pressed the object "
                          f"{(self.obj_half_h - seat_z) * 1000:.0f}mm into the floor "
                          f"(seat z {seat_z:.3f} vs resting {self.obj_half_h:.3f}) — rise is measured "
                          f"from the resting height, not from that", flush=True)
                self.lifted = bool(np.all(np.isfinite(oc2)) and rise > 0.15)
                print(f">>> lift[friction]: object rose {rise * 1000:.0f}mm -> "
                      f"{'HELD by friction' if self.lifted else 'SLIPPED/DROPPED'}", flush=True)
            # Level the boom for the carry: cosmetic, the baked pose keeps a small tilt.
            if (os.environ.get("CARRY_LEVEL", "0") == "1" and self.lifted
                    and getattr(self, "clut", None) is not None):
                _qL = np.asarray(self.robot.get_joint_positions(), float).copy()
                _i1L = self.idx["ColumnLeftBearingJoint_1"]
                _i2L = self.idx["ColumnRightBearingJoint_1"]
                _iaL = self.idx.get("ArmLeftJoint_1")
                _a1L = float(_qL[_iaL]) if _iaL is not None else 0.10
                _dh0L = float(_qL[_i2L]) - float(_qL[_i1L])
                # LIFT_LEVEL=0: skip boom levelling under drives -- straightening dh swings the
                # hand ~145mm and ~45deg with the wrist fixed, and a clamped object follows.
                if os.environ.get("LIFT_LEVEL", "1") == "0":
                    _dh0L = 0.0
                if abs(_dh0L) > 2e-3:
                    _lzL = lambda d_: self._lut_z(d_, _a1L)
                    _CL = float(_qL[_i1L]) + _lzL(_dh0L)
                    _h1fL = _CL - _lzL(0.0)
                    if 0.0 <= _h1fL <= COLUMN_MAX:
                        _nL = max(6, int(0.8 / self.dt))
                        _qbase = _qL.copy()
                        for _kL in range(_nL):
                            _fL = 0.5 - 0.5 * math.cos(math.pi * (_kL + 1) / _nL)
                            _dhkL = _dh0L * (1.0 - _fL)
                            _h1kL = float(np.clip(_CL - _lzL(_dhkL), 0.0, COLUMN_MAX))
                            _qk = _qbase.copy()
                            _qk[_i1L] = _h1kL
                            _qk[_i2L] = float(np.clip(_h1kL + _dhkL, 0.0, COLUMN_MAX))
                            for _pnL, _pvL in self._closure_passives(_dhkL, _a1L).items():
                                if _pnL in self.idx:
                                    _qk[self.idx[_pnL]] = float(_pvL)
                            # A per-step `_force` here is a kinematic write on a held object; it
                            # jolts the object sideways and down.
                            if os.environ.get("LIFT_DRIVE", "0") != "1":
                                self._force(_qk)
                            self._apply(_qk)
                            self.world.step(render=not HEADLESS)
                        self.grip = np.asarray(self.robot.get_joint_positions(), float).copy()
                        print(f">>> carry-level: boom dh {_dh0L:+.3f} -> +0.000 after the lift -- "
                              f"horizontal carry (CARRY_LEVEL=0 restores MuJoCo's tilted carry)",
                              flush=True)
                    else:
                        print(f">>> carry-level: SKIPPED -- level would need h1 {_h1fL:.3f}, out "
                              f"of bounds at this height", flush=True)
            self._phase_report("lift-end", obj)
            held = self.lifted
            # the lift is where a weak hold shows first
            self._carry_check(obj, "lift-end")
        # exact floor pose back, colliders on
        restore_bystanders()
        # `_obj_local` is None when the close gate skipped the grasp -- a clean fail; guard round().
        _gl = np.round(self._obj_local, 3).tolist() if self._obj_local is not None else None
        print(f">>> PICK obj {obj_idx}: held={held} lifted={self.lifted} dock=({dock[0]:.2f},{dock[1]:.2f}) "
              f"grip_local={_gl}", flush=True)
        st.held = held

