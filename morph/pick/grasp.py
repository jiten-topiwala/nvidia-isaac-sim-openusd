"""Seat the object in the cage, capture the pin reference, verify the hold and lift.

Part of the pick cycle — see `morph/pick/__init__.py` for the stage order and the shared
state object. Imports Isaac APIs at module level; only importable after SimulationApp.
"""
import math
import os

import numpy as np

from morph.config import PERFECT, HEADLESS, KNOWN, COLUMN_MAX


class GraspStage:
    """One stage of `pick`. Mixed into `PickMixin`; all `self.*` belong to `Demo`."""

    def _pick_seat(self, st):
        self._pad_trace("seat-entry")
        obj_idx, status = st.obj_idx, st.status
        dock, psi, obj, replay = st.dock, st.psi, st.obj, st.replay
        alive, grip = st.alive, st.grip
        restore_bystanders = st.restore_bystanders

        if status:
            status(f"gripping object {obj_idx}")
        # Did the grip command survive the handoff? `_apply` honours `_finger_cmd` and `_force`
        # excludes the finger DOFs while it is set, so print the COMMAND against the MEASUREMENT: if
        # cmd is open, something re-commanded it; if cmd is still closed while act has sprung open,
        # the drive is losing to the object.
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
        if self.clut is not None:
            # STEP-1 GUARANTEE (all modes): the object NEVER moves during the grasp.  Fine-align
            # brought the pinch onto it; it stays exactly where it stands on the floor.
            seat = floor_pos.copy()
            seat_gap = float(np.linalg.norm(pinch - floor_pos))
        else:
            # teleport fallback keeps the old visible floor->cage seat sweep (no servo available)
            seat = np.array([pinch[0], pinch[1], max(pinch[2], self.obj_half_h)])
            seat_gap = float(np.linalg.norm(seat - floor_pos))
            if PERFECT:
                seat = floor_pos.copy()                      # NEVER force-move a real colliding body
            elif seat_gap > 0.01:
                n_lift = int(0.7 / self.dt)
                for k in range(n_lift):
                    f = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / n_lift)   # smoothstep 0..1
                    p = floor_pos + f * (seat - floor_pos)
                    obj.set_world_poses(positions=np.array([p]), orientations=np.array([[1, 0, 0, 0]]))
                    obj.set_velocities(np.zeros((1, 6)))
                    self.set_base(float(dock[0]), float(dock[1]), psi)
                    self._force(grip)   # KINEMATIC-pin the arm through the seat: forcing   # it to stay AT the settled
                    self._apply(grip)   # pose keeps the loop ok
                    self.world.step(render=not HEADLESS)
        st.seat, st.seat_gap, st.floor_pos, st.grip = seat, seat_gap, floor_pos, grip

    def _pick_capture(self, st):
        if getattr(self, "_gate_blocked", False):
            self._gate_blocked = False
            print(">>> capture SKIPPED: the close gate refused this pose -- no squeeze, no lift; "
                  "fail clean and recover", flush=True)
            st.held = False
            return
        self._pad_trace("capture-entry")
        obj_idx, status = st.obj_idx, st.status
        dock, obj, replay, alive = st.dock, st.obj, st.replay, st.alive
        grip, seat, seat_gap, floor_pos = st.grip, st.seat, st.seat_gap, st.floor_pos
        pin_bystanders = st.pin_bystanders
        self._hold(grip, int(0.3 / self.dt), pin=(obj, seat), kin=True, on_step=pin_bystanders)
        if PERFECT:
            # squeeze settle: give the force-limited drives a moment to load the pads, then read the
            # REAL contact evidence.  _fN accumulates impulse over the window -> avg N = sum/T
            self._fN = {}
            self._ctc = {}
            self._hold(grip, int(0.4 / self.dt), kin=True)
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
        # Pin reference: the object is now properly seated between the fingers and settled — capture
        # its SMALL gripper-frame offset HERE.
        self._phase_report("pin-capture", obj)   # the seat the pin is about to freeze for the carry
        self._capture_grip_offset(obj)
        # ...and in practical mode, hand the hold to a real constraint. `_capture_grip_offset`
        # stays: it is the fallback the pose pin uses when GRIP_WELD=0, and `_obj_local` is read
        # elsewhere.
        self._weld_grip(obj_idx)
        self._carry_check(obj, "grip")           # baseline: everything after is measured against it
        oc = np.asarray(obj.get_world_poses()[0][0], float)
        pbx, pby, _ = self.base_pose()
        # held only if BOTH object and base are finite AND in-arena — the ~20% forced-linkage settle
        # explosion leaves them at huge-but-finite coords (object at 6.7e6, base flung off-floor);
        # guard so we DON'T then run place() on an invalid state (which nav-fails messily).
        held = bool(alive and np.all(np.isfinite(oc)) and np.all(np.abs(oc[:2]) < 20.0)
                    and math.isfinite(pbx) and 0.2 < pbx < 7.8 and -7.8 < pby < -0.2
                    and math.hypot(pbx - float(dock[0]), pby - float(dock[1])) < 0.2)
        #   ^ base must still be AT the dock: a wall-wedged TELEPORT grasp shoves the base slowly
        #     (in-bounds but drifting) — carrying on from a displaced base flies the object around.
        st.held, st.seat, st.grip = held, seat, grip

    def _pick_lift(self, st):
        self._pad_trace("lift-entry")
        obj_idx, status = st.obj_idx, st.status
        dock, obj, replay, seat, held = st.dock, st.obj, st.replay, st.seat, st.held
        pin_bystanders, restore_bystanders = st.pin_bystanders, st.restore_bystanders
        # Visible lift: replay the baked grasp->carry trajectory. Practical mode rides the weld;
        # perfect mode holds on pad friction alone.
        self.lifted = False
        if held and replay and os.environ.get("LIFT_REPLAY", "1") == "1":
            if status:
                status(f"lifting object {obj_idx}")
            seat_z = float(seat[2])
            lift_t = float(os.environ.get("LIFT_T", "2.5"))
            self._fN = {}
            if PERFECT and os.environ.get("LIFT_MODE", "columns") == "columns":
                # Computed column lift, from the pose the arm is actually in. The baked lift replay
                # starts from the RECORDED grasp pose, which is a long way from where the close ends
                # -- so its first frame teleports the arm back and down, and the hand lurches away
                # from the object and leaves it behind even with the squeeze intact and the pads
                # loaded.
                _dz_l = 0.25
                _n_l = max(1, int(lift_t / self.dt))
                _q_l = np.asarray(self.robot.get_joint_positions(), float).copy()
                _ih1, _ih2 = self.idx["ColumnLeftBearingJoint_1"], self.idx["ColumnRightBearingJoint_1"]
                _h10, _h20 = float(_q_l[_ih1]), float(_q_l[_ih2])
                try:
                    _bx_l, _by_l, _byaw_l = self.base_ledger()
                except Exception:
                    _bx_l = None
                _tight = 0.0
                _prev_imp = dict(getattr(self, "_ctcN", {}))   # lift-trace baseline (see below)
                print(f">>> lift[columns]: +{_dz_l * 1000:.0f}mm over {lift_t:.1f}s from h1 {_h10:.3f} "
                      f"(a1/dh/th untouched; the baked replay would have jumped to a1 0.347)", flush=True)
                for _k in range(1, _n_l + 1):
                    _f = 0.5 - 0.5 * math.cos(math.pi * _k / _n_l)          # smoothstep
                    _qc = _q_l.copy()
                    _qc[_ih1] = float(np.clip(_h10 + _f * _dz_l, 0.0, COLUMN_MAX))
                    _qc[_ih2] = float(np.clip(_h20 + _f * _dz_l, 0.0, COLUMN_MAX))
                    if (self._finger_cmd is not None and _k % int(0.25 / self.dt) == 0
                            and _tight < 0.30):
                        _tight += 0.03
                        for pos, nm in enumerate(self._f_names):
                            if nm.startswith("finger_") and nm.endswith("joint_1_1"):
                                self._finger_cmd[pos] = min(self._finger_cmd[pos] + 0.02, 1.0)
                    if _bx_l is not None:
                        self.set_base(_bx_l, _by_l, _byaw_l)
                    _qv = np.zeros_like(_qc)
                    _dz_step = _dz_l * (0.5 * math.pi / _n_l) * math.sin(math.pi * _k / _n_l)
                    _qv[_ih1] = _qv[_ih2] = _dz_step / self.dt
                    self._force(_qc, _qv)
                    self._apply(_qc)
                    if pin_bystanders is not None:
                        pin_bystanders()
                    self.world.step(render=not HEADLESS)
                    # Is there load on the object, and when is it lost? The end state alone cannot
                    # tell a grip that NEVER FORMED from one that formed and broke -- both finish
                    # with the hand up and the object on the floor.
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
                            print(f">>>   lift[{_k:4d}/{_n_l}] h1 cmd {float(_qc[_ih1]):.3f} act "
                                  f"{float(_qa_t[_ih1]):.3f} | obj z {float(_oc_t[2]):.3f} "
                                  f"(rise {(float(_oc_t[2]) - self.obj_half_h) * 1000:+.0f}mm) | "
                                  f"pinch z {float(self._pinch()[2]):.3f} | gaps(mm) {_gaps} | "
                                  f"obj-impulse {_d_imp if _d_imp else 'NONE'}", flush=True)
                        except Exception as _e_lt:
                            print(f">>>   lift trace failed ({_e_lt})", flush=True)
                self.grip = np.asarray(self.robot.get_joint_positions(), float).copy()
                self.lifted = self._finite("lift-columns")
            else:
                self.grip = self._play_traj(self.traj["lift"], lift_t,
                                            grip_pin=obj, hold_fingers=True, on_step=pin_bystanders)
                self.lifted = self._finite("lift-replay")
            if PERFECT and self.lifted:
                pads2 = {k: round(v / lift_t, 1) for k, v in getattr(self, "_fN", {}).items()}
                print(f">>> lift pad loads (N): {pads2}", flush=True)
                oc2 = np.asarray(obj.get_world_poses()[0][0], float)
                # Measure the rise from the RESTING height, not from the seat: `seat_z` is read
                # after the squeeze, and the squeeze presses the object into the floor, so a rebound
                # to its resting height reads as tens of millimetres of lift.
                rise = float(oc2[2]) - max(seat_z, self.obj_half_h)
                if seat_z < self.obj_half_h - 0.005:
                    print(f">>> lift[friction]: NOTE the squeeze pressed the object "
                          f"{(self.obj_half_h - seat_z) * 1000:.0f}mm into the floor "
                          f"(seat z {seat_z:.3f} vs resting {self.obj_half_h:.3f}) — rise is measured "
                          f"from the resting height, not from that", flush=True)
                self.lifted = bool(np.all(np.isfinite(oc2)) and rise > 0.15)
                print(f">>> lift[friction]: object rose {rise * 1000:.0f}mm -> "
                      f"{'HELD by friction' if self.lifted else 'SLIPPED/DROPPED'}", flush=True)
            # Level the boom for the carry. The baked carry pose keeps a small tilt, which tucks the
            # load closer to the chassis, but a horizontal carry is the preferred look.
            if (os.environ.get("CARRY_LEVEL", "1") == "1" and self.lifted
                    and getattr(self, "clut", None) is not None):
                _qL = np.asarray(self.robot.get_joint_positions(), float).copy()
                _i1L = self.idx["ColumnLeftBearingJoint_1"]
                _i2L = self.idx["ColumnRightBearingJoint_1"]
                _iaL = self.idx.get("ArmLeftJoint_1")
                _a1L = float(_qL[_iaL]) if _iaL is not None else 0.10
                _dh0L = float(_qL[_i2L]) - float(_qL[_i1L])
                if abs(_dh0L) > 2e-3:
                    _lzL = lambda d_: self._lut_z(d_, _a1L)
                    _CL = float(_qL[_i1L]) + _lzL(_dh0L)
                    _h1fL = _CL - _lzL(0.0)
                    if 0.0 <= _h1fL <= COLUMN_MAX:
                        # Keep the gripper parallel THROUGH the ramp, in one motion. Two simpler
                        # approaches fail: per-step feedback from a gradient probed once at entry
                        # pushes the wrong way as the attitude changes, and chunked stop-and-level
                        # holds the tilt but pauses the boom repeatedly, each pause a visible
                        # twitch.
                        _nL = max(6, int(0.8 / self.dt))
                        _ckL = max(1, _nL // 6)
                        _wl_on = os.environ.get("WRIST_LEVEL", "1") == "1"
                        _wiL, _wextL, _wstepL = None, 0.0, 0.0
                        _qbase = _qL.copy()
                        for _kL in range(_nL):
                            if _wl_on and _kL % _ckL == 0:
                                _trk = self._level_gradient(obj)      # invisible, single joint
                                if _trk is not None:
                                    _tnow = self._tilt_deg(obj)
                                    _wiL = _trk[0]
                                    _d = float(np.clip(-(_tnow or 0.0) / _trk[1], -0.25, 0.25))
                                    _wstepL = _d / _ckL
                            _fL = 0.5 - 0.5 * math.cos(math.pi * (_kL + 1) / _nL)
                            _dhkL = _dh0L * (1.0 - _fL)
                            _h1kL = float(np.clip(_CL - _lzL(_dhkL), 0.0, COLUMN_MAX))
                            _qk = _qbase.copy()
                            _qk[_i1L] = _h1kL
                            _qk[_i2L] = float(np.clip(_h1kL + _dhkL, 0.0, COLUMN_MAX))
                            for _pnL, _pvL in self._closure_passives(_dhkL, _a1L).items():
                                if _pnL in self.idx:
                                    _qk[self.idx[_pnL]] = float(_pvL)
                            if _wiL is not None:
                                _wextL += _wstepL
                                _qk[_wiL] = float(_qbase[_wiL]) + _wextL
                            self._force(_qk)
                            self._apply(_qk)
                            self.world.step(render=not HEADLESS)
                        _qL = np.asarray(self.robot.get_joint_positions(), float).copy()
                        if os.environ.get("WRIST_LEVEL", "1") == "1":
                            self._level_wrist(obj, "carry-level")
                        self.grip = np.asarray(self.robot.get_joint_positions(), float).copy()
                        print(f">>> carry-level: boom dh {_dh0L:+.3f} -> +0.000 after the lift -- "
                              f"horizontal carry (CARRY_LEVEL=0 restores MuJoCo's tilted carry)",
                              flush=True)
                    else:
                        print(f">>> carry-level: SKIPPED -- level would need h1 {_h1fL:.3f}, out "
                              f"of bounds at this height", flush=True)
            self._phase_report("lift-end", obj)
            held = self.lifted
            self._carry_check(obj, "lift-end")   # the lift is where a weak hold shows first
        restore_bystanders()                                 # exact floor pose back, colliders on
        # _obj_local is None when the close gate skipped the grasp (no capture) -- a clean fail, not
        # a crash. round() on None threw and took the cycle down; guard it.
        _gl = np.round(self._obj_local, 3).tolist() if self._obj_local is not None else None
        print(f">>> PICK obj {obj_idx}: held={held} lifted={self.lifted} dock=({dock[0]:.2f},{dock[1]:.2f}) "
              f"grip_local={_gl}", flush=True)
        st.held = held

