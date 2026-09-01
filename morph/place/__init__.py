"""Carry to the rack, insert into the slot, release and back out; plus the failure recovery.

Extracted verbatim from play_isaac.py — the method bodies are unchanged.
"""
import math
import os

import numpy as np

import scene
from morph.config import PERFECT, HEADLESS, KNOWN, NUM_OBJECTS, PLACE_DOCK_D, RACK_FRONT_Y, SHELF_SLOTS, COLUMN_MAX, A1_MAX
from morph.geometry import quat_yaw
from morph.place.insert import InsertStage
from morph.place.recover import RecoverStage


class PlaceMixin(InsertStage, RecoverStage):
    """Mixed into Demo. Every `self.*` it touches is owned by Demo."""

    def place(self, obj_idx, slot_idx, status=None):
        """Carry (pinned) to the rack dock, lift the object onto the slot, release."""
        obj = self.objs[obj_idx]
        slot = SHELF_SLOTS[slot_idx]
        dock_xy = (float(slot[0]), RACK_FRONT_Y + PLACE_DOCK_D)
        face_south = -math.pi / 2
        # Arm-place mode: pick ended LIFTED (arm at the baked carry pose, object riding the gripper)
        # and a baked insert trajectory exists -> the ARM places the object on the slot.
        lifted = (getattr(self, "lifted", False) and self.traj is not None
                  and os.environ.get("PLACE_REPLAY", "1") == "1")
        if PERFECT and not lifted:
            # no pin exists to glide the object in perfect mode — a pick that is not genuinely
            # lifted CANNOT be placed; fail clean instead of fighting physics
            print(f">>> PLACE obj {obj_idx}: not friction-lifted -> abort", flush=True)
            return
        tname = "place_low" if slot[2] < 0.5 else ("place_mid" if slot[2] < 1.0 else "place_high")
        dock_ins = None
        if lifted:
            g = self.traj[tname]["meta"]["grip_base_frame"]
            c, s = math.cos(face_south), math.sin(face_south)
            # INSERT_STANDOFF keeps the extended ARM outside the rack volume: reaching the gripper
            # all the way to the slot puts fingers and boom between the shelves, and the kinematic-
            # versus-shelf fight can NaN the robot mid-place.
            _stand0 = 0.20
            _half0 = 0.35
            _ymin0 = RACK_FRONT_Y + _half0 + 0.10

            def _dock_for(_thv):
                """(dock after the clearance clamp, reach it demands) for a turret angle."""
                _fs = face_south + _thv
                _c, _s = math.cos(_fs), math.sin(_fs)
                _cg, _sg = math.cos(-_thv), math.sin(-_thv)
                _gv = (float(_cg * g[0] - _sg * g[1]), float(_sg * g[0] + _cg * g[1])) + tuple(g[2:])
                _d = [float(slot[0] - (_c * _gv[0] - _s * _gv[1])),
                      float(slot[1] - (_s * _gv[0] + _c * _gv[1]) + _stand0)]
                _d[1] = max(_d[1], _ymin0)                    # same floor as below
                return (_d, _gv, _fs, _c, _s,
                        float(math.hypot(slot[0] - _d[0], slot[1] - _d[1])))

            _th_g = 0.0
            if os.environ.get("PLACE_TH_ZERO", "1") == "1" and "BaseJoint_1" in self.idx:
                _th_try = float(np.asarray(self.robot.get_joint_positions(),
                                           float)[self.idx["BaseJoint_1"]])
                _, _, _, _, _, _r_keep = _dock_for(0.0)       # inherited turret
                _dz, _gz, _fsz, _cz, _sz, _r_zero = _dock_for(_th_try)
                _lim = 0.78
                if _r_zero <= _lim and _r_zero <= _r_keep + 0.05:
                    _th_g, face_south, g, c, s = _th_try, _fsz, _gz, _cz, _sz
                    print(f">>> place dock: turret {_th_g:+.3f} rad "
                          f"({math.degrees(_th_g):+.1f}deg) absorbed into the base yaw -- arm "
                          f"reaches straight ahead, not across the body (needs {_r_zero:.3f}m vs "
                          f"{_r_keep:.3f}m inherited, limit {_lim:.2f}m)", flush=True)
                else:
                    print(f">>> place dock: KEEPING the inherited turret for this slot -- zeroing "
                          f"it demands {_r_zero:.3f}m of reach (inherited {_r_keep:.3f}m, limit "
                          f"{_lim:.2f}m); the trim cannot close that gap", flush=True)
            stand = 0.20
            dock_ins = (float(slot[0] - (c * g[0] - s * g[1])),
                        float(slot[1] - (s * g[0] + c * g[1]) + stand))
            # The chassis has a footprint: clear the rack by its EDGE, not its origin.
            _half = 0.35
            _edge = 0.10
            _y_min = RACK_FRONT_Y + _half + _edge
            if dock_ins[1] < _y_min:
                print(f">>> insert dock: floored at chassis clearance -- y {dock_ins[1]:.3f} -> "
                      f"{_y_min:.3f} ({_edge * 1000:.0f}mm from the plate edge to the rack face; "
                      f"the plate reaches {_half:.2f}m forward of the base origin)", flush=True)
                dock_ins = (dock_ins[0], _y_min)
            # (A model-based lateral dock re-aim was tried twice and removed: its prediction
            # disagreed with the measured miss by ~80mm and made things worse both times.
            _half = 0.35
            _edge = 0.10
            _y_min = RACK_FRONT_Y + _half + _edge
            if dock_ins[1] < _y_min:
                print(f">>> insert dock: floored at chassis clearance -- y {dock_ins[1]:.3f} -> "
                      f"{_y_min:.3f} ({_edge * 1000:.0f}mm from the plate edge to the rack face; "
                      f"the plate reaches {_half:.2f}m forward of the base origin)", flush=True)
                dock_ins = (dock_ins[0], _y_min)
            # Aim the dock for the pose the slide actually ENDS in, after the y-clamp.
            if self.clut is not None and os.environ.get("DOCK_LAT_FIX", "1") == "1":
                try:
                    _qd = np.asarray(self.robot.get_joint_positions(), float)
                    _thd = float(_qd[self.idx["BaseJoint_1"]])
                    _offd = (np.asarray(obj.get_world_poses()[0][0], float)
                             - self._grip_fk(float(_qd[self.idx["ColumnLeftBearingJoint_1"]]),
                                             float(_qd[self.idx["ColumnRightBearingJoint_1"]]),
                                             float(_qd[self.idx["ArmLeftJoint_1"]]), _thd))
                    _mxd, _myd = self.clut["mount"][0], self.clut["mount"][1]
                    _cd, _sd = math.cos(_thd), math.sin(_thd)

                    def _grip_ch(a1_):
                        _pd = self._lut_interp(self.clut["grip"], 0.0, a1_, 3)
                        _pxd, _pyd = float(_pd[0]) - _mxd, float(_pd[1]) - _myd
                        return (_mxd + _cd * _pxd - _sd * _pyd, _myd + _sd * _pxd + _cd * _pyd)

                    _cb, _sb = math.cos(face_south), math.sin(face_south)
                    # forward reach the slide will need from this dock
                    _gxw = float(slot[0]) - float(_offd[0])
                    _gyw = float(slot[1]) - float(_offd[1])
                    _dxd, _dyd = _gxw - dock_ins[0], _gyw - dock_ins[1]
                    _fxd = _cb * _dxd + _sb * _dyd
                    _hi = self._bisect_a1(lambda a: _grip_ch(a)[0] >= _fxd, 0.0, A1_MAX)
                    _pf, _pl = _grip_ch(_hi)
                    # predicted world object x at slide end from this dock
                    _ox_pred = dock_ins[0] + (_cb * _pf - _sb * _pl) + float(_offd[0])
                    _lat_miss = _ox_pred - float(slot[0])
                    if abs(_lat_miss) > 0.005:
                        print(f">>> insert dock: lateral re-aim {_lat_miss * 1000:+.0f}mm (baked "
                              f"grip frame vs our slide-end pose) -- const-z push now lands ON "
                              f"the slot, no turret at the end", flush=True)
                        dock_ins = (dock_ins[0] - _lat_miss, dock_ins[1])
                except Exception as _e_lf:
                    print(f">>> insert dock: lateral re-aim skipped ({_e_lf})", flush=True)

        def place_obj(pos):                                 # collision-off dynamic puppet -> pin pose + velocity
            obj.set_world_poses(positions=np.array([pos]), orientations=np.array([[1, 0, 0, 0]]))
            obj.set_velocities(np.zeros((1, 6)))

        # Retract to a compact CARRY pose before driving. Carrying at the extended grasp config is
        # unstable: even with the joints forced, the loop joint is EXCLUDED from the articulation,
        # so it accumulates stress as the base yaws with the arm out and the base diverges.
        seat_w = np.asarray(obj.get_world_poses()[0][0], float)
        rbx, rby, rbyaw = self.base_ledger()                # hold base FIXED during retract (don't re-read each step)
        if lifted:
            # arm already AT the baked carry pose from the lift replay, object riding in the hand —
            # nothing retracts, nothing sweeps.
            carry_q = np.asarray(self.grip, float).copy()
            scene.set_arm_gains(self.robot, self.names)
            # No carry fold is needed -- the bake already does it. a1 is already retracted here,
            # because the baked lift ends at the compact carry pose, and `_a1_clearing_chassis`
            # floors it anyway.
            _ia1c = self.idx.get("ArmLeftJoint_1")
            if _ia1c is not None:
                print(f">>> carry: a1 {float(carry_q[_ia1c]):.3f} (baked lift already parks the "
                      f"boom compact -- no transport fold needed)", flush=True)
            self._hold(carry_q, int(0.15 / self.dt), pin=(obj, seat_w), kin=True)
            print(f">>> carry: LIFTED in-hand at {np.round(seat_w, 3).tolist()} (baked carry pose)", flush=True)
            self._carry_check(obj, "carry-start")
        else:
            # CARRY arm pose = PARK (q0), teleport-retract; the object sweeps into the parked hand.
            # (Fallback path — used when replay is off or the reach corridor was rack-blocked.)
            carry_q = np.asarray(self.q0, float).copy()
            col_drop = 0.0
            if col_drop and "ColumnLeftBearingJoint_1" in self.idx:
                carry_q[self.idx["ColumnLeftBearingJoint_1"]] -= col_drop
            self.robot.set_joint_velocities(np.zeros(len(self.names)))
            self.robot.set_joint_positions(carry_q)         # teleport-snap to the carry pose (instant, stable)
            scene.set_arm_gains(self.robot, self.names)      # RESET off the pick's hot 3e5/3e4 settle gains to the
            if PERFECT and self._finger_cmd is not None:
                # keep the HOLD stiffness through the carry carry-tuned 2e5/2e4 — carrying with the
                # settle gains diverged the base (THE carry
                self._grip_stiffen()
            # regression)
            self._hold(carry_q, int(0.15 / self.dt), pin=(obj, seat_w), kin=True)
            # Pin reference is the gripper itself: put the object at the SAME small gripper-frame
            # offset captured when it was grasped — between the fingers of the (now retracted) hand,
            # NOT at some fixed point in the air.
            gp, gR = self._grip_frame()
            palm_w = gp + gR @ self._obj_local
            n_ret = int(0.6 / self.dt)                      # sweep the object from the floor-lift seat into the hand
            for k in range(n_ret):
                f = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / n_ret)
                self.set_base(rbx, rby, rbyaw)
                self._force(carry_q)
                self._apply(carry_q)
                place_obj(seat_w + f * (palm_w - seat_w))
                self.world.step(render=not HEADLESS)
            print(f">>> carry: object in-hand at {np.round(palm_w,3).tolist()} (gripper-frame pin)", flush=True)

        if status:
            status(f"carrying to slot {slot_idx}")
        nav_dock = (dock_ins[0], dock_xy[1]) if lifted else dock_xy   # nav to the SAFE standoff row first
        self.drive_to(nav_dock, face_south, on_step=lambda: self._pin_grip(obj),
                      hold_q=carry_q, status=status, vmax=float(os.environ.get("CARRY_VMAX", "1.2")),
                      freeze_arm=True, discs=self._nav_discs(target=obj_idx))
        # Ramp the arrival, like the pick does: `drive_to` exits within GOAL_TOL and a few degrees
        # of yaw, so snapping onto the exact standoff pose in one frame shows as a jump in the turn
        # before the dock.
        _bxs, _bys, _byas = self.base_ledger()
        _dsn = math.hypot(nav_dock[0] - _bxs, nav_dock[1] - _bys)
        _dyw = abs(wrap(face_south - _byas)) if 'wrap' in dir() else abs(face_south - _byas)
        _nsn = max(2, int(0.3 / self.dt))
        for _ks in range(_nsn):
            _fs = 0.5 - 0.5 * math.cos(math.pi * (_ks + 1) / _nsn)
            self.set_base(_bxs + _fs * (nav_dock[0] - _bxs), _bys + _fs * (nav_dock[1] - _bys),
                          _byas + _fs * (face_south - _byas))
            self._pin_grip(obj)
            self.world.step(render=not HEADLESS)
        self.set_base(nav_dock[0], nav_dock[1], face_south)
        print(f">>> place-dock snap: ramped {_dsn * 1000:.0f}mm / "
              f"{abs(face_south - _byas) * 57.3:.1f}deg over {_nsn} steps (was one frame)",
              flush=True)

        self._carry_check(obj, "nav-end")     # did the transport keep the object in the hand?
        goal = np.array([slot[0], slot[1], slot[2] + self.obj_half_h])   # centre = surface + half-height
        if status:
            status(f"placing on slot {slot_idx}")
        if lifted:
            # Set the height BEFORE driving in, not after. The object rides half a metre ahead of
            # the base, so the dock leg pushes it at the rack well before the chassis gets near --
            # and at carry height it does not fit the opening it is being pushed into.
            _slide_z = float(slot[2]) + self.obj_half_h + 0.03
            _ih1p = self.idx["ColumnLeftBearingJoint_1"]
            _ih2p = self.idx["ColumnRightBearingJoint_1"]
            if os.environ.get("PREDOCK_HEIGHT", "1") == "1":
                _bx_ch, _by_ch, _byaw_ch = self.base_ledger()
                _q0p = np.asarray(self.robot.get_joint_positions(), float).copy()
                _dzp = _slide_z - float(np.asarray(obj.get_world_poses()[0][0], float)[2])
                # Level the boom here too. The carry rides noticeably inclined, and freezing that
                # tilt runs the whole insert tilted for no benefit: level reaches further, the tip
                # height barely moves with a1 when horizontal, and the high slot's column-headroom
                # problem dissolves.
                _ia1p = self.idx.get("ArmLeftJoint_1")
                _a1p = float(_q0p[_ia1p]) if _ia1p is not None else 0.10
                _dh0p = float(_q0p[_ih2p]) - float(_q0p[_ih1p])
                _dhfp = 0.0
                _lz = lambda d_: self._lut_z(d_, _a1p) if self.clut is not None else 0.0
                _h1fp = float(_q0p[_ih1p]) + _dzp + (_lz(_dh0p) - _lz(_dhfp))
                if not (0.0 <= _h1fp and _h1fp + _dhfp <= COLUMN_MAX) or self.clut is None:
                    _dhfp = _dh0p          # level boom infeasible here -> keep the carry tilt
                    _h1fp = float(_q0p[_ih1p]) + _dzp
                    print(f">>> pre-dock: boom stays at dh {_dh0p:+.3f} (level would put h1 at "
                          f"{_h1fp:.3f}, out of bounds)", flush=True)
                else:
                    print(f">>> pre-dock: boom -> LEVEL (dh {_dh0p:+.3f} -> {_dhfp:+.3f}) with the "
                          f"height move -- horizontal slide, columns near-static from here",
                          flush=True)
                _qtp = _q0p.copy()
                # ...and zero the turret in the same ramp. The dock rotated the base by the turret
                # angle, so the turret must come to 0 for the reach to land where the dock solved
                # for.
                _ith_p = self.idx.get("BaseJoint_1")
                # ...to 0 only if the dock above actually absorbed the turret (`_th_g` non-zero);
                # when the reach guard declined, the inherited turret must be held instead.
                if _ith_p is not None and abs(_th_g) > 1e-9:
                    _qtp[_ith_p] = 0.0
                _qtp[_ih1p] = float(np.clip(_h1fp, 0.0, COLUMN_MAX))
                _qtp[_ih2p] = float(np.clip(_h1fp + _dhfp, 0.0, COLUMN_MAX))
                _npd = max(1, int(1.2 / self.dt))
                # Pin the base for this move: it swings the columns ~110mm, and `set_base` alone
                # corrects after the fact without zeroing the wheel joints or root velocity, so the
                # arm's reaction shoves and yaws the chassis.
                _pinned_pd = False
                if os.environ.get("PLACE_PIN_CHASSIS", "1") == "1":
                    try:
                        self.pin_chassis(True, "pre-dock")
                        _pinned_pd = True
                    except Exception:
                        pass
                # Keep the gripper parallel THROUGH the ramp, in one motion -- same design as the
                # carry level, and for the same reason: entry-gradient feedback drives the tilt the
                # wrong way as the attitude changes, and chunked stop-and-level pauses the boom
                # visibly at each pause.
                _z_corr, _z_ext = 0.0, 0.0
                _z_samp = max(1, _npd // 10)
                _ckp = max(1, _npd // 6)
                _wl_on = os.environ.get("WRIST_LEVEL", "1") == "1"
                _wip, _wextp, _wstepp = None, 0.0, 0.0
                for _k in range(_npd):
                    if _wl_on and _k % _ckp == 0:
                        # Single-joint probe. Probing all four wrist joints and taking the strongest
                        # was tried, to cover the case where the chosen joint has no authority, and
                        # it regressed badly -- the extra probe steps per ramp disturb more than the
                        # better joint choice recovers.
                        _trkp2 = self._level_gradient(obj)          # invisible probe
                        if _trkp2 is not None:
                            _tnp = self._tilt_deg(obj)
                            _wip = _trkp2[0]
                            _dp2 = float(np.clip(-(_tnp or 0.0) / _trkp2[1], -0.25, 0.25))
                            _wstepp = _dp2 / _ckp
                    _f = 0.5 - 0.5 * math.cos(math.pi * (_k + 1) / _npd)
                    self.set_base(_bx_ch, _by_ch, _byaw_ch)
                    _qp = _q0p + _f * (_qtp - _q0p)
                    if os.environ.get("PREDOCK_Z_CLOSE", "1") == "1":
                        if _k and _k % _z_samp == 0:            # measure: object vs the slide height
                            _z_corr = _slide_z - float(
                                np.asarray(obj.get_world_poses()[0][0], float)[2])
                        _st_z = float(np.clip(_z_corr, -6e-4, 6e-4))   # <=0.6mm/step
                        _z_ext += _st_z
                        _z_corr -= _st_z
                        _qp[_ih1p] = float(np.clip(_qp[_ih1p] + _z_ext, 0.0, COLUMN_MAX))
                        _qp[_ih2p] = float(np.clip(_qp[_ih2p] + _z_ext, 0.0, COLUMN_MAX))
                    if _wip is not None:
                        _wextp += _wstepp
                        _qp[_wip] = float(_qp[_wip]) + _wextp
                    if self.clut is not None:                # passives follow the closure as dh ramps
                        _dh_kp = _dh0p + _f * (_dhfp - _dh0p)
                        for _pn, _pv in self._closure_passives(_dh_kp, _a1p).items():
                            if _pn in self.idx:
                                _qp[self.idx[_pn]] = float(_pv)
                    self._force(_qp)
                    self._apply(_qp)
                    self._pin_grip(obj)
                    self.world.step(render=not HEADLESS)
                # Level the wrist now, inside the pinned window. The boom ramp above pitches the
                # hand ~20deg, and correcting only after the dock leg lets some of it ride in
                # through the drive.
                if os.environ.get("WRIST_LEVEL", "1") == "1":
                    self._level_wrist(obj, "pre-dock")
                if _pinned_pd:
                    self.pin_chassis(False, "pre-dock-end")   # the dock leg drives; wheels must roll
                carry_q = np.asarray(self.robot.get_joint_positions(), float).copy()
                # Close the loop on the height: the open-loop ramp misses by tens of millimetres,
                # because it solves the column move analytically while the boom is levelling at the
                # same time and the object hangs a pitch-dependent distance below the grip.
                if os.environ.get("PREDOCK_Z_CLOSE", "1") == "1":
                    for _it_z in range(1):
                        _oz_now = float(np.asarray(obj.get_world_poses()[0][0], float)[2])
                        _err_z = _slide_z - _oz_now
                        if abs(_err_z) <= 0.003:
                            break
                        _qz = np.asarray(self.robot.get_joint_positions(), float).copy()
                        _z0 = _qz.copy()
                        _qz[_ih1p] = float(np.clip(float(_qz[_ih1p]) + _err_z, 0.0, COLUMN_MAX))
                        _qz[_ih2p] = float(np.clip(float(_qz[_ih2p]) + _err_z, 0.0, COLUMN_MAX))
                        _nz = max(2, int(0.25 / self.dt))
                        for _kz in range(_nz):
                            _fz = 0.5 - 0.5 * math.cos(math.pi * (_kz + 1) / _nz)
                            self.set_base(_bx_ch, _by_ch, _byaw_ch)
                            _qk = _z0 + _fz * (_qz - _z0)
                            self._force(_qk)
                            self._apply(_qk)
                            self._pin_grip(obj)
                            self.world.step(render=not HEADLESS)
                    _oz_fin = float(np.asarray(obj.get_world_poses()[0][0], float)[2])
                    print(f">>> pre-dock height: closed the loop in {_it_z + 1} pass(es) -> "
                          f"{(_oz_fin - _slide_z) * 1000:+.1f}mm of the slide height", flush=True)
                _ozp = float(np.asarray(obj.get_world_poses()[0][0], float)[2])
                # No levelling here -- tried, does not survive. It corrects a couple of degrees and
                # insert-entry then reads the same value it would without it, so the tilt is
                # introduced by the dock leg through some path that holding the arm pose does not
                # cover.
                print(f">>> pre-dock height: object z -> {_ozp:.3f} (slide {_slide_z:.3f}, spans "
                      f"{_ozp - self.obj_half_h:.3f}..{_ozp + self.obj_half_h:.3f}) at the standoff row, "
                      f"after the transport (MuJoCo order) -- the dock leg drives in at this height", flush=True)

            # Second dock leg: kinematically slide the base from the safe nav row to the insert
            # dock, object riding the gripper. Hold the height through the leg.
            self._pin_z = float(self.robot.get_world_pose()[0][2])
            n_sl = max(2, int(abs(dock_ins[1] - nav_dock[1]) / 0.4 / self.dt))
            _wj0 = {n: float(self._wheel_a[n]) for n in self._wheel_a} if hasattr(self, "_wheel_a") else {}
            _wq0 = np.asarray(self.robot.get_joint_positions(), float).copy()
            for k in range(n_sl):
                f = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / n_sl)
                self.set_base(nav_dock[0], nav_dock[1] + f * (dock_ins[1] - nav_dock[1]), face_south)
                self._force(carry_q)
                self._apply(carry_q)
                self._pin_grip(obj)
                self.world.step(render=not HEADLESS)
            # Dock-leg wheel audit: the commanded arc against what the joint actually holds
            # afterwards, which separates "it was never commanded" from "the write did not survive".
            if _wj0:
                _wq1 = np.asarray(self.robot.get_joint_positions(), float)
                _rows = []
                for _n in _wj0:
                    _i = self.idx[_n]
                    _rows.append(f"{_n.split('_wheel')[0]}: cmd {self._wheel_a[_n] - _wj0[_n]:+.2f} "
                                 f"act {float(_wq1[_i]) - float(_wq0[_i]):+.2f}")
                print(f">>> dock-leg wheels over {n_sl} steps ({abs(dock_ins[1] - nav_dock[1]):.2f}m): "
                      f"{' | '.join(_rows)} rad", flush=True)
            # Once docked, the place is the arm's job -- the same rule the grasp follows.
            if os.environ.get("PLACE_PIN_CHASSIS", "1") == "1":
                self.pin_chassis(True, "place-insert")
            # What is actually nearest the rack at the dock, before the arm has moved.
            try:
                _bxd, _byd, _ = self.base_ledger()
                _ocd = np.asarray(obj.get_world_poses()[0][0], float)
                _hpd, _ = self._hand_frame()
                _r_obj = float(KNOWN["object_radius"])
                _plate = (_byd - 0.35) - RACK_FRONT_Y
                _objf = (float(_ocd[1]) - _r_obj) - RACK_FRONT_Y     # object's leading face
                _hand = float(_hpd[1]) - RACK_FRONT_Y
                _who = min((_plate, "chassis plate"), (_objf, "object face"),
                           (_hand, "hand"), key=lambda t: t[0])
                print(f">>> dock clearance to the rack face (before the arm moves): "
                      f"chassis plate {_plate * 1000:+.0f}mm | object face {_objf * 1000:+.0f}mm "
                      f"(centre {(float(_ocd[1]) - RACK_FRONT_Y) * 1000:+.0f}, r "
                      f"{_r_obj * 1000:.0f}mm) | hand {_hand * 1000:+.0f}mm  -> NEAREST is the "
                      f"{_who[1]} at {_who[0] * 1000:+.0f}mm", flush=True)
            except Exception as _e_dc:
                print(f">>> dock clearance readout failed ({_e_dc})", flush=True)
            # No baked insert replay: the architecture was wrong, not the tuning. Its job was to
            # move the arm from the carry pose to the recorded insert pose, but the carry now
            # arrives at the SLIDE height while the bake was recorded from the original carry height
            # -- so it hauls the columns hundreds of millimetres toward a different geometry, the
            # z-hold fights them back, and the boom's dh change tilts the hand.
            if os.environ.get("INSERT_REPLAY", "0") == "1":
                # Baked carry-to-insert replay, object riding the hand. Level as the boom tilts, not
                # after: the low bake ends at a different dh from the carry, so the replay tilts the
                # boom and the hand pitches with it.
                _n_ck = max(1, 16)
                _fr_n = len(self.traj[tname]["frames"])
                _t_ck = float(os.environ.get("INSERT_T", "2.5")) / _n_ck
                for _ck in range(_n_ck):
                    _a = _fr_n * _ck // _n_ck
                    _b = _fr_n * (_ck + 1) // _n_ck if _ck < _n_ck - 1 else _fr_n
                    q_ins = self._play_traj(self.traj[tname], _t_ck, grip_pin=obj,
                                            hold_fingers=True, hold_wrist=True,
                                            frame_range=(_a, _b))
                    if os.environ.get("WRIST_LEVEL", "1") == "1" and _ck < _n_ck - 1:
                        self._level_wrist(obj, f"insert-replay {_ck + 1}/{_n_ck}")
                    # ...and hold the HEIGHT between chunks too: the bake ends at higher columns
                    # than the pre-set slide height, so the replay walks the object up off the line
                    # and the raise walks it back.
                    if os.environ.get("INSERT_LEVEL", "1") == "1":
                        _zerr_ck = _slide_z - float(np.asarray(obj.get_world_poses()[0][0], float)[2])
                        if abs(_zerr_ck) > 5e-3:
                            _q_ck = np.asarray(self.robot.get_joint_positions(), float).copy()
                            _qz_ck = _q_ck.copy()
                            _i1c, _i2c = self.idx["ColumnLeftBearingJoint_1"], self.idx["ColumnRightBearingJoint_1"]
                            _qz_ck[_i1c] = float(np.clip(float(_q_ck[_i1c]) + _zerr_ck, 0.0, COLUMN_MAX))
                            _qz_ck[_i2c] = float(np.clip(float(_q_ck[_i2c]) + _zerr_ck, 0.0, COLUMN_MAX))
                            _nz_ck = max(1, int(0.12 / self.dt))
                            for _kz in range(_nz_ck):
                                _fz = 0.5 - 0.5 * math.cos(math.pi * (_kz + 1) / _nz_ck)
                                _qq = _q_ck + _fz * (_qz_ck - _q_ck)
                                self._force(_qq)
                                self._apply(_qq)
                                self.world.step(render=not HEADLESS)
            elif os.environ.get("WRIST_LEVEL", "1") == "1":
                self._level_wrist(obj, "place-dock")     # one level at the dock; slide keeps it

            q_ins = np.asarray(self.robot.get_joint_positions(), float).copy()
            bx, by, byaw = self.base_ledger()

            def frozen_step(hold):
                self.set_base(bx, by, byaw)
                self._force(hold)
                self._apply(hold)
                self.world.step(render=not HEADLESS)
            if PERFECT:
                # REAL release: the object is IN the hand over the slot — open the fingers and it
                # settles onto the shelf board (active static collider) under gravity.
                self.grip = None
                open_t = {n: v for n, v in KNOWN["fingers_open"].items()}
                self._animate_fingers(q_ins, open_t, 1.2)
                self._finger_cmd = None                      # fingers back under kinematic control
                for _ in range(int(0.5 / self.dt)):          # let the object settle on the board
                    frozen_step(q_ins)
                self.placed.append((obj_idx, np.asarray(obj.get_world_poses()[0][0], float)))
            else:
                self._carry_check(obj, "insert-entry")
                # Level before inserting. By here the object has accumulated 17.5deg of lean from
                # the lift and the dock approach; sliding it into a shelf opening tilted is how a
                # cylinder catches an edge.
                self._level_wrist(obj, "insert-entry")
                q_end, res = q_ins, None
                if os.environ.get("PLACE_ARM_INSERT", "1") == "1":
                    # Two motions: slide in at a clearance height above the shelf, then lower onto
                    # it. One diagonal move would graze the shelf lip on the way in.
                    _cl = 0.03
                    # Solve first, then let the servo trim -- the solve covers the gross motion
                    # including the tilt/height coupling the servo cannot, and the servo closes the
                    # last centimetres.
                    _seq = None
                    if os.environ.get("INSERT_SOLVE", "1") == "1":
                        _seq = self._insert_sequence(obj, goal, _cl, frozen_step)
                    if _seq is None:
                        # no solve (target outside the envelope, or solve disabled) -> the
                        # incremental servo, which is fine inside the reach it can already cover
                        q_end, _ = self._arm_insert(obj, goal + np.array([0.0, 0.0, _cl]),
                                              2.0,
                                              "slide", frozen_step, q_ref=q_ins)
                        q_end, res = self._arm_insert(obj, goal, 1.2,
                                                "lower", frozen_step, q_ref=q_ins)
                    else:
                        q_end, res = _seq
                        # let the servo trim the last centimetres -- it is stable and accurate at
                        # short range (residual ~6mm), it is only the gross coupled move it cannot
                        # do
                        if res > 0.01:
                            q_end, res = self._arm_insert(obj, goal, 1.0,
                                                    "trim", frozen_step, q_ref=q_ins)
                    q_ins = q_end
                    # Last level after the trim: the trim servo moves a1/dh after the final
                    # waypoint's levelling, which can leave the object leaning while the hand is
                    # still carrying it.
                    if os.environ.get("WRIST_LEVEL", "1") == "1":
                        self._level_wrist(obj, "post-trim")
                    self._carry_check(obj, "insert-end")
                # Release: drop the weld before opening, or any pose write below fights a live
                # joint.
                self._unweld_grip()
                if os.environ.get("PLACE_UPRIGHT", "1") == "1":
                    # Leave it where the arm put it. Correct height and tilt only, never the
                    # horizontal error -- the arm's placement IS the result, and moving the object
                    # afterwards would measure our own correction instead of the robot. Two earlier
                    # paths both moved XY, so the object appeared to travel to the slot by itself.
                    # Grade what is left instead of fixing it:
                    #   xy <= 0.03 precise_success
                    #  <= 0.05 precise_marginal
                    #   xy <= 0.12 approx_fallback
                    #  else failed;  z <= 0.02 to pass
                    # Reporting the tier makes a regression visible as a tier change rather than
                    # hiding in a glide.
                    _oc = np.asarray(obj.get_world_poses()[0][0], float)
                    _ztgt = float(slot[2]) + self.obj_half_h
                    _xy = float(np.linalg.norm(_oc[:2] - goal[:2]))
                    _dz = abs(float(_oc[2]) - _ztgt)
                    _tier = ("precise_success" if _xy <= 0.03 and _dz <= 0.02 else
                             "precise_marginal" if _xy <= 0.05 else
                             "approx_fallback" if _xy <= 0.12 else "failed")
                    _n_up = max(1, int(0.4 / self.dt))
                    _kept = np.array([_oc[0], _oc[1], _ztgt])      # RELEASED xy, corrected z
                    for k in range(_n_up):
                        f = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / _n_up)
                        obj.set_world_poses(positions=np.array([_oc + f * (_kept - _oc)]),
                                            orientations=np.array([[1, 0, 0, 0]]))
                        obj.set_velocities(np.zeros((1, 6)))
                        frozen_step(q_ins)
                    print(f">>> place: upright at the RELEASED xy (no teleport) -- "
                          f"xy err {_xy * 1000:.0f}mm, z corrected {_dz * 1000:.0f}mm -> "
                          f"{_tier}  (MuJoCo scale: pass 30 / marginal 50 / fallback 120mm)",
                          flush=True)
                self.grip = None
                self.placed.append((obj_idx, np.asarray(obj.get_world_poses()[0][0], float)))
                # Visible release: fingers open around the placed object
                open_t = {n: v for n, v in KNOWN["fingers_open"].items()}
                self._animate_fingers(q_ins, open_t, 0.8, pin=(obj, goal))
                self._carry_check(obj, "released")
            # Constant-height retract, not a reverse replay: replaying the bake backwards drags the
            # arm back through its tilts right next to the shelf.
            _qr = np.asarray(self.robot.get_joint_positions(), float).copy()
            _i1r = self.idx["ColumnLeftBearingJoint_1"]
            _i2r = self.idx["ColumnRightBearingJoint_1"]
            _iar = self.idx.get("ArmLeftJoint_1")
            _clr = 0.03
            _qtr = _qr.copy()
            _qtr[_i1r] = float(np.clip(float(_qr[_i1r]) + _clr, 0.0, COLUMN_MAX))
            _qtr[_i2r] = float(np.clip(float(_qr[_i2r]) + _clr, 0.0, COLUMN_MAX))
            _nr = max(2, int(0.4 / self.dt))
            for _kr in range(_nr):                          # raise off the placed object first
                _fr = 0.5 - 0.5 * math.cos(math.pi * (_kr + 1) / _nr)
                frozen_step(_qr + _fr * (_qtr - _qr))
            _qr = np.asarray(self.robot.get_joint_positions(), float).copy()
            if _iar is not None:
                _a1r0 = float(_qr[_iar])
                _a1rt = 0.10
                _dhr = float(_qr[_i2r]) - float(_qr[_i1r])
                _nr2 = max(2, int(1.5 / self.dt))
                for _kr in range(_nr2):                     # const-z a1 retract, boom level
                    _fr = 0.5 - 0.5 * math.cos(math.pi * (_kr + 1) / _nr2)
                    _qr[_iar] = _a1r0 + _fr * (_a1rt - _a1r0)
                    for _pnr, _pvr in self._closure_passives(_dhr, float(_qr[_iar])).items():
                        if _pnr in self.idx:
                            _qr[self.idx[_pnr]] = float(_pvr)
                    frozen_step(_qr)
            hold_ret = np.asarray(self.robot.get_joint_positions(), float).copy()
            print(f">>> backout: raised {_clr * 1000:.0f}mm then const-z a1 retract to "
                  f"{float(os.environ.get('CARRY_A1', '0.10')):.2f} -- no reverse replay, no tilt "
                  f"near the shelf", flush=True)
            for _ in range(int(0.2 / self.dt)):
                if not PERFECT:
                    place_obj(goal)
                frozen_step(hold_ret)
            # do NOT snap to tall PARK here: the insert dock is only ~0.5m from the rack and the
            # parked boom clips the upper shelves (base went NaN in the headed run).
            dock_xy = dock_ins                               # retreat starts from the insert dock
        else:
            # object-glide fallback: smooth xyz trajectory from gripper level onto the slot.
            start = np.asarray(obj.get_world_poses()[0][0], float)
            N = int(2.0 / self.dt)
            bx, by, byaw = self.base_ledger()

            def frozen_step(hold):
                self.set_base(bx, by, byaw)
                self._force(hold)
                self._apply(hold)
                self.world.step(render=not HEADLESS)
            for k in range(N):
                f = 0.5 - 0.5 * math.cos(math.pi * min(1.0, k / (N - 1)))   # smoothstep 0..1
                p = start + f * (goal - start)
                obj.set_world_poses(positions=np.array([p]), orientations=np.array([[1, 0, 0, 0]]))
                obj.set_velocities(np.zeros((1, 6)))
                frozen_step(self.q0)
            # RELEASE: no shelf-surface collider at slot heights -> PERSIST the object kinematically
            # (idle_step re-pins every placed object) — it stays put on the shelf,
            # deterministically.
            self.grip = None
            self.placed.append((obj_idx, goal.copy()))
            # Ramp to park, do not teleport. `frozen_step` writes the pose it is given, so passing
            # q0 straight after the release puts the arm at park on the first step and it visibly
            # snaps.
            self._park_ramp(step_fn=lambda _qk: (place_obj(goal), frozen_step(_qk)))
            for _ in range(int(0.4 / self.dt)):
                place_obj(goal)
                frozen_step(self.q0)
            hold_ret = np.asarray(self.q0, float)
        oc = np.asarray(obj.get_world_poses()[0][0], float)
        ok = bool(np.all(np.isfinite(oc)) and np.linalg.norm(oc[:2] - goal[:2]) < 0.15)
        if PERFECT:
            ok = ok and abs(float(oc[2]) - float(goal[2])) < 0.10   # genuine: on the board, not through it
        print(f">>> PLACE obj {obj_idx} -> slot {slot_idx} at {np.round(oc,3).tolist()} ok={ok} lifted={lifted}", flush=True)
        # RETREAT the base north into the open aisle before returning — parked AT the rack dock the
        # base slowly gets pushed south INTO the rack footprint (contact), leaving the next nav
        # start invalid ("No path").
        try:                                                 # the retreat legitimately drives
            self.pin_chassis(False, "place-end")
        except Exception:
            pass
        retreat = (float(slot[0]), RACK_FRONT_Y + 2.0)      # deep north aisle, well clear of the rack
        for k in range(int(1.0 / self.dt)):
            f = min(1.0, k / (1.0 / self.dt - 1))
            rx = dock_xy[0] + f * (retreat[0] - dock_xy[0])
            ry = dock_xy[1] + f * (retreat[1] - dock_xy[1])
            self.set_base(rx, ry, face_south)
            self._force(hold_ret)   # compact carry pose in lifted mode — the TALL   # park boom clips the rack this close
            self._apply(hold_ret)
            self.world.step(render=not HEADLESS)
        if lifted:                                           # rack cleared -> now park (safe in the open aisle)
            # Ramp into park. A single-step write jumps the arm from the compact retreat pose to the
            # tall park pose in one frame, right after the retreat.
            self._park_ramp(base=self.base_ledger(), note=" after the retreat")
            self._force(self.q0)
        return ok
