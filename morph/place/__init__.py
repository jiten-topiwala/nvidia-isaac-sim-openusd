"""Carry to the rack, insert into the slot, release and back out; plus the failure recovery."""
import math
import os

import numpy as np

import scene
from morph.arm.api import cartesian_retreat, move_linear
from morph.config import (HEADLESS, KNOWN, PLACE_DOCK_D, RACK_FRONT_Y, SHELF_SLOTS, COLUMN_MAX,
                          A1_MAX, PLACE_DH_FLOOR, DOCK_EDGE_M, DOCK_TURRET_LIM_M,
                          dock_floor_y, plate_reach)
from morph.gripper.api import GRIPPER_OPEN, open_gripper
from morph.place.insert import InsertStage
from morph.place.recover import RecoverStage


# ESCAPE_FLOOR_M: below this the collider model is not evidence of air, so no pose is commanded
# there. CLEARANCE_EPS_M: `ArmModel.clearance` bisects, so the tolerance must not outrun it.
ESCAPE_FLOOR_M = 0.008
CLEARANCE_EPS_M = 1.0e-4
class PlaceMixin(InsertStage, RecoverStage):
    """Mixed into Demo. Every `self.*` it touches is owned by Demo."""

    def _escape_ok(self, model, q8, obstacles, base, margin, clear_prev):
        """May this pose be commanded? Returns (ok, clearance). Outside `margin`, yes; inside, only
        while RETREATING (no overlap, no step that gives away clearance): the backout starts inside
        the margin by construction. Nothing is commanded below `ESCAPE_FLOOR_M` (8mm, inherited)."""
        clear = float(model.clearance(q8, obstacles, None, base))
        if model.first_hit(q8, obstacles, margin, None, base) is None:
            return True, clear
        if clear < ESCAPE_FLOOR_M:
            return False, clear                  # overlapping, or too close to trust the geometry
        return clear >= clear_prev - CLEARANCE_EPS_M, clear

    def _checked_a1_retract(self, obj_idx, frozen_step, a1_target=None):
        """Retract a1 to the carry pose, refusing before any pose that would collide.

        The open-loop version of this move was the last unchecked motion of the backout. Every step
        is tested against the live obstacle set BEFORE it is commanded; a refusal latches the
        run-scoped safety abort, because a boom left extended in the rack must not be driven out of
        it. Returns True when the carry pose was reached by checked motion."""
        if getattr(self, "_safety_abort", False):
            # `_recover` refuses to move under a standing latch; so does this. A latch means some
            # motion was already disproved, and 456 more commanded poses do not make it safer.
            print(">>> backout: retract refused -- a safety abort is already latched", flush=True)
            return False
        idx_a1 = self.idx.get("ArmLeftJoint_1")
        if idx_a1 is None:
            self._safety_abort = True
            return False
        try:
            from morph.arm.api import MARGIN
            model = self._arm_model()
            q8_idx = [self.idx[n] for n in model.Q8]
            base = self._arm_base_world()
            obstacles = self._arm_obstacles((f"pickup_obj_{obj_idx}",))
        except Exception as exc:                 # noqa: BLE001 -- no checker means no checked motion
            print(f">>> backout: retract checker unavailable ({exc})", flush=True)
            self._safety_abort = True
            return False

        q = np.asarray(self.robot.get_joint_positions(), float).copy()
        # Lift the columns first: the boom leaves the bay inside MARGIN of the battery box and a1 alone
        # cannot get out from under it. Checked step by step, and refused it still tries the retract.
        lift = float(os.environ.get("BACKOUT_LIFT", "0.03"))
        i_c1, i_c2 = self.idx["ColumnLeftBearingJoint_1"], self.idx["ColumnRightBearingJoint_1"]
        if lift > 0.0:
            q_start = q.copy()                      # the ramp interpolates from a FIXED pose: re-reading
            q_lift = q.copy()                       # the robot inside the loop collapses it to a snap
            _cmax = float(globals().get("COLUMN_MAX", 1.40))
            q_lift[i_c1] = float(np.clip(float(q[i_c1]) + lift, 0.0, _cmax))
            q_lift[i_c2] = float(np.clip(float(q[i_c2]) + lift, 0.0, _cmax))
            n_lift = max(2, int(0.4 / self.dt))
            clear_prev = float(model.clearance(
                np.asarray([q[i] for i in q8_idx], float), obstacles, None, base))
            for k in range(n_lift):
                f = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / n_lift)
                q_k = q_start + f * (q_lift - q_start)
                ok, clear_prev = self._escape_ok(
                    model, np.asarray([q_k[i] for i in q8_idx], float),
                    obstacles, base, MARGIN, clear_prev)
                if not ok:
                    print(f">>> backout: checked lift refused at step {k + 1}/{n_lift} "
                          f"(clearance {clear_prev * 1000:.0f}mm) -- retracting from where the arm "
                          f"stands", flush=True)
                    break
                frozen_step(q_k)
            q = np.asarray(self.robot.get_joint_positions(), float).copy()

        a1_from = float(q[idx_a1])
        a1_to = float(os.environ.get("CARRY_A1", "0.10")) if a1_target is None else float(a1_target)
        a1_to = float(np.clip(a1_to, 0.0, float(globals().get("A1_MAX", 0.60))))   # an env var is not a bound
        dh = float(q[self.idx["ColumnRightBearingJoint_1"]]) - float(q[self.idx["ColumnLeftBearingJoint_1"]])
        # Duration from the JOINT, not from a habit: a cosine ramp peaks at pi/2 times its average rate,
        # so a distance that looks slow can still outrun the slide's own velocity limit.
        from morph.arm.api import V_MAX
        v_max = float(V_MAX[model.Q8.index("ArmLeftJoint_1")])
        travel = abs(a1_to - a1_from)
        secs = max(1.5, (travel * math.pi / (2.0 * v_max)) * 1.05) if v_max > 0 else 1.5
        steps = max(2, int(secs / self.dt))
        clear_r = float(model.clearance(
            np.asarray([q[i] for i in q8_idx], float), obstacles, None, base))
        for k in range(steps):
            f = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / steps)
            q_k = q.copy()
            q_k[idx_a1] = a1_from + f * (a1_to - a1_from)
            for name, value in self._closure_passives(dh, float(q_k[idx_a1])).items():
                if name in self.idx:
                    q_k[self.idx[name]] = float(value)
            ok, clear_r = self._escape_ok(
                model, np.asarray([q_k[i] for i in q8_idx], float),
                obstacles, base, MARGIN, clear_r)
            if not ok:
                hit = model.first_hit(np.asarray([q_k[i] for i in q8_idx], float),
                                      obstacles, MARGIN, None, base)
                print(f">>> backout: const-z retract REFUSED before step {k + 1}/{steps} -- {hit} "
                      f"at clearance {clear_r * 1000:.0f}mm, and this step gives clearance away",
                      flush=True)
                self._safety_abort = True
                return False
            frozen_step(q_k)
        print(f">>> backout: checked const-z retract completed -- {travel * 1000:.0f}mm of a1 "
              f"over {secs:.2f}s, peak {travel * math.pi / (2.0 * secs) * 1000:.0f}mm/s "
              f"(limit {v_max * 1000:.0f}mm/s)", flush=True)
        return True

    def place(self, obj_idx, slot_idx, status=None):
        """Carry (pinned) to the rack dock, lift the object onto the slot, release. A thin wrapper
        whose `finally` is the point: `pin_chassis(True)` monkey-patches `world.step`, so a pin left
        installed by an exception in the implementation freezes the chassis for the process."""
        try:
            return self._place_impl(obj_idx, slot_idx, status)
        finally:
            try:
                self.pin_chassis(False, "place-exit")
            except Exception:                 # noqa: BLE001 -- a cleanup must not mask the real
                # exception that is already unwinding through here
                pass

    def _place_impl(self, obj_idx, slot_idx, status=None):
        self._stage = "place"
        obj = self.objs[obj_idx]
        slot = SHELF_SLOTS[slot_idx]
        dock_xy = (float(slot[0]), RACK_FRONT_Y + PLACE_DOCK_D)
        face_south = -math.pi / 2
        # Arm-place mode: pick ended LIFTED at the baked carry pose with a baked trajectory.
        lifted = getattr(self, "lifted", False) and self.traj is not None
        if not lifted:
            # Nothing glides an unlifted object into the slot: the pick has to have lifted it.
            print(f">>> PLACE obj {obj_idx}: not friction-lifted -> abort", flush=True)
            return
        tname = "place_low" if slot[2] < 0.5 else ("place_mid" if slot[2] < 1.0 else "place_high")
        dock_ins = None
        if lifted:
            g = self.traj[tname]["meta"]["grip_base_frame"]
            c, s = math.cos(face_south), math.sin(face_south)
            # The standoff keeps the extended ARM outside the rack volume: reaching the gripper all
            # the way in puts fingers and boom between the shelves, which can NaN the robot.
            stand = 0.20

            def _dock_for(_thv):
                """(dock after the clearance clamp, reach it demands) for a turret angle."""
                _fs = face_south + _thv
                _c, _s = math.cos(_fs), math.sin(_fs)
                _cg, _sg = math.cos(-_thv), math.sin(-_thv)
                _gv = (float(_cg * g[0] - _sg * g[1]), float(_sg * g[0] + _cg * g[1])) + tuple(g[2:])
                _d = [float(slot[0] - (_c * _gv[0] - _s * _gv[1])),
                      float(slot[1] - (_s * _gv[0] + _c * _gv[1]) + stand)]
                # the floor belongs to the yaw this candidate would park at
                _d[1] = max(_d[1], dock_floor_y(_fs))
                return (_d, _gv, _fs, _c, _s,
                        float(math.hypot(slot[0] - _d[0], slot[1] - _d[1])))

            _th_g = 0.0
            if "BaseJoint_1" in self.idx:
                _th_try = float(np.asarray(self.robot.get_joint_positions(),
                                           float)[self.idx["BaseJoint_1"]])
                _, _, _, _, _, _r_keep = _dock_for(0.0)       # inherited turret
                _dz, _gz, _fsz, _cz, _sz, _r_zero = _dock_for(_th_try)
                _lim = DOCK_TURRET_LIM_M
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
            dock_ins = (float(slot[0] - (c * g[0] - s * g[1])),
                        float(slot[1] - (s * g[0] + c * g[1]) + stand))
            _y_min = dock_floor_y(face_south)
            if dock_ins[1] < _y_min:
                print(f">>> insert dock: floored at chassis clearance -- y {dock_ins[1]:.3f} -> "
                      f"{_y_min:.3f} ({DOCK_EDGE_M * 1000:.0f}mm from the plate CORNER to the rack "
                      f"face; the plate reaches {plate_reach(face_south):.4f}m toward it at base "
                      f"yaw {face_south:+.4f})", flush=True)
                dock_ins = (dock_ins[0], _y_min)
            # Aim the dock for the pose the slide actually ENDS in, after the y-clamp.
            if self.clut is not None:
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

        # Retract to a compact CARRY pose before driving: the loop joint is EXCLUDED from the
        # articulation, so carrying extended accumulates stress and the base diverges.
        seat_w = np.asarray(obj.get_world_poses()[0][0], float)
        rbx, rby, rbyaw = self.base_ledger()                # hold base FIXED during retract (don't re-read each step)
        if lifted:
            # Already at the baked carry pose with the object in hand: nothing retracts or sweeps.
            carry_q = np.asarray(self.grip, float).copy()
            if self._drive_on():
                self._grip_stiffen()
            else:
                scene.set_arm_gains(self.robot, self.names)
            # No carry fold: the baked lift already ends at the compact carry pose.
            _ia1c = self.idx.get("ArmLeftJoint_1")
            if _ia1c is not None:
                print(f">>> carry: a1 {float(carry_q[_ia1c]):.3f} (baked lift already parks the "
                      f"boom compact -- no transport fold needed)", flush=True)
            self._hold(carry_q, int(0.15 / self.dt), kin=True)
            print(f">>> carry: LIFTED in-hand at {np.round(seat_w, 3).tolist()} (baked carry pose)", flush=True)
            self._carry_check(obj, "carry-start")

        if status:
            status(f"carrying to slot {slot_idx}")
        nav_dock = (dock_ins[0], dock_xy[1]) if lifted else dock_xy   # nav to the SAFE standoff row first
        self._stage = "carry"
        with self._time_stage('drive_to'):
            self.drive_to(nav_dock, face_south,
                          hold_q=carry_q, status=status, vmax=float(os.environ.get("CARRY_VMAX", "1.2")),
                          freeze_arm=True, discs=self._nav_discs(target=obj_idx), carry_obj=obj)
        self._stage = "place"
        # Ramp the arrival: `drive_to` exits within GOAL_TOL, so a one-frame snap shows as a jump.
        _bxs, _bys, _byas = self.base_ledger()
        _dsn = math.hypot(nav_dock[0] - _bxs, nav_dock[1] - _bys)
        _wheels = self._drive_on()
        if _wheels:
            # Object in hand: no root write -- close the remaining millimetres on the wheels too.
            self._wheel_move(nav_dock[0], nav_dock[1], face_south, 0.3, carry_q, obj, tag="dock-snap")
        else:
            _nsn = max(2, int(0.3 / self.dt))
            for _ks in range(_nsn):
                _fs = 0.5 - 0.5 * math.cos(math.pi * (_ks + 1) / _nsn)
                self.set_base(_bxs + _fs * (nav_dock[0] - _bxs), _bys + _fs * (nav_dock[1] - _bys),
                              _byas + _fs * (face_south - _byas))
                self.world.step(render=not HEADLESS)
            self.set_base(nav_dock[0], nav_dock[1], face_south)
            print(f">>> place-dock snap: ramped {_dsn * 1000:.0f}mm / "
                  f"{abs(face_south - _byas) * 57.3:.1f}deg over {_nsn} steps (was one frame)",
                  flush=True)

        # did the transport keep the object in the hand?
        self._carry_check(obj, "nav-end")
        goal = np.array([slot[0], slot[1], slot[2] + self.obj_half_h])   # centre = surface + half-height
        if status:
            status(f"placing on slot {slot_idx}")
        if lifted:
            # Set the slide height BEFORE the dock leg drives in: the object rides half a metre
            # ahead of the base, and at carry height it does not fit the opening it is pushed into.
            _slide_z = float(slot[2]) + self.obj_half_h + 0.03
            _ih1p = self.idx["ColumnLeftBearingJoint_1"]
            _ih2p = self.idx["ColumnRightBearingJoint_1"]
            _bx_ch, _by_ch, _byaw_ch = self.base_ledger()
            _q0p = np.asarray(self.robot.get_joint_positions(), float).copy()
            _dzp = _slide_z - float(np.asarray(obj.get_world_poses()[0][0], float)[2])
            # Level the boom with the height move: it reaches further and holds z as a1 moves.
            _ia1p = self.idx.get("ArmLeftJoint_1")
            _a1p = float(_q0p[_ia1p]) if _ia1p is not None else 0.10
            _dh0p = float(_q0p[_ih2p]) - float(_q0p[_ih1p])
            _dhfp = 0.0
            _lz = lambda d_: self._lut_z(d_, _a1p) if self.clut is not None else 0.0
            _h1fp = float(_q0p[_ih1p]) + _dzp + (_lz(_dh0p) - _lz(_dhfp))
            # At low slots a LEVEL boom sits inside the battery and nothing catches it -- self-collision is
            # off and this stage has no geometric guard -- so floor the differential when it would hit.
            _ith_p0 = self.idx.get("BaseJoint_1")
            _seatp = _h1fp - (0.03 + int(os.environ.get("PLACE_SEAT_MM", "6")) / 1000)
            if (PLACE_DH_FLOOR > 0.0 and self.clut is not None
                    and any(self._body_hit(_t, _seatp, 0.0, _a1p) for _t in
                            (float(_q0p[_ith_p0]) if _ith_p0 is not None else 0.0, 0.0))):
                _dhfp = float(PLACE_DH_FLOOR)
                _h1fp = float(_q0p[_ih1p]) + _dzp + (_lz(_dh0p) - _lz(_dhfp))
            if not (0.0 <= _h1fp and _h1fp + _dhfp <= COLUMN_MAX) or self.clut is None:
                _dhfp = _dh0p          # level boom infeasible here -> keep the carry tilt
                _h1fp = float(_q0p[_ih1p]) + _dzp
                print(f">>> pre-dock: boom stays at dh {_dh0p:+.3f} (level would put h1 at "
                      f"{_h1fp:.3f}, out of bounds)", flush=True)
            elif _dhfp:
                print(f">>> pre-dock: boom -> dh {_dh0p:+.3f} -> {_dhfp:+.4f}, the CHASSIS-BODY "
                      f"FLOOR (h1 {_h1fp:.3f}): levelling would put Arm_Left_1 inside the "
                      f"battery at the seat -- still a const-z slide, the object does not move",
                      flush=True)
            else:
                print(f">>> pre-dock: boom -> LEVEL (dh {_dh0p:+.3f} -> {_dhfp:+.3f}) with the "
                      f"height move -- horizontal slide, columns near-static from here",
                      flush=True)
            _qtp = _q0p.copy()
            # Zero the turret in the same ramp -- the dock rotated the base by it -- but only if
            # the dock actually absorbed it; otherwise the inherited turret must be held.
            _ith_p = self.idx.get("BaseJoint_1")
            if _ith_p is not None and abs(_th_g) > 1e-9:
                _qtp[_ith_p] = 0.0
            _qtp[_ih1p] = float(np.clip(_h1fp, 0.0, COLUMN_MAX))
            _qtp[_ih2p] = float(np.clip(_h1fp + _dhfp, 0.0, COLUMN_MAX))
            # Time the ramp from the travel, capped at 0.4 m/s peak with the object in hand.
            _dz_pd = abs(float(_qtp[_ih1p]) - float(_q0p[_ih1p]))
            _npd = max(1, int(max(1.2, 1.6 * _dz_pd / 0.4) / self.dt))
            # Pin the base: `set_base` alone corrects late, so the arm's reaction yaws the base.
            _pinned_pd = False
            if os.environ.get("PLACE_PIN_CHASSIS", "1") == "1":
                try:
                    self.pin_chassis(True, "pre-dock")
                    _pinned_pd = True
                except Exception:
                    pass
            _z_corr, _z_ext = 0.0, 0.0
            _z_samp = max(1, _npd // 10)
            with self._time_stage("pre_dock_loop"):
                for _k in range(_npd):
                    _f = 0.5 - 0.5 * math.cos(math.pi * (_k + 1) / _npd)
                    self.set_base(_bx_ch, _by_ch, _byaw_ch)
                    _qp = _q0p + _f * (_qtp - _q0p)
                    if os.environ.get("PREDOCK_Z_CLOSE", "1") == "1":
                        # measure: object vs the slide height
                        if _k and _k % _z_samp == 0:
                            _z_corr = _slide_z - float(
                                np.asarray(obj.get_world_poses()[0][0], float)[2])
                        _st_z = float(np.clip(_z_corr, -6e-4, 6e-4))   # <=0.6mm/step
                        _z_ext += _st_z
                        _z_corr -= _st_z
                        _qp[_ih1p] = float(np.clip(_qp[_ih1p] + _z_ext, 0.0, COLUMN_MAX))
                        _qp[_ih2p] = float(np.clip(_qp[_ih2p] + _z_ext, 0.0, COLUMN_MAX))
                    # passives follow the closure as dh ramps
                    if self.clut is not None:
                        _dh_kp = _dh0p + _f * (_dhfp - _dh0p)
                        _pk = self._closure_passives(_dh_kp, _a1p)
                        for _pn, _pv in _pk.items():
                            if _pn in self.idx:
                                _qp[self.idx[_pn]] = float(_pv)
                        # Counter-rotate the wrist pitch by the LUT pitch change each step:
                        # the hand's attitude must not move as the boom levels.
                        _iwr = self.idx.get("HandBearingJoint_1")
                        # Drives only: this compensation is unvalidated on the kinematic path.
                        if _iwr is not None and self._drive_on():
                            _p0b = float(self._closure_passives(_dh0p, _a1p)["RotationLeftJoint_1"])
                            _dpk = float(_pk["RotationLeftJoint_1"]) - _p0b
                            _qp[_iwr] = float(_q0p[_iwr]) + float(os.environ.get("PLACE_WRIST_COMP", "-1.0")) * _dpk
                    self._force(_qp)
                    self._apply(_qp)
                    self.world.step(render=not HEADLESS)
            # Close the loop on z through `move_linear`: checked per waypoint, and it refuses
            # rather than clipping one column and tilting the held object.
            if os.environ.get("PREDOCK_Z_CLOSE", "1") == "1":
                _oz_now = float(np.asarray(obj.get_world_poses()[0][0], float)[2])
                _err_z = _slide_z - _oz_now
                if abs(_err_z) > 0.003:
                    # `fingers=None` (the SWEPT box) and the grasped-object exemption, as the
                    # insert raise uses for the same motion with the same payload.
                    _mvz = move_linear(self, np.array([0.0, 0.0, _err_z]), "base", secs=0.25,
                                       allow_contact_with=(f"pickup_obj_{obj_idx}",),
                                       fingers=None, tag="pre-dock-z",
                                       on_step=lambda: self.set_base(_bx_ch, _by_ch, _byaw_ch))
                    if not _mvz:
                        # A REFUSAL here is information, not a failure to route around: the trim is a small
                        # correction and the insert re-measures, so it is not worth ending a cycle over.
                        print(f">>> pre-dock height: REFUSED -- {_mvz.reason}; holding, and the "
                              f"dock leg drives in from here", flush=True)
                _oz_fin = float(np.asarray(obj.get_world_poses()[0][0], float)[2])
                print(f">>> pre-dock height: {(_oz_fin - _slide_z) * 1000:+.1f}mm of the slide "
                      f"height (asked {_err_z * 1000:+.1f}mm)", flush=True)
            # Capture the last COMMANDED pose, not the live articulation, so load droop is not latched
            # into the `_wheel_move` target. Off-drives `_drv_prev` is None and the live pose is used.
            _cmd_q = getattr(self, "_drv_prev", None)
            carry_q = (np.asarray(_cmd_q, float).copy() if _cmd_q is not None
                       else np.asarray(self.robot.get_joint_positions(), float).copy())
            _ozp = float(np.asarray(obj.get_world_poses()[0][0], float)[2])
            print(f">>> pre-dock height: object z -> {_ozp:.3f} (slide {_slide_z:.3f}, spans "
                  f"{_ozp - self.obj_half_h:.3f}..{_ozp + self.obj_half_h:.3f}) at the standoff row, "
                  f"after the transport (MuJoCo order) -- the dock leg drives in at this height", flush=True)

            # Second dock leg: nav row -> insert dock, object riding the gripper at the slide height.
            self._carry_check(obj, "pre-dock-end")
            if _pinned_pd:
                # Released HERE, after the z-close loop: released before it, that loop's `set_base`
                # steps teleport the base with the object in hand. The dock leg drives; wheels roll.
                self.pin_chassis(False, "pre-dock-end")
            self._pin_z = float(self.robot.get_world_pose()[0][2])
            n_sl = max(2, int(abs(dock_ins[1] - nav_dock[1]) / 0.4 / self.dt))
            _wj0 = {n: float(self._wheel_a[n]) for n in self._wheel_a} if hasattr(self, "_wheel_a") else {}
            _wq0 = np.asarray(self.robot.get_joint_positions(), float).copy()
            if self._drive_on():
                # Object in hand: the dock leg drives in on the wheels (no root write).
                self._wheel_move(nav_dock[0], dock_ins[1], face_south, 0.4, carry_q, obj, tag="dock-leg")
            else:
                for k in range(n_sl):
                    f = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / n_sl)
                    self.set_base(nav_dock[0], nav_dock[1] + f * (dock_ins[1] - nav_dock[1]), face_south)
                    self._force(carry_q)
                    self._apply(carry_q)
                    self.world.step(render=not HEADLESS)
            # Dock-leg wheel audit: the commanded arc against what the joints actually hold.
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
                _bxd, _byd, _byawd = self.base_ledger()
                _ocd = np.asarray(obj.get_world_poses()[0][0], float)
                _hpd, _ = self._hand_frame()
                _r_obj = float(KNOWN["object_radius"])
                _plate = (_byd - plate_reach(_byawd)) - RACK_FRONT_Y
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
            q_ins = np.asarray(self.robot.get_joint_positions(), float).copy()
            bx, by, byaw = self.base_ledger()

            def frozen_step(hold):
                self.set_base(bx, by, byaw)
                self._force(hold)
                self._apply(hold)
                self.world.step(render=not HEADLESS)
            # Raise / const-z slide / lower, no pins; release; grade the object where it ends up.
            self._carry_check(obj, "insert-entry")
            q_end, res = q_ins, None
            # CALL-scoped, unlike `_safety_abort`: the branches below must tell "THIS insert
            # refused a pose" from "the abort was latched earlier in the run".
            self._insert_refused = False
            _cl = 0.03
            _seq = self._insert_sequence(obj, goal, _cl, frozen_step, obj_idx=obj_idx)
            if self._insert_refused:
                # The insert refused a pose. The servo below drives the SAME corridor with no
                # geometric check at all, so falling into it executes what was just refused.
                print(">>> place: the insert ABORTED -- the servo fallback would re-drive the "
                      "corridor it refused, so it is not run", flush=True)
            elif _seq is None:
                q_end, _ = self._arm_insert(obj, goal + np.array([0.0, 0.0, _cl]), 2.0,
                                            "slide", frozen_step, q_ref=q_ins,
                                            obj_idx=obj_idx)
                q_end, res = self._arm_insert(obj, goal, 1.2, "lower", frozen_step,
                                              q_ref=q_ins, obj_idx=obj_idx)
            else:
                q_end, res = _seq
                if res > 0.01:
                    q_end, res = self._arm_insert(obj, goal, 1.0, "trim", frozen_step,
                                                  q_ref=q_ins, obj_idx=obj_idx)
            q_ins = q_end
            self._carry_check(obj, "insert-end")
            if self._safety_abort:
                # Everything below RELEASES, and opening the hand on a refused insert drops the object
                # between the boards. Leave it held and name WHERE the latch happened.
                _where = ("mid-insert" if self._insert_refused else
                          "before the insert -- the abort was latched EARLIER in this run")
                print(f">>> PLACE obj {obj_idx} -> slot {slot_idx} ABORTED {_where} -- object "
                      f"still held, arm holding the last cleared pose; no release, no backout, "
                      f"no retreat", flush=True)
                # `pin_chassis` works by MONKEY-PATCHING `world.step`, so left installed it freezes the
                # chassis for the life of the process -- including through `_recover`'s `world.reset()`.
                try:
                    self.pin_chassis(False, "place-abort")
                except Exception:
                    pass
                return False
            self.grip = None
            if self._drive_on():
                # Open at the GENTLE finger gains: stiff pads flick the object off the slot.
                self._grip_gentle()
                # Gentle gains only cap the effort; the wound-up targets keep a live preload. Zero
                # the squeeze (targets = measured), settle, THEN open, so nothing pushes the object.
                if getattr(self, "_f_idx", None) is not None and len(self._f_idx):
                    self._finger_cmd = np.asarray(self.robot.get_joint_positions(), float)[self._f_idx].copy()
                    for _ in range(int(0.25 / self.dt)):
                        frozen_step(q_ins)
                    self._carry_check(obj, "zero-squeeze")
            # BEFORE the open: `_apply` overrides
            self._finger_cmd = None
            _ok = {"k": 0}
            # carry check + contacts every 0.25s
            def _open_watch():
                _ok["k"] += 1
                if _ok["k"] % int(0.25 / self.dt) == 0:
                    self._carry_check(obj, f"open+{_ok['k'] * self.dt:.2f}s")
                    # Check for PhysX SLEEP vs genuine balance; a body frozen by PhysX
                    # sleep reads exactly zero velocity. Read-only diagnostic.
                    try:
                        _ov = np.asarray(obj.get_linear_velocities(), float).reshape(-1)[:3]
                        _ow = np.asarray(obj.get_angular_velocities(), float).reshape(-1)[:3]
                        _osl = getattr(obj, "get_sleep_thresholds", None)
                        _osl = (float(np.asarray(_osl(), float).reshape(-1)[0])
                                if _osl is not None else float("nan"))
                        print(f">>>   obj motion at open+{_ok['k'] * self.dt:.2f}s: "
                              f"|v| {float(np.linalg.norm(_ov)) * 1000:.3f}mm/s  "
                              f"|w| {float(np.linalg.norm(_ow)) * 1000:.3f}mrad/s  "
                              f"(sleep threshold {_osl:.4g}) -- exactly 0 means the body is "
                              f"ASLEEP, not balanced", flush=True)
                    except Exception as _e_ov:
                        print(f">>>   obj motion at open+{_ok['k'] * self.dt:.2f}s: unreadable "
                              f"({_e_ov}) -- the sleep hypothesis stays untested this cycle",
                              flush=True)
                    if getattr(self, "_ctc_any", None):
                        print(f">>>   CONTACT during open: {dict(sorted(self._ctc_any.items(), key=lambda kv: 'pickup_obj' in kv[0])[:6])}", flush=True)
                        self._ctc_any = {}
            with self._time_stage('release'):
                open_gripper(self, q_ins, float(os.environ.get("PLACE_OPEN_T", "2.0")),
                             on_step=_open_watch)
            # let the object settle on the board
            for _ in range(int(0.5 / self.dt)):
                frozen_step(q_ins)
            if self._drive_on() and getattr(self, "_f_idx", None) is not None and len(self._f_idx):
                # Verify RELEASE before any backout: pads can stay buried through the gentle open
                # and carry the object away. Measure against the open targets, re-open if stuck.
                _qm = np.asarray(self.robot.get_joint_positions(), float)
                _err = max(abs(float(_qm[self.idx[n]]) - float(v)) for n, v in GRIPPER_OPEN.items() if n in self.idx)
                if _err > float(os.environ.get("RELEASE_ERR_MAX", "0.10")):
                    print(f">>> release: fingers {_err:.2f} rad short of open after the gentle open -> "
                          f"re-opening at the hold effort", flush=True)
                    self._grip_stiffen()
                    open_gripper(self, q_ins, 1.0)
                    for _ in range(int(0.3 / self.dt)):
                        frozen_step(q_ins)
                    _qm = np.asarray(self.robot.get_joint_positions(), float)
                    _err = max(abs(float(_qm[self.idx[n]]) - float(v)) for n, v in GRIPPER_OPEN.items() if n in self.idx)
                print(f">>> release: fingers within {_err:.3f} rad of open", flush=True)
            self._carry_check(obj, "released")
            _oc = np.asarray(obj.get_world_poses()[0][0], float)
            _ztgt = float(slot[2]) + self.obj_half_h
            _xy = float(np.linalg.norm(_oc[:2] - goal[:2]))
            _dz = abs(float(_oc[2]) - _ztgt)
            _tier = ("precise_success" if _xy <= 0.03 and _dz <= 0.02 else
                     "precise_marginal" if _xy <= 0.05 else
                     "approx_fallback" if _xy <= 0.12 else "failed")
            print(f">>> place: upright at the RELEASED xy (no teleport) -- xy err {_xy * 1000:.0f}mm, "
                  f"z err {_dz * 1000:.0f}mm (object z {_oc[2]:.3f} vs {_ztgt:.3f}) -> {_tier}", flush=True)
            self.placed.append((obj_idx, _oc))
            with self._time_stage("backout"):
                # Constant-height retract: reversing the bake drags tilts back past the shelf.
                try:
                    _sep = cartesian_retreat(self, obj_idx)
                except Exception as _e:          # noqa: BLE001 -- see the raise below
                    # The object is already released and in `self.placed`: an escape here unwinds
                    # past `pin_chassis(False)` and leaves the patch installed. A counted degrade.
                    self._fallback("retreat-raised")
                    print(f">>> backout: cartesian retreat RAISED ({type(_e).__name__}: {_e}) -- "
                          f"no checked motion available", flush=True)
                    _sep = False

                if not _sep:
                    if not self._checked_a1_retract(obj_idx, frozen_step):
                        print(">>> backout: the checked const-z retract refused -- the arm cannot "
                              "reach a carry pose by any route this cycle", flush=True)
                hold_ret = np.asarray(self.robot.get_joint_positions(), float).copy()
                for _ in range(int(0.2 / self.dt)):
                    frozen_step(hold_ret)
                # Final attitude check after backout. Note that slip-in-hand reading here
                # reflects hand retreat rather than object movement; check upright instead.
                self._carry_check(obj, "settled")
            # Do NOT snap to tall PARK here: the parked boom clips the upper shelves at this dock.
            dock_xy = dock_ins                               # retreat starts from the insert dock

        oc = np.asarray(obj.get_world_poses()[0][0], float)
        ok = bool(np.all(np.isfinite(oc)) and np.linalg.norm(oc[:2] - goal[:2]) < 0.15)
        ok = ok and abs(float(oc[2]) - float(goal[2])) < 0.10    # genuine: on the board, not through it
        print(f">>> PLACE obj {obj_idx} -> slot {slot_idx} at {np.round(oc,3).tolist()} ok={ok} lifted={lifted}", flush=True)
        # The object is placed and recorded above; only the way OUT failed. Stop before the drive:
        # the base would otherwise carry an un-retracted boom through the rack it just reached into.
        if self._safety_abort:
            print(">>> backout: no checked route to the carry pose -- refusing the retreat drive "
                  "(an un-retracted boom would be dragged through the rack)", flush=True)
            return False
        # RETREAT north into the open aisle before returning: parked at the dock the base drifts
        # south into the rack footprint and the next nav start is invalid ("No path").

        # the retreat legitimately drives
        try:
            self.pin_chassis(False, "place-end")
        except Exception:
            pass
        with self._time_stage("retreat"):
            retreat = (float(slot[0]), RACK_FRONT_Y + 2.0)      # deep north aisle, well clear of the rack
            _wheels_ret = self._drive_on()
            if _wheels_ret:
                self._wheel_move(retreat[0], retreat[1], face_south, 1.0, hold_ret, tag="retreat")
            for k in (range(int(1.0 / self.dt)) if not _wheels_ret else []):
                f = min(1.0, k / (1.0 / self.dt - 1))
                rx = dock_xy[0] + f * (retreat[0] - dock_xy[0])
                ry = dock_xy[1] + f * (retreat[1] - dock_xy[1])
                self.set_base(rx, ry, face_south)
                # compact carry pose: the tall park boom clips the rack here
                self._force(hold_ret)
                self._apply(hold_ret)
                self.world.step(render=not HEADLESS)
        # rack cleared -> now park (safe in the open aisle)
        if lifted:
            # Ramp into park: a single-step write jumps compact -> tall in one frame.
            with self._time_stage("park"):
                self._park_ramp(base=self.base_ledger(), note=" after the retreat",
                                source="post-retreat")
            self._force(self.q0)
        return ok
