"""The constant-height forward reach, driven on the joints rather than through IK.

Part of the close sequence -- see `morph/close/__init__.py` for the stage order and the
shared state object. Split from `enclose.py`. Imports Isaac APIs at module level; only
importable after SimulationApp.
"""
import os

import numpy as np

from morph.config import HEADLESS, KNOWN, COLUMN_MAX


class JointServoStage:
    """One stage of `_close_replay`. Mixed into `CloseMixin`; all `self.*` belong to `Demo`."""

    def _enclose_joint_servo(self, obj, tol):
        """Constant-height forward reach, driven on the joints rather than through IK.

        Nudges the joints directly and reads the real pinch centroid every iteration, so the gap
        between predicted and runtime kinematics on this compliant arm cannot defeat the reach.
        Per iteration:

          * extend a1 (boom reach) toward the object;
          * when a1 saturates, straighten the boom (reduce the |h2-h1| tilt) for extra forward
            reach;
          * hold the pinch height by moving both columns against the measured z error — extending
            a1 on a tilted boom drops z, but tilting down to chase the object's centre instead
            costs forward reach and stalls the servo well short;
          * stop on arrival, on stall (the true reach limit), or on the iteration cap.

        `self._pinch()` is the right reference here: the thumb-vs-bc midpoint, not the mean of the
        three fingertips, which is skewed by the thumb's much wider opening.
        """
        i_a1 = self.idx.get("ArmLeftJoint_1")
        i_h1 = self.idx.get("ColumnLeftBearingJoint_1")
        i_h2 = self.idx.get("ColumnRightBearingJoint_1")
        if i_a1 is None or i_h1 is None or i_h2 is None:
            print(">>> enclose(joint): arm joints not found, skipped", flush=True)
            return
        a1_max = float(os.environ.get("ENCLOSE_A1_MAX", "0.625"))   # the a1 joint stop
        a1_step = 0.010
        tilt_step = 0.008
        z_kp = 0.6
        z_cap = 0.004   # max column move per iter, m
        z_w = 0.35       # height weight vs reach
        n_it = 70
        n_set = max(1, int(0.06 / self.dt))
        # Centre of the measured reachable band for this hand (du -0.07 .. -0.13).
        du_tgt = float(os.environ.get("ENCLOSE_DU", "-0.100"))
        q = np.asarray(self.robot.get_joint_positions(), float).copy()
        oc = np.asarray(obj.get_world_poses()[0][0], float)
        # What constant height means depends on the mode. `hold` keeps the inherited height, which
        # is the baked descent's depth and belongs to the recording's object.
        _zmode = os.environ.get("ENCLOSE_Z_MODE", "hold")
        z_tol = 0.010
        if _zmode == "obj":
            z_hold = float(oc[2]) + 0.0
            print(f">>> enclose(joint): z TARGET {z_hold:.3f} = object centre "
                  f"{float(oc[2]):.3f} (pinch is at {float(self._pinch()[2]):.3f}, "
                  f"{(float(self._pinch()[2]) - z_hold) * 1000:+.0f}mm to descend)", flush=True)
        else:
            z_hold = float(self._pinch()[2])
        e0 = float(np.linalg.norm((oc - self._pinch())[:2]))
        print(f">>> enclose(joint): const-Z servo, pinch->obj {e0 * 1000:.0f}mm  "
              f"(a1 {float(q[i_a1]):.3f}/{a1_max:.3f}, z hold {z_hold:.3f})", flush=True)
        # The forward-at-constant-height direction is a measured joint PAIR, not a single axis.
        # Taken offline at the grasp pose, per 0.04 of joint travel, projected on the mouth axis:
        #     a1 -0.04 -> mouth +49.5mm  z +20.3mm
        #     h1 +0.04 -> mouth +48.7mm  z +64.3mm
        #     h2       -> hysteretic; it drives the four-bar, so it is not a clean DOF
        # Neither holds height alone, but a1- with h1- cancels the rise and leaves ~34mm of mouth
        # reach per 0.04 of a1. It needs h1 off its lower limit to exist at all. Measured offline
        # rather than probed live, because probing perturbs a compliant arm that is already mid-
        # grasp. `MOUTH_INSERT` backs the hand off for the descent and is dropped here -- this
        # advance exists to close exactly that distance.
        if (os.environ.get("MOUTH_INSERT") and
                os.environ.get("ENCLOSE_KEEP_INSERT", "0") != "1"):
            print(f">>> enclose(joint): dropped the descent insert "
                  f"({float(os.environ['MOUTH_INSERT']) * 1000:+.0f}mm) -- the advance closes it",
                  flush=True)
            self._mouth_ins = 0.0
        _open_j1 = os.environ.get("ENCLOSE_OPEN_J1", "")
        if _open_j1:
            _oj = float(_open_j1)
            _n_open = 0
            for _c in "abc":
                _ij = self.idx.get(f"finger_{_c}_joint_1_1")
                if _ij is not None:
                    q[_ij] = _oj
                    _n_open += 1
                # Open the distal joints too: j1 alone leaves j3 on its lower limit, and with j3
                # against its stop the coupled middle phalanx stops tracking for the whole close.
                if os.environ.get("ENCLOSE_OPEN_DISTAL", "1") == "1":
                    _i2o = self.idx.get(f"finger_{_c}_joint_2_1")
                    _i3o = self.idx.get(f"finger_{_c}_joint_3_1")
                    if _i2o is not None:
                        q[_i2o] = 0.0          # j2 lower limit = the vendor open pose
                    if _i3o is not None:
                        q[_i3o] = -0.0523      # j3 UPPER limit = vendor open, i.e. uncurled
            self._finger_cmd = q[self._f_idx].copy()
            self._force(q)
            self._apply(q)
            for _ in range(int(0.4 / self.dt)):
                self.world.step(render=not HEADLESS)
            print(f">>> enclose(joint): jaw re-opened to j1 {_oj:+.2f} on {_n_open} fingers before "
                  f"the advance (the replay had curled them to the baked grip pose)", flush=True)
        r_h1 = 0.316     # h1 per unit a1, cancels d.z
        mm_per = 34.1  # mouth mm per 0.04 of a1
        step_mm = 4.0 # mouth mm per iteration
        # Headroom for the tilt -- with h1 on its stop only one column can move. Keep the lift
        # small: it raises the pinch faster than the extra tilt lowers it.
        h1_need = 0.030
        if float(q[i_h1]) < h1_need:
            _lift = h1_need - float(q[i_h1])
            q[i_h1] = float(np.clip(q[i_h1] + _lift, 0.0, COLUMN_MAX))
            q[i_h2] = float(np.clip(q[i_h2] + _lift, 0.0, COLUMN_MAX))
            self._force(q); self._apply(q)
            for _ in range(n_set):
                self._fN_step = {}
                self.world.step(render=not HEADLESS)
            print(f">>> enclose(joint): lifted columns {_lift * 1000:.0f}mm so h1 has room to "
                  f"descend (it was on its lower limit, nothing could cancel the z rise)",
                  flush=True)
        # Tilt before reaching: from a flat boom the hand goes over the top of the object. Must run
        # after the headroom lift, or the differential has no vertical authority.
        if os.environ.get("ENCLOSE_LOWEST", "1") == "1":
            try:
                self._reach_lowest(obj)                      # STEP 1: a1 + tilt -> lowest pose
            except Exception as _e_rl:
                print(f">>> reach-lowest: failed ({_e_rl})", flush=True)
        elif os.environ.get("ENCLOSE_POLAR", "0") == "1":
            try:
                self._reach_polar(obj)
            except Exception as _e_rp:
                print(f">>> reach-polar: failed ({_e_rp})", flush=True)
        elif os.environ.get("TILT_FIRST", "1") == "1":
            try:
                self._tilt_to_obj_z(obj)
            except Exception as _e_tz:
                print(f">>> tilt-to-objz: failed ({_e_tz})", flush=True)
        # adaptive gain state: k_du = d(du)/d(a1), learned online, seeded from the probe
        k_du = -0.85   # m of du per unit a1
        k_a = 0.5
        dq_cap = 0.010
        du_prev, last_d = None, None
        best, stall = None, 0
        i_th = self.idx.get("BaseJoint_1")
        dv_hold = os.environ.get("ENCLOSE_DV_HOLD", "1") == "1"
        _z_drop_max = 0.010
        _oz0_e = float(np.asarray(obj.get_world_poses()[0][0], float)[2])
        _dd0 = self._du_dv(obj)
        dv0_e = _dd0[1] if _dd0 else 0.0
        # dv target, not just dv hold. Holding dv wherever it was at entry leaves no way to ask for
        # a jaw-centred object, and an object well off the jaw axis is only ever reached by one
        # finger.
        _dv_tgt_env = os.environ.get("ENCLOSE_DV_TGT", "")
        if _dv_tgt_env:
            dv0_e = float(_dv_tgt_env)
            print(f">>> enclose(joint): dv TARGET {dv0_e * 1000:+.1f}mm "
                  f"(entry dv {(_dd0[1] if _dd0 else 0.0) * 1000:+.1f}mm)", flush=True)
        _k_dv = -0.20   # m of dv per rad of th, learned
        _th_cap = float(os.environ.get("ENCLOSE_TH_CAP", "0.010"))
        _dv_prev, _last_th = None, 0.0
        _k_ff, _last_adv, _last_ff = None, 0.0, 0.0   # advance->jaw coupling, learned
        # Per-link contact ledger, so a body riding on the object -- the palm underside, a knuckle
        # passing over -- can be named rather than guessed.
        self._ctc = {}
        self._ctcN = {}
        _named = False
        _q_prev_e = q.copy()           # for the velocity-consistent kinematic write
        _gap_stop = 0.010
        # Read once, before the loop: both arm-stop guards use it, and the in-loop copy sits inside
        # a try/except that would leave it undefined.
        _stop_on = os.environ.get("ENCLOSE_STOP_ON", "wrap")
        for it in range(n_it):
            # `_ctcN` accumulates impulse, so it must be cleared per iteration -- against the whole
            # servo's history the press guard below stays armed permanently after any incidental
            # touch.
            self._ctcN = {}
            _dd = self._du_dv(obj)
            if _dd is None:
                # Every other exit here prints a reason, so this one must too, or a failed
                # measurement is indistinguishable from normal completion.
                print(f">>> enclose(joint): ABORTED at iter {it} -- _du_dv returned None "
                      f"(object pose or hand frame unavailable). The advance stopped here; this is "
                      f"NOT arrival.", flush=True)
                break
            du, dv = _dd
            # Stop if the object is being pressed down: a grasp cannot begin on an object being
            # pushed into the floor, so the forward reach is done the moment it starts loading it.
            _oz_now = float(np.asarray(obj.get_world_poses()[0][0], float)[2])
            if (_oz0_e - _oz_now) > _z_drop_max:
                print(f">>> enclose(joint): STOP -- object pressed down "
                      f"{(_oz0_e - _oz_now) * 1000:.1f}mm (limit {_z_drop_max * 1000:.0f}mm) at "
                      f"du {du * 1000:+.1f}mm dv {dv * 1000:+.1f}mm", flush=True)
                break
            # du is negative outside the band, so the servo brings it UP. Height is part of arrival
            # too -- du/dv are the XY window only, and arriving on them while high closes on empty
            # air.
            _zres = float(self._pinch()[2]) - z_hold
            if du >= du_tgt and (_zmode != "obj" or abs(_zres) <= z_tol):
                print(f">>> enclose(joint): arrived in the contact band, du {du * 1000:+.1f}mm "
                      f"dv {dv * 1000:+.1f}mm (target du <= {du_tgt * 1000:+.0f}mm) "
                      f"after {it} iters", flush=True)
                break
            need_mm = (du_tgt - du) * 1000.0            # +ve: how much closer du must come
            if best is None or need_mm < best - 1.0:
                best, stall = need_mm, 0
            else:
                stall += 1
                if stall >= 8:
                    print(f">>> enclose(joint): STALLED, du {du * 1000:+.1f}mm "
                          f"(h2 {float(q[i_h2]):.3f}, true reach limit)", flush=True)
                    break
            # Adapt from the servo's own motion: the joint-to-pinch map on this four-bar is strongly
            # pose-dependent, so a fixed sign or scale is valid only where it was measured.
            if last_d is not None and abs(last_d) > 1e-9 and du_prev is not None:
                k_obs = (du - du_prev) / last_d          # m of du per unit a1
                if abs(k_obs) > 1e-4:                    # ignore steps that moved nothing
                    k_du = (1.0 - k_a) * k_du + k_a * k_obs
            want_du = du_tgt - du                        # negative: du must become more negative
            d_a1 = float(np.clip(want_du / k_du, -dq_cap, dq_cap))
            if it == 0:
                d_a1 = -dq_cap * 0.5                     # exploratory first step; k is unknown
            du_prev, last_d = du, d_a1
            _a1_prev = float(q[i_a1])
            q[i_a1] = float(np.clip(q[i_a1] + d_a1, 0.0, a1_max))
            _a1_moved = abs(float(q[i_a1]) - _a1_prev) > 1e-6
            _rh = r_h1
            # Optional: let a1 supply the remaining descent by scaling out its z cancellation. Off
            # by default -- this is a constant-height slide, and relaxing it trades height against
            # reach.
            if os.environ.get("ENCLOSE_A1_DESCEND", "0") == "1":
                _zerr_now = float(self._pinch()[2]) - z_hold      # >0 -> still above the target
                _band = 0.010
                if _zerr_now > 0.0:
                    _rh = r_h1 * float(np.clip(1.0 - _zerr_now / max(1e-6, _band), 0.0, 1.0))
            # When a1 saturates, straighten the boom: reduce the tilt and lower both columns
            # together, so the hand reaches further at the same height -- about 84mm of reach for
            # 85mm of column drop.
            _straightened = False
            if (not _a1_moved) and need_mm > 0.0 \
                    and os.environ.get("ENCLOSE_STRAIGHTEN", "1") == "1":
                _dh_now = float(q[i_h2]) - float(q[i_h1])
                _dh_new = _dh_now - tilt_step
                _off_s = float(self._pinch()[2]) - self._fk_z(
                    float(q[i_h1]), float(q[i_h2]), float(q[i_a1]))
                _h1_new = (z_hold - _off_s
                           - float(self._lut_interp(self.clut["grip"], _dh_new,
                                                    float(q[i_a1]), 3)[2])
                           + self.clut["h1_ref"])
                if _dh_new < float(self.clut["dh"][0]):
                    if it % 10 == 0:
                        print(f">>> enclose(joint): boom straight (dh {_dh_new:+.3f} at the LUT "
                              f"floor), du {du * 1000:+.1f}mm", flush=True)
                elif _h1_new < 0.0 or (_h1_new + _dh_new) < 0.0:
                    if it % 10 == 0:
                        print(f">>> enclose(joint): a1 saturated; holding z at dh {_dh_new:+.3f} "
                              f"needs h1 {_h1_new:.3f} -- COLUMN STOP, du {du * 1000:+.1f}mm",
                              flush=True)
                else:
                    q[i_h1] = float(np.clip(_h1_new, 0.0, COLUMN_MAX))
                    q[i_h2] = float(np.clip(_h1_new + _dh_new, 0.0, COLUMN_MAX))
                    _straightened = True
                    if it % 10 == 0:
                        print(f">>> enclose(joint): a1 saturated -> STRAIGHTEN boom "
                              f"{_dh_now:+.3f} -> {_dh_new:+.3f}, both columns to h1 "
                              f"{_h1_new:.3f} (holds z {z_hold:.3f}), du {du * 1000:+.1f}mm",
                              flush=True)
            if not _straightened:
                q[i_h1] = float(np.clip(q[i_h1] + d_a1 * _rh, 0.0, COLUMN_MAX))  # h1 tracks a1, cancels z
            # Hold dv while du closes. The forward reach is not lateral-neutral -- dv walks tens of
            # millimetres over the du it closes, pushing the object off the jaw axis.
            if i_th is not None and dv_hold and _k_ff is not None and abs(d_a1) > 1e-9:
                _dth_ff = float(np.clip(-(_k_ff * d_a1) / _k_dv, -_th_cap, _th_cap))
                q[i_th] = float(q[i_th] + _dth_ff)
                _ff_applied = _dth_ff
            else:
                _ff_applied = 0.0
            if i_th is not None and dv_hold:
                if _dv_prev is not None and abs(_last_th) > 1e-9:
                    _k_obs = (dv - _dv_prev) / _last_th
                    if abs(_k_obs) > 1e-4:
                        _k_dv = 0.5 * _k_dv + 0.5 * _k_obs
                _d_th = float(np.clip((dv0_e - dv) / _k_dv, -_th_cap, _th_cap))
                if it == 0:
                    _d_th = _th_cap * 0.25          # small probe; the gain is not yet known
                # the advance's own coupling into dv, separated from the th correction applied here
                # -- that is what has to be fed forward next step
                if _dv_prev is not None and abs(_last_adv) > 1e-9:
                    _obs_ff = ((dv - _dv_prev) - _k_dv * (_last_th + _last_ff)) / _last_adv
                    _k_ff = _obs_ff if _k_ff is None else (0.5 * _k_ff + 0.5 * _obs_ff)
                _dv_prev, _last_th, _last_adv, _last_ff = dv, _d_th, d_a1, _ff_applied
                q[i_th] = float(q[i_th] + _d_th)
            # residual height trim, rate-limited -- the ratio cancels z to first order, not exactly
            z_err = z_hold - float(self._pinch()[2])
            _dz = float(np.clip(z_w * z_err, -z_cap, z_cap))
            q[i_h1] = float(np.clip(q[i_h1] + _dz, 0.0, COLUMN_MAX))
            q[i_h2] = float(np.clip(q[i_h2] + _dz, 0.0, COLUMN_MAX))
            # The kinematic arm write is load-bearing and dangerous, in that order. It cannot be
            # dropped: the drives alone cannot hold the four-bar, so a drive-only servo stalls and
            # drifts backwards.
            _gmin_e = None
            try:
                _oc_e = np.asarray(obj.get_world_poses()[0][0], float)
                _oR_e = self._obj_R(obj)
                # Which surface stops the advance. `_wrap_gap` is a minimum over link_1/2/3, so it
                # returns the nearest knuckle, which on this hand leads the pad by ~30mm -- a
                # knuckle-based stop halts the advance with the gripping surfaces still far out.
                _stop_on = os.environ.get("ENCLOSE_STOP_ON", "wrap")
                _gs_e = [(self._finger_surface_gap(f_, _oc_e, KNOWN["object_radius"],
                                                   self.obj_half_h, _oR_e)
                          if _stop_on == "pad" else
                          self._wrap_gap(f_, _oc_e, KNOWN["object_radius"], self.obj_half_h, _oR_e))
                         for f_ in "abc"]
                _pg_e = self._finger_surface_gap("palm", _oc_e, KNOWN["object_radius"],
                                                 self.obj_half_h, _oR_e)
                _gmin_e = min([g for g in _gs_e if g is not None] + [_pg_e])
            except Exception:
                _gmin_e = None
            if _gmin_e is not None and _gmin_e < _gap_stop:
                print(f">>> enclose(joint): ARM STOP -- closest body {_gmin_e * 1000:.1f}mm from the "
                      f"object (margin {_gap_stop * 1000:.0f}mm) at du {du * 1000:+.1f}mm. The "
                      f"fingers close the rest; a kinematic arm write must not land inside it.",
                      flush=True)
                break
            # baselines for attributing the settle's motion to the HAND or the OBJECT
            _h0_tmp, _ = self._hand_frame()
            _hand0_s = np.asarray(_h0_tmp, float)
            _obj0_s = np.asarray(obj.get_world_poses()[0][0], float)
            if os.environ.get("ENCLOSE_QV", "1") == "1":
                # Scale the velocity by the time the move takes: `q_prev` updates per iteration
                # while the settle runs n_set steps, so a single dt overstates the rate n_set-fold
                # and the hand lurches.
                _qv = (q - _q_prev_e) / (self.dt * n_set)
                # Optional ceiling on the declared velocity. Off by default: it reduces the lurch
                # but makes the object eject harder, because a lower declared velocity makes the
                # link look more like a wall.
                _vcap = 1e9
                _vmax = float(np.max(np.abs(_qv))) if _qv.size else 0.0
                if _vmax > _vcap:
                    _qv = _qv * (_vcap / _vmax)
            else:
                _qv = None
            # Ramp the iteration rather than jumping it -- one write plus idle settle steps is a
            # visible stutter, where interpolating gives one write per physics step and the guards
            # see every one.
            _q_from_e = _q_prev_e.copy()
            _q_prev_e = q.copy()
            # Both guards run per physics step, not per iteration -- contact happens inside the
            # settle.
            _brk_z = False
            for _s_i in range(n_set):
                self._fN_step = {}
                _q_i = _q_from_e + (q - _q_from_e) * (float(_s_i + 1) / n_set)
                if _qv is not None:
                    self._force(_q_i, _qv)
                else:
                    self._force(_q_i)
                self._apply(_q_i)
                self.world.step(render=not HEADLESS)
                # Object-motion stop: the clearance guard protects the arm from landing inside the
                # object, not the object from a body the gap model reads as clear but that is
                # already touching.
                try:
                    _oc_m = np.asarray(obj.get_world_poses()[0][0], float)
                    if float(np.linalg.norm(_oc_m[:2] - _obj0_s[:2])) > 0.004:
                        print(f">>> enclose(joint): ARM STOP -- OBJECT moved "
                              f"{float(np.linalg.norm(_oc_m[:2] - _obj0_s[:2])) * 1000:.1f}mm during "
                              f"the settle; the arm is touching it. Holding here.", flush=True)
                        _brk_z = True
                        break
                except Exception:
                    pass
                if os.environ.get("LOWEST_TRACE", "0") == "1":
                    self._fw_tr = getattr(self, "_fw_tr", [])
                    _pf = self._pinch()
                    self._fw_tr.append((float(_pf[0]), float(_pf[1]), float(_pf[2])))
                _oc_s = np.asarray(obj.get_world_poses()[0][0], float)
                _oR_s = self._obj_R(obj)
                try:
                    # Same pad-versus-knuckle choice as the pre-step guard above; this is the copy
                    # that actually fires, so both must be kept in step.
                    _gs = [(self._finger_surface_gap(f_, _oc_s, KNOWN["object_radius"],
                                                     self.obj_half_h, _oR_s)
                            if _stop_on == "pad" else
                            self._wrap_gap(f_, _oc_s, KNOWN["object_radius"], self.obj_half_h,
                                           _oR_s))
                           for f_ in "abc"]
                    _gm = min([g for g in _gs if g is not None])
                except Exception:
                    _gm = None
                _hp_s, _ = self._hand_frame()
                if _gm is not None and _gm < _gap_stop:
                    _dh_s = float(np.linalg.norm(np.asarray(_hp_s, float) - _hand0_s)) * 1000.0
                    _do_s = float(np.linalg.norm(_oc_s - _obj0_s)) * 1000.0
                    print(f">>> enclose(joint): ARM STOP at step {_s_i} of the settle -- closest "
                          f"{_gm * 1000:.1f}mm (margin {_gap_stop * 1000:.0f}mm). Since this "
                          f"iteration began: HAND moved {_dh_s:.1f}mm, OBJECT moved {_do_s:.1f}mm.",
                          flush=True)
                    _brk_z = True
                    break
                if (_oz0_e - float(_oc_s[2])) > _z_drop_max:
                    _hand_ns = sum(v for kk, v in getattr(self, "_ctcN", {}).items()
                                   if "floor" not in kk)
                    if _hand_ns > 0.05:
                        # Name the pressing links -- an early stop that does not say why reads as
                        # arrival.
                        _who = sorted(((kk, vv) for kk, vv in getattr(self, "_ctcN", {}).items()
                                       if "floor" not in kk), key=lambda kv: -kv[1])[:3]
                        print(f">>> enclose(joint): STOP at settle step {_s_i} -- the advance is "
                              f"PRESSING the object down {(_oz0_e - float(_oc_s[2])) * 1000:.1f}mm "
                              f"(limit {_z_drop_max * 1000:.0f}mm) with {_hand_ns:.3f}Ns of hand "
                              f"contact, at du {du * 1000:+.1f}mm. Not arrival. Pressing links: "
                              f"{ {k2: round(v2, 4) for k2, v2 in _who} }", flush=True)
                        _brk_z = True
                        break
            if _brk_z:
                break
            if not _named:
                # Non-floor contacts only: the floor is always present, so including it names the
                # floor rather than the link that is pressing.
                _hot = {kk: vv for kk, vv in getattr(self, "_ctcN", {}).items()
                        if vv > 1e-4 and "floor" not in kk}
                if _hot:
                    _top = sorted(_hot.items(), key=lambda kv: -kv[1])[:4]
                    print(f">>>   reach contact: { {k2: round(v2, 4) for k2, v2 in _top} }Ns "
                          f"at du {du * 1000:+.1f}mm", flush=True)
                    _named = True
            if _brk_z:
                _zn = float(np.asarray(obj.get_world_poses()[0][0], float)[2])
                _ddz = self._du_dv(obj)
                print(f">>> enclose(joint): STOP (mid-step) -- object pressed down "
                      f"{(_oz0_e - _zn) * 1000:.1f}mm at "
                      f"du {(_ddz[0] * 1000 if _ddz else float('nan')):+.1f}mm", flush=True)
                break
            # Re-seed the height setpoint from the live pinch after the first iteration -- but never
            # in `obj` mode, where it would overwrite the object-centre target and make arrival
            # trivially true.
            if it == 0 and _zmode != "obj":
                z_hold = float(self._pinch()[2])
            if it % 5 == 0:
                print(f">>>   enclose(joint)[{it:3d}]: du {du * 1000:+6.1f}mm dv {dv * 1000:+6.1f}mm "
                      f"h2 {float(q[i_h2]):.3f} z {float(self._pinch()[2]):.3f}", flush=True)
        oc = np.asarray(obj.get_world_poses()[0][0], float)
        if os.environ.get("LOWEST_TRACE", "0") == "1" and getattr(self, "_fw_tr", None):
            _t = np.asarray(self._fw_tr, float)
            _dxy = np.linalg.norm(np.diff(_t[:, :2], axis=0), axis=1) * 1000.0
            _dz = np.abs(np.diff(_t[:, 2])) * 1000.0
            print(f">>> enclose(joint): FORWARD SMOOTHNESS over {len(_t)} steps -- xy step max "
                  f"{_dxy.max():.2f}mm p99 {np.percentile(_dxy, 99):.2f}mm mean {_dxy.mean():.3f}mm | "
                  f"z wander max {_dz.max():.2f}mm, z range {(_t[:, 2].max() - _t[:, 2].min()) * 1000:.1f}mm "
                  f"(const-Z held)", flush=True)
            self._fw_tr = []
        print(f">>> enclose(joint): done, pinch->obj "
              f"{float(np.linalg.norm((oc - self._pinch())[:2])) * 1000:.0f}mm  "
              f"pinch z {float(self._pinch()[2]):.3f} (held {z_hold:.3f})", flush=True)
