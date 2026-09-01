"""Lowest-pose solves before the forward reach: get low, tilt-and-reach polar solve, and
the tilt-to-object-height entry.

Part of the close sequence -- see `morph/close/__init__.py` for the stage order and the
shared state object. Split from `enclose.py`. Imports Isaac APIs at module level; only
importable after SimulationApp.
"""
import os

import numpy as np

from morph.config import HEADLESS, COLUMN_MAX, A1_MAX


class LowestStage:
    """One stage of `_close_replay`. Mixed into `CloseMixin`; all `self.*` belong to `Demo`."""

    def _reach_lowest(self, obj, label="reach-lowest"):
        """Get the gripper as low as it will go, before the horizontal reach.

        Order matters: retract a1, tilt, then extend. The column differential only has a vertical
        lever while the boom is short — at a long a1 the same differential raises the hand instead
        of lowering it.

        The gradient is evaluated on the baked closure map (`_fk_z`), not by moving the joint and
        reading back. The arm is a forced linkage whose passive joints are interpolated from
        `usd/_closure_lut.json`, so a physical step reads a transient rather than the settled pose.
        One bounded Newton step is applied per iteration, ramped, and reverted if the live pinch
        did not actually improve. One-sided: a pinch already at or below the object needs no tilt.

        Note that the "does not hit the robot body" bound is enforced geometrically, not by
        contact: articulation self-collision is disabled at load (it produces phantom
        finger-vs-finger contacts), so `LOWEST_A1_MAX` and the LUT dh bounds stand in for it.
        """
        i_h1 = self.idx.get("ColumnLeftBearingJoint_1")
        i_h2 = self.idx.get("ColumnRightBearingJoint_1")
        i_a1 = self.idx.get("ArmLeftJoint_1")
        if i_h1 is None or i_h2 is None or i_a1 is None:
            return
        # Measured limit: above a1 ~0.50 the joint stops tracking and the pinch jumps in one
        # increment, a kinematic discontinuity in the forced four-bar.
        a1_max = A1_MAX
        # Retract only as far as the chassis allows -- not a constant, since at a large dh the
        # retracted gripper sits inside the collision plate. `_clears_chassis` checks the (dh, a1)
        # pair.
        a1_tilt = float(os.environ.get("LOWEST_A1_TILT", "0.15"))
        # Bounded by the closure table (dh -0.100 .. 0.300), outside which the passives extrapolate
        # into a bad articulation state -- and tighter still, because the boom grazes the chassis
        # plate near 0.13.
        dh_lo = -0.100
        dh_hi = 0.15
        z_tol = 0.010
        max_it = 8
        oc = np.asarray(obj.get_world_poses()[0][0], float)
        z_tgt = float(oc[2]) + 0.0
        q = np.asarray(self.robot.get_joint_positions(), float).copy()
        pz0 = float(self._pinch()[2])
        print(f">>> {label}: pinch {pz0:.3f} -> target {z_tgt:.3f} ({(pz0 - z_tgt) * 1000:+.0f}mm); "
              f"retract a1 -> tilt -> a1 forward", flush=True)
        if (pz0 - z_tgt) <= z_tol:
            print(f">>> {label}: pinch at/below obj-Z -- no tilt needed", flush=True)
            return

        # ---- One move: a1 forward-and-down, columns trimmed from the measured a1 gradient ------
        # The descent is one continuous motion -- the boom straightens while both columns sink,
        # sliding the hand forward and down onto the object.
        _a1_now = float(q[i_a1])
        _dh_now = float(q[i_h2] - q[i_h1])
        # Cap below the linkage discontinuity: the four-bar jumps around a1 0.48-0.55 depending on
        # dh and th, and the pinch flies up hard enough to knock the object down. The columns own
        # the last mm.
        _a1_end = min(a1_max, 0.47)
        _a1_safe = self._a1_clearing_chassis(_dh_now, a1_max=a1_max)
        if _a1_safe is not None and _a1_end < _a1_safe:
            _a1_end = _a1_safe               # never finish inside the chassis plate
        print(f">>> {label}: one move -- a1 {_a1_now:.3f} -> {_a1_end:.3f} at dh {_dh_now:+.3f}, "
              f"columns trimmed from the MEASURED dz/da1 (gap {(pz0 - z_tgt) * 1000:+.0f}mm)",
              flush=True)
        # One write per physics step. Fewer increments with settle steps between them let the arm
        # sag and then teleport to the next, which reads as a visible stutter.
        _N = 800
        # Probe long enough to beat the settle, and require a real a1 displacement behind it.
        _probe_frac = 0.40
        _da1_min = 0.040
        _n_probe = max(4, int(_N * _probe_frac))
        if abs(_a1_end - _a1_now) * _probe_frac < _da1_min:      # short move -> probe more of it
            _probe_frac = min(0.7, _da1_min / max(1e-6, abs(_a1_end - _a1_now)))
            _n_probe = max(4, int(_N * _probe_frac))
        _h1_0, _h2_0 = float(q[i_h1]), float(q[i_h2])
        # Let the arm settle first: right after the descent the pinch is still sinking on its own,
        # and that residual motion lands in the gradient's numerator.
        _q_settle = np.asarray(self.robot.get_joint_positions(), float).copy()
        # Pin the base across the settle too -- an extended, tilted arm's reaction rolls a free
        # chassis forward, which also biases every "already over the object" reading taken
        # afterwards.
        try:
            _bx_s, _by_s, _byaw_s = self.base_ledger()
        except Exception:
            _bx_s = None
        for _ in range(int(0.4 / self.dt)):
            if _bx_s is not None:
                self.set_base(_bx_s, _by_s, _byaw_s)
            self._force(_q_settle)
            self._apply(_q_settle)
            self.world.step(render=not HEADLESS)
        _z_0 = float(self._pinch()[2])
        _d_entry = float(np.linalg.norm(
            (np.asarray(obj.get_world_poses()[0][0], float) - self._pinch())[:2]))
        # Choose the actuator from the remaining XY gap, measured after the settle.
        _xy_gap = _d_entry
        q = np.asarray(self.robot.get_joint_positions(), float).copy()
        if _xy_gap < 0.10:
            _gap_z = float(self._pinch()[2]) - z_tgt
            print(f">>> {label}: pinch already OVER the object (xy {_xy_gap * 1000:.0f}mm) -- "
                  f"columns straight down {_gap_z * 1000:+.0f}mm, a1 stays at {_a1_now:.3f}",
                  flush=True)
            q_dn = q.copy()
            q_dn[i_h1] = float(np.clip(q[i_h1] - _gap_z, 0.0, COLUMN_MAX))
            q_dn[i_h2] = float(np.clip(q[i_h2] - _gap_z, 0.0, COLUMN_MAX))
            _i_th_c = self.idx.get("BaseJoint_1")
            _th_c = float(q[_i_th_c]) if _i_th_c is not None else 0.0
            if not self._clears_chassis(_dh_now, _a1_now, th=_th_c):
                print(f">>> {label}: descent pose inside the chassis plate -- refusing", flush=True)
                return
            if not self._ramp_arm(q, q_dn, n_steps=100):
                print(f">>> {label}: diverged on the column descent -- restoring", flush=True)
                self._ramp_arm(np.asarray(self.robot.get_joint_positions(), float), q)
                return
            _qa = np.asarray(self.robot.get_joint_positions(), float)
            print(f">>> {label}: done, pinch {float(self._pinch()[2]):.3f} (target {z_tgt:.3f}, "
                  f"gap {(float(self._pinch()[2]) - z_tgt) * 1000:+.0f}mm) | ACT h1 "
                  f"{float(_qa[i_h1]):.3f} a1 {float(_qa[i_a1]):.3f} dh "
                  f"{float(_qa[i_h2] - _qa[i_h1]):+.3f}; enclose takes the last mm", flush=True)
            return

        _g_meas = None                        # real dz/da1, filled after the probe fraction
        _h1_tgt, _h2_tgt = _h1_0, _h2_0       # column trim, decided once the gradient is known
        _n_settle = 1                            # per-step writes; the step IS the settle
        try:
            _bx, _by, _byaw = self.base_ledger()
        except Exception:
            _bx = None
        # Keep the wrist level through the extension: the recorded wrist values belong to the
        # recorded boom angle, and the boom pitch changes as a1 grows.
        _wl_i = self.idx.get("gripper_y_rotation_1") if os.environ.get("WRIST_LEVEL", "1") == "1" else None
        if _wl_i is not None:
            _wl_wy0 = float(q[_wl_i])
            _wl_bp0 = float(self._lut_interp(self.clut["grid"], _dh_now, _a1_now, 4)[0])
        _i_th_m = self.idx.get("BaseJoint_1")
        _th_m = float(q[_i_th_m]) if _i_th_m is not None else 0.0
        # Command from a clean base vector, never fresh reads -- rebuilding from
        # `get_joint_positions()` feeds every disturbance back out as a kinematic command and locks
        # the drift in.
        _q_base = np.asarray(self.robot.get_joint_positions(), float).copy()
        _best_pz_m, _best_a1_m = float(self._pinch()[2]), _a1_now
        for _k in range(1, _N + 1):
            _f = float(_k) / _N
            _qc = _q_base.copy()
            _qc[i_a1] = _a1_now + _f * (_a1_end - _a1_now)
            if _wl_i is not None:
                _bp = float(self._lut_interp(self.clut["grid"], _dh_now, float(_qc[i_a1]), 4)[0])
                _qc[_wl_i] = _wl_wy0 - (_bp - _wl_bp0)
            # Mid-move body guard: the ramp's endpoints are checked, the path between them is not.
            if _k % 40 == 0 and not self._clears_chassis(
                    _dh_now, float(_qc[i_a1]),
                    margin=0.05, th=_th_m):
                print(f">>> {label}: extension STOPPED at a1 {float(_qc[i_a1]):.3f} -- next pose "
                      f"inside the chassis envelope at th {_th_m:+.2f}", flush=True)
                break
            if _g_meas is None and _k == _n_probe:
                # Measured, not predicted. The probe fraction must be large enough for a1's own
                # motion to dominate the residual settle, or the gradient comes out several times
                # too steep.
                _da1 = float(_qc[i_a1]) - _a1_now
                _dz = float(self._pinch()[2]) - _z_0
                _g_raw = (_dz / _da1) if abs(_da1) >= _da1_min else 0.0
                # Sanity band: the map predicts about -0.83 here and the arm measures about -0.32,
                # so anything outside this is probe noise rather than the linkage.
                _g_meas = float(np.clip(_g_raw, -1.20,
                                        0.20))
                if abs(_da1) < _da1_min:
                    print(f">>>   {label}: a1 travel {_da1:.3f} too small to measure a gradient "
                          f"-- columns take the whole gap", flush=True)
                _z_pred_end = float(self._pinch()[2]) + _g_meas * (_a1_end - float(_qc[i_a1]))
                _trim = _z_pred_end - z_tgt      # >0 -> a1 alone will finish HIGH, columns come down
                _h1_tgt = float(np.clip(_q_base[i_h1] - _trim, 0.0, COLUMN_MAX))
                _h2_tgt = float(np.clip(_q_base[i_h2] - _trim, 0.0, COLUMN_MAX))
                # The columns must never go up while the hand is already high, whatever the gradient
                # says -- which makes a bad probe reading harmless rather than destructive.
                if (pz0 - z_tgt) > 0.0 and _h1_tgt > _q_base[i_h1]:
                    print(f">>>   {label}: trim wanted the columns UP {(_h1_tgt - _qc[i_h1]) * 1000:+.0f}mm "
                          f"with the hand still high -- refused, holding them", flush=True)
                    _h1_tgt, _h2_tgt = float(_q_base[i_h1]), float(_q_base[i_h2])
                print(f">>>   {label}: measured dz/da1 {_g_meas:+.2f} (map said "
                      f"{(self._fk_z(_h1_0, _h2_0, _a1_end) - self._fk_z(_h1_0, _h2_0, _a1_now)) / max(1e-6, _a1_end - _a1_now):+.2f})"
                      f" -> a1 alone ends {(_z_pred_end - z_tgt) * 1000:+.0f}mm off; columns trim "
                      f"{-_trim * 1000:+.0f}mm over the rest of the move", flush=True)
            if _g_meas is not None:            # blend the column trim in over the remaining travel
                _fc = (_f - _probe_frac) / max(1e-6, 1.0 - _probe_frac)
                _fc = float(np.clip(_fc, 0.0, 1.0))
                _qc[i_h1] = float(np.clip(_h1_0 + _fc * (_h1_tgt - _h1_0), 0.0, COLUMN_MAX))
                _qc[i_h2] = float(np.clip(_h2_0 + _fc * (_h2_tgt - _h2_0), 0.0, COLUMN_MAX))
            if _bx is not None:
                self.set_base(_bx, _by, _byaw)
            self._force(_qc)
            self._apply(_qc)
            for _ in range(_n_settle):
                if _bx is not None:
                    self.set_base(_bx, _by, _byaw)
                self.world.step(render=not HEADLESS)
            if _k % 40 == 0:
                _dd_m = self._du_dv(obj)
                # The extension owns the height; du only bounds it at the near edge. Too shallow
                # presses the palm against the object, too early leaves the columns to make up the
                # height and sink into the chassis.
                if _dd_m is not None and _dd_m[0] > -0.075:
                    print(f">>> {label}: DEEP ENOUGH -- du {_dd_m[0] * 1000:+.0f}mm at a1 "
                          f"{float(_qc[i_a1]):.3f}; stopping the extension, columns take the rest",
                          flush=True)
                    break
            _pz_now_m = float(self._pinch()[2])
            if os.environ.get("LOWEST_TRACE", "0") == "1":
                self._lt_tr = getattr(self, "_lt_tr", [])
                self._lt_tr.append((_pz_now_m, float(np.asarray(
                    self.robot.get_joint_positions(), float)[i_a1])))
            if _pz_now_m - z_tgt <= z_tol and float(_qc[i_a1]) > _a1_now + 0.02:
                print(f">>> {label}: AT HEIGHT -- pinch {_pz_now_m:.3f} (target {z_tgt:.3f}) at a1 "
                      f"{float(_qc[i_a1]):.3f}; stopping the extension", flush=True)
                break
            if _pz_now_m < _best_pz_m:
                _best_pz_m, _best_a1_m = _pz_now_m, float(_qc[i_a1])
            elif (_pz_now_m > _best_pz_m + 0.025 and float(_qc[i_a1]) > _a1_now + 0.02):
                print(f">>> {label}: LINKAGE JUMP -- pinch rose {(_pz_now_m - _best_pz_m) * 1000:.0f}mm "
                      f"while extending (best {_best_pz_m:.3f} at a1 {_best_a1_m:.3f}); restoring "
                      f"the best pose, columns take the rest", flush=True)
                _q_r = _q_base.copy()
                _q_r[i_a1] = _best_a1_m
                self._ramp_arm(np.asarray(self.robot.get_joint_positions(), float), _q_r,
                               n_steps=60)
                break
            if _k % max(1, _N // 4) == 0:
                _qa_m = np.asarray(self.robot.get_joint_positions(), float)
                print(f">>>   {label}[{_k:3d}/{_N}]: a1 cmd {float(_qc[i_a1]):.3f} ACT "
                      f"{float(_qa_m[i_a1]):.3f} | pinch {float(self._pinch()[2]):.3f}", flush=True)
                self._arm_body_clearance(f"{label} {_k}")     # boom-vs-chassis during a1 extend
                if abs(float(_qa_m[i_a1]) - float(_qc[i_a1])) > 0.05:
                    print(f">>> {label}: a1 NOT TRACKING (cmd-act "
                          f"{abs(float(_qa_m[i_a1]) - float(_qc[i_a1])):.3f}) -- restoring",
                          flush=True)
                    self._ramp_arm(_qa_m, q, n_steps=60)
                    return
            _pz_chk = float(self._pinch()[2])
            if not np.isfinite(_pz_chk) or abs(_pz_chk) > 10.0:
                print(f">>> {label}: DIVERGED mid-move -- restoring", flush=True)
                self._ramp_arm(np.asarray(self.robot.get_joint_positions(), float), q)
                return
            # Abort if the move is making things worse. The guards above each watch a mechanism, and
            # a bad gradient can satisfy all of them while driving the hand away -- so watch the
            # objective itself.
            if _k % 40 == 0:
                _d_now = float(np.linalg.norm(
                    (np.asarray(obj.get_world_poses()[0][0], float) - self._pinch())[:2]))
                if _d_now > _d_entry + 0.10:
                    print(f">>> {label}: ABORT -- pinch->obj grew {_d_entry * 1000:.0f} -> "
                          f"{_d_now * 1000:.0f}mm during the move; restoring the entry pose",
                          flush=True)
                    self._ramp_arm(np.asarray(self.robot.get_joint_positions(), float), q,
                                   n_steps=40)
                    return
        if os.environ.get("LOWEST_TRACE", "0") == "1" and getattr(self, "_lt_tr", None):
            _tr = np.asarray(self._lt_tr, float)
            _dpz = np.abs(np.diff(_tr[:, 0])) * 1000.0
            _da = np.abs(np.diff(_tr[:, 1])) * 1000.0
            print(f">>> {label}: SMOOTHNESS over {len(_tr)} steps -- pinch step "
                  f"max {_dpz.max():.2f}mm mean {_dpz.mean():.3f}mm p99 "
                  f"{np.percentile(_dpz, 99):.2f}mm | a1 act step max {_da.max():.2f}mrad",
                  flush=True)
            self._lt_tr = []
        _gap_c2 = float(self._pinch()[2]) - z_tgt
        if _gap_c2 > z_tol:
            # The path guard may have stopped the extension early with no trim measured; finish the
            # height with the columns, which is exact 1:1 and leaves the xy footprint unchanged.
            _q_c2 = np.asarray(self.robot.get_joint_positions(), float).copy()
            _q_n2 = _q_c2.copy()
            # Column floor: below this the carriage is inside the chassis envelope. Height the
            # columns cannot legally supply is reported, not stolen from the body clearance.
            _h1_min = 0.20
            _drop = min(_gap_c2, max(0.0, float(_q_c2[i_h1]) - _h1_min))
            if _drop < _gap_c2 - 1e-4:
                print(f">>> {label}: column floor h1 {_h1_min:.2f} -- supplying {_drop * 1000:.0f} of "
                      f"{_gap_c2 * 1000:.0f}mm; the rest stays (arm must not sink into the body)",
                      flush=True)
            _q_n2[i_h1] = float(np.clip(_q_c2[i_h1] - _drop, 0.0, COLUMN_MAX))
            _q_n2[i_h2] = float(np.clip(_q_c2[i_h2] - _drop, 0.0, COLUMN_MAX))
            print(f">>> {label}: columns finish the height ({_drop * 1000:+.0f}mm, dh held "
                  f"{float(_q_c2[i_h2] - _q_c2[i_h1]):+.3f})", flush=True)
            self._ramp_arm(_q_c2, _q_n2, n_steps=80)
        _qa = np.asarray(self.robot.get_joint_positions(), float)
        _gap_f = float(self._pinch()[2]) - z_tgt
        print(f">>> {label}: done in one move, pinch {float(self._pinch()[2]):.3f} "
              f"(target {z_tgt:.3f}, gap {_gap_f * 1000:+.0f}mm) | ACT h1 {float(_qa[i_h1]):.3f} "
              f"a1 {float(_qa[i_a1]):.3f} dh {float(_qa[i_h2] - _qa[i_h1]):+.3f}; boom-straighten "
              f"takes the forward reach from here", flush=True)
        if abs(_gap_f) <= 0.030:
            return

        # The staged fallback below must not run after a good solve -- it opens by retracting a1,
        # undoing the single move. It exists only for when that solve cannot reach the height at
        # all.
        _gap_now = float(self._pinch()[2]) - z_tgt
        if abs(_gap_now) <= 0.030:
            q = np.asarray(self.robot.get_joint_positions(), float)
            print(f">>> {label}: done, pinch {float(self._pinch()[2]):.3f} (target {z_tgt:.3f}, "
                  f"gap {_gap_now * 1000:+.0f}mm) at a1 {float(q[i_a1]):.3f}, dh "
                  f"{float(q[i_h2] - q[i_h1]):+.3f}; const-Z forward reach takes it from here",
                  flush=True)
            return

        # ---- Step A: retract a1, so the tilt has a vertical lever at all ------------------------
        # Floored by the chassis plate: at a large dh a bare retract puts the gripper behind its
        # front edge.
        _dh_fb = float(q[i_h2] - q[i_h1])
        # Wider margin on retracts: the clearance test tracks the gripper ORIGIN, but the fingers
        # hang below and behind it and a retract sweeps them back over the body.
        _m_ret = 0.10
        _i_th_f = self.idx.get("BaseJoint_1")
        _th_f = float(q[_i_th_f]) if _i_th_f is not None else 0.0
        _a1_fb = self._a1_clearing_chassis(_dh_fb, margin=_m_ret, a1_max=a1_max, th=_th_f)
        _a1_ret_fb = a1_tilt if _a1_fb is None else max(a1_tilt, _a1_fb)
        if _a1_ret_fb > a1_tilt + 1e-4:
            print(f">>>   {label}: fallback retract floored at a1 {_a1_ret_fb:.3f} (not "
                  f"{a1_tilt:.3f}) -- below it the gripper is inside the chassis plate at dh "
                  f"{_dh_fb:+.3f}", flush=True)
        if float(q[i_a1]) > _a1_ret_fb + 0.004:
            q_r = q.copy()
            q_r[i_a1] = _a1_ret_fb
            if not self._ramp_arm(q, q_r, n_steps=12):
                print(f">>> {label}: diverged retracting a1 -- stopping", flush=True)
                return
            q = q_r
            print(f">>>   {label}: a1 retracted {float(q[i_a1]):.3f} for the tilt "
                  f"(pinch {float(self._pinch()[2]):.3f})", flush=True)

        # ---- Step B: tilt -- Newton on the FK gradient, verified live, reverted on no gain ------
        for it in range(max_it):
            q = np.asarray(self.robot.get_joint_positions(), float).copy()
            pz = float(self._pinch()[2])
            err = pz - z_tgt
            if err <= z_tol:
                break
            h1, h2, a1 = float(q[i_h1]), float(q[i_h2]), float(q[i_a1])
            _dt = 0.03
            grad = (self._fk_z(h1 - 0.5 * _dt, h2 + 0.5 * _dt, a1) - self._fk_z(h1, h2, a1)) / _dt
            if abs(grad) < 0.05:
                print(f">>> {label}: tilt has no Z authority (grad {grad:+.2f}) -- "
                      f"leaving (z_gap {err * 1000:+.0f}mm)", flush=True)
                break
            cmd = float(np.clip(-err / grad, -0.10, 0.10))
            dh_now = h2 - h1
            dh_new = float(np.clip(dh_now + cmd, dh_lo, dh_hi))
            # The plate bounds the tilt too: deeper tilt pulls the hand back, so a dh that clears at
            # full a1 is inside the chassis at the retracted a1 here.
            _m_ret2 = 0.10
            _th_b = float(q[self.idx["BaseJoint_1"]]) if "BaseJoint_1" in self.idx else 0.0
            while dh_new > dh_now and not self._clears_chassis(dh_new, a1, margin=_m_ret2, th=_th_b):
                dh_new -= 0.005
            cmd = dh_new - dh_now
            if abs(cmd) < 1e-4 and not self._clears_chassis(dh_now + 0.005, a1, margin=_m_ret2, th=_th_b):
                print(f">>> {label}: tilt stopped by the chassis plate (dh {dh_now:+.3f} at a1 "
                      f"{a1:.3f}) -- leaving (z_gap {err * 1000:+.0f}mm)", flush=True)
                break
            if abs(cmd) < 1e-4:
                print(f">>> {label}: tilt at the LUT bound (dh {dh_now:+.3f}) -- "
                      f"leaving (z_gap {err * 1000:+.0f}mm)", flush=True)
                break
            q_n = q.copy()
            q_n[i_h1] = float(np.clip(h1 - 0.5 * cmd, 0.0, COLUMN_MAX))
            q_n[i_h2] = float(np.clip(h2 + 0.5 * cmd, 0.0, COLUMN_MAX))
            if not self._ramp_arm(q, q_n):
                print(f">>> {label}: diverged on tilt it{it + 1} -- restoring", flush=True)
                self._ramp_arm(np.asarray(self.robot.get_joint_positions(), float), q)
                break
            err2 = float(self._pinch()[2]) - z_tgt
            if err2 > err - 0.003:
                print(f">>> {label}: it{it + 1} REVERTED (z not improved "
                      f"{err * 1000:+.0f} -> {err2 * 1000:+.0f}mm)", flush=True)
                self._ramp_arm(q_n, q)
                break
            print(f">>> {label}: it{it + 1} tilt {cmd:+.3f} -> dh {dh_new:+.3f} "
                  f"(grad {grad:+.2f}) z_gap {err * 1000:+.0f} -> {err2 * 1000:+.0f}mm", flush=True)

        # ---- Step C: a1 forward, for as long as the map says it does not lift the hand ---------
        # Extending a1 only descends while the boom angle keeps the slide pointing down, so ask per
        # pose.
        q = np.asarray(self.robot.get_joint_positions(), float).copy()
        while float(q[i_a1]) < a1_max - 1e-3:
            h1, h2, a1 = float(q[i_h1]), float(q[i_h2]), float(q[i_a1])
            a1_n = min(a1_max, a1 + 0.05)
            if (self._fk_z(h1, h2, a1_n) - self._fk_z(h1, h2, a1)) > 0.002:
                print(f">>>   {label}: a1 stops here ({a1:.3f}) -- extending further LIFTS the hand "
                      f"at dh {h2 - h1:+.3f}; the const-Z servo takes the forward reach", flush=True)
                break
            q_n = q.copy()
            q_n[i_a1] = a1_n
            if not self._ramp_arm(q, q_n):
                print(f">>> {label}: diverged extending a1 -- restoring", flush=True)
                self._ramp_arm(np.asarray(self.robot.get_joint_positions(), float), q)
                break
            q = q_n
            if float(self._pinch()[2]) - z_tgt <= z_tol:
                break

        q = np.asarray(self.robot.get_joint_positions(), float)
        print(f">>> {label}: done, pinch {float(self._pinch()[2]):.3f} (target {z_tgt:.3f}, "
              f"gap {(float(self._pinch()[2]) - z_tgt) * 1000:+.0f}mm) at a1 {float(q[i_a1]):.3f}, "
              f"dh {float(q[i_h2] - q[i_h1]):+.3f}; const-Z forward reach takes it from here",
              flush=True)

    def _reach_polar(self, obj, label="reach-polar"):
        """Solve tilt and a1 together for a (forward, down) target.

        The boom is a polar arm: the h2-h1 differential sets its angle, a1 its length along that
        angle, so `forward = r·cos(theta)` and `down = r·sin(theta)`. The two axes are coupled and
        cannot be servoed one after the other —

          * more tilt: the same a1 extension buys more down and less forward
          * max tilt with max a1: the lowest reachable point, but pulled back toward the chassis
          * less tilt: reaches further out, but stays high

        — so servoing them in sequence just trades one error for the other. Rather than assume the
        pivot geometry, measure the 2x2 Jacobian each iteration (one probe on the tilt, one on a1,
        both restored afterwards) and solve `J @ [dtilt, da1] = [d_forward, d_down]`. Bounded per
        iteration, and the best pose is kept so a bad solve cannot leave the arm worse than it
        found it.
        """
        i_h1 = self.idx.get("ColumnLeftBearingJoint_1")
        i_h2 = self.idx.get("ColumnRightBearingJoint_1")
        i_a1 = self.idx.get("ArmLeftJoint_1")
        if i_h1 is None or i_h2 is None or i_a1 is None:
            return
        n_it = 8
        d_pr = 0.03
        cap_t = 0.08
        cap_a = 0.06
        tol = 0.012
        n_set = max(1, int(0.06 / self.dt))
        du_tgt = float(os.environ.get("ENCLOSE_DU", "-0.035"))

        def _settle(q):
            self._force(q)
            self._apply(q)
            for _ in range(n_set):
                self.world.step(render=not HEADLESS)

        def _state():
            """(forward-error, down-error) in metres. forward: du toward its target; down: pinch
            above the object centre."""
            _oc = np.asarray(obj.get_world_poses()[0][0], float)
            _p = np.asarray(self._pinch(), float)
            _dd = self._du_dv(obj)
            _du = float(_dd[0]) if _dd is not None else float("nan")
            return (du_tgt - _du), (float(_p[2]) - float(_oc[2]))

        f0, d0 = _state()
        if not np.isfinite(f0):
            print(f">>> {label}: du unavailable -- skipped", flush=True)
            return
        print(f">>> {label}: forward err {f0 * 1000:+.0f}mm, down err {d0 * 1000:+.0f}mm "
              f"(a1 {float(np.asarray(self.robot.get_joint_positions(), float)[i_a1]):.3f})",
              flush=True)
        best = abs(f0) + abs(d0)
        best_q = np.asarray(self.robot.get_joint_positions(), float).copy()
        for it in range(n_it):
            q = np.asarray(self.robot.get_joint_positions(), float).copy()
            f_e, d_e = _state()
            if not np.isfinite(f_e):
                break
            if abs(f_e) <= tol and abs(d_e) <= tol:
                print(f">>> {label}: reached (forward {f_e * 1000:+.0f}mm, down {d_e * 1000:+.0f}mm) "
                      f"after {it} iters", flush=True)
                return
            qp = q.copy()                                     # probe TILT
            qp[i_h1] = float(np.clip(q[i_h1] - 0.5 * d_pr, 0.0, COLUMN_MAX))
            qp[i_h2] = float(np.clip(q[i_h2] + 0.5 * d_pr, 0.0, COLUMN_MAX))
            _settle(qp)
            f_t, d_t = _state()
            _settle(q)
            qa = q.copy()                                     # probe A1
            qa[i_a1] = float(np.clip(q[i_a1] + d_pr, 0.0, 0.625))
            _settle(qa)
            f_a, d_a = _state()
            _settle(q)
            if not (np.isfinite(f_t) and np.isfinite(f_a)):
                break
            J = np.array([[(f_t - f_e) / d_pr, (f_a - f_e) / d_pr],
                          [(d_t - d_e) / d_pr, (d_a - d_e) / d_pr]], float)
            if abs(float(np.linalg.det(J))) < 1e-4:           # tilt and a1 not independent here
                print(f">>> {label}: axes degenerate (det {float(np.linalg.det(J)):.5f}) -- "
                      f"stopping at forward {f_e * 1000:+.0f}mm down {d_e * 1000:+.0f}mm", flush=True)
                break
            sol = np.linalg.solve(J, np.array([-f_e, -d_e], float))
            d_tilt = float(np.clip(sol[0], -cap_t, cap_t))
            d_a1 = float(np.clip(sol[1], -cap_a, cap_a))
            qn = q.copy()
            qn[i_h1] = float(np.clip(q[i_h1] - 0.5 * d_tilt, 0.0, COLUMN_MAX))
            qn[i_h2] = float(np.clip(q[i_h2] + 0.5 * d_tilt, 0.0, COLUMN_MAX))
            qn[i_a1] = float(np.clip(q[i_a1] + d_a1, 0.0, 0.625))
            _settle(qn)
            f_n, d_n = _state()
            print(f">>>   {label}[{it}]: forward {f_e * 1000:+.0f} -> {f_n * 1000:+.0f}mm, "
                  f"down {d_e * 1000:+.0f} -> {d_n * 1000:+.0f}mm "
                  f"(dtilt {d_tilt:+.3f}, da1 {d_a1:+.3f})", flush=True)
            _score = abs(f_n) + abs(d_n)
            if _score < best:
                best, best_q = _score, qn.copy()
            elif _score > best * 1.5:
                print(f">>> {label}: diverging -- restoring the best pose", flush=True)
                _settle(best_q)
                return
        _settle(best_q)

    def _tilt_to_obj_z(self, obj, label="tilt-to-objz"):
        """Tilt the boom down to the object's centre height, before any horizontal reach.

        Tilt first, reach second. At the a1-retracted pose the h2-h1 differential is grown until
        the pinch drops to the object's centre height, and only then does a1 extend. Approaching
        high and lowering the columns together instead moves the hand straight down rather than
        forward-and-down, so the gripper passes over the object and pushes it away.

        Closed loop on the live pinch with an empirically measured gradient, because the tilt's
        vertical lever depends on the whole four-bar pose: perturb the differential, measure
        d(pinch_z), step. One-sided — a pinch already at or below the object grasps fine — and
        bounded, so the worst case is a no-op.
        """
        i_h1 = self.idx.get("ColumnLeftBearingJoint_1")
        i_h2 = self.idx.get("ColumnRightBearingJoint_1")
        if i_h1 is None or i_h2 is None:
            return
        z_tol = 0.012
        z_bias = -0.005
        n_it = 16
        d_probe = 0.03
        cap = 0.10
        n_set = max(1, int(0.08 / self.dt))
        tgt_z = float(np.asarray(obj.get_world_poses()[0][0], float)[2]) + z_bias

        def _settle(q):
            self._force(q)
            self._apply(q)
            for _ in range(n_set):
                self.world.step(render=not HEADLESS)
            return float(self._pinch()[2])

        q = np.asarray(self.robot.get_joint_positions(), float).copy()
        pz = float(self._pinch()[2])
        if (pz - tgt_z) <= z_tol:
            print(f">>> {label}: pinch already at/below the object "
                  f"(z_gap {(pz - tgt_z) * 1000:+.0f}mm) -- no tilt needed", flush=True)
            return
        print(f">>> {label}: pinch {pz:.3f} vs object centre {tgt_z:.3f} "
              f"(z_gap {(pz - tgt_z) * 1000:+.0f}mm) -- tilting the boom BEFORE the reach", flush=True)
        _best_pz, _best_q, _bad = pz, None, 0
        for it in range(n_it):
            q = np.asarray(self.robot.get_joint_positions(), float).copy()
            pz = float(self._pinch()[2])
            err = pz - tgt_z
            if err <= z_tol:
                print(f">>> {label}: reached (z_gap {err * 1000:+.0f}mm) after {it} iters", flush=True)
                return
            # Probe, then put it back. Leaving it in place makes every later iteration read from a
            # displaced pose, and the stage walks the hand up while trying to lower it.
            qp = q.copy()                                    # probe: GROW the tilt
            qp[i_h1] = float(np.clip(q[i_h1] - 0.5 * d_probe, 0.0, COLUMN_MAX))
            qp[i_h2] = float(np.clip(q[i_h2] + 0.5 * d_probe, 0.0, COLUMN_MAX))
            pz_p = _settle(qp)
            grad = (pz_p - pz) / d_probe                     # d(pinch_z) / d(tilt-grow)
            _settle(q)                                       # restore before commanding
            if abs(grad) < 0.05:
                print(f">>> {label}: tilt has no Z authority here (grad {grad:+.2f}) -- "
                      f"leaving it at z_gap {err * 1000:+.0f}mm", flush=True)
                return
            cmd = float(np.clip(-err / grad, -cap, cap))
            qn = q.copy()                                    # from the RESTORED pose, not the probe
            qn[i_h1] = float(np.clip(qn[i_h1] - 0.5 * cmd, 0.0, COLUMN_MAX))
            qn[i_h2] = float(np.clip(qn[i_h2] + 0.5 * cmd, 0.0, COLUMN_MAX))
            pz_n = _settle(qn)
            print(f">>>   {label}[{it}]: z_gap {err * 1000:+.0f} -> "
                  f"{(pz_n - tgt_z) * 1000:+.0f}mm  (grad {grad:+.2f}, cmd {cmd:+.3f}, "
                  f"dh {float(qn[i_h2] - qn[i_h1]):+.3f})", flush=True)
            # One bad step is noise, not a stall -- the gradient scatters by a factor of two on a
            # compliant arm. Require two in a row, and keep the best pose.
            if pz_n < _best_pz - 1e-4:
                _best_pz, _best_q, _bad = pz_n, qn.copy(), 0
            else:
                _bad += 1
                if _bad >= 2:
                    print(f">>> {label}: no further Z gain in 2 iters -- restoring the best pose "
                          f"(z_gap {(_best_pz - tgt_z) * 1000:+.0f}mm)", flush=True)
                    if _best_q is not None:
                        _settle(_best_q)
                    return
