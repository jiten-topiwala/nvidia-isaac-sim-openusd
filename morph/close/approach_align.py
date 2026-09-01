"""Pre-close XY/Z alignment servos: const-z approach, column-only height, and the
recorded-relation align.

Part of the close sequence -- see `morph/close/__init__.py` for the stage order and the
shared state object. Split from `enclose.py`. Imports Isaac APIs at module level; only
importable after SimulationApp.
"""
import os

import numpy as np

from morph.config import KNOWN, COLUMN_MAX


class ApproachAlignStage:
    """One stage of `_close_replay`. Mixed into `CloseMixin`; all `self.*` belong to `Demo`."""

    def _align_to_grasp_center(self, obj, label="align-pinch"):
        """Single const-z servo that approaches and aligns in one loop.

        Drives the arm (h1, h2, a1, th) so the fingertip pinch lands on the object's axis at
        constant height. The pinch is the grasp point, so approaching it and aligning to it are
        the same motion — split across two stages they target different points and fight.

        Closed-loop and proportional: `dq` comes from `lstsq(J, move)`, damped and clipped, so the
        servo is coarse and fast far out and fine and slow near the target, with a divergence
        guard. Height is held by a decoupled column nudge rather than being mixed into the solve;
        mixing it in makes the least-squares step unstable.
        """
        i_h1 = self.idx.get("ColumnLeftBearingJoint_1"); i_h2 = self.idx.get("ColumnRightBearingJoint_1")
        i_a1 = self.idx.get("ArmLeftJoint_1"); i_th = self.idx.get("BaseJoint_1")
        if None in (i_h1, i_h2, i_a1, i_th):
            return
        tol = 0.006
        n_it = 26   # enough iters to cover the ~150mm approach
        _e = 0.01
        try:
            _bx, _by, _byaw = self.base_ledger()
        except Exception:
            _bx = None
        _q_ctr0 = np.asarray(self.robot.get_joint_positions(), float).copy()
        self._ctr_prev_e = None
        _e_best = float("inf")                                 # tightest XY error seen (for the log)
        for it in range(n_it):
            q = np.asarray(self.robot.get_joint_positions(), float).copy()
            _pinch = np.asarray(self._pinch(), float)          # fingertip pinch = the grasp point
            oc = np.asarray(obj.get_world_poses()[0][0], float)
            err = (oc - _pinch)[:2]                            # drive the PINCH onto the object axis
            _z_err = float(oc[2]) - float(_pinch[2])
            _en = float(np.linalg.norm(err))
            _e_best = min(_e_best, _en)
            # Return on XY alone. z is held by the column nudge below and cannot reach tol exactly,
            # so gating on it runs the servo past convergence and destabilises XY.
            if _en <= tol:
                print(f">>> {label}: centred, pinch->object {_en * 1000:.0f}mm "
                      f"z-err {_z_err * 1000:+.0f}mm in {it} it", flush=True)
                return
            # Divergence guard: if the error grows, stop in place. Ramping a diverged pose back
            # destabilises the articulation, where stopping leaves the arm a few centimetres off for
            # the close to absorb.
            if getattr(self, "_ctr_prev_e", None) is not None and _en > self._ctr_prev_e + 0.010:
                print(f">>> {label}: error GREW {self._ctr_prev_e * 1000:.0f}->{_en * 1000:.0f}mm "
                      f"-- unstable, STOPPING in place (best seen {_e_best * 1000:.0f}mm)", flush=True)
                return
            self._ctr_prev_e = _en
            move = np.array([err[0], err[1], 0.0])             # XY solve only (stable)
            h1, h2, a1, th = float(q[i_h1]), float(q[i_h2]), float(q[i_a1]), float(q[i_th])
            base = np.asarray(self._grip_fk(h1, h2, a1, th), float)
            J = np.column_stack([
                (np.asarray(self._grip_fk(h1 + _e, h2, a1, th), float) - base) / _e,
                (np.asarray(self._grip_fk(h1, h2 + _e, a1, th), float) - base) / _e,
                (np.asarray(self._grip_fk(h1, h2, a1 + _e, th), float) - base) / _e,
                (np.asarray(self._grip_fk(h1, h2, a1, th + _e), float) - base) / _e])
            dq, *_ = np.linalg.lstsq(J, move, rcond=None)
            dq = np.clip(dq * 0.6, -0.03, 0.03)  # damped
            # Decoupled const-z hold: nudge both columns by the z error, outside the lstsq.
            _dz = float(np.clip(0.5 * _z_err, -0.004, 0.004))
            q_n = q.copy()
            q_n[i_h1] = float(np.clip(h1 + dq[0] + _dz, 0.0, COLUMN_MAX)); q_n[i_h2] = float(np.clip(h2 + dq[1] + _dz, 0.0, COLUMN_MAX))
            q_n[i_a1] = float(np.clip(a1 + dq[2], 0.0, 0.47))
            q_n[i_th] = float(th + dq[3])
            if not self._clears_chassis(q_n[i_h2] - q_n[i_h1], q_n[i_a1], margin=0.02, th=q_n[i_th]):
                print(f">>> {label}: next pose hits the chassis plate -- stop at {np.linalg.norm(err) * 1000:.0f}mm", flush=True)
                return
            # Ejection guard. `_ramp_arm` writes joint positions, so the bodies are kinematic:
            # contact cannot push them back, and a step landing inside the object ejects it.
            if os.environ.get("ALIGN_CTR_GUARD", "1") == "1":
                try:
                    _occ = np.asarray(obj.get_world_poses()[0][0], float)
                    _oRc = self._obj_R(obj)
                    _gsc = [self._wrap_gap(_f, _occ, KNOWN["object_radius"], self.obj_half_h, _oRc)
                            for _f in "abc"]
                    _pgc = self._finger_surface_gap("palm", _occ, KNOWN["object_radius"],
                                                    self.obj_half_h, _oRc)
                    _gminc = min([g for g in _gsc if g is not None] + [_pgc])
                except Exception:
                    _gminc = None
                _gsc_stop = 0.010
                if _gminc is not None and _gminc < _gsc_stop:
                    print(f">>> {label}: GAP STOP at iter {it} -- nearest body {_gminc * 1000:+.1f}mm "
                          f"from the surface, below the {_gsc_stop * 1000:.0f}mm floor. Advancing "
                          f"would teleport into the object (residual {np.linalg.norm(err) * 1000:.0f}mm; "
                          f"the force close covers the rest)", flush=True)
                    return
            if _bx is not None:
                self.set_base(_bx, _by, _byaw)
            if not self._ramp_arm(q, q_n, n_steps=max(8, int(np.max(np.abs(dq)) / 0.002))):
                self._ramp_arm(np.asarray(self.robot.get_joint_positions(), float), q); return
            if it % 3 == 0:
                # Object z on every line: reporting only the servo's own error hides a sink it is
                # causing.
                try:
                    _ozc = float(np.asarray(obj.get_world_poses()[0][0], float)[2])
                    _sfx = (f" | obj z {_ozc:.3f}"
                            + ("  <-- SINKING" if _ozc < float(self.obj_half_h) - 0.005 else ""))
                except Exception:
                    _sfx = ""
                print(f">>>   {label}[{it}]: pinch->object {np.round(err * 1000, 0).tolist()}mm -> dq {np.round(dq, 3).tolist()}{_sfx}", flush=True)
        # Iterations exhausted. On a converging run the XY return above exits first, so reaching
        # here means the error never tightened.
        _err2 = (np.asarray(obj.get_world_poses()[0][0], float) - np.asarray(self._pinch(), float))[:2]
        print(f">>> {label}: done, pinch->object {np.linalg.norm(_err2) * 1000:.0f}mm "
              f"(best {_e_best * 1000:.0f}mm)", flush=True)

    def _align_z_to_center(self, obj, label="align-z"):
        """Bring the pinch to the object's mid-height with the columns only, after the XY align.

        The XY Jacobian servo couples a1 into z (dz/da1 ~ -0.78), so nulling XY drifts the pinch
        up above the object's centre and the fingers curl over the top of it. Moving both columns
        by the same dz changes z almost purely — dh is preserved, so tilt, reach and chassis
        clearance are unchanged — which makes a dedicated column step the stable way to land the
        grasp at mid-height.
        """
        if os.environ.get("ALIGN_Z", "1") != "1":
            return
        i_h1 = self.idx.get("ColumnLeftBearingJoint_1"); i_h2 = self.idx.get("ColumnRightBearingJoint_1")
        if i_h1 is None or i_h2 is None:
            return
        tol = 0.008
        try:
            _bx, _by, _byaw = self.base_ledger()
        except Exception:
            _bx = None
        self._az_prev_err = None          # per CALL: a stale error latches the divergence abort
        for it in range(50):
            oc_z = float(np.asarray(obj.get_world_poses()[0][0], float)[2])   # object CENTRE = mid-height
            _z_err = oc_z - float(self._pinch()[2])
            if abs(_z_err) <= tol:
                print(f">>> {label}: pinch at object mid-height (z-err {_z_err * 1000:+.0f}mm) "
                      f"in {it} it", flush=True)
                return
            q = np.asarray(self.robot.get_joint_positions(), float).copy()
            _dz = float(np.clip(0.6 * _z_err, -0.006, 0.006))    # columns down/up; dh preserved

            # Ejection guard, the same rule as `_align_to_grasp_center`. The kinematic write is
            # load-bearing and cannot be dropped, so it must simply never land inside the object --
            # without this the columns descend for up to 50 iterations and expel the object through
            # the pads.
            if os.environ.get("ALIGN_Z_GUARD", "1") == "1" and _dz < 0:
                try:
                    _ocg = np.asarray(obj.get_world_poses()[0][0], float)
                    _oRg = self._obj_R(obj)
                    _gsg = [self._wrap_gap(_f, _ocg, KNOWN["object_radius"], self.obj_half_h, _oRg)
                            for _f in "abc"]
                    _pgg = self._finger_surface_gap("palm", _ocg, KNOWN["object_radius"],
                                                    self.obj_half_h, _oRg)
                    _gming = min([g for g in _gsg if g is not None] + [_pgg])
                except Exception:
                    _gming = None
                if _gming is not None and _gming < 0.010:
                    print(f">>> {label}: GAP STOP at iter {it} -- nearest body {_gming * 1000:+.1f}mm "
                          f"from the surface, below the {float(os.environ.get('ENCLOSE_GAP_STOP', '0.010')) * 1000:.0f}mm "
                          f"floor; lowering further would teleport into the object (z-err "
                          f"{_z_err * 1000:+.0f}mm left, the close covers the rest)", flush=True)
                    return

            # Divergence abort: an error that grows while the columns move the correct way means the
            # object is being driven down, and every further iteration presses harder.
            _pz_e = getattr(self, "_az_prev_err", None)
            if _pz_e is not None and abs(_z_err) > abs(_pz_e) + 0.004:
                print(f">>> {label}: DIVERGING at iter {it} -- z-err {_pz_e * 1000:+.0f} -> "
                      f"{_z_err * 1000:+.0f}mm while lowering; the OBJECT is moving, not the hand. "
                      f"Stopping instead of chasing it down.", flush=True)
                self._az_prev_err = None
                return
            self._az_prev_err = _z_err
            q[i_h1] = float(np.clip(q[i_h1] + _dz, 0.0, COLUMN_MAX))
            q[i_h2] = float(np.clip(q[i_h2] + _dz, 0.0, COLUMN_MAX))
            if _bx is not None:
                self.set_base(_bx, _by, _byaw)
            # Heartbeat -- printing only on exit looks like a hang on a slow sim. Raw z for both
            # bodies, since the error is a difference and cannot say which side moved.
            if os.environ.get("ALIGN_Z_TRACE", "1") == "1":
                _pz_t = float(self._pinch()[2])
                _nc_t = len(getattr(self, "_ctc", {}) or getattr(self, "_fN_step", {}) or {})
                print(f">>>   {label}[{it:2d}]: obj z {oc_z:.3f} (rest {float(self.obj_half_h):.3f}"
                      f"{'  <-- SUNK' if oc_z < float(self.obj_half_h) - 0.005 else ''})  "
                      f"pinch z {_pz_t:.3f}  z-err {_z_err * 1000:+.0f}mm -> columns "
                      f"{_dz * 1000:+.1f}mm | contacts {_nc_t}", flush=True)
            self._ramp_arm(np.asarray(self.robot.get_joint_positions(), float), q, n_steps=8)
        print(f">>> {label}: done, pinch z {float(self._pinch()[2]):.3f} vs object centre "
              f"{float(np.asarray(obj.get_world_poses()[0][0], float)[2]):.3f}", flush=True)

    def _align_to_recorded(self, obj, label="align-rec"):
        """Closed-loop alignment onto a recorded working grasp, expressed in the hand frame.

        Works in the hand frame rather than the world-XY du/dv window, which goes degenerate once
        the hand pitches down: its "depth" is then measured along a near-horizontal direction that
        barely exists.

        Each step measures the object in the grip frame, subtracts the recorded target and moves
        the arm to null the difference. The grip-position Jacobian with respect to (h1, h2, a1, th)
        comes from `_grip_fk` by finite difference, so no probe perturbs the live simulation, and
        the FK-vs-live offset cancels because the servo reads the real object every iteration.
        """
        rec = getattr(self, "_rec_obj_in_grip", None)
        if rec is None:
            print(f">>> {label}: no recorded relation -> skipped", flush=True)
            return
        i_h1 = self.idx.get("ColumnLeftBearingJoint_1"); i_h2 = self.idx.get("ColumnRightBearingJoint_1")
        i_a1 = self.idx.get("ArmLeftJoint_1"); i_th = self.idx.get("BaseJoint_1")
        if None in (i_h1, i_h2, i_a1, i_th):
            return
        tol = 0.010
        n_it = 8
        _e = 0.01                                              # FK finite-diff step
        try:
            _bx, _by, _byaw = self.base_ledger()
        except Exception:
            _bx = None
        for it in range(n_it):
            q = np.asarray(self.robot.get_joint_positions(), float).copy()
            gp, gR = self._grip_frame()
            oc = np.asarray(obj.get_world_poses()[0][0], float)
            cur = gR.T @ (oc - gp)                             # object in grip frame, live
            err_l = cur - rec                                  # want cur -> rec
            if float(np.linalg.norm(err_l)) <= tol:
                print(f">>> {label}: aligned, obj-in-grip {np.round(cur, 4).tolist()} vs recorded "
                      f"{np.round(rec, 4).tolist()} (residual {np.linalg.norm(err_l) * 1000:.0f}mm) "
                      f"in {it} it", flush=True)
                return
            move = oc - gR @ rec - gp                          # world translation the grip needs
            h1, h2, a1, th = float(q[i_h1]), float(q[i_h2]), float(q[i_a1]), float(q[i_th])
            base = np.asarray(self._grip_fk(h1, h2, a1, th), float)
            J = np.column_stack([
                (np.asarray(self._grip_fk(h1 + _e, h2, a1, th), float) - base) / _e,
                (np.asarray(self._grip_fk(h1, h2 + _e, a1, th), float) - base) / _e,
                (np.asarray(self._grip_fk(h1, h2, a1 + _e, th), float) - base) / _e,
                (np.asarray(self._grip_fk(h1, h2, a1, th + _e), float) - base) / _e,
            ])                                                 # 3x4: d(grip_world)/d(h1,h2,a1,th)
            dq, *_ = np.linalg.lstsq(J, move, rcond=None)      # least-norm 4-vector
            dq = np.clip(dq, -0.05, 0.05)                      # bound per iteration
            q_n = q.copy()
            q_n[i_h1] = float(np.clip(h1 + dq[0], 0.0, COLUMN_MAX))
            q_n[i_h2] = float(np.clip(h2 + dq[1], 0.0, COLUMN_MAX))
            q_n[i_a1] = float(np.clip(a1 + dq[2], 0.0, 0.47))
            q_n[i_th] = float(th + dq[3])
            # never drive the arm into the chassis plate
            if not self._clears_chassis(q_n[i_h2] - q_n[i_h1], q_n[i_a1], margin=0.02, th=q_n[i_th]):
                print(f">>> {label}: next pose hits the chassis plate -- stopping at residual "
                      f"{np.linalg.norm(err_l) * 1000:.0f}mm", flush=True)
                return
            if _bx is not None:
                self.set_base(_bx, _by, _byaw)
            if not self._ramp_arm(q, q_n, n_steps=max(8, int(np.max(np.abs(dq)) / 0.002))):
                print(f">>> {label}: diverged -- restoring", flush=True)
                self._ramp_arm(np.asarray(self.robot.get_joint_positions(), float), q)
                return
            if it % 2 == 0:
                print(f">>>   {label}[{it}]: obj-in-grip {np.round(cur, 3).tolist()} err "
                      f"{np.round(err_l * 1000, 0).tolist()}mm -> dq {np.round(dq, 3).tolist()}",
                      flush=True)
        _gp2, _gR2 = self._grip_frame()
        _cur2 = _gR2.T @ (np.asarray(obj.get_world_poses()[0][0], float) - _gp2)
        print(f">>> {label}: done, residual {np.linalg.norm(_cur2 - rec) * 1000:.0f}mm "
              f"(obj-in-grip {np.round(_cur2, 4).tolist()} vs recorded {np.round(rec, 4).tolist()})",
              flush=True)
