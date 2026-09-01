"""Pre-close hand placement: ENCLOSE (get the object inside the finger circle) and ALIGN_EQ
(equalise the three surface gaps so no finger presses alone).

Part of the close sequence — see `morph/close/__init__.py` for the stage order and the
shared state object. Imports Isaac APIs at module level; only importable after SimulationApp.
"""
import math
import os

import numpy as np

from morph.config import PERFECT, HEADLESS, KNOWN


class EncloseStage:
    """One stage of `_close_replay`. Mixed into `CloseMixin`; all `self.*` belong to `Demo`."""

    def _close_enclose(self, st):
        obj = st.obj
        # This hand is built for an ENCOMPASSING grasp, not a pinch: the object rests against the
        # palm and the fingers curl around it.
        if os.environ.get("ENCLOSE", "1") == "1" and PERFECT:
            _tol_e = 0.030
            # Two implementations: the Cartesian jog by default, the direct joint servo behind
            # ENCLOSE_JOINT for poses near the reach limit, where the jog's IK under-reaches and
            # stalls.
            if os.environ.get("ENCLOSE_JOINT", "0") == "1":
                try:
                    self._enclose_joint_servo(obj, _tol_e)
                except Exception as _e_js:
                    print(f">>> enclose(joint): failed ({_e_js})", flush=True)
                return
            try:
                _z0_e = float(self._pinch()[2])
                self.start_jog()
                for _ke in range(int(3.0 / self.dt)):
                    _oc_n = np.asarray(obj.get_world_poses()[0][0], float)
                    _pn = self._pinch()
                    _d_n = (_oc_n - _pn)[:2]
                    _e_n = float(np.linalg.norm(_d_n))
                    if _ke % 120 == 0:
                        print(f">>>   enclose[{_ke:4d}]: pinch->obj {_e_n * 1000:.0f}mm", flush=True)
                    if _e_n < _tol_e:
                        print(f">>> enclose: object inside the finger circle "
                              f"(pinch->obj {_e_n * 1000:.0f}mm)", flush=True)
                        break
                    _step_e = float(np.clip(_e_n, 0.0, 0.004))
                    _dir_e = _d_n / max(1e-9, _e_n)
                    # Constant z means an ABSOLUTE z: rebuilding the target from the live gripper
                    # makes any drift the new setpoint, and the stage climbs out of the grasp height
                    # over the run.
                    _gp_e = self.grip_pos()
                    if _ke == 0:
                        _gz_e = float(_gp_e[2])
                    self.jog["target"] = np.array(
                        [_gp_e[0] + _dir_e[0] * _step_e, _gp_e[1] + _dir_e[1] * _step_e, _gz_e])
                    self._jog_tick()
                    self.world.step(render=not HEADLESS)
                else:
                    _oc_n = np.asarray(obj.get_world_poses()[0][0], float)
                    print(f">>> enclose: STALLED at pinch->obj "
                          f"{np.linalg.norm((_oc_n - self._pinch())[:2]) * 1000:.0f}mm "
                          f"(arm reach limit?)", flush=True)
                print(f">>> enclose: pinch z {float(self._pinch()[2]):.3f} (was {_z0_e:.3f})",
                      flush=True)
            except Exception as _e_en:
                print(f">>> enclose: failed ({_e_en})", flush=True)


    def _ramp_arm(self, q_from, q_to, n_steps=8, settle=0.02):
        """Interpolate to a new arm pose over n steps, settling at each, instead of snapping.

        The arm is a forced parallel linkage, so snapping leaves its LUT-driven passive joints in
        a transient and every measurement taken during it is unusable. Returns False if the
        articulation diverged on the way.
        """
        q_from = np.asarray(q_from, float)
        q_to = np.asarray(q_to, float)
        # Step count scales with distance -- a fixed one turns a large restore into velocity-spiking
        # jumps.
        _dmax = float(np.max(np.abs(q_to - q_from))) if len(q_to) else 0.0
        n_steps = max(int(n_steps), int(_dmax / 0.004))
        n_settle = max(1, int(settle / self.dt))
        # Pin the base for the whole ramp: a boom tilted down and extended forward pushes the
        # chassis, so the no-chassis-push rule holds in every stage that moves the arm, not only the
        # descent.
        try:
            _bx, _by, _byaw = self.base_ledger()
        except Exception:
            _bx = None
        for s_i in range(1, n_steps + 1):
            qi = q_from + (q_to - q_from) * (float(s_i) / n_steps)
            if _bx is not None:
                self.set_base(_bx, _by, _byaw)
            self._force(qi)
            self._apply(qi)
            for _ in range(n_settle):
                if _bx is not None:
                    self.set_base(_bx, _by, _byaw)
                self.world.step(render=not HEADLESS)
            pz = float(self._pinch()[2])
            if not np.isfinite(pz) or abs(pz) > 10.0:
                return False
        return True

    def _close_align_eq(self, st):
        q, obj = st.q, st.obj
        # Equalise the three surface gaps so no finger arrives alone. A rigid hand translation is
        # zero-sum across the jaws, so the shift that makes them equal is exact.
        if os.environ.get("ALIGN_EQ", "1") == "1" and PERFECT:
            try:
                for _it in range(6):
                    _oce = np.asarray(obj.get_world_poses()[0][0], float)
                    _oRe = self._obj_R(obj)
                    _gfn = (self._wrap_gap if os.environ.get("EQ_ALL_LINKS", "1") == "1"
                            else self._finger_surface_gap)
                    _ge = {f_: _gfn(f_, _oce, KNOWN["object_radius"], self.obj_half_h, _oRe)
                           for f_ in "abc"}
                    _pa_e = self._finger_world_pts("a").mean(axis=0)
                    _bc_e = 0.5 * (self._finger_world_pts("b").mean(axis=0)
                                   + self._finger_world_pts("c").mean(axis=0))
                    _jx_e = (_bc_e - _pa_e)[:2]
                    _jx_e = _jx_e / max(1e-9, float(np.linalg.norm(_jx_e)))
                    _gbc = min(_ge["b"], _ge["c"])
                    _d_e = 0.5 * (_ge["a"] - _gbc)          # >0 => thumb further => move toward bc
                    if abs(_d_e) < 0.0015:
                        print(f">>> align-eq: gaps balanced "
                              f"{ {k: round(v * 1000, 1) for k, v in _ge.items()} }mm", flush=True)
                        break
                    _d_e = float(np.clip(_d_e, -0.03, 0.03))
                    # Move the arm, not the chassis. A mobile manipulator does not creep its whole
                    # chassis to correct a few millimetres of finger gap, and dragging the base
                    # couples yaw error into a lateral shove.
                    if os.environ.get("ALIGN_EQ_ARM", "1") == "1":
                        self.start_jog()
                        self.jog["target"] = self.grip_pos() + np.array(
                            [_d_e * _jx_e[0], _d_e * _jx_e[1], 0.0])
                        for _ke in range(int(0.6 / self.dt)):
                            _err_e = self._jog_tick()
                            self.world.step(render=not HEADLESS)
                            if _err_e < 0.002:
                                break
                    else:
                        _bxe, _bye, _byawe = self.base_ledger()
                        _qe = np.asarray(self.robot.get_joint_positions(), float).copy()
                        _ne = int(0.35 / self.dt)
                        for _ke in range(_ne):
                            _fe = 0.5 - 0.5 * math.cos(math.pi * (_ke + 1) / _ne)
                            self.set_base(float(_bxe + _fe * _d_e * _jx_e[0]),
                                          float(_bye + _fe * _d_e * _jx_e[1]), _byawe)
                            self._force(_qe)
                            self._apply(_qe)
                            self.world.step(render=not HEADLESS)
                    print(f">>> align-eq[{_it}]: gaps "
                          f"{ {k: round(v * 1000, 1) for k, v in _ge.items()} }mm -> shifted "
                          f"{_d_e * 1000:+.1f}mm toward b/c", flush=True)
            except Exception as _e_eq:
                print(f">>> align-eq: failed ({_e_eq})", flush=True)
