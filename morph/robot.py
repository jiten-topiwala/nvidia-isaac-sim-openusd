"""Base and articulation write primitives: the lowest layer everything else calls. `set_base`
is the only mover of the chassis, `_force`/`_apply` the only joint writers. Imports Isaac APIs at
module level, so it is importable only after SimulationApp exists."""
import json
import math
import os

import numpy as np
import scene
from isaacsim.core.utils.types import ArticulationAction

from morph.config import (KNOWN, HEADLESS, HERE, ARM_DRIVE, ARM_MODEL)
from morph.geometry import quat_yaw, yaw_of, wrap

# Tracking tolerances are per-joint, from drive limits rather than one global threshold: a static
# load hits drive saturation, so `_track_tols` bounds each joint by its own saturation error.
TRACK_HEADROOM = 1.5    # multiplier above a drive's OWN saturation error before a reading means
                        # "not tracking". Clears the instrument's noise and still fails at 10 mrad.
                        # CHOSEN, not measured.
TRACK_STEP_CLIP_M = 0.004   # the insert trim's per-step prismatic clip (its common-mode and a1
                            # corrections). A drive following at v lags v*kd/kp before anything is wrong;
                            # a PLANNED path is retimed under the joint ceilings and never reaches this.
LUT_REPORT_TOL = 0.005      # REPORTING floor for the four-bar LUT-vs-physics diagnostic below, in
                            # each joint's own units. Not a gate and not derived: nothing fails on
                            # it, it only decides when the disagreement is worth a line.

# Candidates. `_track_tols` filters this at runtime -- a joint the gain rule gives kp = 0, or one
# the closure LUT writes as a four-bar passive, is not doing command tracking and is not gated.
TRACK_GATED_JOINTS = ("ColumnLeftBearingJoint_1", "ColumnRightBearingJoint_1", "ArmLeftJoint_1",
                      "ArmRightJoint_1", "RotationLeftJoint_1", "gripper_x_rotation_1",
                      "gripper_y_rotation_1")

# TRACK_TRACE=1 (OFF by default): one CSV row per watched joint per step. `qd_target` empty means
# no velocity commanded (0.0 means commanded zero); `effort_meas` is joint force, not drive torque.
TRACK_TRACE_COLS = ("step", "stage", "joint", "kind", "q_cmd", "q_act", "qd_meas", "qd_target",
                    "v_written", "effort_meas", "kp", "kd", "max_effort")


class RobotMixin:
    import contextlib
    @contextlib.contextmanager
    def _time_stage(self, stage_name, file_line=None):
        self._current_stage_name = stage_name
        if __import__("os").environ.get("TRACK_ERR", "0") == "1":
            import time
            import inspect
            if file_line is None:
                frame = inspect.currentframe().f_back.f_back
                file_line = f"{frame.f_code.co_filename.split('Isaac_sim/')[-1]}:{frame.f_lineno}"
            t0 = time.time()
            yield
            dt = time.time() - t0
            if not hasattr(self, "_bench_stages"): self._bench_stages = []
            self._bench_stages.append((stage_name, file_line, dt))
        else:
            yield

    """Mixed into Demo. Every `self.*` it touches is owned by Demo."""

    def base_pose(self):
        p, q = self.robot.get_world_pose()
        return float(p[0]), float(p[1]), yaw_of(q)

    def base_ledger(self):
        """Intended base pose (exact): what WE last commanded, immune to read lag."""
        return getattr(self, "_bl", None) or self.base_pose()

    # Mecanum mixer (cf, cl, cw) per hub: rate = (cf*fwd + cl*lat + cw*wz*D) / r.
    _MIX = {"front_left_wheel_rolling_joint": (1.0, -1.0, -1.0),
            "front_right_wheel_rolling_joint": (1.0, 1.0, 1.0),
            "back_left_wheel_rolling_joint": (1.0, 1.0, -1.0),
            "back_right_wheel_rolling_joint": (1.0, -1.0, 1.0)}

    def _hub_rates(self, fwd, lat, wz):
        """Hub rates {dof: rad/s} for body-frame (fwd, lat, wz); r/D/gain are calibrated."""
        r = float(os.environ.get("NAV_WHEEL_R", "0.127"))
        D = float(os.environ.get("NAV_WHEEL_D", "0.64"))
        gl = float(os.environ.get("NAV_LAT_GAIN", "1.96"))
        out = {}
        for n, (cf, cl, cw) in self._MIX.items():
            i = self.idx.get(n)
            if i is not None:
                out[i] = (cf * fwd + cl * gl * lat + cw * wz * D) / r
        return out

    def _hub_effort(self, nm):
        """Max effort on the four hubs (Nm); ~2 Nm accelerates the base, 20 Nm caps the rest."""
        try:
            _g = getattr(self, "_gains", None)
            eff = (np.full(len(self.names), 1.0e6) if _g is None or _g[2] is None else _g[2].copy())
            for n in self._MIX:
                if n in self.idx:
                    eff[self.idx[n]] = float(nm)
            self.ctrl.set_max_efforts(eff)
            if _g is not None:
                self._gains = (_g[0], _g[1], eff)
        except Exception as _e:
            print(f">>> hub effort unavailable ({_e})", flush=True)

    def _spin_wheels_mecanum(self, fwd, lat, yaw_d):
        """Roll each wheel by what a mecanum base would need for THIS body step. Cosmetic; twelve
        rollers repeat every 30 deg, so past ~15 deg per rendered frame the spin aliases."""
        if not hasattr(self, "_wheel_ii"):
            names = ["front_left_wheel_rolling_joint", "front_right_wheel_rolling_joint",
                     "back_left_wheel_rolling_joint", "back_right_wheel_rolling_joint"]
            self._wheel_ii = [(n, self.idx[n]) for n in names if n in self.idx]
            self._wheel_a = {n: 0.0 for n, _ in self._wheel_ii}
            print(f">>> wheel spin: {len(self._wheel_ii)} rolling joints (mecanum mixer)", flush=True)
        if not self._wheel_ii:
            return
        # No wheel motion while pinned: a drift re-assert arrives as a position delta.
        if getattr(self, "_pin_orig_step", None) is not None:
            try:
                _pi = np.array([i for _, i in self._wheel_ii])
                if len(_pi):
                    self.robot.set_joint_velocities(np.zeros(len(_pi)), joint_indices=_pi)
            except Exception:
                pass
            return
        r = float(os.environ.get("WHEEL_R", "0.1"))                 # wheel radius, m
        D = 0.55                                                    # wheelbase + track, m
        mix = self._MIX
        try:
            _wvc = 6.0
            q = np.asarray(self.robot.get_joint_positions(), float)
            for n, i in self._wheel_ii:
                cf, cl, cw = mix[n]
                _d = (cf * fwd + cl * lat + cw * yaw_d * D) / r
                if _wvc > 0.0:
                    _rate = _d / self.dt
                    _d = _wvc * math.tanh(_rate / _wvc) * self.dt
                # Relative, wrapped: an absolute accumulator desyncs wherever the wheels are held.
                q[i] = (float(q[i]) + _d + math.pi) % (2.0 * math.pi) - math.pi
                self._wheel_a[n] += _d
            ii = np.array([i for _, i in self._wheel_ii])
            self.robot.set_joint_positions(q[ii], joint_indices=ii)
            # Zero wheel velocity ONLY while stationary, or the solver integrates between writes.
            if os.environ.get("WHEEL_VEL_ZERO", "1") == "1" and \
                    abs(fwd) < 1e-6 and abs(lat) < 1e-6 and abs(yaw_d) < 1e-6:
                self.robot.set_joint_velocities(np.zeros(len(ii)), joint_indices=ii)
        except Exception as _e_ws:
            if not getattr(self, "_ws_warned", False):
                self._ws_warned = True
                print(f">>> wheel spin: WRITE FAILED ({_e_ws}) -- wheels will not turn; "
                      f"the mixer is fine, the joint write is not", flush=True)

    def _drive_on(self):
        """True when ARM_DRIVE covers `self._stage`. ARM_DRIVE_STAGE: "all" or a comma list."""
        if not ARM_DRIVE:
            return False
        _al = os.environ.get("ARM_DRIVE_STAGE", "all")
        return _al == "all" or getattr(self, "_stage", "") in [x.strip() for x in _al.split(",")]

    def set_base(self, x, y, yaw, force=False):
        # Under the chassis brake only the pin's drift re-assert (force=True) and nav may write.
        if self._drive_on() and getattr(self, "_pin_brake", False) and not force:
            self._base_skip_n = getattr(self, "_base_skip_n", 0) + 1
            return
        # The no-chassis-push rule: during a grasp every gap is closed with the ARM, not the base.
        if getattr(self, "_grasp_locked", None) is not None:
            _px, _py, _pyaw = self._grasp_locked
            if (abs(float(x) - _px) > 1e-6 or abs(float(y) - _py) > 1e-6
                    or abs(wrap(float(yaw) - _pyaw)) > 1e-6):
                _n = getattr(self, "_base_block_n", {})
                try:
                    import traceback
                    _who = traceback.extract_stack()[-2]
                    _key = f"{os.path.basename(_who.filename)}:{_who.lineno}"
                except Exception:
                    _key = "?"
                if _key not in _n:
                    print(f">>> chassis move REFUSED during grasp ({_key} wanted "
                          f"{float(x) - _px:+.3f},{float(y) - _py:+.3f}m) -- close this gap "
                          f"with the ARM, never the base", flush=True)
                _n[_key] = _n.get(_key, 0) + 1
                self._base_block_n = _n
                return
        # ...and only now: a root write invalidates the drive feedforward, but a write REFUSED
        # above did nothing, and killing the feedforward for it costs the next `_apply` v*kd/kp.
        self._drv_prev = None
        if os.environ.get("WHEEL_SPIN", "1") == "1" and getattr(self, "_bl", None) is not None:
            _dx, _dy = float(x) - self._bl[0], float(y) - self._bl[1]
            _hd = self._bl[2]
            _f = _dx * math.cos(_hd) + _dy * math.sin(_hd)
            _l = -_dx * math.sin(_hd) + _dy * math.cos(_hd)
            self._spin_wheels_mecanum(_f, _l, wrap(float(yaw) - self._bl[2]))
        self._bl = (float(x), float(y), float(yaw))
        # Global base ledger: `base_pose()` is stale for a beat after any `set_base`.
        try:
            _zl = float(self.robot.get_world_pose()[0][2])
            _zpin = getattr(self, "_pin_z", None)
            if _zpin is not None:
                # Under a pin, hold the height captured when the pin armed (`_pin_base` is xy/yaw).
                _zp = float(_zpin)
            else:
                _zp = (_zl if abs(_zl - self.z0) < 0.02
                       else self.z0)
        except Exception:
            _zp = self.z0
        self.robot.set_world_pose(position=[x, y, _zp], orientation=quat_yaw(yaw))
        # set_world_pose writes position only; residual velocity keeps pushing the base.
        try:
            self.robot.set_velocities(np.zeros((1, 6)))
        except Exception:
            try:
                self.robot.set_velocities(np.zeros(6))
            except Exception:
                pass

    def _with_closure(self, q):
        """Write the linkage's PASSIVE joints for the commanded (h2-h1, a1) from the closure LUT.
        The arm runs NO_LOOP, so these passives are what actually ANGLE THE BOOM."""
        q = np.asarray(q, float).copy()
        if not getattr(self, "clut", None):
            return q
        i_h1 = self.idx.get("ColumnLeftBearingJoint_1")
        i_h2 = self.idx.get("ColumnRightBearingJoint_1")
        i_a1 = self.idx.get("ArmLeftJoint_1")
        if i_h1 is None or i_h2 is None or i_a1 is None or len(q) <= max(i_h1, i_h2, i_a1):
            return q
        try:
            for _n, _v in self._closure_passives(float(q[i_h2] - q[i_h1]), float(q[i_a1])).items():
                _j = self.idx.get(_n)
                if _j is not None and _j < len(q):
                    q[_j] = float(_v)
        except Exception:
            pass
        return q


    def _force(self, q, qv=None):
        """Kinematic joint write, grasp-mode aware; a no-op under the drive gate. The finger DOFs
        are excluded while `_finger_cmd` is set, so real contact can stop them. qv must be the
        velocities of a MOVING write: friction is velocity-based, and v=0 is a stationary wall."""
        if self._drive_on():
            return
        self._drv_prev = None
        q = self._with_closure(q)
        if self._finger_cmd is not None:
            ii = self._arm_ii
            self.robot.set_joint_positions(q[ii], joint_indices=ii)
            self.robot.set_joint_velocities(
                np.zeros(len(ii)) if qv is None else qv[ii], joint_indices=ii)
        else:
            # ...and never the WHEELS/ROLLERS: parked angles here erase the mecanum roll.
            ii = getattr(self, "_hold_ii", None)
            if ii is None:
                self.robot.set_joint_positions(q)
                self.robot.set_joint_velocities(
                    np.zeros(len(self.names)) if qv is None else qv)
            else:
                self.robot.set_joint_positions(q[ii], joint_indices=ii)
                self.robot.set_joint_velocities(
                    np.zeros(len(ii)) if qv is None else qv[ii], joint_indices=ii)

    def _prismatic_names(self):
        """Prismatic joint names, out of the arm model the FK and the planner already read, plus
        the parked arm's `_2` twins. Every other DOF on this machine is revolute, so an unknown
        name reads as revolute -- which only ever makes a tolerance TIGHTER (no lag term)."""
        _p = getattr(self, "_prism_n", None)
        if _p is None:
            try:
                _p = {j["name"] for j in json.load(open(ARM_MODEL))["joints"]
                      if j.get("type") == "prismatic"}
            except Exception as _e_p:
                _p = set()
                print(f">>> track: no joint types from {ARM_MODEL} ({_e_p}) -- every joint reads "
                      f"as revolute, so prismatic tolerances lose their velocity-lag term and the "
                      f"units below are wrong for them", flush=True)
            _p |= {n[:-2] + "_2" for n in _p if n.endswith("_1")}
            self._prism_n = _p
        return _p

    def _track_tols(self):
        """Per-joint tracking tolerances: rows (name, index, tol, unit, kind); kind 'gate' fails the
        run, 'lut' only diagnoses passive agreement. Tol = TRACK_HEADROOM * (max_effort/kp + v*kd/kp
        for prismatics). Outside ARM_DRIVE_STAGE the arm is teleported, so NO joint is gated."""
        _c = getattr(self, "_track_tab", None)
        if _c is not None and _c[0] == self._drive_on():
            return _c[1]
        _prism = self._prismatic_names()
        _lut = tuple((getattr(self, "clut", None) or {}).get("passives", ()))
        _v = TRACK_STEP_CLIP_M / self.dt            # fastest commanded prismatic rate, m/s
        _tab, _drop = [], []
        for _j in dict.fromkeys(tuple(TRACK_GATED_JOINTS) + _lut):
            if _j not in self.names:
                continue
            _u = "mm" if _j in _prism else "mrad"
            if _j in _lut:
                _tab.append((_j, self.names.index(_j), LUT_REPORT_TOL, _u, "lut"))
                if _j in TRACK_GATED_JOINTS:
                    _drop.append(f"{_j} (four-bar passive written from the closure LUT)")
                continue
            if not self._drive_on():
                continue
            _kp, _kd, _ef = self._arm_gain(_j, 3.0e5, wrist_rule=True)
            if _kp <= 0.0:
                _drop.append(f"{_j} (no position drive at all: kp 0)")
                continue
            _tol = TRACK_HEADROOM * (_ef / _kp + (_v * _kd / _kp if _j in _prism else 0.0))
            _tab.append((_j, self.names.index(_j), _tol, _u, "gate"))
        if _drop and not getattr(self, "_track_drop_said", False):
            self._track_drop_said = True
            print(f">>> track gate: dropped from the gated set, these are not command tracking -- "
                  f"{'; '.join(_drop)}", flush=True)
        self._track_tab = (self._drive_on(), _tab)
        return _tab

    def _track_trace_row(self, watch, stage, q_cmd, q_act, act, v, ff):
        """Write one per-step diagnostic CSV row to track_trace.csv (TRACK_TRACE=1).

        Rides the tracking check's position readback; adds qd_meas and effort_meas reads.
        qd_target is empty when no velocity target was commanded (ff None or False)
        to distinguish from commanded zero velocity. effort_meas is link-incoming force."""
        def _cell(a, i):
            return "" if a is None or i >= len(a) else float(a[i])

        _w = getattr(self, "_track_trace_w", None)
        if _w is None:
            import atexit
            import csv
            # Beside PLACE_RESULTS; unset, the same `<repo>/logs` play_isaac.py defaults it to.
            _dir = os.path.dirname(os.environ.get("PLACE_RESULTS", "")) or os.path.join(HERE, "logs")
            os.makedirs(_dir, exist_ok=True)
            _p = os.path.join(_dir, "track_trace.csv")
            _f = self._track_trace_f = open(_p, "w", newline="")
            # the unflushed tail is data; do not lose it on exit
            atexit.register(_f.close)
            _w = self._track_trace_w = csv.writer(_f)
            _w.writerow(TRACK_TRACE_COLS)
            print(f">>> TRACK_TRACE=1: per-step joint trace -> {_p}. This run takes two extra "
                  f"readbacks per step (joint velocities, measured efforts) that an unflagged "
                  f"run does not, so its timing is NOT a baseline.", flush=True)
        try:
            _qd = np.asarray(self.robot.get_joint_velocities(), float).ravel()
        except Exception:
            _qd = None                       # a missing column is honest; a raise would be a run
        # lost to its own instrument
        try:
            _ef = np.asarray(self.robot.get_measured_joint_efforts(), float).ravel()
        except Exception:
            _ef = None
        _kp, _kd, _ef_max = getattr(self, "_gains", None) or (None, None, None)
        _vt = None if not ff else act.joint_velocities
        _n = getattr(self, "_track_err_n", 0)
        for _j, _i, _t, _u, _k in watch:
            _w.writerow([_n, stage, _j, _k, _cell(q_cmd, _i), _cell(q_act, _i), _cell(_qd, _i),
                         _cell(_vt, _i), _cell(v, _i), _cell(_ef, _i),
                         _cell(_kp, _i), _cell(_kd, _i), _cell(_ef_max, _i)])
        # One flush per step, not per row: ~800 bytes to the page cache against a 4.16 ms step,
        # and the alternative is a run killed at the viewport losing the tail it was taken for.
        self._track_trace_f.flush()

    def _pin_fingers(self, q):
        """Hold ARM ONE's fingers at the pin the running motion published, KINEMATICALLY: their
        drives carry the friction grasp's own compliance and cannot hold a free-space pose. Inert
        while `_finger_cmd` is set, which is the whole of the grip."""
        pin = self._finger_pin
        if pin is None or self._finger_cmd is not None or not self._drive_on():
            return q
        _m = self._f1_mask
        if _m is None:
            # `_f_idx` carries arm TWO's fingers too; the model's joints are the same hand
            # `commanded_fingers` checks, and ARM2_FREEZE below already parks the rest.
            _known = self._arm_model().joints
            _m = self._f1_mask = np.array([self.names[i] in _known for i in self._f_idx])
        if not _m.any():
            return q
        ii, val = np.asarray(self._f_idx)[_m], np.asarray(pin, float)[_m]
        q = np.asarray(q, float).copy()
        q[ii] = val
        self.robot.set_joint_positions(val, joint_indices=ii)
        # Position without velocity leaves the solver integrating the sag the write just undid.
        self.robot.set_joint_velocities(np.zeros(len(ii)), joint_indices=ii)
        return q

    def _apply(self, q, qv=None):
        """Drive-target write through apply_action, surviving a dropped articulation view: a
        paused viewport or USD churn invalidates the physics view, so revive once and retry rather
        than crashing. While the fingers own an object their targets come from `_finger_cmd`."""
        q = self._with_closure(q)
        if self._finger_cmd is not None:
            q = np.asarray(q, float).copy()
            q[self._f_idx] = self._finger_cmd
        q = self._pin_fingers(q)
        # Rolling joints are velocity drives (kp 0): a position target fights the free spin.
        _rii = getattr(self, "_roll_ii", None)
        if _rii is not None and len(_rii):
            _live = np.asarray(self.robot.get_joint_positions(), float)
            # A dropped view returns a 0-d array, not an AttributeError the retry below catches.
            if _live.ndim == 1 and _live.size > int(np.max(_rii)):
                q = np.asarray(q, float).copy()
                q[_rii] = _live[_rii]
        _hq = getattr(self, "_pin_hub_q", None)              # parking brake: hub targets frozen
        if _hq is not None and getattr(self, "_pin_brake", False):
            q = np.asarray(q, float).copy()
            q[self._pin_hub_ii] = _hq
        _track_target = getattr(self, "_drv_prev", None)
        if self._drive_on() and os.environ.get("ARM2_FREEZE", "1") == "1":
            # Park the unused second arm: its loop is not in the closure LUT, so it would sag.
            _a2 = getattr(self, "_arm2_ii", None)
            if _a2 is None:
                _a2 = np.asarray([i for n, i in self.idx.items() if n.endswith("_2")], int)
                self._arm2_ii = _a2
                self._arm2_q = (np.asarray(self.robot.get_joint_positions(), float)[_a2].copy()
                                if _a2.size else None)
                if _a2.size:
                    print(f">>> arm 2 frozen at its park pose ({_a2.size} joints; it is unused and "
                          f"its loop is not in the closure LUT, so on drives it would sag)", flush=True)
            if _a2.size and getattr(self, "_arm2_q", None) is not None:
                q = np.asarray(q, float).copy()
                q[_a2] = self._arm2_q
        if self._drive_on():
            # Clip to the joint limits: a drive fighting its own stop goes non-finite.
            _lim = getattr(self, "_q_lim", None)
            if _lim is None:
                try:
                    _dp = self.robot.dof_properties
                    _lo, _hi = np.asarray(_dp["lower"], float), np.asarray(_dp["upper"], float)
                    _ok = np.isfinite(_lo) & np.isfinite(_hi) & (_hi > _lo) & (np.abs(_lo) < 1e6) & (np.abs(_hi) < 1e6)
                    _lo, _hi = np.where(_ok, _lo, -np.inf), np.where(_ok, _hi, np.inf)
                    _lim = self._q_lim = (_lo, _hi)
                except Exception as _e_l:
                    _lim = self._q_lim = False
                    print(f">>> _apply: joint limits unavailable ({_e_l}); targets unclipped", flush=True)
            if _lim:
                _qc = np.clip(np.asarray(q, float), _lim[0], _lim[1])
                _over = np.abs(_qc - np.asarray(q, float)) > 1e-4
                if np.any(_over):
                    self._drv_clip_n = getattr(self, "_drv_clip_n", 0) + 1
                    if self._drv_clip_n <= 3:
                        print(f">>> _apply: target beyond the joint limit, clipped: "
                              f"{[(self.names[i], round(float(q[i]), 3)) for i in np.flatnonzero(_over)[:4]]}", flush=True)
                    q = _qc
            # Kinematic-stage gains carry the kinematic rule: re-run the last writer on entry.
            _gw = getattr(self, "_gains_writer", None)
            if _gw is not None and not getattr(self, "_gains_rule_drive", False):
                _fn, _kw = _gw
                getattr(self, _fn)(**_kw)
            # Velocity feedforward: without v_t the drive lags by v*kd/kp, and a None entry
            # keeps the PREVIOUS target in the view.
            q = np.asarray(q, float)
            _prev = _track_target
            _ff = True          # was a velocity target really COMMANDED this step? False marks the
                                # zeros filler below, None the kinematic branch that writes none at
                                # all. Read only by the trace: nothing here changes what is written.
            if qv is not None:
                v = np.asarray(qv, float).copy()
            elif _prev is None or _prev.shape != q.shape:
                v = np.zeros(len(q))
                _ff = False
            else:
                v = (q - _prev) / self.dt
                _aj = getattr(self, "_arm_ii", None)         # fingers/rollers excluded
                _dq = np.abs(q - _prev) if _aj is None else np.abs(q - _prev)[_aj]
                _jm = _dq > 0.005                            # a jump (snap/sync), not motion
                if _jm.any():
                    # ONLY the joints that jumped: zeroing qd_target on a joint that is genuinely moving is
                    # a brake pulse, not a reset. `_dq` is in `_aj` coordinates -- scatter the mask back.
                    v[_jm if _aj is None else np.asarray(_aj)[_jm]] = 0.0
                    self._drv_jump_n = getattr(self, "_drv_jump_n", 0) + 1
            if getattr(self, "_f_idx", None) is not None and len(self._f_idx):
                # fingers: force-limited PD, no ff
                v[self._f_idx] = 0.0
            if _rii is not None and len(_rii):
                # rollers free; hubs: brake target 0
                v[_rii] = 0.0
                _hv = getattr(self, "_hub_vel", None)         #   or the mecanum mixer's rate (nav)
                if _hv is not None:
                    for _hi, _hw in _hv.items():
                        v[_hi] = float(_hw)
            self._drv_prev = q.copy()
            _act = ArticulationAction(joint_positions=q, joint_velocities=v)
        else:
            v = _ff = None                       # no velocities in the action at all
            _act = ArticulationAction(joint_positions=q)
        # Non-finite guard: NaN targets cascade into a NaN articulation. Hold them live instead.
        try:
            _qa = np.asarray(_act.joint_positions, float)
            _bad = ~np.isfinite(_qa)
            if _bad.any():
                _live = np.asarray(self.robot.get_joint_positions(), float)
                _prev_ok = getattr(self, "_drv_prev", None)
                _fill = (_prev_ok if _prev_ok is not None and _prev_ok.shape == _qa.shape
                         and np.all(np.isfinite(_prev_ok)) else _live)
                _fill = np.where(np.isfinite(_fill), _fill, 0.0)      # the articulation itself may be NaN
                _qa = np.where(_bad, _fill, _qa)
                self._nonfinite_n = getattr(self, "_nonfinite_n", 0) + 1
                _act.joint_positions = _qa
                if _act.joint_velocities is not None:
                    _va = np.asarray(_act.joint_velocities, float)
                    _act.joint_velocities = np.where(~np.isfinite(_va) | _bad, 0.0, _va)
                if not getattr(self, "_nan_warned", False):
                    self._nan_warned = True
                    print(f">>> _apply: NON-FINITE targets for {[self.names[i] for i in np.where(_bad)[0][:6]]} "
                          f"-> held at their live positions (printed once)", flush=True)
        except Exception:
            pass
        # Never gates anything IN SIM: it logs commanded-vs-actual and lets the run continue, and is the
        # only source of the `TRACK BREACH` lines verify_place grades afterwards.
        if os.environ.get("TRACK_ERR", "0") == "1":
            try:
                self._track_err_n = getattr(self, "_track_err_n", 0) + 1
                _act_pos = np.asarray(self.robot.get_joint_positions(), float)
                _gate = self._track_tols()
                _gate_ready = (
                    self._drive_on() and _track_target is not None
                    and np.asarray(_track_target).shape == _qa.shape
                    and np.all(np.isfinite(_track_target))
                )
                _track_cmd = _qa.copy()
                for _gj, _gi, _gtol, _gu, _gk in _gate:
                    if _gk == "gate":
                        _track_cmd[_gi] = _track_target[_gi] if _gate_ready else np.nan
                _valid = np.isfinite(_track_cmd) & np.isfinite(_act_pos)
                _err = np.full(_track_cmd.shape, np.nan)
                _err[_valid] = np.abs(_track_cmd[_valid] - _act_pos[_valid])
                # Gate check: every step, never throttled -- a transient breach between the informational
                # samples is what must not be missed. Each (joint, stage) prints once.
                _breached = getattr(self, "_track_breached", None)
                if _breached is None:
                    _breached = self._track_breached = set()
                _stage = getattr(self, "_stage", "?")
                # Live per-step signal: how far past tolerance the worst GATED joint is, as a
                # ratio. The breach prints below are deduped per (joint, stage) and cannot serve.
                _over = [float(_err[_i]) / _t for _j, _i, _t, _u, _k in _gate
                         if _k == "gate" and _valid[_i] and _t > 0.0]
                self._track_over = max(_over) if _over else 0.0
                for _gj, _gi, _gtol, _gu, _gk in _gate:
                    if not _valid[_gi] or float(_err[_gi]) <= _gtol:
                        continue
                    _key = (_gj, _stage)
                    if _key in _breached:
                        continue
                    _breached.add(_key)
                    _e = float(_err[_gi]) * 1000.0
                    if _gk == "gate":
                        print(f">>> TRACK BREACH[{_stage}]: {_gj} cmd {float(_track_cmd[_gi]):+.4f} "
                              f"act {float(_act_pos[_gi]):+.4f} err {_e:.2f}{_gu} "
                              f"(tol {_gtol * 1000:.2f}{_gu}, this drive's own floor plus "
                              f"headroom) -- the actuated chain does not track here; every pose "
                              f"computed downstream is suspect.", flush=True)
                    else:
                        print(f">>> LUT-PHYSICS DISAGREE[{_stage}]: {_gj} lut "
                              f"{float(_qa[_gi]):+.4f} act {float(_act_pos[_gi]):+.4f} err "
                              f"{_e:.2f}{_gu} -- the closure LUT and physics disagree on this "
                              f"four-bar passive. Real, and worth knowing; NOT command tracking, "
                              f"and graded by nothing.", flush=True)
                if self._track_err_n % int(0.5 / self.dt) == 0:
                    _wi = int(np.nanargmax(_err))
                    _wu = "mm" if self.names[_wi] in self._prismatic_names() else "mrad"
                    _ng = sum(1 for _r in _gate if _r[4] == "gate")
                    print(f">>> track-err[{_stage}]: worst "
                          f"{self.names[_wi]} cmd {float(_track_cmd[_wi]):+.4f} "
                          f"act {float(_act_pos[_wi]):+.4f} err {float(_err[_wi]) * 1000:+.2f}{_wu} "
                          f"| p95 {float(np.nanpercentile(_err, 95)) * 1000:.2f} | "
                          + (f"gate {_ng} joints" if _gate_ready
                             else "gate awaiting previous target" if _ng
                             else "gate off (this stage teleports; no drive to track)"),
                          flush=True)
                if os.environ.get("TRACK_TRACE", "0") == "1":
                    # Its OWN try: a recorder that throws must not take the gate above down with it. The
                    # command and readback are already in hand, so the trace adds no third position read.
                    try:
                        self._track_trace_row(_gate, _stage, _qa, _act_pos, _act, v, _ff)
                    except Exception as _e_tr:
                        if not getattr(self, "_track_trace_warned", False):
                            self._track_trace_warned = True
                            print(f">>> TRACK TRACE FAILED[{_stage}]: {type(_e_tr).__name__}: "
                                  f"{_e_tr} -- no per-step trace is being recorded (printed once)",
                                  flush=True)
            except Exception as _e_t:
                # NOT silent: swallowed, a readback raising every step looks exactly like an instrument that
                # is quiet because nothing is wrong. Printed once -- this sits in the per-step write path.
                if not getattr(self, "_track_err_warned", False):
                    self._track_err_warned = True
                    print(f">>> TRACK INSTRUMENT FAILED[{getattr(self, '_stage', '?')}]: "
                          f"{type(_e_t).__name__}: {_e_t} -- the tracking check is not reporting, "
                          f"so nothing downstream can tell a tracked run from an unmeasured one "
                          f"(printed once)", flush=True)
        try:
            self.ctrl.apply_action(_act)
        except AttributeError:
            print(">>> articulation view dropped (viewport paused/stopped?) -> revive + retry", flush=True)
            try:
                self.robot.initialize()
            except Exception:
                # Timeline STOPPED: the physics view is destroyed and the sim state is reset.
                import omni.timeline
                omni.timeline.get_timeline_interface().play()
                for _ in range(3):
                    self.world.step(render=not HEADLESS)
                self.robot.initialize()
            self.ctrl = self.robot.get_articulation_controller()
            try:
                _bx_r, _by_r, _byaw_r = (self._grasp_locked
                                         if getattr(self, "_grasp_locked", None) is not None
                                         else self.base_ledger())
                self.set_base(_bx_r, _by_r, _byaw_r)
                print(f">>> revive: base re-asserted at ({_bx_r:.3f}, {_by_r:.3f})", flush=True)
            except Exception as _e_rv:
                print(f">>> revive: base re-assert failed ({_e_rv})", flush=True)
            # Replay the CURRENT gains and brake, not the load-time defaults.
            _g = getattr(self, "_gains", None)
            if self._drive_on() and _g is not None:
                self._set_gains(*_g)
                if getattr(self, "_pin_brake", False) and hasattr(self, "_pin_brake_apply"):
                    self._pin_brake_apply(True)
            else:
                scene.set_arm_gains(self.robot, self.names)
            self.ctrl.apply_action(_act)

    def park(self):
        self.robot.set_joint_positions(self.q0)
        self._drv_prev = None
        for _ in range(int(0.3 / self.dt)):
            # kinematic: NO_LOOP has no loop joint to hold the passives
            self._force(self.q0)
            self._apply(self.q0)
            self.world.step(render=not HEADLESS)

    def _hold(self, cmd, steps, kin=False, on_step=None):
        """Hold base + arm fixed for `steps`. kin=True also pins the arm KINEMATICALLY (forced
        positions + zero velocity each step) so the grasp config cannot oscillate against the
        linkage loop; the drive still commands `cmd` underneath."""
        bx, by, byaw = self.base_ledger()
        # A drive-based hold must not rewrite the root: a teleport drops the friction anchors.
        _root_pin = not ((os.environ.get("LIFT_DRIVE", "0") == "1" and not kin)
                         or getattr(self, "_pin_brake", False)
                         or self._drive_on())
        for _ in range(steps):
            if _root_pin:
                self.set_base(bx, by, byaw)
            if kin:
                self._force(cmd)
            self._apply(cmd)
            if on_step is not None:
                on_step()
            self.world.step(render=not HEADLESS)

    def _park_ramp(self, base=None, step_fn=None, t=1.0, note="", source="unspecified"):
        """Cosine-ramp the arm into the park pose `q0`, closure passives written every step.
        `base` is held; `step_fn(qk)` replaces the default write+step. Skips if already parked."""
        q_now = np.asarray(self.robot.get_joint_positions(), float).copy()
        q_tgt = np.asarray(self.q0, float)
        if t <= 0 or float(np.max(np.abs(q_tgt - q_now))) <= 1e-3:
            return
        i1 = self.idx.get("ColumnLeftBearingJoint_1")
        i2 = self.idx.get("ColumnRightBearingJoint_1")
        ia = self.idx.get("ArmLeftJoint_1")
        n = max(2, int(t / self.dt))
        # Telemetry records the cycle's FIRST park ramp; run_cycle clears it, a later idle re-ramp must not
        # overwrite it.
        _jidx = self.idx.get("gripper_x_rotation_1")
        _rec = (os.environ.get("TRACK_ERR", "0") == "1" and _jidx is not None
                and getattr(self, "_park_telemetry", None) is None)
        if _rec:
            self._park_telemetry = {
                "pre_step": [], "post_step": [], "cmd_step": [], "step_index": [],
                "path": "step_fn" if step_fn is not None else "world.step",
                "source": source, "drive_on": self._drive_on(),
            }
            _prev_cmd = float(q_now[_jidx])

        for k in range(n):
            f = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / n)
            qk = q_now + f * (q_tgt - q_now)
            if getattr(self, "clut", None) is not None and None not in (i1, i2, ia):
                for pn, pv in self._closure_passives(
                        float(qk[i2]) - float(qk[i1]), float(qk[ia])).items():
                    if pn in self.idx:
                        qk[self.idx[pn]] = float(pv)
            
            if _rec:
                _cmd_k = float(qk[_jidx])
                self._park_telemetry["cmd_step"].append(_cmd_k - _prev_cmd)
                self._park_telemetry["pre_step"].append((_cmd_k, float(self.robot.get_joint_positions()[_jidx])))
                try:
                    _step_before = int(self.world.current_time_step_index)
                except (AttributeError, TypeError, ValueError):
                    _step_before = None
                _prev_cmd = _cmd_k

            if step_fn is not None:
                step_fn(qk)
            else:
                if base: self.set_base(*base)
                self._force(qk)
                self._apply(qk)
                self.world.step(render=not HEADLESS)

            if _rec:
                self._park_telemetry["post_step"].append((float(qk[_jidx]), float(self.robot.get_joint_positions()[_jidx])))
                try:
                    _step_after = int(self.world.current_time_step_index)
                except (AttributeError, TypeError, ValueError):
                    _step_after = None
                self._park_telemetry["step_index"].append((_step_before, _step_after))
        print(f">>> park: ramped to the park pose over {t:.1f}s{note}", flush=True)

    def idle_step(self):
        # FIXED anchor: re-reading base_pose() each step integrates whatever physics pushes.
        if self._idle_anchor is None:
            self._idle_anchor = self.base_pose()
        bx, by, byaw = self._idle_anchor
        if getattr(self, "_pin_orig_step", None) is None:
            self.set_base(bx, by, byaw)
        # Freeze kinematically while idle: parked at a rack dock the arm can graze a shelf.
        if self.grip is None:
            # Ramp into park on the first idle step: writing q0 in one step makes the arm snap.
            if not getattr(self, "_parked_ramped", False):
                self._parked_ramped = True
                if self._drive_on() and not getattr(self, "_pin_brake", False):
                    # no unbraked root writes
                    self._pin_brake_apply(True)
                self._park_ramp(base=(bx, by, byaw), source="idle")
            self._force(self.q0)
        self._apply(self.grip if self.grip is not None else self.q0)
        self.world.step(render=not HEADLESS)

    def _grip_stiffen(self, arm_kp=None):
        """Post-touch hold stiffness. Call AFTER the touch freeze: stiffening first lets the higher
        force press where there is no contact yet."""
        self._gains_writer = ("_grip_stiffen", {"arm_kp": arm_kp})
        kp = np.full(len(self.names), 2.0e3)
        kd = np.full(len(self.names), 2.0e2)
        eff = np.full(len(self.names), 1.0e6)
        for i, n in enumerate(self.names):
            if n.startswith(("finger_", "palm_finger")):
                kp[i], kd[i], _I_f = scene.finger_pd(n)
                # HOLD_FINGER_KP: a position drive keeps grip force only through stiffness.
                _hk = os.environ.get("HOLD_FINGER_KP", "")
                if _hk:
                    kp[i] = float(_hk)
                    # critical vs armature
                    kd[i] = 2.0 * (kp[i] * max(_I_f, 1e-3)) ** 0.5
                eff[i] = float(os.environ.get("HOLD_EFFORT", "25"))
            elif "rolling_joint" in n or "slipping" in n:
                kp[i], kd[i] = self._wheel_gain(n)
        # Arm last, at the CALLER's stiffness: a soft gain here undoes `_set_arm_gain`.
        arm_kp = 3.0e5 if arm_kp is None else arm_kp
        # The arm needs a torque limit too: unlimited, a stiff drive is the same infinite wall.
        for n in KNOWN["arm_joints"]:
            if n in self.idx:
                kp[self.idx[n]], kd[self.idx[n]], eff[self.idx[n]] = self._arm_gain(n, arm_kp, wrist_rule=True)
        # The WRIST is not the columns: kp 3e5 on those light links rings at 240 Hz.
        for n in self._WRIST:
            if n in self.idx and n not in KNOWN["arm_joints"] and (ARM_DRIVE or os.environ.get("WRIST_KP", "")):
                kp[self.idx[n]], kd[self.idx[n]], eff[self.idx[n]] = self._arm_gain(n, arm_kp, wrist_rule=True)
        self._arm2_rows(kp, kd, eff, arm_kp)
        self._set_gains(kp, kd, eff)
        _fi = self.idx.get("finger_a_joint_1_1")
        print(f">>> grip stiffened (hold gains: finger j1 kp {kp[_fi] if _fi is not None else float('nan'):.3g}, "
              f"effort {eff[_fi] if _fi is not None else float('nan'):.3g}Nm, arm kp {arm_kp:.3g})", flush=True)

    def _grip_gentle(self):
        """APPROACH gains: soft fingers, arm unchanged. Must run BEFORE the close loop -- at hold
        stiffness first touch is an impact, not a press, and no stop reacts inside that window.
        kd comes down with kp: at kd=40 travelling at 0.29 rad/s alone costs 11 Nm of damping."""
        self._gains_writer = ("_grip_gentle", {})
        kp = np.full(len(self.names), 2.0e3)
        kd = np.full(len(self.names), 2.0e2)
        eff = np.full(len(self.names), 1.0e6)
        g_kp = 30.0
        g_kd = 1.0
        # GENTLE_EFFORT: close-phase torque ceiling; the limit belongs in the drive.
        g_ef = float(os.environ.get("GENTLE_EFFORT", "5.0"))
        for i, n in enumerate(self.names):
            if n.startswith(("finger_", "palm_finger")):
                kp[i], kd[i], eff[i] = g_kp, g_kd, g_ef
            elif "rolling_joint" in n or "slipping" in n:
                kp[i], kd[i] = self._wheel_gain(n)
        arm_kp = 3.0e5
        for n in KNOWN["arm_joints"]:
            if n in self.idx:
                kp[self.idx[n]], kd[self.idx[n]], eff[self.idx[n]] = self._arm_gain(n, arm_kp)
        for n in self._WRIST:
            if n in self.idx and n not in KNOWN["arm_joints"] and self._drive_on():
                kp[self.idx[n]], kd[self.idx[n]], eff[self.idx[n]] = self._arm_gain(n, arm_kp)
        self._arm2_rows(kp, kd, eff, arm_kp)
        self._set_gains(kp, kd, eff)
        print(f">>> grip GENTLE for approach (fingers kp {g_kp} kd {g_kd} effort {g_ef}Nm)",
              flush=True)

    # `scene.set_arm_gains`, `_hub_effort` and `_pin_brake_apply` bypass these tables and write the
    # drives directly. kd = 2*sqrt(kp*m) off reflected mass (implicit drives: zeta 0.7-1.2, wn*dt<=1).
    _ARM_REFLECTED = {"ColumnLeftBearingJoint_1": 14.2, "ColumnRightBearingJoint_1": 14.2,
                      "ArmLeftJoint_1": 13.8, "BaseJoint_1": 6.0, "RotationLeftJoint_1": 3.2}
    _ARM_LEAVES = ("ArmRightJoint_1", "ContactCylinderJoint_1_1", "ContactCylinderJoint2_1")
    _WRIST = ("HandBearingJoint_1", "gripper_z_rotation_1", "gripper_y_rotation_1",
              "gripper_x_rotation_1", "HandBearingJoint_2", "gripper_z_rotation_2",
              "gripper_y_rotation_2", "gripper_x_rotation_2")
    _ARM_EFFORT = {"ColumnLeftBearingJoint_1": 1500.0, "ColumnRightBearingJoint_1": 1500.0,
                   "ArmLeftJoint_1": 500.0, "BaseJoint_1": 300.0, "RotationLeftJoint_1": 300.0}

    def _wrist_gain(self, n, arm_kp):
        """(kp, kd) for a wrist joint. On drives ARM_DRIVE_WRIST_KP (6e3) with kd critical against
        ~0.11 kg m^2: arm gains of 3e5 on a wrist link ring at omega_n*dt ~ 8."""
        # drive rule first: WRIST_KP is the kinematic-era knob
        if self._drive_on():
            _wk = float(os.environ.get("ARM_DRIVE_WRIST_KP", "6e3"))
            return _wk, 2.0 * (_wk * 0.11) ** 0.5      # I ~0.11 kg m^2: hand + fingers + object at 0.2 m
        _wkp = os.environ.get("WRIST_KP", "")
        if _wkp:
            _wk = float(_wkp)
            return _wk, min(100.0, 2.0 * (_wk * 0.5) ** 0.5)
        return arm_kp, arm_kp * 0.1

    def _arm_gain(self, n, arm_kp, wrist_rule=False):
        """(kp, kd, effort) for an arm joint at stiffness arm_kp. Kinematic: kd = 0.1*kp, effort
        unbounded. On drives: damping from the reflected mass, leaves free, efforts bounded."""
        if n in self._WRIST:
            if self._drive_on() or (wrist_rule and os.environ.get("WRIST_KP", "")):
                kp, kd = self._wrist_gain(n, arm_kp)
                return kp, kd, (30.0 if self._drive_on() else 1.0e6)
            return arm_kp, arm_kp * 0.1, 1.0e6
        if not self._drive_on():
            return arm_kp, arm_kp * 0.1, 1.0e6
        if n in self._ARM_LEAVES:
            return 0.0, 1.0, 1.0e6
        m = self._ARM_REFLECTED.get(n, 1.0)
        return arm_kp, 2.0 * (arm_kp * m) ** 0.5, self._ARM_EFFORT.get(n, 1.0e6)

    def _arm2_rows(self, kp, kd, eff, arm_kp):
        """Apply every `_1` gain rule to the parked second arm's `_2` twin. No-op off the gate."""
        if not self._drive_on():
            return
        twins = tuple(KNOWN["arm_joints"]) + tuple(self._WRIST) + self._ARM_LEAVES
        for n1 in twins:
            n2 = n1[:-2] + "_2" if n1.endswith("_1") else None
            if n2 is None or n2 not in self.idx or n2 in KNOWN["arm_joints"]:
                continue
            i2 = self.idx[n2]
            if n1 in self._ARM_LEAVES or n1 in self._WRIST:
                kp[i2], kd[i2], e2 = self._arm_gain(n1, arm_kp, wrist_rule=True)
            else:
                # Loaded arm-2 joints keep the modest kp: its passives are not LUT-driven.
                m2 = self._ARM_REFLECTED.get(n1, 1.0)
                kp[i2], kd[i2], e2 = kp[i2], 2.0 * (kp[i2] * m2) ** 0.5, self._ARM_EFFORT.get(n1, 1.0e6)
            if eff is not None:
                eff[i2] = e2

    def _set_gains(self, kp, kd, eff=None):
        """The one gain writer: applies and REMEMBERS (kp, kd, eff) for the revive path."""
        kp = np.asarray(kp, float); kd = np.asarray(kd, float)
        eff = None if eff is None else np.asarray(eff, float)
        self._gains = (kp.copy(), kd.copy(), None if eff is None else eff.copy())
        self._gains_rule_drive = self._drive_on()
        self.ctrl.set_gains(kp, kd)
        if eff is not None:
            try:
                self.ctrl.set_max_efforts(eff)
            except Exception:
                pass

    def _wheel_gain(self, n):
        """Drive gains for a wheel HUB (rolling) or mecanum ROLLER (slipping) joint; free by
        default (kp 0, kd 1). Under a grasp pin with PIN_DRIVE_BRAKE=1 the hubs become dampers
        (velocity target 0); kp stays 0 because `_apply` writes the parked hub angles."""
        if getattr(self, "_pin_brake", False) and "wheel" in n and "rolling" in n:
            # PIN_BRAKE_KP > 0 = parking brake (hold at the brake-on angles); 0 = damper only.
            return float(os.environ.get("PIN_BRAKE_KP", "0")), float(os.environ.get("PIN_BRAKE_KD", "1e3"))
        # Rollers ALWAYS free (kp 0); damping comes from the implicit-drive stability bound, not from feel.
        # `scene.set_arm_gains` reads the same variable with the same default; keep them together.
        return 0.0, float(os.environ.get("ROLLER_KD", "0.02"))

    def _set_arm_gain(self, arm_kp):
        """Set the forced arm-1 joints to arm_kp (kd=0.1*kp); rollers free; everything else modest."""
        self._gains_writer = ("_set_arm_gain", {"arm_kp": arm_kp})
        kp = np.full(len(self.names), 2.0e3)
        kd = np.full(len(self.names), 2.0e2)
        eff = np.full(len(self.names), 1.0e6)
        for n in KNOWN["arm_joints"]:
            if n in self.idx:
                kp[self.idx[n]], kd[self.idx[n]], eff[self.idx[n]] = self._arm_gain(n, arm_kp)
        for n in self._WRIST:
            if n in self.idx and n not in KNOWN["arm_joints"] and self._drive_on():
                kp[self.idx[n]], kd[self.idx[n]], eff[self.idx[n]] = self._arm_gain(n, arm_kp)
        for i, n in enumerate(self.names):
            if "rolling_joint" in n or "slipping" in n:
                kp[i], kd[i] = self._wheel_gain(n)
        self._arm2_rows(kp, kd, eff, arm_kp)
        # kinematic era never bounded arm efforts here; keep that (eff only applied under ARM_DRIVE)
        self._set_gains(kp, kd, eff if self._drive_on() else None)
