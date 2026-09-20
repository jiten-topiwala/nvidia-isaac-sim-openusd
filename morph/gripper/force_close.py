"""Force-balanced close: advance each finger independently until its own pad reads `FB_TARGET`,
then hold it while the others keep coming, so no finger arrives alone and shoves a free object.
FB_TARGET follows the grasp, m·g/(3·mu); pressing much harder tips the cylinder at m·g·r/h.
"""
import math
import os

# CLOSE_DRIVE=1: run the ARM on drives through the close too. Default off: the arm is pinned
# kinematically, because an infinite-mass arm re-written each step shoves the object into the palm.
_CLOSE_KIN = os.environ.get("CLOSE_DRIVE", "0") != "1"


def _fb_diag(self, tag, k, obj=None, every=24):
    """FB_DIAG=1: every `every` steps, print object speed, the loudest non-finger joints, wrist
    rates, boom-pitch drive error, root motion and pin hits."""
    if os.environ.get("FB_DIAG", "0") != "1" or k % every:
        return
    try:
        st_ = self.__dict__.setdefault("_fbd", {})
        q = np.asarray(self.robot.get_joint_positions(), float)
        oc = None if obj is None else np.asarray(obj.get_world_poses()[0][0], float)
        rp = np.asarray(self.robot.get_world_pose()[0], float)[:2]
        q0, oc0, rp0 = st_.get("q"), st_.get("oc"), st_.get("rp")
        st_.update(q=q.copy(), oc=None if oc is None else oc.copy(), rp=rp.copy())
        if q0 is None:
            return
        dq = np.abs(q - q0)
        nm = {v: kk for kk, v in self.idx.items()}
        top = [i for i in sorted(range(len(dq)), key=lambda i: -dq[i])
               if "finger" not in str(nm.get(i, "")) and "slipping" not in str(nm.get(i, ""))][:3]
        wr = {n: float((q[self.idx[n]] - q0[self.idx[n]]) / (every * self.dt))
              for n in ("HandBearingJoint_1", "gripper_x_rotation_1", "gripper_y_rotation_1") if n in self.idx}
        ip = self.idx.get("RotationLeftJoint_1")
        dp = getattr(self, "_drv_prev", None)
        bp_err = (float(dp[ip] - q[ip]) * 1000.0 if dp is not None and ip is not None and len(dp) > ip
                  else float("nan"))
        ov = (float(np.linalg.norm(oc - oc0)) / (every * self.dt) * 1000.0 if oc is not None and oc0 is not None
              else float("nan"))
        print(f">>>   fb-diag[{tag} {k:5d}]: obj {ov:6.1f} mm/s | movers "
              + ", ".join(f"{nm.get(i, i)} {dq[i] / (every * self.dt):+.2f}/s" for i in top)
              + " | wrist " + " ".join(f"{n.split('_')[0][:4]}{v:+.2f}" for n, v in wr.items())
              + f" | boom-pitch err {bp_err:+.2f} mrad | root {np.linalg.norm(rp - rp0) * 1000:.1f} mm"
              + f" | pin {getattr(self, '_pin_hits', 0)}", flush=True)
    except Exception:
        pass


import numpy as np

from morph.config import HEADLESS, KNOWN, COLUMN_MAX

# Reference full-curl targets and the ratios between them. ABSOLUTE targets, not a mechanical
# coupling: scaling them off j1's limit drives j2/j3 onto their stops and stalls the close.
J1_LEGAL_MAX = 1.2218                     # vendor legal limit; not the grasp's curl target
J2_PER_J1 = 0.855 / 0.595
J3_PER_J1 = -1.152 / 0.595
J2_RANGE = (0.0, 1.5708)
J3_RANGE = (-1.152, -0.0523)
J1_CURL_MAX = float(os.environ.get("FB_J1_MAX", "0.595"))   # the reference grasp's curl


class ForceBalanceStage:
    """One stage of `_close_replay`. Mixed into `CloseMixin`; all `self.*` belong to `Demo`."""

    def _close_force_balance(self, st):
        q, obj = st.q, st.obj
        f_tgt = float(os.environ.get("FB_TARGET", "1.0"))         # N per pad
        d_j1 = float(os.environ.get("FB_STEP", "0.0006"))         # rad per step, per finger
        n_max = int(float(os.environ.get("FB_T", "8.0")) / self.dt)
        ema_a = 0.15
        hi_mult = float(os.environ.get("FB_HI", "3.0"))            # retract above this x target
        use_eff = os.environ.get("FB_EFFORT", "1") == "1"          # effort signal, not impulse
        # Arrival-sync is off by default: it works, but visibly holds the nearer finger back, and
        # from the symmetric open a uniform close loads all three anyway.
        gap_sync = os.environ.get("FB_GAP_SYNC", "0") == "1"       # rate-scale by measured gap
        rate_floor = float(os.environ.get("FB_RATE_MIN", "0.10"))  # slowest a near finger may go
        gap_every = max(1, 8)
        gaps = {f: None for f in "abc"}
        pad_gaps = {f: None for f in "abc"}      # pad-only, for the latch gate (see below)
        gap_prev = {f: None for f in "abc"}
        spd = {f: 0.0 for f in "abc"}          # gap closed per rad of j1, per finger
        ttc = {f: None for f in "abc"}         # rad of j1 remaining until contact
        j1_moved = {f: 0.0 for f in "abc"}
        ttc_sync = os.environ.get("FB_TTC", "0") == "1"           # off: uniform close, as above
        lever = 0.090         # measured finger lever, m
        # `_pad_force_step` is bursty -- a contact either reports on a given step or does not -- so
        # the latch must fire on this filtered value, never on a single sample.
        ema = {f: 0.0 for f in "abc"}
        base_e = {f: 0.0 for f in "abc"}                           # free-space effort per finger
        n_base = 48            # steps to average it over
        ix = {}
        for f in "abc":
            ix[f] = tuple(self.idx.get(f"finger_{f}_joint_{j}_1") for j in (1, 2, 3))
        held = set()
        _imp_win, n_win = {f: 0.0 for f in "abc"}, 0   # contact impulse over this window
        # FB_TRUEFORCE: a steadily pressing pad reports impulse on only ~1 step in 4, so averaging
        # the window over ALL steps reads the duty cycle, not the force. Average over hit steps.
        _hit_win = {f: 0 for f in "abc"}
        # Opt-in: if enabled, hi_mult and the proportional rate need re-scaling with it.
        _trueforce = os.environ.get("FB_TRUEFORCE", "0") == "1"
        def _zst(tag):
            # Height and speed at a boundary: a floor overlap is solved out as velocity, silently.
            _o = np.asarray(obj.get_world_poses()[0][0], float)
            _v = float(np.linalg.norm(np.asarray(obj.get_velocities(), float).reshape(-1)[:3]))
            print(f'>>>   zst[{tag}]: obj z {_o[2]:+.4f} (rest {self.obj_half_h:.3f}) '
                  f'speed {_v * 1000:.1f}mm/s', flush=True)
        _zst('fb-entry')
        _fl_prev = 0.0                      # floor impulse at the last support report
        _oc_prev, _pin_prev, _n_jump = None, None, 0   # jump catcher state
        _pinchprev_j = None
        _bprev_j = None
        _qprev_j = None
        _rev_watch = True                   # report the FIRST step that reverts the finger drive
        # Optional downward preload: brace the object, since floor friction scales with normal load.
        _pre = 0.0
        if _pre > 0.0:
            _h1 = self.idx.get("ColumnLeftBearingJoint_1")
            _h2 = self.idx.get("ColumnRightBearingJoint_1")
            if _h1 is not None and _h2 is not None:
                _oz0 = float(np.asarray(obj.get_world_poses()[0][0], float)[2])
                q[_h1] = float(np.clip(q[_h1] - _pre, 0.0, COLUMN_MAX))
                q[_h2] = float(np.clip(q[_h2] - _pre, 0.0, COLUMN_MAX))
                if _CLOSE_KIN:
                    self._force(q)
                self._apply(q)
                for _ in range(int(float(os.environ.get("PRELOAD_T", "0.8")) / self.dt)):
                    self._fN_step = {}
                    self.world.step(render=not HEADLESS)
                _oz1 = float(np.asarray(obj.get_world_poses()[0][0], float)[2])
                _pf = self._pad_force_step("palm") if hasattr(self, "_pad_force_step") else 0.0
                print(f">>> force-close: PRELOAD {_pre * 1000:.0f}mm of column drop -- object z "
                      f"{_oz0:.3f} -> {_oz1:.3f} ({(_oz1 - _oz0) * 1000:+.1f}mm), palm {_pf:.1f}N. "
                      f"Floor friction now resists the retreat that makes the fingers graze.",
                      flush=True)
        # Optional levelling to equalise finger travel. Off: a finger sweeping the object launches it.
        if os.environ.get("FB_LEVEL", "0") == "1":
            _lv = min(float(q[ix[f][0]]) for f in "abc" if ix[f][0] is not None)
            _q0_lv = {f: float(q[ix[f][0]]) for f in "abc" if ix[f][0] is not None}
            # Interpolate rather than commanding the level pose in one step.
            _n_lv = max(1, int(0.5 / self.dt))
            _q_lv = q.copy()
            for _i_lv in range(_n_lv):
                _f_lv = 0.5 - 0.5 * math.cos(math.pi * (_i_lv + 1) / _n_lv)   # smoothstep
                for _c in "abc":
                    _ii = ix[_c][0]
                    if _ii is not None:
                        q[_ii] = float(_q0_lv[_c] + _f_lv * (_lv - _q0_lv[_c]))
                self._finger_cmd = q[self._f_idx].copy()
                if _CLOSE_KIN:
                    self._force(q)
                self._apply(q)
                self._fN_step = {}
                self.world.step(render=not HEADLESS)
            _ocL = np.asarray(obj.get_world_poses()[0][0], float)
            _RL = self._obj_R(obj)
            _vL = np.asarray(obj.get_velocities(), float).reshape(-1)[:3]
            print(f">>> force-close: levelled j1 to {_lv:+.3f} on all three (symmetric start) "
                  f"-> obj z {_ocL[2]:.3f} axis.z {float(_RL[:, 2][2]) if _RL is not None else float('nan'):+.4f} "
                  f"speed {float(np.linalg.norm(_vL)) * 1000:.0f}mm/s"
                  f"{'   <-- LEVELLING DISTURBED IT' if (_RL is not None and float(_RL[:, 2][2]) < 0.99) else ''}",
                  flush=True)
        oc0 = np.asarray(obj.get_world_poses()[0][0], float)
        # Re-sync the arm: the jog stages drive it outside `st.q`, so a stale pose teleports back.
        try:
            self._close_sync_arm(st, "force-close entry")
        except Exception as _e_sa:
            print(f">>> force-close: arm re-sync failed ({_e_sa})", flush=True)
        # Hold the chassis, or the wheels roll through the close and carry the object at 1:1.
        _wheel_ii = [i for n, i in self.idx.items()
                     if ("wheel" in n and "rolling" in n) or "slipping" in n]
        _wheel_hold = None
        _bx_hold, _by_hold, _byaw_hold = self.base_ledger()
        # Constant-height target: pinch at the object centre, so the curl cannot press it into the floor.
        _i_h1z = self.idx.get("ColumnLeftBearingJoint_1"); _i_h2z = self.idx.get("ColumnRightBearingJoint_1")
        _z_hold_fb = float(np.asarray(obj.get_world_poses()[0][0], float)[2])
        _zk_fb = 0.5; _zc_fb = 0.003
        if os.environ.get("HOLD_WHEELS", "1") == "1" and _wheel_ii:
            _q_live = np.asarray(self.robot.get_joint_positions(), float)
            _wheel_hold = (_wheel_ii, _q_live[_wheel_ii].copy())
            print(f">>> force-close: holding {len(_wheel_ii)} chassis joints + pinning the root "
                  f"(wheels + slipping) for the whole close", flush=True)
        # Standing error = the hand still travelling to a pending target; this names the joint.
        if os.environ.get("FB_CMD_AUDIT", "1") == "1":
            try:
                _ctrl = self.robot.get_articulation_controller()
                _act_a = _ctrl.get_applied_action()
                _tgt = np.asarray(_act_a.joint_positions, float).reshape(-1)
                _now = np.asarray(self.robot.get_joint_positions(), float).reshape(-1)
                _rows = []
                for _jn in ("ArmLeftJoint_1", "ColumnLeftBearingJoint_1", "ColumnRightBearingJoint_1",
                            "BaseJoint_1", "RotationLeftJoint_1", "ArmRightJoint_1",
                            "HandBearingJoint_1"):
                    _i = self.idx.get(_jn)
                    if _i is None or _i >= len(_tgt) or _tgt[_i] is None:
                        continue
                    _e = float(_tgt[_i]) - float(_now[_i])
                    if abs(_e) > 1e-4:
                        _rows.append(f"{_jn.replace('Joint_1','').replace('Bearing','')}: "
                                     f"tgt {float(_tgt[_i]):+.4f} act {float(_now[_i]):+.4f} "
                                     f"err {_e:+.4f}")
                print(">>> force-close: ARM COMMAND AUDIT -> "
                      + (" | ".join(_rows) if _rows else "no standing error"), flush=True)
            except Exception as _e_ca:
                print(f">>> force-close: command audit failed ({_e_ca})", flush=True)
        # Settle the hand: handed over still moving, it drags the object and looks like a finger sweep.
        if os.environ.get("FB_SETTLE", "1") == "1":
            _v_tol = 0.0005     # m per step
            _n_set_max = int(1.5 / self.dt)
            _p_last = np.asarray(self._pinch(), float)
            _sv, _k_s = 1e9, 0
            for _k_s in range(_n_set_max):
                if _CLOSE_KIN:
                    self._force(q)
                self._apply(q)
                self._fN_step = {}
                self.world.step(render=not HEADLESS)
                _fb_diag(self, "hand-settle", _k_s, obj)
                _p_now = np.asarray(self._pinch(), float)
                _sv = float(np.linalg.norm(_p_now - _p_last))
                _p_last = _p_now
                if _sv < _v_tol:
                    break
            print(f">>> force-close: hand settled in {_k_s + 1} steps "
                  f"({_sv * 1000:.3f}mm/step, tol {_v_tol * 1000:.2f})"
                  f"{'   <-- STILL MOVING' if _sv >= _v_tol else ''}", flush=True)
        _zst('before-obj-settle')
        # Then wait for the OBJECT: with the robot stationary it can still be coasting.
        if os.environ.get("FB_OBJ_SETTLE", "1") == "1":
            _v_tol_o = 0.004     # m/s
            _n_o = int(1.5 / self.dt)
            _vo, _k_o = 1e9, 0
            for _k_o in range(_n_o):
                _vv = np.asarray(obj.get_velocities(), float).reshape(-1)[:3]
                _vo = float(np.linalg.norm(_vv))
                if _vo < _v_tol_o:
                    break
                if _CLOSE_KIN:
                    self._force(q)
                self._apply(q)
                self._fN_step = {}
                self.world.step(render=not HEADLESS)
                _fb_diag(self, "obj-settle", _k_o, obj)
            print(f">>> force-close: object settled in {_k_o + 1} steps -> {_vo * 1000:.1f}mm/s "
                  f"(tol {_v_tol_o * 1000:.0f}mm/s)"
                  f"{'   <-- STILL MOVING' if _vo >= _v_tol_o else ''}", flush=True)
        _zst('before-palm-gate')
        # Palm clearance gate: the palm ejects the object, and the pad gap queries do not see it.
        if os.environ.get("PALM_GATE", "1") == "1":
            try:
                _pmin = 0.010
                _oc_p = np.asarray(obj.get_world_poses()[0][0], float)
                _pg = self._finger_surface_gap("palm", _oc_p, KNOWN["object_radius"],
                                               self.obj_half_h, self._obj_R(obj))
                print(f">>> force-close: palm clearance {_pg * 1000:+.1f}mm "
                      f"(target >= {_pmin * 1000:.0f}mm)", flush=True)
                _n_back = 0
                while _pg is not None and _pg < _pmin and _n_back < 40:
                    _bx, _by, _byaw = self.base_ledger()
                    _v = np.array([_bx, _by]) - _oc_p[:2]
                    _v = _v / max(1e-6, float(np.linalg.norm(_v)))
                    _st = 0.004
                    if _CLOSE_KIN:
                        self.set_base(float(_bx + _st * _v[0]), float(_by + _st * _v[1]), _byaw)
                    if _CLOSE_KIN:
                        self._force(q)
                    self._apply(q)
                    self._fN_step = {}
                    self.world.step(render=not HEADLESS)
                    _fb_diag(self, "palm-gate", _n_back, obj, every=8)
                    _oc_p = np.asarray(obj.get_world_poses()[0][0], float)
                    _pg = self._finger_surface_gap("palm", _oc_p, KNOWN["object_radius"],
                                                   self.obj_half_h, self._obj_R(obj))
                    _n_back += 1
                if _n_back:
                    print(f">>> force-close: backed the base out {_n_back * float(os.environ.get('PALM_BACK_STEP', '0.004')) * 1000:.0f}mm "
                          f"-> palm clearance {_pg * 1000:+.1f}mm before any finger moves", flush=True)
            except Exception as _e_pg:
                print(f">>> force-close: palm gate failed ({_e_pg})", flush=True)
        # Precondition: baseline drift alone reads as a few newtons, so force with the pads far is noise.
        try:
            _g0 = {f: self._wrap_gap(f, oc0, KNOWN["object_radius"], self.obj_half_h,
                                     self._obj_R(obj)) for f in "abc"}
            _gmin0 = min(v for v in _g0.values() if v is not None)
            _lim0 = 0.050
            print(f">>> force-close: entry gaps "
                  f"{ {f: (None if v is None else round(v * 1000, 1)) for f, v in _g0.items()} }mm",
                  flush=True)
            # `_wrap_gap` is a minimum over link_1/2/3: the nearest KNUCKLE, which leads the pad.
            try:
                _pad0 = {f: self._finger_surface_gap(f, oc0, KNOWN["object_radius"],
                                                     self.obj_half_h, self._obj_R(obj))
                         for f in "abc"}
                print(f">>> force-close: entry PAD gaps "
                      f"{ {f: (None if v is None else round(v * 1000, 1)) for f, v in _pad0.items()} }mm"
                      f"   (wrap-gap above is the nearest KNUCKLE, not the pad)", flush=True)
            except Exception as _e_pg:
                print(f">>> force-close: pad-gap readout unavailable ({_e_pg})", flush=True)
            if _gmin0 > _lim0:
                print(f">>> force-close: *** NOT IN CONTACT RANGE *** closest pad is "
                      f"{_gmin0 * 1000:.0f}mm from the surface (> {_lim0 * 1000:.0f}mm). Any force "
                      f"this stage reports is BASELINE DRIFT, not a grasp. Fix the approach first.",
                      flush=True)
        except Exception as _e_pc:
            print(f">>> force-close: entry-gap check failed ({_e_pc})", flush=True)
        # First-touch forensics. Reset both ledgers: `_ctcN` accumulates across the approach.
        self._ctc = {}
        self._ctcN = {}
        _first_seen, _fs_k, _fs_oc = None, None, None
        print(f">>> force-close: each finger advances until its own pad reads {f_tgt:.2f}N "
              f"(step {d_j1:.4f} rad, cap {n_max} steps)", flush=True)
        for k in range(n_max):
            # Servo on measured joint effort, not contact impulse: impulse/dt jumps between zero
            # and hundreds of newtons between samples, so a narrow target band is never sampled.
            if k % gap_every == 0:
                try:
                    _ocp = np.asarray(obj.get_world_poses()[0][0], float)
                    _oRp = self._obj_R(obj)
                    for _fp in "abc":
                        pad_gaps[_fp] = self._finger_surface_gap(
                            _fp, _ocp, KNOWN["object_radius"], self.obj_half_h, _oRp)
                except Exception:
                    pass
            if gap_sync and k % gap_every == 0:
                try:
                    _oc_g = np.asarray(obj.get_world_poses()[0][0], float)
                    _oR_g = self._obj_R(obj)
                    # "pad" syncs the gripping surfaces, "wrap" the nearest knuckle; the pads trail
                    # the knuckles by a per-finger amount, but pad-sync measured worse.
                    _sync_on = os.environ.get("FB_SYNC_ON", "wrap")
                    for f in "abc":
                        gaps[f] = (self._finger_surface_gap(f, _oc_g, KNOWN["object_radius"],
                                                            self.obj_half_h, _oR_g)
                                   if _sync_on == "pad" else
                                   self._wrap_gap(f, _oc_g, KNOWN["object_radius"],
                                                  self.obj_half_h, _oR_g))
                    # per-finger closing SPEED (m of gap per rad of j1), from its own last step
                    for f in "abc":
                        g_now, g_old = gaps.get(f), gap_prev.get(f)
                        dj = j1_moved.get(f, 0.0)
                        if g_now is not None and g_old is not None and abs(dj) > 1e-9:
                            _sp = (g_old - g_now) / dj          # +ve = closing
                            if _sp > 1e-6:
                                spd[f] = (1.0 - 0.4) * spd[f] + 0.4 * _sp if spd[f] else _sp
                        if g_now is not None and spd.get(f):
                            # radians of j1 still to travel
                            ttc[f] = max(0.0, g_now) / spd[f]
                    gap_prev = dict(gaps)
                    j1_moved = {f: 0.0 for f in "abc"}
                except Exception:
                    gaps = {f: None for f in "abc"}
            _eff = self._finger_efforts() if use_eff else {}
            # Effort carries the finger's own gravity load, several newtons at this lever, which
            # reads as permanent contact: subtract a free-space baseline taken before anything touches.
            if use_eff and k < n_base:
                for f in "abc":
                    base_e[f] = (base_e[f] * k + abs(float(_eff.get(f, 0.0)))) / (k + 1)
            for f in "abc":
                if use_eff:
                    _s = max(0.0, abs(float(_eff.get(f, 0.0))) - base_e[f]) / lever
                else:
                    # Windowed contact impulse: unlike effort, it exists only on real contact.
                    _den = (max(1, _hit_win.get(f, 0)) if _trueforce else n_win) * self.dt
                    _s = float(_imp_win.get(f, 0.0)) / max(1e-9, _den)
                ema[f] = (1.0 - ema_a) * ema[f] + ema_a * _s
            if not use_eff:
                # Reset the window immediately after reading it, or the accumulator leaks.
                _imp_win = {f2: 0.0 for f2 in "abc"}
                _hit_win = {f2: 0 for f2 in "abc"}
                n_win = 0
            # Jump catcher: the object moving further than any measured impulse accounts for means
            # it is RELOCATED by a position-controlled body, which no force instrument here sees.
            if os.environ.get("FB_JUMP", "1") == "1":
                # The RIGID hand frame, not `_pinch()`: the fingertip midpoint moves as fingers close.
                _oc_j = np.asarray(obj.get_world_poses()[0][0], float)
                _hp_j, _ = self._hand_frame()
                _pin_j = np.asarray(_hp_j, float)
                if _oc_prev is not None:
                    _dj = float(np.linalg.norm(_oc_j - _oc_prev)) * 1000.0
                    if _dj > 2.0 and _n_jump < 6:
                        _dh = float(np.linalg.norm(_pin_j - _pin_prev)) * 1000.0
                        _oR_j = self._obj_R(obj)
                        _gj = {f: self._wrap_gap(f, _oc_j, KNOWN["object_radius"],
                                                 self.obj_half_h, _oR_j) for f in "abc"}
                        _pj = self._finger_surface_gap("palm", _oc_j, KNOWN["object_radius"],
                                                       self.obj_half_h, _oR_j)
                        _pinch_now = np.asarray(self._pinch(), float)
                        _dp = (float(np.linalg.norm(_pinch_now - _pinchprev_j)) * 1000.0
                               if _pinchprev_j is not None else float("nan"))
                        # World-space motion with no joint motion can only be the chassis.
                        _bx_j, _by_j, _byaw_j = self.base_ledger()
                        _db = (float(np.hypot(_bx_j - _bprev_j[0], _by_j - _bprev_j[1])) * 1000.0
                               if _bprev_j is not None else float("nan"))
                        _bprev_j = (_bx_j, _by_j)
                        # Which joints moved this step, so an unexplained wrist jump names its mover.
                        _qn_j = np.asarray(self.robot.get_joint_positions(), float)
                        if _qprev_j is not None:
                            _dq = np.abs(_qn_j - _qprev_j)
                            _nm = {v: k2 for k2, v in self.idx.items()}
                            _top_j = sorted(range(len(_dq)), key=lambda i: -_dq[i])[:5]
                            _bits_j = [f"{_nm.get(i, i)}: {_dq[i]:+.5f}" for i in _top_j
                                       if _dq[i] > 1e-6 and "finger" not in str(_nm.get(i, ""))]
                            if _bits_j:
                                print(f">>>     movers[{k:5d}]: " + " | ".join(_bits_j), flush=True)
                        _qprev_j = _qn_j
                        print(f">>>   JUMP[{k:5d}]: object {_dj:6.2f}mm in ONE step | BASE "
                              f"{_db:5.2f}mm | ARM(wrist) "
                              f"{_dh:5.2f}mm | pinch {_dp:5.2f}mm | gaps "
                              f"{ {f: (None if v is None else round(v * 1000, 1)) for f, v in _gj.items()} }mm "
                              f"palm {_pj * 1000:+.1f}mm", flush=True)
                        _n_jump += 1
                _oc_prev, _pin_prev, _pinchprev_j = _oc_j, _pin_j, np.asarray(self._pinch(), float)
            moved = False
            for f in "abc":
                i1, i2, i3 = ix[f]
                if i1 is None:
                    continue
                # Retract an overloaded finger: a servo that can only advance or hold cannot undo a
                # finger the seat left buried, which then carries the whole reaction.
                if ema[f] > f_tgt * hi_mult:
                    # Give way proportionally and slower than the advance, or the servo is
                    # bang-bang and the command visibly oscillates.
                    _over = (ema[f] - f_tgt * hi_mult) / max(1e-9, f_tgt * hi_mult)
                    _rt = float(np.clip(_over, 0.0, 1.0)) * 0.35
                    _dr = d_j1 * _rt
                    held.discard(f)
                    q[i1] = max(-1.04, q[i1] - _dr)
                    if i2 is not None:
                        q[i2] = float(np.clip(q[i2] - _dr * J2_PER_J1, *J2_RANGE))
                    if i3 is not None:
                        q[i3] = float(np.clip(q[i3] - _dr * J3_PER_J1, *J3_RANGE))
                    moved = True
                    continue
                # A finger counts as loaded only if the geometry agrees: the effort signal carries
                # gravity load, which can latch every finger wide open on iteration one.
                _gate_g = pad_gaps.get(f) if os.environ.get("FB_LATCH_ON", "pad") == "pad" \
                    else gaps.get(f)
                _g_ok = (_gate_g is None
                         or _gate_g <= 0.012)
                if ema[f] >= f_tgt and _g_ok:
                    # in band: stop pushing, hold the command
                    held.add(f)
                    continue
                # force fell away -> resume closing
                held.discard(f)
                if q[i1] >= J1_CURL_MAX:
                    # at full curl; j2/j3 are already at their targets
                    continue
                # Optional arrival sync: gap-proportional scales each advance by its remaining gap;
                # time-to-contact also divides by the measured closing speed, which the thumb needs.
                _r_f = 1.0
                if gap_sync and gaps.get(f) is not None:
                    if ttc_sync and all(ttc.get(x) is not None for x in "abc"):
                        _tmx = max(ttc[x] for x in "abc")
                        if _tmx and _tmx > 1e-9:
                            _r_f = float(np.clip(ttc[f] / _tmx, rate_floor, 1.0))
                    else:
                        _gmx = max((v for v in gaps.values() if v is not None), default=None)
                        if _gmx and _gmx > 1e-6:
                            _r_f = float(np.clip(gaps[f] / _gmx, rate_floor, 1.0))
                # Proportional approach: at full rate the step crossing the target overshoots it and
                # feeds the retract above into a limit cycle. Floored, or a near finger never lands.
                _fe = float(np.clip((f_tgt - ema[f]) / max(1e-9, f_tgt), 0.0, 1.0))
                _r_p = max(0.15, _fe)
                _d = d_j1 * _r_f * _r_p
                j1_moved[f] = j1_moved.get(f, 0.0) + _d
                q[i1] = min(J1_CURL_MAX, q[i1] + _d)
                if i2 is not None:
                    q[i2] = float(np.clip(q[i2] + _d * J2_PER_J1, *J2_RANGE))
                if i3 is not None:
                    q[i3] = float(np.clip(q[i3] + _d * J3_PER_J1, *J3_RANGE))
                moved = True
            if _wheel_hold is not None:
                # Pin the root, not just the wheel joints: the chassis is a floating root body.
                _wi, _wv = _wheel_hold
                q[_wi] = _wv
                self.robot.set_joint_velocities(np.zeros(len(_wi)), joint_indices=np.asarray(_wi))
                if _CLOSE_KIN:
                    self.set_base(_bx_hold, _by_hold, _byaw_hold)
            # Constant-height hold, so the curl squeezes horizontally rather than into the floor.
            if os.environ.get("FB_ZHOLD", "1") == "1" and _i_h1z is not None:
                _zerr = _z_hold_fb - float(self._pinch()[2])
                _dz = float(np.clip(_zk_fb * _zerr, -_zc_fb, _zc_fb))
                q[_i_h1z] = float(np.clip(q[_i_h1z] + _dz, 0.0, COLUMN_MAX))
                q[_i_h2z] = float(np.clip(q[_i_h2z] + _dz, 0.0, COLUMN_MAX))
            self._finger_cmd = q[self._f_idx].copy()
            if _CLOSE_KIN:
                self._force(q)
            self._apply(q)
            # Read the applied action right after `_apply`: separates a write that never reached the
            # articulation from something later re-opening the hand.
            if k % 480 == 0 and os.environ.get("FB_APPLY_AUDIT", "1") == "1":
                try:
                    _aa2 = self.robot.get_articulation_controller().get_applied_action()
                    _tg2 = np.asarray(_aa2.joint_positions, float).reshape(-1)
                    _i1a = ix["a"][0]
                    print(f">>>     apply-audit[{k:5d}]: wrote q[a.j1] {float(q[_i1a]):+.3f} -> "
                          f"drive_tgt {float(_tg2[_i1a]):+.3f} | _finger_cmd holds "
                          f"{float(self._finger_cmd[list(self._f_idx).index(_i1a)]):+.3f}",
                          flush=True)
                    # The distal joints too: j2/j3 can sit on their limits for a whole close while
                    # `q` climbs, with no contact holding them.
                    _fl = list(self._f_idx)
                    for _fj in "abc":
                        _i2a, _i3a = ix[_fj][1], ix[_fj][2]
                        if _i2a is None or _i3a is None:
                            continue
                        print(f">>>     apply-audit[{k:5d}]: {_fj}.j2 q{float(q[_i2a]):+.3f} -> "
                              f"tgt{float(_tg2[_i2a]):+.3f} cmd"
                              f"{float(self._finger_cmd[_fl.index(_i2a)]):+.3f}"
                              f" | {_fj}.j3 q{float(q[_i3a]):+.3f} -> "
                              f"tgt{float(_tg2[_i3a]):+.3f} cmd"
                              f"{float(self._finger_cmd[_fl.index(_i3a)]):+.3f}", flush=True)
                except Exception as _e_aa2:
                    print(f">>>     apply-audit[{k:5d}]: unavailable ({_e_aa2})", flush=True)
            # `_fN_step` accumulates from the contact callback; clear it every step.
            self._fN_step = {}
            self.world.step(render=not HEADLESS)
            # Watch object and fingers mid-close; everything else prints at a stage boundary.
            if os.environ.get("CLOSE_WATCH", "1") == "1":
                try:
                    _ow = np.asarray(obj.get_world_poses()[0][0], float)
                    if not hasattr(self, "_cw_z0"):
                        # RESTING height, not "wherever it
                        self._cw_z0 = float(self.obj_half_h)
                        # started" -- a sink is measured
                        self._cw_xy0 = _ow[:2].copy()
                        # against the floor it sits on
                        self._cw_said = False
                    _Rw = self._obj_R(obj)
                    _azw = float(_Rw[:, 2][2]) if _Rw is not None else float("nan")
                    _vw = float(np.linalg.norm(
                        np.asarray(obj.get_velocities(), float).reshape(-1)[:3])) * 1000.0
                    _dz = float(_ow[2]) - self._cw_z0
                    _dxy = float(np.linalg.norm(_ow[:2] - self._cw_xy0)) * 1000.0
                    _qn = np.asarray(self.robot.get_joint_positions(), float)
                    _fl = list(self._f_idx)
                    _jw = {}
                    for _f in "abc":
                        _i1 = ix[_f][0]
                        if _i1 is None:
                            continue
                        _c = (float(self._finger_cmd[_fl.index(_i1)])
                              if (self._finger_cmd is not None and _i1 in _fl) else float("nan"))
                        _jw[_f] = f"{_c:+.2f}/{float(_qn[_i1]):+.2f}"
                    _flag = ("  <-- SINKING" if _dz < -0.005 else
                             "  <-- TILTING" if _azw < 0.99 else
                             "  <-- SHOVED" if _dxy > 10.0 else "")
                    if (not self._cw_said) and (abs(_dz) > 0.005 or _azw < 0.99 or _dxy > 10.0):
                        _hot = {kk: round(vv, 3) for kk, vv in
                                getattr(self, "_fN_step", {}).items() if vv > 1e-6}
                        _why = _hot or "NONE -- nothing is pushing it, so whatever moved it is KINEMATIC"
                        print(f">>> close-watch: OBJECT DISTURBED at step {k}: dz {_dz * 1000:+.0f}mm "
                              f"(z {float(_ow[2]):.3f} vs rest {self._cw_z0:.3f})  dxy {_dxy:.0f}mm  "
                              f"axis.z {_azw:+.4f}  speed {_vw:.0f}mm/s{_flag}\n"
                              f">>>   impulse THIS step (Ns per link): {_why}\n"
                              f">>>   j1 cmd/act {_jw}", flush=True)
                        self._cw_said = True
                    if k % 25 == 0:
                        import time as _t
                        _prev = getattr(self, "_cw_t", None)
                        _now = _t.time()
                        self._cw_t, self._cw_k = _now, k
                        _rate = ("" if _prev is None else
                                 f" | {(k - getattr(self, '_cw_kprev', 0)) / max(1e-6, _now - _prev):.1f} steps/s")
                        self._cw_kprev = k
                        # Step rate is a symptom: a contact explosion looks like a hang from outside.
                        try:
                            _hy = self._hand_frame()[1][:, 1]
                            _tilt = math.degrees(math.acos(
                                max(-1.0, min(1.0, float(-_hy[2]) / max(1e-9, float(np.linalg.norm(_hy)))))))
                            _wr = f" | wrist {_tilt:.1f}deg off level"
                        except Exception:
                            _wr = ""
                        print(f">>> close-watch[{k:5d}]: obj dz {_dz * 1000:+.0f}mm dxy {_dxy:.0f}mm "
                              f"axis.z {_azw:+.4f} speed {_vw:.0f}mm/s | contacts "
                              f"{len(getattr(self, '_fN_step', {}))} | j1 cmd/act {_jw}{_flag}"
                              f"{_wr}{_rate}", flush=True)
                except Exception as _e_cw:
                    # never fail SILENTLY: a probe that
                    if not getattr(self, "_cw_broke", False):
                        # prints nothing reads as "clean"
                        self._cw_broke = True
                        print(f">>> close-watch: PROBE BROKEN ({_e_cw}) -- object NOT being "
                              f"monitored this run", flush=True)
            # Read the drive again AFTER the step: a disagreement with the audit above brackets the
            # revert to this `world.step`.
            if _rev_watch:
                try:
                    _tgR = np.asarray(self.robot.get_articulation_controller()
                                      .get_applied_action().joint_positions, float).reshape(-1)
                    _i1r = ix["a"][0]
                    if abs(float(_tgR[_i1r]) - float(q[_i1r])) > 0.01:
                        print(f">>>   DRIVE REVERTED across world.step[{k:5d}]: wrote "
                              f"{float(q[_i1r]):+.3f} -> reads {float(_tgR[_i1r]):+.3f} "
                              f"(pin {'ON' if getattr(self, '_pin_orig_step', None) else 'off'})",
                              flush=True)
                        _rev_watch = False
                except Exception:
                    _rev_watch = False
            # accumulate this window's contact impulse
            for _fw in "abc":
                _iw = float(self._fN_step.get(_fw, 0.0))
                _imp_win[_fw] = _imp_win.get(_fw, 0.0) + _iw
                if _iw > 0.0:
                    _hit_win[_fw] = _hit_win.get(_fw, 0) + 1
            n_win += 1
            if len(held) == 3:
                print(f">>> force-close: all three loaded at step {k} "
                      f"{ {f: round(ema[f], 2) for f in 'abc'} }N", flush=True)
                break
            if not moved:
                print(f">>> force-close: no finger can advance at step {k} "
                      f"{ {f: round(ema[f], 2) for f in 'abc'} }N (legal travel exhausted)",
                      flush=True)
                break
            # first contact, and how far the object travels in the steps that follow it
            if _first_seen is None and self._ctc:
                _first_seen = dict(self._ctc)
                _fs_k, _fs_oc = k, np.asarray(obj.get_world_poses()[0][0], float)
                _co = getattr(self, "_ctc_obj", {})
                print(f">>>   FIRST TOUCH at step {k}: "
                      f"{ {kk: _co.get(kk, '?') for kk in sorted(_first_seen)} } "
                      f"| impulse { {kk: round(getattr(self, '_ctcN', {}).get(kk, 0.0), 4) for kk in sorted(_first_seen)} }Ns "
                      f"| focus={getattr(self, '_focus_obj', None)} "
                      f"| gaps { {f: (None if gaps[f] is None else round(gaps[f]*1000,1)) for f in 'abc'} }mm "
                      f"| j1 { {f: round(float(q[ix[f][0]]), 3) for f in 'abc'} }", flush=True)
            elif _first_seen is not None and _fs_k is not None and (k - _fs_k) in (24, 60, 120, 240):
                _d = np.linalg.norm(np.asarray(obj.get_world_poses()[0][0], float) - _fs_oc) * 1000
                # Impulse, not just the link list: a link appears on speculative zero-impulse contacts.
                _cn = getattr(self, "_ctcN", {})
                _top = sorted(_cn.items(), key=lambda kv: -kv[1])[:4]
                print(f">>>   +{k - _fs_k:4d} steps after first touch: object has moved "
                      f"{_d:6.1f}mm  | impulse-so-far "
                      f"{ {kk: round(vv, 3) for kk, vv in _top} }Ns", flush=True)
            # Palm clearance and centreline geometry throughout: the entry gate cannot catch a palm
            # that advances later.
            if k % 240 == 0:
                try:
                    _ocw = np.asarray(obj.get_world_poses()[0][0], float)
                    _pgw = self._finger_surface_gap("palm", _ocw, KNOWN["object_radius"],
                                                    self.obj_half_h, self._obj_R(obj))
                    _ddw = self._du_dv(obj)
                    _inb = (_ddw is not None and -0.13 <= _ddw[0] <= -0.07 and abs(_ddw[1]) <= 0.03)
                    # Vertical support: an under-supported object slides under small lateral force.
                    _mass = float(os.environ.get("OBJ_DENSITY", "800.0")) * math.pi \
                        * KNOWN["object_radius"] ** 2 * (2.0 * self.obj_half_h)
                    _fl_now = float(getattr(self, "_ctcN", {}).get("floor", 0.0))
                    _fl_d = _fl_now - _fl_prev
                    _exp = _mass * 9.81 * (240 * self.dt)
                    _fl_prev = _fl_now
                    print(f">>>     support[{k:5d}]: floor {_fl_d:6.3f}Ns vs weight {_exp:6.3f}Ns "
                          f"({100.0 * _fl_d / max(1e-9, _exp):5.0f}%)"
                          f"{'   <-- UNDER-SUPPORTED' if _fl_d < 0.7 * _exp else ''}", flush=True)
                    print(f">>>     geom[{k:5d}]: palm {_pgw * 1000:+6.1f}mm | "
                          f"du {(_ddw[0] * 1000 if _ddw else float('nan')):+7.1f} "
                          f"dv {(_ddw[1] * 1000 if _ddw else float('nan')):+6.1f}mm "
                          f"{'IN band' if _inb else 'OUT of band'} | "
                          f"obj z {_ocw[2]:.3f}", flush=True)
                except Exception as _e_gw:
                    print(f">>>     geom[{k:5d}]: failed ({_e_gw})", flush=True)
            if k % 480 == 0:
                print(f">>>   force-close[{k:5d}]: F { {f: round(ema[f], 2) for f in 'abc'} }N "
                      f"held={sorted(held)} j1 "
                      f"{ {f: round(float(q[ix[f][0]]), 2) for f in 'abc'} }", flush=True)
                # Do the distal joints actually curl? A finger that rotates at j1 but never wraps
                # ends with the pads far out.
                _anyc = getattr(self, "_ctc_any", {})
                if _anyc:
                    _top = sorted(_anyc.items(), key=lambda kv: -kv[1])[:6]
                    print(f">>>     finger-contact[{k:5d}]: "
                          + " | ".join(f"{kk} x{vv}" for kk, vv in _top), flush=True)
                _flc = getattr(self, "_ctc_floor", {})
                if _flc:
                    print(f">>>     floor-contact[{k:5d}]: "
                          f"{ {kk: vv for kk, vv in sorted(_flc.items())} }", flush=True)
                # Reset both ledgers per sample: unreset, a jam that ENDED reads like one ongoing.
                self._ctc_any = {}
                self._ctc_floor = {}
                try:
                    _qa_t = np.asarray(self.robot.get_joint_positions(), float)
                    _bits_t = []
                    for _f_t in "abc":
                        _i1t, _i2t, _i3t = ix[_f_t]
                        _bits_t.append(
                            f"{_f_t}: j2 cmd{float(q[_i2t]):+.2f}/act{float(_qa_t[_i2t]):+.2f} "
                            f"j3 cmd{float(q[_i3t]):+.2f}/act{float(_qa_t[_i3t]):+.2f}"
                            if _i2t is not None and _i3t is not None else f"{_f_t}: -")
                    print(f">>>     curl[{k:5d}]: " + " | ".join(_bits_t), flush=True)
                    # Why a joint does not move: eff ~ 0 with a large position error is a dead drive; eff
                    # at the cap with vel ~ 0 is mechanically held; vel without motion is another writer.
                    _vv = np.asarray(self.robot.get_joint_velocities(), float)
                    # The target PhysX holds, not `get_applied_action`'s echo of what we just sent.
                    _pt = None
                    for _acc in ("get_dof_position_targets", "get_joint_position_targets"):
                        _v = getattr(getattr(self.robot, "_articulation_view", None), _acc, None)
                        if _v is not None:
                            try:
                                _pt = np.asarray(_v(), float).reshape(-1)
                                break
                            except Exception:
                                pass
                    try:
                        _ef = np.asarray(self.robot.get_measured_joint_efforts(), float)
                    except Exception:
                        _ef = np.full(len(_vv), float("nan"))
                    try:
                        _kp_l, _kd_l = self.ctrl.get_gains()[0], self.ctrl.get_gains()[1]
                    except Exception:
                        _kp_l = _kd_l = np.full(len(_vv), float("nan"))
                    for _f_t in "abc":
                        _i1t, _i2t, _i3t = ix[_f_t]
                        print(f">>>     why[{k:5d}] {_f_t}: "
                              + " ".join(
                                  f"j{_n}(vel{float(_vv[_i]):+.3f} eff{float(_ef[_i]):+.2f} "
                                  f"kp{float(_kp_l[_i]):.0f} kd{float(_kd_l[_i]):.0f} "
                                  f"pxtgt{(float(_pt[_i]) if _pt is not None else float('nan')):+.3f})"
                                  for _n, _i in ((1, _i1t), (2, _i2t), (3, _i3t))), flush=True)
                except Exception as _e_t:
                    print(f">>>     curl[{k:5d}]: unavailable ({_e_t})", flush=True)
                # Is the arrival-time sync running? gap mm, spd mm/rad, ttc rad remaining.
                print(f">>>     ttc: gap { {f: (None if gaps[f] is None else round(gaps[f]*1000,1)) for f in 'abc'} }mm"
                      f"  spd { {f: round(spd[f]*1000, 1) for f in 'abc'} }mm/rad"
                      f"  ttc { {f: (None if ttc[f] is None else round(ttc[f], 3)) for f in 'abc'} }rad"
                      f"  -> {'TTC ACTIVE' if (ttc_sync and all(ttc.get(x) is not None for x in 'abc')) else 'falling back to gap-proportional'}",
                      flush=True)
        oc1 = np.asarray(obj.get_world_poses()[0][0], float)
        # Hold by overdrive, not at contact: stopping on the measurement leaves zero drive error,
        # and a position drive at zero error makes no torque, so the grip is given away.
        _od = float(os.environ.get("FB_OVERDRIVE", "0.06"))
        # Complete the wrap: the loop latches at first pad force, and that pinch cannot lift.
        _wrapN = int(1.2 / self.dt) if _od > 0.0 and held else 0
        if _wrapN:
            # WRAP_CURL scales the distal curl. Cylinders want PARALLEL pads -- a full curl makes each pad a
            # tip touching at one height, and that height spread tilts the held object.
            _wc = float(os.environ.get("WRAP_CURL", "1.0"))
            _wrap_j2 = 0.55 * _wc   # total distal curl to add
            _wrap_j3 = 0.45 * _wc
            _fld = sorted(held)
            _q0w = np.asarray(self.robot.get_joint_positions(), float).copy()
            _stop = {f: False for f in _fld}                          # per-finger distal-contact stop
            for _kw in range(1, _wrapN + 1):
                _fw = float(_kw) / _wrapN
                for f in _fld:
                    i1, i2, i3 = ix[f]
                    if i1 is None or _stop[f]:
                        continue
                    # stop this finger's wrap once its distal links carry real load
                    try:
                        _pff = float(self._pad_force_step(f))
                    except Exception:
                        _pff = 0.0
                    if _pff > f_tgt * 2.5:
                        _stop[f] = True
                        continue
                    if i2 is not None:
                        q[i2] = float(np.clip(_q0w[i2] + _fw * _wrap_j2, *J2_RANGE))
                    if i3 is not None:
                        q[i3] = float(np.clip(_q0w[i3] - _fw * _wrap_j3, *J3_RANGE))
                    # small knuckle follow
                    q[i1] = float(min(J1_LEGAL_MAX, _q0w[i1] + _fw * _od))
                self._finger_cmd = q[self._f_idx].copy()
                if _CLOSE_KIN:
                    self._force(q)
                self._apply(q)
                self._fN_step = {}
                self.world.step(render=not HEADLESS)
            print(f">>> force-close: WRAP-COMPLETION distal curl j2 +{_wrap_j2:.2f} j3 -{_wrap_j3:.2f} "
                  f"on {_fld}, stopped {[f for f in _fld if _stop[f]]} on distal load", flush=True)
        # later stages read this: a 3-finger hold with overdrive must NOT be re-seated to cmd==act
        st.fb_held = set(held)
        # the pick state is a different object: the gate reads this
        self._fb_held = set(held)
        # Close-end census: every contact pair since the last print, non-object partners first,
        # and where the object ended up in the mouth frame.
        try:
            _ca = getattr(self, "_ctc_any", None) or {}
            print(f">>> force-close: contact pairs {dict(sorted(_ca.items(), key=lambda kv: ('pickup_obj' in kv[0], -kv[1]))[:8])}",
                  flush=True)
            self._ctc_any = {}
            _dd_e = self._du_dv(obj)
            if _dd_e is not None:
                print(f">>> force-close: object vs pinch at close end du {_dd_e[0] * 1000:+.1f} dv {_dd_e[1] * 1000:+.1f} mm",
                      flush=True)
        except Exception as _e_c:
            print(f">>> force-close: census unavailable ({_e_c})", flush=True)
        print(f">>> force-close done: loaded {sorted(held)} "
              f"F { {f: round(ema[f], 2) for f in 'abc'} }N  object moved "
              f"{np.linalg.norm(oc1 - oc0) * 1000:.0f}mm  axis "
              f"{np.round(self._obj_R(obj)[:, 2], 3).tolist() if self._obj_R(obj) is not None else '?'}",
              flush=True)
