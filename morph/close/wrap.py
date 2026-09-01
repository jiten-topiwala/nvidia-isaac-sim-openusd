"""The encompassing (wrap) close — the grasp this hand is actually built for.

Part of the close sequence — see `morph/close/__init__.py` for the stage order and the
shared state object. Imports Isaac APIs at module level; only importable after SimulationApp.
"""
import math
import os

import numpy as np

from morph.config import PERFECT, HEADLESS, KNOWN


class WrapStage:
    """One stage of `_close_replay`. Mixed into `CloseMixin`; all `self.*` belong to `Demo`."""

    def _close_wrap(self, st):
        q, obj = st.q, st.obj
        bx, by, byaw = st.bx, st.by, st.byaw
        lj_locked = st.lj_locked
        if os.environ.get("WRAP", "1") == "1" and PERFECT:
            _r_w = KNOWN["object_radius"]
            # The synchroniser measures the nearest LINK by default, not the pad. Synchronising on
            # the pad gap was tried and is worse here -- the thumb ends in the air and the total
            # finger impulse falls.
            _gfn_w = (self._finger_surface_gap
                      if os.environ.get("WRAP_GAP_METRIC", "links") == "pads" else self._wrap_gap)
            _ctrl = float(np.clip(0.20 - 4.0 * _r_w, 0.15, 0.20))
            _i_w = float(np.clip(_ctrl / 0.20, 0.0, 1.0))
            # Curl to the vendor's legal limit and let the closed loop decide where to stop.
            _i_w = 1.0
            _tg = {1: _i_w * 0.70 * 0.85, 2: _i_w * 0.90 * 0.95, 3: -0.052 - _i_w * 1.10}
            # Keep the reference model's deep j3 curl. The vendor's own close pose is shallower, but
            # shortening to match was tried and is worse: only two fingers land and the object
            # topples.
            _j1t = os.environ.get("WRAP_J1_TARGET", "")
            if _j1t:
                _tg[1] = float(_j1t)
            _tg_f = {f_: dict(_tg) for f_ in "abc"}
            _ax_extra = 0.60   # rad, thumb only
            # `c` needs its own offset: it sits about 13mm further from the object than `b` at every
            # j1, a structural asymmetry of this hand — b and c share a command but never a
            # distance.
            _cx_extra = 0.60
            # `b` gets one too, so that no finger is the hard-coded exception. At the base target
            # the closed finger envelope is wider than the objects being grasped, so a finger that
            # stops there closes on air by construction.
            _bx_extra = 0.60
            for _f_x, _ex in (("a", _ax_extra), ("b", _bx_extra), ("c", _cx_extra)):
                for _jn in (1, 2):
                    _tg_f[_f_x][_jn] = _tg_f[_f_x][_jn] + _ex
                _tg_f[_f_x][3] = _tg_f[_f_x][3] - _ex
            print(f">>> WRAP per-finger extra curl: a +{_ax_extra:.2f}  b +{_bx_extra:.2f}  "
                  f"c +{_cx_extra:.2f} rad -> j1 targets "
                  f"a {_tg_f['a'][1]:+.3f}  b {_tg_f['b'][1]:+.3f}  c {_tg_f['c'][1]:+.3f} "
                  f"(vendor legal max +1.222)", flush=True)
            # Symmetric start. A symmetric open puts all three fingertips at the same radius from
            # the palm, so with the palm at the object standoff they are equidistant and can close
            # together.
            if os.environ.get("WRAP_LEVEL", "1") == "1":
                _j1s = {f_: self.idx[f"finger_{f_}_joint_1_1"] for f_ in "abc"
                        if f"finger_{f_}_joint_1_1" in self.idx}
                _v0s = {f_: float(q[_i]) for f_, _i in _j1s.items()}
                _lv = min(_v0s.values())                      # j1 negative = open
                _n_lv = max(2, int(1.0 / self.dt))
                for _k_lv in range(_n_lv):
                    _fr_lv = 0.5 - 0.5 * math.cos(math.pi * (_k_lv + 1) / _n_lv)
                    for f_, _i in _j1s.items():
                        q[_i] = _v0s[f_] + _fr_lv * (_lv - _v0s[f_])
                    self._finger_cmd = q[self._f_idx].copy()
                    self.set_base(bx, by, byaw)
                    if os.environ.get("CLOSE_HOLD_ARM", "1") == "1":
                        self._force(q)
                    self._apply(q)
                    self.world.step(render=not HEADLESS)
                print(f">>> WRAP level: j1 "
                      f"{ {k_: round(v_, 2) for k_, v_ in _v0s.items()} } -> {_lv:+.2f} on all "
                      f"three (symmetric start)", flush=True)

            # Equalise the starting DISTANCES, not the starting angles: levelling j1 starts the
            # fingers at the same angle but preserves whatever spread exists in their distance,
            # because the arcs are not congruent, so the nearest finger still arrives first and
            # takes the whole load.
            if os.environ.get("WRAP_PREGAP", "0") == "1":
                try:
                    _pg_n = max(2, int(1.5 / self.dt))
                    _pg_cap = 0.35   # rad of retract
                    _pg_tol = 0.002
                    _pg_rate = _pg_cap / _pg_n
                    _pg_used = {f_: 0.0 for f_ in "abc"}
                    _pg_g0 = None
                    for _k_pg in range(_pg_n):
                        _oc_pg = np.asarray(obj.get_world_poses()[0][0], float)
                        _oR_pg = self._obj_R(obj)
                        _g_pg = {f_: _gfn_w(f_, _oc_pg, _r_w, self.obj_half_h, _oR_pg)
                                 for f_ in "abc"}
                        if _pg_g0 is None:
                            _pg_g0 = dict(_g_pg)
                        _tgt_pg = max(_g_pg.values())
                        _moved = False
                        for f_ in "abc":
                            _ix_pg = self.idx.get(f"finger_{f_}_joint_1_1")
                            if _ix_pg is None:
                                continue
                            if _g_pg[f_] < _tgt_pg - _pg_tol and _pg_used[f_] < _pg_cap:
                                q[_ix_pg] = float(q[_ix_pg]) - _pg_rate   # -j1 = open
                                _pg_used[f_] += _pg_rate
                                _moved = True
                        self._finger_cmd = q[self._f_idx].copy()
                        self.set_base(bx, by, byaw)
                        if os.environ.get("CLOSE_HOLD_ARM", "1") == "1":
                            self._force(q)
                        self._apply(q)
                        self.world.step(render=not HEADLESS)
                        if not _moved:
                            break
                    _oc_pg = np.asarray(obj.get_world_poses()[0][0], float)
                    _g_pg = {f_: round(_gfn_w(f_, _oc_pg, _r_w, self.obj_half_h,
                                              self._obj_R(obj)) * 1000, 1) for f_ in "abc"}
                    print(f">>> WRAP pre-gap: "
                          f"{ {k_: round(v_ * 1000, 1) for k_, v_ in (_pg_g0 or {}).items()} } -> "
                          f"{_g_pg}mm  (retracted "
                          f"{ {k_: round(v_, 2) for k_, v_ in _pg_used.items()} } rad)", flush=True)
                except Exception as _e_pg:
                    print(f">>> WRAP pre-gap: failed ({_e_pg})", flush=True)
            _n_w = max(2, int(6.0 / self.dt))
            _q0w = {}
            for _f in "abc":
                for _jn in (1, 2, 3):
                    _ix = self.idx.get(f"finger_{_f}_joint_{_jn}_1")
                    if _ix is not None:
                        _q0w[(_f, _jn)] = float(q[_ix])
            # Bound the grip force with the effort cap, not with tracking error. Steady force is
            # stiffness x overdrive capped by max_effort, so a cap below the torque the finger needs
            # at its lever just lets the finger back off however far past contact it is commanded --
            # it has to clear the fingers' own static load or a closing command lets the hand fall
            # open.
            try:
                _fmax = 8.0       # N per pad, target
                _lev = 0.09      # m, measured
                _eff = np.full(len(self.names), 1.0e6)
                _cap = _fmax * _lev + 1.2  # + static load
                for _i_n, _n_n in enumerate(self.names):
                    if _n_n.startswith(("finger_", "palm_finger")):
                        _eff[_i_n] = _cap
                self.ctrl.set_max_efforts(_eff)
                print(f">>> WRAP effort cap: {_cap:.2f} Nm per finger joint "
                      f"(= {_fmax:.1f}N x {_lev:.3f}m lever + static-load bias)", flush=True)
            except Exception as _e_ef:
                print(f">>> WRAP effort cap: not set ({_e_ef})", flush=True)
            print(f">>> WRAP close: encompassing grasp, intensity {_i_w:.2f} -> "
                  f"j1 {_tg[1]:+.3f}  j2 {_tg[2]:+.3f}  j3 {_tg[3]:+.3f}  "
                  f"(radius {_r_w * 1000:.0f}mm)", flush=True)
            # Approach gentle, squeeze stiff. This stage runs after `_grip_stiffen()`, so without
            # softening the drives first contact arrives hard -- more than enough to tip a cylinder
            # whose limit is m*g*r/h.
            if os.environ.get("WRAP_GENTLE", "1") == "1":
                self._grip_gentle()
            _wf = set()
            self._wrap_n = {c_: 0 for c_ in "abc"}
            # Reset per invocation: these are per-stroke minima, and a retry that inherits the
            # previous pick's values compares this stroke's gaps against a stale reference.
            self._wrap_best = {c_: (1e9, None) for c_ in "abc"}
            self._wrap_ema = {c_: 0.0 for c_ in "abc"}
            self._wrap_prox = set()
            self._wrap_land = {}

            # Landing distance as data. Two attempts to PREDICT where each finger comes to rest both
            # failed -- the rate extrapolation diverged and threw the object.
            def _wgap(f_):
                _oc_l = np.asarray(obj.get_world_poses()[0][0], float)
                return self._wrap_gap(f_, _oc_l, _r_w, self.obj_half_h, self._obj_R(obj)) * 1000.0

            # Gap-synchronised curl. Without it the curl is a clock: every finger advances on the
            # same smoothstep to a fixed target with no feedback, and three fingers on one clock,
            # starting from different angles and distances, arrive one at a time. Levelling j1 above
            # makes the START symmetric; it cannot make the ARRIVAL symmetric, because the arcs
            # differ. So drive each finger's phase at a rate proportional to its own measured gap:
            #   * the finger furthest from the surface runs at full rate
            #   * the closest is throttled and waits for the others
            #   * the whole curl slows as the minimum gap shrinks -- approach fast, touch slow
            # Sampled every WRAP_SYNC_EVERY steps; the gap query is the expensive part. Note which
            # gap: this measures the knuckles, while the grasp is made by the distal pads -- see
            # WRAP_GAP_METRIC above.
            _sync_g = os.environ.get("WRAP_GAP_SYNC", "1") == "1"
            _sy_far = 0.020
            _sy_near = 0.003
            _sy_floor = 0.15
            _sy_every = max(1, 8)
            _ph = {f_: 0.0 for f_ in "abc"}          # per-finger phase in [0, 1]
            _gcache = {f_: 1e9 for f_ in "abc"}
            # The synchroniser has to extend the time, not truncate the curl: it scales each phase
            # rate below 1 to make the closest finger wait, so against a fixed step budget the
            # stroke never finishes and the fingers stop part-way -- which looks like "they cannot
            # reach" when the kinematics say they can.
            _n_cap = int(_n_w * 4.0)
            for _k in range(_n_cap):
                if _sync_g and min(_ph.values()) >= 1.0:
                    break                                    # the whole stroke is done
                if _sync_g:
                    if _k % _sy_every == 0:
                        _oc_s = np.asarray(obj.get_world_poses()[0][0], float)
                        _oR_s = self._obj_R(obj)
                        for f_ in "abc":
                            _gcache[f_] = _gfn_w(f_, _oc_s, _r_w, self.obj_half_h, _oR_s)
                        # Re-arm a latched finger whose gap has reopened. The latch drops a finger
                        # out of the live set permanently, which makes the close open-loop at
                        # exactly the moment it must not be: if the object then moves, the finger is
                        # abandoned in mid-air and there is no opposing pair left.
                        if os.environ.get("WRAP_RELATCH", "1") == "1" and _wf:
                            _rg = 0.015
                            for f_ in sorted(_wf):
                                if _gcache[f_] > _rg and _ph[f_] < 1.0:
                                    _wf.discard(f_)
                                    _wn[f_] = 0                  # clear the hold counter too
                                    print(f">>>   WRAP: finger {f_} re-armed, gap reopened to "
                                          f"{_gcache[f_] * 1000:+.1f}mm", flush=True)
                    _live = {f_: max(0.0, _gcache[f_]) for f_ in "abc" if f_ not in _wf}
                    _gmin = min(_live.values()) if _live else 0.0
                    _gmax = max(_live.values()) if _live else 1.0
                    # global: full rate far out, WRAP_SLOW_MIN of it at the surface
                    _s_g = (1.0 if _gmin >= _sy_far else
                            _sy_floor if _gmin <= _sy_near else
                            _sy_floor + (1.0 - _sy_floor) * (_gmin - _sy_near)
                            / max(1e-9, _sy_far - _sy_near))
                    for f_ in "abc":
                        # per-finger: nearest finger throttled to WRAP_SLOW_MIN, furthest at 1.0
                        _r_f = (1.0 if _gmax <= 1e-6 else
                                max(_sy_floor, float(_live.get(f_, _gmax)) / _gmax))
                        _ph[f_] = min(1.0, _ph[f_] + _s_g * _r_f / max(1, _n_w - 1))
                else:
                    for f_ in "abc":
                        _ph[f_] = (_k + 1) / max(1, _n_w - 1)
                _frs = {f_: 0.5 - 0.5 * math.cos(math.pi * min(1.0, _ph[f_])) for f_ in "abc"}
                for _f in "abc":
                    if _f in _wf:
                        continue
                    # Stop THIS finger when its own pads load -- a wrap cages by geometry, so a
                    # finger that has met the object should hold, not keep crushing.
                    _wn = getattr(self, "_wrap_n", None)
                    if _wn is None:
                        _wn = self._wrap_n = {c_: 0 for c_ in "abc"}
                    # Filter the force before the latch. `_pad_force_step` is bursty, so a raw
                    # consecutive-step counter never fills even while the finger accumulates real
                    # load -- the latch never fires and every finger runs to full curl with no hold.
                    _wa = 0.15
                    _we = getattr(self, "_wrap_ema", None)
                    if _we is None:
                        _we = self._wrap_ema = {c_: 0.0 for c_ in "abc"}
                    # Do not let the latch fire while the finger is still open: an early graze
                    # otherwise freezes j1 where it is for the rest of the wrap.
                    _ix_g = self.idx.get(f"finger_{_f}_joint_1_1")
                    _min_j1 = 0.20
                    _min_pr = 0.15
                    if _min_pr > 0 and _ix_g is not None:
                        _min_j1 = min(_min_j1, _q0w[(_f, 1)] + _min_pr)
                    if _ix_g is not None and float(q[_ix_g]) < _min_j1:
                        _we[_f] = 0.0
                        _wn[_f] = 0
                        for _jn in (1, 2, 3):
                            _ixc = self.idx.get(f"finger_{_f}_joint_{_jn}_1")
                            if _ixc is not None:
                                q[_ixc] = _q0w[(_f, _jn)] + _frs[_f] * (_tg_f[_f][_jn]
                                                                       - _q0w[(_f, _jn)])
                        continue
                    _we[_f] = (1.0 - _wa) * _we[_f] + _wa * self._pad_force_step(_f)
                    _wn[_f] = _wn[_f] + 1 if _we[_f] >= 0.5 else 0
                    # A geometric stop alongside the force stop. Even with the synchronised curl, a
                    # force-triggered latch fires hundreds of steps apart, because force onset
                    # depends on which pad bites first -- and by then the first finger has been
                    # pressing alone long enough to tip the cylinder.
                    _wb = getattr(self, "_wrap_best", None)
                    if _wb is None:
                        _wb = self._wrap_best = {c_: (1e9, None) for c_ in "abc"}
                    _g_now = _gcache.get(_f, 1e9) if _sync_g else 1e9
                    _ix_b = self.idx.get(f"finger_{_f}_joint_1_1")
                    if _g_now < _wb[_f][0]:
                        _wb[_f] = (_g_now, None if _ix_b is None else float(q[_ix_b]))
                    elif (os.environ.get("WRAP_MIN_STOP", "1") == "1" and _wb[_f][1] is not None
                          and _g_now > _wb[_f][0] + 0.002
                          and _ph[_f] > 0.15):
                        q[_ix_b] = _wb[_f][1]                # rewind to the closest-approach pose
                        _wn[_f] = 3   # and latch there
                        print(f">>>   WRAP: finger {_f} closest-approach stop at "
                              f"{_wb[_f][0] * 1000:+.1f}mm (now {_g_now * 1000:+.1f}mm)", flush=True)
                    _gstop = 0.003
                    _hit_g = _sync_g and _gstop > 0 and _gcache.get(_f, 1e9) <= _gstop
                    if _hit_g and _wn[_f] < 3:
                        _wn[_f] = 3
                    if _wn[_f] >= 3:
                        _wf.add(_f)
                        _qa_w = np.asarray(self.robot.get_joint_positions(), float)
                        # Hold by overdrive, not by freezing. `command := actual` gives zero steady-
                        # state force, because a position drive makes torque only from error, so the
                        # grip decays away between the close and the lift.
                        if os.environ.get("WRAP_SOFT_LATCH", "1") == "1":
                            try:
                                _kp_s = np.asarray(self.ctrl.get_gains()[0], float).copy()
                                _kd_s = np.asarray(self.ctrl.get_gains()[1], float).copy()
                                _soft = 8.0
                                for _jn_s in (1, 2, 3):
                                    _ix_s = self.idx.get(f"finger_{_f}_joint_{_jn_s}_1")
                                    if _ix_s is not None:
                                        _kp_s[_ix_s] = _soft
                                self.ctrl.set_gains(_kp_s, _kd_s)
                                print(f">>>   WRAP: finger {_f} soft-latched (kp -> {_soft:.0f}) "
                                      f"so it holds contact without pushing", flush=True)
                            except Exception as _e_sl:
                                print(f">>>   WRAP: soft latch unavailable ({_e_sl})", flush=True)
                        _od = 0.10    # rad past contact
                        # Contact-synchronised close. Overdriving the instant one finger lands is
                        # the shove: the nearest finger presses a free-standing body out of the
                        # mouth before anything can oppose it, and raising friction makes that
                        # worse, because a one-sided contact just gets more tangential grip.
                        if os.environ.get("WRAP_SYNC", "1") == "1":
                            _od = 0.0                            # freeze here; squeeze comes later
                        # Proximal stops, distal keeps curling -- the adaptive-linkage behaviour.
                        _ix1 = self.idx.get(f"finger_{_f}_joint_1_1")
                        if _ix1 is not None:
                            q[_ix1] = float(_qa_w[_ix1]) + _od
                        if os.environ.get("WRAP_DISTAL_CONT", "0") != "1":
                            for _jn in (2, 3):
                                _ix = self.idx.get(f"finger_{_f}_joint_{_jn}_1")
                                if _ix is not None:
                                    q[_ix] = float(_qa_w[_ix]) + (-_od if _jn == 3 else _od)
                        else:
                            _wf.discard(_f)                  # not retired: j2/j3 keep curling
                            _wd = getattr(self, "_wrap_prox", None)
                            if _wd is None:
                                _wd = self._wrap_prox = set()
                            if _f not in _wd:
                                _wd.add(_f)
                                _gl = _wgap(_f)
                                self._wrap_land[_f] = _gl
                                print(f">>>   WRAP: finger {_f} proximal contact -> j1 held "
                                      f"({_od:.2f} rad overdrive), j2/j3 keep curling around  "
                                      f"[landed at wrap-gap {_gl:+.1f}mm, j1 {float(q[_ix1]) if _ix1 is not None else float('nan'):+.3f}]",
                                      flush=True)
                            _wn[_f] = 0                       # re-arm for the DISTAL contact
                            continue
                        self._wrap_land[_f] = _wgap(_f)
                        print(f">>>   WRAP: finger {_f} loaded -> holding with {_od:.2f} rad "
                              f"overdrive (keeps the drive pushing)  "
                              f"[landed at wrap-gap {self._wrap_land[_f]:+.1f}mm]", flush=True)
                        continue
                    _held1 = _f in getattr(self, "_wrap_prox", set())
                    for _jn in (1, 2, 3):
                        if _jn == 1 and _held1:
                            continue                          # j1 stalled at proximal contact
                        _ix = self.idx.get(f"finger_{_f}_joint_{_jn}_1")
                        if _ix is not None:
                            q[_ix] = (_q0w[(_f, _jn)]
                                      + _frs[_f] * (_tg_f[_f][_jn] - _q0w[(_f, _jn)]))
                # Per-step trace of the three quantities the latch depends on: the filtered force,
                # the consecutive-step counter, j1 against its arming threshold, and the measured
                # gap.
                if os.environ.get("WRAP_TRACE", "0") == "1" and _k % 40 == 0:
                    _tr = []
                    for _f_t in "abc":
                        _ix_t = self.idx.get(f"finger_{_f_t}_joint_1_1")
                        _j1_t = float(q[_ix_t]) if _ix_t is not None else float("nan")
                        _arm = min(0.20,
                                   _q0w[(_f_t, 1)] + 0.15
                                   ) if _ix_t is not None else float("nan")
                        _tr.append(f"{_f_t}: F {self._pad_force_step(_f_t):5.2f} ema "
                                   f"{_we.get(_f_t, 0.0):5.2f} n {_wn.get(_f_t, 0):2d} "
                                   f"j1 {_j1_t:+.2f}/{_arm:+.2f} gap {_wgap(_f_t):+6.1f}")
                    # ...and the object's own pose, so a sink or a topple during the stroke is
                    # attributable to a step rather than only visible at the end.
                    _oc_tr = np.asarray(obj.get_world_poses()[0][0], float)
                    print(f">>>   WRAP[{_k:4d}] obj z {_oc_tr[2]:.3f} axis "
                          f"{np.round(self._obj_R(obj)[:, 2], 2).tolist()} | " + " | ".join(_tr),
                          flush=True)
                self._finger_cmd = q[self._f_idx].copy()
                self.set_base(bx, by, byaw)
                if os.environ.get("CLOSE_HOLD_ARM", "1") == "1":
                    self._force(q)
                self._apply(q)
                self._fN_step = {}
                self.world.step(render=not HEADLESS)
                if len(_wf) == 3:
                    break
            # Arm seat: the last few millimetres cannot come from the fingers. The gaps bottom out
            # short on all three, because the links swing nearly tangentially there -- the pads are
            # at their arc minimum, so curling harder moves them AWAY.
            if os.environ.get("WRAP_ARM_SEAT", "1") == "1" and _wf and _sync_g:
                try:
                    _as_f = 1.0       # N, mean pad target
                    _as_max = 0.020  # m, travel ceiling
                    _as_step = 0.0004
                    _as_n = max(2, int(2.0 / self.dt))
                    _as_e = {f_: 0.0 for f_ in "abc"}
                    _as_travel = 0.0
                    self.start_jog()
                    _as_z = float(self.grip_pos()[2])            # CONSTANT Z, held absolutely
                    for _ka in range(_as_n):
                        for f_ in "abc":
                            _as_e[f_] = (1.0 - _wa) * _as_e[f_] + _wa * self._pad_force_step(f_)
                        # Stop on the weakest finger, not the mean: a mean lets one finger's spike
                        # halt the advance for all three, leaving the others clear while that one
                        # presses the object over alone.
                        if (min(_as_e.values()) >= _as_f or _as_travel >= _as_max
                                or max(_as_e.values()) >= 3.0):
                            break
                        _oc_a = np.asarray(obj.get_world_poses()[0][0], float)
                        _gp_a = self.grip_pos()
                        _d_a = (_oc_a - _gp_a)[:2]
                        _n_a = float(np.linalg.norm(_d_a))
                        if _n_a < 1e-6:
                            break
                        _dir_a = _d_a / _n_a
                        self.jog["target"] = np.array(
                            [_gp_a[0] + _dir_a[0] * _as_step,
                             _gp_a[1] + _dir_a[1] * _as_step, _as_z])
                        self._jog_tick()
                        self._fN_step = {}
                        self.world.step(render=not HEADLESS)
                        _as_travel += _as_step
                    _oc_a2 = np.asarray(obj.get_world_poses()[0][0], float)
                    _oR_a2 = self._obj_R(obj)
                    _g_a2 = {f_: round(self._wrap_gap(f_, _oc_a2, _r_w, self.obj_half_h,
                                                      _oR_a2) * 1000, 1) for f_ in "abc"}
                    print(f">>> WRAP arm-seat: advanced {_as_travel * 1000:.1f}mm  "
                          f"pads { {k_: round(v_, 1) for k_, v_ in _as_e.items()} }N  "
                          f"gaps {_g_a2}mm  obj z {_oc_a2[2]:.3f} axis "
                          f"{np.round(_oR_a2[:, 2], 2).tolist()}", flush=True)
                    # the jog moved the arm outside `q` — same two-writers trap as _close_sync_arm
                    _qa_a = np.asarray(self.robot.get_joint_positions(), float)
                    q[self._arm_ii] = _qa_a[self._arm_ii]
                    bx, by, byaw = self.base_ledger()
                except Exception as _e_as:
                    print(f">>> WRAP arm-seat: failed ({_e_as})", flush=True)
            if os.environ.get("WRAP_SYNC", "1") == "1" and _wf:
                # Squeeze phase: every landed finger presses at the same time, so the lateral
                # components oppose rather than shove. Ramped, not stepped -- a step command on a
                # stiff drive is an impact.
                if os.environ.get("WRAP_GENTLE", "1") == "1":
                    self._grip_stiffen()
                _od_s = 0.10
                _n_s = max(2, int(1.5 / self.dt))
                _qa_s = np.asarray(self.robot.get_joint_positions(), float)
                _base_s = {}
                for _f_s in sorted(_wf):
                    for _jn in (1, 2, 3):
                        _ixs = self.idx.get(f"finger_{_f_s}_joint_{_jn}_1")
                        if _ixs is not None:
                            _base_s[(_f_s, _jn)] = float(_qa_s[_ixs])
                # Squeeze on all three joints. Squeezing j1 alone was tried, on the theory that
                # j2/j3 press the object down, and is much worse -- the end gaps open by an order of
                # magnitude and the object tips.
                print(f">>> WRAP squeeze: {sorted(_wf)} press together "
                      f"({_od_s:.2f} rad overdrive ceiling over {_n_s * self.dt:.1f}s)", flush=True)
                self._fN = {}                                # report the SQUEEZE force, not the approach
                # Force-regulated squeeze. A fixed overdrive ramp is open loop: the force it
                # produces depends on where each finger happened to stop, its pad compliance and the
                # effort cap, so the three end at very different loads -- and that imbalance is
                # itself what tips the cylinder.
                _sq_f = 3.0       # N per pad, target
                _sq_bal = 2.0   # N, max lead over the min
                _sq_srv = os.environ.get("WRAP_SQ_SERVO", "1") == "1"
                _od_f = {_f_s: 0.0 for _f_s in _wf}                     # per-finger overdrive now
                _ema_s = {_f_s: 0.0 for _f_s in _wf}
                _rate_s = _od_s / max(1, _n_s)                          # ceiling reached in WRAP_SQ_T
                for _ks in range(_n_s):
                    _fs_r = 0.5 - 0.5 * math.cos(math.pi * (_ks + 1) / _n_s)
                    if _sq_srv:
                        for _f_s in _wf:
                            _ema_s[_f_s] = (1.0 - _wa) * _ema_s[_f_s] + _wa * self._pad_force_step(_f_s)
                        _lo = min(_ema_s.values()) if _ema_s else 0.0
                        for _f_s in _wf:
                            # advance while under target AND not more than WRAP_SQ_BAL ahead of the
                            # weakest finger — a lone finger pressing harder is the tipping moment
                            if _ema_s[_f_s] < _sq_f and _ema_s[_f_s] <= _lo + _sq_bal:
                                _od_f[_f_s] = min(_od_s, _od_f[_f_s] + _rate_s)
                    for (_f_s, _jn), _b_s in _base_s.items():
                        _ixs = self.idx.get(f"finger_{_f_s}_joint_{_jn}_1")
                        if _ixs is not None:
                            _amt = _od_f[_f_s] if _sq_srv else _fs_r * _od_s
                            q[_ixs] = _b_s + (-_amt if _jn == 3 else _amt)
                    self._finger_cmd = q[self._f_idx].copy()
                    self.set_base(bx, by, byaw)
                    if os.environ.get("CLOSE_HOLD_ARM", "1") == "1":
                        self._force(q)
                    self._apply(q)
                    self._fN_step = {}
                    self.world.step(render=not HEADLESS)

            _ocw = np.asarray(obj.get_world_poses()[0][0], float)
            # Landing table. `stop` is the gap when the finger latched (`--AIR--` means it never
            # did); `end` is the gap at the end of the stroke.
            try:
                _lend = {f_: _wgap(f_) for f_ in "abc"}
                _lrow = "  ".join(
                    (f"{f_}: stop {self._wrap_land[f_]:+6.1f} end {_lend[f_]:+6.1f}mm"
                     if f_ in self._wrap_land else
                     f"{f_}: stop --AIR-- end {_lend[f_]:+6.1f}mm") for f_ in "abc")
                _dth = _lend["a"] - 0.5 * (_lend["b"] + _lend["c"])
                print(f">>> WRAP landing: {_lrow}", flush=True)
                print(f">>>   thumb-vs-bc landing error {_dth:+.1f}mm "
                      f"-> shift the hand {abs(_dth) / 2:.1f}mm "
                      f"{'toward b/c' if _dth > 0 else 'toward a'} to split it", flush=True)
            except Exception as _e_l:
                print(f">>> WRAP landing: unavailable ({_e_l})", flush=True)
            print(f">>> WRAP done: loaded {sorted(_wf)}  obj {np.round(_ocw, 3).tolist()}  "
                  f"axis {np.round(self._obj_R(obj)[:, 2], 3).tolist()}  "
                  f"pad loads {{{', '.join(f'{f}: {self._fN.get(f, 0.0):.1f}' for f in 'abc')}}}",
                  flush=True)
            lj_locked = True                     # wrap owns the close; skip every legacy stage
        st.lj_locked = lj_locked

