"""The wrap sub-stage of the geometric seat: curl j1/j2/j3 per finger onto the surface,
then the final common-mode approach.

Part of the close sequence — see `morph/close/__init__.py` for the stage order and the
shared state object. Imports Isaac APIs at module level; only importable after SimulationApp.
"""
import math
import os

import numpy as np

import scene
from morph.config import HEADLESS


class SeatWrapStage:
    """One stage of `_close_replay`. Mixed into `CloseMixin`; all `self.*` belong to `Demo`."""

    def _seat_wrap(self, st, j1i, r_obj, touch, sync_tol, grip_done):
        """Curl each finger onto the object after the geometric seat. Returns `grip_done`."""
        q, obj = st.q, st.obj
        bx, by, byaw = st.bx, st.by, st.byaw
        if os.environ.get("SEAT_WRAP", "1") == "1":
            # Sample gaps AND object pose either side of the settle below. A gap that jumps further
            # than the fingers could have travelled means the object moved, not the hand, and only
            # measuring both tells the two apart.
            _oc0 = np.asarray(obj.get_world_poses()[0][0], float)
            _aw0 = self._obj_R(obj)[:, 2]
            _g0 = {fl: round(self._finger_surface_gap(fl, _oc0, r_obj, self.obj_half_h,
                                                      self._obj_R(obj)) * 1000, 1)
                   for fl in j1i}
            # Zero the drive error on seated fingers first. A seated finger still holds a command
            # ahead of where it physically is, so its drive keeps pressing -- and since the thumb
            # reaches contact first, during the settle it presses an unopposed cylinder and shoves
            # it away from the other two.
            _qa_seat = np.asarray(self.robot.get_joint_positions(), float)
            for _fl, _i in j1i.items():
                q[_i] = float(_qa_seat[_i])
            self._finger_cmd = q[self._f_idx].copy()
            # Settle before sampling. The seat rewinds a receding finger's COMMAND, so without
            # letting the joint actually travel there the wrap reads the pre-rewind pose and
            # immediately declares the finger receding again.
            self._hold(q, int(0.3 / self.dt),
                       kin=True)
            _oc1 = np.asarray(obj.get_world_poses()[0][0], float)
            _aw1 = self._obj_R(obj)[:, 2]
            _g1 = {fl: round(self._finger_surface_gap(fl, _oc1, r_obj, self.obj_half_h,
                                                      self._obj_R(obj)) * 1000, 1)
                   for fl in j1i}
            # Print the axis, not just the displacement. A cylinder that topples drops its centre
            # from half its height to its radius, which looks exactly like sinking in a
            # displacement-only reading.
            print(f">>>   settle: gaps {_g0} -> {_g1}   obj moved "
                  f"{np.linalg.norm(_oc1 - _oc0) * 1000:.1f}mm "
                  f"{np.round(_oc1 - _oc0, 4).tolist()}  axis "
                  f"{np.round(_aw0, 3).tolist()} -> {np.round(_aw1, 3).tolist()}", flush=True)
            wrapj = {}
            for fl in j1i:
                i2 = self.idx.get(f"finger_{fl}_joint_2_1")
                if i2 is not None:
                    wrapj[fl] = (i2, self.idx.get(f"finger_{fl}_joint_3_1"))
            w_rate = 0.35      # rad/s on j2
            w_cap = 0.90
            w_j3 = 1.20          # j3 closes NEGATIVE
            # The wrap needs its own recede threshold: the seat's is a couple of millimetres, but
            # the object jitters by more than that while the pads settle, so the tighter test trips
            # on noise and the wrap quits after a few percent of its travel.
            w_back = 0.006
            w_cur = {fl: 0.0 for fl in wrapj}
            oc_w = np.asarray(obj.get_world_poses()[0][0], float)
            w_best = {fl: (self._finger_surface_gap(fl, oc_w, r_obj, self.obj_half_h,
                                                    self._obj_R(obj)), None)
                      for fl in wrapj}
            w_done = {fl for fl in wrapj if w_best[fl][0] <= touch}
            # Which joint reaches depends on the pose, so every finger starts on j1 and falls back
            # to j2, and a finger that recedes on one joint rewinds and switches rather than giving
            # up.
            w_mode = {fl: "j1" for fl in wrapj}
            w_next = {"j1": "j2", "j2": None}            # fallback order when a joint exhausts
            w_moved = {fl: False for fl in wrapj}        # did this finger advance last step?
            # The same direction probe the final pass runs, and it matters most here: the wrap is
            # the stage where the object first gets pushed, so a probe that only fires downstream
            # confirms the wrong direction after the damage is done.
            w_sign = {fl: 1.0 for fl in wrapj}
            w_probe = {fl: None for fl in wrapj}
            w_flip = set()
            wp_n = 12
            w_do_probe = os.environ.get("WRAP_PROBE", "1") == "1"
            # A receded finger must keep BLOCKING the opposing jaw while it retries.
            w_reached = set(w_done)
            # This has to cover the far jaw's whole arc shortfall AND the time spent holding the
            # near jaw still while it catches up.
            n_wrap = max(2, int(3.5 / self.dt))
            for kw in range(n_wrap):
                oc_w = np.asarray(obj.get_world_poses()[0][0], float)
                oR_w = self._obj_R(obj)
                qa_w = np.asarray(self.robot.get_joint_positions(), float)
                # Sync gate. This is the stage where all three fingers CAN reach, so it is where
                # arriving together is meaningful: advance only the finger(s) furthest from the
                # surface, so both jaws land within SYNC_TOL.
                g_w = {fl: self._finger_surface_gap(fl, oc_w, r_obj, self.obj_half_h, oR_w)
                       for fl in wrapj if fl not in w_reached}
                # Sync by jaw, not by finger. This is a two-jaw hand: b and c sit together on one
                # side and the thumb opposes them alone, so a per-finger "advance the furthest" rule
                # still lands one SIDE first -- the same single-sided tipping moment, mirrored.
                g_jaw = {}
                for jn, fs in (("a", ("a",)), ("bc", ("b", "c"))):
                    gj = [g_w[f] for f in fs if f in g_w]
                    if gj:
                        g_jaw[jn] = min(gj)
                w_lead = max(g_jaw.values()) if g_jaw else 0.0
                # Both jaws on the object means the grip is complete: stop. The grip needs ONE
                # finger from each jaw on the surface, not all three -- the third is a passenger,
                # and once the opposed pair is made it usually cannot improve in either direction,
                # so driving it just shoves the object.
                ga_ = 0.0 if "a" in w_reached else min(
                    [g_w[f] for f in ("a",) if f in g_w] or [1e9])
                gbc_ = 0.0 if (w_reached & {"b", "c"}) else min(
                    [g_w[f] for f in ("b", "c") if f in g_w] or [1e9])
                if max(ga_, gbc_) <= touch:
                    qa_j = np.asarray(self.robot.get_joint_positions(), float)
                    for _fl in wrapj:                    # command := actual, nothing pressing
                        for _jn2 in (1, 2, 3):
                            _ji2 = self.idx.get(f"finger_{_fl}_joint_{_jn2}_1")
                            if _ji2 is not None:
                                q[_ji2] = float(qa_j[_ji2])
                    self._finger_cmd = q[self._f_idx].copy()
                    print(f">>>   wrap: BOTH JAWS on the object (a {ga_ * 1000:+.1f}mm, "
                          f"bc {gbc_ * 1000:+.1f}mm) — grip complete, stopping", flush=True)
                    w_done.update(wrapj)
                    w_reached.update(wrapj)
                    grip_done = True
                    # Symmetric preload, and it is only safe here. The jaw-complete stop leaves the
                    # fingers NEAR the object but not on it, and a few millimetres of air carries no
                    # friction -- the cage is geometric only.
                    f_tgt = 3.0         # N per jaw
                    # N — BELOW the 5.4N one-sided tipping limit, so the ceiling itself keeps a
                    # single loaded jaw from tipping the cylinder
                    f_max = 5.0
                    pl_rate = 0.05   # rad/s, gentle
                    pl_cap = 0.12
                    n_pl = max(2, int(float(os.environ.get("PRELOAD_T", "2.0")) / self.dt))
                    lead_f = {"a": "a",
                              "bc": min(("b", "c"),
                                        key=lambda f: self._finger_surface_gap(
                                            f, oc_w, r_obj, self.obj_half_h, oR_w))}
                    # Seat the far jaw by moving the hand. The near jaw loads fine, but the far one
                    # can stop a millimetre or two out with no finger DOF able to take it up -- its
                    # j1 is at the arc minimum and its j2 opens the gap.
                    if os.environ.get("JAW_SEAT_NUDGE", "1") == "1":
                        try:
                            oc_n = np.asarray(obj.get_world_poses()[0][0], float)
                            oR_n = self._obj_R(obj)
                            # Centre the object; do not try to close the far jaw with the hand. A
                            # rigid translation is zero-sum -- both pads move the same way, so one
                            # approaches and the other retreats by the same amount. The governing
                            # invariant is
                            #     a_gap + bc_gap = jaw_span - object_diameter
                            # which hand motion cannot change; only finger closure shrinks the span.
                            # So the hand splits the slack evenly and the finger with real authority
                            # takes it up.
                            gg = {jn: self._finger_surface_gap(fl_, oc_n, r_obj,
                                                               self.obj_half_h, oR_n)
                                  for jn, fl_ in lead_f.items()}
                            far = max(gg.items(), key=lambda kv: kv[1])
                            near = min(gg.items(), key=lambda kv: kv[1])
                            far = (far[0], lead_f[far[0]])
                            nud = float(np.clip((gg[far[0]] - near[1]) * 0.5, 0.0,
                                                0.006))
                            if nud > 0.0005:
                                pa = self._finger_world_pts("a").mean(axis=0)
                                pb = self._finger_world_pts(lead_f["bc"]).mean(axis=0)
                                # Sign: to close the far jaw's gap the hand travels FROM the far jaw
                                # TOWARD the near one. The object sits between them, so moving
                                # toward the far jaw opens it.
                                v = (pa - pb)[:2] if far[0] == "bc" else (pb - pa)[:2]
                                v = v / max(1e-9, float(np.linalg.norm(v)))
                                n_n = int(0.4 / self.dt)
                                for i_n in range(n_n):
                                    f_n = 0.5 - 0.5 * math.cos(math.pi * (i_n + 1) / n_n)
                                    self.set_base(float(bx + f_n * nud * v[0]),
                                                  float(by + f_n * nud * v[1]), byaw)
                                    self._force(q)
                                    self._apply(q)
                                    self.world.step(render=not HEADLESS)
                                bx, by = float(bx + nud * v[0]), float(by + nud * v[1])
                                print(f">>>   preload: hand nudged {nud * 1000:.1f}mm to "
                                      f"CENTRE (gaps a {gg.get('a', 0) * 1000:+.1f} / bc "
                                      f"{gg.get('bc', 0) * 1000:+.1f}mm, slack "
                                      f"{sum(gg.values()) * 1000:+.1f}mm) along "
                                      f"{np.round(v, 3).tolist()}", flush=True)
                        except Exception as e:
                            print(f">>>   preload: jaw-seat nudge skipped ({e})", flush=True)
                    # MEASURE the moment arm; do not assume it.  tau = F*L, so a wrong L scales the
                    # commanded force by exactly that error.
                    lever = {jn: (self._finger_lever(fl_, oc_w, r_obj, self.obj_half_h, oR_w) or 0.10)
                             for jn, fl_ in lead_f.items()}
                    # Compliant gains and a current limit on the two lead j1 DOFs, this window only.
                    kp_all = kd_all = kp_sav = kd_sav = eff_sav = None
                    try:
                        _g = self.ctrl.get_gains()
                        kp_all = np.asarray(_g[0], float).copy()
                        kd_all = np.asarray(_g[1], float).copy()
                        kp_sav, kd_sav = kp_all.copy(), kd_all.copy()
                    except Exception:
                        kp_all = np.full(len(self.names), 2.0e3)
                        kd_all = np.full(len(self.names), 2.0e2)
                        for _i2, _n2 in enumerate(self.names):
                            if _n2.startswith(("finger_", "palm_finger")):
                                kp_all[_i2], kd_all[_i2], _ = scene.finger_pd(_n2)
                    try:
                        eff_sav = np.asarray(self.ctrl.get_max_efforts(), float).copy()
                    except Exception:
                        pass
                    eff_new = None if eff_sav is None else eff_sav.copy()
                    kp_pre = 15.0
                    kd_pre = 3.0
                    for jn, fl_ in lead_f.items():
                        ip_ = j1i.get(fl_)
                        if ip_ is None:
                            continue
                        kp_all[ip_], kd_all[ip_] = kp_pre, kd_pre
                        if eff_new is not None:          # hard ceiling: f_max at the MEASURED lever
                            eff_new[ip_] = f_max * lever[jn]
                    self.ctrl.set_gains(kp_all, kd_all)
                    if eff_new is not None:
                        try:
                            self.ctrl.set_max_efforts(eff_new)
                        except Exception:
                            pass
                    print(f">>>   preload: lever(m) "
                          f"{ {k: round(v, 4) for k, v in lever.items()} }  "
                          f"target {f_tgt:.1f}N  ceiling {f_max:.1f}N "
                          f"(effort {f_max * min(lever.values()):.3f}-"
                          f"{f_max * max(lever.values()):.3f}Nm)  kp {kp_pre} kd {kd_pre}",
                          flush=True)
                    pl_cur = {jn: 0.0 for jn in lead_f}
                    pl_done = set()
                    f_ema = {jn: 0.0 for jn in lead_f}   # PhysX per-step impulses are spiky;
                    f_a = 0.08   # filter before feeding
                    # a position loop, or it oscillates. First-touch impulse carries the collision
                    # velocity change rather than the steady contact force, so impulse/dt spikes
                    # enormously for a single step and a finger can read the whole effort ceiling
                    # and then settle to nothing.
                    f_hold = {jn: 0 for jn in lead_f}
                    f_need = 12   # consecutive steps
                    # ...and a MINIMUM travel past first contact. A position servo produces force
                    # only through deflection: holding F needs err = F*L/kp of commanded overshoot
                    # PAST the surface.
                    pl_min = 0.0    # 0.02 OVERRODE the force
                    # Arc guard. Without it both jaws drive their full travel having never touched
                    # anything: past the j1 arc minimum, more travel OPENS the gap, and force
                    # feedback cannot rescue that because there is no contact to feed back.
                    pl_best = {jn: self._finger_surface_gap(fl_, oc_w, r_obj, self.obj_half_h, oR_w)
                               for jn, fl_ in lead_f.items()}
                    pl_back = 0.001
                    for kp_ in range(n_pl):
                        self._fN_step = {}               # THIS step's contact impulses only
                        for jn, fl_ in lead_f.items():
                            if jn in pl_done:
                                continue
                            ip_ = j1i.get(fl_)
                            if ip_ is None or pl_cur[jn] >= pl_cap:
                                pl_done.add(jn)
                                continue
                            gp_ = self._finger_surface_gap(
                                fl_, np.asarray(obj.get_world_poses()[0][0], float),
                                r_obj, self.obj_half_h, self._obj_R(obj))
                            if gp_ < pl_best[jn]:
                                pl_best[jn] = gp_
                            elif gp_ > pl_best[jn] + pl_back and f_ema[jn] <= 0.0:
                                pl_done.add(jn)      # receding with no contact -> wrong way
                                print(f">>>   preload: jaw {jn} ({fl_}) receding "
                                      f"({pl_best[jn] * 1000:+.1f} -> {gp_ * 1000:+.1f}mm) "
                                      f"with 0N — stopping", flush=True)
                                continue
                            # Neither jaw may press alone. Advancing each jaw to its OWN force
                            # target means whichever contacts first loads while the other sits at
                            # zero and its arc guard stops it -- and the loaded jaw then presses an
                            # unopposed cylinder and moves it, from a cage that was millimetres on
                            # all three and upright.
                            if pl_done and jn not in pl_done:
                                pl_done.add(jn)
                                print(f">>>   preload: jaw {jn} ({fl_}) stopping — the opposing "
                                      f"jaw is already done, a lone jaw only shoves the object",
                                      flush=True)
                                continue
                            f_now = f_ema[jn]
                            if f_now < f_tgt or pl_cur[jn] < pl_min:
                                pl_cur[jn] += pl_rate * self.dt   # advance until SUSTAINED
                                q[ip_] = float(q[ip_]) + pl_rate * self.dt
                            elif f_now > f_max:          # over ceiling -> ease back
                                q[ip_] = float(q[ip_]) - pl_rate * self.dt
                        self._finger_cmd = q[self._f_idx].copy()
                        self.set_base(bx, by, byaw)
                        if os.environ.get("CLOSE_HOLD_ARM", "1") == "1":
                            self._force(q)
                        self._apply(q)
                        self.world.step(render=not HEADLESS)
                        _eff_m = self._finger_efforts()
                        _qa_m = np.asarray(self.robot.get_joint_positions(), float)
                        for jn, fl_ in lead_f.items():   # read AFTER the step
                            f_ema[jn] = ((1 - f_a) * f_ema[jn]
                                         + f_a * self._pad_force_step(fl_))
                            # Where is the force coming from? The effort ceiling is F_max *
                            # measured_lever, so a higher reading means either the joint torque is
                            # at its cap and the excess is reacted by the kinematically forced arm,
                            # or the cap is not applied at all.
                            if (os.environ.get("PRELOAD_DBG", "1") == "1"
                                    and kp_ % 20 == 0 and jn not in pl_done):
                                _ipd = j1i.get(fl_)
                                print(f">>>     preload[{kp_:4d}] jaw {jn}: "
                                      f"cmd {float(q[_ipd]):+.4f} act "
                                      f"{float(_qa_m[_ipd]):+.4f} err "
                                      f"{float(q[_ipd]) - float(_qa_m[_ipd]):+.4f} rad | "
                                      f"effort {_eff_m.get(fl_, 0.0):+.3f}Nm "
                                      f"(cap {f_max * lever[jn]:.3f}) | "
                                      f"F {self._pad_force_step(fl_):6.2f}N "
                                      f"ema {f_ema[jn]:6.2f}N", flush=True)
                            f_hold[jn] = f_hold[jn] + 1 if f_ema[jn] >= f_tgt else 0
                            if (f_hold[jn] >= f_need and pl_cur[jn] >= pl_min
                                    and jn not in pl_done):
                                pl_done.add(jn)
                                print(f">>>   preload: jaw {jn} ({fl_}) LOADED "
                                      f"{f_ema[jn]:.2f}N sustained {f_hold[jn]} steps "
                                      f"at +{pl_cur[jn]:.3f} rad", flush=True)
                        if len(pl_done) == len(lead_f):
                            break
                    # RESTORE — the soft gains are for the preload window ONLY (chatter mode)
                    if kp_sav is not None:
                        self.ctrl.set_gains(kp_sav, kd_sav)
                    else:
                        self._grip_stiffen()             # rebuild the gains it originally set
                    if eff_sav is not None:
                        try:
                            self.ctrl.set_max_efforts(eff_sav)
                        except Exception:
                            pass
                    print(f">>>   preload done: forces "
                          f"{ {k: round(v, 2) for k, v in f_ema.items()} }N  "
                          f"travel { {k: round(v, 3) for k, v in pl_cur.items()} } rad",
                          flush=True)
                    break
                for fl, (i2, i3) in wrapj.items():
                    if fl in w_done:
                        continue
                    g = g_w[fl]
                    if g < w_best[fl][0]:
                        i1b = j1i.get(fl)                            # ACTUAL, not commanded
                        w_best[fl] = (g, (float(qa_w[i2]),
                                          float(qa_w[i3]) if i3 is not None else None,
                                          float(qa_w[i1b]) if i1b is not None else None))
                    # Only a finger that actually MOVED can have receded. The sync gate holds the
                    # near jaw still while the far one catches up, and a stationary finger's gap
                    # still grows when the object jitters -- so without this the gate retires the
                    # very fingers it is protecting for standing still, releasing the thumb to close
                    # alone.
                    if w_do_probe and w_moved.get(fl, False) and fl not in w_flip:
                        if w_probe[fl] is None:
                            w_probe[fl] = g
                        elif w_cur[fl] >= wp_n * w_rate * self.dt:
                            if g > w_probe[fl]:
                                w_sign[fl] = -1.0
                                print(f">>>   wrap: finger {fl} +{w_mode[fl]} opened it "
                                      f"({w_probe[fl] * 1000:+.1f} -> {g * 1000:+.1f}mm)"
                                      f" -> driving -{w_mode[fl]}", flush=True)
                            w_flip.add(fl)
                            i1p = j1i.get(fl)
                            w_best[fl] = (g, (float(qa_w[i2]),
                                              float(qa_w[i3]) if i3 is not None else None,
                                              float(qa_w[i1p]) if i1p is not None else None))
                            w_cur[fl] = 0.0
                            continue
                    receded = (w_moved.get(fl, False) and fl in w_flip
                               and g > w_best[fl][0] + w_back)
                    if g <= touch:
                        w_done.add(fl)
                        w_reached.add(fl)
                        print(f">>>   wrap: finger {fl} REACHED on {w_mode[fl]} "
                              f"+{w_cur[fl]:.3f} rad (gap {g * 1000:+.1f}mm)", flush=True)
                        continue
                    if receded or w_cur[fl] >= w_cap:
                        if receded and w_best[fl][1] is not None:
                            if w_mode[fl] == "j1":                   # rewind the joint we drove
                                i1r = j1i.get(fl)
                                if i1r is not None and w_best[fl][1][2] is not None:
                                    q[i1r] = w_best[fl][1][2]
                            else:
                                q[i2] = w_best[fl][1][0]
                                if i3 is not None and w_best[fl][1][1] is not None:
                                    q[i3] = w_best[fl][1][1]
                        nxt = w_next.get(w_mode[fl])
                        if nxt is not None:
                            # Switch reach joint instead of giving up (see w_mode, above).
                            print(f">>>   wrap: finger {fl} {w_mode[fl]} exhausted "
                                  f"(gap {g * 1000:+.1f}mm) -> switching to {nxt}", flush=True)
                            w_mode[fl] = nxt
                            w_cur[fl] = 0.0
                            w_best[fl] = (g, None)
                            w_moved[fl] = False          # it has not moved on the new joint YET
                            w_sign[fl] = 1.0             # the new joint has its OWN sign —
                            w_probe[fl] = None           #   re-probe rather than inherit
                            w_flip.discard(fl)
                            continue
                        w_done.add(fl)                   # j1 too — genuinely cannot reach, but
                        # it stays in g_w and keeps blocking the other jaw until the wrap ends
                        print(f">>>   wrap: finger {fl} j1 +{w_cur[fl]:.3f} rad "
                              f"(gap {g * 1000:+.1f}mm, best {w_best[fl][0] * 1000:+.1f}mm)",
                              flush=True)
                        continue
                    if sync_tol > 0 and g_jaw.get("bc" if fl in "bc" else "a",
                                                  g) < w_lead - sync_tol:
                        w_moved[fl] = False              # gated: cannot have receded (see above)
                        continue                         # wait for the far jaw
                    w_moved[fl] = True
                    w_cur[fl] += w_rate * self.dt
                    step = w_sign[fl] * w_rate * self.dt
                    if w_mode[fl] == "j1":
                        i1_ = j1i.get(fl)
                        if i1_ is not None:
                            q[i1_] = float(q[i1_]) + step
                    else:
                        q[i2] = float(q[i2]) + step
                        if i3 is not None:
                            q[i3] = float(q[i3]) - step * w_j3
                self._finger_cmd = q[self._f_idx].copy()
                self.set_base(bx, by, byaw)
                if os.environ.get("CLOSE_HOLD_ARM", "1") == "1":
                    self._force(q)
                self._apply(q)
                self.world.step(render=not HEADLESS)
                if len(w_done) == len(wrapj):
                    break
            # Final approach -- back to j1, now that both jaws are close. The seat deliberately
            # stopped short so nothing was in contact through the settle, and the wrap then closes
            # j2/j3, which reaches for b and c but NOT for the thumb: j2 arcs the thumb AWAY.
            if os.environ.get("FINAL_APPROACH", "1") == "1" and not grip_done:
                f_rate = 0.20     # rad/s, half the seat
                f_back = 0.004
                f_cap = 0.50
                n_fin = max(2, int(2.5 / self.dt))
                f_cur = {fl: 0.0 for fl in j1i}
                f_moved = {fl: False for fl in j1i}      # same guard as the wrap's w_moved
                # Direction probe. j1's pad path has an ARC MINIMUM, and b/c land on either side of
                # it depending on cycle-to-cycle scatter in the object's base-frame position.
                f_sign = {fl: 1.0 for fl in j1i}
                f_probe = {fl: None for fl in j1i}       # gap when this finger first moved
                f_flip = set()
                probe_n = 12
                do_probe = os.environ.get("FINAL_PROBE", "1") == "1"
                oc_f = np.asarray(obj.get_world_poses()[0][0], float)
                f_best = {fl: (self._finger_surface_gap(fl, oc_f, r_obj, self.obj_half_h,
                                                        self._obj_R(obj)), None)
                          for fl in j1i}
                f_done = {fl for fl in j1i if f_best[fl][0] <= touch}
                for kf in range(n_fin):
                    oc_f = np.asarray(obj.get_world_poses()[0][0], float)
                    oR_f = self._obj_R(obj)
                    qa_f = np.asarray(self.robot.get_joint_positions(), float)
                    g_f = {fl: self._finger_surface_gap(fl, oc_f, r_obj, self.obj_half_h, oR_f)
                           for fl in j1i if fl not in f_done}
                    gj_f = {}
                    for jn, fs in (("a", ("a",)), ("bc", ("b", "c"))):
                        gj = [g_f[f] for f in fs if f in g_f]
                        if gj:
                            gj_f[jn] = min(gj)
                    lead_f = max(gj_f.values()) if gj_f else 0.0
                    for fl, i in j1i.items():
                        if fl in f_done:
                            continue
                        g = g_f[fl]
                        if g < f_best[fl][0]:
                            f_best[fl] = (g, float(qa_f[i]))     # the ACTUAL pose, not the command
                        # Sign decision (see f_sign, above). Compare the gap now against the gap
                        # when this finger first moved; if driving has made it WORSE, the pad is
                        # past its arc minimum and the closing direction is the other one.
                        if do_probe and f_moved[fl] and fl not in f_flip:
                            if f_probe[fl] is None:
                                f_probe[fl] = g
                            elif f_cur[fl] >= probe_n * f_rate * self.dt:
                                if g > f_probe[fl]:
                                    f_sign[fl] = -1.0
                                    print(f">>>   final: finger {fl} +j1 opened it "
                                          f"({f_probe[fl] * 1000:+.1f} -> {g * 1000:+.1f}mm)"
                                          f" -> driving -j1", flush=True)
                                f_flip.add(fl)
                                f_best[fl] = (g, float(qa_f[i]))     # re-baseline after probe
                                f_cur[fl] = 0.0
                                continue
                        f_rec = (f_moved[fl] and fl in f_flip
                                 and g > f_best[fl][0] + f_back)
                        if g <= touch or f_cur[fl] >= f_cap or f_rec:
                            if f_rec and f_best[fl][1] is not None:
                                q[i] = f_best[fl][1]
                            f_done.add(fl)
                            print(f">>>   final: finger {fl} j1 +{f_cur[fl]:.3f} rad "
                                  f"(gap {g * 1000:+.1f}mm, best {f_best[fl][0] * 1000:+.1f}mm)",
                                  flush=True)
                            continue
                        if sync_tol > 0 and gj_f.get("bc" if fl in "bc" else "a",
                                                     g) < lead_f - sync_tol:
                            continue                     # wait for the far jaw
                        f_moved[fl] = True
                        f_cur[fl] += f_rate * self.dt
                        q[i] = float(q[i]) + f_sign[fl] * f_rate * self.dt
                    self._finger_cmd = q[self._f_idx].copy()
                    self.set_base(bx, by, byaw)
                    if os.environ.get("CLOSE_HOLD_ARM", "1") == "1":
                        self._force(q)
                    self._apply(q)
                    self.world.step(render=not HEADLESS)
                    if len(f_done) == len(j1i):
                        break
        # Settle the rewind. The seat rewinds a receding finger's COMMAND to its closest-approach
        # pose, but the joint needs time to physically get there -- otherwise the squeeze monitor
        # takes its baseline on unsettled geometry and every decision is made against a pose the
        # hand has already left.
        st.bx, st.by = bx, by
        return grip_done

