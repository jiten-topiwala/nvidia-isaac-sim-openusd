"""Vendor legal-range close — continue past the replay's j1 into 0.0495..1.2218.

Part of the close sequence — see `morph/close/__init__.py` for the stage order and the
shared state object. Imports Isaac APIs at module level; only importable after SimulationApp.
"""
import os

import numpy as np

from morph.config import PERFECT, HEADLESS, KNOWN


class LegalStage:
    """One stage of `_close_replay`. Mixed into `CloseMixin`; all `self.*` belong to `Demo`."""

    def _close_legal(self, st):
        q, obj = st.q, st.obj
        bx, by, byaw = st.bx, st.by, st.byaw
        lj_locked = st.lj_locked
        if os.environ.get("LEGAL_CLOSE", "0") == "1" and PERFECT:
            lj_t = 0.75
            lj_r = 0.15        # rad/s
            lj_touch = 0.003
            n_lj = max(2, int(6.0 / self.dt))
            j1L = {f: self.idx[f"finger_{f}_joint_1_1"] for f in "abc"
                   if f"finger_{f}_joint_1_1" in self.idx}
            lj_sq = 0.04    # rad/s once synchronised
            lj_pen = 0.002      # allowed pad penetration
            lj_dmax = 0.020   # squeeze abort on shove
            fgate = 0.0015
            lj_fN = 1.0
            lj_hold = 8         # consecutive loaded steps
            lj_n = {f: 0 for f in j1L}
            lj_done, lj_best = set(), {f: 1e9 for f in j1L}
            lj_bestq = {f: None for f in j1L}      # j1 at each finger's closest approach
            lj_sync, lj_oc0 = False, None
            lj_ax = st.jaw_axis        # measured by _grasp_geometry_report just before this stage
            # Vendor distal curl. The vendor's own SRDF pre-curls the distal phalanx inward in BOTH
            # the open and close poses, while the recorded replay holds it nearly flat -- which is
            # why the pads meet a large cylinder tangentially.
            j3L, j3_rate = {}, {}
            if os.environ.get("VENDOR_CURL", "0") == "1":
                j3_t = -0.65
                for f in j1L:
                    ix3 = self.idx.get(f"finger_{f}_joint_3_1")
                    if ix3 is None:
                        continue
                    span1 = lj_t - float(q[j1L[f]])
                    j3L[f] = ix3
                    j3_rate[f] = ((j3_t - float(q[ix3])) / span1) if abs(span1) > 1e-6 else 0.0
                print(f">>> legal close: VENDOR CURL on — j3 -> {j3_t} alongside j1 "
                      f"(srdf 'close': j1 0.6, j2 0.001, j3 -0.65)", flush=True)
            print(f">>> legal close: driving j1 -> {lj_t:+.3f} (vendor legal range "
                  f"+0.0495..+1.2218) at {lj_r} rad/s", flush=True)
            for kL in range(n_lj):
                ocL = np.asarray(obj.get_world_poses()[0][0], float)
                oRL = self._obj_R(obj)
                gAll = {f: self._finger_surface_gap(f, ocL, KNOWN["object_radius"], self.obj_half_h, oRL)
                        for f in j1L}
                # Two stages, and the distinction is the point. APPROACH: a finger that reaches the
                # surface WAITS -- it does not advance and it is not retired.
                if not lj_sync and max(gAll.values()) <= lj_touch:
                    lj_sync, lj_oc0 = True, ocL.copy()
                    print(f">>>   legal close: ALL THREE within {lj_touch * 1000:.0f}mm "
                          f"({ {k: round(v * 1000, 1) for k, v in gAll.items()} }) — synchronised, "
                          f"squeezing at {lj_sq} rad/s", flush=True)
                # Object shoved during the squeeze -> the jaws are not opposing each other; stop
                # everything rather than push it out of the hand.
                if lj_sync and float(np.linalg.norm(ocL - lj_oc0)) > lj_dmax:
                    # Which way did it go? Along the jaw axis means one jaw out-pushing the other.
                    # ACROSS it = the object squirting OUT OF THE MOUTH, which is a depth problem,
                    # not a size problem.
                    _dv = (ocL - lj_oc0)[:2]
                    _al = _ac = float("nan")
                    if lj_ax is not None:
                        _al = float(_dv @ lj_ax) * 1000.0
                        _ac = float(_dv @ np.array([-lj_ax[1], lj_ax[0]])) * 1000.0
                    print(f">>>   legal close: ABORT — squeeze moved the object "
                          f"{np.linalg.norm(ocL - lj_oc0) * 1000:.1f}mm  "
                          f"(along jaw axis {_al:+.1f}mm, across/out of mouth {_ac:+.1f}mm)",
                          flush=True)
                    qaA = np.asarray(self.robot.get_joint_positions(), float)
                    for f in j1L:
                        for _jn5 in (1, 2, 3):
                            _ji5 = self.idx.get(f"finger_{f}_joint_{_jn5}_1")
                            if _ji5 is not None:
                                q[_ji5] = float(qaA[_ji5])
                    self._finger_cmd = q[self._f_idx].copy()
                    break
                for f, ix in j1L.items():
                    if f in lj_done:
                        continue
                    gL = gAll[f]
                    if gL < lj_best[f]:
                        lj_best[f], lj_bestq[f] = gL, float(q[ix])
                    # Stop at the closest approach, not at a fixed angle: each pad travels an ARC,
                    # so the gap falls to a minimum and then GROWS as the fingertip curls past the
                    # object.
                    if (lj_bestq[f] is not None
                            and lj_best[f] <= 0.015
                            and gL > lj_best[f] + 0.0015):
                        # Stop in place, do NOT rewind to the remembered angle. The object shifts
                        # slightly during the close, so the j1 that was best no longer reproduces
                        # that gap -- a stale setpoint is worse than stopping a hair past the
                        # minimum.
                        if os.environ.get("LEGAL_REWIND", "0") == "1":
                            q[ix] = lj_bestq[f]
                        self._lj_stop(q, f, ix, gL, lj_best, lj_done,
                                      f"past closest approach (best {lj_best[f] * 1000:.1f}mm)")
                        continue
                    if float(q[ix]) >= lj_t:
                        self._lj_stop(q, f, ix, gL, lj_best, lj_done, "j1 limit")
                        continue
                    if not lj_sync:
                        if gL <= lj_touch:
                            continue                 # WAIT for the others — do NOT retire
                        q[ix] = float(q[ix]) + lj_r * self.dt
                        if f in j3L:
                            q[j3L[f]] = float(q[j3L[f]]) + j3_rate[f] * lj_r * self.dt
                        continue
                    # Squeeze: stop this finger only when ITS pad actually loads, and gate the force
                    # on measured geometry.
                    ga_ = 0.0 if "a" in lj_done else gAll["a"]
                    _bc = [0.0 if x in lj_done else gAll[x] for x in ("b", "c") if x in gAll]
                    gbc_ = min(_bc) if _bc else 0.0
                    mine, other = (ga_, gbc_) if f == "a" else (gbc_, ga_)
                    if mine < other - 0.001:
                        continue                     # this jaw is ahead; let the other catch up
                    fL = self._pad_force_step(f)
                    lj_n[f] = lj_n[f] + 1 if (fL >= lj_fN and gL <= fgate) else 0
                    if lj_n[f] >= lj_hold:
                        self._lj_stop(q, f, ix, gL, lj_best, lj_done,
                                      f"loaded {fL:.2f}N for {lj_hold} steps")
                        continue
                    if gL <= -lj_pen:
                        self._lj_stop(q, f, ix, gL, lj_best, lj_done, "penetrating")
                        continue
                    q[ix] = float(q[ix]) + lj_sq * self.dt
                    if f in j3L:
                        q[j3L[f]] = float(q[j3L[f]]) + j3_rate[f] * lj_sq * self.dt
                self._finger_cmd = q[self._f_idx].copy()
                self.set_base(bx, by, byaw)
                if os.environ.get("CLOSE_HOLD_ARM", "1") == "1":
                    self._force(q)
                self._apply(q)
                self._fN_step = {}                           # THIS step's contact impulses only
                self.world.step(render=not HEADLESS)
                if len(lj_done) == len(j1L):
                    break
            ocL = np.asarray(obj.get_world_poses()[0][0], float)
            gL2 = {f: round(self._finger_surface_gap(f, ocL, KNOWN["object_radius"], self.obj_half_h,
                                                     self._obj_R(obj)) * 1000, 1) for f in "abc"}
            print(f">>> legal close done: gaps {gL2}  obj {np.round(ocL, 3).tolist()} "
                  f"axis {np.round(self._obj_R(obj)[:, 2], 3).tolist()}", flush=True)
            # Did it actually grip -- every finger retired, and retired ON THE OBJECT rather than on
            # the j1 limit?
            lj_locked = (len(lj_done) == len(j1L)
                         and max(gL2.values()) <= lj_touch * 1000.0)
            if lj_locked:
                print(">>> legal close: GRIPPED — skipping seat / settle / wrap / preload / "
                      "squeeze (this stage already did their job)", flush=True)

        st.lj_locked = lj_locked

        # j1 sweep diagnostic: map gap against j1 across the hardware-legal range.
        #
        # The vendor URDF gives finger joint_1 a limit of 0.0495 .. 1.2218, i.e. j1 closes in the
        # POSITIVE direction with about 1.17 rad of travel. The recorded replay runs j1 from -1.04
        # (open) to -0.44 (grip), entirely outside that range — the reference simulator permits it
        # because its joint limits are soft, but the hardware cannot occupy that region at all. Any
        # reading about which way a finger's arc goes is only meaningful inside the legal range, so
        # this sweeps it and prints the curve rather than guessing.

