"""The visible finger close: MuJoCo curl profile with a geometric per-finger surface stop.

Part of the pick cycle — see `morph/pick/__init__.py` for the stage order and the shared
state object. Imports Isaac APIs at module level; only importable after SimulationApp.
"""
import math
import os

import numpy as np

from morph.config import PERFECT, HEADLESS, KNOWN


class FingerAnimStage:
    """One stage of `pick`. Mixed into `PickMixin`; all `self.*` belong to `Demo`."""

    def _animate_fingers(self, q_base, targets, secs, pin=None, stagger=0.15, on_step=None,
                         stop_obj=None, delays=None):
        """Kinematically animate the finger joints to `targets` at a FIXED arm config (the whole
        config is re-forced every step, so the arm linkage loop is untouched; the object is a
        collision-free puppet, so finger-object contact — the old driven-close explosion source —
        physically cannot happen).  Per-finger start stagger mimics MuJoCo's close (~0.15s spread).
        stop_obj=(obj, radius): per-finger GEOMETRIC contact-stop — the kinematic twin of MuJoCo's
        force-stop: a finger FREEZES the moment its tip reaches the object surface, so fingertips
        land ON the object, never inside it."""
        q = q_base.copy()
        n = max(2, int(secs / self.dt))
        moves = []                                           # (finger, joint idx, start, target, start-frac)
        for jn, tgt in targets.items():
            if jn in self.idx:
                d = next((c for c in "abc" if f"_{c}_" in jn), "a")
                if delays is not None:
                    # MEASURED stagger (MuJoCo's rule): each finger waits (d_max - d_own)/rate, so
                    # the FARTHEST pad sets off first and all three land on the surface together.
                    d0 = float(delays.get(d, 0.0)) / max(secs, 1e-6)
                else:
                    d0 = {"a": 1.0, "b": 0.0, "c": 0.5}[d] * stagger / max(secs, 1e-6)
                # phalanx window within this finger's own close: j1 leads, j3 hooks last
                w0, w1 = (0.0, 1.0)
                if PERFECT:
                    w0, w1 = {"1": (0.00, 0.55), "2": (0.45, 0.85),
                              "3": (0.75, 1.00)}.get(jn.split("joint_")[-1][0], (0.0, 1.0))
                moves.append((d, self.idx[jn], float(q[self.idx[jn]]), float(tgt), d0, w0, w1, jn))
        # HULL stop (default): measure the finger's real collider against the object surface.
        # FINGER_HULL_STOP=0 falls back to the old link_3-origin + FINGER_STOP_PAD proxy.
        fjoints = {}
        for mv in moves:
            fjoints.setdefault(mv[0], []).append(mv[1])
        safe_q = {}                                          # last pose per finger still clear
        use_hull = os.environ.get("FINGER_HULL_STOP", "1") == "1"
        hull_gap = 0.002   # pad stops this far OUT of the
        # Jaw stop. The recorded grip pose closes this two-jaw hand to a fixed opening, because it
        # was recorded on a smaller cylinder -- replayed against a larger object it drives each pad
        # through the surface, which is why the gripper appears to sink toward the centre.
        jaw_stop = os.environ.get("JAW_STOP", "0") == "1" and stop_obj is not None
        jaw_margin = 0.004
        pad = 0.005   # fingertip standoff from the
        # ...against the object SURFACE. Too large a standoff stops the tips visibly short, and
        # asymmetrically, because each finger freezes wherever its step happened to cross the ring.
        # Measured sweep of tip distance from the object axis against a radius of 80mm:
        #   0.030 -> 93/100/97   0.015 -> 89/95/95   0.005 -> 85/85/85   0.000 -> 80/80/80
        # 0.005 lands all three symmetrically just off the surface, tips ON the object without the
        # distal geometry visibly sinking in.
        frozen = set()
        bx, by, byaw = self.base_ledger()
        force_stop = PERFECT and stop_obj is not None       # closing ON an object -> per-finger force-stop
        if force_stop:
            self._touch = {}                                 # fresh touch ledger for THIS close
        if PERFECT:
            # Perfect mode: the fingers are DRIVE-ONLY from here. The staggered targets go to the
            # force-limited PD drives and real pad contact stops them on the surface -- the
            # geometric stop is the practical-mode substitute for that.
            self._finger_cmd = q[self._f_idx].copy()
            stop_obj = None
        lag_stop = 0.30   # SAFETY jam only — a finger
        # Dynamic close: stop state-writing the arm while the fingers work. `set_joint_positions`
        # cannot write a subset -- the tensor API rewrites the whole articulation and re-targets
        # every drive to its measured position each step, so finger contact forces are wiped before
        # they can build.
        dyn = PERFECT and force_stop and os.environ.get("CLOSE_DYNAMIC", "1") == "1"
        if dyn:
            self._set_arm_gain(3.0e5)
            self._grip_stiffen()
            print(">>> close: DYNAMIC (arm on stiff drives, no state writes)", flush=True)

        f_dbg = os.environ.get("CLOSE_DBG") == "1"
        s0j = {fl: float(q[self.idx[f"finger_{fl}_joint_1_1"]])
               for fl in "abc" if f"finger_{fl}_joint_1_1" in self.idx}
        k_min = int(0.25 / self.dt)                          # skip the start transient
        cons = {f: 0 for f in "abc"}                         # consecutive steps of live pad contact
        n_hold = max(2, int(0.015 / self.dt))
        # A short freeze filter is self-defeating: the pads DO reach the object, but an unfrozen
        # finger keeps curling through it and squeezes it out, so contact never lasts long enough to
        # trip the filter BECAUSE nothing stopped.
        for k in range(n):
            if force_stop:
                # Sustained contact only. Freezing on the FIRST contact event froze fingers on a
                # transient graze from a proximal link sweeping past the object (b froze at k=305 of
                # 600 with the pads still 90mm away, and the "grip" then held nothing).
                now = self._touch_step
                self._touch_step = {}
                for fl in "abc":
                    cons[fl] = cons[fl] + 1 if now.get(fl) else 0
            if force_stop and k > k_min:
                # Per-finger contact stop. Joint REACTION forces cannot be used here -- the
                # structural baseline is noisier than any contact spike -- but the force-limited
                # drive gives a clean signal: a finger pressing the object STALLS behind its
                # commanded position.
                qa = np.asarray(self.robot.get_joint_positions(), float)
                for fl in "abc":
                    if fl in frozen:
                        continue
                    j1 = self.idx.get(f"finger_{fl}_joint_1_1")
                    if j1 is None:
                        continue
                    adv0 = float(q[j1] - s0j.get(fl, q[j1]))
                    # Pad sensor: sustained contact, and only once the finger has actually closed
                    # 0.1 rad (a contact still live from the approach otherwise froze the thumb at
                    # its OPEN pose, k=61)
                    if cons[fl] >= n_hold and adv0 > 0.10:
                        frozen.add(fl)
                        # gentle freeze — the deep load ramps in AFTER all fingers engage (instant
                        # deep biases lunged and batted the object 25cm)
                        q[j1] = qa[j1] + 0.03
                        for jn2, bias in ((f"finger_{fl}_joint_2_1", 0.02),
                                          (f"finger_{fl}_joint_3_1", -0.03)):
                            if jn2 in self.idx:
                                q[self.idx[jn2]] = qa[self.idx[jn2]] + bias
                        print(f">>> close[friction]: finger {fl} TOUCH-stop (k={k})", flush=True)
                        continue
                    lag = float(q[j1] - qa[j1])              # closing dir positive for j1
                    adv = float(q[j1] - s0j.get(fl, q[j1]))  # finger must actually be EN ROUTE
                    if lag > lag_stop and adv > 0.15:
                        # Safety jam-stop only, now that `lag_stop` sits well above free-motion
                        # tracking error.
                        frozen.add(fl)
                        q[j1] = qa[j1] + 0.04                # contact + squeeze bias
                        for jn2, bias in ((f"finger_{fl}_joint_2_1", 0.03),
                                          (f"finger_{fl}_joint_3_1", -0.03)):   # j3 closes negative
                            if jn2 in self.idx:
                                q[self.idx[jn2]] = qa[self.idx[jn2]] + bias
                        print(f">>> close[friction]: finger {fl} stall-stop "
                              f"(lag {lag:.3f} rad, k={k})", flush=True)
                if f_dbg and k % 30 == 0:
                    rows = []
                    for fl in "abc":
                        jj = self.idx.get(f"finger_{fl}_joint_1_1")
                        if jj is not None:
                            rows.append(f"{fl}:lag={float(q[jj] - qa[jj]):.3f},adv={float(q[jj] - s0j.get(fl, q[jj])):.3f},fr={fl in frozen}")
                    print(f"    close k={k} " + " ".join(rows), flush=True)
            if jaw_stop and k % 3 == 0 and len(frozen) < 3:
                tp_ = self._tip_prims()
                pa_ = np.asarray(tp_["a"].get_world_pose()[0], float)
                pbc_ = 0.5 * (np.asarray(tp_["b"].get_world_pose()[0], float)
                              + np.asarray(tp_["c"].get_world_pose()[0], float))
                jaw_ = float(np.linalg.norm(pa_ - pbc_))
                if jaw_ <= 2.0 * stop_obj[1] + jaw_margin:
                    for f_ in "abc":
                        if f_ not in frozen:
                            frozen.add(f_)
                    print(f">>> close: JAW stop at {jaw_ * 1000:.0f}mm "
                          f"(object {2000 * stop_obj[1]:.0f}mm + {jaw_margin * 1000:.0f}mm)",
                          flush=True)
            if stop_obj is not None:
                o, radius = stop_obj
                oc = np.asarray(o.get_world_poses()[0][0], float)
                if use_hull:
                    # Real-geometry stop: freeze each finger when its own distal COLLIDER reaches
                    # the cylinder surface.
                    for f in "abc":
                        if f in frozen:
                            continue
                        g = self._finger_surface_gap(f, oc, radius, self.obj_half_h, self._obj_R(o))
                        if g <= hull_gap:
                            frozen.add(f)
                            if g < 0.0 and f in safe_q:
                                # One-step overshoot. The thumb travels 1.021 rad over the close —
                                # ~4-6mm of pad per step — so it can be outside the surface at step
                                # k and 2.6mm INSIDE at k+1, whatever the threshold (measured: a
                                # ended negative at every hull_gap while b and c tracked it
                                # exactly).
                                for ji_ in fjoints.get(f, ()):
                                    q[ji_] = safe_q[f][ji_]
                                g = self._finger_surface_gap(f, oc, radius, self.obj_half_h,
                                                             self._obj_R(o))
                            print(f">>> close: finger {f} stopped at surface "
                                  f"(pad {g * 1000:+.1f}mm from surface)", flush=True)
                        else:
                            safe_q[f] = {ji_: float(q[ji_]) for ji_ in fjoints.get(f, ())}
                        if os.environ.get("HULL_DBG") == "1" and k % 20 == 0:
                            print(f"    hull k={k} {f} gap={g * 1000:+.1f}mm "
                                  f"safe={f in safe_q}", flush=True)
                    stop_iter = ()
                else:
                    stop_iter = self._tip_prims().items()
                for f, prim in stop_iter:
                    if f not in frozen:
                        tp = np.asarray(prim.get_world_pose()[0], float)
                        dxy = math.hypot(tp[0] - oc[0], tp[1] - oc[1])
                        # pure pad-face stop: freeze when the pad face (which extends ~27mm inward
                        # of the link_3 origin) reaches the surface.
                        if dxy <= radius + pad:
                            frozen.add(f)                    # tip AT the surface -> this finger stops
                            print(f">>> close: finger {f} stopped at surface "
                                  f"(tip {dxy * 1000:.0f}mm from axis)", flush=True)
            for fi, ji, s0, tgt, d0, w0, w1, jname in moves:
                # The whole finger freezes on contact. Letting j2/j3 finish afterwards is worse: the
                # distal phalanges hook inward toward the palm, so the fingertip centroid pulls away
                # from the object and the three tips stop asymmetrically.
                if fi in frozen:
                    continue
                s = float(np.clip(((k + 1) / n - d0) / max(1e-6, 1.0 - d0), 0.0, 1.0))
                # Underactuated wrap: the phalanges close in SEQUENCE — proximal swings in first,
                # then the middle, then the distal hooks.
                s = float(np.clip((s - w0) / max(1e-6, w1 - w0), 0.0, 1.0))
                s = 0.5 - 0.5 * math.cos(math.pi * s)        # smoothstep, like MuJoCo's _set_gripper
                q[ji] = s0 + s * (tgt - s0)
            if PERFECT:
                self._finger_cmd = q[self._f_idx].copy()
            self.set_base(bx, by, byaw)
            if not dyn:
                self._force(q)
            self._apply(q)
            if pin is not None and not PERFECT:
                o, p = pin
                o.set_world_poses(positions=np.array([p]), orientations=np.array([[1, 0, 0, 0]]))
                o.set_velocities(np.zeros((1, 6)))
            if on_step is not None:
                on_step()
            self.world.step(render=not HEADLESS)
        if force_stop:
            # Squeeze stage: the animation targets alone left fingers that never met the object
            # curled in air (2-of-3 contact, object toppled at lift).
            n_sq = int(4.0 / self.dt)
            sq_rate = 0.12    # rad/s
            sq_cap = 1.15      # rad, j1
            for k in range(n_sq):
                if frozen >= {"a", "b", "c"}:
                    break
                now = self._touch_step
                self._touch_step = {}
                for fl in "abc":
                    cons[fl] = cons[fl] + 1 if now.get(fl) else 0
                qa = np.asarray(self.robot.get_joint_positions(), float)
                for fl in "abc":
                    if fl in frozen:
                        continue
                    j1 = self.idx.get(f"finger_{fl}_joint_1_1")
                    if j1 is None:
                        continue
                    if cons[fl] >= n_hold and float(q[j1] - s0j.get(fl, q[j1])) > 0.10:
                        frozen.add(fl)
                        q[j1] = qa[j1] + 0.04
                        print(f">>> squeeze: finger {fl} TOUCH (k={k})", flush=True)
                        continue
                    # safety jam at squeeze pace (slow advance -> small legitimate lag, but this
                    # must still sit above free-motion tracking error, not below it)
                    if float(q[j1] - qa[j1]) > 0.15:
                        frozen.add(fl)
                        q[j1] = qa[j1] + 0.04
                        print(f">>> squeeze: finger {fl} jammed (k={k})", flush=True)
                        continue
                    q[j1] = min(q[j1] + sq_rate * self.dt, sq_cap)   # gentle advance, hard cap
                self._finger_cmd = q[self._f_idx].copy()
                self.set_base(bx, by, byaw)
                if not dyn:
                    self._force(q)
                self._apply(q)
                self.world.step(render=not HEADLESS)
            qa_e = np.asarray(self.robot.get_joint_positions(), float)
            print(f">>> squeeze done: engaged={sorted(frozen)}, j1 actual="
                  f"{ {fl: round(float(qa_e[self.idx[f'finger_{fl}_joint_1_1']]), 2) for fl in 'abc'} }"
                  f" (full curl 0.60)", flush=True)
            # Load phase: with all fingers seated, ramp the rim-hold undercut in slowly — distal
            # hooks under the rim (j3), preload (j1/j2) — 1.5s, no lunge
            n_ld = int(1.5 / self.dt)
            ld = {"1": 0.05, "2": 0.03, "3": -0.07}
            q_base2 = q.copy()
            for k in range(n_ld):
                f3 = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / n_ld)
                for fl in frozen:
                    if fl == "palm":
                        continue
                    for jn3, dv in ld.items():
                        jn4 = f"finger_{fl}_joint_{jn3}_1"
                        if jn4 in self.idx:
                            q[self.idx[jn4]] = q_base2[self.idx[jn4]] + f3 * dv
                self._finger_cmd = q[self._f_idx].copy()
                self.set_base(bx, by, byaw)
                if not dyn:
                    self._force(q)
                self._apply(q)
                self.world.step(render=not HEADLESS)
            self._grip_stiffen()                             # stiff HOLD only after gentle touch
            if dyn:
                qa_d = np.asarray(self.robot.get_joint_positions(), float)
                drift = max(abs(float(qa_d[self.idx[n]] - q_base[self.idx[n]]))
                            for n in KNOWN["arm_joints"] if n in self.idx)
                print(f">>> close: arm held on drives, worst joint drift {drift:.4f} rad",
                      flush=True)
            print(">>> load phase done (grip loaded + stiffened)", flush=True)
        return q

