"""Recorded-close replay with per-finger freeze on real contact.

Part of the close sequence — see `morph/close/__init__.py` for the stage order and the
shared state object. Imports Isaac APIs at module level; only importable after SimulationApp.
"""
import os

import numpy as np

from morph.config import HEADLESS, KNOWN


class ReplayStage:
    """One stage of `_close_replay`. Mixed into `CloseMixin`; all `self.*` belong to `Demo`."""

    def _close_do_replay(self, st):
        """Replay the recorded close, with a per-finger freeze on real contact.

        Runs dynamically — the arm on stiff drives, with no per-step state writes — so contact
        forces can build: a state write re-targets every drive in the articulation and wipes them.
        """
        q_base, secs, obj = st.q_base, st.secs, st.obj
        ct = self.close_traj
        q = q_base.copy()
        fr, ii = ct["frames"], ct["idx"]
        n = max(2, int(secs / self.dt))
        bx, by, byaw = self.base_ledger()
        frozen, cons = set(), {f: 0 for f in "abc"}
        n_hold = max(2, int(0.015 / self.dt))
        k_min = int(0.2 / self.dt)
        self._finger_cmd = q[self._f_idx].copy()
        self._set_arm_gain(3.0e5)
        if os.environ.get("CLOSE_GENTLE", "1") == "1":
            self._grip_gentle()                              # touch first...
        else:
            self._grip_stiffen()
        self._touch_step = {}
        q0 = q[ii].copy()                                    # blend from the live open pose
        # Off by default: without a start gate it fires during the blend-in and freezes b/c at their
        # entry gaps.
        geo_stop = os.environ.get("CLOSE_GEO_STOP", "0") == "1"
        geo_back = 0.003
        cstand = float(os.environ.get("CLOSE_STANDOFF", "0.005"))
        gbest = {fl: (1e9, None) for fl in "abc"}
        for k in range(n):
            now, self._touch_step = self._touch_step, {}
            # A geometric closest-approach stop, alongside the contact-event stop. A finger whose
            # arc never reaches the object passes through its closest approach mid-close and then
            # recedes, so by the time a later stage looks at it its own "best" is already the
            # receded value.
            if k % 4 == 0 and geo_stop:
                oc_g = np.asarray(obj.get_world_poses()[0][0], float)
                for fl in "abc":
                    if fl in frozen:
                        continue
                    g_ = self._finger_surface_gap(fl, oc_g, KNOWN["object_radius"], self.obj_half_h,
                                                 self._obj_R(obj))
                    ji_ = self.idx.get(f"finger_{fl}_joint_1_1")
                    if g_ < gbest[fl][0]:
                        gbest[fl] = (g_, None if ji_ is None else float(q[ji_]))
                    elif k > k_min and g_ > gbest[fl][0] + geo_back and gbest[fl][1] is not None:
                        q[ji_] = gbest[fl][1]
                        frozen.add(fl)
                        print(f">>> close: finger {fl} closest-approach stop (k={k}, "
                              f"best {gbest[fl][0] * 1000:+.1f}mm, now {g_ * 1000:+.1f}mm)",
                              flush=True)
            # Geometric standoff freeze, checked every step and independent of contact events.
            if cstand > 0 and k > k_min:
                oc_c = np.asarray(obj.get_world_poses()[0][0], float)
                oR_c = self._obj_R(obj)
                for fl in "abc":
                    if fl in frozen:
                        continue
                    if self._finger_surface_gap(fl, oc_c, KNOWN["object_radius"], self.obj_half_h,
                                                oR_c) <= cstand:
                        frozen.add(fl)
                        # Snap command := actual. Freezing only stops UPDATING q; the joint keeps
                        # travelling to the last command it was given, which on a force-limited
                        # drive is ahead of where it currently is.
                        qa_c = np.asarray(self.robot.get_joint_positions(), float)
                        for _jn in (1, 2, 3):
                            _ji = self.idx.get(f"finger_{fl}_joint_{_jn}_1")
                            if _ji is not None:
                                q[_ji] = float(qa_c[_ji])
                        print(f">>> close: finger {fl} standoff freeze (k={k})", flush=True)
            for fl in "abc":
                cons[fl] = cons[fl] + 1 if now.get(fl) else 0
                if fl not in frozen and cons[fl] >= n_hold and k > k_min:
                    # Gate the contact event on measured geometry. A contact event only says the
                    # finger touched SOMETHING — the floor, a neighbouring link, the object's top
                    # face — and trusting it blindly freezes a finger while its pads are still far
                    # from the object, leaving the others to shove the object with no opposing jaw.
                    #
                    # The threshold has to be tight. Set loose, every finger "contact-stops"
                    # carrying no load at all, well short of the recorded grip, which reads as the
                    # grasp being unreachable when it is only this gate firing early.
                    gmax = 0.004
                    if gmax > 0:
                        oc_s = np.asarray(obj.get_world_poses()[0][0], float)
                        g_s = self._finger_surface_gap(fl, oc_s, KNOWN["object_radius"],
                                                       self.obj_half_h, self._obj_R(obj))
                        if g_s > gmax:
                            cons[fl] = 0                     # spurious touch — keep closing
                            continue
                    frozen.add(fl)
                    # Snap command := actual here too. A "frozen" finger on a force-limited drive
                    # keeps pressing, because q still holds a command ahead of the joint's actual
                    # position — and a lone finger pressing an unopposed cylinder tips it before the
                    # other two arrive.
                    qa_f = np.asarray(self.robot.get_joint_positions(), float)
                    for _jn3 in (1, 2, 3):
                        _ji3 = self.idx.get(f"finger_{fl}_joint_{_jn3}_1")
                        if _ji3 is not None:
                            q[_ji3] = float(qa_f[_ji3])
                    self._finger_cmd = q[self._f_idx].copy()
                    print(f">>> close: finger {fl} contact-stop (k={k})", flush=True)
            # Approach fast, touch slow. The close cannot run at one speed: at the replay rate a
            # finger needs several times more torque just to beat its own damping than the force
            # that tips this cylinder, so no single effort both moves the finger and stays under the
            # tipping limit.
            if os.environ.get("CLOSE_SLOW", "1") == "1":
                if k == 0:
                    self._cl_ph = 0.0
                _gmin = 1e9
                if k % 5 == 0:
                    _oc_p = np.asarray(obj.get_world_poses()[0][0], float)
                    _oR_p = self._obj_R(obj)
                    _gmin = min(self._finger_surface_gap(fl_, _oc_p, KNOWN["object_radius"],
                                                         self.obj_half_h, _oR_p)
                                for fl_ in "abc" if fl_ not in frozen) if len(frozen) < 3 else 0.0
                    self._cl_gmin = _gmin
                _gm = getattr(self, "_cl_gmin", 1e9)
                _far = 0.020
                _near = 0.004
                _floor = 0.12
                _sc = 1.0 if _gm >= _far else (
                    _floor if _gm <= _near else
                    _floor + (1.0 - _floor) * (_gm - _near) / max(1e-9, _far - _near))
                self._cl_ph += _sc * (len(fr) - 1) / (n - 1)
                f = min(float(self._cl_ph), float(len(fr) - 1))
            else:
                f = k / (n - 1) * (len(fr) - 1)
            i0 = int(f)
            i1 = min(i0 + 1, len(fr) - 1)
            row = (1 - (f - i0)) * fr[i0] + (f - i0) * fr[i1]
            s_in = min(1.0, k / max(1.0, 0.15 * n))          # ease off the live pose onto frame 0
            for pos, nm in enumerate(ct["names"]):
                fl = next((c for c in "abc" if f"_{c}_" in nm), None)
                if fl in frozen:
                    continue
                q[ii[pos]] = (1 - s_in) * q0[pos] + s_in * row[pos]
            # Per-step trace. Phase-boundary snapshots a thousand steps apart cannot show WHEN a
            # replay goes wrong.
            if os.environ.get("CLOSE_TRACE", "0") == "1" and k % 25 == 0:
                _oc_t = np.asarray(obj.get_world_poses()[0][0], float)
                _oR_t = self._obj_R(obj)
                _g_t = {fl: round(self._finger_surface_gap(
                    fl, _oc_t, KNOWN["object_radius"], self.obj_half_h, _oR_t) * 1000, 1)
                    for fl in "abc"}
                print(f">>> TRACE k={k:4d} obj {np.round(_oc_t, 4).tolist()} "
                      f"axis {np.round(_oR_t[:, 2], 3).tolist()} gaps {_g_t} "
                      f"frozen {sorted(frozen)}", flush=True)
            self._finger_cmd = q[self._f_idx].copy()
            self.set_base(bx, by, byaw)
            # Hold the arm. A state write wipes contact forces, which argues for drives alone — but
            # in NO_LOOP mode the four-bar's closure joint is outside the articulation, so the
            # passive links are held ONLY by the kinematic write, and without it the linkage
            # collapses under the hand while the fingers close onto empty space.
            if os.environ.get("CLOSE_HOLD_ARM", "1") == "1":
                self._force(q)
            self._apply(q)
            self.world.step(render=not HEADLESS)
        if os.environ.get("CLOSE_GENTLE", "1") == "1":
            self._grip_stiffen()                             # ...stiffen second (the documented order)
        qa = np.asarray(self.robot.get_joint_positions(), float)
        print(f">>> close-replay done: engaged={sorted(frozen)}, j1 a/b/c = "
              f"{float(qa[self.idx['finger_a_joint_1_1']]):+.2f}/"
              f"{float(qa[self.idx['finger_b_joint_1_1']]):+.2f}/"
              f"{float(qa[self.idx['finger_c_joint_1_1']]):+.2f}", flush=True)
        st.q, st.bx, st.by, st.byaw = q, bx, by, byaw

