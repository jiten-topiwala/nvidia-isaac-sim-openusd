"""Per-finger geometric seat, then the squeeze that turns a cage into a grip.

Part of the close sequence — see `morph/close/__init__.py` for the stage order and the
shared state object. Imports Isaac APIs at module level; only importable after SimulationApp.
"""
import math
import os

import numpy as np

from morph.config import HEADLESS, KNOWN


class SeatStage:
    """One stage of `_close_replay`. Mixed into `CloseMixin`; all `self.*` belong to `Demo`."""

    def _close_grip_seat(self, st):
        q, obj = st.q, st.obj
        bx, by, byaw = st.bx, st.by, st.byaw
        lj_locked = st.lj_locked
        # Squeeze. The recording ends the moment the pads REACH the surface, so replaying it lands
        # the commanded pose exactly on contact and the compliant pads carry F = k*0 = 0 -- a
        # perfectly good cage that the lift drops.
        sq = float(os.environ.get("GRIP_SQUEEZE", "0.0"))
        # SEAT and SQUEEZE are separate things and used to share one knob: `if sq > 0` gated the
        # whole block, so GRIP_SQUEEZE=0 silently skipped the per-finger geometric SEAT too and left
        # every finger 8-15mm out.
        if os.environ.get("GRIP_SEAT", "1") == "1" and not lj_locked:
            # Per-finger geometric seat, then a common over-closure. A single shared over-closure
            # cannot work on this hand: at the recorded grip the fingers sit at very different
            # distances from the surface, one already penetrating while another is centimetres
            # clear, so a common increment either buries one or leaves another in mid-air.
            j1i = {fl: self.idx[f"finger_{fl}_joint_1_1"] for fl in "abc"
                   if f"finger_{fl}_joint_1_1" in self.idx}
            # The arrival tolerance has to be loose enough to accept a finger already on the object.
            touch = 0.003
            standoff = 0.005
            sync_tol = 0.003
            grip_done = False                        # hoisted: SEAT_WRAP=0 must not NameError
            cap = 0.60
            rate = 0.40     # rad/s
            r_obj = KNOWN["object_radius"]
            seated, cur = set(), {fl: 0.0 for fl in j1i}
            best = {fl: (1e9, None) for fl in j1i}           # (min gap, commanded j1 there)
            back = 0.002
            n_seat = max(2, int(2.5 / self.dt))
            for k in range(n_seat):
                oc = np.asarray(obj.get_world_poses()[0][0], float)
                oR_s = self._obj_R(obj)
                # Store the ACTUAL joint position. A rewind must restore the pose the finger was
                # physically in when its gap was minimal; storing the COMMAND stores a target the
                # joint had not reached, which lets the finger keep closing PAST its arc minimum,
                # where the gap grows fast.
                qa_s = np.asarray(self.robot.get_joint_positions(), float)
                # Synchronised seat: advance only the finger(s) currently furthest from the surface.
                g_now = {fl: self._finger_surface_gap(fl, oc, r_obj, self.obj_half_h, oR_s)
                         for fl in j1i if fl not in seated}
                g_lead = max(g_now.values()) if g_now else 0.0
                for fl, i in j1i.items():
                    if fl in seated:
                        continue
                    g = g_now[fl]
                    if g < best[fl][0]:
                        best[fl] = (g, float(qa_s[i]))       # ACTUAL, not commanded
                    # Stop at the MINIMUM, not only on contact: a pad passes through its closest
                    # approach and then arcs away again, so waiting for contact drives a finger that
                    # can never reach all the way to its cap, on the far side of its own arc.
                    if g <= max(touch, standoff) or cur[fl] >= cap or g > best[fl][0] + back:
                        if g > best[fl][0] + back and best[fl][1] is not None:
                            q[i] = best[fl][1]               # rewind to the closest-approach pose
                        seated.add(fl)
                        print(f">>>   seat: finger {fl} at +{cur[fl]:.3f} rad "
                              f"(gap {g * 1000:+.1f}mm, best {best[fl][0] * 1000:+.1f}mm)",
                              flush=True)
                        continue
                    if sync_tol > 0 and g < g_lead - sync_tol:
                        continue                             # wait for the laggards (see above)
                    cur[fl] += rate * self.dt
                    q[i] = float(q[i]) + rate * self.dt
                self._finger_cmd = q[self._f_idx].copy()
                self.set_base(bx, by, byaw)
                if os.environ.get("CLOSE_HOLD_ARM", "1") == "1":
                    self._force(q)                           # same reason as the close loop above
                self._apply(q)
                self.world.step(render=not HEADLESS)
                if len(seated) == len(j1i):
                    break
            # Wrap stage: the underactuated close is j1 -> j2 -> j3, not j1 alone. The seat drives
            # only j1, and j1's pad path has an arc minimum -- on a large cylinder every finger
            # bottoms out short and curling further moves it away.
            grip_done = self._seat_wrap(st, j1i, r_obj, touch, sync_tol, grip_done)
            bx, by = st.bx, st.by
            oc_cg = np.asarray(obj.get_world_poses()[0][0], float)
            caged = grip_done or all(self._finger_surface_gap(fl, oc_cg, r_obj, self.obj_half_h, self._obj_R(obj))
                        <= 0.003 for fl in j1i)
            if caged:
                qa_cg = np.asarray(self.robot.get_joint_positions(), float)
                for _fl in j1i:
                    for _jn in (1, 2, 3):
                        _ji = self.idx.get(f"finger_{_fl}_joint_{_jn}_1")
                        if _ji is not None:
                            q[_ji] = float(qa_cg[_ji])
                self._finger_cmd = q[self._f_idx].copy()
                print(">>>   cage complete on all three — skipping settle + squeeze", flush=True)
            self._hold(q, 0 if caged else
                       int(0.3 / self.dt), kin=True)
            base1 = {fl: float(q[i]) for fl, i in j1i.items()}
            m = 0 if caged else max(2, int(1.2 / self.dt))
            self._fN = {}
            # Per-finger and gap-monitored. A uniform over-closure is wrong here: the seat pose IS
            # the closest approach for the outer fingers, so curling further arcs them AWAY while
            # only the middle one gains.
            mon = os.environ.get("GRIP_SQUEEZE_MONITOR", "1") == "1"
            sq_best = {fl: (self._finger_surface_gap(fl, np.asarray(obj.get_world_poses()[0][0],
                                                                   float),
                                                     KNOWN["object_radius"], self.obj_half_h,
                                                     self._obj_R(obj)),
                            base1[fl]) for fl in j1i} if mon else {}
            sq_done = set()
            for k in range(m):
                f = 0.5 - 0.5 * math.cos(math.pi * min(1.0, (k + 1) / (0.8 * m)))
                if mon and k % 4 == 0:
                    oc_q = np.asarray(obj.get_world_poses()[0][0], float)
                    for fl, i in j1i.items():
                        if fl in sq_done:
                            continue
                        g_q = self._finger_surface_gap(fl, oc_q, KNOWN["object_radius"],
                                                       self.obj_half_h, self._obj_R(obj))
                        if g_q < sq_best[fl][0]:
                            sq_best[fl] = (g_q, float(np.asarray(
                                self.robot.get_joint_positions(), float)[i]))   # ACTUAL
                        elif g_q > sq_best[fl][0] + 0.002:
                            q[i] = sq_best[fl][1]            # rewind to its closest approach
                            sq_done.add(fl)
                            print(f">>>   squeeze: finger {fl} receding "
                                  f"({sq_best[fl][0] * 1000:+.1f} -> {g_q * 1000:+.1f}mm) -> held",
                                  flush=True)
                for fl, i in j1i.items():
                    if fl in sq_done:
                        continue
                    # SEAT_MAX_GAP, not STOP_MAX_GAP: this used to read the replay stage's variable
                    # with a different default (0.012 here, 0.004 there), so setting the knob for
                    # one silently retuned the other.
                    if mon and sq_best[fl][0] > 0.012:
                        sq_done.add(fl)                      # not on the surface -> squeezing it
                        continue                             #   only arcs it further away
                    q[i] = base1[fl] + f * sq
                self._finger_cmd = q[self._f_idx].copy()
                self.set_base(bx, by, byaw)
                if os.environ.get("CLOSE_HOLD_ARM", "1") == "1":
                    # this loop had NO arm hold — which is why the L3 drift climbed back to 73-88mm
                    # right after a 1mm seat
                    self._force(q)
                self._apply(q)
                self.world.step(render=not HEADLESS)
            pads = {k2: round(v / max(1, m) / self.dt, 1) for k2, v in getattr(self, "_fN", {}).items()}
            oc = np.asarray(obj.get_world_poses()[0][0], float)
            gaps = {fl: round(self._finger_surface_gap(fl, oc, r_obj, self.obj_half_h,
                                                       self._obj_R(obj)) * 1000, 1)
                    for fl in j1i}
            print(f">>> seat+squeeze: seat {dict((k2, round(v, 3)) for k2, v in cur.items())} "
                  f"+{sq:.3f} rad  gaps(mm) {gaps}  pad loads (N) {pads}", flush=True)
        st.bx, st.by = bx, by

