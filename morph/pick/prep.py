"""Finger poses used around the close: the pre-close open and the palm spread.

Part of the pick cycle — see `morph/pick/__init__.py` for the stage order and the shared
state object. Imports Isaac APIs at module level; only importable after SimulationApp.
"""
import os

import numpy as np

from morph.config import HEADLESS, KNOWN


class PrepStage:
    """One stage of `pick`. Mixed into `PickMixin`; all `self.*` belong to `Demo`."""

    def _pre_close_open(self, q, obj, secs=0.6):
        """Retract the fingers until every one clears the object, and return that pose.

        The descent can arrive with the THUMB already touching — its collider slightly inside the
        object while b and c are still centimetres out. The contact stop then freezes the thumb on
        step 0, so it never closes at all. A real gripper opens before it closes; do that, sized
        from the measured gap.

        ONE-SHOT and STEPLESS on purpose.  The first version probed in a loop, stepping the sim after
        each 0.04 rad — that ran `_force()`, which (unlike `_apply()`) has no stale-view revive
        guard, and it killed a headed session with "Failed to get DOF positions from backend" after
        the window had idled. The gap is very nearly linear in j1 (0.76 rad moves the thumb
        from -0.9mm to +13.6mm = 19mm/rad), so one analytic solve is enough; the close animation
        interpolates from whatever pose we return, so no settle is needed here either."""
        # Never retract past the widest pose. Measured jaw opening, thumb pad to the fused b/c pad:
        #     j1 -1.571 -> 172mm      j1 -1.040 -> 188mm  (widest, and the recorded open)
        #     j1 -0.490 -> 168mm      j1 -0.018 -> 126mm  (the recorded grip)
        # The jaw NARROWS again past -1.04, because the finger arcs back over -- so retracting
        # further toward the joint limit actually costs aperture.
        j1_widest = float(os.environ.get("J1_WIDEST", "-1.04"))
        need = 0.012      # clear the surface by this much
        # Metres of gap gained per radian of j1 retraction. Deliberately conservative, so the
        # retraction saturates at PRE_OPEN_CAP and the hand opens FULLY before closing rather than
        # creeping open by the bare minimum.
        slope = 0.0010
        cap = 1.20          # rad, safety clamp
        q = q.copy()
        try:
            oc = np.asarray(obj.get_world_poses()[0][0], float)
            r = KNOWN["object_radius"]
            moved, before = {}, {}
            for f in "abc":
                j1 = self.idx.get(f"finger_{f}_joint_1_1")
                if j1 is None:
                    continue
                g = self._finger_surface_gap(f, oc, r, self.obj_half_h)
                before[f] = round(g * 1000, 1)
                back = 0.0
                if g < need:
                    back = min(cap, (need - g) / max(slope, 1e-6))
                    back = min(back, max(0.0, float(q[j1]) - j1_widest))   # stop at the widest pose
                moved[f] = round(back, 3)
            # Open to ONE pose, not per-finger. Sizing each finger's retraction from its own gap
            # equalises GAPS, not ANGLES, which hands the close an asymmetric hand -- the two side
            # fingers end at visibly different openings.
            if os.environ.get("PRE_OPEN_SYM", "1") == "1" and moved:
                _mx = max(moved.values())
                for f in list(moved):
                    j1 = self.idx.get(f"finger_{f}_joint_1_1")
                    if j1 is not None:
                        moved[f] = round(min(_mx, max(0.0, float(q[j1]) - j1_widest)), 3)
            for f in list(moved):
                j1 = self.idx.get(f"finger_{f}_joint_1_1")
                if j1 is not None:
                    q[j1] -= moved[f]                                # negative j1 = opening
            # Settle it. The retraction has to physically take effect before the close runs, because
            # the contact stop measures the ACTUAL articulation transform -- writing q alone leaves
            # the thumb still reading as touching at step 0, so it freezes again.
            self._finger_cmd = q[self._f_idx].copy()
            # Pin the base, as every other stepping loop does: without it the chassis drifts during
            # the settle and a finger's measured gap moves while the finger does not.
            bx_, by_, byaw_ = self.base_ledger()
            for _ in range(max(1, int(0.4 / self.dt))):
                self.set_base(bx_, by_, byaw_)
                self._apply(q)
                self._force(q)
                self.world.step(render=not HEADLESS)
            after = {f: round(self._finger_surface_gap(f, oc, r, self.obj_half_h) * 1000, 1)
                     for f in "abc"}
            print(f">>> pre-close open: gaps(mm) {before} -> retract j1 by {moved} rad "
                  f"-> gaps(mm) {after}", flush=True)
        except Exception as e:
            # never let the pre-open kill a cycle — worst case we close from the approach pose,
            # which is the old behaviour
            print(f">>> pre-close open skipped ({e})", flush=True)
        return q

    def _pick_palm_spread(self, q):
        """SCISSOR the two side fingers apart before closing, and measure which way is 'apart'.

        _grasp_known sets palm_finger_b and palm_finger_c to the SAME value (+0.161).  On this
        hand those two joints are the scissor pair and must MIRROR: with the same sign both fingers
        swing the same way and end up 8mm apart (measured), acting as a single jaw 192mm from the
        thumb.  The opposing thumb-vs-pair pinch that this gripper grasps with then cannot form —
        an empirical 78-placement sweep held the object exactly 0 times.
        The sign convention is not documented anywhere we can trust, so try both and keep whichever
        actually separates the tips."""
        s = 0.0       # 0 = keep the baked spread
        ib = self.idx.get("palm_finger_b_joint_1")
        ic = self.idx.get("palm_finger_c_joint_1")
        if ib is None or ic is None or s <= 0.0:
            return q
        # PIN the base: without this the robot rolled ~0.6m during the measurement and every tip
        # reading came out garbage
        bx_s, by_s, byaw_s = self.base_ledger()
        best, best_sep = None, -1.0
        for sign in (+1.0, -1.0):
            qt = q.copy()
            qt[ib], qt[ic] = -sign * s, sign * s
            self._finger_cmd = qt[self._f_idx].copy()
            for _ in range(int(0.25 / self.dt)):
                self.set_base(bx_s, by_s, byaw_s)
                self._force(qt)
                self._apply(qt)
                self.world.step(render=not HEADLESS)
            t = {f: np.asarray(p.get_world_pose()[0], float)
                 for f, p in self._tip_prims().items()}
            sep = float(np.linalg.norm(t["b"][:2] - t["c"][:2]))
            print(f">>> palm spread sign {sign:+.0f}: b-c tip separation {sep * 1000:.0f}mm",
                  flush=True)
            if sep > best_sep:
                best, best_sep = sign, sep
        q[ib], q[ic] = -best * s, best * s
        self._finger_cmd = q[self._f_idx].copy()
        for _ in range(int(0.25 / self.dt)):
            self.set_base(bx_s, by_s, byaw_s)
            self._force(q)
            self._apply(q)
            self.world.step(render=not HEADLESS)
        print(f">>> palm spread: sign {best:+.0f} -> b-c {best_sep * 1000:.0f}mm apart", flush=True)
        return q

