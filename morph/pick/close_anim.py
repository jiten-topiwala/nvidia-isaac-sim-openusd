"""The driven finger close — the stage that actually grips in friction mode.

Part of the pick cycle — see `morph/pick/__init__.py` for the stage order and the shared
state object. Imports Isaac APIs at module level; only importable after SimulationApp.
"""
import math
import os

import numpy as np

from morph.config import PERFECT, HEADLESS, KNOWN


class CloseAnimStage:
    """One stage of `pick`. Mixed into `PickMixin`; all `self.*` belong to `Demo`."""

    def _pick_close(self, st):
        obj_idx, status = st.obj_idx, st.status
        obj, park, replay = st.obj, st.park, st.replay
        close_anim, alive, grip = st.close_anim, st.alive, st.grip
        pin_bystanders = st.pin_bystanders

        # Visible finger close: the curl profile with a geometric per-finger surface stop, so the
        # fingertips land ON the object rather than inside it.
        if close_anim and alive:
            if status:
                status(f"closing gripper on object {obj_idx}")
            q_now = np.asarray(self.robot.get_joint_positions(), float).copy()
            # Practical path only. In perfect mode the reach is baked to the recorded close-END arm
            # pose with the recorded close-START fingers, so the hand already arrives at the
            # validated grasp relation, where one finger deliberately sits just off the surface as
            # an anvil.
            if os.environ.get("PRE_OPEN", "0" if PERFECT else "1") == "1":
                q_now = self._pre_close_open(q_now, obj)
            # Telemetry is not mode-specific: inside the PERFECT branch, a practical run reports
            # only the phase AFTER the hold has already frozen whatever seat it was given.
            self._phase_report("close-entry(pre)", obj)
            if PERFECT:
                fj = {fl: round(float(q_now[self.idx[f"finger_{fl}_joint_1_1"]]), 3)
                      for fl in "abc" if f"finger_{fl}_joint_1_1" in self.idx}
                print(f">>> close-entry finger j1: {fj}", flush=True)
                if self._finger_cmd is None:
                    self._finger_cmd = q_now[self._f_idx].copy()
                q_now = self._pick_palm_spread(q_now)     # splay b/c into a real tripod first
                self._tip_report("close-entry", obj)
                self._phase_report("close-entry", obj)
                self._ctc = {}
                gaps = self._tip_gaps(obj)
                g_max = max(gaps.values())
                rate = 0.030   # m/s, MuJoCo's
                close_delays = {f: min((g_max - g) / rate, 1.5) for f, g in gaps.items()}
                print(f">>> close stagger: gaps(mm) "
                      f"{ {f: round(g * 1000) for f, g in gaps.items()} } -> delays(s) "
                      f"{ {f: round(d, 2) for f, d in close_delays.items()} }", flush=True)
            # The grasp starts here, so the chassis-stationary rule starts here too, not at the
            # close pin several stages later.
            if (PERFECT and os.environ.get("NO_CHASSIS_GRASP", "1") == "1"
                    and getattr(self, "_grasp_locked", None) is None):
                self._grasp_locked = self.base_ledger()
                self._base_block_n = {}
                print(f">>> chassis LOCKED at grasp entry "
                      f"({self._grasp_locked[0]:.3f}, {self._grasp_locked[1]:.3f}) -- "
                      f"every remaining gap is closed with the ARM", flush=True)
            # Not under the computed descent. These aligns chase the RECORDED relation, which
            # belongs to the baked pipeline, while the computed reach chain owns the approach --
            # running both makes the arm ping-pong and leaves the pinch further from the object than
            # it started.
            _computed_d = os.environ.get("DESCENT_MODE", "baked") == "computed"
            if PERFECT and _computed_d and os.environ.get("RELATION_ALIGN", "1") == "1":
                print(">>> post-descent aligns SKIPPED (computed descent owns the approach; "
                      "RELATION_ALIGN targets the baked relation)", flush=True)
            if (PERFECT and not _computed_d
                    and os.environ.get("RELATION_ALIGN", "1") == "1"):
                # Second align, at GRASP HEIGHT. The hover align lands the world-xy relation to
                # about a millimetre, but the descent does not preserve it: columns-only is a pure
                # vertical move in JOINT space, and the four-bar swings the hand through an arc.
                _tch = self._hand_touching_object(obj)
                if _tch:
                    print(f">>> post-descent align SKIPPED: {_tch} is already in contact -- "
                          f"a base move here drags the object, not the hand", flush=True)
                else:
                    self._align_recorded_relation(obj)
                # Height, then re-null XY: the align above is world-XY only, so without this the
                # hand can arrive with a large gripper-y residual while reporting a sub-millimetre
                # one.
                if abs(self._align_height(obj, tag="grasp-align")) > 0.002:
                    if not self._hand_touching_object(obj):
                        self._align_recorded_relation(obj)
                # Drop to the side wall. At the end of the descent the outer fingers are not out of
                # REACH, they are ABOVE THE RIM -- which a scalar gap hides completely, and which no
                # dock offset or finger closure can fix: driving them further just arcs them over
                # the top.
                _tw = self._hand_touching_object(obj)
                if os.environ.get("DROP_TO_WALL", "1") == "1" and _tw:
                    print(f">>> drop-to-wall SKIPPED: {_tw} is in contact -- the hand is already "
                          f"down on the object, lowering further only presses it", flush=True)
                elif os.environ.get("DROP_TO_WALL", "1") == "1":
                    oc_ = np.asarray(obj.get_world_poses()[0][0], float)
                    r_ = KNOWN["object_radius"]
                    dz_ = max(self._finger_gap_parts(fl_, oc_, r_, self.obj_half_h)[1][1]
                              for fl_ in "abc")
                    drop = float(np.clip(dz_ + 0.010,
                                         0.0, 0.15))
                    if drop > 0.001:
                        cl_, cr_ = self.idx["ColumnLeftBearingJoint_1"], self.idx["ColumnRightBearingJoint_1"]
                        qd2 = np.asarray(self.robot.get_joint_positions(), float).copy()
                        c0 = np.array([qd2[cl_], qd2[cr_]])
                        bx2, by2, byaw2 = self.base_ledger()
                        n2 = int(0.8 / self.dt)
                        for i2 in range(n2):
                            f2 = 0.5 - 0.5 * math.cos(math.pi * (i2 + 1) / n2)
                            qd2[cl_], qd2[cr_] = (c0 - f2 * drop).tolist()
                            self.set_base(bx2, by2, byaw2)
                            self._force(qd2)
                            self._apply(qd2)
                            self.world.step(render=not HEADLESS)
                        print(f">>> drop-to-wall: worst dz {dz_ * 1000:.0f}mm -> lowered columns "
                              f"{drop * 1000:.0f}mm", flush=True)
                        if not self._hand_touching_object(obj):
                            self._align_recorded_relation(obj)   # re-null xy after the drop
                # Dead end, off by default. The centroid of the fingers' CLOSEST POINTS is not a
                # fixed target -- those points are by definition near the object, so they move with
                # the hand and the solve chases itself.
                if os.environ.get("CENTROID_ALIGN", "0") == "1":
                    # Final, size-adaptive centring: put the object's axis at the centroid of the
                    # three fingers so they straddle it. The recorded offset cannot do that for an
                    # object of a different radius.
                    for it2 in range(6):
                        oc3 = np.asarray(obj.get_world_poses()[0][0], float)
                        cen = self._finger_centroid_xy(oc3)
                        if cen is None:
                            break
                        e3 = cen - oc3[:2]                    # move the HAND so the centroid lands
                        if np.linalg.norm(e3) < 0.002:        # on the object axis -> base by -e3
                            break
                        bx3, by3, byaw3 = self.base_ledger()
                        q3 = np.asarray(self.robot.get_joint_positions(), float).copy()
                        n3 = int(0.4 / self.dt)
                        for i3 in range(n3):
                            f3 = 0.5 - 0.5 * math.cos(math.pi * (i3 + 1) / n3)
                            self.set_base(float(bx3 - f3 * e3[0]), float(by3 - f3 * e3[1]), byaw3)
                            self._force(q3)
                            self._apply(q3)
                            self.world.step(render=not HEADLESS)
                        print(f">>> centroid-align[{it2}] err {np.linalg.norm(e3) * 1000:.1f}mm",
                              flush=True)
                self._phase_report("post-align", obj)
                q_now = np.asarray(self.robot.get_joint_positions(), float).copy()
            if PERFECT and self.close_traj is not None:
                # Replay the recorded grasp instead of synthesising a curl.
                grip = self._close_replay(q_now, float(os.environ.get("CLOSE_T", "4.0")), obj,
                                          status=status)
                alive = self._finite("close-replay")
                self._tip_report("close-end", obj)
                self._phase_report("close-end", obj)
                print(f">>> close contacts: "
                      f"{dict(sorted(self._ctc.items(), key=lambda kv: -kv[1])[:8])}", flush=True)
                # ...and how much load each carried: a link with thousands of contact events and no
                # impulse is resting on the object, not gripping it.
                _cn = {k2: round(v2, 2) for k2, v2 in
                       sorted(getattr(self, "_ctcN", {}).items(), key=lambda kv: -kv[1])[:8]}
                print(f">>> close contact impulse (Ns per link): {_cn}", flush=True)
            else:
              grip = self._animate_fingers(q_now, self._finger_curl_targets(),
                                         float(os.environ.get("CLOSE_T", "3.5" if PERFECT else "1.5")),
                                         # a longer window in perfect mode -- the measured stagger
                                         # can hold the nearest finger back for about a second, and
                                         # it still needs time to travel afterwards
                                         pin=(obj, park), on_step=pin_bystanders,
                                         stagger=float(os.environ.get(
                                             "CLOSE_STAGGER", "0.0" if PERFECT else "0.15")),
            # Close all three SIMULTANEOUSLY. Staggering the thumb last was written for a pinned
            # object that could not run away; a free one does, and the other two reach it early and
            # shove it toward a thumb that is not there yet.
                                         stop_obj=(obj, KNOWN["object_radius"]),
                                         delays=close_delays if PERFECT else None)
              alive = self._finite("close-anim")
        st.alive, st.grip = alive, grip

