"""The reach: ONE planned motion from park to the descent's end, then the seat."""
import os

import numpy as np

from morph.arm.api import ArmModel
from morph.config import COLUMN_MAX, A1_MAX, CHASSIS_PLATE_BOX


DECK_PROBE_CAP_M = 0.30          # bisection ceiling for the boom-vs-deck handoff reading


def _deck(clear):
    """The boom-deck field: inside, past the ceiling, or a measured number."""
    if clear < 0.0:
        return "OVERLAPPING"
    if clear >= DECK_PROBE_CAP_M - 1e-4:
        return ">=%.0fmm" % (DECK_PROBE_CAP_M * 1000)
    return "%.1fmm" % (clear * 1000)


class ReachStage:
    """One stage of `pick`. Mixed into `PickMixin`; all `self.*` belong to `Demo`."""

    def _descent_handoff(self, obj, tag):
        """The state the close stack actually reads, printed at the descent boundary.

        `_reach_lowest` consumes the entry a1/dh, the guarded seat needs COLUMN-TRAVEL HEADROOM and
        respects a floor, and force balance inherits any standing joint error -- so hand pose alone
        does not describe this handoff. A migration that lands the hand identically on a different
        IK branch changes every number below.
        """
        try:
            q = np.asarray(self.robot.get_joint_positions(), float)
            ia1 = self.idx.get("ArmLeftJoint_1")
            h1i, h2i = self.idx["ColumnLeftBearingJoint_1"], self.idx["ColumnRightBearingJoint_1"]
            a1 = float(q[ia1]) if ia1 is not None else float("nan")
            h1, h2 = float(q[h1i]), float(q[h2i])
            pin = self._pinch()
            # THREE distinct outcomes, because a saturating field reads the same in health and in failure:
            # OVERLAPPING, at the cap (far, or the wrong pair), or a bisected number.
            clear, cap = float("nan"), DECK_PROBE_CAP_M
            m = self._arm_model()
            q8 = self._arm_q8()
            plate = (CHASSIS_PLATE_BOX, "chassis")
            if m.first_hit(q8, [plate], 0.0, None, None, links=("Arm_Left_1",)) is not None:
                clear = -1.0
            else:
                lo_b, hi_b = 0.0, cap
                for _ in range(16):
                    mid = 0.5 * (lo_b + hi_b)
                    if m.first_hit(q8, [plate], mid, None, None,
                                   links=("Arm_Left_1",)) is not None:
                        hi_b = mid
                    else:
                        lo_b = mid
                clear = lo_b
            # Stashed for the raise probe: NOT a translation of the seat pose -- it sits higher, further
            # back and at a different a1 -- so probing offsets of the seat never visits it.
            self._handoff_q8 = {**getattr(self, "_handoff_q8", {}), tag: np.array(q8, float)}
            fc = self._finger_cmd
            print(f">>> handoff[{tag}]: a1 {a1:.4f} (head {A1_MAX - a1:+.4f}) "
                  f"h1 {h1:.4f} h2 {h2:.4f} dh {h2 - h1:+.4f} "
                  f"(col head {COLUMN_MAX - max(h1, h2):+.4f}) "
                  f"pinch [{pin[0]:.4f} {pin[1]:.4f} {pin[2]:.4f}] "
                  f"boom-deck {_deck(clear)} "
                  f"fingers {'cmd' if fc is not None else 'free'}",
                  flush=True)
        except Exception as _e:                        # noqa: BLE001 -- a probe must not stop a pick
            print(f">>> handoff[{tag}]: unavailable ({_e})", flush=True)

    def _reach_abort(self, st):
        """End the pick attempt: `st.alive = False` is what `pick()` returns False on, and that
        False is what `run_cycle` recovers and retries. Shared by both `_plan_approach` call sites
        because the counter's registry forbids two `_fallback` calls on one tag."""
        self._fallback("reach-abort")
        print(f">>> reach: approach ABORTED at a checkpoint -- NOT replaying the bake (it ends at "
              f"the goal the check rejected); pick obj {st.obj_idx} ends here", flush=True)
        st.alive = False

    def _pick_reach(self, st):
        self._stage = "reach"
        obj_idx, status = st.obj_idx, st.status
        dock, qg, obj, park = st.dock, st.qg, st.obj, st.park
        pin_bystanders = st.pin_bystanders
        # Bystander protection is the corridor gate, not per-pick collider toggling -- toggling
        # invalidates the articulation view in headed mode.
        tr = self.traj["reach"]
        n_fr = len(tr["frames"])
        # Base-shift align only: an arm-servo align invalidates the baked descent geometry.
        t_total = float(os.environ.get("REACH_T", "2.5"))
        _t_appr = t_total * (n_fr - 30) / n_fr
        # Stages 1..6b are one planned motion: the 6b pose is the approach goal with the
        # columns re-solved and a1 folded (7753e6b, docs/FINDINGS.md).
        h1i = self.idx["ColumnLeftBearingJoint_1"]
        h2i = self.idx["ColumnRightBearingJoint_1"]
        _a1_d = self.idx.get("ArmLeftJoint_1")
        # Pinned before the motion: the arm absorbs the dock's error, the base does not move.
        if os.environ.get("PIN_CHASSIS", "1") == "1":
            self.pin_chassis(True, "reach")
        tbx, tby, tbyaw = self.base_ledger()
        dock = np.array([tbx, tby])   # `grasp` gates `held` on base-vs-dock drift; keep it
        # Read ONCE, before anything moves, and used for both the goal and the reports.
        # The old order read the object again after the hover, 2.5s and one flight later.
        _oc0 = np.asarray(obj.get_world_poses()[0][0], float)
        _qd0 = np.asarray(self.robot.get_joint_positions(), float).copy()
        _h1_0, _h2_0 = float(_qd0[h1i]), float(_qd0[h2i])
        h1_0, p0 = _h1_0, self._pinch()[:2].copy()
        # The object's CENTRE, not the seat band: aiming higher leaves the pinch above the
        # object's mid-plane with the columns already floored.
        _z_tgt, _pz_d = float(_oc0[2]), float(self._pinch()[2])
        _h1_t = _h2_t = None
        n_d = int(3.2 / self.dt)
        if self._finite("reach-plan"):
            if status:
                status(f"planning object {obj_idx}: park -> descent end")
            print(f">>> descent: PLANNED (solve at park, one move_to_pose), object top "
                  f"{_oc0[2] + self.obj_half_h:.3f}", flush=True)
            # The body-box filter needs the SWUNG turret: at th 0 the boom is not over the
            # battery at all.
            _th_end = 0.0
            try:
                _tj0 = tr["names"]
                if "BaseJoint_1" in _tj0:
                    _th_end = float(tr["frames"][-1][_tj0.index("BaseJoint_1")])
            except Exception:
                pass
            _dh_end = float(os.environ.get("LOWEST_DH_TGT", "0.13"))
            # Same margin as the mid-move path guard, or the guard condemns this pose.
            _a1_cl = self._a1_clearing_chassis(_dh_end, margin=0.05, th=_th_end)
            _a1_floor = None if _a1_cl is None else _a1_cl + 0.015
            # Under `OBJ_GOAL=2` the wrist is part of the planner's aim and must not be
            # overwritten; these five joints belong in the GOAL, not written mid-flight.
            _wr = []
            if os.environ.get("OBJ_GOAL", "0") != "2":
                try:
                    _tj = tr["names"]
                    for _n in ("BaseJoint_1", "HandBearingJoint_1", "gripper_z_rotation_1",
                               "gripper_y_rotation_1", "gripper_x_rotation_1"):
                        if _n in _tj and _n in ArmModel.Q8:
                            _wr.append((ArmModel.Q8.index(_n),
                                        float(tr["frames"][-1][_tj.index(_n)])))
                except Exception as _e_wr:
                    print(f">>> descent: wrist target unavailable ({_e_wr})", flush=True)
            _solved = {}

            def _to_6b(g):
                """The approach goal lowered and folded to the descent's END pose; applied to a
                re-derived goal too, so a replan aims at the same stage."""
                g = np.asarray(g, float).copy()
                for _qi, _qv in _wr:
                    g[_qi] = _qv
                # `min`, because this stage may only FOLD: it never extends the boom, so a
                # floor above where the goal already sits cannot be honoured.
                _fold = 0.24
                if _a1_floor is not None and _a1_floor > _fold:
                    _fold = _a1_floor
                _a1_t = min(_fold, float(g[3]))
                _s = self._solve_lowest(
                    _z_tgt, a1_max=A1_MAX,
                    # The pose the solve predicts the pinch AT: this turret and THIS WRIST. The lever swings
                    # tens of millimetres across the wrist, so the wrist is part of the answer.
                    q8_t=g, fingers=self._commanded_fingers(),
                    base=self._arm_base_world(),
                    dh_want=_dh_end or None,
                    # No h1 floor: it buys height with tilt the seat then clips.
                    h1_min=0.0, th=_th_end)
                _solved["sol"], _solved["a1"], _solved["floor"] = _s, _a1_t, _a1_floor
                if _s is None:
                    # no solve -> fly the approach pose alone
                    return g
                g[1], g[2] = float(_s[0]), float(_s[1])
                if _a1_d is not None:
                    g[3] = _a1_t
                return g

            # Via `_plan_approach`, where the reach's degrades are tagged for the gate.
            _secs = max(1.0, _t_appr + n_d * self.dt)
            _appr = self._plan_approach(tr, n_fr - 30, _secs, exclude_obj=obj_idx,
                                        on_step=pin_bystanders, tag="reach",
                                        goal_fn=_to_6b)
            if _appr is None:
                # abort != no-plan: the bake ends AT the
                self._reach_abort(st)
                # rejected goal, so it is not a fallback
                return
            _sol_d = _solved.get("sol")
            if _sol_d is not None:
                _h1_t, _h2_t = float(_sol_d[0]), float(_sol_d[1])
                print(f">>> descent(planned): pinch {_pz_d:.3f} -> object {_z_tgt:.3f} "
                      f"({(_pz_d - _z_tgt) * 1000:+.0f}mm). SOLVED h1 {_h1_t:.3f} h2 "
                      f"{_h2_t:.3f} (dh {_h2_t - _h1_t:+.3f}) a1 {_sol_d[2]:.3f} -> reach "
                      f"{_sol_d[3]:.3f}m; flown at a1 {_solved['a1']:.3f}"
                      + ("" if _a1_floor is None else
                         f" (chassis floor {_a1_floor:.3f} at th {_th_end:+.2f}, "
                         + ("held" if _solved["a1"] >= _a1_floor - 1e-9
                            else "above the goal's own a1, and this stage only folds)")),
                      flush=True)
            else:
                print(f">>> descent(planned): NO SOLVE for object z {_z_tgt:.3f} -- flew "
                      f"the approach pose alone; the seat below closes the gap", flush=True)
            # The same solve from the arrived pose: the goal has 3mm of deck clearance to spend.
            try:
                _sv = self._solve_lowest(
                    _z_tgt, a1_max=A1_MAX, q8_t=self._arm_q8(),
                    fingers=self._commanded_fingers(), base=self._arm_base_world(),
                    dh_want=_dh_end or None, h1_min=0.0, th=_th_end)
                print(">>> park-solve: "
                      + ("not comparable (one side had no solve)"
                         if _sv is None or _sol_d is None else
                         f"arrived-minus-park  h1 {(_sv[0] - _sol_d[0]) * 1000:+.2f}mm  "
                         f"h2 {(_sv[1] - _sol_d[1]) * 1000:+.2f}mm  reach "
                         f"{(_sv[3] - _sol_d[3]) * 1000:+.1f}mm -- "
                         + ("inside the 3mm the pad leaves"
                            if abs(_sv[0] - _sol_d[0]) < 0.003
                            else "OUTSIDE the 3mm the pad leaves")), flush=True)
            except Exception as _e_pv:       # noqa: BLE001 -- a probe must not stop a pick
                print(f">>> park-solve: unavailable ({_e_pv})", flush=True)
            _qf = np.asarray(self.robot.get_joint_positions(), float)
            print(f">>> descent(planned): arrived, pinch z "
                  f"{float(self._pinch()[2]):.3f} (target {_z_tgt:.3f}), dh "
                  f"{float(_qf[h2i] - _qf[h1i]):+.3f}", flush=True)
            # The column ramp buys only the mean height; tilt and extend finish it.
            self._descent_handoff(obj, "pre-reach-lowest")
            try:
                self._reach_lowest(obj, label="descent/reach-lowest")
            except Exception as _e_drl:
                print(f">>> descent/reach-lowest: failed ({_e_drl})", flush=True)
            self._descent_handoff(obj, "post-reach-lowest")
            _qf = np.asarray(self.robot.get_joint_positions(), float)
            # `_a1_d`, not a direct lookup: every other site treats the joint as optional (`self.idx.get`),
            # and a bare subscript would make the SUMMARY the one line that raises on a model without it.
            print(f">>> descent(planned): done, pinch z {float(self._pinch()[2]):.3f} "
                  f"(target {_z_tgt:.3f}), dh {float(_qf[h2i] - _qf[h1i]):+.3f}, "
                  f"a1 {float(_qf[_a1_d]) if _a1_d is not None else float('nan'):.3f}",
                  flush=True)

            qd = np.asarray(self.robot.get_joint_positions(), float).copy()
            self._ctc = {}                           # who touches the object during descent
            print(f">>> descent: {-(float(qd[h1i]) - h1_0) * 1000:.0f}mm descended "
                  f"(baked 300mm), pinch xy drift {np.linalg.norm(self._pinch()[:2] - p0) * 1000:.1f}mm",
                  flush=True)
        self._finite("reach-replay")
        # Settle target = the pose the aligned descent reached. The baked end config would
        # teleport the hand back onto the pre-align path, into the object.
        qg = np.asarray(self.robot.get_joint_positions(), float).copy()
        st.qg, st.dock, st._oc0 = qg, dock, _oc0

