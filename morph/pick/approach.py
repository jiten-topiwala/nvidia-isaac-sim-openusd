"""The pick's approach: the goal derived from the baked frame, and the planned move to it."""
import math
import os

import numpy as np

from morph.arm.api import MARGIN, ArmModel, Outcome, goal_ik, move_to_pose
from morph.config import ARM_PLANNER, KNOWN, MOUTH_VEC_HAND, local_obj


class ApproachStage:
    """Mixed into `Demo`. Every `self.*` it touches is owned by `Demo`."""

    @staticmethod
    def _traj_q8(tr, frame_i):
        """The eight actuated joints at frame `frame_i` of a baked trajectory. Frames are indexed by
        COLUMN, not by articulation index, so `tr["names"]` maps a joint to its slot."""
        col = {nm: k for k, nm in enumerate(tr["names"])}
        fr = tr["frames"][frame_i]
        return np.array([float(fr[col[n]]) for n in ArmModel.Q8])

    def _object_relative_goal(self, q8_baked, obj_idx):
        """OBJ_GOAL=1: keep the RECORDED hand->object relation to the LIVE object -- hand pose at the
        baked frame (FK, ARM-ROOT frame) shifted by (live object - recorded object) in WORLD, same
        orientation, re-solved by IK. OBJ_GOAL=2 builds explicit pre-grasp geometry; anything else,
        `obj_idx` None, or any miss returns the baked frame's goal, which is still planned to."""
        _mode = os.environ.get("OBJ_GOAL", "0")
        if _mode not in ("1", "2") or obj_idx is None:
            return q8_baked
        try:
            m = self._arm_model()
            hand = m.m["hand_link"]
            p_b, R_b = m.fk(q8_baked)[hand]
            bp, bR = self._arm_base_world()
            o_w = np.asarray(self.objs[obj_idx].get_world_poses()[0][0], float)
            # The live hand pose is an INPUT of mode 2, not part of the self-check below it.
            hw_live, hq_live = (np.asarray(v, float) for v in self._hand_prims()[hand].get_world_pose())
            # Frame self-check: FK of the CURRENT q8 through this transform must land on the hand.
            try:
                p_now, _ = m.fk(self._arm_q8())[hand]
                hw_fk = bp + bR @ p_now
                print(f">>> obj-goal: frame check |FK(hand) - live hand| = "
                      f"{float(np.linalg.norm(hw_fk - hw_live)) * 1000:.1f} mm", flush=True)
            except Exception as _e_fc:
                print(f">>> obj-goal: frame check unavailable ({_e_fc})", flush=True)
            if _mode == "2":
                # OBJ_GOAL=2: pinch `OBJ_GOAL_STANDOFF` m behind the live object's surface along the
                # mouth axis, orientation from the baked frame. Size-dynamic.
                r_obj = float(os.environ.get("OBJ_RADIUS", KNOWN["object_radius"]))
                so = float(os.environ.get("OBJ_GOAL_STANDOFF", "0.060"))
                R_w = bR @ R_b                                       # baked hand orientation, world
                # object -> hand, NOT the other way: the standoff below sits at `obj + (r+so)*mouth`, on the
                # palm side. Stepping along -mouth drives the hand into the object and then the floor.
                mouth = R_w @ np.asarray(MOUTH_VEC_HAND, float)
                mouth[2] = 0.0
                # near-vertical mouth: no plane axis
                if float(np.linalg.norm(mouth)) < 0.5:
                    self._fallback("mouth-vertical")
                    print(">>> obj-goal[2]: mouth axis near vertical -> baked goal", flush=True)
                    return q8_baked
                mouth /= float(np.linalg.norm(mouth))
                # pinch offset in the hand frame, from the LIVE prims (FK has no pinch point)
                w_, x_, y_, z_ = [float(v) for v in hq_live]
                hR = np.array([[1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - z_ * w_), 2 * (x_ * z_ + y_ * w_)],
                               [2 * (x_ * y_ + z_ * w_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - x_ * w_)],
                               [2 * (x_ * z_ - y_ * w_), 2 * (y_ * z_ + x_ * w_), 1 - 2 * (x_ * x_ + y_ * y_)]])
                pinch_off_h = hR.T @ (np.asarray(self._pinch(), float) - hw_live)
                # z: the baked HOVER height -- at the object's centre the boom hits the chassis.
                hand_b_w = bp + bR @ p_b
                pinch_b_w = hand_b_w + R_w @ pinch_off_h
                # SIGN: MOUTH_VEC_HAND points OUT of the mouth, object -> palm: obj + (r+so)*mouth.
                # The opposite sign put the pinch ~100mm BEYOND the object and shoved it 40-90mm.
                pinch_des = o_w + (r_obj + so) * mouth
                pinch_des[2] = float(pinch_b_w[2]) + float(os.environ.get("OBJ_GOAL_DZ", "0.0"))
                try:
                    rx, ry = float(self.base_pose()[0]), float(self.base_pose()[1])
                    _v_po = pinch_des[:2] - o_w[:2]; _v_bo = np.array([rx, ry]) - o_w[:2]
                    _side = math.degrees(math.acos(float(np.clip(_v_po @ _v_bo /
                            max(1e-9, np.linalg.norm(_v_po) * np.linalg.norm(_v_bo)), -1, 1))))
                    print(f">>> obj-goal[2]: pinch is {_side:.0f} deg from the base side of the object "
                          f"(0 = between base and object, 180 = beyond it)", flush=True)
                except Exception:
                    pass
                hand_des_w = pinch_des - R_w @ pinch_off_h
                p_des = bR.T @ (hand_des_w - bp)
                lo, hi = self._arm_bounds()
                sol = goal_ik(self, m, p_des, R_b, q8_baked, hand, (lo, hi))
                if sol is None:
                    self._fallback("goal-ik-miss")
                    print(f">>> obj-goal[2]: IK miss for pinch {np.round(pinch_des, 3).tolist()} -> baked goal", flush=True)
                    return q8_baked
                q8, ep, er = sol
                # OBJ_GOAL_AIM=1: yaw the mouth onto the a1 advance direction, then re-solve.
                if os.environ.get("OBJ_GOAL_AIM", "1") == "1":
                    # AIM_ITERS DEFAULT 1: iterating converges the mouth onto the boom, but the goal
                    # then approaches from a side whose path sweeps THROUGH the excluded object.
                    _aim_iters = int(os.environ.get("AIM_ITERS", "1"))
                    _aim_tol = math.radians(float(os.environ.get("AIM_TOL_DEG", "2.0")))
                    _ibase = list(ArmModel.Q8).index("BaseJoint_1")
                    _th0_aim = float(np.asarray(q8, float)[_ibase])
                    _aim_n, _aim_last = 0, 0.0
                    for _ in range(_aim_iters):
                        try:
                            _ia1 = list(ArmModel.Q8).index("ArmLeftJoint_1")
                            _qa = np.asarray(q8, float).copy(); _qa[_ia1] += 1e-3
                            _adv = bR @ (m.fk(_qa)[hand][0] - m.fk(q8)[hand][0])
                            _adv[2] = 0.0
                            _adv /= max(1e-9, float(np.linalg.norm(_adv)))
                            # Seat advances along whichever of +-a1 nears the object; mouth opposes it.
                            _tow = o_w[:2] - pinch_des[:2]
                            if float(_adv[:2] @ _tow) < 0.0:
                                _adv = -_adv
                            _tgt = -_adv                             # where the mouth vector should point
                            _ang = math.atan2(_tgt[0] * mouth[1] - _tgt[1] * mouth[0],
                                              _tgt[0] * mouth[0] + _tgt[1] * mouth[1])   # mouth -> tgt
                            _ang = -_ang
                            # AIM_GAIN goes WITH AIM_ITERS: the error reverses at full magnitude each
                            # pass, so iterating needs the half step; one pass needs gain 1.
                            _ang *= float(os.environ.get("AIM_GAIN", "0.5" if _aim_iters > 1 else "1.0"))
                            _c, _s = math.cos(_ang), math.sin(_ang)
                            _Rz = np.array([[_c, -_s, 0.0], [_s, _c, 0.0], [0.0, 0.0, 1.0]])
                            _R_des_w = _Rz @ R_w
                            _mouth2 = _Rz @ mouth
                            _pinch2 = o_w + (r_obj + so) * _mouth2
                            _pinch2[2] = pinch_des[2]
                            _hand2 = _pinch2 - _R_des_w @ pinch_off_h
                            _pos2, _rot2 = bR.T @ (_hand2 - bp), bR.T @ _R_des_w
                            # THE TURRET AIMS, THE WRIST TAKES THE RESIDUAL: seed with the turret
                            # pre-rotated by the aim angle, and keep the wrist FREE.
                            _seed2 = np.asarray(q8, float).copy()
                            if os.environ.get("AIM_TURRET_FIRST", "1") == "1":
                                _lo_b, _hi_b = float(lo[_ibase]), float(hi[_ibase])
                                _seed2[_ibase] = float(np.clip(_seed2[_ibase] + _ang, _lo_b, _hi_b))
                            # Collision-repaired, like the unaimed goal above: a plain `m.ik` here
                            # discarded the repair on every pick, since AIM is on by default.
                            _sol2 = goal_ik(self, m, _pos2, _rot2, _seed2, hand, (lo, hi))
                            if _sol2 is None:
                                _sol2 = goal_ik(self, m, _pos2, _rot2, q8, hand, (lo, hi))
                                if _sol2 is not None:
                                    print(">>> obj-goal[2]: AIM turret-first seed missed -> unaimed seed", flush=True)
                            if _sol2 is not None:
                                _th_before = float(q8[_ibase])
                                q8, ep, er = _sol2
                                pinch_des, mouth = _pinch2, _mouth2
                                _d_th = math.degrees(float(q8[_ibase]) - _th_before)
                                print(f">>> obj-goal[2]: AIM -- mouth rotated {math.degrees(_ang):+.1f} deg to the a1 "
                                      f"advance direction {np.round(_adv[:2], 2).tolist()} (IK err {ep * 1000:.2f} mm / "
                                      f"{er * 1000:.2f} mrad); turret took {_d_th:+.1f} deg", flush=True)
                            else:
                                # Now that the aimed solve is collision-repaired it returns None on an
                                # unrepairable goal, silently switching AIM off. Count that.
                                self._fallback("aim-ik-miss")
                                print(f">>> obj-goal[2]: AIM IK miss ({math.degrees(_ang):+.1f} deg) -> unaimed goal", flush=True)
                            _aim_n += 1
                            _aim_last = _ang
                            if abs(_ang) <= _aim_tol:
                                break
                        except Exception as _e_aim:
                            print(f">>> obj-goal[2]: AIM unavailable ({_e_aim})", flush=True)
                            break
                    if _aim_n:
                        print(f">>> obj-goal[2]: AIM {_aim_n} pass(es), final mouth-vs-boom "
                              f"{math.degrees(_aim_last):+.1f} deg (tol {math.degrees(_aim_tol):.0f} deg); "
                              f"TURRET carried {math.degrees(float(np.asarray(q8, float)[_ibase]) - _th0_aim):+.1f} deg", flush=True)
                print(f">>> obj-goal[2]: pinch target {np.round(pinch_des, 3).tolist()} = object "
                      f"{np.round(o_w, 3).tolist()} - {(r_obj + so) * 1000:.0f} mm along mouth "
                      f"{np.round(mouth, 2).tolist()} (IK err {ep * 1000:.2f} mm / {er * 1000:.2f} mrad; "
                      f"dq8 max {float(np.max(np.abs(q8 - q8_baked))):.3f})", flush=True)
                return np.asarray(q8, float)
            # FRAMES: `local_obj()` is in the chassis ROOT frame (base_pose()). The plane shift is
            # taken in WORLD, then rotated into the ARM-ROOT link frame for the IK; z is left alone.
            rx, ry, ryaw = self.base_pose()
            lx, ly = [float(v) for v in local_obj()[:2]]
            c, sn = math.cos(ryaw), math.sin(ryaw)
            o_rec_w = np.array([rx + c * lx - sn * ly, ry + sn * lx + c * ly, o_w[2]])
            d_w = o_w - o_rec_w
            d_w[2] = 0.0
            d = bR.T @ d_w
            if float(np.linalg.norm(d)) > 0.35:
                self._fallback("shift-too-large")
                print(f">>> obj-goal: shift {np.round(d * 1000).astype(int).tolist()} mm too large "
                      f"-> baked goal", flush=True)
                return q8_baked
            lo, hi = self._arm_bounds()
            sol = goal_ik(self, m, p_b + d, R_b, q8_baked, hand, (lo, hi))
            if sol is None:
                self._fallback("shift-ik-miss")
                print(f">>> obj-goal: IK miss for shift {np.round(d * 1000).astype(int).tolist()} mm "
                      f"-> baked goal", flush=True)
                return q8_baked
            q8, ep, er = sol
            print(f">>> obj-goal: hand goal shifted {np.round(d * 1000).astype(int).tolist()} mm to keep "
                  f"the recorded relation to the live object (IK err {ep * 1000:.2f} mm / "
                  f"{er * 1000:.2f} mrad; dq8 max {float(np.max(np.abs(q8 - q8_baked))):.3f})", flush=True)
            return np.asarray(q8, float)
        except Exception as _e:
            self._fallback("goal-exception")
            print(f">>> obj-goal: unavailable ({_e}) -> baked goal", flush=True)
            return q8_baked

    def _grasp_candidates(self, obj_idx):
        """The approach-yaw fan around the grasp goal that just FAILED: ONE q8 per yaw, ordered by
        |yaw| rather than by the fan's own score, which carries no yaw term.

        CALLED ONLY AFTER A FAILED PICK -- `ArmModel.grasp_candidates` runs `ik_candidates` once
        per yaw and is the most expensive thing in the pick. The nominal yaw is dropped: it
        reproduces the solve that just failed."""
        goal = getattr(self, "_grasp_goal", None)
        if goal is None:
            # No goal was ever planned (the planner is off, or the approach never got that far), so
            # there is nothing to orbit -- and no reason to pay for the sweep to find that out.
            print(">>> grasp-candidates: the approach planned no goal to orbit -- none offered",
                  flush=True)
            return []
        try:
            m = self._arm_model()
            pos, R = m.fk(np.asarray(goal, float))[m.hand_link]
            bp, bR = self._arm_base_world()
            o_w = np.asarray(self.objs[obj_idx].get_world_poses()[0][0], float)
            # EVERY argument in the frame `ik` works in -- `axis` especially, whose default is that
            # frame's +z. On a tilted base the fan would orbit the arm root's vertical, silently.
            obj = bR.T @ (o_w - bp)
            axis = bR.T @ np.array([0.0, 0.0, 1.0])
            lo, hi = self._arm_bounds()
            cands = m.grasp_candidates(
                pos, R, obj, obstacles=self._arm_obstacles(),
                q_now=self._arm_q8(), axis=axis, margin=MARGIN,
                held=None,                   # the PICK goal: nothing is carried yet
                base=(bp, bR), link=m.hand_link, bounds=(lo, hi))
            out, seen = [], set()
            for (q8, _ep, _er), _clr, _s, y in cands:
                yk = round(float(y), 9)
                if abs(yk) < 1e-9 or yk in seen:
                    continue
                seen.add(yk)
                out.append((abs(float(y)), float(y), np.asarray(q8, float)))
            # stable: equal |yaw| keeps the fan's score order
            out.sort(key=lambda e: e[0])
            print(f">>> grasp-candidates: {len(out)} approach yaw(s) offered from {len(cands)} "
                  f"candidate(s) -- "
                  f"{[round(math.degrees(y), 1) for _a, y, _q in out]} deg about the object's "
                  f"vertical, smallest orbit first (the nominal yaw is the grasp that just failed, "
                  f"so it is not offered)", flush=True)
            return [q8 for _a, _y, q8 in out]
        except Exception as _e:
            # Same contract as every other goal-side miss here: a counted degrade, never an
            # exception -- uncaught it escapes to `run_cycle`'s broad `except` and kills the cycle.
            self._fallback("grasp-candidates-unavailable")
            print(f">>> grasp-candidates: unavailable ({_e}) -- no candidates offered", flush=True)
            return []

    def _approach_goal(self, tr, frame_i, exclude_obj=None):
        """The reach's goal, derived WITHOUT moving the arm, plus the closure that re-derives it:
        `(goal_q8, rederive)`, where `rederive` answers None when the anchor is unavailable (a walk
        candidate is already solved; `_object_relative_goal` returns its argument on a miss)."""
        _baked = self._traj_q8(tr, frame_i)
        # A candidate from `pick`'s walk is ALREADY solved and filtered through the same `collides`, so it
        # IS the goal. Re-deriving one here throws the fan away and re-commands the approach that failed.
        _cand = getattr(self, "_grasp_cand", None)
        goal = (self._object_relative_goal(_baked, exclude_obj) if _cand is None
                else np.asarray(_cand, float))
        # what `_grasp_candidates` orbits if this attempt fails
        self._grasp_goal = goal

        def _rederive():
            """The goal is anchored to the OBJECT, so a base slip moves it in world."""
            if _cand is not None:
                return None
            _re = self._object_relative_goal(_baked, exclude_obj)
            if _re is _baked or np.array_equal(np.asarray(_re, float), np.asarray(_baked, float)):
                return None
            return np.asarray(_re, float)

        return goal, _rederive

    def _plan_approach(self, tr, frame_i, secs, exclude_obj=None, on_step=None,
                       hold_fingers=True, tag="reach", held=None, goal_fn=None):
        """Plan the approach to baked frame `frame_i` and execute it. True = executed (NO_ARRIVAL
        included, counted); None = the caller must END THE PICK; False = `ARM_PLANNER=0`, nothing
        planned. `exclude_obj` NAMES the target (still an obstacle); `goal_fn` maps the goal."""
        if not ARM_PLANNER:
            # Without this the planner can be switched OFF and every cycle still reports zero
            # fallbacks, which is the exact defect the counter exists to expose.
            self._fallback("planner-off-approach")
            return False
        goal, _rederive = self._approach_goal(tr, frame_i, exclude_obj)
        if goal_fn is not None:
            goal = goal_fn(goal)

        def _regoal():
            _re = _rederive()
            if _re is None:
                return None
            return _re if goal_fn is None else goal_fn(_re)

        res = move_to_pose(self, goal, secs, held=held, tag=tag, regoal=_regoal,
                           on_step=on_step, hold_fingers=hold_fingers)
        if res.outcome is Outcome.NO_PLAN:
            # Two different failures, two tags: OMPL found no route, or the goal was never a joint vector.
            # The second planned nothing, and a reader chasing it would go looking at the obstacle set.
            if "bad-target" in res.tags:
                self._fallback("bad-target")
                print(f">>> arm plan[{tag}]: {res.reason} -- nothing was planned", flush=True)
            else:
                self._fallback("no-plan")
                print(f">>> arm plan[{tag}]: no route -- ENDING the attempt: a planned primitive "
                      f"that cannot plan REFUSES", flush=True)
            return None
        if res.outcome in (Outcome.REFUSED, Outcome.PAYLOAD_LOST):
            # None, not False: False lets the caller continue from the pose the abort recovered to.
            # The executor already counted the degrade.
            print(f">>> arm plan[{tag}]: ABORTED mid-path -- reporting ABORT, not a fallback",
                  flush=True)
            return None
        # NO_ARRIVAL reports True here: tightening it is the CALLER's migration, since doing it here would
        # change which cycles proceed. The verdict is counted rather than discarded.
        if res.outcome is Outcome.NO_ARRIVAL:
            self._fallback("reach-no-arrival")
            print(f">>> arm plan[{tag}]: arrived SHORT -- {res.reason}. The reach-lowest column "
                  f"move and the seat run from the pose reached, not the planned goal.", flush=True)
        return True
