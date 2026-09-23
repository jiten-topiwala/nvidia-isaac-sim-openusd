"""The place insert: raise, const-z slide, lower -- one module per stage, as `morph/pick/` is.
Imports Isaac APIs at module level; only importable after SimulationApp.
"""
import math
import os

import numpy as np

from morph.arm.api import ArmModel, move_linear, sweep_top
from morph.config import COLUMN_MAX, A1_MAX


# Consecutive steps a gated joint may sit past its tracking tolerance before the servo gives up.
# A single step is the transient every new motion produces; 24 is 0.1 s at 240 Hz.
_SERVO_FOLLOW_STEPS = 24


class InsertStage:
    """One stage of `place`. Mixed into `PlaceMixin`; all `self.*` belong to `Demo`."""

    def _hand_top_model(self, q):
        """Hand-top world z the ARM MODEL gives for joint vector `q` WITHOUT commanding it; None if
        the model cannot be evaluated. Carries the live wrist joints through, because the hand box's
        reach above its origin depends on wrist orientation (why `sweep_top` is per sample)."""
        try:
            m = self._arm_model()
            q8 = np.array([float(q[self.idx[n]]) for n in m.Q8], float)
            p, R = m.fk(q8)[m.hand_link]
            bt, bR = self._arm_base_world()
            bt, bR = np.asarray(bt, float), np.asarray(bR, float)
            return float((bt + bR @ p)[2]) + sweep_top(bR @ R)
        except Exception:
            return None

    def _contact_watch(self, tag):
        """Per-step watch for the column ramps: a forced arm fighting a static shelf board NaNs
        SILENTLY, so the live contacts are the only warning. Handed to `move_linear` as `on_step`
        because the primitive owns the write loop now."""
        state = {"k": 0}

        def _watch():
            state["k"] += 1
            if state["k"] % 30 == 0 and getattr(self, "_ctc_any", None):
                print(f">>>   CONTACT during {tag}[{state['k']}]: "
                      f"{dict(sorted(self._ctc_any.items(), key=lambda kv: 'pickup_obj' in kv[0])[:6])}",
                      flush=True)

        return _watch

    def _seat_step(self, q_from, q_to, secs, hold_fn):
        """Cosine-eased joint write for the seating servo: one small increment, then the loop reads
        the object again. NOT a planned motion -- the servo presses INTO the board it would be
        checked against, so there is nothing here to collision-check, and the planned column ramps
        went to `move_linear`."""
        n = max(1, int(secs / self.dt))
        q = np.asarray(q_from, float).copy()
        for k in range(n):
            f = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / n)
            q = q_from + f * (np.asarray(q_to, float) - q_from)
            hold_fn(q)
            if k % 30 == 0 and getattr(self, "_ctc_any", None):
                print(f">>>   CONTACT during seat[{k}]: "
                      f"{dict(sorted(self._ctc_any.items(), key=lambda kv: 'pickup_obj' in kv[0])[:6])}",
                      flush=True)
        return q

    def _insert_sequence(self, obj, goal_xyz, clear, hold_fn, obj_idx=None):
        """Raise, slide at constant height, then lower -- three phases, never one diagonal, which
        would pass the carried object THROUGH a shelf plate. The SLIDE extends a1 at frozen dh, or
        else one `_solve_reach_z` (tilt couples reach and z); raise/lower are columns-only."""
        ih1, ih2 = self.idx["ColumnLeftBearingJoint_1"], self.idx["ColumnRightBearingJoint_1"]
        ia1, ith = self.idx.get("ArmLeftJoint_1"), self.idx.get("BaseJoint_1")
        if ia1 is None or ith is None or self.clut is None:
            return None
        goal_xyz = np.asarray(goal_xyz, float)
        slide_z = float(goal_xyz[2]) + clear
        a1max = A1_MAX

        def _pose():
            q = np.asarray(self.robot.get_joint_positions(), float).copy()
            return (q, float(q[ih1]), float(q[ih2]), float(q[ia1]), float(q[ith]))

        # object minus LUT grip point, in world
        def _off_world():
            q, h1, h2, a1, th = _pose()
            return np.asarray(obj.get_world_poses()[0][0], float) - self._grip_fk(h1, h2, a1, th)

        # The object stays an obstacle to every link except the hand holding it. `move_linear`
        # builds the set from these names and refuses the motion itself if it cannot.
        _grasp = () if obj_idx is None else (f"pickup_obj_{obj_idx}",)

        # phase 1: RAISE to the slide height, columns only (reach and yaw untouched)
        dz = slide_z - float(np.asarray(obj.get_world_poses()[0][0], float)[2])
        with self._time_stage('raise'):
            # `fingers=None` keeps what this check has always judged, the SWEPT finger box. The commanded
            # pads are a smaller hand, so adopting them here would loosen the check mid-descent.
            _mv = move_linear(self, np.array([0.0, 0.0, dz]), "base", secs=1.0,
                              allow_contact_with=_grasp, fingers=None, tag="raise",
                              on_step=self._contact_watch("raise"))
        q = _mv.q_end if _mv else None
        if q is None:
            print(f">>> insert[raise]: {_mv.reason}", flush=True)
            self._safety_abort = True
            self._insert_refused = True
            print(">>> insert[raise]: refused -- holding the last cleared pose, grip not released",
                  flush=True)
            return None
        print(f">>> insert[raise]: object z -> "
              f"{float(np.asarray(obj.get_world_poses()[0][0], float)[2]):.3f} "
              f"(slide height {slide_z:.3f}, clearance {clear * 1000:.0f}mm over the slot)",
              flush=True)

        # phase 2: const-z slide -- one continuous a1 extension with the boom tilt FROZEN, so the hand
        # attitude is constant and z is held analytically per step from the LUT.
        _bx, _by, _byaw = self.base_ledger()
        q, h1, h2, a1, th = _pose()
        dh0 = h2 - h1
        off = _off_world()
        g_w = np.array([goal_xyz[0], goal_xyz[1], slide_z], float) - off
        c, s_ = math.cos(-_byaw), math.sin(-_byaw)
        dx, dy = float(g_w[0] - _bx), float(g_w[1] - _by)
        fx = c * dx - s_ * dy
        mx, my = self.clut["mount"][0], self.clut["mount"][1]

        # chassis-forward reach of the grip at this shape
        def _reach_rot(dh_, a1_):
            p_ = self._lut_interp(self.clut["grip"], dh_, a1_, 3)
            px_, py_ = float(p_[0]) - mx, float(p_[1]) - my
            return mx + math.cos(th) * px_ - math.sin(th) * py_

        a1max = A1_MAX
        dh_f, a1_f = dh0, None
        if _reach_rot(dh0, a1max) >= fx - 1e-3:
            # reach is monotonic in a1 along a dh row
            a1_f = self._bisect_a1(lambda a: _reach_rot(dh0, a) >= fx, min(a1, a1max), a1max)
            # Second feasibility gate: COLUMN HEADROOM, not just reach. Past the stop the clip
            # crushes dh while the forced passives carry the old value, and the linkage NaNs.
            _h1_f = C_pre = h1 + self._lut_z(dh0, a1) - self._lut_z(dh0, a1_f)
            if _h1_f + dh0 > COLUMN_MAX - 0.01:
                print(f">>> insert[slide]: column headroom EXHAUSTED at the frozen dh -- h2 would "
                      f"end {_h1_f + dh0:.3f} vs the {COLUMN_MAX:.2f} stop (the top-shelf exception: tilt "
                      f"because straight cannot)", flush=True)
                a1_f = None                                   # fall through to the tilt solve
        if a1_f is not None:
            print(f">>> insert[slide]: UNTILTED -- a1 {a1:.3f} -> {a1_f:.3f} reaches {fx:.3f}m at "
                  f"the frozen dh {dh0:+.3f}; no boom change, one continuous ramp", flush=True)
        if a1_f is None:
            sol = self._solve_reach_z(fx, float(g_w[2]), dh_ref=dh0, a1_max=a1max,
                                      th=th, a1_ref=a1, h1_ref=h1)
            if sol is None:
                print(f">>> insert[slide]: NO SOLVE -- reach {fx:.3f}m is outside the arm envelope "
                      f"even with tilt", flush=True)
                return None
            _h1s, _h2s, a1_f, _r_s, _e_s = sol
            dh_f = _h2s - _h1s
            print(f">>> insert[slide]: tilt REQUIRED -- a1 alone reaches "
                  f"{_reach_rot(dh0, a1max):.3f}m of the needed {fx:.3f}m, so dh {dh0:+.3f} -> "
                  f"{dh_f:+.3f} (the exception, not the default)", flush=True)
        C = h1 + self._lut_z(dh0, a1)                      # z invariant: h1 + lut_z is the grip height
        # ...for the GRIP; the OBJECT hangs a pitch-dependent offset below it, so a dh ramp walks
        # it off the line even with the grip invariant exact.
        _C_corr = 0.0
        # The turret correction rides the slide; closing the lateral miss afterwards would swing
        # the turret at the moment of placing.
        _th_corr = _a1_corr = _a1_ext = 0.0
        _ith = self.idx.get("BaseJoint_1")
        _J = None
        if _ith is not None and os.environ.get("SLIDE_TH_TRACK", "1") == "1":
            def _probe(_ji, _amt):
                _o0 = np.asarray(obj.get_world_poses()[0][0], float)[:2].copy()
                _qpr = np.asarray(self.robot.get_joint_positions(), float).copy()
                _base = float(_qpr[_ji])
                _qpr[_ji] = _base + _amt
                self._force(_qpr); self._apply(_qpr)
                self.world.step(render=False)
                _g = (np.asarray(obj.get_world_poses()[0][0], float)[:2] - _o0) / _amt
                _qpr[_ji] = _base
                self._force(_qpr); self._apply(_qpr)
                self.world.step(render=False)
                return _g
            try:
                if self._drive_on():
                    # On drives a one-step probe barely moves the object: use the FK, not a probe.
                    _q0j, _h1j, _h2j, _a1j, _thj = _pose()
                    _pj = self._grip_fk(_h1j, _h2j, _a1j, _thj)
                    _Jm = np.column_stack([(self._grip_fk(_h1j, _h2j, _a1j + 1e-3, _thj) - _pj)[:2] / 1e-3,
                                           (self._grip_fk(_h1j, _h2j, _a1j, _thj + 1e-3) - _pj)[:2] / 1e-3])
                else:
                    _Jm = np.column_stack([_probe(ia1, 0.02), _probe(_ith, 0.02)])   # [d/da1, d/dth]
                # well-conditioned
                if abs(float(np.linalg.det(_Jm))) > 1e-4:
                    _J = np.linalg.inv(_Jm)
            except Exception:
                _J = None
        n = max(2, int(2.0 / self.dt))
        _samp = max(1, n // 6)
        _ht_max = float("-inf")     # the whole-slide maximum the arm-clear gate reads
        # `_ub` is the underside of the board above THIS slot: the slide may not command a pose whose
        # predicted hand top passes it. None means the goal is not a rack slot, so there is no board.

        # the rack owns its own geometry
        from morph.config import ARM_CLEAR_MARGIN, board_above
        _ub = board_above(goal_xyz)
        _ht_bound = None if _ub is None else _ub - ARM_CLEAR_MARGIN
        _ht_bias, _ht_m, _ht_seen = 0.0, None, 0
        if _ht_bound is not None:
            # Anchor the model on the MEASURED hand: the prediction then carries only the model's
            # error over one step, not its absolute offset from the live arm.
            _ht_m = self._hand_top_model(q)
            if _ht_m is not None:
                _hp0, _hR0 = self._hand_frame()
                _ht_bias = float(_hp0[2]) + sweep_top(_hR0) - _ht_m
            print(f">>> insert[slide]: hand-top guard ARMED at {_ht_bound:.3f} "
                  f"({_ub:.3f} board underside less {ARM_CLEAR_MARGIN * 1000:.0f}mm), model "
                  f"anchored on the measured hand at {_ht_bias * 1000:+.0f}mm"
                  if _ht_m is not None else
                  f">>> insert[slide]: hand-top guard OFF -- the arm model would not evaluate, so "
                  f"nothing predicts the {_ht_bound:.3f} bound this slide is under", flush=True)
        with self._time_stage("slide"):
            for k in range(n):
                f = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / n)
                a1_k = a1 + f * (a1_f - a1)
                dh_k = dh0 + f * (dh_f - dh0)
                _step_c = float(np.clip(_C_corr, -5e-4, 5e-4))
                C += _step_c
                _C_corr -= _step_c
                if _J is not None:
                    _st = float(np.clip(_th_corr, -3e-4, 3e-4))
                    th += _st
                    _th_corr -= _st
                    q[_ith] = th

                h1_k = float(np.clip(C - self._lut_z(dh_k, a1_k), 0.0, COLUMN_MAX))
                q[ih1] = h1_k
                q[ih2] = float(np.clip(h1_k + dh_k, 0.0, COLUMN_MAX))
                q[ia1] = a1_k
                # The passives must follow the closure. Frozen at the entry shape they stop satisfying
                # it the moment the tilt fallback ramps dh: the hand leaves the LUT model and NaNs.
                for _pn, _pv in self._closure_passives(dh_k, a1_k).items():
                    if _pn in self.idx:
                        q[self.idx[_pn]] = float(_pv)
                if _ht_bound is not None:
                    _ht_m = self._hand_top_model(q)
                    if _ht_m is not None:
                        _ht_seen += 1
                        if _ht_m + _ht_bias > _ht_bound:
                            # HALT here: a backout is an unchecked reverse between two boards, and
                            # going on would command the pose the check just refused.
                            self._safety_abort = True
                            self._insert_refused = True
                            # no steps ran means no maximum: do not print one
                            if k:
                                print(f">>> insert[slide]: hand top max {_ht_max:.3f} over {k} steps "
                                      f"(every step, up to the ABORT)", flush=True)
                            print(f">>> insert[slide]: ABORTING before step {k + 1}/{n} -- the pose "
                                  f"about to be commanded puts the hand top at "
                                  f"{_ht_m + _ht_bias:.3f}, past the {_ht_bound:.3f} bound under the "
                                  f"{_ub:.3f} board above this slot. Holding the last cleared pose; "
                                  f"the object stays held, nothing is released.", flush=True)
                            return None
                hold_fn(q)
                # The measurement, taken every step, is also what the next step's prediction is anchored on.
                # `sweep_top` accounts for the wrist-orientation dependent offset above the hand origin.
                _hp_k, _hR_k = self._hand_frame()        # highest arm part inside the rack
                _hz_k = float(_hp_k[2])
                _ht_k = _hz_k + sweep_top(_hR_k)
                _ht_max = max(_ht_max, _ht_k)
                if _ht_m is not None:
                    _ht_bias = _ht_k - _ht_m
                # verify_place.py samples: same fields/format
                if (k + 1) % _samp == 0:
                    _zo_k = float(np.asarray(obj.get_world_poses()[0][0], float)[2]) - slide_z
                    _C_corr = -_zo_k                     # object error -> column correction budget
                    if _J is not None:
                        # Lateral only: nulling FORWARD error double-counts the travel still to come.
                        _err_k = np.asarray(obj.get_world_poses()[0][0], float)[:2] - goal_xyz[:2]
                        _gt = _Jm[:, 1]
                        _th_corr = float(np.clip(-float(_err_k @ _gt) / float(_gt @ _gt), -0.30, 0.30))
                    # Re-solve the a1 endpoint against the LIVE th: `a1_f` was bisected against the
                    # ENTRY th, and reach is measured through th, so the turret trim staled it.
                    if a1_f is not None and _J is not None:
                        if _reach_rot(dh_f, a1max) >= fx - 1e-3:
                            a1_f = self._bisect_a1(lambda a: _reach_rot(dh_f, a) >= fx,
                                                   min(a1_k, a1max), a1max, iters=30)
                    print(f">>> insert[slide] wp {(k + 1) // _samp}/6: reach -> "
                          f"{_reach_rot(dh_k, a1_k):.3f}m (final {fx:.3f}, z off slide "
                          f"{_zo_k * 1000:+.0f}mm) | dh {dh_k:+.3f}  a1 {a1_k:.3f}  "
                          f"h1 {h1_k:.3f} -> {h1_k:.3f}  th {th:+.3f} held  hand z {_hz_k:.3f}  "
                          f"hand top {_ht_k:.3f}",
                          flush=True)
        # The value verify_place's arm-clear gate reads. The `wp` lines above are a human's
        # six-point view of this same slide; this is every step of it.
        print(f">>> insert[slide]: hand top max {_ht_max:.3f} over {n} steps (every step; the wp "
              f"lines above are {n // _samp} samples)", flush=True)
        if _ht_bound is not None:
            print(f">>> insert[slide]: hand-top guard cleared {_ht_seen}/{n} steps against "
                  f"{_ht_bound:.3f}"
                  + ("" if _ht_seen == n else " -- the rest went UNPREDICTED"), flush=True)
        op = np.asarray(obj.get_world_poses()[0][0], float)
        print(f">>> insert[slide]: object -> {np.round(op, 3).tolist()} at the slide height "
              f"(lateral residual "
              f"{float(np.linalg.norm(op[:2] - goal_xyz[:2])) * 1000:.1f}mm)", flush=True)

        # phase 2.5: trim BEFORE the lower -- trimming after swings the object sideways
        # millimetres above the board, just before release.
        if os.environ.get("TRIM_BEFORE_LOWER", "1") == "1":
            self._arm_insert(obj, np.array([goal_xyz[0], goal_xyz[1], slide_z]),
                             1.0, "trim-high", hold_fn, obj_idx=obj_idx)
            if self._insert_refused:
                print(">>> insert[trim-high]: refused -- the lower below drives the same "
                      "corridor, so the insert ends here, grip not released", flush=True)
                return None
        # Boom vs the robot's own battery box, at the two poses that bind. Always prints.
        self._body_clear("slide")

        # phase 3: LOWER onto the slot, columns only
        dz = float(goal_xyz[2]) - float(op[2])
        with self._time_stage('lower'):
            _mv = move_linear(self, np.array([0.0, 0.0, dz]), "base", secs=1.2,
                              allow_contact_with=_grasp, fingers=None, tag="lower",
                              on_step=self._contact_watch("lower"))
        q = _mv.q_end if _mv else None
        if q is None:
            print(f">>> insert[lower]: {_mv.reason}", flush=True)
            self._safety_abort = True
            self._insert_refused = True
            print(">>> insert[lower]: refused -- holding the last cleared pose, grip not released",
                  flush=True)
            return None
        if self._drive_on():
            # The lower can end with the object just ABOVE the board, at zero force, where friction
            # cannot resist the open. Settle until it stops following the columns down.
            _seated = False
            for _kz in range(int(os.environ.get("PLACE_SEAT_MM", "6"))):
                _z0s = float(np.asarray(obj.get_world_poses()[0][0], float)[2])
                q_t = q.copy()
                q_t[ih1] = float(np.clip(q[ih1] - 0.001, 0.0, COLUMN_MAX))
                q_t[ih2] = float(np.clip(q[ih2] - 0.001, 0.0, COLUMN_MAX))
                q = self._seat_step(q, q_t, 0.1, hold_fn)
                if float(np.asarray(obj.get_world_poses()[0][0], float)[2]) > _z0s - 0.0005:
                    _seated = True
                    break
            print(f">>> insert[lower]: board carries the object after {_kz} extra mm"
                  if _seated else
                  f">>> insert[lower]: object still following the columns after {_kz + 1} mm -- not seated",
                  flush=True)
        op = np.asarray(obj.get_world_poses()[0][0], float)
        res = float(np.linalg.norm(goal_xyz - op))
        print(f">>> insert[lower]: object -> {np.round(op, 3).tolist()} target "
              f"{np.round(goal_xyz, 3).tolist()} residual {res * 1000:.1f}mm", flush=True)
        # the binding pose: lower + settle, and the release is next
        self._body_clear("seat")
        return q, res

    # Every link that MOVES under the servo's four joints, less the distal hand and the payload: those
    # touch nothing during an insert, while checking a slipping payload refuses working placements.
    SERVO_WATCH = ("Arm_1", "Arm_Left_1", "Arm_Right_1",
                   "Bearing_Column_Left_1", "Bearing_Column_Right_1", "Rotation_Link_Left_1",
                   "Contact_Cylinder_1_1", "Contact_Cylinder_1_2")

    def _servo_blocked(self, q, obs, base):
        """The proximal link colliding at `q`, None if clear, or False if the watch cannot judge.

        False is NOT "clear": a watch that failed to evaluate must refuse the insert, or an
        exception silently converts the only check on this motion into a pass.
        """
        try:
            from morph.arm.api import MARGIN
            m = self._arm_model()
            q8 = np.asarray([q[self.idx[n]] for n in ArmModel.Q8], float)
            return m.first_hit(q8, obs, MARGIN, None, base, links=self.SERVO_WATCH)
        except Exception as e:                     # noqa: BLE001
            print(f">>> insert: the servo watch could not be evaluated ({e}) -- refusing",
                  flush=True)
            return False

    def _arm_insert(self, obj, target, secs, tag, hold_fn, q_ref=None, obj_idx=None):
        """Drive the OBJECT to `target` by servoing the ARM in joint space, held throughout. Not a
        Cartesian line: a line resolves this move's redundancy into the turret, which swings the
        laterally offset object (FINDINGS). Columns for z; a1/th under an explicit yaw budget."""
        if getattr(self, "_insert_refused", False):
            # A refusal means a check rejected THIS corridor, and every later leg drives the same one.
            # The NaN residual is what the servo refusal already returns.
            print(f">>> insert[{tag}]: NOT run -- the insert already refused this corridor",
                  flush=True)
            return np.asarray(self.robot.get_joint_positions(), float).copy(), float("nan")
        ih1, ih2 = self.idx["ColumnLeftBearingJoint_1"], self.idx["ColumnRightBearingJoint_1"]
        ia1, ith = self.idx.get("ArmLeftJoint_1"), self.idx.get("BaseJoint_1")
        n = max(1, int(secs / self.dt))
        kp = 1.2
        tol = 0.008
        # rad of turret yaw the insert may spend; a real correction is tens of mrad, so anything
        # near this is a runaway
        _TH_SPAN = 0.35
        _a1_lo = _a1_hi = (float(np.asarray(self.robot.get_joint_positions(), float)[self.idx["ArmLeftJoint_1"]])
                           if "ArmLeftJoint_1" in self.idx else 0.0)
        _a1_in = _a1_lo
        eps, best = 1e-3, None
        q = np.asarray(self.robot.get_joint_positions(), float).copy()
        q_best = q.copy()
        th0 = float(q[ith]) if ith is not None else 0.0
        e0 = None
        _sv_run = 0
        # once: `_arm_obstacles` walks the whole stage
        try:
            # The carried object is DROPPED (a boom "hitting" its own payload is not a scene
            # collision) -- looser than the ramps' "grasped" tag, deliberately; both pass live.
            _sv_obs = self._arm_obstacles(
                exclude_names=((f"pickup_obj_{obj_idx}",) if obj_idx is not None else ()))
            _sv_base = self._arm_base_world()
        except Exception as e:                     # noqa: BLE001
            # Same rule the ramps get from `move_linear`: asked for a check and could not build
            # it is a REFUSAL, not a silent pass.
            print(f">>> insert[{tag}]: the servo watch could not be BUILT ({e}) -- refusing",
                  flush=True)
            self._fallback("servo-watch-unbuildable")
            self._safety_abort = True
            self._insert_refused = True
            return np.asarray(self.robot.get_joint_positions(), float).copy(), float("nan")
        for k in range(n):
            op = np.asarray(obj.get_world_poses()[0][0], float)
            err = np.asarray(target, float) - op
            e = float(np.linalg.norm(err))
            # Divergence guard: when the joint-space gradient stops describing the arm the servo
            # runs the turret through a radian and puts the object hundreds of metres away.
            if e0 is None:
                e0 = e
            _run = (not np.all(np.isfinite(op))
                    or e > max(4.0 * e0, e0 + 0.5)
                    or (ith is not None and abs(float(q[ith]) - th0) > _TH_SPAN))
            if _run:
                print(f">>> insert[{tag}]: DIVERGED at step {k} (residual {e * 1000:.0f}mm from "
                      f"{e0 * 1000:.0f}mm, th moved "
                      f"{abs(float(q[ith]) - th0) if ith is not None else 0.0:+.3f}) -- "
                      f"restoring the closest pose ({best * 1000 if best else 0:.0f}mm) and "
                      f"handing the remainder to the ease", flush=True)
                q = q_best.copy()
                hold_fn(q)
                break
            if ia1 is not None:
                _a1_lo = min(_a1_lo, float(q[ia1]))
                _a1_hi = max(_a1_hi, float(q[ia1]))
            if best is None or e < best:
                best, q_best = e, q.copy()
            if e < tol:
                break
            h1, h2 = float(q[ih1]), float(q[ih2])
            a1 = float(q[ia1]) if ia1 is not None else 0.0
            th = float(q[ith]) if ith is not None else 0.0
            # The columns are two DOFs; only their COMMON mode (height) is servoed here, so the
            # boom tilt (their differential) is whatever the slide entered with.
            d_mean = float(np.clip(kp * err[2] * self.dt * 20.0, -0.004, 0.004))
            # xy: 2x2 Jacobian of the LUT grip position wrt (a1, th), solved least-squares
            da1 = dth = 0.0
            try:
                p0 = self._grip_fk(h1, h2, a1, th)
                ja = (self._grip_fk(h1, h2, a1 + eps, th) - p0)[:2] / eps
                jt = (self._grip_fk(h1, h2, a1, th + eps) - p0)[:2] / eps
                # Project, do not least-squares: a blind 2x2 solve dumps what a1 cannot supply
                # into th and swings the object sideways once a1 saturates.
                na, nt = float(np.linalg.norm(ja)), float(np.linalg.norm(jt))
                if na > 1e-9:
                    da1 = float(np.clip(kp * float(err[:2] @ (ja / na)) / na
                                        * self.dt * 20.0, -0.004, 0.004))
                if nt > 1e-9:
                    dth = float(np.clip(kp * float(err[:2] @ (jt / nt)) / nt
                                        * self.dt * 20.0, -0.010, 0.010))
            except Exception:
                pass
            q[ih1] = float(np.clip(h1 + d_mean, 0.0, COLUMN_MAX))
            q[ih2] = float(np.clip(h2 + d_mean, 0.0, COLUMN_MAX))
            if ia1 is not None:
                q[ia1] = float(np.clip(a1 + da1, 0.0, A1_MAX))
            if ith is not None:
                # Clamp the turret: unbounded it integrates a step per sample into radians.
                q[ith] = float(np.clip(th + dth, th0 - _TH_SPAN, th0 + _TH_SPAN))
            _blk = self._servo_blocked(q, _sv_obs, _sv_base)
            # the watch could not judge this step
            if _blk is False:
                _blk = "watch-unevaluable"
            if _blk is not None:
                print(f">>> insert[{tag}]: ABORTING at step {k + 1}/{n} -- {_blk} would collide at "
                      f"the pose about to be commanded. Holding the last cleared pose; the object "
                      f"stays held.", flush=True)
                self._fallback("servo-proximal-blocked")
                self._safety_abort = True
                self._insert_refused = True
                return q, float("nan")
            hold_fn(q)
            # Distance-limited guarded move: sustained following error means every pose downstream
            # is computed from a command the arm is not at, with the object held between two boards.
            _sv_over = float(getattr(self, "_track_over", 0.0) or 0.0)
            _sv_run = (_sv_run + 1) if _sv_over > 1.0 else 0
            if _sv_run >= _SERVO_FOLLOW_STEPS:
                print(f">>> insert[{tag}]: ABORTING at step {k + 1}/{n} -- a gated joint has been "
                      f"past its tolerance for {_sv_run} consecutive steps (worst {_sv_over:.2f}x); "
                      f"holding here, the object stays held.", flush=True)
                self._fallback("servo-following-error")
                self._safety_abort = True
                self._insert_refused = True
                return q, float("nan")
        op = np.asarray(obj.get_world_poses()[0][0], float)
        res = float(np.linalg.norm(np.asarray(target, float) - op))
        print(f">>>   insert[{tag}] arm: a1 {float(q[ia1]) if ia1 is not None else float('nan'):.3f} "
              f"th {float(q[ith]) if ith is not None else float('nan'):+.3f} "
              f"dh {float(q[ih2] - q[ih1]):+.3f} "
              f"(baked dh was {float(q_ref[ih2] - q_ref[ih1]):+.3f}; tilt traded for reach)"
              if q_ref is not None else "",
              flush=True)
        print(f">>> insert[{tag}]: object -> {np.round(op, 3).tolist()} target "
              f"{np.round(np.asarray(target, float), 3).tolist()} residual {res * 1000:.1f}mm "
              f"[a1 {_a1_in:.3f} -> {float(np.asarray(self.robot.get_joint_positions(), float)[ia1]) if ia1 is not None else 0.0:.3f}, "
              f"span {_a1_lo:.3f}..{_a1_hi:.3f}] "
              f"(best {best * 1000 if best is not None else float('nan'):.1f}mm, "
              f"{k + 1}/{n} steps)", flush=True)
        return q, res
