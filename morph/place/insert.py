"""The place insert: raise, const-z slide, lower -- one module per stage, as `morph/pick/` is.

Extracted from the single flat `place.py`, whose `place()` had grown to hold a ~200-line nested
`insert_sequence` closure containing a further three closures. The behaviour is unchanged; only the
nesting is. Imports Isaac APIs at module level; only importable after SimulationApp.
"""
import math
import os

import numpy as np

from morph.config import HEADLESS, COLUMN_MAX, A1_MAX


class InsertStage:
    """One stage of `place`. Mixed into `PlaceMixin`; all `self.*` belong to `Demo`."""

    def _ramp(self, q_from, q_to, secs, hold_fn):
        n = max(1, int(secs / self.dt))
        q = np.asarray(q_from, float).copy()
        for k in range(n):
            f = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / n)
            q = q_from + f * (np.asarray(q_to, float) - q_from)
            hold_fn(q)
            # Contact watch. A kinematically forced arm fighting a static shelf board is the
            # documented NaN mode, and it is invisible unless the contact ledger is read before the
            # numbers stop being numbers.
            if k % 30 == 0:
                if getattr(self, "_ctc_any", None):
                    print(f">>>   CONTACT during ramp[{k}]: "
                          f"{dict(list(self._ctc_any.items())[:6])}", flush=True)
                    self._ctc_any = {}
                _rp = np.asarray(self.robot.get_world_pose()[0], float)
                if not np.all(np.isfinite(_rp)) and not getattr(self, "_nan_said", False):
                    self._nan_said = True
                    print(f">>>   NON-FINITE at ramp step {k}/{n}: root {_rp}", flush=True)
        return q

    def _insert_sequence(self, obj, goal_xyz, clear, hold_fn):
        """Raise, slide at constant height, then lower -- three phases, not one move.

        The slide runs at a fixed height above the shelf and the lower is separate, because a
        single diagonal move can pass the carried object THROUGH a shelf plate on the way in.
        Solving one pose and ramping reach, height and yaw together is exactly that diagonal, and
        it also lands further off target than a plain servo does.

        Reach and height are coupled through the tilt (straightening dh 0.110 -> 0.000 gains
        190mm of reach and 440mm of height together), so the SLIDE cannot be serviced
        incrementally -- it is solved with `_solve_reach_z`, which picks the (dh, a1, h1) that
        hits the wanted reach AT the slide height. RAISE and LOWER are columns-only, where the
        coupling does not arise at all.

        The grip->object offset is carried through the turret rotation: it is fixed in the HAND
        frame, so a solve that turns th by dth rotates it by dth about world z. Aiming with the
        pre-rotation offset is what made the earlier attempt miss; iterate the solve twice so
        the offset used matches the th it produces.
        """
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

        def _off_world():          # object minus LUT grip point, in world
            q, h1, h2, a1, th = _pose()
            return np.asarray(obj.get_world_poses()[0][0], float) - self._grip_fk(h1, h2, a1, th)

        # ---- phase 1: RAISE to the slide height, columns only (reach and yaw untouched) ----
        q, h1, h2, a1, th = _pose()
        dz = slide_z - float(np.asarray(obj.get_world_poses()[0][0], float)[2])
        q_t = q.copy()
        q_t[ih1] = float(np.clip(h1 + dz, 0.0, COLUMN_MAX))
        q_t[ih2] = float(np.clip(h2 + dz, 0.0, COLUMN_MAX))
        q = self._ramp(q, q_t, 1.0, hold_fn)
        print(f">>> insert[raise]: object z -> "
              f"{float(np.asarray(obj.get_world_poses()[0][0], float)[2]):.3f} "
              f"(slide height {slide_z:.3f}, clearance {clear * 1000:.0f}mm over the slot)",
              flush=True)

        # ---- phase 2: const-z slide -- one continuous a1 extension, boom tilt FROZEN ---- Two
        # structural rules, both of which a waypoint solve violates:
        #   1. Prefer reaching the height WITHOUT tilt. A cost-based solve straightens the boom on
        # every
        #      slide because its cost function likes it, when the reach it needs is available at the
        #      current tilt from a1 alone -- and every tilt change pitches the hand, forcing mid-
        # slide
        #      wrist re-levels. Freeze dh, and fall back to a solved tilt only when a1 maxed at the
        #      frozen dh cannot reach.
        #   2. The motion has to be smooth. With dh frozen the hand attitude is constant, so z can
        # be held
        #      ANALYTICALLY per step from the LUT (h1 = C - lut_z(dh, a1)): one cosine ramp of a1,
        #      columns compensating exactly -- no waypoints, no pauses, no mid-slide levelling.
        _bx, _by, _byaw = self.base_ledger()
        q, h1, h2, a1, th = _pose()
        dh0 = h2 - h1
        off = _off_world()
        g_w = np.array([goal_xyz[0], goal_xyz[1], slide_z], float) - off
        c, s_ = math.cos(-_byaw), math.sin(-_byaw)
        dx, dy = float(g_w[0] - _bx), float(g_w[1] - _by)
        fx = c * dx - s_ * dy
        mx, my = self.clut["mount"][0], self.clut["mount"][1]

        def _reach_rot(dh_, a1_):        # chassis-forward reach of the grip at this shape
            p_ = self._lut_interp(self.clut["grip"], dh_, a1_, 3)
            px_, py_ = float(p_[0]) - mx, float(p_[1]) - my
            return mx + math.cos(th) * px_ - math.sin(th) * py_

        a1max = A1_MAX
        dh_f, a1_f = dh0, None
        if _reach_rot(dh0, a1max) >= fx - 1e-3:
            # reach is monotonic in a1 along a dh row
            a1_f = self._bisect_a1(lambda a: _reach_rot(dh0, a) >= fx, min(a1, a1max), a1max)
            # Second feasibility gate: COLUMN HEADROOM, not just reach. On the high slot the
            # untilted plan ends past the column stop, the loop's clip then silently crushes dh
            # while the forced passives are still written for the old value, and the inconsistent
            # linkage NaNs the articulation.
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
            sol = self._solve_reach_z(fx, float(g_w[2]), 0.0, dh_ref=dh0, a1_max=a1max,
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
        # ...for the GRIP. The OBJECT hangs a pitch-dependent offset below it, so when the tilt
        # fallback ramps dh the object walks off the line even with a perfect grip invariant and
        # perfect passives.
        _C_corr = 0.0
        # The turret correction rides the slide instead of firing after it. The slide lands a
        # repeatable lateral miss -- the dock is aimed from the BAKED grip frame while the analytic
        # slide ends in a different arm shape -- and closing it afterwards fires the turret last, at
        # the moment of placing.
        _th_corr = _a1_corr = _a1_ext = 0.0
        _ith = self.idx.get("BaseJoint_1")
        _J = None
        if _ith is not None and os.environ.get("SLIDE_TH_TRACK", "1") == "1":
            def _probe(_ji, _amt):
                _o0 = np.asarray(obj.get_world_poses()[0][0], float)[:2].copy()
                _qpr = np.asarray(self.robot.get_joint_positions(), float).copy()
                _base = float(_qpr[_ji])
                _qpr[_ji] = _base + _amt
                self._force(_qpr); self._apply(_qpr); self._pin_grip(obj)
                self.world.step(render=False)
                _g = (np.asarray(obj.get_world_poses()[0][0], float)[:2] - _o0) / _amt
                _qpr[_ji] = _base
                self._force(_qpr); self._apply(_qpr); self._pin_grip(obj)
                self.world.step(render=False)
                return _g
            try:
                _Jm = np.column_stack([_probe(ia1, 0.02), _probe(_ith, 0.02)])   # [d/da1, d/dth]
                if abs(float(np.linalg.det(_Jm))) > 1e-4:                        # well-conditioned
                    _J = np.linalg.inv(_Jm)
            except Exception:
                _J = None
        n = max(2, int(2.0 / self.dt))
        _samp = max(1, n // 6)
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
            # The passives must follow the closure, or the model is fiction. Writing only the
            # columns and a1 leaves the four passive four-bar joints frozen at the entry shape,
            # which is survivable while dh is frozen -- but the moment the tilt fallback ramps dh
            # they stop satisfying the closure, the real hand falls away from the LUT model, and the
            # inconsistent linkage NaNs the articulation.
            for _pn, _pv in self._closure_passives(dh_k, a1_k).items():
                if _pn in self.idx:
                    q[self.idx[_pn]] = float(_pv)
            hold_fn(q)
            if (k + 1) % _samp == 0:                 # verify_place.py samples: same fields/format
                _zo_k = float(np.asarray(obj.get_world_poses()[0][0], float)[2]) - slide_z
                # highest arm part inside the rack -- logged so the checker can GATE arm-vs-board
                # clearance, not just the object's
                _hz_k = float(self._hand_frame()[0][2])
                _C_corr = -_zo_k                     # object error -> column correction budget
                if _J is not None:
                    # Lateral from the measured probe. Correcting FORWARD the same way was tried and
                    # regressed: early in the ramp the xy error still contains the travel the plan
                    # has yet to cover, so a straight error-null double-counts it.
                    _err_k = np.asarray(obj.get_world_poses()[0][0], float)[:2] - goal_xyz[:2]
                    _gt = _Jm[:, 1]
                    _th_corr = float(np.clip(-float(_err_k @ _gt) / float(_gt @ _gt), -0.30, 0.30))
                # Re-solve the a1 endpoint against the LIVE th. `a1_f` was bisected once against the
                # ENTRY th, but the turret correction moves th and reach is measured through it, so
                # the endpoint goes stale and the push overshoots before the trim hauls a1 back.
                if a1_f is not None and _J is not None:
                    if _reach_rot(dh_f, a1max) >= fx - 1e-3:
                        a1_f = self._bisect_a1(lambda a: _reach_rot(dh_f, a) >= fx,
                                               min(a1_k, a1max), a1max, iters=30)
                print(f">>> insert[slide] wp {(k + 1) // _samp}/6: reach -> "
                      f"{_reach_rot(dh_k, a1_k):.3f}m (final {fx:.3f}, z off slide "
                      f"{_zo_k * 1000:+.0f}mm) | dh {dh_k:+.3f}  a1 {a1_k:.3f}  "
                      f"h1 {h1_k:.3f} -> {h1_k:.3f}  th {th:+.3f} held  hand z {_hz_k:.3f}",
                      flush=True)
        op = np.asarray(obj.get_world_poses()[0][0], float)
        print(f">>> insert[slide]: object -> {np.round(op, 3).tolist()} at the slide height "
              f"(lateral residual "
              f"{float(np.linalg.norm(op[:2] - goal_xyz[:2])) * 1000:.1f}mm)", flush=True)

        # ---- phase 2.5: trim at the slide height, before the lower ---- The slide lands a
        # repeatable lateral miss, and trimming it AFTER the lower swings the object sideways
        # millimetres above the board, just before release.
        if os.environ.get("TRIM_BEFORE_LOWER", "1") == "1":
            self._arm_insert(obj, np.array([goal_xyz[0], goal_xyz[1], slide_z]),
                             1.0, "trim-high", hold_fn)

        # ---- phase 3: LOWER onto the slot, columns only ----
        q, h1, h2, a1, th = _pose()
        dz = float(goal_xyz[2]) - float(op[2])
        q_t = q.copy()
        q_t[ih1] = float(np.clip(h1 + dz, 0.0, COLUMN_MAX))
        q_t[ih2] = float(np.clip(h2 + dz, 0.0, COLUMN_MAX))
        q = self._ramp(q, q_t, 1.2, hold_fn)
        op = np.asarray(obj.get_world_poses()[0][0], float)
        res = float(np.linalg.norm(goal_xyz - op))
        print(f">>> insert[lower]: object -> {np.round(op, 3).tolist()} target "
              f"{np.round(goal_xyz, 3).tolist()} residual {res * 1000:.1f}mm", flush=True)
        return q, res

    def _arm_insert(self, obj, target, secs, tag, hold_fn, q_ref=None):
        """Drive the OBJECT to `target` by moving the ARM, in joint space.

        The arm slides from the carry anchor to the slot with the object still held, and only
        releases once it is there -- the object is never moved independently. Without this stage
        the baked trajectory stops `INSERT_STANDOFF` short and a glide carries the OBJECT the rest
        of the way, which is why backing the base off to clear the rack makes the object fly to
        the slot on its own -- one knob doing two jobs.

        MuJoCo closes the gap with Cartesian IK, which this arm cannot use: it is a closed
        four-bar and `_jog_tick` diverges ("asked 71mm, hand moved 97151339mm"). The pick
        descent already solves the same problem in JOINT space against the baked closure LUT, so
        do that here: servo the columns (z, 1:1 by construction in `_grip_fk`) and a1/th
        (horizontal, via a finite-difference Jacobian on the LUT). Every joint touched is one
        the LUT covers, so the linkage stays on its closure manifold.
        """
        ih1, ih2 = self.idx["ColumnLeftBearingJoint_1"], self.idx["ColumnRightBearingJoint_1"]
        ia1, ith = self.idx.get("ArmLeftJoint_1"), self.idx.get("BaseJoint_1")
        n = max(1, int(secs / self.dt))
        kp = 1.2
        tol = 0.008
        # rad of turret yaw the insert may spend; the recorded place poses sit near -0.57 and a
        # healthy correction is tens of milliradians, so anything approaching this is a runaway, not
        # a correction
        _TH_SPAN = 0.35
        _a1_lo = _a1_hi = (float(np.asarray(self.robot.get_joint_positions(), float)[self.idx["ArmLeftJoint_1"]])
                           if "ArmLeftJoint_1" in self.idx else 0.0)
        _a1_in = _a1_lo
        eps, best = 1e-3, None
        q = np.asarray(self.robot.get_joint_positions(), float).copy()
        q_best = q.copy()
        th0 = float(q[ith]) if ith is not None else 0.0
        e0 = None
        for k in range(n):
            op = np.asarray(obj.get_world_poses()[0][0], float)
            err = np.asarray(target, float) - op
            e = float(np.linalg.norm(err))
            # Divergence guard. A closed-loop servo on a joint-space gradient can run away when the
            # gradient stops describing the arm, and this one does -- driving the turret through
            # more than a radian and putting the object hundreds of metres from the shelf, which
            # nothing downstream can recover from.
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
            # The columns are two DOFs, not one. Their COMMON mode is height -- equal motion
            # translates the whole four-bar 1:1 -- and their DIFFERENTIAL is boom tilt, which sets
            # forward REACH.
            d_mean = float(np.clip(kp * err[2] * self.dt * 20.0, -0.004, 0.004))
            d_diff = 0.0
            # xy: 2x2 Jacobian of the LUT grip position wrt (a1, th), solved least-squares
            da1 = dth = 0.0
            try:
                p0 = self._grip_fk(h1, h2, a1, th)
                ja = (self._grip_fk(h1, h2, a1 + eps, th) - p0)[:2] / eps
                jt = (self._grip_fk(h1, h2, a1, th + eps) - p0)[:2] / eps
                # Project, do not least-squares. A blind 2x2 solve dumps whatever a1 cannot supply
                # into th, so once a1 saturates the servo SWINGS the object sideways chasing a
                # radial error it can never close.
                na, nt = float(np.linalg.norm(ja)), float(np.linalg.norm(jt))
                if na > 1e-9:
                    da1 = float(np.clip(kp * float(err[:2] @ (ja / na)) / na
                                        * self.dt * 20.0, -0.004, 0.004))
                if nt > 1e-9:
                    dth = float(np.clip(kp * float(err[:2] @ (jt / nt)) / nt
                                        * self.dt * 20.0, -0.010, 0.010))
                # Tilt picks up what a1 cannot, and only once a1 is against its stop -- while a1 has
                # travel it is the cheaper actuator, because it does not disturb height. Off by
                # default pending a stable formulation, not because tilt is wrong. The reach trade
                # is real and worth having, especially for the high slot; what is wrong is THIS
                # servo's way of spending it. Read off the LUT at a1 0.50:
                #     dh   -0.100  0.000  +0.050  +0.110  +0.150
                #   reach   0.638  0.809   0.746   0.620   0.565
                #       z   1.256  0.835   0.570   0.395   0.350
                # Reach PEAKS at dh 0 and falls off both ways, so tilting either direction loses
                # reach; what negative tilt buys is HEIGHT. There is genuine reach headroom --
                # straightening from the place bake gains ~190mm -- but it drags 440mm of height
                # along with it, and the common-mode term is rate-clipped and cannot track a
                # coupling that violent, so the object leaves the target while the gradient keeps
                # asking for more. That is not fixable by gain tuning: the pair has to be SOLVED,
                # picking the (dh, common-mode) that hits reach and height together, as
                # `_solve_lowest` does for the descent.
                _a1c = A1_MAX
                if (os.environ.get("INSERT_TILT", "0") == "1"
                        and a1 >= _a1c - 1e-3 and na > 1e-9):
                    _resid = float(err[:2] @ (ja / na))    # radial error a1 must leave behind
                    if _resid > 1e-4:                      # only to reach FURTHER, never nearer
                        _d = 2e-3
                        jd = (self._grip_fk(h1 - 0.5 * _d, h2 + 0.5 * _d, a1, th) - p0) / _d
                        _rad = float(jd[:2] @ (ja / na))   # radial gain of straightening
                        if abs(_rad) > 1e-6:
                            d_diff = float(np.clip(kp * _resid / _rad * self.dt * 20.0,
                                                   -0.004, 0.004))
                            # straightening moves z too; cancel it in the common mode so the object
                            # holds its height while the boom trades tilt for reach
                            d_mean = float(np.clip(d_mean - 0.5 * d_diff * float(jd[2]),
                                                   -0.006, 0.006))
            except Exception:
                pass
            q[ih1] = float(np.clip(h1 + d_mean - 0.5 * d_diff, 0.0, COLUMN_MAX))
            q[ih2] = float(np.clip(h2 + d_mean + 0.5 * d_diff, 0.0, COLUMN_MAX))
            if ia1 is not None:
                q[ia1] = float(np.clip(a1 + da1, 0.0, A1_MAX))
            if ith is not None:
                # Clamp the turret. Every other joint here is bounded and th was not, so it was free
                # to integrate a step's worth per sample for the whole servo -- radians over the
                # lower phase, which is what it spent when it ran away.
                q[ith] = float(np.clip(th + dth, th0 - _TH_SPAN, th0 + _TH_SPAN))
            hold_fn(q)
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
