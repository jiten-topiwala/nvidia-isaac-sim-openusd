"""Base and articulation write primitives — the lowest layer everything else calls.

`set_base` is the only thing that moves the chassis, `_force`/`_apply` the only things that
write joint commands, and the ledger is the single source of truth for where the base is.

Split out of play_isaac.py; the method bodies are unchanged. Imports Isaac APIs at module
level, so it is only importable after SimulationApp exists — see `morph/__init__.py`.
"""
import math
import os

import numpy as np
import scene
from isaacsim.core.utils.types import ArticulationAction

from morph.config import (KNOWN, PERFECT, HEADLESS, VMAX)
from morph.geometry import quat_yaw, yaw_of, wrap


class RobotMixin:
    """Mixed into Demo. Every `self.*` it touches is owned by Demo."""

    def base_pose(self):
        p, q = self.robot.get_world_pose()
        return float(p[0]), float(p[1]), yaw_of(q)

    def base_ledger(self):
        """Intended base pose (exact): what WE last commanded, immune to read lag."""
        return getattr(self, "_bl", None) or self.base_pose()

    def _spin_wheels_mecanum(self, fwd, lat, yaw_d):
        """Roll each wheel by what a mecanum base would need to make THIS body step.

        theta_i += (fwd -+ lat -+ yaw*D) / r, the standard mecanum mixer. Cosmetic here -- the
        base is still moved by
        `set_world_pose`, not by wheel torque -- but it is now DERIVED from the motion actually
        achieved rather than approximated from its magnitude, so the wheels cannot disagree with
        what the chassis did.

        IF "THE WHEELS DO NOT TURN" IS REPORTED AGAIN, IT IS PROBABLY ALIASING. Measured, so it
        does not have to be chased a fifth time:
          - the joints track the body. Wheel roll against chassis travel over a full cycle:
            transport 1.702 -> 7.701m travelled against 2.814 -> 9.909m rolled, climbing together
            the whole way including the corridor turn, cmd == act every sample. They are frozen
            only while `pin_chassis` holds the base, which is correct. (WHEEL_DBG=1 reprints this.)
          - the hierarchy is right. `<side>_wheel_rolling_joint` drives base ->
            wheel_intermediate_link, the hub written here, and TWELVE `slipping_N_joint`s hang
            roller_0..11_link off that hub, so the rollers ride it.
          - twelve rollers means the wheel looks IDENTICAL every 30deg. At NAV_VMAX 32 the wheel
            turns 13.5-21.2 deg per frame on a 60Hz display, against a Nyquist limit of 15 for a
            30deg pattern -- at or past the point where direction is ambiguous. At the original
            NAV_VMAX 8 it was ~3-5 deg/frame and plainly visible. RENDER_EVERY does not help; the
            monitor is 60Hz however many frames Isaac draws.
        Cosmetic, and driven by the cruise speed. Slowing nav is NOT the practical fix: the base is
        acceleration-limited on these short legs and never approaches VMAX, so the real ground speed
        is ~1.2-2.2 m/s, and getting the wheel under 10 deg/frame would need it below 1.05 m/s --
        slower than the robot moved before any of the speed work. Hence the rate cap below.
        """
        if not hasattr(self, "_wheel_ii"):
            names = ["front_left_wheel_rolling_joint", "front_right_wheel_rolling_joint",
                     "back_left_wheel_rolling_joint", "back_right_wheel_rolling_joint"]
            self._wheel_ii = [(n, self.idx[n]) for n in names if n in self.idx]
            self._wheel_a = {n: 0.0 for n, _ in self._wheel_ii}
            print(f">>> wheel spin: {len(self._wheel_ii)} rolling joints (mecanum mixer)", flush=True)
        if not self._wheel_ii:
            return
        # No wheel motion while the chassis is pinned. A pin re-asserts the base whenever physics
        # nudges it, and each re-assert arrives here as a position delta -- so a small drift
        # correction spins the wheels through a visible angle in one step.
        if getattr(self, "_pin_orig_step", None) is not None:
            try:
                _pi = np.array([i for _, i in self._wheel_ii])
                if len(_pi):
                    self.robot.set_joint_velocities(np.zeros(len(_pi)), joint_indices=_pi)
            except Exception:
                pass
            # Drop the jump tracker's anchor. The diagnostic below samples the wheel angle every
            # call, so with this early return the first sample AFTER a pin would be differenced
            # against the last one BEFORE it and reported as a single-frame jump -- a measurement
            # artefact, not motion.
            self._ws_prev_q = None
            return
        r = float(os.environ.get("WHEEL_R", "0.1"))                 # wheel radius, m
        D = 0.55                                                    # wheelbase + track, m
        mix = {"front_left_wheel_rolling_joint": (1.0, -1.0, -1.0),
               "front_right_wheel_rolling_joint": (1.0, 1.0, 1.0),
               "back_left_wheel_rolling_joint": (1.0, 1.0, -1.0),
               "back_right_wheel_rolling_joint": (1.0, -1.0, 1.0)}
        try:
            # Visual rate map. Above roughly 15 degrees per rendered frame the twelve-roller pattern
            # aliases and the wheel reads as stationary or backwards -- ordinary wagon-wheel
            # aliasing.
            _wvc = 6.0
            q = np.asarray(self.robot.get_joint_positions(), float)
            for n, i in self._wheel_ii:
                cf, cl, cw = mix[n]
                _d = (cf * fwd + cl * lat + cw * yaw_d * D) / r
                if _wvc > 0.0:
                    _rate = _d / self.dt
                    _d = _wvc * math.tanh(_rate / _wvc) * self.dt
                # Relative and wrapped write. Accumulating an absolute angle desynchronises from the
                # real joint wherever the wheels are held (a pin, a drift re-assert), so the next
                # drive frame teleports the wheel by the whole gap -- an invisible flick, then
                # normal rolling -- and the value grows unbounded, losing float resolution.
                q[i] = (float(q[i]) + _d + math.pi) % (2.0 * math.pi) - math.pi
                self._wheel_a[n] += _d
            ii = np.array([i for _, i in self._wheel_ii])
            self.robot.set_joint_positions(q[ii], joint_indices=ii)
            # Zero the wheel velocity, but ONLY while the chassis is stationary. Parked, the
            # position write holds the wheels while a residual velocity remains, so the solver
            # integrates them between writes and the next write snaps them back -- which the render
            # catches mid-integration.
            if os.environ.get("WHEEL_VEL_ZERO", "1") == "1" and \
                    abs(fwd) < 1e-6 and abs(lat) < 1e-6 and abs(yaw_d) < 1e-6:
                self.robot.set_joint_velocities(np.zeros(len(ii)), joint_indices=ii)
            # Commanded against actual. The write succeeding and the mixer being right does not mean
            # the joint moved: if `act` does not follow `cmd` something is refusing the write, which
            # no amount of fixing the mixer would reveal.
            if os.environ.get("WHEEL_DBG", "0") == "1":
                # Against the chassis, not just against itself: the question is whether the wheels
                # turn by what the BODY travelled.
                self._ws_n = getattr(self, "_ws_n", 0) + 1
                self._ws_travel = getattr(self, "_ws_travel", 0.0) + math.hypot(fwd, lat)
                self._ws_roll = getattr(self, "_ws_roll", 0.0) + abs(
                    self._wheel_a[self._wheel_ii[0][0]] - getattr(self, "_ws_prev_a", 0.0)) * r
                self._ws_prev_a = self._wheel_a[self._wheel_ii[0][0]]
                # Biggest single-frame joint jump. The absolute-accumulator bug showed up as a 6.84
                # rad one-frame teleport after every pin; track the worst so a regression cannot
                # hide.
                _jn, _ji = self._wheel_ii[0]
                _now = float(np.asarray(self.robot.get_joint_positions(), float)[_ji])
                _pv = getattr(self, "_ws_prev_q", None)
                if _pv is not None:
                    _jump = abs((_now - _pv + math.pi) % (2 * math.pi) - math.pi)
                    self._ws_maxjump = max(getattr(self, "_ws_maxjump", 0.0), _jump)
                self._ws_prev_q = _now
                # Per-wheel rates: during combined translate+turn a mecanum wheel's mix term
                # legitimately crosses ZERO -- that wheel truly stops while the chassis moves.
                _rates = {n2: (mix[n2][0] * fwd + mix[n2][1] * lat
                               + mix[n2][2] * yaw_d * D) / r / self.dt
                          for n2, _ in self._wheel_ii}
                if self._ws_n % 120 == 0:
                    _qa = np.asarray(self.robot.get_joint_positions(), float)
                    _n0, _i0 = self._wheel_ii[0]
                    print(f">>>   wheel[{self._ws_n}] cmd {self._wheel_a[_n0]:+.2f} act "
                          f"{float(_qa[_i0]):+.2f} rad | chassis travel "
                          f"{self._ws_travel:.3f}m  wheel roll {self._ws_roll:.3f}m  "
                          f"ratio {self._ws_roll / max(self._ws_travel, 1e-9):.2f} | per-wheel "
                          f"rad/s min {min(_rates.values()):+.1f} max {max(_rates.values()):+.1f} "
                          f"| worst 1-frame jump {getattr(self, '_ws_maxjump', 0.0):.2f} rad",
                          flush=True)
        except Exception as _e_ws:
            # Say it once. As a bare `except: pass` this cannot be told apart from the mixer never
            # being called -- and a swallowed write is exactly the failure the logs could not
            # distinguish.
            if not getattr(self, "_ws_warned", False):
                self._ws_warned = True
                print(f">>> wheel spin: WRITE FAILED ({_e_ws}) -- wheels will not turn; "
                      f"the mixer is fine, the joint write is not", flush=True)

    def _spin_wheels(self, dist, yaw_d):
        """Roll the four wheel joints by the distance the chassis just travelled.

        The base is teleported with `set_world_pose`, never driven by wheel torque, so left alone
        the wheels are geometrically stationary and the robot visibly slides. Rolling them from
        the travelled distance is the honest visual: theta += ds / r, plus a differential term so
        an in-place yaw still counter-rotates left and right.
        """
        if not hasattr(self, "_wheel_ii"):
            names = ["front_right_wheel_rolling_joint", "front_left_wheel_rolling_joint",
                     "back_right_wheel_rolling_joint", "back_left_wheel_rolling_joint"]
            self._wheel_ii = [(n, self.idx[n]) for n in names if n in self.idx]
            self._wheel_a = 0.0
            print(f">>> wheel spin: {len(self._wheel_ii)} rolling joints", flush=True)
        if not self._wheel_ii:
            return
        r = float(os.environ.get("WHEEL_R", "0.127"))
        half = 0.30 / 2.0
        self._wheel_a += dist / r
        try:
            q = np.asarray(self.robot.get_joint_positions(), float)
            for n, i in self._wheel_ii:
                side = -1.0 if "left" in n else 1.0          # yaw counter-rotates the two sides
                q[i] = self._wheel_a + side * yaw_d * half / r
            ii = np.array([i for _, i in self._wheel_ii])
            self.robot.set_joint_positions(q[ii], joint_indices=ii)
        except Exception:
            pass

    def set_base(self, x, y, yaw):
        # The no-chassis-push rule, enforced rather than assumed: during a grasp the chassis is
        # stationary and every gap is closed with the ARM.
        if getattr(self, "_grasp_locked", None) is not None:
            _px, _py, _pyaw = self._grasp_locked
            if (abs(float(x) - _px) > 1e-6 or abs(float(y) - _py) > 1e-6
                    or abs(wrap(float(yaw) - _pyaw)) > 1e-6):
                _n = getattr(self, "_base_block_n", {})
                try:
                    import traceback
                    _who = traceback.extract_stack()[-2]
                    _key = f"{os.path.basename(_who.filename)}:{_who.lineno}"
                except Exception:
                    _key = "?"
                if _key not in _n:
                    print(f">>> chassis move REFUSED during grasp ({_key} wanted "
                          f"{float(x) - _px:+.3f},{float(y) - _py:+.3f}m) -- close this gap "
                          f"with the ARM, never the base", flush=True)
                _n[_key] = _n.get(_key, 0) + 1
                self._base_block_n = _n
                return
        # Base drift watchdog: before re-pinning, compare the LIVE pose against the ledger -- any
        # gap is real chassis motion physics produced between pins.
        if os.environ.get("BASE_WATCH", "1") == "1" and getattr(self, "_bl", None) is not None:
            try:
                # Only when re-pinning. During nav and dock the caller is MOVING the base, so the
                # live pose lagging the ledger is tracking, not drift.
                if (abs(float(x) - self._bl[0]) > 1e-6 or abs(float(y) - self._bl[1]) > 1e-6
                        or abs(wrap(float(yaw) - self._bl[2])) > 1e-6):
                    raise StopIteration          # intentional motion -> skip the check
                _lx, _ly, _lyaw = self.base_pose()
                _ddx, _ddy = _lx - self._bl[0], _ly - self._bl[1]
                _dwz = wrap(_lyaw - self._bl[2])
                _dm = math.hypot(_ddx, _ddy)
                if (_dm > 0.008 or abs(_dwz) > 0.015):
                    self._bw_n = getattr(self, "_bw_n", 0) + 1
                    # Quiet by default: this fires per detection while unpinned, which at 240Hz
                    # floods the terminal, and it is redundant -- every pinned stage prints a
                    # released-with-N-re-asserts summary with the same count.
                    if os.environ.get("BASE_DRIFT_DBG", "0") == "1" and (
                            self._bw_n <= 3 or self._bw_n % 60 == 0):
                        try:
                            import traceback
                            _w = traceback.extract_stack()[-2]
                            _site = f"{os.path.basename(_w.filename)}:{_w.lineno}"
                        except Exception:
                            _site = "?"
                        print(f">>> BASE DRIFT #{self._bw_n}: chassis moved "
                              f"{_dm * 1000:.0f}mm / {math.degrees(_dwz):+.1f}deg off the ledger "
                              f"between pins (dx {_ddx * 1000:+.0f} dy {_ddy * 1000:+.0f}mm, "
                              f"caught at {_site})", flush=True)
            except Exception:
                pass
        if os.environ.get("WHEEL_SPIN", "1") == "1" and getattr(self, "_bl", None) is not None:
            # Per-wheel roll from the real 2D motion, through the mecanum mixer, so a sideways step
            # turns the wheels the way a mecanum base does and reversing turns them backwards.
            _dx, _dy = float(x) - self._bl[0], float(y) - self._bl[1]
            _hd = self._bl[2]
            _f = _dx * math.cos(_hd) + _dy * math.sin(_hd)          # body forward component
            _l = -_dx * math.sin(_hd) + _dy * math.cos(_hd)         # body lateral component
            self._spin_wheels_mecanum(_f, _l, wrap(float(yaw) - self._bl[2]))
        self._bl = (float(x), float(y), float(yaw))          # GLOBAL base ledger: base_pose() reads
        # The global base ledger. `base_pose()` reads are stale for a beat after any `set_base`, so
        # a loop that re-reads it teleports the base back -- the source of the close-time palm
        # shoves.
        try:
            _zl = float(self.robot.get_world_pose()[0][2])
            _zpin = getattr(self, "_pin_z", None)
            if _zpin is not None and os.environ.get("PIN_Z_HARD", "1") == "1":
                _zp = float(_zpin)
            else:
                _zp = (_zl if abs(_zl - self.z0) < 0.02
                       else self.z0)
        except Exception:
            _zp = self.z0
        self.robot.set_world_pose(position=[x, y, _zp], orientation=quat_yaw(yaw))
        # ALSO zero the base ROOT velocity — set_world_pose sets position only, so residual velocity
        # from a rack contact keeps pushing the base (it drifted 2.5m south INTO the rack between
        # cycles -> next nav start invalid).
        try:
            self.robot.set_velocities(np.zeros((1, 6)))
        except Exception:
            try:
                self.robot.set_velocities(np.zeros(6))
            except Exception:
                pass

    def _with_closure(self, q):
        """Write the parallel linkage's PASSIVE joints for whatever (h2-h1, a1) is being commanded.

        THE ARM RUNS `NO_LOOP`: the 4-bar closure is NOT simulated, so its passive joints come from
        `usd/_closure_lut.json` -- and those passives are what actually ANGLE THE BOOM. Commanding
        the two column joints alone barely tilts it, because the right column reaches the boom only
        through a closure that is not being solved (`robot.usda`: Bearing_Column_Right_1 ->
        Rotation_Link_Right_1 is a PhysicsFixedJoint; the arm hangs off the LEFT column).

        `_closure_passives` existed but was called from exactly ONE place -- `align.py:684` -- so
        every other stage commanded a tilt that never materialised. Measured: the descent asked for
        dh +0.150 and the joints read back +0.083 with no guard firing and nothing stopping the ramp.
        It also explains why physically probing the tilt produced three successive wrong direction
        conclusions: the probe moved dh and the linkage did not follow.

        Applied in the write primitives so EVERY caller gets it, rather than adding a fourth
        hand-rolled copy. `CLOSURE_AUTO=0` disables.
        """
        # Perfect mode only. Practical mode is a shipped, working configuration whose baked replays
        # carry RECORDED passive values; overwriting those with LUT values would change a proven
        # path for no reason.
        q = np.asarray(q, float).copy()
        if (not PERFECT or os.environ.get("CLOSURE_AUTO", "1") != "1"
                or not getattr(self, "clut", None)):
            return q
        i_h1 = self.idx.get("ColumnLeftBearingJoint_1")
        i_h2 = self.idx.get("ColumnRightBearingJoint_1")
        i_a1 = self.idx.get("ArmLeftJoint_1")
        if i_h1 is None or i_h2 is None or i_a1 is None or len(q) <= max(i_h1, i_h2, i_a1):
            return q
        try:
            for _n, _v in self._closure_passives(float(q[i_h2] - q[i_h1]), float(q[i_a1])).items():
                _j = self.idx.get(_n)
                if _j is not None and _j < len(q):
                    q[_j] = float(_v)
        except Exception:
            pass
        return q

    def _force(self, q, qv=None):
        """Kinematic joint write, grasp-mode aware.  The ARM (NO_LOOP parallel linkage) is ALWAYS
        forced.  In friction mode, while the fingers own an object (_finger_cmd set), the finger
        DOFs are EXCLUDED — they run on force-limited PD drives so real contact can stop them;
        kinematically forcing a finger into a colliding object is an infinite-force fight.
        qv: CONSISTENT joint velocities for a MOVING kinematic write.  PhysX friction is
        velocity-based — a position-teleported link with v=0 is a stationary wall to the solver,
        so a zero-velocity 'lift' can squeeze an object at 200N and still leave it on the floor
        (the kinematic-platform trap).  Every replay that must CARRY contact passes qv."""
        q = self._with_closure(q)
        if PERFECT and self._finger_cmd is not None:
            ii = self._arm_ii
            self.robot.set_joint_positions(q[ii], joint_indices=ii)
            self.robot.set_joint_velocities(
                np.zeros(len(ii)) if qv is None else qv[ii], joint_indices=ii)
        else:
            # ...and never write the WHEELS/ROLLERS: they spin freely, and writing the hold vector's
            # parked angles here is what erased the mecanum roll every step.
            ii = getattr(self, "_hold_ii", None)
            if ii is None:
                self.robot.set_joint_positions(q)
                self.robot.set_joint_velocities(
                    np.zeros(len(self.names)) if qv is None else qv)
            else:
                self.robot.set_joint_positions(q[ii], joint_indices=ii)
                self.robot.set_joint_velocities(
                    np.zeros(len(ii)) if qv is None else qv[ii], joint_indices=ii)

    def _apply(self, q):
        """apply_action that SURVIVES a dropped articulation view.  In headed mode the user can
        pause/stop the viewport (screenshots!) or other USD churn can invalidate the physics view —
        apply_action then dies with `applied_actions None`, the drives go dead and the arm visibly
        collapses into the chassis. Revive the view once and retry rather than crashing the cycle.

        While the fingers own an object, their drive targets come from `_finger_cmd` — the squeeze
        and curl targets — regardless of what the caller's hold vector says: a hold vector frozen
        at the contact surface would command essentially zero grip force.
        """
        q = self._with_closure(q)                 # linkage passives, same rule as `_force`
        if PERFECT and self._finger_cmd is not None:
            q = np.asarray(q, float).copy()
            q[self._f_idx] = self._finger_cmd
        # Do not drive the wheels to a position either. `scene.set_arm_gains` leaves the rolling
        # joints at kp=0/kd=1 (a velocity drive), so a position target here fights the free spin the
        # mecanum roll writes -- the other half of "the wheels do not turn".
        _rii = getattr(self, "_roll_ii", None)
        if _rii is not None and len(_rii):
            _live = np.asarray(self.robot.get_joint_positions(), float)
            # A dropped articulation view comes back as a 0-d array, not as an AttributeError -- so
            # the retry below, which only catches AttributeError, would index the scalar and take
            # the whole pick down.
            if _live.ndim == 1 and _live.size > int(np.max(_rii)):
                q = np.asarray(q, float).copy()
                q[_rii] = _live[_rii]                        # command them where they already are
        try:
            self.ctrl.apply_action(ArticulationAction(joint_positions=q))
        except AttributeError:
            print(">>> articulation view dropped (viewport paused/stopped?) -> revive + retry", flush=True)
            try:
                self.robot.initialize()
            except Exception:
                # the user STOPPED the timeline: the physics simulation view itself is destroyed
                # (create_articulation_view on None) and stopping also resets the sim state.
                import omni.timeline
                omni.timeline.get_timeline_interface().play()
                for _ in range(3):
                    self.world.step(render=not HEADLESS)
                self.robot.initialize()
            self.ctrl = self.robot.get_articulation_controller()
            # Re-assert the base after the revive. Stopping the timeline also resets the sim state,
            # so without this the robot stands wherever that reset left it.
            try:
                _bx_r, _by_r, _byaw_r = (self._grasp_locked
                                         if getattr(self, "_grasp_locked", None) is not None
                                         else self.base_ledger())
                self.set_base(_bx_r, _by_r, _byaw_r)
                print(f">>> revive: base re-asserted at ({_bx_r:.3f}, {_by_r:.3f})", flush=True)
            except Exception as _e_rv:
                print(f">>> revive: base re-assert failed ({_e_rv})", flush=True)
            scene.set_arm_gains(self.robot, self.names)
            self.ctrl.apply_action(ArticulationAction(joint_positions=q))

    def park(self):
        self.robot.set_joint_positions(self.q0)
        for _ in range(int(0.3 / self.dt)):
            self._force(self.q0)   # kinematic: NO_LOOP mode has no loop joint to   # hold the free passives under dynamics
            self._apply(self.q0)
            self.world.step(render=not HEADLESS)

    def _hold(self, cmd, steps, pin=None, kin=False, on_step=None):
        """Hold base + arm fixed for `steps`.  kin=True KINEMATICALLY pins the arm (force joint positions
        + zero velocity every step) so the forced grasp config can't oscillate against the linkage loop
        (kills the settle shake); the drive still commands cmd underneath."""
        bx, by, byaw = self.base_ledger()
        dbg = os.environ.get("HOLD_DBG") == "1"
        for k in range(steps):
            self.set_base(bx, by, byaw)
            if pin is not None and not PERFECT:            # friction: the object is a real body
                obj, pos = pin                              # collision-off dynamic puppet -> pin pose + velocity
                obj.set_world_poses(positions=np.array([pos]), orientations=np.array([[1, 0, 0, 0]]))
                obj.set_velocities(np.zeros((1, 6)))
            if kin:
                self._force(cmd)
            self._apply(cmd)
            if on_step is not None:
                on_step()
            self.world.step(render=not HEADLESS)
            if dbg:
                q = np.asarray(self.robot.get_joint_positions(), float)
                bad = np.where(~np.isfinite(q))[0]
                if len(bad):
                    print(f">>> HOLD_DBG step {k}: {len(bad)} non-finite DOFs, first: "
                          f"{[(self.names[i]) for i in bad[:6]]}", flush=True)
                    return

    def _park_ramp(self, base=None, step_fn=None, t=1.0, note=""):
        """Cosine-ramp the arm from its live pose into the park pose `q0`, closure passives
        written every step so the 4-bar stays consistent. `base` (x, y, yaw) is held during the
        ramp; `step_fn(qk)` replaces the default write+step when the caller owns the stepping
        (the place stage re-pins the object each frame). Skips entirely if already at park."""
        q_now = np.asarray(self.robot.get_joint_positions(), float).copy()
        q_tgt = np.asarray(self.q0, float)
        if t <= 0 or float(np.max(np.abs(q_tgt - q_now))) <= 1e-3:
            return
        i1 = self.idx.get("ColumnLeftBearingJoint_1")
        i2 = self.idx.get("ColumnRightBearingJoint_1")
        ia = self.idx.get("ArmLeftJoint_1")
        n = max(2, int(t / self.dt))
        for k in range(n):
            f = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / n)
            qk = q_now + f * (q_tgt - q_now)
            if getattr(self, "clut", None) is not None and None not in (i1, i2, ia):
                for pn, pv in self._closure_passives(
                        float(qk[i2]) - float(qk[i1]), float(qk[ia])).items():
                    if pn in self.idx:
                        qk[self.idx[pn]] = float(pv)
            if step_fn is not None:
                step_fn(qk)
            else:
                self.set_base(*base)
                self._force(qk)
                self._apply(qk)
                self.world.step(render=not HEADLESS)
        print(f">>> park: ramped to the park pose over {t:.1f}s{note}", flush=True)

    def idle_step(self):
        # hold the base at a FIXED anchor while idle — re-reading base_pose() each step integrates
        # whatever the physics pushes (loop-joint/wheel residuals moved the parked base 0.5-2.1m per
        # 0.5s idle window, cycle 3 drifting INTO the rack -> next nav "No path").
        if self._idle_anchor is None:
            self._idle_anchor = self.base_pose()
        bx, by, byaw = self._idle_anchor
        self.set_base(bx, by, byaw)
        if not PERFECT:                                     # friction: gravity + the shelf board hold them
            for oi, pos in self.placed:                      # practical mode: re-pin (objects are collision-off)
                self.objs[oi].set_world_poses(positions=np.array([pos]), orientations=np.array([[1, 0, 0, 0]]))
                self.objs[oi].set_velocities(np.zeros((1, 6)))
        # FREEZE the robot kinematically while idle — after a place it sits at the rack dock where
        # the parked arm can graze a shelf; frozen (forced q0 + zero velocity) it physically cannot
        # explode.
        if self.grip is None:
            # Ramp into park on the first idle step rather than teleporting: the cycle ends holding
            # the retracted pose, so writing q0 in one step makes the arm visibly snap.
            if not getattr(self, "_parked_ramped", False):
                self._parked_ramped = True
                self._park_ramp(base=(bx, by, byaw))
            self._force(self.q0)
        self._apply(self.grip if self.grip is not None else self.q0)
        self.world.step(render=not HEADLESS)

    def _grip_stiffen(self, arm_kp=None):
        """Post-touch hold stiffness.

        Left at the approach gains, the distal joints bend backward at contact — the drives are
        too soft to resist their own grip, so the pads flex away and the cage carries no force. A
        real gripper's gearbox is stiff; emulate that AFTER the gentle touch. The order matters:
        freeze on contact first, stiffen second, so the higher force can only press where contact
        already exists.
        """
        kp = np.full(len(self.names), 2.0e3)
        kd = np.full(len(self.names), 2.0e2)
        eff = np.full(len(self.names), 1.0e6)
        for i, n in enumerate(self.names):
            if n.startswith(("finger_", "palm_finger")):
                # One source of truth for finger dynamics: `scene.finger_pd`, from the source
                # model's actuator and joint numbers.
                kp[i], kd[i], _ = scene.finger_pd(n)
                # 25Nm, the figure this docstring always cited, restored from an unbounded 1.0e6.
                eff[i] = float(os.environ.get("HOLD_EFFORT", "25"))
            elif "rolling_joint" in n or "slipping" in n:
                kp[i], kd[i] = 0.0, 1.0
        # Arm last, and at the CALLER's stiffness. Dropping every arm joint outside the columns to a
        # soft gain here silently undoes the `_set_arm_gain` that runs just before it, so the whole
        # close phase runs on a soft arm -- the hand droops centimetres while the object never
        # moves, and the pads close somewhere the object no longer is.
        arm_kp = 3.0e5 if arm_kp is None else arm_kp
        # The arm needs a torque limit as much as the fingers do. Left unlimited, a stiff arm drive
        # is the same infinite-mass wall as a state write: during the squeeze it answers a
        # millimetre of blocked travel with hundreds of newtons at the pad and fires the object
        # away.
        arm_eff = 1.0e6
        for n in KNOWN["arm_joints"]:
            if n in self.idx:
                kp[self.idx[n]], kd[self.idx[n]] = arm_kp, arm_kp * 0.1
                eff[self.idx[n]] = arm_eff
        self.ctrl.set_gains(kp, kd)
        try:
            self.ctrl.set_max_efforts(eff)
        except Exception:
            pass
        print(">>> grip stiffened (hold gains: fingers kp 6e3, effort 25Nm)", flush=True)

    def _grip_gentle(self):
        """APPROACH gains: soft fingers, arm unchanged.  The counterpart to _grip_stiffen, which
        its own docstring already prescribes ("Safe order: touch-freeze first, stiffen second") but
        which the code called BEFORE the close loop — so the whole approach ran at HOLD_EFFORT=25Nm,
        ~250N at the pad.

        Measured consequence (CLOSE_TRACE, failing cycle): at k=350 all three fingers are 4-8mm off
        a still-upright object; 25 steps later the object has jumped 17mm and tilted 9deg and b/c
        are 29/40mm out.  17mm in 0.1s is 170mm/s while the pad closes at only ~26mm/s — the object
        is accelerated far faster than the finger moves, i.e. first touch is an IMPACT, not a press.
        No stop can react inside that window; the fix has to be to arrive gently.
        kd comes down with kp, or the finger cannot move at all: at kd=40 merely travelling at
        0.29 rad/s costs 11Nm of damping, which is why HOLD_EFFORT=1.0 never reached."""
        kp = np.full(len(self.names), 2.0e3)
        kd = np.full(len(self.names), 2.0e2)
        eff = np.full(len(self.names), 1.0e6)
        g_kp = 30.0
        # 5.0 still cost 1.45Nm of damping at the replay rate — more than the whole sub-tipping
        # torque budget
        g_kd = 1.0
        # 0.40 was BELOW the finger's own gravity load (0.64-1.05Nm measured) and sagged j1 to the
        # -1.5708 limit  # ~4.4N at the
        g_ef = 5.0
        # 0.09m lever,
        #   under the 5.4N that tips this cylinder
        for i, n in enumerate(self.names):
            if n.startswith(("finger_", "palm_finger")):
                kp[i], kd[i], eff[i] = g_kp, g_kd, g_ef
            elif "rolling_joint" in n or "slipping" in n:
                kp[i], kd[i] = 0.0, 1.0
        arm_kp = 3.0e5
        for n in KNOWN["arm_joints"]:
            if n in self.idx:
                kp[self.idx[n]], kd[self.idx[n]] = arm_kp, arm_kp * 0.1
                eff[self.idx[n]] = 1.0e6
        self.ctrl.set_gains(kp, kd)
        try:
            self.ctrl.set_max_efforts(eff)
        except Exception:
            pass
        print(f">>> grip GENTLE for approach (fingers kp {g_kp} kd {g_kd} effort {g_ef}Nm)",
              flush=True)

    def _set_arm_gain(self, arm_kp):
        """Set the forced arm-1 joints to arm_kp (kd=0.1·kp); rollers free; everything else modest."""
        kp = np.full(len(self.names), 2.0e3)
        kd = np.full(len(self.names), 2.0e2)
        for n in KNOWN["arm_joints"]:
            if n in self.idx:
                kp[self.idx[n]], kd[self.idx[n]] = arm_kp, arm_kp * 0.1
        for i, n in enumerate(self.names):
            if "rolling_joint" in n or "slipping" in n:
                kp[i], kd[i] = 0.0, 1.0
        self.ctrl.set_gains(kp, kd)
