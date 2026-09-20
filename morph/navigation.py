"""OMPL base navigation: plan a path in a subprocess, then drive the chassis along it."""
import json
import math
import os
import subprocess


from morph.config import (HERE, VENV_PY, NAV_PLAN, HEADLESS, VMAX, WMAX, KP_LIN, KP_ANG,
    WP_TOL, GOAL_TOL)
from morph.geometry import wrap


def arrival_error(goal_xy, goal_yaw, pose):
    """How far a finished drive stopped from its GOAL: (mm, degrees). The drive loops track an
    internal reference pose that is clamped to the live pose, so reference-vs-live cannot say
    whether the base arrived."""
    x, y, yaw = pose
    return (math.hypot(goal_xy[0] - x, goal_xy[1] - y) * 1000.0,
            math.degrees(wrap(goal_yaw - yaw)))


def nav_endpoint_shift(start_xy, goal_xy, path):
    """How far the planner's endpoints sit from the ones it was asked for: (start_mm, goal_mm), or
    None if the path has no segment. `nudge` relocates an invalid endpoint by up to 0.8 m and the
    plan is then rooted somewhere the robot is not."""
    if len(path) < 2:
        return None
    return (math.hypot(path[0][0] - start_xy[0], path[0][1] - start_xy[1]) * 1000.0,
            math.hypot(path[-1][0] - goal_xy[0], path[-1][1] - goal_xy[1]) * 1000.0)


def plan_path(start_xy, goal_xy, solve_time=3.0, tries=3, discs=None):
    """OMPL RRTConnect via nav_plan.py, run in the project venv (has ompl); retried with more time
    because RRTConnect can time out from a cornered start. discs: [[x, y, r]] keep-out circles in
    metres (floor objects) -- objects are real bodies and the chassis would shove them."""
    last = None
    for t in range(tries):
        req = json.dumps({"start": list(start_xy), "goal": list(goal_xy),
                          "solve_time": solve_time * (1 + t), "discs": discs or []})
        # Isaac's PYTHONPATH/PYTHONHOME point at its 3.12 stdlib; inheriting them makes the venv's
        # 3.10 import the wrong `re` ("SRE module mismatch"). Strip them for the child.
        env = {k: v for k, v in os.environ.items()
               if k not in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH")}
        out = subprocess.run([VENV_PY, NAV_PLAN], input=req, capture_output=True, text=True,
                             cwd=HERE, env=env)
        lines = [ln for ln in out.stdout.strip().splitlines() if ln.strip()]
        if lines:
            data = json.loads(lines[-1])
            if not isinstance(data, dict):
                return data
            last = data
        else:
            last = {"error": f"no output. stderr:\n{out.stderr[-300:]}"}
    raise RuntimeError(f"no path after {tries} tries: {last}")


class NavigationMixin:
    """Mixed into Demo. Every `self.*` it touches is owned by Demo."""

    def _wheel_move(self, x1, y1, yaw1, vmax, hold_q, carry_obj=None, tag="leg"):
        """Straight base move ON THE WHEELS (no planner, no root write): an accel-limited
        reference from the live pose to (x1, y1, yaw1) in world metres/radians, tracked with the
        same hub-rate command and P-correction as `drive_to`'s wheel branch. Returns the live pose."""
        _kc = float(os.environ.get("NAV_TRACK_KP", "3.0"))
        _vcap = float(os.environ.get("NAV_TRACK_VCAP", "0.08"))   # above this a lateral step reversed a hub
        # Wheel-mode caps, MEASURED, and set by grip CREEP during the accel/decel transients rather than by
        # traction. Raise them only with a re-seat before the insert, or by slowing the final approach.
        _a = float(os.environ.get("NAV_ACC_WD", "2.0"))
        _aw = float(os.environ.get("NAV_AW_WD", "2.0"))
        _wm = float(os.environ.get("NAV_WMAX_WD", "0.6"))
        vmax = min(float(vmax), float(os.environ.get("NAV_VMAX_WD", "0.5")))
        self._pin_brake_apply(True)
        self._hub_effort(float(os.environ.get("NAV_HUB_EFFORT", "20")))
        x, y, yaw = self.base_pose()
        x0, y0, yaw0 = x, y, yaw
        vx = vy = wz = 0.0
        d_tot = math.hypot(x1 - x0, y1 - y0)
        n_max = int((d_tot / max(vmax, 1e-3) + abs(wrap(yaw1 - yaw0)) / max(_wm, 1e-3) + 3.0) / self.dt)
        _k = 0
        for _k in range(n_max):
            ex, ey = x1 - x, y1 - y
            dist = math.hypot(ex, ey)
            yerr = wrap(yaw1 - yaw)
            _lx, _ly, _lyaw = self.base_pose()
            if dist < 0.003 and abs(yerr) < 0.01 and math.hypot(x1 - _lx, y1 - _ly) < 0.01:
                break
            sp = min(vmax, 0.9 * math.sqrt(2.0 * _a * dist)) if dist > 1e-6 else 0.0
            tvx, tvy = (sp * ex / dist, sp * ey / dist) if dist > 1e-6 else (0.0, 0.0)
            twz = (math.copysign(min(_wm, 0.9 * math.sqrt(2.0 * _aw * abs(yerr))), yerr)
                   if abs(yerr) > 1e-9 else 0.0)
            _dv, _dw = _a * self.dt, _aw * self.dt
            vx = min(max(tvx, vx - _dv), vx + _dv)
            vy = min(max(tvy, vy - _dv), vy + _dv)
            wz = min(max(twz, wz - _dw), wz + _dw)
            x += vx * self.dt
            y += vy * self.dt
            yaw = wrap(yaw + wz * self.dt)
            _ex, _ey, _eyaw = x - _lx, y - _ly, wrap(yaw - _lyaw)
            # stalled: freeze the reference, no wind-up
            if math.hypot(_ex, _ey) > 0.05:
                x, y = _lx + 0.05 * _ex / math.hypot(_ex, _ey), _ly + 0.05 * _ey / math.hypot(_ex, _ey)
                _ex, _ey = x - _lx, y - _ly
            _cvx = vx + max(-_vcap, min(_vcap, _kc * _ex))
            _cvy = vy + max(-_vcap, min(_vcap, _kc * _ey))
            _cwz = wz + max(-0.3, min(0.3, _kc * _eyaw))
            _fwd = _cvx * math.cos(_lyaw) + _cvy * math.sin(_lyaw)
            _lat = -_cvx * math.sin(_lyaw) + _cvy * math.cos(_lyaw)
            self._hub_vel = self._hub_rates(_fwd, _lat, _cwz)
            self._bl = (x, y, yaw)
            self._apply(hold_q)
            self.world.step(render=not HEADLESS)
            if carry_obj is not None and (_k + 1) % int(1.0 / self.dt) == 0:
                self._carry_check(carry_obj, f"{tag}+{(_k + 1) * self.dt:.1f}s")
        # settle on the wheels
        for _ in range(int(0.4 / self.dt)):
            _lx, _ly, _lyaw = self.base_pose()
            _cvx, _cvy, _cwz = _kc * (x1 - _lx), _kc * (y1 - _ly), _kc * wrap(yaw1 - _lyaw)
            _fwd = _cvx * math.cos(_lyaw) + _cvy * math.sin(_lyaw)
            _lat = -_cvx * math.sin(_lyaw) + _cvy * math.cos(_lyaw)
            self._hub_vel = self._hub_rates(_fwd, _lat, _cwz)
            self._apply(hold_q)
            self.world.step(render=not HEADLESS)
        self._hub_vel = None
        _lx, _ly, _lyaw = self.base_pose()
        self._bl = (_lx, _ly, _lyaw)                     # ledger = LIVE, or the next pin teleports
        _emm, _edeg = arrival_error((x1, y1), yaw1, (_lx, _ly, _lyaw))
        print(f">>> wheel-move[{tag}]: {d_tot * 1000:.0f} mm / {math.degrees(wrap(yaw1 - yaw0)):+.1f} deg in "
              f"{(_k + 1) * self.dt:.1f}s -> {_emm:.0f} mm / {_edeg:+.1f} deg off", flush=True)
        self._pin_brake_apply(False)
        return self.base_pose()

    def drive_to(self, goal_xy, final_yaw, on_step=None, hold_q=None, status=None, vmax=None, freeze_arm=False,
                 discs=None, carry_obj=None):
        """OMPL-plan to goal_xy, then holonomically drive the base along the path. on_step() runs
        every physics step (pins the carried object); hold_q holds the arm; freeze_arm kinematically
        pins it so a base near the rack cannot self-explode via the parked arm grazing a shelf."""
        self._nav_v = (0.0, 0.0, 0.0)   # velocity slew state, reset per drive
        x, y, yaw = self.base_pose()
        if status:
            status(f"planning nav -> ({goal_xy[0]:.1f},{goal_xy[1]:.1f})")
        print(f">>> NAV start=({x:.3f},{y:.3f}) goal=({goal_xy[0]:.3f},{goal_xy[1]:.3f})", flush=True)
        try:
            path = plan_path((x, y), goal_xy, discs=discs)
        except RuntimeError:
            self._fallback("nav-no-plan")
            raise
        # Attributes a detour to the planner or the controller: path len vs straight catches an
        # RRTConnect detour the simplifier missed, first leg vs goal bearing a path leaving wrong.
        try:
            _pl = sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                      for i in range(len(path) - 1))
            _sl = math.hypot(goal_xy[0] - x, goal_xy[1] - y)
            _gb = math.atan2(goal_xy[1] - y, goal_xy[0] - x)
            _f1 = math.atan2(path[1][1] - y, path[1][0] - x) if len(path) > 1 else _gb
            _dev = math.degrees(abs(wrap(_f1 - _gb)))
            print(f">>> NAV path: {len(path)} pts, len {_pl:.2f}m vs straight {_sl:.2f}m "
                  f"({_pl / max(_sl, 1e-6):.2f}x), start yaw {math.degrees(yaw):+.0f}deg, "
                  f"goal bearing {math.degrees(_gb):+.0f}deg, first leg {math.degrees(_f1):+.0f}deg"
                  f"{'  <-- PATH LEAVES THE WRONG WAY' if _dev > 60 else ''}", flush=True)
            _sh = nav_endpoint_shift((x, y), goal_xy, path)
            if _sh is not None and max(_sh) > 1.0:
                print(f">>> NAV endpoints moved: start {_sh[0]:.0f} mm, goal {_sh[1]:.0f} mm "
                      f"-- the plan is not rooted where it was asked to be", flush=True)
        except Exception:
            pass
        hold_q = self.q0 if hold_q is None else hold_q
        vmax = VMAX if vmax is None else vmax
        idx, last = 1, len(path) - 1
        # Under the drive gate the base MOVES ON ITS WHEELS: the hub brake is a velocity drive at target 0
        # and nav feeds the mixer's rates, closed on the live pose. A root write would drop a held object.
        _wd = self._drive_on()
        _tele_k = 0
        if _wd:
            self._pin_brake_apply(True)
            self._hub_effort(float(os.environ.get("NAV_HUB_EFFORT", "20")))
            _kc = float(os.environ.get("NAV_TRACK_KP", "3.0"))     # 1/s, pose correction
            _vcap = float(os.environ.get("NAV_TRACK_VCAP", "0.08"))
            # The teleport profile is what a KINEMATIC base can do; on 20 Nm hubs the real limit is ~0.4 g,
            # and the held object feels every jerk.
            vmax = min(vmax, float(os.environ.get("NAV_VMAX_WD", "0.5")))
            print(f">>> NAV on wheel drives: hub kd {os.environ.get('PIN_BRAKE_KD', '1e3')}, "
                  f"effort {os.environ.get('NAV_HUB_EFFORT', '20')} Nm, track kp {_kc}/s, "
                  f"vmax {vmax:.2f} m/s, accel {os.environ.get('NAV_ACC_WD', '2.0')} m/s^2", flush=True)
        for _ in range(int(60.0 / self.dt)):
            tx, ty = path[idx]
            ex, ey = tx - x, ty - y
            dist = math.hypot(ex, ey)
            if dist < WP_TOL and idx < last:
                idx += 1
                continue
            at_goal = idx == last and dist < GOAL_TOL
            # Target heading: travel direction en route, blended into the dock yaw over the last metre.
            # Adopting the dock yaw at the last waypoint makes a short plan spin round and crab in.
            gd_ = math.hypot(goal_xy[0] - x, goal_xy[1] - y)
            yaw_blend = 1.0
            if idx == last and gd_ < yaw_blend:
                w_ = 1.0 - max(0.0, gd_ / max(yaw_blend, 1e-6))
                tgt_yaw = wrap(math.atan2(ey, ex) + w_ * wrap(final_yaw - math.atan2(ey, ex))) \
                    if dist > 1e-6 else final_yaw
            else:
                # Look ahead along the path, not at the next waypoint: consecutive waypoints can
                # be centimetres apart, so one segment's atan2 is quantisation noise the yaw chases.
                _la = 0.6
                _lx, _ly, _acc = tx, ty, dist
                _j = idx
                while _acc < _la and _j < last:
                    _j += 1
                    _nx, _ny = path[_j]
                    _acc += math.hypot(_nx - _lx, _ny - _ly)
                    _lx, _ly = _nx, _ny
                _ax, _ay = _lx - x, _ly - y
                tgt_yaw = (math.atan2(_ay, _ax) if math.hypot(_ax, _ay) > 1e-6
                           else (math.atan2(ey, ex) if dist > 1e-6 else final_yaw))
            yaw_err = wrap(tgt_yaw - yaw)
            if at_goal and abs(yaw_err) < 0.05:
                break
            # Dock profile: full speed until 2.5 m from the FINAL goal, a linear ramp to a floor
            # at 1.0 m, then that floor all the way in -- a proportional creep inches the last half-metre.
            gdist = math.hypot(goal_xy[0] - x, goal_xy[1] - y)
            vfloor = min(vmax, float(os.environ.get("DOCK_VFLOOR", "3.2")))
            if gdist > 2.5:
                vcap = vmax
            elif gdist > 1.0:
                vcap = vfloor + (vmax - vfloor) * (gdist - 1.0) / 1.5
            else:
                vcap = vfloor
            speed = 0.0 if at_goal else min(vcap, KP_LIN * dist)
            _wm = min(WMAX, float(os.environ.get("NAV_WMAX_WD", "0.6"))) if _wd else WMAX
            wz = max(-_wm, min(_wm, KP_ANG * yaw_err))    # a fast final spin whips the hand
            # Never command a speed you cannot stop from: a purely proportional law with a
            # downstream slew limit overshoots and the base spins past the target heading.
            _a = 36.0            # m/s^2
            _aw = 40.0          # rad/s^2
            # wheels, not teleports: what the hubs and the held object can take
            if _wd:
                _a = float(os.environ.get("NAV_ACC_WD", "2.0"))
                _aw = float(os.environ.get("NAV_AW_WD", "2.0"))
            _bk = 0.9
            if _bk > 0.0:
                if abs(yaw_err) > 1e-9:
                    wz = math.copysign(
                        min(abs(wz), _bk * math.sqrt(2.0 * _aw * abs(yaw_err))), yaw_err)
                else:
                    wz = 0.0
                # brake against the FINAL goal, not the lookahead waypoint: the waypoints are ~35mm
                # apart, so braking on `dist` would crawl the whole path.
                speed = min(speed, _bk * math.sqrt(2.0 * _a * max(gdist, 0.0)))
            vx, vy = (speed * ex / dist, speed * ey / dist) if dist > 1e-6 else (0.0, 0.0)
            # Acceleration limit: velocity straight from the position error goes rest -> vcap in
            # one step, and since `set_base` is a teleport that reads as the chassis jumping.
            _pv = getattr(self, "_nav_v", (0.0, 0.0, 0.0))
            _dv, _dw = _a * self.dt, _aw * self.dt
            vx = min(max(vx, _pv[0] - _dv), _pv[0] + _dv)
            vy = min(max(vy, _pv[1] - _dv), _pv[1] + _dv)
            wz = min(max(wz, _pv[2] - _dw), _pv[2] + _dw)
            self._nav_v = (vx, vy, wz)
            x += vx * self.dt
            y += vy * self.dt
            yaw = wrap(yaw + wz * self.dt)
            if _wd:
                # Command = reference velocity + P on the live pose error (world), then to the
                # body frame of the LIVE heading, then the mixer. Wheels do the moving.
                _lx, _ly, _lyaw = self.base_pose()
                _ex, _ey, _eyaw = x - _lx, y - _ly, wrap(yaw - _lyaw)
                # stalled: freeze the reference, no wind-up
                if math.hypot(_ex, _ey) > 0.05:
                    x, y = _lx + 0.05 * _ex / math.hypot(_ex, _ey), _ly + 0.05 * _ey / math.hypot(_ex, _ey)
                    _ex, _ey = x - _lx, y - _ly
                _cvx = vx + max(-_vcap, min(_vcap, _kc * _ex))
                _cvy = vy + max(-_vcap, min(_vcap, _kc * _ey))
                _cwz = wz + max(-0.3, min(0.3, _kc * _eyaw))
                _fwd = _cvx * math.cos(_lyaw) + _cvy * math.sin(_lyaw)
                _lat = -_cvx * math.sin(_lyaw) + _cvy * math.cos(_lyaw)
                self._hub_vel = self._hub_rates(_fwd, _lat, _cwz)
                self._bl = (x, y, yaw)                    # ledger follows the reference
                _tele_k += 1
                if carry_obj is not None and _tele_k % int(0.5 / self.dt) == 0:
                    # WHEN the object goes, not just whether: slip/upright every 0.5 s.
                    self._carry_check(carry_obj, f"nav+{_tele_k * self.dt:.1f}s")
                    print(f">>>   nav-tele: ref v ({vx:+.2f},{vy:+.2f}) m/s wz {wz:+.2f} | "
                          f"track err {math.hypot(x - _lx, y - _ly) * 1000:.0f} mm "
                          f"{math.degrees(wrap(yaw - _lyaw)):+.1f} deg", flush=True)
            else:
                self.set_base(x, y, yaw)
            if freeze_arm:
                # no-op under the drive gate
                self._force(hold_q)
            self._apply(hold_q)
            if on_step:
                on_step()
            self.world.step(render=not HEADLESS)
            if on_step:
                # Re-pin AFTER the step: a transform read right after `set_base` is stale for a
                # beat, so pinning before it anchors the object to where the hand HAD been.
                on_step()
        if _wd:
            # Settle on the wheels, or the goal reads as reached with the base still rolling: zero
            # rates with the pose correction still closing, then hand the hubs back to the brake.
            for _ in range(int(0.4 / self.dt)):
                _lx, _ly, _lyaw = self.base_pose()
                _cvx, _cvy, _cwz = _kc * (x - _lx), _kc * (y - _ly), _kc * wrap(yaw - _lyaw)
                _fwd = _cvx * math.cos(_lyaw) + _cvy * math.sin(_lyaw)
                _lat = -_cvx * math.sin(_lyaw) + _cvy * math.cos(_lyaw)
                self._hub_vel = self._hub_rates(_fwd, _lat, _cwz)
                self._apply(hold_q)
                self.world.step(render=not HEADLESS)
            self._hub_vel = None
            _lx, _ly, _lyaw = self.base_pose()
            self._bl = (_lx, _ly, _lyaw)                  # ledger = LIVE, or the next pin teleports
            _emm, _edeg = arrival_error(goal_xy, final_yaw, (_lx, _ly, _lyaw))
            print(f">>> NAV on wheel drives: stopped ({_lx:.3f}, {_ly:.3f}, {math.degrees(_lyaw):+.1f}deg) "
                  f"-> {_emm:.0f} mm / {_edeg:+.1f} deg off goal", flush=True)
            self._pin_brake_apply(False)
        return self.base_pose()
