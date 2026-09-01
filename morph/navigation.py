"""OMPL base navigation: plan a path in a subprocess, then drive the chassis along it.

Split out of play_isaac.py; the method bodies are unchanged. Imports Isaac APIs at module
level, so it is only importable after SimulationApp exists — see `morph/__init__.py`.
"""
import json
import math
import os
import subprocess


from morph.config import (HERE, VENV_PY, NAV_PLAN, HEADLESS, VMAX, WMAX, KP_LIN, KP_ANG,
    WP_TOL, GOAL_TOL)
from morph.geometry import wrap


def plan_path(start_xy, goal_xy, solve_time=3.0, tries=3, discs=None):
    """OMPL RRTConnect via the isaac-owned nav_plan.py, run in the project venv (has ompl).  RRTConnect
    is randomized and occasionally times out from a cornered start, so retry with more time.
    discs: [[x,y,r]] dynamic keep-outs (floor objects) so the chassis never plows through one —
    in friction mode objects are REAL bodies and a kinematic base shoves them irresistibly."""
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

    def drive_to(self, goal_xy, final_yaw, on_step=None, hold_q=None, status=None, vmax=None, freeze_arm=False,
                 discs=None):
        """OMPL-plan to goal_xy, then holonomically drive the base along the path.  on_step(i)
        runs every physics step (used to pin the carried object); hold_q holds the arm.  freeze_arm=True
        KINEMATICALLY pins the arm (force positions + zero velocity) so a base near the rack can't
        self-explode via the parked arm grazing a shelf (the loaded carry-to-rack path)."""
        self._nav_v = (0.0, 0.0, 0.0)   # velocity slew state, reset per drive
        x, y, yaw = self.base_pose()
        if status:
            status(f"planning nav -> ({goal_xy[0]:.1f},{goal_xy[1]:.1f})")
        print(f">>> NAV start=({x:.3f},{y:.3f}) goal=({goal_xy[0]:.3f},{goal_xy[1]:.3f})", flush=True)
        path = plan_path((x, y), goal_xy, discs=discs)
        # Attribute a detour to the planner or to the controller. `straight` against `len` catches
        # an RRTConnect detour the simplifier failed to shortcut; `first leg` against `goal bearing`
        # catches a path that starts out heading away from the goal.
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
        except Exception:
            pass
        hold_q = self.q0 if hold_q is None else hold_q
        vmax = VMAX if vmax is None else vmax
        idx, last = 1, len(path) - 1
        for _ in range(int(60.0 / self.dt)):
            tx, ty = path[idx]
            ex, ey = tx - x, ty - y
            dist = math.hypot(ex, ey)
            if dist < WP_TOL and idx < last:
                idx += 1
                continue
            at_goal = idx == last and dist < GOAL_TOL
            # Target heading: travel direction en route, dock yaw on the last leg. Adopting the dock
            # yaw as soon as the last waypoint is current takes it from the very first step whenever
            # the plan has only two waypoints -- the common case in a clear corridor -- so the robot
            # spins to a heading often near-opposite the travel direction and then crabs sideways.
            gd_ = math.hypot(goal_xy[0] - x, goal_xy[1] - y)
            yaw_blend = 1.0
            if idx == last and gd_ < yaw_blend:
                w_ = 1.0 - max(0.0, gd_ / max(yaw_blend, 1e-6))
                tgt_yaw = wrap(math.atan2(ey, ex) + w_ * wrap(final_yaw - math.atan2(ey, ex))) \
                    if dist > 1e-6 else final_yaw
            else:
                # Look ahead along the path rather than aiming at the next waypoint. The planner
                # interpolates to a fixed number of states, so on a short hop consecutive waypoints
                # are centimetres apart and the `atan2` of one segment is mostly quantisation noise
                # -- the heading target jitters and the base rotates one way, then back.
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
            # Dock profile: full speed until 2.5m from the FINAL goal, a linear ramp down to a floor
            # at 1.0m, then hold that floor all the way in — never a proportional creep, which
            # inches the base through the last half-metre of the dock.
            gdist = math.hypot(goal_xy[0] - x, goal_xy[1] - y)
            vfloor = min(vmax, float(os.environ.get("DOCK_VFLOOR", "3.2")))
            if gdist > 2.5:
                vcap = vmax
            elif gdist > 1.0:
                vcap = vfloor + (vmax - vfloor) * (gdist - 1.0) / 1.5
            else:
                vcap = vfloor
            speed = 0.0 if at_goal else min(vcap, KP_LIN * dist)
            wz = max(-WMAX, min(WMAX, KP_ANG * yaw_err))
            # Never command a speed you cannot stop from. A purely proportional law with a slew
            # limit downstream overshoots whenever the commanded velocity exceeds what the
            # acceleration limit can bleed off over the error that is LEFT -- and at this cruise
            # speed that is nearly always, so the base spins past the target heading and corrects
            # back.
            _a = 36.0            # m/s^2
            _aw = 40.0          # rad/s^2
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
            # Acceleration limit. Setting the velocity directly from the position error takes the
            # base from rest to `vcap` in one step and back to zero the same way -- and since
            # `set_base` is a teleport, that reads as the chassis jumping rather than driving.
            _pv = getattr(self, "_nav_v", (0.0, 0.0, 0.0))
            _dv, _dw = _a * self.dt, _aw * self.dt
            vx = min(max(vx, _pv[0] - _dv), _pv[0] + _dv)
            vy = min(max(vy, _pv[1] - _dv), _pv[1] + _dv)
            wz = min(max(wz, _pv[2] - _dw), _pv[2] + _dw)
            self._nav_v = (vx, vy, wz)
            x += vx * self.dt
            y += vy * self.dt
            yaw = wrap(yaw + wz * self.dt)
            self.set_base(x, y, yaw)
            if freeze_arm:
                self._force(hold_q)
            self._apply(hold_q)
            if on_step:
                on_step()
            self.world.step(render=not HEADLESS)
            if on_step:
                # Re-pin AFTER the step. `on_step` rebuilds the object pose from the LIVE hand
                # frame, but a transform read right after `set_base` is stale for a beat -- so
                # called before the step it pins the object to where the hand HAD been and the base
                # then moves out from under it.
                on_step()
        return self.base_pose()
