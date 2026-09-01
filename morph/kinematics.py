"""Frames, forward kinematics, the closure LUT and the baked-trajectory player.

Split out of play_isaac.py; the method bodies are unchanged. Imports Isaac APIs at module
level, so it is only importable after SimulationApp exists — see `morph/__init__.py`.
"""
import json
import math
import os

import numpy as np
from isaacsim.core.prims import SingleXFormPrim

from morph.config import HERE, KNOWN, PERFECT, HEADLESS, COLUMN_MAX, A1_MAX
from morph.geometry import wrap
from morph.usd_utils import find_path


class KinematicsMixin:
    """Mixed into Demo. Every `self.*` it touches is owned by Demo."""

    def grip_pos(self):
        return np.asarray(self._grip_prim.get_world_pose()[0], float)

    def _grip_frame(self):
        """World (pos, R) of the gripper body — R rotates a gripper-frame offset into world."""
        p, q = self._grip_prim.get_world_pose()
        p = np.asarray(p, float)
        w, x, y, z = [float(v) for v in q]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
                      [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]])
        return p, R

    def _hand_prims(self):
        """name -> SingleXFormPrim for every arm-1 hand/finger link (contact forensics)."""
        if getattr(self, "_hand_prims_cache", None) is None:
            out = {}
            for prim in self.stage.Traverse():
                nm = prim.GetName()
                pth = prim.GetPath().pathString
                if "Arm_1" not in pth:
                    continue
                if (nm.startswith("finger_") and "_link_" in nm and nm.endswith("_1")) \
                        or nm.startswith("Gripper_Link") and nm.endswith("_1"):
                    out[nm] = SingleXFormPrim(pth)
            self._hand_prims_cache = out
        return self._hand_prims_cache

    def _hand_frame(self):
        """World (pos, R) of Gripper_Link3_1 — the body the FINGERS are children of.

        NOT the same as _grip_frame (Gripper_Link1_1): the wrist joints sit BETWEEN them, so the two
        frames move relative to each other whenever the wrist turns.  Pinning the object to Link1
        while the grip is formed by fingers on Link3 therefore slides the object inside the hand as
        soon as the lift trajectory moves the wrist — the finger gaps open by several millimetres
        between capture and lift-end with the finger joints completely unchanged, i.e. the object
        moves inside the gripper. `_grip_prim` stays Link1 because the baked FK and closure LUT are
        calibrated to it; only the HOLD moves to the hand body.
        """
        if getattr(self, "_hand_prim", None) is None:
            self._hand_prim = SingleXFormPrim(find_path(self.stage, "Gripper_Link3_1"))
        p, q = self._hand_prim.get_world_pose()
        w, x, y, z = [float(v) for v in q]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
                      [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]])
        return np.asarray(p, float), R

    def _obj_R(self, obj):
        """World rotation of the object, for gap measurement.

        Every geometric finger stop used to model the object as a WORLD-Z ALIGNED cylinder while
        `_phase_report` measured it in the object's real frame.  When the fingers tip the cylinder
        the two disagree wildly: the seat reported all three pads at +0.7/+0.9/+1.0mm and the very
        next report — same instant, same colliders — read 20.3/32.0/75.9mm.  The seat was stopping
        against a cylinder that was no longer standing up."""
        try:
            return self._quat_to_R(np.asarray(obj.get_world_poses()[1][0], float))
        except Exception:
            return None

    @staticmethod
    def _lut_cell(vals, v):
        v = min(max(v, vals[0]), vals[-1])
        for i in range(len(vals) - 2, -1, -1):
            if v >= vals[i]:
                return i, (v - vals[i]) / (vals[i + 1] - vals[i])
        return 0, 0.0

    def _lut_interp(self, table, dh, a1, n):
        i, fi = self._lut_cell(self.clut["dh"], dh)
        j, fj = self._lut_cell(self.clut["a1"], a1)
        return [(1 - fi) * (1 - fj) * table[i][j][k] + fi * (1 - fj) * table[i + 1][j][k]
                + (1 - fi) * fj * table[i][j + 1][k] + fi * fj * table[i + 1][j + 1][k]
                for k in range(n)]

    def _closure_passives(self, dh, a1):
        """Closure LUT -> the 4 passive linkage joints for (h2-h1, a1)."""
        return dict(zip(self.clut["passives"], self._lut_interp(self.clut["grid"], dh, a1, 4)))

    def _grip_fk(self, h1, h2, a1, th):
        """TRUE gripper (Gripper_Link1_1) world position from the baked map: chassis-frame grip
        at (dh, a1), z-shifted 1:1 by (h1 - h1_ref) — equal column moves translate the whole
        4-bar rigidly — and rotated about the arm mount by th (BaseJoint, +z CCW).  The old
        boom-tip-only FK missed the wrist lever arm (real dz ~1m vs FK ~0.05m) and made the
        closed-loop servo diverge to NaN — the map IS the fix."""
        p = np.array(self._lut_interp(self.clut["grip"], h2 - h1, a1, 3))
        p[2] += h1 - self.clut["h1_ref"]
        mx, my = self.clut["mount"][0], self.clut["mount"][1]
        c, s = math.cos(th), math.sin(th)
        px, py = p[0] - mx, p[1] - my
        p[0], p[1] = mx + c * px - s * py, my + s * px + c * py
        # base pose: the jog anchor when a jog is active, else the live ledger -- so callers that do
        # not start a jog (the recorded-relation align) still get a correct world grip position.
        bx, by, byaw = (self.jog["anchor"] if getattr(self, "jog", None) else self.base_ledger())
        c, s = math.cos(byaw), math.sin(byaw)
        return np.array([bx + c * p[0] - s * p[1], by + s * p[0] + c * p[1], p[2]])

    def _lut_z(self, dh, a1):
        """Raw LUT pinch z at (dh, a1), h1 at the bake reference. `_fk_z` adds the live h1; the
        const-z ramps use this directly because their `C = h1 + lut_z` invariant cancels h1_ref."""
        return float(self._lut_interp(self.clut["grip"], dh, a1, 3)[2])

    def _bisect_a1(self, pred, lo, hi, iters=40):
        """Monotone bisection over a1: the smallest value in (lo, hi] where `pred` holds. The
        caller checks pred(hi) beforehand; the predicate must be monotone along the a1 row."""
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if pred(mid):
                hi = mid
            else:
                lo = mid
        return float(hi)

    def _fk_z(self, h1, h2, a1):
        """Pinch HEIGHT from the baked closure map -- no physics, no settle, no transient.

        A tilt gradient has to be read this way rather than by moving the real joint: snapping the
        column differential and reading back a few frames later on a FORCED parallel linkage
        measures a transient, not the settled pose, and the resulting map disagrees with live runs
        by hundreds of millimetres.

        Only z is needed, and z is independent of the chassis pose -- `_grip_fk`'s base rotation and
        translation are planar -- so the baked `jog["anchor"]` does not contaminate this.
        """
        return self._lut_z(h2 - h1, a1) + h1 - self.clut["h1_ref"]

    # Chassis COLLISION plate, from the source MJCF: a box with half-extents 0.35 x 0.3 x 0.01, so
    # 0.70 x 0.60 x 0.02m.
    CHASSIS_PLATE_X = 0.35
    CHASSIS_PLATE_Y = 0.30          # same MJCF geom: size="0.35 0.3 0.01" half-extents

    def _clears_chassis(self, dh, a1, margin=0.02, th=0.0):
        """Is the gripper FORWARD of the chassis collision plate at this (dh, a1)?

        Articulation self-collision is off, so physics will not stop the arm entering the body --
        and switching it on is not free either: the selective filter applies correctly (436 pairs)
        but the process is OOM-killed at 5.1GB RSS -- a resource wall. A geometric test costs one
        table lookup and no memory.

        Measured against the poses this pipeline commands:
            descent start   dh 0.058 a1 0.347 -> x 0.583  clear
            solved pose     dh 0.150 a1 0.500 -> x 0.531  clear
            fallback retract dh 0.150 a1 0.150 -> x 0.336  INSIDE the plate
            fallback tilt    dh 0.250 a1 0.150 -> x 0.291  INSIDE
        i.e. every arm-into-body event was the old fallback retracting a1 while tilted.
        """
        try:
            # The LUT is baked at th=0 ("runtime: rotate xy about mount by th" -- its own note).
            p = self._lut_interp(self.clut["grip"], dh, a1, 3)
            mx, my = self.clut["mount"][0], self.clut["mount"][1]
            c, s_ = math.cos(th), math.sin(th)
            px, py = float(p[0]) - mx, float(p[1]) - my
            x = mx + c * px - s_ * py
            y = my + s_ * px + c * py
            return (abs(x) >= self.CHASSIS_PLATE_X + margin
                    or abs(y) >= self.CHASSIS_PLATE_Y + margin)
        except Exception:
            return True                      # no table -> do not block motion on a failed lookup

    def _a1_clearing_chassis(self, dh, margin=0.02, a1_max=A1_MAX, th=0.0):
        """Smallest a1 that keeps the gripper clear of the plate at this tilt.

        The safe bound is NOT a tilt constant -- it is a (dh, a1) pair, which is why
        `LOWEST_TILT_MAX` could never express it. More tilt pulls the hand BACK toward the chassis,
        so deeper tilts need MORE a1 to stay out of it:
            dh 0.05 -> a1 >= 0.072      dh 0.20 -> a1 >= 0.242
            dh 0.15 -> a1 >= 0.175      dh 0.30 -> a1 >= 0.380
        """
        lo, hi = 0.0, float(a1_max)
        if self._clears_chassis(dh, lo, margin, th=th):
            return 0.0
        if not self._clears_chassis(dh, hi, margin, th=th):
            return None                      # no a1 clears the plate at this tilt
        return self._bisect_a1(lambda a: self._clears_chassis(dh, a, margin, th=th), lo, hi)

    def _solve_lowest(self, z_tgt, off_z, a1_max=A1_MAX, h_max=COLUMN_MAX, dh_want=None, h1_min=0.0):
        """Solve (h1, h2, a1) that puts the pinch at `z_tgt` with the MOST forward reach.

        Three DOFs, one hard constraint (height), so there is a whole family of solutions and the
        free parameter is worth spending on reach -- this stage was previously servoing blindly into
        that null space and landing on a bad member of it.

        `off_z` is the measured live-pinch-minus-FK offset. The map gives Gripper_Link1 while the
        pinch is a wrist lever away, and that offset cancels when the FK is used for GRADIENTS —
        but for an absolute solve it has to be measured rather than assumed.

        Grid search, not a Newton solve: the surface is cheap (2k table interps), and a grid cannot
        wander off into the linkage's bad regions the way a gradient step can. Returns None if the
        height is simply not reachable within the joint stops and the LUT's dh range.
        """
        # A requested tilt wins over maximum reach. Maximising reach picks the SMALLEST feasible
        # tilt, which is the opposite of what this arm needs: tilt hard first, spend a1 driving the
        # gripper down along the tilted boom, then trade the tilt back for forward reach.
        if dh_want is not None:
            a1 = a1_max
            p = self._lut_interp(self.clut["grip"], dh_want, a1, 3)
            h1 = z_tgt - off_z - float(p[2]) + self.clut["h1_ref"]
            if (h1_min <= h1 <= h_max and 0.0 <= h1 + dh_want <= h_max
                    and self._clears_chassis(dh_want, a1)):
                return (h1, h1 + dh_want, a1, float(p[0]))
            # requested tilt would sink the carriage below the floor -> fall through: the search
            # below picks the shallowest tilt that keeps h1 >= h1_min at a1_max asked-for tilt is
            # past the column stop at this height -- fall through to max reach
        best = None
        dhs = self.clut["dh"]
        for i in range(81):
            dh = dhs[0] + (dhs[-1] - dhs[0]) * i / 80.0
            for j in range(51):
                a1 = a1_max * j / 50.0
                p = self._lut_interp(self.clut["grip"], dh, a1, 3)
                # z = p[2] + h1 - h1_ref + off_z  ->  h1 that lands exactly on the target
                h1 = z_tgt - off_z - float(p[2]) + self.clut["h1_ref"]
                h2 = h1 + dh
                if not (h1_min <= h1 <= h_max and 0.0 <= h2 <= h_max):
                    continue
                if not self._clears_chassis(dh, a1):    # inside the chassis plate -- not a pose
                    continue
                if best is None or float(p[0]) > best[0]:
                    best = (float(p[0]), h1, h2, a1)
        return None if best is None else (best[1], best[2], best[3], best[0])

    def _solve_reach_z(self, reach_tgt, z_tgt, off_z, a1_max=A1_MAX, h_max=COLUMN_MAX, h1_min=0.0,
                       dh_ref=None, th=0.0, a1_ref=None, h1_ref=None):
        """Solve (h1, h2, a1) that puts the grip at `reach_tgt` forward AND `z_tgt` high.

        The dual of `_solve_lowest`: that one fixes height and spends the free parameter on MAXIMUM
        reach, which is the right call for the descent (get as low as possible, reach as far as the
        pose allows). The place needs a SPECIFIC reach -- the distance from the docked base to the
        slot -- so the free parameter has to be spent hitting it exactly instead.

        Why a solve and not a servo: reach and height are violently coupled through the tilt. Read
        off the LUT at a1 0.50, straightening from dh 0.110 to 0.000 gains 190mm of reach and 440mm
        of HEIGHT at the same time. An incremental servo that nudges tilt for reach and corrects
        height afterwards cannot track that -- measured, the object left the target while the
        gradient kept asking for more (place.py's insert notes). Choosing the (dh, a1) pair up front
        and ramping to it once makes the coupling a non-issue: both are satisfied by construction.

        Same grid-search discipline as `_solve_lowest` -- the surface is cheap and a grid cannot
        wander into the linkage's bad regions the way a gradient step can. Among the pairs that hit
        the target, prefer the one nearest `dh_ref` (the pose we are already in) so the arm makes
        the smallest move that works. Returns None when the (reach, height) pair is simply not in
        the arm's envelope, which is the honest answer for a standoff beyond its reach.
        """
        best = None
        dhs = self.clut["dh"]
        for i in range(81):
            dh = dhs[0] + (dhs[-1] - dhs[0]) * i / 80.0
            for j in range(51):
                a1 = a1_max * j / 50.0
                p = self._lut_interp(self.clut["grip"], dh, a1, 3)
                h1 = z_tgt - off_z - float(p[2]) + self.clut["h1_ref"]
                h2 = h1 + dh
                if not (h1_min <= h1 <= h_max and 0.0 <= h2 <= h_max):
                    continue
                if not self._clears_chassis(dh, a1):
                    continue
                # Rotate by th before comparing. The LUT stores the grip in the chassis frame BEFORE
                # the turret rotation, and `_grip_fk` then swings it about the mount by th -- so the
                # table's p[0] is not the chassis-forward reach unless th is 0.
                _mx, _my = self.clut["mount"][0], self.clut["mount"][1]
                _c, _s = math.cos(th), math.sin(th)
                _px, _py = float(p[0]) - _mx, float(p[1]) - _my
                reach = _mx + _c * _px - _s * _py
                err = abs(reach - reach_tgt)
                # Continuity is not a tie-break, it is half the problem. (reach, z) is two
                # constraints on three DOFs, so the solution set is a whole curve and many (dh, a1,
                # h1) triples hit the same point.
                _w = 0.5
                cost = err
                if dh_ref is not None:
                    cost += _w * abs(dh - dh_ref)
                if a1_ref is not None:
                    cost += _w * abs(a1 - a1_ref)
                if h1_ref is not None:
                    cost += _w * abs(h1 - h1_ref)
                if best is None or cost < best[0]:
                    best = (cost, err, h1, h2, a1, reach)
        if best is None:
            return None
        return (best[2], best[3], best[4], best[5], best[1])   # h1, h2, a1, reach, reach_err

    def _load_trajs(self):
        """Load the baked loop-consistent trajectories from `usd/_traj_*.json`.

        Every frame satisfies the parallel-linkage closure to within a tenth of a millimetre at
        bake time, which is what makes multi-step arm motion replayable without exploding the
        excluded loop joint. Returns None — replay disabled, direct-write fallback — if any file
        or joint is missing.
        """
        out = {}
        for name in ("reach", "lift", "place_low", "place_mid", "place_high"):
            p = os.path.join(HERE, f"usd/_traj_{name}.json")
            if not os.path.exists(p):
                print(f">>> no {p} -> trajectory replay disabled (run gen_traj.py)", flush=True)
                return None
            t = json.load(open(p))
            missing = [j for j in t["joints"] if j not in self.idx]
            if missing:
                print(f">>> traj {name}: unknown joints {missing[:4]} -> replay disabled", flush=True)
                return None
            # Drop the arm-2 columns from the bake. The recorded frames carry a low home pose for
            # arm 2, so replaying them visibly drags the idle hand down.
            keep = [i for i, j in enumerate(t["joints"]) if not j.endswith("_2")]
            out[name] = {"idx": np.array([self.idx[t["joints"][i]] for i in keep]),
                         "frames": np.array(t["frames"], float)[:, keep], "meta": t["meta"],
                         # kept-column joint NAMES: stages that need one specific joint out of the
                         # bake (the descent's wrist target) can look it up; before this the names
                         # were dropped at load and the wrist fix failed with KeyError 'joints'.
                         "names": [t["joints"][i] for i in keep]}
        # Symmetric open for the side grip. The baked reach opens the thumb much wider than b/c,
        # which is the right pose for a TOP-DOWN grasp — the thumb has to clear the object's top
        # face — but on a side grip it starts the thumb splayed well behind the palm, so the palm
        # meets the object first.
        if PERFECT and os.environ.get("SIDE_GRIP_SYMMETRIC", "1") == "1":
            _op = -0.55
            for _nm, _tr in out.items():
                for _c, _jn in enumerate(_tr["names"]):
                    if _jn in ("finger_a_joint_1_1", "finger_b_joint_1_1", "finger_c_joint_1_1"):
                        _col = _tr["frames"][:, _c]
                        _col[_col < 0.0] = _op            # symmetric OPEN; leave any closing frames
            print(f">>> side-grip: baked traj finger-open -> {_op:+.2f} symmetric "
                  f"(was thumb -1.04 / b,c -0.49)", flush=True)
        print(f">>> loaded {len(out)} baked trajectories (loop-consistent replay ON)", flush=True)
        return out

    def _play_traj(self, tr, secs, pin=None, grip_pin=None, reverse=False, hold_fingers=False,
                   on_step=None, frame_range=None, hold_wrist=False):
        """Kinematically replay a baked trajectory: force the full config every step — the proven
        _hold(kin=True) stability pattern extended along a PATH of loop-consistent frames (adjacent
        frames are ~0.04 rad apart; the micro-lerp between them stays on the closure manifold).
        pin=(obj, pos) holds an object at a fixed spot; grip_pin=obj rides it on the LIVE gripper.
        hold_fingers=True keeps the CURRENT finger pose (the curled grip) instead of the baked open
        fingers — used for lift/insert replays while holding an object."""
        frames, ii = tr["frames"], tr["idx"]
        if frame_range is not None:
            frames = frames[frame_range[0]:frame_range[1]]
        if reverse:
            frames = frames[::-1]
        n = max(2, int(secs / self.dt))
        q = np.asarray(self.robot.get_joint_positions(), float).copy()
        fhold = None
        if hold_fingers:
            f_idx = np.array([i for nm, i in self.idx.items()
                              if nm.startswith("finger_") or nm.startswith("palm_finger")])
            fhold = (f_idx, q[f_idx].copy())
        whold = None
        if hold_wrist:
            # Same mechanism as hold_fingers, and the same reason. The baked frames carry the
            # RECORDED wrist angles, and replaying them drives the hand to the recorded attitude no
            # matter what any earlier stage set -- which is exactly how the pre-dock levelling was
            # undone twice (levelled to 1.4deg, insert-entry read 17.5deg regardless; the "dock leg
            # does it by an unknown path" conclusion was wrong, THIS replay does it,
            # deterministically).
            w_idx = np.array([i for nm, i in self.idx.items()
                              if nm in ("HandBearingJoint_1", "gripper_z_rotation_1",
                                        "gripper_y_rotation_1", "gripper_x_rotation_1")])
            whold = (w_idx, q[w_idx].copy())
        bx, by, byaw = self.base_ledger()                    # base held FIXED throughout the replay
        q_in = q[ii].copy()                                  # BLEND-IN: ease from the CURRENT arm pose onto
        n_b = min(int(0.3 / self.dt), n // 3)                #   frame[0] (after a fine-align servo the arm
        # velocity-consistent kinematic write: friction needs the solver to SEE the links move (see
        # _force docstring) — zero-velocity teleports made every lift a stationary wall and the
        # object never rose a millimetre
        q_prev = q.copy()
        for k in range(n):                                   #   sits a few cm off the baked path — no snap)
            if k < n_b:
                s = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / n_b)
                q[ii] = (1 - s) * q_in + s * frames[0]
            else:
                f = (k + 1 - n_b) / (n - n_b) * (len(frames) - 1)   # linear — the bake already eased
                i0 = int(f)
                i1 = min(i0 + 1, len(frames) - 1)
                a = f - i0
                q[ii] = (1 - a) * frames[i0] + a * frames[i1]
            if fhold is not None:
                q[fhold[0]] = fhold[1]
            if whold is not None:
                q[whold[0]] = whold[1]
            if (PERFECT and hold_fingers and self._finger_cmd is not None
                    and k % int(0.25 / self.dt) == 0
                    and getattr(self, "_tighten_total", 0.0) < 0.10):
                # Re-tighten periodically while carrying, capped in total: uncapped it keeps driving
                # the pads and eventually pushes the object out of the hand.
                self._tighten_total = getattr(self, "_tighten_total", 0.0) + 0.02
                for pos, nm in enumerate(self._f_names):
                    if nm.startswith("finger_") and nm.endswith("joint_1_1"):
                        self._finger_cmd[pos] = min(self._finger_cmd[pos] + 0.02, 1.0)
            self.set_base(bx, by, byaw)
            qv = (q - q_prev) / self.dt
            q_prev = q.copy()
            self._force(q, qv)
            self._apply(q)
            if pin is not None and not PERFECT:              # friction: the object RESTS there itself
                o, p = pin
                o.set_world_poses(positions=np.array([p]), orientations=np.array([[1, 0, 0, 0]]))
                o.set_velocities(np.zeros((1, 6)))
            if grip_pin is not None:
                self._pin_grip(grip_pin)                     # (no-op in perfect mode)
            if on_step is not None:
                on_step()
            self.world.step(render=not HEADLESS)
            if os.environ.get("REPLAY_DBG") == "1" and k % 10 == 0:
                rbx, rby, _ = self.base_pose()
                rv = np.asarray(self.robot.get_joint_velocities(), float)
                fi = 0.0 if k < n_b else (k + 1 - n_b) / (n - n_b) * (len(frames) - 1)
                print(f"    dbg k={k} f={fi:.1f} base=({rbx:.3f},{rby:.3f}) "
                      f"|qv|max={np.abs(rv).max():.2f} palm_z={self.grip_pos()[2]:.3f}", flush=True)
            if os.environ.get("REPLAY_DBG") == "1" and k == n - 1:
                qa = np.asarray(self.robot.get_joint_positions(), float)
                dq = np.abs(qa - q)
                worst = np.argsort(dq)[::-1][:6]
                print("    dbg cmd-vs-actual worst: " + ", ".join(
                    f"{self.names[i]}={dq[i]:.3f}" for i in worst), flush=True)
        return q

    def _load_close_traj(self):
        """The finger trajectory of a real successful grasp, from `usd/_grasp_close.json`.

        Replaying it beats synthesising a curl. The recorded grip is far gentler than the close
        formula suggests — the side fingers barely move and the thumb does the work — whereas a
        synthetic close drives every joint about a radian past the real grip, which sweeps the
        fingers through the object and squeezes it out.
        """
        p = os.path.join(HERE, "usd/_grasp_close.json")
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f">>> close-replay unavailable ({e}) -> synthetic curl", flush=True)
            return None
        ph = d.get("phase", [])
        # The `close` phase is thumb-only by construction. Across its rows the thumb closes its full
        # travel while b and c barely move:
        #     finger_a j1  -1.040 -> -0.018      finger_b j1  -0.493 -> -0.446
        #                                        finger_c j1  -0.492 -> -0.395
        # b and c do their real closing in `post_close`, so a bounded number of those rows is
        # included to add the b/c wrap. Only FINGER joints are replayed, so it adds no arm motion.
        #
        # The row count matters: `post_close` also contains the lift, the transport and the release,
        # so its tail frames open the fingers again. The real hold is a plateau part-way through --
        # and even that is held with the reference model's compliant weld installed, so the
        # recording contains no friction-only three-finger cage anywhere.
        n_post = 200    # 0 = close-phase only
        rows = [i for i, s in enumerate(ph) if s == "close"]
        if n_post > 0:
            rows += [i for i, s in enumerate(ph) if s == "post_close"][:n_post]
        rows = rows or list(range(len(d["frames"])))
        names = [j for j in d["joints"] if "finger" in j]
        miss = [n for n in names if n not in self.idx]
        if miss:
            print(f">>> close-replay: joints missing in Isaac {miss[:3]} -> synthetic curl", flush=True)
            return None
        cols = [d["joints"].index(n) for n in names]
        fr = np.array([[d["frames"][r][c] for c in cols] for r in rows], float)
        ob = d.get("obj_in_base")
        if ob and os.environ.get("USE_RECORDED_DOCK", "1") == "1":
            # The dock target, taken from the grasp that actually worked. _grasp_known's base/object
            # pair puts the object at (0.703, +0.019) in the base frame; the recorded successful
            # close has it at (0.682, -0.081) — 99mm off LATERALLY.
            rec = np.array([float(ob[rows[0]][0]), float(ob[rows[0]][1])])
            # LOCAL_OBJ_OFF applies HERE, not in `config.local_obj()` — this line is what the dock
            # actually uses, and it overwrites whatever the config computed.
            _off = os.environ.get("LOCAL_OBJ_OFF", "") if PERFECT else ""
            if _off:
                rec = rec + np.array([float(x) for x in _off.split(",")[:2]], float)
                print(f">>> dock target: LOCAL_OBJ_OFF {_off} applied -> "
                      f"{np.round(rec, 4).tolist()}", flush=True)
            # Clamp the range, keep the lateral offset. What the recording fixes is the LATERAL
            # relation; its RANGE is a separate question. The recorded distance is reachable, but
            # only with a few centimetres to spare, and a1 is the binding constraint on reach — so
            # take the larger margin that DOCK_DIST gives.
            #
            # Shorten x alone rather than rescaling the vector: scaling would drag the lateral
            # offset in with it and give back part of what this block exists to fix. Clamp only — if
            # the recording is already closer than DOCK_DIST, leave it alone.
            _dmax = 0.64
            if _dmax > 0 and float(np.linalg.norm(rec)) > _dmax and abs(rec[1]) < _dmax:
                _x_new = float(math.sqrt(_dmax * _dmax - float(rec[1]) * float(rec[1])))
                print(f">>> dock target: range {float(np.linalg.norm(rec)):.3f} -> {_dmax:.3f}m "
                      f"(the grasp distance); x {float(rec[0]):.4f} -> {_x_new:.4f}, "
                      f"lateral {float(rec[1]):+.4f} UNCHANGED", flush=True)
                rec = np.array([_x_new * (1.0 if rec[0] >= 0 else -1.0), float(rec[1])])
            self.pick_standoff = float(np.linalg.norm(rec))
            print(f">>> dock target: recorded grasp {np.round(rec, 4).tolist()} "
                  f"(was {np.round(self.local_obj, 4).tolist()}, "
                  f"{np.linalg.norm(rec - self.local_obj) * 1000:.0f}mm apart)", flush=True)
            self.local_obj = rec
        print(f">>> close-replay: {len(fr)} recorded grasp frames, {len(names)} finger joints "
              f"(grip pose j1 a/b/c = {fr[-1][names.index('finger_a_joint_1_1')]:+.2f}/"
              f"{fr[-1][names.index('finger_b_joint_1_1')]:+.2f}/"
              f"{fr[-1][names.index('finger_c_joint_1_1')]:+.2f})", flush=True)
        # FRAME.  The recording ships the object's relation in TWO frames, and they are 158mm apart:
        #   obj_in_gripper  -> meta['gripper_frame_body'] = Gripper_Link3_1
        #   obj_in_weldbody -> meta['weld_frame_body']    = Gripper_Link1_1   <- what _grip_frame
        # reads We were feeding the Link3-frame array into a Link1-frame servo.  The x component of
        # that error is -0.0768, which is what ALIGN_OFF_X=-0.090 was silently cancelling — a fudge
        # tuned by sweeping pad gaps, so it fixed x and left the -0.136 Y error untouched.  Gripper
        # +Y is very nearly world-DOWN here, so that residual is a 136mm vertical misplacement: the
        # hand rode high, every finger dz went positive (a +36, c +63 — above the top face entirely)
        # and the PALM became the only part low enough to reach the object, which it did by clipping
        # 13-26mm into its top rim.  Use the frame that matches the prim we actually measure.
        key = "obj_in_weldbody" if os.environ.get("REC_FRAME", "weld") == "weld" else "obj_in_gripper"
        og = d.get(key) or d.get("obj_in_gripper")
        # The grip relation must come from the end of the `close` phase, not from the last row
        # replayed.
        grip_row = max([i for i, s in enumerate(ph) if s == "close"] or rows)
        if og:
            self._rec_obj_in_grip = np.array(og[grip_row][:3], float)
            # keep the LINK3 array too — it is the frame the FINGERS live in, so it is the one to
            # compare against when the pads miss despite a converged Link1 relation
            _g3 = d.get("obj_in_gripper")
            if _g3:
                self._rec_obj_in_grip_L3 = np.array(_g3[grip_row][:3], float)
            print(f">>> recorded {key} at grasp (frame "
                  f"{d['meta'].get('weld_frame_body' if key == 'obj_in_weldbody' else 'gripper_frame_body')}): "
                  f"{np.round(self._rec_obj_in_grip, 4).tolist()}", flush=True)
        return {"names": names, "idx": np.array([self.idx[n] for n in names]), "frames": fr}

    def _sync_object_size(self):
        """Read the pickup cylinder's REAL radius/half-height off the stage and overwrite the
        json-derived constants.  _grasp_known claims half_h 0.14; the shipped cylinder is 0.30 tall
        (half 0.15), so every top/clearance number computed from the json was 10-20mm optimistic."""
        from pxr import UsdGeom as _UG
        # Author the radius onto the prim, or OBJ_RADIUS does nothing: it sets KNOWN at import and
        # this method overwrites that from the stage cylinder a moment later, so a size sweep would
        # re-run the same geometry with a relabelled constant.
        _ro = os.environ.get("OBJ_RADIUS")
        if _ro:
            _rv = float(_ro)
            _n_au = 0
            for prim in self.stage.Traverse():
                if "pickup_obj" in prim.GetPath().pathString and prim.GetTypeName() == "Cylinder":
                    _a = _UG.Cylinder(prim).GetRadiusAttr()
                    if _a and abs(float(_a.Get() or 0.0) - _rv) > 1e-6:
                        _a.Set(_rv)
                        _n_au += 1
            if _n_au:
                print(f">>> object size: AUTHORED radius {_rv:.3f} on {_n_au} pickup cylinders "
                      f"(OBJ_RADIUS)", flush=True)
        if os.environ.get("OBJ_HALF_H"):
            return                                           # explicit override wins
        for prim in self.stage.Traverse():
            p = prim.GetPath().pathString
            if "pickup_obj_0" in p and prim.GetTypeName() == "Cylinder":
                cy = _UG.Cylinder(prim)
                h = float(cy.GetHeightAttr().Get() or 0.0)
                r = float(cy.GetRadiusAttr().Get() or 0.0)
                if h > 0:
                    if abs(h / 2.0 - self.obj_half_h) > 1e-4:
                        print(f">>> object size: measured half-height {h / 2.0:.3f} "
                              f"(json said {self.obj_half_h:.3f}) -> using the measurement", flush=True)
                    self.obj_half_h = h / 2.0
                if r > 0:
                    KNOWN["object_radius"] = r
                    print(f">>> object size: radius {r:.3f}, half-height {self.obj_half_h:.3f}",
                          flush=True)
                return
