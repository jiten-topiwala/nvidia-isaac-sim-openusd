"""Dock search and the forced pre-grasp pose: pick an approach angle with a clear corridor,
nav there, then force the settled arm config and the cupped fingers.

Part of the pick cycle — see `morph/pick/__init__.py` for the stage order and the shared
state object. Imports Isaac APIs at module level; only importable after SimulationApp.
"""
import math
import os

import numpy as np

import scene

from morph.config import PERFECT, HEADLESS, KNOWN, NUM_OBJECTS
from morph.geometry import wrap


class DockStage:
    """One stage of `pick`. Mixed into `PickMixin`; all `self.*` belong to `Demo`."""

    def _pick_setup(self, st):
        """Choose the dock angle, nav to it and snap the base exactly onto it."""
        obj_idx, status = st.obj_idx, st.status
        # Point the contact sensor at THIS object for the whole pick — see `_on_contact`. Without
        # it, a finger brushing any of the nine bystanders reads as a pad load here.
        self._focus_obj = f"pickup_obj_{obj_idx}"
        if PERFECT:
            # Attempt hygiene: a failed attempt leaves `_finger_cmd` at the curl targets and the
            # gains at hold stiffness, so the retry would replay the whole approach with the hand
            # CLOSED and hot gains.
            self._finger_cmd = None
            # Release the chassis lock for the new attempt. It is armed at grasp entry and normally
            # cleared when the close pin releases, but a pick that ABORTS before the pin (cycle 1
            # does, every run) would otherwise leave it set and refuse the next cycle's NAV.
            self._grasp_locked = None
            self._base_block_n = {}
            self._tighten_total = 0.0
            scene.set_arm_gains(self.robot, self.names)
        O = self.obj_xy[obj_idx]
        rx, ry, _ = self.base_pose()
        base_psi = math.atan2(O[1] - ry, O[0] - rx)         # approach from the current side, first choice
        # Try several approach angles and use the first whose dock is a VALID base pose (in-bounds,
        # clear of racks + robot radius) — an edge-spawned object (tight north strip) has no dock on
        # the near side, so we approach it from the aisle side instead.
        def dock_for(ang):
            c, s = math.cos(ang), math.sin(ang)
            off = np.array([c * self.local_obj[0] - s * self.local_obj[1], s * self.local_obj[0] + c * self.local_obj[1]])
            return O - off
        def dock_ok(d):
            x, y = float(d[0]), float(d[1])
            if not (0.35 < x < 7.65 and -7.65 < y < -0.35):
                return False
            return not self._inside_rack(x, y, 0.45)         # keep chassis-radius clear of every rack
        placed_ids = {p[0] for p in self.placed}
        others_xy = [self.obj_xy[i] for i in range(NUM_OBJECTS) if i != obj_idx and i not in placed_ids]

        def corridor_clear(d, ang):
            """Is the reach corridor from base to object clear of racks, walls and other objects?

            The reach sweeps the boom through that corridor, and anything in it meets a
            kinematically forced arm -- an irresistible-force contact, or a pass-through.
            """
            c, s = math.cos(ang), math.sin(ang)
            for r in np.arange(0.3, self.pick_standoff + 0.1, 0.15):   # +0.1: nothing physically sweeps
                # Sample only just past the object, where the descent is vertical: a longer
                # overshoot rejects every rack-adjacent object and forces permanent teleport mode.
                sx, sy = float(d[0] + c * r), float(d[1] + s * r)
                if (math.hypot(sx - O[0], sy - O[1]) > 0.35
                        and self._inside_rack(sx, sy, 0.30)):
                    # rack test skipped NEAR the object: the hand must obviously arrive there;
                    # chassis rack clearance is dock_ok's job
                    return False
                if not (0.45 < sx < 7.55 and -7.55 < sy < -0.45):
                    # The arena walls fight the sweep too, and a tighter margin still lets an object
                    # be shoved metres. Nor may the sweep pass through a bystander.
                    return False
                if any(math.hypot(sx - o[0], sy - o[1]) < 0.35 for o in others_xy):
                    return False
            return True
        psi, dock = base_psi, dock_for(base_psi)
        reach_ok = False
        best = None
        # A dense full circle: a corridor-clear angle almost always exists, and a sparse list misses
        # the wall-safe approaches.
        for da in (0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2, 1.6, -1.6, 2.0, -2.0,
                   2.4, -2.4, 2.8, -2.8, math.pi):
            cand = dock_for(base_psi + da)
            if not dock_ok(cand):
                continue
            if best is None:
                best = (base_psi + da, cand)                 # first valid dock = teleport-mode fallback
            if corridor_clear(cand, base_psi + da):
                psi, dock, reach_ok = base_psi + da, cand, True
                break
        if not reach_ok and best is not None:
            psi, dock = best
        if os.environ.get("PICK_KNOWN_POSE") == "1":        # DIAG: dock at the exact known-grasp base pose
            bx, by, byaw = KNOWN["base_xy_yaw"]
            dock, psi = np.array([bx, by]), byaw
        print(f">>> PICK setup obj {obj_idx}: O={np.round(O,3).tolist()} psi={psi:.3f} "
              f"dock={np.round(dock,3).tolist()} reach_ok={reach_ok}", flush=True)
        if status:
            status(f"nav to object {obj_idx}")
        if os.environ.get("PICK_NO_NAV") != "1":            # DIAG: skip nav, teleport base to dock
            self.drive_to((float(dock[0]), float(dock[1])), psi, status=status,
                          freeze_arm=True,                   # NO_LOOP mode: arm NEVER runs free dynamics
                          discs=self._nav_discs(target=obj_idx))   # never plow a floor object
        # Ramp the final snap. `drive_to` stops within its own tolerance, and putting the base
        # exactly on the dock in a single `set_world_pose` is a visible jump at the end of every
        # approach.
        _bx0, _by0, _byaw0 = self.base_ledger()
        _n_sn = max(1, int(0.4 / self.dt))
        _dsn = math.hypot(float(dock[0]) - _bx0, float(dock[1]) - _by0)
        if _dsn > 0.002:
            _qsn = np.asarray(self.robot.get_joint_positions(), float).copy()
            for _i in range(_n_sn):
                _f = 0.5 - 0.5 * math.cos(math.pi * (_i + 1) / _n_sn)
                self.set_base(float(_bx0 + _f * (float(dock[0]) - _bx0)),
                              float(_by0 + _f * (float(dock[1]) - _by0)),
                              float(_byaw0 + _f * wrap(psi - _byaw0)))
                self._force(_qsn)
                self._apply(_qsn)
                self.world.step(render=not HEADLESS)
            print(f">>> dock snap: ramped {_dsn * 1000:.0f}mm over {_n_sn} steps "
                  f"(was a single-frame teleport)", flush=True)
        self.set_base(float(dock[0]), float(dock[1]), psi)  # exact (grasp is pose-sensitive)
        if PERFECT:
            _op = np.asarray(self.objs[obj_idx].get_world_poses()[0][0], float)
            print(f">>> obj@post-nav: {np.round(_op, 3).tolist()} (spawn {np.round(O, 3).tolist()})", flush=True)
        # bleed off the wheel/roller spin the teleport-drive accumulated — else the fresh 3e5 arm
        # gains diverge on top of the non-finite root and joint velocities (non-finite bounds blow-
        # up).
        self.robot.set_joint_velocities(np.zeros(len(self.names)))

        # Pin the chassis HERE, at the dock -- not at the reach, and not at the close.
        if os.environ.get("PIN_AT_DOCK", "1") == "1":
            try:
                self.set_base(float(dock[0]), float(dock[1]), psi)   # exact pose, then freeze it
                self.pin_chassis(True, "dock")
            except Exception as _e_pd:
                print(f">>> pin at dock skipped ({_e_pd})", flush=True)

        st.O, st.psi, st.dock, st.reach_ok = O, psi, dock, reach_ok

    def _pick_pose(self, st):
        """Force the settled arm-1 config and the cupped finger pose, and set up the bystanders."""
        obj_idx, status = st.obj_idx, st.status
        O, reach_ok = st.O, st.reach_ok
        # Force the settled arm-1 config and a cupped finger pose -- PhysX has no linkage equality,
        # so every joint is pinned.
        qg = np.asarray(self.robot.get_joint_positions(), float).copy()
        forced = {**KNOWN["arm_joints"], **KNOWN["fingers_open"]}
        for n, v in forced.items():
            if n in self.idx:
                qg[self.idx[n]] = v
        # lightly cupped (from open ~-0.49) — a tight cup can overlap the object/palm and
        # destabilise the forced arm; 0.05 hugs enough without the
        grip_j1 = 0.05
        # blow-up risk.

        def cup_fingers(q):
            for f in "abc":
                if f"finger_{f}_joint_1_1" in self.idx:
                    q[self.idx[f"finger_{f}_joint_1_1"]] = grip_j1
            for n, sgn in (("palm_finger_b_joint_1", 1.0), ("palm_finger_c_joint_1", -1.0)):
                if n in self.idx:
                    q[self.idx[n]] = 0.35 * sgn            # modest b/c spread -> 3-point cage
        # One open pose, and the hand descends in it. Cupping the settle target unconditionally
        # means the hand travels and DESCENDS cupped, springing open only at the settle --
        # backwards, and it reads as the fingers entering the object.
        _close_anim_early = os.environ.get("CLOSE_ANIM", "1") == "1"
        if not _close_anim_early:
            cup_fingers(qg)
        replay = (self.traj is not None and os.environ.get("REACH_REPLAY", "1") == "1"
                  and reach_ok)                              # rack in the sweep corridor -> teleport this pick
        if not replay:
            # Z_DROP compensates the EE landing high when the loop-coupled config is teleported and
            # the passives settle dynamically.
            Z_DROP = 0.116
            if "ColumnLeftBearingJoint_1" in self.idx:
                qg[self.idx["ColumnLeftBearingJoint_1"]] -= Z_DROP

        # Turn off the target object's collision -- keeping it a dynamic body -- so it can rest on
        # the floor IN the reach path and later be lifted INTO the cage without any finger contact.
        obj = self.objs[obj_idx]
        park = np.array([O[0], O[1], self.obj_half_h])           # its own floor spot, visible
        if not PERFECT:
            # practical mode only: teleporting a REAL colliding body snaps a displaced object back
            # to its spawn mid-scene (NaN explosions + base<->object contact storms on retries)
            obj.set_world_poses(positions=np.array([park]), orientations=np.array([[1, 0, 0, 0]]))
            obj.set_velocities(np.zeros((1, 6)))
        else:
            O = np.asarray(obj.get_world_poses()[0][0], float)[:2]   # aim at where it REALLY is

        # Visible arm reach: kinematic replay of the baked loop-consistent trajectory.
        if status:
            status(f"reaching for object {obj_idx}")
        if PERFECT:
            _op = np.asarray(obj.get_world_poses()[0][0], float)
            print(f">>> obj@pre-reach: {np.round(_op, 3).tolist()}", flush=True)
        close_anim = os.environ.get("CLOSE_ANIM", "1") == "1"   # surface-stop close in EVERY mode
        by_pose = {}

        def pin_bystanders():
            for i, p in by_pose.items():
                self.objs[i].set_world_poses(positions=np.array([p]), orientations=np.array([[1, 0, 0, 0]]))
                self.objs[i].set_velocities(np.zeros((1, 6)))

        def restore_bystanders():
            if by_pose:
                pin_bystanders()                             # exact floor pose back BEFORE colliders return
                for i in by_pose:
                    self._obj_collision(i, True)
                by_pose.clear()
        st.qg, st.obj, st.park = qg, obj, park
        st.close_anim, st.replay = close_anim, replay
        st.cup_fingers = cup_fingers
        st.pin_bystanders, st.restore_bystanders = pin_bystanders, restore_bystanders

