"""Dock search and the forced pre-grasp pose: pick an approach angle with a clear corridor, nav
there, then force the arm config and the cupped fingers. Import only after SimulationApp."""
import math
import os

import numpy as np

import scene

from morph.config import HEADLESS, KNOWN, NUM_OBJECTS
from morph.geometry import wrap
from morph.gripper.api import GRIPPER_OPEN


class DockStage:
    """One stage of `pick`. Mixed into `PickMixin`; all `self.*` belong to `Demo`."""

    def _pick_setup(self, st):
        """Choose the dock angle, nav to it and snap the base exactly onto it."""
        self._stage = "setup"
        obj_idx, status = st.obj_idx, st.status
        # Point the contact sensor at THIS object, or a finger brushing a bystander reads as a load
        self._focus_obj = f"pickup_obj_{obj_idx}"
        # Attempt hygiene: a failed attempt leaves the curl targets set, so a retry approaches CLOSED
        self._finger_cmd = None
        self._finger_pin = None
        # Release the chassis lock: a pick that aborts before the close pin would refuse the next NAV
        self._grasp_locked = None
        self._base_block_n = {}
        scene.set_arm_gains(self.robot, self.names)
        O = self.obj_xy[obj_idx]
        rx, ry, _ = self.base_pose()
        base_psi = math.atan2(O[1] - ry, O[0] - rx)
        # First angle whose dock is a VALID base pose: an edge-spawned object has none on the near side
        def dock_for(ang):
            c, s = math.cos(ang), math.sin(ang)
            off = np.array([c * self.local_obj[0] - s * self.local_obj[1], s * self.local_obj[0] + c * self.local_obj[1]])
            return O - off
        def dock_ok(d):
            x, y = float(d[0]), float(d[1])
            if not (0.35 < x < 7.65 and -7.65 < y < -0.35):
                return False
            # 0.45 = chassis radius
            return not self._inside_rack(x, y, 0.45)
        placed_ids = {p[0] for p in self.placed}
        others_xy = [self.obj_xy[i] for i in range(NUM_OBJECTS) if i != obj_idx and i not in placed_ids]

        def corridor_clear(d, ang):
            """Corridor clear? The boom sweeps it, and anything inside meets a forced arm."""
            c, s = math.cos(ang), math.sin(ang)
            for r in np.arange(0.3, self.pick_standoff + 0.1, 0.15):
                # Sample only just past the object: a longer overshoot rejects every rack-adjacent one
                sx, sy = float(d[0] + c * r), float(d[1] + s * r)
                if (math.hypot(sx - O[0], sy - O[1]) > 0.35
                        and self._inside_rack(sx, sy, 0.30)):
                    return False
                # walls fight the sweep too
                if not (0.45 < sx < 7.55 and -7.55 < sy < -0.45):
                    return False
                if any(math.hypot(sx - o[0], sy - o[1]) < 0.35 for o in others_xy):
                    return False
            return True
        psi, dock = base_psi, dock_for(base_psi)
        reach_ok = False
        best = None
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
        if not reach_ok:
            # A veto REFUSES the attempt: the reach only exists on a clear corridor, and the base
            # never drives to a dock the sweep cannot use. `best` None: no angle passed `dock_ok`.
            if best is not None:
                self._fallback("dock-no-corridor")
            else:
                self._fallback("dock-invalid")
            print(f">>> PICK setup obj {obj_idx}: O={np.round(O,3).tolist()} -- no dock with a clear "
                  f"sweep corridor -> attempt refused", flush=True)
            st.alive = False
            return
        print(f">>> PICK setup obj {obj_idx}: O={np.round(O,3).tolist()} psi={psi:.3f} "
              f"dock={np.round(dock,3).tolist()}", flush=True)
        if status:
            status(f"nav to object {obj_idx}")
        self.drive_to((float(dock[0]), float(dock[1])), psi, status=status,
                      freeze_arm=True,
                      discs=self._nav_discs(target=obj_idx))
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
        self.set_base(float(dock[0]), float(dock[1]), psi)
        _op = np.asarray(self.objs[obj_idx].get_world_poses()[0][0], float)
        print(f">>> obj@post-nav: {np.round(_op, 3).tolist()} (spawn {np.round(O, 3).tolist()})", flush=True)
        # bleed off the spin the teleport-drive accumulated, or the stiff arm gains diverge on it
        self.robot.set_joint_velocities(np.zeros(len(self.names)))

        # Pin the chassis HERE, at the dock -- not at the reach, and not at the close.
        if os.environ.get("PIN_AT_DOCK", "1") == "1":
            try:
                self.set_base(float(dock[0]), float(dock[1]), psi)
                self.pin_chassis(True, "dock")
            except Exception as _e_pd:
                print(f">>> pin at dock skipped ({_e_pd})", flush=True)

        st.O, st.psi, st.dock = O, psi, dock

    def _pick_pose(self, st):
        """Force the settled arm-1 config and the cupped finger pose, and set up the bystanders."""
        self._stage = "pose"
        obj_idx, status = st.obj_idx, st.status
        O = st.O
        if self.traj is None:
            # The reach is the bake's approach re-solved to the live object: no bake, no reach.
            self._fallback("no-baked-trajectory")
            print(f">>> PICK pose obj {obj_idx}: no baked trajectory -> attempt refused", flush=True)
            st.alive = False
            return
        # PhysX has no linkage equality, so every joint is pinned individually.
        qg = np.asarray(self.robot.get_joint_positions(), float).copy()
        forced = {**KNOWN["arm_joints"], **GRIPPER_OPEN}
        for n, v in forced.items():
            if n in self.idx:
                qg[self.idx[n]] = v

        # Target keeps collision off (still dynamic): it rests IN the reach path, then is lifted
        obj = self.objs[obj_idx]
        park = np.array([O[0], O[1], self.obj_half_h])
        O = np.asarray(obj.get_world_poses()[0][0], float)[:2]       # aim at where it REALLY is

        if status:
            status(f"reaching for object {obj_idx}")
        _op = np.asarray(obj.get_world_poses()[0][0], float)
        print(f">>> obj@pre-reach: {np.round(_op, 3).tolist()}", flush=True)
        by_pose = {}

        def pin_bystanders():
            for i, p in by_pose.items():
                self.objs[i].set_world_poses(positions=np.array([p]), orientations=np.array([[1, 0, 0, 0]]))
                self.objs[i].set_velocities(np.zeros((1, 6)))

        def restore_bystanders():
            if by_pose:
                # exact floor pose back BEFORE colliders return
                pin_bystanders()
                for i in by_pose:
                    self._obj_collision(i, True)
                by_pose.clear()
        st.qg, st.obj, st.park = qg, obj, park
        # Constant: `morph/pick/grasp.py` (off limits) still reads it.
        st.replay = True
        st.pin_bystanders, st.restore_bystanders = pin_bystanders, restore_bystanders

