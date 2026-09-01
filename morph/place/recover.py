"""Failure recovery for the place cycle.

Extracted from the flat `place.py` alongside the insert, so the package mirrors `morph/pick/`.
Imports Isaac APIs at module level; only importable after SimulationApp.
"""
import math

import numpy as np

import scene
from morph.config import PERFECT, HEADLESS, NUM_OBJECTS
from morph.geometry import quat_yaw


class RecoverStage:
    """One stage of `place`. Mixed into `PlaceMixin`; all `self.*` belong to `Demo`."""

    def _recover(self):
        """If a cycle blew the robot to a non-finite or out-of-bounds pose, reset it to a safe home so the
        NEXT cycle isn't bricked (an exploded base -> every subsequent nav 'No path').  A NaN articulation
        can't be fixed by set_world_pose alone — it needs a full world.reset() to rebuild the physics."""
        self._unweld_grip()          # a grasp weld surviving a reset re-attaches the object to the
        # Clearing it matters: a stale command would re-close the hand on the next cycle, and this
        # path is exactly where a half-finished pick lands.
        p, _ = self.robot.get_world_pose()
        x, y = float(p[0]), float(p[1])
        if math.isfinite(x) and math.isfinite(y) and 0.2 < x < 7.8 and -7.8 < y < -0.2:
            if PERFECT:
                # Real dynamic objects remember a failed attempt -- a knocked one stays where it was
                # shoved, a toppled one lies on its side -- so the retry would reach for a target
                # that has moved.
                placed_ids = {pi for pi, _ in self.placed}
                for i in range(NUM_OBJECTS):
                    if i in placed_ids:
                        continue
                    xy = self.obj_xy[i]
                    self.objs[i].set_world_poses(
                        positions=np.array([[xy[0], xy[1], self.obj_half_h]]),
                        orientations=np.array([[1, 0, 0, 0]]))
                    self.objs[i].set_velocities(np.zeros((1, 6)))
            return                                           # base is sane, nothing else to do
        print(f">>> RECOVER: base at ({x:.1f},{y:.1f}) is bad -> world.reset + re-home", flush=True)
        self.grip = None
        self._finger_cmd = None                              # fingers back under kinematic control
        self.world.reset()                                   # rebuild the physics view (clears the NaN state)
        self.robot.initialize()
        scene.set_arm_gains(self.robot, self.names)
        self.robot.set_joint_positions(self.q0)
        self.robot.set_world_pose(position=[3.70, -6.00, self.z0], orientation=quat_yaw(0.0))
        self.robot.set_joint_velocities(np.zeros(len(self.names)))
        # world.reset reverts every object to its USD spawn, so restore the scattered layout and
        # anything already placed on a shelf.
        for i in range(NUM_OBJECTS):
            xy = self.obj_xy[i]
            self.objs[i].set_world_poses(positions=np.array([[xy[0], xy[1], self.obj_half_h]]),
                                         orientations=np.array([[1, 0, 0, 0]]))
            self.objs[i].set_velocities(np.zeros((1, 6)))
        for oi, pos in self.placed:
            self.objs[oi].set_world_poses(positions=np.array([pos]), orientations=np.array([[1, 0, 0, 0]]))
            self.objs[oi].set_velocities(np.zeros((1, 6)))
        for _ in range(int(0.5 / self.dt)):
            self.robot.set_world_pose(position=[3.70, -6.00, self.z0], orientation=quat_yaw(0.0))
            self._force(self.q0)
            self._apply(self.q0)
            self.world.step(render=not HEADLESS)
        pr, _ = self.robot.get_world_pose()
        print(f">>> RECOVER done: base now ({float(pr[0]):.2f},{float(pr[1]):.2f})", flush=True)
