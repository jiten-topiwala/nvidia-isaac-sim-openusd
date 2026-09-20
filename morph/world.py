"""Scene-side helpers: object scatter, colliders, rack keep-outs and the grip material.
Imports Isaac APIs at module level, so it is only importable after SimulationApp exists.
"""
import os

import numpy as np
from pxr import UsdPhysics

from morph.config import (NUM_OBJECTS, SPAWN_ZONES, RACK_RECTS, SPAWN_EDGE_MARGIN,
    MIN_OBJ_SEPARATION, SPAWN_ROBOT_KEEP_CENTER, SPAWN_ROBOT_KEEP_RADIUS, fallback_spawn_xy, spawn_seed)


class WorldMixin:
    """Mixed into Demo. Every `self.*` it touches is owned by Demo."""

    def _apply_grip_material(self):
        from isaacsim.core.api.materials.physics_material import PhysicsMaterial
        from isaacsim.core.prims import SingleGeometryPrim
        from pxr import UsdPhysics
        # Applied LAST, with a strongerThanDescendants binding, so it supersedes scene.py's material on
        # every grasp shape. Friction only: scene.py's compliant contact does NOT survive this rebind.
        mu = float(os.environ.get("GRIP_MU", "5.0"))
        gmat = PhysicsMaterial(prim_path="/World/PhysicsMaterials/grip", name="gripmat",
                               static_friction=mu, dynamic_friction=mu * 0.8, restitution=0.0)
        for p in self.stage.Traverse():
            path = p.GetPath().pathString
            low = path.lower()
            if p.HasAPI(UsdPhysics.CollisionAPI) and (
                    "finger" in path or "Gripper" in path or "pickup_obj" in path
                    or "floor" in low or "ground" in low or "matplane" in low):
                try:
                    SingleGeometryPrim(path).apply_physics_material(gmat)
                except Exception:
                    pass

    @staticmethod
    def _inside_rack(x, y, margin):
        for (x0, x1, y0, y1) in RACK_RECTS:
            if (x0 - margin) <= x <= (x1 + margin) and (y0 - margin) <= y <= (y1 + margin):
                return True
        return False

    def _spawn_keepout(self, x, y):
        if self._inside_rack(x, y, SPAWN_EDGE_MARGIN):
            return True
        return np.linalg.norm(np.array([x, y]) - SPAWN_ROBOT_KEEP_CENTER) < SPAWN_ROBOT_KEEP_RADIUS

    def _obj_collision(self, obj_idx, enabled):
        """NO-OP: objects are kinematic and collision-free from load time (scene.fixup_scene).
        Toggling a collider invalidates the live articulation view (apply_action -> applied_actions
        None -> dead session). NEVER touch USD physics attrs while that view is live."""
        return

    def scatter_objects(self):
        """Place objects randomly: cycle the aisle zones, rejection-sample against the rack and
        robot-start keep-outs and the inter-object spacing."""
        seed = spawn_seed()
        rng = np.random.default_rng(seed)
        placed = []
        for i in range(NUM_OBJECTS):
            done = False
            zone_order = [(i + k) % len(SPAWN_ZONES) for k in range(len(SPAWN_ZONES))]
            for zi in zone_order:
                (zx0, zx1), (zy0, zy1) = SPAWN_ZONES[zi]
                for _ in range(700):
                    x = rng.uniform(zx0 + SPAWN_EDGE_MARGIN, zx1 - SPAWN_EDGE_MARGIN)
                    y = rng.uniform(zy0 + SPAWN_EDGE_MARGIN, zy1 - SPAWN_EDGE_MARGIN)
                    if self._spawn_keepout(x, y):
                        continue
                    p = np.array([x, y])
                    if all(np.linalg.norm(p - q) >= MIN_OBJ_SEPARATION for q in placed):
                        placed.append(p)
                        done = True
                        break
                if done:
                    break
            if not done:
                p = fallback_spawn_xy(placed, self._spawn_keepout)
                if p is None:
                    raise RuntimeError(f"no legal spawn left for object {i} of {NUM_OBJECTS}")
                placed.append(p)
            self.objs[i].set_world_poses(positions=np.array([[placed[i][0], placed[i][1], self.obj_half_h]]),
                                         orientations=np.array([[1, 0, 0, 0]]))
            self.objs[i].set_velocities(np.zeros((1, 6)))
        # Settle on the floor, holding the arm at PARK every step: stepping untargeted lets the
        # heavy arm collapse under its own gain and blow up the articulation.
        for _ in range(int(0.5 / self.dt)):
            # kinematic (NO_LOOP: no joint holds the passives)
            self._force(self.q0)
            self._apply(self.q0)
            self.world.step(render=False)
        for o in self.objs:
            o.set_velocities(np.zeros((1, 6)))
        self.obj_xy = [np.asarray(o.get_world_poses()[0][0], float)[:2].copy() for o in self.objs]
        print(f">>> scattered {NUM_OBJECTS} objects (seed={seed})", flush=True)

    def _nav_discs(self, target=None):
        """Keep-out discs [x, y, r] for nav, one per unplaced floor object; r (m) = chassis
        half-width (~0.45) + object radius + margin. The pick TARGET gets a smaller disc: its dock
        is only ~0.7 m out, so the path may pass near it but never through it."""
        placed_ids = {p[0] for p in self.placed}
        discs = []
        for i in range(NUM_OBJECTS):
            if i in placed_ids:
                continue
            op = np.asarray(self.objs[i].get_world_poses()[0][0], float)   # LIVE (tensor) — a
            x, y = float(op[0]), float(op[1])                              # knocked object moves
            discs.append([x, y, 0.50 if i == target else 0.55])
        return discs
