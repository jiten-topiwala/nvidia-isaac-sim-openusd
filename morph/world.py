"""Scene-side helpers: object scatter, colliders, rack keep-outs and the grip material.

Split out of play_isaac.py; the method bodies are unchanged. Imports Isaac APIs at module
level, so it is only importable after SimulationApp exists — see `morph/__init__.py`.
"""
import os

import numpy as np
import scene
from pxr import UsdPhysics

from morph.config import (NUM_OBJECTS, PERFECT, SPAWN_ZONES, RACK_RECTS, SPAWN_EDGE_MARGIN,
    MIN_OBJ_SEPARATION, SPAWN_ROBOT_KEEP_CENTER, SPAWN_ROBOT_KEEP_RADIUS)


class WorldMixin:
    """Mixed into Demo. Every `self.*` it touches is owned by Demo."""

    def _apply_grip_material(self):
        from isaacsim.core.api.materials.physics_material import PhysicsMaterial
        from isaacsim.core.prims import SingleGeometryPrim
        from pxr import Usd, UsdPhysics
        # The coefficient is GRIP_MU, and the mode is GRIP_MODE. Setting the internal constant's
        # name on the command line does not select a mode — it would silently change mu instead, on
        # the pads AND the floor, since this material is applied to both.
        mu = 5.0
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
        """NO-OP.  Objects are kinematic + collision-free from LOAD TIME (scene.fixup_scene): even
        the single target-object collider toggle turned out to invalidate the live articulation view
        in headed mode (apply_action -> applied_actions None -> dead session mid-reach) — the third
        member of the runtime-USD-churn crash family after the loop-joint and bystander-shield
        toggles.  Rule: NEVER touch USD physics attrs while the articulation view is live."""
        return

    def scatter_objects(self):
        """Randomly place the objects: cycle the aisle zones and rejection-sample against the rack
        and robot-start keep-outs and the inter-object spacing."""
        seed = int(os.environ.get("SEED", "0"))
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
                placed.append(np.array([0.5 + 0.4 * i, -5.0]))   # fallback
            self.objs[i].set_world_poses(positions=np.array([[placed[i][0], placed[i][1], self.obj_half_h]]),
                                         orientations=np.array([[1, 0, 0, 0]]))
            self.objs[i].set_velocities(np.zeros((1, 6)))
        # settle on the floor — hold the arm at PARK every step (never step untargeted, or the heavy
        # arm collapses under its own gain and blows up the articulation)
        for _ in range(int(0.5 / self.dt)):
            self._force(self.q0)   # kinematic (NO_LOOP: no joint holds the passives)
            self._apply(self.q0)
            self.world.step(render=False)
        for o in self.objs:
            o.set_velocities(np.zeros((1, 6)))
        self.obj_xy = [np.asarray(o.get_world_poses()[0][0], float)[:2].copy() for o in self.objs]
        print(f">>> scattered {NUM_OBJECTS} objects (seed={seed})", flush=True)

    def _nav_discs(self, target=None):
        """Keep-out discs for nav: every unplaced floor object.  r = chassis half-width (~0.45)
        + object radius + margin.  The pick TARGET gets a smaller disc (its dock is only ~0.7m
        out) — the path may come NEAR it but never through it."""
        placed_ids = {p[0] for p in self.placed}
        discs = []
        for i in range(NUM_OBJECTS):
            if i in placed_ids:
                continue
            op = np.asarray(self.objs[i].get_world_poses()[0][0], float)   # LIVE (tensor) — a
            x, y = float(op[0]), float(op[1])                              # knocked object moves
            discs.append([x, y, 0.50 if i == target else 0.55])
        return discs
