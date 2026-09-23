#!/usr/bin/env python3
"""Extract arm-1's kinematic tree and per-link collision AABBs from the USD into
usd/_arm_model.json. The output is committed data; re-run only when the USD changes.
Run: "$ISAAC/python.sh" tools/extract_arm_model.py"""
import json
import math
import os
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics, Gf                      # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import morph.config  # noqa: E402,F401 -- the run's profile; without it the scene builds another robot
import scene                                                      # noqa: E402
from morph.usd_utils import find_path                             # noqa: E402

OUT = os.path.join(HERE, "usd", "_arm_model.json")

# The canonical q8 order, used by every consumer downstream: [th, h1, h2, a1, hb, wz, wy, wx].
ACTUATED = ["BaseJoint_1", "ColumnLeftBearingJoint_1", "ColumnRightBearingJoint_1",
            "ArmLeftJoint_1", "HandBearingJoint_1", "gripper_z_rotation_1",
            "gripper_y_rotation_1", "gripper_x_rotation_1"]
# Driven by the closed 4-bar, not by an actuator. The LUT supplies their values.
PASSIVES = ["RotationLeftJoint_1", "ArmRightJoint_1",
            "ContactCylinderJoint_1_1", "ContactCylinderJoint2_1"]
HAND_GROUP = ["Arm_Left_1", "Arm_Right_1", "Hand_Bearing_1",
              "Gripper_Link1_1", "Gripper_Link2_1", "Gripper_Link3_1"]
# Arm_1 is the turret the column carriages ride on: it turns with theta and is part of the swept
# volume, so it belongs here even though it never moves relative to the chassis.
COLUMN_GROUP = ["Arm_1", "Bearing_Column_Left_1", "Bearing_Column_Right_1",
                "Rotation_Link_Left_1", "Rotation_Link_Right_1",
                "Contact_Cylinder_1_1", "Contact_Cylinder_1_2"]


# Gf.Quatf/d -> [w,x,y,z]
def _q(gfq):
    if gfq is None:
        return [1.0, 0.0, 0.0, 0.0]
    im = gfq.GetImaginary()
    return [float(gfq.GetReal()), float(im[0]), float(im[1]), float(im[2])]


def _v(gfv):
    return [0.0, 0.0, 0.0] if gfv is None else [float(gfv[0]), float(gfv[1]), float(gfv[2])]


def main():
    # The demo's own loader, so prims and transforms are identical to a run.
    world, _robot, _names, _q0 = scene.load_scene(app)
    stage = world.stage
    arm_path = find_path(stage, "Arm_1")
    if arm_path is None:
        raise SystemExit("Arm_1 not on the stage")
    arm_root = stage.GetPrimAtPath(arm_path)
    robot_root = arm_root.GetParent()

    joints = []
    for prim in Usd.PrimRange(robot_root):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(prim)
        b0 = j.GetBody0Rel().GetTargets()
        b1 = j.GetBody1Rel().GetTargets()
        if not b0 or not b1:
            continue
        # arm-1's subtree only: the child must live under Arm_1, or this is BaseJoint_1 itself
        if prim.GetName() != "BaseJoint_1" and not str(b1[0]).startswith(str(arm_root.GetPath())):
            continue
        if prim.IsA(UsdPhysics.RevoluteJoint):
            rj = UsdPhysics.RevoluteJoint(prim)
            t, ax = "revolute", rj.GetAxisAttr().Get()
            lo, hi = rj.GetLowerLimitAttr().Get(), rj.GetUpperLimitAttr().Get()
            # USD stores revolute limits in DEGREES; an unlimited joint uses a far-out sentinel.
            lo = math.radians(lo) if lo is not None and lo > -1e6 else -math.pi
            hi = math.radians(hi) if hi is not None and hi < 1e6 else math.pi
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            pj = UsdPhysics.PrismaticJoint(prim)
            t, ax = "prismatic", pj.GetAxisAttr().Get()
            lo, hi = pj.GetLowerLimitAttr().Get(), pj.GetUpperLimitAttr().Get()
            lo = float(lo) if lo is not None and lo > -1e6 else 0.0
            hi = float(hi) if hi is not None and hi < 1e6 else 1.40
        elif prim.IsA(UsdPhysics.FixedJoint):
            t, ax, lo, hi = "fixed", "X", 0.0, 0.0
        else:
            # spherical loop joint etc. -- not part of the tree
            continue
        jname = prim.GetName()
        # USD allows two prims called PhysicsFixedJoint
        if jname in {j["name"] for j in joints}:
            jname = f"{jname}@{b1[0].name}"
        joints.append({"name": jname, "type": t, "axis": str(ax),
                       "parent": b0[0].name, "child": b1[0].name,
                       "localPos0": _v(j.GetLocalPos0Attr().Get()),
                       "localRot0": _q(j.GetLocalRot0Attr().Get()),
                       "localPos1": _v(j.GetLocalPos1Attr().Get()),
                       "localRot1": _q(j.GetLocalRot1Attr().Get()),
                       "lower": float(lo), "upper": float(hi)})

    # Per-link AABB of the link's OWN colliders, in its local frame, carried rigidly by FK. The USD
    # hierarchy nests along the chain, so an unpruned walk gives every parent the whole-arm box.
    child_links = {j["child"] for j in joints}
    # ignoreVisibility and every purpose: hidden collision geometry returns an EMPTY bound.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                              ["default", "render", "proxy", "guide"],
                              useExtentsHint=True, ignoreVisibility=True)
    link_aabb = {}
    for link in HAND_GROUP + COLUMN_GROUP:
        lpath = find_path(stage, link)
        if lpath is None:
            print(f"WARN: {link} is not on the stage", flush=True)
            continue
        lp = stage.GetPrimAtPath(lpath)
        xf_inv = UsdGeom.Xformable(lp).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()).GetInverse()
        box = [Gf.Vec3d(1e9, 1e9, 1e9), Gf.Vec3d(-1e9, -1e9, -1e9)]
        n = [0]

        def _accum(prim, colliders_only=True):
            if colliders_only and not prim.HasAPI(UsdPhysics.CollisionAPI):
                return
            if not colliders_only and not prim.IsA(UsdGeom.Gprim):
                return
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if rng.IsEmpty():
                return
            mn, mx = rng.GetMin(), rng.GetMax()
            lo_, hi_ = box
            for cx in (mn[0], mx[0]):
                for cy in (mn[1], mx[1]):
                    for cz in (mn[2], mx[2]):
                        p = xf_inv.Transform(Gf.Vec3d(cx, cy, cz))
                        lo_ = Gf.Vec3d(min(lo_[0], p[0]), min(lo_[1], p[1]), min(lo_[2], p[2]))
                        hi_ = Gf.Vec3d(max(hi_[0], p[0]), max(hi_[1], p[1]), max(hi_[2], p[2]))
            box[0], box[1] = lo_, hi_
            n[0] += 1

        def _walk(colliders_only):
            # TraverseInstanceProxies: the meshes are instanced; without it the columns come back
            # with no geometry at all.
            it = iter(Usd.PrimRange(lp, Usd.TraverseInstanceProxies()))
            for prim in it:
                if prim != lp and prim.GetName() in child_links:
                    # that link gets its own entry; do not double-count
                    it.PruneChildren()
                    continue
                _accum(prim, colliders_only)
            # Fingers fold in at the authored REST pose, the SMALLER box. `collision.FINGER_SWEEP_AABB`
            # unions the full reachable sweep back at load, so this is safe to re-run at any finger pose.

            # fold every finger/palm collider in
            if link == "Gripper_Link3_1":
                for prim in Usd.PrimRange(arm_root, Usd.TraverseInstanceProxies()):
                    nm = prim.GetName()
                    if nm.startswith("finger_") or nm.startswith("palm"):
                        for sub in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                            _accum(sub, colliders_only)

        _walk(True)
        src = "colliders"
        if n[0] == 0:
            # The column group carries NO PhysX colliders (the arm is driven kinematically) but still has to
            # be avoided. Render geometry is the conservative fallback: a mesh is never smaller.
            _walk(False)
            src = "render geometry (no colliders on this link)"
        if n[0] == 0:
            print(f"    {link}: NO GEOMETRY AT ALL -- omitted", flush=True)
            continue
        link_aabb[link] = [_v(box[0]), _v(box[1])]
        print(f"    {link}: {n[0]} prims from {src}  "
              f"{[round(x, 3) for x in link_aabb[link][0]]} .. "
              f"{[round(x, 3) for x in link_aabb[link][1]]}", flush=True)

    model = {"root": "base", "joints": joints, "actuated": ACTUATED, "passives": PASSIVES,
             "link_aabb": link_aabb, "hand_group": HAND_GROUP, "column_group": COLUMN_GROUP,
             "hand_link": "Gripper_Link3_1"}
    model = carry_forward(_previous(OUT), model)
    with open(OUT, "w") as fh:
        json.dump(model, fh, indent=1)
    print(f">>> wrote {OUT}: {len(joints)} joints, {len(link_aabb)} link AABBs", flush=True)



def _previous(path):
    """The model this run is about to overwrite, or {}."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def carry_forward(old, new):
    """Keep what this extractor cannot produce, and never let a stale claim ride along.

    `link_parts` is per-sub-mesh geometry measured with `tools/audit_colliders.py` and entered by
    hand; this script writes whole-link boxes only, so a plain overwrite deletes it and the arm
    silently returns to the union box it was decomposed to escape. `validated` records a LIVE FK
    check against the articulation: it describes the model it was run on, so a re-extraction does
    not inherit it. Carry it, mark it stale, and say so.
    """
    out = dict(new)
    parts = old.get("link_parts")
    if parts:
        out["link_parts"] = parts
        print(f">>> carried forward link_parts for {sorted(parts)} -- MEASURED geometry this "
              f"script cannot regenerate", flush=True)
    was = old.get("validated")
    if was:
        out["validated"] = dict(was, stale=True,
                                stale_reason="the model was re-extracted after this was recorded")
        print(">>> `validated` marked STALE: re-run tests/test_armfk_live.py before trusting it",
              flush=True)
    return out


main()
app.close()
