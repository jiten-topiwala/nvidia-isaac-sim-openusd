"""Shared Isaac Sim scene setup for the market world.

Import this *after* the entry script has created `SimulationApp(...)`: the isaacsim and pxr
modules only resolve once the Kit app is up. `load_scene()` is the single entry point, so the
USD load and the converter-artifact fixups live in one place.
"""
import json
import math
import os

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

HERE = os.path.dirname(__file__)
# "perfect" = objects are real dynamic bodies held by finger-pad contact friction (real-world
# physics); "practical" = the proven kinematic-puppet fallback.
GRIP_MODE = os.environ.get("GRIP_MODE", "practical")   # practical = the verified demo mode
USD = os.path.join(HERE, "usd/market_world_m1/market_world_m1.usda")
PRODUCTS_USD = os.path.join(HERE, "usd/products_textured.usd")
PRODUCTS_MANIFEST = os.path.join(HERE, "usd/_products.json")
HOME_KEYFRAME = os.path.join(HERE, "usd/_home_keyframe.json")
MARBLE = os.path.join(HERE, "usd/market_world_m1/Textures/white_marble_tile2.png")

# The parallel-linkage equality that holds the arm up in MuJoCo doesn't convert, so force the arm to
# the play.py PARK pose (else it sags into the chassis).
PARK_OVERRIDE = {
    "ColumnLeftBearingJoint": 1.2,
    "ColumnRightBearingJoint": 1.2,
    "ArmLeftJoint": 0.1,
}


def fixup_scene(stage) -> None:
    """Undo two converter artifacts: (1) jointless static env bodies import as dynamic
    rigid bodies with bad mass and fall through the floor -> make them static; (2) the
    `grasp_left*` welds import active and snap the 10 objects to the gripper -> delete."""
    geo = stage.GetPrimAtPath("/World/Geometry")
    n_static = 0
    n_kin = 0
    if geo.IsValid():
        for child in geo.GetChildren():
            name = child.GetName()
            if name == "robot":
                continue
            if name.startswith("pickup_obj"):
                if GRIP_MODE == "perfect":
                    # Perfect mode: real dynamic bodies with gravity and collision, held by pad
                    # friction alone. The depenetration velocity is kept low so a squeeze cannot
                    # eject the object.
                    pb = PhysxSchema.PhysxRigidBodyAPI.Apply(child)
                    # The depenetration kick is what throws the object. The arm is kinematically
                    # forced, so any overlap -- even a fraction of a millimetre -- is resolved by
                    # pushing the dynamic body out at up to this speed.
                    pb.CreateMaxDepenetrationVelocityAttr(
                        0.05)
                    # hard ceiling on how fast a pickup object can ever travel — nothing in this
                    # scene legitimately moves faster, and it bounds every remaining solver artifact
                    pb.CreateMaxLinearVelocityAttr(1.0)
                    # Maximum contact impulse: the bound on what one contact may do in a single
                    # step, and the honest model of a compliant rubber pad.
                    imp = 0.10
                    if imp > 0:
                        pb.CreateMaxContactImpulseAttr(imp)
                    # threshold 0 -> EVERY persistent contact reports each step (the touch sensor /
                    # pad-load telemetry); the default threshold hides resting low-force contacts
                    cr = PhysxSchema.PhysxContactReportAPI.Apply(child)
                    cr.CreateThresholdAttr(0.0)
                    n_kin += 1
                    continue
                # Practical mode: gravity-free, collision-free dynamic puppets, moved by
                # `set_world_poses`.
                pb = PhysxSchema.PhysxRigidBodyAPI.Apply(child)
                pb.CreateDisableGravityAttr(True)
                for p in Usd.PrimRange(child):
                    if p.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
                n_kin += 1
                continue
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(child).CreateRigidBodyEnabledAttr(False)
                for gc in list(child.GetChildren()):
                    if "Joint" in gc.GetTypeName():
                        gc.SetActive(False)  # redundant world-fixed joint (in a payload)
                n_static += 1
    n_weld = 0
    phys = stage.GetPrimAtPath("/World/Physics")
    if phys.IsValid():
        for child in list(phys.GetChildren()):
            if child.GetName().startswith("grasp_left"):
                # RemovePrim on a payload-composed prim FAILS SILENTLY (it edits the root layer,
                # where the prim isn't defined) — the palm<->object FixedJoints stayed alive for the
                # entire project.
                child.SetActive(False)
                n_weld += 1
    left = [p.GetPath().pathString for p in stage.Traverse()
            if p.GetName().startswith("grasp_left") and p.IsActive()]
    if left:
        print(f">>> WARNING: {len(left)} grasp welds STILL ACTIVE: {left[:2]}", flush=True)
    mode_note = "REAL dynamic (friction grip)" if GRIP_MODE == "perfect" else "gravity-free+collision-free"
    print(f">>> fixup: {n_static} env bodies static, {n_weld} grasp welds removed, "
          f"{n_kin} pickup objects {mode_note}", flush=True)


def add_lighting(stage) -> None:
    """MuJoCo's single <light> doesn't convert -> RTX renders black. Add dome + sun."""
    UsdLux.DomeLight.Define(stage, "/World/Lighting/Dome").CreateIntensityAttr(800.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Lighting/Sun")
    sun.CreateIntensityAttr(2500.0)
    sun.CreateAngleAttr(1.0)
    sun.AddRotateXYZOp().Set((-45.0, 0.0, 0.0))


def add_marble_floor(stage) -> None:
    """Converter stretches the floor marble over the whole 8x8 plane; MuJoCo tiles it
    4x4. MatPlane is behind an instance proxy/payload (un-patchable on the session
    layer), so author an 8x8 quad with the marble tiled 4x4 and hide the converter
    floor (its collision plane stays)."""
    if not os.path.exists(MARBLE):
        return
    f = stage.GetPrimAtPath("/World/Geometry/floor")
    if f.IsValid():
        UsdGeom.Imageable(f).MakeInvisible()
    z = 0.002
    q = UsdGeom.Mesh.Define(stage, "/World/MarbleFloor")
    q.CreatePointsAttr([(0, -8, z), (8, -8, z), (8, 0, z), (0, 0, z)])
    q.CreateFaceVertexCountsAttr([4])
    q.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    q.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    q.CreateNormalsAttr([(0, 0, 1)] * 4)
    UsdGeom.PrimvarsAPI(q).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    ).Set(Vt.Vec2fArray([(0, 0), (4, 0), (4, 4), (0, 4)]))
    mat = UsdShade.Material.Define(stage, "/World/MarbleFloor/Mat")
    pbr = UsdShade.Shader.Define(stage, "/World/MarbleFloor/Mat/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    rdr = UsdShade.Shader.Define(stage, "/World/MarbleFloor/Mat/stReader")
    rdr.CreateIdAttr("UsdPrimvarReader_float2")
    rdr.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, "/World/MarbleFloor/Mat/Tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(MARBLE)
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rdr.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    pbr.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(q).Apply(q.GetPrim())
    UsdShade.MaterialBindingAPI(q).Bind(mat)


def swap_in_textured_products(stage) -> None:
    """Reference the re-authored textured products (build_textured_products.py) and hide
    the converter's flat-colored originals. Skips silently if not built."""
    if not os.path.exists(PRODUCTS_USD):
        print(">>> textured products not built (run gen_/build_textured_products.py)", flush=True)
        return
    add_reference_to_stage(usd_path=PRODUCTS_USD, prim_path="/World/TexturedProducts")
    for item in json.load(open(PRODUCTS_MANIFEST)):
        p = stage.GetPrimAtPath(f"/World/Geometry/{item['hide_prim']}")
        if p.IsValid():
            UsdGeom.Imageable(p).MakeInvisible()


def find_articulation_root(stage):
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return prim.GetPath().pathString
    return None


def apply_link_masses(stage, robot_root):
    """Author REAL mass/inertia (from the MuJoCo model) on the robot links. The MJCF->USD
    import left many links with identity {1,1,1} inertia or NEGATIVE mass — especially the
    MASSLESS parallel-linkage frames (base_footprint, Rotation_Link_Right, camera_pov, the
    articulation root): mass 0 in MuJoCo is fine, but PhysX makes them zero/negative-mass
    rigid bodies that explode to non-finite on ANY contact impulse (the finger<->object grasp
    NaN). Transfer MuJoCo's valid masses; give the massless frames a small stable default.
    Must run BEFORE world.reset() so the articulation initialises with real inertia."""
    path = os.path.join(HERE, "usd/_link_masses.json")
    if not os.path.exists(path):
        print(">>> no _link_masses.json (run extract_link_masses.py) — SKIPPING mass fix", flush=True)
        return
    masses = json.load(open(path))
    root = stage.GetPrimAtPath(robot_root)
    n_real = n_default = 0
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        b = masses.get(prim.GetName())
        if b is None:
            continue
        mapi = UsdPhysics.MassAPI.Apply(prim)
        if b["mass"] > 1e-9:
            is_finger = "finger" in prim.GetName()
            # Floor the finger-link mass. The real links are ~0.007kg against a ~1kg object, and a
            # link that light cannot absorb the contact impulse -- it goes non-finite on first
            # touch.
            fm = 0.20
            m_link = max(b["mass"], fm) if is_finger else b["mass"]
            mapi.CreateMassAttr(m_link)
            if is_finger:
                ii = max(m_link * 9.0e-4, *b["inertia"])         # 9e-4 = (0.03m)^2 ; keep real if larger
                inertia = Gf.Vec3f(ii, ii, ii)
            else:
                inertia = Gf.Vec3f(*[max(x, 1e-6) for x in b["inertia"]])
            mapi.CreateDiagonalInertiaAttr(inertia)
            q = b["iquat"]
            mapi.CreatePrincipalAxesAttr(Gf.Quatf(q[0], q[1], q[2], q[3]))   # (w, x, y, z)
            mapi.CreateCenterOfMassAttr(Gf.Vec3f(*b["ipos"]))
            n_real += 1
        else:                                    # massless MuJoCo frame -> small stable default
            mapi.CreateMassAttr(0.1)
            mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-3, 1e-3, 1e-3))
            mapi.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
            n_default += 1
    print(f">>> apply_link_masses: {n_real} real + {n_default} massless-defaulted links", flush=True)


def close_arm_loop(stage, robot_root):
    """Reproduce the MORPH parallel-linkage MuJoCo `<connect>` ball constraint the MJCF->USD import
    dropped: a SPHERICAL joint (3 translation DOF locked, rotation free) between Contact_Cylinder_1_1
    and Contact_Cylinder_1_2, whose origins coincide (0.82mm) at the assembled pose (verified via
    verify_loop_geom.py). `excludeFromArticulation=True` keeps the articulation a clean tree and adds
    this as a MAXIMAL-coordinate loop constraint -> the passive linkage joints (RotationLeft/ArmRight/
    ContactCylinder) are held by PHYSICS, not forced drives. This is the real fix for the 12cm offset
    + the grip-load arm fling. The forced passive-joint drives MUST be slackened (kp=0) or they fight."""
    def path_of(name):
        for p in Usd.PrimRange(stage.GetPrimAtPath(robot_root)):
            if p.GetName() == name:
                return p.GetPath()
        return None
    a, b = path_of("Contact_Cylinder_1_1"), path_of("Contact_Cylinder_1_2")
    if a is None or b is None:
        print(f">>> close_arm_loop: bodies not found (a={a}, b={b}) — SKIP", flush=True)
        return
    j = UsdPhysics.SphericalJoint.Define(stage, Sdf.Path(robot_root + "/LoopJoint_arm1"))
    j.CreateBody0Rel().SetTargets([a])
    j.CreateBody1Rel().SetTargets([b])
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))          # anchor at each body origin (they coincide)
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    j.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    j.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    j.CreateExcludeFromArticulationAttr().Set(True)         # <-- the loop-closure flag
    j.CreateCollisionEnabledAttr().Set(False)
    print(f">>> close_arm_loop: spherical loop joint {a.name}<->{b.name} (excludeFromArticulation)", flush=True)


def fix_finger_collision(stage, robot_root):
    """Replace the crude thin box finger colliders (2.8x0.76x1.8cm pads that UNDER-cover the real
    3.9x2.15x1.8cm fingertip -> glancing point-contact on the 15cm cylinder -> 0.04N, no grip) with a
    CONVEX-HULL collider off the REAL finger mesh (link_*_real_V2) — exactly what MuJoCo collides with.
    The mesh is INSTANCED (shared prototype), so a collider API can't be authored through the instance
    proxy; the prior attempt found 0 meshes. FIX: de-instance the finger meshes FIRST (pass A), then in
    a fresh traversal author convexHull collision on the now-real Mesh + disable the boxes (pass B).
    Arm-1 fingers only (the grasping arm). Isaac-only. PAD_COLLIDER=0 to skip (keep the boxes)."""
    root = stage.GetPrimAtPath(robot_root)

    def is_finger1(prim):                                   # arm-1 finger subtree only
        p = prim.GetPath().pathString
        return "finger_" in p and "Arm_1" in p

    # Which finger links collide. On a large object the proximal and middle links sit inside it at
    # the open pose, so rigid contact flings it -- there, tip-only is correct.
    tip_only = os.environ.get("TIP", "all" if GRIP_MODE == "perfect" else "3") == "3"
    def is_tip(prim):
        p = prim.GetPath().pathString
        return (not tip_only) or "link_3_1" in p

    # pass A: de-instance the fingertip meshes so their Mesh children become editable
    n_deinst = 0
    # De-instance the proximal links too when the intent is to switch them OFF.
    _prox_off = tip_only and os.environ.get("PROX_MESH_OFF", "1") == "1"
    for prim in Usd.PrimRange(root):
        if (is_finger1(prim) and (is_tip(prim) or _prox_off)
                and prim.GetName().endswith("real_V2") and prim.IsInstance()):
            prim.SetInstanceable(False)
            n_deinst += 1

    # pass B: fresh traversal (sees the de-instanced children) — convex collider on each fingertip
    # Mesh, disable EVERY finger box (link_1/2/3).
    n_mesh = n_box = n_prox_off = 0
    for prim in Usd.PrimRange(root):
        if not is_finger1(prim):
            continue
        t = prim.GetTypeName()
        path = prim.GetPath().pathString
        if t == "Cube" and prim.HasAPI(UsdPhysics.CollisionAPI):
            # These Cubes are the source model's fingertip pad geoms and carry the entire grip in
            # the recorded grasp.
            if os.environ.get("KEEP_BOXES", "0") == "1" and is_tip(prim):
                continue
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)   # turn OFF the box collider
            n_box += 1
        elif (t == "Mesh" and "real_V2" in path and is_tip(prim)
              and os.environ.get("MESH_PADS", "1") == "1"):                   # the real fingertip mesh
            # MESH_PADS=0 with KEEP_BOXES=1 reproduces the source collision model exactly: pad boxes
            # only.
            UsdPhysics.CollisionAPI.Apply(prim)
            # convexHull of the NON-convex finger mesh balloons + deep-overlaps; convexDecomposition
            # splits it into shape-following convex chunks (thinner, follows the pad).
            approx = os.environ.get("APPROX", "convexDecomposition")
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approx)
            n_mesh += 1
        elif (t == "Mesh" and "real_V2" in path and not is_tip(prim)
              and prim.HasAPI(UsdPhysics.CollisionAPI)
              and os.environ.get("PROX_MESH_OFF", "1") == "1"):
            # Free the proximal links. Disabling their pad boxes is not enough: the finger meshes
            # carry CollisionAPI from the source asset, so with TIP=3 the link_1/link_2 meshes still
            # collide and still block -- a proximal link ends up inside the object while that
            # finger's pad is far out.
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            n_prox_off += 1
    n_palm = 0
    if os.environ.get("DIAG_NOWHEEL") == "1":
        n_offw = 0
        for prim in Usd.PrimRange(root):
            pth = prim.GetPath().pathString
            if ("roller" in pth or "wheel" in pth) and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                n_offw += 1
        print(f">>> DIAG_NOWHEEL: {n_offw} wheel/roller colliders OFF", flush=True)
    if os.environ.get("DIAG_NOHAND") == "1":
        # Diagnostic: kill every collider in the arm-1 hand subtree — if the descent STILL nudges
        # the object, the toucher was never the hand (wrist/boom/chassis/field effect)
        n_off2 = 0
        for prim in Usd.PrimRange(root):
            pth = prim.GetPath().pathString
            if "Arm_1" in pth and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                n_off2 += 1
        print(f">>> DIAG_NOHAND: {n_off2} arm-1 colliders OFF", flush=True)
    if GRIP_MODE == "perfect":
        # The palm block's lip dips into the object at grasp depth. It is not a grip surface, and
        # where the source model's soft contact tolerated the incidental press, rigid contact shoves
        # the object away.
        def _palm_hit(pth):
            if (("Hand_Bearing_1" in pth or "Gripper_Link1_1" in pth)
                    and "Gripper_Link2_1" not in pth):
                # WRIST bodies (Hand_Bearing, Gripper_Link1): PhysX-verified plowing the object
                # during the mouth slide — structure, not grip surface
                return True
            if "link_0_1" in pth and "link_1_1" not in pth:
                # finger KNUCKLE mounts (link_0): fixed blocks, not grip surfaces
                return True
            if "Gripper_Link2_1/Sphere" in pth:
                # Wrist bearing sphere: bearing structure between Link1 and the palm, never a grip
                # surface — same rule as the other wrist bodies above.
                return True
            # PALM (Gripper_Link2/3 meshes) stays COLLIDABLE: it is the cage's BACKSTOP — with it
            # off the thumb pushed the object straight out the back (13cm, both objects).
            return False
        for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
            pth = prim.GetPath().pathString
            if _palm_hit(pth) and prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsInstanceProxy():
                anc = prim
                while anc and not anc.IsInstance():
                    anc = anc.GetParent()
                if anc and anc.IsInstance():
                    anc.SetInstanceable(False)
        for prim in Usd.PrimRange(root):
            pth = prim.GetPath().pathString
            if _palm_hit(pth) and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                n_palm += 1
    # The mouth is only a mouth if the collider is concave. `palm_real_V2` is a concave mesh
    # carrying a convexHull collider, and a convex hull of a concave palm FILLS THE CAVITY -- the
    # visual shows an opening the physics does not have.
    n_papx = 0
    _papx = os.environ.get("PALM_APPROX", "convexDecomposition")
    if GRIP_MODE == "perfect" and _papx:
        # COLLECT the ancestors first: de-instancing DURING the traversal expires the iterator
        # ("Iterator points to expired 'Mesh' instance proxy prim"), because making an ancestor non-
        # instanceable invalidates every proxy under it.
        _anc = []
        for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
            pth = prim.GetPath().pathString
            if "palm" not in pth or prim.GetTypeName() != "Mesh":
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or not prim.IsInstanceProxy():
                continue
            a = prim
            while a and not a.IsInstance():
                a = a.GetParent()
            if a and a.IsInstance():
                _anc.append(a.GetPath())
        for _pp in set(_anc):
            a = stage.GetPrimAtPath(_pp)
            if a and a.IsInstance():
                a.SetInstanceable(False)
        for prim in Usd.PrimRange(root):
            pth = prim.GetPath().pathString
            if "palm" not in pth or prim.GetTypeName() != "Mesh":
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            try:
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(_papx)
                n_papx += 1
            except Exception:
                pass
        print(f">>> palm collider: {n_papx} meshes -> {_papx} (was convexHull, which fills the "
              f"mouth cavity)", flush=True)
    print(f">>> fix_finger_collision: de-instanced {n_deinst} tip meshes, {n_mesh} convex colliders, "
          f"{n_prox_off} proximal meshes off, {n_box} boxes disabled (TIP_ONLY={tip_only}), {n_palm} palm-block colliders off", flush=True)


def resize_object(stage, radius, half_h=None):
    """Shrink the pickup object's cylinder RADIUS (keep height for z-reach) so it's within the
    gripper's stable-grasp size — a 15cm object is too big to wrap (proximal links penetrate) and a
    fingertip pinch squirts it out; ~11cm (r=0.055) pinches cleanly. Sets radius on every Cylinder
    under pickup_obj_0 (visual + collision) before world.reset so PhysX cooks the new shape. OBJ_RADIUS env."""
    n = 0
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if "pickup_obj" in p and prim.GetTypeName() == "Cylinder":
            cy = UsdGeom.Cylinder(prim)
            old = cy.GetRadiusAttr().Get()
            cy.GetRadiusAttr().Set(radius)
            # extent must track the size or the bbox/broadphase uses the old shape
            if half_h is not None:
                # TALLER objects: the pads then land mid-FLANK (horizontal normals = real clamp)
                # instead of on the top rim
                cy.GetHeightAttr().Set(2.0 * half_h)
            h = cy.GetHeightAttr().Get() or 0.28
            cy.GetExtentAttr().Set([(-radius, -radius, -h / 2), (radius, radius, h / 2)])
            # Author the mass, or resizing silently makes the object a feather. Nothing sets one, so
            # PhysX derives it from the shape at a very low effective density, and shrinking the
            # cylinder cuts the mass with the VOLUME -- a smaller object is also several times
            # lighter.
            _dens = float(os.environ.get("OBJ_DENSITY", "800.0"))
            if _dens > 0:
                body = prim.GetParent() if prim.GetParent().HasAPI(UsdPhysics.RigidBodyAPI) else prim
                _vol = math.pi * radius * radius * h
                UsdPhysics.MassAPI.Apply(body).CreateMassAttr(float(_dens * _vol))
                _m_auth = _dens * _vol
            n += 1
    print(f">>> resize_object: {n} cylinders -> radius {radius} (was ~{old if n else '?'})", flush=True)
    if n and float(os.environ.get("OBJ_DENSITY", "800.0")) > 0:
        print(f">>> object mass authored: {_m_auth:.3f} kg at {os.environ.get('OBJ_DENSITY', '800.0')}"
              f" kg/m3 (tipping threshold at a 0.147m contact height now "
              f"{_m_auth * 9.81 * radius / 0.147:.2f}N)", flush=True)


def tune_physics(stage, robot_root):
    """Tighten the solver for the finger-object grasp.

    PhysX's defaults are too loose for it: rigid-contact impulses spike and the light finger links
    go non-finite. This raises the solver iteration counts (TGS), runs finer substeps, and gives
    the object and finger colliders a small contact offset so contact engages gradually rather
    than all at once.
    """
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            sc = PhysxSchema.PhysxSceneAPI.Apply(prim)
            sc.CreateSolverTypeAttr("TGS")
            sc.CreateTimeStepsPerSecondAttr(240)                 # 240Hz substeps (was 60)
            # resolve dynamic contacts AFTER the articulation — grip-specific, cuts persistent
            # penetration + gives stronger, stabler grasp contact forces
            prim.CreateAttribute("physxScene:solveArticulationContactLast",
                                 Sdf.ValueTypeNames.Bool).Set(True)
            break
    # 64/16 is validated for this KINEMATICALLY-FORCED linkage. The usual TGS advice — velocity
    # iterations 0 — is written for drive-based robots; applied here the forced arm sags well below
    # its commanded pose with several rad/s of joint oscillation.
    VEL_ITERS = 16
    ar = stage.GetPrimAtPath(robot_root)
    if ar.IsValid():
        pa = PhysxSchema.PhysxArticulationAPI.Apply(ar)
        pa.CreateSolverPositionIterationCountAttr(64)
        pa.CreateSolverVelocityIterationCountAttr(VEL_ITERS)
        # Self-collision: opt-in, and selective when on. The reason to disable it is real but narrow
        # -- converted MJCF hands generate phantom finger-versus-finger contacts, a well-known
        # articulation-explosion source.
        _sc = os.environ.get("SELF_COLLIDE", "0") == "1"
        pa.CreateEnabledSelfCollisionsAttr(_sc)
        if _sc:
            _links = {}
            for _pr in Usd.PrimRange(stage.GetPrimAtPath(robot_root)):
                _nm = _pr.GetName()
                if _pr.HasAPI(UsdPhysics.RigidBodyAPI) and (
                        _nm.startswith("finger_") or _nm.startswith("Gripper_Link")
                        or _nm.startswith("palm")):
                    _links[_nm] = _pr
            _n_filt = 0
            for _nm, _pr in _links.items():
                _fp = UsdPhysics.FilteredPairsAPI.Apply(_pr)
                _rel = _fp.CreateFilteredPairsRel()
                for _on, _op in _links.items():
                    if _on == _nm:
                        continue
                    # same finger (its own phalanges) or anything vs the palm/gripper shell
                    _same = (_nm[:8] == _on[:8] and _nm.startswith("finger_"))
                    _palm = _on.startswith("Gripper_Link") or _nm.startswith("Gripper_Link")
                    if _same or _palm:
                        _rel.AddTarget(_op.GetPath())
                        _n_filt += 1
            print(f">>> self-collision ON with {_n_filt} filtered hand-internal pairs "
                  f"(arm/gripper vs CHASSIS now collides; finger-vs-finger and finger-vs-palm "
                  f"stay off -- those are the phantom-contact explosion source)", flush=True)
    # finger/gripper LINKS: cap velocity + depenetration so a swept object contact can't fling a
    # light link to non-finite (values around 1e6). This caps the LINK directly, not just the obj.
    n_fl = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_root)):
        nm = prim.GetName()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and ("finger" in nm or "Gripper" in nm or "palm" in nm):
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rb.CreateMaxDepenetrationVelocityAttr(1.0)
            rb.CreateMaxLinearVelocityAttr(3.0)
            rb.CreateMaxAngularVelocityAttr(20.0)
            rb.CreateSolverPositionIterationCountAttr(64)
            rb.CreateSolverVelocityIterationCountAttr(VEL_ITERS)
            n_fl += 1
    print(f">>> finger-link stabilize: {n_fl} links (vel + depenetration cap + iters)", flush=True)
    # The conversion clamped the b/c finger joint_1 lower limit ABOVE the commanded open pose, so
    # those fingers stick nearly closed and cannot grip.
    n_lim = 0
    for prim in stage.Traverse():
        nm = prim.GetName()
        if prim.IsA(UsdPhysics.RevoluteJoint) and nm.startswith("finger_") and "joint_1" in nm:
            j = UsdPhysics.RevoluteJoint(prim)
            lo = j.GetLowerLimitAttr().Get()
            j.GetLowerLimitAttr().Set(-90.0)
            if n_lim < 2:
                print(f">>>   {nm} lowerLimit {lo} -> -90 (unit tell: 0.05=rad, 2.8=deg)", flush=True)
            n_lim += 1
    print(f">>> widened {n_lim} finger joint_1 lower limits", flush=True)
    n_off = 0
    geo = stage.GetPrimAtPath("/World/Geometry")
    for child in geo.GetChildren() if geo.IsValid() else []:
        if child.GetName().startswith("pickup_obj"):
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(child)
            rb.CreateSolverPositionIterationCountAttr(64)
            rb.CreateSolverVelocityIterationCountAttr(VEL_ITERS)
            # hard velocity cap + damping: a bad swept-penetration contact can't launch the object
            # across the room (was hitting 500+ m/s); it stays put and settles into grip.
            rb.CreateMaxLinearVelocityAttr(1.5)
            rb.CreateMaxAngularVelocityAttr(8.0)
            # Cap the depenetration velocity. This, not the hand, launches the object: the position-
            # controlled hand is effectively infinite mass, so it presses the object into the static
            # FLOOR and the solver resolves that overlap by pushing them apart.
            rb.CreateMaxDepenetrationVelocityAttr(
                0.10)
            # High linear damping resists the early one-sided shove during the close: the first
            # finger to touch would otherwise push the free object out before the others engage.
            rb.CreateLinearDampingAttr(float(os.environ.get("OBJ_DAMP", "0.5")))
            rb.CreateAngularDampingAttr(float(os.environ.get("OBJ_DAMP", "0.5")))
            cr = PhysxSchema.PhysxContactReportAPI.Apply(child)   # enable get_net_contact_forces
            cr.CreateThresholdAttr(0.0)
    # The import bound a mu=0 material to the finger PADS, so the grip was frictionless and every
    # object slid out on lift.
    MU = float(os.environ.get("FRICTION", "5.0"))            # test knob (FRICTION=0.01 to check if it's applied)
    grip_mat = UsdShade.Material.Define(stage, "/World/Physics/GripMaterial")
    gp = UsdPhysics.MaterialAPI.Apply(grip_mat.GetPrim())
    gp.CreateStaticFrictionAttr(MU)                          # match MuJoCo finger geom friction="5 5 5"
    gp.CreateDynamicFrictionAttr(MU * 0.8)
    gp.CreateRestitutionAttr(0.0)
    # combine mode MAX: default 'average' halves the pad friction against a lower-friction partner —
    # with max, the pad's mu always wins the pairing (NVIDIA gripper guidance; the analog of
    # MuJoCo's priority=1 finger geoms winning over the object's weaker triplet)
    PhysxSchema.PhysxMaterialAPI.Apply(grip_mat.GetPrim()).CreateFrictionCombineModeAttr("max")
    touched = set()

    def make_compliant(mprim):
        """COMPLIANT CONTACT = MuJoCo's solref, ported.  MuJoCo's pads declare
        solref="0.005 1": contact behaves as a spring-damper with time constant 5ms and damping
        ratio 1, so force RAMPS with penetration over ~10 steps.  PhysX's default rigid contact
        instead resolves the whole overlap in one step, which is why a pad arriving at 40mm/s
        handed this cylinder metres per second and, once bounded, still
        TIPPED it — a 300mm cylinder on a 160mm base goes over at a few newtons.
        k = m/tau^2 = 1.025/0.005^2 = 4.1e4 N/m; c = 2*sqrt(k*m) for ratio 1."""
        api = PhysxSchema.PhysxMaterialAPI.Apply(mprim)
        k = 4.1e4
        if k <= 0:
            return False
        c = float(os.environ.get("PAD_DAMP", str(2.0 * (k * 1.025) ** 0.5)))
        try:
            api.CreateCompliantContactStiffnessAttr(k)
            api.CreateCompliantContactDampingAttr(c)
            return True
        except Exception:
            return False
    n_compliant = int(make_compliant(grip_mat.GetPrim()))
    # The floor needs it too. The loop below matches the hand and the objects, so the GROUND keeps
    # whatever the converter gave it and the object-versus-floor pair resolves near that instead.
    n_floor = 0
    if GRIP_MODE == "perfect" and os.environ.get("FLOOR_MU_ON", "1") == "1":
        _fmu = float(os.environ.get("FLOOR_MU", str(MU)))
        _fkeys = ("floor", "Floor", "Plane", "ground", "Ground")
        _fanc = []
        for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
            _p = prim.GetPath().pathString
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or not any(s in _p for s in _fkeys):
                continue
            if prim.IsInstanceProxy():
                _a = prim
                while _a and not _a.IsInstance():
                    _a = _a.GetParent()
                if _a and _a.IsInstance():
                    _fanc.append(_a.GetPath())
        for _pp in set(_fanc):
            _a = stage.GetPrimAtPath(_pp)
            if _a and _a.IsInstance():
                _a.SetInstanceable(False)
        for prim in stage.Traverse():
            _p = prim.GetPath().pathString
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or not any(s in _p for s in _fkeys):
                continue
            try:
                _m, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
                if _m and _m.GetPrim().IsValid():
                    _mp = UsdPhysics.MaterialAPI.Apply(_m.GetPrim())
                    _mp.CreateStaticFrictionAttr(_fmu)
                    _mp.CreateDynamicFrictionAttr(_fmu * 0.8)
                    PhysxSchema.PhysxMaterialAPI.Apply(
                        _m.GetPrim()).CreateFrictionCombineModeAttr("max")
                else:
                    UsdShade.MaterialBindingAPI.Apply(prim)
                    UsdShade.MaterialBindingAPI(prim).Bind(grip_mat, materialPurpose="physics")
                n_floor += 1
            except Exception:
                pass
        print(f">>> floor friction: mu {_fmu} on {n_floor} ground colliders "
              f"(was ~0.7 effective; breakaway 10N vs the wrap's 14N thumb)", flush=True)
    # De-instance the grasp colliders first, or mu lands on nothing. The audit above finds enabled
    # grasp colliders that a plain Traverse cannot see: they sit behind instance proxies under the
    # Hand_Bearing links, and a material cannot be authored on a proxy -- so they keep whatever the
    # converter gave them and the pad-versus-object pair resolves at that value whatever MU says.
    if os.environ.get("MU_DEINSTANCE", "1") == "1":
        _gkeys = ("finger", "Gripper", "palm", "pickup_obj")
        _ganc = []
        for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
            _p = prim.GetPath().pathString
            if not (prim.HasAPI(UsdPhysics.CollisionAPI) and any(k in _p for k in _gkeys)):
                continue
            if prim.IsInstanceProxy():
                _a = prim
                while _a and not _a.IsInstance():
                    _a = _a.GetParent()
                if _a and _a.IsInstance():
                    _ganc.append(_a.GetPath())
        for _pp in set(_ganc):
            _a = stage.GetPrimAtPath(_pp)
            if _a and _a.IsInstance():
                _a.SetInstanceable(False)
        if _ganc:
            print(f">>> mu: de-instanced {len(set(_ganc))} ancestors so the grasp colliders behind "
                  f"instance proxies can carry the friction material", flush=True)

    _reported = set()
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not (prim.HasAPI(UsdPhysics.CollisionAPI) and
                any(s in path for s in ("finger", "Gripper", "palm", "pickup_obj"))):
            continue
        co = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        # Contact reporting on the hand itself. PhysX only raises contact events for actors carrying
        # PhysxContactReportAPI, so with it on the objects alone every ledger can only see pairs
        # containing a pickup object -- a finger jammed against the floor, the palm or its own
        # neighbour raises no event at all.
        _bp = prim
        while _bp and _bp.IsValid() and not _bp.HasAPI(UsdPhysics.RigidBodyAPI):
            _bp = _bp.GetParent()
        if (_bp and _bp.IsValid() and _bp.GetPath() not in _reported
                and os.environ.get("HAND_CONTACT_REPORT", "1") == "1"):
            try:
                PhysxSchema.PhysxContactReportAPI.Apply(_bp).CreateThresholdAttr(0.0)
                _reported.add(_bp.GetPath())
            except Exception:
                pass
        # The contact offset is the grasp's shock absorber: contacts start being generated this far
        # out, so a pad decelerates over a band instead of being discovered already overlapping.
        co.CreateContactOffsetAttr(float(os.environ.get(
            "CONTACT_OFF", "0.008" if GRIP_MODE == "perfect" else "0.02")))
        # The rest offset models compliant PAD THICKNESS. The contact offset only decides where
        # contacts are generated; force still needs compression past the rest offset, so at 0.0 a
        # pad sitting a couple of millimetres off the surface carries exactly zero load -- and the
        # fingers cannot close further, because they are already at their arc minimum.
        _rest_def = "0.003" if os.environ.get("GRIP_MODE", "practical") == "perfect" else "0.0"
        co.CreateRestOffsetAttr(float(os.environ.get("REST_OFF", _rest_def)))
        # Torsional friction. The source gripper grips with condim="6" (tangential + torsional +
        # rolling); the PhysX default is tangential-only, so the cylinder rolls out of the grip.
        tpr = 0.04
        co.CreateTorsionalPatchRadiusAttr(tpr)
        co.CreateMinTorsionalPatchRadiusAttr(tpr)
        n_off += 1
        # Force-binding our own material with `strongerThanDescendants` -- tried, failed, off by
        # default.
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        if mat and mat.GetPrim().IsValid():                  # raise friction on the material it uses
            mp = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            mp.CreateStaticFrictionAttr(MU)                  # match MuJoCo finger friction="5 5 5"
            mp.CreateDynamicFrictionAttr(MU * 0.8)
            mp.CreateRestitutionAttr(0.0)
            PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr("max")
            n_compliant += int(make_compliant(mat.GetPrim()))
            touched.add(mat.GetPrim().GetName())
        else:                                                # no material -> bind ours
            UsdShade.MaterialBindingAPI.Apply(prim)
            UsdShade.MaterialBindingAPI(prim).Bind(grip_mat, materialPurpose="physics")
            touched.add("GripMaterial")
    print(f">>> tune_physics: TGS/240Hz, iters 64/16, contact-offset {n_off} colliders, "
          f"mu{MU}(max) on {sorted(touched)}, compliant-contact on {n_compliant} materials",
          flush=True)


def build_home_pose(names, current):
    """Per-joint hold pose = play.py home keyframe, arm forced to PARK."""
    pose = np.array(current, dtype=float)
    kf = json.load(open(HOME_KEYFRAME)) if os.path.exists(HOME_KEYFRAME) else {}
    for i, n in enumerate(names):
        if n in kf:
            pose[i] = kf[n]
        for sub, val in PARK_OVERRIDE.items():
            if sub in n:
                pose[i] = val
    return pose


def finger_pd(n):
    """Finger dynamics carried over from the source model: (kp, kd, armature).

    From the source actuator and joint definitions:
        joint_1        actuator kp=100 kv=20  +  joint stiffness=10 damping=20  -> kd 40
        joint_2/3      actuator kp=10  kv=6   +  joint stiffness=1  damping=10  -> kd 16
        palm_finger_*  actuator kp=10  kv=6   +  joint stiffness=1  damping=10  -> kd 16
    MuJoCo's kv and the joint's own damping both oppose absolute joint velocity, exactly as a
    PhysX drive's kd does against its zero target velocity, so they add.  The passive joint
    stiffness is deliberately NOT folded in: we command the RECORDED qpos, which already contains
    whatever equilibrium that spring produced.

    ARMATURE is the piece PhysX needs and the MJCF has no word for.  MuJoCo integrates damping
    IMPLICITLY at 500Hz (`integrator="implicitfast"`), so kd=40 on a 1e-5 kg.m2 phalanx is fine
    there.  At 240Hz a drive that stiff on that little inertia either rings (measured: j1
    overshooting its command by 0.72 rad, then throwing the object) or, once you cap the torque to
    stop the ringing, spends its whole budget on damping and sags to the joint limit (measured:
    j1 pinned at -1.5708 = the -90deg limit while commanded to -0.21).  Reflected rotor inertia
    fixes it and is PHYSICALLY REAL for this hand — a Robotiq 3F finger is worm-gear driven, and a
    gearbox multiplies motor inertia by the square of its ratio.  armature ~ kd*dt keeps the
    damping resolvable in one step.
    """
    if n.startswith("palm_finger") or "_joint_2_" in n or "_joint_3_" in n:
        kp, kd = 10.0, 16.0
    else:
        kp, kd = 100.0, 40.0
    g = 1.0
    marg = 1.5
    return kp * g, kd * g, kd * g * marg / 240.0


def hide_collision_visuals(stage):
    """Hide the source model's collision-proxy geometry.

    Visual only: PhysX collision is independent of USD visibility, so every collider still
    collides exactly as before.

    In the source MJCF these are debug shapes drawn in a toggleable geom group — the finger
    contact pads (red blocks that otherwise appear poking out of the gripper) and a chassis
    collision box (a green slab on the body). The importer has no notion of geom groups, so they
    arrive as ordinary visible meshes.

    They are identified by their authored display colour rather than by name, which survives
    renaming and catches every copy across both arms.
    """
    if os.environ.get("HIDE_COLLISION_VIS", "1") != "1":
        return
    n = 0
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        img = UsdGeom.Imageable(prim)
        if not img:
            continue
        c = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get() if prim.IsA(UsdGeom.Gprim) else None
        if not c:
            continue
        r, g, b = (float(v) for v in c[0])
        pure_red = r > 0.9 and g < 0.1 and b < 0.1
        pure_green = g > 0.9 and r < 0.1 and b < 0.1
        if pure_red or pure_green:
            img.MakeInvisible()
            n += 1
    print(f">>> hid {n} collision-proxy visuals (pad boxes + chassis collision box); "
          f"HIDE_COLLISION_VIS=0 to show them", flush=True)


def set_finger_armature(stage, arm="_1"):
    """Author physxJoint:armature on every arm-1 finger joint.  Must run at LOAD (before the
    articulation is built) — it is a solver property of the joint, not a runtime gain."""
    n = 0
    for prim in stage.Traverse():
        nm = prim.GetName()
        if not (nm.startswith(("finger_", "palm_finger")) and nm.endswith(arm)):
            continue
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        _, _, a = finger_pd(nm)
        PhysxSchema.PhysxJointAPI.Apply(prim).CreateArmatureAttr(a)
        n += 1
    print(f">>> finger armature: {n} joints (j1 {finger_pd('finger_a_joint_1_1')[2]:.4f}, "
          f"j2/3 {finger_pd('finger_a_joint_2_1')[2]:.4f} kg.m2)", flush=True)


def set_finger_contact_impulse(stage, arm="_1"):
    """Bound the impulse a FINGER PAD contact may deliver, per pad body.

    physxRigidBody:maxContactImpulse is authored on the OBJECT elsewhere, but PhysX applies it to
    every contact that body has — including the floor, which needs at least m*g/dt = 0.042 N.s to
    hold the cylinder up.  So the object-side value cannot be lowered to a grip-sized force without
    the object sinking through the floor.  PhysX takes the MINIMUM of the two bodies' values, so
    authoring it on the pads instead bounds pad contacts alone and leaves the floor untouched.

    Sizing: holding 1.025kg by friction with mu=4.0 needs only m*g/(2*mu) = 1.3N per side, while a
    300mm-tall r=80mm cylinder TIPS above m*g*r/h = 5.4N.  Measured pad loads were 10-96N, i.e. 2-18x
    the tipping limit — that is what walked the object 23-66mm in every trial, not a lack of
    friction (floor and object are already static=5.0 dynamic=4.0).  Default 0.02 N.s = 4.8N at
    240Hz: above the 1.3N needed, below the 5.4N that tips."""
    imp = 0.0
    if imp <= 0:
        return
    n = 0
    for prim in stage.Traverse():
        nm = prim.GetName()
        if not (nm.startswith(("finger_", "palm_finger")) and nm.endswith(arm)):
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateMaxContactImpulseAttr(imp)
        n += 1
    print(f">>> finger contact impulse: {n} pad bodies capped at {imp} N.s "
          f"({imp * 240:.1f}N at 240Hz)", flush=True)


def set_arm_gains(robot, names):
    """The MJCF importer skips drive gains on the arm Column/Arm actuators -> set PD
    gains (re-tuned from scratch on PhysX). Wheels/rollers stay free."""
    ndof = len(names)
    kp = np.full(ndof, 2.0e3)
    kd = np.full(ndof, 2.0e2)
    eff = np.full(ndof, 1.0e6)                               # arm DOFs are kinematically forced anyway
    for i, n in enumerate(names):
        if any(s in n for s in ("ColumnLeftBearingJoint", "ColumnRightBearingJoint", "ArmLeftJoint")):
            kp[i], kd[i] = 2.0e5, 2.0e4
        elif any(s in n for s in ("finger_", "palm_finger", "gripper_", "HandBearingJoint")):
            kp[i], kd[i] = 1.0e3, 1.0e2
            if n.startswith(("finger_", "palm_finger")):
                kp[i], kd[i], _ = finger_pd(n)
                # A finger phalanx has inertia ~1e-5 kg.m2, so at 240Hz a position drive is stable
                # only up to about 0.6 Nm/rad before damping.

                # Force-limited squeeze: the PD saturates at this torque, so the fingers press with
                # a bounded force instead of the default unlimited drive, which punches through
                # contacts.
                eff[i] = 1.0e6
        elif "rolling_joint" in n or "slipping" in n:
            kp[i], kd[i] = 0.0, 1.0
    ctrl = robot.get_articulation_controller()
    ctrl.set_gains(kp, kd)
    if os.environ.get("EFFORT_LIMIT", "1") == "1":
        try:
            ctrl.set_max_efforts(eff)
        except Exception as e:
            print(f">>> set_max_efforts unavailable: {e}", flush=True)


def reveal_viewport(world, simulation_app):
    """Re-enable viewport updates after the scene is fully built AND scattered AND parked, warming
    the renderer a few frames so it appears finished in one shot (not body-by-body, not mid-scatter)."""
    _vp = getattr(world, "_frozen_vp", None)
    if _vp is None:
        return
    try:
        _vp.updates_enabled = True
        for _ in range(30):
            simulation_app.update()
    except Exception:
        pass
    world._frozen_vp = None


def load_scene(simulation_app, name="morph"):
    """Full setup: reference the world USD, apply converter fixups + lighting + floor +
    textured products, add the robot, set gains, park it. Returns (world, robot, names, q0)."""
    if not os.path.exists(USD):
        raise SystemExit(f"USD not found: {USD}\nRun import_world.py first.")
    world = World(stage_units_in_meters=1.0)
    # Frame the camera the moment the world exists: before `World()` it is a silent no-op, because
    # there is no viewport yet. Bodies rendering black during the load are RTX warm-up, and
    # cosmetic.
    _vp = None
    if os.environ.get("ISAAC_HEADLESS") != "1":
        try:
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(eye=[7.5, -8.0, 4.5], target=[3.5, -4.0, 0.3])
        except Exception:
            pass
        # Do not show the scene assembling: bodies render black until their MDL materials compile.
        if os.environ.get("PRESENT_WHEN_LOADED", "1") == "1":
            try:
                from omni.kit.viewport.utility import get_active_viewport
                _vp = get_active_viewport()
                _vp.updates_enabled = False
            except Exception:
                _vp = None
    add_reference_to_stage(usd_path=USD, prim_path="/World")
    while is_stage_loading():
        simulation_app.update()

    fixup_scene(world.stage)
    add_lighting(world.stage)
    add_marble_floor(world.stage)
    swap_in_textured_products(world.stage)

    root = find_articulation_root(world.stage)
    if root is None:
        raise SystemExit("No ArticulationRootAPI prim found in imported USD.")
    print(f">>> articulation root: {root}", flush=True)
    apply_link_masses(world.stage, root)
    if os.environ.get("NO_LOOP", "0") != "1":               # NO_LOOP=1: skip the loop joint (for a fully-
        close_arm_loop(world.stage, root)                   # kinematic arm — the loop constraint fights forcing)
    else:
        print(">>> NO_LOOP: skipping close_arm_loop (kinematic arm mode)", flush=True)
    if os.environ.get("PAD_COLLIDER", "1") == "1":          # real-mesh finger pads (vs the thin under-covering boxes)
        fix_finger_collision(world.stage, root)
    if os.environ.get("OBJ_RADIUS"):                        # reshape objects to a graspable size (isaac-only, runtime)
        resize_object(world.stage, float(os.environ["OBJ_RADIUS"]),
                      float(os.environ["OBJ_HALF_H"]) if os.environ.get("OBJ_HALF_H") else None)
    tune_physics(world.stage, root)
    set_finger_armature(world.stage)          # reflected gearbox inertia (see finger_pd)
    set_finger_contact_impulse(world.stage)   # bound PAD force without throttling the floor
    hide_collision_visuals(world.stage)       # red pad boxes / green chassis box — visual only

    robot = world.scene.add(Robot(prim_path=root, name=name))
    world.reset()
    robot.initialize()
    names = list(robot.dof_names)
    set_arm_gains(robot, names)
    q0 = build_home_pose(names, robot.get_joint_positions())
    robot.set_joint_positions(q0)
    print(f">>> {len(names)} DOFs, robot parked", flush=True)
    # Do not reveal here: the object scatter and the arm park both happen in `Demo.__init__`, after
    # this returns, so revealing now shows the initial line-up and then the scatter and park
    # happening live.
    world._frozen_vp = _vp

    return world, robot, names, q0
