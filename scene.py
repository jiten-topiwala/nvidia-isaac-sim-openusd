"""Shared Isaac-Sim scene setup for the obotx market world.

Import this *after* the entry script has created `SimulationApp(...)` (the isaacsim
and pxr modules only resolve once the kit app is up). Both view_and_jog.py and
nav.py call `load_scene()` so the load + converter-artifact fixups live in one place.
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
# "friction" = objects are real dynamic bodies held by finger-pad contact friction (real-world
# physics); "pin" = the proven kinematic-puppet fallback.  Read at LOAD time — the two modes
# author different physics on the stage, so switching requires a restart.
GRIP_MODE = os.environ.get("GRIP_MODE", "friction")
USD = os.path.join(HERE, "usd/market_world_m1/market_world_m1.usda")
PRODUCTS_USD = os.path.join(HERE, "usd/products_textured.usd")
PRODUCTS_MANIFEST = os.path.join(HERE, "usd/_products.json")
HOME_KEYFRAME = os.path.join(HERE, "usd/_home_keyframe.json")
MARBLE = os.path.join(HERE, "usd/market_world_m1/Textures/white_marble_tile2.png")

# The parallel-linkage equality that holds the arm up in MuJoCo doesn't convert, so
# force the arm to the play.py PARK pose (else it sags into the chassis).
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
                if GRIP_MODE == "friction":
                    # FRICTION MODE: objects are REAL dynamic bodies (gravity + collision from
                    # load) held by finger-pad contact friction — the real-world-transferable
                    # physics the client asked for.  maxDepenetrationVelocity kept low so a squeeze
                    # never ejects the object (PhysX default 100 m/s does).
                    pb = PhysxSchema.PhysxRigidBodyAPI.Apply(child)
                    # DEPENETRATION KICK is what "throws" the object: the arm is kinematically
                    # forced, so any overlap — even 0.1mm from one creep step — is resolved by
                    # pushing the DYNAMIC body out at up to this speed.  At 3.0 m/s a single step
                    # of palm contact launched the object 100mm+ across the floor (measured, every
                    # approach variant).  A real gripper touching a can does not fire it away.
                    pb.CreateMaxDepenetrationVelocityAttr(
                        float(os.environ.get("DEPEN_VEL", "0.05")))
                    # hard ceiling on how fast a pickup object can ever travel — nothing in this
                    # scene legitimately moves faster, and it bounds every remaining solver artifact
                    pb.CreateMaxLinearVelocityAttr(float(os.environ.get("OBJ_VMAX", "1.0")))
                    # MAX CONTACT IMPULSE — the direct bound on what one contact may do in one
                    # step, and the honest model of a compliant rubber pad.  Unbounded (the
                    # default), a 3mm finger curl against a stiff arm answered with ~968N and threw
                    # the object; MuJoCo's whole recorded grasp peaks at 56N.  0.10 N.s at 240Hz
                    # caps a single contact at ~24N.
                    imp = float(os.environ.get("MAX_IMPULSE", "0.10"))
                    if imp > 0:
                        pb.CreateMaxContactImpulseAttr(imp)
                    # threshold 0 -> EVERY persistent contact reports each step (the touch sensor /
                    # pad-load telemetry); the default threshold hides resting low-force contacts
                    cr = PhysxSchema.PhysxContactReportAPI.Apply(child)
                    cr.CreateThresholdAttr(0.0)
                    n_kin += 1
                    continue
                # PIN MODE (fallback): gravity-free, COLLISION-FREE dynamic puppets — every phase
                # moves them by set_world_poses (tensor writes persist on dynamic bodies; a
                # KINEMATIC body snaps back to its USD-authored pose every step, which silently
                # killed the scatter).  Toggling colliders at runtime — even one — invalidates the
                # live articulation view in headed mode (view-killer family); authored here,
                # before the World/view exists, it is categorically safe.
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
                # where the prim isn't defined) — the palm<->object FixedJoints stayed alive for
                # the entire project.  SetActive(False) composes over the payload and sticks.
                child.SetActive(False)
                n_weld += 1
    left = [p.GetPath().pathString for p in stage.Traverse()
            if p.GetName().startswith("grasp_left") and p.IsActive()]
    if left:
        print(f">>> WARNING: {len(left)} grasp welds STILL ACTIVE: {left[:2]}", flush=True)
    mode_note = "REAL dynamic (friction grip)" if GRIP_MODE == "friction" else "gravity-free+collision-free"
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
            # floor finger-link mass: the real ~0.007kg links vs the 1kg object = ~20:1 ratio -> the light
            # link CAN'T absorb the contact impulse and flies to 5e7 on the FIRST touch. Floor to a stable
            # ratio. CRITICAL: floor the INERTIA consistently too — a heavy mass with the real ~1e-6 inertia
            # is a mass/inertia MISMATCH that the drive+contact spin to 1e14. I ~ m*L^2, finger link L~3cm.
            fm = float(os.environ.get("FINGER_MASS", "0.20"))
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

    # collide the FINGERTIPS (link_3) ONLY. Gap probe (grasp_replay): the object is 15cm wide, fingers
    # are ~10cm long, so the PROXIMAL (link_1) + MIDDLE (link_2) links sit INSIDE the fat object at the
    # open pose (-3 to -7cm overlap) -> fling. The fingertip pads (link_3) are clean (+0..+1.7cm) and
    # ARE the grip surface: 3 tip pads at mu5 easily hold 1kg. link_1/2 contact is incidental in MuJoCo
    # (soft contact tolerates it); PhysX rigid contact can't, so drop those colliders. TIP=all to keep all.
    # FRICTION mode: ALL finger-link colliders — the short (r=0.055) object sits BELOW the
    # fingertip curl arc; the MIDDLE-phalanx pads are what actually grip it (in MuJoCo too).
    # Tip-only was sized for the fat r=0.08 original where proximal links overlapped at open.
    tip_only = os.environ.get("TIP", "all" if GRIP_MODE == "friction" else "3") == "3"
    def is_tip(prim):
        p = prim.GetPath().pathString
        return (not tip_only) or "link_3_1" in p

    # pass A: de-instance the fingertip meshes so their Mesh children become editable
    n_deinst = 0
    # ...and the PROXIMAL ones too when we intend to switch them OFF. A prim that is still an
    # INSTANCE hides its Mesh children from `Usd.PrimRange(root)` (which does not descend into
    # instance proxies), so pass B literally cannot see them: measured, `0 proximal meshes off` was
    # reported while `finger hulls: a:3666v` proved the mesh colliders were still live -- the hull
    # builder finds them only because it uses `Usd.TraverseInstanceProxies()`. Without this, TIP=3
    # disables the proximal BOXES and leaves the proximal MESHES colliding, and b's link_1 still
    # sits 2.2mm off the object while b's PAD is 30.8mm away.
    _prox_off = tip_only and os.environ.get("PROX_MESH_OFF", "1") == "1"
    for prim in Usd.PrimRange(root):
        if (is_finger1(prim) and (is_tip(prim) or _prox_off)
                and prim.GetName().endswith("real_V2") and prim.IsInstance()):
            prim.SetInstanceable(False)
            n_deinst += 1

    # pass B: fresh traversal (sees the de-instanced children) — convex collider on each fingertip Mesh,
    # disable EVERY finger box (link_1/2/3).
    n_mesh = n_box = n_prox_off = 0
    for prim in Usd.PrimRange(root):
        if not is_finger1(prim):
            continue
        t = prim.GetTypeName()
        path = prim.GetPath().pathString
        if t == "Cube" and prim.HasAPI(UsdPhysics.CollisionAPI):
            # KEEP_BOXES=1 leaves them ON.  These Cubes ARE MuJoCo's 27 fingertip pad geoms — the
            # geoms that carry the entire grip in the recorded grasp (finger_[abc]_link_3_1,
            # 2-56N, contact from close frame 23).  With them off, the convex hull of the visual
            # shell is the only grip surface and it reaches 1.4-2.2mm SHORT: measured, the b/c
            # fingers hover 2mm off the cylinder for the whole close and never touch.
            # KEEP_BOXES=1 kept EVERY box, including link_1/link_2, which re-creates the exact
            # overlap this function was written to avoid: measured 08-17, b's link_1 sits -0.3mm
            # INSIDE the object while b's PAD is still 28.5mm away, so the advance can never bring
            # the pads to the object without the proximal link penetrating it (opening j1 to -1.55
            # moves it 0.3mm -- the link does not swing clear). Respect TIP here: with TIP=3 keep
            # only the link_3 pad boxes, so the proximal links pass and the PADS do the gripping,
            # which is what the comment above prescribes. TIP=all keeps the old keep-everything.
            if os.environ.get("KEEP_BOXES", "0") == "1" and is_tip(prim):
                continue
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)   # turn OFF the box collider
            n_box += 1
        elif (t == "Mesh" and "real_V2" in path and is_tip(prim)
              and os.environ.get("MESH_PADS", "1") == "1"):                   # the real fingertip mesh
            # MESH_PADS=0 + KEEP_BOXES=1 == MuJoCo's EXACT collision model (pad boxes only).
            # Running both at once double-covers each phalanx: two overlapping convex sets produce
            # duplicate contact constraints on the same patch, and PhysX answers 3mm of curl with
            # ~968N instead of ~55N (the finger drive's own limit at that lever).
            UsdPhysics.CollisionAPI.Apply(prim)
            # convexHull of the NON-convex finger mesh balloons + deep-overlaps; convexDecomposition
            # splits it into shape-following convex chunks (thinner, follows the pad).
            approx = os.environ.get("APPROX", "convexDecomposition")
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approx)
            n_mesh += 1
        elif (t == "Mesh" and "real_V2" in path and not is_tip(prim)
              and prim.HasAPI(UsdPhysics.CollisionAPI)
              and os.environ.get("PROX_MESH_OFF", "1") == "1"):
            # FREE THE PROXIMAL LINKS.  Disabling their pad BOXES is not enough: the finger meshes
            # carry CollisionAPI from the source asset (the hull census proves it -- a:3642v with
            # MESH_PADS=1 and a:3682v with MESH_PADS=0 + KEEP_BOXES=1, i.e. the 3642 mesh verts are
            # present either way and KEEP_BOXES only ADDS 5 cubes). So with TIP=3 the link_1/link_2
            # meshes still collide and still block: measured, b's link_1 sits -0.3mm INSIDE the
            # object while b's PAD is 28.5mm away. Turn them off so the proximal links pass and the
            # link_3 pads do the gripping -- which is what this function's own rationale prescribes.
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
        # DIAGNOSTIC: kill EVERY collider in the arm-1 hand subtree — if the descent STILL
        # nudges the object, the toucher was never the hand (wrist/boom/chassis/field effect)
        n_off2 = 0
        for prim in Usd.PrimRange(root):
            pth = prim.GetPath().pathString
            if "Arm_1" in pth and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                n_off2 += 1
        print(f">>> DIAG_NOHAND: {n_off2} arm-1 colliders OFF", flush=True)
    if GRIP_MODE == "friction":
        # the PALM BLOCK (Gripper_Link2/3) lip dips 21mm into the object cylinder at grasp depth
        # (measured) — it is NOT a grip surface (the finger pads are); MuJoCo's soft contact
        # tolerated the incidental lip press, PhysX rigid shoves the object away.  Exempt it.
        # The palm's colliders live under INSTANCED prototypes — PrimRange skips instance
        # proxies, so a plain traversal never saw them (PhysX contact events proved
        # Gripper_Link3 still hitting the object).  Pass A: find them THROUGH proxies and
        # de-instance their owning ancestor; pass B: author the disable on the now-real prims.
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
                # WRIST BEARING SPHERE: bearing structure between Link1 and the palm, never a grip
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
    # PALM COLLIDER APPROXIMATION — the mouth is only a mouth if the collider is concave.
    # Measured: `palm_real_V2` is a 96,391-vertex CONCAVE mesh carrying a **convexHull** collider,
    # extent 134 x 92 x 130mm. A convex hull of a concave palm FILLS THE CAVITY — the visual shows
    # an opening the physics does not have. That is why the "mouth depth" measures only 43mm, why
    # the palm bulldozes the object in every close that brings it near (243-376 Ns), and why the
    # object can never reach the pad line where a force-closure grasp lives.
    # The finger links already use convexDecomposition; the palm was the one part left on a hull,
    # and it is the one part whose whole function is having a cavity.
    n_papx = 0
    _papx = os.environ.get("PALM_APPROX", "convexDecomposition")
    if GRIP_MODE == "friction" and _papx:
        # COLLECT the ancestors first: de-instancing DURING the traversal expires the iterator
        # ("Iterator points to expired 'Mesh' instance proxy prim"), because making an ancestor
        # non-instanceable invalidates every proxy under it.
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
                cy.GetHeightAttr().Set(2.0 * half_h)         # TALLER objects: the pads then land
                #   mid-FLANK (horizontal normals = real clamp) instead of on the top rim
            h = cy.GetHeightAttr().Get() or 0.28
            cy.GetExtentAttr().Set([(-radius, -radius, -h / 2), (radius, radius, h / 2)])
            # AUTHOR THE MASS, or resizing silently makes the object a FEATHER.
            # Nothing sets a mass for the pickup objects, so PhysX derives one from the shape at a
            # very low effective density (~170 kg/m3 as shipped). Shrinking the cylinder then cuts
            # the mass with the VOLUME: measured, r=0.040 h=0.09 gives **0.154 kg**, six times
            # lighter than the stock object, and every size experiment in this log was quietly
            # testing a lighter object as well as a smaller one.
            # Why it decides the grasp: the object tips about its base edge at m*g*r/h, which at
            # 0.154kg and a 0.147m contact height is **0.41N** -- while the wrap drives the thumb at
            # ~14N. It goes over at 3% of the applied force, so no amount of approach or timing work
            # can hold it. Measured breakaway confirms it: a steady 2N push slides it 62mm.
            # Fix the density instead of the symptom. 800 kg/m3 is a filled plastic container, which
            # is what these objects represent (the stock 150x300mm object is ~1kg, i.e. the scene was
            # authored around roughly this mass and only the RESIZE broke it).
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
    """PhysX defaults are too loose for the finger<->object grasp: rigid-contact impulses
    spike and the light finger links go non-finite. Bump solver iterations (TGS), run finer
    substeps, and give the object+finger colliders a small contact offset so contact engages
    gradually. Also raises contact fidelity for later RL. All Isaac-side; no MuJoCo touch."""
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
    # 64/16 is the VALIDATED config for this KINEMATICALLY-FORCED linkage.  The published TGS
    # advice (velocity iterations 0) is for drive-based robots — with vel iters 0 the forced arm
    # sagged 8cm below its commanded pose with 5-11 rad/s joint oscillation (fingers through the
    # floor, object swept away).  Do NOT re-apply that guidance here.
    VEL_ITERS = 16
    ar = stage.GetPrimAtPath(robot_root)
    if ar.IsValid():
        pa = PhysxSchema.PhysxArticulationAPI.Apply(ar)
        pa.CreateSolverPositionIterationCountAttr(64)
        pa.CreateSolverVelocityIterationCountAttr(VEL_ITERS)
        # OFF: converted MJCF hands generate PHANTOM finger-vs-finger self-contacts (adjacent
        # links overlap by construction) — a documented articulation-explosion source. The
        # fingers only need to contact the OBJECT, not each other.
        pa.CreateEnabledSelfCollisionsAttr(False)
    # finger/gripper LINKS: cap velocity + depenetration so a swept object contact can't fling a
    # light link to non-finite (the ~1e6 garbage). This caps the LINK directly, not just the obj.
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
    # The MJCF->USD conversion clamped finger_b/c_joint_1 lower limit (0.0495 rad) ABOVE the
    # commanded open pose (-0.493 rad) -> those fingers stick nearly-closed and can't grip
    # (only the wide-range thumb works). Widen every finger joint_1 lower limit so the open
    # pose is reachable. -90 is safe whether the attr is stored in deg (-1.57rad) or rad.
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
            # hard velocity cap + damping: a bad swept-penetration contact can't launch the
            # object across the room (was hitting 500+ m/s); it stays put and settles into grip.
            rb.CreateMaxLinearVelocityAttr(1.5)
            rb.CreateMaxAngularVelocityAttr(8.0)
            # CAP THE DEPENETRATION VELOCITY -- the actual launcher.
            # Measured 08-17 with a per-step velocity-jump catcher: the object goes 0 -> 1224mm/s in
            # ONE step, and the impulse that step is `floor 9.19 Ns` against `palm 0.5 Ns`. The FLOOR
            # is throwing it, not the hand. The chain is: the position-controlled hand (effectively
            # infinite mass) presses the object slightly into the static floor, PhysX then resolves
            # that overlap by pushing the bodies apart, and with no cap that recovery is violent
            # enough to launch a 3kg body at over a metre per second. It then flies, lands deeper,
            # and does it again -- which is the "coasting", the "retreat" and the "sweep" that this
            # log has been chasing from six different directions.
            # This is a numerical recovery, not a physical force, so bounding it is the correct fix
            # rather than damping the symptom.
            rb.CreateMaxDepenetrationVelocityAttr(
                float(os.environ.get("MAX_DEPEN_VEL", "0.10")))
            # HIGH linear damping resists the early one-sided shove during close (the first finger to
            # touch pushes the free object out before the others engage -> 0.04N, no grip). Damped, it
            # barely moves while all 3 fingers close symmetrically, then the opposing grip + friction
            # holds it (net hand force ~0). Physical (viscous), not a pin. OBJ_DAMP env-tunable.
            rb.CreateLinearDampingAttr(float(os.environ.get("OBJ_DAMP", "0.5")))
            rb.CreateAngularDampingAttr(float(os.environ.get("OBJ_DAMP", "0.5")))
            cr = PhysxSchema.PhysxContactReportAPI.Apply(child)   # enable get_net_contact_forces
            cr.CreateThresholdAttr(0.0)
    # FRICTION — the MJCF import bound a mu=0.0 material (PhysicsMaterial_1) to the finger PADS, so
    # the grip was frictionless and every object slid straight out on lift. Match colliders by PATH
    # (finger collider prims are named "Box"/"Cylinder", NOT "finger*" — the old name check missed
    # them entirely), give a contact offset, and set high friction on WHATEVER material each collider
    # actually uses (bind a fresh one only if it has none).
    MU = float(os.environ.get("FRICTION", "5.0"))            # test knob (FRICTION=0.01 to check if it's applied)
    grip_mat = UsdShade.Material.Define(stage, "/World/Physics/GripMaterial")
    gp = UsdPhysics.MaterialAPI.Apply(grip_mat.GetPrim())
    gp.CreateStaticFrictionAttr(MU)                          # match MuJoCo finger geom friction="5 5 5"
    gp.CreateDynamicFrictionAttr(MU * 0.8)
    gp.CreateRestitutionAttr(0.0)
    # combine mode MAX: default 'average' halves the pad friction against a lower-friction
    # partner — with max, the pad's mu always wins the pairing (NVIDIA gripper guidance; the
    # analog of MuJoCo's priority=1 finger geoms winning over the object's weaker triplet)
    PhysxSchema.PhysxMaterialAPI.Apply(grip_mat.GetPrim()).CreateFrictionCombineModeAttr("max")
    touched = set()

    def make_compliant(mprim):
        """COMPLIANT CONTACT = MuJoCo's solref, ported.  MuJoCo's pads declare
        solref="0.005 1": contact behaves as a spring-damper with time constant 5ms and damping
        ratio 1, so force RAMPS with penetration over ~10 steps.  PhysX's default rigid contact
        instead resolves the whole overlap in one step, which is why a pad arriving at 40mm/s
        handed this 1.025kg cylinder metres-per-second (measured 2.2 m/s) and, once bounded, still
        TIPPED it — a 300mm cylinder on a 160mm base goes over at a few newtons.
        k = m/tau^2 = 1.025/0.005^2 = 4.1e4 N/m; c = 2*sqrt(k*m) for ratio 1."""
        api = PhysxSchema.PhysxMaterialAPI.Apply(mprim)
        k = float(os.environ.get("PAD_STIFF", "4.1e4"))
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
    # THE FLOOR WAS NEVER IN THE FRICTION LIST.  The loop below matches
    # finger/Gripper/palm/pickup_obj, so the GROUND keeps whatever the converter gave it, and the
    # object-vs-floor pair resolves near that value instead of the mu=5 the scene intends.
    # Measured with probe_floor_friction.py (a steady horizontal push on a resting object):
    #     2N -> 0.0mm    5N -> 0.0mm    10N -> 39mm SLID    14N -> 65mm SLID
    # i.e. an effective pair friction of ~0.7, not 5.0. Breakaway is ~10N and the wrap drives the
    # thumb at ~14N (1.25Nm over a 0.090m lever), so the FIRST finger to touch slides the object
    # out of the hand before the opposing two arrive. That is the 200-300mm escape behind every
    # one-sided close in this log, and it is why balancing the landing to 4.8mm still was not
    # enough -- any residual timing error is spent sliding.
    # A warehouse floor is not ice; mu here is the physical model, not a tuning knob.
    # The ground sits behind an instance proxy (a plain Traverse never sees it), so collect the
    # ancestors first and de-instance before authoring -- same rule as the palm collider above.
    n_floor = 0
    if GRIP_MODE == "friction" and os.environ.get("FLOOR_MU_ON", "1") == "1":
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
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not (prim.HasAPI(UsdPhysics.CollisionAPI) and
                any(s in path for s in ("finger", "Gripper", "palm", "pickup_obj"))):
            continue
        co = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        # CONTACT OFFSET is the grasp's shock absorber: PhysX starts generating (speculative)
        # contacts this far out, so the pad decelerates over a band instead of being discovered
        # already overlapping.  At 0.003 the first pad touch handed the object 2.2 m/s in a single
        # step (measured, whatever the drive effort) — NVIDIA's gripper guidance is ~0.02.
        co.CreateContactOffsetAttr(float(os.environ.get("CONTACT_OFF", "0.02")))
        # REST OFFSET = compliant PAD THICKNESS.  Contact offset only decides where PhysX starts
        # GENERATING contacts; force still needs compression past the rest offset, so at 0.0 a pad
        # sitting 2-3mm off the surface carries exactly zero load.  That is the friction blocker on
        # our 160mm cylinder: at the balanced alignment all three fingers reach their ARC MINIMUM
        # 2-3mm out (the pad faces enclose ~120mm; alignment recovered all but the last few mm of
        # the 6.5mm radius excess over the recording's 147mm object) and simply cannot close
        # further — curling past the minimum moves them AWAY.  Inflating the pad surface by a few mm
        # is the physical model of the rubber pad the real gripper has, and it lets all three
        # fingers load TOGETHER instead of one at a time (single contact = pure tipping moment).
        # Must stay well under CONTACT_OFF.
        # FRICTION DEFAULT 3mm.  Measured 2026-08-14: at 0.0 the thumb's only contact is
        # `finger_a_link_1_1`, the KNUCKLE — link_2 and link_3 never touch the object at all, in
        # every run at every radius. It was never gripping, which is why it read ~0N however the
        # force side was tuned. At 0.003 `finger_a_link_3_1` (the pad) appears for the first time.
        # 0.005 is too much: contact then registers ~10mm out and `b` misses entirely.
        # Pin mode keeps 0.0 — its grasp is a kinematic pin and does not need a pad model.
        _rest_def = "0.003" if os.environ.get("GRIP_MODE", "friction") == "friction" else "0.0"
        co.CreateRestOffsetAttr(float(os.environ.get("REST_OFF", _rest_def)))
        # TORSIONAL friction — the obotx MuJoCo gripper.xml grips with condim="6" (tangential + TORSIONAL
        # + ROLLING friction). PhysX default is tangential-only -> the cylinder ROLLS out of the grip (the
        # "perpendicular squirt" over 33 iters). A non-zero torsionalPatchRadius enables torsional friction
        # so the object can't twist/roll free. TORSION env (default 0.04m patch).
        tpr = float(os.environ.get("TORSION", "0.04"))
        co.CreateTorsionalPatchRadiusAttr(tpr)
        co.CreateMinTorsionalPatchRadiusAttr(tpr)
        n_off += 1
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
          f"mu5.0(max) on {sorted(touched)}, compliant-contact on {n_compliant} materials",
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
    """MuJoCo's OWN finger dynamics, ported — (kp, kd, armature).

    src/env/robot/assets/gripper_actuator_V2.xml + robot/obotx_V2_OBJs.xml (READ-ONLY refs):
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
    g = float(os.environ.get("FINGER_GAIN_SCALE", "1.0"))
    marg = float(os.environ.get("ARMATURE_MARGIN", "1.5"))
    return kp * g, kd * g, kd * g * marg / 240.0


def hide_collision_visuals(stage):
    """Make MuJoCo's COLLISION-PROXY geometry invisible.  Visual only — PhysX collision is
    independent of USD visibility, so every collider still collides exactly as before.

    In the MJCF these are debug shapes MuJoCo draws in a toggleable geom group:
      * `pad_box1` / `pad_box2` on the fingers, `rgba="1 0 0 0.4"` — the 27 contact pads (the red
        blocks the client can see poking out of the gripper)
      * a chassis collision box, `rgba="0 1 0 1" class="collision"` (the green slab on the body)
    The importer has no notion of geom groups, so they came through as ordinary visible meshes.
    Identify them by their authored display colour rather than by name, which survives renaming and
    catches every copy across both arms."""
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
    imp = float(os.environ.get("FINGER_IMPULSE", "0.0"))
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
                # A finger phalanx has inertia ~1e-5 kg.m2.  At 240Hz a position drive is only
                # stable up to kp ~ I/dt^2 ~ 0.6 Nm/rad before damping; 1e3-6e3 is four orders past
                # that and the drive RINGS on contact — measured j1 overshooting its command by
                # 0.72 rad (commanded -0.45, actual +0.27), snapping back open, and throwing the
                # object.  MuJoCo's own finger actuator runs kp=100 (j1) / 10 (j2,j3) with joint
                # damping 10-20, and its whole grasp peaks at 56 N.  FINGER_KP matches that scale.

                # FORCE-LIMITED squeeze (friction grip): PD saturates at this torque, so the
                # fingers press the object with a bounded, realistic force instead of the PhysX
                # default unlimited drive (which punches through contacts).  ~2 Nm at the ~5cm
                # phalanx lever = ~40 N/finger, ~3x MuJoCo's 12 N force-stop target.
                # 8.0 could not track even FREE motion: measured 0.09-0.42 rad of lag closing
                # through empty air, which read as "the finger jammed on something" and froze all
                # three halfway with the pads 90mm short.  At 25 the lag is 0.000 and a stall means
                # a real obstruction.  Grip force stays bounded by this limit, and the contact
                # stop — not the drive weakness — is what halts a finger on the object.
                # UNLIMITED, like MuJoCo (its <position> actuators declare no forcerange).
                # Capping the drive to tame contact was the wrong layer: it starved the damping
                # and the fingers sagged to their limits.  physxRigidBody:maxContactImpulse on the
                # OBJECT bounds the contact instead — that is where MuJoCo bounds it too
                # (solref 1e-4 / impratio 100 / noslip_iterations 3).
                eff[i] = float(os.environ.get("FINGER_EFFORT", "1.0e6"))
        elif "rolling_joint" in n or "slipping" in n:
            kp[i], kd[i] = 0.0, 1.0
    ctrl = robot.get_articulation_controller()
    ctrl.set_gains(kp, kd)
    if os.environ.get("EFFORT_LIMIT", "1") == "1":
        try:
            ctrl.set_max_efforts(eff)
        except Exception as e:
            print(f">>> set_max_efforts unavailable: {e}", flush=True)


def load_scene(simulation_app, name="obotx"):
    """Full setup: reference the world USD, apply converter fixups + lighting + floor +
    textured products, add the robot, set gains, park it. Returns (world, robot, names, q0)."""
    if not os.path.exists(USD):
        raise SystemExit(f"USD not found: {USD}\nRun import_world.py first.")
    world = World(stage_units_in_meters=1.0)
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
    return world, robot, names, q0
