"""Shared Isaac Sim scene setup; import only after the entry script created `SimulationApp(...)`,
since isaacsim and pxr resolve once the Kit app is up. `load_scene()` is the single entry point."""
import json
import math
import os

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

HERE = os.path.dirname(__file__)
USD = os.path.join(HERE, "usd/market_world_m1/market_world_m1.usda")
PRODUCTS_USD = os.path.join(HERE, "usd/products_textured.usd")
PRODUCTS_MANIFEST = os.path.join(HERE, "usd/_products.json")
HOME_KEYFRAME = os.path.join(HERE, "usd/_home_keyframe.json")
MARBLE = os.path.join(HERE, "usd/market_world_m1/Textures/white_marble_tile2.png")

# The parallel-linkage equality doesn't convert -> force the arm to play.py's PARK pose (else it sags).
PARK_OVERRIDE = {
    "ColumnLeftBearingJoint": 1.2,
    "ColumnRightBearingJoint": 1.2,
    "ArmLeftJoint": 0.1,
}


def fixup_scene(stage) -> None:
    """Undo two converter artifacts: jointless env bodies import as dynamic and fall through the
    floor -> make them static; the `grasp_left*` welds import active -> deactivate them."""
    geo = stage.GetPrimAtPath("/World/Geometry")
    n_static = 0
    n_kin = 0
    if geo.IsValid():
        for child in geo.GetChildren():
            name = child.GetName()
            if name == "robot":
                continue
            if name.startswith("pickup_obj"):
                pb = PhysxSchema.PhysxRigidBodyAPI.Apply(child)
                # Max contact impulse: bound on what one contact may do in a step. PhysX takes the MIN
                # of a pair's caps, so this frees the FLOOR while the pads stay pinned by their own.
                imp = float(os.environ.get("OBJ_IMPULSE", "0.10"))   # <= 0: no cap
                if imp > 0:
                    pb.CreateMaxContactImpulseAttr(imp)
                # threshold 0 -> every persistent contact reports each step (the default hides resting
                # low-force ones, i.e. the pad-load telemetry)
                cr = PhysxSchema.PhysxContactReportAPI.Apply(child)
                cr.CreateThresholdAttr(0.0)
                n_kin += 1
                continue
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(child).CreateRigidBodyEnabledAttr(False)
                for gc in list(child.GetChildren()):
                    if "Joint" in gc.GetTypeName():
                        # redundant world-fixed joint (in a payload)
                        gc.SetActive(False)
                n_static += 1
    n_weld = 0
    phys = stage.GetPrimAtPath("/World/Physics")
    if phys.IsValid():
        for child in list(phys.GetChildren()):
            if child.GetName().startswith("grasp_left"):
                # RemovePrim on a payload-composed prim FAILS SILENTLY (it edits the root layer, where
                # the prim isn't defined) -- deactivate instead.
                child.SetActive(False)
                n_weld += 1
    left = [p.GetPath().pathString for p in stage.Traverse()
            if p.GetName().startswith("grasp_left") and p.IsActive()]
    if left:
        print(f">>> WARNING: {len(left)} grasp welds STILL ACTIVE: {left[:2]}", flush=True)
    mode_note = "REAL dynamic (friction grip)"
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
    """Author an 8x8 quad with the marble tiled 4x4 and hide the converter's stretched floor, whose
    collision plane stays and whose material sits behind an un-patchable instance proxy."""
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
    """Reference the textured products and hide the converter's flat-colored originals."""
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
    """Author real mass/inertia from the MuJoCo model on the robot links, with a small stable default
    for massless frames (PhysX turns zero/negative-mass bodies non-finite on any contact impulse).
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
            # Floor the finger-link mass: a real ~0.007kg link goes non-finite on first touch.
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
            # (w, x, y, z)
            mapi.CreatePrincipalAxesAttr(Gf.Quatf(q[0], q[1], q[2], q[3]))
            mapi.CreateCenterOfMassAttr(Gf.Vec3f(*b["ipos"]))
            n_real += 1
        else:                                    # massless MuJoCo frame -> small stable default
            mapi.CreateMassAttr(0.1)
            mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-3, 1e-3, 1e-3))
            mapi.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
            n_default += 1
    print(f">>> apply_link_masses: {n_real} real + {n_default} massless-defaulted links", flush=True)


def close_arm_loop(stage, robot_root):
    """Re-create the parallel-linkage ball constraint the MJCF->USD import dropped: a spherical joint
    between Contact_Cylinder_1_1/1_2, whose origins coincide at the assembled pose.
    `excludeFromArticulation=True` makes it a maximal-coordinate loop constraint instead of an
    articulation edge; the passive linkage drives must then be slack (kp=0) or they fight it."""
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
    # anchor at each body origin (they coincide)
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    j.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    j.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    j.CreateExcludeFromArticulationAttr().Set(True)
    j.CreateCollisionEnabledAttr().Set(False)
    print(f">>> close_arm_loop: spherical loop joint {a.name}<->{b.name} (excludeFromArticulation)", flush=True)


def fix_finger_collision(stage, robot_root):
    """Give the arm-1 fingertips a collider off the REAL finger mesh instead of the thin boxes that
    under-cover it. A collider API cannot be authored through an instance proxy, so de-instancing
    the meshes must be its own first pass."""
    root = stage.GetPrimAtPath(robot_root)

    # arm-1 finger subtree only
    def is_finger1(prim):
        p = prim.GetPath().pathString
        return "finger_" in p and "Arm_1" in p

    # pass A: de-instance the meshes about to be edited so their Mesh children become editable
    n_deinst = 0
    for prim in Usd.PrimRange(root):
        if is_finger1(prim) and prim.GetName().endswith("real_V2") and prim.IsInstance():
            prim.SetInstanceable(False)
            n_deinst += 1

    # pass B: fresh traversal (sees the de-instanced children); boxes off, mesh colliders on
    n_mesh = n_box = 0
    for prim in Usd.PrimRange(root):
        if not is_finger1(prim):
            continue
        t = prim.GetTypeName()
        path = prim.GetPath().pathString
        if t == "Cube" and prim.HasAPI(UsdPhysics.CollisionAPI):
            # KEEP_BOXES=1 with MESH_PADS=0 reproduces the source collision model: pad boxes only.
            if os.environ.get("KEEP_BOXES", "0") == "1":
                continue
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            n_box += 1
        elif t == "Mesh" and "real_V2" in path and os.environ.get("MESH_PADS", "1") == "1":
            UsdPhysics.CollisionAPI.Apply(prim)
            # convexHull of the NON-convex finger balloons and deep-overlaps; decomposition follows it
            approx = os.environ.get("APPROX", "convexDecomposition")
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approx)
            n_mesh += 1
    n_palm = 0
    # WHEEL_HUB_COLLIDE=0: floor contact goes through the ROLLERS, which is what lets the mecanum
    # base translate and yaw. Load-time only -- never touch USD physics while the view is live.
    if os.environ.get("WHEEL_HUB_COLLIDE", "1") == "0":
        n_hub = 0
        for prim in Usd.PrimRange(root):
            pth = prim.GetPath().pathString
            if ("wheel" in pth and "roller" not in pth and "slipping" not in pth
                    and prim.HasAPI(UsdPhysics.CollisionAPI)):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                n_hub += 1
        print(f">>> WHEEL_HUB_COLLIDE=0: {n_hub} wheel HUB colliders OFF (rollers carry the base)",
              flush=True)
    # Structure that plows the object at grasp depth without being a grip surface.
    def _palm_hit(pth):
        if (("Hand_Bearing_1" in pth or "Gripper_Link1_1" in pth)
                and "Gripper_Link2_1" not in pth):
            # wrist bodies
            return True
        if "link_0_1" in pth and "link_1_1" not in pth:
            # finger knuckle mounts
            return True
        if "Gripper_Link2_1/Sphere" in pth:
            # wrist bearing sphere
            return True
        # PALM (Gripper_Link2/3) stays COLLIDABLE: it is the cage's BACKSTOP for the thumb.
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
    # `palm_real_V2` is concave: a convexHull collider FILLS THE MOUTH CAVITY the visual shows.
    n_papx = 0
    _papx = os.environ.get("PALM_APPROX", "convexDecomposition")
    if _papx:
        # COLLECT the ancestors first: de-instancing DURING the traversal expires the iterator,
        # because making an ancestor non-instanceable invalidates every proxy under it.
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
          f"{n_box} boxes disabled, {n_palm} palm-block colliders off", flush=True)


def resize_object(stage, radius, half_h=None):
    """Set radius (and optionally height) on every pickup Cylinder, visual and collision, to a size
    the gripper grasps stably (~11cm, r=0.055). Must run before world.reset so PhysX cooks it."""
    n = 0
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if "pickup_obj" in p and prim.GetTypeName() == "Cylinder":
            cy = UsdGeom.Cylinder(prim)
            old = cy.GetRadiusAttr().Get()
            cy.GetRadiusAttr().Set(radius)
            # extent must track the size or the bbox/broadphase uses the old shape
            if half_h is not None:
                # taller objects land the pads mid-FLANK (horizontal normals = real clamp)
                cy.GetHeightAttr().Set(2.0 * half_h)
            h = cy.GetHeightAttr().Get() or 0.28
            cy.GetExtentAttr().Set([(-radius, -radius, -h / 2), (radius, radius, h / 2)])
            # Author the mass or resizing makes the object a feather: PhysX derives it from the shape.
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
    """Tighten the solver for the finger-object grasp: TGS, 240Hz substeps, raised iteration counts,
    and a contact offset on the object and finger colliders so contact engages gradually."""
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            sc = PhysxSchema.PhysxSceneAPI.Apply(prim)
            sc.CreateSolverTypeAttr("TGS")
            # 240Hz substeps
            sc.CreateTimeStepsPerSecondAttr(240)
            # resolve dynamic contacts AFTER the articulation: less penetration, stabler grasp forces
            prim.CreateAttribute("physxScene:solveArticulationContactLast",
                                 Sdf.ValueTypeNames.Bool).Set(True)
            break
    # 64/16 is PINNED for this KINEMATICALLY-FORCED linkage: the usual TGS advice of zero velocity
    # iterations is written for drive-based robots, and here the forced arm sags and oscillates.
    VEL_ITERS = int(os.environ.get("VEL_ITERS", "16"))
    ar = stage.GetPrimAtPath(robot_root)
    if ar.IsValid():
        pa = PhysxSchema.PhysxArticulationAPI.Apply(ar)
        pa.CreateSolverPositionIterationCountAttr(64)
        pa.CreateSolverVelocityIterationCountAttr(VEL_ITERS)
        # Self-collision is opt-in: converted MJCF hands generate phantom finger-vs-finger contacts.
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
    # finger/gripper LINKS: cap velocity + depenetration on the LINK, not just the object
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
    # The conversion clamped the b/c finger joint_1 lower limit ABOVE the commanded open pose.
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
            # velocity cap: a bad swept-penetration contact can't launch the object across the room
            rb.CreateMaxLinearVelocityAttr(1.5)
            rb.CreateMaxAngularVelocityAttr(8.0)
            # Depenetration launches the object: the hand is effectively infinite mass against the floor
            rb.CreateMaxDepenetrationVelocityAttr(
                float(os.environ.get("DEPEN_VEL", "0.10")))
            # damping resists the one-sided shove from the first finger to touch during the close
            rb.CreateLinearDampingAttr(float(os.environ.get("OBJ_DAMP", "0.5")))
            rb.CreateAngularDampingAttr(float(os.environ.get("OBJ_DAMP", "0.5")))
            cr = PhysxSchema.PhysxContactReportAPI.Apply(child)   # enable get_net_contact_forces
            cr.CreateThresholdAttr(0.0)
    # The import bound a mu=0 material to the finger PADS -- frictionless grip, objects slid out.
    MU = 5.0                                                 # pad/object mu
    grip_mat = UsdShade.Material.Define(stage, "/World/Physics/GripMaterial")
    gp = UsdPhysics.MaterialAPI.Apply(grip_mat.GetPrim())
    # match MuJoCo finger geom friction="5 5 5"
    gp.CreateStaticFrictionAttr(MU)
    gp.CreateDynamicFrictionAttr(MU * 0.8)
    gp.CreateRestitutionAttr(0.0)
    # Combine mode max, so the pad's mu wins the pairing instead of being averaged down. NOTE that
    # morph/world.py rebinds its own material on every grasp shape, which sets no combine mode.
    PhysxSchema.PhysxMaterialAPI.Apply(grip_mat.GetPrim()).CreateFrictionCombineModeAttr("max")
    touched = set()

    def make_compliant(mprim):
        """Port MuJoCo's solref compliant contact onto a material: a 5ms spring-damper, so force RAMPS
        with penetration instead of PhysX resolving the whole overlap in one step (k = m/tau^2)."""
        api = PhysxSchema.PhysxMaterialAPI.Apply(mprim)
        k = float(os.environ.get("PAD_STIFFNESS", "4.1e4"))
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
    # The floor needs friction too: the loop below only matches the hand and the objects.
    n_floor = 0
    _fmu = MU                                            # floor tracks the pad coefficient
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
    # De-instance the grasp colliders first or mu lands on nothing: a plain Traverse cannot see the
    # colliders behind instance proxies, and a material cannot be authored on a proxy anyway.
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
        # PhysX only raises contact events for actors carrying PhysxContactReportAPI, so with it on
        # the objects alone a finger jammed against the floor or the palm raises nothing.
        _bp = prim
        while _bp and _bp.IsValid() and not _bp.HasAPI(UsdPhysics.RigidBodyAPI):
            _bp = _bp.GetParent()
        if _bp and _bp.IsValid() and _bp.GetPath() not in _reported:
            try:
                PhysxSchema.PhysxContactReportAPI.Apply(_bp).CreateThresholdAttr(0.0)
                _reported.add(_bp.GetPath())
            except Exception:
                pass
        # Contact offset 8mm: contacts start being generated this far out, so a pad decelerates over
        # a band instead of being discovered already overlapping.
        co.CreateContactOffsetAttr(float(os.environ.get(
            "CONTACT_OFF", "0.008")))
        # Rest offset 3mm models compliant PAD THICKNESS: force needs compression past it, so at 0.0
        # a pad sitting a couple of millimetres off the surface carries exactly zero load.
        co.CreateRestOffsetAttr(float(os.environ.get("REST_OFF", "0.003")))
        # Torsional friction: PhysX defaults to tangential-only and the cylinder rolls out of the grip.
        tpr = 0.04
        co.CreateTorsionalPatchRadiusAttr(tpr)
        co.CreateMinTorsionalPatchRadiusAttr(tpr)
        n_off += 1
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        # raise friction on the material it uses
        if mat and mat.GetPrim().IsValid():
            mp = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            # match MuJoCo finger friction="5 5 5"
            mp.CreateStaticFrictionAttr(MU)
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
    """Finger drive gains from the source model: (kp, kd, armature). kd sums the actuator kv and the
    joint's own damping, which both oppose joint velocity. ARMATURE is reflected gearbox inertia: at
    240Hz these kd values on a ~1e-5 kg.m2 phalanx ring or sag to the limit without it."""
    # VENDOR_GAINS=1: Tesollo's published Isaac drive gains -- orders of magnitude softer than the
    # MuJoCo-derived numbers below, so contact saturates the drives into torque control.
    if os.environ.get("VENDOR_GAINS", "0") == "1":
        if "_joint_1_" in n:
            kp = 1.9
        elif "_joint_2_" in n:
            kp = 0.82
        elif "_joint_3_" in n:
            kp = 0.53
        else:                                   # palm_finger spread joints
            kp = 0.23
        # The vendor ships kd ~0 because the real hand's damping is mechanical (worm gear), which the
        # sim does not model: vendor stiffness, our own gearbox inertia and damping.
        _I = 0.25 if "_joint_1_" in n else 0.10
        kd = 2.0 * (kp * _I) ** 0.5              # critically damped against the reflected inertia
        return kp, kd, _I
    if n.startswith("palm_finger") or "_joint_2_" in n or "_joint_3_" in n:
        kp, kd = 10.0, 16.0
    else:
        kp, kd = 100.0, 40.0
    g = 1.0
    marg = 1.5
    return kp * g, kd * g, kd * g * marg / 240.0


_SHOW_COLLIDERS = os.environ.get("SHOW_COLLIDERS") == "1"


def fit_battery_collider(stage):
    """Grow the battery's collider by `BATTERY_FIT` mm on every face (default 5, the figure checked
    in the GUI; `BATTERY_FIT=0` keeps the authored box). NOT to the mesh bbox: the terminals make
    a bounds comparison read 26mm where the surfaces are short by ~5 (see FINDINGS)."""
    _mm = os.environ.get("BATTERY_FIT", "5")
    if not _mm or float(_mm) == 0.0:
        print(">>> BATTERY_FIT=0: battery collider left as authored (5mm under the drawn body)",
              flush=True)
        return
    try:
        grow = float(_mm) / 1000.0
    except ValueError:
        print(f">>> BATTERY_FIT: not a number ({_mm!r}) -- expected millimetres", flush=True)
        return
    # today's authored collider, base frame; grown uniformly rather than fitted to the mesh bbox
    lo = np.array([-0.310, -0.290, 0.165]) - grow
    hi = np.array([-0.030, 0.270, 0.305]) + grow
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        p = str(prim.GetPath())
        if not p.endswith("/LifePo4_12V50Ah/Box") or not prim.IsA(UsdGeom.Cube):
            continue
        par = UsdGeom.Xformable(prim.GetParent()).GetLocalTransformation()
        off = np.array([par.ExtractTranslation()[i] for i in range(3)])
        half, ctr = (hi - lo) * 0.5, (lo + hi) * 0.5 - off
        _old = {o.GetOpName(): o for o in UsdGeom.Xformable(prim).GetOrderedXformOps()}
        for _nm, _v in (("xformOp:scale", half), ("xformOp:translate", ctr)):
            if _nm in _old:
                _old[_nm].Set(Gf.Vec3f(*[float(x) for x in _v]) if "scale" in _nm
                              else Gf.Vec3d(*[float(x) for x in _v]))
        print(f">>> BATTERY_FIT: collider grown {grow * 1000:.1f}mm per face -> base frame "
              f"{np.round(lo, 4).tolist()}..{np.round(hi, 4).tolist()} "
              f"(height 140mm -> {(hi[2] - lo[2]) * 1000:.0f}mm); the boom's 19mm retreat clearance "
              f"becomes {19.0 - grow * 1000:.1f}mm", flush=True)
        return
    print(">>> BATTERY_FIT: the battery collider prim was not found", flush=True)


def hide_collision_visuals(stage):
    """Hide the source model's collision-proxy geometry (red pad boxes, green chassis box). Visual
    only: PhysX collision is independent of USD visibility. Matched by authored display colour, not
    name, because the importer has no notion of the MJCF geom groups these were drawn in."""
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
            if _SHOW_COLLIDERS:
                # The check the planner cannot do for itself: these ARE the boxes it reasons about. Authored
                # `purpose = "guide"`, which the viewport does not draw, so force it to default.
                img.MakeVisible()
                img.GetPurposeAttr().Set(UsdGeom.Tokens.default_)
                try:
                    UsdGeom.Gprim(prim).GetDisplayOpacityAttr().Set([0.35])
                except Exception:                                     # noqa: BLE001 -- display only
                    pass
                n += 1
                continue
            img.MakeInvisible()
            n += 1
    print(f">>> {'SHOWING' if _SHOW_COLLIDERS else 'hid'} {n} collision-proxy visuals "
          f"(pad boxes + chassis collision box)", flush=True)


def set_finger_armature(stage, arm="_1"):
    """Author physxJoint:armature on every arm-1 finger joint. Must run at LOAD, before the
    articulation is built: it is a solver property of the joint, not a runtime gain."""
    n = 0
    for prim in stage.Traverse():
        nm = prim.GetName()
        if not (nm.startswith(("finger_", "palm_finger")) and nm.endswith(arm)):
            continue
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        _, _, a = finger_pd(nm)
        _jp = PhysxSchema.PhysxJointAPI.Apply(prim)
        _jp.CreateArmatureAttr(a)
        # Joint friction expresses the worm gear's non-backdrivability (the mechanism holds position,
        # which is why the vendor can ship kp ~1.9).
        _jf = float(os.environ.get("FINGER_JOINT_FRICTION", "0"))
        if _jf > 0:
            try:
                _jp.CreateJointFrictionAttr(_jf)
            except Exception:
                pass
        n += 1
    print(f">>> finger armature: {n} joints (j1 {finger_pd('finger_a_joint_1_1')[2]:.4f}, "
          f"j2/3 {finger_pd('finger_a_joint_2_1')[2]:.4f} kg.m2)", flush=True)


def set_finger_contact_impulse(stage, arm="_1"):
    """Bound the impulse a FINGER PAD contact may deliver, per pad body.

    The same cap on the OBJECT applies to all its contacts, including the floor, which needs m*g/dt
    to hold it up; PhysX takes the MINIMUM of a pair's caps, so authoring it on the pads bounds pad
    contacts alone. Size above the friction need m*g/(2*mu) per side, below tipping m*g*r/h."""
    imp = float(os.environ.get("FINGER_IMPULSE", "0.0"))
    if imp <= 0:
        return
    # The PALM matches neither prefix but carries most of the force; PALM_IMPULSE overrides it alone.
    palm_imp = float(os.environ.get("PALM_IMPULSE", imp))
    n = n_palm = 0
    for prim in stage.Traverse():
        nm = prim.GetName()
        is_palm = nm == "Gripper_Link3" + arm
        if not (is_palm or (nm.startswith(("finger_", "palm_finger")) and nm.endswith(arm))):
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateMaxContactImpulseAttr(
            palm_imp if is_palm else imp)
        n_palm += is_palm
        n += 1
    print(f">>> grasp contact impulse: {n} bodies capped at {imp} N.s "
          f"({imp * 240:.1f}N at 240Hz)"
          + (f", palm at {palm_imp} N.s ({palm_imp * 240:.1f}N)" if n_palm else
             "  -- PALM NOT FOUND, it stays uncapped"), flush=True)


def set_arm_gains(robot, names):
    """Set the PD gains the MJCF importer skips on the arm Column/Arm actuators; wheels stay free."""
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
                # squeeze torque ceiling: the PD saturates here rather than punching through contacts
                eff[i] = 1.0e6
        elif "rolling_joint" in n or "slipping" in n:
            # Mecanum rollers and unbraked wheel hubs. kd comes from the implicit-drive stability bound, not
            # from feel. `RobotMixin._wheel_gain` reads the same variable with the same default.
            kp[i], kd[i] = 0.0, float(os.environ.get("ROLLER_KD", "0.02"))
    ctrl = robot.get_articulation_controller()
    ctrl.set_gains(kp, kd)
    try:
        ctrl.set_max_efforts(eff)
    except Exception as e:
        print(f">>> set_max_efforts unavailable: {e}", flush=True)


def reveal_viewport(world, simulation_app):
    """Re-enable viewport updates and warm the renderer so the scene appears finished in one shot."""
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
    """Reference the world USD, apply fixups, lighting, floor and products, add the robot, set gains
    and park it. Returns (world, robot, names, q0)."""
    if not os.path.exists(USD):
        raise SystemExit(f"USD not found: {USD}\nRun import_world.py first.")
    # physics_dt == rendering_dt, so a HEADED step is ONE physics step. At the default rendering_dt
    # a rendered call advances four substeps and every per-step law runs fast.
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 240, rendering_dt=1.0 / 240)
    # PhysX's stabilization pass suppresses jitter in RESTING contacts, and a held grasp is one.
    try:
        world.get_physics_context().enable_stablization(True)
        print(">>> physx: stabilization ENABLED (resting-contact jitter suppression)",
              flush=True)
    except Exception as _e_stab:
        print(f">>> physx: stabilization unavailable ({_e_stab})", flush=True)
    # Frame the camera the moment the world exists: before `World()` it is a silent no-op, because
    # there is no viewport yet.
    _vp = None
    if os.environ.get("ISAAC_HEADLESS") != "1":
        try:
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(eye=[7.5, -8.0, 4.5], target=[3.5, -4.0, 0.3])
        except Exception:
            pass
        # Do not show the scene assembling: bodies render black until their MDL materials compile.
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
    # NO_LOOP=1: kinematic arm; the loop
    if os.environ.get("NO_LOOP", "0") != "1":
        # constraint fights joint forcing
        close_arm_loop(world.stage, root)
    else:
        print(">>> NO_LOOP: skipping close_arm_loop (kinematic arm mode)", flush=True)
    # real-mesh finger pads (vs the thin under-covering boxes)
    fix_finger_collision(world.stage, root)
    # reshape objects to a graspable size (isaac-only, runtime)
    if os.environ.get("OBJ_RADIUS"):
        resize_object(world.stage, float(os.environ["OBJ_RADIUS"]),
                      float(os.environ["OBJ_HALF_H"]) if os.environ.get("OBJ_HALF_H") else None)
    tune_physics(world.stage, root)
    # reflected gearbox inertia (see finger_pd)
    set_finger_armature(world.stage)
    # bound PAD force without throttling the floor
    set_finger_contact_impulse(world.stage)
    # BATTERY_FIT=1: collider -> the drawn mesh
    fit_battery_collider(world.stage)
    # red pad boxes / green chassis box — visual only
    hide_collision_visuals(world.stage)

    robot = world.scene.add(Robot(prim_path=root, name=name))
    world.reset()
    robot.initialize()
    names = list(robot.dof_names)
    set_arm_gains(robot, names)
    q0 = build_home_pose(names, robot.get_joint_positions())
    robot.set_joint_positions(q0)
    print(f">>> {len(names)} DOFs, robot parked", flush=True)
    # Reveal later, not here: the scatter and the arm park happen in `Demo.__init__` after this
    # returns, and would otherwise be shown happening live.
    world._frozen_vp = _vp

    return world, robot, names, q0
