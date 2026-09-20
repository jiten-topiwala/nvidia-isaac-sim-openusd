"""What the USD stage says the arm must avoid: every collider under the roots, as boxes."""
import os

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from morph.usd_utils import find_path

# Mast sub-meshes that are convex hulls of a plate rather than solid volume -- see
# `ArmModel.SELFCOL_HULL_PARTS`, measured off the live stage. Same model on both arms.
HULL_PARTS = frozenset(("VerticalArm_0", "VerticalArm_1"))
MAX_SPAN = 50.0        # an AABB bigger than this is a degenerate/whole-stage bound, not geometry


def arm_prim_paths(demo):
    """Cached (arm-1, base) prim paths -- `find_path` walks the whole stage."""
    if getattr(demo, "_armp", None) is None:
        demo._armp = (str(find_path(demo.stage, "Arm_1")), str(find_path(demo.stage, "base")))
    return demo._armp


def parked_arm_path(demo):
    """Prim path of the SECOND arm, or None if this stage has one arm.

    It is parked and nothing ever plans for it -- no drive or target write in `morph/` touches
    a `_2` joint -- but it is a solid body bolted to the same chassis and arm-1 must not plan
    through it. Cached: a rigid subtree does not need re-finding."""
    if "_parked_armp" not in demo.__dict__:
        try:
            demo._parked_armp = str(find_path(demo.stage, "Arm_2"))
        except Exception:                        # noqa: BLE001 -- single-arm stage is legal
            demo._parked_armp = None
    return demo._parked_armp


def arm_obstacles(demo, grasp_names=(), *, exclude_names=(), diagnostic_paths=None):
    """AABBs (metres) of every enabled collider outside arm-1, tagged with the frame it is tight
    in ("world" or "chassis" = base); `diagnostic_paths` receives the prim paths in that order.
    `grasp_names` KEEPS a prim as "grasped"; `exclude_names` DROPS one: per-pose probes ONLY."""
    stage = demo.stage
    arm_path, base_path = arm_prim_paths(demo)
    # DEFAULT OFF, and a retreat from a correct change rather than a claim the gap is not real: the
    # parked arm IS solid, but as obstacles it costs the reach its plan at the RRTConnect budget.
    parked = (parked_arm_path(demo)
              if os.environ.get("PARKED_ARM_OBSTACLE", "0") == "1" else None)
    cache_p = demo.__dict__.setdefault("_arm_excl_cache", {})
    for n in tuple(exclude_names) + tuple(grasp_names):
        if n not in cache_p:
            _pth = find_path(stage, n)
            if _pth is None:
                # `str(None)` caches the truthy "None" and `startswith("None")` then matches nothing, so the
                # request silently does nothing while reporting success.
                raise KeyError(f"arm_obstacles: no prim named {n!r} on the stage; the caller "
                               f"named an obstacle that is not there")
            cache_p[n] = str(_pth)
    excl = tuple(cache_p[n] for n in exclude_names)
    grasp = tuple(cache_p[n] for n in grasp_names)
    # ignoreVisibility + every purpose: a default cache silently returns EMPTY for hidden prims.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                              ["default", "render", "proxy", "guide"],
                              useExtentsHint=True, ignoreVisibility=True)
    # World -> base from the STAGE, so a chassis box and its base come from one source.
    inv = UsdGeom.Xformable(stage.GetPrimAtPath(base_path)).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()).GetInverse()

    out, dropped = [], 0
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        p = str(prim.GetPath())
        _parked = bool(parked and p.startswith(parked))
        if _parked and p.rsplit("/", 1)[-1] in HULL_PARTS:
            # The mast's two end caps are convex hulls of PLATES, not solid volume, and the boom passes
            # through them at shipped poses. As obstacles they would wall off a volume that is air.
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False and not _parked:
            # A PHYSICS decision that must not become a PLANNING one: the parked arm's mast and booms
            # are collision-DISABLED while its fingers are not. No collider does not mean no body.
            continue
        if p.startswith(arm_path) or any(e and p.startswith(e) for e in excl):
            continue
        bb = cache.ComputeWorldBound(prim)      # range in prim-local space + local->world
        # WORLD for the parked arm, never "chassis", and this ordering is load-bearing: arm-2 lives under
        # `base/`, and the chassis exemption for column links would blind arm-1's columns to the whole arm.
        tag = ("world" if _parked
               else "chassis" if p.startswith(base_path)
               else "grasped" if any(g and p.startswith(g) for g in grasp) else "world")
        if tag == "chassis":
            # local -> base, THEN align: aligning first and rotating that back re-boxes twice.
            r = Gf.BBox3d(bb.GetRange(), bb.GetMatrix() * inv).ComputeAlignedRange()
        else:
            r = bb.ComputeAlignedRange()
        if r.IsEmpty():
            continue
        mn, mx = np.array(r.GetMin(), float), np.array(r.GetMax(), float)
        # A SPHERE's bound is its authored `extent` BOX carried through the transform, so a roller on a
        # spinning hub arrives as a rotated cube's AABB. A sphere's real bound is centre +- r. Rebuild it.
        if prim.IsA(UsdGeom.Sphere):
            _rad = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
            if _rad:
                # Scale can be anisotropic, which makes the prim an ELLIPSOID; the largest
                # singular value keeps the ball enclosing it rather than cutting into it.
                _xf = bb.GetMatrix() * inv if tag == "chassis" else bb.GetMatrix()
                _lin = np.array([[_xf[i][j] for j in range(3)] for i in range(3)], float)
                _r = float(_rad) * float(np.linalg.svd(_lin, compute_uv=False)[0])
                _c = 0.5 * (mn + mx)
                mn, mx = _c - _r, _c + _r
        if not np.all(np.isfinite(mn)) or not np.all(np.isfinite(mx)):
            dropped += 1
            continue
        if np.any(mx - mn > MAX_SPAN):
            # whole-stage bound, not a real collider
            dropped += 1
            continue
        out.append((np.vstack([mn, mx]), tag))   # each box in the frame its tag names
        if diagnostic_paths is not None:
            diagnostic_paths.append(p)
    if dropped:
        print(f">>> arm obstacles: dropped {dropped} degenerate bound(s) "
              f"(span > {MAX_SPAN:.0f}m or non-finite)", flush=True)
    return out
