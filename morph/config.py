"""Scene constants, paths and the grip-mode selection.

Pure data plus two small environment helpers. This module imports nothing from the rest of the
package, so it cannot create an import cycle, and it imports no Isaac API, so it is safe to import
at any point — before or after `SimulationApp` exists.

Runtime-mutable values (the object half-height, the measured base-to-object relation) deliberately
do not live here: they are read from the live scene and belong to the `Demo` instance.
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root; morph/ is one down
REPO = os.path.dirname(HERE)
NAV_PLAN = os.path.join(HERE, "nav_plan.py")     # isaac-owned copy of the OMPL planner

# The OMPL planner runs in a separate interpreter — Isaac's bundled Python has no `ompl`.
# OMPL_PYTHON points at any interpreter that does; see the README for how to build one.
VENV_PY = os.environ.get("OMPL_PYTHON", os.path.join(REPO, ".venv/bin/python3"))


def env_f(name, default):
    """Float from the environment."""
    return float(os.environ.get(name, default))



def env_on(name, default="0"):
    """Switch gate: true when the variable is exactly "1"."""
    return os.environ.get(name, default) == "1"


# ── Run mode ──────────────────────────────────────────────────────────────────────────
HEADLESS = env_on("ISAAC_HEADLESS")

# The two grip modes, selected with GRIP_MODE:
#   practical  (default) The object is held by a FixedJoint weld, so the grasp cannot slip. The
#              verified demo path: full pick-and-place across all three shelf levels.
#   perfect    A real dynamic body held only by pad friction -- no weld, the physics a real
#              gripper faces. The contact-physics development profile; asked for by name.
# Note that the env var FRICTION is NOT a mode: it is the friction coefficient mu, read in scene.py.
PERFECT = os.environ.get("GRIP_MODE", "practical") == "perfect"


# ── The reference grasp ─────────────────────────────────────────────────────────────── Single
# owner: every module reads KNOWN from here and never re-loads the JSON.
KNOWN = json.load(open(os.path.join(HERE, "usd/_grasp_known.json")))
KNOWN["object_radius"] = env_f("OBJ_RADIUS", KNOWN["object_radius"])
# One open pose, not two. The recorded `fingers_open` is asymmetric -- an artifact of how the
# reference grasp was captured, not something the geometry asks for -- and it shows: the hand
# approaches with the thumb splayed and b/c half shut, the descent arrives with the thumb already
# touching, and a whole stage goes on widening b/c to match.
if os.environ.get("FINGERS_OPEN_SYM", "1") == "1":
    _fo = KNOWN.get("fingers_open", {})
    _j1 = [k for k in _fo if k.endswith("joint_1_1")]
    if _j1:
        _widest = min(float(_fo[k]) for k in _j1)
        if any(abs(float(_fo[k]) - _widest) > 1e-6 for k in _j1):
            print(">>> fingers_open: symmetrised to %+.3f on a/b/c (was %s) -- one open pose"
                  % (_widest, {k.split('_')[1]: round(float(_fo[k]), 3) for k in sorted(_j1)}),
                  flush=True)
            for _k in _j1:
                _fo[_k] = _widest
NUM_OBJECTS = 10

# Symmetric open for the side grip, perfect mode only. The recorded asymmetry belongs to a TOP-DOWN
# grasp, where the thumb opens wide to clear the object's top face; a side grip has nothing to
# clear.
if PERFECT and env_on("SIDE_GRIP_SYMMETRIC", "1"):
    _open = env_f("GRIPPER_OPEN_POS", -0.55)
    for _f in "abc":
        KNOWN["fingers_open"][f"finger_{_f}_joint_1_1"] = _open
    os.environ.setdefault("J1_WIDEST", str(_open))   # prep/reach retract to this open, not to -1.04


def local_obj(known=None):
    """Object position in the BASE frame at the reference grasp.

    The whole robot is yaw-invariant: at ANY approach yaw the fixed arm config puts the object at
    this base-frame point, so navigating to `P = O - R(psi)*LOCAL_OBJ` facing `psi` reproduces the
    grasp from any direction.
    """
    k = known or KNOWN
    bx, by, byaw = k["base_xy_yaw"]
    ox, oy, _ = k["object_xyz"]
    dx, dy = ox - bx, oy - by
    c, s = math.cos(-byaw), math.sin(-byaw)
    v = np.array([c * dx - s * dy, s * dx + c * dy])
    # LOCAL_OBJ_OFF adds a base-frame (dx, dy) to the docking point, in perfect mode only.
    if PERFECT:
        off = os.environ.get("LOCAL_OBJ_OFF", "")
        if off:
            v = v + np.array([float(x) for x in off.split(",")[:2]], float)
    return v


# ── Shelf slots ─────────────────────────────────────────────────────────────────────── The world
# frame matches the source model's frame.
SHELF_SLOTS = np.array([
    [2.80, -3.72, 0.247], [3.15, -3.72, 0.247], [3.70, -3.72, 0.247], [4.05, -3.72, 0.247],
    [2.85, -3.72, 0.687], [3.70, -3.72, 0.687], [4.05, -3.72, 0.687],
    [2.85, -3.72, 1.190], [3.70, -3.72, 1.190], [4.05, -3.72, 1.190],
], dtype=float)

RACK_FRONT_Y = -3.41                 # north face of the middle rack
PLACE_DOCK_D = 1.15                  # base stands this far north of the rack's front face: far enough
                                     # that the raised PARKED arm clears the shelves at every
                                     # x-slot, while staying in the open north aisle.

# ── Object spawn ────────────────────────────────────────────────────────────────────── Taken from
# the reference scene so objects land where the working simulation puts them — never inside or
# beside a rack, so a pick never grazes rack geometry.
SPAWN_ZONES = (
    ((0.25, 7.77), (-7.30, -4.70)),    # south aisle
    ((0.25, 7.77), (-3.30, -0.79)),    # north aisle (stops at the north rack keepout)
    ((0.25, 2.40), (-4.70, -3.30)),    # west strip between the middle racks
    ((0.25, 2.40), (-0.79, -0.30)),    # narrow strip west of the north rack
)
# Rack rectangles (x0,x1,y0,y1) — same as nav_plan.py OBSTACLE_RECTS (minus the walls); objects keep
# a margin off these so they never spawn on a rack.
RACK_RECTS = (
    (2.55, 4.34, -3.96, -3.41), (4.34, 7.96, -3.96, -3.41), (2.54, 7.96, -4.55, -3.97),
    (0.72, 7.97, -7.96, -7.42), (2.54, 7.96, -0.66, -0.04),
)
SPAWN_EDGE_MARGIN = 0.055 + 0.05                   # obj radius + clearance
MIN_OBJ_SEPARATION = 0.45
SPAWN_ROBOT_KEEP_CENTER = np.array([3.70, -6.00])  # robot start
SPAWN_ROBOT_KEEP_RADIUS = 1.15

# Base drive Navigation speed is mode-independent: it runs before the hand does anything.
VMAX = env_f("NAV_VMAX", "32.0")       # cruise
WMAX = env_f("NAV_WMAX", "20.0")       # yaw rate
KP_LIN = env_f("NAV_KP_LIN", "20.0")   # approach gains, matched to VMAX
KP_ANG = env_f("NAV_KP_ANG", "24.0")
print(f">>> nav speed: VMAX {VMAX} WMAX {WMAX} KP_LIN {KP_LIN} KP_ANG {KP_ANG} "
      f"(same in both grip modes)", flush=True)
WP_TOL, GOAL_TOL = 0.15, 0.06

# Shared joint limits -- one definition each; every stage that writes the columns or caps the boom
# extension uses these.
COLUMN_MAX = 1.40   # column prismatic stop, metres; h1 and h2 clip to [0, COLUMN_MAX]
A1_MAX = 0.50       # operational boom-extension cap; above it pinch tracking jumps

# Mouth axis in the HAND frame (Gripper_Link3_1), measured; unit-normalised here so every consumer
# shares one definition.
_mv = np.array([float(x) for x in os.environ.get(
    "MOUTH_VEC_HAND", "0.2207,-0.177,-0.9591").split(",")], float)
MOUTH_VEC_HAND = _mv / max(1e-9, float(np.linalg.norm(_mv)))
   # tight goal tol -> smaller final base snap (less jerk before the grasp)

# Perfect-mode grasp profile: the one place the tuned settings live. Applied with `setdefault`, so
# an explicit environment value still wins -- a floor, not a cage. GRASP_PROFILE=0 opts out.
PERFECT_PROFILE = {
    # --- object and descent ---------------------------------------------------------------------
    "OBJ_RADIUS":       "0.040",  # 90 x 180mm: wider and the knuckles block, narrower and
    "OBJ_HALF_H":       "0.09",   #   the pads miss entirely
    "DESCENT_MODE":     "computed",  # the baked drop lands at the recorded object's height
    "DESCENT_H_FLOOR":  "0.02",   # must allow the solved lowest pose, which bottoms the columns
    "DESCENT_BACKOFF":  "0.0",    # never push the chassis to make room
    # --- dock and approach ----------------------------------------------------------------------
    "DOCK_VFLOOR": "3.2",
    "NO_CHASSIS_APPROACH": "1",   # the base parks at the dock and never moves again
    "ENCLOSE_LOWEST":   "1",      # reach the lowest safe pose before the forward reach
    # Fallbacks only; `_solve_lowest` computes the real targets each run. The tilt phase only has to
    # get LOW, not to reach -- the forward stage straightens the boom back out to buy reach.
    "LOWEST_DH_TGT":    "0.13",
    "LOWEST_A1_LOW":    "0.50",
    "LOWEST_A1_TILT":   "0.15",   # retract to here BEFORE tilting; the tilt needs a short boom
    "ENCLOSE_DU":       "-0.095", # centre of the reachable window (-70..-130mm)
    "ENCLOSE_DV_TGT":   "0.0",
    "ENCLOSE_Z_MODE":   "obj",    # servo z to the OBJECT, not to the inherited height
    "FORCE_CLOSE":      "1",
    "WRAP":             "0",
    "KIN_SETTLE":       "0",
    "GRIP_DROP_AUTO":   "0",      # the automatic drop double-counts the required descent
    # --- collision model ------------------------------------------------------------------------
    "KEEP_BOXES":     "1",        # the reference pad geoms; without them nothing has contact
    "MESH_PADS":      "0",        #   geometry at all, and running both double-covers each phalanx
    # --- force signal ---------------------------------------------------------------------------
    # Windowed contact impulse.
    "FB_EFFORT":      "0",
    # --- approach gates -------------------------------------------------------------------------
    "CLOSE_GEO_STOP": "1",        # without it the replay ploughs the object through the floor
    "CLOSE_STANDOFF": "0.008",
    "ALIGN_EQ":       "1",        # equalise the pad gaps before the close
    "ENCLOSE_A1_MAX": "0.50",     # above this a1 stops tracking and the pinch jumps
    "ENCLOSE_JOINT":  "1",        # direct joint servo for the constant-height forward reach
    # --- lateral authority ----------------------------------------------------------------------
    # The per-iteration dv correction must exceed the per-iteration disturbance or dv can only
    # diverge; much above this the arm sweeps into the object.
    "ENCLOSE_TH_CAP": "0.025",
    # Stop on the NEAREST BODY, not the pads: the arm write is kinematic, so any link landing inside
    # the object makes the solver eject it.
    "ENCLOSE_STOP_ON": "wrap",
    # --- close servo ----------------------------------------------------------------------------
    "FB_RATE_MIN":    "0.35",     # below this the near finger cannot cover its travel in budget
    "FB_T":           "40.0",
    # Contact on a rigid object is bursty, so a narrow band never latches -- every burst trips the
    # retract and the finger hits again.
    "FB_OVERDRIVE":   "0.10",
    "FB_TARGET":      "4.0",
    "FB_HI":          "4.0",
    "FB_STEP":        "0.0006",
}

# Deliberately NOT in the profile, because they depend on the object or the run:
#   OBJ_RADIUS / OBJ_HALF_H  the hand's measured limits -- the pads span about 87mm of diameter,
#                            and an object must be under ~33mm to pass BETWEEN the knuckles
#   ENCLOSE_DU               the advance target, roughly -(radius + 10mm). A large object's value
#                            on a small one stops the servo short and looks like a reach limit
#   ENCLOSE_DV_TGT           the jaw-centring target
#   NAV_SEED                 required for any run-to-run comparison: the planner is randomised

# Path smoothing applies in BOTH grip modes: navigation runs before the grasp, and the planner does
# not know what the hand will do.
os.environ.setdefault("AH_NAV_SIMPLIFY", "1")

# Practical mode with a smaller object, opted into by setting OBJ_RADIUS -- a plain practical run
# sets none, so nothing here fires and the verified configuration is untouched.
if not PERFECT and os.environ.get("OBJ_RADIUS"):
    os.environ.setdefault("DROP_TRIM", "-0.12")
    print(">>> pin + small object: DROP_TRIM %s (descent deepened to the small object's flank)"
          % os.environ["DROP_TRIM"], flush=True)

if PERFECT and os.environ.get("GRASP_PROFILE", "1") not in ("0", "off", "none"):
    for _k, _v in PERFECT_PROFILE.items():
        os.environ.setdefault(_k, _v)     # an explicit environment value still wins
    print(">>> perfect profile: applied %d settings (morph/config.py)"
          % len(PERFECT_PROFILE), flush=True)
    # Re-sync KNOWN to the profile's object size. The OBJ_RADIUS read near the top of this file runs
    # at import, BEFORE the profile sets OBJ_RADIUS — so without this line the scene would build one
    # cylinder while every close-stage gap, seat and stop computed against another.
    KNOWN["object_radius"] = env_f("OBJ_RADIUS", KNOWN["object_radius"])
    print(f">>> object radius for the close stack: {KNOWN['object_radius']:.3f} (synced to "
              f"OBJ_RADIUS after the profile)", flush=True)
