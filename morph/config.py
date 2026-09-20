"""Scene constants, paths and tuning defaults.

Imports no package module and no Isaac API, so it is safe to import at any point. Runtime-mutable
values belong to the `Demo` instance, not here.
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root; morph/ is one down
NAV_PLAN = os.path.join(HERE, "nav_plan.py")     # isaac-owned copy of the OMPL planner

# Isaac's bundled Python has no `ompl`, so the planner runs in a separate interpreter.
VENV_PY = os.environ.get("OMPL_PYTHON", os.path.join(HERE, ".venv/bin/python3"))

# OMPL RRTConnect over the eight actuated joints to the aimed descent-start pose. 0 skips the reach
# (nothing is flown in its place); a failed plan refuses it.
ARM_PLANNER = os.environ.get("ARM_PLANNER", "1") == "1"
# Plan the park-to-grasp move when the dock-corridor heuristic vetoes the reach; 0 = teleport.
ARM_PLAN = os.path.join(HERE, "arm_plan.py")
ARM_MODEL = os.path.join(HERE, "usd", "_arm_model.json")
CLOSURE_LUT = os.path.join(HERE, "usd", "_closure_lut.json")


def env_f(name, default):
    """Float from the environment."""
    return float(os.environ.get(name, default))



def env_on(name, default="0"):
    """Switch gate: true when the variable is exactly "1"."""
    return os.environ.get(name, default) == "1"


# ── Run mode ──────────────────────────────────────────────────────────────────────────
HEADLESS = env_on("ISAAC_HEADLESS")

# Set the ENV, not just the constant: the profile block below and several stage gates read
# `os.environ.get("ARM_DRIVE")` directly, so the constant alone leaves the implied set unapplied.
os.environ.setdefault("ARM_DRIVE", "1")
ARM_DRIVE = os.environ.get("ARM_DRIVE", "1") == "1"   # drive-based arm; 0 restores the kinematic path


# The reference grasp, as recorded. Single owner: every module reads KNOWN from here, never
# re-loads the JSON. The open commanded is `gripper.api`'s `GRIPPER_OPEN`, not anything in here.
KNOWN = json.load(open(os.path.join(HERE, "usd/_grasp_known.json")))
KNOWN["object_radius"] = env_f("OBJ_RADIUS", KNOWN["object_radius"])
NUM_OBJECTS = 10


def local_obj(known=None):
    """Object position in the BASE frame at the reference grasp.

    Yaw-invariant: navigating to `P = O - R(psi)*LOCAL_OBJ` facing `psi` reproduces the grasp
    from any approach direction.
    """
    k = known or KNOWN
    bx, by, byaw = k["base_xy_yaw"]
    ox, oy, _ = k["object_xyz"]
    dx, dy = ox - bx, oy - by
    c, s = math.cos(-byaw), math.sin(-byaw)
    return np.array([c * dx - s * dy, s * dx + c * dy])


# Shelf slots, world frame (= the source model's frame).
SHELF_SLOTS = np.array([
    [2.80, -3.72, 0.247], [3.15, -3.72, 0.247], [3.70, -3.72, 0.247], [4.05, -3.72, 0.247],
    [2.85, -3.72, 0.687], [3.70, -3.72, 0.687], [4.05, -3.72, 0.687],
    [2.85, -3.72, 1.190], [3.70, -3.72, 1.190], [4.05, -3.72, 1.190],
], dtype=float)

SHELF_BOARD_T = env_f("SHELF_BOARD_T", "0.010")   # measured board thickness (docs/FINDINGS.md)
# The rack's own top board carries no slot, so unlike every other ceiling here it is not a
# SHELF_SLOTS row. A calibration knob: re-measure it on the rack you are actually running.
RACK_TOP_SURFACE = env_f("RACK_TOP_SURFACE", "1.686")
# Hand-top clearance the insert keeps under the board above the slot, and the SAME number the
# arm-clear gate passes a run on -- tighter would stop runs that grade clean.
ARM_CLEAR_MARGIN = env_f("ARM_CLEAR_MARGIN", "0.020")
# A goal within this of a slot in xy IS that slot -- half the closest slot-to-slot spacing, so at
# most one slot can ever match.
_SLOT_D = np.linalg.norm(SHELF_SLOTS[:, None, :2] - SHELF_SLOTS[None, :, :2], axis=-1)
SLOT_MATCH_R = 0.5 * float(_SLOT_D[_SLOT_D > 1e-9].min())


def board_above(goal_xyz):
    """UNDERSIDE, world z, of the shelf board above the slot `goal_xyz` fills; None when `goal_xyz`
    is not a slot, because off the rack there is no board to derive a ceiling from.

    Derived, not tabulated: the next shelf SURFACE up in SHELF_SLOTS -- or, above the top slot, the
    rack's own top board -- less SHELF_BOARD_T. Move the shelves and every ceiling moves with them.
    """
    g = np.asarray(goal_xyz, float)
    d = np.linalg.norm(SHELF_SLOTS[:, :2] - g[:2], axis=1)
    # One column of the rack carries a slot at every level, so xy alone names a COLUMN, not a
    # slot: the shelf being filled is the highest surface in that column at or below the goal.
    col = [float(z) for z, dd in zip(SHELF_SLOTS[:, 2], d)
           if dd <= SLOT_MATCH_R and z <= float(g[2]) + 1e-6]
    if not col:
        return None
    ups = sorted({float(z) for z in SHELF_SLOTS[:, 2]} | {float(RACK_TOP_SURFACE)})
    nxt = next((z for z in ups if z > max(col) + 1e-6), None)
    return None if nxt is None else nxt - SHELF_BOARD_T


RACK_FRONT_Y = -3.41                 # north face of the middle rack
# Clearance the insert dock keeps between the chassis plate's nearest CORNER and that face.
DOCK_EDGE_M = 0.10
# Reach the insert dock may demand before it stops absorbing the turret into the BASE YAW. It
# therefore chooses the yaw the chassis parks at, which is what the plate is projected through.
DOCK_TURRET_LIM_M = 0.78
PLACE_DOCK_D = 1.15                  # base stands this far north of the face: the raised PARKED arm
                                     # clears the shelves at every x-slot, still in the aisle

# Object spawn, from the reference scene: never inside or beside a rack.
SPAWN_ZONES = (
    ((0.25, 7.77), (-7.30, -4.70)),    # south aisle
    ((0.25, 7.77), (-3.30, -0.79)),    # north aisle (stops at the north rack keepout)
    ((0.25, 2.40), (-4.70, -3.30)),    # west strip between the middle racks
    ((0.25, 2.40), (-0.79, -0.30)),    # narrow strip west of the north rack
)
# Rack rectangles (x0,x1,y0,y1); spawns keep a margin. Defined in morph/rects.py so nav_plan.py
# can read the same table without importing this module's environment profile.
from morph.rects import RACK_RECTS                                            # noqa: E402
SPAWN_EDGE_MARGIN = 0.055 + 0.05                   # obj radius + clearance
MIN_OBJ_SEPARATION = 0.45
SPAWN_ROBOT_KEEP_CENTER = np.array([3.70, -6.00])  # robot start
SPAWN_ROBOT_KEEP_RADIUS = 1.15


def fallback_spawn_xy(placed, blocked, step=0.25):
    """A legal spawn when rejection sampling has exhausted its tries: the first zone point that is
    neither blocked nor within MIN_OBJ_SEPARATION of `placed`, or None if the zones hold no such
    point. `blocked(x, y)` is the same keep-out the sampled spawns pass."""
    for (zx0, zx1), (zy0, zy1) in SPAWN_ZONES:
        y = zy0 + SPAWN_EDGE_MARGIN
        while y <= zy1 - SPAWN_EDGE_MARGIN:
            x = zx0 + SPAWN_EDGE_MARGIN
            while x <= zx1 - SPAWN_EDGE_MARGIN:
                p = np.array([x, y])
                if not blocked(x, y) and all(
                        np.linalg.norm(p - q) >= MIN_OBJ_SEPARATION for q in placed):
                    return p
                x += step
            y += step
    return None


def spawn_seed():
    """Seed for the object scatter. Unset: a fresh layout every run, drawn here so the scatter can
    print it -- a run is reproduced by passing that value back as SEED."""
    pinned = os.environ.get("SEED")
    return int(pinned) if pinned not in (None, "") else int.from_bytes(os.urandom(4), "big")

# Base drive navigation speed.
VMAX = env_f("NAV_VMAX", "32.0")       # cruise
WMAX = env_f("NAV_WMAX", "20.0")       # yaw rate
KP_LIN = env_f("NAV_KP_LIN", "20.0")   # approach gains, matched to VMAX
KP_ANG = env_f("NAV_KP_ANG", "24.0")
print(f">>> nav speed: VMAX {VMAX} WMAX {WMAX} KP_LIN {KP_LIN} KP_ANG {KP_ANG}", flush=True)
WP_TOL, GOAL_TOL = 0.15, 0.06   # tight goal tol -> smaller final base snap (less jerk)

# Shared joint limits: one definition each, used by every stage that writes columns or caps a1.
COLUMN_MAX = 1.40   # column prismatic stop, metres; h1 and h2 clip to [0, COLUMN_MAX]

# Chassis COLLISION plate, mirroring the authored collider in `payloads/base.usda` -- the base
# body's ONLY enabled collider -- as a [min, max] box in the BASE LINK frame. Keep both in step.
CHASSIS_PLATE_X = 0.35
CHASSIS_PLATE_Y = 0.30
CHASSIS_PLATE_BOX = np.array([[-CHASSIS_PLATE_X, -CHASSIS_PLATE_Y, 0.1386],
                              [+CHASSIS_PLATE_X, +CHASSIS_PLATE_Y, 0.156016]], float)


def plate_reach(yaw):
    """How far the chassis plate reaches toward the rack (world -y) with the base parked at `yaw`.

    The plate box's SUPPORT along -y: equal to CHASSIS_PLATE_X only square on to the face, and
    larger at every other yaw, because off square the leading CORNER is what approaches the rack.
    """
    return CHASSIS_PLATE_X * abs(math.sin(yaw)) + CHASSIS_PLATE_Y * abs(math.cos(yaw))


def dock_floor_y(yaw):
    """Northmost insert dock y keeping the plate DOCK_EDGE_M clear of the rack face at `yaw`."""
    return RACK_FRONT_Y + plate_reach(yaw) + DOCK_EDGE_M


# TRACKING cap, not a reachable ceiling: a1's real ceiling falls with column separation
# (`ArmModel.a1_ceil`) and is below this at working tilts. Commandable = min(A1_MAX, a1_ceil(dh)).
A1_MAX = 0.50

# Smallest column differential keeping the boom out of the chassis through the PLACE insert, and
# the SEAT binds rather than the slide. Object-dependent: it scales with the object's half-height.
PLACE_DH_FLOOR = env_f("PLACE_DH_FLOOR", "0.0195")

# Mouth axis in the HAND frame (Gripper_Link3_1), measured; normalised here for all consumers.
_mv = np.array([float(x) for x in os.environ.get(
    "MOUTH_VEC_HAND", "0.2207,-0.177,-0.9591").split(",")], float)
MOUTH_VEC_HAND = _mv / max(1e-9, float(np.linalg.norm(_mv)))

# Tuned grasp settings, applied with `setdefault` so an explicit environment value still wins.
PROFILE = {
    # object and descent
    "OBJ_RADIUS":       "0.040",  # 90 x 180 mm: wider and the knuckles block, narrower and the
    "OBJ_HALF_H":       "0.09",   #   pads miss
    # dock and approach
    "DOCK_VFLOOR":      "3.2",
    "LOWEST_DH_TGT":    "0.13",   # fallbacks; `_solve_lowest` computes the real targets each run
    "LOWEST_A1_TILT":   "0.15",   # retract before tilting: the tilt needs a short boom
    # collision model
    "KEEP_BOXES":       "1",      # the reference pad geoms; both on double-covers each phalanx
    "MESH_PADS":        "0",
    # close servo
    "FB_EFFORT":        "0",
    "FB_RATE_MIN":      "0.35",   # below this the near finger cannot cover its travel in budget
    "FB_T":             "40.0",
    "FB_OVERDRIVE":     "0.10",   # rigid contact is bursty; a narrow band never latches
    "FB_TARGET":        "8.0",    # the force target IS the grip; 4.0 latches too early
    "FB_HI":            "4.0",
    "FB_STEP":          "0.002",
}

# Nav path smoothing; independent of the grasp.
os.environ.setdefault("AH_NAV_SIMPLIFY", "1")

# Grasp and lift tuning.
PROFILE.update({
    "OBJ_IMPULSE":           "0.5",     # contact impulse cap on the object (code default 0.10)
    "FINGER_IMPULSE":        "0.004",   # per-pad cap: the close latches on force, not on a wall
    "PALM_IMPULSE":          "0.004",   # the palm is a backstop, not a scoop
    "PAD_STIFFNESS":         "2e5",     # compliant pads (code default 4.1e4 was too soft to hold)
    "FINGER_JOINT_FRICTION": "0.05",    # joint friction the vendor hand has and the port dropped
    "VENDOR_GAINS":          "1",       # the source model's finger PD, not Isaac's import defaults
    "HOLD_FINGER_KP":        "30",      # a position drive keeps grip force only through stiffness
    "WRIST_KP":              "2e4",     # 3e5 on the light wrist links rings at 240 Hz
    "WRAP_CURL":             "0.0",     # no extra distal curl: it levers the object out
    "SEAT_GUARDED":          "1",       # the guarded seat owns the approach
    "LIFT_DRIVE":            "1",       # lift on drives
    "LIFT_PIN":              "0",       # ...so no root pin during it (the mixed pair runs away)
    "CARRY_LEVEL":           "0",       # levelling at the end of the lift swung the wrist
    "OBJ_GOAL_SET":          "1",       # Multi-goal IK candidate set; avoids all-miss solve time
    "TRACK_ERR":             "1",       # Arms per-step tracking check; produces TRACK BREACH
                                        # lines, which are the ONLY evidence verify_place's track_gate
                                        # can read: with this off, that gate cannot fail any run.
})

for _k, _v in PROFILE.items():
    # an explicit environment value still wins
    os.environ.setdefault(_k, _v)
# ARM_DRIVE=1 switches the three motion primitives (set_base/_force/_apply) in morph/robot.py
# to drives, and implies the stability set validated with them.
if os.environ.get("ARM_DRIVE", "0") == "1":
    for _k, _v in (("PIN_DRIVE_BRAKE", "1"),
                   ("CAPTURE_DRIVE", "1"), ("REST_OFF", "0.0015"),
                   # rollers must carry the floor contact; hub colliders lock lateral and yaw
                   ("WHEEL_HUB_COLLIDE", "0"),
                   # Every entry must name a stage some code actually SETS: `_drive_on` matches
                   # `self._stage` against this list, so a name nothing sets is silently inert.
                   ("ARM_DRIVE_STAGE", "descend,close,capture,lift,carry,place"),
                   ("FB_ZHOLD", "0"), ("PALM_GATE", "0"),
                   # geometric aimed goal, fine-align off, seat band +20 mm
                   ("OBJ_GOAL", "2"), ("SEAT_BAND_DZ", "0.02"),
                   # the close's align-pinch owns the final approach; the seat servo is the
                   # alternative and is off (SEAT_RELATION=1 re-enables it)
                   ("CLOSE_ALIGN_AFTER_SEAT", "1"), ("SEAT_RELATION", "0")):
        os.environ.setdefault(_k, _v)
    print(">>> ARM_DRIVE=1: drives own the arm (brake, drive capture, "
          "rest offset 1.5mm implied; stages: %s)"
          % os.environ.get("ARM_DRIVE_STAGE", "all"), flush=True)
print(">>> profile: applied %d settings (morph/config.py)"
      % len(PROFILE), flush=True)
# Re-sync: the OBJ_RADIUS read at the top of this file runs before the profile sets it, so
# without this the scene builds one cylinder while the close stack computes against another.
KNOWN["object_radius"] = env_f("OBJ_RADIUS", KNOWN["object_radius"])
print(f">>> object radius for the close stack: {KNOWN['object_radius']:.3f} (synced to "
      f"OBJ_RADIUS after the profile)", flush=True)
