"""Entry point: menu-driven pick and place in Isaac Sim.

An `omni.ui` panel selects an object and a shelf slot. On MOVE, the robot plans a path to the
object with OMPL, drives there, grasps it, plans the carry leg to the rack, places the object on
the chosen slot and releases it. The objects are randomly scattered at start-up.

This module owns only the bootstrap: it constructs `SimulationApp`, composes `Demo` from the
mixins in `morph/`, and runs `main()`. Everything else lives in that package — see the README for
the module map.

    "$ISAAC/python.sh" play_isaac.py

        GRIP_MODE=perfect    friction-only grip (default: practical, the welded hold)
        ISAAC_HEADLESS=1     no window; runs one demo cycle and exits
        DEMO_CYCLES="3:6"    headless run list of object:slot pairs
        SEED=1               object-scatter seed
"""
import json
import math
import os
import subprocess

from isaacsim import SimulationApp

# Safe to import before SimulationApp: config.py touches no Isaac API.
from morph.config import PERFECT, HEADLESS

# An interrupted run leaves /dev/shm/carb-* behind and the next SimulationApp blocks on the dead
# owner's lock. Segments whose pid is gone are safe to remove.
def _sweep_stale_carb_shm():
    import glob, re
    n = 0
    for f in glob.glob("/dev/shm/carb-*") + glob.glob("/dev/shm/sem.carb-*"):
        # pid sits at the end (carb-RStringInternals-<pid>) or mid-name (carb-ringbuffer-<pid>-0x..)
        m = re.search(r"-(\d+)(?:$|-0x)", f)
        if not m:
            continue
        try:
            os.kill(int(m.group(1)), 0)
        except ProcessLookupError:
            try:
                os.remove(f)
                n += 1
            except OSError:
                pass
        except PermissionError:
            pass
    # The global semaphore has no pid suffix; drop it only once no owner is alive.
    _alive = False
    for f in glob.glob("/dev/shm/carb-RStringInternals-*"):
        m = re.search(r"-(\d+)$", f)
        if m:
            try:
                os.kill(int(m.group(1)), 0)
                _alive = True
            except (ProcessLookupError, PermissionError):
                pass
    if not _alive and os.path.exists("/dev/shm/sem.carbonite-sharedmemory"):
        try:
            os.remove("/dev/shm/sem.carbonite-sharedmemory")
            n += 1
        except OSError:
            pass
    if n:
        print(f">>> swept {n} stale carb shm objects from dead runs", flush=True)

_sweep_stale_carb_shm()
simulation_app = SimulationApp({"headless": HEADLESS})

# Mute harmless one-time startup warnings so the terminal shows only the >>> progress lines.
if os.environ.get("LOG_LEVEL") != "warning":
    try:
        import carb
        carb.settings.get_settings().set("/persistent/app/usd/muteUsdDiagnostics", True)
    except Exception:
        pass
    try:                                                    # silence the noisy per-plugin warning channels
        import omni.log
        _log = omni.log.get_log()
        for _ch in ("omni.physx.plugin", "omni.physx.tensors.plugin", "omni.usd",
                    "omni.hydra", "rtx", "pxr.Semantics", "omni.replicator.core"):
            try:
                _log.set_channel_enabled(_ch, False, omni.log.SettingBehavior.OVERRIDE)
            except Exception:
                pass
    except Exception:
        pass

import numpy as np
import scene
# Imported below SimulationApp: these modules use Isaac APIs at module level.
from morph.align import AlignMixin
from morph.close import CloseMixin
from morph.diagnostics import DiagnosticsMixin
from morph.fingers import FingersMixin
from morph.gui import build_menu
from morph.kinematics import KinematicsMixin
from morph.navigation import NavigationMixin, plan_path
from morph.pick import PickMixin
from morph.place import PlaceMixin
from morph.robot import RobotMixin
from morph.world import WorldMixin
# HEADLESS/PERFECT come from config.py above -- they are needed before this import can happen.
from morph.config import (                      # single owner of the scene data — never re-load it
    HERE, REPO, VENV_PY, NAV_PLAN, KNOWN, NUM_OBJECTS,
    SHELF_SLOTS, RACK_FRONT_Y, PLACE_DOCK_D,
    SPAWN_ZONES, RACK_RECTS, SPAWN_EDGE_MARGIN, MIN_OBJ_SEPARATION,
    SPAWN_ROBOT_KEEP_CENTER, SPAWN_ROBOT_KEEP_RADIUS,
    VMAX, WMAX, KP_LIN, KP_ANG, WP_TOL, GOAL_TOL, local_obj as _local_obj)
from morph.geometry import quat_yaw, yaw_of, wrap, quat_to_R, R_to_quat
from morph.usd_utils import find_path
from isaacsim.core.prims import RigidPrim, SingleXFormPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Usd, UsdPhysics

# Startup defaults; Demo measures the real values in __init__. Module-level for the probes.
OBJ_HALF_H = float(os.environ.get("OBJ_HALF_H", KNOWN["object_half_h"]))
# The json's object_half_h does not match the shipped cylinder, so every clearance derived from it
# is optimistic. _sync_object_size() overwrites it from the live USD geometry at startup.
LOCAL_OBJ = _local_obj()
# Park at the distance the grasp happens from. The chassis never pushes in afterwards, so docking at
# the navigation distance would leave the arm reaching from further out than it can manage.
DOCK_DIST = 0.64
if DOCK_DIST > 0:
    _lo_n = float(np.linalg.norm(LOCAL_OBJ))
    if _lo_n > 1e-6:
        LOCAL_OBJ = LOCAL_OBJ * (DOCK_DIST / _lo_n)
        print(f">>> dock distance: {_lo_n:.3f} -> {DOCK_DIST:.3f}m "
              f"(park at the grasp distance; the chassis never pushes in)", flush=True)
PICK_STANDOFF = float(np.linalg.norm(LOCAL_OBJ))   # base sits this far behind the object





class Demo(PickMixin, PlaceMixin, CloseMixin, FingersMixin, AlignMixin, KinematicsMixin,
           NavigationMixin, WorldMixin, RobotMixin, DiagnosticsMixin):
    """The demo robot. This class owns ALL the state; every mixin above only touches `self`.

    The mixins carry no state and no name collisions, so the order above is documentation —
    roughly outermost phase to innermost primitive — not behaviour.
    """

    def __init__(self):
        # The arm is kinematically forced in every phase, so the runtime loop-closure joint is
        # redundant -- and toggling jointEnabled invalidates the articulation view in headed mode.
        os.environ.setdefault("NO_LOOP", "1")
        # Measured at startup, not constant: _sync_object_size() from the USD geometry and
        # _load_close_traj() from the recorded grasp. On the instance so there is one owner.
        self.obj_half_h = OBJ_HALF_H
        self.local_obj = LOCAL_OBJ
        self.pick_standoff = PICK_STANDOFF
        self.world, self.robot, self.names, self.q0 = scene.load_scene(simulation_app)
        # Physics runs at 240Hz and every stage steps with render=True, so a headed run asks for 240
        # frames per simulated second and the draw calls dominate. Render every Nth step.
        _rev = max(1, int(float(os.environ.get("RENDER_EVERY", "4"))))
        if _rev > 1 and not HEADLESS:
            _raw_step, _tick = self.world.step, [0]

            def _decimated_step(render=True, **kw):
                _tick[0] += 1
                return _raw_step(render=bool(render) and _tick[0] % _rev == 0, **kw)
            self.world.step = _decimated_step
            print(f">>> render decimation: drawing 1 frame per {_rev} physics steps "
                  f"({240 // _rev} FPS at 240Hz); RENDER_EVERY=1 to disable", flush=True)
        # Frame the camera before the scene build, or the viewer sits at Kit's default and snaps.
        if not HEADLESS:
            try:
                set_camera_view(eye=[7.5, -8.0, 4.5], target=[3.5, -4.0, 0.3])
            except Exception:
                pass
        self.ctrl = self.robot.get_articulation_controller()
        self.idx = {n: i for i, n in enumerate(self.names)}
        # `_finger_cmd`, when set, means the fingers own an object: kinematic writes skip those DOFs
        # so real contact can stop them, and `_apply` drives them at this target instead.
        self._f_idx = np.array([i for n, i in self.idx.items()
                                if n.startswith("finger_") or n.startswith("palm_finger")])
        self._arm_ii = np.array([i for i in range(len(self.names)) if i not in set(self._f_idx)])
        # Wheels and rollers are never written: the per-step hold vector would erase the roll
        # `set_base` computes on the same step. Free-spinning bodies, no commanded position.
        self._roll_ii = np.array([i for n, i in self.idx.items()
                                  if "rolling_joint" in n or "slipping" in n])
        _excl = set(self._f_idx) | set(self._roll_ii.tolist())
        self._arm_ii = np.array([i for i in range(len(self.names)) if i not in _excl])
        self._hold_ii = np.array([i for i in range(len(self.names))
                                  if i not in set(self._roll_ii.tolist())])
        self._f_names = [self.names[i] for i in self._f_idx]
        self._finger_cmd = None
        # Both are invisible to _ctc/_fN, which require a pickup_obj on one side of the pair.
        self._ctc_floor = {}                                 # finger <-> floor contacts
        self._ctc_any = {}                                   # any partner of a hand link
        self.dt = self.world.get_physics_dt()
        self.stage = self.world.stage
        # per-object RigidPrim handles + the grip-material pass (friction to the PxShapes)
        self._apply_grip_material()
        self.world.reset()                                   # re-flow the material to the PxShapes...
        self.robot.initialize()
        # world.reset() reverts the arm to the USD default and zeros the drive targets, so PARK, the
        # gains and q0 all have to be re-established or the heavy arm slams to zero.
        scene.set_arm_gains(self.robot, self.names)
        self.robot.set_joint_positions(self.q0)
        self.objs = [RigidPrim(f"/World/Geometry/pickup_obj_{i}",
                               prepare_contact_sensors=PERFECT,    # friction: real contact evidence
                               track_contact_forces=PERFECT, max_contact_count=16)
                     for i in range(NUM_OBJECTS)]
        self.z0 = float(self.robot.get_world_pose()[0][2])   # base root height (holonomic drive keeps it)
        self.grip = None                                     # active finger command while carrying
        # Cached: find_path traverses the whole stage, so resolve the gripper link once.
        self._grip_prim = SingleXFormPrim(find_path(self.stage, "Gripper_Link1_1"))
        self._obj_local = None                               # object pose in the hand frame, at capture
        self._weld_path = None                               # live grasp-weld prim path (practical mode)
        self.placed = []                                     # (obj_idx, world_pos) kept resting on the shelf
        self._idle_anchor = None                             # FIXED base pose held while idle (never re-read)
        self.traj = self._load_trajs()                       # baked loop-consistent trajectories
        self.close_traj = self._load_close_traj()            # the recorded finger close
        try:                                                 # closure LUT: (h2-h1, a1) -> passive linkage joints
            self.clut = json.load(open(os.path.join(HERE, "usd/_closure_lut.json")))
        except Exception:
            self.clut = None
        self.jog = None                                      # active cartesian arm-jog state (GUI sliders)
        self._sync_object_size()                             # measure the object, don't trust the json
        # DOCK_DIST is measured to the object's AXIS, so extra radius comes straight off the surface
        # clearance the approach is calibrated against.
        if os.environ.get("DOCK_SIZE_AWARE", "1") == "1":
            _r_ref = 0.040
            _dr = float(KNOWN["object_radius"]) - _r_ref
            _n0 = float(np.linalg.norm(self.local_obj))
            if abs(_dr) > 1e-4 and _n0 > 1e-6:
                self.local_obj = self.local_obj * ((_n0 + _dr) / _n0)
                self.pick_standoff = float(np.linalg.norm(self.local_obj))
                print(f">>> dock size-aware: object radius {float(KNOWN['object_radius']):.3f} vs "
                      f"reference {_r_ref:.3f} -> dock {_n0:.3f} -> {self.pick_standoff:.3f}m "
                      f"({_dr * 1000:+.0f}mm), holding the same "
                      f"{(_n0 - _r_ref) * 1000:.0f}mm surface clearance", flush=True)
        self.scatter_objects()
        self.park()
        scene.reveal_viewport(self.world, simulation_app)    # present now: scene built, scattered, parked
        self._touch = {}
        self._touch_step = {}
        self._fN_step = {}
        self._ctc_obj = {}
        self._ctcN = {}                                       # per-LINK accumulated impulse
        self._hit_step = False
        self._fN = {}
        self._ctc = {}
        if PERFECT or os.environ.get("DIAG_CONTACT") == "1":
            # Contact events double as the per-finger touch sensor: a yielding object never stalls
            # the finger pushing it, so drive lag cannot detect contact.
            from omni.physx import get_physx_simulation_interface
            from pxr import PhysicsSchemaTools
            dbg_c = os.environ.get("DIAG_CONTACT") == "1"

            def _on_contact(contact_headers, contact_data):
                # `_fN` accumulates over a window the READER resets. Per-batch would show only the
                # last step, which for a resting grip is usually empty.
                for h in contact_headers:
                    try:
                        a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
                        a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
                    except Exception:
                        continue
                    # Own ledger, before the pickup_obj filter below: a finger jammed against the
                    # floor reports nothing there while silently stopping the joint tracking.
                    if ("finger_" in a0 or "finger_" in a1) and ("floor" in a0 or "floor" in a1):
                        _fl = a0 if "finger_" in a0 else a1
                        _k = _fl.rsplit("/", 1)[-1]
                        self._ctc_floor[_k] = self._ctc_floor.get(_k, 0) + 1
                    # Catch-all ledger: a finger against the palm or its own neighbour has neither a
                    # pickup_obj nor a floor in the pair.
                    _HAND = ("finger_", "Gripper_Link", "palm")
                    _h0 = any(_k in a0 for _k in _HAND)
                    _h1 = any(_k in a1 for _k in _HAND)
                    if _h0 or _h1:
                        _fa = a0 if _h0 else a1
                        _ob = a1 if _h0 else a0
                        _pair = f"{_fa.rsplit('/', 1)[-1]}<->{_ob.rsplit('/', 1)[-1]}"
                        self._ctc_any[_pair] = self._ctc_any.get(_pair, 0) + 1
                    if "pickup_obj" not in a0 and "pickup_obj" not in a1:
                        continue
                    # It must be THE object being grasped: `_fN`/`_fN_step` are the force sensor
                    # every close stage servos on, and a brush against a bystander would read as pad
                    # load.
                    _fo = getattr(self, "_focus_obj", None)
                    if _fo is not None and _fo not in a0 and _fo not in a1:
                        continue
                    other = a0 if "pickup_obj" in a1 else a1
                    imp = 0.0
                    try:                                     # per-contact impulses = REAL pad load
                        for ci in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                            cd = contact_data[ci]
                            imp += float(np.linalg.norm([cd.impulse.x, cd.impulse.y, cd.impulse.z]))
                    except Exception:
                        pass
                    # Which robot part is touching the object. A proximity scan over prim origins
                    # cannot answer this -- a finger link's origin sits ~10cm from its pad face.
                    key = other.rsplit("/", 1)[-1]
                    self._ctc[key] = self._ctc.get(key, 0) + 1
                    # Record which object, so a gap-metric disagreement can be told apart from a
                    # bystander leaking past the focus filter.
                    _op = a0 if "pickup_obj" in a0 else a1
                    self._ctc_obj[key] = _op.rsplit("/", 1)[-1]
                    # Per-link impulse: an event count cannot tell a load-bearing contact from a
                    # graze, and a per-finger sum cannot say which link is carrying.
                    self._ctcN[key] = self._ctcN.get(key, 0.0) + imp
                    for fl in "abc":
                        if f"finger_{fl}_link" in other:
                            self._touch[fl] = True
                            self._touch_step[fl] = True      # THIS step only (see _animate_fingers)
                            self._fN[fl] = self._fN.get(fl, 0.0) + imp
                            # Per-step, for the force-feedback loops: `_fN` above accumulates over a
                            # whole phase, which a closed loop cannot use.
                            self._fN_step[fl] = self._fN_step.get(fl, 0.0) + imp
                    if "Gripper_Link3_1" in other and "finger" not in other:
                        self._touch["palm"] = True
                        self._touch_step["palm"] = True
                        self._fN["palm"] = self._fN.get("palm", 0.0) + imp
                        # The real contact position. A nearest-hull-vertex proxy can report a vertex
                        # outside the surface, i.e. touching nothing.
                        try:
                            cd0 = contact_data[h.contact_data_offset]
                            self._palm_ct = (np.array([cd0.position.x, cd0.position.y,
                                                       cd0.position.z], float), other)
                        except Exception:
                            pass
                    if "floor" not in other and "robot" in other:
                        self._hit_step = True                # ANY robot part touching the object
                    if dbg_c and "floor" not in other:
                        print(f">>> CONTACT-EVENT: {a0}  <->  {a1}", flush=True)

            self._contact_sub = get_physx_simulation_interface().subscribe_contact_report_events(_on_contact)
            print(">>> contact-event touch sensor ON", flush=True)

    # ---- setup -----------------------------------------------------------------























    # Aliases onto morph/geometry.py, so the call sites keep working without a second copy.
    _quat_to_R = staticmethod(quat_to_R)
    _R_to_quat = staticmethod(R_to_quat)





    # ---- cartesian arm jog: the GUI's X/Y/Z sliders ---------------------------- Damped-least-
    # squares steps on the four-bar FK Jacobian over [h1, h2, a1, th], with the error measured on
    # the LIVE gripper so the loop converges on the real position.










    # There is deliberately no descent palm-clearance gate: the world AABB of the tilted palm shell
    # over-reports overlap against the real mesh, so any such gate cuts the descent short and the
    # fingers close on air.

















    # ---- driving ---------------------------------------------------------------


    # ---- pick ------------------------------------------------------------------















    # ---- place -----------------------------------------------------------------


    def run_cycle(self, obj_idx, slot_idx, status=None):
        self._idle_anchor = None                             # cycle moves the base; idle re-anchors after
        self._parked_ramped = False                          # ...and the park ramp runs once per cycle
        self.jog = None                                      # any active arm jog ends when a cycle starts
        self._finger_cmd = None                              # no object owned at cycle start
        self._tighten_total = 0.0
        try:
            self._recover()                                  # start each cycle from a sane robot pose
            held = self.pick(obj_idx, status=status)
            if not held and os.environ.get("PICK_RETRY", "1") == "1":
                # The residual forced-linkage explosion is nondeterministic: recover and retry once
                # with the replay off, so a demo cycle never ends failed.
                print(f">>> pick obj {obj_idx} failed -> RECOVER + retry", flush=True)
                self._recover()
                prev = os.environ.get("REACH_REPLAY", "1")
                if not PERFECT:
                    os.environ["REACH_REPLAY"] = "0"         # teleport fallback force-moves the object
                try:                                         #   between colliding fingers -> replay-only
                    held = self.pick(obj_idx, status=status)  #   retry in perfect mode
                finally:
                    os.environ["REACH_REPLAY"] = prev
            if held:
                self.place(obj_idx, slot_idx, status=status)
            if status:
                status(f"done: obj {obj_idx} -> slot {slot_idx}")
        except Exception as e:
            import traceback
            print(f"!! cycle failed: {e}\n{traceback.format_exc()}", flush=True)
            if status:
                status(f"error: {e}")
            try:                                             # a dropped articulation view (headed-mode USD
                self.robot.initialize()                      # churn) otherwise bricks every later apply_action
                self.ctrl = self.robot.get_articulation_controller()
                scene.set_arm_gains(self.robot, self.names)
                print(">>> articulation view re-initialized", flush=True)
            except Exception:
                pass
        self._recover()                                      # and leave it sane for the next one





def main():
    demo = Demo()
    grip_c = demo.grip_pos()
    # Framed once in scene.load_scene; re-issuing here makes the view hop on start-up.

    if HEADLESS and os.environ.get("JOG_TEST") == "1":       # cartesian-jog accuracy test, then exit
        demo.start_jog()
        worst = 0.0
        for name, dv in (("+x", [0.15, 0, 0]), ("-x", [-0.15, 0, 0]), ("+y", [0, 0.15, 0]),
                         ("-y", [0, -0.15, 0]), ("+z", [0, 0, 0.15]), ("-z", [0, 0, -0.20]),
                         ("home", [0, 0, 0])):
            tgt = demo.jog["origin"] + np.array(dv, float)
            demo.jog["target"] = tgt
            e = 1e9
            for _ in range(int(5.0 / demo.dt)):
                e = demo._jog_tick()
                if e < 0.005:
                    break
            got = demo.grip_pos()
            err = float(np.linalg.norm(got - tgt))
            worst = max(worst, err)
            q = demo.jog["q"]
            pred = demo._grip_fk(*q)
            print(f">>> JOG {name}: target {np.round(tgt, 3).tolist()} got {np.round(got, 3).tolist()} "
                  f"err {err * 1000:.1f}mm  q=[h1 {q[0]:.3f} h2 {q[1]:.3f} a1 {q[2]:.3f} th {q[3]:.3f}] "
                  f"pred {np.round(pred, 3).tolist()} fkerr {np.linalg.norm(pred - got) * 1000:.0f}mm", flush=True)
        ok = np.all(np.isfinite(np.asarray(demo.robot.get_joint_positions(), float)))
        print(f">>> JOG_TEST done: worst err {worst * 1000:.1f}mm, finite={bool(ok)}", flush=True)
        return

    if HEADLESS:                                             # no UI -> run the demo cycle(s) and exit.
        # DEMO_CYCLES="0:0,3:4,7:8" runs several obj:slot cycles in one boot; else
        # DEMO_OBJ/DEMO_SLOT.
        spec = os.environ.get("DEMO_CYCLES")
        if spec:
            pairs = [tuple(int(x) for x in p.split(":")) for p in spec.split(",")]
        else:
            pairs = [(int(os.environ.get("DEMO_OBJ", "0")), int(os.environ.get("DEMO_SLOT", "0")))]
        for oi, si in pairs:
            print(f">>> ===== CYCLE obj {oi} -> slot {si} =====", flush=True)
            demo.run_cycle(oi, si, status=lambda m: print(f">>> [{m}]", flush=True))
            for _ in range(int(0.5 / demo.dt)):
                demo.idle_step()
            bx, by, _ = demo.base_pose()
            print(f">>> post-idle base=({bx:.2f},{by:.2f})", flush=True)
        return

    state = {"busy": False, "request": None, "status_fn": None}
    build_menu(demo, state)
    status = lambda m: state["status_fn"](m) if state["status_fn"] else None
    print(">>> menu ready — pick an object + slot, click MOVE. Close window to exit.", flush=True)
    while simulation_app.is_running():
        if state["request"] is not None and not state["busy"]:
            state["busy"] = True
            oi, si = state["request"]
            state["request"] = None
            demo.run_cycle(oi, si, status=status)
            state["busy"] = False
        elif demo.jog is not None:
            demo._jog_tick()                                 # slider-driven cartesian arm servo
        else:
            demo.idle_step()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Print before close(): fastShutdown kills the process inside it, losing the traceback.
        import traceback
        traceback.print_exc()
        import sys as _sys
        _sys.stdout.flush()
        _sys.stderr.flush()
    finally:
        simulation_app.close()
