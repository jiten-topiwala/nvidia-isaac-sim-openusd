"""Entry point: menu-driven pick and place in Isaac Sim, run as `"$ISAAC/python.sh" play_isaac.py`.
Owns only the bootstrap: SimulationApp, `Demo` composed from the mixins in `morph/`, and `main()`.
Env: ISAAC_HEADLESS=1 (one cycle, no window), DEMO_CYCLES="3:6" obj:slot list, SEED=1 scatter seed."""
import json
import os
import time

from isaacsim import SimulationApp

# Safe to import before SimulationApp: config.py touches no Isaac API.
from morph.config import HEADLESS


# The wrong-regime decoy asks whether the capture caught the drive following its command, or
# something else. It is NOT a second tracking gate -- verify_place's track gate grades that.
WRONG_REGIME_POST_ERR_M = 2.0 * 7.5e-3


def _bench_metrics(stages, latencies, telemetry, total_cycle_time, abort_at_start, extra=None):
    """Return the instrumentation verdict while retaining the evidence behind it; `extra` carries
    the run metrics that are reported, not judged (slip, clearance, replans)."""
    metrics = {
        "stages": stages, "latencies": latencies, "telemetry": telemetry,
        "total_cycle_time": total_cycle_time, **(extra or {}),
    }
    named = ["_pick_setup", "_pick_pose", "_pick_reach", "_pick_settle", "_pick_descend",
             "_pick_close", "_pick_seat", "_pick_capture", "_pick_lift", "drive_to",
             "pre_dock_loop", "release", "backout", "retreat", "park", "raise", "slide",
             "lower"]
    durations = {name: duration for name, _, duration in stages}
    pre = telemetry.get("pre_step", [])
    post = telemetry.get("post_step", [])
    steps = telemetry.get("step_index", [])
    cmd_steps = telemetry.get("cmd_step", [])
    command_travel = sum(abs(step) for step in cmd_steps)
    response = sum(max(0.0, abs(command - before) - abs(command - after))
                   for (command, before), (_, after) in zip(pre, post))
    post_error = max((abs(command - actual) for command, actual in post),
                     default=float("inf"))

    fail = None
    if abort_at_start:
        fail = "SILENT ABORT DECOY"
    elif (not pre or len(pre) != len(post) or len(pre) != len(steps)
          or len(pre) != len(cmd_steps)):
        fail = "EVENT NOT CAPTURED"
    elif max(abs(command - actual) for command, actual in pre) < 7.0e-3:
        fail = "EVENT NOT CAPTURED"
    else:
        correct_regime = (
            telemetry.get("source") == "post-retreat"
            and telemetry.get("path") == "world.step"
            and telemetry.get("drive_on") is True
            and all(isinstance(before, int) and isinstance(after, int) and after > before
                    for before, after in steps)
            and all(abs(pre[k][0] - post[k][0]) <= 1e-12 for k in range(len(pre)))
            and all(abs(post[k][1] - pre[k + 1][1]) <= 1e-9
                    for k in range(len(pre) - 1))
            and command_travel > 0.0
            and response >= 0.9 * command_travel
            and post_error <= WRONG_REGIME_POST_ERR_M
        )
        if not correct_regime:
            fail = "SILENT WRONG-REGIME CAPTURE"
    if fail is None and total_cycle_time < 5.0:
        fail = "SILENT ABORT DECOY"
    if fail is None and (not all(name in durations for name in named)
                         or not all(duration > 0.0 for duration in durations.values())
                         or sum(durations.values()) < 0.8 * total_cycle_time):
        fail = "UNACCOUNTED TIME"
    # Each entry is (seconds, the child answered with an event record). The EVENT is the signal: a
    # real refusal is a 2 ms plan through the persistent worker, so a duration cannot say.
    if fail is None and (not latencies
                         or not all(len(entry) == 2 and entry[1] for entry in latencies)):
        fail = "WRONG WRAP DECOY"

    metrics.update({"captured": fail is None, "mode": fail})
    return metrics

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

# SHOW_COLLIDERS=1: draw every collider over the visuals, for a human check that the two agree.
# The planner only ever sees colliders, so a disagreement is a body it is not reasoning about.
if os.environ.get("SHOW_COLLIDERS") == "1":
    try:
        import carb
        _s = carb.settings.get_settings()
        _s.set("/persistent/physics/visualizationDisplayColliders", True)
        _s.set("/persistent/physics/visualizationSimulationOutput", True)
        print(">>> SHOW_COLLIDERS=1: collider wireframes drawn over the visuals", flush=True)
    except Exception as _e_sc:
        print(f">>> SHOW_COLLIDERS unavailable ({_e_sc})", flush=True)

# Mute harmless one-time startup warnings so the terminal shows only the >>> progress lines.
if os.environ.get("LOG_LEVEL") != "warning":
    try:
        import carb
        carb.settings.get_settings().set("/persistent/app/usd/muteUsdDiagnostics", True)
    except Exception:
        pass
    # silence the noisy per-plugin warning channels
    try:
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
from morph.arm.api import ArmMixin
from morph.close import CloseMixin
from morph.diagnostics import DiagnosticsMixin
from morph.gripper.api import FingersMixin
from morph.gui import build_menu
from morph.kinematics import KinematicsMixin
from morph.navigation import NavigationMixin
from morph.pick import PickMixin
from morph.place import PlaceMixin
from morph.robot import RobotMixin
from morph.world import WorldMixin
# SHELF_SLOTS is used below (the place-result target) and re-exported: smoke_test.py's
# PUBLIC_API asserts it stays exposed.
from morph.config import HERE, KNOWN, NUM_OBJECTS, SHELF_SLOTS, local_obj as _local_obj
from morph.geometry import quat_to_R, R_to_quat
from morph.usd_utils import find_path
from isaacsim.core.prims import RigidPrim, SingleXFormPrim
from isaacsim.core.utils.viewports import set_camera_view

# Startup defaults; Demo measures the real values in __init__. Module-level for the probes.
_PAD_ONLY = os.environ.get("PAD_ONLY", "0") == "1"
_PAD_FBD = os.environ.get("PAD_FBD", "0") == "1"       # per-point, per-link contact ledger     # read once; contact callback is a hot loop
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
# verify_place.py runs offline with no stage access, so this is the only place a placed object's
# real final xy/z gets measured. Default sits next to the default log (logs/live.log).
PLACE_RESULTS = os.environ.get("PLACE_RESULTS", os.path.join(HERE, "logs", "place_results.json"))


class Demo(PickMixin, PlaceMixin, CloseMixin, FingersMixin, AlignMixin, KinematicsMixin,
           NavigationMixin, ArmMixin, WorldMixin, RobotMixin, DiagnosticsMixin):
    """The demo robot. This class owns ALL the state; every mixin above only touches `self`.
    The mixins carry no state and no name collisions, so their order is documentation, not
    behaviour: roughly outermost phase to innermost primitive."""

    def __init__(self):
        # Records include _run_id so the verification gate only grades the current run.
        self._run_id = f"{os.getpid()}-{time.time():.0f}"
        # Safety abort latch persists across cycles; only process restart resets it.
        self._safety_abort = False
        # Truncate results on construct rather than module import to avoid deleting
        # prior evidence on bare imports. Guard against read-only mounts.
        if os.path.exists(PLACE_RESULTS):
            try:
                os.remove(PLACE_RESULTS)
            except OSError as e:
                print(f">>> could not truncate {PLACE_RESULTS}: {e}", flush=True)
        # Before any cycle can write a record, and unconditionally: verify_place has to be TOLD which
        # run's records are this run's evidence, or it grades whichever id the last record carries.
        self._write_run_manifest()
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
        # `_finger_pin`, when set, is the hand a running motion published for `_apply` to hold.
        self._finger_pin = None
        self._f1_mask = None
        # Both are invisible to _ctc/_fN, which require a pickup_obj on one side of the pair.
        self._ctc_floor = {}                                 # finger <-> floor contacts
        self._ctc_any = {}                                   # any partner of a hand link
        self.dt = self.world.get_physics_dt()
        self.stage = self.world.stage
        # per-object RigidPrim handles + the grip-material pass (friction to the PxShapes)
        self._apply_grip_material()
        # re-flow the material to the PxShapes...
        self.world.reset()
        self.robot.initialize()
        # world.reset() reverts the arm to the USD default and zeros the drive targets, so PARK, the
        # gains and q0 all have to be re-established or the heavy arm slams to zero.
        scene.set_arm_gains(self.robot, self.names)
        self.robot.set_joint_positions(self.q0)
        self.objs = [RigidPrim(f"/World/Geometry/pickup_obj_{i}",
                               prepare_contact_sensors=True,      # real contact evidence
                               track_contact_forces=True, max_contact_count=16)
                     for i in range(NUM_OBJECTS)]
        # base root height (holonomic drive keeps it)
        self.z0 = float(self.robot.get_world_pose()[0][2])
        # active finger command while carrying
        self.grip = None
        # Cached: find_path traverses the whole stage, so resolve the gripper link once.
        self._grip_prim = SingleXFormPrim(find_path(self.stage, "Gripper_Link1_1"))
        # object pose in the hand frame, at capture
        self._obj_local = None
        self.placed = []                                     # (obj_idx, world_pos) kept resting on the shelf
        # FIXED base pose held while idle (never re-read)
        self._idle_anchor = None
        # baked loop-consistent trajectories
        self.traj = self._load_trajs()
        # the recorded finger close
        self.close_traj = self._load_close_traj()
        # Closure LUT: (h2-h1, a1) -> passive linkage joints. This IS the arm's kinematic model, so a
        # swallowed load failure would silently demote every plan to bake replay. Let it raise at boot.
        self.clut = json.load(open(os.path.join(HERE, "usd/_closure_lut.json")))
        # active cartesian arm-jog state (GUI sliders)
        self.jog = None
        # measure the object, don't trust the json
        self._sync_object_size()
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
        # present now: scene built, scattered, parked
        scene.reveal_viewport(self.world, simulation_app)
        self._touch = {}
        self._fN_step = {}
        self._ctc_obj = {}
        self._ctcN = {}                                       # per-LINK accumulated impulse
        self._hit_step = False
        self._fN = {}
        self._fVec = {}                                      # per-link impulse VECTOR (world), for force closure
        self._fPos = {}                                      # last contact position per link, to orient it
        self._ctc = {}
        # Contact events double as the per-finger touch sensor: a yielding object never stalls
        # the finger pushing it, so drive lag cannot detect contact.
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools
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
                # It must be THE object being grasped: `_fN`/`_fN_step` are the force sensor every
                # close stage servos on, and a brush against a bystander would read as pad load.
                _fo = getattr(self, "_focus_obj", None)
                if _fo is not None and _fo not in a0 and _fo not in a1:
                    continue
                other = a0 if "pickup_obj" in a1 else a1
                imp = 0.0
                # Impulse VECTOR too: force closure lives in the directions (opposed normals), so
                # the geometry report needs the sum vector and a contact position to orient it by.
                impv, lastp = np.zeros(3), None
                # per-contact impulses = REAL pad load
                try:
                    for ci in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                        cd = contact_data[ci]
                        _iv = np.array([cd.impulse.x, cd.impulse.y, cd.impulse.z], float)
                        imp += float(np.linalg.norm(_iv))
                        impv += _iv
                        # PAD_FBD: per-POINT decomposition against PhysX's contact normal, keyed by LINK. n
                        # is along the normal (push-positive), t the remainder, nz its vertical part.
                        if _PAD_FBD:
                            _nv = np.array([cd.normal.x, cd.normal.y, cd.normal.z], float)
                            _nn = float(np.linalg.norm(_nv))
                            if _nn > 1e-9:
                                _nv = _nv / _nn
                                _n = float(np.dot(_iv, _nv))
                                if _n < 0.0:
                                    _n, _nv = -_n, -_nv
                                _t = float(np.linalg.norm(_iv - _n * _nv))
                                _lk = other.rsplit("/", 1)[-1]
                                # callback can fire before run_cycle
                                if not hasattr(self, "_fLink"):
                                    self._fLink = {}
                                _acc = self._fLink.setdefault(_lk, {"n": 0.0, "t": 0.0, "pts": 0,
                                                                    "nz": 0.0, "fz": 0.0})
                                _acc["n"] += _n; _acc["t"] += _t; _acc["pts"] += 1
                                _acc["nz"] += _n * float(_nv[2]); _acc["fz"] += float(_iv[2])
                        lastp = (cd.position.x, cd.position.y, cd.position.z)
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
                    # PAD_ONLY=1: only the distal link counts as finger load, or the close latches on the
                    # PROXIMAL link. `_hit_step` stays any-link: the seat stops on any kiss.
                    if (f"finger_{fl}_link" in other
                            and (not _PAD_ONLY or "_link_3_" in other)):
                        self._touch[fl] = True
                        self._fN[fl] = self._fN.get(fl, 0.0) + imp
                        self._fVec[fl] = self._fVec.get(fl, np.zeros(3)) + impv
                        if lastp is not None:
                            self._fPos[fl] = lastp
                        # Per-step, for the force-feedback loops: `_fN` above accumulates over a
                        # whole phase, which a closed loop cannot use.
                        self._fN_step[fl] = self._fN_step.get(fl, 0.0) + imp
                if "Gripper_Link3_1" in other and "finger" not in other:
                    self._touch["palm"] = True
                    self._fN["palm"] = self._fN.get("palm", 0.0) + imp
                    self._fVec["palm"] = self._fVec.get("palm", np.zeros(3)) + impv
                    if lastp is not None:
                        self._fPos["palm"] = lastp
                    # The real contact position. A nearest-hull-vertex proxy can report a vertex
                    # outside the surface, i.e. touching nothing.
                    try:
                        cd0 = contact_data[h.contact_data_offset]
                        self._palm_ct = (np.array([cd0.position.x, cd0.position.y,
                                                   cd0.position.z], float), other)
                    except Exception:
                        pass
                if "floor" not in other and "robot" in other:
                    # ANY robot part touching the object
                    self._hit_step = True

        self._contact_sub = get_physx_simulation_interface().subscribe_contact_report_events(_on_contact)
        print(">>> contact-event touch sensor ON", flush=True)
























    # Aliases onto morph/geometry.py, so the call sites keep working without a second copy.
    _quat_to_R = staticmethod(quat_to_R)
    _R_to_quat = staticmethod(R_to_quat)



















































    def run_cycle(self, obj_idx, slot_idx, status=None):
        self._safety_abort_at_start = getattr(self, "_safety_abort", False)
        # BEFORE `_open_place_result`: a refused cycle must stamp no record and print no grade
        # line, or results_gate's record-count-vs-grade-line equality breaks on every aborted run.
        if self._safety_abort and not getattr(self, "_abort_recovered", False):
            print(f">>> CYCLE obj {obj_idx} -> slot {slot_idx} REFUSED: a checkpoint SAFETY ABORT "
                  f"is latched -- the run halts; restart the process to clear it", flush=True)
            if status:
                status(f"refused: safety abort latched (obj {obj_idx} -> slot {slot_idx})")
            return False
        import time; _cycle_t0 = time.time()
        if getattr(self, "_abort_recovered", False):
            # Only a recovered abort can leave the pin held into the next cycle; an ordinary cycle
            # releases it on its own path, and unpinning those is an unvalidated behaviour change.
            self.pin_chassis(False, "abort-recovery-reset")
        self._abort_recovered = False
        # cycle moves the base; idle re-anchors after
        self._idle_anchor = None
        # ...and the park ramp runs once per cycle
        self._parked_ramped = False
        # telemetry keeps this cycle's first park ramp
        self._park_telemetry = None
        # any active arm jog ends when a cycle starts
        self.jog = None
        # no object owned at cycle start
        self._finger_cmd = None
        # ...and no motion is holding one
        self._finger_pin = None
        self._fLink = {}                                     # PAD_FBD per-link ledger
        try:
            self._open_place_result(obj_idx, slot_idx)
        except Exception as e:                               # an unwritable path must not kill the cycle
            # Open failure leaves no record on disk; record fallback for gating.
            self._fallback("place-results-open-failed")
            print(f">>> place-result open failed: {e}", flush=True)

        # Cycle-entry state stamp: `_recover` returns early whenever the base pose is sane, so gains and
        # residual velocities survive into the next cycle. Print them rather than assume them.
        try:
            _kp, _kd = self.ctrl.get_gains()
            _kp = np.asarray(_kp, float).reshape(-1); _kd = np.asarray(_kd, float).reshape(-1)
            _qv = np.asarray(self.robot.get_joint_velocities(), float).reshape(-1)
            _ii = self._arm_ii
            # ARM indices only. Taken over every joint this is ~2000 rad/s on cycle 1, because the
            # free mecanum rollers spin -- a number that says nothing about carried-over arm state.
            _qva = _qv[_ii]
            print(f">>> cycle-entry[obj {obj_idx}]: arm kp sum {float(_kp[_ii].sum()):.4g} kd sum "
                  f"{float(_kd[_ii].sum()):.4g} | arm |qv| {float(np.linalg.norm(_qva)):.4f} max "
                  f"{float(np.abs(_qva).max()):.4f} | drv_prev "
                  f"{'stale' if getattr(self, '_drv_prev', None) is not None else 'clear'} | brake "
                  f"{'ON' if getattr(self, '_pin_brake', False) else 'off'}", flush=True)
        except Exception as _e:
            print(f">>> cycle-entry[obj {obj_idx}]: unavailable ({_e})", flush=True)
        # STATELESS PER CYCLE: without this, the arm's gains and residual velocity from cycle N-1 are
        # the starting condition of cycle N -- the one carrier that makes a cycle differ when repeated.
        try:
            if getattr(self, "_gains_cycle0", None) is None:
                _k0, _d0 = self.ctrl.get_gains()
                self._gains_cycle0 = (np.asarray(_k0, float).reshape(-1).copy(),
                                      np.asarray(_d0, float).reshape(-1).copy())
            else:
                self.ctrl.set_gains(*(a.copy() for a in self._gains_cycle0))
                self.robot.set_joint_velocities(np.zeros(len(self._arm_ii)),
                                                joint_indices=self._arm_ii)
                # the reset invalidates the drive feedforward
                self._drv_prev = None
                print(">>> cycle-entry: arm gains and velocities restored to the cycle-0 baseline",
                      flush=True)
        except Exception as _e:
            print(f">>> cycle-entry: baseline restore failed ({_e})", flush=True)
        try:
            # start each cycle from a sane robot pose
            self._recover()
            held = self.pick(obj_idx, status=status)
            if not held and self._safety_abort:
                # The retry is an IDENTICAL second attempt, so it re-commands the configuration a checkpoint
                # rejected -- and a successful replan has already flown it by the time this returns True.
                self._fallback("retry-suppressed-after-abort")
                print(f">>> pick obj {obj_idx} failed after a checkpoint SAFETY ABORT -> retry "
                      f"SUPPRESSED (an identical attempt re-commands the rejected configuration)",
                      flush=True)
            elif not held and os.environ.get("PICK_RETRY", "1") == "1":
                # Residual forced-linkage disturbance is nondeterministic: recover and retry once.
                print(f">>> pick obj {obj_idx} failed -> RECOVER + retry", flush=True)
                # Close the books on attempt 1 first: without this the retry's degrades are added
                # to the first attempt's and the cycle reports one merged number.
                self._fallback_tally(obj_idx, attempt=1)
                self._recover()
                held = self.pick(obj_idx, status=status)
            if held:
                self.place(obj_idx, slot_idx, status=status)
            if status:
                status(f"done: obj {obj_idx} -> slot {slot_idx}")
        except Exception as e:
            import traceback
            print(f"!! cycle failed: {e}\n{traceback.format_exc()}", flush=True)
            if status:
                status(f"error: {e}")
            # a dropped articulation view (headed-mode USD
            try:
                # churn) otherwise bricks every later apply_action
                self.robot.initialize()
                self.ctrl = self.robot.get_articulation_controller()
                scene.set_arm_gains(self.robot, self.names)
                print(">>> articulation view re-initialized", flush=True)
            except Exception:
                pass
        # and leave it sane for the next one
        self._recover()
        # Write before tallying so any fallback recorded by _write_place_result
        # is included in the current cycle's tally before clearing.
        try:
            self._write_place_result(obj_idx, slot_idx)
        except Exception as e:
            # Guard against filesystem or pose exceptions unwinding past caller.
            print(f">>> place-result write failed: {e}", flush=True)
        self._fallback_tally(obj_idx)
        self._total_cycle_time = time.time() - _cycle_t0
        # the outcome both schedulers stop on
        return not self._safety_abort

    def _write_run_manifest(self):
        """Name THIS run, beside PLACE_RESULTS, before cycle 1. verify_place.py fails any record
        that does not carry this id -- that is what stops a previous run's records passing as this
        run's when a cycle is graded and then killed before its own record lands."""
        path = os.environ.get("PLACE_RUN_MANIFEST") or PLACE_RESULTS + ".run"
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tmp = path + ".tmp"                              # same write-then-rename as the records
            with open(tmp, "w") as f:
                json.dump({"run": self._run_id,
                           "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
            os.replace(tmp, path)
        except Exception as e:
            # Whatever is already on disk is the PREVIOUS run's manifest and records, and they agree with
            # each other. A run that cannot name itself must leave nothing gradeable behind.
            for stale in (path, PLACE_RESULTS):
                try:
                    os.remove(stale)
                except OSError:
                    pass
            # no manifest means an ungradeable run, so this is counted, not merely printed
            self._fallback("place-results-manifest-failed")
            print(f">>> RUN MANIFEST WRITE FAILED: {path}: {e} -- attempted to remove any stale "
                  f"manifest and results file; verify_place cannot grade this run", flush=True)

    def _place_records(self):
        """PLACE_RESULTS as a list. Non-list JSON resets to empty list so subsequent cycles do not fail."""
        if not os.path.exists(PLACE_RESULTS):
            return []
        try:
            with open(PLACE_RESULTS) as f:
                records = json.load(f)
            if not isinstance(records, list):
                raise ValueError("PLACE_RESULTS is not a list")
            return records
        except Exception:
            # Discarding prior measurements degrades acceptance evidence; track with counter.
            self._fallback("place-results-reset")
            return []

    def _dump_place_records(self, records):
        # Atomic write-and-rename prevents corruption if killed mid-dump.
        os.makedirs(os.path.dirname(PLACE_RESULTS) or ".", exist_ok=True)
        tmp = PLACE_RESULTS + ".tmp"
        with open(tmp, "w") as f:
            json.dump(records, f, indent=2)
        os.replace(tmp, PLACE_RESULTS)

    def _open_place_result(self, obj_idx, slot_idx):
        """Pre-record cycle execution state before pick so crashes leave an ok:false record."""
        # Record the baseline placed count before the dump, so write_place_result checks only
        # this cycle.
        self._placed_at_open = len(self.placed)
        records = self._place_records()
        records.append({"obj": obj_idx, "slot": slot_idx, "ok": False, "attempted": True,
                        "run": self._run_id})
        self._dump_place_records(records)

    def _write_place_result(self, obj_idx, slot_idx):
        """The one measurement verify_place.py cannot scrape from a print: re-read the placed
        object's pose straight from the stage -- not the value morph/place already computed -- and
        hand it to PLACE_RESULTS as data."""
        # `self.placed` is append-only, so only the tail since `_open_place_result` is this cycle's:
        # matching the whole list lets a repeat overwrite an earlier cycle's success.
        if not any(oi == obj_idx for oi, _ in self.placed[getattr(self, "_placed_at_open", 0):]):
            # nothing landed: this cycle's ok:false stamp stands
            return
        pos = np.asarray(self.objs[obj_idx].get_world_poses()[0][0], float)
        slot = SHELF_SLOTS[slot_idx]
        rec = {"obj": obj_idx, "slot": slot_idx, "ok": True, "run": self._run_id,
               "xy": [float(pos[0]), float(pos[1])], "z": float(pos[2]),
               "target_xy": [float(slot[0]), float(slot[1])],
               "target_z": float(slot[2] + self.obj_half_h)}
        records = self._place_records()
        mine = [i for i, r in enumerate(records) if isinstance(r, dict)
                and r.get("obj") == obj_idx and r.get("slot") == slot_idx]
        if mine:
            # UPDATE this cycle's stamp: two records for one
            records[mine[-1]] = rec
        else:                                                 # cycle would trip the gate's own record count
            records.append(rec)
        self._dump_place_records(records)

    def _fallback_tally(self, obj_idx, attempt=None):
        """Print and clear the silent-degrade counter. Called at cycle end, and again between any
        two attempts -- `run_cycle` before its retry, `pick` between candidates -- so no two
        attempts ever merge into one number.

        Lives HERE, not in the headless loop: a GUI run printed no line at all, and verify_place's
        gate is `all()` over the matches -- vacuously true with zero lines, so a GUI run silently
        PASSED the fallback gate."""
        _fb = getattr(self, "_plan_fallbacks", None) or {}
        _tot = sum(_fb.values())
        _parts = ", ".join(f"{k}={v}" for k, v in sorted(_fb.items()))
        _who = f"obj {obj_idx}" + (f" attempt {attempt}" if attempt is not None else "")
        print(f">>> plan-fallbacks[{_who}]: {_tot}" + (f" ({_parts})" if _tot else ""), flush=True)
        self._plan_fallbacks = {}





def main():
    demo = Demo()
    grip_c = demo.grip_pos()
    # Framed once in scene.load_scene; re-issuing here makes the view hop on start-up.

    # no UI -> run the demo cycle(s) and exit.
    if HEADLESS:
        # DEMO_CYCLES="0:0,3:4,7:8" runs several obj:slot cycles in one boot; else
        # DEMO_OBJ/DEMO_SLOT.
        spec = os.environ.get("DEMO_CYCLES")
        if spec:
            pairs = [tuple(int(x) for x in p.split(":")) for p in spec.split(",")]
        else:
            pairs = [(int(os.environ.get("DEMO_OBJ", "0")), int(os.environ.get("DEMO_SLOT", "0")))]
        # PROFILE_OUT=<path>: in-process cProfile, dumped before Kit's fastShutdown skips atexit.
        _prof = None
        if os.environ.get("PROFILE_OUT"):
            import cProfile
            _prof = cProfile.Profile()
            _prof.enable()
        for _k, (oi, si) in enumerate(pairs):
            print(f">>> ===== CYCLE obj {oi} -> slot {si} =====", flush=True)
            demo._safety_abort_at_start = getattr(demo, "_safety_abort", False)
            if not demo.run_cycle(oi, si, status=lambda m: print(f">>> [{m}]", flush=True)):
                # Run halts on safety abort; log skipped cycles explicitly.
                _skipped = pairs[_k + 1:]
                print(f">>> SAFETY ABORT latched -- scheduling STOPPED after obj {oi} -> slot "
                      f"{si}; {len(_skipped)} remaining cycle(s) SKIPPED: "
                      f"{', '.join(f'{a}:{b}' for a, b in _skipped) or '(none)'}", flush=True)
                break
            for _ in range(int(0.5 / demo.dt)):
                demo.idle_step()
            bx, by, _ = demo.base_pose()
            print(f">>> post-idle base=({bx:.2f},{by:.2f})", flush=True)
        if _prof is not None:
            _prof.disable()
            _prof.dump_stats(os.environ["PROFILE_OUT"])
            print(f">>> profile written: {os.environ['PROFILE_OUT']}", flush=True)
        if getattr(demo, "_bench_stages", None) is not None:
            import json, sys
            _stages = demo._bench_stages
            _lats = getattr(demo, "_bench_latencies", [])
            _tele = getattr(demo, "_park_telemetry", None) or {"pre_step": [], "post_step": []}
            _total = getattr(demo, "_total_cycle_time", 0.0)

            _clr = getattr(demo, "_bench_clearance", [])
            _metrics = _bench_metrics(
                _stages, _lats, _tele, _total,
                getattr(demo, "_safety_abort_at_start", False),
                {"slip": getattr(demo, "_bench_slip", []), "clearance": _clr,
                 "clearance_min_m": min((c[2] for c in _clr), default=None),
                 "replans": getattr(demo, "_bench_replans", 0)})
            _fail = _metrics["mode"]
            # A missing directory here kills a COMPLETED run at exit, and /tmp is cleared at boot.
            # Overridable like PLACE_RESULTS, so a run can keep it beside its own log.
            _mpath = os.environ.get("BENCH_METRICS", "/tmp/bench/metrics.json")
            os.makedirs(os.path.dirname(_mpath) or ".", exist_ok=True)
            with open(_mpath, "w") as f:
                json.dump(_metrics, f)
            if _fail:
                print(f"FAILED: {_fail}", flush=True)
                sys.exit(1)

        return

    state = {"busy": False, "request": None, "status_fn": None}
    build_menu(demo, state)
    status = lambda m: state["status_fn"](m) if state["status_fn"] else None
    print(">>> menu ready — pick an object + slot, click MOVE. Close window to exit.", flush=True)
    while simulation_app.is_running():
        if demo._safety_abort:
            # Halt completely on safety abort: idle_step or jog would continue commanding
            # the rejected arm configuration.
            print(">>> SAFETY ABORT latched -- menu loop STOPPED; the jog servo and the idle "
                  "step both command the arm, so the window closes. Restart the run.", flush=True)
            break
        if state["request"] is not None and not state["busy"]:
            oi, si = state["request"]
            state["request"] = None
            if demo._safety_abort:
                # Visible, not silently dropped: an ignored click reads as a broken button.
                print(f">>> MOVE obj {oi} -> slot {si} REFUSED: a checkpoint SAFETY ABORT is "
                      f"latched -- restart the run", flush=True)
                status("REFUSED: safety abort latched -- restart the run to clear it")
            else:
                state["busy"] = True
                demo.run_cycle(oi, si, status=status)
                state["busy"] = False
        elif demo.jog is not None:
            # slider-driven cartesian arm servo
            demo._jog_tick()
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
