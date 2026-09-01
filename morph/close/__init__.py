"""The finger close — the grasp itself.

`_close_replay` is a sequence of env-gated stages that run in order against the live contact
state. Each stage reads the finger/object gaps and decides whether to advance, freeze or back
off, so they share state and run strictly in order:

    replay      the recorded MuJoCo close, per-finger freeze on contact   replay.py
    enclose     get the object inside the circle the fingers sweep        enclose.py
    align_eq    equalise the three surface gaps before closing            enclose.py
    wrap        the encompassing close (owns the grasp when it runs)      wrap.py
    legal       vendor legal-range close                                  legal.py
    seat_bc     seat the b/c jaw so the thumb has something to react on   seat_bc.py
    grip_seat   per-finger geometric seat + wrap + squeeze                seat.py, seat_wrap.py

Everything a stage needs from the stage before it lives on `_CloseState`; anything else is
local to one stage. That is the whole contract — a stage may not reach into another's locals.
"""
import os

import numpy as np

from morph.config import PERFECT
from morph.close.approach_align import ApproachAlignStage
from morph.close.enclose import EncloseStage
from morph.close.joint_servo import JointServoStage
from morph.close.lowest import LowestStage
from morph.close.force_balance import ForceBalanceStage
from morph.close.legal import LegalStage
from morph.close.replay import ReplayStage
from morph.close.seat import SeatStage
from morph.close.seat_bc import SeatBcStage
from morph.close.seat_wrap import SeatWrapStage
from morph.close.wrap import WrapStage

__all__ = ["CloseMixin"]


class _CloseState:
    """The locals that cross stage boundaries. Deliberately tiny — if a name is not here, it
    belongs to exactly one stage and must not be read by another."""

    __slots__ = ("q_base", "secs", "obj", "status", "q", "bx", "by", "byaw", "lj_locked",
                 "jaw_axis", "fb_held")

    def __init__(self, q_base, secs, obj, status):
        self.q_base, self.secs, self.obj, self.status = q_base, secs, obj, status
        self.q = q_base                      # replaced by the replay stage with its own copy
        self.fb_held = set()                 # fingers force-close latched (3 = hold is final)
        self.bx = self.by = self.byaw = 0.0  # set by the replay stage from the base ledger
        self.lj_locked = False               # set by wrap/legal; gates every stage below them
        self.jaw_axis = None                 # measured by the geometry report, used by the legal close


class CloseMixin(ReplayStage, EncloseStage, ApproachAlignStage, LowestStage, JointServoStage,
                 WrapStage, ForceBalanceStage, LegalStage, SeatBcStage,
                 SeatStage, SeatWrapStage):
    """Mixed into Demo. Every `self.*` it touches is owned by Demo, not by this class."""

    def _close_sync_arm(self, st, tag):
        """Re-read the ARM half of the command array, and the base ledger, from the live robot.

        `st.q` is the command every stage writes through `_force`/`_apply`, but the two jog stages
        (ENCLOSE and ALIGN_EQ) drive the arm through `_jog_write` instead. After either of them
        `st.q` still holds the pre-jog arm pose, so the next `self._force(st.q)` would teleport the
        arm back there in a single kinematic step and undo the alignment.

        Only the arm indices are synced. The finger command must not be rebuilt from measurement:
        `command := actual` means zero steady-state error, and therefore zero grip force.
        """
        qa = np.asarray(self.robot.get_joint_positions(), float)
        st.q[self._arm_ii] = qa[self._arm_ii]
        st.bx, st.by, st.byaw = self.base_ledger()
        if os.environ.get("CLOSE_TRACE", "0") == "1":
            print(f">>>   close: arm command re-synced after {tag}", flush=True)

    def pin_chassis(self, on, tag=""):
        """Hold the chassis for the whole grasp, not just one stage.

        The dock deliberately arrives with momentum, so the robot is still rolling when the grasp
        begins — the wrist travels 6-10mm per step and drags the object with it. The chassis is a
        floating root body, so zeroing the wheel joints is not enough; the root pose itself has to
        be re-asserted.

        Stages call `self.world.step` from many places, so the pin wraps that one call for its
        lifetime rather than being threaded through every loop.
        """
        if on:
            if getattr(self, "_pin_orig_step", None) is not None:
                return
            self._pin_ii = np.asarray([i for n, i in self.idx.items()
                                       if ("wheel" in n and "rolling" in n) or "slipping" in n])
            _q = np.asarray(self.robot.get_joint_positions(), float)
            self._pin_qw = _q[self._pin_ii].copy() if len(self._pin_ii) else None
            self._pin_base = self.base_ledger()
            # Capture the height too. `_pin_base` is xy/yaw only, and `set_base`'s 20mm z slack
            # keeps whatever height physics produced -- so the arm's reaction could lift the chassis
            # and the pin would hold it there.
            try:
                self._pin_z = float(self.robot.get_world_pose()[0][2])
            except Exception:
                self._pin_z = None
            self._pin_orig_step = self.world.step

            # Re-assert on drift, not every step. `set_base` teleports the articulation root, and a
            # root teleport rebuilds the articulation state: the stiff j1 drive recovers each step
            # but the weak distal drives never accumulate motion, which freezes j2/j3 on their
            # limits for the whole close.
            _pin_tol = 0.001

            def _pinned_step(*a, **kw):
                bx, by, byaw = self._pin_base
                _drift = _pin_tol + 1.0
                try:
                    _p = np.asarray(self.robot.get_world_pose()[0], float)
                    _drift = float(np.hypot(_p[0] - bx, _p[1] - by))
                except Exception:
                    pass
                if _drift > _pin_tol:
                    self.set_base(bx, by, byaw)
                    self._pin_hits = getattr(self, "_pin_hits", 0) + 1
                else:
                    try:                                     # anti-coast without the teleport
                        self.robot.set_velocities(np.zeros((1, 6)))
                    except Exception:
                        pass
                if self._pin_qw is not None:
                    self.robot.set_joint_velocities(np.zeros(len(self._pin_ii)),
                                                    joint_indices=self._pin_ii)
                return self._pin_orig_step(*a, **kw)

            # ...and lock it. The pin puts the base back after a stage has moved it; the lock
            # refuses the move outright, so the grasp never fights the chassis in the first place.
            if os.environ.get("NO_CHASSIS_GRASP", "1") == "1" and PERFECT:
                self._grasp_locked = tuple(self._pin_base)
                self._base_block_n = {}
            self.world.step = _pinned_step
            print(f">>> chassis PINNED{(' ' + tag) if tag else ''}: root held at "
                  f"({self._pin_base[0]:.3f}, {self._pin_base[1]:.3f}) and "
                  f"{len(self._pin_ii)} wheel joints zeroed for the whole grasp", flush=True)
        else:
            if getattr(self, "_pin_orig_step", None) is not None:
                self.world.step = self._pin_orig_step
                self._pin_orig_step = None
                self._pin_z = None          # z goes back to the slack rule once unpinned
                _blk = getattr(self, "_base_block_n", {})
                print(f">>> chassis released{(' ' + tag) if tag else ''} "
                      f"({getattr(self, '_pin_hits', 0)} drift re-asserts"
                      + (f", REFUSED {sum(_blk.values())} base moves from "
                         f"{ {k: v for k, v in sorted(_blk.items(), key=lambda kv: -kv[1])} }"
                         if _blk else "") + ")", flush=True)
                self._pin_hits = 0
                self._grasp_locked = None
                self._base_block_n = {}

    def _drive_stamp(self, tag):
        """One line: the finger drive target the articulation actually holds.

        Read back from the articulation controller rather than from `_finger_cmd`, which is this
        module's own bookkeeping and can diverge from what was applied.
        """
        try:
            _t = np.asarray(self.robot.get_articulation_controller().get_applied_action()
                            .joint_positions, float).reshape(-1)
            _b = []
            for f in "abc":
                i = self.idx.get(f"finger_{f}_joint_1_1")
                if i is not None and i < len(_t) and _t[i] is not None:
                    _b.append(f"{f} {float(_t[i]):+.3f}")
            print(f">>>   drive[{tag}]: " + " ".join(_b), flush=True)
        except Exception as _e_ds:
            print(f">>>   drive[{tag}]: unavailable ({_e_ds})", flush=True)

    def _obj_watch(self, obj, tag, rest=None):
        """Object tilt, height and speed at a boundary — the ONE readout; `_obj_state` wraps it
        for the pick-stage call sites. axis.z 1.0 is perfectly upright; below ~0.99 it is
        already going over."""
        try:
            oc = np.asarray(obj.get_world_poses()[0][0], float)
            R = self._obj_R(obj)
            az = float(R[:, 2][2]) if R is not None else float("nan")
            v = float(np.linalg.norm(np.asarray(obj.get_velocities(), float).reshape(-1)[:3]))
            _r = f" (rest {rest:.3f})" if rest is not None else ""
            print(f">>>   obj[{tag}]: z {oc[2]:.3f}{_r} axis.z {az:+.4f} speed {v * 1000:.0f}mm/s"
                  + ("   <-- DISTURBED" if (az < 0.99 or v > 0.05) else ""), flush=True)
        except Exception as _e_ow:
            print(f">>>   obj[{tag}]: readout failed ({_e_ow})", flush=True)

    def _close_replay(self, q_base, secs, obj, status=None):
        """Run the close stages in order and return the final joint command."""
        st = _CloseState(q_base, secs, obj, status)
        # Diagnostic only, off by default: place the object at the pinch kinematically so the grasp
        # can be evaluated independently of the approach that has to deliver it there.
        if os.environ.get("GRASP_TELEPORT", "0") == "1":
            try:
                _p = self._pinch()
                _oc = np.asarray(obj.get_world_poses()[0][0], float)
                obj.set_world_poses(
                    positions=np.array([[float(_p[0]), float(_p[1]), float(_oc[2])]]),
                    orientations=np.array([[1.0, 0.0, 0.0, 0.0]]))
                for _ in range(30):
                    self.world.step(render=False)
                _oc2 = np.asarray(obj.get_world_poses()[0][0], float)
                print(f">>> GRASP_TELEPORT: object moved to the pinch "
                      f"{np.round(_oc, 3).tolist()} -> {np.round(_oc2, 3).tolist()} "
                      f"(pinch->obj now {np.linalg.norm(_oc2 - self._pinch()) * 1000:.0f}mm)",
                      flush=True)
            except Exception as _e_tp:
                print(f">>> GRASP_TELEPORT failed ({_e_tp})", flush=True)
        # Measure the geometry before anything moves: the same report taken after the close
        # describes the outcome, not the approach that produced it.
        self._grasp_geometry_report(obj, tag="close-entry")
        # Refuse to close from a pose no downstream servo can recover. The reachable window is the
        # one `_grasp_geometry_report` prints (du -70..-130mm, |dv| <= 30mm); this gate is
        # deliberately wider, because the enclose servo exists to absorb moderate error.
        if PERFECT and os.environ.get("CLOSE_GATE", "1") == "1":
            _dd_g = self._du_dv(obj)
            _pz_g = float(self._pinch()[2])
            _oc_g = np.asarray(obj.get_world_poses()[0][0], float)
            _z_hi = float(_oc_g[2]) + self.obj_half_h + 0.03
            # The bound is what the close itself can survive, not what a later servo might fix. The
            # window edge is -130mm; 140 allows the close's own stagger tolerance, no more.
            _du_min = -0.14
            _dv_max = 0.05
            _bad = []
            if _dd_g is not None:
                if _dd_g[0] < _du_min:
                    _bad.append(f"du {_dd_g[0] * 1000:+.0f}mm < {_du_min * 1000:.0f}")
                if abs(_dd_g[1]) > _dv_max:
                    _bad.append(f"|dv| {abs(_dd_g[1]) * 1000:.0f}mm > {_dv_max * 1000:.0f}")
            if _pz_g > _z_hi:
                _bad.append(f"pinch z {_pz_g:.3f} above obj top+slack {_z_hi:.3f}")
            if _bad:
                print(f">>> CLOSE GATE: pose unreachable ({'; '.join(_bad)}) -- SKIPPING the "
                      f"close instead of thrashing on it; the pick fails clean and recovers",
                      flush=True)
                # Block the rest of the pick too: returning here only skips the close, and
                # capture/lift would otherwise run against no object at all.
                self._gate_blocked = True
                return st.q
        if os.environ.get("PIN_CHASSIS", "1") == "1" and PERFECT:
            self.pin_chassis(True, "close-entry")
        # The symmetric side-grip open is applied at the root: the baked reach trajectory and
        # KNOWN["fingers_open"] are both rewritten to GRIPPER_OPEN_POS, so the fingers arrive here
        # already symmetric and no re-symmetrise stage is needed.
        self._obj_watch(obj, "before-replay")
        # Enclose before the fingers close: the object has to be inside the circle the fingers
        # sweep, or the hand closes on air beside it.
        _am = os.environ.get("ALIGN_MODE", "center")
        _enc_first = PERFECT and os.environ.get("ENCLOSE_FIRST", "1") == "1" and _am != "center"
        if _enc_first:
            self._close_enclose(st)
            self._obj_watch(obj, "after-enclose")
            self._close_sync_arm(st, "enclose")
        # Direct synchronous close, the default in perfect mode. The recorded replay arc encodes a
        # large object's grip -- it curls the thumb through a long reach and barely moves b/c, so on
        # a thinner object only the thumb lands.
        _direct = (PERFECT and os.environ.get("CLOSE_DIRECT", "1") == "1"
                   and os.environ.get("FORCE_CLOSE", "0") == "1")
        if _direct:
            print(">>> close: DIRECT synchronous force-close (recorded-replay arc skipped; "
                  "entry is centered and in-window)", flush=True)
            self._set_arm_gain(3.0e5)
            if os.environ.get("CLOSE_GENTLE", "1") == "1":
                self._grip_gentle()
            self._finger_cmd = st.q[self._f_idx].copy()
            if not _enc_first and _am != "center":
                self._close_enclose(st)
                self._obj_watch(obj, "after-enclose")
                self._close_sync_arm(st, "enclose")
            # Align in the hand frame, not in the du/dv window: du/dv is a world-XY projection and
            # goes degenerate once the hand pitches down (mouth axis z ~ -0.98).
            if _am == "center":
                self._align_to_grasp_center(obj)     # size-independent: object -> finger-arc centre
                self._align_z_to_center(obj)         # then columns bring the pinch to object MID-height
            elif _am == "recorded":
                self._align_to_recorded(obj)
            else:
                self._close_align_eq(st)
            self._obj_watch(obj, "after-align-rec")
            self._close_sync_arm(st, "align-rec")
        else:
            self._close_do_replay(st)
            self._obj_watch(obj, "after-replay")
            if not _enc_first:
                self._close_enclose(st)
                self._obj_watch(obj, "after-enclose")
                self._close_sync_arm(st, "enclose")
            self._obj_watch(obj, "after-sync1")
            self._close_align_eq(st)
            self._obj_watch(obj, "after-align-eq")
            self._close_sync_arm(st, "align-eq")
            self._obj_watch(obj, "after-sync2")
            self._close_wrap(st)
            self._obj_watch(obj, "after-wrap")
        # Close on measured pad force, so a finger the wrap left in mid-air keeps coming instead of
        # leaving its opposite number to push the object alone. See force_balance.py.
        self._close_force_balance(st)
        # Last: take up the remaining gap symmetrically and preload. The force servo above is the
        # right tool for finding contact from an unknown pose and the wrong one for the final few
        # millimetres of an already-symmetric grasp, because it advances each finger independently
        # and so breaks the symmetry the stages before it established.
        self._close_sym_squeeze(st)
        # The drive stamps bracket each stage below, so a stage that re-opens the hand is
        # attributable rather than merely visible at close-return.
        self._drive_stamp("before-geom-report")
        st.jaw_axis = self._grasp_geometry_report(obj, tag="post-wrap")
        self._drive_stamp("after-geom-report")
        # A three-finger hold with overdrive is final. legal/seat_bc/grip_seat exist for the case
        # where the force close did NOT get a hold; run after a successful one they re-derive the
        # finger command from the measured pose, which sets command == actual and therefore drops
        # the squeeze -- the grip is handed away between capture and lift.
        _fb_ok = (PERFECT and os.environ.get("FB_SKIP_SEAT_ON_HOLD", "1") == "1"
                  and len(getattr(st, "fb_held", set())) == 3)
        if _fb_ok:
            print(">>> close: force-close holds all three with overdrive -- legal/seat_bc/grip_seat "
                  "SKIPPED (they would re-seat the command onto the measurement and drop the "
                  "squeeze)", flush=True)
        if not _fb_ok:
            self._close_legal(st)
        self._drive_stamp("after-legal")
        if not _fb_ok:
            self._close_seat_bc(st)
        self._drive_stamp("after-seat-bc")
        if not _fb_ok:
            self._close_grip_seat(st)
        self._drive_stamp("after-grip-seat")
        # Stamp the command on the way out, so what the close hands over is on the record.
        self._pad_trace("close-return")
        return st.q
