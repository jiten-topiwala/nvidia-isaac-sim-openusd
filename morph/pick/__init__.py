"""The pick cycle: a fixed, order-dependent sequence -- setup, pose, reach, settle, descend,
close, seat, capture, lift -- each stage reading what the one before left and either advancing
or setting `alive = False`. State crossing a stage boundary lives on `_PickState`."""
import os
from morph.config import HEADLESS, KNOWN
import numpy as np

from morph.pick.approach import ApproachStage
from morph.pick.close_anim import CloseAnimStage
from morph.pick.descend import DescendStage
from morph.pick.dock import DockStage
from morph.pick.grasp import GraspStage
from morph.pick.reach import ReachStage

__all__ = ["PickMixin"]


class _PickState:
    """The locals that cross stage boundaries; a name not here belongs to exactly one stage."""

    __slots__ = ("obj_idx", "status", "O", "psi", "dock", "qg", "obj", "park",
                 "replay", "pin_bystanders", "restore_bystanders",
                 "alive", "grip", "seat", "seat_gap", "floor_pos", "held", "_oc0")

    def __init__(self, obj_idx, status):
        self.obj_idx, self.status = obj_idx, status
        self.alive, self.held = True, False


class PickMixin(DockStage, ApproachStage, ReachStage, DescendStage,
                CloseAnimStage, GraspStage):
    """Mixed into Demo. Every `self.*` it touches is owned by Demo."""

    def _settle_hand(self, tag, tol=None, t_max=None):
        """Hold the current joint command until the pinch stops moving; returns the final
        per-step pinch speed (m/step)."""
        tol = 0.0002 if tol is None else tol
        n_max = int((float(os.environ.get("SETTLE_T", "2.5")) if t_max is None else t_max) / self.dt)
        q = np.asarray(self.robot.get_joint_positions(), float).copy()
        p_last = np.asarray(self._pinch(), float)
        v, k = 1e9, 0
        # Watch the object while the hand holds still: the hand is kinematically forced, so a palm
        # resting on the object is an infinite-stiffness wall that can drive it through the floor.
        _fo = getattr(self, "_focus_obj", None)
        _ow = self.objs[int(_fo.rsplit("_", 1)[-1])] if _fo else None
        _z0 = float(np.asarray(_ow.get_world_poses()[0][0], float)[2]) if _ow is not None else None
        _z_said = False
        # How close is the hand before the settle starts? A descent reporting zero contacts can
        # still finish INSIDE the contact offset, where the smallest motion makes one.
        if _ow is not None:
            try:
                _oc = np.asarray(_ow.get_world_poses()[0][0], float)
                _r = float(KNOWN["object_radius"])
                _hh = float(self.obj_half_h)
                _rep = {}
                for _ln in ("a", "b", "c", "palm"):
                    _pts = self._finger_world_pts(_ln)
                    _dr = np.hypot(_pts[:, 0] - _oc[0], _pts[:, 1] - _oc[1]) - _r
                    _dz = np.abs(_pts[:, 2] - _oc[2]) - _hh
                    _rep[_ln] = round(float(np.min(np.maximum(_dr, _dz))) * 1000, 1)
                print(f">>> settle[{tag}] CLEARANCE to the object surface before the hold: "
                      f"{_rep}mm  (PhysX contact offset is 20mm -- anything under that is already "
                      f"touching as far as the solver is concerned)", flush=True)
            except Exception as _e_cl:
                print(f">>> settle[{tag}] clearance unavailable ({_e_cl})", flush=True)
        for k in range(n_max):
            if _ow is not None and not _z_said:
                self._ctcN = {}
            # SETTLE_FORCE=0: a forced palm displaces the object with no contact force at all.
            if os.environ.get("SETTLE_FORCE", "1") == "1":
                self._force(q)
            self._apply(q)
            self.world.step(render=not HEADLESS)
            if _ow is not None and not _z_said:
                _zn = float(np.asarray(_ow.get_world_poses()[0][0], float)[2])
                if abs(_zn - _z0) > 0.005:
                    _hot = {kk: round(vv, 4) for kk, vv in getattr(self, "_ctcN", {}).items()
                            if vv > 1e-6}
                    print(f">>> settle[{tag}] OBJECT DEPARTS at step {k}: z {_z0:+.3f} -> "
                          f"{_zn:+.3f} ({(_zn - _z0) * 1000:+.1f}mm) | impulse THIS step {_hot}",
                          flush=True)
                    _z_said = True
            p_now = np.asarray(self._pinch(), float)
            v = float(np.linalg.norm(p_now - p_last))
            p_last = p_now
            if v < tol:
                break
        print(f">>> settle[{tag}]: {k + 1} steps -> {v * 1000:.3f}mm/step "
              f"(tol {tol * 1000:.2f})" + ("   <-- STILL MOVING" if v >= tol else ""), flush=True)
        return v

    def _obj_state(self, st, tag):
        """Log object tilt + height at a stage boundary -- names the stage that destabilises it."""
        obj = getattr(st, "obj", None)
        if obj is not None:
            self._obj_watch(obj, tag, rest=self.obj_half_h)

    def pick(self, obj_idx, status=None):
        """The nominal grasp, then one attempt per APPROACH-YAW candidate if it failed -- each a
        whole hand pose orbited about the object's vertical, already solved and already filtered
        through the same `collides` the planner calls.

        The WHOLE fan is walked: `grasp_candidates` carries no yaw term, so its order is a
        preference, never a correctness condition. A checkpoint SAFETY ABORT ends the walk."""
        _held = self._pick_attempt(obj_idx, status)
        if self._safety_abort:
            # BEFORE the success return: a settle abort does not clear `st.alive`, so the attempt
            # can report a grasp with the flag latched and `place` would open the insert with it.

            # a checkpoint rejected a configuration: the run halts
            return False
        if _held:
            return True
        if os.environ.get("GRASP_FAN", "0") != "1":
            # OFF by default: the fan is solved in the arm-root frame as it stands NOW, but the attempt
            # re-docks before the candidate is commanded, so a base-relative q8 lands elsewhere.
            return False
        if not self._cand_usable:
            # `_plan_approach` is the only reader of `_grasp_cand`, and only an attempt that reached the
            # reach ran it -- so after a refusal the walk would command candidates nothing reads.
            print(f">>> PICK obj {obj_idx}: the attempt was refused before the reach, "
                  f"so a candidate would be discarded unread -> no walk", flush=True)
            return False
        # the failed attempt may have shoved the object; the
        self._recover()
        cands = self._grasp_candidates(obj_idx)   # fan must orbit where the candidates will find it
        for k, q8 in enumerate(cands, 1):
            # Close the books on the attempt that just ended, as `run_cycle` does before its retry.
            # Without it five attempts' degrades merge into the one cycle-end number.
            self._fallback_tally(obj_idx, attempt="nominal" if k == 1 else f"candidate {k - 1}")
            print(f">>> PICK obj {obj_idx}: nominal grasp failed -> approach candidate {k}",
                  flush=True)
            self._grasp_cand = q8
            try:
                held = self._pick_attempt(obj_idx, status)
            finally:
                # the NEXT nominal must solve its own goal again
                self._grasp_cand = None
            if self._safety_abort:
                # BEFORE the success return, exactly as on the nominal path above: a candidate can latch the
                # flag in its settle and still come back holding the object.
                print(f">>> PICK obj {obj_idx}: candidate {k} hit a checkpoint SAFETY ABORT -> "
                      f"walk ENDS (the run halts; the rejected configuration is never re-commanded)",
                      flush=True)
                break
            if held:
                return True
            # the next candidate starts sane, as the retry does
            self._recover()
        return False

    def _release_chassis(self, why):
        """`pin_chassis(True)` monkey-patches `world.step`; every exit past the dock pin comes here."""
        try:
            self.pin_chassis(False, why)
        except Exception:                      # noqa: BLE001 -- a release must not mask the exit
            pass

    def _pick_attempt(self, obj_idx, status=None):
        """One pass of the fixed sequence: nav to a standoff facing the object, force the known arm
        config, seat the object in the finger cage, close + pin.  Returns True on grasp.

        `self._grasp_cand`, when `pick` has set it, is the grasp configuration `_plan_approach`
        aims the reach at instead of solving the nominal one."""
        # close-gate veto from a PREVIOUS attempt must not leak
        self._gate_blocked = False
        # ...and only THIS attempt's planned grasp seeds the fan
        self._grasp_goal = None
        for _a in ("_cw_z0", "_cw_xy0", "_cw_said", "_cw_broke"):
            # close-watch state is PER PICK: stale, its latch silences the probe after cycle one
            self.__dict__.pop(_a, None)
        st = _PickState(obj_idx, status)
        # Only the reach consumes a candidate; until it runs, `pick` must not walk the fan.
        self._cand_usable = False
        with self._time_stage('_pick_setup'):
            self._pick_setup(st)
        if not st.alive:
            print(f">>> PICK obj {obj_idx}: held=False (no dock corridor)", flush=True)
            return False
        # Stability trace: the close can receive an object that is already falling, so the
        # instability arrives UPSTREAM of every close-side conclusion.
        self._obj_state(st, "after-setup")
        with self._time_stage('_pick_pose'):
            self._pick_pose(st)
        if not st.alive:
            self._release_chassis("no-bake")
            print(f">>> PICK obj {obj_idx}: held=False (no baked trajectory)", flush=True)
            return False
        self._obj_state(st, "after-pose")
        try:
            _axz = float(self._obj_R(st.obj)[:, 2][2])
        except Exception:
            _axz = 1.0
        if _axz < 0.9:
            # A fallen cylinder is not pickable by this grasp, and planning to its rest height
            # runs the descent fallback past the boom limit. Fail clean.
            print(f">>> PICK obj {obj_idx}: object is lying on its side (axis.z {_axz:+.2f}) -> no pick", flush=True)
            st.restore_bystanders()
            self._release_chassis("fallen-object")
            return False
        self._cand_usable = True
        with self._time_stage('_pick_reach'):
            self._pick_reach(st)
        self._obj_state(st, "after-reach")
        # Gated, not sequential: a checkpoint abort leaves the arm at an intermediate pose, and the
        # settle's `_hold` would command the rejected goal from there.
        if st.alive:
            with self._time_stage('_pick_settle'):
                self._pick_settle(st)
            self._obj_state(st, "after-settle")
        if st.alive:
            with self._time_stage('_pick_descend'):
                self._pick_descend(st)
            # SEAT_GUARDED=1: touch -> centre -> release seat, sensing depth/offset instead of
            # assuming. Sequenced here, not inside the descent, so the grasp bench can align between.
            if st.alive and os.environ.get("SEAT_GUARDED", "0") == "1":
                self._pick_seat_guarded(st)
            self._obj_state(st, "after-descend")
            # Settle the hand before any geometry is measured: with the hand still travelling
            # millimetres per step the object tracks it -- dragged, not grasped.
            self._settle_hand("approach-end")
            self._obj_state(st, "after-settle-hand")
            with self._time_stage('_pick_close'):
                self._pick_close(st)
            # between close and seat: who opens the hand?
            self._pad_trace("after-pick-close")
        if not st.alive:
            # exploded, shoved, or aborted at a checkpoint: do NOT seat-sweep toward an invalid
            # pinch point (that flings the object across the arena) -- fail clean, run_cycle retries.
            st.restore_bystanders()
            self._release_chassis("abort")
            print(f">>> PICK obj {obj_idx}: held=False (aborted before seat)", flush=True)
            return False
        with self._time_stage('_pick_seat'):
            self._pick_seat(st)
        with self._time_stage('_pick_capture'):
            self._pick_capture(st)
        with self._time_stage('_pick_lift'):
            self._pick_lift(st)
        # Release the chassis only once the grasp is finished; place must be able to drive.
        self._release_chassis("pick-end")
        return st.held
