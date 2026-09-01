"""The pick cycle.

`pick` runs a fixed sequence of stages against the live scene; each one reads what the stage
before it left and either advances or sets `alive = False`:

    setup     dock-angle search with a clear corridor, then nav             dock.py
    pose      force the settled arm config + cupped fingers                 dock.py
    reach     replay the baked reach trajectory                             reach.py
    settle    snap the residue, switch to the settle gains                  descend.py
    descend   fine align: pinch centroid onto the object centre             descend.py
    close     the driven finger close                                       close_anim.py
    seat      compute the seat pose the pin will freeze                     grasp.py
    capture   hold, capture the grip offset, decide `held`                  grasp.py
    lift      replay the lift and check the object actually rose            grasp.py

Everything that crosses a stage boundary lives on `_PickState`; anything else is local to one
stage. `_animate_fingers` (animate.py) and the finger-pose helpers (prep.py) are called by the
stages rather than being stages themselves.
"""
import os
from morph.config import HEADLESS, KNOWN
import numpy as np

from morph.pick.animate import FingerAnimStage
from morph.pick.close_anim import CloseAnimStage
from morph.pick.descend import DescendStage
from morph.pick.dock import DockStage
from morph.pick.grasp import GraspStage
from morph.pick.prep import PrepStage
from morph.pick.reach import ReachStage

__all__ = ["PickMixin"]


class _PickState:
    """The locals that cross stage boundaries. If a name is not here it belongs to exactly one
    stage — that is the contract."""

    __slots__ = ("obj_idx", "status", "O", "psi", "dock", "reach_ok", "qg", "obj", "park",
                 "close_anim", "replay", "cup_fingers", "pin_bystanders", "restore_bystanders",
                 "alive", "grip", "seat", "seat_gap", "floor_pos", "held", "_oc0")

    def __init__(self, obj_idx, status):
        self.obj_idx, self.status = obj_idx, status
        self.alive, self.held = True, False


class PickMixin(PrepStage, FingerAnimStage, DockStage, ReachStage, DescendStage,
                CloseAnimStage, GraspStage):
    """Mixed into Demo. Every `self.*` it touches is owned by Demo."""

    def _settle_hand(self, tag, tol=None, t_max=None):
        """Hold the current joint command until the pinch stops moving. Returns the final
        per-step pinch speed (m/step)."""
        tol = 0.0002 if tol is None else tol
        n_max = int((float(os.environ.get("SETTLE_T", "2.5")) if t_max is None else t_max) / self.dt)
        q = np.asarray(self.robot.get_joint_positions(), float).copy()
        p_last = np.asarray(self._pinch(), float)
        v, k = 1e9, 0
        # Watch the object while the hand holds still. The hand is kinematically forced here, so a
        # palm resting on the object is an infinite-stiffness wall and the object can be driven
        # through the floor before any close stage runs.
        _ow = self._obj_by_focus() if hasattr(self, "_obj_by_focus") else None
        if _ow is None:
            _fo = getattr(self, "_focus_obj", None)
            _ow = self.objs[int(_fo.rsplit("_", 1)[-1])] if _fo else None
        _z0 = float(np.asarray(_ow.get_world_poses()[0][0], float)[2]) if _ow is not None else None
        _z_said = False
        # How close is the hand before the settle even starts? A descent that reports zero object
        # contacts can still leave the object departing within a few steps of this loop, under
        # either hold mode -- consistent with the hand finishing the descent already INSIDE the
        # contact offset, where no event has fired but the smallest motion makes one.
        if _ow is not None and os.environ.get("SETTLE_CLEAR", "1") == "1":
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
            # Do NOT kinematically force the arm while settling. A palm resting on the object is an
            # infinite-stiffness wall, and over a couple of seconds it displaces the object through
            # the floor with no contact force at all -- a kinematic body does not need any.
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
        """Object tilt + height at a pick-stage boundary — the fastest way to name the stage
        that destabilises it. Extracts the stage's object and defers to `_obj_watch`."""
        obj = getattr(st, "obj", None)
        if obj is not None:
            self._obj_watch(obj, tag, rest=self.obj_half_h)

    def pick(self, obj_idx, status=None):
        self._gate_blocked = False           # close-gate veto from a PREVIOUS attempt must not leak
        for _a in ("_cw_z0", "_cw_xy0", "_cw_said", "_cw_broke"):
            # close-watch state is PER PICK: left stale, the "already reported" latch silences the
            # probe for every cycle after the first
            self.__dict__.pop(_a, None)
        """Nav to a standoff facing the object, force the known arm config, seat the object in
        the finger cage, close + pin.  Returns True on grasp."""
        st = _PickState(obj_idx, status)
        self._pick_setup(st)
        # Stability trace. The close can receive an object that is already falling: nothing in
        # contact at close entry, and the object still travelling and tilted -- so the instability
        # arrives UPSTREAM and every close-side conclusion is downstream of whichever stage
        # introduces it.
        self._obj_state(st, "after-setup")
        self._pick_pose(st)
        self._obj_state(st, "after-pose")
        self._pick_reach(st)
        self._obj_state(st, "after-reach")
        self._pick_settle(st)
        self._obj_state(st, "after-settle")
        self._pick_descend(st)
        self._obj_state(st, "after-descend")
        # Settle the hand before any geometry is measured. Entered with the hand still travelling
        # millimetres per step, the object tracks it almost exactly -- dragged, not grasped -- and
        # du and the palm clearance move by tens of millimetres over the settle.
        self._settle_hand("approach-end")
        self._obj_state(st, "after-settle-hand")
        self._pick_close(st)
        self._pad_trace("after-pick-close")   # between the close and the seat: who opens the hand?
        if not st.alive:
            # exploded or shoved: do NOT seat-sweep the object toward an invalid pinch point (that
            # is what "flew" it across the arena) — fail clean, run_cycle retries in teleport mode.
            st.restore_bystanders()
            try:
                self.pin_chassis(False, "abort")
            except Exception:
                pass
            print(f">>> PICK obj {obj_idx}: held=False (aborted before seat)", flush=True)
            return False
        self._pick_seat(st)
        self._pick_capture(st)
        self._pick_lift(st)
        # Release the chassis only once the grasp is finished -- the place stage must be able to
        # drive.
        try:
            self.pin_chassis(False, "pick-end")
        except Exception:
            pass
        return st.held
