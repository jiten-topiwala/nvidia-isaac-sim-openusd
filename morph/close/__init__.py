"""The finger close. `_close_replay` runs the stages in strict order: align, force balance.
State crossing stages lives on `_CloseState`; a stage may not reach into another stage's
locals."""
import os

import numpy as np

from morph.config import MOUTH_VEC_HAND
from morph.gripper.api import ForceBalanceStage, close_gripper
from morph.close.approach_align import ApproachAlignStage
from morph.close.lowest import LowestStage
from morph.close.ramp import RampStage

__all__ = ["CloseMixin"]


class _CloseState:
    """The locals that cross stage boundaries; anything else belongs to exactly one stage."""

    __slots__ = ("obj", "status", "q", "bx", "by", "byaw", "fb_held")

    def __init__(self, q_base, obj, status):
        self.obj, self.status = obj, status
        self.q = q_base
        # fingers force-close latched (3 = hold is final)
        self.fb_held = set()
        # the base ledger, refreshed by `_close_sync_arm`
        self.bx = self.by = self.byaw = 0.0


class CloseMixin(ApproachAlignStage, LowestStage, RampStage, ForceBalanceStage):
    """Mixed into Demo. Every `self.*` it touches is owned by Demo, not by this class."""

    def _close_sync_arm(self, st, tag):
        """Re-read the ARM half of `st.q`, and the base ledger, from the live robot.

        Must follow a stage that drives the arm outside `st.q`; without it the next
        `_force(st.q)` teleports the arm back. Arm indices only: rebuilding the finger command
        from measurement zeroes the tracking error, and so the grip force."""
        qa = np.asarray(self.robot.get_joint_positions(), float)
        st.q[self._arm_ii] = qa[self._arm_ii]
        st.bx, st.by, st.byaw = self.base_ledger()

    def _pin_brake_apply(self, on):
        """Set/clear the hub damper gains in place (only the four hub joints change)."""
        self._pin_brake = bool(on)
        try:
            kp, kd = self.ctrl.get_gains()
            kp = np.asarray(kp, float).reshape(-1).copy(); kd = np.asarray(kd, float).reshape(-1).copy()
            hubs = [i for n, i in self.idx.items() if "wheel" in n and "rolling" in n]
            for i in hubs:
                kp[i], kd[i] = self._wheel_gain(self.names[i])
            self.ctrl.set_gains(kp, kd)
            self._pin_hub_ii = np.asarray(hubs, int)
            self._pin_hub_q = (np.asarray(self.robot.get_joint_positions(), float)[self._pin_hub_ii].copy()
                               if on and kp[hubs[0]] > 0 else None)
            print(f">>> chassis brake {'ON' if on else 'OFF'}: {len(hubs)} wheel hubs kd "
                  f"{kd[hubs[0]] if hubs else float('nan'):g} (kp 0), rollers free", flush=True)
        except Exception as _e:
            self._pin_brake = False
            print(f">>> chassis brake unavailable ({_e}) -- falling back to the velocity pin", flush=True)

    def pin_chassis(self, on, tag="", tol=None, brake=None):
        """Hold the chassis (root pose and wheel joints) for the whole grasp, not one stage.

        The robot is still rolling at grasp entry and the chassis is a floating root, so zeroing
        the wheel joints is not enough. Implemented by wrapping `self.world.step`."""
        if on:
            if getattr(self, "_pin_orig_step", None) is not None:
                # An IDLE pin must YIELD to a purposeful one: the re-pin branch keeps the ORIGINAL
                # anchor, which after an idle holds the chassis nowhere near the dock.
                if getattr(self, "_pin_tag", "") == "idle" and tag != "idle":
                    self.pin_chassis(False, "idle-yield")
                else:
                    # re-pin (dock -> close-entry): re-apply the brake, other gain writers drop it.
                    if os.environ.get("PIN_DRIVE_BRAKE", "0") == "1":
                        self._pin_brake_apply(True)
                    return
            self._pin_tag = tag
            self._pin_ii = np.asarray([i for n, i in self.idx.items()
                                       if ("wheel" in n and "rolling" in n) or "slipping" in n])
            _q = np.asarray(self.robot.get_joint_positions(), float)
            self._pin_qw = _q[self._pin_ii].copy() if len(self._pin_ii) else None
            self._pin_base = self.base_ledger()
            # Capture the height too: `_pin_base` is xy/yaw and `set_base` has 20mm of z slack.
            try:
                self._pin_z = float(self.robot.get_world_pose()[0][2])
            except Exception:
                self._pin_z = None
            self._pin_orig_step = self.world.step
            # PIN_DRIVE_BRAKE=1: brake the wheel hubs with damper drives instead of zeroing joint
            # velocities per step. Right for a grasp; at a sustained IDLE pass `brake=False`.
            self._pin_brake = (bool(os.environ.get("PIN_DRIVE_BRAKE", "0") == "1")
                               if brake is None else bool(brake))
            if self._pin_brake:
                self._pin_brake_apply(True)

            # Re-assert on drift, not every step: a root teleport rebuilds the articulation state
            # and freezes the weak distal drives on their limits. `tol` overrides the 1mm default.
            _pin_tol = float(os.environ.get("PIN_TOL", "0.001")) if tol is None else float(tol)

            def _pinned_step(*a, **kw):
                bx, by, byaw = self._pin_base
                _drift = _pin_tol + 1.0
                try:
                    _p = np.asarray(self.robot.get_world_pose()[0], float)
                    _drift = float(np.hypot(_p[0] - bx, _p[1] - by))
                except Exception:
                    pass
                if _drift > _pin_tol:
                    self.set_base(bx, by, byaw, force=True)
                    self._pin_hits = getattr(self, "_pin_hits", 0) + 1
                    # PIN_TRACE=1: `_drift` is what physics does to the root during ONE world.step
                    # after a correction. Reading straight after `set_base` measures nothing.
                    if (os.environ.get("PIN_TRACE", "0") == "1"
                            and (self._pin_hits == 1 or self._pin_hits % 20 == 0)):
                        print(f">>> pin-trace[{getattr(self, '_stage', '?')}] hit "
                              f"{self._pin_hits}: root moved {_drift * 1000:.1f}mm in one step "
                              f"after the last correction", flush=True)
                elif not self._pin_brake:
                    # anti-coast without the teleport
                    try:
                        self.robot.set_velocities(np.zeros((1, 6)))
                    except Exception:
                        pass
                if self._pin_qw is not None and not self._pin_brake:
                    self.robot.set_joint_velocities(np.zeros(len(self._pin_ii)),
                                                    joint_indices=self._pin_ii)
                return self._pin_orig_step(*a, **kw)

            # ...and lock it: the pin restores the base after a move, the lock refuses the move.
            if os.environ.get("NO_CHASSIS_GRASP", "1") == "1":
                self._grasp_locked = tuple(self._pin_base)
                self._base_block_n = {}
            self.world.step = _pinned_step
            print(f">>> chassis PINNED{(' ' + tag) if tag else ''}: root held at "
                  f"({self._pin_base[0]:.3f}, {self._pin_base[1]:.3f}) and "
                  f"{len(self._pin_ii)} wheel joints zeroed for the whole grasp", flush=True)
        else:
            if getattr(self, "_pin_orig_step", None) is not None:
                self.world.step = self._pin_orig_step
                if getattr(self, "_pin_brake", False):
                    self._pin_brake_apply(False)
                self._pin_orig_step = None
                # z goes back to the slack rule once unpinned
                self._pin_z = None
                _blk = getattr(self, "_base_block_n", {})
                print(f">>> chassis released{(' ' + tag) if tag else ''} "
                      f"({getattr(self, '_pin_hits', 0)} drift re-asserts, "
                      f"{getattr(self, '_drv_jump_n', 0)} drive jump resets, "
                      f"{getattr(self, '_base_skip_n', 0)} base writes skipped"
                      + (f", REFUSED {sum(_blk.values())} base moves from "
                         f"{ {k: v for k, v in sorted(_blk.items(), key=lambda kv: -kv[1])} }"
                         if _blk else "") + ")", flush=True)
                self._pin_hits = 0
                self._grasp_locked = None
                self._base_block_n = {}

    def _drive_stamp(self, tag):
        """The finger drive target the articulation holds, read back from the controller."""
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
        """Object tilt, height and speed at a boundary; axis.z below ~0.99 is already going over."""
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

    def _close_replay(self, q_base, obj, status=None):
        """Run the close stages in order and return the final joint command."""
        st = _CloseState(q_base, obj, status)
        # Diagnostic only: place the object at the pinch kinematically, bypassing the approach.
        if os.environ.get("GRASP_TELEPORT", "0") == "1":
            try:
                _p = self._pinch()
                _oc = np.asarray(obj.get_world_poses()[0][0], float)
                _tx, _ty = float(_p[0]), float(_p[1])
                # GRASP_TELEPORT_OFF: mm along the world mouth axis, +ve = DEEPER, 0 = on the diameter.
                _off_mm = float(os.environ.get("GRASP_TELEPORT_OFF", "0"))
                if abs(_off_mm) > 1e-9:
                    _, _hR = self._hand_frame()
                    _m = (_hR @ MOUTH_VEC_HAND)[:2]
                    _m = _m / max(1e-9, float(np.linalg.norm(_m)))
                    _tx += _off_mm * 1e-3 * float(_m[0])
                    _ty += _off_mm * 1e-3 * float(_m[1])
                obj.set_world_poses(
                    positions=np.array([[_tx, _ty, float(_oc[2])]]),
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
        self._grasp_geometry_report(obj, tag="close-entry")
        # Refuse to close from a pose no downstream servo can recover; deliberately wider than
        # the reachable window, because the aligns below absorb moderate error.
        if os.environ.get("CLOSE_GATE", "1") == "1":
            _dd_g = self._du_dv(obj)
            _pz_g = float(self._pinch()[2])
            _oc_g = np.asarray(obj.get_world_poses()[0][0], float)
            _z_hi = float(_oc_g[2]) + self.obj_half_h + 0.03
            # What the close itself can survive: the -130mm window edge plus its stagger tolerance.
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
                self._fallback("close-gate-unreachable")
                print(f">>> CLOSE GATE: pose unreachable ({'; '.join(_bad)}) -- SKIPPING the "
                      f"close instead of thrashing on it; the pick fails clean and recovers",
                      flush=True)
                # Block the rest of the pick: returning alone would let capture/lift run on nothing.
                self._gate_blocked = True
                return st.q
        if os.environ.get("PIN_CHASSIS", "1") == "1":
            self.pin_chassis(True, "close-entry")
        self._obj_watch(obj, "before-close")
        print(">>> close: DIRECT synchronous force-close (entry is centered and in-window)",
              flush=True)
        self._set_arm_gain(3.0e5)
        if os.environ.get("CLOSE_GENTLE", "1") == "1":
            self._grip_gentle()
        self._finger_cmd = st.q[self._f_idx].copy()
        # Align in the hand frame: du/dv is a world-XY projection, degenerate once pitched down.
        _seat_owns = (os.environ.get("SEAT_GUARDED", "0") == "1"
                      and os.environ.get("OBJ_GOAL", "0") == "2"
                      and os.environ.get("CLOSE_ALIGN_AFTER_SEAT", "0") != "1")
        if _seat_owns:
            # ONE depth authority: the guarded seat already owns depth, lateral and z.
            print(">>> close: aligns SKIPPED after the guarded seat (the seat owns depth, lateral and z)",
                  flush=True)
        else:
            # size-independent: object -> finger-arc centre
            self._align_to_grasp_center(obj)
            # then columns bring the pinch to object MID-height
            self._align_z_to_center(obj)
        self._obj_watch(obj, "after-align")
        self._close_sync_arm(st, "align")
        # Close on measured pad force until every pad reads FB_TARGET.
        close_gripper(self, st)
        self._drive_stamp("before-geom-report")
        self._grasp_geometry_report(obj, tag="post-close")
        self._drive_stamp("after-geom-report")
        if len(st.fb_held) == 0:
            # A close that loaded nothing is a missed grasp: fail clean, never lift on air.
            self._fallback("close-missed-grasp")
            print(">>> close: force-close loaded NO finger -> missed grasp, ABORT (no lift on air)",
                  flush=True)
            # the pick's clean-fail path (capture SKIPPED)
            self._gate_blocked = True
            return
        if len(st.fb_held) in (1, 2):
            self._fallback("close-partial-latch")
            print(f">>> close: force-close latched only {sorted(st.fb_held)} -> partial latch, ABORT "
                  f"(a three-finger hold is the only grip; run_cycle retries)", flush=True)
            self._gate_blocked = True
            return
        self._pad_trace("close-return")
        return st.q
