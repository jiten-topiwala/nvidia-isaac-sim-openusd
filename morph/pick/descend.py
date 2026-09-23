"""Settle onto the reach pose, then the fine align that servos the pinch centroid onto the object
centre. Runs after `dock`; import only after SimulationApp."""
import math

import numpy as np


from morph.arm.api import ArmModel, Q8_TOL_M, Q8_TOL_RAD
from morph.config import HEADLESS, KNOWN


class DescendStage:
    """One stage of `pick`. Mixed into `PickMixin`; all `self.*` belong to `Demo`."""

    def _pick_settle(self, st):
        self._stage = "settle"
        obj_idx, status = st.obj_idx, st.status
        dock, qg = st.dock, st.qg
        pin_bystanders = st.pin_bystanders
        # The snap is a NO-OP while the residue is ~0, which is what a normal reach leaves. Past
        # `q8_resid`'s own bands it teleports the arm a real distance into a pose nothing checked.
        _sm, _sr = ArmModel.q8_resid(
            self._arm_q8(), np.array([qg[self.idx[n]] for n in ArmModel.Q8], float))
        if _sm > Q8_TOL_M or _sr > Q8_TOL_RAD:
            self._fallback("settle-snap-too-far")
            print(f">>> settle: NOT snapping -- the residue is {_sm * 1000:.0f}mm / "
                  f"{_sr * 1000:.0f}mrad, past the {Q8_TOL_M * 1000:.0f}mm / "
                  f"{Q8_TOL_RAD * 1000:.0f}mrad this snap is for; holding the pose the arm "
                  f"actually reached", flush=True)
            # Hold THAT, not `qg`: `_hold` commands its vector every step, so leaving `qg` on the
            # goal drives the refused distance anyway, and a residue this large means no arrival.
            qg = np.asarray(self.robot.get_joint_positions(), float).copy()
            st.alive, st.grip = False, qg.copy()
            return
        # The snap of a ~0 residue.
        self.robot.set_joint_positions(qg)

        kp = np.full(len(self.names), 2.0e3)
        kd = np.full(len(self.names), 2.0e2)
        # 3e5/3e4 = the stable settle gains
        for n in KNOWN["arm_joints"]:
            if n in self.idx:
                kp[self.idx[n]], kd[self.idx[n]], _ = self._arm_gain(n, 3.0e5)
        for i, n in enumerate(self.names):
            if "rolling_joint" in n or "slipping" in n:
                kp[i], kd[i] = self._wheel_gain(n)   # keeps the chassis brake between dock and close
        self._set_gains(kp, kd)
        self._gains_writer = ("_set_arm_gain", {"arm_kp": 3.0e5})
        # SHORT settle: every extra 10th of a second at depth lets a pad graze GRIND the object away
        settle_t = 0.1
        self._hold(qg, int(settle_t / self.dt), on_step=pin_bystanders)
        alive = self._finite("settle")
        if alive:
            sbx, sby, _ = self.base_pose()
            if math.hypot(sbx - float(dock[0]), sby - float(dock[1])) > 0.15:
                # A wall or rack fight can shove the base off the dock while staying finite
                self._fallback("settle-base-shoved")
                print(f">>> BASE SHOVED to ({sbx:.2f},{sby:.2f}) during reach -> abort pick", flush=True)
                alive = False
        grip = qg.copy()
        st.alive, st.grip = alive, grip

    def _pick_seat_guarded(self, st):
        """The guarded seat after the descent: touch, centre, release, sensing depth and offset."""
        from morph.pick.guarded_seat import guarded_seat
        guarded_seat(self, st)

    def _pick_descend(self, st):
        self._stage = "descend"
        O, obj, park = st.O, st.obj, st.park
        alive, grip, _oc0 = st.alive, st.grip, st._oc0
        pin_bystanders = st.pin_bystanders

        # Fine align: the hand comes to the object; the object must not move.
        if alive and self.clut is not None:
            # Already aligned AT HOVER, so the verdict here is OBJECT DRIFT, not pinch centroid
            # A fresh contact ledger for the seat and the close: the descent's touches are theirs to judge.
            self._hit_step = False
            self._touch = {}
            _pz = self._pinch()
            print(f">>> descent end: pinch z {_pz[2]:.3f} vs object top "
                  f"{_oc0[2] + self.obj_half_h:.3f} (tips {(_oc0[2] + self.obj_half_h - _pz[2]) * 1000:.0f}"
                  f"mm below top)", flush=True)
            if self._ctc:
                print(f">>> descent contacts (steps touching object): "
                      f"{dict(sorted(self._ctc.items(), key=lambda kv: -kv[1])[:6])}", flush=True)
            _oc = np.asarray(obj.get_world_poses()[0][0], float)
            # Tensor-only gate: a small seat-nudge from the palm-touch stop is expected
            e_align = float(np.linalg.norm(_oc[:2] - np.asarray(O, float)[:2])) - 0.03
            e_align = max(e_align, 0.0)
            print(f">>> post-approach object displacement: {e_align * 1000:.0f}mm "
                  f"(obj {np.round(_oc, 3).tolist()} pinch {np.round(self._pinch(), 3).tolist()})", flush=True)
            if alive and e_align > 0.05:
                # object displaced: closing here grabs AIR and the curl drives fight whatever they hit
                self._fallback("descend-align-displaced")
                print(f">>> fine-align FAILED ({e_align * 1000:.0f}mm) -> abort pick", flush=True)
                alive = False
        st.alive, st.grip = alive, grip
