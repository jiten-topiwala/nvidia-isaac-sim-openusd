"""The driven finger close — the stage that actually grips in friction mode.
Part of the pick cycle — see `morph/pick/__init__.py` for stage order and shared state.
Imports Isaac APIs at module level; only importable after SimulationApp."""
import os

import numpy as np

from morph.diagnostics import finger_vector_line


class CloseAnimStage:
    """One stage of `pick`. Mixed into `PickMixin`; all `self.*` belong to `Demo`."""

    def _pick_close(self, st):
        self._stage = "close"
        obj_idx, status = st.obj_idx, st.status
        obj = st.obj
        alive, grip = st.alive, st.grip

        # The recorded close replay is the only close; without its asset the stage refuses.
        # Stage order below is load-bearing.
        if alive:
            if status:
                status(f"closing gripper on object {obj_idx}")
            q_now = np.asarray(self.robot.get_joint_positions(), float).copy()
            fj = {fl: round(float(q_now[self.idx[f"finger_{fl}_joint_1_1"]]), 3)
                  for fl in "abc" if f"finger_{fl}_joint_1_1" in self.idx}
            print(f">>> close-entry finger j1: {fj}", flush=True)
            # The pre-grip hand is the stage's authored state, written nowhere in Python.
            print(finger_vector_line(self, q_now, "close-entry fingers"), flush=True)
            if self._finger_cmd is None:
                self._finger_cmd = q_now[self._f_idx].copy()
            self._tip_report("close-entry", obj)
            self._phase_report("close-entry", obj)
            self._ctc = {}
            gaps = self._tip_gaps(obj)
            print(f">>> close entry gaps(mm): "
                  f"{ {f: round(g * 1000) for f, g in gaps.items()} }", flush=True)
            # The grasp starts here, so the chassis-stationary rule starts here, not at the close
            # pin several stages later.
            if (os.environ.get("NO_CHASSIS_GRASP", "1") == "1"
                    and getattr(self, "_grasp_locked", None) is None):
                self._grasp_locked = self.base_ledger()
                self._base_block_n = {}
                print(f">>> chassis LOCKED at grasp entry "
                      f"({self._grasp_locked[0]:.3f}, {self._grasp_locked[1]:.3f}) -- "
                      f"every remaining gap is closed with the ARM", flush=True)
            if self.close_traj is not None:
                # `close_traj` only gates here; the close itself is the direct force-close.
                grip = self._close_replay(q_now, obj, status=status)
                alive = self._finite("close-replay")
                self._tip_report("close-end", obj)
                self._phase_report("close-end", obj)
                print(f">>> close contacts: "
                      f"{dict(sorted(self._ctc.items(), key=lambda kv: -kv[1])[:8])}", flush=True)
                # ...and the load each carried: contact events with no impulse means resting on it.
                _cn = {k2: round(v2, 2) for k2, v2 in
                       sorted(getattr(self, "_ctcN", {}).items(), key=lambda kv: -kv[1])[:8]}
                print(f">>> close contact impulse (Ns per link): {_cn}", flush=True)
            else:
                # The recorded close is the only close: the curl fallback never gripped
                # (docs/FINDINGS.md PART F/15). `_pick_attempt` reads `alive` and fails clean.
                self._fallback("close-no-asset")
                print(">>> close: no recorded close trajectory -> REFUSE; the attempt ends and "
                      "run_cycle retries", flush=True)
                alive = False
        st.alive, st.grip = alive, grip
