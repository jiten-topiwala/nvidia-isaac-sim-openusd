"""The settling joint-space arm ramp shared by the close stages: interpolate to a pose, settling at
each step, with an opt-in `ramp_blocked` refusal. Part of the close sequence — see
`morph/close/__init__.py`. Pure Python: it imports no Isaac API and loads without a running Sim."""
import numpy as np

from morph.arm.api import ramp_blocked
from morph.config import HEADLESS


class RampStage:
    """Mixed into `CloseMixin`; all `self.*` belong to `Demo`."""

    def _ramp_arm(self, q_from, q_to, n_steps=8, settle=0.02, check=False):
        """Interpolate to a new arm pose over n steps, settling at each, instead of snapping. The arm
        is a forced parallel linkage, so snapping leaves its LUT-driven passive joints in a transient
        that makes every measurement taken during it unusable. False if the articulation diverged,
        or if `check` and `ramp_blocked` refuses the target. `check` is opt-in: a refused RESTORE
        would strand the arm at the pose its caller was escaping.
        """
        q_from = np.asarray(q_from, float)
        q_to = np.asarray(q_to, float)
        _ref = ramp_blocked(self, q_to) if check else None
        if _ref is not None:
            if _ref[0] == "unbuildable":
                self._fallback("ramp-unbuildable")
            else:
                self._fallback("ramp-blocked")
            print(f">>> ramp: REFUSED ({_ref[0]}) -- {_ref[1]}; nothing is commanded. The check "
                  f"sees WORLD boxes only: not the chassis, and not the grasp target between the "
                  f"fingers", flush=True)
            return False
        # Step count scales with distance; a fixed one turns a large restore into velocity spikes.
        _dmax = float(np.max(np.abs(q_to - q_from))) if len(q_to) else 0.0
        n_steps = max(int(n_steps), int(_dmax / 0.004))
        n_settle = max(1, int(settle / self.dt))
        # Pin the base for the whole ramp: a boom tilted down and extended forward pushes the
        # chassis, so the no-chassis-push rule holds in every stage that moves the arm.
        try:
            _bx, _by, _byaw = self.base_ledger()
        except Exception:
            _bx = None
        for s_i in range(1, n_steps + 1):
            qi = q_from + (q_to - q_from) * (float(s_i) / n_steps)
            if _bx is not None:
                self.set_base(_bx, _by, _byaw)
            self._force(qi)
            self._apply(qi)
            for _ in range(n_settle):
                if _bx is not None:
                    self.set_base(_bx, _by, _byaw)
                self.world.step(render=not HEADLESS)
            pz = float(self._pinch()[2])
            if not np.isfinite(pz) or abs(pz) > 10.0:
                return False
        return True
