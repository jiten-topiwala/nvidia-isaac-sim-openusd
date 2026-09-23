"""What the arm package needs from the robot: the model, the measured pose, the bounds, the
obstacle set and the fallback tally. Composed into `Demo`; the stage readers import at the call
because they need pxr and this module must import offline."""
import numpy as np

from morph.arm.execute import commanded_fingers
from morph.arm.model import ArmModel
from morph.config import A1_MAX, ARM_MODEL, CLOSURE_LUT, COLUMN_MAX


class ArmMixin:
    """Mixed into `Demo`. Every `self.*` it touches is owned by `Demo`."""

    def _arm_model(self):
        if getattr(self, "_armm", None) is None:
            self._armm = ArmModel.load(ARM_MODEL, CLOSURE_LUT)
        return self._armm

    def _arm_q8(self):
        """The eight actuated joints, in ArmModel.Q8 order, as the articulation currently holds."""
        q = np.asarray(self.robot.get_joint_positions(), float)
        return np.array([float(q[self.idx[n]]) for n in ArmModel.Q8])

    def _arm_bounds(self):
        """Planning bounds: the USD joint limits, tightened by COLUMN_MAX / A1_MAX (metres, rad)."""
        lo, hi = self._arm_model().bounds()
        lo, hi = lo.copy(), hi.copy()
        hi[1] = min(hi[1], COLUMN_MAX)
        hi[2] = min(hi[2], COLUMN_MAX)
        hi[3] = min(hi[3], A1_MAX)
        return lo, hi

    def _commanded_fingers(self):
        """The hand the checker judges: the drive target while the grip owns the fingers."""
        return commanded_fingers(self)

    def _arm_obstacles(self, grasp_names=(), *, exclude_names=(), diagnostic_paths=None):
        """The raw obstacle set off the live stage; the seam every consumer reads it through."""
        from morph.arm.stage import arm_obstacles
        return arm_obstacles(self, grasp_names, exclude_names=exclude_names,
                             diagnostic_paths=diagnostic_paths)

    def _arm_base_world(self):
        """(translation3 metres, rotation3x3) of the `base` link now, in WORLD -- the measured pose
        every plan and checkpoint is validated against. Full rotation, not yaw alone."""
        bp, bq = self._arm_base_prim().get_world_pose()
        w, x, y, z = [float(v) for v in bq]
        return np.asarray(bp, float), np.array(
            [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
             [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
             [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], float)

    def _arm_base_prim(self):
        if getattr(self, "_armbp", None) is None:
            from isaacsim.core.prims import SingleXFormPrim
            from morph.arm.stage import arm_prim_paths
            self._armbp = SingleXFormPrim(arm_prim_paths(self)[1])
        return self._armbp

    def _fallback(self, reason):
        """COUNT a planned-motion degrade by reason: play_isaac.py prints the tally per cycle and
        verify_place.py grades it, so a SAFETY-class degrade can never read as a green run."""
        d = getattr(self, "_plan_fallbacks", None)
        if d is None:
            d = self._plan_fallbacks = {}
        d[reason] = d.get(reason, 0) + 1
