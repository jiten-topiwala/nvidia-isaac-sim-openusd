"""Backing the hand off an object it has released, then planning on to the carry pose."""
import os

import numpy as np

from morph.arm.api import Outcome, move_linear, move_to_pose
from morph.arm.execute import commanded_fingers
from morph.arm.plan import plan_start
from morph.arm.collision import MARGIN, Q8_TOL_M, Q8_TOL_RAD
from morph.config import COLUMN_MAX, MOUTH_VEC_HAND

# A restart can land the next waypoint on another IK branch, unchecked.
_RETREAT_IK_RESTARTS = 0


def cartesian_retreat(demo, obj_idx):
    """Retreat the hand along the mouth axis until its box clears the released object's by
    `MARGIN`, then plan the rest of the way to carry; False if the line aborts or the remainder
    cannot be planned or run. Only the line exempts the hand from the released object's box: the
    remainder is a free-space plan and sees it as an ordinary obstacle."""
    m, q8 = demo._arm_model(), demo._arm_q8()
    bt, bR = demo._arm_base_world()
    # WORLD: the frame `move_linear` reads. Only the hand's rotation is wanted here.
    _, R = m.fk(q8)[m.hand_link]
    axis_w = bR @ (R @ MOUTH_VEC_HAND)
    grasp = (f"pickup_obj_{obj_idx}",)
    obs = demo._arm_obstacles(grasp)
    # THE SAME GEOMETRY THE CHECK USES: derived on the hand `move_linear` is given below, not on the
    # swept union. Sizing on the sweep asks the arm to travel for clearance the checker never wanted.
    _fing = commanded_fingers(demo)

    def _near_edge(loc, lp, lR):
        """Near edge, on the mouth axis, of one link's OBB grown by MARGIN."""
        _lo, _hi = np.asarray(loc, float)
        _R, _p = bR @ lR, bt + bR @ lp
        return float((_p + _R @ ((_lo + _hi) / 2.0)) @ axis_w) - float(
            np.abs(_R.T @ axis_w) @ ((_hi - _lo) / 2.0 + MARGIN))

    # Far edge of the released object's box on the same axis: the gap from a link's near edge
    # to it IS how far the hand must travel for `_hits` to separate.
    _far = [float(ob.mean(axis=0) @ axis_w + np.abs(axis_w) @ ((ob[1] - ob[0]) / 2.0))
            for ob, tg in obs if tg == "grasped"]
    hand_min = min(_near_edge(_loc, _lp, _lR)
                   for _lk, _loc, _lp, _lR in m._links(q8, None, fingers=_fing)
                   if _lk in m.hand_subtree)
    dist = max(_far or [hand_min]) - hand_min
    # SHADOW, refuses nothing: `dist` is the HAND's requirement and the hand subtree is EXEMPT from a
    # grasped box, so the links actually enforced -- the boom, the wrist -- may need less. Report both.
    try:
        _need, _who = 0.0, "none"
        for _lk, _loc, _lp, _lR in m._links(q8, None, fingers=_fing):
            if _lk in m.hand_subtree or _lk == "__held__":
                continue
            _min = _near_edge(_loc, _lp, _lR)
            _d = max(_far or [_min]) - _min
            if _d > _need:
                _need, _who = _d, _lk
        print(f">>> retreat-need: hand asks {dist * 1000:.0f}mm (EXEMPT from the grasped box); "
              f"the checked links need {_need * 1000:.0f}mm (worst {_who})", flush=True)
    except Exception as _e_rn:                 # noqa: BLE001 -- a shadow must not kill a retreat
        print(f">>> retreat-need: unavailable ({_e_rn})", flush=True)
    _why = ""
    if dist <= 0.0:
        _why = f"nothing to separate from under pickup_obj_{obj_idx}"
    else:
        print(f">>> backout: cartesian retreat asks {dist * 1000:.0f}mm along the mouth axis",
              flush=True)
        mv = move_linear(demo, dist * axis_w, "world", link=m.hand_link,
                         allow_contact_with=grasp, tag="backout", fingers=_fing,
                         ik_restarts=_RETREAT_IK_RESTARTS)
        if mv.outcome is Outcome.NO_ARRIVAL:
            demo._fallback("retreat-no-arrival")
            print(f">>> backout: cartesian retreat did NOT arrive -- {mv.reason} "
                  f"(tol {Q8_TOL_M * 1000:.0f}mm / {Q8_TOL_RAD * 1000:.0f}mrad)", flush=True)
            return False
        # BOTH executor aborts: `abort-recovered` leaves `_replan_failed` False and still returns None,
        # having already driven the arm back to park -- the remainder must not drive it again.
        if (mv.outcome is Outcome.PAYLOAD_LOST
                or {"execute-replan-failed", "abort-recovered"} & set(mv.tags)):
            print(f">>> backout: cartesian retreat aborted -- {mv.reason}", flush=True)
            return False
        if mv:
            print(">>> backout: cartesian retreat arrived", flush=True)
        else:
            _why = mv.reason
    # ONE site for this tag: tests/test_fallback_counter.py forbids a second.
    if _why:
        demo._fallback("retreat-no-cartesian")
        print(f">>> backout: {_why}", flush=True)

    # The rest of the way out, from the seed `plan_start` chooses (commanded under drives).
    carry_q8 = plan_start(demo)[0].copy()
    c1 = m.Q8.index("ColumnLeftBearingJoint_1")
    c2 = m.Q8.index("ColumnRightBearingJoint_1")
    a1 = m.Q8.index("ArmLeftJoint_1")

    # Build the carry target matching the open-loop raise + const-z retract logic
    carry_q8[c1] = np.clip(carry_q8[c1] + 0.03, 0.0, COLUMN_MAX)
    carry_q8[c2] = np.clip(carry_q8[c2] + 0.03, 0.0, COLUMN_MAX)
    carry_q8[a1] = float(os.environ.get("CARRY_A1", "0.10"))

    mv = move_to_pose(demo, carry_q8, None, tag="backout-remainder")
    if mv.outcome is Outcome.NO_PLAN:
        demo._fallback("retreat-no-remainder")
        print(">>> backout: remainder plan to carry pose unavailable", flush=True)
        return False
    if not mv:
        print(f">>> backout: remainder to the carry pose did not complete -- {mv.reason}",
              flush=True)
    return bool(mv)
