"""Executing a planned path on the arm: eased writes, checkpoints, re-validation."""
import math
import os

import numpy as np

from morph.arm.cartesian import q8_derate, retime_q8
from morph.arm.obstacles import blame, planned_obstacles
from morph.arm.plan import plan_arm
from morph.arm.model import ArmModel
from morph.arm.collision import MARGIN
from morph.config import HEADLESS

# Base motion worth re-deriving a world goal for: translation moves the hand 1:1 and yaw 0.869 mm/mrad at
# the worst lever inside the bounds, so 5 mm is the band `q8_resid` calls an arrival, a quarter of MARGIN.
SLIP_RETARGET_M = 0.005


def with_closure_forced(demo, q):
    """Closure passives from the LUT for a planned frame, which carries none of its own."""
    q = np.asarray(q, float)
    i1, i2, ia = (demo.idx["ColumnLeftBearingJoint_1"], demo.idx["ColumnRightBearingJoint_1"],
                  demo.idx["ArmLeftJoint_1"])
    for pn, pv in demo._closure_passives(float(q[i2]) - float(q[i1]), float(q[ia])).items():
        if pn in demo.idx:
            q[demo.idx[pn]] = float(pv)
    return q


def execute_path(demo, waypoints, secs=None, held=None, checkpoint_every=8,
                      on_step=None, hold_fingers=True, grasp_names=(), tag="arm",
                      hold_base=True, regoal=None):
    """Write a planned path, timed by `retime` under the measured joint ceilings; returns the
    final full joint vector, or None if it did not complete (blocked checkpoint, stale goal, no
    obstacles, an abort already recovered to park). `secs` is a FLOOR, never a ceiling: without
    one the motion runs as fast as the limits allow. An interval whose endpoints coincide is no
    motion: it writes nothing and steps the simulator zero times. `grasp_names` MUST match the set
    the plan used."""
    m = demo._arm_model()
    goal = waypoints[-1]
    q_full = np.asarray(demo.robot.get_joint_positions(), float).copy()
    fhold = q_full[demo._f_idx].copy() if hold_fingers else None
    # The hand every check here judges, from the SAME condition as `fhold` (two hands in one
    # motion was d723967): the DRIVE TARGET, constant while the grip owns the hand (FINDINGS).
    _fing = commanded_fingers(demo) if hold_fingers else None
    bx, by, byaw = demo.base_ledger()
    b0 = demo._arm_base_world()          # P1.4: what the plan was made against
    q_prev = q_full.copy()
    every = max(1, int(checkpoint_every))
    derate = q8_derate(waypoints, secs)
    replans = 0
    i = 0
    # The hand this motion flies, published for its duration so `_apply` can hold it
    # kinematically: the finger drives carry the grasp's compliance and cannot.
    pin_prev = demo._finger_pin
    demo._finger_pin = fhold
    try:
        while i < len(waypoints) - 1:
            # `i` only ever lands on a chunk boundary below, so this IS the `checkpoint_every`
            # cadence.
            if i:
                # The set the plan was made under: re-fetched (obstacles move), padded (the deck
                # does not). Stricter than the planner here meant seven replans a cycle.
                try:
                    obs, _paths = planned_obstacles(demo, grasp_names)
                except Exception as _e_obs:        # noqa: BLE001 -- nothing checked can run
                    # Unjudgeable remainder: stop here rather than unwind past flown waypoints.
                    print(f">>> arm plan[{tag}]: checkpoint {i}/{len(waypoints)} -- obstacle set "
                          f"unavailable ({_e_obs}); ENDING the motion", flush=True)
                    demo._fallback("checkpoint-obstacles-unavailable")
                    demo._safety_abort = True
                    return None
                _b = demo._arm_base_world()          # re-measured, every checkpoint
                bad = next((k for k in range(i, len(waypoints))
                            if m.collides(waypoints[k], obs, MARGIN, held, _b,
                                          fingers=_fing)), None)
                # The margin the arm has HERE, on the checker's own predicate: a run metric, not a gate.
                try:
                    demo._bench_clearance = getattr(demo, "_bench_clearance", []) + [
                        (tag, i, round(float(m.clearance(waypoints[i], obs, held, _b, fingers=_fing)), 4))]
                except Exception:                  # noqa: BLE001 -- a diagnostic must not stop a motion
                    pass
                # Test hook, half of ARM_PLAN_TEST_FAIL_REPLAN: a real abort needs a blocked checkpoint
                # AND a failed replan; this supplies the second half.
                if (bad is None and os.environ.get("ARM_PLAN_TEST_FAIL_REPLAN") in ("1", tag)
                        and not getattr(demo, "_arm_test_failed_replan", False)):
                    bad = min(i + 1, len(waypoints) - 1)
                if bad is None:
                    print(f">>> arm plan[{tag}]: checkpoint {i}/{len(waypoints)} -- "
                          f"{len(waypoints) - i} remaining waypoints re-validated against the "
                          f"measured pose, all clear", flush=True)
                if bad is not None:
                    # Name the pair under the same predicate that refused it.
                    try:
                        _who = blame(m, waypoints[bad], obs, _paths, held, _b, _fing)
                    except Exception:                  # noqa: BLE001 -- a diagnostic must not stop a motion
                        _who = None
                    print(f">>> arm plan[{tag}]: checkpoint {i}/{len(waypoints)} -- waypoint {bad} "
                          f"is no longer clear against the measured pose"
                          + (f" ({_who[0]} inside the margin of {_who[1]})" if _who else "")
                          + "; replanning", flush=True)
                    # Test hook: force the first replan to fail, yielding the abort path. Never set outside
                    # a test.
                    if os.environ.get("ARM_PLAN_TEST_FAIL_REPLAN") in ("1", tag) and not getattr(demo, "_arm_test_failed_replan", False):
                        demo._arm_test_failed_replan = True
                        new = None
                    else:
                        # The goal may be world-anchored and `waypoints[-1]` is a JOINT vector, so a
                        # base slip puts the hand off in world BY the slip. Re-derive only then.
                        if regoal is not None:
                            _d = float(np.linalg.norm(np.asarray(_b[0], float)
                                                      - np.asarray(b0[0], float)))
                            if _d > SLIP_RETARGET_M:
                                _g = regoal()
                                if _g is None:
                                    demo._fallback("replan-goal-stale")
                                    print(f">>> arm plan[{tag}]: the base moved {_d * 1000:.0f}mm "
                                          f"and the goal could not be re-derived in world -- "
                                          f"refusing rather than replanning to a stale target",
                                          flush=True)
                                    demo._safety_abort = True
                                    return None
                                goal = _g
                                b0 = _b          # the slip is corrected; measure the next from HERE
                                print(f">>> arm plan[{tag}]: base moved {_d * 1000:.0f}mm -- goal "
                                      f"re-derived in world before replanning", flush=True)
                        new = plan_arm(demo, goal, held=held, grasp_names=grasp_names,
                                       fingers=_fing)
                    if new is None:
                        # The check just proved these waypoints unsafe: falling through would
                        # execute what it disproved. Only `Demo.__init__` resets the latch.
                        demo._safety_abort = True
                        print(f">>> arm plan[{tag}]: replan FAILED -- ABORTING at waypoint {i} "
                              f"(waypoint {bad} unsafe, no replan found)", flush=True)

                        _replan_failed = False
                        if tag == "abort-return":
                            _replan_failed = True
                        else:
                            # 1. Controlled stop (taper)
                            last_delta = getattr(demo, "_last_step_delta", np.zeros_like(q_full))
                            steps_taper = 96
                            q_taper = q_full.copy()
                            _taper_steps_ran = 0
                            _taper_travel = np.zeros_like(q_full)
                            _taper_break = False
                            for k_t in range(steps_taper):
                                f_t = 0.5 + 0.5 * math.cos(math.pi * (k_t + 1) / steps_taper)
                                step_delta = last_delta * f_t
                                q_taper = q_taper + step_delta

                                q8_taper = np.array([float(q_taper[demo.idx[n]]) for n in ArmModel.Q8])
                                if m.collides(q8_taper, obs, MARGIN, held, _b, fingers=_fing):
                                    _taper_break = True
                                    break

                                _taper_steps_ran += 1
                                _taper_travel += np.abs(step_delta)
                                q_taper = with_closure_forced(demo, q_taper)
                                if hold_base:
                                    demo.set_base(bx, by, byaw)
                                qv_taper = step_delta / demo.dt
                                demo._force(q_taper, qv_taper)
                                demo._apply(q_taper)
                                if on_step is not None:
                                    on_step()
                                demo.world.step(render=not HEADLESS)
                                q_full = q_taper.copy()
                            _taper_init = float(np.max(np.abs(last_delta)))
                            _taper_final = float(np.max(np.abs(step_delta))) if _taper_steps_ran > 0 else _taper_init
                            _trav_str = ", ".join(
                                f"{n}={_taper_travel[demo.idx[n]] * 1000.0:.3f}"
                                f"{'mm' if j in ArmModel.Q8_LIN else 'mrad'}"
                                for j, n in enumerate(ArmModel.Q8))
                            print(f">>> abort-taper: ran {_taper_steps_ran}/{steps_taper} steps, "
                                  f"travel [{_trav_str}], mag {_taper_init:.5f} -> "
                                  f"{_taper_final:.5f}, ended by "
                                  f"{'collision check' if _taper_break else 'completion'}", flush=True)

                            # `held` is the DECLARATION that there is a payload; `_focus_obj` is set at
                            # dock and never cleared, so alone it measures against a stale baseline.
                            obj = None
                            if held is not None and getattr(demo, "_focus_obj", None):
                                try:
                                    obj = demo.objs[int(demo._focus_obj.split("_")[-1])]
                                except (ValueError, LookupError, TypeError, AttributeError):
                                    obj = None
                            # A declared payload nobody can find cannot be verified: refuse the return.
                            _unverifiable = held is not None and obj is None
                            if _unverifiable:
                                demo._fallback("payload-unverifiable")
                                print(">>> abort-recovery: payload declared but no object to verify it "
                                      "on -- holding here", flush=True)

                            _payload_lost = False
                            if obj is not None:
                                slip = demo._carry_check(obj, "abort-stop")
                                if slip is not None:
                                    threshold = 5.0 # CHOSEN, not derived (no defensible in-hand band exists)
                                    if slip > threshold:
                                        demo._fallback("payload-lost")
                                        print(">>> PAYLOAD-LOST", flush=True)
                                        _payload_lost = True

                            if _payload_lost or _unverifiable:
                                _replan_failed = True
                            else:
                                # 3. Validated return to park
                                q0_8 = np.array([float(demo.q0[demo.idx[n]]) for n in ArmModel.Q8])
                                # RE-DERIVED, not `_fing`: the taper ran physics steps since that
                                # snapshot, so a stale value plans one hand and flies another.
                                _fing_park = commanded_fingers(demo) if hold_fingers else None
                                park_plan = plan_arm(demo, q0_8, held=held,
                                                     grasp_names=grasp_names,
                                                     fingers=_fing_park)
                                if park_plan is None:
                                    _replan_failed = True
                                else:
                                    print(">>> abort-recovery: executing return to park", flush=True)
                                    q_parked = execute_path(demo, park_plan, secs=secs, held=held,
                                                            checkpoint_every=checkpoint_every,
                                                            on_step=on_step, hold_fingers=hold_fingers,
                                                            grasp_names=grasp_names, tag="abort-return",
                                                            hold_base=hold_base)
                                    if q_parked is not None:
                                        q_full = q_parked
                                        demo._fallback("abort-recovered")
                                        demo._abort_recovered = True
                                    else:
                                        _replan_failed = True

                        if _replan_failed:
                            demo._fallback("execute-replan-failed")
                        return None
                    else:
                        replans += 1
                        demo._bench_replans = getattr(demo, "_bench_replans", 0) + 1
                        waypoints = list(waypoints[:i + 1]) + list(new[1:])
                        derate = q8_derate(waypoints, secs)
            # One retimed run per checkpoint interval: the arm rests where it re-validates, so a
            # splice never starts from a moving arm -- a velocity step is what `_apply` masks.
            j = min(len(waypoints) - 1, i + every)
            for q8 in retime_q8(waypoints[i:j + 1], demo.dt, derate)[1][1:]:
                q = q_full.copy()
                for n, v in zip(ArmModel.Q8, q8):
                    q[demo.idx[n]] = float(v)
                if fhold is not None:
                    q[demo._f_idx] = fhold
                q = with_closure_forced(demo, q)
                if hold_base:
                    demo.set_base(bx, by, byaw)
                qv = (q - q_prev) / demo.dt
                demo._last_step_delta = q - q_prev
                q_prev = q.copy()
                demo._force(q, qv)
                demo._apply(q)
                if on_step is not None:
                    on_step()
                demo.world.step(render=not HEADLESS)
                q_full = q
            i = j
        print(f">>> arm plan[{tag}]: executed {len(waypoints)} waypoints, {replans} replan(s)",
              flush=True)
        # Where the executor actually drove: a replan rebinds `waypoints` locally, and a
        # `regoal()` retarget ends that tail elsewhere, so the caller's list is not the target.
        demo._exec_goal8 = np.asarray(waypoints[-1], float).copy()
        return q_full
    finally:
        demo._finger_pin = pin_prev


def commanded_fingers(demo):
    """{finger joint: value the fingers are DRIVEN to}; the measured hand if nothing owns them;
    None if there are none (`first_hit` then uses the swept box). The drive target is constant
    while the grip owns the hand, so every checkpoint judges ONE hand (see FINDINGS)."""
    try:
        _cmd = getattr(demo, "_finger_cmd", None)
        if _cmd is not None:
            # `_f_idx` and `_finger_cmd` are parallel by construction, so the names come back
            # through the same index map rather than being re-derived from a name filter.
            _name = {i: n for n, i in demo.idx.items()}
            got = {_name[int(i)]: float(v)
                   for i, v in zip(demo._f_idx, np.asarray(_cmd, float).ravel())}
        else:
            # "finger" in n, NOT startswith("finger_"): `palm_finger_b/c_joint_1` swing the whole b and c
            # fingers without that prefix, and missing them leaves `fk` defaulting them to 0.0.
            q = np.asarray(demo.robot.get_joint_positions(), float)
            got = {n: float(q[demo.idx[n]]) for n in demo.idx if "finger" in n}
        # Arm TWO's finger joints match both paths, and `fk` drops them in silence -- its
        # `q.get(name, 0.0)` walks the MODEL's joints, which are arm one's.
        return {n: v for n, v in got.items() if n in demo._arm_model().joints} or None
    except Exception:                          # noqa: BLE001 -- geometry must not kill a plan
        return None
