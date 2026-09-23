"""Collision-aware goal repair -- `ArmModel.ik_repair` and the primitives it scores on. Pure
numpy, no Isaac/OMPL: `.venv/bin/python3 tests/test_ik_repair.py`. The obstacle set here is the
two chassis boxes only; arm-2, the ~121 world boxes and self-collision are not covered offline."""
import inspect
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from morph.arm.model import ArmModel, IK_REPAIR_TRAVEL_K, IK_SEED_SPREAD, _rot_vec  # noqa: E402
from morph.arm.collision import Q8_TOL_M, Q8_TOL_RAD                              # noqa: E402
from morph.config import MOUTH_VEC_HAND                                      # noqa: E402

M = ArmModel.load(os.path.join(ROOT, "usd/_arm_model.json"),
                  os.path.join(ROOT, "usd/_closure_lut.json"))

# Both boxes are tagged "chassis": axis-aligned in the BASE frame, the frame `collides` evaluates
# the links in, so `base=None` below is the identical computation at any base pose.
BODY = [(np.array([[-0.3165, -0.2960, 0.1586], [-0.0219, 0.2809, 0.3310]]), "chassis"),
        (np.array([[-0.350, -0.300, 0.139], [0.350, 0.300, 0.159]]), "chassis")]
MARGIN = 0.02                                                   # `morph/arm/collision.py` MARGIN

# The shipped low-slot seat after the 28 mm lower and the settle: level boom (dh 0), wrist home.
SEAT = np.array([0.099, 0.043, 0.043, 0.341, 0.0, 0.0, 0.0, 0.0])


def _shipped_poses():
    """(name, q8) for the grasp config and all 405 baked frames, in Q8 order."""
    cols = list(ArmModel.Q8)
    rows = [("grasp", np.array([json.load(open(os.path.join(ROOT, "usd/_grasp_known.json")))
                                ["arm_joints"][n] for n in cols], float))]
    for name in ("reach", "lift", "place_low", "place_mid", "place_high"):
        tr = json.load(open(os.path.join(ROOT, f"usd/_traj_{name}.json")))
        ix = [tr["joints"].index(n) for n in cols]
        rows += [(f"{name}[{k}]", np.array([fr[i] for i in ix], float))
                 for k, fr in enumerate(tr["frames"])]
    return rows


def _filler(n):
    """`n` obstacle boxes: `BODY` plus far-away filler. For cost measurement only."""
    return BODY + [(np.array([[5.0 + i, 5.0, 5.0], [5.5 + i, 5.5, 5.5]]), "world")
                   for i in range(max(0, n - 2))]


def test_the_seat_pose_collides_and_its_own_solution_set_does_not():
    """The seat's commanded pose interpenetrates the body at margin 0, while the identical hand
    pose has IK solutions that clear it."""
    pos, R = M.fk(SEAT)[M.hand_link]
    assert [round(float(v), 4) for v in pos] == [0.6456, 0.1942, 0.2706], pos
    assert M.collides(SEAT, BODY, 0.0) is True
    assert M.clearance(SEAT, BODY) == -1.0

    lo, hi = M.bounds()
    rng = np.random.default_rng(0)
    sols = [M.ik(pos, R=R, seed=rng.uniform(lo, hi), restarts=0, rng=rng) for _ in range(24)]
    sols = [s[0] for s in sols if s is not None]
    clear = [q for q in sols if not M.collides(q, BODY, MARGIN)]
    assert sols and clear, (len(sols), len(clear))


def test_it_repairs_the_seat_where_plain_ik_returns_the_colliding_branch():
    """Plain `ik` seeded at the seat converges in one step onto the colliding branch; `ik_repair`
    returns a clear member of the same solution set, at the same hand pose."""
    pos, R = M.fk(SEAT)[M.hand_link]
    plain = M.ik(pos, R=R, seed=SEAT, restarts=0)
    assert plain is not None and M.collides(plain[0], BODY, MARGIN) is True

    best = M.ik_repair(pos, R, obstacles=BODY, q_now=SEAT, rng=np.random.default_rng(0))
    assert best is not None
    q, ep, er = best
    assert M.collides(q, BODY, MARGIN) is False
    assert M.clearance(q, BODY) > MARGIN
    assert ep < 1e-4 and er < 1e-3, (ep, er)                    # the hand pose is HELD, not traded


def test_a_clear_goal_comes_back_untouched_and_costs_exactly_one_ik():
    q = np.array([json.load(open(os.path.join(ROOT, "usd/_grasp_known.json")))
                  ["arm_joints"][n] for n in ArmModel.Q8], float)
    pos, R = M.fk(q)[M.hand_link]
    # the premise: this goal is fine
    assert M.collides(q, BODY, MARGIN) is False

    calls = []
    real = ArmModel.ik
    try:
        ArmModel.ik = lambda self, *a, **k: (calls.append(1), real(self, *a, **k))[1]
        sol = M.ik_repair(pos, R, obstacles=BODY, q_now=q, rng=np.random.default_rng(3))
    finally:
        ArmModel.ik = real
    assert len(calls) == 1, len(calls)
    assert np.array_equal(sol[0], real(M, pos, R, seed=q)[0])
    assert np.array_equal(sol[0], q)


def test_the_goal_no_longer_jumps_branches_when_q_now_moves_a_little():
    """A clear goal IS `q_now`'s own solution, so the goal is exactly as continuous as `q_now`:
    zero deviation at three points 4.1 mm apart."""
    tr = json.load(open(os.path.join(ROOT, "usd/_traj_place_low.json")))
    ix = [tr["joints"].index(n) for n in ArmModel.Q8]
    q = np.array([tr["frames"][len(tr["frames"]) // 2][i] for i in ix], float)
    for d in (0.0, 0.0041, 0.0082):
        qq = q.copy()
        qq[1] += d
        qq[2] += d
        sol = M.ik_repair(*M.fk(qq)[M.hand_link], obstacles=BODY, q_now=qq)
        assert float(np.max(np.abs(sol[0] - qq))) == 0.0, (d, sol[0] - qq)


def test_a_free_orientation_solve_is_impossible():
    """`R` is a required positional argument rather than a runtime check: the score has no
    orientation term, so a defaulted `R=None` scored a freely rotated hand as best."""
    pos, _ = M.fk(SEAT)[M.hand_link]
    try:
        M.ik_repair(pos)
    except TypeError as e:
        assert "'R'" in str(e), e
    else:
        raise AssertionError("ik_repair accepted a free-orientation solve")
    assert "R" in inspect.signature(M.ik_repair).parameters
    assert inspect.signature(M.ik_repair).parameters["R"].default is inspect.Parameter.empty


def test_clearance_has_no_margin_argument_because_it_never_read_one():
    assert "margin" not in inspect.signature(M.clearance).parameters
    q = np.array([-0.572, 0.422, 0.480, 0.347, 0.020, -1.877, -0.431, 0.772])
    board = [(np.array([[-3.227, -2.594, 0.559], [3.903, -0.131, 0.572]]), "world")]
    assert round(M.clearance(q, board) * 1000.0, 1) == 42.8


def test_limit_margin_reads_the_bounds_it_is_given():
    """It read `self.bounds()` -- the raw USD limits -- while callers plan against the
    COLUMN_MAX-capped bounds."""
    q = np.array([0.0, 1.395, 1.395, 0.25, 0.0, 0.0, 0.0, 0.0])
    lo, hi = M.bounds()
    usd_lin, _ = M.limit_margin(q)
    capped = hi.copy()
    # COLUMN_MAX
    capped[1] = capped[2] = 1.40
    # A1_MAX
    capped[3] = 0.50
    plan_lin, _ = M.limit_margin(q, bounds=(lo, capped))
    assert round(usd_lin, 3) == 0.035, usd_lin
    assert round(plan_lin, 3) == 0.005, plan_lin


def test_limit_margin_sees_the_passive_stop_the_eight_joints_cannot():
    """The closure can rest `ArmRightJoint_1` on its stop while all eight commanded joints sit
    mid-range. No production caller; kept as the only measure of distance to a passive stop."""
    q = np.array([0.0, 0.60, 0.60, 0.50, 0.0, 0.0, 0.0, 0.0])
    lin, rot = M.limit_margin(q)
    lo, hi = M.bounds()
    active = float(np.minimum(q - lo, hi - q)[ArmModel.Q8_LIN].min())
    # what the eight joints alone say
    assert round(active, 3) == 0.125
    # what the closure actually leaves
    assert round(lin, 3) == 0.025
    # the revolutes are nowhere near
    assert rot > 1.0


def test_the_limit_term_decided_nothing_on_the_family_that_would_have_used_it():
    """Why the third score term was deleted rather than fixed: over the 13 poses that can now
    reach the score it saturates at 1.0 in every scored survivor and never moves a winner."""
    cases = [("seat", SEAT)]
    for traj, ks in (("reach", (77, 78, 79, 80)), ("lift", tuple(range(8)))):
        tr = json.load(open(os.path.join(ROOT, f"usd/_traj_{traj}.json")))
        ix = [tr["joints"].index(n) for n in ArmModel.Q8]
        cases += [(f"{traj}[{k}]", np.array([tr["frames"][k][i] for i in ix], float)) for k in ks]

    scored, saturated, differed = 0, 0, 0
    for name, q in cases:
        pos, R = M.fk(q)[M.hand_link]
        # the premise: these need repair
        assert M.collides(q, BODY, MARGIN) is True, name
        rng = np.random.default_rng(0)
        lo, hi = M.bounds()
        rows = []
        # the same draws `ik_repair` makes
        for _ in range(8):
            seed = np.clip(q + rng.normal(0.0, IK_SEED_SPREAD, 8) * (hi - lo), lo, hi)
            c = M.ik(pos, R, seed=seed, restarts=0, rng=rng)
            if c is None or M.collides(c[0], BODY, MARGIN):
                continue
            lin, rot = M.limit_margin(c[0])
            d_lin, d_rot = M.q8_resid(c[0], q)
            two = (min(M.clearance(c[0], BODY) - MARGIN, MARGIN) / MARGIN
                   - max(d_lin / Q8_TOL_M, d_rot / Q8_TOL_RAD) / IK_REPAIR_TRAVEL_K)
            limit = min(lin / Q8_TOL_M, rot / Q8_TOL_RAD, 1.0)
            rows.append((c[0], two, two + limit, limit))
        assert rows, name
        scored += len(rows)
        saturated += sum(r[3] >= 1.0 for r in rows)
        w2 = max(rows, key=lambda r: r[1])[0]
        w3 = max(rows, key=lambda r: r[2])[0]
        differed += (not np.array_equal(w2, w3))
    # 31 before the IK iterate was held to the dh-dependent a1 ceiling: 23 more seeds now converge
    # to a holdable pose instead of being rejected after convergence.
    assert differed == 0, (scored, saturated, differed)
    assert (scored, saturated) == (54, 54), (scored, saturated)


def test_first_hit_names_the_offender_and_collides_is_exactly_its_bool():
    """One scan, two consumers: `collides` is `first_hit(...) is not None`, so they cannot
    disagree, and the log line gets the offending link name."""
    assert M.first_hit(SEAT, BODY, MARGIN) == "Arm_Left_1"
    for q in (SEAT, np.array([0.0, 0.60, 0.60, 0.25, 0.0, 0.0, 0.0, 0.0])):
        for m in (0.0, 0.02, 0.10):
            assert M.collides(q, BODY, m) is (M.first_hit(q, BODY, m) is not None)


def test_clearance_bisects_collides_and_is_not_the_pessimistic_aabb_gap():
    """`clearance >= margin` is exactly `not collides(margin)`, and the oriented gap (42.8 mm) is
    not the pessimistic link-AABB gap (18.7 mm, inside the planner's own 20 mm margin)."""
    q8 = np.array([-0.572, 0.422, 0.480, 0.347, 0.020, -1.877, -0.431, 0.772])
    board = [(np.array([[-3.227, -2.594, 0.559], [3.903, -0.131, 0.572]]), "world")]
    aabb = M.link_aabbs(q8)["Arm_Left_1"]
    gap = float(np.max(np.maximum(board[0][0][0] - aabb[1], aabb[0] - board[0][0][1])))
    clr = M.clearance(q8, board)
    assert round(gap * 1000.0, 1) == 18.7, gap * 1000.0
    assert round(clr * 1000.0, 1) == 42.8, clr * 1000.0
    assert M.collides(q8, board, MARGIN) is False
    for m in (0.02, 0.10, 0.20):
        assert (clr >= m) is (not M.collides(q8, board, m)), m


def test_none_rather_than_an_unscored_colliding_branch():
    """When nothing survives, None -- the contract both wired call sites (`pick/approach.py`
    and `:568`, via `_goal_ik`) read as "use the baked goal"."""
    pos, R = M.fk(SEAT)[M.hand_link]
    everywhere = [(np.array([[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]]), "chassis")]
    assert M.ik(pos, R=R, seed=SEAT, restarts=0) is not None
    assert M.ik_repair(pos, R, obstacles=everywhere, q_now=SEAT,
                       rng=np.random.default_rng(0)) is None


def test_the_repair_is_reproducible_under_a_pinned_rng():
    """The planner pins `ARM_SEED`, so a sampled goal has to be reproducible or the route is not.
    Reproducible is not stable: across different rngs the clearance still spans ~180 mm."""
    pos, R = M.fk(SEAT)[M.hand_link]
    a = M.ik_repair(pos, R, obstacles=BODY, q_now=SEAT, rng=np.random.default_rng(7))
    b = M.ik_repair(pos, R, obstacles=BODY, q_now=SEAT, rng=np.random.default_rng(7))
    c = M.ik_repair(pos, R, obstacles=BODY, q_now=SEAT)         # the default_rng(0) `ik` also uses
    d = M.ik_repair(pos, R, obstacles=BODY, q_now=SEAT)
    assert np.array_equal(a[0], b[0]) and np.array_equal(c[0], d[0])
    spread = [M.clearance(M.ik_repair(pos, R, obstacles=BODY, q_now=SEAT,
                                      rng=np.random.default_rng(r))[0], BODY) for r in range(10)]
    assert max(spread) - min(spread) > 0.10, spread


def test_the_cost_scales_with_the_obstacle_count():
    """With the IK descents gone from the common path the box scan is all that is left, so the
    clear path is linear in the obstacle count (~1 ms at 2, ~7 ms at 123). The bound is loose
    on purpose -- shared dev machine, not a benchmark rig."""
    tr = json.load(open(os.path.join(ROOT, "usd/_traj_reach.json")))
    ix = [tr["joints"].index(n) for n in ArmModel.Q8]
    qs = [np.array([fr[i] for i in ix], float) for fr in tr["frames"][:40]]
    ms = {}
    for n in (2, 123):
        obs = _filler(n)
        t0 = time.perf_counter()
        for q in qs:
            M.ik_repair(*M.fk(q)[M.hand_link], obstacles=obs, q_now=q, margin=MARGIN)
        ms[n] = (time.perf_counter() - t0) / len(qs) * 1000.0
    print(f"    clear path: {ms[2]:.2f} ms at 2 obstacles, {ms[123]:.2f} ms at 123")
    assert ms[123] < 50.0, ms
    assert ms[123] > ms[2], ms


def test_the_log_tells_repaired_from_already_clear():
    """`log=True` prints ONE line and the three outcomes must be distinguishable in `live.log`."""
    import contextlib
    import io

    def line(**kw):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            M.ik_repair(log=True, **kw)
        return buf.getvalue().strip()

    pos, R = M.fk(SEAT)[M.hand_link]
    q_ok = np.array([json.load(open(os.path.join(ROOT, "usd/_grasp_known.json")))
                     ["arm_joints"][n] for n in ArmModel.Q8], float)
    fixed = line(pos=pos, R=R, obstacles=BODY, q_now=SEAT, rng=np.random.default_rng(0))
    clean = line(pos=M.fk(q_ok)[M.hand_link][0], R=M.fk(q_ok)[M.hand_link][1],
                 obstacles=BODY, q_now=q_ok)
    stuck = line(pos=pos, R=R, q_now=SEAT, rng=np.random.default_rng(0),
                 obstacles=[(np.array([[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]]), "chassis")])
    for got in (fixed, clean, stuck):
        assert got.count("\n") == 0 and got.startswith(">>> ik-repair:"), got
    assert "REPAIRED" in fixed and "Arm_Left_1" in fixed
    assert "OVERLAPPING -> 126.9 mm" in fixed, fixed
    assert "travel" in fixed
    assert "already clear" in clean and "REPAIRED" not in clean
    assert "NO CLEAR SOLUTION" in stuck


def test_every_shipped_pose_still_solves_and_the_colliding_ones_come_back_clear():
    """The gate: all 406 shipped poses (grasp config plus 405 baked frames) must solve, the 394
    already clear must come back bit-identical, and the 12 that violate the planner's 20 mm
    margin must come back clear."""
    rows = _shipped_poses()
    assert len(rows) == 406, len(rows)
    t0 = time.perf_counter()
    misses, before, after, kept = [], [], 0, 0
    for name, q in rows:
        pos, R = M.fk(q)[M.hand_link]
        if M.collides(q, BODY, MARGIN):
            before.append(name)
        sol = M.ik_repair(pos, R, obstacles=BODY, q_now=q, rng=np.random.default_rng(0))
        if sol is None:
            misses.append(name)
            continue
        qb, ep, er = sol
        after += M.collides(qb, BODY, MARGIN)
        assert ep < 1e-4 and er < 1e-3, (name, ep, er)
        # BIT-identical, not just within tol
        kept += np.array_equal(qb, q)
    print(f"    [{time.perf_counter() - t0:.0f}s] solved {len(rows) - len(misses)}/{len(rows)}, "
          f"kept {kept}, shipped-colliding {len(before)} -> {after}")
    assert misses == [], misses
    assert len(before) == 12 and after == 0, (before, after)
    assert kept == 394, kept


def test_ik_candidates_returns_the_whole_clear_set_best_first():
    """The planner needs the SET, not the argmax. `ik_repair` is the single-answer
    consumer of the same call, so its answer must be the set's first entry -- if those two ever
    disagree the planner and the servo are choosing different goals for the same pose."""
    pos, R = M.fk(SEAT)[M.hand_link]
    cands = M.ik_candidates(pos, R, obstacles=BODY, q_now=SEAT, seeds=24, margin=MARGIN)
    assert len(cands) > 1, f"expected several clear branches, got {len(cands)}"
    scores = [c[2] for c in cands]
    assert scores == sorted(scores, reverse=True), scores
    for (q8, _ep, _er), clr, _s in cands:
        assert not M.collides(q8, BODY, MARGIN), "a colliding branch reached the candidate set"
        assert clr >= 0.0, clr
    best = M.ik_repair(pos, R, obstacles=BODY, q_now=SEAT, seeds=24, margin=MARGIN)
    assert best is not None
    assert np.allclose(best[0], cands[0][0][0]), "ik_repair disagrees with its own candidate set"


# The grasp-candidate SET: `ik_candidates` varies the JOINT BRANCH at a fixed hand pose, and this
# is the other axis -- several approach YAWS, each solved, checked and ordered best-first.

GRASP = np.array([json.load(open(os.path.join(ROOT, "usd/_grasp_known.json")))
                  ["arm_joints"][n] for n in ArmModel.Q8], float)


def _nominal_grasp(standoff=0.13):
    """(hand pos, hand R, object centre) in the ARM-ROOT frame, from the shipped grasp config.
    The object sits `standoff` m out along the planar mouth axis -- `MOUTH_VEC_HAND` points OUT
    of the mouth (object -> palm), so the object is at `pos - standoff * mouth`."""
    pos, R = M.fk(GRASP)[M.hand_link]
    mouth = R @ np.asarray(MOUTH_VEC_HAND, float)
    mouth[2] = 0.0
    mouth /= float(np.linalg.norm(mouth))
    return pos, R, pos - standoff * mouth


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], float)


def test_grasp_candidates_span_several_approach_yaws_not_one_pose():
    """`ik_candidates` returns branches at ONE hand pose; object 6 failed
    twice identically because nothing offered a DIFFERENT approach. The set must carry more than
    one yaw, and cost a sweep, not an explosion -- it multiplies `ik_candidates` by the fan size."""
    pos, R, obj = _nominal_grasp()
    t0 = time.perf_counter()
    cands = M.grasp_candidates(pos, R, obj, obstacles=BODY, q_now=GRASP, margin=MARGIN)
    dt = time.perf_counter() - t0
    yaws = sorted({round(float(y), 9) for _c, _clr, _s, y in cands})
    print(f"    [{dt:.1f}s] {len(cands)} candidates over {len(yaws)} yaw(s): "
          f"{[round(math.degrees(y), 1) for y in yaws]}")
    assert len(yaws) > 1, f"only one approach yaw offered: {yaws}"
    assert 0.0 in yaws, f"the nominal yaw is not in the set: {yaws}"
    assert any(abs(y) > 1e-6 for y in yaws), "every candidate is the nominal yaw again"
    assert dt < 20.0, f"the yaw sweep took {dt:.1f}s"


def test_grasp_candidates_are_ordered_best_first():
    """The MODEL's contract, which is score order and nothing else -- the same score as
    `ik_candidates`, so a caller reading `cands[0]` gets the best-scoring member and not an
    arbitrary one. It is not the walk order: the pick-stage consumer re-sorts by |yaw| precisely
    because this score carries no yaw term (`ApproachStage._grasp_candidates`)."""
    pos, R, obj = _nominal_grasp()
    cands = M.grasp_candidates(pos, R, obj, obstacles=BODY, q_now=GRASP, margin=MARGIN)
    scores = [c[2] for c in cands]
    assert len(scores) > 1, len(scores)
    assert scores == sorted(scores, reverse=True), scores


def test_every_grasp_candidate_clears_the_planners_own_checker():
    """`arm_plan.py:43` is `not m.collides(q, obs, margin, held, base)` and `ArmModel.collides` is
    the SOLE collision authority for the arm. A candidate the planner would reject on sight is
    worse than no candidate: it burns the one retry the safety abort now forbids."""
    pos, R, obj = _nominal_grasp()
    cands = M.grasp_candidates(pos, R, obj, obstacles=BODY, q_now=GRASP, margin=MARGIN)
    assert cands
    for (q8, _ep, _er), clr, _s, _y in cands:
        assert not M.collides(q8, BODY, MARGIN), "a colliding candidate reached the set"
        assert clr >= 0.0, clr


def test_each_grasp_candidate_is_the_same_grasp_rigidly_orbited():
    """A yaw candidate is only a GRASP if the whole hand orbited: rotate position but not
    orientation and the mouth stops pointing at the object -- a pose near the object, not a grasp
    of it. Undo the yaw and every candidate must land back on the nominal hand pose."""
    pos, R, obj = _nominal_grasp()
    cands = M.grasp_candidates(pos, R, obj, obstacles=BODY, q_now=GRASP, margin=MARGIN)
    assert cands
    off = [c for c in cands if abs(c[3]) > 1e-6]
    assert off, "no rotated candidate to check"
    for (q8, _ep, _er), _clr, _s, y in cands:
        p_f, R_f = M.fk(q8)[M.hand_link]
        Ry = _rz(y)
        dp = float(np.linalg.norm(Ry.T @ (p_f - obj) - (pos - obj)))
        dr = float(np.linalg.norm(_rot_vec(Ry.T @ R_f @ R.T)))
        assert dp < 2e-4, (math.degrees(y), dp)
        assert dr < 2e-3, (math.degrees(y), dr)


def test_a_blocked_grasp_yields_an_empty_set_not_a_colliding_candidate():
    """Same contract as `ik_repair`'s None: nothing, so the caller falls back -- never a branch
    that collides. A slab through the whole approach volume blocks every yaw at once."""
    pos, R, obj = _nominal_grasp()
    wall = (np.array([[0.30, -0.80, -0.20], [1.30, 0.80, 0.90]]), "world")
    cands = M.grasp_candidates(pos, R, obj, obstacles=BODY + [wall], q_now=GRASP, margin=MARGIN)
    assert cands == [], f"{len(cands)} candidates inside a solid block"


def test_the_nominal_yaw_slice_is_exactly_todays_ik_candidates():
    """The set is a strict SUPERSET of today's answer, not a replacement for it: at yaw 0 it must
    reproduce `ik_candidates` at the nominal pose, entry for entry. That is what makes the set
    safe to wire -- the tuned grasp is still tried first and is still the same solve."""
    pos, R, obj = _nominal_grasp()
    base = M.ik_candidates(pos, R, obstacles=BODY, q_now=GRASP, margin=MARGIN,
                           rng=np.random.default_rng(0))
    cands = M.grasp_candidates(pos, R, obj, obstacles=BODY, q_now=GRASP, margin=MARGIN,
                               rng=np.random.default_rng(0))
    nom = [c for c in cands if abs(c[3]) < 1e-12]
    assert len(nom) == len(base), (len(nom), len(base))
    for (q8, _e, _r), clr, s in base:
        hit = [n for n in nom if np.array_equal(n[0][0], q8)]
        assert len(hit) == 1, f"nominal branch {np.round(q8, 4).tolist()} not reproduced"
        assert hit[0][1] == clr and hit[0][2] == s, (hit[0][1], clr, hit[0][2], s)


# no pytest in the venv? run direct.
if __name__ == "__main__":
    fails = 0
    for name in sorted(n for n in dir() if n.startswith("test_")):
        try:
            globals()[name]()
            print(f"PASS {name}")
        except Exception as e:
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{'all passed' if not fails else str(fails) + ' failed'}")
    sys.exit(1 if fails else 0)
