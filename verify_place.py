# !/usr/bin/env python3
"""Grade a place run from its log: const-z, upright, shelf-board clearance, arm clearance, plus a
NaN gate and a final-grade gate. Exits non-zero if any check fails, so it can gate a visual check
instead of replacing one."""
import json
import os
import re
import sys

Z_TOL_MM = 25.0        # how far off the slide height counts as "not const-z"
TILT_TOL_DEG = 5.0     # how far off vertical counts as "tilted"
DRIFT_MAX = 10          # mm of base drift during place. A band, not a bisected floor: CHOSEN,
                        # not measured. Healthy is 0-3; 280 and 129 preceded the obj-6 abort.
RESULTS_XY_TOL_MM = 20.0  # Read off the STAGE, so tighter than precise_success's self-reported
                          # 30mm (morph/place/__init__.py:525). A historically good 29mm run FAILS
                          # here -- that is the intended verdict, not a regression.
RESULTS_Z_TOL_MM = RESULTS_XY_TOL_MM  # Stage-measured Z placement tolerance in mm.

# The hand's reach past the frame origin is NOT a constant, so insert.py accumulates the hand top
# every step and emits one summary. The per-waypoint prints are for a human and are NOT gated.
HAND_TOP_MAX_LINE = re.compile(r"insert\[slide\]: hand top max ([\d.]+) over \d+ steps")
TRACK_BEAT_LINE = re.compile(r">>> track-err\[[^\]]+\]:")

FALLBACK_TAGS = {
    "mouth-vertical": ("SAFETY", "mouth axis near vertical -> baked goal"),
    "goal-ik-miss": ("SAFETY", "IK miss -> baked goal"),
    "shift-too-large": ("SAFETY", "shift too large -> baked goal"),
    "shift-ik-miss": ("SAFETY", "IK miss for shift -> baked goal"),
    "goal-exception": ("SAFETY", "unavailable -> baked goal"),
    "plan-start-not-finite": ("SAFETY", "the arm's measured pose is not finite: nothing is planned from it"),
    "linear-start-not-finite": ("SAFETY", "the arm's measured pose is not finite: no line is checked or flown from it"),
    "no-plan": ("SAFETY", "no route for the reach: the planned motion refuses and the pick ends"),
    "plan-timeout": ("SAFETY", "the solve timed out: that plan is refused and nothing runs on it"),
    "densify-depth-cap": ("SAFETY", "the emitted path is under-refined, traded for less collision-checked"),
    "aim-ik-miss": ("SAFETY", "AIM IK miss -> unaimed goal"),
    "grasp-candidates-unavailable": ("SAFETY", "candidate generation raised; _safety_abort is not latched, so run_cycle retries and _recover() re-homes open-loop on a path nothing validated"),
    "execute-replan-failed": ("SAFETY", "replan FAILED; one call path discards the return value and an unchecked column raise can follow it (based on what the code permits, not the arithmetic coincidence that checkpoint_every = len(wps) + 1 makes the checkpoint never fire on that path today)"),
    "planner-off-approach": ("SAFETY", "planner off: the reach is neither planned nor flown, and the pick continues from where the arm is"),
    "plan-bad-obstacle": ("SAFETY", "an exemption reached the PLANNED tier, which A4 forbids; the plan is refused so nothing unchecked runs, but the request was malformed and the run must not read green"),
    "retreat-no-cartesian": ("OUTCOME", "no unchecked motion runs"),
    "retreat-no-remainder": ("OUTCOME", "no unchecked motion runs"),
    "checkpoint-obstacles-unavailable": ("SAFETY", "the executor could not build the obstacle set mid-motion; the motion was ended at the last flown waypoint with nothing validated past it"),
    "replan-goal-stale": ("OUTCOME", "the base moved and the world goal could not be re-derived, so nothing was replanned"),
    "servo-following-error": ("OUTCOME", "a gated joint stopped following its command during the insert"),
    "servo-proximal-blocked": ("OUTCOME", "the boom or a column would have collided during the insert servo"),
    "servo-watch-unbuildable": ("OUTCOME", "the insert servo's collision watch could not be built, so the insert refused"),
    "retreat-no-arrival": ("OUTCOME", "no unchecked motion runs"),
    "retreat-raised": ("OUTCOME", "no unchecked motion runs"),
    "settle-snap-too-far": ("SAFETY", "the settle refuses the teleport AND ends the attempt, but like settle-abort-no-teleport it does not latch _safety_abort, so run_cycle retries and _recover() may re-home open-loop on a path nothing validated -- same exposure, same class"),
    "reach-no-arrival": ("SAFETY", "the planned reach stopped short and the caller proceeded anyway: the reach-lowest column move and the seat run from the pose actually reached, not from the planned goal"),
    "bad-target": ("SAFETY", "a motion was asked for with a target that is not a joint vector -- nothing was planned, and the caller cannot tell that from a planner that found no route"),
    "reach-abort": ("OUTCOME", "ends the pick; nothing further runs on that attempt"),
    "retry-suppressed-after-abort": ("OUTCOME", "retry suppressed, cycle fails"),
    "lowest-staged": ("SAFETY", "staged fallback, retracts a1 open-loop"),
    "ramp-blocked": ("SAFETY", "a close-stage ramp target was inside MARGIN of a world box; no ramp ran and the stage continues from the pose reached. SAFETY, not OUTCOME: the shipped seats all clear this check, so a refusal means its scope is wrong and must not read green"),
    "ramp-unbuildable": ("SAFETY", "a close-stage ramp could not be judged -- no focus prim, no commanded hand, or a non-finite target -- so it refused. SAFETY deliberately, unlike the servo-*-unbuildable precedent: an unjudged close-stage ramp is exactly the evidence gap this check exists to close"),
    "dock-no-corridor": ("SAFETY", "no dock angle with a clear sweep corridor; attempt refused"),
    "no-baked-trajectory": ("SAFETY", "pose stage: no baked reach trajectory to re-solve; attempt refused"),
    "dock-invalid": ("SAFETY", "no valid dock at any angle; attempt refused"),
    "place-results-reset": ("SAFETY", "degrades acceptance evidence"),
    "place-results-open-failed": ("SAFETY", "leaves no record on disk"),
    "place-results-manifest-failed": ("SAFETY", "ungradeable run"),
    "abort-recovered": ("OUTCOME", "taper, payload check and validated return all complete"),
    "payload-lost": ("OUTCOME", "payload slipped out of band during abort stop"),
    "close-gate-unreachable": ("OUTCOME", "the close gate refused the pose: no squeeze, the attempt ends and run_cycle retries"),
    "close-no-asset": ("OUTCOME", "no recorded close trajectory: the close has no other close to run, so it refuses and the attempt ends"),
    "close-missed-grasp": ("OUTCOME", "the force close loaded no finger: nothing lifts on air, the attempt ends"),
    "close-partial-latch": ("OUTCOME", "the force close latched one or two fingers: a partial grip is not a hold, the attempt ends and run_cycle retries"),
    "lowest-tilt-diverged": ("OUTCOME", "the articulation diverged on the tilt ramp: the pose is restored and the pick is over"),
    "nav-no-plan": ("OUTCOME", "no base route to the dock: the drive refuses and the attempt ends"),
    "settle-base-shoved": ("OUTCOME", "the base was shoved off the dock during the reach: the attempt ends"),
    "descend-align-displaced": ("OUTCOME", "the object moved more than 50mm before the close: closing would grab air"),
    "capture-grasp-rejected": ("OUTCOME", "grasp quality out of band -- tilt, offset, loaded pads or penetration: no lift"),
    "payload-unverifiable": ("SAFETY", "abort stop with a declared payload no object resolves; no return to park"),
}


def _default_results_path(log_path=None):
    """One formula, not two: these used to diverge CWD-relative vs log-relative whenever the log
    lived outside CWD. results_gate only ever sees log TEXT, so with no log_path it anchors to CWD."""
    return os.path.join((os.path.dirname(log_path) if log_path else "") or ".", "place_results.json")


def grade_gate(txt):
    """EVERY cycle's grade must pass, not just the last: `findall(...)[-1]` let a nine-object run
    report ALL CHECKS PASS on eight failures and one success. Evidence is required the same way
    results_gate does it (`bool(x) and all(...)`), because a run that crashed before any cycle could
    grade itself has NO grade line, and an empty `all(...)` read exactly like a full pass."""
    grades = re.findall(r"-> (precise_success|precise_marginal|approx_fallback|failed)", txt)
    return bool(grades) and all(g in ("precise_success", "precise_marginal") for g in grades)


def fallback_gate(txt):
    """False if any cycle suffered a SAFETY degrade; OUTCOME degrades do not fail the gate.
    Requires positive evidence."""
    m_fb = re.findall(r">>> plan-fallbacks\[([^\]]+)\]: (\d+)(?: \(([^)]+)\))?", txt)
    if not m_fb:
        return False
    for phase, total, parts in m_fb:
        if parts:
            for part in parts.split(', '):
                k, v = part.split('=')
                if int(v) > 0:
                    cls = FALLBACK_TAGS.get(k, ("SAFETY",))[0]
                    if cls == "SAFETY":
                        return False
    return True


def drift_gate(txt):
    """Every cycle's chassis drift must stay within DRIFT_MAX; requires evidence."""
    drifts = re.findall(r"released [a-z-]+ \((\d+) drift", txt)
    return bool(drifts) and all(int(n) <= DRIFT_MAX for n in drifts)


def model_gate(txt):
    """False if Bearing_Column diverged from its kinematic model. `_arm_body_clearance` is called
    from two sites only -- pick/reach.py:378 (the computed descent) and close/lowest.py:227 (the
    a1 extend) -- so this covers the DESCENT and nothing else: not the planned path, not the
    carry, not the place. ABSENCE PASSES: a run that never entered those two loops gates green."""
    return "MODEL-PHYSICS DIVERGENCE" not in txt


TRACK_INSTRUMENT_FAILED = "TRACK INSTRUMENT FAILED"   # morph/robot.py:588 flushes this


def track_gate(txt):
    """False if the actuated chain ever failed to track its command, OR if the instrument that
    would have said so died. ABSENCE OF BOTH PASSES: `TRACK BREACH` prints only ON a breach, so a
    clean run and a dead readback look alike here. Unlike `model_gate` that is not the whole story
    -- the heartbeat branch at :383 fails a placed cycle that reported no `track-err[...]`.

    SCOPE: only stages inside ARM_DRIVE_STAGE have gated joints, and the two that execute a
    planned path -- `reach` and `settle` -- are not among them by default. A green track gate is
    therefore NOT evidence that the planned motion tracked."""
    return "TRACK BREACH" not in txt and TRACK_INSTRUMENT_FAILED not in txt


def _xy_err_mm(rec):
    """Euclidean xy error, in mm, between one results record's measured and target position."""
    dx = rec["xy"][0] - rec["target_xy"][0]
    dy = rec["xy"][1] - rec["target_xy"][1]
    return (dx * dx + dy * dy) ** 0.5 * 1000.0


def _z_err_mm(rec):
    """Absolute z error in mm between measured and target position."""
    return abs(rec["z"] - rec["target_z"]) * 1000.0


GRADE_LINE = re.compile(r"-> (precise_success|precise_marginal|approx_fallback|failed)")


MANIFEST_FAILED = "RUN MANIFEST WRITE FAILED"   # play_isaac.py:569 flushes this; tests pin both ends


def _load_records(path):
    """Return PLACE_RESULTS records, or None if unparseable or inaccessible."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _manifest_path(results_path):
    """Beside PLACE_RESULTS. play_isaac.py's writer derives it the same way, from the same env."""
    return os.environ.get("PLACE_RUN_MANIFEST") or results_path + ".run"


def _manifest_run(results_path):
    """Return the run id Demo.__init__ stamped before any cycle could run, or None if there is no
    readable manifest. The id must NOT be taken from the records: a run killed between its grade
    line -- printed in morph/place, before run_cycle writes anything -- and its record leaves the
    file holding only the PREVIOUS run's records, whose count can match this run's grade count."""
    try:
        with open(_manifest_path(results_path)) as f:
            return json.load(f).get("run")
    except (OSError, ValueError, AttributeError):
        return None


def results_gate(txt, path=None):
    """Compare stage-measured placement poses in PLACE_RESULTS to slot targets.

    Validates that records match manifest run ID, cycle count matches grade count,
    and XY/Z errors remain within thresholds."""
    path = path or os.environ.get("PLACE_RESULTS", _default_results_path())
    if MANIFEST_FAILED in txt:
        # the run could not name itself, so nothing on disk
        return False
                                            # is its evidence -- first, so no later branch
                                            # (absence, pre-writer) can rescue it
    if "!! cycle failed:" in txt:
        # run_cycle's sole except site in play_isaac.py -- printed on BOTH launchers
        return False
    if not os.path.exists(path):
        if os.path.exists(_manifest_path(path)):
            # `Demo.__init__` writes the manifest BEFORE cycle 1,
            return False
                                            # so a manifest with no records is a run that STARTED
                                            # and produced nothing, which the markers below cannot see.
        ran = (bool(CYCLE_BANNER.search(txt)) or bool(GRADE_LINE.search(txt))
               or bool(FALLBACK_LINE.search(txt)))
        # no file and no manifest: fail-closed unless the log
        return not ran
                                            # shows no sign of a cycle, i.e. it predates the
                                            # writer itself
    records = _load_records(path)
    if records is None or not isinstance(records, list):
        # unreadable or non-list evidence fails
        return False
    run = _manifest_run(path)
    if run is None:
        # a results file nothing claims is not evidence: only the
        return False
                                            # missing-file branch above may pass without a manifest
    if any(not isinstance(r, dict) or r.get("run") != run for r in records):
        # a previous run's record is not this run's evidence
        return False
    if len(records) != len(GRADE_LINE.findall(txt)):
        # record count must match grade count
        return False
    try:
        # non-empty list required
        return bool(records) and all(
            isinstance(r, dict)
            and r.get("ok", True)           # absent `ok` is a pre-stamp record: judge it on its numbers
            and _xy_err_mm(r) <= RESULTS_XY_TOL_MM and _z_err_mm(r) <= RESULTS_Z_TOL_MM
            for r in records)
    except KeyError:
        # schema missing target coordinates fails
        return False


CYCLE_BANNER = re.compile(r"^>>> ===== CYCLE (.+?) =====\s*$", re.M)


FALLBACK_LINE = re.compile(r">>> plan-fallbacks\[")


def split_cycles(txt):
    """Split a log into per-cycle segments on the `>>> ===== CYCLE ... =====` banner. A log with no
    banner (single-cycle run) comes back as one segment holding the whole text -- the standing
    baseline behaviour is not allowed to change. Preamble before the first banner (startup noise,
    one-time setup lines) is folded into the first segment rather than dropped."""
    banners = list(CYCLE_BANNER.finditer(txt))
    if not banners:
        return [("(whole log)", txt)]
    segments = []
    for i, m in enumerate(banners):
        start = m.start()
        end = banners[i + 1].start() if i + 1 < len(banners) else len(txt)
        seg = txt[:start] + txt[start:end] if i == 0 and start > 0 else txt[start:end]
        segments.append((m.group(1).strip(), seg))
    return segments


def check(path):
    txt = open(path, errors="replace").read()
    segments = split_cycles(txt)
    multi = len(segments) > 1
    ok = True
    # FAIL-CLOSED: absence of evidence fails the log, unless the log predates
    # the instruments entirely (indicated by absence of plan-fallbacks line).
    pre_instrument = not FALLBACK_LINE.search(txt)
    if pre_instrument:
        print("  fallback-gate: NOT RUN -- no plan-fallbacks line anywhere (pre-counter log)")
    for label, seg in segments:
        if multi:
            print(f"  ===== cycle: {label} =====")
        ok &= check_text(seg, label, pre_instrument=pre_instrument)
        if multi:
            print()
    # Whole-log evaluation: PLACE_RESULTS accumulates across cycles.
    # results_gate is the sole pass/fail authority; following block formats details.
    rpath = os.environ.get("PLACE_RESULTS", _default_results_path(path))
    good = results_gate(txt, rpath)
    ok &= good
    raw = _load_records(rpath) if os.path.exists(rpath) else None
    raw = raw if isinstance(raw, list) else []
    # Filter non-dict entries; discrepancies are reported in detail below.
    records = [r for r in raw if isinstance(r, dict)]
    crashed = [r for r in records if not r.get("ok", True)]
    run = _manifest_run(rpath)
    foreign = [r for r in records if r.get("run") != run]   # named, not counted: whose run is it
    if MANIFEST_FAILED in txt:
        detail = ("this run could not record its identity; any results file present belongs "
                  "to another run")
    elif len(records) != len(raw):
        detail = (f"{len(raw) - len(records)} of {len(raw)} entries in "
                  f"{os.path.basename(rpath)} are not records")
    elif records and run is None:
        detail = f"no run manifest ({os.path.basename(_manifest_path(rpath))})"
    elif foreign:
        detail = f"record(s) from another run (this run is {run}): " + ", ".join(
            f"obj {r.get('obj')} slot {r.get('slot')} run {r.get('run')}" for r in foreign)
    elif crashed:
        detail = "cycle(s) crashed before placement: " + ", ".join(
            f"obj {r.get('obj')} slot {r.get('slot')}" for r in crashed)
    elif "!! cycle failed:" in txt:
        detail = "the log reports a cycle failed (!! cycle failed:) that owns no record"
    elif records:
        try:
            worst_rec = max(records, key=_xy_err_mm)
            detail = (f"{len(records)} record(s), worst xy err {_xy_err_mm(worst_rec):.0f}mm "
                      f"(obj {worst_rec['obj']} slot {worst_rec['slot']})")
        except KeyError:
            detail = f"{os.path.basename(rpath)} has an old-schema record"
    elif os.path.exists(rpath):
        detail = f"{os.path.basename(rpath)} present but empty/unreadable"
    else:
        detail = f"no {os.path.basename(rpath)}"
    print(f"  results-gate: {detail} (tol {RESULTS_XY_TOL_MM:.0f}) -> {'PASS' if good else 'FAIL'}")
    return ok


def check_text(txt, label, pre_instrument=False):
    """`pre_instrument` exempts the ABSENCE checks below, and only those: it means the log predates
    the instruments entirely (see `check`). It never rescues a log that reported a failure."""
    ok = True

    # 1. const-z: the per-waypoint deviation from the slide height.
    devs = [int(m) for m in re.findall(r"z off slide ([-+]\d+)mm", txt)]
    pre = re.search(r"pre-dock height: object z -> ([\d.]+) \(slide ([\d.]+)", txt)
    print(f"  const-z: height set to {pre.group(1)}m for a slide of {pre.group(2)}m"
          if pre else "  const-z: NO pre-dock height line")
    if devs:
        worst = max(abs(d) for d in devs)
        verdict = "PASS" if worst <= Z_TOL_MM else "FAIL"
        ok &= worst <= Z_TOL_MM
        print(f"  const-z: {len(devs)} waypoints, worst deviation {worst}mm "
              f"(tol {Z_TOL_MM:.0f}) -> {verdict}    {devs}")
    else:
        ok = False
        print("  const-z: FAIL -- no waypoint height samples found")

    # 2. upright: every carry checkpoint, and what the object SETTLES at.
    # Judge settled attitude after backout rather than intermediate tilt during release.
    tilts = re.findall(r"carry\[([a-z-]+)\]:.*?upright ([\d.]+)deg", txt)
    if tilts:
        worst_tag, worst_deg = max(tilts, key=lambda t: float(t[1]))
        # the insert-entry spike is corrected by the leveller; judge what it is left at
        after = [(t, float(d)) for t, d in tilts if t in ("insert-end", "settled")]
        # Absence must not pass, the same rule arm-clear and the track gate use: a cycle that reached
        # `released` OWES a reading after the backout, and none at all is UNMEASURED, not upright.
        owed = (any(t == "released" for t, _ in tilts) and not pre_instrument
                and not any(t == "settled" for t, _ in after))
        good = bool(after) and not owed and all(d <= TILT_TOL_DEG for _, d in after)
        ok &= good
        print(f"  upright: {'  '.join(f'{t}={d}deg' for t, d in tilts)}")
        print(f"  upright: worst {worst_deg}deg at {worst_tag}; "
              f"left at {after} (tol {TILT_TOL_DEG:.0f}) -> {'PASS' if good else 'FAIL'}"
              + ("" if after else " -- no checkpoint the shelf can be judged on")
              + (" -- released but never measured again after the backout, so its final resting "
                 "attitude is unverified" if owed else ""))
    else:
        ok = False
        print("  upright: FAIL -- no carry checkpoints found")

    # 3. shelf-board clearance: the object must fit the opening it slides through, since a board strike
    # on a collision-free object passes silently. Half-height comes from the run's own `spans a..b`.
    HALF_H = 0.150
    _sp = re.search(r"pre-dock height: object z -> [\d.]+ \(slide [\d.]+, spans ([\d.]+)\.\.([\d.]+)", txt)
    if _sp:
        HALF_H = 0.5 * (float(_sp.group(2)) - float(_sp.group(1)))
    SURFACES = [0.247, 0.687, 1.190]
    UNDERSIDES_ABOVE = [0.677, 1.180, 1.676]     # board above each level
    m_slot = re.search(r"PLACE obj \d+ -> slot (\d+)", txt)
    if pre and devs and m_slot:
        slot = int(m_slot.group(1))
        lvl = 0 if slot <= 3 else (1 if slot <= 6 else 2)
        slide = float(pre.group(2))
        top = slide + max(devs) / 1000.0 + HALF_H
        bot = slide + min(devs) / 1000.0 - HALF_H
        c_up = (UNDERSIDES_ABOVE[lvl] - top) * 1000.0
        c_dn = (bot - SURFACES[lvl]) * 1000.0
        good = c_up > 20.0 and c_dn > 10.0
        ok &= good
        print(f"  shelf-clear: level {('low','mid','high')[lvl]} opening "
              f"{SURFACES[lvl]:.3f}..{UNDERSIDES_ABOVE[lvl]:.3f} | object spans {bot:.3f}..{top:.3f} "
              f"at worst -> {c_up:.0f}mm below the board above, {c_dn:.0f}mm above the board it "
              f"lands on -> {'PASS' if good else 'FAIL'}")
    else:
        ok = False
        print("  shelf-clear: FAIL -- missing slide height, waypoints, or slot id")

    # 4. arm clearance: the hand must clear the board above through the whole slide. Check 3 covers the
    # payload, this covers the robot, and insert.py's one summary makes whole-slide literal.
    ht = [float(m) for m in HAND_TOP_MAX_LINE.findall(txt)]
    if ht and m_slot:
        lvl_h = 0 if int(m_slot.group(1)) <= 3 else (1 if int(m_slot.group(1)) <= 6 else 2)
        c_arm = (UNDERSIDES_ABOVE[lvl_h] - max(ht)) * 1000.0
        good_a = c_arm > 20.0
        ok &= good_a
        print(f"  arm-clear: hand top max {max(ht):.3f} (whole slide, every step, this wrist) "
              f"vs board underside "
              f"{UNDERSIDES_ABOVE[lvl_h]:.3f} -> {c_arm:.0f}mm -> "
              f"{'PASS' if good_a else 'FAIL'}")
    elif m_slot and not pre_instrument:
        # Absence of hand-top maximum in instrumented logs fails the check.
        ok = False
        print("  arm-clear: FAIL -- this cycle placed but reported no whole-slide hand-top "
              "maximum, so its clearance under the board is unverified")
    else:
        print("  arm-clear: no whole-slide hand-top maximum (pre-instrument log) -- not gated")

    # Hard gates, not context lines: a run that NaN'd mid-place still reported ALL CHECKS PASS while
    # the grade and the ok flag were only printed rather than asserted.
    if re.search(r"residual nanmm|upright nandeg|xy err nanmm|\[nan, nan, nan\]", txt):
        ok = False
        print("  nan-gate: FAIL -- NaN in a measured quantity (physics explosion)")
    m_g = re.findall(r"-> (precise_success|precise_marginal|approx_fallback|failed)", txt)
    if m_g and not grade_gate(txt):
        ok = False
        bad = [g for g in m_g if g not in ("precise_success", "precise_marginal")]
        print(f"  grade-gate: FAIL -- {len(bad)}/{len(m_g)} cycle(s) failed: {bad}")
    m_fb = re.findall(r">>> plan-fallbacks\[([^\]]+)\]: (\d+)(?: \(([^)]+)\))?", txt)
    if m_fb:
        safety_bad = []
        outcome_bad = []
        for phase, total, parts in m_fb:
            if parts:
                for part in parts.split(', '):
                    k, v = part.split('=')
                    if int(v) > 0:
                        cls = FALLBACK_TAGS.get(k, ("SAFETY",))[0]
                        if cls == "SAFETY":
                            safety_bad.append((phase, k, v))
                        else:
                            outcome_bad.append((phase, k, v))
        if safety_bad:
            ok = False
            print(f"  fallback-gate: FAIL -- SAFETY tags degraded (zero tolerance): {safety_bad}")
        elif outcome_bad:
            print(f"  fallback-gate: PASS -- but OUTCOME tags recorded FAILED CYCLES: {outcome_bad}")
        else:
            # No bad tags
            pass
    elif not pre_instrument:
        ok = False
        print("  fallback-gate: FAIL -- this cycle reported no tally, so its degrades are unverified")
    m_dr = [int(n) for n in re.findall(r"released [a-z-]+ \((\d+) drift", txt)]
    if m_dr and not drift_gate(txt):
        ok = False
        bad = [n for n in m_dr if n > DRIFT_MAX]
        print(f"  drift-gate: FAIL -- {len(bad)} cycle(s) over {DRIFT_MAX}: {bad}")
    m_md = re.findall(r"MODEL-PHYSICS DIVERGENCE\[([^\]]+)\]", txt)
    if m_md and not model_gate(txt):
        ok = False
        print(f"  model-gate: FAIL -- {len(m_md)} tag(s) diverged from the kinematic model: {m_md}")
    if TRACK_INSTRUMENT_FAILED in txt:
        # Its own unconditional `if`, and FIRST: the breach/heartbeat pair below is an if/elif, and a run
        # whose readback died mid-way would slip past both. Presence is what this fails on.
        ok = False
        print("  track-gate: FAIL -- the tracking instrument failed (TRACK INSTRUMENT FAILED), "
              "so this run's command tracking is unmeasured whatever else it reported")
    m_tr = re.findall(r"TRACK BREACH\[[^\]]+\]: (\S+)", txt)
    if m_tr and not track_gate(txt):
        ok = False
        print(f"  track-gate: FAIL -- {len(m_tr)} joint(s) failed to track their command: {m_tr}")
    elif m_slot and not pre_instrument and not TRACK_BEAT_LINE.search(txt):
        # `TRACK BREACH` prints only ON a breach, so its absence looks the same for a clean run and a dead
        # instrument. The heartbeat is the evidence the check RAN.
        ok = False
        print("  track-gate: FAIL -- this cycle placed but reported no track-err[...] heartbeat, "
              "so its command tracking is unverified (TRACK_ERR off, or the readback is failing)")

    # Context that decides whether the above is even meaningful.
    for pat, lbl in ((r"PLACE obj \d+ -> slot \d+ .* ok=(\w+)", "place ok"),
                     (r"-> (precise_success|precise_marginal|approx_fallback|failed)", "grade"),
                     (r"EASE FALLBACK", "ease fallback (should be absent)"),
                     (r"released place-end \((\d+) drift", "chassis drift re-asserts")):
        m = re.findall(pat, txt)
        print(f"  {lbl}: {m[-1] if m else 'none'}")
    return ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} LOGFILE [LOGFILE ...]", file=sys.stderr)
        sys.exit(2)
    allok = True
    for p in sys.argv[1:]:
        print(f"=== {p.split('/')[-1]}")
        allok &= check(p)
        print()
    print("ALL CHECKS PASS" if allok else "SOME CHECKS FAILED")
    sys.exit(0 if allok else 1)
