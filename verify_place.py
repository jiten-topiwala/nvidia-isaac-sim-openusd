# !/usr/bin/env python3
"""Verify three properties of a place run, from its log:

  1. CONST-Z: once the height is set for the slot, the object holds it until it is lowered.
  2. UPRIGHT: the object is never meaningfully tilted at any measured checkpoint.

Exits non-zero if either fails, so it can gate a visual check instead of replacing one.
"""
import re
import sys

Z_TOL_MM = 25.0        # how far off the slide height counts as "not const-z"
TILT_TOL_DEG = 5.0     # how far off vertical counts as "tilted"


def check(path):
    txt = open(path, errors="replace").read()
    ok = True

    # ---- 1. const-z: the per-waypoint deviation from the slide height ----
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

    # ---- 2. upright: every carry checkpoint, and what the insert leaves ----
    tilts = re.findall(r"carry\[([a-z-]+)\]:.*?upright ([\d.]+)deg", txt)
    if tilts:
        worst_tag, worst_deg = max(tilts, key=lambda t: float(t[1]))
        # the insert-entry spike is corrected by the leveller; judge what it is left at
        after = [(t, float(d)) for t, d in tilts if t in ("insert-end", "released")]
        verdict = "PASS" if all(d <= TILT_TOL_DEG for _, d in after) else "FAIL"
        ok &= all(d <= TILT_TOL_DEG for _, d in after)
        print(f"  upright: {'  '.join(f'{t}={d}deg' for t, d in tilts)}")
        print(f"  upright: worst {worst_deg}deg at {worst_tag}; "
              f"left at {after} (tol {TILT_TOL_DEG:.0f}) -> {verdict}")
    else:
        ok = False
        print("  upright: FAIL -- no carry checkpoints found")

    # ---- 3. shelf-board clearance: the object must fit the opening it slides through ----
    # Practical-mode objects are collision-free, so a board strike would pass through silently.
    HALF_H = 0.150
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

    # ---- 4. arm clearance: the hand must clear the board above, through the whole slide ---- Check
    # (3) covers the payload; this covers the robot.
    HAND_TOP = 0.10
    hz = [float(m) for m in re.findall(r"hand z ([\d.]+)", txt)]
    if hz and m_slot:
        lvl_h = 0 if int(m_slot.group(1)) <= 3 else (1 if int(m_slot.group(1)) <= 6 else 2)
        c_arm = (UNDERSIDES_ABOVE[lvl_h] - (max(hz) + HAND_TOP)) * 1000.0
        good_a = c_arm > 20.0
        ok &= good_a
        print(f"  arm-clear: hand z max {max(hz):.3f} + {HAND_TOP:.2f} top vs board underside "
              f"{UNDERSIDES_ABOVE[lvl_h]:.3f} -> {c_arm:.0f}mm -> {'PASS' if good_a else 'FAIL'}")
    else:
        print("  arm-clear: no hand-z samples (older log) -- not gated")

    # ---- hard gates on values the context lines used to only print ---- A run that NaN'd mid-place
    # once still reported ALL CHECKS PASS, because the grade and the ok flag were printed rather
    # than asserted.
    if re.search(r"residual nanmm|upright nandeg|xy err nanmm|\[nan, nan, nan\]", txt):
        ok = False
        print("  nan-gate: FAIL -- NaN in a measured quantity (physics explosion)")
    m_g = re.findall(r"-> (precise_success|precise_marginal|approx_fallback|failed)", txt)
    if m_g and m_g[-1] not in ("precise_success", "precise_marginal"):
        ok = False
        print(f"  grade-gate: FAIL -- final grade {m_g[-1]}")

    # ---- context that decides whether the above is even meaningful ----
    for pat, label in ((r"PLACE obj \d+ -> slot \d+ .* ok=(\w+)", "place ok"),
                       (r"-> (precise_success|precise_marginal|approx_fallback|failed)", "grade"),
                       (r"EASE FALLBACK", "ease fallback (should be absent)"),
                       (r"released place-end \((\d+) drift", "chassis drift re-asserts")):
        m = re.findall(pat, txt)
        print(f"  {label}: {m[-1] if m else 'none'}")
    return ok


if __name__ == "__main__":
    allok = True
    for p in sys.argv[1:]:
        print(f"=== {p.split('/')[-1]}")
        allok &= check(p)
        print()
    print("ALL CHECKS PASS" if allok else "SOME CHECKS FAILED")
    sys.exit(0 if allok else 1)
