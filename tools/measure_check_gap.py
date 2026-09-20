"""How far can a collision box move between two consecutive collision checks?

Run: ./.venv/bin/python3 tools/measure_check_gap.py     (Python 3.10 side; needs numpy only)

Two intervals matter and they are different sizes:
  planner  -- one OMPL validity check, RES * subspace extent, whichever subspace binds
  densify  -- one `arm_plan.STEP` segment, the per-joint FLOOR `_densify` starts from before it
              bisects each piece until every link is inside its OWN limit; this is the WIDEST
              interval a bisection can start from, not what `_densify` emits
A box tunnels an obstacle when it moves further, over an unchecked interval, than its OWN thinnest
dimension plus the margin it is grown by on each side -- `ArmModel.tunnel_limits`, printed here as
`safe`. That is PER LINK: 21 mm of `Gripper_Link1_1` and 290 mm of `Arm_1` do not
survive the same displacement. Sampling is over state PAIRS that both pass the planner's own
validity gate, because a bound taken over states the planner would reject is not a bound on
anything.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morph.arm.model import ArmModel  # noqa: E402

# RES and STEP are copied, not imported: `arm_plan` pulls in OMPL and this tool is numpy-only.
# Drift there silently invalidates what the tool measures; tests/test_check_gap.py asserts both.
from morph.arm.collision import MARGIN  # noqa: E402
RES = 0.005            # arm_plan.RES: fraction of a subspace's MAXIMUM EXTENT, not a length
STEP = 0.02            # arm_plan.STEP: the per-joint FLOOR on emitted waypoint spacing
SAMPLES = 4000


def _load():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return ArmModel.load(os.path.join(root, "usd/_arm_model.json"),
                         os.path.join(root, "usd/_closure_lut.json"))


def _valid(m, q):
    """The planner's own gate, arm_plan.Validity.isValid minus the obstacle test."""
    dh = q[2] - q[1]
    return m.in_lut_domain(dh, q[3]) and not m.passive_violations(dh, q[3])


def _step_planner(m, u, lo, hi):
    """The largest move along `u` that no subspace checks across in one go."""
    lin, rot = ArmModel.Q8_LIN, ArmModel.Q8_ROT
    dl, dr = np.linalg.norm(u[lin]), np.linalg.norm(u[rot])
    el = RES * float(np.linalg.norm((hi - lo)[lin]))
    er = RES * float(np.linalg.norm((hi - lo)[rot]))
    return u * min(el / dl if dl > 1e-12 else np.inf, er / dr if dr > 1e-12 else np.inf)


def _step_densify(u):
    """The STEP floor: the max component is the budget, so this is the WIDEST interval `_densify`
    can start a bisection from -- not what it emits."""
    return u / np.max(np.abs(u)) * STEP


def measure(m, kind, samples=SAMPLES, seed=0, direction=None):
    """Worst per-link corner displacement over one unchecked interval of `kind`."""
    lo, hi = m.bounds()
    rng = np.random.default_rng(seed)
    worst, kept = {}, 0
    while kept < samples:
        q = lo + rng.random(8) * (hi - lo)
        if not _valid(m, q):
            continue
        u = np.array(direction, float) if direction is not None else rng.normal(size=8)
        u = u / np.linalg.norm(u)
        q2 = np.clip(q + (_step_planner(m, u, lo, hi) if kind == "planner" else _step_densify(u)),
                     lo, hi)
        if not _valid(m, q2):
            continue
        kept += 1
        for link, d in m.corner_shifts(q, q2).items():
            if d > worst.get(link, 0.0):
                worst[link] = d
    return worst


def _selfcheck():
    """A finer interval cannot displace more than a coarser one, and the differential columns must
    dominate the revolutes -- if either fails, the closure or the interval maths has changed."""
    m = _load()
    fine = max(measure(m, "planner", samples=200, seed=7).values())
    coarse = max(measure(m, "densify", samples=200, seed=7).values())
    assert fine < coarse, f"planner interval {fine} not finer than densify {coarse}"
    diff = max(measure(m, "densify", samples=200, seed=7, direction=[0, 1, -1, 0, 0, 0, 0, 0]).values())
    rot = max(measure(m, "densify", samples=200, seed=7, direction=[1, 0, 0, 0, 1, 1, 1, 1]).values())
    assert diff > 5 * rot, f"differential {diff} no longer dominates revolute {rot}"
    print("selfcheck passed")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
        sys.exit(0)
    model = _load()
    lim = model.tunnel_limits(margin=MARGIN)
    for which in ("planner", "densify"):
        w = measure(model, which)
        print(f"\n{which}: worst corner displacement over one unchecked interval, against each "
              f"link's own tunnelling limit ({SAMPLES} valid state pairs)")
        print(f"  {'moves':>8s} {'safe':>8s} {'':>6s}   link")
        for link, v in sorted(w.items(), key=lambda kv: -kv[1] / lim[kv[0]]):
            r = v / lim[link]
            print(f"  {v * 1000:7.1f}mm {lim[link] * 1000:7.1f}mm {r:5.2f}x   {link}"
                  f"{'   OVERRUN' if r > 1.0 else ''}")
