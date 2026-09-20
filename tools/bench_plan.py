#!/usr/bin/env python3
"""Offline planner benchmark: drive arm_plan.py over a committed problem set, N seeds each.

No Isaac, no GPU: arm_plan.py is a JSON-over-stdin subprocess. Path length is TWO numbers,
metres over Q8_LIN and radians over Q8_ROT -- the state space mixes units, their sum has none.

  tools/bench_plan.py --seeds 100 --out bench.csv --write-baseline tools/bench_plan_baseline.json
  tools/bench_plan.py --seeds 100 --baseline tools/bench_plan_baseline.json   # exit 1 on a change
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from morph.arm.model import ArmModel  # noqa: E402
from morph.arm.plan import PLAN_TIMEOUT, child_result, run_one_shot, try_budgets  # noqa: E402

EVENT = "[arm_plan] event "
DEGRADES = ("bad_goals", "tunnel", "selfcol", "densify_bad")
# What the child reads from its own environment: a run under a different setting is a different
# planner, and the baseline records which one it measured.
CHILD_ENV = ("SELF_COLLIDE_PLAN", "ARM_SIMPLIFY_MODE", "ARM_SIMPLIFY_T")

# CHOSEN above the spread of three 100-seed sweeps on one tree: success, lengths, waypoints and
# degrade rates spread 0 %; p95 latency 51 % between two idle-machine runs, hence a 2x band.
GATE = {"ok_pts": 2.0, "p95_band": 2.0, "pct": 0.20, "length_floor": 0.01, "waypoints_floor": 1.0}


def _stale(req):
    """Why `req` no longer describes the robot: its margin, deck pad and bounds against the live
    constants. A stale set benchmarks a planner for a robot that no longer exists."""
    from morph.arm.collision import CHASSIS_PAD_LINKS, CHASSIS_PAD_M, MARGIN
    from morph.arm.surface import ArmMixin
    why = []
    if req["margin"] != MARGIN:
        why.append(f"margin {req['margin']} vs live {MARGIN}")
    for pad in req["obstacle_pads"]:
        if pad[1] != CHASSIS_PAD_M or set(pad[2]) != set(CHASSIS_PAD_LINKS):
            why.append(f"deck pad {pad[1:]} vs live {(CHASSIS_PAD_M, list(CHASSIS_PAD_LINKS))}")
    model = ArmModel.load(*(os.path.join(ROOT, req[k]) for k in ("model", "lut")))
    lo, hi = ArmMixin._arm_bounds(SimpleNamespace(_arm_model=lambda: model))
    if not (np.allclose(req["bounds"][0], lo) and np.allclose(req["bounds"][1], hi)):
        why.append("bounds differ from the live joint limits")
    return why


def load_problems(folder, only=None):
    """{name: request}; refuses a request the live robot has moved away from (re-capture it)."""
    out = {}
    for name in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(name)
        if ext != ".json" or (only and stem not in only):
            continue
        with open(os.path.join(folder, name), encoding="utf-8") as fh:
            out[stem] = json.load(fh)
        why = _stale(out[stem])
        if why:
            raise SystemExit(f"{name}: stale problem set ({'; '.join(why)}); re-capture, do not edit")
    return out


def run_one(req, seed, solve_time, tries=2):
    """Production's attempt through plan_arm's own child contract: `tries` runs at `try_budgets`,
    the first path wins. (ok, seconds over all tries, waypoints or None, stderr of all tries,
    error). Seeds start at 1: OMPL maps 0 to 1."""
    r = dict(req, seed=int(seed))
    for k in ("model", "lut"):
        r[k] = r[k] if os.path.isabs(r[k]) else os.path.join(ROOT, r[k])
    t0, stderr, error = time.time(), "", ""
    for budget in try_budgets(float(solve_time), tries):
        r["solve_time"] = budget
        p = run_one_shot(r)
        if p is None:
            error = f"timed out after {PLAN_TIMEOUT}s"
            break
        stderr += p.stderr
        out = child_result(p.stdout)
        if isinstance(out, list):
            return True, time.time() - t0, out, stderr, ""
        error = str(out.get("error", out))[:80]
    return False, time.time() - t0, None, stderr, error


def lengths(waypoints):
    """(metres over Q8_LIN, radians over Q8_ROT) summed over consecutive waypoints."""
    d = np.diff(np.asarray(waypoints, float), axis=0)
    return (float(np.linalg.norm(d[:, ArmModel.Q8_LIN], axis=1).sum()),
            float(np.linalg.norm(d[:, ArmModel.Q8_ROT], axis=1).sum()))


def parse_stderr(text):
    """The child's event records, one per try: counts accumulated, the rest from the last try
    that planned (the text lines above each record are for the operator log)."""
    got = {"states_raw": None, "states": None, "solve_s": None, "simplify_s": None,
           "densify_s": None, "hand": None, **{k: 0 for k in DEGRADES}}
    for line in text.splitlines():
        if not line.startswith(EVENT):
            continue
        ev = json.loads(line[len(EVENT):])
        for k in DEGRADES:
            got[k] += int(ev.get(k, 0))
        for k in ("states_raw", "states", "solve_s", "simplify_s", "densify_s", "hand"):
            if ev.get(k) is not None:
                got[k] = ev[k]
    return got


def row_for(problem, seed, solve_time, jobs, ok, dt, wps, err, error=""):
    row = {"problem": problem, "seed": seed, "solve_time": solve_time, "jobs": jobs, "ok": int(ok),
           "latency_ms": round(dt * 1000.0, 1), "waypoints": len(wps) if wps else None,
           "lin_m": None, "rot_rad": None, "error": error}
    # `parse_stderr` adds the child's own counts and timings, and which hand it planned with.
    if wps:
        row["lin_m"], row["rot_rad"] = (round(v, 4) for v in lengths(wps))
    row.update(parse_stderr(err))
    return row


def _mean(rows, k):
    vals = [r[k] for r in rows if r[k] is not None]
    return float(np.mean(vals)) if vals else None


def summarise(rows, env):
    """Per problem: n, ok%, seed 1's own verdict, latency p50/p95/max (ms), mean LIN/ROT/waypoints
    and timing split over successes, the degrade totals, and the run's child environment."""
    out = {}
    for problem in sorted({r["problem"] for r in rows}):
        rs = [r for r in rows if r["problem"] == problem]
        lat = np.asarray([r["latency_ms"] for r in rs], float)
        oks = [r for r in rs if r["ok"]]
        seed1 = [r for r in rs if r["seed"] == 1]
        out[problem] = {
            "n": len(rs), "ok_pct": 100.0 * len(oks) / len(rs),
            "seed1_ok": bool(seed1[0]["ok"]) if seed1 else None,
            "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
            "max_ms": float(lat.max()),
            "lin_m": _mean(oks, "lin_m"), "rot_rad": _mean(oks, "rot_rad"),
            "waypoints": _mean(oks, "waypoints"),
            "solve_s": _mean(oks, "solve_s"), "simplify_s": _mean(oks, "simplify_s"),
            "densify_s": _mean(oks, "densify_s"),
            **{k: sum(r[k] for r in rs) for k in DEGRADES},
            "jobs": rs[0]["jobs"], "solve_time": rs[0]["solve_time"], "env": dict(env),
        }
    return out


def _band(problem, k, was, now, floor, bad):
    if was is None and now is None:
        return
    if was is None or now is None:
        bad.append(f"{problem}: {k} {was} -> {now}")
    elif abs(now - was) > max(GATE["pct"] * abs(was), floor):
        bad.append(f"{problem}: {k} {was:.3f} -> {now:.3f}")


def gate(summary, baseline):
    """(differences, notes): every difference beyond tolerance, in EITHER direction -- a gate
    that reads one direction passes a planner whose checker is gone. Degrade counts compare as
    per-plan rates so seed counts need not match; latency only between runs of the same `jobs`;
    a run under a different child environment is a different planner and does not compare."""
    bad, notes = [], []
    for problem, was in baseline.items():
        now = summary.get(problem)
        if now is None:
            notes.append(f"{problem}: not run, not compared")
            continue
        if now["env"] != was["env"]:
            bad.append(f"{problem}: child env {was['env']} -> {now['env']}: not comparable")
            continue
        if abs(now["ok_pct"] - was["ok_pct"]) > GATE["ok_pts"]:
            bad.append(f"{problem}: success {was['ok_pct']:.1f}% -> {now['ok_pct']:.1f}%")
        if now["seed1_ok"] != was["seed1_ok"]:
            bad.append(f"{problem}: seed 1 (production's) ok {was['seed1_ok']} -> {now['seed1_ok']}")
        if now["jobs"] != was["jobs"]:
            notes.append(f"{problem}: latency not compared (jobs {now['jobs']} vs baseline {was['jobs']})")
        elif not was["p95_ms"] / GATE["p95_band"] <= now["p95_ms"] <= was["p95_ms"] * GATE["p95_band"]:
            bad.append(f"{problem}: p95 {was['p95_ms']:.0f} -> {now['p95_ms']:.0f} ms")
        for k in ("lin_m", "rot_rad"):
            _band(problem, k, was[k], now[k], GATE["length_floor"], bad)
        _band(problem, "waypoints", was["waypoints"], now["waypoints"], GATE["waypoints_floor"], bad)
        for k in DEGRADES:
            was_r, now_r = was[k] / was["n"], now[k] / now["n"]
            if (was_r == 0) != (now_r == 0) or abs(now_r - was_r) > GATE["pct"] * was_r:
                bad.append(f"{problem}: {k} {was_r:.2f} -> {now_r:.2f} per plan")
    return bad, notes


def print_summary(summary):
    print(f"{'problem':22s} {'n':>4s} {'ok%':>6s} {'s1':>3s} {'p50ms':>7s} {'p95ms':>7s} {'max':>7s} "
          f"{'LIN(m)':>8s} {'ROT(rad)':>9s} {'wps':>5s}  degrades")
    for problem, s in summary.items():
        deg = ", ".join(f"{k}={s[k]}" for k in DEGRADES if s[k]) or "-"
        lin = f"{s['lin_m']:.3f}" if s["lin_m"] is not None else "-"
        rot = f"{s['rot_rad']:.3f}" if s["rot_rad"] is not None else "-"
        wps = f"{s['waypoints']:.0f}" if s["waypoints"] is not None else "-"
        s1 = {True: "ok", False: "NO", None: "-"}[s["seed1_ok"]]
        print(f"{problem:22s} {s['n']:4d} {s['ok_pct']:6.1f} {s1:>3s} {s['p50_ms']:7.0f} {s['p95_ms']:7.0f} "
              f"{s['max_ms']:7.0f} {lin:>8s} {rot:>9s} {wps:>5s}  {deg}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", default=os.path.join(ROOT, "tools", "bench_problems"))
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--solve-time", type=float, default=3.0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", default="", help="comma-separated problem names")
    ap.add_argument("--out", default="")
    ap.add_argument("--baseline", default="")
    ap.add_argument("--write-baseline", default="")
    args = ap.parse_args()

    problems = load_problems(args.problems, set(filter(None, args.only.split(","))))
    if not problems:
        print("no problems found", file=sys.stderr)
        return 2
    work = [(name, seed) for name in problems for seed in range(1, args.seeds + 1)]

    def one(item):
        name, seed = item
        return row_for(name, seed, args.solve_time, args.jobs,
                       *run_one(problems[name], seed, args.solve_time))
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        rows = list(pool.map(one, work))

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    summary = summarise(rows, {k: os.environ.get(k) for k in CHILD_ENV})
    print_summary(summary)
    if args.write_baseline:
        with open(args.write_baseline, "w") as fh:
            json.dump({"problems": summary}, fh, indent=1)
        print(f"baseline written: {args.write_baseline}")
    if args.baseline:
        with open(args.baseline) as fh:
            bad, notes = gate(summary, json.load(fh)["problems"])
        for n in notes:
            print(f"note: {n}")
        if bad:
            print("CHANGED against the baseline:\n  " + "\n  ".join(bad))
            return 1
        print("bench gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
