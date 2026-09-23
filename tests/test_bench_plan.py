"""tools/bench_plan.py's own logic: the two-number path length, the stderr parse, the regression
gate, and one real plan on the synthetic free-space problem.
Offline, no Isaac:  .venv/bin/python3 tests/test_bench_plan.py"""
import copy
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bench_plan  # noqa: E402


def test_path_length_is_two_numbers_metres_and_radians():
    """Summed over segments (an out-and-back counts twice), L2 within a family per segment
    (two joints moving together are one hypotenuse), metres and radians never mixed."""
    w = np.zeros((4, 8))
    # Both columns at once: one 0.5 m step, not 0.7; then back, which an endpoint would miss.
    w[1, 1], w[1, 2] = 0.3, 0.4
    w[2, 1], w[2, 2] = 0.0, 0.0
    # Turret and wrist together: one 1.0 rad step.
    w[3, 0], w[3, 4] = 0.6, 0.8
    lin, rot = bench_plan.lengths(w)
    assert abs(lin - 1.0) < 1e-12, lin
    assert abs(rot - 1.0) < 1e-12, rot


def test_the_parse_reads_the_child_event_records_and_accumulates_over_tries():
    ev1 = ('[arm_plan] hand: 12 (swept union)\n'
           '[arm_plan] event {"hand": "swept", "goals": 3, "bad_goals": 2, "tunnel": 1, "selfcol": 3, '
           '"densify_bad": 0, "states_raw": 41, "states": 6, "waypoints": 213, "solve_s": 0.3, '
           '"simplify_s": 0.05, "densify_s": 0.07, "ok": false, "error": "x"}\n')
    ev2 = ('[arm_plan] event {"hand": "swept", "goals": 3, "bad_goals": 2, "tunnel": 0, "selfcol": 4, '
           '"densify_bad": 1, "states_raw": 7, "states": 3, "waypoints": 59, "solve_s": 0.9, '
           '"simplify_s": 0.5, "densify_s": 0.2, "ok": true}\n')
    got = bench_plan.parse_stderr(ev1 + ev2)
    assert (got["bad_goals"], got["tunnel"], got["selfcol"], got["densify_bad"]) == (4, 1, 7, 1), "counts accumulate"
    assert (got["states_raw"], got["states"], got["solve_s"], got["hand"]) == (7, 3, 0.9, "swept"), "the last try that planned"
    clean = bench_plan.parse_stderr("[arm_plan] self-collision: clear over 14 waypoints\n")
    assert all(clean[k] == 0 for k in bench_plan.DEGRADES) and clean["states"] is None, clean


ENV = {"SELF_COLLIDE_PLAN": None, "ARM_SIMPLIFY_MODE": None, "ARM_SIMPLIFY_T": None}


def _summary(**over):
    """(baseline, run) for one problem, the run differing from the baseline by `over`."""
    base = {"reach": {"n": 100, "ok_pct": 100.0, "seed1_ok": True, "p50_ms": 200.0, "p95_ms": 300.0,
                      "max_ms": 400.0, "lin_m": 0.4, "rot_rad": 1.8, "waypoints": 30.0,
                      "solve_s": 1.0, "simplify_s": 0.8, "densify_s": 0.4,
                      "bad_goals": 0, "tunnel": 0, "selfcol": 400, "densify_bad": 0,
                      "jobs": 1, "solve_time": 3.0, "env": dict(ENV)}}
    now = copy.deepcopy(base)
    now["reach"].update(over)
    return base, now


def _bad(**over):
    base, now = _summary(**over)
    return bench_plan.gate(now, base)[0]


def test_the_gate_passes_an_unchanged_run_and_names_a_change_in_either_direction():
    assert bench_plan.gate(*reversed(_summary())) == ([], [])
    assert any("success" in m for m in _bad(ok_pct=97.0))
    assert any("success" in m for m in _bad(ok_pct=100.0)) is False
    base, now = _summary()
    base["reach"]["ok_pct"], base["reach"]["seed1_ok"] = 0.0, False
    now["reach"]["ok_pct"], now["reach"]["seed1_ok"] = 100.0, True
    got = bench_plan.gate(now, base)[0]
    assert any("success 0.0% -> 100.0%" in m for m in got), "a refusal problem that starts succeeding is a change"
    assert any("seed 1" in m for m in got), got
    assert any("p95" in m for m in _bad(p95_ms=700.0)), "a 2.3x rise"
    assert any("p95" in m for m in _bad(p95_ms=100.0)), "a 3x drop is the checker gone, not luck"
    assert not any("p95" in m for m in _bad(p95_ms=453.0)), "the measured 51 % spread must pass"
    assert any("lin_m" in m for m in _bad(lin_m=0.5)) and any("lin_m" in m for m in _bad(lin_m=0.3))
    assert any("rot_rad" in m for m in _bad(rot_rad=None)), "a length that vanished"
    assert any("waypoints" in m for m in _bad(waypoints=80.0)), "simplification degrading: 59 -> 80"
    assert any("selfcol" in m for m in _bad(selfcol=0)), "a silenced reporter reads as a drop"
    assert any("selfcol" in m for m in _bad(selfcol=500))
    assert any("bad_goals" in m for m in _bad(bad_goals=5))
    assert any("densify_bad" in m for m in _bad(densify_bad=1))
    assert any("tunnel" in m for m in _bad(tunnel=1))


def test_a_zero_baseline_still_gates_lengths_above_a_floor():
    base, now = _summary(lin_m=0.05)
    base["reach"]["lin_m"] = 0.0
    assert any("lin_m" in m for m in bench_plan.gate(now, base)[0]), "free-space growing 5 cm of column travel"
    now["reach"]["lin_m"] = 0.005
    assert not any("lin_m" in m for m in bench_plan.gate(now, base)[0]), "under the floor is noise"


def test_degrade_counts_compare_as_rates_across_seed_counts():
    base, now = _summary(n=3, selfcol=12)
    assert bench_plan.gate(now, base)[0] == [], "4 per plan at n=3 is the baseline's 4 per plan"
    now["reach"]["selfcol"] = 177
    assert any("selfcol" in m for m in bench_plan.gate(now, base)[0]), "59 per plan at n=3"
    base, now = _summary(n=200, selfcol=800)
    assert bench_plan.gate(now, base)[0] == [], "healthy at n=200 is not a regression"


def test_latency_is_noted_not_compared_across_job_counts_and_env_never_compares():
    for jobs in (4, 1):
        base, now = _summary(p95_ms=900.0, jobs=jobs)
        base["reach"]["jobs"] = 5 - jobs
        bad, notes = bench_plan.gate(now, base)
        assert not any("p95" in m for m in bad) and any("latency not compared" in n for n in notes), (bad, notes)
    base, now = _summary(env={**ENV, "ARM_SIMPLIFY_MODE": "none"})
    bad, _ = bench_plan.gate(now, base)
    assert len(bad) == 1 and "child env" in bad[0], bad
    base, now = _summary()
    now.pop("reach")
    bad, notes = bench_plan.gate(now, base)
    assert bad == [] and notes == ["reach: not run, not compared"], (bad, notes)


def test_summarise_carries_n_seed_one_and_the_child_env():
    rows = [bench_plan.row_for("p", seed, 3.0, 1, seed != 2, 0.1, [[0.0] * 8, [0.0] * 8] if seed != 2 else None,
                               '[arm_plan] event {"selfcol": 4}\n', "" if seed != 2 else "no")
            for seed in (1, 2, 3)]
    got = bench_plan.summarise(rows, {"SELF_COLLIDE_PLAN": "1"})["p"]
    assert (got["n"], got["seed1_ok"], got["selfcol"], got["env"]) == (3, True, 12, {"SELF_COLLIDE_PLAN": "1"}), got
    assert abs(got["ok_pct"] - 200.0 / 3) < 1e-9
    rows[0]["ok"], rows[0]["lin_m"] = 0, None
    assert bench_plan.summarise(rows, {})["p"]["seed1_ok"] is False


PROBLEMS = os.path.join(ROOT, "tools", "bench_problems")


def test_a_problem_the_robot_has_moved_away_from_is_refused():
    import tempfile
    req = json.load(open(os.path.join(PROBLEMS, "free-space.json")))
    with tempfile.TemporaryDirectory() as d:
        for change, word in (({"margin": 0.03}, "margin"),
                             ({"bounds": [req["bounds"][0], [v + 0.1 for v in req["bounds"][1]]]}, "bounds")):
            with open(os.path.join(d, "free-space.json"), "w") as fh:
                json.dump({**req, **change}, fh)
            try:
                bench_plan.load_problems(d)
            except SystemExit as e:
                assert word in str(e), (word, e)
            else:
                raise AssertionError(f"a stale {word} loaded")
    assert set(bench_plan.load_problems(PROBLEMS)) >= {"reach", "free-space"}, "the committed set is live"


def test_a_free_space_plan_runs_and_reports_per_family_lengths():
    problems = bench_plan.load_problems(PROBLEMS, {"free-space"})
    ok, dt, wps, err, error = bench_plan.run_one(problems["free-space"], seed=1, solve_time=1.0)
    assert ok and len(wps) >= 2, (error, err[-300:])
    row = bench_plan.row_for("free-space", 1, 1.0, 1, ok, dt, wps, err, error)
    assert row["lin_m"] == 0.0 and abs(row["rot_rad"] - 0.25) < 1e-6, row
    # The event record is read from a REAL child here: its numbers must describe this plan.
    hand = "commanded" if problems["free-space"]["fingers"] else "swept"
    assert row["states"] == 2 and row["waypoints"] == len(wps) and row["hand"] == hand, row
    assert row["solve_s"] is not None and err.count(bench_plan.EVENT) == 1, err[-400:]


def test_the_child_runs_under_the_scrubbed_environment():
    """Isaac leaks PYTHONPATH into children; a poisoned one must not reach the planner."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        os.mkdir(os.path.join(d, "numpy"))
        with open(os.path.join(d, "numpy", "__init__.py"), "w") as fh:
            fh.write("raise ImportError('poisoned PYTHONPATH reached the child')\n")
        prev = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = d
        try:
            problems = bench_plan.load_problems(PROBLEMS, {"free-space"})
            ok, _, _, err, error = bench_plan.run_one(problems["free-space"], seed=1, solve_time=1.0)
        finally:
            os.environ.pop("PYTHONPATH", None) if prev is None else os.environ.__setitem__("PYTHONPATH", prev)
    assert ok, (error, err[-300:])


def test_seed_and_solve_time_reach_the_child():
    """Different seeds grow different raw paths on the reach (21 vs 20 states at seeds 1 and 3);
    a 1 ms budget plans nothing."""
    problems = bench_plan.load_problems(PROBLEMS, {"reach"})
    raw = {}
    for seed in (1, 3):
        ok, dt, wps, err, error = bench_plan.run_one(problems["reach"], seed=seed, solve_time=3.0)
        assert ok, (seed, error)
        raw[seed] = bench_plan.parse_stderr(err)["states_raw"]
    assert raw[1] != raw[3], raw
    ok, _, _, _, error = bench_plan.run_one(problems["reach"], seed=1, solve_time=0.001, tries=1)
    assert not ok, "a 1 ms budget must not plan the reach"


def test_run_one_pays_production_two_tries_and_reports_a_timeout_as_an_error():
    calls = []

    class _P:
        def __init__(self, out):
            self.stdout, self.stderr = out, "[arm_plan] stub\n"

    def _refuses(req):
        calls.append(req["solve_time"])
        return _P('{"error": "no"}')
    real = bench_plan.run_one_shot
    bench_plan.run_one_shot = _refuses
    try:
        ok, _, wps, err, error = bench_plan.run_one({"model": "m", "lut": "l"}, seed=1, solve_time=3.0)
        assert (ok, wps, calls) == (False, None, [3.0, 6.0]), (ok, calls)
        assert err.count("[arm_plan] stub") == 2 and error == "no", (err, error)
        calls.clear()
        bench_plan.run_one_shot = lambda req: (calls.append(req["solve_time"]), _P("[[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]"))[1]
        ok, _, wps, _, _ = bench_plan.run_one({"model": "m", "lut": "l"}, seed=1, solve_time=3.0)
        assert ok and len(wps) == 2 and calls == [3.0], calls
        bench_plan.run_one_shot = lambda req: None
        ok, _, _, _, error = bench_plan.run_one({"model": "m", "lut": "l"}, seed=1, solve_time=3.0)
        assert not ok and "timed out" in error, error
    finally:
        bench_plan.run_one_shot = real


if __name__ == "__main__":
    names = [n for n in list(globals()) if n.startswith("test_")]
    failed = 0
    for n in names:
        try:
            globals()[n]()
            print(f"PASS {n}")
        except Exception as e:                     # noqa: BLE001 -- a runner reports, it does not raise
            failed += 1
            print(f"FAIL {n}: {e}")
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
