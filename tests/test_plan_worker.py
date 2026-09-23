"""The persistent planner child: one worker answers every plan, a request that raises inside it
does not end the loop, a stuck one is killed and respawned, and the one-shot contract the
offline tools still use is unchanged.
Offline, no Isaac:  .venv/bin/python3 tests/test_plan_worker.py"""
import json
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import morph.arm.plan as plan  # noqa: E402

EVENT = "[arm_plan] event "


def _problem(name, **over):
    """A committed bench request with its paths made absolute, as every caller sends it."""
    with open(os.path.join(ROOT, "tools", "bench_problems", f"{name}.json"), encoding="utf-8") as fh:
        req = json.load(fh)
    for k in ("model", "lut"):
        req[k] = os.path.join(ROOT, req[k])
    req["seed"] = 1
    return dict(req, **over)


def _event(stderr):
    lines = [ln for ln in stderr.splitlines() if ln.startswith(EVENT)]
    assert len(lines) == 1, f"one event record per request, got {len(lines)}: {stderr}"
    return json.loads(lines[0][len(EVENT):])


def _fresh():
    plan.drop_worker()
    return plan


def test_one_worker_answers_every_request_without_respawning():
    """The whole point of P5.4: the spawn+import is paid once, not once per plan. A worker that
    respawns per call shows a different pid on the second request."""
    _fresh()
    try:
        first = plan.run_child(_problem("free-space"))
        pid = plan._WORKER.pid
        assert isinstance(plan.child_result(first.stdout), list), first.stdout
        for _ in range(3):
            r = plan.run_child(_problem("free-space"))
            assert isinstance(plan.child_result(r.stdout), list), r.stdout
            assert plan._WORKER.pid == pid, "the child was respawned between plans"
    finally:
        plan.drop_worker()


def test_a_request_that_raises_in_the_child_is_an_error_result_and_the_next_one_answers():
    """A raise inside the loop must become one JSON error result, never the end of the worker:
    the plan after it has to be answered by the same child."""
    _fresh()
    try:
        bad = plan.run_child(_problem("free-space", model="/nonexistent/model.json"))
        pid = plan._WORKER.pid
        out = plan.child_result(bad.stdout)
        assert isinstance(out, dict) and "error" in out, out
        good = plan.run_child(_problem("free-space"))
        assert isinstance(plan.child_result(good.stdout), list), good.stdout
        assert plan._WORKER.pid == pid, "the raise killed the worker"
    finally:
        plan.drop_worker()


def test_the_event_record_does_not_carry_the_previous_request_over():
    """The child accumulates its event in module state, so a loop that does not clear it reports
    the previous plan's error and degrade counts against this one."""
    _fresh()
    try:
        bad = plan.run_child(_problem("free-space", model="/nonexistent/model.json"))
        assert _event(bad.stderr).get("error"), bad.stderr
        good = plan.run_child(_problem("free-space"))
        ev = _event(good.stderr)
        assert ev["ok"] is True and "error" not in ev, ev
    finally:
        plan.drop_worker()


def test_the_child_stderr_reaches_the_parent_per_request():
    """T2's constraint: the planner's degrades reach the operator log only through this stderr,
    and `plan_arm` counts one of them by matching its text."""
    _fresh()
    try:
        r = plan.run_child(_problem("free-space"))
        assert "[arm_plan] hand:" in r.stderr, r.stderr
        assert _event(r.stderr)["ok"] is True, r.stderr
    finally:
        plan.drop_worker()


def test_a_stuck_worker_is_killed_and_the_next_call_gets_a_fresh_one():
    """PLAN_TIMEOUT semantics, unchanged: None to the caller (which counts `plan-timeout`), the
    stuck child killed rather than left holding the pipe."""
    _fresh()
    stub = os.path.join(tempfile.mkdtemp(), "stuck_child.py")
    with open(stub, "w", encoding="utf-8") as fh:
        fh.write("import time\ntime.sleep(300)\n")
    saved = (plan.ARM_PLAN, plan.PLAN_TIMEOUT)
    plan.ARM_PLAN, plan.PLAN_TIMEOUT = stub, 1.0
    try:
        t0 = time.time()
        assert plan.run_child(_problem("free-space")) is None, "a stuck child must read as no-plan"
        dt = time.time() - t0
        assert plan.PLAN_TIMEOUT <= dt < plan.PLAN_TIMEOUT + 2.0, (
            f"the wait was not the timeout: {dt:.1f}s at PLAN_TIMEOUT {plan.PLAN_TIMEOUT}")
        assert plan._WORKER is None, "the killed worker was left in place"
        plan.ARM_PLAN, plan.PLAN_TIMEOUT = saved
        r = plan.run_child(_problem("free-space"))
        assert isinstance(plan.child_result(r.stdout), list), r.stdout
    finally:
        plan.ARM_PLAN, plan.PLAN_TIMEOUT = saved
        plan.drop_worker()
        shutil.rmtree(os.path.dirname(stub))


def test_the_childs_stdout_chatter_before_its_answer_stays_bounded():
    """`_read_frame` leaves stdout unread until the event record arrives on stderr, so a child
    that could fill the 64 KiB pipe with console chatter before its answer would deadlock the
    plan to the watchdog. OMPL writes 2 lines / ~136 B per solving request."""
    for name in ("free-space", "reach", "retreat-remainder", "unreachable"):
        lines = plan.run_one_shot(_problem(name)).stdout.splitlines()
        chatter = "\n".join(lines[:-1])
        assert lines[-1][:1] in ("[", "{"), f"{name}: the answer is not the last stdout line"
        assert len(lines) - 1 <= 8 and len(chatter) < 4096, (
            f"{name}: {len(lines) - 1} chatter lines, {len(chatter)} bytes before the answer")


def test_the_killed_worker_is_reaped_not_left_a_zombie():
    """A kill without a wait leaves a zombie per timeout for the life of the simulator."""
    _fresh()
    plan.run_child(_problem("free-space"))
    pid = plan._WORKER.pid
    plan.drop_worker()
    assert not os.path.exists(f"/proc/{pid}"), f"pid {pid} is still there after the kill"


def test_a_dead_worker_is_respawned_on_the_next_call():
    """A C++ crash is the reason the process boundary exists: it must cost one plan, not the run."""
    _fresh()
    try:
        plan.run_child(_problem("free-space"))
        dead = plan._WORKER
        dead.kill()
        dead.wait()
        r = plan.run_child(_problem("free-space"))
        assert plan._WORKER is not dead, "the dead worker was reused"
        assert isinstance(plan.child_result(r.stdout), list), r.stdout
    finally:
        plan.drop_worker()


def _stub_worker(body):
    """Point the worker at a child script that speaks the line protocol however `body` says.
    Returns the path; the caller restores ARM_PLAN and removes the directory."""
    path = os.path.join(tempfile.mkdtemp(), "stub_child.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path


_STRAY = """import json, sys
for line in sys.stdin:
    req = json.loads(line)
    stale = {"seq": req["seq"] - 1, "result": [[9.0] * 8, [9.1] * 8]}
    sys.stdout.write(json.dumps(stale) + "\\n")        # the PREVIOUS request's answer, late
    sys.stdout.flush()
    sys.stderr.write('[arm_plan] event {"ok": true}\\n')
    sys.stderr.flush()
    sys.stdout.write(json.dumps({"seq": req["seq"], "result": [[0.0] * 8, [0.1] * 8]}) + "\\n")
    sys.stdout.flush()
"""

_DEAF = """import json, os, sys, time
sys.stdin.readline()
sys.stderr.write('[arm_plan] event {"ok": true}\\n')
sys.stderr.flush()
os.close(1)
os.close(2)
time.sleep(300)
"""


def test_a_stray_stdout_line_is_a_protocol_error_not_a_mis_paired_path():
    """A late answer on stdout pairs request N's PATH with request N+1 unless it is checked
    against the request's `seq`. It must read as a refused plan, with the worker dropped."""
    _fresh()
    stub, saved = _stub_worker(_STRAY), plan.ARM_PLAN
    plan.ARM_PLAN = stub
    try:
        out = plan.child_result(plan.run_child(_problem("free-space")).stdout)
        assert isinstance(out, dict), f"the previous request's path was returned: {out}"
        assert "protocol error" in out["error"], out
        assert plan._WORKER is None, "the worker kept the desynchronised child"
    finally:
        plan.ARM_PLAN = saved
        plan.drop_worker()
        shutil.rmtree(os.path.dirname(stub))


def test_a_child_that_closes_its_pipes_but_lives_is_dropped():
    """An empty answer means this plan is lost either way; leaving the child in place would write
    the next request into a process that can no longer reply."""
    _fresh()
    stub, saved = _stub_worker(_DEAF), plan.ARM_PLAN
    plan.ARM_PLAN = stub
    try:
        r = plan.run_child(_problem("free-space"))
        assert r is not None and r.stdout == "", r
        assert plan._WORKER is None, "the deaf child was kept for the next request"
    finally:
        plan.ARM_PLAN = saved
        plan.drop_worker()
        shutil.rmtree(os.path.dirname(stub))


def test_each_reply_is_framed_by_its_own_stderr():
    """The event record is the LAST stderr line of a reply and the hand line is its first, so a
    line written after the event would surface at the head of the NEXT reply's stderr."""
    _fresh()
    try:
        for req in (_problem("free-space", fingers={"finger_1": 0.0}), _problem("free-space")):
            r = plan.run_child(req)
            lines = r.stderr.splitlines()
            assert lines[0].startswith("[arm_plan] hand:"), f"stderr starts mid-reply: {lines[:2]}"
            assert lines[-1].startswith(EVENT), f"the event is not the last line: {lines[-1]}"
            assert sum(ln.startswith(EVENT) for ln in lines) == 1, lines
        assert "21 boxes" not in r.stderr, "the previous request's hand line leaked into this one"
    finally:
        plan.drop_worker()


def test_a_seeded_request_plans_in_a_warm_worker_what_a_fresh_child_plans():
    """The seed is applied once per process and re-applied when it CHANGES, so a worker must not
    mean one seed per run. The emitted path is seed-insensitive on this problem, so the signal is
    the raw tree the child reports: `reach` grows 20 states at seed 3 and 21 at seed 1."""
    _fresh()
    try:
        warm = [_event(plan.run_child(_problem("reach", seed=s)).stderr)["states_raw"]
                for s in (3, 1, 3)]
        fresh = [_event(plan.run_one_shot(_problem("reach", seed=s)).stderr)["states_raw"]
                 for s in (3, 1)]
        assert fresh[0] != fresh[1], f"this problem no longer separates the two seeds: {fresh}"
        assert warm == [fresh[0], fresh[1], fresh[0]], (
            f"a warm worker plans {warm} where fresh children plan {fresh} -- the seed of a "
            f"request is not what it plans with")
    finally:
        plan.drop_worker()


def test_the_one_shot_mode_still_answers_one_request_on_stdin():
    """`tools/bench_plan.py`, `tools/p39_step_measure.py` and the armfk suites drive arm_plan.py
    with no arguments: one JSON request in, one JSON result out, the event on stderr."""
    p = plan.run_one_shot(_problem("free-space"))
    assert isinstance(json.loads(p.stdout.strip().splitlines()[-1]), list), p.stdout
    assert _event(p.stderr)["ok"] is True, p.stderr


def test_bench_plans_own_child_stays_one_shot():
    """The benchmark measures a COLD plan and runs its problems in parallel threads; it must not
    share production's single worker."""
    import bench_plan
    ok, _dt, wps, err, error = bench_plan.run_one(
        json.load(open(os.path.join(ROOT, "tools", "bench_problems", "free-space.json"))),
        seed=1, solve_time=3.0)
    assert ok and wps, error
    assert _event(err)["ok"] is True, err
    assert plan._WORKER is None, "the benchmark spawned production's worker"


if __name__ == "__main__":
    names = [n for n in list(globals()) if n.startswith("test_")]
    failed = 0
    for n in names:
        try:
            globals()[n]()
            print(f"PASS {n}")
        except Exception as e:                     # noqa: BLE001 -- a runner reports, it does not raise
            failed += 1
            print(f"FAIL {n}: {type(e).__name__}: {e}")
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
