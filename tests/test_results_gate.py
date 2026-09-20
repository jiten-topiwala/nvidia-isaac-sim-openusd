"""results_gate: the one acceptance check that reads a stage measurement instead of the log.
Offline: no Isaac, no simulator. Run: ./.venv/bin/python3 tests/test_results_gate.py"""
import ast
import contextlib
import io
import json
import os
import sys
import tempfile
import time

import numpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RAN_LOG = ">>> PLACE obj 0 -> slot 0 ok=True -> precise_success\n"
GRADED_2ND = ">>> PLACE obj 1 -> slot 1 ok=True -> precise_success\n"
NO_CYCLE_LOG = ">>> menu ready -- pick an object + slot, click MOVE. Close window to exit.\n"


RUN = "stub-1"                       # the run id every fixture record and manifest here carries
# An ambient override would point the reader at a manifest these fixtures never wrote.
os.environ.pop("PLACE_RUN_MANIFEST", None)
_TMP = tempfile.TemporaryDirectory()  # fixtures live here, so their manifests don't litter /tmp


def _manifest(results_path, run=RUN):
    """The run manifest Demo.__init__ writes before cycle 1, naming the run whose records are THIS
    run's evidence -- without it the gate can only trust the id the last record carries."""
    with open(results_path + ".run", "w") as f:
        json.dump({"run": run, "started": "2026-01-01T00:00:00"}, f)


def _fixture(records, run=RUN):
    """Write records to a throwaway JSON file (the PLACE_RESULTS shape), with the run manifest the
    gate now requires beside it, and return its path."""
    fd, path = tempfile.mkstemp(suffix=".json", dir=_TMP.name)
    with os.fdopen(fd, "w") as f:
        json.dump(records, f)
    _manifest(path, run)
    return path


def _record(err_mm, obj=0, slot=0):
    """A results record whose measured xy sits err_mm from a target parked at the origin."""
    return {"obj": obj, "slot": slot, "run": RUN, "xy": [err_mm / 1000.0, 0.0], "z": 0.25,
            "target_xy": [0.0, 0.0], "target_z": 0.25}


def _play_isaac_ast():
    """play_isaac.py imports isaacsim at module scope and cannot be executed in this 3.10 offline
    venv (see tests/test_fallback_counter.py's own _fallback_sites, same reason). AST-only."""
    src = open(os.path.join(ROOT, "play_isaac.py")).read()
    return ast.parse(src), src


def _demo_init(tree):
    """M3: `next(walk(...) if name == "__init__")` took the FIRST __init__ in the module, whichever
    class it belongs to -- a helper class added above Demo would silently retarget the assertion."""
    demo = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Demo")
    return next(n for n in demo.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")


# Only [0..2] of a slot is ever read. Stubbed rather than imported: morph.config applies the perfect
# profile and prints a banner on import, and none of that is what these tests are about.
SLOT_STUB = [[2.80, -3.72, 0.247], [2.85, -3.72, 0.687]]
HALF_H = 0.14


class _StubObj:
    """`self.objs[i]` as the writer uses it: get_world_poses()[0][0] is the world position."""

    def __init__(self, pos):
        self.pos = list(pos)

    def get_world_poses(self):
        return ([self.pos], None)


def _stub_demo(results_path):
    """The two guards this design rests on -- the merge-in-place and the `_placed_at_open` tail
    slice -- had no runnable test and both mutants survived. Demo cannot be imported here
    (module-scope `isaacsim`), so lift the writer methods out by AST into a synthetic class and
    EXECUTE them against a stub self. Anything they reach for beyond {os, json, np, time,
    PLACE_RESULTS, SHELF_SLOTS, self} raises NameError here, which is itself part of the check."""
    tree, _ = _play_isaac_ast()
    want = ("_place_records", "_dump_place_records", "_open_place_result", "_write_place_result",
            "_write_run_manifest")
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in want]
    assert sorted(f.name for f in fns) == sorted(want), (
        f"writer methods missing or renamed: {sorted(set(want) - {f.name for f in fns})}")
    ns = {"os": os, "json": json, "np": numpy, "time": time, "PLACE_RESULTS": results_path,
          "SHELF_SLOTS": SLOT_STUB}
    exec(compile(ast.Module(body=fns, type_ignores=[]), "<play_isaac writers>", "exec"), ns)
    demo = type("StubDemo", (), {n: ns[n] for n in want})()
    demo.objs = [_StubObj([0.0, 0.0, 0.0]) for _ in range(10)]
    demo.obj_half_h = HALF_H
    demo.placed = []
    demo.fallbacks = []
    demo._fallback = demo.fallbacks.append
    # ...and EXECUTE Demo.__init__'s evidence prologue (run id, truncation, manifest) instead of
    # imitating it: P0-4 lived in that prologue, so a stub that hand-rolls it cannot see the hole.
    init = _demo_init(tree)
    cut = [i for i, s in enumerate(init.body) if any(
        isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
        and c.func.attr == "_write_run_manifest" for c in ast.walk(s))]
    assert cut, "Demo.__init__ no longer writes the run manifest"
    exec(compile(ast.Module(body=init.body[:max(cut) + 1], type_ignores=[]),
                 "<Demo.__init__ evidence prologue>", "exec"), dict(ns, self=demo))
    return demo


def _boom(*a, **kw):
    """A dump that dies mid-write: ENOSPC, or the kill these runs normally end with."""
    raise OSError("no space left on device")


def _killed(*a, **kw):
    """A write the process does not survive -- BaseException, so no `except Exception` runs."""
    raise KeyboardInterrupt


def _boom_once(orig):
    """os.remove that fails its FIRST call only: the truncation that dies on a read-only file and
    leaves the previous run's records in place, with later removes still working."""
    dead = []

    def _rm(path):
        if not dead:
            dead.append(path)
            raise OSError("read-only file system")
        return orig(path)
    return _rm


def _on_target(slot_idx):
    """The world position a perfectly placed object ends at: the slot, lifted by its half-height."""
    s = SLOT_STUB[slot_idx]
    return [s[0], s[1], s[2] + HALF_H]


def _cycle(demo, obj, slot, landed_at):
    """run_cycle's writer path in its real order: stamp the attempt, maybe land the object, write.
    landed_at=None is a pick that returned False -- the cycle whose ok:false stamp must stand."""
    demo._open_place_result(obj, slot)
    if landed_at is not None:
        demo.objs[obj].pos = list(landed_at)
        demo.placed.append((obj, list(landed_at)))
    demo._write_place_result(obj, slot)


def test_two_clean_cycles_yield_one_ok_record_each_and_pass_the_gate():
    """The merge-in-place guard, executed. `records.append(rec)` instead of the update leaves every
    cycle owning a dead ok:false stamp beside its success, so every clean run FAILs the gate."""
    import verify_place
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "place_results.json")
        demo = _stub_demo(path)
        _cycle(demo, 0, 0, _on_target(0))
        _cycle(demo, 1, 1, _on_target(1))
        recs = json.load(open(path))
        assert len(recs) == 2, f"one record per cycle; got {len(recs)}: {recs}"
        assert all(r["ok"] for r in recs), f"a clean cycle left an ok:false stamp: {recs}"
        assert verify_place.results_gate(RAN_LOG + GRADED_2ND, path) is True


def test_a_repeat_cycle_on_an_already_placed_object_keeps_its_own_ok_false_stamp():
    """The `_placed_at_open` tail slice, executed. self.placed is append-only for the process, so
    without the slice cycle B matches cycle A's landing, re-reads the shelf pose the object is STILL
    sitting at and stamps ok:true over its own ok:false -- and B's pick returned False without
    raising, so there is no grade line and no crash line to catch it either."""
    import verify_place
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "place_results.json")
        demo = _stub_demo(path)
        # A: lands on target
        _cycle(demo, 0, 0, _on_target(0))
        # B: same object again, pick returned False
        _cycle(demo, 0, 0, None)
        recs = json.load(open(path))
        assert len(recs) == 2, f"one record per cycle; got {len(recs)}: {recs}"
        assert recs[0]["ok"] is True and recs[1]["ok"] is False, (
            f"cycle B overwrote its own ok:false stamp with cycle A's success: {recs}")
        assert verify_place.results_gate(RAN_LOG, path) is False
        # A's grade line alone already fails the count check; the two-grade log takes the count out
        # of it so this asserts the STAMP.
        assert verify_place.results_gate(RAN_LOG + GRADED_2ND, path) is False


def test_within_tolerance_passes():
    import verify_place
    path = _fixture([_record(10)])
    try:
        assert verify_place.results_gate(RAN_LOG, path) is True
    finally:
        os.remove(path)


def test_60mm_off_fails():
    """60mm is picked to fail BOTH this gate's 20mm and the old precise_marginal's 50mm, so the
    failure isn't an artifact of only one threshold."""
    import verify_place
    path = _fixture([_record(60)])
    try:
        assert verify_place.results_gate(RAN_LOG, path) is False
    finally:
        os.remove(path)


def test_35mm_off_fails_though_precise_marginal_would_have_passed_it():
    """precise_marginal allows up to 50mm; this gate's 20mm is deliberately tighter because it
    reads the stage directly instead of trusting the log's self-reported grade."""
    import verify_place
    path = _fixture([_record(35)])
    try:
        assert verify_place.results_gate(RAN_LOG, path) is False
    finally:
        os.remove(path)


def test_missing_file_fails_when_the_log_shows_cycles_ran():
    """The defect being fixed: every other gate in this file lets absence pass. This is the one
    number that must not go missing when there was something to measure."""
    import verify_place
    assert verify_place.results_gate(RAN_LOG, "/nonexistent/place_results.json") is False


def test_missing_file_passes_when_no_cycles_ran():
    """Legacy logs (pre-existing, before this gate existed) stay gradeable."""
    import verify_place
    assert verify_place.results_gate(NO_CYCLE_LOG, "/nonexistent/place_results.json") is True


def test_missing_file_fails_when_only_the_banner_shows_a_cycle_ran():
    """SPEC MISS: a cycle that crashes at pick prints its CYCLE banner and '!! cycle failed:' but
    never reaches the grade line and writes no record. A grade-line-only 'ran' check let that look
    identical to a log with nothing in it at all -- the exact hole this gate exists to close."""
    import verify_place
    crashed_log = (">>> ===== CYCLE obj 0 -> slot 0 =====\n"
                   "!! cycle failed: some IK exception\n")
    assert verify_place.results_gate(crashed_log, "/nonexistent/place_results.json") is False


def test_missing_file_fails_on_the_gui_path_with_only_a_fallback_tally():
    """CRITICAL: on the GUI path (HEADLESS off, the canonical launcher) a cycle that crashes at
    pick prints no CYCLE_BANNER (that's main()'s headless branch only, play_isaac.py's own
    `if HEADLESS:`) and no grade line (place() never got that far). The one line _fallback_tally
    prints unconditionally on BOTH paths is plan-fallbacks -- that's the exact GUI blind spot its
    own docstring names. Without it in 'ran', this crashed cycle looked identical to an empty log
    and a missing results file passed."""
    import verify_place
    gui_crash_log = (">>> cycle-entry[obj 0]: arm kp sum 12 kd sum 34 | arm |qv| 0.0100 max "
                      "0.0200 | drv_prev clear | brake off\n"
                      "!! cycle failed: some IK exception\n"
                      ">>> plan-fallbacks[obj 0]: 2 (goal-ik-miss=2)\n")
    assert verify_place.results_gate(gui_crash_log, "/nonexistent/place_results.json") is False


def test_an_aborted_gui_run_that_wrote_no_record_at_all_fails():
    """The sharpest form of the same GUI blind spot, and the one the three markers cannot see: a
    run aborted before its FIRST cycle prints no CYCLE banner (that line is play_isaac.py:661,
    main()'s headless branch alone -- the canonical launcher is the GUI one), no grade line and no
    plan-fallbacks tally. All three absent, `return not ran` PASSED it. It passed BECAUSE it died
    early enough to say nothing.

    The positive signal is the run manifest: `Demo.__init__` writes it before cycle 1 can run, so a
    manifest with no results file beside it is a run that STARTED and produced nothing -- which is
    exactly the case that must FAIL, and the only thing that tells it apart from a log written
    before the writer existed."""
    import verify_place
    path = os.path.join(_TMP.name, "aborted-gui", "place_results.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _manifest(path)
    assert not os.path.exists(path), "the fixture wrote a results file; the case needs none"
    assert os.path.exists(path + ".run"), "the fixture wrote no manifest, so nothing below is evidence"
    assert verify_place.results_gate(NO_CYCLE_LOG, path) is False, (
        "an aborted GUI run -- manifest written before cycle 1, no records file, and none of "
        "the three log markers `ran` looks for -- PASSED the results gate")


def test_round3_repro_gui_log_where_a_crashed_cycle_wrote_no_record():
    """ROUND 3, reproduced on HEAD with the repo's real GUI log: no CYCLE banners (the GUI menu
    loop prints none), cycle 1 graded and recorded, cycle 2 crashed at pick so it never graded and
    never wrote. 'ran' was satisfied, one record matched one grade line, the record was on target
    -> PASS. The gate must not infer what ran from the log."""
    import verify_place
    gui_log = (">>> PLACE obj 0 -> slot 0 ok=True -> precise_success\n"
               ">>> plan-fallbacks[obj 0]: 0\n"
               ">>> cycle-entry[obj 7]: arm kp sum 12 kd sum 34 | arm |qv| 0.01\n"
               "!! cycle failed: some IK exception\n"
               ">>> plan-fallbacks[obj 7]: 0\n")
    path = _fixture([_record(5)])
    try:
        assert verify_place.results_gate(gui_log, path) is False
    finally:
        os.remove(path)


def test_an_ok_false_record_fails_and_check_names_the_cycle():
    """Round 3's architectural fix: run_cycle stamps an `ok: false` attempt record BEFORE the pick,
    so a cycle that crashes owns a record instead of owing none. The log here is deliberately CLEAN
    -- no '!! cycle failed:', the shape a SIGKILLed run leaves -- so only the record can fail it."""
    import verify_place
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "live.log")
        open(log, "w").write(RAN_LOG)
        json.dump([_record(5), {"obj": 7, "slot": 8, "ok": False, "attempted": True, "run": RUN}],
                  open(os.path.join(d, "place_results.json"), "w"))
        _manifest(os.path.join(d, "place_results.json"))
        assert verify_place.results_gate(RAN_LOG,
                                         os.path.join(d, "place_results.json")) is False
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            verify_place.check(log)
        line = next(l for l in buf.getvalue().splitlines() if "results-gate:" in l)
        assert "obj 7" in line and "slot 8" in line, f"the crashed cycle is not named: {line}"
    # ok:false must decide on its own. A stamp missing target_xy already fails via _xy_err_mm's
    # KeyError, which SHADOWS the flag -- give this one perfect numbers so only the flag can fail it.
    perfect = _record(0)
    perfect["ok"] = False
    path = _fixture([perfect])
    try:
        assert verify_place.results_gate(RAN_LOG, path) is False
    finally:
        os.remove(path)


def test_the_attempt_record_is_written_before_the_pick_in_run_cycle():
    """The record must exist BEFORE anything that can throw, or a crash at pick leaves no evidence
    that the cycle ran at all -- which is the whole defect. AST-only: no Isaac in this venv."""
    tree, _ = _play_isaac_ast()
    run_cycle = next(n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name == "run_cycle")
    opens = [n.lineno for n in ast.walk(run_cycle)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "_open_place_result"]
    picks = [n.lineno for n in ast.walk(run_cycle)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "pick"]
    assert opens, "run_cycle never stamps an attempt record"
    assert picks, "no self.pick(...) call found in run_cycle"
    assert max(opens) < min(picks), "the attempt record must be written before the pick"


def test_nothing_opts_the_evidence_prologue_out():
    """P0-4: the KEEP opt-out was the hole. It skipped the manifest write AND preserved the previous
    manifest and records, so a KEEP=1 process handed the verifier an internally consistent STALE
    pair. Scratch paths (below) replace it: every Demo() truncates and re-stamps its own file."""
    tree, src = _play_isaac_ast()
    # the CONDITION, not the body: a mention in a comment must not decide this either way.
    assert not [n for n in ast.walk(_demo_init(tree)) if isinstance(n, ast.If)
                and "PLACE_RESULTS_KEEP" in (ast.get_source_segment(src, n.test) or "")], (
        "an opt-out gates the truncation or the manifest write, which is the stale-pair hole")


def test_partial_write_fails_the_count_check():
    """The reverse of the vacuous-all() bug -- 3 records for 9 cycles graded in the log must
    not pass just because the 3 present happen to be on target. Two cycles graded here; only one
    record was written."""
    import verify_place
    two_cycle_log = RAN_LOG + ">>> PLACE obj 1 -> slot 4 ok=True -> precise_success\n"
    path = _fixture([_record(5)])          # one record, well within tolerance on its own
    try:
        assert verify_place.results_gate(two_cycle_log, path) is False
    finally:
        os.remove(path)


def test_empty_results_list_fails_even_when_the_log_shows_no_cycles():
    """Verbatim the defect this codebase's own fallback-gate history documents -- all() over
    an empty list is vacuously True. seen=0 here so the count check (above) can't be what catches
    it; only 'records and all(...)' does."""
    import verify_place
    path = _fixture([])
    try:
        assert verify_place.results_gate(NO_CYCLE_LOG, path) is False
    finally:
        os.remove(path)


def test_corrupt_json_fails_instead_of_crashing():
    """A sim killed mid-json.dump -- the normal way these runs end -- leaves invalid JSON. An
    acceptance gate must return a verdict, not a stack trace."""
    import verify_place
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        # truncated mid-write
        f.write('[{"obj": 0, "slot": 0, "xy": [0.0')
    try:
        assert verify_place.results_gate(RAN_LOG, path) is False
    finally:
        os.remove(path)


def test_old_schema_record_fails_instead_of_crashing():
    """rec["target_xy"] is unguarded; an old-schema record must FAIL the gate, not KeyError."""
    import verify_place
    path = _fixture([{"obj": 0, "slot": 0, "xy": [0.0, 0.0]}])      # no target_xy/target_z
    try:
        assert verify_place.results_gate(RAN_LOG, path) is False
    finally:
        os.remove(path)


def test_non_list_json_fails_instead_of_crashing_on_a_bare_scalar():
    """M1: results_gate does len(records)/iterates records on whatever json.load returned; a file
    containing a bare `5` reaches len(5) -- TypeError, not a graded FAIL."""
    import verify_place
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        f.write("5")
    try:
        assert verify_place.results_gate(RAN_LOG, path) is False
    finally:
        os.remove(path)


def test_non_list_json_fails_instead_of_crashing_on_a_dict():
    """M1: check() already guards records with isinstance(records, list); results_gate did not. A
    file containing `{"a": 1}` is valid, parseable JSON that is not a list of records."""
    import verify_place
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump({"a": 1}, f)
    try:
        assert verify_place.results_gate(RAN_LOG, path) is False
    finally:
        os.remove(path)


def test_z_error_is_checked_even_when_xy_is_perfect():
    """z is recorded and was once silently ignored -- an object that fell straight
    through to the shelf below would show 0mm xy error and pass."""
    import verify_place
    rec = _record(0)                       # xy exactly on target
    # 60mm low
    rec["z"] = rec["target_z"] + 0.06
    path = _fixture([rec])
    try:
        assert verify_place.results_gate(RAN_LOG, path) is False
    finally:
        os.remove(path)


def test_check_defers_to_results_gate_for_the_whole_log():
    """All the tests above call results_gate directly; none drove check(), which held a SECOND,
    duplicated copy of the absence logic. Mutation-tested by the reviewer: replacing `good =
    results_gate(...)` with `good = True` left the whole suite green. Stub out check_text (its own
    gates are covered elsewhere) and replace results_gate with a spy so this test is sensitive to
    exactly that mutation and nothing else."""
    import verify_place
    orig_check_text, orig_gate = verify_place.check_text, verify_place.results_gate
    calls = []

    def fake_check_text(*a, **kw):
        # isolate: only results_gate's verdict should move ok
        return True

    fd, log_path = tempfile.mkstemp(suffix=".log")
    try:
        with os.fdopen(fd, "w") as f:
            f.write("anything\n")
        verify_place.check_text = fake_check_text
        for verdict in (False, True):
            def fake_gate(txt, path=None, _v=verdict):
                calls.append(_v)
                return _v
            verify_place.results_gate = fake_gate
            assert verify_place.check(log_path) is verdict, (
                f"check() did not propagate results_gate() -> {verdict}")
        assert calls == [False, True], "check() did not call results_gate() on the CLI path"
    finally:
        verify_place.check_text = orig_check_text
        verify_place.results_gate = orig_gate
        os.remove(log_path)


def test_writer_treats_a_non_list_results_file_as_corrupt():
    """M2: json.load(f) happily parses a bare `{"a": 1}` or `5` -- valid JSON, wrong shape. Falling
    through to records.append(...) on that raises AttributeError/etc: no counter, no record, and
    the bad file is never repaired, so every later cycle in the run fails the same way. The load
    must treat a non-list result the same as unparseable JSON: same except branch, same
    'place-results-reset' tag, same reset-and-continue. AST-only: play_isaac.py cannot be imported
    here."""
    tree, _ = _play_isaac_ast()
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_place_records")
    load_try = next(n for n in ast.walk(fn) if isinstance(n, ast.Try)
                    and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                            and c.func.attr == "load" for c in ast.walk(n)))
    isinstance_checks = [n for n in ast.walk(load_try) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name) and n.func.id == "isinstance"]
    assert isinstance_checks, (
        "no isinstance() check in the PLACE_RESULTS load path -- a non-list value reaches "
        "records.append() unguarded")


def test_place_results_is_removed_at_process_start():
    """Nothing else clears PLACE_RESULTS between runs, so a killed-at-pick run could inherit a
    PREVIOUS run's good numbers and pass on them. 2(round2): the truncation must sit inside
    Demo.__init__, not module scope -- module scope fired on a bare `import play_isaac`
    (grip_bench.py, tools/idle_probe.py, tools/wheel_calib.py, tests/test_armfk_live.py all do
    that), silently deleting a previous run's acceptance evidence before anything is actually
    re-run, and with no try/except an OSError there killed the process before the sim even booted.
    AST-only: play_isaac.py cannot be imported here."""
    tree, _ = _play_isaac_ast()
    removes = [n for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "remove"
               and n.args and isinstance(n.args[0], ast.Name) and n.args[0].id == "PLACE_RESULTS"]
    assert removes, "no os.remove(PLACE_RESULTS) -- the results file is never truncated"
    init = _demo_init(tree)
    assert all(init.lineno <= n.lineno <= init.end_lineno for n in removes), (
        "os.remove(PLACE_RESULTS) must sit inside Demo.__init__, not module scope")
    guarded = [n for n in ast.walk(init) if isinstance(n, ast.Try)
               and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                       and c.func.attr == "remove" for c in ast.walk(n))]
    assert guarded, "os.remove(PLACE_RESULTS) must be wrapped in try/except OSError"


def test_write_place_result_precedes_the_cycle_end_fallback_tally_in_run_cycle():
    """3(round2): _fallback_tally prints AND CLEARS _plan_fallbacks. _write_place_result's except
    branch is the only site that can set 'place-results-reset' THIS cycle -- call the tally first
    and that increment lands in the NEXT cycle's tally, or is dropped entirely on the last cycle.
    Precisely the defect _fallback_tally's own docstring was written about."""
    tree, _ = _play_isaac_ast()
    run_cycle = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == "run_cycle")
    write_calls = [n.lineno for n in ast.walk(run_cycle)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == "_write_place_result"]
    # the retry-path tally is a distinct call to the same method and stays where it is; only the
    # CYCLE-END tally races `_write_place_result`'s possible fallback-tag increment.
    end_tally_calls = [n.lineno for n in ast.walk(run_cycle)
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr == "_fallback_tally"
                        and not any(kw.arg == "attempt" for kw in n.keywords)]
    assert write_calls, "no self._write_place_result(...) call found in run_cycle"
    assert end_tally_calls, "no cycle-end self._fallback_tally(...) call found in run_cycle"
    assert max(write_calls) < min(end_tally_calls), (
        "_write_place_result must run before the cycle-end _fallback_tally, or a "
        "'place-results-reset' increment from this cycle's write lands in the NEXT cycle's tally")


def test_write_place_result_is_called_inside_a_try_in_run_cycle():
    """_write_place_result is the LAST statement of run_cycle, outside every other instrument's
    try/except. get_world_poses() (stage churn), SHELF_SLOTS[slot_idx] (a bad DEMO_CYCLES), and
    os.makedirs/json.dump (an unwritable path) can each throw there; both callers call run_cycle
    bare, so unguarded it unwinds past them and closes the app before the remaining cycles run."""
    tree, _ = _play_isaac_ast()
    run_cycle = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == "run_cycle")

    def _calls_write(node):
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_write_place_result")

    bare = [s for s in run_cycle.body
            if not isinstance(s, ast.Try) and any(_calls_write(n) for n in ast.walk(s))]
    assert not bare, "_write_place_result is called outside a try/except in run_cycle"
    guarded = [s for s in run_cycle.body
               if isinstance(s, ast.Try) and any(_calls_write(n) for n in ast.walk(s))]
    assert guarded, "no try/except wraps the _write_place_result call in run_cycle"


def test_a_failed_attempt_stamp_increments_a_fallback_counter():
    """The handler prints `place-result open failed` and nothing in verify_place.py matches it,
    so to the gate it does not exist. A WRITE failure self-heals -- the ok:false stamp survives --
    but an open failure leaves nothing: no stamp, and if that cycle's pick then returns False
    without raising there is no grade line and no crash line either, so the count matches and the
    run PASSes on a cycle that is invisible from end to end."""
    tree, _ = _play_isaac_ast()
    run_cycle = next(n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name == "run_cycle")
    opener = next(t for t in ast.walk(run_cycle) if isinstance(t, ast.Try)
                  and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                          and c.func.attr == "_open_place_result" for s_ in t.body
                          for c in ast.walk(s_)))
    tags = [n.args[0].value for h in opener.handlers for n in ast.walk(h)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_fallback" and n.args and isinstance(n.args[0], ast.Constant)]
    assert "place-results-open-failed" in tags, (
        f"a failed attempt stamp is printed but not counted; tags here: {tags}")


def test_stale_records_cannot_stand_in_for_graded_cycles_the_writer_never_wrote():
    """`<` gives up the direction that actually happens. records > grades under this design only
    means a cycle crashed, which already fails on ok:false, so `!=` cost no real FAILs -- while `<`
    let 3 stale records stand in for 2 graded cycles whose writer was dead."""
    import verify_place
    path = _fixture([_record(5), _record(5, obj=1), _record(5, obj=2)])
    try:
        assert verify_place.results_gate(RAN_LOG + GRADED_2ND, path) is False
    finally:
        os.remove(path)


def test_a_list_of_non_dicts_fails_instead_of_raising():
    """results_gate catches only KeyError, so [1,2,3], ["a"] and [good, 7] raise AttributeError
    out of r.get(). check() already guards exactly this one line further down."""
    import verify_place
    for records in ([1, 2, 3], ["a"], [_record(5), 7]):
        path = _fixture(records)
        try:
            assert verify_place.results_gate(RAN_LOG + GRADED_2ND, path) is False, records
        finally:
            os.remove(path)


def test_check_reports_a_verdict_on_a_list_of_non_dicts_instead_of_a_traceback():
    """On the CLI path, results_gate returning False is not enough. check() carries on to build
    its detail line and max(records, key=_xy_err_mm) subscripts the int -- TypeError, which its
    `except KeyError` does not catch, so the acceptance gate dies instead of grading."""
    import verify_place
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "live.log")
        open(log, "w").write(RAN_LOG)
        json.dump([_record(5), 7], open(os.path.join(d, "place_results.json"), "w"))
        _manifest(os.path.join(d, "place_results.json"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert verify_place.check(log) is False
        line = next(l for l in buf.getvalue().splitlines() if "results-gate:" in l)
        assert "not records" in line, f"the bad entry is dropped, not named: {line}"


def test_check_names_the_crashed_cycle_the_log_alone_reports():
    """The branch chain has no case for the `!! cycle failed:` verdict, so a good placement
    followed by a crash printed `1 record(s), worst xy err 0mm ... -> FAIL`: a 0mm FAIL with no
    reason anywhere on the line."""
    import verify_place
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "live.log")
        open(log, "w").write(RAN_LOG + "!! cycle failed: some IK exception\n")
        json.dump([_record(0)], open(os.path.join(d, "place_results.json"), "w"))
        _manifest(os.path.join(d, "place_results.json"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            verify_place.check(log)
        line = next(l for l in buf.getvalue().splitlines() if "results-gate:" in l)
        assert "cycle failed" in line, f"the FAIL names no reason: {line}"


def test_run_gui_exports_place_results_alongside_its_log():
    """M3: run_gui.sh honours MORPH_LOG but never exported PLACE_RESULTS, so an overridden log left
    the writer on play_isaac's logs/ default and the reader on verify_place's dirname(log)."""
    line = next((l for l in open(os.path.join(ROOT, "run_gui.sh")).read().splitlines()
                 if l.strip().startswith("export PLACE_RESULTS")), None)
    assert line, "run_gui.sh never exports PLACE_RESULTS"
    assert "$LOG" in line, f"PLACE_RESULTS must follow MORPH_LOG, not a second default: {line}"


def test_placed_at_open_is_marked_before_the_dump_that_can_throw():
    """M4: `_placed_at_open` marks where THIS cycle's landings start. Set it after the dump and an
    unwritable path leaves it stale from the previous cycle -- the tail slice then sees the previous
    cycle's landing and the guard above is silently disarmed."""
    tree, _ = _play_isaac_ast()
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_open_place_result")
    marks = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Attribute) and t.attr == "_placed_at_open"
                     for t in n.targets)]
    dumps = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "_dump_place_records"]
    assert marks, "_open_place_result never marks where this cycle's landings start"
    assert dumps, "_open_place_result never dumps"
    assert max(marks) < min(dumps), "_placed_at_open must be set before the dump that can throw"


def test_a_dump_that_dies_mid_write_leaves_the_previous_records_intact():
    """M5: the dump truncated PLACE_RESULTS and then wrote into it, so a kill mid-dump destroyed
    the whole run's evidence -- and being killed is how these runs normally end."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "place_results.json")
        demo = _stub_demo(path)
        _cycle(demo, 0, 0, _on_target(0))
        good = json.load(open(path))
        orig_dump = json.dump
        json.dump = _boom
        try:
            demo._open_place_result(1, 1)
        except OSError:
            pass
        finally:
            json.dump = orig_dump
        assert json.load(open(path)) == good, "a dump that died mid-write took the run's evidence"


def test_every_record_carries_the_run_that_wrote_it():
    """M2: PLACE_RESULTS outlives a run whenever the truncation is opted out of or fails, so a
    record has to say which run it belongs to."""
    tree, _ = _play_isaac_ast()
    init = _demo_init(tree)
    assert [n for n in ast.walk(init) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == "_run_id" for t in n.targets)], (
        "Demo.__init__ stamps no run id")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "place_results.json")
        demo = _stub_demo(path)
        _cycle(demo, 0, 0, _on_target(0))
        _cycle(demo, 1, 1, None)
        recs = json.load(open(path))
        assert all(r.get("run") == demo._run_id for r in recs), (
            f"a record does not name its run: {recs}")


def test_records_from_an_older_run_are_not_this_run_s_evidence():
    """M2: one stale record could stand in for a cycle whose record this run never wrote -- the
    count matched and the stale numbers were good. Only the run the MANIFEST names counts."""
    import verify_place
    stale, fresh = _record(5), _record(5, obj=1)
    stale["run"], fresh["run"] = "111-1", "222-2"
    path = _fixture([stale, fresh], run="222-2")
    try:
        assert verify_place.results_gate(RAN_LOG + GRADED_2ND, path) is False
    finally:
        os.remove(path)


def test_a_prior_runs_record_cannot_stand_in_for_this_runs_graded_cycle():
    """reviewK P0-4, the offline mock. This run graded a cycle and was killed before the record
    landed -- the grade is printed in morph/place (`-> {_tier}`) and the record written later, in
    run_cycle -- leaving one record from the run BEFORE it. Record count matched grade count, and
    the gate took its run id off the last record, which was that older run's own: PASS on a
    previous run's numbers. The manifest is written before cycle 1 and says which run to demand."""
    import verify_place
    with tempfile.TemporaryDirectory() as d:
        rp = os.path.join(d, "place_results.json")
        prior = _record(5)
        # the run before this one
        prior["run"] = "111-1"
        json.dump([prior], open(rp, "w"))
        # ...and this one is 222-2
        _manifest(rp, "222-2")
        assert verify_place.results_gate(RAN_LOG, rp) is False
        log = os.path.join(d, "live.log")
        open(log, "w").write(RAN_LOG)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            verify_place.check(log)
        line = next(l for l in buf.getvalue().splitlines() if "results-gate:" in l)
        assert "111-1" in line, f"the record from another run is not named: {line}"


def test_the_run_manifest_names_this_run_and_survives_a_write_that_dies():
    """The writer side, EXECUTED (the stub runs Demo.__init__'s prologue). A manifest left
    half-written by the kill these runs end with is a manifest the gate cannot read, so it is
    written beside-and-renamed like the records; and a manifest that cannot be written at all is a
    run nothing can grade, so it is counted, not just printed -- and it must not raise, or it kills
    the process before the sim boots."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "place_results.json")
        demo = _stub_demo(path)
        assert json.load(open(path + ".run"))["run"] == demo._run_id, "the manifest names no run"
        orig_dump = json.dump
        # the kill: no except branch runs at all
        json.dump = _killed
        try:
            demo._write_run_manifest()
        except KeyboardInterrupt:
            pass
        finally:
            json.dump = orig_dump
        assert json.load(open(path + ".run"))["run"] == demo._run_id, (
            "a write that died mid-manifest left it unreadable")
        json.dump = _boom
        try:
            # must return, not raise
            demo._write_run_manifest()
        finally:
            json.dump = orig_dump
        assert "place-results-manifest-failed" in demo.fallbacks, (
            f"a manifest that could not be written is printed but not counted: {demo.fallbacks}")


def test_the_run_manifest_is_written_before_any_cycle_can_run():
    """It has to name the run BEFORE a record carrying that run id can exist, or cycle 1's record
    is evidence no manifest claims. Demo.__init__ is the only place that runs before both
    launchers' run_cycle."""
    tree, _ = _play_isaac_ast()
    init = _demo_init(tree)

    def _calls(node, name):
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == name)

    writes = [n.lineno for n in ast.walk(init) if _calls(n, "_write_run_manifest")]
    cycles = [n.lineno for n in ast.walk(tree) if _calls(n, "run_cycle")]
    assert writes, "Demo.__init__ writes no run manifest"
    assert cycles, "no run_cycle call site found in play_isaac.py"
    assert max(writes) < min(cycles), "the run manifest must be written before any cycle runs"
    assert not [n for n in ast.walk(init) if isinstance(n, ast.If)
                and any(_calls(c, "_write_run_manifest") for c in ast.walk(n))], (
        "the manifest write is conditional: whatever skips it preserves the PREVIOUS manifest, and "
        "the previous results file beside it grades as this run's evidence")


def test_no_run_manifest_fails_when_a_cycle_wrote_a_record():
    """Absence is the same hole: with nothing naming this run, a whole file of a PREVIOUS run's
    records reads as this run's evidence. The record here predates the run stamp AND is perfectly
    placed, so nothing but the missing manifest can fail it -- comparing run ids alone would find
    None == None and pass. (Legacy logs stay gradeable -- see
    test_missing_file_passes_when_no_cycles_ran, which has no manifest either and still PASSes
    because nothing ran.)"""
    import verify_place
    with tempfile.TemporaryDirectory() as d:
        rp = os.path.join(d, "place_results.json")
        legacy = _record(0)
        # written before records named their run
        legacy.pop("run")
        json.dump([legacy], open(rp, "w"))
        assert verify_place.results_gate(RAN_LOG, rp) is False
        log = os.path.join(d, "live.log")
        open(log, "w").write(RAN_LOG)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            verify_place.check(log)
        line = next(l for l in buf.getvalue().splitlines() if "results-gate:" in l)
        assert "no run manifest" in line, f"the FAIL names no reason: {line}"


def test_a_stale_record_a_failed_truncation_left_behind_fails_beside_this_runs_own():
    """Demo.__init__ PRINTS and continues when it cannot truncate PLACE_RESULTS, so the file can
    hold a previous run's records alongside this run's. Taking the run id off the LAST record hid
    exactly that: the leftovers were filtered out, the count matched the grade lines, and the run
    PASSed on a file it did not own. Every record has to be this run's."""
    import verify_place
    stale = _record(0)
    # left by the run before this one
    stale["run"] = "111-1"
    path = _fixture([stale, _record(0, obj=1)])
    try:
        assert verify_place.results_gate(RAN_LOG, path) is False
    finally:
        os.remove(path)


def test_no_opt_out_may_leave_a_stale_manifest_and_results_pair_to_grade():
    """P0-4, CRITICAL 1: PLACE_RESULTS_KEEP=1 skipped the manifest write AND preserved the previous
    run's manifest and records -- an internally consistent stale pair whose record count matches
    this run's grade lines, so it PASSed as this run's evidence. The prologue is unconditional now:
    even with the old opt-out set and the truncation dying, this run re-stamps the manifest and the
    leftovers show up as foreign."""
    import verify_place
    with tempfile.TemporaryDirectory() as d:
        rp = os.path.join(d, "place_results.json")
        stale = _record(0)
        # a perfect record, from the run before
        stale["run"] = "111-1"
        json.dump([stale], open(rp, "w"))
        # ...and the manifest that agrees with it
        _manifest(rp, "111-1")
        orig_remove = os.remove
        # the opt-out that used to gate both
        os.environ["PLACE_RESULTS_KEEP"] = "1"
        # ...and a truncation that cannot clear it
        os.remove = _boom
        try:
            demo = _stub_demo(rp)
        finally:
            os.remove = orig_remove
            os.environ.pop("PLACE_RESULTS_KEEP", None)
        assert json.load(open(rp)) == [stale], "the stale records were cleared; nothing to grade"
        assert demo._run_id != "111-1"
        assert verify_place.results_gate(RAN_LOG, rp) is False
        log = os.path.join(d, "live.log")
        open(log, "w").write(RAN_LOG)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            verify_place.check(log)
        line = next(l for l in buf.getvalue().splitlines() if "results-gate:" in l)
        assert "111-1" in line, f"the stale record is not named: {line}"


def test_a_manifest_write_that_fails_takes_the_stale_pair_with_it():
    """P0-4, CRITICAL 2: the truncation dies, then the manifest write dies, and the previous run's
    manifest and records are left standing -- agreeing with each other, matching this log's grade
    count, graded as this run's. A run that cannot record its identity must leave NO gradeable
    evidence, so both are removed and the gate fails for absence."""
    import verify_place
    with tempfile.TemporaryDirectory() as d:
        rp = os.path.join(d, "place_results.json")
        stale = _record(0)
        stale["run"] = "111-1"
        json.dump([stale], open(rp, "w"))
        _manifest(rp, "111-1")
        orig_remove, orig_dump = os.remove, json.dump
        os.remove, json.dump = _boom_once(orig_remove), _boom
        try:
            demo = _stub_demo(rp)
        finally:
            os.remove, json.dump = orig_remove, orig_dump
        assert "place-results-manifest-failed" in demo.fallbacks, demo.fallbacks
        assert not os.path.exists(rp + ".run"), "the previous run's manifest survived"
        assert not os.path.exists(rp), "the previous run's records survived"
        assert verify_place.results_gate(RAN_LOG, rp) is False


def test_a_results_file_no_manifest_claims_fails_even_with_no_cycles_in_the_log():
    """The absence rule, pinned: manifest missing AND results missing AND no cycles -> PASS (a
    legacy log, predating the writer); manifest missing and results PRESENT -> FAIL, whatever the
    log says. A file no run claims is not evidence, and a quiet log is not proof it is stale."""
    import verify_place
    with tempfile.TemporaryDirectory() as d:
        rp = os.path.join(d, "place_results.json")
        json.dump([_record(0)], open(rp, "w"))            # left behind; no manifest beside it
        assert verify_place.results_gate(NO_CYCLE_LOG, rp) is False
        assert verify_place.results_gate(NO_CYCLE_LOG, os.path.join(d, "gone.json")) is True


MANIFEST_FAILED_LOG = (">>> RUN MANIFEST WRITE FAILED: /ro/place_results.json.run: [Errno 30] "
                       "Read-only file system -- attempted to remove any stale manifest and "
                       "results file; verify_place cannot grade this run\n")


def test_a_stale_pair_cannot_grade_a_run_whose_manifest_write_failed():
    """P0-4, the last AUTOMATIC false-PASS: the manifest write dies, BOTH best-effort unlinks die,
    and the process dies before its first plan-fallbacks tally. The previous run's manifest and
    records survive agreeing with each other and matching this log's grade count, and `legacy`
    excuses the missing tally -> PASS, with no manual action anywhere. The flushed marker is the
    only witness that this run could not name itself, so it alone has to fail the gate."""
    import verify_place
    with tempfile.TemporaryDirectory() as d:
        rp = os.path.join(d, "place_results.json")
        stale = _record(0)
        # a perfect record, from the run before
        stale["run"] = "111-1"
        json.dump([stale], open(rp, "w"))
        # ...and the manifest that agrees with it
        _manifest(rp, "111-1")
        assert verify_place.results_gate(RAN_LOG, rp) is True, (
            "fixture no longer reproduces the pair that PASSes when the marker is absent")
        assert verify_place.results_gate(RAN_LOG + MANIFEST_FAILED_LOG, rp) is False
        log = os.path.join(d, "live.log")
        open(log, "w").write(RAN_LOG + MANIFEST_FAILED_LOG)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert verify_place.check(log) is False
        line = next(l for l in buf.getvalue().splitlines() if "results-gate:" in l)
        assert "could not record its identity" in line, f"the FAIL names no reason: {line}"


def test_the_marker_the_verifier_greps_is_the_one_play_isaac_prints():
    """The gate above is a string match across two files: rewording the print in play_isaac.py
    would reopen the hole silently, with every test still green."""
    import verify_place
    assert verify_place.MANIFEST_FAILED in _play_isaac_ast()[1], (
        f"play_isaac.py prints no {verify_place.MANIFEST_FAILED!r}")


if __name__ == "__main__":
    fails = 0
    for name in sorted(n for n in dir() if n.startswith("test_")):
        try:
            globals()[name]()
            print(f"PASS {name}")
        except Exception as e:
            fails += 1
            print(f"FAIL {name}: {e}")
    print("\nall passed" if not fails else f"\n{fails} failed")
    sys.exit(1 if fails else 0)
