# Planner benchmark problems

Requests captured verbatim from one graded pick-and-place cycle (`ARM_PLAN_DUMP`, obj 0 -> slot 0),
with only `model`/`lut` rewritten to repo-relative paths. `reach-nofingers` is `reach` with
`fingers: null`: the swept-union hand instead of the commanded one, a different problem, not a
variant. `free-space` and `unreachable` are synthetic floor/ceiling cases. There is no `carry`
problem: no request in the cycle carried a `held` box.

The set is a snapshot of one cycle's obstacle geometry. Any change to what `arm_obstacles` emits
makes it stale: re-capture, do not hand-edit.

Acceptance, run alone on an idle machine (latency is a machine-state number here):

    ./.venv/bin/python3 tools/bench_plan.py --seeds 100 --baseline tools/bench_plan_baseline.json

A change in either direction exits 1 and names the metric; re-baseline only on purpose, with
`--write-baseline tools/bench_plan_baseline.json`, and say why in the commit. The loader refuses
the set once MARGIN, the deck pad or the planning bounds move; `verify_branch.sh` runs one seed of
`free-space` per commit and nothing more.
