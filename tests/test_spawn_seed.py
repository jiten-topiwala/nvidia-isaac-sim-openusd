"""`morph.config.spawn_seed`: the object scatter is random per run unless SEED pins it.

Offline: no Isaac. Run: ./.venv/bin/python3 tests/test_spawn_seed.py

WHAT THIS FILE PINS.

A demo must show a fresh object layout every run, and any run must still be reproducible
afterwards -- so an unpinned seed is drawn, not assumed, and the scatter prints the value it drew.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morph.config import MIN_OBJ_SEPARATION, fallback_spawn_xy, spawn_seed


def _with_seed(value):
    keep = os.environ.pop("SEED", None)
    if value is not None:
        os.environ["SEED"] = value
    try:
        return spawn_seed()
    finally:
        os.environ.pop("SEED", None)
        if keep is not None:
            os.environ["SEED"] = keep


def test_an_explicit_seed_is_used_verbatim():
    assert _with_seed("7") == 7


def test_seed_zero_is_honoured_and_not_read_as_unset():
    """`0` is falsy; reading the env with a truthiness test would silently randomise it."""
    assert _with_seed("0") == 0


def test_an_unset_seed_draws_a_fresh_layout_each_run():
    drawn = {_with_seed(None) for _ in range(8)}
    assert len(drawn) > 1, f"the unpinned scatter repeated one layout: {drawn}"


def test_a_drawn_seed_can_be_replayed_through_the_env():
    drawn = _with_seed(None)
    assert _with_seed(str(drawn)) == drawn


def test_the_fallback_refuses_a_blocked_point():
    """The rejection sampler's fallback used a fixed row that walks through the robot keep-out.
    Whatever it returns must pass the same keep-out every sampled spawn passes."""
    import numpy as np

    blocked_box = lambda x, y: 3.0 <= x <= 5.0 and -6.0 <= y <= -4.0
    p = fallback_spawn_xy([], blocked_box)
    assert p is not None and not blocked_box(p[0], p[1]), p


def test_the_fallback_keeps_its_distance_from_what_is_already_placed():
    import numpy as np

    placed = [np.array([0.5, -5.0])]
    p = fallback_spawn_xy(placed, lambda x, y: False)
    assert float(np.linalg.norm(p - placed[0])) >= MIN_OBJ_SEPARATION, p


def test_the_fallback_returns_nothing_rather_than_a_blocked_point():
    """No legal square left: refuse. Placing an object inside the robot is not a fallback."""
    assert fallback_spawn_xy([], lambda x, y: True) is None


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
