"""`morph/navigation.py`: the arrival error a completed drive reports.

Offline: no Isaac. Run: ./.venv/bin/python3 tests/test_navigation.py

WHAT THIS FILE PINS.

A drive that ends must report how far it is from the GOAL. Both drive loops track an internal
reference pose, and `drive_to`'s wheel branch clamps that reference to within 50 mm of the live
pose every step, so a reference-vs-live readout is bounded by construction and reads healthy
whether the base docked or wedged against a shelf.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morph.navigation import arrival_error, nav_endpoint_shift


def test_a_drive_that_reached_its_goal_reports_zero():
    mm, deg = arrival_error((1.0, -2.0), 0.5, (1.0, -2.0, 0.5))
    assert mm == 0.0 and deg == 0.0, (mm, deg)


def test_the_error_is_measured_against_the_goal_not_the_live_pose():
    """A base 2 m short of its goal reports 2 m, not the reference's bounded residue."""
    mm, _ = arrival_error((3.0, -2.0), 0.0, (1.0, -2.0, 0.0))
    assert abs(mm - 2000.0) < 1e-6, mm


def test_the_yaw_error_wraps_the_short_way():
    """+179 deg against -179 deg is 2 deg apart, not 358."""
    _, deg = arrival_error((0.0, 0.0), math.radians(179), (0.0, 0.0, math.radians(-179)))
    assert abs(abs(deg) - 2.0) < 1e-6, deg


def test_the_readout_a_stalled_wheel_drive_produces_is_not_bounded_by_the_clamp():
    """`drive_to`'s wheel branch pins reference-vs-live to 50 mm. A base wedged 2 m short must
    not be able to report inside that bound, or arrival cannot be told from exhaustion."""
    mm, _ = arrival_error((3.0, -2.0), 0.0, (1.0, -2.0, 0.0))
    assert mm > 50.0, mm


def test_a_no_plan_is_counted_and_still_ends_the_attempt():
    """The refusal must propagate: place/__init__.py dock-snaps and inserts on the statement after
    its drive_to, so a counted return would carry the object into an unplanned straight line."""
    import morph.navigation as nav

    class _Demo:
        _plan_fallbacks = None
        dt = 1.0 / 240

        def _fallback(self, reason):
            d = self._plan_fallbacks
            if d is None:
                d = self._plan_fallbacks = {}
            d[reason] = d.get(reason, 0) + 1

        def base_pose(self):
            return (0.0, 0.0, 0.0)

    def _boom(*a, **k):
        raise RuntimeError("no path after 3 tries: {'error': 'No path found'}")

    demo, keep = _Demo(), nav.plan_path
    nav.plan_path = _boom
    try:
        raised = False
        try:
            nav.NavigationMixin.drive_to(demo, (1.0, -2.0), 0.0)
        except RuntimeError:
            raised = True
    finally:
        nav.plan_path = keep
    assert raised, "the no-plan was swallowed; the caller would drive on"
    assert demo._plan_fallbacks == {"nav-no-plan": 1}, demo._plan_fallbacks


def test_a_nudged_endpoint_is_reported():
    """nav_plan's `nudge` moves an invalid start or goal by up to 0.8 m and returns the path from
    the moved point. drive_to then steers at path[1] of a path not rooted where the robot is."""
    smm, gmm = nav_endpoint_shift((0.0, 0.0), (5.0, 0.0), [[0.10, 0.0], [2.0, 0.0], [4.85, 0.0]])
    assert abs(smm - 100.0) < 1e-6, smm
    assert abs(gmm - 150.0) < 1e-6, gmm


def test_an_unmoved_plan_reports_no_shift():
    smm, gmm = nav_endpoint_shift((0.0, 0.0), (5.0, 0.0), [[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]])
    assert smm == 0.0 and gmm == 0.0, (smm, gmm)


def test_a_path_too_short_to_have_endpoints_reports_nothing():
    """plan_path can in principle return a single point; indexing it blind raises IndexError."""
    assert nav_endpoint_shift((0.0, 0.0), (5.0, 0.0), []) is None


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
