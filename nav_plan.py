import sys, json, math, os
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

# RRTConnect samples randomly, so NAV_SEED is needed for a reproducible run.
_SEED = os.environ.get("NAV_SEED")
if _SEED:
    ou.RNG.setSeed(int(_SEED))

FLOOR_X      = (0.0, 8.0)
FLOOR_Y      = (-8.0, 0.0)
ROBOT_RADIUS = 0.65   # covers the chassis's widest yaw profile

# (x_min, x_max, y_min, y_max), inflated by ROBOT_RADIUS at check time.
#
# The pick-to-shelf detour is geometry, not a planner fault: with the inflation, the only opening
# between the two docks is the west corridor, so a first leg heading away from the goal is optimal.
OBSTACLE_RECTS = [
    (2.55, 4.34, -3.96, -3.41),   # middle-front row: the placement target, carrying the slot sites
    (4.34, 7.96, -3.96, -3.41),   # middle-front row, east of the target
    (2.54, 7.96, -4.55, -3.97),   # middle-back row
    (0.72, 7.97, -7.96, -7.42),   # south rack row
    (2.54, 7.96, -0.66, -0.04),   # north rack row
    # walls: west, east, north
    (-0.1, 0.1,  -8.0,  0.0),
    (7.9,  8.1,  -8.0,  0.0),
    (0.0,  8.0,  -0.1,  0.1),
]

class ValidityChecker(ob.StateValidityChecker):
    def __init__(self, si, discs=None):
        super().__init__(si)
        # [[x, y, r], ...] floor-object keep-outs. r is the TOTAL clearance, not inflated again by
        # ROBOT_RADIUS -- that is the rotating-diagonal profile, and it would swallow the pick dock.
        self.discs = discs or []
    def isValid(self, state):
        x, y = state[0], state[1]
        if not (FLOOR_X[0] <= x <= FLOOR_X[1] and FLOOR_Y[0] <= y <= FLOOR_Y[1]):
            return False
        r = ROBOT_RADIUS
        for (x0, x1, y0, y1) in OBSTACLE_RECTS:
            if (x0 - r) <= x <= (x1 + r) and (y0 - r) <= y <= (y1 + r):
                return False
        for (cx, cy, cr) in self.discs:
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) < cr * cr:
                return False
        return True

def nudge(si, space, xy, radius=0.8, tries=24, steps=16):
    """Nearest valid state to `xy`, searching outward.

    Nearest, not first-valid-at-a-fixed-radius: a goal on an obstacle boundary would otherwise be
    shoved the whole search radius, off the slot it was meant to park in front of.
    """
    state = space.allocState()
    for ri in range(1, int(steps) + 1):
        r = radius * ri / float(steps)
        for i in range(tries):
            angle = 2 * math.pi * i / tries
            state[0] = xy[0] + r * math.cos(angle)
            state[1] = xy[1] + r * math.sin(angle)
            if si.isValid(state):
                return state
    return None

def plan(start_xy, goal_xy, solve_time=1.5, use_rrt_connect=True, discs=None):
    space = ob.RealVectorStateSpace(2)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0, FLOOR_X[0]);  bounds.setHigh(0, FLOOR_X[1])
    bounds.setLow(1, FLOOR_Y[0]);  bounds.setHigh(1, FLOOR_Y[1])
    space.setBounds(bounds)

    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(ValidityChecker(si, discs))
    si.setStateValidityCheckingResolution(0.01)
    si.setup()

    start = space.allocState()
    start[0], start[1] = float(start_xy[0]), float(start_xy[1])
    goal  = space.allocState()
    goal[0],  goal[1]  = float(goal_xy[0]),  float(goal_xy[1])

    if not si.isValid(start): start = nudge(si, space, start_xy)
    if start is None: return None
    goal_was_nudged = False
    if not si.isValid(goal):
        goal  = nudge(si, space, goal_xy)
        goal_was_nudged = True
    if goal is None:  return None
    final_xy = [float(goal[0]), float(goal[1])] if goal_was_nudged else list(goal_xy)

    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start, goal)

    # RRTConnect exits on the first feasible path; RRTstar always spends the full budget.
    if use_rrt_connect:
        planner = og.RRTConnect(si)
    else:
        planner = og.RRTstar(si)
    planner.setProblemDefinition(pdef)
    planner.setRange(0.5)
    planner.setup()

    if not planner.solve(solve_time):
        return None

    path = pdef.getSolutionPath()
    # Simplify before interpolating: the first feasible path overshoots, and a loaded base -- wider
    # than the planning model -- can wedge on those detours. Reverted below if a shortcut breaks.
    if os.environ.get("AH_NAV_SIMPLIFY", "0") == "1":
        try:
            _raw_n = path.getStateCount()
            og.PathSimplifier(si).simplifyMax(path)
            if not path.check():
                path = pdef.getSolutionPath()   # a shortcut broke validity -- revert
            else:
                sys.stderr.write(
                    f"[plan] path simplified {_raw_n}->{path.getStateCount()} states\n")
        except Exception as _e:
            sys.stderr.write(f"[plan] simplify skipped: {_e}\n")
            path = pdef.getSolutionPath()
    path.interpolate(40)
    waypoints = [[path.getState(i)[0], path.getState(i)[1]]
                 for i in range(path.getStateCount())]
    waypoints[-1] = final_xy
    return waypoints

if __name__ == "__main__":
    try:
        inp  = json.loads(sys.stdin.read())
        path = plan(inp["start"], inp["goal"], inp.get("solve_time", 3.0), discs=inp.get("discs"))
        print(json.dumps(path if path else {"error": "No path found"}), flush=True)
    except Exception as e:
        print(json.dumps({"error": str(e)}), flush=True)
