"""The floor plan, as axis-aligned rectangles: one definition, two readers.

`morph/config.py` reads the racks for the spawn keep-outs; `nav_plan.py` reads racks and walls as
its obstacle set, from a separate interpreter that must not pick up this project's environment
profile. Nothing here imports, prints or reads the environment -- keep it that way.
"""

# (x0, x1, y0, y1), metres, world frame. nav_plan.py inflates them by ROBOT_RADIUS at check time.
RACK_RECTS = (
    (2.55, 4.34, -3.96, -3.41),   # middle-front row: the placement target, carrying the slot sites
    (4.34, 7.96, -3.96, -3.41),   # middle-front row, east of the target
    (2.54, 7.96, -4.55, -3.97),   # middle-back row
    (0.72, 7.97, -7.96, -7.42),   # south rack row
    (2.54, 7.96, -0.66, -0.04),   # north rack row
)

# The floor bounds. Obstacles to the base planner; irrelevant to a spawn, which is bounded by its
# own zones.
WALL_RECTS = (
    (-0.1, 0.1, -8.0, 0.0),       # west
    (7.9, 8.1, -8.0, 0.0),        # east
    (0.0, 8.0, -0.1, 0.1),        # north
)
