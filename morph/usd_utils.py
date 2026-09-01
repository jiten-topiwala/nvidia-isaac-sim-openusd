"""USD stage helpers. Extracted from play_isaac.py.

Imports `pxr`, so this module is only importable AFTER `SimulationApp` has been constructed —
which is guaranteed, because the entry script boots the app before it imports anything from
`morph`. See the bootstrap check in `smoke_test.py`.
"""


def find_path(stage, name):
    """Path string of the first prim named `name`, or None.

    Traverses the whole stage, so callers resolve a prim ONCE and cache it — doing this per step is
    what made the gripper lookup a measurable cost in the first port.
    """
    for p in stage.Traverse():
        if p.GetName() == name:
            return p.GetPath().pathString
    return None
