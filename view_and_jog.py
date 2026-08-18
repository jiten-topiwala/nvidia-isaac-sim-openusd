"""Phase 1 — load the imported world and jog the arm joints.

Proves the low-level port: robot + scene render under PhysX and every articulated
joint is controllable. Scene load/fixups live in scene.py.

    cd M1
    /path/to/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh view_and_jog.py
        ISAAC_HEADLESS=1   no window (CI / verification)
        STILL=1            hold the arm instead of jogging
        JOG_SECONDS=20     animation length (default 15)
"""
import math
import os

# `scene.py` sits next to this file, so no path bootstrap is needed — but run from THIS
# directory, because scene.py resolves usd/ relative to itself.
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_HEADLESS", "0") == "1"
STILL = os.environ.get("STILL", "0") == "1"
simulation_app = SimulationApp({"headless": HEADLESS})

import numpy as np
import scene
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view

# (joint-name substring, sine amplitude rad) — small swing around the park pose.
JOG = [
    ("ColumnLeftBearingJoint", 0.15), ("ColumnRightBearingJoint", 0.15),
    ("ArmLeftJoint", 0.15), ("HandBearingJoint", 0.30),
    ("finger_a_joint_1", 0.40), ("finger_b_joint_1", 0.40), ("finger_c_joint_1", 0.40),
]


def main() -> None:
    world, robot, names, q0 = scene.load_scene(simulation_app)
    if not HEADLESS:
        set_camera_view(eye=[6.0, -10.0, 3.2], target=[3.4, -5.2, 0.7])

    targets = [(i, q0[i], amp) for sub, amp in JOG for i, n in enumerate(names) if sub in n]
    if not targets:
        raise SystemExit("No jog joints matched.")
    print(f">>> jogging {len(targets)} joints", flush=True)

    ctrl = robot.get_articulation_controller()
    dt = world.get_physics_dt()
    steps = int(float(os.environ.get("JOG_SECONDS", "15")) / dt)
    moved = np.zeros(len(names))

    def command(k):
        cmd = q0.copy()
        for i, start, amp in targets:
            cmd[i] = start + (0.0 if STILL else amp) * math.sin(2.0 * math.pi * 0.25 * k * dt)
        ctrl.apply_action(ArticulationAction(joint_positions=cmd))

    for k in range(steps):
        command(k)
        world.step(render=not HEADLESS)
        moved = np.maximum(moved, np.abs(np.asarray(robot.get_joint_positions()) - q0))

    if STILL:
        print(">>> STILL — held pose, no jog.", flush=True)
    else:
        ok = all(moved[i] > 0.3 * amp for i, _, amp in targets)
        for i, _, amp in targets:
            print(f"  {'OK ' if moved[i] > 0.3*amp else '!! '}{names[i]:<28} moved {moved[i]:.3f} rad", flush=True)
        print(">>> JOG PASS" if ok else ">>> JOG FAIL", flush=True)

    if not HEADLESS:
        print(">>> window open — close it to exit.", flush=True)
        k = steps
        while simulation_app.is_running():
            command(k)
            world.step(render=True)
            k += 1


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
