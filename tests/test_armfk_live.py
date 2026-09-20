#!/usr/bin/env python3
"""Ground-truth `ArmModel` against the live articulation -- THE GATE. Writes random joint vectors
(actuated plus the closure LUT's passives) into the real robot, steps unrendered, and compares
every link's world pose against `fk`.  ISAAC_HEADLESS=1 "$ISAAC/python.sh" tests/test_armfk_live.py"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("ISAAC_HEADLESS", "1")

# Demo() truncates PLACE_RESULTS and stamps its own run manifest beside it; this is not a demo run,
# so send both to scratch. play_isaac reads PLACE_RESULTS at module scope: this must precede it.
os.environ.setdefault("PLACE_RESULTS",
                      os.path.join(tempfile.gettempdir(), "armfk_live_place_results.json"))

import play_isaac                                                   # noqa: E402  boots SimulationApp
from isaacsim.core.prims import SingleXFormPrim                     # noqa: E402
from morph.usd_utils import find_path                               # noqa: E402
from morph.arm.model import ArmModel, _quat_to_R  # noqa: E402

N = 20
TOL_M = 0.005                                               # model vs live link position
TOL_RAD = 0.005                                             # commanded vs held joint value

demo = play_isaac.Demo()
M = ArmModel.load(os.path.join(ROOT, "usd/_arm_model.json"),
                  os.path.join(ROOT, "usd/_closure_lut.json"))

lo, hi = M.bounds()
# The region we actually plan in; h1 and dh are sampled separately to stay inside the LUT.
lo[1], hi[1] = 0.10, 1.30
lo[3], hi[3] = 0.05, 0.50
lo[4:], hi[4:] = -0.6, 0.6
DH_LO, DH_HI = M.lut["dh"][0], M.lut["dh"][-1]

# Reject floor/chassis poses: PhysX pushes the joint off its command -- contact, not model error.
OBSTACLES = [(np.array([[-5.0, -5.0, -1.0], [5.0, 5.0, 0.02]]), "world"),        # floor
             (np.array([[-0.35, -0.30, 0.0], [0.35, 0.30, 0.35]]), "chassis")]   # chassis plate

prims = {ln: SingleXFormPrim(find_path(demo.stage, ln)) for ln in M.link_aabb}
base_prim = SingleXFormPrim(find_path(demo.stage, "base"))
rng = np.random.default_rng(0)
q_home = np.asarray(demo.robot.get_joint_positions(), float).copy()

SETTLE = 60                                                 # 0.25 s at 240 Hz
worst, worst_at, used = 0.0, None, 0
per_link = {}
ALL = M.Q8 + M.m["passives"]
track, track1 = {}, {}
# rejection sampling; stop at N accepted
for n in range(N * 4):
    if used >= N:
        break
    q8 = rng.uniform(lo, hi)
    dh = float(rng.uniform(DH_LO, DH_HI))
    # h2 from h1 + a dh inside the LUT
    q8[2] = min(max(q8[1] + dh, lo[2]), hi[2])
    dh = q8[2] - q8[1]
    # a config the closure cannot hold is one `_apply` CLIPS, so this would measure the clip
    if (not M.in_lut_domain(dh, q8[3]) or M.passive_violations(dh, q8[3])
            or M.collides(q8, OBSTACLES)):
        continue
    used += 1
    q = q_home.copy()
    for name, v in zip(M.Q8, q8):
        q[demo.idx[name]] = v
    for name, v in M.passives(dh, q8[3]).items():               # the closure, written explicitly
        if name in demo.idx:
            q[demo.idx[name]] = v
    demo._force(q)
    demo._apply(q)
    demo.world.step(render=False)
    act1 = np.asarray(demo.robot.get_joint_positions(), float)
    # let the drives converge
    for _ in range(SETTLE):
        demo._force(q)
        demo._apply(q)
        demo.world.step(render=False)

    act = np.asarray(demo.robot.get_joint_positions(), float)
    for name in ALL:
        if name in demo.idx:
            i = demo.idx[name]
            track[name] = max(track.get(name, 0.0), abs(float(act[i]) - q[i]))
            track1[name] = max(track1.get(name, 0.0), abs(float(act1[i]) - q[i]))
    _wn = max((n2 for n2 in ALL if n2 in demo.idx),
              key=lambda n2: abs(float(act[demo.idx[n2]]) - q[demo.idx[n2]]))
    print(f">>> s{n:02d} h1 {q8[1]:.3f} dh {dh:+.3f} a1 {q8[3]:.3f} | worst joint {_wn} "
          f"cmd {q[demo.idx[_wn]]:+.4f} act {float(act[demo.idx[_wn]]):+.4f}", flush=True)
    q8_act = np.array([float(act[demo.idx[nm]]) for nm in M.Q8])
    pas_act = {nm: float(act[demo.idx[nm]]) for nm in M.m["passives"] if nm in demo.idx}

    bp, bq = base_prim.get_world_pose()
    Rb = _quat_to_R(*[float(v) for v in bq])
    bp = np.asarray(bp, float)
    pred = M.fk(q8_act, passives=pas_act)                   # model judged on the MEASURED state
    for link, prim in prims.items():
        live = np.asarray(prim.get_world_pose()[0], float)
        model = Rb @ pred[link][0] + bp
        err = float(np.linalg.norm(live - model))
        per_link[link] = max(per_link.get(link, 0.0), err)
        if err > worst:
            worst, worst_at = err, (n, link, np.round(q8_act, 3).tolist())

print(f">>> joint tracking, worst |commanded - actual|: after 1 step vs after {SETTLE}:", flush=True)
for nm in sorted(track1, key=lambda k: -track1[k]):
    print(f">>>   {nm:<28} 1-step {track1[nm] * 1000:8.2f}   settled {track[nm] * 1000:8.2f}  mrad/mm",
          flush=True)
for link in sorted(per_link, key=lambda k: -per_link[k]):
    print(f">>>   {link:<24} worst {per_link[link] * 1000:7.2f}mm", flush=True)
print(f">>> armfk live check: {used} collision-free samples, "
      f"worst link error {worst * 1000:.2f}mm at {worst_at}", flush=True)

# Phase 2: IK live -- test_armfk.py's round trip checks IK against FK, which is circular.
ik_worst, ik_solved, ik_n = 0.0, 0, 0
rng2 = np.random.default_rng(7)
for _ in range(10):
    q8 = rng2.uniform(lo, hi)
    dh = float(rng2.uniform(DH_LO, DH_HI))
    q8[2] = min(max(q8[1] + dh, lo[2]), hi[2])
    if (not M.in_lut_domain(q8[2] - q8[1], q8[3]) or M.passive_violations(q8[2] - q8[1], q8[3])
            or M.collides(q8, OBSTACLES)):
        continue
    ik_n += 1
    tgt_p, tgt_R = M.fk(q8)[M.hand_link]                    # a pose known to be reachable
    got = M.ik(tgt_p, tgt_R, seed=np.clip(q8 + rng2.normal(0, 0.15, 8), lo, hi), bounds=(lo, hi))
    if got is None:
        continue
    q_sol = got[0]
    q = q_home.copy()
    for name, v in zip(M.Q8, q_sol):
        q[demo.idx[name]] = v
    for name, v in M.passives(q_sol[2] - q_sol[1], q_sol[3]).items():
        if name in demo.idx:
            q[demo.idx[name]] = v
    for _ in range(SETTLE):
        demo._force(q)
        demo._apply(q)
        demo.world.step(render=False)
    bp2, bq2 = base_prim.get_world_pose()
    live = np.asarray(prims[M.hand_link].get_world_pose()[0], float)
    want = _quat_to_R(*[float(v) for v in bq2]) @ tgt_p + np.asarray(bp2, float)
    ik_worst = max(ik_worst, float(np.linalg.norm(live - want)))
    ik_solved += 1
print(f">>> armfk IK live check: {ik_solved}/{ik_n} solved, worst hand position error "
      f"{ik_worst * 1000:.2f}mm", flush=True)

demo._force(q_home)
demo._apply(q_home)
demo.world.step(render=False)
# Tracking is reported, not gated: fully-extended wrist poses settle ~90 mrad off, on contact.
worst_track = max(track.values()) if track else 1e9
ok = used >= N and worst < TOL_M and ik_solved >= 1 and ik_worst < TOL_M
print(f">>> armfk tracking (diagnostic, not gated): worst {worst_track * 1000:.2f} mrad/mm; "
      f"{sum(1 for v in track.values() if v < TOL_RAD)}/{len(track)} joints under "
      f"{TOL_RAD * 1000:.0f}", flush=True)
print(f">>> ARMFK GATE: {'PASS' if ok else 'FAIL'}  "
      f"(model {worst * 1000:.2f}mm, IK {ik_worst * 1000:.2f}mm vs {TOL_M * 1000:.0f}mm; "
      f"{used}/{N} FK samples, {ik_solved}/{ik_n} IK)", flush=True)
play_isaac.simulation_app.close()
sys.exit(0 if ok else 1)
