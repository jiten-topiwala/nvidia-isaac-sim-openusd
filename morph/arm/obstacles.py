"""The obstacle policy every planned motion shares: the deck pad, the padded set, and blame."""
import numpy as np

from morph.arm.collision import MARGIN, CHASSIS_PAD_LINKS, CHASSIS_PAD_M, CHASSIS_PAD_PRIM


def chassis_pad(obs, paths):
    """`obs` with the DECK box held to `CHASSIS_PAD_M` for `CHASSIS_PAD_LINKS` instead of MARGIN --
    a FLOOR for the one pair with no localisation error. Applied by `planned_obstacles` for every
    planned motion and its checkpoint, never in `_arm_obstacles`. RAISES if the prim is absent."""
    out, hit = [], 0
    for ob, path in zip(obs, paths):
        # EXACT, never a prefix: `base` carries the deck, and the battery and the wheels hang under
        # it with their own colliders and their own boxes.
        if path == CHASSIS_PAD_PRIM:
            out.append((ob[0], ob[1], CHASSIS_PAD_M, CHASSIS_PAD_LINKS))
            hit += 1
        else:
            out.append(ob)
    if hit != 1:
        under = sorted(q for q in paths if q.startswith(CHASSIS_PAD_PRIM.rsplit("/", 1)[0] + "/"))
        raise KeyError(f"chassis_pad: {CHASSIS_PAD_PRIM} matched {hit} of {len(paths)} obstacle "
                       f"prims, expected exactly 1 -- the pad would be a silent no-op. "
                       f"Boxes under its parent: {under[:12]}")
    return out


def planned_obstacles(demo, grasp_names=()):
    """The producer plus the deck floor, for the planner AND its checkpoint: a checkpoint
    stricter than its planner is a replan loop, not a guard."""
    paths = []
    obs = demo._arm_obstacles(grasp_names, diagnostic_paths=paths)
    return chassis_pad(obs, paths), paths


def blame(m, q, obs, paths, held, base, fingers):
    """(link, prim path) for the first link inside `MARGIN` at `q`, or None if the pose is clear.

    Judged with the same predicate, margin and hand that refuse the pose, so it cannot accuse a
    pair the checker accepts. Costs a second pass; call it only on a refusal.
    """
    hit = m.first_hit(q, obs, MARGIN, held, base, fingers=fingers)
    if hit is None:
        return None
    who = next((p for ob, p in zip(obs, paths)
                if m.first_hit(q, [ob], MARGIN, held, base, links=(hit,), fingers=fingers)
                is not None), "?")
    return hit, who


def arm_blame(demo, q8, obs, paths, held, what, base, fingers=None):
    """Print every link/obstacle pair that makes `q8` invalid, naming the obstacle's PRIM.

    `paths` is `planned_obstacles`' second return, index-aligned with `obs`. `fingers` is NOT
    optional in practice: without it this walks the swept hand union while the check that refused
    used the nine pads, so the report describes a different hand."""
    # `zip` would truncate to the shorter of the two and misname every prim past the gap.
    if len(paths) != len(obs):
        raise ValueError(f"arm_blame: {len(paths)} prim paths for {len(obs)} obstacles")
    m = demo._arm_model()
    dh, a1 = q8[2] - q8[1], q8[3]
    if not m.in_lut_domain(dh, a1):
        print(f">>>   {what} outside the closure LUT: dh {dh:+.4f} a1 {a1:.4f} "
              f"(domain dh [{m.lut['dh'][0]}, {m.lut['dh'][-1]}] "
              f"a1 [{m.lut['a1'][0]}, {m.lut['a1'][-1]}])", flush=True)
        return
    # In-domain but past a stop: `isValid`'s second half. No early return -- the scan still counts.
    for n, v, plo, phi in m.passive_violations(dh, a1):
        print(f">>>   {what} demands {n} = {v:+.4f}, outside [{plo:+.3f}, {phi:+.3f}] by "
              f"{max(plo - v, v - phi) * 1000:.0f}mm -- the closure cannot hold this pose",
              flush=True)
    # `first_hit` per (link, obstacle) -- the SAME predicate, frame rule and exemption set the refusal
    # applied. A hand-written copy here reported the broad phase alone and accused cleared links.
    hits = 0
    # The CHECKER's link set, not the model's: `_links` appends the carried box as `__held__`, which
    # `m.link_aabb` does not contain, so iterating the model's keys hides the object in the hand.
    for link in m.link_aabbs(q8, held, base, fingers=fingers):
        for ob, path in zip(obs, paths):
            if m.first_hit(q8, [ob], MARGIN, held, base, links=(link,),
                           fingers=fingers) is None:
                continue
            hits += 1
            if hits <= 6:
                ab, tg = ob[0], ob[1]
                # `clearance` returns -1.0 when the pair ALREADY overlaps at margin 0, printing as -1000mm.
                # Real penetration and a margin shortfall want opposite fixes.
                clr = m.clearance(q8, [ob], held, base, fingers=fingers)
                gap = ("OVERLAPPING" if clr < 0.0 else
                       f"clearance {clr * 1000:.0f}mm of margin {MARGIN * 1000:.0f}mm")
                print(f">>>   {what} hit: {link} vs {tg} obstacle {path} "
                      f"[{np.round(ab[0], 3).tolist()}..{np.round(ab[1], 3).tolist()}] "
                      f"({gap})", flush=True)
    print(f">>>   {what}: {hits} link/obstacle overlap(s) at margin {MARGIN * 1000:.0f}mm",
          flush=True)
