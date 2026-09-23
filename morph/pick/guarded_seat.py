"""Guarded seat, in order: seat by TOUCH, centre laterally, release depth. Every stage is sensed
-- touch by the contact ledger, centring by measured per-finger gaps, release by contact-free
steps -- so no depth or offset is assumed for any diameter. Runs at the end of _pick_descend."""
import math
import os

import numpy as np

from morph.config import HEADLESS, KNOWN


# bench-compat shim: written against play_isaac's globals
class _P:
    HEADLESS = HEADLESS
    KNOWN = KNOWN


P = _P


def probe(demo, qq):
    """Gradient PROBE (1 mrad, one step, then restore): kinematic by design. Under the drive gate
    `_force` is a no-op and a 1 mrad drive step reads ~0 gain, so snap the joints directly; the
    seat's real MOVES still go through the primitives."""
    if demo._drive_on():
        demo.robot.set_joint_positions(np.asarray(demo._with_closure(np.asarray(qq, float)), float))
        demo._drv_prev = None
    else:
        demo._force(qq)
    demo._apply(qq)
    demo.world.step(render=False)


def level_hand(demo, q=None, tag="bench-seat"):
    """Level the palm to vertical with the wrist DOF that has the most pitch authority; returns
    the (possibly updated) q. The baked approach pose arrives pitched ~9-14 deg and a faithful
    parallel grasp then holds the object parallel to that palm. Must run BEFORE the descent --
    levelling after it lifts the pinch and the column floor then blocks the pre-drop."""
    if q is None:
        q = np.asarray(demo.robot.get_joint_positions(), float).copy()
    if os.environ.get("BENCH_HAND_LEVEL", "1") != "1":
        return q

    def _hand_tilt():
        _hyv = demo._hand_frame()[1][:, 1]
        return math.degrees(math.acos(max(-1.0, min(1.0,
            float(-_hyv[2]) / max(1e-9, float(np.linalg.norm(_hyv)))))))
    for _rnd in range(3):
        _t0 = _hand_tilt()
        if _t0 < 2.0:
            break
        _best = None
        for _wn in ("HandBearingJoint_1", "gripper_y_rotation_1", "gripper_x_rotation_1"):
            _wi = demo.idx.get(_wn)
            if _wi is None:
                continue
            qp2 = q.copy(); qp2[_wi] = float(q[_wi]) + 0.02
            probe(demo, qp2)
            _g2 = (_hand_tilt() - _t0) / 0.02
            probe(demo, q)
            if abs(_g2) > 20.0 and (_best is None or abs(_g2) > abs(_best[1])):
                _best = (_wi, _g2)
        if _best is None:
            print(f">>> {tag}: hand-level: no wrist authority at {_t0:.1f}deg -- leaving", flush=True)
            break
        _dw = float(np.clip(-_t0 / _best[1], -0.35, 0.35))
        _nw = max(1, int(0.3 / demo.dt))
        _w0 = float(q[_best[0]])
        for _kk in range(_nw):
            _ff = 0.5 - 0.5 * math.cos(math.pi * (_kk + 1) / _nw)
            q[_best[0]] = _w0 + _ff * _dw
            demo._force(q); demo._apply(q)
            demo.world.step(render=not P.HEADLESS)
    print(f">>> {tag}: hand levelled to {_hand_tilt():.1f}deg off vertical-palm "
          f"(was pitched by the baked pose)", flush=True)
    demo._hand_levelled = True
    return q


def guarded_seat(demo, st):
    """Seat the object by TOUCH, not by model: advance the hand until the palm (or, for an object
    too wide for the throat, the first hand link) makes sustained contact, so the seat holds for
    any diameter -- object between fingers AND palm. The arm is kinematic, so the stop is SENSED
    from the per-step contact ledger and the palm force cap must stay on, or the touch is a ram."""
    def _probe(qq):
        probe(demo, qq)

    ia1 = demo.idx.get("ArmLeftJoint_1")
    if ia1 is None:
        return
    q = np.asarray(demo.robot.get_joint_positions(), float).copy()
    oc0 = np.asarray(st.obj.get_world_poses()[0][0], float)
    # d(pinch)/d(a1), probed kinematically (forced arm -> no settle transient)
    p0 = np.asarray(demo._pinch(), float)
    qp = q.copy(); qp[ia1] = float(q[ia1]) + 1e-3
    _probe(qp)
    dp = (np.asarray(demo._pinch(), float) - p0) / 1e-3
    _probe(q)
    toward = oc0[:2] - p0[:2]
    toward = toward / max(1e-9, float(np.linalg.norm(toward)))
    gain = float(dp[:2] @ toward)                       # m of approach per rad of a1
    if abs(gain) < 1e-3:
        print(">>> bench-seat: a1 has no authority toward the object -- skipped", flush=True)
        return
    step_rad = 0.0005 / gain                            # ~0.5mm of approach per step
    ih1 = demo.idx.get("ColumnLeftBearingJoint_1")
    ih2 = demo.idx.get("ColumnRightBearingJoint_1")
    ith = demo.idx.get("BaseJoint_1")
    z0 = float(p0[2])                                   # hold the approach height: a1 also moves z,
    # ...and hold the LINE: the a1 advance arcs sideways, landing the pads asymmetrically. Probe the
    # turret's lateral gain once and trim it each step, the same pattern as z.
    if not getattr(demo, "_hand_levelled", False):
        q = level_hand(demo, q)
    # once per seat; the next pick levels again
    demo._hand_levelled = False
    p0 = np.asarray(demo._pinch(), float)                # the seat's reference pinch, levelled

    lat_gain = 0.0
    if ith is not None:
        qt = q.copy(); qt[ith] = float(q[ith]) + 1e-3
        _probe(qt)
        dpt = (np.asarray(demo._pinch(), float) - p0) / 1e-3
        _probe(q)
        lat_dir = np.array([-toward[1], toward[0]])     # +90deg left of the approach line
        lat_gain = float(dpt[:2] @ lat_dir)             # m of lateral per rad of th
    th0 = float(q[ith]) if ith is not None else 0.0
    lat_sgn, lat_prev, lat_applied = -1.0, 1e9, False
    # Trim cadence: per-step feedback assumes the command is realised within the step. On drives the
    # hand lags and per-step trims wind up, so trim on a cadence the drives can settle within.
    _trim_every = 12 if demo._drive_on() else 1
    _cmax = P.COLUMN_MAX if hasattr(P, "COLUMN_MAX") else 1.4
    _cmin = float(os.environ.get("COLUMN_FLOOR", "0.20"))   # the descent's rule: below it the carriage is on the chassis cover

    def _cols_step(d):
        """Move BOTH columns by the same d (their difference is the boom pitch), clipped so both
        stay inside [floor, max]; returns the step actually applied. Clipping them independently
        collapses the differential and pitches the boom half a metre."""
        if ih1 is None or ih2 is None:
            return 0.0
        _lo = _cmin - min(float(q[ih1]), float(q[ih2]))
        _hi = _cmax - max(float(q[ih1]), float(q[ih2]))
        d = float(np.clip(d, min(_lo, 0.0), max(_hi, 0.0)))
        q[ih1] = float(q[ih1]) + d
        q[ih2] = float(q[ih2]) + d
        return d

    def _fk_gains():
        """Hand motion per rad of a1 / turret / both columns together, from the FK at the current
        pose: (a1 lateral, turret lateral, a1 z, columns z) in m/rad, or None if unusable. Needed
        because the probe-based gain reads the WRONG SIGN under drives, and because a1 moves z
        wherever the boom is pitched (dz/da1 ~ -0.78), which the trims must cancel."""
        try:
            m = demo._arm_model()
            bp, bR = demo._arm_base_world()
            q8 = demo._arm_q8()
            names = list(m.Q8)
            ia, it = names.index("ArmLeftJoint_1"), names.index("BaseJoint_1")
            ic1, ic2 = names.index("ColumnLeftBearingJoint_1"), names.index("ColumnRightBearingJoint_1")
            h0 = bR @ m.fk(q8)[m.hand_link][0]
            def _d(*ii):
                qq = q8.copy()
                for i in ii:
                    qq[i] += 1e-3
                return (bR @ m.fk(qq)[m.hand_link][0] - h0) / 1e-3
            da, dt, dc = _d(ia), _d(it), _d(ic1, ic2)
            ga, gt = float(da[:2] @ lat_dir), float(dt[:2] @ lat_dir)
            return (ga, gt, float(da[2]), float(dc[2])) if abs(gt) > 1e-3 and abs(dc[2]) > 0.2 else None
        except Exception as _e:
            print(f">>> bench-seat: FK gains unavailable ({_e})", flush=True)
            return None

    _fkg = _fk_gains() if demo._drive_on() and ith is not None else None
    if _fkg is not None:
        print(f">>> bench-seat: FK gains a1 lat {_fkg[0]:+.3f} th lat {_fkg[1]:+.3f} a1 z {_fkg[2]:+.3f} "
              f"cols z {_fkg[3]:+.3f} m/rad (feedforward per rad of a1: turret {-_fkg[0] / _fkg[1]:+.3f} rad, "
              f"columns {-_fkg[2] / _fkg[3]:+.3f} m)", flush=True)

    def _a1_step(s):
        """The seat's ONE fore/aft primitive: a1 by `s` rad plus, on drives, the FK feedforward
        that holds the hand on a straight constant-z line (turret against a1's lateral arc,
        columns against its z coupling). Kinematic mode: a1 alone, the per-step trims hold z."""
        q[ia1] = float(np.clip(q[ia1] + s, 0.0, P.A1_MAX if hasattr(P, "A1_MAX") else 0.5))
        if _fkg is None:
            return
        _th_ff = float(q[ith]) - (_fkg[0] / _fkg[1]) * s
        if abs(_th_ff - th0) < 0.15:
            q[ith] = _th_ff
        _cols_step(-(_fkg[2] / _fkg[3]) * s)

    def _pads(tag):
        """Log pad gaps + du/dv at a seat boundary -- shows WHERE the thumb/pair balance moves."""
        try:
            _ocn = np.asarray(st.obj.get_world_poses()[0][0], float)
            _oRn = demo._obj_R(st.obj)
            _r_o = float(os.environ.get("OBJ_RADIUS", P.KNOWN["object_radius"]))
            _hh_o = float(os.environ.get("OBJ_HALF_H", P.KNOWN["object_half_h"]))
            _gp = {fl: round(float(demo._finger_surface_gap(fl, _ocn, _r_o, _hh_o, _oRn)) * 1000, 1) for fl in "abc"}
            _dd = demo._du_dv(st.obj)
            _pz = float(np.asarray(demo._pinch(), float)[2])
            print(f">>> bench-seat[{tag}]: pad gaps {_gp} mm | du {_dd[0] * 1000:+.1f} dv {_dd[1] * 1000:+.1f} mm | "
                  f"pinch z {_pz:.3f} a1 {float(q[ia1]):.3f} th {float(q[ith]) if ith is not None else float('nan'):+.3f}", flush=True)
        except Exception as _e_p:
            print(f">>> bench-seat[{tag}]: pad gaps unavailable ({_e_p})", flush=True)

    _pads("entry")
    # BACK OFF FIRST, then drop, then advance to touch: the align can land nearly on top of the
    # object, and pre-dropping from a too-near XY presses the fingers onto the near rim.
    _clear = float(os.environ.get("BENCH_SEAT_BACK", "0.012" if os.environ.get("OBJ_GOAL", "0") == "2" else "0.035"))
    for _r in range(240):
        try:
            _ocb = np.asarray(st.obj.get_world_poses()[0][0], float)
            _gb = min(g for g in (demo._wrap_gap(f, _ocb, P.KNOWN["object_radius"],
                                                 demo.obj_half_h, demo._obj_R(st.obj))
                                  for f in "abc") if g is not None)
        except Exception:
            break
        if _gb >= _clear:
            break
        _a1_step(-abs(step_rad))
        demo._force(q); demo._apply(q)
        demo.world.step(render=not P.HEADLESS)
    if _r:
        print(f">>> bench-seat: backed off {_r * 0.5:.1f}mm first (nearest gap now "
              f"{_gb * 1000:.0f}mm >= {_clear * 1000:.0f}mm)", flush=True)
    _pads("after-backoff")

    # Seat at the OBJECT's height band, not the align's inherited height: that height is baked for the
    # tall reference object, and on a shorter one finger c starts above the top face and never lands.
    if os.environ.get("OBJ_GOAL", "0") == "2":
        # Geometric arrival: the band is OBJECT-relative (object centre z + SEAT_BAND_DZ), not
        # arrival-relative, which can target a z below the floor.
        z0 = float(np.asarray(st.obj.get_world_poses()[0][0], float)[2]) \
            + float(os.environ.get("SEAT_BAND_DZ", "0.0"))
    else:
        # Baked arrival: 35 mm below the height the align inherited.
        z0 = z0 - 0.035
    _pz_live = float(np.asarray(demo._pinch(), float)[2])   # after the back-off, not the entry
    if ih1 is not None and ih2 is not None and abs(z0 - _pz_live) > 1e-3:
        for _ in range(240):
            _pz = float(np.asarray(demo._pinch(), float)[2])
            _dz0 = float(np.clip(z0 - _pz, -6e-4, 6e-4))
            if abs(z0 - _pz) < 1e-3:
                break
            if _cols_step(_dz0) == 0.0:
                print(f">>> bench-seat: pre-drop stopped at the column floor (pinch z {_pz:.3f}, target {z0:.3f})", flush=True)
                break
            demo._force(q); demo._apply(q)
            demo.world.step(render=not P.HEADLESS)
        _pz_end = float(np.asarray(demo._pinch(), float)[2])
        print(f">>> bench-seat: pre-drop end pinch z {_pz_end:.3f}, band {z0:.3f}, "
              f"residual {(_pz_end - z0) * 1000:+.1f}mm"
              + (" -- BAND NOT REACHED" if abs(_pz_end - z0) > 0.005 else ""),
              flush=True)
    lat0 = float(np.cross(np.append(toward, 0.0), np.append(p0[:2] - oc0[:2], 0.0))[2])
    touch_n, moved, k = 0, 0.0, 0

    def _advance_to_touch(_tag=""):
        """ADVANCE: a1 forward until sustained touch (touch_n >= 3) or the object moves."""
        nonlocal touch_n, moved, k, _fkg, lat_sgn, lat_prev, lat_applied
        demo._ctc = {}                                       # who touches the object during the advance
        # <= 200mm of travel, far past any seat
        for k in range(400):
            demo._hit_step = False
            _trim_now = (k % _trim_every == 0)
            _a1_step(step_rad)
            if _fkg is not None and k % (12 * 8) == 0 and k:
                _fkg = _fk_gains() or _fkg                      # refresh the gains as the pose changes
            # CONST-Z advance, by measurement: trim both columns toward the entry height each step
            # (<= 0.6mm/step) so the boom extension cannot walk the hand off the object band.
            _pnow = np.asarray(demo._pinch(), float)
            if ih1 is not None and ih2 is not None and _trim_now:
                _cols_step(float(np.clip(z0 - float(_pnow[2]), -6e-4, 6e-4)))
            if ith is not None and abs(lat_gain) > 1e-3 and _trim_now:
                _lerr = float(np.cross(np.append(toward, 0.0), np.append(_pnow[:2] - oc0[:2], 0.0))[2])
                # SELF-CORRECTING sign: the probed gain sign does not survive contact noise. If the
                # error grew after the last correction, flip it; converges within two steps.
                if abs(_lerr) > abs(lat_prev) + 1e-6 and lat_applied:
                    lat_sgn = -lat_sgn
                lat_prev, lat_applied = abs(_lerr), False
                if abs(_lerr) > 0.002:
                    if demo._drive_on():
                        # Drives: sign and scale from the FK gain, 0.3 per 12-step trim, clipped.
                        _gt = _fkg[1] if _fkg is not None else lat_gain
                        _dth = float(np.clip(-0.3 * _lerr / _gt, -6e-4 / abs(_gt), 6e-4 / abs(_gt)))
                    else:
                        _dth = lat_sgn * float(np.clip(abs(_lerr) / abs(lat_gain),
                                                       0.0, 6e-4 / abs(lat_gain)))
                    _th_new = float(q[ith]) + _dth
                    # total authority ~30mm -- a runaway is bounded
                    if abs(_th_new - th0) < 0.05:
                        q[ith] = _th_new
                        lat_applied = True
            demo._force(q); demo._apply(q)
            demo.world.step(render=not P.HEADLESS)
            touch_n = touch_n + 1 if demo._hit_step else 0
            moved = float(np.linalg.norm(
                np.asarray(st.obj.get_world_poses()[0][0], float)[:2] - oc0[:2])) * 1000.0
            if touch_n >= 3 or moved > 5.0:
                try:
                    _dd_t = demo._du_dv(st.obj)
                    _ctc_nf = {k: v for k, v in getattr(demo, '_ctc', {}).items() if "floor" not in k}
                    print(f">>> bench-seat: touch by "
                          f"{dict(sorted(_ctc_nf.items(), key=lambda kv: -kv[1])[:4])} "
                          f"at du {_dd_t[0] * 1000:+.1f} dv {_dd_t[1] * 1000:+.1f} mm{_tag}" if _dd_t is not None else
                          f">>> bench-seat: touch by {dict(sorted(_ctc_nf.items(), key=lambda kv: -kv[1])[:4])}{_tag}",
                          flush=True)
                except Exception as _e_t:
                    print(f">>> bench-seat: touch attribution unavailable ({_e_t})", flush=True)
                break
        return touch_n

    touch_n = _advance_to_touch()
    _moved1 = moved                      # snapshot: the centring loop below reuses `moved`
    # Centre the object between thumb and the b/c pair along the OPPOSITION axis, driven by measured
    # surface gaps and guarded by object displacement and a no-plow floor on the closest gap.
    if ith is not None and touch_n >= 3:
        # ORDER: release the depth FIRST, then centre at the depth the close will see. Release by sensing
        # CONTACT, never a gap threshold -- gaps are in the OPEN-pose frame and are satisfied at once.
        def _release_depth(_tag=""):
            """DEPTH RELEASE: retreat until contact-free, then a 3mm decompression margin."""
            _free, _k3 = 0, 0
            for _k3 in range(60):
                demo._hit_step = False
                _a1_step(-step_rad)
                demo._force(q); demo._apply(q)
                demo.world.step(render=not P.HEADLESS)
                _free = 0 if demo._hit_step else _free + 1
                if _free >= 3:
                    break
            # 3mm decompression margin
            for _ in range(6):
                _a1_step(-step_rad)
                demo._force(q); demo._apply(q)
                demo.world.step(render=not P.HEADLESS)
            _pads("after-release")
            print(f">>> bench-seat: depth release, retreated {(_k3 + 7) * 0.5:.1f}mm "
                  f"(contact-free + 3mm margin){_tag}", flush=True)

        _release_depth()
        _r_o = float(os.environ.get("OBJ_RADIUS", P.KNOWN["object_radius"]))
        _hh_o = float(os.environ.get("OBJ_HALF_H", P.KNOWN["object_half_h"]))

        def _ggaps():
            _ocn = np.asarray(st.obj.get_world_poses()[0][0], float)
            _oRn = demo._obj_R(st.obj)
            return {fl: float(demo._finger_surface_gap(fl, _ocn, _r_o, _hh_o, _oRn))
                    for fl in "abc"}

        def _cerr(g):
            return (g["a"] - 0.5 * (g["b"] + g["c"])) / 2.0

        _g0 = _ggaps(); _e0 = _cerr(_g0); _th0 = float(q[ith])
        # The gradient PROBE snaps under the drive gate: probing through the drives reads a wrong
        # gradient and the Newton loop diverges, pressing the pads into the object.
        q[ith] = _th0 + 5e-3
        _probe(q)
        _de = (_cerr(_ggaps()) - _e0) / 5e-3
        q[ith] = _th0
        _probe(q)
        _e, k2 = _e0, 0
        _drv = demo._drive_on()
        if abs(_de) > 1e-3:
            # drives need more steps at the 12-step cadence
            for k2 in range(240 if _drv else 80):
                _g = _ggaps(); _e = _cerr(_g)
                _ocn = np.asarray(st.obj.get_world_poses()[0][0], float)
                moved = float(np.linalg.norm(_ocn[:2] - oc0[:2])) * 1000.0
                if abs(_e) * 1000.0 <= 2.0 or moved > 5.0 or min(_g.values()) * 1000.0 < -3.0:
                    break
                # Clamp +-0.12 rad (~54mm at this lever); +-0.06 saturates on a large object.

                # drives: 0.3-gain Newton at a 12-step cadence
                if not _drv or k2 % 12 == 0:
                    _gn = 0.3 if _drv else 1.0
                    q[ith] = float(np.clip(_th0 + np.clip(float(q[ith]) - _th0 - _gn * _e / _de,
                                                          -0.12, 0.12), _th0 - 0.12, _th0 + 0.12))
                demo._force(q); demo._apply(q)
                demo.world.step(render=not P.HEADLESS)
        _gf = _ggaps()
        _pads("after-centring")
        print(f">>> bench-seat: lateral centring gap-err {_e0 * 1000:+.1f} -> {_e * 1000:+.1f}mm "
              f"in {k2} steps, gaps(mm) " +
              str({fl: round(v * 1000, 1) for fl, v in _gf.items()}), flush=True)
        # SECOND PASS: centring moves the turret, not a1, so depth lost to an off-centre knuckle is never
        # regained. Skipped when the first pass ended because the object MOVED, not because it touched.
        if _moved1 <= 5.0:
            touch_n = _advance_to_touch(" [pass 2]")
            _pads("after-pass2")
            _release_depth(" [pass 2]")
        # RELATION SERVO: put the pads at the object's RADIUS in MOUTH DEPTH, so the curl travels around
        # the object rather than into it. Contact is NOT a stop here: the knuckles sit inside the offset.
        if os.environ.get("SEAT_RELATION", "1") == "1":
            # size-dynamic: one radius in front of the centre, never a hardcoded distance
            _rel_md = float(os.environ.get("SEAT_MOUTH_DEPTH", "0")) or _r_o
            _rel_dv = float(os.environ.get("SEAT_REL_DV", "0.0"))
            _rel_tol = float(os.environ.get("SEAT_REL_TOL", "0.004"))

            def _md():
                """Mean pad depth in front of the object centre (m), or None."""
                try:
                    _d, _ = demo._mouth_depth(st.obj)
                    return None if _d is None else float(_d)
                except Exception:
                    return None
            # baseline at the commanded pose, as the probe measures
            _probe(q)
            _md0 = _md()
            _dd0 = demo._du_dv(st.obj)
            _g_u = _g_v = None
            if _md0 is not None:
                _q_keep = q.copy()
                # ~2 mm of pinch travel
                _a1_step(4.0 * step_rad)
                _probe(q)
                _mdp = _md()
                if _mdp is not None:
                    _g_u = (_mdp - _md0) / (4.0 * step_rad)                      # d(depth)/d(a1)
                q[:] = _q_keep
                _probe(q)
                if ith is not None and _dd0 is not None:
                    q[ith] = float(_q_keep[ith]) + 5e-3
                    _probe(q)
                    _ddp = demo._du_dv(st.obj)
                    if _ddp is not None:
                        _g_v = (float(_ddp[1]) - float(_dd0[1])) / 5e-3          # d(dv)/d(th)
                    q[:] = _q_keep
                    _probe(q)
                print(f">>> bench-seat: relation gains depth/a1 {_g_u if _g_u is not None else float('nan'):+.3f} "
                      f"dv/th {_g_v if _g_v is not None else float('nan'):+.3f} m/rad (start depth "
                      f"{_md0 * 1000:+.1f} mm, target {_rel_md * 1000:.0f}; dv "
                      f"{_dd0[1] * 1000 if _dd0 is not None else float('nan'):+.1f} mm)", flush=True)
            if _g_u is not None and abs(_g_u) < 0.05:
                _g_u = None
            if _g_v is not None and abs(_g_v) < 0.02:
                _g_v = None
            _su = _sv = 0.0
            _why, _kr = "step cap", 0
            _dd = _dd0
            _oc_r0 = np.asarray(st.obj.get_world_poses()[0][0], float)
            for _kr in range(240):
                _mv = float(np.linalg.norm(np.asarray(st.obj.get_world_poses()[0][0], float)[:2] - _oc_r0[:2]))
                if _mv > 0.003:
                    _why = f"object moved {_mv * 1000:.1f} mm"
                    break
                if _kr % 12 == 0:
                    _mdn = _md()
                    _dd = demo._du_dv(st.obj)
                    if _mdn is None:
                        _why = "no mouth depth"
                        break
                    # depth error signs with the boom: pads too far in FRONT must advance
                    _eu = _rel_md - _mdn
                    _ev = (_rel_dv - float(_dd[1])) if _dd is not None else 0.0
                    if abs(_eu) <= _rel_tol and abs(_ev) <= _rel_tol:
                        _why = "in tolerance"
                        break
                    # 0.5 gain per cadence, spread over the 12 steps and clipped
                    _su = (float(np.clip(0.5 * _eu / _g_u, -4.0 * step_rad, 4.0 * step_rad)) / 12.0
                           if _g_u is not None and abs(_eu) > _rel_tol else 0.0)
                    _sv = (float(np.clip(0.5 * _ev / _g_v, -0.010, 0.010)) / 12.0
                           if _g_v is not None and ith is not None and abs(_ev) > _rel_tol else 0.0)
                    if _su == 0.0 and _sv == 0.0:
                        _why = "no authority"
                        break
                if _fkg is None and _kr % _trim_every == 0:
                    # `_a1_step` cancels a1's z coupling only through the FK feedforward; with no
                    # gains it is the boom alone (dz/da1 ~ -0.78), so trim at the seat's cadence.
                    _cols_step(float(np.clip(z0 - float(np.asarray(demo._pinch(), float)[2]),
                                             -6e-4, 6e-4)))
                if _su:
                    _a1_step(_su)
                if _sv:
                    # Bound the servo's OWN contribution, not the accumulated value: `_a1_step`
                    # also writes the turret, and clipping the sum would silently rewind that.
                    _th_new = float(q[ith]) + _sv
                    if abs(_th_new - _th0) < 0.12:
                        q[ith] = _th_new
                demo._force(q); demo._apply(q)
                demo.world.step(render=not P.HEADLESS)
            _md_fresh, _dd_fresh = _md(), demo._du_dv(st.obj)
            _stale = _md_fresh is None
            _mdn = _md_fresh if _md_fresh is not None else _mdn
            _dd = _dd_fresh if _dd_fresh is not None else _dd
            _pads("after-relation")
            print(f">>> bench-seat: relation servo -> mouth depth "
                  f"{_mdn * 1000 if _mdn is not None else float('nan'):+.1f} mm (target "
                  f"{_rel_md * 1000:.0f}, tol {_rel_tol * 1000:.0f}) | dv "
                  f"{float(_dd[1]) * 1000 if _dd is not None else float('nan'):+.1f} mm "
                  f"in {_kr} steps [{_why}]" + (" (stale reading)" if _stale else ""), flush=True)
    _pe = np.asarray(demo._pinch(), float)
    _lat = float(np.cross(np.append(toward, 0.0), np.append(_pe[:2] - oc0[:2], 0.0))[2])
    _pads("after-advance")
    print(f">>> bench-seat: advanced {k * 0.5:.1f}mm -> "
          f"{'SUSTAINED TOUCH' if touch_n >= 3 else 'OBJECT MOVED' if moved > 5.0 else 'TRAVEL CAP'}"
          f" (object displaced {moved:.1f}mm | z drift {(float(_pe[2]) - z0) * 1000:+.1f}mm"
          f" | lateral drift {(_lat - lat0) * 1000:+.1f}mm)", flush=True)
    st.grip = q.copy()

