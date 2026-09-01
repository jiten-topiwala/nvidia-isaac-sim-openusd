"""Replay of the baked reach trajectory that brings the hand to the object.

Part of the pick cycle — see `morph/pick/__init__.py` for the stage order and the shared
state object. Imports Isaac APIs at module level; only importable after SimulationApp.
"""
import math
import os

import numpy as np

from pxr import Usd, UsdPhysics

from morph.config import PERFECT, HEADLESS, KNOWN, COLUMN_MAX, A1_MAX, MOUTH_VEC_HAND


class ReachStage:
    """One stage of `pick`. Mixed into `PickMixin`; all `self.*` belong to `Demo`."""

    def _pick_reach(self, st):
        obj_idx, status = st.obj_idx, st.status
        O, dock, qg, obj, park = st.O, st.dock, st.qg, st.obj, st.park
        close_anim, replay = st.close_anim, st.replay
        cup_fingers, pin_bystanders = st.cup_fingers, st.pin_bystanders
        # Open above the object, then descend open. The hand arrives cupped and the replay holds the
        # fingers wherever they entered, so without this it descends cupped and springs open at the
        # settle, which reads as the fingers entering the object.
        if close_anim and os.environ.get("OPEN_BEFORE_DESCENT", "0" if PERFECT else "1") == "1":
            _q_ob = np.asarray(self.robot.get_joint_positions(), float).copy()
            _tgt_ob = {n: v for n, v in KNOWN["fingers_open"].items() if n in self.idx}
            _far = max((abs(float(_q_ob[self.idx[n]]) - float(v)) for n, v in _tgt_ob.items()),
                       default=0.0)
            if _far > 0.05:                       # skip when already open (e.g. a retry)
                self._animate_fingers(_q_ob, _tgt_ob, 0.6)
                print(f">>> open-before-descent: fingers -> the open pose over "
                      f"{os.environ.get('OPEN_T', '0.6')}s, {_far:.2f} rad travelled "
                      f"(one open pose everywhere; OPEN_BEFORE_DESCENT=0 restores the late open)",
                      flush=True)
        _oc0 = None
        if replay:
            # Bystander protection is the corridor gate (no sweep within 0.35m of another object).
            # Toggling ~9 objects' colliders per pick instead invalidates the articulation view in
            # headed mode.
            tr = self.traj["reach"]
            n_fr = len(tr["frames"])
            if PERFECT and self.clut is not None:
                # Base-shift alignment. Every arm-servo align mutates the arm shape, which
                # invalidates the baked descent geometry and every depth calibration downstream.
                t_total = float(os.environ.get("REACH_T", "2.5"))
                self._play_traj(tr, t_total * (n_fr - 30) / n_fr,
                                frame_range=(0, n_fr - 30))
                if self._finite("reach-hover"):
                    if status:
                        status(f"aligning over object {obj_idx}")
                    q_hover = np.asarray(self.robot.get_joint_positions(), float).copy()
                    self.start_jog()                         # only to get the FK radial (slide dir)
                    qj0 = self.jog["q"].copy()
                    f0_ = self._grip_fk(*qj0)
                    qq_ = qj0.copy()
                    qq_[2] += 1e-3
                    u_r = (self._grip_fk(*qq_) - f0_)[:2]
                    u_r = u_r / max(float(np.linalg.norm(u_r)), 1e-9)
                    self.jog = None
                    back = 0.0   # no horizontal slide: the
                    # Align to the BAKED geometry, not to the pinch centroid. `_grasp_known` pairs a
                    # base pose with an object pose from a real successful grasp, so the object must
                    # sit at `self.local_obj` in the base frame -- the same relation nav docks to.
                    _no_push = PERFECT and os.environ.get("BASE_ALIGN", "0") != "1"
                    # ledger: base_pose() reads are stale for a beat after any set_base, and re-
                    # reading silently undid each shift
                    tbx, tby, tbyaw = self.base_ledger()
                    Ol = np.asarray(obj.get_world_poses()[0][0], float)[:2]
                    c_a, s_a = math.cos(tbyaw), math.sin(tbyaw)
                    off_a = np.array([c_a * self.local_obj[0] - s_a * self.local_obj[1],
                                      s_a * self.local_obj[0] + c_a * self.local_obj[1]])
                    tgt = Ol - off_a                         # exact baked base pose for THIS object
                    if _no_push:
                        tgt = np.array([tbx, tby], float)     # stay put: arm does the rest
                        print(">>> base-align: NO CHASSIS PUSH (friction) -- base held, arm only",
                              flush=True)
                    # The mouth insert belongs here too. This stage targets `self.local_obj`, the
                    # baked base-frame offset, rather than `_rec_target()`, so `_mouth_offset` never
                    # reached it and base-align simply re-seated the base on the baked relation.
                    _ins_b = float(os.environ.get("MOUTH_INSERT", "0.0"))
                    if abs(_ins_b) > 1e-9:
                        try:
                            _, _hRb = self._hand_frame()
                            _mwb = (_hRb @ MOUTH_VEC_HAND)[:2]
                            _mwb = _mwb / max(1e-9, float(np.linalg.norm(_mwb)))
                            tgt = tgt - _ins_b * _mwb        # hand goes -mouth => object sits deeper
                            print(f">>> base-align: MOUTH INSERT {_ins_b * 1000:.0f}mm along world "
                                  f"{np.round(-_mwb, 3).tolist()}", flush=True)
                        except Exception as _e_bi:
                            print(f">>> base-align: mouth insert skipped ({_e_bi})", flush=True)
                    bx0, by0 = tbx, tby
                    n_a = int(0.6 / self.dt)
                    for ka in range(n_a):                    # smooth, so nothing is jolted
                        fa = 0.5 - 0.5 * math.cos(math.pi * (ka + 1) / n_a)
                        self.set_base(bx0 + fa * (tgt[0] - bx0), by0 + fa * (tgt[1] - by0), tbyaw)
                        self._force(q_hover)
                        self._apply(q_hover)
                        self.world.step(render=not HEADLESS)
                    tbx, tby, tbyaw = self.base_ledger()
                    print(f">>> base-align: moved {math.hypot(tgt[0] - bx0, tgt[1] - by0) * 1000:.1f}mm "
                          f"to the baked grasp relation (pinch->object "
                          f"{np.linalg.norm((np.asarray(obj.get_world_poses()[0][0], float) - self._pinch())[:2]) * 1000:.0f}mm, "
                          f"expected ~95mm)", flush=True)
                    abx, aby, _ = self.base_ledger()
                    dock = np.array([abx, aby])              # the align MOVED the base on purpose —
                    # Pure baked columns-only drop, with the base holding the cage on its vertical
                    # line: the boom swings ~40mm laterally during column motion, enough to eat the
                    # cage margin.
                    fr2 = tr["frames"][n_fr - 30:]
                    ii2 = tr["idx"]
                    qd = q_hover.copy()
                    h1i = self.idx["ColumnLeftBearingJoint_1"]
                    h2i = self.idx["ColumnRightBearingJoint_1"]
                    # Full baked depth, no early stop. At the baked end config the palm clears the
                    # object by 34mm with no vertices inside, and the contact ledger shows the floor
                    # only.
                    _oc0 = np.asarray(obj.get_world_poses()[0][0], float)
                    h1_0 = float(q_hover[h1i])
                    p0 = self._pinch()[:2].copy()
                    oc0 = np.asarray(obj.get_world_poses()[0][0], float)
                    hit = False
                    n_d = int(3.2 / self.dt)
                    # Total metres, ramped in over the descent below -- NOT pre-divided by the step
                    # count.
                    trim = float(os.environ.get("DROP_TRIM", "0.0"))
                    # Say which mode is actually running: an unconditional message here announces
                    # the baked descent even when the computed branch below takes over, which reads
                    # as the computed descent being inert.
                    _dmode = os.environ.get("DESCENT_MODE", "baked")
                    print(f">>> descent: {'COMPUTED (solve + ramp)' if _dmode == 'computed' else 'FULL baked depth'}"
                          f", object top {_oc0[2] + self.obj_half_h:.3f}, "
                          f"trim {trim * 1000:.0f}mm", flush=True)
                    if PERFECT and os.environ.get("PRE_OPEN_DESCENT", "0") == "1":
                        # Open before coming down. In perfect mode the object is a real collider and
                        # the baked descent brings the hand down THROUGH its space, scraping it for
                        # hundreds of steps and shoving it sideways and into the floor -- after
                        # which the far fingers finish centimetres away and the close grabs air.
                        qd = self._pre_close_open(qd, obj)
                        self._finger_cmd = qd[self._f_idx].copy()
                        qd_prev = qd.copy()
                    if PERFECT and os.environ.get("RELATION_ALIGN", "1") == "1":
                        self._align_recorded_relation(
                            obj, standoff=0.0)
                        # A base back-off buys real clearance for the vertical drop -- it takes the
                        # descent's object contacts to zero, where offsetting the recorded gripper-
                        # frame relation does not.
                        bo = float(os.environ.get("DESCENT_BACKOFF",
                                                  "0.0" if PERFECT else "0.18"))
                        # The arm-retraction alternative is off by default: it does NOT buy the
                        # clearance, because the descent is a baked joint replay whose first frame
                        # writes joint positions and wipes the retraction, while a base offset
                        # survives (the replay does not control the base).
                        if os.environ.get("DESCENT_BACKOFF_ARM", "0") == "1":
                            _ia1 = self.idx.get("ArmLeftJoint_1")
                            if _ia1 is not None:
                                _qr = np.asarray(self.robot.get_joint_positions(), float).copy()
                                _a1_0 = float(_qr[_ia1])
                                _a1_t = max(0.0, _a1_0 - 0.18)
                                _nb = int(0.5 / self.dt)
                                for _ib in range(_nb):
                                    _fb = 0.5 - 0.5 * math.cos(math.pi * (_ib + 1) / _nb)
                                    _qr[_ia1] = _a1_0 + _fb * (_a1_t - _a1_0)
                                    self._force(_qr)
                                    self._apply(_qr)
                                    self.world.step(render=not HEADLESS)
                                print(f">>> descent back-off: ARM a1 {_a1_0:.3f} -> {_a1_t:.3f} "
                                      f"(chassis stationary)", flush=True)
                                bo = 0.0
                        if bo > 0:
                            ob4 = np.asarray(obj.get_world_poses()[0][0], float)
                            bx4, by4, byaw4 = self.base_ledger()
                            v = np.array([bx4, by4]) - ob4[:2]
                            v = v / max(1e-6, float(np.linalg.norm(v)))
                            q4 = np.asarray(self.robot.get_joint_positions(), float).copy()
                            n4 = int(0.5 / self.dt)
                            for i4 in range(n4):
                                f4 = 0.5 - 0.5 * math.cos(math.pi * (i4 + 1) / n4)
                                self.set_base(float(bx4 + f4 * bo * v[0]),
                                              float(by4 + f4 * bo * v[1]), byaw4)
                                self._force(q4)
                                self._apply(q4)
                                self.world.step(render=not HEADLESS)
                            print(f">>> descent back-off {bo * 1000:.0f}mm along "
                                  f"{np.round(v, 3).tolist()}", flush=True)
                        qd = np.asarray(self.robot.get_joint_positions(), float).copy()
                    qd_prev = qd.copy()
                    # Descend with the jaw actually wide. The baked `fingers_open` is asymmetric,
                    # and the narrower side descends with only a few millimetres of clearance on a
                    # large object -- less than the align error.
                    wide_i = []
                    # Size-aware: the extra widening exists for a large object, where the baked
                    # opening leaves only a few millimetres per side.
                    _need_wide = float(KNOWN["object_radius"]) > 0.06
                    if not _need_wide and os.environ.get("DESCENT_WIDE", "1") == "1":
                        print(f">>> descent: jaw stays at the baked open (a -1.04 / b,c -0.493) -- "
                              f"object radius {float(KNOWN['object_radius']) * 1000:.0f}mm needs no "
                              f"extra widening", flush=True)
                    if _need_wide and os.environ.get("DESCENT_WIDE", "1") == "1":
                        wide_j1 = float(os.environ.get("J1_WIDEST", "-1.04"))
                        wide_i = [self.idx[f"finger_{c_}_joint_1_1"] for c_ in "abc"
                                  if f"finger_{c_}_joint_1_1" in self.idx]
                        print(f">>> descent: jaw opened to j1 {wide_j1:+.2f} on all three "
                              f"(baked open was a -1.04 / b,c -0.493 = a 20mm-narrower jaw)",
                              flush=True)
                    # how far to hold a1 back for the whole drop (0 = baked path, chassis back-off)
                    _a1_off = 0.0
                    i_a1r = self.idx.get("ArmLeftJoint_1")
                    if _a1_off > 0.0 and i_a1r is not None:
                        print(f">>> descent: holding a1 back {_a1_off:.3f} for the whole drop "
                              f"(arm clearance, chassis stationary)", flush=True)
                    # Computed descent: solve the height for THIS object rather than replaying a
                    # recorded drop.
                    if PERFECT and os.environ.get("PIN_CHASSIS", "1") == "1":
                        self.pin_chassis(True, "reach")
                    if os.environ.get("DESCENT_MODE", "baked") == "computed":
                        _oc_d = np.asarray(obj.get_world_poses()[0][0], float)
                        _z_tgt = float(_oc_d[2]) + 0.0
                        _pz_d = float(self._pinch()[2])
                        _need = _pz_d - _z_tgt                # >0 -> must come DOWN
                        _qd0 = np.asarray(self.robot.get_joint_positions(), float).copy()
                        _h1_0, _h2_0 = float(_qd0[h1i]), float(_qd0[h2i])
                        _a1_d = self.idx.get("ArmLeftJoint_1")
                        _a1_0d = float(_qd0[_a1_d]) if _a1_d is not None else 0.0
                        # Fold a1 during the descent so the extension stage has real travel.
                        # Descending at the recorded a1 already parks the hand OVER the object, so
                        # extending from there overshoots.
                        _a1_fold = 0.24
                        # Floor the fold by chassis clearance at the SWUNG th: a fold chosen at th =
                        # 0 pulls the gripper back over the chassis footprint at the recorded th,
                        # and the descent guard then stops the drop with the columns high.
                        _th_end = 0.0
                        try:
                            _tj0 = tr["names"]
                            if "BaseJoint_1" in _tj0:
                                _th_end = float(tr["frames"][-1][_tj0.index("BaseJoint_1")])
                        except Exception:
                            pass
                        # dh_end from the SAME source the solve uses -- _h1_t/_h2_t are defined by
                        # the solve BELOW this line (third use-before-assign of this campaign; this
                        # one crashed the cycle instead of being swallowed).
                        _dh_end = float(os.environ.get("LOWEST_DH_TGT", "0.13"))
                        # SAME margin as the mid-move path guard, or the floor certifies a start
                        # pose the guard then condemns -- measured: floored at 0.272 (margin 0.02),
                        # STOPPED at 0.283 (margin 0.05).
                        _m_path = 0.05
                        _a1_cl = self._a1_clearing_chassis(_dh_end, margin=_m_path, th=_th_end)
                        if _a1_cl is not None and _a1_cl + 0.015 > _a1_fold:
                            print(f">>> descent(computed): fold floored at a1 "
                                  f"{_a1_cl + 0.015:.3f} (asked {_a1_fold:.3f}) -- below it the "
                                  f"gripper sits over the chassis at th {_th_end:+.2f}", flush=True)
                            _a1_fold = _a1_cl + 0.015
                        _a1_tg = (max(0.0, min(_a1_fold, _a1_0d) - _a1_off)
                                  if _a1_d is not None else 0.0)
                        # Solve the target pose first, then ramp to it. Driving the column MEAN at
                        # the height error with the differential left alone brings the boom down
                        # FLAT until it bottoms out, and a lower column floor makes that worse.
                        _off_zd = _pz_d - self._fk_z(_h1_0, _h2_0, _a1_0d)
                        # Carriage against the chassis covers: the column carriage tracks h1
                        # directly, so a low h1 puts it on the cover surface with the arm slightly
                        # inside the body.
                        _sol_d = self._solve_lowest(
                            _z_tgt, _off_zd,
                            # solve with the nominal 0.50: the extension stops AT HEIGHT on its own
                            # (a1 ~0.45), and the optimistic assumption is what keeps the carriage
                            # at the carriage higher, i.e.
                            a1_max=A1_MAX,
                            dh_want=float(os.environ.get("LOWEST_DH_TGT", "0.15")) or None,
                            # No h1 floor in the SOLVE. Forcing one makes it buy the pinch height
                            # with more tilt, the descent guard clips that tilt, the extension never
                            # runs and the swept boom knocks the object over.
                            h1_min=0.0)
                        if _sol_d is not None:
                            _h1_t, _h2_t, _a1_solved, _reach_d = _sol_d
                            # Fold, do not hold: an overwrite here ("a1 goes forward after the
                            # tilt") silently cancels the fold. The fold IS the setup for the
                            # forward stage -- descend folded, extend after.
                            _a1_tg = min(_a1_0d, _a1_fold)
                            print(f">>> descent(computed): pinch {_pz_d:.3f} -> object {_z_tgt:.3f} "
                                  f"({_need * 1000:+.0f}mm). SOLVED h1 {_h1_t:.3f} h2 {_h2_t:.3f} "
                                  f"(dh {_h2_t - _h1_t:+.3f}) a1 {_a1_solved:.3f} -> reach "
                                  f"{_reach_d:.3f}m; columns tilt INTO it as they descend",
                                  flush=True)
                        else:
                            _h1_t, _h2_t = None, None
                            print(f">>> descent(computed): pinch {_pz_d:.3f} -> object {_z_tgt:.3f} "
                                  f"({_need * 1000:+.0f}mm), columns {_h1_0:.3f}/{_h2_0:.3f}, "
                                  f"a1 {_a1_0d:.3f} -> {_a1_tg:.3f} (no solve -- error ramp)",
                                  flush=True)
                        # The wrist has to go where the recording put it. This branch replaced a
                        # baked replay that wrote every joint per frame; it writes only the columns,
                        # a1 and the jaw, so the wrist would stay at PARK -- visibly rotated wrong
                        # against the recording's end pose.
                        _wr_names = ("BaseJoint_1", "HandBearingJoint_1", "gripper_z_rotation_1",
                                     "gripper_y_rotation_1", "gripper_x_rotation_1")
                        _wr = []
                        try:
                            _tj = tr["names"]        # in-memory traj: names kept at load
                            for _n in _wr_names:
                                if _n in _tj and _n in self.idx:
                                    _wr.append((self.idx[_n], float(tr["frames"][-1][_tj.index(_n)])))
                        except Exception as _e_wr:
                            print(f">>> descent(computed): wrist target unavailable ({_e_wr})",
                                  flush=True)
                        if _wr:
                            print(">>> descent(computed): wrist -> recorded grasp ["
                                  + ", ".join(f"{n.split('_')[1] if '_' in n else n}"
                                              f"={v:+.3f}"
                                              for n, (_, v) in zip(_wr_names, _wr))
                                  + "] (was at park; the baked replay used to set these)",
                                  flush=True)
                        _wr0 = {_i: float(_qd0[_i]) for _i, _ in _wr}
                        _qc = _qd0.copy()
                        # Blend the jaw open. Writing the wide target into `_qc` before the loop
                        # snaps the fingers there in a single physics step; ramp them over the first
                        # part of the descent instead.
                        _jaw0 = {int(wi): float(_qd0[wi]) for wi in wide_i}
                        for _k in range(n_d):
                            _f = 0.5 - 0.5 * math.cos(math.pi * (_k + 1) / n_d)
                            _pz_now = float(self._pinch()[2])
                            _err_now = _pz_now - _z_tgt
                            _step = float(np.clip(0.6 * _err_now, -0.004, 0.004))
                            # Leave column headroom for the tilt. Driving them to their stop gets
                            # the most descent but leaves the boom differential nothing to move
                            # with, and the next stage reports no tilt authority at all.
                            _hfl = float(os.environ.get("DESCENT_H_FLOOR", "0.10"))
                            # Off by default: measured worse. Arriving pre-tilted was meant to dodge
                            # the pinch-height minimum where the tilt has no authority, and the
                            # mechanism does work -- but the result is worse, because the larger dh
                            # puts the probe on a stretch of the curve it misreads and the tilt then
                            # runs to its cap the wrong way.
                            if _h1_t is not None:
                                # Straight cosine ramp onto the solved pair -- both columns move and
                                # their difference grows into the solved tilt on the way down.
                                _qc[h1i] = float(np.clip(_h1_0 + _f * (_h1_t - _h1_0), 0.0, COLUMN_MAX))
                                _qc[h2i] = float(np.clip(_h2_0 + _f * (_h2_t - _h2_0), 0.0, COLUMN_MAX))
                            else:
                                _dht = 0.0   # target dh
                                _mean = 0.5 * (float(_qc[h1i]) + float(_qc[h2i]))
                                _dh_now = float(_qc[h2i]) - float(_qc[h1i])
                                _mean = _mean - _step
                                _dh_new = _dh_now + float(np.clip((_dht - _dh_now) * 0.02,
                                                                  -0.0015, 0.0015))
                                _mean = max(_mean, _hfl + 0.5 * abs(_dh_new))
                                _qc[h1i] = float(np.clip(_mean - 0.5 * _dh_new, 0.0, COLUMN_MAX))
                                _qc[h2i] = float(np.clip(_mean + 0.5 * _dh_new, 0.0, COLUMN_MAX))
                            # Stop if the hand reaches the robot's own body. The column height has
                            # to leave the a1 slide a path that clears the chassis; too low or too
                            # tilted and extending a1 sweeps the gripper into it.
                            if _k % 12 == 0:
                                _self_hit = [kk for kk in getattr(self, "_ctc_any", {})
                                             if ("base" in kk or "chassis" in kk or "Column" in kk
                                                 or "wheel" in kk or "roller" in kk)]
                                if _self_hit:
                                    print(f">>> descent(computed): STOP -- hand reached the robot "
                                          f"body ({_self_hit[0]}); columns held at "
                                          f"{float(_qc[h1i]):.3f}/{float(_qc[h2i]):.3f} so the a1 "
                                          f"slide still clears it", flush=True)
                                    break
                                self._ctc_any = {}
                            if _a1_d is not None:
                                _qc[_a1_d] = _a1_0d + _f * (_a1_tg - _a1_0d)
                            for _wi, _wv in _wr:                 # wrist into the recorded pose
                                _qc[_wi] = _wr0[_wi] + _f * (_wv - _wr0[_wi])
                            _fj = min(1.0, _f / 0.30)            # jaw fully wide by 30% of the drop
                            for wi in wide_i:
                                _qc[wi] = _jaw0[int(wi)] + _fj * (wide_j1 - _jaw0[int(wi)])
                            # Geometric body guard. Self-collision is off and the visible chassis
                            # mesh has no collider at all — only a thin plate — so physics will not
                            # stop the arm entering the body.
                            _i_th_g = self.idx.get("BaseJoint_1")
                            if _k % 12 == 0 and not self._clears_chassis(
                                    float(_qc[h2i] - _qc[h1i]),
                                    float(_qc[_a1_d]) if _a1_d is not None else 0.0,
                                    th=float(_qc[_i_th_g]) if _i_th_g is not None else 0.0):
                                print(f">>> descent(computed): STOP -- dh "
                                      f"{float(_qc[h2i] - _qc[h1i]):+.3f} with a1 "
                                      f"{float(_qc[_a1_d]) if _a1_d is not None else 0.0:.3f} puts "
                                      f"the gripper inside the chassis plate; holding here",
                                      flush=True)
                                break
                            self.set_base(tbx, tby, tbyaw)
                            self._force(_qc)
                            self._apply(_qc)
                            self.world.step(render=not HEADLESS)
                            # COMMANDED vs ACTUAL. The ramp commanded dh +0.150 and the joints read
                            # back +0.083 with no stop and no guard firing, so the columns are not
                            # tracking.
                            if _h1_t is not None and _k % 120 == 0:
                                _qa = np.asarray(self.robot.get_joint_positions(), float)
                                print(f">>>   descent(computed)[{_k:4d}] f={_f:.2f} "
                                      f"cmd h1 {float(_qc[h1i]):.3f}/h2 {float(_qc[h2i]):.3f} "
                                      f"(dh {float(_qc[h2i] - _qc[h1i]):+.3f}) | "
                                      f"act h1 {float(_qa[h1i]):.3f}/h2 {float(_qa[h2i]):.3f} "
                                      f"(dh {float(_qa[h2i] - _qa[h1i]):+.3f}) pinch "
                                      f"{float(self._pinch()[2]):.3f}", flush=True)
                                # ARM-vs-BODY clearance every 120 steps of the tilt+descent, so the
                                # pose where a boom link enters the chassis is readable from the
                                # log.
                                self._arm_body_clearance(f"descent {_k}")
                            if (_h1_t is None and abs(_err_now) < 0.008
                                    and _k > int(0.4 / self.dt)):
                                break
                        _qf = np.asarray(self.robot.get_joint_positions(), float)
                        print(f">>> descent(computed): columns done, pinch z "
                              f"{float(self._pinch()[2]):.3f} (target {_z_tgt:.3f}), dh "
                              f"{float(_qf[h2i] - _qf[h1i]):+.3f}", flush=True)
                        # Now tilt and extend, as part of the descent. The column ramp above buys
                        # only the MEAN height and leaves the boom nearly flat.
                        if os.environ.get("DESCENT_REACH_LOWEST", "1") == "1":
                            try:
                                self._reach_lowest(obj, label="descent/reach-lowest")
                            except Exception as _e_drl:
                                print(f">>> descent/reach-lowest: failed ({_e_drl})", flush=True)
                        _qf = np.asarray(self.robot.get_joint_positions(), float)
                        print(f">>> descent(computed): done, pinch z {float(self._pinch()[2]):.3f} "
                              f"(target {_z_tgt:.3f}), dh {float(_qf[h2i] - _qf[h1i]):+.3f}, "
                              f"a1 {float(_qf[self.idx['ArmLeftJoint_1']]):.3f}", flush=True)
                        qd = np.asarray(self.robot.get_joint_positions(), float).copy()
                        n_d = 0                              # skip the baked replay below
                    self._ctc = {}                           # who touches the object during descent
                    for k in range(n_d):
                        fi2 = (k + 1) / n_d * (len(fr2) - 1)
                        i0 = int(fi2)
                        i1 = min(i0 + 1, len(fr2) - 1)
                        qd[ii2] = (1 - (fi2 - i0)) * fr2[i0] + (fi2 - i0) * fr2[i1]
                        for wi in wide_i:                    # after the baked write, which sets them
                            qd[wi] = wide_j1
                        # Hold the a1 retraction INSIDE the replay -- the arm-side replacement for
                        # the chassis back-off. Retracting before the loop does nothing: the baked
                        # write restores it on frame 1.
                        if _a1_off > 0.0 and i_a1r is not None:
                            qd[i_a1r] = max(0.0, qd[i_a1r] - _a1_off)
                        _tr = trim * (k + 1) / n_d       # ramp to the FULL trim by the last step
                        qd[h1i] += _tr
                        qd[h2i] += _tr
                        # Track the LIVE object, not the start line, so whatever nudges it mid-drop
                        # the cage stays centred.
                        self.set_base(tbx, tby, tbyaw)
                        self._force(qd, (qd - qd_prev) / self.dt)
                        qd_prev = qd.copy()
                        self._apply(qd)
                        self.world.step(render=not HEADLESS)
                        if os.environ.get("DROP_DBG") == "1":
                            if k % 60 == 0:
                                ovv = np.asarray(obj.get_velocities(), float).ravel()
                                print(f"    drop k={k} obj_v={np.round(ovv[:3], 3).tolist()}", flush=True)
                            if not hit:
                                occ = np.asarray(obj.get_world_poses()[0][0], float)
                                if np.linalg.norm(occ[:2] - oc0[:2]) > 0.002:
                                    hit = True
                                    ov = np.asarray(obj.get_velocities(), float).ravel()
                                    print(f"    drop CONTACT k={k} obj {np.round(occ, 3).tolist()} "
                                          f"|v|={np.linalg.norm(ov[:3]):.4f} m/s", flush=True)
                                    from pxr import UsdGeom as _UG
                                    cache = _UG.XformCache()
                                    for prim in self.stage.Traverse(Usd.TraverseInstanceProxies()):
                                        if not prim.HasAPI(UsdPhysics.CollisionAPI):
                                            continue
                                        pth = prim.GetPath().pathString
                                        if f"pickup_obj_{obj_idx}" in pth:
                                            continue
                                        wp = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
                                        dd = math.sqrt((wp[0] - occ[0]) ** 2 + (wp[1] - occ[1]) ** 2
                                                       + (wp[2] - occ[2]) ** 2)
                                        if dd < 0.5:
                                            print(f"      near({dd * 1000:.0f}mm): {pth} "
                                                  f"z={wp[2]:.3f}", flush=True)
                                    for nm2, pr2 in self._hand_prims().items():
                                        pp = np.asarray(pr2.get_world_pose()[0], float)
                                        dxy2 = math.hypot(pp[0] - occ[0], pp[1] - occ[1])
                                        if dxy2 < 0.13 and pp[2] < 0.40:
                                            print(f"      {nm2}: z={pp[2]:.3f} dxy={dxy2 * 1000:.0f}mm", flush=True)
                    print(f">>> descent: {-(float(qd[h1i]) - h1_0) * 1000:.0f}mm descended "
                          f"(baked 300mm), pinch xy drift {np.linalg.norm(self._pinch()[:2] - p0) * 1000:.1f}mm",
                          flush=True)
            else:
                # Hold the fingers: the baked frames carry the recorded finger angles
                # asymmetrically, so replaying them undoes the symmetric open and the pre-close open
                # has to retract twice -- the fingers visibly open two separate times.
                self._play_traj(tr, float(os.environ.get("REACH_T", "2.5")),
                                pin=(obj, park), on_step=pin_bystanders, hold_fingers=True)
            self._finite("reach-replay")
            if PERFECT and self.clut is not None:
                # settle target = the pose the ALIGNED DESCENT actually reached. Snapping to the
                # baked end config here TELEPORTED the hand back onto the pre-align path — straight
                # into the object (the residual ~20cm knock after every otherwise-clean descent).
                qg = np.asarray(self.robot.get_joint_positions(), float).copy()
            else:
                qg[tr["idx"]] = tr["frames"][-1]             # settle target = the baked end config
                # ...but not the fingers, for the same reason: this line would silently undo the
                # symmetric open `_pick_pose` just put in `qg`.
                if os.environ.get("REACH_KEEP_FINGERS", "1") == "1":
                    for _n_f, _v_f in KNOWN["fingers_open"].items():
                        if _n_f in self.idx:
                            qg[self.idx[_n_f]] = _v_f
            if not close_anim:
                cup_fingers(qg)                              # no close anim -> static cup as before
        st.qg, st.dock, st._oc0 = qg, dock, _oc0

