"""Settle onto the reach pose, then the fine align that servos the pinch centroid onto the
object centre.

Part of the pick cycle — see `morph/pick/__init__.py` for the stage order and the shared
state object. Imports Isaac APIs at module level; only importable after SimulationApp.
"""
import math
import os

import numpy as np

from pxr import Usd

from morph.config import PERFECT, HEADLESS, KNOWN


class DescendStage:
    """One stage of `pick`. Mixed into `PickMixin`; all `self.*` belong to `Demo`."""

    def _pick_settle(self, st):
        obj_idx, status = st.obj_idx, st.status
        dock, qg, obj, park = st.dock, st.qg, st.obj, st.park
        pin_bystanders = st.pin_bystanders
        self.robot.set_joint_positions(qg)                   # snap the residue (replay/descent: ~0)

        kp = np.full(len(self.names), 2.0e3)
        kd = np.full(len(self.names), 2.0e2)
        for n in KNOWN["arm_joints"]:                        # 3e5/3e4 = the proven-stable settle gains (6e4 damping
            if n in self.idx:                                # destabilised obj4). Shake is reduced by the longer settle.
                kp[self.idx[n]], kd[self.idx[n]] = 3.0e5, 3.0e4
        for i, n in enumerate(self.names):
            if "rolling_joint" in n or "slipping" in n:
                kp[i], kd[i] = 0.0, 1.0
        self.ctrl.set_gains(kp, kd)
        # SHORT drive settle (proven baseline): the arm is already AT qg (teleported), so the drive
        # sees ~0 error.
        settle_t = 0.1 if PERFECT else float(os.environ.get("SETTLE_T", "0.4"))
        #   friction: the slow descent arrives quiet, and every extra 10th of a second at depth
        #   lets a marginal pad graze GRIND the object away (kinematic hand = irresistible)
        self._hold(qg, int(settle_t / self.dt),
                   pin=(obj, park), kin=os.environ.get("KIN_SETTLE", "1") == "1",
                   on_step=pin_bystanders)
        alive = self._finite("settle")
        if alive:
            sbx, sby, _ = self.base_pose()
            if math.hypot(sbx - float(dock[0]), sby - float(dock[1])) > 0.15:
                # A wall or rack fight during the sweep can shove the base metres off the dock while
                # staying finite, and the whole pick then runs displaced.
                print(f">>> BASE SHOVED to ({sbx:.2f},{sby:.2f}) during reach -> abort pick", flush=True)
                alive = False
        grip = qg.copy()
        st.alive, st.grip = alive, grip

    def _pick_descend(self, st):
        obj_idx, status = st.obj_idx, st.status
        O, obj, park, replay = st.O, st.obj, st.park, st.replay
        alive, grip, _oc0 = st.alive, st.grip, st._oc0
        pin_bystanders = st.pin_bystanders

        # Fine align: servo the pinch centroid ONTO the object centre — the hand comes to the
        # object, and the object does not move.
        if alive and self.clut is not None:
            if PERFECT and replay:
                # Already aligned AT HOVER, before the drop, so the verdict is OBJECT DRIFT rather
                # than the pinch centroid -- the widened descent fingers bias that reading by
                # centimetres.
                qh2 = np.asarray(self.robot.get_joint_positions(), float).copy()
                # Measured seat -- the motion the bake does not contain. At the end of the descent
                # the whole open mouth sits BEHIND the object: it is correctly between thumb and b/c
                # laterally but tens of millimetres too far forward, so the curl closes on air and a
                # marginal tip graze flicks it away.
                pb_ = self._pinch()
                ob_ = np.asarray(obj.get_world_poses()[0][0], float)
                sbx_, sby_, syaw_ = self.base_ledger()
                fwd_ = np.array([math.cos(syaw_), math.sin(syaw_)])
                lat_ = np.array([-math.sin(syaw_), math.cos(syaw_)])
                d_ = (ob_ - pb_)[:2]
                adv_f = float(np.dot(d_, fwd_)) + 0.02
                adv_l = float(np.dot(d_, lat_))
                # The geometric target is ~95mm, but the PALM runs out of room first: at the baked
                # relation it clears the object by only ~34mm, so a full advance buries it and
                # shoves the object.
                adv_f = float(np.clip(adv_f, 0.0, 0.0))
                lat_max = 0.0
                adv_l = float(np.clip(adv_l, -lat_max, lat_max))
                # Mouth seat -- the "approach object" stage this pipeline otherwise lacks: reach a
                # pre-grasp standoff, then run a straight-line Cartesian approach into the grasp, so
                # the object is never swept sideways.
                if os.environ.get("MOUTH_SEAT", "0" if PERFECT else "1") == "1" and self.clut is not None:
                    try:
                        _dep, _mw = self._mouth_depth(obj)
                        _extra = 0.0
                        _cap = 0.070
                        _adv = float(np.clip(_dep + _extra, 0.0, _cap))
                        if _adv > 1e-4:
                            fwd_, lat_ = -_mw, np.zeros(2)   # hand moves -mouth to swallow deeper
                            adv_f, adv_l = _adv, 0.0
                            print(f">>> mouth seat: pads sit {_dep * 1000:.1f}mm in FRONT of the "
                                  f"object centre -> creeping {_adv * 1000:.0f}mm along the mouth "
                                  f"axis {np.round(-_mw, 3).tolist()} (fingers wide)", flush=True)
                    except Exception as _e_ms:
                        print(f">>> mouth seat: skipped ({_e_ms})", flush=True)
                # lateral seat is off too: it was derived from the PINCH CENTROID, which is not the
                # grasp reference for this hand (contact is on the mid-links, not the tips), so it
                # kept commanding a 50mm sideways shove against an object that was already docked
                # correctly — and shoving it is what broke the grasp.
                print(f">>> seat: creeping up to {adv_f * 1000:.0f}mm forward, "
                      f"{adv_l * 1000:+.0f}mm lateral (stop on PALM contact)", flush=True)
                # Throat-centre servo. An object smaller than the throat is still knuckle-struck and
                # shoved, so the insert is MIS-AIMED: the old aim came from a PAD-derived axis, and
                # the pads swing as the fingers close while the knuckles do not, so the pad axis is
                # not the throat line.
                if os.environ.get("THROAT_SERVO", "0" if PERFECT else "1") == "1":
                    try:
                        self._knuckle_hulls()
                        from pxr import Usd as _U3, UsdGeom as _UG3
                        def _k1w(fl_):
                            acc = []
                            for pr_, loc_ in self._k1.get(fl_, ()):
                                m3 = np.array(_UG3.Xformable(pr_).ComputeLocalToWorldTransform(
                                    _U3.TimeCode.Default()), float)
                                acc.append(loc_ @ m3[:3, :3] + m3[3, :3])
                            return np.vstack(acc) if acc else None
                        n_t = int(2.5 / self.dt)
                        rate_t = 0.05   # m/s
                        for kt in range(n_t):
                            ka_, kb_, kc_ = _k1w("a"), _k1w("b"), _k1w("c")
                            if ka_ is None or kb_ is None or kc_ is None:
                                break
                            oc_t = np.asarray(obj.get_world_poses()[0][0], float)
                            # Centre of the GAP, not the mean of the hulls. The knuckles are long
                            # parts extending forward, so their centroids are nowhere near the
                            # middle of the opening between them -- with the object centred by that
                            # definition, a knuckle is still the first thing to hit it.
                            jx_ = (0.5 * (kb_.mean(axis=0) + kc_.mean(axis=0)) - ka_.mean(axis=0))[:2]
                            jx_ = jx_ / max(1e-9, float(np.linalg.norm(jx_)))
                            _j3 = np.array([jx_[0], jx_[1], 0.0])
                            _a_face = float((ka_ @ _j3).max())        # thumb's inner face
                            _bc_face = float((np.concatenate([kb_, kc_]) @ _j3).min())
                            ctr = 0.5 * (_a_face + _bc_face) * _j3    # gap centre, on the jaw axis
                            ctr = ctr + (0.5 * (ka_.mean(axis=0) + 0.5 * (kb_.mean(axis=0)
                                                + kc_.mean(axis=0))) - _j3 * float(
                                0.5 * (ka_.mean(axis=0) + 0.5 * (kb_.mean(axis=0)
                                       + kc_.mean(axis=0))) @ _j3))
                            mx_ = np.array([-jx_[1], jx_[0]])
                            d_ = (oc_t - ctr)[:2]
                            e_lat = float(d_ @ jx_)              # object off the throat centreline
                            e_fwd = float(d_ @ mx_)              # how far ahead of the knuckles
                            if kt % 40 == 0:
                                print(f">>>   throat servo[{kt:4d}]: lateral {e_lat * 1000:+.1f}mm  "
                                      f"forward {e_fwd * 1000:+.1f}mm", flush=True)
                            if abs(e_lat) < 0.002 and abs(e_fwd) < 0.005:
                                print(f">>>   throat servo: object centred in the knuckle throat "
                                      f"(lat {e_lat * 1000:+.1f}mm, fwd {e_fwd * 1000:+.1f}mm)",
                                      flush=True)
                                break
                            step_ = rate_t * self.dt
                            mv = jx_ * np.clip(e_lat, -step_, step_) + mx_ * np.clip(e_fwd, -step_, step_)
                            bxt, byt, byawt = self.base_ledger()
                            self.set_base(float(bxt + mv[0]), float(byt + mv[1]), byawt)
                            self._force(qh2)
                            self._apply(qh2)
                            self._touch_step = {}
                            self._ctc_t = dict(self._ctc)        # snapshot before the step
                            self.world.step(render=not HEADLESS)
                            # Name the part that lands. With the object dead-centred in the throat
                            # something still strikes it ~52mm out; print WHICH prim, the moment it
                            # happens.
                            _new = {k: v - self._ctc_t.get(k, 0) for k, v in self._ctc.items()
                                    if v > self._ctc_t.get(k, 0) and k != "floor"}
                            if _new and not getattr(self, "_ts_named", False):
                                self._ts_named = True
                                print(f">>>   throat servo: FIRST CONTACT at forward "
                                      f"{e_fwd * 1000:+.1f}mm lateral {e_lat * 1000:+.1f}mm -> "
                                      f"{_new}", flush=True)
                    except Exception as _e_ts:
                        print(f">>>   throat servo: failed ({_e_ts})", flush=True)
                self._hit_step = False
                self._touch_step = {}
                self._touch = {}                                 # fresh ledger for THIS seat
                n_s = int(2.0 / self.dt)
                done_f = 0.0
                # Stop on a real touch, not on a contact EVENT. PhysX generates contacts at its
                # contact offset, so a reported "contact" can sit a centimetre clear of the surface
                # — the same trap the finger contact-stop hits.
                _seat_pen = 0.004
                for k in range(n_s):
                    _palm_real = self._touch_step.get("palm")
                    if _palm_real and os.environ.get("SEAT_PEN_GATE", "1") == "1":
                        _ct2, _ = getattr(self, "_palm_ct", (None, "?"))
                        if _ct2 is not None:
                            _oc2 = np.asarray(obj.get_world_poses()[0][0], float)
                            _rad2 = float(np.linalg.norm((_ct2 - _oc2)[:2]))
                            # contact point still outside the surface by more than the tolerance =>
                            # it is an offset-generated proximity event, not a touch
                            _palm_real = _rad2 <= KNOWN["object_radius"] + _seat_pen
                    if _palm_real:
                        # Palm contact means the cage is full: stop. Stopping on ANY contact instead
                        # halts the seat on a fingertip brushing past -- normal mouth entry -- which
                        # leaves the object at the fingers' rear-most reach, so the close converges
                        # BEHIND it and squeezes it forward out of the hand. Report WHERE on the
                        # palm, in (jaw, mouth) coordinates: a large jaw
                        #  means the hand is off to one
                        # side, a large mouth value means the leading EDGE landed first, i.e. an
                        # approach-angle error.
                        try:
                            _ocp = np.asarray(obj.get_world_poses()[0][0], float)
                            _dep_, _mwp = self._mouth_depth(obj)
                            _jwp = np.array([-_mwp[1], _mwp[0]])
                            _ct, _cprim = getattr(self, "_palm_ct", (None, "?"))
                            if _ct is None:
                                print(">>> seat: no PhysX palm contact position recorded", flush=True)
                            else:
                                _rel = _ct[:2] - _ocp[:2]
                                _rad = float(np.linalg.norm(_rel)) * 1000.0
                                _dz = (_ct[2] - _ocp[2]) * 1000.0
                                print(f">>> seat: PALM CONTACT (PhysX) — jaw "
                                      f"{float(_rel @ _jwp) * 1000:+.1f}mm  mouth "
                                      f"{float(_rel @ _mwp) * 1000:+.1f}mm  radial {_rad:.1f}mm "
                                      f"(surface {KNOWN['object_radius'] * 1000:.0f}mm)  z "
                                      f"{_ct[2]:.3f} = centre {_dz:+.0f}mm "
                                      f"(TOP is {self.obj_half_h * 1000:+.0f}mm)  on {_cprim.rsplit('/', 1)[-1]}",
                                      flush=True)
                        except Exception as _e_pt:
                            print(f">>> seat: palm contact readout failed ({_e_pt})", flush=True)
                        print(f">>> seat: PALM contact at {done_f * 1000:.0f}mm -> stop", flush=True)
                        break
                    self._touch_step = {}
                    f2 = (k + 1) / n_s                           # linear creep, no end rush
                    done_f = adv_f * f2
                    self.set_base(sbx_ + fwd_[0] * done_f + lat_[0] * adv_l * f2,
                                  sby_ + fwd_[1] * done_f + lat_[1] * adv_l * f2, syaw_)
                    self._force(qh2)
                    self._apply(qh2)
                    self.world.step(render=not HEADLESS)
                if self._touch.get("palm"):                      # unload the contact before closing
                    # just enough to unload the contact — every mm given back is depth the cage
                    # loses
                    back_off = 0.003
                    bx_n, by_n, _ = self.base_ledger()
                    for k in range(int(0.3 / self.dt)):
                        f3 = (k + 1) / int(0.3 / self.dt)
                        self.set_base(bx_n - fwd_[0] * back_off * f3,
                                      by_n - fwd_[1] * back_off * f3, syaw_)
                        self._force(qh2)
                        self._apply(qh2)
                        self.world.step(render=not HEADLESS)
                    self._hit_step = False
                _pz = self._pinch()
                print(f">>> descent end: pinch z {_pz[2]:.3f} vs object top "
                      f"{_oc0[2] + self.obj_half_h:.3f} (tips {(_oc0[2] + self.obj_half_h - _pz[2]) * 1000:.0f}"
                      f"mm below top)", flush=True)
                if self._ctc:
                    print(f">>> descent contacts (steps touching object): "
                          f"{dict(sorted(self._ctc.items(), key=lambda kv: -kv[1])[:6])}", flush=True)
                _oc = np.asarray(obj.get_world_poses()[0][0], float)
                # Tensor-only gate: a small seat-nudge from the palm-touch stop is expected
                e_align = float(np.linalg.norm(_oc[:2] - np.asarray(O, float)[:2])) - 0.03
                e_align = max(e_align, 0.0)
                print(f">>> post-approach object displacement: {e_align * 1000:.0f}mm "
                      f"(obj {np.round(_oc, 3).tolist()} pinch {np.round(self._pinch(), 3).tolist()})", flush=True)
                # Perfect mode had no closed loop at all here. This branch computes its descent from
                # the LUT, and the LUT/FK model is tens of millimetres off -- which the practical
                # path corrects with `_fine_align` in the else-branch below.
                if os.environ.get("FRICTION_FINE_ALIGN", "1") == "1":
                    _e_fa = self._fine_align(obj, park)
                    alive = self._finite("fine-align")
                    print(f">>> friction fine-align: {_e_fa * 1000:.1f}mm residual "
                          f"(was open-loop: the LUT descent's ~92mm FK error went uncorrected)",
                          flush=True)
                    e_align = max(e_align, 0.0)
            else:
                if status:
                    status(f"aligning hand on object {obj_idx}")
                e_align = self._fine_align(obj, park)
                alive = self._finite("fine-align")
            if alive and e_align > 0.05:
                # object unreachable/displaced (e.g. it was bumped away) — closing here grabs AIR
                # and the curl drives then fight whatever they land on (33kN bogus reaction)
                print(f">>> fine-align FAILED ({e_align * 1000:.0f}mm) -> abort pick", flush=True)
                alive = False
        st.alive, st.grip = alive, grip

