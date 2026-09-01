"""Seat the b/c jaw against the object before anything squeezes.

Part of the close sequence — see `morph/close/__init__.py` for the stage order and the
shared state object. Imports Isaac APIs at module level; only importable after SimulationApp.
"""
import math
import os

import numpy as np

from morph.config import PERFECT, HEADLESS, KNOWN


class SeatBcStage:
    """One stage of `_close_replay`. Mixed into `CloseMixin`; all `self.*` belong to `Demo`."""

    def _close_seat_bc(self, st):
        q, obj = st.q, st.obj
        bx, by, byaw = st.bx, st.by, st.byaw
        lj_locked = st.lj_locked
        # Seat the b/c jaw against the object before anything squeezes. Without it there is nothing
        # to react the thumb against: the thumb takes up the last of the span and pushes the object
        # out of the jaw. The span slack is an invariant of the hand pose,
        #     a_gap + bc_gap = jaw_span - object_diameter
        # and a rigid hand translation is zero-sum across the two jaws -- which is exactly what is
        # wanted here. Translating toward b/c seats their pads and hands the whole slack to the
        # thumb, the one finger with the authority to close it, so the close becomes a squeeze
        # against an existing contact rather than a chase. It must NOT run when a preceding stage
        # has already seated everything: being zero-sum, it then simply opens the thumb by whatever
        # it closes b/c.
        _skip_seatbc = False
        if PERFECT:
            try:
                _oc_sk = np.asarray(obj.get_world_poses()[0][0], float)
                _oR_sk = self._obj_R(obj)
                _g_sk = {f: self._finger_surface_gap(f, _oc_sk, KNOWN["object_radius"],
                                                     self.obj_half_h, _oR_sk) for f in "abc"}
                _tol_sk = 0.006
                if max(_g_sk.values()) <= _tol_sk:
                    _skip_seatbc = True
                    print(f">>> seat-bc SKIPPED — all three already seated "
                          f"({ {k: round(v * 1000, 1) for k, v in _g_sk.items()} }mm)", flush=True)
            except Exception:
                pass
        if os.environ.get("SEAT_BC", "1") == "1" and PERFECT and not _skip_seatbc and not lj_locked:
            try:
                oc_s2 = np.asarray(obj.get_world_poses()[0][0], float)
                oR_s2 = self._obj_R(obj)
                r_s2 = KNOWN["object_radius"]
                g_bc = {f: self._finger_surface_gap(f, oc_s2, r_s2, self.obj_half_h, oR_s2)
                        for f in ("b", "c")}
                own0 = {f: self._finger_gap_owner(f, oc_s2, r_s2, self.obj_half_h, oR_s2)
                        for f in "abc"}
                fl_bc = min(g_bc, key=g_bc.get)
                need = float(np.clip(g_bc[fl_bc], 0.0,
                                     0.008))
                if need > 0.0005:
                    pa2 = self._finger_world_pts("a").mean(axis=0)
                    pb2 = self._finger_world_pts(fl_bc).mean(axis=0)
                    # Sign: a and b sit on OPPOSITE sides of the object, so the vector a->b points
                    # from a's side through the object to b's, and translating along it drives a's
                    # pad into the object while carrying b's away.
                    v2 = (pa2 - pb2)[:2]
                    v2 = v2 / max(1e-9, float(np.linalg.norm(v2)))
                    n_s2 = int(0.5 / self.dt)
                    for i_s2 in range(n_s2):
                        f_s2 = 0.5 - 0.5 * math.cos(math.pi * (i_s2 + 1) / n_s2)
                        self.set_base(float(bx + f_s2 * need * v2[0]),
                                      float(by + f_s2 * need * v2[1]), byaw)
                        self._force(q)
                        self._apply(q)
                        self.world.step(render=not HEADLESS)
                    bx, by = float(bx + need * v2[0]), float(by + need * v2[1])
                    oc_s3 = np.asarray(obj.get_world_poses()[0][0], float)
                    g2 = {f: round(self._finger_surface_gap(f, oc_s3, r_s2, self.obj_half_h,
                                                            self._obj_R(obj)) * 1000, 1)
                          for f in "abc"}
                    g3 = {f: self._finger_gap_owner(f, oc_s3, r_s2, self.obj_half_h,
                                                      self._obj_R(obj)) for f in "abc"}
                    print(f">>> seat-bc: hand moved {need * 1000:.1f}mm toward {fl_bc} "
                          f"along {np.round(v2, 3).tolist()}  gaps now {g2}", flush=True)
                    print(">>> seat-bc: closest-collider OWNER before -> after: "
                          + ", ".join(f"{f}: {own0[f][0] * 1000:+.1f}mm {own0[f][1]}"
                                      f" -> {g3[f][0] * 1000:+.1f}mm {g3[f][1]}"
                                      for f in "abc"), flush=True)
            except Exception as e:
                print(f">>> seat-bc skipped ({e})", flush=True)
        st.bx, st.by = bx, by

