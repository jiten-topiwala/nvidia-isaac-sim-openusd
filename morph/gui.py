"""The omni.ui side panel: pick an object and a slot, press MOVE.

Split out of play_isaac.py; the body is unchanged. Imports omni.ui at module level, so it is
only importable after SimulationApp exists.
"""
import os

import numpy as np
import omni.ui as ui

from morph.config import NUM_OBJECTS, SHELF_SLOTS


def build_menu(demo, state):
    """omni.ui panel: object dropdown + slot dropdown + MOVE button + status line."""
    import omni.ui as ui
    win = ui.Window("MORPH Pick & Place", width=380, height=300,
                    flags=ui.WINDOW_FLAGS_NO_SCROLLBAR)
    win.visible = True
    try:                                                    # float it top-left, above the viewport
        win.position_x, win.position_y = 20, 20
    except Exception:
        pass
    obj_labels = [f"Obj-{i}" for i in range(NUM_OBJECTS)]
    def _lvl(i):
        return "low" if SHELF_SLOTS[i, 2] < 0.5 else "mid" if SHELF_SLOTS[i, 2] < 1.0 else "high"
    slot_labels = [f"Slot-{i}  ({_lvl(i)} z={SHELF_SLOTS[i,2]:.2f}m)" for i in range(NUM_OBJECTS)]
    with win.frame:
        with ui.VStack(spacing=10, height=0):
            ui.Spacer(height=4)
            ui.Label("MORPH  Pick & Place", height=24,
                     style={"font_size": 20, "color": 0xFF33CCFF})
            ui.Label("1) pick an object   2) pick a shelf slot   3) MOVE", height=18,
                     style={"color": 0xFFBBBBBB})
            ui.Separator(height=6)
            ui.Label("Object", height=16, style={"font_size": 15})
            obj_cb = ui.ComboBox(0, *obj_labels, height=26)
            ui.Label("Shelf slot", height=16, style={"font_size": 15})
            slot_cb = ui.ComboBox(0, *slot_labels, height=26)
            ui.Spacer(height=4)

            def on_move():
                if state["busy"]:
                    return
                oi = obj_cb.model.get_item_value_model().get_value_as_int()
                si = slot_cb.model.get_item_value_model().get_value_as_int()
                state["request"] = (oi, si)
            ui.Button("MOVE", height=40, clicked_fn=on_move,
                      style={"font_size": 18, "background_color": 0xFF2E7D32})
            status_lbl = ui.Label("status: idle", height=22, word_wrap=True,
                                  style={"color": 0xFFFFCC33})

            # ---- arm jog: X/Y/Z sliders ----
            ui.Separator(height=6)
            ui.Label("Arm jog (gripper Δx/Δy/Δz from jog start, metres)", height=16,
                     style={"font_size": 14})
            jog_sliders = []
            for ax in "XYZ":
                with ui.HStack(height=22):
                    ui.Label(ax, width=14)
                    sl = ui.FloatSlider(min=-0.5, max=0.5, height=20)
                    sl.model.set_value(0.0)
                    jog_sliders.append(sl)

            def on_jog_change(*_):
                if demo.jog is not None:
                    d = np.array([s.model.get_value_as_float() for s in jog_sliders])
                    demo.jog["target"] = demo.jog["origin"] + d
            for sl in jog_sliders:
                sl.model.add_value_changed_fn(on_jog_change)

            def on_jog_toggle():
                if state["busy"]:
                    return
                if demo.jog is None:
                    for sl in jog_sliders:
                        sl.model.set_value(0.0)
                    demo.start_jog()
                    jog_btn.text = "ARM JOG: ON (drag sliders)"
                else:
                    demo.jog = None
                    demo._idle_anchor = None
                    jog_btn.text = "ARM JOG: OFF"
            jog_btn = ui.Button("ARM JOG: OFF", height=30, clicked_fn=on_jog_toggle)
    def _set_status(msg):
        try:
            status_lbl.text = f"status: {msg}"               # omni.ui.Label uses .text, not set_text()
        except Exception:
            pass
    state["status_fn"] = _set_status
    state["window"] = win                                   # KEEP a reference — else omni.ui GCs the window
    state["widgets"] = (obj_cb, slot_cb, status_lbl, jog_sliders, jog_btn)   # keep widget refs alive
    return win
