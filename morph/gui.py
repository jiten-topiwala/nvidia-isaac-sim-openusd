"""The omni.ui side panel: pick an object and a slot, press MOVE, with the jog below.

Imports omni.ui at module level, so it is only importable after SimulationApp exists.
"""

import math

import numpy as np
import omni.ui as ui

from morph.config import NUM_OBJECTS, SHELF_SLOTS

# The base joystick, in pixels: inner disc translates, outer ring rotates.
JOY_INNER_PX = 46
JOY_RING_PX = 16
JOY_GAP_PX = 10
JOY_OUTER_PX = JOY_INNER_PX + JOY_GAP_PX + JOY_RING_PX
JOY_KNOB_PX = 8
# Below this fraction of the inner disc the stick reads as centred, so a resting hand does not
# creep the chassis.
JOY_DEAD_ZONE = 0.1
from morph.gripper.api import gripper_fraction


def build_menu(demo, state):
    """omni.ui panel: object + slot + MOVE + status, then the arm, base and gripper jog."""
    import omni.ui as ui
    # The ScrollingFrame below owns scrolling, so the window's own bar is off; NO_SAVED_SETTINGS
    # stops ImGui restoring a remembered position a frame after this is built.
    win = ui.Window("MORPH Pick & Place", width=380, height=300,
                    flags=ui.WINDOW_FLAGS_NO_SCROLLBAR | ui.WINDOW_FLAGS_NO_SAVED_SETTINGS)
    win.visible = True
    # top-left of the viewport: x clears Kit's icon toolbar, which owns the first ~45px
    win.position_x, win.position_y = 55, 20
    obj_labels = [f"Obj-{i}" for i in range(NUM_OBJECTS)]
    def _lvl(i):
        return "low" if SHELF_SLOTS[i, 2] < 0.5 else "mid" if SHELF_SLOTS[i, 2] < 1.0 else "high"
    slot_labels = [f"Slot-{i}  ({_lvl(i)} z={SHELF_SLOTS[i,2]:.2f}m)" for i in range(NUM_OBJECTS)]
    with win.frame:
        # the window frame does not scroll on its own; this is what reaches the jog below
        with ui.ScrollingFrame(
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
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

                ui.Separator(height=6)
                # ASCII only: Kit's UI font has no glyph for an em dash or a delta and draws '?'
                ui.Label("Arm jog: gripper offset from jog start, metres", height=16,
                         style={"font_size": 14})
                # +-1.0 spans the column's 1.38m travel; _jog_tick clips every joint at its own
                # stop, so a target past the envelope parks the servo, it cannot drive through
                jog_sliders = []
                for ax in "XYZ":
                    with ui.HStack(height=22):
                        ui.Label(ax, width=14)
                        sl = ui.FloatSlider(min=-1.0, max=1.0, height=20)
                        sl.model.set_value(0.0)
                        jog_sliders.append(sl)

                ui.Label("Base jog: drag the stick to drive, the ring to turn", height=16,
                         style={"font_size": 14})
                # `radius` is only honoured under FIXED; the default STRETCH fills the stack, so
                # every circle would draw at the same size.
                _fix = ui.CircleSizePolicy.FIXED
                # The stack is the joystick's own size: filling the panel, the circles centre
                # themselves while the knob's Placer still measures from the far left edge.
                with ui.HStack(height=JOY_OUTER_PX * 2):
                  ui.Spacer()
                  with ui.ZStack(width=JOY_OUTER_PX * 2, height=JOY_OUTER_PX * 2):
                      joy_hit = ui.Circle(radius=JOY_OUTER_PX, size_policy=_fix,
                                          style={"background_color": 0xFF2B2B2B})
                      ui.Circle(radius=JOY_INNER_PX + JOY_GAP_PX, size_policy=_fix,
                                style={"background_color": 0xFF141414})
                      ui.Circle(radius=JOY_INNER_PX, size_policy=_fix,
                                style={"background_color": 0xFF4A4A4A})
                      joy_knob = ui.Placer(offset_x=JOY_OUTER_PX - JOY_KNOB_PX,
                                           offset_y=JOY_OUTER_PX - JOY_KNOB_PX)
                      with joy_knob:
                          ui.Circle(radius=JOY_KNOB_PX, size_policy=_fix,
                                    width=JOY_KNOB_PX * 2, height=JOY_KNOB_PX * 2,
                                    style={"background_color": 0xFFFFCC33})
                  ui.Spacer()

                # `mode` is chosen where the press LANDS and held for the drag, so a stick pushed
                # out past the inner disc does not become a turn halfway through.
                joy = {"mode": None}

                def _joy_centre():
                    w = float(joy_hit.computed_width) or JOY_OUTER_PX * 2
                    h = float(joy_hit.computed_height) or JOY_OUTER_PX * 2
                    joy_knob.offset_x = w / 2 - JOY_KNOB_PX
                    joy_knob.offset_y = h / 2 - JOY_KNOB_PX

                def _joy_xy(sx, sy):
                    """Mouse point -> (dx, dy) from the widget's centre, in pixels."""
                    w = float(joy_hit.computed_width) or JOY_OUTER_PX * 2
                    h = float(joy_hit.computed_height) or JOY_OUTER_PX * 2
                    return (float(sx) - float(joy_hit.screen_position_x) - w / 2,
                            float(sy) - float(joy_hit.screen_position_y) - h / 2, w, h)

                def _joy_press(sx, sy, *_):
                    dx, dy, _w, _h = _joy_xy(sx, sy)
                    r = math.hypot(dx, dy)
                    joy["mode"] = ("xy" if r <= JOY_INNER_PX else
                                   "yaw" if JOY_INNER_PX + JOY_GAP_PX <= r <= JOY_OUTER_PX else None)
                    _joy_drag(sx, sy)

                def _joy_drag(sx, sy, *_):
                    if demo.jog is None or joy["mode"] is None:
                        return
                    dx, dy, w, h = _joy_xy(sx, sy)
                    if joy["mode"] == "yaw":
                        # A TARGET, not the pose: `_jog_write` slews to it, or a drag across the
                        # ring would snap the chassis round in one frame.
                        demo.jog["yaw_target"] = -math.atan2(dy, dx)
                        return
                    r = math.hypot(dx, dy)
                    if r > JOY_INNER_PX:
                        dx, dy = dx * JOY_INNER_PX / r, dy * JOY_INNER_PX / r
                    joy_knob.offset_x = w / 2 + dx - JOY_KNOB_PX
                    joy_knob.offset_y = h / 2 + dy - JOY_KNOB_PX
                    nx, ny = dx / JOY_INNER_PX, -dy / JOY_INNER_PX
                    mag = math.hypot(nx, ny)
                    if mag <= JOY_DEAD_ZONE:
                        nx = ny = 0.0
                    else:
                        # rescale past the dead zone so the first commanded speed is not a step
                        k = min(1.0, (mag - JOY_DEAD_ZONE) / (1.0 - JOY_DEAD_ZONE)) / mag
                        nx, ny = nx * k, ny * k
                    demo.jog["base_vel"] = (nx, ny)

                def _joy_release(*_):
                    """Spring back: a stick left off-centre would drive the chassis unattended."""
                    joy["mode"] = None
                    if demo.jog is not None:
                        demo.jog["base_vel"] = (0.0, 0.0)
                    _joy_centre()
                joy_hit.set_mouse_pressed_fn(lambda x, y, *_: _joy_press(x, y))
                joy_hit.set_mouse_moved_fn(lambda x, y, *_: _joy_drag(x, y))
                joy_hit.set_mouse_released_fn(lambda *_: _joy_release())

                ui.Label("Gripper: 0 open, 1 closed", height=16, style={"font_size": 14})
                with ui.HStack(height=22):
                    ui.Label("grip", width=26)
                    grip_slider = ui.FloatSlider(min=0.0, max=1.0, height=20)
                    grip_slider.model.set_value(0.0)

                def on_jog_change(*_):
                    if demo.jog is not None:
                        d = np.array([s.model.get_value_as_float() for s in jog_sliders])
                        demo.jog["target"] = demo.jog["origin"] + d
                for sl in jog_sliders:
                    sl.model.add_value_changed_fn(on_jog_change)


                def on_grip_change(*_):
                    if demo.jog is not None:
                        # from the hand the jog started with, so slider 0 commands no motion
                        demo.jog["fcmd"] = gripper_fraction(
                            demo, grip_slider.model.get_value_as_float(),
                            base=demo.jog["fhold"])
                grip_slider.model.add_value_changed_fn(on_grip_change)

                def on_jog_toggle():
                    if state["busy"]:
                        return
                    if demo.jog is None:
                        for sl in (*jog_sliders, grip_slider):
                            sl.model.set_value(0.0)
                        _joy_centre()
                        demo.start_jog()
                        jog_btn.text = "JOG: ON (drag sliders)"
                    else:
                        demo.jog = None
                        demo._idle_anchor = None
                        jog_btn.text = "JOG: OFF"
                jog_btn = ui.Button("JOG: OFF", height=30, clicked_fn=on_jog_toggle)
                ui.Spacer(height=4)

    def _set_status(msg):
        try:
            # omni.ui.Label uses .text, not set_text()
            status_lbl.text = f"status: {msg}"
        except Exception:
            pass
    state["status_fn"] = _set_status
    # omni.ui garbage-collects a window with no live reference, and its widgets with it
    state["window"] = win
    state["widgets"] = (obj_cb, slot_cb, status_lbl, jog_sliders, joy_hit, joy_knob,
                        grip_slider, jog_btn)
    return win
