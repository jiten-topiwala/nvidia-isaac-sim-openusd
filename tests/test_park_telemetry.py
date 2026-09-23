import ast
import copy
import inspect
import json
import math
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_park_ramp_func():
    path = os.path.join(ROOT, "morph", "robot.py")
    tree = ast.parse(open(path).read())
    cd = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "RobotMixin")
    fn = next(n for n in cd.body if isinstance(n, ast.FunctionDef) and n.name == "_park_ramp")
    
    glb = {"np": np, "math": math, "os": os, "HEADLESS": True}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), path, "exec"), glb)
    return glb["_park_ramp"]

def get_bench_metrics_func():
    path = os.path.join(ROOT, "play_isaac.py")
    tree = ast.parse(open(path).read())
    matches = [n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "_bench_metrics"]
    assert matches, "play_isaac.py must expose the pure _bench_metrics validator"
    # The module-level constants it reads come from the SOURCE, not a copy kept here: a duplicated
    # bound drifts, and this one already cost a healthy cycle when it matched the track tolerance.
    reads = {n.id for n in ast.walk(matches[0])
             if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    consts = [n for n in tree.body
              if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in reads]
    glb = {"os": os, "np": np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=consts, type_ignores=[])),
                 path, "exec"), glb)
    exec(compile(ast.fix_missing_locations(
        ast.Module(body=[matches[0]], type_ignores=[])), path, "exec"), glb)
    return glb["_bench_metrics"]

class FakeRobot:
    def __init__(self, q_init):
        self.q = np.array([q_init], dtype=float)

    def get_joint_positions(self):
        return self.q.copy()

class FakeSelf:
    def __init__(self, q_init):
        self.idx = {"gripper_x_rotation_1": 0}
        self.q0 = [0.0]
        self.dt = 1/240.0
        self.clut = None
        self.robot = FakeRobot(q_init)
        self._last_apply = None

    def _drive_on(self):
        return True

    def _force(self, q, qv=None):
        pass

    def _apply(self, q, qv=None):
        self._last_apply = q.copy()

    def set_base(self, *args, **kwargs):
        pass

class FakeWorld:
    def __init__(self, fake_self):
        self.fake_self = fake_self
        self.current_time_step_index = 0
        
    def step(self, render=False):
        if self.fake_self._last_apply is not None:
            self.fake_self.robot.q[0] += 0.8 * (self.fake_self._last_apply[0] - self.fake_self.robot.q[0])
        self.current_time_step_index += 1

class FakeWorldWithoutStepIndex:
    def __init__(self, fake_self):
        self.fake_self = fake_self

    def step(self, render=False):
        if self.fake_self._last_apply is not None:
            self.fake_self.robot.q[0] += 0.5 * (self.fake_self._last_apply[0] - self.fake_self.robot.q[0])

def _stages(include_backout=True):
    names = ["_pick_setup", "_pick_pose", "_pick_reach", "_pick_settle", "_pick_descend",
             "_pick_close", "_pick_seat", "_pick_capture", "_pick_lift", "drive_to",
             "pre_dock_loop", "release", "retreat", "park", "raise", "slide", "lower"]
    if include_backout:
        names.append("backout")
    return [(name, "test:1", 1.0) for name in names]


def test_bench_metrics_carries_the_run_metrics_it_is_handed():
    """T3: slip, clearance and replans are reported beside the verdict, never judged by it."""
    _bench_metrics = get_bench_metrics_func()
    extra = {"slip": [("released", 1.2, 0.5, None)], "clearance": [("reach", 8, 0.031)],
             "clearance_min_m": 0.031, "replans": 1}
    with_extra = _bench_metrics(_stages(), [(0.01, True)], _valid_telemetry(), 20.0, False, extra)
    without = _bench_metrics(_stages(), [(0.01, True)], _valid_telemetry(), 20.0, False)
    assert all(with_extra[k] == v for k, v in extra.items()), with_extra
    assert with_extra["mode"] == without["mode"] and "replans" not in without


def test_a_fast_planner_refusal_is_not_a_wrong_wrap_decoy():
    """The decoy read any plan under 5 ms as proof the wrap never ran. Through the persistent
    worker a refused plan takes 2 ms and still answers with an event record, so the event is the
    signal and the duration is not."""
    _bench_metrics = get_bench_metrics_func()
    quick = _bench_metrics(_stages(), [(0.002, True), (0.5, True)], _valid_telemetry(), 20.0, False)
    assert quick["captured"] is True, quick["mode"]
    silent = _bench_metrics(_stages(), [(0.002, False), (0.5, True)], _valid_telemetry(), 20.0, False)
    assert silent["mode"] == "WRONG WRAP DECOY", silent
    assert _bench_metrics(_stages(), [], _valid_telemetry(), 20.0, False)["mode"] == "WRONG WRAP DECOY"


def test_backout_is_required_and_covers_the_complete_separation():
    _bench_metrics = get_bench_metrics_func()
    complete = _bench_metrics(_stages(), [(0.01, True)], _valid_telemetry(), 20.0, False)
    missing = _bench_metrics(_stages(include_backout=False), [(0.01, True)],
                             _valid_telemetry(), 20.0, False)
    assert complete["captured"] is True, complete
    assert missing["mode"] == "UNACCOUNTED TIME", missing

    path = os.path.join(ROOT, "morph", "place", "__init__.py")
    tree = ast.parse(open(path).read())
    place = next(n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "_place_impl")
    backout = next((n for n in ast.walk(place) if isinstance(n, ast.With)
                    and any(isinstance(item.context_expr, ast.Call)
                            and isinstance(item.context_expr.func, ast.Attribute)
                            and item.context_expr.func.attr == "_time_stage"
                            and item.context_expr.args
                            and isinstance(item.context_expr.args[0], ast.Constant)
                            and item.context_expr.args[0].value == "backout"
                            for item in n.items)), None)
    assert backout is not None, "post-release separation is absent from wall-clock accounting"
    calls = [n for n in ast.walk(backout) if isinstance(n, ast.Call)]
    assert any(isinstance(n.func, ast.Name) and n.func.id == "cartesian_retreat"
               for n in calls), "backout timer excludes the planned separation"
    assert any(isinstance(n.func, ast.Attribute) and n.func.attr == "_carry_check"
               and len(n.args) > 1 and isinstance(n.args[1], ast.Constant)
               and n.args[1].value == "settled" for n in calls), \
        "backout timer stops before the final settled-object measurement"


def _valid_telemetry():
    return {
        "pre_step": [(0.01, 0.0)], "post_step": [(0.01, 0.0095)],
        "cmd_step": [0.01], "step_index": [(1, 2)], "source": "post-retreat",
        "path": "world.step", "drive_on": True,
    }

def test_park_telemetry():
    os.environ["TRACK_ERR"] = "1"
    _park_ramp = get_park_ramp_func()
    assert "source" in inspect.signature(_park_ramp).parameters, \
        "park telemetry must identify the ramp that produced it"
    
    # 1. First call from a far pose
    # A pose of 4.0 keeps the captured event well above the 7 mrad floor.
    fake_self = FakeSelf(4.0)
    fake_self.world = FakeWorld(fake_self)

    _park_ramp(fake_self, source="post-retreat")
    
    assert hasattr(fake_self, "_park_telemetry"), "Telemetry missing after first call"
    tele1 = fake_self._park_telemetry
    assert tele1 is not None, "Telemetry is None after first call"
    assert tele1["source"] == "post-retreat"
    assert tele1["path"] == "world.step"
    assert tele1["drive_on"] is True
    len_first = len(tele1["pre_step"])
    assert len(tele1["post_step"]) == len_first
    assert len(tele1["step_index"]) == len_first
    assert all(after > before for before, after in tele1["step_index"])
    first_record = copy.deepcopy(tele1)
    
    max_err_first = max(abs(cmd - act) for cmd, act in tele1["pre_step"])
    assert max_err_first > 0.03, f"Expected max err > 0.03, got {max_err_first}"

    # 2. Second call from a near pose
    fake_self.robot.q = np.array([0.01], dtype=float)
    _park_ramp(fake_self, source="idle")
    
    tele2 = fake_self._park_telemetry
    
    assert tele1 is tele2, "Telemetry object was overwritten by the second call"
    assert len(tele2["pre_step"]) == len_first, "Telemetry recorded the second ramp instead of keeping only the first"
    max_err_second = max(abs(cmd - act) for cmd, act in tele2["pre_step"])
    assert max_err_second == max_err_first, "Telemetry content changed"
    assert tele2["source"] == "post-retreat", "Idle ramp replaced the post-retreat source"
    assert tele2 == first_record, "Idle ramp changed the retained post-retreat evidence"

    # 3. Third call with reset telemetry
    fake_self._park_telemetry = None
    fake_self.robot.q = np.array([0.01], dtype=float)
    _park_ramp(fake_self, source="post-retreat")
    
    assert fake_self._park_telemetry is not None, "Telemetry not recorded after reset"
    assert fake_self._park_telemetry is not tele1, "Telemetry should be a new dict"
    assert len(fake_self._park_telemetry["pre_step"]) > 0, "Telemetry is empty after reset"
    
    print("all passed")

def test_bench_metrics_accepts_smooth_lagging_post_step_capture():
    os.environ["TRACK_ERR"] = "1"
    _park_ramp = get_park_ramp_func()
    _bench_metrics = get_bench_metrics_func()
    fake_self = FakeSelf(4.0)
    fake_self.world = FakeWorld(fake_self)
    _park_ramp(fake_self, source="post-retreat")

    tele = fake_self._park_telemetry
    post_max = max(abs(cmd - act) for cmd, act in tele["post_step"])
    assert post_max > 1e-3, "Fixture must retain realistic one-step drive lag"
    assert post_max <= 7.5e-3, "Fixture must stay within the measured wrist floor"
    metrics = _bench_metrics(_stages(), [(0.01, True)], tele, 20.0, False)
    assert metrics["captured"] is True, metrics
    assert metrics["mode"] is None


def test_bench_metrics_rejects_stuck_post_step_capture():
    _bench_metrics = get_bench_metrics_func()
    stuck = _valid_telemetry()
    stuck["post_step"] = [(command, actual) for command, actual in stuck["pre_step"]]
    metrics = _bench_metrics(_stages(), [(0.01, True)], stuck, 20.0, False)
    assert metrics["captured"] is False, metrics
    assert metrics["mode"] == "SILENT WRONG-REGIME CAPTURE", metrics


def test_bench_metrics_rejects_underresponsive_near_target_capture():
    _bench_metrics = get_bench_metrics_func()
    slow = _valid_telemetry()
    slow["pre_step"] = [(0.0074, 0.0)]
    slow["post_step"] = [(0.0074, 0.0001)]
    slow["cmd_step"] = [0.0074]
    metrics = _bench_metrics(_stages(), [(0.01, True)], slow, 20.0, False)
    assert metrics["captured"] is False, metrics
    assert metrics["mode"] == "SILENT WRONG-REGIME CAPTURE", metrics


def test_bench_metrics_rejects_capture_without_command_stimulus():
    _bench_metrics = get_bench_metrics_func()
    idle = _valid_telemetry()
    idle["pre_step"] = [(0.0074, 0.0)]
    idle["post_step"] = [(0.0074, 0.0001)]
    idle["cmd_step"] = [0.0]
    metrics = _bench_metrics(_stages(), [(0.01, True)], idle, 20.0, False)
    assert metrics["captured"] is False, metrics
    assert metrics["mode"] == "SILENT WRONG-REGIME CAPTURE", metrics


def test_a_cycle_at_the_drives_own_floor_is_not_called_wrong_regime():
    """A real cycle failed this decoy by 1.878 MICROMETRES: its wrist sat at the drive's own
    tracking floor and the decoy's bound WAS the track tolerance. The decoy asks whether the right
    thing was captured; verify_place's track gate grades how well it tracked. Keep them apart."""
    _bench_metrics = get_bench_metrics_func()
    tele = _valid_telemetry()
    # The measured shape: the commanded travel is answered in full, and the residual sits just past
    # 7.5e-3 -- which is the drive's floor, not evidence that the wrong phase was captured.
    d, lag = 0.003, 0.0075019          # 3 mm per step, a CONSTANT 7.5019 mm behind: the measured shape
    cmd = [d * (k + 1) for k in range(240)]
    tele["pre_step"] = [(c, c - d - lag) for c in cmd]     # measured before this step's physics
    tele["post_step"] = [(c, c - lag) for c in cmd]        # and after it: the lag never grows
    tele["cmd_step"] = [d] * 240
    tele["step_index"] = [(k + 1, k + 2) for k in range(240)]
    metrics = _bench_metrics(_stages(), [(0.01, True)], tele, 20.0, False)
    assert metrics["captured"] is True, metrics["mode"]


def test_bench_metrics_rejects_wildly_lagging_capture():
    _bench_metrics = get_bench_metrics_func()
    lagged = _valid_telemetry()
    lagged["pre_step"] = [(command, actual - 0.1) for command, actual in lagged["pre_step"]]
    lagged["post_step"] = [(command, actual - 0.1) for command, actual in lagged["post_step"]]
    metrics = _bench_metrics(_stages(), [(0.01, True)], lagged, 20.0, False)
    assert metrics["captured"] is False, metrics
    assert metrics["mode"] == "SILENT WRONG-REGIME CAPTURE", metrics

def test_bench_metrics_rejects_pre_step_capture_without_losing_evidence():
    os.environ["TRACK_ERR"] = "1"
    _park_ramp = get_park_ramp_func()
    _bench_metrics = get_bench_metrics_func()
    fake_self = FakeSelf(4.0)
    fake_self.world = FakeWorld(fake_self)
    _park_ramp(fake_self, source="post-retreat")

    tele = dict(fake_self._park_telemetry)
    tele["post_step"] = list(tele["pre_step"])
    metrics = _bench_metrics(_stages(), [(0.01, True)], tele, 20.0, False)
    assert metrics["captured"] is False
    assert metrics["mode"] == "SILENT WRONG-REGIME CAPTURE"
    assert metrics["telemetry"] == tele
    assert metrics["stages"] == _stages()
    assert metrics["latencies"] == [(0.01, True)]
    assert metrics["total_cycle_time"] == 20.0
    json.dumps(metrics)

    for key, value in (("source", "idle"), ("path", "step_fn"), ("drive_on", False)):
        wrong = copy.deepcopy(fake_self._park_telemetry)
        wrong[key] = value
        result = _bench_metrics(_stages(), [(0.01, True)], wrong, 20.0, False)
        assert result["mode"] == "SILENT WRONG-REGIME CAPTURE", (key, result)

def test_park_telemetry_failure_cannot_interrupt_motion():
    os.environ["TRACK_ERR"] = "1"
    _park_ramp = get_park_ramp_func()
    fake_self = FakeSelf(4.0)
    fake_self.world = FakeWorldWithoutStepIndex(fake_self)
    _park_ramp(fake_self, source="post-retreat")
    assert len(fake_self._park_telemetry["step_index"]) == 240

if __name__ == "__main__":
    test_bench_metrics_carries_the_run_metrics_it_is_handed()
    test_a_fast_planner_refusal_is_not_a_wrong_wrap_decoy()
    test_backout_is_required_and_covers_the_complete_separation()
    test_park_telemetry()
    test_bench_metrics_accepts_smooth_lagging_post_step_capture()
    test_bench_metrics_rejects_stuck_post_step_capture()
    test_bench_metrics_rejects_underresponsive_near_target_capture()
    test_bench_metrics_rejects_capture_without_command_stimulus()
    test_a_cycle_at_the_drives_own_floor_is_not_called_wrong_regime()
    test_bench_metrics_rejects_wildly_lagging_capture()
    test_bench_metrics_rejects_pre_step_capture_without_losing_evidence()
    test_park_telemetry_failure_cannot_interrupt_motion()
    print("all passed")
