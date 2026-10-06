import math
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest
import torch

from ocean_compat.controller import AttitudeController, RateController
from ocean_compat.math import quaternion_to_rotation_matrix
from ocean_compat.sensor import orientation_from_view
from ocean_compat.transforms import euler_to_rotation_matrix, rotation_matrix_to_euler
import ocean_compat.sensor as sensor_compat


def test_standalone_import_and_controller_resources() -> None:
    code = """
import importlib.abc
import sys
blocked = ('oceanscale', 'isaacsim', 'isaaclab', 'omni', 'pxr', 'torchrl', 'tensordict')
class BlockOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise ModuleNotFoundError('blocked optional dependency', name=fullname)
sys.meta_path.insert(0, BlockOptionalImports())
original_find_spec = importlib.util.find_spec
def optional_absent_find_spec(name, package=None):
    if name in blocked:
        return None
    return original_find_spec(name, package)
importlib.util.find_spec = optional_absent_find_spec
for name in blocked:
    assert importlib.util.find_spec(name) is None
    try:
        __import__(name)
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError(f'{name} import was not blocked')
import ocean_compat
import ocean_compat.transforms
from ocean_compat.controller import _load_controller_yaml
assert _load_controller_yaml('hummingbird')['position_gain'] == [4, 4, 4]
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "diagonal", [(1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0)]
)
def test_exact_half_turn_preserves_rotation(diagonal: tuple[float, ...]) -> None:
    rotation = torch.diag(torch.tensor(diagonal, dtype=torch.float64))
    recovered = euler_to_rotation_matrix(rotation_matrix_to_euler(rotation))
    torch.testing.assert_close(recovered, rotation, atol=1e-12, rtol=0)


@pytest.mark.parametrize("pitch", [0.3, math.pi / 2, -math.pi / 2, math.pi / 2 - 1e-4])
def test_rotation_round_trip_preserves_batches_and_gimbal_lock(pitch: float) -> None:
    angles = torch.tensor(
        [
            [[0.3, pitch, -0.7], [math.pi - 1e-8, 0.0, 0.0]],
            [[-0.5, pitch, 1.1], [0.0, 0.0, math.pi - 1e-8]],
        ],
        dtype=torch.float64,
    )
    rotation = euler_to_rotation_matrix(angles)
    recovered_angles = rotation_matrix_to_euler(rotation)
    assert recovered_angles.shape == angles.shape
    torch.testing.assert_close(
        euler_to_rotation_matrix(recovered_angles), rotation, atol=1e-9, rtol=0
    )


@pytest.mark.parametrize(
    "direction",
    [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (1, 2, 3)],
)
def test_view_orientation_is_proper_and_points_forward(
    direction: tuple[int, ...],
) -> None:
    quaternion = torch.tensor(
        orientation_from_view([0, 0, 0], direction), dtype=torch.float64
    )
    torch.testing.assert_close(
        quaternion.norm(), torch.tensor(1.0, dtype=torch.float64)
    )
    rotation = quaternion_to_rotation_matrix(quaternion)
    torch.testing.assert_close(rotation.det(), torch.tensor(1.0, dtype=torch.float64))
    expected_forward = torch.tensor(direction, dtype=torch.float64)
    expected_forward /= expected_forward.norm()
    torch.testing.assert_close(-rotation[:, 2], expected_forward, atol=1e-12, rtol=0)


def test_view_orientation_matches_usd_opengl_look_at() -> None:
    pytest.importorskip("pxr")
    from pxr import Gf
    actual = torch.tensor(
        orientation_from_view([0, 0, 0], [1, 0, 0]), dtype=torch.float64
    )
    # USD 0.26.3 reference: SetLookAt(...).GetInverse().ExtractRotationQuat().
    usd_rotation = Gf.Matrix4d(1).SetLookAt(
        Gf.Vec3d(0, 0, 0), Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 0, 1)
    ).GetInverse().ExtractRotationQuat()
    reference = torch.tensor(
        [usd_rotation.GetReal(), *usd_rotation.GetImaginary()], dtype=torch.float64
    )
    assert abs(torch.dot(actual, reference)).item() == pytest.approx(1.0, abs=1e-12)


def test_view_orientation_toward_negative_z_is_identity() -> None:
    quaternion = torch.tensor(
        orientation_from_view([0, 0, 0], [0, 0, -1]), dtype=torch.float64
    )
    assert abs(quaternion[0]).item() == pytest.approx(1.0, abs=1e-12)
    torch.testing.assert_close(
        quaternion[1:], torch.zeros(3, dtype=torch.float64), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    "target", [(0, 0, 0), (float("nan"), 0, 0), (float("inf"), 0, 0)]
)
def test_view_orientation_rejects_undefined_direction(
    target: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        orientation_from_view([0, 0, 0], target)


@pytest.mark.parametrize(
    "wxyz, expected_xyzw, expected_x_axis",
    [
        ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        (
            (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
            (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)),
            (0.0, 1.0, 0.0),
        ),
    ],
)
def test_isaaclab_camera_adapter_converts_wxyz_to_xyzw(
    monkeypatch: pytest.MonkeyPatch,
    wxyz: tuple[float, float, float, float],
    expected_xyzw: tuple[float, float, float, float],
    expected_x_axis: tuple[float, float, float],
) -> None:
    class OffsetCfg:
        def __init__(self, *, pos, rot, convention):
            self.pos = pos
            self.rot = rot
            self.convention = convention

    class CameraCfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    CameraCfg.OffsetCfg = OffsetCfg

    isaaclab = ModuleType("isaaclab")
    isaaclab.__path__ = []
    sensors = ModuleType("isaaclab.sensors")
    sensors.CameraCfg = CameraCfg
    sim = ModuleType("isaaclab.sim")
    sim.PinholeCameraCfg = lambda **kwargs: kwargs
    sim.FisheyeCameraCfg = lambda **kwargs: kwargs
    monkeypatch.setattr(sensor_compat, "_ISAACLAB_AVAILABLE", True)
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.sensors", sensors)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim)

    cfg = sensor_compat.to_isaaclab_camera_cfg(
        sensor_compat.PinholeCameraCfg(), "/World/Camera", offset_rot=wxyz
    )

    assert cfg.offset.rot == pytest.approx(expected_xyzw)
    assert cfg.offset.convention == "opengl"
    x, y, z, w = cfg.offset.rot
    vector = torch.tensor([x, y, z], dtype=torch.float64)
    source_x = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    rotated_x = (
        2 * torch.dot(vector, source_x) * vector
        + (w * w - torch.dot(vector, vector)) * source_x
        + 2 * w * torch.linalg.cross(vector, source_x)
    )
    torch.testing.assert_close(
        rotated_x,
        torch.tensor(expected_x_axis, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )


@pytest.fixture
def vehicle_parameters() -> dict:
    return {
        "mass": 1.0,
        "inertia": {"xx": 0.1, "yy": 0.1, "zz": 0.2},
        "rotor_configuration": {
            "rotor_angles": [0, math.pi / 2, math.pi, 3 * math.pi / 2],
            "arm_lengths": [0.2] * 4,
            "directions": [1, -1, 1, -1],
            "force_constants": [4.4e-7] * 4,
            "moment_constants": [1e-8] * 4,
            "max_rotation_velocities": [3900] * 4,
        },
    }


@pytest.mark.parametrize("controller_class", [AttitudeController, RateController])
@pytest.mark.parametrize("batch_shape", [(2,), (2, 3)])
def test_collective_thrust_feeds_controller_computation(
    vehicle_parameters: dict, controller_class: type, batch_shape: tuple[int, ...]
) -> None:
    controller = controller_class(9.81, vehicle_parameters)
    actions = torch.zeros(*batch_shape, 4)
    targets = controller.process_rl_actions(actions)
    if controller_class is AttitudeController:
        target_thrust, target_yaw_rate, target_roll, target_pitch = targets
        torch.testing.assert_close(target_yaw_rate, torch.zeros_like(target_yaw_rate))
        torch.testing.assert_close(target_roll, torch.zeros_like(target_roll))
        torch.testing.assert_close(target_pitch, torch.zeros_like(target_pitch))
    else:
        target_rate, target_thrust = targets
    assert target_thrust.shape == (*batch_shape, 1)
    torch.testing.assert_close(
        target_thrust,
        torch.full((*batch_shape, 1), float(controller.max_thrusts.sum()) / 2),
    )
    root_state = torch.zeros(*batch_shape, 13)
    root_state[..., 3] = 1.0
    if controller_class is RateController:
        commands = controller.compute(root_state, target_rate, target_thrust)
    else:
        commands = controller(root_state, *targets)
    assert commands.shape == (*batch_shape, 4)
    torch.testing.assert_close(commands, torch.zeros_like(commands), atol=1e-6, rtol=0)
