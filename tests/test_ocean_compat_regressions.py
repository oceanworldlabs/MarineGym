import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from ocean_compat.controller import AttitudeController, RateController
from ocean_compat.math import quaternion_to_rotation_matrix
from ocean_compat.sensor import orientation_from_view
from ocean_compat.transforms import euler_to_rotation_matrix, rotation_matrix_to_euler


def test_standalone_import_and_controller_resources() -> None:
    code = """
import importlib.abc
import sys
class BlockOceanScale(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'oceanscale' or fullname.startswith('oceanscale.'):
            raise ModuleNotFoundError('undeclared OceanScale dependency', name=fullname)
sys.meta_path.insert(0, BlockOceanScale())
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
    torch.testing.assert_close(rotation[:, 2], expected_forward, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "target", [(0, 0, 0), (float("nan"), 0, 0), (float("inf"), 0, 0)]
)
def test_view_orientation_rejects_undefined_direction(
    target: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        orientation_from_view([0, 0, 0], target)


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
    target_rate, target_thrust = controller.process_rl_actions(
        torch.zeros(*batch_shape, 4)
    )
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
        commands = controller.compute(
            root_state, target_thrust, target_yaw_rate=target_rate[..., 2:3]
        )
    assert commands.shape == (*batch_shape, 4)
    torch.testing.assert_close(commands, torch.zeros_like(commands), atol=1e-6, rtol=0)
