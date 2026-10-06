import math

import pytest
import torch
from torch import nn

from ocean_compat.controller import (
    AttitudeController,
    ControllerBase,
    LeePositionController,
    RateController,
)


VEHICLE_PARAMS = {
    "name": "hummingbird",
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

CPU_OUTPUTS = {
    LeePositionController: [
        [-0.23739326, -0.18152070, -0.15260780, -0.17105180],
        [-0.17311889, -0.29252630, -0.20952201, -0.28556436],
    ],
    AttitudeController: [
        [-0.72418898, -0.79459769, -1.22148716, -1.18501461],
        [-1.20433915, -1.03854251, -0.72631764, -0.94114679],
    ],
    RateController: [
        [-1.06499910, -0.97534519, -0.83189887, -1.05304527],
        [-1.01710892, -0.94583410, -0.96271890, -0.98468411],
    ],
}


def make_inputs(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    state = torch.zeros((2, 13), device=device, dtype=dtype)
    state[:, 3] = 1
    state[:, 7:10] = torch.tensor(
        [[0.1, -0.2, 0.05], [-0.1, 0.03, 0.2]], device=device, dtype=dtype
    )
    state[:, 10:13] = torch.tensor(
        [[0.1, -0.2, 0.3], [-0.15, 0.08, 0.05]], device=device, dtype=dtype
    )
    return state


def run_controller(
    controller: ControllerBase, state: torch.Tensor, *, via_module_call: bool = True
) -> torch.Tensor:
    device, dtype = state.device, state.dtype
    invoke = controller if via_module_call else controller.compute
    if isinstance(controller, LeePositionController):
        return invoke(
            state,
            target_pos=torch.tensor(
                [[0.2, -0.1, 0.3], [-0.1, 0.1, 0.2]], device=device, dtype=dtype
            ),
            target_yaw=torch.tensor([[0.3], [-0.4]], device=device, dtype=dtype),
        )
    if isinstance(controller, AttitudeController):
        args = (
            state,
            torch.tensor([[0.25], [0.3]], device=device, dtype=dtype),
        )
        kwargs = {
            "target_yaw_rate": torch.tensor(
                [[0.2], [-0.1]], device=device, dtype=dtype
            ),
            "target_roll": torch.tensor([[0.1], [-0.05]], device=device, dtype=dtype),
            "target_pitch": torch.tensor([[-0.15], [0.12]], device=device, dtype=dtype),
        }
        return invoke(*args, **kwargs)
    actions = torch.tensor(
        [[0.2, 0.1, -0.1, 0.25], [-0.1, 0.15, 0.2, 0.3]], device=device, dtype=dtype
    )
    rate, thrust = controller.process_rl_actions(actions)
    args = (
        state,
        torch.tensor([[0.2, 0.1, -0.1], [-0.1, 0.15, 0.2]], device=device, dtype=dtype),
        torch.tensor([[0.25], [0.3]], device=device, dtype=dtype),
    )
    assert rate.device == device and rate.dtype == dtype
    assert thrust.device == device and thrust.dtype == dtype
    return invoke(*args)


@pytest.mark.parametrize(
    "controller_class", [LeePositionController, AttitudeController, RateController]
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_controller_is_movable_and_preserves_real_outputs(
    controller_class: type[ControllerBase], device: str, dtype: torch.dtype
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    controller = controller_class(9.81, VEHICLE_PARAMS)
    assert callable(controller)
    controller = controller.to(device=device, dtype=dtype)
    assert isinstance(controller, nn.Module)
    assert list(controller.parameters()) == []
    buffers = dict(controller.named_buffers())
    assert buffers
    assert set(controller.state_dict()) == set(buffers)
    assert all(
        buffer.device.type == device and buffer.dtype == dtype
        for buffer in buffers.values()
    )

    state = make_inputs(torch.device(device), dtype)
    commands = run_controller(controller, state)
    direct_commands = run_controller(controller, state, via_module_call=False)

    assert commands.shape == (2, 4)
    assert commands.device.type == device
    assert commands.dtype == dtype
    assert torch.isfinite(commands).all()
    expected = torch.tensor(CPU_OUTPUTS[controller_class], device=device, dtype=dtype)
    torch.testing.assert_close(commands, expected, atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(commands, direct_commands, atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_attitude_actions_match_torchrl_transform_and_compose(
    device: str, dtype: torch.dtype
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    controller = AttitudeController(9.81, VEHICLE_PARAMS).to(device=device, dtype=dtype)
    actions = torch.tensor(
        [[0.2, 0.1, -0.2, 0.3], [-0.4, -0.3, 0.25, -0.15]],
        device=device,
        dtype=dtype,
    )
    targets = controller.process_rl_actions(actions)
    assert len(targets) == 4
    assert all(target.shape == (2, 1) for target in targets)
    assert all(
        target.device.type == device and target.dtype == dtype for target in targets
    )
    expected = (
        ((actions[:, :1] + 1) / 2).clamp(min=0) * controller.max_thrusts.sum(),
        actions[:, 1:2] * torch.pi,
        actions[:, 2:3] * torch.pi,
        actions[:, 3:4] * torch.pi,
    )
    for actual, expected_target in zip(targets, expected):
        torch.testing.assert_close(actual, expected_target)

    state = make_inputs(torch.device(device), dtype)
    commands = controller(state, *targets)
    direct_commands = controller.compute(state, *targets)
    assert commands.shape == (2, 4)
    assert torch.isfinite(commands).all()
    torch.testing.assert_close(commands, direct_commands, atol=0, rtol=0)


def test_optional_targets_follow_controller_input_dtype() -> None:
    controller = AttitudeController(9.81, VEHICLE_PARAMS).double()
    state = make_inputs(torch.device("cpu"), torch.float64)
    state[:, 3] = 1

    commands = controller.compute(state, state.new_full((2, 1), 0.25))

    assert commands.dtype == torch.float64
    assert torch.isfinite(commands).all()
