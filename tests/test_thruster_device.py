"""Device and equation regressions for the stateless thruster models."""

import pytest
import torch

from ocean_compat.thruster import RotorConfig, RotorGroupModel, T200Thruster


ROTOR_CONFIG = {
    "force_constants": [4.4e-7, 4.1e-7, 4.5e-7, 4.2e-7],
    "moment_constants": [1.2e-9, 1.3e-9, 1.4e-9, 1.5e-9],
    "max_rotation_velocities": [3900.0, 3800.0, 3700.0, 3600.0],
    "time_constants": [0.01, 0.02, 0.03, 0.04],
    "directions": [1.0, -1.0, 1.0, -1.0],
    "num_rotors": 4,
}
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_t200_thruster_config_and_batched_state_follow_device(device: str) -> None:
    config = RotorConfig.from_yaml(ROTOR_CONFIG, device=device)
    thruster = T200Thruster(config, dt=0.02)
    commands = torch.tensor([[0.4, -0.2, 0.8, -0.6]], device=device)
    throttle = torch.tensor([[0.1, -0.1, 0.3, -0.3]], device=device)
    rpm = torch.tensor([[300.0, -200.0, 100.0, -50.0]], device=device)
    original_state = (throttle.clone(), rpm.clone())

    result = thruster(commands, throttle, rpm)

    assert config.force_constants.device.type == device
    assert config.time_constants.device.type == device
    assert thruster.tau_up.device.type == device
    assert thruster.tau_down.device.type == device
    for value in result:
        assert value.device.type == device
        assert value.dtype == config.force_constants.dtype
        assert torch.isfinite(value).all()
    torch.testing.assert_close(throttle, original_state[0])
    torch.testing.assert_close(rpm, original_state[1])


def test_rotor_config_defaults_to_cpu() -> None:
    config = RotorConfig.from_yaml(ROTOR_CONFIG)

    assert config.force_constants.device.type == "cpu"
    assert config.time_constants.device.type == "cpu"


@pytest.mark.parametrize("device", DEVICES)
def test_rotor_group_model_matches_upstream_square_law(device: str) -> None:
    config = RotorConfig.from_yaml(ROTOR_CONFIG, device=device)
    model = RotorGroupModel(config, dt=0.02)
    commands = torch.tensor([[-1.0, -0.25, 0.5, 1.0]], device=device)
    throttle = torch.tensor([[0.2, 0.4, 0.3, 0.8]], device=device)
    original_throttle = throttle.clone()

    target_throttle = torch.sqrt(torch.clamp((commands + 1.0) / 2.0, 0.0, 1.0))
    tau = torch.where(target_throttle > throttle, model.tau_up, model.tau_down)
    expected_throttle = throttle + torch.clamp(tau, 0.0, 1.0) * (
        target_throttle - throttle
    )
    thrust_fraction = torch.clamp(expected_throttle.square(), 0.0, 1.0)
    kf = config.max_rotation_velocities.square() * config.force_constants
    km = config.max_rotation_velocities.square() * config.moment_constants
    expected_thrust = thrust_fraction * kf
    expected_moments = (thrust_fraction * km) * -config.directions

    thrust, moments, new_throttle = model(commands, throttle)

    torch.testing.assert_close(thrust, expected_thrust)
    torch.testing.assert_close(moments, expected_moments)
    torch.testing.assert_close(new_throttle, expected_throttle)
    torch.testing.assert_close(throttle, original_throttle)
    assert model.tau_up.device.type == device
    assert model.tau_down.device.type == device
    for value in (thrust, moments, new_throttle):
        assert value.device.type == device
        assert value.dtype == config.force_constants.dtype
        assert torch.isfinite(value).all()


@pytest.mark.parametrize("model_type", [T200Thruster, RotorGroupModel])
def test_tau_tensors_follow_config_dtype(model_type: type) -> None:
    config = RotorConfig.from_yaml(ROTOR_CONFIG)
    config = RotorConfig(
        force_constants=config.force_constants.double(),
        moment_constants=config.moment_constants.double(),
        max_rotation_velocities=config.max_rotation_velocities.double(),
        time_constants=config.time_constants.double(),
        directions=config.directions.double(),
        num_rotors=config.num_rotors,
    )

    model = model_type(config, dt=0.02)

    assert model.tau_up.dtype == config.force_constants.dtype
    assert model.tau_down.dtype == config.force_constants.dtype


@pytest.mark.parametrize("num_rotors", [0, -1])
def test_rotor_config_rejects_nonpositive_num_rotors(num_rotors: int) -> None:
    config = dict(ROTOR_CONFIG, num_rotors=num_rotors)

    with pytest.raises(ValueError, match="num_rotors"):
        RotorConfig.from_yaml(config)


@pytest.mark.parametrize(
    "field",
    [
        "force_constants",
        "moment_constants",
        "max_rotation_velocities",
        "time_constants",
        "directions",
    ],
)
def test_rotor_config_rejects_wrong_vector_length(field: str) -> None:
    config = dict(ROTOR_CONFIG, **{field: ROTOR_CONFIG[field][:-1]})

    with pytest.raises(ValueError, match=field):
        RotorConfig.from_yaml(config)


@pytest.mark.parametrize(
    "field",
    [
        "force_constants",
        "moment_constants",
        "max_rotation_velocities",
        "time_constants",
        "directions",
    ],
)
def test_rotor_config_rejects_non_vector_with_matching_numel(field: str) -> None:
    values = ROTOR_CONFIG[field]
    config = dict(ROTOR_CONFIG, **{field: [values[:2], values[2:]]})

    with pytest.raises(ValueError, match=field):
        RotorConfig.from_yaml(config)
