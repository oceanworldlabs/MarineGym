import pytest
import torch

from ocean_compat.thruster import RotorConfig, T200Thruster


@pytest.mark.parametrize("command", [0.0, 0.05, -0.05])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_zero_state_t200_produces_zero_force(command: float, batch_size: int) -> None:
    """A neutral command from exactly zero RPM has zero force.

    This does not imply a deadband after spin-down: inherited residual RPM
    continues through the upstream RPM-to-force polynomial.
    """
    config = RotorConfig(
        force_constants=torch.full((4,), 4.4e-7),
        moment_constants=torch.zeros(4),
        max_rotation_velocities=torch.full((4,), 3900.0),
        time_constants=torch.full((4,), 0.05),
        directions=torch.ones(4),
        num_rotors=4,
    )
    thruster = T200Thruster(config, dt=0.02)
    zero_state = torch.zeros(batch_size, 4)

    forces, moments, throttle, rpm = thruster(
        torch.full_like(zero_state, command), zero_state, zero_state
    )

    torch.testing.assert_close(rpm, zero_state, atol=0, rtol=0)
    torch.testing.assert_close(forces, zero_state, atol=0, rtol=0)
    torch.testing.assert_close(moments, zero_state, atol=0, rtol=0)
    assert torch.isfinite(throttle).all()


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_spinning_t200_preserves_force_during_spin_down(direction: float) -> None:
    config = RotorConfig(
        force_constants=torch.full((4,), 4.4e-7),
        moment_constants=torch.zeros(4),
        max_rotation_velocities=torch.full((4,), 3900.0),
        time_constants=torch.full((4,), 0.05),
        directions=torch.ones(4),
        num_rotors=4,
    )
    thruster = T200Thruster(config, dt=0.02)
    zeros = torch.zeros(2, 4)
    initial_rpm = torch.full_like(zeros, direction * 1000)

    forces, _, _, rpm = thruster(zeros, zeros, initial_rpm)

    assert (rpm.abs() < initial_rpm.abs()).all()
    assert (rpm * direction > 0).all()
    assert (forces * direction > 0).all()
