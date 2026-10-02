"""T200 thruster model ported from MarineGym.

Pure PyTorch, no functorch/vmap/torchrl/Isaac dependencies.
MIT License, Copyright (c) 2023 Botian Xu, Tsinghua University.

Port changes:
- Removed nn.Module base (standalone class, vehicle manages state tensors)
- Replaced nn.Parameter mutable state with regular tensors
- Removed functorch dependency (not needed, original code used none either)
- Added RotorConfig dataclass for YAML config validation
- Type-annotated all public methods
- forward() is torch.jit.script compatible
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class RotorConfig:
    """Configuration for a rotor group, parsed from vehicle YAML."""

    force_constants: Tensor          # (num_rotors,)
    moment_constants: Tensor         # (num_rotors,)
    max_rotation_velocities: Tensor  # (num_rotors,)
    time_constants: Tensor           # (num_rotors,)
    directions: Tensor               # (num_rotors,)
    num_rotors: int

    @classmethod
    def from_yaml(cls, rotor_config: Dict) -> "RotorConfig":
        """Build from the ``rotor_configuration`` dict in a vehicle YAML."""
        fc = torch.as_tensor(rotor_config["force_constants"], dtype=torch.float32)
        mc = torch.as_tensor(rotor_config["moment_constants"], dtype=torch.float32)
        mrv = torch.as_tensor(rotor_config["max_rotation_velocities"], dtype=torch.float32)
        tc = torch.as_tensor(rotor_config["time_constants"], dtype=torch.float32)
        d = torch.as_tensor(rotor_config["directions"], dtype=torch.float32)
        return cls(
            force_constants=fc,
            moment_constants=mc,
            max_rotation_velocities=mrv,
            time_constants=tc,
            directions=d,
            num_rotors=int(rotor_config["num_rotors"]),
        )


class T200Thruster:
    """Blue Robotics T200 thruster model with RPM dynamics and force curve.

    This is NOT an nn.Module. The caller (vehicle class) owns the state
    tensors (throttle, rpm) and passes them in/out each step so that
    batching across environments is handled externally.

    Usage::

        cfg = RotorConfig.from_yaml(yaml_dict["rotor_configuration"])
        thruster = T200Thruster(cfg, dt=0.02)
        throttle = torch.zeros(cfg.num_rotors)
        rpm = torch.zeros(cfg.num_rotors)
        thrusts, moments, throttle, rpm = thruster(cmds, throttle, rpm)
    """

    __slots__ = (
        "dt",
        "num_rotors",
        "force_constants",
        "moment_constants",
        "max_rot_vels",
        "time_constants",
        "directions",
        "tau_up",
        "tau_down",
        "noise_scale",
    )

    def __init__(
        self,
        rotor_config: RotorConfig,
        dt: float,
        tau_up: float = 0.43,
        tau_down: float = 0.43,
        noise_scale: float = 0.002,
    ) -> None:
        self.dt = dt
        self.num_rotors = rotor_config.num_rotors
        self.force_constants = rotor_config.force_constants
        self.moment_constants = rotor_config.moment_constants
        self.max_rot_vels = rotor_config.max_rotation_velocities
        self.time_constants = rotor_config.time_constants
        self.directions = rotor_config.directions

        # First-order time constants for throttle smoothing
        self.tau_up = torch.full((self.num_rotors,), tau_up, dtype=torch.float32)
        self.tau_down = torch.full((self.num_rotors,), tau_down, dtype=torch.float32)
        self.noise_scale = noise_scale

    def forward(
        self,
        cmds: Tensor,
        throttle: Tensor,
        rpm: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute thrusts and moments for one simulation step.

        Args:
            cmds: Raw command inputs in [-1, 1], shape ``(num_rotors,)``
                or ``(batch, num_rotors)``.
            throttle: Current throttle state, same shape as *cmds*.
            rpm: Current RPM state, same shape as *cmds*.

        Returns:
            (thrusts, moments, new_throttle, new_rpm) — all same shape as *cmds*.
        """
        # --- throttle dynamics (first-order smoothing) ---
        target_throttle = torch.clamp(cmds, -1.0, 1.0)
        tau = torch.where(
            target_throttle > throttle,
            self.tau_up,
            self.tau_down,
        )
        tau = torch.clamp(tau, 0.0, 1.0)
        new_throttle = throttle + tau * (target_throttle - throttle)

        # --- RPM target from throttle (T200 characteristic curve) ---
        # Positive throttle (>0.075): linear region
        # Negative throttle (<-0.075): linear region (reversed)
        # Dead zone [-0.075, 0.075]: zero RPM
        target_rpm = torch.where(
            new_throttle > 0.075,
            3.6599e3 * new_throttle + 3.4521e2,
            torch.where(
                new_throttle < -0.075,
                3.4944e3 * new_throttle - 4.3350e2,
                torch.zeros_like(new_throttle),
            ),
        )

        # --- RPM dynamics (exponential smoothing) ---
        alpha = torch.exp(-self.dt / self.time_constants)
        noise = torch.randn_like(rpm) * self.noise_scale * 0.0  # noise disabled, kept for API compat
        new_rpm_raw = alpha * rpm + (1.0 - alpha) * target_rpm
        new_rpm = torch.clamp(new_rpm_raw + noise, -3900.0, 3900.0)

        # --- RPM-to-force curve (T200 polynomial fit) ---
        # Positive RPM:  F = 4.7368e-7 * rpm^2 - 1.9275e-4 * rpm + 8.4452e-2
        # Negative RPM:  F = -3.8442e-7 * rpm^2 - 1.6186e-4 * rpm - 3.9139e-2
        rpm_sq = new_rpm * new_rpm
        raw_force = torch.where(
            new_rpm > 0,
            4.7368e-7 * rpm_sq - 1.9275e-4 * new_rpm + 8.4452e-2,
            -3.8442e-7 * rpm_sq - 1.6186e-4 * new_rpm - 3.9139e-2,
        )
        raw_force = torch.where(new_rpm == 0, torch.zeros_like(raw_force), raw_force)

        # Scale by force_constants ratio (supports heterogeneous rotors)
        thrusts = self.force_constants / 4.4e-7 * 9.81 * raw_force

        # Moments: thrust * direction (currently zeroed in original — kept as-is)
        moments = thrusts * (-self.directions) * 0.0

        return thrusts, moments, new_throttle, new_rpm

    # Alias so the object is callable like the original
    __call__ = forward


class RotorGroupModel:
    """Simplified rotor-group model (quadrotor-style) from MarineGym.

    Uses a square-law thrust curve instead of the T200 RPM dynamics.
    Suitable for vehicles where a simple quadratic model suffices.

    Usage::

        cfg = RotorConfig.from_yaml(yaml_dict["rotor_configuration"])
        model = RotorGroupModel(cfg, dt=0.02)
        throttle = torch.zeros(cfg.num_rotors)
        thrusts, moments, throttle = model(cmds, throttle)
    """

    __slots__ = (
        "dt",
        "num_rotors",
        "KF",
        "KM",
        "directions",
        "tau_up",
        "tau_down",
        "noise_scale",
    )

    def __init__(
        self,
        rotor_config: RotorConfig,
        dt: float,
        tau_up: float = 0.43,
        tau_down: float = 0.43,
        noise_scale: float = 0.002,
    ) -> None:
        self.dt = dt
        self.num_rotors = rotor_config.num_rotors
        self.directions = rotor_config.directions

        # KF = max_rot_vel^2 * force_constant  (per rotor)
        self.KF = rotor_config.max_rotation_velocities.square() * rotor_config.force_constants
        self.KM = rotor_config.max_rotation_velocities.square() * rotor_config.moment_constants

        self.tau_up = torch.full((self.num_rotors,), tau_up, dtype=torch.float32)
        self.tau_down = torch.full((self.num_rotors,), tau_down, dtype=torch.float32)
        self.noise_scale = noise_scale

    def forward(
        self,
        cmds: Tensor,
        throttle: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute thrusts and moments for one simulation step.

        Args:
            cmds: Raw command inputs in [-1, 1], shape ``(num_rotors,)``
                or ``(batch, num_rotors)``.
            throttle: Current throttle state, same shape as *cmds*.

        Returns:
            (thrusts, moments, new_throttle) — all same shape as *cmds*.
        """
        # Map cmds [-1, 1] -> target throttle [0, 1] via sqrt (inverse of
        # the square thrust curve), then apply time-constant smoothing.
        target_throttle = torch.sqrt(torch.clamp((cmds + 1.0) / 2.0, 0.0, 1.0))

        tau = torch.where(
            target_throttle > throttle,
            self.tau_up,
            self.tau_down,
        )
        tau = torch.clamp(tau, 0.0, 1.0)
        new_throttle = throttle + tau * (target_throttle - throttle)

        noise = torch.randn_like(new_throttle) * self.noise_scale * 0.0
        t = torch.clamp(new_throttle * new_throttle + noise, 0.0, 1.0)

        thrusts = t * self.KF
        moments = (t * self.KM) * (-self.directions)

        return thrusts, moments, new_throttle

    __call__ = forward
