"""Lee attitude / rate / position controllers ported from MarineGym.

Pure PyTorch — no torchrl, tensordict, or nn.Parameter dependencies.
All gains and parameters are regular tensors (not trainable).

Ported from marinegym/controllers/ (MIT License,
Copyright (c) 2023 Botian Xu, Tsinghua University).
"""

from __future__ import annotations

import abc
from importlib.resources import files
from typing import Any, Dict, Optional, Tuple

import torch
import yaml
from torch import Tensor

from .math import (
    axis_angle_to_matrix,
    normalize,
    quat_rotate_inverse,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
)


class ControllerBase(abc.ABC):
    """Base class for MarineGym controllers (plain class, NOT nn.Module)."""

    REGISTRY: Dict[str, type] = {}

    @classmethod
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        ControllerBase.REGISTRY[cls.__name__] = cls
        ControllerBase.REGISTRY[cls.__name__.lower()] = cls

    @abc.abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> Tensor:
        ...

    @abc.abstractmethod
    def process_rl_actions(self, actions: Tensor) -> Any:
        ...


def compute_parameters(
    rotor_config: Dict[str, Any],
    inertia_matrix: Tensor,
) -> Tensor:
    """Compute the control mixer matrix from rotor configuration.

    Args:
        rotor_config: dict with keys rotor_angles, arm_lengths,
            force_constants, moment_constants, directions,
            max_rotation_velocities.
        inertia_matrix: (4, 4) inertia matrix.

    Returns:
        (num_rotors, 4) mixer matrix.
    """
    rotor_angles = torch.as_tensor(rotor_config["rotor_angles"])
    arm_lengths = torch.as_tensor(rotor_config["arm_lengths"])
    force_constants = torch.as_tensor(rotor_config["force_constants"])
    moment_constants = torch.as_tensor(rotor_config["moment_constants"])
    directions = torch.as_tensor(rotor_config["directions"])

    A = torch.stack(
        [
            torch.sin(rotor_angles) * arm_lengths,
            -torch.cos(rotor_angles) * arm_lengths,
            -directions * moment_constants / force_constants,
            torch.ones_like(rotor_angles),
        ]
    )
    mixer = A.T @ (A @ A.T).inverse() @ inertia_matrix
    return mixer


def _load_controller_yaml(vehicle_name: str) -> Dict[str, Any]:
    yaml_path = files("ocean_compat").joinpath("cfg").joinpath(
        f"lee_controller_{vehicle_name}.yaml"
    )
    with open(str(yaml_path), "r") as f:
        return yaml.safe_load(f)


def _build_inertia_matrix(inertia: Dict[str, float]) -> Tensor:
    return torch.diag_embed(
        torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"], 1.0])
    )


class LeePositionController(ControllerBase):
    """Position controller from Lee et al. (2010), https://arxiv.org/abs/1003.2005.

    Inputs:
        root_state: (..., 13) — [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        target_pos: (..., 3) — desired position
        target_vel: (..., 3) — desired velocity
        target_acc: (..., 3) — desired acceleration (feedforward)
        target_yaw: (..., 1) — desired yaw angle

    Output:
        cmd: (..., num_rotors) — normalised rotor commands in [-1, 1]
    """

    def __init__(self, g: float, uav_params: Dict[str, Any]) -> None:
        controller_params = _load_controller_yaml(uav_params["name"])

        self.pos_gain: Tensor = torch.as_tensor(controller_params["position_gain"]).float()
        self.vel_gain: Tensor = torch.as_tensor(controller_params["velocity_gain"]).float()
        self.mass: Tensor = torch.tensor(uav_params["mass"])
        self.g_vec: Tensor = torch.tensor([0.0, 0.0, g]).abs()

        rotor_config = uav_params["rotor_configuration"]
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])

        self.max_thrusts: Tensor = max_rot_vel.square() * force_constants

        inertia_matrix = _build_inertia_matrix(uav_params["inertia"])
        self.mixer: Tensor = compute_parameters(rotor_config, inertia_matrix)
        self.attitude_gain: Tensor = (
            torch.as_tensor(controller_params["attitude_gain"]).float() @ inertia_matrix[:3, :3].inverse()
        )
        self.ang_rate_gain: Tensor = (
            torch.as_tensor(controller_params["angular_rate_gain"]).float() @ inertia_matrix[:3, :3].inverse()
        )

    def compute(
        self,
        root_state: Tensor,
        target_pos: Optional[Tensor] = None,
        target_vel: Optional[Tensor] = None,
        target_acc: Optional[Tensor] = None,
        target_yaw: Optional[Tensor] = None,
        body_rate: bool = False,
    ) -> Tensor:
        batch_shape = root_state.shape[:-1]
        device = root_state.device

        if target_pos is None:
            target_pos = root_state[..., :3]
        else:
            target_pos = target_pos.expand(batch_shape + (3,))
        if target_vel is None:
            target_vel = torch.zeros(*batch_shape, 3, device=device)
        else:
            target_vel = target_vel.expand(batch_shape + (3,))
        if target_acc is None:
            target_acc = torch.zeros(*batch_shape, 3, device=device)
        else:
            target_acc = target_acc.expand(batch_shape + (3,))
        if target_yaw is None:
            target_yaw = quaternion_to_euler(root_state[..., 3:7])[..., -1:]
        else:
            if target_yaw.shape[-1] != 1:
                target_yaw = target_yaw.unsqueeze(-1)
            target_yaw = target_yaw.expand(batch_shape + (1,))

        cmd = self._compute(
            root_state.reshape(-1, 13),
            target_pos.reshape(-1, 3),
            target_vel.reshape(-1, 3),
            target_acc.reshape(-1, 3),
            target_yaw.reshape(-1, 1),
            body_rate,
        )
        return cmd.reshape(*batch_shape, -1)

    def _compute(
        self,
        root_state: Tensor,
        target_pos: Tensor,
        target_vel: Tensor,
        target_acc: Tensor,
        target_yaw: Tensor,
        body_rate: bool,
    ) -> Tensor:
        pos, rot, vel, ang_vel = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        if not body_rate:
            ang_vel = quat_rotate_inverse(rot, ang_vel)

        pos_error = pos - target_pos
        vel_error = vel - target_vel

        acc = (
            pos_error * self.pos_gain
            + vel_error * self.vel_gain
            - self.g_vec
            - target_acc
        )
        R = quaternion_to_rotation_matrix(rot)
        b1_des = torch.cat(
            [torch.cos(target_yaw), torch.sin(target_yaw), torch.zeros_like(target_yaw)],
            dim=-1,
        )
        b3_des = -normalize(acc)
        b2_des = normalize(torch.cross(b3_des, b1_des, 1))
        R_des = torch.stack(
            [b2_des.cross(b3_des, 1), b2_des, b3_des], dim=-1
        )
        ang_error_matrix = 0.5 * (
            torch.bmm(R_des.transpose(-2, -1), R)
            - torch.bmm(R.transpose(-2, -1), R_des)
        )
        ang_error = torch.stack(
            [ang_error_matrix[:, 2, 1], ang_error_matrix[:, 0, 2], ang_error_matrix[:, 1, 0]],
            dim=-1,
        )
        ang_rate_err = ang_vel
        ang_acc = (
            -ang_error * self.attitude_gain
            - ang_rate_err * self.ang_rate_gain
            + torch.linalg.cross(ang_vel, ang_vel)
        )
        thrust = -self.mass * (acc * R[:, :, 2]).sum(-1, keepdim=True)
        ang_acc_thrust = torch.cat([ang_acc, thrust], dim=-1)
        cmd = (self.mixer @ ang_acc_thrust.T).T
        cmd = (cmd / self.max_thrusts) * 2 - 1
        return cmd

    def process_rl_actions(self, actions: Tensor) -> Tuple[Tensor, Tensor]:
        target_vel, target_yaw = actions.split([3, 1], dim=-1)
        return target_vel, target_yaw * torch.pi


class AttitudeController(ControllerBase):
    """Attitude controller — stabilises roll, pitch, and yaw rate.

    Inputs:
        root_state: (..., 13) — [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        target_thrust: (..., 1) — collective thrust
        target_yaw_rate: (..., 1) — desired yaw rate
        target_roll: (..., 1) — desired roll angle
        target_pitch: (..., 1) — desired pitch angle

    Output:
        cmd: (..., num_rotors) — normalised rotor commands in [-1, 1]
    """

    def __init__(self, g: float, uav_params: Dict[str, Any]) -> None:
        rotor_config = uav_params["rotor_configuration"]
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])

        self.mass: Tensor = torch.tensor(uav_params["mass"])
        self.g_scalar: Tensor = torch.tensor(g)
        self.max_thrusts: Tensor = max_rot_vel.square() * force_constants
        inertia_matrix = _build_inertia_matrix(uav_params["inertia"])
        self.mixer: Tensor = compute_parameters(rotor_config, inertia_matrix)
        self.gain_attitude: Tensor = torch.tensor([3.0, 3.0, 0.035]) @ inertia_matrix[:3, :3].inverse()
        self.gain_angular_rate: Tensor = torch.tensor([0.52, 0.52, 0.025]) @ inertia_matrix[:3, :3].inverse()

    def compute(
        self,
        root_state: Tensor,
        target_thrust: Tensor,
        target_yaw_rate: Optional[Tensor] = None,
        target_roll: Optional[Tensor] = None,
        target_pitch: Optional[Tensor] = None,
    ) -> Tensor:
        batch_shape = root_state.shape[:-1]
        device = root_state.device

        if target_yaw_rate is None:
            target_yaw_rate = torch.zeros(*batch_shape, 1, device=device)
        if target_pitch is None:
            target_pitch = torch.zeros(*batch_shape, 1, device=device)
        if target_roll is None:
            target_roll = torch.zeros(*batch_shape, 1, device=device)

        cmd = self._compute(
            root_state.reshape(-1, 13),
            target_thrust.reshape(-1, 1),
            target_yaw_rate=target_yaw_rate.reshape(-1, 1),
            target_roll=target_roll.reshape(-1, 1),
            target_pitch=target_pitch.reshape(-1, 1),
        )
        return cmd.reshape(*batch_shape, -1)

    def _compute(
        self,
        root_state: Tensor,
        target_thrust: Tensor,
        target_yaw_rate: Tensor,
        target_roll: Tensor,
        target_pitch: Tensor,
    ) -> Tensor:
        pos, rot, vel, ang_vel = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        device = pos.device

        R = quaternion_to_rotation_matrix(rot)
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0]).unsqueeze(-1)
        yaw_mat = axis_angle_to_matrix(yaw, torch.tensor([0.0, 0.0, 1.0], device=device))
        roll_mat = axis_angle_to_matrix(target_roll, torch.tensor([1.0, 0.0, 0.0], device=device))
        pitch_mat = axis_angle_to_matrix(target_pitch, torch.tensor([0.0, 1.0, 0.0], device=device))
        R_des = torch.bmm(torch.bmm(yaw_mat, roll_mat), pitch_mat)

        angle_error_matrix = 0.5 * (
            torch.bmm(R_des.transpose(-2, -1), R)
            - torch.bmm(R.transpose(-2, -1), R_des)
        )
        angle_error = torch.stack(
            [
                angle_error_matrix[:, 2, 1],
                angle_error_matrix[:, 0, 2],
                torch.zeros(yaw.shape[0], device=device),
            ],
            dim=-1,
        )

        angular_rate_des = torch.zeros_like(ang_vel)
        angular_rate_des[:, 2] = target_yaw_rate.squeeze(1)
        angular_rate_error = ang_vel - torch.bmm(
            torch.bmm(R_des.transpose(-2, -1), R),
            angular_rate_des.unsqueeze(2),
        ).squeeze(2)

        angular_acc = (
            -angle_error * self.gain_attitude
            - angular_rate_error * self.gain_angular_rate
            + torch.linalg.cross(ang_vel, ang_vel)
        )
        angular_acc_thrust = torch.cat([angular_acc, target_thrust], dim=1)
        cmd = (self.mixer @ angular_acc_thrust.T).T
        cmd = (cmd / self.max_thrusts) * 2 - 1
        return cmd

    def process_rl_actions(self, actions: Tensor) -> Tuple[Tensor, Tensor]:
        target_rate, target_thrust = actions.split([3, 1], dim=-1)
        target_thrust = ((target_thrust + 1) / 2).clamp(min=0.0) * self.max_thrusts.sum()
        return target_rate * torch.pi, target_thrust


class RateController(ControllerBase):
    """Angular-rate controller — direct body-rate + thrust control.

    Inputs:
        root_state: (..., 13) — [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        target_rate: (..., 3) — desired body angular rates
        target_thrust: (..., 1) — collective thrust

    Output:
        cmd: (..., num_rotors) — normalised rotor commands in [-1, 1]
    """

    def __init__(self, g: float, uav_params: Dict[str, Any]) -> None:
        rotor_config = uav_params["rotor_configuration"]
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])

        self.g_scalar: Tensor = torch.tensor(g)
        self.max_thrusts: Tensor = max_rot_vel.square() * force_constants
        inertia_matrix = _build_inertia_matrix(uav_params["inertia"])
        self.mixer: Tensor = compute_parameters(rotor_config, inertia_matrix)
        self.gain_angular_rate: Tensor = torch.tensor([0.52, 0.52, 0.025]) @ inertia_matrix[:3, :3].inverse()

    def compute(
        self,
        root_state: Tensor,
        target_rate: Tensor,
        target_thrust: Tensor,
    ) -> Tensor:
        assert root_state.shape[:-1] == target_rate.shape[:-1]

        batch_shape = root_state.shape[:-1]
        root_state = root_state.reshape(-1, 13)
        target_rate = target_rate.reshape(-1, 3)
        target_thrust = target_thrust.reshape(-1, 1)

        pos, rot, linvel, angvel = root_state.split([3, 4, 3, 3], dim=1)
        body_rate = quat_rotate_inverse(rot, angvel)

        rate_error = body_rate - target_rate
        acc_des = (
            -rate_error * self.gain_angular_rate
            + torch.linalg.cross(angvel, angvel)
        )
        angacc_thrust = torch.cat([acc_des, target_thrust], dim=1)
        cmd = (self.mixer @ angacc_thrust.T).T
        cmd = (cmd / self.max_thrusts) * 2 - 1
        cmd = cmd.reshape(*batch_shape, -1)
        return cmd

    def process_rl_actions(self, actions: Tensor) -> Tuple[Tensor, Tensor]:
        target_rate, target_thrust = actions.split([3, 1], dim=-1)
        target_thrust = ((target_thrust + 1) / 2).clamp(min=0.0) * self.max_thrusts.sum()
        return target_rate * torch.pi, target_thrust
