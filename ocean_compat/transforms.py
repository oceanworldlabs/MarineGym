"""Rotation-matrix and coordinate-frame transform utilities.

Pure PyTorch, no Isaac Sim / torchrl dependencies.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .math import (
    euler_to_quaternion,
    quaternion_to_rotation_matrix,
)


def euler_to_rotation_matrix(euler: Tensor) -> Tensor:
    """Convert Euler angles (roll, pitch, yaw) to a 3x3 rotation matrix.

    Args:
        euler: (..., 3) tensor of Euler angles in radians.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    q = euler_to_quaternion(euler)
    return quaternion_to_rotation_matrix(q)


def rotation_matrix_to_euler(R: Tensor) -> Tensor:
    """Extract Euler angles (roll, pitch, yaw) from rotation matrices.

    Uses the ZYX intrinsic convention, with zero yaw at gimbal lock.

    Args:
        R: (..., 3, 3) rotation matrices.

    Returns:
        (..., 3) Euler angles in radians.
    """
    horizontal = (R[..., 0, 0].square() + R[..., 1, 0].square()).sqrt()
    regular = horizontal > 4 * torch.finfo(horizontal.dtype).eps
    roll = torch.where(
        regular,
        torch.atan2(R[..., 2, 1], R[..., 2, 2]),
        torch.atan2(-R[..., 1, 2], R[..., 1, 1]),
    )
    pitch = torch.atan2(-R[..., 2, 0], horizontal)
    yaw = torch.where(
        regular, torch.atan2(R[..., 1, 0], R[..., 0, 0]), torch.zeros_like(roll)
    )
    return torch.stack([roll, pitch, yaw], dim=-1)
