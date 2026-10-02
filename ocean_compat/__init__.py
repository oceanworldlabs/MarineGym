"""Ocean compat layer for MarineGym on Isaac Sim 6 + Isaac Lab 3.

Pure-PyTorch quaternion, rotation, thruster, controller, and sensor utilities.
No Isaac Sim / torchrl / tensordict dependencies.

Quaternion convention: (w, x, y, z) — aerospace / scalar-first.

Ported from MarineGym (MIT License, Copyright (c) 2023 Botian Xu, Tsinghua University).
"""

__version__ = "0.1.0a0"

from ocean_compat.math import (
    euler_to_quaternion,
    normalize,
    off_diag,
    quat_axis,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
    symexp,
    symlog,
)
from ocean_compat.thruster import RotorConfig, RotorGroupModel, T200Thruster
from ocean_compat.controller import (
    AttitudeController,
    ControllerBase,
    LeePositionController,
    RateController,
)
from ocean_compat.sensor import (
    FisheyeCameraCfg,
    PinholeCameraCfg,
    orientation_from_view,
    load_camera_cfg_from_dict,
)

__all__ = [
    "__version__",
    "euler_to_quaternion",
    "normalize",
    "off_diag",
    "quat_axis",
    "quat_mul",
    "quat_rotate",
    "quat_rotate_inverse",
    "quaternion_to_euler",
    "quaternion_to_rotation_matrix",
    "symexp",
    "symlog",
    "RotorConfig",
    "RotorGroupModel",
    "T200Thruster",
    "ControllerBase",
    "LeePositionController",
    "AttitudeController",
    "RateController",
    "FisheyeCameraCfg",
    "PinholeCameraCfg",
    "orientation_from_view",
    "load_camera_cfg_from_dict",
]
