"""Pure-PyTorch math utilities ported from MarineGym.

Quaternion convention throughout: (w, x, y, z) — scalar-first / aerospace.
All functions operate on batched tensors with the quaternion / vector
dimension as the last axis.

Ported from marinegym/utils/torch.py (MIT License,
Copyright (c) 2023 Botian Xu, Tsinghua University).
"""

from __future__ import annotations

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------


def euler_to_quaternion(euler: Tensor) -> Tensor:
    """Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z).

    Applies intrinsic ZYX rotation: yaw -> pitch -> roll.

    Args:
        euler: (..., 3) tensor of Euler angles in radians.

    Returns:
        (..., 4) tensor of unit quaternions in (w, x, y, z) order.
    """
    r, p, y = torch.unbind(euler, dim=-1)
    cr, sr = torch.cos(r * 0.5), torch.sin(r * 0.5)
    cp, sp = torch.cos(p * 0.5), torch.sin(p * 0.5)
    cy, sy = torch.cos(y * 0.5), torch.sin(y * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return torch.stack([qw, qx, qy, qz], dim=-1)


def quaternion_to_euler(quaternion: Tensor) -> Tensor:
    """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw).

    Uses the ZYX intrinsic convention.

    Args:
        quaternion: (..., 4) tensor of unit quaternions.

    Returns:
        (..., 3) tensor of Euler angles in radians (roll, pitch, yaw).
    """
    w, x, y, z = torch.unbind(quaternion, dim=-1)

    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack([roll, pitch, yaw], dim=-1)


def quaternion_to_rotation_matrix(quaternion: Tensor) -> Tensor:
    """Convert quaternion (w, x, y, z) to a 3x3 rotation matrix.

    Args:
        quaternion: (..., 4) tensor of unit quaternions.

    Returns:
        (..., 3, 3) tensor of rotation matrices.
    """
    w, x, y, z = torch.unbind(quaternion, dim=-1)
    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z

    matrix = torch.stack(
        [
            1 - (tyy + tzz), txy - twz, txz + twy,
            txy + twz, 1 - (txx + tzz), tyz - twx,
            txz - twy, tyz + twx, 1 - (txx + tyy),
        ],
        dim=-1,
    )
    return matrix.unflatten(matrix.dim() - 1, (3, 3))


def quat_mul(a: Tensor, b: Tensor) -> Tensor:
    """Hamilton product of two quaternions (w, x, y, z).

    Args:
        a: (..., 4) tensor.
        b: (..., 4) tensor (same batch shape as *a*).

    Returns:
        (..., 4) tensor — the product quaternion.
    """
    shape = a.shape
    a_flat = a.reshape(-1, 4)
    b_flat = b.reshape(-1, 4)

    w1, x1, y1, z1 = a_flat[:, 0], a_flat[:, 1], a_flat[:, 2], a_flat[:, 3]
    w2, x2, y2, z2 = b_flat[:, 0], b_flat[:, 1], b_flat[:, 2], b_flat[:, 3]

    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))

    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return torch.stack([w, x, y, z], dim=-1).view(shape)


def axis_angle_to_quaternion(angle: Tensor, axis: Tensor) -> Tensor:
    """Convert axis-angle representation to quaternion (w, x, y, z).

    Args:
        angle: (..., 1) rotation angle in radians.
        axis: (..., 3) rotation axis (will be normalized).

    Returns:
        (..., 4) unit quaternion.
    """
    axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
    return torch.cat(
        [torch.cos(angle / 2), torch.sin(angle / 2) * axis], dim=-1
    )


def axis_angle_to_matrix(angle: Tensor, axis: Tensor) -> Tensor:
    """Convert axis-angle representation to a 3x3 rotation matrix.

    Args:
        angle: (..., 1) rotation angle in radians.
        axis: (..., 3) rotation axis.

    Returns:
        (..., 3, 3) rotation matrix.
    """
    q = axis_angle_to_quaternion(angle, axis)
    return quaternion_to_rotation_matrix(q)


def quat_rotate(q: Tensor, v: Tensor) -> Tensor:
    """Rotate vector(s) *v* by quaternion(s) *q*:  v' = q * v * q^{-1}.

    Supports arbitrary batch dimensions (the last dim of *q* is 4,
    the last dim of *v* is 3; all preceding dimensions broadcast).

    Args:
        q: (..., 4) unit quaternion (w, x, y, z).
        v: (..., 3) vector(s) to rotate.

    Returns:
        (..., 3) rotated vector(s).
    """
    # Flatten to 2-D for bmm, keep original prefix shape for output.
    q_flat = q.reshape(-1, 4)
    v_flat = v.reshape(-1, 3)
    q_w = q_flat[:, 0]
    q_vec = q_flat[:, 1:]

    a = v_flat * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v_flat, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.unsqueeze(1), v_flat.unsqueeze(2)).squeeze(-1) * 2.0
    result = a + b + c
    return result.reshape(v.shape)


def quat_rotate_inverse(q: Tensor, v: Tensor) -> Tensor:
    """Rotate vector(s) *v* by the *inverse* of quaternion *q*.

    Equivalent to  v' = q^{-1} * v * q.

    Args:
        q: (..., 4) unit quaternion (w, x, y, z).
        v: (..., 3) vector(s) to rotate.

    Returns:
        (..., 3) rotated vector(s).
    """
    q_flat = q.reshape(-1, 4)
    v_flat = v.reshape(-1, 3)
    q_w = q_flat[:, 0]
    q_vec = q_flat[:, 1:]

    a = v_flat * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v_flat, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.unsqueeze(1), v_flat.unsqueeze(2)).squeeze(-1) * 2.0
    result = a - b + c
    return result.reshape(v.shape)


def quat_axis(q: Tensor, axis: int = 0) -> Tensor:
    """Get the rotated direction of a canonical axis under quaternion *q*.

    Args:
        q: (..., 4) unit quaternion (w, x, y, z).
        axis: 0 = x, 1 = y, 2 = z.

    Returns:
        (..., 3) rotated axis vector.
    """
    basis = torch.zeros(*q.shape[:-1], 3, device=q.device, dtype=q.dtype)
    basis[..., axis] = 1.0
    return quat_rotate(q, basis)


# ---------------------------------------------------------------------------
# General tensor utilities
# ---------------------------------------------------------------------------


def normalize(x: Tensor, eps: float = 1e-6) -> Tensor:
    """L2-normalize along the last dimension.

    Args:
        x: (..., D) tensor.
        eps: small constant to avoid division by zero.

    Returns:
        (..., D) normalized tensor (unit vectors).
    """
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


def off_diag(a: Tensor) -> Tensor:
    """Extract the off-diagonal entries of a square matrix (first two dims).

    For an (N, N, ...) input, returns an (N, N-1, ...) tensor where row *i*
    contains all entries of row *i* except entry (*i*, *i*).

    Args:
        a: (N, N, ...) square matrix (at least 2-D).

    Returns:
        (N, N-1, ...) off-diagonal entries.
    """
    n = a.shape[0]
    flat = a.flatten(0, 1)  # (N*N, ...)
    # Drop the diagonal (indices 0, N+1, 2(N+1), ...)
    trimmed = flat[1:].unflatten(0, (n - 1, n + 1))[:, :-1]
    return trimmed.reshape(n, n - 1, *a.shape[2:])


def symlog(x: Tensor) -> Tensor:
    """Symmetric log transform: sign(x) * log(|x| + 1).

    Inverse of :func:`symexp`.  See DreamerV3 (Hafner et al., 2023).

    Args:
        x: any-shape tensor.

    Returns:
        Same-shape transformed tensor.
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x: Tensor) -> Tensor:
    """Inverse symmetric log: sign(x) * (exp(|x|) - 1).

    Inverse of :func:`symlog`.

    Args:
        x: any-shape tensor.

    Returns:
        Same-shape transformed tensor.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
