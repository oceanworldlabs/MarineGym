"""MarineGym camera sensor compatibility layer.

Ports MarineGym camera configs (PinholeCameraCfg, FisheyeCameraCfg) to
Isaac Lab 3 format. Pure dataclass configs work without Isaac Lab installed;
helper functions lazily import Isaac Lab when available.

Reference: MarineGym sensors/config.py, sensors/camera.py
(MIT License, Copyright (c) 2023 Botian Xu, Tsinghua University)
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def to_camel_case(snake_str: str, to: str = "cC") -> str:
    """Convert a snake_case string to camelCase (cC) or PascalCase (CC)."""
    if to not in ("cC", "CC"):
        raise ValueError("to_camel_case(): Choose a valid `to` argument (CC or cC)")
    components = snake_str.lower().split("_")
    if to == "cC":
        return components[0] + "".join(x.title() for x in components[1:])
    return "".join(x.title() for x in components)


def orientation_from_view(
    camera: Sequence[float],
    target: Sequence[float],
) -> Tuple[float, float, float, float]:
    """Compute a quaternion (w, x, y, z) that orients a camera to look at *target*.

    Local +Z points forward and +Y points up. Requires a finite nonzero view direction.
    """
    cam = torch.as_tensor(camera, dtype=torch.float64)
    tgt = torch.as_tensor(target, dtype=torch.float64)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

    if cam.shape != (3,) or tgt.shape != (3,):
        raise ValueError("camera and target must be three-dimensional points")
    forward = tgt - cam
    if not torch.isfinite(forward).all():
        raise ValueError("camera and target must be finite three-dimensional points")
    magnitude = forward.norm()
    if not torch.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("camera and target must define a finite nonzero view direction")
    forward = forward / magnitude

    right = torch.cross(up, forward, dim=0)
    if right.norm() < 1e-8:
        up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        right = torch.cross(up, forward, dim=0)
    right = right / right.norm()

    up_corrected = torch.cross(forward, right, dim=0)

    rot = torch.stack([right, up_corrected, forward], dim=1)
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s
        y = (rot[0, 2] - rot[2, 0]) * s
        z = (rot[1, 0] - rot[0, 1]) * s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s

    return (float(w), float(x), float(y), float(z))


# ---------------------------------------------------------------------------
# Pure dataclass configs (no Isaac Lab dependency)
# ---------------------------------------------------------------------------

@dataclass
class PinholeCameraCfg:
    """Pinhole camera sensor configuration.

    Field names and semantics are preserved from MarineGym.
    Use :func:`to_isaaclab_camera_cfg` to convert to Isaac Lab 3 format.
    """

    @dataclass
    class UsdCameraCfg:
        clipping_range: Optional[Tuple[float, float]] = None
        focal_length: Optional[float] = None
        focus_distance: Optional[float] = None
        f_stop: Optional[float] = None
        horizontal_aperture: Optional[float] = None
        horizontal_aperture_offset: Optional[float] = None
        vertical_aperture_offset: Optional[float] = None

    sensor_tick: float = 0.0
    data_types: List[str] = field(default_factory=lambda: ["rgb"])
    resolution: Tuple[int, int] = (640, 480)
    semantic_types: List[str] = field(default_factory=lambda: ["class"])
    projection_type: str = "pinhole"
    usd_params: UsdCameraCfg = field(default_factory=UsdCameraCfg)


@dataclass
class FisheyeCameraCfg(PinholeCameraCfg):
    """Fisheye camera sensor configuration (polynomial model)."""

    @dataclass
    class UsdCameraCfg(PinholeCameraCfg.UsdCameraCfg):
        fisheye_nominal_width: Optional[float] = None
        fisheye_nominal_height: Optional[float] = None
        fisheye_optical_centre_x: Optional[float] = None
        fisheye_optical_centre_y: Optional[float] = None
        fisheye_max_fov: Optional[float] = None
        fisheye_polynomial_a: Optional[float] = None
        fisheye_polynomial_b: Optional[float] = None
        fisheye_polynomial_c: Optional[float] = None
        fisheye_polynomial_d: Optional[float] = None
        fisheye_polynomial_e: Optional[float] = None

    projection_type: str = "fisheye_polynomial"
    usd_params: UsdCameraCfg = field(default_factory=UsdCameraCfg)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def load_camera_cfg_from_dict(data: dict[str, Any]) -> Union[PinholeCameraCfg, FisheyeCameraCfg]:
    """Build a camera config from a plain dictionary (e.g. parsed from YAML).

    Expected keys mirror the dataclass fields. Missing keys fall back to defaults.
    """
    projection = data.get("projection_type", "pinhole")
    usd_data = data.get("usd_params", {})

    if projection == "fisheye_polynomial":
        usd_cfg = FisheyeCameraCfg.UsdCameraCfg(**{
            k: v for k, v in usd_data.items()
            if v is not None and k in FisheyeCameraCfg.UsdCameraCfg.__dataclass_fields__
        })
        return FisheyeCameraCfg(
            sensor_tick=data.get("sensor_tick", 0.0),
            data_types=data.get("data_types", ["rgb"]),
            resolution=tuple(data.get("resolution", (640, 480))),
            semantic_types=data.get("semantic_types", ["class"]),
            projection_type=projection,
            usd_params=usd_cfg,
        )

    usd_cfg = PinholeCameraCfg.UsdCameraCfg(**{
        k: v for k, v in usd_data.items()
        if v is not None and k in PinholeCameraCfg.UsdCameraCfg.__dataclass_fields__
    })
    return PinholeCameraCfg(
        sensor_tick=data.get("sensor_tick", 0.0),
        data_types=data.get("data_types", ["rgb"]),
        resolution=tuple(data.get("resolution", (640, 480))),
        semantic_types=data.get("semantic_types", ["class"]),
        projection_type=projection,
        usd_params=usd_cfg,
    )


# ---------------------------------------------------------------------------
# Isaac Lab 3 conversion
# ---------------------------------------------------------------------------

_ISAACLAB_AVAILABLE = importlib.util.find_spec("isaaclab") is not None


def _build_spawn_kwargs(usd_cfg: PinholeCameraCfg.UsdCameraCfg) -> dict[str, Any]:
    """Extract non-None USD params into a flat kwargs dict."""
    kwargs: dict[str, Any] = {}
    for fname in usd_cfg.__dataclass_fields__:
        val = getattr(usd_cfg, fname)
        if val is not None:
            kwargs[fname] = val
    return kwargs


def to_isaaclab_camera_cfg(
    cfg: Union[PinholeCameraCfg, FisheyeCameraCfg],
    prim_path: str,
    offset_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    offset_rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    offset_convention: str = "world",
) -> Any:
    """Convert a MarineGym camera config to an Isaac Lab 3 ``CameraCfg``.

    Requires Isaac Lab to be installed. Raises ``ImportError`` otherwise.

    Args:
        cfg: MarineGym camera config (PinholeCameraCfg or FisheyeCameraCfg).
        prim_path: USD prim path for the camera (may contain ``{ENV_REGEX_NS}``).
        offset_pos: Position offset from parent prim.
        offset_rot: Orientation offset quaternion (w, x, y, z).
        offset_convention: Coordinate convention for the offset rotation.

    Returns:
        An ``isaaclab.sensors.CameraCfg`` instance.
    """
    if not _ISAACLAB_AVAILABLE:
        raise ImportError(
            "Isaac Lab is required for to_isaaclab_camera_cfg(). "
            "Install isaaclab or use the pure dataclass configs directly."
        )

    from isaaclab.sensors import CameraCfg as IsaacLabCameraCfg
    from isaaclab.sim import sim_utils

    spawn_kwargs = _build_spawn_kwargs(cfg.usd_params)

    if isinstance(cfg, FisheyeCameraCfg):
        spawn = sim_utils.FisheyeCameraCfg(**spawn_kwargs)
    else:
        spawn = sim_utils.PinholeCameraCfg(**spawn_kwargs)

    return IsaacLabCameraCfg(
        prim_path=prim_path,
        update_period=cfg.sensor_tick if cfg.sensor_tick > 0 else 0.0,
        height=cfg.resolution[1],
        width=cfg.resolution[0],
        data_types=list(cfg.data_types),
        spawn=spawn,
        offset=IsaacLabCameraCfg.OffsetCfg(
            pos=offset_pos,
            rot=offset_rot,
            convention=offset_convention,
        ),
    )


def is_isaaclab_available() -> bool:
    """Return ``True`` if Isaac Lab is importable."""
    return _ISAACLAB_AVAILABLE
