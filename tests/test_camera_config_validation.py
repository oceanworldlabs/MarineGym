"""Validation for camera configuration dictionaries."""

import pytest

from ocean_compat.sensor import (
    FisheyeCameraCfg,
    PinholeCameraCfg,
    load_camera_cfg_from_dict,
)


def test_rejects_unsupported_projection_type() -> None:
    with pytest.raises(ValueError, match="Unsupported camera projection_type"):
        load_camera_cfg_from_dict({"projection_type": "fisheye_kannala_brandt"})


def test_none_usd_params_uses_camera_defaults() -> None:
    config = load_camera_cfg_from_dict({"usd_params": None})

    assert isinstance(config, PinholeCameraCfg)
    assert config.usd_params == PinholeCameraCfg.UsdCameraCfg()


@pytest.mark.parametrize("usd_params", [[("focal_length", 8.0)], "not-a-mapping", 17])
def test_rejects_nonmapping_usd_params(usd_params: object) -> None:
    with pytest.raises(TypeError, match="usd_params must be a mapping or None"):
        load_camera_cfg_from_dict({"usd_params": usd_params})


def test_rejects_unknown_usd_param_instead_of_ignoring_typo() -> None:
    with pytest.raises(ValueError, match="focal_lenght"):
        load_camera_cfg_from_dict({"usd_params": {"focal_lenght": 8.0}})


def test_pinhole_accepts_known_fields_and_none_defaults() -> None:
    config = load_camera_cfg_from_dict(
        {
            "usd_params": {
                "clipping_range": (0.1, 50.0),
                "focal_length": None,
            }
        }
    )

    assert isinstance(config, PinholeCameraCfg)
    assert config.projection_type == "pinhole"
    assert config.usd_params.clipping_range == (0.1, 50.0)
    assert config.usd_params.focal_length is None


def test_fisheye_accepts_its_known_fields_and_none_defaults() -> None:
    config = load_camera_cfg_from_dict(
        {
            "projection_type": "fisheye_polynomial",
            "usd_params": {
                "fisheye_nominal_width": 640.0,
                "fisheye_polynomial_a": 0.5,
                "fisheye_polynomial_b": None,
            },
        }
    )

    assert isinstance(config, FisheyeCameraCfg)
    assert config.usd_params.fisheye_nominal_width == 640.0
    assert config.usd_params.fisheye_polynomial_a == 0.5
    assert config.usd_params.fisheye_polynomial_b is None
