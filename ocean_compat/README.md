# `ocean_compat` source utilities

This experimental source package contains PyTorch math, thruster, controller,
transform, and camera-configuration utilities ported from MarineGym. It uses
PyTorch and PyYAML.

The code targets Isaac Sim 6 APIs and Isaac Lab 3 beta-era APIs. The Lab API
references checked for this snapshot are tag `v3.0.0-beta2` and the
`release/3.0.0` branch snapshot `72cd51194381caa93ee024f93e4007b1fb6c5b92`;
neither a final Isaac Lab 3 tag nor runtime execution was verified. Isaac Sim
and Isaac Lab runtime compatibility and camera rendering remain unverified.

## Modules

| Module | Description |
|--------|-------------|
| `math.py` | Quaternion operations, rotation matrices, and symlog/symexp |
| `transforms.py` | Euler-angle and rotation-matrix conversions |
| `thruster.py` | T200 thruster model with RPM dynamics |
| `controller.py` | Lee position, attitude, and rate controller utilities |
| `sensor.py` | Pinhole/fisheye camera configuration utilities |

## Source-tree example

From the repository root, this example reads the checked-in BlueROV rotor
configuration and exercises the standalone thruster and quaternion utilities:

```python
from pathlib import Path

import torch
import yaml

from ocean_compat import RotorConfig, T200Thruster, euler_to_quaternion

vehicle_path = Path("marinegym/robots/assets/usd/BlueROV/BlueROV.yaml")
vehicle = yaml.safe_load(vehicle_path.read_text(encoding="utf-8"))
rotor_config = RotorConfig.from_yaml(vehicle["rotor_configuration"])
thruster = T200Thruster(rotor_config, dt=0.02)

commands = torch.zeros(rotor_config.num_rotors)
throttle = torch.zeros_like(commands)
rpm = torch.zeros_like(commands)
thrusts, moments, throttle, rpm = thruster(commands, throttle, rpm)

q_wxyz = euler_to_quaternion(torch.tensor([0.1, 0.2, 0.3]))
```

The repository does not include a usable `LeePositionController` parameter set
for a supported vehicle, so the controller class is not presented as a
ready-to-run example. The source-tree snippet is not an installation or
simulator acceptance test. `setup.py` installs the broader MarineGym package
and retains its upstream dependencies, including pinned `torchrl==0.4.0` and
`tensordict==0.4.0`; this directory is not independently installable as a
dependency-free package.

## Interface conventions and limits

Quaternion values exposed by these utilities use public WXYZ order
`(w, x, y, z)`. Isaac Lab `OffsetCfg.rot` uses XYZW order `(x, y, z, w)` and
requires an explicit conversion at that boundary. OpenGL camera coordinates
use local `-Z` forward and `+Y` up. The camera helpers' mapping between these
conventions has not passed runtime or rendered-frame validation; do not infer
camera pose correctness from configuration construction alone.

The adapter selects a pinhole or polynomial fisheye spawn class. It does not
forward custom projection metadata or map `semantic_types` to Isaac Lab's
`semantic_filter`.

Controller objects are callable `torch.nn.Module` instances with non-trainable
registered buffers. Use `.to(device=..., dtype=...)` to align controller state
with the input tensors. `RotorConfig.from_yaml(..., device=...)` places thruster
configuration and smoothing tensors on the requested device. The tensor tests
cover CPU behavior and include CUDA cases that run when CUDA is available.
They do not establish Isaac runtime or vehicle simulation acceptance.

`RateController.process_rl_actions` returns collective thrust with shape
`(..., 1)` and bounds based on the sum of rotor maximum thrusts. This matches
the upstream torchrl `RateController` transform, which also uses the summed
maximum. It differs from the upstream `RateController.process_rl_actions`
method, which returns per-rotor values and is not directly composable with the
upstream controller computation; this distinction is an API-shape note, not a
claim of changed physical behavior.

Inherited scientific limitations remain: the controllers' gyroscopic term
`cross(angular_velocity, angular_velocity)` is zero, and the T200 model retains
the upstream zero torque/noise terms and RPM-to-force polynomial. Zero force
at exactly zero initial RPM does not imply zero residual force after spin-down.

## License and attribution

The repository is MIT licensed, Copyright (c) 2025 Shuguang Chu, Zhejiang
University. The ported MarineGym math routines also retain their upstream
attribution to Copyright (c) 2023 Botian Xu, Tsinghua University.
