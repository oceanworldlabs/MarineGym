# ocean_compat — Isaac Sim 6 + Isaac Lab 3 Compatibility

Pure-PyTorch compatibility layer extracted from MarineGym for use with Isaac Sim 6 and Isaac Lab 3.

## Modules

| Module | Description |
|--------|-------------|
| `math.py` | Quaternion ops, rotation matrices, symlog/symexp |
| `transforms.py` | Euler ↔ rotation matrix conversions |
| `thruster.py` | T200 thruster model with RPM dynamics |
| `controller.py` | Lee position, attitude, and rate controllers |
| `sensor.py` | Pinhole/fisheye camera configs |

## Usage

```python
from ocean_compat import (
    euler_to_quaternion,
    T200Thruster,
    LeePositionController,
    PinholeCameraCfg,
)

# Quaternion math (w, x, y, z convention)
q = euler_to_quaternion(torch.tensor([0.1, 0.2, 0.3]))

# T200 thruster
thruster = T200Thruster(rotor_config, dt=0.02)
thrusts, moments, new_throttle, new_rpm = thruster(cmds, throttle, rpm)

# Controller
controller = LeePositionController(g=9.81, uav_params=params)
cmd = controller.compute(root_state, target_pos=target)
```

## Migration from Isaac 4.x

| Isaac 4.x (original MarineGym) | Isaac 6 (this compat layer) |
|-------------------------------|------------------------------|
| `omni.isaac.core` | `isaacsim.core.api` |
| `omni.isaac.cloner` | `isaacsim.core.cloner` |
| `omni.isaac.debug_draw` | `isaacsim.util.debug_draw` |
| `carb` | Removed |
| `functorch.vmap` | Direct batched ops (PyTorch native) |
| `torchrl.EnvBase` | `isaaclab.envs.DirectRLEnv` |
| `ArticulationView` | `isaaclab.assets.RigidObject` |

## License

MIT License — Copyright (c) 2023 Botian Xu, Tsinghua University.
