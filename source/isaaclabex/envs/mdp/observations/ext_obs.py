
from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.utils import math
from isaaclab.managers import SceneEntityCfg

import numpy as np
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import RigidObject

"""
Root state.
"""

def phase_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    commands = env.command_manager.get_command(command_name)
    sin_phase = torch.sin(2 * np.pi * commands[:, :1])
    cos_phase = torch.cos(2 * np.pi * commands[:, :1])
    return torch.cat((sin_phase, cos_phase), dim = -1)


def root_euler_w(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root orientation (w, x, y, z) in the environment frame.

    If :attr:`make_quat_unique` is True, then returned quaternion is made unique by ensuring
    the quaternion has non-negative real component. This is because both ``q`` and ``-q`` represent
    the same orientation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    roll, pitch, yaw = math.euler_xyz_from_quat(quat)
    # make the quaternion real-part positive if configured
    return torch.stack((roll, pitch, yaw), dim=-1)
