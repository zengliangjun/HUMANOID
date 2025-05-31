from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.utils import math
from collections.abc import Sequence
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

def reward_orientation_euler_gravity_b(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """
    Calculates the reward for maintaining a flat base orientation. It penalizes deviation
    from the desired base orientation using the base euler angles and the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    roll, pitch, yaw = math.euler_xyz_from_quat(quat)
    # make the quaternion real-part positive if configured
    euler_mismatch = torch.exp(-(torch.abs(roll) + torch.abs(pitch)) * 10)
    gravity_mismatch = torch.exp(-torch.norm(asset.data.projected_gravity_b[:, :2], dim=1) * 20)
    return (euler_mismatch + gravity_mismatch) / 2.

