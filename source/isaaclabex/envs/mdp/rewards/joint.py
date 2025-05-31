from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

def energy_cost(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    '''
    penalizes output torques to reduce energy consumption.
    '''

    asset: Articulation = env.scene[asset_cfg.name]
    joint_torque = asset.data.applied_torque
    joint_vel = asset.data.joint_vel

    return torch.sum(torch.abs(joint_torque * joint_vel), dim = -1)

'''
def reward_yaw_rool_joint_pos(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    " ""
    Calculates the reward for keeping joint positions close to default positions, with a focus
    on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
    " ""

    asset: Articulation = env.scene[asset_cfg.name]

    pose = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pose = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    diff_pose = pose - default_pose

    left_ids = asset_cfg.joint_ids[::2]
    right_ids = asset_cfg.joint_ids[1::2]
    left_yaw_roll = diff_pose[:, left_ids]
    right_yaw_roll = diff_pose[:, right_ids]
    yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
    yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
    return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(diff_pose, dim=1)
'''

def reward_yaw_rool_joint_pos(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """
    Calculates the reward for keeping joint positions close to default positions, with a focus
    on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    diff_full = asset.data.joint_pos - asset.data.default_joint_pos
    diff = diff_full[: , asset_cfg.joint_ids]

    left = diff[:, ::2]
    right = diff[:, 1::2]
    diff = torch.norm(left, dim=1) + torch.norm(right, dim=1)
    diff = torch.clamp(diff - 0.1, 0, 50)
    return torch.exp(-diff * 100) - 0.01 * torch.norm(diff_full, dim=1)
