from __future__ import annotations
import math
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

def reward_penalize_joint(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    diff_range = 0.1,
    diff_std = 0.2,
    penalize_weight = 0.5
    ):
    """
    Calculates the reward for keeping joint positions close to default positions, with a focus
    on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    diff_full = asset.data.joint_pos - asset.data.default_joint_pos
    diff_full = diff_full[: , asset_cfg.joint_ids]

    diff_full = torch.norm(diff_full, dim=1) / math.sqrt(len(asset_cfg.joint_ids))
    diff = torch.clamp(diff_full - diff_range, 0, 50)
    return torch.exp(- diff / diff_std) - penalize_weight * diff_full

