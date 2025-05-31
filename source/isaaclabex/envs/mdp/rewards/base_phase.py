from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.utils import math

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reward_base_height(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        target_height: float = 0.8,
        phase_command_name: str = 'phase',
        stance_phase = 0.55,
        feet_contact_height: float = 0.05,  # feet base height when feet contact on plane
        ):
    """
    Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
    The reward is computed based on the height difference between the robot's base and the average height
    of its feet when they are in contact with the ground.
    """
    # Compute stance mask
    leg_phase = env.command_manager.get_command(phase_command_name)
    stance_mask = leg_phase < stance_phase

    asset: Articulation = env.scene[asset_cfg.name]

    feet_heights = torch.sum(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
    base_height = asset.data.root_pos_w[:, 2] - (feet_heights - feet_contact_height)

    return torch.exp(-torch.abs(base_height - target_height) * 100)
