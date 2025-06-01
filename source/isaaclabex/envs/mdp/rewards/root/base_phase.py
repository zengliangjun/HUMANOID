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
        feet_contact_height: float = 0.05,  # Reference feet height when in contact with plane
        ):
    """
    Calculates the reward based on the robot's base height relative to the feet contact plane.
    Penalizes deviations from the desired target height.

    Args:
        env (ManagerBasedRLEnv): Environment instance with simulation scene and command manager.
        asset_cfg (SceneEntityCfg): Configuration for the asset. Default is "robot".
        target_height (float): Desired base height of the robot.
        phase_command_name (str): Command name for retrieving leg phase information.
        stance_phase: Threshold phase value to decide if a leg is in stance.
        feet_contact_height (float): Reference contact height for the feet with the ground.

    Returns:
        torch.Tensor: Reward value computed using an exponential penalty on height deviation.
    """
    # Retrieve leg phase from command manager
    leg_phase = env.command_manager.get_command(phase_command_name)
    # Determine legs in stance phase (phase value below threshold)
    stance_mask = leg_phase < stance_phase

    asset: Articulation = env.scene[asset_cfg.name]

    # Compute weighted average height of feet in stance phase
    feet_heights = torch.sum(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] * stance_mask, dim=1
    ) / torch.sum(stance_mask, dim=1)
    # Adjust base height by aligning it relative to feet contact height
    base_height = asset.data.root_pos_w[:, 2] - (feet_heights - feet_contact_height)

    # Return reward: higher deviation from target yields lower reward
    return torch.exp(-torch.abs(base_height - target_height) * 100)

# The function documentation and inline comments clearly describe the reward computation.
