from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reward_distance(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
    max_threshold: float = 0.2,
    min_threshold: float = 0.3):
    """
    Calculates the reward based on the distance between the feet.
    Penalizes if the feet are too close or too far from each other.

    Parameters:
        env (ManagerBasedRLEnv): The environment containing scene and asset data.
        asset_cfg (SceneEntityCfg): Configuration for the scene entity including name and body identifiers.
        max_threshold (float): Upper distance threshold; distances above are penalized. Default is 0.2.
        min_threshold (float): Lower distance threshold; distances below are penalized. Default is 0.3.

    Returns:
        torch.Tensor: Computed reward tensor based on the feet distance.
    """
    # Retrieve the articulation asset from the scene using the asset configuration name.
    asset: Articulation = env.scene[asset_cfg.name]

    # Extract the 2D positions (x, y) of the bodies based on provided body_ids.
    pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]

    # Compute Euclidean distance between paired body positions (assumes feet are ordered alternately).
    pos_dist = torch.norm(pos[:, ::2] - pos[:, 1::2], dim=1)

    # Compute a penalty for distances below the minimum threshold.
    d_min = torch.clamp(pos_dist - min_threshold, -0.5, 0.)
    # Compute a penalty for distances above the maximum threshold.
    d_max = torch.clamp(pos_dist - max_threshold, 0, 0.5)

    # Calculate and return the final reward as the average penalty over the two conditions.
    return torch.sum((torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2, dim=-1)
