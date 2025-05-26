from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def penalize_base_height(env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    feet_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    feet_height = torch.min(feet_heights, dim= -1)[0]
    body_height = torch.abs(asset.data.root_pos_w[:, 2] - feet_height)

    # Compute the L2 squared penalty
    return torch.square(body_height - target_height)
