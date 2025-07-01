from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm


def orientation_xywiththreshold(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    xy_threshold: float = 0.7,
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    abs_projected_gravity = torch.abs(asset.data.projected_gravity_b)

    return torch.any(
        torch.logical_or(
            abs_projected_gravity[:, 0].unsqueeze(1) > xy_threshold,
            abs_projected_gravity[:, 1].unsqueeze(1) > xy_threshold,
        ),
        dim=1,
        keepdim=True,
    )
