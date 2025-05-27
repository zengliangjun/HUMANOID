
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import numpy as np
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import RigidObject
    from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

"""
Root state.
"""

def phase_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    commands = env.command_manager.get_command(command_name)
    sin_phase = torch.sin(2 * np.pi * commands[:, :1])
    cos_phase = torch.cos(2 * np.pi * commands[:, :1])
    return torch.cat((sin_phase, cos_phase), dim = -1)

"""
def gravity_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    "" "Gravity projection on the asset's root frame."" "
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.GRAVITY_VEC_W
"""