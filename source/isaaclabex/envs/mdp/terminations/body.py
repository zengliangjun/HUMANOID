
from __future__ import annotations
from collections.abc import Sequence

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def bad_width(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = 'base_velocity',
    #target_width: float = 0.2,
    stand_ranges: Sequence[float] = [0.16, 0.24],
    line_ranges: Sequence[float] = [0.16, 0.24],
    noline_ranges: Sequence[float] = [0.1, 0.4],
) -> torch.Tensor:

    command_term = env.command_manager.get_term(command_name)
    standing = command_term.is_standing_env
    line = command_term.is_line_env
    noline = command_term.is_noline_env

    asset: Articulation = env.scene[asset_cfg.name]

    pos = asset.data.body_pos_w[:, asset_cfg.body_ids]

    quat_w = torch.repeat_interleave(asset.data.root_link_quat_w[:, None, :], pos.shape[1], dim=1)

    pos_b = math_utils.quat_rotate_inverse(quat_w, pos)

    error = (pos_b[:, 0::2, 1] - pos_b[:, 1::2, 1])
    stand_over = torch.sum(((error < stand_ranges[0]).float() + (error < stand_ranges[1]).float()), dim = -1) * standing.float()
    line_over =  torch.sum(((error < line_ranges[0]).float() + (error < line_ranges[1]).float()), dim = -1) * line.float()
    noline_over =  torch.sum(((error < noline_ranges[0]).float() + (error < noline_ranges[1]).float()), dim = -1) * noline.float()

    over = (stand_over + line_over + noline_over)
    return over > 0
