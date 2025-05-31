from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

def rew_joint_pos_withrefpose(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    phase_name: str,
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    pose = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pose = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    diff_pose = pose - default_pose

    phase_command = env.command_manager.get_term(phase_name)
    ref_dof_pos = phase_command.ref_dof_pos

    diff = diff_pose - ref_dof_pos
    r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
    return r
