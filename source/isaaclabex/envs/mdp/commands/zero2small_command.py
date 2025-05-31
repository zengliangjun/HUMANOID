from __future__ import annotations
from collections.abc import Sequence

import torch
from typing import TYPE_CHECKING
from isaaclab.envs.mdp.commands import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from . commands_cfg import ZeroSmallCommandCfg


class ZeroSmallCommand(UniformVelocityCommand):
    cfg: ZeroSmallCommandCfg

    def __init__(self, cfg: ZeroSmallCommandCfg, env: ManagerBasedEnv):
        super(ZeroSmallCommand, self).__init__(cfg, env)

    def _resample_command(self, env_ids: Sequence[int]):
        super(ZeroSmallCommand, self)._resample_command(env_ids)

        zero_line_flags = (torch.norm(self.vel_command_b[env_ids, :2], dim=1) < self.cfg.small2zero_threshold_line)
        zero_angle_flags = torch.abs(self.vel_command_b[env_ids, 2]) < self.cfg.small2zero_threshold_angle

        zero_flags = torch.logical_and(zero_line_flags, zero_angle_flags)

        zero_line_ids = env_ids[zero_line_flags]
        zero_angle_ids = env_ids[zero_angle_flags]
        zero_ids = env_ids[zero_flags]
        # set small commands to zero
        self.vel_command_b[zero_line_ids, :2] *= 0
        self.vel_command_b[zero_angle_ids, 2] *= 0
        self.is_standing_env[zero_ids] = True

