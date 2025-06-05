from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

from .. import reward_collect, pbrs_base

class lin_xy_exp_pbrs(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float = 0.25,
        command_name: str = "base_velocity",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sigma: float = 0.1,
        gamma: float = 1,
        method: int = pbrs_base.PBRSLite5Clamp0
        ) -> torch.Tensor:

        _reward = reward_collect.reward_lin_xy_exp(
            env=env,
            std = std,
            command_name = command_name,
            asset_cfg=asset_cfg)
        return self._calculate(_reward)


class ang_z_exp_pbrs(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float = 0.25,
        command_name: str = "base_velocity",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sigma: float = 0.1,
        gamma: float = 1,
        method: int = pbrs_base.PBRSLite5Clamp0
        ) -> torch.Tensor:

        _reward = reward_collect.reward_ang_z_exp(
            env=env,
            std = std,
            command_name = command_name,
            asset_cfg=asset_cfg)
        return self._calculate(_reward)
