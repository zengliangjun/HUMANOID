from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

from .. import reward_collect, pbrs_base


class episode_pbrs(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        start_ids: Sequence[int],
        end_ids: Sequence[int],
        mean_std,
        variance_target,
        command_name,
        calcute_method: int = 0,
        sigma: float = 0.25,
        gamma: float = 1,
        method: int = pbrs_base.PBRSNormal
        ) -> torch.Tensor:

        _penalize = reward_collect.reward_episode(
            env=env,
            asset_cfg=asset_cfg,
            start_ids=start_ids,
            end_ids=end_ids,
            mean_std=mean_std,
            variance_target=variance_target,
            command_name=command_name,
            method=calcute_method,
        )
        return self._calculate(_penalize)

class episode2zero_pbrs(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        start_ids: Sequence[int],
        end_ids: Sequence[int],
        mean_std,
        variance_target,
        command_name,
        calcute_method: int = 0,
        sigma: float = 0.25,
        gamma: float = 1,
        method: int = pbrs_base.PBRSNormal
        ) -> torch.Tensor:

        _penalize = reward_collect.reward_episode2zero(
            env=env,
            asset_cfg=asset_cfg,
            start_ids=start_ids,
            end_ids=end_ids,
            mean_std=mean_std,
            variance_target=variance_target,
            command_name=command_name,
            method=calcute_method,
        )
        return self._calculate(_penalize)

class step_pbrs(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name,
        calcute_method: int = 0,
        center: int = 0,  # 1 is mean_mean is to zero
        sigma: float = 0.25,
        gamma: float = 1,
        method: int = pbrs_base.PBRSNormal
        ) -> torch.Tensor:

        _penalize = reward_collect.reward_step(
            env=env,
            asset_cfg=asset_cfg,
            center=center,
            command_name=command_name,
            method=calcute_method,
        )
        return self._calculate(_penalize)
