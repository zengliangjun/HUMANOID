from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

from .. import reward_collect, pbrs_base

class slide_pbrs(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sigma: float = 0.25,
        gamma: float = 1,
        method: int = pbrs_base.PBRSNormal
        ) -> torch.Tensor:

        _penalize = reward_collect.penalize_slide(env=env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg)
        return self._calculate(_penalize)

class clearance_pbrs(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        target_height: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sigma: float = 0.25,
        gamma: float = 1,
        method: int = pbrs_base.PBRSNormal
        ) -> torch.Tensor:

        _penalize = reward_collect.penalize_clearance \
            (env=env, target_height=target_height, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg)
        return self._calculate(_penalize)

