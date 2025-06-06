from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

from .. import reward_collect, pbrs_base

class jacc_l2_pbrs(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sigma: float = 0.25,
        gamma: float = 1,
        method: int = pbrs_base.PBRSNormal
        ) -> torch.Tensor:

        _penalize = reward_collect.penalize_jacc_l2(env=env, asset_cfg=asset_cfg)
        return self._calculate(_penalize)


class jvel_l2_pbrs(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sigma: float = 0.25,
        gamma: float = 1,
        method: int = pbrs_base.PBRSNormal
        ) -> torch.Tensor:

        _penalize = reward_collect.penalize_jvel_l2(env=env, asset_cfg=asset_cfg)
        return self._calculate(_penalize)

class jpos_limits_l1_pbrs(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sigma: float = 0.25,
        gamma: float = 1,
        method: int = pbrs_base.PBRSNormal
        ) -> torch.Tensor:

        _penalize = reward_collect.penalize_jpos_limits_l1(env=env, asset_cfg=asset_cfg)
        return self._calculate(_penalize)

class jpos_deviation_l1_pbrs(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sigma: float = 0.25,
        gamma: float = 1,
        method: int = pbrs_base.PBRSNormal
        ) -> torch.Tensor:

        _penalize = reward_collect.penalize_jpos_deviation_l1(env=env, asset_cfg=asset_cfg)
        return self._calculate(_penalize)

class torques_l2_pbrs(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sigma: float = 0.25,
        gamma: float = 1,
        method: int = pbrs_base.PBRSNormal
        ) -> torch.Tensor:

        _penalize = reward_collect.penalize_torques_l2(env=env, asset_cfg=asset_cfg)
        return self._calculate(_penalize)


class total2zero_pbrs(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sigma: float = 0.25,
        gamma: float = 1,
        method: int = pbrs_base.PBRSNormal
        ) -> torch.Tensor:

        _penalize = reward_collect.reward_left_right_symmetry \
                (env=env, asset_cfg=asset_cfg, command_name = command_name, std = sigma)
        return self._calculate(_penalize)


class total2zero_one_way_pbrs(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sigma: float = 0.25,
        gamma: float = 1,
        method: int = pbrs_base.PBRSNormal
        ) -> torch.Tensor:

        _penalize = reward_collect.reward_hip_roll_symmetry \
                (env=env, asset_cfg=asset_cfg, command_name = command_name, std = sigma)
        return self._calculate(_penalize)


class equals_pbrs(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sigma: float = 0.25,
        gamma: float = 1,
        method: int = pbrs_base.PBRSNormal
        ) -> torch.Tensor:

        _penalize = reward_collect.reward_equals_symmetry \
                (env=env, asset_cfg=asset_cfg, command_name = command_name, std = sigma)
        return self._calculate(_penalize)

class meanvar_pbrs(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.term = reward_collect.reward_pose_mean_var_symmetry(cfg, env)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self.term.reset(env_ids)



    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        mean_std = 0.15,
        variance_std = 0.05,
        mean_weight = 1,
        variance_weight = 1,
        episode_length_threshold = 0,
        gamma: float = 1,
        method: int = pbrs_base.PBRSNormal
        ) -> torch.Tensor:

        _penalize = self.term \
                (env=env,
                asset_cfg=asset_cfg,
                mean_std = mean_std,
                variance_std = variance_std,
                mean_weight = mean_weight,
                variance_weight = variance_weight,
                episode_length_threshold = episode_length_threshold)

        return self._calculate(_penalize)

