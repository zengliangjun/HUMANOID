from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

PBRSNormal = 0
PBRSExp = 1
PBRSLite5Clamp0 = 2
PBRSNegativeScale = 3

class PbrsBase(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev_penalize = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        self.gamma = cfg.params.get("gamma", 1)
        self.sigma = cfg.params.get("sigma", 0.25)
        self.method = cfg.params.get("method", PBRSNormal)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Reset stored velocities for specified environments
        if len(env_ids) == 0:
            return
        self.prev_penalize[env_ids] = 0

    def _calculate(self, _penalize) -> torch.Tensor:
        if self.method == PBRSNormal:
            return self._calculate_normal(_penalize)
        elif self.method == PBRSExp:
            return self._calculate_exp(_penalize)
        elif self.method == PBRSLite5Clamp0:
            return self._calculate_reward_lite5clamp0(_penalize)
        elif self.method == PBRSNegativeScale:
            return self._calculate_reward_negative_scale(_penalize)
        else:
            raise ValueError(f"Unsupported method {self.method}")

    def _calculate_normal(self, _penalize) -> torch.Tensor:
        reward = ~self._env.reset_buf * (self.gamma * _penalize - self.prev_penalize) / self.sigma
        self.prev_penalize = _penalize
        return reward

    def _calculate_exp(self, _penalize) -> torch.Tensor:
        _penalize = torch.exp(-_penalize / self.sigma)
        reward = ~self._env.reset_buf * (self.gamma * _penalize - self.prev_penalize)
        self.prev_penalize = _penalize
        return reward

    def _calculate_reward_lite5clamp0(self, _penalize) -> torch.Tensor:
        reward = ~self._env.reset_buf * (self.gamma * _penalize - self.prev_penalize) / self.sigma
        self.prev_penalize = _penalize
        return torch.where(_penalize > 0.5, reward, torch.clamp_min(reward, 0))

    def _calculate_reward_negative_scale(self, _penalize) -> torch.Tensor:
        # NOTE _penalize calcute by negative exponential
        reward = ~self._env.reset_buf * (self.gamma * _penalize - self.prev_penalize) / self.sigma
        self.prev_penalize = _penalize
        negative_mask = reward < 0
        reward[negative_mask] *= _penalize[negative_mask]
        return reward
