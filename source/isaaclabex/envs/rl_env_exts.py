# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.envs.common import VecEnvStepReturn
from isaaclabex.envs.rl_env_exts_cfg import ManagerBasedRLExtendsCfg
from isaaclabex.envs.managers.constraint_manager import ConstraintManager
from collections.abc import Sequence

class ManagerBasedRLEnv_Extends(ManagerBasedRLEnv):
    cfg : ManagerBasedRLExtendsCfg

    def __init__(self, cfg: ManagerBasedRLExtendsCfg, render_mode: str | None = None, **kwargs):
        super(ManagerBasedRLEnv_Extends, self).__init__(cfg=cfg)
        '''
        for reward penalty curriculum
        '''
        self.average_episode_length = torch.tensor(0, device=self.device, dtype=torch.long)
        self.max_iterations_steps = cfg.num_transitions_per_env * cfg.max_iterations

    def load_managers(self):
        super(ManagerBasedRLEnv_Extends, self).load_managers()
        self.termination_manager = ConstraintManager(self.cfg.terminations, self)


    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        super(ManagerBasedRLEnv_Extends, self).step(action)

        # positive reward
        if self.cfg.reward_positive_flag:
            torch.clamp_min_(self.reward_buf, 0)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids: Sequence[int]):
        '''
        for reward penalty curriculum
        averag episode length
        '''
        num = len(env_ids)
        current_average_episode_length = torch.mean(self.episode_length_buf[env_ids], dtype=torch.float)

        num_compute_average_epl = self.cfg.num_compute_average_epl
        self.average_episode_length = self.average_episode_length * (1 - num / num_compute_average_epl) \
                                     + current_average_episode_length * (num / num_compute_average_epl)

        # called super
        super()._reset_idx(env_ids)
