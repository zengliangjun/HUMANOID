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
from collections.abc import Sequence

class ManagerBasedRLEnv_Extends(ManagerBasedRLEnv):
    cfg : ManagerBasedRLExtendsCfg

    def __init__(self, cfg: ManagerBasedRLExtendsCfg, render_mode: str | None = None, **kwargs):
        super(ManagerBasedRLEnv_Extends, self).__init__(cfg=cfg)
        '''
        for reward penalty curriculum
        '''
        self.average_episode_length = torch.tensor(0, device=self.device, dtype=torch.long)


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
