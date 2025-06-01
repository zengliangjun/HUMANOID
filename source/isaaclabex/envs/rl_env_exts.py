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
    """
    Extended reinforcement learning environment with additional manager functionalities,
    including termination constraints and reward penalty adjustments.

    Attributes:
        cfg (ManagerBasedRLExtendsCfg): Configuration for extended RL settings.
        average_episode_length (torch.Tensor): Running average of episode lengths.
        max_iterations_steps (int): Total allowed steps computed from configuration.
        termination_manager: Manager that handles termination constraints.
    """
    cfg : ManagerBasedRLExtendsCfg

    def __init__(self, cfg: ManagerBasedRLExtendsCfg, render_mode: str | None = None, **kwargs):
        """
        Initialize the extended RL environment.

        Parameters:
            cfg (ManagerBasedRLExtendsCfg): Environment configuration.
            render_mode (str | None): Render mode to be used, if any.
            **kwargs: Additional keyword arguments.
        """
        super(ManagerBasedRLEnv_Extends, self).__init__(cfg=cfg)
        '''
        for reward penalty curriculum
        '''
        # Initialize variables for reward penalty and curriculum
        self.average_episode_length = torch.tensor(0, device=self.device, dtype=torch.long)
        self.max_iterations_steps = cfg.num_steps_per_env * cfg.max_iterations

    def load_managers(self):
        """
        Load and initialize all necessary managers for the environment.

        This method first loads base managers, then initializes the termination manager
        using constraints provided in the configuration.
        """
        super(ManagerBasedRLEnv_Extends, self).load_managers()
        # Initialize termination manager with constraints from config
        self.termination_manager = ConstraintManager(self.cfg.terminations, self)


    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """
        Perform an environment step based on the given action.

        Parameters:
            action (torch.Tensor): Action to be executed.

        Returns:
            VecEnvStepReturn: A tuple containing observation buffer, reward buffer,
                              termination flags, timeout flags, and additional extras.
        """
        super(ManagerBasedRLEnv_Extends, self).step(action)

        # Apply reward constraint to ensure non-negative values if flag is set
        if self.cfg.reward_positive_flag:
            torch.clamp_min_(self.reward_buf, 0)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids: Sequence[int]):
        """
        Reset specific environments and update the average episode length.

        Parameters:
            env_ids (Sequence[int]): List of environment indices to reset.
        """
        num = len(env_ids)
        # Calculate the current average episode length for the selected environments
        current_average_episode_length = torch.mean(self.episode_length_buf[env_ids], dtype=torch.float)

        num_compute_average_epl = self.cfg.num_compute_average_epl
        # Update the running average using a weighted average formula
        self.average_episode_length = self.average_episode_length * (1 - num / num_compute_average_epl) \
                                     + current_average_episode_length * (num / num_compute_average_epl)

        # called super
        super()._reset_idx(env_ids)
