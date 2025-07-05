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
from isaaclabex.envs.managers import constraint_manager, statistics_manager
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
    average_episode_length: torch.Tensor
    max_iterations_steps: int

    recovery_counters: torch.Tensor

    def __init__(self, cfg: ManagerBasedRLExtendsCfg, render_mode: str | None = None, **kwargs):
        """
        Initialize the extended RL environment.

        Parameters:
            cfg (ManagerBasedRLExtendsCfg): Environment configuration.
            render_mode (str | None): Render mode to be used, if any.
            **kwargs: Additional keyword arguments.
        """
        super(ManagerBasedRLEnv_Extends, self).__init__(cfg=cfg, render_mode = render_mode, **kwargs)
        '''
        for reward penalty curriculum
        '''
        # Initialize variables for reward penalty and curriculum
        self.average_episode_length = torch.tensor(0, device=self.device, dtype=torch.long)
        self.max_iterations_steps = cfg.num_steps_per_env * cfg.max_iterations

        self.recovery_counters = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def load_managers(self):
        """
        Load and initialize all necessary managers for the environment.

        This method first loads base managers, then initializes the termination manager
        using constraints provided in the configuration.
        """

        self.statistics_manager = statistics_manager.StatisticsManager(self.cfg.statistics, self)
        print("[INFO] Statistic Manager: ", self.statistics_manager)

        super(ManagerBasedRLEnv_Extends, self).load_managers()
        # Initialize termination manager with constraints from config
        self.termination_manager = constraint_manager.ConstraintManager(self.cfg.terminations, self)
        print("[INFO] Constraint Manager: ", self.termination_manager)

    def _super_step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        # -- statistics computation
        self.statistics_manager.compute()
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """
        Perform an environment step based on the given action.

        Parameters:
            action (torch.Tensor): Action to be executed.

        Returns:
            VecEnvStepReturn: A tuple containing observation buffer, reward buffer,
                              termination flags, timeout flags, and additional extras.
        """
        # Update recovery counters
        self.recovery_counters -= 1
        torch.clamp_min_(self.recovery_counters, 0)

        # super(ManagerBasedRLEnv_Extends, self).step(action)

        self._super_step(action)

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

        self.recovery_counters[env_ids] = 0

        # called super

        super()._reset_idx(env_ids)
        info = {"Train/average_episode_length": self.average_episode_length}
        self.extras["log"].update(info)

        # -- statistics manager
        info = self.statistics_manager.reset(env_ids)
        self.extras["log"].update(info)

