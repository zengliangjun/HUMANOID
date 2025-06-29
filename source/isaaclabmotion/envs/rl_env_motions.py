from __future__ import annotations

import gymnasium as gym
import torch
from collections.abc import Sequence
from isaaclab.envs.common import VecEnvObs

from isaaclab.envs.common import VecEnvStepReturn

from isaaclabex.envs import rl_env_exts

from isaaclabmotion.envs.managers import motions_manager
from isaaclabmotion.envs import rl_env_motions_cfg


class RLMotionsENV(rl_env_exts.ManagerBasedRLEnv_Extends):
    """
    Extended reinforcement learning environment with additional manager functionalities,
    including termination constraints and reward penalty adjustments.

    Attributes:
        cfg (ManagerBasedRLExtendsCfg): Configuration for extended RL settings.
        average_episode_length (torch.Tensor): Running average of episode lengths.
        max_iterations_steps (int): Total allowed steps computed from configuration.
        termination_manager: Manager that handles termination constraints.
    """
    cfg : rl_env_motions_cfg.RLMotionsENVCfg

    def __init__(self, cfg: rl_env_motions_cfg.RLMotionsENVCfg, render_mode: str | None = None, **kwargs):
        super(RLMotionsENV, self).__init__(cfg=cfg, render_mode = render_mode, **kwargs)


    def load_managers(self):
        """
        Load and initialize all necessary managers for the environment.

        This method first loads base managers, then initializes the termination manager
        using constraints provided in the configuration.
        """
        self.motions_manager = motions_manager.MotionsManager(self.cfg.motions, self)
        print("[INFO] Event Manager: ", self.event_manager)

        super(RLMotionsENV, self).load_managers()

    def reset(
        self, seed: int | None = None, env_ids: Sequence[int] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[VecEnvObs, dict]:

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        self.episode_length_buf[env_ids] = 0
        self.motions_manager.reset(env_ids)
        result = super(RLMotionsENV, self).reset(seed, env_ids, options)
        return result

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

        # -- motions computation
        reset_time_outs = self.motions_manager.compute()
        self.reset_buf |= reset_time_outs
        self.reset_time_outs |= reset_time_outs

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
        # super(ManagerBasedRLEnv_Extends, self).step(action)
        self._super_step(action)

        # Apply reward constraint to ensure non-negative values if flag is set
        if self.cfg.reward_positive_flag:
            torch.clamp_min_(self.reward_buf, 0)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

        info = self.motions_manager.reset(env_ids)
        self.extras["log"].update(info)
