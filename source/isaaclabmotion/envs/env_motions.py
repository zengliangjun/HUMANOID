# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import gymnasium as gym
import torch
from collections.abc import Sequence
from typing import Any
from isaaclab.envs.common import VecEnvObs
from isaaclab.envs import ManagerBasedEnv
from isaaclabmotion.envs.managers import motions_manager

from . import env_motions_cfg

class ManagerMotionsEnv(ManagerBasedEnv):

    cfg: env_motions_cfg.ManagerMotionsCfg

    def __init__(self, cfg: env_motions_cfg.ManagerMotionsCfg):
        super(ManagerMotionsEnv, self).__init__(cfg)
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)


    def load_managers(self):
        super(ManagerMotionsEnv, self).load_managers()

        self.motions_manager = motions_manager.MotionsManager(self.cfg.motions, self)
        print("[INFO] Event Manager: ", self.event_manager)


    def reset(
        self, seed: int | None = None, env_ids: Sequence[int] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[VecEnvObs, dict]:

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        self.episode_length_buf[env_ids] = 0
        self.motions_manager.reset(env_ids)
        result = super(ManagerMotionsEnv, self).reset(seed, env_ids, options)
        return result

    def _super_step(self) -> tuple[VecEnvObs, dict]:
        #self.action_manager.process_action(action.to(self.device))

        #self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            # self.action_manager.apply_action()
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

        # post-step: step interval event
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # -- compute observations
        self.obs_buf = self.observation_manager.compute()
        self.recorder_manager.record_post_step()

        # return observations and extras
        return self.obs_buf, self.extras

    def step(self) -> tuple[VecEnvObs, dict]:
        self.common_step_counter += 1
        self.episode_length_buf += 1

        _reset = self.motions_manager.compute(isplay = True)
        #return super(ManagerMotionsEnv, self).step(action)
        self._super_step()
        return _reset
