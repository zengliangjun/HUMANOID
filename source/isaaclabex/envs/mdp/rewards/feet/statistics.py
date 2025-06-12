# Module for calculating episode rewards based on joint status.
# This module computes rewards based on differences in joint positions and their statistical properties.
from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING, Optional
from abc import ABC, abstractmethod

from isaaclab.sensors import ContactSensor
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

class ContactStatistics(ManagerTermBase, ABC):
    """Base class for joint statistics calculation with common functionality."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg = cfg.params["sensor_cfg"]

        self._init_buffers()

        self.contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    def _init_buffers(self):
        """Initialize statistical buffers."""
        # Basic statistics
        self.forces_vect_mean_buf = torch.zeros((self.num_envs, 3),
                                  device=self.device, dtype=torch.float)
        self.forces_vect_var_buf = torch.zeros_like(self.forces_vect_mean_buf)

        self.forces_mean_buf = torch.zeros((self.num_envs),
                                  device=self.device, dtype=torch.float)
        self.forces_var_buf = torch.zeros_like(self.forces_mean_buf)

        self.vect_mean_buf = torch.zeros((self.num_envs, 3),
                                  device=self.device, dtype=torch.float)
        self.vect_var_buf = torch.zeros_like(self.vect_mean_buf)

    def _calculate_contact(self, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        forces = self.contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
        forces = torch.sum(forces, dim = -2)
        return forces

    def _update_stats(self, data: torch.Tensor, target: float = 500) -> None:
        """Update running statistics with new data.

        Args:
            data: New joint data tensor of shape (num_envs, joint_size)
            debug: Whether to update debug statistics
        """
        # Update mean and variance
        delta0 = data - self.forces_vect_mean_buf
        self.forces_vect_mean_buf += delta0 / self._env.episode_length_buf[:, None]

        delta1 = data - self.forces_vect_mean_buf
        self.forces_vect_var_buf = (self.forces_vect_var_buf * (self._env.episode_length_buf[:, None] - 2) +
                       delta0 * delta1) / (self._env.episode_length_buf[:, None] - 1)
        # Update debug stats if enabled
        norm = torch.linalg.norm(data, dim=-1)
        #vect = data / (norm.unsqueeze(-1) + 1e-6)

        norm_delta0 = norm - self.forces_mean_buf
        self.forces_mean_buf += norm_delta0 / self._env.episode_length_buf

        norm_delta1 = norm - self.forces_mean_buf
        self.forces_var_buf = (self.forces_var_buf * (self._env.episode_length_buf - 2) +
                       norm_delta0 * norm_delta1) / (self._env.episode_length_buf - 1)

        # Reset buffers for new episodes
        invalid_flag = torch.logical_or(norm < 1e-1, norm > target * 1.5)
        self._reset_new_episodes(invalid_flag)

    def _reset_new_episodes(self, invalid_flag):
        """Reset buffers for environments with new episodes."""
        zero_flag = torch.logical_or(self._env.episode_length_buf == 0, invalid_flag)
        self.forces_vect_mean_buf[zero_flag] = 0
        self.forces_mean_buf[zero_flag] = 0

        zero_flag = torch.logical_or(self._env.episode_length_buf <= 1, invalid_flag)
        self.forces_vect_var_buf[zero_flag] = 0
        self.forces_var_buf[zero_flag] = 0

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset buffers for specified environments."""
        if env_ids is None or len(env_ids) == 0:
            return
        self.forces_vect_mean_buf[env_ids] = 0
        self.forces_mean_buf[env_ids] = 0
        self.forces_vect_var_buf[env_ids] = 0
        self.forces_var_buf[env_ids] = 0

    def _calculate_meanmin_variancemin(self, target, std):
        # Reward mean_buf approaching 0
        diff = torch.abs(torch.norm(self.forces_vect_mean_buf, dim=-1) - target) / std
        mean_reward = torch.exp(-diff)  # Higher reward when mean is closer to 0

        var_buf = torch.norm(self.forces_var_buf, dim=-1) / std ** 2
        var_reward = torch.exp(-var_buf)

        norm = torch.abs(self.forces_mean_buf - target) / std
        norm_reward = torch.exp(-norm)
        return (mean_reward + var_reward + norm_reward) / 3.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        target: float = 500,
        std: float = 50,
    ) -> torch.Tensor:

        forces = self._calculate_contact(sensor_cfg)
        self._update_stats(forces, target)
        reward = self._calculate_meanmin_variancemin(target, std)

        zero_flag = self._env.episode_length_buf <= 1

        norm = torch.norm(forces, dim = -1)

        diff_reward = torch.exp(-torch.abs(norm - target) / target)
        reward[zero_flag] = diff_reward[zero_flag]
        invalid_flag = torch.logical_or(norm < 1e-1, norm > target * 1.5)
        reward[invalid_flag] = 0
        return reward
