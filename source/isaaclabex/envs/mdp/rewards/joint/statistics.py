# Module for calculating episode rewards based on joint status.
# This module computes rewards based on differences in joint positions and their statistical properties.
from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING, Optional
from abc import ABC, abstractmethod

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

import math
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

class BaseStatistics(ManagerTermBase, ABC):
    """Base class for joint statistics calculation with common functionality."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        joint_size = len(cfg.params["asset_cfg"].joint_ids)
        self._init_buffers(joint_size)

    def _init_buffers(self, joint_size: int):
        """Initialize statistical buffers."""
        # Basic statistics
        self.mean_buf = torch.zeros((self.num_envs, joint_size),
                                  device=self.device, dtype=torch.float)
        self.var_buf = torch.zeros_like(self.mean_buf)

        # Debug statistics (optional)
        self.max_buf = torch.zeros_like(self.mean_buf)
        self.min_buf = torch.zeros_like(self.mean_buf)

    @abstractmethod
    def _get_joint_data(self, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        """Abstract method to get joint data (position/velocity/acceleration)."""
        pass

    def _update_stats(self, data: torch.Tensor, debug: bool = False) -> None:
        """Update running statistics with new data.

        Args:
            data: New joint data tensor of shape (num_envs, joint_size)
            debug: Whether to update debug statistics
        """
        # Update mean and variance
        delta = data - self.mean_buf
        self.mean_buf += delta / self._env.episode_length_buf[:, None]

        delta2 = data - self.mean_buf
        self.var_buf = (self.var_buf * (self._env.episode_length_buf[:, None] - 2) +
                       delta * delta2) / (self._env.episode_length_buf[:, None] - 1)

        # Update debug stats if enabled
        if debug:
            self.max_buf = torch.maximum(self.max_buf, data)
            self.min_buf = torch.minimum(self.min_buf, data)

        # Reset buffers for new episodes
        self._reset_new_episodes()

    def _reset_new_episodes(self):
        """Reset buffers for environments with new episodes."""
        zero_flag = self._env.episode_length_buf == 0
        self.mean_buf[zero_flag] = 0

        zero_flag = self._env.episode_length_buf <= 1
        self.var_buf[zero_flag] = 0
        if hasattr(self, 'max_buf'):
            self.max_buf[zero_flag] = 0
            self.min_buf[zero_flag] = 0

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset buffers for specified environments."""
        if env_ids is None or len(env_ids) == 0:
            return
        self.mean_buf[env_ids] = 0
        self.var_buf[env_ids] = 0
        if hasattr(self, 'max_buf'):
            self.max_buf[env_ids] = 0
            self.min_buf[env_ids] = 0

class PositionStatistics(BaseStatistics):
    """Statistics for joint position data."""

    def _get_joint_data(self, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        asset: Articulation = self._env.scene[asset_cfg.name]
        return asset.data.joint_pos[:, asset_cfg.joint_ids]

    def _calculate_withzero(self, asset_cfg: SceneEntityCfg):
        """Get raw joint positions."""
        return self._get_joint_data(asset_cfg)

    def _calculate_withdefault(self, asset_cfg: SceneEntityCfg):
        """Get joint positions relative to default positions."""
        asset: Articulation = self._env.scene[asset_cfg.name]
        jpos = self._get_joint_data(asset_cfg)
        default_jpos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        return jpos - default_jpos

    def _calculate_diff(self, start_ids: Sequence[int], end_ids: Sequence[int]) -> torch.Tensor:
        diff_mean = torch.abs(self.mean_buf[:, start_ids] - self.mean_buf[:, end_ids])
        diff_variance = torch.abs(self.var_buf[:, start_ids] - self.var_buf[:, end_ids])
        return (torch.mean(torch.exp(-diff_mean), dim=-1) +
                torch.mean(torch.exp(-diff_variance), dim=-1)) / 2

    def _calculate_meanmin_variancemax(self, mean_std: float, variance_target: float) -> torch.Tensor:
        # mean_std is only used for scaling, not as target
        mean_std = max(mean_std, 1e-6)
        if variance_target != -1:
            variance_target = max(variance_target, 1e-6)

        # Reward mean_buf approaching 0
        mean = torch.norm(self.mean_buf, dim=-1) / math.sqrt(2)
        mean_reward = torch.exp(-mean / (mean_std + 1e-8))  # Higher reward when mean is closer to 0
        mean_reward = torch.clamp(mean_reward, min=0.01, max=1.0)

        if variance_target != -1:
            # Reward var_buf approaching variance_target
            variance = torch.norm(self.var_buf, dim=-1) / math.sqrt(2)
            # Use Gaussian-like reward centered at variance_target
            variance_diff = (variance - variance_target) / (variance_target + 1e-8)
            variance_reward = torch.exp(-torch.square(variance_diff))
            variance_reward = torch.clamp(variance_reward, min=0.01, max=1.0)
            return (variance_reward + mean_reward) / 2  # Equal weighting
        return mean_reward

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        start_ids: Sequence[int],
        end_ids: Sequence[int],
        mean_std,
        variance_target,
        command_name,
        method: int = 0
    ) -> torch.Tensor:
        if method == 0:
            diff = self._calculate_withdefault(asset_cfg)
        else:
            diff = self._calculate_withzero(asset_cfg)

        self._update_stats(diff)
        reward0 = self._calculate_diff(start_ids, end_ids)
        reward1 = self._calculate_meanmin_variancemax(mean_std, variance_target)
        reward = (reward0 + reward1 * 10) / 11

        command = env.command_manager.get_command(command_name)
        stand_flag = torch.norm(command, dim=1) < 0.1
        zero_flag = self._env.episode_length_buf <= 1

        flag = torch.logical_or(stand_flag, zero_flag)
        diff_reward = torch.exp(-torch.norm(diff, dim = -1))
        reward[flag] = diff_reward[flag]

        if torch.isnan(reward).sum() > 0:
            raise ValueError(f"NaN detected in reward calculation for envs: {torch.isnan(reward).sum()}.")
        if torch.isinf(reward).sum() > 0:
            raise ValueError(f"Inf detected in reward calculation for envs: {torch.isinf(reward).sum()}.")

        return reward

class Episode2Zero(PositionStatistics):
    """Variant that calculates differences differently."""

    def _calculate_diff(self, start_ids: Sequence[int], end_ids: Sequence[int]) -> torch.Tensor:
        diff_mean = torch.abs(self.mean_buf[:, start_ids] + self.mean_buf[:, end_ids])
        diff_variance = torch.abs(self.var_buf[:, start_ids] - self.var_buf[:, end_ids])
        return torch.mean(torch.exp(-diff_mean), dim=-1) + torch.mean(torch.exp(-diff_variance), dim=-1)

class VelocityStatistics(BaseStatistics):
    """Statistics for joint velocity data."""

    def _get_joint_data(self, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        asset: Articulation = self._env.scene[asset_cfg.name]
        return asset.data.joint_vel[:, asset_cfg.joint_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        max_speed: float = -1,
        debug: bool = False
    ) -> torch.Tensor:
        velocity = self._get_joint_data(asset_cfg)
        self._update_stats(velocity, debug)

        # Basic reward based on velocity variance
        reward = torch.mean(torch.exp(-torch.abs(self.var_buf)), dim=-1)

        # Penalize exceeding max speed if specified
        if max_speed > 0 and debug:
            speed_violation = (torch.abs(self.max_buf) > max_speed).any(dim=-1)
            reward[speed_violation] *= 0.5  # Reduce reward for violations

        command = env.command_manager.get_command(command_name)
        command_norm = torch.norm(command, dim=1)
        reward *= command_norm > 0.1

        zero_flag = self._env.episode_length_buf <= 1
        reward[zero_flag] = 0

        if torch.isnan(reward).sum() > 0 or torch.isinf(reward).sum() > 0:
            raise ValueError("Invalid reward values detected")

        return reward

class AccelerationStatistics(BaseStatistics):
    """Statistics for joint acceleration data."""

    def _get_joint_data(self, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        asset: Articulation = self._env.scene[asset_cfg.name]
        return asset.data.joint_acc[:, asset_cfg.joint_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        max_accel: float = -1,
        debug: bool = False
    ) -> torch.Tensor:
        acceleration = self._get_joint_data(asset_cfg)
        self._update_stats(acceleration, debug)

        # Basic reward based on acceleration variance
        reward = torch.mean(torch.exp(-torch.abs(self.var_buf)), dim=-1)

        # Penalize exceeding max acceleration if specified
        if max_accel > 0 and debug:
            accel_violation = (torch.abs(self.max_buf) > max_accel).any(dim=-1)
            reward[accel_violation] *= 0.5  # Reduce reward for violations

        command = env.command_manager.get_command(command_name)
        command_norm = torch.norm(command, dim=1)
        reward *= command_norm > 0.1

        zero_flag = self._env.episode_length_buf <= 1
        reward[zero_flag] = 0

        if torch.isnan(reward).sum() > 0 or torch.isinf(reward).sum() > 0:
            raise ValueError("Invalid reward values detected")

        return reward


# Class for episode reward calculation using joint statistical measures.
class BaseStepStats(ManagerTermBase):

    # Initialize buffers for tracking mean and variance statistics.
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        count = len(cfg.params["asset_cfg"].joint_names) // 2
        self.mean_mean_buf = torch.zeros((self.num_envs, count), device=self.device, dtype=torch.float)
        self.mean_variance_buf = torch.zeros((self.num_envs, count), device=self.device, dtype=torch.float)

        self.variance_mean_buf = torch.zeros((self.num_envs, count), device=self.device, dtype=torch.float)
        self.variance_variance_buf = torch.zeros((self.num_envs, count), device=self.device, dtype=torch.float)

    # Reset the statistical buffers for the specified environments.
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if len(env_ids) == 0:
            return
        self.mean_mean_buf[env_ids] = 0
        self.mean_variance_buf[env_ids] = 0
        self.variance_mean_buf[env_ids] = 0
        self.variance_variance_buf[env_ids] = 0


    # Update statistical buffers (mean and variance) for joint differences.
    def _calcute_mean_variance(self, diff) -> None:
        # Compute variance and mean of 'diff' along joints.
        diff0 = diff[:, :: 2, None]
        diff1 = diff[:, 1:: 2, None]
        diff = torch.cat((diff0, diff1), dim=-1)
        var, mean = torch.var_mean(diff, dim=-1)
        # Update buffers for variance.
        delta_var0 = var - self.variance_mean_buf
        self.variance_mean_buf += delta_var0 / self._env.episode_length_buf[:, None]
        delta_var1 = var - self.variance_mean_buf
        self.variance_variance_buf = (self.variance_variance_buf * (self._env.episode_length_buf[:, None] - 2) + delta_var0 * delta_var1) / (self._env.episode_length_buf[:, None] - 1)
        # Update buffers for mean.
        delta_mean0 = mean - self.mean_mean_buf
        self.mean_mean_buf += delta_mean0 / self._env.episode_length_buf[:, None]
        delta_mean1 = mean - self.mean_mean_buf
        self.mean_variance_buf = (self.mean_variance_buf * (self._env.episode_length_buf[:, None] - 2) + delta_mean0 * delta_mean1) / (self._env.episode_length_buf[:, None] - 1)
        # Combine reset: For environments with episode_length <= 1, reset all statistical buffers.

        reset_flag = self._env.episode_length_buf == 0
        self.mean_mean_buf[reset_flag] = 0
        self.variance_mean_buf[reset_flag] = 0

        reset_flag = self._env.episode_length_buf <= 1
        self.mean_variance_buf[reset_flag] = 0
        self.variance_variance_buf[reset_flag] = 0

        if torch.isnan(self.mean_mean_buf).sum() > 0:
            raise ValueError(f"NaN detected in reward calculation for envs: {torch.isnan(self.mean_mean_buf).sum()}. ")

        if torch.isnan(self.variance_mean_buf).sum() > 0:
            raise ValueError(f"NaN detected in reward calculation for envs: {torch.isnan(self.variance_mean_buf).sum()}. ")

        if torch.isnan(self.mean_variance_buf).sum() > 0:
            raise ValueError(f"NaN detected in reward calculation for envs: {torch.isnan(self.mean_variance_buf).sum()}. ")

        if torch.isnan(self.variance_variance_buf).sum() > 0:
            raise ValueError(f"NaN detected in reward calculation for envs: {torch.isnan(self.variance_variance_buf).sum()}. ")

        if torch.isinf(self.mean_mean_buf).sum() > 0:
            raise ValueError(f"NaN detected in reward calculation for envs: {torch.isinf(self.mean_mean_buf).sum()}. ")
        if torch.isinf(self.variance_mean_buf).sum() > 0:
            raise ValueError(f"NaN detected in reward calculation for envs: {torch.isinf(self.variance_mean_buf).sum()}. ")
        if torch.isinf(self.mean_variance_buf).sum() > 0:
            raise ValueError(f"NaN detected in reward calculation for envs: {torch.isinf(self.mean_variance_buf).sum()}. ")
        if torch.isinf(self.variance_variance_buf).sum() > 0:
            raise ValueError(f"NaN detected in reward calculation for envs: {torch.isinf(self.variance_variance_buf).sum()}. ")


    # Compute reward based on the exponential decay of deviations.
    def _calcute_exp(self, center):
        reward0 = torch.mean(torch.exp(-torch.abs(self.mean_variance_buf)), dim=-1)
        reward1 = torch.mean(torch.exp(-torch.abs(self.variance_variance_buf)), dim=-1)
        if center == 1:
            reward2 = torch.mean(torch.exp(-torch.abs(self.mean_mean_buf)), dim=-1)
            return (reward0 + reward1 + reward2) / 3
        else:
            return (reward0 + reward1) / 2

    # Main entry point to compute the episode reward using statistical measures.
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name,
        center: int = 0,  # 1 is mean_mean is to zero
        method: int = 0
    ) -> torch.Tensor:
        # Select the proper joint difference calculation method.
        diff = self._calcute(asset_cfg, method)
        # Update statistical buffers with the current joint differences.
        self._calcute_mean_variance(diff)
        # Compute overall reward from the exponential scoring.
        reward = self._calcute_exp(center)
        # Optimize by computing command norm only once.
        command = env.command_manager.get_command(command_name)
        stand_flag = torch.norm(command, dim=1) < 0.1
        # Filter out rewards for commands under the threshold.
        zero_flag = self._env.episode_length_buf <= 1

        flag = torch.logical_or(stand_flag, zero_flag)
        diff_reward = torch.exp(-torch.norm(diff, dim = -1))
        reward[flag] = diff_reward[flag]

        if torch.isnan(reward).sum() > 0:
            raise ValueError(f"NaN detected in reward calculation for envs: {torch.isnan(reward).sum()}. ")
        if torch.isinf(reward).sum() > 0:
            raise ValueError(f"NaN detected in reward calculation for envs: {torch.isinf(reward).sum()}. ")

        return reward

class StepPositionStats(BaseStepStats):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    # Return current joint positions (no default subtraction).
    def _calcute_withzero(self, asset_cfg: SceneEntityCfg):
        asset: Articulation = self._env.scene[asset_cfg.name]
        jpos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        return jpos

    # Return differences in joint positions (default positions subtracted).
    def _calcute_withdefault(self, asset_cfg: SceneEntityCfg):
        asset: Articulation = self._env.scene[asset_cfg.name]
        jpos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        default_jpos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        return jpos - default_jpos

    def _calcute(self, asset_cfg: SceneEntityCfg, method: int = 0):
        if method == 0:
            return self._calcute_withdefault(asset_cfg)
        else:
            return self._calcute_withzero(asset_cfg)


class StepVelocityStats(BaseStepStats):
    """Step-based statistics for joint velocities."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _calcute(self, asset_cfg: SceneEntityCfg, method: int = 0):
        asset: Articulation = self._env.scene[asset_cfg.name]
        return asset.data.joint_vel[:, asset_cfg.joint_ids]


class StepAccelerationStats(BaseStepStats):
    """Step-based statistics for joint accelerations."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _calcute(self, asset_cfg: SceneEntityCfg, method: int = 0):
        asset: Articulation = self._env.scene[asset_cfg.name]
        return asset.data.joint_acc[:, asset_cfg.joint_ids]
