# Module for calculating episode rewards based on joint status.
# This module computes rewards based on differences in joint positions and their statistical properties.
from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

# Class for episode reward calculation using joint differences.
class EpisodeStatus(ManagerTermBase):

    # Initialize with reward config and environment.
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        joint_size = len(cfg.params["asset_cfg"].joint_ids)
        self.jvariance_buf = torch.zeros((self.num_envs, joint_size), device=self.device, dtype=torch.float)
        self.jmean_buf = torch.zeros((self.num_envs, joint_size), device=self.device, dtype=torch.float)

    # Reset the mean and variance buffers for the specified environments.
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if len(env_ids) == 0:
            return
        self.jvariance_buf[env_ids] = 0
        self.jmean_buf[env_ids] = 0

    # Return current joint positions (no default offset subtracted).
    def _calcute_withzero(self, asset_cfg: SceneEntityCfg):
        asset: Articulation = self._env.scene[asset_cfg.name]
        jpos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        return jpos

    # Return joint differences where default positions are subtracted.
    def _calcute_withdefault(self, asset_cfg: SceneEntityCfg):
        asset: Articulation = self._env.scene[asset_cfg.name]
        jpos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        default_jpos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        return jpos - default_jpos

    # Update the running mean and variance using the new difference (diff).
    def _calcute_mean_variance(self, diff) -> None:
        # Calculate delta for updating the mean buffer.
        delta = diff - self.jmean_buf
        self.jmean_buf += delta / self._env.episode_length_buf[:, None]
        # Recalculate delta using the updated mean and update variance.
        delta2 = diff - self.jmean_buf
        self.jvariance_buf = (self.jvariance_buf * (self._env.episode_length_buf[:, None] - 2) + delta * delta2) / (self._env.episode_length_buf[:, None] - 1)
        # Reset buffers for environments with no or insufficient episodes.
        zero_flag = self._env.episode_length_buf == 0
        self.jmean_buf[zero_flag] = 0
        zero_flag = self._env.episode_length_buf <= 1
        self.jvariance_buf[zero_flag] = 0

    # Compute reward from normalized differences in means and variances.
    def _calcute_diff(self, start_ids: Sequence[int], end_ids: Sequence[int]) -> None:
        # Normalize differences and compute exponential rewards.
        diff_mean = torch.abs(self.jmean_buf[:, start_ids] - self.jmean_buf[:, end_ids]) / torch.norm(self.jmean_buf, dim=-1, keepdim=True)
        diff_variance = torch.abs(self.jvariance_buf[:, start_ids] - self.jvariance_buf[:, end_ids]) / torch.norm(self.jvariance_buf, dim=-1, keepdim=True)
        return torch.mean(torch.exp(-diff_mean), dim=-1) + torch.mean(torch.exp(-diff_variance), dim=-1)

    # Compute reward component based on overall mean and variance norms.
    def _calcute_meanmin_variancemax(self, variance_target) -> None:
        variance_target = torch.tensor(variance_target, device=self.device, dtype=torch.float)[None, :]

        mean = torch.norm(self.jmean_buf, dim=-1)
        variance = torch.norm((self.jvariance_buf - variance_target) / variance_target, dim=-1)
        return torch.exp(-mean) + torch.exp(-variance)

    # Main entry point to compute the episode reward.
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        start_ids: Sequence[int],
        end_ids: Sequence[int],
        variance_target,
        command_name,
        method: int = 0
    ) -> torch.Tensor:
        # Select the method for computing joint differences.
        if method == 0:
            diff = self._calcute_withdefault(asset_cfg)
        else:
            diff = self._calcute_withzero(asset_cfg)
        # Update running statistics.
        self._calcute_mean_variance(diff)
        # Compute reward components from differences.
        reward0 = self._calcute_diff(start_ids, end_ids)
        reward1 = self._calcute_meanmin_variancemax(variance_target)
        reward = (reward0 + reward1) / 4
        # Optimize by computing the command norm only once.
        command = env.command_manager.get_command(command_name)
        command_norm = torch.norm(command, dim=1)
        # Only consider rewards when command magnitude exceeds threshold.
        reward *= command_norm > 0.1
        return reward

class Episode2Zero(EpisodeStatus):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _calcute_diff(self, start_ids: Sequence[int], end_ids: Sequence[int]) -> None:
        # Normalize differences and compute exponential rewards.
        diff_mean = torch.abs(self.jmean_buf[:, start_ids] + self.jmean_buf[:, end_ids]) / torch.norm(self.jmean_buf, dim=-1, keepdim=True)
        diff_variance = torch.abs(self.jvariance_buf[:, start_ids] - self.jvariance_buf[:, end_ids]) / torch.norm(self.jvariance_buf, dim=-1, keepdim=True)
        return torch.mean(torch.exp(-diff_mean), dim=-1) + torch.mean(torch.exp(-diff_variance), dim=-1)




# Class for episode reward calculation using joint statistical measures.
class StepStatus(ManagerTermBase):

    # Initialize buffers for tracking mean and variance statistics.
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.mean_mean_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)
        self.mean_variance_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)
        self.variance_mean_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)
        self.variance_variance_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)

    # Reset the statistical buffers for the specified environments.
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if len(env_ids) == 0:
            return
        self.mean_mean_buf[env_ids] = 0
        self.mean_variance_buf[env_ids] = 0
        self.variance_mean_buf[env_ids] = 0
        self.variance_variance_buf[env_ids] = 0

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

    # Update statistical buffers (mean and variance) for joint differences.
    def _calcute_mean_variance(self, diff) -> None:
        # Compute variance and mean of 'diff' along joints.
        var, mean = torch.var_mean(diff, dim=1)
        # Update buffers for variance.
        self.variance_mean_buf += var / self._env.episode_length_buf
        delta_var = var - self.variance_mean_buf
        self.variance_variance_buf = (self.variance_variance_buf * (self._env.episode_length_buf - 2) + var * delta_var) / (self._env.episode_length_buf - 1)
        # Update buffers for mean.
        self.mean_mean_buf += mean / self._env.episode_length_buf
        delta_mean = mean - self.mean_mean_buf
        self.mean_variance_buf = (self.mean_variance_buf * (self._env.episode_length_buf - 2) + mean * delta_mean) / (self._env.episode_length_buf - 1)
        # Combine reset: For environments with episode_length <= 1, reset all statistical buffers.
        reset_flag = self._env.episode_length_buf == 0
        self.mean_mean_buf[reset_flag] = 0
        self.mean_variance_buf[reset_flag] = 0

        reset_flag = self._env.episode_length_buf <= 1

        self.variance_mean_buf[reset_flag] = 0
        self.variance_variance_buf[reset_flag] = 0

    # Compute reward based on the exponential decay of deviations.
    def _calcute_exp(self, variance_target):
        # Each term promotes low deviation from target statistics.
        reward0 = torch.exp(-torch.abs(self.mean_mean_buf))          # Reward for lower mean value.

        reward2 = torch.exp(-torch.abs(self.variance_mean_buf - self.mean_variance_buf) / self.mean_variance_buf)  # Reward for matching variance mean.
        reward3 = torch.exp(-torch.abs(self.variance_variance_buf / self.mean_variance_buf ** 2))  # Reward for lower dispersion.
        return (reward0 + reward2 + reward3) / 4

    # Main entry point to compute the episode reward using statistical measures.
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name,
        variance_target = 0,
        method: int = 0
    ) -> torch.Tensor:
        # Select the proper joint difference calculation method.
        if method == 0:
            diff = self._calcute_withdefault(asset_cfg)
        else:
            diff = self._calcute_withzero(asset_cfg)
        # Update statistical buffers with the current joint differences.
        self._calcute_mean_variance(diff)
        # Compute overall reward from the exponential scoring.
        reward = self._calcute_exp(variance_target)
        # Optimize by computing command norm only once.
        command = env.command_manager.get_command(command_name)
        command_norm = torch.norm(command, dim=1)
        # Filter out rewards for commands under the threshold.
        reward *= command_norm > 0.1
        return reward
