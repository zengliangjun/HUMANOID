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
    def _calculate_withzero(self, asset_cfg: SceneEntityCfg):
        asset: Articulation = self._env.scene[asset_cfg.name]
        jpos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        return jpos

    # Return joint differences where default positions are subtracted.
    def _calculate_withdefault(self, asset_cfg: SceneEntityCfg):
        asset: Articulation = self._env.scene[asset_cfg.name]
        jpos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        default_jpos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        return jpos - default_jpos

    # Update the running mean and variance using the new difference (diff).
    def _calculate_mean_variance(self, diff) -> None:
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
    def _calculate_diff(self, start_ids: Sequence[int], end_ids: Sequence[int]) -> torch.Tensor:
        """计算基于关节平均值和方差差异的奖励分量

        参数:
            start_ids: 起始关节索引序列
            end_ids: 结束关节索引序列

        返回:
            torch.Tensor: 计算得到的奖励分量
        """
        diff_mean = torch.abs(self.jmean_buf[:, start_ids] - self.jmean_buf[:, end_ids])
        diff_variance = torch.abs(self.jvariance_buf[:, start_ids] - self.jvariance_buf[:, end_ids])
        return (torch.mean(torch.exp(-diff_mean), dim=-1) + torch.mean(torch.exp(-diff_variance), dim=-1)) / 2

    # Compute reward component based on overall mean and variance norms.
    def _calculate_meanmin_variancemax(self, mean_std: float, variance_target: float) -> torch.Tensor:
        """计算基于关节平均值最小化和方差最大化的奖励分量

        使用sigmoid函数替代指数衰减，提供更好的梯度信号和数值稳定性

        参数:
            mean_std (float): 控制均值奖励过渡区的参数，必须为正
            variance_target (float): 目标方差值，-1表示不考虑方差奖励

        返回:
            torch.Tensor: 形状为(num_envs,)的奖励张量
        """
        # 参数校验
        mean_std = max(mean_std, 1e-6)
        if variance_target != -1:
            variance_target = max(variance_target, 1e-6)

        # 计算均值奖励 (sigmoid形式)
        mean = torch.mean(torch.abs(self.jmean_buf), dim=-1)
        mean = torch.clamp(mean, max=3*mean_std)  # 限制最大值
        safe_exp_input = (mean_std - mean) / (mean_std + 1e-8)
        safe_exp_input = torch.clamp(safe_exp_input, min=-10, max=10)
        mean_reward = 1 / (1 + torch.exp(-safe_exp_input))
        mean_reward = torch.clamp(mean_reward, min=0.01)  # 最小奖励

        if variance_target != -1:
            # 计算方差奖励 (sigmoid形式)
            variance = torch.mean(torch.abs(self.jvariance_buf), dim=-1)
            ratio = variance / (variance_target + 1e-8)
            ratio = torch.clamp(ratio, 0.1, 10)  # 限制比率范围
            variance_reward = 2 / (1 + torch.exp(5 * (ratio - 1))) - 1
            variance_reward = torch.clamp(variance_reward, min=0.01, max=1.0)

            return (variance_reward + mean_reward) / 4
        return mean_reward

    # Main entry point to compute the episode reward.
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
        # Select the method for computing joint differences.
        if method == 0:
            diff = self._calculate_withdefault(asset_cfg)
        else:
            diff = self._calculate_withzero(asset_cfg)
        # Update running statistics.
        self._calculate_mean_variance(diff)
        # Compute reward components from differences.
        reward0 = self._calculate_diff(start_ids, end_ids)
        reward1 = self._calculate_meanmin_variancemax(mean_std, variance_target)
        reward = (reward0 + reward1 * 3) / 4
        # Optimize by computing the command norm only once.
        command = env.command_manager.get_command(command_name)
        command_norm = torch.norm(command, dim=1)
        # Only consider rewards when command magnitude exceeds threshold.
        reward *= command_norm > 0.1
        zero_flag = self._env.episode_length_buf <= 1
        reward[zero_flag] = 0

        if torch.isnan(reward).sum() > 0:
            raise ValueError(f"NaN detected in reward calculation for envs: {torch.isnan(reward).sum()}. ")

        if torch.isinf(reward).sum() > 0:
            raise ValueError(f"NaN detected in reward calculation for envs: {torch.isinf(reward).sum()}. ")

        return reward

class Episode2Zero(EpisodeStatus):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _calculate_diff(self, start_ids: Sequence[int], end_ids: Sequence[int]) -> torch.Tensor:
        """计算基于关节平均值和方差差异的奖励分量(Episode2Zero专用实现)

        参数:
            start_ids: 起始关节索引序列
            end_ids: 结束关节索引序列

        返回:
            torch.Tensor: 计算得到的奖励分量
        """
        diff_mean = torch.abs(self.jmean_buf[:, start_ids] + self.jmean_buf[:, end_ids])
        diff_variance = torch.abs(self.jvariance_buf[:, start_ids] - self.jvariance_buf[:, end_ids])
        return torch.mean(torch.exp(-diff_mean), dim=-1) + torch.mean(torch.exp(-diff_variance), dim=-1)

# Class for episode reward calculation using joint statistical measures.
class StepStatus(ManagerTermBase):

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
        if method == 0:
            diff = self._calcute_withdefault(asset_cfg)
        else:
            diff = self._calcute_withzero(asset_cfg)
        # Update statistical buffers with the current joint differences.
        self._calcute_mean_variance(diff)
        # Compute overall reward from the exponential scoring.
        reward = self._calcute_exp(center)
        # Optimize by computing command norm only once.
        command = env.command_manager.get_command(command_name)
        command_norm = torch.norm(command, dim=1)
        # Filter out rewards for commands under the threshold.
        reward *= command_norm > 0.1

        zero_flag = self._env.episode_length_buf <= 1
        reward[zero_flag] = 0
        if torch.isnan(reward).sum() > 0:
            raise ValueError(f"NaN detected in reward calculation for envs: {torch.isnan(reward).sum()}. ")
        if torch.isinf(reward).sum() > 0:
            raise ValueError(f"NaN detected in reward calculation for envs: {torch.isinf(reward).sum()}. ")

        return reward
