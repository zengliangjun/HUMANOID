from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

class PoseMeanVariance(ManagerTermBase):
    """
    Compute a reward based on the running mean and variance of joint differences.

    Parameters (in cfg.params):
        asset_cfg: SceneEntityCfg holding joint configuration and asset name.
        mean_std: Std. deviation for normalizing the mean difference.
        variance_std: Std. deviation for normalizing the variance difference.
        mean_weight: Weight multiplier for the mean reward component.
        variance_weight: Weight multiplier for the variance reward component.
        episode_length_weight: A threshold value; rewards are valid only if episode length is above this.
    """

    '''
    rew_pos_variance = RewardTermCfg(
            func=PoseMeanVariance,
            weight=1.0,
            params={"asset_cfg": SceneEntityCfg("robot",
                                joint_names=["left_hip_pitch_joint",
                                             "right_hip_pitch_joint",
                                             "left_knee_joint",
                                             "right_knee_joint" ]),
                    "weight": []},
        )
    '''

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Determine the number of joints from the asset configuration
        joint_size = len(cfg.params["asset_cfg"].joint_ids)

        # Initialize buffers for the running variance and mean for each joint.
        self.variance_buf = torch.zeros((self.num_envs, joint_size), device=self.device, dtype=torch.float)
        self.mean_buf = torch.zeros((self.num_envs, joint_size), device=self.device, dtype=torch.float)

        # Convert provided parameters to tensors for vectorized operations.
        self.mean_std = torch.tensor(cfg.params["mean_std"], device=self.device, dtype=torch.float)
        self.variance_std = torch.tensor(cfg.params["variance_std"], device=self.device, dtype=torch.float)
        self.mean_weight = torch.tensor(cfg.params["mean_weight"], device=self.device, dtype=torch.float)
        self.variance_weight = torch.tensor(cfg.params["variance_weight"], device=self.device, dtype=torch.float)

        # Establish the threshold for a valid episode length.
        self.episode_length_threshold = cfg.params["episode_length_threshold"]

        # Retrieve the articulation asset using its name.
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """
        Reset the mean and variance buffers for the specified environment IDs.

        Parameters:
            env_ids: Sequence of indices for the environments to reset.
        """
        if len(env_ids) == 0:
            return
        self.variance_buf[env_ids] = 0
        self.mean_buf[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        mean_std,
        variance_std,
        mean_weight,
        variance_weight,
        episode_length_threshold
    ) -> torch.Tensor:
        """
        Calculate the reward based on joint position differences.

        Parameters:
            asset_cfg: Joint configuration asset.
            env: The environment object containing simulation data.
            mean_std, variance_std: Normalization constants.
            mean_weight, variance_weight: Multiplicative factors for reward components.
            episode_length_threshold: Minimum episode length before rewarding.

        Returns:
            Tensor with computed rewards.
        """
        # Retrieve current and default joint positions using joint_ids.
        pose = self.asset.data.joint_pos[:, asset_cfg.joint_ids]
        default_pose = self.asset.data.default_joint_pos[:, asset_cfg.joint_ids]

        # Compute the difference between current pose and default pose.
        diff = pose - default_pose
        mean = self.mean_buf.clone()

        # Update mean using an incremental running average.
        self.mean_buf[...] = (mean * (self._env.episode_length_buf[:, None] - 1) + diff) / self._env.episode_length_buf[:, None]
        # Update variance using incremental variance formula.
        self.variance_buf[...] = self.variance_buf + (diff - mean) * (diff - self.mean_buf)

        # Reset buffers for environments where episode length is zero.
        zero_flag = self._env.episode_length_buf == 0
        self.mean_buf[zero_flag] = 0
        self.variance_buf[zero_flag] = 0

        # Compute differences between left-right joint pairs.
        diff_mean = self.mean_buf[:, ::2] - self.mean_buf[:, 1::2]
        diff_variance = self.variance_buf[:, ::2] - self.variance_buf[:, 1::2]

        # Exponential decay rewards for mean and variance differences.
        reward_mean = torch.mean(torch.exp(-torch.abs(diff_mean) / self.mean_std), dim=-1)
        reward_variance = torch.mean(torch.exp(-torch.abs(diff_variance) / self.variance_std), dim=-1)

        # Combine rewards using provided weights.
        reward = reward_mean * self.mean_weight + reward_variance * self.variance_weight

        # Invalidate rewards for episodes shorter than the threshold.
        novalid_flag = self._env.episode_length_buf < self.episode_length_threshold
        reward[novalid_flag] = 0
        return reward


class MeanMinVarianceMax(ManagerTermBase):
    """
    Compute a reward based on the running mean and variance of joint differences.

    Parameters (in cfg.params):
        asset_cfg: SceneEntityCfg holding joint configuration and asset name.
        mean_std: Std. deviation for normalizing the mean difference.
        variance_std: Std. deviation for normalizing the variance difference.
        mean_weight: Weight multiplier for the mean reward component.
        variance_weight: Weight multiplier for the variance reward component.
        episode_length_weight: A threshold value; rewards are valid only if episode length is above this.
    """

    '''
    rew_pos_variance = RewardTermCfg(
            func=PoseMeanVariance,
            weight=1.0,
            params={"asset_cfg": SceneEntityCfg("robot",
                                joint_names=["left_hip_pitch_joint",
                                             "right_hip_pitch_joint",
                                             "left_knee_joint",
                                             "right_knee_joint" ]),
                    "weight": []},
        )
    '''

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Determine the number of joints from the asset configuration
        joint_size = len(cfg.params["asset_cfg"].joint_ids)

        # Initialize buffers for the running variance and mean for each joint.
        self.mean_std = torch.tensor(cfg.params["mean_std"], device=self.device, dtype=torch.float)
        self.variance_std = torch.tensor(cfg.params["variance_std"], device=self.device, dtype=torch.float)
        self.variance_buf = torch.zeros((self.num_envs, joint_size), device=self.device, dtype=torch.float)
        self.mean_buf = torch.zeros((self.num_envs, joint_size), device=self.device, dtype=torch.float)

        # Convert provided parameters to tensors for vectorized operations.
        self.mean_weight = torch.tensor(cfg.params["mean_weight"], device=self.device, dtype=torch.float)
        self.variance_weight = torch.tensor(cfg.params["variance_weight"], device=self.device, dtype=torch.float)

        # Establish the threshold for a valid episode length.
        self.episode_length_threshold = cfg.params["episode_length_threshold"]

        # Retrieve the articulation asset using its name.
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """
        Reset the mean and variance buffers for the specified environment IDs.

        Parameters:
            env_ids: Sequence of indices for the environments to reset.
        """
        if len(env_ids) == 0:
            return
        self.variance_buf[env_ids] = 0
        self.mean_buf[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        mean_std,
        variance_std,
        mean_weight,
        variance_weight,
        episode_length_threshold
    ) -> torch.Tensor:
        """
        Calculate the reward based on joint position differences.

        Parameters:
            env: The environment object containing simulation data.
            asset_cfg: Joint configuration asset.
            mean_std, variance_std: Normalization constants.
            mean_weight, variance_weight: Multiplicative factors for reward components.
            episode_length_threshold: Minimum episode length before rewarding.

        Returns:
            Tensor with computed rewards.
        """
        # Retrieve current and default joint positions using joint_ids.
        pose = self.asset.data.joint_pos[:, asset_cfg.joint_ids]
        default_pose = self.asset.data.default_joint_pos[:, asset_cfg.joint_ids]

        # Compute the difference between current pose and default pose.
        diff = pose - default_pose
        mean = self.mean_buf.clone()

        # Update mean using an incremental running average.
        self.mean_buf[...] = (mean * (self._env.episode_length_buf[:, None] - 1) + diff) / self._env.episode_length_buf[:, None]
        # Update variance using incremental variance formula.
        self.variance_buf[...] = self.variance_buf + (diff - mean) * (diff - self.mean_buf)

        # Reset buffers for environments where episode length is zero.
        zero_flag = self._env.episode_length_buf == 0
        self.mean_buf[zero_flag] = 0
        self.variance_buf[zero_flag] = 0

        # Exponential decay rewards for mean and variance differences.
        reward_mean = torch.mean(torch.exp(-torch.abs(self.mean_buf) / self.mean_std), dim=-1)
        reward_variance = torch.clamp_max(torch.sum(torch.abs(self.variance_buf), dim=-1), self.variance_std)

        # Combine rewards using provided weights.
        reward = reward_mean * self.mean_weight + reward_variance * self.variance_weight

        # Invalidate rewards for episodes shorter than the threshold.
        novalid_flag = self._env.episode_length_buf < self.episode_length_threshold
        reward[novalid_flag] = 0
        return reward



'''
rew_hip_knee_pitch_total2zero = RewardTermCfg(
        func=rew_hip_knee_pitch_total2zero,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot",
                            joint_names=["left_hip_pitch_joint",
                                        "left_knee_joint",
                                        "right_hip_pitch_joint",
                                        "right_knee_joint"
                                        ]),
                "weight": [1, 0.5]},
    )
'''

# Function to compute a reward that drives hip and knee joint differences toward zero.
def rew_hip_knee_pitch_total2zero(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    std: float,
    weight: Sequence[float] | list[float] | torch.Tensor = [1, 1]
) -> torch.Tensor:
    """
    Compute a reward based on a weighted absolute difference between hip and knee joint positions.

    Parameters:
        env: The environment instance.
        asset_cfg: Joint configuration asset.
        command_name: The name of the command influencing the reward.
        weight: A list-like object with weighting factors for left and right joint pairs.

    Returns:
        Tensor containing the reward values.
    """
    # Retrieve the articulation asset.
    asset: Articulation = env.scene[asset_cfg.name]
    # Extract current and default joint positions.
    pose = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pose = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    # Calculate the joint differences.
    diff = pose - default_pose

    # Compute weighted sum of differences for left/right joint pairs.
    total = diff[:, ::2] * weight[0] + diff[:, 1::2] * weight[1]
    total = torch.mean(torch.square(total), dim=-1)
    reward = torch.exp(-total / std**2)
    # Only award reward if the command vector norm is above 0.1.
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

'''
rew_left_right_total2zero = RewardTermCfg(
        func=rew_left_right_total2zero,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot",
                            joint_names=["left_hip_pitch_joint",
                                        "right_hip_pitch_joint",
                                        "left_knee_joint",
                                        "right_knee_joint"
                                        ])},
    )
'''

# Wrapper function for computing reward with equal weight for left/right joints.
def rew_left_right_total2zero(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    std: float
) -> torch.Tensor:
    """
    Calculate a reward for joint symmetry using equal weights for left and right joints.

    Parameters:
        env: The environment instance.
        asset_cfg: Joint configuration asset.
        command_name: The name of the command influencing the reward.

    Returns:
        Tensor with the computed reward.
    """
    return rew_hip_knee_pitch_total2zero(env, asset_cfg, command_name, std, [1, 1])

# Function to compute a reward that drives hip and knee joint differences toward zero.
def rew_hip_roll_total2zero(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    std: float
) -> torch.Tensor:
    """
    Compute a reward based on a weighted absolute difference between hip and knee joint positions.

    Parameters:
        env: The environment instance.
        asset_cfg: Joint configuration asset.
        command_name: The name of the command influencing the reward.
        weight: A list-like object with weighting factors for left and right joint pairs.

    Returns:
        Tensor containing the reward values.
    """
    # Retrieve the articulation asset.
    asset: Articulation = env.scene[asset_cfg.name]
    # Extract current and default joint positions.
    pose = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pose = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    # Calculate the joint differences.
    diff = pose - default_pose

    # Compute weighted sum of differences for left/right joint pairs.
    total = diff[:, ::2] + diff[:, 1::2]
    total = torch.mean(torch.square(total), dim=-1)
    reward = torch.exp(-total / std**2) * 0.25

    _negative = torch.clamp_max(diff[:, ::2], max = 0)
    _positive = torch.clamp_min(diff[:, 1::2], min = 0)
    _error = torch.mean((torch.square(_negative) + torch.square(_positive)), dim=-1) # to 0
    reward += torch.exp(-_error / std**2)

    # Only award reward if the command vector norm is above 0.1.
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def reward_equals_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    std: float,
    weight: Sequence[float] | list[float] | torch.Tensor = [1, 1]
) -> torch.Tensor:
    """
    Compute a reward based on a weighted absolute difference between hip and knee joint positions.

    Parameters:
        env: The environment instance.
        asset_cfg: Joint configuration asset.
        command_name: The name of the command influencing the reward.
        weight: A list-like object with weighting factors for left and right joint pairs.

    Returns:
        Tensor containing the reward values.
    """
    # Retrieve the articulation asset.
    asset: Articulation = env.scene[asset_cfg.name]
    # Extract current and default joint positions.
    pose = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pose = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    # Calculate the joint differences.
    diff = pose - default_pose

    # Compute weighted sum of differences for left/right joint pairs.
    total = diff[:, ::2] * weight[0] - diff[:, 1::2] * weight[1]
    total = torch.mean(torch.square(total), dim=-1)
    reward = torch.exp(-total / std**2)
    # Only award reward if the command vector norm is above 0.1.
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward
