from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.utils import math
from collections.abc import Sequence
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

def penalize_base_height(env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Penalizes deviation of the robot's base height from the target height using L2 squared error.

    Args:
        env (ManagerBasedRLEnv): Environment instance for simulation.
        target_height (float): Desired target height for the robot's base.
        asset_cfg (SceneEntityCfg): Configuration for the asset, defaulting to "robot".

    Returns:
        torch.Tensor: L2 squared penalty of the base height deviation.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get the individual feet heights and select the minimum (ground contact)
    feet_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    feet_height = torch.min(feet_heights, dim=-1)[0]
    # Compute body height as absolute difference between base height and lowest foot height
    body_height = torch.abs(asset.data.root_pos_w[:, 2] - feet_height)

    # Return L2 squared penalty based on the difference from target height
    return torch.square(body_height - target_height)

def reward_penalize_height_upper(env: ManagerBasedRLEnv,
    target_height: float,
    error_std: float = 0.025,
    penalize_weight: float = -3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Penalizes deviation of the robot's base height from the target height using L2 squared error.

    Args:
        env (ManagerBasedRLEnv): Environment instance for simulation.
        target_height (float): Desired target height for the robot's base.
        asset_cfg (SceneEntityCfg): Configuration for the asset, defaulting to "robot".

    Returns:
        torch.Tensor: L2 squared penalty of the base height deviation.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get the individual feet heights and select the minimum (ground contact)
    upper_height = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    # Compute body height as absolute difference between base height and lowest foot height
    height = torch.abs(asset.data.root_pos_w[:, 2] - upper_height)
    error = torch.abs(height - target_height)

    # Return L2 squared penalty based on the difference from target height
    return torch.exp(- error / error_std) + penalize_weight * (error / error_std)


def reward_mismatch_vel_exp(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    linear_weight = 10,
    angle_weight = 5):
    """
    Computes reward based on mismatches in linear and angular velocities.
    Uses exponential kernels to reward deviations from stability.

    Args:
        env (ManagerBasedRLEnv): Environment instance.
        asset_cfg (SceneEntityCfg): Asset configuration.
        linear_weight: Weight factor for linear velocity mismatch.
        angle_weight: Weight factor for angular velocity mismatch.

    Returns:
        torch.Tensor: Combined reward of velocity mismatches.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Exponential penalty on vertical linear velocity error
    lin_mismatch = torch.exp(-torch.square(asset.data.root_lin_vel_b[:, 2]) * linear_weight)
    # Exponential penalty on angular velocity error in x-y plane
    ang_mismatch = torch.exp(-torch.norm(asset.data.root_ang_vel_b[:, :2], dim=1) * angle_weight)
    # Combine both penalties
    return (lin_mismatch + ang_mismatch) / 2.


def reward_mismatch_speed(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = 'base_velocity',
    ):
    """
    Rewards or penalizes based on the robot's speed relative to the commanded speed.

    Args:
        env (ManagerBasedRLEnv): Environment instance.
        asset_cfg (SceneEntityCfg): Asset configuration.
        command_name (str): Name of the command containing desired base velocity.

    Returns:
        torch.Tensor: Speed reward where deviations and direction mismatches are penalized.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Compute absolute base linear speed in x-direction
    absolute_speed = torch.abs(asset.data.root_lin_vel_b[:, 0])
    # Get commanded velocity and compute its absolute value
    command = env.command_manager.get_command(command_name)
    absolute_command = torch.abs(command[:, 0])

    # Determine if current speed is too low, too high, or within desired range
    speed_too_low = absolute_speed < 0.5 * absolute_command
    speed_too_high = absolute_speed > 1.2 * absolute_command
    speed_desired = ~(speed_too_low | speed_too_high)

    # Detect if movement direction mismatches the command
    sign_mismatch = torch.sign(asset.data.root_lin_vel_b[:, 0]) != torch.sign(command[:, 0])

    reward = torch.zeros_like(absolute_speed)
    # Penalize low speed
    reward[speed_too_low] = -1.0
    # Neutral reward for too high speed
    reward[speed_too_high] = 0.
    # Positive reward for desired speed range
    reward[speed_desired] = 1.2
    # Highest penalty if direction is incorrect
    reward[sign_mismatch] = -2.0
    # Only apply reward if command is significant
    return reward * (torch.abs(command[:, 0]) > 0.1)


def reward_track_vel_hard(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = 'base_velocity',
    std: float = 0.5):
    """
    Calculates reward for accurately tracking linear (xy) and angular (yaw) velocity commands.

    Args:
        env (ManagerBasedRLEnv): Environment instance.
        asset_cfg (SceneEntityCfg): Asset configuration.
        command_name (str): Command name for base velocity.

    Returns:
        torch.Tensor: Reward computed from tracking errors in velocity commands.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Compute error for linear velocity on xy axes
    lin_vel_error = torch.norm(command[:, :2] - asset.data.root_lin_vel_b[:, :2], dim=1)
    lin_vel_error_exp = torch.exp(-lin_vel_error / std)

    # Compute error for angular velocity (yaw)
    ang_vel_error = torch.abs(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    ang_vel_error_exp = torch.exp(-ang_vel_error / std)

    # Apply extra penalty proportional to total linear error
    linear_error = 0.2 * (lin_vel_error + ang_vel_error)

    return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error


class reward_base_acc(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # Parameter docstrings and inline comments sufficiently explain the logic.
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.prev_root_lin_vel_b = torch.zeros_like(asset.data.root_lin_vel_b)
        self.prev_root_ang_vel_b = torch.zeros_like(asset.data.root_ang_vel_b)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Reset stored velocities for specified environments
        if len(env_ids) == 0:
            return
        self.prev_root_lin_vel_b[env_ids] = 0
        self.prev_root_ang_vel_b[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """
        Computes a reward based on penalizing high base accelerations.
        Encourages smoother motion by comparing current and previous velocities.

        Args:
            env (ManagerBasedRLEnv): The environment instance.
            asset_cfg (SceneEntityCfg): Asset configuration, default is "robot".

        Returns:
            torch.Tensor: Reward value computed from the base's acceleration.
        """
        asset: Articulation = env.scene[asset_cfg.name]

        # Compute difference between previous and current velocities (acceleration)
        root_acc = self.prev_root_lin_vel_b - asset.data.root_lin_vel_b
        ang_acc = self.prev_root_ang_vel_b - asset.data.root_ang_vel_b

        # Exponential penalty based on norm of acceleration (both linear and angular)
        rew = torch.exp(-(torch.norm(root_acc, dim=1) * 2 + torch.norm(ang_acc, dim=1)))

        # Update stored velocities for the next call
        self.prev_root_lin_vel_b[...] = asset.data.root_lin_vel_b
        self.prev_root_ang_vel_b[...] = asset.data.root_ang_vel_b

        return rew
