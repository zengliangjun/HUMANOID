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

    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    feet_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    feet_height = torch.min(feet_heights, dim= -1)[0]
    body_height = torch.abs(asset.data.root_pos_w[:, 2] - feet_height)

    # Compute the L2 squared penalty
    return torch.square(body_height - target_height)


def reward_mismatch_vel_exp(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    linear_weight = 10,
    angle_weight = 5):
    """
    Computes a reward based on the mismatch in the robot's linear and angular velocities.
    Encourages the robot to maintain a stable velocity by penalizing large deviations.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    lin_mismatch = torch.exp(-torch.square(asset.data.root_lin_vel_b[:, 2]) * linear_weight)
    ang_mismatch = torch.exp(-torch.norm(asset.data.root_ang_vel_b[:, :2], dim=1) * angle_weight)
    c_update = (lin_mismatch + ang_mismatch) / 2.
    return c_update

def reward_mismatch_speed(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = 'base_velocity',
    ):
    """
    Rewards or penalizes the robot based on its speed relative to the commanded speed.
    This function checks if the robot is moving too slow, too fast, or at the desired speed,
    and if the movement direction matches the command.
    """
    # Calculate the absolute value of speed and command for comparison

    asset: Articulation = env.scene[asset_cfg.name]
    absolute_speed = torch.abs(asset.data.root_lin_vel_b[:, 0])

    command = env.command_manager.get_command(command_name)
    absolute_command = torch.abs(command[:, 0])

    # Define speed criteria for desired range
    speed_too_low = absolute_speed < 0.5 * absolute_command
    speed_too_high = absolute_speed > 1.2 * absolute_command
    speed_desired = ~(speed_too_low | speed_too_high)

    # Check if the speed and command directions are mismatched
    sign_mismatch = torch.sign(
        asset.data.root_lin_vel_b[:, 0]) != torch.sign(command[:, 0])

    # Initialize reward tensor
    reward = torch.zeros_like(absolute_speed)

    # Assign rewards based on conditions
    # Speed too low
    reward[speed_too_low] = -1.0
    # Speed too high
    reward[speed_too_high] = 0.
    # Speed within desired range
    reward[speed_desired] = 1.2
    # Sign mismatch has the highest priority
    reward[sign_mismatch] = -2.0
    return reward * (torch.abs(command[:, 0]) > 0.1)

def reward_track_vel_hard(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = 'base_velocity'):
    """
    Calculates a reward for accurately tracking both linear and angular velocity commands.
    Penalizes deviations from specified linear and angular velocity targets.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Tracking of linear velocity commands (xy axes)
    lin_vel_error = torch.norm(
        command[:, :2] - asset.data.root_lin_vel_b[:, :2], dim=1)
    lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

    # Tracking of angular velocity commands (yaw)
    ang_vel_error = torch.abs(
        command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

    linear_error = 0.2 * (lin_vel_error + ang_vel_error)

    return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error



class reward_base_acc(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.prev_root_lin_vel_b = torch.zeros_like(asset.data.root_lin_vel_b)
        self.prev_root_ang_vel_b = torch.zeros_like(asset.data.root_ang_vel_b)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if len(env_ids) == 0:
            return
        self.prev_root_lin_vel_b[env_ids] = 0
        self.prev_root_ang_vel_b[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        asset: Articulation = env.scene[asset_cfg.name]

        root_acc = self.prev_root_lin_vel_b - asset.data.root_lin_vel_b
        ang_acc = self.prev_root_ang_vel_b - asset.data.root_ang_vel_b

        rew = torch.exp(-(torch.norm(root_acc, dim=1) * 2 + torch.norm(ang_acc, dim=1)))

        self.prev_root_lin_vel_b[...] = asset.data.root_lin_vel_b
        self.prev_root_ang_vel_b[...] = asset.data.root_ang_vel_b

        return rew
