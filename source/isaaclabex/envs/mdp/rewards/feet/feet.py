from __future__ import annotations


import torch
from typing import TYPE_CHECKING
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject, Articulation
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def penalize_foot_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float = 0.08) -> torch.Tensor:
    """
    Rewards the swinging feet for clearing a specified height off the ground.

    Args:
        env (ManagerBasedRLEnv): The simulation environment instance.
        asset_cfg (SceneEntityCfg): Configuration for the asset; used to access the corresponding RigidObject.
        sensor_cfg (SceneEntityCfg): Configuration for the contact sensor, including sensor name and monitored body IDs.
        target_height (float): The desired clearance height for the feet.

    Returns:
        torch.Tensor: A penalty value computed as the squared error between the feet elevation and target height,
                      applied only when no contact is detected.
    """
    # Retrieve the asset object representing the feet.
    asset: RigidObject = env.scene[asset_cfg.name]
    # Retrieve the contact sensor instance using sensor configuration.
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    # Determine if there has been significant contact over history (force threshold > 1.0).
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # Compute squared error between actual height and the target height.
    # The error is only considered if no contact is detected (~contacts).
    pos_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height) * ~contacts
    # Sum error across the body parts.
    return torch.sum(pos_error, dim=-1)

def penalize_feet_slide(
    env,
    sensor_cfg: SceneEntityCfg,
    contacts_threshold: float = 5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Penalizes feet sliding on the ground by measuring the linear velocity of the feet
    when they are in contact with the ground.

    Args:
        env (ManagerBasedRLEnv): The simulation environment instance.
        sensor_cfg (SceneEntityCfg): Configuration for the contact sensor; provides sensor name and body IDs.
        contacts_threshold (float): Threshold for contact forces to determine ground contact.
        asset_cfg (SceneEntityCfg): Configuration for the asset, defaulting to "robot".

    Returns:
        torch.Tensor: A penalty value computed from the linear velocity (norm) of the feet while in contact.
    """
    # Retrieve the contact sensor instance.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Determine if the contact force exceeds the threshold over history.
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > contacts_threshold
    # Retrieve the asset instance.
    asset = env.scene[asset_cfg.name]
    # Extract linear velocity of the feet (only consider x and y components).
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # Compute norm of velocity for each foot and apply penalty only when contact is detected.
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def penalize_feet_orientation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """
    Computes the penalty on feet orientation to make no x and y projected gravity.

    This function is adapted from _reward_feet_ori in legged_gym.

    Returns:
        torch.Tensor: A float tensor of shape (num_envs) representing the computed penalty for each environment.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids]

    vec = asset.data.GRAVITY_VEC_W[:, None, :].repeat((1, quat.shape[1], 1))
    gravity = math_utils.quat_rotate_inverse(quat, vec)

    return torch.sum(torch.sum(torch.square(gravity[..., :2]), dim=-1) ** 0.5, dim=-1)
