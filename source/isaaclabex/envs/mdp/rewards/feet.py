from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def penalize_foot_clearance(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float = 0.08) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]

    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    pos_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height) * ~ contacts
    return torch.sum(pos_error, dim=-1)

def penalize_feet_slide(env,
    sensor_cfg: SceneEntityCfg,
    contacts_threshold: float = 5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > contacts_threshold
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward