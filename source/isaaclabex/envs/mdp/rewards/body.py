from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reward_distance(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
    max_threshold: float = 0.2,
    min_threshold: float = 0.3):
    """
    Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    pos_dist = torch.norm(pos[:, ::2] - pos[:, 1::2], dim=1)

    d_min = torch.clamp(pos_dist - min_threshold, -0.5, 0.)
    d_max = torch.clamp(pos_dist - max_threshold, 0, 0.5)
    return torch.sum((torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2, dim = -1)

def penalize_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg):
    """
    Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
    This encourages the robot to avoid undesired contact with objects or surfaces.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    contact = torch.norm(contact, dim=-1) > 0.1
    return torch.sum(contact, dim=1)
