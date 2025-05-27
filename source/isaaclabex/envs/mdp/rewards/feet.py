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

def rew_contact_with_phase(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str):

    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

    leg_phase = env.command_manager.get_command(command_name)
    is_stance = leg_phase < 0.55

    reward = ~(contacts ^ is_stance)
    return torch.sum(reward, dim=-1)
