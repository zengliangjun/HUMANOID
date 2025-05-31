from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reward_feet_forces_z(env: ManagerBasedRLEnv,
   sensor_cfg: SceneEntityCfg, threshold: float = 500, max_forces: float = 400) -> torch.Tensor:

    '''
    reward high contact forces.
    '''

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    _reward = torch.clamp(forces_z - threshold, min = 0, max = max_forces)
    _reward = torch.max(_reward, dim = 1)[0]
    return _reward

def penalize_feet_forces(env: ManagerBasedRLEnv,
   sensor_cfg: SceneEntityCfg, threshold: float = 500, max_over_penalize_forces: float = 400) -> torch.Tensor:

    """
    Calculates the reward for keeping contact forces within a specified range. Penalizes
    high contact forces on the feet.
    forces above threshold value are penalized
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    _reward = torch.clamp(forces - threshold, min = 0, max = max_over_penalize_forces)
    _reward = torch.sum(_reward, dim = -1)
    return _reward

def penalty_feet_stumble(env: ManagerBasedRLEnv,
   sensor_cfg: SceneEntityCfg):
    '''
    penalizes lateral forces that indicate stumbling
    '''
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    forces_xy = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    return torch.any(forces_xy > forces_z, dim=1).float()

def penalty_feet_airborne(env: ManagerBasedRLEnv,
   sensor_cfg: SceneEntityCfg, threshold: float = 1e-3):
    '''
    penalizes the robot when it is airborne.
    '''
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.any(last_contact_time < threshold, dim=1).float()

