from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def penalize_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg):
    """
    Penalizes collisions detected by the contact sensor.

    Args:
        env (ManagerBasedRLEnv): Environment instance containing simulation scene and sensor data.
        sensor_cfg (SceneEntityCfg): Configuration for the sensor, including the sensor name and body part indices to monitor.

    Returns:
        torch.Tensor: A tensor representing the collision penalty, computed by summing contacts exceeding a force threshold.
    """
    # Retrieve the contact sensor instance using the provided sensor configuration
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Extract the net force values for the specified body parts from the sensor data
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    # Identify collisions where the force norm exceeds 0.1
    collision = torch.norm(forces, dim=-1) > 0.1
    # Sum the collision flags across the body parts to compute the overall penalty per environment sample
    return torch.sum(collision, dim=1)
# Inline and parameter annotations already improve code readability.
