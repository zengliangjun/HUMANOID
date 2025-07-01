from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reward_feet_forces_z(env: ManagerBasedRLEnv,
   sensor_cfg: SceneEntityCfg, threshold: float = 500, max_forces: float = 400) -> torch.Tensor:
    """
    Rewards high vertical contact forces recorded by the feet sensor.

    Args:
        env (ManagerBasedRLEnv): The simulation environment instance.
        sensor_cfg (SceneEntityCfg): Sensor configuration including sensor name and body IDs.
        threshold (float): Force threshold above which reward begins.
        max_forces (float): Maximum force value to be considered for reward clipping.

    Returns:
        torch.Tensor: The reward value, computed as the maximum clamped vertical force across specified body parts.
    """
    # Retrieve the contact sensor instance using the sensor configuration.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Extract vertical (z-axis) force values for specified body parts.
    forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    # Compute reward: clamp the difference between forces and threshold.
    _reward = torch.clamp(forces_z - threshold, min=0, max=max_forces)
    # Return the maximum reward among all monitored body parts.
    _reward = torch.max(_reward, dim=1)[0]
    return _reward

def penalize_feet_forces(env: ManagerBasedRLEnv,
   sensor_cfg: SceneEntityCfg, threshold: float = 500, max_over_penalize_forces: float = 400) -> torch.Tensor:
    """
    Penalizes excessive contact forces on the feet to discourage high impact.

    Args:
        env (ManagerBasedRLEnv): The simulation environment instance.
        sensor_cfg (SceneEntityCfg): Sensor configuration including sensor name and body IDs.
        threshold (float): Force threshold above which penalties are applied.
        max_over_penalize_forces (float): Maximum force value to clip the penalty.

    Returns:
        torch.Tensor: The calculated penalty as the sum of clamped excessive forces from specified body parts.
    """
    # Retrieve the contact sensor instance.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Compute the norm of forces for specified body parts.
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    # Calculate penalty by clamping forces exceeding the threshold.
    _reward = torch.clamp(forces - threshold, min=0, max=max_over_penalize_forces)
    # Sum penalties across all relevant body parts.
    _reward = torch.sum(_reward, dim=-1)
    return _reward

def penalty_feet_stumble(env: ManagerBasedRLEnv,
   sensor_cfg: SceneEntityCfg):
    """
    Penalizes lateral forces on the feet that could indicate stumbling.

    Args:
        env (ManagerBasedRLEnv): The simulation environment instance.
        sensor_cfg (SceneEntityCfg): Sensor configuration including sensor name and body IDs.

    Returns:
        torch.Tensor: A binary tensor (float) where 1.0 indicates a stumbling event and 0.0 otherwise.
    """
    # Retrieve the contact sensor instance.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Get vertical forces (z-axis) for relevant body parts.
    forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    # Compute lateral forces from the x and y components.
    forces_xy = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize if any lateral force exceeds the vertical force; return as binary (float) indicator.
    return torch.any(forces_xy > forces_z, dim=1).float()

def penalty_feet_airborne(env: ManagerBasedRLEnv,
   sensor_cfg: SceneEntityCfg, threshold: float = 1e-3):
    """
    Penalizes the robot when it is airborne, i.e., when feet lose contact with the ground.

    Args:
        env (ManagerBasedRLEnv): The simulation environment instance.
        sensor_cfg (SceneEntityCfg): Sensor configuration including sensor name and body IDs.
        threshold (float): Contact time threshold below which a foot is considered airborne.

    Returns:
        torch.Tensor: A binary tensor (float) where 1.0 indicates the robot is airborne and 0.0 otherwise.

    Raises:
        RuntimeError: If the sensor's air time tracking is not activated.
    """
    # Retrieve the contact sensor instance.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Ensure that track_air_time is enabled in the sensor configuration.
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # Get the last contact times for monitored body parts.
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    # Determine if any contact time is below the threshold; indicates airborne condition.
    return torch.any(last_contact_time < threshold, dim=1).float()

# The reward and penalty functions include detailed parameter and inline documentation.


def penalize_both_feet_in_air(env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg, threshold: float = 1e-3) -> torch.Tensor:

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_in_air = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] <= 1.0

    return torch.all(feet_in_air, dim=1).float()
