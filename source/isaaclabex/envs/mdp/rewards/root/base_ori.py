from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.utils import math
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reward_orientation_euler_gravity_b(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """
    Calculates the reward for maintaining a flat base orientation. It penalizes deviation
    from the desired base orientation using the base euler angles and the projected gravity vector.

    Args:
        env (ManagerBasedRLEnv): The environment instance, which contains the simulation scene
            and provides access to the assets and their states.
        asset_cfg (SceneEntityCfg): Configuration for the asset whose orientation is being evaluated.
            Defaults to a configuration with the name "robot".

    Returns:
        torch.Tensor: The calculated reward value, which is a combination of penalties for
        deviations in Euler angles and the projected gravity vector.
    """
    # Extract the asset from the environment using the provided configuration
    asset: Articulation = env.scene[asset_cfg.name]

    # Get the root quaternion of the asset
    quat = asset.data.root_quat_w

    # Convert the quaternion to Euler angles (roll, pitch, yaw)
    roll, pitch, yaw = math.euler_xyz_from_quat(quat)

    # Calculate mismatch based on Euler angles (penalizes deviation from flat orientation)
    euler_mismatch = torch.exp(-(torch.abs(roll) + torch.abs(pitch)) * 10)

    # Calculate mismatch based on the projected gravity vector (penalizes deviation from vertical alignment)
    gravity_mismatch = torch.exp(-torch.norm(asset.data.projected_gravity_b[:, :2], dim=1) * 20)

    # Combine the two mismatch values into a single reward (average of both components)
    return (euler_mismatch + gravity_mismatch) / 2.

