from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

def rew_contact_with_phase(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contacts_threshold: float = 1,
    command_name: str = 'phase',
    stance_phase = 0.55):

    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > contacts_threshold

    leg_phase = env.command_manager.get_command(command_name)
    is_stance = leg_phase < stance_phase

    reward = ~(contacts ^ is_stance)
    return torch.sum(reward, dim=-1)

def reward_feet_contact_number(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contacts_threshold: float = 5,
    phase_command_name: str = 'phase',
    stance_phase = 0.55):
    """
    Calculates a reward based on the number of feet contacts aligning with the gait phase.
    Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > contacts_threshold

    # Compute swing mask
    leg_phase = env.command_manager.get_command(phase_command_name)
    stance_mask = leg_phase < stance_phase

    reward = torch.where(contacts == stance_mask, 1.0, -0.3)
    return torch.mean(reward, dim=1)

class reward_feet_clearance(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)
        self.last_feet_z = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.last_feet_z[...] = cfg.params["feet_contact_height"]


    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if len(env_ids) == 0:
            return
        self.last_feet_z[env_ids] = self.cfg.params["feet_contact_height"]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        contacts_threshold: float = 5,
        phase_command_name: str = 'phase',
        feet_contact_height: float = 0.05,  # feet base height when feet contact on plane
        feet_target_height: float = 0.08,
        stance_phase = 0.55
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
        contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > contacts_threshold

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - feet_contact_height
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z[...] = feet_z

        # Compute swing mask
        leg_phase = env.command_manager.get_command(phase_command_name)
        swing_mask = leg_phase > stance_phase

        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - feet_target_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contacts
        return rew_pos

class reward_feet_air_time(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.last_contacts = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)
        self.feet_air_time = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if len(env_ids) == 0:
            return
        self.last_contacts[env_ids] = 0
        self.feet_air_time[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        contacts_threshold: float = 5,
        phase_command_name: str = 'phase',
        stance_phase = 0.55):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
        contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > contacts_threshold

        # Compute stance mask
        leg_phase = env.command_manager.get_command(phase_command_name)
        stance_mask = leg_phase < stance_phase

        contact_filt = torch.logical_or(torch.logical_or(contacts, stance_mask), self.last_contacts)
        self.last_contacts = contacts

        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self._env.step_dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~contact_filt
        return air_time.sum(dim=1)
