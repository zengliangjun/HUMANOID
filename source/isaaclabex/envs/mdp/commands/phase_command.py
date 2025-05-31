from __future__ import annotations
from collections.abc import Sequence

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import CommandTerm
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from . commands_cfg import PhaseCommandCfg, RefPoseWithPhaseCfg

class PhaseCommand(CommandTerm):
    cfg: PhaseCommandCfg

    def __init__(self, cfg: PhaseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.phase_command_b = torch.zeros(self.num_envs, 2, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "PhaseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.phase_command_b

    def _update_command(self):
        if not hasattr(self, 'velocity_command'):
            self.velocity_command = self._env.command_manager.get_term(self.cfg.velocity_name)


        self.phase[:] = (self._env.episode_length_buf * self._env.step_dt) % self.cfg.period / self.cfg.period
        self.phase_command_b[:, 0] = self.phase
        self.phase_command_b[:, 1] = (self.phase + self.cfg.offset) % 1

        if hasattr(self.velocity_command, 'is_standing_env'):
            standing_env_ids = self.velocity_command.is_standing_env.nonzero(as_tuple=False).flatten()
            self.phase[standing_env_ids] = 0
            self.phase_command_b[standing_env_ids, :] = 0.0

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        pass


class RefPoseWithPhase(PhaseCommand):

    cfg: RefPoseWithPhaseCfg

    def __init__(self, cfg: RefPoseWithPhaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.asset_name]
        self.left_ids, _ = asset.find_joints(cfg.left_names, preserve_order=True)
        self.right_ids, _ = asset.find_joints(cfg.right_names, preserve_order=True)
        self.joint_scales = torch.tensor(cfg.joint_scales, dtype=torch.float32, device=env.device)

        self.ref_dof_pos = torch.zeros_like(asset.data.joint_pos)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "RefPoseWithPhase:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    def _update_command(self):
        super()._update_command()

        sin_pos = torch.sin(2 * torch.pi * self.phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()

        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        sin_pos_r[sin_pos_r < 0] = 0

        self.ref_dof_pos[:, self.left_ids] = sin_pos_l[:, None] * self.joint_scales[None, :]
        self.ref_dof_pos[:, self.right_ids] = sin_pos_r[:, None] * self.joint_scales[None, :]
        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0
