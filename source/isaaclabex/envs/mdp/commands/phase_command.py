from __future__ import annotations
from collections.abc import Sequence

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from . commands_cfg import PhaseCommandCfg

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
        self.phase[:] = (self._env.episode_length_buf * self._env.step_dt) % self.cfg.period / self.cfg.period
        self.phase_command_b[:, 0] = self.phase
        self.phase_command_b[:, 1] = (self.phase + self.cfg.offset) % 1

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        pass
