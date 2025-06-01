
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

from isaaclab.envs.mdp.actions import joint_actions
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from . import actions_cfg
    from ..commands.phase_command import RefPoseWithPhase


class JPWithRefAction(joint_actions.JointPositionAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.JPWithRefActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JPWithRefActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

    def process_actions(self, actions: torch.Tensor):
        _command: RefPoseWithPhase = self._env.command_manager.get_term(self.cfg.refcommand_name)
        actions = actions + _command.ref_dof_pos
        super().process_actions(actions)
