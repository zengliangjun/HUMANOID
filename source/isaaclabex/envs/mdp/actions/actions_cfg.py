from isaaclab.envs.mdp.actions import actions_cfg as mdp_actions_cfg
from isaaclab.utils import configclass
from isaaclab.managers.action_manager import ActionTerm

from . import actions_withref


@configclass
class JPWithRefActionCfg(mdp_actions_cfg.JointPositionActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = actions_withref.JPWithRefAction

    refcommand_name: str = "phase_command"