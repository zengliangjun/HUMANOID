
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from . phase_command import PhaseCommand

@configclass
class PhaseCommandCfg(CommandTermCfg):
    """Configuration for the null command generator."""
    class_type: type = PhaseCommand

    period: float = 0.8
    offset: float = 0.5
