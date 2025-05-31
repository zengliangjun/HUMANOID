
from dataclasses import MISSING
from collections.abc import Sequence
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands import commands_cfg

from . phase_command import PhaseCommand, RefPoseWithPhase
from .zero2small_command import ZeroSmallCommand

@configclass
class PhaseCommandCfg(CommandTermCfg):
    """Configuration for the null command generator."""
    class_type: type = PhaseCommand
    velocity_name: str = 'base_velocity'

    period: float = 0.8  # step cycle sec
    offset: float = 0.5  # left right leg phase offset


@configclass
class RefPoseWithPhaseCfg(PhaseCommandCfg):
    """Configuration for the null command generator."""
    class_type: type = RefPoseWithPhase

    asset_name: str = "robot"
    left_names: Sequence[str] = MISSING
    right_names: Sequence[str] = MISSING
    joint_scales: Sequence[float] = MISSING



@configclass
class ZeroSmallCommandCfg(commands_cfg.UniformVelocityCommandCfg):
    class_type: type = ZeroSmallCommand

    small2zero_threshold_line: float = 0.2
    small2zero_threshold_angle: float = 0.1
