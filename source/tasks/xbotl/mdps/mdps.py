from isaaclab.utils import configclass
import math
from dataclasses import MISSING

from isaaclab.managers import TerminationTermCfg
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclabex.envs.mdp.commands import commands_cfg
from isaaclabex.envs.mdp.actions import actions_cfg

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = commands_cfg.ZeroSmallCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=commands_cfg.ZeroSmallCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.6), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.3, 0.3), heading=(-math.pi, math.pi)
        ),
    )

    phase_command = commands_cfg.RefPoseWithPhaseCfg(
        resampling_time_range=(10.0, 10.0),
        period = 0.64,

        left_names = ['left_leg_pitch_joint', 'left_knee_joint', 'left_ankle_pitch_joint'],
        right_names = ['right_leg_pitch_joint', 'right_knee_joint', 'right_ankle_pitch_joint'],
        joint_scales = [0.17, 0.34, 0.17]
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    #joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)
    joint_pos = actions_cfg.JPWithRefActionCfg(asset_name="robot",
                    joint_names=[".*"],
                    scale=0.25,
                    use_default_offset=True,
                    refcommand_name= "phase_command")

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    height = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.4})

    out_of_terrain = TerminationTermCfg(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )
    contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces",
                            body_names=['base_link']), "threshold": 1.0},
    )
