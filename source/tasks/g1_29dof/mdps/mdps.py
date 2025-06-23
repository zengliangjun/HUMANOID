from isaaclab.utils import configclass
import math
from dataclasses import MISSING

from isaaclab.managers import TerminationTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclabex.envs.managers import term_cfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclabex.envs.mdp.commands import commands_cfg
from isaaclabex.envs.mdp.statistics import joints

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = commands_cfg.ZeroSmallCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=commands_cfg.ZeroSmallCommandCfg.Ranges(
            lin_vel_x=(0, 2.8), lin_vel_y=(-0.35, 0.35), ang_vel_z=(-2., 2.), heading=(0., 0)
        ),
        small2zero_threshold_line=0.25,
        small2zero_threshold_angle=0.25
    )

    def __post_init__(self):
        self.base_velocity.goal_vel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)
        self.base_velocity.current_vel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

    orientation = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 3.14 * 45 / 180})

    height = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.4})

    contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces",
                            body_names=[
                                'pelvis',
                                'imu_in_pelvis',
                                'left_hip_pitch_link',
                                'left_hip_roll_link',
                                'left_hip_yaw_link',
                                'left_knee_link',
                                'pelvis_contour_link',
                                'right_hip_pitch_link',
                                'right_hip_roll_link',
                                'right_hip_yaw_link',
                                'right_knee_link',
                                'torso_link',
                                'd435_link',
                                'head_link',
                                'imu_in_torso',
                                'left_shoulder_pitch_link',
                                'left_shoulder_roll_link',
                                'left_shoulder_yaw_link',
                                'left_elbow_link',
                                'left_wrist_roll_rubber_hand',
                                'logo_link',
                                'mid360_link',
                                'right_shoulder_pitch_link',
                                'right_shoulder_roll_link',
                                'right_shoulder_yaw_link',
                                'right_elbow_link',
                                'right_wrist_roll_rubber_hand'
                                        ]), "threshold": 1.0},
    )
    out_of_terrain = TerminationTermCfg(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


@configclass
class StatisticsCfg:
    pos = term_cfg.StatisticsTermCfg(
        func= joints.StatusJPos,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot")},

        # episode_truncation = 80,
        export_interval = 1000000
    )
    action = term_cfg.StatisticsTermCfg(
        func= joints.StatusAction,
        params={
            "action_name": "joint_pos",
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot")},

        # episode_truncation = 80,
        export_interval = 1000000
    )
