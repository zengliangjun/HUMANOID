from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


from isaaclabex.envs.mdp.commands import commands_cfg

from isaaclabex.envs.mdp.statistics import fldstatus, joints
from isaaclabex.envs.managers import term_cfg

from rsl_rlex.fld.modules import modules_cfg

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

    fld_status = term_cfg.StatisticsTermCfg(
        func= fldstatus.FLDCollect,
        params={
            "training": True,
            "training_noise_level": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
            "status_names": [
                "ang_vel",
                #"gravity",
                #"commands",
                "joint_pos",
                #"joint_vel",
                #"action"
            ],
            "fld_module_cfg": modules_cfg.FLDCfg(
                                fldmodel_prefix = "tspmodel",
                                step_dt = 0.02,
                                observation_dim = 15,
                                observation_history_horizon = 51,
                                encoder_hidden_dims = [64, 64, 32],
                                decoder_hidden_dims = [32, 64, 64]
                            ),
            "fld_loss_scales": [
                0.5, 0.5, 0.5,
                # 1.0, 1.0, 1.0,
                # 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                # 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                # 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
            "fld_learning_rate": 0.0001,
            "fld_weight_decay": 0.0005,
        },
    )


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
            #lin_vel_x=(0, 4.5), lin_vel_y=(-0.75, 0.75), ang_vel_z=(-2., 2.), heading=(0., 0)
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

    out_of_terrain = TerminationTermCfg(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )

    orientation = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 3.14 * 45 / 180})

    height = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.4})
