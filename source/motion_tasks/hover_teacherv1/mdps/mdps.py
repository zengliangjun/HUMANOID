from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


from isaaclabex.envs.mdp.terminations import body
from isaaclabmotion.envs.mdp.terminations import body as mbody, motion_terminations

from isaaclabex.envs.managers import term_cfg
from isaaclabmotion.envs.mdp.statistics import robotbody

@configclass
class StatisticsCfg:
    rbpos_head_diff = term_cfg.StatisticsTermCfg(
        func= robotbody.RBPosHeadDiff,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "motions_name": "hoverh1"
        },

        # episode_truncation = 80,
        export_interval = 1000000
    )
    root_diff = term_cfg.StatisticsTermCfg(
        func= robotbody.RBRootDiff,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "motions_name": "hoverh1"
        },

        # episode_truncation = 80,
        export_interval = 1000000
    )




@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


motions_name = "hoverh1"
extend_body_names = [
    "left_hand_link",
    "right_hand_link",
    "head_link"
]

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    motion_termination = TerminationTermCfg(func=motion_terminations.motion_termination,
                                            params = {"motions_name": motions_name},  time_out=True)
    """
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

    out_of_terrain = TerminationTermCfg(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )
    """
    orientation = TerminationTermCfg(
        func=body.orientation_xywiththreshold,
        params={"asset_cfg": SceneEntityCfg("robot"), "xy_threshold": 0.7})
    """
    contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_cfg":
                SceneEntityCfg("contact_forces",
                body_names=[
                    "pelvis",
                    ".*_yaw_link",
                    ".*_roll_link",
                    ".*_pitch_link",
                    ".*_knee_link"
                ]), "threshold": 1.0},
    )
    height = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.4})

    """

    distance = TerminationTermCfg(
        func=mbody.terminate_by_reference_motion_distance,
        params={"asset_cfg": SceneEntityCfg("robot"),
                "training_mode": True,
                "max_ref_motion_dist": 0.5,
                "motions_name": motions_name,
                "extend_body_names": extend_body_names},
    )

