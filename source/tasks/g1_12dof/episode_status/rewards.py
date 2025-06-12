
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclabex.envs.mdp.rewards import reward_collect, pbrs_collect, pbrs_base


@configclass
class RewardsCfg:
    # -- task
    rew_lin_xy_exp = RewardTermCfg(
        func=reward_collect.reward_lin_xy_exp,
        weight=5,
        params={"std": 0.5,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    rew_ang_z_exp = RewardTermCfg(
        func=reward_collect.reward_ang_z_exp,
        weight=4,
        params={"std": 0.3,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    rew_motion_speed = RewardTermCfg(
        func=reward_collect.reward_motion_speed,
        weight=3,
        params={"command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    rew_motion_hard = RewardTermCfg(
        func=reward_collect.reward_motion_hard,
        weight=3,
        params={"command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    # joint step
    rew_step1 = RewardTermCfg(
        func=reward_collect.reward_step,
        weight=0.05,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                        "left_hip_roll_joint",
                        "right_hip_roll_joint",
                        "left_knee_joint",
                        "right_knee_joint",
                        "left_ankle_pitch_joint",
                        "right_ankle_pitch_joint"
                    ],
                    preserve_order = True),
                "command_name": "base_velocity",
                "method": 1 # withzero
                },
    )
    rew_step_center = RewardTermCfg(
        func=reward_collect.reward_step,
        weight=0.05,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_yaw_joint",
                        "right_hip_yaw_joint",
                        "left_ankle_roll_joint",
                        "right_ankle_roll_joint",
                    ],
                    preserve_order = True),
                "command_name": "base_velocity",
                "method": 0,  # default
                "center": 1,
                },
    )
    # joint episode
    rew_hipp_episode = RewardTermCfg(
        func=reward_collect.reward_episode,
        weight=0.8,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                    ]),
                "start_ids": [0],
                "end_ids": [1],
                "mean_std": 0.15,
                "variance_target": 0.04,
                "command_name": "base_velocity",
                "method": 0, # default
                },
    )
    rew_hipy_episode = RewardTermCfg(
        func=reward_collect.reward_episode,
        weight=0.05,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_yaw_joint",
                        "right_hip_yaw_joint",
                    ]),
                "start_ids": [0],
                "end_ids": [1],
                "mean_std": 0.08,
                "variance_target": -1,
                "command_name": "base_velocity",
                "method": 0 # default
                },
    )
    rew_hipr_episode = RewardTermCfg(
        func=reward_collect.reward_episode,
        weight=0.05,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_roll_joint",
                        "right_hip_roll_joint",
                    ]),
                "start_ids": [0],
                "end_ids": [1],
                "mean_std": 0.08,
                "variance_target": -1,
                "command_name": "base_velocity",
                "method": 0 # default
                },
    )
    rew_knee_episode = RewardTermCfg(
        func=reward_collect.reward_episode,
        weight=0.8,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_knee_joint",
                        "right_knee_joint",
                    ]),
                "start_ids": [0],
                "end_ids": [1],
                "mean_std": 0.3,
                "variance_target": 0.09,
                "command_name": "base_velocity",
                "method": 0 # default
                },
    )
    rew_anklep_episode = RewardTermCfg(
        func=reward_collect.reward_episode,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_ankle_pitch_joint",
                        "right_ankle_pitch_joint",
                    ]),
                "start_ids": [0],
                "end_ids": [1],
                "mean_std": 0.15,
                "variance_target": -1,# 0.02,
                "command_name": "base_velocity",
                "method": 1 # withzero
                },
    )
    rew_ankler_episode = RewardTermCfg(
        func=reward_collect.reward_episode,
        weight=0.05,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_ankle_roll_joint",
                        "right_ankle_roll_joint",
                    ]),
                "start_ids": [0],
                "end_ids": [1],
                "mean_std": 0.08,
                "variance_target": -1,
                "command_name": "base_velocity",
                "method": 0 # default
                },
    )
    rew_hipp2zero = RewardTermCfg(
        func=reward_collect.reward_left_right_symmetry,
        weight=0.3,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot",
                                        joint_names=[
                                                     "left_hip_pitch_joint",
                                                     "right_hip_pitch_joint",
                                        ]),
            "std": 0.25},
    )
    rew_hip_knee2zero = RewardTermCfg(
        func=reward_collect.reward_hip_knee_symmetry,
        weight=0.3,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot",
                                        joint_names=[
                                                     "left_hip_pitch_joint",
                                                     "left_knee_joint",
                                                     "right_hip_pitch_joint",
                                                     "right_knee_joint",
                                        ]),
            "weight": [1, -0.3],
            "std": 0.25},
    )
    rew_stability= RewardTermCfg(
        func=reward_collect.reward_stability,
        weight=1.0,
        params={"asset_cfg":
                SceneEntityCfg("robot", body_names=[
                                     ".*left_ankle_roll_link",
                                     ".*right_ankle_roll_link"])},
    )
    pbrs_ankle = RewardTermCfg(
        func=pbrs_collect.jpos_deviation_l1_pbrs,
        weight=1.0,
        params={"asset_cfg":
                SceneEntityCfg("robot", joint_names=[ ".*_ankle_roll_joint",
                                                      ".*_ankle_pitch_joint"]),
                "sigma": 0.25,
                "gamma": 1,
                "method": pbrs_base.PBRSExp},
    )
    pbrs_hipp = RewardTermCfg(
        func=pbrs_collect.jpos_deviation_l1_pbrs,
        weight=1.0,
        params={"asset_cfg":
                SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint"]),
                "sigma": 0.35, # 0.25,
                "gamma": 1,
                "method": pbrs_base.PBRSExp},
    )
    p_hipry = RewardTermCfg(
        func=reward_collect.penalize_jpos_deviation_l1,
        weight=-0.3,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[ ".*_hip_roll_joint", ".*_hip_yaw_joint"])
                },
    )

    # action
    p_action_rate = RewardTermCfg(
        func=reward_collect.penalize_action_rate_l2, weight=-0.1)
    # action
    p_action_smoothness = RewardTermCfg(
        func=reward_collect.penalize_action_smoothness,
        weight=-0.02,
        params={
            "weight1": 1,
            "weight2": 1,
            "weight3": 0.05,
            },
    )
    p_torques = RewardTermCfg(
        func=reward_collect.penalize_torques_l2,
        weight=-0.001, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    p_torque_limits = RewardTermCfg(
        func=reward_collect.penalize_torque_limits,
        weight=-1e-1, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    p_pos_limits = RewardTermCfg(
        func=reward_collect.penalize_jpos_limits_l1,
        weight=-20.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    # body
    p_width = RewardTermCfg(
        func=reward_collect.penalize_width,
        weight=-10,
        params={
            "target_width": 0.238,  # Adjusting for the foot clearance
            "target_height": 0.78,
            "center_velocity": 1.8,
            "asset_cfg": SceneEntityCfg("robot",
                         body_names=[".*left_ankle_roll_link",
                                     ".*right_ankle_roll_link",
                                     ".*left_knee_link",
                                     ".*right_knee_link"])
            },
    )

    p_orientation = RewardTermCfg(
        func=reward_collect.penalize_ori_l2,
        weight=-10, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    p_height = RewardTermCfg(
        func=reward_collect.penalize_height_flat_or_rayl2,
        weight=-40.0, params={
            "target_height": 0.78 + 0.035,  # Adjusting for the foot clearance
            "asset_cfg": SceneEntityCfg("robot")}
    )
    # feet
    p_foot_slide = RewardTermCfg(
        func=reward_collect.penalize_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    p_foot_clearance = RewardTermCfg(
        func=reward_collect.penalize_clearance,
        weight=-20.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            'target_height': 0.16 + 0.055
        },
    )
