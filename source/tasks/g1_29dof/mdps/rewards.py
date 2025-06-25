
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclabex.envs.mdp.rewards import reward_collect


@configclass
class RewardsCfg:
    # -- task
    rew_lin_xy_exp = RewardTermCfg(
        func=reward_collect.reward_lin_xy_exp,
        weight=7,
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
        weight=2,
        params={"command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    rew_motion_hard = RewardTermCfg(
        func=reward_collect.reward_motion_hard,
        weight=3,
        params={"command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )

    #
    rew_pitch_total2zero = RewardTermCfg(
        func=reward_collect.rew_pitch_total2zero,
        weight=0.4,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                        "left_knee_joint",
                        "right_knee_joint",
                        "left_ankle_pitch_joint",
                        "right_ankle_pitch_joint"
                    ],
                    preserve_order = True)},
    )

    # shoulder
    rew_mean_shoulderp = RewardTermCfg(
        func=reward_collect.rew_mean_self,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint"]),
                "pos_statistics_name": "pos",
                "std": 0.10
                }
    )

    rew_mean_hipp = RewardTermCfg(
        func=reward_collect.rew_mean_self,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint"]),
                "pos_statistics_name": "pos",
                "std": 0.25
                }
    )

    rew_mean_knee = RewardTermCfg(
        func=reward_collect.rew_mean_self,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_knee_joint",
                        "right_knee_joint"]),
                "pos_statistics_name": "pos",
                "std": 0.25
                }
    )

    rew_mean_ankler_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero_nostep,
        weight=0.08,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_ankle_roll_joint","right_ankle_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.08
                }
    )
    rew_mean_hipr_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero_nostep,
        weight=0.08,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_roll_joint",  "right_hip_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.08
                }
    )
    rew_mean_hipy_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero_nostep,
        weight=0.08,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_yaw_joint",   "right_hip_yaw_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.03
                }
    )
    rew_mean_waistrpy_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero_nostep,
        weight=0.08,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint"
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.01
                }
    )
    rew_mean_shoulderr_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero_nostep,
        weight=0.08,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_roll_joint",   "right_shoulder_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.05
                }
    )
    rew_mean_shouldery_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero_nostep,
        weight=0.08,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_yaw_joint",   "right_shoulder_yaw_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.05
                }
    )
    rew_mean_elbow_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero_nostep,
        weight=0.08,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_elbow_joint",   "right_elbow_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.03
                }
    )
    rew_mean_wristr_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero_nostep,
        weight=0.08,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_wrist_roll_joint",   "right_wrist_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.03
                }
    )
    rew_mean_wristp_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero_nostep,
        weight=0.08,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_wrist_pitch_joint",   "right_wrist_pitch_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.03
                }
    )
    rew_mean_wristy_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero_nostep,
        weight=0.08,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_wrist_yaw_joint",   "right_wrist_yaw_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.05
                }
    )
    # shoulder
    rew_shoulderp_self = RewardTermCfg(
        func=reward_collect.rew_variance_self,
        weight=0.15,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint",
                        ]),
                "pos_statistics_name": "pos",
                "std": 0.01
                }
    )

    rew_hipp_self = RewardTermCfg(
        func=reward_collect.rew_variance_self,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                        ]),
                "pos_statistics_name": "pos",
                "std": 0.09
                }
    )

    rew_knee_self = RewardTermCfg(
        func=reward_collect.rew_variance_self,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_knee_joint",
                        "right_knee_joint"
                        ]),
                "pos_statistics_name": "pos",
                "std": 0.09
                }
    )

    rew_ankler_zero = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.08,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_ankle_roll_joint","right_ankle_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.01
                },
    )

    rew_hipr_zero = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.08,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_roll_joint",  "right_hip_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.006
                },
    )

    rew_hipy_zero = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.08,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_yaw_joint",   "right_hip_yaw_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.008
                },
    )
    #
    rew_waistrpy_zero = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.08,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint"
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.003
                },
    )
    rew_shoulderr_zero = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.08,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_roll_joint",   "right_shoulder_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.006
                },
    )
    rew_shouldery_zero = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.08,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_yaw_joint",   "right_shoulder_yaw_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.006
                },
    )
    rew_elbow_zero = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.08,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_elbow_joint",   "right_elbow_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.006
                },
    )
    rew_wristr_zero = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.08,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_wrist_roll_joint",   "right_wrist_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.004
                },
    )
    rew_wristp_zero = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.08,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_wrist_pitch_joint",   "right_wrist_pitch_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.004
                },
    )
    rew_wristy_zero = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.08,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_wrist_yaw_joint",   "right_wrist_yaw_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.004
                },
    )

    rew_step_vv_mv = RewardTermCfg(
        func=reward_collect.rew_step_vv_mv,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot"),
                "pos_statistics_name": "pos"},
    )

    rew_stability= RewardTermCfg(
        func=reward_collect.reward_stability,
        weight=3.0,
        params={"asset_cfg":
                SceneEntityCfg("robot", body_names=[
                                     ".*left_ankle_roll_link",
                                     ".*right_ankle_roll_link"])},
    )
    p_hipry = RewardTermCfg(
        func=reward_collect.penalize_jpos_deviation_l1,
        weight=-0.06,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[ ".*_hip_roll_joint", ".*_hip_yaw_joint"])
                },
    )
    p_waistrpy = RewardTermCfg(
        func=reward_collect.penalize_jpos_deviation_l1,
        weight=-0.06,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[ "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"])
                },
    )
    p_shoulderry = RewardTermCfg(
        func=reward_collect.penalize_jpos_deviation_l1,
        weight=-0.06,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[ ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint"])
                },
    )
    p_shoulderp = RewardTermCfg(
        func=reward_collect.penalize_jpos_deviation_l1,
        weight=-0.08,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[ ".*_shoulder_pitch_joint"])
                },
    )
    p_elbow = RewardTermCfg(
        func=reward_collect.penalize_jpos_deviation_l1,
        weight=-0.08,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[ ".*_elbow_joint"])
                },
    )
    p_wristrpy = RewardTermCfg(
        func=reward_collect.penalize_jpos_deviation_l1,
        weight=-0.03,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[ ".*_wrist_roll_joint", ".*_wrist_pitch_joint", ".*_wrist_yaw_joint"])
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
        func=reward_collect.penalize_height_base2feet,
        weight=-40.0, params={
            "target_height": 0.78,  # Adjusting for the foot clearance
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

    # -------------------- Episode Penalty --------------------
    p_termination = RewardTermCfg(
        func=reward_collect.penalize_eps_terminated,
        weight=-200,
    )
