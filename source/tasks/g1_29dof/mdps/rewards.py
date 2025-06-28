
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclabex.envs.mdp.rewards import reward_collect


@configclass
class RewardsCfg:
    # -- task
    rew_lin_xy_exp = RewardTermCfg(
        func=reward_collect.reward_lin_xy_exp,
        weight=5 * 2,
        params={"std": 0.35 * 1.2,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    rew_ang_z_exp = RewardTermCfg(
        func=reward_collect.reward_ang_z_exp,
        weight=3 * 1.8,
        params={"std": 0.25 * 1.2,
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
        params={"std": 0.25 * 1.2,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )

    #
    rew_pitch_total2zero = RewardTermCfg(
        func=reward_collect.rew_pitch_total2zero,
        weight=0.4 * 1.2,
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

    rew_mean_hipp = RewardTermCfg(
        func=reward_collect.rew_mean_self,
        weight=0.3 * 3,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint"]),
                "pos_statistics_name": "pos",
                "std": 0.25 * 1.2,
                "diff_scale": 2
                }
    )

    rew_mean_knee = RewardTermCfg(
        func=reward_collect.rew_mean_self,
        weight=0.3 * 3,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_knee_joint",
                        "right_knee_joint"]),
                "pos_statistics_name": "pos",
                "std": 0.25 * 1.2,
                "diff_scale": 2
                }
    )

    # shoulder
    rew_mean_shoulderp_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero,
        weight=0.1 * 2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint"]),
                "pos_statistics_name": "pos",
                "std": 0.05 * 4 * 1.2
                }
    )
    rew_mean_ankler_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero_nostep,
        weight=0.08 * 1.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_ankle_roll_joint","right_ankle_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.10 * 2 * 1.2
                }
    )
    rew_mean_hipr_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero_nostep,
        weight=0.08 * 2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_roll_joint",  "right_hip_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.06 * 2 * 1.2
                }
    )
    rew_mean_hipy_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero_nostep,
        weight=0.08 * 2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_yaw_joint",   "right_hip_yaw_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.03 * 3 * 1.2
                }
    )

    rew_mean_waistrpy_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero_nosymmetry,
        weight=0.24 * 2.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint"
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.01 * 4 * 1.2
                }
    )
    rew_mean_shoulderr_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero_nosymmetry,
        weight=0.16,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_roll_joint",   "right_shoulder_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.03 * 2 * 1.2
                }
    )
    rew_mean_shouldery_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero_nosymmetry,
        weight=0.24 * 2.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_yaw_joint",   "right_shoulder_yaw_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.03 * 2 * 1.2
                }
    )
    rew_mean_elbow_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero_nosymmetry,
        weight=0.12 * 1.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_elbow_joint",   "right_elbow_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.01 * 4 * 1.2
                }
    )
    rew_mean_wristr_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero_nosymmetry,
        weight=0.12 * 1.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_wrist_roll_joint",   "right_wrist_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.02 * 4 * 1.2
                }
    )
    rew_mean_wristp_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero_nosymmetry,
        weight=0.12 * 1.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_wrist_pitch_joint",   "right_wrist_pitch_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.02 * 4 * 1.2
                }
    )
    rew_mean_wristy_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero_nosymmetry,
        weight=0.12 * 1.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_wrist_yaw_joint",   "right_wrist_yaw_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.02 * 4 * 1.2
                }
    )

    # shoulder
    rew_shoulderp_self = RewardTermCfg(
        func=reward_collect.rew_variance_self_noencourage,
        weight=0.15 * 1.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint",
                        ]),
                "pos_statistics_name": "pos",
                "std": 0.01 * 4 * 1.2,
                "diff_scale": 1.2,
                }
    )

    rew_hipp_self = RewardTermCfg(
        func=reward_collect.rew_variance_self,
        weight=0.3 * 2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                        ]),
                "pos_statistics_name": "pos",
                "std": 0.09 * 1.2,
                "diff_scale": 1.6,
                }
    )

    rew_knee_self = RewardTermCfg(
        func=reward_collect.rew_variance_self,
        weight=0.3 * 2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_knee_joint",
                        "right_knee_joint"
                        ]),
                "pos_statistics_name": "pos",
                "std": 0.09 * 1.2,
                "diff_scale": 1.6,
                }
    )
    rew_ankler_zero = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.08 * 1.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_ankle_roll_joint","right_ankle_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.008 * 2 * 1.2
                },
    )

    rew_hipr_zero = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.08 * 2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_roll_joint",  "right_hip_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.003 * 2 * 1.2
                },
    )

    rew_hipy_zero = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.08 * 2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_yaw_joint",   "right_hip_yaw_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.003 * 2 * 1.2
                },
    )
    #

    rew_waistrpy_zero = RewardTermCfg(
        func=reward_collect.rew_variance_zero_nosymmetry,
        weight=0.24 * 3,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint"
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.001 * 3 * 1.2
                },
    )
    rew_shoulderr_zero = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_roll_joint",   "right_shoulder_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.002 * 3 * 1.2
                },
    )
    rew_shouldery_zero = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.2 * 1.5,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_yaw_joint",   "right_shoulder_yaw_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.002 * 3 * 1.2
                },
    )
    rew_elbow_zero = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.16 * 1.5,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_elbow_joint",   "right_elbow_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.001 * 3 * 1.2
                },
    )
    rew_wristr_zero = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.16,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_wrist_roll_joint",   "right_wrist_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.001 * 3 * 1.2
                },
    )
    rew_wristp_zero = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.16 * 1.5,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_wrist_pitch_joint",   "right_wrist_pitch_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.001 * 3 * 1.2
                },
    )
    rew_wristy_zero = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.16 * 1.5,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_wrist_yaw_joint",   "right_wrist_yaw_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.001 * 3 * 1.2
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
        func=reward_collect.reward_penalize_joint,
        weight=0.1,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[ ".*_hip_roll_joint", ".*_hip_yaw_joint"]),
                "diff_range": 0.05,
                "diff_std": 0.1 * 2.5,
                "penalize_weight": - 0.2
                },
    )
    p_shoulderp = RewardTermCfg(
        func=reward_collect.reward_penalize_joint,
        weight=0.10,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[ ".*_shoulder_pitch_joint"]),
                "diff_range": 0.03,
                "diff_std": 0.1 * 2.5,
                "penalize_weight": - 0.1
                },
    )
    p_waistrpy = RewardTermCfg(
        func=reward_collect.reward_penalize_joint,
        weight=0.2,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[ "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]),
                "diff_range": 0,
                "diff_std": 0.03 * 2.5,
                "penalize_weight": - 0.1
                },
    )
    p_shoulderry = RewardTermCfg(
        func=reward_collect.reward_penalize_joint,
        weight=0.15,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[ ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint"]),
                "diff_range": 0,
                "diff_std": 0.03 * 2.5,
                "penalize_weight": - 0.1
                },
    )
    p_elbow = RewardTermCfg(
        func=reward_collect.reward_penalize_joint,
        weight=0.1,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[ ".*_elbow_joint"]),
                "diff_range": 0.03,
                "diff_std": 0.05 * 2.5,
                "penalize_weight": - 0.1
                },
    )
    p_wristrpy = RewardTermCfg(
        func=reward_collect.reward_penalize_joint,
        weight=0.2,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[ ".*_wrist_roll_joint", ".*_wrist_pitch_joint", ".*_wrist_yaw_joint"]),
                "diff_range": 0,
                "diff_std": 0.03 * 2.5,
                "penalize_weight": - 0.1
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
        weight=-0.001, params={"asset_cfg":
                               SceneEntityCfg("robot",
                               joint_names=[ ".*_hip_pitch_joint",
                                             ".*_hip_roll_joint",
                                             ".*_hip_yaw_joint",
                                             ".*_knee_joint",
                                             ".*_ankle_pitch_joint",
                                             ".*_ankle_roll_joint",
                                             ".*_shoulder_pitch_joint"]
                                              )}
    )
    p_torques_upper = RewardTermCfg(
        func=reward_collect.penalize_torques_l2,
        weight=-0.001, params={"asset_cfg": SceneEntityCfg("robot",
                               joint_names=[ "waist_yaw_joint",
                                             "waist_roll_joint",
                                             "waist_pitch_joint",
                                             ".*_shoulder_roll_joint",
                                             ".*_shoulder_yaw_joint",
                                             ".*_elbow_joint",
                                             ".*_wrist_roll_joint",
                                             ".*_wrist_pitch_joint",
                                             ".*_wrist_yaw_joint"]
                                             )}
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
    reward_orientation = RewardTermCfg(
        func=reward_collect.reward_ori_euler_gravity_b,
        weight=1, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    p_orientation = RewardTermCfg(
        func=reward_collect.penalize_ori_l2,
        weight=-10, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    p_height = RewardTermCfg(
        func=reward_collect.penalize_height_base2feet,
        weight=-40.0, params={
            "target_height": 0.78,  # Adjusting for the foot clearance
            "asset_cfg": SceneEntityCfg("robot",
                         body_names=".*_ankle_roll_link")}
    )
    rp_height_upper = RewardTermCfg(
        func=reward_collect.rp_height_upper,
        weight=0.25, params={
            "target_height": 0.46018,  # Adjusting for the foot clearance
            "error_std": 0.025,
            "penalize_weight": -1.5,
            "asset_cfg": SceneEntityCfg("robot",
                         body_names="mid360_link")}
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
        weight=-300,
    )
    p_uncontacts = RewardTermCfg(
        func=reward_collect.penalize_undesired_contacts,
        weight=-1,
        params={
            'threshold': 5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[
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
                                'waist_yaw_link',
                                'waist_roll_link',
                                'torso_link',
                                'd435_link',
                                'head_link',
                                'imu_in_torso',
                                'left_shoulder_pitch_link',
                                'left_shoulder_roll_link',
                                'left_shoulder_yaw_link',
                                'left_elbow_link',
                                'left_wrist_roll_link',
                                'left_wrist_pitch_link',
                                'left_wrist_yaw_link',
                                'left_rubber_hand',
                                'logo_link',
                                'mid360_link',
                                'right_shoulder_pitch_link',
                                'right_shoulder_roll_link',
                                'right_shoulder_yaw_link',
                                'right_elbow_link',
                                'right_wrist_roll_link',
                                'right_wrist_pitch_link',
                                'right_wrist_yaw_link',
                                'right_rubber_hand'
            ])

        }
    )
