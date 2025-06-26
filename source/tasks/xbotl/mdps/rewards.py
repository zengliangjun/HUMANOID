from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclabex.envs.mdp.rewards import reward_collect

@configclass
class RewardsCfg:
    # -- task
    joint_pos = RewardTermCfg(
        func=reward_collect.reward_jpos_withrefpose,
        weight=1.6,
        params={"asset_cfg": SceneEntityCfg("robot"),
                "phase_name": "phase_command"}
    )
    reward_jpos_yaw_rool = RewardTermCfg(
        func=reward_collect.reward_penalize_joint,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot",
                joint_names=[".*left_leg_roll_joint",
                            ".*right_leg_roll_joint",
                            ".*left_leg_yaw_joint",
                            ".*right_leg_yaw_joint",
                            ".*left_ankle_roll_joint",
                            ".*right_ankle_roll_joint",
                            ])}
    )
    reward_feet_clearance = RewardTermCfg(
        func=reward_collect.reward_clearance_phase,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot",
                         body_names=[".*left_ankle_roll_link", ".*right_ankle_roll_link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces",
                         body_names=[".*left_ankle_roll_link", ".*right_ankle_roll_link"]),
            "contacts_threshold": 5,
            "phase_command_name": 'phase_command',
            "feet_contact_height": 0.05,  # feet base height when feet contact on plane
            "feet_target_height": 0.06,
            "stance_phase": 0.55
                },
    )
    reward_feet_contact_number = RewardTermCfg(
        func=reward_collect.reward_contact_with_phase_number,
        weight=1.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces",
                         body_names=[".*left_ankle_roll_link", ".*right_ankle_roll_link"]),
            "phase_command_name": 'phase_command',
            "contacts_threshold": 5,
            "stance_phase": 0.55
            },
    )
    reward_feet_air_time = RewardTermCfg(
        func=reward_collect.reward_air_time_phase,
        weight=1.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces",
                         body_names=[".*left_ankle_roll_link", ".*right_ankle_roll_link"]),
            "phase_command_name": 'phase_command',
            "contacts_threshold": 5,
            "stance_phase": 0.55
            },
    )
    # feet , knee distance
    reward_body_distance = RewardTermCfg(
        func=reward_collect.reward_body_distance,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot",
                         body_names=[".*left_ankle_roll_link",
                                     ".*right_ankle_roll_link",
                                     ".*left_knee_link",
                                     ".*right_knee_link"]),
            "min_threshold": 0.2,
            "max_threshold": 0.5,
            },
    )
    # vel tracking
    rew_tracking_linear = RewardTermCfg(
        func=reward_collect.reward_lin_xy_exp,
        weight=1.2,
        params={"std": 0.13,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    rew_tracking_z = RewardTermCfg(
        func=reward_collect.reward_ang_z_exp,
        weight=1.1,
        params={"std": 0.13,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    reward_motion_lin_ang = RewardTermCfg(
        func=reward_collect.reward_motion_lin_ang,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot"),
                "linear_weight": 10,
                "angle_weight": 5},
    )
    reward_motion_speed = RewardTermCfg(
        func=reward_collect.reward_motion_speed,
        weight=0.2,
        params={"command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    reward_motion_hard = RewardTermCfg(
        func=reward_collect.reward_motion_hard,
        weight=0.5,
        params={"command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    reward_ori_euler_gravity_b = RewardTermCfg(
        func=reward_collect.reward_ori_euler_gravity_b,
        weight=1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_height_base2feet_phase = RewardTermCfg(
        func=reward_collect.reward_height_base2feet_phase,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                         body_names=[".*left_ankle_roll_link", ".*right_ankle_roll_link"]),
                "target_height": 0.89,
                "phase_command_name": 'phase_command',
                "stance_phase": 0.55,
                "feet_contact_height": 0.05,
        },
    )
    reward_base_acc_exp_norm = RewardTermCfg(
        func=reward_collect.reward_base_acc_exp_norm,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


    # penalize
    penalize_feet_slide = RewardTermCfg(
        func=reward_collect.penalize_slide_threshold,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot",
                         body_names=[".*left_ankle_roll_link", ".*right_ankle_roll_link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces",
                         body_names=[".*left_ankle_roll_link", ".*right_ankle_roll_link"]),
            "contacts_threshold": 5,
            },
    )

    penalize_feet_forces = RewardTermCfg(
        func=reward_collect.penalize_forces2,
        weight=-0.01,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces",
                         body_names=[".*left_ankle_roll_link", ".*right_ankle_roll_link"]),
            "threshold": 700,
            "max_over_penalize_forces": 400,
            },
    )
    penalize_action_smoothness = RewardTermCfg(
        func=reward_collect.penalize_action_smoothness,
        weight=-0.002,
        params={
            "weight1": 1,
            "weight2": 1,
            "weight3": 0.05,
            },
    )
    penalize_joint_torques_l2 = RewardTermCfg(
        func=reward_collect.penalize_torques_l2,
        weight=-1e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot")
            },
    )
    penalize_joint_vel_l2 = RewardTermCfg(
        func=reward_collect.penalize_jvel_l2,
        weight=-5e-4,
        params={
            "asset_cfg": SceneEntityCfg("robot")
            },
    )
    penalize_joint_acc_l2 = RewardTermCfg(
        func=reward_collect.penalize_jacc_l2,
        weight=-1e-7,
        params={
            "asset_cfg": SceneEntityCfg("robot")
            },
    )
    penalize_collision = RewardTermCfg(
        func=reward_collect.penalize_contacts,
        weight=-1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces",
                body_names=["base_link"])
            },
    )