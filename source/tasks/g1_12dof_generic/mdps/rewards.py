from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclabex.envs.mdp.rewards import reward_collect

@configclass
class RewardsCfg:
    # Movement Rewards
    linear_xy_reward = RewardTermCfg(
        func=reward_collect.reward_lin_xy_exp,
        weight=1.0,
        params={"std": 0.25,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    angular_z_reward = RewardTermCfg(
        func=reward_collect.reward_ang_z_exp,
        weight=1.0,
        params={"std": 0.25,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    motion_speed_reward = RewardTermCfg(
        func=reward_collect.reward_motion_speed,
        weight=0.2,
        params={"command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    motion_hard_reward = RewardTermCfg(
        func=reward_collect.reward_motion_hard,
        weight=0.5,
        params={"command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    symmetry_lr_reward = RewardTermCfg(
        func=reward_collect.reward_left_right_symmetry,
        weight=1.0,
        params={"std": 0.25,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot",
                            joint_names=["left_hip_pitch_joint",
                                        "right_hip_pitch_joint",
                                        "left_knee_joint",
                                        "right_knee_joint"
                                        ],
                            preserve_order = True)},
    )
    symmetry_hip_knee = RewardTermCfg(
        func=reward_collect.reward_hip_knee_symmetry,
        weight=1.0,
        params={"std": 0.25,
                "weight":  [1, 0.6], # [hip, knee]
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot",
                            joint_names=["left_hip_pitch_joint",
                                        "left_knee_joint",
                                        "right_hip_pitch_joint",
                                        "right_knee_joint"
                                        ],
                            preserve_order = True)},
    )
    # Foot Rewards
    foot_air_time_reward = RewardTermCfg(
        func=reward_collect.reward_air_time_biped,
        weight= 0.25, #0.2,
        params={"command_name": "base_velocity",
                "threshold": 0.4,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    foot_force_z_reward = RewardTermCfg(
        func=reward_collect.reward_forces_z,
        weight=5e-3,
        params={
            "threshold": 500,
            "max_forces": 400,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])},
    )

    # Episodic Penalty
    episode_termination_penalty = RewardTermCfg(
        func=reward_collect.penalize_eps_terminated,
        weight=-200,
    )

    # Base Penalties
    base_ang_xy_penalty = RewardTermCfg(
        func=reward_collect.penalize_ang_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_linear_z_penalty = RewardTermCfg(
        func=reward_collect.penalize_lin_z_l2,
        weight=-1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_orientation_penalty = RewardTermCfg(
        func=reward_collect.reward_ori_euler_gravity_b,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # Joint Penalties
    joint_energy_penalty = RewardTermCfg(
        func=reward_collect.penalize_energy,
        weight=-0.0005,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_acceleration_penalty = RewardTermCfg(
        func=reward_collect.penalize_jacc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    joint_limits_penalty = RewardTermCfg(
        func=reward_collect.penalize_jpos_limits_l1,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    joint_pitch_penalty = RewardTermCfg(
        func=reward_collect.penalize_jpos_deviation_l1,
        weight=-0.015,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[".*_hip_pitch_joint",
                             ".*_knee_joint",
                             ".*_ankle_pitch_joint"])}
    )
    joint_ry_penalty = RewardTermCfg(
        func=reward_collect.penalize_jpos_deviation_l1,
        weight=-0.5, #-0.1,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[".*_hip_yaw_joint",
                             ".*_hip_roll_joint",
                             ".*_ankle_roll_joint"])}
    )

    # Foot Contact Penalties
    foot_stumble_penalty = RewardTermCfg(
        func=reward_collect.penalty_stumble,
        weight=5e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])
        },
    )
    foot_slide_penalty = RewardTermCfg(
        func=reward_collect.penalize_slide,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
        },
    )
    foot_airborne_penalty = RewardTermCfg(
        func=reward_collect.penalize_airborne,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "threshold": 0.12}
    )

    # Action Penalty
    action_rate_penalty = RewardTermCfg(
        func=reward_collect.penalize_action_rate_l2,
        weight=-0.01
    )

    # Contact Penalty
    contact_undesired_penalty = RewardTermCfg(
        func=reward_collect.penalize_undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces",
                body_names=[
                    "torso_link",
                    ".*_hip_yaw_link",
                    ".*_hip_roll_link",
                    ".*_hip_pitch_link",
                    ".*_knee_link",
                    ".*_shoulder_pitch_link",
                    ".*_shoulder_roll_link",
                    ".*_shoulder_yaw_link",
                    ".*_elbow_link"
                ]),
            "threshold": 1.0
        }
    )
