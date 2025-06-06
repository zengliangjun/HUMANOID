from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclabex.envs.mdp.rewards import reward_collect

@configclass
class RewardsCfg:
    # -------------------- Locomotion Rewards --------------------
    forward_velocity_exp = RewardTermCfg(
        func=reward_collect.reward_lin_xy_exp,
        weight=5.0,
        params={"std": 0.25,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    yaw_velocity_exp = RewardTermCfg(
        func=reward_collect.reward_ang_z_exp,
        weight=1.0,
        params={"std": 0.25,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )

    # -------------------- Foot Rewards --------------------
    foot_air_time = RewardTermCfg(
        func=reward_collect.reward_air_time,
        weight=0.2,
        params={"command_name": "base_velocity",
                "threshold": 0.4,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    foot_vertical_force = RewardTermCfg(
        func=reward_collect.reward_forces_z,
        weight=5e-3,
        params={
            "threshold": 500,
            "max_forces": 400,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])},
    )
    foot_stumble = RewardTermCfg(
        func=reward_collect.penalty_stumble,
        weight=5e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])
        },
    )
    foot_slide = RewardTermCfg(
        func=reward_collect.penalize_slide,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
        },
    )
    foot_airborne = RewardTermCfg(
        func=reward_collect.penalize_airborne,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")}
    )

    # -------------------- Trunk/Base Penalties --------------------
    base_ang_xy = RewardTermCfg(
        func=reward_collect.penalize_ang_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_lin_z = RewardTermCfg(
        func=reward_collect.penalize_lin_z_l2,
        weight=-1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_orientation = RewardTermCfg(
        func=reward_collect.penalize_ori_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # -------------------- Joint Penalties --------------------
    joint_energy = RewardTermCfg(
        func=reward_collect.penalize_energy,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_acceleration = RewardTermCfg(
        func=reward_collect.penalize_jacc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    joint_limits = RewardTermCfg(
        func=reward_collect.penalize_jpos_limits_l1,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    joint_pitch_deviation = RewardTermCfg(
        func=reward_collect.penalize_jpos_deviation_l1,
        weight=-0.025,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[".*_hip_pitch_joint",
                             ".*_knee_joint",
                             ".*_ankle_pitch_joint"])}
    )
    joint_hip_deviation = RewardTermCfg(
        func=reward_collect.penalize_jpos_deviation_l1,
        weight=-0.1,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[".*_hip_yaw_joint",
                             ".*_hip_roll_joint"])}
    )
    joint_other_deviation = RewardTermCfg(
        func=reward_collect.penalize_jpos_deviation_l1,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot",
                joint_names=[
                    ".*_ankle_roll_joint",
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_roll_joint",
                    ".*_wrist_yaw_joint"
                ])
        },
    )
    joint_waist_deviation = RewardTermCfg(
        func=reward_collect.penalize_jpos_deviation_l1,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot",
                joint_names=[
                    "waist_yaw_joint",
                    "waist_roll_joint",
                    "waist_pitch_joint"
                ])
        },
    )

    # -------------------- Action Penalty --------------------
    action_rate = RewardTermCfg(
        func=reward_collect.penalize_action_rate_l2,
        weight=-0.01
    )

    # -------------------- Contact Penalty --------------------
    contact_undesired = RewardTermCfg(
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
                    ".*_elbow_link",
                    ".*_wrist_pitch_link",
                    ".*_wrist_roll_link",
                    ".*_wrist_yaw_link"
                ]),
            "threshold": 1.0
        }
    )

    # -------------------- Episode Penalty --------------------
    episode_termination = RewardTermCfg(
        func=reward_collect.penalize_eps_terminated,
        weight=-200,
    )
