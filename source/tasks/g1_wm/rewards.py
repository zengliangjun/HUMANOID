
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclabex.envs.mdp.rewards import reward_collect

@configclass
class RewardsCfg:
    # -- task
    rew_tracking_linear = RewardTermCfg(
        func=reward_collect.track_lin_vel_xy_exp,
        weight=1.0,
        params={"std": 0.25,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    rew_tracking_z = RewardTermCfg(
        func=reward_collect.track_ang_vel_z_exp,
        weight=1.0,
        params={"std": 0.25,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )

    # feet
    rew_air_time = RewardTermCfg(
        func=reward_collect.feet_air_time,
        weight=0.2,
        params={"command_name": "base_velocity",
                "threshold": 0.4,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )

    rew_feet_forces_z = RewardTermCfg(
            func=reward_collect.reward_feet_forces_z,
            weight=5e-3,
            params={
                "threshold": 500,
                "max_forces": 400,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])},
    )

    # episodic
    penalize_terminated = RewardTermCfg(
        func=reward_collect.is_terminated,
        weight=-200,
    )
    # base
    penalize_linear_xy = RewardTermCfg(
        func=reward_collect.ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    penalize_linear_z = RewardTermCfg(
        func=reward_collect.lin_vel_z_l2,
        weight=-1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    penalize_orientation = RewardTermCfg(
        func=reward_collect.flat_orientation_l2,
        weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # joint
    penalize_energy = RewardTermCfg(
        func=reward_collect.energy_cost,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    penalize_joint_acc = RewardTermCfg(
        func=reward_collect.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    penalize_pos_limits = RewardTermCfg(
        func=reward_collect.joint_pos_limits,
        weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )

    penalize_pitch = RewardTermCfg(
        func=reward_collect.joint_deviation_l1,
        weight=-.05,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[".*_hip_pitch_joint",
                             ".*_knee_joint",
                             ".*_ankle_pitch_joint"
                             ])},
    )

    penalize_hip = RewardTermCfg(
        func=reward_collect.joint_deviation_l1,
        weight=-.1,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ])},
    )

    penalize_other = RewardTermCfg(
        func=reward_collect.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_ankle_roll_joint",
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_roll_joint",
                    ".*_wrist_yaw_joint",
                    "waist_pitch_joint"
                ],
            )
        },
    )

    penalize_waist = RewardTermCfg(
        func=reward_collect.joint_deviation_l1,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist_yaw_joint",
                    "waist_roll_joint"
                ],
            )
        },
    )

    penalize_feet_stumble = RewardTermCfg(
            func=reward_collect.penalty_feet_stumble,
            weight=5e-3,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])},
    )

    penalize_foot_slide = RewardTermCfg(
        func=reward_collect.feet_slide,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )

    penalize_feet_airborne = RewardTermCfg(
        func=reward_collect.penalty_feet_airborne,
        weight=-.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")}
    )
    # action
    penalize_action_rate = RewardTermCfg(
        func=reward_collect.action_rate_l2, weight=-0.01)

    # contacts
    penalize_undesired_contacts = RewardTermCfg(
        func=reward_collect.undesired_contacts,
        weight=-.1,
        params={
            "sensor_cfg":
            SceneEntityCfg("contact_forces",
                body_names=[
                    'torso_link',
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
                    ".*_wrist_yaw_link",
                ]),
            "threshold": 1.0})
