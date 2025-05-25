
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.envs import mdp
from isaaclab_tasks.manager_based.locomotion.velocity import mdp as velocity_mdp
from isaaclabex.envs.mdp.rewards import energy, feet_contact

@configclass
class RewardsCfg:
    # -- task
    rew_tracking_linear = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"std": 0.25,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    rew_tracking_z = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"std": 0.25,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )

    rew_air_time = RewardTermCfg(
        func=velocity_mdp.feet_air_time,
        weight=0.2,
        params={"command_name": "base_velocity",
                "threshold": 0.4,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )

    rew_feet_forces_z = RewardTermCfg(
            func=feet_contact.reward_feet_forces_z,
            weight=5e-3,
            params={
                "threshold": 500,
                "max_forces": 400,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])},
    )

    penalize_terminated = RewardTermCfg(
        func=mdp.is_terminated,
        weight=-200,
    )
    penalize_linear_xy = RewardTermCfg(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    penalize_linear_z = RewardTermCfg(
        func=mdp.lin_vel_z_l2,
        weight=-1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    penalize_energy = RewardTermCfg(
        func=energy.energy_cost,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    penalize_joint_acc = RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    penalize_action_rate = RewardTermCfg(
        func=mdp.action_rate_l2, weight=-0.01)

    penalize_orientation = RewardTermCfg(
        func=mdp.flat_orientation_l2,
        weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )

    penalize_pos_limits = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )

    penalize_pitch = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-.05,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[".*_hip_pitch_joint",
                             ".*_knee_joint",
                             ".*_ankle_pitch_joint"
                             ])},
    )

    penalize_hip = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-.1,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                             ])},
    )

    penalize_other = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "torso_joint",
                    ".*_ankle_roll_joint",

                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
                ],
            )
        },
    )

    penalize_fingers = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_five_joint",
                    ".*_three_joint",
                    ".*_six_joint",
                    ".*_four_joint",
                    ".*_zero_joint",
                    ".*_one_joint",
                    ".*_two_joint",
                ],
            )
        },
    )

    penalize_feet_stumble = RewardTermCfg(
            func=feet_contact.penalty_feet_stumble,
            weight=5e-3,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])},
    )

    penalize_foot_slide = RewardTermCfg(
        func=velocity_mdp.feet_slide,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )

    penalize_feet_airborne = RewardTermCfg(
        func=feet_contact.penalty_feet_airborne,
        weight=-.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")}
    )

    penalize_undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-.1,
        params={
            "sensor_cfg":
            SceneEntityCfg("contact_forces",
                body_names=[
                    "torso_link",
                    ".*_hip_yaw_link",
                    ".*_hip_roll_link",
                    ".*_shoulder_pitch_link",
                    ".*_shoulder_roll_link",
                    ".*_shoulder_yaw_link",
                    ".*_elbow_pitch_link",
                    ".*_elbow_roll_link",
                ]),
            "threshold": 1.0})
