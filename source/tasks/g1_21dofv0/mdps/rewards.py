from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclabex.envs.mdp.rewards import reward_collect

@configclass
class RewardsCfg:
    rew_lin_xy_exp = RewardTermCfg(
        func=reward_collect.reward_lin_xy_exp,
        weight=5,
        params={"std": 0.35,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    rew_ang_z_exp = RewardTermCfg(
        func=reward_collect.reward_ang_z_exp,
        weight=4,
        params={"std": 0.25,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    rew_motion_speed = RewardTermCfg(
        func=reward_collect.reward_motion_speed,
        weight=5,
        params={"command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    rew_motion_hard = RewardTermCfg(
        func=reward_collect.reward_motion_hard,
        weight=6,
        params={"std": 0.25,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )

    rew_step_vv_mv = RewardTermCfg(
        func=reward_collect.rew_step_vv_mv,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot"),
                "pos_statistics_name": "pos"},
    )

from .rewards_uper import RewardsUperCfg
from .rewards_leg import RewardsLegCfg

@configclass
class RewardsG21Cfg(RewardsUperCfg, RewardsLegCfg, RewardsCfg):
    rew_stability= RewardTermCfg(
        func=reward_collect.reward_stability,
        weight=1.0,
        params={"asset_cfg":
                SceneEntityCfg("robot", body_names=[
                                     ".*left_ankle_roll_link",
                                     ".*right_ankle_roll_link"])},
    )

    # action
    p_action_rate = RewardTermCfg(
        func=reward_collect.penalize_action_rate_l2, weight=-0.1)

    p_action_smoothness = RewardTermCfg(
        func=reward_collect.penalize_action_smoothness,
        weight=-0.02,
        params={
            "weight1": 1,
            "weight2": 1,
            "weight3": 0.05,
            },
    )
    p_energy = RewardTermCfg(
        func=reward_collect.penalize_energy,
        weight=-1e-4,
        params={
            "asset_cfg":
                SceneEntityCfg("robot")
            },
    )

    p_torques_pitch = RewardTermCfg(
        func=reward_collect.penalize_torques_l2,
        weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot",
                               joint_names=[ ".*_hip_pitch_joint",
                                             ".*_knee_joint",
                                             ".*_ankle_pitch_joint",
                                             ".*_ankle_roll_joint",
                                             ".*_shoulder_pitch_joint",
                                             ".*_elbow_joint"]
                                              )}
    )
    p_torques_other = RewardTermCfg(
        func=reward_collect.penalize_torques_l2,
        weight=-1e-3, params={"asset_cfg": SceneEntityCfg("robot",
                               joint_names=[ "waist_yaw_joint",
                                             ".*_hip_roll_joint",
                                             ".*_hip_yaw_joint",
                                             ".*_shoulder_roll_joint",
                                             ".*_shoulder_yaw_joint"
                                             ]
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
            "error_std": 0.06,
            "penalize_weight": -0.25,
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
        weight=-200,
    )
    p_uncontacts = RewardTermCfg(
        func=reward_collect.penalize_undesired_contacts,
        weight=-1,
        params={
            'threshold': 1,
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
                                'torso_link',
                                'd435_link',
                                'head_link',
                                'imu_in_torso',
                                'left_shoulder_pitch_link',
                                'left_shoulder_roll_link',
                                'left_shoulder_yaw_link',
                                'left_elbow_link',
                                'left_rubber_hand',
                                'logo_link',
                                'mid360_link',
                                'right_shoulder_pitch_link',
                                'right_shoulder_roll_link',
                                'right_shoulder_yaw_link',
                                'right_elbow_link',
                                'right_rubber_hand'
            ])

        }
    )

