
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclabex.envs.mdp.rewards import reward_collect, pbrs_collect, pbrs_base

@configclass
class RewardsCfg:
    # -- task
    rew_lin_xy_exp = RewardTermCfg(
        func=reward_collect.reward_lin_xy_exp,
        weight=10.0,
        params={"std": 0.5,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    rew_ang_z_exp = RewardTermCfg(
        func=reward_collect.reward_ang_z_exp,
        weight=5,
        params={"std": 0.3,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    # action
    p_action_rate = RewardTermCfg(
        func=reward_collect.penalize_action_rate_l2, weight=-0.01)
    # action
    p_action_smoothness = RewardTermCfg(
        func=reward_collect.penalize_action_smoothness,
        weight=-0.002,
        params={
            "weight1": 1,
            "weight2": 1,
            "weight3": 0.05,
            },
    )
    p_torques = RewardTermCfg(
        func=reward_collect.penalize_torques_l2,
        weight=-0.0001, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    p_torque_limits = RewardTermCfg(
        func=reward_collect.penalize_torque_limits,
        weight=-1e-2, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    p_pos_limits = RewardTermCfg(
        func=reward_collect.penalize_jpos_limits_l1,
        weight=-10.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    # body
    pbrs_orientation = RewardTermCfg(
        func=pbrs_collect.ori_l2_pbrs,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot"),
                "sigma": 0.025, #0.5,
                "gamma": 1,
                "method": pbrs_base.PBRSExp}
    )
    pbrs_height = RewardTermCfg(
        func=pbrs_collect.height_flat_or_rayl2_pbrs,
        weight=1.0,
        params={
            "target_height": 0.78 + 0.035,  # Adjusting for the foot clearance
            "asset_cfg": SceneEntityCfg("robot"),
            "sigma": 0.5 * (0.01) ** 2, # 0.5 * (0.65) ** 2,
            "gamma": 1,
            "method": pbrs_base.PBRSExp}
    )
    pbrs_width = RewardTermCfg(
        func=pbrs_collect.width_exp_pbrs,
        weight=1,
        params={
            "target_width": 0.238,  # Adjusting for the foot clearance
            "asset_cfg": SceneEntityCfg("robot",
                         body_names=[".*left_ankle_roll_link",
                                     ".*right_ankle_roll_link",
                                     ".*left_knee_link",
                                     ".*right_knee_link"]),
            "sigma": 0.1,
            "gamma": 1,
            "method": pbrs_base.PBRSNormal
            },
    )

    pbrs_hipry = RewardTermCfg(
        func=pbrs_collect.jpos_deviation_l1_pbrs,
        weight=1.0,
        params={"asset_cfg":
                SceneEntityCfg("robot", joint_names=[ ".*_hip_roll_joint", ".*_hip_yaw_joint"]),
                "sigma": 0.5, # 0.25,
                "gamma": 1,
                "method": pbrs_base.PBRSExp},
    )
    pbrs_hipp = RewardTermCfg(
        func=pbrs_collect.jpos_deviation_l1_pbrs,
        weight=1.0,
        params={"asset_cfg":
                SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint"]),
                "sigma": 0.7, # 0.25,
                "gamma": 1,
                "method": pbrs_base.PBRSExp},
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
    pbrs_total2zero = RewardTermCfg(
        func=pbrs_collect.total2zero_pbrs,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot",
                                        joint_names=[
                                                     "left_hip_pitch_joint",
                                                     "right_hip_pitch_joint",
                                        ]),
            "sigma": 0.25,
            "gamma": 1,
            "method": pbrs_base.PBRSNormal},
    )
    pbrs_hip_roll = RewardTermCfg(
        func=pbrs_collect.total2zero_one_way_pbrs,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot",
                                        joint_names=[
                                                    "left_hip_roll_joint",
                                                    "right_hip_roll_joint",
                                        ]),
            "sigma": 0.25,
            "gamma": 1,
            "method": pbrs_base.PBRSNormal},
    )

    pbrs_equals = RewardTermCfg(
        func=pbrs_collect.equals_pbrs,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot",
                                        joint_names=[
                                                    "left_hip_yaw_joint",
                                                    "right_hip_yaw_joint",
                                        ]),
            "sigma": 0.25,
            "gamma": 1,
            "method": pbrs_base.PBRSNormal},
    )
    hipp_meanvar_pbrs = RewardTermCfg(
        func=pbrs_collect.meanvar_pbrs,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot",
                                        joint_names=[
                                                     "left_hip_pitch_joint",
                                                     "right_hip_pitch_joint",
                                        ]),
            "mean_std": 0.15,
            "variance_std": 0.05,
            "mean_weight": 1,
            "variance_weight": 1,
            "episode_length_threshold": 0,
            "gamma": 1,
            "method": pbrs_base.PBRSNormal},
    )
    knee_meanvar_pbrs = RewardTermCfg(
        func=pbrs_collect.meanvar_pbrs,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot",
                                        joint_names=[
                                                     "left_knee_joint",
                                                     "right_knee_joint",
                                        ]),
            "mean_std": 0.15,
            "variance_std": 0.05,
            "mean_weight": 1,
            "variance_weight": 1,
            "episode_length_threshold": 0,
            "gamma": 1,
            "method": pbrs_base.PBRSNormal},
    )
