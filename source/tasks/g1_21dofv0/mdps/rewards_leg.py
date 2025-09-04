from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclabex.envs.mdp.rewards import reward_collect


@configclass
class RewardsLegCfg():
    rew_legp_total2zero = RewardTermCfg(
        func=reward_collect.rew_pitch_total2zero,
        weight=0.2,
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
    # hipp
    rew_hipp_mean = RewardTermCfg(
        func=reward_collect.rew_mean_self,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint"]),
                "pos_statistics_name": "pos",
                "std": 0.3,
                "diff_scale": 1
                }
    )
    rew_hipp_var = RewardTermCfg(
        func=reward_collect.rew_variance_self,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                        ]),
                "pos_statistics_name": "pos",
                "std": 0.09,
                "diff_scale": 1,
                }
    )

    # knee
    rew_knee_mean = RewardTermCfg(
        func=reward_collect.rew_mean_self,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_knee_joint",
                        "right_knee_joint"]),
                "pos_statistics_name": "pos",
                "std": 0.3,
                "diff_scale": 1
                }
    )
    rew_knee_var = RewardTermCfg(
        func=reward_collect.rew_variance_self,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_knee_joint",
                        "right_knee_joint"
                        ]),
                "pos_statistics_name": "pos",
                "std": 0.09,
                "diff_scale": 1,
                }
    )
    # anklep
    rew_anklep_mean = RewardTermCfg(
        func=reward_collect.rew_mean_self,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_ankle_pitch_joint",
                        "right_ankle_pitch_joint"]),
                "pos_statistics_name": "pos",
                "std": 0.3,
                "diff_scale": 1
                }
    )
    rew_anklep_var = RewardTermCfg(
        func=reward_collect.rew_variance_self,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_ankle_pitch_joint",
                        "right_ankle_pitch_joint"
                        ]),
                "pos_statistics_name": "pos",
                "std": 0.09,
                "diff_scale": 1,
                }
    )

    #################
    # hipr
    rew_hipr_mean_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_roll_joint",  "right_hip_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.3
                }
    )
    rew_hipr_var_zero = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_roll_joint",  "right_hip_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.04
                },
    )
    rew_hipr_rp = RewardTermCfg(
        func=reward_collect.reward_penalize_joint,
        weight=0.2,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[ ".*_hip_roll_joint"]),
                "diff_range": 0.15,
                "diff_std": 0.15,
                "penalize_weight": - 0.5
                },
    )
    # hipy
    rew_hipy_mean_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_yaw_joint",   "right_hip_yaw_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.3
                }
    )
    rew_hipy_var_zero = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_yaw_joint",   "right_hip_yaw_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.04
                },
    )
    rew_hipy_rp = RewardTermCfg(
        func=reward_collect.reward_penalize_joint,
        weight=0.2,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[ ".*_hip_yaw_joint"]),
                "diff_range": 0.15,
                "diff_std": 0.15,
                "penalize_weight": - 0.5
                },
    )

    # ankler
    rew_ankler_mean_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_ankle_roll_joint","right_ankle_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.3
                }
    )
    rew_ankler_var = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_ankle_roll_joint","right_ankle_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.04
                },
    )