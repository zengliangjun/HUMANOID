from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclabex.envs.mdp.rewards import reward_collect
from isaaclabex.envs.mdp.rewards.joint import statisticsv4_pos

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
        func=statisticsv4_pos.rew_mean_self2,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint"]),
                "pos_statistics_name": "pos",
                "symmetry": True,
                "constraint_range": 0.1,
                "std": 0.15,
                }
    )
    rew_hipp_var = RewardTermCfg(
        func=statisticsv4_pos.rew_variance,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                        ]),
                "pos_statistics_name": "pos",
                "symmetry": True,
                "iszero": False,
                "constraint_range": 0.05,
                "std": 0.09,
                }
    )

    # knee
    rew_knee_mean = RewardTermCfg(
        func=statisticsv4_pos.rew_mean_self2,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_knee_joint",
                        "right_knee_joint"]),
                "pos_statistics_name": "pos",
                "symmetry": True,
                "constraint_range": 0.1,
                "std": 0.15,
                }
    )
    rew_knee_var = RewardTermCfg(
        func=statisticsv4_pos.rew_variance,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_knee_joint",
                        "right_knee_joint"
                        ]),
                "pos_statistics_name": "pos",
                "symmetry": True,
                "iszero": False,
                "constraint_range": 0.05,
                "std": 0.09,
                }
    )
    # anklep
    rew_anklep_mean = RewardTermCfg(
        func=statisticsv4_pos.rew_mean_self2,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_ankle_pitch_joint",
                        "right_ankle_pitch_joint"]),
                "pos_statistics_name": "pos",
                "symmetry": True,
                "constraint_range": 0.05,
                "std": 0.15
                }
    )
    rew_anklep_var = RewardTermCfg(
        func=statisticsv4_pos.rew_variance,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_ankle_pitch_joint",
                        "right_ankle_pitch_joint"
                        ]),
                "pos_statistics_name": "pos",
                "symmetry": True,
                "iszero": False,
                "constraint_range": 0.04,
                "std": 0.09,
                }
    )

    #################
    # hipr
    rew_hipr_mean_zero = RewardTermCfg(
        func=statisticsv4_pos.rew_mean_zero2,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_roll_joint",  "right_hip_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "symmetry": True,
                "std": 0.15
                }
    )
    rew_hipr_var_zero = RewardTermCfg(
        func=statisticsv4_pos.rew_variance,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_roll_joint",  "right_hip_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "symmetry": True,
                "iszero": True,
                "constraint_range": None,
                "std": 0.04
                },
    )
    # hipy
    rew_hipy_mean_zero = RewardTermCfg(
        func=statisticsv4_pos.rew_mean_zero2,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_yaw_joint",   "right_hip_yaw_joint",
                                ]),
                "pos_statistics_name": "pos",
                "symmetry": True,
                "std": 0.15
                }
    )
    rew_hipy_var_zero = RewardTermCfg(
        func=statisticsv4_pos.rew_variance,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_yaw_joint",   "right_hip_yaw_joint",
                                ]),
                "pos_statistics_name": "pos",
                "symmetry": True,
                "iszero": True,
                "constraint_range": None,
                "std": 0.04
                },
    )

    # ankler
    rew_ankler_mean_zero = RewardTermCfg(
        func=statisticsv4_pos.rew_mean_zero2,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_ankle_roll_joint","right_ankle_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "symmetry": True,
                "std": 0.15
                }
    )
    rew_ankler_var = RewardTermCfg(
        func=statisticsv4_pos.rew_variance,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_ankle_roll_joint","right_ankle_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "symmetry": True,
                "iszero": True,
                "constraint_range": None,
                "std": 0.04
                },
    )
