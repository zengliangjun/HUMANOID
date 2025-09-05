from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclabex.envs.mdp.rewards.joint import statisticsv4_pos


@configclass
class RewardsUperCfg():
    # shoulderp
    rew_shoulderp_mean = RewardTermCfg(
        func=statisticsv4_pos.rew_mean_self2,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint"]),
                "pos_statistics_name": "pos",
                "symmetry": True,
                "constraint_range": 0.1,
                "std": 0.12,
                }
    )
    rew_shoulderp_var = RewardTermCfg(
        func=statisticsv4_pos.rew_variance,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint"
                        ]),
                "pos_statistics_name": "pos",
                "symmetry": True,
                "constraint_range": 0.04,
                "std": 0.075
                }
    )

    # shoulderr
    rew_shoulderr_mean = RewardTermCfg(
        func=statisticsv4_pos.rew_mean_zero2,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_roll_joint",  "right_shoulder_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "symmetry": False,
                "std": 0.05
                }
    )
    rew_shoulderr_var = RewardTermCfg(
        func=statisticsv4_pos.rew_variance,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_roll_joint",  "right_shoulder_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "symmetry": True,
                "iszero": True,
                "constraint_range": None,
                "std": 0.0025
                }
    )

    # shouldery
    rew_shouldery_mean_zero = RewardTermCfg(
        func=statisticsv4_pos.rew_mean_zero2,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_yaw_joint",   "right_shoulder_yaw_joint",
                                ]),
                "pos_statistics_name": "pos",
                "symmetry": True,
                "std": 0.05
                }
    )

    rew_shouldery_var = RewardTermCfg(
        func=statisticsv4_pos.rew_variance,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_yaw_joint",   "right_shoulder_yaw_joint",
                                ]),
                "pos_statistics_name": "pos",
                "symmetry": True,
                "iszero": True,
                "constraint_range": None,
                "std": 0.0025
                }
    )

    # elbow
    rew_elbow_mean = RewardTermCfg(
        func=statisticsv4_pos.rew_mean_self2,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_elbow_joint",
                        "right_elbow_joint"]),
                "pos_statistics_name": "pos",
                "symmetry": True,
                "constraint_range": 0.04,
                "std": 0.12,
                }
    )

    rew_elbow_var = RewardTermCfg(
        func=statisticsv4_pos.rew_variance,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_elbow_joint",
                        "right_elbow_joint"
                        ]),
                "pos_statistics_name": "pos",
                "symmetry": True,
                "iszero": False,
                "constraint_range": 0.0025,
                }
    )

    # waisty
    rew_waisty_mean_zero = RewardTermCfg(
        func=statisticsv4_pos.rew_mean_zero2,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "waist_yaw_joint"
                                ]),
                "pos_statistics_name": "pos",
                "symmetry": False,
                "std": 0.05
                }
    )
    rew_waisty_var = RewardTermCfg(
        func=statisticsv4_pos.rew_variance,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "waist_yaw_joint"
                                ]),
                "pos_statistics_name": "pos",
                "symmetry": False,
                "std": 0.02
                },
    )
