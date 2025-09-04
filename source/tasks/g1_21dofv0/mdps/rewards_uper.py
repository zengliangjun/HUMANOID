from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclabex.envs.mdp.rewards import reward_collect


@configclass
class RewardsUperCfg():
    # shoulderp
    rew_shoulderp_mean = RewardTermCfg(
        func=reward_collect.rew_mean_self,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint"]),
                "pos_statistics_name": "pos",
                "std": 0.15,
                "diff_scale": 1
                }
    )
    rew_shoulderp_var = RewardTermCfg(
        func=reward_collect.rew_variance_self,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint"
                        ]),
                "pos_statistics_name": "pos",
                "std": 0.075,
                "diff_scale": 1,
                }
    )

    # shoulderr
    rew_shoulderr_mean = RewardTermCfg(
        func=reward_collect.rew_mean_default,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_roll_joint",  "right_shoulder_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.05
                }
    )
    rew_shoulderr_var = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_roll_joint",  "right_shoulder_roll_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.075
                }
    )
    rew_shoulderr_rp = RewardTermCfg(
        func=reward_collect.reward_penalize_joint,
        weight=0.2,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[ ".*_shoulder_roll_joint"]),
                "diff_range": 0.05,
                "diff_std": 0.1,
                "penalize_weight": - 0.5
                },
    )

    # shouldery
    rew_shouldery_mean_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_yaw_joint",   "right_shoulder_yaw_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.15
                }
    )

    rew_shouldery_var = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_yaw_joint",   "right_shoulder_yaw_joint",
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.075
                }
    )

    rew_shouldery_rp = RewardTermCfg(
        func=reward_collect.reward_penalize_joint,
        weight=0.2,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[ ".*_shoulder_yaw_joint"]),
                "diff_range": 0.05,
                "diff_std": 0.1,
                "penalize_weight": - 0.5
                },
    )

    # elbow
    rew_elbow_mean = RewardTermCfg(
        func=reward_collect.rew_mean_self,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_elbow_joint",
                        "right_elbow_joint"]),
                "pos_statistics_name": "pos",
                "std": 0.15,
                "diff_scale": 1
                }
    )

    rew_elbow_var = RewardTermCfg(
        func=reward_collect.rew_variance_zero,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_elbow_joint",
                        "right_elbow_joint"
                        ]),
                "pos_statistics_name": "pos",
                "std": 0.075
                }
    )

    rew_elbow_rp = RewardTermCfg(
        func=reward_collect.reward_penalize_joint,
        weight=0.2,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[ ".*_elbow_joint"]),
                "diff_range": 0.15,
                "diff_std": 0.15,
                "penalize_weight": - 0.5
                },
    )

    # waisty
    rew_waisty_mean_zero = RewardTermCfg(
        func=reward_collect.rew_mean_zero_nosymmetry,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "waist_yaw_joint"
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.15
                }
    )
    rew_waisty_var = RewardTermCfg(
        func=reward_collect.rew_variance_zero_nosymmetry,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "waist_yaw_joint"
                                ]),
                "pos_statistics_name": "pos",
                "std": 0.075
                },
    )

    rew_waisty_rp = RewardTermCfg(
        func=reward_collect.reward_penalize_joint,
        weight=0.2,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[ "waist_yaw_joint"]),
                "diff_range": 0.2,
                "diff_std": 0.15,
                "penalize_weight": - 0.5
                },
    )

