from .rewards import RewardsCfg

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclabex.envs.mdp.rewards import reward_collect

@configclass
class HKSymmetryCfg(RewardsCfg):
    symmetry_hip_knee = RewardTermCfg(
        func=reward_collect.reward_hip_knee_symmetry,
        weight=0.1,
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


@configclass
class LRSymmetryCfg(RewardsCfg):
    symmetry_lr_reward = RewardTermCfg(
        func=reward_collect.reward_left_right_symmetry,
        weight=0.1,
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

@configclass
class MeanVRSymmetryCfg(RewardsCfg):
    symmetry_vr_reward = RewardTermCfg(
        func=reward_collect.reward_pose_mean_var_symmetry,
        weight=0.1,
        params={
                "asset_cfg": SceneEntityCfg("robot",
                            joint_names=["left_hip_pitch_joint",
                                        "right_hip_pitch_joint",
                                        "left_knee_joint",
                                        "right_knee_joint"
                                        ],
                            preserve_order = True),
                "mean_std": 0.25,
                "variance_std": 0.125,
                "mean_weight": 0.25,
                "variance_weight": 0.25,
                "episode_length_threshold": 40},
    )

@configclass
class MMVMCfg(RewardsCfg):
    MMVM_reward = RewardTermCfg(
        func=reward_collect.reward_pose_mean_min_var_max,
        weight=0.1,
        params={
                "asset_cfg": SceneEntityCfg("robot",
                            joint_names=["left_hip_pitch_joint",
                                        "right_hip_pitch_joint",
                                        "left_knee_joint",
                                        "right_knee_joint"
                                        ],
                            preserve_order = True),
                "mean_std": 0.25,
                "variance_std": 0.125,
                "mean_weight": 0.25,
                "variance_weight": 0.25,
                "episode_length_threshold": 40},
    )

@configclass
class MMVMCfg2(RewardsCfg):
    MMVM_reward = RewardTermCfg(
        func=reward_collect.reward_pose_mean_min_var_max,
        weight=0.3,
        params={
                "asset_cfg": SceneEntityCfg("robot",
                            joint_names=["left_hip_pitch_joint",
                                        "right_hip_pitch_joint",
                                        "left_knee_joint",
                                        "right_knee_joint"
                                        ],
                            preserve_order = True),
                "mean_std": 0.25,
                "variance_std": 0.125,
                "mean_weight": 0.25,
                "variance_weight": 0.25,
                "episode_length_threshold": 40},
    )
    def __post_init__(self):
        self.penalize_height.target_height = 0.78 + 0.35
