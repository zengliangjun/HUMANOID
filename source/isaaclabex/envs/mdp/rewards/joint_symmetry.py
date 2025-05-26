
from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

class PoseMeanVariance(ManagerTermBase):

    '''
    rew_pos_variance = RewardTermCfg(
            func=PoseMeanVariance,
            weight=1.0,
            params={"asset_cfg": SceneEntityCfg("robot",
                                joint_names=["left_hip_pitch_joint",
                                             "right_hip_pitch_joint",
                                             "left_knee_joint",
                                             "right_knee_joint" ]),
                    "weight": []},
        )
    '''

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        joint_size = len(cfg.params["asset_cfg"].joint_ids)

        self.variance_buf = torch.zeros((self.num_envs, joint_size), device=self.device, dtype=torch.float)
        self.mean_buf = torch.zeros((self.num_envs, joint_size), device=self.device, dtype=torch.float)

        self.mean_std = torch.tensor(cfg.params["mean_std"], device=self.device, dtype=torch.float)[None, :]
        self.variance_std = torch.tensor(cfg.params["variance_std"], device=self.device, dtype=torch.float)[None, :]
        self.mean_weight = torch.tensor(cfg.params["mean_weight"], device=self.device, dtype=torch.float)[None, :]
        self.variance_weight = torch.tensor(cfg.params["variance_weight"], device=self.device, dtype=torch.float)[None, :]

        self.threshold = cfg.params["episode_length_weight"]

        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if len(env_ids) == 0:
            return
        self.variance_buf[env_ids] = 0
        self.mean_buf[env_ids] = 0

    def __call__(
        self,
        asset_cfg: SceneEntityCfg,
        env: ManagerBasedRLEnv,
        mean_std,
        variance_std,
        mean_weight,
        variance_weight,
        episode_length_weight
    ) -> torch.Tensor:

        pose = self.asset.data.joint_pos[:, asset_cfg.joint_ids]
        default_pose = self.asset.data.default_joint_pos[:, asset_cfg.joint_ids]

        diff = pose - default_pose
        mean = self.mean_buf.clone()

        self.mean_buf = (mean * (self.env.episode_length_buf[:, None] - 1) + diff) / self.env.episode_length_buf[:, None]
        self.variance_buf = self.variance_buf + (diff - mean) * (diff - self.mean_buf)

        zero_flag = self.env.episode_length_buf == 0
        self.mean_buf[zero_flag] = 0
        self.variance_buf[zero_flag] = 0

        diff_mean = self.mean_buf[:, ::2] - self.mean_buf[:, 1::2]
        diff_variance = self.variance_buf[:, ::2] - self.variance_buf[:, 1::2]

        reward_mean = torch.mean(torch.exp(-torch.abs(diff_mean) / self.mean_std), dim = -1)
        reward_variance = torch.sum(torch.exp(-torch.abs(diff_variance) / self.variance_std), dim = -1)

        reward = reward_mean * self.mean_weight  + reward_variance * self.variance_weight

        novalid_flag = self.env.episode_length_buf < self.threshold
        reward[novalid_flag] = 0
        return reward

'''
rew_hip_knee_pitch_total2zero = RewardTermCfg(
        func=rew_hip_knee_pitch_total2zero,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot",
                            joint_names=["left_hip_pitch_joint",
                                        "left_knee_joint",
                                        "right_hip_pitch_joint",
                                        "right_knee_joint"
                                        ]),
                "weight": [1, 0.5]},
    )
'''

def rew_hip_knee_pitch_total2zero(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    weight
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    pose = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pose = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    diff = pose - default_pose

    total = diff[:, ::2] * weight[0] + diff[:, 1::2] * weight[1]
    total = torch.mean(torch.abs(total), dim = -1)
    reward = torch.exp(-total)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

'''
rew_left_right_total2zero = RewardTermCfg(
        func=rew_left_right_total2zero,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot",
                            joint_names=["left_hip_pitch_joint",
                                        "right_hip_pitch_joint",
                                        "left_knee_joint",
                                        "right_knee_joint"
                                        ])},
    )
'''

def rew_left_right_total2zero(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:

    return rew_hip_knee_pitch_total2zero(env, asset_cfg, command_name, [1, 1])
