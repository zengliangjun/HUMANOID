from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg

class Stability(ManagerTermBase):

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = self._env.scene[self.asset_cfg.name]


    def reset(self, env_ids: Sequence[int] | None = None):
        # 如果指定环境列表为空则不执行重置
        if len(env_ids) == 0:
            return

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:

        # 提取机器人中指定 body_ids 的3D位置（世界坐标）
        pos = self.asset.data.body_pos_w[:, asset_cfg.body_ids]
        # 重复根链接的旋转四元数，使其与 pos 的维度匹配
        # 即将根链接的四元数应用到每个 body 上
        quat_w = torch.repeat_interleave(self.asset.data.root_link_quat_w[:, None, :], pos.shape[1], dim=1)

        # 对提取的位置进行逆旋转转换，将世界坐标系位置转换到机器人基座坐标系
        pos_b = math_utils.quat_rotate_inverse(quat_w, pos)
        pos = torch.mean(pos_b[:, :, :2], dim=1)
        com_b = math_utils.quat_rotate_inverse(self.asset.data.root_link_quat_w, self.asset.data.root_com_pos_w)

        # 双足支撑稳定性计算 (只有两个支撑点)
        left_ankle = pos_b[:, 0, :2]  # 左脚踝位置(x,y)
        right_ankle = pos_b[:, 1, :2]  # 右脚踝位置(x,y)

        # 1. 计算支撑线向量和法向量
        support_vec = right_ankle - left_ankle
        support_len = torch.linalg.norm(support_vec, dim=1)
        support_dir = support_vec / (support_len.unsqueeze(1) + 1e-6)
        normal_vec = torch.stack([-support_dir[:, 1], support_dir[:, 0]], dim=1)

        # 2. 计算质心在支撑线上的投影
        com_proj = com_b[:, :2] - left_ankle
        com_dist = torch.sum(com_proj * normal_vec, dim=1)  # 垂直距离
        com_along = torch.sum(com_proj * support_dir, dim=1)  # 沿线距离

        # 3. 稳定性判断
        # 投影是否在支撑线范围内 (考虑10%的裕度)
        margin = support_len * 0.1
        stable = (com_along >= -margin) & (com_along <= (support_len + margin))

        # 4. 计算稳定性奖励
        # 距离越小越稳定，使用指数衰减函数
        dist_reward = torch.exp(-2.0 * torch.abs(com_dist))
        range_reward = torch.where(stable, 1.0, 0.2)
        total_reward = dist_reward * range_reward

        return total_reward
