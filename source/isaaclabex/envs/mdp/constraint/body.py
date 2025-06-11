
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def constraint_width(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_width: float = 0.2
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    # 提取机器人中指定 body_ids 的3D位置（世界坐标）
    pos = asset.data.body_pos_w[:, asset_cfg.body_ids]

    # 重复根链接的旋转四元数，使其与 pos 的维度匹配
    # 即将根链接的四元数应用到每个 body 上
    quat_w = torch.repeat_interleave(asset.data.root_link_quat_w[:, None, :], pos.shape[1], dim=1)

    # 对提取的位置进行逆旋转转换，将世界坐标系位置转换到机器人基座坐标系
    pos_b = math_utils.quat_rotate_inverse(quat_w, pos)

    # width 计算：
    # 1. 从基座坐标系 pos_b 中提取 y 轴坐标（索引 1 表示 y 分量）
    # 2. 假定 body 按照成对顺序排列，其中偶数索引和奇数索引代表一对
    # 3. 计算每一对中两个身体的 y 坐标差的绝对值，得到当前宽度
    # 4. 减去目标宽度 target_width，得到超出或不足目标的偏差
    error = (pos_b[:, 0::2, 1] - pos_b[:, 1::2, 1]) - target_width
    flag_negative =  error < 0
    error[flag_negative] = error[flag_negative] * 3  # 确保负值被置为 0，避免不必要的惩罚
    error = torch.abs(error)  # 将偏差放大（乘以 100）
    error[error < target_width] = 0  # 将小于 4 的偏差置为 0

    # 将偏差放大（乘以 100），计算其指数惩罚，最后求所有对的平均值作为最终 reward
    return torch.sum(error.float(), dim=-1)
