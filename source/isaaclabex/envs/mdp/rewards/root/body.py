from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
import isaaclab.utils.math as math_utils
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reward_distance(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
    max_threshold: float = 0.2,
    min_threshold: float = 0.3):
    """
    Calculates the reward based on the distance between the feet.
    Penalizes if the feet are too close or too far from each other.

    Parameters:
        env (ManagerBasedRLEnv): The environment containing scene and asset data.
        asset_cfg (SceneEntityCfg): Configuration for the scene entity including name and body identifiers.
        max_threshold (float): Upper distance threshold; distances above are penalized. Default is 0.2.
        min_threshold (float): Lower distance threshold; distances below are penalized. Default is 0.3.

    Returns:
        torch.Tensor: Computed reward tensor based on the feet distance.
    """
    # Retrieve the articulation asset from the scene using the asset configuration name.
    asset: Articulation = env.scene[asset_cfg.name]

    # Extract the 2D positions (x, y) of the bodies based on provided body_ids.
    pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]

    # Compute Euclidean distance between paired body positions (assumes feet are ordered alternately).
    pos_dist = torch.norm(pos[:, ::2] - pos[:, 1::2], dim=1)

    # Compute a penalty for distances below the minimum threshold.
    d_min = torch.clamp(pos_dist - min_threshold, -0.5, 0.)
    # Compute a penalty for distances above the maximum threshold.
    d_max = torch.clamp(pos_dist - max_threshold, 0, 0.5)

    # Calculate and return the final reward as the average penalty over the two conditions.
    return torch.sum((torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2, dim=-1)

def reward_width(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
    target_width: float = 0.2):
    """
    计算基于机器人身体宽度偏差的 reward。

    参数:
        env (ManagerBasedRLEnv): 环境对象，包含场景和资产数据。
        asset_cfg (SceneEntityCfg): 包含资产名称和需要计算的 body_id 的配置。
        target_width (float): 希望达到的目标宽度（默认值为 0.2）。

    返回:
        torch.Tensor: 基于宽度偏差计算出的 reward 张量（求平均值）。
    """

    # 从环境中获取指定资产对象（机器人）
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
    width = (pos_b[:, 0::2, 1] - pos_b[:, 1::2, 1] - target_width) * 100
    width[width < 0] = torch.square(width[width < 0] * 2)  # 确保负值被置为 0，避免不必要的惩罚
    width[width < 4] = 0

    # 将偏差放大（乘以 100），计算其指数惩罚，最后求所有对的平均值作为最终 reward
    return torch.mean(torch.exp(-width / 6), dim=-1)


def penalize_width(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
    target_width: float = 0.2):

    # 从环境中获取指定资产对象（机器人）
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
    penalize = (pos_b[:, 0::2, 1] - pos_b[:, 1::2, 1]) < target_width * 0.95

    # 将偏差放大（乘以 100），计算其指数惩罚，最后求所有对的平均值作为最终 reward
    return torch.sum(penalize.float(), dim=-1)
