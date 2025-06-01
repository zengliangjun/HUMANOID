from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

def rew_joint_pos_withrefpose(
    env: ManagerBasedRLEnv,  # RL环境实例
    asset_cfg: SceneEntityCfg,  # 资产配置，包括名称与关节ID列表
    phase_name: str,  # 阶段命令名称，用于获取参考关节位置信息
) -> torch.Tensor:
    """
    计算关节位置奖励：
        - 使用当前关节位置信息与默认姿态计算差值
        - 使用参考关节位置进一步计算奖励值

    参数:
        env (ManagerBasedRLEnv): RL环境实例，包含场景和命令管理器。
        asset_cfg (SceneEntityCfg): 资产配置，指定资产名称和关节ID。
        phase_name (str): 阶段命令名称，用于获取参考关节目标位置。

    返回:
        torch.Tensor: 计算得到的奖励张量。
    """
    # 获取对应的机械臂（Articulation）实例
    asset: Articulation = env.scene[asset_cfg.name]

    # 当前关节位置
    pose = asset.data.joint_pos[:, asset_cfg.joint_ids]
    # 默认关节位置
    default_pose = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    # 计算当前姿态与默认姿态的差值
    diff_pose = pose - default_pose

    # 通过阶段名称获取对应的参考指令
    phase_command = env.command_manager.get_term(phase_name)
    ref_dof_pos = phase_command.ref_dof_pos  # 参考关节位置

    # 计算当前差值与参考差值之间的误差
    diff = diff_pose - ref_dof_pos
    norm_diff = torch.norm(diff, dim=1)  # 计算diff的范数

    # 根据误差计算奖励值: 奖励由指数衰减项和惩罚项组成
    r = torch.exp(-2 * norm_diff) - 0.2 * norm_diff.clamp(0, 0.5)
    return r
