'''
**Constraint Manager**: [Constraints as Termination (CaT)](https://arxiv.org/abs/2403.18765) method implementation added.
'''

from __future__ import annotations
import torch

from isaaclab.utils import configclass
from dataclasses import MISSING
from typing import Callable

from typing import TYPE_CHECKING, Any
from isaaclab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

if TYPE_CHECKING:
    from .motions_manager import MotionsTerm

@configclass
class MotionsTermCfg(ManagerTermBaseCfg):
    # 运动管理器配置

    func: MotionsTerm = MISSING  # 运动函数

    step_dt: float = MISSING  # 步长

    assert_cfg: SceneEntityCfg = MISSING  # 资产配置

    motion_file: str = MISSING  # 运动文件路径

    random_sample: bool = True  # 是否随机采样

    resample_interval_s: float = -1  # 重新采样间隔， -1 表示不调整

    params: dict[str, Any] = MISSING  # 其他参数

    debug_vis: bool = False
    """Whether to visualize debug information. Defaults to False."""
