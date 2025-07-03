from __future__ import annotations
import torch
import numpy as np
from collections.abc import Sequence
from isaaclab.utils import configclass
from typing import TYPE_CHECKING, Union
from dataclasses import MISSING
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
    from isaaclabex.envs.rl_env_exts import ManagerBasedRLEnv_Extends
    from isaaclab.managers.manager_base import ManagerBase, ManagerTermBaseCfg
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg, CurriculumTermCfg

@configclass
class EventCurriculumStepItem:
    #start_steps: int = MISSING
    #end_steps: int = MISSING

    start_range: tuple[float, float] = MISSING
    end_range: tuple[float, float] = MISSING

def _calcute_curriculum_step(_scale, cfg: EventTermCfg, curriculum_dicts: dict):
    for curriculum_name, subitem in curriculum_dicts.items():
        if 0 == _scale:
            cfg.params[curriculum_name] = subitem.start_range
        elif 1 == _scale:
            cfg.params[curriculum_name] = subitem.end_range
        else:
            if isinstance(subitem.start_range, dict):

                for key, start_range, in subitem.start_range.items():
                    end_range = subitem.end_range[key]
                    cfg.params[curriculum_name][key] = (
                    float((end_range[0] - start_range[0]) * _scale + start_range[0]),
                    float((end_range[1] - start_range[1]) * _scale + start_range[1]),
                )

            else:
                cfg.params[curriculum_name] = (
                    float((subitem.end_range[0] - subitem.start_range[0]) * _scale + subitem.start_range[0]),
                    float((subitem.end_range[1] - subitem.start_range[1]) * _scale + subitem.start_range[1]),
                )


def curriculum_with_steps(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],

    start_steps,
    end_steps,
    curriculums
) -> torch.Tensor:

    steps = env.common_step_counter
    if start_steps > steps:
        scale = 0
    elif end_steps < steps:
        scale = 1
    else:
        scale = (steps - start_steps) / (end_steps - start_steps)

    event_manager = env.event_manager
    for event_name, curriculum_dicts in curriculums.items():
        cfg: EventTermCfg = event_manager.get_term_cfg(event_name)

        _calcute_curriculum_step(scale, cfg, curriculum_dicts)

    return torch.tensor(scale, device=env.device)


class range_with_degree(ManagerTermBase):

    _env: ManagerBasedRLEnv_Extends

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv_Extends):
        """初始化课程学习管理器

        参数:
            cfg: 配置对象，必须包含以下参数:
                - degree: 调整系数
                - down_up_lengths: 触发调整的episode长度阈值
                - scale_range
                - manager_name: 目标管理器名称
                - curriculums: list
            env: 环境实例
        """
        super().__init__(cfg, env)
        # 验证必须参数
        assert "degree" in cfg.params
        assert "down_up_lengths" in cfg.params
        assert "scale_range" in cfg.params
        assert "manager_name" in cfg.params
        assert "curriculums" in cfg.params

        # 保存配置参数
        self.degree = cfg.params["degree"]  # 调整系数
        self.down_up_lengths = cfg.params["down_up_lengths"]  # 长度阈值
        self.scale_range = cfg.params["scale_range"]
        self.curriculums = cfg.params["curriculums"]

        # 获取目标配置对象
        name = cfg.params["manager_name"]
        self.manager: ManagerBase = env.__getattribute__(f"{name}_manager")

        self.scale = cfg.params["scale"]
        self._update(self.scale)

    def _update(self, scale):
        for name, dicts in self.curriculums.items():
            term_cfg: ManagerTermBaseCfg = self.manager.get_term_cfg(name)
            _calcute_curriculum_step(scale, term_cfg, dicts)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:

        update = True

        scale = self.scale
        # 根据episode长度决定调整方向
        if self._env.average_episode_length < self.down_up_lengths[0]:
            scale -= self.degree
        elif self._env.average_episode_length > self.down_up_lengths[1]:
            scale += self.degree  # 增加参数值
        else:
            update = False  # 不调整

        if update:
            # 限制值在有效范围内
            scale = np.clip(scale, self.scale_range[0], self.scale_range[1])

            # 如果值有变化，更新配置
            if scale != self.scale:
                self._update(scale)

                self.scale = scale  # 更新当前值

    def __call__(
        self,
        env: ManagerBasedRLEnv_Extends,
        env_ids: Sequence[int],
        degree: float,
        down_up_lengths: Union[list, tuple],
        scale_range: Union[list, tuple],
        manager_name: str,
        curriculums: dict,
        scale: float = 0,
    ) -> torch.Tensor:

        return torch.tensor(self.scale, device=self._env.device)
