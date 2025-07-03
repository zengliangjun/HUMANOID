from __future__ import annotations
import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Union
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
    from isaaclabex.envs.rl_env_exts import ManagerBasedRLEnv_Extends
    from isaaclab.managers.manager_base import ManagerBase, ManagerTermBaseCfg
    from isaaclab.managers import CurriculumTermCfg

class curriculum_with_degree(ManagerTermBase):
    """
    基于degree参数的课程学习管理器

    根据环境表现动态调整配置参数:
    - 当episode_length < down_up_lengths[0]: 参数值减少(1 - degree)倍
    - 当episode_length > down_up_lengths[0]: 参数值增加(1 + degree)倍
    调整后的值会被限制在value_range范围内
    """

    _env: ManagerBasedRLEnv_Extends

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv_Extends):
        """初始化课程学习管理器

        参数:
            cfg: 配置对象，必须包含以下参数:
                - degree: 调整系数
                - down_up_lengths: 触发调整的episode长度阈值
                - value_range: 参数值允许范围
                - manager_name: 目标管理器名称
                - term_name: 目标term名称
                - param_name: 要调整的参数名
            env: 环境实例
        """
        super().__init__(cfg, env)
        # 验证必须参数
        assert "degree" in cfg.params
        assert "down_up_lengths" in cfg.params
        assert "value_range" in cfg.params
        assert "manager_name" in cfg.params
        assert "term_name" in cfg.params
        assert "param_name" in cfg.params

        # 保存配置参数
        self.degree = cfg.params["degree"]  # 调整系数
        self.down_up_lengths = cfg.params["down_up_lengths"]  # 长度阈值
        self.value_range = cfg.params["value_range"]  # 值范围

        # 获取目标配置对象
        name = cfg.params["manager_name"]
        manager: ManagerBase = env.__getattribute__(f"{name}_manager")
        self.term_cfg: ManagerTermBaseCfg = manager.get_term_cfg(cfg.params["term_name"])
        self.param_name = cfg.params["param_name"]  # 要调整的参数名

        # 确定参数位置(属性或params字典中)
        if hasattr(self.term_cfg, self.param_name):
            self.is_attr = True  # 参数是类属性
            self.cur_value = getattr(self.term_cfg, self.param_name)
        elif self.param_name in self.term_cfg.params:
            self.is_attr = False  # 参数在params字典中
            self.cur_value = self.term_cfg.params[self.param_name]
        else:
            raise Exception(f"未知参数: {self.param_name}")

    def reset(self, env_ids: Sequence[int] | None = None) -> None:

        update = True
        # 根据episode长度决定调整方向
        if self._env.average_episode_length < self.down_up_lengths[0]:
            value = (1 - self.degree) * self.cur_value  # 减少参数值
        elif self._env.average_episode_length > self.down_up_lengths[1]:
            value = (1 + self.degree) * self.cur_value  # 增加参数值
        else:
            update = False  # 不调整

        if update:
            # 限制值在有效范围内
            value = np.clip(value, self.value_range[0], self.value_range[1])
            value = float(value)

            # 如果值有变化，更新配置
            if value != self.cur_value:
                if self.is_attr:
                    setattr(self.term_cfg, self.param_name, value)  # 更新属性
                else:
                    self.term_cfg.params[self.param_name] = value  # 更新字典

                self.cur_value = value  # 更新当前值

    def __call__(
        self,
        env: ManagerBasedRLEnv_Extends,
        env_ids: Sequence[int],
        degree: float,
        down_up_lengths: Union[list, tuple],
        value_range: Union[list, tuple],
        manager_name: str,
        term_name: str,
        param_name: str
    ) -> torch.Tensor:
        """
        执行参数调整逻辑

        根据当前episode长度决定是否调整参数值:
        - 小于阈值: 减少参数值
        - 大于阈值: 增加参数值
        - 在阈值范围内: 不调整

        返回:
            调整后的参数值(tensor)
        """
        return torch.tensor(self.cur_value, device=self._env.device)


class scale_with_degree(ManagerTermBase):

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

            param_name = dicts["param_name"]
            value = dicts[f"start_{param_name}"] + \
                        (dicts[f"end_{param_name}"] - dicts[f"start_{param_name}"]) * scale

            value = float(value)

            if hasattr(term_cfg, param_name):
                setattr(term_cfg, param_name, value)
            elif param_name in term_cfg.params:
                term_cfg.params[param_name] = value
            else:
                raise Exception(f"未知参数: {param_name}")

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
