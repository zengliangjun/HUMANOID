from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclabex.envs.managers.term_cfg import StatisticsTermCfg

class StatusBase(ManagerTermBase):

    cfg: StatisticsTermCfg

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        # 初始化StatusBase，加载配置和资产对象，并初始化统计数据缓冲区
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = self._env.scene[asset_cfg.name]
        self.asset_cfg = asset_cfg

        self._init_buffers()

    def _episode_length(self) -> torch.Tensor:
        # 获取episode长度，当cfg中设置了截断时进行最大值截断
        if -1 == self.cfg.episode_truncation:
            return self._env.episode_length_buf
        else:
            return torch.clamp_max(self._env.episode_length_buf, self.cfg.episode_truncation)

    def _calculate_episode(self, diff: torch.Tensor) -> None:
        # 利用增量更新方法计算当前episode的均值和方差
        episode_length_buf = self._episode_length()

        # 计算均值：根据新差值delta0更新均值缓冲区
        delta0 = diff - self.episode_mean_buf
        self.episode_mean_buf += delta0 / episode_length_buf[:, None]

        # 计算方差：利用delta0和新均值计算更新方差缓冲区
        delta1 = diff - self.episode_mean_buf
        self.episode_variance_buf = (
            self.episode_variance_buf * (episode_length_buf[:, None] - 2)
            + delta0 * delta1
        ) / (episode_length_buf[:, None] - 1)

        # 当episode刚开始时重置方差，防止数值异常
        new_episode_mask = episode_length_buf <= 1
        # self.episode_mean_buf[new_episode_mask] = 0
        self.episode_variance_buf[new_episode_mask] = 0


class StatusVel(StatusBase):

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _init_buffers(self):
        # 初始化速度统计的均值与方差缓冲区 (二维数据)
        self.episode_variance_buf = torch.zeros((self.num_envs, 2),
                              device=self.device, dtype=torch.float)
        self.episode_mean_buf = torch.zeros_like(self.episode_variance_buf)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        # 重置指定环境ID的缓冲区，并根据周期导出统计数据
        if env_ids is None or len(env_ids) == 0:
            return {}

        items = {}
        if 0 == self._env.common_step_counter % self.cfg.export_interval:

            mean = self.episode_mean_buf[env_ids]
            items[f"em"] = torch.mean(torch.norm(mean, dim=1))

            variance = self.episode_variance_buf[env_ids]
            items[f"ev"] = torch.mean(torch.norm(variance, dim=1))

        # 重置所有相关缓冲区
        buffers = [
            self.episode_variance_buf,
            self.episode_mean_buf,
        ]

        for buf in buffers:
            buf[env_ids] = 0

        return items

    def __call__(self):
        """执行统计计算：计算机器人根链速度的x和y分量的绝对值"""
        diff = torch.abs(self.asset.data.root_lin_vel_b[:, :2])  # 只取x和y分量
        self._calculate_episode(diff)

class StatusFootHeight(StatusBase):

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _init_buffers(self):
        # 初始化足高统计相关的缓冲区 (一维数据)
        self.episode_variance_buf = torch.zeros((self.num_envs, 1),
                              device=self.device, dtype=torch.float)
        self.episode_mean_buf = torch.zeros_like(self.episode_variance_buf)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        # 重置指定环境的缓冲区，并导出足高相关统计数据
        if env_ids is None or len(env_ids) == 0:
            return {}

        items = {}
        if 0 == self._env.common_step_counter % self.cfg.export_interval:

            mean = self.episode_mean_buf[env_ids]
            items[f"em"] = torch.mean(mean, dim=1)

            variance = self.episode_variance_buf[env_ids]
            items[f"ev"] = torch.mean(variance, dim=1)

        # 清空对应环境的缓冲数据
        buffers = [
            self.episode_variance_buf,
            self.episode_mean_buf,
        ]

        for buf in buffers:
            buf[env_ids] = 0

        return items

    def __call__(self):
        """执行统计计算：计算双足足高差"""
        diff = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, 2]
        diff = torch.abs(diff[:, :1] - diff[:, 1:])
        self._calculate_episode(diff)


class StatusFootContact(StatusBase):

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _init_buffers(self):
        # 初始化足接触统计的均值与方差缓冲区 (一维数据)
        self.episode_variance_buf = torch.zeros((self.num_envs, 1),
                              device=self.device, dtype=torch.float)
        self.episode_mean_buf = torch.zeros_like(self.episode_variance_buf)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        # 重置足接触统计的缓冲区，并导出导数数据
        if env_ids is None or len(env_ids) == 0:
            return {}

        items = {}
        if 0 == self._env.common_step_counter % self.cfg.export_interval:

            mean = self.episode_mean_buf[env_ids]
            items[f"em"] = torch.mean(mean, dim=1)

            variance = self.episode_variance_buf[env_ids]
            items[f"ev"] = torch.mean(variance, dim=1)

        # 清空所有足接触统计缓冲区
        buffers = [
            self.episode_variance_buf,
            self.episode_mean_buf,
        ]

        for buf in buffers:
            buf[env_ids] = 0

        return items

    def __call__(self):
        """执行统计计算：直接根据足部接触位置计算统计数据"""
        diff = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, 2]
        self._calculate_episode(diff)


from .joints import CovarianceStatistics

class CovarVel(StatusVel, CovarianceStatistics):
    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv) -> None:
        StatusVel.__init__(self, cfg, env)
        CovarianceStatistics.__init__(self, cfg, env)

    def _calculate_episode(self, diff: torch.Tensor) -> None:
        CovarianceStatistics._calculate_episode(self, diff)


class CovarFootHeight(StatusFootHeight, CovarianceStatistics):
    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv) -> None:
        StatusFootHeight.__init__(self, cfg, env)
        CovarianceStatistics.__init__(self, cfg, env)

    def _calculate_episode(self, diff: torch.Tensor) -> None:
        CovarianceStatistics._calculate_episode(self, diff)


class CovarFootContact(StatusFootContact, CovarianceStatistics):
    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv) -> None:
        StatusFootContact.__init__(self, cfg, env)
        CovarianceStatistics.__init__(self, cfg, env)

    def _calculate_episode(self, diff: torch.Tensor) -> None:
        CovarianceStatistics._calculate_episode(self, diff)


