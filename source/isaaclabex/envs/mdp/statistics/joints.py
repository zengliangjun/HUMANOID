"""
关节状态统计模块
功能：计算机器人关节位置、速度、加速度和扭矩的统计量（均值和方差）
支持两种统计粒度：
1. Episode级别：整个训练周期的统计
2. Step级别：按关节分组（如左右关节）的统计
"""

from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionManager, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclabex.envs.managers.term_cfg import StatisticsTermCfg

class StatusBase(ManagerTermBase):
    """关节统计基类，提供公共统计方法和缓冲区"""

    cfg: StatisticsTermCfg

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        """
        初始化统计缓冲区
        Args:
            cfg: 统计项配置，包含asset_cfg参数
            env: RL环境实例
        """
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]

        self.command_name = cfg.params["command_name"]

        # 初始化关节统计缓冲区
        self.asset: Articulation = self._env.scene[asset_cfg.name]
        joint_count = len(self.asset.data.joint_names)

        # Episode级别统计
        self.episode_variance_buf = torch.zeros((self.num_envs, joint_count),
                              device=self.device, dtype=torch.float)
        self.episode_mean_buf = torch.zeros_like(self.episode_variance_buf)

        # Step级别统计（按关节分组）
        self.step_mean_mean_buf = torch.zeros((self.num_envs, joint_count // 2),
                              device=self.device, dtype=torch.float)
        self.step_mean_variance_buf = torch.zeros_like(self.step_mean_mean_buf)
        self.step_variance_mean_buf = torch.zeros_like(self.step_mean_mean_buf)
        self.step_variance_variance_buf = torch.zeros_like(self.step_mean_mean_buf)

        # 初始化标志位
        self.stand_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.zero_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        """重置指定环境的统计缓冲区
        Args:
            env_ids: 需要重置的环境ID列表
        Returns:
            空字典（保持接口统一）
        """
        if env_ids is None or len(env_ids) == 0:
            return {}

        items = {}
        if 0 == self._env.common_step_counter % self.cfg.export_interval:

            mean = self.episode_mean_buf[env_ids]
            for id, name in enumerate(self.asset.data.joint_names):
                name = name.replace("_joint", "")
                items[f"em/{name}"] = torch.mean(mean[:, id])
                items[f"em2/{name}"] = torch.sqrt(torch.mean(torch.square(mean[:, id])))

            variance = self.episode_variance_buf[env_ids]
            for id, name in enumerate(self.asset.data.joint_names):
                name = name.replace("_joint", "")
                items[f"ev/{name}"] = torch.mean(variance[:, id])
                items[f"ev2/{name}"] = torch.sqrt(torch.mean(torch.square(variance[:, id])))

            mean = self.step_mean_mean_buf[env_ids]
            for id, name in enumerate(self.asset.data.joint_names[::2]):
                name = name.replace("_joint", "")
                name = name.replace("left_", "")
                items[f"smm/{name}"] = torch.mean(mean[:, id])
                items[f"smm2/{name}"] = torch.sqrt(torch.mean(torch.square(mean[:, id])))

            variance = self.step_mean_variance_buf[env_ids]
            for id, name in enumerate(self.asset.data.joint_names[::2]):
                name = name.replace("_joint", "")
                name = name.replace("left_", "")
                items[f"smv/{name}"] = torch.mean(variance[:, id])
                items[f"smv2/{name}"] = torch.sqrt(torch.mean(torch.square(variance[:, id])))

            mean = self.step_variance_mean_buf[env_ids]
            for id, name in enumerate(self.asset.data.joint_names[::2]):
                name = name.replace("_joint", "")
                name = name.replace("left_", "")
                items[f"svm/{name}"] = torch.mean(mean[:, id])
                items[f"svm2/{name}"] = torch.sqrt(torch.mean(torch.square(mean[:, id])))

            variance = self.step_variance_variance_buf[env_ids]
            for id, name in enumerate(self.asset.data.joint_names[::2]):
                name = name.replace("_joint", "")
                name = name.replace("left_", "")
                items[f"svv/{name}"] = torch.mean(variance[:, id])
                items[f"svv2/{name}"] = torch.sqrt(torch.mean(torch.square(variance[:, id])))

        # 重置所有缓冲区
        buffers = [
            self.episode_variance_buf,
            self.episode_mean_buf,
            self.step_mean_mean_buf,
            self.step_mean_variance_buf,
            self.step_variance_mean_buf,
            self.step_variance_variance_buf
        ]

        for buf in buffers:
            buf[env_ids] = 0

        return items

    def _episode_length(self) -> torch.Tensor:
        if -1 == self.cfg.episode_truncation:
            return self._env.episode_length_buf
        else:
            return torch.clamp_max(self._env.episode_length_buf, self.cfg.episode_truncation)

    def _calculate_episode(self, diff: torch.Tensor) -> None:
        """更新episode级别的统计量（Welford算法）
        Args:
            diff: 当前step的关节状态差值
        """
        episode_length_buf = self._episode_length()
        # 计算均值
        delta0 = diff - self.episode_mean_buf
        self.episode_mean_buf += delta0 / episode_length_buf[:, None]

        # 计算方差
        delta1 = diff - self.episode_mean_buf
        self.episode_variance_buf = (
            self.episode_variance_buf * (episode_length_buf[:, None] - 2)
            + delta0 * delta1
        ) / (episode_length_buf[:, None] - 1)

        # 处理新episode
        new_episode_mask = episode_length_buf <= 1
        # self.episode_mean_buf[new_episode_mask] = 0
        self.episode_variance_buf[new_episode_mask] = 0

    def _calculate_step(self, diff: torch.Tensor) -> None:
        """更新step级别的分组统计量"""
        # 将关节分为两组（如左右关节）
        diff = torch.stack([diff[:, ::2], diff[:, 1::2]], dim=-1)

        # 计算每组均值和方差
        var, mean = torch.var_mean(diff, dim=-1)

        # 更新统计量（使用Welford算法）
        self._update_step_stats(mean, var)

    def _update_step_stats(self, mean: torch.Tensor, var: torch.Tensor) -> None:
        """更新step级别统计缓冲区"""
        episode_length_buf = self._episode_length()
        # 更新均值统计
        delta_mean0 = mean - self.step_mean_mean_buf
        self.step_mean_mean_buf += delta_mean0 / episode_length_buf[:, None]
        # 计算均值方差
        delta_mean1 = mean - self.step_mean_mean_buf
        self.step_mean_variance_buf = (
            self.step_mean_variance_buf * (episode_length_buf[:, None] - 2)
            + delta_mean0 * delta_mean1
        ) / (episode_length_buf[:, None] - 1)

        # 更新方差统计
        delta_var0 = var - self.step_variance_mean_buf
        self.step_variance_mean_buf += delta_var0 / episode_length_buf[:, None]
        # 计算方差方差
        delta_var1 = var - self.step_variance_mean_buf
        self.step_variance_variance_buf = (
            self.step_variance_variance_buf * (episode_length_buf[:, None] - 2)
            + delta_var0 * delta_var1
        ) / (episode_length_buf[:, None] - 1)

        reset_mask = episode_length_buf <= 1
        self.step_mean_variance_buf[reset_mask] = 0
        self.step_variance_variance_buf[reset_mask] = 0

    def _update_flag(self):
        command = self._env.command_manager.get_command(self.command_name)
        self.stand_flag[...] = torch.norm(command, dim=1) < 0.1
        self.zero_flag[...] = self._env.episode_length_buf <= 1


class StatusJPos(StatusBase):
    """关节位置统计"""

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.method = cfg.params.get("method", 0)  # 0: 相对默认位置, 1: 绝对位置
        self.diff = torch.zeros_like(self.episode_variance_buf)

    def __call__(self):
        """执行统计计算"""
        diff = (
            self._calculate_withdefault()
            if self.method == 0
            else self._calculate_withzero()
        )
        self.diff[...] = diff
        self._calculate_episode(diff)
        self._calculate_step(diff)
        self._update_flag()


    def _calculate_withdefault(self) -> torch.Tensor:
        """计算相对于默认位置的差值"""
        return self.asset.data.joint_pos - self.asset.data.default_joint_pos

    def _calculate_withzero(self) -> torch.Tensor:
        """直接返回关节位置"""
        return self.asset.data.joint_pos

class StatusJVel(StatusBase):
    """关节速度统计"""

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(self):
        """执行统计计算"""
        diff = self.asset.data.joint_vel
        self._calculate_episode(diff)
        self._calculate_step(diff)

class StatusJAcc(StatusBase):
    """关节加速度统计"""

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(self):
        """执行统计计算"""
        diff = self.asset.data.joint_acc
        self._calculate_episode(diff)
        self._calculate_step(diff)

class StatusTorque(StatusBase):
    """关节扭矩统计"""

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(self):
        """执行统计计算"""
        diff = self.asset.data.applied_torque
        self._calculate_episode(diff)
        self._calculate_step(diff)


class StatusAction(StatusBase):
    """关节扭矩统计"""

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.action_name = cfg.params.get("action_name")

    def __call__(self):
        """执行统计计算"""
        manager: ActionManager = self._env.action_manager
        if self.action_name is None:
            action = manager.action
        else:
            action = manager.get_term(self.action_name).raw_actions

        self._calculate_episode(action)
        self._calculate_step(action)


class CovarianceStatistics():

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv) -> None:
        joint_count = len(self.asset.data.joint_names)
        self.episode_covariance_buf = torch.zeros((self.num_envs, joint_count, joint_count),
                                                    device=self.device, dtype=torch.float)

    def _calculate_episode(self, diff: torch.Tensor) -> None:
        episode_length_buf = self._episode_length()
        # 计算均值
        delta0 = diff - self.episode_mean_buf
        self.episode_mean_buf += delta0 / episode_length_buf[:, None]

        # 计算方差
        delta1 = diff - self.episode_mean_buf
        self.episode_variance_buf = (
            self.episode_variance_buf * (episode_length_buf[:, None] - 2)
            + delta0 * delta1
        ) / (episode_length_buf[:, None] - 1)


        # 使用 torch.einsum 计算外积后更新协方差缓冲值
        covariance = torch.einsum("bi,bj->bij", delta1, delta0)
        self.episode_covariance_buf = (\
            self.episode_covariance_buf * (episode_length_buf[:, None, None] - 2) + \
            covariance) / (episode_length_buf[:, None, None] - 1)

        episode_mask = episode_length_buf <= 1
        self.episode_variance_buf[episode_mask] = 0
        self.episode_covariance_buf[episode_mask] = 0

class CovarAction(StatusAction, CovarianceStatistics):
    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv) -> None:
        StatusAction.__init__(self, cfg, env)
        CovarianceStatistics.__init__(self, cfg, env)

    def _calculate_episode(self, diff: torch.Tensor) -> None:
        CovarianceStatistics._calculate_episode(self, diff)


class CovarJPos(StatusJPos, CovarianceStatistics):
    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv) -> None:
        StatusJPos.__init__(self, cfg, env)
        CovarianceStatistics.__init__(self, cfg, env)

    def _calculate_episode(self, diff: torch.Tensor) -> None:
        CovarianceStatistics._calculate_episode(self, diff)


class CovarJVel(StatusJVel, CovarianceStatistics):
    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv) -> None:
        StatusJVel.__init__(self, cfg, env)
        CovarianceStatistics.__init__(self, cfg, env)

    def _calculate_episode(self, diff: torch.Tensor) -> None:
        CovarianceStatistics._calculate_episode(self, diff)

class CovarJAcc(StatusJAcc, CovarianceStatistics):
    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv) -> None:
        StatusJAcc.__init__(self, cfg, env)
        CovarianceStatistics.__init__(self, cfg, env)

    def _calculate_episode(self, diff: torch.Tensor) -> None:
        CovarianceStatistics._calculate_episode(self, diff)

class CovarTorque(StatusTorque, CovarianceStatistics):
    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv) -> None:
        StatusTorque.__init__(self, cfg, env)
        CovarianceStatistics.__init__(self, cfg, env)

    def _calculate_episode(self, diff: torch.Tensor) -> None:
        CovarianceStatistics._calculate_episode(self, diff)

