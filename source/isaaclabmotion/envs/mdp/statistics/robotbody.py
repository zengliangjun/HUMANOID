from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from isaaclabmotion.envs.managers import motions_manager
import isaaclab.utils.math as isaac_math_utils
from extends.isaac_utils import math_utils, rotations, torch_utils

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
        #motion
        motions_name: str = cfg.params["motions_name"]
        self.motions: motions_manager.MotionsTerm = env.motions_manager.get_term(motions_name)
        # 初始化标志位
        self.zero_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self._init_buffers()

    @property
    def body_num(self):
        return len(self.motions.motion_body_names)

    @property
    def status_motion(self) -> dict:
        return self.motions.motion_ref(0)

    @property
    def heading_inv_wxyz(self):
        return self.motions.heading_inv_wxyz

    @property
    def heading_wxyz(self):
        return self.motions.heading_wxyz

    def _episode_length(self) -> torch.Tensor:
        # 获取episode长度，当cfg中设置了截断时进行最大值截断
        if -1 == self.cfg.episode_truncation:
            return self._env.episode_length_buf
        else:
            return torch.clamp_max(self._env.episode_length_buf, self.cfg.episode_truncation)

    def _calculate_episode(self, diff: torch.Tensor) -> None:
        # 利用增量更新方法计算当前episode的均值和方差
        episode_length_buf = self._episode_length()

        if 2 == diff.dim():
            episode_length_buf_expand = episode_length_buf[:, None]
        elif 3 == diff.dim():
            episode_length_buf_expand = episode_length_buf[:, None, None]
        else:
            raise Exception(f"Don\'t support {diff.dim()}")

        # 计算均值：根据新差值delta0更新均值缓冲区
        delta0 = diff - self.episode_mean_buf
        self.episode_mean_buf += delta0 / episode_length_buf_expand

        # 计算方差：利用delta0和新均值计算更新方差缓冲区
        delta1 = diff - self.episode_mean_buf
        self.episode_variance_buf = (
            self.episode_variance_buf * (episode_length_buf_expand - 2)
            + delta0 * delta1
        ) / (episode_length_buf_expand - 1)

        # 当episode刚开始时重置方差，防止数值异常
        new_episode_mask = episode_length_buf <= 1
        # self.episode_mean_buf[new_episode_mask] = 0
        self.episode_variance_buf[new_episode_mask] = 0

    def _update_flag(self):
        self.zero_flag[...] = self._env.episode_length_buf <= 1

    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        # 重置指定环境ID的缓冲区，并根据周期导出统计数据
        if env_ids is None or len(env_ids) == 0:
            return {}

        items = {}
        if 0 == self._env.common_step_counter % self.cfg.export_interval:

            mean = self.episode_mean_buf[env_ids]
            items[f"em"] = torch.mean(torch.norm(mean, dim=-1))

            variance = self.episode_variance_buf[env_ids]
            items[f"ev"] = torch.mean(torch.norm(variance, dim=-1))

        # 重置所有相关缓冲区
        buffers = [
            self.episode_variance_buf,
            self.episode_mean_buf,
        ]

        for buf in buffers:
            buf[env_ids] = 0

        return items

class RBPosHeadDiffXY(StatusBase):

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _init_buffers(self):
        # 初始化速度统计的均值与方差缓冲区 (二维数据)
        self.episode_variance_buf = torch.zeros((self.num_envs, self.body_num, 2),
                              device=self.device, dtype=torch.float)

        self.episode_mean_buf = torch.zeros_like(self.episode_variance_buf)

    def __call__(self):
        ref_motions = self.status_motion
        ref_pos = ref_motions['rg_pos_t']
        ref_quat_xyzw = ref_motions['root_rot']

        ref_inv_xyzw = torch_utils.calc_heading_quat_inv(ref_quat_xyzw)  # xyzw
        ref_inv_wxyz = math_utils.convert_quat(ref_inv_xyzw, to="wxyz")
        ref_inv_wxyz = ref_inv_wxyz[:, None, :]
        ref_inv_wxyz = ref_inv_wxyz.repeat((1, self.body_num, 1))

        ref_rbpos = isaac_math_utils.quat_rotate(ref_inv_wxyz, ref_pos)

        assetpos = self.asset.data.body_pos_w[: , self.motions.body_ids]
        assetpos = torch.cat((assetpos, self.motions.extend_body_pos), dim = 1)

        inv_wxyz = self.motions.heading_inv_wxyz[:, None, :]
        inv_wxyz = inv_wxyz.repeat((1, self.body_num, 1))

        rbpos = isaac_math_utils.quat_rotate(inv_wxyz, assetpos)

        diff = (ref_rbpos - rbpos)[:, :, :2]
        self._update_flag()
        self._calculate_episode(diff)


class RBPosHeadDiff(StatusBase):

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _init_buffers(self):
        # 初始化速度统计的均值与方差缓冲区 (二维数据)
        self.episode_variance_buf = torch.zeros((self.num_envs, self.body_num, 3),
                              device=self.device, dtype=torch.float)

        self.episode_mean_buf = torch.zeros_like(self.episode_variance_buf)

    def __call__(self):
        ref_motions = self.status_motion
        ref_pos = ref_motions['rg_pos_t']
        ref_quat_xyzw = ref_motions['root_rot']

        ref_inv_xyzw = torch_utils.calc_heading_quat_inv(ref_quat_xyzw)  # xyzw
        ref_inv_wxyz = math_utils.convert_quat(ref_inv_xyzw, to="wxyz")
        ref_inv_wxyz = ref_inv_wxyz[:, None, :]
        ref_inv_wxyz = ref_inv_wxyz.repeat((1, self.body_num, 1))

        ref_rbpos = isaac_math_utils.quat_rotate(ref_inv_wxyz, ref_pos)

        assetpos = self.asset.data.body_pos_w[: , self.motions.body_ids]
        assetpos = torch.cat((assetpos, self.motions.extend_body_pos), dim = 1)

        inv_wxyz = self.motions.heading_inv_wxyz[:, None, :]
        inv_wxyz = inv_wxyz.repeat((1, self.body_num, 1))

        rbpos = isaac_math_utils.quat_rotate(inv_wxyz, assetpos)

        diff = (ref_rbpos - rbpos)
        self._update_flag()
        self._calculate_episode(diff)

class RBRootDiff(StatusBase):

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _init_buffers(self):
        # 初始化速度统计的均值与方差缓冲区 (二维数据)
        self.episode_variance_buf = torch.zeros((self.num_envs, 3),
                              device=self.device, dtype=torch.float)

        self.episode_mean_buf = torch.zeros_like(self.episode_variance_buf)

    def __call__(self):
        ref_motions = self.status_motion
        ref_root_wxyz = rotations.xyzw_to_wxyz(ref_motions['root_rot'])

        root_quat = self.asset.data.root_quat_w

        diff_quat_wxyz = isaac_math_utils.quat_mul(ref_root_wxyz, isaac_math_utils.quat_conjugate(root_quat))
        diff_angle = isaac_math_utils.axis_angle_from_quat(diff_quat_wxyz)
        self._update_flag()
        self._calculate_episode(diff_angle)

class RBPosDiff(StatusBase):

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _init_buffers(self):
        # 初始化速度统计的均值与方差缓冲区 (二维数据)
        self.episode_variance_buf = torch.zeros((self.num_envs, self.body_num, 3),
                              device=self.device, dtype=torch.float)

        self.episode_mean_buf = torch.zeros_like(self.episode_variance_buf)

    def __call__(self):
        ref_motions = self.status_motion
        ref_pos = ref_motions['rg_pos_t']

        assetpos = self.asset.data.body_pos_w[: , self.motions.body_ids]
        assetpos = torch.cat((assetpos, self.motions.extend_body_pos), dim = 1)

        diff = (ref_pos - assetpos)
        self._update_flag()
        self._calculate_episode(diff)
