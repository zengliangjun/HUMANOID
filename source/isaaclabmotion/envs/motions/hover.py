from __future__ import annotations
from typing import TYPE_CHECKING

from extends.motion_lib.hover import motion_lib_robot
from . import base

if TYPE_CHECKING:
    from isaaclabmotion.envs.env_motions import ManagerMotionsEnv
    from isaaclabmotion.envs.managers.term_cfg import MotionsTermCfg


class HOVERMotions(base.MotionsBase):

    def _init_motion_lib(self):
        # 构建运动库配置
        multi_thread = False
        if "multi_thread" in self.cfg.params:
            multi_thread = self.cfg.params["multi_thread"]


        libcfg = motion_lib_robot.HOVERMotionlibCfg(
            num_envs = self.num_envs,
            device = self.device,
            step_dt = self._env.step_dt,
            motion_file = self.cfg.motion_file,
            mjcf_file = self.cfg.params["mjcf_file"],
            extend_config = self.cfg.params["extend_config"],
            multi_thread = multi_thread,
        )
        # 初始化运动库
        self.motion_lib = motion_lib_robot.MotionLibRobot(libcfg)

    def __init__(self, cfg: MotionsTermCfg, env: ManagerMotionsEnv):
        super(HOVERMotions, self).__init__(cfg, env)
