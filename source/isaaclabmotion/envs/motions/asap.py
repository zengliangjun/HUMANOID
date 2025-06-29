from __future__ import annotations
import numpy as np
from dataclasses import MISSING
from typing import TYPE_CHECKING
from collections.abc import Sequence
import torch
from isaaclab.utils import configclass

from isaaclabmotion.envs.managers.motions_manager import MotionsTerm
from extends.motion_lib.aosp import motion_lib_robot
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.assets import Articulation

#import isaaclab.utils.math as math_utils
from extends.isaac_utils import rotations

if TYPE_CHECKING:
    from isaaclabmotion.envs.env_motions import ManagerMotionsEnv
    from isaaclabmotion.envs.managers.term_cfg import MotionsTermCfg

"""
- joint_name: "left_hand_link"
        parent_name: "left_elbow_link"
        pos: [0.3, 0.0, 0.0]
        rot: [1.0, 0.0, 0.0, 0.0]
- joint_name: "right_hand_link"
        parent_name: "right_elbow_link"
        pos: [0.3, 0.0, 0.0]
        rot: [1.0, 0.0, 0.0, 0.0]

"""

@configclass
class ASAPMotionlibCfg:
    """
    配置类，用于存储ASAP运动库的相关参数。
    """
    num_envs: int = MISSING
    device: str = MISSING
    ##
    step_dt: float = MISSING

    motion_file: str = MISSING

    mjcf_file: str = MISSING

    extend_config: list = MISSING

class ASAPMotions(MotionsTerm):
    """
    继承自MotionsTerm，实现ASAP运动库的管理与接口。
    """

    cfg: MotionsTermCfg

    def __init__(self, cfg: MotionsTermCfg, env: ManagerMotionsEnv):
        """
        初始化ASAPMotions对象，加载运动库并建立资产与运动的映射关系。
        """
        super(ASAPMotions, self).__init__(cfg, env)
        assert "mjcf_file" in self.cfg.params
        assert "extend_config" in self.cfg.params
        assert "body_names" in self.cfg.params
        assert "joint_names" in self.cfg.params

        # 构建运动库配置
        libcfg = ASAPMotionlibCfg(
            num_envs = self.num_envs,
            device = self.device,
            step_dt = self._env.step_dt,
            motion_file = self.cfg.motion_file,
            mjcf_file = self.cfg.params["mjcf_file"],
            extend_config = self.cfg.params["extend_config"]
        )

        # 初始化运动库
        self.motion_lib = motion_lib_robot.MotionLibRobot(libcfg)

        # 加载运动数据
        self.motion_lib.load_motions(random_sample=self.cfg.random_sample)
        self.motion_len[:] = self.motion_lib.get_motion_length(self.motion_ids)

        # 初始化起始时间
        if self.cfg.random_sample:
            self.start_times[...] = self.motion_lib.sample_time(self.motion_ids)
        else:
            self.start_times[...] = 0

        self.num_motions = self.motion_lib._num_unique_motions
        if -1 != self.cfg.resample_interval_s:
            self.resample_time_interval = np.ceil(self.cfg.resample_interval_s / self._env.step_dt)

        # 初始化资产到运动的映射
        body_names = cfg.params["body_names"]
        joint_names = cfg.params["joint_names"]

        assert_cfg: SceneEntityCfg = cfg.assert_cfg
        asset: Articulation = self._env.scene[assert_cfg.name]

        bodyAssetToMotionIds = []
        for name in body_names:
            # 资产body名称到运动body名称的索引映射
            bodyAssetToMotionIds.append(asset.body_names.index(name))

        jointMotionToAssetIds = []
        for name in asset.joint_names:
            # 运动joint名称到资产joint名称的索引映射
            jointMotionToAssetIds.append(joint_names.index(name))

        self._bodyAssetToMotionIds = bodyAssetToMotionIds
        self._jointMotionToAssetIds = jointMotionToAssetIds

        ## extend
        # 扩展body名称列表，添加额外的关节
        body_names = []
        parent_names = []
        poses = []
        rots = []

        for item in self.cfg.params["extend_config"]:
            body_names.append(item["joint_name"])
            parent_names.append(item["parent_name"])
            poses.append(item["pos"])
            rots.append(item["rot"])

        self._extend_body_names = body_names
        self._extend_body_parent_names = parent_names
        self._extend_body_parent_ids = [asset.body_names.index(n) for n in parent_names]
        self._extend_body_parent_poses = torch.tensor(poses, dtype= torch.float32, device = self.device).repeat(self.num_envs, 1, 1) # envs * n * 3
        self._extend_body_parent_rots_wxyz = torch.tensor(rots, dtype= torch.float32, device = self.device).repeat(self.num_envs, 1, 1) # envs * n * 4

        # compute extend status
        self._extend_body_pos = torch.zeros_like(self._extend_body_parent_poses)
        self._extend_body_rot_wxyz = torch.zeros_like(self._extend_body_parent_rots_wxyz)
        self._extend_body_ang_vel = torch.zeros_like(self._extend_body_pos)
        self._extend_body_lin_vel = torch.zeros_like(self._extend_body_pos)

        self._motion_body_names = cfg.params["body_names"] + self._extend_body_names

    def motion_ref(self, step: int):
        """
        获取指定步数下的参考运动状态。
        """
        offset = self._env.scene.env_origins
        motion_times = self.motion_times(step)
        return self.motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)

    def motion_times(self, step: int):
        """
        计算每个环境下当前步数对应的运动时间。
        """
        if hasattr(self._env, "episode_length_buf"):
            return (self._env.episode_length_buf + step) * self._env.step_dt + self.start_times # next frames so +1
        else:
            env_step_count = self._env._sim_step_counter // self._env.cfg.decimation
            return (env_step_count + step) * self._env.step_dt + self.start_times

    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        """
        重置指定环境的运动起始时间。
        """
        if len(env_ids) == 0:
            return {}

        if self.cfg.random_sample:
            self.start_times[env_ids] = self.motion_lib.sample_time(self.motion_ids[env_ids])
        else:
            self.start_times[env_ids] = 0

        self.step_play(env_ids)
        return {}

    def termination_compute(self) -> torch.Tensor:
        """
        检查当前运动是否需要重采样或重置。
        """
        current_time = self._env.episode_length_buf * self._env.step_dt + self.start_times
        reset_buf = current_time >= self.motion_len

        # 定期重采样运动
        if -1 != self.cfg.resample_interval_s and self._env.common_step_counter % self.resample_time_interval == 0:
            ## update reset flag with True
            self.motion_lib.load_motions(random_sample=self.cfg.random_sample)
            self.motion_len[:] = self.motion_lib.get_motion_length(self.motion_ids)

            if self.cfg.random_sample:
                self.start_times[...] = self.motion_lib.sample_time(self.motion_ids)
            else:
                self.start_times[...] = 0

            reset_buf[:] = 1

        return reset_buf


    def step_play(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        ref_motions = self.motion_ref(0)
        asset: Articulation = self._env.scene[self.cfg.assert_cfg.name]

        dof_pos = ref_motions['dof_pos'][env_ids]
        dof_vel = ref_motions['dof_vel'][env_ids]
        dof_pos = dof_pos[:, self.jointMotionToAssetIds]
        dof_vel = dof_vel[:, self.jointMotionToAssetIds]

        actions = torch.cat((dof_pos, dof_vel), dim = -1)

        root_pos = ref_motions['root_pos'][env_ids]                 # 0: 3
        root_rot = rotations.xyzw_to_wxyz(ref_motions['root_rot'][env_ids])   # 3: 7
        # root_rot = math_utils.convert_quat(ref_motions['root_rot'], to="wxyz")
        root_vel = ref_motions['root_vel'][env_ids]                 # 7: 10
        root_ang_vel = ref_motions['root_ang_vel'][env_ids]         # 7: 10

        pose = torch.cat((root_pos, root_rot), dim = -1)
        vel = torch.cat((root_vel, root_ang_vel), dim = -1)

        asset.set_joint_position_target(target = dof_pos, env_ids = env_ids)
        asset.set_joint_velocity_target(target = dof_vel, env_ids = env_ids)
        asset.write_root_pose_to_sim(root_pose = pose, env_ids = env_ids)
        asset.write_root_velocity_to_sim(root_velocity = vel, env_ids = env_ids)

        self.prev_motions = ref_motions

        return actions

    def extend_compute(self):
        asset: Articulation = self._env.scene[self.cfg.assert_cfg.name]

        if not hasattr(self, "_extend_body_parent_rots_xyzw"):
            self._extend_body_parent_rots_xyzw = rotations.wxyz_to_xyzw(self._extend_body_parent_rots_wxyz.reshape(-1, 4))

        ################### EXTEND Rigid body POS #####################
        parent_quatwxyz = asset.data.body_quat_w[:, self._extend_body_parent_ids].reshape(-1, 4)
        parent_quatxyzw = rotations.wxyz_to_xyzw(parent_quatwxyz)

        rotated_pos_in_parent = rotations.my_quat_rotate(parent_quatxyzw,
                        self._extend_body_parent_poses.reshape(-1, 3)
                    )
        extend_pos = rotations.my_quat_rotate(
                self._extend_body_parent_rots_xyzw,
                rotated_pos_in_parent
            ).view(self.num_envs, -1, 3) + asset.data.body_pos_w[:, self._extend_body_parent_ids]

        self._extend_body_pos[...] = extend_pos
        ################### EXTEND Rigid body Rotation #####################
        extend_rot_xyzw = rotations.quat_mul(parent_quatxyzw,
                                    self._extend_body_parent_rots_xyzw,
                                    w_last=True).view(self.num_envs, -1, 4)

        self._extend_body_rot_wxyz[...] = rotations.xyzw_to_wxyz(extend_rot_xyzw)
        ################### EXTEND Rigid Body Angular Velocity #####################
        self._extend_body_ang_vel[...] = asset.data.body_ang_vel_w[:, self._extend_body_parent_ids]

        ################### EXTEND Rigid Body Linear Velocity #####################
        angular_contribution = torch.cross(self._extend_body_ang_vel, self._extend_body_parent_poses, dim=2)
        _extend_curr_vel = asset.data.body_lin_vel_w[:, self._extend_body_parent_ids] + angular_contribution
        self._extend_body_lin_vel[...] = _extend_curr_vel

        if False:
            extend_count = len(self.extend_body_names)
            extend_pos = self.prev_motions["rg_pos_t"][:, - extend_count:]
            extend_rot = self.prev_motions["rg_rot_t"][:, - extend_count:] # xyzw
            extend_line_vel = self.prev_motions["body_vel_t"][:, - extend_count:]
            extend_ang_vel = self.prev_motions["body_ang_vel_t"][:, - extend_count:]

            root_pos = self.prev_motions["root_pos"]
            r_pos = asset.data.root_pos_w
            diff_pos = root_pos - r_pos

            root_rot = rotations.xyzw_to_wxyz(self.prev_motions["root_rot"])
            r_rot = asset.data.root_quat_w
            diff_rot = root_rot - r_rot

            root_vel = self.prev_motions["root_vel"]
            r_vel = asset.data.root_lin_vel_w
            diff_vel = root_vel - r_vel

            root_ang = self.prev_motions["root_ang_vel"]
            r_ang = asset.data.root_ang_vel_w
            diff_ang = root_ang - r_ang


            rg_pos = self.prev_motions["rg_pos"]
            bpos = asset.data.body_pos_w[: , self.bodyAssetToMotionIds]
            diff_bpos = rg_pos - bpos


            rb_rot = rotations.xyzw_to_wxyz(self.prev_motions["rb_rot"])
            brot = asset.data.body_quat_w[: , self.bodyAssetToMotionIds]
            diff_brot = rb_rot - brot

            body_vel = self.prev_motions["body_vel"]
            bvel = asset.data.body_lin_vel_w[: , self.bodyAssetToMotionIds]
            diff_bvel = body_vel - bvel

            body_ang = self.prev_motions["body_ang_vel"]
            bang = asset.data.body_ang_vel_w[: , self.bodyAssetToMotionIds]
            diff_bang = body_ang - bang

            dof_pos = self.prev_motions["dof_pos"][:, self.jointMotionToAssetIds]
            jpos = asset.data.joint_pos
            diff_jpos = dof_pos - jpos

            dof_vel = self.prev_motions["dof_vel"][:, self.jointMotionToAssetIds]
            jvel = asset.data.joint_vel
            diff_jvel = dof_vel - jvel

            print(">>>")