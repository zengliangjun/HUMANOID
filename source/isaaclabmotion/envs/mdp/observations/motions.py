from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.assets import Articulation

from isaaclabmotion.envs.managers import motions_manager
import isaaclab.utils.math as isaac_math_utils
from extends.isaac_utils import math_utils, rotations, torch_utils
import copy

DEBUG = False

class Base(ManagerTermBase):
    """Base class for joint statistics calculation with common functionality."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        motions_name: str = cfg.params["motions_name"]

        self.motions: motions_manager.MotionsTerm = env.motions_manager.get_term(motions_name)

        self.asset: Articulation = env.scene[asset_cfg.name]

        if asset_cfg.body_names is None or len(asset_cfg.body_names) == 0:
            motion_names = copy.deepcopy(self.asset.body_names)
        else:
            motion_names = copy.deepcopy(asset_cfg.body_names)

        if "extend_body_names" in cfg.params:
            motion_names += cfg.params["extend_body_names"]

        self._obs_motions_bodyids, _ = self.motions.resolve_motion_bodies(motion_names)

    def _obs_motion(self) -> dict:
        return self.motions.motion_ref(0)

    @property
    def obs_motions_bodyids(self):
        # TODO
        return self._obs_motions_bodyids

    def heading_inv_wxyz(self):
        return self.motions.heading_inv_wxyz

    def heading_wxyz(self):
        return self.motions.heading_wxyz


class obs_diff_rbpos(Base):

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)


    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        motions_name: str,
        extend_body_names: Sequence[str] = None
    ) -> torch.Tensor:

        ref_motions = self._obs_motion()
        motions_pos = ref_motions['rg_pos_t']

        inv_wxyz = self.heading_inv_wxyz()
        inv_wxyz2 = inv_wxyz[:, None, :]

        pos = self.asset.data.body_pos_w[: , self.motions.body_ids]
        pos = torch.cat((pos, self.motions.extend_body_pos), dim = 1)

        diff_pos = (motions_pos - pos)[:, self.obs_motions_bodyids]

        inv_wxyz2 = inv_wxyz2.repeat((1, diff_pos.shape[1], 1))

        diff_rbpos = isaac_math_utils.quat_rotate(inv_wxyz2, diff_pos)

        if DEBUG:
            num_envs, num_bodies, _ = diff_pos.shape

            root_rot = math_utils.convert_quat(self.asset.data.root_quat_w, to="xyzw")
            heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)  # xyzw
            heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
            heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, num_bodies, 1))
            flat_heading_rot_inv = heading_rot_inv_expand.reshape(num_envs * num_bodies, 4)

            diff_local_body_pos_flat = torch_utils.my_quat_rotate(
                    flat_heading_rot_inv, diff_pos.view(-1, 3))  # input xyzw

            diff_local_body_pos_flat = torch.reshape(diff_local_body_pos_flat, (num_envs, num_bodies, -1))

            diff = diff_local_body_pos_flat - diff_rbpos

        return diff_rbpos.flatten(1)

class obs_diff_rbquat(Base):

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)


    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        motions_name: str,
        extend_body_names: Sequence[str] = None
    ) -> torch.Tensor:

        ref_motions = self._obs_motion()

        ref_quat_wxyz = rotations.xyzw_to_wxyz(ref_motions['rg_rot_t'])[:, self.obs_motions_bodyids]

        inv_wxyz = self.heading_inv_wxyz()
        inv_wxyz2 = inv_wxyz[:, None, :]

        heading_wxyz = self.heading_wxyz()
        heading_wxyz2 = heading_wxyz[:, None, :]

        quat_wxyz = self.asset.data.body_quat_w.clone()
        quat_wxyz = quat_wxyz[: , self.motions.body_ids]
        quat_wxyz = torch.cat((quat_wxyz, self.motions.extend_body_rot_wxyz), dim = 1)[:, self.obs_motions_bodyids]

        inv_wxyz2 = inv_wxyz2.repeat((1, quat_wxyz.shape[1], 1))
        heading_wxyz2 = heading_wxyz2.repeat((1, quat_wxyz.shape[1], 1))

        diff_quat_wxyz = isaac_math_utils.quat_mul(ref_quat_wxyz, isaac_math_utils.quat_conjugate(quat_wxyz))
        #diff_quat = isaac_math_utils.quat_error_magnitude(motions_quat_wxyz, quat)
        diff_rbquat_wxyz = isaac_math_utils.quat_mul(inv_wxyz2, diff_quat_wxyz)
        diff_rbquat_wxyz = isaac_math_utils.quat_mul(diff_rbquat_wxyz, heading_wxyz2)

        diff_rbquat_xyzw = rotations.wxyz_to_xyzw(diff_rbquat_wxyz)

        diff_rbquat_xyzw = torch.reshape(diff_rbquat_xyzw, (-1, diff_rbquat_xyzw.shape[-1]))
        diff_norm = torch_utils.quat_to_tan_norm(diff_rbquat_xyzw)
        diff_norm = torch.reshape(diff_norm, (quat_wxyz.shape[0], -1))

        if DEBUG:
            num_envs, num_bodies, _ = ref_quat_wxyz.shape

            root_rot = math_utils.convert_quat(self.asset.data.root_quat_w, to="xyzw")
            heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)  # xyzw
            heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
            heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, num_bodies, 1))
            flat_heading_rot_inv = heading_rot_inv_expand.reshape(num_envs * num_bodies, 4)


            heading_rot = torch_utils.calc_heading_quat(root_rot)  # xyzw
            heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, num_bodies, 1))

            ##
            diff_global_body_rot = math_utils.quat_mul(
                                        ref_quat_wxyz,
                                        math_utils.quat_conjugate(quat_wxyz),
                                    )  # input wxyz out wxyz
            diff_local_body_rot_flat = math_utils.quat_mul(
                    math_utils.quat_mul(
                        math_utils.convert_quat(flat_heading_rot_inv, to="wxyz"),
                        diff_global_body_rot.view(-1, 4)
                    ),
                    math_utils.convert_quat(heading_rot_expand.view(-1, 4), to="wxyz"),
            )  # Need to be change of basis  # input wxyz

            diff_local_body_rot_flat = math_utils.convert_quat(diff_local_body_rot_flat, to="xyzw")  # out xyzw

            diff_norm2 = torch_utils.quat_to_tan_norm(diff_local_body_rot_flat)
            diff_norm2 = torch.reshape(diff_norm2, (num_envs, -1))

            diff = diff_norm - diff_norm2

        return diff_norm


class obs_diff_rblin(Base):

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)


    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        motions_name: str,
        extend_body_names: Sequence[str] = None
    ) -> torch.Tensor:

        ref_motions = self._obs_motion()
        motions_lin = ref_motions['body_vel_t']

        inv_wxyz = self.heading_inv_wxyz()
        inv_wxyz2 = inv_wxyz[:, None, :]

        lin = self.asset.data.body_lin_vel_w[: , self.motions.body_ids]
        lin = torch.cat((lin, self.motions.extend_body_lin_vel), dim = 1)

        diff_lin = (motions_lin - lin)[:, self.obs_motions_bodyids]

        inv_wxyz2 = inv_wxyz2.repeat((1, diff_lin.shape[1], 1))

        diff_rblin = isaac_math_utils.quat_rotate(inv_wxyz2, diff_lin)

        if DEBUG:
            num_envs, num_bodies, _ = diff_lin.shape

            root_rot = math_utils.convert_quat(self.asset.data.root_quat_w, to="xyzw")
            heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)  # xyzw
            heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
            heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, num_bodies, 1))
            flat_heading_rot_inv = heading_rot_inv_expand.reshape(num_envs * num_bodies, 4)

            diff_local_body_lin_flat = torch_utils.my_quat_rotate(
                    flat_heading_rot_inv, diff_lin.view(-1, 3))  # input xyzw

            diff_local_body_lin_flat = torch.reshape(diff_local_body_lin_flat, (num_envs, num_bodies, -1))

            diff = diff_local_body_lin_flat - diff_rblin

        return diff_rblin.flatten(1)


class obs_diff_rbang(Base):

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)


    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        motions_name: str,
        extend_body_names: Sequence[str] = None
    ) -> torch.Tensor:

        ref_motions = self._obs_motion()
        motions_ang = ref_motions['body_ang_vel_t']

        inv_wxyz = self.heading_inv_wxyz()
        inv_wxyz2 = inv_wxyz[:, None, :]

        ang = self.asset.data.body_ang_vel_w[: , self.motions.body_ids]
        ang = torch.cat((ang, self.motions.extend_body_ang_vel), dim = 1)

        diff_ang = (motions_ang - ang)[:, self.obs_motions_bodyids]

        inv_wxyz2 = inv_wxyz2.repeat((1, diff_ang.shape[1], 1))

        diff_rbang = isaac_math_utils.quat_rotate(inv_wxyz2, diff_ang)

        if DEBUG:
            num_envs, num_bodies, _ = diff_ang.shape

            root_rot = math_utils.convert_quat(self.asset.data.root_quat_w, to="xyzw")
            heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)  # xyzw
            heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
            heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, num_bodies, 1))
            flat_heading_rot_inv = heading_rot_inv_expand.reshape(num_envs * num_bodies, 4)

            diff_local_body_ang_flat = torch_utils.my_quat_rotate(
                    flat_heading_rot_inv, diff_ang.view(-1, 3))  # input xyzw

            diff_local_body_ang_flat = torch.reshape(diff_local_body_ang_flat, (num_envs, num_bodies, -1))

            diff = diff_local_body_ang_flat - diff_rbang

        return diff_rbang.flatten(1)

class obs_diff_root_rbpos(Base):

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)


    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        motions_name: str,
        extend_body_names: Sequence[str] = None
    ) -> torch.Tensor:

        ref_motions = self._obs_motion()
        ref_root_pos = ref_motions['root_pos']

        inv_wxyz = self.heading_inv_wxyz()
        root_pos = self.asset.data.root_pos_w

        diff_root_pos = ref_root_pos - root_pos
        diff_root_rbpos = isaac_math_utils.quat_rotate(inv_wxyz, diff_root_pos)

        if DEBUG:
            num_envs, _ = diff_root_pos.shape

            root_rot = math_utils.convert_quat(self.asset.data.root_quat_w, to="xyzw")
            heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)  # xyzw

            diff_local_root = torch_utils.my_quat_rotate(
                    heading_rot_inv, diff_root_pos.view(-1, 3))  # input xyzw

            diff = diff_local_root - diff_root_rbpos

        return diff_root_rbpos.flatten(1)

class obs_diff_root_rbquat(Base):

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)


    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        motions_name: str,
        extend_body_names: Sequence[str] = None
    ) -> torch.Tensor:

        ref_motions = self._obs_motion()
        ref_root_wxyz = rotations.xyzw_to_wxyz(ref_motions['root_rot'])

        root_quat = self.asset.data.root_quat_w

        diff_quat_wxyz = isaac_math_utils.quat_mul(ref_root_wxyz, isaac_math_utils.quat_conjugate(root_quat))
        diff_quat_xyzw = rotations.wxyz_to_xyzw(diff_quat_wxyz)
        #
        diff_norm = torch_utils.quat_to_tan_norm(diff_quat_xyzw)
        return diff_norm

class obs_motions_rbpos(Base):

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        motions_name: str,
        extend_body_names: Sequence[str] = None
    ) -> torch.Tensor:

        ref_motions = self._obs_motion()
        ref_pos = ref_motions['rg_pos_t']

        #
        inv_wxyz = self.heading_inv_wxyz()
        inv_wxyz2 = inv_wxyz[:, None, :]

        #
        root_pos = self.asset.data.root_pos_w[:, None, :]
        diff_pos = (ref_pos - root_pos)

        #
        inv_wxyz2 = inv_wxyz2.repeat((1, diff_pos.shape[1], 1))

        motions_rbpos = isaac_math_utils.quat_rotate(inv_wxyz2, diff_pos)
        return motions_rbpos.flatten(1)

class obs_motions_rbquat(Base):

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        motions_name: str,
        extend_body_names: Sequence[str] = None
    ) -> torch.Tensor:

        ref_motions = self._obs_motion()
        ref_quat_wxyz = rotations.xyzw_to_wxyz(ref_motions['rg_rot_t'][:, self.obs_motions_bodyids])

        inv_wxyz = self.heading_inv_wxyz()
        inv_wxyz2 = inv_wxyz[:, None, :]
        inv_wxyz2 = inv_wxyz2.repeat((1, ref_quat_wxyz.shape[1], 1))

        ref_quat_wxyz = isaac_math_utils.quat_mul(inv_wxyz2, isaac_math_utils.quat_conjugate(ref_quat_wxyz))
        ref_rbquat = rotations.wxyz_to_xyzw(ref_quat_wxyz)

        ref_rbquat = torch.reshape(ref_rbquat, (-1, ref_rbquat.shape[-1]))
        ref_norm = torch_utils.quat_to_tan_norm(ref_rbquat)
        ref_norm = torch.reshape(ref_norm, (inv_wxyz.shape[0], -1))
        return ref_norm
