from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.assets import Articulation

from isaaclabmotion.envs.managers import motions_manager
import isaaclab.utils.math as math_utils
from extends.isaac_utils import rotations, torch_utils

class Base(ManagerTermBase):
    """Base class for joint statistics calculation with common functionality."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        motions_name: str = cfg.params["motions_name"]

        self.motions: motions_manager.MotionsTerm = env.motions_manager.get_term(motions_name)

        self.asset: Articulation = env.scene[asset_cfg.name]

        if asset_cfg.body_names is None or len(asset_cfg.body_names) == 0:
            motion_names = self.asset.body_names
        else:
            motion_names = asset_cfg.body_names

        if "extend_body_names" in cfg.params:
            motion_names += cfg.params["extend_body_names"]

        self._obs_motions_bodyids, _ = self.motions.resolve_motion_bodies(motion_names)

    def _obs_motion(self) -> dict:
        return self.motions.motion_ref(0)

    @property
    def obs_motions_bodyids(self):
        # TODO
        return self._obs_motions_bodyids

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


        root_quat = self.asset.data.root_quat_w
        root_quat2 = root_quat[:, None, :]
        pos = self.asset.data.body_pos_w[: , self.motions.bodyAssetToMotionIds]
        pos = torch.cat((pos, self.motions.extend_body_pos), dim = 1)

        diff_pos = (motions_pos - pos)[:, self.obs_motions_bodyids]

        root_quat2 = root_quat2.repeat((1, diff_pos.shape[1], 1))

        diff_rbpos = math_utils.quat_rotate_inverse(root_quat2, diff_pos)
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

        motions_quat_wxyz = rotations.xyzw_to_wxyz(ref_motions['rg_rot_t'])[:, self.obs_motions_bodyids]


        root_quat = self.asset.data.root_quat_w
        root_quat2 = root_quat[:, None, :]

        quat = self.asset.data.body_quat_w.clone()
        quat = quat[: , self.motions.bodyAssetToMotionIds]
        quat = torch.cat((quat, self.motions.extend_body_rot_wxyz), dim = 1)[:, self.obs_motions_bodyids]

        root_quat2 = root_quat2.repeat((1, quat.shape[1], 1))

        diff_quat = math_utils.quat_mul(motions_quat_wxyz, math_utils.quat_conjugate(quat))
        #diff_quat = math_utils.quat_error_magnitude(motions_quat_wxyz, quat)
        diff_rbquat = math_utils.quat_mul(math_utils.quat_inv(root_quat2), diff_quat)
        diff_rbquat = math_utils.quat_mul(diff_rbquat, root_quat2)
        diff_rbquat = rotations.wxyz_to_xyzw(diff_rbquat)

        diff_rbquat = torch.reshape(diff_rbquat, (-1, diff_rbquat.shape[-1]))
        diff_rbquat = torch_utils.quat_to_tan_norm(diff_rbquat)
        diff_rbquat = torch.reshape(diff_rbquat, (quat.shape[0], -1))

        return diff_rbquat


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


        root_quat = self.asset.data.root_quat_w
        root_quat2 = root_quat[:, None, :]
        lin = self.asset.data.body_lin_vel_w[: , self.motions.bodyAssetToMotionIds]
        lin = torch.cat((lin, self.motions.extend_body_lin_vel), dim = 1)

        diff_lin = (motions_lin - lin)[:, self.obs_motions_bodyids]

        root_quat2 = root_quat2.repeat((1, diff_lin.shape[1], 1))

        diff_lin = math_utils.quat_rotate_inverse(root_quat2, diff_lin)
        return diff_lin.flatten(1)


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


        root_quat = self.asset.data.root_quat_w
        root_quat2 = root_quat[:, None, :]
        ang = self.asset.data.body_ang_vel_w[: , self.motions.bodyAssetToMotionIds]
        ang = torch.cat((ang, self.motions.extend_body_ang_vel), dim = 1)

        diff_ang = (motions_ang - ang)[:, self.obs_motions_bodyids]

        root_quat2 = root_quat2.repeat((1, diff_ang.shape[1], 1))

        diff_ang = math_utils.quat_rotate_inverse(root_quat2, diff_ang)
        return diff_ang.flatten(1)

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
        motions_root_pos = ref_motions['root_pos']

        root_quat = self.asset.data.root_quat_w
        root_pos = self.asset.data.root_pos_w

        diff_root_pos = motions_root_pos - root_pos
        diff_root_rbpos = math_utils.quat_rotate_inverse(root_quat, diff_root_pos)
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
        motions_root_quat_wxyz = rotations.xyzw_to_wxyz(ref_motions['root_rot'])

        root_quat = self.asset.data.root_quat_w

        diff_root_rbquat = math_utils.quat_mul(root_quat, math_utils.quat_conjugate(motions_root_quat_wxyz))
        #diff_root_rbquat = math_utils.quat_error_magnitude(root_quat, motions_root_quat_wxyz)
        diff_root_rbquat = rotations.wxyz_to_xyzw(diff_root_rbquat)

        diff_root_rbquat = torch.reshape(diff_root_rbquat, (-1, diff_root_rbquat.shape[-1]))
        diff_root_rbquat = torch_utils.quat_to_tan_norm(diff_root_rbquat)
        diff_root_rbquat = torch.reshape(diff_root_rbquat, (root_quat.shape[0], -1))
        return diff_root_rbquat

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
        motions_pos = ref_motions['rg_pos_t']

        root_quat = self.asset.data.root_quat_w[:, None, :]
        root_pos = self.asset.data.root_pos_w[:, None, :]

        diff_pos = (motions_pos - root_pos)

        root_quat = root_quat.repeat((1, diff_pos.shape[1], 1))

        motions_rbpos = math_utils.quat_rotate_inverse(root_quat, diff_pos)
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
        motions_quat_wxyz = rotations.xyzw_to_wxyz(ref_motions['rg_rot_t'][:, self.obs_motions_bodyids])

        root_quat = self.asset.data.root_quat_w[:, None, :]
        root_quat = root_quat.repeat((1, motions_quat_wxyz.shape[1], 1))

        motions_rbquat = math_utils.quat_mul(root_quat, math_utils.quat_conjugate(motions_quat_wxyz))
        #motions_rbquat = math_utils.quat_error_magnitude(root_quat, motions_quat_wxyz)
        motions_rbquat = rotations.wxyz_to_xyzw(motions_rbquat)

        motions_rbquat = torch.reshape(motions_rbquat, (-1, motions_rbquat.shape[-1]))
        motions_rbquat = torch_utils.quat_to_tan_norm(motions_rbquat)
        motions_rbquat = torch.reshape(motions_rbquat, (root_quat.shape[0], -1))
        return motions_rbquat
