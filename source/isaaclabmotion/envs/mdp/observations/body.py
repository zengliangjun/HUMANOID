from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.assets import Articulation

from isaaclabmotion.envs.managers import motions_manager
import isaaclab.utils.math as isaac_math_utils
from extends,isaac_utils import math_utils, torch_utils, rotations

DEBUG = False

class Base(ManagerTermBase):
    """Base class for joint statistics calculation with common functionality."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        motions_name: str = cfg.params["motions_name"]

        self.motions: motions_manager.MotionsTerm = env.motions_manager.get_term(motions_name)

        if "extend_body_names" in cfg.params:
            extend_body_names: list = cfg.params["extend_body_names"]
            self.extend_body_ids, _ = self.motions.resolve_extend_bodies(extend_body_names)

        self.asset: Articulation = env.scene[asset_cfg.name]
        self.body_ids = asset_cfg.body_ids

    def heading_quat_inv_wxyz(self):
        root_rot = math_utils.convert_quat(self.asset.data.root_quat_w, to="xyzw")
        heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)  # xyzw
        return math_utils.convert_quat(heading_rot_inv, to="wxyz")


class obs_body_pos(Base):
    """Base class for joint statistics calculation with common functionality."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        motions_name: str,
        extend_body_names: Sequence[str] = None
    ) -> torch.Tensor:

        body_pos = self.asset.data.body_pos_w[:, asset_cfg.body_ids]
        if extend_body_names != None and len(extend_body_names) > 0:
            extend_pos = self.motions.extend_body_pos[:, self.extend_body_ids]
            body_pos = torch.cat((body_pos, extend_pos), dim = 1)

        root_pos = self.asset.data.root_pos_w

        root_pos_extend = root_pos.unsqueeze(-2)
        local_body_pos = body_pos - root_pos_extend

        inv_wxyz = self.heading_quat_inv_wxyz()[:, None, :].repeat((1, local_body_pos.shape[1], 1))

        body_pos_lb = isaac_math_utils.quat_rotate(inv_wxyz, local_body_pos)

        if DEBUG:
            num_envs, num_bodies, _ = body_pos.shape

            root_rot = math_utils.convert_quat(self.asset.data.root_quat_w, to="xyzw")
            heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)  # xyzw
            heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
            heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, num_bodies, 1))
            flat_heading_rot_inv = heading_rot_inv_expand.reshape(num_envs * num_bodies, 4)

            flat_local_body_pos = local_body_pos.reshape(num_envs * num_bodies, 3)
            flat_local_body_pos = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_local_body_pos)  # input xyzw
            flat_local_body_pos = flat_local_body_pos.reshape(num_envs, num_bodies, 3)
            diff = body_pos_lb - flat_local_body_pos

        return body_pos_lb.flatten(1)

class obs_body_rotwxyz(Base):

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        motions_name: str,
        extend_body_names: Sequence[str] = None
    ) -> torch.Tensor:

        bodyrot_wxyz = self.asset.data.body_quat_w[:, asset_cfg.body_ids]
        if extend_body_names != None and len(extend_body_names) > 0:
            extend_rot = self.motions.extend_body_rot_wxyz[:, self.extend_body_ids]
            bodyrot_wxyz = torch.cat((bodyrot_wxyz, extend_rot), dim = 1)

        quatinv_wxyz = self.heading_quat_inv_wxyz()[:, None, :].repeat((1, bodyrot_wxyz.shape[1], 1))

        lbody_wxyz = isaac_math_utils.quat_mul(quatinv_wxyz, bodyrot_wxyz)
        lbody_xyzw = rotations.wxyz_to_xyzw(lbody_wxyz)
        lbody_xyzw = lbody_xyzw.reshape(-1, 4)
        tan_norm = torch_utils.quat_to_tan_norm(lbody_xyzw)
        tan_norm = tan_norm.reshape(bodyrot_wxyz.shape[0], -1)

        if DEBUG:
            num_envs, num_bodies, _ = bodyrot_wxyz.shape

            root_rot = math_utils.convert_quat(self.asset.data.root_quat_w, to="xyzw")
            heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)  # xyzw
            heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
            heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, num_bodies, 1))
            flat_heading_rot_inv = heading_rot_inv_expand.reshape(num_envs * num_bodies, 4)

            # body quat and normalize to egocentric (for angle only yaw)
            flat_body_rot = bodyrot_wxyz.reshape(num_envs * num_bodies, 4)
            flat_local_body_rot = math_utils.quat_mul(
                math_utils.convert_quat(flat_heading_rot_inv, to="wxyz"), flat_body_rot
            )  # input wxyz, output wxyz
            flat_local_body_rot = math_utils.convert_quat(flat_local_body_rot, to="xyzw")
            flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(
                flat_local_body_rot
            )  # Shape becomes (num_envs, num_bodies * 6)
            flat_local_body_rot_obs = flat_local_body_rot_obs.reshape(bodyrot_wxyz.shape[0], -1)
            diff = tan_norm - flat_local_body_rot_obs

        return tan_norm.flatten(1)

class obs_body_lin_vel(Base):
    """Base class for joint statistics calculation with common functionality."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        motions_name: str,
        extend_body_names: Sequence[str] = None
    ) -> torch.Tensor:

        lin_vel = self.asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
        if extend_body_names != None and len(extend_body_names) > 0:
            extend_vel = self.motions.extend_body_lin_vel[:, self.extend_body_ids]
            lin_vel = torch.cat((lin_vel, extend_vel), dim = 1)

        inv_wxyz = self.heading_quat_inv_wxyz()[:, None, :].repeat((1, lin_vel.shape[1], 1))

        lin_vel_lb = isaac_math_utils.quat_rotate(inv_wxyz, lin_vel)

        if DEBUG:
            num_envs, num_bodies, _ = lin_vel.shape

            root_rot = math_utils.convert_quat(self.asset.data.root_quat_w, to="xyzw")
            heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)  # xyzw
            heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
            heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, num_bodies, 1))
            flat_heading_rot_inv = heading_rot_inv_expand.reshape(num_envs * num_bodies, 4)

            flat_body_vel = lin_vel.reshape(num_envs * num_bodies, 3)
            flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_body_vel)
            local_body_vel = flat_local_body_vel.reshape(num_envs, num_bodies, 3)

            diff = lin_vel_lb - local_body_vel

        return lin_vel_lb.flatten(1)

class obs_body_ang_vel(Base):
    """Base class for joint statistics calculation with common functionality."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        motions_name: str,
        extend_body_names: Sequence[str] = None
    ) -> torch.Tensor:

        ang_vel = self.asset.data.body_ang_vel_w[:, asset_cfg.body_ids]
        if extend_body_names != None and len(extend_body_names) > 0:
            extend_ang = self.motions.extend_body_ang_vel[:, self.extend_body_ids]
            ang_vel = torch.cat((ang_vel, extend_ang), dim = 1)

        inv_wxyz = self.heading_quat_inv_wxyz()[:, None, :].repeat((1, ang_vel.shape[1], 1))

        ang_vel_lb = isaac_math_utils.quat_rotate(inv_wxyz, ang_vel)

        if DEBUG:
            num_envs, num_bodies, _ = ang_vel.shape

            root_rot = math_utils.convert_quat(self.asset.data.root_quat_w, to="xyzw")
            heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)  # xyzw
            heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
            heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, num_bodies, 1))
            flat_heading_rot_inv = heading_rot_inv_expand.reshape(num_envs * num_bodies, 4)

            flat_body_vel = ang_vel.reshape(num_envs * num_bodies, 3)
            flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_body_vel)
            local_body_vel = flat_local_body_vel.reshape(num_envs, num_bodies, 3)

            diff = ang_vel_lb - local_body_vel

        return ang_vel_lb.flatten(1)
