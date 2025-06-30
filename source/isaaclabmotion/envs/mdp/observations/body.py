from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.assets import Articulation

from isaaclabmotion.envs.managers import motions_manager
import isaaclab.utils.math as math_utils

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

        body_pos_lb = math_utils.quat_rotate_inverse(self.asset.data.root_quat_w[:, None, :], local_body_pos)
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

        rot_wxyz = self.asset.data.body_quat_w[:, asset_cfg.body_ids]
        if extend_body_names != None and len(extend_body_names) > 0:
            extend_rot = self.motions.extend_body_rot_wxyz[:, self.extend_body_ids]
            rot_wxyz = torch.cat((rot_wxyz, extend_rot), dim = 1)

        rot_lb = math_utils.quat_error_magnitude(self.asset.data.root_quat_w[:, None, :], rot_wxyz)
        return rot_lb.flatten(1)

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

        lin_vel_lb = math_utils.quat_rotate_inverse(self.asset.data.root_quat_w[:, None, :], lin_vel)
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

        ang_vel = self.asset.data.body_lin_ang_w[:, asset_cfg.body_ids]
        if extend_body_names != None and len(extend_body_names) > 0:
            extend_ang = self.motions.extend_body_ang_vel[:, self.extend_body_ids]
            ang_vel = torch.cat((ang_vel, extend_ang), dim = 1)

        ang_vel_lb = math_utils.quat_rotate_inverse(self.asset.data.root_quat_w[:, None, :], ang_vel)
        return ang_vel_lb.flatten(1)
