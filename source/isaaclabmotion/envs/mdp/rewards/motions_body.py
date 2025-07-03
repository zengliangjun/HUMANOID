import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.assets import Articulation

from isaaclabmotion.envs.managers import motions_manager
import isaaclab.utils.math as math_utils
from extends.isaac_utils import rotations

class Base(ManagerTermBase):
    """Base class for joint statistics calculation with common functionality."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        motions_name: str = cfg.params["motions_name"]

        self.motions: motions_manager.MotionsTerm = env.motions_manager.get_term(motions_name)

        self.asset: Articulation = env.scene[asset_cfg.name]

        if asset_cfg.body_names is None or len(asset_cfg.body_names) == 0:
            motion_names = []
        else:
            motion_names = asset_cfg.body_names

        if "extend_body_names" in cfg.params:
            motion_names += cfg.params["extend_body_names"]

        self._reward_motions_bodyids, _ = self.motions.resolve_motion_bodies(motion_names)

    def _reward_motion(self) -> dict:
        return self.motions.motion_ref(0)

    @property
    def reward_motions_bodyids(self):
        # TODO
        return self._reward_motions_bodyids


class reward_track_body_lin_vel(Base):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        motions_name: str,
        extend_body_names: list = None,
        sigma: float = 1
    ) -> torch.Tensor:

        ref_motions = self._reward_motion()
        motions_lin = ref_motions['body_vel_t']

        lin = self.asset.data.body_lin_vel_w[: , self.motions.body_ids]
        lin = torch.cat((lin, self.motions.extend_body_lin_vel), dim = 1)

        diff_lin = (motions_lin - lin)[:, self.reward_motions_bodyids]
        diff_lin = (diff_lin **2).mean(dim=-1).mean(dim=-1)
        return torch.exp(-diff_lin / sigma)


class reward_track_body_ang_vel(Base):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        motions_name: str,
        extend_body_names: list = None,
        sigma: float = 1
    ) -> torch.Tensor:

        ref_motions = self._reward_motion()
        motions_ang = ref_motions['body_ang_vel_t']

        ang = self.asset.data.body_ang_vel_w[: , self.motions.body_ids]
        ang = torch.cat((ang, self.motions.extend_body_ang_vel), dim = 1)

        diff_ang = (motions_ang - ang)[:, self.reward_motions_bodyids]
        diff_ang = (diff_ang **2).mean(dim=-1).mean(dim=-1)
        return torch.exp(-diff_ang / sigma)


class reward_track_body_pos(Base):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        motions_name: str,
        extend_body_names: list = None,
        sigma: float = 1
    ) -> torch.Tensor:

        ref_motions = self._reward_motion()
        motions_pos = ref_motions['rg_pos_t']

        pos = self.asset.data.body_pos_w[: , self.motions.body_ids]
        pos = torch.cat((pos, self.motions.extend_body_pos), dim = 1)

        diff_pos = (motions_pos - pos)[:, self.reward_motions_bodyids]
        diff_pos = (diff_pos **2).mean(dim=-1).mean(dim=-1)
        return torch.exp(-diff_pos / sigma)

class reward_track_body_quat(Base):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        motions_name: str,
        extend_body_names: list = None,
        sigma: float = 1
    ) -> torch.Tensor:

        ref_motions = self._reward_motion()
        motions_quat_wxyz = rotations.xyzw_to_wxyz(ref_motions['rg_rot_t'])[:, self.reward_motions_bodyids]

        quat = self.asset.data.body_quat_w[: , self.motions.body_ids]
        quat = torch.cat((quat, self.motions.extend_body_rot_wxyz), dim = 1)[:, self.reward_motions_bodyids]

        diff_quat = math_utils.quat_error_magnitude(motions_quat_wxyz, quat)
        diff_angle = math_utils.euler_xyz_from_quat(diff_quat)
        diff_angle = torch.stack(diff_angle, dim = 2)

        diff_angle = (diff_angle **2).mean(dim=-1).mean(dim=-1)
        return torch.exp(-diff_angle / sigma)
