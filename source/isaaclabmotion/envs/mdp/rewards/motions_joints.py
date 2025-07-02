from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclabmotion.envs.managers import motions_manager

class Base(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        motions_name: str = cfg.params["motions_name"]

        self.motions: motions_manager.MotionsTerm = env.motions_manager.get_term(motions_name)

        self.asset: Articulation = env.scene[asset_cfg.name]


    def _reward_motion(self) -> dict:
        return self.motions.motion_ref(0)


class reward_track_joint_positions(Base):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        motions_name: str,
        sigma: float = 1
    ) -> torch.Tensor:

        ref_motions = self._reward_motion()
        motions_dof_pos = ref_motions['dof_pos']

        dof_pos = self.asset.data.joint_pos[:, self.motions.joint_ids]

        diff = (motions_dof_pos - dof_pos)[:, asset_cfg.joint_ids]
        diff_squared = torch.mean(torch.square(diff), dim=1)
        return torch.exp(- diff_squared / sigma)


class reward_track_joint_velocities(Base):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        motions_name: str,
        sigma: float = 1
    ) -> torch.Tensor:

        ref_motions = self._reward_motion()
        motions_dof_pos = ref_motions['dof_vel']

        dof_pos = self.asset.data.joint_vel[:, self.motions.joint_ids]

        diff = (motions_dof_pos - dof_pos)[:, asset_cfg.joint_ids]
        diff_squared = torch.mean(torch.square(diff), dim=1)
        return torch.exp(- diff_squared / sigma)
