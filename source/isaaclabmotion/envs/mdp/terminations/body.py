from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

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

        if asset_cfg.body_names is None or len(asset_cfg.body_names) == 0:
            motion_names = self.asset.body_names
        else:
            motion_names = asset_cfg.body_names

        if "extend_body_names" in cfg.params:
            motion_names += cfg.params["extend_body_names"]

        self._motions_bodyids, _ = self.motions.resolve_motion_bodies(motion_names)

    @property
    def motions_bodyids(self):
        # TODO
        return self._motions_bodyids


    def _terminate_motion(self) -> dict:
        return self.motions.motion_ref(0)

class terminate_by_reference_motion_distance(Base):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        motions_name: str,
        max_ref_motion_dist: float,
        training_mode: bool = True
        ) -> torch.Tensor:
            """
            Determines if the distance between current body positions and reference motion positions exceeds the allowed threshold.

            Args:
                is_training (bool): Flag indicating if the system is in training mode.
                body_state (BodyState): Current state of the humanoid bodies.
                ref_motion_state (ReferenceMotionState): Reference motion state for the humanoid to track.
                max_ref_motion_dist (float): Maximum allowable distance between current and reference body positions.
                in_recovery (torch.Tensor | None): Boolean tensor (num_envs, 1) indicating if each environment is in recovery mode.
                                                If None, the recovery condition is ignored.

            Returns:
                torch.Tensor: Boolean tensor (num_envs, 1) indicating if the termination condition is met for each instance.
            """

            ref_motions = self._terminate_motion()
            motions_pos = ref_motions['rg_pos_t']

            pos = self.asset.data.body_pos_w[: , self.motions.bodyAssetToMotionIds]
            pos = torch.cat((pos, self.motions.extend_body_pos), dim = 1)

            diff_pos = (motions_pos - pos)[:, self.motions_bodyids]

            # Calculate the distance between current and reference positions
            distance = torch.norm(diff_pos, dim=-1)

            if training_mode:
                # Check if any distance exceeds the threshold
                exceeds_threshold = torch.any(distance > max_ref_motion_dist, dim=-1, keepdim=True)
            else:
                # Check if the mean distance exceeds the threshold
                mean_distance = distance.mean(dim=-1, keepdim=True)
                exceeds_threshold = torch.any(mean_distance > max_ref_motion_dist, dim=-1, keepdim=True)

            if not hasattr(env, "recovery_counters"):
                return exceeds_threshold

            # If in recovery, ensure we only terminate if not in recovery mode
            in_recovery = env.recovery_counters[:, None] > 0
            return torch.logical_and(exceeds_threshold, ~in_recovery)
