from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from collections.abc import Sequence
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

class penalize_action_smoothness(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev_prev_action = torch.zeros_like(env.action_manager.action)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if len(env_ids) == 0:
            return
        self.prev_prev_action[env_ids] = 0

    def __call__(self, env: ManagerBasedRLEnv,
                 weight1: float = 1,
                 weight2: float = 1,
                 weight3: float = 0.05,
                 ):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1) * weight1
        term_2 = torch.sum(torch.square(
            env.action_manager.action + self.prev_prev_action - 2 * env.action_manager.prev_action), dim=1) * weight2
        term_3 =  torch.sum(torch.abs(env.action_manager.action), dim=1) * weight3

        self.prev_prev_action[...] = env.action_manager.prev_action
        return term_1 + term_2 + term_3
