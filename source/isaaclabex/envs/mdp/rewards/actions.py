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
        # Initialize previous-previous action for smoothness penalty computation.
        self.prev_prev_action = torch.zeros_like(env.action_manager.action)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Resets the stored previous action for the specified environment IDs.
        # env_ids: sequence of environment indices to reset.
        if len(env_ids) == 0:
            return
        self.prev_prev_action[env_ids] = 0

    def __call__(self, env: ManagerBasedRLEnv,
                 weight1: float = 1,    # Weight for penalizing first-order difference (current vs previous)
                 weight2: float = 1,    # Weight for penalizing second-order difference (acceleration)
                 weight3: float = 0.05, # Weight for penalizing absolute action magnitude
                 ):
        """
        Computes the smoothness penalty on actions.

        Parameters:
            env (ManagerBasedRLEnv): Environment holding the current and previous actions.
            weight1 (float): Factor for penalizing the difference between current and previous actions.
            weight2 (float): Factor for penalizing the second derivative (difference of differences).
            weight3 (float): Factor for penalizing the absolute magnitude of actions.

        Returns:
            Tensor: Total smoothness penalty.
        """
        # Term for penalizing deviation between current and previous action.
        term_1 = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1) * weight1
        # Term for penalizing second-order difference (action acceleration).
        term_2 = torch.sum(torch.square(
            env.action_manager.action + self.prev_prev_action - 2 * env.action_manager.prev_action), dim=1) * weight2
        # Term for penalizing high action magnitude via L1 norm.
        term_3 = torch.sum(torch.abs(env.action_manager.action), dim=1) * weight3

        # Update previous previous action for the next iteration.
        self.prev_prev_action[...] = env.action_manager.prev_action
        return term_1 + term_2 + term_3

def action_rate2_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    diff = env.action_manager.action - env.action_manager.prev_action
    diff = diff[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(diff), dim=1)
