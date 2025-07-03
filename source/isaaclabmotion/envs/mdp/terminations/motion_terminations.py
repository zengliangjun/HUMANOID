from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclabmotion.envs.managers import motions_manager
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def motion_termination(env: ManagerBasedRLEnv,
        motions_name: str) -> torch.Tensor:

    motions: motions_manager.MotionsTerm = env.motions_manager.get_term(motions_name)
    return motions.termination_compute()
