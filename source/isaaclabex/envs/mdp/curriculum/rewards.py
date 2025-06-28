from __future__ import annotations
import torch
from collections.abc import Sequence
from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass
from typing import TYPE_CHECKING
from dataclasses import MISSING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

def curriculum_with_steps(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],

    start_steps,
    end_steps,
    curriculums
) -> torch.Tensor:

    steps = env.common_step_counter
    if start_steps > steps:
        scale = 0
    elif end_steps < steps:
        scale = 1
    else:
        scale = (steps - start_steps) / (end_steps - start_steps)

    reward_manager = env.reward_manager
    for reward_name, curriculum_dicts in curriculums.items():
        cfg: RewardTermCfg = reward_manager.get_term_cfg(reward_name)

        cfg.weight = curriculum_dicts["start_weight"] + \
                     (curriculum_dicts["end_weight"] - curriculum_dicts["start_weight"]) * scale

    return torch.tensor(scale, device=env.device)

