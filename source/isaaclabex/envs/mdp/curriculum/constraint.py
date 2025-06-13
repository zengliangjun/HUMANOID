from __future__ import annotations
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclabex.envs.managers.term_cfg import ConstraintTermCfg

def curriculum_with_steps(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    start_curriculum_steps,
    end_curriculum_steps,
    start_probability,
    end_probability,
    curriculums
) -> torch.Tensor:

    steps = env.common_step_counter
    if steps < start_curriculum_steps:
        scale = 0
    else:
        scale = (steps - start_curriculum_steps) / (end_curriculum_steps - start_curriculum_steps)

    if 0 == scale:
        probability_max = start_probability
    else:
        probability_max = end_probability - start_probability
        T_start = 20
        T_end = max(1.0 / probability_max, 1e-6)
        probability_max = 1.0 / (T_start + scale * (T_end - T_start)) + start_probability

    termination_manager = env.termination_manager
    for event_name in curriculums:
        cfg: ConstraintTermCfg = termination_manager.get_term_cfg(event_name)
        cfg.probability_max = probability_max

    return torch.tensor(probability_max, device=env.device)
