from __future__ import annotations
import torch
from collections.abc import Sequence
from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass
from typing import TYPE_CHECKING
from dataclasses import MISSING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg

@configclass
class EventCurriculumStepItem:
    #start_steps: int = MISSING
    #end_steps: int = MISSING

    start_range: tuple[float, float] = MISSING
    end_range: tuple[float, float] = MISSING

def _calcute_curriculum_step(_scale, cfg: EventTermCfg, curriculum_dicts: dict):
    for curriculum_name, subitem in curriculum_dicts.items():
        if 0 == _scale:
            cfg.params[curriculum_name] = subitem.start_range
        elif 1 == _scale:
            cfg.params[curriculum_name] = subitem.end_range
        else:
            if isinstance(subitem.start_range, dict):

                for key, start_range, in subitem.start_range.items():
                    end_range = subitem.end_range[key]
                    cfg.params[curriculum_name][key] = (
                    (end_range[0] - start_range[0]) * _scale + start_range[0],
                    (end_range[1] - start_range[1]) * _scale + start_range[1],
                )


            else:
                cfg.params[curriculum_name] = (
                    (subitem.end_range[0] - subitem.start_range[0]) * _scale + subitem.start_range[0],
                    (subitem.end_range[1] - subitem.start_range[1]) * _scale + subitem.start_range[1],
                )


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

    event_manager = env.event_manager
    for event_name, curriculum_dicts in curriculums.items():
        cfg: EventTermCfg = event_manager.get_term_cfg(event_name)

        _calcute_curriculum_step(scale, cfg, curriculum_dicts)

    return torch.tensor(scale, device=env.device)
