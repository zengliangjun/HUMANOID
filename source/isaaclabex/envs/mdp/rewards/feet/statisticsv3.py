from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING, Optional
from abc import ABC, abstractmethod

import math
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclabex.envs.rl_env_exts import ManagerBasedRLEnv


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclabex.envs.managers.statistics_manager import StatisticsManager
    from isaaclabex.envs.mdp.statistics import body

def penalize_footclearance(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    statistics_name: str = "footclearance",
) -> torch.Tensor:

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: body.StatusFootClearance = manager.get_term(statistics_name)

    episode_mean = term.episode_mean_buf
    variance_mean = term.episode_variance_buf

    error_mean = episode_mean - target_height / 2
    error_mean = torch.clamp_max(error_mean, max = 0)

    std_mean = torch.sqrt(variance_mean)
    error_std = std_mean - target_height / 2
    error_std = torch.clamp_max(error_std, max = 0)

    error = error_mean + error_std

    flag = torch.logical_or(term.stand_flag, term.zero_flag)
    stand_error = - (episode_mean + std_mean)
    error[flag] = stand_error[flag]
    return torch.abs(error[:, 0]) / std
