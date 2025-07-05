from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Sequence

import torch
from isaaclabex.envs.rl_env_exts import ManagerBasedRLEnv


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclabex.envs.managers.statistics_manager import StatisticsManager

    from isaaclabmotion.envs.managers import motions_manager
    from isaaclabmotion.envs.mdp.statistics import robotbody

def rew_mean_rbpos_headdiff(
    env: ManagerBasedRLEnv,
    motions_name: str,
    statistics_name: str,
    body_names: Sequence[str] = None,
    std: float = 0.25,
) -> torch.Tensor:

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    statistics: robotbody.RBPosHeadDiff = manager.get_term(statistics_name)

    manager: motions_manager.MotionsManager = env.motions_manager
    motions: motions_manager.MotionsTerm = manager.get_term(motions_name)

    if body_names is None:
        body_ids = slice(None)
    else:
        body_ids, _ = motions.resolve_motion_bodies(body_names)

    episode_mean = statistics.episode_mean_buf[:, body_ids]
    episode_mean = torch.reshape(episode_mean, (env.num_envs, -1))

    r = torch.exp(- torch.square(episode_mean) / std)
    r = torch.mean(r, dim=-1)

    r[statistics.zero_flag] = 0
    return r

def rew_variance_rbpos_headdiff(
    env: ManagerBasedRLEnv,
    motions_name: str,
    statistics_name: str,
    body_names: Sequence[str] = None,
    std: float = 0.25,
) -> torch.Tensor:

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    statistics: robotbody.RBPosHeadDiff = manager.get_term(statistics_name)

    manager: motions_manager.MotionsManager = env.motions_manager
    motions: motions_manager.MotionsTerm = manager.get_term(motions_name)

    if body_names is None:
        body_ids = slice(None)
    else:
        body_ids, _ = motions.resolve_motion_bodies(body_names)

    episode_var = statistics.episode_variance_buf[:, body_ids]
    episode_var = torch.reshape(episode_var, (env.num_envs, -1))

    r = torch.exp(- episode_var / std)
    r = torch.mean(r, dim=-1)

    r[statistics.zero_flag] = 0
    return r


def rew_mean_root_headdiff(
    env: ManagerBasedRLEnv,
    statistics_name: str,
    std: float = 0.25,
    z_weight: float = 3,
) -> torch.Tensor:

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    statistics: robotbody.RBPosHeadDiff = manager.get_term(statistics_name)

    episode_mean = statistics.episode_mean_buf
    episode_mean = torch.abs(episode_mean)

    rxy = torch.exp(- torch.mean(episode_mean[:, :2], dim = 1)  / std)
    rz = torch.exp(- episode_mean[:, 2] * z_weight / std) * z_weight

    r = (rxy + rz) / (1 + z_weight)
    return r

def rew_variance_root_headdiff(
    env: ManagerBasedRLEnv,
    statistics_name: str,
    std: float = 0.25,
    z_weight: float = 3,
) -> torch.Tensor:

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    statistics: robotbody.RBPosHeadDiff = manager.get_term(statistics_name)

    episode_var = statistics.episode_variance_buf
    episode_var = torch.sqrt(episode_var)

    rxy = torch.exp(- torch.mean(episode_var[:, :2], dim = 1)  / std)
    rz = torch.exp(- episode_var[:, 2] * z_weight / std) * z_weight

    r = (rxy + rz) / (1 + z_weight)
    return r
