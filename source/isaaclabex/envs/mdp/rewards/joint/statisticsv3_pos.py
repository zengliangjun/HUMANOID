from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING, Optional
from abc import ABC, abstractmethod

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclabex.envs.rl_env_exts import ManagerBasedRLEnv


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclabex.envs.managers.statistics_manager import StatisticsManager
    from isaaclabex.envs.mdp.statistics import joints

def _exp_decay(std, values: list[float]):
    count = 0
    total = None
    for id0 in range(len(values)):
        for id1 in range(1, len(values)):
            if id0 == id1:
                continue
            diff = torch.abs(values[id0] - values[id1]) / std
            if None == total:
                total = torch.exp(-diff)
            else:
                total += torch.exp(-diff)

            count += 1

    return total / count

def _exp_zero(std, values: list[float]):

    total = None
    for id, value in enumerate(values):
        if None == total:
            total = torch.exp(-torch.abs(value) / std)
        else:
            total += torch.exp(-torch.abs(value) / std)

    return total / len(values)


def rew_mean_self(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_statistics_name: str = "pos",
    std: float = 0.25
) -> torch.Tensor:

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    episode_mean = term.episode_mean_buf[:, asset_cfg.joint_ids]

    step_ids = [x // 2 for x in asset_cfg.joint_ids[::2]]
    step_mean_mean = term.step_mean_mean_buf[:, step_ids]
    episode_mean0 = episode_mean[:, ::2]
    episode_mean1 = episode_mean[:, ::2]
    reward = _exp_decay(std, [step_mean_mean, episode_mean0, episode_mean1])
    reward = torch.mean(reward, dim=-1)

    flag = torch.logical_or(term.stand_flag, term.zero_flag)
    diff_reward = torch.exp(-torch.norm(term.diff, dim = -1))
    reward[flag] = diff_reward[flag]
    return reward

def rew_mean_zero(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_statistics_name: str = "pos",
    std: float = 0.25
) -> torch.Tensor:

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    episode_mean = term.episode_mean_buf[:, asset_cfg.joint_ids]

    step_ids = [x // 2 for x in asset_cfg.joint_ids[::2]]
    step_mean_mean = term.step_mean_mean_buf[:, step_ids]
    episode_mean0 = episode_mean[:, ::2]
    episode_mean1 = episode_mean[:, ::2]
    reward = _exp_zero(std, [step_mean_mean, episode_mean0, episode_mean1])
    reward = torch.mean(reward, dim=-1)

    flag = torch.logical_or(term.stand_flag, term.zero_flag)
    diff_reward = torch.exp(-torch.norm(term.diff, dim = -1))
    reward[flag] = diff_reward[flag]
    return reward

def rew_variance_self(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_statistics_name: str = "pos",
    std: float = 0.05
) -> torch.Tensor:

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    episode_variance = term.episode_variance_buf[:, asset_cfg.joint_ids]

    step_ids = [x // 2 for x in asset_cfg.joint_ids[::2]]

    step_mean_variance = term.step_mean_variance_buf[:, step_ids]
    step_variance_mean = term.step_variance_mean_buf[:, step_ids]

    episode_variance0 = episode_variance[:, ::2]
    episode_variance1 = episode_variance[:, ::2]

    reward = _exp_decay(std, [step_mean_variance,
                            step_variance_mean,
                            episode_variance0,
                            episode_variance1])
    reward = torch.mean(reward, dim=-1)

    flag = torch.logical_or(term.stand_flag, term.zero_flag)
    diff_reward = torch.exp(-torch.norm(term.diff, dim = -1))
    reward[flag] = diff_reward[flag]
    return reward

def rew_variance_zero(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_statistics_name: str = "pos",
    std: float = 0.05
) -> torch.Tensor:

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    episode_variance = term.episode_variance_buf[:, asset_cfg.joint_ids]
    episode_variance0 = episode_variance[:, ::2]
    episode_variance1 = episode_variance[:, ::2]

    reward = _exp_zero(std, [episode_variance0, episode_variance1])
    reward = torch.mean(reward, dim=-1)

    flag = torch.logical_or(term.stand_flag, term.zero_flag)
    diff_reward = torch.exp(-torch.norm(term.diff, dim = -1))
    reward[flag] = diff_reward[flag]
    return reward

