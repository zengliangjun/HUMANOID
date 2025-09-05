# Module for calculating episode rewards based on joint status.
# This module computes rewards based on differences in joint positions and their statistical properties.
from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclabex.envs.rl_env_exts import ManagerBasedRLEnv


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclabex.envs.managers.statistics_manager import StatisticsManager
    from isaaclabex.envs.mdp.statistics import joints


def _exp_decay(std, values: list[float]):
    count = 0
    total = None
    for id0 in range(len(values)):
        for id1 in range(id0 + 1, len(values)):
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


def rew_mean_self2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_statistics_name: str = "pos",
    symmetry: bool = True,
    constraint_range: float = None,
    std: float = 0.1
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    episode_mean = term.episode_mean_buf[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    means = [episode_mean[:, ::2], episode_mean[:, 1::2]]

    if symmetry:

        step_ids = term.step_ids(asset_cfg)
        step_mean_mean = term.step_mean_mean_buf[:, step_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids[::2]]
        means.append(step_mean_mean)

    reward = _exp_decay(std, means)

    if constraint_range is not None:
        diff_std = std * 0.5
        constraint_reward = None
        for mean in means:
            mean = torch.abs(mean)
            diff = torch.clamp(mean - constraint_range, 0, 50)
            step_reward = torch.exp(- diff / diff_std) - 0.5 * mean / diff_std

            constraint_reward = step_reward if constraint_reward is None else constraint_reward + step_reward

        reward += constraint_reward / len(means)

    reward = torch.mean(reward, dim=-1)

    flag = torch.logical_or(term.stand_flag, term.zero_flag)
    diff_reward = torch.exp(-torch.norm(term.diff, dim = -1))
    reward[flag] = diff_reward[flag]
    return reward


def rew_mean_zero2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_statistics_name: str = "pos",
    symmetry: bool = True,
    std: float = 0.25
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    episode_mean = term.episode_mean_buf[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    if symmetry:
        means = [episode_mean[:, ::2], episode_mean[:, 1::2]]

        step_ids = term.step_ids(asset_cfg)
        step_mean_mean = term.step_mean_mean_buf[:, step_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids[::2]]
        means.append(step_mean_mean)

        symmetry_diff = episode_mean[:, ::2] - episode_mean[:, 1::2]
        means.append(symmetry_diff)

    else:
        means = [episode_mean]


    reward = _exp_zero(std, means)
    reward = torch.mean(reward, dim=-1)

    flag = torch.logical_or(term.stand_flag, term.zero_flag)
    diff_reward = torch.exp(-torch.norm(term.diff, dim = -1))
    reward[flag] = diff_reward[flag]
    return reward


def rew_variance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_statistics_name: str = "pos",
    symmetry: bool = True,
    iszero: bool = False,
    constraint_range: float = None,
    std: float = 0.05,
) -> torch.Tensor:

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    episode_variance = term.episode_variance_buf[:, asset_cfg.joint_ids]

    if symmetry:
        step_ids = term.step_ids(asset_cfg)

        step_mean_variance = term.step_mean_variance_buf[:, step_ids]
        step_variance_mean = term.step_variance_mean_buf[:, step_ids]

        episode_variance0 = episode_variance[:, ::2]
        episode_variance1 = episode_variance[:, 1::2]

        means = [step_variance_mean,
                 episode_variance0,
                 episode_variance1]

        if iszero:
            reward = torch.exp(- (step_mean_variance / std)) / 4 + \
                    _exp_zero(std, means) * 3 / 4

        else:
            reward = torch.exp(- (step_mean_variance / std)) / 4 + \
                _exp_decay(std, means) * 3 / 4

            if constraint_range is not None:
                diff_std = std * 0.5
                constraint_reward = None
                for mean in means:
                    mean = torch.abs(mean)
                    diff = torch.clamp(mean - constraint_range, 0, 50)
                    step_reward = torch.exp(- diff / diff_std) - 0.5 * mean / diff_std

                    constraint_reward = step_reward if constraint_reward is None else constraint_reward + step_reward

                reward += constraint_reward / len(means)

    else:

        if constraint_range is not None:
            # TODO
            pass
        reward = _exp_zero(std, [episode_variance])

    reward = torch.mean(reward, dim=-1)


    flag = torch.logical_or(term.stand_flag, term.zero_flag)
    diff_reward = torch.exp(-torch.norm(term.diff, dim = -1))
    reward[flag] = diff_reward[flag]
    return reward
