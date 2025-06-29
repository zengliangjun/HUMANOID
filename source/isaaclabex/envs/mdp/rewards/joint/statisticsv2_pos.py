# Module for calculating episode rewards based on joint status.
# This module computes rewards based on differences in joint positions and their statistical properties.
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

def rew_pitch_total2zero(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    # Retrieve the articulation asset.
    asset: Articulation = env.scene[asset_cfg.name]
    # Extract current and default joint positions.
    pose = asset.data.joint_pos[:, asset_cfg.joint_ids]

    total0 = torch.abs(torch.sum(pose[:, ::2], dim=-1))
    total1 = torch.abs(torch.sum(pose[:, 1::2], dim=-1))

    reward0 = torch.exp(-total0)
    reward1 = torch.exp(-total1)

    hip_ids = asset_cfg.joint_ids[:2]
    vel = asset.data.joint_vel[:, hip_ids]
    invalid_flag0 = vel[:, 0] > 0
    invalid_flag1 = vel[:, 1] > 0
    reward0[invalid_flag0] = 0
    reward1[invalid_flag1] = 0

    return (reward0 + reward1) / 2

def _step_find_joints(asset: Articulation, j_names):
    ids = asset.find_joints(j_names, preserve_order = True)[0]
    ids = [x//2 for x in ids]
    return ids

def rew_step_mean_mean(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_statistics_name: str = "pos",
    jhipp_name: str = "left_hip_pitch_joint",
    jknee_name: str = "left_knee_joint",
    jother_names: Sequence[str] = [
                                #"left_ankle_pitch_joint",
                                "left_ankle_roll_joint",
                                "left_hip_roll_joint",
                                "left_hip_yaw_joint",
                                    ],
    jhipp_target: float = - 0.1,
    jknee_target: float = 0.4,
    jother_target: float = 0,
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    jhipp_ids = _step_find_joints(asset, [jhipp_name])
    jknee_ids = _step_find_joints(asset, [jknee_name])
    jother_ids = _step_find_joints(asset, jother_names)

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    jhipp = term.step_mean_mean_buf[:, jhipp_ids[0]]
    jknee = term.step_mean_mean_buf[:, jknee_ids[0]]
    jother = term.step_mean_mean_buf[:, jother_ids]

    jhipp = torch.abs(jhipp - jhipp_target)
    jknee = torch.abs(jknee - jknee_target)

    jother = torch.sqrt(torch.mean(torch.square(jother - jother_target), dim=1))

    reward = (torch.exp(-jhipp) + torch.exp(-jknee) + torch.exp(- jother)) / 3
    reward[term.zero_flag] = 0
    reward[term.stand_flag] = 0
    return reward


def rew_step_variance_mean(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_statistics_name: str = "pos",
    jhipp_name: str = "left_hip_pitch_joint",
    jknee_name: str = "left_knee_joint",
    jother_names: Sequence[str] = [
                                #"left_ankle_pitch_joint",
                                "left_ankle_roll_joint",
                                "left_hip_roll_joint",
                                "left_hip_yaw_joint",
                                    ],
    jhipp_target: float = 0.16,
    jknee_target: float = 0.16,
    jother_target: float = 0,
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    jhipp_ids = _step_find_joints(asset, [jhipp_name])
    jknee_ids = _step_find_joints(asset, [jknee_name])
    jother_ids = _step_find_joints(asset, jother_names)


    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    jhipp = term.step_variance_mean_buf[:, jhipp_ids[0]]
    jknee = term.step_variance_mean_buf[:, jknee_ids[0]]
    jother = term.step_variance_mean_buf[:, jother_ids]

    jhipp = torch.sqrt(torch.abs(jhipp - jhipp_target))
    jknee = torch.sqrt(torch.abs(jknee - jknee_target))

    jother = torch.sqrt(torch.sqrt(torch.mean(torch.square(jother - jother_target), dim=1)))

    reward = (torch.exp(-jhipp) + torch.exp(-jknee) + torch.exp(-jother)) / 3
    reward[term.zero_flag] = 0
    reward[term.stand_flag] = 0
    return reward

def rew_step_vv_mv(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_statistics_name: str = "pos",
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    vv = term.step_variance_variance_buf
    mv = term.step_mean_variance_buf

    vv = torch.sqrt(torch.mean(torch.square(vv), dim=1))
    mv = torch.sqrt(torch.mean(torch.square(mv), dim=1))

    reward = (torch.exp(-vv) + torch.exp(-mv)) / 2
    reward[term.zero_flag] = 0
    reward[term.stand_flag] = 0
    return reward


def rew_episode_mean(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_statistics_name: str = "pos",
    jhipp_names: str = ["left_hip_pitch_joint","right_hip_pitch_joint"],
    jknee_names: str = ["left_knee_joint","right_knee_joint"],
    jother_names: Sequence[str] = [
                                #"left_ankle_pitch_joint",
                                "left_ankle_roll_joint","right_ankle_roll_joint",
                                "left_hip_roll_joint",  "right_hip_roll_joint",
                                "left_hip_yaw_joint",   "right_hip_yaw_joint",
                                    ],
    jhipp_target: float = - 0.1,
    jknee_target: float = 0.4,
    jother_target: float = 0,
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    jhipp_ids = asset.find_joints(jhipp_names, preserve_order = True)[0]
    jknee_ids = asset.find_joints(jknee_names, preserve_order = True)[0]
    jother_ids = asset.find_joints(jother_names, preserve_order = True)[0]

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    jhipp = term.episode_mean_buf[:, jhipp_ids]
    jknee = term.episode_mean_buf[:, jknee_ids]
    jother = term.episode_mean_buf[:, jother_ids]

    jhipp = torch.linalg.norm(jhipp - jhipp_target, dim=-1)
    jknee = torch.linalg.norm(jknee - jknee_target, dim=-1)

    jother = torch.sqrt(torch.mean(torch.square(jother - jother_target), dim=1))

    reward = (torch.exp(-jhipp) + torch.exp(-jknee) + torch.exp(- jother)) / 3
    flag = torch.logical_or(term.stand_flag, term.zero_flag)
    diff_reward = torch.exp(-torch.norm(term.diff, dim = -1))
    reward[flag] = diff_reward[flag]
    return reward

def rew_episode_mean_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_statistics_name: str = "pos",
    jhipp_names: str = ["left_hip_pitch_joint","right_hip_pitch_joint"],
    jknee_names: str = ["left_knee_joint","right_knee_joint"],
    jother_names: Sequence[str] = [
                                #"left_ankle_pitch_joint",
                                "left_ankle_roll_joint","right_ankle_roll_joint",
                                "left_hip_roll_joint",  "right_hip_roll_joint",
                                "left_hip_yaw_joint",   "right_hip_yaw_joint",
                                    ],
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    jhipp_ids = asset.find_joints(jhipp_names, preserve_order = True)[0]
    jknee_ids = asset.find_joints(jknee_names, preserve_order = True)[0]
    jother_ids = asset.find_joints(jother_names, preserve_order = True)[0]

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    jhipp = term.episode_mean_buf[:, jhipp_ids]
    jknee = term.episode_mean_buf[:, jknee_ids]
    jother = term.episode_mean_buf[:, jother_ids]

    jhipp = torch.abs((jhipp[:, 0] - jhipp[:, 1]) / (jhipp[:, 0] + jhipp[:, 1] + 1e-6)) * 2
    jknee = torch.abs((jknee[:, 0] - jknee[:, 1]) / (jknee[:, 0] + jknee[:, 1] + 1e-6)) * 2
    jother = torch.sqrt(torch.mean(torch.square(jother), dim=1))

    reward = (torch.exp(-jhipp) + torch.exp(-jknee) + torch.exp(- jother)) / 3
    flag = torch.logical_or(term.stand_flag, term.zero_flag)
    diff_reward = torch.exp(-torch.norm(term.diff, dim = -1))
    reward[flag] = diff_reward[flag]
    return reward

def rew_episode_variance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_statistics_name: str = "pos",
    jhipp_names: str = ["left_hip_pitch_joint","right_hip_pitch_joint"],
    jknee_names: str = ["left_knee_joint","right_knee_joint"],
    jother_names: Sequence[str] = [
                                #"left_ankle_pitch_joint",
                                "left_ankle_roll_joint","right_ankle_roll_joint",
                                "left_hip_roll_joint",  "right_hip_roll_joint",
                                "left_hip_yaw_joint",   "right_hip_yaw_joint",
                                    ],
    jhipp_target: float = 0.16,
    jknee_target: float = 0.16,
    jother_target: float = 0,
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    jhipp_ids = asset.find_joints(jhipp_names, preserve_order = True)[0]
    jknee_ids = asset.find_joints(jknee_names, preserve_order = True)[0]
    jother_ids = asset.find_joints(jother_names, preserve_order = True)[0]

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    jhipp = term.episode_variance_buf[:, jhipp_ids]
    jknee = term.episode_variance_buf[:, jknee_ids]
    jother = term.episode_variance_buf[:, jother_ids]

    jhipp = torch.square((jhipp - jhipp_target) / jhipp_target)
    jknee = torch.square((jknee - jknee_target) / jknee_target)
    jother = torch.square(jother - jother_target)

    jhipp = torch.sqrt(torch.mean(jhipp, dim=1))
    jknee = torch.sqrt(torch.mean(jknee, dim=1))
    jother = torch.sqrt(torch.mean(jother, dim=1))

    reward = (torch.exp(-jhipp) + torch.exp(-jknee) + torch.exp(- jother)) / 3
    flag = torch.logical_or(term.stand_flag, term.zero_flag)
    diff_reward = torch.exp(-torch.norm(term.diff, dim = -1))
    reward[flag] = diff_reward[flag]
    return reward

def rew_episode_variance_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_statistics_name: str = "pos",
) -> torch.Tensor:
    """
    jnames: str = ["left_hip_pitch_joint", "right_hip_pitch_joint",
                   "left_knee_joint", "right_knee_joint",
                   "left_ankle_pitch_joint", "right_ankle_pitch_joint",
                   ],
    """
    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    jbuf = term.episode_variance_buf[:, asset_cfg.joint_ids]

    jbuf = (jbuf[:, ::2] - jbuf[:, 1::2]) / (jbuf[:, ::2] + jbuf[:, 1::2] + 1e-6) * 2
    jbuf = torch.sqrt(torch.mean(torch.square(jbuf), dim=1))

    reward = torch.exp(-jbuf)
    flag = torch.logical_or(term.stand_flag, term.zero_flag)
    diff_reward = torch.exp(-torch.norm(term.diff, dim = -1))
    reward[flag] = diff_reward[flag]
    return reward
