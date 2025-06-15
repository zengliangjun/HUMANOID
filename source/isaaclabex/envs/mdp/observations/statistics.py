from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
import torch
from isaaclab.assets import Articulation
from isaaclabex.envs.mdp.statistics import joints

def obs_episode_mean(env: ManagerBasedEnv,
    pos_statistics_name: str = "pos") -> torch.Tensor:

    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    return term.episode_mean_buf

def obs_episode_variance(env: ManagerBasedEnv,
    pos_statistics_name: str = "pos") -> torch.Tensor:

    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    return torch.sqrt(term.episode_variance_buf)

def obs_step_mean_mean(env: ManagerBasedEnv,
    pos_statistics_name: str = "pos") -> torch.Tensor:

    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    return term.step_mean_mean_buf

def obs_step_mean_variance(env: ManagerBasedEnv,
    pos_statistics_name: str = "pos") -> torch.Tensor:

    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    return torch.sqrt(term.step_mean_variance_buf)

def obs_step_variance_mean(env: ManagerBasedEnv,
    pos_statistics_name: str = "pos") -> torch.Tensor:

    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    return torch.sqrt(term.step_variance_mean_buf)
