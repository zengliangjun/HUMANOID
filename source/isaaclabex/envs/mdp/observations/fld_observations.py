from isaaclab.envs import ManagerBasedEnv
from isaaclabex.envs.managers.statistics_manager import StatisticsManager
import torch
from isaaclabex.envs.mdp.statistics import fldstatus

def obs_fld_params(env: ManagerBasedEnv,
    statistics_name: str = "fld_status") -> torch.Tensor:

    manager: StatisticsManager = env.statistics_manager
    term: fldstatus.FLDCollect = manager.get_term(statistics_name)

    return term.fld_params.flatten(start_dim=1)

