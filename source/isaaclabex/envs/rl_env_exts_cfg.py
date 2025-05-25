
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg

@configclass
class ManagerBasedRLExtendsCfg(ManagerBasedRLEnvCfg):
    '''
    used for calcute average_episode_length
    '''
    debug_flags: bool = False

    '''
    for reward penalty curriculum
    # NOTE it is used after reset
    '''
    num_compute_average_epl: int = 10000
