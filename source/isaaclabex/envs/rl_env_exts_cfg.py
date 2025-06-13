from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg

@configclass
class ManagerBasedRLExtendsCfg(ManagerBasedRLEnvCfg):
    """
    Environment extension configuration for humanoid simulation.

    This extension provides advanced features such as:
      - Computation of average episode length.
      - Reward penalty and curriculum adaptations.
      - Custom environment step and reset handling.

    Inherits from ManagerBasedRLEnvCfg.
    """
    debug_flags: bool = False

    # For reward penalty and curriculum (applied after reset)
    num_compute_average_epl: int = 10000

    # Reward positive flag
    reward_positive_flag: bool = False

    # copy from agent cfg
    num_steps_per_env: int = 24
    max_iterations: int = 5000


    # Statistics manager configuration
    statistics: object | None = None

