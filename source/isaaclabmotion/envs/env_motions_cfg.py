from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.envs.manager_based_env_cfg import ManagerBasedEnvCfg

@configclass
class ManagerMotionsCfg(ManagerBasedEnvCfg):

    motions: object = MISSING

