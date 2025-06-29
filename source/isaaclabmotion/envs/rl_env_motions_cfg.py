from __future__ import annotations

from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclabex.envs import rl_env_exts_cfg


@configclass
class RLMotionsENVCfg(rl_env_exts_cfg.ManagerBasedRLExtendsCfg):

    motions: object = MISSING

