import gymnasium as gym
from . import ppo_cfg, env_cfg

gym.register(
    id="HOVERH1-v0",
    entry_point="isaaclabmotion.envs.rl_env_motions:RLMotionsENV",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:OMNIH2OH1Cfg",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:OMNIH2OH1CfgV0",
    },
)

gym.register(
    id="HOVERH1-Play-v0",
    entry_point="isaaclabmotion.envs.rl_env_motions:RLMotionsENV",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:OMNIH2OH1Cfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:OMNIH2OH1CfgV0",
    },
)

"""

"""
from . import env_cfgv1

gym.register(
    id="HOVERH1-v1",
    entry_point="isaaclabmotion.envs.rl_env_motions:RLMotionsENV",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfgv1.__name__}:OMNIH2OH1CfgV1",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:OMNIH2OH1CfgV1",
    },
)

gym.register(
    id="HOVERH1-Play-v1",
    entry_point="isaaclabmotion.envs.rl_env_motions:RLMotionsENV",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfgv1.__name__}:OMNIH2OH1CfgV1_PLAY",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:OMNIH2OH1CfgV1",
    },
)

from . import env_cfgv2

gym.register(
    id="HOVERH1-v2",
    entry_point="isaaclabmotion.envs.rl_env_motions:RLMotionsENV",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfgv2.__name__}:OMNIH2OH1CfgV2",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:OMNIH2OH1CfgV2",
    },
)

gym.register(
    id="HOVERH1-Play-v2",
    entry_point="isaaclabmotion.envs.rl_env_motions:RLMotionsENV",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfgv2.__name__}:OMNIH2OH1CfgV2_PLAY",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:OMNIH2OH1CfgV2",
    },
)
