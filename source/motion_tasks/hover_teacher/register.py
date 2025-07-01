import gymnasium as gym
from . import ppo_cfg, env_cfg

gym.register(
    id="OMNIH2OH1-v0",
    entry_point="isaaclabmotion.envs.rl_env_motions:RLMotionsENV",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:OMNIH2OH1Cfg",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:OMNIH2OH1CfgV0",
    },
)

gym.register(
    id="OMNIH2OH1-Play-v0",
    entry_point="isaaclabmotion.envs.rl_env_motions:RLMotionsENV",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:OMNIH2OH1Cfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:OMNIH2OH1CfgV0",
    },
)
