import gymnasium as gym
from . import ppo_cfg, env_cfg

gym.register(
    id="BOTHOVERH1-v0",
    entry_point="isaaclabmotion.envs.rl_env_motions:RLMotionsENV",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:BOTH1Cfg",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:BOTH1CfgV0",
    },
)

gym.register(
    id="BOTHOVERH1-Play-v0",
    entry_point="isaaclabmotion.envs.rl_env_motions:RLMotionsENV",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:BOTH1Cfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:BOTH1CfgV0",
    },
)
