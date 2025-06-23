import gymnasium as gym
from . import ppo_cfg, env_cfg

gym.register(
    id="G129dofObsStatistic-v0",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G129dofObsStatisticCfgV0",
    },
)

gym.register(
    id="G129dofObsStatistic-Play-v0",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G129dofObsStatisticCfgV0",
    },
)
