import gymnasium as gym
from . import ppo_cfg, env_cfg

gym.register(
    id="G1ObsStatistic-v15",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1ObsStatisticsCfg",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1ObsStatisticCfgV1",
    },
)

gym.register(
    id="G1ObsStatistic-Play-v15",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1ObsStatisticsCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1ObsStatisticCfgV1",
    },
)
