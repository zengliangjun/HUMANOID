import gymnasium as gym
from . import ppo_cfg, env_cfg

gym.register(
    id="G1ObsStatistic-v1",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1ObsStatisticsCfg",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1ObsStatisticCfgV1",
    },
)

gym.register(
    id="G1ObsStatistic-PLANE-v1",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1ObsStatistics_PLANE",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1ObsStatisticCfgV1",
    },
)

gym.register(
    id="G1ObsStatistic-Play-v1",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1ObsStatisticsCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1ObsStatisticCfgV1",
    },
)

gym.register(
    id="G1ObsStatistic-PLANE-Play-v1",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1ObsStatisticsCfg_PLANE_PLAY",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1ObsStatisticCfgV1",
    },
)


from tasks.g1_12dofv0 import ppo_cfg as org_ppo_cfg

gym.register(
    id="G1ObsStatistic-TSPTrain-v1",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1ObsStatistics_TSPTrainCfg",
        "rsl_rl_cfg_entry_point": f"{org_ppo_cfg.__name__}:G1ObsStatisticCfgV1",
    },
)
