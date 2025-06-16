import gymnasium as gym
from . import ppo_cfg, env_cfg

gym.register(
    id="G1episodeflat-v0",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1FlatCfg",
    },
)

gym.register(
    id="G1episodeflat-Play-v0",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1FlatCfg",
    },
)


gym.register(
    id="G1episodev2flat-v0",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1FlatEnvV2Cfg",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1FlatCfgV2",
    },
)

gym.register(
    id="G1episodev2flat-Play-v0",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1FlatEnvV2Cfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1FlatCfgV2",
    },
)


gym.register(
    id="G1episodev3flat-v0",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1FlatEnvV3Cfg",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1FlatCfgV3",
    },
)

gym.register(
    id="G1episodev3flat-Play-v0",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1FlatEnvV3Cfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1FlatCfgV3",
    },
)

from . import env_obs_cfg

gym.register(
    id="G1ObsStatistic-v0",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_obs_cfg.__name__}:G1ObsStatisticsCfg",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1ObsStatisticCfgV0",
    },
)

gym.register(
    id="G1ObsStatistic-Play-v0",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_obs_cfg.__name__}:G1ObsStatisticsCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1ObsStatisticCfgV0",
    },
)

from . import env_covar_cfg

gym.register(
    id="G1ObsCovar-v0",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_covar_cfg.__name__}:G1ObsCovarCfg",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1ObsCovarCfgV0",
    },
)

gym.register(
    id="G1ObsCovar-Play-v0",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_covar_cfg.__name__}:G1ObsCovarCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1ObsCovarCfgV0",
    },
)
