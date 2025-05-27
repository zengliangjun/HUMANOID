
import gymnasium as gym
from . import rsl_rl_g1wm_cfg, rsl_rl_g1_cfg
from . import env_cfg

gym.register(
    id="G1WM-v0",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{rsl_rl_g1wm_cfg.__name__}:G1WMCfg",
    },
)

gym.register(
    id="G1NOWM-v0",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1NoWMEnvCfg",
        "rsl_rl_cfg_entry_point": f"{rsl_rl_g1_cfg.__name__}:G1RoughCfg",
    },
)

gym.register(
    id="G1NOWM-PLAY-v0",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1NoWMEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{rsl_rl_g1_cfg.__name__}:G1RoughCfg",
    },
)