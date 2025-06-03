import gymnasium as gym
from . import rsl_rl_ppo_cfg, env_cfg


gym.register(
    id="G112Flat-v0",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:G1FlatCfg",
    },
)

gym.register(
    id="G112Flat-Play-v0",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:G1FlatCfg",
    },
)

gym.register(
    id="G112FlatEx-v0",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1FlatEnvExCfg",
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:G1FlatExCfg",
    },
)

gym.register(
    id="G112FlatEx-Play-v0",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1FlatEnvExCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:G1FlatExCfg",
    },
)
