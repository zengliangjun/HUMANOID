
import gymnasium as gym
from . import rsl_rl_ppo_cfg
from . import g1_orgenv

gym.register(
    id="G1Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{g1_orgenv.__name__}:G1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:G1RoughCfg",
    },
)

gym.register(
    id="G1Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{g1_orgenv.__name__}:G1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:G1RoughCfg",
    },
)

from . import g1_no_scan

gym.register(
    id="G1RoughNoScan-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{g1_no_scan.__name__}:G1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:G1RoughNoSCanCfg",
    },
)

gym.register(
    id="G1RoughNoScan-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{g1_no_scan.__name__}:G1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:G1RoughNoSCanCfg",
    },
)
