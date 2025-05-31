
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
import torch
from isaaclab.assets import Articulation, RigidObject
import numpy as np

def stance_status(env: ManagerBasedEnv, phase_name: str, velocity_name: str) -> torch.Tensor:
    leg_phase = env.command_manager.get_command(phase_name)
    is_stance = leg_phase < 0.55
    return is_stance

