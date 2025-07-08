
from dataclasses import MISSING
from typing import Literal
import torch

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

@configclass
class BOTActorCriticCfg(RslRlPpoActorCriticCfg):

    class_name: str = "BOTActorCritic"

    actor_mapping: str = MISSING
    critic_mapping: str = MISSING
    shortest_path_matrix: str = MISSING
    embedding_dim: int = MISSING
    feedforward_dim: int = MISSING

    nheads: int = MISSING
    nlayers: int = MISSING

    #encode_hidden_dims: dict[str, list[int]] = MISSING
    #f"encode_{name}_hidden_dims": list[int] = MISSING
