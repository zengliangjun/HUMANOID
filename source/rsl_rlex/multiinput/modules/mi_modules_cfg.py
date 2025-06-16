
from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

@configclass
class MIEncodeActorCriticCfg(RslRlPpoActorCriticCfg):

    class_name: str = "MIEncodeActorCritic"

    policy_groups: list[str] = MISSING
    critic_groups: list[str] = MISSING
    encode_groups: list[str] = MISSING

    #encode_hidden_dims: dict[str, list[int]] = MISSING
    #f"encode_{name}_hidden_dims": list[int] = MISSING
