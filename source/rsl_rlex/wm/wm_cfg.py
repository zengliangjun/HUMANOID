
from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

@configclass
class WMActorCriticCfg(RslRlPpoActorCriticCfg):

    class_name: str = "WMActorCritic"

    # Continuous Decoder
    continuous_decoder_dims: list[int] = MISSING
    # Discrete Decoder
    discrete_decoder_dims: list[int] = MISSING

    proprioceptive_names: list[str] = MISSING
    continuous_names: list[str] = MISSING
    discreate_names: list[str] = MISSING

    rnn_type: str ="lstm"
    # History Encoder
    rnn_encoder_hidden_size: int = 256
    rnn_encoder_num_layers: int = 1

    rnn_actor_hidden_size: int = 256
    rnn_actor_num_layers: int = 1

    rnn_critic_hidden_size: int = 256
    rnn_critic_num_layers: int = 1

    num_actions: int = None

    input_dim: int = None
    continuous_dim: int = None
    discreate_dim: int = None
