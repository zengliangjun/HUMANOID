from dataclasses import MISSING
from isaaclab.utils import configclass


@configclass
class FLDCfg:

    class_name: str = "FLD"

    step_dt: float = MISSING

    observation_dim: int = MISSING

    observation_history_horizon: int = MISSING

    encoder_hidden_dims: list[int] = MISSING

    decoder_hidden_dims: list[int] = MISSING
