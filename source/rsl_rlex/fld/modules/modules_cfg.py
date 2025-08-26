from dataclasses import MISSING
from isaaclab.utils import configclass


@configclass
class FLDCfg:

    class_name: str = "FLD"

    fldmodel_prefix: str = "tspmodel"

    step_dt: float = MISSING

    observation_dim: int = MISSING

    observation_history_horizon: int = MISSING

    encoder_hidden_dims: list[int] = MISSING

    decoder_hidden_dims: list[int] = MISSING



@configclass
class FLDExtendCfg(FLDCfg):

    forecast_horizon: int = MISSING

    num_mini_batches: int = MISSING   # 80

    num_epochs: int = MISSING         # 4

    mini_batch_size: int = MISSING    # 4000
