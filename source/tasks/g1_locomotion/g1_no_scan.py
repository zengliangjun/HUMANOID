from . import g1_orgenv
from isaaclab.utils import configclass
from isaaclabex.terrains.config import rough_low_level_cfg

@configclass
class G1RoughEnvCfg(g1_orgenv.G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        self.scene.terrain.terrain_generator=rough_low_level_cfg.ROUGH_TERRAINS_CFG
        super().__post_init__()
        self.observations.policy.height_scan = None

@configclass
class G1RoughEnvCfg_PLAY(g1_orgenv.G1RoughEnvCfg_PLAY):
    def __post_init__(self):
        # post init of parent
        self.scene.terrain.terrain_generator=rough_low_level_cfg.ROUGH_TERRAINS_CFG
        super().__post_init__()
        self.observations.policy.height_scan = None
