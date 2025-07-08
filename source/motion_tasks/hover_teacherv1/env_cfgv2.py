from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclabex.envs.managers import term_cfg
from isaaclabmotion.envs.mdp.statistics import robotbody
from isaaclabmotion.envs.mdp.rewards import statistics

from isaaclabmotion.assets.motions import hover_h1
from . import env_cfg, env_cfgv1



@configclass
class StatisticsCfg:
    rbpos_diff = term_cfg.StatisticsTermCfg(
        func= robotbody.RBPosDiff,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "motions_name": "hoverh1"
        },

        # episode_truncation = 80,
        export_interval = 1000000
    )


@configclass
class OMNIH2OH1CfgV2(env_cfg.OMNIH2OH1Cfg):

    rewards = env_cfgv1.RewardsCfg()
    statistics = StatisticsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.motions.hoverh1.motion_file = f"{hover_h1._ASSETS_ROOT}/motions/asap/omnih2o/amass_phc_filtered.pkl"


@configclass
class OMNIH2OH1CfgV2_PLAY(env_cfg.OMNIH2OH1Cfg_PLAY):

    rewards = env_cfgv1.RewardsCfg()
    statistics = StatisticsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.motions.hoverh1.motion_file = f"{hover_h1._ASSETS_ROOT}/motions/asap/omnih2o/amass_phc_filtered.pkl"
