from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclabex.envs.managers import term_cfg
from isaaclabmotion.envs.mdp.statistics import robotbody
from isaaclabmotion.envs.mdp.rewards import statistics

from isaaclabmotion.assets.motions import hover_h1
from .mdps  import rewards
from . import env_cfg



@configclass
class StatisticsCfg:
    rbpos_diff = term_cfg.StatisticsTermCfg(
        func= robotbody.RBPosHeadDiff,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "motions_name": "hoverh1"
        },

        # episode_truncation = 80,
        export_interval = 1000000
    )



@configclass
class RewardsCfg(rewards.RewardsCfg):

    rew_mean_pos_diff = RewardTermCfg(
        func=statistics.rew_mean_rbpos_headdiff,
        weight=8,

        params={"motions_name": "hoverh1",
                "statistics_name": "rbpos_diff",
                # "body_names": None,
                "std":  0.05},
    )
    rew_var_pos_diff = RewardTermCfg(
        func=statistics.rew_variance_rbpos_headdiff,
        weight=8,

        params={"motions_name": "hoverh1",
                "statistics_name": "rbpos_diff",
                # "body_names": None,
                "std":  0.05},
    )
    rewmean_upper_diff = RewardTermCfg(
        func=statistics.rew_mean_rbpos_headdiff,
        weight=16,

        params={"motions_name": "hoverh1",
                "statistics_name": "rbpos_diff",
                "body_names": rewards.bnames[:11] + rewards.extend_body_names,
                "std":  0.05},
    )
    rewvar_upper_diff = RewardTermCfg(
        func=statistics.rew_variance_rbpos_headdiff,
        weight=16,

        params={"motions_name": "hoverh1",
                "statistics_name": "rbpos_diff",
                "body_names": rewards.bnames[:11] + rewards.extend_body_names,
                "std":  0.05},
    )
    rewmean_extend_diff = RewardTermCfg(
        func=statistics.rew_mean_rbpos_headdiff,
        weight=24,

        params={"motions_name": "hoverh1",
                "statistics_name": "rbpos_diff",
                "body_names": rewards.extend_body_names,
                "std":  0.05},
    )
    rewvar_extend_diff = RewardTermCfg(
        func=statistics.rew_variance_rbpos_headdiff,
        weight=24,

        params={"motions_name": "hoverh1",
                "statistics_name": "rbpos_diff",
                "body_names": rewards.extend_body_names,
                "std":  0.05},
    )

    def __post_init__(self):
        self.rew_mean_pos_headdiff = None
        self.rew_var_pos_headdiff = None
        self.rewmean_upper_headdiff = None
        self.rewvar_upper_headdiff = None
        self.rewmean_extend_headdiff = None
        self.rewvar_extend_headdiff = None
        self.rew_mean_rootdiff = None
        self.rew_var_rootdiff = None


@configclass
class OMNIH2OH1CfgV1(env_cfg.OMNIH2OH1Cfg):

    rewards = RewardsCfg()
    statistics = StatisticsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.motions.hoverh1.motion_file = f"{hover_h1._ASSETS_ROOT}/motions/asap/omnih2o/amass_phc_filtered.pkl"


@configclass
class OMNIH2OH1CfgV1_PLAY(env_cfg.OMNIH2OH1Cfg_PLAY):

    rewards = RewardsCfg()
    statistics = StatisticsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.motions.hoverh1.motion_file = f"{hover_h1._ASSETS_ROOT}/motions/asap/omnih2o/amass_phc_filtered.pkl"
