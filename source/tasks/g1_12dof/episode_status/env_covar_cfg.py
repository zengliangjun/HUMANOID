from isaaclab.utils import configclass
from ..mdps  import obs
from . import env_cfg

from isaaclabex.envs.mdp.statistics import joints #, body
from isaaclabex.envs.managers import term_cfg
from isaaclab.managers import SceneEntityCfg, ObservationTermCfg, ObservationGroupCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclabex.envs.mdp.observations import statistics

@configclass
class StatisticsCfg:
    pos = term_cfg.StatisticsTermCfg(
        func= joints.CovarJPos,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot")},
        export_interval = 1000000
    )
    action = term_cfg.StatisticsTermCfg(
        func= joints.CovarAction,
        params={
            "action_name": "joint_pos",
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot")},
        export_interval = 1000000
    )
    """
    footclearance = term_cfg.StatisticsTermCfg(
        func= body.StatusFootClearance,
        params={
            "asset_cfg": SceneEntityCfg("robot",
                     body_names=[".*left_ankle_roll_link", ".*right_ankle_roll_link"])},
        export_interval = 1000000
    )
    """

@configclass
class ObservationsCfg(obs.ObservationsCfg):

    @configclass
    class ActionStatisticsCfg(ObservationGroupCfg):
        action_episode_mean = ObservationTermCfg(func=statistics.obs_episode_mean,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.1, n_max=0.1))
        action_episode_variance = ObservationTermCfg(func=statistics.obs_episode_variance,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.1, n_max=0.1))
        action_step_mean_mean = ObservationTermCfg(func=statistics.obs_step_mean_mean,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.1, n_max=0.1))
        action_step_mean_variance = ObservationTermCfg(func=statistics.obs_step_mean_variance,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.1, n_max=0.1))
        action_step_variance_mean = ObservationTermCfg(func=statistics.obs_step_variance_mean,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.1, n_max=0.1))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PosStatisticsCfg(ObservationGroupCfg):

        pos_episode_mean = ObservationTermCfg(func=statistics.obs_episode_mean,
                                       params={"pos_statistics_name": "pos"},
                                       noise=Unoise(n_min=-0.1, n_max=0.1))
        pos_episode_variance = ObservationTermCfg(func=statistics.obs_episode_variance,
                                       params={"pos_statistics_name": "pos"},
                                       noise=Unoise(n_min=-0.1, n_max=0.1))
        pos_step_mean_mean = ObservationTermCfg(func=statistics.obs_step_mean_mean,
                                       params={"pos_statistics_name": "pos"},
                                       noise=Unoise(n_min=-0.1, n_max=0.1))
        pos_step_mean_variance = ObservationTermCfg(func=statistics.obs_step_mean_variance,
                                       params={"pos_statistics_name": "pos"},
                                       noise=Unoise(n_min=-0.1, n_max=0.1))
        pos_step_variance_mean = ObservationTermCfg(func=statistics.obs_step_variance_mean,
                                       params={"pos_statistics_name": "pos"},
                                       noise=Unoise(n_min=-0.1, n_max=0.1))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ActionCovarCfg(ObservationGroupCfg):
        action_episode_mean = ObservationTermCfg(func=statistics.obs_covar_mean,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.1, n_max=0.1))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PosCovarCfg(ObservationGroupCfg):
        pos_episode_mean = ObservationTermCfg(func=statistics.obs_covar_mean,
                                       params={"pos_statistics_name": "pos"},
                                       noise=Unoise(n_min=-0.1, n_max=0.1))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True



    def __post_init__(self):
        self.policy.phase = None
        self.critic.phase = None

    # observation groups
    action_statistics: ActionStatisticsCfg = ActionStatisticsCfg()
    pos_statistics: PosStatisticsCfg = PosStatisticsCfg()
    action_covar: ActionCovarCfg = ActionCovarCfg()
    pos_covar: PosCovarCfg = PosCovarCfg()


@configclass
class G1ObsCovarCfg(env_cfg.G1FlatEnvV3Cfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.observations = ObservationsCfg()
        self.statistics = StatisticsCfg()

@configclass
class G1ObsCovarCfg_PLAY(env_cfg.G1FlatEnvV3Cfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.observations = ObservationsCfg()
        self.statistics = StatisticsCfg()

