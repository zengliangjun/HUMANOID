from isaaclab.utils import configclass
from ..mdps  import obs
from . import env_cfg

from isaaclabex.envs.mdp.statistics import joints
from isaaclabex.envs.managers import term_cfg
from isaaclab.managers import SceneEntityCfg, ObservationTermCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclabex.envs.mdp.observations import statistics

@configclass
class StatisticsCfg:
    pos = term_cfg.StatisticsTermCfg(
        func= joints.StatusJPos,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot")},
        export_interval = 1000000
    )
    action = term_cfg.StatisticsTermCfg(
        func= joints.StatusAction,
        params={
            "action_name": "joint_pos",
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot")},
        export_interval = 1000000
    )


class ObservationsCfg(obs.ObservationsCfg):
    @configclass
    class PolicyCfg(obs.ObservationsCfg.PolicyCfg):
        action_episode_mean = ObservationTermCfg(func=statistics.obs_episode_mean,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))
        action_episode_variance = ObservationTermCfg(func=statistics.obs_episode_variance,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))
        action_step_mean_mean = ObservationTermCfg(func=statistics.obs_step_mean_mean,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))
        action_step_mean_variance = ObservationTermCfg(func=statistics.obs_step_mean_variance,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))
        action_step_variance_mean = ObservationTermCfg(func=statistics.obs_step_variance_mean,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))


    @configclass
    class CriticCfg(obs.ObservationsCfg.CriticCfg):
        action_episode_mean = ObservationTermCfg(func=statistics.obs_episode_mean,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))
        action_episode_variance = ObservationTermCfg(func=statistics.obs_episode_variance,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))
        action_step_mean_mean = ObservationTermCfg(func=statistics.obs_step_mean_mean,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))
        action_step_mean_variance = ObservationTermCfg(func=statistics.obs_step_mean_variance,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))
        action_step_variance_mean = ObservationTermCfg(func=statistics.obs_step_variance_mean,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))


        pos_episode_mean = ObservationTermCfg(func=statistics.obs_episode_mean,
                                       params={"pos_statistics_name": "pos"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))
        pos_episode_variance = ObservationTermCfg(func=statistics.obs_episode_variance,
                                       params={"pos_statistics_name": "pos"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))
        pos_step_mean_mean = ObservationTermCfg(func=statistics.obs_step_mean_mean,
                                       params={"pos_statistics_name": "pos"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))
        pos_step_mean_variance = ObservationTermCfg(func=statistics.obs_step_mean_variance,
                                       params={"pos_statistics_name": "pos"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))
        pos_step_variance_mean = ObservationTermCfg(func=statistics.obs_step_variance_mean,
                                       params={"pos_statistics_name": "pos"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))


    def __post_init__(self):
        self.policy.phase = None
        self.critic.phase = None

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1ObsStatisticsCfg(env_cfg.G1FlatEnvV3Cfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.observations = ObservationsCfg()
        self.statistics = StatisticsCfg()

@configclass
class G1ObsStatisticsCfg_PLAY(env_cfg.G1FlatEnvV3Cfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.observations = ObservationsCfg()
        self.statistics = StatisticsCfg()

