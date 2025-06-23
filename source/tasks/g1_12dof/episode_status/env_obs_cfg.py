from isaaclab.utils import configclass
from ..mdps  import obs
from . import env_cfg

from isaaclabex.envs.mdp.statistics import joints
from isaaclabex.envs.managers import term_cfg
from isaaclab.managers import SceneEntityCfg, ObservationTermCfg, ObservationGroupCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclabex.envs.mdp.observations import statistics

@configclass
class StatisticsCfg:
    pos = term_cfg.StatisticsTermCfg(
        func= joints.StatusJPos,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot")},

        # episode_truncation = 80,
        export_interval = 1000000
    )
    action = term_cfg.StatisticsTermCfg(
        func= joints.StatusAction,
        params={
            "action_name": "joint_pos",
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot")},

        # episode_truncation = 80,
        export_interval = 1000000
    )

@configclass
class ObservationsCfg(obs.ObservationsCfg):

    @configclass
    class ActionStatisticsCfg(ObservationGroupCfg):
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

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PosStatisticsCfg(ObservationGroupCfg):

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
            self.enable_corruption = True
            self.concatenate_terms = True


    def __post_init__(self):
        self.policy.phase = None
        self.critic.phase = None

    # observation groups
    action_statistics: ActionStatisticsCfg = ActionStatisticsCfg()
    pos_statistics: PosStatisticsCfg = PosStatisticsCfg()


@configclass
class G1ObsStatisticsCfg(env_cfg.G1FlatEnvV3Cfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.observations = ObservationsCfg()
        self.statistics = StatisticsCfg()

        # final valid cfg
        """
        self.curriculum.penalize_with_steps = None
        self.curriculum.events_with_steps = None

        self.rewards.rew_stability.weight = 3

        self.rewards.p_action_rate.weight = -0.05
        self.rewards.p_action_smoothness.weight=-0.008
        self.rewards.p_torques.weight=-1e-5
        self.rewards.p_pos_limits.weight=-2.0
        self.rewards.p_foot_slide.weight=-0.5
        """

@configclass
class G1ObsStatisticsCfg_PLAY(env_cfg.G1FlatEnvV3Cfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.observations = ObservationsCfg()
        self.statistics = StatisticsCfg()
        self.commands.base_velocity.ranges.lin_vel_x=(0, 2)
        self.events.interval_push = None
        self.events.interval_gravity = None
        self.events.interval_actuator = None
        self.events.interval_mass = None
        self.events.interval_coms = None

