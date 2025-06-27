from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg, \
                              ObservationGroupCfg, SceneEntityCfg

from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclabex.envs.mdp.observations import statistics, privileged

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""
        ## proprioceptive
        lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, scale = 2.0, noise=Unoise(n_min=-0.1, n_max=0.1))
        ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, scale = 0.25, noise=Unoise(n_min=-0.2, n_max=0.2))
        gravity = ObservationTermCfg(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        commands = ObservationTermCfg(func=mdp.generated_commands, scale = 0.25, params={"command_name": "base_velocity"})
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, scale = 0.05, noise=Unoise(n_min=-0.5, n_max=0.5))

        joint_acc = ObservationTermCfg(func=privileged.joint_acc, scale = 0.05, noise=Unoise(n_min=-0.2, n_max=0.2),
                                       params={"asset_cfg": SceneEntityCfg("robot", body_names=".*")})
        joint_torques = ObservationTermCfg(func=privileged.joint_torques, scale = 0.05, noise=Unoise(n_min=-0.1, n_max=0.1),
                                       params={"asset_cfg": SceneEntityCfg("robot", body_names=".*")})
        body_mass = ObservationTermCfg(func=privileged.body_mass, scale = 0.6, noise=Unoise(n_min=-0.1, n_max=0.1),
                                       params={"asset_cfg": SceneEntityCfg("robot", body_names=".*")})
        feet_contact_forces = ObservationTermCfg(func=privileged.feet_contact_forces, scale = 0.02, noise=Unoise(n_min=-0.1, n_max=0.1),
                                       params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")})
        feet_pos = ObservationTermCfg(func=privileged.feet_pos, noise=Unoise(n_min=-0.1, n_max=0.1),
                                       params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")})

        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


    @configclass
    class ActionStatisticsCfg(ObservationGroupCfg):
        action_episode_mean = ObservationTermCfg(func=statistics.obs_episode_mean,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.05, n_max=0.05))
        action_episode_variance = ObservationTermCfg(func=statistics.obs_episode_variance,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.05, n_max=0.05))
        action_step_mean_mean = ObservationTermCfg(func=statistics.obs_step_mean_mean,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.05, n_max=0.05))
        action_step_mean_variance = ObservationTermCfg(func=statistics.obs_step_mean_variance,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.05, n_max=0.05))
        action_step_variance_mean = ObservationTermCfg(func=statistics.obs_step_variance_mean,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.05, n_max=0.05))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PosStatisticsCfg(ObservationGroupCfg):

        pos_episode_mean = ObservationTermCfg(func=statistics.obs_episode_mean,
                                       params={"pos_statistics_name": "pos"},
                                       noise=Unoise(n_min=-0.05, n_max=0.05))
        pos_episode_variance = ObservationTermCfg(func=statistics.obs_episode_variance,
                                       params={"pos_statistics_name": "pos"},
                                       noise=Unoise(n_min=-0.05, n_max=0.05))
        pos_step_mean_mean = ObservationTermCfg(func=statistics.obs_step_mean_mean,
                                       params={"pos_statistics_name": "pos"},
                                       noise=Unoise(n_min=-0.05, n_max=0.05))
        pos_step_mean_variance = ObservationTermCfg(func=statistics.obs_step_mean_variance,
                                       params={"pos_statistics_name": "pos"},
                                       noise=Unoise(n_min=-0.05, n_max=0.05))
        pos_step_variance_mean = ObservationTermCfg(func=statistics.obs_step_variance_mean,
                                       params={"pos_statistics_name": "pos"},
                                       noise=Unoise(n_min=-0.05, n_max=0.05))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    action_statistics: ActionStatisticsCfg = ActionStatisticsCfg()
    pos_statistics: PosStatisticsCfg = PosStatisticsCfg()

